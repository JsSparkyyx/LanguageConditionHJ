"""The sliding-tray task (T8 ``tray_slide``) for the residual policy (residual/PLAN.md section 8).

One arm pinches the tray's handle tab and drags it 32-38 cm sideways into the pad. On the tray stand a wine glass
(item ``glass``, tall: tips at a gentle stop) and a can (item ``can``, short: tips only at a hard stop). The nominal
is the careless ``fast`` drag (0.70 m/s, 3 cm brake: both fall); the residual must slow it down / ease the stop so
that the NAMED item stays standing (``upright|<item>|-``, 45 deg). The other item may fall: it is logged, never
part of the reward or the termination, so the policy has a reason to slow down less for the can than for the glass.

State (``build_obs``, 186 numbers; every position in the world frame):

- robot: arm joints q [12] and velocities [12], gripper openings [2], both hands' pose (position, rotation 6D)
  [2 x 9], which arm drags (one-hot left / right) [2];
- planner: a_nom [12], the plan rows 5, 10, 15, 20, 25 ticks ahead as offsets from the live joints [5 x 12], per arm
  whether a plan plays [2] and the cursor's share of its plan [2], whether the risky drag is the current motion [1];
- tray: position [3], rotation 6D [6], linear and angular velocity [6], linear acceleration over the last tick [3],
  pad centre minus tray centre (x, y) [2];
- per item, the NAMED one first, then the other [2 x 20]: position relative to the tray [3], rotation 6D [6], linear
  and angular velocity [6], tilt (rad) [1], upright margin (45 deg minus tilt, rad) [1], height [1], base radius
  [1], tipping acceleration g r / c (m/s^2, the task's own estimate) [1];
- which item is named (one-hot glass / can) [1 x 2];
- w [1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ensure_paths
from .config import N_ARM

ensure_paths()

TASK = "tray_slide"
ITEM_IDS = ("glass", "can")
PHYSICS_DT = 1.0 / 250.0
# which items the named behaviour's own drag tips in most runs (envs/docs/item_envs/task_cards/T8.md, final check):
# the tray's "in the planner's way"
AT_RISK = {"slow": (), "medium": ("glass",), "fast": ("glass", "can"), "recover_glass": (), "recover_can": ("glass",)}
N_ITEM_FEAT = 20


@dataclass
class TrayNamed:
    item_id: str
    kind: str            # "upright"
    in_way: bool         # the behaviour's drag tips it in most runs (AT_RISK)
    reason: str          # the behaviour's name
    rule_index: int
    item_index: int
    category: str = ""
    behaviour: str = "fast"


def _rule_index(runner, item_id: str) -> int | None:
    rid = f"upright|{item_id}|-"
    for j, rule in enumerate(runner.rules):
        if rule.rule_id == rid:
            return j
    return None


def behaviour_of(runner) -> str:
    return str((runner.risk_params.task or {}).get("behaviour") or "fast")


def make_named(runner, item_id: str) -> TrayNamed:
    if item_id not in ITEM_IDS:
        raise ValueError(f"tray_slide names the glass or the can, got {item_id!r}")
    j = _rule_index(runner, item_id)
    if j is None:
        raise ValueError(f"no upright rule for {item_id!r}")
    k = next(i for i, s in enumerate(runner.items) if s.item_id == item_id)
    b = behaviour_of(runner)
    return TrayNamed(item_id=item_id, kind="upright", in_way=item_id in AT_RISK.get(b, ITEM_IDS), reason=b,
                     rule_index=j, item_index=k, category=str(runner.items[k].category), behaviour=b)


class ItemBalancer:
    """Names the glass and the can equally often (the one behind; a tie is drawn)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = {i: 0 for i in ITEM_IDS}

    def __call__(self, runner, rng, job):
        if job.item_id is not None:
            return make_named(runner, job.item_id)
        lo = min(self.n.values())
        behind = [i for i in ITEM_IDS if self.n[i] == lo]
        iid = behind[int(rng.integers(len(behind)))] if len(behind) > 1 else behind[0]
        named = make_named(runner, iid)
        self.n[iid] += 1
        return named


# --------------------------------------------------------------------------------------------- measuring ----
def _rot6d(R) -> np.ndarray:
    return np.asarray(R, dtype=np.float64)[:3, :2].T.reshape(-1)


class TrayProbe:
    """Per-episode measuring tool: the tray, both items, the named item's upright margin (the runner's one_margin)."""

    def __init__(self, runner, named, cfg, clock=None):
        self.runner, self.named, self.cfg = runner, named, cfg
        self.env = runner.env
        self.task = self.env.task
        self.specs = {s.item_id: s for s in runner.items}
        self.tray = self.specs["tray"]
        self.k = int(named.item_index)
        self.spec = runner.items[self.k]
        self.rule = runner.rules[named.rule_index]
        self.other_id = next(i for i in ITEM_IDS if i != named.item_id)
        j_other = _rule_index(runner, self.other_id)
        self.other_rule = runner.rules[j_other] if j_other is not None else None
        self.other_k = next(i for i, s in enumerate(runner.items) if s.item_id == self.other_id)
        # the task's own item design: height, base radius, tipping acceleration g r / c
        self.design = {str(it["item_id"]): it for it in getattr(self.task, "standing", [])}
        self.clock = clock               # physics steps so far (the episode's counter), for the acceleration
        self._prev = None                # (steps, tray linear velocity)
        self.acc = np.zeros(3)
        self.start_pos = self._T(self.spec)[:3, 3].copy()
        # episode extremes (episode_extra)
        self.named_max_tilt = 0.0
        self.other_max_tilt = 0.0
        self.other_min_margin = float("inf")
        self.tray_v_max = 0.0
        self.tray_acc_max = 0.0

    @staticmethod
    def _T(spec) -> np.ndarray:
        from main.item_envs.runner import _world_T

        return _world_T(spec)

    def _upright_margin(self, rule, k, spec) -> float:
        from main.item_envs.runner import one_margin

        return float(one_margin(rule, [(k, spec)], [], None, self.runner._stay_ref))

    def _item(self, spec, rule, k) -> dict:
        from main.item_envs import geometry
        from main.item_envs.runner import _velocity6

        T = self._T(spec)
        return {"T": T, "vel": _velocity6(spec), "tilt": float(geometry.tilt_angle(T, spec.up_axis_local)),
                "margin": self._upright_margin(rule, k, spec) if rule is not None else float("inf")}

    def drag_active(self) -> bool:
        t = self.task
        risky = getattr(t, "_time_scale_risky", None)
        safe = getattr(t, "_time_scale_safe", None)
        act = getattr(t, "active_time_scale", None)
        if risky is None or act is None or safe is None or abs(float(risky) - float(safe)) < 1e-12:
            return False
        return abs(float(act) - float(risky)) < 1e-9

    def measure(self) -> dict:
        from main.item_envs.runner import _velocity6

        T_tray = self._T(self.tray)
        v_tray = _velocity6(self.tray)
        steps = int(self.clock()) if self.clock is not None else None
        if steps is not None and self._prev is not None and steps > self._prev[0]:
            self.acc = (v_tray[:3] - self._prev[1]) / ((steps - self._prev[0]) * PHYSICS_DT)
        if steps is not None:
            self._prev = (steps, v_tray[:3].copy())
        named = self._item(self.spec, self.rule, self.k)
        other = self._item(self.specs[self.other_id], self.other_rule, self.other_k)
        self.named_max_tilt = max(self.named_max_tilt, named["tilt"])
        self.other_max_tilt = max(self.other_max_tilt, other["tilt"])
        self.other_min_margin = min(self.other_min_margin, other["margin"])
        self.tray_v_max = max(self.tray_v_max, float(np.linalg.norm(v_tray[:2])))
        self.tray_acc_max = max(self.tray_acc_max, float(np.linalg.norm(self.acc[:2])))
        T = named["T"]
        origin = T[:3, :3] @ np.asarray(self.spec.obb_center_local, dtype=np.float64) + T[:3, 3]
        return {
            "margin": named["margin"], "tilt": named["tilt"], "T": T, "vel": named["vel"],
            "obb": (origin, T[:3, :3].copy(), np.asarray(self.spec.half_extents, dtype=np.float64)),
            # the T1 contact bookkeeping in Episode._check: nothing touches here
            "dist": float("inf"), "d_arm": float("inf"), "d_held": float("inf"), "named_held": False,
            "named": named, "other": other, "tray_T": T_tray, "tray_vel": v_tray, "tray_acc": self.acc.copy(),
            "drag": self.drag_active(),
        }

    def item_features(self, it: dict, item_id: str, tray_pos: np.ndarray) -> np.ndarray:
        d = self.design.get(item_id, {})
        T = it["T"]
        return np.concatenate([T[:3, 3] - tray_pos, _rot6d(T[:3, :3]), it["vel"], [it["tilt"], it["margin"]],
                               [float(d.get("h", 0.0)), float(d.get("base_r", 0.0)), float(d.get("a_tip", 0.0))]])


def build_obs(cfg, probe: TrayProbe, m: dict, rob: dict, view: dict, w: float) -> np.ndarray:
    from .state import _quat_to_R

    task = probe.task
    carry = getattr(task, "carry_arm", None)
    T_tray = m["tray_T"]
    tray_pos = T_tray[:3, 3]
    target = np.asarray(getattr(task, "target_xy", tray_pos[:2]), dtype=np.float64).reshape(2)
    parts = [
        rob["q_left"], rob["q_right"], rob["qd_left"], rob["qd_right"], rob["grip"],
        rob["ee_left"][:3], _rot6d(_quat_to_R(rob["ee_left"][3:7])),
        rob["ee_right"][:3], _rot6d(_quat_to_R(rob["ee_right"][3:7])),
        [1.0 if carry == "left" else 0.0, 1.0 if carry == "right" else 0.0],
        view["a_nom"], view["future"].reshape(-1), view["active"], view.get("progress", np.zeros(2)),
        [1.0 if m["drag"] else 0.0],
        tray_pos, _rot6d(T_tray[:3, :3]), m["tray_vel"], m["tray_acc"], target - tray_pos[:2],
        probe.item_features(m["named"], probe.named.item_id, tray_pos),
        probe.item_features(m["other"], probe.other_id, tray_pos),
        [1.0 if probe.named.item_id == i else 0.0 for i in ITEM_IDS],
        [float(w)],
    ]
    obs = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts])
    if not np.all(np.isfinite(obs)):
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return obs.astype(np.float32)


def obs_dim(cfg) -> int:
    n_fut = len(cfg.future_ticks)
    robot = 4 * N_ARM + 2 + 2 * 9 + 2
    planner = 2 * N_ARM * (1 + n_fut) + 2 + 2 + 1
    tray = 3 + 6 + 6 + 3 + 2
    return robot + planner + tray + 2 * N_ITEM_FEAT + len(ITEM_IDS) + 1


# ------------------------------------------------------------------------------------- scenes and evaluation ----
def scan_record(runner) -> dict:
    task = runner.env.task
    cands = [s for s in runner.items if s.item_id in ITEM_IDS]
    return {
        "host": TASK,
        "n_candidates": len(cands),
        "items": [s.item_id for s in cands],
        "categories": {s.item_id: str(s.category) for s in cands},
        "referent": getattr(task, "referent_id", None),
        "carry_arm": getattr(task, "carry_arm", None),
        "rules_ok": all(_rule_index(runner, i) is not None for i in ITEM_IDS),
        "time_scale": float(runner.risk_params.time_scale),
    }


def select_eval_scenes(scans: list[dict], n_pairs: int, behaviours) -> list[dict]:
    """The first ``n_pairs`` good seeds, each run twice: the glass named, then the can (paired: same scene, the other
    rule). The nominal behaviour cycles through ``behaviours`` by seed."""
    ok = sorted([s for s in scans if not s.get("error") and s.get("n_candidates", 0) == len(ITEM_IDS)
                 and s.get("rules_ok", True)], key=lambda s: s["seed"])
    out = []
    behaviours = list(behaviours)
    for p, s in enumerate(ok[:int(n_pairs)]):
        b = behaviours[p % len(behaviours)]
        for iid in ITEM_IDS:
            out.append({"seed": s["seed"], "host": TASK, "item_id": iid, "reason": b, "behaviour": b,
                        "in_way": iid in AT_RISK.get(b, ITEM_IDS), "rule_kind": "upright", "group": iid,
                        "category": s.get("categories", {}).get(iid, iid), "pair": p})
    return out
