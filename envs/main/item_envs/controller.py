"""RiskTickController (PLAN v3 §2.5): TickChunkedController + plan retiming.

Every successful plan is retimed with the task's ``active_time_scale`` via
``retime.retime``. Velocities are NEVER clipped: retime reports
``info["v_ratio"]`` (max |v'| / URDF joint velocity limit) which lands in
``retime_log`` / ``v_ratio_events`` and on stdout.

Time-scale ownership (T1 mechanism, do not combine paths):

- T2/T3 (and smoke_items): the task's ``risky(task)`` context manager
  flips ``task.active_time_scale`` around a motion primitive; plan() only
  READS the attribute for the retime factor.
- T1: host tasks run unmodified, so risky segments are chosen by skill
  type. When ``risky_skill_labels`` is non-empty, plan() first calls
  ``_sync_time_scale_from_skill()``: the current skill label (the CEHJ
  skill tracker stamps ``task._cehj_skill`` with the label of the action
  being planned) is substring-matched against ``risky_skill_labels`` and
  ``task.active_time_scale`` is set to the risky or safe scale
  accordingly. With an empty ``risky_skill_labels`` this path is inert.

Realized-vs-commanded tracking: ``attach()`` chains this controller's
``tick_hook`` around any pre-existing hook (saved as ``_user_tick_hook``
and always honored). At each tick boundary the upcoming chunk's commanded
joint displacement (``pos[min(i+spt, n-1)] - qpos_live``, per arm, in plan
joint order) is recorded; the next boundary measures the achieved
displacement. Pairs accumulate only after the first v_ratio > 1 event
(the only regime where tracking matters); ``mean_realized_ratio`` is the
rollout.py definition — summed achieved norms / summed commanded norms
over chunks with nonzero command. All tracking is wrapped in try/except:
diagnostics never break control.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.controller lives in the sibling CEHJ copy

from main.envs.controller import TickChunkedController, current_skill  # noqa: E402

from .retime import retime  # noqa: E402

DEFAULT_JOINT_VEL = 2.0  # rad/s (or m/s prismatic) fallback, body_features.py policy


class RiskTickController(TickChunkedController):
    """Vanilla tick controller whose plans are uniformly time-scaled."""

    controller_name = "risk_tick"

    def __init__(self, env, risky_skill_labels: tuple[str, ...] = (), **kw):
        super().__init__(env, **kw)
        self.risky_skill_labels = tuple(str(s) for s in risky_skill_labels)
        self._v_limit_cache: dict[str, np.ndarray] = {}
        self.retime_log: list[dict] = []  # one entry per retimed plan
        self.v_ratio_events: list[dict] = []  # subset with v_ratio > 1 (never clipped)
        self.n_plan_failures: int = 0
        # realized-vs-commanded tracking (see module docstring)
        self._user_tick_hook = None
        self._tracking_enabled = False
        self._pending_chunk = None  # (plan ids, qpos per arm, commanded norms)
        self._realized_pairs: list[tuple[float, float]] = []

    # ----------------------------------------------------------------- plan ----
    def plan(self, arm: str, target_pose, **kw):
        self._sync_time_scale_from_skill()
        result = super().plan(arm, target_pose, **kw)
        if not result or result.get("status") != "Success":
            self.n_plan_failures += 1
            return result
        pos = np.asarray(result.get("position"), dtype=np.float64)
        vel_raw = result.get("velocity")
        vel = np.zeros_like(pos) if vel_raw is None else np.asarray(vel_raw, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2 or vel.shape != pos.shape:
            # degenerate plan (0/1 rows or missing velocity): leave untouched —
            # retime needs >= 2 rows; never crash a sweep on a short plan.
            return result
        s = float(getattr(getattr(self.env, "task", None), "active_time_scale", 1.0))
        pos_out, vel_out, info = retime(pos, vel, s, v_limit=self._v_limit(arm))
        result["position"], result["velocity"] = pos_out, vel_out
        result["_item_retime"] = info
        self.retime_log.append({"arm": arm, **info})
        v_ratio = info.get("v_ratio")
        if v_ratio is not None and v_ratio > 1.0:
            event = {
                "arm": arm,
                "s": info["s"],
                "v_ratio": float(v_ratio),
                "T_in": info["T_in"],
                "T_out": info["T_out"],
            }
            self.v_ratio_events.append(event)
            self._tracking_enabled = True  # start realized-vs-commanded tracking
            print(
                f"[risk_tick] v_ratio={v_ratio:.3f} > 1 (arm={arm} s={info['s']} "
                f"T {info['T_in']}->{info['T_out']}) — NOT clipped; tracking "
                f"realized-vs-commanded displacement"
            )
        return result

    def _sync_time_scale_from_skill(self) -> None:
        """T1 path: set task.active_time_scale from the current skill label.

        Inert when risky_skill_labels is empty (T2/T3 use the risky() CM).
        The skill tracker stamps task._cehj_skill with the label of the
        action being planned BEFORE the planner runs, so the label read here
        is the one this plan belongs to.
        """
        if not self.risky_skill_labels:
            return
        task = getattr(self.env, "task", None)
        if task is None:
            return
        label = current_skill(task)
        risky = any(tag in label for tag in self.risky_skill_labels)
        scale = getattr(task, "_time_scale_risky", 1.0) if risky else getattr(
            task, "_time_scale_safe", 1.0
        )
        task.active_time_scale = float(scale)

    # ------------------------------------------------------------- tracking ----
    def attach(self) -> None:
        super().attach()  # plan() routing + skill tracker + tick chunking
        self._user_tick_hook = self.tick_hook  # RolloutController's, if any
        self.tick_hook = self._tracking_tick_hook

    @property
    def mean_realized_ratio(self) -> float | None:
        """Summed achieved / summed commanded joint displacement (cmd > 0 chunks)."""
        num = sum(a for a, c in self._realized_pairs if c > 0)
        den = sum(c for a, c in self._realized_pairs if c > 0)
        if den <= 0:
            return None
        return float(num / den)

    def _tracking_tick_hook(self, ctx):
        try:
            self._observe_tick(ctx)
        except Exception:
            pass  # diagnostics never break control
        if self._user_tick_hook is not None:
            return self._user_tick_hook(ctx)
        return False

    def _observe_tick(self, ctx) -> None:
        qpos = self._arm_qpos()  # {"left": [n_l], "right": [n_r]} plan-order arm joints
        if self._pending_chunk is not None and self._tracking_enabled:
            plan_ids, start_qpos, cmd_norms = self._pending_chunk
            # a replan between boundaries discards the pending chunk: its
            # commanded rows never played (plan object identity changed)
            if plan_ids == (id(ctx.get("left_arm")), id(ctx.get("right_arm"))):
                achieved = 0.0
                for arm in ("left", "right"):
                    if arm in start_qpos:
                        n = start_qpos[arm].shape[0]
                        achieved += float(np.linalg.norm(qpos[arm][:n] - start_qpos[arm]))
                self._realized_pairs.append((achieved, sum(cmd_norms.values())))
        spt = self.steps_per_tick
        i = int(ctx.get("i", 0))
        start_qpos, cmd_norms = {}, {}
        for arm, key in (("left", "left_arm"), ("right", "right_arm")):
            plan = ctx.get(key)
            pos = None if plan is None else plan.get("position")
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            n_rows, n_cols = pos.shape[0], pos.shape[1]
            if n_rows < 2 or i >= n_rows or qpos[arm].shape[0] < n_cols:
                continue
            q_now = qpos[arm][:n_cols]
            target_row = pos[min(i + spt, n_rows - 1)]
            start_qpos[arm] = q_now.copy()
            cmd_norms[arm] = float(np.linalg.norm(target_row - q_now))
        self._pending_chunk = (
            (id(ctx.get("left_arm")), id(ctx.get("right_arm"))),
            start_qpos,
            cmd_norms,
        )

    def _arm_qpos(self) -> dict[str, np.ndarray]:
        """Live arm-joint qpos per arm, plan joint order (gripper excluded)."""
        robot = self.env.robot
        return {
            "left": np.asarray(robot.get_left_arm_real_jointState(), dtype=np.float64)[:-1],
            "right": np.asarray(robot.get_right_arm_real_jointState(), dtype=np.float64)[:-1],
        }

    # ------------------------------------------------------------- v_limit ----
    def _v_limit(self, arm: str) -> np.ndarray:
        """URDF <limit velocity> per plan column, in plan joint order.

        Parses the embodiment URDF directly — the same <limit velocity>
        pattern as main.network.body_features._parse_urdf_joints — WITHOUT
        instantiating BodyTokenExtractor (that needs torch, pytorch_kinematics
        and a live sim). Missing/implausible limits fall back to
        DEFAULT_JOINT_VEL and are logged, mirroring body_features policy.
        """
        if arm in self._v_limit_cache:
            return self._v_limit_cache[arm]
        if arm == "left":
            urdf_path = self.robot.left_urdf_path
            joint_names = list(self.robot.left_arm_joints_name)
        else:
            urdf_path = self.robot.right_urdf_path
            joint_names = list(self.robot.right_arm_joints_name)
        limits = _urdf_velocity_limits(urdf_path)
        vels, fallbacks = [], []
        for name in joint_names:
            v = limits.get(name)
            if v is None or not (0.1 < v < 10.0):
                v = DEFAULT_JOINT_VEL
                fallbacks.append(name)
            vels.append(float(v))
        if fallbacks:
            print(
                f"[risk_tick] URDF {Path(str(urdf_path)).name}: missing/implausible "
                f"<limit velocity> on {fallbacks}; fallback {DEFAULT_JOINT_VEL}"
            )
        out = np.asarray(vels, dtype=np.float64)
        self._v_limit_cache[arm] = out
        return out


def _urdf_velocity_limits(urdf_path) -> dict[str, float]:
    """joint name -> <limit velocity> (revolute rad/s, prismatic m/s)."""
    root = ET.parse(str(urdf_path)).getroot()
    out: dict[str, float] = {}
    for joint in root.iter("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("velocity") is None:
            continue
        try:
            out[joint.get("name")] = float(limit.get("velocity"))
        except ValueError:
            continue
    return out
