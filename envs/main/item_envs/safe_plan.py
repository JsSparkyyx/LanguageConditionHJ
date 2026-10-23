"""Obstacle-aware planning for the "safe" nominal mode (Yuxuan 2026-10-22: the expert for safe demonstrations).

The careless nominal plans with a cuRobo world that holds the table only, so the arm goes straight through the
items. The safe mode (T1 so far) plans around the PROTECTED items instead:

- the items' boxes (ItemSpec OBB: entity pose, ``obb_center_local``, ``half_extents``), grown by a margin, are the
  cuRobo world of both arms (``push_protected_world``; RoboTwin's table cuboid is left out, as in vla_finetuning:
  its base-frame placement misses the real table, and with the hand modelled every grasp then starts "in
  collision");
- RoboTwin's ur5-wsg cuRobo model ends at ee_link, so the WSG hand and the held object are invisible to the
  planner. ``install_safe_yml`` makes every ur5-wsg CuroboPlanner built afterwards load a copy of the model with an
  ``attached_object`` link of 256 sphere slots (configs/curobo_safe/ur5-wsg.yml), and ``install_plan_hooks`` fills
  them before every plan with spheres covering the hand's collision boxes and the held object (``sync_arm``);
- the planner must be cuRobo on both arms (``require_curobo``): the mplib fallback ignores the world.

The hand model is vla_finetuning/hand_model.py's (the CrossSafe safe demonstrations, 500 kept); the item world is
new. The keep rule (no contact at any physics step) belongs to the demo collector, ``contact_with`` helps it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SAFE_YML = Path(__file__).resolve().parent / "configs" / "curobo_safe" / "ur5-wsg.yml"
N_SLOTS = 256
LINK = "attached_object"
PROTECTED_PREFIX = "protected_"

# link -> (lo, hi) of its collision mesh in the link frame, metres (vla_finetuning/hand_model.py)
HAND_BOXES = {
    "wsg_50_base_link": ((-0.073, -0.025, 0.0), (0.073, 0.025, 0.0725)),     # WSG50_110.stl
    "gripper_left": ((-0.018, -0.015, 0.0594), (0.0, 0.015, 0.0965)),        # GUIDE_WSG50_110.stl
    "gripper_right": ((-0.018, -0.015, 0.0594), (0.0, 0.015, 0.0965)),
    "finger_left": ((-0.02, -0.015, 0.0575), (0.003, 0.015, 0.1305)),        # WSG-FMF.stl
    "finger_right": ((-0.02, -0.015, 0.0575), (0.003, 0.015, 0.1305)),
}
HAND_R = 0.02              # largest hand sphere radius (m): 74 spheres for the 5 hand boxes
ROUND = ("bowl", "cup", "mug", "plate")
FALLBACK = ("full", "hand", "none")


# ------------------------------------------------------------------ robot model ----
def install_safe_yml() -> None:
    """Make ur5-wsg CuroboPlanners built from now on in this process load configs/curobo_safe/ur5-wsg.yml.
    Idempotent; call before the Env is built."""
    from main.envs.controller import CuroboIKController

    CuroboIKController.install()                  # prepare(): RoboTwin's root on sys.path, then the cache
    from envs.robot.planner import CuroboPlanner

    if getattr(CuroboPlanner.__init__, "_item_safe_yml", False) or getattr(CuroboPlanner.__init__, "_vf_safe_yml", False):
        return
    prev = CuroboPlanner.__init__
    stock = os.path.realpath(str(Path(os.environ.get("ROBOTWIN_ROOT", "/work/nvme/bifq/project/RoboTwin"))
                                 / "assets/embodiments/ur5-wsg/curobo.yml"))

    def init(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
        if yml_path is not None and os.path.realpath(str(yml_path)) == stock:
            yml_path = str(SAFE_YML)
        prev(self, robot_origion_pose, active_joints_name, all_joints, yml_path=yml_path)

    init._item_safe_yml = True
    CuroboPlanner.__init__ = init


def require_curobo(robot) -> None:
    """Both arms must plan with cuRobo in this process: the mplib fallback ignores the protected items."""
    from envs.robot.planner import CuroboPlanner

    if getattr(robot, "communication_flag", False):
        raise RuntimeError("cuRobo planners run in subprocesses: the world cannot be updated")
    for side in ("left", "right"):
        p = getattr(robot, f"{side}_planner", None)
        if not (isinstance(p, CuroboPlanner) and hasattr(p, "motion_gen") and hasattr(p, "motion_gen_batch")):
            raise RuntimeError(f"{side} planner is {type(p).__name__}, not cuRobo: the protected items would be ignored")


# ----------------------------------------------------------------- the world ----
def item_box(spec, margin: float = 0.0):
    """(world centre xyz, quat wxyz, half extents + margin) of an item's OBB, from its live pose."""
    from .items import entity_of

    pose = entity_of(spec).get_pose()
    T = np.asarray(pose.to_transformation_matrix(), np.float64)
    centre = T[:3, :3] @ np.asarray(spec.obb_center_local, np.float64) + T[:3, 3]
    return centre, np.asarray(pose.q, np.float64), np.asarray(spec.half_extents, np.float64) + float(margin)


def _check_enabled(mg, name: str) -> None:
    wcc = mg.world_coll_checker
    names = wcc._env_obbs_names[0]
    if name not in names:
        raise RuntimeError(f"cuRobo world has no cuboid {name!r} after update_world")
    idx = names.index(name)
    n = int(wcc._env_n_obbs[0])
    enabled = int(wcc._cube_tensor_list[2][0, idx])
    if not (idx < n and enabled == 1):
        raise RuntimeError(f"cuRobo cuboid {name!r} slot {idx} not enabled (n={n}, flag={enabled})")


def push_protected_world(robot, specs, margin: float = 0.02) -> list[str]:
    """Write the protected items' boxes (grown by ``margin``) as the cuRobo world of both MotionGens of both arms,
    each in its arm's base frame (no table, see the module docstring). Checks every slot is enabled. -> cuboid
    names. cuRobo reserves 16 box slots (main/envs/obstacle.py), enough for T1's 2-3 objects."""
    from curobo.geom.types import WorldConfig
    from main.envs.obstacle import _obstacle_in_base

    require_curobo(robot)
    boxes = {f"{PROTECTED_PREFIX}{s.item_id}": item_box(s, margin) for s in specs}
    for planner in (robot.left_planner, robot.right_planner):
        world = {"cuboid": {name: _obstacle_in_base(planner, c, q, h) for name, (c, q, h) in boxes.items()}}
        cfg = WorldConfig.from_dict(world)
        for mg in (planner.motion_gen, planner.motion_gen_batch):
            mg.update_world(cfg)
            for name in boxes:
                _check_enabled(mg, name)
    return list(boxes)


class PlanStatusLog:
    """Failure statuses of cuRobo's plans (RoboTwin's CuroboPlanner drops them; vla_finetuning's PlannerStatusLog):
    wraps plan_single / plan_batch of the four MotionGens once (they are cached across episodes); ``sink`` is reset
    per episode. install_plan_hooks reads it to spot a start state in collision."""

    sink: list = []

    @classmethod
    def install(cls, robot) -> None:
        for planner in (robot.left_planner, robot.right_planner):
            for mg in (planner.motion_gen, planner.motion_gen_batch):
                for name in ("plan_single", "plan_batch"):
                    fn = getattr(mg, name, None)
                    if fn is None or getattr(fn, "_item_status", False) or getattr(fn, "_vf_status", False):
                        continue

                    def wrapped(*a, _fn=fn, _name=name, **k):
                        res = _fn(*a, **k)
                        try:
                            ok = bool(res.success.any()) if hasattr(res.success, "any") else bool(res.success)
                            if not ok:
                                cls.sink.append(f"{_name}:{getattr(res, 'status', '?')}")
                        except Exception:  # noqa: BLE001
                            pass
                        return res

                    wrapped._item_status = True
                    setattr(mg, name, wrapped)


# ------------------------------------------------------------ the hand model ----
def box_spheres(T_w: np.ndarray, lo, hi, r_max: float) -> np.ndarray:
    """Spheres [n, 4] (world xyz, r) inscribed in the box [lo, hi] given in a frame with world pose T_w (radius
    min(r_max, half the smallest side), centres on a grid spaced at most 1.2 r; none sticks out of the box)."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    ext = np.maximum(hi - lo, 1e-4)
    r = min(float(r_max), 0.5 * float(ext.min()))
    axes = []
    for i in range(3):
        span = ext[i] - 2 * r
        n = 1 if span <= 1e-9 else int(np.ceil(span / (1.2 * r))) + 1
        axes.append(np.linspace(lo[i] + r, hi[i] - r, n) if n > 1 else np.array([0.5 * (lo[i] + hi[i])]))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    pw = (T_w[:3, :3] @ grid.T).T + T_w[:3, 3]
    return np.concatenate([pw, np.full((len(pw), 1), r)], 1)


def _pose_T(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), np.float64)


def hand_spheres_world(entity) -> np.ndarray:
    from main.envs.distance import _entity_link, _link_pose

    out = []
    for name, (lo, hi) in HAND_BOXES.items():
        link = _entity_link(entity, name)
        if link is None:
            raise RuntimeError(f"ur5-wsg link {name!r} not found")
        out.append(box_spheres(_pose_T(_link_pose(link)), lo, hi, HAND_R))
    return np.concatenate(out, 0)


def round_axis(half, label: str = "") -> int | None:
    """The symmetry axis of a round held object (bowl / cup / mug / plate by name), else None (a box)."""
    if not any(k in str(label).lower() for k in ROUND):
        return None
    h = np.asarray(half, np.float64)
    gaps = [abs(h[u] - h[v]) / max(h[u], h[v]) for u, v in ((1, 2), (0, 2), (0, 1))]
    return int(np.argmin(gaps))


def payload_spheres_world(actor, budget: int, margin: float, label: str = "") -> np.ndarray:
    """Spheres inscribed in the held object's box grown by ``margin`` (round objects: in the elliptic cylinder),
    the largest radius up to 2 cm that fits in ``budget`` slots."""
    from main.envs.distance import _actor_entity_name, _obb_from_actor

    origin, rot, half = _obb_from_actor(actor)
    half = np.asarray(half, np.float64) + float(margin)
    axis = round_axis(half, f"{label} {_actor_entity_name(actor)}")
    r = 0.02
    while True:
        s = box_spheres(np.eye(4), -half, half, r)
        if axis is not None:
            u, v = [k for k in range(3) if k != axis]
            ru, rv = max(half[u] - s[0, 3], 1e-4), max(half[v] - s[0, 3], 1e-4)
            s = s[(s[:, u] / ru) ** 2 + (s[:, v] / rv) ** 2 <= 1.0 + 1e-9]
        if len(s) <= budget or r > float(half.max()):
            s = s[:budget]
            break
        r *= 1.15
    pw = (rot @ s[:, :3].T).T + origin
    return np.concatenate([pw, s[:, 3:4]], 1)


def sync_arm(env, arm: str, payload_margin: float = 0.01, level: str = "full") -> dict:
    """Write the hand (+ held object) spheres of ``arm`` into its MotionGens' attached_object slots (ee_link frame).
    level "full" = hand + held object, "hand" = hand only, "none" = no spheres."""
    import torch

    from main.envs.distance import _entity_link, _link_pose, detect_held_by_arm

    robot = env.robot
    entity = robot.left_entity if arm == "left" else robot.right_entity
    planner = robot.left_planner if arm == "left" else robot.right_planner
    ee = _entity_link(entity, "ee_link")
    if ee is None:
        raise RuntimeError("ee_link not found")
    T_ee = _pose_T(_link_pose(ee))
    sph = [hand_spheres_world(entity) if level != "none" else np.zeros((0, 4))]
    n_hand = len(sph[0])
    held = (detect_held_by_arm(env) or {}).get(arm)
    n_pay = 0
    if held and level == "full":
        p = payload_spheres_world(held[0], N_SLOTS - n_hand, payload_margin, label=str(held[1]))
        n_pay = len(p)
        sph.append(p)
    allw = np.concatenate(sph, 0)
    if len(allw) > N_SLOTS:
        raise RuntimeError(f"{len(allw)} spheres for {N_SLOTS} slots")
    R, t = T_ee[:3, :3], T_ee[:3, 3]
    local = (R.T @ (allw[:, :3] - t).T).T
    tens = np.zeros((N_SLOTS, 4), np.float32)
    tens[:, 3] = -100.0                                       # unused slots: disabled
    tens[: len(allw), :3] = local
    tens[: len(allw), 3] = allw[:, 3]
    for mg in (planner.motion_gen, planner.motion_gen_batch):
        t_dev = mg.tensor_args.to_device(torch.as_tensor(tens))
        mg.attach_spheres_to_robot(sphere_tensor=t_dev, link_name=LINK)
    return {"n_hand": n_hand, "n_payload": n_pay, "held": held[1] if held else "", "level": level}


def detach_all(robot) -> None:
    for planner in (robot.left_planner, robot.right_planner):
        for mg in (getattr(planner, "motion_gen", None), getattr(planner, "motion_gen_batch", None)):
            if mg is not None:
                try:
                    mg.detach_spheres_from_robot(LINK)
                except Exception as exc:  # noqa: BLE001 - a stock model has no such link
                    print(f"[safe_plan] detach skipped: {exc}")


def _failed(res) -> bool:
    st = res.get("status") if isinstance(res, dict) else res
    if isinstance(st, str):
        return st != "Success"
    try:
        return not any(str(s) == "Success" for s in st)
    except TypeError:
        return True


def install_plan_hooks(env, log: list | None = None, payload_margin: float = 0.01, status_sink=None):
    """Run sync_arm before every RoboTwin plan of either arm (plan_path and the batched grasp search). A plan whose
    START state cuRobo finds in collision (INVALID_START_STATE in ``status_sink``) is retried without the held
    object, then without the hand (only that plan; vla_finetuning smoke B3). Install BEFORE the controller is built,
    so it wraps these hooks. Returns an uninstall function."""
    robot = env.robot
    names = {"left_plan_path": "left", "right_plan_path": "right",
             "left_plan_multi_path": "left", "right_plan_multi_path": "right"}
    orig = {}
    for name, arm in names.items():
        fn = getattr(robot, name)
        orig[name] = fn

        def wrapped(*a, _fn=fn, _arm=arm, _name=name, **k):
            res = None
            for i, level in enumerate(FALLBACK):
                n0 = len(status_sink) if status_sink is not None else 0
                info = sync_arm(env, _arm, payload_margin, level)
                if log is not None:
                    log.append({"call": _name, **info})
                res = _fn(*a, **k)
                start_bad = status_sink is not None and any("INVALID_START_STATE" in s for s in status_sink[n0:])
                if not (start_bad and _failed(res)) or i == len(FALLBACK) - 1:
                    return res
            return res

        setattr(robot, name, wrapped)

    def uninstall():
        for name in orig:
            if name in vars(robot):
                delattr(robot, name)
        detach_all(robot)

    return uninstall


# ---------------------------------------------------------------- contact ----
def contact_with(env, entities) -> tuple[float, bool, set]:
    """(max force N, touched, partner names) for contacts of ``entities`` with anything but the table and each
    other at this physics step: vla_finetuning's contact_scan (CrossSafe's rules: touch = a contact point with
    separation <= 0; force = |impulse| / dt over all points) for several protected entities."""
    names = {e.name or "" for e in entities}
    max_f, touched, partners = 0.0, False, set()
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:  # noqa: BLE001
        return 0.0, False, partners
    dt = float(getattr(env.task.scene, "timestep", 1.0 / 250.0))
    for c in contacts:
        n0, n1 = c.bodies[0].entity.name or "", c.bodies[1].entity.name or ""
        if (n0 in names) == (n1 in names):
            continue
        other = n1 if n0 in names else n0
        if "table" in other.lower():
            continue
        for pt in c.points:
            if float(pt.separation) <= 0.0:
                touched = True
                partners.add(other)
            max_f = max(max_f, float(np.linalg.norm(pt.impulse)) / dt)
    return max_f, touched, partners
