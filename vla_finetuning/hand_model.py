"""The WSG hand and the held object in cuRobo's robot model, for the safe demonstrator (PLAN.md).

RoboTwin's ur5-wsg cuRobo model ends at ee_link (one 1 mm sphere): the WSG body (14.6 x 5 x 7.3 cm) and
its fingers, which reach 15 cm below ee_link, and the object in the hand are invisible to the planner, so
an "obstacle-aware" plan only keeps the arm links off the box (review 2026-10-15, planner F1; the smoke
run's safe attempts came within 0.8 cm or hit it). Here:

- curobo_cfg/ur5-wsg/curobo_safe.yml is RoboTwin's model plus an 'attached_object' link on ee_link with
  256 sphere slots; install_safe_yml() makes every ur5-wsg CuroboPlanner built afterwards in this process
  load it (RoboTwin's files stay untouched).
- sync_arm(env, arm) fills that arm's slots, in both of its MotionGens, with spheres covering the hand's
  collision boxes at the current gripper opening (link poses from SAPIEN) plus the box of the object the
  arm holds (CrossSafe's held-object latch, the one h uses), all expressed in the ee_link frame.
- install_plan_hooks(env) runs sync_arm before every plan RoboTwin makes (plan_path and the batched
  grasp-pose search), so a plan always sees the hand and the payload as they are when it starts. The
  gripper does not move during an arm plan, and the payload moves with ee_link.

Hand boxes are the axis-aligned bounds of the gripper's collision meshes in their link frames
(assets/embodiments/ur5-wsg/meshes/gripper/*.stl; URDF collision origins are all zero).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import common as C

SAFE_YML = C.VF / "curobo_cfg" / "ur5-wsg" / "curobo_safe.yml"
N_SLOTS = 256
LINK = "attached_object"

# link -> (lo, hi) of its collision mesh in the link frame, metres
HAND_BOXES = {
    "wsg_50_base_link": ((-0.073, -0.025, 0.0), (0.073, 0.025, 0.0725)),     # WSG50_110.stl
    "gripper_left": ((-0.018, -0.015, 0.0594), (0.0, 0.015, 0.0965)),        # GUIDE_WSG50_110.stl
    "gripper_right": ((-0.018, -0.015, 0.0594), (0.0, 0.015, 0.0965)),
    "finger_left": ((-0.02, -0.015, 0.0575), (0.003, 0.015, 0.1305)),        # WSG-FMF.stl
    "finger_right": ((-0.02, -0.015, 0.0575), (0.003, 0.015, 0.1305)),
}
HAND_R = 0.02              # largest hand sphere radius (m): 74 spheres for the 5 hand boxes


def install_safe_yml() -> None:
    """Make ur5-wsg CuroboPlanners built from now on in this process load curobo_safe.yml. Call after
    CuroboIKController.install() (it wraps the cached constructor, so the cache keys on our path)."""
    from main.envs.controller import CuroboIKController

    CuroboIKController.install()                  # prepare(): RoboTwin's root on sys.path, then the cache
    from envs.robot.planner import CuroboPlanner

    if getattr(CuroboPlanner.__init__, "_vf_safe_yml", False):
        return
    prev = CuroboPlanner.__init__
    stock = os.path.realpath(str(Path(os.environ.get("ROBOTWIN_ROOT", "/work/nvme/bifq/project/RoboTwin"))
                                 / "assets/embodiments/ur5-wsg/curobo.yml"))

    def init(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
        if yml_path is not None and os.path.realpath(str(yml_path)) == stock:
            yml_path = str(SAFE_YML)
        prev(self, robot_origion_pose, active_joints_name, all_joints, yml_path=yml_path)

    init._vf_safe_yml = True
    CuroboPlanner.__init__ = init


def uses_safe_yml(robot) -> bool:
    return all(os.path.realpath(str(getattr(p, "yml_path", ""))) == os.path.realpath(str(SAFE_YML))
               for p in (robot.left_planner, robot.right_planner))


def box_spheres(T_w: np.ndarray, lo, hi, r_max: float) -> np.ndarray:
    """Spheres [n, 4] (world xyz, r) inscribed in the box [lo, hi] given in a frame with world pose T_w:
    radius r = min(r_max, half the box's smallest side), centres on a grid over [lo + r, hi - r] spaced at
    most 1.2 r apart. No sphere sticks out of the box (the cell-sized spheres of the first version bulged
    about 1 cm past every face, and the hand at a grasp next to the box then started "in collision": smoke B2,
    10-15); the uncovered pockets along edges and corners are at most 0.2 r deep and are left to the box
    margin and the keep rule's contact check."""
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


def payload_spheres_world(actor, budget: int, margin: float, label: str = "") -> np.ndarray:
    """Spheres inscribed in the held object's box (CrossSafe's OBB of the actor) grown by `margin`, with
    the largest radius up to 2 cm that fits in `budget` slots (a larger radius means fewer spheres)."""
    from main.envs.distance import _obb_from_actor

    origin, rot, half = _obb_from_actor(actor)
    half = np.asarray(half, np.float64) + float(margin)
    from main.envs.distance import _actor_entity_name

    axis = round_axis(half, f"{label} {_actor_entity_name(actor)}")   # task attribute name + model name
    r = 0.02
    while True:
        s = box_spheres(np.eye(4), -half, half, r)            # in the object's frame
        if axis is not None:                                  # bowls, cups: drop the box's square corners
            u, v = [k for k in range(3) if k != axis]
            ru, rv = max(half[u] - s[0, 3], 1e-4), max(half[v] - s[0, 3], 1e-4)
            s = s[(s[:, u] / ru) ** 2 + (s[:, v] / rv) ** 2 <= 1.0 + 1e-9]
        if len(s) <= budget or r > float(half.max()):
            if len(s) > budget:
                print(f"[hand] payload needs {len(s)} spheres for {budget} slots: truncated")
            s = s[:budget]
            break
        r *= 1.15
    pw = (rot @ s[:, :3].T).T + origin
    return np.concatenate([pw, s[:, 3:4]], 1)


ROUND = ("bowl", "cup", "mug", "plate")


def round_axis(half, label: str = "") -> int | None:
    """The symmetry axis of a round held object (the model name says bowl / cup / mug / plate): the axis whose
    two other half extents are closest to each other (002_bowl 8.95 / 8.95 cm around its 2.6 cm axis). Other
    objects (blocks, bread, the burger and fries) stay boxes: None. The spheres are then inscribed in the elliptic
    cylinder around that axis (smoke B4, 10-19: the box corners of a grasped bowl overlapped the obstacle while the
    bowl itself was 3 cm away, so every plan after the grasp started 'in collision')."""
    if not any(k in str(label).lower() for k in ROUND):
        return None
    h = np.asarray(half, np.float64)
    gaps = [abs(h[u] - h[v]) / max(h[u], h[v]) for u, v in ((1, 2), (0, 2), (0, 1))]
    return int(np.argmin(gaps))

def sync_arm(env, arm: str, payload_margin: float = 0.01, level: str = "full") -> dict:
    """Write the hand (+ held object) spheres of `arm` into its MotionGens' attached_object slots, in the
    ee_link frame. level "full" = hand + held object, "hand" = hand only, "none" = no spheres (RoboTwin's
    stock model). -> {"n_hand", "n_payload", "held", "level"}."""
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
    local = (R.T @ (allw[:, :3] - t).T).T                     # world -> ee_link frame
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
                    print(f"[hand] detach skipped: {exc}")


FALLBACK = ("full", "hand", "none")


def _failed(res) -> bool:
    """RoboTwin's plan results: plan_path -> {"status": "Success" | "Fail", ...}; the batched grasp search ->
    {"status": [per-pose statuses], ...}. Failed = no pose succeeded."""
    st = res.get("status") if isinstance(res, dict) else res
    if isinstance(st, str):
        return st != "Success"
    try:
        return not any(str(s) == "Success" for s in st)
    except TypeError:
        return True


def install_plan_hooks(env, log: list | None = None, payload_margin: float = 0.01, status_sink=None):
    """Run sync_arm before every RoboTwin plan of either arm (plan_path and the batched grasp search).

    A plan that fails because cuRobo finds the START state in collision (INVALID_START_STATE_WORLD_COLLISION
    in `status_sink`, the PlannerStatusLog list) is retried with a lighter model: first without the held
    object, then without the hand (smoke B3, 10-16: right after a grasp next to the box, the box around a
    round bowl already overlaps the grown obstacle while the bowl itself does not). Only that one plan is
    relaxed; the next plan starts from the full model again, and the keep rule still checks the real contact
    at every physics step. Returns an uninstall function. `log` collects one entry per sync."""
    robot = env.robot
    names = {"left_plan_path": "left", "right_plan_path": "right",
             "left_plan_multi_path": "left", "right_plan_multi_path": "right"}
    orig = {}
    for name, arm in names.items():
        fn = getattr(robot, name)
        orig[name] = fn

        def wrapped(*a, _fn=fn, _arm=arm, _name=name, **k):
            for i, level in enumerate(FALLBACK):
                n0 = len(status_sink) if status_sink is not None else 0
                info = sync_arm(env, _arm, payload_margin, level)
                if log is not None:
                    log.append({"call": _name, **info})
                res = _fn(*a, **k)
                start_bad = status_sink is not None and any(
                    "INVALID_START_STATE" in s for s in status_sink[n0:])
                if not (start_bad and _failed(res)) or i == len(FALLBACK) - 1:
                    return res
                if log is not None:
                    log[-1]["fallback"] = True
            return res

        setattr(robot, name, wrapped)

    def uninstall():
        for name in orig:
            if name in vars(robot):
                delattr(robot, name)          # back to the class's bound method
        detach_all(robot)

    return uninstall
