"""Obstacle-aware planning for the "safe" nominal mode (Yuxuan 2026-10-22: the expert for safe demonstrations).

The careless nominal plans with a cuRobo world that holds the table only, so the arm goes straight through the
items. The safe mode (T1 so far) plans around the PROTECTED items instead (every extra object when no rule is named):

- the world (``ProtectedWorld``, written to both MotionGens of both arms, each in its arm's base frame): every
  protected item as the convex hull of its PhysX collision shapes with each face pushed out by a margin, as a cuRobo
  mesh (2026-10-27; the OBB boxes used before overshoot round and tapered objects by 1.4-3.7 cm, item_shapes.py), its
  OBB box when no hull can be built or past the reserved mesh slots; plus the real tabletop with its top 2 cm low
  (RoboTwin's own table cuboid misses the real table, vla_finetuning/common.py push_world; the 2 cm keep an object
  just grasped, grown by its margin, clear of it). It is re-written from the live poses before every plan when an item
  moved (items can be nudged);
- RoboTwin's ur5-wsg cuRobo model ends at ee_link, so the WSG hand and the held object are invisible to the
  planner. ``install_safe_yml`` makes every ur5-wsg CuroboPlanner built afterwards load a copy of the model with an
  ``attached_object`` link of 256 sphere slots (configs/curobo_safe/ur5-wsg.yml), and ``install_plan_hooks`` fills
  them before every plan with spheres covering the hand's collision boxes and the held object (``sync_arm``);
- a failed plan is retried down a ladder (``plan_levels``): the same model at smaller margins (1 -> 0.5 -> 0.2 cm, the
  held object's margin too), each failure kind down to its floor (``RETRY_FLOORS``, yaml ``retry_floors``: a start in
  collision, "Invalid Problem" included, and a goal without IK to 0.2 cm; no route and a goal in collision to 0.5 cm,
  a route blocked at 0.5 cm is not taken at 0.2); then, for a start in collision only, without the held object, then
  without the hand. The batched grasp search's start state is checked first: cuRobo's batch planner checks none, and
  its graph planner then rejects the whole batch ("Invalid Problem", which the 10-22 ladder missed);
- the grasp search escalates as a whole (the stock grasp at the largest margin wins whenever it exists): RoboTwin's
  search over every contact point with the stock poses at the base margin first; only if no contact point plans, the
  whole search again one ladder level per pass, with ten approach tilts for ur5-wsg's one stock pose, up to the first
  level at which a contact point plans. Each plan's level and every failed try go to task.episode_info
  (``safe_plan_log``, the last entries; the plan the task used is booked by the controller, ``record_plan``);
- no fallback planner: a plan the ladder cannot find ends the episode as a plan failure with its reason in
  task.episode_info["safe_fail"], and the planner must be cuRobo on both arms (``require_curobo``).

The hand model is vla_finetuning/hand_model.py's (the CrossSafe safe demonstrations, 500 kept); the item world is
new. The keep rule (no contact at any physics step) belongs to the demo collector, ``contact_with`` helps it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

SAFE_YML = Path(__file__).resolve().parent / "configs" / "curobo_safe" / "ur5-wsg.yml"
N_SLOTS = 256
LINK = "attached_object"
PROTECTED_PREFIX = "protected_"
TABLE_NAME = "safe_table"
# cuRobo world slots reserved by main/envs/obstacle.py patch_curobo_collision_cache ({"obb": 16, "mesh": 8}): more
# meshes make cuRobo build a new mesh cache, which the MotionGens' recorded CUDA graphs never read
OBB_SLOTS = 16
MESH_SLOTS = 8
TABLE_CLEARANCE = 0.02           # m: the planner's tabletop sits this far below the real one
MARGINS = (0.01, 0.005, 0.002)   # m: the margin ladder (protected shapes and held object grown by it), tried in turn
DROP_MODELS = ("hand", "none")   # then, for a start in collision only: without the held object, without the hand
LOG_MAX = 64                     # task.episode_info["safe_plan_log"] keeps the last this many entries

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

# a cuRobo status as PlanStatusLog records it (lower case, spaces as "_") -> the failure kind; first match wins
_KINDS = (
    ("exc:", "planner_error"),            # the planner raised (RoboTwin's plan_batch swallows it)
    ("joint_limits", "joint_limits"),     # before "start_state": INVALID_START_STATE_JOINT_LIMITS
    ("start_state", "start_in_collision"),  # INVALID_START_STATE_* (plan_single, check_start_state)
    ("invalid_problem", "start_in_collision"),  # plan_batch: the graph planner found the start (or a goal) in collision
    ("ik_fail", "ik_fail"),
    ("invalid_query", "invalid_query"),   # plan_single: the graph planner found the goal in collision
    ("partial_pose", "constraint"),       # a constrained (held-orientation) move that cannot keep it
    ("trajopt", "no_route"),
    ("graph_fail", "no_route"),
    ("dt_exce", "no_route"),
)
# failure kind -> the smallest margin (m) a failed plan of that kind is retried at; a kind not listed is not retried
# (review of 2026-10-27: a smaller margin lets a blocked route pass closer to the items, and the hand's box corners
# stick out of its spheres by up to 1.5 cm, so no route and a goal in collision stop at 0.5 cm)
RETRY_FLOORS = {"start_in_collision": 0.002, "ik_fail": 0.002, "no_route": 0.005, "invalid_query": 0.005}
DROP_RETRY = ("start_in_collision",)


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


def hull_points(spec) -> np.ndarray | None:
    """Vertices (entity frame) of every PhysX collision piece of the item (item_shapes.shape_of: the shapes the
    simulator touches with; the OBB box for an item without any), or None."""
    from . import item_shapes

    shape = item_shapes.shape_of(spec, fallback_obb=(spec.obb_center_local, spec.half_extents))
    if shape is None or not shape.pieces:
        return None
    return np.concatenate([np.asarray(p.verts, np.float64) for p in shape.pieces], axis=0)


def grown_hull(verts, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [V,3], triangles [F,3]) of the convex hull of ``verts`` with every face pushed out by ``margin``: the
    hull's planes n.x <= d moved to d + margin and intersected, a polytope containing every point within ``margin``
    of the hull (its corners stick out a little more than a rounded offset would)."""
    from scipy.spatial import HalfspaceIntersection

    from .item_shapes import _hull

    v, f, planes = _hull(np.asarray(verts, np.float64))
    if float(margin) <= 0.0:
        return v, f
    planes = np.unique(np.round(planes, 9), axis=0)       # one halfspace per face plane (trimesh splits faces)
    hs = np.concatenate([planes[:, :3], -(planes[:, 3:4] + float(margin))], axis=1)   # n.x - (d + m) <= 0
    v, f, _ = _hull(HalfspaceIntersection(hs, v.mean(axis=0)).intersections)
    return v, f


def table_box(task, clearance: float = TABLE_CLEARANCE):
    """(world centre, quat wxyz, half extents) of the planner's tabletop: the real one (the largest box collision shape
    of ``task.table``; RoboTwin create_table: 1.2 x 0.7 x 0.05 m, its top face at the table height) moved ``clearance``
    down, its thickness kept. None when the task has no table entity or the table has no box shape."""
    import sapien.physx as px

    from .items import entity_of

    table = getattr(task, "table", None)
    if table is None:
        return None
    ent = entity_of(table)
    comp = ent.find_component_by_type(px.PhysxRigidBaseComponent)
    boxes = [s for s in (comp.get_collision_shapes() if comp is not None else [])
             if isinstance(s, px.PhysxCollisionShapeBox)]
    if not boxes:
        return None
    top = max(boxes, key=lambda s: float(np.prod(np.asarray(s.get_half_size(), np.float64)[:2])))
    pose = ent.get_pose() * top.get_local_pose()
    centre = np.asarray(pose.p, np.float64) - np.array([0.0, 0.0, float(clearance)])
    return centre, np.asarray(pose.q, np.float64), np.asarray(top.get_half_size(), np.float64)


def world_in_base(planner, obstacles) -> dict:
    """cuRobo world dict of the world-frame ``obstacles`` (ProtectedWorld.obstacles) in ``planner``'s arm base frame:
    CrossSafe's _obstacle_in_base, RoboTwin's world -> base transform plus the planner's frame bias, as its targets
    get. A mesh's vertices stay in its item frame, only its pose moves."""
    from main.envs.obstacle import _obstacle_in_base

    out = {"cuboid": {}, "mesh": {}}
    for o in obstacles:
        if o["kind"] == "mesh":
            pose = _obstacle_in_base(planner, o["p"], o["q"], np.zeros(3))["pose"]
            out["mesh"][o["name"]] = {"pose": pose, "vertices": np.asarray(o["verts"], np.float64).tolist(),
                                      "faces": np.asarray(o["faces"], np.int64).tolist()}
        else:
            out["cuboid"][o["name"]] = _obstacle_in_base(planner, o["p"], o["q"], o["half"])
    return out


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


def _check_mesh_enabled(mg, name: str) -> None:
    wcc = mg.world_coll_checker
    names = wcc._env_mesh_names[0]
    if name not in names:
        raise RuntimeError(f"cuRobo world has no mesh {name!r} after update_world")
    idx = names.index(name)
    n = int(wcc._env_n_mesh[0])
    enabled = int(wcc._mesh_tensor_list[2][0, idx])
    if not (idx < n and enabled == 1):
        raise RuntimeError(f"cuRobo mesh {name!r} slot {idx} not enabled (n={n}, flag={enabled})")


def _mesh_cache_ref(wcc):
    t = getattr(wcc, "_mesh_tensor_list", None)
    if t is None:
        return None
    return t[0].data_ptr() if hasattr(t[0], "data_ptr") else id(t[0])


def load_world(mg, world: dict) -> None:
    """Replace a MotionGen's world with ``world`` (world_in_base's dict) and check every slot is enabled. Guards two
    cuRobo behaviours: a world without cuboids (meshes) leaves the previous ones enabled, so those slots are switched
    off here; and meshes past the reserved slots rebuild the mesh cache, which the recorded CUDA graphs never see."""
    from curobo.geom.types import WorldConfig

    boxes, meshes = dict(world.get("cuboid") or {}), dict(world.get("mesh") or {})
    if len(boxes) > OBB_SLOTS or len(meshes) > MESH_SLOTS:
        raise RuntimeError(f"{len(boxes)} cuboids / {len(meshes)} meshes for {OBB_SLOTS} / {MESH_SLOTS} cuRobo slots")
    wcc = mg.world_coll_checker
    ref = _mesh_cache_ref(wcc)
    mg.update_world(WorldConfig.from_dict({"cuboid": boxes, "mesh": meshes}))
    if meshes and _mesh_cache_ref(wcc) != ref:
        raise RuntimeError("cuRobo rebuilt its mesh cache: the recorded CUDA graphs would not see the meshes")
    if not boxes and getattr(wcc, "_cube_tensor_list", None) is not None:
        wcc._cube_tensor_list[2][0, :] = 0
        wcc._env_n_obbs[0] = 0
    if not meshes and getattr(wcc, "_mesh_tensor_list", None) is not None:
        wcc._mesh_tensor_list[2][0, :] = 0
        wcc._env_n_mesh[0] = 0
    for name in boxes:
        _check_enabled(mg, name)
    for name in meshes:
        _check_mesh_enabled(mg, name)


class ProtectedWorld:
    """The safe mode's cuRobo world: the protected items and the table, written from the live poses to the two
    MotionGens of each arm, in its base frame (``push``).

    shape "hull": an item is the convex hull of its collision shapes grown by the margin (a cuRobo mesh, named by its
    geometry: cuRobo caches meshes by name, the MotionGens outlive the episode), its OBB box when no hull can be
    built or past MESH_SLOTS items; "obb": the OBB boxes grown by the margin (the 10-22 world). table: the real
    tabletop, its top ``table_clearance`` low (table_box). ``stats`` (a dict, updated in place) is the world's part
    of meta.diag.safe_plan."""

    def __init__(self, robot, specs, task=None, margin: float = MARGINS[0], shape: str = "hull", table: bool = True,
                 table_clearance: float = TABLE_CLEARANCE):
        if shape not in ("hull", "obb"):
            raise ValueError(f"shape must be 'hull' or 'obb', got {shape!r}")
        require_curobo(robot)
        self.robot = robot
        self.specs = list(specs)
        self.margin = float(margin)
        self.shape = shape
        self._points: dict = {}       # item_id -> hull points | None
        self._hulls: dict = {}        # (item_id, margin) -> (verts, faces, key) | None
        self._pushed: dict = {}       # arm -> (signature, obstacle names) of its last write
        self._mesh_names: set = set()
        self.table = None
        table_error = None
        if table:
            try:
                self.table = table_box(task, table_clearance) if task is not None else None
            except Exception as exc:  # noqa: BLE001 - no table is a weaker world, not a crash
                table_error = f"{type(exc).__name__}: {exc}"
        self.stats = {
            "margin": self.margin, "shape": shape, "items": {}, "n_writes": 0,
            "table": None if self.table is None else {"top_z": float(self.table[0][2] + self.table[2][2]),
                                                      "clearance": float(table_clearance)},
        }
        if table and self.table is None:
            self.stats["table_missing"] = table_error or "no table entity"

    def _hull(self, spec, margin: float):
        key = (spec.item_id, round(float(margin), 6))
        if key not in self._hulls:
            out = None
            try:
                if spec.item_id not in self._points:
                    self._points[spec.item_id] = hull_points(spec)
                pts = self._points[spec.item_id]
                if pts is not None:
                    v, f = grown_hull(pts, margin)
                    out = (v, f, hashlib.sha1(np.round(v, 6).tobytes()).hexdigest()[:10])
            except Exception as exc:  # noqa: BLE001 - the item falls back to its box
                self.stats["items"].setdefault(spec.item_id, {})["hull_error"] = f"{type(exc).__name__}: {exc}"[:200]
            self._hulls[key] = out
        return self._hulls[key]

    def obstacles(self, margin: float | None = None, table: bool = True) -> list[dict]:
        """World-frame obstacles at ``margin`` (the base margin by default): {"name", "kind": "mesh" | "cuboid",
        "item_id", "p", "q", and "verts", "faces" (item frame) | "half"}; ``table`` False leaves the table out."""
        from .items import entity_of

        m = self.margin if margin is None else float(margin)
        out = []
        for k, spec in enumerate(self.specs):
            hull = self._hull(spec, m) if self.shape == "hull" and k < MESH_SLOTS else None
            if hull is not None:
                pose = entity_of(spec).get_pose()
                v, f, key = hull
                out.append({"name": f"{PROTECTED_PREFIX}{spec.item_id}_{key}", "kind": "mesh", "item_id": spec.item_id,
                            "p": np.asarray(pose.p, np.float64), "q": np.asarray(pose.q, np.float64),
                            "verts": v, "faces": f})
            else:
                c, q, h = item_box(spec, m)
                out.append({"name": f"{PROTECTED_PREFIX}{spec.item_id}", "kind": "cuboid", "item_id": spec.item_id,
                            "p": c, "q": q, "half": h})
            mesh = out[-1]["kind"] == "mesh"
            self.stats["items"].setdefault(spec.item_id, {}).update(
                {"shape": "hull" if mesh else "obb", "category": spec.category,
                 "n_vertices": int(len(out[-1]["verts"])) if mesh else 8})
        if self.table is not None and table:
            c, q, h = self.table
            out.append({"name": TABLE_NAME, "kind": "cuboid", "item_id": None, "p": c, "q": q, "half": h})
        return out

    def names(self, arm: str) -> list[str]:
        """The obstacle names last written to ``arm``'s MotionGens."""
        return list(self._pushed.get(arm, (None, []))[1])

    def push(self, arm: str | None = None, margin: float | None = None, table: bool = True) -> bool:
        """Write the world at ``margin`` (the base margin by default) from the live poses to the MotionGens of
        ``arm`` (both arms by default); an arm whose last write had the same obstacles at the same poses (0.1 mm,
        1e-3 quaternion) is skipped. ``table`` False: without the table (install_plan_hooks' table-free descent
        retry). -> True if anything was written."""
        m = self.margin if margin is None else float(margin)
        obs = self.obstacles(m, table=table)
        sig = (round(m, 6), tuple((o["name"], tuple(np.round(o["p"], 4)), tuple(np.round(o["q"], 3))) for o in obs))
        wrote = False
        for side in (("left", "right") if arm is None else (str(arm),)):
            if self._pushed.get(side, (None,))[0] == sig:
                continue
            planner = self.robot.left_planner if side == "left" else self.robot.right_planner
            world = world_in_base(planner, obs)
            for mg in (planner.motion_gen, planner.motion_gen_batch):
                load_world(mg, world)
            self._pushed[side] = (sig, [o["name"] for o in obs])
            self._mesh_names |= set(world["mesh"])
            self.stats["n_writes"] += 1
            wrote = True
        return wrote

    def clear(self) -> None:
        """End of the episode: switch off every mesh slot of the four MotionGens and drop this world's warp meshes
        (the MotionGens outlive the episode, and the next episode's table-only world would leave the meshes on)."""
        for planner in (getattr(self.robot, "left_planner", None), getattr(self.robot, "right_planner", None)):
            for mg in (getattr(planner, "motion_gen", None), getattr(planner, "motion_gen_batch", None)):
                wcc = getattr(mg, "world_coll_checker", None)
                if wcc is None:
                    continue
                if getattr(wcc, "_mesh_tensor_list", None) is not None:
                    wcc._mesh_tensor_list[2][:] = 0
                    wcc._env_n_mesh[:] = 0
                cache = getattr(wcc, "_wp_mesh_cache", None)
                if isinstance(cache, dict):
                    for name in self._mesh_names:
                        cache.pop(name, None)
        self._pushed = {}


def push_protected_world(robot, specs, margin: float = 0.02) -> list[str]:
    """The 10-22 world (kept for scripts): the protected items' OBB boxes grown by ``margin`` as the whole cuRobo
    world of both MotionGens of both arms, no table. -> obstacle names."""
    world = ProtectedWorld(robot, specs, margin=margin, shape="obb", table=False)
    world.push()
    return world.names("left")


def world_options(safe: dict) -> dict:
    """ProtectedWorld keyword arguments from a task yaml's safe_mode section."""
    return {"margin": float(safe.get("margin", MARGINS[0])), "shape": str(safe.get("shape", "hull")),
            "table": bool(safe.get("table", True)),
            "table_clearance": float(safe.get("table_clearance", TABLE_CLEARANCE))}


def margin_ladder(safe: dict) -> tuple[float, ...]:
    """The margins of the retry ladder from a safe_mode section: ``margin``, then the smaller ``retry_margins``."""
    base = float(safe.get("margin", MARGINS[0]))
    retry = sorted({float(m) for m in (safe.get("retry_margins") or []) if float(m) < base}, reverse=True)
    return (base, *retry)


def retry_floors(safe: dict) -> dict:
    """The smallest retry margin per failure kind from a safe_mode section (``retry_floors``, RETRY_FLOORS without)."""
    raw = safe.get("retry_floors")
    return dict(RETRY_FLOORS) if raw is None else {str(k): float(v) for k, v in dict(raw).items()}


def hook_options(safe: dict) -> dict:
    """install_plan_hooks keyword arguments from a task yaml's safe_mode section."""
    drop = safe.get("drop_models", DROP_MODELS)
    lim = safe.get("grasp_rotate_lim")
    return {"payload_margin": float(safe.get("payload_margin", 0.01)), "margins": margin_ladder(safe),
            "drop_models": tuple(str(d) for d in (drop or ())),
            "wide_grasp": None if not lim else [float(lim[0]), float(lim[1])], "floors": retry_floors(safe),
            "table_free_descents": bool(safe.get("table_free_descents", False)),
            "descent_tip_len": None if safe.get("descent_tip_len") is None else float(safe["descent_tip_len"]),
            "descent_tip_clear": float(safe.get("descent_tip_clear", 0.005))}


class PlanStatusLog:
    """Failure statuses of cuRobo's plans (RoboTwin's CuroboPlanner drops them; vla_finetuning's PlannerStatusLog):
    wraps plan_single / plan_batch of the four MotionGens once (they are cached across episodes); ``sink`` is reset
    per episode. An exception is recorded as "<name>:EXC:<type>: <message>" and raised on (RoboTwin's plan_batch
    swallows it and returns a blank failure). install_plan_hooks reads it to classify each failed plan."""

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
                        try:
                            res = _fn(*a, **k)
                        except Exception as exc:
                            cls.sink.append(f"{_name}:EXC:{type(exc).__name__}: {str(exc)[:160]}")
                            raise
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


# --------------------------------------------------------- the retry ladder ----
def _failed(res) -> bool:
    st = res.get("status") if isinstance(res, dict) else res
    if isinstance(st, str):
        return st != "Success"
    try:
        return not any(str(s) == "Success" for s in st)
    except TypeError:
        return True


def failure_kind(statuses) -> str:
    """The kind of a failed plan from the cuRobo statuses PlanStatusLog recorded for it (the last one that names a
    kind): planner_error, joint_limits, start_in_collision, ik_fail, invalid_query, constraint, no_route; plan_fail
    when none does."""
    for s in reversed(list(statuses)):
        t = str(s).lower().replace(" ", "_")
        for needle, kind in _KINDS:
            if needle in t:
                return kind
    return "plan_fail"


def plan_levels(margins=MARGINS, drop_models=DROP_MODELS) -> list[tuple[float, str]]:
    """The retry ladder, (margin, model) per try: the full model (hand + held object) at each margin, largest first,
    then each dropped model at the smallest margin."""
    ms = [float(m) for m in margins]
    return [(m, "full") for m in ms] + [(ms[-1], str(d)) for d in drop_models]


def escalate(kind: str, margin: float, model: str, floors: dict | None = None) -> bool:
    """Whether a try that failed with ``kind`` goes on to the level (``margin``, ``model``). A smaller margin helps
    a start in collision, a goal without IK, a blocked route and a goal in collision, each down to its floor
    (``floors``, RETRY_FLOORS by default; a kind without one is not retried); a lighter model only a start in collision
    (the start is a real, contact-free state the model reads as colliding; a lighter model at the goal would let the
    hand into the items), at a margin its floor allows. The ladder never grows its margin, so the first level
    refused ends it."""
    floors = RETRY_FLOORS if floors is None else floors
    if kind not in floors or float(margin) < float(floors[kind]) - 1e-9:
        return False
    return model == "full" or kind in DROP_RETRY


def _self_collision_cost(mg):
    return getattr(getattr(mg, "rollout_fn", None), "robot_self_collision_constraint", None)


def _restore_self_collision(mg, was_on: bool) -> None:
    """cuRobo 0.7.8's check_start_state (plan_single runs it too) switches both collision costs off to name the
    problem and returns WORLD_COLLISION before switching the self-collision one back on: put it back."""
    cost = _self_collision_cost(mg)
    if cost is not None and was_on and not bool(getattr(cost, "enabled", True)):
        cost.enable_cost()


def _joint_state(planner, mg, qpos):
    """cuRobo JointState of the arm's joints in ``qpos`` (its entity's full qpos), as RoboTwin's planner builds it."""
    import torch
    from curobo.types.robot import JointState

    names = list(planner.active_joints_name)
    q = [round(float(qpos[planner.all_joints.index(n)]), 5) for n in names if n in planner.all_joints]
    return JointState.from_position(mg.tensor_args.to_device(torch.as_tensor(q).reshape(1, -1)), joint_names=names)


def check_start(planner, mg, qpos, names=()) -> dict:
    """cuRobo's start-state check (MotionGen.check_start_state: joint limits, world and self collision of the arm's
    sphere model with its attached spheres; plan_single runs it, the batch planner does not) for ``qpos``. A start
    in collision is checked again with each of ``names`` (world obstacles) switched off in turn: those that free it are
    its ``culprits``. -> {"ok": bool | None (the check could not run), "status", "culprits"}"""
    cost = _self_collision_cost(mg)
    was_on = bool(getattr(cost, "enabled", False))
    try:
        js = _joint_state(planner, mg, qpos)
        ok, status = mg.check_start_state(js)
    except Exception as exc:  # noqa: BLE001 - a check that cannot run never blocks the plan
        _restore_self_collision(mg, was_on)
        return {"ok": None, "status": f"check_error:{type(exc).__name__}: {str(exc)[:120]}", "culprits": []}
    _restore_self_collision(mg, was_on)
    out = {"ok": bool(ok), "status": "ok" if ok else str(status), "culprits": []}
    if not ok:
        wcc = mg.world_coll_checker
        for name in names:
            try:
                wcc.enable_obstacle(name, False)
                freed = bool(mg.check_start_state(js)[0])
            except Exception:  # noqa: BLE001
                freed = False
            finally:
                try:
                    wcc.enable_obstacle(name, True)
                except Exception:  # noqa: BLE001
                    pass
                _restore_self_collision(mg, was_on)
            if freed:
                out["culprits"].append(name)
    return out


def _pose_list(pose) -> list | None:
    if pose is None:
        return None
    if hasattr(pose, "p") and hasattr(pose, "q"):
        arr = np.concatenate([np.asarray(pose.p, np.float64), np.asarray(pose.q, np.float64)])
    else:
        arr = np.asarray(pose, np.float64).reshape(-1)
    return [round(float(x), 4) for x in arr[:7]]


def raised_goal(goal, table_top: float, tip_len: float, clear: float = 0.005) -> list | None:
    """``goal`` (a gripper pose [x, y, z, qw, qx, qy, qz]) moved straight up until the fingertips, ``tip_len`` along
    the approach axis (the gripper x axis), sit ``clear`` above ``table_top``; None when the approach does not point
    down (its z component above -0.5) or the fingertips already clear the table. ur5-wsg: tip_len 0.164 m, the
    distance push_via's tip_len was measured at (fingertips 8 mm above the table at gripper z = table + 0.172)."""
    g = _pose_list(goal)
    if g is None or len(g) < 7:
        return None
    w, x, y, z = (float(v) for v in g[3:7])
    a_z = 2.0 * (x * z - y * w)                    # z component of the rotation's first column (the approach axis)
    if a_z > -0.5:
        return None
    dz = float(table_top) + float(clear) - (float(g[2]) + float(tip_len) * a_z)
    if dz <= 1e-4:
        return None
    return [g[0], g[1], float(g[2]) + dz, *g[3:7]]


def _note(info: dict, entry: dict) -> None:
    """Append to the plan log, a ring of the last LOG_MAX entries (a list, JSON-safe); the older ones are counted."""
    log = info.setdefault("safe_plan_log", [])
    log.append(entry)
    if len(log) > LOG_MAX:
        n = len(log) - LOG_MAX
        del log[:n]
        info["safe_plan_log_dropped"] = int(info.get("safe_plan_log_dropped", 0)) + n


def _count(info: dict, level) -> None:
    levels = info.setdefault("safe_plan_levels", {})
    levels[str(level)] = int(levels.get(str(level), 0)) + 1


def _fail_final(info: dict, kind: str, entry: dict | None) -> None:
    """A plan the task used failed for good: "fail" counted, its kind and its last failed try kept."""
    _count(info, "fail")
    info["safe_last_fail"] = kind
    if entry is not None:
        info["safe_fail_entry"] = entry


def record_plan(info: dict, result) -> str | None:
    """Book the plan the controller kept (RiskTickController.plan in T1 safe mode, from the tags install_plan_hooks
    put on it; the controller's branch-flip re-plans it did not keep, and the batch plans of a grasp search, are not
    booked here): planned, its level into ``safe_plan_levels`` and ``safe_last_fail`` cleared; failed, "fail", its
    kind as ``safe_last_fail`` and its last failed try as ``safe_fail_entry``. -> the failure kind, None if planned."""
    res = result if isinstance(result, dict) else {}
    if res.get("status") == "Success":
        if "_safe_level" in res:
            _count(info, res["_safe_level"])
        info.pop("safe_last_fail", None)
        return None
    kind = str(res.get("_safe_fail") or "plan_fail")
    _fail_final(info, kind, res.get("_safe_fail_entry"))
    return kind


def _pose_key(pose):
    """A pose (list / array, or None) as a hashable key, 0.01 mm; the grasp search keys its contact points by it."""
    if pose is None:
        return None
    try:
        return tuple(np.round(np.asarray(pose, np.float64).reshape(-1), 5).tolist())
    except (TypeError, ValueError):
        return str(pose)


def install_plan_hooks(env, log: list | None = None, payload_margin: float = 0.01, status_sink=None, world=None,
                       margins=None, drop_models=DROP_MODELS, info: dict | None = None, wide_grasp=None,
                       sync=None, floors=None, table_free_descents: bool = False, descent_tip_len: float | None = None,
                       descent_tip_clear: float = 0.005):
    """Run sync_arm before every RoboTwin plan of either arm (plan_path and the batched grasp search), down the retry
    ladder ``plan_levels(margins, drop_models)``: before each try ``world`` (a ProtectedWorld) is re-written at the
    try's margin from the live poses and the hand + held object synced (the held object grown by the same margin, at
    most ``payload_margin``); a batch's start state is checked first and a start in collision skips the planner
    (except on the last level). A failed try goes on only as ``escalate`` allows (``floors``: the smallest margin per
    failure kind, RETRY_FLOORS by default). margins None: one level at ``payload_margin`` (the 10-22 ladder, plus
    "Invalid Problem"). ``table_free_descents``: a straight-line move (a constraint pose: RoboTwin's grasp and place
    descents) whose goal has no IK on every rung is tried once more per allowed margin without the table (level
    "no_table"; the items stay), since RoboTwin's stock grasp of a flat object asks for the fingertips below the
    tabletop. Every result is tagged: "_safe_level" (the level it planned at), or "_safe_fail" (its kind) and
    "_safe_fail_entry" (its last failed try); RiskTickController.plan books the plan it keeps (record_plan).

    The grasp search (choose_grasp_pose, guarded) escalates as a whole, so the stock grasp at the largest margin wins
    whenever one exists: RoboTwin's search over every contact point with the stock poses at the base margin first;
    only if no contact point plans, the whole search again one level per pass (from the base margin again when
    ``wide_grasp`` [lo, hi] rad gives an arm without a rotate range of its own, ur5-wsg: ten copies of one pose, ten
    approach tilts, the first the stock pose), up to the first level at which any contact point plans. A contact point
    is planned at a pass with the same candidates as its previous one only as ``escalate`` allows that pass's failure.
    No contact point gives a pose: ``safe_fail`` "no_grasp" ("planner_error" when cuRobo raised during the search,
    which then stops), plan_success False, RoboTwin's placeholder pose instead of an assertion.

    Diagnostics go to ``info`` (default task.episode_info): ``safe_plan_log`` (the last LOG_MAX entries: every failed
    try with its status, kind, skill, goal, start check and culprits, every plan that needed a level > 0, every grasp
    search that escalated or failed; ``safe_plan_log_dropped`` counts the older ones), ``safe_plan_levels`` (per plan
    the task used, a controller plan or a whole grasp search: the level it needed, or "fail"), ``safe_last_fail`` (the
    kind of the latest such plan if it failed; finish_episode turns it into ``safe_fail``), ``safe_fail_entry`` (the
    entry of the latest such failure), ``safe_wide_grasp`` (grasp searches past the stock pass). Install BEFORE the
    controller is built, so it wraps these hooks. Returns an uninstall function."""
    from main.envs.controller import _actor_label, current_skill

    robot, task = env.robot, env.task
    sync = sync or sync_arm
    floors = dict(RETRY_FLOORS if floors is None else floors)
    if info is None:
        if not isinstance(getattr(task, "episode_info", None), dict):
            task.episode_info = {}
        info = task.episode_info
    levels_all = plan_levels(margins if margins is not None else (payload_margin,), drop_models)
    info["safe_plan_ladder"] = [[m, model] for m, model in levels_all]
    # rungs: the levels a grasp-search pass pins every plan to (None: the whole ladder); search: the grasp search in
    # progress (choose_grasp_pose); kind / entry: the latest plan's failure kind (None: it planned) and last failed try
    state = {"rungs": None, "batches": 0, "search": None, "kind": None, "entry": None}
    names = {"left_plan_path": "left", "right_plan_path": "right",
             "left_plan_multi_path": "left", "right_plan_multi_path": "right"}
    orig = {}
    for name, arm in names.items():
        fn = getattr(robot, name)
        orig[name] = fn

        def wrapped(*a, _fn=fn, _arm=arm, _name=name, **k):
            batch = _name.endswith("multi_path")
            planner = robot.left_planner if _arm == "left" else robot.right_planner
            mg = planner.motion_gen_batch if batch else planner.motion_gen
            rungs = list(range(len(levels_all))) if state["rungs"] is None else list(state["rungs"])
            target = a[0] if a else k.get("target_lst" if batch else "target_pose")
            goals = list(target) if batch and target is not None else [target]
            base = {"call": _name, "skill": getattr(task, "_safe_plan_ctx", None) or current_skill(task),
                    "goal": _pose_list(goals[0]) if goals else None,
                    "constrained": k.get("constraint_pose") is not None}
            if batch:
                base["n_goals"] = len(goals)
                state["batches"] += 1
            res, kind, entry = None, None, None
            for j, i in enumerate(rungs):
                m, model = levels_all[i]
                if j > 0 and not escalate(kind, m, model, floors):
                    break
                if world is not None:
                    world.push(_arm, m)
                s = sync(env, _arm, min(float(payload_margin), m), model)
                if log is not None:
                    log.append({"call": _name, **s})
                n0 = len(status_sink) if status_sink is not None else 0
                obstacles = world.names(_arm) if world is not None else ()
                entity = robot.left_entity if _arm == "left" else robot.right_entity
                qpos = k.get("last_qpos") if k.get("last_qpos") is not None else entity.get_qpos()
                start = None
                if batch and i < len(levels_all) - 1:      # cuRobo's batch planner checks no start state
                    start = check_start(planner, mg, qpos, obstacles)
                entry = {**base, "level": i, "margin": m, "model": model}
                if start is not None and start["ok"] is False:
                    res = {"status": np.array(["Failure"] * max(len(goals), 1), dtype=object)}
                    statuses = [f"check_start_state:{start['status']}"]
                else:
                    was_on = bool(getattr(_self_collision_cost(mg), "enabled", False))
                    try:
                        res = _fn(*a, **k)
                    except Exception as exc:
                        _note(info, {**entry, "ok": False, "kind": "planner_error",
                                     "exc": f"{type(exc).__name__}: {str(exc)[:160]}"})
                        raise
                    finally:
                        _restore_self_collision(mg, was_on)
                    statuses = list(status_sink[n0:]) if status_sink is not None else []
                if not _failed(res):
                    if i > 0:
                        _note(info, {**entry, "ok": True})
                    if isinstance(res, dict):
                        res["_safe_level"] = i
                    state["kind"], state["entry"] = None, None
                    return res
                kind = failure_kind(statuses)
                entry.update({"ok": False, "kind": kind, "status": statuses[-1] if statuses else None,
                              "held": s.get("held", ""), "n_payload": s.get("n_payload", 0)})
                if start is not None:
                    entry["start"] = start["status"]
                    entry["culprits"] = start["culprits"]
                elif kind == "start_in_collision" and obstacles:
                    entry["culprits"] = check_start(planner, mg, qpos, obstacles)["culprits"]
                _note(info, entry)
            if (table_free_descents and kind == "ik_fail" and base["constrained"] and not batch
                    and state["search"] is None and world is not None and getattr(world, "table", None) is not None):
                # RoboTwin's straight grasp / place descent onto a flat object asks for the fingertips below the
                # tabletop (2026-10-28 pilot: the stock grasps of 006_hamburg and 075_bread put them 2.5-3 cm into
                # it, 1 cm below the planner's table; physics stops them on the table), which no margin of the
                # table-aware world allows. A grasp descent (nothing held) first goes to the goal raised until the
                # fingertips stop descent_tip_clear above the real tabletop, the whole world kept (10-29: pressed on
                # the table, the closing fingers tipped 005_french-fries 30-68 deg in 7 of 9 lifts); then, or for a
                # place descent, once more without the table, the items kept at the ladder's margins
                tb = (getattr(world, "stats", None) or {}).get("table") or {}
                up = None
                if descent_tip_len and not (entry or {}).get("held") and "top_z" in tb:
                    up = raised_goal(goals[0], float(tb["top_z"]) + float(tb.get("clearance", 0.0)),
                                     float(descent_tip_len), float(descent_tip_clear))
                if up is not None:
                    ua, uk = ((up, *a[1:]), k) if a else ((), {**k, "target_pose": up})
                    for m, model in levels_all:
                        if model != "full" or float(m) < float(floors.get("ik_fail", 0.0)) - 1e-9:
                            continue
                        world.push(_arm, m)
                        s = sync(env, _arm, min(float(payload_margin), m), "full")
                        n0 = len(status_sink) if status_sink is not None else 0
                        entry = {**base, "goal": _pose_list(up), "level": "raised", "margin": m, "model": "full",
                                 "raise_m": round(float(up[2]) - float(_pose_list(goals[0])[2]), 4)}
                        was_on = bool(getattr(_self_collision_cost(mg), "enabled", False))
                        try:
                            res = _fn(*ua, **uk)
                        except Exception as exc:
                            _note(info, {**entry, "ok": False, "kind": "planner_error",
                                         "exc": f"{type(exc).__name__}: {str(exc)[:160]}"})
                            raise
                        finally:
                            _restore_self_collision(mg, was_on)
                        if not _failed(res):
                            _note(info, {**entry, "ok": True})
                            if isinstance(res, dict):
                                res["_safe_level"] = "raised"
                            state["kind"], state["entry"] = None, None
                            return res
                        statuses = list(status_sink[n0:]) if status_sink is not None else []
                        entry.update({"ok": False, "kind": failure_kind(statuses),
                                      "status": statuses[-1] if statuses else None, "held": s.get("held", "")})
                        _note(info, entry)
                for m, model in levels_all:
                    if model != "full" or float(m) < float(floors.get("ik_fail", 0.0)) - 1e-9:
                        continue
                    world.push(_arm, m, table=False)
                    s = sync(env, _arm, min(float(payload_margin), m), "full")
                    n0 = len(status_sink) if status_sink is not None else 0
                    entry = {**base, "level": "no_table", "margin": m, "model": "full", "table": False}
                    was_on = bool(getattr(_self_collision_cost(mg), "enabled", False))
                    try:
                        res = _fn(*a, **k)
                    except Exception as exc:
                        _note(info, {**entry, "ok": False, "kind": "planner_error",
                                     "exc": f"{type(exc).__name__}: {str(exc)[:160]}"})
                        raise
                    finally:
                        _restore_self_collision(mg, was_on)
                    if not _failed(res):
                        _note(info, {**entry, "ok": True})
                        if isinstance(res, dict):
                            res["_safe_level"] = "no_table"
                        state["kind"], state["entry"] = None, None
                        return res
                    statuses = list(status_sink[n0:]) if status_sink is not None else []
                    entry.update({"ok": False, "kind": failure_kind(statuses),
                                  "status": statuses[-1] if statuses else None, "held": s.get("held", "")})
                    _note(info, entry)
            kind = kind or "plan_fail"
            if isinstance(res, dict):
                res["_safe_fail"], res["_safe_fail_entry"] = kind, entry
            state["kind"], state["entry"] = kind, entry
            return res

        setattr(robot, name, wrapped)

    def rotate_attr(arm_tag) -> str:
        return "left_rotate_lim" if arm_tag == "left" else "right_rotate_lim"   # robot.create_target_pose_list

    def tilts(arm_tag) -> bool:
        """Whether ``wide_grasp`` gives this arm new grasp candidates: set, and the arm has no rotate range."""
        stock = list(getattr(robot, rotate_attr(arm_tag), None) or [0.0, 0.0])
        return wide_grasp is not None and abs(float(stock[1]) - float(stock[0])) <= 1e-9

    def eligible(last, r: int, wide: bool) -> bool:
        """Whether a contact point whose latest pass failed as ``last`` = (kind, wide) is planned at level ``r``: always
        with new candidates (the first pass with the tilts), else as escalate allows its failure."""
        kind, last_wide = last
        return last_wide != wide or escalate(kind, *levels_all[r], floors)

    guards = {}
    if callable(getattr(task, "choose_grasp_pose", None)):
        orig_choose = task.choose_grasp_pose

        def choose_grasp_pose(actor, *ga, **gk):
            arm_tag = gk.get("arm_tag", ga[0] if ga else None)
            ctx = f"{arm_tag} grasp_search {_actor_label(task, actor)}"
            prev, n0 = (getattr(task, "_safe_plan_ctx", None), state["search"]), state["batches"]
            # search: wide = this pass's tilts on; last = per contact point (kind, wide) of its latest failed pass;
            # errors = planner errors (cuRobo raised: no further pass)
            search = {"wide": False, "last": {}, "errors": 0}
            wide_ok = tilts(arm_tag)
            passes = [(0, False)] + [(r, wide_ok) for r in range(0 if wide_ok else 1, len(levels_all))]
            out, found, n_pass = None, None, 0
            task._safe_plan_ctx, state["search"] = ctx, search
            try:
                for p, (r, wide) in enumerate(passes):
                    if p > 0 and (search["errors"] or not any(eligible(v, r, wide) for v in search["last"].values())):
                        break
                    if p == 1:
                        info["safe_wide_grasp"] = int(info.get("safe_wide_grasp", 0)) + 1
                    search["wide"], state["rungs"], n_pass = wide, [r], p + 1
                    out = orig_choose(actor, *ga, **gk)
                    if out is None:                     # plan_success already False: RoboTwin's bare return
                        return out
                    if out[0] is not None and out[1] is not None:
                        found = (p, r, wide)
                        break
            finally:
                state["rungs"] = None
                task._safe_plan_ctx, state["search"] = prev
            n_batches = state["batches"] - n0
            if found is not None:
                p, r, wide = found
                _count(info, r)
                info.pop("safe_last_fail", None)
                if p > 0:
                    _note(info, {"call": "choose_grasp_pose", "skill": ctx, "ok": True, "level": r, "wide": wide,
                                 "n_passes": n_pass, "n_batches": n_batches})
                return out
            if getattr(task, "plan_success", True):     # no contact point gave a pose: end the motion, no fallback
                kind = "planner_error" if search["errors"] else "no_grasp"
                cp = gk.get("contact_point_id")
                entry = {"call": "choose_grasp_pose", "skill": ctx, "ok": False, "kind": kind, "n_passes": n_pass,
                         "n_batches": n_batches, "kinds": [v[0] for v in search["last"].values()],
                         "last_try": state["entry"],
                         "contact_point_id": cp if cp is None or isinstance(cp, (int, list)) else str(cp)}
                _note(info, entry)
                _fail_final(info, kind, entry)
                info["safe_fail"] = info.get("safe_fail") or kind
                task.plan_success = False
            dummy = [-1.0] * 7          # RoboTwin's own placeholder pose once plan_success is False (get_grasp_pose)
            return dummy, list(dummy)

        guards["choose_grasp_pose"] = choose_grasp_pose
    if callable(getattr(task, "choose_best_pose", None)):
        orig_best = task.choose_best_pose

        def choose_best_pose(res_pose, center_pose, arm_tag=None):
            search = state["search"]
            if search is None:                      # outside a grasp search: RoboTwin's, the whole ladder per plan
                return orig_best(res_pose, center_pose, arm_tag)
            key = (_pose_key(res_pose), _pose_key(center_pose))     # the contact point (its stock grasp pose)
            last = search["last"].get(key)
            if last is not None and not eligible(last, state["rungs"][0], search["wide"]):
                return None                         # its failure is not retried at this level: RoboTwin skips it
            attr = rotate_attr(arm_tag)
            stock = getattr(robot, attr, None)
            if search["wide"]:
                setattr(robot, attr, [float(wide_grasp[0]), float(wide_grasp[1])])   # ten tilts over [lo, hi)
            state["kind"], state["entry"] = None, None
            try:
                pose = orig_best(res_pose, center_pose, arm_tag)
            finally:
                if search["wide"]:
                    setattr(robot, attr, stock)
            if pose is None:
                kind = state["kind"] or "plan_fail"
                search["last"][key] = (kind, search["wide"])
                search["errors"] += int(kind == "planner_error")
            return pose

        guards["choose_best_pose"] = choose_best_pose
    shadowed = {name: vars(task)[name] for name in guards if name in vars(task)}   # instance attributes before ours
    for name, fn in guards.items():
        setattr(task, name, fn)

    def uninstall():
        for name in orig:
            if name in vars(robot):
                delattr(robot, name)
        for name, fn in guards.items():
            if vars(task).get(name) is fn:
                if name in shadowed:
                    setattr(task, name, shadowed[name])
                else:
                    delattr(task, name)             # back to the task class's method
        detach_all(robot)

    return uninstall


def finish_episode(task) -> None:
    """End of a safe episode: a plan failure (plan_success False) without a reason yet gets one in
    episode_info["safe_fail"]: the kind of the latest plan the task used, if it failed (``safe_last_fail``), else
    "plan_fail" (plan_success tripped outside a plan, e.g. a target pose None)."""
    info = getattr(task, "episode_info", None)
    if isinstance(info, dict) and getattr(task, "plan_success", True) is False and not info.get("safe_fail"):
        info["safe_fail"] = info.get("safe_last_fail") or "plan_fail"


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
