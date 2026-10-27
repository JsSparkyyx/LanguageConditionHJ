"""T1 safe nominal mode, planning side (safe_plan.py, 2026-10-27), off-GPU: the protected world (grown hulls, the
table, live refresh, cuRobo slot bookkeeping), the retry ladder (status kinds, margins, dropped models, the batch
start check, "Invalid Problem"), the grasp guards, the diagnostics, and the yaml knobs.

cuRobo's MotionGen / world collision checker are stood in by fakes that keep cuRobo's slot rules (world.py /
world_mesh.py: a world without cuboids or meshes leaves the old ones enabled; more meshes than slots rebuild the
cache); sapien entities with PhysX collision shapes are real (no scene, no GPU)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sapien = pytest.importorskip("sapien")
pytest.importorskip("mplib")
pytest.importorskip("trimesh")
pytest.importorskip("curobo")

import sapien.physx as px  # noqa: E402
import transforms3d as t3d  # noqa: E402

from main.item_envs import safe_plan  # noqa: E402
from main.item_envs.items import ItemSpec  # noqa: E402
from main.item_envs.retime import retime  # noqa: E402
from main.item_envs.risk import sample_risk  # noqa: E402
from main.item_envs.runner import apply_nominal_mode  # noqa: E402

TASKS = Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs" / "tasks"
_MAT = None


def _mat():
    global _MAT
    if _MAT is None:
        _MAT = px.PhysxMaterial(0.5, 0.5, 0.0)
    return _MAT


def _entity(shapes, pose):
    e = sapien.Entity()
    comp = px.PhysxRigidStaticComponent()
    for s in shapes:
        comp.attach(s)
    e.add_component(comp)
    e.set_pose(pose)
    return e


def _cylinder_spec(item_id="obj_0", xyz=(0.1, 0.0, 0.80), r=0.03, half_len=0.06):
    cyl = px.PhysxCollisionShapeCylinder(r, half_len, _mat())
    cyl.set_local_pose(sapien.Pose(q=t3d.quaternions.axangle2quat([0, 1, 0], -np.pi / 2)))   # PhysX axis x -> z
    return ItemSpec(item_id=item_id, entity=_entity([cyl], sapien.Pose(list(xyz))), category="cylinder",
                    half_extents=[r, r, half_len])


def _table_task():
    top = px.PhysxCollisionShapeBox([0.6, 0.35, 0.025], _mat())
    top.set_local_pose(sapien.Pose([0, 0, -0.025]))
    leg = px.PhysxCollisionShapeBox([0.025, 0.025, 0.37], _mat())
    leg.set_local_pose(sapien.Pose([0.55, 0.3, -0.37]))
    return SimpleNamespace(table=_entity([top, leg], sapien.Pose([0.0, 0.0, 0.74])))


# ------------------------------------------------------------ cuRobo fakes ----
class _WCC:
    """WorldMeshCollision's slot bookkeeping as MotionGen.update_world fills it."""

    def __init__(self, n_obb=safe_plan.OBB_SLOTS, n_mesh=safe_plan.MESH_SLOTS):
        self._cube_tensor_list = [np.zeros((1, n_obb, 4)), np.zeros((1, n_obb, 8)), np.zeros((1, n_obb), np.uint8)]
        self._env_n_obbs = np.zeros(1, np.int32)
        self._env_obbs_names = [[None] * n_obb]
        self._mesh_tensor_list = [np.zeros((1, n_mesh), np.int64), np.zeros((1, n_mesh, 8)), np.zeros((1, n_mesh), np.uint8)]
        self._env_n_mesh = np.zeros(1, np.int32)
        self._env_mesh_names = [[None] * n_mesh]
        self._wp_mesh_cache = {}
        self.pose, self.dims = {}, {}

    def load(self, cfg):
        meshes, boxes = list(cfg.mesh or []), list(cfg.cuboid or [])
        if meshes:
            if len(meshes) > self._mesh_tensor_list[0].shape[1]:     # cuRobo: "Creating new Mesh cache"
                n = len(meshes)
                self._mesh_tensor_list = [np.zeros((1, n), np.int64), np.zeros((1, n, 8)), np.zeros((1, n), np.uint8)]
                self._env_mesh_names = [[None] * n]
            for i, m in enumerate(meshes):
                self._wp_mesh_cache.setdefault(m.name, np.asarray(m.vertices))
                self._env_mesh_names[0][i] = m.name
                self.pose[m.name] = list(m.pose)
            self._mesh_tensor_list[2][0, : len(meshes)] = 1
            self._mesh_tensor_list[2][0, len(meshes):] = 0
            self._env_n_mesh[0] = len(meshes)
        if boxes:                                                    # no cuboids: cuRobo returns early
            for i, b in enumerate(boxes):
                self._env_obbs_names[0][i] = b.name
                self.pose[b.name], self.dims[b.name] = list(b.pose), list(b.dims)
            self._cube_tensor_list[2][0, : len(boxes)] = 1
            self._cube_tensor_list[2][0, len(boxes):] = 0
            self._env_n_obbs[0] = len(boxes)

    def enabled(self) -> set:
        out = set()
        for names, n, flags in ((self._env_obbs_names[0], self._env_n_obbs[0], self._cube_tensor_list[2][0]),
                                (self._env_mesh_names[0], self._env_n_mesh[0], self._mesh_tensor_list[2][0])):
            out |= {names[i] for i in range(int(n)) if flags[i] == 1}
        return out

    def enable_obstacle(self, name, enable=True, env_idx=0):
        if name in self._env_mesh_names[0]:
            self._mesh_tensor_list[2][0, self._env_mesh_names[0].index(name)] = int(enable)
        elif name in self._env_obbs_names[0]:
            self._cube_tensor_list[2][0, self._env_obbs_names[0].index(name)] = int(enable)
        else:
            raise ValueError(f"Obstacle not found in world model: {name}")


class _Cost:
    def __init__(self):
        self._on = True

    @property
    def enabled(self):
        return self._on

    def enable_cost(self):
        self._on = True


class _MG:
    """MotionGen: update_world, check_start_state (a start is in collision while an obstacle of ``blockers`` is
    enabled; like cuRobo 0.7.8 it then leaves the self-collision cost off), plan_single / plan_batch."""

    def __init__(self):
        self.world_coll_checker = _WCC()
        self.tensor_args = SimpleNamespace(to_device=lambda t: t.float())
        self.rollout_fn = SimpleNamespace(robot_self_collision_constraint=_Cost())
        self.n_updates = 0
        self.blockers: set = set()
        self.n_start_checks = 0

    def update_world(self, cfg):
        self.n_updates += 1
        self.world_coll_checker.load(cfg)

    def check_start_state(self, js):
        self.n_start_checks += 1
        assert tuple(js.position.shape) == (1, 6)
        ok = not (self.blockers & self.world_coll_checker.enabled())
        if not ok:
            self.rollout_fn.robot_self_collision_constraint._on = False
        return ok, None if ok else "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"

    def plan_single(self, *a, **k):
        return SimpleNamespace(success=np.array([False]), status="MotionGenStatus.IK_FAIL")

    def plan_batch(self, *a, **k):
        raise RuntimeError("boom")

    def detach_spheres_from_robot(self, link_name):
        self.detached = link_name


def _trans_from_world_to_base(base_pose, target_pose):   # RoboTwin CuroboPlanner._trans_from_world_to_base
    base_p, base_q = base_pose[0:3], base_pose[3:]
    target_p, target_q = target_pose[0:3], target_pose[3:]
    wRb = t3d.quaternions.quat2mat(base_q)
    wRt = t3d.quaternions.quat2mat(target_q)
    return wRb.T @ (target_p - base_p), t3d.quaternions.mat2quat(wRb.T @ wRt)


class _Planner:
    def __init__(self, x):
        self.robot_origion_pose = sapien.Pose([x, -0.65, 0.75])
        self.frame_bias = [0.01, -0.02, 0.03]
        self.yml_path = "/RoboTwin/assets/embodiments/ur5-wsg/curobo.yml"
        self.active_joints_name = [f"j{i}" for i in range(6)]
        self.all_joints = self.active_joints_name + ["g0", "g1"]
        self.motion_gen, self.motion_gen_batch = _MG(), _MG()
        self._trans_from_world_to_base = _trans_from_world_to_base


class _Arm:
    def get_qpos(self):
        return np.linspace(-1.0, 1.0, 8)


class _Robot:
    """RoboTwin Robot's four plan calls, scripted: ``single`` / ``batch`` hold one entry per planner call, None =
    success, else the cuRobo status PlanStatusLog would record (a batch entry may also be a per-goal callable)."""

    def __init__(self, sink):
        self.left_planner, self.right_planner = _Planner(-0.4), _Planner(0.4)
        self.left_entity, self.right_entity = _Arm(), _Arm()
        self.left_rotate_lim, self.right_rotate_lim = [0, 0], [0, 0]
        self.sink, self.single, self.batch, self.calls = sink, [], [], []

    def _single(self, arm, target):
        self.calls.append(("single", arm))
        st = self.single.pop(0)
        if st is None:
            return {"status": "Success", "position": np.zeros((5, 6)), "velocity": np.zeros((5, 6))}
        self.sink.append(f"plan_single:{st}")
        return {"status": "Fail"}

    def _batch(self, arm, targets):
        self.calls.append(("batch", arm))
        st = self.batch.pop(0)
        if callable(st):
            ok = [bool(st(t)) for t in targets]
            st = None if any(ok) else "MotionGenStatus.IK_FAIL"
        else:
            ok = [st is None] * len(targets)
        if st is not None:
            self.sink.append(f"plan_batch:{st}")
        return {"status": np.array(["Success" if o else "Failure" for o in ok], dtype=object)}

    def left_plan_path(self, target_pose, constraint_pose=None, use_point_cloud=False, use_attach=False, last_qpos=None):
        return self._single("left", target_pose)

    def right_plan_path(self, target_pose, constraint_pose=None, use_point_cloud=False, use_attach=False, last_qpos=None):
        return self._single("right", target_pose)

    def left_plan_multi_path(self, target_lst, constraint_pose=None, use_point_cloud=False, use_attach=False,
                             last_qpos=None):
        return self._batch("left", target_lst)

    def right_plan_multi_path(self, target_lst, constraint_pose=None, use_point_cloud=False, use_attach=False,
                              last_qpos=None):
        return self._batch("right", target_lst)


class _World:
    def __init__(self):
        self.pushes = []

    def push(self, arm=None, margin=None):
        self.pushes.append((arm, margin))
        return True

    def names(self, arm):
        return []


def _sync_log():
    calls = []

    def sync(env, arm, payload_margin, level):
        calls.append((arm, round(payload_margin, 6), level))
        return {"n_hand": 74 if level != "none" else 0, "n_payload": 12 if level == "full" else 0,
                "held": "container" if level == "full" else "", "level": level}

    return calls, sync


def _env(sink=None):
    sink = [] if sink is None else sink
    robot = _Robot(sink)
    task = SimpleNamespace(episode_info={}, plan_success=True, _cehj_skill="left pre_place container")
    return SimpleNamespace(robot=robot, task=task), sink


@pytest.fixture(autouse=True)
def _no_curobo_check(monkeypatch):
    monkeypatch.setattr(safe_plan, "require_curobo", lambda robot: None)


# ---------------------------------------------------------------- the world ----
def test_grown_hull_contains_the_margin_and_beats_the_box():
    spec = _cylinder_spec()
    pts = safe_plan.hull_points(spec)
    v, f = safe_plan.grown_hull(pts, 0.01)
    from main.item_envs.item_shapes import _hull

    _, _, planes = _hull(v)
    inside = lambda p: float((np.asarray(p) @ planes[:, :3].T - planes[:, 3]).max()) <= 1e-9   # noqa: E731
    for a in np.linspace(0, 2 * np.pi, 13):   # every point 1 cm (minus a hair) off the cylinder's side is covered
        assert inside([0.0399 * np.cos(a), 0.0399 * np.sin(a), 0.0])
    assert inside([0.0, 0.0, 0.0699]) and not inside([0.0, 0.0, 0.0705])
    assert not inside([0.039, 0.039, 0.0])   # inside the grown OBB (corner 4 x 4 cm): the hull is 1.6 cm tighter there
    assert np.linalg.norm(v[:, :2], axis=1).max() < 0.0405
    assert len(f) and len(v) == len(np.unique(np.round(v, 9), axis=0))


def test_world_hull_mesh_obb_fallback_and_table():
    spec = _cylinder_spec()
    robot = SimpleNamespace(left_planner=_Planner(-0.4), right_planner=_Planner(0.4))
    w = safe_plan.ProtectedWorld(robot, [spec], task=_table_task(), margin=0.01)
    obs = {o["kind"]: o for o in w.obstacles()}
    assert obs["mesh"]["name"].startswith("protected_obj_0_")
    np.testing.assert_allclose(obs["mesh"]["p"], [0.1, 0.0, 0.80])
    table = next(o for o in w.obstacles() if o["name"] == safe_plan.TABLE_NAME)
    assert table["p"][2] + table["half"][2] == pytest.approx(0.74 - 0.02)            # top 2 cm low
    np.testing.assert_allclose(table["half"], [0.6, 0.35, 0.025], atol=1e-6)           # the real top, thickness kept
    assert w.stats["items"]["obj_0"]["shape"] == "hull" and w.stats["table"]["clearance"] == 0.02
    # the 10-22 world: boxes grown by the margin, centred on the OBB
    box = safe_plan.ProtectedWorld(robot, [spec], margin=0.01, shape="obb", table=False).obstacles()
    assert [o["kind"] for o in box] == ["cuboid"]
    np.testing.assert_allclose(box[0]["half"], [0.04, 0.04, 0.07])
    # no table entity: recorded, not a crash; table=False: none at all
    w2 = safe_plan.ProtectedWorld(robot, [spec], task=SimpleNamespace(), margin=0.01)
    assert w2.table is None and "table_missing" in w2.stats
    assert all(o["name"] != safe_plan.TABLE_NAME for o in safe_plan.ProtectedWorld(robot, [spec], table=False).obstacles())


def test_world_hull_failure_falls_back_to_the_box(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("qhull")

    monkeypatch.setattr(safe_plan, "grown_hull", boom)
    robot = SimpleNamespace(left_planner=_Planner(-0.4), right_planner=_Planner(0.4))
    w = safe_plan.ProtectedWorld(robot, [_cylinder_spec()], margin=0.01, table=False)
    assert [o["kind"] for o in w.obstacles()] == ["cuboid"]
    assert w.stats["items"]["obj_0"]["shape"] == "obb" and "qhull" in w.stats["items"]["obj_0"]["hull_error"]


def test_world_mesh_slots_are_capped():
    specs = [_cylinder_spec(f"obj_{i}", xyz=(0.05 * i, 0.0, 0.8)) for i in range(safe_plan.MESH_SLOTS + 1)]
    robot = SimpleNamespace(left_planner=_Planner(-0.4), right_planner=_Planner(0.4))
    kinds = [o["kind"] for o in safe_plan.ProtectedWorld(robot, specs, table=False).obstacles()]
    assert kinds == ["mesh"] * safe_plan.MESH_SLOTS + ["cuboid"]


def test_world_push_base_frame_refresh_and_clear():
    spec = _cylinder_spec()
    robot = SimpleNamespace(left_planner=_Planner(-0.4), right_planner=_Planner(0.4))
    w = safe_plan.ProtectedWorld(robot, [spec], task=_table_task(), margin=0.01)
    assert w.push() is True
    for planner in (robot.left_planner, robot.right_planner):
        for mg in (planner.motion_gen, planner.motion_gen_batch):
            assert mg.n_updates == 1
            wcc = mg.world_coll_checker
            assert wcc.enabled() == set(w.names("left"))
            mesh = next(n for n in wcc.enabled() if n.startswith("protected_"))
            o = np.asarray(planner.robot_origion_pose.p)
            np.testing.assert_allclose(wcc.pose[mesh][:3], np.array([0.1, 0.0, 0.8]) - o + planner.frame_bias, atol=1e-6)
            np.testing.assert_allclose(wcc.dims[safe_plan.TABLE_NAME], [1.2, 0.7, 0.05], atol=1e-6)
    assert w.push() is False                                   # nothing moved: no write
    spec.entity.set_pose(sapien.Pose([0.102, 0.0, 0.80]))      # nudged 2 mm
    assert w.push("left") is True
    assert robot.left_planner.motion_gen.n_updates == 2 and robot.right_planner.motion_gen.n_updates == 1
    assert w.push("left", 0.005) is True                       # another margin: another mesh
    assert w.stats["n_writes"] == 4
    wcc = robot.left_planner.motion_gen.world_coll_checker
    assert len(wcc._wp_mesh_cache) == 2
    w.clear()
    for planner in (robot.left_planner, robot.right_planner):
        wcc = planner.motion_gen_batch.world_coll_checker
        assert int(wcc._env_n_mesh[0]) == 0 and not wcc._mesh_tensor_list[2].any() and not wcc._wp_mesh_cache


def test_load_world_switches_stale_slots_off_and_guards_the_mesh_cache():
    mg = _MG()
    safe_plan.load_world(mg, {"cuboid": {"table": {"dims": [0.7, 2, 0.04], "pose": [0, 0, 0, 1, 0, 0, 0]}}})
    mesh = {"pose": [0, 0, 0, 1, 0, 0, 0], "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]}
    safe_plan.load_world(mg, {"cuboid": {}, "mesh": {"m0": mesh}})
    assert mg.world_coll_checker.enabled() == {"m0"}           # RoboTwin's (misplaced) table no longer on
    safe_plan.load_world(mg, {"cuboid": {"b0": {"dims": [0.1, 0.1, 0.1], "pose": [0, 0, 0, 1, 0, 0, 0]}}})
    assert mg.world_coll_checker.enabled() == {"b0"}           # the old mesh no longer on
    with pytest.raises(RuntimeError, match="slots"):
        safe_plan.load_world(mg, {"mesh": {f"m{i}": mesh for i in range(safe_plan.MESH_SLOTS + 1)}})
    mg.world_coll_checker = _WCC(n_mesh=1)                     # a cache too small for two meshes: cuRobo rebuilds it
    with pytest.raises(RuntimeError, match="CUDA graphs"):
        safe_plan.load_world(mg, {"mesh": {"m0": mesh, "m1": mesh}})


# --------------------------------------------------------------- the ladder ----
def test_failure_kinds_of_the_10_22_statuses():
    k = safe_plan.failure_kind
    assert k(["plan_batch:Invalid Problem"]) == "start_in_collision"          # the 10-22 bug: was never retried
    assert k(["plan_single:MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"]) == "start_in_collision"
    assert k(["check_start_state:MotionGenStatus.INVALID_START_STATE_SELF_COLLISION"]) == "start_in_collision"
    assert k(["plan_single:MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS"]) == "joint_limits"
    assert k(["plan_batch:MotionGenStatus.IK_FAIL"]) == "ik_fail"
    assert k(["plan_single:MotionGenStatus.TRAJOPT_FAIL"]) == k(["plan_batch:MotionGenStatus.GRAPH_FAIL"]) == "no_route"
    assert k(["plan_single:MotionGenStatus.INVALID_PARTIAL_POSE_COST_METRIC"]) == "constraint"
    assert k(["plan_batch:Invalid Hold Partial Pose"]) == "constraint"
    assert k(["plan_single:MotionGenStatus.INVALID_QUERY"]) == "invalid_query"
    assert k(["plan_batch:EXC:RuntimeError: boom"]) == "planner_error"
    assert k([]) == "plan_fail"
    assert safe_plan.plan_levels() == [(0.01, "full"), (0.005, "full"), (0.002, "full"), (0.002, "hand"), (0.002, "none")]
    assert safe_plan.plan_levels((0.01,)) == [(0.01, "full"), (0.01, "hand"), (0.01, "none")]   # the 10-22 ladder
    assert safe_plan.escalate("ik_fail", "full") and not safe_plan.escalate("ik_fail", "hand")
    assert safe_plan.escalate("start_in_collision", "none") and not safe_plan.escalate("constraint", "full")


def test_ladder_shrinks_the_margin_and_records_the_level():
    env, sink = _env()
    env.robot.single = ["MotionGenStatus.IK_FAIL", "MotionGenStatus.IK_FAIL", None]
    world, (calls, sync) = _World(), _sync_log()
    safe_plan.install_plan_hooks(env, status_sink=sink, world=world, margins=safe_plan.MARGINS, sync=sync)
    res = env.robot.left_plan_path([0.1, -0.1, 0.9, 1, 0, 0, 0])
    assert res["status"] == "Success"
    assert world.pushes == [("left", 0.01), ("left", 0.005), ("left", 0.002)]
    assert calls == [("left", 0.01, "full"), ("left", 0.005, "full"), ("left", 0.002, "full")]   # payload shrinks too
    info = env.task.episode_info
    assert info["safe_plan_levels"] == {"2": 1} and "safe_last_fail" not in info
    log = info["safe_plan_log"]
    assert [(e["level"], e["ok"], e.get("kind")) for e in log] == [(0, False, "ik_fail"), (1, False, "ik_fail"), (2, True, None)]
    assert log[0]["skill"] == "left pre_place container" and log[0]["goal"] == [0.1, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0]
    assert log[0]["status"] == "plan_single:MotionGenStatus.IK_FAIL" and log[0]["held"] == "container"


def test_ladder_drops_the_models_only_for_a_start_in_collision():
    env, sink = _env()
    start = "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"
    env.robot.single = [start, start, start, None]
    calls, sync = _sync_log()
    safe_plan.install_plan_hooks(env, status_sink=sink, world=_World(), margins=safe_plan.MARGINS, sync=sync)
    assert env.robot.left_plan_path([0, 0, 1, 1, 0, 0, 0])["status"] == "Success"
    assert [c[2] for c in calls] == ["full", "full", "full", "hand"]
    assert env.task.episode_info["safe_plan_levels"] == {"3": 1}


def test_no_unsafe_fallback_the_plan_fails_with_its_reason():
    env, sink = _env()
    env.robot.single = ["MotionGenStatus.IK_FAIL"] * 3
    safe_plan.install_plan_hooks(env, status_sink=sink, world=_World(), margins=safe_plan.MARGINS, sync=_sync_log()[1])
    assert env.robot.right_plan_path([0, 0, 1, 1, 0, 0, 0])["status"] == "Fail"
    assert env.robot.calls == [("single", "right")] * 3                  # the margins only: never without the hand
    info = env.task.episode_info
    assert info["safe_plan_levels"] == {"fail": 1} and info["safe_last_fail"] == "ik_fail"
    safe_plan.finish_episode(env.task)
    assert "safe_fail" not in info                                       # plan_success still True: not final
    env.task.plan_success = False                                        # RoboTwin's move_to_pose trips it
    safe_plan.finish_episode(env.task)
    assert info["safe_fail"] == "ik_fail"


def test_invalid_problem_batch_is_retried_and_the_start_checked_first():
    env, sink = _env()
    robot = env.robot
    spec = _cylinder_spec()
    world = safe_plan.ProtectedWorld(robot, [spec], task=_table_task(), margin=0.01)
    world.push()
    mg = robot.left_planner.motion_gen_batch
    name0 = next(n for n in world.names("left") if n.startswith("protected_obj_0_"))
    mg.blockers = {name0}                       # the 1 cm hull holds the start: the 0.5 cm one does not
    robot.batch = [None]
    safe_plan.install_plan_hooks(env, status_sink=sink, world=world, margins=safe_plan.MARGINS, sync=_sync_log()[1])
    res = robot.left_plan_multi_path([[0.1, 0.0, 0.9, 1, 0, 0, 0]] * 10)
    assert list(res["status"]) == ["Success"] * 10
    assert robot.calls == [("batch", "left")]                        # level 0 never called the planner
    log = env.task.episode_info["safe_plan_log"]
    assert log[0]["kind"] == "start_in_collision" and log[0]["culprits"] == [name0] and log[0]["n_goals"] == 10
    assert "INVALID_START_STATE_WORLD_COLLISION" in log[0]["start"]
    assert env.task.episode_info["safe_plan_levels"] == {"1": 1}
    assert mg.rollout_fn.robot_self_collision_constraint.enabled     # cuRobo's leaked switch put back
    # the 10-22 failure itself: no pre-check verdict, the batch says "Invalid Problem" -> retried, not dropped
    env2, sink2 = _env()
    env2.robot.batch = ["Invalid Problem", None]
    safe_plan.install_plan_hooks(env2, status_sink=sink2, world=_World(), margins=safe_plan.MARGINS, sync=_sync_log()[1])
    assert list(env2.robot.right_plan_multi_path([[0, 0, 1, 1, 0, 0, 0]] * 10)["status"]) == ["Success"] * 10
    assert env2.task.episode_info["safe_plan_log"][0]["kind"] == "start_in_collision"


def test_swallowed_planner_exceptions_are_logged(monkeypatch):
    monkeypatch.setattr(safe_plan.PlanStatusLog, "sink", [])
    planner = _Planner(0.4)
    robot = SimpleNamespace(left_planner=_Planner(-0.4), right_planner=planner)
    safe_plan.PlanStatusLog.install(robot)
    with pytest.raises(RuntimeError, match="boom"):
        planner.motion_gen_batch.plan_batch()
    planner.motion_gen.plan_single()
    assert safe_plan.PlanStatusLog.sink == ["plan_batch:EXC:RuntimeError: boom", "plan_single:MotionGenStatus.IK_FAIL"]
    # through the hook: RoboTwin's plan_batch returned its blank failure, the kind says why, nothing is retried
    env, sink = _env()
    env.robot.batch = ["EXC:RuntimeError: boom"]
    safe_plan.install_plan_hooks(env, status_sink=sink, world=_World(), margins=safe_plan.MARGINS, sync=_sync_log()[1])
    env.robot.left_plan_multi_path([[0, 0, 1, 1, 0, 0, 0]] * 10)
    assert env.robot.calls == [("batch", "left")]
    assert env.task.episode_info["safe_plan_log"][0]["kind"] == "planner_error"


def test_grasp_guard_and_the_wide_search():
    env, sink = _env()
    robot, task = env.robot, env.task
    tilts = []

    def choose_best_pose(res_pose, center_pose, arm_tag=None):   # RoboTwin: ten tilts over rotate_lim, first success
        lo, hi = robot.left_rotate_lim if arm_tag == "left" else robot.right_rotate_lim
        cands = [list(res_pose[:3]) + [lo + (hi - lo) / 10 * i] for i in range(10)]
        out = (robot.left_plan_multi_path if arm_tag == "left" else robot.right_plan_multi_path)(cands)
        ok = [i for i, s in enumerate(out["status"]) if s == "Success"]
        return cands[ok[0]] if ok else None

    def choose_grasp_pose(actor, arm_tag, pre_dis=0.1, target_dis=0, contact_point_id=None):
        pose = task.choose_best_pose([0.1, -0.1, 0.9], None, arm_tag)
        tilts.append(None if pose is None else pose[3])
        return (pose, pose) if pose is not None else (None, None)

    task.choose_best_pose, task.choose_grasp_pose = choose_best_pose, choose_grasp_pose
    task.container = actor = object()
    # the stock pose (tilt 0) is blocked at every margin; a tilt of 0.24 rad plans at the base margin
    robot.batch = ["MotionGenStatus.IK_FAIL", lambda c: abs(c[3] - 0.24) < 1e-9]
    uninstall = safe_plan.install_plan_hooks(env, status_sink=sink, world=_World(), margins=safe_plan.MARGINS,
                                             sync=_sync_log()[1], wide_grasp=[0.0, 0.8])
    pre, pose = task.choose_grasp_pose(actor, arm_tag="left")
    assert tilts == [pytest.approx(0.24)] and robot.left_rotate_lim == [0, 0]        # the stock range is back
    assert robot.calls == [("batch", "left")] * 2           # the stock search at the base margin only, then the wide one
    assert task.episode_info["safe_wide_grasp"] == 1
    assert task.episode_info["safe_plan_log"][0]["skill"] == "left grasp_search container"
    # nothing plans: no grasp pose -> the motion ends as a plan failure, RoboTwin's placeholder, no assertion
    robot.batch = ["MotionGenStatus.IK_FAIL"] * 4
    assert task.choose_grasp_pose(actor, arm_tag="left") == ([-1.0] * 7, [-1.0] * 7)
    assert task.plan_success is False and task.episode_info["safe_fail"] == "no_grasp"
    assert task.episode_info["safe_plan_log"][-1]["kind"] == "no_grasp"
    uninstall()
    assert "left_plan_path" not in vars(robot)
    assert task.choose_grasp_pose is choose_grasp_pose and task.choose_best_pose is choose_best_pose


def test_plan_log_is_bounded(monkeypatch):
    monkeypatch.setattr(safe_plan, "LOG_MAX", 4)
    env, sink = _env()
    env.robot.single = ["MotionGenStatus.TRAJOPT_FAIL"] * 6
    safe_plan.install_plan_hooks(env, status_sink=sink, world=_World(), margins=safe_plan.MARGINS, sync=_sync_log()[1])
    env.robot.left_plan_path([0, 0, 1, 1, 0, 0, 0])
    env.robot.left_plan_path([0, 0, 1, 1, 0, 0, 0])
    info = env.task.episode_info
    assert len(info["safe_plan_log"]) == 4 and info["safe_plan_log_dropped"] == 2


def test_world_aware_unwrap_is_counted(tmp_path, monkeypatch):
    from main.envs.controller import TickChunkedController
    from main.item_envs.controller import RiskTickController

    urdf = tmp_path / "fake.urdf"
    urdf.write_text('<robot name="f"><joint name="j0" type="revolute"><limit velocity="3.0" effort="1"/></joint>'
                    '<joint name="j1" type="revolute"><limit velocity="3.0" effort="1"/></joint></robot>')
    pos = np.linspace([0.0, 0.0], [2 * np.pi - 0.1, 0.2], 50)            # joint 0 the long way round: unwrapped
    plan = {"status": "Success", "position": pos, "velocity": np.gradient(pos, 1 / 250, axis=0)}
    monkeypatch.setattr(TickChunkedController, "plan", lambda self, arm, target_pose, **kw: dict(plan))
    for aware in (False, True):
        entity = SimpleNamespace(get_qlimits=lambda: np.array([[-2 * np.pi, 2 * np.pi]] * 2))
        robot = SimpleNamespace(left_urdf_path=str(urdf), right_urdf_path=str(urdf), left_arm_joints_name=["j0", "j1"],
                                right_arm_joints_name=["j0", "j1"], left_plan_path=None, right_plan_path=None,
                                left_entity=entity, right_entity=entity)
        task = SimpleNamespace(active_time_scale=1.0, _cehj_skill="", episode_info={})
        ctrl = RiskTickController(SimpleNamespace(robot=robot, task=task, PHYSICS_FREQ=250.0, control_freq=25.0))
        ctrl.world_aware_only = aware
        out = ctrl.plan("left", [0, 0, 1, 1, 0, 0, 0])
        assert out["position"][-1][0] == pytest.approx(-0.1)             # the unwrap itself is unchanged
        assert task.episode_info.get("safe_unwraps") == (1 if aware else None)
        # a plan the ladder could not find: the controller's failure is final, its reason lands in safe_fail
        task.episode_info["safe_last_fail"] = "no_route"
        monkeypatch.setattr(TickChunkedController, "plan", lambda self, arm, target_pose, **kw: {"status": "Fail"})
        ctrl.plan("left", [0, 0, 1, 1, 0, 0, 0])
        assert ctrl.n_plan_failures == 1
        assert task.episode_info.get("safe_fail") == ("no_route" if aware else None)
        monkeypatch.setattr(TickChunkedController, "plan", lambda self, arm, target_pose, **kw: dict(plan))


# ----------------------------------------------------------------- the knobs ----
def test_t1_safe_yaml_knobs():
    cfg = yaml.safe_load((TASKS / "T1_selective_avoid.yaml").read_text())
    safe = cfg["safe_mode"]
    assert safe["plan"] == "obstacle_aware" and safe["protect"] == "named"
    assert safe_plan.world_options(safe) == {"margin": 0.01, "shape": "hull", "table": True, "table_clearance": 0.02}
    assert safe_plan.hook_options(safe) == {"payload_margin": 0.01, "margins": (0.01, 0.005, 0.002),
                                            "drop_models": ("hand", "none"), "wide_grasp": [0.0, 0.8]}
    for seed in range(6):   # the risk_collect cells pin 0.5-1.3 and 0.6: safe runs at stock speed whatever was drawn
        p, ref = sample_risk(cfg, np.random.RandomState(seed)), sample_risk(cfg, np.random.RandomState(seed))
        apply_nominal_mode(p, cfg, "safe", "selective_avoid")
        assert p.time_scale == 1.0 and p.safe_time_scale == 1.0 and p.task == ref.task
        apply_nominal_mode(ref, cfg, "careless", "selective_avoid")
        assert ref.as_dict() == sample_risk(cfg, np.random.RandomState(seed)).as_dict()
    rng = np.random.default_rng(0)
    pos, vel = rng.normal(size=(40, 6)), rng.normal(size=(40, 6))
    p1, v1, info = retime(pos, vel, 1.0)                  # time_scale 1 = the plan untouched (rows and values)
    assert info["T_out"] == 40 and np.array_equal(p1, pos) and np.array_equal(v1, vel)
