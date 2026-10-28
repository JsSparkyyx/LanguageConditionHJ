"""T3 safe mode: push_via (robotwin_tasks/_item_utils.py) on a fake task, no simulator.

The fake "physics" extends test_push_to_goal's: a box with a yaw, pushed only when the closed fingers move at push
height and their front face sweeps into its back face (the box then sits flush ahead of them, optionally creeping
sideways and turning per metre pushed); backed-off, sideways and lifted moves leave it alone. The fake EE takes every
commanded pose exactly. The fake arm joints (_Ur5Wrist) are the review's ur5 wrist_3 model."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.item_envs.test_push_zone import _FakePose, _load_task_module

TABLE_Z = 0.74
TIP_LEN = 0.164
PUSH_Z = TABLE_Z + 0.008 + TIP_LEN   # EE height of a push (fingertips 8 mm above the table)
CONTACT = 0.015                      # finger front face ahead of the EE


class _Box:
    def __init__(self, xy, half=(0.03, 0.025, 0.025), yaw=0.0):
        self.p = np.array([xy[0], xy[1], TABLE_Z + half[2]], dtype=float)
        self.half = np.asarray(half, dtype=float)
        self.yaw = float(yaw)
        self.config = {"extents": list(half), "scale": list(half), "center": [0, 0, 0]}
        self.path = [self.p[:2].copy()]

    def get_pose(self):
        return _FakePose(self.p.copy(), [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)])

    def depth(self, d):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return abs(d[0] * c + d[1] * s) * self.half[0] + abs(d[1] * c - d[0] * s) * self.half[1]


# ur5-wsg at RoboTwin's homestate (wrist_3 = 0): the EE poses of the careless log's first plans
# (runs/lang/t6_1023/tilt.log, plan #1 left / #2 right); the arm bases (ARM_DISTANCE 0.8, y = -0.65)
HOME_EE = {"left": [-0.305, -0.168, 1.115, 0.502, -0.495, 0.484, 0.518],
           "right": [0.495, -0.168, 1.215, 0.501, -0.495, 0.484, 0.518]}
ARM_BASE = {"left": (-0.4, -0.65), "right": (0.4, -0.65)}


class _Ur5Wrist:
    """A fake RoboTwin robot, one arm: the review's ur5 wrist_3 model (review_t3/wrist3_check_model.py, fit on the
    careless T3 log runs/lang/t6_1023/tilt.log, seeds 6400 / 6401, within 0.15 rad): shoulder pan = the EE's
    bearing about the arm base - 2.96, wrist_3 = pan - yaw + 1.583, followed move by move (the hand turns the short
    way round); a wrist_3 past +-pi is reached the other way round, as the real IK must (a turn of 2 pi less the
    asked one, the wrap the review predicted). The other joints stay at the homestate. ``kick``: {move index: rad}
    extra wrist_3 jumps (an IK branch flip)."""

    def __init__(self, ee_yaw, arm, ee, w3=0.0, kick=None):
        self.yaw_of, self.arm = ee_yaw, arm
        self.base = np.asarray(ARM_BASE[arm], dtype=float)
        setattr(self, f"{arm}_entity_origion_pose", SimpleNamespace(p=np.array([*ARM_BASE[arm], 0.65])))
        setattr(self, f"get_{arm}_arm_real_jointState", self.joints)
        self.ee = np.asarray(ee, dtype=float).copy()
        self.pan, self.w3 = self._pan(self.ee), float(w3)
        self.kick, self.n, self.w3_log = dict(kick or {}), 0, [float(w3)]

    def _pan(self, ee):
        return math.atan2(ee[1] - self.base[1], ee[0] - self.base[0]) - 2.96

    def follow(self, ee):
        ee = np.asarray(ee, dtype=float)
        pan = self._pan(ee)
        turn = (self.yaw_of(ee[3:]) - self.yaw_of(self.ee[3:]) + math.pi) % (2.0 * math.pi) - math.pi
        w = self.w3 + (pan - self.pan) - turn + self.kick.get(self.n, 0.0)
        if abs(w) > math.pi:
            w -= math.copysign(2.0 * math.pi, w)
        self.n += 1
        self.pan, self.w3, self.ee = pan, w, ee.copy()
        self.w3_log.append(w)

    def joints(self):
        return [self.pan, -1.5447, -1.5447, -1.5794, 1.5794, self.w3, 0.0]   # + the gripper value


class _Task:
    table_z_bias = 0.0

    def __init__(self, box, drift=0.0, turn=0.0):
        self.box, self.drift, self.turn = box, drift, turn
        self.ee = np.array([-0.30, -0.17, 1.12, -0.5, 0.5, -0.5, -0.5])   # home, TOP_DOWN
        self.moves = []
        self.robot = None

    def get_arm_pose(self, arm):
        return self.ee.copy()

    def move_to_pose(self, arm, pose):
        return arm, [SimpleNamespace(kind="move", pose=np.asarray(pose, dtype=float))]

    def close_gripper(self, arm):
        return arm, [SimpleNamespace(kind="close", pose=None)]

    def execute(self, actions):
        for a in actions:
            if a.pose is None:
                continue
            p0, self.ee = self.ee.copy(), a.pose.copy()
            self.moves.append(self.ee.copy())
            if self.robot is not None:
                self.robot.follow(self.ee)
            self._contact(p0, self.ee)

    def _contact(self, p0, p1):
        if max(p0[2], p1[2]) > PUSH_Z + 0.01:   # lifted (or coming down): the fingers pass over the box
            return
        seg = p1[:2] - p0[:2]
        n = float(np.linalg.norm(seg))
        if n < 1e-6:
            return
        d = seg / n
        perp = np.array([-d[1], d[0]])
        rel = self.box.p[:2] - p0[:2]
        gap = float(rel @ d) - self.box.depth(d) - CONTACT   # finger face -> box face along the move
        if -0.005 <= gap <= n and abs(float(rel @ perp)) < 0.03:
            pushed = n - max(gap, 0.0)
            self.box.yaw += self.turn * pushed   # a box that turns under the fingers (``turn`` rad per m)
            self.box.p[:2] = p1[:2] + d * (CONTACT + self.box.depth(d)) + perp * self.drift * pushed
            self.box.path.append(self.box.p[:2].copy())


@pytest.fixture(scope="module")
def utils_mod():
    pz, restore = _load_task_module()
    import sys

    yield sys.modules["pz_replay._item_utils"]
    restore()


def _run(utils_mod, task, steps, hook=None):
    for k, entry in enumerate(steps):
        if hook is not None:
            hook(k, task)
        step = utils_mod.realize_step(task, entry)
        arm_actions = step.step if isinstance(step, utils_mod.RiskyStep) else step
        task.execute(arm_actions[1])


def _push_via(utils_mod, task, via, **kw):
    kw.setdefault("tip_len", TIP_LEN)
    return utils_mod.push_via(task, "left", task.box, via, **kw)


def _yaws(utils_mod, task):
    return [utils_mod._ee_yaw(m[3:]) for m in task.moves if m[2] <= PUSH_Z + 1e-6]


def _wrap_deg(a):
    return abs((math.degrees(a) + 180.0) % 360.0 - 180.0)


def test_route_followed_through_every_via_point(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    via = [[-0.20, -0.20], [0.05, -0.20], [0.20, -0.10]]
    log = {}
    _run(utils_mod, task, _push_via(utils_mod, task, via, log=log))
    assert log["fail"] is None, log
    assert np.linalg.norm(box.p[:2] - via[-1]) < 0.01, box.p
    path = np.stack(box.path)
    for v in via[:-1]:
        assert np.min(np.linalg.norm(path - np.asarray(v), axis=1)) < 0.015, (v, path.round(3))
    # never pushed past a via point: every push ends at most seg_len on, no overshoot at the goal
    assert float((box.p[:2] - np.asarray(via[-1])) @ np.array([1.0, 0.0])) < 0.01


def test_first_two_steps_pair_with_the_idle_arm(utils_mod):
    task = _Task(_Box([-0.30, -0.10]))
    steps = _push_via(utils_mod, task, [[0.10, -0.10]], max_segments=5)
    assert steps[0][1][0].kind == "close"
    assert isinstance(steps[1], utils_mod._Deferred) and not isinstance(steps[1], utils_mod.RiskyStep)
    assert len(steps) == 2 + 2 * 5 + 1
    assert all(isinstance(s, utils_mod.RiskyStep) for s in steps[3:-1:2])   # the pushes run risky


def test_no_yaw_flip_when_the_heading_crosses_pure_x(utils_mod):
    """Legs at -20 then +20 deg to +x: _push_quat folds the two yaws 140 deg apart (the careless 180-deg flip,
    seed 6400); push_via keeps the one nearest the live yaw."""
    d1 = np.array([math.cos(math.radians(-20)), math.sin(math.radians(-20))])
    d2 = np.array([math.cos(math.radians(20)), math.sin(math.radians(20))])
    jump = _wrap_deg(utils_mod._ee_yaw(utils_mod._push_quat(d1)) - utils_mod._ee_yaw(utils_mod._push_quat(d2)))
    assert jump > 100.0   # the fold flips
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    a = np.array([-0.30, -0.10])
    via = [list(a + d1 * 0.22), list(a + d1 * 0.22 + d2 * 0.22)]
    log = {}
    _run(utils_mod, task, _push_via(utils_mod, task, via, log=log))
    assert log["fail"] is None and np.linalg.norm(box.p[:2] - via[-1]) < 0.01
    yaws = _yaws(utils_mod, task)
    assert max(_wrap_deg(b - a) for a, b in zip(yaws[:-1], yaws[1:])) < 100.0
    assert log["yaw_jump_max_deg"] < 100.0


@pytest.mark.parametrize("turn_deg, lifts", [(60.0, 1), (20.0, 0)])
def test_corner_lifts_the_hand_over_the_box(utils_mod, turn_deg, lifts):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    d2 = np.array([math.cos(math.radians(-turn_deg)), math.sin(math.radians(-turn_deg))])
    via = [[-0.12, -0.10], list(np.array([-0.12, -0.10]) + d2 * 0.15)]
    log = {}
    _run(utils_mod, task, _push_via(utils_mod, task, via, log=log))
    assert log["fail"] is None and np.linalg.norm(box.p[:2] - via[-1]) < 0.01, log
    assert sum(1 for s in log["steps"] if s["kind"] == "lift") == lifts
    first_push = next(i for i, m in enumerate(task.moves) if m[2] <= PUSH_Z + 1e-6 and i > 1)
    high = [m for m in task.moves[first_push:] if m[2] > PUSH_Z + 0.02]
    if lifts:
        top = TABLE_Z + 2 * box.half[2]
        assert high and min(m[2] for m in high) >= top + 0.02 + TIP_LEN - 1e-6   # fingertips 2 cm over the box
    else:
        assert not high[:-1]   # only the retreat goes up


def test_lane_drift_replans_and_stops_without_a_route(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    calls, fails = [], []

    def replan(c, yaw):
        calls.append(np.asarray(c).copy())
        return None

    log = {}
    steps = _push_via(utils_mod, task, [[0.20, -0.10]], replan=replan, on_fail=fails.append, log=log)

    def hook(k, task):
        if k == 6:
            box.p[1] += 0.03   # knocked 3 cm off its leg line (lane_tol 2 cm)

    _run(utils_mod, task, steps, hook)
    assert len(calls) == 1 and fails == ["lane"] and log["fail"] == "lane"
    n_after = len(task.moves)
    assert all(utils_mod.realize_step(task, e) is not None for e in steps[-3:])
    assert len(task.moves) == n_after


def test_replan_continues_on_the_new_route(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    goal = np.array([0.20, -0.10])

    def replan(c, yaw):
        return [list(goal)]

    log = {}
    steps = _push_via(utils_mod, task, [list(goal)], replan=replan, log=log)

    def hook(k, task):
        if k == 6:
            box.p[1] += 0.03

    _run(utils_mod, task, steps, hook)
    assert log["fail"] is None and len(log["replans"]) == 1 and log["replans"][0]["why"] == "lane"
    assert np.linalg.norm(box.p[:2] - goal) < 0.01


def test_guard_failure_stops_the_push(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    seen, fails = [], []

    def guard(poses, box_from, box_to, yaw):
        seen.append(len(poses))
        return (len(seen) < 4), {"check": "hand_cyl", "elem": "obstacle_0", "leg": 0}

    log = {}
    _run(utils_mod, task, _push_via(utils_mod, task, [[0.20, -0.10]], guard=guard, on_fail=fails.append, log=log))
    assert fails == ["guard"] and log["fail"] == "guard"
    assert len(seen) == 4   # nothing is checked (or moved) after the stop
    blocked = next(s for s in log["steps"] if "guard" in s)
    assert blocked["guard"]["check"] == "hand_cyl"


def test_guard_sees_the_live_pose_first_and_the_box_slide(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    calls = []

    def guard(poses, box_from, box_to, yaw):
        calls.append((np.asarray(poses[0]).copy(), task.ee.copy(), np.asarray(box_from), np.asarray(box_to)))
        return True, None

    _run(utils_mod, task, _push_via(utils_mod, task, [[0.20, -0.10]], guard=guard))
    assert calls and all(np.allclose(p0, ee) for p0, ee, _, _ in calls)
    moving = [(b0, b1) for _, _, b0, b1 in calls if np.linalg.norm(b1 - b0) > 1e-9]
    assert moving and all(np.linalg.norm(b1 - b0) <= 0.06 + 1e-9 for b0, b1 in moving)   # seg_len


def test_retreat_backs_off_then_lifts(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    _run(utils_mod, task, _push_via(utils_mod, task, [[0.10, -0.10]]))
    last_push, back, up = task.moves[-3], task.moves[-2], task.moves[-1]
    assert back[0] == pytest.approx(last_push[0] - 0.015) and back[2] == pytest.approx(last_push[2])
    assert up[:2] == pytest.approx(back[:2]) and up[2] == pytest.approx(back[2] + 0.08)


def test_box_off_the_table_stops(utils_mod):
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    fails = []
    steps = _push_via(utils_mod, task, [[0.20, -0.10]], on_fail=fails.append)

    def hook(k, task):
        if k == 5:
            box.p[2] = 0.3

    _run(utils_mod, task, steps, hook)
    assert fails == ["box_fell"]


def test_out_of_segments_is_a_failure(utils_mod):
    task = _Task(_Box([-0.30, -0.10]))
    fails = []
    _run(utils_mod, task, _push_via(utils_mod, task, [[0.30, -0.10]], max_segments=3, on_fail=fails.append))
    assert fails == ["segments"]


class _Exec:
    """Fake RoboTwin executor on a replayed push_zone task (test_push_zone's loader): two EEs taking every pose
    exactly, the push box (a stub entity) pushed by the push arm's fingers as in _Task; move() does nothing once
    plan_success is False, as RoboTwin's."""

    def __init__(self, task, wrist=False, kick=None, turn=0.0):
        import sys

        self.task, self.turn = task, turn
        rec = task.push_objects[0]
        self.arm, self.ent = rec["arm"], rec["obj"].actor
        self.box = _Box(np.asarray(self.ent.get_pose().p)[:2], half=tuple(rec["obj"].config["extents"]))
        top = [-0.5, 0.5, -0.5, -0.5]
        self.ee = {"left": np.array([-0.30, -0.17, 1.12] + top), "right": np.array([0.30, -0.17, 1.12] + top)}
        self.robot = None
        if wrist:   # the real home poses and the fake ur5 joints of the pushing arm
            self.ee = {arm: np.array(HOME_EE[arm]) for arm in ("left", "right")}
            self.robot = _Ur5Wrist(sys.modules["pz_replay._item_utils"]._ee_yaw, self.arm, self.ee[self.arm],
                                   kick=kick)
            task.robot = self.robot
        self.moves = {"left": [], "right": []}
        self.pushers = {}
        task.get_arm_pose = lambda arm: self.ee[arm].copy()
        task.move_to_pose = lambda arm, pose: (arm, [SimpleNamespace(kind="move", pose=np.asarray(pose, dtype=float))])
        task.close_gripper = lambda arm, pos=0.0: (arm, [SimpleNamespace(kind="close", pose=None)])
        task.move = self.move

    def move(self, a1, a2=None):
        if self.task.plan_success is False:
            return False
        for step in (a1, a2):
            if step is None:
                continue
            arm, actions = step
            for a in actions:
                if a.pose is None:
                    continue
                p0, self.ee[arm] = self.ee[arm].copy(), a.pose.copy()
                self.moves[arm].append(a.pose.copy())
                if arm == self.arm:
                    if self.robot is not None:
                        self.robot.follow(a.pose)
                    pusher = self.pushers.setdefault(arm, _Task(self.box, turn=self.turn))
                    pusher._contact(p0, a.pose)
                    yaw = self.box.yaw
                    self.ent.set_pose(_FakePose(self.box.p.copy(), [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
        return True


def _safe_task(seed, protect=None, **exec_kw):
    import yaml
    from pathlib import Path

    from tests.item_envs.test_push_zone import _replay_layout

    envs = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((envs / "main" / "item_envs" / "configs" / "tasks" / "T3_push_zone.yaml").read_text())
    pz = __import__("sys").modules["pz_replay.push_zone"]
    task = _replay_layout(pz, seed)
    task.info, task.plan_success = {}, True
    task.embodiment_params = dict(cfg["embodiment_overrides"]["ur5-wsg"])
    npz = envs / "main" / "item_envs" / "configs" / "workspace_ur5-wsg.npz"
    task.safe_route = {**cfg["safe_mode"]["route"], "protect": protect,
                       "reach_npz": str(npz) if npz.is_file() else None}
    return task, _Exec(task, **exec_kw)


def test_push_zone_safe_play_once_follows_the_route(utils_mod):
    """push_zone.play_once in the safe mode (protect every element), on replayed layouts: the route is planned,
    recorded, pushed in the fake physics to the goal; the idle arm closes and tucks; no plan failure."""
    pz = __import__("sys").modules["pz_replay.push_zone"]
    done = 0
    for seed in (0, 6, 7, 13):
        task, ex = _safe_task(seed)
        task.play_once()
        info = task.episode_info
        if not info["route"]["ok"]:
            continue
        done += 1
        assert "safe_fail" not in info and task.plan_success is True, (seed, info.get("safe_fail"), info["push"])
        goal = np.asarray(task.push_objects[0]["goal_xy"])
        assert np.linalg.norm(ex.box.p[:2] - goal) <= pz.SUCCESS_TOL, (seed, ex.box.p, goal)
        idle = "right" if ex.arm == "left" else "left"
        assert len(ex.moves[idle]) == 1 and ex.moves[idle][0][2] == pytest.approx(0.74 + pz.TUCK_POSE[2])
        assert info["push"]["fail"] is None and info["push"]["yaw_jump_max_deg"] < 100.0
        assert task.last_push_info[0]["via"] == info["route"]["via"]
    assert done >= 2


def test_push_zone_safe_play_once_without_a_route_does_not_push(utils_mod):
    """A layout with no route (seed 19: a cylinder on the line, the near side off the table, an off-path mat on
    the far side) is not pushed at all: a plan failure with safe_fail "no_route" and the planner's reasons."""
    task, ex = _safe_task(19)
    task.play_once()
    info = task.episode_info
    assert info["route"]["ok"] is False and info["safe_fail"] == "no_route" and task.plan_success is False
    assert info["route"]["reason"] and info["route"]["detail"]["blocking"] == ["obstacle_0"]
    assert ex.moves == {"left": [], "right": []}
    # the same layout, only the off-path mat named: nothing protected blocks the line, the push goes straight
    task, ex = _safe_task(19, protect=[e["id"] for e in task.layout_info["elements"] if not e["on_path"]])
    task.play_once()
    assert task.episode_info["route"]["ok"] and len(task.episode_info["route"]["via"]) == 1


def test_push_quat_near_keeps_the_fingers_across_and_the_yaw_continuous(utils_mod):
    def R(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    live = list(utils_mod.TOP_DOWN_QUAT)
    prev = None
    for ang in np.radians(np.arange(-170.0, 171.0, 4.0)):
        d = np.array([math.cos(ang), math.sin(ang)])
        q = utils_mod._push_quat_near(d, live)
        m = R(q)
        assert np.allclose(m[:, 0], [0, 0, -1], atol=1e-9)   # top-down
        assert abs(float(m[:2, 1] @ d)) < 1e-9                # finger axis across the push
        psi = utils_mod._ee_yaw(q)
        if prev is not None:
            assert _wrap_deg(psi - prev) <= 4.0 + 1e-6        # follows the heading, never flips
        prev, live = psi, q
        qn = utils_mod._push_quat_near(d, live, hand="narrow")
        assert abs(abs(float(R(qn)[:2, 1] @ d)) - 1.0) < 1e-9  # narrow: the finger axis along the push
    # nearest the live yaw: the same rotation as _push_quat when the live yaw is _push_quat's own
    for ang in np.radians([-80.0, -30.0, 10.0, 60.0]):
        d = np.array([math.cos(ang), math.sin(ang)])
        q0 = utils_mod._push_quat(d)
        assert abs(abs(float(np.dot(utils_mod._push_quat_near(d, q0), q0))) - 1.0) < 1e-9


# ------------------------------------------------------------- planner failures, re-plans, a box that turns ----


def _failing_move(task, ex, at, controller_reason=None):
    """task.move that fails on its ``at``-th call as a cuRobo failure does in RoboTwin (plan_success False, nothing
    moves; ``controller_reason``: the T1 controller's safe_fail, written first)."""
    calls = {"n": 0, "after": 0, "steps_then": None}

    def move(a1, a2=None):
        calls["n"] += 1
        if calls["n"] == at:
            task.plan_success = False
            if controller_reason is not None:
                task.episode_info.setdefault("safe_fail", controller_reason)
            calls["steps_then"] = len(task.episode_info["push"]["steps"])
            return False
        if task.plan_success is False:
            calls["after"] += 1
        return ex.move(a1, a2)

    task.move = move
    return calls


@pytest.mark.parametrize("controller_reason", [None, "plan_fail"])
def test_planner_failure_mid_push_is_a_plan_failure(utils_mod, controller_reason):
    """A cuRobo failure in the middle of the push (the 6th move; review_t3/plan_fail_label.py): the push stops at
    once, no safe_fail of its own (the controller's reason, when it wrote one, stands), plan_fail_step names the
    failed step, no phantom push steps are logged and nothing moves after it."""
    done = 0
    for seed in (0, 6, 7, 13):
        task, ex = _safe_task(seed)
        calls = _failing_move(task, ex, 6, controller_reason)
        task.episode_info = {}
        task.play_once()
        info = task.episode_info
        if not info["route"]["ok"]:
            continue
        done += 1
        assert info.get("safe_fail") == controller_reason, info.get("safe_fail")
        assert info["plan_fail_step"] is not None and info["push"]["fail"] == "plan", info["push"]["fail"]
        assert task.plan_success is False and calls["after"] == 0
        assert len(info["push"]["steps"]) == calls["steps_then"], info["push"]["steps"]   # nothing logged after it
        assert sum(len(v) for v in ex.moves.values()) == 5
    assert done >= 2


def test_push_zone_replans_through_play_route(utils_mod):
    """push_zone._play_route's re-plan: the box knocked 3 cm off its leg mid-push (lane_tol 2 cm) asks the planner
    again from the live box (prefer the old sides, the start at the guard's floor), and the push goes on to the
    goal along the new route (seeds 0, 14, 20; on 6, 7, 13 the knock puts the box inside an element's clearance and
    the push stops with "guard")."""
    pz = __import__("sys").modules["pz_replay.push_zone"]
    done = 0
    for seed in (0, 6, 7, 13, 14, 20):
        task, ex = _safe_task(seed)
        orig, n = ex.move, {"push": 0}

        def move(a1, a2=None, _orig=orig, _ex=ex):
            ok = _orig(a1, a2)
            n["push"] += sum(1 for st in (a1, a2) if st is not None and st[0] == _ex.arm and st[1])
            if n["push"] == 8:   # mid-push: knocked sideways, away from the table edge
                _ex.box.p[1] += 0.03 if _ex.box.p[1] < 0.0 else -0.03
                _ex.ent.set_pose(_FakePose(_ex.box.p.copy(), [1, 0, 0, 0]))
            return ok

        task.move = move
        task.play_once()
        info = task.episode_info
        if not info["route"]["ok"]:
            continue
        lane = [r for r in info["push"]["replans"] if r["why"] == "lane"]
        if not lane or "safe_fail" in info:
            continue   # the knock left no route: covered by test_lane_drift_replans_and_stops_without_a_route
        done += 1
        assert lane[0]["ok"] and info["replans"] and info["replans"][0]["ok"], info["replans"]
        assert task.plan_success is True and info["push"]["fail"] is None
        goal = np.asarray(task.push_objects[0]["goal_xy"])
        assert np.linalg.norm(ex.box.p[:2] - goal) <= pz.SUCCESS_TOL, (seed, ex.box.p, goal)
    assert done >= 2


def test_a_box_that_turns_is_followed(utils_mod):
    """A box that turns under the fingers (40 deg per metre pushed): every segment re-aims from its live pose and
    yaw (the OBB depth along the new heading), the guard sees the live yaw, and the push still ends at the goal."""
    box = _Box([-0.30, -0.10])
    task = _Task(box, turn=math.radians(40.0))
    seen = []

    def guard(poses, box_from, box_to, yaw):
        seen.append((float(yaw), box.yaw))
        return True, None

    log = {}
    via = [[-0.10, -0.16], [0.15, -0.16]]
    _run(utils_mod, task, _push_via(utils_mod, task, via, guard=guard, log=log))
    assert log["fail"] is None, log
    assert np.linalg.norm(box.p[:2] - via[-1]) < 0.012, box.p
    assert abs(math.degrees(box.yaw)) > 10.0
    assert seen and all(abs((a - b + math.pi) % (2 * math.pi) - math.pi) < 1e-6 for a, b in seen)


def test_push_zone_safe_play_once_with_a_turning_box(utils_mod):
    """The same on replayed layouts through play_once (the route checks allow the box +-15 deg)."""
    pz = __import__("sys").modules["pz_replay.push_zone"]
    done = 0
    for seed in (0, 6, 7, 13):
        task, ex = _safe_task(seed, turn=math.radians(25.0))
        task.play_once()
        info = task.episode_info
        if not info["route"]["ok"]:
            continue
        if "safe_fail" in info:   # a guard stop is allowed (it re-checks the turned box); never an unchecked move
            assert info["safe_fail"] in ("guard", "lane", "replans"), info["safe_fail"]
            continue
        done += 1
        goal = np.asarray(task.push_objects[0]["goal_xy"])
        assert np.linalg.norm(ex.box.p[:2] - goal) <= pz.SUCCESS_TOL, (seed, ex.box.p, goal)
        assert abs(math.degrees(ex.box.yaw)) > 5.0
    assert done >= 3


# ------------------------------------------------------------------------------------ the wrist_3 limit ----


BAND = math.pi - 0.3


def _dir_for_yaw(psi):
    """A push direction whose finger-across yaws are psi and psi + pi."""
    return np.array([math.sin(psi), -math.cos(psi)])


def test_pick_yaw_near_the_limit(utils_mod):
    pick = utils_mod.pick_yaw
    d = _dir_for_yaw(-0.5)
    near, other = utils_mod._push_yaws(d, 0.0)
    assert near == pytest.approx(-0.5) and other == pytest.approx(-0.5 + math.pi)
    assert pick(d, 0.0, 0.0, False) == (near, None)
    assert pick(d, 0.0, None, False) == (near, None)                 # no joint readings: the nearest
    # w3 2.5: the nearest would end at 3.0 > pi - 0.3
    psi, why = pick(d, 0.0, 2.5, False)
    assert psi is None and why["pred"] == [pytest.approx(3.0)] and why["lifted"] is False
    psi, why = pick(d, 0.0, 2.5, True)                               # lifted: the 180 deg switch
    assert psi == pytest.approx(other) and why is None
    assert 2.5 - utils_mod._wrap_pi(psi - 0.0) == pytest.approx(2.5 - (math.pi - 0.5))
    # the band edge counts as inside; just past it does not
    assert pick(d, 0.0, BAND - 0.5, False)[0] == pytest.approx(near)
    assert pick(d, 0.0, BAND - 0.5 + 1e-6, False)[0] is None
    # the other side: w3 -2.6 with a turn the other way
    d2 = _dir_for_yaw(0.5)
    assert pick(d2, 0.0, -2.6, False)[0] is None
    assert utils_mod._wrap_pi(pick(d2, 0.0, -2.6, True)[0]) == pytest.approx(0.5 - math.pi)
    # lifted, one of the two always lands in the band (they are pi apart): a wrist already past it comes back
    psi, why = pick(d, 0.0, 3.1, True)
    assert psi == pytest.approx(other) and abs(3.1 - utils_mod._wrap_pi(psi)) <= BAND


def test_wrist_track_follows_the_pan_and_the_yaw(utils_mod):
    """wrist_track = the review's model step by step: a straight +x push of the left arm keeps the yaw, and wrist_3
    follows the pan (the EE's bearing about the arm base)."""
    base = ARM_BASE["left"]
    path = [((x, -0.15), np.array([1.0, 0.0])) for x in np.arange(-0.30, 0.21, 0.06)]
    w = utils_mod.wrist_track(0.0, 0.0, (-0.30, -0.15), path, 0, base_xy=base)
    bearing = lambda x: math.atan2(-0.15 - base[1], x - base[0])   # noqa: E731
    near = utils_mod._push_yaws(np.array([1.0, 0.0]), 0.0)[0]
    expect = [-utils_mod._wrap_pi(near) + bearing(xy[0]) - bearing(-0.30) for xy, _ in path]
    assert np.allclose(w, expect)
    assert w[-1] < w[0] - 0.4   # the pan swings ~0.6 rad over the push
    w_none = utils_mod.wrist_track(0.0, 0.0, (-0.30, -0.15), path, 0, base_xy=None)
    assert np.allclose(w_none, -utils_mod._wrap_pi(near))


def test_wrist_drift_stops_a_straight_push_before_the_limit(utils_mod):
    """wrist_3 drifting toward -pi on a long straight push (the pan swings, the yaw stays): the push stops with
    "wrist_limit" on the move whose prediction leaves the band, before the wrist reaches the limit (no wrap)."""
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    task.robot = _Ur5Wrist(utils_mod._ee_yaw, "left", task.ee)
    fails, log = [], {}
    steps = _push_via(utils_mod, task, [[0.20, -0.10]], on_fail=fails.append, log=log)

    def hook(k, task):
        if k == 2:   # after the approach: the arm came down in a configuration near the limit
            task.robot.w3 = -2.55
            task.robot.w3_log.append(-2.55)

    _run(utils_mod, task, steps, hook)
    assert fails == ["wrist_limit"] and log["fail"] == "wrist_limit", log["fail"]
    stop = log["steps"][-1]
    assert stop["wrist"]["pred"][0] < -BAND and stop["wrist"]["lifted"] is False
    w = np.asarray(task.robot.w3_log[4:])
    assert w.min() > -math.pi and np.all(np.abs(np.diff(w)) < 0.2), w   # it drifted, never wrapped
    assert box.p[0] > -0.25 and np.linalg.norm(box.p[:2] - [0.20, -0.10]) > 0.05   # it pushed, then stopped


def test_lifted_corner_switches_the_yaw_in_parts(utils_mod):
    """At a lifted corner whose nearest yaw would carry wrist_3 past the band, push_via takes the other yaw (the
    180 deg switch): it turns the hand in place, lifted, in parts of at most travel_max - margin (1.7 rad), then
    goes on; no move turns the wrist by more, and the push ends at the goal."""
    box = _Box([-0.30, -0.10])
    task = _Task(box)
    task.robot = _Ur5Wrist(utils_mod._ee_yaw, "left", task.ee)
    d2 = np.array([math.cos(math.radians(-60.0)), math.sin(math.radians(-60.0))])
    via = [[-0.12, -0.10], list(np.array([-0.12, -0.10]) + d2 * 0.15)]
    log = {}
    steps = _push_via(utils_mod, task, via, log=log)

    def hook(k, task):
        if k == 2:
            task.robot.w3 = 2.2
            task.robot.w3_log.append(2.2)

    _run(utils_mod, task, steps, hook)
    assert log["fail"] is None and np.linalg.norm(box.p[:2] - via[-1]) < 0.01, log["fail"]
    assert log["wrist_switches"] == 1
    w = np.asarray(task.robot.w3_log[4:])
    assert np.abs(w).max() <= BAND + 0.05 and np.all(np.abs(np.diff(w)) <= 2.0 - 0.3 + 0.05), w
    lift = next(s for s in log["steps"] if s["kind"] == "lift")
    assert lift["turn_deg"] == pytest.approx(60.0, abs=2.0)
    # the turn happens in place at the lifted back-off pose: consecutive lifted moves with one xy
    high = [m for m in task.moves if m[2] > PUSH_Z + 0.02]
    same_xy = [(a, b) for a, b in zip(high[:-1], high[1:]) if np.linalg.norm(a[:2] - b[:2]) < 1e-9
               and abs(utils_mod._wrap_pi(utils_mod._ee_yaw(a[3:]) - utils_mod._ee_yaw(b[3:]))) > 0.5]
    assert len(same_xy) >= 2, len(same_xy)   # 120 deg in two parts


def test_first_approach_takes_the_yaw_with_headroom(utils_mod):
    """Replayed seed 14 (right arm, around the near side, the last two corners turning clockwise: the review's
    wrist_3 ~ 3.7 rad case): from the home pose the yaw nearest the live one would carry wrist_3 past the band,
    so the first approach takes the other one (a turn above the box in two parts); the push ends at the goal with
    wrist_3 inside the band all the way and no move above 2 rad."""
    pz = __import__("sys").modules["pz_replay.push_zone"]
    task, ex = _safe_task(14, wrist=True)
    task.play_once()
    info = task.episode_info
    assert info["route"]["ok"], info["route"]["reason"]
    first = info["push"]["wrist_first"]
    assert first["pick"] == "other" and first["peaks"][0] > BAND >= first["peaks"][1], first
    assert info["push"]["steps"][0]["kind"] == "approach" and info["push"]["steps"][0]["n_turns"] == 2
    assert "safe_fail" not in info and task.plan_success is True, (info.get("safe_fail"), info["push"]["fail"])
    goal = np.asarray(task.push_objects[0]["goal_xy"])
    assert np.linalg.norm(ex.box.p[:2] - goal) <= pz.SUCCESS_TOL
    assert np.abs(ex.robot.w3_log).max() <= BAND + 0.05
    assert 0.5 < info["push"]["max_joint_travel"] <= 2.0 - 0.3 + 0.1
    assert info["push"]["wrist_switches"] == 0


def test_joint_travel_over_the_cap_stops_the_push(utils_mod):
    """A move whose joints swing more than 2 rad (an IK flip the prediction cannot see: a 2.5 rad wrist_3 kick on
    the 7th move) stops the push at once: safe_fail "joint_travel", the travel recorded, nothing moves after."""
    task, ex = _safe_task(0, wrist=True, kick={6: 2.5})
    task.play_once()
    info = task.episode_info
    assert info["route"]["ok"]
    assert info["safe_fail"] == "joint_travel" and info["push"]["fail"] == "joint_travel"
    assert info["push"]["max_joint_travel"] == pytest.approx(2.5, abs=0.3) and task.plan_success is False
    assert "plan_fail_step" not in info
    assert ex.robot.n == 7   # the kicked move was the last one
