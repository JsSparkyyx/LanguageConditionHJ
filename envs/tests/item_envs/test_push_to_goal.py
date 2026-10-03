"""push_to_goal segment logic, no simulator (fake task + fake box).

The fake "physics" teleports the box to sit just ahead of the gripper after
every push action, optionally with a sideways drift — the failure the closed
loop must correct (diag job 22636182: open-loop pushes walked 3-5 cm off)."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.item_envs.test_push_zone import _FakePose, _load_task_module


class _Box:
    def __init__(self, xy, half=0.03):
        self.p = np.array([xy[0], xy[1], 0.74 + half])
        self.config = {"extents": [half] * 3, "scale": [half] * 3, "center": [0, 0, 0]}

    def get_pose(self):
        return _FakePose(self.p.copy(), [1, 0, 0, 0])


class _Task:
    table_z_bias = 0.0

    def __init__(self, box, drift):
        self.box, self.drift = box, drift
        self.ee = np.array([-0.4, -0.2, 1.0, 0.5, -0.5, 0.5, 0.5])
        self.log = []

    def get_arm_pose(self, arm):
        return self.ee.copy()

    def move_to_pose(self, arm, pose):
        return arm, [SimpleNamespace(kind="move", pose=np.asarray(pose, dtype=float))]

    def move_by_displacement(self, arm, x=0.0, y=0.0, z=0.0):
        return arm, [SimpleNamespace(kind="disp", pose=self.ee + np.array([x, y, z, 0, 0, 0, 0]))]

    def close_gripper(self, arm):
        return arm, [SimpleNamespace(kind="close", pose=None)]

    def execute(self, actions):
        for a in actions:
            if a.pose is None:
                continue
            start = self.ee[:2].copy()
            self.ee = np.array(a.pose, dtype=float)
            seg = self.ee[:2] - start
            n = float(np.linalg.norm(seg))
            # contact: the box ends one half-width ahead of a forward-moving gripper
            if n > 1e-6:
                d = seg / n
                ahead = self.box.p[:2] - start
                if 0.0 < float(ahead @ d) < 0.08 and abs(float(ahead @ np.array([-d[1], d[0]]))) < 0.03:
                    self.box.p[:2] = self.ee[:2] + d * 0.03 + np.array([-d[1], d[0]]) * self.drift * n
            self.log.append((a.kind, self.ee[:2].round(3).tolist()))


@pytest.fixture(scope="module")
def utils_mod():
    pz, restore = _load_task_module()
    import sys

    mod = sys.modules["pz_replay._item_utils"]
    yield mod
    restore()


def _run(utils_mod, drift, overshoot=0.0):
    box = _Box([-0.15, -0.10])
    task = _Task(box, drift)
    goal = np.array([0.15, -0.10])
    steps = utils_mod.push_to_goal(task, "left", box, goal, overshoot=overshoot, seg_len=0.10)
    for entry in steps:
        step = utils_mod.realize_step(task, entry)
        arm_actions = step.step if isinstance(step, utils_mod.RiskyStep) else step
        task.execute(arm_actions[1])
    return box, goal, task


def test_reaches_goal_without_drift(utils_mod):
    box, goal, _ = _run(utils_mod, drift=0.0)
    assert np.linalg.norm(box.p[:2] - goal) < 0.01


def test_closed_loop_corrects_sideways_drift(utils_mod):
    # 20% sideways creep per metre pushed: open loop would end ~6 cm off
    box, goal, task = _run(utils_mod, drift=0.2)
    assert np.linalg.norm(box.p[:2] - goal) < 0.02, (box.p[:2], task.log)


def test_overshoot_is_applied_once_on_the_last_segment(utils_mod):
    box, goal, _ = _run(utils_mod, drift=0.0, overshoot=0.05)
    past = float((box.p[:2] - goal) @ np.array([1.0, 0.0]))
    assert 0.03 < past < 0.07, past
