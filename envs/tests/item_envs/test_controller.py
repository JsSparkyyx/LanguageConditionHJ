"""Non-sim tests for main.item_envs.controller.RiskTickController.

Uses a fake env/robot (no sapien): the CEHJ parent plan() is monkeypatched
with a synthetic plan, and the retime / failure-counting / v_ratio / skill
sync / realized-ratio tracking paths are exercised directly.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.controller lives in the sibling CEHJ copy

from main.envs.controller import TickChunkedController
from main.item_envs.controller import RiskTickController, _urdf_velocity_limits

URDF = """<robot name="fake">
  <joint name="j0" type="revolute"><limit velocity="1.0" effort="1"/></joint>
  <joint name="j1" type="revolute"><limit velocity="2.0" effort="1"/></joint>
</robot>
"""


class FakeRobot:
    def __init__(self, urdf_path):
        self.left_urdf_path = str(urdf_path)
        self.right_urdf_path = str(urdf_path)
        self.left_arm_joints_name = ["j0", "j1"]
        self.right_arm_joints_name = ["j0", "j1"]
        self.left_plan_path = None
        self.right_plan_path = None
        self.q = np.zeros(2)

    def get_left_arm_real_jointState(self):
        return [float(self.q[0]), float(self.q[1]), 0.5]  # + gripper val

    def get_right_arm_real_jointState(self):
        return [float(self.q[0]), float(self.q[1]), 0.5]


def make_ctrl(tmp_path, monkeypatch, plan_result, skill="", labels=(), scale=1.0):
    urdf = tmp_path / "fake.urdf"
    urdf.write_text(URDF)
    task = types.SimpleNamespace(
        active_time_scale=scale,
        _time_scale_risky=1.3,
        _time_scale_safe=0.6,
        _cehj_skill=skill,
    )
    env = types.SimpleNamespace(
        robot=FakeRobot(urdf), task=task, PHYSICS_FREQ=250.0, control_freq=25.0
    )
    monkeypatch.setattr(TickChunkedController, "plan", lambda self, arm, target_pose, **kw: plan_result)
    ctrl = RiskTickController(env, risky_skill_labels=labels)
    return ctrl, env


def success_plan(n_rows=30, n_cols=2, vel=0.5):
    pos = np.linspace(np.zeros(n_cols), np.ones(n_cols), n_rows)
    return {"status": "Success", "position": pos, "velocity": np.full_like(pos, vel)}


# ---------------------------------------------------------------------------
# retime wiring
# ---------------------------------------------------------------------------


def test_plan_retime_applies_active_scale(tmp_path, monkeypatch):
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, success_plan(), scale=2.0)
    result = ctrl.plan("left", [[0, 0, 0, 1, 0, 0, 0]])
    info = result["_item_retime"]
    assert info["s"] == pytest.approx(2.0)
    assert info["T_in"] == 30
    assert info["T_out"] < 30  # s > 1 shortens the plan
    # velocity scaled by s, never clipped: 0.5 * 2 = 1.0 -> ratio 1.0/1.0 = 1
    assert np.allclose(result["velocity"], 1.0)
    assert ctrl.retime_log[-1]["arm"] == "left"
    assert result["position"][0] == pytest.approx([0.0, 0.0])
    assert result["position"][-1] == pytest.approx([1.0, 1.0])  # exact endpoint


def test_plan_failure_counted_not_retimed(tmp_path, monkeypatch):
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, None)
    assert ctrl.plan("left", None) is None
    ctrl2, _ = make_ctrl(tmp_path, monkeypatch, {"status": "Timeout"})
    ctrl2.plan("right", None)
    assert ctrl.n_plan_failures == 1
    assert ctrl2.n_plan_failures == 1
    assert ctrl.retime_log == [] and ctrl2.retime_log == []


def test_degenerate_plan_untouched(tmp_path, monkeypatch):
    one_row = {"status": "Success", "position": np.zeros((1, 2)), "velocity": np.zeros((1, 2))}
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, one_row, scale=2.0)
    result = ctrl.plan("left", None)
    assert "_item_retime" not in result
    assert ctrl.n_plan_failures == 0
    assert ctrl.retime_log == []


def test_v_ratio_event_enables_tracking(tmp_path, monkeypatch):
    # v' = 0.5 * 3 = 1.5 on a 1.0-limit joint -> v_ratio 1.5 > 1
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, success_plan(vel=0.5), scale=3.0)
    assert not ctrl._tracking_enabled
    ctrl.plan("left", None)
    assert len(ctrl.v_ratio_events) == 1
    assert ctrl.v_ratio_events[0]["v_ratio"] == pytest.approx(1.5)
    assert ctrl._tracking_enabled


def test_slow_scale_no_v_ratio_event(tmp_path, monkeypatch):
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, success_plan(vel=0.5), scale=0.6)
    ctrl.plan("left", None)
    assert ctrl.v_ratio_events == []
    assert not ctrl._tracking_enabled


# ---------------------------------------------------------------------------
# T1 skill-label time-scale sync
# ---------------------------------------------------------------------------


def test_skill_sync_sets_risky_and_safe(tmp_path, monkeypatch):
    ctrl, env = make_ctrl(
        tmp_path, monkeypatch, success_plan(), skill="left move dx=0.1", labels=("move", "place")
    )
    ctrl.plan("left", None)
    assert env.task.active_time_scale == pytest.approx(1.3)  # risky label matched
    env.task._cehj_skill = "left grasp box_0"
    ctrl.plan("left", None)
    assert env.task.active_time_scale == pytest.approx(0.6)  # non-risky -> safe
    assert ctrl.retime_log[-1]["s"] == pytest.approx(0.6)


def test_skill_sync_inert_without_labels(tmp_path, monkeypatch):
    ctrl, env = make_ctrl(tmp_path, monkeypatch, success_plan(), skill="left move dx=0.1", labels=())
    env.task.active_time_scale = 1.0
    ctrl.plan("left", None)
    assert env.task.active_time_scale == pytest.approx(1.0)  # untouched by sync


# ---------------------------------------------------------------------------
# realized-vs-commanded tracking
# ---------------------------------------------------------------------------


def _ctx(ctrl, pos=None, i=0):
    pos = np.linspace(np.zeros(2), np.ones(2), 30) if pos is None else pos
    left = {"status": "Success", "position": pos}
    return {"left_arm": left, "right_arm": None, "i": i, "left_g": None, "right_g": None, "g_i": 0}


def test_realized_ratio_accumulates(tmp_path, monkeypatch):
    ctrl, env = make_ctrl(tmp_path, monkeypatch, success_plan())
    ctrl._tracking_enabled = True
    ctx = _ctx(ctrl)
    ctrl._tracking_tick_hook(ctx)  # chunk start recorded: cmd = pos[10] - q0
    env.robot.q[:] = [0.1, 0.1]  # arm moved 0.1 rad per joint before next boundary
    ctrl._tracking_tick_hook(ctx)  # same plan objects -> pair appended
    assert len(ctrl._realized_pairs) == 1
    achieved, commanded = ctrl._realized_pairs[0]
    assert achieved == pytest.approx(np.linalg.norm([0.1, 0.1]))
    spt = ctrl.steps_per_tick
    expected_cmd = float(np.linalg.norm(np.linspace(np.zeros(2), np.ones(2), 30)[spt]))
    assert commanded == pytest.approx(expected_cmd)
    assert ctrl.mean_realized_ratio == pytest.approx(achieved / commanded)


def test_pending_chunk_discarded_on_replan(tmp_path, monkeypatch):
    ctrl, env = make_ctrl(tmp_path, monkeypatch, success_plan())
    ctrl._tracking_enabled = True
    ctrl._tracking_tick_hook(_ctx(ctrl))
    env.robot.q[:] = [0.1, 0.1]
    ctrl._tracking_tick_hook(_ctx(ctrl))  # NEW plan objects -> old chunk discarded
    assert ctrl._realized_pairs == []


def test_tracking_gated_until_first_v_ratio_event(tmp_path, monkeypatch):
    ctrl, env = make_ctrl(tmp_path, monkeypatch, success_plan())
    ctx = _ctx(ctrl)
    ctrl._tracking_tick_hook(ctx)
    env.robot.q[:] = [0.1, 0.1]
    ctrl._tracking_tick_hook(ctx)
    assert ctrl._realized_pairs == []  # tracking disabled: nothing accumulated


def test_user_hook_chained(tmp_path, monkeypatch):
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, success_plan())
    ctx = _ctx(ctrl)
    assert ctrl._tracking_tick_hook(ctx) is False  # no user hook -> play on
    ctrl._user_tick_hook = lambda c: True
    assert ctrl._tracking_tick_hook(ctx) is True  # intervention passes through


def test_mean_realized_ratio_none_without_command(tmp_path, monkeypatch):
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, success_plan())
    assert ctrl.mean_realized_ratio is None
    ctrl._realized_pairs.append((0.3, 0.0))  # zero-command chunks are masked
    assert ctrl.mean_realized_ratio is None


# ---------------------------------------------------------------------------
# URDF velocity limits
# ---------------------------------------------------------------------------


def test_urdf_velocity_limits(tmp_path):
    urdf = tmp_path / "fake.urdf"
    urdf.write_text(URDF)
    assert _urdf_velocity_limits(urdf) == {"j0": 1.0, "j1": 2.0}


def test_v_limit_fallback_on_missing(tmp_path, monkeypatch):
    ctrl, _env = make_ctrl(tmp_path, monkeypatch, success_plan())
    # overwrite the URDF AFTER make_ctrl wrote the good one (read lazily here)
    urdf = tmp_path / "fake.urdf"
    urdf.write_text(
        '<robot name="x"><joint name="j0" type="revolute"/>'
        '<joint name="j1" type="revolute"><limit velocity="2.0" effort="1"/></joint></robot>'
    )
    v = ctrl._v_limit("left")
    assert v.tolist() == [2.0, 2.0]  # j0 fell back to DEFAULT_JOINT_VEL (logged)
