"""Sim tests for the smoke_items Phase 2 acceptance path (PLAN v3 §6 Phase 2).

Runs the full CEHJ Env construction (cuRobo controller install, RoboTwin
setup_demo, table-only cuRobo world) so these need the GPU node — they are
marked ``sim`` and skipped in the login-node suite.

- 5 seeds x 2 embodiments: setup() builds without UnStableError, the rule
  table matches the config contract, and every margin is positive at t = 0.
- One full episode per embodiment at low risk (both scales 0.6): error is
  None, task_success, the t = 0 margins are positive, the time_scale trace
  column follows the risky/safe convention, the grasped box registers as
  held, and meta.json / trace.npz / retime.json / keyframes are written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]

EXPECTED_RULE_IDS = [
    "inside|box_0|table_top",
    "inside|box_1|table_top",
    "outside|box_0|zone_green",
    "outside|box_1|zone_green",
    "avoid|box_0|-",
    "avoid|box_1|-",
]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_setup_margins_positive_at_t0(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner(
        "smoke_items", embodiment=embodiment, seed=seed, render="none", out_root=tmp_path
    )
    try:
        runner.setup()  # raises UnStableError on a bad layout — that is the check
        assert [r.rule_id for r in runner.rules] == EXPECTED_RULE_IDS
        assert [s.item_id for s in runner.items] == ["box_0", "box_1"]
        assert [r.region_id for r in runner.regions] == ["table_top", "zone_green"]
        margins = runner._rule_margins()
        assert margins.shape == (len(EXPECTED_RULE_IDS),)
        assert np.all(np.isfinite(margins))
        assert (margins > 0.0).all(), f"non-positive t=0 margins: {margins}"
        # physics was sampled and applied: masses stay inside the 0.05-0.15 kg
        # smoke range scaled by mass_scale U(0.7, 1.3)
        for spec in runner.items:
            assert 0.03 < spec.mass < 0.20, (spec.item_id, spec.mass)
        # time-scale convention installed for the risky() CM
        task = runner.env.task
        assert task._time_scale_safe == pytest.approx(runner.risk_params.safe_time_scale)
        assert task._time_scale_risky == pytest.approx(runner.risk_params.time_scale)
        assert task.active_time_scale == pytest.approx(runner.risk_params.safe_time_scale)
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_full_episode_low_risk(embodiment, tmp_path):
    runner = ItemEpisodeRunner(
        "smoke_items",
        embodiment=embodiment,
        seed=0,
        risk={"time_scale": 0.6, "safe_time_scale": 0.6},
        render="full",
        video=False,
        out_root=tmp_path,
    )
    meta, error = runner.run()
    assert error is None
    assert meta.task_success is True

    data = np.load(runner.out_dir / "trace.npz")
    assert data["margins"].shape[1] == len(EXPECTED_RULE_IDS)
    assert (data["margins"][0] > 0.0).all()  # all margins positive at t = 0
    assert len(data["t"]) > 50  # a real episode, not a stub
    # both scales are 0.6 in this run: the whole trace sits at the safe scale
    assert np.allclose(data["time_scale"], 0.6)
    # the grasped box (index 0) registers as held by the left arm at some tick
    assert (data["held"][:, 0] == 0).any()
    # the grasp brings the robot close to box_0; the floor is the gripper's
    # collision-sphere coverage of the contact patch, not the contact itself
    # (WSG bottoms at 4.26 cm on seed 0, panda below 2 cm — a missed grasp
    # would stay above the 8 cm pre-grasp hover)
    rule_ids = list(data["rule_ids"])
    j = rule_ids.index("avoid|box_0|-")
    assert data["margins"][:, j].min() < 0.05

    assert (runner.out_dir / "meta.json").exists()
    retime = json.loads((runner.out_dir / "retime.json").read_text())
    assert "plans" in retime and "v_ratio_events" in retime
    assert retime["v_ratio_events"] == []  # s = 0.6 only slows plans down
    assert (runner.out_dir / "00_start.png").exists()
    assert (runner.out_dir / "30_end.png").exists()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_full_episode_fast_risk_retime(embodiment, tmp_path):
    """time_scale 1.3 retimes the risky segment: shorter plans, trace s=1.3 window."""
    runner = ItemEpisodeRunner(
        "smoke_items",
        embodiment=embodiment,
        seed=0,
        risk={"time_scale": 1.3, "safe_time_scale": 0.6},
        render="none",
        video=False,
        out_root=tmp_path,
    )
    meta, error = runner.run()
    assert error is None
    assert meta.task_success is True
    data = np.load(runner.out_dir / "trace.npz")
    scales = np.unique(np.round(data["time_scale"], 3))
    assert 1.3 in scales  # the risky translation ran at the risky scale
    assert 0.6 in scales  # approach/grasp/retreat at the safe scale
    retime = json.loads((runner.out_dir / "retime.json").read_text())
    s_values = {round(p["s"], 3) for p in retime["plans"]}
    assert {0.6, 1.3} <= s_values
