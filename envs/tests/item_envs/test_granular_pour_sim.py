"""Sim tests for T2 granular_pour (PLAN v3 §7 T2) — GPU node only.

Marked ``sim``: setup() builds the full CEHJ Env (cuRobo controller install,
RoboTwin setup_demo, bead settle), and the episode tests run play_once end to
end. NOT run in the login-node suite.

- 20 seeds x 2 embodiments: setup() builds without UnStableError (this
  exercises the bead-skip in the task's check_stable), the rule table matches
  the T2 config contract, every margin is positive at t = 0, and the
  task-specific risk knobs land on task.risk_knobs.
- One low-risk episode per embodiment (transit 0.5, tilt_rate 1.0, pour
  pinned at 0.05 — clamped by the task to the rim-ring interlink floor
  min_pour_height = src outer radius + 0.015, ~0.07-0.08 m; centered):
  error is None, task_success (>= 80% of beads in
  the target — P2 poured everything at omega = 1 rad/s, so a failure here is
  a tuning signal, not necessarily an implementation bug), and the trace
  shows the risky/safe time-scale windows.

NOTE: these tests exercise the runner's granular_pour registry entry, its
container-region construction (region_meta kind "container" -> rim_radius /
rim_height / up_axis_local), and task.risk_knobs — the T1-side runner
integration. They are written against that post-merge interface.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]

EXPECTED_RULE_IDS = [
    "particles_inside|beads|bowl_src_interior+target_interior",
    "particles_outside|beads|zone_red",
    "particles_outside|beads|zone_blue",
]

# flat override keys — runner._merge_risk_override puts every
# non-time_scale key into risk_params.task verbatim. pour_height 0.05 is
# below the rim-ring interlink floor: the task clamps the EFFECTIVE height
# to min_pour_height (src outer radius + 0.015), so this pin also exercises
# the clamp.
LOW_RISK = {
    "time_scale": 0.5,
    "safe_time_scale": 0.6,
    "tilt_rate": 1.0,
    "pour_height": 0.05,
    "pour_offset": 0.0,
}


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", list(range(20)))
def test_setup_margins_positive_at_t0(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner(
        "granular_pour", embodiment=embodiment, seed=seed, render="none", out_root=tmp_path
    )
    try:
        runner.setup()  # raises UnStableError on a bad layout — that is the check
        assert [r.rule_id for r in runner.rules] == EXPECTED_RULE_IDS
        assert [s.item_id for s in runner.items] == ["bowl_src", "beads"]
        assert [r.region_id for r in runner.regions] == [
            "table_top",
            "bowl_src_interior",
            "target_interior",
            "zone_red",
            "zone_blue",
        ]
        beads = {s.item_id: s for s in runner.items}["beads"]
        assert beads.kind == "particle_set"
        assert len(beads.particles) == 60
        margins = runner._rule_margins()
        assert margins.shape == (len(EXPECTED_RULE_IDS),)
        assert np.all(np.isfinite(margins))
        assert (margins > 0.0).all(), f"non-positive t=0 margins: {margins}"
        # time-scale convention installed for the risky() CM
        task = runner.env.task
        assert task._time_scale_safe == pytest.approx(runner.risk_params.safe_time_scale)
        assert task._time_scale_risky == pytest.approx(runner.risk_params.time_scale)
        assert task.active_time_scale == pytest.approx(runner.risk_params.safe_time_scale)
        # sampled task knobs installed for play_once, inside the yaml ranges
        knobs = task.risk_knobs
        assert 1.0 <= knobs["tilt_rate"] <= 4.0
        assert 0.05 <= knobs["pour_height"] <= 0.20
        assert 0.0 <= knobs["pour_offset"] <= 0.05
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_full_episode_low_risk(embodiment, tmp_path):
    runner = ItemEpisodeRunner(
        "granular_pour",
        embodiment=embodiment,
        seed=0,
        risk=dict(LOW_RISK),
        render="none",
        video=False,
        out_root=tmp_path,
    )
    meta, error = runner.run()
    assert error is None
    # >= 80% of beads inside the target (check_success); P2 poured the full
    # load at omega = 1 rad/s, so treat a failure here as a knob-tuning
    # signal first (pour is 5 cm above the rim, centered).
    assert meta.task_success is True

    data = np.load(runner.out_dir / "trace.npz")
    assert list(data["rule_ids"]) == EXPECTED_RULE_IDS
    assert data["margins"].shape[1] == len(EXPECTED_RULE_IDS)
    assert (data["margins"][0] > 0.0).all()  # all margins positive at t = 0
    assert len(data["t"]) > 50  # a real episode, not a stub
    # transit ran at the risky scale 0.5, approach/grasp/retreat at 0.6;
    # the tilt steps pin active_time_scale to clip(tilt_rate / 2.0) = 0.5
    scales = np.unique(np.round(data["time_scale"], 3))
    assert 0.5 in scales
    assert 0.6 in scales
    # the grasped source bowl registers as held at some tick
    item_ids = [s.item_id for s in runner.items]
    src_col = item_ids.index("bowl_src")
    assert (data["held"][:, src_col] >= 0).any()

    assert (runner.out_dir / "meta.json").exists()
    retime = json.loads((runner.out_dir / "retime.json").read_text())
    assert "plans" in retime and "v_ratio_events" in retime
