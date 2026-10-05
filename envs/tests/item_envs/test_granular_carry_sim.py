"""Sim tests for T4 granular_carry — GPU node only (marked ``sim``).

- Both modes x 2 embodiments x 3 seeds: setup() (incl. the held-mode
  pre-roll ``prepare_episode``) builds, the rule table is the T4 contract,
  the bead set is non-empty and every margin is positive at t = 0 (the
  pre-roll deletes every bead that left the bowl, so a held episode starts
  clean), and the raised-floor bowl is in place.
- held mode: at t = 0 the bowl is lifted off the table and level (the
  pre-roll grasped, lifted and levelled it).
"""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]
EXPECTED_RULE_IDS = ["particles_inside|beads|bowl_src_interior"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("mode", ["table", "held"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_setup_margins_positive_at_t0(embodiment, mode, seed, tmp_path, monkeypatch):
    monkeypatch.setenv("ITEM_T4_MODE", mode)
    runner = ItemEpisodeRunner("granular_carry", embodiment=embodiment, seed=seed, render="none",
                               out_root=tmp_path)
    try:
        runner.setup()
        task = runner.env.task
        assert task.carry_mode == mode
        assert [r.rule_id for r in runner.rules] == EXPECTED_RULE_IDS
        beads = {s.item_id: s for s in runner.items}["beads"]
        assert beads.kind == "particle_set" and len(beads.particles) > 20
        assert beads.particles is task.beads  # shared list (the pre-roll edits it in place)
        assert task._src_geom.get("raised_floor", 0.0) > 0.0  # low bowl (user review 5)
        margins = runner._rule_margins()
        assert np.all(np.isfinite(margins)) and (margins > 0.0).all(), margins
        if mode == "held":
            # held, not a fallback (a 3 cm jaw made all 3 ur5 seeds fall back,
            # audit3), and off the table by the task's own held criterion,
            # 1.5 mm (a loaded bowl sags 2-5 mm in the pinch: 2.9 mm on ur5
            # seed 1, t6c)
            assert not task.episode_info.get("held_fallback"), task.episode_info
            assert float(task.container.get_pose().p[2]) > 0.74 + 0.0015
            assert task._bowl_tilt_deg() < 10.0  # levelled
    finally:
        runner.close()
