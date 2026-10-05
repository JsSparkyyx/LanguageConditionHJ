"""Sim tests for T7 gap_place — GPU node only (marked ``sim``).

- setup() builds on both embodiments: 2-3 standing blocks, the flankers
  in front of / behind the slot with the drawn clearance, the rule table is
  upright + stay per block, every margin is positive at t = 0.
- a zero-offset episode places the payload in the slot, nothing toppled.
"""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_setup_blocks_stand_margins_positive(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner("gap_place", embodiment=embodiment, seed=seed, render="none", out_root=tmp_path)
    try:
        runner.setup()
        task = runner.env.task
        ids = [r.rule_id for r in runner.rules]
        n = len(task.standing)
        assert 2 <= n <= 3
        assert ids == [f"upright|block_{k}|-" for k in range(n)] + [f"stay|block_{k}|-" for k in range(n)]
        assert task.primary_rule_id in ids
        for b in task.standing:
            p = np.asarray(b["entity"].get_pose().p, dtype=np.float64)
            assert task._tilt_deg(b["entity"], [0.0, 0.0, 1.0]) < 3.0
            assert abs(p[2] - (0.74 + 0.5 * b["h"])) < 0.01
        f0, f1 = task.standing[0], task.standing[1]
        gap = (f1["xy"][1] - 0.5 * f1["w"]) - (f0["xy"][1] + 0.5 * f0["w"])  # flankers along y
        assert gap == pytest.approx(task.payload_w + 2.0 * task.clearance, abs=1e-6)
        margins = runner._rule_margins()
        assert np.all(np.isfinite(margins)) and (margins > 0.0).all(), margins
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_zero_offset_episode_places_nothing_topples(embodiment, tmp_path):
    runner = ItemEpisodeRunner("gap_place", embodiment=embodiment, seed=0, render="none", out_root=tmp_path,
                               risk={"time_scale": 0.75, "safe_time_scale": 0.6, "place_offset": 0.0})
    meta, error = runner.run()
    assert error is None
    assert meta.task_success is True
    up = {k: v for k, v in meta.violations.items() if k.startswith("upright|")}
    assert up and not any(up.values()), meta.violations
