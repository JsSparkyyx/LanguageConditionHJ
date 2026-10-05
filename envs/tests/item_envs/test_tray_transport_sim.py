"""Sim tests for T5 tray_transport — GPU node only (marked ``sim``).

- setup() builds on both embodiments: the rule table is upright per
  standing block, every block stands on the tray at t = 0 and every
  margin is positive; the primary rule is the referent's upright.
- a low-risk episode (time_scale 0.5) reaches the pad with nothing toppled.
"""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_setup_items_stand_margins_positive(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner("tray_transport", embodiment=embodiment, seed=seed, render="none", out_root=tmp_path)
    try:
        runner.setup()
        task = runner.env.task
        ids = [r.rule_id for r in runner.rules]
        assert ids == [f"upright|{it['item_id']}|-" for it in task.standing]
        assert task.primary_rule_id in ids
        for it in task.standing:
            p = np.asarray(it["entity"].get_pose().p, dtype=np.float64)
            assert task._tilt_deg(it["entity"], [0.0, 0.0, 1.0]) < 3.0
            assert abs(p[2] - (0.74 + 0.006 + 0.5 * it["h"])) < 0.01  # on the tray floor
        margins = runner._rule_margins()
        assert np.all(np.isfinite(margins)) and (margins > 0.0).all(), margins
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_low_risk_episode_reaches_pad_nothing_topples(embodiment, tmp_path):
    runner = ItemEpisodeRunner("tray_transport", embodiment=embodiment, seed=0, render="none", out_root=tmp_path,
                               risk={"time_scale": 0.5, "safe_time_scale": 0.6})
    meta, error = runner.run()
    assert error is None
    assert meta.task_success is True
    assert all(v is False or v == 0 for v in meta.violations.values()), meta.violations
