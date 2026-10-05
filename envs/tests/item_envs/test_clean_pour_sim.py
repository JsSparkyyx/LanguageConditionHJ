"""Sim tests for T6 clean_pour — GPU node only (marked ``sim``).

- setup() builds on both embodiments: 2-3 static neighbours around the
  target, the rule table is one particles_outside per neighbour (capped
  column) + T2's spill rule, the primary rule is the referent's, every
  margin is positive at t = 0 (no bead in any neighbour).
- an aim_error 0 episode keeps every neighbour clean.
"""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_setup_neighbours_rules_margins_positive(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner("clean_pour", embodiment=embodiment, seed=seed, render="none", out_root=tmp_path)
    try:
        runner.setup()
        task = runner.env.task
        nbs = task.neighbors
        assert 2 <= len(nbs) <= 3, task.episode_info
        ids = [r.rule_id for r in runner.rules]
        keep = sorted(f"particles_outside|beads|{nb['region_id']}" for nb in nbs)
        assert ids == keep + ["particles_inside|beads|bowl_src_interior+target_interior"]
        assert task.primary_rule_id in ids
        regions = {r.region_id: r for r in runner.regions}
        for nb in nbs:
            reg = regions[nb["region_id"]]
            assert reg.role == "keep_clean" and reg.z_cap == pytest.approx(0.012)
            # region frame z = 0 at the rim plane, on the table
            z_rim = float(reg.world_T()[2, 3])
            assert abs(z_rim - (0.74 + nb["rim_h"])) < 0.003
        c = np.asarray(task.target.get_pose().p, dtype=np.float64)[:2]
        for nb in nbs:
            assert np.linalg.norm(np.asarray(nb["xy"]) - c) > float(task._target_geom["outer_radius"])
        margins = runner._rule_margins()
        assert np.all(np.isfinite(margins)) and (margins > 0.0).all(), margins
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_zero_aim_error_keeps_neighbours_clean(embodiment, tmp_path):
    runner = ItemEpisodeRunner("clean_pour", embodiment=embodiment, seed=0, render="none", out_root=tmp_path,
                               risk={"time_scale": 0.75, "safe_time_scale": 0.6,
                                     "aim_error": 0.0, "tilt_rate": 1.5, "pour_height": 0.08})
    meta, error = runner.run()
    assert error is None
    keep = {k: v for k, v in meta.violations.items() if k.startswith("particles_outside|")}
    assert keep and not any(keep.values()), meta.violations
