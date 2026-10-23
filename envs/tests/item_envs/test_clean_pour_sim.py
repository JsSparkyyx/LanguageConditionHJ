"""Sim tests for T6 clean_pour (the three-bowl design of 2026-10-23, robotwin_tasks/bowl_pour.py) — GPU node only
(marked ``sim``).

- setup() builds on both embodiments: the target bowl plus the two other static 002_bowl bowls (white / blue / green,
  one colour each), the rule table is one particles_outside per non-target bowl (a capped and floored column) + the
  spill rule (recorded, never named), the primary rule is the referent's, every margin is positive at t = 0 (no bead
  in any bowl).
- the safe nominal mode (aim at the target, low and slow) keeps the NAMED bowl clean on ur5-wsg.
"""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_setup_bowls_rules_margins_positive(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner("clean_pour", embodiment=embodiment, seed=seed, render="none", out_root=tmp_path)
    try:
        runner.setup()
        task = runner.env.task
        nbs = task.neighbors
        assert len(nbs) == 2, task.episode_info
        assert {nb["color"] for nb in nbs} | {task.target_color} == {"white", "blue", "green"}
        ids = [r.rule_id for r in runner.rules]
        keep = sorted(f"particles_outside|beads|{nb['region_id']}" for nb in nbs)
        assert ids == keep + ["particles_inside|beads|bowl_src_interior+target_interior"]
        assert task.primary_rule_id in ids and task.primary_rule_id.startswith("particles_outside|")
        regions = {r.region_id: r for r in runner.regions}
        for nb in nbs:
            reg = regions[nb["region_id"]]
            assert reg.role == "keep_clean" and reg.z_cap == pytest.approx(0.012)
            assert reg.z_floor is not None and reg.z_floor < 0.0   # below the rim plane, above the interior floor
            # region frame z = 0 at the bowl's rim plane (the bowls stand at rand_pose's 0.741)
            z_rim = float(reg.world_T()[2, 3])
            assert abs(z_rim - (0.741 + float(nb["geom"]["rim_y"]))) < 0.003
        c = np.asarray(task.target.get_pose().p, dtype=np.float64)[:2]
        for nb in nbs:
            assert np.linalg.norm(np.asarray(nb["xy"]) - c) > float(task._target_geom["outer_radius"]) + nb["radius"]
        margins = runner._rule_margins()
        assert np.all(np.isfinite(margins)) and (margins > 0.0).all(), margins
    finally:
        runner.close()


@pytest.mark.parametrize("seed", [0, 1])
def test_safe_mode_keeps_the_named_bowl_clean(seed, tmp_path):
    runner = ItemEpisodeRunner("clean_pour", embodiment="ur5-wsg", seed=seed, render="none", out_root=tmp_path,
                               nominal_mode="safe")
    meta, error = runner.run()
    assert error is None
    named = meta.diag["primary_rule"]   # the referent's particles_outside (bowl_pour.primary_rule_id)
    assert named.startswith("particles_outside|") and named in meta.violations
    assert not meta.violations[named], meta.violations
