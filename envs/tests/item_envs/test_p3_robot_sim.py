"""Sim test for probe p3_robot (PLAN v3 §6 Phase 3 P3-robot): ONE push per
embodiment at time_scale 1.0 through the planner, GPU node only. The
20-seed x 3-scale sweep is the orchestrator's job.
"""

from __future__ import annotations

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")

from main.item_envs.probes import get_probe
from main.item_envs.probes._common import KeyframeWriter

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_p3_robot_single_push(embodiment, tmp_path):
    p3r = get_probe("p3_robot")
    keywriter = KeyframeWriter(tmp_path, budget=2)
    row, vis_rows = p3r._episode(
        embodiment, seed=0, time_scale=1.0,
        dist_range=p3r.DEFAULT_DIST_RANGE, vis_seed_set={0},
        max_steps=7500, keywriter=keywriter,
    )
    assert row["error"] is None
    assert row["plan_failures"] == 0
    assert row["success"] is True  # plan chain executed cleanly
    assert p3r.DEFAULT_DIST_RANGE[0] <= row["commanded_m"] <= p3r.DEFAULT_DIST_RANGE[1]
    # the box actually moved along the push direction (loose sanity — the
    # 3 cm gate is at time_scale 0.5 over 20 seeds, not a single seed at 1.0)
    assert row["achieved_m"] > 0.02
    # loose sanity only — the box coasts after the pusher stops (the p3
    # coast probe measured up to ~5.8 cm at v=0.6/μ=0.3, and the ts=1.0
    # push is fast); the real 3 cm gate lives at time_scale 0.5 x 20 seeds
    assert abs(row["achieved_m"] - row["commanded_m"]) <= 0.08
    assert abs(row["lateral_m"]) <= 0.05
    assert row["box_tilt_deg"] < 10.0  # pushed, not tipped
    # P4 hook fired: one visibility row for box_0
    assert len(vis_rows) == 1
    assert vis_rows[0]["item_id"] == "box_0"
    assert 0.0 <= vis_rows[0]["vis_max"] <= 1.0


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_p3_robot_run_and_resume(embodiment, tmp_path):
    p3r = get_probe("p3_robot")
    out = p3r.run(
        tmp_path, embodiments=[embodiment], seeds=[0], time_scales=(1.0,),
        visibility_seeds=0,
    )
    assert out["ran"] == 1
    out2 = p3r.run(
        tmp_path, embodiments=[embodiment], seeds=[0], time_scales=(1.0,),
        visibility_seeds=0,
    )
    assert out2["ran"] == 0 and out2["skipped"] == 1
    s = p3r.summarize(tmp_path)
    assert s["n_rows"] == 1
