"""Sim test for probe p1 (PLAN v3 §6 Phase 3 P1): ONE episode on one host
(place_bread_basket — layout-feasible on DIAG seeds 0-3; place_container_plate
and stack_bowls_two stay hostile and are the P1-sweep finding) per embodiment,
GPU node only. Objects mode and baseline mode each run once; the sweep itself
is the orchestrator's job.
"""

from __future__ import annotations

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")

from main.item_envs.probes import get_probe
from main.item_envs.probes._common import KeyframeWriter

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]
HOST = "place_bread_basket"


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_p1_single_episode(embodiment, tmp_path):
    p1 = get_probe("p1")
    task_cfg = p1.load_t1_cfg()
    keywriter = KeyframeWriter(tmp_path, budget=4)

    # Friendly host, but keep CEHJ scene-retry parity: some seeds can still
    # admit no on+near layout within MAX_LAYOUTS (or error), so retry seeds
    # 0-2 and require one clean episode with a full spawn.
    row = vis_rows = None
    for seed in (0, 1, 2, 3):
        row, vis_rows = p1._episode(
            HOST, embodiment, seed=seed, baseline=False, task_cfg=task_cfg,
            vis_seed_set={0, 1, 2, 3}, max_steps=7500, keywriter=keywriter,
        )
        if row["error"] is None and row["k"] >= 3:  # prefer full; k=2 partial-K is legal
            break
    assert row["error"] is None
    assert 2 <= row["k"] <= 5  # fix-2 partial-K: k >= max(2, quota)
    assert len(row["items"]) == row["k"]
    assert row["min_dist_overall"] is not None
    assert np.isfinite(row["min_dist_overall"])
    for item in row["items"]:
        assert np.isfinite(item["min_dist"])
        assert item["max_force"] >= 0.0
        assert item["cls"] in ("on_path", "near_path", "off_path")
    assert isinstance(row["success"], bool)
    # P4 hook fired for seed 0: one visibility row per candidate
    assert len(vis_rows) == row["k"]
    for vr in vis_rows:
        assert vr["n_cam"] >= 1
        assert len(vr["vis"]) == vr["n_cam"]
        assert 0.0 <= vr["vis_max"] <= 1.0
    assert keywriter.saved >= 1  # t=0 and/or end keyframes landed

    # baseline mode: no objects, success recorded
    row_b, vis_b = p1._episode(
        HOST, embodiment, seed=0, baseline=True, task_cfg=task_cfg,
        vis_seed_set=set(), max_steps=7500, keywriter=keywriter,
    )
    assert row_b["error"] is None
    assert row_b["k"] == 0
    assert row_b["items"] == []
    assert vis_b == []
    assert isinstance(row_b["success"], bool)


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_p1_run_and_resume(embodiment, tmp_path):
    """The run() driver writes rows and resumes over them (single embodiment)."""
    p1 = get_probe("p1")
    out = p1.run(
        tmp_path, embodiments=[embodiment], seeds=[0], hosts=[HOST],
        baseline=False, visibility_seeds=0,
    )
    assert out["ran"] == 1
    out2 = p1.run(
        tmp_path, embodiments=[embodiment], seeds=[0], hosts=[HOST],
        baseline=False, visibility_seeds=0,
    )
    assert out2["ran"] == 0 and out2["skipped"] == 1
    s = p1.summarize(tmp_path)
    assert s["n_rows"] == 1
