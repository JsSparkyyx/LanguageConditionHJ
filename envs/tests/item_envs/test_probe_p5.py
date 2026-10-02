"""GPU test for probes/p5.py (PLAN v3 §6 P5): token-mask vs pos_mean alignment.

Runs the single-scene probe end to end (Env on smoke_items + HoloBrain
encoder) and asserts the P5 gate: the token cells with >= 50% coverage
contain the encoder token positions closest to the item centroid.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sapien")
pytest.importorskip("torch")

from main.item_envs.probes import p5


@pytest.mark.gpu
def test_p5_token_alignment_head_camera(tmp_path):
    row = p5.check_once(tmp_path, seed=0, embodiment="ur5-wsg", camera="head", keyframe=True)
    assert row["k"] >= 1, "box_0 has no token cell at >= 50% coverage in the head camera"
    # the P5 gate: the top-coverage cells contain the centroid-closest tokens
    assert row["overlap"] is not None and row["overlap"] >= 0.5, (
        f"token alignment failed: overlap={row['overlap']} k={row['k']} "
        f"top={row['top_cells']} nearest={row['nearest']}"
    )
    # and the top-coverage tokens sit closer to the item than the average token
    assert row["dist_top_mean"] < row["dist_all_mean"]


def test_p5_alignment_metrics_pure():
    import numpy as np

    mask = np.zeros(400, dtype=np.uint8)
    mask[[5, 6, 7]] = 200  # three cells above the 50% threshold
    pos = np.random.RandomState(0).rand(400, 3) * 10.0
    centroid = np.array([0.0, 0.0, 0.0])
    pos[5] = [0.01, 0.0, 0.0]  # the three closest points land on covered cells
    pos[6] = [0.02, 0.0, 0.0]
    pos[7] = [0.03, 0.0, 0.0]
    out = p5.alignment_metrics(mask, pos, centroid)
    assert out["k"] == 3
    assert out["overlap"] == 1.0
    assert sorted(out["nearest"]) == [5, 6, 7]
    # an uncovered closest point drops the overlap to 2/3
    pos[9] = [0.005, 0.0, 0.0]
    pos[7] = [5.0, 0.0, 0.0]
    out2 = p5.alignment_metrics(mask, pos, centroid)
    assert out2["overlap"] == pytest.approx(2 / 3)
    # zero coverage -> k=0, no crash
    out3 = p5.alignment_metrics(np.zeros(400, dtype=np.uint8), pos, centroid)
    assert out3["k"] == 0 and out3["overlap"] is None
