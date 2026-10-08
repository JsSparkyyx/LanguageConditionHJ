"""R2 (2026-10-07): planner_avoids flags on T1 placements and the cuRobo boxes built from them."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from main.item_envs.spawn import draw_planner_avoids, planner_avoid_boxes


def _pl(cls):
    return SimpleNamespace(cls=cls, planner_avoids=False)


def test_flags_only_on_or_near_path_and_seeded():
    rate, n = [], 0
    for seed in range(400):
        pls = [_pl("on_path"), _pl("near_path"), _pl("off_path"), _pl("near_path")]
        draw_planner_avoids(pls, 0.2, np.random.RandomState(seed))
        assert not pls[2].planner_avoids                          # off_path never
        rate += [pls[0].planner_avoids, pls[1].planner_avoids, pls[3].planner_avoids]
        again = [_pl("on_path"), _pl("near_path"), _pl("off_path"), _pl("near_path")]
        draw_planner_avoids(again, 0.2, np.random.RandomState(seed))
        assert [p.planner_avoids for p in again] == [p.planner_avoids for p in pls]   # same seed, same flags
    assert 0.15 < np.mean(rate) < 0.25


def test_boxes_for_avoided_candidates_only():
    pose = SimpleNamespace(p=np.array([0.1, -0.2, 0.8]), q=np.array([0.7071068, 0.0, 0.0, 0.7071068]))  # 90 deg yaw
    ent = SimpleNamespace(get_pose=lambda: pose)
    from main.item_envs.items import ItemSpec

    a = ItemSpec(item_id="obj_0", entity=ent, category="cup", half_extents=np.array([0.03, 0.02, 0.05]),
                 obb_center_local=np.array([0.01, 0.0, 0.05]))
    b = ItemSpec(item_id="obj_1", entity=ent, category="bottle", half_extents=np.array([0.03, 0.03, 0.1]))
    report = {"placements": [{"item_id": "obj_0", "planner_avoids": True}, {"item_id": "obj_1", "planner_avoids": False}]}
    boxes = planner_avoid_boxes([a, b], report, pad=0.03)
    assert len(boxes) == 1
    c, q, half = boxes[0]
    assert np.allclose(c, [0.1, -0.19, 0.85], atol=1e-6)            # local +x offset turns to world +y
    assert np.allclose(half, [0.06, 0.05, 0.08]) and np.allclose(q, pose.q)
