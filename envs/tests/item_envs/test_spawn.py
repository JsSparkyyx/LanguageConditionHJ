"""Non-sim tests for main.item_envs.spawn (PLAN v3 §7 T1 spawner).

Placement logic (_class_counts / _find_xy / _layout_plan), the settle
verdict, and the spawn_candidates driver's rejection accounting — all with
mocked corridors/keepaways and injected geometry, no sapien scene.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from main.item_envs import spawn
from main.item_envs.spawn import (
    MAX_LAYOUTS,
    Placement,
    _Geometry,
    _ObjectDraw,
    settle_verdict,
)

CFG = {
    "k": [3, 5],
    "max_on_path": 2,
    "min_on_or_near_path": 2,
    "near_path_offset": [0.02, 0.06],
    "off_path_min_dist": 0.10,
    "min_separation": 0.03,
    "static_objects": False,
}
TASK_CFG = {"objects": dict(CFG), "corridor_t": [0.22, 0.48]}
P0 = np.array([-0.30, 0.00])
P1 = np.array([0.30, 0.00])  # 60 cm corridor along x
XYLIM = np.array([[-0.48, 0.48], [-0.32, 0.28]])
GEO = _Geometry(
    center_local=np.zeros(3),
    half=np.array([0.03, 0.03, 0.05]),
    world_half=np.array([0.03, 0.03, 0.05]),
    up_local=np.array([0.0, 0.0, 1.0]),
    quat=(1.0, 0.0, 0.0, 0.0),
)
POOL = [
    {"asset": "prim_box", "category": "box", "side_xy": [0.04, 0.08], "height": [0.05, 0.15]},
    {"asset": "prim_tall_cylinder", "category": "cylinder", "radius": [0.02, 0.035], "height": [0.10, 0.18]},
]


def fake_geometry(draw):
    return GEO


def _quat_rotz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg) / 2
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def _quat_rotx(deg: float) -> np.ndarray:
    a = np.deg2rad(deg) / 2
    return np.array([np.cos(a), np.sin(a), 0.0, 0.0])


# ---------------------------------------------------------------------------
# _class_counts
# ---------------------------------------------------------------------------


def test_class_counts_quotas():
    rng = np.random.RandomState(0)
    for _ in range(200):
        k = int(rng.randint(3, 6))
        classes = spawn._class_counts(k, CFG, TASK_CFG, rng)
        assert len(classes) == k
        n_on = classes.count("on_path")
        n_near = classes.count("near_path")
        assert 1 <= n_on <= CFG["max_on_path"]
        assert n_on + n_near >= min(CFG["min_on_or_near_path"], k)
        assert set(classes) <= {"on_path", "near_path", "off_path"}


def test_class_counts_injected_on_path_count():
    cfg = dict(CFG, on_path_count=2)
    rng = np.random.RandomState(3)
    for _ in range(50):
        classes = spawn._class_counts(4, cfg, TASK_CFG, rng)
        assert classes.count("on_path") == 2


def test_class_counts_risk_knob_fallback():
    task_cfg = {
        **TASK_CFG,
        "risk": {"task": {"on_path_count": {"dist": "choice", "values": [1, 2]}}},
    }
    rng = np.random.RandomState(5)
    for _ in range(50):
        classes = spawn._class_counts(5, CFG, task_cfg, rng)
        assert classes.count("on_path") in (1, 2)


def test_class_counts_min_on_near_with_small_k():
    # k == min_on_or_near_path: every object is on- or near-path
    rng = np.random.RandomState(7)
    for _ in range(50):
        classes = spawn._class_counts(2, CFG, TASK_CFG, rng)
        assert classes.count("on_path") + classes.count("near_path") == 2


# ---------------------------------------------------------------------------
# _find_xy
# ---------------------------------------------------------------------------


def test_find_xy_on_path_on_corridor():
    rng = np.random.RandomState(1)
    for _ in range(30):
        xy, t, lat, fb = spawn._find_xy(
            "on_path", P0, P1, [], [], GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
        )
        assert xy is not None
        assert 0.22 - 1e-9 <= t <= 0.48 + 1e-9
        assert abs(xy[1]) < 1e-9  # corridor is the x axis
        np.testing.assert_allclose(xy[0], P0[0] + t * (P1[0] - P0[0]), atol=1e-9)
        assert lat == 0.0
        assert fb is False  # base window feasible -> fallback never fires


def test_find_xy_near_path_lateral_band():
    rng = np.random.RandomState(2)
    for _ in range(30):
        xy, t, lat, fb = spawn._find_xy(
            "near_path", P0, P1, [], [], GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
        )
        assert xy is not None
        assert 0.02 - 1e-9 <= abs(lat) <= 0.06 + 1e-9
        assert abs(abs(xy[1]) - abs(lat)) < 1e-9  # corridor is the x axis
        # near-path t rides the FULL corridor (near_path_t default [0.05, 0.95]),
        # not the on-path [0.22, 0.48] window
        assert 0.05 - 1e-9 <= t <= 0.95 + 1e-9
        assert fb is False


def test_find_xy_near_path_t_configurable():
    cfg = dict(CFG, near_path_t=[0.3, 0.4])
    rng = np.random.RandomState(2)
    for _ in range(10):
        xy, t, lat, fb = spawn._find_xy(
            "near_path", P0, P1, [], [], GEO.world_half, cfg, rng, [0.22, 0.48], XYLIM
        )
        assert xy is not None
        assert 0.3 - 1e-9 <= t <= 0.4 + 1e-9
        assert fb is False


def test_find_xy_off_path_min_dist():
    rng = np.random.RandomState(4)
    for _ in range(10):
        xy, t, lat, fb = spawn._find_xy(
            "off_path", P0, P1, [], [], GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
        )
        assert xy is not None
        assert spawn.ob._line_dist(xy, P0, P1) >= CFG["off_path_min_dist"]
        assert t is None and lat is None
        assert fb is False


def test_find_xy_respects_separation_and_keepaways():
    rng = np.random.RandomState(6)
    placed = [(np.array([0.0, 0.0]), 0.03)]  # blocks the corridor middle
    keepaways = [(np.array([-0.132, 0.0]), 0.03)]  # blocks t=0.22
    for _ in range(20):
        xy, t, _, _fb = spawn._find_xy(
            "on_path", P0, P1, keepaways, placed, GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
        )
        if xy is None:
            continue
        # keepaways: CEHJ KEEPAWAY_GAP (0.02); placed objects: min_separation (0.03)
        for pt, r in keepaways:
            assert np.linalg.norm(xy - pt) >= 0.03 + r + 0.02 - 1e-9
        for pxy, pr in placed:
            assert np.linalg.norm(xy - pxy) >= 0.03 + pr + CFG["min_separation"] - 1e-9


def test_find_xy_keepaway_gap_configurable():
    """The literal PLAN reading (3 cm vs keepaway circles) is available as a knob."""
    cfg = dict(CFG, keepaway_gap=0.03)
    rng = np.random.RandomState(6)
    keepaways = [(np.array([-0.132, 0.0]), 0.03)]
    xy, t, _, _fb = spawn._find_xy(
        "on_path", P0, P1, keepaways, [], GEO.world_half, cfg, rng, [0.22, 0.48], XYLIM
    )
    if xy is not None:
        assert np.linalg.norm(xy - keepaways[0][0]) >= 0.03 + 0.03 + 0.03 - 1e-9


def test_find_xy_infeasible_returns_none():
    rng = np.random.RandomState(8)
    # keepaways covering the whole table
    keepaways = [
        (np.array([x, y]), 0.10)
        for x in np.arange(-0.5, 0.51, 0.1)
        for y in np.arange(-0.35, 0.31, 0.1)
    ]
    for cls in ("on_path", "near_path", "off_path"):
        xy, t, lat, fb = spawn._find_xy(
            cls, P0, P1, keepaways, [], GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
        )
        assert xy is None and t is None and lat is None
        assert fb is False


def test_find_xy_on_path_fallback_window_fires():
    """Fix 2.1: [0.22,0.48] blocked but [0.10,0.75] has room -> fallback hit.

    Keepaways (r = 0.03) at x = -0.20..0.00 cover the on-line interval
    [-0.28, +0.08] (blocked radius 0.03 + 0.02 gap + 0.03 obj = 0.08): the
    base window x in [-0.168, -0.012] is fully blocked, while the fallback
    grid reaches x > 0.08 (t > 0.633).
    """
    keepaways = [(np.array([x, 0.0]), 0.03) for x in np.arange(-0.20, 0.01, 0.05)]
    rng = np.random.RandomState(11)
    xy, t, lat, fb = spawn._find_xy(
        "on_path", P0, P1, keepaways, [], GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
    )
    assert xy is not None
    assert fb is True
    assert lat == 0.0
    lo, hi = spawn.ON_PATH_FALLBACK_T
    assert lo - 1e-9 <= t <= hi + 1e-9
    assert t > 0.48  # outside the (blocked) base window
    for pt, r in keepaways:  # placement is genuinely feasible
        assert np.linalg.norm(xy - pt) >= 0.03 + r + 0.02 - 1e-9

    # fully blocked on-line -> fallback also fails
    hard = [(np.array([x, 0.0]), 0.10) for x in np.arange(-0.40, 0.41, 0.05)]
    xy2, t2, lat2, fb2 = spawn._find_xy(
        "on_path", P0, P1, hard, [], GEO.world_half, CFG, rng, [0.22, 0.48], XYLIM
    )
    assert xy2 is None and t2 is None and lat2 is None and fb2 is False


# ---------------------------------------------------------------------------
# _layout_plan
# ---------------------------------------------------------------------------


def test_layout_plan_constraints_and_determinism():
    # 90 cm corridor: two on-path objects (9 cm center separation) fit
    # comfortably — this test checks constraint correctness, not the
    # congestion-driven None path (covered by test_layout_plan_none_when_infeasible).
    p0, p1 = np.array([-0.45, 0.00]), np.array([0.45, 0.00])

    def run(seed):
        rng = np.random.RandomState(seed)
        return spawn._layout_plan(
            p0, p1, [], CFG, TASK_CFG, rng, POOL, ["red", "green"],
            [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM,
        )

    for seed in (0, 1, 2, 3):
        out = run(seed)
        assert out is not None
        placements, k_dropped = out
        assert k_dropped == 0  # roomy corridor: no partial-K
        assert 3 <= len(placements) <= 5
        n_on = sum(p.cls == "on_path" for p in placements)
        n_near = sum(p.cls == "near_path" for p in placements)
        assert 1 <= n_on <= 2
        assert n_on + n_near >= 2
        for i, p in enumerate(placements):
            assert p.item_id == f"obj_{i}"
            if p.cls == "off_path":
                assert spawn.ob._line_dist(p.xyz[:2], p0, p1) >= CFG["off_path_min_dist"]
            if p.cls == "near_path":
                assert 0.02 - 1e-9 <= abs(p.lateral) <= 0.06 + 1e-9
        # pairwise separation (surface-to-surface >= min_separation)
        for a, b in zip(placements, placements[1:]):
            d = np.linalg.norm(a.xyz[:2] - b.xyz[:2])
            ra = np.max(a.geo.world_half[:2])
            rb = np.max(b.geo.world_half[:2])
            assert d >= ra + rb + CFG["min_separation"] - 1e-9
        # determinism: same seed -> identical layout
        again, again_dropped = run(seed)
        assert again_dropped == 0
        assert [tuple(p.xyz) for p in placements] == [tuple(p.xyz) for p in again]
        assert [p.draw.asset for p in placements] == [p.draw.asset for p in again]


def test_layout_plan_feasible_layout_never_degrades():
    """Fix 2, no-op path: a layout feasible as drawn sets NONE of the fix flags."""
    p0, p1 = np.array([-0.45, 0.00]), np.array([0.45, 0.00])
    for seed in (0, 1, 2, 3):
        out = spawn._layout_plan(
            p0, p1, [], CFG, TASK_CFG, np.random.RandomState(seed), POOL, ["red"],
            [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM,
        )
        assert out is not None
        placements, k_dropped = out
        assert k_dropped == 0
        assert all(p.t_fallback is False for p in placements)
        assert all(p.degraded_from is None for p in placements)
        assert all(p.report_dict()["t_fallback"] is False for p in placements)
        assert all(p.report_dict()["degraded_from"] is None for p in placements)


def test_layout_plan_none_when_infeasible():
    rng = np.random.RandomState(0)
    keepaways = [
        (np.array([x, y]), 0.10)
        for x in np.arange(-0.5, 0.51, 0.1)
        for y in np.arange(-0.35, 0.31, 0.1)
    ]
    out = spawn._layout_plan(
        P0, P1, keepaways, CFG, TASK_CFG, rng, POOL, [],
        [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM,
    )
    assert out is None


# ---------------------------------------------------------------------------
# degradation chain + partial-K (fix 2.2/2.3) — scripted _find_xy / _class_counts
# ---------------------------------------------------------------------------

XY_NEAR = np.array([0.00, 0.05])
XY_OFF = np.array([0.40, 0.20])


def _scripted_find_xy(monkeypatch, near_outcomes):
    """_find_xy fake: on_path always fails; near_path follows near_outcomes
    (True -> XY_NEAR, False -> None); off_path always lands at XY_OFF."""
    calls = iter(near_outcomes)

    def fake(cls, p0, p1, keepaways, placed, world_half, cfg, rng, t_range, xylim=None, **_kw):
        if cls == "near_path" and next(calls, False):
            return XY_NEAR.copy(), 0.5, 0.05, False
        if cls == "off_path":
            return XY_OFF.copy(), None, None, False
        return None, None, None, False

    monkeypatch.setattr(spawn, "_find_xy", fake)
    return fake


def _layout_with_classes(monkeypatch, classes, quota, k):
    monkeypatch.setattr(spawn, "_class_counts", lambda k_, cfg_, task_cfg_, rng_: list(classes))
    cfg = dict(CFG, k=[k, k], min_on_or_near_path=quota)
    return spawn._layout_plan(
        P0, P1, [], cfg, TASK_CFG, np.random.RandomState(0), POOL, ["red"],
        [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM,
    )


def test_layout_degrade_on_to_near_counts_toward_quota(monkeypatch):
    """Fix 2.2: drawn-on -> near succeeds; that near counts for the quota, so a
    second object's degrade into off_path is ALLOWED (quota 1, k 2)."""
    _scripted_find_xy(monkeypatch, [True, False])  # near: 1st object lands, 2nd fails
    out = _layout_with_classes(monkeypatch, ["on_path", "on_path"], quota=1, k=2)
    assert out is not None  # None here would mean degraded-to-near did not count
    placements, k_dropped = out
    assert k_dropped == 0
    assert [p.cls for p in placements] == ["near_path", "off_path"]
    assert [p.degraded_from for p in placements] == ["on_path", "on_path"]
    assert all(p.t_fallback is False for p in placements)
    d = placements[0].report_dict()
    assert d["cls"] == "near_path" and d["degraded_from"] == "on_path"


def test_layout_degrade_near_to_off_refused_when_quota_unmet(monkeypatch):
    """Fix 2.2: near -> off is refused while the on/near quota is unmet, and the
    exhausted chain with < max(2, quota) placements returns None (no partial)."""
    _scripted_find_xy(monkeypatch, [False])  # near always fails; off would succeed
    out = _layout_with_classes(monkeypatch, ["near_path", "off_path"], quota=2, k=2)
    assert out is None  # refusal is the quota gate, not off_path infeasibility


def test_layout_partial_k_acceptance(monkeypatch):
    """Fix 2.3: two slots placed with quota met, third slot's chain exhausted ->
    stop and return the partial layout with k_dropped."""
    calls = {"off": 0}

    def fake(cls, p0, p1, keepaways, placed, world_half, cfg, rng, t_range, xylim=None, **_kw):
        if cls == "on_path":
            return np.array([-0.10, 0.0]), 0.3, 0.0, False
        if cls == "near_path":
            return XY_NEAR.copy(), 0.5, 0.05, False
        calls["off"] += 1
        return None, None, None, False

    monkeypatch.setattr(spawn, "_find_xy", fake)
    out = _layout_with_classes(
        monkeypatch, ["on_path", "near_path", "off_path", "off_path"], quota=2, k=4
    )
    assert out is not None
    placements, k_dropped = out
    assert [p.cls for p in placements] == ["on_path", "near_path"]
    assert all(p.degraded_from is None for p in placements)
    assert k_dropped == 2  # k=4 drawn, 2 placed
    assert calls["off"] == 1  # stopped at the first exhausted slot

    # same shape but quota unmet by what is placed -> None, not a partial layout
    def fake2(cls, p0, p1, keepaways, placed, world_half, cfg, rng, t_range, xylim=None, **_kw):
        if cls == "on_path":
            return np.array([-0.10, 0.0]), 0.3, 0.0, False
        return None, None, None, False

    monkeypatch.setattr(spawn, "_find_xy", fake2)
    out2 = _layout_with_classes(
        monkeypatch, ["on_path", "off_path", "off_path", "off_path"], quota=2, k=4
    )
    # slots 2-4 are drawn off_path (ungated) and fail; on+near count is 1 < 2
    assert out2 is None


# ---------------------------------------------------------------------------
# settle_verdict
# ---------------------------------------------------------------------------


def test_settle_verdict_pass():
    up = [np.array([0.0, 0.0, 1.0])]
    before = [(np.array([0.1, 0.2, 0.75]), _quat_rotz(30.0))]
    after = [(np.array([0.1 + 0.002, 0.2, 0.75]), _quat_rotz(33.0))]  # 2 mm, no tilt
    ok, details = settle_verdict(before, after, up)
    assert ok
    assert details[0]["moved_m"] == pytest.approx(0.002, abs=1e-9)
    assert abs(details[0]["tilt_deg"]) < 1e-3  # quat->mat->arccos float noise


def test_settle_verdict_reject_move():
    up = [np.array([0.0, 0.0, 1.0])]
    before = [(np.zeros(3), _quat_rotz(0.0))]
    after = [(np.array([0.006, 0.0, 0.0]), _quat_rotz(0.0))]  # 6 mm > 5 mm
    ok, details = settle_verdict(before, after, up)
    assert not ok
    assert details[0]["moved_m"] > 0.005


def test_settle_verdict_reject_tilt():
    up = [np.array([0.0, 0.0, 1.0])]
    before = [(np.zeros(3), _quat_rotz(0.0))]
    after = [(np.zeros(3), _quat_rotx(4.0))]  # 4 deg > 3 deg, no translation
    ok, details = settle_verdict(before, after, up)
    assert not ok
    assert details[0]["tilt_deg"] == pytest.approx(4.0, abs=1e-6)
    # yaw about the up axis is NOT a tilt
    ok_yaw, _ = settle_verdict(before, [(np.zeros(3), _quat_rotz(45.0))], up)
    assert ok_yaw


# ---------------------------------------------------------------------------
# spawn_candidates driver (mocked sim side)
# ---------------------------------------------------------------------------


OBJECTS_CFG = {
    "assets": {
        "prim_box": {"category": "box", "side_xy": [0.04, 0.08], "height": [0.05, 0.15]},
    },
    "t1_pool": ["prim_box"],
    "colors": {"train": ["red", "green"]},
    "color_rgb": {"red": [1, 0, 0], "green": [0, 1, 0]},
    "categories": {"box": {"mass_range": [0.05, 0.15]}},
}


def _make_placement(i=0):
    draw = _ObjectDraw(
        asset="prim_box", category="box", color="red",
        size=np.array([0.05, 0.05, 0.08]),
    )
    return Placement(
        item_id=f"obj_{i}", cls="on_path", draw=draw, geo=GEO,
        xyz=np.array([0.0, 0.0, 0.0]), corridor_t=0.3, lateral=0.0,
    )


def _fake_env():
    return SimpleNamespace(
        task_name="place_container_plate",
        task=SimpleNamespace(table_z_bias=0.0, item_meta=[]),
        robot=SimpleNamespace(),
    )


def _patch_driver(monkeypatch, poses_script):
    """Patch every sim-side seam of spawn_candidates. poses_script: per-attempt
    movers mapping the spawn xyz -> the after-settle pose (k = 1 object)."""
    calls = {"spawn_one": 0, "masses": [], "layouts": 0}
    state = {"xyz": None, "pose_calls": 0}  # per attempt: 1 before + 1 after

    monkeypatch.setattr(spawn, "resolve_arm", lambda task, spec, hint: "left")
    monkeypatch.setattr(
        spawn, "longest_corridor",
        lambda task, spec, robot, arm: (P0.copy(), P1.copy(), "left"),
    )
    monkeypatch.setattr(spawn.ob, "keepaways_from_task", lambda task, spec: [])
    monkeypatch.setattr(spawn.ob, "keepaways_from_robot", lambda env: [])
    monkeypatch.setattr(
        spawn.physics, "load_objects_cfg", lambda path=None: OBJECTS_CFG
    )
    monkeypatch.setattr(
        spawn.physics, "set_mass_props",
        lambda entity, mass: calls["masses"].append(float(mass)),
    )

    def fake_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **_kw):
        calls["layouts"] += 1
        return [_make_placement(0)], 0

    monkeypatch.setattr(spawn, "_layout_plan", fake_layout)

    def fake_spawn_one(task, placement, static, objects_cfg):
        calls["spawn_one"] += 1
        state["xyz"] = placement.xyz.copy()
        return SimpleNamespace(name=placement.item_id)

    monkeypatch.setattr(spawn, "_spawn_one", fake_spawn_one)
    monkeypatch.setattr(spawn, "_settle", lambda env, steps: None)

    def fake_pose_of(spec):
        # call 0 of an attempt = "before" (spawn pose), call 1 = "after" (scripted)
        attempt = state["pose_calls"] // 2
        is_after = state["pose_calls"] % 2 == 1
        state["pose_calls"] += 1
        xyz = state["xyz"]
        if not is_after:
            return (xyz.copy(), _quat_rotz(0.0))
        mover = poses_script[min(attempt, len(poses_script) - 1)]
        return mover(xyz)

    monkeypatch.setattr(spawn, "_pose_of", fake_pose_of)
    monkeypatch.setattr(spawn, "_remove_layout", lambda env, specs: None)
    return calls, state


def test_spawn_candidates_counts_settle_rejections(monkeypatch):
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    moved = lambda xyz: (xyz + np.array([0.02, 0.0, 0.0]), _quat_rotz(0.0))
    # attempt 0 settles bad, attempt 1 settles fine -> exactly one rejection
    _patch_driver(monkeypatch, [moved, still])
    env = _fake_env()
    rng = np.random.RandomState(0)
    specs, report = spawn.spawn_candidates(env, TASK_CFG, rng)
    assert report["spawn_rejections"] == 1
    assert report["settle_rejections"] == 1
    assert report["layout_retries"] == 0
    assert report["max_layouts_hit"] is False
    assert report["k"] == 1
    assert len(specs) == 1
    assert specs[0].item_id == "obj_0"
    assert specs[0].role == "candidate"
    assert len(env.task.item_meta) == 1
    assert env.task.item_meta[0]["item_id"] == "obj_0"
    assert len(report["placements"]) == 1
    assert report["placements"][0]["cls"] == "on_path"


def test_spawn_candidates_max_layouts(monkeypatch):
    moved = lambda xyz: (xyz + np.array([0.02, 0.0, 0.0]), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [moved])
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert report["max_layouts_hit"] is True
    assert report["settle_rejections"] == MAX_LAYOUTS
    assert report["spawn_rejections"] == MAX_LAYOUTS - 1
    assert len(specs) == 1  # last layout kept, never crashes


def test_spawn_candidates_layout_retries(monkeypatch):
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    calls, state = _patch_driver(monkeypatch, [still])
    layouts = {"n": 0}

    def flaky_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **_kw):
        layouts["n"] += 1
        if layouts["n"] == 1:
            return None  # first layout infeasible
        return [_make_placement(0)], 0

    monkeypatch.setattr(spawn, "_layout_plan", flaky_layout)
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert report["layout_retries"] == 1
    assert report["spawn_rejections"] == 1  # the failed layout counts once
    assert report["settle_rejections"] == 0
    assert len(specs) == 1


def test_spawn_candidates_fix2_report_fields(monkeypatch):
    """t_fallbacks / degraded / k_dropped describe the FINAL kept layout."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    calls, _state = _patch_driver(monkeypatch, [still])

    def flagged_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **_kw):
        p0_ = _make_placement(0)
        p0_.t_fallback = True
        p1_ = _make_placement(1)
        p1_.cls = "near_path"
        p1_.degraded_from = "on_path"
        return [p0_, p1_], 1  # k_dropped = 1

    monkeypatch.setattr(spawn, "_layout_plan", flagged_layout)
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert report["k"] == 2
    assert report["t_fallbacks"] == 1
    assert report["degraded"] == 1
    assert report["k_dropped"] == 1
    assert report["placements"][0]["t_fallback"] is True
    assert report["placements"][1]["degraded_from"] == "on_path"
    assert report["placements"][1]["t_fallback"] is False
    assert report["placements"][0]["degraded_from"] is None

    # clean layout: all counters zero
    calls2, _state2 = _patch_driver(monkeypatch, [still])
    env2 = _fake_env()
    specs2, report2 = spawn.spawn_candidates(env2, TASK_CFG, np.random.RandomState(0))
    assert report2["t_fallbacks"] == 0
    assert report2["degraded"] == 0
    assert report2["k_dropped"] == 0


def test_spawn_candidates_mass_applied(monkeypatch):
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    calls, _state = _patch_driver(monkeypatch, [still])
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert len(calls["masses"]) == 1
    assert 0.05 <= calls["masses"][0] <= 0.15  # objects.yaml category range
    assert specs[0].mass == pytest.approx(calls["masses"][0])
    assert report["static_objects"] is False


def test_spawn_candidates_static_parity_skips_mass(monkeypatch):
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    calls, _state = _patch_driver(monkeypatch, [still])
    env = _fake_env()
    cfg = {"objects": {**CFG, "static_objects": True}, "corridor_t": [0.22, 0.48]}
    spawn.spawn_candidates(env, cfg, np.random.RandomState(0))
    assert calls["masses"] == []  # no set_mass_props for static parity


def test_spawn_candidates_rejects_non_host(monkeypatch):
    env = _fake_env()
    env.task_name = "not_a_host"
    with pytest.raises(KeyError):
        spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))


# ---------------------------------------------------------------------------
# F5: off_path visibility band (visibility_band_<embodiment>.npz)
# ---------------------------------------------------------------------------

BAND_RECT = (-0.15, 0.15, -0.25, 0.25)  # (x_min, x_max, y_min, y_max), cell-center coords


def _write_band(dir_path, embodiment, rect):
    """Fabricate a band npz in the shape visibility_band.py writes."""
    dir_path = Path(dir_path)
    xs = np.arange(rect[0], rect[1] + 0.01, 0.02)
    ys = np.arange(rect[2], rect[3] + 0.01, 0.02)
    grid_xy = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    vis = np.ones((len(ys), len(xs), 3), dtype=np.float32)
    np.savez(
        dir_path / f"visibility_band_{embodiment}.npz",
        xs=xs, ys=ys, grid_xy=grid_xy, vis=vis, vis_max=vis.max(axis=2),
        covered=np.ones((len(ys), len(xs)), dtype=bool),
        band=np.ones((len(ys), len(xs)), dtype=bool),
        rect=np.asarray(rect, dtype=np.float64),
        embodiment=embodiment, cell=0.02, thresh=0.3,
    )


def test_layout_plan_off_path_vis_band(monkeypatch):
    """F5 (a): with vis_band set, every off_path placement lands inside the rect
    (and still honors off_path_min_dist); on/near placements are NOT restricted."""
    monkeypatch.setattr(
        spawn, "_class_counts",
        lambda k_, cfg_, task_cfg_, rng_: [
            "on_path", "near_path", "off_path", "off_path", "off_path"],
    )
    cfg = dict(CFG, k=[5, 5])
    n_off = 0
    for seed in range(5):
        out = spawn._layout_plan(
            P0, P1, [], cfg, TASK_CFG, np.random.RandomState(seed), POOL, ["red"],
            [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM, vis_band=BAND_RECT,
        )
        assert out is not None
        placements, k_dropped = out
        assert k_dropped == 0
        for p in placements:
            if p.cls != "off_path":
                continue
            n_off += 1
            assert BAND_RECT[0] - 1e-9 <= p.xyz[0] <= BAND_RECT[1] + 1e-9
            assert BAND_RECT[2] - 1e-9 <= p.xyz[1] <= BAND_RECT[3] + 1e-9
            assert spawn.ob._line_dist(p.xyz[:2], P0, P1) >= CFG["off_path_min_dist"]
    assert n_off == 15  # 3 off_path draws x 5 seeds, all placed


def test_layout_plan_vis_band_none_byte_identical():
    """F5 (b), layout level: vis_band=None (flag off / npz missing) reproduces
    the unrestricted layout exactly, same seed."""
    def run(seed, **kw):
        return spawn._layout_plan(
            P0, P1, [], CFG, TASK_CFG, np.random.RandomState(seed), POOL, ["red"],
            [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM, **kw,
        )

    for seed in (0, 1, 2, 3):
        base, base_kd = run(seed)
        explicit, explicit_kd = run(seed, vis_band=None)
        assert base_kd == explicit_kd
        assert [tuple(p.xyz) for p in base] == [tuple(p.xyz) for p in explicit]
        assert [(p.cls, p.corridor_t, p.lateral) for p in base] == [
            (p.cls, p.corridor_t, p.lateral) for p in explicit
        ]


def test_vis_band_for_resolution(monkeypatch, tmp_path):
    """F5 (c), unit: own embodiment's npz wins; otherwise the other embodiment's
    band resolves as 'shared-fallback'; no file at all -> (None, None)."""
    other_rect = (-0.20, 0.20, -0.22, 0.22)
    _write_band(tmp_path, "franka-panda", other_rect)
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)

    rect, label = spawn._vis_band_for("franka-panda")
    assert rect == pytest.approx(other_rect)
    assert label == "franka-panda"

    rect2, label2 = spawn._vis_band_for("ur5-wsg")  # own file missing -> fallback
    assert rect2 == pytest.approx(other_rect)
    assert label2 == "shared-fallback"

    rect3, label3 = spawn._vis_band_for(None)  # no embodiment at all -> fallback
    assert rect3 == pytest.approx(other_rect)
    assert label3 == "shared-fallback"

    # own file takes precedence over the fallback
    _write_band(tmp_path, "ur5-wsg", BAND_RECT)
    rect4, label4 = spawn._vis_band_for("ur5-wsg")
    assert rect4 == pytest.approx(BAND_RECT)
    assert label4 == "ur5-wsg"

    # empty configs dir -> no restriction
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", empty)
    assert spawn._vis_band_for("ur5-wsg") == (None, None)


def test_spawn_candidates_vis_band_threading(monkeypatch, tmp_path):
    """F5 (b/c), driver level: the flag loads the band once per spawn_candidates
    call, threads it into _layout_plan, and labels the report; a missing npz
    falls back to vis_band=None with the old behavior."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still])
    _write_band(tmp_path, "franka-panda", BAND_RECT)
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)
    seen = {}

    def capture_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **kw):
        seen.update(kw)
        return [_make_placement(0)], 0

    monkeypatch.setattr(spawn, "_layout_plan", capture_layout)
    flag_cfg = {"objects": {**CFG, "off_path_visible_only": True}, "corridor_t": [0.22, 0.48]}

    # own band missing, franka's present -> shared-fallback, rect threaded through
    env = _fake_env()
    env.embodiment = "ur5-wsg"
    specs, report = spawn.spawn_candidates(env, flag_cfg, np.random.RandomState(0))
    assert report["vis_band"] == "shared-fallback"
    assert seen["vis_band"] == pytest.approx(BAND_RECT)

    # no band files anywhere -> vis_band None threaded, report None (old behavior)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", empty)
    seen.clear()
    env2 = _fake_env()
    env2.embodiment = "ur5-wsg"
    specs2, report2 = spawn.spawn_candidates(env2, flag_cfg, np.random.RandomState(0))
    assert report2["vis_band"] is None
    assert seen["vis_band"] is None

    # flag OFF: band never loaded even though files exist
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)
    seen.clear()
    env3 = _fake_env()
    env3.embodiment = "ur5-wsg"
    specs3, report3 = spawn.spawn_candidates(env3, TASK_CFG, np.random.RandomState(0))
    assert report3["vis_band"] is None
    assert seen["vis_band"] is None


def test_spawn_candidates_vis_band_report_default(monkeypatch):
    """F5 (d): the report always carries vis_band (None when the flag is off)."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still])
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert "vis_band" in report
    assert report["vis_band"] is None
