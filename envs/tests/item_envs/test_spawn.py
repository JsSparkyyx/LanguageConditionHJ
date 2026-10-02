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


def _patch_driver(monkeypatch, poses_script, real_layout=False):
    """Patch every sim-side seam of spawn_candidates. poses_script: per-attempt
    movers mapping the spawn xyz -> the after-settle pose (k = 1 object).
    real_layout=True keeps the genuine _layout_plan (F5b end-to-end tests;
    the OBJECTS_CFG pool is prim_box only, so geometry stays trimesh-free)."""
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

    if not real_layout:
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


def _write_band(dir_path, embodiment, rect, covered=None):
    """Fabricate a band npz in the shape visibility_band.py writes.

    covered: optional bool [ny, nx] mask (F5b tests craft partial coverage);
    default all-True (every cell covered, as the F5 tests assume).
    """
    dir_path = Path(dir_path)
    xs = np.arange(rect[0], rect[1] + 0.01, 0.02)
    ys = np.arange(rect[2], rect[3] + 0.01, 0.02)
    grid_xy = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    vis = np.ones((len(ys), len(xs), 3), dtype=np.float32)
    if covered is None:
        covered = np.ones((len(ys), len(xs)), dtype=bool)
    np.savez(
        dir_path / f"visibility_band_{embodiment}.npz",
        xs=xs, ys=ys, grid_xy=grid_xy, vis=vis, vis_max=vis.max(axis=2),
        covered=np.asarray(covered, dtype=bool),
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


# ---------------------------------------------------------------------------
# fix 4: measured mesh AABB (bottom-rest z) + arm-link keepaways
# ---------------------------------------------------------------------------


def _mesh_draw(asset="068_boxdrink", model_id=0):
    return _ObjectDraw(asset=asset, category="drink_box", model_id=model_id)


def test_geometry_mesh_measured_aabb(monkeypatch):
    """_geometry_of uses the measured collision AABB, not the json metadata."""
    lo = np.array([-0.02, 0.00, -0.03])
    hi = np.array([0.04, 0.10, 0.05])  # bottom-origin along +y -> up axis 1
    monkeypatch.setattr(spawn, "_mesh_aabb", lambda a, i: (lo.copy(), hi.copy()))
    geo = spawn._geometry_of(_mesh_draw())
    np.testing.assert_allclose(geo.center_local, [0.01, 0.05, 0.01])
    np.testing.assert_allclose(geo.half, [0.03, 0.05, 0.04])
    # dominant |center| axis is y -> up=1: obstacle.py's _STAND_R permutation
    assert geo.quat == spawn.ob.OBSTACLE_STAND_QUAT
    np.testing.assert_allclose(geo.world_half, np.abs(spawn.ob._STAND_R) @ geo.half)
    np.testing.assert_allclose(geo.up_local, spawn.ob._STAND_R.T @ np.array([0.0, 0.0, 1.0]))


def test_geometry_mesh_bottom_rest_z(monkeypatch):
    """Offset-origin asset: the z-pack rests the AABB BOTTOM at table + 0.002,
    and the packed bottom equals the rotated mesh min-z (not the origin)."""
    lo = np.array([-0.02, 0.00, -0.03])
    hi = np.array([0.04, 0.10, 0.05])
    monkeypatch.setattr(spawn, "_mesh_aabb", lambda a, i: (lo.copy(), hi.copy()))
    geo = spawn._geometry_of(_mesh_draw())
    table_z = 0.75
    xyz_z = table_z + float(geo.world_half[2]) + 0.002  # spawn_candidates' pack
    r = spawn._quat_wxyz_to_mat(np.asarray(geo.quat))  # _STAND_R (up=1 here)
    pose_z = xyz_z - float((r @ geo.center_local)[2])  # _spawn_one compensation
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    bottom_local = float((corners @ r.T)[:, 2].min())  # world min-z rel. pose origin
    assert pose_z + bottom_local == pytest.approx(table_z + 0.002, abs=1e-9)
    # the asset origin is NOT at the AABB center along z — the compensation matters
    assert abs(bottom_local + geo.world_half[2]) > 1e-3


def test_geometry_mesh_json_fallback(monkeypatch):
    """Mesh unloadable -> the old json-metadata path (obstacle.py semantics)."""
    monkeypatch.setattr(spawn, "_mesh_aabb", lambda a, i: None)
    cfg = {"center": [0.0, 0.05, 0.0], "extents": [0.06, 0.10, 0.08], "scale": [1, 1, 1]}
    monkeypatch.setattr(spawn, "_load_model_cfg_local", lambda a, i: dict(cfg))
    geo = spawn._geometry_of(_mesh_draw())
    np.testing.assert_allclose(geo.center_local, [0.0, 0.05, 0.0])
    np.testing.assert_allclose(geo.half, [0.03, 0.05, 0.04])


def test_geometry_prim_bottom_rest_z():
    """Prims are center-origin: bottom rests at table + 0.002 with no compensation."""
    draw = _ObjectDraw(asset="prim_box", category="box", color="red",
                       size=np.array([0.05, 0.06, 0.10]))
    geo = spawn._geometry_of(draw)
    table_z = 0.75
    xyz_z = table_z + float(geo.world_half[2]) + 0.002
    pose_z = xyz_z - float(geo.center_local[2])  # prims: identity quat, zero center
    assert pose_z - float(geo.world_half[2]) == pytest.approx(table_z + 0.002, abs=1e-9)


def test_draw_object_cap_uses_geometry_fn():
    """The on/near size cap reads geometry_fn(draw).world_half (measured),
    not the json globals — same geometry source as the placement itself."""
    pool = [{"asset": "068_boxdrink", "category": "drink_box", "ids": [0]}]

    def geo_with(r):
        return _Geometry(center_local=np.zeros(3), half=np.full(3, r),
                         world_half=np.full(3, r), up_local=np.array([0.0, 0.0, 1.0]),
                         quat=(1.0, 0.0, 0.0, 0.0))

    draw = spawn._draw_object(pool, np.random.RandomState(0), [], max_r_xy=0.035,
                              geometry_fn=lambda d: geo_with(0.03))
    assert draw is not None
    draw2 = spawn._draw_object(pool, np.random.RandomState(0), [], max_r_xy=0.035,
                               geometry_fn=lambda d: geo_with(0.05))
    assert draw2 is None  # 5 cm radius exceeds the 3.5 cm cap on every retry


def _link(x, y, z):
    return SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.array([x, y, z])))


def test_arm_link_keepaways_filter():
    """Only low links over the (padded) table become circles; knobs retune."""
    env = SimpleNamespace(robot=SimpleNamespace(
        left_entity=SimpleNamespace(get_links=lambda: [
            _link(0.30, -0.20, 0.90),   # over the table, low -> keepaway
            _link(0.30, -0.20, 1.20),   # too high -> excluded
            _link(0.90, -0.20, 0.90),   # outside the padded table xy -> excluded
        ]),
        right_entity=SimpleNamespace(get_links=lambda: [_link(-0.30, -0.20, 0.88)]),
    ))
    out = spawn._arm_link_keepaways(env, 0.74, {})
    assert len(out) == 2
    assert out[0][1] == pytest.approx(spawn.ARM_LINK_KEEPAWAY_R)
    np.testing.assert_allclose(out[0][0], [0.30, -0.20])
    np.testing.assert_allclose(out[1][0], [-0.30, -0.20])
    # a broken/headless robot yields no circles and never raises
    assert spawn._arm_link_keepaways(SimpleNamespace(robot=SimpleNamespace()), 0.74, {}) == []
    # knobs: higher z_max admits the 1.20 m link; custom radius
    out2 = spawn._arm_link_keepaways(env, 0.74, {"arm_link_z_max": 0.60,
                                                 "arm_link_keepaway_r": 0.05})
    assert len(out2) == 3
    assert out2[0][1] == pytest.approx(0.05)


def test_spawn_candidates_arm_link_keepaways(monkeypatch):
    """Franka-style overhanging links reach the keepaways _layout_plan sees;
    objects.arm_link_keepaways: false opts out."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still])
    seen = {}

    def capture_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **kw):
        seen["keepaways"] = list(keepaways)
        return [_make_placement(0)], 0

    monkeypatch.setattr(spawn, "_layout_plan", capture_layout)

    def env_with_links():
        env = _fake_env()
        env.robot = SimpleNamespace(
            left_entity=SimpleNamespace(get_links=lambda: [_link(0.30, -0.20, 0.90)]),
            right_entity=SimpleNamespace(get_links=lambda: []),
        )
        return env

    specs, report = spawn.spawn_candidates(env_with_links(), TASK_CFG, np.random.RandomState(0))
    assert report["arm_link_keepaways"] == 1
    assert any(np.allclose(xy, [0.30, -0.20]) for xy, _r in seen["keepaways"])

    seen.clear()
    cfg_off = {"objects": {**CFG, "arm_link_keepaways": False}, "corridor_t": [0.22, 0.48]}
    specs2, report2 = spawn.spawn_candidates(env_with_links(), cfg_off, np.random.RandomState(0))
    assert report2["arm_link_keepaways"] == 0
    assert not any(np.allclose(xy, [0.30, -0.20]) for xy, _r in seen["keepaways"])


# ---------------------------------------------------------------------------
# fix 4, real assets (login-node: trimesh only; skips when the tree is absent)
# ---------------------------------------------------------------------------


def _assets_tree():
    for root in spawn._objects_roots():
        if Path(root).is_dir():
            return Path(root)
    return None


@pytest.mark.skipif(_assets_tree() is None, reason="RoboTwin asset tree not present")
def test_real_assets_bottom_rest_z():
    """Every T1 pool mesh id: the measured AABB + z-pack rest the world AABB
    bottom within [0, 4 mm] of the table — for any asset origin convention."""
    objects_cfg = spawn.physics.load_objects_cfg()
    pool = [e for e in spawn.t1_pool_entries(objects_cfg)
            if not str(e["asset"]).startswith("prim_")]
    table_z = 0.75
    checked = 0
    for entry in pool:
        for model_id in entry.get("ids") or [0]:
            draw = _ObjectDraw(asset=entry["asset"], category=entry["category"],
                               model_id=int(model_id))
            aabb = spawn._mesh_aabb(draw.asset, draw.model_id)
            assert aabb is not None, f"{draw.asset} id {model_id}: collision mesh unloadable"
            lo, hi = aabb
            geo = spawn._geometry_of(draw)
            r = spawn._quat_wxyz_to_mat(np.asarray(geo.quat))  # geometry's own quat
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                              for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            bottom_local = float((corners @ r.T)[:, 2].min())
            xyz_z = table_z + float(geo.world_half[2]) + 0.002  # the pack
            pose_z = xyz_z - float((r @ geo.center_local)[2])   # _spawn_one
            offset = pose_z + bottom_local - table_z
            assert 0.0 <= offset <= 0.004, (
                f"{draw.asset} id {model_id}: AABB bottom at {offset:+.4f} m rel. table")
            checked += 1
    assert checked >= 70  # the full 79-id pool (guard against a silently empty pool)


# ---------------------------------------------------------------------------
# F5b: near/on_path visibility coverage (covered mask in visibility_band npz)
# ---------------------------------------------------------------------------

COV_XS = np.arange(-0.48, 0.481, 0.02)
COV_YS = np.arange(-0.32, 0.281, 0.02)


def _vis_cov(covered):
    return (COV_XS, COV_YS, np.asarray(covered, dtype=bool))


def _mask_pos_y():
    """Covered = all cells with y > 0.005 (the +y half of the table)."""
    return np.broadcast_to(COV_YS[:, None] > 0.005, (len(COV_YS), len(COV_XS))).copy()


def test_covered_at_nearest_cell():
    cov = _vis_cov(_mask_pos_y())
    assert spawn._covered_at(cov, np.array([0.10, 0.03])) is True
    assert spawn._covered_at(cov, np.array([0.10, -0.03])) is False
    # between-cell points snap to the nearest cell center
    assert spawn._covered_at(cov, np.array([0.10, 0.011])) is True   # -> 0.02 row
    assert spawn._covered_at(cov, np.array([0.10, 0.009])) is False  # -> ~0.00 row


def test_find_xy_near_path_covered_only():
    """F5b (a): with near_path_visible_only, near placements land only in
    covered cells — the +y mask forces sign=+1 draws, no fallback needed."""
    cfg = dict(CFG, near_path_visible_only=True)
    vis_cov = _vis_cov(_mask_pos_y())
    rng = np.random.RandomState(2)
    for _ in range(30):
        vis_out = {}
        xy, t, lat, fb = spawn._find_xy(
            "near_path", P0, P1, [], [], GEO.world_half, cfg, rng, [0.22, 0.48],
            XYLIM, vis_cov=vis_cov, vis_out=vis_out,
        )
        assert xy is not None
        assert xy[1] > 0.005  # only +y cells are covered
        assert lat > 0.0
        assert spawn._covered_at(vis_cov, xy)
        assert vis_out == {}  # covered pass succeeded -> no fallback
        assert fb is False


def test_find_xy_on_path_covered_segment():
    """F5b (b): the base window is uncovered, a fallback-window segment is
    covered -> the covered pass lands there (t_fallback=True, no vis fallback)."""
    # cover only corridor cells with x in [0.06, 0.18] (t in [0.6, 0.8])
    covered = (np.abs(COV_YS[:, None]) <= 0.005) & (COV_XS[None, :] >= 0.06) & (COV_XS[None, :] <= 0.18)
    vis_cov = _vis_cov(covered)
    cfg = dict(CFG, on_path_visible_only=True)
    rng = np.random.RandomState(3)
    for _ in range(10):
        vis_out = {}
        xy, t, lat, fb = spawn._find_xy(
            "on_path", P0, P1, [], [], GEO.world_half, cfg, rng, [0.22, 0.48],
            XYLIM, vis_cov=vis_cov, vis_out=vis_out,
        )
        assert xy is not None
        assert spawn._covered_at(vis_cov, xy)
        assert fb is True  # base window [0.22,0.48] (x <= -0.012) is uncovered
        assert vis_out == {}
        lo, hi = spawn.ON_PATH_FALLBACK_T
        assert lo - 1e-9 <= t <= hi + 1e-9


def test_find_xy_vis_fallback_when_no_covered_cell():
    """F5b (c): no covered cell anywhere -> each class falls back to the
    ORIGINAL logic and flags vis_fallback. on_path's covered pass consumes no
    rng, so its fallback placement is byte-identical to the no-vis_cov run."""
    vis_cov = _vis_cov(np.zeros((len(COV_YS), len(COV_XS)), dtype=bool))

    cfg_near = dict(CFG, near_path_visible_only=True)
    rng = np.random.RandomState(4)
    vis_out = {}
    xy, t, lat, fb = spawn._find_xy(
        "near_path", P0, P1, [], [], GEO.world_half, cfg_near, rng, [0.22, 0.48],
        XYLIM, vis_cov=vis_cov, vis_out=vis_out,
    )
    assert xy is not None  # the old unrestricted loop still places
    assert vis_out == {"vis_fallback": True}

    cfg_on = dict(CFG, on_path_visible_only=True)
    for seed in (5, 6, 7):
        vis_out = {}
        xy, t, lat, fb = spawn._find_xy(
            "on_path", P0, P1, [], [], GEO.world_half, cfg_on,
            np.random.RandomState(seed), [0.22, 0.48], XYLIM,
            vis_cov=vis_cov, vis_out=vis_out,
        )
        # flag off, same seed: identical placement (covered pass is draw-free)
        xy0, t0_, lat0, fb0 = spawn._find_xy(
            "on_path", P0, P1, [], [], GEO.world_half, CFG,
            np.random.RandomState(seed), [0.22, 0.48], XYLIM,
        )
        assert vis_out == {"vis_fallback": True}
        np.testing.assert_allclose(xy, xy0, atol=0.0)
        assert (t, lat, fb) == (t0_, lat0, fb0)


def test_layout_plan_vis_cov_none_byte_identical():
    """F5b (d), layout level: class flags ON but vis_cov=None (npz missing)
    reproduces the flags-off layout exactly, same seed."""
    cfg_flags = dict(CFG, near_path_visible_only=True, on_path_visible_only=True)

    def run(cfg, seed, **kw):
        return spawn._layout_plan(
            P0, P1, [], cfg, TASK_CFG, np.random.RandomState(seed), POOL, ["red"],
            [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM, **kw,
        )

    for seed in (0, 1, 2, 3):
        base, base_kd = run(CFG, seed)
        flagged, flagged_kd = run(cfg_flags, seed, vis_cov=None)
        assert base_kd == flagged_kd == 0
        assert [tuple(p.xyz) for p in base] == [tuple(p.xyz) for p in flagged]
        assert [(p.cls, p.corridor_t, p.lateral) for p in base] == [
            (p.cls, p.corridor_t, p.lateral) for p in flagged
        ]
        assert all(p.vis_fallback is False for p in flagged)


def test_layout_plan_vis_cov_determinism_and_coverage():
    """F5b (a/e), layout level: +y-only mask -> every near_path placement is
    covered (y > 0); same seed twice -> identical layout."""
    cfg = dict(CFG, near_path_visible_only=True)
    vis_cov = _vis_cov(_mask_pos_y())

    def run(seed):
        return spawn._layout_plan(
            P0, P1, [], cfg, TASK_CFG, np.random.RandomState(seed), POOL, ["red"],
            [0.22, 0.48], geometry_fn=fake_geometry, xylim=XYLIM, vis_cov=vis_cov,
        )

    n_near = 0
    for seed in (0, 1, 2, 3):
        out = run(seed)
        assert out is not None
        placements, k_dropped = out
        assert k_dropped == 0
        for p in placements:
            if p.cls == "near_path":
                n_near += 1
                assert spawn._covered_at(vis_cov, p.xyz[:2])
                assert p.xyz[1] > 0.005
                assert p.vis_fallback is False
        again, again_kd = run(seed)
        assert again_kd == k_dropped
        assert [tuple(p.xyz) for p in placements] == [tuple(p.xyz) for p in again]
        assert [p.vis_fallback for p in placements] == [p.vis_fallback for p in again]
    assert n_near > 0  # the draw actually produced near_path objects


def test_vis_covered_for_resolution(monkeypatch, tmp_path):
    """F5b (g), unit: own embodiment's npz wins; otherwise the other
    embodiment's mask resolves as 'shared-fallback'; no file -> (None, None)."""
    rect = (-0.20, 0.20, -0.22, 0.22)
    _write_band(tmp_path, "franka-panda", rect)
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)

    cov, label = spawn._vis_covered_for("franka-panda")
    assert label == "franka-panda"
    xs, ys, covered = cov
    np.testing.assert_allclose(xs, np.arange(rect[0], rect[1] + 0.01, 0.02))
    np.testing.assert_allclose(ys, np.arange(rect[2], rect[3] + 0.01, 0.02))
    assert covered.shape == (len(ys), len(xs)) and covered.dtype == bool
    assert covered.all()  # _write_band's default mask

    cov2, label2 = spawn._vis_covered_for("ur5-wsg")  # own file missing
    assert label2 == "shared-fallback"
    assert cov2[2].shape == covered.shape

    _, label3 = spawn._vis_covered_for(None)
    assert label3 == "shared-fallback"

    # partial mask round-trips (F5b tests rely on fabricated partial coverage)
    partial = np.zeros_like(covered)
    partial[3, 5] = True
    _write_band(tmp_path, "ur5-wsg", rect, covered=partial)
    cov4, label4 = spawn._vis_covered_for("ur5-wsg")
    assert label4 == "ur5-wsg"
    assert cov4[2].sum() == 1 and cov4[2][3, 5]

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", empty)
    assert spawn._vis_covered_for("ur5-wsg") == (None, None)


def test_spawn_candidates_vis_cov_threading(monkeypatch, tmp_path):
    """F5b (f), driver level: either class flag loads the mask once per
    spawn_candidates call, threads it into _layout_plan, and labels the
    report; a missing npz falls back to vis_cov=None (old behavior)."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still])
    _write_band(tmp_path, "franka-panda", BAND_RECT)
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)
    seen = {}

    def capture_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **kw):
        seen.update(kw)
        return [_make_placement(0)], 0

    monkeypatch.setattr(spawn, "_layout_plan", capture_layout)
    near_cfg = {"objects": {**CFG, "near_path_visible_only": True}, "corridor_t": [0.22, 0.48]}
    on_cfg = {"objects": {**CFG, "on_path_visible_only": True}, "corridor_t": [0.22, 0.48]}

    # own mask missing, franka's present -> shared-fallback, mask threaded
    env = _fake_env()
    env.embodiment = "ur5-wsg"
    specs, report = spawn.spawn_candidates(env, near_cfg, np.random.RandomState(0))
    assert report["vis_cov"] == "shared-fallback"
    assert seen["vis_cov"] is not None and seen["vis_cov"][2].all()
    assert report["near_path_vis_fallbacks"] == 0
    assert report["on_path_vis_fallbacks"] == 0

    # on_path flag alone also loads the mask
    seen.clear()
    env1 = _fake_env()
    env1.embodiment = "ur5-wsg"
    specs1, report1 = spawn.spawn_candidates(env1, on_cfg, np.random.RandomState(0))
    assert report1["vis_cov"] == "shared-fallback"
    assert seen["vis_cov"] is not None

    # no npz anywhere -> None threaded, report None (old behavior)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", empty)
    seen.clear()
    env2 = _fake_env()
    env2.embodiment = "ur5-wsg"
    specs2, report2 = spawn.spawn_candidates(env2, near_cfg, np.random.RandomState(0))
    assert report2["vis_cov"] is None
    assert seen["vis_cov"] is None

    # flags OFF: mask never loaded even though files exist
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)
    seen.clear()
    env3 = _fake_env()
    env3.embodiment = "ur5-wsg"
    specs3, report3 = spawn.spawn_candidates(env3, TASK_CFG, np.random.RandomState(0))
    assert report3["vis_cov"] is None
    assert seen["vis_cov"] is None


def test_spawn_candidates_vis_cov_report_default(monkeypatch):
    """F5b (f): the report always carries the F5b keys (None/0 when flags off)."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still])
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert report["vis_cov"] is None
    assert report["near_path_vis_fallbacks"] == 0
    assert report["on_path_vis_fallbacks"] == 0
    assert report["placements"][0]["vis_fallback"] is False


def test_spawn_candidates_vis_fallback_counters(monkeypatch):
    """F5b (c/f), driver level: the counters describe the FINAL kept layout,
    split by final class (a degraded on->near vis fallback counts as near)."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still])

    def flagged_layout(p0, p1, keepaways, cfg, task_cfg, rng, pool, colors, t_range, **_kw):
        p_near = _make_placement(0)
        p_near.cls = "near_path"
        p_near.vis_fallback = True
        p_on = _make_placement(1)
        p_on.cls = "on_path"
        p_on.vis_fallback = True
        p_ok = _make_placement(2)
        p_ok.cls = "near_path"
        return [p_near, p_on, p_ok], 0

    monkeypatch.setattr(spawn, "_layout_plan", flagged_layout)
    env = _fake_env()
    specs, report = spawn.spawn_candidates(env, TASK_CFG, np.random.RandomState(0))
    assert report["near_path_vis_fallbacks"] == 1
    assert report["on_path_vis_fallbacks"] == 1
    flags = [p["vis_fallback"] for p in report["placements"]]
    assert flags == [True, True, False]


def test_spawn_candidates_f5b_end_to_end(monkeypatch, tmp_path):
    """F5b (a/b/d), driver + real layout: +y-only mask in the embodiment npz
    restricts every near_path placement to covered cells; flag-off and
    missing-npz runs are byte-identical; same seed -> same layout."""
    still = lambda xyz: (xyz.copy(), _quat_rotz(0.0))
    _patch_driver(monkeypatch, [still], real_layout=True)
    # table-wide grid, +y half covered (spawn_candidates uses ob.TABLE_XYLIM)
    rect = (-0.48, 0.48, -0.32, 0.28)
    xs = np.arange(rect[0], rect[1] + 0.01, 0.02)
    ys = np.arange(rect[2], rect[3] + 0.01, 0.02)
    covered = np.broadcast_to(ys[:, None] > 0.005, (len(ys), len(xs))).copy()
    _write_band(tmp_path, "ur5-wsg", rect, covered=covered)
    monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)
    flag_cfg = {
        "objects": {**CFG, "near_path_visible_only": True, "on_path_visible_only": True},
        "corridor_t": [0.22, 0.48],
    }

    def run(cfg, seed):
        env = _fake_env()
        env.embodiment = "ur5-wsg"
        return spawn.spawn_candidates(env, cfg, np.random.RandomState(seed))

    vis_cov = (xs, ys, covered)
    n_near = 0
    for seed in (0, 1, 2):
        specs, report = run(flag_cfg, seed)
        assert report["vis_cov"] == "ur5-wsg"
        assert len(specs) >= 2
        for p in report["placements"]:
            if p["cls"] == "near_path":
                n_near += 1
                xy = np.array(p["xyz"][:2])
                if not p["vis_fallback"]:
                    assert spawn._covered_at(vis_cov, xy)
                    assert xy[1] > 0.005
        # determinism: same seed -> identical placements
        specs2, report2 = run(flag_cfg, seed)
        assert [p["xyz"] for p in report["placements"]] == [
            p["xyz"] for p in report2["placements"]
        ]
        # missing npz -> byte-identical to flags OFF, same seed
        empty = tmp_path / "empty"
        empty.mkdir(exist_ok=True)
        monkeypatch.setattr(spawn, "_CONFIGS_DIR", empty)
        _, rep_missing = run(flag_cfg, seed)
        _, rep_off = run(TASK_CFG, seed)
        monkeypatch.setattr(spawn, "_CONFIGS_DIR", tmp_path)
        assert rep_missing["vis_cov"] is None
        assert [p["xyz"] for p in rep_missing["placements"]] == [
            p["xyz"] for p in rep_off["placements"]
        ]
    assert n_near > 0
