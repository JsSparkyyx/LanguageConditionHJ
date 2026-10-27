"""T3 safe mode: the route planner (robotwin_tasks/_push_route.py), pure numpy, no simulator.

Synthetic scenes (a box pushed +x across the near half of the table): the straight route when nothing protected
is in the way, detours around a mat and around a cylinder (the WSG hand included), two elements, the no-route
cases with their reasons, the reach and table limits, the executor's guard; then replayed T3 layouts
(test_push_zone's loader). Every route the planner returns is re-checked here independently (shapely polygons,
the box at its yaw +- the allowance, the hand behind it, sampled every 2 mm along the route)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "robotwin_tasks" / "_push_route.py"


@pytest.fixture(scope="module")
def pr():
    spec = importlib.util.spec_from_file_location("push_route_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


START, GOAL = (-0.30, -0.10), (0.25, -0.10)
BOX = (0.03, 0.025)   # half sides
BOX_H = 0.05


def _route(pr, elems, protect=None, reach=None, start=START, goal=GOAL, **cfg):
    return pr.plan_route(start, goal, BOX, 0.0, elems, protect=protect, cfg=cfg, reach_ok=reach, box_h=BOX_H)


def _rect(c, ang, a, b):
    from shapely.geometry import Polygon

    ca, sa = math.cos(ang), math.sin(ang)
    pts = [(c[0] + ca * u - sa * v, c[1] + sa * u + ca * v) for u, v in ((a, b), (-a, b), (-a, -b), (a, -b))]
    return Polygon(pts)


def _shape(e):
    from shapely.geometry import Point

    if e.kind == "zone":
        return _rect(e.xy, 0.0, e.half[0], e.half[1])
    return Point(e.xy).buffer(e.half[0], 64)


def _clearances(pr, start, via, elems, box=BOX, step=0.002):
    """Independent re-check: per element, the least distance from the box and from the WSG base behind it (EE =
    centre - d (depth + 1.5 cm), 14.6 x 5 cm across the push) along the route. The box keeps its yaw for the first
    6 cm of a leg, then lies flush with the fingers (the planner's model, which adds +- 15 deg on top)."""
    pts = [np.asarray(start, dtype=float)] + [np.asarray(v, dtype=float) for v in via]
    out = {e.id: np.inf for e in elems}
    th_prev = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        d = (b - a) / length
        phi = math.atan2(d[1], d[0])
        th_al = pr._aligned(th_prev, phi)[0]
        for s in np.arange(0.0, length + 1e-9, step):
            c = a + d * s
            for th in ((th_prev, th_al) if s < pr.ALIGN_LEN else (th_al,)):
                bx = _rect(c, th, box[0], box[1])
                depth = abs(math.cos(phi - th)) * box[0] + abs(math.sin(phi - th)) * box[1]
                ee = c - d * (depth + 0.015)
                hand = _rect(ee, phi + math.pi / 2.0, 0.073, 0.025)
                for e in elems:
                    dist = bx.distance(_shape(e))
                    if e.kind == "obstacle":
                        dist = min(dist, hand.distance(_shape(e)))
                    out[e.id] = min(out[e.id], dist)
        th_prev = th_al if length >= pr.ALIGN_LEN else th_prev
    return out


def test_straight_route_when_nothing_is_in_the_way(pr):
    r = _route(pr, [])
    assert r.ok and r.reason is None
    assert np.allclose(r.via, [GOAL]) and r.turns_deg == []
    assert r.length == pytest.approx(0.55)


def test_unprotected_elements_are_ignored(pr):
    mat = pr.Elem("zone_red", "zone", (-0.05, -0.10), (0.04, 0.035))
    r = _route(pr, [mat], protect=[])
    assert r.ok and np.allclose(r.via, [GOAL])          # named mode, another rule named: straight through
    r = _route(pr, [mat], protect=["zone_red"])
    assert r.ok and len(r.via) >= 2 and set(r.sides) == {"zone_red"}


@pytest.mark.parametrize("mat_y", [-0.115, -0.085])   # on-path mats sit within 2 cm of the line
def test_mat_detour_keeps_the_box_off_the_mat(pr, mat_y):
    pytest.importorskip("shapely")
    mat = pr.Elem("zone_red", "zone", (-0.05, mat_y), (0.04, 0.035))
    cfg = pr.RouteCfg()
    r = _route(pr, [mat])
    assert r.ok, r.as_dict()
    assert all(t <= cfg.max_turn_deg + 1e-6 for t in r.turns_deg) and r.first_deg <= cfg.max_first_deg + 1e-6
    clear = _clearances(pr, START, r.via, [mat])
    assert clear["zone_red"] >= cfg.mat_clear - 1e-3, clear
    assert r.min_clear["zone_red"] >= cfg.mat_clear - 1e-9 and r.min_clear["table"] >= cfg.table_clear


def test_cylinder_detour_clears_the_hand(pr):
    pytest.importorskip("shapely")
    cyl = pr.Elem("obstacle_0", "obstacle", (0.0, -0.06), (0.03, 0.03), 0.15)
    cfg = pr.RouteCfg()
    r = _route(pr, [cyl])
    assert r.ok, r.as_dict()
    clear = _clearances(pr, START, r.via, [cyl])
    assert clear["obstacle_0"] >= cfg.cyl_clear_hand - 1e-3, clear
    # the lane abreast of the cylinder clears it by the hand's half width, not the box's
    lane_y = min(v[1] for v in r.via[:-1]) if r.sides["obstacle_0"] == "near" else max(v[1] for v in r.via[:-1])
    assert abs(lane_y - cyl.xy[1]) >= cyl.half[0] + 0.073 + cfg.cyl_clear_hand - 1e-6


def test_a_cylinder_needs_a_wider_lane_than_a_mat(pr):
    mat = pr.Elem("zone_red", "zone", (0.0, -0.08), (0.03, 0.03))
    cyl = pr.Elem("obstacle_0", "obstacle", (0.0, -0.08), (0.03, 0.03), 0.15)
    lane = {}
    for e in (mat, cyl):
        r = _route(pr, [e])
        assert r.ok
        lane[e.kind] = max(abs(v[1] - e.xy[1]) for v in r.via[:-1])
    assert lane["obstacle"] > lane["zone"] + 0.03, lane


def test_two_elements_on_the_line(pr):
    pytest.importorskip("shapely")
    elems = [pr.Elem("zone_red", "zone", (-0.12, -0.10), (0.035, 0.035)),
             pr.Elem("obstacle_0", "obstacle", (0.10, -0.14), (0.025, 0.025), 0.12)]
    cfg = pr.RouteCfg()
    r = _route(pr, elems)
    assert r.ok, r.as_dict()
    assert set(r.sides) == {"zone_red", "obstacle_0"}
    clear = _clearances(pr, START, r.via, elems)
    assert clear["zone_red"] >= cfg.mat_clear - 1e-3 and clear["obstacle_0"] >= cfg.cyl_clear_hand - 1e-3, clear


def test_a_short_lead_in_takes_a_sideways_first_leg(pr):
    """A mat 8 cm past the box (T3's LEAD_IN is 10 cm to the box front): no gentle entry exists, the first leg
    pushes the box sideways (up to 90 deg) and the hand lifts over it at the corner."""
    mat = pr.Elem("zone_red", "zone", (-0.30 + 0.03 + 0.08 + 0.04, -0.10), (0.04, 0.04))
    r = _route(pr, [mat])
    assert r.ok, r.as_dict()
    assert r.first_deg > 45.0 and max(r.turns_deg) > pr.RouteCfg().lift_turn_deg
    narrow = _route(pr, [mat], max_turn_deg=45.0, max_first_deg=45.0)
    assert not narrow.ok


def test_no_route_reports_the_element_side_and_check(pr):
    """A cylinder on the line next to the near table edge, the far side out of reach: no route, and the reasons
    name the element, the side and the check that failed."""
    start, goal = (-0.30, -0.27), (0.25, -0.27)
    cyl = pr.Elem("obstacle_0", "obstacle", (0.0, -0.26), (0.03, 0.03), 0.15)
    near_only = lambda xy: np.asarray(xy)[..., 1] < -0.15  # noqa: E731
    r = _route(pr, [cyl], start=start, goal=goal, reach=near_only)
    assert not r.ok and r.via == []
    d = r.as_dict()
    assert d["reason"] == "lanes_blocked", d
    assert d["detail"]["blocking"] == ["obstacle_0"]
    lanes = d["detail"]["lanes"]
    assert any(k.startswith("obstacle_0:near") and v == "table:None" for k, v in lanes.items()), lanes
    assert any(k.startswith("obstacle_0:far") and v == "reach:None" for k, v in lanes.items()), lanes


def test_reach_limits_the_detour(pr):
    mat = pr.Elem("zone_red", "zone", (-0.05, -0.10), (0.04, 0.035))
    free = _route(pr, [mat])
    assert free.ok
    side = free.sides["zone_red"]
    other = "far" if side == "near" else "near"
    # forbid the chosen side: the route takes the other one
    sign = -1.0 if side == "near" else 1.0
    no_side = lambda xy: (np.asarray(xy)[..., 1] - (-0.10)) * sign < 0.04  # noqa: E731
    r = _route(pr, [mat], reach=no_side)
    assert r.ok and r.sides["zone_red"] == other, r.as_dict()
    # forbid both: no route, reach named as the failure
    narrow_band = lambda xy: np.abs(np.asarray(xy)[..., 1] + 0.10) < 0.04  # noqa: E731
    r = _route(pr, [mat], reach=narrow_band)
    assert not r.ok
    d = r.as_dict()["detail"]
    assert any(v == "reach:None" for v in d["lanes"].values()) or "reach:None" in (d.get("fails") or {}), d


def test_table_edge_limits_the_detour(pr):
    """Near the robot edge the near side leaves the table (table_clear): the route goes round the far side."""
    start, goal = (-0.30, -0.25), (0.25, -0.25)
    mat = pr.Elem("zone_red", "zone", (-0.05, -0.25), (0.04, 0.035))
    r = _route(pr, [mat], start=start, goal=goal)
    assert r.ok and r.sides["zone_red"] == "far", r.as_dict()
    assert r.min_clear["table"] >= pr.RouteCfg().table_clear
    assert any(k.startswith("zone_red:near") and v == "table:None" for k, v in r.detail["lanes"].items()), r.detail


def test_start_blocked(pr):
    mat = pr.Elem("zone_red", "zone", (-0.31, -0.10), (0.03, 0.03))   # under the box
    r = _route(pr, [mat])
    assert not r.ok and r.reason == "start_blocked"
    assert r.detail["fail"]["elem"] == "zone_red"


def test_deterministic(pr):
    elems = [pr.Elem("zone_red", "zone", (-0.12, -0.10), (0.035, 0.035)),
             pr.Elem("obstacle_0", "obstacle", (0.10, -0.14), (0.025, 0.025), 0.12)]
    assert _route(pr, elems).as_dict() == _route(pr, elems).as_dict()


def test_replan_prefers_the_previous_sides(pr):
    mat = pr.Elem("zone_red", "zone", (-0.05, -0.10), (0.04, 0.035))
    first = _route(pr, [mat])
    s = first.detail["lane_sides"]["zone_red"]
    again = pr.plan_route(START, GOAL, BOX, 0.0, [mat], box_h=BOX_H, prefer={"zone_red": -s})
    assert again.ok and again.detail["lane_sides"]["zone_red"] == -s


def test_guard_flags_a_hand_sweep_through_a_cylinder(pr):
    """check_motion (push_via's guard): a push-height EE move past a cylinder fails on the hand; the same move
    with the hand lifted above the cylinder passes; a box slide into a mat fails on the box."""
    cyl = pr.Elem("obstacle_0", "obstacle", (0.0, -0.02), (0.03, 0.03), 0.12)
    q = [-0.5, 0.5, -0.5, -0.5]   # TOP_DOWN: the finger axis along world x, across a +y push
    low = [[-0.10, -0.06, 0.74 + 0.172] + q, [0.10, -0.06, 0.74 + 0.172] + q]
    ok, why, _ = pr.check_motion(low, (0.0, -0.30), (0.0, -0.30), BOX, 0.0, [cyl], box_h=BOX_H)
    assert not ok and why["check"] == "hand_cyl" and why["elem"] == "obstacle_0"
    high = [[-0.10, -0.06, 0.74 + 0.12 + 0.164 + 0.02] + q, [0.10, -0.06, 0.74 + 0.12 + 0.164 + 0.02] + q]
    ok, why, _ = pr.check_motion(high, (0.0, -0.30), (0.0, -0.30), BOX, 0.0, [cyl], box_h=BOX_H)
    assert ok, why
    mat = pr.Elem("zone_red", "zone", (0.0, -0.10), (0.03, 0.03))
    ok, why, _ = pr.check_motion([high[0]], (-0.12, -0.10), (-0.05, -0.10), BOX, 0.0, [mat], box_h=BOX_H)
    assert not ok and why["check"] == "box_mat"


def test_reach_npz_mirrors_the_right_arm(pr):
    npz = ROOT / "main" / "item_envs" / "configs" / "workspace_ur5-wsg.npz"
    if not npz.is_file():
        pytest.skip("no workspace npz")
    left, right = pr.reach_npz(npz, "left"), pr.reach_npz(npz, "right")
    xy = np.stack(np.meshgrid(np.linspace(-0.6, 0.6, 31), np.linspace(-0.35, 0.35, 15)), axis=-1)
    mirror = xy * np.array([-1.0, 1.0])
    assert np.array_equal(left(xy), right(mirror))
    assert left(np.array([-0.3, -0.1])) and not left(np.array([0.5, 0.3]))   # own half near / far corner
    eroded = pr.reach_npz(npz, "left", erode=2)
    assert eroded(xy).sum() < left(xy).sum() and not (eroded(xy) & ~left(xy)).any()


def test_constants_mirror_push_zone(pr):
    from tests.item_envs.test_push_zone import _const

    assert pr.TABLE_HALF_XY == (_const("TABLE_HALF_X"), _const("TABLE_HALF_Y"))
    assert pr.OWN_SIDE_Y_MAX == _const("OWN_SIDE_Y_MAX")
    assert pr.REACH_X_MAX == _const("REACH_X_MAX")


def test_wsg_hand_matches_the_safe_plan_hand_boxes(pr):
    from main.item_envs.safe_plan import HAND_BOXES

    hand = pr.wsg_hand("wide", tip_len=0.164)
    lo, hi = HAND_BOXES["wsg_50_base_link"]
    base = hand.layer("base")
    assert (base.a, base.b, -base.z_lo) == pytest.approx((hi[0], hi[1], hi[2]))
    fl, fh = HAND_BOXES["finger_left"]
    fingers = hand.layer("fingers")
    assert fingers.b == pytest.approx(fh[1])                  # 3 cm thick along the push
    assert fingers.a >= -fl[0] + 0.0027 - 1e-9                # closed: the pads meet in the middle
    assert hand.contact == pytest.approx(0.015) and pr.wsg_hand("narrow").contact == pytest.approx(fingers.a)


def test_yaml_route_knobs_are_route_cfg_fields(pr):
    import yaml

    cfg = yaml.safe_load((ROOT / "main" / "item_envs" / "configs" / "tasks" / "T3_push_zone.yaml").read_text())
    names = {f for f in pr.RouteCfg.__dataclass_fields__}
    assert set(cfg["safe_mode"]["route"]) <= names, set(cfg["safe_mode"]["route"]) - names


# ------------------------------------------------------- replayed layouts ----


@pytest.fixture(scope="module")
def replay():
    from tests.item_envs.test_push_zone import _load_task_module, _replay_layout

    pz, restore = _load_task_module()
    yield pz, _replay_layout
    restore()


def test_replayed_layouts_routes_are_clear(pr, replay):
    """Protect every element (the demo collector's case) on 12 replayed layouts: each route found passes the
    independent re-check against every element, and the planner reports a reason for each layout it skips."""
    pytest.importorskip("shapely")
    pz, replay_layout = replay
    found = 0
    for seed in range(12):
        task = replay_layout(pz, seed)
        start, goal, bhalf, yaw, box_h, elems = task._route_inputs()
        r = pr.plan_route(start, goal, bhalf, yaw, elems, box_h=box_h,
                          reach_ok=pr.make_reach("linear", task.push_objects[0]["arm"]))
        if not r.ok:
            assert r.reason and r.detail.get("blocking") is not None, (seed, r.as_dict())
            continue
        found += 1
        cfg = pr.RouteCfg()
        clear = _clearances(pr, start, r.via, elems, box=bhalf, step=0.004)
        for e in elems:
            need = cfg.mat_clear if e.kind == "zone" else min(cfg.cyl_clear_box, cfg.cyl_clear_hand)
            assert clear[e.id] >= need - 1e-3, (seed, e.id, clear, r.as_dict())
    assert found >= 6, found
