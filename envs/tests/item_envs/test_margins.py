"""Pure-numpy tests for main.item_envs.margins."""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.margins import (
    m_avoid,
    m_inside,
    m_outside,
    m_particles_inside,
    m_particles_outside,
    m_stay,
    m_upright,
    set_rule_margin,
    softmin,
)
from main.item_envs.regions import Region

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def make_region(polygon=UNIT_SQUARE, z=0.0, z_band=0.01, kind="surface", region_id="r"):
    return Region(
        region_id=region_id,
        kind=kind,
        frame_fn=lambda: np.eye(4),
        polygon=polygon,
        z=z,
        z_band=z_band,
    )


def footprint_xy(x0, y0, hx=0.005, hy=0.005, z=0.0):
    """Small 4-corner footprint centred at (x0, y0)."""
    return np.array(
        [[x0 - hx, y0 - hy, z], [x0 + hx, y0 - hy, z], [x0 + hx, y0 + hy, z], [x0 - hx, y0 + hy, z]]
    )


def test_m_inside_values():
    region = make_region()
    inside = m_inside(footprint_xy(0.5, 0.5), region, require_on_surface=False)
    assert np.isclose(inside, 0.495)  # limited by the corner nearest an edge
    straddling = m_inside(footprint_xy(1.0, 0.5), region, require_on_surface=False)
    assert np.isclose(straddling, -0.005)  # outer corners stick out by hx
    outside = m_inside(footprint_xy(1.3, 0.5), region, require_on_surface=False)
    assert np.isclose(outside, -0.305)  # min over corners = farthest outside corner


def test_m_inside_continuity_polygon_edge():
    region = make_region()
    for require in (False, True):
        f_minus = m_inside(footprint_xy(1.0 - 1e-4, 0.5), region, require_on_surface=require)
        f_plus = m_inside(footprint_xy(1.0 + 1e-4, 0.5), region, require_on_surface=require)
        assert abs(f_plus - f_minus) < 2.5e-4  # slope 1 over a 2e-4 step


def test_m_inside_continuity_z_band():
    region = make_region(z=0.0, z_band=0.01)
    pts_low = footprint_xy(0.5, 0.5, z=0.01 - 1e-4)
    pts_high = footprint_xy(0.5, 0.5, z=0.01 + 1e-4)
    f_low = m_inside(pts_low, region, require_on_surface=True)
    f_high = m_inside(pts_high, region, require_on_surface=True)
    assert abs(f_high - f_low) < 2.5e-4
    # just below the band the vertical slack dominates; just above, the margin
    # enters the violation branch min(xy, -(|dz| - z_band)) continuously
    assert np.isclose(f_low, 1e-4)
    assert np.isclose(f_high, -1e-4)
    # deeper lift: exact violation-branch value
    assert np.isclose(m_inside(footprint_xy(0.5, 0.5, z=0.015), region, True), -0.005)
    # item outside the polygon and lifted: min keeps the more negative xy part
    assert np.isclose(m_inside(footprint_xy(1.3, 0.5, z=0.015), region, True), -0.305)


def test_m_inside_require_on_surface_flag():
    region = make_region(z_band=0.01)
    lifted = footprint_xy(0.5, 0.5, z=0.05)
    assert m_inside(lifted, region, require_on_surface=False) > 0
    assert m_inside(lifted, region, require_on_surface=True) < 0
    # lowest corner drives the check: one corner resting keeps it "on surface"
    pts = footprint_xy(0.5, 0.5, z=0.0)
    pts[3, 2] = 0.05
    assert m_inside(pts, region, require_on_surface=True) > 0


def test_m_outside():
    region = make_region()
    assert np.isclose(m_outside(footprint_xy(1.3, 0.5), region), 0.295)
    assert np.isclose(m_outside(footprint_xy(0.5, 0.5), region), -0.495)
    assert np.isclose(m_outside(footprint_xy(1.0, 0.5), region), -0.005)  # straddling


def test_m_avoid():
    assert m_avoid(0.10) == 0.10  # margin defaults to 0 (T1 setting)
    assert np.isclose(m_avoid(0.10, margin=0.02), 0.08)
    assert np.isclose(m_avoid(0.02, margin=0.02), 0.0)  # zero at the boundary
    assert m_avoid(0.01, margin=0.02) < 0  # closer than the margin violates
    assert m_avoid(np.inf) == np.inf  # no item / nothing reachable stays safe


def test_set_rule_margin():
    assert set_rule_margin([0.10, 0.02, 0.30]) == 0.02  # min over the items
    assert set_rule_margin([-0.01, 0.02]) == -0.01  # one violating item fails the rule
    assert set_rule_margin(np.array([0.05])) == 0.05
    assert set_rule_margin(0.05) == 0.05  # scalar tolerated
    with pytest.raises(ValueError):
        set_rule_margin([])


def test_m_upright_zero_at_theta_max():
    theta_max = np.deg2rad(45.0)
    assert m_upright(theta_max, theta_max) == 0.0
    assert m_upright(theta_max - 0.1, theta_max) > 0
    assert m_upright(theta_max + 0.1, theta_max) < 0


def test_m_stay():
    p_ref = np.array([0.3, 0.4, 0.7])
    assert np.isclose(m_stay(p_ref, p_ref, 0.0, 0.05, 0.35), 0.05)
    moved = np.array([0.32, 0.4, 0.9])  # z ignored, xy displacement 0.02
    assert np.isclose(m_stay(moved, p_ref, 0.0, 0.05, 0.35), 0.03)
    # tilt term, in metres: the tilt room as a fraction of theta_tol, times d_tol
    assert np.isclose(m_stay(p_ref, p_ref, 0.30, 0.05, 0.35), 0.05 * 0.05 / 0.35)
    assert np.isclose(m_stay(moved, p_ref, 0.34, 0.05, 0.35), 0.05 * 0.01 / 0.35)
    # the sign follows either limit: past theta_tol or past d_tol
    assert m_stay(p_ref, p_ref, 0.36, 0.05, 0.35) < 0.0
    assert m_stay(np.array([0.36, 0.4, 0.7]), p_ref, 0.0, 0.05, 0.35) < 0.0


def test_rule_h_units_and_clip():
    """h = clip(margin / h_unit(rule), -1, 1): 5 cm for distance rules, the
    rule's own theta_max (upright) / d_tol (stay) — 1.0 = comfortably safe
    for every rule (user 2026-10-05)."""
    from types import SimpleNamespace

    from main.item_envs.margins import h_unit, rule_h
    from main.item_envs.rules import Consequence

    def rule(cons, **params):
        return SimpleNamespace(consequence=cons, params=params)

    avoid = rule(Consequence.AVOID, margin=0.01)
    upright = rule(Consequence.UPRIGHT, theta_max=np.deg2rad(10.0))
    stay = rule(Consequence.STAY, d_tol=0.015, theta_tol=np.deg2rad(10.0))
    beads = rule(Consequence.PARTICLES_INSIDE)
    assert h_unit(avoid) == 0.05 and h_unit(beads) == 0.01  # a resting bead sits ~1 cm inside
    assert h_unit(rule(Consequence.PARTICLES_OUTSIDE)) == 0.05
    assert np.isclose(h_unit(upright), np.deg2rad(10.0)) and h_unit(stay) == 0.015
    assert np.isclose(rule_h(avoid, 0.02), 0.4)  # 20 x metres, CrossSafe's kappa
    assert rule_h(avoid, 0.3) == 1.0 and rule_h(avoid, -23.0) == -1.0  # clipped both ways
    assert np.isclose(rule_h(upright, np.deg2rad(10.0)), 1.0)  # standing straight
    assert np.isclose(rule_h(upright, np.deg2rad(5.0)), 0.5)
    assert rule_h(upright, np.deg2rad(-30.0)) == -1.0
    assert np.isclose(rule_h(stay, 0.0075), 0.5)
    assert rule_h(beads, 0.0) == 0.0


def test_softmin_properties():
    rng = np.random.RandomState(0)
    for n in (1, 2, 7, 50):
        x = rng.uniform(-0.1, 0.5, size=n)
        for T in (0.005, 0.02):
            s = softmin(x, T)
            assert s <= x.min() + 1e-15  # conservative
            assert s >= x.min() - T * np.log(n) - 1e-12
    assert np.isclose(softmin([0.3], 0.005), 0.3)
    # approaches the hard min as T -> 0
    x = np.array([0.1, 0.4, 0.4])
    assert abs(softmin(x, 1e-4) - 0.1) < 1e-6


def test_particles_inside_single_region():
    region = make_region()
    pts = np.array([[0.5, 0.5], [0.1, 0.5], [0.9, 0.9]])
    m = m_particles_inside([pts], [region], T=0.005)
    per_particle = np.array([0.5, 0.1, 0.1])
    assert m <= per_particle.min()
    assert m >= per_particle.min() - 0.005 * np.log(3)
    assert m > 0


def test_particles_inside_two_regions_column_semantics():
    region_a = make_region(UNIT_SQUARE, region_id="a")
    region_b = make_region(np.array([[3, 0], [4, 0], [4, 1], [3, 1]], dtype=float), region_id="b")
    # particle safe in either column counts safe; same world xy passed per frame
    pts = np.array([[0.5, 0.5], [3.5, 0.5], [2.0, 0.5]])
    m = m_particles_inside([pts, pts], [region_a, region_b], T=0.005)
    per_particle = np.array([0.5, 0.5, -1.0])  # max over regions
    assert m <= per_particle.min()
    assert m >= per_particle.min() - 0.005 * np.log(3)
    assert m < 0  # the particle between the regions violates
    # all particles inside one region or the other -> positive
    pts_ok = np.array([[0.5, 0.5], [3.5, 0.5]])
    assert m_particles_inside([pts_ok, pts_ok], [region_a, region_b], T=0.005) > 0


def test_particles_outside():
    region = make_region()
    pts_out = np.array([[1.5, 0.5], [-0.3, 0.2]])
    m = m_particles_outside([pts_out], [region], T=0.005)
    per_particle = np.array([0.5, 0.3])  # min over regions of -sd_inside
    assert m <= per_particle.min()
    assert m >= per_particle.min() - 0.005 * np.log(2)
    assert m > 0
    # one particle deep inside -> violation dominates the soft-min
    pts = np.array([[0.5, 0.5], [1.5, 0.5]])
    m = m_particles_outside([pts], [region], T=0.005)
    assert m < -0.49


def test_particles_softmin_conservative_bound_random():
    rng = np.random.RandomState(3)
    region = make_region()
    T = 0.005
    for _ in range(10):
        pts = rng.uniform(-0.5, 1.5, size=(12, 2))
        m = m_particles_inside([pts], [region], T=T)
        from main.item_envs.geometry import signed_dist_inside_polygon

        per_particle = signed_dist_inside_polygon(pts, region.polygon)
        assert m <= per_particle.min() + 1e-15
        assert m >= per_particle.min() - T * np.log(len(pts)) - 1e-12


def test_particles_outside_capped_column():
    """Region.z_cap (T6): a particle over the region but above the cap is
    outside it (margin = its height above the cap); below the cap the polygon
    distance rules; xy-only input ignores the cap."""
    region = make_region()
    region.z_cap = 0.01
    above = np.array([[0.5, 0.5, 0.05]])  # 4 cm above the cap, over the centre
    m = m_particles_outside([above], [region], T=0.001)
    assert m == pytest.approx(0.04, abs=1e-6)
    inside = np.array([[0.5, 0.5, 0.0]])  # dropped in: below the cap
    assert m_particles_outside([inside], [region], T=0.001) < -0.009
    beside = np.array([[1.5, 0.5, 0.0]])  # on the table beside it
    assert m_particles_outside([beside], [region], T=0.001) == pytest.approx(0.5, abs=1e-6)
    assert m_particles_outside([above[:, :2]], [region], T=0.001) < -0.49  # xy only: uncapped


def test_particle_h_counts_beads():
    """Bead rules (2026-10-07): -0.5 for one violating bead, linearly to -1 at 20 or more;
    the safe side is clip(margin / unit, 0, 1)."""
    from main.item_envs.margins import PARTICLE_N_FULL, particle_h, rule_h
    from main.item_envs.rules import Consequence, RuleSpec

    inside = RuleSpec("particles_inside|beads|bowl", Consequence.PARTICLES_INSIDE, ("beads",), ("bowl",), {})
    outside = RuleSpec("particles_outside|beads|bin", Consequence.PARTICLES_OUTSIDE, ("beads",), ("bin",), {})
    assert particle_h(1, -0.03, 0.01) == -0.5
    assert np.isclose(particle_h(2, -0.001, 0.01), -0.5 - 0.5 / 19)
    assert particle_h(PARTICLE_N_FULL, -0.001, 0.01) == -1.0 and particle_h(57, -0.001, 0.01) == -1.0
    hs = [particle_h(n, -0.01, 0.01) for n in range(1, 25)]
    assert all(a > b for a, b in zip(hs[:19], hs[1:20])) and len(set(hs[19:])) == 1   # strictly worse to 20
    assert np.isclose(rule_h(inside, 0.005, 0), 0.5)          # 5 mm inside, unit 1 cm
    assert rule_h(inside, 0.03, 0) == 1.0
    assert rule_h(inside, -0.002, 0) == 0.0                    # (a soft-min margin can dip below 0 with no bead out)
    assert np.isclose(rule_h(outside, 0.025, 0), 0.5)          # 2.5 cm from the column, unit 5 cm
    assert rule_h(outside, -0.004, 1) == -0.5
    assert rule_h(inside, -0.004, 3) == rule_h(outside, -0.2, 3)   # depth no longer matters once violated
    assert np.isclose(rule_h(inside, -0.004), -0.4)            # without a count: the old clip(margin / unit)
    avoid = RuleSpec("avoid|cup|-", Consequence.AVOID, ("cup",), (), {"margin": 0.01})
    assert np.isclose(rule_h(avoid, -0.02, 5), -0.4)           # counts only apply to bead rules


def test_particle_margin_hard_min_by_default():
    from main.item_envs.margins import m_particles_inside, m_particles_outside

    region = make_region(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]), kind="container")
    pts = np.array([[0.0, 0.0, 0.0]] * 60 + [[0.04, 0.0, 0.0]])            # 60 at the centre, one 1 cm from the wall
    assert np.isclose(m_particles_inside([pts], [region]), 0.01)           # the worst bead, no soft-min pull
    assert m_particles_inside([pts], [region], T=0.005) < 0.01              # an explicit T keeps the soft-min
    assert m_particles_outside([pts + [0.2, 0.0, 0.0]], [region]) > 0


def test_particle_margin_sign_is_the_bead_count():
    """Hard min (no T): margin < 0 exactly when at least one bead breaks the rule, for random bead clouds
    over one or two regions (the count margins.particle_h uses)."""
    from main.item_envs.margins import (m_particles_inside, m_particles_outside, per_particle_inside,
                                        per_particle_outside)

    sq = lambda c: np.array([[c[0] - 0.05, c[1] - 0.05], [c[0] + 0.05, c[1] - 0.05],  # noqa: E731
                             [c[0] + 0.05, c[1] + 0.05], [c[0] - 0.05, c[1] + 0.05]])
    rng = np.random.default_rng(0)
    seen = {True: 0, False: 0}
    for trial in range(300):
        regs = [make_region(sq((0.0, 0.0)), kind="container")]
        if trial % 2:
            regs.append(make_region(sq((0.2, 0.0)), kind="container"))
        n = int(rng.integers(1, 120))
        xy = rng.normal([0.0, 0.0], 0.004 + 0.03 * rng.random(), size=(n, 2))   # tight to loose clouds
        per_region = [xy for _ in regs]
        out_in = int((per_particle_inside(per_region, regs) < 0).sum())
        assert (m_particles_inside(per_region, regs) < 0) == (out_in > 0)
        out_out = int((per_particle_outside(per_region, regs) < 0).sum())
        assert (m_particles_outside(per_region, regs) < 0) == (out_out > 0)
        seen[out_in > 0] += 1
    assert min(seen.values()) > 20   # both cases exercised


def test_particle_margin_with_no_beads_is_safe():
    """No bead yet (a T4 episode's first tick): +inf, not a crash; h reads 1 (safe)."""
    from main.item_envs.margins import m_particles_inside, m_particles_outside, particle_h

    reg = make_region(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]), kind="container")
    empty = [np.zeros((0, 2))]
    assert m_particles_inside(empty, [reg]) == float("inf") and m_particles_outside(empty, [reg]) == float("inf")
    assert particle_h(0, float("inf"), 0.01) == 1.0
