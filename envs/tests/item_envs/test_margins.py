"""Pure-numpy tests for main.item_envs.margins."""

from __future__ import annotations

import numpy as np

from main.item_envs.margins import (
    m_inside,
    m_outside,
    m_particles_inside,
    m_particles_outside,
    m_stay,
    m_upright,
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
    # tilt term dominates
    assert np.isclose(m_stay(p_ref, p_ref, 0.30, 0.05, 0.35), 0.05)
    assert np.isclose(m_stay(moved, p_ref, 0.34, 0.05, 0.35), 0.01)


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
