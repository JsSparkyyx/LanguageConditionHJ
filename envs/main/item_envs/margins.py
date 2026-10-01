"""Failure-margin functions for item-safety rules.

Every function returns a float in raw SI units (metres or radians), positive
meaning safe, zero the failure boundary, never clipped. Pure numpy.
"""

from __future__ import annotations

import numpy as np

from .geometry import signed_dist_inside_polygon


def softmin(x: np.ndarray, T: float) -> float:
    """-T * logsumexp(-x / T), numerically stable. <= min(x), >= min(x) - T*log(N)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("softmin of an empty set")
    a = float(x.min())
    return a - T * float(np.log(np.exp(-(x - a) / T).sum()))


def m_inside(item_pts_region_frame: np.ndarray, region, require_on_surface: bool) -> float:
    """Item footprint inside the region polygon (optionally resting on its surface).

    item_pts_region_frame: [4, 3] footprint corners in the region frame.
    xy margin = min over corners of the polygon signed distance. When
    require_on_surface, the margin also accounts for the vertical offset of the
    lowest corner from region.z: m = min(xy_margin, z_band - |dz|). This equals
    min(xy_margin, -(|dz| - z_band)) once |dz| > z_band (the violation branch in
    the plan) and stays continuous at the band edge.
    """
    pts = np.asarray(item_pts_region_frame, dtype=np.float64).reshape(-1, 3)
    xy_margin = float(signed_dist_inside_polygon(pts[:, :2], region.polygon).min())
    if not require_on_surface:
        return xy_margin
    dz = abs(float(pts[:, 2].min()) - float(region.z))
    return min(xy_margin, float(region.z_band) - dz)


def m_outside(item_pts_region_frame: np.ndarray, region) -> float:
    """Item footprint outside the region polygon: -max over corners of signed_dist_inside."""
    pts = np.asarray(item_pts_region_frame, dtype=np.float64).reshape(-1, 3)
    return float(-signed_dist_inside_polygon(pts[:, :2], region.polygon).max())


def m_upright(tilt: float, theta_max: float) -> float:
    """theta_max - tilt (radians)."""
    return float(theta_max) - float(tilt)


def m_stay(p_now: np.ndarray, p_ref: np.ndarray, tilt_now: float, d_tol: float, theta_tol: float) -> float:
    """min(d_tol - horizontal displacement from the reference, theta_tol - tilt)."""
    p_now = np.asarray(p_now, dtype=np.float64).reshape(-1)
    p_ref = np.asarray(p_ref, dtype=np.float64).reshape(-1)
    d = float(np.linalg.norm(p_now[:2] - p_ref[:2]))
    return min(float(d_tol) - d, float(theta_tol) - float(tilt_now))


def m_particles_inside(particle_xy_by_region_frame: list[np.ndarray], regions, T: float = 0.005) -> float:
    """Every particle stays in the column above one of the regions.

    Per particle: max over regions of the polygon signed distance; then a
    soft-min over particles with temperature T (metres).
    """
    per_region = _per_region_signed_dist(particle_xy_by_region_frame, regions)  # [R, N]
    per_particle = per_region.max(axis=0)
    return softmin(per_particle, T)


def m_particles_outside(particle_xy_by_region_frame: list[np.ndarray], regions, T: float = 0.005) -> float:
    """No particle enters the column above any of the regions.

    Per particle: min over regions of -signed_dist_inside; then the same soft-min.
    """
    per_region = _per_region_signed_dist(particle_xy_by_region_frame, regions)  # [R, N]
    per_particle = (-per_region).min(axis=0)
    return softmin(per_particle, T)


def _per_region_signed_dist(particle_xy_by_region_frame: list[np.ndarray], regions) -> np.ndarray:
    """[R, N] polygon signed distances of the particles expressed in each region frame."""
    if len(particle_xy_by_region_frame) != len(regions):
        raise ValueError("need one particle xy array per region")
    cols = []
    n_particles = None
    for pts, region in zip(particle_xy_by_region_frame, regions):
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        if n_particles is None:
            n_particles = pts.shape[0]
        elif pts.shape[0] != n_particles:
            raise ValueError("particle count differs across region frames")
        cols.append(signed_dist_inside_polygon(pts[:, :2], region.polygon))
    return np.stack(cols, axis=0)
