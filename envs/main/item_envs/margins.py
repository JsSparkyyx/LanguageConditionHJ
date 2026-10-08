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


def m_avoid(robot_item_dist: float, margin: float = 0.0) -> float:
    """robot_item_dist - margin (metres). The distance comes from robot_distance.py
    (robot links + held payload to the item); it is not computed here."""
    return float(robot_item_dist) - float(margin)


def m_inside(item_pts_region: np.ndarray, region, require_on_surface: bool) -> float:
    """Item footprint inside the region polygon (optionally resting on its surface).

    item_pts_region: [4, 3] footprint corners in the region frame.
    xy margin = min over corners of the polygon signed distance. When
    require_on_surface, the margin also accounts for the vertical offset of the
    lowest corner from region.z: m = min(xy_margin, z_band - |dz|). This equals
    min(xy_margin, -(|dz| - z_band)) once |dz| > z_band (the violation branch in
    the plan) and stays continuous at the band edge.
    """
    pts = np.asarray(item_pts_region, dtype=np.float64).reshape(-1, 3)
    xy_margin = float(signed_dist_inside_polygon(pts[:, :2], region.polygon).min())
    if not require_on_surface:
        return xy_margin
    dz = abs(float(pts[:, 2].min()) - float(region.z))
    return min(xy_margin, float(region.z_band) - dz)


def m_outside(item_pts_region: np.ndarray, region) -> float:
    """Item footprint outside the region polygon: -max over corners of signed_dist_inside."""
    pts = np.asarray(item_pts_region, dtype=np.float64).reshape(-1, 3)
    return float(-signed_dist_inside_polygon(pts[:, :2], region.polygon).max())


def m_upright(tilt: float, theta_max: float) -> float:
    """theta_max - tilt (radians)."""
    return float(theta_max) - float(tilt)


def m_stay(p_now: np.ndarray, p_ref: np.ndarray, tilt_now: float, d_tol: float, theta_tol: float) -> float:
    """min(d_tol - horizontal displacement, d_tol * (theta_tol - tilt) / theta_tol),
    metres: the tilt counts as movement, the full tilt allowance = the full
    movement allowance (2026-10-05; it was min(metres, radians): 1 deg of tilt
    weighed like 1.7 cm of shove). The sign — the violation — is unchanged."""
    p_now = np.asarray(p_now, dtype=np.float64).reshape(-1)
    p_ref = np.asarray(p_ref, dtype=np.float64).reshape(-1)
    d = float(np.linalg.norm(p_now[:2] - p_ref[:2]))
    return min(float(d_tol) - d, float(d_tol) * (float(theta_tol) - float(tilt_now)) / float(theta_tol))


H_UNIT_DISTANCE = 0.05  # m: one h unit for distance-type rules (h = 20 x metres, CrossSafe's kappa)
H_UNIT_PARTICLES_INSIDE = 0.01  # m: a bead resting against the container wall sits 1.0-1.2 cm inside (median at rest, T2/T4/T6 calibration)


def h_unit(rule) -> float:
    """The margin that reads h = 1 ("comfortably safe") for this rule, in the
    margin's own units (user 2026-10-05: the h ranges differed ~4x between
    rules — T5's upright topped out at 2/rad x 10 deg = 0.35 while distance
    rules reached 1.0 at 5 cm):

    - avoid / inside / outside / particles_outside: 5 cm (h = 20 x metres);
    - particles_inside ("don't spill"): 1 cm — the worst bead of a container
      at rest sits against its wall, 1.0-1.2 cm inside (median at the start
      of the T2 / T4 / T6 calibration episodes), so 5 cm capped it at ~0.2;
    - upright: the rule's own tilt limit theta_max (h = 1 standing straight,
      0 at the limit, -1 at twice the limit);
    - stay: the rule's own movement limit d_tol (its margin is in metres,
      the tilt converted, see m_stay).
    """
    name = getattr(getattr(rule, "consequence", None), "value", str(getattr(rule, "consequence", "")))
    params = getattr(rule, "params", None) or {}
    if name == "upright":
        return float(params["theta_max"])
    if name == "stay":
        return float(params["d_tol"])
    if name == "particles_inside":
        return H_UNIT_PARTICLES_INSIDE
    return H_UNIT_DISTANCE


PARTICLE_H_FIRST = 0.5  # h = -0.5 when one bead violates a bead rule (user 2026-10-07)
PARTICLE_N_FULL = 20    # ... falling linearly to h = -1 at 20 or more violating beads

PARTICLE_CONSEQUENCES = ("particles_inside", "particles_outside")


def is_particle_rule(rule) -> bool:
    name = getattr(getattr(rule, "consequence", None), "value", str(getattr(rule, "consequence", "")))
    return name in PARTICLE_CONSEQUENCES


def particle_h(n_violating: int, margin: float, unit: float) -> float:
    """Bead rules (user 2026-10-07: one stray bead and fifty stray beads both read -1 before).

    Safe side (no bead violating): clip(margin / unit, 0, 1), margin = the worst bead's distance.
    Violated side: by the number n of violating beads, -0.5 at n = 1, linearly to -1 at
    n >= PARTICLE_N_FULL: h = -(0.5 + 0.5 (min(n, 20) - 1) / 19).
    """
    n = int(n_violating)
    if n <= 0:
        return float(np.clip(float(margin) / float(unit), 0.0, 1.0))
    frac = (min(n, PARTICLE_N_FULL) - 1) / float(PARTICLE_N_FULL - 1)
    return -(PARTICLE_H_FIRST + (1.0 - PARTICLE_H_FIRST) * frac)


def rule_h(rule, margin: float, n_violating: int | None = None) -> float:
    """The failure function: clip(margin / h_unit(rule), -1, 1). The sign is
    the margin's (h < 0 = harmed); +1 / -1 = comfortably safe / badly harmed.

    Bead rules (particles_inside / particles_outside) given ``n_violating`` (the
    number of beads breaking the rule this tick, ``rule_violation_counts``) use
    ``particle_h``: the violated side counts beads instead of the worst bead's depth."""
    if n_violating is not None and int(n_violating) >= 0 and is_particle_rule(rule):
        return particle_h(n_violating, margin, h_unit(rule))
    return float(np.clip(float(margin) / h_unit(rule), -1.0, 1.0))


def per_particle_inside(particle_xy_per_region: list[np.ndarray], regions) -> np.ndarray:
    """[N] per bead: max over regions of the column signed distance (> 0 = inside one of them)."""
    return _per_region_signed_dist(particle_xy_per_region, regions).max(axis=0)


def per_particle_outside(particle_xy_per_region: list[np.ndarray], regions) -> np.ndarray:
    """[N] per bead: min over regions of -signed distance (> 0 = outside every column)."""
    return (-_per_region_signed_dist(particle_xy_per_region, regions)).min(axis=0)


def _bead_min(per_particle: np.ndarray, T: float | None) -> float:
    """Hard min (or soft-min with T) over beads; +inf with no bead (none can break the rule: what the
    soft-min returned, and what the bead count of 0 says). A T4 episode whose beads did not exist yet at
    its first tick crashed on the empty min (round-2 run, val episode 64000001)."""
    per_particle = np.asarray(per_particle, dtype=np.float64).reshape(-1)
    if per_particle.size == 0:
        return float("inf")
    return softmin(per_particle, T) if T else float(per_particle.min())


def m_particles_inside(particle_xy_per_region: list[np.ndarray], regions, T: float | None = None) -> float:
    """Every particle stays in the column above one of the regions.

    Per particle: max over regions of the polygon signed distance; then the
    min over particles (T = None, the default since 2026-10-07: margin < 0 iff a
    bead is outside, matching the bead count of ``particle_h``) or a soft-min with
    temperature T (metres), which can read < 0 with every bead inside.
    """
    per_particle = per_particle_inside(particle_xy_per_region, regions)
    return _bead_min(per_particle, T)


def m_particles_outside(particle_xy_per_region: list[np.ndarray], regions, T: float | None = None) -> float:
    """No particle enters the column above any of the regions.

    Per particle: min over regions of -signed_dist_inside; then the min (or soft-min) as above.
    """
    per_particle = per_particle_outside(particle_xy_per_region, regions)
    return _bead_min(per_particle, T)


def set_rule_margin(per_item_margins) -> float:
    """min over the per-item margins of a set rule (the rule holds iff every item does)."""
    vals = np.asarray(per_item_margins, dtype=np.float64).ravel()
    if vals.size == 0:
        raise ValueError("set_rule_margin of an empty set")
    return float(vals.min())


def _per_region_signed_dist(particle_xy_per_region: list[np.ndarray], regions) -> np.ndarray:
    """[R, N] polygon signed distances of the particles expressed in each region frame."""
    if len(particle_xy_per_region) != len(regions):
        raise ValueError("need one particle xy array per region")
    cols = []
    n_particles = None
    for pts, region in zip(particle_xy_per_region, regions):
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        if n_particles is None:
            n_particles = pts.shape[0]
        elif pts.shape[0] != n_particles:
            raise ValueError("particle count differs across region frames")
        sd = signed_dist_inside_polygon(pts[:, :2], region.polygon)
        z_cap = getattr(region, "z_cap", None)
        if z_cap is not None and pts.shape[1] >= 3:
            # capped column: inside iff inside the polygon AND below the cap
            sd = np.minimum(sd, float(z_cap) - pts[:, 2])
        cols.append(sd)
    return np.stack(cols, axis=0)
