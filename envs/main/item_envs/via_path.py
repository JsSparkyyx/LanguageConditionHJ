"""One continuous joint trajectory through a chain of IK waypoints.

The item tasks used to execute a long Cartesian move (T4 carry, T2 roll) as
a chain of short local plans, each with its own rest-to-rest min-jerk
profile: the payload stopped at every ~5 cm / 5 deg boundary ("move and
stop, move and stop", user review 7). Here the same waypoints (dense enough
that joint-space interpolation tracks the Cartesian path) are joined by one
C2 cubic spline over the Cartesian path fraction and played on ONE time law:

- ``minjerk``: rest-to-rest minimum jerk over ``duration`` s;
- ``hard_stop``: smooth start (velocity smoothstep over ``accel_dist`` m),
  cruise at ``v`` m/s, then a short brake (smoothstep over ``stop_dist``
  m): peak deceleration = 0.75 v^2 / stop_dist — the free surface of a
  carried bead layer tips by atan(a / g), so a fast nominal sloshes beads
  over a low rim at the stop while a slow one does not.

Pure numpy / scipy; no sim imports (unit-tested without a GPU).
"""

from __future__ import annotations

import numpy as np

PHYSICS_DT = 1.0 / 250.0


def path_fractions(points: np.ndarray, angles: np.ndarray | None = None, rot_radius: float = 0.1) -> np.ndarray:
    """Cumulative, normalised path parameter of waypoint poses: Cartesian
    distance plus ``rot_radius`` x rotation angle (a pure re-orientation
    still advances). ``points`` [n+1, 3]; ``angles`` [n] per-segment rad."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if angles is not None:
        seg = seg + rot_radius * np.abs(np.asarray(angles, dtype=np.float64))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return cum / cum[-1]


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def time_law(profile: dict, dt: float = PHYSICS_DT) -> tuple[np.ndarray, np.ndarray]:
    """Path fraction lam(t) in [0, 1] and dlam/dt sampled every dt (first
    row t = 0, last row exactly lam = 1). See module docstring for kinds."""
    kind = profile.get("kind", "minjerk")
    if kind == "hard_stop":
        L = float(profile["length"])
        v = float(profile["v"])
        da = float(profile.get("accel_dist", 0.05))
        ds = float(profile.get("stop_dist", 0.02))
        if L <= 1e-9 or v <= 1e-9:
            return np.array([0.0, 1.0]), np.zeros(2)
        if da + ds > L:  # short move: shrink both ramps proportionally
            f = L / (da + ds)
            da, ds = da * f, ds * f
        ta, td = 2.0 * da / v, 2.0 * ds / v
        tc = (L - da - ds) / v
        T = ta + tc + td
        t = np.arange(0.0, T, dt)
        t = np.append(t, T) if t[-1] < T - 1e-12 else t
        vel = np.where(t < ta, v * _smoothstep(t / ta),
                       np.where(t < ta + tc, v, v * (1.0 - _smoothstep((t - ta - tc) / td))))
        # distance = integral of the velocity (closed form per phase)
        def _int_ss(x):  # integral_0^x smoothstep = x^3 - x^4 / 2
            x = np.clip(x, 0.0, 1.0)
            return x ** 3 - 0.5 * x ** 4
        s = np.where(t < ta, v * ta * _int_ss(t / ta),
                     np.where(t < ta + tc, da + v * (t - ta),
                              da + v * tc + v * td * (np.clip((t - ta - tc) / td, 0.0, 1.0) - _int_ss((t - ta - tc) / td))))
        lam = np.clip(s / L, 0.0, 1.0)
        lam[-1] = 1.0
        return lam, vel / L
    T = max(float(profile.get("duration", 1.0)), 2.0 * dt)
    n = max(2, int(np.ceil(T / dt)) + 1)
    tau = np.linspace(0.0, 1.0, n)
    lam = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5
    dlam = (30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4) / T
    return lam, dlam


def profiled_via_path(q_waypoints, frac, profile: dict, dt: float = PHYSICS_DT):
    """Joint rows (position, velocity) [N, dof] through ``q_waypoints``
    [n+1, dof] (row 0 = the live joints) placed at path fractions ``frac``
    [n+1], timed by ``profile`` (time_law)."""
    from scipy.interpolate import CubicSpline

    W = np.asarray(q_waypoints, dtype=np.float64)
    lam_k = np.asarray(frac, dtype=np.float64)
    if W.ndim != 2 or len(W) != len(lam_k) or len(W) < 2:
        raise ValueError("q_waypoints [n+1, dof] and frac [n+1] must match, n >= 1")
    lam, dlam = time_law(profile, dt)
    if len(W) == 2:
        pos = W[0] + lam[:, None] * (W[1] - W[0])
        vel = dlam[:, None] * (W[1] - W[0])
    else:
        spline = CubicSpline(lam_k, W, axis=0, bc_type="natural")
        pos = spline(lam)
        vel = spline(lam, 1) * dlam[:, None]
    pos[0], pos[-1] = W[0], W[-1]
    vel[0] = 0.0
    vel[-1] = 0.0
    return pos, vel
