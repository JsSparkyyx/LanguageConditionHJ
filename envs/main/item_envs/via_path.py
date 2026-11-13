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
- ``speed_change`` (2026-11-12, the tray-slide task): ``hard_stop`` with a
  second cruise speed: smooth start to ``v`` over ``accel_dist``, cruise,
  then from ``switch`` m along the path a smooth change to ``v2`` over
  ``change_dist`` m (peak deceleration 0.75 |v^2 - v2^2| / change_dist),
  cruise at ``v2``, brake within ``stop_dist``. Without ``v2`` (or with
  ``v2 == v``) it is exactly ``hard_stop``.

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
    if kind == "speed_change":
        v2 = profile.get("v2")
        if v2 is None or abs(float(v2) - float(profile["v"])) < 1e-9:
            return time_law({**profile, "kind": "hard_stop"}, dt)
        return _speed_change_law(profile, dt)
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


def speed_change_phases(profile: dict) -> dict:
    """The distances (m) of a ``speed_change`` profile after clamping: accel, switch (where the change starts),
    change, stop, and the speeds v, v2. Ramps that do not fit the length shrink proportionally (as hard_stop's)."""
    L = float(profile["length"])
    v, v2 = float(profile["v"]), float(profile["v2"])
    da = float(profile.get("accel_dist", 0.05))
    dc = float(profile.get("change_dist", 0.05))
    ds = float(profile.get("stop_dist", 0.02))
    if da + dc + ds > L:
        f = L / (da + dc + ds)
        da, dc, ds = da * f, dc * f, ds * f
    sw = float(np.clip(float(profile.get("switch", da)), da, L - dc - ds))
    return {"length": L, "v": v, "v2": v2, "accel_dist": da, "switch": sw, "change_dist": dc, "stop_dist": ds}


def _speed_change_law(profile: dict, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity piecewise in time (smoothstep ramps, as hard_stop), distance by closed-form integrals per phase."""
    p = speed_change_phases(profile)
    L, v, v2 = p["length"], p["v"], p["v2"]
    if L <= 1e-9 or v <= 1e-9 or v2 <= 1e-9:
        return np.array([0.0, 1.0]), np.zeros(2)
    da, sw, dc, ds = p["accel_dist"], p["switch"], p["change_dist"], p["stop_dist"]
    ta = 2.0 * da / v                       # 0 -> v (mean speed v / 2)
    t1 = (sw - da) / v                      # cruise at v
    tc = 2.0 * dc / (v + v2)                # v -> v2 (mean speed (v + v2) / 2)
    t2 = (L - sw - dc - ds) / v2            # cruise at v2
    td = 2.0 * ds / v2                      # v2 -> 0
    e = np.cumsum([0.0, ta, t1, tc, t2, td])   # phase boundaries in time
    T = float(e[-1])
    t = np.arange(0.0, T, dt)
    t = np.append(t, T) if t[-1] < T - 1e-12 else t

    def _int_ss(x):  # integral_0^x smoothstep = x^3 - x^4 / 2
        x = np.clip(x, 0.0, 1.0)
        return x ** 3 - 0.5 * x ** 4

    def ramp(t0, dur, a, b):  # velocity and distance (from the ramp start) of a smoothstep a -> b
        x = np.clip((t - t0) / dur, 0.0, 1.0)
        return a + (b - a) * _smoothstep(x), a * dur * x + (b - a) * dur * _int_ss(x)

    va, sa = ramp(e[0], ta, 0.0, v)
    vc, sc = ramp(e[2], tc, v, v2)
    vd, sd = ramp(e[4], td, v2, 0.0)
    vel = np.select([t < e[1], t < e[2], t < e[3], t < e[4]], [va, v, vc, v2], vd)
    s = np.select([t < e[1], t < e[2], t < e[3], t < e[4]],
                  [sa, da + v * (t - e[1]), sw + sc, sw + dc + v2 * (t - e[3])], L - ds + sd)
    lam = np.clip(s / L, 0.0, 1.0)
    lam[-1] = 1.0
    return lam, vel / L


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
