"""Non-sim tests: main/item_envs/via_path.py — one continuous trajectory
through IK waypoints (user review 7: the step chain stopped every 5 cm)."""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.via_path import PHYSICS_DT, path_fractions, profiled_via_path, time_law


@pytest.mark.parametrize("v", [0.125, 0.325])
def test_hard_stop_cruises_and_brakes_within_stop_dist(v):
    L, da, ds = 0.27, 0.06, 0.02
    lam, dlam = time_law({"kind": "hard_stop", "length": L, "v": v, "accel_dist": da, "stop_dist": ds})
    s, vel = lam * L, dlam * L
    acc = np.diff(vel) / PHYSICS_DT
    assert lam[0] == 0.0 and lam[-1] == 1.0 and np.all(np.diff(lam) >= -1e-12)
    assert vel.max() == pytest.approx(v, rel=1e-3)
    # no intermediate stop: the speed never drops below v between the ramps
    cruise = (s > da + 1e-3) & (s < L - ds - 1e-3)
    assert cruise.any() and np.all(vel[cruise] >= v * (1 - 1e-6))
    assert -acc.min() == pytest.approx(0.75 * v * v / ds, rel=0.02)  # the brake
    assert acc.max() == pytest.approx(0.75 * v * v / da, rel=0.02)  # the gentler start


def test_minjerk_rest_to_rest_single_peak():
    lam, dlam = time_law({"kind": "minjerk", "duration": 2.0})
    assert lam[0] == 0.0 and lam[-1] == pytest.approx(1.0)
    assert dlam.max() * 2.0 == pytest.approx(1.875, rel=1e-3)
    k = int(np.argmax(dlam))  # one peak: monotone up, then down
    assert np.all(np.diff(dlam[: k + 1]) >= -1e-12) and np.all(np.diff(dlam[k:]) <= 1e-12)


def test_profiled_via_path_hits_waypoints_and_is_continuous():
    rng = np.random.RandomState(0)
    W = np.cumsum(rng.normal(0.0, 0.02, (7, 6)), axis=0)
    frac = path_fractions(np.linspace([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], 7))
    pos, vel = profiled_via_path(W, frac, {"kind": "minjerk", "duration": 1.5})
    assert np.allclose(pos[0], W[0]) and np.allclose(pos[-1], W[-1])
    assert np.abs(np.diff(pos, axis=0)).max() < 0.01  # no jumps between rows
    fd = np.diff(pos, axis=0) / PHYSICS_DT
    assert np.abs(fd - 0.5 * (vel[1:] + vel[:-1])).max() < 1e-2  # velocity rows match the path
    # passes (near) every interior waypoint
    lam, _ = time_law({"kind": "minjerk", "duration": 1.5})
    for k in range(1, 6):
        i = int(np.argmin(np.abs(lam - frac[k])))
        assert np.abs(pos[i] - W[k]).max() < 5e-3


def test_path_fractions_counts_pure_rotation():
    pts = np.zeros((4, 3))
    frac = path_fractions(pts, angles=np.array([0.1, 0.1, 0.1]))
    assert np.allclose(frac, [0.0, 1 / 3, 2 / 3, 1.0])
