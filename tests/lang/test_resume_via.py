"""ResumeViaController's pure parts (main/lang/controller.py): what is left of a via path, and its time law."""

import numpy as np
import pytest

from main.item_envs.via_path import PHYSICS_DT
from main.lang.controller import (ResumeViaController, controller_for, path_speed, remaining_via, resume_law,
                                  resume_segments, retime)
from main.lang.config import LangConfig

Q = [1.0, 0.0, 0.0, 0.0]
MEDIUM = {"kind": "hard_stop", "length": 0.36, "v": 0.45, "accel_dist": 0.15, "stop_dist": 0.04}
RECOVER = {"kind": "speed_change", "length": 0.36, "v": 0.45, "accel_dist": 0.15, "stop_dist": 0.04, "v2": 0.15,
           "switch": 0.15, "change_dist": 0.12}


def straight(n=12, L=0.36):
    """The T8 drag: n waypoints on a straight line from x = 0 (the start, not a waypoint) to x = L."""
    poses = [[L * k / n, 0.0, 0.8] + Q for k in range(1, n + 1)]
    return poses[:-1], poses[-1]


def accel(s, dt=PHYSICS_DT):
    v = np.diff(s) / dt
    return np.diff(v) / dt


def test_remaining_via_keeps_what_is_ahead():
    via, target = straight()
    kept, R = remaining_via([0.20, 0.0, 0.8], via, target)
    assert [round(p[0], 3) for p in kept] == [0.21, 0.24, 0.27, 0.30, 0.33]
    assert R == pytest.approx(0.16)
    kept, R = remaining_via([0.0, 0.0, 0.8], via, target)          # before the first waypoint: the whole path
    assert len(kept) == len(via) - 1 and R == pytest.approx(0.36)
    kept, R = remaining_via([0.35, 0.01, 0.8], via, target)        # past the last waypoint: straight to the target
    assert kept == [] and R == pytest.approx(np.hypot(0.01, 0.01))
    kept, R = remaining_via([0.10, 0.0, 0.8], [], target)          # a move without via poses
    assert kept == [] and R == pytest.approx(0.26)


@pytest.mark.parametrize("R, v0", [(0.30, 0.0), (0.16, 0.20), (0.16, 0.45), (0.06, 0.0), (0.06, 0.30), (0.02, 0.40)])
def test_resume_law_medium(R, v0):
    s = resume_law(R, v0, MEDIUM, MEDIUM["length"] - R)
    assert s[0] == 0.0 and s[-1] == pytest.approx(R) and (np.diff(s) >= -1e-12).all()
    v = np.diff(s) / PHYSICS_DT
    assert v[0] == pytest.approx(v0, abs=0.02)                     # starts at the arm's speed, not from rest
    assert v.max() <= max(0.45, v0) + 0.01                         # never faster than the behaviour's cruise
    a = accel(s[:-1])                                              # (the last sample closes the path)
    a_acc, a_stop = 0.75 * 0.45 ** 2 / 0.15, 0.75 * 0.45 ** 2 / 0.04
    if 0.75 * v0 ** 2 / a_stop <= R:                               # the nominal's own brake fits in what is left
        assert a.max() <= a_acc * 1.05 + 0.05 and -a.min() <= a_stop * 1.05 + 0.05


def test_resume_reaches_the_cruise_speed_when_there_is_room():
    segs = resume_segments(0.30, 0.2, MEDIUM, 0.06)
    assert [k for k, *_ in segs] == ["ramp", "cruise", "ramp"]
    assert segs[0][1:3] == (0.2, 0.45) and segs[2][1:3] == (0.45, 0.0) and segs[2][3] == pytest.approx(0.04)
    assert segs[0][3] == pytest.approx(0.15 * (1 - (0.2 / 0.45) ** 2))     # the same peak acceleration as the start
    short = resume_segments(0.06, 0.0, MEDIUM, 0.30)                        # too short: a lower peak speed
    assert [k for k, *_ in short] == ["ramp", "ramp"] and short[0][2] < 0.45
    assert sum(x[3] for x in short) == pytest.approx(0.06)


def test_resume_of_a_recovery_profile():
    before = resume_segments(0.33, 0.1, RECOVER, 0.03)             # released before its slow-down: it still slows
    assert [x[2] for x in before] == [0.45, 0.45, 0.15, 0.15, 0.0]
    assert sum(x[3] for x in before) == pytest.approx(0.33)
    tight = resume_segments(0.30, 0.1, RECOVER, 0.06)              # no room left for speed-up + slow-down: second speed
    assert max(x[2] for x in tight) == pytest.approx(0.15)
    after = resume_segments(0.10, 0.1, RECOVER, 0.26)              # released after it: on at the second speed
    assert max(x[2] for x in after) == pytest.approx(0.15) and sum(x[3] for x in after) == pytest.approx(0.10)


def test_retime_and_path_speed():
    lam = np.linspace(0, 1, 501) ** 2                              # any monotone sampling of the geometry
    pos = np.stack([0.3 * lam, -0.1 * lam], 1)                     # a straight joint path
    R = 0.2
    s = resume_law(R, 0.2, MEDIUM, 0.16)
    new, vel = retime(pos, lam, s, R)
    assert np.allclose(new[0], pos[0]) and np.allclose(new[-1], pos[-1]) and np.allclose(vel[-1], 0)
    assert np.allclose(new[:, 0], 0.3 * s / R, atol=1e-4)          # the rows follow the distance law
    assert path_speed(pos, lam, np.array([0.3, -0.1]) * (0.2 / R), R) == pytest.approx(0.2, rel=0.05)
    assert path_speed(pos, lam, -np.array([0.3, -0.1]), R) == 0.0  # moving backward counts as standing


def test_controller_choice():
    import dataclasses

    from main.item_envs.controller import RiskTickController

    assert controller_for(LangConfig()) is RiskTickController
    assert controller_for(dataclasses.replace(LangConfig(), resume_via_replan=True)) is ResumeViaController


def test_capped_finish_never_exceeds_the_cap():
    """d_*_finish (2026-11-18): the resumed drag cruises at most at the cap, with the original peak acceleration and a
    brake over the original stop distance, so it brakes gently; an arm handed over faster than the cap slows to it."""
    from main.lang.controller import capped_profile

    for prof in (MEDIUM, RECOVER):
        cp = capped_profile(prof, 0.15)
        assert cp["v"] == pytest.approx(0.15) and cp["kind"] == "hard_stop" and "v2" not in cp
        for R, v0 in ((0.20, 0.0), (0.20, 0.10), (0.12, 0.30), (0.03, 0.05)):
            segs = resume_segments(R, v0, cp, 0.15)
            assert max(max(a, b) for _, a, b, _ in segs) <= max(0.15, v0) + 1e-9
            assert sum(x[3] for x in segs) == pytest.approx(R)
            s = resume_law(R, v0, cp, 0.15)
            v = np.diff(s) / PHYSICS_DT
            assert v[2:].max() <= max(0.15, v0) * 1.02 and s[-1] == pytest.approx(R)
            if v0 <= 0.15 and R >= 0.1:                     # the brake: 0.75 v^2 / stop_dist, 0.42 m/s^2 for medium
                assert np.abs(accel(s)[-200:]).max() <= 0.75 * 0.15 ** 2 / prof["stop_dist"] * 1.05
    fast = resume_segments(0.20, 0.30, capped_profile(MEDIUM, 0.15), 0.15)   # handed over at 0.30 m/s: down to 0.15
    assert fast[0][1:3] == (0.30, 0.15) and fast[1][1:3] == (0.15, 0.15)
