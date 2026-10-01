"""Pure-numpy tests for main.item_envs.retime."""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.retime import retime


def make_plan(T=101, n=6, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, T)[:, None]
    freq = rng.uniform(0.5, 3.0, size=(1, n))
    phase = rng.uniform(0, np.pi, size=(1, n))
    position = np.sin(2 * np.pi * freq * t + phase)
    velocity = 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
    return position, velocity


def test_identity_s1():
    position, velocity = make_plan()
    pos, vel, info = retime(position, velocity, 1.0)
    assert np.array_equal(pos, position)
    assert np.array_equal(vel, velocity)
    assert info == {"s": 1.0, "T_in": len(position), "T_out": len(position), "v_ratio": None}


@pytest.mark.parametrize("s", [0.3, 0.7, 1.3, 2.0])
def test_endpoints_preserved(s):
    position, velocity = make_plan()
    pos, vel, info = retime(position, velocity, s)
    assert np.allclose(pos[0], position[0], atol=1e-12)
    assert np.array_equal(pos[-1], position[-1])
    K = int(np.floor((len(position) - 1) / s))
    assert info["T_out"] == K + 1
    assert info["T_in"] == len(position)
    assert info["s"] == s


@pytest.mark.parametrize("s", [0.3, 0.7, 1.3, 2.0])
def test_duration_scales_inverse_s_within_one_row(s):
    position, velocity = make_plan()
    _, _, info = retime(position, velocity, s)
    rows_in = info["T_in"] - 1
    rows_out = info["T_out"] - 1
    assert abs(rows_out - rows_in / s) <= 1.0 + 1e-9


@pytest.mark.parametrize("s", [0.3, 0.7, 1.3, 2.0])
def test_velocity_scales_by_s(s):
    position, velocity = make_plan()
    pos, vel, info = retime(position, velocity, s)
    K = info["T_out"] - 1
    tau = np.arange(K) * s
    for k in (0, K // 2, K - 1):
        expected = np.stack([np.interp(tau[k], np.arange(len(velocity)), velocity[:, j]) for j in range(velocity.shape[1])])
        assert np.allclose(vel[k], s * expected, atol=1e-10)
        assert np.allclose(
            pos[k],
            [np.interp(tau[k], np.arange(len(position)), position[:, j]) for j in range(position.shape[1])],
            atol=1e-10,
        )
    # final row velocity is also scaled
    assert np.allclose(vel[-1], s * velocity[-1])


def test_v_ratio_reported_not_clipped():
    position, velocity = make_plan()
    v_limit = np.max(np.abs(velocity), axis=0) * 0.9  # below the input peak
    _, vel, info = retime(position, velocity, 2.0, v_limit=v_limit)
    assert info["v_ratio"] > 1.0
    assert np.isclose(info["v_ratio"], np.max(np.abs(vel) / v_limit[None, :]))
    # the velocity is NOT clipped to the limit
    assert np.max(np.abs(vel)) > np.max(v_limit)


def test_slow_then_fast_roundtrip_keeps_endpoint():
    position, velocity = make_plan(T=40)
    for s in (0.5, 1.7):
        pos, _, _ = retime(position, velocity, s)
        assert np.array_equal(pos[-1], position[-1])


def test_bad_inputs_raise():
    position, velocity = make_plan()
    with pytest.raises(ValueError):
        retime(position, velocity, 0.0)
    with pytest.raises(ValueError):
        retime(position, velocity, -1.0)
    with pytest.raises(ValueError):
        retime(position, velocity[:, :-1], 1.0)
    with pytest.raises(ValueError):
        retime(position[:1], velocity[:1], 1.0)
    with pytest.raises(ValueError):
        retime(position, velocity, 1.0, v_limit=np.zeros(position.shape[1]))
