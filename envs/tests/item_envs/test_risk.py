"""Pure-numpy tests for main.item_envs.risk."""

from __future__ import annotations

import numpy as np

from main.item_envs.risk import RiskParams, sample_risk

TASK_CFG = {
    "risk": {
        "time_scale": {"dist": "choice", "values": [0.5, 0.75, 1.0, 1.15, 1.3]},
        "safe_time_scale": 0.6,
        "task": {
            "stop_abruptly": {"dist": "choice", "values": [0, 1]},
            "tilt_rate": {"dist": "uniform", "range": [1.0, 4.0]},
            "note": "constant",
        },
    }
}


def test_defaults():
    rp = RiskParams()
    assert rp.time_scale == 1.0
    assert rp.safe_time_scale == 0.6
    assert rp.task == {}
    assert rp.as_dict() == {"time_scale": 1.0, "safe_time_scale": 0.6, "task": {}}


def test_sample_risk_value_sets_and_ranges():
    rng = np.random.RandomState(0)
    for _ in range(200):
        rp = sample_risk(TASK_CFG, rng)
        assert rp.time_scale in {0.5, 0.75, 1.0, 1.15, 1.3}
        assert rp.safe_time_scale == 0.6  # plain scalar passes through
        assert rp.task["stop_abruptly"] in {0, 1}
        assert 1.0 <= rp.task["tilt_rate"] <= 4.0
        assert rp.task["note"] == "constant"
        # plain python types (JSON-safe)
        assert type(rp.time_scale) is float
        assert type(rp.task["stop_abruptly"]) is int
        assert type(rp.task["tilt_rate"]) is float


def test_sample_risk_deterministic_per_seed():
    a = sample_risk(TASK_CFG, np.random.RandomState(42))
    b = sample_risk(TASK_CFG, np.random.RandomState(42))
    assert a.as_dict() == b.as_dict()


def test_sample_risk_empty_and_partial_cfg():
    rng = np.random.RandomState(1)
    assert sample_risk({}, rng).as_dict() == RiskParams().as_dict()
    rp = sample_risk({"risk": {"time_scale": {"dist": "uniform", "range": [0.4, 0.4]}}}, rng)
    assert rp.time_scale == 0.4
    assert rp.safe_time_scale == 0.6
    assert rp.task == {}


def test_as_dict_roundtrip_types():
    rp = sample_risk(TASK_CFG, np.random.RandomState(7))
    d = rp.as_dict()
    assert set(d) == {"time_scale", "safe_time_scale", "task"}
    assert all(type(v) in (float, int, str) for v in d["task"].values())
