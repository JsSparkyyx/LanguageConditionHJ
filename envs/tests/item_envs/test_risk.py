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


COLLECT_CFG = {
    **TASK_CFG,
    "risk_collect": {
        "safe_time_scale": 0.55,
        "cells": [
            {"weight": 1.0, "time_scale": 1.0, "task": {"tilt_rate": 2.0}},
            {"weight": 3.0, "time_scale": 1.3, "task": {"tilt_rate": 4.0}},
        ],
    },
}


def test_risk_collect_cells_mixture_weights():
    rng = np.random.RandomState(0)
    draws = [sample_risk(COLLECT_CFG, rng).time_scale for _ in range(4000)]
    frac_fast = sum(1 for s in draws if s == 1.3) / len(draws)
    assert 0.70 <= frac_fast <= 0.80  # expected 0.75
    assert set(draws) == {1.0, 1.3}


def test_risk_collect_cell_pins_and_fallbacks():
    rng = np.random.RandomState(3)
    for _ in range(200):
        rp = sample_risk(COLLECT_CFG, rng)
        assert rp.safe_time_scale == 0.55  # from risk_collect, not the risk section
        assert rp.task["tilt_rate"] == (4.0 if rp.time_scale == 1.3 else 2.0)  # pinned by cell
        assert rp.task["stop_abruptly"] in {0, 1}  # unlisted knob still sampled from risk section
        assert rp.task["note"] == "constant"


def test_risk_collect_deterministic_per_seed():
    a = sample_risk(COLLECT_CFG, np.random.RandomState(11))
    b = sample_risk(COLLECT_CFG, np.random.RandomState(11))
    assert a.as_dict() == b.as_dict()


def test_risk_collect_null_or_empty_falls_back_to_risk_section():
    for collect in (None, {}, {"cells": []}, {"cells": None}):
        cfg = {**TASK_CFG, "risk_collect": collect}
        a = sample_risk(cfg, np.random.RandomState(5))
        b = sample_risk(TASK_CFG, np.random.RandomState(5))
        assert a.as_dict() == b.as_dict()


def test_risk_collect_invalid_weights_raise():
    for cells in (
        [{"weight": 0.0, "time_scale": 1.0}],
        [{"weight": -1.0, "time_scale": 1.0}, {"weight": 2.0, "time_scale": 1.1}],
    ):
        cfg = {**TASK_CFG, "risk_collect": {"cells": cells}}
        try:
            sample_risk(cfg, np.random.RandomState(0))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for cells={cells}")
