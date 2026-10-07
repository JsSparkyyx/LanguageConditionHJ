"""eval_closed metrics and aggregates on synthetic traces / rows."""

import numpy as np

from main.lang.eval_closed import aggregate, hsel, run_metrics, tables


def test_run_metrics():
    t = np.arange(10) * 0.04
    trace = {"t": t.tolist(), "h_named": [0.5, 0.4, 0.3, 0.1, -0.1, -0.2, 0.1, 0.3, 0.4, 0.5],
             "V_t": t.tolist(), "V": [0.4] * 3 + [-0.1] * 7, "intervened": [False] * 3 + [True] * 7,
             "h_other_min": [0.9] * 10, "Q_cf": [0.6] * 10, "status": "ok", "success": True,
             "intervention_rate": 0.7}
    nom = {"runs": [{"first_violation_t": {"r": 0.5}}, {"first_violation_t": {"r": 0.3}}]}
    m = run_metrics(trace, nom, "r", "harm")
    assert m["harm_named"] and m["any_intervention"] and not m["harm_other_any"]
    assert abs(m["first_intervention_t"] - 0.12) < 1e-9
    assert abs(m["lead_time"] - (0.4 - 0.12)) < 1e-9
    assert abs(m["q_cf_gap"] - 0.2) < 1e-9


def _row(task, kind, harm, interv, nom_v, succ=True):
    return {"type": "filter", "task": task, "embodiment": "ur5-wsg", "axis": "train", "kind": kind,
            "harm_named": harm, "any_intervention": interv, "intervention_rate": 0.1 if interv else 0.0,
            "nominal_violated": nom_v, "nominal_success": [True, True], "task_success": succ}


def test_aggregate_rates():
    rows = [_row("gap_place", "harm", False, True, [True, True]),
            _row("gap_place", "harm", True, True, [True, False]),
            _row("gap_place", "benign", False, False, [False, False]),
            _row("gap_place", "benign", False, True, [False, False]),
            {"type": "filter", "missing": True}]
    S = aggregate(rows)
    p = S["pooled/train"]
    assert abs(p["harm_rate_nominal"] - 0.75) < 1e-9 and p["harm_rate_filter"] == 0.5
    assert abs(p["protect_rate"] - (1 - 0.5 / 0.75)) < 1e-9
    assert p["benign_false_alarm"] == 0.5 and abs(p["nominal_disagreement"] - 0.25) < 1e-9
    assert S["_counts"]["missing"] == 1
    assert "pooled/train" in tables(S, "t")


def test_hsel_deterministic():
    assert hsel(["a", "b", "c"], (1, "x")) == hsel(["a", "b", "c"], (1, "x"))
    assert hsel([], 1) is None
