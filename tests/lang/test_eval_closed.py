"""eval_closed metrics and aggregates on synthetic traces / rows."""

import json
from types import SimpleNamespace

import numpy as np

from main.lang.eval_closed import (aggregate, env_code_hash, hsel, merged_rows, pick_rules, run_metrics, tables,
                                  trace_violations)


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


def _row(task, kind, harm, interv, nom_v, succ=True, scene=0):
    return {"type": "filter", "task": task, "embodiment": "ur5-wsg", "axis": "train", "kind": kind, "scene_idx": scene,
            "harm_named": harm, "any_intervention": interv, "intervention_rate": 0.1 if interv else 0.0,
            "nominal_violated": nom_v, "nominal_success": [True, True], "task_success": succ}


def test_aggregate_rates():
    rows = [_row("gap_place", "harm", False, True, [True, True], scene=0),
            _row("gap_place", "harm", True, True, [True, False], scene=1),
            _row("gap_place", "benign", False, False, [False, False], scene=0),
            _row("gap_place", "benign", False, True, [False, False], scene=1),
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


def test_run_metrics_ticks_and_nominal_duration():
    """M7: intervened_tick_frac counts actor ticks; E6: harm within the nominal runs' length."""
    t = np.arange(10) * 0.04
    trace = {"t": t.tolist(), "h_named": [0.5] * 6 + [-0.1] * 4, "V_t": t.tolist(), "V": [0.2] * 10,
             "intervened": [False] * 10, "action_source": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]}
    nom = {"runs": [{"n_ticks": 5, "first_violation_t": {"r": 0.0}}, {"n_ticks": 6, "first_violation_t": {"r": None}}]}
    m = run_metrics(trace, nom, "r", "harm")
    assert abs(m["intervened_tick_frac"] - 0.3) < 1e-9
    assert m["harm_named"] and not m["harm_named_nomdur"]      # harm only after tick 6 (nominal ran 6 ticks)


def test_trace_violations_use_bead_counts(tmp_path):
    """B2: a negative margin with no bead out is not a violation (h by count); other rules by margin."""
    from main.item_envs.rules import Consequence, RuleSpec

    bead = RuleSpec("particles_inside|beads|bowl", Consequence.PARTICLES_INSIDE, ("beads",), ("bowl",), {})
    avoid = RuleSpec("avoid|cup|-", Consequence.AVOID, ("cup",), (), {"margin": 0.01})
    t = np.arange(4) * 0.04
    margins = np.array([[0.01, 0.05], [-0.002, 0.02], [-0.004, -0.01], [-0.01, 0.03]])
    counts = np.array([[0, -1], [0, -1], [0, -1], [3, -1]])
    np.savez(tmp_path / "trace.npz", t=t, margins=margins, violation_counts=counts)
    viol, first = trace_violations(SimpleNamespace(out_dir=tmp_path, rules=[bead, avoid]))
    assert viol == {bead.rule_id: True, avoid.rule_id: True}
    assert first[bead.rule_id] == t[3]            # the first tick with a bead out, not the first margin < 0
    assert first[avoid.rule_id] == t[2]


def test_pick_rules_same_rule_on_every_axis():
    """M5: where a rule is eligible on several axes, every axis names it."""
    rec = {"rules": [{"rule_id": f"r{k}"} for k in range(6)]}
    des = {f"r{k}": ("harm" if k % 2 else "benign") for k in range(6)}
    picks = pick_rules(rec, des, 7, ["train", "heldout_template", "heldout_word"],
                       lambda r, ax: not (ax == "heldout_word" and r["rule_id"] in ("r1", "r3")))
    assert picks["train"]["harm"] is picks["heldout_template"]["harm"]
    assert picks["train"]["benign"] is picks["heldout_template"]["benign"]
    assert picks["heldout_word"]["harm"]["rule_id"] not in ("r1", "r3")


def test_aggregate_intervals_and_selectivity_scenes():
    """E7: protect has a scene-bootstrap interval; pooled selectivity uses scenes with both kinds only."""
    rows = []
    for sc in range(8):
        rows.append(_row("gap_place", "harm", sc % 3 == 0, True, [True, sc % 2 == 0], scene=sc))
        rows.append(_row("gap_place", "benign", False, sc % 4 == 0, [False, False], scene=sc))
    rows.append(_row("granular_pour", "harm", False, True, [True, True], scene=0))   # harm only: not in pooled sel.
    S = aggregate(rows)
    p = S["pooled/train"]
    lo, hi = p["protect_ci"]
    assert lo <= p["protect_rate"] <= hi
    assert p["selectivity_scenes"] == 8 and S["gap_place/train"]["selectivity_scenes"] == 8
    assert p["n_harm_pos"] == 9


def test_merged_rows_dedupe_and_hash(tmp_path):
    """O8: one row per (embodiment, key), last written wins; rows under another code hash are dropped."""
    h = env_code_hash()
    a = [{"key": ["t", 0, "train", "harm"], "embodiment": "e", "v": 1, "code_hash": h},
         {"key": ["t", 0, "train", "harm"], "embodiment": "f", "v": 2, "code_hash": h},
         {"key": ["t", 1, "train", "harm"], "embodiment": "e", "v": 3, "code_hash": "old"}]
    (tmp_path / "episodes_emb-e.jsonl").write_text("".join(json.dumps(x) + "\n" for x in a))
    b = [{"key": ["t", 0, "train", "harm"], "embodiment": "e", "v": 9, "code_hash": h}]
    (tmp_path / "episodes_emb-e_s1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in b))
    rows = merged_rows(tmp_path)
    assert sorted((r["embodiment"], r["v"]) for r in rows) == [("e", 9), ("f", 2)]
