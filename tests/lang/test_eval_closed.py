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
            "nominal_violated": nom_v, "nominal_success": [True, True], "task_success": succ, "n_ticks": 200}


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
    import os
    os.utime(tmp_path / "episodes_emb-e.jsonl", (1000, 1000))      # the s1 file is the newer one
    os.utime(tmp_path / "episodes_emb-e_s1.jsonl", (2000, 2000))
    rows = merged_rows(tmp_path)
    assert sorted((r["embodiment"], r["v"]) for r in rows) == [("e", 9), ("f", 2)]


def test_aggregate_ungrouped_all_runs():
    """Ungrouped stats: every filter run of a group, harm and benign together."""
    rows = [_row("gap_place", "harm", False, True, [True, True], succ=False, scene=0),
            _row("gap_place", "harm", True, True, [True, False], scene=1),
            _row("gap_place", "benign", False, False, [False, False], scene=0),
            _row("gap_place", "benign", False, True, [True, False], scene=1)]
    a = aggregate(rows)["pooled/train"]["all"]
    assert a["n_runs"] == 4 and abs(a["broken_nominal"] - 4 / 8) < 1e-9 and a["broken_filter"] == 0.25
    assert abs(a["reduction"] - (1 - 0.25 / 0.5)) < 1e-9 and a["any_intervention"] == 0.75
    assert abs(a["success_cost"] - (1.0 - 0.75)) < 1e-9
    assert "All runs, no grouping" in tables(aggregate(rows), "t")


def test_protect_pos_is_cross_fitted():
    """C3: a filter that does nothing (harm independent of the nominal runs, same probability) scores about 0
    on protect_rate_pos; selecting and measuring on the same runs gave about (1 - p)^2."""
    from main.lang.eval_closed import _protect_pos

    rng = np.random.default_rng(0)
    harm = []
    for sc in range(4000):
        p = 0.3
        harm.append({"nominal_violated": [bool(rng.random() < p), bool(rng.random() < p)],
                     "harm_named": bool(rng.random() < p), "task": "t", "embodiment": "e", "scene_idx": sc})
    assert abs(_protect_pos(harm)) < 0.08


def test_merged_rows_drop_filter_rows_of_other_filter_code(tmp_path):
    """C7: a filter row written under another filter code hash is left out of the merge."""
    from main.lang.eval_closed import filter_code_hash

    h, fh = env_code_hash(), filter_code_hash()
    rows = [{"key": ["t", 0, "train", "harm"], "embodiment": "e", "type": "filter", "code_hash": h, "filter_hash": fh},
            {"key": ["t", 1, "train", "harm"], "embodiment": "e", "type": "filter", "code_hash": h, "filter_hash": "old"},
            {"key": ["t", 0, "nominal", "-"], "embodiment": "e", "type": "nominal", "code_hash": h, "filter_hash": "old"}]
    (tmp_path / "episodes_emb-e.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    keys = sorted(tuple(r["key"]) for r in merged_rows(tmp_path))
    assert keys == [("t", 0, "nominal", "-"), ("t", 0, "train", "harm")]


def test_pick_rules_prefers_a_rule_eligible_everywhere():
    """C8: when one rule is eligible on every axis, every axis names it."""
    rec = {"rules": [{"rule_id": f"r{k}"} for k in range(6)]}
    des = {f"r{k}": "harm" for k in range(6)}
    picks = pick_rules(rec, des, 3, ["train", "heldout_word"], lambda r, ax: ax == "train" or r["rule_id"] == "r4")
    assert picks["train"]["harm"]["rule_id"] == picks["heldout_word"]["harm"]["rule_id"] == "r4"


def test_closed_loop_defaults_to_the_trained_embodiments(tmp_path):
    from main.lang.config import CANON_EMBODIMENTS, LangConfig
    from main.lang.eval_closed import trained_embodiments

    assert trained_embodiments(tmp_path) == CANON_EMBODIMENTS          # no config: every embodiment
    import dataclasses

    dataclasses.replace(LangConfig(), embodiment_choices=("ur5-wsg",)).save(tmp_path / "lang_config.json")
    assert trained_embodiments(tmp_path) == ("ur5-wsg",)


def test_exception_runs_are_scored_and_the_control_summarized():
    """Review round 3, B2 / B5: a run that threw mid-episode keeps its observed harm and interventions and counts
    as a task failure; only a run with no tick is left out (counted per task and kind); filter-off control rows
    are summarized against the nominal runs of their scenes."""
    ok = _row("gap_place", "benign", False, False, [False, False], scene=0)
    exc = {**_row("gap_place", "benign", False, True, [False, False], scene=1), "status": "exception"}
    dead = {**_row("gap_place", "benign", False, False, [False, False], scene=2), "status": "exception", "n_ticks": 0}
    S = aggregate([ok, exc, dead])
    g = S["pooled/train"]
    assert g["n_benign"] == 2 and g["false_alarm_k"] == 1        # the exception run's intervention counts
    assert S["_counts"]["not_run"] == 1 and S["_counts"]["not_run_by_task_kind"] == {"gap_place|benign": 1}
    assert S["_counts"]["exceptions_scored"] == 1
    assert g["all"]["success_cost"] > 0                           # the exception run is a task failure
    ctl = {"type": "control", "task": "gap_place", "embodiment": "ur5-wsg", "scene_idx": 0, "harm_named": True,
           "task_success": True, "n_ticks": 300, "intervention_rate": 0.0, "nominal_violated": [True, False],
           "nominal_success": [True, True], "nominal_ticks": [280, 320]}
    c = aggregate([ok, ctl])["_control"]
    assert c["scenes"] == 1 and c["harm_control"] == 1.0 and c["harm_nominal"] == 0.5 and c["ticks_nominal"] == 300
