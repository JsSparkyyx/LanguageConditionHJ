"""Tests for scripts/item_envs/calibration_report.py (pure stdlib, no SAPIEN)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "item_envs" / "calibration_report.py"
spec = importlib.util.spec_from_file_location("calibration_report", SCRIPT)
calib = importlib.util.module_from_spec(spec)
sys.modules["calibration_report"] = calib  # dataclass() needs the module registered
spec.loader.exec_module(calib)


def _write_meta(root: Path, cell: str, task: str, emb: str, seed: int, *,
                host: str | None = None, success: bool = True,
                rules: list | None = None, violations: dict | None = None,
                min_margin: dict | None = None, first_t: dict | None = None,
                time_scale: float = 1.0, task_risk: dict | None = None):
    rules = rules if rules is not None else [{"rule_id": "inside|box_0|table_top"}]
    violations = violations or {}
    d = {
        "task": task, "host_task": host, "embodiment": emb, "seed": seed,
        "risk": {"time_scale": time_scale, "safe_time_scale": 0.6, "task": task_risk or {}},
        "rules": rules, "task_success": success, "error": None,
        "plan_failures": 0, "spawn_rejections": 0, "wall_time_s": 42.0,
        "violations": violations,
        "min_margin": min_margin if min_margin is not None else {r: 0.01 for r in violations},
        "first_violation_t": first_t if first_t is not None else {r: (1.5 if v else None) for r, v in violations.items()},
    }
    out = root / cell / task / emb / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(d))


def _t3_tree(root: Path, primary_rates: dict[str, float], success_low: float = 0.8):
    """One T3-like group; keys are cells, values the primary-rule violation rate."""
    for cell, rate in primary_rates.items():
        ts = calib.parse_cell(cell)["time_scale"]
        ov = calib.parse_cell(cell).get("overshoot", 0.0)
        n = 10
        n_v = int(round(rate * n))
        n_s = int(round((success_low if ts == 0.5 else 0.6) * n))
        for seed in range(n):
            violated = seed < n_v
            _write_meta(
                root, cell, "push_zone", "ur5-wsg", seed,
                success=seed < n_s,
                violations={"inside|box_0|table_top": violated},
                min_margin={"inside|box_0|table_top": -0.01 if violated else 0.02},
                time_scale=ts, task_risk={"overshoot": ov},
            )


def test_parse_cell():
    assert calib.parse_cell("") == {}
    assert calib.parse_cell("time_scale=1.0_overshoot=0.04") == {"time_scale": 1.0, "overshoot": 0.04}
    assert calib.parse_cell("time_scale=0.6_on_path_count=2") == {"time_scale": 0.6, "on_path_count": 2}


def test_load_and_aggregate(tmp_path):
    _t3_tree(tmp_path, {"time_scale=0.5_overshoot=0.0": 0.1, "time_scale=1.3_overshoot=0.08": 0.4})
    eps = calib.load_episodes(tmp_path)
    assert len(eps) == 20
    aggs = calib.aggregate(eps)
    assert [a.cell for a in aggs] == ["time_scale=0.5_overshoot=0.0", "time_scale=1.3_overshoot=0.08"]
    low, high = aggs
    assert low.n == 10 and low.primary_rate == pytest.approx(0.1)
    assert high.primary_rate == pytest.approx(0.4)
    assert low.primary_rule == "inside|box_0|table_top"
    assert low.success_rate == pytest.approx(0.8)


def test_evaluate_gates_and_proposal(tmp_path):
    _t3_tree(tmp_path, {
        "time_scale=0.5_overshoot=0.0": 0.0,
        "time_scale=1.0_overshoot=0.04": 0.3,
        "time_scale=1.3_overshoot=0.08": 0.9,
    })
    aggs = calib.aggregate(calib.load_episodes(tmp_path))
    ev = calib.evaluate_group(aggs)
    assert ev["gates"]["low_cell"] == "time_scale=0.5_overshoot=0.0"
    assert ev["gates"]["low_cell_success_ok"] is True
    # first single cell inside the [0.2, 0.5] band wins
    assert ev["proposal"]["cells"] == ["time_scale=1.0_overshoot=0.04"]
    assert ev["proposal"]["primary_rate"] == pytest.approx(0.3)
    snippet = calib.risk_collect_yaml(aggs, ev["proposal"])
    assert "risk_collect:" in snippet and "time_scale: 1.0" in snippet and "overshoot: 0.04" in snippet


def test_no_proposal_when_all_out_of_band(tmp_path):
    _t3_tree(tmp_path, {"time_scale=0.5_overshoot=0.0": 0.05, "time_scale=1.3_overshoot=0.08": 0.95},
             success_low=0.5)
    aggs = calib.aggregate(calib.load_episodes(tmp_path))
    ev = calib.evaluate_group(aggs)
    assert ev["gates"]["low_cell_success_ok"] is False  # 0.5 < 0.7
    # singles 0.05/0.95 and the all-cell mixture 0.5 are all outside (0.2, 0.5) exclusive-ish band edges
    assert ev["proposal"] is None or ev["proposal"]["primary_rate"] == 0.5


def test_coviolation_gate_for_avoid_rules(tmp_path):
    # T1-like: three avoid rules; co-violation (>= 2 violated) rate 0.1 < 0.3 gate
    cell = "time_scale=1.15_on_path_count=2"
    rules = [{"rule_id": f"avoid|obj_{k}|-"} for k in range(3)]
    for seed in range(10):
        if seed == 0:
            v = {"avoid|obj_0|-": True, "avoid|obj_1|-": True, "avoid|obj_2|-": False}
        elif seed < 4:
            v = {"avoid|obj_0|-": True, "avoid|obj_1|-": False, "avoid|obj_2|-": False}
        else:
            v = {"avoid|obj_0|-": False, "avoid|obj_1|-": False, "avoid|obj_2|-": False}
        _write_meta(tmp_path, cell, "selective_avoid", "ur5-wsg", seed, host="stack_blocks_two",
                    rules=rules, violations=v, time_scale=1.15, task_risk={"on_path_count": 2})
    aggs = calib.aggregate(calib.load_episodes(tmp_path))
    (agg,) = aggs
    assert agg.primary_rule == "avoid|obj_0|-"
    assert agg.primary_rate == pytest.approx(0.4)
    assert agg.any_avoid_rate == pytest.approx(0.4)
    assert agg.coviol_rate == pytest.approx(0.1)
    ev = calib.evaluate_group(aggs)
    # 0.4 is in the band but co-violation 0.1 < 0.3 -> no proposal
    assert ev["proposal"] is None
    md = calib.render_markdown(aggs, {("selective_avoid", "stack_blocks_two", "ur5-wsg"): ev})
    assert "NO mixture meets" in md and "selective_avoid on stack_blocks_two" in md


def test_main_end_to_end(tmp_path, capsys):
    _t3_tree(tmp_path, {"time_scale=1.0_overshoot=0.04": 0.3})
    rc = calib.main([str(tmp_path), "--md", str(tmp_path / "out.md")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "push_zone" in out and "time_scale=1.0_overshoot=0.04" in out
    assert (tmp_path / "out.md").exists()


def test_main_empty_root(tmp_path):
    assert calib.main([str(tmp_path)]) == 1
