"""Non-sim tests for the probes package (PLAN v3 §6 Phase 3).

Resumable-row machinery (load_rows / append_row / done_keys / shard_cells),
probe discovery, and the PURE gate evaluators of p1 / p3 / p3_robot on
synthetic rows. No sapien, no GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from main.item_envs.probes import KNOWN, get_probe
from main.item_envs.probes import _common


# ---------------------------------------------------------------------------
# row log
# ---------------------------------------------------------------------------


def test_rows_roundtrip_and_done_keys(tmp_path):
    path = tmp_path / "deep" / "rows.jsonl"  # append_row mkdirs
    rows = [
        {"host": "a", "embodiment": "ur5-wsg", "seed": 0, "baseline": False, "v": 1},
        {"host": "a", "embodiment": "ur5-wsg", "seed": 0, "baseline": True, "v": 2},
    ]
    for r in rows:
        _common.append_row(path, r)
    assert _common.load_rows(path) == rows
    keys = _common.done_keys(path, ("host", "embodiment", "seed", "baseline"))
    assert keys == {("a", "ur5-wsg", 0, False), ("a", "ur5-wsg", 0, True)}
    # resume logic: a third append lands, both old keys still done
    _common.append_row(path, {"host": "a", "embodiment": "ur5-wsg", "seed": 1, "baseline": False})
    assert len(_common.load_rows(path)) == 3


def test_load_rows_drops_truncated_last_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"a": 1}\n{"a": 2}\n{"a": 3, "trun', encoding="utf-8"
    )
    assert _common.load_rows(path) == [{"a": 1}, {"a": 2}]


def test_load_rows_skips_corrupt_middle_line(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nNOT JSON\n{"a": 2}\n', encoding="utf-8")
    assert _common.load_rows(path) == [{"a": 1}, {"a": 2}]
    assert "corrupt mid-file line" in capsys.readouterr().out


def test_load_rows_missing_file(tmp_path):
    assert _common.load_rows(tmp_path / "nope.jsonl") == []


def test_jsonable_numpy():
    row = {
        "f": np.float64(1.5),
        "i": np.int64(3),
        "b": np.bool_(True),
        "arr": np.array([1.0, 2.0]),
        "nested": {"m": np.array([[1, 2], [3, 4]])},
        "inf": float("inf"),
    }
    text = json.dumps(_common._jsonable(row))
    back = json.loads(text)
    assert back == {"f": 1.5, "i": 3, "b": True, "arr": [1.0, 2.0],
                    "nested": {"m": [[1, 2], [3, 4]]}, "inf": float("inf")}


# ---------------------------------------------------------------------------
# shard_cells
# ---------------------------------------------------------------------------


def test_shard_cells_partitions():
    cells = list(range(10))
    assert _common.shard_cells(cells, "0/1") == cells
    a = _common.shard_cells(cells, "0/2")
    b = _common.shard_cells(cells, "1/2")
    assert a == [0, 2, 4, 6, 8]
    assert b == [1, 3, 5, 7, 9]
    assert sorted(a + b) == cells
    thirds = [_common.shard_cells(cells, f"{i}/3") for i in range(3)]
    assert sorted(sum(thirds, [])) == cells
    assert all(len(set(t)) == len(t) for t in thirds)
    # tuples keep their identity (strided by position, not by value)
    tcells = [("h", i) for i in range(6)]
    assert _common.shard_cells(tcells, "1/3") == [("h", 1), ("h", 4)]


# ---------------------------------------------------------------------------
# probe discovery
# ---------------------------------------------------------------------------


def test_known_probes():
    assert set(KNOWN) == {"p1", "p2", "p3", "p3_robot", "p4", "p5"}


def test_get_probe_imports_and_caches():
    m1 = get_probe("p3")
    assert m1 is get_probe("p3")  # cached
    assert callable(m1.run) and callable(m1.summarize)
    for name in ("p1", "p3_robot"):
        m = get_probe(name)
        assert callable(m.run) and callable(m.summarize), name
    with pytest.raises(KeyError):
        get_probe("p99")


# ---------------------------------------------------------------------------
# p3 gates (pure)
# ---------------------------------------------------------------------------


def _p3_rows(medians, seeds=(0, 1, 2)):
    rows = []
    for (v, mu), coast in medians.items():
        for s in seeds:
            rows.append({"v": v, "mu": mu, "seed": s, "coast_m": coast})
    return rows


def test_p3_gates_pass():
    p3 = get_probe("p3")
    rows = _p3_rows({
        (0.1, 0.3): 0.0017, (0.1, 0.6): 0.0008, (0.1, 0.9): 0.0006,
        (0.3, 0.3): 0.015, (0.3, 0.6): 0.0076, (0.3, 0.9): 0.005,
        (0.6, 0.3): 0.061, (0.6, 0.6): 0.031, (0.6, 0.9): 0.020,
    })
    gates = p3.evaluate_gates(rows)
    assert gates["pass"]
    assert gates["coast_positive"] and gates["increases_with_v"] and gates["decreases_with_mu"]
    assert gates["n_cells"] == 9


def test_p3_gates_detect_nonmonotone_v():
    p3 = get_probe("p3")
    rows = _p3_rows({
        (0.1, 0.6): 0.05, (0.3, 0.6): 0.01,  # decreases with v: violation
        (0.6, 0.6): 0.02,
    })
    gates = p3.evaluate_gates(rows)
    assert not gates["increases_with_v"]
    assert not gates["pass"]


def test_p3_gates_detect_nonmonotone_mu():
    p3 = get_probe("p3")
    rows = _p3_rows({
        (0.3, 0.3): 0.005, (0.3, 0.9): 0.02,  # increases with mu: violation
    })
    gates = p3.evaluate_gates(rows)
    assert not gates["decreases_with_mu"]
    assert not gates["pass"]


def test_p3_gates_detect_zero_coast():
    p3 = get_probe("p3")
    rows = _p3_rows({(0.1, 0.9): 0.0})
    gates = p3.evaluate_gates(rows)
    assert not gates["coast_positive"]
    assert not gates["pass"]


# ---------------------------------------------------------------------------
# p1 / p3_robot summarize on synthetic rows (pure)
# ---------------------------------------------------------------------------


def test_p1_summarize_gates(tmp_path):
    p1 = get_probe("p1")
    rows = []
    for seed in range(10):
        rows.append({"host": "h", "embodiment": "e", "seed": seed, "baseline": False,
                     "success": True, "error": None, "k": 4, "spawn_rejections": 0,
                     "n_within_2cm": 2 if seed < 6 else 0,
                     "n_touched": 1 if seed < 4 else 0})
        rows.append({"host": "h", "embodiment": "e", "seed": seed, "baseline": True,
                     "success": seed != 9, "error": None, "k": 0, "spawn_rejections": 0})
    for r in rows:
        _common.append_row(tmp_path / "p1_rows.jsonl", r)
    s = p1.summarize(tmp_path)
    assert s["near2_rate"] == pytest.approx(0.6)
    assert s["touch_rate"] == pytest.approx(0.4)
    assert s["baseline_success"] == pytest.approx(0.9)
    assert s["pass"]  # 0.6 >= 0.5, 0.4 >= 0.3, 0.9 >= 0.7
    assert (tmp_path / "summary.json").is_file()


def test_p1_summarize_failing_gates(tmp_path):
    p1 = get_probe("p1")
    for seed in range(10):
        _common.append_row(tmp_path / "p1_rows.jsonl",
                           {"host": "h", "embodiment": "e", "seed": seed, "baseline": False,
                            "success": True, "error": None, "k": 3, "spawn_rejections": 0,
                            "n_within_2cm": 0, "n_touched": 0})
    s = p1.summarize(tmp_path)
    assert not s["pass"]
    assert s["baseline_success"] is None  # no baseline rows -> gate cannot pass


def test_p1_summarize_error_rows_excluded(tmp_path):
    p1 = get_probe("p1")
    _common.append_row(tmp_path / "p1_rows.jsonl",
                       {"host": "h", "embodiment": "e", "seed": 0, "baseline": False,
                        "success": False, "error": "UnStableError: x", "k": 0,
                        "spawn_rejections": 0, "n_within_2cm": 0, "n_touched": 0})
    s = p1.summarize(tmp_path)
    assert s["n_error_rows"] == 1
    assert s["error_kinds"] == ["UnStableError"]
    assert s["n_object_episodes"] == 0


def test_p3_robot_gates(tmp_path):
    p3r = get_probe("p3_robot")
    rows = []
    for seed in range(10):
        rows.append({"embodiment": "ur5-wsg", "seed": seed, "time_scale": 0.5,
                     "error": None, "within_3cm": seed < 9, "abs_err_m": 0.01,
                     "plan_failures": 0})
        rows.append({"embodiment": "ur5-wsg", "seed": seed, "time_scale": 1.3,
                     "error": None, "within_3cm": seed < 4, "abs_err_m": 0.05,
                     "plan_failures": 0})
    for r in rows:
        _common.append_row(tmp_path / "p3_robot_rows.jsonl", r)
    s = p3r.summarize(tmp_path)
    assert s["cells"]["ur5-wsg/ts=0.5"]["frac_within_3cm"] == pytest.approx(0.9)
    assert s["cells"]["ur5-wsg/ts=1.3"]["frac_within_3cm"] == pytest.approx(0.4)
    assert s["pass"]  # gate is at ts=0.5 only
    # below the 0.8 bar at the gate scale -> fail
    for r in rows:
        r["time_scale"] = 0.5
        r["within_3cm"] = r["seed"] < 7
    (tmp_path / "p3_robot_rows.jsonl").unlink()
    for r in rows:
        if r["time_scale"] == 0.5:
            _common.append_row(tmp_path / "p3_robot_rows.jsonl", r)
    s2 = p3r.summarize(tmp_path)
    assert not s2["pass"]


def test_run_signatures_tolerate_extra_kwargs(tmp_path):
    """The orchestrator passes shared flags; probes must swallow them."""
    p3 = get_probe("p3")
    out = p3.run(tmp_path / "p3", seeds=(0,), velocities=(0.3,), frictions=(0.6,),
                 render="none", video=False, shard="0/1")
    assert out["ran"] == 1
