"""Non-sim tests: T5 tray_transport rule table + task-file/yaml contract.

Mock scene mirrors robotwin_tasks/tray_transport.py (tray task_object,
2-3 standing cylinders as candidates); rules are enumerated from the REAL
T5 yaml + objects.yaml + splits.yaml. The task file's module constants are
pinned to the yaml; the speed physics (a_tip vs the brake) is checked on
the configured ranges.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import yaml

from main.item_envs.items import ItemSpec
from main.item_envs.registry import get_task_entry
from main.item_envs.rules import enumerate_rules, resolve_splits

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "main" / "item_envs" / "configs"
TASK_SRC = ROOT / "robotwin_tasks" / "tray_transport.py"


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _const(name):
    src = TASK_SRC.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", src, re.M)
    assert m, name
    return ast.literal_eval(re.sub(r"\s+#.*$", "", m.group(1)))


def _f(v):
    if isinstance(v, dict):
        return {k: _f(x) for k, x in v.items()}
    return tuple(_f(x) for x in v) if isinstance(v, (list, tuple)) else float(v)


def _rules(n_items=3):
    items = {"tray": ItemSpec(item_id="tray", entity=object(), category="tray", color="", role="task_object")}
    for k, c in enumerate(["red", "green", "blue"][:n_items]):
        items[f"block_{k}"] = ItemSpec(item_id=f"block_{k}", entity=object(), category="box", color=c,
                                     role="candidate")
    return enumerate_rules(items, {}, _load("tasks/T5_tray_transport.yaml"), objects_cfg=_load("objects.yaml"),
                           splits_cfg=resolve_splits(_load("splits.yaml"), "tray_transport"))


def test_registry_entry():
    entry = get_task_entry("tray_transport")
    assert entry.robotwin_task == "tray_transport"
    assert entry.config.name == "T5_tray_transport.yaml"


def test_rule_table_upright_per_block():
    for n in (2, 3):
        rules = _rules(n)
        assert [r.rule_id for r in rules] == [f"upright|block_{k}|-" for k in range(n)]
        for r in rules:
            assert r.phrases and not any(ch.isdigit() for p in r.phrases for ch in p)


def test_primary_rule_id_format_matches_rule_table():
    src = TASK_SRC.read_text(encoding="utf-8")
    assert 'self.primary_rule_id = f"upright|{ref[\'item_id\']}|-"' in src
    assert "upright|block_0|-" in [r.rule_id for r in _rules(2)]


def test_constants_mirror_yaml():
    cfg = _load("tasks/T5_tray_transport.yaml")
    assert _f(_const("START_ABS_X")) == _f(cfg["start"]["abs_x"])
    assert _f(_const("START_Y")) == _f(cfg["start"]["y"])
    assert _f(_const("TARGET_ABS_X")) == _f(cfg["target"]["abs_x"])
    assert _f(_const("TARGET_Y")) == _f(cfg["target"]["y"])
    assert _f(_const("MIN_CARRY_DIST")) == _f(cfg["target"]["min_dist"])
    assert _f(_const("PAD_SIZE")) == _f(cfg["target"]["pad_size"])
    t = cfg["tray"]
    assert _f(_const("TRAY_SIZE")) == _f(t["size"])
    assert _f(_const("TRAY_FLOOR_T")) == _f(t["floor_t"])
    assert _f(_const("TRAY_RIM_H")) == _f(t["rim_h"])
    assert _f(_const("TRAY_RIM_T")) == _f(t["rim_t"])
    assert _f(_const("HANDLE_LEN")) == _f(t["handle"]["len"])
    assert _f(_const("HANDLE_T")) == _f(t["handle"]["t"])
    assert _f(_const("HANDLE_H")) == _f(t["handle"]["h"])
    assert _f(_const("TRAY_MASS_RANGE")) == _f(t["mass"])
    assert _f(_load("objects.yaml")["categories"]["tray"]["mass_range"]) == _f(t["mass"])
    it = cfg["items"]
    for kind, (wk, hk) in {"tall": ("TALL_W", "TALL_H"), "short": ("SHORT_W", "SHORT_H"),
                           "mid": ("MID_W", "MID_H")}.items():
        assert _f(_const(wk)) == _f(it[kind]["w"]) and _f(_const(hk)) == _f(it[kind]["h"])
    assert _f(_const("THIRD_ITEM_P")) == _f(it["third_item_p"])
    assert _f(_const("ITEM_MASS_RANGE")) == _f(it["mass"])
    assert _f(_load("objects.yaml")["categories"]["box"]["mass_range"]) == _f(it["mass"])
    assert _f(_const("ITEM_EDGE_CLEAR")) == _f(it["edge_clear"])
    assert _f(_const("ITEM_GAP")) == _f(it["gap"])
    assert _f(_const("HANDLE_CLEAR")) == _f(it["handle_clear"])
    assert tuple(_const("ITEM_COLORS")) == tuple(it["colors"])
    assert _f(_const("REFERENT_P")) == _f(cfg["referent"])
    m = cfg["motion"]
    assert _f(_const("TRAY_SPEED")) == _f(m["speed"])
    assert _f(_const("TRAY_ACCEL_DIST")) == _f(m["accel_dist"])
    assert _f(_const("TRAY_STOP_DIST")) == _f(m["stop_dist"])
    assert _f(_const("SAFE_SPEED")) == _f(m["safe_speed"])
    assert _f(_const("GRASP_TIP_CLEAR")) == _f(m["tip_clear"])
    assert _f(_const("GRIPPER_STIFFNESS")) == _f(m["gripper_stiffness"])
    assert _f(_const("SUCCESS_TOL")) == _f(cfg["success_tol"])
    assert _f(_const("UPRIGHT_MAX_DEG")) == _f(cfg["rules"][0]["theta_max_deg"])


def test_speed_is_the_hazard_on_the_configured_ranges():
    """The brake's peak deceleration 0.75 v^2 / stop_dist must exceed every
    tall block's a_tip at the fastest draw and stay under it (with 2x margin
    for stick-slip jolts, measured 1.3-1.6 m/s^2 at the slowest, d68) at the
    slowest; the short ones never tip (a_tip = g w / h)."""
    cfg = _load("tasks/T5_tray_transport.yaml")
    g = 9.81
    v = cfg["motion"]["speed"] * np.asarray(cfg["risk"]["time_scale"]["values"])
    a_brake = 0.75 * v**2 / cfg["motion"]["stop_dist"]
    it = cfg["items"]
    tip = {k: (g * it[k]["w"][0] / it[k]["h"][1], g * it[k]["w"][1] / it[k]["h"][0]) for k in it
           if isinstance(it[k], dict) and "w" in it[k]}
    assert 2.0 * max(a_brake.min(), 1.25) < tip["tall"][0] + 0.1  # slowest draw (+ jolts): no tall block tips
    assert a_brake.max() > tip["tall"][1]  # fastest draw: every tall cylinder can tip
    assert a_brake.max() < tip["short"][0]  # short cylinders never tip from the brake
    # the start ramp is gentler than the brake (accel_dist > stop_dist)
    assert cfg["motion"]["accel_dist"] > cfg["motion"]["stop_dist"]


def test_task_file_never_imports_main():
    tree = ast.parse(TASK_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.startswith("main") for n in names), names
