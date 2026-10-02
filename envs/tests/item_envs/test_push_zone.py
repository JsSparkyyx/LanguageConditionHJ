"""Non-sim tests for T3 push_zone (PLAN v3 §7 T3).

Two contracts, no SAPIEN:

1. ``configs/tasks/T3_push_zone.yaml`` + objects.yaml + splits.yaml against a
   synthetic push_zone scene (two push candidates, two bystanders, two zones):
   the rule table has one ``inside`` and one ``outside``-per-zone rule per push
   candidate and one ``upright`` + one ``stay`` rule per bystander, rule_ids
   are stable across enumerations, and phrases never contain digits.

2. Source-grep contract: the module constants in
   ``robotwin_tasks/push_zone.py`` (parsed with ast.literal_eval off single-
   line assignments) mirror the task yaml — the task file cannot import
   ``main.*``, so the yaml is the CEHJ-side copy of the same static config.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import enumerate_rules, resolve_splits

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "main" / "item_envs" / "configs"
TASK_SRC = ROOT / "robotwin_tasks" / "push_zone.py"

REQUIRED_KEYS = [
    "task", "robotwin_task", "embodiments", "n_push", "push_objects", "goal",
    "zones", "bystanders", "risk", "rules", "embodiment_overrides", "risk_collect",
]

EXPECTED_RULE_IDS = [
    "inside|push_0|table_top",
    "inside|push_1|table_top",
    "outside|push_0|zone_blue",
    "outside|push_0|zone_red",
    "outside|push_1|zone_blue",
    "outside|push_1|zone_red",
    "upright|bystander_0|-",
    "upright|bystander_1|-",
    "stay|bystander_0|-",
    "stay|bystander_1|-",
]


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _task_cfg():
    return _load("tasks/T3_push_zone.yaml")


def _const(name: str):
    """ast.literal_eval of a single-line module assignment in push_zone.py."""
    src = TASK_SRC.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", src, re.M)
    assert m, f"constant {name} not found as a single-line assignment in {TASK_SRC.name}"
    rhs = re.sub(r"\s+#.*$", "", m.group(1))  # strip trailing comment
    return ast.literal_eval(rhs)


def _norm(v):
    """Tuples <-> lists and ints -> floats for yaml/python comparison."""
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in v.items()}
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


def mock_push_scene():
    """push_0 (red prim box) + push_1 (tea box) candidates, two bystanders,
    a candidate outside the push group (decoy), table_top + two zones."""
    items = {
        "push_0": ItemSpec(item_id="push_0", entity=object(), category="box", color="red", role="candidate", group="push"),
        "push_1": ItemSpec(item_id="push_1", entity=object(), category="tea_box", color=None, role="candidate", group="push"),
        "bystander_0": ItemSpec(item_id="bystander_0", entity=object(), category="bottle", color=None, role="bystander"),
        "bystander_1": ItemSpec(item_id="bystander_1", entity=object(), category="cylinder", color="yellow", role="bystander"),
        "decoy_0": ItemSpec(item_id="decoy_0", entity=object(), category="box", color="green", role="candidate"),
    }
    table_poly = np.array([[-0.6, -0.35], [0.6, -0.35], [0.6, 0.35], [-0.6, 0.35]])
    zone_poly = np.array([[-0.07, -0.06], [0.07, -0.06], [0.07, 0.06], [-0.07, 0.06]])
    regions = {
        "table_top": Region(
            region_id="table_top", kind="surface", frame_fn=lambda: np.eye(4),
            polygon=table_poly, z=0.74, z_band=0.01, nameable=True,
        ),
        "zone_red": Region(
            region_id="zone_red", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=zone_poly, z=0.0, z_band=0.01, nameable=True, color="red",
        ),
        "zone_blue": Region(
            region_id="zone_blue", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=zone_poly, z=0.0, z_band=0.01, nameable=True, color="blue",
        ),
    }
    return items, regions


def _rules():
    items, regions = mock_push_scene()
    return enumerate_rules(
        items,
        regions,
        _task_cfg(),
        objects_cfg=_load("objects.yaml"),
        splits_cfg=resolve_splits(_load("splits.yaml"), "push_zone"),
    )


# --------------------------------------------------------------------- yaml ----


def test_t3_yaml_required_keys():
    cfg = _task_cfg()
    for key in REQUIRED_KEYS:
        assert key in cfg, f"missing key {key!r} in T3_push_zone.yaml"
    assert cfg["task"] == "push_zone"
    assert cfg["robotwin_task"] == "push_zone"
    assert cfg["embodiments"] == ["ur5-wsg", "franka-panda"]
    assert cfg["risk_collect"] is None
    risk = cfg["risk"]
    assert risk["time_scale"] == {"dist": "choice", "values": [0.5, 0.75, 1.0, 1.15, 1.3]}
    assert risk["safe_time_scale"] == 0.6
    assert risk["task"]["overshoot"] == {"dist": "uniform", "range": [0.0, 0.08]}
    assert risk["task"]["lateral_offset"] == {"dist": "uniform", "range": [-0.02, 0.02]}
    # play_once's knob defaults sit at the low end of the sampled ranges
    assert risk["task"]["overshoot"]["range"][0] <= 0.0 <= risk["task"]["overshoot"]["range"][1]
    assert risk["task"]["lateral_offset"]["range"][0] <= 0.0 <= risk["task"]["lateral_offset"]["range"][1]
    # the four plan rules entries, in order
    entries = cfg["rules"]
    assert [e["consequence"] for e in entries] == ["inside", "outside", "upright", "stay"]
    assert entries[0]["items"] == {"role": "candidate", "group": "push"}
    assert entries[0]["regions"] == ["table_top"]
    assert entries[1]["items"] == {"role": "candidate", "group": "push"}
    assert entries[1]["regions"] == {"kind": "zone"}
    assert entries[2]["items"] == {"role": "bystander"}
    assert entries[2]["theta_max_deg"] == 45
    assert entries[3]["items"] == {"role": "bystander"}
    assert entries[3]["d_tol"] == 0.03
    assert entries[3]["theta_tol_deg"] == 20


# ------------------------------------------------------------------- rules ----


def test_push_rule_table_order_and_ids():
    assert [r.rule_id for r in _rules()] == EXPECTED_RULE_IDS


def test_push_rule_ids_stable():
    first = [r.rule_id for r in _rules()]
    second = [r.rule_id for r in _rules()]
    assert first == second


def test_push_rule_semantics():
    rules = _rules()
    by_id = {r.rule_id: r for r in rules}
    for rid, rule in by_id.items():
        consequence, item_part, region_part = rid.split("|")
        assert rule.consequence.value == consequence
        assert item_part != "decoy_0"  # the non-push-group candidate gets no rules
        assert rule.phrases, f"{rid} rendered no phrases"
        for phrase in rule.phrases:
            assert not any(c.isdigit() for c in phrase), phrase
        assert rule.split in ("train", "heldout_item", "heldout_region", "heldout_phrase")
    # inside/outside only on push candidates; upright/stay only on bystanders
    for rid in EXPECTED_RULE_IDS:
        consequence, item_part, _ = rid.split("|")
        if consequence in ("inside", "outside"):
            assert item_part in ("push_0", "push_1")
        else:
            assert item_part in ("bystander_0", "bystander_1")
    assert by_id["upright|bystander_0|-"].params["theta_max"] == pytest.approx(np.deg2rad(45))
    stay = by_id["stay|bystander_1|-"]
    assert stay.params["d_tol"] == pytest.approx(0.03)
    assert stay.params["theta_tol"] == pytest.approx(np.deg2rad(20))
    # colored items name their color in the naming metadata
    assert by_id["inside|push_0|table_top"].params["color"] == "red"
    assert by_id["upright|bystander_1|-"].params["color"] == "yellow"
    # splits.yaml: tea_box is a heldout item category for push_zone
    assert by_id["inside|push_1|table_top"].split == "heldout_item"
    assert by_id["inside|push_0|table_top"].split == "train"


# ----------------------------------------------------------------- contract ----


def test_module_constants_mirror_yaml():
    cfg = _task_cfg()
    assert _norm(_const("N_PUSH")) == _norm(cfg["n_push"])
    assert _norm(_const("PUSH_OBJECT_POOL")) == _norm(cfg["push_objects"]["pool"])
    assert _norm(_const("PRIM_SIDE_XY")) == _norm(cfg["push_objects"]["side"])
    assert _norm(_const("PRIM_HEIGHT")) == _norm(cfg["push_objects"]["height"])
    assert _norm(_const("PUSH_START")) == _norm(cfg["push_objects"]["start"])
    mesh_ids = _const("MESH_MODEL_IDS")
    assert _norm({k: mesh_ids[k] for k in cfg["push_objects"]["ids"]}) == _norm(cfg["push_objects"]["ids"])
    assert _norm({k: mesh_ids[k] for k in cfg["bystanders"]["ids"]}) == _norm(cfg["bystanders"]["ids"])
    assert _norm(_const("GOAL_EDGE_DIST")) == _norm(cfg["goal"]["edge_dist"])
    assert _norm(_const("GOAL_EDGE_JITTER")) == _norm(cfg["goal"]["edge_jitter"])
    assert _norm(_const("GOAL_ZONE_DIST")) == _norm(cfg["goal"]["zone_dist"])
    assert _norm(_const("GOAL_ZONE_ANGLE_DEG")) == _norm(cfg["goal"]["zone_angle_deg"])
    assert _norm(_const("GOAL_ZONE_GAP")) == _norm(cfg["goal"]["zone_gap"])
    assert _norm(_const("N_ZONES")) == _norm(cfg["zones"]["n"])
    assert _norm(_const("ZONE_SIZE_XY")) == _norm(cfg["zones"]["size_xy"])
    assert _norm(_const("ZONE_NEAR_LINE_GAP")) == _norm(cfg["zones"]["near_push_line"])
    assert _norm(_const("ZONE_FREE_GAP")) == _norm(cfg["zones"]["free_gap"])
    assert _norm(_const("N_BYSTANDERS")) == _norm(cfg["bystanders"]["n"])
    assert _norm(_const("BYSTANDER_POOL")) == _norm(cfg["bystanders"]["pool"])
    assert _norm(_const("CYLINDER_RADIUS")) == _norm(cfg["bystanders"]["cylinder"]["radius"])
    assert _norm(_const("CYLINDER_HEIGHT")) == _norm(cfg["bystanders"]["cylinder"]["height"])
    assert _norm(_const("BYSTANDER_LINE_OFFSET")) == _norm(cfg["bystanders"]["line_offset"])
    assert _norm(_const("BYSTANDER_T")) == _norm(cfg["bystanders"]["t"])
    for emb in ("ur5-wsg", "franka-panda"):
        assert _norm(_const("PREPUSH_DIST")) == _norm(cfg["embodiment_overrides"][emb]["prepush_dist"])
        assert _norm(_const("PUSH_HEIGHT_OFFSET")) == _norm(cfg["embodiment_overrides"][emb]["push_height_offset"])
    # LAT_OFFSET_MAX keeps placements safe against the full lateral_offset range
    lat_range = cfg["risk"]["task"]["lateral_offset"]["range"]
    assert _const("LAT_OFFSET_MAX") == pytest.approx(max(abs(v) for v in lat_range))


def test_color_palette_matches_objects_yaml():
    objects_cfg = _load("objects.yaml")
    assert _norm(_const("TRAIN_COLORS")) == _norm(objects_cfg["colors"]["train"])
    for name in _const("TRAIN_COLORS"):
        assert _norm(_const("COLOR_RGB")[name]) == _norm(objects_cfg["color_rgb"][name])
