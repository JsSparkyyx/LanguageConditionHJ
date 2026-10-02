"""Non-sim tests: the T2 granular_pour rule table and task-file/config contract.

Mirrors the robotwin_tasks/granular_pour.py scene (one ``beads`` particle_set
candidate, bowl_src task_object, container interiors bowl_src_interior /
target_interior, zone mats zone_red / zone_blue) with mock items/regions and
enumerates rules from the REAL configs/tasks/T2_granular_pour.yaml +
objects.yaml + splits.yaml. Also pins the task-file module constants against
the yaml (they are two copies of the same numbers by design — the task file
owns the layout, the yaml documents it — and this test is the drift alarm).
"""

from __future__ import annotations

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
TASK_SRC = ROOT / "robotwin_tasks" / "granular_pour.py"

EXPECTED_RULE_IDS = [
    "particles_inside|beads|bowl_src_interior+target_interior",
    "particles_outside|beads|zone_red",
    "particles_outside|beads|zone_blue",
]


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mock_pour_scene():
    """beads particle_set candidate + the four T2 regions (two containers, two mats)."""
    items = {
        "beads": ItemSpec(
            item_id="beads", entity=object(), category="beads", color="",
            kind="particle_set", particles=[object() for _ in range(4)], role="candidate",
        ),
        "bowl_src": ItemSpec(
            item_id="bowl_src", entity=object(), category="bowl", color="", role="task_object",
        ),
    }
    disk = np.array([[0.05 * np.cos(t), 0.05 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    rect = np.array([[-0.07, -0.05], [0.07, -0.05], [0.07, 0.05], [-0.07, 0.05]])
    regions = {
        "bowl_src_interior": Region(
            region_id="bowl_src_interior", kind="container", frame_fn=lambda: np.eye(4),
            polygon=disk, z=0.0, z_band=0.01, nameable=True,
        ),
        "target_interior": Region(
            region_id="target_interior", kind="container", frame_fn=lambda: np.eye(4),
            polygon=disk, z=0.0, z_band=0.01, nameable=True,
        ),
        "zone_red": Region(
            region_id="zone_red", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=rect, z=0.0, z_band=0.01, nameable=True, color="red",
        ),
        "zone_blue": Region(
            region_id="zone_blue", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=rect, z=0.0, z_band=0.01, nameable=True, color="blue",
        ),
    }
    return items, regions


def _enumerate():
    items, regions = mock_pour_scene()
    task_cfg = _load("tasks/T2_granular_pour.yaml")
    return enumerate_rules(
        items,
        regions,
        task_cfg,
        objects_cfg=_load("objects.yaml"),
        splits_cfg=resolve_splits(_load("splits.yaml"), "granular_pour"),
    )


def test_t2_cfg_shape():
    task_cfg = _load("tasks/T2_granular_pour.yaml")
    for key in (
        "task", "robotwin_task", "embodiments", "bimanual", "bowl_model",
        "bowl_ids", "beads", "target", "zones", "risk", "rules", "risk_collect",
    ):
        assert key in task_cfg, f"T2 yaml missing {key}"
    assert task_cfg["task"] == "granular_pour"
    assert task_cfg["robotwin_task"] == "granular_pour"
    assert task_cfg["embodiments"][0] == "ur5-wsg"
    assert "franka-panda" in task_cfg["embodiments"]
    assert task_cfg["bimanual"] is False
    assert task_cfg["risk"]["safe_time_scale"] == 0.6
    assert task_cfg["risk"]["time_scale"]["dist"] == "choice"
    assert task_cfg["risk"]["task"]["tilt_rate"]["range"] == [1.0, 4.0]
    assert task_cfg["risk"]["task"]["pour_height"]["range"] == [0.05, 0.20]
    assert task_cfg["risk"]["task"]["pour_offset"]["range"] == [0.0, 0.05]


def test_t2_rule_table_order_and_ids():
    rules = _enumerate()
    assert [r.rule_id for r in rules] == EXPECTED_RULE_IDS
    # deterministic: a second enumeration is identical
    assert [r.rule_id for r in _enumerate()] == EXPECTED_RULE_IDS


def test_t2_rule_semantics():
    rules = _enumerate()
    by_id = {r.rule_id: r for r in rules}
    for rid, rule in by_id.items():
        assert rule.consequence.value == rid.split("|")[0]
        assert rule.item_ids == ("beads",), rid
        assert rule.phrases, f"{rid} rendered no phrases"
        for phrase in rule.phrases:
            assert not any(c.isdigit() for c in phrase), phrase
        assert rule.split in ("train", "heldout_item", "heldout_region", "heldout_phrase")
    # the joint particles_inside rule spans BOTH container interiors
    assert by_id["particles_inside|beads|bowl_src_interior+target_interior"].region_ids == (
        "bowl_src_interior", "target_interior",
    )
    # one particles_outside per zone (a joint rule over both zones would make
    # the referent meaningless)
    assert by_id["particles_outside|beads|zone_red"].region_ids == ("zone_red",)
    assert by_id["particles_outside|beads|zone_blue"].region_ids == ("zone_blue",)
    assert "T" in by_id["particles_inside|beads|bowl_src_interior+target_interior"].params


def test_t2_task_file_matches_yaml():
    """The task-file module constants and the yaml must tell the same story."""
    src = TASK_SRC.read_text(encoding="utf-8")
    cfg = _load("tasks/T2_granular_pour.yaml")

    beads = cfg["beads"]
    assert f"N_BEADS = {beads['n']}" in src
    assert f"BEAD_R = {beads['radius']}" in src
    lo, hi = beads["mass"]
    assert f"BEAD_MASS_RANGE = ({lo}, {hi})" in src

    ids = ", ".join(str(i) for i in cfg["bowl_ids"])
    assert f"BOWL_IDS = ({ids})" in src
    assert cfg["bowl_model"] == "002_bowl"

    assert f"BIMANUAL = {cfg['bimanual']}" in src  # False vs False

    m = re.search(r"^ZONE_DEFS = \((.*)\)\s*$", src, flags=re.MULTILINE)
    assert m, "ZONE_DEFS not found in task file"
    for c in cfg["zones"]["colors"]:
        assert f'"{c}"' in m.group(1), f"zone color {c} missing from ZONE_DEFS"

    # per-zone full-extent sampling ranges (compare numerically: yaml 0.10
    # stringifies as 0.1)
    ms = re.search(r"^ZONE_SIZE_XY = (\(.*\))\s*(?:#.*)?$", src, flags=re.MULTILINE)
    assert ms, "ZONE_SIZE_XY not found in task file"
    src_sizes = [tuple(float(v) for v in pair.split(",")) for pair in re.findall(r"\(([^()]+)\)", ms.group(1))]
    assert src_sizes == [tuple(pair) for pair in cfg["zones"]["size_xy"]]

    # pour knob defaults + tilt range
    assert "POUR_HEIGHT_DEFAULT = 0.10" in src
    assert "POUR_OFFSET_DEFAULT = 0.0" in src
    assert "TILT_RATE_DEFAULT = 2.0" in src
    assert "TILT_ANGLE_DEG = (100.0, 120.0)" in src

    # every zone the yaml names a rule for must be one the task registers
    for entry in cfg["rules"]:
        for rid in entry["regions"]:
            if rid.startswith("zone_"):
                assert rid == f"zone_{rid.split('_', 1)[1]}"
                assert f'"{rid.split("_", 1)[1]}"' in m.group(1)
