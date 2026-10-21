"""Non-sim tests: the smoke_items.yaml rule table against a mock smoke scene.

Mirrors the robotwin_tasks/smoke_items.py layout (box_0 red, box_1 blue,
zone_green mat) with mock items/regions and enumerates rules from the REAL
configs/tasks/smoke_items.yaml + objects.yaml + splits.yaml — the test pins
the config-to-rule-table contract the Phase 2 acceptance sim tests rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import enumerate_rules, resolve_splits

CONFIGS = Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs"

EXPECTED_RULE_IDS = [
    "inside|box_0|table_top",
    "inside|box_1|table_top",
    "outside|box_0|zone_green",
    "outside|box_1|zone_green",
    "avoid|box_0|-",
    "avoid|box_1|-",
]


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mock_smoke_scene():
    """box_0 (red) + box_1 (blue) candidates; table_top + zone_green regions."""
    items = {
        "box_0": ItemSpec(item_id="box_0", entity=object(), category="box", color="red", role="candidate"),
        "box_1": ItemSpec(item_id="box_1", entity=object(), category="box", color="blue", role="candidate"),
    }
    table_poly = np.array([[-0.6, -0.35], [0.6, -0.35], [0.6, 0.35], [-0.6, 0.35]])
    zone_poly = np.array([[-0.07, -0.06], [0.07, -0.06], [0.07, 0.06], [-0.07, 0.06]])
    regions = {
        "table_top": Region(
            region_id="table_top", kind="surface", frame_fn=lambda: np.eye(4),
            polygon=table_poly, z=0.74, z_band=0.01, nameable=True,
        ),
        "zone_green": Region(
            region_id="zone_green", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=zone_poly, z=0.0, z_band=0.01, nameable=True, color="green",
        ),
    }
    return items, regions


def test_smoke_rule_table_order_and_ids():
    items, regions = mock_smoke_scene()
    task_cfg = _load("tasks/smoke_items.yaml")
    rules = enumerate_rules(
        items,
        regions,
        task_cfg,
        objects_cfg=_load("objects.yaml"),
        splits_cfg=resolve_splits(_load("splits.yaml"), "smoke_items"),
    )
    assert [r.rule_id for r in rules] == EXPECTED_RULE_IDS


def test_smoke_rule_semantics():
    items, regions = mock_smoke_scene()
    task_cfg = _load("tasks/smoke_items.yaml")
    rules = enumerate_rules(
        items,
        regions,
        task_cfg,
        objects_cfg=_load("objects.yaml"),
        splits_cfg=resolve_splits(_load("splits.yaml"), "smoke_items"),
    )
    by_id = {r.rule_id: r for r in rules}
    for rid, rule in by_id.items():
        consequence = rid.split("|")[0]
        assert rule.consequence.value == consequence
        assert len(rule.item_ids) == 1  # no set rules in the smoke config
        if consequence == "inside":   # keep-on-the-table is never named (Yuxuan, 2026-10-20): no wording
            assert rule.phrases == [], rule.phrases
            continue
        assert rule.phrases, f"{rid} rendered no phrases"
        for phrase in rule.phrases:
            assert not any(c.isdigit() for c in phrase), phrase
        assert rule.split in ("train", "heldout_item", "heldout_region", "heldout_phrase")
    assert by_id["inside|box_0|table_top"].region_ids == ("table_top",)
    assert by_id["outside|box_1|zone_green"].region_ids == ("zone_green",)
    assert by_id["avoid|box_0|-"].region_ids == ()
    assert by_id["avoid|box_0|-"].params["margin"] == 0.0
    # colored items name their color in at least the naming metadata
    assert by_id["avoid|box_0|-"].params["color"] == "red"
    assert by_id["avoid|box_1|-"].params["color"] == "blue"


def test_smoke_cfg_shape():
    task_cfg = _load("tasks/smoke_items.yaml")
    assert task_cfg["task"] == "smoke_items"
    assert task_cfg["robotwin_task"] == "smoke_items"
    assert task_cfg["embodiments"][0] == "ur5-wsg"
    assert task_cfg["risk"]["safe_time_scale"] == 0.6
    assert task_cfg["risk"]["time_scale"]["dist"] == "choice"
