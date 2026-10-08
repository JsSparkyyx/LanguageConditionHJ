"""Non-sim tests: T6 clean_pour rule table + task-file/yaml contract.

Mock scene mirrors robotwin_tasks/clean_pour.py (beads candidate, the source
mug / target regions, 2-3 keep_clean neighbour regions named
<kind>_<color>); rules are enumerated from the REAL T6 yaml + objects.yaml +
splits.yaml. The task file's module constants are pinned to the yaml.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import yaml

from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.registry import get_task_entry
from main.item_envs.rules import enumerate_rules, resolve_splits

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "main" / "item_envs" / "configs"
TASK_SRC = ROOT / "robotwin_tasks" / "clean_pour.py"
SQ = [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]


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
    if isinstance(v, str):
        return v
    return tuple(_f(x) for x in v) if isinstance(v, (list, tuple)) else float(v)


def _scene(neighbors):
    items = {
        "bowl_src": ItemSpec(item_id="bowl_src", entity=object(), category="pour_mug", color="", role="task_object"),
        "beads": ItemSpec(item_id="beads", entity=object(), category="beads", color="", role="candidate",
                          kind="particle_set"),
    }
    regions = {rid: Region(region_id=rid, kind="container", frame_fn=lambda: np.eye(4), polygon=SQ, z=0.0,
                           z_band=0.01, nameable=True) for rid in ("bowl_src_interior", "target_interior")}
    for rid in neighbors:
        regions[rid] = Region(region_id=rid, kind="container", frame_fn=lambda: np.eye(4), polygon=SQ, z=0.0,
                              z_band=0.01, nameable=True, color=rid.split("_")[1], z_cap=0.012, role="keep_clean")
    return items, regions


def _rules(neighbors):
    items, regions = _scene(neighbors)
    return enumerate_rules(items, regions, _load("tasks/T6_clean_pour.yaml"), objects_cfg=_load("objects.yaml"),
                           splits_cfg=resolve_splits(_load("splits.yaml"), "clean_pour"))


def test_registry_entry():
    entry = get_task_entry("clean_pour")
    assert entry.robotwin_task == "clean_pour"
    assert entry.config.name == "T6_clean_pour.yaml"


def test_rule_table_one_keep_clean_rule_per_neighbour_plus_spill():
    nbs = ["bin_blue", "plate_green", "bin_yellow"]
    ids = [r.rule_id for r in _rules(nbs)]
    assert ids == [f"particles_outside|beads|{rid}" for rid in sorted(nbs)] + [
        "particles_inside|beads|bowl_src_interior+target_interior"]
    assert all("T" not in r.params for r in _rules(nbs))   # hard min over beads (2026-10-07)


def test_keep_clean_phrases_name_exactly_one_neighbour():
    """Two bins: every phrase is color-qualified ("the blue bin"); the only
    plate may be called "the plate"; never a digit."""
    rules = {r.rule_id: r for r in _rules(["bin_blue", "bin_yellow", "plate_green"])}
    for rid, color in (("particles_outside|beads|bin_blue", "blue"), ("particles_outside|beads|bin_yellow", "yellow")):
        assert rules[rid].phrases
        for p in rules[rid].phrases:
            assert f"{color} " in p, p
            assert not any(ch.isdigit() for ch in p)
    assert any("plate" in p or "dish" in p or "saucer" in p for p in rules["particles_outside|beads|plate_green"].phrases)


def test_primary_rule_id_format_matches_rule_table():
    src = TASK_SRC.read_text(encoding="utf-8")
    assert 'self.primary_rule_id = f"particles_outside|beads|{ref[\'region_id\']}"' in src
    assert 'rid = f"{nb[\'kind\']}_{color}"' in src
    assert "particles_outside|beads|bin_green" in [r.rule_id for r in _rules(["bin_green", "plate_blue"])]


def test_constants_mirror_yaml():
    cfg = _load("tasks/T6_clean_pour.yaml")
    nb = cfg["neighbors"]
    assert _const("NEIGHBOR_SLOTS") == tuple(nb["slots"])
    assert _const("OUTER_SLOTS") == tuple(nb["outer_slots"])
    assert _const("AIM_SLOTS") == {k: tuple(v) for k, v in nb["aim_slots"].items()}
    assert all(set(v) <= set(nb["outer_slots"]) for v in nb["aim_slots"].values())
    assert _f(_const("N_NEIGHBORS")) == _f(nb["n"]["values"])
    assert _f(_const("N_NEIGHBORS_P")) == _f(nb["n"]["p"])
    assert _const("SLOT_JITTER_DEG") == nb["slot_jitter_deg"]
    assert _f(_const("NEIGHBOR_GAP")) == _f(nb["gap"])
    assert _const("SRC_CLEAR") == nb["src_clear"]
    assert _const("KINDS") == tuple(nb["kinds"]["values"])
    assert _f(_const("KIND_P")) == _f(nb["kinds"]["p"])
    assert _f(_const("BIN_SIDE")) == _f(nb["bin"]["side"])
    assert _f(_const("BIN_WALL_H")) == _f(nb["bin"]["wall_h"])
    assert _const("BIN_WALL_T") == nb["bin"]["wall_t"]
    assert _const("BIN_FLOOR_T") == nb["bin"]["floor_t"]
    assert _f(_const("PLATE_SIDE")) == _f(nb["plate"]["side"])
    assert _const("PLATE_T") == nb["plate"]["t"]
    assert _const("NEIGHBOR_COLORS") == tuple(nb["colors"])
    assert _const("Z_CAP") == nb["z_cap"]
    assert _const("REFERENT_P") == cfg["referent"]
    assert _const("SUCCESS_MIN_FRAC") == cfg["success_min_frac"]
    assert sum(cfg["referent"].values()) == 1.0


def test_neighbour_colors_avoid_bead_red_and_white():
    cfg = _load("tasks/T6_clean_pour.yaml")
    assert not {"red", "white"} & set(cfg["neighbors"]["colors"])
    assert len(cfg["neighbors"]["colors"]) >= max(cfg["neighbors"]["n"]["values"])


def test_aim_error_knob_and_pour_offset_pinned():
    cfg = _load("tasks/T6_clean_pour.yaml")
    knobs = cfg["risk"]["task"]
    assert "aim_error" in knobs and knobs["aim_error"]["range"][0] >= 0.0
    assert "pour_offset" not in knobs  # the aim error replaces T2's lateral offset (default 0)
