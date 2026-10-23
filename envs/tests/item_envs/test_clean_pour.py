"""Non-sim tests: T6 clean_pour (redesigned 2026-10-23: three bowls, robotwin_tasks/bowl_pour.py) rule table +
task-file/yaml contract.

Mock scene mirrors bowl_pour.py (beads candidate, the source mug / target regions, two keep_clean bowl regions named
bowl_<colour>); rules are enumerated from the REAL T6 yaml + objects.yaml + splits.yaml. The task file's module
constants are pinned to the yaml.
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
TASK_SRC = ROOT / "robotwin_tasks" / "bowl_pour.py"
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


def _scene(bowls, target_color="green"):
    items = {
        "bowl_src": ItemSpec(item_id="bowl_src", entity=object(), category="pour_mug", color="", role="task_object"),
        "beads": ItemSpec(item_id="beads", entity=object(), category="beads", color="", role="candidate",
                          kind="particle_set"),
    }
    regions = {
        "bowl_src_interior": Region(region_id="bowl_src_interior", kind="container", frame_fn=lambda: np.eye(4),
                                    polygon=SQ, z=0.0, z_band=0.01, nameable=True),
        "target_interior": Region(region_id="target_interior", kind="container", frame_fn=lambda: np.eye(4),
                                  polygon=SQ, z=0.0, z_band=0.01, nameable=True, color=target_color, role="target"),
    }
    for rid in bowls:
        regions[rid] = Region(region_id=rid, kind="container", frame_fn=lambda: np.eye(4), polygon=SQ, z=0.0,
                              z_band=0.01, nameable=True, color=rid.split("_")[1], z_cap=0.012, role="keep_clean")
    return items, regions


def _rules(bowls, target_color="green"):
    items, regions = _scene(bowls, target_color)
    return enumerate_rules(items, regions, _load("tasks/T6_clean_pour.yaml"), objects_cfg=_load("objects.yaml"),
                           splits_cfg=resolve_splits(_load("splits.yaml"), "clean_pour"))


def test_registry_entry():
    entry = get_task_entry("clean_pour")
    assert entry.robotwin_task == "bowl_pour"   # 2026-10-23; the old design stays in clean_pour.py for old runs
    assert entry.config.name == "T6_clean_pour.yaml"
    assert (ROOT / "robotwin_tasks" / "clean_pour.py").exists()


def test_rule_table_one_keep_clean_rule_per_other_bowl_plus_spill():
    bowls = ["bowl_blue", "bowl_white"]
    ids = [r.rule_id for r in _rules(bowls)]
    assert ids == [f"particles_outside|beads|{rid}" for rid in sorted(bowls)] + [
        "particles_inside|beads|bowl_src_interior+target_interior"]
    assert all("T" not in r.params for r in _rules(bowls))   # hard min over beads (2026-10-07)


def test_bowl_phrases_name_the_colour_and_the_spill_rule_is_never_named():
    """Two bowls: every phrase is colour-qualified ("the white bowl"), white included (objects.yaml colors.fixed);
    the spill rule has no sentence (Yuxuan 2026-10-23: only the bowl rules are named)."""
    rules = {r.rule_id: r for r in _rules(["bowl_white", "bowl_blue"], target_color="green")}
    for rid, color in (("particles_outside|beads|bowl_white", "white"), ("particles_outside|beads|bowl_blue", "blue")):
        assert rules[rid].phrases, rid
        for p in rules[rid].phrases:
            assert f"{color} bowl" in p, p
            assert not any(ch.isdigit() for ch in p)
    assert rules["particles_inside|beads|bowl_src_interior+target_interior"].phrases == []


def test_primary_rule_id_format_matches_rule_table():
    src = TASK_SRC.read_text(encoding="utf-8")
    assert 'self.primary_rule_id = f"particles_outside|beads|{ref[\'region_id\']}"' in src
    assert 'rid = f"bowl_{color}"' in src
    assert "particles_outside|beads|bowl_green" in [r.rule_id for r in _rules(["bowl_green", "bowl_blue"], "white")]


def test_constants_mirror_yaml():
    cfg = _load("tasks/T6_clean_pour.yaml")
    b = cfg["bowls"]
    assert cfg["robotwin_task"] == "bowl_pour"
    assert _const("BOWL_MODEL") == b["model"]
    assert _const("BOWL_COLORS") == {int(k): v for k, v in b["colors"].items()}
    assert _const("NEIGHBOR_SLOTS") == tuple(b["slots"])
    assert _const("SLOT_JITTER_DEG") == b["slot_jitter_deg"]
    assert _f(_const("NEIGHBOR_GAP")) == _f(b["gap"])
    assert _const("SRC_CLEAR") == b["src_clear"]
    assert _const("NEIGHBOR_CLEAR") == b["neighbor_clear"]
    assert _const("HAND_CLEAR") == b["hand_clear"]
    assert _const("Z_CAP") == b["z_cap"]
    assert _const("FLOOR_BEAD_FRAC") == b["floor_bead_frac"]
    assert _const("AIM_SLOTS") == {k: tuple(v) for k, v in b["aim_slots"].items()}
    assert all(set(v) <= set(b["slots"]) for v in b["aim_slots"].values())
    assert _const("REFERENT_P") == cfg["referent"] == {"hit": 0.5, "other": 0.5}
    assert _const("SUCCESS_MIN_FRAC") == cfg["success_min_frac"]


def test_bowl_colours_are_distinct_named_words_and_not_bead_red():
    cfg = _load("tasks/T6_clean_pour.yaml")
    colors = list(cfg["bowls"]["colors"].values())
    assert len(set(colors)) == 3 and "red" not in colors   # the beads are red
    cl = _load("objects.yaml")["colors"]
    assert set(colors) <= set(cl["train"]) | set(cl["fixed"]) and not set(colors) & set(cl["heldout"])
    assert sorted(cfg["bowls"]["colors"]) == [3, 5, 7]   # 002_bowl base 5 / 3 / 7 (Yuxuan 2026-10-23: base 1 is black inside)


def test_aim_error_knob_pour_offset_pinned_and_safe_mode():
    cfg = _load("tasks/T6_clean_pour.yaml")
    knobs = cfg["risk"]["task"]
    assert "aim_error" in knobs and knobs["aim_error"]["range"][0] >= 0.0
    assert "pour_offset" not in knobs  # the aim error replaces T2's lateral offset (default 0)
    assert cfg["safe_mode"]["risk"]["aim_error"] == 0.0
