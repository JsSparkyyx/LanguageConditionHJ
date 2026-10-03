"""Non-sim tests: T4 granular_carry rule table + task-file/yaml contract.

Mock scene mirrors robotwin_tasks/granular_carry.py (beads particle_set,
bowl_src task_object, bowl_src_interior container, target_pad, zone_red /
zone_blue mats); rules are enumerated from the REAL T4 yaml + objects.yaml +
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
TASK_SRC = ROOT / "robotwin_tasks" / "granular_carry.py"

EXPECTED_RULE_IDS = [
    "particles_inside|beads|bowl_src_interior",
    "particles_outside|beads|zone_red",
    "particles_outside|beads|zone_blue",
]


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _const(name):
    src = TASK_SRC.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", src, re.M)
    assert m, name
    return ast.literal_eval(re.sub(r"\s+#.*$", "", m.group(1)))


def _f(v):
    return tuple(_f(x) for x in v) if isinstance(v, (list, tuple)) else float(v)


def _scene():
    items = {
        "beads": ItemSpec(item_id="beads", entity=object(), category="beads", color="",
                          kind="particle_set", particles=[object()] * 4, role="candidate"),
        "bowl_src": ItemSpec(item_id="bowl_src", entity=object(), category="bowl", color="", role="task_object"),
    }
    disk = np.array([[0.05 * np.cos(t), 0.05 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    rect = np.array([[-0.06, -0.06], [0.06, -0.06], [0.06, 0.06], [-0.06, 0.06]])
    mk = lambda rid, kind, poly, color=None: Region(region_id=rid, kind=kind, frame_fn=lambda: np.eye(4),
                                                    polygon=poly, z=0.0, z_band=0.01, nameable=True, color=color)
    regions = {
        "bowl_src_interior": mk("bowl_src_interior", "container", disk),
        "target_pad": mk("target_pad", "zone", rect),
        "zone_red": mk("zone_red", "zone", rect, "red"),
        "zone_blue": mk("zone_blue", "zone", rect, "blue"),
    }
    return items, regions


def _rules():
    items, regions = _scene()
    return enumerate_rules(items, regions, _load("tasks/T4_granular_carry.yaml"),
                           objects_cfg=_load("objects.yaml"),
                           splits_cfg=resolve_splits(_load("splits.yaml"), "granular_carry"))


def test_registry_entry():
    entry = get_task_entry("granular_carry")
    assert entry.robotwin_task == "granular_carry"
    assert entry.config.name == "T4_granular_carry.yaml"


def test_rule_table_and_phrases():
    rules = _rules()
    assert [r.rule_id for r in rules] == EXPECTED_RULE_IDS
    for r in rules:
        assert r.phrases and not any(c.isdigit() for p in r.phrases for c in p)
        assert r.params.get("T") == 0.001
    # the target pad is never a referent
    assert not any("target_pad" in r.rule_id for r in rules)


def test_constants_mirror_yaml():
    cfg = _load("tasks/T4_granular_carry.yaml")
    assert _f(_const("START_X")) == _f(cfg["start"]["x"])
    assert _f(_const("START_Y")) == _f(cfg["start"]["y"])
    assert _f(_const("TARGET_DX")) == _f(cfg["target"]["dx"])
    assert _f(_const("TARGET_DY")) == _f(cfg["target"]["dy"])
    assert _f(_const("PAD_SIZE")) == _f(cfg["target"]["pad_size"])
    assert _const("N_BEADS_MAX") == cfg["beads"]["n_max"]
    assert _f(_const("BRIM_CLEAR")) == _f(cfg["beads"]["brim_clear"])
    assert _f(_const("ZONE_SIZE_XY")) == _f(cfg["zones"]["size_xy"])
    assert _f(_const("ZONE_T")) == _f(cfg["zones"]["t"])
    assert _f(_const("ZONE_EDGE_GAP")) == _f(cfg["zones"]["edge_gap"])
    assert _f(_const("CARRY_STEP")) == _f(cfg["carry"]["step"])
    assert _f(_const("CARRY_SPEED")) == _f(cfg["carry"]["speed"])
    lo, hi = cfg["risk"]["task"]["carry_tilt_deg"]["range"]
    assert lo <= _const("CARRY_TILT_DEFAULT") <= hi  # vanilla default inside the knob range
    lo, hi = cfg["risk"]["task"]["carry_height"]["range"]
    assert lo <= _const("CARRY_HEIGHT_DEFAULT") <= hi


def test_task_file_never_imports_main():
    tree = ast.parse(TASK_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.startswith("main") for n in names), names
