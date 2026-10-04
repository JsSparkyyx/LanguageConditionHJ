"""Non-sim tests: T4 granular_carry rule table + task-file/yaml contract.

Mock scene mirrors robotwin_tasks/granular_carry.py (beads particle_set,
bowl_src task_object, bowl_src_interior container — no mats since user
review 3); rules are enumerated from the REAL T4 yaml + objects.yaml +
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
    regions = {"bowl_src_interior": mk("bowl_src_interior", "container", disk)}
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
    # the only constraint is "spill no bead" (user decision 2026-10-03)
    assert len(rules) == 1


def test_constants_mirror_yaml():
    cfg = _load("tasks/T4_granular_carry.yaml")
    assert _f(_const("START_ABS_X")) == _f(cfg["start"]["abs_x"])
    assert _f(_const("START_Y")) == _f(cfg["start"]["y"])
    assert _f(_const("TARGET_ABS_X")) == _f(cfg["target"]["abs_x"])
    assert _f(_const("TARGET_Y")) == _f(cfg["target"]["y"])
    assert _f(_const("MIN_CARRY_DIST")) == _f(cfg["target"]["min_dist"])
    assert tuple(_const("CARRY_MODES")) == tuple(cfg["modes"])
    assert _f(_const("CARRY_MODE_P")) == _f(tuple(cfg["modes"].values()))
    assert _f(_const("HELD_PACK_FILL")) == _f(cfg["held"]["pack_fill"])
    assert _f(_const("CARRY_INTERIOR_DEPTH")) == _f(cfg["bowl"]["interior_depth"])
    assert _f(_const("TABLE_FILL_FRAC")) == _f(cfg["table"]["fill_frac"])
    assert _const("HELD_SETTLE_STEPS") == cfg["held"]["settle_steps"]
    assert _f(_const("PAD_SIZE")) == _f(cfg["target"]["pad_size"])
    assert _const("N_BEADS_MAX") == cfg["beads"]["n_max"]
    assert _f(_const("BRIM_CLEAR")) == _f(cfg["beads"]["brim_clear"])
    assert _const("BEAD_PACK") == cfg["beads"]["pack"]
    assert _f(_const("PACK_FILL")) == _f(cfg["beads"]["pack_fill"])
    assert _f(_const("CARRY_STEP")) == _f(cfg["carry"]["step"])
    assert _f(_const("CARRY_SPEED")) == _f(cfg["carry"]["speed"])
    assert _f(_const("SAFE_SPEED")) == _f(cfg["carry"]["safe_speed"])
    assert _f(_const("RISKY_STEP_MIN_T")) == _f(cfg["carry"]["risky_step_min_t"])
    assert _f(_const("LIFTOFF")) == _f(cfg["carry"]["liftoff"])
    assert _const("LEVEL_ITERS") == cfg["carry"]["level_iters"]
    assert _f(_const("GRIPPER_STIFFNESS")) == _f(cfg["carry"]["gripper_stiffness"])
    assert _f(_const("CARRY_BOWL_MASS_RANGE")) == _f(cfg["carry"]["bowl_mass"])
    assert _f(_const("GRASP_DIS")) == _f(cfg["carry"]["grasp_dis"])
    assert _f(_const("GRASP_RADIAL_OUT")) == _f(cfg["carry"]["grasp_radial_out"])
    lo, hi = cfg["risk"]["task"]["carry_tilt_deg"]["range"]
    assert lo <= _const("CARRY_TILT_DEFAULT") <= hi
    lo, hi = cfg["risk"]["task"]["carry_height"]["range"]
    assert lo <= _const("CARRY_HEIGHT_DEFAULT") <= hi


def test_task_file_never_imports_main():
    tree = ast.parse(TASK_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.startswith("main") for n in names), names
