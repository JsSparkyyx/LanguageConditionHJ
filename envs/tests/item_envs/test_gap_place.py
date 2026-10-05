"""Non-sim tests: T7 gap_place rule table + task-file/yaml contract.

Mock scene mirrors robotwin_tasks/gap_place.py (wooden payload task_object,
2-3 standing colored blocks as candidates); rules are enumerated from the
REAL T7 yaml + objects.yaml + splits.yaml. The task file's module constants
are pinned to the yaml; the slot geometry keeps the payload's offset edge
on the flanker's top (partial overlap, never a full landing).
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
TASK_SRC = ROOT / "robotwin_tasks" / "gap_place.py"


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
    items = {"payload": ItemSpec(item_id="payload", entity=object(), category="wooden_block", color="",
                                 role="task_object")}
    for k, c in enumerate(["red", "green", "blue"][:n_items]):
        items[f"block_{k}"] = ItemSpec(item_id=f"block_{k}", entity=object(), category="block", color=c,
                                     role="candidate")
    return enumerate_rules(items, {}, _load("tasks/T7_gap_place.yaml"), objects_cfg=_load("objects.yaml"),
                           splits_cfg=resolve_splits(_load("splits.yaml"), "gap_place"))


def test_registry_entry():
    entry = get_task_entry("gap_place")
    assert entry.robotwin_task == "gap_place"
    assert entry.config.name == "T7_gap_place.yaml"


def test_rule_table_upright_and_stay_per_block():
    for n in (2, 3):
        ids = [r.rule_id for r in _rules(n)]
        assert ids == [f"upright|block_{k}|-" for k in range(n)] + [f"stay|block_{k}|-" for k in range(n)]


def test_phrases_color_qualified_never_the_payload():
    """Every phrase names its block by color (two+ blocks share the bare
    names, and the wooden payload answers to "block" too); no digits."""
    rules = _rules(3)
    colors = {"block_0": "red", "block_1": "green", "block_2": "blue"}
    for r in rules:
        assert r.phrases
        for p in r.phrases:
            assert f"{colors[r.item_ids[0]]} " in p, p
            assert not any(ch.isdigit() for ch in p)


def test_primary_rule_id_format_matches_rule_table():
    src = TASK_SRC.read_text(encoding="utf-8")
    assert 'self.primary_rule_id = f"upright|{ref[\'item_id\']}|-"' in src
    assert "upright|block_0|-" in [r.rule_id for r in _rules(2)]


def test_constants_mirror_yaml():
    cfg = _load("tasks/T7_gap_place.yaml")
    assert _f(_const("START_ABS_X")) == _f(cfg["start"]["abs_x"])
    assert _f(_const("START_Y")) == _f(cfg["start"]["y"])
    assert _f(_const("SLOT_ABS_X")) == _f(cfg["slot"]["abs_x"])
    assert _f(_const("SLOT_Y")) == _f(cfg["slot"]["y"])
    assert _const("MIN_TRAVEL") == cfg["slot"]["min_travel"]
    assert _f(_const("SLOT_CLEARANCE")) == _f(cfg["slot"]["clearance"])
    assert _f(_const("PAYLOAD_W")) == _f(cfg["payload"]["w"])
    assert _f(_const("PAYLOAD_H")) == _f(cfg["payload"]["h"])
    assert _f(_const("PAYLOAD_MASS_RANGE")) == _f(cfg["payload"]["mass"])
    assert _f(_const("TALL_W")) == _f(cfg["items"]["tall"]["w"])
    assert _f(_const("TALL_H")) == _f(cfg["items"]["tall"]["h"])
    assert _f(_const("SHORT_W")) == _f(cfg["items"]["short"]["w"])
    assert _f(_const("SHORT_H")) == _f(cfg["items"]["short"]["h"])
    assert _const("FLANKER_KIND_P") == cfg["items"]["flanker_kind_p"]
    assert _const("BYSTANDER_P") == cfg["items"]["bystander_p"]
    assert _f(_const("BYSTANDER_DY")) == _f(cfg["items"]["bystander_dy"])
    assert _f(_const("BYSTANDER_DX")) == _f(cfg["items"]["bystander_dx"])
    assert _f(_const("ITEM_MASS_RANGE")) == _f(cfg["items"]["mass"])
    assert _const("ITEM_COLORS") == tuple(cfg["items"]["colors"])
    assert _const("REFERENT_P") == cfg["referent"]
    m = cfg["motion"]
    assert _const("SAFE_SPEED") == m["safe_speed"] and _const("DESCENT_SPEED") == m["descent_speed"]
    assert _const("TRANSIT_CLEAR") == m["transit_clear"] and _const("GRASP_TIP_CLEAR") == m["tip_clear"]
    assert _const("PLACE_CLEAR") == m["place_clear"] and _const("JAW_MARGIN") == m["jaw_margin"]
    assert _const("GRIPPER_STIFFNESS") == m["gripper_stiffness"]
    assert _const("SUCCESS_TOL") == cfg["success_tol"]
    up = next(r for r in cfg["rules"] if r["consequence"] == "upright")
    st = next(r for r in cfg["rules"] if r["consequence"] == "stay")
    assert _const("UPRIGHT_MAX_DEG") == up["theta_max_deg"] and _const("STAY_D_TOL") == st["d_tol"]
    assert abs(sum(cfg["referent"].values()) - 1.0) < 1e-9


def test_offset_overlap_is_partial():
    """The largest offset beyond the smallest clearance stays below the
    slimmest flanker's width: the payload edge always lands ON the
    flanker's inner top edge (a shove), never fully on its top."""
    cfg = _load("tasks/T7_gap_place.yaml")
    off_max = cfg["risk"]["task"]["place_offset"]["range"][1]
    overlap_max = off_max - cfg["slot"]["clearance"][0]
    assert 0.0 < overlap_max < cfg["items"]["tall"]["w"][0]
    # and the smallest clearance > 0: a centred payload never touches a flanker
    assert cfg["slot"]["clearance"][0] > 0.0


def test_flanker_tipping_angles():
    cfg = _load("tasks/T7_gap_place.yaml")
    (w0, w1), (h0, h1) = cfg["items"]["tall"]["w"], cfg["items"]["tall"]["h"]
    for w, h in ((w0, h1), (w1, h0)):
        assert 15.0 < np.degrees(np.arctan(w / h)) < 24.0


def test_every_block_shorter_than_the_payload():
    """The palm sits at ~the payload's top (a low grip): every standing block
    is >= 7 mm shorter than the shortest payload, so only the payload can
    touch a flanker."""
    cfg = _load("tasks/T7_gap_place.yaml")
    tallest = max(cfg["items"]["tall"]["h"][1], cfg["items"]["short"]["h"][1])
    assert cfg["payload"]["h"][0] - tallest >= 0.007 - 1e-9
