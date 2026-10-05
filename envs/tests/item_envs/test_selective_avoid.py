"""Non-sim tests for T1 selective_avoid (PLAN v3 §7 T1, Phase 4).

Covers the T1 task YAML (rule enumeration contract, risky_skill_labels
honesty against the CEHJ skill tracker, host_objects table consistency with
RoboTwin host sources and CEHJ TASK_SPECS), the runner's host argument
handling, and the runner's spawn-report surfacing (mock level — spawn.py
Fix 1's t_fallbacks / degraded / k_dropped counters must reach
meta.diag["spawn"]). Everything here runs without a simulator; the sim
side lives in test_selective_avoid_sim.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.tasks (TASK_SPECS) lives in the sibling CEHJ copy

from main.envs.tasks import TASK_SPECS
from main.item_envs.items import ItemSpec
from main.item_envs.registry import get_task_entry
from main.item_envs.runner import ItemEpisodeRunner, _host_category
from main.item_envs.rules import Consequence, enumerate_rules, resolve_splits

ENVs_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ENVs_DIR.parent  # LanguageConditionHJ checkout
ROBOTWIN_ENVS = REPO_ROOT.parent / "RoboTwin" / "envs"
CONFIGS = ENVs_DIR / "main" / "item_envs" / "configs"
T1_YAML = CONFIGS / "tasks" / "T1_selective_avoid.yaml"
CEHJ_CONTROLLER = REPO_ROOT / "main" / "envs" / "controller.py"

HOSTS = (
    "place_container_plate",
    "place_burger_fries",
    "stack_blocks_two",
    "stack_bowls_two",
    "place_bread_basket",
)


def load_t1() -> dict:
    return yaml.safe_load(T1_YAML.read_text()) or {}


def make_item(item_id, category, role="candidate", color=None):
    return ItemSpec(item_id=item_id, entity=object(), category=category, color=color, role=role)


def t1_items():
    """Synthetic T1 scene: 4 candidates (two shared categories) + host items."""
    items = {
        "cup_0": make_item("cup_0", "cup"),
        "cup_1": make_item("cup_1", "cup"),
        "bottle_0": make_item("bottle_0", "bottle"),
        "box_0": make_item("box_0", "box", color="red"),
        # host items: registered but NEVER rule referents (PLAN v3 §7 T1)
        "container": make_item("container", "bowl", role="task_object"),
        "plate": make_item("plate", "plate", role="support"),
    }
    return items


# ---------------------------------------------------------------------------
# YAML structure
# ---------------------------------------------------------------------------


def test_yaml_required_keys():
    cfg = load_t1()
    assert cfg["task"] == "selective_avoid"
    assert tuple(cfg["host_tasks"]) == HOSTS
    assert cfg["embodiments"] == ["ur5-wsg", "franka-panda"]
    assert cfg["host_tasks"] == list(get_task_entry("selective_avoid").host_tasks)
    # spawn-required keys (spawn.spawn_candidates reads objects + corridor_t)
    assert "objects" in cfg and "corridor_t" in cfg
    objects = cfg["objects"]
    for key in ("k", "max_on_path", "min_on_or_near_path", "near_path_offset",
                "off_path_min_dist", "min_separation", "static_objects"):
        assert key in objects, key
    assert objects["k"] == [3, 5]
    assert objects["static_objects"] is False
    assert cfg["corridor_t"] == [0.22, 0.48]
    # risk knobs per PLAN v3 §7 T1
    assert cfg["risk"]["safe_time_scale"] == 0.6
    assert set(cfg["risk"]["time_scale"]["values"]) == {0.6, 0.8, 1.0, 1.15, 1.3}
    assert cfg["risk"]["task"]["on_path_count"]["values"] == [1, 2]
    assert "risk_collect" in cfg  # filled by the 2026-10-05 calibration
    assert cfg["risk_collect"] is None or cfg["risk_collect"]["cells"]


def test_yaml_rule_entries_match_plan():
    """PLAN v3 §9 T1: avoid singles, same-category sets, all-set baseline, stay."""
    rules = load_t1()["rules"]
    assert [r["consequence"] for r in rules] == ["avoid", "avoid", "avoid", "stay"]
    assert rules[0]["items"] == {"role": "candidate"}
    assert "set_rule" not in rules[0]
    assert rules[1]["items"] == {"role": "candidate", "same_category_min": 2}
    assert rules[1]["set_rule"] is True
    assert rules[2]["set_rule"] == "all"  # CrossSafe baseline
    assert rules[3]["d_tol"] == 0.03
    for r in rules[:3]:
        assert r["margin"] == 0.01  # keep 1 cm away (user 2026-10-05)


# ---------------------------------------------------------------------------
# risky_skill_labels vs the CEHJ skill tracker source
# ---------------------------------------------------------------------------


def test_risky_skill_labels_nonempty_and_in_controller_source():
    cfg = load_t1()
    labels = cfg.get("risky_skill_labels")
    assert labels, "T1 needs risky_skill_labels (the skill-type risk path)"
    source = CEHJ_CONTROLLER.read_text(encoding="utf-8")
    # the labels must be strings the CEHJ skill tracker actually stamps
    # (_install_skill_tracker); a label missing from the source would never
    # match and would silently make every segment safe
    for label in labels:
        assert isinstance(label, str) and label
        assert label in source, f"{label!r} not found in {CEHJ_CONTROLLER}"


def test_risky_skill_labels_are_transport_and_place():
    """Transport (pre_place, lift) + place are risky; grasp/approach/retreat safe."""
    labels = set(load_t1()["risky_skill_labels"])
    assert {"pre_place", "place", "lift"} <= labels
    for safe in ("pre_grasp", "grasp", "close", "open", "home", "move_to_pose"):
        assert safe not in labels, f"{safe!r} must stay at safe_time_scale"


def test_risky_skill_labels_match_stamped_labels():
    """The labels substring-match the tracker's stamped transport/place labels
    and NOT the grasp/approach/retreat labels (RiskTickController semantics)."""
    labels = tuple(load_t1()["risky_skill_labels"])
    stamped_risky = [
        "left pre_place container", "right place hamburg", "left lift dz=0.1",
        "right pre_place block | left home",  # paired label: one risky arm wins
    ]
    stamped_safe = [
        "left pre_grasp container", "right grasp bowl1", "left close container",
        "right open container", "left home", "right move_to_pose",
    ]
    for label in stamped_risky:
        assert any(tag in label for tag in labels), label
    for label in stamped_safe:
        assert not any(tag in label for tag in labels), label


# ---------------------------------------------------------------------------
# host_objects table
# ---------------------------------------------------------------------------


def test_host_objects_covers_all_hosts():
    host_objects = load_t1().get("host_objects")
    assert host_objects is not None, "host_objects table missing"
    assert tuple(host_objects) == HOSTS


def _spec_actor_attrs(host: str) -> set[str]:
    """Actor attributes CEHJ's TASK_SPECS names for the host (poses excluded)."""
    spec = TASK_SPECS[host]
    names = set(spec.get("grasp_objects") or ())
    if spec.get("object"):
        names.add(spec["object"])
    target = spec.get("target")
    if target and not str(target).endswith("_pose"):  # *_target_pose are arrays
        names.add(str(target))
    return names


@pytest.mark.parametrize("host", HOSTS)
def test_host_objects_attrs_match_task_specs(host):
    entries = load_t1()["host_objects"][host]
    attrs = [e["attr"] for e in entries]
    assert set(attrs) == _spec_actor_attrs(host), (
        f"{host}: host_objects attrs {sorted(attrs)} != TASK_SPECS actor attrs "
        f"{sorted(_spec_actor_attrs(host))}"
    )
    for e in entries:
        assert e["role"] in ("task_object", "support"), e
        assert e.get("category") or e.get("category_from"), e


@pytest.mark.parametrize("host", HOSTS)
def test_host_objects_attrs_exist_in_robotwin_source(host):
    """Every registered attr is really assigned on the host task (self.<attr>)."""
    source = (ROBOTWIN_ENVS / f"{host}.py").read_text(encoding="utf-8")
    for e in load_t1()["host_objects"][host]:
        attr = e["attr"]
        assert re.search(rf"self\.{re.escape(attr)}\b", source), (
            f"{host}: attribute self.{attr} not found in {ROBOTWIN_ENVS / (host + '.py')}"
        )
        src = e.get("category_from")
        if src:
            assert re.search(rf"self\.{re.escape(src['attr'])}\b", source), (
                f"{host}: category_from attr self.{src['attr']} not in the host source"
            )
            for model in (src.get("map") or {}):
                assert model in source, f"{host}: model {model!r} not in the host source"


def test_host_objects_never_register_pose_targets():
    """stack_* targets are pose arrays (block1_target_pose/bowl1_target_pose),
    not actors — they must NOT appear as items."""
    for host in ("stack_blocks_two", "stack_bowls_two"):
        attrs = {e["attr"] for e in load_t1()["host_objects"][host]}
        assert all(not a.endswith("_pose") for a in attrs), attrs


def test_host_objects_roles_cover_pick_and_support():
    """Plan §7 T1: pick objects are task_object, targets support."""
    host_objects = load_t1()["host_objects"]
    for host in HOSTS:
        roles = [e["role"] for e in host_objects[host]]
        assert "task_object" in roles, host  # every host picks something
    # hosts with a dedicated support actor
    for host, attr in (("place_container_plate", "plate"),
                       ("place_burger_fries", "tray"),
                       ("place_bread_basket", "breadbasket")):
        entry = next(e for e in host_objects[host] if e["attr"] == attr)
        assert entry["role"] == "support", (host, entry)


def test_host_category_resolution():
    import types

    entry = {"attr": "container", "role": "task_object",
             "category_from": {"attr": "actor_name",
                               "map": {"002_bowl": "bowl", "021_cup": "cup"},
                               "default": "bowl"}}
    assert _host_category(types.SimpleNamespace(actor_name="002_bowl"), entry) == "bowl"
    assert _host_category(types.SimpleNamespace(actor_name="021_cup"), entry) == "cup"
    assert _host_category(types.SimpleNamespace(actor_name="???"), entry) == "bowl"  # default
    assert _host_category(types.SimpleNamespace(), {"category": "plate"}) == "plate"
    assert _host_category(types.SimpleNamespace(), {}) == "object"


# ---------------------------------------------------------------------------
# enumerate_rules over the T1 yaml (synthetic scene)
# ---------------------------------------------------------------------------


def t1_rules(items=None):
    cfg = load_t1()
    objects_cfg = yaml.safe_load((CONFIGS / "objects.yaml").read_text())
    splits_yaml = yaml.safe_load((CONFIGS / "splits.yaml").read_text())
    return enumerate_rules(
        items or t1_items(),
        {},  # T1 enumerates no regional rules; the runner still passes table_top
        cfg,
        objects_cfg=objects_cfg,
        splits_cfg=resolve_splits(splits_yaml, "selective_avoid"),
        rng=np.random.RandomState(0),
    )


def test_enumerate_rules_t1_contract():
    rules = t1_rules()
    ids = [r.rule_id for r in rules]
    # one avoid per candidate (4), never on host task_object/support items
    singles = [i for i in ids if i.startswith("avoid|") and i.endswith("|-") and "+" not in i]
    assert singles == ["avoid|bottle_0|-", "avoid|box_0|-", "avoid|cup_0|-", "avoid|cup_1|-"]
    # same-category set rules (same_category_min: 2): cup set and... box has one
    # member only here, so just the cup set
    assert "avoid|cup_0+cup_1|-" in ids
    assert "avoid|box_0|-" in ids and "avoid|box_0+cup_0|-" not in ids
    # the all-candidates baseline (CrossSafe setting)
    assert "avoid|bottle_0+box_0+cup_0+cup_1|-" in ids
    # stay rules per candidate
    stays = [i for i in ids if i.startswith("stay|")]
    assert stays == ["stay|bottle_0|-", "stay|box_0|-", "stay|cup_0|-", "stay|cup_1|-"]
    # host items never appear in any rule
    for rule in rules:
        assert "container" not in rule.item_ids
        assert "plate" not in rule.item_ids
    # totals: 4 singles + 1 category set + 1 all-set + 4 stay
    assert len(ids) == 10
    assert len(ids) == len(set(ids)), "rule_ids must be unique"


def test_enumerate_rules_second_category_set():
    """Two members of one more category -> one more set rule."""
    items = t1_items()
    items["box_1"] = make_item("box_1", "box", color="blue")
    ids = [r.rule_id for r in t1_rules(items)]
    assert "avoid|box_0+box_1|-" in ids
    assert "avoid|cup_0+cup_1|-" in ids
    assert "avoid|bottle_0+box_0+box_1+cup_0+cup_1|-" in ids
    assert len(ids) == 5 + 2 + 1 + 5


def test_enumerate_rules_stable_ids_and_no_digits():
    items = t1_items()
    a = [r.rule_id for r in t1_rules(items)]
    shuffled = dict(reversed(list(items.items())))
    b = [r.rule_id for r in t1_rules(shuffled)]
    assert a == b
    rules = t1_rules()
    for rule in rules:
        assert rule.phrases, rule.rule_id
        for phrase in rule.phrases:
            assert not any(c.isdigit() for c in phrase), phrase


def test_enumerate_rules_params():
    rules = {r.rule_id: r for r in t1_rules()}
    assert rules["avoid|cup_0|-"].params["margin"] == 0.01
    assert rules["stay|cup_0|-"].params["d_tol"] == 0.03
    assert np.isclose(rules["stay|cup_0|-"].params["theta_tol"], np.deg2rad(20))
    assert rules["avoid|cup_0+cup_1|-"].consequence is Consequence.AVOID
    assert rules["avoid|cup_0+cup_1|-"].item_ids == ("cup_0", "cup_1")
    # the all-set baseline margin is the min over members (set semantics)
    assert rules["avoid|bottle_0+box_0+cup_0+cup_1|-"].item_ids == (
        "bottle_0", "box_0", "cup_0", "cup_1"
    )


def test_enumerate_rules_splits():
    rules = {r.rule_id: r for r in t1_rules()}
    # splits.yaml selective_avoid: heldout categories pill_bottle/sauce_can/
    # tea_box — none present here, so everything is train or heldout_phrase
    assert rules["avoid|cup_0|-"].split in ("train", "heldout_phrase")
    items = t1_items()
    items["pill_0"] = make_item("pill_0", "pill_bottle")
    with_pill = {r.rule_id: r for r in t1_rules(items)}
    assert with_pill["avoid|pill_0|-"].split == "heldout_item"


# ---------------------------------------------------------------------------
# runner host argument handling (constructor only — no simulator)
# ---------------------------------------------------------------------------


def test_runner_host_rotates_over_host_tasks_by_seed():
    # no explicit host -> round-robin over the 5 CrossSafe hosts by episode seed
    hosts = get_task_entry("selective_avoid").host_tasks
    for seed, expected in [(0, hosts[0]), (4, hosts[-1]), (5, hosts[0]), (7, hosts[2])]:
        runner = ItemEpisodeRunner("selective_avoid", seed=seed, render="none", out_root="/tmp/t1")
        assert runner.host == expected
        assert runner.task_cfg_path.name == "T1_selective_avoid.yaml"


@pytest.mark.parametrize("host", HOSTS)
def test_runner_accepts_every_registered_host(host):
    runner = ItemEpisodeRunner("selective_avoid", host=host, seed=0, render="none", out_root="/tmp/t1")
    assert runner.host == host


def test_runner_rejects_unregistered_host():
    with pytest.raises(ValueError, match="not an allowed host"):
        ItemEpisodeRunner("selective_avoid", host="place_empty_cup", render="none", out_root="/tmp/t1")


# ---------------------------------------------------------------------------
# spawn report (spawn.py Fix 1) -> meta.diag["spawn"] — mock level, no sim
# ---------------------------------------------------------------------------


def _stub_play_runner(out_root, report):
    """Runner with _play's sim dependencies stubbed; returns (runner, meta)."""
    from types import SimpleNamespace

    from main.item_envs.risk import sample_risk

    runner = ItemEpisodeRunner("selective_avoid", seed=0, render="none", out_root=out_root)
    runner.task_cfg = load_t1()
    runner.risk_params = sample_risk(runner.task_cfg, np.random.RandomState(0))
    task = SimpleNamespace(
        scene=SimpleNamespace(step=lambda: None),
        play_once=lambda: None,
        check_success=lambda: True,
    )
    runner.env = SimpleNamespace(task=task, control_freq=25.0)
    runner.ctrl = SimpleNamespace(n_plan_failures=0)
    runner._spawn_report = report
    runner._capture_tick = lambda *a, **k: None  # the sim-dependent capture
    runner.out_dir.mkdir(parents=True, exist_ok=True)
    return runner, SimpleNamespace(diag={})


def test_spawn_diag_surfaces_fix1_fields(tmp_path):
    """spawn.py Fix 1 (adaptive placement + quota degradation): the runner
    copies the spawn report verbatim into meta.diag["spawn"], so the new
    t_fallbacks / degraded / k_dropped counters reach meta.json. A partial-K
    layout (k=2 with the quota met) is VALID — no infeasible flag."""
    report = {
        "spawn_rejections": 4,
        "layout_retries": 3,
        "settle_rejections": 1,
        "max_layouts_hit": False,
        "static_objects": False,
        "k": 2,
        "t_fallbacks": 1,
        "degraded": 1,
        "k_dropped": 1,
        "placements": [
            {"item_id": "obj_0", "cls": "on_path", "t_fallback": True, "degraded_from": None},
            {"item_id": "obj_1", "cls": "near_path", "t_fallback": False,
             "degraded_from": "on_path"},
        ],
        "settle": [],
        "corridor": {"p0": [0.0, 0.0], "p1": [0.3, 0.0]},
    }
    runner, meta = _stub_play_runner(tmp_path, report)
    assert runner._play(object(), meta) is None
    assert meta.diag["spawn"] is report
    assert meta.diag["spawn"]["t_fallbacks"] == 1
    assert meta.diag["spawn"]["degraded"] == 1
    assert meta.diag["spawn"]["k_dropped"] == 1
    assert meta.spawn_rejections == 4
    assert "spawn_layout_infeasible" not in meta.diag


def test_spawn_diag_marks_zero_candidate_layout_infeasible(tmp_path):
    """k=0 (every layout failed even with Fix 1) still sets the infeasible flag."""
    report = {
        "spawn_rejections": 39,
        "layout_retries": 39,
        "settle_rejections": 0,
        "max_layouts_hit": False,
        "k": 0,
        "t_fallbacks": 0,
        "degraded": 0,
        "k_dropped": 0,
        "placements": [],
    }
    runner, meta = _stub_play_runner(tmp_path, report)
    assert runner._play(object(), meta) is None
    assert meta.diag["spawn"]["k"] == 0
    assert meta.diag["spawn_layout_infeasible"] is True
    assert meta.spawn_rejections == 39
