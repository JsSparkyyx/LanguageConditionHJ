"""Pure-numpy tests for main.item_envs.rules (+ items.entity_of duck-typing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs.items import ItemSpec, entity_of
from main.item_envs.regions import Region
from main.item_envs.rules import (
    Consequence,
    RuleSpec,
    assign_split,
    enumerate_rules,
    render_phrases,
    resolve_splits,
)

CONFIGS = Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs"

OBJECTS_CFG = {
    "colors": {"train": ["red", "green", "blue"], "heldout": ["purple", "orange", "cyan"]},
    "categories": {
        "bottle": {"names": ["bottle"], "synonyms": ["flask"], "mass_range": [0.05, 0.30]},
        "cup": {"names": ["cup"], "synonyms": ["glass"], "mass_range": [0.05, 0.25]},
        "box": {"names": ["box"], "synonyms": ["crate"], "mass_range": [0.05, 0.15], "colored": True},
        "beads": {"names": ["beads"], "synonyms": ["beans", "rice", "grains"], "mass_range": [0.0005, 0.002]},
    },
    "assets": {"001_bottle": {"category": "bottle"}},
    "regions": {
        "table_top": {"names": ["table"], "synonyms": ["tabletop"]},
        "zone": {"names": ["zone", "mat"], "synonyms": [], "colored": True},
        "bowl_src_interior": {"names": ["source bowl", "bowl"], "synonyms": []},
        "target_interior": {"names": ["target", "target container"], "synonyms": ["container"]},
    },
}

SQUARE = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]])


def make_item(item_id, category, role="candidate", color=None, kind="rigid", group=None, **attrs):
    item = ItemSpec(
        item_id=item_id, entity=object(), category=category, color=color, kind=kind, role=role, group=group
    )
    for key, value in attrs.items():
        setattr(item, key, value)
    return item


def make_region(region_id, kind="surface", nameable=False, color=None):
    return Region(
        region_id=region_id,
        kind=kind,
        frame_fn=lambda: np.eye(4),
        polygon=SQUARE,
        z=0.0,
        z_band=0.01,
        nameable=nameable,
        color=color,
    )


def t1_scene():
    """T1-flavoured scene: four candidates, two sharing a category."""
    items = {
        "cup_0": make_item("cup_0", "cup"),
        "cup_1": make_item("cup_1", "cup"),
        "bottle_0": make_item("bottle_0", "bottle"),
        "pill_0": make_item("pill_0", "pill_bottle"),
    }
    return items, {}


def t3_scene():
    """T3-flavoured scene: two push objects, one bystander, two zones."""
    items = {
        "box_0": make_item("box_0", "box", color="red", group="push"),
        "box_1": make_item("box_1", "box", color="blue", group="push"),
        "bottle_1": make_item("bottle_1", "bottle", role="bystander"),
    }
    regions = {
        "table_top": make_region("table_top"),
        "zone_red": make_region("zone_red", kind="zone", nameable=True, color="red"),
        "zone_green": make_region("zone_green", kind="zone", nameable=True, color="green"),
    }
    return items, regions


def t2_scene():
    """T2-flavoured scene: one bead particle set, two containers, two zones."""
    items = {"beads": make_item("beads", "beads", kind="particle_set")}
    regions = {
        "bowl_src_interior": make_region("bowl_src_interior", kind="container", nameable=True),
        "target_interior": make_region("target_interior", kind="container", nameable=True),
        "zone_red": make_region("zone_red", kind="zone", nameable=True, color="red"),
        "zone_blue": make_region("zone_blue", kind="zone", nameable=True, color="blue"),
    }
    return items, regions


# ---------------------------------------------------------------------------
# Consequence / RuleSpec contract


def test_consequence_avoid_exists():
    assert Consequence.AVOID.value == "avoid"
    assert set(Consequence) == {
        Consequence.AVOID,
        Consequence.INSIDE,
        Consequence.OUTSIDE,
        Consequence.UPRIGHT,
        Consequence.STAY,
        Consequence.PARTICLES_INSIDE,
        Consequence.PARTICLES_OUTSIDE,
    }


def test_rulespec_item_ids_tuple():
    rule = RuleSpec(
        rule_id="avoid|obj_2|-",
        consequence="avoid",
        item_ids=["obj_2"],  # lists are coerced
        region_ids=[],
        params={"margin": 0.0},
    )
    assert rule.item_ids == ("obj_2",)
    assert rule.region_ids == ()
    assert rule.consequence is Consequence.AVOID
    assert rule.phrases == [] and rule.split == "train"


# ---------------------------------------------------------------------------
# enumerate_rules: selectors and set rules


def test_enumerate_single_avoid_rules():
    items, regions = t1_scene()
    task_cfg = {"rules": [{"consequence": "avoid", "items": {"role": "candidate"}, "margin": 0.0}]}
    rules = enumerate_rules(items, regions, task_cfg)
    assert [r.rule_id for r in rules] == [
        "avoid|bottle_0|-",
        "avoid|cup_0|-",
        "avoid|cup_1|-",
        "avoid|pill_0|-",
    ]
    for rule in rules:
        assert rule.consequence is Consequence.AVOID
        assert rule.region_ids == ()
        assert rule.params["margin"] == 0.0
        assert len(rule.item_ids) == 1
        assert rule.params["categories"] == (rule.params["category"],)


def test_enumerate_set_rule_same_category_min():
    items, regions = t1_scene()
    task_cfg = {
        "rules": [
            {"consequence": "avoid", "items": {"role": "candidate", "same_category_min": 2},
             "set_rule": True, "margin": 0.0}
        ]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    # only "cup" has >= 2 members -> one set rule carrying all its members
    assert [r.rule_id for r in rules] == ["avoid|cup_0+cup_1|-"]
    rule = rules[0]
    assert rule.item_ids == ("cup_0", "cup_1")
    assert rule.params["category"] == "cup"
    assert rule.params["categories"] == ("cup", "cup")
    assert "color" not in rule.params  # members have no color


def test_enumerate_set_rule_all():
    items, regions = t1_scene()
    task_cfg = {
        "rules": [{"consequence": "avoid", "items": {"role": "candidate"}, "set_rule": "all", "margin": 0.0}]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    assert [r.rule_id for r in rules] == ["avoid|bottle_0+cup_0+cup_1+pill_0|-"]
    rule = rules[0]
    assert rule.item_ids == ("bottle_0", "cup_0", "cup_1", "pill_0")
    assert "category" not in rule.params  # mixed set: no uniform category
    assert set(rule.params["categories"]) == {"bottle", "cup", "pill_bottle"}


def test_enumerate_set_rule_true_defaults_min_two():
    items, regions = t1_scene()
    # set_rule: true without same_category_min still requires >= 2 per category
    task_cfg = {"rules": [{"consequence": "avoid", "items": {"role": "candidate"}, "set_rule": True}]}
    assert [r.rule_id for r in enumerate_rules(items, regions, task_cfg)] == ["avoid|cup_0+cup_1|-"]
    # no category repeats -> no set rule at all
    single = {"bottle_0": items["bottle_0"]}
    assert enumerate_rules(single, regions, task_cfg) == []


def test_enumerate_group_selector_and_region_list():
    items, regions = t3_scene()
    task_cfg = {
        "rules": [
            {"consequence": "inside", "items": {"role": "candidate", "group": "push"}, "regions": ["table_top"]},
            {"consequence": "upright", "items": {"role": "bystander"}, "theta_max_deg": 45},
            {"consequence": "stay", "items": {"role": "bystander"}, "d_tol": 0.03, "theta_tol_deg": 20},
        ]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    assert [r.rule_id for r in rules] == [
        "inside|box_0|table_top",
        "inside|box_1|table_top",
        "upright|bottle_1|-",
        "stay|bottle_1|-",
    ]
    assert np.isclose(rules[2].params["theta_max"], np.deg2rad(45))  # *_deg converted to radians
    assert rules[3].params["d_tol"] == 0.03
    assert np.isclose(rules[3].params["theta_tol"], np.deg2rad(20))


def test_enumerate_region_kind_selector_expands_per_region():
    items, regions = t3_scene()
    task_cfg = {
        "rules": [
            {"consequence": "outside", "items": {"role": "candidate", "group": "push"}, "regions": {"kind": "zone"}}
        ]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    # one rule per (push object, zone); zones sorted by region_id
    assert [r.rule_id for r in rules] == [
        "outside|box_0|zone_green",
        "outside|box_0|zone_red",
        "outside|box_1|zone_green",
        "outside|box_1|zone_red",
    ]


def test_enumerate_particles_single_joint_region_rule():
    items, regions = t2_scene()
    task_cfg = {
        "rules": [
            {"consequence": "particles_inside", "items": {"role": "candidate", "kind": "particle_set"},
             "regions": ["bowl_src_interior", "target_interior"]},
            {"consequence": "particles_outside", "items": {"role": "candidate", "kind": "particle_set"},
             "regions": {"kind": "zone"}},
        ]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    assert [r.rule_id for r in rules] == [
        "particles_inside|beads|bowl_src_interior+target_interior",
        "particles_outside|beads|zone_blue+zone_red",
    ]
    assert rules[0].region_ids == ("bowl_src_interior", "target_interior")
    assert rules[0].params["T"] == 0.005  # default soft-min temperature
    assert rules[1].params["T"] == 0.005


def test_enumerate_missing_regions_skips_not_errors():
    items, _ = t3_scene()
    regions = {"table_top": make_region("table_top")}  # no zones in this scene
    task_cfg = {
        "rules": [
            {"consequence": "outside", "items": {"role": "candidate"}, "regions": ["zone_red", "table_top"]},
            {"consequence": "outside", "items": {"role": "candidate"}, "regions": {"kind": "zone"}},
        ]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    # zone_red is absent from the scene -> skipped; the {kind: zone} entry matches nothing
    assert [r.rule_id for r in rules] == [
        "outside|box_0|table_top",
        "outside|box_1|table_top",
    ]


def test_enumerate_stable_across_input_order():
    items, regions = t1_scene()
    task_cfg = {
        "rules": [
            {"consequence": "avoid", "items": {"role": "candidate"}, "margin": 0.0},
            {"consequence": "avoid", "items": {"role": "candidate", "same_category_min": 2}, "set_rule": True},
            {"consequence": "stay", "items": {"role": "candidate"}, "d_tol": 0.03},
        ]
    }
    shuffled = dict(reversed(list(items.items())))
    a = [r.rule_id for r in enumerate_rules(items, regions, task_cfg)]
    b = [r.rule_id for r in enumerate_rules(shuffled, regions, task_cfg)]
    assert a == b


def test_enumerate_config_errors_raise():
    items, regions = t1_scene()
    with pytest.raises(ValueError):
        enumerate_rules(items, regions, {"rules": [{"consequence": "upright", "items": {}}]})
    with pytest.raises(ValueError):  # regional consequence without a regions selector
        enumerate_rules(items, regions, {"rules": [{"consequence": "inside", "items": {}}]})


def test_stay_default_theta_tol():
    items, regions = t1_scene()
    task_cfg = {"rules": [{"consequence": "stay", "items": {"role": "candidate"}, "d_tol": 0.05}]}
    rules = enumerate_rules(items, regions, task_cfg)
    assert rules[0].params["theta_tol"] == 0.35  # documented default


# ---------------------------------------------------------------------------
# render_phrases


def _rule(consequence, item_ids=("bottle_0",), region_ids=(), params=None, phrases=()):
    return RuleSpec(
        rule_id=f"{consequence}|{'+'.join(item_ids)}|{'+'.join(region_ids) if region_ids else '-'}",
        consequence=Consequence(consequence),
        item_ids=tuple(item_ids),
        region_ids=tuple(region_ids),
        params=params if params is not None else {"category": "bottle", "categories": ("bottle",), "colors": (None,)},
        phrases=list(phrases),
    )


def test_render_phrases_no_digits_all_consequences():
    cases = [
        _rule("avoid"),
        _rule("inside", region_ids=("table_top",)),
        _rule("outside", region_ids=("zone_red",)),
        _rule("upright"),
        _rule("stay"),
        _rule("particles_inside", item_ids=("beads",), region_ids=("bowl_src_interior", "target_interior"),
              params={"category": "beads", "categories": ("beads",), "colors": (None,)}),
        _rule("particles_outside", item_ids=("beads",), region_ids=("zone_red",),
              params={"category": "beads", "categories": ("beads",), "colors": (None,)}),
        _rule("avoid", item_ids=("cup_0", "cup_1"),
              params={"category": "cup", "categories": ("cup", "cup"), "colors": (None, None)}),  # set rule
    ]
    for rule in cases:
        phrases = render_phrases(rule, OBJECTS_CFG, np.random.RandomState(0), n=3)
        assert phrases, rule.rule_id
        for phrase in phrases:
            assert not any(c.isdigit() for c in phrase), phrase


def test_render_phrases_avoid_templates_and_determinism():
    rule = _rule("avoid")
    a = render_phrases(rule, OBJECTS_CFG, np.random.RandomState(1), n=3)
    b = render_phrases(rule, OBJECTS_CFG, np.random.RandomState(1), n=3)
    assert a == b
    for phrase in a:
        assert any(phrase.startswith(p) for p in ("don't touch the", "avoid the", "stay away from the"))
        assert any(nm in phrase for nm in ("bottle", "flask"))


def test_render_phrases_color_prefix_and_region_color():
    rule = _rule(
        "outside",
        item_ids=("box_0",),
        region_ids=("zone_blue",),
        params={"category": "box", "categories": ("box",), "colors": ("red",), "color": "red"},
    )
    seen = set()
    for seed in range(10):
        seen.update(render_phrases(rule, OBJECTS_CFG, np.random.RandomState(seed), n=3))
    assert any("red box" in p or "red crate" in p for p in seen)
    assert any("blue zone" in p or "blue mat" in p for p in seen)
    assert all(not any(c.isdigit() for c in p) for p in seen)


def test_render_phrases_set_rule_uses_category_templates():
    rule = _rule(
        "avoid",
        item_ids=("cup_0", "cup_1"),
        params={"category": "cup", "categories": ("cup", "cup"), "colors": (None, None)},
    )
    seen = set()
    for seed in range(10):
        seen.update(render_phrases(rule, OBJECTS_CFG, np.random.RandomState(seed), n=3))
    assert seen  # set templates only
    assert all(p.startswith("don't touch any ") or p.startswith("avoid all the ") for p in seen)
    assert any("cup" in p or "glass" in p for p in seen)


def test_render_phrases_mixed_set_falls_back_to_generic():
    rule = _rule(
        "avoid",
        item_ids=("bottle_0", "cup_0"),
        params={"categories": ("bottle", "cup"), "colors": (None, None)},  # no uniform category
    )
    seen = set()
    for seed in range(10):
        seen.update(render_phrases(rule, OBJECTS_CFG, np.random.RandomState(seed), n=3))
    assert seen
    assert all(any(g in p for g in ("object", "item")) for p in seen)
    assert all(not any(c.isdigit() for c in p) for p in seen)


def test_render_phrases_particles_inside_names_no_region():
    rule = _rule("particles_inside", item_ids=("beads",), region_ids=("bowl_src_interior", "target_interior"),
                 params={"category": "beads", "categories": ("beads",), "colors": (None,)})
    phrases = render_phrases(rule, OBJECTS_CFG, np.random.RandomState(0), n=3)
    assert phrases
    assert all(p.startswith("don't spill the") for p in phrases)
    assert any("beans" in p or "rice" in p or "grains" in p or "beads" in p for p in phrases)


# ---------------------------------------------------------------------------
# assign_split / resolve_splits


SPLITS = {
    "heldout_item_categories": ["pill_bottle", "sauce_can", "tea_box"],
    "heldout_colors": ["purple", "orange", "cyan"],
    "heldout_regions": ["laptop"],
    "heldout_phrase_templates": ["stay away from the {item}", "don't get any {item} in the {region}"],
}


def test_assign_split_heldout_item_category():
    rule = _rule("avoid", item_ids=("pill_0",),
                 params={"category": "pill_bottle", "categories": ("pill_bottle",), "colors": (None,)})
    assert assign_split(rule, SPLITS) == "heldout_item"
    # hyphen in asset-flavoured ids normalizes to the category alias token
    rule = _rule("avoid", item_ids=("105_sauce-can_0",),
                 params={"category": "sauce_can", "categories": ("sauce_can",), "colors": (None,)})
    assert assign_split(rule, SPLITS) == "heldout_item"


def test_assign_split_heldout_color():
    rule = _rule("avoid", item_ids=("box_0",),
                 params={"category": "box", "categories": ("box",), "colors": ("purple",), "color": "purple"})
    assert assign_split(rule, SPLITS) == "heldout_item"


def test_assign_split_set_rule_any_member_heldout():
    rule = _rule(
        "avoid",
        item_ids=("cup_0", "pill_0"),
        params={"categories": ("cup", "pill_bottle"), "colors": (None, None)},
        phrases=["avoid all the objects"],
    )
    assert assign_split(rule, SPLITS) == "heldout_item"


def test_assign_split_heldout_region_and_precedence():
    region_rule = _rule("outside", region_ids=("laptop",),
                        params={"category": "beads", "categories": ("beads",), "colors": (None,)})
    assert assign_split(region_rule, SPLITS) == "heldout_region"
    # heldout item wins over heldout region
    both = _rule("outside", item_ids=("pill_0",), region_ids=("laptop",),
                 params={"category": "pill_bottle", "categories": ("pill_bottle",), "colors": (None,)})
    assert assign_split(both, SPLITS) == "heldout_item"


def test_assign_split_heldout_phrase_template():
    rule = _rule("avoid", phrases=["stay away from the bottle"])
    assert assign_split(rule, SPLITS) == "heldout_phrase"
    rule = _rule("particles_outside", item_ids=("beads",), region_ids=("zone_red",),
                 params={"category": "beads", "categories": ("beads",), "colors": (None,)},
                 phrases=["don't get any beans in the red zone"])
    assert assign_split(rule, SPLITS) == "heldout_phrase"
    assert assign_split(_rule("avoid", phrases=["avoid the bottle"]), SPLITS) == "train"
    assert assign_split(_rule("avoid"), None) == "train"


def test_resolve_splits_merges_all_under_task():
    splits_yaml = {
        "selective_avoid": {"heldout_item_categories": ["tea_box"], "heldout_colors": ["cyan"]},
        "all": {"heldout_phrase_templates": ["stay away from the {item}"]},
    }
    merged = resolve_splits(splits_yaml, "selective_avoid")
    assert merged == {
        "heldout_item_categories": ["tea_box"],
        "heldout_colors": ["cyan"],
        "heldout_phrase_templates": ["stay away from the {item}"],
    }
    assert resolve_splits(None, "nothing") == {}


def test_enumerate_rules_fills_phrases_and_split():
    items, regions = t1_scene()
    task_cfg = {"rules": [{"consequence": "avoid", "items": {"role": "candidate"}, "margin": 0.0}]}
    rules = enumerate_rules(
        items, regions, task_cfg, objects_cfg=OBJECTS_CFG, splits_cfg=SPLITS, rng=np.random.RandomState(0)
    )
    by_id = {r.rule_id: r for r in rules}
    assert by_id["avoid|pill_0|-"].split == "heldout_item"
    assert by_id["avoid|bottle_0|-"].split in ("train", "heldout_phrase")
    assert all(len(r.phrases) > 0 for r in rules)
    assert all(not any(c.isdigit() for c in p) for r in rules for p in r.phrases)


# ---------------------------------------------------------------------------
# Real configs on disk


def test_objects_yaml_structure_and_v3_mass_ranges():
    cfg = yaml.safe_load((CONFIGS / "objects.yaml").read_text())
    pool = [
        "068_boxdrink", "086_woodenblock", "001_bottle", "114_bottle", "021_cup", "039_mug",
        "071_can", "080_pillbottle", "105_sauce-can", "112_tea-box", "113_coffee-box",
    ]
    for asset_id in pool:
        assert asset_id in cfg["assets"], asset_id
        assert cfg["assets"][asset_id]["category"] in cfg["categories"], asset_id
    # nominal mass ranges from PLAN v3 §7
    assert cfg["categories"]["bottle"]["mass_range"] == [0.05, 0.30]
    assert cfg["categories"]["cup"]["mass_range"] == [0.05, 0.25]
    assert cfg["categories"]["mug"]["mass_range"] == [0.05, 0.25]
    assert cfg["categories"]["can"]["mass_range"] == [0.05, 0.35]
    assert cfg["categories"]["wooden_block"]["mass_range"] == [0.05, 0.15]
    for boxed in ("drink_box", "tea_box", "coffee_box"):
        assert cfg["categories"][boxed]["mass_range"] == [0.05, 0.30], boxed
    assert cfg["categories"]["beads"]["mass_range"] == [0.0005, 0.002]
    assert set(cfg["categories"]["beads"]["synonyms"]) == {"beans", "rice", "grains"}
    # 6 training colors + 3 heldout
    assert len(cfg["colors"]["train"]) == 6
    assert set(cfg["colors"]["heldout"]) == {"purple", "orange", "cyan"}
    # names/synonyms never contain digits
    for key, entry in cfg["categories"].items():
        for nm in list(entry.get("names", [])) + list(entry.get("synonyms", [])):
            assert not any(c.isdigit() for c in nm), (key, nm)


def test_splits_yaml_is_v3_excerpt():
    cfg = yaml.safe_load((CONFIGS / "splits.yaml").read_text())
    assert cfg == {
        "selective_avoid": {
            "heldout_item_categories": ["pill_bottle", "sauce_can", "tea_box"],
            "heldout_colors": ["purple", "orange", "cyan"],
        },
        "granular_pour": {
            "heldout_regions": ["laptop"],
            "heldout_colors": ["purple", "orange", "cyan"],
        },
        "push_zone": {
            "heldout_colors": ["purple", "orange", "cyan"],
            "heldout_item_categories": ["tea_box"],
        },
        "granular_carry": {"heldout_colors": ["purple", "orange", "cyan"]},  # T4 (2026-10-03)
        "tray_transport": {"heldout_colors": ["purple", "orange", "cyan"]},  # T5 (2026-10-04)
        "clean_pour": {"heldout_colors": ["purple", "orange", "cyan"]},  # T6 (2026-10-04)
        "gap_place": {"heldout_colors": ["purple", "orange", "cyan"]},  # T7 (2026-10-04)
        "all": {
            "heldout_phrase_templates": ["stay away from the {item}", "don't get any {item} in the {region}"],
        },
    }


def test_task_yamls_parse_and_rule_entries_enumerate():
    objects_cfg = yaml.safe_load((CONFIGS / "objects.yaml").read_text())
    splits_yaml = yaml.safe_load((CONFIGS / "splits.yaml").read_text())

    # T1: avoid singles + category set rules + the CrossSafe-baseline all-set + stay
    t1 = yaml.safe_load((CONFIGS / "tasks" / "T1_selective_avoid.yaml").read_text())
    assert t1["task"] == "selective_avoid"
    items, _ = t1_scene()
    rules = enumerate_rules(
        items, {}, t1, objects_cfg=objects_cfg,
        splits_cfg=resolve_splits(splits_yaml, "selective_avoid"), rng=np.random.RandomState(0),
    )
    ids = [r.rule_id for r in rules]
    assert "avoid|cup_0+cup_1|-" in ids  # same_category_min set rule
    assert "avoid|bottle_0+cup_0+cup_1+pill_0|-" in ids  # set_rule: all baseline
    assert "stay|cup_0|-" in ids
    assert len(ids) == 4 + 1 + 1 + 4
    by_id = {r.rule_id: r for r in rules}
    assert by_id["avoid|pill_0|-"].split == "heldout_item"
    assert by_id["avoid|bottle_0+cup_0+cup_1+pill_0|-"].split == "heldout_item"  # pill_0 is a member
    assert all(r.phrases and not any(c.isdigit() for c in p) for r in rules for p in r.phrases)

    # T2: one joint particles_inside over both containers — the only rule
    # (forbidden-mat rules dropped, user decision 2026-10-03)
    t2 = yaml.safe_load((CONFIGS / "tasks" / "T2_granular_pour.yaml").read_text())
    assert t2["task"] == "granular_pour"
    assert t2["bimanual"] is False
    items, regions = t2_scene()
    rules = enumerate_rules(
        items, regions, t2, objects_cfg=objects_cfg,
        splits_cfg=resolve_splits(splits_yaml, "granular_pour"), rng=np.random.RandomState(0),
    )
    assert [r.rule_id for r in rules] == [
        "particles_inside|beads|bowl_src_interior+target_interior",
    ]
    assert all(r.phrases and not any(c.isdigit() for c in p) for r in rules for p in r.phrases)

    # T3: group: push selector, {kind: zone} expansion, bystander upright/stay
    t3 = yaml.safe_load((CONFIGS / "tasks" / "T3_push_zone.yaml").read_text())
    assert t3["task"] == "push_zone"
    items, regions = t3_scene()
    rules = enumerate_rules(
        items, regions, t3, objects_cfg=objects_cfg,
        splits_cfg=resolve_splits(splits_yaml, "push_zone"), rng=np.random.RandomState(0),
    )
    ids = [r.rule_id for r in rules]
    assert ids == [
        "inside|box_0|table_top",
        "inside|box_1|table_top",
        "outside|box_0|zone_green",
        "outside|box_0|zone_red",
        "outside|box_1|zone_green",
        "outside|box_1|zone_red",
        "upright|bottle_1|-",
        "stay|bottle_1|-",
        # region_set: pairs (A-A scenarios); one bystander -> no stay pair
        "outside|box_0|zone_green+zone_red",
        "outside|box_1|zone_green+zone_red",
    ]
    assert all(r.phrases and not any(c.isdigit() for c in p) for r in rules for p in r.phrases)
    # a tea-box push object would be held out
    items = {**items, "tea_0": make_item("tea_0", "tea_box", group="push")}
    rules = enumerate_rules(items, regions, t3, splits_cfg=resolve_splits(splits_yaml, "push_zone"))
    by_id = {r.rule_id: r for r in rules}
    assert by_id["inside|tea_0|table_top"].split == "heldout_item"


def test_entity_of_duck_typing():
    class Raw:
        def get_pose(self):
            return "pose"

    class Wrapper:  # RoboTwin Actor shape
        def __init__(self, raw):
            self.actor = raw

        def get_pose(self):
            return self.actor.get_pose()

    raw = Raw()
    assert entity_of(raw) is raw
    assert entity_of(Wrapper(raw)) is raw
    item = make_item("bottle_0", "bottle")
    item.entity = Wrapper(raw)
    assert entity_of(item) is raw
    plain = object()
    assert entity_of(plain) is plain  # no wrapper attributes -> returned as-is


def test_pair_rule_modes_and_joint_phrases():
    """set_rule: pairs / region_set: pairs emit one rule per 2-combination,
    and every phrase names both members."""
    objects_cfg = yaml.safe_load((CONFIGS / "objects.yaml").read_text())
    items, regions = t3_scene()
    items = {**items, "cyl_0": make_item("cyl_0", "cylinder", color="blue", role="bystander")}
    cfg = {"rules": [
        {"consequence": "outside", "items": {"role": "candidate", "group": "push"},
         "regions": {"kind": "zone"}, "region_set": "pairs"},
        {"consequence": "stay", "items": {"role": "bystander"}, "set_rule": "pairs",
         "d_tol": 0.03, "theta_tol_deg": 20},
    ]}
    rules = enumerate_rules(items, regions, cfg, objects_cfg=objects_cfg, rng=np.random.RandomState(0))
    by_id = {r.rule_id: r for r in rules}
    assert set(by_id) == {
        "outside|box_0|zone_green+zone_red", "outside|box_1|zone_green+zone_red",
        "stay|bottle_1+cyl_0|-",
    }
    for phrase in by_id["outside|box_0|zone_green+zone_red"].phrases:
        assert "green" in phrase and "red" in phrase, phrase
    stay = by_id["stay|bottle_1+cyl_0|-"]
    assert stay.params["theta_tol"] == pytest.approx(np.deg2rad(20))
    for phrase in stay.phrases:
        assert "blue" in phrase and " and the " in phrase, phrase
    with pytest.raises(ValueError):
        enumerate_rules(items, regions, {"rules": [
            {"consequence": "outside", "items": {"group": "push"}, "regions": {"kind": "zone"}, "region_set": "bogus"}]})


def test_phrases_never_use_a_name_another_scene_item_answers_to():
    """T5 audit (2026-10-04): with two blocks on the tray, "don't knock over
    the brick" named no referent. Bare / synonym names shared with another
    item are dropped; distinct colors keep the phrase unique; a category
    that is unique in the scene keeps its bare names."""
    from main.item_envs.items import ItemSpec

    cfg = {"rules": [{"consequence": "upright", "items": {"role": "candidate"}, "theta_max_deg": 10}]}
    items = {
        "block_0": ItemSpec(item_id="block_0", entity=object(), category="block", color="blue", role="candidate"),
        "block_1": ItemSpec(item_id="block_1", entity=object(), category="block", color="red", role="candidate"),
        "can_0": ItemSpec(item_id="can_0", entity=object(), category="can", color="", role="candidate"),
    }
    import yaml
    from pathlib import Path

    real = yaml.safe_load((Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs" / "objects.yaml")
                          .read_text(encoding="utf-8"))
    rules = {r.rule_id: r for r in enumerate_rules(items, {}, cfg, objects_cfg=real)}
    for rid, color in (("upright|block_0|-", "blue"), ("upright|block_1|-", "red")):
        for phrase in rules[rid].phrases:
            assert f"{color} " in phrase, phrase  # always color-qualified
    assert any(" can" in p or " tin" in p for p in rules["upright|can_0|-"].phrases)


def test_rule_without_a_unique_name_is_flagged_ambiguous():
    """Two uncolored bottles: no name tells them apart — their single-item
    rules are flagged (the runner never makes them the primary rule); a
    uniquely named item's rule is not."""
    import yaml
    from pathlib import Path

    from main.item_envs.items import ItemSpec

    real = yaml.safe_load((Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs" / "objects.yaml")
                          .read_text(encoding="utf-8"))
    cfg = {"rules": [{"consequence": "avoid", "items": {"role": "candidate"}, "margin": 0.0}]}
    items = {
        "obj_0": ItemSpec(item_id="obj_0", entity=object(), category="bottle", color="", role="candidate"),
        "obj_1": ItemSpec(item_id="obj_1", entity=object(), category="bottle", color="", role="candidate"),
        "obj_2": ItemSpec(item_id="obj_2", entity=object(), category="can", color="", role="candidate"),
    }
    rules = {r.rule_id: r for r in enumerate_rules(items, {}, cfg, objects_cfg=real)}
    assert rules["avoid|obj_0|-"].ambiguous and rules["avoid|obj_1|-"].ambiguous
    assert not rules["avoid|obj_2|-"].ambiguous


def test_region_phrases_never_use_a_name_another_scene_region_answers_to():
    """Audit (2026-10-04): with two zones, "keep the box out of the zone"
    named neither — the bare kind words are dropped, the colored ones stay;
    a region kind unique in the scene keeps its bare names."""
    import yaml
    from pathlib import Path

    from main.item_envs.items import ItemSpec
    from main.item_envs.regions import Region

    real = yaml.safe_load((Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs" / "objects.yaml")
                          .read_text(encoding="utf-8"))
    sq = [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]]
    regions = {rid: Region(region_id=rid, kind=kind, frame_fn=lambda: np.eye(4), polygon=sq, z=0.0, z_band=0.01,
                           nameable=True, color=color)
               for rid, kind, color in (("zone_red", "zone", "red"), ("zone_blue", "zone", "blue"),
                                        ("table_top", "surface", None))}
    items = {"push_0": ItemSpec(item_id="push_0", entity=object(), category="box", color="green", role="candidate")}
    cfg = {"rules": [{"consequence": "outside", "items": {"role": "candidate"}, "regions": ["zone_red", "zone_blue"]},
                     {"consequence": "inside", "items": {"role": "candidate"}, "regions": ["table_top"]}]}
    rules = {r.rule_id: r for r in enumerate_rules(items, regions, cfg, objects_cfg=real)}
    for rid, color in (("outside|push_0|zone_red", "red"), ("outside|push_0|zone_blue", "blue")):
        assert rules[rid].phrases
        for phrase in rules[rid].phrases:
            assert f"{color} zone" in phrase or f"{color} mat" in phrase or f"{color} marked area" in phrase, phrase
    assert any("table" in p for p in rules["inside|push_0|table_top"].phrases)


def test_particle_rule_region_set_each_gives_one_rule_per_region():
    """region_set: each (T6) — a particle rule per region instead of one
    joint rule over all of them."""
    from main.item_envs.items import ItemSpec
    from main.item_envs.regions import Region

    sq = [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]]
    regions = {rid: Region(region_id=rid, kind="container", frame_fn=lambda: np.eye(4), polygon=sq, z=0.0,
                           z_band=0.01, nameable=True)
               for rid in ("dish_a_interior", "dish_b_interior")}
    items = {"beads": ItemSpec(item_id="beads", entity=object(), category="beads", color="", role="candidate",
                               kind="particle_set")}
    each = {"rules": [{"consequence": "particles_outside", "items": {"kind": "particle_set"},
                       "regions": ["dish_a_interior", "dish_b_interior"], "region_set": "each", "T": 0.001}]}
    joint = {"rules": [{"consequence": "particles_outside", "items": {"kind": "particle_set"},
                        "regions": ["dish_a_interior", "dish_b_interior"], "T": 0.001}]}
    assert [r.rule_id for r in enumerate_rules(items, regions, each)] == [
        "particles_outside|beads|dish_a_interior", "particles_outside|beads|dish_b_interior"]
    assert [r.rule_id for r in enumerate_rules(items, regions, joint)] == [
        "particles_outside|beads|dish_a_interior+dish_b_interior"]
