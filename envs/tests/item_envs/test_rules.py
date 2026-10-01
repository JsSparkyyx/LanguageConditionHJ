"""Pure-numpy tests for main.item_envs.rules (+ items.entity_of duck-typing)."""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.items import ItemSpec, entity_of
from main.item_envs.regions import Region
from main.item_envs.rules import (
    Consequence,
    RuleSpec,
    assign_split,
    enumerate_rules,
    render_phrases,
)

PHRASES_CFG = {
    "colors": ["red", "blue", "purple"],
    "items": {
        "bottle": {"names": ["bottle"], "synonyms": ["flask"]},
        "block": {"names": ["block", "wooden block"], "synonyms": ["cube"]},
        "beans": {"names": ["beans"], "synonyms": ["beads", "rice"]},
        "ball": {"names": ["ball", "sphere"], "synonyms": []},
    },
    "regions": {
        "tray_top": {"names": ["tray"], "synonyms": ["serving tray"]},
        "table_top": {"names": ["table"], "synonyms": ["tabletop"]},
        "cup_interior": {"names": ["cup"], "synonyms": []},
        "box_interior": {"names": ["box"], "synonyms": ["container"]},
        "zone": {"names": ["zone", "mat"], "synonyms": [], "colored": True},
    },
}

SQUARE = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]])


def make_item(item_id, category, role="constrained_candidate", color=None, kind="rigid", **attrs):
    item = ItemSpec(item_id=item_id, entity=object(), category=category, color=color, kind=kind, role=role)
    for key, value in attrs.items():
        setattr(item, key, value)
    return item


def make_region(region_id, kind="surface"):
    return Region(
        region_id=region_id, kind=kind, frame_fn=lambda: np.eye(4), polygon=SQUARE, z=0.0, z_band=0.01
    )


def toy_scene():
    items = {
        "bottle_0": make_item("bottle_0", "bottle", tippable=True),
        "block_0": make_item("block_0", "block", tippable=False),
        "mug_0": make_item("mug_0", "mug", role="bystander", tippable=True),
    }
    regions = {
        "tray_top": make_region("tray_top"),
        "table_top": make_region("table_top"),
        "cup_interior": make_region("cup_interior", kind="container"),
        "box_interior": make_region("box_interior", kind="container"),
    }
    return items, regions


TOY_TASK_CFG = {
    "rules": [
        {"consequence": "inside", "items": {"role": "constrained_candidate"}, "regions": ["tray_top"]},
        {"consequence": "upright", "items": {"role": "constrained_candidate", "tippable": True}, "theta_max_deg": 45},
    ]
}


def test_enumerate_rules_count_and_ids():
    items, regions = toy_scene()
    rules = enumerate_rules(items, regions, TOY_TASK_CFG)
    # inside: bottle_0, block_0 (mug_0 is a bystander); upright: bottle_0 only
    assert [r.rule_id for r in rules] == [
        "inside|block_0|tray_top",
        "inside|bottle_0|tray_top",
        "upright|bottle_0|-",
    ]
    assert rules[0].consequence is Consequence.INSIDE
    assert rules[0].region_ids == ("tray_top",)
    assert rules[2].region_ids == ()
    assert np.isclose(rules[2].params["theta_max"], np.deg2rad(45))
    assert rules[2].params["category"] == "bottle"


def test_enumerate_rules_stable_across_input_order():
    items, regions = toy_scene()
    shuffled = dict(reversed(list(items.items())))
    a = [r.rule_id for r in enumerate_rules(items, regions, TOY_TASK_CFG)]
    b = [r.rule_id for r in enumerate_rules(shuffled, regions, TOY_TASK_CFG)]
    assert a == b


def test_enumerate_rules_particles_single_rule_joined_regions():
    items = {"beans_0": make_item("beans_0", "beans", kind="particle_set")}
    _, regions = toy_scene()
    task_cfg = {
        "rules": [
            {
                "consequence": "particles_inside",
                "items": {"role": "constrained_candidate", "kind": "particle_set"},
                "regions": ["cup_interior", "box_interior"],
            },
            {"consequence": "stay", "items": {"role": "constrained_candidate"}, "d_tol": 0.05},
            {"consequence": "outside", "items": {"role": "constrained_candidate"}, "regions": ["zone_red", "table_top"]},
        ]
    }
    rules = enumerate_rules(items, regions, task_cfg)
    assert [r.rule_id for r in rules] == [
        "particles_inside|beans_0|cup_interior+box_interior",
        "stay|beans_0|-",
        "outside|beans_0|table_top",  # zone_red is not in the scene -> skipped
    ]
    assert rules[0].params["T"] == 0.005
    assert rules[1].params["d_tol"] == 0.05
    assert rules[1].params["theta_tol"] == 0.35  # documented default


def test_enumerate_rules_upright_requires_theta_max():
    items, regions = toy_scene()
    with pytest.raises(ValueError):
        enumerate_rules(items, regions, {"rules": [{"consequence": "upright", "items": {}}]})


def _rule(consequence, item_id="bottle_0", region_ids=(), params=None, phrases=()):
    return RuleSpec(
        rule_id=f"{consequence}|{item_id}|{'+'.join(region_ids) if region_ids else '-'}",
        consequence=Consequence(consequence),
        item_id=item_id,
        region_ids=tuple(region_ids),
        params=params or {"category": "bottle"},
        phrases=list(phrases),
    )


def test_render_phrases_no_digits_all_consequences():
    cases = [
        _rule("inside", region_ids=("tray_top",)),
        _rule("outside", region_ids=("zone_red",)),
        _rule("upright"),
        _rule("stay"),
        _rule("particles_inside", item_id="beans_0", region_ids=("cup_interior", "box_interior"),
              params={"category": "beans"}),
        _rule("particles_outside", item_id="beans_0", region_ids=("table_top",), params={"category": "beans"}),
    ]
    for rule in cases:
        phrases = render_phrases(rule, PHRASES_CFG, np.random.RandomState(0), n=3)
        assert phrases, rule.rule_id
        for phrase in phrases:
            assert not any(c.isdigit() for c in phrase), phrase


def test_render_phrases_content_and_determinism():
    rule = _rule("inside", region_ids=("tray_top",))
    a = render_phrases(rule, PHRASES_CFG, np.random.RandomState(1), n=3)
    b = render_phrases(rule, PHRASES_CFG, np.random.RandomState(1), n=3)
    assert a == b
    for phrase in a:
        assert any(nm in phrase for nm in ("bottle", "flask"))
        assert any(nm in phrase for nm in ("tray", "serving tray"))


def test_render_phrases_color_prefix_and_region_color():
    rule = _rule("outside", params={"category": "ball", "color": "red"}, item_id="ball_0",
                 region_ids=("zone_blue",))
    seen = set()
    for seed in range(10):
        seen.update(render_phrases(rule, PHRASES_CFG, np.random.RandomState(seed), n=3))
    assert any("red ball" in p or "red sphere" in p for p in seen)
    assert any("blue zone" in p or "blue mat" in p for p in seen)
    assert all(not any(c.isdigit() for c in p) for p in seen)


def test_render_phrases_custom_templates_override():
    cfg = dict(PHRASES_CFG)
    cfg["templates"] = {"upright": ["please do not tip the {item}"]}
    rule = _rule("upright")
    phrases = render_phrases(rule, cfg, np.random.RandomState(0), n=2)
    assert phrases
    assert all(p.startswith("please do not tip the") for p in phrases)


def test_assign_split():
    splits = {
        "heldout_item_categories": ["mug", "sauce_can"],
        "heldout_colors": ["purple"],
        "heldout_regions": ["zone"],
        "heldout_phrase_templates": ["don't let the {item} slide off the {region}"],
    }
    assert assign_split(_rule("upright", params={"category": "mug"}, item_id="mug_0"), splits) == "heldout_item"
    # hyphen in asset ids normalizes to the category alias token
    assert assign_split(_rule("upright", params={"category": "105_sauce-can"}, item_id="105_sauce-can_0"), splits) == "heldout_item"
    assert assign_split(_rule("upright", params={"category": "ball", "color": "purple"}, item_id="ball_0"), splits) == "heldout_item"
    assert assign_split(_rule("outside", region_ids=("zone_red",)), splits) == "heldout_region"
    assert (
        assign_split(
            _rule("inside", region_ids=("tray_top",), phrases=["don't let the bottle slide off the tray"]), splits
        )
        == "heldout_phrase"
    )
    assert assign_split(_rule("inside", region_ids=("tray_top",)), splits) == "train"
    assert assign_split(_rule("upright"), None) == "train"


def test_enumerate_rules_fills_phrases_and_split():
    items, regions = toy_scene()
    splits = {"heldout_item_categories": ["block"], "heldout_colors": [], "heldout_regions": [],
              "heldout_phrase_templates": []}
    rules = enumerate_rules(items, regions, TOY_TASK_CFG, phrases_cfg=PHRASES_CFG, splits_cfg=splits,
                            rng=np.random.RandomState(0))
    by_id = {r.rule_id: r for r in rules}
    assert by_id["inside|block_0|tray_top"].split == "heldout_item"
    assert by_id["inside|bottle_0|tray_top"].split == "train"
    assert all(len(r.phrases) > 0 for r in rules)
    assert all(not any(c.isdigit() for c in p) for r in rules for p in r.phrases)


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
