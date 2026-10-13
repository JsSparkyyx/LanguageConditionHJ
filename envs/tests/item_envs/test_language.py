"""Non-sim tests for the relabeling language side (main/item_envs/language).

The paraphrase bank (size, template split, no digits, placeholders), the
unique-in-scene referent phrases (describe / describe_region), the sentence
splits, and the episode record built from a saved meta.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml

from main.item_envs import language as L
from main.item_envs.rules import SET_TEMPLATES, TEMPLATES

PLACEHOLDERS = {
    "avoid": {"obj"}, "upright": {"obj"}, "stay": {"obj"},
    "zone_exclude": {"obj", "region"}, "zone_include": {"obj", "region"},
    "contain_particles": {"material"}, "exclude_particles": {"material", "region"},
}


def _keys(t: str) -> set:
    return set(re.findall(r"{(\w+)}", t))


def test_bank_size_split_and_placeholders():
    bank = L.load_bank()
    assert set(L.SPEC_CONSEQUENCE.values()) == set(PLACEHOLDERS)
    for cons, allowed in PLACEHOLDERS.items():
        train, held = bank[cons]["train"], bank[cons]["heldout"]
        assert len(train) + len(held) >= 30, cons  # "about 30 templates per consequence"
        assert len(held) >= 5 and not set(train) & set(held), cons
        for t in train + held:
            assert not re.search(r"\d", t), t
            assert _keys(t) and _keys(t) <= allowed, (cons, t)


def _to_bank(template: str, cons: str) -> str:
    """rules.py's {item}/{region} wording -> the bank's ({obj} / {material}
    carry their article from describe())."""
    t = template.replace("the {region}", "{region}")
    if cons in ("contain_particles", "exclude_particles"):
        return t.replace("{item}", "{material}")
    return t.replace("the {item}", "{obj}")


def test_rules_templates_and_heldout_split_are_in_the_bank():
    """Every rules.py template is in the bank, and splits.yaml's held-out
    phrase templates are held out here too (one split, not two)."""
    bank = L.load_bank()
    for cons, templates in TEMPLATES.items():
        name = L.SPEC_CONSEQUENCE[cons.value]
        for t in templates:
            assert _to_bank(t, name) in bank[name]["train"] + bank[name]["heldout"], t
    splits = yaml.safe_load(open("main/item_envs/configs/splits.yaml"))
    held = splits["all"]["heldout_phrase_templates"]
    names = {t: next(L.SPEC_CONSEQUENCE[c.value] for c, ts in TEMPLATES.items() if t in ts) for t in held}
    for t, name in names.items():
        assert _to_bank(t, name) in bank[name]["heldout"], t
    assert SET_TEMPLATES  # set rules exist, but have no bank entry (never sampled)


def _scene(objects, regions=()):
    objs = [dict(id=i, item_id=f"o{i}", category=o[0], synonyms=list(o[1]), color=o[2], size=o[3],
                 role="candidate", kind=o[4] if len(o) > 4 else "rigid") for i, o in enumerate(objects)]
    regs = [dict(id=i, region_id=f"r{i}", kind=r[0], names=list(r[1]), color=r[2], role=None, nameable=True)
            for i, r in enumerate(regions)]
    return {"objects": objs, "regions": regs, "rules": []}


def test_describe_only_unique_phrases():
    s = _scene([("cup", ["glass"], "red", None), ("cup", ["glass"], None, None), ("bowl", [], None, None)])
    red, plain, bowl = s["objects"]
    assert "the red cup" in L.describe(red, s) and "the cup" not in L.describe(red, s)
    assert L.describe(plain, s) == []  # "the cup" fits the red cup too: no unique phrase
    assert L.describe(bowl, s) == ["the bowl"]
    # size tells two red blocks apart; "the red block" does not
    s2 = _scene([("block", ["brick"], "red", "tall"), ("block", ["brick"], "red", "short")])
    tall = L.describe(s2["objects"][0], s2)
    assert "the tall block" in tall and "the tall red block" in tall
    assert "the red block" not in tall and "the block" not in tall


def test_bare_head_noun_is_not_unique_next_to_a_compound():
    """D1 (2026-10-09): "the box" fits a coffee box and "the bottle" a pill
    bottle, so a bare head noun is not unique next to a compound with that
    head; the compound still names only its own object."""
    s = _scene([("box", ["crate"], None, None), ("coffee box", ["coffee carton", "coffee bag"], None, None),
                ("bottle", ["water bottle"], None, None), ("pill bottle", ["pill jar", "small bottle"], None, None)])
    box, coffee, bottle, pill = s["objects"]
    # "the box" is not unique next to the coffee box; the box's other word still names it ("crate" is no longer a
    # held-out word: names check of 2026-10-12)
    assert L.describe(box, s) == ["the crate"] and L.describe(box, s, "heldout") == []
    assert "the coffee box" in L.describe(coffee, s)
    assert "the bottle" not in L.describe(bottle, s) and "the water bottle" in L.describe(bottle, s)
    assert "the pill bottle" in L.describe(pill, s)
    s2 = _scene([("box", [], "red", None), ("coffee box", [], "blue", None)])
    assert L.describe(s2["objects"][0], s2) == ["the red box"]
    # the region path checks objects through the same match
    s3 = _scene([("coffee box", [], None, None)], [("bin", ["bin", "box"], "green")])
    green = L.describe_region(s3["regions"][0], s3)
    assert "the bin" in green and "the box" not in green


def test_heldout_words():
    s = _scene([("cup", ["mug", "beaker"], "red", None)])
    train = L.describe(s["objects"][0], s, "train")
    held = L.describe(s["objects"][0], s, "heldout")
    assert not any("mug" in p for p in train) and any("beaker" in p for p in train)
    assert held and all("mug" in p for p in held)  # "mug" is a held-out word (names check, 2026-10-12)


def test_describe_region_unique_and_not_an_object_name():
    s = _scene([("box", [], "yellow", None)],
               [("zone", ["zone", "mat"], "red"), ("zone", ["zone", "mat"], "blue"), ("bin", ["bin", "box"], "green")])
    red = L.describe_region(s["regions"][0], s)
    assert "the red zone" in red and "the red mat" in red and "the zone" not in red
    green = L.describe_region(s["regions"][2], s)
    assert "the bin" in green and "the green box" in green
    assert "the box" not in green  # the yellow box (an object) answers to it


def _rule(cons, subject, reference=None, **kw):
    return dict(idx=0, rule_id="r", consequence=cons, subject=subject, reference=reference, params={},
                split="train", ambiguous=False, **kw)


def test_sentence_splits_and_particles():
    bank = L.load_bank()
    s = _scene([("beads", ["balls", "little balls"], None, None, "particle_set"), ("mug", [], None, None)],
               [("bin", ["bin", "box"], "green"), ("bowl_src_interior", ["source bowl", "bowl"], None)])
    spill = _rule("contain_particles", {"particles": 0}, {"regions": [1]})
    train = L.sentences(spill, s, "train")
    assert "don't spill the balls" in train and not any("little balls" in x for x in train)
    held_t = L.sentences(spill, s, "heldout_template")
    assert held_t and not set(held_t) & set(train)
    assert all("little balls" in x for x in L.sentences(spill, s, "heldout_word"))
    keep = _rule("exclude_particles", {"particles": 0}, {"region": 0})
    out = L.sentences(keep, s, "train")
    assert "keep the beads out of the green bin" in out and "keep the green box clean" in out
    assert L.sentences(_rule("avoid", {"objects": [0, 1]}, set=True), s) == []  # set rules: none
    rng = np.random.default_rng(0)
    assert L.sample_sentence(keep, s, rng) in out
    for x in train + out:
        assert not re.search(r"\d", x) and "{" not in x


def test_record_from_meta_columns_and_params():
    meta = {
        "task": "push_zone", "embodiment": "ur5-wsg", "seed": 3, "active_rule_id": "stay|obstacle_0|-",
        "items": [{"item_id": "push_0", "category": "box", "color": "yellow", "role": "candidate", "kind": "rigid"},
                  {"item_id": "obstacle_0", "category": "cylinder", "color": "green", "role": "bystander", "kind": "rigid"}],
        "regions": [{"region_id": "table_top", "color": None}, {"region_id": "zone_red", "color": "red"}],
        "rules": [
            {"rule_id": "outside|push_0|zone_red", "consequence": "outside", "item_ids": ["push_0"],
             "region_ids": ["zone_red"], "params": {}, "phrases": ["keep the box out of the red zone"], "split": "train"},
            {"rule_id": "stay|obstacle_0|-", "consequence": "stay", "item_ids": ["obstacle_0"], "region_ids": [],
             "params": {"d_tol": 0.015, "theta_tol": float(np.deg2rad(10))}, "phrases": ["don't disturb the cylinder"],
             "split": "train"},
        ],
    }
    rec = L.record_from_meta(meta, 12)
    assert rec["episode_id"] == 12 and rec["active_rule"] == 1
    assert rec["active_instruction"] == "don't disturb the cylinder"
    z, st = rec["rules"]
    assert z["consequence"] == "zone_exclude" and z["subject"] == {"object": 0} and z["reference"] == {"region": 1}
    assert st["params"]["disp_max_m"] == 0.015 and st["params"]["tilt_max_deg"] == pytest.approx(10.0)
    assert st["params"]["h_unit"] == 0.015 and z["params"]["h_unit"] == 0.05
    assert z["sampleable"] and st["sampleable"]
    assert "keep the yellow box out of the red zone" in L.sentences(z, rec)


def test_rule_phrases_skip_heldout_words():
    """rules.render_phrases (the on-screen instruction / active_instruction)
    never uses the bank's held-out referent words when another name exists."""
    from main.item_envs import physics
    from main.item_envs.rules import Consequence, RuleSpec, render_phrases

    cfg = physics.load_objects_cfg()
    held = set(L.load_bank()["heldout_words"])
    for cat, iid in (("bottle", "obj_0"), ("can", "obj_1"), ("cup", "obj_2")):
        rule = RuleSpec(rule_id=f"avoid|{iid}|-", consequence=Consequence.AVOID, item_ids=(iid,),
                        region_ids=(), params={"margin": 0.01, "category": cat})
        out = render_phrases(rule, cfg, np.random.RandomState(0), n=12)
        assert out and not any(w in p for p in out for w in held), out


def test_per_model_names_and_old_records():
    """Names checked against renders (2026-10-12): per-model words (objects.yaml assets), the conservative words of
    a pool object recorded without its model id, and refresh_record re-naming a stored record."""
    cfg = L._objects_cfg()
    assert L.model_words(cfg, "001_bottle", 19)[1] == ["bottle", "water bottle"]
    assert "water bottle" not in L.model_words(cfg, "001_bottle", 2)[1]
    assert L.model_words(cfg, "068_boxdrink", 2)[0] == "bottle"            # ids 1-3 are bottles
    assert L.model_words(cfg, "068_boxdrink", 0)[2] is False                # id 0 (a plastic cup) is never named
    assert "cube" in L.model_words(cfg, "086_woodenblock", 4)[1] and "cube" not in L.model_words(cfg, "086_woodenblock", 3)[1]
    assert L.unknown_model_words(cfg, "bottle") == ("bottle", ["bottle"], True)   # the word every bottle model shares
    assert L.unknown_model_words(cfg, "drink_box")[2] is False                   # could be the plastic cup
    assert L.unknown_model_words(cfg, "wooden_block")[2] is False                # could be the toy house
    # a stored T1 record: the drink box gets its model id from the recovered map
    rec = {"task": "selective_avoid", "objects": [
        {"id": 0, "item_id": "obj_0", "category_key": "drink_box", "category": "drink box", "synonyms": ["carton"],
         "color": None, "size": None, "role": "candidate", "kind": "rigid"},
        {"id": 1, "item_id": "obj_1", "category_key": "can", "category": "can", "synonyms": ["tin"],
         "color": None, "size": None, "role": "candidate", "kind": "rigid"}],
        "regions": [], "rules": [
        {"idx": 0, "rule_id": "avoid|obj_0|-", "consequence": "avoid", "subject": {"object": 0}, "reference": None,
         "params": {}, "set": False, "split": "train", "ambiguous": False, "sampleable": True},
        {"idx": 1, "rule_id": "avoid|obj_1|-", "consequence": "avoid", "subject": {"object": 1}, "reference": None,
         "params": {}, "set": False, "split": "train", "ambiguous": False, "sampleable": True}]}
    import copy
    old = L.refresh_record(copy.deepcopy(rec))                                   # no id: the drink box is not named
    assert old["rules"][0]["sampleable"] is False and old["objects"][1]["synonyms"] == ["soda can", "beverage can"]
    new = L.refresh_record(copy.deepcopy(rec), {"obj_0": ("068_boxdrink", 1)})
    assert new["objects"][0]["category_key"] == "bottle" and new["rules"][0]["sampleable"] is True
    assert "the bottle" in L.describe(new["objects"][0], new) and "the tin" not in L.describe(new["objects"][1], new)
    assert L.refresh_record(copy.deepcopy(rec), {"obj_0": ("068_boxdrink", 0)})["rules"][0]["sampleable"] is False


def test_load_model_ids(tmp_path):
    (tmp_path / "item_buffer").mkdir()
    (tmp_path / "item_buffer" / "model_ids.json").write_text(
        '{"episodes": {"7": {"obj_0": {"asset": "001_bottle", "model_id": 19}, '
        '"obj_1": {"asset": "prim_box", "model_id": null}}}, "mismatch": []}')   # primitives have no model id
    assert L.load_model_ids(tmp_path) == {7: {"obj_0": ("001_bottle", 19), "obj_1": ("prim_box", None)}}
    assert L.load_model_ids(tmp_path / "item_buffer") == L.load_model_ids(tmp_path)
    assert L.load_model_ids(tmp_path / "nowhere") == {}
