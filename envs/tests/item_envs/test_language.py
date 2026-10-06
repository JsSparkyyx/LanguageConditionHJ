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


def test_heldout_words():
    s = _scene([("cup", ["glass", "mug"], "red", None)])
    train = L.describe(s["objects"][0], s, "train")
    held = L.describe(s["objects"][0], s, "heldout")
    assert not any("glass" in p for p in train) and any("mug" in p for p in train)
    assert held and all("glass" in p for p in held)  # "glass" is a held-out word


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
    s = _scene([("beads", ["rice", "grains"], None, None, "particle_set"), ("mug", [], None, None)],
               [("bin", ["bin", "box"], "green"), ("bowl_src_interior", ["source bowl", "bowl"], None)])
    spill = _rule("contain_particles", {"particles": 0}, {"regions": [1]})
    train = L.sentences(spill, s, "train")
    assert "don't spill the rice" in train and not any("grains" in x for x in train)
    held_t = L.sentences(spill, s, "heldout_template")
    assert held_t and not set(held_t) & set(train)
    assert all("grains" in x for x in L.sentences(spill, s, "heldout_word"))
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
