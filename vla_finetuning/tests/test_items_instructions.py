"""Plain task instructions of the item demos (vla_finetuning/items/instructions.py) on fake task objects, no simulator.
RoboTwin's description/ tree is read for the generator helpers and the T1 host templates:

    python -m pytest -q vla_finetuning/tests/test_items_instructions.py
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace as NS

import pytest

from vla_finetuning.items import instructions as I

INFO_KEYS = {"push_zone": {"{A}"}, "tray_transport": {"{A}"}, "clean_pour": {"{A}", "{B}"},
             "gap_place": {"{A}", "{B}", "{C}"}}


def _gap(colors=("red", "blue"), bystander="green"):
    blocks = [{"item_id": f"block_{k}", "color": c, "role_in_slot": "flanker"} for k, c in enumerate(colors)]
    if bystander:
        blocks.append({"item_id": "block_2", "color": bystander, "role_in_slot": "bystander"})
    return NS(standing=blocks)


FAKE = {"push_zone": NS(push_objects=[{"item_id": "push_0", "color": "green"}]),
        "tray_transport": NS(),
        "clean_pour": NS(target_color="white", src_kind="mug"),
        "gap_place": _gap()}


def test_placeholders_from_the_scene():
    assert I.item_instruction_info("push_zone", FAKE["push_zone"]) == {"{A}": "the green box"}
    assert I.item_instruction_info("tray_transport", FAKE["tray_transport"]) == {"{A}": "the tray"}
    assert I.item_instruction_info("clean_pour", FAKE["clean_pour"]) == {"{A}": "the white bowl", "{B}": "the mug"}
    seen = set()
    for seed in range(20):
        info = I.item_instruction_info("gap_place", FAKE["gap_place"], seed=seed)
        assert info == I.item_instruction_info("gap_place", FAKE["gap_place"], seed=seed)   # the seed decides
        assert info["{A}"] == "the wooden block"
        assert {info["{B}"], info["{C}"]} == {"the red block", "the blue block"}    # flankers only, never the bystander
        seen.add(info["{B}"])
    assert seen == {"the red block", "the blue block"}                              # both orders occur


def test_placeholder_guards():
    with pytest.raises(ValueError):
        I.item_instruction_info("push_zone", NS(push_objects=[{"color": None}]))        # textured box: no colour word
    with pytest.raises(ValueError):
        I.item_instruction_info("clean_pour", NS(target_color="white", src_kind="bowl"))   # the templates say the mug
    with pytest.raises(ValueError):
        I.item_instruction_info("gap_place", _gap(("red", "red")))
    with pytest.raises(ValueError):
        I.item_instruction_info("gap_place", _gap(("red",), bystander=None))


def test_t1_uses_the_host_placeholders():
    assert I.item_instruction_info("selective_avoid", NS(), host="stack_blocks_two") == \
        {"{A}": "red block", "{B}": "green block"}
    host = NS(bread_id=[2], bread=[object()], basket_id=1)
    assert I.item_instruction_info("selective_avoid", host, host="place_bread_basket") == \
        {"{A}": "076_breadbasket/base1", "{B}": "075_bread/base2"}


@pytest.mark.parametrize("task", I.TEMPLATE_TASKS)
def test_template_files(task):
    data = json.loads(I.template_path(task).read_text())
    assert {"full_description", "schema", "preference", "seen", "unseen"} <= set(data)
    for key in I.KEYS:
        assert data[key], key
        for tpl in data[key]:
            assert {"{" + p + "}" for p in re.findall(r"{([^}]+)}", tpl)} == INFO_KEYS[task], tpl
            assert I.has_heldout_word(tpl) is None, tpl
            assert len(tpl.split()) <= 20, tpl
            assert tpl[0].isupper() and tpl.endswith("."), tpl
    assert not set(data["seen"]) & set(data["unseen"])


@pytest.mark.parametrize("task", I.TEMPLATE_TASKS)
def test_make_item_instructions(task):
    info = I.item_instruction_info(task, FAKE[task], seed=3)
    out = I.make_item_instructions(task, info, seed=11)
    assert set(out) == {"seen", "unseen"}
    assert len(out["seen"]) == 100 and len(out["unseen"]) == 100
    for s in out["seen"] + out["unseen"]:
        assert "{" not in s and s.strip()
        for v in info.values():                     # every string names every placeholder value
            assert v in s, (v, s)
    assert out == I.make_item_instructions(task, info, seed=11)           # reproducible from the demo seed
    templates = I.load_templates(task)
    assert len(set(out["seen"])) == len(templates["seen"])                # every template used, cycled to 100
    assert len(set(out["unseen"])) == len(templates["unseen"])


def test_seed_shuffles_the_order_only():
    info = I.item_instruction_info("push_zone", FAKE["push_zone"])
    a, b = (I.make_item_instructions("push_zone", info, seed=s) for s in (1, 2))
    assert a["seen"] != b["seen"] and set(a["seen"]) == set(b["seen"])


def test_a_path_value_is_refused():
    with pytest.raises(ValueError):
        I.make_item_instructions("tray_transport", {"{A}": "008_tray/base1"}, seed=0)


def test_heldout_words():
    assert set(I.heldout_words()) >= {"container", "serving tray", "cube", "little balls"}
    assert I.has_heldout_word("Put the bowl in the white container.") == "container"
    assert I.has_heldout_word("two containers") == "container"
    assert I.has_heldout_word("the rectangular bright orange Serving Tray") == "serving tray"
    assert I.has_heldout_word("Pour the beads into the white bowl.") is None
    assert I.has_heldout_word("the cubed apple") is None                     # whole words only


def test_t1_host_templates_without_heldout_words():
    """place_burger_fries' tray and fries descriptions say "serving tray" / "container": those strings are dropped
    and the lists topped up from a larger draw."""
    info = {"{A}": "006_hamburg/base0", "{B}": "008_tray/base1", "{C}": "005_french-fries/base0"}
    out = I.make_item_instructions("selective_avoid", info, seed=5, host="place_burger_fries")
    assert len(out["seen"]) == 100 and len(out["unseen"]) == 100
    assert not any(I.has_heldout_word(s) for s in out["seen"] + out["unseen"])
    out2 = I.make_item_instructions("selective_avoid", {"{A}": "red block", "{B}": "green block"}, seed=5,
                                    host="stack_blocks_two")
    assert all("red block" in s and "green block" in s for s in out2["seen"])
    with pytest.raises(ValueError):
        I.make_item_instructions("selective_avoid", {"{A}": "red block"}, seed=5)       # no host


def test_check_instructions():
    good = {"seen": ["Push the red box into the black frame."], "unseen": ["Shove the red box into the black square."]}
    I.check_instructions(good)
    for bad in ({"seen": good["seen"]},                                             # a key missing
                {"seen": good["seen"], "unseen": []},                               # an empty list
                {"seen": ["  "], "unseen": good["unseen"]},                         # an empty string
                {"seen": ["Push {A} into the black frame."], "unseen": good["unseen"]},   # unfilled
                {"seen": ["Pour the beads into the white container."], "unseen": good["unseen"]}):   # held-out word
        with pytest.raises(ValueError):
            I.check_instructions(bad)


def test_ambiguous_t1_objects():
    box = lambda c, i="obj_0": NS(item_id=i, role="candidate", category="box", color=c)   # noqa: E731
    assert I.ambiguous_objects("stack_blocks_two", [box("red"), box("blue", "obj_1")]) == ["obj_0"]
    assert I.ambiguous_objects("stack_blocks_two", [box("Green")]) == ["obj_0"]
    assert I.ambiguous_objects("stack_blocks_two", [box("blue"), NS(item_id="obj_2", role="candidate",
                                                                    category="cylinder", color="red")]) == []
    assert I.ambiguous_objects("place_container_plate", [box("red")]) == []


def test_instruction_md5_follows_the_template_file():
    assert I.instruction_md5("gap_place") == I.instruction_md5("gap_place")
    assert I.instruction_md5("gap_place") != I.instruction_md5("push_zone")
    assert I.instruction_md5("selective_avoid", "stack_blocks_two") != I.instruction_md5("selective_avoid",
                                                                                          "stack_bowls_two")
