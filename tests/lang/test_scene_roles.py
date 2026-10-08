"""scene_roles (VANILLA_PLAN 3.5): referent conversion, in_way per task,
designation, token_masks rows."""

import json
from types import SimpleNamespace as NS

import pytest

from main.lang import scene_roles as SR


def _obj(k, item_id, kind="rigid", size=None, role="candidate"):
    return {"id": k, "item_id": item_id, "kind": kind, "size": size, "role": role}


def _reg(k, region_id, nameable=True):
    return {"id": k, "region_id": region_id, "nameable": nameable}


def _rule(rid, cons, subject, reference=None, **kw):
    r = {"rule_id": rid, "consequence": cons, "subject": subject, "reference": reference,
         "set": False, "ambiguous": False, "sampleable": True}
    r.update(kw)
    return r


def _env(task_attrs=None, runner=None):
    return NS(task=NS(**(task_attrs or {})), _item_runner=runner)


def test_referent_conversion():
    rec = {"objects": [_obj(0, "payload", role="task_object"), _obj(1, "block_0"), _obj(2, "beads", kind="particle_set")],
           "regions": [_reg(0, "table_top"), _reg(1, "bin_0")]}
    assert SR.referent(_rule("u", "upright", {"object": 1}), rec) == "block_0"
    assert SR.referent(_rule("s", "contain_particles", {"particles": 2}, {"region": 1}), rec) == "beads"
    assert SR.referent(_rule("x", "exclude_particles", {"particles": 2}, {"region": 1}), rec) == "bin_0"
    assert SR.referent(_rule("z", "zone_include", {"object": 0}, {"region": 0}), rec) == "table_top"
    assert SR.referent(_rule("set", "avoid", {"objects": [1, 2]}, set=True), rec) is None


def test_designation_values():
    rec = {"objects": [_obj(0, "a"), _obj(1, "b"), _obj(2, "c")], "regions": [_reg(0, "table_top")]}
    roles = ({"a"}, {"b"})
    assert SR.designation(_rule("1", "avoid", {"object": 0}), rec, roles) == "harm"
    assert SR.designation(_rule("2", "stay", {"object": 1}), rec, roles) == "unclear"
    assert SR.designation(_rule("3", "upright", {"object": 2}), rec, roles) == "benign"
    assert SR.designation(_rule("4", "zone_include", {"object": 0}, {"region": 0}), rec, roles) == "benign"
    assert SR.designation(_rule("5", "avoid", {"object": 0}, ambiguous=True), rec, roles) is None
    assert SR.designation(_rule("6", "avoid", {"objects": [0, 1]}, set=True), rec, roles) is None


def test_in_way_t1_placements_and_second_corridor(monkeypatch):
    # off_path candidate within 10 cm of a SECOND corridor -> unclear
    placements = [
        {"item_id": "c0", "cls": "on_path", "xyz": [0.0, 0.0, 0.8]},
        {"item_id": "c1", "cls": "near_path", "xyz": [0.1, 0.0, 0.8]},
        {"item_id": "c2", "cls": "off_path", "xyz": [0.0, 0.30, 0.8]},
        {"item_id": "c3", "cls": "off_path", "xyz": [0.5, -0.4, 0.8]},
    ]
    import numpy as np

    import main.envs.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "iter_corridors", lambda task, spec: [
        (np.array([-0.2, 0.0]), np.array([0.2, 0.0]), "a", "t"),
        (np.array([-0.2, 0.25]), np.array([0.2, 0.25]), "b", "t"),
    ])
    monkeypatch.setitem(tasks_mod.TASK_SPECS, "place_burger_fries", tasks_mod.TASK_SPECS["place_burger_fries"])
    runner = NS(_spawn_report={"placements": placements}, host="place_burger_fries")
    inway, unclear = SR.in_way(_env(runner=runner), {"task": "selective_avoid", "objects": []})
    assert inway == {"c0"} and unclear == {"c1", "c2"}


def test_in_way_other_tasks():
    beads = _obj(9, "beads", kind="particle_set")
    rec2 = {"task": "granular_pour", "objects": [_obj(0, "mug", role="task_object"), beads]}
    assert SR.in_way(_env(), rec2) == ({"beads"}, set())
    elems = [{"id": "mat_0", "on_path": True}, {"id": "obj_1", "on_path": False}, {"id": "obj_2", "on_path": True}]
    assert SR.in_way(_env({"layout_info": {"elements": elems}}), {"task": "push_zone", "objects": []}) == ({"mat_0", "obj_2"}, set())
    rec5 = {"task": "tray_transport", "objects": [_obj(0, "tray", role="task_object"), _obj(1, "block_0", size="tall"),
                                                  _obj(2, "block_1", size="short")]}
    assert SR.in_way(_env(), rec5) == ({"block_0"}, set())
    rec6 = {"task": "clean_pour", "objects": [beads]}
    # R1: the diagonal slot is unclear, the other outer slot is benign even when jitter puts it < 90 deg away
    nbs = [{"region_id": "bin_0", "az_deg": 90.0, "slot": "far"}, {"region_id": "bin_1", "az_deg": 128.0, "slot": "far_across"},
           {"region_id": "plate_2", "az_deg": 171.0, "slot": "across"}]
    inway, unclear = SR.in_way(_env({"episode_info": {"hit": "bin_0", "neighbors": nbs}}), rec6)
    assert inway == {"bin_0", "beads"} and unclear == {"bin_1"}
    # records without slot names: the 67.5 deg azimuth cut (45 +- 20 vs 90 +- 20)
    nbs = [{"region_id": "bin_0", "az_deg": 0.0}, {"region_id": "bin_1", "az_deg": 45.0},
           {"region_id": "plate_2", "az_deg": 81.0}]
    inway, unclear = SR.in_way(_env({"episode_info": {"hit": "bin_0", "neighbors": nbs}}), rec6)
    assert inway == {"bin_0", "beads"} and unclear == {"bin_1"}
    assert SR.in_way(_env({"episode_info": {"hit": "block_1"}}), {"task": "gap_place", "objects": []}) == ({"block_1"}, set())
    with pytest.raises(KeyError):
        SR.in_way(_env(), {"task": "nope", "objects": []})


def test_mask_rows():
    runner = NS(items=[1, 2, 3], regions=[NS(region_id="table_top", nameable=True, entity=None),
                                          NS(region_id="mat_0", nameable=True, entity=object()),
                                          NS(region_id="bin_x", nameable=False, entity=object()),
                                          NS(region_id="bin_1", nameable=True, entity=object())])
    rows = SR.mask_region_rows(runner)
    assert rows == {"mat_0": 3, "bin_1": 4}
    rec = {"objects": [_obj(0, "a"), _obj(1, "b"), _obj(2, "beads", kind="particle_set")],
           "regions": [_reg(0, "table_top"), _reg(1, "mat_0"), _reg(2, "bin_x", False), _reg(3, "bin_1")]}
    assert SR.ref_rows(rec, _rule("s", "stay", {"object": 1}), rows) == [1]
    assert SR.ref_rows(rec, _rule("c", "contain_particles", {"particles": 2}, {"region": 3}), rows) == [2]
    assert SR.ref_rows(rec, _rule("x", "exclude_particles", {"particles": 2}, {"region": 3}), rows) == [4]
    assert SR.ref_rows(rec, _rule("z", "zone_exclude", {"object": 0}, {"region": 1}), rows) == [3]
    assert SR.ref_rows(rec, _rule("t", "zone_include", {"object": 0}, {"region": 0}), rows) == []  # no entity


def test_fixture_records(fixtures_dir):
    """One saved record per task (written by the Phase-0 recon): every
    non-set rule's referent resolves to an id of the record."""
    files = sorted((fixtures_dir / "records").glob("*.json"))
    if not files:
        pytest.skip("no fixture records")
    seen = set()
    for f in files:
        rec = json.loads(f.read_text())
        seen.add(rec["task"])
        ids = {o["item_id"] for o in rec["objects"]} | {r["region_id"] for r in rec["regions"]}
        for rule in rec["rules"]:
            ref = SR.referent(rule, rec)
            if rule.get("set"):
                assert ref is None
            else:
                assert ref in ids, (f.name, rule["rule_id"], ref)
                # the rule id names its referent (consequence|item|region)
                parts = rule["rule_id"].split("|")
                assert ref in parts, (f.name, rule["rule_id"], ref)
    assert len(seen) == len(files)
