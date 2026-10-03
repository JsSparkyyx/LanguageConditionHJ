"""Held-item selection among several closed-gripper contacts (no simulator).

Regression for diag job 22636048 (selective_avoid / place_bread_basket /
franka seed 1): the gripper pinched bread0 while a can brushed the outside of
one finger; "first contact wins" latched the can.
"""

from types import SimpleNamespace

from main.item_envs import robot_distance as rd


def _ent(psid):
    return SimpleNamespace(per_scene_id=psid)


def _contact(a, b):
    return SimpleNamespace(bodies=[SimpleNamespace(entity=_ent(a)), SimpleNamespace(entity=_ent(b))])


FINGER_L, FINGER_R = 101, 102  # left-arm gripper links
CAN, BREAD = 7, 9


def _setup(monkeypatch, contacts, tcp_dist):
    monkeypatch.setattr(rd, "_robot_side_psids", lambda env: {FINGER_L: "left", FINGER_R: "left"})
    env = SimpleNamespace(task=SimpleNamespace(scene=SimpleNamespace(get_contacts=lambda: contacts)))
    cehj = SimpleNamespace(
        _is_gripper_entity=lambda env, e: e.per_scene_id in (FINGER_L, FINGER_R),
        _arm_is_holding=lambda env, side: side == "left",
        _hold_metrics=lambda env, entity, side: {"origin_dist_tcp": tcp_dist[entity]},
    )
    return env, cehj


def test_pinched_item_beats_earlier_single_finger_contact(monkeypatch):
    # the can is listed FIRST and touches one finger; the bread touches both
    contacts = [_contact(FINGER_L, CAN), _contact(BREAD, FINGER_L), _contact(FINGER_R, BREAD)]
    env, cehj = _setup(monkeypatch, contacts, {"can": 0.02, "bread": 0.05})
    got = rd._closed_gripper_contacts(env, {CAN: "can", BREAD: "bread"}, {"can": "can", "bread": "bread"}, cehj)
    assert got == {"left": "bread"}


def test_tie_breaks_by_tcp_distance(monkeypatch):
    contacts = [_contact(FINGER_L, CAN), _contact(FINGER_R, BREAD)]
    env, cehj = _setup(monkeypatch, contacts, {"can": 0.08, "bread": 0.01})
    got = rd._closed_gripper_contacts(env, {CAN: "can", BREAD: "bread"}, {"can": "can", "bread": "bread"}, cehj)
    assert got == {"left": "bread"}


def test_open_gripper_holds_nothing(monkeypatch):
    contacts = [_contact(FINGER_L, BREAD), _contact(FINGER_R, BREAD)]
    env, cehj = _setup(monkeypatch, contacts, {"bread": 0.0})
    cehj._arm_is_holding = lambda env, side: False
    assert rd._closed_gripper_contacts(env, {BREAD: "bread"}, {"bread": "bread"}, cehj) == {}
