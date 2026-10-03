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


def test_single_link_contacts_never_start_a_hold(monkeypatch):
    # an item resting against the OUTSIDE of one finger is not held — this
    # latched the basket the carried bread touched (diag 22636122 seed 3)
    contacts = [_contact(FINGER_L, CAN), _contact(FINGER_R, BREAD)]
    env, cehj = _setup(monkeypatch, contacts, {"can": 0.08, "bread": 0.01})
    got = rd._closed_gripper_contacts(env, {CAN: "can", BREAD: "bread"}, {"can": "can", "bread": "bread"}, cehj)
    assert got == {}


def test_two_pinched_items_tie_break_by_tcp_distance(monkeypatch):
    contacts = [_contact(FINGER_L, CAN), _contact(FINGER_R, CAN),
                _contact(FINGER_L, BREAD), _contact(FINGER_R, BREAD)]
    env, cehj = _setup(monkeypatch, contacts, {"can": 0.08, "bread": 0.01})
    got = rd._closed_gripper_contacts(env, {CAN: "can", BREAD: "bread"}, {"can": "can", "bread": "bread"}, cehj)
    assert got == {"left": "bread"}


def test_open_gripper_holds_nothing(monkeypatch):
    contacts = [_contact(FINGER_L, BREAD), _contact(FINGER_R, BREAD)]
    env, cehj = _setup(monkeypatch, contacts, {"bread": 0.0})
    cehj._arm_is_holding = lambda env, side: False
    assert rd._closed_gripper_contacts(env, {BREAD: "bread"}, {"bread": "bread"}, cehj) == {}


def test_tall_cylinder_stands_upright():
    """SAPIEN cylinders run along local x: the spawn quat must put that axis on
    world z, and the registered OBB / up axis must live in the same frame."""
    import numpy as np

    from main.item_envs import spawn

    draw = spawn._ObjectDraw(asset="prim_tall_cylinder", category="cylinder", color="red",
                             size=np.array([0.06, 0.06, 0.14]))
    geo = spawn._geometry_of(draw)
    R = spawn._quat_wxyz_to_mat(np.asarray(geo.quat, dtype=np.float64))
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-9)  # axis vertical
    assert np.allclose(R @ geo.up_local, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(np.abs(R) @ geo.half, geo.world_half, atol=1e-9)
    assert np.allclose(geo.world_half, [0.03, 0.03, 0.07])


def test_joint_unwrap_takes_the_short_way():
    import numpy as np
    from types import SimpleNamespace
    from main.item_envs.controller import RiskTickController

    ctrl = RiskTickController.__new__(RiskTickController)
    lim = np.array([[-2 * np.pi, 2 * np.pi]] * 6)
    ctrl.robot = SimpleNamespace(left_entity=SimpleNamespace(get_qlimits=lambda: lim))
    n = 50
    pos = np.zeros((n, 6))
    pos[:, 5] = np.linspace(0.1, 0.1 - np.radians(350), n)  # -350 deg spin
    pos[:, 2] = np.linspace(0.0, 0.3, n)
    out = ctrl._unwrap_revolutions("left", {"status": "Success", "position": pos})
    assert np.isclose(out["position"][-1, 5] - out["position"][0, 5], np.radians(10))
    assert np.allclose(out["position"][:, 2], pos[:, 2])  # other joints untouched
    # a genuine > pi move that cannot wrap inside the limits stays as planned
    lim2 = lim.copy(); lim2[5] = [-0.2, 0.2 + np.radians(350)]
    ctrl.robot = SimpleNamespace(left_entity=SimpleNamespace(get_qlimits=lambda: lim2))
    pos2 = pos.copy(); pos2[:, 5] = np.linspace(-0.1, -0.1 + np.radians(200), n)
    out2 = ctrl._unwrap_revolutions("left", {"status": "Success", "position": pos2})
    assert out2["position"] is pos2 or np.allclose(out2["position"], pos2)
