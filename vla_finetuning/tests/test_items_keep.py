"""The item demos' keep rule (vla_finetuning/items/keep.py) on synthetic rule tables, traces and episode facts, no
simulator:

    python -m pytest -q vla_finetuning/tests/test_items_keep.py
"""

from __future__ import annotations

import numpy as np
import pytest

import main.lang  # noqa: F401 - main.item_envs on main.__path__
from main.item_envs.rules import RuleSpec
from vla_finetuning.items import keep as K

DEG = np.pi / 180.0


def _r(rid, consequence, items, regions=(), **params):
    return RuleSpec(rid, consequence, items, regions, params)


RULES = {
    "selective_avoid": [_r("avoid|obj_0|-", "avoid", ("obj_0",), margin=0.01),
                        _r("avoid|obj_1|-", "avoid", ("obj_1",), margin=0.01),
                        _r("avoid|obj_0+obj_1|-", "avoid", ("obj_0", "obj_1"), margin=0.01),
                        _r("stay|obj_0|-", "stay", ("obj_0",), d_tol=0.03, theta_tol=20 * DEG),
                        _r("stay|obj_1|-", "stay", ("obj_1",), d_tol=0.03, theta_tol=20 * DEG)],
    "push_zone": [_r("inside|push_0|table_top", "inside", ("push_0",), ("table_top",)),
                  _r("outside|push_0|zone_red", "outside", ("push_0",), ("zone_red",)),
                  _r("outside|push_0|zone_blue", "outside", ("push_0",), ("zone_blue",)),
                  _r("upright|cyl_0|-", "upright", ("cyl_0",), theta_max=45 * DEG),
                  _r("stay|cyl_0|-", "stay", ("cyl_0",), d_tol=0.015, theta_tol=45 * DEG),
                  _r("outside|push_0|zone_blue+zone_red", "outside", ("push_0",), ("zone_blue", "zone_red"))],
    "tray_transport": [_r("upright|block_0|-", "upright", ("block_0",), theta_max=45 * DEG),
                       _r("upright|block_1|-", "upright", ("block_1",), theta_max=45 * DEG)],
    "clean_pour": [_r("particles_outside|beads|bowl_blue", "particles_outside", ("beads",), ("bowl_blue",)),
                   _r("particles_outside|beads|bowl_green", "particles_outside", ("beads",), ("bowl_green",)),
                   _r("particles_inside|beads|bowl_src_interior+target_interior", "particles_inside", ("beads",),
                      ("bowl_src_interior", "target_interior"))],
    "gap_place": [_r("upright|block_0|-", "upright", ("block_0",), theta_max=45 * DEG),
                  _r("stay|block_0|-", "stay", ("block_0",), d_tol=0.02, theta_tol=45 * DEG)],
}


def test_protected_rules_per_task():
    p = K.protected_rules("selective_avoid", RULES["selective_avoid"])
    assert p == {"stay|obj_0|-": 0.0, "stay|obj_1|-": 0.0}                # the contact check replaces avoid
    p = K.protected_rules("push_zone", RULES["push_zone"])
    assert list(p) == [r.rule_id for r in RULES["push_zone"]]             # every element, the table, the pairs
    assert p["inside|push_0|table_top"] == 0.0 and p["upright|cyl_0|-"] == 0.0 and p["stay|cyl_0|-"] == 0.0
    assert p["outside|push_0|zone_red"] == p["outside|push_0|zone_blue+zone_red"] == K.OUTSIDE_H_MIN == 0.1
    assert K.protected_rules("tray_transport", RULES["tray_transport"]) == {"upright|block_0|-": 0.0,
                                                                            "upright|block_1|-": 0.0}
    p = K.protected_rules("clean_pour", RULES["clean_pour"])
    assert p == {"particles_outside|beads|bowl_blue": 0.0, "particles_outside|beads|bowl_green": 0.0}   # no spill rule
    assert K.protected_rules("gap_place", RULES["gap_place"]) == {"upright|block_0|-": 0.0, "stay|block_0|-": 0.0}
    # an outside rule outside T3 keeps the plain bar
    assert K.protected_rules("gap_place", [RULES["push_zone"][1]]) == {"outside|push_0|zone_red": 0.0}


def _trace(n, rules, safe=0.03):
    t = np.arange(n) * 10 / 250.0
    return t, np.full((n, len(rules)), safe), np.full((n, len(rules)), -1, np.int64)


def test_score_trace_bead_counts_and_order():
    rules = RULES["clean_pour"]
    prot = K.protected_rules("clean_pour", rules)
    t, m, c = _trace(10, rules, safe=0.02)
    c[:, :] = 0                                  # bead rules: counts per tick
    c[6:, 1] = 1                                 # one bead in the green bowl from capture 6: h = -0.5
    c[3:, 0] = 25                                # 25 beads in the blue bowl from capture 3: h = -1
    c[2:, 2] = 4                                 # beads on the table: the spill rule is not protected
    per, broken = K.score_trace(rules, prot, t, m, c)
    assert broken == ["particles_outside|beads|bowl_blue", "particles_outside|beads|bowl_green"]   # earliest first
    assert per["particles_outside|beads|bowl_green"]["first_t"] == pytest.approx(t[6])
    assert per["particles_outside|beads|bowl_green"]["first_tick"] == 6
    assert per["particles_outside|beads|bowl_green"]["min_h"] == pytest.approx(-0.5)
    assert per["particles_outside|beads|bowl_blue"]["min_h"] == pytest.approx(-1.0)
    spill = "particles_inside|beads|bowl_src_interior+target_interior"
    assert per[spill]["first_t"] == pytest.approx(t[2]) and spill not in broken     # reported, never required


def test_score_trace_outside_needs_5mm():
    rules = RULES["push_zone"]
    prot = K.protected_rules("push_zone", rules)
    t, m, c = _trace(8, rules)
    m[4, 1] = 0.004                              # 4 mm past the red mat: h = 0.08 < 0.1
    m[5, 2] = 0.006                              # 6 mm: h = 0.12, kept
    per, broken = K.score_trace(rules, prot, t, m, c)
    assert broken == ["outside|push_0|zone_red"]
    assert per["outside|push_0|zone_red"]["min_h"] == pytest.approx(0.08)
    assert per["outside|push_0|zone_blue"]["first_t"] is None
    m[3, 3] = 0.0 - 1e-6                         # the cylinder tips past 45 deg at capture 3: first
    _, broken = K.score_trace(rules, prot, t, m, c)
    assert broken == ["upright|cyl_0|-", "outside|push_0|zone_red"]


def test_score_trace_nan_is_broken_and_ties_keep_table_order():
    rules = RULES["gap_place"]
    prot = K.protected_rules("gap_place", rules)
    t, m, c = _trace(5, rules)
    m[2, :] = [np.nan, -0.001]
    per, broken = K.score_trace(rules, prot, t, m, c)
    assert broken == ["upright|block_0|-", "stay|block_0|-"]
    assert per["upright|block_0|-"]["min_h"] is not None                  # NaN left out of the min
    _, broken = K.score_trace(rules, prot, *_trace(5, rules))
    assert broken == []


def test_avoid_breaks_t1_only_by_contact():
    rules = RULES["selective_avoid"]
    prot = K.protected_rules("selective_avoid", rules)
    t, m, c = _trace(6, rules)
    m[3, 0] = -0.005                             # 5 mm from obj_0: inside the filter's 1 cm buffer, no touch
    per, broken = K.score_trace(rules, prot, t, m, c)
    assert broken == [] and per["avoid|obj_0|-"]["first_t"] == pytest.approx(t[3])


def test_read_trace(tmp_path):
    rules = RULES["gap_place"]
    t, m, c = _trace(4, rules)
    p = tmp_path / "trace.npz"
    np.savez_compressed(p, t=t, margins=m, violation_counts=c, rule_ids=np.array([r.rule_id for r in rules]))
    t2, m2, c2 = K.read_trace(p, rules)
    assert np.allclose(t2, t) and np.allclose(m2, m) and (c2 == c).all()
    np.savez_compressed(p, t=t, margins=m, rule_ids=np.array([r.rule_id for r in rules]))   # an old trace: no counts
    assert (K.read_trace(p, rules)[2] == -1).all()
    with pytest.raises(K.TraceMismatch):
        K.read_trace(p, rules[::-1])            # columns in another order: the code, not the seed (code_error)
    np.savez_compressed(p, t=t, margins=m)
    with pytest.raises(K.TraceMismatch):
        K.read_trace(p, rules[:1])              # no rule_ids: the column count


def test_render_errors():
    assert K.is_render_error("RuntimeError: cannot create buffer")
    assert K.is_render_error("RuntimeError: vk::Queue::submit: ErrorDeviceLost")
    assert K.is_render_error("RuntimeError: CUDA error: an illegal memory access was encountered")
    assert not K.is_render_error("AssertionError: target_pose cannot be None for move action.")
    assert not K.is_render_error(None)


GOOD = {"err": None, "safe_fail": None, "plan_success": True, "task_success": True, "broken": [], "contact": None,
        "n_frames": 120}


@pytest.mark.parametrize("change, reason", [
    ({}, "kept"),
    ({"err": "RuntimeError: cannot create buffer"}, "render_error"),
    ({"err": "ContactAbort: extra object touched at physics step 812"}, "contact"),
    ({"err": "ItemTimeout: rollout exceeded 7500 physics steps"}, "timeout"),
    ({"err": "AssertionError: target_pose cannot be None for move action."}, "error"),        # no grasp pose
    ({"err": "RuntimeError: push_zone: no valid B-A/mix layout after 1000 tries (seed 3)"}, "error"),
    ({"err": "AttributeError: 'NoneType' object has no attribute 'get_pose'"}, "code_error"),
    ({"err": "NotImplementedError: push_zone: no safe nominal mode yet"}, "code_error"),
    ({"err": "FileNotFoundError: RoboTwin missing: /x/RoboTwin"}, "code_error"),
    # a trace-scoring failure keeps its type first (collect_items): a bug there stops the process, as a code_error
    ({"err": "TraceMismatch: trace: trace has 3 rule columns, the runner 4 rules"}, "code_error"),
    ({"err": "KeyError: trace: 'margins'"}, "code_error"),
    ({"err": "trace: KeyError: 'margins'"}, "error"),                  # the old prefix hid the type
    ({"safe_fail": "no_route", "plan_success": False, "task_success": False}, "safe_fail:no_route"),
    ({"plan_success": False}, "plan_failed"),                         # the task's flag at the end of the episode
    ({"plan_success": False, "task_success": False}, "plan_failed"),
    ({"plan_failures": 2}, "kept"),          # failed plans the task recovered from (T5 / T7 step chain, T6 pour)
    ({"task_success": False}, "task_failed"),
    ({"broken": ["stay|block_0|-", "upright|block_0|-"]}, "rule:stay|block_0|-"),
    ({"contact": {"touched": True, "force_max": 3.0}}, "contact"),
    ({"contact": {"touched": False, "force_max": 25.0}}, "force"),
    ({"contact": {"touched": False, "force_max": 19.9}}, "kept"),
    ({"n_frames": 9}, "few_frames"),
    # the order: an error beats everything, a safe_fail beats the plan failure it causes, the rule beats contact
    ({"err": "ContactAbort: x", "task_success": False, "broken": ["stay|obj_0|-"]}, "contact"),
    ({"broken": ["stay|obj_0|-"], "contact": {"touched": True, "force_max": 50.0}}, "rule:stay|obj_0|-"),
    ({"task_success": False, "broken": ["stay|obj_0|-"]}, "task_failed"),
])
def test_keep_reason(change, reason):
    assert K.keep_reason({**GOOD, **change}) == reason


def test_planner_error_from_the_t1_grasp_search_is_retried():
    from vla_finetuning.items.collect_items import RETRY_REASONS

    r = K.keep_reason({"err": None, "safe_fail": "planner_error", "plan_success": False, "task_success": False})
    assert r == "planner_error" and r in RETRY_REASONS

