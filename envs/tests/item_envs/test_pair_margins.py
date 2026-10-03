"""Non-sim margins for pair rules (T3 rev. 3 A-A scenarios): outside over a
zone pair = min over the zones; stay over an obstacle pair keeps one
reference pose PER member (the old single-ref keying reused member 0's)."""

from __future__ import annotations

import numpy as np

from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import Consequence, RuleSpec
from main.item_envs.runner import one_margin, stay_ref_key


class _Pose:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def to_transformation_matrix(self):
        T = np.eye(4)
        T[:3, 3] = self.p
        return T


class _Ent:
    def __init__(self, p):
        self.pose = _Pose(p)

    def get_pose(self):
        return self.pose


def _item(iid, p, role="candidate"):
    return ItemSpec(item_id=iid, entity=_Ent(p), category="box", color="red", role=role,
                    half_extents=[0.02, 0.02, 0.02], obb_center_local=[0.0, 0.0, 0.0], up_axis_local=[0.0, 0.0, 1.0])


def _zone(rid, cx):
    T = np.eye(4)
    T[:3, 3] = [cx, 0.0, 0.74]
    sq = np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    return Region(region_id=rid, kind="zone", frame_fn=lambda T=T: T, polygon=sq, z=0.0, z_band=0.01,
                  nameable=True, color=rid.split("_")[1])


def test_outside_pair_is_min_over_zones():
    box = _item("push_0", [0.30, 0.0, 0.76])
    red, blue = _zone("zone_red", 0.0), _zone("zone_blue", 0.40)
    single = RuleSpec("outside|push_0|zone_red", Consequence.OUTSIDE, ("push_0",), ("zone_red",), {})
    pair = RuleSpec("outside|push_0|zone_blue+zone_red", Consequence.OUTSIDE, ("push_0",), ("zone_blue", "zone_red"), {})
    m_red = one_margin(single, [(0, box)], [red], None, {})
    m_blue = one_margin(RuleSpec("outside|push_0|zone_blue", Consequence.OUTSIDE, ("push_0",), ("zone_blue",), {}),
                        [(0, box)], [blue], None, {})
    m_pair = one_margin(pair, [(0, box)], [blue, red], None, {})
    assert m_red > 0.0 and m_blue > 0.0
    assert m_pair == min(m_red, m_blue)
    box.entity.pose.p[0] = 0.40  # into the blue zone
    assert one_margin(pair, [(0, box)], [blue, red], None, {}) < 0.0


def test_stay_pair_keeps_per_member_refs():
    a = _item("obstacle_0", [0.0, 0.0, 0.8], role="bystander")
    b = _item("obstacle_1", [0.3, 0.0, 0.8], role="bystander")
    rule = RuleSpec("stay|obstacle_0+obstacle_1|-", Consequence.STAY, ("obstacle_0", "obstacle_1"), (),
                    {"d_tol": 0.03, "theta_tol": np.deg2rad(20)})
    stay_ref = {}
    m0 = one_margin(rule, [(0, a), (1, b)], [], None, stay_ref)
    assert m0 > 0.0  # both at their own reference (member 1 is NOT compared to member 0's pose)
    assert set(stay_ref) == {stay_ref_key(rule.rule_id, "obstacle_0"), stay_ref_key(rule.rule_id, "obstacle_1")}
    b.entity.pose.p[0] += 0.05  # knock member 1 by 5 cm
    assert one_margin(rule, [(0, a), (1, b)], [], None, stay_ref) < 0.0
