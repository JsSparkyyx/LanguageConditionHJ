"""Non-sim tests for ItemMarginHFn (Phase 5) with fabricated fixtures.

The sim seam (_compute_tick / _per_link) is stubbed on the instance; the
pure logic under test is real: active-rule selection, the per-rule h
(scale * rule_h(rule, margin)), diag assembly (all margins + item state +
compute_h compat keys), per_link target selection for T1 single-item avoid
rules, and the sidecar row offered to the lockstep sink.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from main.item_envs.margin_hook import ItemMarginHFn, _TickData
from main.item_envs.items import ItemSpec
from main.item_envs.rules import Consequence, RuleSpec


def make_item(item_id, category="cup"):
    return ItemSpec(item_id=item_id, entity=object(), category=category, role="candidate")


def make_rule(rule_id, consequence=Consequence.AVOID, item_ids=("cup_0",), region_ids=(), params=None):
    return RuleSpec(
        rule_id=rule_id, consequence=consequence, item_ids=tuple(item_ids),
        region_ids=tuple(region_ids), params=dict(params or {"margin": 0.0}),
        phrases=["p"], split="train",
    )


def make_ctx(rules, items, active_rule_id=None):
    return SimpleNamespace(
        items=list(items), regions=[], rules=list(rules),
        _stay_ref={}, active_rule_id=active_rule_id, task_cfg={},
    )


def make_tick(margins, k=2, g=1, n_cam=3, n_tok=4):
    data = _TickData()
    data.margins = np.asarray(margins, dtype=np.float64)
    data.dists = np.full(k, 0.3, dtype=np.float64)
    data.held = np.array([-1, -1], dtype=np.int64)
    data.item_pose = np.zeros((k, 7), dtype=np.float64)
    data.item_vel = np.zeros((k, 6), dtype=np.float64)
    data.region_T = np.stack([np.eye(4)] * g) if g else np.zeros((0, 4, 4))
    data.visibility = np.zeros((k, n_cam), dtype=np.float32)
    data.token_masks = np.zeros((k + g, n_cam, n_tok), dtype=np.uint8)
    return data


def make_hfn(ctx, tick, active_rule_id=None, scale=1.0, row_sink=None):
    hfn = ItemMarginHFn(active_rule_id, scale, ctx=ctx, row_sink=row_sink)
    hfn._compute_tick = lambda env: tick
    hfn._per_link_calls = []
    hfn._per_link = lambda env, spec: (
        hfn._per_link_calls.append(spec.item_id),
        {"L/link1": 0.01, "R/link3": 0.02},
    )[1]
    return hfn


ENV = SimpleNamespace(task=SimpleNamespace())
CFG = SimpleNamespace()


def test_h_active_rule_scale():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
             make_rule("avoid|cup_1|-", item_ids=("cup_1",))]
    hfn = make_hfn(make_ctx(rules, items, "avoid|cup_1|-"),
                   make_tick([0.5, -0.02]), active_rule_id="avoid|cup_1|-",
                   scale=0.05)
    h, diag = hfn(ENV, CFG)
    assert h == pytest.approx(0.05 * (-0.02 / 0.05))  # scale x rule_h (20 x metres)
    assert diag["margin_active"] == pytest.approx(-0.02)
    assert diag["active_rule_id"] == "avoid|cup_1|-"


def test_h_min_over_rule_h_without_active():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
             make_rule("avoid|cup_1|-", item_ids=("cup_1",))]
    hfn = make_hfn(make_ctx(rules, items), make_tick([0.5, 0.03]))
    h, diag = hfn(ENV, CFG)
    assert h == pytest.approx(0.6)  # min(clip(0.5 / 0.05), 0.03 / 0.05)
    assert diag["active_rule_id"] is None
    assert diag["margin_active"] == pytest.approx(0.03)  # min raw margin


def test_h_per_rule_unit():
    """1.0 = comfortably safe for every rule: upright's own tilt limit, the
    don't-spill rule's 1 cm (user 2026-10-05; the old kappa 2/rad capped a
    10 deg upright rule at 0.35)."""
    items = [make_item("box_0"), make_item("beads")]
    rules = [
        make_rule("upright|box_0|-", Consequence.UPRIGHT, ("box_0",), params={"theta_max": np.deg2rad(10.0)}),
        make_rule("particles_inside|beads|bowl_src_interior",
                  Consequence.PARTICLES_INSIDE, ("beads",), ("bowl_src_interior",)),
    ]
    hfn = make_hfn(make_ctx(rules, items), make_tick([np.deg2rad(10.0), 0.005]))
    h, diag = hfn(ENV, CFG)
    assert h == pytest.approx(min(1.0, 0.005 / 0.01))  # upright straight = 1.0, beads 5 mm inside = 0.5
    hfn2 = make_hfn(make_ctx(rules, items, "upright|box_0|-"), make_tick([np.deg2rad(5.0), 0.04]),
                    active_rule_id="upright|box_0|-")
    assert hfn2(ENV, CFG)[0] == pytest.approx(0.5)


def test_diag_carries_all_margins_and_compat_keys():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
             make_rule("avoid|cup_1|-", item_ids=("cup_1",)),
             make_rule("avoid|cup_0+cup_1|-", item_ids=("cup_0", "cup_1"))]
    tick = make_tick([0.5, 0.03, 0.03])
    hfn = make_hfn(make_ctx(rules, items), tick)
    hfn._held_tracker = SimpleNamespace(latched={"left": "cup_1", "right": None})
    h, diag = hfn(ENV, CFG)
    np.testing.assert_array_equal(diag["margins"], np.array([0.5, 0.03, 0.03]))
    assert diag["rule_ids"] == [r.rule_id for r in rules]
    # item state
    assert diag["item_pose"].shape == (2, 7)
    assert diag["item_vel"].shape == (2, 6)
    assert diag["held"].tolist() == [-1, -1]
    assert diag["token_masks"].shape == (3, 3, 4)
    assert diag["visibility"].shape == (2, 3)
    assert diag["region_T"].shape == (1, 4, 4)
    # compute_h compat keys (rollout._collect_tick / video extras consumers)
    assert isinstance(diag["per_link"], dict) and diag["per_link"]
    assert diag["d_arm_arm"] == float("inf")
    assert diag["d_left"] == float("inf") and diag["d_right"] == float("inf")
    assert diag["d_left_held"] == float("inf") and diag["d_right_held"] == float("inf")
    assert diag["d_held"] == float("inf")
    assert diag["true_argmin"] == "L/link1"  # min of the stubbed per_link
    assert diag["holding"] == "cup_1"
    assert diag["holding_left"] == "cup_1" and diag["holding_right"] == ""
    assert diag["hold_debug"] == {}
    assert diag["contact"] is False
    assert diag["n_below_table"] == 0
    assert diag["h"] == h


def test_per_link_target_prefers_active_single_item_avoid():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
             make_rule("avoid|cup_1|-", item_ids=("cup_1",))]
    ctx = make_ctx(rules, items, "avoid|cup_0|-")
    hfn = make_hfn(ctx, make_tick([0.5, 0.01]), active_rule_id="avoid|cup_0|-")
    hfn(ENV, CFG)
    # the active rule wins even though cup_1 has the smaller margin
    assert hfn._per_link_calls == ["cup_0"]


def test_per_link_target_falls_back_to_min_margin_single_item():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [
        make_rule("avoid|cup_0+cup_1|-", item_ids=("cup_0", "cup_1")),  # set rule
        make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
        make_rule("avoid|cup_1|-", item_ids=("cup_1",)),
    ]
    hfn = make_hfn(make_ctx(rules, items), make_tick([0.02, 0.5, 0.03]))
    _, diag = hfn(ENV, CFG)
    assert hfn._per_link_calls == ["cup_1"]  # min-margin single-item avoid
    assert diag["true_argmin"] == "L/link1"


def test_per_link_empty_without_single_item_avoid():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0+cup_1|-", item_ids=("cup_0", "cup_1"))]
    hfn = make_hfn(make_ctx(rules, items), make_tick([0.02]))
    _, diag = hfn(ENV, CFG)
    assert hfn._per_link_calls == []
    assert diag["per_link"] == {} and diag["true_argmin"] == ""


def test_row_sink_offer_unpadded_row():
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
             make_rule("avoid|cup_1|-", item_ids=("cup_1",))]
    sink = SimpleNamespace(rows=[], offer=lambda row: sink.rows.append(row))
    hfn = make_hfn(make_ctx(rules, items), make_tick([0.5, 0.03]), row_sink=sink)
    hfn(ENV, CFG)
    assert len(sink.rows) == 1
    row = sink.rows[0]
    assert row["item_pose"].shape == (2, 7) and row["item_pose"].dtype == np.float32
    assert row["item_obb"].shape == (2, 9)  # center_local + half + up_axis
    np.testing.assert_array_equal(
        row["item_obb"][0],
        np.concatenate([items[0].obb_center_local, items[0].half_extents,
                        items[0].up_axis_local]).astype(np.float32),
    )
    assert row["item_valid"].tolist() == [True, True]
    assert row["held"].dtype == np.int8
    assert row["margins"].tolist() == pytest.approx([0.5, 0.03])  # unpadded, raw SI
    assert row["rule_valid"].tolist() == [True, True]
    assert row["region_T"].shape == (1, 4, 4)
    assert row["token_masks"].dtype == np.uint8
    assert row["visibility"].dtype == np.float16


def test_empty_rule_table_h_inf():
    hfn = make_hfn(make_ctx([], []), make_tick([], k=0, g=0))
    h, diag = hfn(ENV, CFG)
    assert h == float("inf")
    assert diag["margins"].size == 0
    assert diag["per_link"] == {}


def test_active_rule_missing_raises():
    items = [make_item("cup_0")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",))]
    hfn = make_hfn(make_ctx(rules, items), make_tick([0.1]),
                   active_rule_id="avoid|nope|-")
    with pytest.raises(KeyError, match="not in the episode's rule table"):
        hfn(ENV, CFG)


def test_bind_reads_env_item_runner():
    ctx = make_ctx([], [])
    hfn = ItemMarginHFn()
    with pytest.raises(ValueError, match="no _item_runner"):
        hfn.bind(SimpleNamespace())
    env = SimpleNamespace(_item_runner=ctx)
    assert hfn.bind(env) is hfn and hfn.ctx is ctx


def test_failure_depth_clip_bounds_h_but_keeps_the_sign():
    """h is clipped to [-1, 1]: a bead 0.9 m off the table / an object
    launched 23 m (audit 2026-10-04) must not read h = -18 / -464; the raw
    margin is reported unchanged; a very safe margin reads +1."""
    items = [make_item("cup_0"), make_item("cup_1")]
    rules = [make_rule("avoid|cup_0|-", item_ids=("cup_0",)),
             make_rule("avoid|cup_1|-", item_ids=("cup_1",))]
    hfn = make_hfn(make_ctx(rules, items, "avoid|cup_1|-"), make_tick([0.30, -23.0]),
                   active_rule_id="avoid|cup_1|-")
    h, diag = hfn(ENV, CFG)
    assert h == pytest.approx(-1.0) and h < 0.0
    assert diag["margin_active"] == pytest.approx(-23.0)  # the raw margin is reported unchanged
    hfn_safe = make_hfn(make_ctx(rules, items, "avoid|cup_0|-"), make_tick([0.30, -23.0]),
                        active_rule_id="avoid|cup_0|-")
    h_safe, _ = hfn_safe(ENV, CFG)
    assert h_safe == pytest.approx(1.0)
