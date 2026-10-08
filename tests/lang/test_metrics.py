"""metrics: recomputed GeoCrossAttention weights reproduce its output; pair stats; enrichment."""

import numpy as np
import torch

from main.lang.metrics import enrichment, geo_attention_weights, geo_output_from_weights, pair_stats, wilson
from main.network.trunk import GeoCrossAttention


def test_geo_weights_reproduce_output():
    torch.manual_seed(0)
    for dir_per_head in (True, False):
        geo = GeoCrossAttention(256, 8, dir_per_head=dir_per_head).eval()
        B, N, M = 2, 22, 50
        body, pos = torch.randn(B, N, 256), torch.randn(B, N, 3)
        scene, spos = torch.randn(B, M, 256), torch.randn(B, M, 3)
        smask = torch.ones(B, M, dtype=torch.bool)
        smask[1, :5] = False
        with torch.no_grad():
            ref = geo(body, pos, scene, spos, smask)
            A = geo_attention_weights(geo, body, pos, scene, spos, smask)
            rebuilt = geo_output_from_weights(geo, A, scene, pos, spos)
        assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-5)
        assert float(A[1, :, :, :5].abs().max()) == 0.0          # masked tokens get no weight
        assert torch.allclose(rebuilt, ref, atol=1e-5)


def test_pair_stats():
    q = np.array([0.1, 0.5, 0.4, 0.2])
    y = np.array([0.0, 0.6, 0.5, 0.1])
    fs = np.array([True, False, False, True])
    out = pair_stats(q, y, fs, np.array([0, 0, 1, 1]))
    assert out["lang/pair_rank_acc"] == 1.0 and out["lang/pair_target_rank_acc"] == 1.0
    assert abs(out["lang/pair_q_gap"] - 0.3) < 1e-9


def test_enrichment():
    A = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert float(enrichment(A, ref)[0]) == 2.0     # half the mass on a quarter of the tokens
    assert torch.isnan(enrichment(A, torch.zeros(1, 4)))[0]


def test_wilson():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and 0.2 < lo < 0.5 < hi < 0.8


def test_episode_row_reads_q_and_ticks():
    """M1: Q comes from trace["V"]; M7: intervened_tick_frac from the per-tick action_source."""
    from main.lang.metrics import LangRoundMetrics

    t = (np.arange(10) * 0.04).tolist()
    trace = {"h_named": [0.5 - 0.05 * k for k in range(10)], "t": t, "V": [0.4 - 0.05 * k for k in range(10)],
             "V_t": t, "intervened": [False] * 8 + [True] * 2, "action_source": [0] * 4 + [1] * 6,
             "h_other_min": [0.9] * 10}
    line = {"task": "gap_place", "embodiment": "ur5-wsg", "episode_id": 1, "success": True,
            "named_consequence": "stay", "designation": "harm"}
    row = LangRoundMetrics.episode_row(3, line, trace)
    assert row["v_le_h_frac"] == 1.0 and np.isfinite(row["hq_corr"])
    assert abs(row["intervention_rate"] - 0.2) < 1e-9 and abs(row["intervened_tick_frac"] - 0.6) < 1e-9


def test_spill_rule_counts_as_another_rule():
    """M2: with a T6 exclude rule named, the two-container spill rule enters h_other_min."""
    from types import SimpleNamespace

    from main.item_envs.rules import Consequence, RuleSpec
    from main.lang.rollout import LangRolloutController, single_rule

    excl = RuleSpec("particles_outside|beads|bin_blue", Consequence.PARTICLES_OUTSIDE, ("beads",), ("bin_blue",), {})
    spill = RuleSpec("particles_inside|beads|bowl_src_interior+target_interior", Consequence.PARTICLES_INSIDE,
                     ("beads",), ("bowl_src_interior", "target_interior"), {})
    zones = RuleSpec("inside|box|mat_a+mat_b", Consequence.INSIDE, ("box",), ("mat_a", "mat_b"), {})
    pair = RuleSpec("avoid|cup+mug|-", Consequence.AVOID, ("cup", "mug"), (), {"margin": 0.01})
    assert single_rule(excl) and single_rule(spill) and not single_rule(zones) and not single_rule(pair)
    fake = SimpleNamespace(
        _last_diag={"margins": np.array([0.03, -0.004]), "violation_counts": np.array([0, 2]),
                    "robot_item_dist": np.array([]), "held": np.array([-1, -1])},
        env=SimpleNamespace(_item_runner=SimpleNamespace(rules=[excl, spill], items=[])),
        named_rule_id=excl.rule_id, _tick_source=0,
        trace={k: [] for k in ("h_all", "h_named", "h_other_min", "action_source", "named_dist", "held")})
    LangRolloutController._trace_rules(fake)
    assert fake.trace["h_named"][0] > 0
    assert fake.trace["h_other_min"][0] < 0          # the spill (2 beads out) reaches the "other" minimum
