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
