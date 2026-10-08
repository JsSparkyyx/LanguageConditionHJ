"""LangPolicyShim frame history (VANILLA_PLAN 7.5), CPU, with a stand-in controller."""

import dataclasses
from types import SimpleNamespace as NS

import torch

from main.lang.config import LangConfig
from main.lang.network import LangPolicyEncoder
from main.lang.rollout import LangPolicyShim

from .test_network import body_inputs

NTOK_CAM = 4   # 3 cameras x 4 tokens = 12 per frame


def make(k=2):
    cfg = dataclasses.replace(LangConfig(), k_prev=k)
    torch.manual_seed(0)
    pe = LangPolicyEncoder(cfg, n_tok=3 * NTOK_CAM).eval()
    ro = NS(_in_capture=False, text_out=None, cf_text_out=None)
    w = torch.randn(1, 24, 256)
    m = torch.zeros(1, 24, dtype=torch.bool)
    m[:, :8] = True
    with torch.no_grad():
        ro.text_out = pe.encode_text(w, m)
    return ro, LangPolicyShim(ro, pe, k), pe


def frame(i):
    """[1, n_cam, 4, 256] tokens whose value encodes the capture index i."""
    return torch.full((1, 3, NTOK_CAM, 256), float(i)), torch.full((1, 3, NTOK_CAM, 3), float(i))


def call(ro, shim, i, capture: bool):
    body, _ = body_inputs("ur5-wsg", 1, torch.Generator().manual_seed(0))
    seen = {}
    orig = shim.pe.forward

    def spy(*a, **kw):
        seen["st"] = a[5].clone()
        return orig(*a, **kw)

    shim.pe.forward = spy
    ro._in_capture = capture
    tok, pos = frame(i)
    shim(body["state_tokens"], body["body_tokens"], body["link_pos"], body["body_mask"], body["joint_index"], tok, pos)
    ro._in_capture = False
    shim.pe.forward = orig
    return [int(seen["st"][0, f, 0, 0]) for f in range(3)]   # frame indices [t, t-k, 0]


def test_history_t_tk_0():
    ro, shim, _ = make(k=2)
    got = [call(ro, shim, i, capture=True) for i in range(6)]
    # t - k is frame 0 until k + 1 frames exist (the sampler's max(t - k, 0))
    assert got[0] == [0, 0, 0]
    assert got[1] == [1, 0, 0]
    assert got[2] == [2, 0, 0]
    assert got[3] == [3, 1, 0]
    assert got[5] == [5, 3, 0]
    assert shim.n_pushed == 6


def test_cache_miss_does_not_push():
    ro, shim, _ = make(k=2)
    for i in range(3):
        call(ro, shim, i, capture=True)
    n = shim.n_pushed
    got = call(ro, shim, 99, capture=False)   # base cache-miss path: fresh frame as t, not pushed
    assert got == [99, 0, 0] and shim.n_pushed == n
    assert call(ro, shim, 3, capture=True) == [3, 1, 0]   # the history is unchanged by the miss


def test_counterfactual_encoding():
    ro, shim, pe = make()
    w = torch.randn(1, 24, 256)
    m = torch.ones(1, 24, dtype=torch.bool)
    with torch.no_grad():
        ro.cf_text_out = pe.encode_text(w, m)
    call(ro, shim, 0, capture=True)
    assert shim.enc_cf is not None


def test_filter_margins_in_h_units():
    """O5: the language filter compares Q with the margins as LangConfig states them (h units), not x h_scale."""
    from types import SimpleNamespace

    from main.lang.rollout import LangRolloutController

    fake = SimpleNamespace(cfg=SimpleNamespace(filter_margin=0.02, filter_release_margin=0.05, h_scale=20.0),
                           flt=SimpleNamespace(margin=0.4, release_margin=1.0))     # what the base class set
    LangRolloutController._h_unit_margins(fake)
    assert (fake.flt.margin, fake.flt.release_margin) == (0.02, 0.05)
