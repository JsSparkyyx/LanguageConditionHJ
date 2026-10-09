"""Model tests (VANILLA_PLAN 5.10), CPU-sized (n_tok per frame = 40)."""

import dataclasses

import pytest
import torch

from main.lang.config import LangConfig
from main.lang.network import (LangPolicyEncoder, LangTwinCritic, TextAdapter, action_effects, n_params,
                               param_groups, set_probe, softmin)

NTOK = 40
SWITCHES = ("lang_scene", "lang_body", "lang_film", "lang_queries", "lang_rtoken")


def body_inputs(emb: str, B: int, g: torch.Generator):
    """CrossSafe layout: 20 body slots; per arm n links (UR5 7, Franka 8: joints + gripper),
    joint_index -1 for grippers and padding, column 14 of body_tokens = arm id."""
    n = 7 if emb == "ur5-wsg" else 8
    nj = n + 1                       # action columns per arm: arm joints + 2 prismatic
    A = 18
    st = torch.randn(B, 16, 256, generator=g)
    st[:, 2 * n:] = 0                # padded HoloBrain state tokens
    bt = torch.randn(B, 20, 16, generator=g)
    lp = torch.randn(B, 20, 3, generator=g) * 0.3
    bm = torch.zeros(B, 20, dtype=torch.bool)
    ji = torch.full((B, 20), -1, dtype=torch.long)
    for a in range(2):
        for i in range(n):
            t = a * n + i
            bm[:, t] = True
            bt[:, t, 14] = float(a)
            if i < n - 1:
                ji[:, t] = a * nj + i
    bt[:, 2 * n:, 14] = 0.0
    Jlin = torch.randn(B, 20, 3, A, generator=g) * 0.1
    Jang = torch.randn(B, 20, 3, A, generator=g) * 0.1
    cols = torch.zeros(A, dtype=torch.bool)
    cols[ji[0][ji[0] >= 0]] = True
    dtheta = torch.randn(B, A, generator=g) * 0.01 * cols
    dmax = torch.full((B, A), 0.02) * cols
    return dict(state_tokens=st, body_tokens=bt, link_pos=lp, body_mask=bm, joint_index=ji), \
        dict(dtheta=dtheta, Jlin=Jlin, Jang=Jang, joint_index=ji, dtheta_max=dmax)


def scene_inputs(B, g):
    return torch.randn(B, 3, NTOK, 256, generator=g), torch.randn(B, 3, NTOK, 3, generator=g)


def words(B, g, L=24, lens=(9, 14)):
    w = torch.randn(B, L, 256, generator=g)
    m = torch.zeros(B, L, dtype=torch.bool)
    for i in range(B):
        m[i, : lens[i % len(lens)]] = True
    return w, m


def build(**kw):
    # this file tests the softmin readout (per-token values V, padding filled with 1e4); the MLP readout, the
    # default since decision 6, is covered by test_critic_readout.py
    cfg = dataclasses.replace(LangConfig(), **{"critic_readout": "softmin", **kw})
    torch.manual_seed(0)
    return cfg, LangPolicyEncoder(cfg, n_tok=NTOK).eval(), LangTwinCritic(cfg).eval()


def run(pe, cr, body, act, scene, w, wm, scene_mask=None):
    enc = pe(**body, scene_tokens=scene[0], scene_pos=scene[1], words=w, word_mask=wm, scene_mask=scene_mask)
    q1, q2, v1, v2 = cr(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
    return enc, q1, q2, v1, v2


@pytest.mark.parametrize("emb", ["ur5-wsg", "franka-panda"])
def test_shapes(emb):
    g = torch.Generator().manual_seed(1)
    cfg, pe, cr = build()
    body, act = body_inputs(emb, 2, g)
    enc, q1, q2, v1, v2 = run(pe, cr, body, act, scene_inputs(2, g), *words(2, g))
    assert enc.body.shape == (2, 22, 256) and enc.pos.shape == (2, 22, 3) and enc.mask.shape == (2, 22)
    assert enc.scene.shape == (2, 3 * NTOK, 256) and enc.scene_pos.shape == (2, 3 * NTOK, 3)
    assert enc.words.shape == (2, 24, 256) and enc.r0.shape == (2, 256)
    assert enc.r.shape == (2, 1, 256) and enc.queries.shape == (2, 4, 256)
    assert q1.shape == q2.shape == (2,) and v1.shape == (2, 26)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()


def test_scene_frames_assert():
    g = torch.Generator().manual_seed(1)
    cfg, pe, cr = build()
    body, act = body_inputs("ur5-wsg", 2, g)
    bad = torch.randn(2, 3 * NTOK, 256)  # cameras/frames not split
    with pytest.raises(AssertionError):
        pe(**body, scene_tokens=bad, scene_pos=torch.randn(2, 3 * NTOK, 3), words=torch.randn(2, 24, 256),
           word_mask=torch.ones(2, 24, dtype=torch.bool))


def test_padding_invariance():
    g = torch.Generator().manual_seed(2)
    cfg, pe, cr = build()
    body, act = body_inputs("ur5-wsg", 2, g)     # 14 real tokens, 6 padded
    scene = scene_inputs(2, g)
    w, wm = words(2, g)
    smask = torch.ones(2, 3, NTOK, dtype=torch.bool)
    smask[:, 1, :7] = False
    with torch.no_grad():
        _, q_ref, _, _, _ = run(pe, cr, body, act, scene, w, wm, smask)
        b2 = {k: v.clone() for k, v in body.items()}
        b2["body_tokens"][:, 14:, :14] += 5.0          # padded body slots (arm id col untouched)
        b2["link_pos"][:, 14:] += 1.0
        b2["state_tokens"][:, 14:] += 3.0              # beyond the 14 real state tokens
        _, q_b, _, _, _ = run(pe, cr, b2, act, scene, w, wm, smask)
        w2 = w.clone()
        w2[~wm] = torch.randn(int((~wm).sum()), 256)    # padded words
        _, q_w, _, _, _ = run(pe, cr, body, act, scene, w2, wm, smask)
        s2 = (scene[0].clone(), scene[1].clone())
        s2[0][:, 1, :7] += 4.0                          # masked scene tokens
        s2[1][:, 1, :7] -= 1.0
        _, q_s, _, _, _ = run(pe, cr, body, act, s2, w, wm, smask)
    assert torch.allclose(q_ref, q_b, atol=1e-5)
    assert torch.allclose(q_ref, q_w, atol=1e-5)
    assert torch.allclose(q_ref, q_s, atol=1e-5)


def test_batch_permutation():
    g = torch.Generator().manual_seed(3)
    cfg, pe, cr = build()
    bu, au = body_inputs("ur5-wsg", 2, g)
    bf, af = body_inputs("franka-panda", 2, g)
    body = {k: torch.cat([bu[k], bf[k]]) for k in bu}
    act = {k: torch.cat([au[k], af[k]]) for k in au}
    scene = scene_inputs(4, g)
    w, wm = words(4, g, lens=(5, 9, 14, 20))
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        enc, q1, q2, v1, v2 = run(pe, cr, body, act, scene, w, wm)
        encp, q1p, q2p, v1p, v2p = run(pe, cr, {k: v[perm] for k, v in body.items()},
                                       {k: v[perm] for k, v in act.items()},
                                       (scene[0][perm], scene[1][perm]), w[perm], wm[perm])
    for a, b in ((q1, q1p), (q2, q2p), (v1, v1p), (v2, v2p), (enc.body, encp.body), (enc.scene, encp.scene),
                 (enc.queries, encp.queries), (enc.r0, encp.r0)):
        assert torch.allclose(a[perm], b, atol=1e-5)


def _two_sentences(cfg, pe, cr, seed=4):
    g = torch.Generator().manual_seed(seed)
    body, act = body_inputs("franka-panda", 1, g)
    scene = scene_inputs(1, g)
    w1, m1 = words(1, g, lens=(9,))
    w2, m2 = words(1, g, lens=(13,))
    with torch.no_grad():
        e1, qa, _, _, _ = run(pe, cr, body, act, scene, w1, m1)
        e2, qb, _, _, _ = run(pe, cr, body, act, scene, w2, m2)
    return e1, e2, qa, qb


def test_all_switches_off_text_cannot_matter():
    cfg, pe, cr = build(**{s: False for s in SWITCHES})
    _, _, qa, qb = _two_sentences(cfg, pe, cr)
    assert torch.equal(qa, qb)
    cfg, pe, cr = build()
    _, _, qa, qb = _two_sentences(cfg, pe, cr)
    assert not torch.allclose(qa, qb, atol=1e-6)


@pytest.mark.parametrize("switch", SWITCHES)
def test_each_switch(switch):
    # off alone: its parameters do not exist
    cfg, pe, cr = build(**{switch: False})
    names = {n for n, _ in pe.named_parameters()} | {"critic." + n for n, _ in cr.named_parameters()}
    absent = {
        "lang_scene": lambda: not any(n.startswith("scene_reads_words") for n in names),
        "lang_body": lambda: not any(".attn_bw." in n or ".norm_b." in n for n in names),
        "lang_film": lambda: not any(".film." in n for n in names),
        "lang_queries": lambda: not any(n.endswith("text_adapter.q") or ".attn_qs." in n or ".ffn_q." in n
                                        or "value_head_q" in n for n in names),
        "lang_rtoken": lambda: "r_static" in names,   # the sentence-independent token replaces r0
    }[switch]
    assert absent(), switch
    if switch == "lang_scene":
        e1, e2, _, _ = _two_sentences(cfg, pe, cr)
        assert torch.equal(e1.scene, e2.scene)
    # on alone (every other path off): the path alone carries the sentence to Q
    only = {s: (s == switch) for s in SWITCHES}
    cfg, pe, cr = build(film_zero_init=False, **only)
    _, _, qa, qb = _two_sentences(cfg, pe, cr)
    assert not torch.allclose(qa, qb, atol=1e-7), f"{switch} alone does not reach Q"


def test_film_zero_init_is_identity():
    cfg, pe, cr = build(**{s: (s == "lang_film") for s in SWITCHES})
    _, _, qa, qb = _two_sentences(cfg, pe, cr)
    assert torch.equal(qa, qb)   # FiLM starts as the identity


def test_gradients_reach_every_parameter():
    g = torch.Generator().manual_seed(5)
    cfg, pe, cr = build()
    pe.train(), cr.train()
    body, act = body_inputs("ur5-wsg", 3, g)
    scene = scene_inputs(3, g)
    w, wm = words(3, g, lens=(9, 14, 24))
    enc, q1, q2, _, _ = run(pe, cr, body, act, scene, w, wm)
    y = torch.randn(3) * 0.3
    loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
    loss.backward()
    dead = [n for n, p in list(pe.named_parameters()) + [("critic." + n, p) for n, p in cr.named_parameters()]
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead
    # frozen inputs (HoloBrain scene / state tokens, text-store words) never get gradients
    assert not w.requires_grad and not scene[0].requires_grad and not body["state_tokens"].requires_grad


def test_action_effects_parity():
    from main.network.heads import CriticHead, Encoded

    g = torch.Generator().manual_seed(6)
    for emb in ("ur5-wsg", "franka-panda"):
        body, act = body_inputs(emb, 3, g)
        head = CriticHead(temperature=0.02).eval()
        seen = {}
        head.act_proj.register_forward_hook(lambda m, i, o: seen.__setitem__("eff", i[0].detach()))
        head.blocks[0].cross.register_forward_pre_hook(lambda m, a: seen.__setitem__("pos", a[1].detach()))
        pos = torch.randn(3, 22, 3)
        enc = Encoded(torch.randn(3, 22, 256), pos, torch.ones(3, 22, dtype=torch.bool),
                      torch.randn(3, 10, 256), torch.randn(3, 10, 3), torch.ones(3, 10, dtype=torch.bool))
        with torch.no_grad():
            head(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
        eff, dp = action_effects(act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"], 2, torch.float32)
        assert (eff - seen["eff"]).abs().max() < 1e-6
        assert (pos + dp - seen["pos"]).abs().max() < 1e-6


def test_softmin_equal_values():
    T = 0.05
    for n in (1, 5, 26):
        v = torch.full((2, n), 0.37)
        assert torch.allclose(softmin(v, T), torch.full((2,), 0.37 - T * torch.log(torch.tensor(float(n)))), atol=1e-6)


def test_padding_loses_the_min():
    cfg, pe, cr = build()
    g = torch.Generator().manual_seed(7)
    body, act = body_inputs("ur5-wsg", 1, g)
    with torch.no_grad():
        enc, q1, _, v1, _ = run(pe, cr, body, act, scene_inputs(1, g), *words(1, g, lens=(9,)))
    assert (v1[0, 14:20] == 1e4).all()
    valid = torch.cat([v1[0, :14], v1[0, 20:]])
    assert float(q1[0]) <= float(valid.min()) + 1e-6


def test_text_adapter_backends():
    for backend, d in (("bert_raw", 768), ("sentence_generic", 768)):
        cfg = dataclasses.replace(LangConfig(), text_backend=backend)
        ad = TextAdapter(cfg, d_in=d)
        r0, q, w = ad(torch.randn(2, 24, d), torch.ones(2, 24, dtype=torch.bool))
        assert r0.shape == (2, 256) and q.shape == (2, 4, 256) and w.shape == (2, 24, 256)
        assert isinstance(ad.in_proj, torch.nn.Linear)


def test_deployment_text_out_broadcast_and_probe():
    g = torch.Generator().manual_seed(8)
    cfg, pe, cr = build()
    body, act = body_inputs("ur5-wsg", 2, g)
    scene = scene_inputs(2, g)
    w, wm = words(1, g, lens=(11,))
    with torch.no_grad():
        text_out = pe.encode_text(w, wm)
        enc_a = pe(**body, scene_tokens=scene[0], scene_pos=scene[1], text_out=text_out)
        enc_b = pe(**body, scene_tokens=scene[0], scene_pos=scene[1], words=w.expand(2, -1, -1),
                   word_mask=wm.expand(2, -1))
        assert torch.allclose(enc_a.body, enc_b.body, atol=1e-5)
        stores = set_probe(pe, True)
        pe(**body, scene_tokens=scene[0], scene_pos=scene[1], text_out=text_out)
        assert "scene_delta_norm" in stores[0] and "body_word_attn" in stores[1]
        assert stores[1]["body_word_attn"].shape == (2, 22, 24)
        set_probe(pe, False)


def test_param_groups_cover_encoder():
    cfg, pe, cr = build()
    groups = param_groups(pe)
    ids = {id(p) for ps in groups.values() for p in ps}
    assert ids == {id(p) for p in pe.parameters()}
    assert n_params(pe.parameters()) > 0 and n_params(cr.parameters()) > 0
