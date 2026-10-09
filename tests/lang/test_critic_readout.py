"""Critic readout (decision 6, FUSION_PLAN 3.6), CPU-sized (n_tok per frame = 40): the MLP over [r, q_1..q_K,
masked body mean] that new runs use, the softmin that configs saved before the field keep, and the config rules
that come with the new fields (legacy from_dict, collect_hash)."""

import dataclasses
from pathlib import Path

import pytest
import torch

from main.lang.collect import TRAIN_ONLY_FIELDS, collect_hash
from main.lang.config import LangConfig
from main.lang.network import (DIM, LangEncoded, LangPolicyEncoder, LangTwinCritic, action_effects, n_queries,
                               softmin)

from .test_network import NTOK, SWITCHES, body_inputs, scene_inputs, words

REPO = Path(__file__).resolve().parents[2]
# fields added on 2026-10-09 (V5, decision 6, E6); a config saved before them has none of these keys
NEW_FIELDS = ("scene_fusion", "fusion_words", "fusion_dtype", "fusion_chunk", "critic_readout", "critic_mlp_hidden",
              "train_episodes_per_task")
# a finished run saved before the field (E2 V0, softmin critic); the test that reads it skips when it is absent
OLD_CKPT = REPO / "runs/lang/grounding_1008/g1008_v0_s0/checkpoint_final.pt"


def old_dict() -> dict:
    """LangConfig().to_dict() as a run saved before the new fields would have written it."""
    return {k: v for k, v in LangConfig().to_dict().items() if k not in NEW_FIELDS}


def build(readout: str = "mlp", seed: int = 0, **kw):
    cfg = dataclasses.replace(LangConfig(), critic_readout=readout, **kw)
    torch.manual_seed(seed)
    return cfg, LangPolicyEncoder(cfg, n_tok=NTOK).eval(), LangTwinCritic(cfg).eval()


def mixed_batch(g):
    """ur5-wsg (14 real link tokens, 6 padded) and franka-panda (16 real, 4 padded) in one batch of 4."""
    bu, au = body_inputs("ur5-wsg", 2, g)
    bf, af = body_inputs("franka-panda", 2, g)
    return {k: torch.cat([bu[k], bf[k]]) for k in bu}, {k: torch.cat([au[k], af[k]]) for k in au}


def run(pe, cr, body, act, scene, w, wm):
    enc = pe(**body, scene_tokens=scene[0], scene_pos=scene[1], words=w, word_mask=wm)
    return (enc,) + tuple(cr(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"]))


def head_inputs(g, n_real=(14, 16), K=4, M=20, A=18):
    """A LangEncoded built directly (the critic head's input), B = len(n_real): sample i has n_real[i] valid link
    tokens first, then M - n_real[i] padded slots, then the 2 always-valid residual tokens."""
    B = len(n_real)
    mask = torch.zeros(B, M + 2, dtype=torch.bool)
    ji = torch.full((B, M), -1, dtype=torch.long)
    for i, n in enumerate(n_real):
        mask[i, :n] = True
        ji[i, :n] = torch.arange(n) % A
    mask[:, M:] = True
    enc = LangEncoded(torch.randn(B, M + 2, DIM, generator=g), torch.randn(B, M + 2, 3, generator=g) * 0.3, mask,
                      torch.randn(B, 3 * NTOK, DIM, generator=g), torch.randn(B, 3 * NTOK, 3, generator=g),
                      torch.ones(B, 3 * NTOK, dtype=torch.bool), torch.randn(B, 24, DIM, generator=g),
                      torch.arange(24)[None].expand(B, -1) < 11, torch.randn(B, DIM, generator=g),
                      torch.randn(B, 1, DIM, generator=g), torch.randn(B, K, DIM, generator=g))
    act = dict(dtheta=torch.randn(B, A, generator=g) * 0.01, Jlin=torch.randn(B, M, 3, A, generator=g) * 0.1,
               Jang=torch.randn(B, M, 3, A, generator=g) * 0.1, joint_index=ji)
    return enc, act


def add_pads(enc, act, g, n=4):
    """The same samples with n more padded link slots (before the 2 residual tokens), holding random content."""
    B, M = act["joint_index"].shape
    ins = lambda t, new: torch.cat([t[:, :M], new, t[:, M:]], 1)
    enc = dataclasses.replace(enc, body=ins(enc.body, torch.randn(B, n, DIM, generator=g)),
                              pos=ins(enc.pos, torch.randn(B, n, 3, generator=g)),
                              mask=ins(enc.mask, torch.zeros(B, n, dtype=torch.bool)))
    A = act["dtheta"].shape[1]
    act = dict(act, Jlin=torch.cat([act["Jlin"], torch.zeros(B, n, 3, A)], 1),
               Jang=torch.cat([act["Jang"], torch.zeros(B, n, 3, A)], 1),
               joint_index=torch.cat([act["joint_index"], torch.full((B, n), -1)], 1))
    return enc, act


def take(t, idx):
    """t [B, N, ...] reordered along dim 1 with a per-sample index idx [B, N]."""
    return torch.take_along_dim(t, idx.view(*idx.shape, *([1] * (t.dim() - 2))), dim=1)


def q_head(head, enc, act):
    return head(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])


def softmin_reference(head, enc, dtheta, Jlin, Jang, joint_index):
    """LangCriticHead.forward as it was before decision 6 (commit bca6bc5), copied: the softmin path must stay
    this computation, op for op, so old checkpoints score bit for bit."""
    M = joint_index.shape[-1]
    n_res = enc.body.shape[1] - M
    eff, dp = action_effects(dtheta.to(enc.body.device), Jlin, Jang, joint_index, n_res, enc.body.dtype)
    b = enc.body + head.act_proj(eff)
    x = enc.pos + dp
    r, q = enc.r, enc.queries
    for blk in head.blocks:
        b, r, q = blk(b, x, enc.mask, r, q, enc.scene, enc.scene_pos, enc.scene_mask, enc.words, enc.word_mask,
                      enc.r0)
    V = head.value_head(b).squeeze(-1).masked_fill(~enc.mask, 1e4)
    if head.K:
        V = torch.cat([V, head.value_head_q(q).squeeze(-1)], dim=-1)
    return softmin(V, head.temperature), V


# ---------------------------------------------------------------------------
# the MLP readout
# ---------------------------------------------------------------------------


def test_mlp_shapes_and_no_value_heads():
    g = torch.Generator().manual_seed(1)
    cfg, pe, cr = build()
    assert LangConfig().critic_readout == "mlp" and cfg.critic_mlp_hidden == 512
    body, act = mixed_batch(g)
    with torch.no_grad():
        enc, q1, q2, v1, v2 = run(pe, cr, body, act, scene_inputs(4, g), *words(4, g, lens=(5, 9, 14, 20)))
        qm = cr.qmin(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
    assert q1.shape == q2.shape == (4,) and v1 is None and v2 is None
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all() and not torch.equal(q1, q2)
    assert torch.equal(qm, torch.minimum(q1, q2))
    names = [n for n, _ in cr.named_parameters()]
    assert not any("value_head" in n for n in names), names
    for head in (cr.c1, cr.c2):
        assert head.readout == "mlp" and not hasattr(head, "temperature")   # softmin_T_h is not read
        lin1, lin2 = head.readout_mlp[0], head.readout_mlp[2]
        assert (lin1.in_features, lin1.out_features, lin2.out_features) == ((4 + 2) * DIM, 512, 1)
        n = sum(p.numel() for p in list(head.readout_norm.parameters()) + list(head.readout_mlp.parameters()))
        assert n == DIM + (6 * DIM + 1) * 512 + 513     # ~0.8 M per head, each twin its own (FUSION_PLAN 3.6)


@pytest.mark.parametrize("queries,rtoken", [(True, True), (False, True), (True, False), (False, False)])
def test_mlp_builds_without_queries_or_rtoken(queries, rtoken):
    """K = 0 (lang_queries off): the input is [r, b_mean] = 2 x 256; lang_rtoken off: r is the static token after
    the blocks, still read."""
    g = torch.Generator().manual_seed(2)
    cfg, pe, cr = build(lang_queries=queries, lang_rtoken=rtoken, critic_mlp_hidden=64)
    K = 4 if queries else 0
    assert cr.c1.readout_mlp[0].in_features == (K + 2) * DIM and cr.c1.readout_mlp[0].out_features == 64
    body, act = mixed_batch(g)
    with torch.no_grad():
        enc, q1, q2, v1, v2 = run(pe, cr, body, act, scene_inputs(4, g), *words(4, g))
    assert enc.queries.shape == (4, K, DIM) and q1.shape == (4,) and v1 is None and torch.isfinite(q1).all()


def test_mlp_reads_r_queries_and_masked_body_mean():
    """The readout's input is [r, q_1..q_K, mean of the body tokens over enc.mask] after the critic blocks."""
    g = torch.Generator().manual_seed(3)
    cfg, pe, cr = build()
    head = cr.c1
    enc, act = head_inputs(g, n_real=(14, 16, 7))
    seen = {}
    h1 = head.blocks[-1].register_forward_hook(lambda m, i, o: seen.__setitem__("out", [t.detach() for t in o]))
    h2 = head.readout_norm.register_forward_pre_hook(lambda m, a: seen.__setitem__("parts", a[0].detach()))
    with torch.no_grad():
        q, v = q_head(head, enc, act)
    h1.remove(), h2.remove()
    b, r, qs = seen["out"]
    parts = seen["parts"]
    assert parts.shape == (3, 4 + 2, DIM) and v is None
    assert torch.equal(parts[:, :1], r) and torch.equal(parts[:, 1:5], qs)
    for i in range(3):
        mean = b[i][enc.mask[i]].mean(0)
        assert torch.allclose(parts[i, 5], mean, atol=1e-6)


def test_mlp_padding_content_order_and_count():
    """Q does not depend on what the padded body slots hold, on where the pads sit among the link tokens, or on
    how many pads there are (each sample of a mixed-embodiment batch uses its own count)."""
    g = torch.Generator().manual_seed(4)
    cfg, pe, cr = build()
    head = cr.c1
    enc, act = head_inputs(g, n_real=(14, 16))
    pad = ~enc.mask
    with torch.no_grad():
        q_ref, _ = q_head(head, enc, act)
        # content: padded slots refilled (body and positions)
        body2, pos2 = enc.body.clone(), enc.pos.clone()
        body2[pad] = torch.randn(int(pad.sum()), DIM, generator=g) * 5.0
        pos2[pad] += 2.0
        q_content, _ = q_head(head, dataclasses.replace(enc, body=body2, pos=pos2), act)
        # order: the 20 link slots of each sample shuffled (pads interleaved), with their action inputs
        M = act["joint_index"].shape[1]
        perm = torch.stack([torch.randperm(M, generator=g) for _ in range(2)])
        full = torch.cat([perm, torch.arange(M, M + 2).expand(2, -1)], 1)     # residual tokens stay last
        enc_p = dataclasses.replace(enc, body=take(enc.body, full), pos=take(enc.pos, full), mask=take(enc.mask, full))
        act_p = dict(act, Jlin=take(act["Jlin"], perm), Jang=take(act["Jang"], perm),
                     joint_index=take(act["joint_index"], perm))
        q_order, _ = q_head(head, enc_p, act_p)
        assert not torch.equal(enc_p.mask, enc.mask)     # the pads did move
        # count: 4 more padded slots, the same real tokens
        q_count, _ = q_head(head, *add_pads(enc, act, g))
        # per sample: each sample of the batch alone
        q_alone = torch.cat([q_head(head, enc.index(slice(i, i + 1)),
                                    {k: v[i:i + 1] for k, v in act.items()})[0] for i in range(2)])
    assert torch.allclose(q_ref, q_content, atol=1e-6)
    assert torch.allclose(q_ref, q_order, atol=1e-5)
    assert torch.allclose(q_ref, q_count, atol=1e-5)
    assert torch.allclose(q_ref, q_alone, atol=1e-5)


def test_mlp_full_model_padding_and_batch():
    """Through the encoder too: padded body slots never reach Q, and a mixed-embodiment batch scores every sample
    as it scores alone (and in any batch order)."""
    g = torch.Generator().manual_seed(5)
    cfg, pe, cr = build()
    body, act = mixed_batch(g)
    scene = scene_inputs(4, g)
    w, wm = words(4, g, lens=(5, 9, 14, 20))
    with torch.no_grad():
        _, q_ref, q2_ref, _, _ = run(pe, cr, body, act, scene, w, wm)
        b2 = {k: v.clone() for k, v in body.items()}
        n = (2 * 7, 2 * 7, 2 * 8, 2 * 8)
        for i, ni in enumerate(n):
            b2["body_tokens"][i, ni:, :14] += 5.0       # padded body slots (arm id col untouched)
            b2["link_pos"][i, ni:] += 1.0
            b2["state_tokens"][i, ni:] += 3.0
        _, q_pad, _, _, _ = run(pe, cr, b2, act, scene, w, wm)
        alone = [run(pe, cr, {k: v[i:i + 1] for k, v in body.items()}, {k: v[i:i + 1] for k, v in act.items()},
                     (scene[0][i:i + 1], scene[1][i:i + 1]), w[i:i + 1], wm[i:i + 1])[1] for i in range(4)]
        perm = torch.tensor([2, 0, 3, 1])
        _, q_perm, q2_perm, _, _ = run(pe, cr, {k: v[perm] for k, v in body.items()},
                                       {k: v[perm] for k, v in act.items()}, (scene[0][perm], scene[1][perm]),
                                       w[perm], wm[perm])
    assert torch.allclose(q_ref, q_pad, atol=1e-5)
    assert torch.allclose(q_ref, torch.cat(alone), atol=1e-5)
    assert torch.allclose(q_ref[perm], q_perm, atol=1e-5) and torch.allclose(q2_ref[perm], q2_perm, atol=1e-5)


def test_mlp_gradients_reach_r_queries_and_body():
    """Every sample sends gradient into r, every query and every valid body token (the dense gradient decision 6
    is for); padded body tokens get exactly none."""
    g = torch.Generator().manual_seed(6)
    cfg, pe, cr = build()
    head = cr.c1.train()
    enc, act = head_inputs(g, n_real=(14, 16, 9))
    leaves = {k: getattr(enc, k).clone().requires_grad_(True) for k in ("body", "r", "queries")}
    q, _ = q_head(head, dataclasses.replace(enc, **leaves), act)
    q.sum().backward()
    gb, gr, gq = (leaves[k].grad for k in ("body", "r", "queries"))
    assert (gr.norm(dim=-1) > 0).all() and (gq.norm(dim=-1) > 0).all()          # [B, 1], [B, K]
    assert (gb.norm(dim=-1)[enc.mask] > 0).all() and (gb[~enc.mask] == 0).all()


def test_mlp_gradients_reach_every_critic_parameter():
    g = torch.Generator().manual_seed(7)
    cfg, pe, cr = build()
    pe.train(), cr.train()
    body, act = body_inputs("ur5-wsg", 3, g)
    enc, q1, q2, _, _ = run(pe, cr, body, act, scene_inputs(3, g), *words(3, g, lens=(9, 14, 24)))
    y = torch.randn(3) * 0.3
    (((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()).backward()
    dead = [n for n, p in list(pe.named_parameters()) + [("critic." + n, p) for n, p in cr.named_parameters()]
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, dead


def _two_sentences(pe, cr, seed=8):
    g = torch.Generator().manual_seed(seed)
    body, act = body_inputs("franka-panda", 1, g)
    scene = scene_inputs(1, g)
    w1, m1 = words(1, g, lens=(9,))
    w2, m2 = words(1, g, lens=(13,))
    with torch.no_grad():
        qa = run(pe, cr, body, act, scene, w1, m1)[1]
        qb = run(pe, cr, body, act, scene, w2, m2)[1]
    return qa, qb


def test_mlp_text_paths():
    """DESIGN_CONTEXT 5.1 under the new readout: all five paths off, the sentence cannot affect Q; each path
    alone reaches Q."""
    _, pe, cr = build(**{s: False for s in SWITCHES})
    qa, qb = _two_sentences(pe, cr)
    assert torch.equal(qa, qb)
    for switch in SWITCHES:
        _, pe, cr = build(film_zero_init=False, **{s: (s == switch) for s in SWITCHES})
        qa, qb = _two_sentences(pe, cr)
        assert not torch.allclose(qa, qb, atol=1e-7), f"{switch} alone does not reach Q"


# ---------------------------------------------------------------------------
# the softmin readout: configs and checkpoints saved before the field
# ---------------------------------------------------------------------------


def test_softmin_is_the_old_computation_and_round_trips():
    """A config without critic_readout builds the softmin head with the old parameter names; its Q equals the
    pre-decision-6 forward bit for bit, and a state_dict round trip (strict) reproduces Q bit for bit."""
    cfg = LangConfig.from_dict(old_dict())
    assert cfg.critic_readout == "softmin"
    g = torch.Generator().manual_seed(9)
    torch.manual_seed(0)
    cr = LangTwinCritic(cfg).eval()
    tops = {n.split(".")[1] for n in cr.state_dict()}
    assert tops == {"act_proj", "blocks", "value_head", "value_head_q"}, tops
    assert cr.c1.temperature == cfg.softmin_T_h
    enc, act = head_inputs(g, n_real=(14, 16))
    with torch.no_grad():
        q1, q2, v1, v2 = cr(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
        for head, q, v in ((cr.c1, q1, v1), (cr.c2, q2, v2)):
            q_ref, v_ref = softmin_reference(head, enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
            assert torch.equal(q, q_ref) and torch.equal(v, v_ref)
        assert v1.shape == (2, 22 + 4) and (v1[0, 14:20] == 1e4).all()
        torch.manual_seed(1)
        back = LangTwinCritic(LangConfig.from_dict(cfg.to_dict())).eval()
        back.load_state_dict(cr.state_dict())
        b1, b2, _, _ = back(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
    assert torch.equal(q1, b1) and torch.equal(q2, b2)
    # the two readouts' state dicts never load into each other (a mismatch fails loudly, never silently)
    torch.manual_seed(0)
    mlp = LangTwinCritic(dataclasses.replace(cfg, critic_readout="mlp"))
    with pytest.raises(RuntimeError):
        mlp.load_state_dict(cr.state_dict())
    with pytest.raises(RuntimeError):
        cr.load_state_dict(mlp.state_dict())


def test_mlp_state_dict_round_trip():
    g = torch.Generator().manual_seed(10)
    cfg, _, cr = build()
    enc, act = head_inputs(g)
    torch.manual_seed(1)
    back = LangTwinCritic(LangConfig.from_dict(cfg.to_dict())).eval()
    assert back.c1.readout == "mlp"
    back.load_state_dict(cr.state_dict())
    with torch.no_grad():
        a = cr(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
        b = back(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.skipif(not OLD_CKPT.exists(), reason=f"{OLD_CKPT} not on this machine")
def test_old_checkpoint_loads_strictly_as_softmin():
    """A real checkpoint saved before the field (E2 V0, 2026-10-08): its config reads softmin, the critics load
    strictly, and Q equals the pre-decision-6 forward bit for bit."""
    ck = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    assert "critic_readout" not in ck["lang_config"]
    cfg = LangConfig.from_dict(ck["lang_config"])
    assert cfg.critic_readout == "softmin" and cfg.scene_fusion is False
    cr = LangTwinCritic(cfg).eval()
    cr.load_state_dict(ck["critics"])                    # strict
    g = torch.Generator().manual_seed(11)
    enc, act = head_inputs(g, n_real=(14, 14), K=n_queries(cfg))
    with torch.no_grad():
        q1, q2, _, _ = cr(enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
        r1, _ = softmin_reference(cr.c1, enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
        r2, _ = softmin_reference(cr.c2, enc, act["dtheta"], act["Jlin"], act["Jang"], act["joint_index"])
    assert torch.equal(q1, r1) and torch.equal(q2, r2) and torch.isfinite(q1).all()


# ---------------------------------------------------------------------------
# config: legacy from_dict, collect_hash
# ---------------------------------------------------------------------------


def test_from_dict_legacy_defaults():
    old = LangConfig.from_dict(old_dict())
    assert old.critic_readout == "softmin" and old.scene_fusion is False and old.fused_words is False
    assert (old.fusion_words, old.fusion_dtype, old.critic_mlp_hidden, old.train_episodes_per_task) == \
        (True, "fp32", 512, 0)
    new = LangConfig.from_dict(LangConfig().to_dict())
    assert new.critic_readout == "mlp" and new.scene_fusion is False
    on = LangConfig.from_dict({**LangConfig().to_dict(), "scene_fusion": True, "critic_readout": "softmin"})
    assert on.scene_fusion is True and on.fused_words is True and on.critic_readout == "softmin"
    with pytest.raises(AssertionError):
        dataclasses.replace(LangConfig(), critic_readout="max")


def test_collect_hash_unchanged_by_new_fields():
    """The new fields are training-side only: a config without them, and one with every new field changed, hash
    like LangConfig() (shards being collected keep resuming); lang_version stays in the hash (decision 4)."""
    cfg = LangConfig()
    base = collect_hash(cfg)
    assert set(NEW_FIELDS) <= TRAIN_ONLY_FIELDS and "lang_version" not in TRAIN_ONLY_FIELDS

    class Saved:   # a config whose dict has the new keys removed
        def to_dict(self):
            return old_dict()

    assert collect_hash(Saved()) == base
    assert collect_hash(LangConfig.from_dict(old_dict())) == base
    changed = dataclasses.replace(cfg, scene_fusion=True, fusion_words=False, fusion_dtype="bf16", fusion_chunk=8,
                                  critic_readout="softmin", critic_mlp_hidden=64, train_episodes_per_task=8)
    assert collect_hash(changed) == base
    assert collect_hash(dataclasses.replace(cfg, lang_version=2)) != base
