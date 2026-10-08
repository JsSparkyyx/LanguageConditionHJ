"""Vanilla language-conditioned HJ filter model (VANILLA_PLAN Section 5).

CrossSafe (RobotInjection, GeoCrossAttention, TokenActor, SafetyFilter: all
reused unchanged) plus text. The scene is three frames [t, t-k, 0] of frozen
HoloBrain tokens (3 x 1200 = 3600 tokens, world positions per frame, a learned
time embedding per frame). The sentence enters through five switchable paths
(Section 5.9): scene reads words once (lang_scene), body reads words in every
block (lang_body), FiLM from the sentence summary r0 (lang_film), K readout
queries (lang_queries), the rule token r in self-attention (lang_rtoken).

Width 256, 8 heads, FFN 1024 SiLU, RMSNorm, pre-norm residual sub-layers, no
dropout, batch_first=True everywhere. HoloBrain vision, the robot state
encoder, BERT and text_feat_map are frozen OUTSIDE this module (the buffer
stores their outputs; the text store holds the word tokens).

Probe mode (``set_probe(model, True)``): blocks keep the inputs and attention
weights metrics.py needs (Section 8.2). Off by default: ``need_weights=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from main.network.encoder import RobotInjection
from main.network.trunk import GeoCrossAttention

DIM = 256
HEADS = 8
FFN_DIM = 1024
N_FRAMES = 3          # [t, t-k, 0]
N_TOK_FRAME = 1200    # 3 cameras x 400 tokens, flattened within a frame
TEXT_DIMS = {"holobrain_bert": 256, "bert_raw": 768, "sentence_generic": 768}


def ffn(dim: int = DIM, hidden: int = FFN_DIM) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))


def mlp_head(dim: int = DIM) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))


def n_queries(cfg) -> int:
    return int(cfg.n_queries) if cfg.lang_queries else 0


def _mha() -> nn.MultiheadAttention:
    return nn.MultiheadAttention(DIM, HEADS, batch_first=True)


def _attend(mha, q, kv, pad, probe_store=None, key=None):
    """mha(q, kv, kv) with key padding; weights only in probe mode (per head too when the store asks: "_heads")."""
    if probe_store is None:
        return mha(q, kv, kv, key_padding_mask=pad, need_weights=False)[0]
    heads = bool(probe_store.get("_heads"))
    out, w = mha(q, kv, kv, key_padding_mask=pad, need_weights=True, average_attn_weights=not heads)
    if heads:   # [B, H, Nq, L]: per-head entropies (review round 3, B4)
        probe_store[key + "_heads"] = w.detach()
        w = w.mean(1)
    probe_store[key] = w.detach()
    return out


# ---------------------------------------------------------------------------
# text adapter (5.3)
# ---------------------------------------------------------------------------


class TextBlock(nn.Module):
    """Pre-norm self-attention block (NOT nn.TransformerEncoderLayer: its
    defaults are post-norm, dropout 0.1, ReLU, FFN 2048, batch_first=False)."""

    def __init__(self):
        super().__init__()
        self.norm1, self.attn = nn.RMSNorm(DIM), _mha()
        self.norm2, self.ffn = nn.RMSNorm(DIM), ffn()

    def forward(self, x, pad):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class TextAdapter(nn.Module):
    """[r, q_1..q_K, words + position] through 2 pre-norm blocks.

    in : words [B, L, d], word_mask [B, L] (True = real token)
    out: r0 [B, 256] (sentence summary), q [B, K, 256] (readout queries),
         w' [B, L, 256] (adapted words; replaces the frozen words downstream)
    """

    def __init__(self, cfg, d_in: int = 256, depth: int = 2):
        super().__init__()
        self.K = n_queries(cfg)
        L = int(cfg.max_text_tokens)
        self.r = nn.Parameter(torch.randn(1, DIM) * 0.02)
        self.q = nn.Parameter(torch.randn(self.K, DIM) * 0.02) if self.K else None
        self.pos_emb = nn.Parameter(torch.randn(L, DIM) * 0.02)
        backend = getattr(cfg, "text_backend", "holobrain_bert")
        self.in_proj = nn.Identity() if (backend == "holobrain_bert" and d_in == DIM) else nn.Linear(d_in, DIM)
        self.blocks = nn.ModuleList([TextBlock() for _ in range(depth)])
        if getattr(cfg, "text_out_norm", False):   # V1 (round 3 E2): the outputs enter the model normalized
            self.norm_r0, self.norm_w = nn.RMSNorm(DIM), nn.RMSNorm(DIM)
            self.norm_q = nn.RMSNorm(DIM) if self.K else None

    def forward(self, words, word_mask):
        B, L, _ = words.shape
        w = self.in_proj(words.to(self.pos_emb.dtype)) + self.pos_emb[:L]
        parts = [self.r.expand(B, -1, -1)]
        if self.K:
            parts.append(self.q.expand(B, -1, -1))
        x = torch.cat(parts + [w], dim=1)                                   # [B, 1+K+L, 256]
        pad = torch.cat([torch.zeros(B, 1 + self.K, dtype=torch.bool, device=x.device), ~word_mask.bool()], 1)
        for blk in self.blocks:
            x = blk(x, pad)
        r0, q, w = x[:, 0], x[:, 1:1 + self.K], x[:, 1 + self.K:]
        if hasattr(self, "norm_r0"):
            r0, w = self.norm_r0(r0), self.norm_w(w)
            q = self.norm_q(q) if self.norm_q is not None else q
        return r0, q, w


def drop_special(wm: torch.Tensor) -> torch.Tensor:
    """V2 (round 3 E2): the word mask without each sentence's first and last real token ([CLS], [SEP]), for the
    word reads (body / scene-reads-words); a sentence left with no token keeps its mask."""
    n = wm.sum(-1)
    idx = torch.arange(wm.shape[1], device=wm.device)[None]
    out = wm & (idx != 0) & (idx != (n - 1)[:, None])
    return torch.where(out.any(-1, keepdim=True), out, wm)


def oracle_tag_weights(scene_pos: torch.Tensor, center: torch.Tensor, valid: torch.Tensor,
                       sigma: float) -> torch.Tensor:
    """V3 (round 3 E2): w_j = exp(-|p_j - c_f|^2 / (2 sigma^2)) for every scene token j of frame f, 0 in frames
    where the named object's centre c_f is unknown. scene_pos [B, F * n, 3], center [B, F, 3], valid [B, F]."""
    B, Fn = scene_pos.shape[:2]
    F = center.shape[1]
    p = scene_pos.view(B, F, Fn // F, 3)
    d2 = ((p - center[:, :, None, :]) ** 2).sum(-1)
    return (torch.exp(-d2 / (2.0 * sigma ** 2)) * valid[:, :, None].to(p.dtype)).reshape(B, Fn)


# ---------------------------------------------------------------------------
# scene set + scene reads words (5.4)
# ---------------------------------------------------------------------------


class SceneSet(nn.Module):
    """Frames [t, t-k, 0] -> one 3600-token set: CrossSafe's scene adapter
    (shared over frames) + a learned time embedding per frame role."""

    def __init__(self, n_tok: int = N_TOK_FRAME):
        super().__init__()
        self.n_tok = int(n_tok)
        self.tau = nn.Parameter(torch.randn(N_FRAMES, DIM) * 0.02)

    def forward(self, scene_tokens, scene_pos, injection: RobotInjection):
        # frames first, never cameras: [B, 3, 1200, 256]
        assert scene_tokens.dim() == 4 and tuple(scene_tokens.shape[1:]) == (N_FRAMES, self.n_tok, DIM), \
            f"scene_tokens must be [B, {N_FRAMES}, {self.n_tok}, {DIM}] (frames, flattened cameras), got {tuple(scene_tokens.shape)}"
        s = injection.scene_adapter(scene_tokens.to(self.tau.dtype)) + self.tau[None, :, None, :]
        return s.flatten(1, 2), scene_pos.to(self.tau.dtype).flatten(1, 2)


class SceneReadsWords(nn.Module):
    """s = s + MHA(RMSNorm(s), w', w'), once; the scene is read-only afterwards."""

    def __init__(self):
        super().__init__()
        self.norm, self.attn = nn.RMSNorm(DIM), _mha()
        self.probe = None

    def forward(self, s, w, wm):
        delta = _attend(self.attn, self.norm(s), w, ~wm, self.probe, "scene_word_attn")
        if self.probe is not None:
            self.probe["scene_delta_norm"] = delta.detach().norm(dim=-1)   # [B, 3600]
        return s + delta


# ---------------------------------------------------------------------------
# trunk block (5.5)
# ---------------------------------------------------------------------------


class LangTrunkBlock(nn.Module):
    """Updated: body b [B, 22, 256] (positions x, body_mask m), rule token r
    [B, 1, 256], queries q [B, K, 256]. Read-only: scene s / p / scene_mask,
    words w' / word_mask, r0.

        a. body reads scene (CrossSafe GeoCrossAttention, unchanged module)
        b. body reads words                        (lang_body)
        c. queries read scene, no geometric bias   (lang_queries)
        d. self-attention over [body, r, queries]
        e. FiLM from r0 (lang_film), then FFN (separate FFNs for b, q, r)
    """

    def __init__(self, cfg, update_r: bool = True):
        super().__init__()
        self.K = n_queries(cfg)
        self.lang_body = bool(cfg.lang_body)
        self.update_r = bool(update_r)
        self.norm_a, self.geo = nn.RMSNorm(DIM), GeoCrossAttention(DIM, HEADS)
        if self.lang_body:
            self.norm_b, self.attn_bw = nn.RMSNorm(DIM), _mha()
        if self.K:
            self.norm_c, self.attn_qs = nn.RMSNorm(DIM), _mha()
        self.norm_d, self.attn_self = nn.RMSNorm(DIM), _mha()
        if cfg.lang_film:
            self.film = nn.Linear(DIM, 2 * DIM)
            if getattr(cfg, "film_zero_init", True):
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
        self.norm_eb, self.ffn_b = nn.RMSNorm(DIM), ffn()
        if self.K:
            self.norm_eq, self.ffn_q = nn.RMSNorm(DIM), ffn()
        if self.update_r:
            self.norm_er, self.ffn_r = nn.RMSNorm(DIM), ffn()
        self.probe = None   # dict in probe mode

    def forward(self, b, x, m, r, q, s, p, smask, w, wm, r0):
        P = self.probe
        y = self.norm_a(b)
        if P is not None:   # metrics.geo_attention_weights recomputes A from these
            P["geo_inputs"] = (y.detach(), x.detach())
        b = b + self.geo(y, x, s, p, smask)
        if self.lang_body:
            b = b + _attend(self.attn_bw, self.norm_b(b), w, ~wm, P, "body_word_attn")
        if self.K:
            q = q + _attend(self.attn_qs, self.norm_c(q), s, ~smask, P, "query_scene_attn")
        nb, B = b.shape[1], b.shape[0]
        z = torch.cat([b, r, q], 1)
        y = self.norm_d(z)
        pad = torch.cat([~m, torch.zeros(B, 1 + q.shape[1], dtype=torch.bool, device=b.device)], 1)
        z = z + self.attn_self(y, y, y, key_padding_mask=pad, need_weights=False)[0]
        b, r, q = z[:, :nb], z[:, nb:nb + 1], z[:, nb + 1:]
        if hasattr(self, "film"):
            gamma, beta = self.film(r0).unsqueeze(1).chunk(2, dim=-1)       # [B, 1, 256] each
            if P is not None:
                P["film_gamma_rms"] = gamma.detach().pow(2).mean().sqrt()
                P["film_beta_rms"] = beta.detach().pow(2).mean().sqrt()
            b = b + self.ffn_b((1 + gamma) * self.norm_eb(b) + beta)
            if self.K:
                q = q + self.ffn_q((1 + gamma) * self.norm_eq(q) + beta)
        else:
            b = b + self.ffn_b(self.norm_eb(b))
            if self.K:
                q = q + self.ffn_q(self.norm_eq(q))
        if self.update_r:
            r = r + self.ffn_r(self.norm_er(r))
        return b, r, q


def _run_blocks(blocks, use_ckpt: bool, b, x, m, r, q, s, p, smask, w, wm, r0):
    for blk in blocks:
        if use_ckpt and torch.is_grad_enabled():
            b, r, q = checkpoint(blk, b, x, m, r, q, s, p, smask, w, wm, r0, use_reentrant=False)
        else:
            b, r, q = blk(b, x, m, r, q, s, p, smask, w, wm, r0)
    return b, r, q


# ---------------------------------------------------------------------------
# policy encoder (5.6)
# ---------------------------------------------------------------------------


@dataclass
class LangEncoded:
    body: torch.Tensor; pos: torch.Tensor; mask: torch.Tensor                  # [B,22,256] [B,22,3] [B,22]
    scene: torch.Tensor; scene_pos: torch.Tensor; scene_mask: torch.Tensor     # [B,3600,256] [B,3600,3] [B,3600]
    words: torch.Tensor; word_mask: torch.Tensor                               # [B,L,256] [B,L]
    r0: torch.Tensor; r: torch.Tensor; queries: torch.Tensor                   # [B,256] [B,1,256] [B,K,256]

    def detach(self) -> "LangEncoded":
        return LangEncoded(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def index(self, idx) -> "LangEncoded":
        return LangEncoded(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})


class LangPolicyEncoder(nn.Module):
    """injection + scene set + text adapter + scene-reads-words + 3 LangTrunkBlocks.

    forward(state_tokens, body_tokens, link_pos, body_mask, joint_index,
            scene_tokens [B, 3, 1200, 256], scene_pos [B, 3, 1200, 3],
            words, word_mask | text_out) -> LangEncoded
    text_out = encode_text(words, word_mask) = (r0, q, w', word_mask): at
    deployment computed once per episode and reused every tick.
    """

    def __init__(self, cfg, n_tok: int = N_TOK_FRAME):
        super().__init__()
        self.cfg = cfg
        self.K = n_queries(cfg)
        self.injection = RobotInjection()
        self.scene_set = SceneSet(n_tok)
        self.text_adapter = TextAdapter(cfg, d_in=TEXT_DIMS[cfg.text_backend])
        if cfg.lang_scene:
            self.scene_reads_words = SceneReadsWords()
        if not cfg.lang_rtoken:
            self.r_static = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)
        self.blocks = nn.ModuleList([LangTrunkBlock(cfg) for _ in range(int(cfg.trunk_depth))])
        self.use_ckpt = bool(getattr(cfg, "activation_checkpointing", False))
        if getattr(cfg, "oracle_ref_tag", False):   # V3: one learned tag vector, zero-init
            self.e_tag = nn.Parameter(torch.zeros(DIM))

    def encode_text(self, words, word_mask):
        word_mask = word_mask.bool()
        r0, q, w = self.text_adapter(words, word_mask)
        return r0, q, w, word_mask

    def forward(self, state_tokens, body_tokens, link_pos, body_mask, joint_index,
                scene_tokens, scene_pos, words=None, word_mask=None, text_out=None,
                scene_mask=None, ref_center=None, ref_valid=None) -> LangEncoded:
        b, x, m = self.injection(state_tokens, body_tokens, link_pos, body_mask, joint_index)
        s, p = self.scene_set(scene_tokens.to(b.device), scene_pos.to(b.device), self.injection)
        B = b.shape[0]
        if hasattr(self, "e_tag"):   # V3: s_j += w_j e_tag before scene-reads-words (the pointer's answer, given)
            if ref_center is None or ref_valid is None:
                raise ValueError("oracle_ref_tag needs ref_center / ref_valid (simulator state; offline only)")
            w_tag = oracle_tag_weights(p, ref_center.to(p.device, p.dtype), ref_valid.to(p.device),
                                       float(self.cfg.oracle_tag_sigma))
            s = s + w_tag[..., None] * self.e_tag
        # every token is valid in training and deployment; scene_mask [B, 3, n_tok] exists for tests
        smask = (torch.ones(s.shape[:2], dtype=torch.bool, device=s.device) if scene_mask is None
                 else scene_mask.to(s.device).bool().flatten(1))
        if text_out is None:
            text_out = self.encode_text(words.to(b.device), word_mask.to(b.device))
        r0, q, w, wm = text_out
        if r0.shape[0] != B:   # one sentence for the whole batch (deployment)
            r0, q, w, wm = (t.expand(B, *t.shape[1:]) for t in (r0, q, w, wm))
        if getattr(self.cfg, "mask_special_tokens", False):   # V2: also the critic's body-reads-words (enc.word_mask)
            wm = drop_special(wm)
        if hasattr(self, "scene_reads_words"):
            s = self.scene_reads_words(s, w, wm)
        r = r0[:, None] if self.cfg.lang_rtoken else self.r_static.expand(B, 1, -1)
        if not self.K:
            q = b.new_zeros(B, 0, DIM)
        b, r, q = _run_blocks(self.blocks, self.use_ckpt, b, x, m, r, q, s, p, smask, w, wm, r0)
        return LangEncoded(b, x, m, s, p, smask, w, wm, r0, r, q)


# ---------------------------------------------------------------------------
# critic (5.8)
# ---------------------------------------------------------------------------


def action_effects(dtheta, Jlin, Jang, joint_index, n_res: int = 2, dtype=None):
    """CriticHead's step 1, copied: per-token Cartesian effects of a joint
    displacement. -> eff [B, M + n_res, 8] = [own dtheta, dp, dw, valid] (zeros
    for the residual tokens), dp_full [B, M + n_res, 3] (post-action offset)."""
    B = dtheta.shape[0]
    dev = dtheta.device
    if joint_index.dim() == 1:
        joint_index = joint_index.unsqueeze(0).expand(B, -1)
    joint_index = joint_index.to(dev).long()
    dtype = dtype or dtheta.dtype
    dtheta = dtheta.to(dtype)
    Jlin, Jang = Jlin.to(dev).to(dtype), Jang.to(dev).to(dtype)
    dp = torch.einsum("bnca,ba->bnc", Jlin, dtheta)                          # [B, M, 3]
    dw = torch.einsum("bnca,ba->bnc", Jang, dtheta)
    valid = joint_index >= 0
    dth = torch.where(valid, dtheta.gather(1, joint_index.clamp_min(0)),
                      torch.zeros_like(joint_index, dtype=dtheta.dtype))     # [B, M]
    eff = torch.cat([dth.unsqueeze(-1), dp, dw, valid.to(dtheta.dtype).unsqueeze(-1)], dim=-1)
    eff = torch.cat([eff, eff.new_zeros(B, n_res, 8)], dim=1)
    dp_full = torch.cat([dp, dp.new_zeros(B, n_res, 3)], dim=1)
    return eff, dp_full


def softmin(V, T: float):
    return -T * torch.logsumexp(-V / T, dim=-1)


class LangCriticHead(nn.Module):
    """Action injection (Linear(8, 256) on the effects), post-action geometry,
    critic_depth LangTrunkBlocks (last: update_r=False), value heads on the 22
    body tokens and the K queries, Q = softmin_T over them (T in h units)."""

    def __init__(self, cfg):
        super().__init__()
        self.K = n_queries(cfg)
        self.temperature = float(cfg.softmin_T_h)    # plain attribute, NOT scaled by h_scale
        self.act_proj = nn.Linear(8, DIM)
        depth = int(cfg.critic_depth)
        self.blocks = nn.ModuleList([LangTrunkBlock(cfg, update_r=(i < depth - 1)) for i in range(depth)])
        self.value_head = mlp_head()
        if self.K:
            self.value_head_q = mlp_head()
        self.use_ckpt = bool(getattr(cfg, "activation_checkpointing", False))

    def forward(self, enc: LangEncoded, dtheta, Jlin, Jang, joint_index):
        M = joint_index.shape[-1]
        n_res = enc.body.shape[1] - M
        eff, dp = action_effects(dtheta.to(enc.body.device), Jlin, Jang, joint_index, n_res, enc.body.dtype)
        b = enc.body + self.act_proj(eff)
        x = enc.pos + dp
        b, r, q = _run_blocks(self.blocks, self.use_ckpt, b, x, enc.mask, enc.r, enc.queries, enc.scene,
                              enc.scene_pos, enc.scene_mask, enc.words, enc.word_mask, enc.r0)
        V = self.value_head(b).squeeze(-1).masked_fill(~enc.mask, 1e4)       # padding must lose the min
        if self.K:
            V = torch.cat([V, self.value_head_q(q).squeeze(-1)], dim=-1)
        return softmin(V, self.temperature), V


class LangTwinCritic(nn.Module):
    """Two LangCriticHeads on the same LangEncoded (TwinCritic's interface:
    forward -> (q1, q2, v1, v2), qmin) so SafetyFilter is reused unchanged."""

    def __init__(self, cfg):
        super().__init__()
        self.c1, self.c2 = LangCriticHead(cfg), LangCriticHead(cfg)

    def forward(self, enc, dtheta, Jlin, Jang, joint_index):
        q1, v1 = self.c1(enc, dtheta, Jlin, Jang, joint_index)
        q2, v2 = self.c2(enc, dtheta, Jlin, Jang, joint_index)
        return q1, q2, v1, v2

    def qmin(self, enc, dtheta, Jlin, Jang, joint_index):
        q1, q2, _, _ = self.forward(enc, dtheta, Jlin, Jang, joint_index)
        return torch.minimum(q1, q2)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def set_probe(module: nn.Module, on: bool, heads: bool = False) -> list[dict]:
    """Probe mode on every LangTrunkBlock / SceneReadsWords inside module;
    returns their probe dicts in module order (empty when off). heads: also keep per-head weights."""
    stores = []
    for mod in module.modules():
        if isinstance(mod, (LangTrunkBlock, SceneReadsWords)):
            mod.probe = ({"_heads": True} if heads else {}) if on else None
            if on:
                stores.append(mod.probe)
    return stores


def param_groups(policy_enc: LangPolicyEncoder) -> dict:
    """Parameter groups for logging / gradient norms (Section 8.1)."""
    groups = {"injection": list(policy_enc.injection.parameters()),
              "scene_set": list(policy_enc.scene_set.parameters()),
              "text_adapter": list(policy_enc.text_adapter.parameters()),
              "trunk": list(policy_enc.blocks.parameters())}
    if hasattr(policy_enc, "scene_reads_words"):
        groups["scene_reads_words"] = list(policy_enc.scene_reads_words.parameters())
    if hasattr(policy_enc, "r_static"):
        groups["r_static"] = [policy_enc.r_static]
    return groups


def n_params(params) -> int:
    return int(sum(p.numel() for p in params))
