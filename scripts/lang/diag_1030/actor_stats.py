#!/usr/bin/env python
"""diag_1030 check 1: the actor's statistics and the critic's action response on stored states.

Read-only on the run and its data. For a set of states (T5 tray_transport first: planner ticks of the drag split
into "tips within 1 s" / "tips later" / "never tips", and the actor-driven ticks of the run's T5 filter episodes;
then planner and actor ticks of the other tasks), for every checkpoint given:

  (a) the actor: pre-tanh mu, sigma = exp(clamped log_std), |tanh(mu)| = |a_det| / dtheta_max, the share of joints
      at the sigma cap, the tanh-Gaussian entropy (Gauss-Hermite) summed over the valid joints and the Monte Carlo
      -log pi the training log reports, ||a_det|| vs ||a_ref|| and their cosine (a_ref = the stored command);
  (b) the critic's action response: Q(s, c a_ref) for c in C_LIST, Q(s, a_det), Q over 16 uniform actions in the
      box, the logged critic/action_sens_rel recomputed, ||dQ/da * dtheta_max||, the same Q in TF32 (the training
      step's precision) for the noise floor;
  (c) where the action signal dies: ||act_proj(eff(a)) - act_proj(eff(0))|| vs ||body token||, the relative change
      of the critic's hidden state after each block and of the per-token values V between a = 0 and saturation;
  (d) the critic target with four continuations at s': the stochastic actor (as trained), its mean, the stored next
      command and a' = 0.

Plus numpy-only motion statistics of the run's T5 episodes (what the actor commanded, what the robot did, the
tray's deceleration when the filter takes over).

    scripts/lang/in_container.sh python scripts/lang/diag_1030/actor_stats.py \\
        --run runs/lang/rounds_ur5_g095_1023 --data data/lang_1023/warmup \\
        --ckpts checkpoint_round10.pt checkpoint_round20.pt checkpoint_round30.pt checkpoint.pt \\
        --out runs/lang/diag_1030/actor_stats/vanilla
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from main.lang import ensure_item_path  # noqa: E402

ensure_item_path()

from main.lang.config import CANON_TASKS, LangConfig  # noqa: E402
from main.lang.network import LangPolicyEncoder, LangTwinCritic, action_effects, make_actor, softmin  # noqa: E402
from main.lang.train import LangTrainer, caches_of, load_policy, matmul_tf32, roots_of  # noqa: E402

C_LIST = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
N_RAND = 16
N_STOCH = 8
CTRL_DT = 0.04
T5 = "tray_transport"
FLOAT_KEYS = ("state_tokens", "body_tokens", "link_pos", "Jlin", "Jang", "dtheta_max", "scene_tokens", "scene_pos")

GH_Z, GH_W = np.polynomial.hermite_e.hermegauss(96)
GH_W = GH_W / GH_W.sum()


def qstat(x) -> dict:
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}


def tanh_gauss_entropy(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Differential entropy (nats) of u = tanh(x), x ~ N(mu, sigma^2): H(x) + E[log(1 - tanh(x)^2)]."""
    z = torch.as_tensor(GH_Z, dtype=mu.dtype, device=mu.device)
    w = torch.as_tensor(GH_W, dtype=mu.dtype, device=mu.device)
    x = mu[..., None] + sigma[..., None] * z
    log_jac = 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    return 0.5 * torch.log(2 * math.pi * math.e * sigma ** 2) + (log_jac * w).sum(-1)


# ---------------------------------------------------------------------------------------------------- models --
class ModelSet:
    def __init__(self, label: str, path: Path, cfg: LangConfig, n_tok: int, state_path: Path | None = None):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        if ck.get("switches") != cfg.path_switches():
            raise SystemExit(f"{path}: switches {ck.get('switches')} != {cfg.path_switches()}")
        self.label, self.path = label, str(path)
        self.step, self.round = int(ck.get("step", -1)), int(ck.get("round", -1))
        self.pe = LangPolicyEncoder(cfg, n_tok=n_tok).cuda().eval()
        load_policy(self.pe, ck)
        self.actor = make_actor(cfg).cuda().eval()
        self.actor.load_state_dict(ck["actor"])
        self.cr = LangTwinCritic(cfg).cuda().eval()
        self.cr.load_state_dict(ck["critics"])
        self.pe_t, self.cr_t = copy.deepcopy(self.pe), copy.deepcopy(self.cr)
        self.target_src = "live weights (no target copy in the checkpoint)"
        if state_path is not None and state_path.exists():
            st = torch.load(state_path, map_location="cuda", weights_only=False)
            if int(st["step"]) == self.step:
                self.pe_t.load_state_dict(st["policy_enc_targ"])
                self.cr_t.load_state_dict(st["critics_targ"])
                self.target_src = f"state.pt target copies (step {int(st['step'])})"
            else:
                self.target_src += f"; state.pt step {int(st['step'])} != checkpoint step {self.step}, not used"
            del st
        for mod in (self.pe, self.actor, self.cr, self.pe_t, self.cr_t):
            for p in mod.parameters():
                p.requires_grad_(False)
        del ck


class FreshModelSet(ModelSet):
    """A freshly initialized model (the run's config, random weights, seed fixed): the action response at step 0,
    before any training (is the critic action-blind from initialization, and are the body tokens already large?)."""

    def __init__(self, cfg: LangConfig, n_tok: int, seed: int = 0):
        torch.manual_seed(seed)
        self.label, self.path = "fresh_init", "(random init)"
        self.step, self.round = 0, -1
        self.pe = LangPolicyEncoder(cfg, n_tok=n_tok).cuda().eval()
        self.actor = make_actor(cfg).cuda().eval()
        self.cr = LangTwinCritic(cfg).cuda().eval()
        self.pe_t, self.cr_t = copy.deepcopy(self.pe), copy.deepcopy(self.cr)
        self.target_src = "fresh weights (target = live copy, as at step 0)"
        for mod in (self.pe, self.actor, self.cr, self.pe_t, self.cr_t):
            for p in mod.parameters():
                p.requires_grad_(False)


def upload(b: dict) -> dict:
    out = dict(b)
    for suf in ("", "_next"):
        for k in FLOAT_KEYS:
            out[k + suf] = torch.from_numpy(np.ascontiguousarray(b[k + suf])).cuda().float()
        out["body_mask" + suf] = torch.from_numpy(np.ascontiguousarray(b["body_mask" + suf])).cuda().bool()
        out["joint_index" + suf] = torch.from_numpy(np.ascontiguousarray(b["joint_index" + suf])).cuda().long()
    out["dtheta"] = torch.from_numpy(np.ascontiguousarray(b["dtheta"])).cuda().float()
    out["dtheta_next_cmd"] = torch.from_numpy(np.ascontiguousarray(b["dtheta_next_cmd"])).cuda().float()
    out["h"] = torch.from_numpy(np.asarray(b["h"], np.float32)).cuda()
    return out


def encode(pe, b, words, wmask, suffix=""):
    g = lambda k: b[k + suffix]  # noqa: E731
    enc = pe(g("state_tokens"), g("body_tokens"), g("link_pos"), g("body_mask"), g("joint_index"),
             g("scene_tokens"), g("scene_pos"), words=words, word_mask=wmask)
    return enc, g("Jlin"), g("Jang"), g("joint_index")


def encoder_norms(pe, b, words, wmask) -> dict:
    """LangPolicyEncoder.forward (no V3 tag; V2 special-token mask honoured), recording the mean body-token norm
    after the RobotInjection and after each trunk block, and the query-token norm: where the residual stream grows."""
    from main.lang.network import drop_special

    g = lambda k: b[k]  # noqa: E731
    bt, x, m = pe.injection(g("state_tokens"), g("body_tokens"), g("link_pos"), g("body_mask"), g("joint_index"))
    s, p = pe.scene_set(g("scene_tokens"), g("scene_pos"), pe.injection)
    B = bt.shape[0]
    smask = torch.ones(s.shape[:2], dtype=torch.bool, device=s.device)
    r0, q, w, wm = pe.encode_text(words, wmask)
    if getattr(pe.cfg, "mask_special_tokens", False):
        wm = drop_special(wm)
    if hasattr(pe, "scene_reads_words"):
        s = pe.scene_reads_words(s, w, wm)
    r = r0[:, None] if pe.cfg.lang_rtoken else pe.r_static.expand(B, 1, -1)
    mf = m.float()
    nrm = lambda t: ((t.norm(dim=-1) * mf).sum(-1) / mf.sum(-1)).cpu().numpy()  # noqa: E731
    out = {"enc_body_norm_after_injection": nrm(bt), "enc_scene_norm": s.norm(dim=-1).mean(-1).cpu().numpy()}
    for k, blk in enumerate(pe.blocks):
        bt, r, q = blk(bt, x, m, r, q, s, p, smask, w, wm, r0)
        out[f"enc_body_norm_after_block{k}"] = nrm(bt)
        if q.shape[1]:
            out[f"enc_query_norm_after_block{k}"] = q.norm(dim=-1).mean(-1).cpu().numpy()
    return out


def head_trace(head, enc, dtheta, Jlin, Jang, ji) -> dict:
    """LangCriticHead.forward (softmin readout) with its intermediates."""
    M = ji.shape[-1]
    n_res = enc.body.shape[1] - M
    eff, dp = action_effects(dtheta, Jlin, Jang, ji, n_res, enc.body.dtype)
    inj = head.act_proj(eff)
    b = enc.body + inj
    x = enc.pos + dp
    r, q = enc.r, enc.queries
    hb, hq = [b], [q]
    for blk in head.blocks:
        b, r, q = blk(b, x, enc.mask, r, q, enc.scene, enc.scene_pos, enc.scene_mask, enc.words, enc.word_mask, enc.r0)
        hb.append(b)
        hq.append(q)
    V = head.value_head(b).squeeze(-1).masked_fill(~enc.mask, 1e4)
    if head.K:
        V = torch.cat([V, head.value_head_q(q).squeeze(-1)], dim=-1)
    return {"eff": eff, "inj": inj, "hb": hb, "hq": hq, "V": V, "Q": softmin(V, head.temperature)}


def out_t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), device="cuda", dtype=torch.float32)


def rel_tok(a, b, mask=None):
    """Per-sample mean over (valid) tokens of ||a - b|| / ||b||."""
    num = (a - b).norm(dim=-1)
    den = b.norm(dim=-1).clamp_min(1e-12)
    r = num / den
    if mask is None:
        return r.mean(-1)
    m = mask.float()
    return (r * m).sum(-1) / m.sum(-1).clamp_min(1)


# --------------------------------------------------------------------------------------------- per batch ----
@torch.no_grad()
def batch_eval(m: ModelSet, b: dict, words, wmask, words_n, wmask_n, gamma: float, seed: int) -> dict:
    out = {}
    enc, Jlin, Jang, ji = encode(m.pe, b, words, wmask)
    B, M = ji.shape
    dmax, a_ref = b["dtheta_max"], b["dtheta"]
    valid = ji >= 0                                                   # [B, M] joint tokens with an action column
    cols = torch.zeros_like(dmax, dtype=torch.bool)
    cols[torch.arange(B, device=ji.device)[:, None].expand(-1, M)[valid], ji[valid]] = True   # actor columns
    nv = int(valid.sum(1)[0])
    assert bool((valid.sum(1) == nv).all()), "mixed joint counts in one batch"

    # ---- (a) actor -------------------------------------------------------------------------------------------
    head_out = m.actor.head(enc.body[:, :M])
    mu, ls_raw = head_out.unbind(-1)
    ls = ls_raw.clamp(-5, 0)
    sig = ls.exp()
    mu_v, ls_raw_v, sig_v = mu[valid].view(B, nv), ls_raw[valid].view(B, nv), sig[valid].view(B, nv)
    out["mu"] = mu_v.cpu().numpy()
    out["log_std_raw"] = ls_raw_v.cpu().numpy()
    out["sigma"] = sig_v.cpu().numpy()
    out["tanh_mu_abs"] = torch.tanh(mu_v).abs().cpu().numpy()
    out["H_analytic"] = tanh_gauss_entropy(mu_v, sig_v).sum(-1).cpu().numpy()
    hmc = []
    for _ in range(4):
        _, logp, _, _ = m.actor(enc.body, ji, dmax)
        hmc.append(-logp)
    out["H_mc"] = torch.stack(hmc).mean(0).cpu().numpy()
    a_det, _, _, _ = m.actor(enc.body, ji, dmax, deterministic=True)
    cm = cols.float()
    nd = (a_det * cm).norm(dim=1)
    nr = (a_ref * cm).norm(dim=1)
    out["norm_adet"], out["norm_aref"] = nd.cpu().numpy(), nr.cpu().numpy()
    out["norm_dmax"] = (dmax * cm).norm(dim=1).cpu().numpy()
    out["cos_adet_aref"] = ((a_det * a_ref * cm).sum(1) / (nd * nr).clamp_min(1e-12)).cpu().numpy()
    out["aref_over_dmax"] = (a_ref.abs() / dmax.clamp_min(1e-9))[cols].view(B, nv).cpu().numpy()

    # ---- (b) critic's action response -------------------------------------------------------------------------
    qmin = lambda a: m.cr.qmin(enc, a, Jlin, Jang, ji)  # noqa: E731
    box = lambda a: torch.maximum(torch.minimum(a, dmax), -dmax)  # noqa: E731
    qc = torch.stack([qmin(box(c * a_ref)) for c in C_LIST], 1)                      # [B, len(C)]
    out["Q_c"] = qc.cpu().numpy()
    out["clip_frac_c15"] = ((1.5 * a_ref).abs() > dmax + 1e-9)[cols].view(B, nv).float().mean(1).cpu().numpy()
    q_stored = qmin(a_ref)
    out["Q_stored"] = q_stored.cpu().numpy()
    out["Q_det"] = qmin(a_det).cpu().numpy()
    a_sat = torch.where(cols, dmax * torch.where(a_ref >= 0, 1.0, -1.0), torch.zeros_like(dmax))
    out["Q_sat"] = qmin(a_sat).cpu().numpy()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qr = []
    for _ in range(N_RAND):
        u = torch.rand(dmax.shape, generator=gen, device="cuda") * 2 - 1
        qr.append(qmin(u * dmax * cm))
    qr = torch.stack(qr, 1)
    out["Q_rand"] = qr.cpu().numpy()
    # the logged metric, exactly (_diagnostics: every column with dtheta_max, critic outputs at the stored action)
    r1 = (torch.rand(dmax.shape, generator=gen, device="cuda") * 2 - 1) * dmax
    r2 = (torch.rand(dmax.shape, generator=gen, device="cuda") * 2 - 1) * dmax
    out["sens_abs_logged_way"] = (qmin(r1) - qmin(r2)).abs().cpu().numpy()
    q1s, q2s, _, _ = m.cr(enc, a_ref, Jlin, Jang, ji)
    out["absQ_stored_minpair"] = torch.minimum(q1s, q2s).abs().cpu().numpy()
    # TF32, the training step's matmul precision: the same actions, the noise floor
    with matmul_tf32(True):
        qc32 = torch.stack([qmin(box(c * a_ref)) for c in C_LIST], 1)
        out["Q_c_tf32"] = qc32.cpu().numpy()
        r1b = (torch.rand(dmax.shape, generator=gen, device="cuda") * 2 - 1) * dmax
        r2b = (torch.rand(dmax.shape, generator=gen, device="cuda") * 2 - 1) * dmax
        out["sens_abs_logged_way_tf32"] = (qmin(r1b) - qmin(r2b)).abs().cpu().numpy()
    # the best of the scored candidates vs the actor's own action: does a_det sit where Q is highest?
    cand = torch.cat([qc, qr, out_t(out["Q_sat"])[:, None]], 1)
    out["Q_best_cand"] = cand.max(1).values.cpu().numpy()
    out["Q_best_minus_Q_det"] = out["Q_best_cand"] - out["Q_det"]
    # gradient of Q with respect to the action, at a_ref, at 0 and at a_det, scaled by the box half-width
    with torch.enable_grad():
        for name, a0 in (("aref", a_ref), ("zero", torch.zeros_like(a_ref)), ("adet", a_det)):
            a = a0.clone().requires_grad_(True)
            g = torch.autograd.grad(m.cr.qmin(enc, a, Jlin, Jang, ji).sum(), a)[0]
            out[f"gradQ_dmax_{name}"] = (g * dmax * cm).norm(dim=1).cpu().numpy()
            out[f"gradQ_dmax_{name}_l1"] = (g * dmax * cm).abs().sum(1).cpu().numpy()

    # ---- (c) where the action signal dies (critic twin 1 and 2) ----------------------------------------------
    for k, v in encoder_norms(m.pe, b, words, wmask).items():
        out["c1_" + k] = v
    if getattr(m.cr.c1, "readout", "softmin") == "softmin":
        bm = enc.mask
        for hn, head in (("c1", m.cr.c1), ("c2", m.cr.c2)):
            tr0 = head_trace(head, enc, torch.zeros_like(a_ref), Jlin, Jang, ji)
            if hn == "c1":
                q_chk = head(enc, a_ref, Jlin, Jang, ji)[0]
                tr_ref = head_trace(head, enc, a_ref, Jlin, Jang, ji)
                out["trace_check_maxdiff"] = (q_chk - tr_ref["Q"]).abs().cpu().numpy()
            mf = bm.float()
            bnorm = enc.body.norm(dim=-1).clamp_min(1e-12)
            if hn == "c1":
                # raw ||act_proj(eff(a))|| (bias and the valid-flag column included) next to ||body token||
                out["c1_inj_norm_zero"] = ((tr0["inj"].norm(dim=-1) * mf).sum(-1) / mf.sum(-1)).cpu().numpy()
            for an, a in (("aref", a_ref), ("sat", a_sat)):
                tra = head_trace(head, enc, a, Jlin, Jang, ji)
                dinj = (tra["inj"] - tr0["inj"]).norm(dim=-1)
                if hn == "c1":
                    out[f"c1_inj_norm_{an}"] = ((tra["inj"].norm(dim=-1) * mf).sum(-1) / mf.sum(-1)).cpu().numpy()
                    # per token index [B, 22] (padding NaN): which link carries the action signal
                    out[f"c1_{an}_inj_delta_over_body_tok"] = (dinj / bnorm).masked_fill(~bm, float("nan")).cpu().numpy()
                out[f"{hn}_{an}_inj_delta_over_body"] = ((dinj / bnorm * mf).sum(-1) / mf.sum(-1)).cpu().numpy()
                out[f"{hn}_{an}_inj_delta_over_body_max"] = (dinj / bnorm).masked_fill(~bm, 0).amax(-1).cpu().numpy()
                out[f"{hn}_{an}_inj_delta_abs"] = ((dinj * mf).sum(-1) / mf.sum(-1)).cpu().numpy()
                for k in range(len(tra["hb"])):
                    out[f"{hn}_{an}_hb{k}_rel"] = rel_tok(tra["hb"][k], tr0["hb"][k], bm).cpu().numpy()
                    if tra["hq"][k].shape[1]:
                        out[f"{hn}_{an}_hq{k}_rel"] = rel_tok(tra["hq"][k], tr0["hq"][k]).cpu().numpy()
                Vm = torch.cat([bm, torch.ones(B, tr0["V"].shape[1] - bm.shape[1], dtype=torch.bool,
                                               device=bm.device)], 1)
                dV = (tra["V"] - tr0["V"]).abs().masked_fill(~Vm, 0)
                out[f"{hn}_{an}_dV_max"] = dV.amax(-1).cpu().numpy()
                out[f"{hn}_{an}_dV_rel_mean"] = ((dV / tr0["V"].abs().clamp_min(1e-6)).masked_fill(~Vm, 0).sum(-1)
                                                 / Vm.sum(-1)).cpu().numpy()
                out[f"{hn}_{an}_dQ"] = (tra["Q"] - tr0["Q"]).abs().cpu().numpy()
            if hn == "c1":
                # how far above the softmin the body (link) values sit: link-carried action effects matter only near it
                nb = bm.shape[1]
                out["c1_V_body_min_minus_Q"] = (tr0["V"][:, :nb].min(-1).values - tr0["Q"]).cpu().numpy()
                if tr0["V"].shape[1] > nb:
                    out["c1_V_query_min_minus_Q"] = (tr0["V"][:, nb:].min(-1).values - tr0["Q"]).cpu().numpy()
                out["c1_argmin_token"] = tr0["V"].argmin(-1).cpu().numpy()
                out["c1_n_body_tokens"] = np.full(B, bm.shape[1])
                out["c1_body_norm_mean"] = ((enc.body.norm(dim=-1) * bm.float()).sum(-1) / bm.float().sum(-1)).cpu().numpy()
                out["c1_body_norm_tok"] = enc.body.norm(dim=-1).masked_fill(~bm, float("nan")).cpu().numpy()
                eff_ref = tr_ref["eff"][:, :M]
                vv = valid.float()
                out["eff_dth_abs"] = ((eff_ref[..., 0].abs() * vv).sum(-1) / vv.sum(-1)).cpu().numpy()
                tokm = (ji >= -1).float()   # every joint-row token (dp / dw exist for all 20 rows)
                out["eff_dp_norm_mm"] = (1000 * (eff_ref[..., 1:4].norm(dim=-1) * tokm).sum(-1) / tokm.sum(-1)).cpu().numpy()
                out["eff_dp_norm_mm_max"] = (1000 * eff_ref[..., 1:4].norm(dim=-1).amax(-1)).cpu().numpy()
                out["eff_dw_norm"] = ((eff_ref[..., 4:7].norm(dim=-1) * tokm).sum(-1) / tokm.sum(-1)).cpu().numpy()

    # ---- (d) target composition --------------------------------------------------------------------------------
    enc_n, Jlin_n, Jang_n, ji_n = encode(m.pe_t, b, words_n, wmask_n, "_next")
    dmax_n = b["dtheta_max_next"]
    h = b["h"]

    def qbar(a):
        q1, q2, _, _ = m.cr_t(enc_n, a, Jlin_n, Jang_n, ji_n)
        return 0.5 * (q1 + q2)

    stoch = torch.stack([qbar(m.actor(enc_n.body, ji_n, dmax_n)[0]) for _ in range(N_STOCH)], 1)
    a_det_n, _, _, _ = m.actor(enc_n.body, ji_n, dmax_n, deterministic=True)
    qd, qn, qz = qbar(a_det_n), qbar(b["dtheta_next_cmd"]), qbar(torch.zeros_like(a_det_n))
    y = lambda qb: (1 - gamma) * h[..., None] + gamma * torch.minimum(h[..., None], qb)  # noqa: E731
    out["qbar_stoch"], out["qbar_det"] = stoch.cpu().numpy(), qd.cpu().numpy()
    out["qbar_next"], out["qbar_zero"] = qn.cpu().numpy(), qz.cpu().numpy()
    out["y_stoch"] = y(stoch).cpu().numpy()
    out["y_det"], out["y_next"], out["y_zero"] = (y(v[:, None])[:, 0].cpu().numpy() for v in (qd, qn, qz))
    out["h"] = h.cpu().numpy()
    out["norm_adet_next"] = (a_det_n * cm).norm(dim=1).cpu().numpy()
    out["norm_anext_cmd"] = (b["dtheta_next_cmd"] * cm).norm(dim=1).cpu().numpy()
    return out


# --------------------------------------------------------------------------------------------- selection ----
def select_states(tr: LangTrainer, ring_root: Path, rng, caps: dict, smoke: bool) -> dict:
    """{group: [(global index, rule col, info dict)]} over the trainer's sampler (warmup + ring)."""
    smp = tr.sampler
    offs = np.concatenate([[0], np.cumsum([s.rows.size for s in smp.sources])])
    groups: dict[str, list] = {}
    for j, s in enumerate(smp.sources):
        A = s.rs.buf.arrays
        src_all = np.load(s.root / "buffer" / "action_source.npy", mmap_mode="r")
        is_ring = s.root.resolve() == ring_root.resolve()
        slots = s.rows
        src = np.asarray(src_all[slots]).astype(np.int64)
        for task_i in np.unique(s.task):
            task = CANON_TASKS[int(task_i)]
            sel = np.flatnonzero(s.task == task_i)
            if task == T5:
                vel = np.asarray(A["item_vel"])
                h_rules = np.asarray(A["h_rules"])
                fs_all = np.asarray(A["fails_soon"])
                # per episode: the tall and the short block's rule columns, future-min of the tall block's h
                eids = np.unique(s.eid[sel])
                for e in eids:
                    rec = s.rs.records[int(e)]
                    tall = [o["item_id"] for o in rec["objects"] if o.get("size") == "tall"]
                    short = [o["item_id"] for o in rec["objects"] if o.get("size") == "short"]
                    col = {r["rule_id"]: r["idx"] for r in rec["rules"]}
                    if not tall or not short:
                        continue
                    ct, cs = col.get(f"upright|{tall[0]}|-"), col.get(f"upright|{short[0]}|-")
                    if ct is None or cs is None:
                        continue
                    ii = sel[s.eid[sel] == e]
                    ii = ii[np.argsort(s.t[ii])]
                    sl = slots[ii]
                    ht = h_rules[sl, ct]
                    fut = np.minimum.accumulate(ht[::-1])[::-1]
                    first_neg = np.full(ii.size, -1)
                    nxt = -1
                    for k in range(ii.size - 1, -1, -1):
                        if ht[k] < 0:
                            nxt = int(s.t[ii[k]])
                        first_neg[k] = nxt
                    v = np.linalg.norm(vel[sl, 0, :3], axis=1)
                    fs = fs_all[sl, ct]
                    line = s.lines.get(int(e)) or {}
                    filt = bool(line.get("filter", False))
                    named = line.get("named_rule_id")
                    tipped_before = np.concatenate([[False], np.logical_or.accumulate(ht < 0)[:-1]])
                    for k, i in enumerate(ii):
                        info = {"src": int(src[i]), "v_tray": float(v[k]), "h_tall": float(ht[k]), "fs_tall": bool(fs[k]),
                                "fut_min_tall": float(fut[k]), "t": int(s.t[i]), "eid": int(e), "source": j,
                                "ring": is_ring, "filter_ep": filt, "named_tall": bool(named == f"upright|{tall[0]}|-"),
                                "tipped_before": bool(tipped_before[k]),
                                "ticks_to_tip": (first_neg[k] - int(s.t[i])) if first_neg[k] >= 0 else -1}
                        g = None
                        if src[i] == 0 and v[k] > 0.02:
                            if ht[k] >= 0 and fs[k]:
                                g = "t5_plan_soon"
                            elif ht[k] >= 0 and fut[k] < 0:
                                g = "t5_plan_later"
                            elif fut[k] >= 0:
                                g = "t5_plan_safe"
                        elif src[i] == 1 and is_ring:
                            # the actor's ticks before the tall block has tipped (h >= 0 so far) vs after
                            g = "t5_actor_pretip" if (ht[k] >= 0 and not tipped_before[k]) else "t5_actor_posttip"
                        if g is None:
                            continue
                        groups.setdefault(g, []).append((int(offs[j] + i), int(ct), info))
                        groups.setdefault(g + "__short", []).append((int(offs[j] + i), int(cs), info))
            else:
                for i in sel:
                    if src[i] == 0:
                        g = f"plan_{task}"
                    elif src[i] == 1 and is_ring:
                        g = f"actor_{task}"
                    else:
                        continue
                    groups.setdefault(g, []).append((int(offs[j] + i), None,
                                                     {"src": int(src[i]), "t": int(s.t[i]), "eid": int(s.eid[i]),
                                                      "source": j, "ring": is_ring}))
    out = {}
    for g, items in sorted(groups.items()):
        base = g.split("__")[0]
        cap = caps.get(base, caps.get("default", 64))
        if smoke:
            cap = min(cap, 16)
        if len(items) > cap:
            # the __short twin draws the same states as its tall group
            r = np.random.default_rng(zlib.crc32(base.encode()))
            keep = np.sort(r.choice(len(items), cap, replace=False))
            items = [items[k] for k in keep]
        out[g] = items
    # other tasks: the rule as training draws it (harmful / benign from fails_soon, 50 / 50 when both exist)
    for g, items in out.items():
        if items and items[0][1] is None:
            fixed = []
            for gi, _, info in items:
                s = smp.sources[int(smp.sid[gi])]
                col = smp._pick(s, int(smp.loc[gi]), rng)
                info = dict(info, fs_named=bool(np.asarray(s.rs.buf.arrays["fails_soon"][int(s.rows[int(smp.loc[gi])]), col])))
                fixed.append((gi, int(col), info))
            out[g] = fixed
    # training-like draws (task-balanced, 50 % harmful, half paired, LangSampler.plan): the population of the logged
    # critic/action_sens_rel and entropy
    srcs = [np.load(s.root / "buffer" / "action_source.npy", mmap_mode="r") for s in smp.sources]
    mix = []
    for gi, col, p in smp.plan(32 if smoke else int(caps.get("train_mix", 256)), np.random.default_rng(12345)):
        j, i = int(smp.sid[gi]), int(smp.loc[gi])
        s = smp.sources[j]
        mix.append((int(gi), int(col), {"src": int(srcs[j][int(s.rows[i])]), "t": int(s.t[i]), "eid": int(s.eid[i]),
                                         "source": j, "ring": s.root.resolve() == ring_root.resolve(),
                                         "task": CANON_TASKS[int(s.task[i])], "pair": int(p)}))
    out["train_mix"] = mix
    return out


def make_batch(tr: LangTrainer, items, rng) -> dict:
    smp = tr.sampler
    picks = [(gi, col, -1) for gi, col, _ in items]
    b = smp.sample(len(picks), rng=rng, picks=picks)
    # the stored command of row t + 1 (the "next action" continuation of check d)
    nxt = []
    for gi, _, _ in items:
        s = smp.sources[int(smp.sid[gi])]
        nxt.append(np.asarray(s.step["dtheta"][int(s.next[int(smp.loc[gi])])], np.float32))
    b["dtheta_next_cmd"] = np.stack(nxt)
    return b


# ------------------------------------------------------------------------------------------- motion stats ----
def motion_stats(ring: Path, warm_roots) -> dict:
    """numpy only: T5 episodes of the ring (filter and nominal) and of the warmup."""
    out = {}

    def episodes(root):
        L = [json.loads(x) for x in (root / "lang_episodes.jsonl").read_text().splitlines() if x.strip()]
        return [d for d in L if d["task"] == T5 and not d.get("skipped") and d.get("rows")]

    def arrays(root):
        bf, ib = root / "buffer", root / "item_buffer"
        g = lambda p: np.load(p, mmap_mode="r")  # noqa: E731
        return {"src": g(bf / "action_source.npy"), "dtheta": g(bf / "dtheta.npy"), "dmax": g(bf / "dtheta_max.npy"),
                "qraw": g(bf / "qpos_raw.npy"), "link": g(bf / "link_pos.npy"), "bmask": g(bf / "body_mask.npy"),
                "ji": g(bf / "joint_index.npy"), "vel": g(ib / "item_vel.npy"), "h": g(ib / "h_rules.npy")}

    def tall_col(root, eid):
        rec = json.loads((root / "item_buffer" / "episodes" / f"{eid}.json").read_text())
        tall = [o["item_id"] for o in rec["objects"] if o.get("size") == "tall"][0]
        return [r["idx"] for r in rec["rules"] if r["rule_id"] == f"upright|{tall}|-"][0]

    per = {}   # key -> lists
    switches = []
    brakes = []
    col_check = {"num": 0.0, "den": 0.0, "corr": []}
    for root, kind in [(ring, "ring")] + [(r, "warm") for r in warm_roots]:
        eps = episodes(root)
        if not eps:
            continue
        A = arrays(root)
        for d in eps:
            a, bnd = d["rows"]
            if bnd - a < 3:
                continue
            ct = tall_col(root, d["episode_id"])
            src = np.asarray(A["src"][a:bnd]).astype(int)
            dth = np.asarray(A["dtheta"][a:bnd], np.float64)
            dmx = np.asarray(A["dmax"][a:bnd], np.float64)
            qr = np.asarray(A["qraw"][a:bnd], np.float64)
            lp = np.asarray(A["link"][a:bnd], np.float64)
            bm = np.asarray(A["bmask"][a:bnd]).astype(bool)
            ji = np.asarray(A["ji"][a]).astype(int)
            cols = np.unique(ji[ji >= 0])
            v = np.linalg.norm(np.asarray(A["vel"][a:bnd, 0, :3], np.float64), axis=1)
            ht = np.asarray(A["h"][a:bnd, ct], np.float64)
            rnd = int(d.get("round", int(d["episode_id"]) // 1_000_000 - 100))
            filt = bool(d.get("filter"))
            n = bnd - a - 1
            cmd = dth[:n][:, cols]
            real = (qr[1:] - qr[:-1])[:, cols]
            disp = np.linalg.norm(lp[1:] - lp[:-1], axis=-1)
            disp = np.where(bm[:-1] & bm[1:], disp, 0).max(-1)
            ratio_j = np.abs(cmd) / np.maximum(dmx[:n][:, cols], 1e-9)
            for t in range(n):
                key = (kind, "filter" if filt else "nominal", int(src[t]))
                p = per.setdefault(key, {"ratio": [], "cmd_norm": [], "real_norm": [], "disp_mm": [], "v": [],
                                         "rounds": []})
                p["ratio"].append(ratio_j[t])
                p["cmd_norm"].append(np.linalg.norm(cmd[t]))
                p["real_norm"].append(np.linalg.norm(real[t]))
                p["disp_mm"].append(1000 * disp[t])
                p["v"].append(v[t])
                p["rounds"].append(rnd)
            pl = src[:n] == 0
            if pl.sum() > 5:
                col_check["num"] += float(np.linalg.norm(real[pl], axis=1).sum())
                col_check["den"] += float(np.linalg.norm(cmd[pl], axis=1).sum())
                c = np.corrcoef(cmd[pl].ravel(), real[pl].ravel())[0, 1]
                col_check["corr"].append(float(c))
            # the tray's peak deceleration over the episode (planner brake) and the tall block's outcome
            dec = (v[:-1] - v[1:]) / CTRL_DT
            if not filt:
                k = int(np.argmax(dec))
                brakes.append({"root": kind, "eid": int(d["episode_id"]), "peak_decel": float(dec[k]),
                               "v_before": float(v[k]), "tick": k, "tall_tipped": bool((ht < 0).any()),
                               "v_max": float(v.max())})
            # planner -> actor switches while the tray moves
            for t in range(1, n):
                if src[t] == 1 and src[t - 1] != 1 and v[t - 1] > 0.02:
                    w = slice(t - 1, min(t + 4, n))
                    dw = (v[w][:-1] - v[w][1:]) / CTRL_DT
                    switches.append({"root": kind, "eid": int(d["episode_id"]), "round": rnd, "t": t,
                                     "v_prev": float(v[t - 1]), "v_after3": float(v[min(t + 3, n)]),
                                     "peak_decel_next4": float(dw.max()) if dw.size else float("nan"),
                                     "h_tall_at": float(ht[t]), "h_tall_min_next25": float(ht[t:t + 25].min()),
                                     "tall_tipped_later": bool((ht[t:] < 0).any()), "tipped_before": bool((ht[:t] < 0).any()),
                                     "actor_ratio_max": float(ratio_j[t].max()), "actor_ratio_med": float(np.median(ratio_j[t])),
                                     "planner_ratio_med_prev": float(np.median(ratio_j[t - 1]))})
    for key, p in sorted(per.items()):
        name = f"{key[0]}/{key[1]}/src{key[2]}"
        r = np.stack(p["ratio"]) if p["ratio"] else np.zeros((0, 1))
        cn, rn = np.asarray(p["cmd_norm"]), np.asarray(p["real_norm"])
        dm = np.asarray(p["disp_mm"])
        out[name] = {"ticks": int(len(cn)), "abs_cmd_over_dmax_per_joint": qstat(r),
                     "frac_joints_below_0.05": float((r < 0.05).mean()) if r.size else None,
                     "max_joint_ratio_per_tick": qstat(r.max(1)) if r.size else None,
                     "cmd_norm_rad": qstat(cn), "realized_norm_rad": qstat(rn),
                     "realized_over_cmd_summed": float(rn.sum() / max(cn.sum(), 1e-12)),
                     "max_link_disp_mm": qstat(dm), "frac_ticks_link_disp_below_0.5mm": float((dm < 0.5).mean()),
                     "frac_ticks_link_disp_below_1mm": float((dm < 1.0).mean()),
                     "tray_speed_mps": qstat(p["v"])}
        if key[2] == 1:
            rr = np.asarray(p["rounds"])
            byr = {}
            for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40)):
                mk = (rr >= lo) & (rr <= hi)
                if mk.any():
                    byr[f"rounds_{lo}-{hi}"] = {"ticks": int(mk.sum()), "median_joint_ratio": float(np.median(r[mk])),
                                                "p90_joint_ratio": float(np.percentile(r[mk], 90)),
                                                "median_link_disp_mm": float(np.median(dm[mk])),
                                                "frac_link_disp_below_0.5mm": float((dm[mk] < 0.5).mean())}
            out[name]["by_round"] = byr
    out["qpos_raw_vs_dtheta_column_check"] = {"realized_over_cmd_planner_summed": col_check["num"] / max(col_check["den"], 1e-12),
                                              "median_corr_cmd_real_planner": float(np.median(col_check["corr"])) if col_check["corr"] else None}
    out["planner_brakes"] = {"n": len(brakes), "peak_decel_mps2": qstat([x["peak_decel"] for x in brakes]),
                             "peak_decel_tipped": qstat([x["peak_decel"] for x in brakes if x["tall_tipped"]]),
                             "peak_decel_not_tipped": qstat([x["peak_decel"] for x in brakes if not x["tall_tipped"]]),
                             "v_max": qstat([x["v_max"] for x in brakes]), "rows": brakes}
    sw = [x for x in switches if x["root"] == "ring"]
    out["filter_switches_while_moving"] = {
        "n": len(sw), "peak_decel_next4_mps2": qstat([x["peak_decel_next4"] for x in sw]),
        "v_prev": qstat([x["v_prev"] for x in sw]), "v_after3": qstat([x["v_after3"] for x in sw]),
        "actor_ratio_max": qstat([x["actor_ratio_max"] for x in sw]),
        "actor_ratio_med": qstat([x["actor_ratio_med"] for x in sw]),
        "planner_ratio_med_prev": qstat([x["planner_ratio_med_prev"] for x in sw]),
        "first_switch_per_episode": [x for k, x in enumerate(sw) if k == 0 or sw[k - 1]["eid"] != x["eid"]],
    }
    return out


# ------------------------------------------------------------------------------------------------ summary ----
def summarize(acc: dict, gamma: float) -> dict:
    s = {}
    n = len(acc["h"])
    s["n_states"] = int(n)
    s["h_named"] = qstat(acc["h"])
    s["mu"] = qstat(acc["mu"])
    s["abs_mu"] = qstat(np.abs(acc["mu"]))
    s["sigma"] = qstat(acc["sigma"])
    s["log_std_raw"] = qstat(acc["log_std_raw"])
    s["frac_joints_at_sigma_cap"] = float((acc["log_std_raw"] >= -0.01).mean())
    s["abs_tanh_mu_eq_abs_adet_over_dmax"] = qstat(acc["tanh_mu_abs"])
    s["per_joint_abs_tanh_mu_median"] = np.median(acc["tanh_mu_abs"], 0).round(5).tolist()
    s["per_joint_sigma_median"] = np.median(acc["sigma"], 0).round(4).tolist()
    s["per_joint_mu_median"] = np.median(acc["mu"], 0).round(5).tolist()
    s["per_joint_mu_std_across_states"] = np.std(acc["mu"], 0).round(5).tolist()
    s["entropy_analytic_sum"] = qstat(acc["H_analytic"])
    s["entropy_mc"] = qstat(acc["H_mc"])
    s["norm_adet_rad"] = qstat(acc["norm_adet"])
    s["norm_aref_rad"] = qstat(acc["norm_aref"])
    s["norm_dmax_rad"] = qstat(acc["norm_dmax"])
    s["ratio_norm_adet_over_aref"] = qstat(acc["norm_adet"] / np.maximum(acc["norm_aref"], 1e-12))
    s["cos_adet_aref"] = qstat(acc["cos_adet_aref"])
    s["aref_over_dmax_per_joint"] = qstat(acc["aref_over_dmax"])
    s["adet_near_zero_frac"] = float((acc["tanh_mu_abs"].max(1) < 0.05).mean())
    Qc = acc["Q_c"]
    c = np.asarray(C_LIST)
    i0, ih, i1 = list(c).index(0.0), list(c).index(0.5), list(c).index(1.0)
    q0, qh, q1 = Qc[:, i0], Qc[:, ih], Qc[:, i1]
    absq = np.abs(acc["Q_stored"])
    spread_c = Qc.max(1) - Qc.min(1)
    spread_r = acc["Q_rand"].max(1) - acc["Q_rand"].min(1)
    s["Q_stored"] = qstat(acc["Q_stored"])
    s["frac_Q_stored_below_0"] = float((acc["Q_stored"] < 0).mean())
    s["Q_at_c"] = {str(cc): qstat(Qc[:, k]) for k, cc in enumerate(C_LIST)}
    s["spread_over_c_abs"] = qstat(spread_c)
    s["spread_over_c_rel_absQ"] = qstat(spread_c / np.maximum(absq, 1e-9))
    s["spread_16_random_abs"] = qstat(spread_r)
    s["std_16_random_abs"] = qstat(acc["Q_rand"].std(1))
    s["spread_16_random_rel_absQ"] = qstat(spread_r / np.maximum(absq, 1e-9))
    s["Q_det_minus_Q_stored"] = qstat(acc["Q_det"] - acc["Q_stored"])
    s["Q_sat_minus_Q_zero"] = qstat(acc["Q_sat"] - q0)
    s["sens_rel_logged_way_fp32"] = float(acc["sens_abs_logged_way"].mean() / max(acc["absQ_stored_minpair"].mean(), 1e-9))
    s["sens_rel_logged_way_tf32"] = float(acc["sens_abs_logged_way_tf32"].mean() / max(acc["absQ_stored_minpair"].mean(), 1e-9))
    tf_noise = np.abs(acc["Q_c_tf32"] - Qc)
    s["tf32_minus_fp32_abs"] = qstat(tf_noise)
    s["spread_over_c_tf32_abs"] = qstat(acc["Q_c_tf32"].max(1) - acc["Q_c_tf32"].min(1))
    s["gradQ_times_dmax_l2_at_aref"] = qstat(acc["gradQ_dmax_aref"])
    s["gradQ_times_dmax_l1_at_aref"] = qstat(acc["gradQ_dmax_aref_l1"])
    s["gradQ_times_dmax_l2_at_zero"] = qstat(acc["gradQ_dmax_zero"])
    s["ranking"] = {
        "Q0_minus_Qhalf": qstat(q0 - qh), "Qhalf_minus_Qnom": qstat(qh - q1), "Q0_minus_Qnom": qstat(q0 - q1),
        "frac_Q0_gt_Qhalf": float((q0 > qh).mean()), "frac_Qhalf_gt_Qnom": float((qh > q1).mean()),
        "frac_Q0_gt_Qnom": float((q0 > q1).mean()),
        "frac_order_stop_gt_slow_gt_nom": float(((q0 > qh) & (qh > q1)).mean()),
        "frac_order_nom_gt_slow_gt_stop": float(((q1 > qh) & (qh > q0)).mean()),
        "frac_slow_best_of_three": float(((qh > q0) & (qh > q1)).mean()),
        "argmax_c_hist": {str(cc): int((Qc.argmax(1) == k).sum()) for k, cc in enumerate(C_LIST)},
        "frac_abs_Q0_minus_Qhalf_above_tf32_noise_p90": float((np.abs(q0 - qh) > np.percentile(tf_noise, 90)).mean()),
    }
    bd = acc["Q_best_minus_Q_det"]
    s["actor_vs_Q"] = {
        "Q_best_candidate_minus_Q_det": qstat(bd),
        "frac_best_candidate_beats_det_by_tf32_noise_p90": float((bd > np.percentile(tf_noise, 90)).mean()),
        "frac_best_candidate_beats_det_by_0.01": float((bd > 0.01).mean()),
        "frac_best_candidate_beats_det_by_0.001": float((bd > 0.001).mean()),
        "gradQ_times_dmax_l2_at_adet": qstat(acc["gradQ_dmax_adet"]),
    }
    s["frac_log_std_raw_above_0_clamp_grad_dead"] = float((acc["log_std_raw"] > 0).mean())
    for key in [k for k in acc if k.startswith(("c1_", "c2_", "eff_", "trace_"))]:
        if key in ("c1_argmin_token", "c1_n_body_tokens"):
            continue
        s.setdefault("trace", {})[key] = qstat(acc[key])
        if key.endswith("_tok"):
            with np.errstate(all="ignore"):
                s["trace"][key + "_median_per_token"] = [None if not np.isfinite(x) else round(float(x), 6)
                                                         for x in np.nanmedian(acc[key], 0)]
    if "c1_argmin_token" in acc:
        nb = int(acc["c1_n_body_tokens"][0])
        am = acc["c1_argmin_token"]
        s.setdefault("trace", {})["c1_softmin_argmin_is_query_frac"] = float((am >= nb).mean())
        s["trace"]["c1_softmin_argmin_hist"] = {str(int(k)): int(v) for k, v in zip(*np.unique(am, return_counts=True))}
    ys = acc["y_stoch"]
    s["target"] = {
        "y_stoch_mean": qstat(ys.mean(1)), "y_stoch_std_over_8_samples": qstat(ys.std(1)),
        "abs_ystoch_minus_ydet": qstat(np.abs(ys.mean(1) - acc["y_det"])),
        "abs_ystoch_minus_ynext": qstat(np.abs(ys.mean(1) - acc["y_next"])),
        "abs_ystoch_minus_yzero": qstat(np.abs(ys.mean(1) - acc["y_zero"])),
        "abs_ydet_minus_yzero": qstat(np.abs(acc["y_det"] - acc["y_zero"])),
        "abs_ynext_minus_yzero": qstat(np.abs(acc["y_next"] - acc["y_zero"])),
        "qbar_stoch_std_over_8": qstat(acc["qbar_stoch"].std(1)),
        "abs_qbar_stoch_minus_det": qstat(np.abs(acc["qbar_stoch"].mean(1) - acc["qbar_det"])),
        "abs_qbar_next_minus_zero": qstat(np.abs(acc["qbar_next"] - acc["qbar_zero"])),
        "abs_qbar_det_minus_zero": qstat(np.abs(acc["qbar_det"] - acc["qbar_zero"])),
        "frac_h_below_qbar_stoch": float((acc["h"] < acc["qbar_stoch"].mean(1)).mean()),
        "td_y_stoch_minus_Q_stored": qstat(ys.mean(1) - acc["Q_stored"]),
        "norm_adet_next_rad": qstat(acc["norm_adet_next"]), "norm_anext_cmd_rad": qstat(acc["norm_anext_cmd"]),
    }
    return s


def logged_reference(run: Path, step: int, window: int = 2048) -> dict:
    keys = ("entropy", "critic/action_sens_rel", "actor/action_std", "data/action_std", "alpha", "q/mean", "h/mean",
            "grad/actor", "actor/sat_frac")
    rows = []
    with open(run / "train_log.jsonl") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "entropy" in d and step - window < d.get("step", -1) <= step:
                rows.append(d)
    out = {"window_steps": [step - window, step], "n_log_rows": len(rows)}
    for k in keys:
        v = np.asarray([r[k] for r in rows if r.get(k) is not None], float)
        if v.size:
            out[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "last": float(v[-1])}
    return out


# ---------------------------------------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--state", default="state.pt", help="state.pt of the run: target copies for the checkpoint "
                                                        "whose step matches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-motion", action="store_true")
    ap.add_argument("--fresh-init", action="store_true", help="also score a freshly initialized model (step 0)")
    ap.add_argument("--caps", default="", help='JSON overrides of the per-group state caps, e.g. {"default": 32}')
    ap.add_argument("--groups", default="", help="comma-separated group-name prefixes to keep (default: all)")
    ap.add_argument("--drop-groups", default="", help="comma-separated exact group names to drop")
    args = ap.parse_args(argv)
    t0 = time.time()
    run = (REPO / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    out_dir = (REPO / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ring = run / "ring"
    ck_paths = [run / c for c in args.ckpts]
    ck0 = torch.load(ck_paths[-1], map_location="cpu", weights_only=False)
    cfg = LangConfig.from_dict(ck0["lang_config"])
    from main.lang.legacy_pool import checkpoint_vocab

    vocab = checkpoint_vocab(ck0["lang_config"])
    if vocab != cfg.vocab:
        import dataclasses

        cfg = dataclasses.replace(cfg, vocab=vocab, bank_hash=cfg.bank_hash)
    del ck0
    roots = roots_of([str(REPO / d) for d in args.data]) + [ring]
    caches = caches_of([str(REPO / d) for d in args.data])
    if (run / "text_cache_extra.pt").exists():
        caches.append(run / "text_cache_extra.pt")
    print(f"[actor_stats] run {run} encoder {cfg.encoder} roots {len(roots)} caches {[str(c) for c in caches]}", flush=True)
    tr = LangTrainer(cfg, out_dir / "_trainer", roots, (), caches, use_wandb=False, prefetch=False, seed=args.seed)
    n_tok = int(tr.sampler.sources[0].step["scene_tokens"].shape[1])
    models = []
    for p in ck_paths:
        st = run / args.state if (args.state and p.name == "checkpoint.pt") else None
        models.append(ModelSet(p.stem, p, cfg, n_tok, state_path=st))
        print(f"[actor_stats] loaded {p.name}: step {models[-1].step} round {models[-1].round}; target "
              f"{models[-1].target_src}", flush=True)
    if args.fresh_init:
        models.insert(0, FreshModelSet(cfg, n_tok, seed=args.seed))
    del tr.policy_enc, tr.actor, tr.critics, tr.policy_enc_targ, tr.critics_targ
    torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    caps = {"t5_plan_soon": 320, "t5_plan_later": 192, "t5_plan_safe": 320, "t5_actor_pretip": 320,
            "t5_actor_posttip": 128, "default": 128}
    for t in CANON_TASKS:
        caps[f"actor_{t}"] = 64
    if args.caps:
        caps.update(json.loads(args.caps))
    groups = select_states(tr, ring, rng, caps, args.smoke)
    if args.groups:
        keep = tuple(x for x in args.groups.split(",") if x)
        groups = {g: v for g, v in groups.items() if g.startswith(keep)}
    if args.drop_groups:
        drop = set(x for x in args.drop_groups.split(",") if x)
        groups = {g: v for g, v in groups.items() if g not in drop}
    print("[actor_stats] groups: " + ", ".join(f"{g} {len(v)}" for g, v in groups.items()), flush=True)

    gamma = float(cfg.gamma)
    acc = {m.label: {} for m in models}
    infos = {}
    bidx = 0
    for g, items in groups.items():
        infos[g] = [it[2] for it in items]
        for k0 in range(0, len(items), args.batch):
            chunk = items[k0:k0 + args.batch]
            b = make_batch(tr, chunk, rng)
            infos.setdefault(g + "/sentences", []).extend(list(b["sentence"]))
            if cfg.qwen:
                b = tr.qwen_words(b, target=True)
                words, wmask = b["w_qwen"], b["wm_qwen"]
                words_n, wmask_n = b["w_qwen_next"], b["wm_qwen_next"]
            else:
                words, wmask = tr.text.lookup(b["sentence"], splits=["train"] * len(b["sentence"]))
                words_n, wmask_n = words, wmask
            bd = upload(b)
            for m in models:
                o = batch_eval(m, bd, words, wmask, words_n, wmask_n, gamma, seed=1000 * bidx + 7)
                A = acc[m.label].setdefault(g, {})
                for k, v in o.items():
                    A.setdefault(k, []).append(np.asarray(v))
            bidx += 1
            del bd
        print(f"[actor_stats] group {g} done ({len(items)} states) at {time.time() - t0:.0f} s", flush=True)

    results = {"run": str(run), "encoder": cfg.encoder, "gamma": gamma, "C_LIST": list(C_LIST),
               "n_rand": N_RAND, "n_stoch": N_STOCH, "smoke": bool(args.smoke), "checkpoints": {},
               "groups": {g: len(v) for g, v in groups.items()}}
    npz = {}
    for m in models:
        res = {"path": m.path, "step": m.step, "round": m.round, "target_src": m.target_src,
               "logged": logged_reference(run, m.step) if m.step > 0 else logged_reference(run, 100, window=100),
               "groups": {},
               "act_proj_weight_col_norms": {hn: getattr(m.cr, hn).act_proj.weight.norm(dim=0).cpu().numpy().round(4).tolist()
                                             for hn in ("c1", "c2")},
               "act_proj_bias_norm": {hn: float(getattr(m.cr, hn).act_proj.bias.norm()) for hn in ("c1", "c2")}}
        allg = {}
        for g, A in acc[m.label].items():
            cat = {k: np.concatenate(v) for k, v in A.items()}
            res["groups"][g] = summarize(cat, gamma)
            for k, v in cat.items():
                if k in ("mu", "sigma", "log_std_raw", "tanh_mu_abs", "Q_c", "Q_rand", "Q_det", "Q_stored", "Q_sat",
                         "H_analytic", "H_mc", "h", "y_stoch", "y_det", "y_next", "y_zero", "gradQ_dmax_aref",
                         "norm_adet", "norm_aref", "cos_adet_aref", "Q_c_tf32", "Q_best_minus_Q_det",
                         "gradQ_dmax_adet", "gradQ_dmax_zero"):
                    npz[f"{m.label}/{g}/{k}"] = v
            if not g.endswith("__short"):
                for k, v in cat.items():
                    allg.setdefault(k, []).append(v)
        res["all_states_entropy_analytic_mean"] = float(np.concatenate(allg["H_analytic"]).mean())
        res["all_states_entropy_mc_mean"] = float(np.concatenate(allg["H_mc"]).mean())
        res["all_states_frac_joints_at_sigma_cap"] = float((np.concatenate(allg["log_std_raw"]) >= -0.01).mean())
        res["all_states_abs_tanh_mu_median"] = float(np.median(np.concatenate(allg["tanh_mu_abs"])))
        res["all_states_sigma_median"] = float(np.median(np.concatenate(allg["sigma"])))
        for prec in ("", "_tf32"):
            sa = np.concatenate(allg["sens_abs_logged_way" + prec])
            res[f"all_states_sens_rel_logged_way{prec or '_fp32'}"] = float(sa.mean() / max(
                np.concatenate(allg["absQ_stored_minpair"]).mean(), 1e-9))
        results["checkpoints"][m.label] = res
    for g, inf in infos.items():
        if g.endswith("/sentences"):
            continue
        for key in ("ticks_to_tip", "v_tray", "t", "eid", "src", "h_tall", "fs_named", "named_tall", "tipped_before"):
            vals = [x.get(key) for x in inf]
            if all(v is not None for v in vals):
                npz[f"info/{g}/{key}"] = np.asarray(vals)
    results["sentences_example"] = {g.split("/")[0]: v[:6] for g, v in infos.items() if g.endswith("/sentences")}
    if not args.skip_motion:
        results["motion"] = motion_stats(ring, [r for r in roots if r.resolve() != ring.resolve()])
    results["wall_s"] = time.time() - t0
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    np.savez_compressed(out_dir / "samples.npz", **npz)
    print(f"[actor_stats] wrote {out_dir / 'results.json'} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
