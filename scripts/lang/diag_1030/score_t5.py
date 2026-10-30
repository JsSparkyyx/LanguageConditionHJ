#!/usr/bin/env python
"""Score check: the retrained critics (check 4) and the vanilla checkpoints on check 2's T5 branch states, against
check 2's three ground truths (diag 2026-10-30).

  score      (GPU)  python scripts/lang/diag_1030/score_t5.py score --ckpt A.pt B.pt ... [--states ...] [--out DIR]
  aggregate  (CPU, light: numpy + json only)  python scripts/lang/diag_1030/score_t5.py aggregate [--out DIR]

score: per checkpoint and branch-state file (runs/lang/diag_1030/t5_branch/states/<episode>.pt, loaded with
score_states.load_states) writes <out>/npz/<episode>__<tag>.npz, tag = <run dir>__<ckpt stem>__s<step>, with
  qmin, q1, q2 [N, 5]   Q(s, a) at the 5 candidate first actions (a_nom, zero, half_lit, half_cmd, slow_cmd) under
                        the tall block's sentence (the named rule of the branch truths)
  qmin_cf [N, 5]        the same under the short block's sentence (cf_sentences)
  q_rand16 [N, 16]      Q at 16 fixed uniform random actions u * dtheta_max (same RNG as score_states.py)
  a_mean, q_mean, q_zero, q_samp [N, 8]   the actor's mean action, Q there, Q(s, 0), Q at 8 actor samples
  u_abs, sigma [N]      mean |tanh mu| and mean sigma over the actor's valid arm joints (dtheta_max > 0)
  v_anom, v_zero [N, H] the softmin readout's per-head values (twin 1) at a_nom and at 0 (softmin only)
  extras of the state file (t, j, is_drag, lead_to_brake_s, min_h, V95)
Already scored (file exists) is skipped, so a rerun resumes.

aggregate: joins the npz files with check 2's results.json rows (episode, t) and writes <out>/results.json:
  truths  full = the four full branches (plan, stop, half, slow; mean over the 4 runs, rep 0 = exact restore),
          det  = the branch's first tick, then the vanilla final checkpoint's actor mean (15 drag ticks, 1 rollout),
          sto  = the branch's first tick, then that actor sampled (15 drag ticks, mean of 4 rollouts);
          values V95 (the gamma 0.95 recursion on the h_tall trace, the quantity Q estimates) and min_h.
  per tag and truth: pairwise sign agreement over branch pairs whose truths differ by > 0.05 (strict: also
  |dQ| >= 1e-3; raw: any nonzero dQ), Kendall tau-b per state, signed / absolute error Q - V95 per branch, Pearson r
  of Q vs V95 over cells, the Q spread over the 4 branch actions and over the 16 random actions vs 0.02.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

OUT = REPO / "runs/lang/diag_1030/score"
STATES = REPO / "runs/lang/diag_1030/t5_branch/states"
BRANCH_RESULTS = REPO / "runs/lang/diag_1030/t5_branch/results.json"
N_SAMP = 8
BRANCHES = ("plan", "stop", "half", "slow")
# first action of each branch = column of the states file's candidates
#   (a_nom, zero, half_lit, half_cmd, slow_cmd): plan <- a_nom, stop <- zero, half <- half_cmd, slow <- slow_cmd
BR_COL = {"plan": 0, "stop": 1, "half": 3, "slow": 4}
GT_TOL, Q_TOL, SPREAD_THR = 0.05, 1e-3, 0.02


# ----------------------------------------------------------------------------------------------------------- score
class T5Scorer:
    """score_states.Scorer's model loading (same classes, same text caches), with this check's own extra text cache
    (never the retrain check's) and the actor / readout diagnostics this check needs."""

    def __init__(self, ckpt: Path, device: str = "cuda"):
        import torch
        from main.lang.config import LangConfig
        from main.lang.legacy_pool import checkpoint_vocab, install_vocab
        from main.lang.network import LangPolicyEncoder, LangTwinCritic, make_actor
        from main.lang.text import TextStore
        from main.lang.train import load_policy
        from score_states import CACHES

        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        install_vocab(checkpoint_vocab(ck["lang_config"]))
        self.step, self.target = int(ck.get("step", -1)), ck.get("diag_target", "orig")
        self.norm = ck.get("diag_action_norm")
        self.readout = getattr(self.cfg, "critic_readout", "softmin")
        self.pe = LangPolicyEncoder(self.cfg).to(device).eval()
        load_policy(self.pe, ck)
        self.actor = make_actor(self.cfg).to(device).eval()
        self.actor.load_state_dict(ck["actor"])
        self.critics = LangTwinCritic(self.cfg).to(device).eval()
        self.critics.load_state_dict(ck["critics"])
        run = Path(ckpt).parent
        extra = [run / "text_cache_extra.pt"] if (run / "text_cache_extra.pt").exists() else []
        OUT.mkdir(parents=True, exist_ok=True)
        self.text = TextStore([c for c in CACHES if c.exists()] + extra, self.cfg.text_backend, device,
                              extra_path=OUT / "text_cache_extra_score.pt", backend_paths=self.cfg.backend_paths)
        self.dev = device
        del ck

    def _encode(self, e, sents):
        from main.lang.train import _t
        import torch

        d = self.dev
        words, wmask = self.text.lookup(sents, splits=["probe"] * len(sents))
        return self.pe(_t(e["state_tokens"], dev=d), _t(e["body_tokens"], dev=d), _t(e["link_pos"], dev=d),
                       _t(e["body_mask"], torch.bool, d), _t(e["joint_index"], torch.int64, d),
                       _t(e["scene_tokens"], dev=d), _t(e["scene_pos"], dev=d), words=words, word_mask=wmask)

    def score(self, S: dict, chunk: int = 32) -> dict:
        import torch
        from critic_retrain import ActionNorm
        from main.lang.train import _t

        if self.norm is not None or ActionNorm._orig is not None:
            ActionNorm.install(self.norm)          # a stoch_norm checkpoint is scored with its own action scaling
        N, K, A = S["N"], S["actions"].shape[1], S["actions"].shape[2]
        U = np.random.default_rng(20261030).uniform(-1, 1, size=(N, 16, A))     # = score_states.py's draw
        cf = [str(x) for x in S["extra"]["cf_sentences"]] if "cf_sentences" in S["extra"] else None
        z = lambda *s: np.zeros(s, np.float32)                                   # noqa: E731
        out = {"q1": z(N, K), "q2": z(N, K), "qmin_cf": z(N, K), "q_rand16": z(N, 16), "a_mean": z(N, A),
               "q_mean": z(N), "q_zero": z(N), "q_samp": z(N, N_SAMP), "u_abs": z(N), "sigma": z(N),
               "a_mean_frac": z(N)}
        vs = {"v_anom": [], "v_zero": []}
        gen = torch.Generator(device=self.dev).manual_seed(1030)
        d = self.dev
        with torch.no_grad():
            for c0 in range(0, N, chunk):
                sl = slice(c0, min(N, c0 + chunk))
                e = {k: v[sl] for k, v in S["enc"].items()}
                enc = self._encode(e, S["sentences"][sl])
                Jl, Ja, ji = _t(e["Jlin"], dev=d), _t(e["Jang"], dev=d), _t(e["joint_index"], torch.int64, d)
                dmax = _t(e["dtheta_max"], dev=d)
                Acand = _t(S["actions"][sl], dev=d)
                for k in range(K):
                    a, b, v1, _ = self.critics(enc, Acand[:, k], Jl, Ja, ji)
                    out["q1"][sl, k], out["q2"][sl, k] = a.cpu().numpy(), b.cpu().numpy()
                    if k == 0 and v1 is not None:
                        vs["v_anom"].append(v1.float().cpu().numpy())
                zero = torch.zeros_like(Acand[:, 0])
                a, b, v1, _ = self.critics(enc, zero, Jl, Ja, ji)
                out["q_zero"][sl] = torch.minimum(a, b).cpu().numpy()
                if v1 is not None:
                    vs["v_zero"].append(v1.float().cpu().numpy())
                Ur = _t(U[sl], dev=d)
                for k in range(16):
                    out["q_rand16"][sl, k] = self.critics.qmin(enc, Ur[:, k] * dmax, Jl, Ja, ji).cpu().numpy()
                # actor: mean action, Q there, 8 samples, |tanh mu| and sigma over valid arm joints
                ad = self.actor(enc.body, ji, dmax, deterministic=True)[0]
                out["a_mean"][sl] = ad.cpu().numpy()
                out["q_mean"][sl] = self.critics.qmin(enc, ad, Jl, Ja, ji).cpu().numpy()
                M = ji.shape[1]
                mu, log_std = self.actor.head(enc.body[:, :M]).unbind(-1)
                sig = log_std.clamp(-5, 0).exp()
                own = torch.gather(dmax, 1, ji.clamp_min(0))
                valid = ((ji >= 0) & (own > 0)).float()
                nv = valid.sum(1).clamp_min(1)
                out["u_abs"][sl] = ((torch.tanh(mu).abs() * valid).sum(1) / nv).cpu().numpy()
                out["sigma"][sl] = ((sig * valid).sum(1) / nv).cpu().numpy()
                arm = dmax > 0
                out["a_mean_frac"][sl] = ((ad.abs() / dmax.clamp_min(1e-12)) * arm).sum(1).div(
                    arm.sum(1).clamp_min(1)).cpu().numpy()
                for s in range(N_SAMP):
                    pre = mu + sig * torch.randn(mu.shape, generator=gen, device=mu.device)
                    u = torch.tanh(pre)
                    act = u.new_zeros(u.shape[0], dmax.shape[1])
                    bi, ti = (ji >= 0).nonzero(as_tuple=True)
                    act[bi, ji[bi, ti]] = u[bi, ti]
                    out["q_samp"][sl, s] = self.critics.qmin(enc, act * dmax, Jl, Ja, ji).cpu().numpy()
                if cf is not None:
                    enc_cf = self._encode(e, cf[sl])
                    for k in range(K):
                        out["qmin_cf"][sl, k] = self.critics.qmin(enc_cf, Acand[:, k], Jl, Ja, ji).cpu().numpy()
        out["qmin"] = np.minimum(out["q1"], out["q2"])
        for k, v in vs.items():
            if v:
                out[k] = np.concatenate(v)
        if self.readout == "softmin":
            out["n_query_heads"] = np.array(int(getattr(self.critics.c1, "K", 0) or 0))
        return out


def ckpt_tag(ck: Path, step: int) -> str:
    return f"{ck.parent.name}__{ck.stem}__s{step}"


def cmd_score(a) -> int:
    import torch
    from score_states import load_states

    out = Path(a.out) / "npz"
    out.mkdir(parents=True, exist_ok=True)
    states = sorted({Path(p) for s in a.states for p in glob.glob(s)})
    states = [p for p in states if p.suffix == ".pt"]
    print(f"[score_t5] {len(states)} state files, {len(a.ckpt)} checkpoints", flush=True)
    loaded = {}
    for ck in a.ckpt:
        ck = Path(ck)
        if not ck.exists():
            print(f"[score_t5] MISSING {ck}", flush=True)
            continue
        t0 = time.time()
        try:
            sc = T5Scorer(ck)
        except Exception as ex:                       # a checkpoint being replaced: report, go on
            print(f"[score_t5] load failed {ck}: {ex!r}", flush=True)
            continue
        tag = ckpt_tag(ck, sc.step)
        todo = [p for p in states if not (out / f"{p.stem}__{tag}.npz").exists()]
        for p in todo:
            if p not in loaded:
                loaded[p] = load_states(p)
            S = loaded[p]
            r = sc.score(S)
            meta = {"ckpt": str(ck), "step": sc.step, "target": sc.target, "readout": sc.readout,
                    "states": str(p), "names": S["names"]}
            np.savez_compressed(out / f"{p.stem}__{tag}.npz", meta=np.array(json.dumps(meta)), **r,
                                **{f"extra_{k}": v for k, v in S["extra"].items()
                                   if v.dtype != object and v.dtype.kind != "U"})
        print(f"[score_t5] {tag} ({sc.target}, {sc.readout}): {len(todo)} file(s) in {time.time() - t0:.0f} s",
              flush=True)
        del sc
        torch.cuda.empty_cache()
    return 0


# ------------------------------------------------------------------------------------------------------- aggregate
def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def pair_stats(Q: np.ndarray, T: np.ndarray, gt_tol=GT_TOL, q_tol=Q_TOL) -> dict:
    """Q, T [n, K] (states x candidates). Over pairs (i < j) of one state with |T_i - T_j| > gt_tol."""
    n, K = Q.shape
    agree_s = agree_r = ties = npair = 0
    taus_s, taus_r = [], []
    for s in range(n):
        if not np.isfinite(T[s]).all():
            continue
        C_s = D_s = C_r = D_r = tq_s = tq_r = tt = n0 = 0
        for i in range(K):
            for j in range(i + 1, K):
                n0 += 1
                dt, dq = T[s, i] - T[s, j], Q[s, i] - Q[s, j]
                t_tie = abs(dt) <= gt_tol
                tt += t_tie
                q_tie_s, q_tie_r = abs(dq) < q_tol, dq == 0
                tq_s += q_tie_s
                tq_r += q_tie_r
                if t_tie:
                    continue
                npair += 1
                same = np.sign(dq) == np.sign(dt)
                agree_r += bool(same and not q_tie_r)
                agree_s += bool(same and not q_tie_s)
                ties += bool(q_tie_r)
                if not q_tie_s:
                    C_s += same
                    D_s += not same
                if not q_tie_r:
                    C_r += same
                    D_r += not same
        for C, D, tq, lst in ((C_s, D_s, tq_s, taus_s), (C_r, D_r, tq_r, taus_r)):
            den = np.sqrt(float(n0 - tq) * float(n0 - tt))
            if den > 0:
                lst.append((C - D) / den)
    return {"n_states": int(n), "n_pairs": int(npair),
            "agree_strict": _f(agree_s / npair) if npair else None,
            "agree_raw": _f(agree_r / npair) if npair else None,
            "q_exact_tie_frac": _f(ties / npair) if npair else None,
            "kendall_tau_b_strict": _f(np.mean(taus_s)) if taus_s else None, "n_tau_strict": len(taus_s),
            "kendall_tau_b_raw": _f(np.mean(taus_r)) if taus_r else None, "n_tau_raw": len(taus_r)}


def err_stats(Q: np.ndarray, T: np.ndarray) -> dict:
    m = np.isfinite(T) & np.isfinite(Q)
    if not m.any():
        return {"n": 0}
    e = (Q - T)[m]
    r = np.corrcoef(Q[m], T[m])[0, 1] if m.sum() > 2 and Q[m].std() > 0 and T[m].std() > 0 else np.nan
    return {"n": int(m.sum()), "signed": _f(e.mean()), "mae": _f(np.abs(e).mean()), "pearson_r": _f(r),
            "q_mean": _f(Q[m].mean()), "truth_mean": _f(T[m].mean())}


VARIANT_TRUTH = {
    "vanilla": ("sto", "trained with a' sampled from its own stochastic actor; the measured T_sto continuation IS the "
                "vanilla final checkpoint's actor (exact for the final checkpoint, a proxy for rounds 10/20/30)"),
    "stoch": ("sto", "a' sampled from its own stochastic actor; T_sto uses the vanilla final actor (proxy: both have "
              "|tanh mu| < 0.1 and sigma ~0.86-0.90)"),
    "stoch_norm": ("sto", "as stoch (only the critic's action inputs are rescaled)"),
    "stoch_mlp": ("sto", "as stoch (MLP readout)"),
    "det": ("det", "a' = its own actor's mean; T_det uses the vanilla final actor's mean (1.8% of dtheta_max, a "
            "drifting near-freeze); the retrained det actor's own mean is measured here (a_mean_frac)"),
    "data": ("plan@a_nom", "a' = the stored next command (behaviour policy: in T5 drag states the planner); its exact "
             "truth exists only for the first action a_nom (= the full plan branch); 'a then planner' for the other "
             "first actions was not simulated"),
    "zero": ("stop@zero", "a' = 0 (freeze forever); its exact truth exists only for the first action 0 (= the full "
             "stop branch); for other first actions T_det (a then near-freeze) is the nearest measured proxy"),
}


def variant_of(tag: str) -> str:
    run = tag.split("__")[0]
    return "vanilla" if run.startswith("rounds_") else run


def load_truth_rows():
    d = json.loads(BRANCH_RESULTS.read_text())
    rows = {}
    for r in d["rows"]:
        rows[(r["episode"], round(float(r["t"]), 3))] = r
    return rows, d


def truth_properties(br: dict) -> dict:
    """How much the first action matters under each truth (within-state max - min of V95 over the 4 branch first
    actions), the 4-rollout noise floor of T_sto, a constant predictor's MAE, and whether T_det / T_sto rank the
    first actions like the full branches."""
    ra = [r for r in br["rows"] if "plan_det" in r]
    dr = [r for r in br["rows"] if r["kind"] == "drag"]
    Tf = np.array([[r[b]["V95"] for b in BRANCHES] for r in dr])
    out = {"full": {"n_states": len(dr), "within_state_range_mean": _f(np.ptp(Tf, 1).mean()),
                    "frac_range_gt_0.02": _f((np.ptp(Tf, 1) > SPREAD_THR).mean()),
                    "V95_mean": _f(Tf.mean()), "V95_sd": _f(Tf.std()),
                    "constant_predictor_mae": _f(np.abs(Tf - Tf.mean()).mean())}}
    Tfa = np.array([[r[b]["V95"] for b in BRANCHES] for r in ra])
    for tr in ("det", "sto"):
        T = np.array([[r[f"{b}_{tr}"]["V95"] for b in BRANCHES] for r in ra])
        e = {"n_states": len(ra), "within_state_range_mean": _f(np.ptp(T, 1).mean()),
             "frac_range_gt_0.02": _f((np.ptp(T, 1) > SPREAD_THR).mean()), "V95_mean": _f(T.mean()),
             "V95_sd": _f(T.std()), "constant_predictor_mae": _f(np.abs(T - T.mean()).mean()),
             "tip_rate": _f(np.mean([[r[f"{b}_{tr}"]["tip"] for b in BRANCHES] for r in ra]))}
        ok = n = 0
        for s in range(len(ra)):
            for i in range(4):
                for j in range(i + 1, 4):
                    if abs(Tfa[s, i] - Tfa[s, j]) > GT_TOL and abs(T[s, i] - T[s, j]) > GT_TOL:
                        n += 1
                        ok += bool(np.sign(Tfa[s, i] - Tfa[s, j]) == np.sign(T[s, i] - T[s, j]))
        e["ranking_agrees_with_full"] = {"agree": ok, "n_pairs": n}
        if tr == "sto":
            sd = np.array([[r[f"{b}_sto"]["V95_sd"] for b in BRANCHES] for r in ra])
            nn = np.array([[r[f"{b}_sto"]["n"] for b in BRANCHES] for r in ra])
            se = float((sd / np.sqrt(nn)).mean())
            e["noise"] = {"rollout_sd_mean": _f(sd.mean()), "se_of_4_rollout_mean": se,
                          "expected_range_of_4_noise_only": _f(2.059 * se)}
        out[tr] = e
    return out


def cmd_aggregate(a) -> int:
    out = Path(a.out)
    rows, br = load_truth_rows()
    files = sorted((out / "npz").glob("*.npz"))
    by_tag: dict[str, list] = {}
    for f in files:
        ep, tag = f.stem.split("__", 1)
        by_tag.setdefault(tag, []).append((ep, f))
    n_ep_all = len({r[0] for r in rows})
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "states_dir": str(STATES),
           "branch_results": str(BRANCH_RESULTS), "gt_tol": GT_TOL, "q_tol": Q_TOL, "spread_threshold": SPREAD_THR,
           "branch_first_action": {b: ["a_nom", "zero", "half_lit", "half_cmd", "slow_cmd"][c] for b, c in BR_COL.items()},
           "variant_truth": {k: {"truth": v[0], "why": v[1]} for k, v in VARIANT_TRUTH.items()},
           "tags": {}}
    for tag, lst in sorted(by_tag.items()):
        cells = {"full": {"Q": [], "V95": [], "min_h": [], "tip": [], "V95_0": [], "lead": [], "ptip": []},
                 "det": {"Q": [], "V95": [], "min_h": [], "tip": []},
                 "sto": {"Q": [], "V95": [], "min_h": [], "tip": []}}
        extra = {"pause3_V95": [], "qzero_drag": [], "qanom_drag": []}
        diag = {k: [] for k in ("spread4", "spread16", "q_anom", "q_cf_anom", "u_abs", "sigma", "a_mean_frac",
                                "samp_minus_mean", "mean_minus_anom", "min_on_query", "body_med", "head_move_query",
                                "head_move_body", "truth_spread4", "vs_check2_absdiff")}
        meta0, n_drag, n_missing = None, 0, 0
        for ep, f in sorted(lst):
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            meta0 = meta0 or meta
            t = z["extra_t"]
            isd = z["extra_is_drag"].astype(bool)
            Q = z["qmin"]
            for i in range(len(t)):
                r = rows.get((ep, round(float(t[i]), 3)))
                if r is None:
                    n_missing += 1
                    continue
                if not isd[i] or r["kind"] != "drag":
                    continue
                n_drag += 1
                qb = np.array([Q[i, BR_COL[b]] for b in BRANCHES])
                cells["full"]["Q"].append(qb)
                for k in ("V95", "min_h", "tip", "V95_0"):
                    cells["full"][k].append([r[b][k] for b in BRANCHES])
                cells["full"]["lead"].append(r["lead_brake"] if r["lead_brake"] is not None else np.nan)
                cells["full"]["ptip"].append(bool(r["nominal_tipped"]) and r["plan"]["tip"] > 0.5)
                extra["pause3_V95"].append(r["pause3"]["V95"])
                for tr in ("det", "sto"):
                    if f"plan_{tr}" in r:
                        cells[tr]["Q"].append(qb)
                        for k in ("V95", "min_h", "tip"):
                            cells[tr][k].append([r[f"{b}_{tr}"][k] for b in BRANCHES])
                if variant_of(tag) == "vanilla":            # consistency with check 2's own scoring of this checkpoint
                    qrec = r.get("Q", {}).get(f"{tag.split('__')[1]}|tall", {}).get("X")
                    if qrec:
                        diag["vs_check2_absdiff"].append(max(abs(qrec[b] - qb[k]) for k, b in enumerate(BRANCHES)))
                diag["spread4"].append(qb.max() - qb.min())
                diag["truth_spread4"].append(np.ptp([r[b]["V95"] for b in BRANCHES]))
                diag["spread16"].append(np.ptp(z["q_rand16"][i]))
                diag["q_anom"].append(Q[i, 0])
                diag["q_cf_anom"].append(z["qmin_cf"][i, 0])
                diag["u_abs"].append(z["u_abs"][i])
                diag["sigma"].append(z["sigma"][i])
                diag["a_mean_frac"].append(z["a_mean_frac"][i])
                diag["samp_minus_mean"].append(np.abs(z["q_samp"][i] - z["q_mean"][i]).mean())
                diag["mean_minus_anom"].append(z["q_mean"][i] - Q[i, 0])
                if "v_anom" in z.files:
                    V, Vz = z["v_anom"][i], z["v_zero"][i]
                    Kq = int(z["n_query_heads"]) if "n_query_heads" in z.files else 4
                    nb = V.shape[0] - Kq
                    mb = np.abs(V[:nb] - 1e4) > 1e-3          # padding heads carry exactly 1e4 (masked_fill)
                    body = V[:nb][mb]
                    diag["min_on_query"].append(float(np.argmin(V) >= nb))
                    diag["body_med"].append(np.median(body) if body.size else np.nan)
                    diag["head_move_query"].append(np.abs(V[nb:] - Vz[nb:]).max())
                    diag["head_move_body"].append(np.abs(V[:nb][mb] - Vz[:nb][mb]).max() if mb.any() else np.nan)
        T = {}
        for tr, c in cells.items():
            T[tr] = {k: np.array(v, dtype=np.float64) for k, v in c.items() if len(v)}
        ent = {"ckpt": meta0["ckpt"], "step": meta0["step"], "target": meta0["target"], "readout": meta0["readout"],
               "variant": variant_of(tag), "n_episodes": len(lst), "n_episodes_expected": n_ep_all,
               "n_drag_states": n_drag, "n_unmatched_rows": n_missing}
        # spreads
        sp4, sp16, tsp = (np.array(diag[k]) for k in ("spread4", "spread16", "truth_spread4"))
        ent["spread"] = {"q_spread4_mean": _f(sp4.mean()), "q_spread4_max": _f(sp4.max()),
                         "q_spread16_mean": _f(sp16.mean()), "q_spread16_max": _f(sp16.max()),
                         "frac_states_spread16_ge_thr": _f((sp16 >= SPREAD_THR).mean()),
                         "frac_states_spread4_ge_thr": _f((sp4 >= SPREAD_THR).mean()),
                         "truth_V95_spread4_mean": _f(tsp.mean()),
                         "ratio_q_to_truth_spread4": _f(sp4.mean() / tsp.mean()),
                         "clear_spread_0.02": bool(sp16.mean() >= SPREAD_THR), "n": int(len(sp4))}
        # ranking and error vs each truth
        ent["vs"] = {}
        for tr in ("full", "det", "sto"):
            if "Q" not in T[tr]:
                continue
            Qc = T[tr]["Q"]
            e = {"n_states": int(len(Qc))}
            for key in ("V95", "min_h"):
                e[f"rank_{key}"] = pair_stats(Qc, T[tr][key])
            e["err_V95"] = err_stats(Qc, T[tr]["V95"])
            e["err_V95_per_branch"] = {b: err_stats(Qc[:, i], T[tr]["V95"][:, i]) for i, b in enumerate(BRANCHES)}
            e["truth_tip_rate"] = {b: _f(T[tr]["tip"][:, i].mean()) for i, b in enumerate(BRANCHES)}
            ent["vs"][tr] = e
        # full truth on the planner-tipping / planner-safe subsets
        F = T["full"]
        pt = F["ptip"].astype(bool)
        ent["vs_full_subsets"] = {
            "planner_tips": {"n": int(pt.sum()), "rank_V95": pair_stats(F["Q"][pt], F["V95"][pt]),
                             "err_V95": err_stats(F["Q"][pt], F["V95"][pt])},
            "planner_safe": {"n": int((~pt).sum()), "rank_V95": pair_stats(F["Q"][~pt], F["V95"][~pt]),
                             "err_V95": err_stats(F["Q"][~pt], F["V95"][~pt])}}
        # the variant's own continuation
        v = variant_of(tag)
        own = {}
        if v in ("vanilla", "stoch", "stoch_norm", "stoch_mlp") and "Q" in T["sto"]:
            own = {"truth": "sto (4 branch first actions x 15 ticks)", **err_stats(T["sto"]["Q"], T["sto"]["V95"])}
        elif v == "det" and "Q" in T["det"]:
            own = {"truth": "det (4 x 15)", **err_stats(T["det"]["Q"], T["det"]["V95"])}
        elif v == "data":
            own = {"truth": "full plan branch at a_nom (117 drag ticks)", **err_stats(F["Q"][:, 0], F["V95"][:, 0])}
            own["proxy_zero_then_planner_pause3"] = err_stats(F["Q"][:, 1], np.array(extra["pause3_V95"]))
        elif v == "zero":
            own = {"truth": "full stop branch at a = 0 (117 drag ticks)", **err_stats(F["Q"][:, 1], F["V95"][:, 1])}
            if "Q" in T["det"]:
                own["proxy_det_other_first_actions"] = err_stats(T["det"]["Q"][:, [0, 2, 3]],
                                                                 T["det"]["V95"][:, [0, 2, 3]])
        own["comparison_all_truths_mae"] = {tr: ent["vs"][tr]["err_V95"].get("mae") for tr in ent["vs"]}
        ent["own_continuation"] = own
        # actor and readout
        ent["actor"] = {"u_abs_mean": _f(np.mean(diag["u_abs"])), "sigma_mean": _f(np.mean(diag["sigma"])),
                        "a_mean_frac_of_dtheta_max": _f(np.mean(diag["a_mean_frac"])),
                        "q_samples_minus_q_mean_abs": _f(np.mean(diag["samp_minus_mean"])),
                        "q_mean_minus_q_anom": _f(np.mean(diag["mean_minus_anom"])),
                        "u_abs_ge_0.05": bool(np.mean(diag["u_abs"]) >= 0.05),
                        "sigma_lt_0.9": bool(np.mean(diag["sigma"]) < 0.9),
                        "note": "a tanh-Gaussian with mu = 0 has maximum entropy at sigma = 0.874 (0.684 nats per "
                                "joint, 8.20 for 12): sigma < 0.9 alone is where an actor with no Q gradient rests, "
                                "not a sign that it left zero"}
        if diag["min_on_query"]:
            ent["readout"] = {"min_on_query_frac": _f(np.mean(diag["min_on_query"])),
                              "body_head_median": _f(np.nanmedian(diag["body_med"])),
                              "query_head_move_anom_vs_zero_mean": _f(np.mean(diag["head_move_query"])),
                              "body_head_move_anom_vs_zero_mean": _f(np.nanmean(diag["head_move_body"]))}
        if diag["vs_check2_absdiff"]:
            ent["consistency_vs_check2_Q"] = {"n_states": len(diag["vs_check2_absdiff"]),
                                              "max_abs_diff": _f(np.max(diag["vs_check2_absdiff"])),
                                              "mean_abs_diff": _f(np.mean(diag["vs_check2_absdiff"])),
                                              "note": "check 2 scored the live fp32 tokens; the saved states hold fp16 "
                                                      "scene tokens (t5_branch.py line 783: .half())"}
        ent["sentence"] = {"q_anom_tall_mean": _f(np.mean(diag["q_anom"])),
                           "q_anom_short_mean": _f(np.mean(diag["q_cf_anom"])),
                           "short_minus_tall_mean": _f(np.mean(np.array(diag["q_cf_anom"]) - np.array(diag["q_anom"])))}
        res["tags"][tag] = ent
    res["truth_properties"] = truth_properties(br)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    # console table
    print(f"{'tag':58s} {'step':>6s} {'ep':>3s} {'sp4':>8s} {'sp16':>8s} | full: agS agR tauR  MAE  sgn"
          f" | det: agR MAE sgn | sto: agR MAE sgn | own MAE | u_abs sigma")
    for tag, e in res["tags"].items():
        s = e["spread"]
        line = f"{tag[:58]:58s} {e['step']:6d} {e['n_episodes']:3d} {s['q_spread4_mean']:8.1e} {s['q_spread16_mean']:8.1e} |"
        for tr in ("full", "det", "sto"):
            v = e["vs"].get(tr)
            if v is None:
                line += " -- |"
                continue
            rk, er = v["rank_V95"], v["err_V95"]
            fmt = lambda x: "  nan" if x is None else f"{x:5.2f}"        # noqa: E731
            if tr == "full":
                line += f" {fmt(rk['agree_strict'])} {fmt(rk['agree_raw'])} {fmt(rk['kendall_tau_b_raw'])}"
            else:
                line += f" {fmt(rk['agree_raw'])}"
            line += f" {fmt(er['mae'])} {er['signed']:+.2f} |"
        om = e["own_continuation"].get("mae")
        line += f" {('%5.2f' % om) if om is not None else '  -- '} | {e['actor']['u_abs_mean']:.3f} {e['actor']['sigma_mean']:.3f}"
        print(line)
    print(f"[aggregate] {len(res['tags'])} tags -> {out / 'results.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--ckpt", nargs="+", required=True)
    s.add_argument("--states", nargs="+", default=[str(STATES / "*.pt")])
    s.add_argument("--out", default=str(OUT))
    g = sub.add_parser("aggregate")
    g.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    return cmd_score(a) if a.cmd == "score" else cmd_aggregate(a)


if __name__ == "__main__":
    raise SystemExit(main())
