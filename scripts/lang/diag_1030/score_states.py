#!/usr/bin/env python
"""Score language-filter checkpoints on a saved states file (check 4 of the 2026-10-30 diagnosis).

    python scripts/lang/diag_1030/score_states.py --states runs/lang/diag_1030/t5_branch/states/*.pt \\
        --ckpt runs/lang/diag_1030/retrain/{stoch,det,data,zero}/checkpoint.pt runs/lang/rounds_ur5_g095_1023/checkpoint.pt \\
        --out runs/lang/diag_1030/retrain/scores

A states file is a torch.save dict (or a list of per-state dicts, stacked here) with:
  the encoder inputs LangTrainer.encode consumes, either at the top level or under "inputs" / "enc_inputs" / "obs":
      state_tokens, body_tokens, link_pos, body_mask, joint_index, Jlin, Jang, dtheta_max,
      scene_tokens [N, 3, n_tok, 256] (frames t, t-k, 0), scene_pos [N, 3, n_tok, 3]
  candidate first actions: "actions" / "candidates" / "cand_actions" [N, K, A] (rad per tick) or a dict {name: [N, A]}
  their names: "action_names" / "names" / "branches" (K strings)
  sentences: "sentences" (N strings) / "sentence" (one string for all)
  anything else (meta, ground truth: "min_h" [N, K] etc.) is copied into the output when it is array-like of length N.
Per checkpoint and states file: <out>/<states stem>__<ckpt tag>.npz with qmin / q1 / q2 [N, K], q at the actor's mean
action and at zero, the actor's mean action, Q at 16 fixed uniform random actions in the box, and a json summary (mean
Q per candidate, the share of states where each candidate ranks first, the spread of Q over the 16 random actions, and
for EVERY ground-truth array [N, K] (a key containing "min_h", one of GT_KEYS, or named with --gt): Spearman rank
correlation, top-1 agreement and the pairwise SIGN agreement sign(Q_i - Q_j) == sign(gt_i - gt_j) over candidate pairs
whose ground truths differ by more than --gt-tol and whose Q gap reaches --q-tol (else no ranking), overall and per
named pair).

A checkpoint trained with normalized critic action inputs (critic_retrain.py stoch_norm: "diag_action_norm" in the
checkpoint) is scored with the same scaling (critic_retrain.ActionNorm).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
N_RAND = 16

ENC_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max",
              "scene_tokens", "scene_pos")
CACHES = [REPO / "data/lang_1023/warmup/text_cache.pt", REPO / "data/lang_1023/val/text_cache.pt",
          REPO / "data/lang_1023/val20/text_cache.pt"]
GT_KEYS = ("min_h", "gt_min_h", "branch_min_h", "h_min", "value", "gt")


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_states(path: Path, sentence_key: str | None = None) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if sentence_key:   # score under another sentence list (e.g. the T5 states' cf_sentences)
        if sentence_key not in d:
            raise SystemExit(f"{path}: no {sentence_key!r} (keys {sorted(d)[:30]})")
        d = {**d, "sentences": d[sentence_key]}
    if isinstance(d, (list, tuple)):   # per-state dicts
        keys = d[0].keys()
        d = {k: [x[k] for x in d] for k in keys}
        for k, v in list(d.items()):
            if k in ENC_FIELDS or k in ("actions", "candidates", "cand_actions") or k in GT_KEYS:
                d[k] = np.stack([_np(x) for x in v])
    inp = None
    for k in ("inputs", "enc_inputs", "obs", "encoder_inputs"):
        if isinstance(d.get(k), dict):
            inp = d[k]
            break
    inp = inp if inp is not None else d
    missing = [f for f in ENC_FIELDS if f not in inp]
    if missing:
        raise SystemExit(f"{path}: no encoder fields {missing} (keys {sorted(d)[:30]})")
    enc = {f: _np(inp[f]) for f in ENC_FIELDS}
    N = enc["state_tokens"].shape[0]
    acts, names = None, None
    for k in ("actions", "candidates", "cand_actions", "branch_actions"):
        if k in d:
            a = d[k]
            if isinstance(a, dict):
                names = list(a)
                acts = np.stack([_np(a[n]) for n in names], 1)
            else:
                acts = _np(a)
            break
    if acts is None:
        raise SystemExit(f"{path}: no candidate actions (keys {sorted(d)[:30]})")
    if acts.ndim == 2:
        acts = acts[:, None]
    for k in ("action_names", "names", "branches", "branch_names"):
        if names is None and k in d:
            names = [str(x) for x in d[k]]
    names = names or [f"a{k}" for k in range(acts.shape[1])]
    if "sentences" in d:
        s = d["sentences"]
        sents = [str(s)] * N if isinstance(s, str) else [str(x) for x in s]
    elif "sentence" in d:
        s = d["sentence"]
        sents = [str(s)] * N if isinstance(s, str) else [str(x) for x in s]
    else:
        raise SystemExit(f"{path}: no sentences")
    extra = {}
    for k, v in d.items():
        if k in ENC_FIELDS or k in ("inputs", "enc_inputs", "obs", "encoder_inputs", "sentences", "sentence"):
            continue
        try:
            a = _np(v)
        except Exception:
            continue
        if a.dtype != object and a.ndim >= 1 and a.shape[0] == N:
            extra[k] = a
    return {"enc": enc, "actions": acts.astype(np.float32), "names": names, "sentences": sents, "extra": extra, "N": N,
            "meta": d.get("meta")}


class Scorer:
    def __init__(self, ckpt: Path, device: str = "cuda"):
        from main.lang.config import LangConfig
        from main.lang.legacy_pool import checkpoint_vocab, install_vocab
        from main.lang.network import LangPolicyEncoder, LangTwinCritic, make_actor
        from main.lang.text import TextStore
        from main.lang.train import load_policy

        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        install_vocab(checkpoint_vocab(ck["lang_config"]))
        self.step, self.target = int(ck.get("step", -1)), ck.get("diag_target", "orig")
        self.norm = ck.get("diag_action_norm")
        self.pe = LangPolicyEncoder(self.cfg).to(device).eval()
        load_policy(self.pe, ck)
        self.actor = make_actor(self.cfg).to(device).eval()
        self.actor.load_state_dict(ck["actor"])
        self.critics = LangTwinCritic(self.cfg).to(device).eval()
        self.critics.load_state_dict(ck["critics"])
        run = Path(ckpt).parent
        extra = [run / "text_cache_extra.pt"] if (run / "text_cache_extra.pt").exists() else []
        out = REPO / "runs/lang/diag_1030/retrain/scores"
        out.mkdir(parents=True, exist_ok=True)
        self.text = TextStore([c for c in CACHES if c.exists()] + extra, self.cfg.text_backend, device,
                              extra_path=out / "text_cache_extra_score.pt", backend_paths=self.cfg.backend_paths)
        self.dev = device

    @torch.no_grad()
    def score(self, S: dict, chunk: int = 32) -> dict:
        from critic_retrain import ActionNorm
        from main.lang.train import _t

        if self.norm is not None or ActionNorm._orig is not None:
            ActionNorm.install(self.norm)
        U = np.random.default_rng(20261030).uniform(-1, 1, size=(S["N"], N_RAND, S["actions"].shape[2]))
        q_rand = np.zeros((S["N"], N_RAND), np.float32)

        N, K = S["N"], S["actions"].shape[1]
        q1 = np.zeros((N, K), np.float32)
        q2 = np.zeros((N, K), np.float32)
        q_det, q_zero = np.zeros(N, np.float32), np.zeros(N, np.float32)
        a_det = np.zeros((N, S["actions"].shape[2]), np.float32)
        d = self.dev
        for c0 in range(0, N, chunk):
            sl = slice(c0, min(N, c0 + chunk))
            e = {k: v[sl] for k, v in S["enc"].items()}
            n = e["state_tokens"].shape[0]
            words, wmask = self.text.lookup(S["sentences"][sl], splits=["probe"] * n)
            enc = self.pe(_t(e["state_tokens"], dev=d), _t(e["body_tokens"], dev=d), _t(e["link_pos"], dev=d),
                          _t(e["body_mask"], torch.bool, d), _t(e["joint_index"], torch.int64, d),
                          _t(e["scene_tokens"], dev=d), _t(e["scene_pos"], dev=d), words=words, word_mask=wmask)
            Jl, Ja, ji = _t(e["Jlin"], dev=d), _t(e["Jang"], dev=d), _t(e["joint_index"], torch.int64, d)
            dmax = _t(e["dtheta_max"], dev=d)
            A = _t(S["actions"][sl], dev=d)
            for k in range(K):
                a, b, _, _ = self.critics(enc, A[:, k], Jl, Ja, ji)
                q1[sl, k], q2[sl, k] = a.cpu().numpy(), b.cpu().numpy()
            ad = self.actor(enc.body, ji, dmax, deterministic=True)[0]
            a_det[sl] = ad.cpu().numpy()
            q_det[sl] = self.critics.qmin(enc, ad, Jl, Ja, ji).cpu().numpy()
            q_zero[sl] = self.critics.qmin(enc, torch.zeros_like(ad), Jl, Ja, ji).cpu().numpy()
            Ur = _t(U[sl], dev=d)
            for k in range(N_RAND):
                q_rand[sl, k] = self.critics.qmin(enc, Ur[:, k] * dmax, Jl, Ja, ji).cpu().numpy()
        return {"q1": q1, "q2": q2, "qmin": np.minimum(q1, q2), "q_actor_mean": q_det, "q_zero": q_zero,
                "a_actor_mean": a_det, "q_rand16": q_rand}


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def gt_arrays(S: dict, extra_keys=()) -> dict:
    K = S["actions"].shape[1]
    out = {}
    for k, v in S["extra"].items():
        if (k in GT_KEYS or "min_h" in k or k in extra_keys) and v.ndim == 2 and v.shape == (S["N"], K) \
                and np.issubdtype(v.dtype, np.number):
            out[k] = v.astype(np.float64)
    return out


def sign_agreement(q: np.ndarray, gt: np.ndarray, names, tol: float, q_tol: float = 1e-3) -> dict:
    """Over candidate pairs (i < j) whose ground truths differ by more than tol: the share with
    sign(Q_i - Q_j) == sign(gt_i - gt_j) AND |Q_i - Q_j| >= q_tol (a Q gap below q_tol is no ranking: counted as a
    miss); overall and per named pair, with counts, plus the share of pairs whose Q gap reaches q_tol at all."""
    N, K = q.shape
    tot_ok, tot_n, tot_dec, per = 0, 0, 0, {}
    for i in range(K):
        for j in range(i + 1, K):
            dg = gt[:, i] - gt[:, j]
            m = np.isfinite(dg) & (np.abs(dg) > tol)
            if not m.any():
                continue
            dq = q[m, i] - q[m, j]
            dec = np.abs(dq) >= q_tol
            ok = (np.sign(dq) == np.sign(dg[m])) & dec
            per[f"{names[i]}>{names[j]}"] = {"n": int(m.sum()), "agree": float(ok.mean()),
                                             "decided_frac": float(dec.mean()),
                                             "raw_sign_agree": float((np.sign(dq) == np.sign(dg[m])).mean()),
                                             "gt_i_better_frac": float((dg[m] > 0).mean()),
                                             "q_gap_mean": float(dq.mean()), "gt_gap_mean": float(dg[m].mean())}
            tot_ok += int(ok.sum())
            tot_n += int(m.sum())
            tot_dec += int(dec.sum())
    return {"n_pairs": tot_n, "agree": (tot_ok / tot_n) if tot_n else None,
            "decided_frac": (tot_dec / tot_n) if tot_n else None, "q_tol": q_tol, "per_pair": per}


def summarize(S: dict, r: dict, gt_keys=(), tol: float = 0.05, q_tol: float = 1e-3) -> dict:
    q = r["qmin"]
    names = S["names"]
    qr = r["q_rand16"]
    out = {"n_states": S["N"], "names": names,
           "q_mean": {n: float(q[:, k].mean()) for k, n in enumerate(names)},
           "best_frac": {n: float((q.argmax(1) == k).mean()) for k, n in enumerate(names)},
           "q_spread_mean": float((q.max(1) - q.min(1)).mean()),
           "spread16_mean": float((qr.max(1) - qr.min(1)).mean()),
           "q_actor_mean": float(r["q_actor_mean"].mean()), "q_zero": float(r["q_zero"].mean())}
    for g, gt in gt_arrays(S, gt_keys).items():
        rho = [spearman(q[i], gt[i]) for i in range(S["N"]) if np.isfinite(gt[i]).all()]
        out[f"gt/{g}"] = {"spearman": float(np.nanmean(rho)) if len(rho) and np.isfinite(rho).any() else None,
                          "top1_agree": float((q.argmax(1) == np.nan_to_num(gt, nan=-9).argmax(1)).mean()),
                          "gt_mean": {n: float(np.nanmean(gt[:, k])) for k, n in enumerate(names)},
                          "sign": sign_agreement(q, gt, names, tol, q_tol)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO / "runs/lang/diag_1030/retrain/scores"))
    ap.add_argument("--gt", nargs="*", default=[], help="extra ground-truth keys [N, K] to compare against")
    ap.add_argument("--gt-tol", type=float, default=0.05, help="ground-truth gap (h units) below which a pair is a tie")
    ap.add_argument("--q-tol", type=float, default=1e-3, help="Q gap below which the critic does not rank a pair")
    ap.add_argument("--sentence-key", default=None, help="score under this sentence list instead of 'sentences'")
    ap.add_argument("--tag", default="", help="suffix of the output names (e.g. _cf)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    states = [Path(p) for s in a.states for p in sorted(glob.glob(s))]
    loaded = {p: load_states(p, a.sentence_key) for p in states}
    summary = {}
    for ck in a.ckpt:
        sc = Scorer(Path(ck))
        tag = f"{Path(ck).parent.name}__{Path(ck).stem}{a.tag}"
        for p, S in loaded.items():
            r = sc.score(S)
            np.savez_compressed(out / f"{p.stem}__{tag}.npz", names=np.array(S["names"]), **r,
                                **{f"extra_{k}": v for k, v in S["extra"].items() if v.dtype != object})
            sm = summarize(S, r, a.gt, a.gt_tol, a.q_tol)
            sm.update({"ckpt": str(ck), "step": sc.step, "target": sc.target, "states": str(p)})
            summary[f"{p.stem}__{tag}"] = sm
            print(f"[score] {p.name} x {tag} (step {sc.step}, {sc.target}): "
                  + " ".join(f"{n}={v:+.4f}" for n, v in sm["q_mean"].items())
                  + f" spread {sm['q_spread_mean']:.5f} spread16 {sm['spread16_mean']:.5f} "
                  + " ".join(f"{k}: sign {v['sign']['agree']}" for k, v in sm.items() if k.startswith("gt/")),
                  flush=True)
        del sc
        torch.cuda.empty_cache()
    sp = out / "summary.json"
    old = json.loads(sp.read_text()) if sp.exists() else {}
    old.update(summary)
    sp.write_text(json.dumps(old, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
