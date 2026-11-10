#!/usr/bin/env python
"""Step 1 of ACTION_SENSITIVITY_PLAN (2026-11-10): can the critic fit action differences when they are given directly?

No simulator. Data: the drag states of the 10-30 branch tool (runs/lang/diag_1030/t5_branch/states/*.pt, stored
encoder inputs), three first actions per state with clearly different commands (the planner's, zero, half speed's
first command), each with the V95 of its whole continuation (plan / stop / half branch). Fit: start from the vanilla
final checkpoint and regress Q(s, a) on V95 (Huber, both twins, no Bellman), everything trainable (encoder, text
adapter, critics). 3 folds by episode (8 train, 4 held out). Variants:

    softmin   the checkpoint's critic exactly as trained (softmin readout, Linear injection): the reference
    linear    current injection with the MLP readout (blocks and act_proj from the checkpoint, a fresh readout)
    revised   the revised injection (plan 3.3: body RMSNorm, bounded features, MLP + RMSNorm, action in the readout)

Report per variant, on training and held-out states: rank agreement (pairs of a state's actions whose V95 differ by
more than 0.05: the share Q orders as the truth does), the spread of Q over the three actions against the true
spread, and the mean absolute error.

    python scripts/lang/action_sens/fit_check.py [--steps 300] [--variants softmin linear revised]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STATES = REPO / "runs/lang/diag_1030/t5_branch/states"
CKPT = REPO / "runs/lang/rounds_ur5_g095_1023/checkpoint.pt"
NORM = REPO / "runs/lang/diag_1030/retrain/action_norm.json"
OUT = REPO / "runs/lang/action_sens/step1"
CACHES = [REPO / "data/lang_1023/warmup/text_cache.pt", REPO / "data/lang_1023/val/text_cache.pt",
          REPO / "data/lang_1023/val20/text_cache.pt"]
ACTIONS = {"planner": 0, "zero": 1, "half": 3}   # columns of the states files' actions / V95 (a_nom, zero, half_cmd)
ENC = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "scene_tokens", "scene_pos")
GT_TOL = 0.05


def load_states():
    eps = []
    for p in sorted(STATES.glob("s*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = d["is_drag"].bool()
        if not m.any():
            continue
        cols = list(ACTIONS.values())
        eps.append({"name": p.stem, "n": int(m.sum()), **{k: d[k][m] for k in ENC},
                    "Jlin": d["Jlin"][m], "Jang": d["Jang"][m], "actions": d["actions"][m][:, cols].float(),
                    "V95": d["V95"][m][:, cols].float(), "sentence": d["sentences"][0],
                    "lead": d["lead_to_brake_s"][m]})
    return eps


def cat(eps, idx):
    out = {k: torch.cat([eps[i][k] for i in idx]) for k in ENC + ("Jlin", "Jang", "actions", "V95")}
    out["sentence"] = [eps[i]["sentence"] for i in idx for _ in range(eps[i]["n"])]
    out["episode"] = [eps[i]["name"] for i in idx for _ in range(eps[i]["n"])]
    return out


def build(variant: str, ck: dict, norm: dict):
    from main.lang.config import LangConfig
    from main.lang.legacy_pool import checkpoint_vocab, install_vocab
    from main.lang.network import LangPolicyEncoder, LangTwinCritic
    from main.lang.train import load_policy

    cfg = LangConfig.from_dict(ck["lang_config"])
    vocab = checkpoint_vocab(ck["lang_config"])
    install_vocab(vocab)
    if vocab != cfg.vocab:
        cfg = dataclasses.replace(cfg, vocab=vocab, bank_hash=cfg.bank_hash)
    if variant != "softmin":
        cfg = dataclasses.replace(cfg, critic_readout="mlp",
                                  critic_injection="revised" if variant == "revised" else "linear")
    pe = LangPolicyEncoder(cfg).cuda()
    load_policy(pe, ck)
    critics = LangTwinCritic(cfg).cuda()
    own = critics.state_dict()
    take = {k: v for k, v in ck["critics"].items() if k in own and own[k].shape == v.shape}
    missing = sorted(set(own) - set(take))
    critics.load_state_dict(take, strict=False)
    if variant == "revised":
        critics.set_action_ref(norm["dtheta_max"], norm["s_p"], norm["s_w"])
    return cfg, pe, critics, missing


def rep_enc(enc, K: int):
    """Each state K times, action-major blocks: row k * n + i = state i with action k."""
    n = enc.body.shape[0]
    return enc.index(torch.arange(n, device=enc.body.device).repeat(K))


def q_all(pe, critics, text, D, idx, grad: bool):
    """Q1, Q2 [n, K] for states idx of D at its K actions."""
    g = lambda k: D[k][idx].cuda()  # noqa: E731
    words, wmask = text.lookup([D["sentence"][i] for i in idx.tolist()])
    with torch.set_grad_enabled(grad):
        enc = pe(g("state_tokens").float(), g("body_tokens").float(), g("link_pos").float(), g("body_mask").bool(),
                 g("joint_index").long(), g("scene_tokens").float(), g("scene_pos").float(), words=words,
                 word_mask=wmask)
        n, K = len(idx), D["actions"].shape[1]
        A = g("actions").transpose(0, 1).reshape(n * K, -1)              # action-major
        rep = lambda x: x.repeat(K, *([1] * (x.dim() - 1)))  # noqa: E731
        q1, q2, _, _ = critics(rep_enc(enc, K), A, rep(g("Jlin").float()), rep(g("Jang").float()),
                               rep(g("joint_index").long()))
    return q1.view(K, n).T, q2.view(K, n).T


@torch.no_grad()
def evaluate(pe, critics, text, D, bs: int = 16) -> dict:
    pe.eval(), critics.eval()
    Q = []
    for c0 in range(0, D["V95"].shape[0], bs):
        idx = torch.arange(c0, min(c0 + bs, D["V95"].shape[0]))
        q1, q2 = q_all(pe, critics, text, D, idx, grad=False)
        Q.append(torch.minimum(q1, q2).float().cpu())
    pe.train(), critics.train()
    Q, V = torch.cat(Q).numpy().astype(np.float64), D["V95"].numpy().astype(np.float64)
    ok = n = 0
    K = V.shape[1]
    for i in range(K):
        for j in range(i + 1, K):
            m = np.abs(V[:, i] - V[:, j]) > GT_TOL
            n += int(m.sum())
            ok += int((np.sign(Q[m, i] - Q[m, j]) == np.sign(V[m, i] - V[m, j])).sum())
    qs, vs = Q.max(1) - Q.min(1), V.max(1) - V.min(1)
    return {"n_states": int(V.shape[0]), "n_pairs": n, "rank_agree": ok / n if n else None,
            "q_spread_mean": float(qs.mean()), "true_spread_mean": float(vs.mean()),
            "spread_ratio": float(qs.mean() / max(vs.mean(), 1e-9)), "mae": float(np.abs(Q - V).mean()),
            "q_mean_per_action": Q.mean(0).tolist(), "v_mean_per_action": V.mean(0).tolist()}


def fit(variant: str, fold: int, eps, args, text, ck, norm) -> dict:
    torch.manual_seed(1000 * fold + 7)
    te = [i for i in range(len(eps)) if i % 3 == fold]
    tr = [i for i in range(len(eps)) if i % 3 != fold]
    Dtr, Dte = cat(eps, tr), cat(eps, te)
    cfg, pe, critics, missing = build(variant, ck, norm)
    params = list(pe.parameters()) + list(critics.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    n = Dtr["V95"].shape[0]
    log = [{"step": 0, "train": evaluate(pe, critics, text, Dtr), "held": evaluate(pe, critics, text, Dte)}]
    print(f"[fit] {variant} fold {fold}: {n} train / {Dte['V95'].shape[0]} held-out states; fresh parameters "
          f"{len(missing)} tensors; step 0 train rank {log[0]['train']['rank_agree']} spread ratio "
          f"{log[0]['train']['spread_ratio']:.2e} | held rank {log[0]['held']['rank_agree']} spread ratio "
          f"{log[0]['held']['spread_ratio']:.2e}", flush=True)
    t0 = time.time()
    g = torch.Generator().manual_seed(fold)
    for step in range(1, args.steps + 1):
        idx = torch.randperm(n, generator=g)[: args.batch]
        q1, q2 = q_all(pe, critics, text, Dtr, idx, grad=True)
        y = Dtr["V95"][idx].cuda()
        loss = F.huber_loss(q1, y, delta=args.huber) + F.huber_loss(q2, y, delta=args.huber)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            r = {"step": step, "loss": float(loss), "train": evaluate(pe, critics, text, Dtr),
                 "held": evaluate(pe, critics, text, Dte), "s_per_step": (time.time() - t0) / step}
            log.append(r)
            print(f"[fit] {variant} fold {fold} step {step} loss {r['loss']:.4f} | train rank "
                  f"{r['train']['rank_agree']:.2f} spread {r['train']['q_spread_mean']:.3f} (true "
                  f"{r['train']['true_spread_mean']:.3f}) mae {r['train']['mae']:.3f} | held rank "
                  f"{r['held']['rank_agree']:.2f} ({r['held']['n_pairs']} pairs) spread "
                  f"{r['held']['q_spread_mean']:.3f} (true {r['held']['true_spread_mean']:.3f}) mae "
                  f"{r['held']['mae']:.3f} | {r['s_per_step']:.2f} s/step", flush=True)
    return {"variant": variant, "fold": fold, "train_episodes": [eps[i]["name"] for i in tr],
            "held_episodes": [eps[i]["name"] for i in te], "fresh_tensors": missing, "log": log, "final": log[-1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", default=["softmin", "linear", "revised"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--huber", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    import main.lang  # noqa: F401
    from main.lang.config import LangConfig
    from main.lang.text import TextStore

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    eps = load_states()
    print(f"[fit] {len(eps)} episodes, {sum(e['n'] for e in eps)} drag states, actions {list(ACTIONS)}", flush=True)
    ck = torch.load(CKPT, map_location="cuda", weights_only=False)
    norm = json.loads(NORM.read_text())
    cfg0 = LangConfig.from_dict(ck["lang_config"])
    text = TextStore([c for c in CACHES if c.exists()], cfg0.text_backend, "cuda",
                     extra_path=out / "text_cache_extra.pt", backend_paths=cfg0.backend_paths)
    for v in args.variants:
        for f in args.folds:
            p = out / f"{v}_fold{f}.json"
            if p.exists():
                continue
            r = fit(v, f, eps, args, text, ck, norm)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, indent=1))
            tmp.replace(p)
    # the table
    rows = []
    for v in args.variants:
        fs = [json.loads((out / f"{v}_fold{f}.json").read_text())["final"] for f in args.folds
              if (out / f"{v}_fold{f}.json").exists()]
        if not fs:
            continue
        agg = lambda part, k: float(np.mean([x[part][k] for x in fs]))  # noqa: E731
        rows.append({"variant": v, "folds": len(fs), **{f"{part}_{k}": agg(part, k) for part in ("train", "held")
                                                         for k in ("rank_agree", "q_spread_mean", "true_spread_mean",
                                                                   "spread_ratio", "mae")},
                     "held_pairs": int(sum(x["held"]["n_pairs"] for x in fs))})
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print("\nvariant   | train rank / spread ratio / MAE | held-out rank / spread ratio / MAE (pairs)")
    for r in rows:
        print(f"{r['variant']:9s} | {r['train_rank_agree']:.2f} / {r['train_spread_ratio']:.2f} / "
              f"{r['train_mae']:.3f} | {r['held_rank_agree']:.2f} / {r['held_spread_ratio']:.2f} / "
              f"{r['held_mae']:.3f} ({r['held_pairs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
