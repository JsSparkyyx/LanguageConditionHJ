#!/usr/bin/env python
"""Plan section 8.3 (2026-11-12): two checks on the final checkpoints of the second round, no training.

    python scripts/lang/action_sens/target_checks.py [--names D2 B2 H D3] [--ks 3 12 25 50]
        -> runs/lang/action_sens/checks/check1.json, check2.json (and a printed table)

Check 1. Do the Bellman targets differ between a state's decisions? For every branch state (both collections,
training and held-out apart) and each decision the run scores (plan + the decisions of its length), the decision
row (the branch's row at the branch tick) gives three numbers: the true value V95 (pass 1 of the collector), the
Bellman target y exactly as the run's training step builds it (CriticTrainer._target: its target encoder and
target twins at the end of the decision, the recursion over the decision's rows), and Q (min of the twins, the
stored action or window). Per state the spread (max - min over its decisions) of each, averaged over states;
reported with target / truth and Q / target.

Check 2. Does the critic follow the block after the decision? For every state, its branches' rows k ticks after
the branch tick (all 7 or 9 decisions, the rows that are stored): Q(s_t+k, the stored action or window there) and
the true value from that tick (V95 of the rest of the branch's recorded h, measured_table). Per k: rank agreement
and spread ratio across a state's branches (section4's definitions over the branches present), the share of branch
pairs whose h already differs by more than 0.1 at that tick, and on those separated pairs mean |dQ| / mean |dV|.

Every branch of a state is scored with the state's probe sentence (tall_rules). Both use the run's state.pt (step 6,000; it holds the target networks) and the tall block's rule, the rule the
branch truth is on. TF32 as in the training step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_critic as tc  # noqa: E402
from train_critic import BRANCH, BRANCH2, CHUNK, OUT, T5, EpIndex, chunk_fields, measured_table  # noqa: E402

from main.lang.train import _t, matmul_tf32, roots_of  # noqa: E402

GT_TOL = 0.05
H_SEP = 0.1


def branch_sampler(cfg, seed: int = 0):
    from main.lang.data import LangSampler

    roots = roots_of([str(d / "branches") for d in (BRANCH, BRANCH2)])
    return LangSampler(roots, cfg, tasks=(T5,), embodiments=("ur5-wsg",), seed=seed, n_threads=8)


def tall_rules() -> dict:
    """episode name -> (the tall block's rule id, the probe sentence) from the states files. Every branch of a state
    is scored with that one sentence, as the probes and the scoring do: the sampler draws a random paraphrase per
    row, and Q (and y) would then differ between a state's decisions through the sentence, not the action."""
    out = {}
    for d in (BRANCH, BRANCH2):
        for p in sorted((d / "states").glob("s*.pt")):
            st = torch.load(p, map_location="cpu", weights_only=False)
            out[p.stem] = (st["meta"]["tall_rule"], st["sentences"][0])
    return out


def index_rows(bs, tall: dict) -> dict:
    """(state_id, decision, k) -> {gi, col, held, V95, H, source} for every indexed branch row, k = ticks after the
    branch tick (k = 0: the state row). Terminal rows are not indexed (the sampler drops them)."""
    idx = {}
    for g in range(bs.sid.size):
        j, i = int(bs.sid[g]), int(bs.loc[g])
        s = bs.sources[j]
        eid = int(s.eid[i])
        ln = s.lines.get(eid) or {}
        br = ln.get("branch")
        if br is None:
            continue
        k = int(s.t[i]) - int(ln.get("n_ctx", 3))
        if k < 0:
            continue
        rule, sent = tall[br["episode"]]
        col = ln["rule_ids"].index(rule)
        idx[(br["state_id"], br["decision"], k)] = {"gi": g, "col": col, "sentence": sent, "held": bool(br["held_out"]),
                                                    "V95": float(br["V95"]), "H": int(br["H"]), "source": s.name}
    return idx


def load_run(name: str):
    tr = tc.make_trainer(name, OUT / "score_tmp" / f"checks_{name}", 0, with_batcher=False)
    st = OUT / "runs" / name / "state.pt"
    tr.load_state(st)
    for m in (tr.policy_enc, tr.policy_enc_targ, tr.critics, tr.critics_targ):
        m.eval()
    print(f"[checks] {name}: decision {tr.H} tick(s), step {tr.step} from {st}", flush=True)
    return tr


@torch.no_grad()
def q_and_y(tr, b, want_y: bool):
    """Q (min of the twins) at the batch rows with the stored action / window, and the training target y."""
    d = tr.dev
    words, wmask = tr.text.lookup(b["sentence"], splits=["probe"] * len(b["sentence"]))
    with matmul_tf32(getattr(tr.cfg, "tf32_train", False)):
        enc, Jl, Ja, ji = tr.encode(b, words, wmask)
        if tr.H == 1:
            act = _t(b["dtheta"], dev=d)
        else:
            cm = _t(b["chunk_cmds"], dev=d)[:, : tr.H]
            act = tr.act_of(cm)
        q1, q2, _, _ = tr.critics(enc, act, Jl, Ja, ji)
        q = torch.minimum(q1, q2).float().cpu().numpy()
        y = tr._target(b, words, wmask, _t(b["h"], dev=d)).float().cpu().numpy() if want_y else None
    return q, y


def spread_stats(V, Y, Q):
    """Per-state arrays (lists of [D]) -> spreads and ratios (check 1)."""
    sv = np.array([v.max() - v.min() for v in V])
    sy = np.array([y.max() - y.min() for y in Y])
    sq = np.array([q.max() - q.min() for q in Q])

    def rank(a, b):
        ok = n = 0
        for x, t in zip(a, b):
            for i in range(len(t)):
                for j in range(i + 1, len(t)):
                    if abs(t[i] - t[j]) > GT_TOL:
                        n += 1
                        ok += int(np.sign(x[i] - x[j]) == np.sign(t[i] - t[j]))
        return (ok / n if n else None), n

    ry, n = rank(Y, V)
    rq, _ = rank(Q, V)
    return {"n_states": len(V), "n_pairs": n, "truth_spread": float(sv.mean()), "target_spread": float(sy.mean()),
            "q_spread": float(sq.mean()), "target_over_truth": float(sy.mean() / max(sv.mean(), 1e-9)),
            "q_over_target": float(sq.mean() / max(sy.mean(), 1e-9)),
            "q_over_truth": float(sq.mean() / max(sv.mean(), 1e-9)),
            "rank_target_vs_truth": ry, "rank_q_vs_truth": rq}


def check1(tr, bs, idx, eidx, rng, bs_n: int = 32) -> dict:
    names = [n for n, H, _ in DEC if H == 0 or H == tr.H]
    states = sorted({k[0] for k in idx if k[2] == 0})
    rows = []                         # (state, decision, entry)
    for st in states:
        ent = [(st, n, idx.get((st, n, 0))) for n in names]
        if all(e is not None for _, _, e in ent):
            rows += ent
    vals = {}
    for c0 in range(0, len(rows), bs_n):
        part = rows[c0:c0 + bs_n]
        picks = [(e["gi"], e["col"], -1) for _, _, e in part]
        b = bs.sample(len(picks), rng=rng, picks=picks)
        b["sentence"] = [e["sentence"] for _, _, e in part]
        if tr.H > 1:
            chunk_fields(bs, picks, b, tr.H, eidx, rng, need_end=True)
        q, y = q_and_y(tr, b, True)
        for (st, n, e), qq, yy in zip(part, q, y):
            vals.setdefault(st, {"held": e["held"]})[n] = (e["V95"], float(yy), float(qq))
    out = {}
    for split, held in (("held", True), ("train", False)):
        S = [v for v in vals.values() if v["held"] == held]
        V = [np.array([s[n][0] for n in names]) for s in S]
        Y = [np.array([s[n][1] for n in names]) for s in S]
        Q = [np.array([s[n][2] for n in names]) for s in S]
        out[split] = spread_stats(V, Y, Q) if S else None
        if S:
            out[split]["mean_per_decision"] = {n: {"V95": float(np.mean([s[n][0] for s in S])),
                                                   "y": float(np.mean([s[n][1] for s in S])),
                                                   "Q": float(np.mean([s[n][2] for s in S]))} for n in names}
    out["decisions"] = names
    return out


def check2(trs: dict, bs, idx, eidx, meas: dict, ks, rng, bs_n: int = 48) -> dict:
    names = [n for n, _, _ in DEC]
    keys = [key for key in idx if key[2] in ks]
    rec = {name: {} for name in trs}             # run -> (state, k) -> {decision: (q, v, h)}
    t0 = time.time()
    for c0 in range(0, len(keys), bs_n):
        part = keys[c0:c0 + bs_n]
        ents = [idx[k] for k in part]
        picks = [(e["gi"], e["col"], -1) for e in ents]
        b = bs.sample(len(picks), rng=rng, picks=picks)
        b["sentence"] = [e["sentence"] for e in ents]
        chunk_fields(bs, picks, b, CHUNK, eidx, rng, need_end=False)
        v = np.array([meas[e["source"]][int(r), e["col"]] for e, r in zip(ents, b["row"])])
        h = np.asarray(b["h"])
        for name, tr in trs.items():
            q, _ = q_and_y(tr, b, False)
            for (st, dn, k), qq, vv, hh in zip(part, q, v, h):
                rec[name].setdefault((st, k), {})[dn] = (float(qq), float(vv), float(hh))
        if (c0 // bs_n) % 50 == 0:
            print(f"[checks] check 2: {c0 + len(part)} of {len(keys)} rows ({time.time() - t0:.0f} s)", flush=True)
    held_of = {k[0]: e["held"] for k, e in idx.items()}
    out = {}
    for name in trs:
        out[name] = {}
        for split, held in (("held", True), ("train", False)):
            out[name][split] = {}
            for k in ks:
                groups = [d for (st, kk), d in rec[name].items() if kk == k and held_of[st] == held and len(d) >= 2]
                ok = n = nsep = npair = 0
                qs, vs, dq_sep, dv_sep = [], [], [], []
                for d in groups:
                    arr = np.array([d[x] for x in names if x in d])          # [n_br, 3] (q, v, h)
                    q, v, hh = arr[:, 0], arr[:, 1], arr[:, 2]
                    qs.append(q.max() - q.min()), vs.append(v.max() - v.min())
                    for i in range(len(v)):
                        for j in range(i + 1, len(v)):
                            npair += 1
                            if abs(hh[i] - hh[j]) > H_SEP:
                                nsep += 1
                                dq_sep.append(abs(q[i] - q[j])), dv_sep.append(abs(v[i] - v[j]))
                            if abs(v[i] - v[j]) > GT_TOL:
                                n += 1
                                ok += int(np.sign(q[i] - q[j]) == np.sign(v[i] - v[j]))
                out[name][split][str(k)] = {
                    "n_states": len(groups), "n_branches": int(sum(len(d) for d in groups)), "n_pairs": n,
                    "rank_agree": (ok / n) if n else None,
                    "spread_ratio": float(np.mean(qs) / max(np.mean(vs), 1e-9)) if groups else None,
                    "q_spread": float(np.mean(qs)) if groups else None, "true_spread": float(np.mean(vs)) if groups else None,
                    "h_separated_frac": (nsep / npair) if npair else None,
                    "sep_dq_over_dv": float(np.mean(dq_sep) / max(np.mean(dv_sep), 1e-9)) if dq_sep else None}
    return out


# collect_branches.DECISIONS (copied: importing the collector pulls in the simulator): (name, H ticks, speed scale c)
DEC = [("plan", 0, 1.0), ("t1_c0", 1, 0.0), ("t1_c50", 1, 0.5), ("ch_c0", CHUNK, 0.0), ("ch_c25", CHUNK, 0.25),
       ("ch_c50", CHUNK, 0.5), ("ch_c75", CHUNK, 0.75), ("t3_c0", 3, 0.0), ("t3_c50", 3, 0.5)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--names", nargs="+", default=["D2", "B2", "H", "D3"])
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 12, 25, 50])
    ap.add_argument("--skip", nargs="*", default=[], help="check1 / check2")
    args = ap.parse_args(argv)
    outdir = OUT / "checks"
    outdir.mkdir(parents=True, exist_ok=True)
    trs = {n: load_run(n) for n in args.names}
    tr0 = next(iter(trs.values()))
    bs = branch_sampler(tr0.cfg)
    for tr in trs.values():
        tr.add_source_index(bs)
    tall = tall_rules()
    idx = index_rows(bs, tall)
    eidx = {id(s): EpIndex(s) for s in bs.sources}
    meas = {s.name: measured_table(s, tr0.gamma(), None) for s in bs.sources}
    print(f"[checks] {len(idx)} indexed branch rows, {len({k[0] for k in idx})} states", flush=True)
    rng = np.random.default_rng(0)
    if "check1" not in args.skip:
        c1 = {}
        for name, tr in trs.items():
            c1[name] = check1(tr, bs, idx, eidx, rng)
            for split in ("held", "train"):
                m = c1[name][split]
                print(f"[check1] {name} {split}: {m['n_states']} states; spread truth {m['truth_spread']:.3f} target "
                      f"{m['target_spread']:.3f} Q {m['q_spread']:.4f} | target/truth {m['target_over_truth']:.3f} "
                      f"Q/target {m['q_over_target']:.3f} | rank target {m['rank_target_vs_truth']} Q "
                      f"{m['rank_q_vs_truth']} ({m['n_pairs']} pairs)", flush=True)
        (outdir / "check1.json").write_text(json.dumps(c1, indent=1))
    if "check2" not in args.skip:
        c2 = check2(trs, bs, idx, eidx, meas, args.ks, rng)
        for name in trs:
            for split in ("held", "train"):
                for k in args.ks:
                    m = c2[name][split][str(k)]
                    print(f"[check2] {name} {split} k={k}: {m['n_states']} states {m['n_branches']} branches; rank "
                          f"{m['rank_agree']} ({m['n_pairs']} pairs) spread ratio {m['spread_ratio']} | h separated "
                          f"{m['h_separated_frac']} | separated |dQ|/|dV| {m['sep_dq_over_dv']}", flush=True)
        (outdir / "check2.json").write_text(json.dumps(c2, indent=1))
    print("[checks] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
