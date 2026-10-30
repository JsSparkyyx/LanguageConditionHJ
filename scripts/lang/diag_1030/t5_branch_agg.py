#!/usr/bin/env python
"""Aggregate the T5 branch episodes (t5_branch.py run) into results.json and tables (numpy + json only, no torch).

    python scripts/lang/diag_1030/t5_branch_agg.py [--out runs/lang/diag_1030/t5_branch]

Truths per (branch tick, first action X in plan / stop / half / slow), all on the tall block's upright h:
  T_full   the four full branches; rep 0 (exact restore) and 3 perturbed repeats (a tiny kick to the tall block),
           summarized by the mean over the 4 and the tip fraction (min h < 0)
  T_det    X's first tick, then the final checkpoint's deterministic actor
  T_sto    X's first tick, then its stochastic actor (mean over 4 seeds, and their spread)
V_g = the backward recursion y(t) = (1 - g) h_t + g min(h_t, y(t+1)), y(T) = h_T on the branch's h trace.
Critic: Q(s, a) of each vanilla checkpoint (min of the twins), tall sentence, at the four first actions
(a_nom, 0, half_cmd = B_half's first-tick command, slow_cmd = B_slow's first-tick command).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs/lang/diag_1030/t5_branch"
X4 = ("plan", "stop", "half", "slow")
CAND = {"plan": "a_nom", "stop": "zero", "half": "half_cmd", "slow": "slow_cmd"}
TICK = 0.04
TOL = 0.05       # truth differences below this (h units) are ties: not counted in the pairwise agreement


def vrec(h, g):
    h = np.asarray(h, np.float64)
    y = np.empty_like(h)
    y[-1] = h[-1]
    for t in range(h.size - 2, -1, -1):
        y[t] = (1 - g) * h[t] + g * min(h[t], y[t + 1])
    return y


def pair_agree(q, g, tol=TOL):
    ok = n = 0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            d = g[i] - g[j]
            if not np.isfinite(d) or abs(d) <= tol:
                continue
            n += 1
            ok += int(np.sign(q[i] - q[j]) == np.sign(d))
    return ok, n


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rows_of(path: Path):
    d = json.loads(path.read_text())
    meta = d["meta"]
    ckpts = [c[0] for c in meta["ckpts"]]
    out = []
    for bp in d["branch_points"]:
        if "branches" not in bp:
            continue
        B = bp["branches"]
        sc = bp["score"]
        names = sc["names"]
        r = {"episode": path.parent.name, "seed": meta["seed"], "ts": meta["time_scale"], "kind": bp["kind"],
             "j": bp["j"], "t": bp["t"], "lead_brake": bp.get("lead_to_brake_s"),
             "nominal_tipped": d["nominal"]["tipped"]}
        fp = B["plan"].get("first_neg_tick")
        r["lead_plan_harm"] = fp * TICK if fp is not None else None
        for X in X4 + ("pause3", "plan_live", "plan2"):
            if X not in B:
                continue
            reps = [B[X]] + [B[k] for k in sorted(B) if k.startswith(f"{X}~p")] if X in X4 else [B[X]]
            mh = np.array([b["min_h_tall"] for b in reps])
            v = np.array([b["V95"] for b in reps])
            r[X] = {"min_h0": float(mh[0]), "V95_0": float(v[0]), "min_h": float(mh.mean()), "V95": float(v.mean()),
                    "V95_sd": float(v.std()), "tip": float((mh < 0).mean()), "n": int(mh.size),
                    "V98": float(np.mean([b["V98"] for b in reps])), "V99": float(np.mean([b["V99"] for b in reps])),
                    "ticks": [b["n_ticks"] for b in reps], "peak_decel": float(np.mean([b["peak_decel"] for b in reps])),
                    "success": float(np.mean([b["success"] for b in reps])),
                    "min_h_short": float(np.mean([b["min_h_short"] for b in reps]))}
        for X in X4:
            if f"{X}_det" in B:
                b = B[f"{X}_det"]
                r[f"{X}_det"] = {"min_h": b["min_h_tall"], "V95": b["V95"], "ticks": b["n_ticks"],
                                 "actor_abs_frac": b.get("actor_abs_frac_mean"), "tip": float(b["min_h_tall"] < 0)}
                st = [B[k] for k in sorted(B) if k.startswith(f"{X}_sto")]
                v = np.array([b["V95"] for b in st])
                mh = np.array([b["min_h_tall"] for b in st])
                r[f"{X}_sto"] = {"min_h": float(mh.mean()), "V95": float(v.mean()), "V95_sd": float(v.std()),
                                 "tip": float((mh < 0).mean()), "n": int(v.size),
                                 "actor_abs_frac": float(np.mean([b.get("actor_abs_frac_mean") or np.nan for b in st]))}
        r["Q"] = {}
        for c in ckpts:
            for key in ("tall", "short"):
                s = sc.get(f"{c}|{key}")
                if not s:
                    continue
                q = dict(zip(names, s["qmin"]))
                r["Q"][f"{c}|{key}"] = {"X": {X: float(q[CAND[X]]) for X in X4}, "half_lit": float(q["half_lit"]),
                                        "det": float(s["q_det"]), "samples": float(np.mean(s["q_samples"])),
                                        "rand_spread": float(np.max(s["q_rand"]) - np.min(s["q_rand"])),
                                        "rand_mean": float(np.mean(s["q_rand"])),
                                        "abs_tanh_mu": s.get("abs_tanh_mu"),
                                        "sigma_mean": float(np.mean(s["sigma"])) if "sigma" in s else None}
        f = bp["fidelity"]
        r["fid"] = {k: f.get(k) for k in ("h_tall", "tilt_tall", "tray_p", "h_tall_tick1", "tray_p_tick1",
                                          "live_h_tall", "live_tilt_tall", "live_tray_p_tick1", "img_ab_maxabs",
                                          "img_ac_maxabs", "img_cd_maxabs", "img_ad_maxabs", "img_ac_meanabs",
                                          "depth_ac_maxabs", "tok_ab_maxabs", "tok_ac_maxabs", "tok_scale")}
        r["fid"]["state_check_max"] = max(max(v.values()) for v in f["state_checks"].values())
        r["fid"]["vs_real"] = f.get("vs_real")
        out.append(r)
    return d, out, ckpts


def nominal_curve(d, ckpts):
    """Q(s, a_nom) along the nominal episode vs the true V95 of the planner's own continuation (the real episode's
    h_tall at the captures, recursion from the end), at the hook ticks."""
    caps = d["caps"]
    h = np.array([c["h_tall"] for c in caps])
    tc = np.array([c["t"] for c in caps])
    v95 = vrec(h, 0.95) if h.size else h
    ticks = [t for t in d["ticks"] if "score" in t]
    out = {"t_first_h_neg": float(tc[np.argmax(h < 0)]) if (h < 0).any() else None,
           "t_first_V95_neg": float(tc[np.argmax(v95 < 0)]) if (v95 < 0).any() else None}
    for c in ckpts:
        q = np.array([t["score"][f"{c}|tall"]["q_nom"] for t in ticks])
        tt = np.array([t["t"] for t in ticks])
        neg = np.nonzero(q < 0)[0]
        indrag = np.array([t["in_drag"] for t in ticks], bool)
        t0 = float(tt[indrag][0]) if indrag.any() else None
        pre = tt < t0 if t0 is not None else np.ones_like(tt, bool)
        negd = np.nonzero((q < 0) & ~pre)[0]
        out[c] = {"t_first_Q_neg": float(tt[negd[0]]) if negd.size else None,   # from the drag start on
                  "t_first_Q_neg_any": float(tt[neg[0]]) if neg.size else None,
                  "frac_Q_neg_predrag": float((q[pre] < 0).mean()) if pre.any() else None,
                  "frac_Q_neg": float((q < 0).mean()) if q.size else None}
        # Q vs true V95 at the hook ticks (the capture just before each hook)
        vi = np.interp(tt, tc, v95) if tc.size else tt * np.nan
        out[c]["mae_vs_V95"] = float(np.mean(np.abs(q - vi))) if q.size else None
        out[c]["signed_vs_V95"] = float(np.mean(q - vi)) if q.size else None
        dr = [t for t in ticks if t["in_drag"]]
        if dr:
            qd = np.array([t["score"][f"{c}|tall"]["q_nom"] for t in dr])
            out[c]["drag_q_nom_first_last_min"] = [float(qd[0]), float(qd[-1]), float(qd.min())]
            mu = np.array([t["score"][f"{c}|tall"]["abs_tanh_mu"] for t in dr])
            sg = np.array([np.mean(t["score"][f"{c}|tall"]["sigma"]) for t in dr])
            out[c]["drag_abs_tanh_mu_mean"] = float(mu.mean())
            out[c]["drag_sigma_mean"] = float(sg.mean())
            out[c]["drag_q_spread_nom_det_zero"] = float(np.mean([max(t["score"][f"{c}|tall"][k] for k in ("q_nom", "q_det", "q_zero")) - min(t["score"][f"{c}|tall"][k] for k in ("q_nom", "q_det", "q_zero")) for t in dr]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    paths = sorted((out / "episodes").glob("*/result.json"))
    R = {"episodes": [], "rows": [], "summary": {}}
    ckpts = None
    for p in paths:
        d, rows, ckpts = rows_of(p)
        meta = d["meta"]
        R["episodes"].append({"name": p.parent.name, "seed": meta["seed"], "time_scale": meta["time_scale"],
                              "cruise_speed": meta["cruise_speed"], "status": d["status"], "success": d["success"],
                              "nominal": d["nominal"], "wall_s": d.get("wall_s"), "gpu_mem_max_gb": d.get("gpu_mem_max_gb"),
                              "timing": d.get("timing"), "rt": meta.get("rt"),
                              "items": meta.get("items"), "sentence_tall": meta.get("sentence_tall"),
                              "sentence_short": meta.get("sentence_short"),
                              "drag": {k: v for k, v in (d.get("drag") or {}).items() if k != "plan_id"},
                              "nominal_curve": nominal_curve(d, ckpts)})
        R["rows"] += rows
    rows = R["rows"]
    drag = [r for r in rows if r["kind"] == "drag"]
    S = R["summary"]
    S["n_episodes"], S["n_branch_ticks"], S["n_drag_ticks"] = len(paths), len(rows), len(drag)
    S["n_actor_points"] = sum(1 for r in drag if "plan_det" in r)
    # 1. ranking of the four first actions: Q vs T_full (mean over the repeats)
    rank = {}
    for c in ckpts or []:
        for key in ("tall", "short"):
            k = f"{c}|{key}"
            for truth in ("V95", "min_h"):
                ok = n = 0
                spreads, rho = [], []
                for r in drag:
                    if k not in r["Q"]:
                        continue
                    q = [r["Q"][k]["X"][X] for X in X4]
                    g = [r[X][truth] for X in X4]
                    a, b = pair_agree(q, g)
                    ok += a
                    n += b
                    spreads.append(max(q) - min(q))
                    rho.append(spearman(q, g))
                rank[f"{k}|{truth}"] = {"pair_agree": ok / n if n else None, "n_pairs": n, "n_ticks": len(spreads),
                                        "Q_spread_4_mean": float(np.mean(spreads)) if spreads else None,
                                        "Q_spread_4_max": float(np.max(spreads)) if spreads else None,
                                        "spearman_mean": float(np.nanmean(rho)) if np.isfinite(rho).any() else None}
    S["ranking_vs_full"] = rank
    # the truth's own spread over the four first actions
    S["truth_spread_V95"] = {"mean": float(np.mean([max(r[X]["V95"] for X in X4) - min(r[X]["V95"] for X in X4)
                                                   for r in drag])) if drag else None}
    # 2. truth: which first action is best, and the last lead at which some branch saves the block
    best = {X: 0 for X in X4}
    for r in drag:
        best[max(X4, key=lambda X: r[X]["V95"])] += 1
    S["best_by_V95_count"] = best
    save = []
    for ep in sorted({r["episode"] for r in drag}):
        er = sorted([r for r in drag if r["episode"] == ep], key=lambda r: r["j"])
        if not er or not any(r["plan"]["tip"] > 0 for r in er):
            continue
        saved = [r for r in er if r["plan"]["tip"] > 0 and any(r[X]["tip"] == 0 for X in ("stop", "half", "slow"))]
        lost = [r for r in er if all(r[X]["tip"] > 0 for X in X4)]
        save.append({"episode": ep, "ts": er[0]["ts"],
                     "last_saving_t": max((r["t"] for r in saved), default=None),
                     "first_lost_t": min((r["t"] for r in lost), default=None),
                     "last_saving_lead_brake": min((r["lead_brake"] for r in saved), default=None),
                     "last_saving_lead_plan_harm": min((r["lead_plan_harm"] for r in saved if r["lead_plan_harm"] is not None), default=None),
                     "first_lost_lead_brake": max((r["lead_brake"] for r in lost), default=None),
                     "per_tick": [{"j": r["j"], "lead_brake": r["lead_brake"], "lead_plan_harm": r["lead_plan_harm"],
                                   **{X: (round(r[X]["tip"], 2), round(r[X]["V95"], 3)) for X in X4}} for r in er]})
    S["last_saving"] = save
    # 3. Q vs the three truths at the actor points (V95)
    fit = {}
    for c in ckpts or []:
        k = f"{c}|tall"
        for truth in ("full", "det", "sto"):
            err, ok, n, rho = [], 0, 0, []
            for r in drag:
                if "plan_det" not in r or k not in r["Q"]:
                    continue
                q = [r["Q"][k]["X"][X] for X in X4]
                g = [r[X]["V95"] if truth == "full" else r[f"{X}_{truth}"]["V95"] for X in X4]
                err += [a - b for a, b in zip(q, g)]
                a_, b_ = pair_agree(q, g)
                ok += a_
                n += b_
                rho.append(spearman(q, g))
            err = np.asarray(err)
            fit[f"{c}|{truth}"] = {"n": int(err.size), "mae": float(np.abs(err).mean()) if err.size else None,
                                   "signed": float(err.mean()) if err.size else None,
                                   "pair_agree": ok / n if n else None, "n_pairs": n,
                                   "spearman_mean": float(np.nanmean(rho)) if np.isfinite(rho).any() else None}
    S["q_vs_truths_V95"] = fit
    # over every drag tick: Q at each first action vs that action's own full-branch V95, and vs the PLANNER's
    # full-branch V95 (an action-blind Q fit to planner data should match the planner's value for every action)
    own = {}
    for c in ckpts or []:
        k = f"{c}|tall"
        for X in X4:
            e_own = np.array([r["Q"][k]["X"][X] - r[X]["V95"] for r in drag if k in r["Q"]])
            e_pl = np.array([r["Q"][k]["X"][X] - r["plan"]["V95"] for r in drag if k in r["Q"]])
            own[f"{c}|{X}"] = {"n": int(e_own.size), "mae_own": float(np.abs(e_own).mean()) if e_own.size else None,
                               "signed_own": float(e_own.mean()) if e_own.size else None,
                               "mae_vs_plan": float(np.abs(e_pl).mean()) if e_pl.size else None,
                               "signed_vs_plan": float(e_pl.mean()) if e_pl.size else None}
        rs = [r["Q"][k]["rand_spread"] for r in drag if k in r["Q"]]
        sm = [abs(r["Q"][k]["samples"] - r["Q"][k]["det"]) for r in drag if k in r["Q"]]
        own[f"{c}|rand16_spread_mean"] = float(np.mean(rs)) if rs else None
        own[f"{c}|samples_vs_det_absdiff_mean"] = float(np.mean(sm)) if sm else None
        own[f"{c}|abs_tanh_mu_mean"] = float(np.mean([r["Q"][k]["abs_tanh_mu"] for r in drag if k in r["Q"]]))
        own[f"{c}|sigma_mean"] = float(np.mean([r["Q"][k]["sigma_mean"] for r in drag if k in r["Q"]]))
    S["q_own_vs_plan"] = own
    # 4. fidelity
    fids = [r["fid"] for r in rows]

    def mx(k):
        v = [f[k] for f in fids if f.get(k) is not None]
        return float(np.max(v)) if v else None

    S["fidelity"] = {k: mx(k) for k in ("state_check_max", "h_tall", "tilt_tall", "tray_p", "h_tall_tick1",
                                        "tray_p_tick1", "live_h_tall", "live_tilt_tall", "live_tray_p_tick1",
                                        "img_ab_maxabs", "img_ac_maxabs", "img_cd_maxabs", "img_ad_maxabs",
                                        "depth_ac_maxabs", "tok_ab_maxabs", "tok_ac_maxabs")}
    vr = [f["vs_real"] for f in fids if f.get("vs_real")]
    S["fidelity"]["vs_real_max"] = {k: float(np.max([v[k] for v in vr if v.get(k) is not None]))
                                    for k in (vr[0] if vr else {}) if k != "n_ticks"}
    S["fidelity"]["vs_real_n_ticks"] = [v["n_ticks"] for v in vr]
    # 5. nominal curves
    S["nominal"] = [{"name": e["name"], "ts": e["time_scale"], "tipped": e["nominal"]["tipped"],
                     "min_h_tall": e["nominal"]["min_h_tall"], "drag_start_t": (e["drag"] or {}).get("start_t"),
                     "brake_t": ((e["drag"] or {}).get("start_t") or 0) + ((e["drag"] or {}).get("brake_row") or 0) / 250.0,
                     **e["nominal_curve"]} for e in R["episodes"]]
    ls = {x["episode"]: x for x in save}
    for x in S["nominal"]:
        y = ls.get(x["name"], {})
        x["last_saving_t"], x["first_lost_t"] = y.get("last_saving_t"), y.get("first_lost_t")
        for c in ckpts or []:
            tq = (x.get(c) or {}).get("t_first_Q_neg")
            if tq is not None and x["last_saving_t"] is not None:
                x[c]["Q_neg_minus_last_saving_s"] = tq - x["last_saving_t"]
    # per-branch-tick table (markdown)
    fin = (ckpts or ["checkpoint"])[-1]
    lines = ["| episode | ts | j | lead brake s | lead plan harm s | plan tip / V95 | stop | half | slow | pause3 | "
             f"Q {fin} a_nom | Q spread 4 | Q r10 a_nom | T_det V95 p/s/h/sl | T_sto V95 p/s/h/sl |",
             "|" + "---|" * 15]
    for r in sorted(drag, key=lambda r: (r["episode"], r["j"])):
        def tv(X):
            return f"{r[X]['tip']:.2f} / {r[X]['V95']:.2f}" if X in r else "-"
        qf = r["Q"].get(f"{fin}|tall")
        q10 = r["Q"].get(f"checkpoint_round10|tall")
        sp = (max(qf["X"].values()) - min(qf["X"].values())) if qf else float("nan")
        det = "/".join(f"{r[f'{X}_det']['V95']:.2f}" for X in X4) if "plan_det" in r else ""
        sto = "/".join(f"{r[f'{X}_sto']['V95']:.2f}" for X in X4) if "plan_sto" in r else ""
        lph = f"{r['lead_plan_harm']:.2f}" if r["lead_plan_harm"] is not None else "none"
        lines.append(f"| {r['episode']} | {r['ts']:g} | {r['j']} | {r['lead_brake']:.2f} | {lph} | {tv('plan')} | "
                     f"{tv('stop')} | {tv('half')} | {tv('slow')} | {tv('pause3')} | "
                     f"{qf['X']['plan']:.3f} | {sp:.1e} | {q10['X']['plan']:.3f} | {det} | {sto} |")
    (out / "branch_ticks.md").write_text("\n".join(lines) + "\n")
    # binned by lead time to the planner's brake: tip rate (over ticks x 4 repeats) and mean V95 per first action,
    # the final checkpoint's Q(s, a_nom), separately for drag ticks where the planner's branch tips and where not
    bins = [(0.6, 9), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2), (-9, 0.0)]
    tab = {}
    blines = ["| planner branch | lead to brake s | n ticks | plan tip / V95 | stop | half | slow | pause3 | "
              f"mean Q {fin}(s, a_nom) | mean Q r10 |", "|" + "---|" * 10]
    for grp, sel in (("tips", lambda r: r["plan"]["tip"] > 0), ("no tip", lambda r: r["plan"]["tip"] == 0)):
        for lo, hi in bins:
            rr = [r for r in drag if sel(r) and lo <= r["lead_brake"] < hi]
            if not rr:
                continue
            e = {X: (float(np.mean([r[X]["tip"] for r in rr])), float(np.mean([r[X]["V95"] for r in rr])))
                 for X in X4 + ("pause3",)}
            qf = float(np.mean([r["Q"][f"{fin}|tall"]["X"]["plan"] for r in rr]))
            q10 = float(np.mean([r["Q"]["checkpoint_round10|tall"]["X"]["plan"] for r in rr])) \
                if "checkpoint_round10|tall" in rr[0]["Q"] else float("nan")
            tab[f"{grp}|{lo}..{hi}"] = {"n": len(rr), **{X: e[X] for X in e}, "Q_final": qf, "Q_r10": q10}
            lab = f">= {lo}" if hi == 9 else (f"< {hi}" if lo == -9 else f"{lo}-{hi}")
            blines.append(f"| {grp} | {lab} | {len(rr)} | " + " | ".join(f"{e[X][0]:.2f} / {e[X][1]:.2f}" for X in e)
                          + f" | {qf:.2f} | {q10:.2f} |")
    S["by_lead"] = tab
    (out / "by_lead.md").write_text("\n".join(blines) + "\n")
    tmp = out / ".results.json.tmp"
    tmp.write_text(json.dumps(R, indent=1, default=float))
    tmp.replace(out / "results.json")
    print(json.dumps({k: v for k, v in S.items() if k not in ("last_saving", "nominal")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
