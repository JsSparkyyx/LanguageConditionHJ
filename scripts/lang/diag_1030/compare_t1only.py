#!/usr/bin/env python
"""Check 1 side by side, T1-only run vs five-task run (11-02: is the action blindness from training many tasks in one
model?). Reads actor_stats.py results.json files; numpy only, safe on the login node.

    python scripts/lang/diag_1030/compare_t1only.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
A = REPO / "runs/lang/diag_1030/actor_stats"
SETS = [("T1-only", A / "t1only/results.json"), ("five-task", A / "vanilla_t1states/results.json"),
        ("five-task (10-30, 32 states)", A / "vanilla/results.json")]
GROUPS = ["plan_selective_avoid", "actor_selective_avoid", "train_mix"]


def g(d, *path, stat="median"):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    if isinstance(d, dict):
        return d.get(stat)
    return d


def f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float) and (abs(x) < 1e-2 or abs(x) >= 1e4) and x != 0:
        return f"{x:.1e}"
    return f"{x:.{nd}g}"


ROWS = [
    ("n states", lambda G: G.get("n_states")),
    ("|tanh mu| med (share of dtheta_max)", lambda G: g(G, "abs_tanh_mu_eq_abs_adet_over_dmax")),
    ("sigma med", lambda G: g(G, "sigma")),
    ("entropy med (max 8.204)", lambda G: g(G, "entropy_analytic_sum")),
    ("Q spread, 16 random actions, med", lambda G: g(G, "spread_16_random_abs")),
    ("Q spread, 16 random actions, p90", lambda G: g(G, "spread_16_random_abs", stat="p90")),
    ("Q spread over c*a_ref, med", lambda G: g(G, "spread_over_c_abs")),
    ("|Q(stop) - Q(half)| > TF32 noise", lambda G: g(G, "ranking", "frac_abs_Q0_minus_Qhalf_above_tf32_noise_p90")),
    ("TF32 noise p90", lambda G: g(G, "tf32_minus_fp32_abs", stat="p90")),
    ("readout min on a query head", lambda G: g(G, "trace", "c1_softmin_argmin_is_query_frac")),
    ("lowest link head - Q, med", lambda G: g(G, "trace", "c1_V_body_min_minus_Q")),
    ("body token norm, med", lambda G: g(G, "trace", "c1_body_norm_mean")),
    ("action part of body token (sat), med", lambda G: g(G, "trace", "c1_sat_inj_delta_abs")),
    ("Q(stored) < 0 share", lambda G: G.get("frac_Q_stored_below_0")),
    ("h of named rule, mean", lambda G: g(G, "h_named", stat="mean")),
]


def main() -> int:
    R = {}
    for name, p in SETS:
        if p.exists():
            R[name] = json.loads(p.read_text())
        else:
            print(f"(missing {p})", file=sys.stderr)
    # 1. final checkpoints, T1 groups, both models
    for grp in GROUPS:
        cols = [(n, r["checkpoints"]["checkpoint"]["groups"].get(grp)) for n, r in R.items()
                if "checkpoint" in r["checkpoints"] and grp in r["checkpoints"]["checkpoint"]["groups"]]
        if not cols:
            continue
        print(f"\n### {grp}, final checkpoints\n")
        print("| | " + " | ".join(n for n, _ in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for lab, fn in ROWS:
            print(f"| {lab} | " + " | ".join(f(fn(G)) for _, G in cols) + " |")
    # 2. over training, T1-only run
    for name in ("T1-only", "five-task (10-30, 32 states)"):
        if name not in R:
            continue
        cks = R[name]["checkpoints"]
        for grp in GROUPS[:2]:
            cols = [(c, v["groups"].get(grp), v.get("step")) for c, v in cks.items() if grp in v.get("groups", {})]
            if not cols:
                continue
            print(f"\n### {name}, {grp}, over training\n")
            print("| | " + " | ".join(f"{c} (step {s})" for c, _, s in cols) + " |")
            print("|---|" + "---|" * len(cols))
            for lab, fn in ROWS[1:13]:
                print(f"| {lab} | " + " | ".join(f(fn(G)) for _, G, _ in cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
