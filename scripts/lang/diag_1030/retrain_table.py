#!/usr/bin/env python
"""Check 4: tabulate the critic-retrain variants' diag.jsonl over steps (json + numpy only; login-node safe).

    python scripts/lang/diag_1030/retrain_table.py [--root runs/lang/diag_1030/retrain] [--steps 250 1000 ...]

Writes <root>/results.json: per variant and step the key probe metrics, the original run's logged
critic/action_sens_rel / entropy at the same steps (cross-check), and the verdict by the pre-registered criteria
(<root>/criteria.json) at the latest step every variant reached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VARIANTS = ("stoch", "det", "data", "zero", "stoch_norm")
EXPLORATORY = ("stoch_mlp",)   # added after the smoke, not in the pre-registered verdict
KEYS = ("all/spread16", "val_t5drag/spread16", "train_t5drag/spread16", "all/sens_rel", "all/sens_abs",
        "val_planner/q0_minus_qnom", "val_planner/qhalf_minus_qnom", "val_planner/slope_c_planner",
        "val_planner/gnorm_planner", "val_t5drag/q0_minus_qnom", "val_t5drag/qhalf_minus_qnom",
        "train_t5drag/q0_minus_qnom", "all/gnorm", "all/q_mean", "all/h_mean", "val/pair_rank_acc",
        "train/pair_rank_acc", "all/actor_u_abs", "all/actor_sigma", "all/actor_cap_frac", "all/actor_a_frac",
        "batch/loss/q", "batch/critic/action_sens_rel", "batch/entropy", "train_s_per_step", "probe_s",
        "all/min_on_query_frac", "all/w_query", "all/head_spread16_body", "all/head_spread16_query",
        "all/body_min_minus_qmin", "all/body_val_median", "val_t5drag/min_on_query_frac")


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def nearest(rows: list[dict], step: int, key: str = "step", tol: int = 20):
    best = None
    for r in rows:
        s = r.get(key)
        if s is None:
            continue
        if abs(int(s) - step) <= tol and (best is None or abs(int(s) - step) < abs(int(best[key]) - step)):
            best = r
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "runs/lang/diag_1030/retrain"))
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000])
    ap.add_argument("--orig", default=str(REPO / "runs/lang/rounds_ur5_g095_1023/train_log.jsonl"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    crit = json.loads((root / "criteria.json").read_text()) if (root / "criteria.json").exists() else {}
    thr = float(crit.get("clear_spread", {}).get("threshold_h_units", 0.02))
    ratio = float(crit.get("departs", {}).get("ratio", 2.0))
    orig = read_jsonl(Path(a.orig))
    diag = {v: {int(r["step"]): r for r in read_jsonl(root / v / "diag.jsonl")} for v in VARIANTS + EXPLORATORY}
    res = {"criteria": crit, "variants": {}, "orig_train_log": {}}
    for st in a.steps:
        o = nearest(orig, st)
        if o is not None:
            res["orig_train_log"][st] = {"step": o.get("step"), "critic/action_sens_rel": o.get("critic/action_sens_rel"),
                                         "entropy": o.get("entropy"), "loss/q": o.get("loss/q")}
    last = {}
    for v in VARIANTS + EXPLORATORY:
        rows = diag[v]
        last[v] = max(rows) if rows else None
        res["variants"][v] = {"last_step": last[v],
                              "steps": {st: {k: rows[st].get(k) for k in KEYS} for st in a.steps if st in rows}}
    # verdict at the latest step every variant has
    shared = set.intersection(*[set(diag[v]) for v in VARIANTS]) if all(diag[v] for v in VARIANTS) else set()
    common = max(shared) if shared else None
    res["latest_step_all_variants"] = common
    # the interim comparison: every step at which stoch and another variant both have a probe line
    res["vs_stoch"] = {v: {st: {"spread16": diag[v][st].get("all/spread16"),
                                "stoch_spread16": diag["stoch"][st].get("all/spread16"),
                                "ratio": (diag[v][st].get("all/spread16") or 0)
                                / max(diag["stoch"][st].get("all/spread16") or 1e-12, 1e-12)}
                           for st in sorted(set(diag[v]) & set(diag["stoch"])) if st in a.steps or st == max(
                               set(diag[v]) & set(diag["stoch"]))}
                       for v in VARIANTS + EXPLORATORY if v != "stoch"}
    if common is not None:
        sp = {v: diag[v][common].get("all/spread16") for v in VARIANTS}
        clear = {v: (sp[v] is not None and sp[v] >= thr) for v in VARIANTS}
        dep = {v: clear[v] and sp["stoch"] is not None and sp[v] >= ratio * max(sp["stoch"], 1e-12)
               for v in VARIANTS if v != "stoch"}
        leaves = {v: (diag[v][common].get("all/actor_u_abs") or 0) >= crit.get("actor_leaves_zero", {}).get(
            "u_abs_threshold", 0.05) or (diag[v][common].get("all/actor_sigma") or 1) < crit.get(
            "actor_leaves_zero", {}).get("sigma_threshold", 0.9) for v in VARIANTS}
        if clear["stoch"]:
            verdict = "mixed: the stoch control itself has a clear spread"
        elif any(dep[v] for v in ("det", "data", "zero")):
            verdict = "H1 supported: a non-random continuation departs (" + ", ".join(
                v for v in ("det", "data", "zero") if dep[v]) + ")"
        else:
            verdict = ("target not the cause: det/data/zero flat; "
                       + ("stoch_norm departs -> input scale" if dep["stoch_norm"] else "stoch_norm flat too -> data / time step"))
        res["verdict"] = {"at_step": common, "final": common >= int(crit.get("verdict_step", 20000)),
                          "spread16": sp, "clear": clear, "departs": dep, "actor_leaves_zero": leaves,
                          "verdict": verdict}
    # the vanilla run's own checkpoints and the smoke checkpoints scored on the same probe set (score-ckpt)
    res["scored_checkpoints"] = {}
    for d in ("orig_scoring", "smoke_scoring"):
        for f in sorted((root / d).glob("diag_*.jsonl")):
            r = read_jsonl(f)
            if r:
                res["scored_checkpoints"][f"{d}/{f.stem[5:]}"] = {"step": r[-1].get("step"),
                                                                    **{k: r[-1].get(k) for k in KEYS}}
    # check 2's T5 branch states scored per checkpoint (score_states.py / score_new.sh): pooled over states files
    sp = root / "scores_t5" / "summary.json"
    if sp.exists():
        agg: dict = {}
        for key, sm in json.loads(sp.read_text()).items():
            tag = key.split("__", 1)[1] if "__" in key else key
            a_ = agg.setdefault(tag, {"step": sm.get("step"), "target": sm.get("target"), "n_states": 0, "files": 0,
                                      "_qs": 0.0, "_s16": 0.0, "gt": {}})
            n = int(sm.get("n_states") or 0)
            a_["n_states"] += n
            a_["files"] += 1
            a_["_qs"] += n * float(sm.get("q_spread_mean") or 0)
            a_["_s16"] += n * float(sm.get("spread16_mean") or 0)
            for g in ("gt/min_h", "gt/V95"):
                sg = (sm.get(g) or {}).get("sign") or {}
                if sg.get("n_pairs"):
                    t = a_["gt"].setdefault(g, {"n_pairs": 0, "_agree": 0.0, "_dec": 0.0})
                    t["n_pairs"] += sg["n_pairs"]
                    t["_agree"] += sg["n_pairs"] * float(sg.get("agree") or 0)
                    t["_dec"] += sg["n_pairs"] * float(sg.get("decided_frac") or 0)
        for tag, a_ in agg.items():
            n = max(a_["n_states"], 1)
            a_["q_spread_candidates"] = a_.pop("_qs") / n
            a_["spread16"] = a_.pop("_s16") / n
            for g, t in a_["gt"].items():
                t["sign_agree"] = t.pop("_agree") / max(t["n_pairs"], 1)
                t["decided_frac"] = t.pop("_dec") / max(t["n_pairs"], 1)
        res["t5_branch_scores"] = agg
    if "verdict" in res:
        res["verdict"]["exploratory"] = {v: {"last_step": last.get(v),
                                             "spread16": diag[v][last[v]].get("all/spread16") if last.get(v) else None,
                                             "stoch_spread16_same_step": diag["stoch"].get(last[v], {}).get(
                                                 "all/spread16") if last.get(v) else None}
                                         for v in EXPLORATORY}
    (root / "results.json").write_text(json.dumps(res, indent=1))
    if not a.quiet:
        cols = ("all/spread16", "val_t5drag/spread16", "all/sens_rel", "val_planner/q0_minus_qnom",
                "val_planner/slope_c_planner", "all/q_mean", "all/h_mean", "val/pair_rank_acc", "all/actor_u_abs",
                "all/actor_sigma", "batch/loss/q", "train_s_per_step")
        print("variant step " + " ".join(c.split("/", 1)[-1][:14].rjust(14) for c in cols))
        for v in VARIANTS + EXPLORATORY:
            for st, r in sorted(res["variants"][v]["steps"].items()):
                print(f"{v:10s} {st:5d} " + " ".join(
                    (f"{r[c]:14.5g}" if isinstance(r.get(c), (int, float)) else " " * 13 + "-") for c in cols))
        for st, o in sorted(res["orig_train_log"].items()):
            print(f"orig       {st:5d} action_sens_rel {o['critic/action_sens_rel']} entropy {o['entropy']}")
        if "verdict" in res:
            print(json.dumps(res["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
