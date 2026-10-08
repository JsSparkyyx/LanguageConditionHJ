"""Section 5 table and the E4 decision for the grounding runs (review round 3, Part II E2 / E4).

    python scripts/lang/grounding_summary.py --runs runs/lang/grounding_1008/g1008_v*_s* \
        --out-name grounding --md runs/lang/grounding_1008/summary.md

Reads each run's eval_offline/<out-name>/rows.json.gz (every (state, rule) row, scored on the same validation
states: the pairs of all runs are the same pairs). Per variant and pair the seeds' outcomes (1 right, 0.5 tie,
0 wrong) are averaged, then rank accuracy per group with an episode-cluster bootstrap 95% interval of that
seed mean; each seed's own value is listed too. Groups: referent pairs (same consequence, other object) and
consequence pairs (same object, other consequence), T1 + T7 pooled and per task, already harmed / not yet;
the sentence-only prior on the same pairs. E4 (Yuxuan, 2026-10-08): primary = referent, already harmed, T1 +
T7 pooled. Text fix: V1 or V2 >= 0.85 with its interval above V0's interval. Grounding bottleneck: V3 >= 0.85
while V1 and V2 < 0.75. Not grounding: V3 < 0.75. Otherwise in between.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LABELS = {"v0": "V0 baseline", "v1": "V1 words normalized", "v2": "V2 V1 + [CLS] / [SEP] hidden",
          "v3": "V3 oracle tag (ceiling)"}


def pair_outcomes(rows: list[dict], ax: str = "train") -> dict:
    """{pair key: (episode cluster, task, state, kind, model ok, prior ok)} over the symmetric pairs of `ax`."""
    from main.lang.eval_offline import _ep, _pairs, _rank, _ttv_bin

    out = {}
    for hr, bn in _pairs(rows):
        if not (hr["eligible"][ax] and bn["eligible"][ax]) or hr.get("censored"):
            continue
        b = _ttv_bin(hr.get("ttv"))
        if b is None:
            continue
        same_cons = hr["consequence"] == bn["consequence"]
        same_ref = hr["referent"] == bn["referent"] and hr["referent"] is not None
        kind = "referent" if (same_cons and not same_ref) else "consequence" if (same_ref and not same_cons) else None
        if kind is None:
            continue
        ph, pb = (hr.get("prior") or {}).get(ax), (bn.get("prior") or {}).get(ax)
        key = (hr["src"], hr["i"], hr["rule"], bn["rule"])
        out[key] = (_ep(hr), hr["task"], "already_harmed" if b == "violated" else "not_yet", kind,
                    _rank(hr["q"][ax], bn["q"][ax]), _rank(-ph, -pb) if ph is not None and pb is not None else None)
    return out


def boot(obs, B: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    from main.lang.eval_offline import boot_mean_ci

    if not obs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = boot_mean_ci(obs, B=B, seed=seed)
    return float(np.mean([v for _, v in obs])), lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-name", default="grounding")
    ap.add_argument("--md", required=True)
    args = ap.parse_args(argv)
    runs = sorted({Path(p) for s in args.runs for p in glob.glob(s) if Path(p).is_dir()})
    by_var = defaultdict(dict)   # variant -> seed -> pair outcomes
    meta = {}
    for r in runs:
        m = re.search(r"_(v\d)_s(\d+)$", r.name)
        f = r / "eval_offline" / args.out_name / "rows.json.gz"
        if not m or not f.exists():
            continue
        with gzip.open(f, "rt") as fh:
            rows = json.load(fh)
        by_var[m.group(1)][int(m.group(2))] = pair_outcomes(rows)
        mj = r / "eval_offline" / args.out_name / "metrics.json"
        meta[r.name] = json.loads(mj.read_text()).get("step") if mj.exists() else None
    groups = [("T1 + T7", None), ("T1", "selective_avoid"), ("T7", "gap_place")]
    res = {}
    for v, seeds in sorted(by_var.items()):
        keys = set.intersection(*[set(p) for p in seeds.values()])
        res[v] = {"seeds": sorted(seeds)}
        for kind in ("referent", "consequence"):
            for state in ("already_harmed", "not_yet"):
                for gname, task in groups:
                    sel = [k for k in keys if seeds[next(iter(seeds))][k][3] == kind and
                           seeds[next(iter(seeds))][k][2] == state and (task is None or seeds[next(iter(seeds))][k][1] == task)]
                    first = seeds[next(iter(seeds))]
                    mean_obs = [(first[k][0], float(np.mean([seeds[s][k][4] for s in seeds]))) for k in sel]
                    prior_obs = [(first[k][0], first[k][5]) for k in sel if first[k][5] is not None]
                    per_seed = {s: float(np.mean([seeds[s][k][4] for k in sel])) if sel else float("nan") for s in seeds}
                    res[v][f"{kind}/{state}/{gname}"] = {"n_pairs": len(sel), "mean": boot(mean_obs),
                                                         "prior": boot(prior_obs), "per_seed": per_seed}
    # E4
    def prim(v):
        return res.get(v, {}).get("referent/already_harmed/T1 + T7", {}).get("mean", (float("nan"),) * 3)

    v0, v1, v2, v3 = (prim(v) for v in ("v0", "v1", "v2", "v3"))
    text_fix = any(x[0] >= 0.85 and x[1] > v0[2] for x in (v1, v2) if x[0] == x[0])
    bottleneck = v3[0] >= 0.85 and all(x[0] < 0.75 for x in (v1, v2) if x[0] == x[0])
    not_grounding = v3[0] == v3[0] and v3[0] < 0.75
    outcome = ("the text path is the fix" if text_fix else "grounding is the bottleneck" if bottleneck else
               "not grounding" if not_grounding else "in between")
    res["_e4"] = {"primary": {k: x for k, x in zip(("v0", "v1", "v2", "v3"), (v0, v1, v2, v3))}, "outcome": outcome}
    f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if t[0] == t[0] else "–"  # noqa: E731
    L = ["# Grounding runs: summary (review round 3, Part II E2 / E4)", "",
         f"Runs: {', '.join(f'{k} (step {s})' for k, s in sorted(meta.items()))}. Cells: rank accuracy, the seed mean "
         "[episode-bootstrap 95% interval] (each seed in brackets after it); prior = the sentence-only reader on "
         "the same pairs. Referent pairs: same consequence, other object; consequence pairs: same object, other "
         "consequence.", ""]
    for kind in ("referent", "consequence"):
        L += [f"## {kind.capitalize()} pairs", "",
              "| run | " + " | ".join(f"{g} {s.replace('_', ' ')}" for s in ("already_harmed", "not_yet") for g, _ in groups) + " |",
              "|---|" + "---|" * 6]
        for v in sorted(by_var):
            cells = []
            for state in ("already_harmed", "not_yet"):
                for gname, _ in groups:
                    c = res[v][f"{kind}/{state}/{gname}"]
                    seeds_s = " / ".join(f"{x:.2f}" for x in c["per_seed"].values())
                    cells.append(f"{f(c['mean'])} ({seeds_s}; {c['n_pairs']})")
            L.append(f"| {LABELS.get(v, v)} | " + " | ".join(cells) + " |")
        c0 = res[sorted(by_var)[0]] if by_var else {}
        L.append("| sentence prior | " + " | ".join(f(c0.get(f"{kind}/{s}/{g}", {}).get("prior", (float('nan'),) * 3))
                                                   for s in ("already_harmed", "not_yet") for g, _ in groups) + " |")
        L.append("")
    L += ["## E4 decision", "", "Primary: referent, already harmed, T1 + T7 pooled: " +
          ", ".join(f"{k} {f(x)}" for k, x in res["_e4"]["primary"].items()) + f". Outcome: **{outcome}**.", ""]
    Path(args.md).write_text("\n".join(L) + "\n")
    Path(args.md).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
