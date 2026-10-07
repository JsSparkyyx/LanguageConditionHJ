"""Tables and figures per research question (VANILLA_PLAN 9.3).

    python -m main.lang.report --runs runs/lang/* --out docs/lang_filter/results/

Reads each run's ``lang_config.json``, the latest ``eval_offline/<ckpt>/metrics.json`` and, when present,
``eval_closed/<ckpt>/summary.json``. A run's variant is read from its config (relabel mode, shuffled
text, path switches, text backend, task filter), never from its name.

- Q1 (does relabeling teach the filter to read?): paired vs random vs fixed vs shuffled, rank_acc on the
  train and held-out-template axes, AUROC vs its other-episode baseline;
- Q2 (where must language enter?): the five path switches, rank_acc_referent vs rank_acc_consequence;
- Q3 (which text space?): text backends;
- Q4 (which axis fails first?): rank_acc per axis for every run;
- closed loop: protect_rate, benign false alarms, selectivity per run (runs trained the same way only).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

AXES = ("train", "heldout_template", "heldout_word", "heldout_item", "shuffled")
SWITCHES = ("lang_scene", "lang_body", "lang_film", "lang_queries", "lang_rtoken")


def variant(cfg: dict) -> str:
    parts = []
    if cfg.get("shuffle_text"):
        parts.append("shuffled")
    if cfg.get("relabel", "paired") != "paired":
        parts.append(cfg["relabel"])
    off = [s.replace("lang_", "no") for s in SWITCHES if not cfg.get(s, True)]
    parts += off
    if cfg.get("text_backend", "holobrain_bert") != "holobrain_bert":
        parts.append(cfg["text_backend"])
    tc = cfg.get("task_choices")
    if tc and len(tc) < 7:
        parts.append("tasks=" + ",".join(tc))
    return "+".join(parts) or "full"


def load_run(d: Path) -> dict | None:
    cfgp = d / "lang_config.json"
    if not cfgp.exists():
        return None
    cfg = json.loads(cfgp.read_text())
    off = sorted((d / "eval_offline").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    clo = sorted((d / "eval_closed").glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not off and not clo:
        return None
    r = {"name": d.name, "cfg": cfg, "variant": variant(cfg), "mode": "rounds" if (d / "ring").exists() else "offline"}
    if off:
        m = json.loads(off[-1].read_text())
        r["offline"], r["offline_ckpt"], r["step"] = m["metrics"], m.get("ckpt"), m.get("step")
    if clo:
        r["closed"] = json.loads(clo[-1].read_text())
    return r


def g(m: dict, key: str, field: str):
    v = (m or {}).get(key, {}).get(field)
    return float("nan") if v is None else float(v)


def fmt(x, nd=3) -> str:
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def q1(runs) -> str:
    L = ["## Q1: does HJ loss + relabeling teach the filter to read the sentence?", "",
         "| run | variant | mode | step | rank_acc train | rank_acc heldout_template | auroc train | auroc base | "
         "shuffled-axis rank_acc |", "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r.get("offline")
        if m is None:
            continue
        L.append(f"| {r['name']} | {r['variant']} | {r['mode']} | {r.get('step')} | "
                 f"{fmt(g(m, 'train/symmetric/pooled', 'rank_acc'))} | {fmt(g(m, 'heldout_template/symmetric/pooled', 'rank_acc'))} | "
                 f"{fmt(g(m, 'train/state/pooled', 'auroc'))} | {fmt(g(m, 'train/state/pooled', 'auroc_base'))} | "
                 f"{fmt(g(m, 'shuffled/symmetric/pooled', 'rank_acc'))} |")
    return "\n".join(L)


def q2(runs) -> str:
    L = ["## Q2: where must language enter? (referent vs consequence decisions)", "",
         "| run | variant | rank_acc train | referent decisions | consequence decisions | text_sens (probe) |",
         "|---|---|---|---|---|---|"]
    for r in runs:
        m = r.get("offline")
        if m is None:
            continue
        L.append(f"| {r['name']} | {r['variant']} | {fmt(g(m, 'train/symmetric/pooled', 'rank_acc'))} | "
                 f"{fmt(g(m, 'train/symmetric/pooled', 'rank_acc_referent'))} | "
                 f"{fmt(g(m, 'train/symmetric/pooled', 'rank_acc_consequence'))} | – |")
    return "\n".join(L)


def q3(runs) -> str:
    L = ["## Q3: which text space?", "", "| run | backend | rank_acc train | rank_acc heldout_template | heldout_word |",
         "|---|---|---|---|---|"]
    for r in runs:
        m = r.get("offline")
        if m is None:
            continue
        L.append(f"| {r['name']} | {r['cfg'].get('text_backend')} | {fmt(g(m, 'train/symmetric/pooled', 'rank_acc'))} | "
                 f"{fmt(g(m, 'heldout_template/symmetric/pooled', 'rank_acc'))} | "
                 f"{fmt(g(m, 'heldout_word/symmetric/pooled', 'rank_acc'))} |")
    return "\n".join(L)


def q4(runs) -> str:
    L = ["## Q4: which generalization axis fails first? (rank_acc, symmetric pairs; mixed: difference to base)", "",
         "| run | " + " | ".join(AXES) + " | heldout_word mixed | heldout_item mixed |",
         "|---|" + "---|" * (len(AXES) + 2)]
    for r in runs:
        m = r.get("offline")
        if m is None:
            continue
        cells = [fmt(g(m, f"{a}/symmetric/pooled", "rank_acc")) for a in AXES]
        cells += [fmt(g(m, f"{a}/mixed/pooled", "rank_acc_minus_base")) for a in ("heldout_word", "heldout_item")]
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def closed(runs) -> str:
    L = ["## Closed loop (compare runs trained the same way)", "",
         "| run | mode | axis | harm nominal | harm filter | protect | benign false alarm | selectivity | nominal disagreement |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        S = r.get("closed")
        if not S:
            continue
        for k, v in sorted(S.items()):
            if not k.startswith("pooled/"):
                continue
            L.append(f"| {r['name']} | {r['mode']} | {k.split('/')[1]} | {fmt(v['harm_rate_nominal'], 2)} | "
                     f"{fmt(v['harm_rate_filter'], 2)} | {fmt(v['protect_rate'], 2)} | {fmt(v['benign_false_alarm'], 2)} | "
                     f"{fmt(v['selectivity'], 2)} | {fmt(v['nominal_disagreement'], 2)} |")
    return "\n".join(L)


def figure(runs, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rr = [r for r in runs if r.get("offline")]
    if not rr:
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(rr)), 3.5))
    w = 0.8 / len(AXES)
    for k, a in enumerate(AXES):
        ax.bar(np.arange(len(rr)) + (k - 2) * w, [g(r["offline"], f"{a}/symmetric/pooled", "rank_acc") for r in rr], w, label=a)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(rr)), [r["variant"] for r in rr], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rank_acc")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(out / "rank_acc_by_run.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dirs = sorted({Path(p) for s in args.runs for p in glob.glob(s) if Path(p).is_dir()})
    runs = [r for r in (load_run(d) for d in dirs) if r]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = "\n\n".join(["# Results by research question", f"{len(runs)} runs: " + ", ".join(r["name"] for r in runs),
                      q1(runs), q2(runs), q3(runs), q4(runs), closed(runs)]) + "\n"
    (out / "results.md").write_text(md)
    figure(runs, out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
