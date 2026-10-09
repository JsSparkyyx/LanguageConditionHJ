"""Tables and figures per research question (VANILLA_PLAN 9.3).

    python -m main.lang.report --runs runs/lang/* --out docs/lang_filter/results/

Reads each run's ``lang_config.json``, ``eval_offline/<ckpt>/metrics.json`` and, when present,
``eval_closed/<ckpt>/summary.json`` for the checkpoint named by ``--ckpt`` (review M6: not the newest file,
which mixed checkpoints). A run's variant is read from its config (relabel mode, shuffled text, path
switches, text backend, task filter), never from its name. Runs whose offline numbers cover different task
sets (e.g. bead tasks left out for their old bead h) are never put in one table: each task set gets its own.

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


def load_run(d: Path, ckpt: str = "checkpoint_final") -> dict | None:
    cfgp = d / "lang_config.json"
    if not cfgp.exists():
        return None
    cfg = json.loads(cfgp.read_text())
    stem = Path(ckpt).stem
    off = [p for p in [d / "eval_offline" / stem / "metrics.json"] if p.exists()]
    clo = [p for p in [d / "eval_closed" / stem / "summary.json"] if p.exists()]
    if not off and not clo:
        return None
    r = {"name": d.name, "cfg": cfg, "variant": variant(cfg), "mode": "rounds" if (d / "ring").exists() else "offline"}
    if off:
        m = json.loads(off[-1].read_text())
        r["offline"], r["offline_ckpt"], r["step"] = m["metrics"], m.get("ckpt"), m.get("step")
        r["data"] = m.get("data")
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
    """Referent vs consequence decisions where both exist: T1, T3, T7 and their pool (round 2 C6: the pooled
    referent score also held T5 / T6, which have no consequence pairs)."""
    tasks = (("T1", "selective_avoid"), ("T3", "push_zone"), ("T7", "gap_place"))
    L = ["## Q2: where must language enter? (referent vs consequence decisions, T1 / T3 / T7 only)", "",
         "| run | variant | " + " | ".join(f"{n} referent | {n} consequence" for n, _ in tasks) +
         " | T1+T3+T7 referent | T1+T3+T7 consequence |",
         "|---|---|" + "---|---|" * len(tasks) + "---|---|"]
    for r in runs:
        m = r.get("offline")
        if m is None:
            continue
        cells = []
        for _, t in tasks:
            cells += [fmt(g(m, f"train/symmetric/task/{t}", "rank_acc_referent")),
                      fmt(g(m, f"train/symmetric/task/{t}", "rank_acc_consequence"))]
        cells += [fmt(g(m, "train/symmetric/rc_pool", "rank_acc_referent")),
                  fmt(g(m, "train/symmetric/rc_pool", "rank_acc_consequence"))]
        L.append(f"| {r['name']} | {r['variant']} | " + " | ".join(cells) + " |")
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


def _actor_hidden(r) -> int:
    """The actor head width from the run's saved config (a config without the field is the linear head, 0)."""
    return int((r.get("cfg") or {}).get("actor_hidden") or 0)


def closed(runs) -> str:
    """One table per (task set, env code hash, filter code hash) of the closed-loop summaries (round 2 C6)."""
    groups = {}
    for r in runs:
        S = r.get("closed")
        if not S:
            continue
        meta = S.get("_meta") or {}
        ah = meta.get("actor_hidden", _actor_hidden(r))
        groups.setdefault((tuple(meta.get("tasks") or ()), meta.get("code_hash"), meta.get("filter_hash"), ah),
                          []).append(r)
    L = ["## Closed loop (compare runs trained the same way)"]
    for (tasks, ch, fh, ah), rr in groups.items():
        head = f"actor_hidden {ah} ({'MLP' if ah else 'linear'} actor head)"
        what = (f"tasks {', '.join(tasks)}; env code {ch}, filter code {fh}; {head}" if tasks else
                f"task set / code hashes not recorded (summary.json from before 2026-10-08: rerun the merge); {head}")
        L += ["", f"### {what}", "",
              "| run | mode | axis | harm nominal | harm filter | protect | protect+ (cross-fit) | benign false alarm | "
              "selectivity | all runs: broken nominal / filter |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rr:
            for k, v in sorted(r["closed"].items()):
                if not k.startswith("pooled/"):
                    continue
                a = v.get("all") or {}
                L.append(f"| {r['name']} | {r['mode']} | {k.split('/')[1]} | {fmt(v['harm_rate_nominal'], 2)} | "
                         f"{fmt(v['harm_rate_filter'], 2)} | {fmt(v['protect_rate'], 2)} | {fmt(v.get('protect_rate_pos'), 2)} | "
                         f"{fmt(v['benign_false_alarm'], 2)} | {fmt(v['selectivity'], 2)} | "
                         f"{fmt(a.get('broken_nominal'), 2)} / {fmt(a.get('broken_filter'), 2)} |")
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
    ap.add_argument("--ckpt", default="checkpoint_final", help="the checkpoint whose evaluations every run reports")
    args = ap.parse_args(argv)
    dirs = sorted({Path(p) for s in args.runs for p in glob.glob(s) if Path(p).is_dir()})
    runs = [r for r in (load_run(d, args.ckpt) for d in dirs) if r]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in runs:   # one section per task set the offline numbers cover
        d = r.get("data") or {}
        # the TRAINING data's bead h (round 3, B1; the validation's bead_h said nothing about the weights), the
        # scored embodiments and the evaluation code: runs differing in any of them never share a table
        key = (tuple(d.get("tasks") or ()), d.get("bead_h_train", "not recorded"), tuple(d.get("embodiments") or ()),
               d.get("eval_code_hash", "not recorded"), d.get("actor_hidden", _actor_hidden(r)))
        groups.setdefault(key, []).append(r)
    parts = ["# Results by research question", f"checkpoint {args.ckpt}; {len(runs)} runs: " + ", ".join(r["name"] for r in runs)]
    for (tasks, bead_h, embs, code, ah), rr in groups.items():
        what = (f"tasks {', '.join(tasks)}; trained on bead h: {bead_h}; embodiments {', '.join(embs) or 'not recorded'}; "
                f"eval code {code}; actor_hidden {ah}" if tasks else
                "task set not recorded (metrics.json from before 2026-10-07: rerun eval_offline)")
        dropped = sum(int((r.get("data") or {}).get("stale_bead_rows_dropped", 0)) for r in rr)
        parts += [f"# Runs scored on {what}" + (f" ({dropped} old-h bead rows left out)" if dropped else ""),
                  q1(rr), q2(rr), q3(rr), q4(rr), closed(rr)]
    md = "\n\n".join(parts) + "\n"
    (out / "results.md").write_text(md)
    figure(runs, out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
