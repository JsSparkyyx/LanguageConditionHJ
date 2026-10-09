"""E6 (FUSION_PLAN section 4, fusion_impl.md decision 3): the scene-count curve.

    python scripts/lang/scene_curve.py --runs 'runs/lang/fusion_1009/e6_n*_s*' \
        'runs/lang/grounding_1008/g1008_v0_s*' 'runs/lang/grounding_1008/g1008_d2_s*' \
        --out runs/lang/fusion_1009/e6/scene_curve

V0-style runs (critic_readout softmin, gamma 0.95, actor_hidden 0, recall 7) that differ only in the scenes they
trained on: train_episodes_per_task = 8, 15 (new runs), 0 = all (the report's V0: 29 T1 + 30 T7 ur5 scenes) and the
d2 runs (both robots, about 120 scenes). One point per (train_episodes_per_task, embodiments), its seeds pooled:

- training scenes: the episodes the run's sampler drew from (grounding_summary.training_episodes: the training
  tasks and embodiments of the checkpoint's data roots, train_episodes_per_task's fixed subset);
- the primary score (referent pairs, already harmed, T1 + T7 pooled; grounding_summary.pair_outcomes) on new scenes
  (eval_offline/<out-name>: the validation pairs) and on training scenes (eval_offline/<out-name>_train, restricted
  to the run's OWN training episodes: a subsampled run scored on the whole warmup would count scenes it never saw),
  the seed mean over the pairs every seed scored [episode-bootstrap 95% interval], each seed's value;
- repeats per scene: batch_size x steps / training episodes (FUSION_PLAN C1).

Held equal to the reference run (--reference; default the first run that trained on all scenes of one robot), else
left out and listed: gamma, actor head, TF32, steps, tasks, the critic readout (a config saved before the field
reads softmin), the model switches (grounding_summary.run_settings' switches: text_out_norm, mask_special_tokens,
oracle_ref_tag, aux_target_loss, aux_attention_loss, scene_fusion / fusion_words; so V1-V4 and V5 runs cannot join a
V0 point), the training data version, the scored set (roots, tasks, embodiments, episodes) and the scored sentences
(grounding_summary.val_sentences: rows scored before the D1 head-noun fix of 2026-10-09, such as the report's V0 and
the d2 runs scored in the pre-D1 checkout, carry other sentences for some T1 rules and are refused: re-score them under
the current language code). Embodiments and train_episodes_per_task differ by design. A second run with the same
point and seed is refused too (it would replace the first). Writes <out>.json, <out>.md and <out>.png (score vs
training scenes, log x).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

NAN = float("nan")
BAR, COIN = 0.85, 0.5   # the grounding bar (FUSION_PLAN decision 4) and a coin flip
NEW_COLOR, TRAIN_COLOR, INK, MUTED = "#2a78d6", "#eb6834", "#1b2230", "#586374"   # dataviz reference slots 1, 2


def _sibling(name: str):
    """A script next to this one, as a module (scripts/lang is not a package)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gs = _sibling("grounding_summary")
FIXED = ("gamma", "actor_hidden", "tf32_train", "steps", "tasks", "critic_readout", "model_switches",
         "train_data_version")
CHECKED = FIXED + gs.SCORED + gs.SCORED_V2


def model_switches(switches) -> list:
    """run_settings' switches without what a curve point differs in by design (the episodes per task) or holds
    equal on its own (the critic readout, a config without the field reading softmin): the model's switches."""
    return sorted(s for s in switches if not s.endswith(" episodes / task") and not s.startswith("critic_readout "))


def point_score(by_seed: dict) -> dict:
    """{seed: pair outcomes} -> the primary score of a point: the seed mean over the primary pairs every seed scored,
    with its episode-bootstrap interval, and each seed's value on those pairs."""
    if not by_seed:
        return {"mean": [NAN] * 3, "per_seed": {}, "n_pairs": 0, "n_episodes": 0}
    keys = sorted(set.intersection(*[set(p) for p in by_seed.values()]))
    first = by_seed[next(iter(by_seed))]
    sel = [k for k in keys if first[k][3] == "referent" and first[k][2] == "already_harmed"
           and first[k][1] in gs.PRIMARY_TASKS]
    obs = [(first[k][0], float(np.mean([by_seed[s][k][4] for s in by_seed]))) for k in sel]
    return {"mean": list(gs.boot(obs)), "n_pairs": len(sel), "n_episodes": len({first[k][0] for k in sel}),
            "per_seed": {str(s): float(np.mean([p[k][4] for k in sel])) if sel else NAN
                         for s, p in sorted(by_seed.items())}}


def own_rows(rows: list[dict], val_roots, episodes) -> list[dict]:
    """The rows of a training-scene scoring on the run's own training episodes ((root, episode_id); rows name their
    root by its index in the scoring's val_roots)."""
    keep = {(str(Path(r).resolve()), int(e)) for r, e in episodes}
    roots = [str(Path(r).resolve()) for r in val_roots]
    return [x for x in rows if 0 <= int(x.get("src", 0)) < len(roots)
            and (roots[int(x.get("src", 0))], int(x["episode_id"])) in keep]


def run_info(run: Path, out_name: str) -> dict:
    """grounding_summary.run_settings plus what the curve needs: the critic readout, the training episodes, the
    scored validation episodes and both pair-outcome sets."""
    st = gs.run_settings(run, out_name)   # critic_readout as LangConfig.from_dict reads it (gs.lang_fields)
    cfg = json.loads((run / "lang_config.json").read_text())
    st["model_switches"] = model_switches(st["switches"])
    st["tasks"] = tuple(sorted(st["tasks"]))   # the sampler draws by task, not by its place in --tasks
    rows = gs._rows(run / "eval_offline" / out_name / "rows.json.gz")
    st["val_episodes"] = gs.val_episodes(rows)
    st["val_sentences"] = gs.val_sentences(rows)
    st["new"] = gs.pair_outcomes(rows)
    del rows
    ck = run / (st.get("ckpt") or "checkpoint_final.pt")
    ident = gs._ckpt_keys(ck)[1]
    tr = gs.training_episodes(tuple(str(x) for x in ident.get("data_roots") or ()), st["tasks"], st["embodiments"],
                              st["train_episodes_per_task"], bool(cfg.get("drop_stale_bead_rows", True)))
    st["train_set"] = tr
    st["train"], st["train_note"] = None, None
    tdir = run / "eval_offline" / f"{out_name}_train"
    if (tdir / "rows.json.gz").exists():
        rows = gs._rows(tdir / "rows.json.gz")
        mj = tdir / "metrics.json"
        val_roots = ((json.loads(mj.read_text()).get("data") or {}).get("val_roots")) if mj.exists() else None
        if tr and val_roots:
            rows = own_rows(rows, val_roots, tr["episodes"])
        else:
            st["train_note"] = "not restricted to the run's own episodes (no data roots or no scored roots)"
        st["train"] = gs.pair_outcomes(rows)
    m = re.search(r"_s(\d+)$", run.name)
    st["seed"] = int(m.group(1)) if m else run.name
    return st


def label(per_task: int, embodiments) -> str:
    return (f"{per_task} / task" if per_task else "all") + (f" ({', '.join(embodiments)})" if embodiments else "")


def plot(points: list[dict], path: Path) -> None:
    """Primary score vs training scenes (log x): new scenes and training scenes, seed mean with its interval; the
    bar (0.85) and a coin flip (0.5) as reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, color, name in (("new", NEW_COLOR, "new scenes"), ("train", TRAIN_COLOR, "training scenes")):
        pts = sorted((p["train_episodes"], p[key]["mean"], len(p["embodiments"]) > 1) for p in points
                     if p["train_episodes"] and p[key]["mean"][0] == p[key]["mean"][0])
        if not pts:
            continue
        x = np.array([q[0] for q in pts], float)
        m = np.array([q[1][0] for q in pts])
        lo = np.array([q[1][1] if q[1][1] == q[1][1] else q[1][0] for q in pts])
        hi = np.array([q[1][2] if q[1][2] == q[1][2] else q[1][0] for q in pts])
        ax.plot(x, m, color=color, lw=2, label=name, zorder=3)
        # clamped at 0: a degenerate interval can sit one ulp off the mean (matplotlib refuses a negative bar)
        ax.errorbar(x, m, yerr=[np.maximum(m - lo, 0.0), np.maximum(hi - m, 0.0)], fmt="none", ecolor=color,
                    elinewidth=1.5, capsize=3, zorder=3)
        for xi, mi, both in zip(x, m, [q[2] for q in pts]):   # both robots: a hollow marker
            ax.plot([xi], [mi], "o", ms=8, mfc="white" if both else color, mec=color, mew=2, zorder=4)
        ax.annotate(name, (x[-1], m[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    for y, txt in ((BAR, f"bar {BAR}"), (COIN, "coin flip")):
        ax.axhline(y, color=MUTED, lw=1, zorder=1)
        ax.annotate(txt, (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, color=MUTED)
    ax.set_xscale("log")
    xs = sorted({p["train_episodes"] for p in points if p["train_episodes"]})
    if xs:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(v)) for v in xs])
        ax.set_xlim(xs[0] / 1.4, xs[-1] * 2.2)
    ax.minorticks_off()
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("training scenes (episodes, T1 + T7; hollow: both robots)", color=INK)
    ax.set_ylabel("rank accuracy, different objects, already harmed", color=INK)
    ax.grid(axis="y", color="#d9dfe7", lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("E6: does more scenes help? (seed mean, 95% interval)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or globs")
    ap.add_argument("--out-name", default="grounding", help="eval_offline output name (and <name>_train)")
    ap.add_argument("--out", required=True, help="output path without suffix: <out>.json, .md, .png")
    ap.add_argument("--reference", default=None, help="run name the others are held against")
    args = ap.parse_args(argv)
    runs = sorted({Path(p) for s in args.runs for p in glob.glob(s)
                   if (Path(p) / "lang_config.json").exists()})
    info, pending = {}, []
    for r in runs:
        if (r / "eval_offline" / args.out_name / "rows.json.gz").exists():
            info[r.name] = run_info(r, args.out_name)
        else:
            pending.append(r.name)
    names = sorted(info)
    ref = args.reference or next((n for n in names if not info[n]["train_episodes_per_task"]
                                  and len(info[n]["embodiments"]) == 1), names[0] if names else None)
    if ref is not None and ref not in info:
        raise SystemExit(f"--reference {ref}: not among the scored runs ({', '.join(names)})")
    refused, taken = {}, {}
    groups = defaultdict(list)
    for n in sorted(names, key=lambda x: x != ref):   # the reference first: it keeps its (point, seed)
        why = [gs.differs(k, info[n][k], info[ref][k], f"{ref}'s") for k in CHECKED if info[n][k] != info[ref][k]]
        if info[n]["train_set"] is None:
            why.append("no training episodes found (the checkpoint's data roots have no episode lines)")
        point = (info[n]["train_episodes_per_task"], tuple(info[n]["embodiments"]))
        if not why and (point, info[n]["seed"]) in taken:   # by_new / by_tr are keyed by seed: it would replace one
            why.append(f"same point and seed as {taken[(point, info[n]['seed'])]}")
        if why:
            refused[n] = why
            print(f"[scene_curve] refused {n}: {'; '.join(why)}", flush=True)
            continue
        taken[(point, info[n]["seed"])] = n
        groups[point].append(n)
    points = []
    for (per_task, embs), ns in groups.items():
        ns = sorted(ns)
        sts = [info[n] for n in ns]
        n_ep = sorted({x["train_episodes"] for x in sts})
        spe = [x["samples_per_episode"] for x in sts if x["samples_per_episode"] is not None]
        by_new = {x["seed"]: x["new"] for x in sts}
        by_tr = {x["seed"]: x["train"] for x in sts if x["train"] is not None}
        points.append({"label": label(per_task, embs), "per_task": per_task, "embodiments": list(embs),
                       "runs": ns, "seeds": sorted(str(x["seed"]) for x in sts),
                       "train_episodes": n_ep[0] if len(n_ep) == 1 else float(np.mean(n_ep)),
                       "train_episodes_runs": n_ep, "per_group": sts[0]["train_set"]["per_group"],
                       "samples_per_episode": float(np.mean(spe)) if spe else None,
                       "new": point_score(by_new), "train": point_score(by_tr),
                       "train_notes": sorted({x["train_note"] for x in sts if x["train_note"]})})
    points.sort(key=lambda p: (p["train_episodes"], len(p["embodiments"])))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ("gamma", "actor_hidden", "tf32_train", "steps", "tasks", "embodiments", "critic_readout", "recall_ticks",
            "train_data_version", "val_episodes", "val_sentences", "train_episodes", "train_episodes_per_task",
            "samples_per_episode", "git_head", "switches", "model_switches")
    res = {"reference": ref, "out_name": args.out_name, "points": points, "refused": refused, "pending": pending,
           "settings": {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in info[n].items() if k in keep}
                        for n in names}, "held_equal": list(CHECKED), "bar": BAR}
    f = lambda t: "–" if t[0] != t[0] else (f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if t[1] == t[1] else f"{t[0]:.2f}")  # noqa: E731
    L = ["# E6: the scene-count curve (FUSION_PLAN section 4)", "",
         f"Primary score (referent pairs, already harmed, T1 + T7 pooled) by the number of training scenes. Cells: the "
         "seed mean over the pairs every seed scored [episode-bootstrap 95% interval] (each seed; pairs). New scenes: "
         f"eval_offline/{args.out_name}; training scenes: eval_offline/{args.out_name}_train on each run's own training "
         "episodes. Repeats per scene = batch_size x steps / training episodes. Held equal to "
         f"{ref}: {', '.join(CHECKED)}.", "",
         "| training scenes | point | runs | new scenes | training scenes | repeats per scene |", "|---|---|---|---|---|---|"]
    for p in points:
        cells = []
        for key in ("new", "train"):
            c = p[key]
            seeds = " / ".join(f"{v:.2f}" for v in c["per_seed"].values())
            cells.append(f"{f(c['mean'])} ({seeds}; {c['n_pairs']})" if c["n_pairs"] else "–")
        n_ep = " / ".join(str(x) for x in p["train_episodes_runs"])
        spe = "–" if p["samples_per_episode"] is None else f"{p['samples_per_episode']:,.0f}"
        L.append(f"| {n_ep} | {p['label']} | {', '.join(p['runs'])} | {cells[0]} | {cells[1]} | {spe} |")
    L.append("")
    notes = sorted({x for p in points for x in p["train_notes"]})
    if notes:
        L += [f"Training-scene scores {'; '.join(notes)}.", ""]
    if refused:
        L += ["Left out: " + "; ".join(f"{n} ({'; '.join(w)})" for n, w in refused.items()) + ".", ""]
    if pending:
        L += [f"Not scored yet: {', '.join(pending)}.", ""]
    L += [f"Figure: `{out.with_suffix('.png').name}`.", ""]
    if points:
        plot(points, out.with_suffix(".png"))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
