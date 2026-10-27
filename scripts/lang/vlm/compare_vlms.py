"""Compare VLM labelers against the simulator's h on the same episodes (VLM_FAILURE_PLAN, 2026-10-27).

    python scripts/lang/vlm/compare_vlms.py --roots 'data/lang_vlm_1026/warmup/shard_g*' \
        --vlms qwen25vl32b robobrain2_32b qwen25vl7b qwen3vl8b [--out runs/lang/vlm_1026/compare.json]

A cell is one labelled rule (T1 "don't touch X", T5 "keep X upright", T6 "no beads in bowl X") at one tick; the
VLM's h at a tick is the score of the nearest asked tick, as in training (hsource.spread). "Harmed" = h < 0, so an
"uncertain" answer (0) counts as not harmed. Per task:
- caught / right: of the cells the simulator calls harmed, the share the VLM also calls harmed / of the cells the
  VLM calls harmed, the share the simulator also calls harmed;
- detected: of the (episode, rule) pairs the simulator calls harmed at some tick, the share the VLM calls harmed at
  some tick; delay: median ticks from the simulator's first harmed tick to the VLM's first (negative = earlier);
- false alarm: of the pairs the simulator never calls harmed, the share the VLM calls harmed at some tick;
- T1 also against "touched so far" (the avoid h was < 0 at this tick or before), what the T1 prompt asks;
- uncertain / unreadable: shares of the answers; s/question: labeling wall time per question (GPUs in the files).
Only episodes every listed VLM has labelled are scored, so the models see the same cells.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import main.lang  # noqa: E402,F401
from main.lang.hsource import spread  # noqa: E402

TASK_NAMES = {"selective_avoid": "T1 don't touch", "tray_transport": "T5 keep upright", "clean_pour": "T6 no beads in bowl"}


def episodes(roots, vlms):
    """(root, episode_id) pairs labelled by every VLM."""
    out = []
    for r in roots:
        sets = [{int(p.stem) for p in (r / "vlm_labels" / v).glob("[0-9]*.json")} for v in vlms]
        out += [(r, e) for e in sorted(set.intersection(*sets))]
    return out


def score(roots, vlms):
    eps = episodes(roots, vlms)
    res = {v: defaultdict(lambda: defaultdict(float)) for v in vlms}
    delays = {v: defaultdict(list) for v in vlms}
    cache = {}
    for root, eid in eps:
        if root not in cache:
            n = json.loads((root / "item_buffer/header.json").read_text())["n"]
            cache = {root: (np.load(root / "item_buffer/episode_id.npy", mmap_mode="r")[:n].reshape(-1),
                            np.load(root / "item_buffer/t_in_ep.npy", mmap_mode="r")[:n].reshape(-1),
                            np.load(root / "item_buffer/h_rules.npy", mmap_mode="r"))}
        ep, t, H = cache[root]
        rows = np.flatnonzero(ep == eid)
        rows = rows[np.argsort(t[rows])]
        tt = t[rows]
        task = json.loads((root / "item_buffer/episodes" / f"{eid}.json").read_text())["task"]
        for v in vlms:
            lab = json.loads((root / "vlm_labels" / v / f"{eid}.json").read_text())
            c = res[v][task]
            c["episodes"] += 1
            c["s_total"] += float(lab.get("wall_s") or 0.0)
            c["host"] = lab.get("host", "")
            for r in lab["rules"].values():
                vn = spread(lab["ticks"], r["score"], tt) < 0
                sn = np.asarray(H[rows, r["idx"]]) < 0
                c["cells"] += vn.size
                c["both"] += int((vn & sn).sum())
                c["vlm_neg"] += int(vn.sum())
                c["sim_neg"] += int(sn.sum())
                if task == "selective_avoid":
                    so = np.maximum.accumulate(sn)
                    c["so_far_both"] += int((vn & so).sum())
                    c["so_far_neg"] += int(so.sum())
                if sn.any():
                    c["harm_pairs"] += 1
                    if vn.any():
                        c["detected"] += 1
                        delays[v][task].append(int(np.argmax(vn)) - int(np.argmax(sn)))
                else:
                    c["safe_pairs"] += 1
                    c["false_alarm"] += int(vn.any())
                c["answers"] += len(r["score"])
                c["uncertain"] += sum(1 for s in r["score"] if s == 0)
                c["unreadable"] += sum(1 for p in r["parsed"] if not p)
    table = {}
    for v in vlms:
        for task, c in res[v].items():
            d = delays[v][task]
            row = {
                "caught": c["both"] / max(c["sim_neg"], 1), "right": c["both"] / max(c["vlm_neg"], 1),
                "detected": c["detected"] / max(c["harm_pairs"], 1), "harm_pairs": int(c["harm_pairs"]),
                "delay_ticks": float(np.median(d)) if d else None,
                "false_alarm": c["false_alarm"] / max(c["safe_pairs"], 1), "safe_pairs": int(c["safe_pairs"]),
                "uncertain": c["uncertain"] / max(c["answers"], 1), "unreadable": c["unreadable"] / max(c["answers"], 1),
                "s_per_question": c["s_total"] / max(c["answers"], 1), "episodes": int(c["episodes"]),
                "cells": int(c["cells"]), "host": c["host"]}
            if task == "selective_avoid":
                row["caught_so_far"] = c["so_far_both"] / max(c["so_far_neg"], 1)
                row["right_so_far"] = c["so_far_both"] / max(c["vlm_neg"], 1)
            table.setdefault(task, {})[v] = row
    return table, len(eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--vlms", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    roots = [Path(p) for r in a.roots for p in sorted(glob.glob(r))]
    table, n = score(roots, a.vlms)
    print(f"{n} episodes labelled by all of {a.vlms}\n")
    for task in ("selective_avoid", "tray_transport", "clean_pour"):
        if task not in table:
            continue
        print(f"## {TASK_NAMES[task]}")
        print("| VLM | caught | right | detected | delay (ticks) | false alarm | uncertain | unreadable | s/question |"
              + (" caught (touched so far) | right (touched so far) |" if task == "selective_avoid" else ""))
        print("|---" * (11 if task == "selective_avoid" else 9) + "|")
        for v, r in table[task].items():
            dl = "-" if r["delay_ticks"] is None else f"{r['delay_ticks']:+.0f}"
            line = (f"| {v} | {r['caught']:.2f} | {r['right']:.2f} | {r['detected']:.2f} of {r['harm_pairs']} | {dl} | "
                    f"{r['false_alarm']:.2f} of {r['safe_pairs']} | {r['uncertain']:.1%} | {r['unreadable']:.1%} | "
                    f"{r['s_per_question']:.2f} |")
            if task == "selective_avoid":
                line += f" {r['caught_so_far']:.2f} | {r['right_so_far']:.2f} |"
            print(line)
        print()
    if a.out:
        Path(a.out).write_text(json.dumps({"episodes": n, "table": table}, indent=1))


if __name__ == "__main__":
    main()
