"""The reused part of the safety-agnostic VLA's training data (Yuxuan 2026-11-21, collect_nominal.py has the design):
the constraint experiment's careless A demos of T1, T3 and T7, with the task alone as their prompt. An A demo is the
careless nominal run of its scene, kept when it finished the task (harm allowed); half of its prompts named an item
NOT in the way ("<task>, and <constraint>."), so this view writes new instruction files with the other half only (the
strings that end with none of the run's constraint sentences, cycled back to 100 per list).

    python -m vla_finetuning.items.nominal_view --out /abs/vla_finetuning/data/demos_nominal

Per task PER_TASK demos (T1 PER_TASK / 5 per host), the lowest seeds, the main set before the 11-08 T1 set
(demos_constraint_t1x). The T7 A demos ran the experiment's careless offset (constraint.KNOBS t7_offset 2.75-3.25 cm,
success window + 1.5 cm), the T1 / T3 ones the tasks as built.

<out>/<task>/<config>/ mirrors the source config: seed.txt copied (the packer indexes it by episode number), data/
symlinks to the chosen episodes, instructions/ new files, attempts.jsonl their rows. Each task is built into
<out>/<task>.tmp and renamed, so an existing <out>/<task> is complete (kept as is). Counts: <out>/view.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from .. import common as C

VF = Path(__file__).resolve().parents[1]
SOURCES = (VF / "data" / "demos_constraint", VF / "data" / "demos_constraint_t1x")
TASKS = ("selective_avoid", "push_zone", "gap_place")
PER_TASK = 100
T1_HOSTS = ("place_container_plate", "place_burger_fries", "stack_blocks_two", "stack_bowls_two", "place_bread_basket")


def host_of(cfg: str) -> str:
    head = cfg.rsplit("_", 2)[0] if cfg.endswith("_A") else cfg
    return head


def candidates(task: str) -> list[dict]:
    """Every kept A demo of the task: {"root", "config", "episode", "seed", "row", "run"} (run = its runs.jsonl row,
    which lists the constraint sentences its prompts could use)."""
    out = []
    for order, root in enumerate(SOURCES):
        tdir = root / task
        if not tdir.is_dir():
            continue
        for cfg in sorted(p for p in tdir.iterdir() if p.is_dir() and p.name.endswith("_A")):
            shard = cfg.name[:-2]
            runs = {}
            for r in C.read_jsonl(Path(f"{root}_logs") / task / shard / "runs.jsonl"):
                if r.get("kind") == "A" and r.get("reason") == "kept" and r.get("config") == cfg.name:
                    runs[int(r["episode"])] = r
            for r in C.read_jsonl(cfg / "attempts.jsonl"):
                if not r.get("kept"):
                    continue
                e = int(r["episode"])
                if e not in runs or int(runs[e]["seed"]) != int(r["seed"]):
                    raise SystemExit(f"{cfg}: episode {e} (seed {r['seed']}) has no matching kept row in runs.jsonl")
                out.append({"root": root, "order": order, "config": cfg.name, "episode": e, "seed": int(r["seed"]),
                            "row": r, "run": runs[e]})
    return out


def choose(task: str) -> list[dict]:
    cands = sorted(candidates(task), key=lambda d: (d["order"], d["seed"]))
    if task != "selective_avoid":
        return cands[:PER_TASK]
    per = PER_TASK // len(T1_HOSTS)
    by = defaultdict(list)
    for d in cands:
        by[host_of(d["config"])].append(d)
    bad = sorted(set(by) - set(T1_HOSTS))
    if bad:
        raise SystemExit(f"T1 configs of unknown hosts: {bad}")
    return [d for h in T1_HOSTS for d in by[h][:per]]


def constraint_suffixes(run: dict) -> list[str]:
    cs = run.get("constraints") or []
    if isinstance(cs, str):
        cs = json.loads(cs)
    return [", and " + re.sub(r"[\s.!]+$", "", str(c).strip()) + "." for c in cs]


def task_only(instr: dict, suffixes: list[str]) -> tuple[dict, dict]:
    """The prompt lists with the joined strings dropped, cycled back to their length. -> (lists, kept counts)."""
    out, n = {}, {}
    for key, lst in instr.items():
        if not isinstance(lst, list):
            out[key] = lst
            continue
        keep = [s for s in lst if not any(s.endswith(x) for x in suffixes)]
        n[key] = len(keep)
        out[key] = keep
    pool = [s for key in n for s in out[key]]
    if not pool:
        raise SystemExit("a demo with no task-only prompt")
    for key in n:
        src = out[key] or pool
        L = len(instr[key])
        out[key] = (src * (L // len(src) + 1))[:L]
    return out, n


def build_task(task: str, out: Path) -> dict:
    dst_t, tmp = out / task, out / f"{task}.tmp"
    if dst_t.exists():
        return json.loads((out / "view.json").read_text()).get(task, {}) if (out / "view.json").exists() else {}
    if tmp.exists():
        shutil.rmtree(tmp)
    picks = choose(task)
    by_cfg = defaultdict(list)
    for d in picks:
        by_cfg[(d["root"], d["config"])].append(d)
    kept_counts = []
    for (root, cfg), ds in sorted(by_cfg.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        src, dst = root / task / cfg, tmp / cfg
        if dst.exists():
            raise SystemExit(f"two sources with config {cfg}")
        (dst / "data").mkdir(parents=True)
        (dst / "instructions").mkdir()
        shutil.copy2(src / "seed.txt", dst / "seed.txt")
        seeds = (src / "seed.txt").read_text().split()
        rows = []
        for d in sorted(ds, key=lambda d: d["episode"]):
            e = d["episode"]
            if int(seeds[e]) != d["seed"]:
                raise SystemExit(f"{src}: episode {e} has seed {seeds[e]} in seed.txt, {d['seed']} in attempts")
            f = (src / "data" / f"episode{e}.hdf5").resolve()
            if not f.is_file():
                raise SystemExit(f"missing {f}")
            (dst / "data" / f.name).symlink_to(f)
            instr = json.loads((src / "instructions" / f"episode{e}.json").read_text())
            new, n = task_only(instr, constraint_suffixes(d["run"]))
            kept_counts.append(n)
            C.atomic_write(dst / "instructions" / f"episode{e}.json", json.dumps(new, indent=1))
            rows.append({**d["row"], "instruction_example": new["seen"][0], "source": str(src)})
        (dst / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(dst_t)
    per_host = defaultdict(int)
    for d in picks:
        per_host[host_of(d["config"]) if task == "selective_avoid" else "all"] += 1
    return {"demos": len(picks), "per_host": dict(per_host),
            "task_only_prompts_min": {k: min(n[k] for n in kept_counts) for k in kept_counts[0]},
            "task_only_prompts_mean": {k: round(sum(n[k] for n in kept_counts) / len(kept_counts), 1)
                                       for k in kept_counts[0]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for t in TASKS:
        counts[t] = build_task(t, out)
        print(f"[nominal_view] {t}: {counts[t]}")
    C.atomic_write(out / "view.json", json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
