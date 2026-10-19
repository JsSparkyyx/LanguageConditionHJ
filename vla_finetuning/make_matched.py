"""Matched demo sets for the safe-vs-nominal contrast (PLAN.md row 5; review 10-15, B2): per task, the seeds
kept by BOTH demonstrators, the same number for both, so the two fine-tuning sets cover the same layouts.

    python -m vla_finetuning.make_matched --safe-root .../data/demos_safe --nominal-root .../data/demos_nominal \
        --out-safe .../data/matched_safe --out-nominal .../data/matched_nominal [--per-task 60]

Each output is a packer root: <task>/matched/{seed.txt, data/episode{j}.hdf5 -> symlink to the source,
instructions/episode{j}.json -> symlink, attempts.jsonl (the source rows, re-indexed)}. Seeds are taken in
the order the safe demonstrator kept them. Light CPU work (symlinks and json), fine on a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import common as C
from .demo_io import write_seed_txt


def kept_by_seed(root: Path, task: str) -> dict:
    """seed -> (config dir, kept row) of every kept demo of the task under root."""
    out = {}
    for cfg in sorted(p for p in (root / task).iterdir() if p.is_dir()):
        for r in C.read_jsonl(cfg / "attempts.jsonl"):
            if r.get("kept"):
                out[int(r["seed"])] = (cfg, r)
    return out


def write_set(out_root: Path, task: str, picks: list) -> None:
    d = out_root / task / "matched"
    if d.exists():
        raise SystemExit(f"{d} exists")
    (d / "data").mkdir(parents=True)
    (d / "instructions").mkdir()
    rows = []
    for j, (cfg, r) in enumerate(picks):
        k = int(r["episode"])
        (d / "data" / f"episode{j}.hdf5").symlink_to(cfg / "data" / f"episode{k}.hdf5")
        (d / "instructions" / f"episode{j}.json").symlink_to(cfg / "instructions" / f"episode{k}.json")
        rows.append({**r, "episode": j, "source": f"{cfg}/episode{k}"})
    write_seed_txt(d / "seed.txt", [int(r["seed"]) for _, r in picks])
    C.atomic_write(d / "attempts.jsonl", "".join(json.dumps(r, default=C._json_default) + "\n" for r in rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe-root", required=True)
    ap.add_argument("--nominal-root", required=True)
    ap.add_argument("--out-safe", required=True)
    ap.add_argument("--out-nominal", required=True)
    ap.add_argument("--per-task", type=int, default=None, help="cap per task (default: the whole intersection)")
    ap.add_argument("--tasks", nargs="*", default=list(C.TASKS))
    args = ap.parse_args(argv)
    sroot, nroot = Path(args.safe_root), Path(args.nominal_root)
    summary = {}
    for task in args.tasks:
        s, n = kept_by_seed(sroot, task), kept_by_seed(nroot, task)
        order = sorted(s, key=lambda seed: (str(s[seed][0]), int(s[seed][1]["episode"])))
        common = [seed for seed in order if seed in n][: args.per_task]
        write_set(Path(args.out_safe), task, [s[x] for x in common])
        write_set(Path(args.out_nominal), task, [n[x] for x in common])
        fr_s = sum(int(s[x][1]["n_frames"]) for x in common)
        fr_n = sum(int(n[x][1]["n_frames"]) for x in common)
        summary[task] = {"safe_kept": len(s), "nominal_kept": len(n), "matched": len(common),
                         "frames_safe": fr_s, "frames_nominal": fr_n}
        print(f"[matched] {task}: safe {len(s)}, nominal {len(n)}, both {len(common)}; frames safe {fr_s}, "
              f"nominal {fr_n}")
    for out in (args.out_safe, args.out_nominal):
        C.atomic_write(Path(out).parent / f"{Path(out).name}_summary.json", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
