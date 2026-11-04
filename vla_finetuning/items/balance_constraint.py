"""The balanced training view of the constraint demos (Yuxuan 2026-11-04: except T6, every task gets the same
number of training demos). Per complete scene (a kept careless A demo and a kept careful B demo) the view holds the
A demo and ONE B demo: a scene with two items in the way has a B demo for each, and one of them is drawn, seeded by
the scene. So every task but T6 has 100 careless + 100 careful demos; T6 keeps its 100 safe pours. A scene whose B
runs all failed is left out, its A demo too.

    python -m vla_finetuning.items.balance_constraint --root <demos_constraint> --out <demos_constraint_bal>

The view mirrors <root>/<task>/<config>/: seed.txt copied (the packer indexes it by episode number), data/ and
instructions/ hold symlinks to the chosen episodes only, attempts.jsonl only their rows (pack.py's count check).
Built into <out>.tmp and renamed, so <out> is complete when it exists (then it is kept as is). The counts go to
<out>.json.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def choose(root: Path) -> dict:
    """{task: {config: [attempts rows of the chosen demos]}}"""
    out = {}
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        a_rows, b_rows, s_rows = {}, defaultdict(list), []
        for cfg in sorted(p for p in tdir.iterdir() if p.is_dir()):
            shard, kind = cfg.name.rsplit("_", 1)
            p = cfg / "attempts.jsonl"
            for line in (p.read_text().splitlines() if p.exists() else []):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("kept"):
                    continue
                key = (shard, int(r["seed"]))
                if kind == "A":
                    if key in a_rows:
                        raise SystemExit(f"{tdir.name}: two kept A demos of {key}")
                    a_rows[key] = (cfg.name, r)
                elif kind.startswith("B"):
                    b_rows[key].append((cfg.name, r))
                elif kind == "S":
                    s_rows.append((cfg.name, r))
                else:
                    raise SystemExit(f"unknown config kind: {cfg}")
        pick = defaultdict(list)
        for cfg, r in s_rows:
            pick[cfg].append(r)
        for key in sorted(a_rows):
            if key not in b_rows:
                continue
            cfg, r = a_rows[key]
            pick[cfg].append(r)
            cfg, r = random.Random(key[1]).choice(sorted(b_rows[key], key=lambda x: x[0]))
            pick[cfg].append(r)
        out[tdir.name] = dict(pick)
    return out


def build(root: Path, out: Path, picks: dict) -> dict:
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    counts = {}
    for task, cfgs in picks.items():
        n = defaultdict(int)
        for cfg, rows in sorted(cfgs.items()):
            src, dst = root / task / cfg, tmp / task / cfg
            (dst / "data").mkdir(parents=True)
            (dst / "instructions").mkdir()
            shutil.copy2(src / "seed.txt", dst / "seed.txt")
            seeds = (src / "seed.txt").read_text().split()
            for r in rows:
                e = int(r["episode"])
                if int(seeds[e]) != int(r["seed"]):
                    raise SystemExit(f"{src}: episode {e} has seed {seeds[e]} in seed.txt, {r['seed']} in attempts")
                for sub, ext in (("data", "hdf5"), ("instructions", "json")):
                    f = (src / sub / f"episode{e}.{ext}").resolve()
                    if not f.is_file():
                        raise SystemExit(f"missing {f}")
                    (dst / sub / f.name).symlink_to(f)
            (dst / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            n[cfg.rsplit("_", 1)[1][0]] += len(rows)
        counts[task] = {"careless (A)": n.get("A", 0), "careful (B)": n.get("B", 0), "safe only (S)": n.get("S", 0)}
    tmp.rename(out)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if out.exists():
        print(f"[balance] {out} exists: kept ({out.with_suffix('.json').read_text().strip()})")
        return 0
    counts = build(root, out, choose(root))
    out.with_suffix(".json").write_text(json.dumps(counts, indent=1))
    for t, c in counts.items():
        print(f"[balance] {t}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
