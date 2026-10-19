"""Pack a demo root into HoloBrain's LMDB with HoloBrain's own packer, then check it loads through the
training config (PLAN.md, "Fine-tuning").

    scripts/lang/in_container.sh python -m vla_finetuning.pack \
        --root /abs/vla_finetuning/data/demos_safe --out /abs/vla_finetuning/data/lmdb/safe

--root holds <task>/<config>/{seed.txt, data/, instructions/} only (collect_demos.py). Every config
directory under it is packed (uuid = <task>_<config>_seed<seed>, unique across shards). --out must not
exist: the packer never clears old keys. --check builds the training dataset from configs/ with
VF_LMDB=<out>, loads a few samples, and compares the packed episode and frame counts with attempts.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import common as C


def expected(root: Path, tasks=None) -> dict:
    """{task: (episodes, frames)} of the kept demos under root (only `tasks`, if given)."""
    out = {}
    for t in sorted(p for p in root.iterdir() if p.is_dir()):
        if tasks and t.name not in tasks:
            continue
        n_ep = n_fr = 0
        for cfg in sorted(p for p in t.iterdir() if p.is_dir()):
            for r in C.read_jsonl(cfg / "attempts.jsonl"):
                if r.get("kept"):
                    n_ep += 1
                    n_fr += int(r["n_frames"])
        out[t.name] = (n_ep, n_fr)
    return out


def check(out: Path, root: Path, n_samples: int = 4, tasks=None) -> None:
    import numpy as np

    os.environ["VF_LMDB"] = str(out)
    sys.path.insert(0, str(C.VF / "configs"))
    cwd = os.getcwd()
    os.chdir(C.VF)                                   # the config's BERT path is relative to vla_finetuning/
    try:
        import config_holobrain_gd_common as cfgmod

        ds = cfgmod.build_training_dataset(cfgmod.config, lazy_init=False)
        print(f"[pack] training dataset: {len(ds)} samples")
        rng = np.random.default_rng(0)
        for i in rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False):
            item = ds[int(i)]
            desc = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in item.items()}
            print(f"[pack] sample {int(i)}: text {item.get('text')!r}\n        {desc}")
    finally:
        os.chdir(cwd)
    exp = expected(root, tasks)
    tot_ep = sum(v[0] for v in exp.values())
    tot_fr = sum(v[1] for v in exp.values())
    print(f"[pack] attempts.jsonl: {tot_ep} kept episodes, {tot_fr} frames ({exp})")
    if len(ds) != tot_fr:
        raise SystemExit(f"[pack] dataset has {len(ds)} samples, the demos {tot_fr} frames")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    stray = [p.name for p in root.iterdir() if not p.is_dir()]
    if stray:
        raise SystemExit(f"{root} must hold task directories only (the packer lists every entry): {stray}")
    if out.exists():
        raise SystemExit(f"{out} exists: pack into a fresh directory")
    cmd = [sys.executable, "-m", "robo_orchard_lab.dataset.robotwin.robotwin_packer",
           "--input_path", str(root), "--output_path", str(out)]
    if args.tasks:
        cmd += ["--task_names", ",".join(args.tasks)]
    print("[pack]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    meta = {"root": str(root), "tasks": args.tasks, "expected": expected(root, args.tasks)}
    (out / "vf_pack.json").write_text(json.dumps(meta, indent=1))
    if not args.no_check:
        check(out, root, tasks=args.tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
