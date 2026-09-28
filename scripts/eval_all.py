#!/usr/bin/env python3
"""Scheduler for full checkpoint evals: runs x ckpts x release-margins -> jobs.

For each run dir it reads config.json, picks the matching eval_ckpt.py copy
(CEHJ_safe for HoloBrain runs, CEHJ_XVLA for X-VLA runs), selects the
checkpoints (latest by default: checkpoint.pt; --last-k 3 also takes the two
newest checkpoint_epoch<N>.pt snapshots), and emits one sbatch command per
(run, ckpt, margin, shard).

Sharding: 5 embodiments x 5 tasks x 50 episodes = 1250 sim episodes per
ckpt — far over one job's wallclock. Default --shard embodiment submits 5
jobs per ckpt (one embodiment each, 250 episodes); scene seeds are derived
from (task, embodiment, episode) so shards tile the same fixed scene grid.

Examples:
  # dry-run: print the commands for the latest ckpt of two runs
  python scripts/eval_all.py $TMP/cehj/runs/loo_franka $TMP/cehj/runs/l4o_franka

  # submit latest 3 ckpts, both release margins, one job per embodiment
  python scripts/eval_all.py --submit --last-k 3 \
      --release-margins 0 0.005 $TMP/cehj/runs/loo_franka

  # nominal baseline (no ckpt; config.json still used for the scene setup)
  python scripts/eval_all.py --submit --nominal $TMP/cehj/runs/loo_franka

  # every run under a directory
  python scripts/eval_all.py --submit --runs-dir $TMP/cehj/runs
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts" / "slurm_eval.sbatch"
EVAL_HB = ROOT / "main" / "train" / "eval_ckpt.py"
EVAL_XVLA = ROOT / "CEHJ_XVLA" / "main" / "train" / "eval_ckpt.py"
ALL_EMBODIMENTS = ["piper", "franka-panda", "ARX-X5", "ur5-wsg", "aloha-agilex"]


def is_xvla_run(run: Path) -> bool:
    cfg = run / "config.json"
    if not cfg.exists():
        return False
    return "X-VLA" in str(json.loads(cfg.read_text()).get("encoder_variant", ""))


def pick_ckpts(run: Path, last_k: int, snapshots_only: bool = False) -> list[str]:
    """checkpoint.pt first (latest published), then newest epoch snapshots.
    snapshots_only: skip checkpoint.pt — the "other two" after a recent run."""
    out = []
    if not snapshots_only and (run / "checkpoint.pt").exists():
        out.append("checkpoint.pt")
    epochs = []
    for f in run.glob("checkpoint_epoch*.pt"):
        m = re.match(r"checkpoint_epoch(\d+)\.pt$", f.name)
        if m:
            epochs.append((int(m.group(1)), f.name))
    for _, name in sorted(epochs, reverse=True):
        if len(out) < last_k:
            out.append(name)
    return out[: max(last_k, 1)]


def merge_eval_shards(run: Path) -> None:
    """Combine every eval_full/<tag>/episodes*.jsonl shard -> summary.json.

    summarize() lives in eval_ckpt.py; load it by path (the `main` package
    name differs per architecture, so a normal import won't do)."""
    import importlib.util

    eval_full = run / "eval_full"
    if not eval_full.exists():
        print(f"[merge] {run.name}: no eval_full/ — skipping")
        return
    spec = importlib.util.spec_from_file_location("eval_ckpt_mod", EVAL_HB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # leaf dirs holding episode shards: eval_full/<tag>[/t<budget>]/
    leafs = sorted({jf.parent for jf in eval_full.rglob("episodes*.jsonl")})
    for leaf in leafs:
        rows = []
        for jf in sorted(leaf.glob("episodes*.jsonl")):
            for line in jf.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                # error rows carry no metric keys — pad so they count as
                # failed episodes with NaN metrics, like in-run handling
                if "error" in r:
                    r.setdefault("success", 0.0)
                    for m in mod.SUMMARY_METRICS:
                        r.setdefault(m, float("nan"))
                rows.append(r)
        if not rows:
            continue
        summary = mod.summarize(rows)
        (leaf / "summary.json").write_text(json.dumps(summary, indent=2))
        ov = summary["overall"]
        print(f"[merge] {run.name}/{leaf.relative_to(eval_full)}: "
              f"{len(rows)} episodes  "
              f"success={ov['success']['mean']:.3f} "
              f"viol_any={ov['viol_any_frac']['mean']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", type=Path, help="run directories")
    p.add_argument("--runs-dir", type=Path, default=None,
                   help="evaluate every run dir directly under this path")
    p.add_argument("--last-k", type=int, default=1,
                   help="ckpts per run: checkpoint.pt + newest epoch snaps")
    p.add_argument("--snapshots-only", action="store_true",
                   help="skip checkpoint.pt; eval only the newest epoch "
                        "snapshots (--last-k of them)")
    p.add_argument("--release-margins", type=float, nargs="*", default=None,
                   help="override margins to sweep (default: config value, "
                        "one job set); e.g. --release-margins 0 0.005")
    p.add_argument("--shard", choices=["embodiment", "none"],
                   default="embodiment")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--video-every", type=int, default=1,
                   help="forwarded to eval_ckpt.py: record a video every N "
                        "episodes per combo")
    p.add_argument("--nominal", action="store_true",
                   help="nominal controller only (no ckpt)")
    p.add_argument("--submit", action="store_true",
                   help="actually sbatch; default is a dry-run print")
    p.add_argument("--merge", action="store_true",
                   help="no submission: for each run, merge every "
                        "eval_full/<tag>/episodes*.jsonl shard into a "
                        "combined summary.json")
    args = p.parse_args()

    runs = [Path(r).resolve() for r in args.runs]
    if args.runs_dir:
        runs += sorted(d for d in args.runs_dir.resolve().iterdir()
                       if d.is_dir() and (d / "checkpoint.pt").exists())

    if args.merge:
        if not runs:
            p.error("--merge needs run dirs")
        for run in runs:
            merge_eval_shards(run)
        return

    if not runs:
        p.error("no runs given")

    margins = args.release_margins if args.release_margins else [None]
    cmds = []
    for run in runs:
        run = run.resolve()
        script = EVAL_XVLA if is_xvla_run(run) else EVAL_HB
        if args.nominal:
            ckpts = [None]
        else:
            ckpts = pick_ckpts(run, args.last_k, args.snapshots_only)
            if not ckpts:
                print(f"[skip] {run.name}: no checkpoints", file=sys.stderr)
                continue
        shards = ALL_EMBODIMENTS if args.shard == "embodiment" else [None]
        for ckpt in ckpts:
            for rm in margins:
                for emb in shards:
                    tag = ("nominal" if ckpt is None
                           else Path(ckpt).stem.replace("checkpoint_", ""))
                    name = f"ev_{run.name[:24]}_{tag}"
                    if rm is not None:
                        name += f"_rm{rm:g}"
                    if emb:
                        name += f"_{emb.split('-')[0][:6]}"
                    cmd = ["sbatch", f"--job-name={name}", str(SBATCH),
                           str(script), "--run", str(run),
                           "--episodes", str(args.episodes),
                           "--video-every", str(args.video_every)]
                    if ckpt is None:
                        cmd.append("--nominal")
                    else:
                        cmd += ["--ckpt", ckpt]
                    if rm is not None:
                        cmd += ["--filter-release-margin", str(rm)]
                    if emb:
                        cmd += ["--embodiments", emb]
                    cmds.append(cmd)
    for cmd in cmds:
        print(" ".join(cmd))
        if args.submit:
            subprocess.run(cmd, check=True)
    print(f"[eval_all] {len(cmds)} job(s) "
          f"{'submitted' if args.submit else '(dry run — pass --submit)'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
