"""Workqueue command file for the item-scene safe demonstrations (Yuxuan 2026-10-27: 100 kept per task, T1's 100 =
20 per host): one line per (task, config), each a jobs/collect_items_loop.sh shard of collect_items.py.

    python -m vla_finetuning.items.make_queue [--out runs/vf_items/collect_items_safe.cmds] [--tasks ...]
    python -m main.lang.workqueue --cmds <out> --state <state dir> --slots 2      # inside a GPU stage

- selective_avoid (T1): one config per host, named after the host, 20 kept each; host h (its index in the T1 yaml's
  host_tasks) walks seeds 300000 + 1000 h + 100 ... 300000 + 1000 (h + 1) - 1 in steps of 1. The +100 keeps clear of
  the collision-avoidance evaluation grid, whose stack_blocks_two (host 2) scenes and redraws use 302000-302049
  (eval_ckpt.episode_seed: 70000 + 2 x 100000 + 3 x 10000 + ep, redraws + 1000 r; 50 episodes).
- every other task (index i in ITEM_TASKS: push_zone 1, tray_transport 2, clean_pour 3, gap_place 4): 4 configs
  s0..s3 of 25 each, config c walking 300000 + 10000 i + c in steps of 4 below 300000 + 10000 (i + 1).

Every shard writes head-camera videos (--video: episodes 0, 10, 20 and the first 5 rejected attempts of the config,
<config>/videos/). All seeds stay below 1M (the language filter's episode ids, main/lang/config.py). The demo root is
vla_finetuning/data/demos_items_safe of this checkout unless --root says otherwise; nothing is written into it here
(pack.py wants task directories only there).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import common as C
from .instructions import ITEM_TASKS, T1

SEED_BASE = 300000
T1_HOST_SPAN = 1000           # seeds per T1 host
T1_HOST_OFFSET = 100          # clear of the collision-avoidance evaluation grid's 302000-302049 (stack_blocks_two)
TASK_SPAN = 10000             # seeds per task
N_CONFIGS = 4                 # configs per non-T1 task
STEP = 4                      # their seed step (configs interleave)
TARGET = {"t1_per_host": 20, "per_config": 25}
MAX_ATTEMPTS = 40
LOOP = "vla_finetuning/items/jobs/collect_items_loop.sh"


def t1_hosts() -> tuple[str, ...]:
    import main.lang  # noqa: F401
    from main.item_envs.registry import get_task_entry

    return tuple(get_task_entry(T1).host_tasks)


def shards(tasks=ITEM_TASKS) -> list[dict]:
    """One dict per (task, config): task, host, config, seed_start, seed_step, seed_stop, target."""
    out = []
    for task in tasks:
        i = ITEM_TASKS.index(task)
        if task == T1:
            for h, host in enumerate(t1_hosts()):
                lo = SEED_BASE + T1_HOST_SPAN * h
                out.append({"task": task, "host": host, "config": host, "seed_start": lo + T1_HOST_OFFSET,
                            "seed_step": 1, "seed_stop": lo + T1_HOST_SPAN, "target": TARGET["t1_per_host"]})
        else:
            lo = SEED_BASE + TASK_SPAN * i
            for c in range(N_CONFIGS):
                out.append({"task": task, "host": None, "config": f"s{c}", "seed_start": lo + c, "seed_step": STEP,
                            "seed_stop": lo + TASK_SPAN, "target": TARGET["per_config"]})
    return out


def command(sh: dict, root: Path) -> str:
    host = f" --host {sh['host']}" if sh["host"] else ""
    return (f"bash {LOOP} --task {sh['task']}{host} --root {root} --config-name {sh['config']} "
            f"--seed-start {sh['seed_start']} --seed-step {sh['seed_step']} --seed-stop {sh['seed_stop']} "
            f"--target {sh['target']} --max-attempts {MAX_ATTEMPTS} --video")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.REPO / "runs" / "vf_items" / "collect_items_safe.cmds"))
    ap.add_argument("--root", default=str(C.VF / "data" / "demos_items_safe"), help="absolute demo root")
    ap.add_argument("--tasks", nargs="*", default=list(ITEM_TASKS), choices=ITEM_TASKS)
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_absolute():
        raise SystemExit("--root must be absolute (Env chdirs to the RoboTwin root)")
    out = Path(args.out)
    if root.resolve() in out.resolve().parents:
        raise SystemExit(f"{out} lies inside the demo root {root}: pack.py wants task directories only there")
    lines = [command(sh, root) for sh in shards(args.tasks)]
    out.parent.mkdir(parents=True, exist_ok=True)
    head = "# item-scene safe demos for HoloBrain (vla_finetuning/items/make_queue.py)\n"
    out.write_text(head + "\n".join(lines) + "\n")
    print(f"[make_queue] {len(lines)} commands -> {out}")
    print(f"  python -m main.lang.workqueue --cmds {out} --state {out.with_suffix('')}_state --slots 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
