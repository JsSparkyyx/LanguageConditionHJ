"""Careless demonstrations for the safety-agnostic VLA (Yuxuan 2026-11-21: "finetune a vla that is safe agnostic ...
with unsafe success traj, the ultimate goal is to present as a variant of the nominal curobo planner"). The tasks the
existing careless demos do not cover; T1 / T3 / T7 reuse the constraint experiment's careless A demos
(nominal_view.py). One careless run per seed, the prompt the task alone:

    clean_pour  T6, the careless pour as built (nominal_mode "careless", the task's own sampled aim / height / tilt
                rate = the language filter's nominal; collect_constraint.run_demo kind "A" with constraint.KNOBS,
                whose T6 knobs are all unset). Yuxuan 11-21 ("careless pours"): it never fills the target bowl, so the
                success check is waived: kept when the run ended cleanly (no error, no failed plan), beads left the
                mug (some bead outside the mug and the target, or success), nothing flown, enough frames.
    tray_slide  T8 (glass + can; Yuxuan 11-21: the sliding tray is this task, not T5), the careless drag of the
                language filter's nominal mix without its safe slow drag (Yuxuan 11-21, "medium + fast, 2:1"):
                fast when seed % 3 == 0, else medium (collect_tray.run_demo kind "N"); kept on the task alone (tray
                in the frame, drag planned, nothing flown, enough frames); either item may fall.

    scripts/lang/in_container.sh python -m vla_finetuning.items.collect_nominal --task tray_slide \
        --root /abs/vla_finetuning/data/demos_nominal --shard n0 --seed-start 730000 --seed-step 4 \
        --seed-stop 740000 --target 25
    python -m vla_finetuning.items.collect_nominal --shards     # "task:shard:target:start:step:stop" per shard

Files: <root>/<task>/<shard>_N/ (seed.txt, data/episode{k}.hdf5, instructions/episode{k}.json, attempts.jsonl: the
packer's layout); every run's row in <root>_logs/<task>/<shard>/runs.jsonl ("started" lines first), settings.json,
runner/<n>/, videos/ (the first --videos kept demos, head camera). Exit 0: --target kept or seeds exhausted; 75:
continue in a fresh process (--max-runs, the stage deadline, a renderer error); 3: the planners or the code are broken.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from .. import common as C
from ..demo_io import write_video
from . import collect_constraint as CC
from . import collect_tray as CT
from . import constraint as X
from . import keep as K
from .collect_items import MAX_STEPS as ITEM_MAX_STEPS, _plain, diag_overrides, pin_settings
from .monitor import FRAME_EVERY, SETTLE_STEPS

TASKS = ("clean_pour", "tray_slide")
KIND = "N"
PER_TASK = 100
N_SHARDS = 4
SEED_BASE = {"tray_slide": 730000, "clean_pour": 740000}   # free ranges (demos 500000-709999, tests 710000+, 9xxxxx)
SEED_SPAN = 10000
RETRY_REASONS = ("render_error", "planner_error", "code_error")
RETRY_MAX = 3
CODE_STREAK = 3
_hard_exit = False


def shards() -> list[dict]:
    return [{"task": t, "shard": f"n{i}", "target": PER_TASK // N_SHARDS + (1 if i < PER_TASK % N_SHARDS else 0),
             "seed_start": SEED_BASE[t] + i, "seed_step": N_SHARDS, "seed_stop": SEED_BASE[t] + SEED_SPAN}
            for i in range(N_SHARDS) for t in TASKS]        # interleaved: a job's first slots run both tasks


def tray_behaviour(seed: int) -> str:
    """Medium : fast = 2 : 1 over the seeds (each shard's seeds step by 4, so seed % 3 cycles in every shard)."""
    return "fast" if int(seed) % 3 == 0 else "medium"


def poured(row: dict) -> bool:
    return bool(row.get("task_success")) or sum(int(v) for v in (row.get("bead_counts") or {}).values()) > 0


def run_one(task: str, seed: int, out_root: Path, max_steps: int):
    """-> (row, frames, task-only instructions | None); row["reason"] = "kept" or why not."""
    if task == "tray_slide":
        beh = tray_behaviour(seed)
        row, frames, sc = CT.run_demo(seed, KIND, out_root, max_steps, behaviour=beh)
        return row, frames, (sc or {}).get("task_instr")
    row, mon, sc = CC.run_demo(task, seed, None, "A", out_root, max_steps, X.KNOBS)
    frames = list(mon.frames) if mon is not None else []
    row["kind"] = KIND
    row.pop("constraints", None)
    row.pop("constraint_groups", None)
    if row.get("reason") == "task_failed":                    # T6: the careless pour misses the target by design
        if row.get("flown"):
            row["reason"] = f"flown:{row['flown'][0]}"
        elif not poured(row):
            row["reason"] = "no_pour"
        elif len(frames) < K.MIN_FRAMES:
            row["reason"] = "few_frames"
        else:
            row["reason"] = "kept"
    return row, frames, (sc or {}).get("task_instr")


def settings_of(args) -> dict:
    task_file = C.REPO / "envs" / "robotwin_tasks" / f"{'bowl_pour' if args.task == 'clean_pour' else args.task}.py"
    return {"task": args.task, "embodiment": C.EMBODIMENT, "design": "nominal_2026-11-21", "kind": KIND,
            "behaviour": "seed % 3 == 0 fast, else medium" if args.task == "tray_slide" else "careless as built",
            "knobs": X.task_knobs(args.task) if args.task != "tray_slide" else None,
            "task_file_md5": hashlib.md5(task_file.read_bytes()).hexdigest()[:12] if task_file.exists() else None,
            "max_steps": int(args.max_steps), "frame_every": FRAME_EVERY, "settle_steps": SETTLE_STEPS,
            "min_frames": K.MIN_FRAMES, "seed_start": int(args.seed_start), "seed_step": int(args.seed_step),
            "seed_stop": int(args.seed_stop)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--root", required=True, help="absolute demo root (packer input_path)")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--target", type=int, default=25, help="kept demos")
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-step", type=int, default=1)
    ap.add_argument("--seed-stop", type=int, required=True)
    ap.add_argument("--max-runs", type=int, default=10, help="runs (Env builds) per process, then exit 75")
    ap.add_argument("--max-steps", type=int, default=None, help="default: the task collector's")
    ap.add_argument("--videos", type=int, default=3, help="head-camera videos of the first N kept demos")
    args = ap.parse_args(argv)
    if args.max_steps is None:
        args.max_steps = CT.MAX_STEPS if args.task == "tray_slide" else ITEM_MAX_STEPS[args.task]
    if not (0 <= args.seed_start < args.seed_stop <= 1_000_000) or args.seed_step < 1:
        ap.error("want 0 <= --seed-start < --seed-stop <= 1000000 and --seed-step >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    bad = diag_overrides()
    if bad:
        raise SystemExit(f"diagnostic overrides set: {bad}: unset them")
    root = Path(args.root)
    if not root.is_absolute():
        raise SystemExit("--root must be absolute (Env chdirs to the RoboTwin root)")
    log_d = Path(f"{root}_logs") / args.task / args.shard
    log_d.mkdir(parents=True, exist_ok=True)
    import main.lang  # noqa: F401
    from main.lang.locks import WriterLock

    lock = WriterLock(log_d, wait_s=200.0)
    try:
        return _collect(args, root, log_d, lock)
    finally:
        lock.release()


def _collect(args, root: Path, log_d: Path, lock) -> int:
    global _hard_exit
    pin_settings(log_d, settings_of(args))
    runs_p = log_d / "runs.jsonl"
    rows = C.read_jsonl(runs_p)
    cfg = CT.Config(root / args.task / f"{args.shard}_{KIND}")
    name = f"{args.task}/{args.shard}"
    # a seed is tried when it has a result for a reason of the scene, or RETRY_MAX starts (a killed process)
    starts, done = {}, set()
    for r in rows:
        s = int(r["seed"])
        if "started" in r:
            starts[s] = starts.get(s, 0) + 1
        elif r.get("reason") not in RETRY_REASONS:
            done.add(s)
    tried = done | {s for s, n in starts.items() if n >= RETRY_MAX}
    n_vid = sum(1 for r in rows if r.get("video"))
    print(f"[collect_nominal] {name}: {len(cfg.kept)}/{args.target} kept, {len(tried)} seeds tried", flush=True)
    n_run, code_streak, times = 0, 0, []
    n_started = sum(1 for r in rows if "started" in r)
    for seed in range(args.seed_start, args.seed_stop, args.seed_step):
        if len(cfg.kept) >= args.target:
            break
        if seed in tried:
            continue
        if n_run + 1 > args.max_runs:
            print(f"[collect_nominal] {n_run} runs in this process: continue in a fresh one")
            return 75
        mean_t = sum(times) / len(times) if times else 240.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            print(f"[collect_nominal] stage deadline near (mean run {mean_t:.0f} s): continue later")
            return 75
        n_run += 1
        idx, n_started = n_started, n_started + 1
        att = log_d / "runner" / str(idx)
        if att.exists():
            shutil.rmtree(att)
        lock.check()
        C.append_jsonl(runs_p, {"seed": seed, "started": KIND, "run": idx, "at": time.strftime("%F %T")})
        t0 = time.time()
        try:
            row, frames, instr = run_one(args.task, seed, att, args.max_steps)
        except Exception as exc:  # noqa: BLE001 - this collector's own code
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row, frames, instr = ({"task": args.task, "seed": seed, "kind": KIND, "reason": K.error_reason(err),
                                   "error": err[:500], "detail": "run_one"}, [], None)
        reason = row.get("reason") or "error"
        if reason == "kept" and not instr:
            reason = row["reason"] = "code_error"
            row["detail"] = "no task instructions"
        row.update(kind=KIND, run=idx, shard=args.shard, n_frames=len(frames), wall_s=time.time() - t0)
        lock.check()
        if reason == "kept":
            k = cfg.add(row, frames, instr)
            row.update(episode=k, config=cfg.d.name, instruction_example=instr["seen"][0])
            if n_vid < args.videos:
                vname = f"{args.task}_{row.get('behaviour') or 'careless'}_seed{seed}.mp4"
                try:
                    write_video(log_d / "videos" / vname, frames)
                    row["video"] = vname
                    n_vid += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[collect_nominal] video failed: {exc}")
        row.pop("probe", None)
        C.append_jsonl(runs_p, _plain(row))
        times.append(row["wall_s"])
        print(f"[collect_nominal] {name} seed {seed} ({row.get('behaviour') or 'careless'}): {reason} "
              f"(success {row.get('task_success')}, broken {row.get('broken')}, beads {row.get('bead_counts')}, "
              f"peak tilt {row.get('peak_tilt')}, v max {row.get('max_speed')}, {row['n_frames']} frames, "
              f"{row['wall_s']:.0f} s) -> {len(cfg.kept)}/{args.target} kept", flush=True)
        if reason == "render_error":
            _hard_exit = True
            return 75
        if reason == "planner_error":
            print(f"[collect_nominal] refusing to run: {row.get('detail')}", flush=True)
            return 3
        code_streak = code_streak + 1 if reason == "code_error" else 0
        if code_streak >= CODE_STREAK:
            print(f"[collect_nominal] {code_streak} code errors in a row (last: {row.get('error')})", flush=True)
            return 3
        del frames
    print(f"[collect_nominal] {name}: {len(cfg.kept)}/{args.target} kept: done")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--shards"]:
        print(" ".join(f"{d['task']}:{d['shard']}:{d['target']}:{d['seed_start']}:{d['seed_step']}:{d['seed_stop']}"
                       for d in shards()))
        sys.exit(0)
    rc = main()
    if _hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)
