"""Closed-loop test of a HoloBrain model in the item scenes (T1 / T3 / T5 / T6 / T7): the same scenes for every model,
judged by the demos' keep rule (vla_finetuning/items/keep.py).

    scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_items \
        --model-dir /abs/vla_finetuning/models/<name>/model --model-name <name> \
        --out /abs/vla_finetuning/results_items/<name> [--tasks ...] [--episodes 20]

The scene is the demo collector's (collect_items.run_attempt: the same ItemEpisodeRunner, the same filters: a T1
layout with no extra object, or whose host sentence would also name an extra object, is no scene), built from held-
out seeds (EVAL_SEED_BASE: 800000 + 10000 x task index, T1 + 1000 x host index, walked upward and skipping seeds
with no scene, so every model sees the identical list). The model replaces the expert: task.play_once becomes
HoloBrain's loop (vla_finetuning.evaluate.HoloBrainPolicy, task.get_obs() + one held-out ("unseen") instruction of
the demos' generator; --chunk of the 64 predicted actions per inference through RoboTwin's take_action, until the
latched eval_success or the action limit STEP_LIMIT). The runner records every rule's h per control tick as in the
collection, and the DemoMonitor checks T1's contact with any extra object at every physics step (no abort).

Per episode: success (eval_success or check_success at the end), broken (the protected rules with h < 0 at some
tick, keep.score_trace), contact (T1: an extra object touched, or force >= KEEP_FORCE_N), flown (an item above
MAX_ITEM_Z), safe = success and nothing broken / touched / flown (the demos' keep rule without its planner checks).
Rows: <out>/episodes_<task>.jsonl (resumable); <out>/summary.json, per task and overall, with Wilson intervals. One
child process per task (bounds RoboTwin task inits per process). Exit 75 = stopped at the stage deadline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .. import common as C
from . import instructions as I
from . import keep as K
from .instructions import ITEM_TASKS, T1

EVAL_SEED_BASE = 800000
TASK_SPAN = 10000
HOST_SPAN = 1000
# actions per episode: T1 RoboTwin's own limits for its hosts (common.step_limit); the item tasks 2.5-3.5 x the median
# kept demo (2026-10-29: push 340 frames, tray 139, pour 260, gap 244)
STEP_LIMIT = {"push_zone": 1200, "tray_transport": 500, "clean_pour": 800, "gap_place": 800}
MAX_SEED_TRIES = 200            # seeds walked per wanted episode before giving up on a task
_DEVNULL = open(os.devnull, "w")


class _NoRecorder:
    """DemoMonitor's recorder slot: no demo frames are written in a test."""

    def __init__(self):
        self.frames: list = []

    def __call__(self, n) -> None:
        return None


def eval_seed(task: str, ep: int, host_idx: int = 0) -> int:
    i = ITEM_TASKS.index(task)
    return EVAL_SEED_BASE + TASK_SPAN * i + (HOST_SPAN * host_idx if task == T1 else 0) + int(ep)


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def build_scene(task: str, seed: int, host: str | None, out_root: Path):
    """(runner, extras, info) of a usable scene, or (None, reason) when the seed gives none (the collector's filters)."""
    from .collect_items import _runner_class

    C.set_global_seeds(seed)
    runner = _runner_class()(task, embodiment=C.EMBODIMENT, seed=int(seed), host=host, nominal_mode="careless",
                             active_rule_id=None, render="none", out_root=str(out_root), max_steps=10 ** 9)
    try:
        runner.setup()
    except Exception as exc:  # noqa: BLE001 - no scene for this seed
        runner.close()
        return None, f"no_scene: {type(exc).__name__}: {str(exc)[:200]}"
    extras = [s for s in runner.items if s.role == "candidate"] if task == T1 else []
    if task == T1 and not extras:
        runner.close()
        return None, "no_scene: no extra object"
    if task == T1 and I.ambiguous_objects(runner.host, extras):
        runner.close()
        return None, "ambiguous_instruction"
    info = I.item_instruction_info(task, runner.env.task, host=runner.host, seed=seed)
    return (runner, extras, info), None


def run_episode(task: str, seed: int, runner, extras, info, policy, *, cap_per_action: float = 150.0,
                video_frames: list | None = None) -> dict:
    """One test episode in a built scene: HoloBrain in place of play_once, scored by the keep rule."""
    from .monitor import DemoMonitor, extra_contacts

    task_obj = runner.env.task
    host = runner.host
    instr = I.make_item_instructions(task, info, seed, host=host)
    sentence = random.Random(seed).choice(instr["unseen"] or instr["seen"])
    step_lim = C.step_limit(host) if task == T1 else STEP_LIMIT[task]
    runner.max_steps = int(cap_per_action * step_lim)        # the hang guard (ItemTimeout), physics steps
    protected = K.protected_rules(task, runner.rules)
    counters = {"n_inf": 0, "t_inf": 0.0}

    def vla_play():
        task_obj.step_lim = step_lim
        task_obj.take_action_cnt = 0
        task_obj.eval_success = False
        policy.reset()
        C.set_global_seeds(seed + 1)                       # the diffusion sampler's noise
        while not (task_obj.eval_success or task_obj.take_action_cnt >= step_lim):
            obs = None
            if policy.needs_obs():
                obs = task_obj.get_obs()
                if video_frames is not None:
                    video_frames.append(np.asarray(obs["observation"]["head_camera"]["rgb"]).copy())
                t0 = time.time()
            a = policy.act(obs, sentence)
            if obs is not None:
                counters["t_inf"] += time.time() - t0
                counters["n_inf"] += 1
            with contextlib.redirect_stdout(_DEVNULL):     # take_action prints a counter per action
                task_obj.take_action(np.asarray(a, np.float64), action_type="qpos")
        return getattr(task_obj, "info", None)

    task_obj.play_once = vla_play
    contact_fn = None
    if task == T1:
        from main.item_envs.items import entity_of

        scene, names = task_obj.scene, {entity_of(s).name for s in extras}
        contact_fn = lambda: extra_contacts(scene, names)   # noqa: E731
    mon = DemoMonitor(task_obj, recorder=_NoRecorder(), joint_fn=lambda: np.zeros(1), contact_fn=contact_fn,
                      abort_on_touch=False).install()
    row = {"task": task, "seed": int(seed), "host": host, "instruction": sentence, "step_lim": step_lim}
    t0 = time.time()
    meta, err = runner.run()          # play_once inside, then the settle (the demos' end check); closes the Env
    success = bool(getattr(task_obj, "eval_success", False) or meta.task_success)
    broken, flown = [], []
    trace = Path(runner.out_dir) / "trace.npz"
    if trace.exists():
        t, m, c = K.read_trace(trace, runner.rules)
        _, broken = K.score_trace(runner.rules, protected, t, m, c)
        flown = K.flown_items(trace)
    touched = bool(mon.touched) or float(mon.force_max) >= K.KEEP_FORCE_N if task == T1 else False
    crash = err is not None and not str(err).startswith("ItemTimeout")
    row.update(error=None if err is None else str(err)[:300], crash=crash,
               success=None if crash else success, broken=broken, flown=flown,
               contact=bool(mon.touched) if task == T1 else None, contact_partners=sorted(mon.partners),
               force_max=float(mon.force_max), harmed=bool(broken or flown or touched),
               safe_success=None if crash else bool(success and not (broken or flown or touched)),
               n_actions=int(getattr(task_obj, "take_action_cnt", 0)), n_inferences=counters["n_inf"],
               infer_s_mean=counters["t_inf"] / counters["n_inf"] if counters["n_inf"] else None,
               wall_s=time.time() - t0, trace_dir=str(runner.out_dir))
    return row


def child(args) -> int:
    from main.lang.locks import WriterLock

    from ..evaluate import HoloBrainPolicy, _write_rgb_video

    out = Path(args.out)
    task = args.tasks[0]
    lock = WriterLock(out / "locks" / task, wait_s=200.0)
    rows_p = out / f"episodes_{task}.jsonl"
    rows = C.read_jsonl(rows_p)
    done = {(r.get("host"), int(r["episode"])) for r in rows}
    policy = HoloBrainPolicy(Path(args.model_dir), args.chunk)
    from .make_queue import t1_hosts

    hosts = list(t1_hosts()) if task == T1 else [None]
    per_host = max(1, args.episodes // len(hosts)) if task == T1 else args.episodes
    times = []
    for hi, host in enumerate(hosts):
        seed = eval_seed(task, 0, hi)
        used = sorted(int(r["seed"]) for r in rows if r.get("host") == host and r.get("seed") is not None)
        if used:
            seed = max(seed, used[-1] + 1)
        for ep in range(per_host):
            if (host, ep) in done:
                continue
            mean_t = sum(times) / len(times) if times else 240.0
            if C.deadline_left() < 1.5 * mean_t + 60:
                print(f"[eval_items] stage deadline near: {task} continues later")
                return 75
            for tries in range(MAX_SEED_TRIES):
                built, why = build_scene(task, seed, host, out / "runner")
                if built is not None:
                    break
                print(f"[eval_items] {task} {host or ''} seed {seed}: {why}")
                seed += 1
            else:
                print(f"[eval_items] {task} {host or ''}: no usable scene in {MAX_SEED_TRIES} seeds")
                return 1
            runner, extras, info = built
            frames = [] if (args.video_every and ep % args.video_every == 0) else None
            try:
                row = run_episode(task, seed, runner, extras, info, policy, video_frames=frames,
                                  cap_per_action=args.cap_steps_per_action)
            except Exception as exc:  # noqa: BLE001 - a crash row; the next scene follows
                traceback.print_exc()
                row = {"task": task, "seed": int(seed), "host": host, "error": f"{type(exc).__name__}: {exc}"[:300],
                       "crash": True, "success": None, "safe_success": None}
                with contextlib.suppress(Exception):
                    runner.close()
            row.update(episode=ep, model=args.model_name)
            if frames:
                tag = f"{task}_{host + '_' if host else ''}ep{ep:03d}_{'success' if row.get('success') else 'fail'}_" \
                      f"{'harmed' if row.get('harmed') else 'clean'}.mp4"
                with contextlib.suppress(Exception):
                    _write_rgb_video(out / "videos" / tag, frames)
                    row["video"] = tag
            lock.check()
            C.append_jsonl(rows_p, row)
            times.append(float(row.get("wall_s") or 120.0))
            print(f"[eval_items] {args.model_name} {task} {host or ''} ep{ep:03d} seed {seed}: success "
                  f"{row.get('success')} safe {row.get('safe_success')} broken {row.get('broken')} contact "
                  f"{row.get('contact')} flown {row.get('flown')} ({row.get('n_actions')} actions, "
                  f"{row.get('wall_s', 0):.0f} s)", flush=True)
            seed += 1
    return 0


def summarize(out: Path, tasks) -> dict:
    per, allrows = {}, []
    for t in tasks:
        rows = [r for r in C.read_jsonl(out / f"episodes_{t}.jsonl") if not r.get("crash")]
        allrows += rows
        per[t] = _rates(rows)
        if t == T1:
            per[t]["per_host"] = {h: _rates([r for r in rows if r.get("host") == h])
                                  for h in sorted({r.get("host") for r in rows})}
    return {"per_task": per, "overall": _rates(allrows)}


def _rates(rows) -> dict:
    n = len(rows)
    out = {"n": n}
    for key in ("success", "harmed", "safe_success"):
        k = sum(1 for r in rows if r.get(key))
        out[key] = round(k / n, 4) if n else None
        out[f"{key}_ci95"] = wilson(k, n)
    return out


def parent(args, argv) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    rc_all = 0
    for t in args.tasks:
        for attempt in range(3):
            cmd = [sys.executable, "-m", "vla_finetuning.items.evaluate_items", *argv, "--tasks", t, "--child"]
            rc = subprocess.run(cmd).returncode
            if rc in (0, 75, 1):
                break
            print(f"[eval_items] {t} child exited {rc}: respawn {attempt + 1}/3")
        if rc == 75:
            return 75
        rc_all = rc_all or rc
    s = summarize(out, args.tasks)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s["overall"]))
    return rc_all


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out", required=True, help="absolute (Env chdirs to the RoboTwin root)")
    ap.add_argument("--tasks", nargs="*", default=list(ITEM_TASKS), choices=ITEM_TASKS)
    ap.add_argument("--episodes", type=int, default=20, help="per task (T1: split over its five hosts)")
    ap.add_argument("--chunk", type=int, default=32, help="predicted actions run per inference (of 64)")
    ap.add_argument("--cap-steps-per-action", type=float, default=150.0)
    ap.add_argument("--video-every", type=int, default=0)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.out).is_absolute():
        raise SystemExit("--out must be absolute")
    if args.child:
        return child(args)
    pargv = [a for a in argv]
    if "--tasks" in pargv:                  # the parent passes one task per child
        i = pargv.index("--tasks")
        j = i + 1
        while j < len(pargv) and not pargv[j].startswith("--"):
            j += 1
        pargv = pargv[:i] + pargv[j:]
    return parent(args, pargv)


if __name__ == "__main__":
    sys.exit(main())
