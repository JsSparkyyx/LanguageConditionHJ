"""Closed-loop test of the item scenes' two scripted controllers on the scenes the fine-tuned models were tested on
(Yuxuan 11-01: compare the model with the nominal and with the safe nominal that collected its training data):

- careless: the task's nominal controller with the scene's sampled risk knobs (nominal_mode "careless"), the
  controller the language filter guards;
- safe: the safe nominal, nominal_mode "safe" (the task yaml's safe_mode knobs, T1's obstacle-aware planner), the
  expert of the 500 safe demos (collect_items.run_attempt).

    scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_nominal --mode careless|safe \
        --out /abs/vla_finetuning/results_items/nominal_<mode> [--ref /abs/vla_finetuning/results_items/base]

The scenes are the reference model's rows (task, host, episode, seed), so every controller and model meets the same
layouts (runner.apply_nominal_mode: a seed builds the same scene in both modes). Judged like the models
(evaluate_items.run_episode): success = the task's eval_success or the runner's end check, broken = protected rules
with h < 0 at some tick, contact = T1's touch of an extra object (or force >= KEEP_FORCE_N), flown = an item above
MAX_ITEM_Z, safe = success and none of these. The rows also keep the collector's keep reason (keep.keep_reason) and
the planner diagnostics. A scene whose planners did not load (cuRobo init failed, T1's safe model missing) is retried
twice and then recorded as a crash row (left out of the rates, as in evaluate_items). One child process per task;
--tasks runs a subset (several children may run at once: one writer lock per task). No videos. Exit 75 = stopped at
the stage deadline (STAGE_DEADLINE).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .. import common as C
from . import instructions as I
from . import keep as K
from .evaluate_items import _NoRecorder, summarize
from .instructions import ITEM_TASKS, T1

PLANNER_TRIES = 3


def run_one(task: str, seed: int, host: str | None, mode: str, out_root: Path) -> dict:
    """One episode of the scripted controller in <mode> on the scene of <seed> (and T1's <host>), scored."""
    from .collect_items import MAX_STEPS, _runner_class, planner_problem
    from .monitor import DemoMonitor, extra_contacts
    from main.item_envs import safe_plan

    row = {"task": task, "seed": int(seed), "host": host, "mode": mode}
    C.set_global_seeds(seed)
    runner = _runner_class()(task, embodiment=C.EMBODIMENT, seed=int(seed), host=host, nominal_mode=mode,
                             active_rule_id=None, render="none", out_root=str(out_root), max_steps=MAX_STEPS[task])
    try:
        runner.setup()
    except Exception as exc:  # noqa: BLE001 - the reference model had a scene here: a crash row
        traceback.print_exc()
        with contextlib.suppress(Exception):
            runner.close()
        return {**row, "crash": True, "error": f"no_scene: {type(exc).__name__}: {str(exc)[:200]}",
                "success": None, "safe_success": None}
    if mode == "safe":
        problem = planner_problem(runner, task)
    else:   # the careless nominal plans with cuRobo too (no T1 hand model)
        try:
            safe_plan.require_curobo(runner.env.robot)
            problem = None
        except Exception as exc:  # noqa: BLE001
            problem = f"{type(exc).__name__}: {exc}"
    if problem:
        with contextlib.suppress(Exception):
            runner.close()
        return {**row, "crash": True, "planner_error": True, "error": f"planner_error: {problem}"[:300],
                "success": None, "safe_success": None}
    task_obj = runner.env.task
    extras = [s for s in runner.items if s.role == "candidate"] if task == T1 else []
    protected = K.protected_rules(task, runner.rules)
    contact_fn = None
    if task == T1:
        from main.item_envs.items import entity_of

        scene, names = task_obj.scene, {entity_of(s).name for s in extras}
        contact_fn = lambda: extra_contacts(scene, names)   # noqa: E731
    mon = DemoMonitor(task_obj, recorder=_NoRecorder(), joint_fn=lambda: np.zeros(1), contact_fn=contact_fn,
                      abort_on_touch=False).install()
    t0 = time.time()
    meta, err = runner.run()          # the controller's own play_once, then the settle; closes the Env
    success = bool(getattr(task_obj, "eval_success", False) or meta.task_success)
    broken, flown = [], []
    trace = Path(runner.out_dir) / "trace.npz"
    if trace.exists():
        t, m, c = K.read_trace(trace, runner.rules)
        _, broken = K.score_trace(runner.rules, protected, t, m, c)
        flown = K.flown_items(trace)
    touched = bool(mon.touched) or float(mon.force_max) >= K.KEEP_FORCE_N if task == T1 else False
    crash = err is not None and not str(err).startswith("ItemTimeout")
    diag = dict(getattr(meta, "diag", None) or {})
    task_diag = diag.get("task") if isinstance(diag.get("task"), dict) else (getattr(task_obj, "episode_info", None) or {})
    plan_ok = bool(getattr(task_obj, "plan_success", True))
    facts = {"err": err, "safe_fail": task_diag.get("safe_fail"), "plan_success": plan_ok,
             "task_success": bool(meta.task_success), "flown": flown, "broken": broken,
             "contact": {"touched": mon.touched, "force_max": mon.force_max} if task == T1 else None,
             "n_frames": 10 ** 6}
    with contextlib.suppress(Exception):
        row["keep_reason"] = K.keep_reason(facts)
    row.update(error=None if err is None else str(err)[:300], crash=crash,
               success=None if crash else success, broken=broken, flown=flown,
               contact=bool(mon.touched) if task == T1 else None, contact_partners=sorted(mon.partners),
               force_max=float(mon.force_max), harmed=bool(broken or flown or touched),
               safe_success=None if crash else bool(success and not (broken or flown or touched)),
               plan_success=plan_ok, plan_failures=int(getattr(meta, "plan_failures", 0) or 0),
               safe_fail=task_diag.get("safe_fail"), play_steps=mon.play_steps,
               wall_s=time.time() - t0, trace_dir=str(runner.out_dir))
    return row


def child(args) -> int:
    from main.lang.locks import WriterLock

    out = Path(args.out)
    task = args.tasks[0]
    lock = WriterLock(out / "locks" / task, wait_s=200.0)
    ref = C.read_jsonl(Path(args.ref) / f"episodes_{task}.jsonl")
    rows_p = out / f"episodes_{task}.jsonl"
    done = {(r.get("host"), int(r["episode"])) for r in C.read_jsonl(rows_p)}
    seen, times = set(), []
    for r in ref:
        key = (r.get("host"), int(r["episode"]))
        if key in seen or key in done:
            continue
        seen.add(key)
        mean_t = sum(times) / len(times) if times else 240.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            print(f"[eval_nominal] stage deadline near: {task} continues later")
            return 75
        for attempt in range(PLANNER_TRIES):
            try:
                row = run_one(task, int(r["seed"]), r.get("host"), args.mode, out / "runner")
            except Exception as exc:  # noqa: BLE001 - a crash row; the next scene follows
                traceback.print_exc()
                row = {"task": task, "seed": int(r["seed"]), "host": r.get("host"), "mode": args.mode,
                       "error": f"{type(exc).__name__}: {exc}"[:300], "crash": True, "success": None,
                       "safe_success": None}
            if not row.get("planner_error"):
                break
            print(f"[eval_nominal] {task} seed {r['seed']}: {row['error']} (try {attempt + 1}/{PLANNER_TRIES})")
        row.update(episode=key[1], model=f"nominal_{args.mode}", ref_model=Path(args.ref).name)
        lock.check()
        C.append_jsonl(rows_p, row)
        times.append(float(row.get("wall_s") or 120.0))
        print(f"[eval_nominal] {args.mode} {task} {r.get('host') or ''} ep{key[1]:03d} seed {r['seed']}: success "
              f"{row.get('success')} safe {row.get('safe_success')} broken {row.get('broken')} contact "
              f"{row.get('contact')} flown {row.get('flown')} keep {row.get('keep_reason')} "
              f"({row.get('wall_s', 0):.0f} s)", flush=True)
    return 0


def parent(args, argv) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    rc_all = 0
    for t in args.tasks:
        for attempt in range(3):
            cmd = [sys.executable, "-m", "vla_finetuning.items.evaluate_nominal", *argv, "--tasks", t, "--child"]
            rc = subprocess.run(cmd).returncode
            if rc in (0, 75, 1):
                break
            print(f"[eval_nominal] {t} child exited {rc}: respawn {attempt + 1}/3")
        if rc == 75:
            return 75
        rc_all = rc_all or rc
    if set(args.tasks) == set(ITEM_TASKS):     # a partial run never writes the summary
        s = summarize(out, args.tasks)
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s["overall"]))
    return rc_all


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["careless", "safe"])
    ap.add_argument("--out", required=True, help="absolute (Env chdirs to the RoboTwin root)")
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[1] / "results_items" / "base"),
                    help="the model test whose scenes (seeds) are replayed")
    ap.add_argument("--tasks", nargs="*", default=list(ITEM_TASKS), choices=ITEM_TASKS)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.out).is_absolute() or not Path(args.ref).is_absolute():
        raise SystemExit("--out and --ref must be absolute")
    if args.child:
        return child(args)
    pargv = list(argv)
    if "--tasks" in pargv:                  # the parent passes one task per child
        i = pargv.index("--tasks")
        j = i + 1
        while j < len(pargv) and not pargv[j].startswith("--"):
            j += 1
        pargv = pargv[:i] + pargv[j:]
    return parent(args, pargv)


if __name__ == "__main__":
    sys.exit(main())
