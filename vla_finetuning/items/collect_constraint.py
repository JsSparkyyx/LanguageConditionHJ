"""Collect the constraint-following demonstrations of one item task (Yuxuan 2026-11-03; constraint.py has the design):
every demo does what its WHOLE prompt says.

    scripts/lang/in_container.sh python -m vla_finetuning.items.collect_constraint --task gap_place \
        --root /abs/vla_finetuning/data/demos_constraint --shard s0 --seed-start 540000 --seed-step 4 \
        --seed-stop 550000 --target 25

Per scene (one seed):

    A  the careless run (nominal_mode "careless", the experiment's careless knobs: constraint.careless_risk /
       after_setup / patch_task_module). Kept if it finishes the task (harm allowed; no error, no failed plan, nothing
       flown, enough frames). Its prompts: half the task alone, half the task + a constraint on an item NOT in the
       way that this run left unharmed (all the task alone when there is none). Scenes with nothing in the way are
       skipped before the run.
    B  one careful run per item in the way, only if A was kept (nominal_mode "safe" with active_rule_id = that item's
       rule: T1 plans around that object only, T3 pushes around that mat / cylinder only, T5 / T6 / T7 run their
       safe knobs). Kept if it finishes and none of that item's rules broke (T1: the object never touched, force
       < 20 N; aborted at the first touch); other items may be harmed. Its prompts: the task + a constraint on that
       item (every rule of it the run kept). Up to B_TRIES runs per item (cuRobo differs run to run).

Prompts: the task sentences of the 10-27 demos (instructions.make_item_instructions: 100 seen + 100 unseen, RoboTwin
format), joined as "<task>, and <constraint>." with the paraphrase bank's TRAINING templates and words
(language.sentences "train"); the held-out template of each rule never appears (evaluate_constraint tests it).

Files (HoloBrain's packer names an episode <task>_<config>_seed<seed>, so the demos of one scene go to different
config directories): <root>/<task>/<shard>_A/ and <shard>_B<j>/ (j = the item's index among the scene's in-the-way
items), each with seed.txt, data/episode{k}.hdf5, instructions/episode{k}.json and attempts.jsonl (its kept rows, for
pack.py's count check). Every run's row: <root>_logs/<task>/<shard>/runs.jsonl ("started" lines first), with
settings.json, runner/<n>/ and videos/ (the first --videos kept demos of each kind, head camera).

Exit 0: --target complete scenes (A and at least one B kept) or seeds exhausted; 75: continue in a fresh process
(--max-runs, the stage deadline, a renderer error); 3: the planners or the code are broken.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from .. import common as C
from ..demo_io import write_episode, write_instructions, write_seed_txt, write_video
from . import constraint as X
from . import instructions as I
from . import keep as K
from .collect_items import MAX_STEPS, _plain, _runner_class, diag_overrides, expert_diag, pin_settings, \
    planner_problem, scene_facts
from .monitor import FRAME_EVERY, SETTLE_STEPS, DemoMonitor

TASKS = I.ITEM_TASKS
T1 = I.T1
B_TRIES = 2
RETRY_REASONS = ("render_error", "planner_error", "code_error")
RETRY_MAX = 3
CODE_STREAK = 3
_hard_exit = False


# ------------------------------------------------------------------ one run ----
def run_demo(task: str, seed: int, host: str | None, kind: str, out_root: Path, max_steps: int, knobs: dict,
             ref: str | None = None, active: str | None = None, frames: bool = True):
    """One careless (kind "A") or careful (kind "B", for in-the-way referent ``ref``) run. A failed episode never
    raises. -> (row, DemoMonitor | None, scene: {"roles", "task_instr", "info"} | None); row["reason"] = "kept" or
    why not, row["constraints"] = the constraint sentences its prompts may use."""
    mode = "careless" if kind == "A" else "safe"
    active = active if kind == "B" else None
    row = {"task": task, "seed": int(seed), "host": host, "kind": kind, "ref": ref, "active_rule": active}
    runner = mon = None
    try:
        try:
            row["patched"] = X.patch_task_module(task, knobs)
            C.set_global_seeds(seed)
            risk = X.careless_risk(task, seed, knobs) if kind == "A" else None
            runner = _runner_class()(task, embodiment=C.EMBODIMENT, seed=int(seed), host=host, nominal_mode=mode,
                                     active_rule_id=active, render="none", out_root=str(out_root),
                                     max_steps=int(max_steps), risk=risk)
            row["host"] = runner.host
        except Exception as exc:  # noqa: BLE001 - before any scene: never the seed's
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=err[:500], detail="patch / seeding / runner constructor")
            return row, None, None
        try:
            runner.setup()
        except Exception as exc:  # noqa: BLE001 - no scene for this seed
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            kind_err = K.error_reason(err)
            problem = planner_problem(runner, task) if mode == "safe" else None
            reason = ("render_error" if kind_err == "render_error" else "planner_error" if problem
                      else "code_error" if kind_err == "code_error" else "no_scene")
            row.update(reason=reason, error=err[:500], detail=problem)
            return row, None, None
        if mode == "safe":
            problem = planner_problem(runner, task)
        else:
            from main.item_envs import safe_plan

            try:
                safe_plan.require_curobo(runner.env.robot)
                problem = None
            except Exception as exc:  # noqa: BLE001
                problem = f"{type(exc).__name__}: {exc}"
        if problem:
            row.update(reason="planner_error", detail=problem)
            return row, None, None
        try:
            task_obj = runner.env.task
            row["scene"] = scene_facts(task, runner)
            extras = [s for s in runner.items if s.role == "candidate"] if task == T1 else []
            if task == T1 and not extras:
                row.update(reason="no_scene", detail="no extra object")
                return row, None, None
            amb = I.ambiguous_objects(runner.host, extras) if task == T1 else []
            if amb:
                row.update(reason="ambiguous_instruction", detail=amb)
                return row, None, None
            roles = X.scene_rules(task, runner)
            row["in_way"] = sorted(roles["harm"])
            row["benign"] = sorted(roles["benign"])
            row["reasons"] = roles["reasons"]
            row["protect"] = {r: X.protect_rule(roles["harm"][r]) for r in sorted(roles["harm"])}
            if kind == "A" and not roles["harm"]:
                row.update(reason="nothing_in_the_way")
                return row, None, None
            if kind == "B" and ref not in roles["harm"]:
                row.update(reason="scene_mismatch", detail=f"{ref} not in the way in the rebuilt scene")
                return row, None, None
            info = I.item_instruction_info(task, task_obj, host=runner.host, seed=seed)
            task_instr = I.make_item_instructions(task, info, seed, host=runner.host)
            row["info"] = {k: str(v) for k, v in info.items()}
            row["after_setup"] = X.after_setup(task, runner, mode, seed, knobs)
            probe = X.SuccessProbe(task, task_obj)
            contacts = X.t1_contact_log(runner, watch=None if kind == "A" else {ref}) if task == T1 else None
            recorder = None
            if not frames:
                from .evaluate_items import _NoRecorder

                recorder = _NoRecorder()
            mon = DemoMonitor(task_obj, recorder=recorder, contact_fn=contacts, abort_on_touch=(kind == "B")).install()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=err[:500])
            return row, mon, None
        try:
            meta, err = runner.run()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=f"run: {err}"[:500])
            return row, mon, None
        diag = dict(getattr(meta, "diag", None) or {})
        task_diag = diag.get("task") if isinstance(diag.get("task"), dict) else (getattr(task_obj, "episode_info", None)
                                                                                 or {})
        trace = Path(runner.out_dir) / "trace.npz"
        status, flown = {}, []
        try:
            if trace.exists():
                status = X.rule_status(task, runner, trace, contacts)
                flown = K.flown_items(trace)
            elif err is None:
                err = f"no trace.npz in {runner.out_dir}"
        except Exception as exc:  # noqa: BLE001 - a bug here is a code_error
            traceback.print_exc()
            err = err or f"{type(exc).__name__}: trace: {exc}"
        plan_ok = bool(getattr(task_obj, "plan_success", True))
        broken = sorted(r for r, v in status.items() if v["broken"])
        row.update(error=None if err is None else str(err)[:500], safe_fail=task_diag.get("safe_fail"),
                   task_success=bool(meta.task_success), probe=probe.last, plan_success=plan_ok,
                   plan_failures=int(meta.plan_failures), broken=broken, flown=flown,
                   risk=_plain(getattr(meta, "risk", None)), hit=task_diag.get("hit"),
                   contacts=contacts.summary() if contacts is not None else None,
                   n_skipped_frames=mon.n_skipped, play_steps=mon.play_steps, settle_steps=mon.settled,
                   n_steps=mon.n, duration_s=mon.n / C.PHYS_HZ, diag_task=_plain(task_diag),
                   runner_dir=str(runner.out_dir))
        if task == "clean_pour" and trace.exists():
            row["bead_counts"] = X.final_counts(task, runner, trace)
        row.update(expert_diag(task_diag))
        # the keep rule: the task finished, cleanly; B also spares its item
        reason = None
        if err:
            reason = K.error_reason(err)
        elif kind == "B" and task_diag.get("safe_fail") == "planner_error":
            reason = "planner_error"
        elif kind == "B" and task_diag.get("safe_fail"):
            reason = f"safe_fail:{task_diag['safe_fail']}"
        elif not plan_ok:
            reason = "plan_failed"
        elif not meta.task_success:
            reason = "task_failed"
        elif flown:
            reason = f"flown:{flown[0]}"
        elif kind == "B":
            mine = [r["rule_id"] for r in roles["harm"][ref]]
            bad = [r for r in mine if status.get(r, {}).get("broken")]
            if task == T1 and contacts is not None and contacts.harmed(ref, K.KEEP_FORCE_N):
                reason = "contact" if contacts.touched.get(ref) else "force"
            elif bad:
                reason = f"rule:{bad[0]}"
        if reason is None and len(mon.frames) < K.MIN_FRAMES and frames:
            reason = "few_frames"
        row["reason"] = reason or "kept"
        # the constraints its prompts may use: A the benign items' rules it left unbroken, B its item's
        if kind == "A":
            pool = [r for rules in roles["benign"].values() for r in rules if not status.get(r["rule_id"], {}).get(
                "broken", True)]
        else:
            pool = [r for r in roles["harm"][ref] if not status.get(r["rule_id"], {}).get("broken", True)]
        row["constraint_rules"] = [r["rule_id"] for r in pool]
        row["constraint_groups"] = [list(r["train"]) for r in pool]      # one sentence list per rule
        row["constraints"] = sorted({s for r in pool for s in r["train"]})
        return row, mon, {"roles": roles, "task_instr": task_instr, "info": info}
    finally:
        if runner is not None:
            runner.close()


# ------------------------------------------------------------------ bookkeeping ----
class Config:
    """One packer config directory (<root>/<task>/<shard>_<kind>): its kept episodes, 0..k-1."""

    def __init__(self, d: Path):
        self.d = Path(d)
        self.kept = sorted(C.read_jsonl(self.d / "attempts.jsonl"), key=lambda r: r["episode"])
        if [r["episode"] for r in self.kept] != list(range(len(self.kept))):
            raise SystemExit(f"{self.d}: kept episode indices are not 0..{len(self.kept) - 1}")
        for sub, pat in (("data", "episode*.hdf5"), ("instructions", "episode*.json")):
            for p in sorted((self.d / sub).glob(pat)) + sorted((self.d / sub).glob("*.tmp")):
                idx = p.name.replace("episode", "").split(".")[0]
                if p.suffix == ".tmp" or not idx.isdigit() or int(idx) >= len(self.kept):
                    print(f"[collect_constraint] removing orphan {p}")
                    p.unlink()
        if self.kept:
            write_seed_txt(self.d / "seed.txt", [r["seed"] for r in self.kept])

    def add(self, row: dict, frames: list, instr: dict) -> int:
        k = len(self.kept)
        write_episode(self.d / "data" / f"episode{k}.hdf5", frames)
        write_instructions(self.d / "instructions" / f"episode{k}.json", instr)
        write_seed_txt(self.d / "seed.txt", [r["seed"] for r in self.kept] + [row["seed"]])
        kept_row = {"episode": k, "seed": row["seed"], "kept": True, "n_frames": len(frames), "kind": row["kind"],
                    "ref": row.get("ref"), "host": row.get("host"), "constraint_rules": row.get("constraint_rules"),
                    "instruction_example": instr["seen"][0] if instr.get("seen") else None}
        C.append_jsonl(self.d / "attempts.jsonl", kept_row)
        self.kept.append(kept_row)
        return k


def settings_of(args) -> dict:
    """What a shard's directory pins: a rerun with other settings is refused. Only this task's own knobs (review 11-03:
    a knob of another task must not stop this shard)."""
    return {"task": args.task, "host": args.host, "embodiment": C.EMBODIMENT, "design": "constraint_2026-11-03",
            "knobs": X.task_knobs(args.task), "instruction_md5": I.instruction_md5(args.task, args.host),
            "b_tries": B_TRIES, "max_steps": int(args.max_steps), "frame_every": FRAME_EVERY,
            "settle_steps": SETTLE_STEPS, "keep_force_n": K.KEEP_FORCE_N, "outside_h_min": K.OUTSIDE_H_MIN,
            "min_frames": K.MIN_FRAMES, "seed_start": int(args.seed_start), "seed_step": int(args.seed_step),
            "seed_stop": int(args.seed_stop)}


def scene_state(rows: list[dict]) -> dict:
    """The shard's progress from runs.jsonl.

    tried: seeds whose A run ended for a reason of the seed (kept or not), or started RETRY_MAX times (process or
    code errors, or a process killed mid-run, are tried again until then). complete: A kept and a B kept. pending:
    seed -> [(ref, active rule, j)] of the B runs a kept A still owes: an in-the-way item with no kept B, fewer than
    B_TRIES B runs that ended for a reason of the scene, and fewer than B_TRIES + RETRY_MAX starts."""
    a_rows, a_started = {}, Counter()
    b_rows, b_started = {}, Counter()
    for r in rows:
        s = int(r["seed"])
        if r.get("started") == "A":
            a_started[s] += 1
        elif r.get("started") == "B":
            b_started[(s, r.get("ref"))] += 1
        elif r.get("kind") == "A" and "reason" in r:
            a_rows.setdefault(s, []).append(r)
        elif r.get("kind") == "B" and "reason" in r:
            b_rows.setdefault((s, r.get("ref")), []).append(r)
    tried = {s for s, rs in a_rows.items() if any(r["reason"] not in RETRY_REASONS for r in rs)}
    tried |= {s for s, n in a_started.items() if n >= RETRY_MAX}
    kept_a = {s: next(r for r in rs if r["reason"] == "kept") for s, rs in a_rows.items()
              if any(r["reason"] == "kept" for r in rs)}
    complete, pending = set(), {}
    for s, a in kept_a.items():
        refs = list(a.get("in_way") or [])
        if any(r["reason"] == "kept" for ref in refs for r in b_rows.get((s, ref), [])):
            complete.add(s)
        for j, ref in enumerate(refs):
            rs = b_rows.get((s, ref), [])
            if any(r["reason"] == "kept" for r in rs):
                continue
            tries = sum(1 for r in rs if r["reason"] not in RETRY_REASONS)
            if tries < B_TRIES and b_started[(s, ref)] < B_TRIES + RETRY_MAX:
                active = (a.get("protect") or {}).get(ref)
                if active:
                    pending.setdefault(s, []).append((ref, active, j))
    return {"tried": tried, "complete": complete, "pending": pending, "kept_a": kept_a,
            "n_runs": sum(1 for r in rows if "reason" in r)}


# ------------------------------------------------------------------ the loop ----
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--host", default=None, help="selective_avoid only, required: the host task of this shard")
    ap.add_argument("--root", required=True, help="absolute demo root (packer input_path)")
    ap.add_argument("--shard", required=True, help="shard name (config dirs <shard>_A, <shard>_B<j>)")
    ap.add_argument("--target", type=int, default=25, help="complete scenes (A kept and at least one B kept)")
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-step", type=int, default=1)
    ap.add_argument("--seed-stop", type=int, required=True)
    ap.add_argument("--max-runs", type=int, default=30, help="runs (Env builds) per process, then exit 75")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--videos", type=int, default=3, help="head-camera videos of the first N kept demos per kind")
    args = ap.parse_args(argv)
    if args.task == T1:
        import main.lang  # noqa: F401
        from main.item_envs.registry import get_task_entry

        if args.host not in get_task_entry(T1).host_tasks:
            ap.error(f"--task {T1} needs --host, one of {list(get_task_entry(T1).host_tasks)}")
    elif args.host:
        ap.error(f"--host is for --task {T1} only")
    args.max_steps = int(args.max_steps or MAX_STEPS[args.task])
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
    st = scene_state(rows)
    configs: dict[str, Config] = {}

    def config(name: str) -> Config:
        if name not in configs:
            configs[name] = Config(root / args.task / f"{args.shard}_{name}")
        return configs[name]

    config("A")
    for d in sorted((root / args.task).glob(f"{args.shard}_B*")):   # every config's orphans are cleaned at start
        tail = d.name[len(args.shard) + 1:]
        if d.is_dir() and tail[1:].isdigit():
            config(tail)
    n_vid = Counter(r.get("kind") for r in rows if r.get("video"))
    name = f"{args.task}/{args.shard}"
    complete = set(st["complete"])
    print(f"[collect_constraint] {name}: {len(complete)}/{args.target} complete scenes, {st['n_runs']} runs, "
          f"{sum(len(v) for v in st['pending'].values())} careful runs owed", flush=True)
    n_run, code_streak, times = 0, 0, []
    n_started = sum(1 for r in rows if "started" in r)
    seeds = iter(range(args.seed_start, args.seed_stop, args.seed_step))
    # the scenes still owed careful runs first (a process that stopped between a scene's A and its Bs), then new ones
    queue = [(s, [("B", ref, act, j) for ref, act, j in jobs]) for s, jobs in sorted(st["pending"].items())]

    def stop_now(need: int = 1) -> str | None:
        if n_run + need > args.max_runs:
            return f"{n_run} runs in this process: continue in a fresh one"
        mean_t = sum(times) / len(times) if times else 240.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            return f"stage deadline near (mean run {mean_t:.0f} s): continue later"
        return None

    while len(complete) < args.target or queue:
        if queue:
            seed, jobs = queue.pop(0)
        else:
            why = stop_now()
            if why:                                    # before the scene line: no seed is skipped by a stop
                print(f"[collect_constraint] {why}")
                return 75
            seed = next((s for s in seeds if s not in st["tried"]), None)
            if seed is None:
                print(f"[collect_constraint] seeds exhausted: {len(complete)}/{args.target}")
                return 0
            lock.check()
            C.append_jsonl(runs_p, {"seed": seed, "started": "scene", "at": time.strftime("%F %T")})
            jobs = [("A", None, None, None)]
        while jobs:
            kind, ref, active, j = jobs.pop(0)
            why = stop_now()
            if why:                                    # owed B runs resume in the next process (scene_state pending)
                print(f"[collect_constraint] {why}")
                return 75
            n_run += 1
            idx, n_started = n_started, n_started + 1
            att = log_d / "runner" / str(idx)
            if att.exists():
                shutil.rmtree(att)
            C.append_jsonl(runs_p, {"seed": seed, "started": kind, "ref": ref, "run": idx})
            t0 = time.time()
            try:
                row, mon, sc = run_demo(args.task, seed, args.host, kind, att, args.max_steps, X.KNOBS, ref, active)
            except Exception as exc:  # noqa: BLE001 - this collector's own code
                traceback.print_exc()
                err = f"{type(exc).__name__}: {exc}"
                row, mon, sc = ({"task": args.task, "seed": seed, "host": args.host, "kind": kind, "ref": ref,
                                 "reason": K.error_reason(err), "error": err[:500], "detail": "run_demo"}, None, None)
            reason = row.get("reason") or "error"
            frames = mon.frames if mon is not None else []
            row.update(run=idx, shard=args.shard, n_frames=len(frames), wall_s=time.time() - t0)
            if kind == "B":
                row["b_index"] = j
            lock.check()
            if reason == "kept":
                if kind == "A":
                    groups = row["constraint_groups"]
                    instr = X.demo_instructions(sc["task_instr"], groups, 0.5 if groups else 0.0, seed)
                    cfg = config("A")
                    jobs = [("B", r, row["protect"][r], jj) for jj, r in enumerate(row["in_way"])]
                else:
                    instr = X.demo_instructions(sc["task_instr"], row["constraint_groups"], 1.0, seed)
                    cfg = config(f"B{j}")
                    jobs = [jb for jb in jobs if jb[1] != ref]           # its other tries are not needed
                    complete.add(seed)
                k = cfg.add(row, frames, instr)
                row.update(episode=k, config=cfg.d.name, instruction_example=instr["seen"][0])
                if n_vid[kind] < args.videos:
                    vname = f"{kind}{'' if kind == 'A' else '_' + str(ref)}_seed{seed}.mp4"
                    try:
                        write_video(log_d / "videos" / vname, frames)
                        row["video"] = vname
                        n_vid[kind] += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"[collect_constraint] video failed: {exc}")
            elif kind == "B" and reason not in RETRY_REASONS:
                tries = sum(1 for r in C.read_jsonl(runs_p) if r.get("seed") == seed and r.get("kind") == "B"
                            and r.get("ref") == ref and "reason" in r and r["reason"] not in RETRY_REASONS) + 1
                if tries < B_TRIES:
                    jobs.append(("B", ref, active, j))
            C.append_jsonl(runs_p, _plain(row))
            times.append(row["wall_s"])
            print(f"[collect_constraint] {name} seed {seed} {kind}{'' if ref is None else ' ' + ref}: {reason} "
                  f"(success {row.get('task_success')}, broken {row.get('broken')}, probe {row.get('probe')}, "
                  f"in way {row.get('in_way')}, {row['n_frames']} frames, {row['wall_s']:.0f} s) -> "
                  f"{len(complete)}/{args.target} complete", flush=True)
            if reason == "render_error":
                _hard_exit = True
                return 75
            if reason == "planner_error":
                print(f"[collect_constraint] refusing to run: {row.get('detail')}", flush=True)
                return 3
            code_streak = code_streak + 1 if reason == "code_error" else 0
            if code_streak >= CODE_STREAK:
                print(f"[collect_constraint] {code_streak} code errors in a row (last: {row.get('error')})", flush=True)
                return 3
            if kind == "A" and reason != "kept":
                jobs = []
            del frames, mon
    print(f"[collect_constraint] {name}: target reached ({len(complete)} complete scenes)")
    return 0


if __name__ == "__main__":
    rc = main()
    if _hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)
