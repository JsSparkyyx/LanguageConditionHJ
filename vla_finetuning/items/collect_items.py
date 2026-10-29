"""Collect safe demonstrations of one item task on ur5-wsg for HoloBrain fine-tuning, in the collision-avoidance
set's files and setting (Yuxuan 2026-10-27, docs/lang_filter/safe_controllers.md: every item protected, the plain
task instruction, 100 kept per task, T1's 100 = 20 per host):

    scripts/lang/in_container.sh python -m vla_finetuning.items.collect_items --task gap_place \
        --root /work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/data/demos_items_safe \
        --config-name s0 --seed-start 340000 --seed-step 4 --seed-stop 350000 --target 25

Tasks: selective_avoid (T1, --host pins one of its 5 hosts: one config per host), push_zone (T3), tray_transport
(T5), clean_pour (T6, the three-bowl bowl_pour), gap_place (T7). The expert is the task's safe nominal mode,
ItemEpisodeRunner(..., nominal_mode="safe", active_rule_id=None), which protects every item of the scene. Per attempt:

    a. a "started" line; the runner for the seed (out_root = <config>/runner/<attempt>), setup(); a setup error or a
       T1 layout with no extra object -> no_scene;
    b. T1 stack_blocks_two: a red or green box among the extra objects -> ambiguous_instruction (the host's sentences
       say "red block" / "green block");
    c. the instruction placeholders and the 100 + 100 instructions (instructions.py; written only if kept), the
       protected rules (keep.py), T1's extra objects for the contact check;
    d. the hooks (monitor.py): frames every 15 physics steps of play_once (pure waits skipped), T1's contact check at
       every physics step (an extra object touching anything but the table, another extra object included; the
       first touch aborts the attempt), a 1 s settle after play_once;
    e. run(); every rule's h per capture tick from the runner's trace.npz;
    f. the keep rule (keep.keep_reason): error / timeout / contact (T1's abort), safe_fail:<why>, plan_failed (the
       task's plan_success False at the end, as collect_demos.py; meta.plan_failures also counts the failed plans the
       task recovered from and is a row diagnostic only), task_failed, rule:<rule_id>, contact / force (T1),
       few_frames, else kept;
    g. kept: data/episode{k}.hdf5 + instructions/episode{k}.json + entry k of seed.txt (demo_io, the layout
       HoloBrain's packer reads; vla_finetuning/pack.py packs the root unchanged), k = the kept count.

Bookkeeping as in collect_demos.py (its seed walk is reused): settings.json pins the directory's settings (a rerun
with others is refused), attempts.jsonl gets a "started" line before the scene is built and one result row per
attempt with the per-rule min h and first violation, contacts, frames, wall time, scene facts and the safe experts'
diagnostics (expert_diag), a resume deletes orphan files and rewrites seed.txt. Every exception of an attempt ends in
its row (keep.error_reason; the seeding and the runner's constructor included), so no process dies between a
"started" line and its row (a seed that started but never finished, a killed process, counts as tried). Three
reasons blame the process or the code, not the seed, which is then not counted as tried (up to 3 times per seed):
render_error (Vulkan / CUDA: "cannot create buffer", DeviceLost, ...; the process ends at once with 75, since a broken
renderer stays broken in-process: the collision-avoidance attempts show streaks of up to 35), planner_error (not
cuRobo; exit 3) and code_error (an AttributeError, ImportError, NotImplementedError, ...: RoboTwin missing beside the
checkout, a safe mode not merged yet, a bug; 3 in a row end the process with 3). The ITEM_* diagnostic overrides of
the task files must be unset. Exit 0 when --target episodes are kept (or the seeds run out), 75 to be continued
(--max-attempts, the stage deadline, a renderer error; jobs/collect_items_loop.sh relaunches), 3 when the planners or
the code cannot make safe demos (a planner not cuRobo, T1 without the safe model's hand slots, 3 code errors in a row).
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
from types import SimpleNamespace

from .. import collect_demos as CD
from .. import common as C
from ..demo_io import write_episode, write_instructions, write_seed_txt, write_video
from . import instructions as I
from . import keep as K
from .monitor import FRAME_EVERY, SETTLE_STEPS, DemoMonitor, extra_contacts

TASKS = I.ITEM_TASKS
T1 = I.T1
# physics steps per attempt (the runner's max_steps; its first step and the 250-step settle count too); push_zone
# 15000: the T3 safe mode builder's estimate for the routed push (the GPU pilot refines it)
MAX_STEPS = {"selective_avoid": 10000, "push_zone": 15000, "tray_transport": 7500, "clean_pour": 10000,
             "gap_place": 7500}
SEED_STOP = 1_000_000       # the language filter's episode ids start at 1M (main/lang/config.py)
RETRY_REASONS = ("render_error", "planner_error", "code_error")   # the process or the code was broken, not the
RETRY_MAX = 3                                                     # seed: tried again, up to this many times per seed
CODE_STREAK = 3             # this many code errors in a row: the checkout or the code is broken, stop (exit 3)
ALLOWED_ENV = ("ITEM_VIDEO_STYLE",)                 # every other ITEM_* variable changes a scene or the expert
_hard_exit = False          # set on a renderer error: __main__ skips the interpreter teardown (see there)


# ------------------------------------------------------------------ bookkeeping ----
def diag_overrides() -> list[str]:
    """ITEM_* environment variables set (the task files' diagnostic overrides: ITEM_T6_REFERENT, ITEM_T5_STOP_DIST,
    ITEM_SRC_CONTAINER, ...), which a demo collection must not run under."""
    return sorted(k for k in os.environ if k.startswith("ITEM_") and k not in ALLOWED_ENV)


def settings_of(args) -> dict:
    return {"task": args.task, "host": args.host, "embodiment": C.EMBODIMENT, "nominal_mode": "safe",
            "instruction": "plain", "instruction_md5": I.instruction_md5(args.task, args.host),
            "max_steps": int(args.max_steps), "frame_every": FRAME_EVERY, "settle_steps": SETTLE_STEPS,
            "keep_force_n": K.KEEP_FORCE_N, "outside_h_min": K.OUTSIDE_H_MIN, "min_frames": K.MIN_FRAMES,
            "seed_start": int(args.seed_start), "seed_step": int(args.seed_step), "seed_stop": int(args.seed_stop)}


def pin_settings(d: Path, settings: dict) -> None:
    """settings.json pins the directory's settings; a run asking for others is refused (collect_demos.py)."""
    sp = Path(d) / "settings.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        if old != settings:
            diff = {k: (old.get(k), settings.get(k)) for k in sorted(set(old) | set(settings))
                    if old.get(k) != settings.get(k)}
            raise SystemExit(f"{d}: collected with other settings (key: (old, new)) {diff}: use another --config-name")
    else:
        C.atomic_write(sp, json.dumps(settings, indent=1))


def tried_counts(rows: list[dict]) -> Counter:
    """Attempts per seed that count as tried: one per "started" line, minus the renderer / planner errors (the seed is
    tried again), except once a seed had RETRY_MAX of them."""
    started = Counter(int(r["seed"]) for r in rows if "started" in r)
    retry = Counter(int(r["seed"]) for r in rows if r.get("reason") in RETRY_REASONS)
    out = Counter()
    for s, a in started.items():
        r = retry[s]
        out[s] = max(0, a - r + (1 if r >= RETRY_MAX else 0))
    return out


def resume_state(d: Path, rows: list[dict]) -> dict:
    """The kept rows (episode indices must be 0..k-1), with the orphan files of a process that died before its row
    removed and seed.txt rewritten; the tried counts per seed and the attempt count so far."""
    d = Path(d)
    done = [r for r in rows if "reason" in r]
    kept = sorted((r for r in done if r.get("kept")), key=lambda r: r["episode"])
    if [r["episode"] for r in kept] != list(range(len(kept))):
        raise SystemExit(f"{d / 'attempts.jsonl'}: kept episode indices are not 0..{len(kept) - 1}")
    for sub, pat in (("data", "episode*.hdf5"), ("instructions", "episode*.json")):
        for p in sorted((d / sub).glob(pat)):
            idx = p.name.replace("episode", "").split(".")[0]
            if not idx.isdigit() or int(idx) >= len(kept):
                print(f"[collect_items] removing orphan {p}")
                p.unlink()
        for p in sorted((d / sub).glob("*.tmp")):    # a write cut by a crash (demo_io writes tmp + rename)
            print(f"[collect_items] removing {p}")
            p.unlink()
    seeds = [int(r["seed"]) for r in kept]
    write_seed_txt(d / "seed.txt", seeds)
    return {"kept": kept, "seeds": seeds, "counts": tried_counts(rows), "n_done": len(done),
            "n_started": sum(1 for r in rows if "started" in r),
            "n_rej_vid": sum(1 for r in done if r.get("video") and not r.get("kept"))}


def _plain(x):
    """JSON-safe copy (numpy scalars / arrays as common's encoder does, anything else as its str)."""
    def default(v):
        try:
            return C._json_default(v)
        except TypeError:
            return str(v)

    return json.loads(json.dumps(x, default=default))


# ------------------------------------------------------------------ one attempt ----
def _runner_class():
    import main.lang  # noqa: F401 - puts envs/main on main.__path__ (main.item_envs)
    from main.item_envs.runner import ItemEpisodeRunner

    return ItemEpisodeRunner


def planner_problem(runner, task: str) -> str | None:
    """Why the planners cannot make a safe demo, or None: both arms must plan with cuRobo (the mplib fallback of a
    failed cuRobo init ignores the world and the planning constraints), T1 with the safe model (hand + held-object
    sphere slots, safe_plan.install_safe_yml). None as well when the Env was never built."""
    env = getattr(runner, "env", None)
    if env is None:
        return None
    from main.item_envs import safe_plan

    try:
        safe_plan.require_curobo(env.robot)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    if task == T1:
        want = os.path.realpath(str(safe_plan.SAFE_YML))
        for side in ("left", "right"):
            got = os.path.realpath(str(getattr(getattr(env.robot, f"{side}_planner", None), "yml_path", "")))
            if got != want:
                return f"{side} planner loaded {got}, not {want}: the hand and the held object are not modelled"
    return None


def scene_facts(task: str, runner) -> dict:
    """What the scene holds, for the yield per scenario / host / object kind (selection bias, safe_controllers.md 9)."""
    out = {"items": [{"item_id": s.item_id, "category": s.category, "color": s.color, "role": s.role}
                     for s in runner.items if s.kind != "particle_set"]}
    if task == T1:
        rep = getattr(runner, "_spawn_report", None) or {}
        out["placements"] = [{k: p.get(k) for k in ("item_id", "cls", "category", "color", "model_id")}
                             for p in rep.get("placements") or []]
    layout = getattr(runner.env.task, "layout_info", None)
    if isinstance(layout, dict) and "scenario" in layout:     # T3
        out["layout"] = {k: layout[k] for k in ("scenario", "mode", "referent", "elements") if k in layout}
    return _plain(out)


PUSH_KEYS = ("yaw_jump_max_deg", "ee_miss_max", "ee_miss_flag", "box_tilt_max_deg", "fail")
ROUTE_KEYS = ("ok", "reason", "length", "hand", "sides", "turns_deg", "first_deg", "min_clear")
DIAG_KEYS = ("plan_fail_step", "safe_plan_levels", "safe_last_fail")


def expert_diag(task_diag: dict) -> dict:
    """The safe experts' diagnostics at the row's top level, the ones present (the whole task.episode_info stays in
    diag_task), so a pilot reads from attempts.jsonl alone. T3: push (push_via's log: steps per kind, re-plans, the
    largest yaw jump / EE miss / box tilt, the failure), route (plan_route's answer without its via points and search
    detail, n_via instead), plan_fail_step (the step pair a planner failure stopped). T1: safe_plan_levels (plans per
    ladder level, "fail"), safe_last_fail (the kind of the latest plan the ladder could not find)."""
    d = task_diag if isinstance(task_diag, dict) else {}
    out = {}
    push = d.get("push")
    if isinstance(push, dict):
        steps = [s for s in push.get("steps") or [] if isinstance(s, dict)]
        out["push"] = {"steps": dict(Counter(str(s.get("kind")) for s in steps)),
                       "n_replans": len(push.get("replans") or []), **{k: push[k] for k in PUSH_KEYS if k in push}}
    route = d.get("route")
    if isinstance(route, dict):
        out["route"] = {"n_via": len(route.get("via") or []), **{k: route[k] for k in ROUTE_KEYS if k in route}}
    out.update({k: d[k] for k in DIAG_KEYS if k in d})
    return _plain(out)


def run_attempt(task: str, seed: int, host: str | None, out_root: Path, max_steps: int):
    """One attempt, start to keep reason; a failed episode never raises. -> (row, DemoMonitor | None,
    instructions | None); row["reason"] is the keep reason ("kept" or why not)."""
    row = {"task": task, "seed": int(seed), "host": host}
    runner = mon = instr = None
    try:
        try:
            C.set_global_seeds(seed)
            runner = _runner_class()(task, embodiment=C.EMBODIMENT, seed=int(seed), host=host, nominal_mode="safe",
                                     active_rule_id=None, render="none", out_root=str(out_root),
                                     max_steps=int(max_steps))
            row["host"] = runner.host
        except Exception as exc:  # noqa: BLE001 - before any scene, never the seed's: error_reason (a bug: code_error)
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=err[:500], detail="seeding / runner constructor")
            return row, None, None
        try:
            runner.setup()
        except Exception as exc:  # noqa: BLE001 - no scene for this seed (UnStableError, no extra object, ...)
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            problem = planner_problem(runner, task)
            kind = K.error_reason(err)
            if kind == "render_error":
                reason = kind
            elif problem:
                reason = "planner_error"
            elif kind == "code_error":          # RoboTwin missing, a safe mode not merged, a bug: not the scene
                reason = kind
            else:
                reason = "no_scene"
            row.update(reason=reason, error=err[:500], detail=problem)
            return row, None, None
        problem = planner_problem(runner, task)
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
            info = I.item_instruction_info(task, task_obj, host=runner.host, seed=seed)
            instr = I.make_item_instructions(task, info, seed, host=runner.host)
            row["info"] = {k: str(v) for k, v in info.items()}
            protected = K.protected_rules(task, runner.rules)
            row["protected"] = protected
            contact_fn = None
            if task == T1:
                from main.item_envs.items import entity_of

                scene, names = task_obj.scene, {entity_of(s).name for s in extras}
                contact_fn = lambda: extra_contacts(scene, names)   # noqa: E731 - two extra objects count too
            mon = DemoMonitor(task_obj, contact_fn=contact_fn).install()
        except Exception as exc:  # noqa: BLE001 - a broken attempt is logged, the next seed follows
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=err[:500])
            return row, mon, None
        try:
            meta, err = runner.run()
        except Exception as exc:  # noqa: BLE001 - run() records episode errors itself; this is anything else
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=f"run: {err}"[:500])
            return row, mon, None
        diag = dict(getattr(meta, "diag", None) or {})
        task_diag = diag.get("task")                  # task.episode_info as the runner copied it at the end
        if not isinstance(task_diag, dict):
            task_diag = getattr(task_obj, "episode_info", None) or {}
        broken, flown = [], []
        trace = Path(runner.out_dir) / "trace.npz"
        try:
            if trace.exists():
                t, m, c = K.read_trace(trace, runner.rules)
                per, broken = K.score_trace(runner.rules, protected, t, m, c)
                flown = K.flown_items(trace)
                row["flown"] = flown
                row["min_h"] = {rid: v["min_h"] for rid, v in per.items()}
                row["first_violation_tick"] = {rid: v["first_tick"] for rid, v in per.items()
                                               if v["first_tick"] is not None}
                row["first_violation_t"] = {rid: v["first_t"] for rid, v in per.items() if v["first_t"] is not None}
            elif err is None:
                err = f"no trace.npz in {runner.out_dir}"
        except Exception as exc:  # noqa: BLE001 - the type first: error_reason reads it (a bug here is a code_error)
            traceback.print_exc()
            err = err or f"{type(exc).__name__}: trace: {exc}"
        plan_ok = bool(getattr(task_obj, "plan_success", True))   # the end-of-episode flag (collect_demos.py)
        contact = None
        if task == T1:
            contact = {"touched": mon.touched, "force_max": mon.force_max}
            row.update(contact=mon.touched, contact_partners=sorted(mon.partners), force_max=float(mon.force_max),
                       first_touch_step=mon.first_touch,
                       first_touch_t=None if mon.first_touch is None else mon.first_touch / C.PHYS_HZ,
                       plan_status=list(((diag.get("safe_plan") or {}).get("plan_status") or [])[:20]))
        row.update(error=None if err is None else str(err)[:500], safe_fail=task_diag.get("safe_fail"),
                   task_success=bool(meta.task_success), plan_success=plan_ok,
                   plan_failures=int(meta.plan_failures), broken=broken,   # plan_failures: recovered ones too
                   risk=_plain(getattr(meta, "risk", None)), diag_task=_plain(task_diag),
                   n_skipped_frames=mon.n_skipped, play_steps=mon.play_steps, settle_steps=mon.settled,
                   n_steps=mon.n, duration_s=mon.n / C.PHYS_HZ,
                   runner_dir=os.path.relpath(runner.out_dir, Path(out_root).resolve().parent.parent))
        row.update(expert_diag(task_diag))            # push / route (T3), safe_plan_levels / safe_last_fail (T1)
        row["reason"] = K.keep_reason({"err": err, "safe_fail": task_diag.get("safe_fail"), "plan_success": plan_ok,
                                       "task_success": meta.task_success, "flown": flown, "broken": broken,
                                       "contact": contact,
                                       "n_frames": len(mon.frames)})
        return row, mon, instr
    finally:
        if runner is not None:
            runner.close()


# ------------------------------------------------------------------ the loop ----
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--host", default=None, help="selective_avoid only, required: the host task of this config")
    ap.add_argument("--root", required=True, help="absolute demo root (packer input_path)")
    ap.add_argument("--config-name", required=True, help="config directory name (part of the packer uuid)")
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--seed-start", type=int, default=300000)
    ap.add_argument("--seed-step", type=int, default=1)
    ap.add_argument("--seed-stop", type=int, default=SEED_STOP, help=f"first seed not tried (<= {SEED_STOP})")
    ap.add_argument("--max-attempts", type=int, default=40,
                    help="attempts per process, then exit 75 (one Env build each: far below the ~750 per process)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help=f"physics steps per attempt (default per task: {MAX_STEPS})")
    ap.add_argument("--video", action="store_true",
                    help="head-camera videos of every --video-every-th kept episode and the first --rejected-videos "
                         "rejected attempts, in <config>/videos/")
    ap.add_argument("--video-every", type=int, default=10)
    ap.add_argument("--rejected-videos", type=int, default=5)
    args = ap.parse_args(argv)
    if args.task == T1:
        import main.lang  # noqa: F401
        from main.item_envs.registry import get_task_entry

        hosts = get_task_entry(T1).host_tasks
        if args.host not in hosts:
            ap.error(f"--task {T1} needs --host, one of {list(hosts)} (one config per host)")
    elif args.host:
        ap.error(f"--host is for --task {T1} only")
    args.max_steps = int(args.max_steps or MAX_STEPS[args.task])
    if not (0 <= args.seed_start < args.seed_stop <= SEED_STOP) or args.seed_step < 1:
        ap.error(f"want 0 <= --seed-start < --seed-stop <= {SEED_STOP} and --seed-step >= 1")
    return args


def main(argv=None) -> int:
    global _hard_exit
    _hard_exit = False
    args = parse_args(argv)
    bad = diag_overrides()
    if bad:
        raise SystemExit(f"diagnostic overrides set: {bad}: unset them (they change the scenes or the expert)")
    root = Path(args.root)
    if not root.is_absolute():
        raise SystemExit("--root must be absolute (Env chdirs to the RoboTwin root)")
    d = root / args.task / args.config_name
    d.mkdir(parents=True, exist_ok=True)
    from main.lang.locks import WriterLock

    lock = WriterLock(d, wait_s=200.0)
    try:
        return _collect(args, d, lock)
    finally:
        lock.release()


def _collect(args, d: Path, lock) -> int:
    global _hard_exit
    settings = settings_of(args)
    pin_settings(d, settings)
    log = d / "attempts.jsonl"
    st = resume_state(d, C.read_jsonl(log))
    kept, seeds, counts, n_started, n_rej_vid = st["kept"], st["seeds"], st["counts"], st["n_started"], st["n_rej_vid"]
    kept_seeds = set(seeds)
    name = f"{args.task}/{args.config_name}"
    print(f"[collect_items] {name}: {len(kept)}/{args.target} kept, {st['n_done']} attempts so far")
    walk = SimpleNamespace(seeds_from=None, seed_start=args.seed_start, seed_step=args.seed_step,
                           seed_stop=args.seed_stop)
    seed_iter = CD._seed_iter(walk, counts, kept_seeds)        # each seed once, skipping the tried ones
    n_proc, times, code_streak = 0, [], 0
    while len(kept) < args.target:
        seed = next(seed_iter, None)
        if seed is None:
            print(f"[collect_items] seeds exhausted: {len(kept)}/{args.target} kept")
            return 0
        if n_proc >= args.max_attempts:
            print(f"[collect_items] {n_proc} attempts in this process: continue in a fresh one")
            return 75
        mean_t = sum(times) / len(times) if times else 240.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            print(f"[collect_items] stage deadline near (mean attempt {mean_t:.0f} s): continue later")
            return 75
        n_proc += 1
        idx, n_started = n_started, n_started + 1
        att_root = d / "runner" / str(idx)
        if att_root.exists():                         # never two attempts in one runner directory
            shutil.rmtree(att_root)
        lock.check()
        C.append_jsonl(log, {"seed": seed, "started": time.strftime("%F %T"), "attempt": idx})
        counts[seed] += 1
        t0 = time.time()
        try:
            row, mon, instr = run_attempt(args.task, seed, args.host, att_root, args.max_steps)
        except Exception as exc:  # noqa: BLE001 - this collector's own code: classified, the "started" line gets a row
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row, mon, instr = ({"task": args.task, "seed": int(seed), "host": args.host, "reason": K.error_reason(err),
                                "error": err[:500], "detail": "run_attempt"}, None, None)
        reason = row.get("reason") or "error"
        frames = mon.frames if mon is not None else []
        row.update(attempt=idx, config=args.config_name, kept=reason == "kept", reason=reason, n_frames=len(frames))
        lock.check()
        if reason == "kept":
            k = len(kept)
            write_episode(d / "data" / f"episode{k}.hdf5", frames)
            write_instructions(d / "instructions" / f"episode{k}.json", instr)
            write_seed_txt(d / "seed.txt", seeds + [seed])
            row.update(episode=k, instruction_example=instr["seen"][0], frame_steps=list(mon.recorder.steps))
        want_video = args.video and frames and (
            (reason == "kept" and args.video_every and row["episode"] % args.video_every == 0)
            or (reason != "kept" and n_rej_vid < args.rejected_videos))
        if want_video:
            vname = (f"episode{row['episode']}_seed{seed}.mp4" if reason == "kept"
                     else f"rejected_seed{seed}_{reason.replace(':', '_').replace('|', '_')}.mp4")
            try:
                write_video(d / "videos" / vname, frames)
                row["video"] = vname
                n_rej_vid += reason != "kept"
            except Exception as exc:  # noqa: BLE001 - a video is for looking, never worth the attempt
                print(f"[collect_items] video failed: {exc}")
        row["wall_s"] = time.time() - t0
        C.append_jsonl(log, _plain(row))
        if reason == "kept":
            seeds.append(seed)
            kept.append(row)
            kept_seeds.add(seed)
        times.append(row["wall_s"])
        print(f"[collect_items] {name} seed {seed}: {reason} (success {row.get('task_success')}, plan failures "
              f"{row.get('plan_failures')}, contact {row.get('contact')} {row.get('contact_partners')}, "
              f"{row['n_frames']} frames, {row['wall_s']:.0f} s) -> {len(kept)}/{args.target} kept", flush=True)
        if reason == "render_error":
            print(f"[collect_items] renderer error ({row.get('error')}): this process stops, seed {seed} is tried "
                  "again by the next one", flush=True)
            _hard_exit = True
            return 75
        if reason == "planner_error":
            print(f"[collect_items] refusing to run: {row.get('detail')}", flush=True)
            return 3
        code_streak = code_streak + 1 if reason == "code_error" else 0
        if code_streak >= CODE_STREAK:
            print(f"[collect_items] {code_streak} code errors in a row (last: {row.get('error')}): the checkout or "
                  "the code is broken, not the scenes; refusing to go on", flush=True)
            return 3
        del frames, mon, instr
    print(f"[collect_items] {name}: target reached ({len(kept)} kept)")
    return 0


if __name__ == "__main__":
    rc = main()
    if _hard_exit:   # renderer error: the interpreter's teardown can abort on a lost device (vk::Device::waitIdle)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)
