"""Collect demonstrations of one task in the CrossSafe obstacle scenes (PLAN.md, "Demonstrations").

    scripts/lang/in_container.sh python -m vla_finetuning.collect_demos --task stack_blocks_two \
        --mode safe --root /work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/data/demos_safe \
        --config-name s0 --seed-start 0 --seed-step 2 --target 50

--mode safe:    RoboTwin's expert plans with cuRobo's world = RoboTwin's table + the box grown by --margin
                (and --top-extra on top), and with the WSG hand and the held object in cuRobo's robot model
                (hand_model.py); kept only if it succeeds with no contact at any physics step, force < 20 N
                and min h >= --keep-h (--keep safe).
--mode nominal: the careless expert (RoboTwin's stock model, table only), for the control arm (--keep
                success: success only, collisions allowed). --seeds-from <safe attempts.jsonl> walks the
                seeds the safe run kept, in order, so both sets cover the same layouts (review B2).

Frames are recorded where RoboTwin's own collector records them (task._take_picture, called by the
executors every save_freq = 15 physics steps of a motion and at its start and end), so the frame spacing
matches HoloBrain's RoboTwin training data (review 10-15, data F1).

Seeds run from --seed-start in steps of --seed-step (shards: same step, different starts), below the
evaluation grid (70000+). Every attempt is logged in <root>/<task>/<config>/attempts.jsonl: a "started"
line before the scene is built, then the result (kept or not, with the reason); a seed that started but
never finished (the process died) counts as tried. Kept episode i is data/episode{i}.hdf5 +
instructions/episode{i}.json + entry i of seed.txt, the layout HoloBrain's packer reads. settings.json
pins the demonstrator settings of the directory; a rerun with other settings is refused. Exit 0 when
--target episodes are kept (or the seeds run out), 75 to be continued (process recycling after
--max-attempts, which bounds RoboTwin task inits per process, or the stage deadline), 3 when the planner
is not cuRobo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
import time
import traceback
from pathlib import Path

from . import common as C
from .demo_io import FrameRecorder, write_episode, write_instructions, write_seed_txt, write_video

SETTINGS = ("mode", "keep", "margin", "top_extra", "keep_h", "cap_s", "payload_margin")


class PlannerStatusLog:
    """Failure statuses of cuRobo's plans (RoboTwin's CuroboPlanner drops them): wraps plan_single /
    plan_batch of the four MotionGens once (they are cached across episodes); `sink` is reset per episode."""

    sink: list = []

    @classmethod
    def install(cls, robot) -> None:
        for planner in (robot.left_planner, robot.right_planner):
            for mg in (planner.motion_gen, planner.motion_gen_batch):
                for name in ("plan_single", "plan_batch"):
                    fn = getattr(mg, name, None)
                    if fn is None or getattr(fn, "_vf_status", False):
                        continue

                    def wrapped(*a, _fn=fn, _name=name, **k):
                        res = _fn(*a, **k)
                        try:
                            ok = bool(res.success.any()) if hasattr(res.success, "any") else bool(res.success)
                            if not ok:
                                cls.sink.append(f"{_name}:{getattr(res, 'status', '?')}")
                        except Exception:  # noqa: BLE001
                            pass
                        return res

                    wrapped._vf_status = True
                    setattr(mg, name, wrapped)


def run_planner_episode(env, mode: str, *, margin: float, top_extra: float, cap_s: float, record: bool,
                        payload_margin: float = 0.01):
    """One play_once of RoboTwin's expert in this scene. mode "safe": the box in cuRobo's world and the
    hand + held object in its robot model; "nominal": table only, stock model.
    -> (result dict, FrameRecorder or None, play_once info)."""
    from . import hand_model as H

    C.require_curobo(env.robot)
    if mode == "safe":           # the box only: RoboTwin's misplaced table cuboid blocks the modelled hand
        C.push_world(env.robot, C.obstacle_pose(env), margin=margin, top_extra=top_extra, table=False)
    else:                        # RoboTwin's stock world, as the careless expert plans
        C.push_world(env.robot, None)
    PlannerStatusLog.install(env.robot)
    PlannerStatusLog.sink = []
    syncs: list = []
    uninstall = (H.install_plan_hooks(env, syncs, payload_margin, status_sink=PlannerStatusLog.sink)
                 if mode == "safe" else None)
    rec = FrameRecorder(env.task) if record else None
    if rec is not None:
        env.task._take_picture = lambda: rec(mon.n)     # RoboTwin's recorder hook, at its own cadence
    mon = C.PhysicsMonitor(env, int(cap_s * C.PHYS_HZ), poll_success=True)
    err = None
    t0 = time.time()
    try:
        with mon:
            mon.sample()
            env.task.play_once()
    except C.SimTimeout as exc:
        err = f"timeout: {exc}"
    except Exception as exc:  # noqa: BLE001 - a failed plan is an outcome, not a crash
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if rec is not None:
            del env.task._take_picture
        if uninstall is not None:
            uninstall()
    plan_ok = bool(getattr(env.task, "plan_success", True))
    try:
        success = err is None and plan_ok and bool(env.task.check_success())
    except Exception as exc:  # noqa: BLE001
        err = err or f"check_success: {type(exc).__name__}: {exc}"
        success = False
    info = getattr(env.task, "info", None)
    info = dict(info.get("info", {})) if isinstance(info, dict) else {}
    try:
        C.push_world(env.robot, None)              # the MotionGens are cached across episodes
    except Exception as exc:  # noqa: BLE001
        print(f"[collect] world reset failed: {exc}")
    res = {"error": err, "plan_success": plan_ok, "success": bool(success), "wall_s": time.time() - t0,
           "timeout": bool(err and err.startswith("timeout")), "plan_failures": PlannerStatusLog.sink[:20],
           "n_plans": len(syncs) if mode == "safe" else None,
           "n_plans_with_payload": sum(1 for s in syncs if s["n_payload"]) if mode == "safe" else None,
           "n_fallbacks": sum(1 for s in syncs if s.get("fallback")) if mode == "safe" else None,
           "fallback_levels": sorted({s["level"] for s in syncs if s.get("level") != "full"}) if mode == "safe" else None,
           **C.episode_metrics(mon)}
    if success and res.get("t_success") is None:
        res["t_success"] = res.get("duration_s")
    return res, rec, info


def keep_reason(res: dict, keep: str, keep_h: float) -> str:
    """'kept' or why not."""
    if res.get("error"):
        return "timeout" if res.get("timeout") else "error"
    if not res.get("plan_success"):
        return "plan_failed"
    if not res.get("success"):
        return "task_failed"
    if not res.get("n_ticks"):
        return "no_ticks"
    if keep == "success":
        return "kept"
    if res["contact"]:
        return "contact"
    if res["force20"]:
        return "force"
    if keep_h is not None and not (res["min_h_cm"] >= 100.0 * keep_h):
        return "too_close"
    return "kept"


def _seed_iter(args, counts: Counter, kept_seeds: set):
    """Seeds to attempt. counts (attempts per seed so far) and kept_seeds are updated by the caller as it goes.
    A seed range: each seed once. --seeds-from: the source's kept seeds in order, each retried up to
    --tries-per-seed times until kept (the careless expert's outcome on one layout varies run to run: cuRobo
    is nondeterministic), so the matched sets lose fewer layouts."""
    if args.seeds_from:
        src = C.read_jsonl(Path(args.seeds_from))
        seeds = [int(r["seed"]) for r in sorted((r for r in src if r.get("kept")), key=lambda r: r["episode"])]
        for s in seeds:
            while s not in kept_seeds and counts[s] < args.tries_per_seed:
                yield s
        return
    s = args.seed_start
    while s < args.seed_stop:
        if counts[s] == 0:
            yield s
        s += args.seed_step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=C.TASKS)
    ap.add_argument("--mode", choices=("safe", "nominal"), default="safe")
    ap.add_argument("--keep", choices=("safe", "success"), default=None,
                    help="keep rule (default: safe for --mode safe, success for --mode nominal)")
    ap.add_argument("--root", required=True, help="absolute demo root (packer input_path)")
    ap.add_argument("--config-name", required=True, help="config directory name (part of the packer uuid)")
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-step", type=int, default=1)
    ap.add_argument("--seed-stop", type=int, default=C.SEED_BASE)
    ap.add_argument("--seeds-from", default=None, help="attempts.jsonl whose kept seeds to walk, in order")
    ap.add_argument("--tries-per-seed", type=int, default=3, help="--seeds-from: attempts per seed until kept")
    ap.add_argument("--margin", type=float, default=0.01, help="box growth on every side for the planner (m)")
    ap.add_argument("--top-extra", type=float, default=0.0, help="extra box height for the planner (m)")
    ap.add_argument("--payload-margin", type=float, default=0.01, help="held-object growth in cuRobo (m)")
    ap.add_argument("--keep-h", type=float, default=None,
                    help="also require min h >= this (m); off by default: h measures the held object by its box, and "
                         "a round bowl's box corners read 'inside' the obstacle with nothing touching (smoke B4)")
    ap.add_argument("--cap-s", type=float, default=40.0, help="physics cap per attempt (s)")
    ap.add_argument("--max-attempts", type=int, default=40, help="attempts per process, then exit 75")
    ap.add_argument("--video-every", type=int, default=10, help="video of every k-th kept episode")
    ap.add_argument("--rejected-videos", type=int, default=5)
    args = ap.parse_args(argv)
    args.keep = args.keep or ("safe" if args.mode == "safe" else "success")
    root = Path(args.root)
    if not root.is_absolute():
        raise SystemExit("--root must be absolute (Env chdirs to the RoboTwin root)")
    d = root / args.task / args.config_name
    d.mkdir(parents=True, exist_ok=True)
    from main.lang.locks import WriterLock

    lock = WriterLock(d, wait_s=200.0)
    settings = {k: getattr(args, k) for k in SETTINGS}
    sp = d / "settings.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        if old != settings:
            raise SystemExit(f"{d}: collected with {old}, this run asks for {settings}: use another --config-name")
    else:
        C.atomic_write(sp, json.dumps(settings, indent=1))
    if args.mode == "safe":
        from .hand_model import install_safe_yml

        install_safe_yml()
    log = d / "attempts.jsonl"
    rows = C.read_jsonl(log)
    done_rows = [r for r in rows if "reason" in r]
    kept = sorted((r for r in done_rows if r.get("kept")), key=lambda r: r["episode"])
    if [r["episode"] for r in kept] != list(range(len(kept))):
        raise SystemExit(f"{log}: kept episode indices are not 0..{len(kept) - 1}")
    for sub, pat in (("data", "episode*.hdf5"), ("instructions", "episode*.json")):
        for p in (d / sub).glob(pat):            # written by a process that died before its log line
            idx = p.name.replace("episode", "").split(".")[0]
            if not idx.isdigit() or int(idx) >= len(kept):
                print(f"[collect] removing orphan {p}")
                p.unlink()
    seeds = [int(r["seed"]) for r in kept]
    write_seed_txt(d / "seed.txt", seeds)
    counts = Counter(int(r["seed"]) for r in rows if "started" in r)   # one "started" line per attempt
    for r in rows:                                    # rows from before the "started" lines (smoke dirs)
        if "reason" in r and not counts[int(r["seed"])]:
            counts[int(r["seed"])] = 1
    kept_seeds = {int(r["seed"]) for r in kept}
    n_rej_vid = sum(1 for r in done_rows if r.get("video") and not r.get("kept"))
    print(f"[collect] {args.task}/{args.config_name}: {len(kept)}/{args.target} kept, {len(done_rows)} attempts so far")
    n_proc, times = 0, []
    seed_iter = _seed_iter(args, counts, kept_seeds)
    checked_planner = False
    while len(kept) < args.target:
        seed = next(seed_iter, None)
        if seed is None:
            print(f"[collect] seeds exhausted: {len(kept)}/{args.target} kept")
            return 0
        if n_proc >= args.max_attempts:
            print(f"[collect] {n_proc} attempts in this process: continue in a fresh one")
            return 75
        mean_t = sum(times) / len(times) if times else 240.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            print(f"[collect] stage deadline near (mean attempt {mean_t:.0f} s): continue later")
            return 75
        n_proc += 1
        t0 = time.time()
        row = {"task": args.task, "seed": seed, **settings}
        lock.check()
        C.append_jsonl(log, {"seed": seed, "started": time.strftime("%F %T")})
        counts[seed] += 1
        C.set_global_seeds(seed)
        try:
            env = C.build_scene(args.task, seed)
        except C.NoScene as exc:
            row.update(kept=False, reason="no_scene", detail=str(exc), wall_s=time.time() - t0)
            lock.check()
            C.append_jsonl(log, row)
            continue
        except Exception as exc:  # noqa: BLE001 - a scene that cannot be built is skipped, not fatal
            traceback.print_exc()
            row.update(kept=False, reason="scene_error", detail=f"{type(exc).__name__}: {exc}",
                       wall_s=time.time() - t0)
            lock.check()
            C.append_jsonl(log, row)
            continue
        if not checked_planner:                      # cuRobo init failure -> mplib, which ignores the box
            try:
                C.require_curobo(env.robot)
                if args.mode == "safe":
                    from .hand_model import uses_safe_yml

                    if not uses_safe_yml(env.robot):
                        raise RuntimeError("the planners did not load curobo_safe.yml")
            except RuntimeError as exc:
                env.close()
                print(f"[collect] refusing to run: {exc}")
                return 3
            checked_planner = True
        rec, info_attr, instr, reason = None, {}, None, "error"
        try:
            row["obstacle_xyz"] = [float(v) for v in env.obstacle_xyz]
            info_attr = C.instruction_info(env.task, args.task)
            res, rec, info_play = run_planner_episode(env, args.mode, margin=args.margin, top_extra=args.top_extra,
                                                      cap_s=args.cap_s, record=True,
                                                      payload_margin=args.payload_margin)
            row.update(res)
            if info_play:
                row["info_match"] = C.strip_arm(info_play) == C.strip_arm(info_attr)
            reason = keep_reason(res, args.keep, args.keep_h)
            if reason == "kept":
                try:
                    instr = C.make_instructions(args.task, info_attr, seed)
                except SystemExit as exc:            # RoboTwin's generator exit()s on a missing description
                    raise RuntimeError(f"instruction generator exited: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - logged as a failed attempt; the next seed follows
            traceback.print_exc()
            row["error"] = f"{type(exc).__name__}: {exc}"
            reason = "error"
        finally:
            env.close()
        row.update(kept=reason == "kept", reason=reason, n_frames=len(rec.frames) if rec else 0,
                   info=C.strip_arm(info_attr))
        lock.check()
        if reason == "kept":
            k = len(kept)
            write_episode(d / "data" / f"episode{k}.hdf5", rec.frames)
            write_instructions(d / "instructions" / f"episode{k}.json", instr)
            write_seed_txt(d / "seed.txt", seeds + [seed])
            row["episode"] = k
            row["instruction_example"] = instr["seen"][0] if instr["seen"] else instr["unseen"][0]
        row["wall_s"] = time.time() - t0
        C.append_jsonl(log, row)
        if reason == "kept":
            seeds.append(seed)
            kept.append(row)
            kept_seeds.add(seed)
        times.append(row["wall_s"])
        want_video = (reason == "kept" and args.video_every and row["episode"] % args.video_every == 0) or (
            reason != "kept" and rec is not None and rec.frames and n_rej_vid < args.rejected_videos)
        if want_video:
            name = (f"episode{row['episode']}_seed{seed}.mp4" if reason == "kept"
                    else f"rejected_seed{seed}_{reason}.mp4")
            try:
                write_video(d / "videos" / name, rec.frames)
                n_rej_vid += reason != "kept"
            except Exception as exc:  # noqa: BLE001 - a video is for looking, never worth the attempt
                print(f"[collect] video failed: {exc}")
        print(f"[collect] {args.task} seed {seed}: {reason} (success {row.get('success')}, contact "
              f"{row.get('contact')} {row.get('contact_partners')}, min h {row.get('min_h_cm', float('nan')):.1f} cm, "
              f"{row.get('n_frames')} frames, plan failures {row.get('plan_failures')}, {row['wall_s']:.0f} s) "
              f"-> {len(kept)}/{args.target} kept")
    print(f"[collect] {args.task}/{args.config_name}: target reached ({len(kept)} kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
