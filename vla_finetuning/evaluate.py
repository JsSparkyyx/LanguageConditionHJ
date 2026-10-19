"""Closed-loop test on the CrossSafe evaluation scenes (PLAN.md, "Test"): the same 50 scenes per task for
every agent, collision and success per episode.

    # a HoloBrain model directory (exported, or models/base_post_training_robotwin)
    scripts/lang/in_container.sh python -m vla_finetuning.evaluate --agent vla \
        --model-dir /abs/vla_finetuning/models/<name>/model --out /abs/vla_finetuning/results/<name>
    # references on the same scenes: RoboTwin's expert, obstacle-unaware or obstacle-aware (hand model)
    ... --agent nominal --out .../results/planner_nominal
    ... --agent safe    --out .../results/planner_safe

--agent vla: the observation of HoloBrain's HoloBrainRoboTwinPolicy (task.get_obs() + the instruction:
RoboTwin's generator, the task's arm-free templates, one 'unseen' sentence per scene) goes through the
model's inference pipeline; --chunk of the 64 predicted actions run per inference through RoboTwin's
take_action (qpos), until RoboTwin's per-task action limit or the latched eval_success. The physics cap
is a hang guard only: --cap-steps-per-action x the action limit (RoboTwin's protocol has no time cap).

Rows: <out>/episodes_<task>.jsonl, one per scene, resumable. A row is 'crash' when the episode broke with
an exception that is not the physics cap; report.py leaves crash and no-scene rows out of the rates and
lists them. The parent runs one child process per task (bounds RoboTwin task inits per process) and writes
<out>/summary.json. Exit 75 = stopped at the stage deadline (rerun the same command), 1 = a task is still
incomplete after the respawns, 3 = the planner is not cuRobo.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import common as C

_DEVNULL = open(os.devnull, "w")


class HoloBrainPolicy:
    """HoloBrain's RoboTwin policy (projects/holobrain/policy/robotwin_policy.py: _process_raw_obs and act,
    RoboOrchardLab 241bbc6) without its import chain: `projects` is shadowed by pytorch3d's package here, and
    robotwin_policy imports robo_orchard_lab.envs.robotwin.env, which needs foxglove / mcap (not installed;
    review 10-15, eval F1). Same observation, same pipeline call, same action cache."""

    def __init__(self, model_dir: Path, chunk: int):
        from robo_orchard_lab.models.holobrain.pipeline import HoloBrainInferencePipeline

        self.pipe = HoloBrainInferencePipeline.load_pipeline(
            directory=str(model_dir), inference_prefix=C.INFERENCE_PREFIX, device="cuda",
            load_weights=True, load_impl="native")
        self.pipe.model.eval()
        self.chunk = int(chunk)
        self.cache: list = []

    def reset(self) -> None:
        self.cache = []

    @staticmethod
    def _homog(m, shapes, name: str) -> np.ndarray:
        m = np.asarray(m, np.float64)
        if m.shape == (4, 4):
            return m.copy()
        if m.shape in shapes:
            out = np.eye(4)
            out[: m.shape[0], : m.shape[1]] = m
            return out
        raise ValueError(f"{name} shape {m.shape}")

    def _input(self, obs: dict, instruction: str):
        from robo_orchard_lab.models.holobrain.processor import MultiArmManipulationInput

        images, depths, t_w2c, intr = {}, {}, {}, {}
        for cam, d in obs["observation"].items():
            images[cam] = [d["rgb"]]
            depths[cam] = [d["depth"] / 1000]
            t_w2c[cam] = self._homog(d["extrinsic_cv"], ((3, 4), (4, 4)), "extrinsic_cv")
            intr[cam] = self._homog(d["intrinsic_cv"], ((3, 3),), "intrinsic_cv")
        return MultiArmManipulationInput(image=images, depth=depths, intrinsic=intr, t_world2cam=t_w2c,
                                         history_joint_state=[obs["joint_action"]["vector"]],
                                         instruction=instruction)

    def needs_obs(self) -> bool:
        return not self.cache

    def act(self, obs: dict | None, instruction: str) -> np.ndarray:
        if not self.cache:
            out = self.pipe(self._input(obs, instruction))
            a = out.action.cpu().numpy()
            if a.ndim != 2 or a.shape[1] != 14:
                raise ValueError(f"pipeline action {a.shape}, expected [steps, 14]")
            self.cache = [a[i] for i in range(min(len(a), self.chunk))]
        return self.cache.pop(0)


def run_vla_episode(env, policy: HoloBrainPolicy, instruction: str, *, cap_steps_per_action: float,
                    video: list | None = None) -> dict:
    task = env.task
    task.step_lim = C.step_limit(env.task_name)
    policy.reset()

    def grab(_n):
        video.append(np.asarray(task.get_obs()["observation"]["head_camera"]["rgb"]).copy())

    cap = int(cap_steps_per_action * task.step_lim)
    mon = C.PhysicsMonitor(env, cap, on_frame=grab if video is not None else None)
    err, crash, n_inf, t_inf = None, False, 0, 0.0
    t0 = time.time()
    try:
        with mon:
            mon.sample()
            while not (task.eval_success or task.take_action_cnt >= task.step_lim):
                obs = None
                if policy.needs_obs():
                    obs = task.get_obs()
                    ti = time.time()
                a = policy.act(obs, instruction)
                if obs is not None:
                    t_inf += time.time() - ti
                    n_inf += 1
                with contextlib.redirect_stdout(_DEVNULL):   # take_action prints a counter per action
                    task.take_action(np.asarray(a, np.float64), action_type="qpos")
    except C.SimTimeout as exc:
        err = f"timeout: {exc}"
    except Exception as exc:  # noqa: BLE001 - logged as a crash row, left out of the rates
        err, crash = f"{type(exc).__name__}: {exc}", True
        traceback.print_exc()
    success = bool(task.eval_success)
    if not success and err is None:
        try:
            success = bool(task.check_success())
        except Exception:  # noqa: BLE001
            success = False
    if success and mon.t_success is None:
        mon.t_success = mon.n / C.PHYS_HZ       # latched inside the last take_action, after the last sample
    n_act = int(task.take_action_cnt)
    end = "success" if success else ("timeout" if err and not crash else ("crash" if crash else "step_limit"))
    return {"error": err, "crash": crash, "success": None if crash else success, "end": end,
            "n_actions": n_act, "step_lim": int(task.step_lim), "n_inferences": n_inf,
            "infer_s_mean": t_inf / n_inf if n_inf else float("nan"), "wall_s": time.time() - t0,
            "steps_per_action": mon.n / n_act if n_act else float("nan"),
            "timeout": bool(err and not crash), **C.episode_metrics(mon)}


def child(args) -> int:
    from main.lang.locks import WriterLock

    from .collect_demos import run_planner_episode
    from .demo_io import write_video

    out = Path(args.out)
    task = args.tasks[0]
    lock = WriterLock(out / "locks" / task, wait_s=200.0)
    rows_p = out / f"episodes_{task}.jsonl"
    done = {int(r["episode"]) for r in C.read_jsonl(rows_p)}
    if args.agent == "safe":
        from .hand_model import install_safe_yml

        install_safe_yml()
    policy = HoloBrainPolicy(Path(args.model_dir), args.chunk) if args.agent == "vla" else None
    times, checked = [], args.agent == "vla"
    for ep in range(args.episodes):
        if ep in done:
            continue
        mean_t = sum(times) / len(times) if times else 300.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            print(f"[eval] stage deadline near: {task} continues later")
            return 75
        t0 = time.time()
        row = {"task": task, "episode": ep, "agent": args.agent, "model": args.model_name, "scene": args.scene}
        try:
            if args.scene == "standard":
                env, seed = C.standard_scene(task, ep, args.seed_base)
            else:
                env, seed = C.eval_scene(task, ep, args.seed_base)
        except Exception as exc:  # noqa: BLE001 - NoScene or a build error: an error row, the next scene follows
            traceback.print_exc()
            row.update(error=f"no_scene: {type(exc).__name__}: {exc}", success=None)
            lock.check()
            C.append_jsonl(rows_p, row)
            continue
        if not checked:
            try:
                C.require_curobo(env.robot)
                if args.agent == "safe":
                    from .hand_model import uses_safe_yml

                    if not uses_safe_yml(env.robot):
                        raise RuntimeError("the planners did not load curobo_safe.yml")
            except RuntimeError as exc:
                env.close()
                print(f"[eval] refusing to run: {exc}")
                return 3
            checked = True
        video = [] if (args.video_every and ep % args.video_every == 0) else None
        rec = None
        try:
            row.update(seed=seed, obstacle_xyz=[float(v) for v in env.obstacle_xyz] if env.obstacle is not None else None)
            info = C.instruction_info(env.task, task)
            if args.agent == "vla":
                instr = C.make_instructions(task, info, seed)
                pool = instr["unseen"] or instr["seen"]
                row["instruction"] = random.Random(seed).choice(pool)
                C.set_global_seeds(seed + 1)    # the diffusion sampler's noise
                res = run_vla_episode(env, policy, row["instruction"], cap_steps_per_action=args.cap_steps_per_action,
                                      video=video)
            else:
                res, rec, _ = run_planner_episode(env, args.agent, margin=args.margin, top_extra=args.top_extra,
                                                  cap_s=args.planner_cap_s, record=video is not None)
                res["end"] = "success" if res["success"] else ("timeout" if res["timeout"] else "failed")
            row.update(res)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            row.update(error=f"{type(exc).__name__}: {exc}", crash=True, success=None)
        finally:
            env.close()
        if video is not None and row.get("n_ticks"):
            tag = f"{task}_ep{ep:03d}_{'success' if row.get('success') else 'fail'}_" \
                  f"{'collision' if row.get('collision') else 'clean'}.mp4"
            try:
                if rec is not None:
                    write_video(out / "videos" / tag, rec.frames)
                elif video:
                    _write_rgb_video(out / "videos" / tag, video)
                row["video"] = tag
            except Exception as exc:  # noqa: BLE001
                print(f"[eval] video failed: {exc}")
        row["wall_s_total"] = time.time() - t0
        lock.check()
        C.append_jsonl(rows_p, row)
        times.append(row["wall_s_total"])
        print(f"[eval] {args.agent} {task} ep{ep:03d} seed {row.get('seed')}: {row.get('end')} success "
              f"{row.get('success')} collision {row.get('collision')} {row.get('contact_partners')} min h "
              f"{row.get('min_h_cm', float('nan')):.1f} cm, {row.get('n_actions')} actions, "
              f"{row.get('steps_per_action', float('nan')):.1f} steps/action ({row['wall_s_total']:.0f} s)")
    return 0


def _write_rgb_video(path: Path, frames: list, fps: float = 250 / 15) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.mp4")
    w = imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    try:
        for f in frames:
            w.append_data(f)
    finally:
        w.close()
    os.replace(tmp, path)


def parent(args, argv) -> int:
    from .report import summarize_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_args.json").write_text(json.dumps(vars(args), indent=1))
    rc_all = 0
    for task in args.tasks:
        for attempt in range(4):
            n_rows = len(C.read_jsonl(out / f"episodes_{task}.jsonl"))
            if n_rows >= args.episodes:
                break
            cmd = [sys.executable, "-m", "vla_finetuning.evaluate", *argv, "--child", "--tasks", task]
            rc = subprocess.run(cmd, cwd=str(C.REPO)).returncode
            if rc in (75, 3):
                rc_all = rc
                break
            if rc != 0:
                print(f"[eval] child {task} exited {rc} (attempt {attempt + 1}); respawning")
        if rc_all in (75, 3):
            break
        if len(C.read_jsonl(out / f"episodes_{task}.jsonl")) < args.episodes:
            print(f"[eval] {task}: still incomplete after the respawns")
            rc_all = rc_all or 1
    summarize_dir(out)
    return rc_all


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=("vla", "nominal", "safe"), required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model-name", default=None, help="label in the rows (default: the out dir name)")
    ap.add_argument("--out", required=True, help="absolute result directory")
    ap.add_argument("--tasks", nargs="+", default=list(C.TASKS))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=C.SEED_BASE)
    ap.add_argument("--chunk", type=int, default=32, help="actions executed per inference (HoloBrain default 32)")
    ap.add_argument("--cap-steps-per-action", type=float, default=150.0,
                    help="VLA hang guard: physics cap = this x the task's action limit")
    ap.add_argument("--planner-cap-s", type=float, default=60.0, help="physics cap of a planner episode (s)")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--top-extra", type=float, default=0.0)
    ap.add_argument("--video-every", type=int, default=10)
    ap.add_argument("--scene", choices=("obstacle", "standard"), default="obstacle",
                    help="standard = plain RoboTwin layout, no stretch and no box (pipeline sanity check, vla only)")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.out).is_absolute():
        raise SystemExit("--out must be absolute (Env chdirs to the RoboTwin root)")
    if args.agent == "vla" and not args.model_dir:
        raise SystemExit("--agent vla needs --model-dir")
    if args.scene == "standard" and args.agent != "vla":
        raise SystemExit("--scene standard is a sanity check of the policy pipeline (--agent vla)")
    if args.model_dir and not Path(args.model_dir).is_absolute():
        raise SystemExit("--model-dir must be absolute")
    args.model_name = args.model_name or Path(args.out).name
    bad = [t for t in args.tasks if t not in C.TASKS]
    if bad:
        raise SystemExit(f"unknown tasks {bad}")
    if args.child:
        return child(args)
    parent_argv = list(argv)
    if "--tasks" in parent_argv:                 # the child gets its own --tasks
        i = parent_argv.index("--tasks")
        j = i + 1
        while j < len(parent_argv) and not parent_argv[j].startswith("--"):
            j += 1
        parent_argv = parent_argv[:i] + parent_argv[j:]
    return parent(args, parent_argv)


if __name__ == "__main__":
    sys.exit(main())
