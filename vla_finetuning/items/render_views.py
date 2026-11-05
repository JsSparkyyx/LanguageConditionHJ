"""The three camera views HoloBrain reads (head_camera, left_camera, right_camera: the ur5-wsg RoboTwin cameras),
rendered at the start of one scene per item task, for slides (Yuxuan 2026-11-03). Ray traced on an A40, as in the
demos.

    scripts/lang/in_container.sh python -m vla_finetuning.items.render_views --out /abs/imgs [--cam D435|Large_D435]

<out>/<task>_<camera>[_<cam type>].png, plus <task>_observer.png (the overview camera of the demo videos) and
<task>_views.png (the three side by side). D435 = 320 x 240, what the model sees; Large_D435 = 640 x 480, the same
view (same field of view) at twice the resolution. The scenes: the constraint experiment's smoke scenes (T1 host
place_container_plate) with its careless knobs; T6 as built.

Two robots in one task, every joint at 0 (Yuxuan 2026-11-05: the agent view of each robot, joint state 0):

    ... render_views --out /abs/imgs/zero_qpos --tasks selective_avoid --embodiments ur5-wsg franka-panda \
        --zero-qpos --cams head_camera

With --embodiments the files are <task>_<camera>_<embodiment>[_q0][_640x480].png (one scene per robot: the same seed,
each robot's own head camera from its RoboTwin config), plus <task>_<camera>_robots[_q0].png with the robots side by
side. --zero-qpos sets every joint of both arm articulations (arm and gripper) to 0 after the scene is built, with
the drive targets at 0 too; nothing is simulated after that, so the render shows the joints exactly at 0 even where
0 lies outside a joint's limits (those joints are printed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .. import common as C

SCENES = [("selective_avoid", "place_container_plate", 509000), ("push_zone", None, 519000),
          ("tray_transport", None, 529000), ("clean_pour", None, 980000), ("gap_place", None, 549000)]
CAMS = ("head_camera", "left_camera", "right_camera")


def zero_joints(task_env) -> None:
    """Every joint of the arm articulations (arm and gripper) to 0, velocities and drive targets too; prints the
    joints whose limits exclude 0 and the qpos read back."""
    rob = task_env.robot
    ents = [rob.left_entity] if rob.left_entity is rob.right_entity else [rob.left_entity, rob.right_entity]
    for ent in ents:
        n = ent.dof
        ent.set_qpos(np.zeros(n, dtype=np.float32))
        ent.set_qvel(np.zeros(n, dtype=np.float32))
        outside = []
        for j in ent.get_active_joints():
            j.set_drive_target(0.0)
            lo, hi = np.asarray(j.get_limits(), dtype=float).reshape(-1, 2)[0]
            if not lo <= 0.0 <= hi:
                outside.append(f"{j.get_name()} [{lo:.3f}, {hi:.3f}]")
        q = np.asarray(ent.get_qpos(), dtype=float)
        print(f"[render_views] {ent.get_name()}: {n} joints set to 0, max |qpos| read back {np.abs(q).max():.2e}"
              + (f"; 0 outside the limits of {', '.join(outside)}" if outside else ""), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="D435", choices=["D435", "Large_D435"])
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--embodiments", nargs="*", default=None, help="robots to render (default: C.EMBODIMENT only)")
    ap.add_argument("--zero-qpos", action="store_true", help="every arm joint at 0 before the render")
    ap.add_argument("--cams", nargs="*", default=list(CAMS), choices=list(CAMS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import main.lang  # noqa: F401
    from main.envs import env as E
    from main.envs.record import observer_rgb
    from PIL import Image

    from . import constraint as X
    from .collect_items import _runner_class

    orig = E._task_args

    def task_args(*a, **k):           # every camera at the requested type (same field of view, other resolution)
        args_ = orig(*a, **k)
        args_["camera"]["head_camera_type"] = args.cam
        args_["camera"]["wrist_camera_type"] = args.cam
        return args_

    E._task_args = task_args
    suffix = "" if args.cam == "D435" else "_640x480"
    q0 = "_q0" if args.zero_qpos else ""
    for task, host, seed in SCENES:
        if args.tasks and task not in args.tasks:
            continue
        per_robot = {}
        for emb in (args.embodiments or [C.EMBODIMENT]):
            tag = f"_{emb}{q0}" if args.embodiments else q0
            X.patch_task_module(task, X.KNOBS)
            C.set_global_seeds(seed)
            runner = _runner_class()(task, embodiment=emb, seed=seed, host=host, nominal_mode="careless",
                                     active_rule_id=None, render="none", out_root=str(out / "_runner"),
                                     max_steps=10 ** 9, risk=X.careless_risk(task, seed, X.KNOBS))
            try:
                runner.setup()
                t = runner.env.task
                if args.zero_qpos:
                    zero_joints(t)
                t._update_render()
                obs = t.get_obs()["observation"]
                imgs = []
                for cam in args.cams:
                    rgb = np.asarray(obs[cam]["rgb"])[:, :, :3].astype(np.uint8)
                    Image.fromarray(rgb).save(out / f"{task}_{cam}{tag}{suffix}.png")
                    imgs.append(rgb)
                per_robot[emb] = imgs
                if len(imgs) > 1:
                    Image.fromarray(np.concatenate(imgs, axis=1)).save(out / f"{task}_views{tag}{suffix}.png")
                if not suffix:
                    Image.fromarray(observer_rgb(t)).save(out / f"{task}_observer{tag}.png")
                print(f"[render_views] {task} {emb} seed {seed}: {[i.shape for i in imgs]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[render_views] {task} {emb} seed {seed}: {type(exc).__name__}: {exc}", flush=True)
            finally:
                runner.close()
        if len(per_robot) > 1:      # the robots side by side, one row per camera, an 8 px white gap
            for k, cam in enumerate(args.cams):
                row = [per_robot[e][k] for e in per_robot]
                gap = np.full((row[0].shape[0], 8, 3), 255, np.uint8)
                parts = [p for img in row for p in (img, gap)][:-1]
                Image.fromarray(np.concatenate(parts, axis=1)).save(out / f"{task}_{cam}_robots{q0}{suffix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
