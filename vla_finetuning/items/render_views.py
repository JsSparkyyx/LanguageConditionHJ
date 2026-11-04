"""The three camera views HoloBrain reads (head_camera, left_camera, right_camera: the ur5-wsg RoboTwin cameras),
rendered at the start of one scene per item task, for slides (Yuxuan 2026-11-03). Ray traced on an A40, as in the
demos.

    scripts/lang/in_container.sh python -m vla_finetuning.items.render_views --out /abs/imgs [--cam D435|Large_D435]

<out>/<task>_<camera>[_<cam type>].png, plus <task>_observer.png (the overview camera of the demo videos) and
<task>_views.png (the three side by side). D435 = 320 x 240, what the model sees; Large_D435 = 640 x 480, the same
view (same field of view) at twice the resolution. The scenes: the constraint experiment's smoke scenes (T1 host
place_container_plate) with its careless knobs; T6 as built.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="D435", choices=["D435", "Large_D435"])
    ap.add_argument("--tasks", nargs="*", default=None)
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
    for task, host, seed in SCENES:
        if args.tasks and task not in args.tasks:
            continue
        X.patch_task_module(task, X.KNOBS)
        C.set_global_seeds(seed)
        runner = _runner_class()(task, embodiment=C.EMBODIMENT, seed=seed, host=host, nominal_mode="careless",
                                 active_rule_id=None, render="none", out_root=str(out / "_runner"),
                                 max_steps=10 ** 9, risk=X.careless_risk(task, seed, X.KNOBS))
        try:
            runner.setup()
            t = runner.env.task
            t._update_render()
            obs = t.get_obs()["observation"]
            imgs = []
            for cam in CAMS:
                rgb = np.asarray(obs[cam]["rgb"])[:, :, :3].astype(np.uint8)
                Image.fromarray(rgb).save(out / f"{task}_{cam}{suffix}.png")
                imgs.append(rgb)
            Image.fromarray(np.concatenate(imgs, axis=1)).save(out / f"{task}_views{suffix}.png")
            if not suffix:
                Image.fromarray(observer_rgb(t)).save(out / f"{task}_observer.png")
            print(f"[render_views] {task} seed {seed}: {[i.shape for i in imgs]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[render_views] {task} seed {seed}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
