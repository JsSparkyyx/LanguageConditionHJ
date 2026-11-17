"""Videos of prompt B (the task + "don't touch X") from the T1 constraint test: per model, PER_CLASS videos where X was
touched ("collision") and PER_CLASS where it was not ("no_collision") (Yuxuan 2026-11-17).

Scenes are the test's own (scenes_<task>.jsonl). A scene whose B episode is already in the test's rows is picked by its
recorded outcome (the class still short first); scenes not tested yet come after. The episode is rerun with the
cameras recorded (evaluate_constraint.run_prompt, same noise seed), and the video is filed under the RERUN's outcome,
which can differ from the recorded one (cuRobo for the script, GPU nondeterminism). Frames: head | left wrist | right
wrist every 15 physics steps (real time at 250/15 fps), a caption with the model, scene and sentence, "TOUCHED <x>"
from X's first contact on, and an end card with the result.

    python -m vla_finetuning.items.video_constraint --agent model --model-dir /abs/models/<name>/model \
        --model-name <name> --results /abs/results_constraint_t1/<name> --out /abs/results_constraint_t1/videos/<name> \
        --scenes 100 --scenes-dir /abs/results_constraint_t1/scenes

Rows: <out>/videos.jsonl, one per rerun (resumable). Exit 0 = both classes full (or MAX_TRIES reruns), 75 = stage
deadline or a broken renderer (run again to continue).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from .. import common as C
from . import evaluate_constraint as E
from . import keep as K
from .instructions import T1

PROMPT = "B"
CAMS = ("head_camera", "left_camera", "right_camera")
FPS = 250 / 15


class VideoRecorder:
    """DemoMonitor's recorder: the three camera images side by side, with the physics step of each frame."""

    def __init__(self, task_obj):
        self.task = task_obj
        self.frames: list[np.ndarray] = []
        self.steps: list[int] = []

    def __call__(self, n: int = 0) -> None:
        obs = self.task.get_obs()["observation"]
        imgs = [np.asarray(obs[c]["rgb"]) for c in CAMS if c in obs]
        h = imgs[0].shape[0]
        imgs = [im if im.shape[0] == h else _resize(im, h) for im in imgs]
        self.frames.append(np.concatenate(imgs, axis=1).astype(np.uint8))
        self.steps.append(int(n))


def _resize(im: np.ndarray, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h))


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return out + ([line] if line else [])


def write_video(path: Path, frames: list, steps: list, head: str, sentence: str, touch_step, x_name: str,
                end_lines: list[str]) -> None:
    import cv2
    import imageio.v2 as imageio

    w = frames[0].shape[1]
    lines = [head] + _wrap(sentence, max(40, w // 7))
    bar = 16 * len(lines) + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.mp4")
    wr = imageio.get_writer(str(tmp), fps=FPS, codec="libx264", quality=7, macro_block_size=8)

    def caption(img, extra=None, red=False):
        top = np.full((bar, w, 3), 255, np.uint8)
        for i, s in enumerate(lines):
            cv2.putText(top, s, (6, 16 * i + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        img = np.ascontiguousarray(img)
        if extra:
            cv2.putText(img, extra, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 0, 0) if red else (0, 140, 0), 2, cv2.LINE_AA)
        return np.concatenate([top, img], axis=0)

    try:
        for fr, st in zip(frames, steps):
            t = f"t {st / 250.0:5.1f} s"
            touched = touch_step is not None and st >= touch_step
            wr.append_data(caption(fr, f"{t}  TOUCHED {x_name}" if touched else t, red=touched))
        card = np.full_like(frames[-1], 255)
        for i, s in enumerate(end_lines):
            cv2.putText(card, s, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (220, 0, 0) if i == 0 and "TOUCHED" in s else (0, 0, 0), 2, cv2.LINE_AA)
        for _ in range(int(2 * FPS)):
            wr.append_data(caption(card))
    finally:
        wr.close()
    tmp.replace(path)


def next_scene(scenes: list[dict], recorded: dict, tried: set, need: list[str]):
    for cls in need:
        for sc in scenes:
            s = int(sc["scene"])
            if s not in tried and s in recorded and recorded[s] == (cls == "collision"):
                return sc
    for sc in scenes:
        s = int(sc["scene"])
        if s not in tried and s not in recorded:
            return sc
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["model", "scripted"], default="model")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--results", required=True, help="the model's test rows (<RES>/<name>), absolute")
    ap.add_argument("--out", required=True, help="absolute")
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--scenes-dir", required=True)
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--max-tries", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--cap-steps-per-action", type=float, default=150.0)
    args = ap.parse_args(argv)
    for p in (args.out, args.results, args.scenes_dir):
        if not Path(p).is_absolute():
            raise SystemExit("--out, --results and --scenes-dir must be absolute")
    if args.agent == "model" and not args.model_dir:
        raise SystemExit("--agent model needs --model-dir")
    import main.lang  # noqa: F401
    from main.lang.locks import WriterLock

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = WriterLock(out / "lock", wait_s=200.0)
    rows_p = out / "videos.jsonl"
    scenes = E.scene_list(T1, args.scenes, Path(args.scenes_dir))
    recorded = {}
    for p in sorted(Path(args.results).glob(f"episodes_{T1}*.jsonl")):
        for r in C.read_jsonl(p):
            if r.get("prompt") == PROMPT and not r.get("crash") and r.get("x_harmed") is not None:
                recorded[int(r["scene"])] = bool(r["x_harmed"])
    policy = None
    if args.agent == "model":
        from ..evaluate import HoloBrainPolicy

        policy = HoloBrainPolicy(Path(args.model_dir), args.chunk)
    times = []
    while True:
        prev = C.read_jsonl(rows_p)
        have = Counter(r["label"] for r in prev if r.get("video"))
        need = [c for c in ("collision", "no_collision") if have[c] < args.per_class]
        if not need or len(prev) >= args.max_tries:
            print(f"[video] {args.model_name}: {dict(have)} after {len(prev)} reruns: done")
            return 0
        sc = next_scene(scenes, recorded, {int(r["scene"]) for r in prev}, need)
        if sc is None:
            print(f"[video] {args.model_name}: no scene left ({dict(have)})")
            return 0
        mean_t = sum(times) / len(times) if times else 300.0
        if C.deadline_left() < 1.5 * mean_t + 90:
            print(f"[video] stage deadline near: {args.model_name} continues later")
            return 75
        holder = {}

        def make(task_obj):
            holder["rec"] = VideoRecorder(task_obj)
            return holder["rec"]

        t0 = time.time()
        try:
            row = E.run_prompt(sc, PROMPT, args.agent, policy, out / "runner", args.cap_steps_per_action,
                               recorder_fn=make)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            row = {"scene": sc["scene"], "crash": True, "error": f"{type(exc).__name__}: {exc}"[:300]}
        row.update(model=args.model_name, recorded_x_harmed=recorded.get(int(sc["scene"])), video=None, label=None)
        rec = holder.get("rec")
        if not row.get("crash") and row.get("x_harmed") is not None and rec is not None and rec.frames:
            label = "collision" if row["x_harmed"] else "no_collision"
            row["label"] = label
            if have[label] < args.per_class:
                x = sc["x"]
                xc = (row.get("contacts") or {}).get(x) or {}
                touch = xc.get("first_step") if xc.get("touched") else None
                name = f"{label}_scene{int(sc['scene']):03d}.mp4"
                end = [f"{x}: TOUCHED (force max {xc.get('force_max', 0):.0f} N)" if row["x_harmed"]
                       else f"{x}: not touched",
                       f"task success: {row.get('success')}",
                       f"recorded run of this scene: "
                       + {None: "not tested yet", True: "touched", False: "not touched"}[row["recorded_x_harmed"]]]
                try:
                    write_video(out / name, rec.frames, rec.steps,
                                f"{args.model_name}  scene {sc['scene']}  ({sc.get('host')})  prompt B",
                                sc["prompts"][PROMPT], touch, x, end)
                    row["video"] = name
                except Exception as exc:  # noqa: BLE001
                    print(f"[video] writing failed: {exc}")
        row.pop("probe", None)
        lock.check()
        C.append_jsonl(rows_p, row)
        times.append(time.time() - t0)
        print(f"[video] {args.model_name} scene {sc['scene']}: harmed {row.get('x_harmed')} (recorded "
              f"{row['recorded_x_harmed']}) -> {row.get('video')} ({times[-1]:.0f} s)", flush=True)
        if K.is_render_error(row.get("error")):
            print("[video] renderer error: continue in a fresh process")
            return 75


if __name__ == "__main__":
    sys.exit(main())
