"""Videos of the sliding-tray test (Yuxuan 2026-11-19): per (scene, prompt), the fine-tuned model, the careless nominal
and the safe nominal (the scripted demo expert) on the same scene and sentence, stacked top to bottom and aligned in
time (all three count physics steps from the same start).

Each episode is a fresh run of evaluate_tray.run_prompt (the test's own code: same scene, same noise seed, the same
take_action execution of the model), so its outcome can differ from the test's row (cuRobo, GPU nondeterminism); the
end card gives both. Frames: head | left wrist | right wrist, one every 15 physics steps (real time at 250/15 fps),
none skipped, through the 1 s settle after the episode. Each row shows the time, the tray speed, its distance to the
frame and each item's tilt (red from 45 deg = fallen); in C the model row says when the model takes over.

    python -m vla_finetuning.items.video_tray --model-dir /abs/vla_finetuning/models/ft_tray/model \
        --model-name ft_tray --res /abs/vla_finetuning/results_tray --scenes 0 1 2

Writes <res>/videos/<prompt>_scene<NN>.mp4 and one row per episode to <res>/videos/videos.jsonl. Resumable (a finished
video is skipped). Exit 0 = all written, 75 = stage deadline or a broken renderer (run again to continue).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .. import common as C
from . import evaluate_tray as E
from . import keep as K

CAMS = ("head_camera", "left_camera", "right_camera")
EVERY = 15
FPS = 250 / EVERY
FALL_DEG = 45.0
AGENTS = ("model", "careless", "scripted")


class StepRecorder:
    """Wraps scene.step: the cameras and the tray state every EVERY physics steps."""

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.steps: list[int] = []
        self.status: list[dict] = []
        self.n = 0
        self.task = None

    def install(self, task_obj) -> None:
        self.task = task_obj
        inner = task_obj.scene.step

        def step():
            try:
                inner()
            finally:
                self.n += 1
                if self.n % EVERY == 0:
                    self.grab()

        task_obj.scene.step = step

    def grab(self) -> None:
        try:
            obs = self.task.get_obs()["observation"]
            imgs = [np.asarray(obs[c]["rgb"]) for c in CAMS if c in obs]
            h = imgs[0].shape[0]
            imgs = [im if im.shape[0] == h else _resize(im, h) for im in imgs]
            self.frames.append(np.concatenate(imgs, axis=1).astype(np.uint8))
            self.steps.append(self.n)
            self.status.append(self.state())
        except Exception as exc:  # noqa: BLE001  (a frame is never worth an episode)
            print(f"[video_tray] frame at step {self.n} failed: {exc}")

    def state(self) -> dict:
        import sapien

        t = self.task
        tilt = {}
        for it in getattr(t, "standing", []) or []:
            tilt[str(it["kind"])] = float(t._tilt_deg(it["entity"], [0.0, 1.0, 0.0]))
        v, d = None, None
        try:
            rb = t.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            v = float(np.linalg.norm(np.asarray(rb.get_linear_velocity(), np.float64)[:2]))
            d = float(np.linalg.norm(np.asarray(t.tray.get_pose().p, np.float64)[:2] - np.asarray(t.target_xy)[:2]))
        except Exception:  # noqa: BLE001
            pass
        return {"tilt": tilt, "v": v, "d": d}


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


def agent_label(agent: str, prompt: str, model_name: str) -> str:
    beh = E.behaviour_for(agent, prompt)
    if agent == "model":
        return f"fine-tuned model ({model_name})"
    if agent == "careless":
        return f"careless nominal ({beh} drag, ignores the sentence)"
    return f"safe nominal = demo expert ({beh} drag)"


def outcome_text(row: dict) -> str:
    if row.get("crash"):
        return f"crashed: {row.get('error')}"[:90]
    fell = [it for it in E.ITEMS if row.get(f"{it}_fell")]
    s = ("FELL: " + ", ".join(fell)) if fell else "nothing fell"
    return f"{s}; tray in frame: {'yes' if row.get('success') else 'no'}"


def row_image(rec: StepRecorder, k: int, label: str, row: dict, prompt: str, w: int, h: int) -> np.ndarray:
    import cv2

    if not rec.frames:
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.putText(img, f"{label}: no frames ({outcome_text(row)})", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img
    i = min(k, len(rec.frames) - 1)
    img = np.ascontiguousarray(rec.frames[i].copy())
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h))
    ended = k >= len(rec.frames)
    st, step = rec.status[i], rec.steps[i]
    # the label box
    cv2.rectangle(img, (0, 0), (min(w, 12 + 8 * len(label)), 24), (255, 255, 255), -1)
    cv2.putText(img, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ho = row.get("handover_t")
    if E.prompt_kind(prompt) == "C" and row.get("agent") == "model":
        if isinstance(ho, (int, float)) and step >= ho * 250:
            txt, col = "MODEL IN CONTROL", (0, 90, 220)
        else:
            txt, col = "scripted fast drag; model takes over at 90% speed", (90, 90, 90)
        cv2.putText(img, txt, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    # the status line
    parts = [f"t {step / 250.0:5.2f} s"]
    if st.get("v") is not None:
        parts.append(f"tray {st['v']:.2f} m/s")
    if st.get("d") is not None:
        parts.append(f"to frame {st['d'] * 100:4.1f} cm")
    base = "   ".join(parts)
    cv2.rectangle(img, (0, h - 26), (w, h), (255, 255, 255), -1)
    cv2.putText(img, base, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    x = 12 + int(cv2.getTextSize(base, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0])
    for it, deg in sorted(st.get("tilt", {}).items()):
        fell = deg >= FALL_DEG
        s = f"   {it} {deg:3.0f} deg" + (" FELL" if fell else "")
        cv2.putText(img, s, (x, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 0, 0) if fell else (0, 120, 0),
                    2 if fell else 1, cv2.LINE_AA)
        x += int(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0])
    if ended:
        img = (img.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
        cv2.putText(img, "episode over: " + outcome_text(row), (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return img


def write_video(path: Path, recs: dict, rows: dict, labels: dict, sc: dict, prompt: str, recorded: dict) -> None:
    import cv2
    import imageio.v2 as imageio

    have = [a for a in AGENTS if recs[a].frames]
    ref = recs[have[0]].frames[0]
    h, w = ref.shape[0], ref.shape[1]
    kind = E.prompt_kind(prompt)
    head = (f"scene {sc['scene']}  prompt {prompt}"
            + ("  (starts from the unsafe fast drag; the model takes over at 90% speed)" if kind == "C" else ""))
    lines = [head] + _wrap(f"sentence: {sc['prompts'][prompt]}", max(50, w // 8))
    bar = 18 * len(lines) + 10
    total_h = bar + 3 * h
    pad_h = (-total_h) % 16
    n = max(len(recs[a].frames) for a in AGENTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.mp4")
    wr = imageio.get_writer(str(tmp), fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    top = np.full((bar, w, 3), 255, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(top, s, (8, 18 * i + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def stack(parts):
        return np.concatenate([top] + parts + [np.full((pad_h, w, 3), 255, np.uint8)], axis=0)

    try:
        for k in range(n):
            parts = []
            for a in AGENTS:
                im = row_image(recs[a], k, labels[a], rows[a], prompt, w, h)
                im[:2] = 0          # a thin line between the rows
                parts.append(im)
            wr.append_data(stack(parts))
        card = np.full((3 * h, w, 3), 255, np.uint8)
        y = 34
        for a in AGENTS:
            r = rows[a]
            cv2.putText(card, labels[a], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            fell_named = r.get("named_fell")
            col = (220, 0, 0) if fell_named or (kind == "T" and r.get("any_fell")) else (0, 120, 0)
            cv2.putText(card, "this run: " + outcome_text(r), (30, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1,
                        cv2.LINE_AA)
            rec = recorded.get(a)
            cv2.putText(card, "test run of this scene: " + (outcome_text(rec) if rec else "none"), (30, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
            y += 92
        for _ in range(int(3 * FPS)):
            wr.append_data(stack([card[:h], card[h:2 * h], card[2 * h:]]))
    finally:
        wr.close()
    tmp.replace(path)


def recorded_rows(res: Path, model_name: str) -> dict:
    out = {}
    for a, d in (("model", model_name), ("careless", "careless"), ("scripted", "scripted")):
        for p in sorted((res / d).glob("episodes*.jsonl")):
            for r in C.read_jsonl(p):
                if not r.get("crash"):
                    out[(a, int(r["scene"]), r["prompt"])] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-name", default="ft_tray")
    ap.add_argument("--res", required=True, help="results_tray, absolute")
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prompts", nargs="+", default=list(E.BASE_PROMPTS))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--cap-steps-per-action", type=float, default=150.0)
    args = ap.parse_args(argv)
    res = Path(args.res)
    if not res.is_absolute():
        raise SystemExit("--res must be absolute")
    import main.lang  # noqa: F401

    out = res / "videos"
    out.mkdir(parents=True, exist_ok=True)
    scenes = {int(s["scene"]): s for s in E.scene_list(max(args.scenes) + 1, res / "scenes")}
    recorded = recorded_rows(res, args.model_name)
    from ..evaluate import HoloBrainPolicy

    policy = HoloBrainPolicy(Path(args.model_dir), args.chunk)
    times = []
    for s in args.scenes:
        sc = scenes[s]
        for pr in args.prompts:
            path = out / f"{pr}_scene{s:02d}.mp4"
            if path.exists():
                continue
            mean_t = sum(times) / len(times) if times else 240.0
            if C.deadline_left() < 1.5 * mean_t + 60:
                print("[video_tray] stage deadline near: continue later")
                return 75
            t0 = time.time()
            recs, rows = {}, {}
            for a in AGENTS:
                rec = StepRecorder()
                try:
                    row = E.run_prompt(sc, pr, a, policy if a == "model" else None, out / "runner" / a / pr,
                                       args.cap_steps_per_action, step_hook=rec.install)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    row = {"scene": s, "prompt": pr, "agent": a, "crash": True,
                           "error": f"{type(exc).__name__}: {exc}"[:300]}
                row.update(model=args.model_name if a == "model" else a, frames=len(rec.frames))
                recs[a], rows[a] = rec, row
                C.append_jsonl(out / "videos.jsonl", row)
                print(f"[video_tray] scene {s} {pr} {a}: {outcome_text(row)} ({len(rec.frames)} frames)", flush=True)
                if K.is_render_error(row.get("error")):
                    print("[video_tray] renderer error: continue in a fresh process")
                    return 75
            if not any(recs[a].frames for a in AGENTS):
                print(f"[video_tray] scene {s} {pr}: no frames at all, skipped")
                continue
            labels = {a: agent_label(a, pr, args.model_name) for a in AGENTS}
            rec_rows = {a: recorded.get((a, s, pr)) for a in AGENTS}
            write_video(path, recs, rows, labels, sc, pr, rec_rows)
            times.append(time.time() - t0)
            print(f"[video_tray] wrote {path.name} ({times[-1]:.0f} s)", flush=True)
    print("[video_tray] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
