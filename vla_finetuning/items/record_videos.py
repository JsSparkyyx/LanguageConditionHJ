"""Videos of the item-scene tests (Yuxuan 11-03): per controller and task, PER_KIND (2) "success" and PER_KIND "fail"
episodes, re-run on the scenes of the controller's own test (results_items/<controller>/episodes_<task>.jsonl) and
filmed with the observer camera (the third-person view of the CrossSafe videos), one frame every FRAME_EVERY = 10
physics steps (25 fps = real time), the 1 s settle included, plus an outcome card at the end.

    scripts/lang/in_container.sh python -m vla_finetuning.items.record_videos --controller ft_items_safe_x2 \
        --tasks tray_transport --out /abs/vla_videos

Controllers: the three models (base, ft_items_safe, ft_items_safe_x2: run like evaluate_items.run_episode, the same
scene builder, instruction and scoring) and the two scripted controllers (nominal_careless, nominal_safe: run like
evaluate_nominal.run_one). success = safe success (the task done, no protected rule broken, no T1 touch, nothing
flown); fail = everything else (not done, or done but harmed). A scene can turn out differently on the re-run (cuRobo
planning is nondeterministic; the model's GPU sampler may differ slightly), so every video is labelled by the re-run's
own outcome: candidates are tried (the test's successes for "success", its harmed and its unfinished scenes in turn
for "fail", T1's hosts in turn) until PER_KIND videos of each kind exist or MAX_TRIES scenes per kind were run; a
re-run that lands in the other kind is kept there if that kind still needs one.

Output: <out>/<controller>/<task>/<kind>_<k>_<task>[_<host>]_seed<seed>_<reason>.mp4, videos.jsonl (one row per
kept video: scene, instruction, re-run outcome, the test's outcome), attempts.jsonl (every scene run, so a rerun of
the same command resumes); --index writes <out>/README.md. Runner traces go to <out>/_work. Exit 75 = stopped at
STAGE_DEADLINE.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .. import common as C
from . import instructions as I
from .instructions import ITEM_TASKS, T1

RES = Path(__file__).resolve().parents[1] / "results_items"
MODELS = ("base", "ft_items_safe", "ft_items_safe_x2")
NOMINALS = {"nominal_careless": "careless", "nominal_safe": "safe"}
CONTROLLERS = MODELS + tuple(NOMINALS)
LABEL = {"base": "HoloBrain base (no fine-tune)", "ft_items_safe": "HoloBrain fine-tuned 15k steps",
         "ft_items_safe_x2": "HoloBrain fine-tuned 30k steps (x2)",
         "nominal_careless": "careless nominal (scripted)", "nominal_safe": "safe nominal (scripted, demo expert)"}
TASK_LABEL = {"selective_avoid": "T1 selective_avoid", "push_zone": "T3 push_zone",
              "tray_transport": "T5 tray_transport", "clean_pour": "T6 clean_pour", "gap_place": "T7 gap_place"}
PER_KIND = 2
NO_SUCCESS = {"nominal_careless"}   # Yuxuan 11-03: no success videos of the careless nominal
MAX_TRIES = 5
FRAME_EVERY = 10          # physics steps per frame: 25 fps at 250 Hz
FPS = 25.0
SIZE = (640, 480)         # observer frame (w, h)
BANNER = 64               # header rows above the frame: 640 x 544 in all
CARD_S = 2.0              # outcome card length


# ------------------------------------------------------------------------------------------------ video ----
def _text(img, lines, color=(255, 255, 255), scale=0.48, y0=18, dy=20):
    import cv2

    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _wrap(s: str, n: int = 86) -> list[str]:
    out, cur = [], ""
    for w in str(s).split():
        if cur and len(cur) + 1 + len(w) > n:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return out + ([cur] if cur else [])


class ObserverVideo:
    """scene.step hook: one observer frame every FRAME_EVERY physics steps, streamed into an mp4 (tmp until finish)."""

    def __init__(self):
        self.armed = None       # (path, header lines) of the next run
        self.reset()

    def reset(self):
        self.task = self.writer = self.tmp = self.path = self.last = None
        self.header, self.n, self.frames, self._orig = [], 0, 0, None

    def arm(self, path: Path, header: list[str]):
        self.armed = (Path(path), list(header))

    def install(self, task_obj):
        """Called at the start of the runner's run(): its own scene.step wrapper then sits outside this one."""
        if self.armed is None:
            return
        self.reset()
        self.path, self.header = self.armed
        self.armed = None
        self.task = task_obj
        self._orig = task_obj.scene.step
        task_obj.scene.step = self.step
        self.grab()

    def step(self):
        self._orig()
        self.n += 1
        if self.n % FRAME_EVERY == 0:
            self.grab()

    def _compose(self, rgb, header, color=(255, 255, 255), bg=(30, 30, 30)):
        import cv2

        if rgb.shape[1] != SIZE[0] or rgb.shape[0] != SIZE[1]:
            rgb = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
        img = np.zeros((SIZE[1] + BANNER, SIZE[0], 3), np.uint8)
        img[:BANNER] = bg
        img[BANNER:] = rgb
        _text(img, header, color=color)
        return img

    def grab(self):
        from main.envs.record import observer_rgb

        try:
            rgb = np.asarray(observer_rgb(self.task))
        except Exception as exc:  # noqa: BLE001 - one bad render: skip the frame
            if self.frames == 0:
                print(f"[record] observer frame failed: {exc}")
            return
        self.last = rgb.copy()
        lines = self.header + [f"t = {self.n / 250.0:5.1f} s"]
        self._write(self._compose(rgb, lines))

    def _write(self, img):
        import imageio.v2 as imageio

        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp.mp4")
            self.writer = imageio.get_writer(str(self.tmp), fps=FPS, codec="libx264", quality=7, macro_block_size=8)
        self.writer.append_data(img)
        self.frames += 1

    def finish(self, path: Path, card: list[str], ok: bool) -> bool:
        """Outcome card (the last frame held CARD_S), close, move to <path>. -> a video was written."""
        if self.writer is None:
            self.reset()
            return False
        if self.last is not None:
            img = self._compose(self.last, card, bg=(20, 110, 30) if ok else (20, 20, 150))
            for _ in range(int(CARD_S * FPS)):
                self._write(img)
        self.writer.close()
        os.replace(self.tmp, path)
        self.reset()
        return True

    def discard(self):
        self.armed = None
        if self.writer is not None:
            try:
                self.writer.close()
            finally:
                if self.tmp is not None and self.tmp.exists():
                    self.tmp.unlink()
        self.reset()


VIDEO = ObserverVideo()


def _patch_runner():
    from .collect_items import _runner_class

    cls = _runner_class()
    if getattr(cls.run, "_record_videos", False):
        return
    orig = cls.run

    def run(self, *a, **k):
        VIDEO.install(self.env.task)
        return orig(self, *a, **k)

    run._record_videos = True
    cls.run = run


# --------------------------------------------------------------------------------------------- choosing ----
def _rr_hosts(rows):
    """T1: round-robin over the hosts, keeping each host's order."""
    by = {}
    for r in rows:
        by.setdefault(r.get("host"), []).append(r)
    out, lists = [], list(by.values())
    while any(lists):
        for L in lists:
            if L:
                out.append(L.pop(0))
    return out


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        out += [x[i] for x in (a, b) if i < len(x)]
    return out


def candidates(ctrl: str, task: str) -> dict:
    rows = [r for r in C.read_jsonl(RES / ctrl / f"episodes_{task}.jsonl") if not r.get("crash")]
    succ = _rr_hosts([r for r in rows if r.get("safe_success")])
    harmed = _rr_hosts([r for r in rows if r.get("safe_success") is False and r.get("harmed")])
    unfin = _rr_hosts([r for r in rows if r.get("safe_success") is False and not r.get("harmed")])
    return {"success": succ, "fail": _interleave(harmed, unfin)}


def reason(row: dict) -> str:
    if row.get("safe_success"):
        return "safe"
    parts = []
    if row.get("contact"):
        parts.append("touched_" + "-".join(sorted(row.get("contact_partners") or [])[:1]))
    for b in row.get("broken") or []:
        c, item = (b.split("|") + ["", ""])[:2]
        parts.append(f"{c}_{item}")
    for f in row.get("flown") or []:
        parts.append(f"flown_{f}")
    if not parts:
        if row.get("success"):
            parts.append("done_but_harmed")
        elif row.get("safe_fail"):
            parts.append(f"planner_gave_up_{row['safe_fail']}")
        elif row.get("step_lim") and (row.get("n_actions") or 0) >= row["step_lim"]:
            parts.append("out_of_actions")
        else:
            parts.append("not_done")
    s = parts[0] + (f"_and_{len(parts) - 1}_more" if len(parts) > 1 else "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)[:60]


def card_lines(row: dict) -> list[str]:
    ok = bool(row.get("safe_success"))
    head = "SAFE SUCCESS" if ok else ("DONE BUT HARMED" if row.get("success") else "FAIL: NOT DONE")
    harms = []
    if row.get("contact"):
        harms.append("touched " + ", ".join(sorted(row.get("contact_partners") or [])))
    harms += [f"broke {b}" for b in row.get("broken") or []]
    harms += [f"{f} flown" for f in row.get("flown") or []]
    extra = row.get("keep_reason") or ""
    tail = "; ".join(harms) if harms else (extra if not ok else "no rule broken, nothing touched")
    if not ok and not harms and row.get("step_lim") and (row.get("n_actions") or 0) >= row["step_lim"]:
        tail = f"ran out of actions ({row['n_actions']} of {row['step_lim']})"
    return [head] + _wrap(tail)[:2]


# ---------------------------------------------------------------------------------------------- running ----
def run_model(task, seed, host, policy, work: Path, header):
    from .evaluate_items import build_scene, run_episode

    built, why = build_scene(task, seed, host, work)     # the test built a scene from this seed
    if built is None:
        return None, why
    runner, extras, info = built
    instr = I.make_item_instructions(task, info, seed, host=host)
    sentence = random.Random(seed).choice(instr["unseen"] or instr["seen"])
    VIDEO.arm(work / "pending.mp4", header + _wrap(f'"{sentence}"')[:1])
    row = run_episode(task, seed, runner, extras, info, policy, cap_per_action=150.0)
    return row, None


def run_nominal(task, seed, host, mode, work: Path, header):
    from .evaluate_nominal import PLANNER_TRIES, run_one

    for _ in range(PLANNER_TRIES):
        VIDEO.arm(work / "pending.mp4", header + ["(scripted controller: acts on the simulator state, no sentence)"])
        row = run_one(task, seed, host, mode, work)
        if not row.get("planner_error"):
            return row, None
        VIDEO.discard()
    return row, row.get("error")


def record(args) -> int:
    ctrl, task = args.controller, args.tasks[0]
    out = Path(args.out) / ctrl / task
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.out) / "_work" / ctrl
    vids_p, att_p = out / "videos.jsonl", out / "attempts.jsonl"
    vids = C.read_jsonl(vids_p)
    tried = {(r.get("host"), int(r["seed"])) for r in C.read_jsonl(att_p)}
    need = {k: args.per_kind - sum(1 for v in vids if v["kind"] == k) for k in ("success", "fail")}
    if ctrl in NO_SUCCESS:
        need["success"] = 0
    cands = candidates(ctrl, task)
    n_tried = {k: sum(1 for r in C.read_jsonl(att_p) if r.get("for") == k) for k in ("success", "fail")}
    if all(need[k] <= 0 or not [r for r in cands[k] if (r.get("host"), int(r["seed"])) not in tried]
           or n_tried[k] >= args.max_tries for k in need):
        print(f"[record] {ctrl} {task}: done ({json.dumps(need)} still wanted)")
        return 0
    _patch_runner()
    policy = None
    if ctrl in MODELS:
        from ..evaluate import HoloBrainPolicy

        cfg = json.loads((RES / ctrl / "config.json").read_text())
        policy = HoloBrainPolicy(Path(cfg["model_dir"]), int(cfg.get("chunk", 32)))
    times = []
    for kind in ("success", "fail"):
        for r in cands[kind]:
            if need[kind] <= 0 or n_tried[kind] >= args.max_tries:
                break
            seed, host = int(r["seed"]), r.get("host")
            if (host, seed) in tried:
                continue
            mean_t = sum(times) / len(times) if times else 300.0
            if C.deadline_left() < 1.5 * mean_t + 60:
                print(f"[record] stage deadline near: {ctrl} {task} continues later")
                return 75
            header = [f"{LABEL[ctrl]} | {TASK_LABEL[task]}{' ' + host if host else ''} | seed {seed}"]
            t0 = time.time()
            try:
                if ctrl in MODELS:
                    row, err = run_model(task, seed, host, policy, work, header)
                else:
                    row, err = run_nominal(task, seed, host, NOMINALS[ctrl], work, header)
            except Exception as exc:  # noqa: BLE001 - this scene is skipped
                traceback.print_exc()
                row, err = None, f"{type(exc).__name__}: {exc}"
            times.append(time.time() - t0)
            tried.add((host, seed))
            n_tried[kind] += 1
            att = {"for": kind, "seed": seed, "host": host, "error": err, "wall_s": round(times[-1], 1)}
            if row is None or row.get("crash") or row.get("safe_success") is None:
                VIDEO.discard()
                C.append_jsonl(att_p, {**att, "kept": None, "error": err or (row or {}).get("error")})
                print(f"[record] {ctrl} {task} seed {seed}: no video ({att['error']})", flush=True)
                continue
            got = "success" if row["safe_success"] else "fail"
            keep = need[got] > 0
            name = None
            if keep:
                k = args.per_kind - need[got] + 1
                name = f"{got}_{k}_{task}{'_' + host if host else ''}_seed{seed}_{reason(row)}.mp4"
                keep = VIDEO.finish(out / name, card_lines(row), got == "success")
            else:
                VIDEO.discard()
            C.append_jsonl(att_p, {**att, "got": got, "kept": name if keep else None})
            if keep:
                need[got] -= 1
                C.append_jsonl(vids_p, {
                    "file": name, "kind": got, "controller": ctrl, "task": task, "seed": seed, "host": host,
                    "instruction": row.get("instruction"), "success": row.get("success"),
                    "safe_success": row.get("safe_success"), "harmed": row.get("harmed"), "broken": row.get("broken"),
                    "contact": row.get("contact"), "contact_partners": row.get("contact_partners"),
                    "flown": row.get("flown"), "n_actions": row.get("n_actions"), "step_lim": row.get("step_lim"),
                    "keep_reason": row.get("keep_reason"), "reason": reason(row),
                    "test_outcome": {k: r.get(k) for k in ("success", "safe_success", "harmed", "broken")}})
            print(f"[record] {ctrl} {task} seed {seed} (for {kind}): {got} {reason(row)} -> "
                  f"{name if keep else 'not kept'} ({times[-1]:.0f} s)", flush=True)
    print(f"[record] {ctrl} {task}: finished, still wanted {json.dumps(need)}")
    return 0


def index(out: Path) -> None:
    lines = ["# Item-scene test videos", "",
             "Per controller and task: two safe-success and two fail episodes, re-run on scenes of that controller's "
             "test (vla_finetuning/results_items) with the observer camera, real time (25 fps), the 1 s settle included, "
             "and an outcome card at the end. success = the task done with no protected rule broken, no T1 touch and "
             "nothing flown; fail = anything else (not done, or done but harmed). A kind with fewer than two videos had "
             "no more such scenes (or none in 5 tries); the careless nominal has fail videos only. Made by vla_finetuning/items/record_videos.py.", ""]
    for ctrl in CONTROLLERS:
        lines += [f"## {LABEL[ctrl]} (`{ctrl}`)", "", "| task | success | fail |", "|---|---|---|"]
        for t in ITEM_TASKS:
            vids = C.read_jsonl(out / ctrl / t / "videos.jsonl")
            cell = {k: "<br>".join(f"[{v['reason'] if k == 'fail' else 'seed ' + str(v['seed'])}]({ctrl}/{t}/{v['file']})"
                                   for v in vids if v["kind"] == k) or "none" for k in ("success", "fail")}
            if ctrl in NO_SUCCESS:
                cell["success"] = "not recorded"
            lines.append(f"| {TASK_LABEL[t]} | {cell['success']} | {cell['fail']} |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines))
    print(f"[record] wrote {out / 'README.md'}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=CONTROLLERS)
    ap.add_argument("--tasks", nargs="*", default=list(ITEM_TASKS), choices=ITEM_TASKS)
    ap.add_argument("--out", required=True, help="absolute (Env chdirs to the RoboTwin root)")
    ap.add_argument("--per-kind", type=int, default=PER_KIND)
    ap.add_argument("--max-tries", type=int, default=MAX_TRIES, help="scenes run per kind at most")
    ap.add_argument("--index", action="store_true", help="only write <out>/README.md")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    if not Path(args.out).is_absolute():
        raise SystemExit("--out must be absolute")
    if args.index:
        index(Path(args.out))
        return 0
    if not args.controller:
        raise SystemExit("--controller is required")
    if args.child or len(args.tasks) == 1:
        return record(args)
    rc_all = 0
    for t in args.tasks:          # one child per task: bounds RoboTwin task inits per process
        cmd = [sys.executable, "-m", "vla_finetuning.items.record_videos", "--controller", args.controller,
               "--tasks", t, "--out", args.out, "--per-kind", str(args.per_kind), "--max-tries", str(args.max_tries),
               "--child"]
        rc = subprocess.run(cmd).returncode
        if rc == 75:
            return 75
        rc_all = rc_all or rc
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
