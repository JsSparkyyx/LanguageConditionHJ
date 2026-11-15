"""Demonstrations of the sliding-tray task alone (T8 tray_slide, Yuxuan 2026-11-12): a wine glass (tall, falls at a
gentle stop) and a can (short, falls only at a hard stop) stand on a tray that one arm drags into a black frame. Three
kinds of demo per scene, each doing what its prompt says:

    T   task only               "Slide the tray into the black frame."            behaviour slow: nothing falls
    TC  task + constraint       "Slide the tray ..., and don't tip over the can."  the named item stays standing:
                                                                                     glass -> slow, can -> medium
                                                                                     (medium tips the glass, not the can)
    C   constraint only         "Don't tip over the can."                         starts from an unsafe fast drag and
                                                                                     slows down enough for the named
                                                                                     item: recover_glass (0.45 -> 0.15
                                                                                     m/s) / recover_can (0.70 -> 0.45)

The named item is the scene's own draw (tray_slide.REFERENT_P: glass or can, 50 / 50, fixed by the seed), the same in
TC and C. The behaviours are the task's (robotwin_tasks/tray_slide.py BEHAVIOURS; calibration in
envs/docs/item_envs/task_cards/T8.md).

C starts mid-drag: the whole recovery runs, and its frames before the HANDOVER are dropped, where the handover is the
first frame at which the tray moves at 90% of the first (unsafe) cruise speed; the slow-down starts right there. So a
C demo's first frame shows the arm already dragging fast, and the demo is the slow-down, the stop, the release and the
retreat.

    scripts/lang/in_container.sh python -m vla_finetuning.items.collect_tray \
        --root /abs/vla_finetuning/data/demos_tray_slide --shard s0 --seed-start 700000 --seed-step 4 \
        --seed-stop 710000 --target 25

Keep rule (every kind): no error, the drag planned (no fallback), the task succeeded (tray in the frame), nothing
flown, at least keep.MIN_FRAMES frames; T: neither item tipped (upright rules, 45 deg, measured through a 1 s settle);
TC and C: the named item did not tip (the other one may); C: a handover frame exists and the named item stands there
(tilt < 10 deg). Up to TRIES runs per kind and scene (cuRobo differs run to run). A scene is complete when all three
kinds are kept; the kept demos of an incomplete scene stay (each is a valid demo).

Prompts (100 "seen" + 100 "unseen" strings per demo, HoloBrain draws one per sample): the task sentences from
task_instruction/tray_slide.json ({A} = "the tray", RoboTwin's generator, seeded by the seed); the constraint sentences
are the named item's upright rule in the paraphrase bank's TRAINING templates and words (language.sentences "train";
the held-out template "make sure ... stays standing" never appears). TC = "<task>, and <constraint>." on every string;
C = "<Constraint>." alone.

Files (the packer names an episode <task>_<config>_seed<seed>, so each kind has its own config):
<root>/tray_slide/<shard>_T|_TC|_C/ with seed.txt, data/episode{k}.hdf5, instructions/episode{k}.json, attempts.jsonl
(kept rows; C rows carry the handover). Every run's row: <root>_logs/tray_slide/<shard>/runs.jsonl, with
settings.json, runner/<n>/ and videos/ (the first --videos kept demos of each kind, head camera).

Exit 0: --target complete scenes or seeds exhausted; 75: continue in a fresh process (--max-runs, the stage deadline,
a renderer error); 3: the planners or the code are broken.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from .. import common as C
from ..demo_io import FrameRecorder, write_episode, write_instructions, write_seed_txt, write_video
from . import constraint as X
from . import instructions as I
from . import keep as K
from .collect_items import _plain, _runner_class, diag_overrides, pin_settings
from .monitor import FRAME_EVERY, SETTLE_STEPS, DemoMonitor

TASK = "tray_slide"
KINDS = ("T", "TC", "C")
TRIES = 2
MAX_STEPS = 7500
TRAIN_SEED_BASE = 700000         # disjoint from every other demo / test range (constraint.py: 500000-609999, 9xxxxx)
EVAL_SEED_BASE = 710000
HANDOVER_FRAC = 0.9              # C: the first frame at this share of the first cruise speed
HANDOVER_TILT_MAX = 10.0         # deg: the named item must stand at the handover
TEMPLATES = Path(__file__).resolve().parent / "task_instruction" / f"{TASK}.json"
RETRY_REASONS = ("render_error", "planner_error", "code_error")
RETRY_MAX = 3
CODE_STREAK = 3
# the collection: N_SHARDS shards of SCENES // N_SHARDS complete scenes, seeds interleaved over the training range
SCENES = 100
N_SHARDS = 4
_hard_exit = False


def shards() -> list[dict]:
    """{"shard", "target", "seed_start", "seed_step", "seed_stop"} of every shard of the collection."""
    return [{"shard": f"s{i}", "target": SCENES // N_SHARDS + (1 if i < SCENES % N_SHARDS else 0),
             "seed_start": TRAIN_SEED_BASE + i, "seed_step": N_SHARDS, "seed_stop": EVAL_SEED_BASE}
            for i in range(N_SHARDS)]


def behaviour_of(kind: str, ref: str) -> str:
    """The task's named behaviour for a demo kind and the named item (glass / can)."""
    if ref not in ("glass", "can"):
        raise ValueError(f"tray_slide: named item {ref!r}")
    if kind == "T":
        return "slow"
    if kind == "TC":
        return "slow" if ref == "glass" else "medium"
    if kind == "C":
        return f"recover_{ref}"
    raise ValueError(kind)


def rule_id(ref: str) -> str:
    return f"upright|{ref}|-"


# ------------------------------------------------------------------ prompts ----
def task_instructions(seed: int, n: int = 100) -> dict:
    """{"seen": [n], "unseen": [n]} task sentences ({A} = "the tray"), RoboTwin's generator seeded by the seed."""
    templates = json.loads(TEMPLATES.read_text())
    state = random.getstate()
    random.seed(int(seed))
    try:
        out = I._generate(templates, {"{A}": "the tray"}, n)
    finally:
        random.setstate(state)
    I.check_instructions(out, heldout=True)
    return out


def constraint_sentences(runner, ref: str) -> list[str]:
    """The named item's upright rule in the paraphrase bank's training templates and words."""
    import main.lang  # noqa: F401
    from main.item_envs import language as L

    record = L.episode_record(runner)
    rule = next((r for r in record["rules"] if r["rule_id"] == rule_id(ref)), None)
    if rule is None:
        raise ValueError(f"no rule {rule_id(ref)} in {[r['rule_id'] for r in record['rules']]}")
    sents = L.sentences(rule, record, "train")
    if not sents:
        raise ValueError(f"{rule_id(ref)}: no training sentence")
    return sents


def sentence(s: str) -> str:
    s = s.strip().rstrip(".!")
    return s[:1].upper() + s[1:] + "."


def constraint_only(sents: list[str], seed: int, n: int = 100) -> dict:
    """{"seen", "unseen"}: the constraint sentences alone, each list going round a seeded shuffle of all of them."""
    rng = np.random.default_rng([int(seed), 8008])
    out = {}
    for key in I.KEYS:
        order = []
        while len(order) < n:
            order += [sents[i] for i in rng.permutation(len(sents))]
        out[key] = [sentence(s) for s in order[:n]]
    I.check_instructions(out, heldout=True)
    return out


# ------------------------------------------------------------------ one run ----
def _tilt_deg(spec) -> float:
    from main.item_envs.items import entity_of

    T = np.asarray(entity_of(spec).get_pose().to_transformation_matrix(), np.float64)
    up = T[:3, :3] @ (np.asarray(spec.up_axis_local, np.float64) / np.linalg.norm(spec.up_axis_local))
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


class TrayRecorder(FrameRecorder):
    """FrameRecorder plus, per frame, the tray's speed (m/s) and each standing item's tilt (deg)."""

    def __init__(self, task_obj, items):
        super().__init__(task_obj)
        import sapien

        self._rb = task_obj.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self._items = list(items)
        self.speed: list[float] = []
        self.tilt: list[dict] = []

    def __call__(self, n: int = 0) -> None:
        super().__call__(n)
        self.speed.append(float(np.linalg.norm(np.asarray(self._rb.get_linear_velocity(), np.float64)[:2])))
        self.tilt.append({s.item_id: round(_tilt_deg(s), 2) for s in self._items})


def handover(rec: TrayRecorder, v1: float, ref: str) -> int | None:
    """C: the index of the first frame at HANDOVER_FRAC x the first cruise speed, if the named item stands there."""
    for k, v in enumerate(rec.speed):
        if v >= HANDOVER_FRAC * v1:
            return k if rec.tilt[k].get(ref, 90.0) < HANDOVER_TILT_MAX else None
    return None


def run_demo(seed: int, kind: str, out_root: Path, max_steps: int, frames: bool = True):
    """One run of a demo kind. A failed episode never raises. -> (row, frames, scene: {"ref", "task_instr",
    "constraints"} | None); row["reason"] = "kept" or why not."""
    row = {"task": TASK, "seed": int(seed), "kind": kind}
    runner = None
    try:
        try:
            C.set_global_seeds(seed)
            runner = _runner_class()(TASK, embodiment=C.EMBODIMENT, seed=int(seed), nominal_mode="careless",
                                     render="none", out_root=str(out_root), max_steps=int(max_steps),
                                     risk={"behaviour": "slow"})
        except Exception as exc:  # noqa: BLE001 - before any scene: never the seed's
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=err[:500], detail="seeding / runner constructor")
            return row, [], None
        try:
            runner.setup()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            kind_err = K.error_reason(err)
            row.update(reason=kind_err if kind_err in RETRY_REASONS else "no_scene", error=err[:500])
            return row, [], None
        from main.item_envs import safe_plan

        try:
            safe_plan.require_curobo(runner.env.robot)
        except Exception as exc:  # noqa: BLE001
            row.update(reason="planner_error", detail=f"{type(exc).__name__}: {exc}")
            return row, [], None
        try:
            task_obj = runner.env.task
            ref = str(task_obj.referent_id)
            beh = behaviour_of(kind, ref)
            task_obj.risk_knobs = {**(getattr(task_obj, "risk_knobs", None) or {}), "behaviour": beh}
            runner.risk_params.task["behaviour"] = beh               # meta.risk records it
            row.update(ref=ref, behaviour=beh)
            standing = [s for s in runner.items if s.role == "candidate"]
            row["items"] = {s.item_id: s.category for s in standing}
            sents = constraint_sentences(runner, ref)
            task_instr = task_instructions(seed)
            if frames:
                rec = TrayRecorder(task_obj, standing)
            else:
                from .evaluate_items import _NoRecorder

                rec = _NoRecorder()
            mon = DemoMonitor(task_obj, recorder=rec, contact_fn=None, abort_on_touch=False).install()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=err[:500])
            return row, [], None
        try:
            meta, err = runner.run()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            row.update(reason=K.error_reason(err), error=f"run: {err}"[:500])
            return row, [], None
        info = dict(getattr(task_obj, "episode_info", None) or {})
        trace = Path(runner.out_dir) / "trace.npz"
        status, flown = {}, []
        try:
            if trace.exists():
                status = X.rule_status(TASK, runner, trace)
                flown = K.flown_items(trace)
            elif err is None:
                err = f"no trace.npz in {runner.out_dir}"
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            err = err or f"{type(exc).__name__}: trace: {exc}"
        plan_ok = bool(getattr(task_obj, "plan_success", True))
        broken = sorted(r for r, v in status.items() if v["broken"])
        drag = info.get("drag_profile") or {}
        row.update(error=None if err is None else str(err)[:500], task_success=bool(meta.task_success),
                   plan_success=plan_ok, plan_failures=int(meta.plan_failures), broken=broken, flown=flown,
                   drag=_plain(drag), fallback=bool(info.get("drag_fallback")),
                   design=_plain(info.get("physical_design")), risk=_plain(getattr(meta, "risk", None)),
                   n_skipped_frames=mon.n_skipped, play_steps=mon.play_steps, settle_steps=mon.settled,
                   n_steps=mon.n, duration_s=mon.n / C.PHYS_HZ, runner_dir=str(runner.out_dir))
        fr = list(mon.frames)
        if frames:
            row["max_speed"] = round(max(rec.speed), 3) if rec.speed else None
            row["peak_tilt"] = {i: round(max((t[i] for t in rec.tilt), default=0.0), 1) for i in row["items"]}
        reason = None
        if err:
            reason = K.error_reason(err)
        elif not plan_ok:
            reason = "plan_failed"
        elif row["fallback"]:
            reason = "drag_fallback"
        elif not meta.task_success:
            reason = "task_failed"
        elif flown:
            reason = f"flown:{flown[0]}"
        elif kind == "T" and broken:
            reason = f"rule:{broken[0]}"
        elif kind in ("TC", "C") and rule_id(ref) in broken:
            reason = f"rule:{rule_id(ref)}"
        if reason is None and kind == "C" and frames:
            k = handover(rec, float(drag.get("v") or 0.0), ref)
            if k is None:
                reason = "no_handover"
            else:
                row.update(handover_frame=k, handover_speed=round(rec.speed[k], 3), frames_dropped=k,
                           handover_tilt=rec.tilt[k])
                fr = fr[k:]
        if reason is None and frames and len(fr) < K.MIN_FRAMES:
            reason = "few_frames"
        row["reason"] = reason or "kept"
        return row, fr, {"ref": ref, "task_instr": task_instr, "constraints": sents}
    finally:
        if runner is not None:
            runner.close()


def demo_instructions(kind: str, sc: dict, seed: int) -> dict:
    if kind == "T":
        return sc["task_instr"]
    if kind == "TC":
        return X.demo_instructions(sc["task_instr"], [sc["constraints"]], 1.0, seed)
    return constraint_only(sc["constraints"], seed)


# ------------------------------------------------------------------ bookkeeping ----
class Config:
    """One packer config directory (<root>/tray_slide/<shard>_<kind>): its kept episodes, 0..k-1."""

    def __init__(self, d: Path):
        self.d = Path(d)
        self.kept = sorted(C.read_jsonl(self.d / "attempts.jsonl"), key=lambda r: r["episode"])
        if [r["episode"] for r in self.kept] != list(range(len(self.kept))):
            raise SystemExit(f"{self.d}: kept episode indices are not 0..{len(self.kept) - 1}")
        for sub, pat in (("data", "episode*.hdf5"), ("instructions", "episode*.json")):
            for p in sorted((self.d / sub).glob(pat)) + sorted((self.d / sub).glob("*.tmp")):
                idx = p.name.replace("episode", "").split(".")[0]
                if p.suffix == ".tmp" or not idx.isdigit() or int(idx) >= len(self.kept):
                    print(f"[collect_tray] removing orphan {p}")
                    p.unlink()
        if self.kept:
            write_seed_txt(self.d / "seed.txt", [r["seed"] for r in self.kept])

    def add(self, row: dict, frames: list, instr: dict) -> int:
        k = len(self.kept)
        write_episode(self.d / "data" / f"episode{k}.hdf5", frames)
        write_instructions(self.d / "instructions" / f"episode{k}.json", instr)
        write_seed_txt(self.d / "seed.txt", [r["seed"] for r in self.kept] + [row["seed"]])
        kept_row = {"episode": k, "seed": row["seed"], "kept": True, "n_frames": len(frames), "kind": row["kind"],
                    "ref": row.get("ref"), "behaviour": row.get("behaviour"),
                    "handover_frame": row.get("handover_frame"), "handover_speed": row.get("handover_speed"),
                    "instruction_example": instr["seen"][0]}
        C.append_jsonl(self.d / "attempts.jsonl", kept_row)
        self.kept.append(kept_row)
        return k


def settings_of(args) -> dict:
    """What a shard's directory pins: a rerun with other settings is refused."""
    import hashlib

    task_file = C.REPO / "envs" / "robotwin_tasks" / f"{TASK}.py"
    return {"task": TASK, "embodiment": C.EMBODIMENT, "design": "tray_slide_2026-11-13", "kinds": list(KINDS),
            "behaviours": {f"{k}_{r}": behaviour_of(k, r) for k in KINDS for r in ("glass", "can")},
            "task_file_md5": hashlib.md5(task_file.read_bytes()).hexdigest()[:12],
            "templates_md5": hashlib.md5(TEMPLATES.read_bytes()).hexdigest()[:12],
            "tries": TRIES, "max_steps": int(args.max_steps), "frame_every": FRAME_EVERY,
            "settle_steps": SETTLE_STEPS, "min_frames": K.MIN_FRAMES, "handover_frac": HANDOVER_FRAC,
            "handover_tilt_max": HANDOVER_TILT_MAX, "seed_start": int(args.seed_start),
            "seed_step": int(args.seed_step), "seed_stop": int(args.seed_stop)}


def scene_state(rows: list[dict]) -> dict:
    """Per seed from runs.jsonl: kinds kept, tries per kind (runs that ended for a reason of the scene), starts per
    kind. A seed is done when every kind is kept or out of tries (or a no_scene); complete when all three are kept."""
    kept, tries, starts, dead = {}, Counter(), Counter(), set()
    for r in rows:
        s = int(r["seed"])
        if r.get("started") in KINDS:
            starts[(s, r["started"])] += 1
        elif r.get("kind") in KINDS and "reason" in r:
            if r["reason"] == "kept":
                kept.setdefault(s, set()).add(r["kind"])
            elif r["reason"] not in RETRY_REASONS:
                tries[(s, r["kind"])] += 1
                if r["reason"] == "no_scene":
                    dead.add(s)
    seen = {int(r["seed"]) for r in rows if "seed" in r}

    def left(s: int) -> list[str]:
        if s in dead:
            return []
        return [k for k in KINDS if k not in kept.get(s, set()) and tries[(s, k)] < TRIES
                and starts[(s, k)] < TRIES + RETRY_MAX]

    complete = {s for s, ks in kept.items() if set(KINDS) <= ks}
    pending = {s: left(s) for s in sorted(seen) if left(s) and s not in complete}
    return {"complete": complete, "pending": pending, "seen": seen, "kept": kept,
            "n_runs": sum(1 for r in rows if "reason" in r)}


# ------------------------------------------------------------------ the loop ----
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="absolute demo root (packer input_path)")
    ap.add_argument("--shard", required=True, help="shard name (config dirs <shard>_T, _TC, _C)")
    ap.add_argument("--target", type=int, default=25, help="complete scenes (all three kinds kept)")
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-step", type=int, default=1)
    ap.add_argument("--seed-stop", type=int, required=True)
    ap.add_argument("--max-runs", type=int, default=10, help="runs (Env builds) per process, then exit 75")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--videos", type=int, default=3, help="head-camera videos of the first N kept demos per kind")
    args = ap.parse_args(argv)
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
    log_d = Path(f"{root}_logs") / TASK / args.shard
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
    configs = {k: Config(root / TASK / f"{args.shard}_{k}") for k in KINDS}
    n_vid = Counter(r.get("kind") for r in rows if r.get("video"))
    complete = set(st["complete"])
    name = f"{TASK}/{args.shard}"
    print(f"[collect_tray] {name}: {len(complete)}/{args.target} complete scenes, {st['n_runs']} runs, "
          f"{len(st['pending'])} scenes in progress", flush=True)
    n_run, code_streak, times = 0, 0, []
    n_started = sum(1 for r in rows if "started" in r)
    seeds = iter(range(args.seed_start, args.seed_stop, args.seed_step))
    queue = [(s, list(ks)) for s, ks in sorted(st["pending"].items())]

    def stop_now() -> str | None:
        if n_run + 1 > args.max_runs:
            return f"{n_run} runs in this process: continue in a fresh one"
        mean_t = sum(times) / len(times) if times else 240.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            return f"stage deadline near (mean run {mean_t:.0f} s): continue later"
        return None

    while len(complete) < args.target or queue:
        if queue:
            seed, kinds = queue.pop(0)
        else:
            why = stop_now()
            if why:
                print(f"[collect_tray] {why}")
                return 75
            seed = next((s for s in seeds if s not in st["seen"]), None)
            if seed is None:
                print(f"[collect_tray] seeds exhausted: {len(complete)}/{args.target}")
                return 0
            lock.check()
            C.append_jsonl(runs_p, {"seed": seed, "started": "scene", "at": time.strftime("%F %T")})
            st["seen"].add(seed)
            kinds = list(KINDS)
        tries = Counter()
        while kinds:
            kind = kinds[0]
            why = stop_now()
            if why:                                  # the scene's other kinds resume in the next process
                print(f"[collect_tray] {why}")
                return 75
            n_run += 1
            idx, n_started = n_started, n_started + 1
            att = log_d / "runner" / str(idx)
            if att.exists():
                shutil.rmtree(att)
            C.append_jsonl(runs_p, {"seed": seed, "started": kind, "run": idx})
            t0 = time.time()
            try:
                row, frames, sc = run_demo(seed, kind, att, args.max_steps)
            except Exception as exc:  # noqa: BLE001 - this collector's own code
                traceback.print_exc()
                err = f"{type(exc).__name__}: {exc}"
                row, frames, sc = ({"task": TASK, "seed": seed, "kind": kind, "reason": K.error_reason(err),
                                    "error": err[:500], "detail": "run_demo"}, [], None)
            reason = row.get("reason") or "error"
            row.update(run=idx, shard=args.shard, n_frames=len(frames), wall_s=time.time() - t0)
            lock.check()
            if reason == "kept":
                instr = demo_instructions(kind, sc, seed)
                k = configs[kind].add(row, frames, instr)
                row.update(episode=k, config=configs[kind].d.name, instruction_example=instr["seen"][0])
                if n_vid[kind] < args.videos:
                    vname = f"{kind}_{row.get('ref')}_seed{seed}.mp4"
                    try:
                        write_video(log_d / "videos" / vname, frames)
                        row["video"] = vname
                        n_vid[kind] += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"[collect_tray] video failed: {exc}")
                kinds.pop(0)
                st["kept"].setdefault(seed, set()).add(kind)
            elif reason == "no_scene":
                kinds = []
            elif reason not in RETRY_REASONS:
                tries[kind] += 1
                prev = sum(1 for r in C.read_jsonl(runs_p) if r.get("seed") == seed and r.get("kind") == kind
                           and "reason" in r and r["reason"] not in RETRY_REASONS)
                if prev + 1 >= TRIES:
                    kinds.pop(0)
            C.append_jsonl(runs_p, _plain(row))
            times.append(row["wall_s"])
            if set(KINDS) <= st["kept"].get(seed, set()):
                complete.add(seed)
            print(f"[collect_tray] {name} seed {seed} {kind} ({row.get('ref')}, {row.get('behaviour')}): {reason} "
                  f"(success {row.get('task_success')}, broken {row.get('broken')}, peak tilt {row.get('peak_tilt')}, "
                  f"v max {row.get('max_speed')}, handover {row.get('handover_frame')}, {row['n_frames']} frames, "
                  f"{row['wall_s']:.0f} s) -> {len(complete)}/{args.target} complete", flush=True)
            if reason == "render_error":
                _hard_exit = True
                return 75
            if reason == "planner_error":
                print(f"[collect_tray] refusing to run: {row.get('detail')}", flush=True)
                return 3
            code_streak = code_streak + 1 if reason == "code_error" else 0
            if code_streak >= CODE_STREAK:
                print(f"[collect_tray] {code_streak} code errors in a row (last: {row.get('error')})", flush=True)
                return 3
            del frames
    print(f"[collect_tray] {name}: target reached ({len(complete)} complete scenes)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--shards"]:   # for the job scripts: "name:target:start:step:stop" per shard
        print(" ".join(f"{d['shard']}:{d['target']}:{d['seed_start']}:{d['seed_step']}:{d['seed_stop']}" for d in shards()))
        sys.exit(0)
    rc = main()
    if _hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)
