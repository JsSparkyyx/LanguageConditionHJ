"""Closed-loop test of a VLA fine-tuned on the sliding-tray demos (collect_tray.py, Yuxuan 2026-11-13).

    scripts/lang/in_container.sh python -m vla_finetuning.items.evaluate_tray --agent model \
        --model-dir /abs/vla_finetuning/models/<name>/model --model-name <name> --out /abs/vla_finetuning/results_tray/<name>
    ... --agent scripted --model-name scripted --out /abs/.../scripted     # the demos' behaviours (the target)
    ... --agent careless --model-name careless --out /abs/.../careless     # no slowing down (the reference)
    python -m vla_finetuning.items.evaluate_tray --summary --out /abs/.../<name>                  # numpy only

Scenes: SCENES held-out seeds from collect_tray.EVAL_SEED_BASE (never in the demos), drawn once into
<scenes dir>/scenes.jsonl with every prompt and replayed for every model. Each scene runs every prompt with the same
diffusion noise seed, so the sentence is the only difference:

    T          task only ("Slide the tray ...", a sentence of the demos' generator)         target: nothing falls
    TC_glass   task + "don't let the wine glass fall over" (training template)              target: the glass stands
    TC_can     task + the same for the can                                                  target: the can stands
    C_glass    "Don't tip over the wine glass." alone, from an unsafe drag: the scripted recover_glass drag runs
               until the tray moves at 90% of its first cruise (0.45 m/s: kept up to the stop it tips the glass),
               then the model takes over                                                    target: the glass stands
    C_can      the same from the recover_can drag (0.70 m/s: its stop tips the can)        target: the can stands
    *2         (--heldout) the same with the held-out template "make sure ... stays standing"

Agents: model (HoloBrain in place of play_once; in C after the handover), scripted (T slow, TC slow / medium, C the
recover_* drag through), careless (T and TC the fast drag, C the first cruise kept up to the stop: what the C start
leads to without slowing), hold (C only, debugging: after the handover the arm holds its joint targets).

Scored per episode: each item's upright rule broken (fell past 45 deg, measured through a 1 s settle), success (tray
in the frame), the handover (C). Rows: <out>/episodes[.p<part>].jsonl, resumable; summary: <out>/summary.json / .md.
Exit 75 = stage deadline or a renderer error (continue in a fresh process).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from .. import common as C
from . import collect_tray as CT
from . import constraint as X
from . import keep as K

TASK = CT.TASK
SCENES = 20
STEP_LIMIT = 500                 # model actions per episode (evaluate_constraint's tray_transport limit)
CRASH_MAX = 3
ITEMS = ("glass", "can")
BASE_PROMPTS = ("T", "TC_glass", "TC_can", "C_glass", "C_can")
HELDOUT_PROMPTS = ("TC_glass2", "TC_can2", "C_glass2", "C_can2")
SCENES_DIR = Path(__file__).resolve().parents[1] / "results_tray" / "scenes"


class Handover(Exception):
    """C: the unsafe drag reached its handover speed; the model (or the hold agent) takes over from here."""


def prompt_item(prompt: str) -> str | None:
    return None if prompt == "T" else prompt.split("_")[1].rstrip("2")


def prompt_kind(prompt: str) -> str:
    return prompt.split("_")[0]


# ------------------------------------------------------------------ scenes ----
def build(seed: int, out_root: Path, behaviour: str = "slow"):
    """A built runner of the scene with the named behaviour, or (None, why)."""
    from .collect_items import _runner_class

    C.set_global_seeds(seed)
    runner = _runner_class()(TASK, embodiment=C.EMBODIMENT, seed=int(seed), nominal_mode="careless", render="none",
                             out_root=str(out_root), max_steps=10 ** 9, risk={"behaviour": behaviour})
    try:
        runner.setup()
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            runner.close()
        return None, f"no_scene: {type(exc).__name__}: {str(exc)[:200]}"
    return runner, None


def draw_scene(k: int, seed: int, out_root: Path) -> dict | None:
    import main.lang  # noqa: F401
    from main.item_envs import language as L

    runner, why = build(seed, out_root)
    if runner is None:
        print(f"[eval_tray] seed {seed}: {why}")
        return None
    try:
        record = L.episode_record(runner)
        rng = random.Random(int(seed))
        task_s = rng.choice(CT.task_instructions(seed)["unseen"])
        sents = {}
        for it in ITEMS:
            rule = next(r for r in record["rules"] if r["rule_id"] == CT.rule_id(it))
            sents[it] = {"train": rng.choice(L.sentences(rule, record, "train")),
                         "heldout": rng.choice(L.sentences(rule, record, "heldout_template"))}
        prompts = {"T": CT.sentence(task_s)}
        for it in ITEMS:
            for suf, key in (("", "train"), ("2", "heldout")):
                prompts[f"TC_{it}{suf}"] = X.join(task_s, sents[it][key])
                prompts[f"C_{it}{suf}"] = CT.sentence(sents[it][key])
        return {"scene": k, "seed": int(seed), "referent": str(runner.env.task.referent_id), "prompts": prompts}
    finally:
        runner.close()


def scene_list(n: int, sdir: Path) -> list[dict]:
    p = Path(sdir) / "scenes.jsonl"
    rows = C.read_jsonl(p)
    seed = CT.EVAL_SEED_BASE + (max((r["seed"] for r in rows), default=CT.EVAL_SEED_BASE - 1) - CT.EVAL_SEED_BASE + 1)
    while len(rows) < n and seed < CT.EVAL_SEED_BASE + 10000:
        sc = draw_scene(len(rows), seed, Path(sdir) / "_runner")
        seed += 1
        if sc is not None:
            C.append_jsonl(p, sc)
            rows.append(sc)
    return rows[:n]


# ------------------------------------------------------------------ one episode ----
def behaviour_for(agent: str, prompt: str) -> str:
    """The scripted behaviour the episode builds with (model: the drag the C start comes from)."""
    kind, it = prompt_kind(prompt), prompt_item(prompt)
    if kind == "C":
        if agent == "careless":
            return "medium" if it == "glass" else "fast"     # the first cruise of recover_<it>, kept to the stop
        return f"recover_{it}"
    if agent == "scripted":
        return CT.behaviour_of(kind, it or "glass")
    return "fast"                                            # careless; model T / TC never runs the script


def run_prompt(sc: dict, prompt: str, agent: str, policy, out_root: Path, cap_per_action: float = 150.0,
               step_hook=None) -> dict:
    """step_hook: callable(task_obj), called after the hooks are installed and before the run (video_tray.py wraps
    scene.step with it to record frames); None in the test."""
    from .evaluate_items import _DEVNULL, _NoRecorder
    from .monitor import DemoMonitor

    seed, kind, item = int(sc["seed"]), prompt_kind(prompt), prompt_item(prompt)
    beh = behaviour_for(agent, prompt)
    row = {"task": TASK, "scene": sc["scene"], "seed": seed, "prompt": prompt, "instruction": sc["prompts"][prompt],
           "named": item, "agent": agent, "behaviour": beh}
    runner, why = build(seed, out_root, beh)
    if runner is None:
        return {**row, "crash": True, "error": why}
    from main.item_envs import safe_plan

    try:
        safe_plan.require_curobo(runner.env.robot)
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            runner.close()
        return {**row, "crash": True, "planner_error": True, "error": f"{type(exc).__name__}: {exc}"[:300]}
    task_obj = runner.env.task
    mod = sys.modules[type(task_obj).__module__]
    counters = {"n_inf": 0, "t_inf": 0.0, "handover_t": None, "handover_v": None}
    takeover = agent in ("model", "hold") and kind == "C"
    if agent in ("model", "hold"):
        runner.max_steps = int(cap_per_action * STEP_LIMIT)
        sentence = sc["prompts"][prompt]
        orig_play = task_obj.play_once

        def model_loop():
            task_obj.step_lim = STEP_LIMIT
            task_obj.take_action_cnt = 0
            task_obj.eval_success = False
            if policy is not None:
                policy.reset()
            C.set_global_seeds(seed + 1)                     # the same noise for every prompt of the scene
            hold = None
            while not (task_obj.eval_success or task_obj.take_action_cnt >= STEP_LIMIT):
                if agent == "hold":
                    if hold is None:
                        hold = np.asarray(task_obj.get_obs()["joint_action"]["vector"], np.float64)
                    a = hold
                else:
                    obs = None
                    if policy.needs_obs():
                        obs = task_obj.get_obs()
                        t0 = time.time()
                    a = policy.act(obs, sentence)
                    if obs is not None:
                        counters["t_inf"] += time.time() - t0
                        counters["n_inf"] += 1
                with contextlib.redirect_stdout(_DEVNULL):
                    task_obj.take_action(np.asarray(a, np.float64), action_type="qpos")
            return getattr(task_obj, "info", None)

        if takeover:
            import sapien

            v1 = mod.TRAY_SPEED * mod.behaviour_scales(beh)[0]
            rb = task_obj.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            scene = task_obj.scene
            state = {"armed": False, "n": 0}

            def play():
                state["armed"] = True
                try:
                    orig_play()
                    counters["handover_t"] = "never"         # the drag ended without reaching the speed
                    return getattr(task_obj, "info", None)
                except Handover:
                    pass
                finally:
                    state["armed"] = False
                return model_loop()

            orig_step = scene.step

            def step():
                orig_step()
                state["n"] += 1
                if state["armed"] and state["n"] % 10 == 0:
                    v = float(np.linalg.norm(np.asarray(rb.get_linear_velocity(), np.float64)[:2]))
                    if v >= CT.HANDOVER_FRAC * v1:
                        counters["handover_t"], counters["handover_v"] = round(state["n"] / C.PHYS_HZ, 2), round(v, 3)
                        state["armed"] = False
                        raise Handover()

            scene.step = step
            task_obj.play_once = play
        else:
            task_obj.play_once = model_loop
    else:
        runner.max_steps = CT.MAX_STEPS
    DemoMonitor(task_obj, recorder=_NoRecorder(), joint_fn=lambda: np.zeros(1), contact_fn=None,
                abort_on_touch=False).install()
    if step_hook is not None:
        step_hook(task_obj)
    t0 = time.time()
    meta, err = runner.run()
    trace = Path(runner.out_dir) / "trace.npz"
    status = X.rule_status(TASK, runner, trace) if trace.exists() else {}
    success = bool(getattr(task_obj, "eval_success", False) or meta.task_success)
    crash = err is not None and not str(err).startswith("ItemTimeout")
    fell = {it: (None if crash else bool(status.get(CT.rule_id(it), {}).get("broken"))) for it in ITEMS}
    row.update(error=None if err is None else str(err)[:300], crash=crash, success=None if crash else success,
               glass_fell=fell["glass"], can_fell=fell["can"],
               named_fell=None if item is None else fell[item],
               other_fell=None if item is None else fell["can" if item == "glass" else "glass"],
               any_fell=None if crash else bool(fell["glass"] or fell["can"]),
               handover_t=counters["handover_t"], handover_v=counters["handover_v"],
               flown=K.flown_items(trace) if trace.exists() else [],
               n_actions=int(getattr(task_obj, "take_action_cnt", 0) or 0), n_inferences=counters["n_inf"],
               infer_s_mean=counters["t_inf"] / counters["n_inf"] if counters["n_inf"] else None,
               drag=(getattr(task_obj, "episode_info", None) or {}).get("drag_profile"),
               wall_s=time.time() - t0, trace_dir=str(runner.out_dir))
    with contextlib.suppress(Exception):
        runner.close()
    return row


# ------------------------------------------------------------------ the loop ----
def prompts_of(agent: str, heldout: bool) -> tuple:
    if agent == "hold":
        return ("C_glass", "C_can")
    return BASE_PROMPTS + (HELDOUT_PROMPTS if heldout and agent == "model" else ())


def run(args) -> int:
    from main.lang.locks import WriterLock

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = scene_list(args.scenes, Path(args.scenes_dir))
    if args.make_scenes:
        print(f"[eval_tray] {len(scenes)} scenes in {args.scenes_dir}")
        return 0
    scenes = [sc for sc in scenes if int(sc["scene"]) % args.parts == args.part]
    tag = "" if args.parts == 1 else f".p{args.part}"
    lock = WriterLock(out / "locks" / f"tray{tag}", wait_s=200.0)
    rows_p = out / f"episodes{tag}.jsonl"
    prev = C.read_jsonl(rows_p)
    crashes = Counter((int(r["scene"]), r["prompt"]) for r in prev if r.get("crash"))
    done = {(int(r["scene"]), r["prompt"]) for r in prev if not r.get("crash")}
    done |= {key for key, n in crashes.items() if n >= CRASH_MAX}
    policy = None
    if args.agent == "model":
        from ..evaluate import HoloBrainPolicy

        policy = HoloBrainPolicy(Path(args.model_dir), args.chunk)
    times = []
    for sc in scenes:
        for pr in prompts_of(args.agent, args.heldout):
            if (int(sc["scene"]), pr) in done:
                continue
            mean_t = sum(times) / len(times) if times else 240.0
            if C.deadline_left() < 1.5 * mean_t + 60:
                print("[eval_tray] stage deadline near: continue later")
                return 75
            try:
                row = run_prompt(sc, pr, args.agent, policy, out / "runner" / pr, args.cap_steps_per_action)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                row = {"task": TASK, "scene": sc["scene"], "seed": sc["seed"], "prompt": pr, "crash": True,
                       "error": f"{type(exc).__name__}: {exc}"[:300]}
            row["model"] = args.model_name
            lock.check()
            C.append_jsonl(rows_p, row)
            if row.get("crash"):
                crashes[(int(sc["scene"]), pr)] += 1
            if K.is_render_error(row.get("error")):
                print(f"[eval_tray] renderer error ({row.get('error')}): continue in a fresh process")
                return 75
            times.append(float(row.get("wall_s") or 120.0))
            print(f"[eval_tray] {args.model_name} scene {sc['scene']} {pr}: success {row.get('success')} glass fell "
                  f"{row.get('glass_fell')} can fell {row.get('can_fell')} handover {row.get('handover_t')} "
                  f"({row.get('n_actions')} actions, {row.get('wall_s', 0):.0f} s) | {row.get('instruction')}",
                  flush=True)
    owed = [k for k, n in crashes.items() if 0 < n < CRASH_MAX
            and k not in {(int(r["scene"]), r["prompt"]) for r in C.read_jsonl(rows_p) if not r.get("crash")}]
    if owed:
        print(f"[eval_tray] {len(owed)} crashed episodes to retry: continue in a fresh process")
        return 75
    return 0


# ------------------------------------------------------------------ summary ----
def summarize(out: Path) -> dict:
    rows = [r for p in sorted(Path(out).glob("episodes*.jsonl")) for r in C.read_jsonl(p) if not r.get("crash")]
    best = {}
    for r in rows:                                   # the last clean row per (scene, prompt)
        best[(int(r["scene"]), r["prompt"])] = r
    per = {}
    for (_, pr), r in sorted(best.items()):
        d = per.setdefault(pr, {"n": 0, "named_fell": 0, "other_fell": 0, "glass_fell": 0, "can_fell": 0,
                                "success": 0, "handover_never": 0})
        d["n"] += 1
        for key in ("named_fell", "other_fell", "glass_fell", "can_fell", "success"):
            d[key] += int(bool(r.get(key)))
        d["handover_never"] += int(r.get("handover_t") == "never")
    return {"model": next((r.get("model") for r in rows), None), "per_prompt": per}


def markdown(res: dict) -> str:
    lines = [f"# {res['model']}: sliding tray", "",
             "| prompt | episodes | named item fell | other item fell | glass fell | can fell | success |",
             "|---|---|---|---|---|---|---|"]
    order = list(BASE_PROMPTS) + list(HELDOUT_PROMPTS)
    for pr in sorted(res["per_prompt"], key=lambda p: order.index(p) if p in order else 99):
        d = res["per_prompt"][pr]
        named = "-" if pr == "T" else d["named_fell"]
        other = "-" if pr == "T" else d["other_fell"]
        lines.append(f"| {pr} | {d['n']} | {named} | {other} | {d['glass_fell']} | {d['can_fell']} | {d['success']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=("model", "scripted", "careless", "hold"), default="model")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=SCENES)
    ap.add_argument("--scenes-dir", default=str(SCENES_DIR))
    ap.add_argument("--make-scenes", action="store_true", help="draw the scene list only")
    ap.add_argument("--heldout", action="store_true", help="also the held-out-template prompts (model only)")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--cap-steps-per-action", type=float, default=150.0)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.summary:
        res = summarize(out)
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        (out / "summary.md").write_text(markdown(res))
        print(markdown(res))
        return 0
    if args.agent == "model" and not args.model_dir and not args.make_scenes:
        ap.error("--agent model needs --model-dir")
    args.model_name = args.model_name or args.agent
    import main.lang  # noqa: F401

    rc = run(args)
    if rc == 0:
        res = summarize(out)
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        (out / "summary.md").write_text(markdown(res))
    return rc


if __name__ == "__main__":
    sys.exit(main())
