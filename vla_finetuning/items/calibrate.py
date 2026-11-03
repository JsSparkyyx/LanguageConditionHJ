"""Calibration of the careless knobs for the constraint-following fine-tune (constraint.py; Yuxuan 2026-11-03): the
careless motion must still FINISH the task while harming the item in its way.

T6 (clean_pour): the bowls closer (rim to rim GAPS) x the careless aim (the target's outer radius + DELTA, toward the
aimed-at bowl); per run the share of beads in the target and the beads in each other bowl. Want, careless: most beads
in the target AND beads in the aimed-at bowl, none in the other one; safe (aimed at the centre, low, slow): the
target filled, both other bowls clean.

T7 (gap_place): the careless slide offset (OFFSETS); per run where the block ended (dy from the slot centre along the
flanker axis, the scene's clearance) and whether the flanker it aims at was disturbed (its stay / upright rule
broken), so any success window (clearance + tol) can be scored afterwards.

    scripts/lang/in_container.sh python -m vla_finetuning.items.calibrate --task clean_pour \
        --out /abs/vla_finetuning/results_constraint/calib --slot 0 --slots 3
    python -m vla_finetuning.items.calibrate --summary --out ...     # numpy only: the login node is fine

Runs: every (cell, mode, seed) of the grid, the same seeds in every cell (CALIB_SEED_BASE), slot k of n taking every
n-th; rows in <out>/<task>/rows_s<k>.jsonl (resumable: done keys are skipped). Exit 75 at the stage deadline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .. import common as C

N_SEEDS = {"clean_pour": 8, "gap_place": 10}
T6_GAPS = {"g10": (0.010, 0.020), "g20": (0.020, 0.035)}
T6_DELTAS = (-0.03, -0.02, -0.01, 0.0)
# round 2 (11-03: up to the rim the stream still lands in the target, 85-95%; the bowls nearly touching and the
# aim past the rim, so the stream splits between the target and the aimed-at bowl)
T6_GAPS2 = {"g03": (0.003, 0.008), "g10": (0.010, 0.020)}
T6_DELTAS2 = (0.01, 0.02, 0.03, 0.04)
# round 3: aim 4 cm past the rim still put 84-93% in the target, 16-22 cm off (as built) ~0%: the switch lies between
# round 3 (11-03): per scene the stream lands all in the target or all in the aimed-at bowl (its landing spot varies
# by several cm between scenes, the stream is narrow): a high, fast careless pour scatters the beads, aimed just off
# the centre toward the aimed-at bowl, the bowls nearly touching
T6_ROUND3 = [("hf_d-10", -0.01, 0.15, 3.0), ("hf_d+0", 0.0, 0.15, 3.0), ("hf_d+10", 0.01, 0.15, 3.0),
             ("hf20_d+0", 0.0, 0.20, 3.0)]
T7_OFFSETS = (0.020, 0.025, 0.030, 0.035)
T7_TOLS = (0.005, 0.010, 0.015, 0.020)


def grid(task: str, rnd: int = 1) -> list[dict]:
    from .constraint import CALIB_SEED_BASE, seed_of

    seeds = [seed_of(CALIB_SEED_BASE, task, k) for k in range(N_SEEDS[task])]
    cells = []
    if task == "clean_pour":
        if rnd == 3:
            return [{**c, "seed": s, "key": f"{c['cell']}|{s}"} for s in seeds for c in
                    ({"cell": f"g03_{name}", "mode": "careless",
                      "knobs": {"t6_gap": T6_GAPS2["g03"], "t6_aim_delta": (d, d), "t6_pour_height": h,
                                "t6_tilt_rate": tr}} for name, d, h, tr in T6_ROUND3)]
        gaps, deltas = {1: (T6_GAPS, T6_DELTAS), 2: (T6_GAPS2, T6_DELTAS2)}[rnd]
        for g, gap in gaps.items():
            for d in deltas:
                cells.append({"cell": f"{g}_d{int(round(d * 1000)):+d}", "mode": "careless",
                              "knobs": {"t6_gap": gap, "t6_aim_delta": (d, d)}})
            if rnd == 1 or (rnd == 2 and g not in T6_GAPS):
                cells.append({"cell": f"{g}_safe", "mode": "safe", "knobs": {"t6_gap": gap}})
    elif task == "gap_place":
        for o in T7_OFFSETS:
            cells.append({"cell": f"o{int(round(o * 1000))}", "mode": "careless", "knobs": {"t7_offset": (o, o)}})
        cells.append({"cell": "safe", "mode": "safe", "knobs": {}})
    else:
        raise KeyError(task)
    return [{**c, "seed": s, "key": f"{c['cell']}|{s}"} for s in seeds for c in cells]   # seed-major: cells mix early


def run_one(task: str, item: dict, out_root: Path) -> dict:
    import main.lang  # noqa: F401 - puts envs/main on main.__path__ (main.item_envs)

    from . import constraint as X
    from .collect_items import MAX_STEPS, _runner_class
    from .evaluate_items import _NoRecorder
    from .monitor import DemoMonitor
    from main.item_envs import safe_plan

    seed, mode, knobs = int(item["seed"]), item["mode"], item["knobs"]
    row = {"task": task, "key": item["key"], "cell": item["cell"], "mode": mode, "seed": seed, "knobs": knobs}
    t0 = time.time()
    row["patched"] = X.patch_task_module(task, knobs)
    C.set_global_seeds(seed)
    risk = X.careless_risk(task, seed, knobs) if mode == "careless" else None
    runner = _runner_class()(task, embodiment=C.EMBODIMENT, seed=seed, host=None, nominal_mode=mode,
                             active_rule_id=None, render="none", out_root=str(out_root / item["cell"]),
                             max_steps=MAX_STEPS[task], risk=risk)
    try:
        runner.setup()
    except Exception as exc:  # noqa: BLE001 - no scene for this seed
        with contextlib.suppress(Exception):
            runner.close()
        return {**row, "no_scene": True, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    try:
        safe_plan.require_curobo(runner.env.robot)
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            runner.close()
        return {**row, "planner_error": True, "error": f"{type(exc).__name__}: {exc}"[:300]}
    row["after_setup"] = X.after_setup(task, runner, mode, seed, knobs)
    task_obj = runner.env.task
    probe = X.SuccessProbe(task, task_obj)
    DemoMonitor(task_obj, recorder=_NoRecorder(), joint_fn=lambda: np.zeros(1), contact_fn=None,
                abort_on_touch=False).install()
    meta, err = runner.run()
    info = dict((getattr(meta, "diag", None) or {}).get("task") or getattr(task_obj, "episode_info", None) or {})
    trace = Path(runner.out_dir) / "trace.npz"
    status = X.rule_status(task, runner, trace) if trace.exists() else {}
    row.update(error=None if err is None else str(err)[:300], success=bool(meta.task_success), probe=probe.last,
               risk=getattr(meta, "risk", None), hit=info.get("hit"), safe_fail=info.get("safe_fail"),
               plan_success=bool(getattr(task_obj, "plan_success", True)),
               broken=sorted(r for r, v in status.items() if v["broken"]), wall_s=time.time() - t0)
    if task == "clean_pour":
        counts = X.final_counts(task, runner, trace) if trace.exists() else {}
        row["counts"] = counts
        others = [k for k in counts if not k.startswith("_") and k != row["hit"]]
        row["other"] = others[0] if len(others) == 1 else None
    if task == "gap_place":
        hit = row["hit"]
        row["hit_disturbed"] = any(v["broken"] for r, v in status.items() if r.endswith(f"|{hit}|-"))
        row["other_disturbed"] = any(v["broken"] for r, v in status.items()
                                     if r.split("|")[0] in ("stay", "upright") and f"|{hit}|" not in r)
    from .collect_items import _plain

    return _plain(row)


def calibrate(args) -> int:
    out = Path(args.out) / args.task
    out.mkdir(parents=True, exist_ok=True)
    done = {r["key"] for p in out.glob("rows_s*.jsonl") for r in C.read_jsonl(p)}
    items = [it for i, it in enumerate(grid(args.task, args.round)) if i % args.slots == args.slot
             and it["key"] not in done]
    rows_p = out / f"rows_s{args.slot}.jsonl"
    times, crashes = [], 0
    print(f"[calib] {args.task} slot {args.slot}/{args.slots}: {len(items)} runs to go", flush=True)
    for it in items:
        mean_t = sum(times) / len(times) if times else 180.0
        if C.deadline_left() < 1.5 * mean_t + 60:
            print("[calib] stage deadline near: continue later", flush=True)
            return 75
        try:
            row = run_one(args.task, it, out / "runner")
            crashes = 0
        except Exception as exc:  # noqa: BLE001 - logged apart (not a done key: a fixed bug reruns it)
            traceback.print_exc()
            row = {"task": args.task, "key": it["key"], "cell": it["cell"], "mode": it["mode"], "seed": it["seed"],
                   "crash": True, "error": f"{type(exc).__name__}: {exc}"[:300]}
            C.append_jsonl(out / f"crashes_s{args.slot}.jsonl", row)
            crashes += 1
            if crashes >= 3:
                print(f"[calib] 3 crashes in a row (last: {row['error']}): stopping", flush=True)
                return 3
            continue
        C.append_jsonl(rows_p, row)
        times.append(float(row.get("wall_s") or 120.0))
        print(f"[calib] {row['key']}: success {row.get('success')} probe {row.get('probe')} counts "
              f"{row.get('counts')} hit {row.get('hit')} disturbed {row.get('hit_disturbed')} "
              f"err {row.get('error')} ({row.get('wall_s', 0):.0f} s)", flush=True)
    return 0


# ------------------------------------------------------------------ summary ----
def _frac(xs) -> str:
    xs = [bool(x) for x in xs]
    return f"{sum(xs)}/{len(xs)}" if xs else "-"


def summary(out: Path) -> str:
    lines = []
    t6 = [r for p in (out / "clean_pour").glob("rows_s*.jsonl") for r in C.read_jsonl(p)]
    if t6:
        lines += ["## T6: bowls closer, careless aim near the rim", "",
                  "| cell | mode | runs | share in target, median (min-max) | share >= 0.5 | share >= 0.3 | "
                  "aimed-at bowl harmed | other bowl harmed | beads in aimed-at at the end, median | "
                  "share >= 0.5, aimed-at harmed, other clean | the same with share >= 0.3 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for cell in sorted({r["cell"] for r in t6}):
            rs = [r for r in t6 if r["cell"] == cell and not r.get("no_scene") and not r.get("crash")
                  and (r.get("probe") or {}).get("share") is not None]
            if not rs:
                continue
            sh = np.array([r["probe"]["share"] for r in rs])
            hit = [int((r.get("counts") or {}).get(r.get("hit"), 0)) for r in rs]
            # harm as the collection and the test score it: the bowl's rule broken at ANY tick (review 11-03)
            hb = np.array([f"particles_outside|beads|{r.get('hit')}" in (r.get("broken") or []) for r in rs])
            ob = np.array([bool(r.get("other")) and f"particles_outside|beads|{r['other']}" in (r.get("broken") or [])
                           for r in rs])
            lines.append(f"| {cell} | {rs[0]['mode']} | {len(rs)} | {np.median(sh):.2f} ({sh.min():.2f}-{sh.max():.2f})"
                         f" | {_frac(sh >= 0.5)} | {_frac(sh >= 0.3)} | {_frac(hb)} | {_frac(ob)} | "
                         f"{np.median(hit):.0f} | {_frac((sh >= 0.5) & hb & ~ob)} | {_frac((sh >= 0.3) & hb & ~ob)} |")
        lines.append("")
    t7 = [r for p in (out / "gap_place").glob("rows_s*.jsonl") for r in C.read_jsonl(p)]
    if t7:
        head = " | ".join(f"placed (tol {int(t * 1000)} mm)" for t in T7_TOLS)
        both = " | ".join(f"placed + disturbed ({int(t * 1000)} mm)" for t in T7_TOLS)
        lines += ["## T7: careless slide offset vs the success window (clearance + tol)", "",
                  f"| cell | mode | runs | hit flanker disturbed | other disturbed | {head} | {both} |",
                  "|---|---|---|---|---|" + "---|" * (2 * len(T7_TOLS))]
        for cell in sorted({r["cell"] for r in t7}):
            rs = [r for r in t7 if r["cell"] == cell and not r.get("no_scene") and not r.get("crash")
                  and "dy" in (r.get("probe") or {})]
            if not rs:
                continue

            def placed(r, tol):
                p = r["probe"]
                return (abs(p["dy"]) <= p["clearance"] + tol and abs(p["dx"]) <= 0.03 and p["tilt_deg"] < 10.0
                        and r.get("plan_success", True))

            dist = [bool(r.get("hit_disturbed")) for r in rs]
            pl = [_frac([placed(r, t) for r in rs]) for t in T7_TOLS]
            pb = [_frac([placed(r, t) and d for r, d in zip(rs, dist)]) for t in T7_TOLS]
            lines.append(f"| {cell} | {rs[0]['mode']} | {len(rs)} | {_frac(dist)} | "
                         f"{_frac([r.get('other_disturbed') for r in rs])} | {' | '.join(pl)} | {' | '.join(pb)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["clean_pour", "gap_place"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--round", type=int, default=1, help="T6: the grid of round 1 or 2")
    args = ap.parse_args(argv)
    if args.summary:
        s = summary(Path(args.out))
        (Path(args.out) / "summary.md").write_text(s + "\n")
        print(s)
        return 0
    if not Path(args.out).is_absolute():
        raise SystemExit("--out must be absolute (Env chdirs to the RoboTwin root)")
    if not args.task:
        raise SystemExit("--task is required")
    return calibrate(args)


if __name__ == "__main__":
    sys.exit(main())
