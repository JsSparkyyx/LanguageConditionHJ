"""P3-robot push primitive through the planner (PLAN v3 §6 Phase 3 P3-robot, T3 side).

Minimal scene: ``smoke_items`` (its red ``box_0`` is the push object; the
task's own ``play_once`` is NOT run). One push per episode with the
``robotwin_tasks/_item_utils.push`` primitive, executed through the
RiskTickController-retimed planner at time_scale s ∈ {0.5, 1.0, 1.3} (the
risky push segment is retimed; approach/retreat run at safe_time_scale 0.6),
20 seeds x both embodiments. The cuRobo world is refreshed TABLE-ONLY, so
the plan passes straight through the box — that is how a push works.

box_0 spawns at x ∈ [-0.18, -0.08] (left side of the table), so the LEFT
arm pushes in +x toward the table center; commanded dist ~ U(0.08, 0.15)
keeps the final x <= 0.07, clear of box_1 (x >= 0.10). Push angle jitter
U(±15°); layout rng = RandomState(seed + 4241) (spawn convention).

Gate (PLAN): at time_scale 0.5 the box's achieved displacement along the
push direction is within 3 cm of the commanded distance in >= 80% of seeds.

P4 hook: the first ``visibility_seeds`` seeds per embodiment record t=0
per-camera visibility of box_0 (masks.item_masks) into
``visibility_rows.jsonl``, keyed (embodiment, seed, item_id).

Rows: <out_dir>/p3_robot_rows.jsonl keyed (embodiment, seed, time_scale);
resumable. 2 x 20 x 3 = 120 Env inits per process — under the ~750 pybind11
TLS-key ceiling, so no child re-exec (unlike p1).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from ._common import (
    KeyframeWriter,
    ProbeTimeout,
    append_row,
    done_keys,
    load_rows,
    shard_cells,
)

NAME = "p3_robot"
KEY_FIELDS = ("embodiment", "seed", "time_scale")
VIS_KEY_FIELDS = ("embodiment", "seed", "item_id")
SCENE_TASK = "smoke_items"
PUSH_ITEM = "box_0"
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_EMBODIMENTS = ("ur5-wsg", "franka-panda")
DEFAULT_TIME_SCALES = (0.5, 1.0, 1.3)
DEFAULT_DIST_RANGE = (0.08, 0.15)
ANGLE_JITTER_DEG = 15.0
SAFE_TIME_SCALE = 0.6
GATE_TOL_M = 0.03
GATE_TIME_SCALE = 0.5
GATE_FRAC = 0.8


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------


def _episode(embodiment, seed, time_scale, dist_range, vis_seed_set, max_steps, keywriter):
    """One push; returns (row, vis_rows). Never raises."""
    from .._bootstrap import ensure_main_path

    ensure_main_path()
    from main.envs.obstacle import update_curobo_world

    from .. import masks
    from ..controller import RiskTickController
    from ..items import entity_of
    from ..runner import _item_from_meta
    from ._common import build_env, observer_frame, tick_wrapper

    rng = np.random.RandomState(int(seed) + 4241)
    row = {
        "embodiment": embodiment,
        "seed": int(seed),
        "time_scale": float(time_scale),
        "success": False,  # "the push executed cleanly" (not a task success)
        "error": None,
        "commanded_m": None,
        "achieved_m": None,
        "abs_err_m": None,
        "within_3cm": False,
        "lateral_m": None,
        "box_tilt_deg": None,
        "plan_failures": 0,
        "v_ratio_events": 0,
        "mean_realized_ratio": None,
    }
    vis_rows: list[dict] = []
    t_start = time.perf_counter()
    env = None
    try:
        env = build_env(SCENE_TASK, embodiment, seed)
        task = env.task
        # RoboTwin's envs package is importable once the Env exists
        from envs._item_utils import execute_steps, push

        ctrl = RiskTickController(env)  # risky segments via the risky() CM
        ctrl.attach()
        task._time_scale_safe = float(SAFE_TIME_SCALE)
        task._time_scale_risky = float(time_scale)
        task.active_time_scale = float(SAFE_TIME_SCALE)
        update_curobo_world(env.robot)  # TABLE ONLY — plan goes through the box

        specs = [_item_from_meta(d) for d in getattr(task, "item_meta", [])]
        box = next(s for s in specs if s.item_id == PUSH_ITEM)

        if int(seed) in vis_seed_set:
            try:
                visibility, _tok = masks.item_masks(task, [box], [])
                vis = [float(v) for v in visibility[0]] if visibility.size else []
                vis_rows.append(
                    {
                        "embodiment": embodiment,
                        "seed": int(seed),
                        "item_id": box.item_id,
                        "category": box.category,
                        "n_cam": len(vis),
                        "vis": vis,
                        "vis_max": max(vis) if vis else 0.0,
                    }
                )
            except Exception as exc:
                print(f"[p3_robot] visibility hook failed ({embodiment}/s{seed}): {exc}")
        keywriter.maybe_save(f"p3r_{embodiment}_s{seed}_ts{time_scale}_t0", observer_frame(task))

        dist = float(rng.uniform(float(dist_range[0]), float(dist_range[1])))
        ang = float(np.deg2rad(rng.uniform(-ANGLE_JITTER_DEG, ANGLE_JITTER_DEG)))
        direction = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
        start_xy = np.asarray(entity_of(box).get_pose().p, dtype=np.float64).reshape(3)[:2].copy()

        steps = push(task, "left", box.entity, direction, dist)
        with tick_wrapper(task, env.control_freq, max_steps, lambda _n: None):
            execute_steps(task, steps)

        end_xy = np.asarray(entity_of(box).get_pose().p, dtype=np.float64).reshape(3)[:2].copy()
        delta = end_xy - start_xy
        perp = np.array([-direction[1], direction[0]])
        achieved = float(delta @ direction)
        q = np.asarray(entity_of(box).get_pose().q, dtype=np.float64).reshape(4)
        up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)  # box spawned upright: body +z

        keywriter.maybe_save(
            f"p3r_{embodiment}_s{seed}_ts{time_scale}_end", observer_frame(task)
        )
        row.update(
            {
                "success": bool(getattr(task, "plan_success", True)),
                "commanded_m": dist,
                "achieved_m": achieved,
                "abs_err_m": abs(achieved - dist),
                "within_3cm": bool(abs(achieved - dist) <= GATE_TOL_M),
                "lateral_m": float(delta @ perp),
                "box_tilt_deg": float(np.rad2deg(np.arccos(np.clip(up_z, -1.0, 1.0)))),
                "plan_failures": int(ctrl.n_plan_failures),
                "v_ratio_events": len(ctrl.v_ratio_events),
                "mean_realized_ratio": ctrl.mean_realized_ratio,
            }
        )
    except ProbeTimeout as exc:
        row["error"] = f"ProbeTimeout: {exc}"
        print(f"[p3_robot] {embodiment}/s{seed}/ts{time_scale}: {exc}")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[p3_robot] {embodiment}/s{seed}/ts{time_scale}: {row['error']}")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    row["wall_time_s"] = time.perf_counter() - t_start
    return row, vis_rows


# ---------------------------------------------------------------------------
# gates (pure)
# ---------------------------------------------------------------------------


def evaluate_gates(rows: list[dict]) -> dict:
    """PLAN P3-robot gate: >=80% within 3 cm at time_scale 0.5 (per embodiment)."""
    per_scale: dict[str, dict] = {}
    for r in rows:
        if r.get("error") is not None:
            continue
        key = f"{r['embodiment']}/ts={r['time_scale']}"
        c = per_scale.setdefault(key, {"n": 0, "within": 0, "errs": [], "plan_failures": 0})
        c["n"] += 1
        c["within"] += int(bool(r.get("within_3cm")))
        if r.get("abs_err_m") is not None:
            c["errs"].append(float(r["abs_err_m"]))
        c["plan_failures"] += int(r.get("plan_failures", 0))
    cells = {}
    for key, c in sorted(per_scale.items()):
        cells[key] = {
            "n": c["n"],
            "frac_within_3cm": (c["within"] / c["n"]) if c["n"] else None,
            "median_abs_err_m": float(np.median(c["errs"])) if c["errs"] else None,
            "plan_failures": c["plan_failures"],
        }
    gate_cells = {
        k: c for k, c in cells.items()
        if k.endswith(f"/ts={GATE_TIME_SCALE}") and c["n"]
    }
    per_emb_pass = {k: c["frac_within_3cm"] >= GATE_FRAC for k, c in gate_cells.items()}
    return {
        "cells": cells,
        "gate": {
            "time_scale": GATE_TIME_SCALE,
            "tol_m": GATE_TOL_M,
            "frac_required": GATE_FRAC,
            "per_cell_pass": per_emb_pass,
        },
        "pass": bool(per_emb_pass) and all(per_emb_pass.values()),
    }


# ---------------------------------------------------------------------------
# probe interface
# ---------------------------------------------------------------------------


def run(out_dir, embodiments=None, seeds=None, shard="0/1",
        time_scales=DEFAULT_TIME_SCALES, dist_range=DEFAULT_DIST_RANGE,
        max_steps=7500, visibility_seeds=20, **_ignored) -> dict:
    """Run the embodiment x seed x time_scale grid (resumable, single process)."""
    out_dir = Path(out_dir)
    embodiments = list(embodiments or DEFAULT_EMBODIMENTS)
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    rows_path = out_dir / f"{NAME}_rows.jsonl"
    vis_path = out_dir / "visibility_rows.jsonl"
    cells = [
        (emb, int(s), float(ts))
        for emb in embodiments
        for s in sorted(int(x) for x in seeds)
        for ts in time_scales
    ]
    cells = shard_cells(cells, shard)
    done = done_keys(rows_path, KEY_FIELDS)
    vis_done = done_keys(vis_path, VIS_KEY_FIELDS)
    vis_seed_set = set(sorted(int(x) for x in seeds)[: int(visibility_seeds)])
    keywriter = KeyframeWriter(out_dir, budget=8)

    ran, skipped = 0, 0
    for emb, seed, ts in cells:
        if (emb, seed, ts) in done:
            skipped += 1
            continue
        row, vis_rows = _episode(emb, seed, ts, dist_range, vis_seed_set, max_steps, keywriter)
        append_row(rows_path, row)
        for vr in vis_rows:
            key = tuple(vr.get(f) for f in VIS_KEY_FIELDS)
            if key not in vis_done:
                append_row(vis_path, vr)
                vis_done.add(key)
        ran += 1
        print(
            f"[p3_robot] {emb}/s{seed}/ts{ts}: achieved={row['achieved_m']} "
            f"cmd={row['commanded_m']} within3={row['within_3cm']} err={row['error']}"
        )
    return {"probe": NAME, "ran": ran, "skipped": skipped, "rows": str(rows_path)}


def summarize(out_dir) -> dict:
    """Gate evaluation over the rows on disk; writes <out_dir>/summary.json."""
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / f"{NAME}_rows.jsonl")
    errors = [r for r in rows if r.get("error") is not None]
    summary = {
        "probe": NAME,
        "n_rows": len(rows),
        "n_error_rows": len(errors),
        "error_kinds": sorted({str(r["error"]).split(":")[0] for r in errors}),
        "setup": {
            "scene": SCENE_TASK, "push_item": PUSH_ITEM, "arm": "left",
            "dist_range": list(DEFAULT_DIST_RANGE),
            "angle_jitter_deg": ANGLE_JITTER_DEG,
            "safe_time_scale": SAFE_TIME_SCALE,
        },
        **evaluate_gates(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embodiments", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--time-scales", type=float, nargs="*", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-steps", type=int, default=7500)
    ap.add_argument("--visibility-seeds", type=int, default=20)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        run(
            args.out_dir,
            embodiments=args.embodiments,
            seeds=args.seeds,
            shard=args.shard,
            time_scales=tuple(args.time_scales) if args.time_scales else DEFAULT_TIME_SCALES,
            max_steps=args.max_steps,
            visibility_seeds=args.visibility_seeds,
        )
    print(json.dumps(summarize(args.out_dir), indent=2))
