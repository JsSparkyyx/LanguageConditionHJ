"""P1 multi-object path crossing (PLAN v3 §6 Phase 3, T1 side).

For each CrossSafe host task x embodiment x seed: build the CEHJ Env, attach
a RiskTickController at time_scale 1.0 (retiming inert — the nominal runs at
planner speed), spawn K = 3-5 dynamic candidates with
``spawn.spawn_candidates``, refresh the cuRobo world TABLE-ONLY (the nominal
never plans around the objects), and run the nominal ``task.play_once()``
under the runner's tick wrapper. Per episode: per-candidate minimum
robot->item distance, max contact force, touch flags (PhysX separation <= 0,
CEHJ ``obstacle_contact`` semantics), success, and spawn_rejections.

Baseline mode (``baseline=True``): NO objects are spawned and only success is
recorded — the "with all candidates removed" gate arm.

Gates (PLAN): in >= 50% of object episodes the nominal comes within 2 cm of
>= 2 candidates; in >= 30% it touches >= 1; baseline success >= 70%.

P4 hook: the first ``visibility_seeds`` seeds per (host, embodiment) also
record t=0 per-camera visibility of every candidate (``masks.item_masks``)
into ``visibility_rows.jsonl``, keyed (host, embodiment, seed, item_id).

Process model: one Env per episode; a full sweep is 5 hosts x 50 seeds x 2
modes = 500 Env inits per embodiment. eval_ckpt.py measured a pybind11
TLS-key leak that aborts a single process after ~750 episode inits, so
``run()`` with more than one embodiment re-execs ONE CHILD PROCESS per
embodiment (``python -m main.item_envs.probes.p1 ... --child``) — same
pattern as eval_ckpt.py. Rows are keyed (host, embodiment, seed, baseline)
and reruns resume pair-level; a crash-truncated final line is dropped.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from ._common import (
    KeyframeWriter,
    ProbeTimeout,
    append_row,
    done_keys,
    load_rows,
    shard_cells,
)

NAME = "p1"
KEY_FIELDS = ("host", "embodiment", "seed", "baseline")
VIS_KEY_FIELDS = ("host", "embodiment", "seed", "item_id")
NEAR_DIST_M = 0.02  # "comes within 2 cm"
DEFAULT_SEEDS = tuple(range(50))
DEFAULT_EMBODIMENTS = ("ur5-wsg", "franka-panda")
ENVS_DIR = Path(__file__).resolve().parents[3]
T1_YAML = Path(__file__).resolve().parents[1] / "configs" / "tasks" / "T1_selective_avoid.yaml"

# gate thresholds (PLAN v3 §6 Phase 3 P1 row)
GATE_NEAR2 = 0.5
GATE_TOUCH = 0.3
GATE_BASELINE_SUCCESS = 0.7


def load_t1_cfg(path=None) -> dict:
    with open(path or T1_YAML, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------


def _visibility_rows(task, host, embodiment, seed, specs, report) -> list[dict]:
    """P4 hook: t=0 per-camera visibility of every candidate."""
    from .. import masks

    placements = {p["item_id"]: p for p in report.get("placements", [])}
    try:
        visibility, _tok = masks.item_masks(task, specs, [])
    except Exception as exc:
        print(f"[p1] visibility hook failed ({host}/{embodiment}/s{seed}): {exc}")
        return []
    rows = []
    for i, spec in enumerate(specs):
        vis = [float(v) for v in visibility[i]] if visibility.size else []
        rows.append(
            {
                "host": host,
                "embodiment": embodiment,
                "seed": int(seed),
                "item_id": spec.item_id,
                "category": spec.category,
                "cls": placements.get(spec.item_id, {}).get("cls"),
                "n_cam": len(vis),
                "vis": vis,
                "vis_max": max(vis) if vis else 0.0,
            }
        )
    return rows


def _episode(host, embodiment, seed, baseline, task_cfg, vis_seed_set, max_steps, keywriter):
    """One P1 episode; returns (row, vis_rows). Never raises."""
    from .._bootstrap import ensure_main_path

    ensure_main_path()
    from main.envs.obstacle import update_curobo_world

    from .. import robot_distance
    from ..controller import RiskTickController
    from ..spawn import spawn_candidates
    from ._common import build_env, item_contacts, observer_frame, tick_wrapper

    rng = np.random.RandomState(int(seed) + 4241)
    row = {
        "host": host,
        "embodiment": embodiment,
        "seed": int(seed),
        "baseline": bool(baseline),
        "success": False,
        "error": None,
        "k": 0,
        "spawn_rejections": 0,
        "layout_retries": 0,
        "settle_rejections": 0,
        "max_layouts_hit": False,
        "plan_failures": 0,
        "n_within_2cm": 0,
        "n_touched": 0,
        "min_dist_overall": None,
        "first_touch_step": None,
        "items": [],
    }
    vis_rows: list[dict] = []
    t_start = time.perf_counter()
    env = None
    try:
        env = build_env(host, embodiment, seed)
        task = env.task
        ctrl = RiskTickController(env)  # no risky_skill_labels: uniform scale
        ctrl.attach()
        task._time_scale_safe = 1.0  # time_scale 1.0: retime is a no-op
        task._time_scale_risky = 1.0
        task.active_time_scale = 1.0

        specs, report = [], {"spawn_rejections": 0, "placements": []}
        if not baseline:
            specs, report = spawn_candidates(env, task_cfg, rng)
        update_curobo_world(env.robot)  # TABLE ONLY — nominal stays unaware

        k = len(specs)
        min_d = np.full(k, np.inf)
        max_f = np.zeros(k)
        touched = np.zeros(k, dtype=bool)
        state = {"first_touch_step": None}

        if not baseline and int(seed) in vis_seed_set:
            vis_rows = _visibility_rows(task, host, embodiment, seed, specs, report)
        if not baseline:
            keywriter.maybe_save(
                f"p1_{host}_{embodiment}_s{seed}_t0", observer_frame(task)
            )

        def on_tick(n_physics: int) -> None:
            if not specs:
                return
            d = robot_distance.robot_item_distances(env, specs)
            f, t = item_contacts(env, specs)
            np.minimum(min_d, d, out=min_d)
            np.maximum(max_f, f, out=max_f)
            if state["first_touch_step"] is None and bool((t & ~touched).any()):
                state["first_touch_step"] = int(n_physics)
            np.logical_or(touched, t, out=touched)

        with tick_wrapper(task, env.control_freq, max_steps, on_tick):
            task.play_once()
        row["success"] = bool(task.check_success())

        if not baseline:
            keywriter.maybe_save(
                f"p1_{host}_{embodiment}_s{seed}_end", observer_frame(task)
            )
        row["plan_failures"] = int(ctrl.n_plan_failures)
        if getattr(task, "plan_success", True) is False and row["plan_failures"] == 0:
            row["plan_failures"] = 1  # plan_success tripped outside plan()
        row.update(
            {
                "k": k,
                "spawn_rejections": int(report.get("spawn_rejections", 0)),
                "layout_retries": int(report.get("layout_retries", 0)),
                "settle_rejections": int(report.get("settle_rejections", 0)),
                "max_layouts_hit": bool(report.get("max_layouts_hit", False)),
                "n_within_2cm": int(np.sum(min_d <= NEAR_DIST_M)),
                "n_touched": int(np.sum(touched)),
                "min_dist_overall": float(np.min(min_d)) if k else None,
                "first_touch_step": state["first_touch_step"],
                "items": [
                    {
                        "item_id": s.item_id,
                        "category": s.category,
                        "cls": next(
                            (p.get("cls") for p in report.get("placements", [])
                             if p.get("item_id") == s.item_id),
                            None,
                        ),
                        "min_dist": float(min_d[i]),
                        "max_force": float(max_f[i]),
                        "touched": bool(touched[i]),
                    }
                    for i, s in enumerate(specs)
                ],
            }
        )
    except ProbeTimeout as exc:
        row["error"] = f"ProbeTimeout: {exc}"
        print(f"[p1] {host}/{embodiment}/s{seed} baseline={baseline}: {exc}")
    except Exception as exc:  # UnStableError, plan failures, spawn errors
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[p1] {host}/{embodiment}/s{seed} baseline={baseline}: {row['error']}")
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
# probe interface
# ---------------------------------------------------------------------------


def run(out_dir, embodiments=None, seeds=None, hosts=None, shard="0/1",
        max_steps=7500, visibility_seeds=20, baseline=True, child=False,
        **_ignored) -> dict:
    """Run the host x seed x mode grid (resumable, one child process per embodiment)."""
    out_dir = Path(out_dir)
    embodiments = list(embodiments or DEFAULT_EMBODIMENTS)
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    task_cfg = load_t1_cfg()
    hosts = list(hosts or task_cfg["host_tasks"])

    if len(embodiments) > 1 and not child:
        # TLS-key leak: ~750 Env inits abort a process; one child per embodiment.
        results = []
        for emb in embodiments:
            cmd = [
                sys.executable, "-m", "main.item_envs.probes.p1",
                "--out-dir", str(out_dir), "--embodiments", emb, "--child",
                "--seeds", *[str(s) for s in seeds],
                "--hosts", *hosts,
                "--shard", str(shard),
                "--max-steps", str(int(max_steps)),
                "--visibility-seeds", str(int(visibility_seeds)),
            ]
            if not baseline:
                cmd.append("--no-baseline")
            print(f"[p1] child: {' '.join(cmd)}")
            proc = subprocess.run(cmd, cwd=str(ENVS_DIR))
            results.append({"embodiment": emb, "returncode": proc.returncode})
        return {"probe": NAME, "children": results}

    rows_path = out_dir / f"{NAME}_rows.jsonl"
    vis_path = out_dir / "visibility_rows.jsonl"
    modes = [False, True] if baseline else [False]
    cells = [(h, s, b) for h in hosts for s in sorted(int(x) for x in seeds) for b in modes]
    cells = shard_cells(cells, shard)
    done = done_keys(rows_path, KEY_FIELDS)
    vis_done = done_keys(vis_path, VIS_KEY_FIELDS)
    vis_seed_set = set(sorted(int(x) for x in seeds)[: int(visibility_seeds)])
    keywriter = KeyframeWriter(out_dir, budget=8)

    emb = embodiments[0]
    ran, skipped = 0, 0
    for host, seed, is_baseline in cells:
        if (host, emb, seed, is_baseline) in done:
            skipped += 1
            continue
        row, vis_rows = _episode(
            host, emb, seed, is_baseline, task_cfg, vis_seed_set, max_steps, keywriter
        )
        append_row(rows_path, row)
        for vr in vis_rows:
            key = tuple(vr.get(f) for f in VIS_KEY_FIELDS)
            if key not in vis_done:
                append_row(vis_path, vr)
                vis_done.add(key)
        ran += 1
        print(
            f"[p1] {host}/{emb}/s{seed} baseline={is_baseline}: "
            f"success={row['success']} k={row['k']} near2={row['n_within_2cm']} "
            f"touched={row['n_touched']} err={row['error']}"
        )
    return {"probe": NAME, "embodiment": emb, "ran": ran, "skipped": skipped,
            "rows": str(rows_path)}


def summarize(out_dir) -> dict:
    """Gate evaluation over the rows on disk; writes <out_dir>/summary.json."""
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / f"{NAME}_rows.jsonl")
    obj = [r for r in rows if not r.get("baseline") and r.get("error") is None]
    base = [r for r in rows if r.get("baseline") and r.get("error") is None]
    errors = [r for r in rows if r.get("error") is not None]

    def rate(rs, pred):
        return float(np.mean([bool(pred(r)) for r in rs])) if rs else None

    near2_rate = rate(obj, lambda r: r.get("n_within_2cm", 0) >= 2)
    touch_rate = rate(obj, lambda r: r.get("n_touched", 0) >= 1)
    base_success = rate(base, lambda r: r.get("success"))
    gates = {
        "near2_rate": near2_rate,
        "touch_rate": touch_rate,
        "baseline_success": base_success,
        "near2_pass": near2_rate is not None and near2_rate >= GATE_NEAR2,
        "touch_pass": touch_rate is not None and touch_rate >= GATE_TOUCH,
        "baseline_pass": base_success is not None and base_success >= GATE_BASELINE_SUCCESS,
        "thresholds": {"near2": GATE_NEAR2, "touch": GATE_TOUCH,
                       "baseline_success": GATE_BASELINE_SUCCESS},
    }
    gates["pass"] = bool(
        gates["near2_pass"] and gates["touch_pass"] and gates["baseline_pass"]
    )
    per_cell = {}
    for r in rows:
        key = f"{r['host']}/{r['embodiment']}"
        c = per_cell.setdefault(key, {"n": 0, "n_obj": 0, "near2": 0, "touch": 0,
                                      "succ_obj": 0, "n_base": 0, "succ_base": 0,
                                      "errors": 0, "spawn_rejections": 0})
        c["n"] += 1
        if r.get("error") is not None:
            c["errors"] += 1
            continue
        c["spawn_rejections"] += int(r.get("spawn_rejections", 0))
        if r.get("baseline"):
            c["n_base"] += 1
            c["succ_base"] += int(bool(r.get("success")))
        else:
            c["n_obj"] += 1
            c["near2"] += int(r.get("n_within_2cm", 0) >= 2)
            c["touch"] += int(r.get("n_touched", 0) >= 1)
            c["succ_obj"] += int(bool(r.get("success")))
    summary = {
        "probe": NAME,
        "n_rows": len(rows),
        "n_object_episodes": len(obj),
        "n_baseline_episodes": len(base),
        "n_error_rows": len(errors),
        "error_kinds": sorted({str(r["error"]).split(":")[0] for r in errors}),
        **gates,
        "per_cell": per_cell,
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
    ap.add_argument("--hosts", nargs="*", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-steps", type=int, default=7500)
    ap.add_argument("--visibility-seeds", type=int, default=20)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        run(
            args.out_dir,
            embodiments=args.embodiments,
            seeds=args.seeds,
            hosts=args.hosts,
            shard=args.shard,
            max_steps=args.max_steps,
            visibility_seeds=args.visibility_seeds,
            baseline=not args.no_baseline,
            child=args.child,
        )
    print(json.dumps(summarize(args.out_dir), indent=2))
