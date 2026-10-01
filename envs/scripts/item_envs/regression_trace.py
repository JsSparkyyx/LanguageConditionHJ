#!/usr/bin/env python
"""Phase-0 regression reference: one NOMINAL episode, exactly as eval_ckpt.py runs it.

Reproduces the --nominal path of main/train/eval_ckpt.py for a single
(task, embodiment) pair:

  * FrozenConfig() defaults, cfg.obstacle_mode = "on_path"
    (eval_ckpt.py:481,494)
  * build_models(cfg, None) -> HoloBrain encoder + per-embodiment kinematics,
    no actor/critics (eval_ckpt.py:559-560, build_models at eval_ckpt.py:115-151)
  * scene seed = episode_seed(seed_base, task_idx, emb_idx, ep)
    (eval_ckpt.py:97-99,601-603) with the same 10-attempt x 5-retry spawn
    loop so the obstacle exists (eval_ckpt.py:625-662)
  * RolloutController(cfg_ep, seed, mode="filtered", encoder=..., kin=...,
    video=None, max_steps=7500) (eval_ckpt.py:632-638; max_steps from the
    default --cutoffs 18,30 at eval_ckpt.py:445,590)
  * set_global_seeds before each construction (eval_ckpt.py:631,102-108)

Seed note: place_empty_cup is NOT in eval_ckpt.ALL_TASKS (eval_ckpt.py:83-86),
so this recon uses a single-task list -> task_idx = 0. emb_idx =
ALL_EMBODIMENTS.index(embodiment), ep = --ep (0-based, default 0), seed_base =
--seed (default 0). Derived scene seeds with --seed 0: franka-panda -> 10000,
ur5-wsg -> 30000. The seed actually used after spawn retries is recorded in
meta.json as used_seed.

h is stored in the trace in h*h_scale units (h_scale = 20; rollout.py:215-222,
config.py:35). trace.npz keeps both the raw scaled values and h_m (metres);
meta.json and the REGRESSION line report metres (SI).

Usage (cwd = /work/nvme/bifq/project/LanguageConditionHJ):
    python envs/scripts/item_envs/regression_trace.py \
        --embodiment ur5-wsg --task place_empty_cup --seed 0 --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "main" / "train" / "eval_ckpt.py").is_file():
    ROOT = Path("/work/nvme/bifq/project/LanguageConditionHJ")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # only LanguageConditionHJ — never CEHJ_safe (task spec)

from main.train.config import FrozenConfig  # config.py:19
from main.train.eval_ckpt import (  # eval_ckpt.py:82,97,102,115
    ALL_EMBODIMENTS,
    build_models,
    episode_seed,
    set_global_seeds,
)
from main.train.rollout import RolloutController  # rollout.py:49

EVAL_CUTOFFS_S = (18.0, 30.0)  # eval_ckpt.py:445 (--cutoffs default "18,30")
H_SCALE = 20.0  # FrozenConfig.h_scale (config.py:35); trace h is h*20 (rollout.py:218)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embodiment", required=True,
                   choices=["ur5-wsg", "franka-panda"])
    p.add_argument("--task", default="place_empty_cup")
    p.add_argument("--seed", type=int, default=0,
                   help="seed_base for episode_seed (see module docstring)")
    p.add_argument("--ep", type=int, default=0,
                   help="episode index for episode_seed (0-based)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # Resolve BEFORE Env creation: RoboTwin chdirs into its own root
    # (main/envs/env.py:137-142 prepare()).
    out_dir = args.out.resolve() / args.embodiment
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = args.embodiment
    cwd_at_start = str(Path.cwd())  # before prepare() chdirs into RoboTwin
    t0 = time.perf_counter()

    cfg = FrozenConfig()            # eval_ckpt.py:481 (no config.json + --nominal)
    cfg.obstacle_mode = "on_path"   # eval_ckpt.py:494
    task_idx = 0                    # single recon task list (place_empty_cup not in ALL_TASKS)
    emb_idx = ALL_EMBODIMENTS.index(emb)
    seed = episode_seed(args.seed, task_idx, emb_idx, args.ep)  # eval_ckpt.py:97-99
    cfg_ep = dataclasses.replace(cfg, task=args.task, embodiment=emb)   # eval_ckpt.py:604
    max_steps = int(max(EVAL_CUTOFFS_S) * 250.0)                        # eval_ckpt.py:590

    import torch  # noqa: F401 — CUDA sanity before any sim work (eval_ckpt.py:557)

    kind, encoder, kin_fn, model_kw = build_models(cfg, None)  # eval_ckpt.py:559-560
    if kind != "nominal":
        raise RuntimeError(f"build_models returned kind={kind!r}, expected 'nominal'")
    kin = kin_fn(emb)                                            # eval_ckpt.py:595
    t_setup = time.perf_counter() - t0
    print(f"[recon] models ready (kind={kind}); scene seed {seed} "
          f"(seed_base={args.seed} task_idx={task_idx} emb_idx={emb_idx} "
          f"ep={args.ep})")

    ro, trace, used_seed, error = None, None, seed, None
    t_roll = time.perf_counter()
    try:
        # attempt/spawn-retry loop, mirrored from eval_ckpt.py:625-662
        for attempt in range(10):
            s0 = seed + attempt * 10_000_000
            for retry in range(5):
                set_global_seeds(s0 + retry * 1000)                # eval_ckpt.py:631
                cand = RolloutController(
                    cfg_ep, s0 + retry * 1000,
                    mode="filtered",
                    encoder=encoder, kin=kin, video=None,
                    max_steps=max_steps,
                    **model_kw,
                )                                                   # eval_ckpt.py:632-638
                if cand.env.obstacle is not None:
                    ro, used_seed = cand, s0 + retry * 1000
                    break
                print(f"[recon] spawn failed (seed {s0 + retry * 1000}); re-drawing")
                cand.env.close()                                     # eval_ckpt.py:644
            if ro is None:
                continue
            trace = ro.run()
            if trace["h"]:
                break                                                # eval_ckpt.py:653-654
            print(f"[recon] {emb}/{args.task}/ep{args.ep} produced no ticks "
                  f"(task aborted); redrawing scene (attempt {attempt + 1})")
            ro = None
        if ro is None or trace is None or not trace["h"]:
            raise RuntimeError("no valid episode after 10 attempts")   # eval_ckpt.py:660-662
    except Exception as exc:
        import traceback

        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    wall_roll = time.perf_counter() - t_roll

    meta = {
        "cmd": " ".join([sys.executable] + sys.argv),
        "cwd_at_start": cwd_at_start,
        "embodiment": emb,
        "task": args.task,
        "seed_base": args.seed,
        "task_idx": task_idx,
        "emb_idx": emb_idx,
        "ep": args.ep,
        "seed": seed,
        "used_seed": used_seed,
        "h_scale": H_SCALE,
        "h_units_note": "trace stores h*h_scale; min/mean/max below are metres",
        "obstacle_mode": cfg_ep.obstacle_mode,
        "obstacle_model": cfg_ep.obstacle_model,
        "max_steps_physics": max_steps,
        "wall_time_s": wall,
        "wall_setup_s": t_setup,
        "wall_rollout_s": wall_roll,
    }
    if error is not None:
        meta["error"] = error
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[recon] FAILED: {error} (meta written to {out_dir / 'meta.json'})")
        return 1

    t = np.asarray(trace["t"], dtype=np.float64)
    h = np.asarray(trace["h"], dtype=np.float64)      # h*h_scale units (rollout.py:218)
    h_m = h / H_SCALE
    npz = {
        "t": t,
        "h_scaled": h,
        "h_m": h_m,
        "contact_force": np.asarray(trace.get("contact_force", []), dtype=np.float64),
        "contact_touch": np.asarray(trace.get("contact_touch", []), dtype=bool),
        "success": bool(trace.get("success", False)),
        "n_physics": int(trace.get("n_physics", 0)),
    }
    # Q trace only exists when the filter models ran (nominal: empty)
    if len(trace.get("V", [])):
        npz["V"] = np.asarray(trace["V"], dtype=np.float64)
        npz["V_t"] = np.asarray(trace["V_t"], dtype=np.float64)
        npz["intervened"] = np.asarray(trace["intervened"], dtype=bool)
    np.savez(out_dir / "trace.npz", **npz)

    meta.update({
        "n_ticks": int(len(t)),
        "min_h_m": float(h_m.min()),
        "mean_h_m": float(h_m.mean()),
        "max_h_m": float(h_m.max()),
        "success": bool(trace.get("success", False)),
        "n_physics": int(trace.get("n_physics", 0)),
        "mean_realized_ratio": trace.get("mean_realized_ratio"),
        "trace_npz": str(out_dir / "trace.npz"),
    })
    # planner identity (closed env objects are still readable) — the runner
    # should confirm "CuroboPlanner", not the mplib fallback
    try:
        meta["planner_class"] = type(ro.env.robot.left_planner).__name__
    except Exception:
        meta["planner_class"] = None
    try:
        meta["obstacle_xyz"] = [float(v) for v in ro.env.obstacle_xyz]
        meta["obstacle_half"] = [float(v) for v in ro.env.obstacle_half]
    except Exception:
        pass
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"REGRESSION {emb} ticks={len(t)} min_h={h_m.min():.4f} wall={wall:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
