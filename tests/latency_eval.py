#!/usr/bin/env python
"""Latency profile of the per-tick policy pipeline over a few eval episodes.

Measures, per control tick (25 Hz), with CUDA-synchronized wall timers:
  obs      env.get_encoder_obs  — RT render + observation assembly (sim side)
  repr     _model_inputs        — input -> representation before the heads
                                  (frozen HoloBrain encoder + PolicyEncoder)
  value    flt.q_nom            — representation -> Q(s, a_nom)  (critic)
  action   flt.actor_action     — representation -> dtheta       (actor,
                                  only on engaged ticks)
  pipeline repr + value + action — the full model cost per tick

Run on a GPU node (login nodes have no CUDA):

    python tests/latency_eval.py --run <run_dir> [--ckpt checkpoint.pt]
        [--task stack_blocks_two] [--embodiment franka-panda]
        [--episodes 5] [--warmup-episodes 1] [--budget 18]

Writes <run>/eval_full/latency_<emb>_<task>.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "tests"

from main.train.eval_ckpt import (ALL_EMBODIMENTS, ALL_TASKS, build_models,
                                  episode_seed, set_global_seeds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help="run dir holding config.json + checkpoints")
    p.add_argument("--ckpt", type=str, default="checkpoint.pt")
    p.add_argument("--task", type=str, default="stack_blocks_two")
    p.add_argument("--embodiment", type=str, default="franka-panda")
    p.add_argument("--embodiments", type=str, default=None,
                   help="comma list; overrides --embodiment (one loop each)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--warmup-episodes", type=int, default=1,
                   help="unmeasured episodes first (CUDA/cuRobo warmup)")
    p.add_argument("--budget", type=float, default=18.0,
                   help="seconds of sim time per episode")
    p.add_argument("--seed-base", type=int, default=70_000,
                   help="same scene schedule as the full eval")
    p.add_argument("--force-engage", action="store_true",
                   help="latency only: pin the filter engaged so the actor "
                        "is timed every tick")
    args = p.parse_args()

    import torch

    from main.train.config import FrozenConfig
    from main.train.rollout import RolloutController

    run_dir = Path(args.run).resolve()
    cfg = FrozenConfig.load(run_dir / "config.json")
    cfg.obstacle_mode = "on_path"
    if args.force_engage:
        # latency only: margins are metres x h_scale — a huge value keeps the
        # filter engaged so the actor is queried every tick
        cfg.filter_margin = 1e9
        cfg.filter_release_margin = 1e9

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    kind, encoder, kin_fn, model_kw = build_models(cfg, ckpt_path)
    if not model_kw:
        sys.exit("[latency] need a trained ckpt (nominal has no model to time)")

    def timed_factory(rec):
        def timed(fn, key):
            def w(*a, **k):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = fn(*a, **k)
                torch.cuda.synchronize()
                rec[key].append((time.perf_counter() - t0) * 1e3)
                return out
            return w
        return timed

    def profile_embodiment(emb: str) -> dict:
        rec = defaultdict(list)      # stage -> [ms per call]
        timed = timed_factory(rec)
        k = kin_fn(emb)

        def run_episode(seed: int, measure: bool) -> None:
            """Retries on planner abort like the full eval."""
            cfg_ep = dataclasses.replace(cfg, task=args.task, embodiment=emb)
            for attempt in range(10):
                s0 = seed + attempt * 10_000_000
                set_global_seeds(s0)
                ro = RolloutController(cfg_ep, s0, mode="filtered",
                                       encoder=encoder, kin=k, video=None,
                                       max_steps=int(args.budget * 250.0),
                                       **model_kw)
                if measure:
                    ro.env.get_encoder_obs = timed(ro.env.get_encoder_obs,
                                                   "obs")
                    # the real per-tick encoding happens in _capture_tick
                    # (rollout.py:230-240): _encode_obs + policy_enc, then
                    # cached for the hook — NOT via _model_inputs
                    ro._encode_obs = timed(ro._encode_obs, "encode")
                    ro.policy_enc.forward = timed(ro.policy_enc.forward,
                                                  "policy_enc")
                    ro.flt.q_nom = timed(ro.flt.q_nom, "value")
                    ro.flt.actor_action = timed(ro.flt.actor_action, "action")
                trace = ro.run()
                if trace["h"]:
                    print(f"[latency] {emb} ep seed={s0} "
                          f"ticks={len(trace['h'])} "
                          f"engaged={int(np.sum(trace['intervened']))}",
                          flush=True)
                    return
                print(f"[latency] {emb} no ticks (planner abort); redrawing "
                      f"(attempt {attempt + 1})", flush=True)
            raise RuntimeError(f"{emb}: no valid episode after 10 attempts")

        t_idx = ALL_TASKS.index(args.task)
        e_idx = ALL_EMBODIMENTS.index(emb)
        for ep in range(args.warmup_episodes):
            run_episode(episode_seed(args.seed_base, t_idx, e_idx, 10_000 + ep),
                        measure=False)
        for ep in range(args.episodes):
            run_episode(episode_seed(args.seed_base, t_idx, e_idx, ep),
                        measure=True)

        def stats(key):
            v = np.asarray(rec[key], dtype=np.float64)
            if len(v) == 0:
                return {"n": 0, "mean_ms": float("nan"), "std_ms": float("nan")}
            return {"n": int(len(v)), "mean_ms": float(v.mean()),
                    "std_ms": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "p50_ms": float(np.median(v))}

        out = {
            "run": str(run_dir), "ckpt": str(ckpt_path), "kind": kind,
            "task": args.task, "embodiment": emb,
            "episodes": args.episodes, "vars": vars(args),
            "stages": {kk: stats(kk) for kk in ("obs", "encode", "policy_enc",
                                                "value", "action")},
        }
        # repr = encode + policy_enc (input -> representation before heads);
        # pipeline = repr + value + action. repr/value are 1:1 per measured
        # tick, action only on engaged ticks.
        e_, pe, v, a = (out["stages"][kk] for kk in
                        ("encode", "policy_enc", "value", "action"))
        out["repr_ms"] = {
            "mean_ms": e_["mean_ms"] + pe["mean_ms"],
            "std_ms": float(np.sqrt(e_["std_ms"]**2 + pe["std_ms"]**2)),
        }
        out["pipeline_ms"] = {
            "mean_ms": out["repr_ms"]["mean_ms"] + v["mean_ms"] + a["mean_ms"],
            "std_ms": float(np.sqrt(out["repr_ms"]["std_ms"]**2
                                    + v["std_ms"]**2 + a["std_ms"]**2)),
            "note": "sum of stage means; std assumes independent stages",
        }
        out_path = (run_dir / "eval_full" / f"latency_{emb}_{args.task}.json")
        print(f"\n=== {emb} / {args.task} ===", flush=True)
        print(f"{'stage':<28s} {'n':>6s} {'mean ms':>9s} {'std ms':>8s}")
        for kk, label in (("obs", "obs render+assembly (sim side)"),
                          ("encode", "  frozen encoder + body"),
                          ("policy_enc", "  policy encoder (trunk)")):
            s = out["stages"][kk]
            print(f"{label:<28s} {s['n']:>6d} {s['mean_ms']:>9.2f} "
                  f"{s['std_ms']:>8.2f}")
        rms = out["repr_ms"]
        print(f"{'input -> representation':<28s} {'':>6s} "
              f"{rms['mean_ms']:>9.2f} {rms['std_ms']:>8.2f}")
        for kk, label in (("value", "representation -> value"),
                          ("action", "representation -> action")):
            s = out["stages"][kk]
            print(f"{label:<28s} {s['n']:>6d} {s['mean_ms']:>9.2f} "
                  f"{s['std_ms']:>8.2f}")
        pms = out["pipeline_ms"]
        print(f"{'pipeline (repr+value+action)':<28s} {'':>6s} "
              f"{pms['mean_ms']:>9.2f} {pms['std_ms']:>8.2f}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"[latency] wrote {out_path}", flush=True)
        return out

    embs = (args.embodiments.split(",") if args.embodiments
            else [args.embodiment])
    results = {e: profile_embodiment(e) for e in embs}

    if len(results) > 1:
        print("\n=== summary: pipeline ms (mean +/- std) per embodiment ===")
        for e, out in results.items():
            pms = out["pipeline_ms"]
            print(f"{e:<14s} {pms['mean_ms']:>9.2f} +/- {pms['std_ms']:.2f}")
        means = np.array([o["pipeline_ms"]["mean_ms"] for o in results.values()])
        stds = np.array([o["pipeline_ms"]["std_ms"] for o in results.values()])
        print(f"{'ALL (mean of embs)':<14s} {means.mean():>9.2f} +/- "
              f"{float(np.sqrt((stds**2).mean())):.2f}")


if __name__ == "__main__":
    main()
