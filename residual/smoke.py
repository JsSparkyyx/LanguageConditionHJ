"""Simulator smoke test of the residual package, in one process (no worker pool), on an A40:

    scripts/lang/in_container.sh python -m residual.smoke --out /abs/runs/residual/smoke

1. scan a few seeds (which objects are in the planner's way);
2. RoboTwin's own 250 Hz playback, delta = 0 (reference);
3. the 25 Hz controller, delta = 0, deviation trigger, with the per-tick check that the named object's distance and
   margin equal the runner's own (robot_item_distances over every item, rule_margins);
4. a training episode with a randomly initialised policy, fixed-rate trigger, a_nom noise on: transitions, rewards,
   observation size, how it ended;
5. an evaluation episode with a video (deterministic policy, out-of-the-way object).

Writes ``<out>/smoke.json`` and the video; exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=9_500_000)
    ap.add_argument("--n-scan", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(2)
    from .config import N_ACT, ResidualConfig
    from .episode import Episode, Job, run_episode
    from .policy import build_model
    from .scene import build_runner, make_named, scan_record
    from .state import obs_dim

    cfg = ResidualConfig()
    scratch = str(out / "scratch")
    report: dict = {"checks": {}, "episodes": {}}
    fails = []

    def check(name, ok, detail=None):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            fails.append(name)

    # 1. scan
    scans = []
    for s in range(args.seed, args.seed + args.n_scan):
        t0 = time.perf_counter()
        try:
            r = build_runner(cfg, s, scratch)
            try:
                scans.append({"seed": s, **scan_record(r), "build_s": time.perf_counter() - t0})
            finally:
                r.close()
        except Exception as exc:
            traceback.print_exc()
            scans.append({"seed": s, "error": f"{type(exc).__name__}: {exc}"})
    report["scans"] = scans
    print(json.dumps(scans, indent=1), flush=True)
    ok_scans = [s for s in scans if not s.get("error") and s.get("n_candidates", 0) > 0]
    check("scan", len(ok_scans) > 0, f"{len(ok_scans)}/{len(scans)} scenes with extra objects")
    in_seed = next((s for s in ok_scans if any(r != "out_of_way" for r in s["roles"].values())), None)
    out_seed = next((s for s in ok_scans if any(r == "out_of_way" for r in s["roles"].values())), None)
    if in_seed is None:
        print("[smoke] no in-the-way scene: abort", flush=True)
        (out / "smoke.json").write_text(json.dumps(report, indent=2, default=str))
        return 1

    def pick(scan, in_way):
        items = sorted([i for i, r in scan["roles"].items() if (r != "out_of_way") == in_way])
        return items[0]

    model = build_model(cfg, obs_dim(cfg), N_ACT)
    rng = np.random.default_rng(0)

    def fixed(item, kind):
        return lambda runner, _rng: make_named(runner, item, kind)

    def run(name, job, choose, consistency=False):
        t0 = time.perf_counter()
        old = Episode.check_consistency
        Episode.check_consistency = consistency
        try:
            res = run_episode(cfg, job, model, rng, scratch, choose=choose)
        except Exception as exc:
            traceback.print_exc()
            res = {"crash": f"{type(exc).__name__}: {exc}"}
        finally:
            Episode.check_consistency = old
        res["wall"] = time.perf_counter() - t0
        traj = res.pop("traj", None)
        report["episodes"][name] = res
        print(f"[smoke] {name}: " + json.dumps({k: v for k, v in res.items() if k not in ("grip_events",
                                                                                      "replan_log")}, default=str),
              flush=True)
        return res, traj

    # 2. playback reference
    item = pick(in_seed, True)
    res, _ = run("playback_delta0", Job(kind="throughput", seed=in_seed["seed"], policy="zero", noise=False,
                                        terminate=False, controller="playback"), fixed(item, "avoid"))
    check("playback runs", "crash" not in res and not res.get("skipped") and res.get("end") in ("finished", "truncated"),
          {k: res.get(k) for k in ("end", "end_reason", "success", "violated", "sim_ticks", "wall")})

    # 3. 25 Hz delta = 0, deviation trigger, consistency
    res, _ = run("tick25_delta0_deviation", Job(kind="throughput", seed=in_seed["seed"], policy="zero", noise=False,
                                                terminate=False, trigger="deviation"), fixed(item, "avoid"),
                 consistency=True)
    ok = "crash" not in res and not res.get("skipped") and res.get("policy_ticks", 0) > 50
    check("25 Hz loop runs", ok, {k: res.get(k) for k in ("end", "end_reason", "success", "violated", "policy_ticks",
                                                          "gripper_ticks", "untagged_steps", "replans", "max_dev",
                                                          "clip_frac", "settle_ticks", "wall_play")})
    cons = res.get("consistency") or {}
    check("named distance = robot_item_distances", cons.get("n", 0) > 50 and cons.get("max_margin_err_near", 1) < 1e-6
          and cons.get("sign_flips", 1) == 0 and cons.get("max_margin_err", 1) < 0.01, cons)
    check("gripper segments go through the policy loop", res.get("gripper_ticks", 0) > 0,
          {"gripper_ticks": res.get("gripper_ticks"), "untagged_steps": res.get("untagged_steps")})
    ge = res.get("grip_events") or []
    check("grip events logged", len(ge) > 0, ge[:6])

    # 3b. the same with the stay rule (margin from one_margin)
    res, _ = run("tick25_delta0_stay", Job(kind="throughput", seed=in_seed["seed"], policy="zero", noise=False,
                                           terminate=False), fixed(item, "stay"), consistency=True)
    cons = res.get("consistency") or {}
    check("stay margin = rule_margins", cons.get("n", 0) > 50 and cons.get("max_margin_err", 1) < 1e-9
          and cons.get("sign_flips", 1) == 0, cons)

    # 4. training episode, random policy, fixed trigger, noise
    res, traj = run("train_random_fixed", Job(kind="train", seed=in_seed["seed"], noise=True, terminate=True,
                                              trigger="fixed"), fixed(item, "avoid"))
    ok = traj is not None and len(traj["rew"]) > 0
    detail = {k: res.get(k) for k in ("end", "end_reason", "policy_ticks", "replans", "replan_time", "corr_rms",
                                       "rew_mean", "boot", "last_value")}
    if ok:
        detail.update({"T": len(traj["rew"]), "obs_shape": list(traj["obs"].shape), "rew_min": float(traj["rew"].min()),
                       "rew_max": float(traj["rew"].max()), "disc_min": float(traj["disc"].min())})
        ok = (traj["obs"].shape[1] == obs_dim(cfg) and np.all(np.isfinite(traj["obs"])) and traj["rew"].min() > 0
              and traj["rew"].max() <= 1.0 + 1e-6 and res.get("replans", 0) > 0)
    check("training episode transitions", ok, detail)
    rlog = res.get("replan_log") or []
    report["replan_log_fixed"] = rlog[:20]

    # 5. evaluation episode with video
    if out_seed is not None:
        vid = str(out / "eval_out_of_way.mp4")
        res, _ = run("eval_video", Job(kind="eval", seed=out_seed["seed"], policy="model", deterministic=True,
                                       noise=False, terminate=False, w=1.0, video=vid),
                     fixed(pick(out_seed, False), "avoid"))
        check("eval video written", bool(res.get("video")) and Path(vid).exists(), res.get("video"))
    (out / "smoke.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[smoke] {len(fails)} failed: {fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
