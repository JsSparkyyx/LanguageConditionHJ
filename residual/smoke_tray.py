"""Simulator smoke test of the residual policy on the sliding tray (tray_slide), one process, no cameras:

    scripts/lang/in_container.sh python -m residual.smoke_tray --out /abs/runs/residual/smoke_tray [--n-seeds 3]

1. scan the seeds (both items and their upright rules present);
2. RoboTwin's own 250 Hz playback of the careless fast drag, delta = 0 (reference: which items fall);
3. the 25 Hz loop, delta = 0, deviation trigger, the glass and then the can named on each seed, with the per-tick
   check that the named margin equals the runner's own (rule_margins); resyncs at delta = 0 should be rare;
4. a scripted residual that holds the arm back during the drag (delta = -alpha a_nom): the via-path resync must let
   it slow the tray (lower peak speed, resyncs > 0) and the task must still finish (tray in the pad);
5. a training episode with a randomly initialised policy: transitions, rewards, observation size.

Writes ``<out>/smoke_tray.json``; exits 1 if a check fails. The rates over a few seeds are a sanity check only.
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
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.7, help="hold-back share of a_nom during the drag")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_num_threads(2)
    from .config import N_ACT, ResidualConfig
    from .episode import Episode, Job, run_episode
    from .policy import build_model
    from .tasks import task_hooks
    from .tray import make_named

    cfg = ResidualConfig(task="tray_slide")
    hooks = task_hooks(cfg.task)
    scratch = str(out / "scratch")
    report: dict = {"checks": {}, "episodes": {}}
    fails = []

    def check(name, ok, detail=None):
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"[smoke-tray] {'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail, default=str)}", flush=True)
        if not ok:
            fails.append(name)

    # 1. scan
    scans = []
    for s in range(args.seed, args.seed + args.n_seeds):
        t0 = time.perf_counter()
        try:
            r = hooks.build_runner(cfg, s, scratch, Job(kind="scan"))
            try:
                scans.append({"seed": s, **hooks.scan_record(r), "build_s": time.perf_counter() - t0})
            finally:
                r.close()
        except Exception as exc:
            traceback.print_exc()
            scans.append({"seed": s, "error": f"{type(exc).__name__}: {exc}"})
    report["scans"] = scans
    ok_seeds = [s["seed"] for s in scans if not s.get("error") and s.get("n_candidates") == 2 and s.get("rules_ok")]
    check("scan", len(ok_seeds) == len(scans), scans)
    if not ok_seeds:
        (out / "smoke_tray.json").write_text(json.dumps(report, indent=2, default=str))
        return 1

    model = build_model(cfg, hooks.obs_dim(cfg), N_ACT)
    rng = np.random.default_rng(0)
    keys = ("end", "end_reason", "success", "violated", "other_broken", "named_max_tilt_deg", "other_max_tilt_deg",
            "tray_v_max", "tray_acc_max", "pad_dist", "resyncs", "resync_rows", "replans", "policy_ticks",
            "gripper_ticks", "untagged_steps", "max_dev", "clip_frac", "corr_rms", "settle_ticks", "wall_play",
            "behaviour", "drag_fallback")

    def summary(res):
        return {k: res.get(k) for k in keys}

    def run(name, job, item, consistency=False):
        t0 = time.perf_counter()
        old = Episode.check_consistency
        Episode.check_consistency = consistency
        try:
            res = run_episode(cfg, job, model, rng, scratch, choose=lambda runner, _r: make_named(runner, item))
        except Exception as exc:
            traceback.print_exc()
            res = {"crash": f"{type(exc).__name__}: {exc}"}
        finally:
            Episode.check_consistency = old
        res["wall"] = time.perf_counter() - t0
        traj = res.pop("traj", None)
        report["episodes"][name] = res
        print(f"[smoke-tray] {name}: {json.dumps(summary(res), default=str)}", flush=True)
        return res, traj

    # 2. playback reference (250 Hz), fast drag
    s0 = ok_seeds[0]
    res, _ = run("playback_fast_glass", Job(kind="throughput", seed=s0, policy="zero", noise=False, terminate=False,
                                            controller="playback"), "glass")
    check("playback runs", "crash" not in res and not res.get("skipped") and res.get("end") in ("finished", "truncated"),
          summary(res))

    # 3. 25 Hz loop, delta = 0, both items named on every seed
    zero = {"glass": [], "can": []}
    for s in ok_seeds:
        for item in ("glass", "can"):
            res, _ = run(f"tick25_delta0_{item}_s{s}", Job(kind="throughput", seed=s, policy="zero", noise=False,
                                                            terminate=False), item, consistency=True)
            zero[item].append(res)
    flat = [r for v in zero.values() for r in v]
    check("25 Hz loop runs", all("crash" not in r and not r.get("skipped") and r.get("policy_ticks", 0) > 50
                                 for r in flat), [summary(r) for r in flat])
    cons = [r.get("consistency") or {} for r in flat]
    check("named margin = rule_margins", all(c.get("n", 0) > 50 and c.get("max_margin_err", 1) < 1e-9
                                             and c.get("sign_flips", 1) == 0 for c in cons), cons)
    check("drag profile is the fast one", all((r.get("drag_profile") or {}).get("behaviour") == "fast" for r in flat),
          [r.get("drag_profile") for r in flat])
    report["delta0_rates"] = {item: {"broken": float(np.mean([r.get("violated", False) for r in v])),
                                     "success": float(np.mean([r.get("success", False) for r in v])),
                                     "resyncs": float(np.mean([r.get("resyncs", 0) for r in v])),
                                     "tray_v_max": float(np.mean([r.get("tray_v_max", 0) for r in v]))}
                              for item, v in zero.items()}
    print(f"[smoke-tray] delta = 0, 25 Hz: {json.dumps(report['delta0_rates'])}", flush=True)

    # 4. a scripted residual that holds the arm back during the drag
    alpha = float(args.alpha)

    class HoldBack(Episode):
        def policy_tick(self, view):
            super().policy_tick(view)   # keeps the reward bookkeeping (delta = 0 under policy "zero")
            m = self.measure()
            if not m["drag"]:
                return np.zeros(N_ACT)
            delta = np.clip(-alpha * np.asarray(view["a_nom"]), -cfg.delta_bound, cfg.delta_bound)
            c = float(np.sum((delta / cfg.delta_bound) ** 2) / N_ACT)
            self.corr_sq[-1] = c
            self.rews[-1] = 1.0 / (1.0 + self.w * c)
            self._last_corr = float(np.sqrt(c))
            return delta

    hb = []
    for s in ok_seeds:
        job = Job(kind="throughput", seed=s, policy="zero", noise=False, terminate=False, w=1.0)
        hooks.prepare_job(cfg, job, rng)
        t0 = time.perf_counter()
        runner = hooks.build_runner(cfg, s, scratch, job)
        try:
            ep = HoldBack(cfg, job, runner, make_named(runner, "glass"), None, rng)
            res = ep.play()
        except Exception as exc:
            traceback.print_exc()
            res = {"crash": f"{type(exc).__name__}: {exc}"}
        finally:
            runner.close()
        res["wall"] = time.perf_counter() - t0
        report["episodes"][f"holdback_glass_s{s}"] = res
        print(f"[smoke-tray] holdback_glass_s{s}: {json.dumps(summary(res), default=str)}", flush=True)
        hb.append(res)
    v0 = [r.get("tray_v_max", 0.0) for r in zero["glass"]]
    v1 = [r.get("tray_v_max", 0.0) for r in hb]
    check("holding back slows the drag (resync)", all("crash" not in r for r in hb)
          and all(r.get("resyncs", 0) > 0 for r in hb) and float(np.mean(v1)) < 0.7 * float(np.mean(v0)),
          {"tray_v_max_delta0": v0, "tray_v_max_holdback": v1, "resyncs": [r.get("resyncs") for r in hb]})
    check("holding back still finishes the task", sum(bool(r.get("success")) for r in hb) >= max(1, len(hb) - 1),
          [summary(r) for r in hb])
    report["holdback_rates"] = {"glass_broken": float(np.mean([r.get("violated", False) for r in hb])),
                                "can_broken": float(np.mean([r.get("other_broken", False) for r in hb])),
                                "success": float(np.mean([r.get("success", False) for r in hb]))}
    print(f"[smoke-tray] hold back (alpha {alpha}): {json.dumps(report['holdback_rates'])}", flush=True)

    # 5. training episode, random policy, noise on
    res, traj = run("train_random", Job(kind="train", seed=s0, noise=True, terminate=True), "can")
    ok = traj is not None and len(traj["rew"]) > 0
    detail = summary(res)
    if ok:
        detail.update({"T": len(traj["rew"]), "obs_shape": list(traj["obs"].shape),
                       "rew_min": float(traj["rew"].min()), "rew_max": float(traj["rew"].max())})
        ok = (traj["obs"].shape[1] == hooks.obs_dim(cfg) and np.all(np.isfinite(traj["obs"]))
              and traj["rew"].min() > 0 and traj["rew"].max() <= 1.0 + 1e-6)
    check("training episode transitions", ok, detail)

    (out / "smoke_tray.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[smoke-tray] {len(fails)} failed: {fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
