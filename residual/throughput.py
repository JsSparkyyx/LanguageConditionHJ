"""First step of residual/PLAN.md: throughput and the delta = 0 baselines.

Runs delta = 0 episodes (the planner as a 25 Hz policy, no noise, the episode runs to its end so task success is
measured) under each controller setting, with P worker processes on the job's GPUs, for a fixed wall time each:

- ``deviation`` / ``fixed``: the 25 Hz controller with that replan trigger;
- ``playback``: RoboTwin's own 250 Hz plan playback (RiskTickController), the calibrated reference.

Per setting it reports ticks per second (per process and in total; scene building and cuRobo planning included),
seconds per episode, Env-build time, replans per episode and their cost, whether a replanned path starts from rest,
the named rule's break rate (in the way / out of the way), task success, which part first came within 1 cm of an
in-the-way object (held item or arm sphere), the hand's offset at gripper close / open, and GPU memory and RSS per
process. Everything lands in ``<out>/<setting>_p<P>/episodes.jsonl`` and ``<out>/summary.json``.

    scripts/lang/in_container.sh python -m residual.throughput --out /abs/runs/residual/throughput_1109 \
        --settings deviation:8,fixed:8,playback:8,deviation:1,deviation:16 --minutes 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import ResidualConfig, apply_sets
from .episode import Job
from .pool import WorkerPool


def _gpu_mem_by_pid() -> dict[int, float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    mem = {}
    for line in out.strip().splitlines():
        try:
            pid, mb = [x.strip() for x in line.split(",")]
            mem[int(pid)] = float(mb)
        except ValueError:
            continue
    return mem


def _rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _rate(xs) -> dict:
    xs = [bool(x) for x in xs]
    n = len(xs)
    k = int(sum(xs))
    if n == 0:
        return {"n": 0, "k": 0, "rate": None}
    try:
        from main.lang.eval_closed import wilson_str

        ci = wilson_str(k, n)
    except Exception:
        ci = None
    return {"n": n, "k": k, "rate": k / n, "ci": ci}


def summarize(eps: list[dict], wall: float, n_procs: int) -> dict:
    eps = [e for e in eps if not e.get("skipped")]
    if not eps:
        return {"episodes": 0}
    ticks = np.array([e["sim_ticks"] for e in eps], dtype=float)
    wall_job = np.array([e.get("wall_job", e["wall_episode"]) for e in eps], dtype=float)
    inw = [e for e in eps if e["in_way"]]
    outw = [e for e in eps if not e["in_way"]]
    replans = np.array([e["replans"] for e in eps], dtype=float)
    rlog = [r for e in eps for r in (e.get("replan_log") or [])]
    grips = [g for e in eps for g in (e.get("grip_events") or [])]
    first = [e for e in inw if e.get("contact_part")]

    def _med(xs):
        xs = [x for x in xs if x is not None]
        return None if not xs else float(np.median(xs))

    out = {
        "episodes": len(eps), "procs": n_procs, "wall_s": wall,
        "ticks_per_s_total": float(ticks.sum() / max(wall, 1e-9)),
        "ticks_per_s_per_proc": float(ticks.sum() / max(wall_job.sum(), 1e-9)),
        "episode_s_mean": float(wall_job.mean()),
        "build_s_mean": float(np.mean([e.get("wall_build", 0.0) for e in eps])),
        "builds_per_episode": float(np.mean([e.get("builds", 1) for e in eps])),
        "play_s_mean": float(np.mean([e["wall_play"] for e in eps])),
        "settle_s_mean": float(np.mean([e["wall_settle"] for e in eps])),
        "sim_ticks_mean": float(ticks.mean()),
        "policy_ticks_mean": float(np.mean([e["policy_ticks"] for e in eps])),
        "gripper_ticks_mean": float(np.mean([e["gripper_ticks"] for e in eps])),
        "off_ticks_mean": float(np.mean([e["off_ticks"] for e in eps])),
        "settle_ticks_mean": float(np.mean([e["settle_ticks"] for e in eps])),
        "settle_capped": _rate([e["settle_capped"] for e in eps]),
        "untagged_steps_mean": float(np.mean([e["untagged_steps"] for e in eps])),
        "replans_per_episode": float(replans.mean()),
        "replan_s_each": float(sum(e["replan_time"] for e in eps) / max(1.0, replans.sum())),
        "replan_failures": int(sum(e["replan_failures"] for e in eps)),
        "replan_start_speed": {
            "n": len(rlog),
            "v_plan_old_median": _med([r["v_plan_old"] for r in rlog]),
            "v_new_row0_median": _med([r["v_new_row0"] for r in rlog]),
            "v_new_tick1_median": _med([r["v_new_tick1"] for r in rlog]),
        },
        "max_dev_median": _med([e["max_dev"] for e in eps]),
        "clip_frac_mean": float(np.mean([e["clip_frac"] for e in eps])),
        "ends": {k: sum(e["end"] == k for e in eps) for k in ("finished", "violation", "truncated")},
        "end_reasons": {r: sum(e["end_reason"] == r for e in eps) for r in sorted({e["end_reason"] for e in eps})},
        "in_way": {
            "broken": _rate([e["violated"] for e in inw]),
            "success": _rate([e["success"] for e in inw]),
            "safe_success": _rate([e["success"] and not e["violated"] for e in inw]),
            "by_kind": {k: _rate([e["violated"] for e in inw if e["rule_kind"] == k]) for k in ("avoid", "stay")},
            "first_within_1cm": {"held": sum(e["contact_part"] == "held" for e in first),
                                 "arm": sum(e["contact_part"] == "arm" for e in first), "n_in_way": len(inw)},
            "first_touch": {"held": sum(e.get("touch_part") == "held" for e in inw),
                            "arm": sum(e.get("touch_part") == "arm" for e in inw)},
        },
        "out_of_way": {
            "broken": _rate([e["violated"] for e in outw]),
            "success": _rate([e["success"] for e in outw]),
        },
        "success": _rate([e["success"] for e in eps]),
        "grip_offsets": {
            k: {"n": sum(g["kind"] == k for g in grips),
                "joint_off_median": _med([g["joint_off"] for g in grips if g["kind"] == k]),
                "ee_off_median": _med([g["ee_off"] for g in grips if g["kind"] == k]),
                "ee_off_max": max([g["ee_off"] for g in grips if g["kind"] == k and g["ee_off"] is not None],
                                  default=None)}
            for k in ("close", "open")
        },
        "time_scales": sorted({e["time_scale"] for e in eps}),
    }
    return out


def run_setting(cfg, out: Path, name: str, controller: str, trigger: str | None, procs: int, minutes: float,
                seed_offset: int) -> dict:
    d = out / f"{name}_p{procs}"
    d.mkdir(parents=True, exist_ok=True)
    pool = WorkerPool(cfg, d, procs, d / "logs", d / "scratch", next_block=seed_offset)
    pool.start()
    n_jobs = procs * 200   # more than can finish: the wall-time budget decides
    for _ in range(n_jobs):
        pool.submit(Job(kind="throughput", policy="zero", noise=False, terminate=False, controller=controller,
                        trigger=trigger, tag=name).to_dict())
    t0 = time.time()
    t_stop = t0 + 60.0 * minutes
    eps, mem_samples = [], []
    with open(d / "episodes.jsonl", "a") as f:
        while time.time() < t_stop:
            for res in pool.poll(timeout=5.0):
                res.pop("traj", None)
                eps.append(res)
                f.write(json.dumps(res) + "\n")
                f.flush()
            if len(mem_samples) < 40 and time.time() - t0 > 90:
                gm = _gpu_mem_by_pid()
                mem_samples.append({str(w): {"gpu_mb": gm.get(pid), "rss_mb": _rss_mb(pid)}
                                    for w, pid in pool.pids().items()})
        t_wall = time.time() - t0
    # episodes still running at the stop are lost: ticks_per_s_per_proc (ticks / busy seconds of finished
    # episodes) is the unbiased rate, ticks_per_s_total undercounts by about one episode per process
    pool.close(timeout=30)
    gpu = [v["gpu_mb"] for s in mem_samples for v in s.values() if v["gpu_mb"]]
    rss = [v["rss_mb"] for s in mem_samples for v in s.values() if v["rss_mb"]]
    summ = summarize(eps, t_wall, procs)
    summ.update({"setting": name, "controller": controller, "trigger": trigger,
                 "gpu_mb_per_proc_max": max(gpu, default=None), "gpu_mb_per_proc_median": (
                     float(np.median(gpu)) if gpu else None),
                 "rss_mb_per_proc_max": max(rss, default=None),
                 "worker_restarts": dict(pool.restarts)})
    (d / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[throughput] {name} P={procs}: {json.dumps(summ)[:2000]}", flush=True)
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default="deviation:8,fixed:8,playback:8",
                    help="comma list of <deviation|fixed|playback>:<processes>")
    ap.add_argument("--minutes", type=float, default=8.0, help="wall minutes per setting")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = apply_sets(ResidualConfig(), args.set)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    allsum = {}
    if (out / "summary.json").exists():
        allsum = json.loads((out / "summary.json").read_text())
    for k, item in enumerate(args.settings.split(",")):
        name, procs = item.split(":")
        controller = "playback" if name == "playback" else "residual"
        trigger = None if name == "playback" else name
        key = f"{name}_p{procs}"
        if key in allsum and allsum[key].get("episodes", 0) > 0:
            print(f"[throughput] {key} done before: skipped", flush=True)
            continue
        # the same seed blocks for every setting: worker k sees the same scenes under each trigger
        allsum[key] = run_setting(cfg, out, name, controller, trigger, int(procs), args.minutes, seed_offset=1000)
        (out / "summary.json").write_text(json.dumps(allsum, indent=2))
    print(f"[throughput] all done: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
