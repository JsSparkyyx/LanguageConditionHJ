#!/usr/bin/env python
"""T3 push_zone safe nominal mode, GPU pilot (ur5-wsg, Yuxuan 2026-10-27 demo set).

nominal_mode="safe" with no rule named (every element protected: the demo collector's case), optionally the careless
nominal on the same seeds for comparison. One row per (seed, mode) in <out_dir>/t3_safe_pilot.jsonl, printed too:

- the outcome: task success, every rule's lowest margin and the rules broken (min margin < 0), the plan failures,
  the runner's error, ``keep`` (success, no rule broken, no cylinder touched, no plan failure, no safe_fail: the
  demo collector's keep rule without its 5 mm mat bar, which needs the trace);
- the cylinders, at EVERY physics step: physics steps with a contact of each cylinder with anything but the table
  (PhysX separation <= 0, safe_plan.contact_with) and the largest contact force;
- the safe mode: safe_fail, plan_fail_step, the route (ok, reason, via points, turns, sides, clearances, blocking
  elements), the push log (fail, re-plans, yaw jumps, EE miss, box tilt, lifted corners, the wrist_3 choice at the
  first approach and the 180-deg switches, the largest joint travel of one move and where) and its full step list;
- the pushing arm's wrist_3 measured at every physics step (its range, and the largest change within 10 steps):
  push_via's yaw choice predicts it with the review's model (wrist_3 ~= pan - yaw + 1.58, fit on two careless
  runs), so compare it with push_wrist_first["peaks"] (the predicted peak of the yaw taken, and of the other);
- the cost: physics steps (the runner's max_steps must cover them) and wall time.

    python envs/scripts/item_envs/t3_safe_pilot.py <first_seed> <n_seeds> <out_dir> [safe,careless] [max_steps]

From the repo root inside the container, on a GPU node (RoboTwin must sit next to the checkout, ../RoboTwin):

    srun --account=bifq-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 \\
      --cpus-per-task=8 --mem=48g --time=00:55:00 \\
      scripts/lang/in_container.sh python envs/scripts/item_envs/t3_safe_pilot.py 7400 20 \\
      /work/nvme/bifq/project/tmp/t3_pilot safe,careless 15000

(LANG_REPO=<a frozen worktree> runs another checkout; its ../RoboTwin must exist too.) Rows append: a rerun with
the same out_dir adds rows, so give each pilot its own directory.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import main.lang  # noqa: F401,E402  (appends envs/main to main.__path__: main.item_envs)
from main.item_envs import safe_plan  # noqa: E402
from main.item_envs.items import entity_of  # noqa: E402
from main.item_envs.runner import ItemEpisodeRunner  # noqa: E402

PUSH_KEYS = ("fail", "yaw_jump_max_deg", "ee_miss_max", "ee_miss_flag", "box_tilt_max_deg", "wrist_first",
             "wrist_switches", "max_joint_travel", "max_joint_travel_step")


def episode(seed: int, mode: str, out: Path, max_steps: int) -> dict:
    with tempfile.TemporaryDirectory(dir=str(out)) as tmp:
        r = ItemEpisodeRunner("push_zone", embodiment="ur5-wsg", seed=seed, render="none", out_root=tmp,
                              nominal_mode=mode, active_rule_id=None, max_steps=max_steps)
        stat = {"touch": {}, "force": 0.0, "steps": 0, "w3_lo": None, "w3_hi": None, "w3_jump": 0.0}
        t0 = time.perf_counter()
        try:
            r.setup()
            cylinders = [entity_of(s) for s in r.items if s.role == "bystander"]
            task = r.env.task
            scene = task.scene
            step0 = scene.step
            joints = getattr(task.robot, f"get_{task.push_objects[0]['arm']}_arm_real_jointState")
            w3_ring = []

            def step():
                step0()
                stat["steps"] += 1
                w3 = float(joints()[5])   # ur5: pan, lift, elbow, wrist_1, wrist_2, wrist_3, then the gripper
                stat["w3_lo"] = w3 if stat["w3_lo"] is None else min(stat["w3_lo"], w3)
                stat["w3_hi"] = w3 if stat["w3_hi"] is None else max(stat["w3_hi"], w3)
                w3_ring.append(w3)
                if len(w3_ring) > 10:
                    stat["w3_jump"] = max(stat["w3_jump"], abs(w3_ring[-1] - w3_ring.pop(0)))
                if cylinders:
                    force, touched, partners = safe_plan.contact_with(r.env, cylinders)
                    stat["force"] = max(stat["force"], force)
                    if touched:
                        for name in partners:
                            stat["touch"][name] = stat["touch"].get(name, 0) + 1

            scene.step = step   # the runner's own step hook wraps this one (it counts and captures ticks)
            layout = dict(getattr(r.env.task, "layout_info", {}) or {})
            meta, err = r.run()
            diag = meta.diag.get("task") or {}
            route = diag.get("route") or {}
            push = diag.get("push") or {}
            broken = sorted(k for k, v in meta.violations.items() if v)
            safe_fail = diag.get("safe_fail")
            return {
                "seed": seed, "mode": mode, "scenario": layout.get("scenario"), "layout_mode": layout.get("mode"),
                "success": bool(meta.task_success), "rules_broken": broken,
                "keep": bool(meta.task_success) and not broken and not stat["touch"] and meta.plan_failures == 0
                and not safe_fail and not err,
                "min_margin": {k: round(float(v), 4) for k, v in meta.min_margin.items()},
                "cyl_touch_steps": stat["touch"], "max_force_N": round(stat["force"], 1),
                "plan_failures": meta.plan_failures, "error": (err or "")[:300],
                "safe_fail": safe_fail, "plan_fail_step": diag.get("plan_fail_step"),
                "route_ok": route.get("ok"), "route_reason": route.get("reason"), "via": route.get("via"),
                "turns_deg": route.get("turns_deg"), "first_deg": route.get("first_deg"), "sides": route.get("sides"),
                "min_clear": route.get("min_clear"), "blocking": (route.get("detail") or {}).get("blocking"),
                "replans": diag.get("replans") or [],
                "n_lifts": sum(1 for s in push.get("steps") or [] if s.get("kind") == "lift"),
                **{f"push_{k}": push.get(k) for k in PUSH_KEYS},
                "push_replans": push.get("replans") or [], "push_steps": push.get("steps") or [],
                "wrist3_min": None if stat["w3_lo"] is None else round(stat["w3_lo"], 3),
                "wrist3_max": None if stat["w3_hi"] is None else round(stat["w3_hi"], 3),
                "wrist3_jump_10steps": round(stat["w3_jump"], 3),
                "physics_steps": stat["steps"], "wall_s": round(time.perf_counter() - t0, 1),
            }
        except Exception as exc:  # noqa: BLE001  (one bad seed never ends the pilot)
            return {"seed": seed, "mode": mode, "error": f"{type(exc).__name__}: {exc}"[:300],
                    "physics_steps": stat["steps"], "wall_s": round(time.perf_counter() - t0, 1)}
        finally:
            r.close()


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    seed0, n, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]).resolve()
    modes = tuple(m for m in sys.argv[4].split(",") if m) if len(sys.argv) > 4 else ("safe",)
    max_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 15000
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(seed0, seed0 + n):
        for mode in modes:
            row = episode(seed, mode, out, max_steps)
            line = json.dumps(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("push_steps", "min_margin")}), flush=True)
            with open(out / "t3_safe_pilot.jsonl", "a") as fh:
                fh.write(line + "\n")


if __name__ == "__main__":
    main()
