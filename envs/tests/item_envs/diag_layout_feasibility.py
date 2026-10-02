"""One-off Phase 3 diagnostic: per-host corridor geometry + layout feasibility.

For each T1 host x seeds 0-3 (ur5-wsg): build the Env, compute the longest
corridor + keepaways exactly as spawn_candidates does, then run the pure
layout loop (no spawning) and report per-seed: corridor length, keepaway
radii, attempts used (0 = infeasible within MAX_LAYOUTS), and the achieved
class histogram. This is the layout-feasibility slice of probe P1 — it
decides whether hostile hosts need a documented layout fix.

GPU job (Env needs Vulkan/cuRobo install); prints a JSON line per cell.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/work/nvme/bifq/project/LanguageConditionHJ/envs")

from main.item_envs import spawn  # noqa: E402
from main.item_envs.probes._common import build_env  # noqa: E402
from main.item_envs.probes.p1 import load_t1_cfg  # noqa: E402

HOSTS = ["place_container_plate", "place_burger_fries", "stack_blocks_two",
         "stack_bowls_two", "place_bread_basket"]
SEEDS = [0, 1, 2, 3]


def main() -> None:
    task_cfg = load_t1_cfg()
    cfg = task_cfg["objects"]
    objects_cfg = spawn.physics.load_objects_cfg()
    pool = spawn.t1_pool_entries(objects_cfg)
    colors = list((objects_cfg.get("colors") or {}).get("train") or [])
    for host in HOSTS:
        for seed in SEEDS:
            env = None
            try:
                env = build_env(host, "ur5-wsg", seed)
                spec = spawn.TASK_SPECS[host]
                arm = spawn.resolve_arm(env.task, spec, "auto")
                p0, p1, _ = spawn.longest_corridor(env.task, spec, env.robot, arm)
                keepaways = (
                    spawn.ob.keepaways_from_task(env.task, spec)
                    + spawn.ob.keepaways_from_robot(env)
                )
                L = float(np.linalg.norm(np.asarray(p1[:2]) - np.asarray(p0[:2])))
                rng = np.random.RandomState(seed + 4241)
                attempts, layout, dropped = 0, None, 0
                for attempt in range(spawn.MAX_LAYOUTS):
                    out = spawn._layout_plan(
                        p0, p1, keepaways, cfg, task_cfg, rng, pool, colors,
                        task_cfg.get("corridor_t", [0.22, 0.48]),
                    )
                    attempts = attempt + 1
                    if out is not None:
                        layout, dropped = out
                        break
                rec = {
                    "host": host, "seed": seed, "L": round(L, 3),
                    "keepaways": [[round(float(v), 3) for v in np.asarray(pt[:2])] + [round(float(r), 3)]
                                  for pt, r in keepaways],
                    "attempts": attempts if layout is not None else 0,
                    "classes": [p.cls for p in layout] if layout else None,
                    "assets": [p.draw.asset for p in layout] if layout else None,
                    "k_dropped": dropped if layout else None,
                    "t_fallbacks": sum(p.t_fallback for p in layout) if layout else None,
                    "degraded": sum(p.degraded_from is not None for p in layout) if layout else None,
                }
            except Exception as exc:
                rec = {"host": host, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            print("DIAG " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
