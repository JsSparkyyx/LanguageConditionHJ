"""Sim tests for the T1 spawner (PLAN v3 §7 T1): spawn_candidates on a real
host Env (place_bread_basket — layout-feasible on DIAG seeds 0-3, unlike
place_container_plate/stack_bowls_two whose hostile corridors are the P1
finding), one seed per embodiment, GPU node only.

Checks: 2-5 objects spawn (k = 2 is legal under fix-2 partial-K acceptance),
every object is DYNAMIC (static_objects: false default) with a mass inside
its objects.yaml category range, poses sit on the table, the layout survived
the 250-step settle inside spawn_candidates (spawn_rejections accounting
present), and task.item_meta is fed.
"""

from __future__ import annotations

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")
import sapien.physx as physx

from main.item_envs import spawn
from main.item_envs.items import entity_of
from main.item_envs.probes._common import build_env
from main.item_envs.probes.p1 import load_t1_cfg

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_spawn_candidates_dynamic_layout(embodiment):
    # Friendly host (place_bread_basket: DIAG layout-feasible on seeds 0-3,
    # att 1-23) — this test validates the spawn chain, not hostile-corridor
    # behavior. Keep the CEHJ scene-retry parity anyway: try seeds in
    # order, require a feasible layout on at least one.
    env = None
    specs = report = None
    for seed in (0, 1, 2, 3):
        env = build_env("place_bread_basket", embodiment, seed=seed)
        task_cfg = load_t1_cfg()
        rng = np.random.RandomState(seed + 4241)
        specs, report = spawn.spawn_candidates(env, task_cfg, rng)
        if len(specs) >= 3:  # prefer a full layout; k = 2 (partial-K) is legal
            break
        env.close()
        env = None
    try:
        assert env is not None, "no feasible T1 layout on seeds 0-3"
        assert 2 <= len(specs) <= 5  # fix-2 partial-K: k >= max(2, quota)
        assert report["k"] == len(specs)
        assert report["static_objects"] is False
        assert report["max_layouts_hit"] is False
        assert "spawn_rejections" in report
        assert report["spawn_rejections"] == (
            report["layout_retries"] + report["settle_rejections"]
        )

        objects_cfg = spawn.physics.load_objects_cfg()
        mass_ranges = {
            k: (v or {}).get("mass_range")
            for k, v in (objects_cfg.get("categories") or {}).items()
        }
        table_z = spawn.ob.TABLE_Z + float(getattr(env.task, "table_z_bias", 0.0))
        for s in specs:
            ent = entity_of(s)
            comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
            assert comp is not None, f"{s.item_id}: not a dynamic body"
            assert not comp.kinematic, f"{s.item_id}: kinematic (static parity leak)"
            lo_hi = mass_ranges.get(s.category)
            if lo_hi is not None:
                assert float(lo_hi[0]) - 1e-9 <= s.mass <= float(lo_hi[1]) + 1e-9, (
                    s.item_id, s.category, s.mass,
                )
            pose = ent.get_pose()
            # resting on the table: reference point within a few cm of the top
            assert table_z - 0.02 < pose.p[2] < table_z + 0.25, (s.item_id, pose.p[2])
            assert np.all(np.isfinite(np.asarray(pose.p, dtype=np.float64)))

        # per-object placement records and the runner's item_meta channel
        assert len(report["placements"]) == len(specs)
        assert all(
            p["cls"] in ("on_path", "near_path", "off_path")
            for p in report["placements"]
        )
        n_on = sum(p["cls"] == "on_path" for p in report["placements"])
        n_near = sum(p["cls"] == "near_path" for p in report["placements"])
        # fix-2 degradation: a drawn-on object may land near/off, so n_on can
        # be 0; the on/near quota holds on FINAL classes (partial-K guarantee)
        assert 0 <= n_on <= 2
        assert n_on + n_near >= 2
        assert report["k"] + report["k_dropped"] >= 3  # drawn K was in 3..5
        assert report["t_fallbacks"] <= report["k"]
        assert report["degraded"] <= report["k"]
        meta_ids = {d["item_id"] for d in env.task.item_meta}
        assert {s.item_id for s in specs} <= meta_ids

        # settle details recorded per object
        assert len(report["settle"]) == len(specs)
        assert all(d["moved_m"] <= spawn.MOVE_TOL_M for d in report["settle"])
        assert all(d["tilt_deg"] <= 3.0 + 1e-6 for d in report["settle"])
    finally:
        if env is not None:
            env.close()
