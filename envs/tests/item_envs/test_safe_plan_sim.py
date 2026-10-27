"""Sim tests and the GPU pilot for the T1 safe nominal mode (safe_plan.py, 2026-10-27). GPU node only.

pytest -m sim, per host, the first scene (seeds from 7300) with an extra object, every extra object protected
(active_rule_id None, as the demo collector runs it):

- the safe world is cuRobo's: every candidate a grown-hull mesh plus the lowered tabletop, enabled in all four
  MotionGens, RoboTwin's misplaced "table" cuboid gone, and both arms' home poses free in the safe model;
- one safe episode runs without an exception, and a plan failure always carries its reason (meta.diag.task.safe_fail).

The pilot (5 hosts x ~10 scenes; one row per attempt, contacts checked at every physics step), from envs/:

    python -m tests.item_envs.test_safe_plan_sim --scenes 10 --out <dir> [--hosts a,b] [--seed-start 7300]
        [--modes safe[,careless]] [--max-steps 10000]

writes <dir>/pilot.jsonl and prints the per-host yield, the safe_fail reasons and the ladder levels used. Kept =
the task succeeded, no extra object touched anything but the table (PhysX separation <= 0) at any physics step, force
< 20 N, and no stay rule broken (Yuxuan's keep rule of 2026-10-27).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pytest

sapien = pytest.importorskip("sapien")

from main.item_envs import safe_plan  # noqa: E402
from main.item_envs.items import entity_of  # noqa: E402
from main.item_envs.runner import ItemEpisodeRunner  # noqa: E402

pytestmark = [pytest.mark.sim, pytest.mark.gpu]

HOSTS = ("place_container_plate", "place_burger_fries", "stack_blocks_two", "stack_bowls_two", "place_bread_basket")
SEED_START = 7300
FORCE_MAX = 20.0


def _runner(host, seed, mode, out_root, max_steps=10000):
    return ItemEpisodeRunner("selective_avoid", embodiment="ur5-wsg", seed=seed, host=host, nominal_mode=mode,
                             active_rule_id=None, render="none", out_root=out_root, max_steps=max_steps)


def _enabled(wcc) -> set:
    out = {wcc._env_obbs_names[0][i] for i in range(int(wcc._env_n_obbs[0])) if int(wcc._cube_tensor_list[2][0, i]) == 1}
    if getattr(wcc, "_mesh_tensor_list", None) is not None:
        out |= {wcc._env_mesh_names[0][i] for i in range(int(wcc._env_n_mesh[0]))
                if int(wcc._mesh_tensor_list[2][0, i]) == 1}
    return out


def attempt(host, seed, mode, out_root, max_steps=10000) -> dict:
    """One episode with every candidate protected; contacts of the candidates checked at every physics step."""
    t0 = time.time()
    row = {"host": host, "seed": int(seed), "mode": mode}
    r = _runner(host, seed, mode, Path(out_root) / f"{host}_{seed}_{mode}", max_steps)
    try:
        try:
            r.setup()
        except ValueError as exc:   # safe mode on a layout without extra objects: nothing to protect
            if "no item to protect" not in str(exc):
                raise
            row["skip"] = "no_candidate"
            return row
        cands = [s for s in r.items if s.role == "candidate"]
        placements = {p.get("item_id"): p.get("cls") for p in (r._spawn_report or {}).get("placements") or []}
        row.update({"k": len(cands), "objects": {s.item_id: [s.category, placements.get(s.item_id)] for s in cands}})
        if not cands:
            row["skip"] = "no_candidate"
            return row
        ents = [entity_of(s) for s in cands]
        stat = {"touch": False, "force": 0.0, "partners": set(), "steps": 0}
        scene = r.env.task.scene
        orig = scene.step

        def step():
            orig()
            stat["steps"] += 1
            f, touched, partners = safe_plan.contact_with(r.env, ents)
            stat["touch"] |= touched
            stat["partners"] |= partners
            stat["force"] = max(stat["force"], f)

        scene.step = step
        meta, err = r.run()
        task_diag = meta.diag.get("task") or {}
        stay = sorted(rid for rid, v in (meta.violations or {}).items() if rid.startswith("stay|") and v)
        row.update({
            "success": bool(meta.task_success), "error": (err or "")[:200], "plan_failures": int(meta.plan_failures),
            "touched": stat["touch"], "max_force_N": round(stat["force"], 1), "partners": sorted(stat["partners"]),
            "stay_broken": stay, "steps": stat["steps"],
            "safe_fail": task_diag.get("safe_fail"), "levels": task_diag.get("safe_plan_levels"),
            "wide_grasp": task_diag.get("safe_wide_grasp", 0),
            "log_tail": [{k: e.get(k) for k in ("call", "skill", "level", "kind", "status", "start", "culprits")}
                         for e in (task_diag.get("safe_plan_log") or []) if not e.get("ok")][-4:],
            "world": (meta.diag.get("safe_plan") or {}).get("world"),
        })
        row["kept"] = bool(row["success"] and not row["error"] and not row["touched"] and row["max_force_N"] < FORCE_MAX
                           and not stay)
        return row
    except Exception as exc:  # noqa: BLE001 - a pilot row, never a crash
        row["exception"] = f"{type(exc).__name__}: {exc}"[:300]
        return row
    finally:
        r.close()
        row["wall_s"] = round(time.time() - t0, 1)


def _first_scene(host, out_root, tries: int = 30):
    """setup() of the first safe scene with an extra object (seeds from SEED_START); None if none in ``tries``."""
    for seed in range(SEED_START, SEED_START + tries):
        r = _runner(host, seed, "safe", out_root)
        try:
            r.setup()
        except ValueError as exc:   # safe mode: no item to protect (a layout without extra objects)
            r.close()
            if "no item to protect" in str(exc):
                continue
            raise
        return r
    return None


@pytest.mark.parametrize("host", HOSTS)
def test_safe_world_in_curobo(host, tmp_path):
    r = _first_scene(host, tmp_path)
    if r is None:
        pytest.skip(f"{host}: no scene with an extra object in 30 seeds")
    try:
        world = r._safe_world
        cands = sorted(s.item_id for s in r.items if s.role == "candidate")
        assert sorted(r.safe_plan_info["protected"]) == cands
        assert all(world.stats["items"][i]["shape"] == "hull" for i in cands), world.stats
        robot = r.env.robot
        for side, planner, entity in (("left", robot.left_planner, robot.left_entity),
                                      ("right", robot.right_planner, robot.right_entity)):
            names = set(world.names(side))
            assert safe_plan.TABLE_NAME in names and len(names) == len(cands) + 1
            for mg in (planner.motion_gen, planner.motion_gen_batch):
                assert _enabled(mg.world_coll_checker) == names     # RoboTwin's misplaced "table" cuboid is off
            safe_plan.sync_arm(r.env, side, 0.01, "full")
            chk = safe_plan.check_start(planner, planner.motion_gen_batch, entity.get_qpos(), sorted(names))
            assert chk["ok"], f"{host} {side} home pose in collision in the safe model: {chk}"
    finally:
        r.close()


@pytest.mark.parametrize("host", HOSTS)
def test_safe_episode_reports_every_failure(host, tmp_path):
    for seed in range(SEED_START, SEED_START + 30):
        row = attempt(host, seed, "safe", tmp_path)
        if row.get("skip") != "no_candidate":
            break
    else:
        pytest.skip(f"{host}: no scene with an extra object in 30 seeds")
    print(json.dumps(row))
    assert "exception" not in row, row
    if row["plan_failures"]:
        assert row["safe_fail"], row                  # every safe plan failure carries its reason
    assert row["levels"], row                         # the ladder ran (plans per level)


def _summary(rows) -> dict:
    out = {}
    for host in sorted({r["host"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows}):
            rs = [r for r in rows if r["host"] == host and r["mode"] == mode and not r.get("skip")]
            if not rs:
                continue
            levels = Counter()
            for r in rs:
                levels.update({k: int(v) for k, v in (r.get("levels") or {}).items()})
            out[f"{host}/{mode}"] = {
                "attempts": len(rs), "kept": sum(bool(r.get("kept")) for r in rs),
                "success": sum(bool(r.get("success")) for r in rs), "touched": sum(bool(r.get("touched")) for r in rs),
                "stay_broken": sum(bool(r.get("stay_broken")) for r in rs),
                "safe_fail": dict(Counter(r.get("safe_fail") for r in rs if r.get("safe_fail"))),
                "exceptions": sum("exception" in r for r in rs), "plan_levels": dict(levels),
                "wide_grasp": sum(int(r.get("wide_grasp") or 0) for r in rs),
            }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="T1 safe-mode GPU pilot (every extra object protected, ur5-wsg)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hosts", default=",".join(HOSTS))
    ap.add_argument("--scenes", type=int, default=10, help="scenes with an extra object per host")
    ap.add_argument("--seed-start", type=int, default=SEED_START)
    ap.add_argument("--modes", default="safe", help="safe and/or careless, comma-separated")
    ap.add_argument("--max-steps", type=int, default=10000)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()                    # absolute: Env() chdirs to the RoboTwin root
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for host in args.hosts.split(","):
        found, seed = 0, args.seed_start
        while found < args.scenes and seed < args.seed_start + 20 * args.scenes:
            scene_rows = [attempt(host, seed, mode, out / "episodes", args.max_steps) for mode in args.modes.split(",")]
            for row in scene_rows:
                print(json.dumps(row), flush=True)
                with open(out / "pilot.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
            rows += scene_rows
            found += int(not any(r.get("skip") for r in scene_rows))
            seed += 1
    summary = _summary(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
