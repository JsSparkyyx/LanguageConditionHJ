#!/usr/bin/env python
"""Phase-0 workspace sweep: top-down reachability grid over the table.

Builds ONE Env (place_empty_cup, seed 0) for the embodiment, then probes a
2 cm xy grid spanning the table footprint (1.2 m x 0.7 m, centered at the
task's table_xy_bias) at z = table_top + {0.02, 0.10, 0.20} m with a top-down
gripper pose, WITHOUT moving the robot.

Feasibility test: robot.left_plan_path(target_pose) -> CuroboPlanner.plan_path
(cuRobo MotionGen plan_single, max_attempts=10, planner.py:93-154) from the
settled home qpos — Success means a collision-free trajectory to the pose
exists against the planner's table-cuboid world + self-collision. This is the
same reachability test RoboTwin's own choose_grasp_pose uses
(_base_task.py:1123-1132); there is no bare-IK entry point in CuroboPlanner.

Target pose convention: RoboTwin "gripper pose" [x, y, z, qw, qx, qy, qz] —
the frame whose +x axis is the approach direction, whose origin sits 0.12 m
behind the fingertips (get_grasp_pose, _base_task.py:1061-1062;
_trans_from_gripper_to_endlink, robot.py:330-339). The top-down orientation
is GRASP_DIRECTION_DIC["top_down"] = [-0.5, 0.5, -0.5, -0.5] (wxyz)
(_GLOBAL_CONFIGS.py:24; used as the top-down reference in choose_grasp_pose,
_base_task.py:1146-1149). With --z-mode gripper (default) the probed z is the
gripper-pose origin height, so the FINGERTIPS would be 0.12 m lower;
--z-mode tcp shifts the target up by 0.12 m so the probed z is the fingertip
height.

Arm: the LEFT arm of the pair is swept (robot.left_plan_path). ur5-wsg and
franka-panda are dual_arm: False configs — RoboTwin instantiates the SAME arm
twice at +/- arm_distance/2 (robot.py:104-119), so the left map mirrored
about x=0 approximates the right arm's. Which arm place_empty_cup actually
grasps with at this seed is reported in the facts (task_grasp_arm_seed0).

A planner exception in any cell marks that cell infeasible and the sweep
continues; the script never crashes on planner errors.

Outputs:
  <out>/workspace_<embodiment>.npz   x/y/z vectors, feasible [nz, ny, nx],
                                     per-cell planner-exception counts, table
                                     pose, quat, embodiment, swept arm
  <out>/facts_<embodiment>.json      the EMBODIMENT_FACTS dict (also printed)
  stdout: one line "EMBODIMENT_FACTS <embodiment> {json}"

Usage (cwd = /work/nvme/bifq/project/LanguageConditionHJ):
    python envs/scripts/item_envs/workspace_sweep.py \
        --embodiment ur5-wsg --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "main" / "envs" / "env.py").is_file():
    ROOT = Path("/work/nvme/bifq/project/LanguageConditionHJ")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # only LanguageConditionHJ — never CEHJ_safe (task spec)

TABLE_SIZE_XY = (1.2, 0.7)   # create_table(length=1.2, width=0.7) (_base_task.py:306-315)
TABLE_TOP_NOMINAL = 0.74     # create_table_and_wall(table_height=0.74) (_base_task.py:120,271-274)
APPROACH_TO_TCP_M = 0.12     # gripper-pose origin -> fingertips along +x (_base_task.py:1062)


def parse_urdf_limits(urdf_path: Path) -> dict:
    """{joint_name: {type, lower, upper, velocity, effort}} for revolute/prismatic
    joints — the same fields body_features._parse_urdf_joints reads
    (main/network/body_features.py:79-89)."""
    import xml.etree.ElementTree as ET

    out = {}
    for j in ET.parse(str(urdf_path)).getroot().iter("joint"):
        if j.get("type") not in ("revolute", "prismatic"):
            continue
        rec = {"type": j.get("type")}
        lim = j.find("limit")
        for k in ("lower", "upper", "velocity", "effort"):
            if lim is not None and lim.get(k) is not None:
                try:
                    rec[k] = float(lim.get(k))
                except ValueError:
                    pass
        out[j.get("name")] = rec
    return out


def opening_range(grip_joints: list[dict]) -> tuple[float | None, float | None]:
    """(min, max) finger separation from URDF limits of the two finger joints.

    Side sign = sign(lower + upper): franka fingers are both positive
    (separation = q1 + q2), the WSG left finger lives in negative space
    (separation = q_right - q_left)."""
    pos = [g for g in grip_joints
           if g.get("lower") is not None and g.get("upper") is not None
           and (g["lower"] + g["upper"]) >= 0]
    neg = [g for g in grip_joints
           if g.get("lower") is not None and g.get("upper") is not None
           and (g["lower"] + g["upper"]) < 0]
    if not pos and not neg:
        return None, None
    sep_max = sum(g["upper"] for g in pos) - sum(g["lower"] for g in neg)
    sep_min = sum(g["lower"] for g in pos) - sum(g["upper"] for g in neg)
    return float(sep_min), float(sep_max)


def _pose_pq(pose) -> dict:
    return {"p": [float(v) for v in pose.p],
            "q_wxyz": [float(v) for v in pose.q]}


def _scene_root_pose(entity) -> dict:
    # same fallback chain as main/envs/obstacle.py:647
    p = entity.get_root_pose() if hasattr(entity, "get_root_pose") else entity.get_pose()
    return _pose_pq(p)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embodiment", required=True,
                   choices=["ur5-wsg", "franka-panda"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dx", type=float, default=0.02, help="grid spacing (m)")
    p.add_argument("--heights", type=str, default="0.02,0.10,0.20",
                   help="comma list of heights above the table top (m)")
    p.add_argument("--z-mode", choices=["gripper", "tcp"], default="gripper",
                   help="gripper: probed z = gripper-pose origin (RoboTwin "
                        "target convention, fingertips 0.12 m lower). tcp: "
                        "probed z = fingertip height (target shifted +0.12 m)")
    args = p.parse_args()

    # Resolve BEFORE Env creation: RoboTwin chdirs into its own root
    # (main/envs/env.py:137-142 prepare()).
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from main.envs.controller import CuroboIKController  # controller.py:38
    from main.envs.env import Env                        # env.py:338
    try:
        from main.train.collect import ARM_DISTANCE      # collect.py:123
    except Exception:
        ARM_DISTANCE = {"ur5-wsg": 0.8}                  # collect.py:123 (fallback copy)

    t0 = time.perf_counter()
    # cached cuRobo + mplib fallback + collision-cache patch, exactly like the
    # training/eval stack (collect.py:132-136 make_training_env)
    CuroboIKController.install()
    env = Env("place_empty_cup", args.embodiment, 0,
              arm_distance=ARM_DISTANCE.get(args.embodiment, 0.6))  # collect.py:137-139
    setup_s = time.perf_counter() - t0

    try:
        task = env.task
        robot = env.robot

        table_xy = np.asarray(getattr(task, "table_xy_bias", [0.0, 0.0]),
                              dtype=np.float64)          # _base_task.py:272
        # top surface = pose z of create_table = 0.74 + table_z_bias
        # (_base_task.py:111,274 + tabletop centered -thickness/2, create_actor.py:346)
        table_top_z = TABLE_TOP_NOMINAL + float(getattr(task, "table_z_bias", 0.0))

        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC  # after prepare()
        quat = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=np.float64)  # _GLOBAL_CONFIGS.py:24

        planner = robot.left_planner
        planner_class = type(planner).__name__

        # ---------------- grid ----------------
        dx = float(args.dx)
        xs = table_xy[0] + np.arange(-TABLE_SIZE_XY[0] / 2,
                                     TABLE_SIZE_XY[0] / 2 + dx / 2, dx)
        ys = table_xy[1] + np.arange(-TABLE_SIZE_XY[1] / 2,
                                     TABLE_SIZE_XY[1] / 2 + dx / 2, dx)
        heights = [float(h) for h in args.heights.split(",")]
        zs = table_top_z + np.asarray(heights, dtype=np.float64)
        z_shift = APPROACH_TO_TCP_M if args.z_mode == "tcp" else 0.0
        feasible = np.zeros((len(zs), len(ys), len(xs)), dtype=bool)
        errors = np.zeros((len(zs), len(ys), len(xs)), dtype=np.int64)

        print(f"[sweep] {args.embodiment}: grid {len(xs)}x{len(ys)}x{len(zs)} "
              f"= {feasible.size} cells, table_top_z={table_top_z:.3f}, "
              f"planner={planner_class}, z_mode={args.z_mode}")
        t_sweep = time.perf_counter()
        for zi, z in enumerate(zs):
            t_layer = time.perf_counter()
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    target = [float(x), float(y), float(z + z_shift)] + quat.tolist()
                    try:
                        # robot.py:425-457: gripper->endlink transform, then
                        # left_planner.plan_path (cuRobo plan_single)
                        res = robot.left_plan_path(target)
                        feasible[zi, yi, xi] = bool(res) and res.get("status") == "Success"
                    except Exception as exc:
                        errors[zi, yi, xi] += 1
                        if errors.sum() <= 5:
                            print(f"[sweep] planner exception at "
                                  f"z={z:.3f} y={y:.3f} x={x:.3f}: "
                                  f"{type(exc).__name__}: {exc}")
            print(f"[sweep] layer z={z:.3f}: {int(feasible[zi].sum())}/"
                  f"{feasible[zi].size} feasible "
                  f"({int(errors[zi].sum())} exceptions, "
                  f"{time.perf_counter() - t_layer:.0f}s)")
        sweep_s = time.perf_counter() - t_sweep

        # ---------------- embodiment facts ----------------
        urdf_path = Path(robot.left_urdf_path).resolve()  # robot.py:40 (relative; cwd=RoboTwin root)
        limits = parse_urdf_limits(urdf_path)
        arm_joint_names = list(robot.left_arm_joints_name)   # robot.py:52
        arm_dof = len(robot.get_left_arm_real_jointState()) - 1  # robot.py:508-515 (gripper scalar last)

        import yaml  # pyyaml, same as env.py:13
        cyml_path = Path(robot.left_curobo_yml_path).resolve()  # robot.py:42
        cyml = {}
        if cyml_path.is_file():
            cyml = yaml.safe_load(cyml_path.read_text()) or {}
        kin_cfg = (cyml.get("robot_cfg", {}) or {}).get("kinematics", {}) or {}
        cyml_urdf = kin_cfg.get("urdf_path")
        cspace_cfg = kin_cfg.get("cspace", {}) or {}
        frame_bias = getattr(planner, "frame_bias",
                             (cyml.get("planner", {}) or {}).get("frame_bias"))

        grip_names = [g[0].get_name() for g in robot.left_gripper]  # robot.py:182-189
        grip_joints = [{"name": n,
                        "lower": limits.get(n, {}).get("lower"),
                        "upper": limits.get(n, {}).get("upper"),
                        "velocity": limits.get(n, {}).get("velocity")}
                       for n in grip_names]
        grip_min, grip_max = opening_range(grip_joints)

        cup_x = float(task.cup.get_pose().p[0])     # place_empty_cup.py:57-59
        task_arm = "right" if cup_x > 0 else "left"

        layer_stats = [
            {"height_above_table_m": heights[zi],
             "z_m": float(zs[zi]),
             "n_feasible": int(feasible[zi].sum()),
             "n_cells": int(feasible[zi].size),
             "frac_feasible": float(feasible[zi].mean()),
             "n_planner_exceptions": int(errors[zi].sum())}
            for zi in range(len(zs))
        ]

        facts = {
            "embodiment": args.embodiment,
            "swept_arm": "left",
            "task": "place_empty_cup",
            "env_seed": 0,
            "task_grasp_arm_seed0": task_arm,
            "dual_arm_config": False,
            "arm_distance_m": float(ARM_DISTANCE.get(args.embodiment, 0.6)),
            "arm_base_pose_scene": {
                "left": _scene_root_pose(robot.left_entity),
                "right": _scene_root_pose(robot.right_entity),
            },
            "arm_base_pose_configured": {
                "left": _pose_pq(robot.left_entity_origion_pose),   # robot.py:63-65,112
                "right": _pose_pq(robot.right_entity_origion_pose),  # robot.py:90-93,113
            },
            "arm_dof": arm_dof,
            "arm_joint_names": arm_joint_names,
            "joint_vel_limits": {
                n: limits.get(n, {}).get("velocity") for n in arm_joint_names
            },
            "joint_pos_limits": {
                n: [limits.get(n, {}).get("lower"), limits.get(n, {}).get("upper")]
                for n in arm_joint_names
            },
            "urdf_path_sim": str(urdf_path),
            "urdf_path_sim_exists": urdf_path.is_file(),
            "curobo_yml_path": str(cyml_path),
            "curobo_yml_urdf_path": cyml_urdf,
            "curobo_yml_urdf_exists": (Path(cyml_urdf).is_file()
                                       if cyml_urdf else None),
            "curobo_frame_bias": frame_bias,
            "curobo_max_acceleration": cspace_cfg.get("max_acceleration"),
            "curobo_max_jerk": cspace_cfg.get("max_jerk"),
            "planner_class": planner_class,
            "gripper": {
                "joints": grip_joints,
                "opening_min_m": grip_min,
                "opening_max_m": grip_max,
                "gripper_bias": robot.left_gripper_bias,   # robot.py:54
                "gripper_scale": robot.left_gripper_scale,  # robot.py:55
            },
            "table": {
                "xy_bias": [float(v) for v in table_xy],
                "top_z_m": table_top_z,
                "size_xy_m": list(TABLE_SIZE_XY),
            },
            "target_quat_wxyz": quat.tolist(),
            "z_mode": args.z_mode,
            "approach_to_tcp_m": APPROACH_TO_TCP_M,
            "grid": {
                "dx_m": dx,
                "x": [float(xs[0]), float(xs[-1]), int(len(xs))],
                "y": [float(ys[0]), float(ys[-1]), int(len(ys))],
                "z_m": [float(v) for v in zs],
                "n_cells": int(feasible.size),
            },
            "layers": layer_stats,
            "wall_time_s": time.perf_counter() - t0,
            "wall_setup_s": setup_s,
            "wall_sweep_s": sweep_s,
        }

        np.savez(
            out_dir / f"workspace_{args.embodiment}.npz",
            x=xs, y=ys, z=zs,
            heights_above_table=np.asarray(heights, dtype=np.float64),
            feasible=feasible,
            planner_errors=errors,
            table_xy_bias=table_xy,
            table_top_z=table_top_z,
            quat_wxyz=quat,
            embodiment=args.embodiment,
            swept_arm="left",
            z_mode=args.z_mode,
            approach_to_tcp_m=APPROACH_TO_TCP_M,
        )
        (out_dir / f"facts_{args.embodiment}.json").write_text(
            json.dumps(facts, indent=2, default=str))
        print("EMBODIMENT_FACTS " + args.embodiment + " "
              + json.dumps(facts, sort_keys=True, default=str))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
