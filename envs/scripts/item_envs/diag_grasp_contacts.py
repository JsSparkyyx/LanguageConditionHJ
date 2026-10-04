"""Diag: log robot contacts + EE tracking error after each task.move() of the
first N moves of an item episode (GPU node only).

    python scripts/item_envs/diag_grasp_contacts.py --task granular_pour \
        --embodiment ur5-wsg --seed 132 --moves 8 --out runs/diag/dXX/contacts

Per move: arm, commanded EE target (task._cehj_ee_target), achieved EE pose
and their position / angle error, and every contact with separation < 2 mm
that involves a robot link and a body outside the robot (entity names).
Answers "the arm did not reach the commanded pose — what is it touching?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main.item_envs.runner import ItemEpisodeRunner  # noqa: E402


def _quat_angle_deg(q1, q2):
    d = abs(float(np.dot(np.asarray(q1, float), np.asarray(q2, float))))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))))


def _robot_contacts(task, robot_names, sep_max=0.002):
    out = []
    for c in task.scene.get_contacts():
        names = [b.entity.name for b in c.bodies]
        in_robot = [n in robot_names for n in names]
        if not any(in_robot) or all(in_robot):
            continue
        seps = [p.separation for p in c.points]
        if not seps or min(seps) > sep_max:
            continue
        imp = float(sum(np.linalg.norm(p.impulse) for p in c.points))
        out.append((names[0], names[1], float(min(seps)), imp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--embodiment", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--moves", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-after", type=int, default=0, help="abort the episode after this many moves (0 = run it out)")
    args = ap.parse_args()

    runner = ItemEpisodeRunner(args.task, embodiment=args.embodiment, seed=args.seed, render="none",
                               out_root=args.out)
    runner.setup()
    task = runner.env.task
    robot_names = set()
    for art in (task.robot.left_entity, task.robot.right_entity):
        for link in art.get_links():
            robot_names.add(link.get_name())
    ctrl = runner.ctrl
    last_plan: dict = {}
    orig_plan = ctrl.plan

    def plan(arm, target_pose, **kw):
        res = orig_plan(arm, target_pose, **kw)
        if isinstance(res, dict) and res.get("status") == "Success" and len(res.get("position", [])):
            last_plan[arm] = np.asarray(res["position"], dtype=np.float64)[-1]
        return res

    ctrl.plan = plan
    task.robot.left_plan_path = lambda target_pose, **kw: plan("left", target_pose, **kw)
    task.robot.right_plan_path = lambda target_pose, **kw: plan("right", target_pose, **kw)
    orig_move = task.move
    count = {"n": 0}

    def move(*a, **kw):
        ret = orig_move(*a, **kw)
        count["n"] += 1
        if count["n"] <= args.moves:
            tgts = getattr(task, "_cehj_ee_target", {}) or {}
            for arm, tgt in tgts.items():
                ee = np.asarray(task.get_arm_pose(arm), dtype=np.float64).reshape(7)
                tgt = np.asarray(tgt, dtype=np.float64).reshape(7)
                print(f"[diag] move {count['n']} {arm}: target {np.round(tgt[:3], 3).tolist()} "
                      f"ee {np.round(ee[:3], 3).tolist()} pos_err {np.linalg.norm(ee[:3] - tgt[:3]) * 100:.1f} cm "
                      f"ang_err {_quat_angle_deg(ee[3:], tgt[3:]):.1f} deg")
            for arm, q_end in last_plan.items():
                q_now = np.asarray(task.robot.get_left_arm_jointState() if arm == "left"
                                   else task.robot.get_right_arm_jointState(), dtype=np.float64).reshape(-1)
                n = min(len(q_end), len(q_now))
                print(f"[diag]    {arm} plan-end q {np.round(q_end[:n], 3).tolist()} achieved "
                      f"{np.round(q_now[:n], 3).tolist()} max|dq| {np.abs(q_end[:n] - q_now[:n]).max():.3f} rad")
            for n0, n1, sep, imp in _robot_contacts(task, robot_names):
                print(f"[diag]    contact {n0} <-> {n1} sep {sep * 1000:.1f} mm impulse {imp:.4f}")
        if args.stop_after and count["n"] >= args.stop_after:
            print("[diag] stop-after reached", flush=True)
            import os
            os._exit(0)
        return ret

    task.move = move
    meta, error = runner.run()
    print(f"[diag] done success={meta.task_success} error={error}")


if __name__ == "__main__":
    main()
