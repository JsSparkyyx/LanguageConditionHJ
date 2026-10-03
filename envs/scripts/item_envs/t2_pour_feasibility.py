#!/usr/bin/env python
"""T2 pour kinematic feasibility probe (GPU node; dry runs only after the lift).

For one (embodiment, seed, grasp contact ids): build granular_pour through
ItemEpisodeRunner, grasp + lift the source bowl exactly like play_once, then
dry-run (RiskTickController.screw_path_ok — chained mplib screws, nothing
executes) the transit + tilt sequence over a grid of

  pivot fraction f (0 = EE origin ... 1 = pouring lip),
  mouth direction m (far lip, -y, +y),
  tilt angle theta.

The lip at theta_ref sits on the pour point; the table reports feasibility
and the lip's horizontal sweep over the pouring range [60, theta] deg (the
spread of the bead stream over the target).

    python scripts/item_envs/t2_pour_feasibility.py --embodiment ur5-wsg --seed 0 --cp 1,3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", default="ur5-wsg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cp", default="0,2", help="grasp contact ids right,left")
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()

    from main.item_envs.runner import ItemEpisodeRunner

    runner = ItemEpisodeRunner("granular_pour", embodiment=args.embodiment, seed=args.seed,
                               render="none", out_root="runs/diag/pourfeas")
    runner.setup()
    task = runner.env.task
    import importlib

    gp = importlib.import_module(type(task).__module__)
    cp = [int(v) for v in args.cp.split(",")]
    arm = "right" if task.container.get_pose().p[0] > 0 else "left"
    task.move(task.grasp_actor(task.container, arm_tag=arm, contact_point_id=cp[int(arm == "left")],
                               pre_grasp_dis=0.1))
    task.move(task.move_by_displacement(arm, z=0.1, move_axis="arm"))
    ok_lift = bool(getattr(task, "plan_success", True))

    ee = np.asarray(task.get_arm_pose(arm), dtype=np.float64).reshape(7)
    pour_point = task._pour_point(gp.min_pour_height(task._src_geom, clearance=gp.POUR_HEIGHT_CLEAR), 0.0)
    sign = -1.0 if task.container.get_pose().p[0] > 0 else 1.0
    up = np.array([0.0, 0.0, 1.0])
    rows = []
    for m_name, m in (("far", np.array([sign, 0.0, 0.0])), ("-y", np.array([0.0, -1.0, 0.0])),
                      ("+y", np.array([0.0, 1.0, 0.0]))):
        axis = np.cross(up, m)
        lip = task._pour_lip(m)
        l_rel = lip - ee[:3]
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            pv = f * l_rel
            for theta_deg in (90.0, 110.0):
                theta = np.deg2rad(theta_deg)
                th_ref = np.deg2rad(min(80.0, theta_deg))
                pivot = pour_point - gp._rot_axis(axis, th_ref) @ (l_rel - pv)
                ee_t = pivot - pv
                chain, p = [], ee[:3].copy()
                for leg in task._transit_legs(arm, ee_t):
                    p = p + leg
                    chain.append(p.tolist() + list(ee[3:]))
                for k in range(1, 11):
                    R = gp._rot_axis(axis, k * theta / 10)
                    chain.append((pivot + R @ (-pv)).tolist()
                                 + list(gp._quat_mul_wxyz(gp._mat_to_quat_wxyz(R), ee[3:])))
                feas = bool(task.screw_path_ok(arm, chain))
                lips = [pivot + gp._rot_axis(axis, np.deg2rad(a)) @ (l_rel - pv)
                        for a in np.linspace(60.0, theta_deg, 6)]
                sweep = float(max(np.linalg.norm(a[:2] - b[:2]) for a in lips for b in lips))
                ee_travel = float(np.linalg.norm(chain[-1][:3] - np.asarray(ee_t)))
                rows.append({"mouth": m_name, "f": f, "theta": theta_deg, "ok": feas,
                             "lip_sweep_cm": round(100 * sweep, 1), "ee_tilt_travel_cm": round(100 * ee_travel, 1)})
    rec = {"embodiment": args.embodiment, "seed": args.seed, "cp": cp, "arm": arm,
           "lift_ok": ok_lift, "rows": rows}
    for r in rows:
        print(f"{args.embodiment} s{args.seed} cp{cp} {r['mouth']:>3} f={r['f']:.2f} th={r['theta']:.0f} "
              f"ok={int(r['ok'])} lip_sweep={r['lip_sweep_cm']:5.1f}cm ee_travel={r['ee_tilt_travel_cm']:5.1f}cm")
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
