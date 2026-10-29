"""T1 safe mode: which part of the planner's world blocks a failed plan (2026-10-28 pilot: place_burger_fries lost
10 of 12 seeds and place_bread_basket 6 of 12 to "ik_fail" on the hand's last straight descent onto a flat object,
whether or not an extra object stood near it).

Runs vla_finetuning.items.collect_items on the given seeds with one change: when a plan of either arm fails, the
same plan is asked again in seven variant worlds before the ladder goes on, and the answers are printed
("[t1_debug] {...}", at most --max-reports per process):

    no_table          the protected items, no table          no_items       the table only
    empty             nothing                                 all_no_hand    every obstacle, the hand + held object not modelled
    empty_no_hand     nothing, no hand model                  all_unconstr   every obstacle, the straight-line constraint dropped
    empty_no_hand_unconstr   nothing, no hand model, no constraint

The real world is written back afterwards (the next try re-pushes it); nothing kept by this run is a demo (own root).
    python envs/scripts/item_envs/t1_safe_debug.py --root <abs dir> -- <collect_items args without --root>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import main.lang  # noqa: E402,F401  (the item-env path)
from main.item_envs import safe_plan  # noqa: E402

VARIANTS = (   # label, keep obstacle?, hand model, keep the constraint
    ("no_table", lambda o: o["name"] != safe_plan.TABLE_NAME, "full", True),
    ("no_items", lambda o: o["name"] == safe_plan.TABLE_NAME, "full", True),
    ("empty", lambda o: False, "full", True),
    ("all_no_hand", lambda o: True, "none", True),
    ("empty_no_hand", lambda o: False, "none", True),
    ("all_unconstr", lambda o: True, "full", False),
    ("empty_no_hand_unconstr", lambda o: False, "none", False),
)


def install(max_reports: int, margin: float = 0.002):
    orig = safe_plan.install_plan_hooks
    seen = {"n": 0}

    def patched(env, *a, world=None, **k):
        robot = env.robot
        for name in ("left_plan_path", "right_plan_path"):
            fn = getattr(robot, name)
            arm = name.split("_")[0]
            planner = getattr(robot, f"{arm}_planner")

            def wrapped(*aa, _fn=fn, _arm=arm, _planner=planner, **kk):
                res = _fn(*aa, **kk)
                if world is None or not safe_plan._failed(res) or seen["n"] >= max_reports:
                    return res
                seen["n"] += 1
                goal = aa[0] if aa else kk.get("target_pose")
                rep = {"arm": _arm, "skill": getattr(env.task, "_safe_plan_ctx", None),
                       "goal": safe_plan._pose_list(goal), "constrained": kk.get("constraint_pose") is not None,
                       "table": world.stats.get("table"),
                       "obstacles": [[o["name"], [round(float(x), 3) for x in o["p"]]] for o in world.obstacles(margin)]}
                for label, keep, model, constr in VARIANTS:
                    obs = [o for o in world.obstacles(margin) if keep(o)]
                    wb = safe_plan.world_in_base(_planner, obs)
                    for mg in (_planner.motion_gen, _planner.motion_gen_batch):
                        safe_plan.load_world(mg, wb)
                    s = safe_plan.sync_arm(env, _arm, margin, model)
                    kk2 = dict(kk)
                    if not constr:
                        kk2["constraint_pose"] = None
                    try:
                        rep[label] = "ok" if not safe_plan._failed(_fn(*aa, **kk2)) else "fail"
                    except Exception as exc:  # noqa: BLE001
                        rep[label] = f"error {type(exc).__name__}: {str(exc)[:120]}"
                    if label == "no_table":
                        rep["sync"] = {kk_: s.get(kk_) for kk_ in ("held", "n_payload", "n_hand")}
                world._pushed = {}   # the next try writes the real world again
                print("[t1_debug] " + json.dumps(rep), flush=True)
                return res

            setattr(robot, name, wrapped)
        return orig(env, *a, world=world, **k)

    safe_plan.install_plan_hooks = patched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-reports", type=int, default=4)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    rest = [x for x in args.rest if x != "--"]
    install(args.max_reports)
    from vla_finetuning.items import collect_items

    return collect_items.main(["--root", args.root, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
