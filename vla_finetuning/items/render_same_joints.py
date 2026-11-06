"""Two robots in one original RoboTwin task scene, both at the SAME arm joint values, in a pose that looks natural on
both (Yuxuan 2026-11-05: ur5-wsg and piper, 6 arm joints each; the agent view = the head camera, identical in both
robots' RoboTwin configs). Ray traced on an A40.

    scripts/lang/in_container.sh python -m vla_finetuning.items.render_same_joints --out /abs/imgs/same_joints

The same values do not give the same shape: the robots differ in links, joint zeros and mounting (ur5's base faces
sideways, piper's faces the table), and their joint limits overlap only at shoulder = 0. So the values are SEARCHED:
random joint vectors (then a local refinement around the best), each scored on both robots with the arm's link
origins in the world (sapien forward kinematics of the real scene):
  hard: no moving link origin within 6 cm of the table top over the table; the gripper 8-40 cm above the table, over
        the workspace, on its own arm's side, at least 15 cm in front of its base; the elbow above the gripper;
  soft: gripper 18 cm above the table at y = -0.15 (the front of the workspace; the bases stand at different
        depths, ur5 -0.65, piper -0.45), 22 cm off the middle on its arm's side, pointing down, joint-limit excess
        (weighted 3 per rad).
Left and right arms are searched separately (each robot's left arm gets q_left, its right arm q_right). The grippers
stay at the robot's home opening.

Same scene: the task is built with the same seed for every robot, the CrossSafe spawn stretch is off (the original
RoboTwin layout), settle off, and every movable object of the later robots is set to the first robot's pose
(the before-copy difference is printed). Nothing is simulated after the joints are set.

Outputs in <out>: cand<k>_<emb>_head.png / _observer.png (640 x 480) for the top --top candidates,
cand<k>_head_robots.png (the robots side by side), search.json (the chosen joint values, limits, scores).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TABLE_Z = 0.74
TABLE_X, TABLE_Y = (-0.62, 0.62), (-0.38, 0.38)     # the RoboTwin table with a margin
WORK_X, WORK_Y = (-0.45, 0.45), (-0.30, 0.20)        # where the gripper may hang
ARM_DISTANCE = {"ur5-wsg": 0.8}                     # the item runner's; others the Env default 0.6
SIDES = ("left", "right")


def build_env(task: str, emb: str, seed: int, cam: str):
    from main.envs import env as E
    from main.envs import obstacle

    orig = E._task_args

    def task_args(*a, **k):
        args_ = orig(*a, **k)
        args_["camera"]["head_camera_type"] = cam
        args_["camera"]["wrist_camera_type"] = cam
        return args_

    E._task_args = task_args
    stretch = obstacle.stretch_task_spawns
    obstacle.stretch_task_spawns = lambda env: None            # the original RoboTwin layout
    try:
        return E.Env(task, emb, seed, arm_distance=ARM_DISTANCE.get(emb, 0.6), settle=False)
    finally:
        obstacle.stretch_task_spawns = stretch
        E._task_args = orig


def handles(env) -> dict:
    rob = env.robot
    out = {}
    for side in SIDES:
        ent = getattr(rob, f"{side}_entity")
        joints = ent.get_active_joints()
        names = [j.get_name() for j in joints]
        idx = [names.index(n) for n in getattr(rob, f"{side}_arm_joints_name")]
        lim = np.array([np.asarray(joints[i].get_limits(), float).reshape(-1, 2)[0] for i in idx])
        links = list(ent.get_links())
        lnames = [l.get_name() for l in links]
        ee = getattr(rob, f"{side}_ee")
        try:
            ee_link = ee.get_child_link()
            ee_i = lnames.index(ee_link.get_name())
        except Exception:  # noqa: BLE001
            ee_i = max(i for i, n in enumerate(lnames) if "finger" not in n and "gripper" not in n)
        tip_i = list(range(ee_i + 1, len(links)))
        out[side] = dict(ent=ent, idx=idx, lim=lim, links=links, lnames=lnames, ee_i=ee_i, tip_i=tip_i,
                         q0=np.asarray(ent.get_qpos(), dtype=np.float64), names=[names[i] for i in idx])
    return out


def link_points(h: dict, Q: np.ndarray) -> np.ndarray:
    """[N, n_links, 3] world link origins with the arm joints at each row of Q."""
    ent, idx, q0, links = h["ent"], h["idx"], h["q0"], h["links"]
    P = np.empty((len(Q), len(links), 3))
    for k, q in enumerate(Q):
        qq = q0.copy()
        qq[idx] = q
        ent.set_qpos(qq.astype(np.float32))
        for m, l in enumerate(links):
            P[k, m] = l.get_pose().p
    ent.set_qpos(h["q0"].astype(np.float32))
    return P


def arm_score(P: np.ndarray, h: dict, side: str, Q: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict]:
    pts = P[:, moving]
    ee = P[:, h["ee_i"]]
    tip = P[:, h["tip_i"]].mean(1) if h["tip_i"] else ee
    base = P[:, 0]
    over = ((pts[..., 0] > TABLE_X[0]) & (pts[..., 0] < TABLE_X[1])
            & (pts[..., 1] > TABLE_Y[0]) & (pts[..., 1] < TABLE_Y[1]))
    clear = np.where(over, pts[..., 2] - TABLE_Z, np.inf).min(1)
    tz = tip[:, 2] - TABLE_Z
    reach = tip[:, 1] - base[:, 1]
    own = tip[:, 0] < -0.03 if side == "left" else tip[:, 0] > 0.03
    inside = ((tip[:, 0] > WORK_X[0]) & (tip[:, 0] < WORK_X[1]) & (tip[:, 1] > WORK_Y[0]) & (tip[:, 1] < WORK_Y[1]))
    elbow_up = pts[..., 2].max(1) >= tip[:, 2] + 0.05
    ok = (clear >= 0.06) & (tz >= 0.08) & (tz <= 0.40) & inside & own & (reach >= 0.15) & elbow_up
    d = tip - ee
    dz = d[:, 2] / np.maximum(np.linalg.norm(d, axis=1), 1e-9) if h["tip_i"] else np.zeros(len(P))
    lo, hi = h["lim"][:, 0], h["lim"][:, 1]
    viol = (np.clip(lo - Q, 0, None) + np.clip(Q - hi, 0, None)).sum(1)
    side_x = -0.22 if side == "left" else 0.22
    s = (2.0 * np.abs(tz - 0.18) + np.abs(tip[:, 1] + 0.15) + 0.5 * np.abs(tip[:, 0] - side_x) + 0.3 * (1 + dz)
         + 3.0 * viol)
    s = np.where(ok, s, np.inf)
    return s, dict(tip=tip, viol=viol, clear=clear, tz=tz, reach=reach, dz=dz)


def evaluate(task, emb, seed, cam, qsets: dict) -> dict:
    """{side: [N, n_links, 3]} per joint set, one env build."""
    env = build_env(task, emb, seed, cam)
    try:
        H = handles(env)
        res = {side: link_points(H[side], qsets[side]) for side in SIDES}
        meta = {side: dict(lim=H[side]["lim"], ee_i=H[side]["ee_i"], tip_i=H[side]["tip_i"],
                           lnames=H[side]["lnames"], names=H[side]["names"]) for side in SIDES}
        return dict(P=res, meta=meta)
    finally:
        env.close()


def score_all(evals: dict, Q: np.ndarray, side: str) -> tuple[np.ndarray, dict]:
    total = np.zeros(len(Q))
    per = {}
    for emb, ev in evals.items():
        P = ev["P"][side]
        moving = P.std(0).max(1) > 1e-4                     # links that move with the arm joints
        h = ev["meta"][side]
        s, info = arm_score(P, h, side, Q, moving)
        total = total + s
        per[emb] = (s, info)
    return total, per


def distinct_top(Q: np.ndarray, s: np.ndarray, k: int, sep: float = 0.6) -> list[int]:
    order = [i for i in np.argsort(s) if np.isfinite(s[i])]
    keep = []
    for i in order:
        if all(np.abs(Q[i] - Q[j]).max() > sep for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return keep


def save_scene(env) -> dict:
    import sapien

    out = {}
    for e in env.task.scene.get_entities():
        if e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) is None:
            continue
        p = e.get_pose()
        out.setdefault(e.get_name(), []).append((np.asarray(p.p, float), np.asarray(p.q, float)))
    return out


def load_scene(env, poses: dict) -> float:
    """Sets every movable object to the recorded pose (by name, in order); returns the largest position change."""
    import sapien

    seen, worst = {}, 0.0
    for e in env.task.scene.get_entities():
        dyn = e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if dyn is None:
            continue
        name = e.get_name()
        k = seen.get(name, 0)
        seen[name] = k + 1
        if name not in poses or k >= len(poses[name]):
            print(f"[same_joints] no recorded pose for {name!r} #{k}", flush=True)
            continue
        p, q = poses[name][k]
        worst = max(worst, float(np.linalg.norm(np.asarray(e.get_pose().p) - p)))
        e.set_pose(sapien.Pose(p, q))
        try:
            dyn.set_linear_velocity(np.zeros(3))
            dyn.set_angular_velocity(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="place_container_plate")
    ap.add_argument("--seed", type=int, default=509000)
    ap.add_argument("--embodiments", nargs="+", default=["ur5-wsg", "piper"])
    ap.add_argument("--n", type=int, default=120000, help="random joint vectors per arm")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--cam", default="Large_D435", choices=["D435", "Large_D435"])
    ap.add_argument("--save-samples", action="store_true",
                    help="stage 1 only: write <out>/samples_<emb>.npz (Q, link origins, limits, link names) and stop")
    ap.add_argument("--q-file", default=None,
                    help="skip the search: render the candidates of this json {left: [[6 values], ...], right: ...}")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import main.lang  # noqa: F401  (main.item_envs path; RoboTwin on the path)
    from main.envs.record import observer_rgb
    from PIL import Image

    if args.q_file:
        qf = json.loads(Path(args.q_file).read_text())
        chosen = {side: [np.asarray(q, dtype=np.float64) for q in qf[side]] for side in SIDES}
        return render(args, chosen, out, observer_rgb, Image)
    rng = np.random.default_rng(0)
    # stage 1: random vectors in [-pi, pi] (wrist_1 of ur5 is +-1.75, the scores charge any excess)
    Q1 = {side: rng.uniform(-np.pi, np.pi, size=(args.n, 6)) for side in SIDES}
    ev1 = {emb: evaluate(args.task, emb, args.seed, args.cam, Q1) for emb in args.embodiments}
    if args.save_samples:
        for emb, ev in ev1.items():
            arrs = {}
            for side in SIDES:
                m = ev["meta"][side]
                arrs[f"{side}_Q"] = Q1[side].astype(np.float32)
                arrs[f"{side}_P"] = ev["P"][side].astype(np.float32)
                arrs[f"{side}_lim"] = m["lim"]
                arrs[f"{side}_ee_i"] = np.int64(m["ee_i"])
                arrs[f"{side}_tip_i"] = np.asarray(m["tip_i"], dtype=np.int64)
                arrs[f"{side}_lnames"] = np.asarray(m["lnames"])
                arrs[f"{side}_jnames"] = np.asarray(m["names"])
            np.savez_compressed(out / f"samples_{emb}.npz", **arrs)
            print(f"[same_joints] wrote samples_{emb}.npz", flush=True)
        return 0
    # stage 2: local refinement around the 40 best distinct vectors of each arm
    Q2 = {}
    for side in SIDES:
        s, _ = score_all(ev1, Q1[side], side)
        top = distinct_top(Q1[side], s, 40, sep=0.3)
        print(f"[same_joints] {side}: {int(np.isfinite(s).sum())} of {len(s)} feasible on every robot; "
              f"best {s[top[0]]:.3f}" if top else f"[same_joints] {side}: none feasible", flush=True)
        if not top:
            return 1
        base = Q1[side][top]
        Q2[side] = np.concatenate([base] + [base + rng.normal(0, sig, base.shape) for sig in (0.05, 0.12, 0.25)
                                            for _ in range(25)])
    ev2 = {emb: evaluate(args.task, emb, args.seed, args.cam, Q2) for emb in args.embodiments}
    chosen, report = {}, {"task": args.task, "seed": args.seed, "embodiments": args.embodiments, "arms": {}}
    for side in SIDES:
        s, per = score_all(ev2, Q2[side], side)
        top = distinct_top(Q2[side], s, args.top)
        chosen[side] = [Q2[side][i] for i in top]
        report["arms"][side] = []
        for i in top:
            row = {"q": [round(float(v), 4) for v in Q2[side][i]], "score": round(float(s[i]), 4), "robots": {}}
            for emb in args.embodiments:
                si, info = per[emb]
                m = ev2[emb]["meta"][side]
                row["robots"][emb] = {
                    "joints": m["names"], "limits": np.round(m["lim"], 3).tolist(),
                    "limit_excess_rad": round(float(info["viol"][i]), 4),
                    "gripper_xyz": np.round(info["tip"][i], 3).tolist(),
                    "gripper_height_cm": round(100 * float(info["tz"][i]), 1),
                    "reach_cm": round(100 * float(info["reach"][i]), 1),
                    "table_clearance_cm": round(100 * float(info["clear"][i]), 1),
                    "gripper_down": round(float(-info["dz"][i]), 2), "score": round(float(si[i]), 4)}
            report["arms"][side].append(row)
            print(f"[same_joints] {side} cand: {json.dumps(row)}", flush=True)
    (out / "search.json").write_text(json.dumps(report, indent=1))
    return render(args, chosen, out, observer_rgb, Image)


def render(args, chosen: dict, out: Path, observer_rgb, Image) -> int:
    """Every candidate k on every robot, the first robot's object poses copied to the others."""
    n_cand = min(len(chosen["left"]), len(chosen["right"]))
    heads = {k: [] for k in range(n_cand)}
    poses = None
    for emb in args.embodiments:
        env = build_env(args.task, emb, args.seed, args.cam)
        try:
            if poses is None:
                poses = save_scene(env)
            else:
                print(f"[same_joints] {emb}: objects moved by up to {100 * load_scene(env, poses):.2f} cm "
                      "to the first robot's poses", flush=True)
            H = handles(env)
            t = env.task
            for k in range(n_cand):
                for side in SIDES:
                    h = H[side]
                    qq = h["q0"].copy()
                    qq[h["idx"]] = chosen[side][k]
                    h["ent"].set_qpos(qq.astype(np.float32))
                    h["ent"].set_qvel(np.zeros_like(qq, dtype=np.float32))
                    for i, j in zip(h["idx"], [h["ent"].get_active_joints()[i] for i in h["idx"]]):
                        j.set_drive_target(float(qq[i]))
                t._update_render()
                obs = t.get_obs()["observation"]
                rgb = np.asarray(obs["head_camera"]["rgb"])[:, :, :3].astype(np.uint8)
                Image.fromarray(rgb).save(out / f"cand{k}_{emb}_head.png")
                Image.fromarray(observer_rgb(t)).save(out / f"cand{k}_{emb}_observer.png")
                heads[k].append(rgb)
                print(f"[same_joints] rendered cand{k} {emb}", flush=True)
        finally:
            env.close()
    for k, row in heads.items():
        if len(row) > 1:
            gap = np.full((row[0].shape[0], 8, 3), 255, np.uint8)
            parts = [p for img in row for p in (img, gap)][:-1]
            Image.fromarray(np.concatenate(parts, axis=1)).save(out / f"cand{k}_head_robots.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
