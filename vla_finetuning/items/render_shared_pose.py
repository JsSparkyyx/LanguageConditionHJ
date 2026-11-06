"""ur5-wsg and piper in one original RoboTwin task scene with ALL FOUR ARMS at the same joint values (Yuxuan
2026-11-05: "piper and ur5 set in the same joint value, the both arm of the same robot are in the same joint value"),
both arms of each robot wholly in the observer view and no contact. Ray traced on an A40.

    scripts/lang/in_container.sh python -m vla_finetuning.items.render_shared_pose --out /abs/imgs/shared_pose

The values are searched inside every arm's joint limits (their intersection; joint 2: ur5 shoulder_lift [-pi, 0] and
piper joint2 [0, pi] share only 0, so it is fixed at 0), closest to all zeros first (the first request):
  1. per robot, --n vectors (the same ones on every robot): every cuRobo collision sphere of both arms (bases
     included) fully inside the observer picture, 8 px from its edge; no sphere overlap within an arm
     (render_robot_poses.self_check: not parent-child, not structural) and the two arms' spheres 1 cm apart;
  2. the vectors passing on every robot, by distance to zero (sum of |q|): the physics engine stepped once per robot,
     no contact with separation <= 0 involving a robot link (it reports a link touching its own arm, a contact the
     spheres can miss); the first --top that pass on every robot are kept;
  3. each kept vector rendered on every robot (observer and head camera, 640 x 480), the spheres drawn in debug/.
With --ignore-limits the vectors are drawn from [-pi, pi] instead (the excess is reported).
Same scene for every robot (render_same_joints.py's helpers): one seed, the original RoboTwin layout, every movable
object set to the first robot's poses.

Outputs in <out>: cand<k>_<emb>_observer.png / _head.png, cand<k>_robots_observer.png (side by side),
debug/cand<k>_<emb>_observer_spheres.png, shared_pose.json (values, limits, counts, contacts, picture margins).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vla_finetuning.items.render_robot_poses import (centres, contacts_after_step, edge_px, head_rgb, link_mats,
                                                     pair_gaps, project, self_check, set_arms, sphere_model)
from vla_finetuning.items.render_same_joints import SIDES, build_env, handles, load_scene, save_scene

EDGE_PX = 8
ARM_GAP = 0.01


class Robot:
    """One env build with the scene copied from the first robot's poses."""

    def __init__(self, args, emb, poses):
        self.emb = emb
        self.env = build_env(args.task, emb, args.seed, args.cam)
        if poses is not None:
            moved = load_scene(self.env, poses)
            print(f"[shared] {emb}: objects moved by up to {100 * moved:.2f} cm to the first robot's poses", flush=True)
        self.t = self.env.task
        self.H = handles(self.env)
        cam = self.t.cameras.observer_camera
        self.K = np.asarray(cam.get_intrinsic_matrix(), float)
        self.E = np.asarray(cam.get_extrinsic_matrix(), float)
        try:
            self.W, self.Hh = int(cam.get_width()), int(cam.get_height())
        except Exception:  # noqa: BLE001
            from main.envs.record import observer_rgb
            img = observer_rgb(self.t)
            self.W, self.Hh = img.shape[1], img.shape[0]

    def close(self):
        self.env.close()


def prefilter(rb: Robot, Q: np.ndarray, chunk: int = 20000) -> np.ndarray:
    ok = np.ones(len(Q), bool)
    C, R = {}, {}
    for side in SIDES:
        h = rb.H[side]
        li, ctr, rad = sphere_model(h, rb.emb)
        C[side] = centres(link_mats(h, Q), li, ctr)
        R[side] = rad
        a, b, structural = self_check(h, li, rad, C[side][:3000], len(h["lnames"]))
        if structural:
            print(f"[shared] {rb.emb} {side}: structural sphere pairs not checked {structural}", flush=True)
        for s in range(0, len(Q), chunk):
            sl = slice(s, s + chunk)
            ok[sl] &= edge_px(rb.K, rb.E, C[side][sl], rad, rb.W, rb.Hh).min(1) >= EDGE_PX
            if len(a):
                ok[sl] &= pair_gaps(C[side][sl], rad, a, b).min(1) >= 0
    for i in np.flatnonzero(ok):
        g = (np.linalg.norm(C["left"][i][:, None] - C["right"][i][None], axis=-1)
             - R["left"][:, None] - R["right"][None]).min()
        ok[i] = g >= ARM_GAP
    return ok


def no_contact(rb: Robot, q: np.ndarray, poses) -> list:
    set_arms(rb.H, {side: q for side in SIDES})
    c = contacts_after_step(rb.env, rb.H)
    load_scene(rb.env, poses)
    return c


def render(rb: Robot, q: np.ndarray, k: int, out: Path, poses, Image, ImageDraw, observer_rgb) -> tuple[dict, np.ndarray]:
    set_arms(rb.H, {side: q for side in SIDES})
    obs = observer_rgb(rb.t)
    Image.fromarray(obs).save(out / f"cand{k}_{rb.emb}_observer.png")
    Image.fromarray(head_rgb(rb.t)).save(out / f"cand{k}_{rb.emb}_head.png")
    dbg = Image.fromarray(obs.copy())
    dr = ImageDraw.Draw(dbg)
    row = {"arms": {}}
    for side, col in (("left", (255, 60, 60)), ("right", (60, 160, 255))):
        h = rb.H[side]
        li, ctr, rad = sphere_model(h, rb.emb)
        C = centres(link_mats(h, q[None]), li, ctr)
        e = edge_px(rb.K, rb.E, C, rad, rb.W, rb.Hh)[0]
        u, v, z = project(rb.K, rb.E, C[0])
        for uu, vv, zz, r in zip(u, v, z, rad):
            if zz > 0.05:
                rp = rb.K[0, 0] * r / zz
                dr.ellipse([uu - rp, vv - rp, uu + rp, vv + rp], outline=col)
        lo, hi = h["lim"][:, 0], h["lim"][:, 1]
        row["arms"][side] = {
            "joints": h["names"], "limits_rad": np.round(h["lim"], 3).tolist(),
            "limit_excess_rad": round(float((np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)).sum()), 4),
            "picture_edge_px": round(float(e.min()), 1), "closest_to_edge": h["lnames"][li[int(np.argmin(e))]]}
    dbg.save(out / "debug" / f"cand{k}_{rb.emb}_observer_spheres.png")
    set_arms(rb.H, {side: q for side in SIDES})
    row["contacts_after_one_step"] = contacts_after_step(rb.env, rb.H)
    load_scene(rb.env, poses)
    return row, obs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="place_container_plate")
    ap.add_argument("--seed", type=int, default=509000)
    ap.add_argument("--embodiments", nargs="+", default=["ur5-wsg", "piper"])
    ap.add_argument("--cam", default="Large_D435", choices=["D435", "Large_D435"])
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-physics", type=int, default=800, help="physics checks on the first robot at most")
    ap.add_argument("--ignore-limits", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "debug").mkdir(parents=True, exist_ok=True)

    import main.lang  # noqa: F401  (main.item_envs path; RoboTwin on the path)
    from main.envs.record import observer_rgb
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    embs = args.embodiments
    report = {"task": args.task, "seed": args.seed, "embodiments": embs, "ignore_limits": args.ignore_limits}

    # 0. the scene poses and every arm's limits
    poses, lims = None, []
    for emb in embs:
        rb = Robot(args, emb, poses)
        try:
            if poses is None:
                poses = save_scene(rb.env)
            lims += [rb.H[side]["lim"] for side in SIDES]
        finally:
            rb.close()
    if args.ignore_limits:
        lo, hi = np.full(len(lims[0]), -np.pi), np.full(len(lims[0]), np.pi)
    else:
        lo = np.maximum(np.max([l[:, 0] for l in lims], 0), -np.pi)
        hi = np.minimum(np.min([l[:, 1] for l in lims], 0), np.pi)
    report["common_limits_rad"] = np.round(np.stack([lo, hi], 1), 4).tolist()
    print(f"[shared] common limits {report['common_limits_rad']}", flush=True)
    if (hi < lo - 1e-9).any():
        print("[shared] the limits do not overlap on some joint; rerun with --ignore-limits", flush=True)
        (out / "shared_pose.json").write_text(json.dumps(report, indent=1))
        return 1
    Q = rng.uniform(lo, hi, size=(args.n, len(lo)))
    Q[0] = np.clip(0.0, lo, hi)                                   # all zeros (clipped) is tried too

    # 1. sphere prefilter on every robot
    ok, counts = np.ones(len(Q), bool), {"sampled": len(Q)}
    for emb in embs:
        rb = Robot(args, emb, poses)
        try:
            m = prefilter(rb, Q)
        finally:
            rb.close()
        counts[f"{emb}: arms in view, spheres apart"] = int(m.sum())
        ok &= m
    counts["on every robot"] = int(ok.sum())
    order = np.flatnonzero(ok)
    order = order[np.argsort(np.abs(Q[order]).sum(1))]
    print(f"[shared] {json.dumps(counts)}", flush=True)

    # 2. physics contact check, robot by robot on the survivors of the previous one
    alive = list(order[: args.max_physics])
    for emb in embs:
        rb = Robot(args, emb, poses)
        keep = []
        try:
            for i in alive:
                if not no_contact(rb, Q[i], poses):
                    keep.append(i)
                    if emb == embs[-1] and len(keep) == args.top:
                        break
        finally:
            rb.close()
        counts[f"{emb}: no contact (of {len(alive)} checked)"] = len(keep)
        alive = keep
    report["counts"] = counts
    print(f"[shared] {json.dumps(counts)}", flush=True)
    chosen = alive[: args.top]
    report["candidates"] = [{"q_rad": [round(float(v), 4) for v in Q[i]],
                             "q_deg": [round(float(np.degrees(v)), 1) for v in Q[i]], "robots": {}} for i in chosen]
    if not chosen:
        print("[shared] no shared value passes on every robot", flush=True)
        (out / "shared_pose.json").write_text(json.dumps(report, indent=1))
        return 1

    # 3. render
    imgs = {k: [] for k in range(len(chosen))}
    for emb in embs:
        rb = Robot(args, emb, poses)
        try:
            for k, i in enumerate(chosen):
                row, obs = render(rb, Q[i], k, out, poses, Image, ImageDraw, observer_rgb)
                report["candidates"][k]["robots"][emb] = row
                imgs[k].append(obs)
                print(f"[shared] cand{k} {emb}: {json.dumps(row)}", flush=True)
        finally:
            rb.close()
    for k, row in imgs.items():
        gap = np.full((row[0].shape[0], 8, 3), 255, np.uint8)
        parts = [p for img in row for p in (img, gap)][:-1]
        Image.fromarray(np.concatenate(parts, axis=1)).save(out / f"cand{k}_robots_observer.png")
    (out / "shared_pose.json").write_text(json.dumps(report, indent=1))
    print(f"[shared] wrote {out}/shared_pose.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
