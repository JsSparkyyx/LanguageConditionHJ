"""ur5-wsg and piper in one original RoboTwin task scene, both arms of each robot at the SAME joint values, seen by the
observer camera (Yuxuan 2026-11-05: "just ensure the arms are in the observer view and no self collision, and both arm
in the same joint value"; the two robots may differ). Ray traced on an A40.

    scripts/lang/in_container.sh python -m vla_finetuning.items.render_same_arm_pose --out /abs/imgs/same_arm_pose

Poses tried per robot, both arms at the same values: "zero" (every arm joint 0, the first request) and "home" (the
embodiment's RoboTwin homestate, config.yml; equal for the two arms in both robots: piper all 0, ur5
[-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0]). Checks per pose:
  - inside the joint limits;
  - no contact: the physics engine stepped once, then every contact with separation <= 0 that involves a robot link
    is listed (it reports a link touching its own arm too: the 11-05 render_robot_poses run caught ur5 wrist_2_link
    16 mm inside forearm_link that the collision spheres missed);
  - in the observer view: every cuRobo collision sphere of both arms (bases included) projects fully inside the
    observer picture, 8 px from its edge (the sphere outlines are drawn in debug/ to check the projection).
Same scene for every robot (render_same_joints.py's helpers): one seed, the original RoboTwin layout, every movable
object set to the first robot's poses. The chosen pose per robot = "zero" if it passes every check, else "home", else
"search" (search_same_q: shared values closest to home with both arms in view, ending with the physics check).

Outputs in <out>: <emb>_<pose>_observer.png and _head.png (640 x 480), robots_observer.png (the chosen poses side by
side), debug/<emb>_<pose>_observer_spheres.png, pose.json (values, limits, contacts, picture margins, the choice).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vla_finetuning.items.render_robot_poses import (ROBOTWIN, centres, contacts_after_step, edge_px, head_rgb,
                                                     link_mats, project, set_arms, sphere_model)
from vla_finetuning.items.render_same_joints import SIDES, build_env, handles, load_scene, save_scene

EDGE_PX = 8


def search_same_q(env, H, emb, K, E, Wo, Ho, home, rng, poses, n: int, max_physics: int = 400):
    """Joint values shared by both arms, closest to `home` first, with every collision sphere of both arms inside the
    observer picture (8 px), no sphere overlap within an arm (render_robot_poses.self_check: not parent-child, not
    structural) or between the arms (1 cm); then the physics contact check (the ground truth) on up to max_physics of
    them in that order. Returns (q, info) or (None, info)."""
    from vla_finetuning.items.render_robot_poses import pair_gaps, self_check

    lo = np.maximum(np.maximum(H["left"]["lim"][:, 0], H["right"]["lim"][:, 0]), -np.pi)
    hi = np.minimum(np.minimum(H["left"]["lim"][:, 1], H["right"]["lim"][:, 1]), np.pi)
    parts = [home + rng.normal(0, sig, size=(n // 10, len(home))) for sig in (0.05, 0.1, 0.2, 0.4, 0.8)]
    parts.append(rng.uniform(lo, hi, size=(n // 2, len(home))))
    Q = np.clip(np.concatenate(parts), lo, hi)
    ok = np.ones(len(Q), bool)
    C, R = {}, {}
    for side in SIDES:
        h = H[side]
        li, ctr, rad = sphere_model(h, emb)
        C[side] = centres(link_mats(h, Q), li, ctr)
        R[side] = rad
        a, b, _ = self_check(h, li, rad, C[side][:3000], len(h["lnames"]))
        ok &= edge_px(K, E, C[side], rad, Wo, Ho).min(1) >= EDGE_PX
        ok &= (pair_gaps(C[side], rad, a, b).min(1) >= 0) if len(a) else True
    cnt = {"sampled": len(Q), "in_view_no_sphere_overlap_in_arm": int(ok.sum())}
    idx = np.flatnonzero(ok)
    g_arm = np.array([(np.linalg.norm(C["left"][i][:, None] - C["right"][i][None], axis=-1)
                       - R["left"][:, None] - R["right"][None]).min() for i in idx])
    idx = idx[g_arm >= 0.01]
    cnt["arms_1cm_apart"] = len(idx)
    idx = idx[np.argsort(np.abs(Q[idx] - home).sum(1))]
    tried = 0
    for i in idx[:max_physics]:
        tried += 1
        set_arms(H, {side: Q[i] for side in SIDES})
        c = contacts_after_step(env, H)
        load_scene(env, poses)
        if not c:
            cnt["physics_checked"] = tried
            return Q[i], cnt
    cnt["physics_checked"] = tried
    return None, cnt


def homestate(emb: str) -> list:
    import yaml

    return yaml.safe_load((ROBOTWIN / "assets/embodiments" / emb / "config.yml").read_text())["homestate"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="place_container_plate")
    ap.add_argument("--seed", type=int, default=509000)
    ap.add_argument("--embodiments", nargs="+", default=["ur5-wsg", "piper"])
    ap.add_argument("--cam", default="Large_D435", choices=["D435", "Large_D435"])
    ap.add_argument("--n", type=int, default=100000, help="joint vectors tried when neither zero nor home passes")
    args = ap.parse_args(argv)
    out = Path(args.out)
    rng = np.random.default_rng(0)
    (out / "debug").mkdir(parents=True, exist_ok=True)

    import main.lang  # noqa: F401  (main.item_envs path; RoboTwin on the path)
    from main.envs.record import observer_rgb
    from PIL import Image, ImageDraw

    report = {"task": args.task, "seed": args.seed, "robots": {}}
    chosen_imgs, poses, rc = [], None, 0
    for emb in args.embodiments:
        env = build_env(args.task, emb, args.seed, args.cam)
        try:
            if poses is None:
                poses = save_scene(env)
            else:
                print(f"[same_arm] {emb}: objects moved by up to {100 * load_scene(env, poses):.2f} cm to the "
                      "first robot's poses", flush=True)
            t = env.task
            H = handles(env)
            n = len(H["left"]["idx"])
            hs = homestate(emb)
            cands = {"zero": np.zeros(n)}
            if np.allclose(hs[0], hs[1]):
                cands["home"] = np.asarray(hs[0], float)
            else:
                print(f"[same_arm] {emb}: homestate differs between the arms {hs}, not tried", flush=True)
            obs_cam = t.cameras.observer_camera
            K = np.asarray(obs_cam.get_intrinsic_matrix(), float)
            E = np.asarray(obs_cam.get_extrinsic_matrix(), float)
            rep = {"poses": {}}
            imgs = {}

            def evaluate(name, q):
                set_arms(H, {side: q for side in SIDES})
                obs = observer_rgb(t)
                Wo, Ho = obs.shape[1], obs.shape[0]
                Image.fromarray(obs).save(out / f"{emb}_{name}_observer.png")
                Image.fromarray(head_rgb(t)).save(out / f"{emb}_{name}_head.png")
                imgs[name] = obs
                dbg = Image.fromarray(obs.copy())
                dr = ImageDraw.Draw(dbg)
                row = {"q_rad": [round(float(v), 4) for v in q], "q_deg": [round(float(np.degrees(v)), 1) for v in q],
                       "arms": {}}
                in_view = True
                for side, col in (("left", (255, 60, 60)), ("right", (60, 160, 255))):
                    h = H[side]
                    li, ctr, rad = sphere_model(h, emb)
                    C = centres(link_mats(h, q[None]), li, ctr)                  # [1, S, 3]
                    e = edge_px(K, E, C, rad, Wo, Ho)[0]
                    u, v, z = project(K, E, C[0])
                    for uu, vv, zz, r in zip(u, v, z, rad):
                        if zz > 0.05:
                            rp = K[0, 0] * r / zz
                            dr.ellipse([uu - rp, vv - rp, uu + rp, vv + rp], outline=col)
                    lo, hi = h["lim"][:, 0], h["lim"][:, 1]
                    worst = int(np.argmin(e))
                    row["arms"][side] = {
                        "joints": h["names"], "limits_rad": np.round(h["lim"], 3).tolist(),
                        "inside_limits": bool(((q >= lo - 1e-6) & (q <= hi + 1e-6)).all()),
                        "picture_edge_px": round(float(e.min()), 1),
                        "closest_to_edge": h["lnames"][li[worst]]}
                    in_view &= bool(e.min() >= EDGE_PX)
                dbg.save(out / "debug" / f"{emb}_{name}_observer_spheres.png")
                set_arms(H, {side: q for side in SIDES})
                row["contacts_after_one_step"] = contacts_after_step(env, H)
                load_scene(env, poses)
                row["in_view"] = in_view
                row["inside_limits"] = all(row["arms"][s]["inside_limits"] for s in SIDES)
                row["no_contact"] = not row["contacts_after_one_step"]
                row["passes"] = row["in_view"] and row["inside_limits"] and row["no_contact"]
                rep["poses"][name] = row
                print(f"[same_arm] {emb} {name}: {json.dumps(row)}", flush=True)
                return obs.shape[1], obs.shape[0]

            for name, q in cands.items():
                Wo, Ho = evaluate(name, q)
            pick = next((nm for nm in ("zero", "home") if nm in rep["poses"] and rep["poses"][nm]["passes"]), None)
            if pick is None:
                home_q = cands.get("home", np.zeros(n))
                q, cnt = search_same_q(env, H, emb, K, E, Wo, Ho, home_q, rng, poses, args.n)
                rep["search"] = cnt
                print(f"[same_arm] {emb}: search {json.dumps(cnt)}", flush=True)
                if q is not None:
                    evaluate("search", q)
                    if rep["poses"]["search"]["passes"]:
                        pick = "search"
            rep["chosen"] = pick
            if pick is None:
                print(f"[same_arm] {emb}: no pose passes every check", flush=True)
                rc = 1
            else:
                chosen_imgs.append(imgs[pick])
            report["robots"][emb] = rep
        finally:
            env.close()
    if len(chosen_imgs) > 1:
        gap = np.full((chosen_imgs[0].shape[0], 8, 3), 255, np.uint8)
        parts = [p for img in chosen_imgs for p in (img, gap)][:-1]
        Image.fromarray(np.concatenate(parts, axis=1)).save(out / "robots_observer.png")
    (out / "pose.json").write_text(json.dumps(report, indent=1))
    print(f"[same_arm] wrote {out}/pose.json", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
