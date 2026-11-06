"""ur5-wsg and piper in one original RoboTwin task scene, each robot in its OWN natural pose (Yuxuan 2026-11-05: the
two poses may differ; no part of a robot may touch itself or its other arm, and both arms must be in the agent view =
the head camera picture). Ray traced on an A40.

    scripts/lang/in_container.sh python -m vla_finetuning.items.render_robot_poses --out /abs/imgs/robot_poses

Same scene for every robot (render_same_joints.py's helpers): the task built with one seed, the CrossSafe spawn
stretch off (the original RoboTwin layout), every movable object set to the first robot's poses. Per robot, per arm:
  1. --n random joint vectors inside the joint limits; link origins by sapien forward kinematics; kept when the pose
     looks natural (render_same_joints.arm_score: gripper 8-40 cm above the table, in front of the base, over the
     workspace on its own side, elbow above the gripper, every moving link origin 6 cm above the table top) and the
     origins of the links past arm joint --view-from (3: ur5 wrist_2_link on, piper link4 on = wrist and gripper)
     project into the head camera picture; then 90 Gaussian steps (0.03 / 0.08 / 0.2 rad) around each of the 300
     best distinct passing poses, scored the same; the best --keep by arm_score's target (gripper 18 cm up at
     y = -0.15, 22 cm off the middle, pointing down, here weighted twice) go on.
     The WHOLE arm cannot be in view: the head camera (fovy 37 deg) stands at y = -0.45, 61 cm above the table,
     looking forward and 53 deg down, over piper's bases and in front of ur5's (y = -0.65). Of 120,000 random ur5 poses
     inside the joint limits 10 (left) and 0 (right) have even the whole forearm (from the elbow) in the picture
     (samples of render_same_joints --save-samples); the share of the arm in view is preferred (step 2), not required.
  2. The cuRobo collision spheres of every link (the embodiment's collision yml, plus 1.8 cm spheres on the gripper
     links it lacks, as main/envs/distance.py): no overlap between the spheres of two links that are not parent and
     child, except link pairs whose spheres overlap in over 30% of random poses (structural: they always touch in the
     sphere model; listed in poses.json); every moving sphere over the table 1 cm above it; every sphere 1.5 cm from
     every task object's box (container and plate); every sphere past arm joint --view-from fully inside the picture,
     8 px from its edge; ranked by the step-1 score + 0.5 x the share of the arm's moving spheres outside the picture.
  3. Pairs of the best --pairs left and right poses: the two arms' spheres 3 cm apart; cost = both arm scores + the
     mirror asymmetry of the two grippers about the table's middle (x = 0).
The best pair and up to --top - 1 distinct others are rendered (head camera and observer, 640 x 480); the head picture
again with the projected spheres drawn (debug/, to check the projection); then the physics engine is stepped once and
every contact with separation <= 0 that involves a robot link is listed (a second check; the arms and the objects are
put back after it).

Outputs in <out>: cand<k>_<emb>_head.png, cand<k>_<emb>_observer.png, cand<k>_head_robots.png (the robots side by
side), debug/cand<k>_<emb>_head_spheres.png, poses.json (joint values, limits, every check's margin, the contacts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vla_finetuning.items.render_same_joints import (SIDES, TABLE_X, TABLE_Y, TABLE_Z, arm_score, build_env,
                                                     distinct_top, handles, link_points, save_scene, load_scene)

ROBOTWIN = Path(__file__).resolve().parents[3] / "RoboTwin"      # ../RoboTwin next to the repo (AGENTS.md)
STRUCTURAL = 0.30          # a link pair overlapping in more random poses than this is never checked
TABLE_GAP, OBJ_GAP, ARM_GAP, EDGE_PX = 0.01, 0.015, 0.03, 8
GRIP_R = 0.018             # main/envs/distance.py's sphere on gripper links missing from the yml


# ------------------------------------------------------------------ geometry ----
def sphere_model(h: dict, emb: str):
    """(link index [S], local centre [S, 3], radius [S]) of the arm's collision spheres."""
    import yaml
    from main.envs.distance import is_gripper_link
    from main.envs.obstacle import COLLISION_YML

    cfg = yaml.safe_load((ROBOTWIN / "assets/embodiments" / COLLISION_YML[emb]).read_text())["collision_spheres"]
    names = h["lnames"]
    li, ctr, rad = [], [], []
    for n, specs in cfg.items():
        if n not in names:
            continue
        for s in specs:
            if float(s["radius"]) > 0:
                li.append(names.index(n))
                ctr.append([float(v) for v in s["center"]])
                rad.append(float(s["radius"]))
    for i, n in enumerate(names):
        if n not in cfg and is_gripper_link(n, emb):
            li.append(i)
            ctr.append([0.0, 0.0, 0.0])
            rad.append(GRIP_R)
    return np.asarray(li), np.asarray(ctr, float), np.asarray(rad, float)


def link_mats(h: dict, Q: np.ndarray) -> np.ndarray:
    """[N, n_links, 4, 4] world link poses with the arm joints at each row of Q."""
    ent, idx, q0, links = h["ent"], h["idx"], h["q0"], h["links"]
    M = np.empty((len(Q), len(links), 4, 4))
    for k, q in enumerate(Q):
        qq = q0.copy()
        qq[idx] = q
        ent.set_qpos(qq.astype(np.float32))
        for m, l in enumerate(links):
            M[k, m] = l.get_pose().to_transformation_matrix()
    ent.set_qpos(q0.astype(np.float32))
    return M


def centres(M: np.ndarray, li: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    Ml = M[:, li]                      # [N, S, 4, 4] (one fancy index only: mixed ones move the axes)
    return np.einsum("nsij,sj->nsi", Ml[..., :3, :3], ctr) + Ml[..., :3, 3]


def chains(h: dict):
    """chain(link name) -> [the link, its parent, ..., the root] (joints, fixed ones included)."""
    par = {}
    for j in h["ent"].get_joints():
        c, p = j.get_child_link(), j.get_parent_link()
        if c is not None and p is not None:
            par[c.get_name()] = p.get_name()

    def chain(n):
        out = [n]
        while out[-1] in par:
            out.append(par[out[-1]])
        return out

    return chain


def outer_links(h: dict, j: int) -> np.ndarray:
    """[L] bool: the links past arm joint j (0-based; j = 3: ur5 wrist_1_joint -> wrist_2_link, piper joint4 ->
    link4, the wrist and the gripper)."""
    chain = chains(h)
    root = h["ent"].get_active_joints()[h["idx"][j]].get_child_link().get_name()
    return np.array([root in chain(n) for n in h["lnames"]])


def hops(h: dict) -> np.ndarray:
    """[L, L] link distance in the kinematic tree (joints, fixed ones included)."""
    names = h["lnames"]
    chain = chains(h)
    L = len(names)
    H = np.full((L, L), 99)
    for a in range(L):
        ca = chain(names[a])
        for b in range(L):
            cb = chain(names[b])
            common = [x for x in ca if x in cb]
            if common:
                H[a, b] = ca.index(common[0]) + cb.index(common[0])
    return H


def pair_gaps(C: np.ndarray, rad: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[N, P] surface gaps of the sphere pairs (a, b)."""
    return np.linalg.norm(C[:, a] - C[:, b], axis=-1) - rad[a] - rad[b]


def self_check(h, li, rad, C_rand, L):
    """The link pairs to check (parent-child and structural ones out) and the structural list."""
    H = hops(h)
    i, j = np.triu_indices(len(li), 1)
    diff = li[i] != li[j]
    i, j = i[diff], j[diff]
    G = pair_gaps(C_rand, rad, i, j)
    check = np.zeros((L, L), bool)
    structural = []
    for la in range(L):
        for lb in range(la + 1, L):
            m = ((li[i] == la) & (li[j] == lb)) | ((li[i] == lb) & (li[j] == la))
            if not m.any() or H[la, lb] < 2:
                continue
            frac = float((G[:, m].min(1) < 0).mean())
            if frac > STRUCTURAL:
                structural.append([h["lnames"][la], h["lnames"][lb], round(frac, 2)])
            else:
                check[la, lb] = check[lb, la] = True
    keep = check[li[i], li[j]]
    return i[keep], j[keep], structural


def object_boxes(env) -> list:
    """Every RoboTwin Actor the task keeps as an attribute (container AND plate here), table / walls / ground out."""
    from main.envs.distance import _obb_from_actor
    from main.envs.tasks import iter_named_actors

    def is_actor(v):
        return type(v).__name__ == "Actor"

    t = env.task
    names = [n for n, v in vars(t).items()
             if not any(w in n.lower() for w in ("table", "wall", "ground", "robot"))
             and (is_actor(v) or (isinstance(v, (list, tuple)) and v and all(is_actor(x) for x in v)))]
    out = []
    try:
        for name, actor in iter_named_actors(t, names):
            o, R, hf = _obb_from_actor(actor)
            out.append((name, np.asarray(o, float), np.asarray(R, float), np.asarray(hf, float)))
    except Exception as exc:  # noqa: BLE001
        print(f"[robot_poses] object boxes unavailable: {exc!r}", flush=True)
    return out


def obj_gaps(C: np.ndarray, rad: np.ndarray, boxes: list) -> np.ndarray:
    g = np.full(C.shape[0], np.inf)
    for _, o, R, hf in boxes:
        loc = (C - o) @ R
        d = np.abs(loc) - hf
        sd = np.where((d <= 0).all(-1), d.max(-1), np.linalg.norm(np.maximum(d, 0), axis=-1))
        g = np.minimum(g, (sd - rad).min(1))
    return g


def project(K, E, P):
    Pc = P @ E[:3, :3].T + E[:3, 3]
    z = Pc[..., 2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return K[0, 0] * Pc[..., 0] / zs + K[0, 2], K[1, 1] * Pc[..., 1] / zs + K[1, 2], z


def edge_px(K, E, P, r, W, Hh) -> np.ndarray:
    """[N, S] distance in px from each sphere's disc (centre point if r = 0) to the nearest picture edge;
    negative = (partly) outside or behind the camera."""
    u, v, z = project(K, E, P)
    rp = K[0, 0] * r / np.maximum(z, 1e-6)
    e = np.minimum.reduce([u - rp, W - u - rp, v - rp, Hh - v - rp])
    return np.where(z > 0.05, e, -1e9)


# ------------------------------------------------------------------ search ----
def search_robot(env, emb: str, args, rng, cam):
    K, E, W, Hh = cam
    H = handles(env)
    boxes = object_boxes(env)
    print(f"[robot_poses] {emb}: object boxes " + ", ".join(f"{n} half {np.round(hf, 3).tolist()}"
                                                        for n, _, _, hf in boxes), flush=True)
    arms, info = {}, {"structural_pairs": {}, "counts": {}}
    for side in SIDES:
        h = H[side]
        lo = np.maximum(h["lim"][:, 0], -np.pi)
        hi = np.minimum(h["lim"][:, 1], np.pi)
        Q = rng.uniform(lo, hi, size=(args.n, len(lo)))
        P = link_points(h, Q)
        moving = P.std(0).max(1) > 1e-4
        outer = outer_links(h, args.view_from)

        def score(Q, P):
            s, ai = arm_score(P, h, side, Q, moving)
            s = s + 0.3 * (1 + ai["dz"])                 # gripper pointing down counts twice arm_score's weight
            e_link = edge_px(K, E, P[:, outer], np.zeros(int(outer.sum())), W, Hh).min(1)
            return np.where(e_link >= EDGE_PX, s, np.inf), ai

        s, ai = score(Q, P)
        cnt = {"sampled": args.n, "natural_and_wrist_in_view": int(np.isfinite(s).sum())}
        seeds = distinct_top(Q, s, 300, sep=0.15)        # refine: Gaussian steps around the passing poses
        if seeds:
            base = Q[seeds]
            Qr = np.clip(np.concatenate([base + rng.normal(0, sig, base.shape) for sig in (0.03, 0.08, 0.2)
                                         for _ in range(30)]), lo, hi)
            Q = np.concatenate([Q, Qr])
            P = np.concatenate([P, link_points(h, Qr)])
            s, ai = score(Q, P)
            cnt.update(refined=len(Qr), natural_and_wrist_in_view_after_refining=int(np.isfinite(s).sum()))
        order = [i for i in np.argsort(s)[: args.keep] if np.isfinite(s[i])]
        li, ctr, rad = sphere_model(h, emb)
        C_rand = centres(link_mats(h, Q[:3000]), li, ctr)
        mov_s = C_rand.std(0).max(1) > 1e-4              # spheres that move (a link origin on its own axis does not)
        out_s = outer[li]
        a, b, structural = self_check(h, li, rad, C_rand, len(h["lnames"]))
        info["structural_pairs"][side] = structural
        info["outer_links"] = info.get("outer_links", {})
        info["outer_links"][side] = [n for n, o in zip(h["lnames"], outer) if o]
        if not order:
            arms[side] = None
            info["counts"][side] = cnt
            print(f"[robot_poses] {emb} {side}: {json.dumps(cnt)}", flush=True)
            continue
        Qk = Q[order]
        C = centres(link_mats(h, Qk), li, ctr)
        g_self = pair_gaps(C, rad, a, b).min(1) if len(a) else np.full(len(Qk), np.inf)
        over = ((C[..., 0] > TABLE_X[0]) & (C[..., 0] < TABLE_X[1]) & (C[..., 1] > TABLE_Y[0])
                & (C[..., 1] < TABLE_Y[1]) & mov_s)
        g_table = np.where(over, C[..., 2] - rad - TABLE_Z, np.inf).min(1)
        g_obj = obj_gaps(C, rad, boxes)
        g_img = edge_px(K, E, C[:, out_s], rad[out_s], W, Hh).min(1)
        vis = (edge_px(K, E, C[:, mov_s], np.zeros(int(mov_s.sum())), W, Hh) >= 0).mean(1)   # share of the arm in view
        ok = (g_self >= 0) & (g_table >= TABLE_GAP) & (g_obj >= OBJ_GAP) & (g_img >= EDGE_PX)
        cnt.update(spheres_checked=len(Qk), no_self_overlap=int((g_self >= 0).sum()),
                   table_clear=int((g_table >= TABLE_GAP).sum()), objects_clear=int((g_obj >= OBJ_GAP).sum()),
                   wrist_spheres_in_view=int((g_img >= EDGE_PX).sum()), all_checks=int(ok.sum()))
        info["counts"][side] = cnt
        print(f"[robot_poses] {emb} {side}: {json.dumps(cnt)}; structural {structural}; "
              f"outer {info['outer_links'][side]}", flush=True)
        s2 = s[order] + 0.5 * (1 - vis)                  # more of the arm in the picture is better
        sel = [i for i in np.argsort(s2) if ok[i]][: args.pairs]
        idx = np.asarray(order)[sel]
        arms[side] = dict(Q=Q[idx], s=s2[sel], tip=ai["tip"][idx], C=C[sel], rad=rad, li=li, mov=mov_s,
                          g_self=g_self[sel], g_table=g_table[sel], g_obj=g_obj[sel], g_img=g_img[sel],
                          vis=vis[sel], tz=ai["tz"][idx], reach=ai["reach"][idx], dz=ai["dz"][idx], h=h)
    if arms["left"] is None or arms["right"] is None or not len(arms["left"]["Q"]) or not len(arms["right"]["Q"]):
        return H, [], info
    Lf, Rt = arms["left"], arms["right"]
    pairs = []
    for i in range(len(Lf["Q"])):
        D = (np.linalg.norm(Lf["C"][i][None, :, None, :] - Rt["C"][:, None, :, :], axis=-1)
             - Lf["rad"][None, :, None] - Rt["rad"][None, None, :])
        g_arm = D.reshape(len(Rt["Q"]), -1).min(1)
        tl, tr = Lf["tip"][i], Rt["tip"]
        sym = np.abs(tl[0] + tr[:, 0]) + np.abs(tl[1] - tr[:, 1]) + np.abs(tl[2] - tr[:, 2])
        cost = Lf["s"][i] + Rt["s"] + sym
        for j in np.flatnonzero(g_arm >= ARM_GAP):
            pairs.append((float(cost[j]), i, int(j), float(g_arm[j]), float(sym[j])))
    pairs.sort()
    chosen = []
    for cost, i, j, g_arm, sym in pairs:
        if all(max(np.abs(Lf["Q"][i] - Lf["Q"][ci]).max(), np.abs(Rt["Q"][j] - Rt["Q"][cj]).max()) > 0.5
               for _, ci, cj, _, _ in chosen):
            chosen.append((cost, i, j, g_arm, sym))
        if len(chosen) == args.top:
            break
    info["pairs_checked"] = len(Lf["Q"]) * len(Rt["Q"])
    info["pairs_apart"] = len(pairs)
    cands = []
    for cost, i, j, g_arm, sym in chosen:
        c = {"cost": round(cost, 4), "arms_gap_cm": round(100 * g_arm, 1), "gripper_asymmetry_cm": round(100 * sym, 1),
             "arms": {}}
        for side, A, k in (("left", Lf, i), ("right", Rt, j)):
            h = A["h"]
            c["arms"][side] = {
                "joints": h["names"], "q_rad": [round(float(v), 4) for v in A["Q"][k]],
                "q_deg": [round(float(np.degrees(v)), 1) for v in A["Q"][k]],
                "limits_rad": np.round(h["lim"], 3).tolist(),
                "gripper_xyz": np.round(A["tip"][k], 3).tolist(),
                "gripper_height_cm": round(100 * float(A["tz"][k]), 1),
                "reach_cm": round(100 * float(A["reach"][k]), 1),
                "gripper_down": round(float(-A["dz"][k]), 2),
                "self_gap_cm": round(100 * float(A["g_self"][k]), 2),
                "table_gap_cm": round(100 * float(A["g_table"][k]), 2),
                "object_gap_cm": round(100 * float(A["g_obj"][k]), 2),
                "wrist_picture_edge_px": round(float(A["g_img"][k]), 1),
                "arm_share_in_view": round(float(A["vis"][k]), 2),
                "score": round(float(A["s"][k]), 4)}
            c["arms"][side]["_q"] = A["Q"][k]
            c["arms"][side]["_C"] = A["C"][k]
            c["arms"][side]["_rad"] = A["rad"]
        cands.append(c)
    return H, cands, info


# ------------------------------------------------------------------ render ----
def set_arms(H: dict, qs: dict) -> None:
    for side in SIDES:
        h = H[side]
        qq = h["q0"].copy()
        qq[h["idx"]] = qs[side]
        h["ent"].set_qpos(qq.astype(np.float32))
        h["ent"].set_qvel(np.zeros_like(qq, dtype=np.float32))
        joints = h["ent"].get_active_joints()
        for i in h["idx"]:
            joints[i].set_drive_target(float(qq[i]))


def head_rgb(t) -> np.ndarray:
    t._update_render()
    return np.asarray(t.get_obs()["observation"]["head_camera"]["rgb"])[:, :, :3].astype(np.uint8)


def contacts_after_step(env, H) -> list:
    def key(e):
        return getattr(e, "per_scene_id", id(e))

    lab = {}
    for side in SIDES:
        for l in H[side]["links"]:
            lab[key(l.entity)] = f"{side}/{l.get_name()}"
    t = env.task
    t.scene.step()
    out = []
    for c in t.scene.get_contacts():
        e0, e1 = c.bodies[0].entity, c.bodies[1].entity
        n0, n1 = lab.get(key(e0)), lab.get(key(e1))
        if n0 is None and n1 is None:
            continue
        sep = min((float(p.separation) for p in c.points), default=1.0)
        if sep <= 0:
            out.append({"a": n0 or e0.get_name(), "b": n1 or e1.get_name(), "separation_mm": round(1000 * sep, 2)})
    return out


def camera(t, W, Hh):
    cfg = t.cameras.get_config()["head_camera"]
    K = np.asarray(cfg["intrinsic_cv"], float)
    E = np.asarray(cfg["extrinsic_cv"], float)
    return K, E, W, Hh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="place_container_plate")
    ap.add_argument("--seed", type=int, default=509000)
    ap.add_argument("--embodiments", nargs="+", default=["ur5-wsg", "piper"])
    ap.add_argument("--n", type=int, default=200000, help="random joint vectors per arm")
    ap.add_argument("--keep", type=int, default=6000, help="natural, in-view poses per arm given the sphere checks")
    ap.add_argument("--pairs", type=int, default=150, help="best checked poses per arm paired with the other arm's")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--view-from", type=int, default=3,
                    help="the links past this arm joint (0-based) must be fully in the picture (3: wrist + gripper)")
    ap.add_argument("--cam", default="Large_D435", choices=["D435", "Large_D435"])
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "debug").mkdir(parents=True, exist_ok=True)

    import main.lang  # noqa: F401  (main.item_envs path; RoboTwin on the path)
    from main.envs.record import observer_rgb
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    report = {"task": args.task, "seed": args.seed, "camera": args.cam, "robots": {}}
    heads, poses, rc = {}, None, 0
    for emb in args.embodiments:
        env = build_env(args.task, emb, args.seed, args.cam)
        try:
            if poses is None:
                poses = save_scene(env)
            else:
                print(f"[robot_poses] {emb}: objects moved by up to {100 * load_scene(env, poses):.2f} cm "
                      "to the first robot's poses", flush=True)
            t = env.task
            rgb0 = head_rgb(t)
            cam = camera(t, rgb0.shape[1], rgb0.shape[0])
            print(f"[robot_poses] {emb}: head camera {rgb0.shape[1]}x{rgb0.shape[0]}, world->camera "
                  f"{np.round(cam[1], 3).tolist()}", flush=True)
            H, cands, info = search_robot(env, emb, args, rng, cam)
            rep = {"checks": info, "candidates": []}
            if not cands:
                print(f"[robot_poses] {emb}: no pose passed every check", flush=True)
                rc = 1
            for k, c in enumerate(cands):
                set_arms(H, {side: c["arms"][side]["_q"] for side in SIDES})
                rgb = head_rgb(t)
                Image.fromarray(rgb).save(out / f"cand{k}_{emb}_head.png")
                Image.fromarray(observer_rgb(t)).save(out / f"cand{k}_{emb}_observer.png")
                heads.setdefault(k, []).append(rgb)
                dbg = Image.fromarray(rgb.copy())
                dr = ImageDraw.Draw(dbg)
                K, E = cam[0], cam[1]
                for side, col in (("left", (255, 60, 60)), ("right", (60, 160, 255))):
                    Cs, rs = c["arms"][side]["_C"], c["arms"][side]["_rad"]
                    u, v, z = project(K, E, Cs)
                    for uu, vv, zz, r in zip(u, v, z, rs):
                        if zz > 0.05:
                            rp = K[0, 0] * r / zz
                            dr.ellipse([uu - rp, vv - rp, uu + rp, vv + rp], outline=col)
                dbg.save(out / "debug" / f"cand{k}_{emb}_head_spheres.png")
                c["contacts_after_one_step"] = contacts_after_step(env, H)
                load_scene(env, poses)
                for side in SIDES:
                    for key_ in ("_q", "_C", "_rad"):
                        c["arms"][side].pop(key_)
                rep["candidates"].append(c)
                print(f"[robot_poses] {emb} cand{k}: {json.dumps(c)}", flush=True)
            report["robots"][emb] = rep
        finally:
            env.close()
    for k, row in heads.items():
        if len(row) > 1:
            gap = np.full((row[0].shape[0], 8, 3), 255, np.uint8)
            parts = [p for img in row for p in (img, gap)][:-1]
            Image.fromarray(np.concatenate(parts, axis=1)).save(out / f"cand{k}_head_robots.png")
    (out / "poses.json").write_text(json.dumps(report, indent=1))
    print(f"[robot_poses] wrote {out}/poses.json", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
