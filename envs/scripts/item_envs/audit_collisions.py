"""v3 Phase-0: collision audit for the T1/T2 asset pools.

For every asset below, per model id:
  - enumerate available ids (union of model_data*.json, collision/, visual/,
    top-level base*.glb / textured*.obj ids),
  - resolve the collision + visual files EXACTLY as RoboTwin's
    `envs/utils/create_actor.py:create_actor` does (collision/base{id}.glb ->
    collision/textured{id}.obj -> top-level),
  - count convex collision pieces two ways: trimesh scene-geometry nodes of
    the resolved collision file, and the number of PhysX convex shapes sapien
    actually builds (`add_multiple_convex_collisions_from_file` on a headless
    PhysxCpuSystem scene — no Vulkan needed),
  - volumes via trimesh: V_shell = sum of per-piece convex-hull volumes
    (what PhysX collides with), V_union = convex hull of all collision
    vertices, fill = V_shell / V_union (<< 1 for a hollow container shell,
    ~1 for a solid), r_hull_mesh = V_shell / V_visual (voxelized visual
    volume, container assets only — the visual meshes are open, so signed
    volumes are unreliable),
  - hollowness: occupancy of a grid in the central 70% of the collision bbox
    and of the three bbox-center axis lines, inside the union of piece hulls,
  - model_data json: contact_points_pose count (rigid) / contact_points
    (articulation), scale, extents, stable flag,
  - world height/footprint when spawned upright with RoboTwin's convention
    quat [0.5, 0.5, 0.5, 0.5] (file-y -> world-z).

015_laptop is an articulation asset (numeric subdirs with mobility.urdf);
it is audited from the URDF XML instead (collision meshes per link, joint
limits, contact_points count).

Usage:
    python audit_collisions.py                 # markdown table to stdout
    python audit_collisions.py --csv out.csv
    python audit_collisions.py --no-runtime    # skip the sapien shape builds
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

OBJ = Path("/work/nvme/bifq/project/RoboTwin/assets/objects")

# (asset, role, ids-of-interest note). role drives the verdict rule.
RIGID = [
    ("002_bowl", "container", "T2 bowl; host uses ids [1,2,3,5]"),
    ("003_plate", "support", "T2 plate (static place target)"),
    ("062_plasticbox", "container", "T2 target-container candidate"),
    ("068_boxdrink", "candidate", "T1 pool (CrossSafe obstacle)"),
    ("086_woodenblock", "candidate", "T1 pool"),
    ("001_bottle", "candidate", "T1 pool / T3 bystander"),
    ("114_bottle", "candidate", "T1 pool (no id 0)"),
    ("021_cup", "candidate", "T1 pool / T3 bystander / T2 alt container"),
    ("039_mug", "candidate", "T1 pool"),
    ("071_can", "candidate", "T1 pool"),
    ("080_pillbottle", "candidate", "T1 pool (no id 0)"),
    ("105_sauce-can", "candidate", "T1 pool"),
    ("112_tea-box", "candidate", "T1 pool / T3 push object"),
    ("113_coffee-box", "candidate", "T1 pool / T3 push object"),
]
ARTICULATED = [("015_laptop", "zone-prop", "T2 static forbidden-region prop")]

# RoboTwin spawn convention: qpos=[0.5,0.5,0.5,0.5] rotates file-x->world-y,
# file-y->world-z, file-z->world-x (verified: rotation matrix of that quat is
# [[0,0,1],[1,0,0],[0,1,0]]; bowl AABB is y-vertical in file coords).
SPAWN_PERM = (2, 0, 1)  # world (x,y,z) sizes = file sizes permuted by this


def get_glb_or_obj_file(modeldir: Path, model_id):
    """Exact copy of create_actor.py:get_glb_or_obj_file."""
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        file = modeldir / ("textured.obj" if model_id is None else f"textured{model_id}.obj")
    return file


def resolve_files(modeldir: Path, model_id: int):
    """Mirror create_actor.py:518-532. Returns (collision, visual, loadable)."""
    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)
    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)
    ok = collision_file.exists() and visual_file.exists()
    return (collision_file if collision_file != "" else None,
            visual_file if visual_file != "" else None, ok)


def enumerate_ids(modeldir: Path) -> list[int]:
    ids = set()
    for pat in ("model_data*.json", "collision/base*.glb", "visual/base*.glb",
                "base*.glb", "collision/textured*.obj", "visual/textured*.obj",
                "textured*.obj"):
        for p in modeldir.glob(pat):
            m = re.search(r"(\d+)", p.name)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def load_pieces(path: Path):
    """Collision sub-meshes == hulls sapien builds (one per scene node)."""
    loaded = trimesh.load(str(path), force="scene", process=False)
    geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    return [g for g in geoms if getattr(g, "vertices", None) is not None and len(g.vertices) >= 4]


def geometry_stats(pieces):
    """V_shell, V_union, fill, grid occupancy, per-axis line occupancy."""
    hulls = []
    for p in pieces:
        try:
            h = p.convex_hull
            if h.volume > 1e-12:
                hulls.append(h)
        except Exception:
            pass
    if not hulls:
        return None
    v_shell = float(sum(abs(h.volume) for h in hulls))
    all_v = np.concatenate([h.vertices for h in hulls], axis=0)
    v_union = float(abs(trimesh.points.PointCloud(all_v).convex_hull.volume))
    lo, hi = all_v.min(0), all_v.max(0)

    def inside_any(pts):
        occ = np.zeros(len(pts), dtype=bool)
        for h in hulls:
            occ |= h.contains(pts)
        return occ

    # grid over central 70% of the bbox
    ax = [np.linspace(lo[i] + 0.15 * (hi[i] - lo[i]), hi[i] - 0.15 * (hi[i] - lo[i]), 8)
          for i in range(3)]
    grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_occ = float(inside_any(grid).mean())
    # center lines along each axis, [5%, 95%] of the range
    ctr = (lo + hi) / 2
    line_occ = []
    for i in range(3):
        t = np.linspace(lo[i] + 0.05 * (hi[i] - lo[i]), hi[i] - 0.05 * (hi[i] - lo[i]), 24)
        pts = np.tile(ctr, (len(t), 1))
        pts[:, i] = t
        line_occ.append(float(inside_any(pts).mean()))
    return dict(v_shell=v_shell, v_union=v_union,
                fill=v_shell / v_union if v_union > 0 else float("nan"),
                grid_occ=grid_occ, line_occ=line_occ,
                bbox_size=hi - lo)


# ids for which the (expensive) voxelized visual volume is computed.
# "all" = every id; otherwise a list of ids. Voxel volume at pitch
# extents/160 costs ~2-5 s per file and is ~10-15% accurate on thin shells
# (coarser pitches inflate badly — measured: plate 7.78 -> 11.7 going
# div=220 -> 80). Solid candidates (fill ~ 1) don't need it.
VOXEL_IDS = {
    "002_bowl": "all",
    "003_plate": "all",
    "062_plasticbox": "all",
    "021_cup": [0],
    "039_mug": [0],
}


def visual_volume_voxel(path: Path, div: int = 160):
    """Approximate visual-mesh volume via voxelization (handles open meshes).

    These visual meshes are winding-consistent but NOT watertight (verified
    on 002_bowl/003_plate/068_boxdrink id0 visuals), so trimesh's signed
    `volume` is unreliable; voxelize+fill instead. Returns (volume, None)
    or (None, reason).
    """
    try:
        loaded = trimesh.load(str(path), force="scene", process=False)
        geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
        vol = 0.0
        for g in geoms:
            pitch = float(g.extents.max()) / div
            if pitch <= 0:
                return None, "degenerate"
            vol += g.voxelized(pitch=pitch).fill().volume
        return float(vol), None
    except Exception as e:
        return None, type(e).__name__


def read_json(modeldir: Path, model_id):
    jp = modeldir / ("model_data.json" if model_id is None else f"model_data{model_id}.json")
    if not jp.exists():
        return None
    try:
        return json.loads(jp.read_text())
    except Exception:
        return None


def runtime_shape_count(scene, path: Path, scale):
    """Convex shapes sapien actually cooks from this file (headless CPU)."""
    b = scene.create_actor_builder()
    b.add_multiple_convex_collisions_from_file(filename=str(path), scale=tuple(scale))
    import sapien.physx as physx
    a = b.build(name=f"audit_{path.stem}")
    comp = a.find_component_by_type(physx.PhysxRigidDynamicComponent)
    return len(comp.get_collision_shapes())


def verdict(role, loadable, stats, ncp, height_m, pieces):
    if not loadable:
        return "drop (no model file)"
    if role == "container":
        hollow = stats is not None and stats["fill"] < 0.65 and stats["grid_occ"] < 0.5
        if hollow and pieces >= 4 and ncp >= 1:
            return "ok as container"
        if not hollow:
            return "needs CoACD overlay (filled)"
        return "needs CoACD overlay" if pieces < 4 else "ok as container (no grasp pts)"
    if role == "support":
        return "ok as support" if pieces >= 1 else "needs CoACD overlay"
    if role == "candidate":
        if height_m is None:
            return "ok as candidate (no json)"
        if not (0.03 <= height_m <= 0.25):
            return f"drop (height {height_m * 100:.1f} cm)"
        note = ""
        if height_m > 0.20:
            note = f" (tall: {height_m * 100:.1f} cm > 20 cm plan limit)"
        return "ok as candidate" + note
    return "?"


def audit_rigid(name, role, note, obj_root, runtime=True):
    d = obj_root / name
    rows = []
    if not d.is_dir():
        return [dict(asset=name, id="-", role=role, verdict="drop (asset MISSING)", note=note)]
    ids = enumerate_ids(d)
    scene = None
    if runtime:
        import sapien
        import sapien.physx as physx
        scene = sapien.Scene([physx.PhysxCpuSystem()])
    for mid in ids:
        coll, vis, loadable = resolve_files(d, mid)
        jd = read_json(d, mid)
        ncp = len(jd["contact_points_pose"]) if jd and "contact_points_pose" in jd else (
            -1 if jd is None else 0)
        scale = jd.get("scale", [1, 1, 1]) if jd else [1, 1, 1]
        extents = jd.get("extents") if jd else None
        stable = jd.get("stable") if jd else None
        world = [extents[SPAWN_PERM[i]] * scale[SPAWN_PERM[i]] for i in range(3)] if extents else None
        height_m = world[2] if world else None
        foot = f"{world[0] * 100:.1f}x{world[1] * 100:.1f}" if world else "-"
        pieces, stats, n_rt, r_hull = -1, None, -1, None
        coll_rel = "-"
        if loadable:
            coll_rel = str(coll.relative_to(d))
            try:
                pieces_list = load_pieces(coll)
                pieces = len(pieces_list)
                stats = geometry_stats(pieces_list)
            except Exception as e:
                stats = None
                coll_rel += f" (trimesh: {type(e).__name__})"
            if runtime:
                try:
                    n_rt = runtime_shape_count(scene, coll, scale)
                except Exception as e:
                    n_rt = f"ERR:{type(e).__name__}"
            vox = VOXEL_IDS.get(name)
            want_vox = vox == "all" or (isinstance(vox, list) and mid in vox)
            if want_vox:
                vv, err = visual_volume_voxel(vis)
                if vv and stats and vv > 0:
                    r_hull = stats["v_shell"] / vv
                elif err:
                    r_hull = None
        v = verdict(role, loadable, stats, ncp, height_m, max(pieces, 0))
        rows.append(dict(
            asset=name, id=mid, role=role, loadable=loadable, coll=coll_rel,
            pieces=pieces, sapien=n_rt,
            fill=f"{stats['fill']:.2f}" if stats else "-",
            grid=f"{stats['grid_occ']:.2f}" if stats else "-",
            lines="/".join(f"{x:.2f}" for x in stats["line_occ"]) if stats else "-",
            r_hull_mesh=f"{r_hull:.2f}" if r_hull else "-",
            h_cm=f"{height_m * 100:.1f}" if height_m else "-", foot_cm=foot,
            ncp=ncp if ncp >= 0 else "nojson", stable=stable, verdict=v, note=note,
        ))
    return rows


def audit_laptop(name, role, note, obj_root):
    d = obj_root / name
    rows = []
    subdirs = sorted([p for p in d.iterdir()
                      if p.is_dir() and p.name not in ("visual", "collision")],
                     key=lambda p: int(re.search(r"\d+", p.name).group()))
    for sd in subdirs:
        urdf = sd / "mobility.urdf"
        jd = read_json(sd, None)  # model_data.json (no id suffix)
        ncp = len(jd.get("contact_points", [])) if jd else -1
        scale = jd.get("scale") if jd else None
        links, joints = {}, []
        if urdf.exists():
            root = ET.parse(urdf).getroot()
            for link in root.iter("link"):
                links[link.get("name")] = len(list(link.iter("collision")))
            for j in root.iter("joint"):
                lim = j.find("limit")
                joints.append((j.get("name"), j.get("type"),
                               float(lim.get("lower", "nan")) if lim is not None else None,
                               float(lim.get("upper", "nan")) if lim is not None else None))
        n_coll = sum(links.values())
        loadable = urdf.exists() and jd is not None
        rows.append(dict(
            asset=f"{name}/{sd.name}", id=sd.name, role=role, loadable=loadable,
            coll=f"mobility.urdf ({n_coll} coll meshes: "
                 + ",".join(f"{k}={v}" for k, v in links.items()) + ")",
            pieces=n_coll, sapien="-",
            fill="-", grid="-", lines="-", r_hull_mesh="-",
            h_cm="-", foot_cm="-", ncp=ncp if ncp >= 0 else "nojson",
            stable=f"scale={scale}",
            verdict="ok as zone prop (fix_root_link)" if loadable else "drop",
            note=(note + "; joints " + "; ".join(
                f"{n}:{t}[{lo if lo is not None else '-'},{hi if hi is not None else '-'}]"
                for n, t, lo, hi in joints)) if joints else note,
        ))
    if not subdirs:
        rows.append(dict(asset=name, id="-", role=role, verdict="drop (no subdirs)", note=note))
    return rows


MD_COLS = ["asset", "id", "loadable", "pieces", "sapien", "fill", "grid", "lines",
           "r_hull_mesh", "h_cm", "foot_cm", "ncp", "stable", "verdict"]


def to_md(rows):
    out = ["| " + " | ".join(MD_COLS) + " |", "|" + "---|" * len(MD_COLS)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in MD_COLS) + " |")
    return "\n".join(out)


def to_csv(rows):
    import io
    buf = io.StringIO()
    buf.write(",".join(MD_COLS + ["coll", "note"]) + "\n")
    for r in rows:
        buf.write(",".join(f"\"{r.get(c, '')}\"" for c in MD_COLS + ["coll", "note"]) + "\n")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--objects-dir", type=Path, default=OBJ)
    ap.add_argument("--no-runtime", action="store_true")
    args = ap.parse_args()
    obj_root = args.objects_dir

    rows = []
    for name, role, note in RIGID:
        rows.extend(audit_rigid(name, role, note, obj_root, runtime=not args.no_runtime))
        print(f"[audit] {name} done", file=sys.stderr)
    for name, role, note in ARTICULATED:
        rows.extend(audit_laptop(name, role, note, obj_root))

    md = to_md(rows)
    print(md)
    if args.csv:
        args.csv.write_text(to_csv(rows))
        print(f"[audit] csv -> {args.csv}", file=sys.stderr)
    if args.md:
        args.md.write_text(md + "\n")
        print(f"[audit] md -> {args.md}", file=sys.stderr)


if __name__ == "__main__":
    main()
