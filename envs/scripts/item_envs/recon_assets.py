"""Phase-0 recon Q12: audit the Section-7 candidate asset pools.

For each asset under RoboTwin/assets/objects/<asset>:
  - exists on disk
  - model_data*.json: count, and which have "contact_points_pose"
  - collision file(s): resolution per create_actor.py (collision/base{id}.glb ->
    textured{id}.obj -> top-level), number of convex pieces in the model_id=0
    collision file (trimesh scene nodes / connected components)
"""
import json
import re
from pathlib import Path

import trimesh

OBJ = Path("/work/nvme/bifq/project/RoboTwin/assets/objects")

ASSETS = [
    "001_bottle", "002_bowl", "008_tray", "011_dustbin", "015_laptop",
    "019_coaster", "021_cup", "036_cabinet", "039_mug", "063_tabletrashbin",
    "071_can", "075_bread", "080_pillbottle", "086_woodenblock", "102_roller",
    "105_sauce-can", "110_basket", "114_bottle", "048_stapler", "113_coffee-box",
]


def get_glb_or_obj_file(modeldir: Path, model_id):
    """Exact copy of create_actor.py's resolution."""
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        file = modeldir / ("textured.obj" if model_id is None else f"textured{model_id}.obj")
    return file


def collision_file_for(modeldir: Path, model_id: int):
    """Mirror create_actor.py's collision_file resolution."""
    collision_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)
    return collision_file if collision_file.exists() else None


def count_pieces(path: Path) -> tuple[int, str]:
    """Number of convex pieces sapien would build from this file.

    sapien's add_multiple_convex_collisions_from_file builds one hull per
    sub-mesh. For a GLB trimesh exposes scene.geometry entries; for a single
    mesh we count connected components.
    """
    try:
        loaded = trimesh.load(str(path), force="scene", process=False)
        geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
        n = len(geoms)
        detail = f"scene nodes={n}"
        return n, detail
    except Exception as e:
        return -1, f"load failed: {type(e).__name__}: {e}"


print(f"{'asset':<18} {'exists':<7} {'#json':<6} {'cpp':<6} {'coll_file(mid0)':<34} {'pieces':<7} notes")
rows = []
for name in ASSETS:
    d = OBJ / name
    if not d.is_dir():
        print(f"{name:<18} MISSING")
        rows.append((name, False, 0, "", "", -1, ""))
        continue
    jsons = sorted(d.glob("model_data*.json"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    n_json = len(jsons)
    cpp = []  # model_ids with contact_points_pose
    scales = set()
    for jp in jsons:
        try:
            data = json.loads(jp.read_text())
        except Exception as e:
            cpp.append(f"{jp.name}:BADJSON")
            continue
        mid = re.search(r"(\d+)", jp.name).group(1)
        if isinstance(data, dict) and "contact_points_pose" in data:
            cpp.append(mid)
        if isinstance(data, dict) and "scale" in data:
            scales.add(str(data["scale"]))
    mid0 = mids[0] if (mids := [int(re.search(r"(\d+)", p.name).group(1)) for p in jsons]) else 0
    cf = collision_file_for(d, mid0)
    used_mid = mid0
    if cf is None:
        cf_rel, pieces, note = "-", -1, "no collision file"
    else:
        cf_rel = str(cf.relative_to(d))
        pieces, note = count_pieces(cf)
    extra = f" first_id={used_mid}(id0 {'exists' if 0 in mids or not mids else 'MISSING'})"
    if (d / "mobility.urdf").exists():
        extra += " articulation(mobility.urdf)"
    subdirs = [p.name for p in d.iterdir() if p.is_dir() and p.name not in ("visual", "collision")]
    if subdirs:
        extra += f" urdf_model_ids={sorted(subdirs)}"
    n_coll = len(list((d / "collision").glob("*"))) if (d / "collision").exists() else 0
    print(f"{name:<18} {'yes':<7} {n_json:<6} {len(cpp)}/{n_json:<4} {cf_rel:<34} {pieces:<7} "
          f"coll_dir_files={n_coll}{extra} scales={sorted(scales)[:3]}")
    rows.append((name, True, n_json, f"{len(cpp)}/{n_json}", cf_rel, pieces, note, extra))

# also count pieces for EVERY model_id of container-ish assets (cheap sanity)
for name in ["036_cabinet", "063_tabletrashbin", "002_bowl", "008_tray", "011_dustbin"]:
    d = OBJ / name
    if not d.is_dir():
        continue
    jsons = sorted(d.glob("model_data*.json"),
                   key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    mids = [int(re.search(r"(\d+)", p.name).group(1)) for p in jsons]
    per = {}
    for mid in mids[:14]:
        cf = collision_file_for(d, mid)
        if cf is not None:
            n, _ = count_pieces(cf)
            per[mid] = (cf.name, n)
    print(f"\n{name} per-model pieces:", per)
