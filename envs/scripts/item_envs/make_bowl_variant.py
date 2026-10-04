#!/usr/bin/env python
"""Generate a rescaled 002_bowl asset variant under robotwin_assets/<name>/.

    python scripts/item_envs/make_bowl_variant.py --name 002_bowl_deep --depth 0.09 --radial 1.2

Each model_data<id>.json is RoboTwin's 002_bowl json with ``scale`` multiplied
per axis: file y (up) by k_id = depth / (rim_y - floor_y) of the original bowl,
file x / z by ``--radial``; the collision / visual meshes are RELATIVE symlinks
to the originals (RoboTwin applies ``scale`` element-wise to meshes, contact
points and OBBs). No RoboTwin file is modified;
scripts/item_envs/install_robotwin_tasks.sh links the folder into
RoboTwin/assets/objects/. Used for the T2 pour TARGET (user review 6: a
deeper, wider target so poured beads do not bounce out).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN = (ROOT / ".." / ".." / "RoboTwin").resolve()
SRC = ROBOTWIN / "assets" / "objects" / "002_bowl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--depth", type=float, required=True, help="interior depth (m), rim plane above the floor")
    ap.add_argument("--radial", type=float, default=1.0, help="scale of the horizontal axes")
    ap.add_argument("--ids", default="1,2,3,5")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("iu", ROOT / "robotwin_tasks" / "_item_utils.py")
    iu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iu)
    dst = ROOT / "robotwin_assets" / args.name
    for sub in ("collision", "visual"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(ROBOTWIN)  # bowl_geometry resolves assets/objects from the RoboTwin root
    for i in (int(v) for v in args.ids.split(",")):
        g = iu.bowl_geometry("002_bowl", i)
        k = args.depth / (float(g["rim_y"]) - float(g["floor_y"]))
        data = json.loads((SRC / f"model_data{i}.json").read_text())
        s = data["scale"]
        data["scale"] = [float(s[0]) * args.radial, float(s[1]) * k, float(s[2]) * args.radial]
        data["variant"] = {"of": "002_bowl", "depth": args.depth, "radial": args.radial, "k_up": k}
        (dst / f"model_data{i}.json").write_text(json.dumps(data, indent=1))
        for sub in ("collision", "visual"):
            link = dst / sub / f"base{i}.glb"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(SRC / sub / f"base{i}.glb", link.parent))
        print(f"{args.name} id {i}: depth {100 * (g['rim_y'] - g['floor_y']):.1f} -> {100 * args.depth:.1f} cm (k_up {k:.3f}), radial x{args.radial}")
    pts = dst / "points_info.json"
    if pts.is_symlink() or pts.exists():
        pts.unlink()
    pts.symlink_to(os.path.relpath(SRC / "points_info.json", dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
