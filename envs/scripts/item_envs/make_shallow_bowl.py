#!/usr/bin/env python
"""Generate the 002_bowl_shallow asset variant (user review 5: a container
with a smaller height spills at small tilts and needs velocity control).

Writes robotwin_assets/002_bowl_shallow/model_data<id>.json — a copy of
RoboTwin's 002_bowl json with the UP axis (file y) of ``scale`` multiplied by
k_id = TARGET_DEPTH / (rim_y - floor_y) of the original bowl — and symlinks
the original collision / visual meshes (RoboTwin applies ``scale``
element-wise to meshes, contact points and OBBs). No RoboTwin file is
modified; scripts/item_envs/install_robotwin_tasks.sh links the folder into
RoboTwin/assets/objects/.

    python scripts/item_envs/make_shallow_bowl.py   # from envs/
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN = (ROOT / ".." / ".." / "RoboTwin").resolve()
SRC = ROBOTWIN / "assets" / "objects" / "002_bowl"
DST = ROOT / "robotwin_assets" / "002_bowl_shallow"
IDS = (1, 2, 3, 5, 6, 7)
TARGET_DEPTH = 0.025  # m, rim plane above the interior floor


def main() -> int:
    spec = importlib.util.spec_from_file_location("iu", ROOT / "robotwin_tasks" / "_item_utils.py")
    iu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iu)
    for sub in ("collision", "visual"):
        (DST / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(ROBOTWIN)  # bowl_geometry resolves assets/objects relative to the RoboTwin root
    for i in IDS:
        g = iu.bowl_geometry("002_bowl", i)
        k = TARGET_DEPTH / (float(g["rim_y"]) - float(g["floor_y"]))
        data = json.loads((SRC / f"model_data{i}.json").read_text())
        s = data["scale"]
        data["scale"] = [float(s[0]), float(s[1]) * k, float(s[2])]
        data["shallow_k"] = k
        (DST / f"model_data{i}.json").write_text(json.dumps(data, indent=1))
        for sub in ("collision", "visual"):
            link = DST / sub / f"base{i}.glb"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(SRC / sub / f"base{i}.glb", link.parent))  # relative: survives a moved checkout with the same sibling layout
        print(f"id {i}: depth {100 * (g['rim_y'] - g['floor_y']):.1f} cm -> k {k:.3f}")
    pts = DST / "points_info.json"
    if pts.is_symlink() or pts.exists():
        pts.unlink()
    pts.symlink_to(os.path.relpath(SRC / "points_info.json", DST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
