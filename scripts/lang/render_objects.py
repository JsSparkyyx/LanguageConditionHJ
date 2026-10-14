"""Contact sheet of RoboTwin object models (visual/base<id>.glb), rendered on the CPU with matplotlib, so object
names can be checked against what the models look like (names check of 2026-10-12, unseen-object check 2026-10-13).

    python scripts/lang/render_objects.py <asset> <id,id,...> "<title>" <out.png>

Each panel: the textured mesh shaded by one light, standing up (glTF is y-up), titled with its model id and its real
size (model_data<id>.json extents x scale). CPU only, no simulator.
"""

from __future__ import annotations

import json
import sys

import numpy as np

ROOT = "/work/nvme/bifq/project/RoboTwin/assets/objects"


def load(asset: str, mid: int, max_faces: int = 40000):
    import trimesh

    sc = trimesh.load(f"{ROOT}/{asset}/visual/base{mid}.glb", force="scene")
    tris, cols = [], []
    for name, geom in sc.geometry.items():
        g = geom.copy()
        try:
            g.apply_transform(sc.graph[name][0])
        except Exception:
            pass
        try:
            vc = g.visual.to_color().vertex_colors[:, :3] / 255.0
        except Exception:
            vc = np.full((len(g.vertices), 3), 0.7)
        tris.append(g.vertices[g.faces])
        cols.append(vc[g.faces].mean(1))
    tris, cols = np.concatenate(tris), np.concatenate(cols)
    if len(tris) > max_faces:
        idx = np.random.default_rng(0).choice(len(tris), max_faces, replace=False)
        tris, cols = tris[idx], cols[idx]
    return tris, cols


def draw(ax, tris, cols, elev=20, azim=-60):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
    shade = 0.45 + 0.55 * np.abs(n @ light)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=np.clip(cols * shade[:, None], 0, 1), edgecolors="none"))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def sheet(asset: str, ids: list[int], title: str, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ids)
    cols_n = min(n, 5)
    rows = (n + cols_n - 1) // cols_n
    fig = plt.figure(figsize=(3.2 * cols_n, 3.4 * rows))
    fig.suptitle(title, fontsize=13)
    for k, mid in enumerate(ids):
        ax = fig.add_subplot(rows, cols_n, k + 1, projection="3d")
        try:
            tris, cols = load(asset, mid)
            draw(ax, tris[..., [0, 2, 1]] * np.array([1, -1, 1]), cols)   # y-up -> z-up
            md = json.load(open(f"{ROOT}/{asset}/model_data{mid}.json"))
            e = np.asarray(md["extents"]) * np.asarray(md.get("scale", [1, 1, 1])) * 100   # [x, height, z] cm
            ax.set_title(f"id {mid}: {e[0]:.1f} x {e[2]:.1f} cm, {e[1]:.1f} cm tall", fontsize=9)
        except Exception as exc:
            ax.set_title(f"id {mid}: {type(exc).__name__}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    sheet(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3], sys.argv[4])
