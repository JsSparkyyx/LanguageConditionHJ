"""P5 token alignment (PLAN v3 §6 Phase 3, gpu).

One item at a known pose (``smoke_items`` box_0, registered via
``task.item_meta``) inside a CEHJ ``Env``. For one camera:

(a) the item's token-grid mask from ``masks.item_masks`` (400 cells: stride
    16 -> 320 cells finest-first, then stride 32 -> 80, row-major — recon Q15);
(b) the encoder's token world positions: ``Env.get_encoder_obs`` ->
    ``HoloBrainEncoder.encode_visual_tokens`` -> ``pos_mean`` [1, 3, 400, 3],
    depth-weighted back-projected anchors in world frame via T_ego2world
    (``main/network/encoder.py:467-543``).

Gate (PLAN §6 P5): the token-mask cells with the highest coverage CONTAIN the
encoder token positions closest to the item centroid. Implementation: k =
number of cells with coverage >= 50% (uint8 >= 128); take the k ``pos_mean``
tokens nearest the item's OBB centroid; overlap = |nearest ∩ top-coverage| /
k. Reported alongside: the mean centroid distance of the top-coverage tokens
vs all tokens, and the same overlap against the relaxed >= 25% coverage set
(diagnostic for edge cells whose pos_mean is pulled toward the background).

Rows -> ``<out_dir>/p5_rows.jsonl``; one keyframe (mask grid vs projected
pos_mean) under ``<out_dir>/keyframes/``. GPU-only (HoloBrain checkpoint +
RT rendering). The pytest ``@pytest.mark.gpu`` test calls ``check_once``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .p2 import append_row, done_cells, load_rows

ROWS_FILE = "p5_rows.jsonl"
COVERAGE_MIN = 128  # uint8, ~50% of a token cell


def _item_centroid(spec) -> np.ndarray:
    """World-frame OBB centroid of the item."""
    from ..items import entity_of

    ent = entity_of(spec.entity)
    pose = ent.get_pose()
    T = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
    return T[:3, :3] @ np.asarray(spec.obb_center_local, dtype=np.float64) + T[:3, 3]


def alignment_metrics(mask_row: np.ndarray, pos_world: np.ndarray, centroid: np.ndarray,
                      coverage_min: int = COVERAGE_MIN) -> dict:
    """The P5 overlap numbers (pure numpy — unit-testable).

    mask_row: [n_tok] uint8 coverage per token cell of ONE camera.
    pos_world: [n_tok, 3] token world positions of the same camera.
    centroid: [3] item OBB centroid (world).
    """
    mask_row = np.asarray(mask_row)
    pos = np.asarray(pos_world, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(centroid, dtype=np.float64).reshape(3)
    if mask_row.shape[0] != pos.shape[0]:
        raise ValueError(f"mask {mask_row.shape} vs pos {pos.shape} token count mismatch")
    top = np.flatnonzero(mask_row >= coverage_min)
    k = int(len(top))
    if k == 0:
        return {"k": 0, "overlap": None, "top_cells": [], "nearest": []}
    dist = np.linalg.norm(pos - c[None, :], axis=1)
    nearest = np.argsort(dist)[:k]
    overlap = float(len(set(int(i) for i in nearest) & set(int(i) for i in top)) / k)
    relaxed = np.flatnonzero(mask_row >= coverage_min // 2)
    overlap_relaxed = (
        float(len(set(int(i) for i in nearest) & set(int(i) for i in relaxed)) / k) if k else None
    )
    return {
        "k": k,
        "overlap": overlap,
        "overlap_relaxed_25pct": overlap_relaxed,
        "top_cells": [int(i) for i in top],
        "nearest": [int(i) for i in nearest],
        "dist_top_mean": float(dist[top].mean()),
        "dist_all_mean": float(dist.mean()),
        "dist_nearest_mean": float(dist[nearest].mean()),
        "max_coverage": int(mask_row.max()),
    }


def check_once(out_dir, seed: int = 0, embodiment: str = "ur5-wsg", camera: str = "head",
               keyframe: bool = True) -> dict:
    """Build the scene once, compute the P5 row, append it, return it."""
    import torch

    from .._bootstrap import ensure_main_path

    ensure_main_path()
    from main.envs.env import Env  # noqa: E402
    from main.network.encoder import HoloBrainEncoder  # noqa: E402
    from main.train.collect import ARM_DISTANCE, CKPT_DIR  # noqa: E402

    from .. import masks
    from ..items import ItemSpec, entity_of

    out_dir = Path(out_dir).resolve()  # Env creation chdirs to the RoboTwin root
    env = Env("smoke_items", embodiment, int(seed),
              arm_distance=ARM_DISTANCE.get(embodiment, 0.6), control_freq=25.0)
    try:
        task = env.task
        meta = {m["item_id"]: m for m in getattr(task, "item_meta", [])}["box_0"]
        spec = ItemSpec(
            item_id="box_0", entity=task.box_0, category="box", color="red",
            half_extents=meta["half_extents"],
            obb_center_local=meta.get("obb_center_local") or [0.0, 0.0, 0.0],
        )
        centroid = _item_centroid(spec)

        cam_idx = list(masks.CAMERA_NAMES).index(camera)
        _, token_masks = masks.item_masks(task, [spec], [], None)
        mask_row = token_masks[0, cam_idx]

        obs = env.get_encoder_obs()  # kin=None: T_base2ego unused for visual tokens
        encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
        dev = torch.device("cuda")
        _, pos_mean, pos_max = encoder.encode_visual_tokens(
            obs["imgs"].to(dev),
            obs["depths"].to(dev),
            obs["image_wh"].to(dev),
            obs["projection_mat"].to(dev),
            out_extrinsic=obs["T_ego2world"],
        )
        pos = pos_mean[0, cam_idx].double().cpu().numpy()  # [400, 3] world
        pos_mx = pos_max[0, cam_idx].double().cpu().numpy()

        row = {
            "cell": ["p5", camera, int(seed)],
            "seed": int(seed),
            "embodiment": embodiment,
            "camera": camera,
            "centroid": [float(v) for v in centroid],
            "box_pose_p": [float(v) for v in entity_of(spec.entity).get_pose().p],
            **alignment_metrics(mask_row, pos, centroid),
        }
        # pos_mean vs pos_max spread over the top cells (depth-anchor sanity)
        if row["k"]:
            spread = np.linalg.norm(pos[row["top_cells"]] - pos_mx[row["top_cells"]], axis=1)
            row["pos_mean_max_spread_m"] = float(spread.mean())
        append_row(out_dir / ROWS_FILE, row)
        if keyframe:
            row["keyframe"] = _keyframe(env, obs, mask_row, pos, centroid, row, out_dir)
        print(f"[p5] k={row['k']} overlap={row['overlap']} "
              f"dist_top={row.get('dist_top_mean', float('nan')):.3f} m "
              f"dist_all={row.get('dist_all_mean', float('nan')):.3f} m", flush=True)
        return row
    finally:
        try:
            env.close()
        except Exception:
            pass


def _keyframe(env, obs, mask_row, pos, centroid, row, out_dir) -> str:
    """Head RGB + stride-16/32 grid, top-coverage cells, projected pos_mean."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .. import masks
    from ..masks import token_grid

    task = env.task
    head = task.cameras.static_camera_list[task.cameras.head_camera_id]
    ext = np.asarray(head.get_extrinsic_matrix(), dtype=np.float64)
    K = np.asarray(head.get_intrinsic_matrix(), dtype=np.float64)
    W, H = masks.ENCODER_IMG_WH
    h0, w0 = obs["rgb"][2].shape[:2]
    K = K.copy()
    K[0, :] *= W / w0
    K[1, :] *= H / h0

    def project(pts_world):
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        cam = pts @ ext[:3, :3].T + ext[:3, 3]
        front = cam[:, 2] > 1e-4
        u = K[0, 0] * cam[:, 0] / np.where(front, cam[:, 2], 1.0) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / np.where(front, cam[:, 2], 1.0) + K[1, 2]
        return u, v, front

    rgb = obs["rgb"][2]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.imshow(rgb)
    scales, _ = token_grid()
    offset = 0
    for s, rows_n, cols_n in scales:
        for r_i in range(rows_n):
            for c_i in range(cols_n):
                cov = int(mask_row[offset + r_i * cols_n + c_i])
                if cov >= COVERAGE_MIN:
                    rect = plt.Rectangle((c_i * s, r_i * s), s, s, fill=True,
                                         facecolor=(1, 0, 0, 0.35), edgecolor="red", lw=0.8)
                    ax.add_patch(rect)
        offset += rows_n * cols_n
    u, v, front = project(pos)
    ax.scatter(u[front], v[front], s=4, c="cyan", label="pos_mean")
    if row["k"]:
        nn = np.array(row["nearest"], dtype=int)
        ax.scatter(u[nn], v[nn], s=42, facecolors="none", edgecolors="yellow",
                   linewidths=1.6, label=f"{row['k']} nearest to centroid")
    cu, cv, cfront = project(centroid)
    if bool(cfront[0]):
        ax.scatter([cu[0]], [cv[0]], s=90, marker="x", c="lime", linewidths=2.4,
                   label="item centroid")
    ax.set_title(f"P5 {row['camera']}: overlap={row['overlap']} (k={row['k']})")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    path = Path(out_dir) / "keyframes" / f"p5_{row['camera']}_seed{row['seed']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    # sidecar: the raw head frame for inspection
    imageio.imwrite(str(path.with_name(path.stem + "_rgb.png")), rgb)
    return str(path)


def run(out_dir, cells=None, shard: str = "0/1", seed: int = 0,
        embodiment: str = "ur5-wsg", camera: str = "head", **kw) -> dict:
    """Single-scene probe; resumable on cell ("p5", camera, seed)."""
    out_dir = Path(out_dir).resolve()
    rows = load_rows(out_dir / ROWS_FILE)
    if ("p5", camera, int(seed)) in done_cells(rows):
        print(f"[p5] cell ('p5', {camera!r}, {seed}) already done; skipping", flush=True)
        return {"ran": 0, "rows": str(out_dir / ROWS_FILE)}
    row = check_once(out_dir, seed=seed, embodiment=embodiment, camera=camera)
    return {"ran": 1, "overlap": row["overlap"], "k": row["k"], "rows": str(out_dir / ROWS_FILE)}


def summarize(out_dir) -> dict:
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / ROWS_FILE)
    latest = rows[-1] if rows else {}
    summary = {
        "n_rows": len(rows),
        "overlap": latest.get("overlap"),
        "k": latest.get("k"),
        "overlap_relaxed_25pct": latest.get("overlap_relaxed_25pct"),
        "dist_top_mean": latest.get("dist_top_mean"),
        "dist_all_mean": latest.get("dist_all_mean"),
        "gate_p5": bool(latest.get("k")) and latest.get("overlap") is not None
        and latest["overlap"] >= 0.5,
    }
    with open(out_dir / "p5_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", default="ur5-wsg")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2))
    else:
        run(args.out, seed=args.seed, embodiment=args.embodiment, camera=args.camera)
