"""Item token masks on the HoloBrain-Qwen grid (V6, qwen_impl.md 4).

The GD masks (envs/main/item_envs/masks.py) pool each item's segmentation on the 320 x 256 image at strides 16
and 32 (400 cells per camera). The Qwen scene tokens are one 28-pixel grid on the 308 x 252 image: 9 rows x 11
columns = 99 per camera, cameras [left_wrist, right_wrist, head], row-major, the order the 297 scene tokens are
stored in. ``qwen_item_masks`` is masks.item_masks with that grid (the same pictures, the same NEAREST resize of
the id map, the same visibility from the native picture); ``QwenGridMarginHFn`` writes them into the item sidecar
rows, so a Qwen shard's ``token_masks`` are [K + G, 3, 99] and the probe's ``ref_mask`` lines up with the scene
tokens. ``use_qwen_grid(ro)`` switches a wired controller's h_fn (wire_item_episode builds an ItemMarginHFn).
"""

from __future__ import annotations

import numpy as np

import main.lang  # noqa: F401  (puts envs/main on main.__path__)
from main.item_envs import masks
from main.item_envs.items import entity_of
from main.item_envs.margin_hook import ItemMarginHFn

QWEN_IMAGE_WH = (308, 252)
QWEN_STRIDES = (28,)


def qwen_token_grid():
    """([(28, 9, 11)], 99)."""
    return masks.token_grid(QWEN_IMAGE_WH, QWEN_STRIDES)


QWEN_N_TOK = qwen_token_grid()[1]


def pool_ids_to_qwen(ids_native: np.ndarray, ent_ids) -> np.ndarray:
    """A native id map [H0, W0] and an entity id set -> [99] uint8 cell coverage on the Qwen grid."""
    ids = masks.resize_segmentation_nearest(np.asarray(ids_native), QWEN_IMAGE_WH)
    return masks.pool_mask_to_tokens(np.isin(ids, ent_ids), image_wh=QWEN_IMAGE_WH, strides=QWEN_STRIDES)


def n_mask_rows(items, regions) -> int:
    return len(list(items)) + sum(1 for r in (regions or []) if r.nameable and r.entity is not None)


def qwen_item_masks(task, items, regions, cameras=None) -> tuple[np.ndarray, np.ndarray]:
    """(visibility [K, n_cam] float32, token_masks [K + G_nameable, n_cam, 99] uint8): masks.item_masks on the
    Qwen grid (row order: the items, then the nameable regions with an entity)."""
    specs = list(items.values()) if hasattr(items, "values") else list(items)
    regions = list(regions or [])
    targets = [(spec, True) for spec in specs]
    targets += [(r, False) for r in regions if r.nameable and r.entity is not None]
    cams = masks.resolve_cameras(task, cameras)
    visibility = np.zeros((len(specs), len(cams)), dtype=np.float32)
    token_masks = np.zeros((len(targets), len(cams), QWEN_N_TOK), dtype=np.uint8)
    if not targets:
        return visibility, token_masks
    target_ids = [
        masks._item_entity_ids(obj) if is_item else np.array([int(entity_of(obj.entity).per_scene_id)], dtype=np.int64)
        for obj, is_item in targets
    ]
    task._update_render()  # sync physics -> RT renderer before any picture
    for j, cam in enumerate(cams):
        cam.take_picture()
        seg = cam.get_picture("Segmentation")  # [H, W, 4]; ch 1 = actor id, RAW
        ids_native = np.ascontiguousarray(seg[..., 1]).astype(np.int64)
        h0, w0 = ids_native.shape[:2]
        ids_qwen = masks.resize_segmentation_nearest(ids_native, QWEN_IMAGE_WH)
        K_mat = ext = None
        for t, ent_ids in enumerate(target_ids):
            obj, is_item = targets[t]
            if is_item:
                if K_mat is None:
                    K_mat = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
                    ext = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
                visibility[t, j] = masks._visible_fraction(np.isin(ids_native, ent_ids), obj, K_mat, ext, (w0, h0))
            token_masks[t, j] = masks.pool_mask_to_tokens(np.isin(ids_qwen, ent_ids), image_wh=QWEN_IMAGE_WH,
                                                          strides=QWEN_STRIDES)
    return visibility, token_masks


class QwenGridMarginHFn(ItemMarginHFn):
    """ItemMarginHFn whose token_masks are on the Qwen grid (one render per tick, as the parent's)."""

    def _compute_tick(self, env):
        want = self.with_masks
        self.with_masks = False          # the parent then skips its GD-grid pictures (zeros we overwrite)
        try:
            data = super()._compute_tick(env)
        finally:
            self.with_masks = want
        ctx = self.ctx
        n_cam = len(masks.CAMERA_NAMES)
        n_rows = n_mask_rows(ctx.items, ctx.regions)
        if want:
            try:
                data.visibility, data.token_masks = qwen_item_masks(env.task, ctx.items, ctx.regions)
            except Exception as exc:   # the parent's fallback: NaN visibility, zero masks
                print(f"[qwen_masks] item_masks failed once (NaN/zeros): {exc}")
                data.visibility = np.full((len(list(ctx.items)), n_cam), np.nan, dtype=np.float32)
                data.token_masks = np.zeros((n_rows, n_cam, QWEN_N_TOK), dtype=np.uint8)
        else:
            data.token_masks = np.zeros((n_rows, n_cam, QWEN_N_TOK), dtype=np.uint8)
        return data


def use_qwen_grid(ro) -> None:
    """Switch ro.h_fn (an ItemMarginHFn from wire_item_episode) to the Qwen grid, keeping its state."""
    h = ro.h_fn
    if type(h) is QwenGridMarginHFn:
        return
    if type(h) is not ItemMarginHFn:
        raise TypeError(f"use_qwen_grid: ro.h_fn is {type(h).__name__}, expected the ItemMarginHFn of "
                        "wire_item_episode")
    h.__class__ = QwenGridMarginHFn
