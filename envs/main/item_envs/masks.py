"""Per-item visibility and token-grid masks from segmentation (PLAN v3 §2.4).

Segmentation: ``camera.get_picture("Segmentation")``, channel 1 = actor-level
id, read RAW (RoboTwin's helper casts to uint8 and truncates ids > 255 — recon
Q8). The id equals ``entity.per_scene_id`` (static evidence in recon Q8;
runtime confirmation is the sim test / PENDING-GPU item there).

Token grid (recon Q15, matches HoloBrainEncoder.encode_visual_tokens):
cameras [left_wrist, right_wrist, head]; native 320x240 images resized to
(W, H) = (320, 256) exactly as ``Env.get_encoder_obs`` does — for the id map
that resize is NEAREST, never bilinear. Feature strides 16 and 32 give
16x20 = 320 tokens (finest first) then 8x10 = 80 tokens per camera, row-major
within each scale, so n_tok = 400 per camera. token_masks is
[K + G_nameable, n_cam, 400] uint8 (fraction of each token cell covered,
0..255); ``token_masks.reshape(K + G, -1)`` yields the encoder's flattened
1200-token camera-major order (buffer token index = cam*400 + scale_offset +
row*cols + col).

A particle_set item's mask is the union of its particles. Nameable regions
with an entity (zone mats, containers) are appended after the K items in the
order regions are passed in. ``task._update_render()`` runs before any
picture (RT renderer keeps a stale scene copy otherwise).

Cadence (every tick in collection mode, only t=0 in no-render calibration) is
the runner's concern, not this module's.
"""

from __future__ import annotations

import numpy as np

from .geometry import obb_corners
from .items import entity_of

CAMERA_NAMES = ("left_wrist", "right_wrist", "head")  # encoder order (head LAST)
ENCODER_IMG_WH = (320, 256)  # (W, H) — Env.ENCODER_IMG_WH in main/envs/env.py
TOKEN_STRIDES = (16, 32)  # encoder feature_level (1, 2) -> strides 16, 32


def token_grid(image_wh=ENCODER_IMG_WH, strides=TOKEN_STRIDES) -> tuple[list, int]:
    """(scales, n_tok): per-scale (stride, rows, cols) and tokens per camera.

    Scale order: finest stride first; token order row-major (row, col) within
    each scale. token (stride s, row r, col c) covers pixels rows
    [s*r, s*(r+1)) x cols [s*c, s*(c+1)) of the resized image.
    """
    w, h = int(image_wh[0]), int(image_wh[1])
    scales = []
    for s in strides:
        s = int(s)
        if h % s or w % s:
            raise ValueError(f"image {image_wh} not divisible by stride {s}")
        scales.append((s, h // s, w // s))
    return scales, sum(r * c for _, r, c in scales)


def token_index(stride: int, row: int, col: int, image_wh=ENCODER_IMG_WH, strides=TOKEN_STRIDES) -> int:
    """Index of token (stride, row, col) within one camera's token axis."""
    scales, _ = token_grid(image_wh, strides)
    offset = 0
    for s, rows, cols in scales:
        if s == int(stride):
            if not (0 <= row < rows and 0 <= col < cols):
                raise IndexError(f"token ({stride}, {row}, {col}) outside the {rows}x{cols} grid")
            return offset + int(row) * cols + int(col)
        offset += rows * cols
    raise ValueError(f"stride {stride} not in {tuple(strides)}")


def pool_mask_to_tokens(mask: np.ndarray, image_wh=ENCODER_IMG_WH, strides=TOKEN_STRIDES) -> np.ndarray:
    """Average-pool a binary [H, W] mask into token cells; [n_tok] uint8 0..255.

    mask at the encoder resolution (H, W) = image_wh[::-1]. Value = round(255
    x fraction of the cell's pixels covered). Pure numpy.
    """
    m = np.asarray(mask)
    if m.shape != (int(image_wh[1]), int(image_wh[0])):
        raise ValueError(f"mask shape {m.shape} != {(int(image_wh[1]), int(image_wh[0]))}")
    m = m.astype(np.float64)
    scales, _ = token_grid(image_wh, strides)
    parts = []
    for s, rows, cols in scales:
        cells = m.reshape(rows, s, cols, s).mean(axis=(1, 3))
        parts.append(cells.reshape(-1))
    frac = np.concatenate(parts)
    return np.rint(frac * 255.0).clip(0, 255).astype(np.uint8)


def resize_segmentation_nearest(ids: np.ndarray, image_wh=ENCODER_IMG_WH) -> np.ndarray:
    """The encoder's resize (to image_wh) applied to an id map, NEAREST only.

    Index convention matches cv2.resize(..., interpolation=cv2.INTER_NEAREST)
    on this upsampling path (verified empirically): src = floor(dst *
    src_size / dst_size) in exact integer arithmetic — each stride-16 output
    cell then draws on exactly 15 native rows (240/16), one of them repeated.
    Pure numpy, dtype-preserving (cv2 silently retypes int64 input).
    """
    ids = np.asarray(ids)
    w, h = int(image_wh[0]), int(image_wh[1])
    h0, w0 = ids.shape[:2]
    if (w0, h0) == (w, h):
        return ids
    rows = np.minimum((np.arange(h, dtype=np.int64) * h0) // h, h0 - 1)
    cols = np.minimum((np.arange(w, dtype=np.int64) * w0) // w, w0 - 1)
    return ids[rows][:, cols]


def _seg_id_map(task) -> dict[int, object]:
    """per_scene_id -> entity for every scene entity (the seg channel-1 key)."""
    return {int(ent.per_scene_id): ent for ent in task.scene.entities}


def project_obb_area(
    world_T_item: np.ndarray,
    center_local: np.ndarray,
    half: np.ndarray,
    K: np.ndarray,
    ext: np.ndarray,
    image_wh: tuple[int, int],
) -> float:
    """Cheap upper bound on the item's image area, in pixels.

    The axis-aligned bounding rect of the 8 projected OBB corners, clipped to
    the image. ext is the world->camera extrinsic (OpenCV convention: camera
    looks along +z, x right, y down — same as main/envs/env.py
    camera_extrinsic). Returns 0 when fewer than 2 corners are in front of the
    camera or the projection falls fully off-screen.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (3, 4):
        ext = np.vstack([ext, [0.0, 0.0, 0.0, 1.0]])
    corners = obb_corners(world_T_item, center_local, half)  # [8, 3] world
    cam = corners @ ext[:3, :3].T + ext[:3, 3]
    front = cam[:, 2] > 1e-4
    if int(front.sum()) < 2:
        return 0.0
    pts = cam[front]
    u = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
    v = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]
    w, h = image_wh
    u0, u1 = np.clip([u.min(), u.max()], 0.0, float(w))
    v0, v1 = np.clip([v.min(), v.max()], 0.0, float(h))
    return float(max(u1 - u0, 0.0) * max(v1 - v0, 0.0))


def resolve_cameras(task, cameras=None) -> list:
    """Camera objects for item_masks.

    None -> the encoder's three cameras in CAMERA_NAMES order (left wrist,
    right wrist, head last). A sequence of camera objects is used as-is; a
    sequence of names resolves on task.cameras ("left_wrist"/"right_wrist"/
    "head" aliases plus the RoboTwin attribute names).
    """
    if cameras is None:
        cameras = CAMERA_NAMES
    alias = {
        "left_wrist": "left_camera",
        "right_wrist": "right_camera",
        "head": "head_camera",
    }
    out = []
    for cam in cameras:
        if isinstance(cam, str):
            cams = task.cameras  # lazy: only needed when resolving names
            name = alias.get(cam, cam)
            if name == "head_camera":
                out.append(cams.static_camera_list[cams.head_camera_id])
            else:
                out.append(getattr(cams, name))
        else:
            out.append(cam)
    return out


def _item_entity_ids(spec) -> np.ndarray:
    ents = [entity_of(spec.entity)]
    if spec.kind == "particle_set":
        ents += [entity_of(p) for p in (spec.particles or [])]
    # unique: spec.entity often doubles as the first particle of a particle_set
    return np.unique(np.array([int(e.per_scene_id) for e in ents], dtype=np.int64))


def _visible_fraction(binary_native: np.ndarray, spec, K: np.ndarray, ext: np.ndarray, image_wh) -> float:
    """Segmentation pixels / projected-OBB area estimate, clipped to [0, 1]."""
    pixels = int(binary_native.sum())
    if pixels == 0:
        return 0.0
    ent = entity_of(spec.entity)
    world_T = np.asarray(ent.get_pose().to_transformation_matrix(), dtype=np.float64)
    area = project_obb_area(world_T, spec.obb_center_local, spec.half_extents, K, ext, image_wh)
    if area <= 0:
        return 0.0
    return float(min(1.0, pixels / area))


def item_masks(task, items, regions, cameras=None) -> tuple[np.ndarray, np.ndarray]:
    """(visibility [K, n_cam] float32, token_masks [K + G_nameable, n_cam, n_tok] uint8).

    items: iterable of ItemSpec (or mapping id -> ItemSpec); row order follows
    the iteration order. regions: iterable of Region; nameable regions with a
    non-None entity are appended after the K items in the given order.
    cameras: None (the encoder's three) or a sequence of camera objects/names;
    the sequence order defines the n_cam axis.
    """
    specs = list(items.values()) if hasattr(items, "values") else list(items)
    regions = list(regions or [])
    targets = [(spec, True) for spec in specs]
    targets += [(r, False) for r in regions if r.nameable and r.entity is not None]
    _, n_tok = token_grid()
    cams = resolve_cameras(task, cameras)
    K = len(specs)
    visibility = np.zeros((K, len(cams)), dtype=np.float32)
    token_masks = np.zeros((len(targets), len(cams), n_tok), dtype=np.uint8)
    if not targets:
        return visibility, token_masks
    target_ids = [
        _item_entity_ids(obj) if is_item else np.array([int(entity_of(obj.entity).per_scene_id)], dtype=np.int64)
        for obj, is_item in targets
    ]
    task._update_render()  # sync physics -> RT renderer before any picture
    W, H = ENCODER_IMG_WH
    for j, cam in enumerate(cams):
        cam.take_picture()
        seg = cam.get_picture("Segmentation")  # [H, W, 4]; ch 1 = actor id, RAW
        ids_native = np.ascontiguousarray(seg[..., 1]).astype(np.int64)
        h0, w0 = ids_native.shape[:2]
        if (w0, h0) != (W, H):
            ids_encoder = resize_segmentation_nearest(ids_native)
        else:
            ids_encoder = ids_native
        K_mat, ext = None, None
        for t, ent_ids in enumerate(target_ids):
            binary_native = np.isin(ids_native, ent_ids)
            obj, is_item = targets[t]
            if is_item:
                if K_mat is None:
                    K_mat = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
                    ext = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
                visibility[t, j] = _visible_fraction(binary_native, obj, K_mat, ext, (w0, h0))
            token_masks[t, j] = pool_mask_to_tokens(np.isin(ids_encoder, ent_ids))
    return visibility, token_masks
