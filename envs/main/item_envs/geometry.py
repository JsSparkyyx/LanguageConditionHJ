"""Rigid-body geometry helpers: quaternions, OBBs, footprints, polygons.

Pure numpy, SI units (metres, radians). Quaternions are wxyz. No SAPIEN imports.
"""

from __future__ import annotations

import itertools

import numpy as np

_CORNER_SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))  # [8, 3], z fastest


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> [3, 3] rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_to_mat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """position [3] + wxyz quaternion -> [4, 4] homogeneous transform."""
    mat = np.eye(4)
    mat[:3, :3] = quat_wxyz_to_mat(q)
    mat[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return mat


def obb_corners(world_T_item: np.ndarray, center_local: np.ndarray, half: np.ndarray) -> np.ndarray:
    """World-frame [8, 3] corners of the item OBB (center/half extents in the item frame)."""
    world_T_item = np.asarray(world_T_item, dtype=np.float64)
    center_local = np.asarray(center_local, dtype=np.float64).reshape(3)
    half = np.asarray(half, dtype=np.float64).reshape(3)
    local = center_local[None, :] + _CORNER_SIGNS * half[None, :]
    return local @ world_T_item[:3, :3].T + world_T_item[:3, 3]


def footprint(
    world_T_item: np.ndarray,
    center_local: np.ndarray,
    half: np.ndarray,
    up_local: np.ndarray,
) -> np.ndarray:
    """World-frame [4, 3] corners of the OBB face the item stands on.

    The face most aligned with -world_z; when the item is tilted, the lowest
    face's 4 corners. `up_local` is accepted for interface symmetry with
    `tilt_angle` (the standing face follows from the rotation alone).
    """
    del up_local
    world_T_item = np.asarray(world_T_item, dtype=np.float64)
    rot = world_T_item[:3, :3]
    center_local = np.asarray(center_local, dtype=np.float64).reshape(3)
    half = np.asarray(half, dtype=np.float64).reshape(3)
    # World +z in the item frame is the third row of rot; the item axis with the
    # largest |z component| is the most vertical one.
    z_in_item = rot[2, :]
    axis = int(np.argmax(np.abs(z_in_item)))
    face_center = center_local.copy()
    face_center[axis] -= np.sign(z_in_item[axis]) * half[axis]
    others = [i for i in range(3) if i != axis]
    corners = []
    for sa, sb in itertools.product([-1.0, 1.0], repeat=2):
        pt = face_center.copy()
        pt[others[0]] += sa * half[others[0]]
        pt[others[1]] += sb * half[others[1]]
        corners.append(pt)
    return np.asarray(corners) @ rot.T + world_T_item[:3, 3]


def signed_dist_inside_polygon(pts_xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance to the nearest polygon edge, metres.

    Positive inside, negative outside, zero on the boundary. pts_xy [N, 2] (or
    [2]), poly [M, 2] (CCW or CW; the inside test is even-odd, so orientation
    does not matter).
    """
    pts = np.atleast_2d(np.asarray(pts_xy, dtype=np.float64))
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    a = poly  # [M, 2] edge starts
    b = np.roll(poly, -1, axis=0)  # [M, 2] edge ends
    ab = b - a  # [M, 2]
    ap = pts[:, None, :] - a[None, :, :]  # [N, M, 2]
    denom = np.einsum("me,me->m", ab, ab)
    denom = np.where(denom < 1e-24, 1.0, denom)
    t = np.clip(np.einsum("nme,me->nm", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # [N, M, 2]
    dist = np.linalg.norm(pts[:, None, :] - closest, axis=-1).min(axis=1)  # [N]
    # Even-odd inside test: count edges crossing the horizontal ray to +inf x.
    y0, y1 = a[:, 1], b[:, 1]
    x0, x1 = a[:, 0], b[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):  # horizontal edges: x_cross unused
        x_cross = x0[None, :] + (pts[:, None, 1] - y0[None, :]) * (x1 - x0)[None, :] / (y1 - y0)[None, :]
    crosses = ((y0[None, :] > pts[:, None, 1]) != (y1[None, :] > pts[:, None, 1])) & (pts[:, None, 0] < x_cross)
    inside = crosses.sum(axis=1) % 2 == 1
    return np.where(inside, dist, -dist)


def circle_polygon(radius: float, n: int = 16) -> np.ndarray:
    """CCW n-gon approximation of a circle in the xy plane, centered at the origin.

    Returns [n, 2]; vertex k at angle 2*pi*k/n (counterclockwise seen from +z).
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if int(n) < 3:
        raise ValueError("need at least 3 vertices")
    theta = 2.0 * np.pi * np.arange(int(n)) / int(n)
    return radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def tilt_angle(world_T_item: np.ndarray, up_local: np.ndarray) -> float:
    """Angle between the item's up axis and world +z, radians."""
    world_T_item = np.asarray(world_T_item, dtype=np.float64)
    up_local = np.asarray(up_local, dtype=np.float64).reshape(3)
    n = np.linalg.norm(up_local)
    if n < 1e-12:
        raise ValueError("zero up axis")
    up_world = world_T_item[:3, :3] @ (up_local / n)
    return float(np.arccos(np.clip(up_world[2], -1.0, 1.0)))


def to_frame(points_world: np.ndarray, world_T_frame: np.ndarray) -> np.ndarray:
    """Express world-frame points [..., 3] in `world_T_frame`'s local frame."""
    world_T_frame = np.asarray(world_T_frame, dtype=np.float64)
    points_world = np.asarray(points_world, dtype=np.float64)
    return (points_world - world_T_frame[:3, 3]) @ world_T_frame[:3, :3]
