"""Pure-numpy tests for main.item_envs.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from main.item_envs.geometry import (
    footprint,
    obb_corners,
    pose_to_mat,
    quat_wxyz_to_mat,
    signed_dist_inside_polygon,
    tilt_angle,
    to_frame,
)

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_quat_identity_and_known_rotation():
    assert np.allclose(quat_wxyz_to_mat([1, 0, 0, 0]), np.eye(3))
    # 90 deg about z: x -> y, y -> -x
    r = quat_wxyz_to_mat([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(r @ [0, 1, 0], [-1, 0, 0], atol=1e-12)


def test_quat_random_orthonormal():
    rng = np.random.RandomState(0)
    for _ in range(20):
        q = rng.randn(4)
        r = quat_wxyz_to_mat(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(r), 1.0)


def test_quat_unnormalized_matches_normalized():
    q = [0.5, 0.5, -0.5, 0.5]
    assert np.allclose(quat_wxyz_to_mat(q), quat_wxyz_to_mat(7.0 * np.array(q)))


def test_pose_to_mat_blocks_and_inverse():
    p = np.array([0.1, -0.2, 0.3])
    q = np.array([np.cos(0.2), 0, np.sin(0.2), 0.0])
    mat = pose_to_mat(p, q)
    assert mat.shape == (4, 4)
    assert np.allclose(mat[:3, 3], p)
    assert np.allclose(mat[:3, :3], quat_wxyz_to_mat(q))
    assert np.allclose(mat[3], [0, 0, 0, 1])
    pts = np.array([[1.0, 2.0, 3.0]])
    back = to_frame(pts @ mat[:3, :3].T + p, mat)
    assert np.allclose(back, pts, atol=1e-12)


def test_obb_corners_identity():
    corners = obb_corners(np.eye(4), np.zeros(3), np.array([1.0, 2.0, 3.0]))
    assert corners.shape == (8, 3)
    for c in corners:
        assert abs(abs(c[0]) - 1.0) < 1e-12
        assert abs(abs(c[1]) - 2.0) < 1e-12
        assert abs(abs(c[2]) - 3.0) < 1e-12
    # offset center and translation
    mat = pose_to_mat([10.0, 0, 0], [1, 0, 0, 0])
    corners = obb_corners(mat, np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isclose(corners[:, 0].min(), 9.0) and np.isclose(corners[:, 0].max(), 11.0)
    assert np.isclose(corners[:, 1].min(), -1.0) and np.isclose(corners[:, 1].max(), 3.0)


def _corner_set(pts):
    return {tuple(np.round(p, 9)) for p in pts}


def test_footprint_upright_box_is_bottom_face():
    center = np.array([0.01, 0.02, 0.05])
    half = np.array([0.03, 0.02, 0.05])
    fp = footprint(np.eye(4), center, half, np.array([0.0, 0.0, 1.0]))
    expected = {
        (0.01 + sx * 0.03, 0.02 + sy * 0.02, 0.0)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    }
    assert _corner_set(fp) == {tuple(np.round(p, 9)) for p in expected}


def test_footprint_tilted_90deg_is_side_face():
    # Rotate 90 deg about world x: item y axis becomes world z, so the face now
    # facing down is the former -y side face.
    center = np.array([0.01, 0.02, 0.05])
    half = np.array([0.03, 0.02, 0.05])
    rot_x90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    mat = np.eye(4)
    mat[:3, :3] = rot_x90
    fp = footprint(mat, center, half, np.array([0.0, 0.0, 1.0]))
    # Former -y face: local (+-hx, cy - hy, cz +- hz) -> world (x, -z, cy - hy).
    expected = set()
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            local = np.array([center[0] + sx * half[0], center[1] - half[1], center[2] + sz * half[2]])
            expected.add(tuple(np.round(rot_x90 @ local, 9)))
    assert _corner_set(fp) == expected
    # all four corners share the lowest world z
    assert np.allclose(fp[:, 2], fp[:, 2].min())


def test_signed_dist_unit_square_values():
    pts = np.array([[0.5, 0.5], [0.5, 0.0], [1.2, 0.5]])
    d = signed_dist_inside_polygon(pts, UNIT_SQUARE)
    assert np.isclose(d[0], 0.5)  # center
    assert np.isclose(d[1], 0.0, atol=1e-12)  # on the edge
    assert np.isclose(d[2], -0.2)  # outside
    # deeper checks: interior offset, exterior corner (nearest point is a vertex)
    d = signed_dist_inside_polygon(np.array([[0.1, 0.5], [-0.1, -0.1], [0.5, 1.0]]), UNIT_SQUARE)
    assert np.isclose(d[0], 0.1)
    assert np.isclose(d[1], -np.hypot(0.1, 0.1))
    assert np.isclose(d[2], 0.0, atol=1e-12)


def test_signed_dist_concave_polygon():
    # L-shape: unit square with the top-right quadrant removed.
    poly = np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]], dtype=float)
    assert np.isclose(signed_dist_inside_polygon(np.array([[0.25, 0.75]]), poly)[0], 0.25)
    assert np.isclose(signed_dist_inside_polygon(np.array([[0.75, 0.75]]), poly)[0], -0.25)  # in the notch
    assert np.isclose(signed_dist_inside_polygon(np.array([[0.6, 0.75]]), poly)[0], -0.1)  # nearest edge x=0.5


def test_signed_dist_orientation_independent():
    pts = np.array([[0.5, 0.5], [1.2, 0.5]])
    cw = UNIT_SQUARE[::-1]
    assert np.allclose(signed_dist_inside_polygon(pts, UNIT_SQUARE), signed_dist_inside_polygon(pts, cw))


def test_tilt_angle():
    up = np.array([0.0, 0.0, 1.0])
    assert np.isclose(tilt_angle(np.eye(4), up), 0.0)
    rot_x90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    mat = np.eye(4)
    mat[:3, :3] = rot_x90
    assert np.isclose(tilt_angle(mat, up), np.pi / 2)
    rot_y30 = np.array(
        [[np.cos(np.pi / 6), 0, np.sin(np.pi / 6)], [0, 1, 0], [-np.sin(np.pi / 6), 0, np.cos(np.pi / 6)]]
    )
    mat[:3, :3] = rot_y30
    assert np.isclose(tilt_angle(mat, up), np.pi / 6)


def test_to_frame_roundtrip_and_single_point():
    mat = pose_to_mat([1.0, 2.0, 3.0], [np.cos(0.3), 0, 0, np.sin(0.3)])
    pts = np.random.RandomState(1).randn(5, 3)
    local = to_frame(pts, mat)
    back = local @ mat[:3, :3].T + mat[:3, 3]
    assert np.allclose(back, pts, atol=1e-12)
    assert to_frame(mat[:3, 3], mat).shape == (3,)
    assert np.allclose(to_frame(mat[:3, 3], mat), np.zeros(3), atol=1e-12)


def test_quat_zero_raises():
    with pytest.raises(ValueError):
        quat_wxyz_to_mat(np.zeros(4))
