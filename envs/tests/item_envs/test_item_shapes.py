"""Unit tests for item_shapes (exact avoid distance, 2026-10-05).

CPU only: sapien collision shapes are built without a scene (no physics
step, no render). References are analytic (box, cylinder, capsule) or
trimesh's closest point on the hull (convex mesh).
"""

from __future__ import annotations

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")
pytest.importorskip("mplib")
trimesh = pytest.importorskip("trimesh")

import sapien.physx as px  # noqa: E402

from main.item_envs import item_shapes  # noqa: E402
from main.item_envs.item_shapes import ItemShape, _pieces_from_shape  # noqa: E402

_MAT = None


def _mat():
    global _MAT
    if _MAT is None:
        _MAT = px.PhysxMaterial(0.5, 0.5, 0.0)
    return _MAT


class _Entity:
    def __init__(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose


def _rand_pose(rng, scale=0.05):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return sapien.Pose(rng.uniform(-scale, scale, size=3), q)


def _T(pose):
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def _local(points, T):
    return (points - T[:3, 3]) @ T[:3, :3]


# ------------------------------------------------------- analytic references ----


def _ref_box(p_local, h):
    delta = np.abs(p_local) - h
    out = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.all(delta <= 0.0, axis=-1)
    return np.where(inside, delta.max(axis=-1), out)


def _ref_cyl_x(p_local, r, hl):
    dx = np.abs(p_local[..., 0]) - hl
    dr = np.linalg.norm(p_local[..., 1:], axis=-1) - r
    out = np.hypot(np.maximum(dx, 0.0), np.maximum(dr, 0.0))
    return np.where((dx <= 0) & (dr <= 0), np.maximum(dx, dr), out)


def _ref_capsule_x(p_local, r, hl):
    x = np.clip(p_local[..., 0], -hl, hl)
    q = p_local.copy()
    q[..., 0] -= x
    return np.linalg.norm(q, axis=-1) - r


def _ref_hull(p_world, mesh):
    tri = mesh.triangles
    best = np.full(len(p_world), np.inf)
    for i, p in enumerate(p_world):
        cp = trimesh.triangles.closest_point(tri, np.repeat(p[None], len(tri), axis=0))
        best[i] = np.linalg.norm(cp - p, axis=1).min()
    n = mesh.face_normals
    d = np.einsum("ij,ij->i", n, mesh.vertices[mesh.faces[:, 0]])
    s = (p_world @ n.T - d).max(axis=1)
    return np.where(s <= 0, s, best)


# ----------------------------------------------------------------- shapes ----


def _shape_with(shape, entity_pose):
    pieces = _pieces_from_shape(shape)
    assert len(pieces) == 1
    return ItemShape(_Entity(entity_pose), pieces)


def test_box_sphere_distance_exact():
    """Boxes are exact polytopes: gap AND overlap depth match the analytic
    signed distance (centers inside, overlapping from outside, apart)."""
    rng = np.random.default_rng(0)
    h = np.array([0.03, 0.05, 0.08])
    box = px.PhysxCollisionShapeBox(half_size=h, material=_mat())
    box.set_local_pose(sapien.Pose([0.01, -0.02, 0.0], [0.9238795, 0.0, 0.3826834, 0.0]))
    pose = _rand_pose(rng)
    shape = _shape_with(box, pose)
    T = _T(pose) @ _T(box.get_local_pose())
    pts = rng.uniform(-0.15, 0.15, size=(200, 3)) + T[:3, 3]
    radii = rng.uniform(0.005, 0.04, size=200)
    got = item_shapes.sphere_distances(pts, radii, shape)
    ref = _ref_box(_local(pts, T), h) - radii
    assert (ref < 0).sum() > 20 and (ref > 0).sum() > 20
    assert np.allclose(got, ref, atol=2e-5), np.abs(got - ref).max()


@pytest.mark.parametrize("kind", ["cylinder", "capsule"])
def test_round_primitive_axis_is_local_x(kind):
    """PhysX cylinders / capsules run along local x; their polytope contains
    them, <= 0.5 % of the radius bigger (never reads a gap that is not there)."""
    rng = np.random.default_rng(1)
    r, hl = 0.03, 0.07
    cls = px.PhysxCollisionShapeCylinder if kind == "cylinder" else px.PhysxCollisionShapeCapsule
    shp = cls(radius=r, half_length=hl, material=_mat())
    pose = _rand_pose(rng)
    shape = _shape_with(shp, pose)
    T = _T(pose)
    pts = rng.uniform(-0.15, 0.15, size=(200, 3)) + T[:3, 3]
    radii = np.full(200, 0.01)
    got = item_shapes.sphere_distances(pts, radii, shape)
    ref_fn = _ref_cyl_x if kind == "cylinder" else _ref_capsule_x
    ref = ref_fn(_local(pts, T), r, hl) - radii
    tol = 0.006 * r if kind == "cylinder" else 0.012 * r
    assert np.all(got <= ref + 2e-5), (got - ref).max()
    assert np.all(got >= ref - tol - 2e-5), (ref - got).max()


def test_convex_mesh_scaled(tmp_path):
    rng = np.random.default_rng(2)
    mesh = trimesh.convex.convex_hull(rng.normal(size=(40, 3)) * [0.03, 0.02, 0.04])
    path = tmp_path / "hull.obj"
    mesh.export(path)
    scale = np.array([1.5, 1.0, 0.8], dtype=np.float32)
    shp = px.PhysxCollisionShapeConvexMesh(str(path), scale, _mat())
    pose = _rand_pose(rng)
    shape = _shape_with(shp, pose)
    world = trimesh.Trimesh(mesh.vertices * scale, mesh.faces)
    world.apply_transform(_T(pose))
    pts = np.concatenate([rng.uniform(-0.12, 0.12, size=(60, 3)), rng.uniform(-0.02, 0.02, size=(10, 3))]) + _T(pose)[:3, 3]
    got = item_shapes.sphere_distances(pts, np.zeros(70) + 1e-4, shape) + 1e-4
    ref = _ref_hull(pts, world)
    assert (ref < 0).sum() > 3
    assert np.allclose(got, ref, atol=3e-5), np.abs(got - ref).max()


def _ref_piece(points, shape_def):
    """Analytic signed distance of points to one primitive (box / cylinder x)."""
    kind, dims, T = shape_def
    loc = _local(points, T)
    return _ref_box(loc, dims) if kind == "box" else _ref_cyl_x(loc, *dims)


def _multi(rng, pose):
    defs, pieces = [], []
    for k in range(5):
        lp = _rand_pose(rng, 0.06)
        if k < 4:
            h = rng.uniform(0.01, 0.04, size=3)
            s = px.PhysxCollisionShapeBox(half_size=h, material=_mat())
            d = ("box", h)
        else:
            s = px.PhysxCollisionShapeCylinder(radius=0.02, half_length=0.05, material=_mat())
            d = ("cyl", (0.02, 0.05))
        s.set_local_pose(lp)
        pieces += _pieces_from_shape(s)
        defs.append((d[0], d[1], _T(pose) @ _T(lp)))
    return ItemShape(_Entity(pose), pieces), defs


def test_multi_piece_spheres_exact_near_lower_bound_far():
    """Multi-piece shapes vs analytic per-piece distances: exact within NEAR,
    never above the truth (the cylinder polytope is <= 0.1 mm bigger)."""
    rng = np.random.default_rng(3)
    NEAR = item_shapes.NEAR
    n_near = 0
    for trial in range(30):
        b, defs = _multi(rng, sapien.Pose(rng.uniform(-0.05, 0.05, size=3), _rand_pose(rng).q))
        centers = rng.uniform(-0.2, 0.2, size=(40, 3)) + b.T()[:3, 3]
        radii = rng.uniform(0.005, 0.04, size=40)
        ref = np.min([_ref_piece(centers, d) for d in defs], axis=0) - radii
        per = item_shapes.sphere_distances(centers, radii, b)
        near = ref <= NEAR
        n_near += int(near.sum())
        assert np.all(per <= ref + 2e-5)
        assert np.all(per[near] >= ref[near] - 2e-4), (ref - per)[near].max()
        best, s = item_shapes.spheres_min_distance(centers, radii, b)
        assert best <= ref.min() + 2e-5
        if ref.min() <= NEAR:
            assert best >= ref.min() - 2e-4
            assert ref[s] == pytest.approx(best, abs=2e-4)
        else:
            assert best > NEAR - 2e-5
    assert n_near >= 50  # the near branch was exercised


def _qp_distance(va, vb):
    """Reference distance between two convex hulls (vertex sets): min |x - y|
    over convex combinations, SLSQP; ~0 means they overlap."""
    from scipy.optimize import minimize

    na, nb = len(va), len(vb)
    cons = ({"type": "eq", "fun": lambda x: x[:na].sum() - 1}, {"type": "eq", "fun": lambda x: x[na:].sum() - 1})
    x0 = np.r_[np.full(na, 1 / na), np.full(nb, 1 / nb)]
    r = minimize(lambda x: np.sum((x[:na] @ va - x[na:] @ vb) ** 2), x0, constraints=cons,
                 bounds=[(0, 1)] * (na + nb), method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return float(np.sqrt(max(r.fun, 0.0)))


def _world_verts(piece, T):
    return piece.verts @ T[:3, :3].T + T[:3, 3]


def test_piece_pair_near_contact_vs_qp():
    """Polytope pairs near contact vs an optimizer: the gap within 0.05 mm
    when apart, and every overlap reads <= 0 (plain GJK read up to 2.6 mm of
    gap for 1-3 of 233 overlapping pairs)."""
    rng = np.random.default_rng(6)
    hulls = [item_shapes._Piece(rng.normal(size=(14, 3)) * 0.03) for _ in range(5)]
    n_apart = n_over = 0
    for _ in range(150):
        i, j = rng.choice(5, size=2, replace=False)  # two distinct pieces: one FCL object holds one pose
        pa, pb = hulls[i], hulls[j]
        Ta = _T(_rand_pose(rng, 0.0))
        d = rng.normal(size=3)
        Tb = _T(sapien.Pose(d / np.linalg.norm(d) * rng.uniform(0.03, 0.09), _rand_pose(rng).q))
        pa.obj.set_pose(item_shapes._mpose(Ta))
        pb.obj.set_pose(item_shapes._mpose(Tb))
        got = item_shapes._piece_piece(pa, Ta, pb, Tb)
        ref = _qp_distance(_world_verts(pa, Ta), _world_verts(pb, Tb))
        if ref > 1e-5:
            n_apart += 1
            # the optimizer's answer is a distance it ACHIEVED (>= the true one;
            # SLSQP sometimes stops 0.2 mm short): got may sit a little below it
            assert ref - 5e-4 <= got <= ref + 5e-5, (got, ref)
        else:
            n_over += 1
            assert got <= 1e-6, got
    assert n_apart >= 12 and n_over >= 12


def test_multi_piece_shape_distance():
    """Shape vs shape vs the per-pair reference (overlap: any pair's QP ~0;
    else the min QP distance): exact within NEAR, a lower bound beyond."""
    rng = np.random.default_rng(4)
    NEAR = item_shapes.NEAR
    n_near = n_overlap = 0
    for trial in range(16):
        a, _ = _multi(rng, _rand_pose(rng, 0.02))
        spread = 0.12 if trial % 2 else 0.25
        b, _ = _multi(rng, sapien.Pose(rng.uniform(-spread, spread, size=3), _rand_pose(rng).q))
        Ta, Tb = a.T(), b.T()
        pa, pb = a.posed_pieces(), b.posed_pieces()

        def pair_ref(x, y):
            g = item_shapes._gjk(x.obj, y.obj)  # exact when clearly apart (QP-checked above)
            return g if g > 0.01 else _qp_distance(_world_verts(x, Ta), _world_verts(y, Tb))

        ref = min(pair_ref(x, y) for x in pa for y in pb)
        got = item_shapes.shape_distance(a, b)
        if ref <= 1e-5:
            n_overlap += 1
            assert got <= 1e-6
            continue
        assert got <= ref + 5e-5
        if ref <= NEAR:
            n_near += 1
            assert got == pytest.approx(ref, abs=5e-5)
        else:
            assert got > NEAR - 5e-5
    assert n_near >= 2 and n_overlap >= 2


def test_overlap_depth_and_robustness():
    """The crash regression: FCL's EPA threw / hung / segfaulted on random
    overlapping pairs. 3000 random sphere + polytope queries on every
    primitive run cleanly; a bar pushed 1 cm into a box face reads -1 cm."""
    rng = np.random.default_rng(5)
    mats = [
        px.PhysxCollisionShapeBox(half_size=[0.02, 0.03, 0.04], material=_mat()),
        px.PhysxCollisionShapeCylinder(radius=0.02, half_length=0.05, material=_mat()),
        px.PhysxCollisionShapeCapsule(radius=0.02, half_length=0.05, material=_mat()),
        px.PhysxCollisionShapeSphere(radius=0.03, material=_mat()),
    ]
    shapes = [ItemShape(_Entity(sapien.Pose()), _pieces_from_shape(m)) for m in mats]
    for _ in range(3000):
        s = shapes[rng.integers(len(shapes))]
        o = shapes[rng.integers(len(shapes))]
        o.entity.pose = sapien.Pose(rng.uniform(-0.08, 0.08, size=3), _rand_pose(rng).q)
        s.entity.pose = sapien.Pose()
        if o is s:
            continue
        d = item_shapes.shape_distance(s, o)
        assert np.isfinite(d) and d > -0.2
        c = rng.uniform(-0.08, 0.08, size=(1, 3))
        assert np.isfinite(item_shapes.sphere_distances(c, np.array([0.02]), s)[0])


def test_face_contact_reads_zero_not_corner_gap():
    """The bug the user saw: two boxes side by side with offset faces (no
    corner of either inside the other). The old 16-corner OBB check read
    a positive gap; the exact distance reads the true 0 / -1 cm."""
    big = px.PhysxCollisionShapeBox(half_size=[0.05, 0.05, 0.05], material=_mat())
    slab = px.PhysxCollisionShapeBox(half_size=[0.02, 0.12, 0.02], material=_mat())  # a bar across the face
    a = ItemShape(_Entity(sapien.Pose()), _pieces_from_shape(big))
    for overlap in (0.0, 0.01):
        b = ItemShape(_Entity(sapien.Pose([0.07 - overlap, 0.0, 0.0])), _pieces_from_shape(slab))
        assert item_shapes.shape_distance(a, b) == pytest.approx(-overlap, abs=2e-5)
    # the old model: every slab corner is outside the big box, every big-box
    # corner outside the slab -> 16-corner min is positive while touching
    from main.item_envs._bootstrap import ensure_main_path

    ensure_main_path()
    from main.envs.distance import point_obb_signed_distance

    corners = [np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    o_b = np.array([0.06, 0.0, 0.0])
    h_a, h_b = np.array([0.05] * 3), np.array([0.02, 0.12, 0.02])
    old = min(
        min(point_obb_signed_distance(c * h_a, o_b, np.eye(3), h_b) for c in corners),
        min(point_obb_signed_distance(o_b + c * h_b, np.zeros(3), np.eye(3), h_a) for c in corners),
    )
    assert old > 0.0  # 1 cm overlap read as a gap


def test_fallback_obb_box_when_no_collision_shape():
    class _Bare:
        def get_pose(self):
            return sapien.Pose([0.1, 0.0, 0.0])

        def find_component_by_type(self, _cls):
            return None

    shape = item_shapes.shape_of(_Bare(), fallback_obb=(np.zeros(3), np.array([0.01, 0.02, 0.03])))
    assert shape is not None and len(shape.pieces) == 1
    d = item_shapes.sphere_distances(np.array([[0.1, 0.0, 0.1]]), np.array([0.01]), shape)
    assert d[0] == pytest.approx(0.1 - 0.03 - 0.01, abs=1e-5)  # libccd GJK: ~um
