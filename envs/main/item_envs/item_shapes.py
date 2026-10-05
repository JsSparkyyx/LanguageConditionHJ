"""Shape-exact item geometry for the avoid distance (user report 2026-10-05).

The avoid distance used to treat every item as its bounding box (OBB) and
the held payload as 8 box corners: a held bowl TOUCHING a bottle read
+0.8 cm ("safe": no corner of either box was inside the other box — a
face or edge contact has no corner in it) and the same bowl ~5 cm away
read -0.6 cm (the boxes' corners stick out of the round bowl by ~3.7 cm
and of the bottle by ~1.4 cm). Here an item is its PhysX COLLISION shapes
— the same convex pieces the simulator touches with — each a convex
polytope ``P = {x : n_f . x <= d_f}`` (boxes and meshes exactly; cylinders
/ capsules / spheres as a 32-sided polytope CONTAINING them, <= 0.5 % of
the radius larger).

Distances (m; positive = gap, negative = penetration depth, the CEHJ
convention):

- sphere -> polytope: exact. Center inside: ``max_f (n_f . c - d_f) - r``;
  outside: the GJK point distance - r.
- polytope -> polytope: apart by more than 5 mm, FCL's GJK distance
  (mplib's bindings), exact; closer, FCL's collision test decides whether
  they overlap; overlapping, the depth along the best FACE normal of either
  polytope (at least the true depth: edge-edge axes are not tried —
  conservative). Overlap past 5 cm stops the search (h is floored there).

FCL's own signed distance (EPA) is NOT used: on random overlapping pairs
it threw ("origin is outside of the polytope"), hung, or segfaulted within
a few thousand calls (box / cylinder / capsule / convex; 2026-10-05). GJK
distances (GST_INDEP) and collision tests ran tens of thousands of random
calls cleanly at 2-5 us each; vs an optimizer, GJK was exact when apart
(<= 0.01 mm) and the collision test right on 302/302 near-contact pairs.

Cost (T1 objects are split into tens of convex pieces): each shape also
keeps the convex HULL of all its pieces. Beyond NEAR = 2 cm the hull
answers — a lower bound of the true distance, exact for a convex object;
the far values only set how big a positive h is. Within NEAR the pieces
answer exactly: pairs are visited by increasing LOWER BOUND until the next
bound cannot beat the best value found (branch and bound), at most
``MAX_EXACT`` per query (then the min of the remaining bounds: still a
lower bound, i.e. conservative).

Lower bounds used: point c vs polytope: ``max_f (n_f . c - d_f)`` (exact
inside, <= the distance outside); a piece's bounding sphere vs the other
shape's hull; two bounding spheres' ``|c_a - c_b| - r_a - r_b``.
"""

from __future__ import annotations

import weakref

import numpy as np

__all__ = ["ItemShape", "shape_of", "sphere_distances", "spheres_min_distance", "shape_distance"]

MAX_EXACT = 64  # exact queries per (spheres | payload) x item query
NEAR = 0.02  # m: below this hull bound the convex pieces are evaluated exactly
KDOP_N = 98  # planes of the multi-piece bounding polytope
DEPTH_ENOUGH = 0.05  # m: overlap this deep reads h = -1 (physics.yaml margin_floor 0.05): stop searching
_ROUND_SIDES = 32  # polygon sides for cylinders / capsules / spheres (circumscribed)
_POINT_R = 1e-7  # m: the "point" of a point-to-polytope GJK query is a sphere this small


def _fcl():
    from mplib.collision_detection import fcl

    return fcl


def _mpose(T: np.ndarray):
    from mplib import Pose

    return Pose(np.ascontiguousarray(T, dtype=np.float64))


def _pose_T(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def _hull(verts: np.ndarray):
    """(vertices, faces, planes[F,4] = n, d with n.x <= d inside) of the hull."""
    import trimesh

    verts = np.asarray(verts, dtype=np.float64)
    try:
        hull = trimesh.convex.convex_hull(verts)
    except Exception:  # flat / degenerate input: joggle
        hull = trimesh.convex.convex_hull(verts, qhull_options="QbB Pp QJ")
    v = np.asarray(hull.vertices, dtype=np.float64)
    f = np.asarray(hull.faces, dtype=np.int32)
    n = np.asarray(hull.face_normals, dtype=np.float64)
    d = np.einsum("ij,ij->i", n, v[f[:, 0]])
    return v, f, np.concatenate([n, d[:, None]], axis=1)


def _ring(radius: float) -> np.ndarray:
    """[K,2] polygon CONTAINING a circle of this radius."""
    r = radius / np.cos(np.pi / _ROUND_SIDES)
    a = 2.0 * np.pi * (np.arange(_ROUND_SIDES) + 0.5) / _ROUND_SIDES
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def _cylinder_x(radius: float, half_len: float) -> np.ndarray:
    ring = _ring(radius)
    side = np.concatenate([np.full((len(ring), 1), half_len), ring], axis=1)
    return np.concatenate([side, side * [-1.0, 1.0, 1.0]], axis=0)


def _ball(radius: float) -> np.ndarray:
    """Vertices of a polytope containing a ball: circumscribed rings at
    latitudes, every ring (and the poles) pushed out by 1/cos of the
    latitude step so the facets between them stay outside the sphere."""
    k = _ROUND_SIDES // 4
    grow = 1.0 / np.cos(np.pi / (4 * k))
    out = []
    for i in range(-k + 1, k):
        phi = (i / k) * (np.pi / 2)
        ring = _ring(radius * np.cos(phi) * grow)
        out.append(np.concatenate([ring, np.full((len(ring), 1), radius * np.sin(phi) * grow)], axis=1))
    out.append(np.array([[0.0, 0.0, radius * grow], [0.0, 0.0, -radius * grow]]))
    return np.concatenate(out, axis=0)


class _Piece:
    """One convex polytope in the ENTITY frame: FCL convex + its planes."""

    __slots__ = ("geom", "verts", "planes", "center", "radius", "obj")

    def __init__(self, verts: np.ndarray):
        v, f, planes = _hull(verts)
        self.geom = _fcl().Convex(v, f, False)  # kept alive: the CollisionObject does not own it
        self.verts = v
        self.planes = planes
        self.center = 0.5 * (v.min(axis=0) + v.max(axis=0))
        self.radius = float(np.linalg.norm(v - self.center, axis=1).max())
        self.obj = _fcl().CollisionObject(self.geom)


_CUBE = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)


def _shape_verts(shape) -> np.ndarray | None:
    """Vertices (shape frame) of the polytope for one sapien.physx collision
    shape; None for shapes that are not item geometry (plane)."""
    import sapien.physx as px

    if isinstance(shape, px.PhysxCollisionShapeBox):
        return _CUBE * np.asarray(shape.get_half_size(), dtype=np.float64)
    if isinstance(shape, px.PhysxCollisionShapeSphere):
        return _ball(float(shape.get_radius()))
    if isinstance(shape, px.PhysxCollisionShapeCylinder):  # PhysX: axis along local x
        return _cylinder_x(float(shape.get_radius()), float(shape.get_half_length()))
    if isinstance(shape, px.PhysxCollisionShapeCapsule):  # axis x, hemispherical caps
        r, hl = float(shape.get_radius()), float(shape.get_half_length())
        b = _ball(r)
        return np.concatenate([b + [hl, 0, 0], b - [hl, 0, 0]], axis=0)
    if isinstance(shape, (px.PhysxCollisionShapeConvexMesh, px.PhysxCollisionShapeTriangleMesh)):
        scale = np.asarray(shape.get_scale(), dtype=np.float64).reshape(-1)
        return np.asarray(shape.get_vertices(), dtype=np.float64) * scale  # a triangle mesh -> its hull
    return None


def _pieces_from_shape(shape) -> list[_Piece]:
    """Polytope pieces (entity frame) for one sapien.physx collision shape."""
    v = _shape_verts(shape)
    if v is None or len(v) < 4:
        return []
    T = _pose_T(shape.get_local_pose())
    try:
        return [_Piece(v @ T[:3, :3].T + T[:3, 3])]
    except Exception:
        return []


_KDOP_DIRS = None


def _kdop(verts: np.ndarray) -> np.ndarray:
    """Vertices of the polytope bounded by KDOP_N planes (Fibonacci-sphere
    normals) touching the point set: it CONTAINS the set's hull (a lower-bound
    stand-in for it) with ~2 x KDOP_N faces — a T1 container's exact hull had
    2964 faces (one hull-hull face test ~20 ms). Bulge over the true hull:
    a few % of the object's size (only used beyond NEAR)."""
    global _KDOP_DIRS
    from scipy.spatial import HalfspaceIntersection

    if _KDOP_DIRS is None:
        n = KDOP_N
        i = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        theta = np.pi * (1.0 + 5.0**0.5) * i
        _KDOP_DIRS = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    d = (verts @ _KDOP_DIRS.T).max(axis=0)
    inner = verts.mean(axis=0)
    try:
        hs = HalfspaceIntersection(np.concatenate([_KDOP_DIRS, -d[:, None]], axis=1), inner)
        return np.asarray(hs.intersections, dtype=np.float64)
    except Exception:  # degenerate (flat) set: its own hull
        return verts


class ItemShape:
    """An entity's collision geometry: polytope pieces + a polytope containing
    all of them (the pieces' hull, or a k-DOP of it), in the entity frame."""

    def __init__(self, entity, pieces: list[_Piece]):
        self.entity = entity
        self.pieces = pieces
        self.hull = pieces[0] if len(pieces) == 1 else _Piece(_kdop(np.concatenate([q.verts for q in pieces], axis=0)))
        self.centers = np.stack([q.center for q in pieces])  # [P,3] piece bounding spheres
        self.radii = np.array([q.radius for q in pieces])
        self._key = None
        self._T = np.eye(4)
        self._pieces_posed = False

    def T(self) -> np.ndarray:
        """Entity transform now; poses the hull's FCL object (pieces lazily)."""
        T = _pose_T(self.entity.get_pose())
        key = T[:3, :].tobytes()
        if key != self._key:
            self._key, self._T, self._pieces_posed = key, T, False
            self.hull.obj.set_pose(_mpose(T))
        return self._T

    def posed_pieces(self) -> list[_Piece]:
        T = self.T()
        if not self._pieces_posed:
            if len(self.pieces) > 1:
                pose = _mpose(T)
                for q in self.pieces:
                    q.obj.set_pose(pose)
            self._pieces_posed = True
        return self.pieces


def _collision_shapes(entity) -> list:
    import sapien.physx as px

    comp = entity.find_component_by_type(px.PhysxRigidBaseComponent)
    return list(comp.get_collision_shapes()) if comp is not None else []


_SHAPE_CACHE: dict = {}  # id(obj) -> (weakref to obj, ItemShape | None); the ref guards id reuse


def shape_of(obj, fallback_obb=None) -> ItemShape | None:
    """Cached ItemShape of an ItemSpec / actor wrapper / entity (keyed on the
    python object, which is PINNED all episode for item specs and task
    actors; a dead weakref means a new episode reused the id).
    No collision shapes: the OBB (center_local, half) as a box piece when
    given, else None."""
    from .items import entity_of

    hit = _SHAPE_CACHE.get(id(obj))
    if hit is not None and hit[0]() is obj:
        return hit[1]
    entity = entity_of(obj)
    pieces: list[_Piece] = []
    try:
        for s in _collision_shapes(entity):
            pieces += _pieces_from_shape(s)
    except Exception:
        pieces = []
    if not pieces and fallback_obb is not None:
        c, h = (np.asarray(x, dtype=np.float64) for x in fallback_obb)
        pieces = [_Piece(_CUBE * h + c)]
    shape = ItemShape(entity, pieces) if pieces else None
    if len(_SHAPE_CACHE) > 512:  # drop entries of finished episodes
        for k in [k for k, (ref, _s) in _SHAPE_CACHE.items() if ref() is None]:
            del _SHAPE_CACHE[k]
    try:
        _SHAPE_CACHE[id(obj)] = (weakref.ref(obj), shape)
    except TypeError:
        pass  # not weak-referenceable: recomputed next call
    return shape


# ------------------------------------------------------------- primitives ----

_REQ: dict = {}
_POINT = None


def _gjk(o1, o2, solver: str = "GST_INDEP") -> float:
    """FCL GJK distance (unsigned: <= 0 means the two overlap)."""
    fcl = _fcl()
    req = _REQ.get(solver)
    if req is None:
        req = _REQ[solver] = fcl.DistanceRequest(
            enable_signed_distance=False, gjk_solver_type=getattr(fcl.GJKSolverType, solver)
        )
    return float(fcl.distance(o1, o2, req).min_distance)


def _gjk_gap(o1, o2) -> float | None:
    """Gap between two convex objects: the SMALLER of FCL's two GJK solvers'
    positive answers (None if both read overlap). A GJK gap is the distance
    between two points it found on the shapes, so it can only be too large
    when the search stops early — which each solver does on its own cases
    (point -> 56-vertex hull, 1176 queries: GST_INDEP 5x up to +1.1 mm,
    libccd 1x +0.05 mm; libccd also +3.5 mm for a point exactly above a box
    face's centre)."""
    vals = [v for v in (_gjk(o1, o2, "GST_INDEP"), _gjk(o1, o2, "GST_LIBCCD")) if v > 0.0]
    return min(vals) if vals else None


def _at(geom, c):
    T = np.eye(4)
    T[:3, 3] = c
    return _fcl().CollisionObject(geom, _mpose(T))


def _point_distance(c_world, piece: _Piece, lower: float) -> float:
    """Distance point -> polytope for a point known to be OUTSIDE it (lower:
    the plane bound, > 0). A point is a 1e-7 m sphere."""
    global _POINT
    if _POINT is None:
        _POINT = _fcl().Sphere(_POINT_R)  # module-level: kept alive
    d = _gjk_gap(_at(_POINT, c_world), piece.obj)
    return lower if d is None else max(d + _POINT_R, lower)


def _planemax(local: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """[N] max_f (n_f . x - d_f): signed distance of each point to the
    polytope when inside (<= 0), a lower bound when outside."""
    return (local @ planes[:, :3].T - planes[:, 3]).max(axis=1)


def _sphere_piece(c_world, c_local, r: float, piece: _Piece) -> float:
    """Exact signed distance sphere -> polytope (c_local: entity frame):
    center inside -> the depth below the nearest face; outside -> the point
    distance - r (GJK on the sphere itself read +0.6 mm for a 1.3 mm
    overlap: libccd's tolerance; the point query was exact)."""
    s = float(_planemax(c_local[None], piece.planes)[0])
    if s <= 0.0:
        return s - r
    return _point_distance(c_world, piece, s) - r


SAT_CHECK = 0.005  # m: a GJK gap below this is confirmed by FCL's collision test
_CREQ = None


def _collide(o1, o2) -> bool:
    """FCL boolean collision test (302/302 right on random pairs within 4 mm
    of contact, vs an optimizer)."""
    global _CREQ
    fcl = _fcl()
    if _CREQ is None:
        _CREQ = fcl.CollisionRequest()
    return bool(fcl.collide(o1, o2, _CREQ).is_collision())


def _face_gap(pa: _Piece, Ta: np.ndarray, pb: _Piece, Tb: np.ndarray) -> float:
    """Max over both pieces' FACE normals of the gap along it: > 0 proves
    they are apart (and is <= the distance); when they overlap it is
    -(the depth along the best face axis), at least the true depth (the
    edge-edge axes of a full separating-axis test are not tried: too slow,
    ~7 ms for two 24-vertex pieces)."""
    R = Ta[:3, :3].T @ Tb[:3, :3]
    t = Ta[:3, :3].T @ (Tb[:3, 3] - Ta[:3, 3])
    vb_in_a = pb.verts @ R.T + t
    va_in_b = (pa.verts - t) @ R
    ga = (vb_in_a @ pa.planes[:, :3].T - pa.planes[:, 3]).min(axis=0).max()
    gb = (va_in_b @ pb.planes[:, :3].T - pb.planes[:, 3]).min(axis=0).max()
    return float(max(ga, gb))


def _piece_piece(pa: _Piece, Ta: np.ndarray, pb: _Piece, Tb: np.ndarray) -> float:
    """Signed distance polytope -> polytope: the GJK gap when clearly apart;
    within SAT_CHECK FCL's collision test decides (GJK alone read up to
    2.6 mm of gap for 1-3 of 233 overlapping random pairs); overlapping,
    the face-axis depth (conservative)."""
    d = _gjk_gap(pa.obj, pb.obj)
    if d is not None and d > SAT_CHECK:
        return d
    g = _face_gap(pa, Ta, pb, Tb)
    if g > 0.0:  # a separating face plane: certainly apart, by at least g
        return g if d is None else max(d, g)
    if _collide(pa.obj, pb.obj):
        return min(g, 0.0)
    return 0.0 if d is None else d


def _branch_and_bound(bounds: np.ndarray, exact) -> float:
    """min over pairs of exact(k), visiting pairs by increasing lower bound;
    stops once the best is <= -DEPTH_ENOUGH (h is floored there anyway)."""
    if bounds.size == 0:
        return float("inf")
    order = np.argsort(bounds, kind="stable")
    best = float("inf")
    for n, k in enumerate(order):
        b = float(bounds[k])
        if b >= best or best <= -DEPTH_ENOUGH:
            break
        if n >= MAX_EXACT:
            return min(best, b)
        best = min(best, float(exact(int(k))))
    return best


# -------------------------------------------------------------- distances ----


def _local(centers: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (centers - T[:3, 3]) @ T[:3, :3]


def sphere_distances(centers: np.ndarray, radii: np.ndarray, shape: ItemShape) -> np.ndarray:
    """[N] signed distance of each sphere to the shape (m): the hull's exact
    distance beyond NEAR, the pieces' exact distance within it."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    out = np.full(len(radii), np.inf)
    if len(radii) == 0 or not shape.pieces:
        return out
    local = _local(centers, shape.T())
    for i in range(len(radii)):
        out[i] = _sphere_piece(centers[i], local[i], radii[i], shape.hull)
        if len(shape.pieces) == 1 or out[i] > NEAR:
            continue
        pieces = shape.posed_pieces()
        bounds = np.array([_planemax(local[i : i + 1], q.planes)[0] for q in pieces]) - radii[i]
        out[i] = _branch_and_bound(
            np.maximum(bounds, out[i]), lambda k: _sphere_piece(centers[i], local[i], radii[i], pieces[k])
        )
    return out


def spheres_min_distance(centers: np.ndarray, radii: np.ndarray, shape: ItemShape) -> tuple[float, int]:
    """(min over the spheres of ``sphere_distances``, argmin index) without
    evaluating every sphere: exact whenever the result is <= NEAR or the
    shape is one convex piece."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(radii) == 0 or not shape.pieces:
        return float("inf"), -1
    local = _local(centers, shape.T())
    hb = _planemax(local, shape.hull.planes) - radii  # lower bounds vs the hull
    hit = {"best": float("inf"), "s": -1}

    def hull_exact(s: int) -> float:
        d = _sphere_piece(centers[s], local[s], radii[s], shape.hull)
        if d < hit["best"]:
            hit["best"], hit["s"] = d, s
        return d

    d_hull = _branch_and_bound(hb, hull_exact)  # level 1: the hull
    arg = hit["s"] if hit["s"] >= 0 else int(np.argmin(hb))
    if len(shape.pieces) == 1 or d_hull > NEAR:
        return d_hull, arg
    near = np.flatnonzero(hb <= NEAR)  # level 2: the pieces, for spheres near the hull
    far = hb[hb > NEAR]
    pieces = shape.posed_pieces()
    bounds = np.stack([_planemax(local[near], q.planes) for q in pieces], axis=1) - radii[near][:, None]
    n_p = len(pieces)
    hit2 = {"best": float("inf"), "s": -1}

    def piece_exact(k: int) -> float:
        row, j = divmod(k, n_p)
        s = int(near[row])
        d = _sphere_piece(centers[s], local[s], radii[s], pieces[j])
        if d < hit2["best"]:
            hit2["best"], hit2["s"] = d, s
        return d

    best = _branch_and_bound(np.maximum(bounds, d_hull).reshape(-1), piece_exact)
    if far.size and float(far.min()) < best:  # only when the pieces are > NEAR away (a concave hull)
        return float(far.min()), int(np.flatnonzero(hb > NEAR)[np.argmin(far)])
    return best, (hit2["s"] if hit2["s"] >= 0 else int(near[np.argmin(bounds.min(axis=1))]))


def shape_distance(a: ItemShape, b: ItemShape) -> float:
    """Signed distance between two shapes (m): the hulls' distance when it
    exceeds NEAR (or both are one convex piece), else the min over piece
    pairs near the other shape (branch and bound)."""
    if not a.pieces or not b.pieces:
        return float("inf")
    Ta, Tb = a.T(), b.T()
    d_hull = _piece_piece(a.hull, Ta, b.hull, Tb)
    if d_hull > NEAR or (len(a.pieces) == 1 and len(b.pieces) == 1):
        return d_hull
    # each piece's bounding sphere vs the OTHER shape's hull: a lower bound
    ca_w = a.centers @ Ta[:3, :3].T + Ta[:3, 3]
    cb_w = b.centers @ Tb[:3, :3].T + Tb[:3, 3]
    ba = np.maximum(_planemax(_local(ca_w, Tb), b.hull.planes) - a.radii, d_hull)
    bb = np.maximum(_planemax(_local(cb_w, Ta), a.hull.planes) - b.radii, d_hull)
    ia, ib = np.flatnonzero(ba <= NEAR), np.flatnonzero(bb <= NEAR)
    left_out = np.concatenate([ba[ba > NEAR], bb[bb > NEAR]])
    rest = float(left_out.min()) if left_out.size else float("inf")  # lower bound of every pair left out
    if ia.size == 0 or ib.size == 0:
        return rest  # every pair involves a piece > NEAR from the other shape
    pa, pb = a.posed_pieces(), b.posed_pieces()
    gap = np.linalg.norm(ca_w[ia][:, None, :] - cb_w[ib][None, :, :], axis=2) - a.radii[ia][:, None] - b.radii[ib][None, :]
    bounds = np.maximum(gap, np.maximum(ba[ia][:, None], bb[ib][None, :]))
    n_b = ib.size

    def exact(k: int) -> float:
        i, j = divmod(k, n_b)
        return _piece_piece(pa[int(ia[i])], Ta, pb[int(ib[j])], Tb)

    return min(_branch_and_bound(bounds.reshape(-1), exact), rest)
