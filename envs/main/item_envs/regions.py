"""Region: the region-side data contract (PLAN v3 Section 5) and constructors.

The dataclass is the contract; table_region / zone_region / container_region
(PLAN v3 §2.3) build the standard regions. Constructors are duck-typed
(anything with ``get_pose()``); no module-level sapien import. Polygons are
CCW in the region frame xy plane, SI units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import circle_polygon
from .items import entity_of


@dataclass(kw_only=True)
class Region:
    """A named 2D region on a (possibly moving) frame, evaluated every tick."""

    region_id: str  # "table_top", "zone_red", "bowl_src_interior", "target_interior"
    kind: str  # "surface" | "zone" | "container"
    frame_fn: Callable[[], np.ndarray]  # 4x4 world_T_region, evaluated every tick (bowls move)
    polygon: np.ndarray  # [M, 2] CCW polygon in the region frame xy plane (m)
    z: float  # surface height in the region frame
    z_band: float  # vertical tolerance for "on surface" checks (m)
    nameable: bool = False  # True if a rule may name it (zones, containers)
    color: str | None = None
    entity: object | None = None  # the mat or container entity, for masks
    # particle rules only: a particle above region z = z_cap is NOT in the
    # region (a container interior capped just above its rim: a bead flying
    # over a dish does not count as in it, one that drops in does — T6)
    z_cap: float | None = None
    # particle rules only: a particle below region z = z_floor is NOT in the region either (2026-10-23: T6's bowl
    # column, footprint radius = the outer wall + 2 mm, would otherwise count a bead lying on the table against the
    # bowl's outside wall)
    z_floor: float | None = None
    role: str | None = None  # rule selector tag (T6 "keep_clean" neighbours); None for the classic regions

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        self.z = float(self.z)
        self.z_band = float(self.z_band)
        self.nameable = bool(self.nameable)

    def world_T(self) -> np.ndarray:
        """Current 4x4 world_T_region."""
        return np.asarray(self.frame_fn(), dtype=np.float64)


# ---------------------------------------------------------------------------
# Constructors (PLAN v3 §2.3)


def _pose_frame_fn(obj) -> Callable[[], np.ndarray]:
    """frame_fn following obj's world pose every tick (duck-typed get_pose())."""
    ent = entity_of(obj)

    def frame_fn() -> np.ndarray:
        return np.asarray(ent.get_pose().to_transformation_matrix(), dtype=np.float64)

    return frame_fn


def _rect_polygon(hx: float, hy: float) -> np.ndarray:
    """CCW axis-aligned rectangle [-hx,hx] x [-hy,hy] in the xy plane."""
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)


def _up_frame(up_local: np.ndarray, rim_height: float) -> np.ndarray:
    """4x4 actor_T_region whose +z is the actor-local up axis.

    The region xy plane is perpendicular to ``up_local``; the frame origin sits
    ``rim_height`` along the up axis from the actor origin, so the rim plane is
    region z = 0. The x/y basis is an arbitrary orthonormal completion.
    """
    w = np.asarray(up_local, dtype=np.float64).reshape(3)
    n = np.linalg.norm(w)
    if n < 1e-12:
        raise ValueError("zero up axis")
    w = w / n
    a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(a, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    mat = np.eye(4)
    mat[:3, 0], mat[:3, 1], mat[:3, 2] = u, v, w
    mat[:3, 3] = w * float(rim_height)
    return mat


def _mat_size_xy(mat_actor) -> tuple[float, float]:
    """Full (sx, sy) xy size of a zone mat, in the mat frame.

    Reads a ``size_xy`` attribute first (Python wrappers and mocks can carry
    one; pybind11 entities reject attributes, so task-side code registers
    sizes in ``task.region_meta`` instead); else falls back to the union AABB
    of box collision shapes in the actor frame.
    """
    for obj in (mat_actor, entity_of(mat_actor)):
        size_xy = getattr(obj, "size_xy", None)
        if size_xy is not None:
            arr = np.asarray(size_xy, dtype=np.float64).reshape(-1)
            return float(arr[0]), float(arr[1])
    import sapien.physx as physx  # lazy: no module-level sapien import

    ent = entity_of(mat_actor)
    comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        comp = ent.find_component_by_type(physx.PhysxRigidStaticComponent)
    if comp is None:
        raise TypeError(
            f"no rigid component on {ent!r}; the mat footprint needs a "
            "size_xy attribute on a Python wrapper (pybind11 entities reject "
            "attributes — register sizes in task.region_meta) or box "
            "collision shapes for the AABB fallback"
        )
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    n = 0
    for shape in comp.get_collision_shapes():
        half = getattr(shape, "half_size", None)
        if half is None:
            continue  # only box shapes carry an axis-aligned half_size
        center = np.asarray(shape.local_pose.p, dtype=np.float64).reshape(3)
        h = np.abs(np.asarray(half, dtype=np.float64).reshape(3))
        lo = np.minimum(lo, center - h)
        hi = np.maximum(hi, center + h)
        n += 1
    if n == 0:
        raise ValueError(
            f"cannot derive the mat footprint of {ent!r}: no box collision "
            "shapes for the AABB fallback; pybind11 entities reject Python "
            "attributes, so supply size_xy through the task.region_meta "
            "channel or give the mat a box collision shape"
        )
    size = hi - lo
    return float(size[0]), float(size[1])


def table_region(task) -> Region:
    """The RoboTwin tabletop: 1.2 m x 0.7 m, top at 0.74 + task.table_z_bias.

    Centered at task.table_xy_bias (recon.md §3.1 / create_table_and_wall).
    kind="surface" so ``inside`` margins may require resting on it; nameable
    (T3 names it), region_id "table_top", z_band 1 cm.
    """
    xy = np.asarray(getattr(task, "table_xy_bias", [0.0, 0.0]), dtype=np.float64).reshape(2)
    z_top = 0.74 + float(getattr(task, "table_z_bias", 0.0))
    frame = np.eye(4)
    frame[:3, 3] = [xy[0], xy[1], 0.0]
    return Region(
        region_id="table_top",
        kind="surface",
        frame_fn=lambda: frame.copy(),
        polygon=_rect_polygon(0.6, 0.35),
        z=z_top,
        z_band=0.01,
        nameable=True,
        color=None,
        entity=getattr(task, "table", None),
    )


def zone_region(mat_actor, color: str | None, region_id: str | None = None, size_xy=None) -> Region:
    """A zone mat's footprint as a nameable region, following the mat.

    The polygon is the mat's xy footprint in the mat frame (from its
    ``size_xy`` attribute or its box collision shapes); the frame follows the
    mat's pose every tick. kind="zone" — margins use the xy polygon only.
    """
    # registered size first (task.region_meta): collision-free (painted) mats
    # have no box shape for the AABB fallback
    if size_xy is not None:
        sx, sy = (float(v) for v in np.asarray(size_xy, dtype=np.float64).reshape(-1)[:2])
    else:
        sx, sy = _mat_size_xy(mat_actor)
    if region_id is None:
        region_id = f"zone_{color}" if color else "zone"
    return Region(
        region_id=str(region_id),
        kind="zone",
        frame_fn=_pose_frame_fn(mat_actor),
        polygon=_rect_polygon(0.5 * sx, 0.5 * sy),
        z=0.0,
        z_band=0.01,
        nameable=True,
        color=color,
        entity=entity_of(mat_actor),
    )


def container_region(actor, rim_radius: float, region_id: str, up_axis_local=None, rim_height: float = 0.0,
                     z_cap: float | None = None, color: str | None = None, polygon=None,
                     role: str | None = None, z_floor: float | None = None) -> Region:
    """A container interior as a nameable region, following the actor every tick.

    The polygon is a 16-gon at the inner rim radius in the actor's up-axis
    frame (the frame whose +z is ``up_axis_local``). ``up_axis_local`` defaults
    to the actor's ``up_axis_local`` attribute, then to [0, 1, 0] — the
    RoboTwin glb convention (Y-up file frame; recon.md Q13 / v3 surprises).
    ``rim_height`` shifts the region origin along the up axis so the rim plane
    is region z = 0. ``z_cap`` (region frame, m above the rim plane) caps the
    column for particle rules (Region.z_cap), ``z_floor`` floors it (Region.z_floor). ``polygon`` (region frame xy,
    CCW) replaces the default circle — a square bin's footprint (T6).
    """
    ent = entity_of(actor)
    up = up_axis_local
    if up is None:
        up = getattr(actor, "up_axis_local", None)
    if up is None:
        up = getattr(ent, "up_axis_local", None)
    if up is None:
        up = [0.0, 1.0, 0.0]
    actor_T_region = _up_frame(up, rim_height)

    def frame_fn() -> np.ndarray:
        world_T_actor = np.asarray(ent.get_pose().to_transformation_matrix(), dtype=np.float64)
        return world_T_actor @ actor_T_region

    return Region(
        region_id=str(region_id),
        kind="container",
        frame_fn=frame_fn,
        polygon=circle_polygon(float(rim_radius), 16) if polygon is None else np.asarray(polygon, dtype=np.float64),
        z=0.0,
        z_band=0.01,
        nameable=True,
        color=color,
        role=role,
        entity=ent,
        z_cap=None if z_cap is None else float(z_cap),
        z_floor=None if z_floor is None else float(z_floor),
    )
