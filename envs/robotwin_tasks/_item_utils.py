"""RoboTwin-side helpers for item-safety tasks (plan v3 §2.2/§4.1/§4.5).

Lives INSIDE the symlinked ``RoboTwin/envs/`` tree, so it may import
RoboTwin modules — but it must NEVER import CEHJ ``main.*``: task files
run inside RoboTwin's process (which has its own vendored ``envs/``
package and cannot reach the CEHJ repo), and the CEHJ runner re-reads
all metadata back from plain attribute dicts. The metadata dicts this
module writes onto the task instance are the ONLY interface between a
RoboTwin task and ``main.item_envs``:

- ``task.item_meta``   — list of item dicts, exactly the v3 §4.1
  ``items`` schema (``item_id``, ``entity``, ``category``, ``color``,
  ``kind``, ``role``, ``group``, ``particles``, ``half_extents``,
  ``obb_center_local``, ``up_axis_local``, ``mass``). ``entity`` is the
  live sapien Actor/Articulation; the runner derives ``pose_fn`` /
  ``obb`` / ``vel_fn`` from it.
- ``task.region_meta`` — list of region dicts (``region_id``,
  ``entity``, ``kind``, ``color``, plus zone extras ``size_xy`` /
  ``z`` / ``z_band`` / ``nameable``). The runner builds Region objects
  from these (``main.item_envs.regions.zone_region`` reads the
  ``size_xy`` attribute when present and falls back to the actor's
  collision-shape AABB otherwise).

Time-scale convention (T1): the runner sets ``task._time_scale_safe``
and ``task._time_scale_risky`` once after setup. ``risky(task)`` below
flips ``task.active_time_scale`` to the risky value around a motion
primitive and restores the previous value on exit (nesting-safe).
``RiskTickController.plan()`` only READS ``task.active_time_scale``
for its retime factor — the controller's separate
``risky_skill_labels`` path is the ONLY other writer and must not be
combined with this context manager.

Optional live mass override (``register_item(..., mass=...)``) writes
through PhysX immediately, so ``physics.apply_physics`` /
``physics.read_back`` stay the single source of truth for sampled
physics.
"""

from contextlib import contextmanager


def _unwrap(entity):
    """Accept a raw sapien Entity or a RoboTwin Actor wrapper."""
    return getattr(entity, "actor", entity)


def zone_mat(scene, pose, size_xy, color, thickness=0.002, name=None):
    """Spawn a thin static colored box that marks a zone on the table.

    ``pose`` is the CENTER pose of the mat (sapien convention);
    ``size_xy`` is the full (x, y) extent. Built with the same
    Entity + PhysxRigidStaticComponent idiom as RoboTwin's own
    ``create_entity_box`` (no ActorBuilder). A render body is attached
    ONLY when the scene actually has a render system, queried via
    ``scene.get_system("render")`` (sapien raises RuntimeError when no
    such system is registered — the same name-based lookup
    ``add_entity`` performs, so a false query means ``add_entity``
    would reject the entity). Physics-only scenes get a
    collision-only mat.

    The returned entity is a raw pybind11 ``sapien.Entity`` and carries
    NO Python attributes (pybind11 rejects them). ``size_xy`` reaches
    the CEHJ side through ``register_zone``'s meta dict instead;
    ``regions.zone_region`` derives the same footprint from the mat's
    collision-shape AABB when no attribute is present.
    """
    import sapien

    half = [float(size_xy[0]) / 2.0, float(size_xy[1]) / 2.0, float(thickness) / 2.0]
    entity = sapien.Entity()
    entity.set_name(name or f"zone_{int(color[0]*255)}_{int(color[1]*255)}_{int(color[2]*255)}")
    material = getattr(scene, "default_physical_material", None)
    if material is None:
        material = scene.create_physical_material(0.5, 0.5, 0.0)
    rigid = sapien.physx.PhysxRigidStaticComponent()
    rigid.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half, material=material))
    entity.add_component(rigid)
    try:  # query, not construction: get_system raises when absent
        scene.get_system("render")
        has_render = True
    except RuntimeError:
        has_render = False
    if has_render:
        visual = sapien.render.RenderMaterial(base_color=[float(color[0]), float(color[1]), float(color[2]), 1.0])
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(half, visual))
        entity.add_component(render)
    entity.set_pose(pose)
    scene.add_entity(entity)
    return entity


@contextmanager
def risky(task, risk=None):
    """Mark the enclosed motion primitive as risk-bearing (plan v3 §2.5 T1).

    Sets ``task.active_time_scale`` to the risky time scale for the
    duration of the block and restores the previous value on exit
    (nesting-safe). The scale comes from ``risk`` when given (a
    RiskParams-like object with ``.time_scale``, or a plain float),
    else from ``task._time_scale_risky`` which the runner installs
    after controller attach; a missing attribute falls back to 1.0 so
    the task also runs under vanilla RoboTwin collection.
    """
    if risk is not None:
        scale = float(getattr(risk, "time_scale", risk))
    else:
        scale = float(getattr(task, "_time_scale_risky", 1.0))
    prev = getattr(task, "active_time_scale", getattr(task, "_time_scale_safe", 1.0))
    task.active_time_scale = scale
    try:
        yield
    finally:
        task.active_time_scale = prev


def register_item(task, item_id, entity, category, color, kind, role, group=None,
                  particles=None, half_extents=None, mass=None,
                  obb_center_local=None, up_axis_local=None):
    """Append a v3 §4.1 item-metadata dict to ``task.item_meta``.

    ``mass`` (optional) is applied to the live PhysX body immediately —
    the same write ``physics.apply_physics`` performs — so mass never
    diverges from what the runner later reads back.
    """
    if not hasattr(task, "item_meta"):
        task.item_meta = []
    meta = {
        "item_id": str(item_id),
        "entity": entity,
        "category": str(category),
        "color": str(color),
        "kind": str(kind),
        "role": str(role),
        "group": group,
        "particles": particles,
        "half_extents": None if half_extents is None else [float(v) for v in half_extents],
        "obb_center_local": None if obb_center_local is None else [float(v) for v in obb_center_local],
        "up_axis_local": None if up_axis_local is None else [float(v) for v in up_axis_local],
        "mass": None if mass is None else float(mass),
    }
    task.item_meta.append(meta)
    if mass is not None:
        _apply_mass(entity, float(mass))
    return meta


def register_zone(task, region_id, entity, color, size_xy=None, kind="zone",
                  z=0.0, z_band=0.01, nameable=True, **extra):
    """Append a region-metadata dict to ``task.region_meta``.

    The runner turns each entry into a Region. Raw sapien entities
    cannot carry Python attributes, so this meta dict is the channel
    for ``size_xy``; ``regions.zone_region`` otherwise derives the mat
    footprint from its collision-shape AABB.
    """
    if not hasattr(task, "region_meta"):
        task.region_meta = []
    meta = {
        "region_id": str(region_id),
        "entity": entity,
        "kind": str(kind),
        "color": str(color),
        "size_xy": None if size_xy is None else [float(v) for v in size_xy],
        "z": float(z),
        "z_band": float(z_band),
        "nameable": bool(nameable),
    }
    meta.update(extra)
    task.region_meta.append(meta)
    return meta


def _apply_mass(entity, mass):
    """Set the mass of every dynamic rigid body under ``entity``."""
    import sapien.physx as physx

    raw = _unwrap(entity)
    comp = raw.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        return
    comp.set_mass(float(mass))
