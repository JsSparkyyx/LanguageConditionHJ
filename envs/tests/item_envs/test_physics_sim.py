"""Sim tests for main.item_envs.physics (PLAN v3 §2.1): apply then read_back.

Headless physics via sapien.Scene([physx.PhysxCpuSystem()]) — the only scene
that works without Vulkan (recon.md). Values come back with float32 noise, so
comparisons use tolerances. v3: CoM offsets apply in x and y only (z = 0),
masses scale around the configs/objects.yaml category nominal, no damping
randomization.
"""

from __future__ import annotations

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")
import sapien.physx as physx

from main.item_envs import physics as phys
from main.item_envs.items import ItemSpec

pytestmark = pytest.mark.sim

DT = 1.0 / 250.0
ATOL = 2e-6  # float32 readback noise (recon.md Q1 readback shows ~1e-7 relative)


def make_scene():
    scene = sapien.Scene([physx.PhysxCpuSystem()])
    scene.set_timestep(DT)
    return scene


def add_box(scene, half=0.03, xyz=(0, 0, 0.1), name="box", static=False):
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[half, half, half])
    if static:
        b.set_physx_body_type("static")
    b.set_initial_pose(sapien.Pose(list(xyz)))
    return b.build(name=name)


def add_sphere(scene, radius=0.02, xyz=(0.3, 0, 0.1), name="sphere"):
    b = scene.create_actor_builder()
    b.add_sphere_collision(radius=radius)
    b.set_initial_pose(sapien.Pose(list(xyz)))
    return b.build(name=name)


def add_two_part_box(scene, xyz=(0, 0, 0.1), name="two_part"):
    b = scene.create_actor_builder()
    b.add_box_collision(pose=sapien.Pose([0, 0, 0.02]), half_size=[0.03, 0.03, 0.01])
    b.add_box_collision(pose=sapien.Pose([0, 0, -0.02]), half_size=[0.02, 0.02, 0.01])
    b.set_initial_pose(sapien.Pose(list(xyz)))
    return b.build(name=name)


class _Wrapper:
    """Duck-typed RoboTwin-Actor-style wrapper (entity_of unwraps via .actor)."""

    def __init__(self, entity):
        self.actor = entity


class _MockTask:
    """Duck-typed task for apply_physics (needs .scene and .items only)."""

    def __init__(self, scene, items):
        self.scene = scene
        self.items = items


# ---------------------------------------------------------------- round-trips


@pytest.mark.parametrize("shape", ["box", "sphere"])
def test_material_roundtrip(shape):
    scene = make_scene()
    ent = add_box(scene) if shape == "box" else add_sphere(scene)
    phys.apply_material(ent, 0.77, 0.55, 0.08, scene)
    rb = phys.read_back(ent)
    assert rb["body"] == "dynamic"
    assert rb["static_friction"] == pytest.approx(0.77, abs=ATOL)
    assert rb["dynamic_friction"] == pytest.approx(0.55, abs=ATOL)
    assert rb["restitution"] == pytest.approx(0.08, abs=ATOL)


@pytest.mark.parametrize("shape", ["box", "sphere"])
def test_mass_com_roundtrip(shape):
    scene = make_scene()
    ent = add_box(scene) if shape == "box" else add_sphere(scene)
    com = np.array([0.005, -0.004, 0.0])  # v3: CoM offsets live in x and y only
    phys.set_mass_props(ent, 0.42, com)
    rb = phys.read_back(ent)
    assert rb["mass"] == pytest.approx(0.42, abs=1e-6)
    assert np.allclose(rb["com_offset"], com, atol=ATOL)
    # mass change must not clobber the CoM pose, and vice versa (recon.md Q3)
    phys.set_mass_props(ent, 0.5)
    rb = phys.read_back(ent)
    assert rb["mass"] == pytest.approx(0.5, abs=1e-6)
    assert np.allclose(rb["com_offset"], com, atol=ATOL)


def test_wrapper_and_multishape_roundtrip():
    """Materials hit EVERY collision shape; wrappers unwrap through entity_of."""
    scene = make_scene()
    ent = add_two_part_box(scene)
    wrapped = _Wrapper(ent)
    phys.apply_material(wrapped, 0.66, 0.44, 0.02, scene)
    phys.set_mass_props(wrapped, 0.31, [0.002, 0.001, 0.0])
    rb = phys.read_back(wrapped)  # read_back on the wrapper itself
    assert rb["n_shapes"] == 2
    assert rb["static_friction"] == pytest.approx(0.66, abs=ATOL)
    assert rb["dynamic_friction"] == pytest.approx(0.44, abs=ATOL)
    assert rb["mass"] == pytest.approx(0.31, abs=1e-6)
    assert np.allclose(rb["com_offset"], [0.002, 0.001, 0.0], atol=ATOL)
    # both shapes carry the new material, not just the first
    comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
    for shape in comp.get_collision_shapes():
        assert shape.physical_material.static_friction == pytest.approx(0.66, abs=ATOL)


def test_read_back_static_body():
    scene = make_scene()
    mat = add_box(scene, xyz=(0, 0, 0.001), name="mat", static=True)
    phys.apply_material(mat, 0.9, 0.8, 0.0, scene)
    rb = phys.read_back(mat)
    assert rb["body"] == "static"
    assert rb["mass"] is None
    assert rb["static_friction"] == pytest.approx(0.9, abs=ATOL)
    assert rb["dynamic_friction"] == pytest.approx(0.8, abs=ATOL)
    with pytest.raises(TypeError):
        phys.set_mass_props(mat, 1.0)  # no dynamic component on a static body


def test_no_rigid_component_raises():
    scene = make_scene()
    entity = sapien.Entity()
    scene.add_entity(entity)
    with pytest.raises(TypeError):
        phys.read_back(entity)
    with pytest.raises(TypeError):
        phys.apply_material(entity, 0.5, 0.5, 0.0, scene)


# ---------------------------------------------------------------- sampling


def _spec(item_id, category, half, kind="rigid", mass=0.0):
    return ItemSpec(
        item_id=item_id,
        entity=None,
        category=category,
        kind=kind,
        half_extents=np.asarray(half, dtype=np.float64),
        mass=mass,
    )


def test_sample_physics_determinism_and_ranges():
    specs = [
        _spec("bottle_0", "bottle", [0.035, 0.035, 0.10]),
        _spec("cyl_0", "cylinder", [0.03, 0.03, 0.06]),
        _spec("beads_0", "beads", [0.006, 0.006, 0.006], kind="particle_set"),
    ]
    cfg = phys.load_physics_cfg()
    a = phys.sample_physics(specs, cfg, np.random.RandomState(123))
    b = phys.sample_physics(specs, cfg, np.random.RandomState(123))
    assert set(a) == {"bottle_0", "cyl_0", "beads_0"}
    for item_id in a:
        for key, val in a[item_id].items():
            other = b[item_id][key]
            if isinstance(val, np.ndarray):
                assert np.allclose(val, other)
            else:
                assert val == other
    bottle = a["bottle_0"]
    nominal = phys.category_nominal_masses(phys.load_objects_cfg())["bottle"]
    lo, hi = cfg["mass_scale"]
    assert lo * nominal <= bottle["mass"] <= hi * nominal
    assert 0.3 <= bottle["static_friction"] <= 0.9
    assert 0.2 <= bottle["dynamic_friction"] <= bottle["static_friction"]
    assert 0.0 <= bottle["restitution"] <= 0.1
    half = np.array([0.035, 0.035, 0.10])
    assert np.all(bottle["com_offset"][:2] >= -0.2 * half[:2])
    assert np.all(bottle["com_offset"][:2] <= 0.2 * half[:2])
    # v3: CoM offsets apply in x and y only
    assert bottle["com_offset"][2] == 0.0
    # particle sets skip the CoM offset
    assert a["beads_0"]["com_offset"] is None


def test_sample_physics_mass_base_fallback_and_unknown():
    # unknown category with explicit spec.mass -> scale around spec.mass
    spec = _spec("gadget_0", "gadget", [0.02, 0.02, 0.02], mass=0.5)
    out = phys.sample_physics([spec], phys.load_physics_cfg(), np.random.RandomState(7))
    assert 0.7 * 0.5 <= out["gadget_0"]["mass"] <= 1.3 * 0.5
    # unknown category without a mass -> KeyError
    with pytest.raises(KeyError):
        phys.sample_physics([_spec("gadget_1", "gadget", [0.02] * 3)], phys.load_physics_cfg(), np.random.RandomState(7))


# ---------------------------------------------------------------- apply


def test_apply_physics_then_read_back():
    scene = make_scene()
    box = add_box(scene, xyz=(0, 0, 0.05))
    cyl = add_box(scene, half=0.02, xyz=(0.3, 0, 0.05), name="cyl")
    items = {
        "bottle_0": _spec("bottle_0", "bottle", [0.03, 0.03, 0.03]),
        "cyl_0": _spec("cyl_0", "cylinder", [0.02, 0.02, 0.02]),
    }
    items["bottle_0"].entity = box
    items["cyl_0"].entity = cyl
    task = _MockTask(scene, items)
    sampled = phys.sample_physics(items, phys.load_physics_cfg(), np.random.RandomState(42))

    phys.apply_physics(task, sampled)
    for item_id, ent in (("bottle_0", box), ("cyl_0", cyl)):
        vals, rb = sampled[item_id], phys.read_back(ent)
        assert rb["static_friction"] == pytest.approx(vals["static_friction"], abs=ATOL)
        assert rb["dynamic_friction"] == pytest.approx(vals["dynamic_friction"], abs=ATOL)
        assert rb["restitution"] == pytest.approx(vals["restitution"], abs=ATOL)
        assert rb["mass"] == pytest.approx(vals["mass"], abs=1e-6)
        assert np.allclose(rb["com_offset"], vals["com_offset"], atol=ATOL)
        assert rb["com_offset"][2] == pytest.approx(0.0, abs=ATOL)
    # PLAN v3 §5: ItemSpec.mass reflects the value actually set
    assert items["bottle_0"].mass == pytest.approx(sampled["bottle_0"]["mass"])
    assert items["cyl_0"].mass == pytest.approx(sampled["cyl_0"]["mass"])


def test_apply_physics_particle_set():
    scene = make_scene()
    particles = [add_sphere(scene, radius=0.006, xyz=(0.1 * i, 0.5, 0.05), name=f"bead_{i}") for i in range(3)]
    spec = _spec("beads_0", "beads", [0.006, 0.006, 0.006], kind="particle_set")
    spec.entity = particles[0]
    spec.particles = particles
    task = _MockTask(scene, {"beads_0": spec})
    sampled = phys.sample_physics(task.items, phys.load_physics_cfg(), np.random.RandomState(5))
    phys.apply_physics(task, sampled)
    for ent in particles:
        rb = phys.read_back(ent)
        assert rb["mass"] == pytest.approx(sampled["beads_0"]["mass"], abs=1e-9)
        assert rb["static_friction"] == pytest.approx(sampled["beads_0"]["static_friction"], abs=ATOL)
