"""Physical-property randomization: friction, mass, center of mass (PLAN v3 §2.1).

Sampling rules (configs/physics.yaml): static friction drawn first, dynamic
friction drawn below it; restitution small; mass = per-category nominal
(configs/objects.yaml ``mass_range`` midpoint) x U(mass_scale); center-of-mass
offset = U(com_offset_xy) x half extents in x and y only (z stays 0).

sapien API facts per docs/item_envs/recon.md: material on EXISTING collision
shapes via ``scene.create_physical_material`` + the ``shape.physical_material``
setter (Q1); ``cmass_local_pose`` and ``mass`` are read/write properties,
independent of each other (Q3). Raw sapien entities and RoboTwin Actor
wrappers both work; everything is unwrapped through items.entity_of().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import os

import numpy as np
import sapien
import sapien.physx as physx
import yaml

from .items import ItemSpec, entity_of

_CONFIGS = Path(__file__).resolve().parent / "configs"


@lru_cache(maxsize=4)
def load_physics_cfg(path: str | None = None) -> dict:
    """Parsed configs/physics.yaml (randomization ranges + reference kappas)."""
    p = Path(path) if path else _CONFIGS / "physics.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def load_objects_cfg(path: str | None = None) -> dict:
    """Parsed configs/objects.yaml (categories with names, mass ranges, colors)."""
    p = Path(path) if path else _CONFIGS / "objects.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def category_nominal_masses(objects_cfg: dict) -> dict[str, float]:
    """category -> nominal mass (kg), the objects.yaml mass_range midpoint."""
    categories = objects_cfg.get("categories", {}) or {}
    out: dict[str, float] = {}
    for key, entry in categories.items():
        mass_range = (entry or {}).get("mass_range")
        if not mass_range:
            continue
        out[str(key)] = 0.5 * (float(mass_range[0]) + float(mass_range[1]))
    return out


def _rigid_comp(entity):
    """The PhysxRigidDynamicComponent, else the PhysxRigidStaticComponent, else None."""
    ent = entity_of(entity)
    comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        comp = ent.find_component_by_type(physx.PhysxRigidStaticComponent)
    return comp


def apply_material(entity, static_f: float, dynamic_f: float, restitution: float, scene) -> None:
    """Set the physical material on every existing collision shape of the entity.

    recon.md Q1: materials come from scene.create_physical_material (works on
    both RoboTwin's sapien.core scene and the headless sapien.Scene), and the
    shape.physical_material setter sticks post-build.
    """
    comp = _rigid_comp(entity)
    if comp is None:
        raise TypeError(f"no rigid component on {entity_of(entity)!r}")
    mat = scene.create_physical_material(float(static_f), float(dynamic_f), float(restitution))
    for shape in comp.get_collision_shapes():
        shape.physical_material = mat


def set_mass_props(entity, mass: float, com_offset_local=None) -> None:
    """Explicit mass (kg); optional CoM offset in the actor frame (recon.md Q3).

    com_offset_local=None leaves the auto-computed center of mass untouched.
    """
    ent = entity_of(entity)
    comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        raise TypeError(f"no dynamic rigid component on {ent!r}")
    comp.mass = float(mass)
    if com_offset_local is not None:
        off = np.asarray(com_offset_local, dtype=np.float64).reshape(3)
        comp.cmass_local_pose = sapien.Pose(p=off, q=[1, 0, 0, 0])


def sample_physics(item_specs, physics_cfg: dict, rng: np.random.RandomState) -> dict[str, dict]:
    """One sampled physics dict per item_id (deterministic via the passed rng).

    item_specs: mapping id -> ItemSpec, or an iterable of ItemSpec. The mass
    base is the objects.yaml per-category nominal; spec.mass > 0 is the
    fallback when the category is unknown. CoM offsets apply in x and y only
    (z = 0, PLAN v3 §2.1) and are skipped for particle sets (per-particle CoM
    shifts on spheres are meaningless). For particle sets every value is per
    particle.
    """
    if hasattr(item_specs, "values"):
        specs = list(item_specs.values())
    else:
        specs = list(item_specs)
    objects_cfg = load_objects_cfg()
    nominal = category_nominal_masses(objects_cfg)
    cat_cfg = objects_cfg.get("categories", {}) or {}
    s_lo, s_hi = (float(v) for v in physics_cfg["static_friction"])
    d_lo, d_hi = physics_cfg["dynamic_friction"]
    r_lo, r_hi = (float(v) for v in physics_cfg["restitution"])
    m_lo, m_hi = (float(v) for v in physics_cfg["mass_scale"])
    cxy_lo, cxy_hi = (float(v) for v in physics_cfg["com_offset_xy"])

    out: dict[str, dict] = {}
    for spec in specs:
        static_f = float(rng.uniform(s_lo, s_hi))
        hi = static_f if d_hi == "static" else float(d_hi)
        dynamic_f = float(min(rng.uniform(float(d_lo), hi), static_f))
        restitution = float(rng.uniform(r_lo, r_hi))
        pf = os.environ.get("ITEM_PARTICLE_FRICTION")  # diag-only: "lo,hi" remaps particle-set friction (same RNG stream)
        f_range = ((cat_cfg.get(str(spec.category)) or {}).get("friction_range"))
        if pf and spec.kind == "particle_set":
            f_range = [float(v) for v in pf.split(",")]
        if f_range:
            # objects.yaml categories.<cat>.friction_range: this category's
            # static friction is drawn in that range instead (the same RNG
            # draw, remapped: every other item's sample is unchanged)
            f_lo, f_hi = (float(v) for v in f_range)
            frac = (static_f - s_lo) / max(s_hi - s_lo, 1e-9)
            new_static = f_lo + frac * (f_hi - f_lo)
            dynamic_f = float(min(dynamic_f * new_static / max(static_f, 1e-9), new_static))
            static_f = float(new_static)
        d_ratio = (cat_cfg.get(str(spec.category)) or {}).get("dynamic_ratio")
        if d_ratio is not None:
            # objects.yaml categories.<cat>.dynamic_ratio: dynamic = ratio x
            # static (no stick-slip: T5's tray must slide without jolts)
            dynamic_f = float(d_ratio) * static_f
        base = nominal.get(str(spec.category))
        if base is None and float(spec.mass) > 0:
            base = float(spec.mass)
        if base is None:
            raise KeyError(f"no nominal mass for category {spec.category!r} (item {spec.item_id!r})")
        mass = float(base * rng.uniform(m_lo, m_hi))
        com_offset = None
        if spec.kind != "particle_set":
            half = np.asarray(spec.half_extents, dtype=np.float64).reshape(3)
            com_offset = np.array(
                [
                    rng.uniform(cxy_lo, cxy_hi) * half[0],
                    rng.uniform(cxy_lo, cxy_hi) * half[1],
                    0.0,
                ]
            )
        out[spec.item_id] = {
            "static_friction": static_f,
            "dynamic_friction": dynamic_f,
            "restitution": restitution,
            "mass": mass,
            "com_offset": com_offset,
        }
    return out


def apply_physics(task, sampled: dict) -> None:
    """Apply sampled values to the task's items.

    ``task`` needs a ``.scene`` (for material creation) and ``.items``
    (item_id -> ItemSpec). Everything sampled is contact-affecting, so call
    this before check_stable / the settle steps. ItemSpec.mass is updated to
    the value actually set (PLAN v3 §5 contract); for particle sets that is
    the per-particle mass.
    """
    for item_id, vals in sampled.items():
        spec: ItemSpec = task.items[item_id]
        targets = list(spec.particles or []) if spec.kind == "particle_set" else [spec.entity]
        for ent in targets:
            apply_material(ent, vals["static_friction"], vals["dynamic_friction"], vals["restitution"], task.scene)
            set_mass_props(ent, vals["mass"], vals["com_offset"])
        spec.mass = float(vals["mass"])


def read_back(entity) -> dict:
    """Actual values from the simulator, for tests and logging.

    Returns body ("dynamic" | "static"), n_shapes, the material triple, and
    for dynamic bodies mass / com_offset (actor frame) / kinematic. Float32
    noise from PhysX applies; compare with tolerances.
    """
    ent = entity_of(entity)
    dyn = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if dyn is not None:
        shapes = dyn.get_collision_shapes()
        out = {
            "body": "dynamic",
            "mass": float(dyn.mass),
            "com_offset": np.asarray(dyn.cmass_local_pose.p, dtype=np.float64).reshape(3).copy(),
            "kinematic": bool(dyn.kinematic),
        }
    else:
        sta = ent.find_component_by_type(physx.PhysxRigidStaticComponent)
        if sta is None:
            raise TypeError(f"no rigid component on {ent!r}")
        shapes = sta.get_collision_shapes()
        out = {
            "body": "static",
            "mass": None,
            "com_offset": None,
            "kinematic": None,
        }
    if shapes:
        mat = shapes[0].physical_material
        out["static_friction"] = float(mat.static_friction)
        out["dynamic_friction"] = float(mat.dynamic_friction)
        out["restitution"] = float(mat.restitution)
    out["n_shapes"] = len(shapes)
    return out
