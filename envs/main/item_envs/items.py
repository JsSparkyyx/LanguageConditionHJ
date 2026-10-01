"""ItemSpec: the item-side data contract (PLAN Section 5).

No SAPIEN import at module level: entities are generic objects, unwrapped via
duck-typing in entity_of().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(kw_only=True)
class ItemSpec:
    """One safety-relevant item in a scene.

    Extra attributes (e.g. tippable, rolls) may be attached by tasks; rule
    templates match on them via getattr.
    """

    item_id: str  # unique within episode, e.g. "bottle_0"
    entity: object  # sapien Entity (raw) or RoboTwin Actor; use entity_of()
    category: str  # e.g. "bottle"; key into phrases.yaml
    color: str | None = None  # for primitives and zone mats; None for textured assets
    kind: str = "rigid"  # "rigid" | "particle_set"
    particles: list | None = None  # list of entities when kind == "particle_set"
    half_extents: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [3] OBB half extents (m)
    obb_center_local: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [3] OBB center, item frame
    up_axis_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # [3] unit
    mass: float = 0.0  # kg, the value actually set
    role: str = "constrained_candidate"  # "constrained_candidate" | "task_object" | "support" | "bystander"

    def __post_init__(self):
        self.half_extents = np.asarray(self.half_extents, dtype=np.float64).reshape(3)
        self.obb_center_local = np.asarray(self.obb_center_local, dtype=np.float64).reshape(3)
        self.up_axis_local = np.asarray(self.up_axis_local, dtype=np.float64).reshape(3)
        self.mass = float(self.mass)


def entity_of(obj) -> object:
    """Unwrap to the raw simulator entity.

    Accepts an ItemSpec (its .entity field), a RoboTwin Actor wrapper (raw
    entity under .actor), or a raw entity (returned as-is). Duck-typing only.
    """
    if isinstance(obj, ItemSpec):
        return entity_of(obj.entity)
    inner = getattr(obj, "actor", None)
    if inner is not None and hasattr(inner, "get_pose"):
        return inner
    return obj
