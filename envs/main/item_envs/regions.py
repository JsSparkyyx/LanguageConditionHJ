"""Region: the region-side data contract (PLAN v3 Section 5).

Only the dataclass lives here; the sim constructors (table_region,
zone_region, container_region) are added by the Phase 2 relaunch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


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

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        self.z = float(self.z)
        self.z_band = float(self.z_band)
        self.nameable = bool(self.nameable)

    def world_T(self) -> np.ndarray:
        """Current 4x4 world_T_region."""
        return np.asarray(self.frame_fn(), dtype=np.float64)
