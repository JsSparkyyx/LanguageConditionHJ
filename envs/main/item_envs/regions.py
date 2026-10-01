"""Region: the region-side data contract (PLAN Section 5).

Only the dataclass lives here for now; the sim constructors
(region_from_actor_top, region_from_link, table_region, zone_region) are
Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(kw_only=True)
class Region:
    """A named 2D region on a (possibly moving) frame, evaluated every tick."""

    region_id: str  # e.g. "tray_top", "zone_red", "table_top", "drawer_floor"
    kind: str  # "surface" | "zone" | "container"
    frame_fn: Callable[[], np.ndarray]  # returns 4x4 world_T_region, evaluated every tick
    polygon: np.ndarray  # [M, 2] CCW polygon in the region frame's xy plane (m)
    z: float  # surface height in the region frame (usually 0)
    z_band: float  # vertical tolerance for "on surface" checks (m)
    phrases_key: str | None = None  # key into phrases.yaml, None if not nameable

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        self.z = float(self.z)
        self.z_band = float(self.z_band)

    def world_T(self) -> np.ndarray:
        """Current 4x4 world_T_region."""
        return np.asarray(self.frame_fn(), dtype=np.float64)
