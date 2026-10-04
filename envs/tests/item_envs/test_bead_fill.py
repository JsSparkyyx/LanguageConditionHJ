"""Non-sim tests for the bead-fill helpers (T2 / T4, user reviews 4-5):
``bead_random_positions`` (random-pack spawn) and ``bowl_bead_capacity``
(volume-based rim capacity), on a synthetic straight-walled bowl."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ITEM_UTILS = Path(__file__).resolve().parents[2] / "robotwin_tasks" / "_item_utils.py"


def _iu():
    spec = importlib.util.spec_from_file_location("_item_utils_fill", ITEM_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cylinder_geom(iu, radius=0.055, floor=0.005, rim=0.030):
    bands = [[y, radius] for y in np.arange(floor, rim, iu.POUR_BAND_H)]
    return {"inner_radius": radius, "outer_radius": radius + 0.005, "floor_y": floor, "rim_y": rim, "bands": bands}


def test_capacity_matches_random_pack_volume():
    iu = _iu()
    g = _cylinder_geom(iu)
    r = 0.006
    cap = iu.bowl_bead_capacity(g, r, phi=0.58)
    n_bands = len([b for b in g["bands"] if b[0] + iu.POUR_BAND_H <= g["rim_y"]])
    vol = np.pi * (g["inner_radius"] - 0.0005) ** 2 * iu.POUR_BAND_H * n_bands
    assert cap == int(round(0.58 * vol / (4.0 / 3.0 * np.pi * r ** 3)))
    assert iu.bowl_bead_capacity(g, r, phi=0.29) < cap  # monotone in phi


def test_random_positions_non_overlapping_inside_the_walls():
    iu = _iu()
    g = _cylinder_geom(iu)
    r = 0.006
    n = int(1.15 * iu.bowl_bead_capacity(g, r))  # held-mode over-fill: spills above the rim
    pts = iu.bead_random_positions(n, r, g, np.random.RandomState(0))
    assert pts.shape == (n, 3)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(n)
    assert d.min() >= 2.0 * r - 1e-9  # no overlap
    rho = np.hypot(pts[:, 0], pts[:, 2])  # file frame: y up
    assert (rho <= g["inner_radius"] - r).all()  # inside the wall / rim column
    assert (pts[:, 1] >= g["floor_y"] + r).all()  # above the floor
    assert pts[:, 1].max() > g["rim_y"]  # the over-fill stacks into the column above the rim


def test_random_positions_deterministic_per_seed():
    iu = _iu()
    g = _cylinder_geom(iu)
    a = iu.bead_random_positions(40, 0.006, g, np.random.RandomState(3))
    b = iu.bead_random_positions(40, 0.006, g, np.random.RandomState(3))
    assert np.array_equal(a, b)
