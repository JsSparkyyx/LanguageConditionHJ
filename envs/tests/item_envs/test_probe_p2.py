"""Tests for probes/p2.py (PLAN v3 §6 P2) — pure-numpy plus headless sim cells.

Pure functions (bead grid geometry, trapezoid/tilt profiles, containment and
tunneling classification, resumable rows, cell enumeration) run anywhere. The
two ``@pytest.mark.sim`` tests build real PhysxCpuSystem scenes (bowl asset +
beads) and run headless on the login node.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from main.item_envs.probes import p2


# ---------------------------------------------------------------------------
# Bead spawn grid (pure numpy, synthetic bowl)


def _synthetic_grid():
    rng = None
    return p2.bead_grid_positions(n=30, r=0.006, inner_radius=0.040, floor_y=0.004, rng=rng)


def test_bead_grid_count_and_layers():
    pts = _synthetic_grid()
    assert pts.shape == (30, 3)
    # first layer one bead radius above the floor, layers stack at pitch s
    r, floor, s = 0.006, 0.004, 2.0 * 0.006 * 1.05
    ys = np.sort(np.unique(pts[:, 1]))
    assert abs(ys[0] - (floor + r + 5e-4)) < 1e-9
    if len(ys) > 1:
        assert abs((ys[1] - ys[0]) - s) < 1e-9


def test_bead_grid_spacing_and_inside_rim():
    pts = _synthetic_grid()
    r, inner = 0.006, 0.040
    # every bead fully inside the rim (with the 1 mm margin)
    rho = np.linalg.norm(pts[:, [0, 2]], axis=1)
    assert (rho <= inner - r - 0.001 + 1e-9).all()
    # min 3D pairwise distance = the grid pitch (no overlaps), hex packed
    d = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
    d += np.eye(len(pts)) * 1e6
    assert d.min() >= 2.0 * r * 1.05 - 1e-9


def test_bead_grid_stacks_above_rim_when_full():
    # a tiny bowl that fits only one small layer: the rest stacks upward
    pts = p2.bead_grid_positions(n=12, r=0.006, inner_radius=0.010, floor_y=0.0, rng=None)
    assert pts.shape == (12, 3)
    assert pts[:, 1].max() > 2.0 * 0.006  # more than one layer


def test_bead_grid_jitter_respects_layer_height():
    rng = np.random.RandomState(0)
    pts = p2.bead_grid_positions(n=20, r=0.006, inner_radius=0.04, floor_y=0.0, rng=rng)
    ys = np.sort(np.unique(pts[:, 1]))
    base = p2.bead_grid_positions(n=20, r=0.006, inner_radius=0.04, floor_y=0.0, rng=None)
    assert np.array_equal(ys, np.sort(np.unique(base[:, 1])))  # y untouched by jitter


def test_bead_grid_bands_tapered_cup():
    # tapered cup: wall radius 0.020 at the floor widening to 0.040 at the rim
    floor, r = 0.005, 0.006
    nb = 8
    bands = [[floor + 0.5 * p2.BAND_H + i * p2.BAND_H, 0.020 + 0.020 * i / (nb - 1)] for i in range(nb)]
    pts = p2.bead_grid_positions(n=40, r=r, inner_radius=0.040, floor_y=floor, rng=None, bands=bands)
    rho = np.linalg.norm(pts[:, [0, 2]], axis=1)
    # every bead respects the wall radius over ITS vertical extent
    for k in range(len(pts)):
        lim = p2.wall_radius_interval({"bands": bands}, pts[k, 1] - r, pts[k, 1] + r) - r - 0.001
        assert rho[k] <= max(lim, 0.0) + 1e-9
    # the bottom layer is narrower than the flat rim grid would allow ...
    y0 = pts[:, 1].min()
    assert rho[pts[:, 1] == y0].max() <= 0.020 - r - 0.001 + 1e-9
    # ... and upper layers actually use the widening wall
    assert rho.max() > 0.020


def test_wall_radius_interval_clamps():
    bands = [[0.005, 0.030], [0.0075, 0.035], [0.010, 0.040]]  # BAND_H = 0.0025
    g = {"bands": bands}
    assert p2.wall_radius_interval(g, 0.005, 0.0075) == pytest.approx(0.030)  # first band only
    assert p2.wall_radius_interval(g, 0.005, 0.0125) == pytest.approx(0.030)  # min over all
    assert p2.wall_radius_interval(g, 0.004, 0.006) == pytest.approx(0.030)  # clamped below
    assert p2.wall_radius_interval(g, 0.011, 0.020) == pytest.approx(0.040)  # clamped above
    assert p2.wall_radius_interval(g, 0.0076, 0.0099) == pytest.approx(0.035)  # middle band


# ---------------------------------------------------------------------------
# Motion profiles (pure numpy)


def test_trapezoid_full_profile_math():
    prof = p2.trapezoid_profile(distance=1.0, v_max=0.5, accel=1.0)
    assert not prof["triangular"]
    assert prof["v_peak"] == 0.5
    assert prof["t_acc"] == 0.5
    assert prof["t_cruise"] == pytest.approx(1.5)
    assert prof["t_total"] == pytest.approx(2.5)
    x = prof["x_fn"]
    assert x(0.0) == 0.0
    assert x(prof["t_total"]) == pytest.approx(1.0)
    assert x(0.5) == pytest.approx(0.125)  # 0.5*a*t^2 at the corner
    assert x(2.0) == pytest.approx(0.875)
    # exact peak accel: second derivative of x inside the ramp
    h = 1e-4
    for t in (0.1, 0.25, 0.4):
        acc = (x(t + h) - 2 * x(t) + x(t - h)) / h**2
        assert acc == pytest.approx(1.0, abs=1e-3)
    # and inside the decel ramp
    acc = (x(2.4 + h) - 2 * x(2.4) + x(2.4 - h)) / h**2
    assert acc == pytest.approx(-1.0, abs=1e-3)


def test_trapezoid_degenerates_to_triangle():
    # P2's a=1 cell: 0.2 m stroke cannot reach 0.5 m/s
    prof = p2.trapezoid_profile(distance=0.2, v_max=0.5, accel=1.0)
    assert prof["triangular"]
    assert prof["v_peak"] == pytest.approx(math.sqrt(0.2))
    assert prof["v_peak"] < 0.5
    assert prof["peak_accel"] == 1.0  # the exact peak accel is the sweep value
    assert prof["x_fn"](prof["t_total"]) == pytest.approx(0.2)
    # a=8 reaches v_max (the real trapezoid of the P2 grid)
    prof8 = p2.trapezoid_profile(distance=0.2, v_max=0.5, accel=8.0)
    assert not prof8["triangular"]
    assert prof8["v_peak"] == 0.5
    assert prof8["x_fn"](prof8["t_total"]) == pytest.approx(0.2)
    # displacement is continuous and monotone across the corners
    ts = np.linspace(0, prof8["t_total"], 500)
    xs = prof8["x_fn"](ts)
    assert (np.diff(xs) >= -1e-12).all()
    assert xs[-1] == pytest.approx(0.2)


def test_tilt_profile():
    prof = p2.tilt_profile(theta_max=math.radians(110), omega=3.0, t_ramp=0.1)
    th = prof["theta_fn"]
    assert th(0.0) == 0.0
    assert th(prof["t_total"]) == pytest.approx(math.radians(110))
    # constant-omega stretch in the middle
    ts = np.linspace(prof["t_ramp"] + 0.05, prof["t_total"] - 0.05, 20)
    slope = np.gradient(th(ts), ts)
    assert slope == pytest.approx(3.0, abs=1e-6)
    # omega=1 never saturates the ramp logic either
    prof1 = p2.tilt_profile(theta_max=math.radians(110), omega=1.0, t_ramp=0.1)
    assert prof1["theta_fn"](prof1["t_total"]) == pytest.approx(math.radians(110))


# ---------------------------------------------------------------------------
# Containment / tunneling (synthetic positions)


def _synthetic_geom():
    return {
        "inner_radius": 0.04,
        "rim_y": 0.05,
        "bottom_y": 0.0,
        "half_extents": [0.045, 0.025, 0.045],
        "obb_center_local": [0.0, 0.025, 0.0],
    }


def test_containment_counts_synthetic():
    geom = _synthetic_geom()
    # bowl sitting on the table with the real RoboTwin spawn rotation:
    # file-x -> world-y, file-y -> world-z (up), file-z -> world-x
    from main.item_envs.geometry import pose_to_mat

    world_T = pose_to_mat([0.0, 0.0, 0.74], list(p2.SPAWN_QUAT))
    r = 0.006

    def to_world(local):
        local = np.asarray(local, dtype=np.float64).reshape(-1, 3)
        return local @ world_T[:3, :3].T + world_T[:3, 3]

    local_pts = [
        [0.010, 0.020, 0.010],   # inside: rho ~0.014, y=0.02 below the rim
        [0.039, 0.049, 0.000],   # inside: just within inner radius, below rim
        [0.039, 0.051, 0.000],   # spilled: above the rim edge
        [0.000, 0.003, 0.200],   # spilled: on the table beside the bowl (world y=0)
        [0.000, -0.020, 0.000],  # tunneled: 2 cm below the exterior bottom
    ]
    pts = to_world(local_pts)
    out = p2.containment_counts(pts, world_T, geom, r, table_top=0.74)
    assert out["inside"] == 2
    assert out["spilled"] == 2
    assert out["tunneled"] == 1
    assert out["n"] == 5
    # a bead just below the table top but by less than r/2 is NOT tunneling
    pts2 = np.array([[0.2, 0.0, 0.74 - 0.002]])
    out2 = p2.containment_counts(pts2, world_T, geom, r, table_top=0.74)
    assert out2["tunneled"] == 0 and out2["spilled"] == 1


def test_under_bowl_suppressed_when_tilted():
    geom = _synthetic_geom()
    r = 0.006
    # bowl lifted 0.10 m above the table; the bead rests on the table directly
    # below the bowl centre — inside its world xy-AABB, below its world min z
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # local +y -> world +z
    c, s = math.cos(math.radians(60)), math.sin(math.radians(60))
    roty = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def world_T(rot):
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [0.0, 0.0, 0.84]
        return T

    bead = np.array([[0.0, 0.0, 0.74]])
    upright = p2.containment_counts(bead, world_T(R0), geom, r, table_top=0.74)
    assert upright["inside"] == 0
    assert upright["tunneled"] == 1  # the under-bowl clause fires when upright
    tilted = p2.containment_counts(bead, world_T(roty @ R0), geom, r, table_top=0.74)
    assert tilted["tunneled"] == 0  # suppressed under a tilted bowl's overhang
    assert tilted["inside"] == 0
    assert tilted["spilled"] == 1


# ---------------------------------------------------------------------------
# Resumable rows


def test_rows_drop_crash_truncated_last_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    for i in range(3):
        p2.append_row(path, {"cell": ["rest", 1, 30, 6.0, i], "inside": 30})
    with open(path, "a") as fh:
        fh.write('{"cell": ["rest", 1, 30, 6.0, 3], "ins')  # crash mid-write
    rows = p2.load_rows(path)
    assert len(rows) == 3
    # the corrupt line is really gone from the file
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3
    assert p2.done_cells(rows) == {("rest", 1, 30, 6.0, 0), ("rest", 1, 30, 6.0, 1), ("rest", 1, 30, 6.0, 2)}
    # appending after the truncate yields valid jsonl again
    p2.append_row(path, {"cell": ["rest", 1, 30, 6.0, 3], "inside": 30})
    assert len(p2.load_rows(path)) == 4


def test_rows_missing_file(tmp_path):
    assert p2.load_rows(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# Bowl geometry from synthetic vertices


def test_bowl_geometry_from_verts_synthetic_cup():
    # hollow cup: outer rho=0.05, inner rho=0.04, height 0.05, floor at y=0.005
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    verts = []
    for rho in (0.05, 0.04):  # walls
        for y in (0.005, 0.05):
            verts.append(np.stack([rho * np.cos(th), np.full_like(th, y), rho * np.sin(th)], axis=1))
    rim = np.stack([0.045 * np.cos(th), np.full_like(th, 0.05), 0.045 * np.sin(th)], axis=1)
    verts.append(rim)
    rr = np.linspace(0, 0.04, 6)  # interior floor disk + exterior bottom disk
    for y in (0.005, 0.0):
        disk = [np.array([r_ * math.cos(a), y, r_ * math.sin(a)]) for r_ in rr for a in th[::8]]
        verts.append(np.array(disk))
    verts = np.concatenate(verts, axis=0)
    geom = p2.bowl_geometry_from_verts(verts)
    assert geom["rim_y"] == pytest.approx(0.05)
    assert geom["bottom_y"] == pytest.approx(0.0)
    assert geom["outer_radius"] == pytest.approx(0.05)
    assert geom["inner_radius"] == pytest.approx(0.04, abs=1e-6)
    assert geom["floor_y"] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Cell enumeration / sharding


def test_iter_cells_grid_and_shard_partition():
    cells = p2.iter_cells()
    assert len(cells) == 4 * 3 * 3 * 3 + 4 * (4 + 2) * 3  # 108 rest + 72 kinematic
    assert cells[0] == ("rest", 1, 30, 5.0, 0)
    kinds = {c[0] for c in cells}
    assert kinds == {"rest", "translate", "tilt"}
    # 3 shards partition the grid exactly
    shards = [[c for i, c in enumerate(cells) if i % 3 == s] for s in range(3)]
    flat = [c for shard in shards for c in shard]
    assert sorted(flat) == sorted(cells)
    # kinematic cells ride the default cell
    for c in cells:
        if c[0] in ("translate", "tilt"):
            assert len(c) == 4


def test_cell_rng_deterministic_and_distinct():
    c1 = ("rest", 1, 60, 6.0, 0)
    a = p2._cell_rng(c1).rand(4)
    b = p2._cell_rng(c1).rand(4)
    c = p2._cell_rng(("rest", 1, 60, 6.0, 1)).rand(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Sim cells (headless PhysxCpuSystem — login node)

sim = pytest.mark.sim


@sim
def test_rest_containment_cell_sim(tmp_path):
    """One rest cell of the P2 grid: bowl id 1, N=30, r=6 mm, seed 0."""
    pytest.importorskip("sapien")
    row = p2.run_rest_cell(("rest", 1, 30, 6.0, 0), tmp_path, visuals=False, keyframes=False)
    assert row["inside"] == 30, f"beads escaped at rest: {row}"
    assert row["tunneled"] == 0
    assert row["spilled"] == 0
    assert row["end_max_speed"] < 1e-2


@sim
def test_translate_cell_sim(tmp_path):
    """One kinematic translate cell: bowl id 1, a=2 m/s^2, seed 0."""
    pytest.importorskip("sapien")
    row = p2.run_translate_cell(("translate", 1, 2.0, 0), tmp_path, visuals=False, keyframes=False)
    assert row["tunneled"] == 0
    assert row["inside"] + row["spilled"] + row["off_table"] == 60
    assert row["v_peak"] == 0.5  # a=2 reaches v_max on the 0.2 m stroke
    assert not row["triangular"]
    # the kinematic stroke leaves the bowl at x = +0.1
    assert row["traj_min_z"] > 0.74 - 0.004
    assert row["active_ms_per_tick"] < 20.0
