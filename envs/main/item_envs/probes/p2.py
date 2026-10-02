"""P2 bowl containment and beads + P4 T2-layout visibility (PLAN v3 §6 Phase 3).

P2 verifies dynamically what the Phase-0 collision audit showed statically:
``002_bowl`` ids [1, 2, 3, 5] are hollow containers that hold beads. Three
families, all on a bare scene (no RoboTwin task — physics isolated from the
planner, PLAN §6):

- rest containment: N x r bead grid (full cross N in {30, 60, 100}, r in
  {5, 6, 8} mm) settles inside the bowl on the table; every bead must end
  inside the bowl interior, none may tunnel. This is also where the physics
  WALL TIME per control tick (10 substeps of 1/250 s) is measured at N=60.
- translate: the bowl is a KINEMATIC body (recon Q5: dynamic build +
  ``component.kinematic = True`` + ``set_kinematic_target``) driven along a
  0.2 m stroke with a trapezoid velocity profile (v_max 0.5 m/s, peak accel
  {1, 2, 4, 8} m/s^2 — the trapezoid degenerates to a triangle when the
  stroke is too short to reach v_max). Spill must increase with accel.
- tilt: kinematic rotation to 110 deg about a horizontal axis through the
  bowl centre at omega in {1, 3} rad/s. Spill must increase with omega.

Measurement definitions (the PLAN leaves room; these are the choices):

- INSIDE (rest containment): the bead centre, expressed in the bowl's actor
  frame (the glb file frame, y = up, origin at the bowl's exterior bottom),
  has horizontal radius < the inner rim radius AND y < rim height AND
  y > exterior bottom - r/2. The rim/inner radii come from the collision
  pieces themselves (``bowl_geometry``), not the json extents.
- TUNNELED (zero tolerance): the bead centre is below the table top by more
  than r/2 (through the bowl bottom and/or the table — the bowl rests on the
  table, so its exterior bottom plane IS the table top), OR it lies inside
  the bowl's current world xy-AABB and below the bowl's world AABB minimum z
  by more than r/2 — this under-bowl clause only while the bowl is upright
  within 45 deg (a tilted bowl's AABB legitimately overhangs the table during
  the pour). ``traj_min_z`` records the minimum bead z over the whole
  trajectory as the transient diagnostic.
- SPILLED: not inside and not tunneled (on the table, legitimately out).

Headless path (login node, CPU partition): ``sapien.Scene([PhysxCpuSystem()])``,
collision-only bodies — RoboTwin's ``create_actor`` cannot be used (it adds
visuals, which need Vulkan; recon "CPU-node gotcha"), so ``add_bowl`` mirrors
its collision path exactly (same file resolution as create_actor.py:518-532,
same json scale, same ``add_multiple_convex_collisions_from_file``). With
``visuals=True`` (GPU node) the same builders run on a full ``sapien.Scene()``
and keyframes are real renders from a scene camera; headless keyframes are
matplotlib top/side scatter views. Physics is identical (PhysX CPU both ways).

P4 T2 visibility (bottom of this module): 20 layout seeds of the T2
prototype — source bowl + 60 beads + a static target bowl where the host
places its plate + two zone mats near it — built inside ONE CEHJ ``Env`` on
the ``smoke_items`` robotwin task (the camera-rig choice: reusing the Env
gives RoboTwin's exact D435 head/wrist layout and the
``get_encoder_obs``-parity preprocessing for free; layouts are re-rolled in
the same scene to avoid 20 Env constructions and their TLS-key cost).
Gate (PLAN §6 P4): every candidate (bowl, target, each zone mat) has visible
fraction >= 0.3 in at least one camera in >= 95% of seeds, measured with
``masks.item_masks`` at t = 0. Rendering required -> GPU node.

Rows: ``<out_dir>/p2_rows.jsonl`` (P2) and ``<out_dir>/p4_t2_rows.jsonl``
(P4); resumable per the probes/ package convention.
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..geometry import pose_to_mat, quat_wxyz_to_mat, to_frame
from ..items import ItemSpec

# ---------------------------------------------------------------------------
# Constants (PLAN v3 §6 P2 / §7 T2)

OBJ_ROOT = Path(
    os.environ.get(
        "ROBOTWIN_ASSETS",
        "/work/nvme/bifq/project/RoboTwin/assets/objects",
    )
)

BOWL_MODEL = "002_bowl"
BOWL_IDS = (1, 2, 3, 5)  # the host/T2 id set
N_BEADS = (30, 60, 100)
R_BEADS_MM = (5.0, 6.0, 8.0)
SEEDS = (0, 1, 2)
DEFAULT_N = 60
DEFAULT_R_MM = 6.0

ACCELS = (1.0, 2.0, 4.0, 8.0)  # m/s^2, translate family
OMEGAS = (1.0, 3.0)  # rad/s, tilt family
V_MAX = 0.5  # m/s trapezoid cruise speed
STROKE = 0.20  # m translate stroke
TILT_DEG = 110.0
TILT_LIFT = 0.10  # m, bowl lifted above the table before tilting

PHYS_DT = 1.0 / 250.0  # RoboTwin scene timestep
TICK_STEPS = 10  # one control tick = 10 physics steps (25 Hz control)
TABLE_TOP = 0.74  # RoboTwin table height (table_z_bias = 0 in the bare scene)
TABLE_HALF = (0.6, 0.35, 0.02)  # 1.2 x 0.7 m top, thin slab
SPAWN_QUAT = (0.5, 0.5, 0.5, 0.5)  # RoboTwin upright convention (file y -> world z)

SETTLE_MAX_STEPS = 1500  # 6 s cap
SETTLE_TOL_SPEED = 2e-3  # m/s, every bead below this ...
SETTLE_QUIET_CHECKS = 4  # ... for 4 consecutive checks (checked every 25 steps)
TIMING_TICKS = 50  # physics-timing measurement length at N=60
GATE_MS_PER_TICK = 20.0

ROWS_FILE = "p2_rows.jsonl"
P4_ROWS_FILE = "p4_t2_rows.jsonl"


def _cpu_model() -> str:
    """First 'model name' from /proc/cpuinfo (empty string when unreadable)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Resumable rows (shared probes/ convention)


def load_rows(path: Path) -> list[dict]:
    """Valid rows of a jsonl file; a crash-truncated last line is dropped.

    The drop is real: the file is truncated back to the end of the last
    valid line (mid-file corruption is left untouched and reported).
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows: list[dict] = []
    good_end = 0
    for line in raw.split(b"\n"):
        if not line.strip():
            good_end += len(line) + 1
            continue
        try:
            rows.append(json.loads(line))
            good_end += len(line) + 1
        except json.JSONDecodeError:
            break  # truncated/corrupt: everything from here on is dropped
    if good_end < len(raw):
        with open(path, "r+b") as fh:
            fh.truncate(good_end)
    return rows


def append_row(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def done_cells(rows: list[dict]) -> set[tuple]:
    """Cell-id tuples that already have a complete row."""
    return {tuple(r["cell"]) for r in rows if "cell" in r}


# ---------------------------------------------------------------------------
# Asset resolution (mirrors create_actor.py:487-532 / audit_collisions.py)


def _glb_or_obj(modeldir: Path, model_id: int) -> Path:
    f = modeldir / f"base{model_id}.glb"
    return f if f.exists() else modeldir / f"textured{model_id}.obj"


def resolve_asset_files(modelname: str, model_id: int, obj_root: Path = OBJ_ROOT):
    """(collision_file, visual_file) exactly the way create_actor resolves them."""
    modeldir = Path(obj_root) / modelname
    collision = _glb_or_obj(modeldir / "collision", model_id) if (modeldir / "collision").exists() else None
    if collision is None or not collision.exists():
        collision = _glb_or_obj(modeldir, model_id)
    visual = _glb_or_obj(modeldir / "visual", model_id) if (modeldir / "visual").exists() else None
    if visual is None or not visual.exists():
        visual = _glb_or_obj(modeldir, model_id)
    if not collision.exists() or not visual.exists():
        raise FileNotFoundError(f"{modelname} id {model_id}: collision/visual file missing under {modeldir}")
    return collision, visual


def asset_scale(modelname: str, model_id: int, obj_root: Path = OBJ_ROOT):
    """The model_data json scale — a scalar or a 3-list, as create_actor uses it."""
    jp = Path(obj_root) / modelname / f"model_data{model_id}.json"
    with open(jp, "r", encoding="utf-8") as fh:
        return json.load(fh)["scale"]


# ---------------------------------------------------------------------------
# Bowl geometry from the collision pieces (pure numpy/trimesh — no sapien)


def _collision_verts(modelname: str, model_id: int, obj_root: Path = OBJ_ROOT) -> np.ndarray:
    """All collision-piece vertices in the file frame (scaled to metres).

    The glb pieces sit in per-node LOCAL frames (piece centres at the
    origin); the scene-graph node transforms place them around the bowl —
    bake them in, matching what sapien cooks (recon Q13 runtime AABB:
    y [0.0002, 1.0394] for id 1, raw concatenation gives y [-0.52, 0.52]).
    """
    import trimesh

    collision, _ = resolve_asset_files(modelname, model_id, obj_root)
    scale = asset_scale(modelname, model_id, obj_root)
    loaded = trimesh.load(str(collision), force="scene", process=False)
    verts = []
    if hasattr(loaded, "graph") and hasattr(loaded, "geometry"):
        for node in loaded.graph.nodes_geometry:
            T, geom_name = loaded.graph[node]
            g = loaded.geometry[geom_name]
            v = getattr(g, "vertices", None)
            if v is None or len(v) < 4:
                continue
            verts.append(np.asarray(v, dtype=np.float64) @ T[:3, :3].T + T[:3, 3])
    else:
        v = getattr(loaded, "vertices", None)
        if v is not None and len(v) >= 4:
            verts.append(np.asarray(v, dtype=np.float64))
    if not verts:
        raise ValueError(f"no collision geometry in {collision}")
    return np.concatenate(verts, axis=0) * np.asarray(scale, dtype=np.float64)


BAND_H = 0.0025  # m, interior-radius band height for taper-aware spawning


def bowl_geometry_from_verts(verts: np.ndarray) -> dict:
    """Container geometry estimates in the file frame (y = up, origin at bottom).

    - rim_y / bottom_y: collision-AABB top / bottom along the up axis.
    - outer_radius: half the smaller horizontal AABB extent.
    - inner_radius: minimum horizontal radius over vertices in the top 15%
      band — the inner edge of the rim ring. Used by the containment check.
    - floor_y: highest vertex below half height within 0.4 x inner radius of
      the axis — the interior floor surface (spawn reference only; the
      containment check never uses it).
    - bands: [[y_lo, min_rho], ...] — inner wall radius per BAND_H-tall band
      from just above the floor to the rim. The bowls TAPER toward the
      bottom (bowl 5: 4.2 cm at the floor vs 6.0 at the rim), so the spawn
      grid sizes each layer by the bands overlapping its vertical extent.
    - half_extents / obb_center_local: the full AABB, for ItemSpec/masks.
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    lo, hi = v.min(0), v.max(0)
    rim_y, bottom_y = float(hi[1]), float(lo[1])
    outer_radius = 0.5 * float(min(hi[0] - lo[0], hi[2] - lo[2]))
    rho = np.linalg.norm(v[:, [0, 2]], axis=1)
    top_band = v[v[:, 1] > rim_y - 0.15 * (rim_y - bottom_y)]
    inner_radius = float(np.linalg.norm(top_band[:, [0, 2]], axis=1).min()) if len(top_band) else outer_radius
    central = v[(rho < 0.4 * inner_radius) & (v[:, 1] < 0.5 * rim_y)]
    floor_y = float(central[:, 1].max()) if len(central) else bottom_y
    bands = []
    y_lo = floor_y + 0.5 * BAND_H
    while y_lo < rim_y - BAND_H:
        band = (v[:, 1] >= y_lo) & (v[:, 1] < y_lo + BAND_H)
        if band.any():
            bands.append([float(y_lo), float(rho[band].min())])
        y_lo += BAND_H
    if not bands:
        bands = [[floor_y, inner_radius]]
    return {
        "rim_y": rim_y,
        "bottom_y": bottom_y,
        "outer_radius": outer_radius,
        "inner_radius": inner_radius,
        "floor_y": floor_y,
        "bands": bands,
        "half_extents": (0.5 * (hi - lo)).tolist(),
        "obb_center_local": (0.5 * (hi + lo)).tolist(),
    }


def wall_radius_interval(geom: dict, y_lo: float, y_hi: float) -> float:
    """Min inner wall radius over the vertical interval [y_lo, y_hi].

    Band lookup with clamping to the bands' y-extent (a bead reaching above
    the measured bands is sized by the top band; below them, the bottom
    band). Conservative within the BAND_H resolution.
    """
    bands = geom["bands"]
    lo0, hi0 = bands[0][0], bands[-1][0] + BAND_H
    y_lo = max(float(y_lo), lo0)
    y_hi = min(max(float(y_hi), y_lo), hi0)
    radii = [mr for lo, mr in bands if lo + BAND_H > y_lo and lo < y_hi]
    if not radii:  # interval entirely between two bands (sub-band gap)
        mid = 0.5 * (y_lo + y_hi)
        radii = [min(bands, key=lambda b: abs(b[0] + 0.5 * BAND_H - mid))[1]]
    return float(min(radii))


@lru_cache(maxsize=16)
def bowl_geometry(model_id: int, obj_root: str = str(OBJ_ROOT)) -> dict:
    """Cached bowl_geometry_from_verts for a 002_bowl model id."""
    return bowl_geometry_from_verts(_collision_verts(BOWL_MODEL, model_id, Path(obj_root)))


# ---------------------------------------------------------------------------
# Bead spawn grid (pure numpy)


def bead_grid_positions(
    n: int,
    r: float,
    inner_radius: float,
    floor_y: float,
    spacing_factor: float = 1.05,
    rng: np.random.RandomState | None = None,
    bands: list | None = None,
) -> np.ndarray:
    """[n, 3] bead centres in the bowl up-frame (y = up, origin at exterior bottom).

    Hexagonal layers (row spacing s*sqrt(3)/2, alternate rows offset s/2),
    first layer one bead radius above ``floor_y``, layer pitch = s. Cells
    keep the bead fully inside ``inner_radius - 1 mm``; with ``bands`` (the
    taper profile from ``bowl_geometry_from_verts``) each layer is instead
    sized by the wall radius over the bead's vertical extent — the bowls
    taper toward the bottom (bowl 5: 4.2 cm at the floor vs 6.0 at the rim)
    and a flat rim-sized grid spawns the lower layers INTO the wall. Layers
    stack above the rim when the bowl cannot hold ``n`` beads — that is the
    measurement, not an error (the overflow spills during settle).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if inner_radius <= r:
        raise ValueError(f"inner_radius {inner_radius} too small for bead r {r}")
    s = 2.0 * r * float(spacing_factor)
    keep_r = inner_radius - r - 0.001
    dy = s * math.sqrt(3.0) / 2.0
    # rows/columns CENTERED on the bowl axis (row j at z = j*dy, odd rows
    # offset by s/2 in x) so even a tiny interior always contains (0, 0);
    # jmax/imax cover the WIDEST layer (bands only shrink the per-layer keep)
    wide = keep_r
    if bands is not None:
        wide = max(keep_r, max(mr for _, mr in bands) - r - 0.001)
    jmax = int(math.ceil(wide / dy)) + 1
    imax = int(math.ceil(wide / s)) + 1
    pts: list[list[float]] = []
    layer = 0
    while len(pts) < n:
        y = floor_y + r + 5e-4 + layer * s
        keep = keep_r
        if bands is not None:
            keep = max(0.0, wall_radius_interval({"bands": bands}, y - r, y + r) - r - 0.001)
        added = 0
        for j in range(-jmax, jmax + 1):
            if len(pts) >= n:
                break
            z = j * dy
            if abs(z) > keep:
                continue
            x_off = (s / 2.0) if (j % 2) else 0.0
            for i in range(-imax, imax + 1):
                x = i * s + x_off
                if x * x + z * z <= keep * keep:
                    pts.append([float(x), float(y), float(z)])
                    added += 1
        if added == 0:
            raise ValueError(f"no grid cell fits inner_radius {inner_radius} with bead r {r}")
        layer += 1
    out = np.asarray(pts[:n], dtype=np.float64)
    if rng is not None:
        jitter = rng.uniform(-2e-4, 2e-4, size=(n, 3))
        jitter[:, 1] = 0.0  # keep the layer height exact
        out = out + jitter
    return out


# ---------------------------------------------------------------------------
# Motion profiles (pure numpy)


def trapezoid_profile(distance: float, v_max: float, accel: float) -> dict:
    """1D rest-to-rest trapezoid: +a to v_max, cruise, -a to 0.

    Degenerates to a triangle (peak v = sqrt(a*D) < v_max) when the stroke
    cannot reach v_max. Returns x(t) plus the exact metadata; the profile is
    analytic, so ``x_fn`` is vector-safe and exact at every t.
    """
    D, vm, a = float(distance), float(v_max), float(accel)
    if D <= 0 or vm <= 0 or a <= 0:
        raise ValueError("distance, v_max, accel must be positive")
    d_acc = 0.5 * vm * vm / a
    if 2.0 * d_acc >= D:  # triangular
        v_peak = math.sqrt(a * D)
        t_acc = v_peak / a
        t_cruise = 0.0
    else:
        v_peak = vm
        t_acc = vm / a
        t_cruise = (D - 2.0 * d_acc) / vm
    t_total = 2.0 * t_acc + t_cruise

    def x_fn(t):
        tt = np.clip(np.asarray(t, dtype=np.float64), 0.0, t_total)
        x = np.empty_like(tt)
        m1 = tt <= t_acc
        x[m1] = 0.5 * a * tt[m1] ** 2
        m2 = (~m1) & (tt <= t_acc + t_cruise)
        x[m2] = 0.5 * a * t_acc**2 + v_peak * (tt[m2] - t_acc)
        m3 = ~(m1 | m2)
        td = tt[m3] - t_acc - t_cruise
        x[m3] = D - 0.5 * a * (t_acc - td) ** 2
        return x

    return {
        "x_fn": x_fn,
        "distance": D,
        "v_max": vm,
        "v_peak": v_peak,
        "peak_accel": a,
        "t_acc": t_acc,
        "t_cruise": t_cruise,
        "t_total": t_total,
        "triangular": bool(2.0 * d_acc >= D),
    }


def tilt_profile(theta_max: float, omega: float, t_ramp: float = 0.1) -> dict:
    """Constant-omega rotation to theta_max with a short linear accel ramp.

    theta(t): ramp at angular accel omega/t_ramp up to omega, hold omega,
    stop dead at theta_max (kinematic body — the stop is a velocity step,
    physically an impulse on the beads; that is the point of the probe).
    """
    th, w, tr = float(theta_max), float(omega), float(t_ramp)
    if th <= 0 or w <= 0 or tr <= 0:
        raise ValueError("theta_max, omega, t_ramp must be positive")
    alpha = w / tr
    th_ramp = 0.5 * w * tr
    if 2.0 * th_ramp >= th:  # never reaches omega
        tr_eff = math.sqrt(th / alpha)
        w_peak = alpha * tr_eff
        t_total = tr_eff
    else:
        tr_eff = tr
        w_peak = w
        t_total = tr + (th - th_ramp) / w

    def theta_fn(t):
        tt = np.clip(np.asarray(t, dtype=np.float64), 0.0, t_total)
        return np.where(tt <= tr_eff, 0.5 * alpha * tt**2, th_ramp + w_peak * (tt - tr_eff))

    return {
        "theta_fn": theta_fn,
        "theta_max": th,
        "omega": w,
        "omega_peak": w_peak,
        "t_ramp": tr_eff,
        "t_total": t_total,
    }


# ---------------------------------------------------------------------------
# Containment / tunneling / spill (pure numpy)


def containment_counts(
    bead_world: np.ndarray,
    world_T_bowl: np.ndarray,
    geom: dict,
    bead_r: float,
    table_top: float = TABLE_TOP,
    table_half_xy: tuple[float, float] = (TABLE_HALF[0], TABLE_HALF[1]),
) -> dict:
    """Classify bead centres: inside the bowl interior / tunneled / spilled / off_table.

    INSIDE: actor-frame horizontal radius < inner_radius AND
    bottom_y - r/2 < y < rim_y (the actor frame is the glb file frame,
    y = up, origin at the exterior bottom). TUNNELED: centre below the table
    top by more than r/2 WHILE OVER the table footprint (through the bowl
    bottom and/or the table), or — only while the bowl is upright within
    45 deg — inside the bowl's world xy-AABB and below the bowl's world AABB
    min z by more than r/2. OFF_TABLE: below the table top by more than r/2
    outside the table footprint (fell off the edge — there is no floor in
    the probe scene). SPILLED: the rest (out of the bowl, on the table).
    """
    pts = np.asarray(bead_world, dtype=np.float64).reshape(-1, 3)
    world_T_bowl = np.asarray(world_T_bowl, dtype=np.float64).reshape(4, 4)
    local = to_frame(pts, world_T_bowl)  # actor (file) frame: y up
    rho = np.linalg.norm(local[:, [0, 2]], axis=1)
    inner = float(geom["inner_radius"])
    rim = float(geom["rim_y"])
    bottom = float(geom["bottom_y"])
    inside = (rho < inner) & (local[:, 1] < rim) & (local[:, 1] > bottom - 0.5 * bead_r)

    hx, hy = float(table_half_xy[0]), float(table_half_xy[1])
    over_table = (np.abs(pts[:, 0]) <= hx) & (np.abs(pts[:, 1]) <= hy)
    below = pts[:, 2] < table_top - 0.5 * bead_r
    below_table = below & over_table
    off_table = below & ~over_table
    half = np.asarray(geom["half_extents"], dtype=np.float64)
    center = np.asarray(geom["obb_center_local"], dtype=np.float64)
    corners_local = center[None, :] + np.array(
        [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    corners_world = corners_local @ world_T_bowl[:3, :3].T + world_T_bowl[:3, 3]
    wlo, whi = corners_world.min(0), corners_world.max(0)
    # the under-bowl tunnel test is only meaningful when the bowl is upright:
    # tilted (pour family), its world AABB overhangs the table and beads
    # legitimately resting on the table below the overhang are NOT tunneling
    up_world_z = float((world_T_bowl[:3, :3] @ np.array([0.0, 1.0, 0.0]))[2])
    upright = up_world_z > math.cos(math.radians(45.0))
    under_bowl = upright & (
        (pts[:, 0] >= wlo[0]) & (pts[:, 0] <= whi[0])
        & (pts[:, 1] >= wlo[1]) & (pts[:, 1] <= whi[1])
        & (pts[:, 2] < wlo[2] - 0.5 * bead_r)
    )
    tunneled = below_table | under_bowl
    spilled = ~inside & ~tunneled & ~off_table
    return {
        "inside": int(inside.sum()),
        "tunneled": int(tunneled.sum()),
        "spilled": int(spilled.sum()),
        "off_table": int(off_table.sum()),
        "n": int(len(pts)),
        "inside_frac": float(inside.mean()) if len(pts) else 0.0,
        "min_z_world": float(pts[:, 2].min()) if len(pts) else float("nan"),
        "spill_xy": pts[spilled][:, :2].tolist(),
        "tunnel_xyz": pts[tunneled].tolist(),
    }


# ---------------------------------------------------------------------------
# Scene building (sapien; headless or full-render)


def build_scene(visuals: bool = False):
    """Bare scene: table slab + gravity, 1/250 s timestep.

    visuals=False -> sapien.Scene([PhysxCpuSystem()]) (the ONLY headless
    path, recon "CPU-node gotcha"); True -> sapien.Scene() with the render
    system (GPU node) so keyframes are real renders. PhysX CPU either way.
    """
    import sapien
    import sapien.physx as physx

    if visuals:
        scene = sapien.Scene()
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0.2, 1.0, -0.8], [0.85, 0.85, 0.85], shadow=False)
        scene.add_point_light([0.0, -0.6, 1.6], [0.35, 0.35, 0.35], shadow=False)
    else:
        scene = sapien.Scene([physx.PhysxCpuSystem()])
    scene.set_timestep(PHYS_DT)
    mat = scene.create_physical_material(0.5, 0.5, 0.0)  # RoboTwin default
    tb = scene.create_actor_builder()
    tb.add_box_collision(
        pose=sapien.Pose([0, 0, TABLE_TOP - TABLE_HALF[2]]),
        half_size=list(TABLE_HALF),
        material=mat,
    )
    if visuals:
        tb.add_box_visual(
            pose=sapien.Pose([0, 0, TABLE_TOP - TABLE_HALF[2]]),
            half_size=list(TABLE_HALF),
            material=[0.6, 0.5, 0.4, 1.0],
        )
    tb.set_physx_body_type("static")
    tb.build(name="table")
    return scene


def add_bowl(scene, model_id: int, pose, kinematic: bool = False, visuals: bool = False):
    """002_bowl actor, create_actor's collision path (convex hulls per sub-mesh).

    Headless-safe: visuals are added only when requested (they need Vulkan).
    ``kinematic=True`` follows recon Q5 path B (post-build flag), driven with
    ``set_kinematic_target`` by the caller every physics step.
    """
    import sapien.physx as physx

    collision, visual = resolve_asset_files(BOWL_MODEL, model_id)
    scale = asset_scale(BOWL_MODEL, model_id)
    scale3 = tuple(scale) if isinstance(scale, (list, tuple)) else (scale,) * 3
    b = scene.create_actor_builder()
    b.add_multiple_convex_collisions_from_file(filename=str(collision), scale=scale3)
    if visuals:
        b.add_visual_from_file(filename=str(visual), scale=scale3)
    actor = b.build(name=f"{BOWL_MODEL}_{model_id}")
    actor.set_pose(pose)
    if kinematic:
        comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
        comp.kinematic = True
    return actor


def add_beads(scene, positions_world: np.ndarray, r: float, rng: np.random.RandomState, visuals: bool = False):
    """N dynamic spheres; mass U(objects.yaml bead range), friction/restitution
    from physics.yaml ranges (per bead, PLAN v3 §2.1 ranges)."""
    import sapien
    import sapien.physx as physx

    from ..physics import load_objects_cfg, load_physics_cfg

    mass_lo, mass_hi = (float(v) for v in load_objects_cfg()["categories"]["beads"]["mass_range"])
    pcfg = load_physics_cfg()
    s_lo, s_hi = (float(v) for v in pcfg["static_friction"])
    d_lo = float(pcfg["dynamic_friction"][0])
    e_lo, e_hi = (float(v) for v in pcfg["restitution"])

    pts = np.asarray(positions_world, dtype=np.float64).reshape(-1, 3)
    beads = []
    for i, p in enumerate(pts):
        static_f = float(rng.uniform(s_lo, s_hi))
        dynamic_f = float(min(rng.uniform(d_lo, static_f), static_f))
        rest = float(rng.uniform(e_lo, e_hi))
        mat = scene.create_physical_material(static_f, dynamic_f, rest)
        b = scene.create_actor_builder()
        b.add_sphere_collision(radius=float(r), material=mat)
        if visuals:
            b.add_sphere_visual(radius=float(r), material=[0.85, 0.15, 0.15, 1.0])
        bead = b.build(name=f"bead_{i}")
        bead.set_pose(sapien.Pose(p=p))
        comp = bead.find_component_by_type(physx.PhysxRigidDynamicComponent)
        comp.mass = float(rng.uniform(mass_lo, mass_hi))
        beads.append(bead)
    return beads


def _bead_positions(beads) -> np.ndarray:
    return np.array([np.asarray(b.get_pose().p, dtype=np.float64) for b in beads])


def _max_bead_speed(beads) -> float:
    import sapien.physx as physx

    v_max = 0.0
    for b in beads:
        comp = b.find_component_by_type(physx.PhysxRigidDynamicComponent)
        v_max = max(v_max, float(np.linalg.norm(np.asarray(comp.linear_velocity, dtype=np.float64))))
    return v_max


def settle(scene, beads, max_steps: int = SETTLE_MAX_STEPS) -> int:
    """Step until every bead is slower than SETTLE_TOL_SPEED for
    SETTLE_QUIET_CHECKS consecutive checks (every 25 steps), or the cap."""
    quiet = 0
    step = 0
    while step < max_steps:
        scene.step()
        step += 1
        if step % 25 == 0:
            quiet = quiet + 1 if _max_bead_speed(beads) < SETTLE_TOL_SPEED else 0
            if quiet >= SETTLE_QUIET_CHECKS:
                break
    return step


# ---------------------------------------------------------------------------
# Cell definitions


def iter_cells(kinds=("rest", "translate", "tilt"), seeds=SEEDS) -> list[tuple]:
    """All P2 cell ids in deterministic order.

    ("rest", bowl_id, n, r_mm, seed) — the full 4 x 3 x 3 x 3 grid;
    ("translate", bowl_id, accel, seed) and ("tilt", bowl_id, omega, seed)
    at the default cell N=60, r=6 mm.
    """
    cells: list[tuple] = []
    if "rest" in kinds:
        for bowl in BOWL_IDS:
            for n in N_BEADS:
                for r_mm in R_BEADS_MM:
                    for seed in seeds:
                        cells.append(("rest", bowl, int(n), float(r_mm), int(seed)))
    if "translate" in kinds:
        for bowl in BOWL_IDS:
            for a in ACCELS:
                for seed in seeds:
                    cells.append(("translate", bowl, float(a), int(seed)))
    if "tilt" in kinds:
        for bowl in BOWL_IDS:
            for w in OMEGAS:
                for seed in seeds:
                    cells.append(("tilt", bowl, float(w), int(seed)))
    return cells


def _cell_rng(cell: tuple) -> np.random.RandomState:
    """Per-cell deterministic stream (bead physics + spawn jitter)."""
    key = "|".join(str(c) for c in cell)
    seed = int.from_bytes(key.encode("utf-8"), "little") % (2**31 - 1)
    return np.random.RandomState(seed)


def _bowl_spawn_pose():
    import sapien

    return sapien.Pose([0.0, 0.0, TABLE_TOP], list(SPAWN_QUAT))


# Showcase cells that get keyframes (6 total for P2; P4 adds 2 more).
def _showcase_moments(cell: tuple) -> tuple[str, ...]:
    if cell == ("rest", 1, DEFAULT_N, DEFAULT_R_MM, 0):
        return ("start", "end")
    if cell == ("translate", 1, max(ACCELS), 0):
        return ("mid", "end")
    if cell == ("tilt", 1, max(OMEGAS), 0):
        return ("mid", "end")
    return ()


# ---------------------------------------------------------------------------
# Keyframes


def _look_at_pose(eye, target, up=(0.0, 0.0, 1.0)):
    """sapien.Pose at `eye` aimed at `target` (camera convention: +x forward,
    +y left, +z up)."""
    import sapien

    x_c = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    x_c = x_c / np.linalg.norm(x_c)
    up = np.asarray(up, dtype=np.float64)
    z_c = up - float(up @ x_c) * x_c
    z_c = z_c / np.linalg.norm(z_c)
    y_c = np.cross(z_c, x_c)
    rot = np.stack([x_c, y_c, z_c], axis=1)  # columns = camera basis in world
    return sapien.Pose(list(np.asarray(eye, dtype=np.float64)), _mat_to_quat_wxyz(rot))


def _make_camera(scene):
    """A scene camera looking at the bowl from a high front angle."""
    cam = scene.add_camera(name="probe_cam", width=640, height=480, fovy=0.9, near=0.01, far=10.0)
    cam.set_local_pose(_look_at_pose([0.0, -0.75, 1.35], [0.0, 0.0, TABLE_TOP + 0.03]))
    return cam


def _render_keyframe(scene, cam, path: Path) -> None:
    import imageio.v2 as imageio

    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")
    img = (np.clip(rgba[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)


def _scatter_keyframe(bead_world: np.ndarray, world_T_bowl: np.ndarray, geom: dict, path: Path, title: str) -> None:
    """Headless keyframe: top + side scatter of beads vs the bowl outline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(bead_world, dtype=np.float64).reshape(-1, 3)
    local = to_frame(pts, world_T_bowl)
    inner = float(geom["inner_radius"])
    rim = float(geom["rim_y"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].add_patch(plt.Circle((0, 0), inner, fill=False, color="k", lw=1.5))
    axes[0].scatter(local[:, 0], local[:, 2], s=8, c="tab:red")
    axes[0].set_title(f"{title} — top (actor xz)")
    axes[0].set_aspect("equal")
    axes[1].add_patch(plt.Rectangle((-inner, 0), 2 * inner, rim, fill=False, color="k", lw=1.5))
    axes[1].scatter(local[:, 0], local[:, 1], s=8, c="tab:red")
    axes[1].set_title(f"{title} — side (actor xy, y=up)")
    axes[1].set_aspect("equal")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _keyframe(cell, moment, scene, beads, bowl, geom, out_dir, visuals, cam) -> str | None:
    name = "_".join(str(c).replace(".", "p") for c in cell)
    path = Path(out_dir) / "keyframes" / f"p2_{name}_{moment}.png"
    if visuals and cam is not None:
        _render_keyframe(scene, cam, path)
    else:
        _scatter_keyframe(
            _bead_positions(beads),
            pose_to_mat(bowl.get_pose().p, bowl.get_pose().q),
            geom,
            path,
            title=f"{name} {moment}",
        )
    return str(path)


# ---------------------------------------------------------------------------
# Cell runners (sapien)


def _spawn_beads_in_bowl(scene, bowl, geom, n, r, rng, visuals):
    """Grid positions (bowl frame) -> world -> bead entities."""
    local = bead_grid_positions(
        n, r, float(geom["inner_radius"]), float(geom["floor_y"]), rng=rng,
        bands=geom.get("bands"),
    )
    pose = bowl.get_pose()
    world_T = pose_to_mat(pose.p, pose.q)
    world = local @ world_T[:3, :3].T + world_T[:3, 3]
    return add_beads(scene, world, r, rng, visuals=visuals), local


def run_rest_cell(cell, out_dir, visuals: bool = False, keyframes: bool = True) -> dict:
    """("rest", bowl_id, n, r_mm, seed): settle + containment + timing at N=60."""
    _, bowl_id, n, r_mm, seed = cell
    r = float(r_mm) / 1000.0
    rng = _cell_rng(cell)
    t0 = time.perf_counter()
    scene = build_scene(visuals=visuals)
    cam = _make_camera(scene) if visuals else None
    bowl = add_bowl(scene, bowl_id, _bowl_spawn_pose(), visuals=visuals)
    geom = bowl_geometry(bowl_id)
    beads, _ = _spawn_beads_in_bowl(scene, bowl, geom, int(n), r, rng, visuals)

    frames: list[str] = []
    moments = _showcase_moments(cell) if keyframes else ()
    if "start" in moments:
        frames.append(_keyframe(cell, "start", scene, beads, bowl, geom, out_dir, visuals, cam))
    settle_steps = settle(scene, beads)
    world_T = pose_to_mat(bowl.get_pose().p, bowl.get_pose().q)
    meas = containment_counts(_bead_positions(beads), world_T, geom, r)
    if "end" in moments:
        frames.append(_keyframe(cell, "end", scene, beads, bowl, geom, out_dir, visuals, cam))

    ms_per_tick = None
    if int(n) == DEFAULT_N:
        t1 = time.perf_counter()
        for _ in range(TIMING_TICKS * TICK_STEPS):
            scene.step()
        ms_per_tick = (time.perf_counter() - t1) / TIMING_TICKS * 1000.0

    row = {
        "cell": list(cell),
        "kind": "rest",
        "bowl_id": int(bowl_id),
        "n": int(n),
        "r_mm": float(r_mm),
        "seed": int(seed),
        "settle_steps": int(settle_steps),
        "end_max_speed": _max_bead_speed(beads),
        "rim_y": float(geom["rim_y"]),
        "inner_radius": float(geom["inner_radius"]),
        "floor_y": float(geom["floor_y"]),
        "ms_per_tick": ms_per_tick,
        "cpu": _cpu_model(),
        "wall_s": round(time.perf_counter() - t0, 2),
        "visuals": bool(visuals),
        "keyframes": [f for f in frames if f],
        **{k: v for k, v in meas.items() if k != "spill_xy"},
    }
    return row


def run_translate_cell(cell, out_dir, visuals: bool = False, keyframes: bool = True) -> dict:
    """("translate", bowl_id, accel, seed): kinematic trapezoid stroke, N=60 r=6."""
    import sapien
    import sapien.physx as physx

    _, bowl_id, accel, seed = cell
    n, r = DEFAULT_N, DEFAULT_R_MM / 1000.0
    rng = _cell_rng(cell)
    t0 = time.perf_counter()
    scene = build_scene(visuals=visuals)
    cam = _make_camera(scene) if visuals else None
    p0 = np.array([-STROKE / 2.0, 0.0, TABLE_TOP])
    # spawn the bowl AT the stroke start so the bead grid drops into it there
    # (spawning at the origin and jumping the kinematic target ejects beads)
    bowl = add_bowl(scene, bowl_id, sapien.Pose(p0, list(SPAWN_QUAT)), kinematic=True, visuals=visuals)
    comp = bowl.find_component_by_type(physx.PhysxRigidDynamicComponent)
    geom = bowl_geometry(bowl_id)
    beads, _ = _spawn_beads_in_bowl(scene, bowl, geom, n, r, rng, visuals)

    frames: list[str] = []
    moments = _showcase_moments(cell) if keyframes else ()

    # settle with the kinematic bowl pinned at x0
    comp.set_kinematic_target(sapien.Pose(p0, list(SPAWN_QUAT)))
    settle_steps = settle(scene, beads)

    prof = trapezoid_profile(STROKE, V_MAX, float(accel))
    traj_min_z = float("inf")
    mid_done = False
    t_step = 0.0
    n_steps = int(math.ceil(prof["t_total"] / PHYS_DT))
    for i in range(1, n_steps + 1):
        x = p0 + np.array([float(prof["x_fn"](i * PHYS_DT)), 0.0, 0.0])
        comp.set_kinematic_target(sapien.Pose(x, list(SPAWN_QUAT)))
        ta = time.perf_counter()
        scene.step()
        t_step += time.perf_counter() - ta
        traj_min_z = min(traj_min_z, float(_bead_positions(beads)[:, 2].min()))
        if not mid_done and i * PHYS_DT >= 0.5 * prof["t_total"]:
            if "mid" in moments:
                frames.append(_keyframe(cell, "mid", scene, beads, bowl, geom, out_dir, visuals, cam))
            mid_done = True
    # hold at the end pose for 1 s
    hold_steps = int(1.0 / PHYS_DT)
    for _ in range(hold_steps):
        comp.set_kinematic_target(sapien.Pose(p0 + [STROKE, 0.0, 0.0], list(SPAWN_QUAT)))
        ta = time.perf_counter()
        scene.step()
        t_step += time.perf_counter() - ta
        traj_min_z = min(traj_min_z, float(_bead_positions(beads)[:, 2].min()))
    # physics-only wall time per control tick with the bead pile fully active
    active_ms_per_tick = t_step / ((n_steps + hold_steps) / TICK_STEPS) * 1000.0

    world_T = pose_to_mat(bowl.get_pose().p, bowl.get_pose().q)
    meas = containment_counts(_bead_positions(beads), world_T, geom, r)
    if "end" in moments:
        frames.append(_keyframe(cell, "end", scene, beads, bowl, geom, out_dir, visuals, cam))
    return {
        "cell": list(cell),
        "kind": "translate",
        "bowl_id": int(bowl_id),
        "accel": float(accel),
        "seed": int(seed),
        "n": n,
        "r_mm": DEFAULT_R_MM,
        "settle_steps": int(settle_steps),
        "v_peak": prof["v_peak"],
        "t_total": prof["t_total"],
        "triangular": prof["triangular"],
        "traj_min_z": traj_min_z,
        "active_ms_per_tick": active_ms_per_tick,
        "cpu": _cpu_model(),
        "wall_s": round(time.perf_counter() - t0, 2),
        "visuals": bool(visuals),
        "keyframes": [f for f in frames if f],
        **{k: v for k, v in meas.items() if k != "spill_xy"},
    }


def run_tilt_cell(cell, out_dir, visuals: bool = False, keyframes: bool = True) -> dict:
    """("tilt", bowl_id, omega, seed): kinematic 110 deg tilt, N=60 r=6."""
    import sapien
    import sapien.physx as physx

    _, bowl_id, omega, seed = cell
    n, r = DEFAULT_N, DEFAULT_R_MM / 1000.0
    rng = _cell_rng(cell)
    t0 = time.perf_counter()
    scene = build_scene(visuals=visuals)
    cam = _make_camera(scene) if visuals else None
    bowl = add_bowl(scene, bowl_id, _bowl_spawn_pose(), kinematic=True, visuals=visuals)
    comp = bowl.find_component_by_type(physx.PhysxRigidDynamicComponent)
    geom = bowl_geometry(bowl_id)
    beads, _ = _spawn_beads_in_bowl(scene, bowl, geom, n, r, rng, visuals)

    frames: list[str] = []
    moments = _showcase_moments(cell) if keyframes else ()

    p_rest = np.array([0.0, 0.0, TABLE_TOP])
    comp.set_kinematic_target(sapien.Pose(p_rest, list(SPAWN_QUAT)))
    settle_steps = settle(scene, beads)

    prof = tilt_profile(math.radians(TILT_DEG), float(omega))
    # rotation centre: the bowl's interior-floor centre, lifted TILT_LIFT above
    # the table (pour pose); axis = world +y (perpendicular to the pour side).
    pivot = p_rest + np.array([0.0, 0.0, TILT_LIFT + geom["floor_y"]])
    q0 = quat_wxyz_to_mat(SPAWN_QUAT)
    floor_local = np.array([0.0, geom["floor_y"], 0.0])  # actor frame

    lift_steps = int(0.5 / PHYS_DT)
    for i in range(1, lift_steps + 1):
        z = p_rest + np.array([0.0, 0.0, TILT_LIFT * i / lift_steps])
        comp.set_kinematic_target(sapien.Pose(z, list(SPAWN_QUAT)))
        scene.step()

    traj_min_z = float("inf")
    pour_start_deg = None
    mid_done = False
    n_steps = int(math.ceil(prof["t_total"] / PHYS_DT))
    inside_prev = n
    for i in range(1, n_steps + 1):
        th = float(prof["theta_fn"](i * PHYS_DT))
        rot = _roty(th) @ q0  # world-frame rotation about the +y axis through the pivot
        pos = pivot - rot @ floor_local  # keep the interior-floor centre on the pivot
        q = _mat_to_quat_wxyz(rot)
        comp.set_kinematic_target(sapien.Pose(pos, q))
        scene.step()
        pts = _bead_positions(beads)
        traj_min_z = min(traj_min_z, float(pts[:, 2].min()))
        if pour_start_deg is None and i % 5 == 0:
            world_T = pose_to_mat(pos, q)
            inside_now = containment_counts(pts, world_T, geom, r)["inside"]
            if inside_now < inside_prev:
                pour_start_deg = math.degrees(th)
            inside_prev = min(inside_prev, inside_now)
        if not mid_done and th >= 0.5 * math.radians(TILT_DEG):
            if "mid" in moments:
                frames.append(_keyframe(cell, "mid", scene, beads, bowl, geom, out_dir, visuals, cam))
            mid_done = True
    # hold at full tilt for 1 s — keep watching for the pour start: at high
    # omega the beads ride the bowl through the whole rotation and only flow
    # out during the stop/hold, so pour_start would otherwise stay None
    th_hold = math.radians(TILT_DEG)
    for i in range(1, int(1.0 / PHYS_DT) + 1):
        comp.set_kinematic_target(sapien.Pose(pos, q))
        scene.step()
        pts = _bead_positions(beads)
        traj_min_z = min(traj_min_z, float(pts[:, 2].min()))
        if pour_start_deg is None and i % 5 == 0:
            world_T = pose_to_mat(pos, q)
            inside_now = containment_counts(pts, world_T, geom, r)["inside"]
            if inside_now < inside_prev:
                pour_start_deg = math.degrees(th_hold)
            inside_prev = min(inside_prev, inside_now)

    final_pts = _bead_positions(beads)
    world_T = pose_to_mat(bowl.get_pose().p, bowl.get_pose().q)
    meas = containment_counts(final_pts, world_T, geom, r)
    if "end" in moments:
        frames.append(_keyframe(cell, "end", scene, beads, bowl, geom, out_dir, visuals, cam))
    spill_xy = np.asarray(meas.pop("spill_xy"), dtype=np.float64).reshape(-1, 2)
    scatter = (
        float(np.linalg.norm(spill_xy - pivot[:2], axis=1).mean()) if len(spill_xy) else 0.0
    )
    return {
        "cell": list(cell),
        "kind": "tilt",
        "bowl_id": int(bowl_id),
        "omega": float(omega),
        "seed": int(seed),
        "n": n,
        "r_mm": DEFAULT_R_MM,
        "settle_steps": int(settle_steps),
        "t_total": prof["t_total"],
        "pour_start_deg": pour_start_deg,
        "spill_scatter_m": scatter,
        "traj_min_z": traj_min_z,
        "cpu": _cpu_model(),
        "wall_s": round(time.perf_counter() - t0, 2),
        "visuals": bool(visuals),
        "keyframes": [f for f in frames if f],
        **meas,
    }


def _roty(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _mat_to_quat_wxyz(rot: np.ndarray) -> list[float]:
    """3x3 rotation -> wxyz quaternion (Shepperd's method)."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 1e-12)) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


CELL_RUNNERS = {"rest": run_rest_cell, "translate": run_translate_cell, "tilt": run_tilt_cell}


# ---------------------------------------------------------------------------
# run / summarize


def _parse_shard(shard: str) -> tuple[int, int]:
    i, _, n = str(shard).partition("/")
    return int(i), max(1, int(n or 1))


def run(out_dir, cells=None, shard: str = "0/1", kinds=("rest", "translate", "tilt"),
        visuals: bool = False, keyframes: bool = True, **kw) -> dict:
    """Run the pending P2 cells of this shard; append rows to p2_rows.jsonl.

    cells: optional explicit subset of cell-id tuples. shard "i/n": cells
    whose index in iter_cells(kinds) is i mod n. Returns a small run report.
    """
    out_dir = Path(out_dir)
    rows_path = out_dir / ROWS_FILE
    rows = load_rows(rows_path)
    done = done_cells(rows)
    all_cells = iter_cells(kinds=tuple(kinds))
    if cells is not None:
        wanted = {tuple(c) for c in cells}
        all_cells = [c for c in all_cells if c in wanted]
    si, sn = _parse_shard(shard)
    my_cells = [c for idx, c in enumerate(all_cells) if idx % sn == si]
    pending = [c for c in my_cells if c not in done]

    ran = 0
    for cell in pending:
        row = CELL_RUNNERS[cell[0]](cell, out_dir, visuals=visuals, keyframes=keyframes)
        append_row(rows_path, row)
        ran += 1
        meas = f"inside={row['inside']} spill={row['spilled']} tunn={row['tunneled']}"
        extra = f" ms/tick={row['ms_per_tick']:.2f}" if row.get("ms_per_tick") is not None else ""
        print(f"[p2] {cell}: {meas}{extra} ({row['wall_s']}s)", flush=True)
    report = {"ran": ran, "skipped_done": len(my_cells) - len(pending),
              "shard": f"{si}/{sn}", "rows": str(rows_path)}
    print(f"[p2] run done: {report}", flush=True)
    return report


def _mean_spill_by(rows: list[dict], key: str, values: list[float]) -> list[float]:
    """Pooled mean out-of-bowl fraction (1 - inside/n) per swept value."""
    out = []
    for v in values:
        sel = [r for r in rows if float(r.get(key, -1)) == float(v)]
        out.append(
            float(np.mean([1.0 - r["inside"] / max(1, r["n"]) for r in sel])) if sel else float("nan")
        )
    return out


def _nondecreasing(xs: list[float]) -> bool:
    xs = [x for x in xs if not math.isnan(x)]
    return all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))


def summarize(out_dir) -> dict:
    """Aggregate p2_rows.jsonl (+ p4_t2_rows.jsonl) into gate metrics.

    Writes <out_dir>/p2_summary.json and returns the summary dict.
    """
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / ROWS_FILE)
    rest = [r for r in rows if r.get("kind") == "rest"]
    trans = [r for r in rows if r.get("kind") == "translate"]
    tilt = [r for r in rows if r.get("kind") == "tilt"]

    # rest containment per (bowl, n, r)
    rest_cells: dict[tuple, dict] = {}
    for r in rest:
        key = (r["bowl_id"], r["n"], r["r_mm"])
        agg = rest_cells.setdefault(
            key, {"inside_frac": [], "tunneled": 0, "spilled": 0, "off_table": 0, "seeds": 0}
        )
        agg["inside_frac"].append(r["inside_frac"])
        agg["tunneled"] += r["tunneled"]
        agg["spilled"] += r["spilled"]
        agg["off_table"] += r.get("off_table", 0)
        agg["seeds"] += 1
    rest_table = [
        {
            "bowl_id": k[0], "n": k[1], "r_mm": k[2],
            "mean_inside_frac": float(np.mean(v["inside_frac"])),
            "tunneled": v["tunneled"], "spilled": v["spilled"],
            "off_table": v["off_table"], "seeds": v["seeds"],
        }
        for k, v in sorted(rest_cells.items())
    ]
    default_full = all(
        abs(e["mean_inside_frac"] - 1.0) < 1e-9
        for e in rest_table
        if e["n"] == DEFAULT_N and e["r_mm"] == DEFAULT_R_MM
    )
    all_full = all(abs(e["mean_inside_frac"] - 1.0) < 1e-9 for e in rest_table)

    # spill monotonicity
    spill_vs_accel = _mean_spill_by(trans, "accel", list(ACCELS))
    spill_vs_omega = _mean_spill_by(tilt, "omega", list(OMEGAS))
    per_bowl_accel = {
        b: _mean_spill_by([r for r in trans if r["bowl_id"] == b], "accel", list(ACCELS))
        for b in BOWL_IDS
    }

    # timing at N=60: rest cells measure the settled (possibly sleeping) pile;
    # translate cells measure physics-only wall time with the pile fully active
    timed = [r["ms_per_tick"] for r in rest if r.get("ms_per_tick") is not None]
    active = [r["active_ms_per_tick"] for r in trans if r.get("active_ms_per_tick") is not None]
    timing = {
        "n_cells": len(timed),
        "mean_ms_per_tick": float(np.mean(timed)) if timed else None,
        "max_ms_per_tick": float(np.max(timed)) if timed else None,
        "active_mean_ms_per_tick": float(np.mean(active)) if active else None,
        "active_max_ms_per_tick": float(np.max(active)) if active else None,
        "cpu": next((r["cpu"] for r in rest if r.get("cpu")), _cpu_model()),
    }

    gates = {
        "rest_inside_default_cell": bool(default_full),
        "rest_inside_all_cells": bool(all_full),
        "zero_tunneling": all(r["tunneled"] == 0 for r in rows) and all(
            r.get("traj_min_z", TABLE_TOP) > TABLE_TOP - 0.5 * DEFAULT_R_MM / 1000.0
            for r in rows if "traj_min_z" in r
        ),
        "spill_increases_with_accel": bool(
            _nondecreasing(spill_vs_accel)
            and spill_vs_accel[-1] > spill_vs_accel[0] + 1e-9
        ),
        "spill_increases_with_omega": bool(
            _nondecreasing(spill_vs_omega)
            and spill_vs_omega[-1] > spill_vs_omega[0] + 1e-9
        ),
        "physics_le_20ms_per_tick": bool(timed)
        and float(np.max(timed)) <= GATE_MS_PER_TICK
        and (not active or float(np.max(active)) <= GATE_MS_PER_TICK),
    }
    summary = {
        "n_rows": len(rows),
        "rest_table": rest_table,
        "spill_vs_accel": dict(zip([str(a) for a in ACCELS], spill_vs_accel)),
        "spill_vs_omega": dict(zip([str(w) for w in OMEGAS], spill_vs_omega)),
        "spill_vs_accel_per_bowl": {str(b): v for b, v in per_bowl_accel.items()},
        "tilt_extra": {
            "pour_start_deg": {
                str(w): float(np.nanmean([r["pour_start_deg"] for r in tilt if r["omega"] == w and r.get("pour_start_deg") is not None]))
                for w in OMEGAS
                if any(r["omega"] == w and r.get("pour_start_deg") is not None for r in tilt)
            },
            "spill_scatter_m": {
                str(w): float(np.mean([r["spill_scatter_m"] for r in tilt if r["omega"] == w]))
                for w in OMEGAS
                if any(r["omega"] == w for r in tilt)
            },
        },
        "timing": timing,
        "gates": gates,
    }
    p4_rows_path = out_dir / P4_ROWS_FILE
    if p4_rows_path.exists():
        summary["p4"] = summarize_p4(out_dir, write=False)
    with open(out_dir / "p2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# P4 — T2-prototype layout visibility (rendering required -> GPU node)

P4_N_SEEDS = 20
P4_GATE_VIS = 0.3
P4_ZONE_COLORS = [((1.0, 0.0, 0.0), "red"), ((0.0, 0.0, 1.0), "blue")]


def _t2_layout(rng: np.random.RandomState, table_top: float) -> dict:
    """One T2-prototype layout (PLAN v3 §7 T2 geometry, host pose ranges)."""
    side = float(rng.choice([-1.0, 1.0]))
    bowl_xy = np.array([side * rng.uniform(0.20, 0.28), rng.uniform(-0.10, 0.05)])
    bowl_id = int(rng.choice(list(BOWL_IDS)))
    target_id = int(rng.choice([b for b in BOWL_IDS if b != bowl_id]))
    target_xy = np.array([-side * 0.05 + rng.uniform(-0.03, 0.03), rng.uniform(-0.15, -0.10)])
    zones = []
    for rgb, name in P4_ZONE_COLORS:
        size = np.array([rng.uniform(0.10, 0.16), rng.uniform(0.08, 0.14)])
        for _ in range(50):
            off = np.array([rng.uniform(-0.16, 0.16), rng.uniform(0.05, 0.18)])
            zxy = target_xy + off
            if abs(zxy[0]) > 0.55 - size[0] / 2 or abs(zxy[1]) > 0.30 - size[1] / 2:
                continue
            if np.linalg.norm(zxy - bowl_xy) < 0.5 * (size[0] + 0.12):
                continue
            if np.linalg.norm(zxy - target_xy) < 0.5 * (size[0] + 0.10):
                continue
            if zones and np.linalg.norm(zxy - zones[0]["xy"]) < 0.5 * (size[0] + zones[0]["size"][0]) + 0.02:
                continue
            break
        else:
            zxy = np.clip(target_xy + np.array([0.14, 0.12]), [-0.5, -0.25], [0.5, 0.25])
        zones.append({"xy": zxy, "size": size, "rgb": rgb, "name": name})
    return {
        "bowl_xy": bowl_xy, "bowl_id": bowl_id, "target_xy": target_xy,
        "target_id": target_id, "zones": zones, "table_top": table_top,
    }


def _build_t2_scene(env, layout: dict, rng: np.random.RandomState) -> dict:
    """(Re)build the T2-prototype objects inside the Env's smoke_items scene.

    First call removes the smoke task's own actors; every call removes the
    previous probe layout. Returns ItemSpecs (bowl_src, target, zone_red,
    zone_blue) + zone Regions + the spawned entities (for the next cleanup).
    """
    import sapien
    from envs._item_utils import zone_mat  # symlinked robotwin_tasks helper
    from envs.utils.create_actor import create_actor

    from ..items import entity_of
    from ..regions import zone_region

    task = env.task
    scene = task.scene
    if not getattr(task, "_p4_smoke_cleared", False):
        for attr in ("box_0", "box_1", "zone"):
            ent = getattr(task, attr, None)
            if ent is not None:
                scene.remove_entity(entity_of(ent))
        task._p4_smoke_cleared = True
    for ent in getattr(task, "_p4_entities", []):
        scene.remove_entity(ent)
    task._p4_entities = []

    z = float(layout["table_top"])
    bowl = create_actor(
        scene,
        pose=sapien.Pose([float(layout["bowl_xy"][0]), float(layout["bowl_xy"][1]), z], list(SPAWN_QUAT)),
        modelname=BOWL_MODEL,
        model_id=int(layout["bowl_id"]),
        convex=True,
    )
    target = create_actor(
        scene,
        pose=sapien.Pose([float(layout["target_xy"][0]), float(layout["target_xy"][1]), z], list(SPAWN_QUAT)),
        modelname=BOWL_MODEL,
        model_id=int(layout["target_id"]),
        is_static=True,
        convex=True,
    )
    geom = bowl_geometry(int(layout["bowl_id"]))
    beads, _ = _spawn_beads_in_bowl(scene, bowl, geom, DEFAULT_N, DEFAULT_R_MM / 1000.0, rng, visuals=True)
    mats = []
    for zdef in layout["zones"]:
        mat = zone_mat(
            scene,
            sapien.Pose([float(zdef["xy"][0]), float(zdef["xy"][1]), z + 0.001]),
            size_xy=zdef["size"],
            color=zdef["rgb"],
            name=f"zone_{zdef['name']}",
        )
        mats.append((mat, zdef))
    for _ in range(100):  # let the bowl and beads rest before the t=0 read
        scene.step()

    spawned = [entity_of(bowl), entity_of(target), *beads, *(m for m, _ in mats)]
    task._p4_entities = spawned

    items = [
        ItemSpec(
            item_id="bowl_src", entity=bowl, category="bowl", role="task_object",
            half_extents=geom["half_extents"], obb_center_local=geom["obb_center_local"],
            up_axis_local=[0.0, 1.0, 0.0],
        ),
        ItemSpec(
            item_id="target", entity=target, category="bowl", role="support",
            half_extents=bowl_geometry(int(layout["target_id"]))["half_extents"],
            obb_center_local=bowl_geometry(int(layout["target_id"]))["obb_center_local"],
            up_axis_local=[0.0, 1.0, 0.0],
        ),
    ]
    regions = []
    for mat, zdef in mats:
        items.append(
            ItemSpec(
                item_id=f"zone_{zdef['name']}", entity=mat, category="zone",
                color=zdef["name"], role="support",
                half_extents=[zdef["size"][0] / 2, zdef["size"][1] / 2, 0.001],
            )
        )
        regions.append(zone_region(mat, zdef["name"]))
    return {"items": items, "regions": regions, "entities": spawned}


def run_p4(out_dir, cells=None, shard: str = "0/1", embodiment: str = "ur5-wsg",
           n_seeds: int = P4_N_SEEDS, keyframes: bool = True, **kw) -> dict:
    """20 T2-prototype layouts in ONE Env; visibility per candidate at t=0.

    The Env (smoke_items) is built once for the camera rig; layouts are
    re-rolled by removing and re-adding the probe entities. Rows ->
    p4_t2_rows.jsonl, cell key ("p4_t2", seed). out_dir is resolved to an
    absolute path up front: Env creation chdirs to the RoboTwin root.
    """
    from .._bootstrap import ensure_main_path

    ensure_main_path()
    from main.envs.env import Env  # noqa: E402
    from main.train.collect import ARM_DISTANCE  # noqa: E402

    from .. import masks

    out_dir = Path(out_dir).resolve()
    rows_path = out_dir / P4_ROWS_FILE
    rows = load_rows(rows_path)
    done = done_cells(rows)
    si, sn = _parse_shard(shard)
    seeds = [s for s in range(int(n_seeds)) if s % sn == si]
    pending = [s for s in seeds if ("p4_t2", s) not in done]

    env = Env("smoke_items", embodiment, 0,
              arm_distance=ARM_DISTANCE.get(embodiment, 0.6), control_freq=25.0)
    ran = 0
    try:
        for seed in pending:
            rng = np.random.RandomState(10_000 + seed)
            layout = _t2_layout(rng, 0.74 + float(getattr(env.task, "table_z_bias", 0.0)))
            scene_objs = _build_t2_scene(env, layout, rng)
            items, regions = scene_objs["items"], scene_objs["regions"]
            visibility, token_masks = masks.item_masks(env.task, items, regions, None)
            names = [s.item_id for s in items]
            row = {
                "cell": ["p4_t2", seed],
                "seed": seed,
                "embodiment": embodiment,
                "items": names,
                "cameras": list(masks.CAMERA_NAMES),
                "visibility": {nm: [float(v) for v in visibility[i]] for i, nm in enumerate(names)},
                "max_visibility": {nm: float(visibility[i].max()) for i, nm in enumerate(names)},
                "token_mask_nonzero": {
                    nm: int((token_masks[i] > 0).sum()) for i, nm in enumerate(names)
                },
                "layout": {
                    "bowl_id": layout["bowl_id"], "target_id": layout["target_id"],
                    "bowl_xy": [float(v) for v in layout["bowl_xy"]],
                    "target_xy": [float(v) for v in layout["target_xy"]],
                },
            }
            append_row(rows_path, row)
            ran += 1
            if keyframes and seed < 2:
                _p4_keyframe(env, out_dir, seed, scene_objs)
            mx = ", ".join(f"{nm}={row['max_visibility'][nm]:.2f}" for nm in names)
            print(f"[p4] seed={seed}: {mx}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
    report = {"ran": ran, "rows": str(rows_path)}
    print(f"[p4] run done: {report}", flush=True)
    return report


def _p4_keyframe(env, out_dir, seed: int, scene_objs) -> None:
    """Head-camera RGB + the item segmentation overlay for one layout."""
    import imageio.v2 as imageio

    from ..items import entity_of

    task = env.task
    task._update_render()
    head = task.cameras.static_camera_list[task.cameras.head_camera_id]
    head.take_picture()
    rgb = (np.clip(head.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
    seg = head.get_picture("Segmentation")[..., 1]
    ids = {int(entity_of(s).per_scene_id): s.item_id for s in scene_objs["items"]}
    overlay = np.zeros((*seg.shape, 3), dtype=np.uint8)
    palette = [(255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 255, 60), (255, 60, 255)]
    for k, (pid, nm) in enumerate(ids.items()):
        overlay[seg == pid] = palette[k % len(palette)]
    blend = (0.55 * rgb + 0.45 * overlay).astype(np.uint8)
    path = Path(out_dir) / "keyframes" / f"p4_t2_seed{seed}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), np.concatenate([rgb, blend], axis=1))


def summarize_p4(out_dir, write: bool = True) -> dict:
    """P4 gate: per candidate kind, fraction of seeds with max visibility >= 0.3."""
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / P4_ROWS_FILE)
    kinds = ("bowl_src", "target", "zone_red", "zone_blue")
    per_kind: dict[str, dict] = {}
    for kind in kinds:
        vals = [float(r["max_visibility"].get(kind, 0.0)) for r in rows]
        ok = [v >= P4_GATE_VIS for v in vals]
        per_kind[kind] = {
            "n_seeds": len(vals),
            "frac_ge_0.3": (sum(ok) / len(ok)) if ok else None,
            "min_max_visibility": min(vals) if vals else None,
            "mean_max_visibility": float(np.mean(vals)) if vals else None,
        }
    gate = bool(per_kind) and all(
        (v["frac_ge_0.3"] is not None and v["frac_ge_0.3"] >= 0.95) for v in per_kind.values()
    )
    summary = {"n_rows": len(rows), "per_kind": per_kind, "gate_p4": gate}
    if write:
        with open(out_dir / "p4_t2_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--kinds", default="rest,translate,tilt")
    ap.add_argument("--visuals", action="store_true")
    ap.add_argument("--no-keyframes", action="store_true")
    ap.add_argument("--p4", action="store_true", help="run the P4 T2 visibility probe instead")
    ap.add_argument("--p4-seeds", type=int, default=P4_N_SEEDS)
    ap.add_argument("--embodiment", default="ur5-wsg")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2))
    elif args.p4:
        run_p4(args.out, shard=args.shard, embodiment=args.embodiment,
               n_seeds=args.p4_seeds, keyframes=not args.no_keyframes)
    else:
        run(args.out, shard=args.shard, kinds=tuple(args.kinds.split(",")),
            visuals=args.visuals, keyframes=not args.no_keyframes)
