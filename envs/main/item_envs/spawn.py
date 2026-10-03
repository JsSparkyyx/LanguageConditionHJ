"""T1 selective_avoid: multi-object spawner on the CrossSafe host tasks (PLAN v3 §7 T1).

Generalizes CEHJ's ``main/envs/obstacle.py::choose_and_spawn`` from one static
obstacle to K dynamic candidates WITHOUT modifying obstacle.py — its corridor
geometry (``longest_corridor``), keepaway construction (``keepaways_from_task``
/ ``keepaways_from_robot``), and placement primitives (``_in_table`` /
``_line_dist`` / ``_table_grid``) are reused verbatim; the near-path class,
pairwise separations, dynamic spawning, and the settle-reject loop are new.
The stand-quat/half-extents machinery is NOT reused (layout fix 4): the
model_data json center/extents proved unreliable per id, so ``_geometry_of``
measures the collision mesh AABB directly (``_mesh_aabb``).

Layout (all draws from the CALLER's rng — ``np.random.RandomState(seed +
4241)`` per PLAN v3 §2; this module never reseeds):

- K ~ U{objects.k} (T1 yaml: 3-5). Per-object placement class:
  **on-path** (corridor t ~ U(task_cfg.corridor_t = [0.22, 0.48]), at most
  ``max_on_path``), **near-path** (lateral offset ~ U(``near_path_offset``) =
  2-6 cm from the corridor, corridor parameter t ~ U(``near_path_t``) =
  [0.05, 0.95] — the PLAN pins t to [0.22, 0.48] only for ON-path; near-path
  rides the full corridor so the on-path window does not congest),
  **off-path** (>= ``off_path_min_dist`` = 10 cm from the corridor line).
  At least ``min_on_or_near_path`` (= 2) objects land on- or near-path. The on-path target count comes from
  ``objects.on_path_count`` when the caller injects a sampled value (the
  Phase-4 runner's risk sampler owns that knob), else it is sampled from
  ``risk.task.on_path_count`` here, else U{1..max_on_path}.
- Every object keeps >= ``min_separation`` (3 cm, surface-to-surface) from the
  other objects. Clearance from the host's keepaways (pick/target actors, arm
  bases) uses CEHJ's own ``KEEPAWAY_GAP`` = 2 cm (``keepaway_gap`` knob) —
  choose_and_spawn parity. PLAN v3 §7 T1 reads "3 cm from each other and from
  the keepaways"; applying 3 cm against the keepaway CIRCLES makes on-path
  placement infeasible on the shortest host corridors (L ≈ 0.33 m: the
  pick/target keepaways + 3 cm leave a sub-cm window), so the 3 cm is honored
  object-object and the CEHJ gap keepaway-side (documented deviation).
- **Layout fix (documented, PLAN §6 "layout change")**: on/near-path slots
  draw from the size-capped sub-pool (``on_path_max_r_xy`` /
  ``near_path_max_r_xy``, default 3.5 cm world xy-radius) via rejection
  sampling; off-path slots draw the full pool. On place_container_plate's
  shortest corridors (L ≈ 0.33 m, plate keepaway 0.115 m) an exhaustive
  grid search shows on+near co-placement is feasible ONLY for r_xy <= ~3 cm
  objects — an uncapped pool produces ~0% layouts there.
- **Layout fix 2 (documented, PLAN §6 "layout change" — adaptive placement
  + quota degradation, after the P1 sweep showed 408/500 object episodes
  with k=0)**: three cooperating mechanisms, all pure-additive over a
  byte-identical first attempt (feasible hosts see no behavior change):
  1. on_path placements whose t0 + 9-point scan of ``corridor_t`` fails get
     a second, deterministic 25-point scan over ``ON_PATH_FALLBACK_T`` =
     [0.10, 0.75] (flagged per object via ``Placement.t_fallback``);
  2. an object whose drawn class cannot be placed retries the SAME draw
     down the chain on_path -> near_path -> off_path, where the step into
     off_path is allowed only when the ``min_on_or_near_path`` quota is
     already met by the final classes placed so far (a degraded-to-near
     object counts); the Placement records the FINAL class and
     ``degraded_from`` (None when placed as drawn);
  3. when a slot's chain is exhausted but >= max(2, quota) objects are
     already placed with the on/near quota met, the layout stops and
     returns the partial placements (``k_dropped`` slots unplaced) — a
     2-object layout with quota met keeps the "referent matters" property,
     an all-None layout wastes the episode.
- **Layout fix 3 (documented, P4-T1 finding F5 — off_path visibility band)**:
  the P1 rerun's t=0 visibility gate measured per-item max-camera visible
  fraction 0.642 for off_path (< 0.95 gate): far-corner table cells sit
  outside every camera frustum. When ``objects.off_path_visible_only`` is
  set (T1 yaml), the OFF-path grid candidates are restricted to an
  empirically measured visibility band — ``configs/visibility_band_<emb>.npz``
  written by ``scripts/item_envs/visibility_band.py`` (a 2 cm grid sweep of a
  kinematic reference cylinder through ``masks.item_masks``; band = largest
  all-covered axis-aligned rect, vis >= 0.3, inset one cell). Resolution
  order: the env's own embodiment band, else the OTHER embodiment's
  (``report["vis_band"] == "shared-fallback"``), else no restriction (None).
  With the flag off or the npz missing the off_path branch is byte-identical
  to before (``vis_band=None`` skips the filter).
- **Layout fix 4 (documented, P4 acceptance-wave finding — settle-time
  toppling)**: the P4 sweep logged settle rejects with 25-131 mm slides and
  35-149 deg tilts (franka >> ur5), after which MAX_LAYOUTS kept an unsettled
  layout with negative t=0 margins. Offline audit (trimesh on the collision
  glbs, no GPU): (a) the model_data json center/extents are unreliable per id
  — 086_woodenblock id 2's json AABB bottom sits 24 mm high (drop-topple),
  cup/mug/can/bottle halves are inflated up to 35 mm (conservative, but it
  starved the size-capped draws) — so ``_geometry_of`` MEASURES the collision
  mesh AABB (``_mesh_aabb``, cached per (asset, id)) and the z-pack then
  rests the world AABB BOTTOM at ``table_z + 0.002`` for ANY asset origin
  convention (origin-center, origin-bottom, …); (b) the dominant mechanism
  is not z at all: franka's HOME pose rests the hand/fingers at z 0.85-0.98 m
  over (x ±0.3, y −0.2) — INSIDE the spawn region — while the base keepaway
  circles (r=0.18 at the bases, y=−0.65) never bind on the table, so
  candidates >= ~11 cm tall spawn interpenetrating the fingers and PhysX
  ejects them during the 250-step settle (ur5's folded home clears the
  tallest pool object by >= 7 cm). Fix for (b): ``_arm_link_keepaways`` adds
  one keepaway circle per robot link overhanging the table (default r 8 cm,
  link z < table + 30 cm; ``objects.arm_link_keepaways: false`` opts out,
  ``arm_link_keepaway_r`` / ``arm_link_z_max`` retune) — every placement
  class already routes through ``_clearances_ok``, so the circles bind
  on/near/off placements uniformly (``report["arm_link_keepaways"]`` counts
  them).
- **Layout fix 5 (documented, P4-T1 finding F5b — near/on_path visibility
  coverage)**: the F5 band rerun still measured near_path 0.947 / on_path
  0.948 max-camera visible fraction (< 0.95 gate) — the band RECT fix only
  restricted off_path, and single cells inside the rect can still be
  occluded (robot body, frustum notches). When
  ``objects.near_path_visible_only`` / ``objects.on_path_visible_only``
  are set (T1 yaml), near/on-path candidates PREFER the npz's measured
  per-cell ``covered`` mask (vis_max >= 0.3 at that cell —
  ``_vis_covered_for`` / ``_covered_at``, nearest-cell lookup), NOT the
  inset band rect off_path uses. The preference runs INSIDE the original
  candidate sequence: near_path keeps its single 40-draw loop and takes
  the first ok()+covered draw; when no covered cell appears, the FIRST
  ok() draw is kept — byte-identical to the pre-F5b placement — and
  flagged ``vis_fallback=True``. on_path scans its usual t sequence (t0 +
  9-point base window, then the 25-point fallback window) for the first
  covered ok() — zero extra rng draws — else takes the original first-ok
  hit, also flagged. (An earlier two-loop near_path version that re-rolled
  the fallback from a shifted stream was A/B-measured on ur5 to REGRESS
  near visibility 0.959 -> 0.897: it discarded good first-ok placements,
  and the empty-table probe sweep cannot see occlusion by sibling items —
  tall bottles/cans dominated the failures. The fused pass keeps the
  covered preference where it is free and never trades a feasible
  placement for a mask cell.) Resolution order and labels mirror fix 3
  (own embodiment npz, else ``shared-fallback``, else None); the counters
  ``report["near_path_vis_fallbacks"]`` / ``report["on_path_vis_fallbacks"]``
  describe the FINAL kept layout, and ``report["vis_cov"]`` labels the
  mask applied. With both flags off or the npz missing every branch is
  skipped (``vis_cov=None``) and the layout is byte-identical to before;
  off_path is untouched by this fix.
- Objects are DYNAMIC with masses sampled uniformly from their objects.yaml
  category ``mass_range`` (set via ``physics.set_mass_props``) —
  ``objects.static_objects: true`` gives the CrossSafe parity layout.
- After spawning, the scene settles 250 physics steps; the layout is rejected
  and resampled when any object moved > 5 mm or tilted > 3 deg
  (``report["spawn_rejections"]`` counts every resample; ``layout_retries``
  vs ``settle_rejections`` break down why). After ``MAX_LAYOUTS`` attempts the
  last layout is kept and ``max_layouts_hit`` is set — a degraded layout never
  crashes a sweep.
- ``update_curobo_world`` is NOT called here (the caller decides; the probes
  refresh the table-only world so the nominal stays unaware of the objects).

The reference corridor is the host's LONGEST pick->place corridor
(``longest_corridor``) — the same corridor CrossSafe's single obstacle uses.
``stack_blocks_two``'s ``obstacle_bbox_gap`` keepaway padding is applied with
the same semantics as choose_and_spawn (drawn from the passed rng).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.obstacle / main.envs.tasks live in the sibling CEHJ copy

from main.envs import obstacle as ob  # noqa: E402  (sapien-free module import)
from main.envs.tasks import TASK_SPECS, iter_corridors, longest_corridor, resolve_arm  # noqa: E402

from . import physics  # noqa: E402
from .items import ItemSpec, entity_of  # noqa: E402
from .risk import _sample_knob  # noqa: E402

PLACE_CLASSES = ("on_path", "near_path", "off_path")
MAX_LAYOUTS = 40  # pure-numpy redraws are ~ms; hostile corridors need the retries
SETTLE_STEPS = 250
MOVE_TOL_M = 0.005
TILT_TOL_RAD = float(np.deg2rad(3.0))
NEAR_PATH_TRIES = 40
DRAW_CAP_TRIES = 24
ON_PATH_FALLBACK_T = (0.10, 0.75)  # fix 2.1: second scan window (corridor t)
ON_PATH_FALLBACK_N = 25  # fix 2.1: grid points in the fallback scan
# fix 2.2: per drawn class, the order placement is retried in when a class
# fails (same draw/geometry); stepping INTO off_path is quota-gated
_DEGRADE_CHAIN = {
    "on_path": ("on_path", "near_path", "off_path"),
    "near_path": ("near_path", "off_path"),
    "off_path": ("off_path",),
}
_FALLBACK_COLOR = (0.55, 0.32, 0.12)  # spawn_obstacle's box-fallback color


@dataclass
class _ObjectDraw:
    """One sampled pool object (asset + instance parameters)."""

    asset: str  # model folder ("068_boxdrink") or prim name ("prim_box")
    category: str
    model_id: int | None = None  # mesh assets only
    color: str | None = None  # prims only
    size: np.ndarray | None = None  # prims only: box [sx, sy, sz] / cylinder [r, r, h] full extents


CYL_STAND_QUAT = (0.7071067811865476, 0.0, -0.7071067811865476, 0.0)  # wxyz, R_y(-90deg): local x -> world z


@dataclass
class _Geometry:
    """OBB + placement geometry for one object draw."""

    center_local: np.ndarray  # [3] OBB center, item frame
    half: np.ndarray  # [3] OBB half extents, item frame
    world_half: np.ndarray  # [3] world-axis half extents at spawn orientation
    up_local: np.ndarray  # [3] body axis that points at world +z at spawn
    quat: tuple  # spawn orientation (wxyz)


@dataclass
class Placement:
    """One fully placed object (layout-plan output; sapien-free)."""

    item_id: str
    cls: str  # FINAL class: "on_path" | "near_path" | "off_path"
    draw: _ObjectDraw
    geo: _Geometry
    xyz: np.ndarray  # [3] spawn reference point (OBB center lands here; prims: pose origin)
    corridor_t: float | None = None
    lateral: float | None = None  # signed lateral offset (near_path)
    t_fallback: bool = False  # on_path t came from the ON_PATH_FALLBACK_T window
    degraded_from: str | None = None  # drawn class when cls was degraded; None = as drawn
    vis_fallback: bool = False  # fix 5 (F5b): covered cell infeasible -> unrestricted placement

    def report_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "cls": self.cls,
            "asset": self.draw.asset,
            "model_id": self.draw.model_id,
            "category": self.draw.category,
            "color": self.draw.color,
            "xyz": [float(v) for v in np.asarray(self.xyz).reshape(3)],
            "corridor_t": None if self.corridor_t is None else float(self.corridor_t),
            "lateral": None if self.lateral is None else float(self.lateral),
            "t_fallback": bool(self.t_fallback),
            "degraded_from": self.degraded_from,
            "vis_fallback": bool(self.vis_fallback),
        }


# ---------------------------------------------------------------------------
# object pool + geometry
# ---------------------------------------------------------------------------


def t1_pool_entries(objects_cfg: dict) -> list[dict]:
    """The T1 candidate pool as entry dicts (objects.yaml assets + t1_pool)."""
    assets = objects_cfg.get("assets", {}) or {}
    names = objects_cfg.get("t1_pool") or []
    out = []
    for name in names:
        entry = assets.get(str(name))
        if entry is None:
            raise KeyError(f"t1_pool asset {name!r} missing from objects.yaml assets")
        out.append({"asset": str(name), **dict(entry)})
    return out


def _draw_object(pool: list[dict], rng: np.random.RandomState, train_colors: list[str],
                 max_r_xy: float | None = None, geometry_fn=None) -> _ObjectDraw | None:
    """Sample one pool object: asset uniform, then id/color/size in a fixed draw order.

    max_r_xy (layout fix, see module docstring): reject draws whose world
    xy-radius exceeds the cap (<= DRAW_CAP_TRIES tries), returning None when
    the cap cannot be met — the layout attempt then fails and resamples. The
    radius comes from ``geometry_fn(draw).world_half`` (fix 4: the measured
    collision AABB — the json metadata inflated cup/mug radii up to 35 mm and
    starved the capped draws); ``_layout_plan`` passes its own geometry_fn so
    the test seam covers capping too. Prims use the drawn size, as before.
    """
    tries = DRAW_CAP_TRIES if max_r_xy is not None else 1
    for _ in range(tries):
        draw = _draw_one(pool, rng, train_colors)
        if max_r_xy is None:
            return draw
        if draw.asset.startswith("prim_"):
            half = np.asarray(draw.size, dtype=np.float64) * 0.5
        else:
            half = (geometry_fn or _geometry_of)(draw).world_half
        if float(np.max(half[:2])) <= max_r_xy:
            return draw
    return None


def _draw_one(pool: list[dict], rng: np.random.RandomState, train_colors: list[str]) -> _ObjectDraw:
    """Single uncapped draw (the original fixed-order sampler)."""
    entry = pool[int(rng.randint(len(pool)))]
    asset = entry["asset"]
    if asset.startswith("prim_"):
        color = str(train_colors[int(rng.randint(len(train_colors)))]) if train_colors else None
        if asset == "prim_box":
            sx_lo, sx_hi = (float(v) for v in entry.get("side_xy", [0.04, 0.08]))
            h_lo, h_hi = (float(v) for v in entry.get("height", [0.05, 0.15]))
            size = np.array(
                [rng.uniform(sx_lo, sx_hi), rng.uniform(sx_lo, sx_hi), rng.uniform(h_lo, h_hi)],
                dtype=np.float64,
            )
        else:  # prim_tall_cylinder
            r_lo, r_hi = (float(v) for v in entry.get("radius", [0.02, 0.035]))
            h_lo, h_hi = (float(v) for v in entry.get("height", [0.10, 0.18]))
            r = rng.uniform(r_lo, r_hi)
            size = np.array([2.0 * r, 2.0 * r, rng.uniform(h_lo, h_hi)], dtype=np.float64)
        return _ObjectDraw(asset=asset, category=str(entry["category"]), color=color, size=size)
    ids = list(entry.get("ids") or [0])
    model_id = int(ids[int(rng.randint(len(ids)))])
    return _ObjectDraw(asset=asset, category=str(entry["category"]), model_id=model_id)


# ---------------------------------------------------------------------------
# collision-mesh AABB measurement (fix 4; pure numpy + lazy trimesh)
# ---------------------------------------------------------------------------

_MESH_AABB_CACHE: dict = {}
# obstacle.py::_stand_R_quat's up==0 rotation (inline there, not a constant)
_UP0_R = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
_UP0_QUAT = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)


def _objects_roots() -> list[Path]:
    """RoboTwin assets/objects candidates (mirrors main/envs/env.py's root logic)."""
    roots = [Path("assets/objects")]  # runtime: Env prepare() chdirs to the RoboTwin root
    env_root = os.environ.get("ROBOTWIN_ROOT")
    if env_root:
        roots.append(Path(env_root) / "assets" / "objects")
    repo = Path(__file__).resolve().parents[3]  # <repo>/envs/main/item_envs/spawn.py
    roots.append(repo / "RoboTwin" / "assets" / "objects")
    roots.append(repo.parent / "RoboTwin" / "assets" / "objects")
    return roots


def _load_model_cfg_local(asset: str, model_id: int) -> dict:
    """model_data<id>.json (or model_data.json) from the first resolvable root."""
    import json

    for root in _objects_roots():
        folder = root / str(asset)
        for name in (f"model_data{int(model_id)}.json", "model_data.json"):
            path = folder / name
            if path.is_file():
                return json.loads(path.read_text())
    return {}


def _mesh_aabb(asset: str, model_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Measured collision-mesh AABB (lo, hi) in the actor frame, json scale applied.

    Loads the SAME file create_actor collides with (collision/base<id>.glb ->
    base<id>.glb -> textured<id>.obj; scale from the model_data json, which
    overrides create_actor's argument). The json center/extents metadata is
    NOT trustworthy (P4 audit: 086_woodenblock id 2's json AABB bottom is
    24 mm off; cup/mug halves inflated up to 35 mm), so placement geometry is
    measured, not read. Cached per (asset, id); None when the mesh cannot be
    loaded — callers fall back to the json metadata.
    """
    key = (str(asset), int(model_id or 0))
    if key in _MESH_AABB_CACHE:
        return _MESH_AABB_CACHE[key]
    out = None
    cfg = _load_model_cfg_local(*key)
    scale = cfg.get("scale")
    try:
        import trimesh  # lazy: keeps the module import sapien-free

        for root in _objects_roots():
            folder = root / key[0]
            if not folder.is_dir():
                continue
            cand = folder / "collision" / f"base{key[1]}.glb"
            if not cand.is_file():
                cand = folder / f"base{key[1]}.glb"
            if not cand.is_file():
                cand = folder / f"textured{key[1]}.obj"
            if not cand.is_file():
                continue
            mesh = trimesh.load(str(cand), force="scene")
            if hasattr(mesh, "dump"):
                mesh = mesh.dump(concatenate=True)
            s = np.asarray(scale if scale is not None else [1.0, 1.0, 1.0],
                           dtype=np.float64).reshape(-1)
            s = np.full(3, abs(float(s[0]))) if s.size == 1 else np.abs(np.resize(s, 3))
            verts = np.asarray(mesh.vertices, dtype=np.float64) * s
            if verts.size:
                out = (verts.min(axis=0), verts.max(axis=0))
            break
    except Exception as exc:
        print(f"[spawn] mesh AABB load failed for {key[0]} id {key[1]}: {exc}")
    _MESH_AABB_CACHE[key] = out
    return out


def _stand_R_quat_for(asset: str, center: np.ndarray, half: np.ndarray):
    """obstacle._stand_R_quat with the up-axis decided on GIVEN center/half.

    Same rules (presets pin the axis, else the dominant |center| axis when
    >= 1e-3, else the longest half axis) but evaluated on the measured AABB
    instead of the json globals, so the orientation matches the geometry the
    OBB is computed from.
    """
    if asset in ob.OBSTACLE_UP_AXIS:
        up = int(ob.OBSTACLE_UP_AXIS[asset])
    elif float(np.max(np.abs(center))) >= 1e-3:
        up = int(np.argmax(np.abs(center)))
    else:
        up = int(np.argmax(half))
    if up == 1:
        return ob._STAND_R, ob.OBSTACLE_STAND_QUAT
    if up == 0:
        return _UP0_R, _UP0_QUAT
    return np.eye(3, dtype=np.float64), (1.0, 0.0, 0.0, 0.0)


def _geometry_of(draw: _ObjectDraw) -> _Geometry:
    """OBB/stand geometry for a draw, from the MEASURED collision mesh (fix 4).

    Prim path unchanged (create_box/create_cylinder are center-origin). Mesh
    path: center/half are the measured collision-mesh AABB (``_mesh_aabb``),
    the stand rotation is chosen on those measured values
    (``_stand_R_quat_for``), and the json metadata is only a fallback when
    the mesh cannot be loaded. With the z-pack ``table_z + world_half[2] +
    0.002`` and ``_spawn_one``'s center compensation, the world AABB BOTTOM
    then rests at ``table_z + 0.002`` for ANY asset origin convention
    (origin-center, origin-bottom, …) — this function no longer calls
    ``ob.set_obstacle_model`` (no obstacle.py global state).
    """
    if draw.asset == "prim_tall_cylinder":
        # SAPIEN's PhysxCollisionShapeCylinder runs along its LOCAL x axis: an
        # identity pose spawns the "tall" cylinder lying down, and it rolls
        # (diag job 22636182: push_zone bystander cylinders rolled ~0.37 m off
        # the table during setup). Stand it up with CYL_STAND_QUAT (local x ->
        # world z); the item frame then has the axis on x.
        r, h2 = 0.5 * float(draw.size[0]), 0.5 * float(draw.size[2])
        return _Geometry(
            center_local=np.zeros(3),
            half=np.array([h2, r, r], dtype=np.float64),
            world_half=np.array([r, r, h2], dtype=np.float64),
            up_local=np.array([1.0, 0.0, 0.0]),
            quat=CYL_STAND_QUAT,
        )
    if draw.asset.startswith("prim_"):
        half = np.asarray(draw.size, dtype=np.float64) * 0.5
        return _Geometry(
            center_local=np.zeros(3),
            half=half,
            world_half=half.copy(),
            up_local=np.array([0.0, 0.0, 1.0]),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    asset, model_id = draw.asset, int(draw.model_id or 0)
    aabb = _mesh_aabb(asset, model_id)
    if aabb is not None:
        lo, hi = aabb
        center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    else:  # mesh unloadable: the old json-metadata path (obstacle.py semantics)
        center, half = ob.world_center_and_half(_load_model_cfg_local(asset, model_id))
        if float(np.max(np.abs(half))) < 1e-6:
            half = np.array([0.051, 0.051, 0.052], dtype=np.float64)  # obstacle._model_half_extents
    r, quat = _stand_R_quat_for(asset, center, half)
    return _Geometry(
        center_local=center,
        half=half,
        world_half=np.abs(r) @ half,
        up_local=r.T @ np.array([0.0, 0.0, 1.0]),  # body axis that spawn puts vertical
        quat=quat,
    )


# ---------------------------------------------------------------------------
# placement (pure numpy — unit-tested without sapien)
# ---------------------------------------------------------------------------


def _class_counts(k: int, cfg: dict, task_cfg: dict, rng: np.random.RandomState) -> list[str]:
    """Shuffled per-object placement classes honoring the on/near quotas."""
    max_on = int(cfg.get("max_on_path", 2))
    min_on_near = int(cfg.get("min_on_or_near_path", 2))
    if "on_path_count" in cfg:
        n_on = int(cfg["on_path_count"])  # caller-injected (runner's risk sampler)
    else:
        knob = ((task_cfg.get("risk") or {}).get("task") or {}).get("on_path_count")
        n_on = int(_sample_knob(knob, rng)) if knob is not None else int(rng.randint(1, max_on + 1))
    n_on = int(np.clip(n_on, 1, min(max_on, k)))
    n_near_min = max(0, min_on_near - n_on)
    n_on = min(n_on, k - n_near_min)  # leave room for the on/near minimum
    n_near = n_near_min
    for _ in range(k - n_on - n_near_min):
        if rng.rand() < 0.5:  # spare slots split evenly between near and off
            n_near += 1
    classes = ["on_path"] * n_on + ["near_path"] * n_near + ["off_path"] * (k - n_on - n_near)
    rng.shuffle(classes)
    return classes


def _in_table_xy(xy: np.ndarray, r_xy: float, xylim: np.ndarray) -> bool:
    """obstacle._in_table against an explicit xylim (test seam)."""
    m = float(r_xy) + 0.005
    return bool(
        xylim[0, 0] + m <= xy[0] <= xylim[0, 1] - m
        and xylim[1, 0] + m <= xy[1] <= xylim[1, 1] - m
    )


def _clearances_ok(
    xy: np.ndarray,
    r_xy: float,
    keepaways: list,
    placed: list,
    sep: float,
    keepaway_gap: float,
) -> bool:
    """Surface-to-surface clearance: >= keepaway_gap vs keepaways, >= sep vs placed."""
    for pt, r_keep in keepaways:
        if float(np.linalg.norm(xy - np.asarray(pt[:2], dtype=np.float64))) < r_xy + float(r_keep) + keepaway_gap:
            return False
    for pxy, pr in placed:
        if float(np.linalg.norm(xy - np.asarray(pxy, dtype=np.float64))) < r_xy + float(pr) + sep:
            return False
    return True


def _min_clearance(xy: np.ndarray, r_xy: float, keepaways: list, placed: list) -> float:
    best = float("inf")
    for pt, r_keep in keepaways:
        best = min(best, float(np.linalg.norm(xy - np.asarray(pt[:2], dtype=np.float64))) - float(r_keep) - r_xy)
    for pxy, pr in placed:
        best = min(best, float(np.linalg.norm(xy - np.asarray(pxy, dtype=np.float64))) - float(pr) - r_xy)
    return best


def _find_xy(
    cls: str,
    p0: np.ndarray,
    p1: np.ndarray,
    keepaways: list,
    placed: list,
    world_half: np.ndarray,
    cfg: dict,
    rng: np.random.RandomState,
    t_range,
    xylim: np.ndarray | None = None,
    vis_band: tuple | None = None,
    vis_cov: tuple | None = None,
    vis_out: dict | None = None,
) -> tuple[np.ndarray | None, float | None, float | None, bool]:
    """(xy, corridor_t, lateral, t_fallback) for one object, or all-None/False.

    on_path mirrors obstacle.geometric_pose's t candidate pattern (drawn t0,
    then a 9-point scan of the corridor range); when that scan fails, a
    second deterministic 25-point scan over ON_PATH_FALLBACK_T runs and its
    hits return t_fallback=True (fix 2.1 — the first scan is byte-identical,
    so corridors feasible in the base window see no behavior change).
    near_path draws (t, offset, sign) up to NEAR_PATH_TRIES times; off_path
    scans the obstacle.py table grid >= off_path_min_dist from the corridor
    and takes the max-clearance feasible cell (rng-shuffled tie order).
    vis_band (fix 3 / F5): (x_min, x_max, y_min, y_max) cell-center rect from
    the measured visibility band — off_path grid candidates outside it are
    skipped (±1e-9 tolerance); None (flag off / npz missing) is
    byte-identical to the unfiltered scan. on/near_path ignore it.
    vis_cov (fix 5 / F5b): (xs, ys, covered) measured per-cell coverage from
    the same npz. With the class's flag set (``near_path_visible_only`` /
    ``on_path_visible_only``) the class prefers covered cells within the
    ORIGINAL candidate sequence — near: one pass, first ``ok()+covered``
    draw wins, and when no covered cell appears the FIRST ``ok()`` draw is
    kept (byte-identical to the pre-F5b placement) and flagged via
    ``vis_out["vis_fallback"]``; on: the t0 + 9-point base scan then the
    25-point fallback window accept the first covered ``ok()`` (the pass
    consumes NO rng draws — t0 is already drawn — so the rng stream matches
    the flag-off run), else the original first-``ok()`` scan runs and is
    flagged. vis_cov=None skips both preferences byte-identically; off_path
    never reads it.
    """
    xylim = ob.TABLE_XYLIM if xylim is None else np.asarray(xylim, dtype=np.float64)
    p0 = np.asarray(p0[:2], dtype=np.float64)
    p1 = np.asarray(p1[:2], dtype=np.float64)
    delta = p1 - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:  # obstacle.py's degenerate-corridor fallback
        delta = np.array([0.0, -0.12], dtype=np.float64)
        dist = 0.12
    perp = np.array([-delta[1], delta[0]], dtype=np.float64) / dist
    r_xy = float(np.max(np.asarray(world_half, dtype=np.float64)[:2]))
    sep = float(cfg.get("min_separation", 0.03))
    keepaway_gap = float(cfg.get("keepaway_gap", ob.KEEPAWAY_GAP))
    t_lo, t_hi = (float(v) for v in (t_range or [0.22, 0.48]))

    def ok(xy: np.ndarray) -> bool:
        return _in_table_xy(xy, r_xy, xylim) and _clearances_ok(
            xy, r_xy, keepaways, placed, sep, keepaway_gap
        )

    if cls == "on_path":
        t0 = float(np.clip(rng.uniform(t_lo, t_hi), t_lo, t_hi))
        t_tries = [t0]
        for t in np.linspace(t_lo, t_hi, 9):
            t = float(t)
            if all(abs(t - x) > 1e-4 for x in t_tries):
                t_tries.append(t)
        if vis_cov is not None and cfg.get("on_path_visible_only", False):
            # fix 5 (F5b): same candidate sequence, covered cells only (no rng)
            for t in t_tries:
                cand = p0 + t * delta
                if ok(cand) and _covered_at(vis_cov, cand):
                    return cand, t, 0.0, False
            for t in np.linspace(ON_PATH_FALLBACK_T[0], ON_PATH_FALLBACK_T[1], ON_PATH_FALLBACK_N):
                t = float(t)
                cand = p0 + t * delta
                if ok(cand) and _covered_at(vis_cov, cand):
                    return cand, t, 0.0, True
            if vis_out is not None:
                vis_out["vis_fallback"] = True
        for t in t_tries:
            cand = p0 + t * delta
            if ok(cand):
                return cand, t, 0.0, False
        # fix 2.1: base window infeasible -> scan the wider fallback window
        for t in np.linspace(ON_PATH_FALLBACK_T[0], ON_PATH_FALLBACK_T[1], ON_PATH_FALLBACK_N):
            t = float(t)
            cand = p0 + t * delta
            if ok(cand):
                return cand, t, 0.0, True
        return None, None, None, False

    if cls == "near_path":
        t_near_lo, t_near_hi = (float(v) for v in cfg.get("near_path_t", [0.05, 0.95]))
        off_lo, off_hi = (float(v) for v in cfg.get("near_path_offset", [0.02, 0.06]))
        # fix 5 (F5b, fused after the ur5 A/B): ONE pass over the original
        # candidate sequence. The first ok()+covered draw wins; when no
        # covered cell appears in NEAR_PATH_TRIES draws, the FIRST ok() draw
        # is kept — byte-identical to the pre-F5b placement — and flagged
        # vis_fallback. The earlier two-loop version re-rolled the fallback
        # from a shifted stream and discarded good first-ok placements;
        # measured on ur5 it REGRESSED near_path visibility (0.959 -> 0.897,
        # tall thin items occluded by siblings the empty-table probe sweep
        # cannot see).
        vis_checked = vis_cov is not None and cfg.get("near_path_visible_only", False)
        first_ok = None
        for _ in range(NEAR_PATH_TRIES):
            t = float(np.clip(rng.uniform(t_near_lo, t_near_hi), t_near_lo, t_near_hi))
            off = float(rng.uniform(off_lo, off_hi))
            sign = 1.0 if rng.rand() < 0.5 else -1.0
            cand = p0 + t * delta + sign * off * perp
            if not ok(cand):
                continue
            if not vis_checked:
                return cand, t, sign * off, False
            if first_ok is None:
                first_ok = (cand, t, sign * off)
            if _covered_at(vis_cov, cand):
                return cand, t, sign * off, False
        if first_ok is not None:
            if vis_out is not None:
                vis_out["vis_fallback"] = True
            cand, t, lat = first_ok
            return cand, t, lat, False
        return None, None, None, False

    # off_path
    min_dist = float(cfg.get("off_path_min_dist", 0.10))
    grid = ob._table_grid(r_xy)
    order = rng.permutation(grid.shape[0])
    best_xy, best_score = None, -np.inf
    for idx in order:
        cand = grid[idx]
        if vis_band is not None and not (  # fix 3 (F5): stay inside the visible band
            vis_band[0] - 1e-9 <= cand[0] <= vis_band[1] + 1e-9
            and vis_band[2] - 1e-9 <= cand[1] <= vis_band[3] + 1e-9
        ):
            continue
        if ob._line_dist(cand, p0, p1) < min_dist:
            continue
        if not ok(cand):
            continue
        score = _min_clearance(cand, r_xy, keepaways, placed)
        if score > best_score:
            best_xy, best_score = cand, score
    if best_xy is None:
        return None, None, None, False
    return np.asarray(best_xy, dtype=np.float64), None, None, False


def _layout_plan(
    p0: np.ndarray,
    p1: np.ndarray,
    keepaways: list,
    cfg: dict,
    task_cfg: dict,
    rng: np.random.RandomState,
    pool: list[dict],
    train_colors: list[str],
    t_range,
    geometry_fn=_geometry_of,
    xylim: np.ndarray | None = None,
    vis_band: tuple | None = None,
    vis_cov: tuple | None = None,
) -> tuple[list[Placement], int] | None:
    """Draw one layout or None; returns (placements, k_dropped).

    Draw order per object (deterministic given rng): class list -> per object
    asset/id/color/size -> placement (t / offset / grid). ``geometry_fn`` is
    the test seam (non-sim tests inject fake extents). ``vis_band`` (fix 3 /
    F5) is the off_path visibility-band rect, passed through to _find_xy;
    None keeps the historical unrestricted off_path scan. ``vis_cov`` (fix 5
    / F5b) is the (xs, ys, covered) measured coverage mask for the
    near/on_path covered preference; None keeps the historical
    unrestricted draws byte-identically.

    Fix 2.2 (degradation chain): an object whose drawn class cannot be
    placed retries the SAME draw down _DEGRADE_CHAIN; stepping into
    off_path is allowed only when the min_on_or_near_path quota is already
    met by the final classes placed so far (a degraded-to-near object
    counts). The Placement records the FINAL class and degraded_from.

    Fix 2.3 (partial-K): when a slot's chain is exhausted but
    len(placements) >= max(2, quota) with the on/near quota met, the loop
    stops and returns the partial layout with k_dropped = unplaced slots;
    otherwise the whole layout is None (a size-cap draw failure also
    returns None, as before).
    """
    k_lo, k_hi = (int(v) for v in cfg.get("k", [3, 5]))
    k = int(rng.randint(k_lo, k_hi + 1))
    classes = _class_counts(k, cfg, task_cfg, rng)
    caps = {
        "on_path": cfg.get("on_path_max_r_xy", 0.035),
        "near_path": cfg.get("near_path_max_r_xy", 0.035),
        "off_path": None,
    }
    quota = int(cfg.get("min_on_or_near_path", 2))
    placements: list[Placement] = []
    placed: list[tuple[np.ndarray, float]] = []

    def on_near_count() -> int:
        return sum(p.cls in ("on_path", "near_path") for p in placements)

    for i, cls in enumerate(classes):
        draw = _draw_object(pool, rng, train_colors, max_r_xy=caps[cls], geometry_fn=geometry_fn)
        if draw is None:
            return None
        geo = geometry_fn(draw)
        final_cls = None
        xy = t_used = lateral = None
        t_fallback = vis_fallback = False
        for cand_cls in _DEGRADE_CHAIN[cls]:
            if cand_cls != cls and cand_cls == "off_path" and on_near_count() < quota:
                break  # degrade into off_path refused: on/near quota unmet
            vis_out: dict = {}
            xy, t_used, lateral, t_fallback = _find_xy(
                cand_cls, p0, p1, keepaways, placed, geo.world_half, cfg, rng, t_range,
                xylim, vis_band=vis_band, vis_cov=vis_cov, vis_out=vis_out,
            )
            if xy is not None:
                final_cls = cand_cls
                vis_fallback = bool(vis_out.get("vis_fallback", False))
                break
        if final_cls is None:
            if len(placements) >= max(2, quota) and on_near_count() >= quota:
                return placements, k - len(placements)  # partial-K acceptance
            return None
        placed.append((np.asarray(xy, dtype=np.float64), float(np.max(geo.world_half[:2]))))
        placements.append(
            Placement(
                item_id=f"obj_{i}",
                cls=final_cls,
                draw=draw,
                geo=geo,
                xyz=np.array([xy[0], xy[1], 0.0], dtype=np.float64),  # z packed by the caller
                corridor_t=t_used,
                lateral=lateral,
                t_fallback=t_fallback,
                degraded_from=None if final_cls == cls else cls,
                vis_fallback=vis_fallback,
            )
        )
    return placements, 0


# ---------------------------------------------------------------------------
# off_path visibility band (fix 3 / F5; pure numpy, test seam = _CONFIGS_DIR)
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _read_band_rect(path: Path) -> tuple:
    """(x_min, x_max, y_min, y_max) from a visibility_band npz."""
    with np.load(path) as z:
        return tuple(float(v) for v in z["rect"].reshape(4))


def _vis_band_for(embodiment: str | None, configs_dir=None) -> tuple[tuple | None, str | None]:
    """(rect, label) of the measured off_path visibility band, or (None, None).

    Resolution: the embodiment's own ``visibility_band_<embodiment>.npz``
    (label = the embodiment name), else any OTHER embodiment's band file
    (label "shared-fallback" — camera rigs differ per embodiment, but a
    foreign band is strictly better than the unmeasured far corners), else
    (None, None) = no restriction. Called only when the task cfg sets
    ``objects.off_path_visible_only``.
    """
    cfg_dir = Path(configs_dir) if configs_dir is not None else _CONFIGS_DIR
    if embodiment:
        own = cfg_dir / f"visibility_band_{embodiment}.npz"
        if own.is_file():
            return _read_band_rect(own), str(embodiment)
    for path in sorted(cfg_dir.glob("visibility_band_*.npz")):
        if embodiment and path.stem == f"visibility_band_{embodiment}":
            continue
        return _read_band_rect(path), "shared-fallback"
    return None, None


def _read_vis_cov(path: Path) -> tuple:
    """(xs, ys, covered) from a visibility_band npz — the measured per-cell mask."""
    with np.load(path) as z:
        xs = np.asarray(z["xs"], dtype=np.float64).reshape(-1)
        ys = np.asarray(z["ys"], dtype=np.float64).reshape(-1)
        covered = np.asarray(z["covered"], dtype=bool)
    return xs, ys, covered


def _vis_covered_for(embodiment: str | None, configs_dir=None) -> tuple[tuple | None, str | None]:
    """((xs, ys, covered), label) of the measured visibility coverage, or (None, None).

    Same resolution order and labels as ``_vis_band_for`` (own embodiment npz,
    else the other embodiment's as "shared-fallback", else none) — fix 5 (F5b)
    restricts near/on_path to the npz's measured ``covered`` cells, a stricter
    per-cell mask than the off_path band rect. Called only when the task cfg
    sets ``objects.near_path_visible_only`` / ``objects.on_path_visible_only``.
    """
    cfg_dir = Path(configs_dir) if configs_dir is not None else _CONFIGS_DIR
    if embodiment:
        own = cfg_dir / f"visibility_band_{embodiment}.npz"
        if own.is_file():
            return _read_vis_cov(own), str(embodiment)
    for path in sorted(cfg_dir.glob("visibility_band_*.npz")):
        if embodiment and path.stem == f"visibility_band_{embodiment}":
            continue
        return _read_vis_cov(path), "shared-fallback"
    return None, None


def _covered_at(vis_cov: tuple, xy: np.ndarray) -> bool:
    """Nearest-cell lookup into (xs, ys, covered): is this xy in a covered cell?"""
    xs, ys, covered = vis_cov
    ix = int(np.argmin(np.abs(xs - float(xy[0]))))
    iy = int(np.argmin(np.abs(ys - float(xy[1]))))
    return bool(covered[iy, ix])


# ---------------------------------------------------------------------------
# settle verdict (pure numpy)
# ---------------------------------------------------------------------------


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _tilt_rad(q_wxyz: np.ndarray, up_local: np.ndarray) -> float:
    """Angle between world +z and the body axis that spawn put vertical."""
    up_world = _quat_wxyz_to_mat(q_wxyz) @ np.asarray(up_local, dtype=np.float64).reshape(3)
    cos = float(np.clip(up_world[2] / (np.linalg.norm(up_world) + 1e-12), -1.0, 1.0))
    return float(np.arccos(cos))


def settle_verdict(
    before: list[tuple[np.ndarray, np.ndarray]],
    after: list[tuple[np.ndarray, np.ndarray]],
    up_locals: list[np.ndarray],
    move_tol: float = MOVE_TOL_M,
    tilt_tol: float = TILT_TOL_RAD,
) -> tuple[bool, list[dict]]:
    """(ok, per-item details): reject when moved > move_tol or tilted > tilt_tol.

    before/after: (p[3], q_wxyz[4]) per item, same order as up_locals.
    """
    ok = True
    details = []
    for i, ((p0, _q0), (p1, q1)) in enumerate(zip(before, after)):
        moved = float(np.linalg.norm(np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)))
        tilt = _tilt_rad(q1, up_locals[i])
        details.append({"index": i, "moved_m": moved, "tilt_deg": float(np.rad2deg(tilt))})
        if moved > move_tol or tilt > tilt_tol:
            ok = False
    return ok, details


# ---------------------------------------------------------------------------
# spawning (sapien, lazy imports)
# ---------------------------------------------------------------------------


def _color_rgb(objects_cfg: dict, name: str | None) -> tuple:
    rgb = (objects_cfg.get("color_rgb") or {}).get(str(name))
    if rgb is None:
        return _FALLBACK_COLOR
    return tuple(float(v) for v in rgb[:3])


def _spawn_one(task, placement: Placement, static: bool, objects_cfg: dict):
    """Spawn one placed object; returns the RoboTwin Actor (or raw entity for cylinders).

    Mesh path mirrors obstacle.spawn_obstacle (stand quat, center compensation,
    create_box fallback on missing mesh/scale) but HONORS is_static —
    spawn_obstacle itself forces static=True. Prims go through RoboTwin's
    create_box / create_cylinder with the sampled color.
    """
    import sapien
    from envs.utils.create_actor import create_actor, create_box, create_cylinder

    pl, geo = placement, placement.geo
    xyz = np.asarray(pl.xyz, dtype=np.float64)
    if pl.draw.asset == "prim_box":
        actor = create_box(
            task,
            sapien.Pose(xyz.tolist(), [1, 0, 0, 0]),
            half_size=geo.half.tolist(),
            color=_color_rgb(objects_cfg, pl.draw.color),
            is_static=static,
            name=pl.item_id,
        )
    elif pl.draw.asset == "prim_tall_cylinder":
        actor = create_cylinder(
            task,
            sapien.Pose(xyz.tolist(), list(geo.quat)),  # CYL_STAND_QUAT: axis vertical
            radius=float(geo.half[1]),
            half_length=float(geo.half[0]),
            color=_color_rgb(objects_cfg, pl.draw.color),
            name=pl.item_id,
        )
        if static:  # create_cylinder is dynamic-only; kinematic is the static parity
            import sapien.physx as physx

            comp = actor.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if comp is not None:
                comp.kinematic = True
    else:
        r = _quat_wxyz_to_mat(np.asarray(geo.quat, dtype=np.float64))
        pose_xyz = xyz - r @ geo.center_local  # spawn_obstacle's center compensation
        actor = create_actor(
            task,
            sapien.Pose(pose_xyz.tolist(), list(geo.quat)),
            pl.draw.asset,
            convex=True,
            is_static=static,
            model_id=int(pl.draw.model_id or 0),
        )
        if actor is None:  # mesh/json missing: spawn_obstacle's create_box fallback
            actor = create_box(
                task,
                sapien.Pose(xyz.tolist(), [1, 0, 0, 0]),
                half_size=geo.world_half.tolist(),
                color=_FALLBACK_COLOR,
                is_static=static,
                name=pl.item_id,
            )
            print(f"[spawn] {pl.draw.asset} id {pl.draw.model_id} missing; create_box fallback")
        else:
            actor.actor.set_name(pl.item_id)
    return actor


def _spec_from(placement: Placement, actor, mass: float) -> ItemSpec:
    return ItemSpec(
        item_id=placement.item_id,
        entity=actor,
        category=placement.draw.category,
        color=placement.draw.color,
        kind="rigid",
        half_extents=placement.geo.half,
        obb_center_local=placement.geo.center_local,
        up_axis_local=placement.geo.up_local,
        mass=float(mass),
        role="candidate",
        group=None,
    )


def _remove_layout(env, specs: list[ItemSpec]) -> None:
    scene = env.task.scene
    for spec in specs:
        try:
            scene.remove_entity(entity_of(spec))
        except Exception as exc:
            print(f"[spawn] remove_entity({spec.item_id}) failed: {exc}")
    meta = getattr(env.task, "item_meta", None)
    if meta:
        ids = {s.item_id for s in specs}
        env.task.item_meta = [d for d in meta if d.get("item_id") not in ids]


def _settle(env, steps: int) -> None:
    scene = env.task.scene
    for _ in range(int(steps)):
        scene.step()


def _pose_of(spec: ItemSpec) -> tuple[np.ndarray, np.ndarray]:
    pose = entity_of(spec).get_pose()
    return (
        np.asarray(pose.p, dtype=np.float64).reshape(3).copy(),
        np.asarray(pose.q, dtype=np.float64).reshape(4).copy(),
    )


# ---------------------------------------------------------------------------
# arm-link keepaways (fix 4b — the P4 settle-toppling root cause)
# ---------------------------------------------------------------------------

ARM_LINK_KEEPAWAY_R = 0.08  # covers hand/finger collision volumes about the link pose
ARM_LINK_Z_MAX = 0.30  # only links below table_z + this threaten standing candidates
ARM_LINK_XY_PAD = 0.10  # table xy inflated by this much for the overhang test


def _arm_link_keepaways(env, table_z: float, cfg: dict) -> list:
    """One keepaway circle per robot link pose overhanging the table.

    obstacle.keepaways_from_robot's base circles (r=0.18 at the arm BASES,
    y=-0.65) never bind on the table — but franka's home pose rests the
    hand/fingers at z 0.85-0.98 m over (x ±0.3, y −0.2), INSIDE the spawn
    region (offline FK on the embodiment URDFs at the config homestate), so
    candidates >= ~11 cm tall spawned there interpenetrate the gripper and
    PhysX ejects them during the settle (the P4 slides/topples; ur5's folded
    home clears the tallest pool object by >= 7 cm). One circle per link with
    pose z < table_z + ``arm_link_z_max`` and xy over the padded table; the
    usual keepaway_gap applies on top via _clearances_ok. Pure pose reads —
    a broken/headless robot yields no circles, never an exception.
    """
    r = float(cfg.get("arm_link_keepaway_r", ARM_LINK_KEEPAWAY_R))
    z_max = float(table_z) + float(cfg.get("arm_link_z_max", ARM_LINK_Z_MAX))
    x0, x1 = ob.TABLE_XYLIM[0, 0] - ARM_LINK_XY_PAD, ob.TABLE_XYLIM[0, 1] + ARM_LINK_XY_PAD
    y0, y1 = ob.TABLE_XYLIM[1, 0] - ARM_LINK_XY_PAD, ob.TABLE_XYLIM[1, 1] + ARM_LINK_XY_PAD
    out = []
    robot = getattr(env, "robot", None)
    for entity in (getattr(robot, "left_entity", None), getattr(robot, "right_entity", None)):
        if entity is None:
            continue
        try:
            links = entity.get_links()
        except Exception:
            continue
        for link in links:
            try:
                pose = link.get_root_pose() if hasattr(link, "get_root_pose") else link.get_pose()
                x, y, z = (float(v) for v in pose.p[:3])
            except Exception:
                continue
            if z >= z_max or not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            out.append((np.array([x, y], dtype=np.float64), r))
    return out


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def spawn_candidates(env, task_cfg: dict, rng: np.random.RandomState) -> tuple[list[ItemSpec], dict]:
    """Spawn K dynamic candidate objects on a CrossSafe host task.

    env: a built CEHJ ``Env`` whose ``task_name`` is one of the T1 hosts.
    task_cfg: the T1 yaml dict (uses ``objects`` + ``corridor_t`` + the
    ``risk.task.on_path_count`` fallback knob). rng: the caller's
    ``np.random.RandomState(seed + 4241)`` (PLAN v3 §2) — never reseeded here.

    Returns (specs, report): specs are role="candidate" ItemSpecs (entity set,
    realistic mass applied); the report carries spawn_rejections (total
    resamples), layout_retries / settle_rejections, the per-object placement
    records (each with its FINAL cls plus t_fallback / degraded_from), and
    max_layouts_hit. Fix-2 counters describe the final kept layout:
    t_fallbacks (objects placed via the on-path fallback window), degraded
    (objects whose final class differs from the drawn class), k_dropped
    (drawn slots left unplaced by partial-K acceptance; k + k_dropped = the
    drawn K). Also appends schema-compatible dicts to
    ``task.item_meta`` (the same channel robotwin_tasks/_item_utils writes)
    so runner-style consumers see the candidates. Does NOT call
    update_curobo_world — that is the caller's job.

    Fix 3 (F5): when ``objects.off_path_visible_only`` is set, the measured
    visibility band (``_vis_band_for``) is loaded ONCE here and threaded into
    every ``_layout_plan`` call; ``report["vis_band"]`` records which band
    was applied — the embodiment's own name, "shared-fallback" (the other
    embodiment's npz), or None (flag off / no band file anywhere — the
    off_path scan is then byte-identical to the historical behavior).

    Fix 4 (P4 settle-toppling): placement geometry is MEASURED from the
    collision meshes (``_geometry_of`` / ``_mesh_aabb`` — the model_data json
    metadata is a fallback only), and the z-pack rests each object's world
    AABB BOTTOM at ``table_z + 0.002`` for any asset origin convention. The
    keepaway list gains ``_arm_link_keepaways`` (fix 4b: franka's home-pose
    hand/fingers overhang the spawn region — the dominant settle-ejection
    mechanism); ``report["arm_link_keepaways"]`` counts the added circles.

    Fix 5 (F5b): when ``objects.near_path_visible_only`` and/or
    ``objects.on_path_visible_only`` are set, the measured per-cell coverage
    (``_vis_covered_for``) is loaded ONCE here and threaded into every
    ``_layout_plan`` call; ``report["vis_cov"]`` records which mask was
    applied (embodiment name | "shared-fallback" | None — None means the
    covered preference was skipped and the layout is byte-identical to the
    historical behavior). The counters ``near_path_vis_fallbacks`` /
    ``on_path_vis_fallbacks`` describe the FINAL kept layout: objects that
    kept the original first-ok placement because no covered cell appeared
    in their candidate sequence (``vis_fallback=True`` on their placement
    records). off_path is unaffected (fix 3 owns it).
    """
    spec = TASK_SPECS.get(getattr(env, "task_name", ""))
    if spec is None:
        raise KeyError(
            f"{getattr(env, 'task_name', '')!r} is not in main.envs.tasks.TASK_SPECS; "
            "spawn_candidates runs on the five T1 host tasks only"
        )
    cfg = dict(task_cfg.get("objects") or {})
    t_range = task_cfg.get("corridor_t", [0.22, 0.48])
    static = bool(cfg.get("static_objects", False))
    # fix 3 (F5): the off_path visibility band, loaded once per call. The env's
    # canonical embodiment name threads through (Env.embodiment; fall back to
    # the task attribute for runner-built envs); None -> fallback/none.
    embodiment = getattr(env, "embodiment", None) or getattr(env.task, "embodiment", None)
    vis_band = band_label = None
    if cfg.get("off_path_visible_only", False):
        vis_band, band_label = _vis_band_for(embodiment)
    # fix 5 (F5b): the measured per-cell coverage mask for near/on_path,
    # loaded once per call when either class flag is set (same npz, same
    # resolution order as the band; None -> both covered passes skipped)
    vis_cov = cov_label = None
    if cfg.get("near_path_visible_only", False) or cfg.get("on_path_visible_only", False):
        vis_cov, cov_label = _vis_covered_for(embodiment)
    objects_cfg = physics.load_objects_cfg()
    pool = t1_pool_entries(objects_cfg)
    train_colors = list((objects_cfg.get("colors") or {}).get("train") or [])
    nominal_ranges = {
        key: (entry or {}).get("mass_range")
        for key, entry in (objects_cfg.get("categories", {}) or {}).items()
    }

    arm = resolve_arm(env.task, spec, "auto")
    p0, p1, _arm_hint = longest_corridor(env.task, spec, env.robot, arm)
    table_z = ob.TABLE_Z + float(getattr(env.task, "table_z_bias", 0.0))
    keepaways = ob.keepaways_from_task(env.task, spec) + ob.keepaways_from_robot(env)
    # fix 4b: franka's home-pose hand/fingers overhang the spawn region — one
    # keepaway circle per link over the table (default on; objects.arm_link_keepaways)
    arm_links = _arm_link_keepaways(env, table_z, cfg) if cfg.get("arm_link_keepaways", True) else []
    keepaways += arm_links
    gap_range = spec.get("obstacle_bbox_gap")  # choose_and_spawn parity (stack_blocks_two)
    if gap_range is not None:
        extra = float(rng.uniform(float(gap_range[0]), float(gap_range[1])))
        keepaways = [(xy, float(r) + extra) for xy, r in keepaways]
        print(f"[spawn] bbox gap +{extra:.3f} m (task {env.task_name})")

    report: dict = {
        "spawn_rejections": 0,
        "layout_retries": 0,
        "settle_rejections": 0,
        "max_layouts_hit": False,
        "static_objects": static,
        "k": 0,
        "t_fallbacks": 0,
        "degraded": 0,
        "k_dropped": 0,
        "vis_band": band_label,  # fix 3 (F5): embodiment name | "shared-fallback" | None
        "vis_cov": cov_label,  # fix 5 (F5b): embodiment name | "shared-fallback" | None
        "near_path_vis_fallbacks": 0,  # fix 5 (F5b): final-layout uncovered near placements
        "on_path_vis_fallbacks": 0,  # fix 5 (F5b): final-layout uncovered on placements
        "arm_link_keepaways": len(arm_links),  # fix 4b: link circles added (0 = off/none)
        "placements": [],
        "settle": [],
        "corridor": {"p0": [float(v) for v in p0[:2]], "p1": [float(v) for v in p1[:2]]},
    }

    specs: list[ItemSpec] = []
    placements: list[Placement] | None = None
    k_dropped = 0
    # no try/finally: nothing below flips obstacle.py globals anymore
    # (_geometry_of measures meshes instead of set_obstacle_model), so there
    # is no module state to restore on the way out.
    # fallback ladder (only used when EVERY layout of a tier fails to plan —
    # the first tier is the historical longest corridor, so every seed that
    # produced a layout before is byte-identical): longest corridor -> the
    # host's other pick->place corridors -> the longest corridor with the
    # on/near quota relaxed to 1. A layout-infeasible seed produced an
    # obstacle-free T1 episode (franka stack_blocks_two seed 1: 40/40
    # layouts failed on the 0.36 m corridor with 12 arm-link keepaways).
    tiers = [(p0, p1, cfg)]
    try:
        others = sorted(iter_corridors(env.task, spec), key=lambda c: -float(np.linalg.norm(c[1] - c[0])))
    except Exception:
        others = []
    for q0, q1, _label, _tgt in others:
        if np.linalg.norm(np.asarray(q0)[:2] - np.asarray(p0)[:2]) + np.linalg.norm(np.asarray(q1)[:2] - np.asarray(p1)[:2]) > 1e-3:
            tiers.append((q0, q1, cfg))
    tiers.append((p0, p1, {**cfg, "min_on_or_near_path": 1}))
    tier_i, attempt_total = 0, 0
    report["fallback_tier"] = 0
    while tier_i < len(tiers):
        p0, p1, tier_cfg = tiers[tier_i]
        planned = False
        for attempt in range(MAX_LAYOUTS):
            if attempt_total:
                report["spawn_rejections"] += 1
            attempt_total += 1
            out = _layout_plan(
                p0, p1, keepaways, tier_cfg, task_cfg, rng, pool, train_colors, t_range,
                vis_band=vis_band, vis_cov=vis_cov,
            )
            if out is None:
                report["layout_retries"] += 1
                continue
            planned = True
            break
        if planned:
            break
        tier_i += 1
    if tier_i:
        report["fallback_tier"] = int(tier_i)
        report["corridor"] = {"p0": [float(v) for v in np.asarray(p0)[:2]], "p1": [float(v) for v in np.asarray(p1)[:2]]}
        print(f"[spawn] layout fallback tier {tier_i} ({'other corridor' if tier_i < len(tiers) - 1 else 'relaxed quota'})")
    if tier_i >= len(tiers):
        attempts = []  # every tier infeasible: k = 0 (recorded, never a crash)
    else:
        attempts = range(MAX_LAYOUTS)
        first_out = out
    for attempt in attempts:
        if attempt:
            report["spawn_rejections"] += 1
            out = _layout_plan(
                p0, p1, keepaways, tiers[tier_i][2], task_cfg, rng, pool, train_colors, t_range,
                vis_band=vis_band, vis_cov=vis_cov,
            )
        else:
            out = first_out
        if out is None:
            report["layout_retries"] += 1
            continue
        placements, k_dropped = out
        for pl in placements:
            # bottom-rest z (fix 4): pl.xyz is where the world AABB CENTER
            # lands (_spawn_one compensates the measured center_local), so
            # the AABB BOTTOM rests at table_z + 0.002 for any asset origin
            # convention; world_half[2] is the measured rotated half-z.
            pl.xyz = np.array([pl.xyz[0], pl.xyz[1], table_z + float(pl.geo.world_half[2]) + 0.002])
        specs = []
        for pl in placements:
            actor = _spawn_one(env.task, pl, static, objects_cfg)
            mass = 0.0
            if not static:
                lo_hi = nominal_ranges.get(pl.draw.category) or [0.05, 0.15]
                mass = float(rng.uniform(float(lo_hi[0]), float(lo_hi[1])))
                physics.set_mass_props(actor, mass)
            specs.append(_spec_from(pl, actor, mass))
        before = [_pose_of(s) for s in specs]
        _settle(env, SETTLE_STEPS)
        after = [_pose_of(s) for s in specs]
        ok, details = settle_verdict(before, after, [s.up_axis_local for s in specs])
        for d, s in zip(details, specs):
            d["item_id"] = s.item_id
        report["settle"] = details
        if ok:
            break
        report["settle_rejections"] += 1
        if attempt < MAX_LAYOUTS - 1:
            _remove_layout(env, specs)
            specs = []
            print(
                f"[spawn] settle reject (attempt {attempt + 1}/{MAX_LAYOUTS}): "
                + ", ".join(
                    f"{d['item_id']} moved {1e3 * d['moved_m']:.1f} mm tilt {d['tilt_deg']:.1f} deg"
                    for d in details
                )
            )
        else:
            report["max_layouts_hit"] = True
            print(f"[spawn] MAX_LAYOUTS hit; keeping the last (unsettled) layout")

    if placements is not None and specs:
        report["placements"] = [pl.report_dict() for pl in placements]
        # fix-2 counters describe the FINAL kept layout (rejected attempts
        # are not counted); k + k_dropped = the drawn K of that layout
        report["t_fallbacks"] = sum(1 for pl in placements if pl.t_fallback)
        report["degraded"] = sum(1 for pl in placements if pl.degraded_from is not None)
        report["k_dropped"] = int(k_dropped)
        # fix-5 counters describe the FINAL kept layout, per FINAL class
        report["near_path_vis_fallbacks"] = sum(
            1 for pl in placements if pl.cls == "near_path" and pl.vis_fallback
        )
        report["on_path_vis_fallbacks"] = sum(
            1 for pl in placements if pl.cls == "on_path" and pl.vis_fallback
        )
    report["k"] = len(specs)

    # runner-style consumers: same metadata channel as register_item
    if not hasattr(env.task, "item_meta"):
        env.task.item_meta = []
    for s in specs:
        env.task.item_meta.append(
            {
                "item_id": s.item_id,
                "entity": s.entity,
                "category": s.category,
                "color": s.color,
                "kind": s.kind,
                "role": s.role,
                "group": s.group,
                "particles": None,
                "half_extents": [float(v) for v in s.half_extents],
                "obb_center_local": [float(v) for v in s.obb_center_local],
                "up_axis_local": [float(v) for v in s.up_axis_local],
                "mass": float(s.mass),
            }
        )
    return specs, report
