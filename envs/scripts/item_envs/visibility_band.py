#!/usr/bin/env python
"""P4-T1 F5 fix: empirical visibility band for off_path candidate placement.

The P1-rerun visibility gate measured per-item max-camera visible fraction at
t=0 (``masks.item_masks``): near_path 0.951 / on_path 0.917 pass, off_path
0.642 fails (>= 0.3 in >= 1 camera in >= 95% of seeds per item). Off-path
objects in the far table corners sit outside every camera frustum (some
vis_max = 0.0), and no placement-time rule keeps them in view. This script
measures WHERE on the table an object is actually visible, per embodiment,
and turns that into a placement constraint:

BASE MODE (default) — build ONE ``smoke_items`` Env (cheapest full-rig
scene), then repose ONE kinematic reference cylinder (r 3 cm, h 12 cm —
a typical tall prim) over a 2 cm table xy grid. Per cell: brief settle,
``masks.item_masks`` -> per-camera visible fraction, max over cameras.
Repose (not rebuild) because repeated scene creation hits the ~26 svulkan2
render-scene creations/process limit. Output
``main/item_envs/configs/visibility_band_<embodiment>.npz`` with the grid,
per-cell visibility, the covered mask (vis_max >= 0.3), the derived boolean
band, and the band rectangle — the quantity ``spawn._vis_band_for`` reads.

Band-rectangle algorithm (keep in sync with spawn.py's docstring):
  1. covered = vis_max >= 0.3, [ny, nx] boolean (rows = y, cols = x);
  2. largest all-covered axis-aligned rectangle = maximal rectangle in a
     binary matrix via the histogram stack DP;
  3. inset the winner by one cell on each side (boundary cells measured
     at the frustum edge are brittle); if the inset would empty the rect
     (<= 0 cells on an axis), keep the non-inset rect;
  4. rect = cell-CENTER coords (x_min, x_max, y_min, y_max).

--validate MODE — the F5 re-measure: 20 seeds x 5 hosts x both embodiments,
setup-only episodes (build env -> ``spawn.spawn_candidates`` with the band
active via the T1 yaml's ``off_path_visible_only: true``), t=0
``masks.item_masks`` over the spawned specs. One JSONL row per item
{host, embodiment, seed, item_id, cls, xyz, corridor_t, lateral,
t_fallback, vis_fallback, vis_max, ...}; a per-class summary
(fraction with vis_max >= 0.3) prints at the end. Resumable: rows keyed
(host, embodiment, seed, item_id) are skipped on rerun. The spawn side
mirrors probes/p1.py exactly (rng = RandomState(seed + 4241)); the
RiskTickController attach and cuRobo world refresh are skipped — neither
consumes the caller rng nor moves the placed objects, so placements match
P4-T1's episodes one for one (modulo the intended band restriction).

GPU ONLY: needs the RT render system for camera segmentation. Do not run on
the login node.

Usage (cwd = /work/nvme/bifq/project/LanguageConditionHJ):
    python envs/scripts/item_envs/visibility_band.py --embodiment ur5-wsg
    python envs/scripts/item_envs/visibility_band.py --embodiment both --validate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "envs" / "main" / "item_envs" / "spawn.py").is_file():
    ROOT = Path("/work/nvme/bifq/project/LanguageConditionHJ")
ENVS_DIR = ROOT / "envs"
if str(ENVS_DIR) not in sys.path:
    sys.path.insert(0, str(ENVS_DIR))  # `main` = envs/main; ensure_main_path adds ROOT/main

# RoboTwin's `envs` package (envs.utils.*) — same resolution as
# main/envs/env.py:16-18. Needed BEFORE any lazy RoboTwin import: nothing
# calls Env/prepare() first in this script, so sys.path would otherwise lack it.
ROBOTWIN_ROOT = ROOT / "RoboTwin"
if not ROBOTWIN_ROOT.is_dir():
    ROBOTWIN_ROOT = ROOT.parent / "RoboTwin"
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()

CONFIGS_DIR = ROOT / "envs" / "main" / "item_envs" / "configs"
T1_YAML = CONFIGS_DIR / "tasks" / "T1_selective_avoid.yaml"

CELL_M = 0.02  # grid spacing
MARGIN_M = 0.05  # grid stays this far inside TABLE_XYLIM
VIS_THRESH = 0.3  # the F5 gate threshold (max-camera visible fraction)
PROBE_R = 0.03  # reference cylinder radius (m)
PROBE_H = 0.12  # reference cylinder height (m)
SETTLE_STEPS = 3  # scene.step()s after each repose
EMBODIMENTS = ("ur5-wsg", "franka-panda")


# ---------------------------------------------------------------------------
# band rectangle (pure numpy — mirrored by tests)
# ---------------------------------------------------------------------------


def _largest_covered_rect(covered: np.ndarray):
    """Largest all-True axis-aligned rectangle in a [ny, nx] boolean matrix.

    Maximal-rectangle-in-binary-matrix via the histogram stack DP (rows = y,
    cols = x). Returns inclusive index bounds (y0, y1, x0, x1) or None when
    no cell is covered.
    """
    ny, nx = covered.shape
    heights = np.zeros(nx, dtype=np.int64)
    best_area = 0
    best = None
    for y in range(ny):
        heights = np.where(covered[y], heights + 1, 0)
        stack = []  # (left_index, height), heights increasing
        for x in range(nx + 1):
            h = int(heights[x]) if x < nx else 0  # sentinel flushes the stack
            left = x
            while stack and stack[-1][1] > h:
                l, hh = stack.pop()
                area = hh * (x - l)
                if area > best_area:
                    best_area = area
                    best = (y - hh + 1, y, l, x - 1)
                left = l
            stack.append((left, h))
    return best


def band_rect(covered: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """(rect, band) from the covered mask: largest rect, one-cell inset.

    rect = (x_min, x_max, y_min, y_max) in cell-center coordinates; band is
    the [ny, nx] boolean of cell centers inside the final rect. Returns
    (None, zeros) when no cell is covered.
    """
    idx = _largest_covered_rect(covered)
    band = np.zeros_like(covered)
    if idx is None:
        return None, band
    y0, y1, x0, x1 = idx
    iy0, iy1, ix0, ix1 = y0 + 1, y1 - 1, x0 + 1, x1 - 1  # one-cell inset
    if iy0 <= iy1 and ix0 <= ix1:
        y0, y1, x0, x1 = iy0, iy1, ix0, ix1
    rect = (float(xs[x0]), float(xs[x1]), float(ys[y0]), float(ys[y1]))
    band[y0 : y1 + 1, x0 : x1 + 1] = True
    return rect, band


def _ascii_map(mask: np.ndarray, xs, ys, char="#") -> str:
    head = "     " + "".join(f"{x:+.2f}"[1:4] + " " for x in xs)
    lines = [head]
    for iy in range(mask.shape[0] - 1, -1, -1):  # print high y first
        lines.append(f"{ys[iy]:+.2f} " + "".join(char if v else "." for v in mask[iy]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# base mode: per-cell visibility sweep
# ---------------------------------------------------------------------------


def sweep_embodiment(embodiment: str, out_dir: Path) -> dict:
    """Build one smoke_items Env, repose one probe cylinder over the grid."""
    import sapien
    import sapien.physx as physx

    from main.envs import obstacle as ob
    from main.item_envs import masks
    from main.item_envs.items import ItemSpec
    from main.item_envs.probes._common import build_env

    t0 = time.perf_counter()
    env = build_env("smoke_items", embodiment, 0)
    # AFTER build_env: Env init has chdir'd to the RoboTwin root, and
    # envs.utils.create_actor reads assets cwd-relative AT IMPORT TIME.
    from envs.utils.create_actor import create_cylinder
    try:
        task = env.task
        table_z = ob.TABLE_Z + float(getattr(task, "table_z_bias", 0.0))
        half_len = PROBE_H / 2.0
        probe = create_cylinder(
            task,
            sapien.Pose([0.0, 0.0, table_z + half_len + 0.001], [1, 0, 0, 0]),
            radius=PROBE_R,
            half_length=half_len,
            color=(0.9, 0.6, 0.1),
            name="vis_probe",
        )
        comp = probe.find_component_by_type(physx.PhysxRigidDynamicComponent)
        if comp is not None:
            comp.kinematic = True  # repose teleports; never falls
        spec = ItemSpec(
            item_id="vis_probe",
            entity=probe,
            category="probe",
            kind="rigid",
            half_extents=np.array([PROBE_R, PROBE_R, half_len]),
            obb_center_local=np.zeros(3),
        )

        xs = np.arange(ob.TABLE_XYLIM[0, 0] + MARGIN_M,
                       ob.TABLE_XYLIM[0, 1] - MARGIN_M + 0.5 * CELL_M, CELL_M)
        ys = np.arange(ob.TABLE_XYLIM[1, 0] + MARGIN_M,
                       ob.TABLE_XYLIM[1, 1] - MARGIN_M + 0.5 * CELL_M, CELL_M)
        vis = np.zeros((len(ys), len(xs), len(masks.CAMERA_NAMES)), dtype=np.float32)
        print(f"[band] {embodiment}: grid {len(ys)}x{len(xs)} = {vis.shape[0] * vis.shape[1]} cells "
              f"(x [{xs[0]:+.2f},{xs[-1]:+.2f}], y [{ys[0]:+.2f},{ys[-1]:+.2f}]), "
              f"table_z={table_z:.3f}")
        for iy, y in enumerate(ys):
            t_row = time.perf_counter()
            for ix, x in enumerate(xs):
                probe.set_pose(sapien.Pose([float(x), float(y), table_z + half_len + 0.001],
                                           [1, 0, 0, 0]))
                for _ in range(SETTLE_STEPS):
                    task.scene.step()
                visibility, _tok = masks.item_masks(task, [spec], [])
                vis[iy, ix] = visibility[0]
            print(f"[band] row y={y:+.2f}: vis_max mean {vis[iy].max(axis=1).mean():.3f} "
                  f"({time.perf_counter() - t_row:.1f}s)")
        vis_max = vis.max(axis=2)
        covered = vis_max >= VIS_THRESH
        rect, band = band_rect(covered, xs, ys)
        if rect is None:
            raise SystemExit(f"[band] {embodiment}: NO cell reaches vis {VIS_THRESH} — band empty")
        print(f"[band] {embodiment} covered ({int(covered.sum())}/{covered.size} cells):\n"
              + _ascii_map(covered, xs, ys))
        print(f"[band] {embodiment} band ({int(band.sum())} cells):\n"
              + _ascii_map(band, xs, ys, char="o"))
        print(f"[band] {embodiment} RECT x [{rect[0]:+.3f}, {rect[1]:+.3f}] "
              f"y [{rect[2]:+.3f}, {rect[3]:+.3f}]")

        grid_xy = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"visibility_band_{embodiment}.npz"
        np.savez(
            npz_path,
            xs=xs, ys=ys, grid_xy=grid_xy,
            vis=vis, vis_max=vis_max, covered=covered, band=band,
            rect=np.asarray(rect, dtype=np.float64),
            embodiment=embodiment, cell=CELL_M, thresh=VIS_THRESH,
            probe_r=PROBE_R, probe_h=PROBE_H, task="smoke_items",
        )
        print(f"[band] wrote {npz_path} ({time.perf_counter() - t0:.0f}s total)")
        return {"embodiment": embodiment, "rect": rect, "npz": str(npz_path),
                "n_covered": int(covered.sum()), "n_band": int(band.sum())}
    finally:
        env.close()


# ---------------------------------------------------------------------------
# --validate: the F5 re-measure with the band active
# ---------------------------------------------------------------------------


def validate_embodiment(embodiment: str, seeds: list[int], out_path: Path,
                        task_cfg: dict) -> dict:
    """Setup-only P1-style episodes; per-item t=0 visibility -> JSONL rows.

    Resume is per ITEM row (episodes rebuild but never duplicate rows); the
    spawn side mirrors probes/p1.py (rng = RandomState(seed + 4241)).
    """
    from main.item_envs import masks
    from main.item_envs.probes._common import append_row, build_env, done_keys
    from main.item_envs.spawn import spawn_candidates

    hosts = list(task_cfg["host_tasks"])
    key_fields = ("host", "embodiment", "seed", "item_id")
    done = done_keys(out_path, key_fields)
    ran = 0
    for host in hosts:
        for seed in sorted(int(s) for s in seeds):
            env = None
            try:
                env = build_env(host, embodiment, seed)
                task = env.task
                rng = np.random.RandomState(int(seed) + 4241)  # p1 parity
                specs, report = spawn_candidates(env, task_cfg, rng)
                visibility, _tok = masks.item_masks(task, specs, [])
                pl_of = {p["item_id"]: p for p in report.get("placements", [])}
                for i, spec in enumerate(specs):
                    key = (host, embodiment, seed, spec.item_id)
                    if key in done:
                        continue
                    vis = [float(v) for v in visibility[i]] if visibility.size else []
                    pl = pl_of.get(spec.item_id) or {}
                    append_row(out_path, {
                        "host": host, "embodiment": embodiment, "seed": int(seed),
                        "item_id": spec.item_id, "category": spec.category,
                        "cls": pl.get("cls"),
                        "vis_band": report.get("vis_band"),
                        "vis_cov": report.get("vis_cov"),
                        "xyz": pl.get("xyz"),
                        "corridor_t": pl.get("corridor_t"),
                        "lateral": pl.get("lateral"),
                        "t_fallback": pl.get("t_fallback"),
                        "vis_fallback": pl.get("vis_fallback"),
                        "n_cam": len(vis), "vis": vis,
                        "vis_max": max(vis) if vis else 0.0,
                    })
                    done.add(key)
                ran += 1
                print(f"[validate] {host}/{embodiment}/s{seed}: k={len(specs)} "
                      f"band={report.get('vis_band')} cov={report.get('vis_cov')}")
            except Exception as exc:
                print(f"[validate] {host}/{embodiment}/s{seed} FAILED: "
                      f"{type(exc).__name__}: {exc}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
    return {"embodiment": embodiment, "episodes": ran, "rows": str(out_path)}


def validate_summary(paths: list[Path]) -> dict:
    """Per-class fraction of items with vis_max >= VIS_THRESH (the F5 gate)."""
    from main.item_envs.probes._common import load_rows

    rows = [r for p in paths for r in load_rows(p)]
    per_class: dict[str, dict] = {}
    for r in rows:
        c = per_class.setdefault(str(r.get("cls")), {"n": 0, "n_vis": 0, "vis_sum": 0.0})
        c["n"] += 1
        c["n_vis"] += int(float(r.get("vis_max", 0.0)) >= VIS_THRESH)
        c["vis_sum"] += float(r.get("vis_max", 0.0))
    summary = {
        "n_rows": len(rows),
        "thresh": VIS_THRESH,
        "per_class": {
            cls: {
                "n": c["n"],
                "frac_ge_thresh": (c["n_vis"] / c["n"]) if c["n"] else None,
                "mean_vis_max": (c["vis_sum"] / c["n"]) if c["n"] else None,
            }
            for cls, c in sorted(per_class.items())
        },
    }
    print("[validate] summary: " + json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embodiment", required=True,
                    choices=[*EMBODIMENTS, "both"])
    ap.add_argument("--out", type=Path, default=CONFIGS_DIR,
                    help="band npz directory (default: the item_envs configs dir)")
    ap.add_argument("--validate", action="store_true",
                    help="run the F5 re-measure instead of the band sweep")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(20)),
                    help="validate mode: episode seeds (default 0..19)")
    ap.add_argument("--task-yaml", type=Path, default=T1_YAML)
    args = ap.parse_args()

    out_dir = args.out.resolve()  # BEFORE Env creation: RoboTwin chdirs
    embodiments = list(EMBODIMENTS) if args.embodiment == "both" else [args.embodiment]

    if not args.validate:
        results = [sweep_embodiment(emb, out_dir) for emb in embodiments]
        for r in results:
            print("BAND_RECT " + json.dumps(r, sort_keys=True))
        return 0

    import yaml

    with open(args.task_yaml, "r", encoding="utf-8") as fh:
        task_cfg = yaml.safe_load(fh) or {}
    if not (task_cfg.get("objects") or {}).get("off_path_visible_only", False):
        print("[validate] WARNING: objects.off_path_visible_only is not set in "
              f"{args.task_yaml} — placements will NOT use the band")
    val_dir = out_dir / "validate"
    val_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for emb in embodiments:
        paths.append(val_dir / f"visibility_rows_{emb}.jsonl")
        validate_embodiment(emb, list(args.seeds), paths[-1], task_cfg)
    validate_summary(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
