"""P3 push-coast probe (PLAN v3 §6 Phase 3, T3 side) — HEADLESS, login-node safe.

A kinematic pusher plate drives a 5 cm, 0.1 kg box across a table at
v ∈ {0.1, 0.3, 0.6} m/s for 0.15 m, then stops dead; the box coasts on
Coulomb friction (static = dynamic ∈ {0.3, 0.6, 0.9}). Measured: coast
distance after the pusher stops, and box yaw/tilt at rest.

Gate (PLAN): coast > 0 in every cell, coast increases with v at fixed
friction, coast decreases with friction at fixed v. Medians over the seeds
are compared; a 2% relative tolerance absorbs solver noise (documented —
PLAN's wording is qualitative). The energy estimate coast ≈ v²/(2μg) gives
e.g. ~6 cm at (0.6, 0.3) and ~0.6 mm at (0.1, 0.9).

HEADLESS by design: the scene is CPU-only
(``sapien.Scene([physx.PhysxCpuSystem()])`` — the only scene that works
without Vulkan, recon.md). No GPU, no planner, no RoboTwin imports; the
whole 3x3x3 grid runs on the login node in seconds. ``embodiments`` is
accepted for probe-interface uniformity and ignored.

Rows: <out_dir>/p3_rows.jsonl keyed (v, mu, seed); resumable. Seed controls
only the pusher's lateral jitter U(±2 mm) (RandomState(seed + 4241), the
spawn convention). Keyframes: none (no renderer headless).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._common import append_row, done_keys, load_rows, shard_cells

NAME = "p3"
KEY_FIELDS = ("v", "mu", "seed")

DT = 1.0 / 250.0  # RoboTwin physics frequency parity
PUSH_DIST = 0.15
BOX_HALF = 0.025  # 5 cm cube
BOX_MASS = 0.1
SETTLE_STEPS = 50
REST_SPEED_TOL = 1e-3  # m/s and rad/s: "at rest"
REST_MAX_STEPS = 750  # 3 s to come to rest after the stop
DEFAULT_V = (0.1, 0.3, 0.6)
DEFAULT_MU = (0.3, 0.6, 0.9)
DEFAULT_SEEDS = (0, 1, 2)

# gate tolerances (see module docstring)
POS_TOL_M = 1e-6
MONO_REL_TOL = 0.02


# ---------------------------------------------------------------------------
# one cell (headless physics)
# ---------------------------------------------------------------------------


def _simulate(v: float, mu: float, seed: int, push_dist: float = PUSH_DIST) -> dict:
    """One push-coast cell; pure result dict (no file I/O)."""
    import sapien
    import sapien.physx as physx

    rng = np.random.RandomState(int(seed) + 4241)
    jitter_y = float(rng.uniform(-0.002, 0.002))

    scene = sapien.Scene([physx.PhysxCpuSystem()])
    scene.set_timestep(DT)
    material = scene.create_physical_material(float(mu), float(mu), 0.0)

    table = scene.create_actor_builder()
    table.add_box_collision(half_size=[0.4, 0.4, 0.02], material=material)
    table.set_physx_body_type("static")
    table.set_initial_pose(sapien.Pose([0, 0, -0.02]))
    table.build(name="table")

    box_b = scene.create_actor_builder()
    box_b.add_box_collision(half_size=[BOX_HALF] * 3, material=material)
    box_b.set_initial_pose(sapien.Pose([0, 0, BOX_HALF + 1e-4]))
    box = box_b.build(name="box")
    box_comp = box.find_component_by_type(physx.PhysxRigidDynamicComponent)
    box_comp.mass = BOX_MASS

    pusher_half = [0.005, 0.04, 0.024]
    push_x0 = -(BOX_HALF + pusher_half[0] + 0.002)
    push_z = BOX_HALF
    pusher_b = scene.create_actor_builder()
    pusher_b.add_box_collision(half_size=pusher_half, material=material)
    pusher_b.set_physx_body_type("kinematic")
    pusher_b.set_initial_pose(sapien.Pose([push_x0, jitter_y, push_z]))
    pusher = pusher_b.build(name="pusher")
    pusher_comp = pusher.find_component_by_type(physx.PhysxRigidDynamicComponent)

    def drive(x: float) -> None:
        pusher_comp.kinematic_target = sapien.Pose([float(x), jitter_y, push_z])

    def box_state():
        pose = box.get_pose()
        return (
            np.asarray(pose.p, dtype=np.float64).reshape(3).copy(),
            np.asarray(pose.q, dtype=np.float64).reshape(4).copy(),
            float(np.linalg.norm(np.asarray(box_comp.linear_velocity, dtype=np.float64))),
            float(np.linalg.norm(np.asarray(box_comp.angular_velocity, dtype=np.float64))),
        )

    drive(push_x0)
    x_box_0 = float(box.get_pose().p[0])
    for _ in range(SETTLE_STEPS):
        drive(push_x0)
        scene.step()
    settle_drift = abs(float(box.get_pose().p[0]) - x_box_0)

    # push phase: kinematic target moves at v until it covers push_dist
    n_push = int(np.ceil(push_dist / (float(v) * DT)))
    for i in range(1, n_push + 1):
        drive(push_x0 + min(float(v) * i * DT, push_dist))
        scene.step()
    p_stop, _q_stop, v_stop, _w_stop = box_state()
    x_stop = float(p_stop[0])

    # coast phase: pusher held; step until the box rests (or the step cap)
    rest_steps = 0
    came_to_rest = False
    for _ in range(REST_MAX_STEPS):
        drive(push_x0 + push_dist)
        scene.step()
        rest_steps += 1
        _p, _q, lv, av = box_state()
        if lv < REST_SPEED_TOL and av < REST_SPEED_TOL:
            came_to_rest = True
            break
    p_end, q_end, _lv, _av = box_state()
    x_final = float(p_end[0])

    w, x, y, z = (float(c) for c in q_end)
    yaw = float(np.rad2deg(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    up_z = 1 - 2 * (x * x + y * y)  # world-z of the body +z axis
    tilt = float(np.rad2deg(np.arccos(float(np.clip(up_z, -1.0, 1.0)))))

    return {
        "v": float(v),
        "mu": float(mu),
        "seed": int(seed),
        "coast_m": x_final - x_stop,
        "x_stop": x_stop,
        "x_final": x_final,
        "disp_total_m": x_final - x_box_0,
        "v_stop_box": v_stop,
        "yaw_deg": yaw,
        "tilt_deg": tilt,
        "rest_steps": rest_steps,
        "came_to_rest": came_to_rest,
        "settle_drift_m": settle_drift,
        "pusher_jitter_y": jitter_y,
        "push_dist_cmd": float(push_dist),
        "pusher_travel_m": float(n_push) * float(v) * DT,
    }


# ---------------------------------------------------------------------------
# gates (pure — unit-tested without sapien)
# ---------------------------------------------------------------------------


def cell_medians(rows: list[dict], field: str = "coast_m") -> dict:
    """{(v, mu): median field} over complete rows."""
    cells: dict[tuple[float, float], list[float]] = {}
    for r in rows:
        if r.get("error") is not None:
            continue
        cells.setdefault((float(r["v"]), float(r["mu"])), []).append(float(r[field]))
    return {k: float(np.median(v)) for k, v in sorted(cells.items())}


def evaluate_gates(rows: list[dict]) -> dict:
    """PLAN P3 gate on per-cell medians; pure (importable on the login node)."""
    med = cell_medians(rows)
    vs = sorted({k[0] for k in med})
    mus = sorted({k[1] for k in med})
    pairs_v = [
        {"mu": mu, "v_lo": v1, "v_hi": v2, "coast_lo": med[(v1, mu)], "coast_hi": med[(v2, mu)]}
        for mu in mus
        for v1, v2 in zip(vs, vs[1:])
        if (v1, mu) in med and (v2, mu) in med
    ]
    pairs_mu = [
        {"v": v, "mu_lo": m1, "mu_hi": m2, "coast_lo": med[(v, m1)], "coast_hi": med[(v, m2)]}
        for v in vs
        for m1, m2 in zip(mus, mus[1:])
        if (v, m1) in med and (v, m2) in med
    ]
    coast_positive = bool(med) and all(m > POS_TOL_M for m in med.values())
    increases_with_v = all(
        p["coast_hi"] >= p["coast_lo"] * (1.0 - MONO_REL_TOL) for p in pairs_v
    )
    decreases_with_mu = all(
        p["coast_hi"] <= p["coast_lo"] * (1.0 + MONO_REL_TOL) for p in pairs_mu
    )
    return {
        "n_cells": len(med),
        "medians": {f"v={k[0]}_mu={k[1]}": m for k, m in med.items()},
        "coast_positive": coast_positive,
        "increases_with_v": bool(increases_with_v),
        "decreases_with_mu": bool(decreases_with_mu),
        "pairs_v": pairs_v,
        "pairs_mu": pairs_mu,
        "tolerances": {"pos_tol_m": POS_TOL_M, "mono_rel_tol": MONO_REL_TOL},
        "pass": bool(coast_positive and increases_with_v and decreases_with_mu),
    }


# ---------------------------------------------------------------------------
# probe interface
# ---------------------------------------------------------------------------


def run(out_dir, embodiments=None, seeds=None, shard="0/1",
        velocities=DEFAULT_V, frictions=DEFAULT_MU, **_ignored) -> dict:
    """Run the v x mu x seed grid (resumable). `embodiments` ignored (headless)."""
    out_dir = Path(out_dir)
    rows_path = out_dir / f"{NAME}_rows.jsonl"
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    cells = [(float(v), float(mu), int(s)) for v in velocities for mu in frictions for s in seeds]
    cells = shard_cells(cells, shard)
    done = done_keys(rows_path, KEY_FIELDS)
    ran, skipped = 0, 0
    for v, mu, seed in cells:
        if (v, mu, seed) in done:
            skipped += 1
            continue
        row = _simulate(v, mu, seed)
        append_row(rows_path, row)
        ran += 1
        print(
            f"[p3] v={v} mu={mu} seed={seed}: coast {1e2 * row['coast_m']:.2f} cm "
            f"(rest after {row['rest_steps']} steps, yaw {row['yaw_deg']:.1f} deg)"
        )
    return {"probe": NAME, "ran": ran, "skipped": skipped, "rows": str(rows_path)}


def summarize(out_dir) -> dict:
    """Gate evaluation over the rows on disk; writes <out_dir>/summary.json."""
    out_dir = Path(out_dir)
    rows = load_rows(out_dir / f"{NAME}_rows.jsonl")
    summary = {
        "probe": NAME,
        "n_rows": len(rows),
        "setup": {
            "box": f"{2 * BOX_HALF * 100:.0f} cm cube, {BOX_MASS} kg",
            "push_dist_m": PUSH_DIST,
            "dt": DT,
            "seeds_note": "seed = pusher lateral jitter U(±2 mm)",
        },
        **evaluate_gates(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        run(args.out_dir, seeds=args.seeds, shard=args.shard)
    print(json.dumps(summarize(args.out_dir), indent=2))
