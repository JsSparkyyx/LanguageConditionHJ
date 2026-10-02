"""P3 push-coast micro-grid (PLAN v3 §6 Phase 3 P3) — HEADLESS.

Three cells on the CPU-only scene (PhysxCpuSystem — the only scene that
works without Vulkan): coast > 0 everywhere, coast grows with v at fixed
friction, coast shrinks with friction at fixed v. Runs on the login node;
marked ``sim`` because it steps a PhysX scene (test_physics_sim.py
precedent), NOT ``gpu``.
"""

from __future__ import annotations

import pytest

sapien = pytest.importorskip("sapien")

from main.item_envs.probes import get_probe

pytestmark = pytest.mark.sim


def test_micro_grid_coast_ordering():
    p3 = get_probe("p3")
    slow = p3._simulate(0.3, 0.6, seed=0)
    fast = p3._simulate(0.6, 0.6, seed=0)
    grippy = p3._simulate(0.6, 0.3, seed=0)

    for name, row in (("slow", slow), ("fast", fast), ("grippy", grippy)):
        assert row["coast_m"] > 0.0, f"{name}: no coast: {row}"
        assert row["settle_drift_m"] < 1e-3, f"{name}: box drifted during settle: {row}"
        assert abs(row["tilt_deg"]) < 5.0, f"{name}: box tipped: {row}"

    # ordering both directions
    assert fast["coast_m"] > slow["coast_m"]  # v up -> coast up (mu = 0.6)
    assert grippy["coast_m"] > fast["coast_m"]  # mu down -> coast up (v = 0.6)

    # energy sanity: coast <= v^2/(2*mu*g) (pusher stop is lossy, never additive)
    assert fast["coast_m"] <= 0.6**2 / (2 * 0.6 * 9.81) + 1e-3


def test_run_and_summarize_one_cell(tmp_path):
    p3 = get_probe("p3")
    out = p3.run(tmp_path, seeds=(0,), velocities=(0.3,), frictions=(0.6,))
    assert out["ran"] == 1
    # resume: same cell is skipped
    out2 = p3.run(tmp_path, seeds=(0,), velocities=(0.3,), frictions=(0.6,))
    assert out2["ran"] == 0 and out2["skipped"] == 1
    s = p3.summarize(tmp_path)
    assert s["n_rows"] == 1
    assert (tmp_path / "summary.json").is_file()
