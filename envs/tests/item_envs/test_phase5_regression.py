"""Phase 5 regression (gpu, GPU node only — NOT run on the login node).

Reruns the Phase-0 reference episode (place_empty_cup, seed_base 0, both
embodiments) through RolloutController with DEFAULT args — the h_fn /
controller_cls hooks must not change the default path. Mirrors
scripts/item_envs/regression_trace.py (itself a mirror of eval_ckpt.py's
--nominal path): FrozenConfig defaults + obstacle_mode "on_path",
build_models(cfg, None) nominal encoder, episode_seed(0, 0, emb_idx, 0),
the 10-attempt x 5-retry spawn loop, max_steps = 30 s * 250.

WHY NOT BYTE-IDENTITY (redesign 2026-10-02): the original test asserted
element-for-element h equality against docs/item_envs/regression/<emb>/
trace.npz. The reference is NOT reproducible — not across nodes, not even
across two back-to-back processes on the SAME node:

- p5-verify 22611951 (batch gpuA40x4, current code): ur5 t shape (142,)
  vs ref (143,); franka h 119/149 mismatched, max 0.8437 h*20 (4.22 cm).
- p5-regdiag 22624018 (interactive gpub001, worktree at 14c6180 — the
  exact pre-Phase-5 code that generated the references): ur5 92/143, max
  0.2502 (1.25 cm); franka 119/149, max 0.8845 (4.42 cm); head (>=5) and
  tail (8) elements bit-identical. Phase 5 was NOT the cause.
- p5-determ 22624498 (gpub001, ONE job, 14c6180, two separate processes):
  every h pairing splits at tick 30 (bit-onset 30; >1e-3 h*20 by ticks
  31-34) and never reconverges — ur5 run1 vs run2 max 1.27 h*20 (6.35 cm);
  franka run1 vs run2 differs even in tick COUNT (149 vs 150). The scene
  is NOT the source: obstacle xyz/half and used_seed are bit-identical
  across all runs and nodes (numpy seeding), while cuRobo's plans differ
  run-to-run from the very first plan (ur5 plan #1 n_wp 75/76/77) — GPU
  planner nondeterminism. Execution tracks the plan tightly enough that h
  stays bit-identical for 30 ticks (1.2 s), then trajectory chaos wins.

So the test asserts the REPRODUCIBLE contract instead (observed spreads in
parentheses; h is stored h*20, 20/m):

1. scene parity — obstacle xyz/half EXACTLY equal to the reference meta
   (bit-identical in every observed run, any node);
2. cadence — t uniform dt = 0.04 s with t[0] = 0.04 (any length);
3. tick count — |n_new - n_ref| <= 2 (observed same-node spread 1);
4. deterministic head — first tick with |h_new - h_ref| > 1e-3 (50 µm) is
   >= 15 (observed onsets 31-34; a scaling/structural h bug breaks tick 0);
5. episode statistics — min h within +-0.03 m of the reference (chaotic
   for franka: 4 measured samples -3.95/-3.88/-2.67/-1.74 cm; ur5 stable,
   2.2 mm spread — the depth guard is the coarse backstop; the sharp
   guards are the head and the mean), mean h within +-0.01 m (observed
   <= 4.7 mm);
6. outcome — success flag equal (False in all observed runs).

h<0 tick counts (47-55 ur5, 57-70 franka across reruns) are too noisy to
assert. A Phase-5 hook that merely re-seeds RNG on the default path is
indistinguishable from planner noise below trajectory level — the guards
above are the strongest statements that survive the measured noise floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()  # main.train.* live in the sibling CEHJ copy

pytestmark = pytest.mark.gpu

REGRESSION_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "item_envs" / "regression"
)
EMBODIMENTS = ["ur5-wsg", "franka-panda"]
EVAL_CUTOFFS_S = (18.0, 30.0)  # eval_ckpt.py --cutoffs default

TICK_DT = 0.04
TICK_COUNT_TOL = 2      # observed same-node spread: 1
PREFIX_ATOL = 1e-3      # h*20 units = 50 µm; scaling bugs are off by cm..x20
PREFIX_MIN_TICKS = 15   # divergence never observed before tick 30
MIN_H_TOL_M = 0.03      # observed spread: 2.2 mm (ur5) / 21.4 mm (franka, 4 samples)
MEAN_H_TOL_M = 0.01     # observed spread: <= 4.7 mm


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_phase0_h_trace_robust(embodiment):
    import dataclasses

    from main.train.config import FrozenConfig
    from main.train.eval_ckpt import (
        ALL_EMBODIMENTS,
        build_models,
        episode_seed,
        set_global_seeds,
    )
    from main.train.rollout import RolloutController

    ref = np.load(REGRESSION_DIR / embodiment / "trace.npz")
    meta = json.loads((REGRESSION_DIR / embodiment / "meta.json").read_text())

    cfg = FrozenConfig()  # eval_ckpt.py:481
    cfg.obstacle_mode = "on_path"  # eval_ckpt.py:494
    seed = episode_seed(0, 0, ALL_EMBODIMENTS.index(embodiment), 0)
    cfg_ep = dataclasses.replace(cfg, task="place_empty_cup", embodiment=embodiment)
    max_steps = int(max(EVAL_CUTOFFS_S) * 250.0)

    kind, encoder, kin_fn, model_kw = build_models(cfg, None)
    assert kind == "nominal"
    kin = kin_fn(embodiment)

    # spawn-retry loop mirrored from regression_trace.py / eval_ckpt.py —
    # DEFAULT RolloutController args: no h_fn, no controller_cls
    ro = trace = None
    for attempt in range(10):
        s0 = seed + attempt * 10_000_000
        for retry in range(5):
            set_global_seeds(s0 + retry * 1000)
            cand = RolloutController(
                cfg_ep, s0 + retry * 1000, mode="filtered",
                encoder=encoder, kin=kin, video=None, max_steps=max_steps,
                **model_kw,
            )
            if cand.env.obstacle is not None:
                ro = cand
                break
            cand.env.close()
        if ro is None:
            continue
        trace = ro.run()
        if trace["h"]:
            break
        ro = None
    assert ro is not None and trace is not None and trace["h"], (
        f"{embodiment}: no valid episode after 10 attempts"
    )

    h_new = np.asarray(trace["h"], dtype=np.float64)
    t_new = np.asarray(trace["t"], dtype=np.float64)
    h_ref = np.asarray(ref["h_scaled"], dtype=np.float64)
    t_ref = np.asarray(ref["t"], dtype=np.float64)
    h_ref_m = np.asarray(ref["h_m"], dtype=np.float64)

    # 1. scene parity — numpy seeding is platform-stable: EXACT equality
    np.testing.assert_array_equal(
        np.asarray(ro.env.obstacle_xyz, dtype=np.float64),
        np.asarray(meta["obstacle_xyz"], dtype=np.float64),
        err_msg=f"{embodiment}: obstacle pose changed (scene parity broken)",
    )
    np.testing.assert_array_equal(
        np.asarray(ro.env.obstacle_half, dtype=np.float64),
        np.asarray(meta["obstacle_half"], dtype=np.float64),
        err_msg=f"{embodiment}: obstacle extents changed (scene parity broken)",
    )

    # 2. cadence: uniform 0.04 s ticks regardless of episode length
    assert t_new[0] == pytest.approx(TICK_DT, abs=1e-9)
    np.testing.assert_allclose(
        np.diff(t_new), TICK_DT, atol=1e-9,
        err_msg=f"{embodiment}: tick cadence is not uniform {TICK_DT} s",
    )

    # 3. tick count within the observed nondeterminism spread
    assert abs(len(t_new) - len(t_ref)) <= TICK_COUNT_TOL, (
        f"{embodiment}: {len(t_new)} ticks vs reference {len(t_ref)} "
        f"(> {TICK_COUNT_TOL} apart — not planner jitter)"
    )

    # 4. deterministic head: the episode must start on the reference
    # trajectory (scene + plan #1 + the first ~0.6 s of execution)
    n_common = min(len(h_new), len(h_ref))
    over = np.nonzero(np.abs(h_new[:n_common] - h_ref[:n_common]) > PREFIX_ATOL)[0]
    onset = int(over[0]) if len(over) else n_common
    assert onset >= PREFIX_MIN_TICKS, (
        f"{embodiment}: h diverges from the reference at tick {onset} "
        f"(< {PREFIX_MIN_TICKS}) — the deterministic episode head changed "
        "(scene, h scaling, or hook semantics)"
    )

    # 5. episode statistics within the measured nondeterminism spread
    min_new_m = float(h_new.min()) / 20.0
    mean_new_m = float(h_new.mean()) / 20.0
    assert abs(min_new_m - float(h_ref_m.min())) <= MIN_H_TOL_M, (
        f"{embodiment}: min h {min_new_m:.4f} m vs reference "
        f"{float(h_ref_m.min()):.4f} m (> {MIN_H_TOL_M} m apart)"
    )
    assert abs(mean_new_m - float(h_ref_m.mean())) <= MEAN_H_TOL_M, (
        f"{embodiment}: mean h {mean_new_m:.4f} m vs reference "
        f"{float(h_ref_m.mean()):.4f} m (> {MEAN_H_TOL_M} m apart)"
    )

    # 6. outcome parity
    assert bool(trace.get("success", False)) == bool(ref["success"]), (
        f"{embodiment}: success={trace.get('success')} vs reference "
        f"{bool(ref['success'])}"
    )
