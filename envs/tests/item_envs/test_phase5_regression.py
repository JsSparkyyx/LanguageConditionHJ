"""Phase 5 regression (gpu, GPU node only — NOT run on the login node).

Reruns the Phase-0 reference (place_empty_cup, seed_base 0, both
embodiments) through RolloutController with DEFAULT args — the h_fn /
controller_cls hooks must reproduce the pre-Phase-5 path byte for byte.
The h traces in docs/item_envs/regression/<embodiment>/trace.npz (produced
by envs/scripts/item_envs/regression_trace.py) must come back IDENTICAL.

Mirrors scripts/item_envs/regression_trace.py (itself a mirror of
eval_ckpt.py's --nominal path): FrozenConfig defaults + obstacle_mode
"on_path", build_models(cfg, None) nominal encoder, episode_seed(0, 0,
emb_idx, 0), the 10-attempt x 5-retry spawn loop, max_steps = 30 s * 250.
"""

from __future__ import annotations

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


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_phase0_h_trace_byte_identical(embodiment):
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
    np.testing.assert_array_equal(t_new, np.asarray(ref["t"]))
    # THE regression: h*h_scale traces identical, element for element
    np.testing.assert_array_equal(
        h_new,
        np.asarray(ref["h_scaled"]),
        err_msg=f"{embodiment}: h trace diverged from the Phase-0 reference",
    )
