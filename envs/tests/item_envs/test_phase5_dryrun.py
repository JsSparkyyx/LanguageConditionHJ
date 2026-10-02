"""Phase 5 dry run (gpu, GPU node only — NOT run on the login node).

3 nominal-only collection episodes per item task x both embodiments into a
FRESH StepBuffer + ItemSidecarBuffer pair, with the HoloBrain encoder
running — the full Phase-5 path: make_training_env's registry routing,
RiskTickController, ItemMarginHFn, the lockstep writer, episodes.jsonl.

For each episode an ItemEpisodeRunner run with the SAME seed (render="none",
its own out dir) produces the reference trace.npz. What MUST hold:

- STRUCTURAL (hard, per episode — these cannot be caused by planner
  noise, only by wiring bugs): the rollout captured ticks; the lockstep
  writer's end_episode teardown is clean (every offered sidecar row paired
  with a StepBuffer entry); rule_valid marks exactly the trace's rule-table
  width on the episode's rows; the two rings end with identical n/head;
  one episodes.jsonl record per episode.
- TRAJECTORY (per episode PAIR, chaos-tolerant): the sidecar margins at
  the episode's StepBuffer slots and the trace's margins START as the same
  physical episode — tick counts within TICK_COUNT_TOL, and the first
  divergence beyond HEAD_ATOL (+inf == +inf counts as equal) no earlier
  than HEAD_MIN_ONSET. At least MIN_MATCHING_PAIRS of the
  EPISODES_PER_TASK pairs must match per (task, embodiment); non-matching
  pairs are reported with their signatures, not asserted.

WHY NOT STRICT EQUALITY (redesign 2026-10-02): the original test required
the margins to equal the trace EXACTLY. That is unachievable — the
comparison crosses TWO Env constructions, and cuRobo planning is not
run-deterministic even on one node (p5-determ 22624498: identical
back-to-back episodes on ONE node split at tick ~30 and can differ in tick
count; see test_phase5_regression.py's docstring). Strict-equality failure
record, all noise-signed:

- p5-verify 22611951 (batch): selective_avoid ur5 (133,12)v(134,12);
  selective_avoid franka (150,12)v(157,12); granular_pour ur5
  (152,3)v(181,3); granular_pour franka equal-shape but 100% content
  mismatch (max 2.82); push_zone ur5 (84,8)v(85,8); push_zone franka
  (115,8)v(114,8).
- p5v-pieces 22625346/22625424 (gpub001, current code): selective_avoid
  ur5 (131,7)v(134,7); granular_pour ur5 (212,3)v(210,3); push_zone ur5
  equal shape, ulp-level from tick 0 (max 6 cm over the episode);
  selective_avoid franka ulp-drift + chaotic +inf locations; push_zone
  franka (80,8)v(81,8); granular_pour franka a chaotic plan-failure storm
  ("all fallbacks failed", 250 v 153 ticks) — a legitimately different
  episode outcome, not a wiring signal.

(selective_avoid's rule-table width also changed 12 -> 7 between the two
jobs — the rules code moved; within each run sidecar width == trace width,
which is what rule_valid asserts.)

A wiring bug (margins column order, scene parity, h_fn semantics) breaks
the margins at tick ~0 in EVERY pair; chaos breaks pairs at random onsets
and counts. The same-trajectory window ends at the task's first replan
boundary — plan #1 spans ~30 ticks for place_empty_cup but only ~4-5 for
selective_avoid (measured onsets 4/5/12 on 22625594), so the onset gate
sits at 3, far below any observed noise onset and far above the tick-0/1
signature of a real bug. Episode LENGTHS diverge chaotically once the
trajectories decorrelate (plan-failure storms: 253v206, 157v281 ticks on
22625594) — the count gate is deliberately loose (8) and only a
systematic capture bias across ALL pairs would trip it. Requiring >=1
matching pair per combo keeps the wiring guards at full strength while
absorbing the measured nondeterminism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()  # main.train.* live in the sibling CEHJ copy

pytestmark = pytest.mark.gpu

EPISODES_PER_TASK = 3
# (task, host) — T1 on its first host; T2/T3 are robotwin-kind (host None)
TASKS = [
    ("selective_avoid", "place_container_plate"),
    ("granular_pour", None),
    ("push_zone", None),
]
EMBODIMENTS = ["ur5-wsg", "franka-panda"]

TICK_COUNT_TOL = 8      # chaotic lengths (plan-failure storms) run 47-124; systematic capture bias would hit every pair
HEAD_ATOL = 5e-3        # observed early drift <= ~1e-4; permuted columns are cm..m off
HEAD_MIN_ONSET = 3      # observed noise onsets 4-34; a wiring bug breaks tick 0/1
MIN_MATCHING_PAIRS = 1


def _pair_signature(got: np.ndarray, ref: np.ndarray) -> tuple[bool, str]:
    """Do `got` (sidecar margins at the episode's slots) and `ref` (trace
    margins) START as the same physical episode? (matches, diagnostic)."""
    n_got, n_ref = got.shape[0], ref.shape[0]
    count_ok = abs(n_got - n_ref) <= TICK_COUNT_TOL
    c = min(n_got, n_ref)
    close = np.isclose(got[:c], ref[:c], rtol=0, atol=HEAD_ATOL)
    bad = np.nonzero(~close.all(axis=1))[0]
    onset = int(bad[0]) if len(bad) else c
    head_ok = onset >= HEAD_MIN_ONSET
    diag = (
        f"n={n_got}v{n_ref}{'' if count_ok else ' COUNT'}"
        f" onset={onset}{'' if head_ok else ' HEAD'}"
    )
    return count_ok and head_ok, diag


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("task,host", TASKS)
def test_item_collection_dryrun(task, host, embodiment, tmp_path):
    import dataclasses

    from main.item_envs.controller import RiskTickController
    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.item_envs.margin_hook import wire_item_episode
    from main.item_envs.masks import token_grid
    from main.item_envs.runner import ItemEpisodeRunner
    from main.train.buffer import StepBuffer
    from main.train.collect import CKPT_DIR, make_kinematics
    from main.train.config import FrozenConfig
    from main.train.rollout import RolloutController
    from main.network.encoder import HoloBrainEncoder

    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, embodiment)
    _, n_tok = token_grid()

    buf = StepBuffer(tmp_path / "buffer", capacity=4096, header={"dryrun": 1})
    sidecar = ItemSidecarBuffer.for_step_buffer(buf, n_tok=n_tok)
    cfg = FrozenConfig(task=task, embodiment=embodiment, n_episodes=EPISODES_PER_TASK)

    matches, diags = 0, []
    for ep in range(EPISODES_PER_TASK):
        seed = 1000 + ep
        episode_id = ep

        # collection episode through the Phase-5 wiring (collect.py's path)
        cfg_ep = dataclasses.replace(cfg, task=task, embodiment=embodiment)
        if host is not None:
            cfg_ep.item_host = host
        ro = RolloutController(
            cfg_ep, seed=seed, mode="collect",
            encoder=encoder, kin=kin,
            policy_enc=None, actor=None, critics=None, filter_active=False,
            buf=buf, episode_id=episode_id, max_steps=7500,
            controller_cls=RiskTickController,
        )
        wire_item_episode(ro, sidecar)
        trace = ro.run()
        assert trace["h"], f"{task}/{embodiment}/ep{ep}: no ticks captured"
        # teardown: the done-flush must have finalized the last capture —
        # a row left queued here means the rollout dropped the final tick
        n_step, n_side = ro.buf.end_episode()
        assert n_step == n_side

        # reference episode: the runner's own trace.npz at the same seed
        runner = ItemEpisodeRunner(
            task, host=host, embodiment=embodiment, seed=seed,
            render="none", out_root=tmp_path / "ref",
        )
        meta, error = runner.run()
        assert error is None, f"{task}/{embodiment}/ep{ep}: {error}"
        ref = np.load(runner.out_dir / "trace.npz")

        # the sidecar rows for this episode sit at the same slot indices as
        # the StepBuffer's episode_id rows (lockstep)
        ep_slots = np.nonzero(
            np.asarray(buf.arrays["episode_id"][: buf.n]) == episode_id
        )[0]
        ref_margins = np.asarray(ref["margins"], dtype=np.float64)
        got = np.asarray(
            sidecar.arrays["margins"][ep_slots][:, : ref_margins.shape[1]],
            dtype=np.float64,
        )
        # rule_valid marks exactly the rule-table width on those rows
        rv = np.asarray(sidecar.arrays["rule_valid"][ep_slots])
        assert rv[:, : ref_margins.shape[1]].all()
        assert not rv[:, ref_margins.shape[1]:].any()

        same, diag = _pair_signature(got, ref_margins)
        matches += same
        diags.append(f"ep{ep}: {diag}")
        print(f"[dryrun] {task}/{embodiment}/ep{ep}: {'MATCH' if same else 'diverged'} ({diag})")

    assert matches >= MIN_MATCHING_PAIRS, (
        f"{task}/{embodiment}: no episode pair matched the reference "
        f"trajectory contract ({'; '.join(diags)}) — a wiring bug breaks "
        "every pair (head onset ~0 / column mismatch); chaos breaks pairs "
        "at random onsets"
    )

    # lockstep end state: the pair advanced together
    assert (buf.n, buf.head) == (sidecar.n, sidecar.head)
    # episode tables: one JSONL record per episode, keyed by episode_id
    lines = (tmp_path / "item_buffer" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == EPISODES_PER_TASK
    ids = sorted(json.loads(line)["episode_id"] for line in lines)
    assert ids == list(range(EPISODES_PER_TASK))
