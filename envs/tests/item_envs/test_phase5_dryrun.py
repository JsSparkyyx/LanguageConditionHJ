"""Phase 5 dry run (gpu, GPU node only — NOT run on the login node).

3 nominal-only collection episodes per item task x both embodiments into a
FRESH StepBuffer + ItemSidecarBuffer pair, with the HoloBrain encoder
running — the full Phase-5 path: make_training_env's registry routing,
RiskTickController, ItemMarginHFn, the lockstep writer, episodes.jsonl.

For each episode an ItemEpisodeRunner run with the SAME seed (render="none",
its own out dir) produces the reference trace.npz; the sidecar's margins
for that episode's rows must equal the trace's margins EXACTLY (difference
0), at the same slot indices as the StepBuffer's episode rows.
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

        # reference episode: the runner's own trace.npz at the same seed
        runner = ItemEpisodeRunner(
            task, host=host, embodiment=embodiment, seed=seed,
            render="none", out_root=tmp_path / "ref",
        )
        meta, error = runner.run()
        assert error is None, f"{task}/{embodiment}/ep{ep}: {error}"
        ref = np.load(runner.out_dir / "trace.npz")

        # the sidecar rows for this episode sit at the same slot indices as
        # the StepBuffer's episode_id rows (lockstep), and their margins
        # columns (rule table order) match the trace EXACTLY
        ep_slots = np.nonzero(
            np.asarray(buf.arrays["episode_id"][: buf.n]) == episode_id
        )[0]
        ref_margins = np.asarray(ref["margins"], dtype=np.float64)
        got = np.asarray(
            sidecar.arrays["margins"][ep_slots][:, : ref_margins.shape[1]],
            dtype=np.float64,
        )
        assert got.shape == ref_margins.shape, (
            f"{task}/{embodiment}/ep{ep}: sidecar rows {got.shape} vs "
            f"trace margins {ref_margins.shape}"
        )
        np.testing.assert_array_equal(
            got, ref_margins,
            err_msg=f"{task}/{embodiment}/ep{ep}: sidecar margins diverge "
                    "from trace.npz",
        )
        # rule_valid marks exactly the rule-table width on those rows
        rv = np.asarray(sidecar.arrays["rule_valid"][ep_slots])
        assert rv[:, : ref_margins.shape[1]].all()
        assert not rv[:, ref_margins.shape[1]:].any()

    # lockstep end state: the pair advanced together
    assert (buf.n, buf.head) == (sidecar.n, sidecar.head)
    # episode tables: one JSONL record per episode, keyed by episode_id
    lines = (tmp_path / "item_buffer" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == EPISODES_PER_TASK
    ids = sorted(json.loads(line)["episode_id"] for line in lines)
    assert ids == list(range(EPISODES_PER_TASK))
