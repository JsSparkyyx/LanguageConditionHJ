"""Non-sim tests for ItemSidecarBuffer + LockstepItemWriter (Phase 5).

Lockstep contract with main.train.buffer.StepBuffer: same ring semantics,
same slot indices — after a wrap, slot i of either ring is the same
transition. Covered here: paired writes past a wrap, header persistence
across close/reopen, discard_to parity, row padding, for_step_buffer's
fresh/resume gates, and the episode-table JSONL.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()  # main.train.buffer lives in the sibling CEHJ copy

from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter
from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import Consequence, RuleSpec
from main.train.buffer import StepBuffer

DIMS = dict(k_max=3, r_max=4, g_max=2, n_cam=2, n_tok=5)


def make_pair(tmp_path, capacity=8):
    buf = StepBuffer(tmp_path / "buffer", capacity=capacity, header={"test": 1})
    sidecar = ItemSidecarBuffer.for_step_buffer(buf, **DIMS)
    return buf, sidecar


def make_row(i: int, k: int = 2, r: int = 2, g: int = 1) -> dict:
    """Synthetic sidecar row for transition i; margins[0] / item_pose[0,0]
    encode i so cross-ring pairing is checkable."""
    return {
        "item_pose": np.full((k, 7), i, dtype=np.float32),
        "item_vel": np.zeros((k, 6), dtype=np.float32),
        "item_obb": np.ones((k, 9), dtype=np.float32),
        "item_valid": np.ones(k, dtype=np.bool_),
        "held": np.array([-1, -1], dtype=np.int8),
        "robot_item_dist": np.full(k, 0.5, dtype=np.float32),
        "margins": np.array([i, -i][:r] + [0.0] * max(0, r - 2), dtype=np.float32),
        "rule_valid": np.ones(r, dtype=np.bool_),
        "region_T": np.stack([np.eye(4)] * g).astype(np.float32),
        "token_masks": np.zeros((k + g, DIMS["n_cam"], DIMS["n_tok"]), dtype=np.uint8),
        "visibility": np.zeros((k, DIMS["n_cam"]), dtype=np.float16),
    }


def make_entry(i: int, episode: int = 0) -> dict:
    """Minimal StepBuffer entry; h encodes the transition id."""
    return {
        "episode_id": np.int32(episode),
        "h": np.float32(i),
        "done": np.bool_(False),
    }


def test_lockstep_past_wrap(tmp_path):
    buf, sidecar = make_pair(tmp_path, capacity=8)
    lock = LockstepItemWriter(buf, sidecar)
    for i in range(20):  # 2.5x capacity: the ring wraps twice
        lock.offer(make_row(i))
        lock.append(make_entry(i))
    assert len(buf) == len(sidecar) == 8
    assert buf.head == sidecar.head == 20 % 8
    # slot parity: h and margins[0] encode the same transition id per slot.
    # Slot order is by ring position, not recency: writes 12..19 landed in
    # slots 4..7,0..3, so the raw arrays read [16..19, 12..15]
    h = np.asarray(buf.arrays["h"][:])
    m = np.asarray(sidecar.arrays["margins"][:, 0])
    np.testing.assert_array_equal(h, m)
    np.testing.assert_array_equal(
        h, np.array([16.0, 17.0, 18.0, 19.0, 12.0, 13.0, 14.0, 15.0])
    )
    # and the item rows at those slots carry the same ids
    np.testing.assert_array_equal(
        np.asarray(sidecar.arrays["item_pose"][:, 0, 0]), h
    )


def test_header_persistence_across_reopen(tmp_path):
    buf, sidecar = make_pair(tmp_path, capacity=8)
    lock = LockstepItemWriter(buf, sidecar)
    for i in range(11):
        lock.offer(make_row(i))
        lock.append(make_entry(i))
    buf.flush()
    sidecar.flush()
    del lock, buf, sidecar

    buf2 = StepBuffer(tmp_path / "buffer")
    sidecar2 = ItemSidecarBuffer(tmp_path / "item_buffer")
    assert (buf2.n, buf2.head) == (sidecar2.n, sidecar2.head) == (8, 3)
    assert sidecar2.k_max == DIMS["k_max"] and sidecar2.n_tok == DIMS["n_tok"]
    # rows survive the reopen (wrapped: slot 3 is the oldest, id 3)
    assert float(sidecar2.arrays["margins"][3, 0]) == 3.0
    # for_step_buffer resume parity: matching n/head reopens cleanly
    sidecar3 = ItemSidecarBuffer.for_step_buffer(buf2)
    assert sidecar3.n == 8


def test_discard_to_parity(tmp_path):
    buf, sidecar = make_pair(tmp_path, capacity=16)
    lock = LockstepItemWriter(buf, sidecar)
    for i in range(6):
        lock.offer(make_row(i))
        lock.append(make_entry(i))
    # success-only rollback: drop the tail (rows 3..5) on BOTH rings
    buf.discard_to(3)
    sidecar.discard_to(3)
    assert (buf.n, buf.head) == (sidecar.n, sidecar.head) == (3, 3)
    for i in range(2):  # appends continue in lockstep
        lock.offer(make_row(10 + i))
        lock.append(make_entry(10 + i))
    assert (buf.n, buf.head) == (sidecar.n, sidecar.head) == (5, 5)
    np.testing.assert_array_equal(
        np.asarray(buf.arrays["h"][:5]),
        np.asarray(sidecar.arrays["margins"][:5, 0]),
    )
    with pytest.raises(AssertionError):
        sidecar.discard_to(9)  # beyond n


def test_row_padding(tmp_path):
    buf, sidecar = make_pair(tmp_path, capacity=4)
    lock = LockstepItemWriter(buf, sidecar)
    lock.offer(make_row(0, k=1, r=1, g=1))  # short in every leading dim
    lock.append(make_entry(0))
    assert sidecar.arrays["item_valid"][0].tolist() == [True, False, False]
    assert sidecar.arrays["rule_valid"][0].tolist() == [True, False, False, False]
    # invalid item slots read +inf in robot_item_dist, 0 elsewhere
    assert np.isinf(sidecar.arrays["robot_item_dist"][0, 1:]).all()
    assert float(sidecar.arrays["robot_item_dist"][0, 0]) == 0.5
    assert not sidecar.arrays["token_masks"][0, 2:].any()
    with pytest.raises(ValueError, match="does not fit"):
        sidecar.append(make_row(1, k=DIMS["k_max"] + 1))  # over the maximum


def test_lockstep_append_without_row_raises(tmp_path):
    buf, sidecar = make_pair(tmp_path, capacity=4)
    lock = LockstepItemWriter(buf, sidecar)
    with pytest.raises(RuntimeError, match="no queued item row"):
        lock.append(make_entry(0))
    # validate-first protocol: the raise happens BEFORE either ring moves,
    # so the pair is still aligned (a step entry with no paired item row
    # would be unrecoverable)
    assert len(buf) == 0 and len(sidecar) == 0


def test_for_step_buffer_gates(tmp_path):
    buf = StepBuffer(tmp_path / "buffer", capacity=4, header={})
    buf.append(make_entry(0))
    with pytest.raises(ValueError, match="fresh buffer pair"):
        ItemSidecarBuffer.for_step_buffer(buf, **DIMS)  # non-empty StepBuffer
    buf2 = StepBuffer(tmp_path / "b2", capacity=4, header={})
    with pytest.raises(ValueError, match="n_tok is required"):
        ItemSidecarBuffer.for_step_buffer(buf2)  # n_tok missing
    # resume mismatch: a sidecar at n != the StepBuffer's n is rejected
    buf3 = StepBuffer(tmp_path / "b3", capacity=8, header={})
    sc3 = ItemSidecarBuffer.for_step_buffer(buf3, **DIMS)
    lock = LockstepItemWriter(buf3, sc3)
    lock.offer(make_row(0))
    lock.append(make_entry(0))
    buf3.flush()
    sc3.flush()
    buf3.discard_to(0)  # desynchronize the pair on purpose
    buf3.flush()
    with pytest.raises(ValueError, match="out of lockstep"):
        ItemSidecarBuffer.for_step_buffer(buf3)


def test_write_episode_jsonl(tmp_path):
    buf, sidecar = make_pair(tmp_path, capacity=4)
    item = ItemSpec(item_id="obj_0", entity=object(), category="cup", role="candidate")
    region = Region(
        region_id="zone_red", kind="zone", frame_fn=lambda: np.eye(4),
        polygon=np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]),
        z=0.74, z_band=0.05, nameable=True, color="red", entity=None,
    )
    rule = RuleSpec(
        rule_id="avoid|obj_0|-", consequence=Consequence.AVOID,
        item_ids=("obj_0",), region_ids=(), params={"margin": 0.0},
        phrases=["do not touch the cup"], split="train",
    )
    runner = SimpleNamespace(
        task_name="selective_avoid", entry=SimpleNamespace(kind="host"),
        host="stack_blocks_two", embodiment="ur5-wsg", seed=0,
        active_rule_id=None, items=[item], regions=[region], rules=[rule],
    )
    sidecar.write_episode(7, runner)
    sidecar.write_episode(8, runner)
    lines = (tmp_path / "item_buffer" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["episode_id"] == 7
    assert rec["host_task"] == "stack_blocks_two"
    assert [d["item_id"] for d in rec["items"]] == ["obj_0"]
    assert [d["rule_id"] for d in rec["rules"]] == ["avoid|obj_0|-"]
    assert "entity" not in rec["items"][0]  # no handles in the tables


def test_episode_lifecycle_counts_match(tmp_path):
    """The rollout's exact capture pattern — initial capture offers a row,
    each later capture finalizes the previous entry, the done-flush
    finalizes the last — leaves both rings with identical row counts, and
    end_episode confirms nothing is left queued."""
    buf, sidecar = make_pair(tmp_path, capacity=16)
    lock = LockstepItemWriter(buf, sidecar)
    n = 5
    lock.offer(make_row(0))  # initial _capture_tick: the h_fn offers row 0
    for i in range(1, n):  # each later capture finalizes the previous entry
        lock.append(make_entry(i - 1))
        lock.offer(make_row(i))
    lock.append(make_entry(n - 1))  # done-flush finalizes the last entry
    assert len(buf) == len(sidecar) == n
    assert lock.end_episode() == (n, n)
    # slot i of either ring is the same transition
    np.testing.assert_array_equal(
        np.asarray(buf.arrays["h"][:n]),
        np.asarray(sidecar.arrays["margins"][:n, 0]),
    )


def test_end_episode_raises_on_bypassed_done_flush(tmp_path):
    """The p5-verify death point reproduced at the seam: the final capture's
    entry reaches the StepBuffer DIRECTLY (done-flush bypassing the writer),
    so its offered row never lands in the sidecar — end_episode must raise
    instead of letting the surplus row die silently with the writer."""
    buf, sidecar = make_pair(tmp_path, capacity=16)
    lock = LockstepItemWriter(buf, sidecar)
    lock.offer(make_row(0))
    lock.append(make_entry(0))
    lock.offer(make_row(1))
    buf.append(make_entry(1))  # the final entry, appended around the writer
    assert len(buf) == 2 and len(sidecar) == 1
    with pytest.raises(RuntimeError, match="never appended"):
        lock.end_episode()


def test_append_validates_row_before_step_write(tmp_path):
    """A malformed row raises BEFORE either ring advances and stays queued
    for diagnosis — previously the StepBuffer moved +1 and the row was
    popped and lost, leaving the pair silently misaligned."""
    buf, sidecar = make_pair(tmp_path, capacity=16)
    lock = LockstepItemWriter(buf, sidecar)
    lock.offer(make_row(0, k=DIMS["k_max"] + 1))  # over the item maximum
    with pytest.raises(ValueError, match="does not fit"):
        lock.append(make_entry(0))
    assert len(buf) == len(sidecar) == 0
    assert len(lock._rows) == 1  # the row was not silently dropped
