"""Section 7.2 schedule balance and Section 7.1 named-rule choice."""

from collections import Counter

import numpy as np
import pytest

from main.lang.collect import choose_named_rule, shard_schedule
from main.lang.config import CANON_T1_HOSTS, CANON_TASKS, shard_g


@pytest.mark.parametrize("n_shards", [1, 3, 5, 15])
def test_union_balanced(n_shards):
    per_task = 30
    slots = []
    for emb in ("ur5-wsg", "franka-panda"):
        for i in range(n_shards):
            s = shard_schedule(CANON_TASKS, emb, i, n_shards, per_task, shard_g(emb, i, n_shards))
            c = Counter(t for t, _, _ in s)
            assert set(c.values()) == {per_task // n_shards}  # every shard exactly balanced
            slots += s
    c = Counter((t, e) for t, e, _ in slots)
    assert set(c.values()) == {per_task}
    hosts = Counter((e, h) for t, e, h in slots if t == "selective_avoid")
    assert set(hosts.values()) == {per_task // len(CANON_T1_HOSTS)}  # 6 per host per embodiment
    assert all(h is None for t, _, h in slots if t != "selective_avoid")


def test_shards_shuffle_differently():
    a = shard_schedule(CANON_TASKS, "ur5-wsg", 0, 15, 30, 0)
    b = shard_schedule(CANON_TASKS, "ur5-wsg", 1, 15, 30, 1)
    assert [t for t, _, _ in a] != [t for t, _, _ in b]


def test_n_shards_must_divide():
    with pytest.raises(ValueError):
        shard_schedule(CANON_TASKS, "ur5-wsg", 0, 4, 30, 0)


def _rec():
    rules = [
        {"rule_id": "a", "sampleable": True}, {"rule_id": "b", "sampleable": True},
        {"rule_id": "c", "sampleable": False}, {"rule_id": "d", "sampleable": True},
    ]
    return {"rules": rules}


def test_named_rule_kinds():
    desig = {"a": "harm", "b": "benign", "c": "harm", "d": "unclear"}
    rng = np.random.default_rng(0)
    picks = Counter(choose_named_rule(_rec(), desig, rng, 0.5)["rule_id"] for _ in range(2000))
    assert set(picks) == {"a", "b"}  # never unclear, never unsampleable
    assert 0.45 < picks["a"] / 2000 < 0.55


def test_named_rule_fallbacks():
    rng = np.random.default_rng(0)
    only_harm = {"a": "harm", "b": "unclear", "c": None, "d": "unclear"}
    assert {choose_named_rule(_rec(), only_harm, rng)["rule_id"] for _ in range(50)} == {"a"}
    assert choose_named_rule(_rec(), {"a": None, "b": "unclear", "c": "harm", "d": None}, rng) is None


def test_resume_point_rolls_back_partial_episode(tmp_path):
    import json

    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.lang.collect import read_lines, resume_point
    from main.train.buffer import StepBuffer

    from .fake_data import make_root

    root = make_root(tmp_path / "s", episodes=3, capacity=400)   # 3 x 40 rows
    # lang_episodes.jsonl records only the first two episodes (+ a truncated line)
    lines = [{"episode_id": 1000, "rows": [0, 40], "attempt_next": 1, "slot": 0},
             {"skipped": True, "attempt_next": 6, "slot": 1},
             {"episode_id": 1001, "rows": [40, 80], "attempt_next": 7, "slot": 2}]
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines) + '{"episode_id": 10')
    buf, side = StepBuffer(root / "buffer"), ItemSidecarBuffer(root / "item_buffer")
    assert len(buf) == len(side) == 120
    done, attempt, rows = resume_point(root, buf, side)
    assert (done, attempt, rows) == (3, 7, 80)
    assert len(StepBuffer(root / "buffer")) == len(ItemSidecarBuffer(root / "item_buffer")) == 80
    assert len(read_lines(root / "lang_episodes.jsonl")) == 3     # truncated line dropped


def test_resume_point_refuses_rows_of_another_episode(tmp_path):
    """B4: lines whose rows hold another episode (a mid-write copy, two writers) stop the resume."""
    import json

    import pytest

    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.lang.collect import resume_point
    from main.train.buffer import StepBuffer

    from .fake_data import make_root

    root = make_root(tmp_path / "s", episodes=3, capacity=400)   # episodes 1000..1002, 40 rows each
    lines = [{"episode_id": 1000, "rows": [0, 40], "attempt_next": 1, "slot": 0},
             {"episode_id": 1002, "rows": [40, 80], "attempt_next": 2, "slot": 1}]   # rows 40..79 are 1001's
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))
    with pytest.raises(RuntimeError, match="do not hold that episode"):
        resume_point(root, StepBuffer(root / "buffer"), ItemSidecarBuffer(root / "item_buffer"))


def test_h_code_hash_depends_on_the_tasks():
    """B2: shard.json's h_code covers margins.py, rules.py and the shard's task yaml files."""
    from main.lang.collect import h_code_hash

    a = h_code_hash(["granular_pour", "clean_pour"])
    assert a == h_code_hash(["clean_pour", "granular_pour"]) and a != h_code_hash(["gap_place"])
