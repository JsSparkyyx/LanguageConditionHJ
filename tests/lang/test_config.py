"""Seed / episode-id scheme (VANILLA_PLAN 4.2) and LangConfig round trips."""

import dataclasses

import pytest

from main.lang import config as C


def _ranges():
    """[lo, hi) of every stream at its extreme bounds."""
    out = {
        "warmup": (C.warmup_offset(0), C.warmup_offset(C.WARMUP_G_MAX - 1) + C.STREAM_SPAN),
        "validation": (C.val_offset(0), C.val_offset(C.VAL_G_MAX - 1) + C.STREAM_SPAN),
        "rounds": (C.round_offset(0), C.round_offset(1399) + C.STREAM_SPAN),
    }
    seeds = [C.eval_seed(t, e, s, a, h)
             for t in C.CANON_TASKS for e in C.CANON_EMBODIMENTS
             for h in (C.CANON_T1_HOSTS if t == "selective_avoid" else (None,))
             for s in (0, 999) for a in (0, 9)]
    out["eval"] = (min(seeds), max(seeds) + 1)
    return out


def test_streams_disjoint_and_int32():
    r = sorted(_ranges().items(), key=lambda kv: kv[1][0])
    for (na, (a0, a1)), (nb, (b0, b1)) in zip(r, r[1:]):
        assert a1 <= b0, f"{na} {a0}..{a1} overlaps {nb} {b0}..{b1}"
    assert max(hi for _, hi in _ranges().values()) - 1 <= C.INT32_MAX


def test_bounds_asserted():
    with pytest.raises(AssertionError):
        C.warmup_offset(C.WARMUP_G_MAX)
    with pytest.raises(AssertionError):
        C.val_offset(C.VAL_G_MAX)
    with pytest.raises(AssertionError):
        C.round_offset(1400)
    with pytest.raises(AssertionError):
        C.check_stream(C.warmup_offset(0), C.STREAM_SPAN)
    with pytest.raises(AssertionError):
        C.eval_seed("granular_pour", "ur5-wsg", 1000)
    with pytest.raises(AssertionError):
        C.shard_g("ur5-wsg", 15, 15)


def test_eval_seeds_unique():
    seeds = [C.eval_seed(t, e, s, a, h)
             for t in C.CANON_TASKS for e in C.CANON_EMBODIMENTS
             for h in (C.CANON_T1_HOSTS if t == "selective_avoid" else (None,))
             for s in range(0, 1000, 37) for a in range(10)]
    assert len(seeds) == len(set(seeds))


def test_canonical_indices_ignore_filters():
    # a single-embodiment run evaluates on the same seeds as `full`
    assert C.eval_seed("gap_place", "franka-panda", 3) == (
        1_500_000_000 + 6 * 10_000_000 + 1 * 100_000 + 3)
    assert C.shard_g("franka-panda", 2, 15) == 17


def test_langconfig_roundtrip(tmp_path):
    cfg = C.LangConfig()
    assert cfg.gamma == cfg.gamma_final == 0.9
    assert cfg.item_out_root and cfg.bank_hash
    p = cfg.save(tmp_path / "c.json")
    back = C.LangConfig.load(p)
    assert back == cfg and back.hash() == cfg.hash()
    assert isinstance(back.task_choices, tuple)


def test_langconfig_replace_keeps_item_fields():
    cfg = dataclasses.replace(C.LangConfig(), item_host="stack_bowls_two", item_active_rule="primary")
    cfg2 = dataclasses.replace(cfg, task="push_zone")
    assert cfg2.item_host == "stack_bowls_two" and cfg2.item_active_rule == "primary"


def test_langconfig_guards():
    with pytest.raises(AssertionError):
        C.LangConfig(gamma_final=0.95)
    with pytest.raises(AssertionError):
        C.LangConfig(item_out_root="")
    with pytest.raises(AssertionError):
        C.LangConfig(relabel="pairs")
