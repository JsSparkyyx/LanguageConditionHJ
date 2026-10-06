"""Non-sim tests for the relabeling sampler (main/item_envs/relabel.py) on a
synthetic lockstep pair: transitions and their next / previous / first rows,
the harmful / benign balance, paired samples, sentences valid for the drawn
rule, h_gain + clip, and the text-cache contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()

from main.item_envs import language as L  # noqa: E402
from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter  # noqa: E402
from main.item_envs.relabel import RelabelSampler  # noqa: E402
from main.train.buffer import StepBuffer  # noqa: E402

from .test_item_buffer import DIMS, make_entry, make_row  # noqa: E402


def _record(eid: int) -> dict:
    """A T7-like scene: two standing blocks, upright + stay each (4 columns)."""
    objects = [dict(id=0, item_id="block_0", category="block", synonyms=["brick", "pillar"], color="blue",
                    size=None, role="candidate", kind="rigid"),
               dict(id=1, item_id="block_1", category="block", synonyms=["brick", "pillar"], color="red",
                    size=None, role="candidate", kind="rigid")]
    rules = []
    for j, (cons, o) in enumerate([("upright", 0), ("stay", 0), ("upright", 1), ("stay", 1)]):
        rules.append(dict(idx=j, rule_id=f"{cons}|block_{o}|-", consequence=cons, subject={"object": o},
                          reference=None, params={}, set=False, split="train", ambiguous=False))
    rec = {"episode_id": eid, "task": "gap_place", "active_rule": 1, "objects": objects, "regions": [], "rules": rules}
    for r in rules:
        r["sampleable"] = L.sampleable(r, rec)
    return rec


def _build(tmp_path, episodes=((0, 30), (1, 20)), capacity=128):
    buf = StepBuffer(tmp_path / "buffer", capacity=capacity, header={"test": 1})
    side = ItemSidecarBuffer.for_step_buffer(buf, **DIMS)
    i = 0
    for eid, n in episodes:
        lock = LockstepItemWriter(buf, side)
        for t in range(n):
            # column 1 (stay, blue) fails at the episode's tick 20; the others never
            h = [0.9, -0.3 if t >= 20 else 0.2 + 0.01 * t, 0.8, 0.7]
            lock.offer(make_row(i, r=4, h_rules=h, t_in_ep=t, episode=eid))
            lock.append(make_entry(i, episode=eid))
            i += 1
        ep_dir = side.root / "episodes"
        ep_dir.mkdir(exist_ok=True)
        (ep_dir / f"{eid}.json").write_text(json.dumps(_record(eid)))
    side.flush()
    return buf, side


def test_rows_next_prev_first_and_terminal(tmp_path):
    _, side = _build(tmp_path)
    s = RelabelSampler(side.root, pair_frac=0.0)
    # 30 + 20 rows; the newest row (episode 1, t = 19) has no next yet -> excluded
    assert len(s) == 49
    b = s.sample(256, np.random.default_rng(0))
    t = np.asarray(side.arrays["t_in_ep"]); ep = np.asarray(side.arrays["episode_id"])
    for row, nxt, prev, first, done in zip(b["row"], b["next_row"], b["prev_row"], b["first_row"], b["done"]):
        assert ep[first] == ep[row] and t[first] == 0
        assert prev == row - min(1, t[row])
        if done:
            assert row == 29 and nxt == row  # episode 0 ended at row 29
        else:
            assert nxt == row + 1 and ep[nxt] == ep[row]
    h = np.asarray(side.arrays["h_rules"])
    np.testing.assert_allclose(b["h"], h[b["row"], b["rule"]], atol=1e-6)
    np.testing.assert_allclose(b["h_next"], h[b["next_row"], b["rule"]], atol=1e-6)


def test_harmful_benign_balance_and_pairs(tmp_path):
    _, side = _build(tmp_path)
    s = RelabelSampler(side.root, pair_frac=0.5)
    b = s.sample(400, np.random.default_rng(1))
    fs = np.asarray(side.arrays["fails_soon"])
    mixed = np.array([fs[r, :4].any() and not fs[r, :4].all() for r in b["row"]])
    harm = b["fails_soon"][mixed & (b["pair"] < 0)]
    assert 0.35 < harm.mean() < 0.65  # ~50/50 where both exist
    pairs = b["pair"][b["pair"] >= 0]
    assert pairs.size >= 150
    for p in np.unique(pairs):
        k = np.flatnonzero(b["pair"] == p)
        assert k.size == 2 and b["row"][k[0]] == b["row"][k[1]]
        assert b["fails_soon"][k[0]] != b["fails_soon"][k[1]]  # one harmful, one benign


def test_sentences_match_the_rule(tmp_path):
    _, side = _build(tmp_path)
    s = RelabelSampler(side.root)
    b = s.sample(64, np.random.default_rng(2))
    rec = _record(0)
    for col, sent in zip(b["rule"], b["sentence"]):
        assert sent in L.sentences(rec["rules"][col], rec, "train")
        assert ("blue" in sent) == (col in (0, 1))


def test_h_gain_and_clip(tmp_path):
    _, side = _build(tmp_path)
    s = RelabelSampler(side.root, h_gain={"upright": 3.0})
    b = s.sample(200, np.random.default_rng(3))
    h = np.asarray(side.arrays["h_rules"])[b["row"], b["rule"]]
    up = np.array([c == "upright" for c in b["consequence"]])
    np.testing.assert_allclose(b["h"][up], np.clip(3.0 * h[up], -1, 1), atol=1e-6)
    np.testing.assert_allclose(b["h"][~up], h[~up], atol=1e-6)


def test_text_cache_contract(tmp_path):
    _, side = _build(tmp_path)
    rec = _record(0)
    sents = sorted({x for r in rec["rules"] for x in L.sentences(r, rec, "train")})
    cache = {"index": {x: i for i, x in enumerate(sents)}}
    b = RelabelSampler(side.root, text_cache=cache).sample(32, np.random.default_rng(4))
    assert [sents[i] for i in b["text_index"]] == b["sentence"]
    with pytest.raises(KeyError, match="text cache miss"):
        RelabelSampler(side.root, text_cache={"index": {}}).sample(4, np.random.default_rng(5))


def test_benign_rules(tmp_path):
    _, side = _build(tmp_path)
    s = RelabelSampler(side.root)
    assert s.benign_rules(0) == [0, 2, 3]  # column 1 fails at tick 20 of episode 0
    assert s.benign_rules(1) == [0, 1, 2, 3]  # episode 1 has 20 ticks: column 1 never fails
