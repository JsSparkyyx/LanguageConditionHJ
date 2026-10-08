"""LangSampler tests (VANILLA_PLAN 4.4)."""

import dataclasses
import json

import numpy as np
import pytest

from main.item_envs import language
from main.lang.config import LangConfig
from main.lang.data import LangSampler, LangSource, PrefetchSampler

from .fake_data import make_root, template


def cfg(**kw):
    return dataclasses.replace(LangConfig(), **kw)


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path / "r0", episodes=4, capacity=400)


def ident(pos_row):
    w, t, e = pos_row[0]
    return int(w), int(t), int(e)


def test_terminal_rows_excluded(root):
    s = LangSource(root, k_prev=2)
    done = np.load(root / "buffer/done.npy", mmap_mode="r")
    assert not done[s.rows].any()
    assert s.rows.size == 4 * 39  # every row but each episode's last


def test_frames_and_next_prev(root):
    s = LangSource(root, k_prev=2)
    for t_want in (0, 1, 2, 7):
        i = int(np.flatnonzero(s.t == t_want)[0])
        r = s.read(i)
        rows = s.rows_of(i)
        w0, t0, e0 = ident(r["scene_pos"][0])
        assert t0 == t_want and w0 == rows["row"]
        assert ident(r["scene_pos"][1])[1] == max(t_want - 2, 0)          # t - k, clamped to 0
        assert ident(r["scene_pos"][2])[1] == 0                           # frame 0
        assert ident(r["scene_pos_next"][0])[1] == t_want + 1
        assert ident(r["scene_pos_next"][1])[1] == t_want + 1 - min(2, t_want + 1)
        assert rows["next_prev_row"] == rows["next_row"] - min(2, t_want + 1)
        assert np.isclose(r["dtheta"][0], rows["row"] * 1e-4)


def test_ring_wrap_seam_and_truncated(tmp_path):
    # capacity 100, 4 episodes x 40 rows: wraps once; episode 0 is overwritten,
    # episode 1 loses its first rows (truncated -> excluded entirely)
    root = make_root(tmp_path / "ring", episodes=4, capacity=100)
    s = LangSource(root, k_prev=2)
    eps = set(s.eid.tolist())
    assert 1000 not in eps and 1001 not in eps     # overwritten / truncated
    assert {1002, 1003} <= eps
    # every sampled frame is from the right row, including across the physical array end
    for i in range(s.rows.size):
        r, rows = s.read(i), s.rows_of(i)
        w, t, e = ident(r["scene_pos"][0])
        assert t == s.t[i] and (w % 100) == rows["row"]
        wn, tn, en = ident(r["scene_pos_next"][0])
        assert en == e and tn == t + 1 and (wn % 100) == rows["next_row"]
        assert ident(r["scene_pos"][2])[1] == 0 and ident(r["scene_pos"][2])[2] == e
    crossing = [i for i in range(s.rows.size) if s.rows[i] in (98, 99, 0, 1)]
    assert crossing


def test_pairs_always_mixed_and_balance(root):
    smp = LangSampler([root], cfg(pair_frac=0.5), seed=0)
    for _ in range(20):
        b = smp.sample(16)
        for p in set(b["pair"].tolist()) - {-1}:
            fs = b["fails_soon"][b["pair"] == p]
            assert fs.size == 2 and fs.any() and not fs.all()
            rows = b["row"][b["pair"] == p]
            assert rows[0] == rows[1]
            # one transition, read once: identical frames
            k = np.flatnonzero(b["pair"] == p)
            assert np.array_equal(b["scene_tokens"][k[0]], b["scene_tokens"][k[1]])
    assert smp.health()["data/pair_mixed_frac"] == 1.0


def test_random_mode_has_no_pairs(root):
    smp = LangSampler([root], cfg(relabel="random"), seed=0)
    assert (smp.sample(32)["pair"] == -1).all()


def test_fixed_mode_never_changes_rule(root, tmp_path):
    smp = LangSampler([root], cfg(relabel="fixed"), seed=0)
    seen = {}
    for _ in range(30):
        b = smp.sample(16)
        assert (b["pair"] == -1).all()
        for e, c in zip(b["episode_id"], b["rule_col"]):
            assert seen.setdefault(int(e), int(c)) == int(c)
    # a filter episode's named rule (record active_rule) is the fixed rule
    r2 = make_root(tmp_path / "act", episodes=2, capacity=200, active_rule=2)
    smp2 = LangSampler([r2], cfg(relabel="fixed"), seed=0)
    assert set(smp2.sample(32)["rule_col"].tolist()) == {2}


def test_no_heldout_sentences(root):
    smp = LangSampler([root], cfg(), seed=0)
    rec = template("gap_place")
    held = set()
    for rule in rec["rules"]:
        for sp in ("heldout_template", "heldout_word"):
            held |= set(language.sentences(rule, rec, sp))
    n = 0
    while n < 10000:
        b = smp.sample(64)
        assert not (set(b["sentence"]) & held)
        n += 64


def test_shuffled_sentences_come_from_other_episodes(tmp_path):
    a = make_root(tmp_path / "a", episodes=2, capacity=200, task="gap_place", eid0=1000)
    b_ = make_root(tmp_path / "b", episodes=2, capacity=200, task="selective_avoid", eid0=5000)
    smp = LangSampler([a, b_], cfg(shuffle_text=True), seed=0)
    own = {}
    for root, task in ((a, "gap_place"), (b_, "selective_avoid")):
        rec = template(task)
        own[task] = {s for r in rec["rules"] if r.get("sampleable") for s in language.sentences(r, rec, "train")}
    b = smp.sample(64)
    cross = sum(1 for s, t in zip(b["sentence"], b["task"]) if s not in own[t])
    assert cross > 16   # most sentences come from the other task's episodes


def test_task_filter_and_balance(tmp_path):
    a = make_root(tmp_path / "a", episodes=4, capacity=400, task="gap_place", eid0=1000)
    b_ = make_root(tmp_path / "b", episodes=1, capacity=100, lengths=[30], task="selective_avoid", eid0=5000)
    smp = LangSampler([a, b_], cfg(relabel="random"), seed=0)
    tasks = [t for _ in range(20) for t in smp.sample(64)["task"]]
    frac = tasks.count("selective_avoid") / len(tasks)
    assert 0.4 < frac < 0.6     # task-balanced although T1 has 30 of 186 rows
    smp2 = LangSampler([a, b_], cfg(), tasks=("gap_place",), seed=0)
    assert set(smp2.sample(32)["task"]) == {"gap_place"}


def test_recall_label(root):
    s = LangSource(root, k_prev=2, recall_horizon=7)
    A = s.rs.buf.arrays
    for i in range(s.rows.size):
        slot = int(s.rows[i])
        L = 40
        t = int(s.t[i])
        H = np.asarray(A["h_rules"])[slot - t: slot - t + L]
        want = (H[t:t + 7] < 0).any(0)
        assert np.array_equal(s.recall[slot, :H.shape[1]], want)


def test_probe_ref_mask(root):
    smp = LangSampler([root], cfg(), seed=0, probe=True)
    b = smp.sample(8)
    assert b["ref_mask"].shape == (8, 3, 12)
    # rules about block_0 (record object 1) get the fake mask on camera 0
    for k, rid in enumerate(b["rule_id"]):
        if "block_0" in rid:
            assert b["ref_mask"][k, 0, :4].max() == 1.0


def test_prefetch(root):
    smp = LangSampler([root], cfg(), seed=0)
    pf = PrefetchSampler(smp, 8)
    b1 = pf.next()
    b2 = pf.next()
    assert b1["scene_tokens"].shape == (8, 3, 12, 256)
    pf.stop()
    assert pf.thread is None


def test_frozen_rows_cap(tmp_path):
    root = make_root(tmp_path / "f", episodes=4, capacity=400)       # 4 x 40 rows
    full = LangSampler([root], cfg(), seed=0)
    capped = LangSampler([root], cfg(), seed=0, max_rows={str(root): 80})
    assert len(full) == 4 * 39 and len(capped) == 2 * 39             # the first two episodes only
    assert set(capped.sources[0].eid.tolist()) == {1000, 1001}
    capped.refresh()                                                  # stays frozen on refresh
    assert len(capped) == 2 * 39
    other = make_root(tmp_path / "g", episodes=1, capacity=100)
    assert len(LangSampler([root, other], cfg(), seed=0, max_rows={str(root): 80})) == 2 * 39


def test_stale_bead_rows_dropped(tmp_path):
    """Shards written before the bead-count h (2026-10-07: no item_buffer/n_violating.npy) give no
    T2 / T4 / T6 rows; their other tasks, and every newer shard, are kept."""
    a = make_root(tmp_path / "a", episodes=2, capacity=200, task="gap_place", eid0=1000)
    b_ = make_root(tmp_path / "b", episodes=2, capacity=200, task="gap_place", eid0=2000)
    for p in (b_ / "item_buffer" / "episodes").glob("*.json"):   # make b a clean_pour shard
        d = json.loads(p.read_text())
        d["task"] = "clean_pour"
        p.write_text(json.dumps(d))
    assert LangSource(b_, k_prev=2).rows.size == 2 * 39           # new bead shard: kept
    for r in (a, b_):
        (r / "item_buffer" / "n_violating.npy").unlink()          # what an older shard looks like
    assert LangSource(a, k_prev=2).rows.size == 2 * 39            # old non-bead shard: kept
    assert LangSource(b_, k_prev=2).rows.size == 0                # old bead shard: dropped
    assert LangSource(b_, k_prev=2, drop_stale_bead=False).rows.size == 2 * 39
    smp = LangSampler([a, b_], cfg(relabel="random"), seed=0)
    assert set(smp.sample(16)["task"]) == {"gap_place"}


def test_rows_capped_at_the_last_episode_line(tmp_path):
    """Review round 3, C2: rows past the last recorded episode are invisible (an episode in progress, a stale
    writer's rows), and a header with fewer rows than the episode lines record is refused."""
    import json

    import pytest

    root = make_root(tmp_path / "s", episodes=3, capacity=200)              # 3 x 40 rows
    lj = root / "lang_episodes.jsonl"
    lines = [json.loads(x) for x in lj.read_text().splitlines()]
    for k, d in enumerate(lines):
        d["rows"] = [40 * k, 40 * k + 40]
    lj.write_text("".join(json.dumps(d) + "\n" for d in lines[:2]))         # episode 1002 has no line yet
    assert set(LangSource(root, k_prev=2).eid.tolist()) == {1000, 1001}
    lj.write_text("".join(json.dumps(d) + "\n" for d in lines))
    assert set(LangSource(root, k_prev=2).eid.tolist()) == {1000, 1001, 1002}
    hp = root / "item_buffer" / "header.json"
    h = json.loads(hp.read_text())
    h["n"] = h["head"] = 100                                                 # a stale writer's header
    hp.write_text(json.dumps(h))
    with pytest.raises(RuntimeError, match="episode lines end"):
        LangSource(root, k_prev=2)
