"""LangSampler on version-2 shards (fusion_impl.md 4.3): neck frames, the frame-0 cache, the version-1 error, the
episode subset (E6), and the default path unchanged."""

import dataclasses
import json
import re

import numpy as np
import pytest

from main.lang.config import CANON_TASKS, LangConfig
from main.lang.data import NECK_FRAMES, LangSampler, LangSource, stable_hash
from main.lang.neck_buffer import NeckBuffer

from .fake_data import make_root, neck_ident


def cfg(**kw):
    return dataclasses.replace(LangConfig(), **kw)


def ident(pos_row):
    w, t, e = pos_row[0]
    return int(w), int(t), int(e)


def snapshot(d):
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*"))}


def same_batch(a, b, skip=()):
    assert set(a) - set(skip) == set(b) - set(skip)
    for k in set(a) - set(skip):
        if isinstance(a[k], np.ndarray):
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.fixture
def v2(tmp_path):
    return make_root(tmp_path / "v2", episodes=4, capacity=400, neck=True)


def want_t(t, f):
    """t_in_ep of frame f (NECK_FRAMES order: t, t-k, 0, t+1, t+1-k) of a sample at tick t, k = 2."""
    return (t, t - min(2, t), 0, t + 1, t + 1 - min(2, t + 1))[f]


def check_frames(b, C):
    """Every neck_idx entry points at the frame the sampler's row arithmetic names, and at the same row as the
    scene tokens of that frame (frames t, t-k, 0 in scene_pos; t+1, t+1-k in scene_pos_next)."""
    neck, idx = b["neck"], b["neck_idx"]
    assert idx.shape == (len(b["row"]), 5) and idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < neck["neck_maps"].shape[0]
    for k in range(len(b["row"])):
        t = int(b["t_in_ep"][k])
        scene = [ident(b["scene_pos"][k, 0]), ident(b["scene_pos"][k, 1]), ident(b["scene_pos"][k, 2]),
                 ident(b["scene_pos_next"][k, 0]), ident(b["scene_pos_next"][k, 1])]
        for f, name in enumerate(NECK_FRAMES):
            u = idx[k, f]
            w, tf, e = neck_ident(neck["neck_maps"][u])
            assert w % C == b[name][k] and tf == want_t(t, f), (k, name)
            assert (w, tf, e) == scene[f]
            assert neck["episode_id"][u] == b["episode_id"][k] and neck["t_in_ep"][u] == tf
            assert tuple(neck["T_ego2world"][u, :3, 3].astype(int)) == (w, tf, e)


def test_neck_idx_points_at_the_five_frames(v2):
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    for _ in range(10):
        b = smp.sample(32)
        check_frames(b, 400)
        assert b["neck"]["neck_maps"].dtype == np.float16 and b["neck"]["neck_maps"].shape[1:] == (3, 13, 256)
        keys = {(0, int(b[name][k])) for k in range(32) for name in NECK_FRAMES}
        assert b["neck"]["neck_maps"].shape[0] == len(keys)        # each unique (source, ring row) once


def pick_at(smp, t, eid=None, j=0):
    s = smp.sources[j]
    i = int(np.flatnonzero((s.t == t) & ((s.eid == eid) if eid is not None else True))[0])
    gi = int(np.flatnonzero((smp.sid == j) & (smp.loc == i))[0])
    return (gi, int(s.rs._sample_cols[int(s.eid[i])][0]), -1)


@pytest.mark.parametrize("t, idx, U", [(0, [0, 0, 0, 1, 0], 2), (1, [0, 1, 1, 2, 1], 3), (2, [0, 1, 1, 2, 3], 4),
                                       (5, [0, 1, 2, 3, 4], 5)])
def test_episode_start_shares_rows(v2, t, idx, U):
    """t < k: t-k clamps to frame 0 (and t+1-k for t + 1 <= k), so those frames are one entry."""
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    b = smp.sample(1, picks=[pick_at(smp, t, eid=1001)])
    assert b["neck_idx"][0].tolist() == idx and b["neck"]["neck_maps"].shape[0] == U
    check_frames(b, 400)


def test_samples_share_frames(v2):
    """t = 5 and t = 6 of one episode: 6's t-k is 5's t+1-k, 6's t+1-k is 5's t, 6's t is 5's t+1, frame 0 is one
    entry; one new frame (6's t+1)."""
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    b = smp.sample(2, picks=[pick_at(smp, 5, eid=1002), pick_at(smp, 6, eid=1002)])
    assert b["neck"]["neck_maps"].shape[0] == 6
    i5, i6 = b["neck_idx"]
    assert (i6[0], i6[1], i6[2], i6[4]) == (i5[3], i5[4], i5[2], i5[0])
    check_frames(b, 400)


def test_pairs_share_neck_indices(v2):
    smp = LangSampler([v2], cfg(pair_frac=0.5), seed=0, neck=True)
    n_pairs = 0
    for _ in range(10):
        b = smp.sample(16)
        check_frames(b, 400)
        for p in set(b["pair"].tolist()) - {-1}:
            k = np.flatnonzero(b["pair"] == p)
            assert np.array_equal(b["neck_idx"][k[0]], b["neck_idx"][k[1]])   # no copy of neck data
            n_pairs += 1
    assert n_pairs


def test_ring_wrap_frames(tmp_path):
    root = make_root(tmp_path / "ring", episodes=4, capacity=100, neck=True)   # wraps; frames cross the seam
    smp = LangSampler([root], cfg(relabel="random"), seed=0, neck=True)
    s = smp.sources[0]
    crossing = [pick_at(smp, int(s.t[i]), eid=int(s.eid[i])) for i in range(s.rows.size) if s.rows[i] in (98, 99, 0, 1)]
    assert crossing
    check_frames(smp.sample(len(crossing), picks=crossing), 100)
    for _ in range(5):
        check_frames(smp.sample(32), 100)


def test_frame0_cache(v2):
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    s = smp.sources[0]
    b = smp.sample(1, picks=[pick_at(smp, 7, eid=1003)])
    assert set(s.neck_first_cache) == {1003}
    v = s.neck_first_cache[1003]
    assert neck_ident(v["neck_maps"])[1:] == (0, 3)
    s.neck_first_cache[1003] = dict(v, neck_maps=np.full_like(v["neck_maps"], 7))   # served from RAM, not re-read
    b = smp.sample(2, picks=[pick_at(smp, 7, eid=1003), pick_at(smp, 0, eid=1003)])
    for k in range(2):
        assert (b["neck"]["neck_maps"][b["neck_idx"][k, 2]] == 7).all()
    assert (b["neck"]["neck_maps"][b["neck_idx"][1, 0]] == 7).all()   # t = 0: frame t is frame 0
    s.refresh()
    assert 1003 in s.neck_first_cache                                  # a live episode keeps its entry


def test_version1_source_raises(tmp_path, v2):
    v1 = make_root(tmp_path / "v1", episodes=2, capacity=200)
    assert not LangSource(v1, k_prev=2).has_neck and LangSource(v2, k_prev=2).has_neck
    with pytest.raises(ValueError, match="version-1"):
        LangSource(v1, k_prev=2).neck
    for roots in ([v1], [v2, v1]):
        with pytest.raises(ValueError, match=re.escape(str(v1.resolve()))):
            LangSampler(roots, cfg(), seed=0, neck=True)
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    with pytest.raises(ValueError, match="version-1 shard"):
        smp.add_source(v1)
    LangSampler([v1], cfg(), seed=0).sample(4)       # neck off: version 1 as before


def test_mismatched_neck_fields_raise(tmp_path, v2):
    """Sources whose neck rows cannot share one batch array (other level shapes) are refused at construction."""
    other = make_root(tmp_path / "o", episodes=1, capacity=100)
    NeckBuffer(other / "neck_buffer", capacity=100, level_shapes=((1, 1),) * 4)
    with pytest.raises(ValueError, match="differ"):
        LangSampler([v2, other], cfg(), seed=0, neck=True)
    short = make_root(tmp_path / "s", episodes=1, capacity=100)
    NeckBuffer(short / "neck_buffer", capacity=50, level_shapes=((1, 1),) * 4)
    with pytest.raises(ValueError, match="capacity"):     # a neck ring that is not the item ring's twin
        LangSource(short, k_prev=2).neck


def test_out_of_lockstep_neck_raises(v2):
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    s = smp.sources[0]
    got = s.neck_rows(np.array([5, 3, 5]))
    assert [neck_ident(m)[0] for m in got["neck_maps"]] == [5, 3, 5]
    ep = np.load(v2 / "neck_buffer" / "episode_id.npy", mmap_mode="r+")
    ep[50] += 1                                     # row 50: a rollback mirrored on one side only
    ep.flush()
    with pytest.raises(RuntimeError, match="out of lockstep"):
        s.neck_rows(np.array([49, 50]))
    assert s.neck_rows(np.array([49, 50]), check=False)["episode_id"].tolist() == [1001, 1002]
    ep[:] += 1
    ep.flush()
    with pytest.raises(RuntimeError, match="neck_buffer row"):
        smp.sample(8)


def test_neck_reads_create_nothing(v2):
    before = snapshot(v2)
    smp = LangSampler([v2], cfg(), seed=0, neck=True)
    smp.sample(16), smp.refresh(), smp.sources[0].neck_rows(np.arange(10))
    assert snapshot(v2) == before


def test_neck_draws_no_random_numbers(v2):
    """neck on adds two keys and changes nothing else: same batches, same RNG state."""
    for kw in (dict(), dict(relabel="random"), dict(shuffle_text=True)):
        a = LangSampler([v2], cfg(**kw), seed=3)
        b = LangSampler([v2], cfg(**kw), seed=3, neck=True)
        for _ in range(3):
            same_batch(a.sample(16), b.sample(16), skip=("neck", "neck_idx"))
        assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_default_path_unchanged(tmp_path, v2):
    """No new kwargs == neck False, episodes_per_task 0, and a version-2 shard reads as its version-1 twin."""
    v1 = make_root(tmp_path / "v1", episodes=4, capacity=400)
    other = make_root(tmp_path / "t1", episodes=2, capacity=200, task="selective_avoid", eid0=5000)
    for kw in (dict(), dict(relabel="random"), dict(relabel="fixed"), dict(shuffle_text=True)):
        for seed in (0, 1):
            base = LangSampler([v1, other], cfg(**kw), seed=seed)
            expl = LangSampler([v1, other], cfg(**kw), seed=seed, neck=False, episodes_per_task=0)
            twin = LangSampler([v2, other], cfg(**kw), seed=seed)
            for _ in range(3):
                x = base.sample(16)
                same_batch(x, expl.sample(16))
                y = twin.sample(16)
                same_batch(x, y, skip=("source",))
                assert [s.replace("v2", "v1") for s in y["source"]] == x["source"]
            assert base.episodes == expl.episodes and base.episode_keep is None


# ------------------------------------------------------- episodes_per_task ----
@pytest.fixture
def pool(tmp_path):
    """(gap_place, ur5-wsg): a 4 + c 2 episodes (c repeats a's ids 1000, 1001); (gap_place, franka-panda): d 2;
    (selective_avoid, franka-panda): b 3."""
    a = make_root(tmp_path / "a", episodes=4, capacity=400, eid0=1000)
    c = make_root(tmp_path / "c", episodes=2, capacity=200, eid0=1000)
    b = make_root(tmp_path / "b", episodes=3, capacity=300, task="selective_avoid", eid0=5000)
    d = make_root(tmp_path / "d", episodes=2, capacity=200, eid0=3000)
    for p in (d / "item_buffer" / "episodes").glob("*.json"):
        rec = json.loads(p.read_text())
        rec["embodiment"] = "franka-panda"
        p.write_text(json.dumps(rec))
    return [a, b, c, d]


GROUPS = {("gap_place", "ur5-wsg"): 6, ("gap_place", "franka-panda"): 2, ("selective_avoid", "franka-panda"): 3}


def drawn(smp) -> set:
    """(source name, episode id) of the rows the sampler draws from."""
    return {(s.name, int(e)) for j, s in enumerate(smp.sources) for e in np.unique(s.eid[smp.ok[smp.sid == j]])}


def test_episode_subset_nested_and_counted(pool):
    full = LangSampler(pool, cfg(), seed=0)
    assert full.episode_counts() == dict(sorted(GROUPS.items())) and len(drawn(full)) == sum(GROUPS.values())
    prev = set()
    for N in (1, 2, 3, 4, 6, 7):
        smp = LangSampler(pool, cfg(), seed=0, episodes_per_task=N)
        assert smp.episode_counts() == {g: min(N, n) for g, n in sorted(GROUPS.items())}
        got = drawn(smp)
        assert prev <= got and len(got) == sum(min(N, n) for n in GROUPS.values())
        prev = got
        rows = {(s.name, int(e)) for j, s in enumerate(smp.sources) for e in s.eid}
        assert len(smp) == sum(39 for _ in got) and got <= rows
    assert prev == drawn(full)


def test_episode_subset_order(pool):
    """The contract's order: sorted by stable_hash(("e6", task, embodiment, eid)), ties (a repeated id) by
    source name."""
    smp = LangSampler(pool, cfg(), seed=0, episodes_per_task=3)
    eps = {}
    for s in smp.sources:
        for e in np.unique(s.eid).tolist():
            i0 = int(np.flatnonzero(s.eid == e)[0])
            eps.setdefault((CANON_TASKS[int(s.task[i0])], str(s.emb[i0])), []).append((s.name, int(e)))
    want = set()
    for (task, emb), xs in eps.items():
        want |= set(sorted(xs, key=lambda ne: (stable_hash(("e6", task, emb, ne[1])), ne[0]))[:3])
    assert drawn(smp) == want


def test_episode_subset_deterministic(pool):
    ref = LangSampler(pool, cfg(), seed=0, episodes_per_task=2)
    for kw in (dict(seed=5), dict(seed=1, n_threads=2)):
        assert drawn(LangSampler(pool, cfg(), episodes_per_task=2, **kw)) == drawn(ref)
    rev = LangSampler(pool[::-1], cfg(), seed=0, episodes_per_task=2)       # root order does not matter
    assert drawn(rev) == drawn(ref) and rev.episode_counts() == ref.episode_counts()


def test_episode_subset_batches_and_shuffled_pool(pool):
    smp = LangSampler(pool, cfg(shuffle_text=True), seed=0, episodes_per_task=1)
    keep = drawn(smp)
    assert {(smp.sources[j].name, e) for j, e in smp.episodes} == keep    # the shuffled-sentence pool, restricted
    for _ in range(10):
        b = smp.sample(32)
        assert {(s, int(e)) for s, e in zip(b["source"], b["episode_id"])} <= keep
    # task / embodiment filters: only that group is drawn; the shuffled pool keeps every group's first N
    f = LangSampler(pool, cfg(shuffle_text=True), embodiments=("ur5-wsg",), seed=0, episodes_per_task=2)
    assert f.episode_counts() == {("gap_place", "ur5-wsg"): 2}
    assert {(f.sources[j].name, e) for j, e in f.episodes} == drawn(LangSampler(pool, cfg(), seed=0,
                                                                                episodes_per_task=2))
    t = LangSampler(pool, cfg(), tasks=("selective_avoid",), seed=0, episodes_per_task=2)
    assert t.episode_counts() == {("selective_avoid", "franka-panda"): 2}
    assert set(t.sample(32)["task"]) == {"selective_avoid"}


def test_episode_subset_all_equals_default(pool):
    for kw in (dict(), dict(shuffle_text=True)):
        a = LangSampler(pool, cfg(**kw), seed=2)
        b = LangSampler(pool, cfg(**kw), seed=2, episodes_per_task=100)
        assert a.episodes == b.episodes
        for _ in range(3):
            same_batch(a.sample(16), b.sample(16))


def test_episode_subset_with_neck(tmp_path):
    a = make_root(tmp_path / "a", episodes=4, capacity=400, neck=True)
    smp = LangSampler([a], cfg(), seed=0, episodes_per_task=2, neck=True)
    b = smp.sample(32)
    assert len(set(b["episode_id"].tolist())) <= 2 and smp.episode_counts() == {("gap_place", "ur5-wsg"): 2}
    check_frames(b, 400)
