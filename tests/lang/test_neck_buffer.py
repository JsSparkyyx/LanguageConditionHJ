"""NeckBuffer / NeckLockstepWriter tests (fusion_impl.md 3, 4.2): ring semantics, read-only readers, and the
lockstep with a StepBuffer + item sidecar through the rollout's one-capture-late appends, a rollback and a resume."""

import json

import numpy as np
import pytest

from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter
from main.lang.neck_buffer import NECK_FIELDS, NeckBuffer, NeckLockstepWriter, neck_fields
from main.train.buffer import StepBuffer

from .fake_data import FAKE_LEVEL_SHAPES, N_TOK_CAM, h_pattern, item_row, neck_ident, neck_row, step_entry, template


def full_row(w, t=0, e=0, eid=1000):
    return dict(neck_row(w, t, e), episode_id=eid, t_in_ep=t)


def small(root, capacity):
    return NeckBuffer(root, capacity=capacity, level_shapes=FAKE_LEVEL_SHAPES)


def snapshot(d):
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*"))}


def test_default_fields_are_the_contract():
    want = {"neck_maps": ((3, 1700, 256), np.float16), "neck_3d": ((3, 400, 128), np.float16),
            "projection_mat": ((3, 4, 4), np.float32), "image_wh": ((3, 2), np.float32),
            "T_ego2world": ((4, 4), np.float32), "episode_id": ((), np.int64), "t_in_ep": ((), np.int32)}
    assert {k: (shp, np.dtype(dt)) for k, (shp, dt) in want.items()} == NECK_FIELDS
    # fake level shapes: 13 cells, levels 1 + 2 give the fake shards' 4 scene tokens per camera
    assert neck_fields(FAKE_LEVEL_SHAPES)["neck_3d"][0] == (3, N_TOK_CAM, 128)


def test_create_append_flush_reopen_read(tmp_path):
    root = tmp_path / "neck_buffer"
    nb = small(root, 10)
    h = json.loads((root / "header.json").read_text())       # a fresh buffer publishes its header at once
    assert (h["capacity"], h["n"], h["head"], h["neck_version"]) == (10, 0, 0, 1)
    assert h["level_shapes"] == [list(x) for x in FAKE_LEVEL_SHAPES]
    assert h["fields"]["neck_maps"] == [[3, 13, 256], "float16"] and h["fields"]["episode_id"] == [[], "int64"]
    for w in range(3):
        nb.append(full_row(w, t=w, eid=1000 + w))
    reader = NeckBuffer(root)
    assert len(reader) == 0                                   # the header is published by flush only
    nb.flush()
    assert reader.refresh() == 3 and reader.head == 3
    got = reader.read(np.array([2, 0, 0]))
    assert [neck_ident(m) for m in got["neck_maps"]] == [(2, 2, 0), (0, 0, 0), (0, 0, 0)]
    assert got["episode_id"].tolist() == [1002, 1000, 1000] and got["t_in_ep"].tolist() == [2, 0, 0]
    assert got["episode_id"].dtype == np.int64 and got["t_in_ep"].dtype == np.int32
    assert got["neck_3d"].shape == (3, 3, N_TOK_CAM, 128) and got["T_ego2world"][0, 2, 3] == 0
    assert got["T_ego2world"][0, 0, 3] == 2
    # reopen for writing (a resumed shard): continues at the published head
    nb2 = small(root, 10)
    assert (nb2.n, nb2.head) == (3, 3)
    nb2.append(full_row(3, t=3))
    nb2.flush()
    assert reader.refresh() == 4 and neck_ident(reader.read([3])["neck_maps"][0]) == (3, 3, 0)


def test_readers_never_create_or_change_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        NeckBuffer(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    root = tmp_path / "nb"
    nb = small(root, 6)
    nb.append(full_row(0))
    nb.flush()
    before = snapshot(root)
    reader = NeckBuffer(root)
    reader.refresh(), reader.read([0])
    assert snapshot(root) == before
    with pytest.raises(ValueError):            # read-only memmaps
        reader.arrays["neck_maps"][0] = 1
    for call in (lambda: reader.append(full_row(1)), reader.flush, lambda: reader.discard_to(0)):
        with pytest.raises(RuntimeError, match="read-only"):
            call()
    assert snapshot(root) == before
    (root / "neck_3d.npy").unlink()            # a partial copy: refused, not recreated
    with pytest.raises(FileNotFoundError, match="neck_3d"):
        NeckBuffer(root)
    assert not (root / "neck_3d.npy").exists()


def test_writable_reopen_refuses_a_missing_array(tmp_path):
    """V2-2 hardening: an existing buffer (header present) that lost an array is refused on a writable reopen (the
    resume path), never recreated as zeros under rows already recorded; a fresh buffer still creates every array,
    also after a first creation killed before its header."""
    root = tmp_path / "nb"
    nb = small(root, 6)
    for w in range(3):
        nb.append(full_row(w, t=w))
    nb.flush()
    del nb
    for name in ("neck_maps", "t_in_ep"):
        p = root / f"{name}.npy"
        keep = p.read_bytes()
        p.unlink()
        before = snapshot(root)
        with pytest.raises(FileNotFoundError, match=f"{name}.npy: missing in an existing neck buffer"):
            small(root, 6)
        assert not p.exists() and snapshot(root) == before            # nothing created or changed
        p.write_bytes(keep)
    nb2 = small(root, 6)                                              # restored: reopens and keeps its rows
    assert nb2.n == 3 and [neck_ident(m)[0] for m in nb2.read(np.arange(3))["neck_maps"]] == [0, 1, 2]
    # a first creation killed between the arrays and the header: no header, so still fresh, every array created
    part = tmp_path / "part"
    small(part, 6)
    (part / "header.json").unlink()
    for name in ("neck_3d", "episode_id"):
        (part / f"{name}.npy").unlink()
    nb3 = small(part, 6)
    assert set(nb3.arrays) == set(NECK_FIELDS) and (part / "header.json").exists() and nb3.n == 0
    assert all((part / f"{name}.npy").exists() for name in NECK_FIELDS)


def test_header_mismatch_refused(tmp_path):
    root = tmp_path / "nb"
    small(root, 6)
    with pytest.raises(ValueError, match="capacity"):
        small(root, 7)
    with pytest.raises(ValueError, match="fields|level_shapes"):
        NeckBuffer(root, capacity=6, level_shapes=((2, 4), (1, 3), (1, 1), (1, 2)))


def test_validate_row_writes_nothing(tmp_path):
    nb = small(tmp_path / "nb", 6)
    good = full_row(0)
    bad = [({k: v for k, v in good.items() if k != "neck_3d"}, KeyError),
           (dict(good, extra=1), ValueError),
           (dict(good, neck_maps=good["neck_maps"][:, :5]), ValueError),
           (dict(good, episode_id=1000.5), ValueError)]          # a float into an integer field
    for row, exc in bad:
        with pytest.raises(exc):
            nb.append(row)
    assert (nb.n, nb.head) == (0, 0)
    # fp32 maps (the live neck dict) are stored as fp16; float64 matrices as fp32
    nb.append(dict(good, neck_maps=good["neck_maps"].astype(np.float32),
                   projection_mat=good["projection_mat"].astype(np.float64)))
    assert nb.arrays["neck_maps"].dtype == np.float16 and nb.n == 1


def test_discard_to_before_wrap(tmp_path):
    nb = small(tmp_path / "nb", 10)
    for w in range(6):
        nb.append(full_row(w))
    nb.discard_to(4)
    assert (nb.n, nb.head) == (4, 4) and (nb.header["n"], nb.header["head"]) == (4, 4)
    nb.append(full_row(40))
    assert neck_ident(nb.read([4])["neck_maps"][0])[0] == 40
    with pytest.raises(IndexError):            # never written
        nb.read([5])
    with pytest.raises(AssertionError):
        nb.discard_to(7)


def test_ring_wrap(tmp_path):
    nb = small(tmp_path / "nb", 5)
    for w in range(8):
        nb.append(full_row(w, t=w))
    assert (nb.n, nb.head) == (5, 3)
    got = nb.read(np.arange(5))
    assert [neck_ident(m)[0] for m in got["neck_maps"]] == [5, 6, 7, 3, 4]   # the oldest rows overwritten
    assert got["t_in_ep"].tolist() == [5, 6, 7, 3, 4]
    with pytest.raises(AssertionError, match="wrapped"):
        nb.discard_to(4)
    nb.flush()
    r = NeckBuffer(nb.root)
    assert (r.n, r.head) == (5, 3) and r.read([4])["t_in_ep"][0] == 4


def test_for_step_buffer_pairing(tmp_path):
    buf = StepBuffer(tmp_path / "a" / "buffer", capacity=8, header={"fake": True})
    nb = NeckBuffer.for_step_buffer(buf, level_shapes=FAKE_LEVEL_SHAPES)
    assert nb.root == (tmp_path / "a" / "neck_buffer").resolve() and nb.capacity == 8
    rng = np.random.default_rng(0)
    buf.append(step_entry(rng, 0, 0, 0, 1000, 0.5, False))
    with pytest.raises(ValueError, match="out of lockstep"):        # an existing neck one row behind
        NeckBuffer.for_step_buffer(buf, level_shapes=FAKE_LEVEL_SHAPES)
    with pytest.raises(ValueError, match="fresh neck buffer"):      # a fresh neck next to rows it never saw
        NeckBuffer.for_step_buffer(buf, root=tmp_path / "other", level_shapes=FAKE_LEVEL_SHAPES)
    big = StepBuffer(tmp_path / "b" / "buffer", capacity=9, header={"fake": True})
    with pytest.raises(ValueError, match="capacity"):
        NeckBuffer.for_step_buffer(big, root=nb.root, level_shapes=FAKE_LEVEL_SHAPES)


# ------------------------------------------------------------------ lockstep ----
class SpyStepBuffer(StepBuffer):
    """Records the keys of every entry that reaches StepBuffer.append."""

    def append(self, step):
        self.__dict__.setdefault("seen", []).append(set(step))
        super().append(step)


class TickError(RuntimeError):
    pass


def open_pair(root, fresh: bool):
    buf = SpyStepBuffer(root / "buffer", capacity=200, header={"fake": True}) if fresh \
        else SpyStepBuffer(root / "buffer")
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=N_TOK_CAM, k_max=8, r_max=16, g_max=4)
    nb = NeckBuffer.for_step_buffer(buf, level_shapes=FAKE_LEVEL_SHAPES)
    return buf, side, nb


class Capture:
    """A capture counter: every capture gets a unique identity w (never reused after a rollback)."""
    w = 0


def run_episode(writer, inner, rng, e, eid, L, fail_at=None):
    """The rollout protocol: each capture offers its item row to the h_fn's row sink (the LockstepItemWriter),
    attaches _neck to its entry and appends the PREVIOUS entry to ro.buf (one capture late); _flush_pending
    writes the last one with done."""
    rec = template("gap_place")
    H = h_pattern(len(rec["rules"]), L, e)
    pending = None
    for t in range(L):
        inner.offer(item_row(rec, H[t], t, eid))             # h_fn at the capture
        if t == fail_at:
            raise TickError(t)
        entry = step_entry(rng, Capture.w, t, e, eid, float(H[t].min()), False)
        entry["_neck"] = neck_row(Capture.w, t, e)
        Capture.w += 1
        if pending is not None:
            writer.append(pending)
        pending = entry
    pending["done"] = np.bool_(True)
    writer.append(pending)


def collect(buf, side, nb, rng, e, eid, L, fail_at=None):
    """collect.py's loop around one episode: wire_item_episode wraps the StepBuffer in a LockstepItemWriter,
    version 2 wraps that in a NeckLockstepWriter (ro.buf); end_episode at the end; on a tick error the queued
    rows are cleared through ro.buf and the three rings roll back to the episode start."""
    r0 = len(buf)
    inner = LockstepItemWriter(buf, side)
    writer = NeckLockstepWriter(inner, nb)
    try:
        run_episode(writer, inner, rng, e, eid, L, fail_at)
        writer.end_episode()
        return True
    except TickError:
        getattr(writer, "_rows", []).clear()
        assert not inner._rows
        for b in (buf, side, nb):
            b.discard_to(r0)
        return False


def check_aligned(root):
    """Read back with read-only readers: slot i of the three rings is the same capture."""
    ep_b = np.load(root / "buffer" / "episode_id.npy", mmap_mode="r")
    pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")
    side = json.loads((root / "item_buffer" / "header.json").read_text())
    ep_s = np.load(root / "item_buffer" / "episode_id.npy", mmap_mode="r")
    t_s = np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")
    nb = NeckBuffer(root / "neck_buffer")
    n = nb.n
    assert n == side["n"] == json.loads((root / "buffer" / "header.json").read_text())["n"]
    got = nb.read(np.arange(n))
    assert np.array_equal(got["episode_id"], ep_b[:n]) and np.array_equal(got["episode_id"], ep_s[:n])
    assert np.array_equal(got["t_in_ep"], t_s[:n])
    for r in range(n):
        assert neck_ident(got["neck_maps"][r]) == tuple(int(x) for x in pos[r, 0])
    return n


def test_lockstep_append_rollback_resume(tmp_path):
    root = tmp_path / "shard"
    rng = np.random.default_rng(0)
    buf, side, nb = open_pair(root, fresh=True)
    assert collect(buf, side, nb, rng, 0, 1000, 12)
    assert collect(buf, side, nb, rng, 1, 1001, 9)
    assert not collect(buf, side, nb, rng, 2, 1002, 10, fail_at=5)    # 4 rows written, then rolled back
    assert len(buf) == len(side) == len(nb) == 21
    assert collect(buf, side, nb, rng, 3, 1003, 7)
    for b in (buf, side, nb):
        b.flush()
    assert check_aligned(root) == 28
    seen = buf.seen
    assert seen and not any("_neck" in keys for keys in seen)          # the neck never reached the StepBuffer
    assert "_neck" not in buf.arrays and not (root / "buffer" / "_neck.npy").exists()
    # resume: reopen the three rings (for_step_buffer checks n / head), collect one more episode
    buf2, side2, nb2 = open_pair(root, fresh=False)
    assert (nb2.n, nb2.head) == (buf2.n, buf2.head) == (28, 28)
    assert collect(buf2, side2, nb2, rng, 4, 1004, 6)
    for b in (buf2, side2, nb2):
        b.flush()
    assert check_aligned(root) == 34
    eids = NeckBuffer(root / "neck_buffer").read(np.arange(34))["episode_id"]
    assert 1002 not in eids.tolist() and eids[-1] == 1004


def test_resume_with_a_stale_neck_header_is_refused(tmp_path):
    root = tmp_path / "shard"
    rng = np.random.default_rng(0)
    buf, side, nb = open_pair(root, fresh=True)
    nb.flush()                                   # the neck header published at 0 rows only
    assert collect(buf, side, nb, rng, 0, 1000, 5)
    buf.flush(), side.flush()
    with pytest.raises(ValueError, match="out of lockstep"):
        open_pair(root, fresh=False)


def test_lockstep_writer_failures_move_nothing(tmp_path):
    rng = np.random.default_rng(0)
    buf, side, nb = open_pair(tmp_path / "shard", fresh=True)
    inner = LockstepItemWriter(buf, side)
    writer = NeckLockstepWriter(inner, nb)
    rec = template("gap_place")
    H = h_pattern(len(rec["rules"]), 4, 0)
    assert writer.capacity == buf.capacity and writer._rows is inner._rows    # passed through
    writer.offer(item_row(rec, H[0], 0, 1000))
    entry = step_entry(rng, 0, 0, 0, 1000, 0.5, False)
    with pytest.raises(KeyError, match="_neck"):                             # keep_neck off
        writer.append(entry)
    entry["_neck"] = dict(neck_row(0, 0, 0), neck_3d=np.zeros((3, 2, 128), np.float16))
    with pytest.raises(ValueError, match="neck_3d"):                         # malformed: before either ring moves
        writer.append(entry)
    assert "_neck" in entry and (len(buf), len(side), len(nb)) == (0, 0, 0) and len(inner._rows) == 1
    entry["_neck"] = neck_row(0, 0, 0)
    entry["episode_id"] = np.int32(999)                                      # not the queued item row's episode
    with pytest.raises(RuntimeError, match="episode"):
        writer.append(entry)
    assert (len(buf), len(side), len(nb)) == (0, 0, 0) and "_neck" in entry
    entry["episode_id"] = np.int32(1000)
    writer.append(entry)
    assert (len(buf), len(side), len(nb)) == (1, 1, 1) and "_neck" not in entry
    assert nb.arrays["t_in_ep"][0] == 0 and nb.arrays["episode_id"][0] == 1000
    # an append with no queued item row: the inner writer refuses, nothing moves, the entry keeps its neck
    e2 = dict(step_entry(rng, 1, 1, 0, 1000, 0.5, True), _neck=neck_row(1, 1, 0))
    with pytest.raises(RuntimeError, match="queued item row"):
        writer.append(e2)
    assert (len(buf), len(side), len(nb)) == (1, 1, 1) and "_neck" in e2
    writer.end_episode()
    nb.discard_to(0)                                                         # a neck ring left behind
    with pytest.raises(RuntimeError, match="neck ring"):
        writer.end_episode()


def test_entry_t_in_ep_wins_when_present(tmp_path):
    rng = np.random.default_rng(0)
    buf, side, nb = open_pair(tmp_path / "shard", fresh=True)
    inner = LockstepItemWriter(buf, side)
    writer = NeckLockstepWriter(inner, nb)
    rec = template("gap_place")
    H = h_pattern(len(rec["rules"]), 4, 0)
    writer.offer(item_row(rec, H[3], 3, 1000))
    entry = dict(step_entry(rng, 0, 3, 0, 1000, 0.5, False), _neck=neck_row(0, 3, 0))
    assert writer._align(entry)["t_in_ep"] == 3                              # from the queued item row
    assert writer._align(dict(entry, t_in_ep=7))["t_in_ep"] == 7
