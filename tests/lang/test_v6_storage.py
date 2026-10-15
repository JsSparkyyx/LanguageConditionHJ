"""V6 storage tests (qwen_impl.md 4): the kv / frames ring sidecars, their lockstep with a StepBuffer + item
sidecar (rollback, resume, malformed rows), the bf16 bit round trip, the pre-10-12 T1 pool against objects.yaml at
7c2476d^, and the item masks on the Qwen grid. CPU only."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter
from main.lang import legacy_pool, qwen_masks
from main.lang.frames_buffer import FramesBuffer, frames_to_row
from main.lang.kv_buffer import KvBuffer, kv_to_row, rows_to_kv
from main.lang.sidecar import ALIGN_FIELDS, SidecarLockstepWriter
from main.train.buffer import StepBuffer

from .fake_data import N_TOK_CAM, h_pattern, item_row, step_entry, template

KV_DIMS = {"prefix_len": 5, "n_kv_heads": 2, "head_dim": 4}
FR_DIMS = {"native_wh": (8, 6)}
FIXTURE = Path(__file__).parent / "fixtures" / "objects_pre_1012.yaml"
REPO = Path(__file__).resolve().parents[2]


def kv_row(w):
    k = torch.full((2, 5, 4), float(w), dtype=torch.bfloat16)
    return kv_to_row(k, -k)


def frames_row(w):
    native = {"rgb": np.full((3, 6, 8, 3), w % 256, np.uint8), "depth": np.full((3, 6, 8), 0.5 + w * 1e-3, np.float32),
              "K": np.tile(np.eye(3), (3, 1, 1)), "ext": np.tile(np.eye(4), (3, 1, 1))}
    return frames_to_row(native, T_base2ego=np.eye(4) * (w + 1))


def kv_ident(row_k) -> int:
    return int(torch.from_numpy(np.ascontiguousarray(row_k)).view(torch.bfloat16)[0, 0, 0].float())


# ------------------------------------------------------------- the rings ----
def test_kv_contract_fields_and_header(tmp_path):
    kb = KvBuffer(tmp_path / "kv_buffer", capacity=4)
    assert kb.fields["k"] == ((2, 325, 128), np.dtype(np.int16)) and kb.fields["v"] == kb.fields["k"]
    assert {k: kb.fields[k] for k in ALIGN_FIELDS} == ALIGN_FIELDS
    h = json.loads((tmp_path / "kv_buffer" / "header.json").read_text())
    assert (h["kv_version"], h["prefix_len"], h["rope_next"], h["n_kv_heads"], h["head_dim"]) == (1, 325, 61, 2, 128)
    assert h["value_dtype"] == "bfloat16 bits in int16" and h["key_rope"] == "applied"
    row_bytes = sum(int(np.prod(s)) * d.itemsize for s, d in kb.fields.values())
    assert row_bytes == 332_812                                     # qwen_impl.md 4
    fb = FramesBuffer(tmp_path / "frames", capacity=2)
    assert fb.fields["rgb"] == ((3, 240, 320, 3), np.dtype(np.uint8))
    assert fb.fields["depth"] == ((3, 240, 320), np.dtype(np.float16))
    assert json.loads((tmp_path / "frames" / "header.json").read_text())["camera_names"] == \
        ["left_wrist", "right_wrist", "head"]


def test_bf16_bits_round_trip_exactly():
    g = torch.Generator().manual_seed(0)
    k = (torch.randn(1, 2, 7, 4, generator=g) * 50).to(torch.bfloat16)
    v = torch.randn(2, 7, 4, generator=g).to(torch.bfloat16)
    row = kv_to_row(k, v)
    assert row["k"].dtype == np.int16 and row["k"].shape == (2, 7, 4)
    K, V = rows_to_kv(row["k"][None], row["v"][None], device="cpu")
    assert K.dtype == torch.bfloat16 and torch.equal(K[0], k[0]) and torch.equal(V[0], v)
    with pytest.raises(TypeError, match="bfloat16"):
        kv_to_row(k.float(), v)
    with pytest.raises(ValueError, match="one frame"):
        kv_to_row(torch.cat([k, k]), v)
    with pytest.raises(ValueError, match="non-finite"):
        kv_to_row(k, torch.full_like(v, float("nan")))
    with pytest.raises(TypeError, match="int16"):
        rows_to_kv(row["k"].astype(np.float32), row["v"], device="cpu")


def test_kv_reader_refuses_another_checkpoint(tmp_path):
    kb = KvBuffer(tmp_path / "kv", capacity=3, ckpt_id="aaaa", attn="sdpa", **KV_DIMS)
    kb.append(dict(kv_row(1), episode_id=5, t_in_ep=0))
    kb.flush()
    assert len(KvBuffer.open_read(tmp_path / "kv", ckpt_id="aaaa")) == 1
    with pytest.raises(ValueError, match="checkpoint"):
        KvBuffer.open_read(tmp_path / "kv", ckpt_id="bbbb")
    with pytest.raises(ValueError, match="ckpt_id"):            # a writable reopen under another checkpoint
        KvBuffer(tmp_path / "kv", capacity=3, ckpt_id="bbbb", attn="sdpa", **KV_DIMS)


def test_ring_semantics_and_readers(tmp_path):
    kb = KvBuffer(tmp_path / "kv", capacity=3, **KV_DIMS)
    for w in range(5):
        kb.append(dict(kv_row(w), episode_id=10, t_in_ep=w))
    kb.flush()
    assert (kb.n, kb.head) == (3, 2)
    got = KvBuffer(tmp_path / "kv").read([0, 1, 2])
    assert [kv_ident(k) for k in got["k"]] == [3, 4, 2] and got["t_in_ep"].tolist() == [3, 4, 2]
    with pytest.raises(AssertionError, match="wrapped"):
        kb.discard_to(1)
    ro = KvBuffer(tmp_path / "kv")
    with pytest.raises(RuntimeError, match="read-only"):
        ro.append(dict(kv_row(0), episode_id=0, t_in_ep=0))
    with pytest.raises(IndexError):
        FramesBuffer(tmp_path / "fr", capacity=2, **FR_DIMS).read([0])   # nothing written yet
    ro.check_rows([0, 1], [10, 10], [3, 4])
    with pytest.raises(RuntimeError, match="out of lockstep"):
        ro.check_rows([0], [10], [4])


def test_frames_row_checks():
    native = {"rgb": np.zeros((3, 6, 8, 3), np.uint8), "depth": np.zeros((3, 6, 8), np.float32),
              "K": np.zeros((3, 3, 3)), "ext": np.zeros((3, 4, 4))}
    assert np.isnan(frames_to_row(native)["T_base2ego"]).all()
    with pytest.raises(ValueError, match="non-finite"):
        frames_to_row(dict(native, depth=np.full((3, 6, 8), np.inf, np.float32)))
    with pytest.raises(TypeError, match="uint8"):
        frames_to_row(dict(native, rgb=np.zeros((3, 6, 8, 3), np.float32)))


# ------------------------------------------------------------- lockstep ----
class TickError(RuntimeError):
    pass


class Capture:
    w = 0


def open_rings(root, fresh: bool, frames: bool = True):
    buf = StepBuffer(root / "buffer", capacity=200, header={"fake": True}) if fresh else StepBuffer(root / "buffer")
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=N_TOK_CAM, k_max=8, r_max=16, g_max=4)
    rings = {"_kv": KvBuffer.for_step_buffer(buf, ckpt_id="c0", **KV_DIMS)}
    if frames:
        rings["_frames"] = FramesBuffer.for_step_buffer(buf, **FR_DIMS)
    return buf, side, rings


def run_episode(writer, inner, rng, e, eid, L, keys, fail_at=None):
    """The rollout protocol: one capture late, each sidecar row riding in the pending entry."""
    rec = template("gap_place")
    H = h_pattern(len(rec["rules"]), L, e)
    pending = None
    for t in range(L):
        inner.offer(item_row(rec, H[t], t, eid))
        if t == fail_at:
            raise TickError(t)
        entry = step_entry(rng, Capture.w, t, e, eid, float(H[t].min()), False)
        if "_kv" in keys:
            entry["_kv"] = kv_row(Capture.w)
        if "_frames" in keys:
            entry["_frames"] = frames_row(Capture.w)
        Capture.w += 1
        if pending is not None:
            writer.append(pending)
        pending = entry
    pending["done"] = np.bool_(True)
    writer.append(pending)


def collect(buf, side, rings, rng, e, eid, L, fail_at=None):
    r0 = len(buf)
    inner = LockstepItemWriter(buf, side)
    writer = SidecarLockstepWriter(inner, rings)
    try:
        run_episode(writer, inner, rng, e, eid, L, set(rings), fail_at)
        writer.end_episode()
        return True
    except TickError:
        inner._rows.clear()
        for b in (buf, side, *rings.values()):
            b.discard_to(r0)
        return False


def check_aligned(root):
    ep_b = np.load(root / "buffer" / "episode_id.npy", mmap_mode="r")
    tok = np.load(root / "buffer" / "scene_tokens.npy", mmap_mode="r")
    t_s = np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")
    kb, fb = KvBuffer(root / "kv_buffer"), FramesBuffer(root / "frames")
    n = kb.n
    assert n == fb.n == json.loads((root / "buffer" / "header.json").read_text())["n"]
    k = kb.read(np.arange(n))
    f = fb.read(np.arange(n))
    assert np.array_equal(k["episode_id"], ep_b[:n]) and np.array_equal(f["episode_id"], ep_b[:n])
    assert np.array_equal(k["t_in_ep"], t_s[:n]) and np.array_equal(f["t_in_ep"], t_s[:n])
    for r in range(n):
        w = kv_ident(k["k"][r])
        assert round(float(tok[r, 0, 0]) * 1000) == w % 1000 and int(f["rgb"][r, 0, 0, 0, 0]) == w % 256
        assert f["T_base2ego"][r, 0, 0] == w + 1
    return n


def test_lockstep_append_rollback_resume(tmp_path):
    root = tmp_path / "shard"
    rng = np.random.default_rng(0)
    buf, side, rings = open_rings(root, fresh=True)
    assert collect(buf, side, rings, rng, 0, 1000, 12)
    assert collect(buf, side, rings, rng, 1, 1001, 9)
    assert not collect(buf, side, rings, rng, 2, 1002, 10, fail_at=5)
    assert len(buf) == len(side) == len(rings["_kv"]) == len(rings["_frames"]) == 21
    assert collect(buf, side, rings, rng, 3, 1003, 7)
    for b in (buf, side, *rings.values()):
        b.flush()
    assert check_aligned(root) == 28
    assert not any(p.name.startswith("_") for p in (root / "buffer").iterdir())
    buf2, side2, rings2 = open_rings(root, fresh=False)
    assert (rings2["_kv"].n, rings2["_frames"].head) == (buf2.n, buf2.head) == (28, 28)
    assert collect(buf2, side2, rings2, rng, 4, 1004, 6)
    for b in (buf2, side2, *rings2.values()):
        b.flush()
    assert check_aligned(root) == 34


def test_resume_with_a_stale_sidecar_header_is_refused(tmp_path):
    root = tmp_path / "shard"
    rng = np.random.default_rng(0)
    buf, side, rings = open_rings(root, fresh=True)
    rings["_kv"].flush()
    assert collect(buf, side, rings, rng, 0, 1000, 5)
    buf.flush(), side.flush(), rings["_frames"].flush()
    with pytest.raises(ValueError, match="out of lockstep"):
        open_rings(root, fresh=False)


def test_fresh_sidecar_next_to_a_full_buffer_is_refused(tmp_path):
    root = tmp_path / "shard"
    rng = np.random.default_rng(0)
    buf, side, rings = open_rings(root, fresh=True, frames=False)
    assert collect(buf, side, rings, rng, 0, 1000, 4)
    with pytest.raises(ValueError, match="fresh"):
        FramesBuffer.for_step_buffer(buf, **FR_DIMS)


def test_lockstep_failures_move_nothing(tmp_path):
    rng = np.random.default_rng(0)
    buf, side, rings = open_rings(tmp_path / "shard", fresh=True)
    inner = LockstepItemWriter(buf, side)
    writer = SidecarLockstepWriter(inner, rings)
    rec = template("gap_place")
    H = h_pattern(len(rec["rules"]), 4, 0)
    writer.offer(item_row(rec, H[0], 0, 1000))
    sizes = lambda: (len(buf), len(side), len(rings["_kv"]), len(rings["_frames"]))   # noqa: E731
    entry = dict(step_entry(rng, 0, 0, 0, 1000, 0.5, False), _kv=kv_row(0))
    with pytest.raises(KeyError, match="_frames"):
        writer.append(entry)
    assert "_kv" in entry and sizes() == (0, 0, 0, 0)
    entry["_frames"] = dict(frames_row(0), depth=np.zeros((3, 2, 2), np.float16))
    with pytest.raises(ValueError, match="depth"):
        writer.append(entry)
    assert "_kv" in entry and "_frames" in entry and sizes() == (0, 0, 0, 0) and len(inner._rows) == 1
    entry["_frames"] = frames_row(0)
    entry["episode_id"] = np.int32(999)
    with pytest.raises(RuntimeError, match="episode"):
        writer.append(entry)
    entry["episode_id"] = np.int32(1000)
    writer.append(entry)
    assert sizes() == (1, 1, 1, 1) and "_kv" not in entry and "_frames" not in entry
    assert rings["_kv"].arrays["episode_id"][0] == 1000
    writer.end_episode()
    rings["_frames"].discard_to(0)
    with pytest.raises(RuntimeError, match="_frames ring"):
        writer.end_episode()


# ------------------------------------------------------------ old T1 pool ----
def test_pre_1012_pool_rebuilds_the_old_pool():
    from main.item_envs import spawn

    cur = yaml.safe_load((REPO / "envs/main/item_envs/configs/objects.yaml").read_text())
    old = yaml.safe_load(FIXTURE.read_text())
    original = spawn.t1_pool_entries
    want = original(old)
    try:
        assert original(cur) != want                                     # the pools did change on 10-12
        legacy_pool.install_spawn_pool("pre_1012")
        legacy_pool.install_spawn_pool("pre_1012")                       # idempotent: no double wrap
        assert legacy_pool.installed_pool() == "pre_1012"
        got = spawn.t1_pool_entries(cur)
        assert [e["asset"] for e in got] == [e["asset"] for e in want]
        for g, w in zip(got, want):
            assert {k: g.get(k) for k in w} == w, w["asset"]             # every old field, ids and category included
        cats = old.get("categories") or {}
        for e in got:                                                    # physics reads the category's mass range
            assert (cur["categories"].get(e["category"]) or {}).get("mass_range") == \
                (cats.get(e["category"]) or {}).get("mass_range"), e["category"]
    finally:
        legacy_pool.install_spawn_pool("current")
    assert spawn.t1_pool_entries is original and legacy_pool.installed_pool() == "current"
    with pytest.raises(ValueError, match="spawn_pool"):
        legacy_pool.install_spawn_pool("old")


# ------------------------------------------------------------- Qwen grid ----
def test_qwen_grid_is_the_scene_token_raster():
    scales, n = qwen_masks.qwen_token_grid()
    assert scales == [(28, 9, 11)] and n == 99 == qwen_masks.QWEN_N_TOK
    ids = np.zeros((240, 320), np.int64)
    ids[:, 300:] = 7                                   # the right 20 native columns: the last Qwen column only
    cov = qwen_masks.pool_ids_to_qwen(ids, [7]).reshape(9, 11)
    assert (cov[:, :10] == 0).all() and (cov[:, 10] > 0).all()
    ids = np.zeros((240, 320), np.int64)
    ids[:20] = 3                                       # the top 20 native rows -> the first Qwen row (21 of 28 px)
    cov = qwen_masks.pool_ids_to_qwen(ids, [3]).reshape(9, 11)
    assert (cov[0] > 0).all() and (cov[1:] == 0).all()


def test_qwen_grid_h_fn_replaces_the_masks(monkeypatch):
    from main.item_envs import margin_hook

    seen = []

    def parent(self, env):
        seen.append(self.with_masks)
        d = margin_hook._TickData()
        d.visibility = np.zeros((2, 3), np.float32)
        d.token_masks = np.zeros((3, 3, 400), np.uint8)
        return d

    class Region:
        nameable, entity = True, object()

    ctx = type("Ctx", (), {"items": [0, 1], "regions": [Region()]})()
    monkeypatch.setattr(margin_hook.ItemMarginHFn, "_compute_tick", parent)
    monkeypatch.setattr(qwen_masks, "qwen_item_masks",
                        lambda task, items, regions: (np.ones((2, 3), np.float32), np.full((3, 3, 99), 9, np.uint8)))
    ro = type("Ro", (), {})()
    ro.h_fn = margin_hook.ItemMarginHFn("r", 1.0, ctx=ctx)
    qwen_masks.use_qwen_grid(ro)
    qwen_masks.use_qwen_grid(ro)
    assert type(ro.h_fn) is qwen_masks.QwenGridMarginHFn and ro.h_fn.active_rule_id == "r"
    env = type("Env", (), {"task": None})()
    d = ro.h_fn._compute_tick(env)
    assert seen == [False] and ro.h_fn.with_masks and d.token_masks.shape == (3, 3, 99) and (d.token_masks == 9).all()
    ro.h_fn.with_masks = False
    d = ro.h_fn._compute_tick(env)
    assert d.token_masks.shape == (3, 3, 99) and not d.token_masks.any() and not d.visibility.any()
    ro.h_fn = object()
    with pytest.raises(TypeError, match="ItemMarginHFn"):
        qwen_masks.use_qwen_grid(ro)
