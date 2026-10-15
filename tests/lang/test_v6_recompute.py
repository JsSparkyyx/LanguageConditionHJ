"""Front-end version 2 (pre-run review 2026-10-15): scripts/lang/qwen_recompute.py rewrites a V6 shard's encoder
outputs from frames/ (CPU, a stub encoder whose outputs encode the frame they were given), the trainer refuses kv rows
of another front-end version or kernel, and the encoder refuses rotary inv_freq buffers that are not float32."""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from main.item_envs.item_buffer import ItemSidecarBuffer
from main.lang.frames_buffer import FramesBuffer
from main.lang.kv_buffer import KvBuffer, kv_to_row
from main.train.buffer import StepBuffer

from .fake_data import N_TOK_CAM, h_pattern, item_row, step_entry, template

REPO = Path(__file__).resolve().parents[2]
KV_DIMS = {"prefix_len": 5, "n_kv_heads": 2, "head_dim": 4}


def _script():
    spec = importlib.util.spec_from_file_location("qwen_recompute_t", REPO / "scripts" / "lang" / "qwen_recompute.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubEnc:
    """encode_frame: every output carries the red value of the frame's left-camera pixel (0, 0)."""

    ENCODER_VERSION, attn, ckpt_id, N_TOK = 2, "flash_attention_2", "c0", 297

    def __init__(self):
        self.frames = 0

    def encode_frame(self, rgb, depth, proj, T):
        rgb = np.asarray(rgb)
        assert rgb.shape[1:] == (3, 252, 308, 3) and np.asarray(depth).shape[1:] == (3, 252, 308)
        F = rgb.shape[0]
        self.frames += F
        w = torch.tensor(rgb[:, 0, 0, 0, 0].astype(np.float32))
        tok = w.view(F, 1, 1, 1).expand(F, 3, 99, 256) / 100.0
        pos = w.view(F, 1, 1, 1).expand(F, 3, 99, 3) / 10.0
        K = (w.view(F, 1, 1, 1).expand(F, 2, 5, 4) + 0.5).to(torch.bfloat16)
        return tok.contiguous(), pos.contiguous(), (K, -K)


def make_v6_shard(root: Path, episodes=2, L=6, complete=True):
    buf = StepBuffer(root / "buffer", capacity=64, header={"fake": True, "encoder": "holobrain_qwen"})
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=N_TOK_CAM, k_max=8, r_max=16, g_max=4)
    kb = KvBuffer.for_step_buffer(buf, ckpt_id="c0", attn="sdpa", **KV_DIMS)
    fb = FramesBuffer.for_step_buffer(buf, native_wh=(8, 6))
    rec = template("gap_place")
    rng = np.random.default_rng(0)
    w = 1
    for e in range(episodes):
        eid = 1000 + e
        H = h_pattern(len(rec["rules"]), L, e)
        for t in range(L):
            entry = step_entry(rng, w, t, e, eid, float(H[t].min()), t == L - 1)
            entry["scene_tokens"] = np.zeros((297, 256), np.float16)
            entry["scene_pos"] = np.zeros((297, 3), np.float32)
            buf.append(entry)
            side.append(item_row(rec, H[t], t, eid))
            z = torch.zeros(2, 5, 4, dtype=torch.bfloat16)
            kb.append(dict(kv_to_row(z, z), episode_id=eid, t_in_ep=t))
            fb.append({"rgb": np.full((3, 6, 8, 3), w, np.uint8), "depth": np.full((3, 6, 8), 0.6, np.float16),
                       "K": np.tile(np.diag([5.0, 5.0, 1.0]), (3, 1, 1)), "ext": np.tile(np.eye(4), (3, 1, 1)),
                       "T_base2ego": np.eye(4), "episode_id": eid, "t_in_ep": t})
            w += 1
    for b in (buf, side, kb, fb):
        b.flush()
    (root / "shard.json").write_text(json.dumps({"encoder": "holobrain_qwen", "frames": True,
                                                 "kv": {"ckpt_id": "c0", "attn": "sdpa", "prefix_len": 5}}))
    (root / "summary.json").write_text(json.dumps({"complete": complete}))
    return root, episodes * L


def test_recompute_rewrites_every_row_and_stamps_the_headers(tmp_path):
    rc_mod = _script()
    root, n = make_v6_shard(tmp_path / "shard")
    enc = StubEnc()
    assert rc_mod.recompute_shard(root, enc, batch=4, log=lambda *_: None) == 0
    tok = np.load(root / "buffer" / "scene_tokens.npy", mmap_mode="r")
    pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")
    kb = KvBuffer(root / "kv_buffer")
    k = torch.from_numpy(np.ascontiguousarray(kb.read(np.arange(n))["k"])).view(torch.bfloat16).float()
    for r in range(n):   # frame r (1-based value r + 1 in its pixels) went to row r, after the bilinear resize
        assert np.allclose(tok[r], (r + 1) / 100.0, atol=1e-3) and np.allclose(pos[r], (r + 1) / 10.0)
        assert float(k[r, 0, 0, 0]) == pytest.approx(r + 1.5, abs=0.06)
    assert enc.frames == n
    assert kb.header["encoder_version"] == 2 and kb.header["attn"] == "flash_attention_2"
    meta = json.loads((root / "shard.json").read_text())
    assert meta["kv"]["encoder_version"] == 2 and meta["kv"]["attn"] == "flash_attention_2" and meta["recomputed"]
    prog = json.loads((root / "recompute.json").read_text())
    assert prog["complete"] and prog["rows_done"] == n
    assert rc_mod.recompute_shard(root, enc, batch=4, log=lambda *_: None) == 0 and enc.frames == n   # done already
    assert prog["frames_per_call"] == 4 and kb.header["frames_per_call"] == 4
    assert rc_mod.recompute_shard(root, enc, log=lambda *_: None) == 0 and enc.frames == 2 * n      # 1 per call: again
    assert json.loads((root / "recompute.json").read_text())["frames_per_call"] == 1
    ep = np.load(root / "buffer" / "episode_id.npy", mmap_mode="r")
    assert ep[:n].tolist() == [1000] * 6 + [1001] * 6          # nothing else moved


def test_recompute_resumes_after_the_deadline(tmp_path):
    rc_mod = _script()
    root, n = make_v6_shard(tmp_path / "shard")
    enc = StubEnc()
    assert rc_mod.recompute_shard(root, enc, batch=4, deadline=time.time() - 1, log=lambda *_: None) == 75
    assert json.loads((root / "recompute.json").read_text())["rows_done"] == 0 and enc.frames == 0
    assert rc_mod.recompute_shard(root, enc, batch=4, flush_every=1, log=lambda *_: None) == 0
    assert enc.frames == n
    enc2 = StubEnc()
    enc2.attn = "sdpa"                                          # another kernel: start over
    assert rc_mod.recompute_shard(root, enc2, batch=4, log=lambda *_: None) == 0 and enc2.frames == n
    enc3 = StubEnc()
    assert rc_mod.recompute_shard(root, enc3, batch=1, deadline=time.time() + 1e6, log=lambda *_: None) == 0
    assert enc3.frames == n                                     # another frames-per-call: start over too


def test_recompute_refuses_what_it_must(tmp_path):
    rc_mod = _script()
    root, _ = make_v6_shard(tmp_path / "busy", complete=False)
    assert rc_mod.recompute_shard(root, StubEnc(), log=lambda *_: None) == 75   # still collecting
    root, _ = make_v6_shard(tmp_path / "other")
    enc = StubEnc()
    enc.ckpt_id = "c9"
    with pytest.raises(SystemExit, match="checkpoint"):
        rc_mod.recompute_shard(root, enc, log=lambda *_: None)
    t = np.load(root / "frames" / "t_in_ep.npy", mmap_mode="r+")
    t[3] = 9
    t.flush()
    with pytest.raises(RuntimeError, match="frames/ is out of lockstep"):
        rc_mod.recompute_shard(root, StubEnc(), log=lambda *_: None)


def test_trainer_refuses_rows_of_another_version_or_kernel(tmp_path):
    from .test_v6_train import StubQwen, make_qwen_root, qtrainer

    root = make_qwen_root(tmp_path / "q")
    stub = StubQwen()
    stub.ENCODER_VERSION = 2
    with pytest.raises(ValueError, match="front-end version 1"):
        qtrainer(tmp_path / "a", root, qwen=stub)
    kb = KvBuffer.reopen(root / "kv_buffer")
    kb.header["encoder_version"] = 2
    kb._write_header()
    with pytest.raises(ValueError, match="attention sdpa"):
        qtrainer(tmp_path / "b", root, qwen=stub, qwen_attn="flash_attention_2")
    assert qtrainer(tmp_path / "c", root, qwen=stub).qwen is stub


def test_encoder_refuses_bf16_rope_buffers():
    from main.lang import qwen_encoder as qe

    from .test_qwen_encoder import tiny_parts

    vlm, fm, b3, se, proc = tiny_parts()
    vlm.to(torch.bfloat16)                                      # what version 1 did: buffers rounded too
    with pytest.raises(RuntimeError, match="inv_freq"):
        qe.HoloBrainQwenEncoder.from_parts(vlm, fm, b3, se, proc, device="cpu")
