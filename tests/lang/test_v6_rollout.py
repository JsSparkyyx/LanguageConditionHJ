"""The rollout side of V6 (qwen_impl.md 5), CPU, no simulator, a stub HoloBrainQwenEncoder.

- ``_encode_obs`` under qwen: the Qwen frame encoding (never the GD visual path), GD's robot-state encoder, the
  prefix K / V kept as ``_kv_now`` always and as this capture's row source only inside a capture;
- ``_collect_tick``: each capture's kv / frames rows ride in the entry the REAL base class writes one capture late;
  through the real StepBuffer + item sidecar + KvBuffer + FramesBuffer and SidecarLockstepWriter every slot is the
  same capture; a missing row and a token beyond fp16 raise before anything is stashed;
- LangPolicyShim under qwen: the sentence (and the counterfactual one) read the current frame's prefix every call,
  "blank" reads the black-image prefix, set_instruction checks the token budget;
- the constructor's encoder / config checks (before any env is built).
"""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter
from main.lang.config import LangConfig
from main.lang.frames_buffer import FramesBuffer
from main.lang.kv_buffer import KvBuffer
from main.lang.network import LangPolicyEncoder
from main.lang.rollout import LangPolicyShim, LangRolloutController
from main.lang.sidecar import SidecarLockstepWriter
from main.train.buffer import StepBuffer

from .fake_data import N_TOK_CAM, h_pattern, item_row, template
from .test_collect_v2 import DIAG, StubExtractor
from .test_network import body_inputs

KV_DIMS = {"prefix_len": 5, "n_kv_heads": 2, "head_dim": 4}
FR_DIMS = {"native_wh": (8, 6)}


class StubQwen:
    """encode_frame: tokens / positions / K carry the frame identity w (obs["rgb"][0, 0, 0, 0]); encode_text: the
    words carry (the prefix's w, the sentence's length)."""

    def __init__(self):
        self.frames = self.texts = 0
        self.seen = []

    def encode_frame(self, rgb, depth, proj, T):
        self.frames += 1
        w = float(np.asarray(rgb)[0, 0, 0, 0, 0])
        tok = torch.full((1, 3, N_TOK_CAM, 256), w / 1000.0)
        pos = torch.zeros(1, 3, N_TOK_CAM, 3)
        pos[0, 0, 0, 0] = w
        K = torch.full((1, 2, 5, 4), w, dtype=torch.bfloat16)
        return tok, pos, (K, -K)

    def blank_kv(self):
        K = torch.full((1, 2, 5, 4), -7.0, dtype=torch.bfloat16)
        return K, K

    def instruction_ids(self, s):
        if len(s.split()) > 24:
            raise ValueError("word budget")
        return list(range(len(s.split())))

    def encode_text(self, kv, sentences, kv_index=None):
        self.texts += 1
        self.seen.append((float(kv[0][0, 0, 0, 0]), list(sentences)))
        w = torch.zeros(len(sentences), 24, 256)
        m = torch.zeros(len(sentences), 24, dtype=torch.bool)
        for i, s in enumerate(sentences):
            w[i, 0, 0] = float(kv[0][0, 0, 0, 0])
            m[i, :len(s.split())] = True
        return w, m


class StubGD:
    def __init__(self):
        self.visual = 0

    def encode_visual_tokens(self, *a, **k):
        self.visual += 1
        raise AssertionError("the GD visual path ran under qwen")

    def encode_joint_angles(self, joint_state, kin, embodiedment_mat=None):
        return torch.zeros(1, 14, 1, 256)


def qobs(w, t=0):
    native = {"rgb": np.full((3, 6, 8, 3), w % 256, np.uint8), "depth": np.full((3, 6, 8), 0.5, np.float32),
              "K": np.tile(np.eye(3), (3, 1, 1)), "ext": np.tile(np.eye(4), (3, 1, 1))}
    rgb = np.zeros((3, 252, 308, 3), np.uint8)
    rgb[0, 0, 0, 0] = w % 256
    return {"qwen": True, "rgb": rgb, "depth": np.zeros((3, 252, 308), np.float32),
            "projection_mat": np.tile(np.eye(4), (3, 1, 1)), "T_ego2world": np.eye(4), "T_base2ego": np.eye(4) * (w + 1),
            "joint_state": torch.zeros(1, 1, 14), "native": native}


def bare(**attrs):
    ro = LangRolloutController.__new__(LangRolloutController)
    ro.__dict__.update(dict(
        fusion=None, keep_neck=False, _in_capture=False, _neck_now=None, _neck_capture=None, kin=None,
        qwen=StubQwen(), keep_kv=False, keep_frames=False, _kv_now=None, _kv_capture=None, _frames_capture=None,
        encoder=StubGD(), extractor=StubExtractor(), episode_id=1000, buf=None, _pending=None, trace={},
        cfg=NS(embodiment="ur5-wsg", task="gap_place", task_choices=("gap_place",)),
        _tick_action=np.zeros(16), _tick_source=0, sentence=None, cf_sentence=None, text_out=None, cf_text_out=None,
    ), **attrs)
    return ro


def test_encode_obs_under_qwen():
    ro = bare(keep_kv=True, keep_frames=True)
    ro._in_capture = True
    tokens, pos, state, body = ro._encode_obs(qobs(7))
    assert ro.qwen.frames == 1 and ro.encoder.visual == 0
    assert tokens.shape == (1, 3, N_TOK_CAM, 256) and float(pos[0, 0, 0, 0]) == 7 and state.shape == (1, 14, 256)
    assert ro._kv_capture is ro._kv_now and float(ro._kv_now[0][0, 0, 0, 0]) == 7
    assert ro._frames_capture[0]["rgb"][0, 0, 0, 0] == 7
    ro._in_capture = False
    ro._encode_obs(qobs(8))                                   # a cache miss: _kv_now moves, the capture's stays
    assert float(ro._kv_now[0][0, 0, 0, 0]) == 8 and float(ro._kv_capture[0][0, 0, 0, 0]) == 7
    with pytest.raises(RuntimeError, match="GD observation"):
        ro._encode_obs({"imgs": None})


def test_collect_tick_rows_line_up(tmp_path):
    root = tmp_path / "shard"
    buf = StepBuffer(root / "buffer", capacity=100, header={"fake": True})
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=N_TOK_CAM, k_max=8, r_max=16, g_max=4)
    kb = KvBuffer.for_step_buffer(buf, **KV_DIMS)
    fb = FramesBuffer.for_step_buffer(buf, **FR_DIMS)
    rec = template("gap_place")
    w = 1
    for e, (eid, L) in enumerate(((1000, 6), (1001, 4))):
        inner = LockstepItemWriter(buf, side)
        ro = bare(keep_kv=True, keep_frames=True, episode_id=eid,
                  buf=SidecarLockstepWriter(inner, {"_kv": kb, "_frames": fb}))
        H = h_pattern(len(rec["rules"]), L, e)
        for t in range(L):
            ro._in_capture = True
            inner.offer(item_row(rec, H[t], t, eid))
            encoded = ro._encode_obs(qobs(w, t))
            ro._collect_tick(qobs(w, t), float(H[t].min()), DIAG, encoded)
            ro._in_capture = False
            if t % 2:
                ro._encode_obs(qobs(200))                    # the tick hook's fresh encoding: never a row
            w += 1
        ro._flush_pending()
        ro.buf.end_episode()
    for b in (buf, side, kb, fb):
        b.flush()
    n = 10
    k = KvBuffer(root / "kv_buffer").read(np.arange(n))
    f = FramesBuffer(root / "frames").read(np.arange(n))
    pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")
    ep = np.load(root / "buffer" / "episode_id.npy", mmap_mode="r")
    t_s = np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")
    kw = torch.from_numpy(k["k"][:, 0, 0, 0].copy()).view(torch.bfloat16).float().numpy()
    assert kw.tolist() == pos[:n, 0, 0].tolist() == f["rgb"][:, 0, 0, 0, 0].astype(float).tolist() == \
        list(range(1, 11))
    assert np.array_equal(k["episode_id"], ep[:n]) and np.array_equal(f["t_in_ep"], t_s[:n])
    assert f["T_base2ego"][:, 0, 0].tolist() == [x + 1 for x in range(1, 11)]
    assert json.loads((root / "buffer" / "header.json").read_text())["n"] == n


def test_collect_tick_failures_stash_nothing():
    appended = []
    ro = bare(keep_kv=True, buf=NS(append=appended.append))
    encoded = ro._encode_obs(qobs(1))                         # outside a capture
    with pytest.raises(RuntimeError, match="prefix keys"):
        ro._collect_tick(qobs(1), 0.5, DIAG, encoded)
    assert ro._pending is None
    ro._in_capture = True
    tok, pos, state, body = ro._encode_obs(qobs(2))
    with pytest.raises(RuntimeError, match="fp16"):
        ro._collect_tick(qobs(2), 0.5, DIAG, (tok * 1e8, pos, state, body))
    assert ro._pending is None and not appended


def make_shim(prefix="frame"):
    cfg = LangConfig()
    torch.manual_seed(0)
    pe = LangPolicyEncoder(cfg, n_tok=3 * N_TOK_CAM).eval()
    ro = bare(cfg=NS(qwen_words_prefix=prefix, k_prev=2), timing={})
    return ro, LangPolicyShim(ro, pe, 2)


def call(ro, shim, w, capture=True):
    body, _ = body_inputs("ur5-wsg", 1, torch.Generator().manual_seed(0))
    ro._in_capture = capture
    tok, pos, state, _ = ro._encode_obs(qobs(w))
    shim(body["state_tokens"], body["body_tokens"], body["link_pos"], body["body_mask"], body["joint_index"], tok,
         pos)
    ro._in_capture = False


def test_shim_reads_each_frames_prefix():
    ro, shim = make_shim()
    ro.policy_enc = shim
    LangRolloutController.set_instruction(ro, "keep the mug still", "do not touch the can", rule_id="r1",
                                          cf_rule_id="r2")
    assert ro.text_out is None and ro.named_rule_id == "r1"
    call(ro, shim, 3)
    out3 = ro.text_out
    call(ro, shim, 4)
    assert ro.qwen.seen == [(3.0, ["keep the mug still", "do not touch the can"]),
                            (4.0, ["keep the mug still", "do not touch the can"])]
    assert ro.text_out is not out3 and ro.cf_text_out is not None and shim.enc_cf is not None
    assert ro.timing["text"] > 0
    with pytest.raises(ValueError, match="word budget"):
        LangRolloutController.set_instruction(ro, " ".join(["x"] * 30))


def test_shim_blank_prefix_and_no_sentence():
    ro, shim = make_shim("blank")
    ro.policy_enc = shim
    with pytest.raises(RuntimeError, match="set_instruction"):
        call(ro, shim, 3)
    LangRolloutController.set_instruction(ro, "keep the mug still")
    call(ro, shim, 3)
    assert ro.qwen.seen[-1] == (-7.0, ["keep the mug still"]) and ro.cf_text_out is None


def test_constructor_checks_the_encoder():
    gd = NS(encoder="holobrain_gd", qwen=False, scene_fusion=False, oracle_ref_tag=False)
    qw = NS(encoder="holobrain_qwen", qwen=True, scene_fusion=False, oracle_ref_tag=False)
    with pytest.raises(ValueError, match="exactly when"):
        LangRolloutController(gd, 0, qwen=StubQwen())
    with pytest.raises(ValueError, match="exactly when"):
        LangRolloutController(qw, 0)
    with pytest.raises(ValueError, match="neck"):
        LangRolloutController(qw, 0, qwen=StubQwen(), keep_neck=True)
    with pytest.raises(ValueError, match="need qwen"):
        LangRolloutController(gd, 0, keep_kv=True)
