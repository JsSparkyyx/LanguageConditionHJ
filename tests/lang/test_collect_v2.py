"""Version-2 collection and the rollout side of V5 (fusion_impl.md 4.6), CPU, no simulator.

- LangRolloutController: ``_encode_obs`` keeps the neck (the base path when both switches are off);
  ``_collect_tick`` attaches each capture's neck row to the entry the base class writes one capture late,
  checked through the REAL base ``_collect_tick`` / ``_flush_pending`` and the real lockstep writers.
- LangPolicyShim with a stub FusionStage (a deterministic function of frame and sentence): one fused history per
  sentence, frames [t, t-k, 0], text_out recomputed per frame with fused_words, the C5 / R4 checks.
- collect.run_episodes with a stand-in controller that follows the base rollout's row protocol: the neck ring
  guarded, rolled back after a tick error, flushed, lang_version on every line; replayed slots keep their seeds.
- resume_point / open_neck with a neck buffer; plan_shard / shard_meta_diff (a version-1 meta is unchanged);
  ``--replay`` end to end (slots and ids equal the recorded ones, resumes after kills, a mismatch exits 1).
"""

import dataclasses
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter
from main.lang import collect as C
from main.lang.config import LangConfig
from main.lang.fusion import neck_to_row
from main.lang.neck_buffer import NeckBuffer, NeckLockstepWriter
from main.lang.network import LangPolicyEncoder
from main.lang.rollout import LangPolicyShim, LangRolloutController
from main.train.buffer import StepBuffer

from .fake_data import (FAKE_LEVEL_SHAPES, N_TOK_CAM, h_pattern, item_row, make_root, neck_ident, neck_row,
                        step_entry, template)
from .test_neck_buffer import check_aligned
from .test_network import body_inputs

REPO = Path(__file__).resolve().parents[2]
DIAG = {"per_link": {}, "d_arm_arm": 1.0}


# --------------------------------------------------------------------- controller: neck kept, rows aligned --
class StubEncoder:
    """The HoloBrainEncoder calls of the base _encode_obs; obs["imgs"] carries the frame identity (w, t, e)."""

    def __init__(self):
        self.visual = 0

    def encode_visual_tokens(self, imgs, depths, image_wh, projection_mat, out_extrinsic=None):
        self.visual += 1
        return (*frame_tokens(*imgs), None)

    def encode_joint_angles(self, joint_state, kin, embodiedment_mat=None):
        return torch.zeros(1, 14, 1, 256)


class StubExtractor:
    """What the base _collect_tick reads of the BodyTokenExtractor (UR5: 8 action columns per arm)."""

    spec = NS(link_names=[f"link{i}" for i in range(10)], n_joints=8)
    delta_theta_max = np.full(16, 0.02, np.float32)

    def raw_qpos(self):
        return np.zeros(16, np.float32)

    def extract(self):
        return {"body_tokens": torch.zeros(20, 16), "link_pos": torch.zeros(20, 3),
                "body_mask": torch.ones(20, dtype=torch.bool), "joint_index": torch.full((20,), -1),
                "Jlin": torch.zeros(20, 3, 16), "Jang": torch.zeros(20, 3, 16)}


def frame_tokens(w, t, e):
    """[1, 3, N_TOK_CAM, 256] tokens and positions; (w, t, e) in the first token's position (fake_data's rows)."""
    tok = torch.full((1, 3, N_TOK_CAM, 256), (w % 1000) / 1000.0)
    pos = torch.zeros(1, 3, N_TOK_CAM, 3)
    pos[0, 0, 0] = torch.tensor([w, t, e], dtype=torch.float32)
    return tok, pos


def live_neck(w, t, e):
    """A live neck dict (torch, F = 1, fake level shapes) carrying (w, t, e) like fake_data.neck_row."""
    return {k: torch.from_numpy(np.asarray(v, np.float32))[None] for k, v in neck_row(w, t, e).items()}


def fake_encode_obs_with_neck(encoder, obs):
    return frame_tokens(*obs["imgs"]), live_neck(*obs["imgs"])


def obs_of(w, t=0, e=0):
    return {"imgs": (w, t, e), "depths": None, "image_wh": None, "projection_mat": None, "T_ego2world": None,
            "joint_state": torch.zeros(1, 1, 14), "T_base2ego": None}


def bare_controller(**attrs):
    """A LangRolloutController without an env: the attributes _encode_obs / _collect_tick / _flush_pending read."""
    ro = LangRolloutController.__new__(LangRolloutController)
    ro.__dict__.update(dict(
        fusion=None, keep_neck=False, _in_capture=False, _neck_now=None, _neck_capture=None, kin=None,
        encoder=StubEncoder(), extractor=StubExtractor(), episode_id=1000, buf=None, _pending=None, trace={},
        cfg=NS(embodiment="ur5-wsg", task="gap_place", task_choices=("gap_place",)),
        _tick_action=np.zeros(16), _tick_source=0,
    ), **attrs)
    return ro


@pytest.fixture
def stub_encode(monkeypatch):
    monkeypatch.setattr("main.lang.fusion.encode_obs_with_neck", fake_encode_obs_with_neck)


def test_encode_obs_off_is_the_base_path(monkeypatch):
    called = []
    monkeypatch.setattr("main.lang.fusion.encode_obs_with_neck", lambda *a: called.append(a))
    ro = bare_controller()
    tokens, pos, state, body = ro._encode_obs(obs_of(5))
    assert ro.encoder.visual == 1 and not called and ro._neck_now is None
    assert tuple(pos[0, 0, 0].tolist()) == (5, 0, 0) and state.shape == (1, 14, 256) and "Jlin" in body


@pytest.mark.parametrize("which", ["keep_neck", "fusion"])
def test_encode_obs_keeps_the_neck(stub_encode, which):
    ro = bare_controller(**({"keep_neck": True} if which == "keep_neck" else {"fusion": object()}))
    ro._in_capture = True
    out = ro._encode_obs(obs_of(7))
    ref = bare_controller()._encode_obs(obs_of(7))           # the base path on the same stub encoder
    assert all(torch.equal(a, b) for a, b in zip(out[:3], ref[:3]))
    assert ro.encoder.visual == 0                            # one backbone pass (encode_obs_with_neck) only
    assert neck_ident(ro._neck_now["neck_maps"][0]) == (7, 0, 0) and ro._neck_capture is ro._neck_now
    ro._in_capture = False
    ro._encode_obs(obs_of(8))                                # the base class's cache-miss path
    assert neck_ident(ro._neck_now["neck_maps"][0]) == (8, 0, 0)
    assert neck_ident(ro._neck_capture["neck_maps"][0]) == (7, 0, 0)


def test_collect_tick_rows_carry_their_own_capture_neck(tmp_path, stub_encode):
    """The real base _collect_tick writes each entry one capture late and _flush_pending the last one: every row
    of the three rings is the same capture, cache misses between captures never reach a row."""
    root = tmp_path / "shard"
    buf = StepBuffer(root / "buffer", capacity=200, header={"fake": True})
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=N_TOK_CAM, k_max=8, r_max=16, g_max=4)
    nb = NeckBuffer.for_step_buffer(buf, level_shapes=FAKE_LEVEL_SHAPES)
    rec = template("gap_place")
    w = 0
    for e, (eid, L) in enumerate(((1000, 7), (1001, 5))):
        inner = LockstepItemWriter(buf, side)
        ro = bare_controller(keep_neck=True, episode_id=eid, buf=NeckLockstepWriter(inner, nb))
        H = h_pattern(len(rec["rules"]), L, e)
        for t in range(L):
            ro._in_capture = True                                # LangRolloutController._capture_tick
            inner.offer(item_row(rec, H[t], t, eid))             # the h_fn, before the encoding
            encoded = ro._encode_obs(obs_of(w, t, e))
            ro._collect_tick(obs_of(w, t, e), float(H[t].min()), DIAG, encoded)
            ro._in_capture = False
            assert neck_ident(ro._pending["_neck"]["neck_maps"]) == (w, t, e)   # this capture's entry
            if t % 2:                                            # the tick hook's fresh encoding
                ro._encode_obs(obs_of(1999, 99, 9))
            w += 1
        ro._flush_pending()
        ro.buf.end_episode()
    for b in (buf, side, nb):
        b.flush()
    assert check_aligned(root) == 12
    done = np.load(root / "buffer" / "done.npy", mmap_mode="r")[:12]
    assert np.flatnonzero(done).tolist() == [6, 11]


def test_collect_tick_needs_this_captures_neck(stub_encode):
    appended = []
    ro = bare_controller(keep_neck=True, buf=NS(append=appended.append))
    encoded = ro._encode_obs(obs_of(1))                      # outside a capture (a cache miss)
    with pytest.raises(RuntimeError, match="no neck"):
        ro._collect_tick(obs_of(1), 0.5, DIAG, encoded)
    assert ro._pending is None and not appended
    ro = bare_controller(buf=NS(append=appended.append))     # keep_neck off: today's entries
    for t in range(3):
        ro._collect_tick(obs_of(t, t), 0.5, DIAG, ro._encode_obs(obs_of(t, t)))
    assert len(appended) == 2 and not any("_neck" in a for a in appended) and "_neck" not in ro._pending


def test_controller_refuses_a_fusion_mismatch():
    """Checked before the env is built (no simulator needed)."""
    cfg = dataclasses.replace(LangConfig(), scene_fusion=True)
    with pytest.raises(ValueError, match="scene_fusion"):
        LangRolloutController(cfg, 0, policy_enc=LangPolicyEncoder(cfg, n_tok=12))       # the model, no stage
    with pytest.raises(ValueError, match="scene_fusion"):
        LangRolloutController(LangConfig(), 0, policy_enc=LangPolicyEncoder(LangConfig(), n_tok=12),
                              fusion=StubFusion())                                       # a stage, a V0 model
    with pytest.raises(ValueError, match="LangPolicyEncoder"):
        LangRolloutController(LangConfig(), 0, fusion=StubFusion())                      # nothing would call it


# ------------------------------------------------------------------------------------ shim with fusion --
NTOK = 4   # tokens per camera of the stub frames


class StubFusion:
    """FusionStage.fuse stand-in, a deterministic function of (frame, sentence): tokens and positions carry the
    frame (neck_maps[f, 0, 0, 0]) in channel 0 and the sentence id (words[f, 0, 0]) in channel 1; the fused words
    are the words plus the frame (zeros past n)."""

    def __init__(self):
        self.calls = []

    def fuse(self, neck, words, wmask):
        F = words.shape[0]
        frame, sent = neck["neck_maps"][:, 0, 0, 0].float(), words[:, 0, 0].float()
        self.calls.append((frame.tolist(), sent.tolist()))
        tok, pos = torch.zeros(F, 3 * NTOK, 256), torch.zeros(F, 3 * NTOK, 3)
        for x in (tok, pos):
            x[..., 0], x[..., 1] = frame[:, None], sent[:, None]
        return tok, pos, (words + frame[:, None, None]) * wmask[..., None]


def sentence(sid: float, n: int):
    w = torch.randn(1, 24, 256)
    w[0, 0, 0] = sid
    m = torch.zeros(1, 24, dtype=torch.bool)
    m[:, :n] = True
    return w, m


def make_fused(k=2, cf=False, fused_words=False):
    cfg = dataclasses.replace(LangConfig(), k_prev=k)
    torch.manual_seed(0)
    pe = LangPolicyEncoder(cfg, n_tok=3 * NTOK).eval()
    (wa, ma), (wb, mb) = sentence(1.0, 8), sentence(2.0, 5)
    ro = NS(_in_capture=False, fusion=StubFusion(), cfg=NS(fused_words=fused_words), text_words=(wa, ma),
            cf_text_words=(wb, mb) if cf else None, named_rule_id=None, cf_rule_id=None, timing={}, _neck_now=None)
    with torch.no_grad():
        ro.text_out = None if fused_words else pe.encode_text(wa, ma)
        ro.cf_text_out = (None if fused_words else pe.encode_text(wb, mb)) if cf else None
    return ro, LangPolicyShim(ro, pe, k), pe


def call_fused(ro, shim, i, capture):
    """One shim call on frame i (the handed tokens are junk) -> per encoder call: (frames [t, t-k, 0] of the
    tokens, their sentence ids, frames of the positions, the text_out it got)."""
    body, _ = body_inputs("ur5-wsg", 1, torch.Generator().manual_seed(0))
    seen = []
    orig = shim.pe.forward

    def spy(*a, **kw):
        seen.append((a[5].clone(), a[6].clone(), kw["text_out"]))
        return orig(*a, **kw)

    shim.pe.forward = spy
    ro._in_capture = capture
    ro._neck_now = {"neck_maps": torch.full((1, 3, 13, 256), float(i))}
    junk = torch.full((1, 3, NTOK, 256), -7.0), torch.full((1, 3, NTOK, 3), -7.0)
    shim(body["state_tokens"], body["body_tokens"], body["link_pos"], body["body_mask"], body["joint_index"], *junk)
    ro._in_capture = False
    shim.pe.forward = orig
    return [([int(st[0, f, 0, 0]) for f in range(3)], [int(st[0, f, 0, 1]) for f in range(3)],
             [int(sp[0, f, 0, 0]) for f in range(3)], text_out) for st, sp, text_out in seen]


def test_fused_histories_per_sentence():
    ro, shim, _ = make_fused(k=2, cf=True)
    want = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 2, 0], [5, 3, 0]]
    for i in range(6):
        named, cf = call_fused(ro, shim, i, capture=True)
        assert named[0] == named[2] == want[i] and named[1] == [1, 1, 1]     # tokens AND positions fused
        assert cf[0] == cf[2] == want[i] and cf[1] == [2, 2, 2]              # the counterfactual's own history
        assert named[3] is ro.text_out and cf[3] is ro.cf_text_out
        assert ro.fusion.calls[-1] == ([i, i], [1.0, 2.0])                   # one fuse call: the neck twice
    assert shim.n_pushed == 6 and len(shim.hist) == len(shim.hist_cf) == 3
    assert int(shim.frame0[0][0, 0, 1]) == 1 and int(shim.frame0_cf[0][0, 0, 1]) == 2
    ro, shim, _ = make_fused(cf=False)                                       # no counterfactual: one sentence
    assert len(call_fused(ro, shim, 0, capture=True)) == 1 and shim.enc_cf is None
    assert ro.fusion.calls == [([0.0], [1.0])]


def test_fused_cache_miss_is_not_pushed():
    ro, shim, _ = make_fused(k=2, cf=True)
    for i in range(3):
        call_fused(ro, shim, i, capture=True)
    named, cf = call_fused(ro, shim, 99, capture=False)
    assert named[0] == cf[0] == [99, 0, 0] and shim.n_pushed == 3
    named, cf = call_fused(ro, shim, 3, capture=True)
    assert named[0] == cf[0] == [3, 1, 0]


def test_fused_words_recompute_text_out_every_frame():
    ro, shim, pe = make_fused(cf=True, fused_words=True)
    got = []
    orig = pe.encode_text

    def spy(w, m):
        got.append((float(w[0, 0, 0]), float(w[0, 7, 0] - ro.text_words[0][0, 7, 0]), m))
        return orig(w, m)

    pe.encode_text = spy
    for i in range(3):
        got.clear()
        named, cf = call_fused(ro, shim, i, capture=True)
        # named then counterfactual, each on its own fused words of frame t (word 0 = sentence id + frame) and
        # the cached mask; the encoder gets exactly that text_out
        assert [a for a, _, _ in got] == [1.0 + i, 2.0 + i] and got[0][1] == pytest.approx(i)
        assert got[0][2] is ro.text_words[1] and got[1][2] is ro.cf_text_words[1]
        assert named[3] is ro.text_out and cf[3] is ro.cf_text_out
    ro, shim, pe = make_fused(cf=True, fused_words=False)                    # cached words: text_out per episode
    pe.encode_text = lambda *a: pytest.fail("encode_text per frame without fused_words")
    t_a, t_b = ro.text_out, ro.cf_text_out
    for i in range(2):
        named, cf = call_fused(ro, shim, i, capture=True)
        assert named[3] is t_a and cf[3] is t_b


def test_fused_push_needs_a_sentence_first():
    """C5: frame 0 must never be fused without the episode's sentence."""
    ro, shim, _ = make_fused()
    ro.text_words = None
    with pytest.raises(RuntimeError, match="set_instruction"):
        call_fused(ro, shim, 0, capture=True)
    assert shim.n_pushed == 0


class StubStore:
    """TextStore.lookup: one (words, mask) per sentence, word 0 = len(sentence)."""

    def lookup(self, sentences):
        return sentence(float(len(sentences[0])), 6)


@pytest.mark.parametrize("fused_words", [False, True])
def test_set_instruction_with_fusion(fused_words):
    ro, shim, _ = make_fused(fused_words=fused_words)
    ctl = NS(policy_enc=shim, fusion=ro.fusion, cfg=NS(fused_words=fused_words), text_store=StubStore(),
             text_out=None, cf_text_out=None, text_words=None, cf_text_words=None, named_rule_id=None)
    LangRolloutController.set_instruction(ctl, "abc", "abcdefg", rule_id="r")
    assert float(ctl.text_words[0][0, 0, 0]) == 3 and float(ctl.cf_text_words[0][0, 0, 0]) == 7
    assert (ctl.text_out is None) == fused_words and (ctl.cf_text_out is None) == fused_words
    LangRolloutController.set_instruction(ctl, "abcd")       # before any push: replaced, no counterfactual
    assert float(ctl.text_words[0][0, 0, 0]) == 4 and ctl.cf_text_words is None
    call_fused(ro, shim, 0, capture=True)
    with pytest.raises(RuntimeError, match="R4"):            # the history is fused with the first sentence
        LangRolloutController.set_instruction(ctl, "xyz")
    ctl.fusion = None                                        # without fusion: today's behaviour, no check
    LangRolloutController.set_instruction(ctl, "xyz")
    assert ctl.text_out is not None


def test_shim_without_fusion_uses_the_handed_tokens():
    ro, shim, _ = make_fused(cf=True)
    ro.fusion = None
    body, _ = body_inputs("ur5-wsg", 1, torch.Generator().manual_seed(0))
    seen = []
    orig = shim.pe.forward
    shim.pe.forward = lambda *a, **kw: seen.append(a[5].clone()) or orig(*a, **kw)
    ro._in_capture = True
    shim(body["state_tokens"], body["body_tokens"], body["link_pos"], body["body_mask"], body["joint_index"],
         torch.full((1, 3, NTOK, 256), 5.0), torch.zeros(1, 3, NTOK, 3))
    shim.pe.forward = orig
    assert len(seen) == 2 and all(float(s[0, 0, 0, 0]) == 5.0 for s in seen)   # both sentences, the same frames
    assert len(shim.hist_cf) == 0 and shim.n_pushed == 1


def test_shim_with_the_real_stage():
    """The seam to fusion.py (integration, CPU): the real FusionStage (checkpoint weights) on synthetic necks inside
    the shim. Its one fuse call (the neck twice, two sentence lengths) gives each sentence's history exactly what a
    separate fuse call per sentence gives, in the [1, 3, 1200, *] frames a 1200-token encoder reads; with fused_words
    text_out is the adapter on that call's fused words."""
    from main.lang import fusion as fz

    ckpt = Path(__import__("os").environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD"))
    if not ((ckpt / "pretrain" / "model.safetensors").exists() or (ckpt / "model.safetensors").exists()):
        pytest.skip(f"HoloBrain checkpoint not found at {ckpt}")
    stage = fz.FusionStage(str(ckpt), device="cpu")
    cfg = dataclasses.replace(LangConfig(), scene_fusion=True)
    torch.manual_seed(0)
    pe = LangPolicyEncoder(cfg, n_tok=1200).eval()
    (wa, ma), (wb, mb) = sentence(1.0, 7), sentence(2.0, 10)
    ro = NS(_in_capture=False, fusion=stage, cfg=cfg, text_words=(wa, ma), cf_text_words=(wb, mb),
            named_rule_id=None, cf_rule_id=None, timing={}, _neck_now=None, text_out=None, cf_text_out=None)
    shim = LangPolicyShim(ro, pe, 2)
    body, _ = body_inputs("ur5-wsg", 1, torch.Generator().manual_seed(0))
    seen = []
    orig = pe.forward
    pe.forward = lambda *a, **kw: seen.append((a[5], a[6], kw["text_out"])) or orig(*a, **kw)
    necks = [fz.synthetic_neck(1, seed=s) for s in (3, 4)]
    for neck in necks:
        ro._neck_now, ro._in_capture = neck, True
        shim(body["state_tokens"], body["body_tokens"], body["link_pos"], body["body_mask"], body["joint_index"],
             torch.zeros(1, 3, 400, 256), torch.zeros(1, 3, 400, 3))
        ro._in_capture = False
    pe.forward = orig
    assert ro.timing["fuse"] > 0 and shim.n_pushed == 2 and len(seen) == 4
    for (w, m), (st, sp, text_out) in zip(((wa, ma), (wb, mb)), seen[2:]):   # the second frame: [t, t-k, 0]
        assert st.shape == (1, 3, 1200, 256) and sp.shape == (1, 3, 1200, 3)
        ref = [stage.fuse(neck, w, m) for neck in necks]
        for f, src in enumerate((1, 0, 0)):                                   # t = frame 1; t-k and 0 = frame 0
            torch.testing.assert_close(st[0, f], ref[src][0][0], atol=2e-5, rtol=0)
            torch.testing.assert_close(sp[0, f], ref[src][1][0], atol=2e-5, rtol=0)
        with torch.no_grad():
            want = pe.encode_text(ref[1][2], m)
        for a, b in zip(text_out, want):   # encode_text's (r0, queries, words, mask) of frame t's fused words
            if a.is_floating_point():
                torch.testing.assert_close(a, b, atol=2e-5, rtol=0)
            else:
                assert torch.equal(a, b)


# ------------------------------------------------------------------- run_episodes with a fake controller --
class Killed(BaseException):
    """A hard kill: nothing after it runs."""


class FakeLock:
    """WriterLock stand-in (no heartbeat, no files); kill_at_check: Killed at that check() call."""

    class LockHeld(RuntimeError):
        pass

    class LockLost(BaseException):
        pass

    kill_at_check = None

    def __init__(self, root, **kw):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.n_check = self.n_fast = 0

    def check(self):
        self.n_check += 1
        if self.n_check == FakeLock.kill_at_check:
            raise Killed(f"killed at writer check {self.n_check}")

    def check_fast(self):
        self.n_fast += 1

    def release(self):
        pass


def overflow_neck(w, t, e):
    """A live neck whose neck_maps hold one value past the fp16 range: the real neck_to_row raises on it."""
    neck = live_neck(w, t, e)
    neck["neck_maps"][0, 1, 2, 5] = 1e6
    return neck


class FakeRollout:
    """LangRolloutController stand-in for run_episodes, following the base rollout's collection protocol: the h_fn
    offers one item row per capture, each capture's entry is appended one capture late (with its neck row when
    keep_neck), the last one with done. PLAN[seed]: ticks, fail_at (a tick error after the h_fn: the base run
    catches it and flushes the pending entry), bad_neck_at (the same through the REAL neck_to_row, which raises on
    an fp16 overflow before the entry is stashed, as LangRolloutController._collect_tick does), error_after (an
    error between captures, e.g. the planner, after that tick's capture: status exception, every row paired),
    kill_at (Killed after that tick's append), build_fails."""

    PLAN: dict = {}
    built: list = []
    w = 0

    def __init__(self, cfg, seed, mode="collect", **kw):
        self.plan = FakeRollout.PLAN.get(seed, {})
        FakeRollout.built.append((seed, kw))
        if self.plan.get("build_fails"):
            raise RuntimeError(f"fake build failure of seed {seed}")
        self.cfg, self.seed, self.mode = cfg, seed, mode
        self.buf, self.episode_id, self.keep_neck = kw["buf"], kw["episode_id"], kw.get("keep_neck", False)
        self.filter_active, self.video, self.h_fn, self.writer = kw.get("filter_active"), None, None, None
        runner = NS(host=cfg.item_host, active_rule_id=None, _primary_rule_id=lambda task: None)
        self.env = NS(obstacle=object(), _item_runner=runner, task=NS(), close=lambda: None)

    def run(self):
        L = self.plan.get("ticks", 6)
        rec = template("gap_place")
        H = h_pattern(len(rec["rules"]), L, 0)
        rng = np.random.default_rng(self.seed)
        e, pending, status, err = self.seed % 1000, None, "ok", None
        for t in range(L):
            self.h_fn.row_sink.offer(item_row(rec, H[t], t, self.seed))
            if t == self.plan.get("fail_at"):
                status = "exception"
                break
            entry = step_entry(rng, FakeRollout.w, t, e, self.seed, float(H[t].min()), False)
            if self.keep_neck:
                try:
                    entry["_neck"] = (neck_to_row(overflow_neck(FakeRollout.w, t, e))
                                      if t == self.plan.get("bad_neck_at") else neck_row(FakeRollout.w, t, e))
                except Exception as exc:   # the base run catches the tick error, then flushes the pending entry
                    status, err = "exception", f"{type(exc).__name__}: {exc}"
                    break
            FakeRollout.w += 1
            if pending is not None:
                self.buf.append(pending)
            pending = entry
            if t == self.plan.get("kill_at"):
                raise Killed(f"seed {self.seed} killed at tick {t}")
            if t == self.plan.get("error_after"):
                status, err = "exception", "RuntimeError: fake planner error"
                break
        if pending is not None:
            pending["done"] = np.bool_(True)
            self.buf.append(pending)
        return {"status": status, "success": status == "ok", "n_ticks": L, **({"exception": err} if err else {})}


def fake_wire(ro, sidecar):
    """wire_item_episode without a runner: the LockstepItemWriter as ro.buf and the h_fn's row sink."""
    lockstep = LockstepItemWriter(ro.buf, sidecar)
    ro.h_fn = NS(row_sink=lockstep, with_masks=True)
    ro.buf = lockstep


@pytest.fixture
def fake_env(monkeypatch):
    import main.item_envs.language as language
    import main.item_envs.margin_hook as margin_hook
    import main.lang.rollout as rollout
    import main.lang.scene_roles as roles
    import main.train.collect as train_collect

    FakeRollout.PLAN, FakeRollout.built, FakeRollout.w = {}, [], 0
    FakeLock.kill_at_check = None
    monkeypatch.setattr(rollout, "LangRolloutController", FakeRollout)
    monkeypatch.setattr(margin_hook, "wire_item_episode", fake_wire)
    monkeypatch.setattr(language, "episode_record",
                        lambda runner, seed: dict(template("gap_place"), episode_id=seed, seed=seed))
    monkeypatch.setattr(roles, "in_way", lambda env, rec: (set(), set()))
    for name in ("all_designations", "harm_reasons", "in_way_reasons"):
        monkeypatch.setattr(roles, name, lambda a, b: {})
    monkeypatch.setattr(roles, "mask_region_rows", lambda runner: {})
    monkeypatch.setattr(train_collect, "make_kinematics", lambda ckpt, emb: None)
    enc_mod = types.ModuleType("main.network.encoder")
    enc_mod.HoloBrainEncoder = lambda *a, **k: "stub-encoder"
    monkeypatch.setitem(sys.modules, "main.network.encoder", enc_mod)
    monkeypatch.setattr("main.lang.locks.WriterLock", FakeLock)
    # fake shards: tiny neck maps (13 cells per camera) for every NeckBuffer created without explicit shapes
    monkeypatch.setattr(NeckBuffer.__init__, "__kwdefaults__", {"level_shapes": FAKE_LEVEL_SHAPES, "n_cam": 3})
    monkeypatch.delenv("STAGE_DEADLINE", raising=False)
    return FakeRollout


def cfg_of(version=1, tasks=("gap_place",)):
    return dataclasses.replace(LangConfig(), task_choices=tasks, lang_version=version)


def test_run_episodes_v2_neck_in_lockstep(tmp_path, fake_env):
    cfg, root = cfg_of(2), tmp_path / "shard"
    buf, side, nb = C.make_pair(root, cfg, 200, neck=True)
    fake_env.PLAN = {5001: {"ticks": 9, "fail_at": 4}}
    lock = FakeLock(root)
    C.run_episodes(cfg, root, [("gap_place", "ur5-wsg", None)] * 3, 5000, buf, side, encoder="stub",
                   kins={"ur5-wsg": None}, with_masks=False, writer=lock, neck_buf=nb)
    lines = C.read_lines(root / "lang_episodes.jsonl")
    assert [d["status"] for d in lines] == ["ok", "exception", "ok"]
    assert [d["rows"] for d in lines] == [[0, 6], [6, 6], [6, 12]]   # the tick error rolled back all three rings
    assert all(d["lang_version"] == 2 for d in lines)
    assert [kw["keep_neck"] for _, kw in fake_env.built] == [True] * 3
    assert nb._guarded_by is lock and lock.n_fast >= 12             # the lock is checked before each neck row
    assert json.loads((root / "neck_buffer" / "header.json").read_text())["n"] == 12   # flushed
    assert check_aligned(root) == 12


def test_run_episodes_v1_unchanged(tmp_path, fake_env):
    cfg, root = cfg_of(1), tmp_path / "shard"
    buf, side = C.make_pair(root, cfg, 200)
    C.run_episodes(cfg, root, [("gap_place", "ur5-wsg", None)] * 2, 5000, buf, side, encoder="stub",
                   kins={"ur5-wsg": None}, with_masks=False)
    lines = C.read_lines(root / "lang_episodes.jsonl")
    assert [d["rows"] for d in lines] == [[0, 6], [6, 12]] and not any("lang_version" in d for d in lines)
    assert [kw["keep_neck"] for _, kw in fake_env.built] == [False, False]
    assert not (root / "neck_buffer").exists()
    with pytest.raises(ValueError, match="lang_version"):            # a version-2 config needs its neck buffer
        C.run_episodes(cfg_of(2), root, [], 5000, buf, side, encoder="stub")
    nb = NeckBuffer(tmp_path / "other" / "neck_buffer", capacity=200)
    with pytest.raises(ValueError, match="lang_version"):            # ... and a version-1 one has none
        C.run_episodes(cfg, root, [], 5000, buf, side, encoder="stub", neck_buf=nb)


def test_run_episodes_refuses_rings_out_of_lockstep(tmp_path, fake_env):
    cfg, root = cfg_of(2), tmp_path / "shard"
    buf, side, nb = C.make_pair(root, cfg, 200, neck=True)
    nb.append(dict(neck_row(0, 0, 0), episode_id=1, t_in_ep=0))      # a neck row the pair does not have
    with pytest.raises(RuntimeError, match="out of lockstep"):
        C.run_episodes(cfg, root, [("gap_place", "ur5-wsg", None)], 5000, buf, side, encoder="stub",
                       kins={"ur5-wsg": None}, neck_buf=nb)
    assert len(buf) == len(side) == 0


def test_replayed_slots_keep_their_seeds(tmp_path, fake_env):
    cfg, root = cfg_of(1, ("selective_avoid", "gap_place")), tmp_path / "shard"
    buf, side = C.make_pair(root, cfg, 400)
    fake_env.PLAN = {777: {"build_fails": True}, 5003: {"build_fails": True}}
    slots = [("gap_place", "ur5-wsg", None, 555), ("gap_place", "ur5-wsg", None, 777),
             ("selective_avoid", "ur5-wsg", "place_container_plate", 999), ("gap_place", "ur5-wsg", None)]
    summ = C.run_episodes(cfg, root, slots, 5000, buf, side, encoder="stub", kins={"ur5-wsg": None})
    # one build per replayed slot (777 fails: skipped, no new seed; every build still counts as an attempt); a
    # stream slot still draws offset + attempt and retries (5003 -> 5004)
    assert [s for s, _ in fake_env.built] == [555, 777, 999, 5003, 5004]
    lines = C.read_lines(root / "lang_episodes.jsonl")
    assert [d.get("episode_id") for d in lines] == [555, None, 999, 5004]
    assert lines[1]["skipped"] and "777" in lines[1]["errors"][0] and lines[2]["host"] == "place_container_plate"
    assert summ["attempt"] == 5 and [d["attempt_next"] for d in lines] == [1, 2, 3, 5]


# ------------------------------------------------------------------------------ resume with a neck buffer --
def write_lines(root, lines):
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))


def test_resume_point_rolls_back_the_neck_too(tmp_path):
    root = make_root(tmp_path / "s", episodes=3, capacity=400, neck=True)   # 3 x 40 rows, three rings
    write_lines(root, [{"episode_id": 1000, "rows": [0, 40], "attempt_next": 1, "slot": 0},
                       {"episode_id": 1001, "rows": [40, 80], "attempt_next": 2, "slot": 1}])
    buf, side = StepBuffer(root / "buffer"), ItemSidecarBuffer(root / "item_buffer")
    nb = C.open_neck(root, buf)
    assert C.resume_point(root, buf, side, nb) == (2, 3, 80)                # the third episode: next seed
    assert len(StepBuffer(root / "buffer")) == len(ItemSidecarBuffer(root / "item_buffer")) == 80
    assert len(NeckBuffer(root / "neck_buffer")) == 80                      # rolled back AND flushed
    assert check_aligned(root) == 80


def test_resume_point_refuses_a_misaligned_or_short_neck(tmp_path):
    root = make_root(tmp_path / "s", episodes=2, capacity=200, neck=True)
    write_lines(root, [{"episode_id": 1000, "rows": [0, 40], "attempt_next": 1, "slot": 0},
                       {"episode_id": 1001, "rows": [40, 80], "attempt_next": 2, "slot": 1}])
    buf, side = StepBuffer(root / "buffer"), ItemSidecarBuffer(root / "item_buffer")
    nb = C.open_neck(root, buf)
    assert C.resume_point(root, buf, side, nb) == (2, 2, 80)
    for field, row, bad in (("episode_id", 39, 1001), ("t_in_ep", 40, 3)):  # an end row of another capture
        good = int(nb.arrays[field][row])
        nb.arrays[field][row] = bad
        with pytest.raises(RuntimeError, match="do not hold that episode"):
            C.resume_point(root, buf, side, nb)
        nb.arrays[field][row] = good
    nb.discard_to(40)                                                       # a neck ring behind the pair
    with pytest.raises(RuntimeError, match=r"rings \(80, 80, 40\) shorter"):
        C.resume_point(root, buf, side, nb)


def test_open_neck(tmp_path, monkeypatch):
    root = make_root(tmp_path / "v2", episodes=1, capacity=100, neck=True)
    buf = StepBuffer(root / "buffer")
    nb = C.open_neck(root, buf)                                             # reopened r+ with its header's shapes
    assert nb.writable and nb.level_shapes == FAKE_LEVEL_SHAPES and (nb.n, nb.head) == (buf.n, buf.head)
    empty = StepBuffer(tmp_path / "e" / "buffer", capacity=50, header={"fake": True})
    monkeypatch.setattr(NeckBuffer.__init__, "__kwdefaults__", {"level_shapes": FAKE_LEVEL_SHAPES, "n_cam": 3})
    assert len(C.open_neck(tmp_path / "e", empty)) == 0                     # never created: paired fresh
    assert (tmp_path / "e" / "neck_buffer" / "header.json").exists()
    v1 = make_root(tmp_path / "v1", episodes=1, capacity=100)
    with pytest.raises(SystemExit, match="no neck_buffer"):
        C.open_neck(v1, StepBuffer(v1 / "buffer"))
    with pytest.raises(ValueError, match="capacity"):                      # a StepBuffer of another capacity
        C.open_neck(root, NS(capacity=200))
    del nb
    (root / "neck_buffer" / "neck_3d.npy").unlink()                         # a resume of a shard that lost an array
    with pytest.raises(FileNotFoundError, match="neck_3d.npy: missing in an existing neck buffer"):
        C.open_neck(root, buf)
    assert not (root / "neck_buffer" / "neck_3d.npy").exists()


# ------------------------------------------------------------------------------ shard meta, --replay --
V1_KEYS = ["kind", "embodiment", "shard", "n_shards", "g", "episode_offset", "episodes_per_task", "slots",
           "config_hash", "collect_hash", "bank_hash", "masks", "h_code", "h_code_v2"]


def with_(args, **kw):
    return NS(**{**vars(args), **kw})


def shard_args(**kw):
    d = dict(out="unused", embodiment="ur5-wsg", shard=0, n_shards=4, episodes_per_task=20,
             tasks="selective_avoid,gap_place", video_every=0, no_masks=False, time_budget=None, stream="grounding",
             g_base=0, lang_version=1, replay=None)
    d.update(kw)
    return NS(**d)


def test_plan_shard_v1_meta_unchanged():
    cfg, meta, slots, offset, masks = C.plan_shard(shard_args(), "validation")
    assert list(meta) == V1_KEYS and cfg.lang_version == 1 and offset == 91_000_000 and masks
    assert meta["slots"] == [list(s) for s in slots] and len(slots) == 10
    ref = REPO / "data/lang/val_grounding/shard_g0/shard.json"   # the command that wrote the report's validation
    if ref.exists():
        old = json.loads(ref.read_text())
        # the names check of 2026-10-12 changed the vocabulary (bank_hash, so collect_hash): a v1 shard of the old
        # vocabulary is not resumed or replayed with this code (the T1 model pool changed with it)
        same = [k for k in V1_KEYS if k not in ("config_hash", "h_code", "h_code_v2", "bank_hash", "collect_hash")]
        assert {k: meta[k] for k in same} == {k: old[k] for k in same}
        assert meta["bank_hash"] != old["bank_hash"]


def test_plan_shard_v2_and_meta_diff(tmp_path):
    cfg1, m1, s1, o1, _ = C.plan_shard(shard_args(), "validation")
    cfg2, m2, s2, o2, _ = C.plan_shard(shard_args(lang_version=2), "validation")
    assert cfg2.lang_version == 2 and m2["lang_version"] == 2 and (s2, o2) == (s1, o1)
    assert m2["collect_hash"] != m1["collect_hash"]
    rest = {"lang_version", "collect_hash", "config_hash"}
    assert {k: v for k, v in m2.items() if k not in rest} == {k: v for k, v in m1.items() if k not in rest}
    cfg1.save(tmp_path / "config.json")
    assert C.shard_meta_diff(m1, m1, tmp_path) == []
    assert C.shard_meta_diff(m1, m2, tmp_path) == ["collect_hash", "lang_version"]   # never resumed as the other
    assert C.shard_meta_diff(m2, m1, tmp_path) == ["collect_hash", "lang_version"]
    moved = {**m2, "replay_of": "/elsewhere"}
    assert C.shard_meta_diff(moved, {**m2, "replay_of": "/here"}, tmp_path) == []
    with pytest.raises(SystemExit, match="--shard"):
        C.plan_shard(shard_args(shard=None), "validation")
    assert C.plan_shard(shard_args(episodes_per_task=None, n_shards=5, stream="default"), "validation")[1][
        "episodes_per_task"] == 5                                                     # the old CLI default


SRC_LINES = [
    {"slot": 0, "episode_id": 92000000, "task": "selective_avoid", "host": "place_container_plate",
     "embodiment": "ur5-wsg", "status": "exception", "rows": [0, 1], "attempt_next": 1},
    {"slot": 1, "episode_id": 92000001, "task": "gap_place", "host": None, "embodiment": "ur5-wsg",
     "status": "ok", "rows": [1, 300], "attempt_next": 2},
    {"skipped": True, "slot": 2, "task": "selective_avoid", "host": "stack_blocks_two", "embodiment": "ur5-wsg",
     "errors": ["seed 92000002: no rules"], "attempt_next": 7},
    {"slot": 3, "episode_id": 92000007, "task": "push_zone", "host": None, "embodiment": "ur5-wsg",
     "status": "ok", "rows": [300, 500], "attempt_next": 8},
    {"slot": 4, "episode_id": 92000008, "task": "selective_avoid", "host": "place_bread_basket",
     "embodiment": "ur5-wsg", "status": "ok", "rows": [500, 700], "attempt_next": 9},
]
REPLAY = [("selective_avoid", "ur5-wsg", "place_container_plate", 92000000), ("gap_place", "ur5-wsg", None, 92000001),
          ("selective_avoid", "ur5-wsg", "place_bread_basket", 92000008)]


def v1_source(root: Path, kind="validation", masks=True, complete=True, cfg=None) -> Path:
    """A finished version-1 shard's records (no buffers: a replay reads only shard.json, config.json and lines)."""
    cfg = cfg or cfg_of(1, ("selective_avoid", "gap_place", "push_zone"))
    root.mkdir(parents=True)
    meta = {"kind": kind, "embodiment": "ur5-wsg", "shard": 1, "n_shards": 4, "g": 1, "episode_offset": 92_000_000,
            "episodes_per_task": 20, "slots": [[d["task"], "ur5-wsg", d["host"]] for d in SRC_LINES],
            "config_hash": cfg.hash(), "collect_hash": C.collect_hash(cfg), "bank_hash": cfg.bank_hash,
            "masks": masks}
    cfg.save(root / "config.json")
    (root / "shard.json").write_text(json.dumps(meta, indent=1))
    write_lines(root, SRC_LINES)
    (root / "summary.json").write_text(json.dumps({"complete": complete}))
    return root


def snapshot(root: Path) -> dict:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(root.iterdir())}


def test_replay_plan_reads_the_recorded_episodes(tmp_path):
    src = v1_source(tmp_path / "v1")
    before = snapshot(src)
    rp = C.replay_plan(src, "validation", "ur5-wsg", ("selective_avoid", "gap_place"))
    assert rp["slots"] == REPLAY and rp["src_slots"] == [0, 1, 4]   # every status; the skipped slot left out
    assert [s[3] for s in C.replay_plan(src, "validation", "ur5-wsg")["slots"]] == [92000000, 92000001, 92000007,
                                                                                     92000008]
    with pytest.raises(SystemExit, match="validation shard"):
        C.replay_plan(src, "warmup", "ur5-wsg")
    with pytest.raises(SystemExit, match="ur5-wsg"):
        C.replay_plan(src, "validation", "franka-panda")
    with pytest.raises(SystemExit, match="no recorded"):
        C.replay_plan(src, "validation", "ur5-wsg", ("tray_transport",))
    assert snapshot(src) == before
    with open(src / "lang_episodes.jsonl", "a") as f:
        f.write('{"slot": 5, "episode')                            # a truncated line: refused, never repaired
    before = snapshot(src)
    with pytest.raises(SystemExit, match="not JSON"):
        C.replay_plan(src, "validation", "ur5-wsg")
    assert snapshot(src) == before
    with pytest.raises(SystemExit, match="not a finished shard"):
        C.replay_plan(v1_source(tmp_path / "running", complete=False), "validation", "ur5-wsg")


def test_plan_shard_replay(tmp_path):
    src = v1_source(tmp_path / "v1")
    args = shard_args(shard=None, n_shards=None, episodes_per_task=None, stream="default", replay=str(src),
                      lang_version=2)
    cfg, meta, slots, offset, masks = C.plan_shard(args, "validation")
    assert slots == REPLAY and offset == 92_000_000 and masks is True
    assert (meta["shard"], meta["n_shards"], meta["g"], meta["episodes_per_task"]) == (1, 4, 1, 20)
    assert meta["slots"] == [list(s) for s in REPLAY] and meta["replay_slots"] == [0, 1, 4]
    assert meta["replay_of"] == str(src.resolve()) and meta["lang_version"] == 2 and cfg.lang_version == 2
    assert cfg.task_choices == ("selective_avoid", "gap_place", "push_zone")   # the source's (StepBuffer task_id)
    assert C.plan_shard(with_(args, shard=1, n_shards=4, episodes_per_task=20), "validation")[1] == meta
    for bad, match in ((dict(shard=0), "--shard"), (dict(episodes_per_task=30), "--episodes-per-task"),
                       (dict(no_masks=True), "--no-masks")):
        with pytest.raises(SystemExit, match=match):
            C.plan_shard(with_(args, **bad), "validation")
    drift = v1_source(tmp_path / "drift", cfg=dataclasses.replace(cfg_of(1), max_steps_per_episode=10))
    with pytest.raises(SystemExit, match="max_steps_per_episode"):
        C.plan_shard(with_(args, replay=str(drift)), "validation")
    warm = v1_source(tmp_path / "warm", kind="warmup", masks=False)
    assert C.plan_shard(with_(args, replay=str(warm)), "warmup")[4] is False   # its masks setting


REPLAY_ARGV = ["validation", "--lang-version", "2", "--embodiment", "ur5-wsg", "--tasks", "selective_avoid,gap_place"]


def test_replay_end_to_end_with_resumes(tmp_path, fake_env, capsys):
    src = v1_source(tmp_path / "v1")
    out = tmp_path / "v2" / "shard_g1"
    argv = REPLAY_ARGV + ["--replay", str(src), "--out", str(out)]
    # 1: episode 2 flushes its rows, then the process dies before its line (writer check 6: 3 per episode)
    FakeLock.kill_at_check = 6
    with pytest.raises(Killed):
        C.main(argv)
    assert [d["episode_id"] for d in C.read_lines(out / "lang_episodes.jsonl")] == [92000000]
    assert json.loads((out / "neck_buffer" / "header.json").read_text())["n"] == 12
    # 2: the resume rolls the three rings back to 6 rows; episode 2 again, killed mid-run (rows past the headers)
    FakeLock.kill_at_check, fake_env.PLAN = None, {92000001: {"kill_at": 3}}
    with pytest.raises(Killed):
        C.main(argv)
    assert "discarding rows 6..12" in capsys.readouterr().out
    assert json.loads((out / "neck_buffer" / "header.json").read_text())["n"] == 6
    # 3: done; every slot holds its recorded episode, the killed one collected again with ITS seed
    fake_env.PLAN = {}
    assert C.main(argv) == 0
    assert [s for s, _ in fake_env.built] == [92000000, 92000001, 92000001, 92000001, 92000008]
    lines = C.read_lines(out / "lang_episodes.jsonl")
    assert [(d["task"], d["embodiment"], d["host"], d["episode_id"]) for d in lines] == REPLAY
    assert all(d["lang_version"] == 2 and d["masks"] is True for d in lines)
    meta = json.loads((out / "shard.json").read_text())
    assert meta["lang_version"] == 2 and meta["replay_of"] == str(src.resolve())
    summ = json.loads((out / "summary.json").read_text())
    assert summ["complete"] and summ["replay_mismatch"] == []
    assert check_aligned(out) == 18
    assert json.loads((out / "buffer" / "header.json").read_text())["lang_version"] == 2   # the data version
    assert C.main(argv) == 0                                                # complete: nothing more to do


def test_replay_mismatch_exits_1(tmp_path, fake_env):
    src = v1_source(tmp_path / "v1")
    argv = REPLAY_ARGV + ["--replay", str(src), "--out", str(tmp_path / "v2")]
    fake_env.PLAN = {92000008: {"build_fails": True}}                       # its one build fails: no new seed
    assert C.main(argv) == 1
    summ = json.loads((tmp_path / "v2" / "summary.json").read_text())
    assert summ["complete"] and len(summ["replay_mismatch"]) == 1 and "slot 2: skipped" in summ["replay_mismatch"][0]
    assert C.main(argv) == 1                                                # still reported on a rerun
    assert [s for s, _ in fake_env.built] == [92000000, 92000001, 92000008]
    before = snapshot(src)
    with pytest.raises(SystemExit, match="replayed shard itself"):          # never written into the source
        C.main(REPLAY_ARGV + ["--replay", str(src), "--out", str(src)])
    assert snapshot(src) == before


def check_shards_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_shards_t", REPO / "scripts" / "lang" / "check_shards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay_line(i, **kw):
    task, emb, host, eid = REPLAY[i]
    return dict(task=task, embodiment=emb, host=host, episode_id=eid, **kw)


def test_replay_mismatch_flags_zero_rows_not_status(tmp_path):
    """V2-1: a replayed episode rolled back to zero rows (a tick error between the h_fn's sidecar row and the
    StepBuffer append) lost its layout, in replay_mismatch and in check_shards.check_replay; one whose status differs
    from the source's but that kept its rows is not a difference (cuRobo redraws the motion)."""
    cs = check_shards_mod()
    root = tmp_path / "replay"
    root.mkdir()
    (root / "shard.json").write_text(json.dumps({"replay_of": "/src", "slots": [list(s) for s in REPLAY]}))
    ok = [replay_line(0, status="ok", rows=[0, 6]), replay_line(1, status="ok", rows=[6, 12]),
          replay_line(2, status="ok", rows=[12, 18])]
    # slot 0 raised in the source (SRC_LINES) and is ok here; slot 1 was ok there and raises here after 4 rows
    mixed = [ok[0], replay_line(1, status="exception", exception="RuntimeError: planner", rows=[6, 10]),
             replay_line(2, status="ok", rows=[10, 16])]
    assert SRC_LINES[0]["status"] == "exception" and SRC_LINES[1]["status"] == "ok"
    for lines in (ok, mixed):
        assert C.replay_mismatch(lines, REPLAY) == [] and cs.check_replay(root, lines) == []
    tick_error = "lockstep after a tick error (lockstep: 1 queued item row(s) never appended)"
    rolled = [ok[0], replay_line(1, status="exception", exception=tick_error, rows=[6, 6]),
              replay_line(2, status="ok", rows=[6, 12])]
    got = C.replay_mismatch(rolled, REPLAY)
    assert len(got) == 1 and got[0].startswith(f"slot 1: no rows for {REPLAY[1]}") and tick_error in got[0]
    bad = cs.check_replay(root, rolled)
    assert len(bad) == 1 and bad[0].startswith("replay slot 1: episode gap_place 92000001 has no rows")
    no_rows = [ok[0], {k: v for k, v in ok[1].items() if k != "rows"}, ok[2]]   # a line without rows: no rows
    assert [m[:16] for m in C.replay_mismatch(no_rows, REPLAY)] == ["slot 1: no rows "]
    # a zero-row line that is ALSO another episode: both differences listed
    got = C.replay_mismatch([ok[0], dict(rolled[1], episode_id=5), ok[2]], REPLAY)
    assert len(got) == 2 and got[0] == f"slot 1: {('gap_place', 'ur5-wsg', None, 5)} instead of {REPLAY[1]}"
    assert got[1].startswith("slot 1: no rows")


@pytest.mark.parametrize("plan", [{"bad_neck_at": 2}, {"fail_at": 3}])
def test_replay_zero_row_episode_exits_1(tmp_path, fake_env, plan):
    """V2-1 end to end: a replayed slot whose capture raises in neck_to_row (an fp16 overflow) or after its h_fn
    row is rolled back to zero rows; the command writes a non-empty replay_mismatch and exits 1, also on a rerun,
    and check_shards fails the shard."""
    with pytest.raises(ValueError, match="overflows fp16"):            # the real conversion refuses the fake's neck
        neck_to_row(overflow_neck(0, 0, 0))
    src = v1_source(tmp_path / "v1")
    out = tmp_path / "v2"
    argv = REPLAY_ARGV + ["--replay", str(src), "--out", str(out)]
    fake_env.PLAN = {92000001: plan}
    assert C.main(argv) == 1
    lines = C.read_lines(out / "lang_episodes.jsonl")
    assert [d["status"] for d in lines] == ["ok", "exception", "ok"]
    assert [d["rows"] for d in lines] == [[0, 6], [6, 6], [6, 12]]           # rolled back, the next one follows
    assert "lockstep after a tick error" in lines[1]["exception"]
    summ = json.loads((out / "summary.json").read_text())
    assert summ["complete"] and len(summ["replay_mismatch"]) == 1
    assert summ["replay_mismatch"][0].startswith("slot 1: no rows for ('gap_place', 'ur5-wsg', None, 92000001)")
    assert C.main(argv) == 1                                                # still reported on a rerun
    assert [s for s, _ in fake_env.built] == [92000000, 92000001, 92000008]
    assert check_aligned(out) == 12
    bad = check_shards_mod().check(out)
    assert any(b.startswith("replay slot 1: episode gap_place 92000001 has no rows") for b in bad)
    assert any(b.startswith("summary.json replay_mismatch (1): slot 1: no rows") for b in bad)


def test_replay_exception_episode_with_rows_is_not_a_mismatch(tmp_path, fake_env):
    """An error between captures (the planner) after 3 ticks: status exception, its rows kept and paired. The
    replay differs from the source in status only (slot 0 raised there and is ok here, slot 1 the other way round),
    which is no mismatch: exit 0 and check_shards passes."""
    src = v1_source(tmp_path / "v1")
    out = tmp_path / "v2"
    fake_env.PLAN = {92000001: {"error_after": 2}}
    assert C.main(REPLAY_ARGV + ["--replay", str(src), "--out", str(out)]) == 0
    lines = C.read_lines(out / "lang_episodes.jsonl")
    assert [d["status"] for d in lines] == ["ok", "exception", "ok"] and lines[1]["exception"].startswith("Runtime")
    assert [d["rows"] for d in lines] == [[0, 6], [6, 9], [9, 15]]
    assert json.loads((out / "summary.json").read_text())["replay_mismatch"] == []
    assert check_aligned(out) == 15 and check_shards_mod().check(out) == []


@pytest.mark.parametrize("version", [1, 2])
def test_shard_without_replay(tmp_path, fake_env, version):
    out = tmp_path / "shard"
    argv = ["validation", "--embodiment", "ur5-wsg", "--shard", "0", "--n-shards", "4", "--episodes-per-task", "4",
            "--tasks", "selective_avoid,gap_place", "--stream", "grounding", "--out", str(out)]
    assert C.main(argv + (["--lang-version", "2"] if version == 2 else [])) == 0
    meta = json.loads((out / "shard.json").read_text())
    lines = C.read_lines(out / "lang_episodes.jsonl")
    assert [d["episode_id"] for d in lines] == [91_000_000, 91_000_001]
    assert [kw["keep_neck"] for _, kw in fake_env.built] == [version == 2] * 2
    summ = json.loads((out / "summary.json").read_text())
    assert "replay_mismatch" not in summ
    if version == 1:   # exactly the shard before V5
        assert list(meta) == V1_KEYS and not (out / "neck_buffer").exists()
        assert not any("lang_version" in d for d in lines)
    else:
        assert list(meta) == V1_KEYS + ["lang_version"] and check_aligned(out) == 12
        assert all(d["lang_version"] == 2 for d in lines)


def test_rounds_refuse_scene_fusion(tmp_path):
    from main.lang.rounds import run_rounds

    cfg = dataclasses.replace(LangConfig(), scene_fusion=True)
    with pytest.raises(ValueError, match="scene_fusion"):
        run_rounds(cfg, NS(run=str(tmp_path / "run")), [], [], [], None, None, trainer=object(), collect_fn=object())
    assert not (tmp_path / "run").exists()
