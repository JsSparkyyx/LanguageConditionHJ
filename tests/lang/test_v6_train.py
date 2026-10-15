"""V6 training (qwen_impl.md 7, 8) on CPU: fake Qwen shards (kv rows carrying their frame's identity) and a stub
HoloBrain-Qwen text pass whose words carry (the prefix row's identity, the sentence).

- the sampler gathers the prefix rows of frames t and t+1, checked against the item sidecar, and refuses sources
  of another front end;
- qwen_words: one pass per unique (row, sentence); frame t's words for the live encode, t+1's for the target;
  "blank" reads the black-image prefix;
- grad_step, the probe and checkpoints under Qwen; load_weights refuses a GD checkpoint; another checkpoint's kv
  rows are refused;
- rounds: the ring gains kv_buffer/ (the trainer's checkpoint id), collection gets the shared front end, a resume
  rolls the kv ring back with the pair.
"""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.item_envs.item_buffer import ItemSidecarBuffer
from main.lang.config import LangConfig
from main.lang.kv_buffer import KvBuffer
from main.lang.rounds import run_rounds
from main.lang.train import EXIT_RESUME, LangTrainer, probe_kv

from .fake_data import make_root
from .test_rounds import FakeCollect
from .test_train import StubText
from .test_v6_storage import KV_DIMS, kv_row

torch.set_num_threads(4)


def code(s: str) -> float:
    import hashlib

    return float(int(hashlib.md5(s.encode()).hexdigest()[:4], 16) % 97) / 10.0


class StubQwen:
    """encode_text: word 0 = (the prefix row's K value = its frame's write index, the sentence's code)."""

    def __init__(self, ckpt_id="c0"):
        self.ckpt_id = ckpt_id
        self.calls = []

    def blank_kv(self):
        K = torch.full((1, 2, 5, 4), -5.0, dtype=torch.bfloat16)
        return K, K

    def encode_text(self, kv, sentences, kv_index=None):
        K, _ = kv
        N = len(sentences)
        idx = kv_index if kv_index is not None else (torch.zeros(N, dtype=torch.long) if K.shape[0] == 1
                                                     else torch.arange(N))
        self.calls.append(N)
        w = torch.zeros(N, 24, 256)
        m = torch.zeros(N, 24, dtype=torch.bool)
        for i, s in enumerate(sentences):
            w[i, 0, 0] = float(K[int(idx[i]), 0, 0, 0])
            w[i, 0, 1] = code(s)
            w[i, 1:4] = 0.1
            m[i, :4] = True
        return w, m


def make_qwen_root(path, episodes=4, capacity=400, ckpt="c0"):
    root = make_root(path, episodes=episodes, capacity=capacity)
    hp = root / "buffer" / "header.json"
    h = json.loads(hp.read_text())
    hp.write_text(json.dumps({**h, "encoder": "holobrain_qwen"}))
    side = ItemSidecarBuffer(root / "item_buffer")
    pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")
    kb = KvBuffer(root / "kv_buffer", capacity=capacity, ckpt_id=ckpt, attn="sdpa", **KV_DIMS)
    for i in range(side.n):
        kb.append(dict(kv_row(int(pos[i, 0, 0])), episode_id=int(side.arrays["episode_id"][i]),
                       t_in_ep=int(side.arrays["t_in_ep"][i])))
    kb.flush()
    return root


def qcfg(**kw):
    return dataclasses.replace(LangConfig(), batch_size=8, encoder="holobrain_qwen", text_backend="holobrain_qwen",
                               **kw)


def qtrainer(run, root, val=(), qwen=None, **kw):
    return LangTrainer(qcfg(**kw), run, [root], list(val), (), use_wandb=False, prefetch=False, device="cpu", seed=0,
                       probe_n=16, qwen=qwen or StubQwen())


@pytest.fixture
def qroot(tmp_path):
    return make_qwen_root(tmp_path / "q")


def test_sampler_gathers_frames_t_and_t1(tmp_path, qroot):
    tr = qtrainer(tmp_path / "run", qroot)
    assert tr.text is None and tr.policy_enc.scene_set.n_tok == 12
    b = tr.sampler.sample(8)
    kv, kidx = b["kv"], np.asarray(b["kv_idx"])
    assert kidx.shape == (8, 2) and kv["k"].shape[1:] == (2, 5, 4)
    ident = torch.from_numpy(kv["k"][:, 0, 0, 0].copy()).view(torch.bfloat16).float().numpy()
    assert np.array_equal(ident[kidx[:, 0]], b["scene_pos"][:, 0, 0, 0])            # frame t's row
    assert np.array_equal(ident[kidx[:, 1]], b["scene_pos_next"][:, 0, 0, 0])       # frame t+1's row
    assert np.array_equal(kv["t_in_ep"][kidx[:, 0]], b["t_in_ep"])
    assert len(set(map(tuple, kidx.tolist()))) <= 8 and kv["k"].shape[0] == len(np.unique(kidx))


def test_qwen_words_route_frames_and_sentences(tmp_path, qroot):
    tr = qtrainer(tmp_path / "run", qroot)
    rng = np.random.default_rng(0)
    b = tr.sampler.sample(8, rng, picks=tr.sampler.plan(8, rng, pair_frac=1.0))     # 4 pairs
    q = tr.qwen_words(b)
    want_t, want_n = b["scene_pos"][:, 0, 0, 0], b["scene_pos_next"][:, 0, 0, 0]
    assert np.array_equal(q["w_qwen"][:, 0, 0].numpy(), want_t)
    assert np.array_equal(q["w_qwen_next"][:, 0, 0].numpy(), want_n)
    assert np.allclose(q["w_qwen"][:, 0, 1].numpy(), [code(s) for s in b["sentence"]])
    keys = {(int(r), s) for r, s in zip(b["row"], b["sentence"])} | {(int(r), s) for r, s in zip(b["next_row"],
                                                                                                    b["sentence"])}
    assert tr.last_qwen_passes == len(keys) and q["_qwen"] == ("", "_next")
    live = tr.qwen_words(b, target=False)
    assert "w_qwen_next" not in live and live["_qwen"] == ("",)
    with pytest.raises(RuntimeError, match="no words"):
        tr.encode(live, live["w_qwen"], live["wm_qwen"], "_next", target=True)
    bl = qtrainer(tmp_path / "blank", qroot, qwen_words_prefix="blank").qwen_words(b)
    assert (bl["w_qwen"][:, 0, 0] == -5).all() and (bl["w_qwen_next"][:, 0, 0] == -5).all()


def test_grad_step_reads_each_frames_words(tmp_path, qroot):
    tr = qtrainer(tmp_path / "run", qroot)
    seen = {}

    def hook(name):
        def f(_m, args, kwargs):
            seen[name] = (kwargs["words"][:, 0, 0].detach().clone(), args[6][:, 0, 0, 0].detach().clone())
        return f

    h1 = tr.policy_enc.register_forward_pre_hook(hook("live"), with_kwargs=True)
    h2 = tr.policy_enc_targ.register_forward_pre_hook(hook("target"), with_kwargs=True)
    out = tr.grad_step(log=True)
    h1.remove(), h2.remove()
    for name in ("live", "target"):   # the words of frame t (live) and of t+1 (target): the frame the scene shows
        w, frame_id = seen[name]
        assert torch.equal(w, frame_id), name
    assert np.isfinite(out["loss/q"]) and "perf/qwen_text_ms" in out and out["perf/qwen_passes"] > 0
    assert "data/text_miss_frac" not in out


def test_probe_and_checkpoints(tmp_path, qroot):
    val = make_qwen_root(tmp_path / "val", episodes=3)
    tr = qtrainer(tmp_path / "run", qroot, val=[val])
    tr.grad_step()
    out = tr.run_probe()
    assert out and "kv" in tr.prober.batch and (np.asarray(tr.prober.batch["kv_idx"])[:, 1] == -1).all()
    p = tr.save_ckpt()
    ck = torch.load(p, weights_only=False)
    assert ck["encoder"] == "holobrain_qwen" and ck["qwen_ckpt_id"] == "c0"
    tr.save_state()
    tr2 = qtrainer(tmp_path / "run2", qroot)
    tr2.load_weights(p)
    groot = make_root(tmp_path / "gd", episodes=2, capacity=200)
    g = LangTrainer(dataclasses.replace(LangConfig(), batch_size=8), tmp_path / "grun", [groot], (), (),
                    use_wandb=False, prefetch=False, device="cpu", text_encoder=StubText(), seed=0)
    gp = g.save_ckpt()
    ck = torch.load(gp, weights_only=False)
    ck["text_backend"] = "holobrain_qwen"                       # past the backend check: the encoder check refuses
    torch.save(ck, gp)
    with pytest.raises(ValueError, match="front end"):
        tr2.load_weights(gp)


def test_probe_kv_keeps_frame_t():
    batch = {"kv": {"k": np.arange(5)[:, None] * np.ones((5, 3), np.int16), "episode_id": np.arange(5)},
             "kv_idx": np.array([[3, 4], [1, 3], [3, 0]])}
    out = probe_kv(batch)
    assert out["kv_idx"].tolist() == [[1, -1], [0, -1], [1, -1]]
    assert out["kv"]["episode_id"].tolist() == [1, 3] and out["kv"][2] is out["kv"]


def test_sources_of_another_front_end_or_checkpoint_are_refused(tmp_path, qroot):
    groot = make_root(tmp_path / "gd", episodes=2, capacity=200)
    with pytest.raises(ValueError, match="holobrain_gd front end"):
        qtrainer(tmp_path / "a", groot)
    with pytest.raises(ValueError, match="holobrain_qwen front end"):
        LangTrainer(dataclasses.replace(LangConfig(), batch_size=8), tmp_path / "b", [qroot], (), (),
                    use_wandb=False, prefetch=False, device="cpu", text_encoder=StubText())
    with pytest.raises(ValueError, match="never mix checkpoints"):
        qtrainer(tmp_path / "c", qroot, qwen=StubQwen("other"))
    kb = KvBuffer.reopen(qroot / "kv_buffer")
    kb.arrays["t_in_ep"][:] = 99
    kb.flush()
    tr = qtrainer(tmp_path / "d", qroot)
    with pytest.raises(RuntimeError, match="kv_buffer row"):
        tr.sampler.sample(8)


# ---------------------------------------------------------------------------------------------------- rounds --
def _qrounds(run, root, collect, resume, qwen):
    cfg = qcfg(max_steps_per_episode=10)
    tr = LangTrainer(cfg, run, [root], (), (), use_wandb=False, prefetch=False, device="cpu", seed=0, qwen=qwen)
    collect.tr = tr
    args = SimpleNamespace(run=str(run), resume=resume, collect_rounds=2, episodes_per_round=1,
                           grad_steps_per_round=2, probe_every_rounds=0, video_every=0, no_wandb=True, seed=0,
                           time_budget=None)
    return run_rounds(cfg, args, [root], [], [], None, None, trainer=tr, collect_fn=collect), tr


class KwCollect(FakeCollect):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.kw = []

    def __call__(self, *a, **kw):
        self.kw.append({k: kw.get(k) for k in ("qwen", "kv_buf", "text_store")})
        return super().__call__(*a, **kw)


def test_rounds_kv_ring(tmp_path, qroot):
    qwen = StubQwen()
    col = KwCollect(skip_at=None)
    rc, tr = _qrounds(tmp_path / "run", qroot, col, False, qwen)
    assert rc == 0
    ring = tmp_path / "run" / "ring"
    kb = KvBuffer(ring / "kv_buffer")
    assert kb.header["ckpt_id"] == "c0" and kb.capacity == json.loads((ring / "buffer" / "header.json").read_text())["capacity"]
    assert col.kw and all(d["qwen"] is qwen and d["kv_buf"] is not None and d["text_store"] is None for d in col.kw)
    # a resume rolls an extra kv row back with the pair (the ring itself holds no rows here)
    kb = KvBuffer.reopen(ring / "kv_buffer")
    from main.lang.kv_buffer import kv_to_row

    z = torch.zeros(2, 325, 128, dtype=torch.bfloat16)               # the ring's rows have the real prefix shape
    kb.append(dict(kv_to_row(z, z), episode_id=1, t_in_ep=0))
    kb.flush()
    rc, _ = _qrounds(tmp_path / "run", qroot, KwCollect(skip_at=None), True, qwen)
    assert rc == 0 and len(KvBuffer(ring / "kv_buffer")) == 0
    other = make_qwen_root(tmp_path / "q_other", ckpt="other")   # data of another checkpoint, the ring's is c0
    with pytest.raises(SystemExit, match="ckpt_id"):
        _qrounds(tmp_path / "run", other, KwCollect(skip_at=None), True, StubQwen("other"))
