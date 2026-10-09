"""V5 training and evaluation (fusion_impl.md 4.4, 4.6) on CPU: fake version-2 shards and a stub fusion stage.

The stub's output is a deterministic function of (neck row, sentence): fake neck rows carry their identity (write
index, t_in_ep, episode index) in neck_maps (fake_data.neck_row), the same identity the StepBuffer row keeps in
scene_pos[0], and StubText words are a function of the sentence. So a test reads, from any fused token or position,
which frame it came from and which sentence it was fused with.
"""

import dataclasses
import functools
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.lang import train as train_mod
from main.lang.config import LangConfig
from main.lang.eval_offline import Evaluator, data_info, model_info, model_line, summarize
from main.lang.train import LangTrainer, SharedNeck, fusion_flags, neck_table

from .fake_data import make_root
from .test_train import StubText

torch.set_num_threads(4)


class StubFusion:
    """fuse(neck, words, wmask), FusionStage's interface. Per frame f with identity ident = neck_maps[f, 0, 0, :3]
    and sentence code = words[f, 0, :3]: tokens [ident, code, 0 ...] on every token; positions token 0 = ident,
    token 1 = code, the rest 0 (plus `offset` on every token); fused words = (words + ident[0]) on the sentence's
    real tokens, zeros past n (as the real stage)."""

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)
        self.frames = []   # (ident, code) of every frame fused

    def fuse(self, neck, words, wmask):
        maps = neck["neck_maps"].float()                       # [F, n_cam, cells, 256]
        F = int(maps.shape[0])
        n_tok = int(neck["neck_3d"].shape[1] * neck["neck_3d"].shape[2])
        assert words.shape[0] == wmask.shape[0] == F
        ident, code = maps[:, 0, 0, :3], words[:, 0, :3].float()
        tok = torch.zeros(F, n_tok, 256)
        tok[:, :, :3], tok[:, :, 3:6] = ident[:, None], code[:, None]
        pos = torch.zeros(F, n_tok, 3)
        pos[:, 0], pos[:, 1] = ident, code
        pos = pos + self.offset
        wf = (words.float() + ident[:, 0, None, None]) * wmask[..., None].float()
        self.frames += [(tuple(ident[f].tolist()), tuple(code[f].tolist())) for f in range(F)]
        return tok, pos, wf


def make_trainer(run, root, val=(), stub=None, **kw):
    cfg = dataclasses.replace(LangConfig(), batch_size=8, **kw)
    return LangTrainer(cfg, run, [root], list(val), (), use_wandb=False, prefetch=False, device="cpu",
                       text_encoder=StubText(), seed=0, probe_n=16,
                       fusion=(stub or StubFusion()) if cfg.scene_fusion else None)


def fusing(run, root, **kw):
    return make_trainer(run, root, scene_fusion=True, **kw)


@pytest.fixture
def v2(tmp_path):
    return make_root(tmp_path / "v2", episodes=4, capacity=400, neck=True)


def codes(words):
    return words[:, 0, :3].float()


def spy(mod, store: dict, name: str):
    """Record the scene frames and words of every forward of a LangPolicyEncoder (the last call wins)."""
    def hook(_m, args, kwargs):
        store[name] = {"scene_tokens": args[5].detach().clone(), "scene_pos": args[6].detach().clone(),
                       "words": kwargs["words"].detach().clone()}
    return mod.register_forward_pre_hook(hook, with_kwargs=True)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


def test_fused_batch_routes_every_frame_and_sentence(tmp_path, v2):
    """Live stack = frames t, t-k, 0 and target stack = t+1, t+1-k, 0 (the frames the stored tokens of the same
    sample come from), every frame fused with the sample's own sentence, frame 0 one pass shared by both stacks,
    w_fused of frame t / w_fused_next of frame t+1, one pass per unique (frame, sentence)."""
    tr = fusing(tmp_path / "run", v2)
    assert tr.sampler.with_neck and tr.fusion is not None
    b = tr.sampler.sample(16)
    words, wmask = tr.text.lookup(b["sentence"])
    fb = tr.fused(b, words, wmask)
    assert fb is not b and b["scene_tokens"].dtype == np.float16 and "_fused" not in b   # a shallow copy
    assert fb["_fused"] == ("", "_next")
    for k in ("scene_pos", "scene_pos_next"):
        assert fb[k].shape == b[k].shape and fb["scene_tokens" + k[9:]].shape == b["scene_tokens" + k[9:]].shape
        # the fused frame of each slot is the row the stored tokens of that slot come from
        assert torch.equal(fb[k][:, :, 0], torch.as_tensor(b[k][:, :, 0])), k
        # ... fused with the sample's own sentence
        assert torch.equal(fb[k][:, :, 1], codes(words)[:, None].expand(-1, 3, -1)), k
    assert torch.equal(fb["scene_tokens"][:, 2], fb["scene_tokens_next"][:, 2])      # frame 0, both stacks
    w_t, w_n = (torch.as_tensor(b[k][:, 0, 0, 0]) for k in ("scene_pos", "scene_pos_next"))
    m = wmask[..., None].float()
    assert torch.equal(fb["w_fused"], (words + w_t[:, None, None]) * m)
    assert torch.equal(fb["w_fused_next"], (words + w_n[:, None, None]) * m)
    want = {(tuple(b[k][i, f, 0].tolist()), b["sentence"][i]) for i in range(16)
            for k, fs in (("scene_pos", (0, 1, 2)), ("scene_pos_next", (0, 1))) for f in fs}
    assert tr.last_fused_frames == len(want) == len(set(tr.fusion.frames))
    # the two members of a pair: the same frames, their own sentences
    pairs = [np.flatnonzero(b["pair"] == p) for p in np.unique(b["pair"][b["pair"] >= 0])]
    assert pairs
    for i, j in pairs:
        assert torch.equal(fb["scene_pos"][i, :, 0], fb["scene_pos"][j, :, 0])
        if b["sentence"][i] != b["sentence"][j]:
            assert not torch.equal(fb["scene_tokens"][i], fb["scene_tokens"][j])
    # live frames only (probes): no target stack, and encode refuses to read it
    fl = tr.fused(b, words, wmask, target=False)
    assert fl["_fused"] == ("",) and fl["scene_tokens_next"] is b["scene_tokens_next"]
    with pytest.raises(RuntimeError, match="not fused"):
        tr.encode(fl, words, wmask, "_next", target=True)
    with pytest.raises(RuntimeError, match="not fused"):
        tr.encode(b, words, wmask)


@pytest.mark.parametrize("fusion_words", [True, False])
def test_grad_step_feeds_fused_frames_and_words(tmp_path, v2, fusion_words):
    """The live encoder gets the fused frames t, t-k, 0 and w_fused(t), the target encoder t+1, t+1-k, 0 and
    w_fused(t+1); with fusion_words off both get the cached words. perf/* and data/samples_per_episode logged."""
    tr = fusing(tmp_path / "run", v2, fusion_words=fusion_words)
    seen = {}
    spy(tr.policy_enc, seen, "live"), spy(tr.policy_enc_targ, seen, "target")
    b = tr.sampler.sample(8)
    out = tr.grad_step(b, log=True)
    words, wmask = tr.text.lookup(b["sentence"])
    fb = tr.fused(b, words, wmask)
    for name, sfx in (("live", ""), ("target", "_next")):
        assert torch.equal(seen[name]["scene_tokens"], fb["scene_tokens" + sfx]), name
        assert torch.equal(seen[name]["scene_pos"], fb["scene_pos" + sfx]), name
        assert torch.equal(seen[name]["words"], fb["w_fused" + sfx] if fusion_words else words), name
    assert np.isfinite(out["loss/q"]) and np.isfinite(out["loss/pi"])
    assert out["perf/fusion_ms"] > 0 and out["perf/step_ms"] >= out["perf/fusion_ms"]
    assert out["perf/fusion_frames"] == tr.last_fused_frames
    assert out["data/samples_per_episode"] == pytest.approx(8 / 4)   # 1 step x 8 samples, 4 episodes
    tr.reset_perf()
    assert tr.timer.read() == {}


def test_v4b_reads_the_fused_positions(tmp_path, v2):
    """V4b's attention-distance loss uses the positions the encoder saw: the stub moves every fused position by
    1 km, so the attention-weighted distance to the named item is >= 1 km (the stored positions are < 200 m)."""
    tr = fusing(tmp_path / "run", v2, stub=StubFusion(offset=1000.0), aux_attention_loss=True)
    out = tr.grad_step(tr.sampler.sample(8), log=True)
    assert out["aux/att_dist_cm_t"] > 1000.0 * 100
    off = make_trainer(tmp_path / "off", v2, aux_attention_loss=True)
    assert off.grad_step(off.sampler.sample(8), log=True)["aux/att_dist_cm_t"] < 300.0 * 100


def test_fusion_off_is_unchanged_on_a_version2_shard(tmp_path):
    """scene_fusion off: a version-2 shard trains exactly like its version-1 twin (no neck rows read, no stage)."""
    v1 = make_root(tmp_path / "v1", episodes=4, capacity=400)
    v2 = make_root(tmp_path / "v2", episodes=4, capacity=400, neck=True)
    a = make_trainer(tmp_path / "a", v1)          # each trainer seeds torch's global RNG: run them in turn
    oa = [a.grad_step(log=True) for _ in range(2)]
    c = make_trainer(tmp_path / "c", v2)
    oc = [c.grad_step(log=True) for _ in range(2)]
    assert c.fusion is None and not c.sampler.with_neck and "neck" not in c.sampler.sample(4)
    for x, y in zip(oa, oc):
        assert x["loss/q"] == y["loss/q"] and x["loss/pi"] == y["loss/pi"]
        assert "perf/fusion_ms" not in x and "perf/step_ms" in x
    for x, y in zip(a.policy_enc.state_dict().values(), c.policy_enc.state_dict().values()):
        assert torch.equal(x, y)


def test_fusion_needs_version2_shards(tmp_path):
    v1 = make_root(tmp_path / "v1", episodes=2, capacity=200)
    with pytest.raises(ValueError, match="version-1 shard"):
        fusing(tmp_path / "run", v1)


def test_init_weights_refuses_other_fusion_settings(tmp_path, v2):
    """V5 adds no parameter, so load_weights compares (scene_fusion, fused words) of the checkpoint's config; a
    config saved before the fields counts as unfused."""
    off, on = make_trainer(tmp_path / "off", v2), fusing(tmp_path / "on", v2)
    p_off, p_on = off.save_ckpt(), on.save_ckpt()
    with pytest.raises(ValueError, match="scene_fusion"):
        on.load_weights(p_off)
    with pytest.raises(ValueError, match="scene_fusion"):
        off.load_weights(p_on)
    with pytest.raises(ValueError, match="fused words"):
        fusing(tmp_path / "w", v2, fusion_words=False).load_weights(p_on)
    fusing(tmp_path / "on2", v2).load_weights(p_on)
    # fusion off: fusion_words does not matter (the effective flag)
    make_trainer(tmp_path / "off2", v2, fusion_words=False).load_weights(p_off)
    ck = torch.load(p_off, weights_only=False)
    for k in ("scene_fusion", "fusion_words"):
        del ck["lang_config"][k]
    torch.save(ck, tmp_path / "old.pt")
    off.load_weights(tmp_path / "old.pt")
    assert fusion_flags(ck["lang_config"]) == (False, False)
    with pytest.raises(ValueError):
        on.load_weights(tmp_path / "old.pt")


def test_init_weights_refuses_the_other_critic_readout(tmp_path, v2):
    """Decision 6: a checkpoint of the other readout is refused by name (not by the strict load's key list); a
    config saved before the field trained the softmin."""
    soft, mlp = make_trainer(tmp_path / "s", v2, critic_readout="softmin"), make_trainer(tmp_path / "m", v2)
    p_soft, p_mlp = soft.save_ckpt(), mlp.save_ckpt()
    with pytest.raises(ValueError, match="critic readout softmin"):
        mlp.load_weights(p_soft)
    with pytest.raises(ValueError, match="critic readout mlp"):
        soft.load_weights(p_mlp)
    make_trainer(tmp_path / "s2", v2, critic_readout="softmin").load_weights(p_soft)
    ck = torch.load(p_soft, weights_only=False)
    del ck["lang_config"]["critic_readout"]
    torch.save(ck, tmp_path / "old.pt")
    soft.load_weights(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="critic readout softmin"):
        mlp.load_weights(tmp_path / "old.pt")


def test_rounds_refuses_scene_fusion(tmp_path, v2):
    """Through the CLI (the refusal is rounds.run_rounds', before the ring's lock or files)."""
    run = tmp_path / "r"
    with pytest.raises(ValueError, match="rounds mode"):
        train_mod.main(["rounds", "--data", str(v2), "--run", str(run), "--no-wandb", "--set", "scene_fusion=true"])
    assert not (run / "ring").exists()


def test_train_episodes_per_task_reaches_the_training_sampler_only(tmp_path):
    root = make_root(tmp_path / "data", episodes=4, capacity=400, neck=True)
    tr = make_trainer(tmp_path / "run", root, val=[root], train_episodes_per_task=2)
    assert tr.cfg.train_episodes_per_task == 2 and tr.sampler.episodes_per_task == 2
    assert tr.val_sampler.episodes_per_task == 0 and tr.val_sampler.cfg.train_episodes_per_task == 0
    assert sum(tr.sampler.episode_counts().values()) == 2 and sum(tr.val_sampler.episode_counts().values()) == 4
    keep = {e for _, e in tr.sampler.episode_keep}
    assert set(tr.sampler.sample(32)["episode_id"].tolist()) <= keep
    out = tr.grad_step(log=True)
    assert out["data/samples_per_episode"] == pytest.approx(8 / 2)


def test_probes_fuse_with_the_probed_sentences(tmp_path, v2):
    """probe_forward fuses the frames with the sentences it is called with (shuffled / paraphrase probes too); the
    probe set keeps live frames only, one table its per-state sub-batches share; run_probe and aux_probe run."""
    tr = fusing(tmp_path / "run", v2, val=[v2], aux_target_loss=True)
    out = tr.run_probe()
    assert np.isfinite(out["lang/text_sens"]) and "lang/para_std" in out and "probe/aux_center_err_cm_t" in out
    pb = tr.prober.batch
    assert isinstance(pb["neck"], SharedNeck) and (pb["neck_idx"][:, 3:] == -1).all()
    assert pb["neck"][3] is pb["neck"] and neck_table([pb["neck"]] * 3) is pb["neck"]
    used = np.unique(pb["neck_idx"][:, :3])
    assert used.tolist() == list(range(pb["neck"]["neck_maps"].shape[0]))
    sub = {k: (v[:4] if isinstance(v, (np.ndarray, list)) else v) for k, v in pb.items()}
    other = list(reversed(sub["sentence"]))
    seen = {}
    spy(tr.policy_enc, seen, "live")
    q, _, _ = tr.probe_forward(sub, other, False)
    words, wmask = tr.text.lookup(other)
    assert q.shape == (4,) and torch.equal(seen["live"]["scene_pos"][:, :, 1], codes(words)[:, None].expand(-1, 3, -1))
    assert torch.equal(seen["live"]["scene_pos"][:, :, 0], torch.as_tensor(sub["scene_pos"][:, :, 0]))
    with pytest.raises(ValueError, match="live frames only"):
        tr.fused(sub, words, wmask)   # the probe set has no t+1 rows


def test_sanity_turns_the_fusion_off_for_the_switches_off_check(tmp_path, v2, monkeypatch):
    monkeypatch.setattr(train_mod, "make_fusion",
                        lambda cfg, device="cuda", spatial_enhancer=None: StubFusion() if cfg.scene_fusion else None)
    monkeypatch.setattr(train_mod, "LangTrainer", functools.partial(LangTrainer, device="cpu", text_encoder=StubText()))
    run = tmp_path / "sanity"
    train_mod.main(["sanity", "--data", str(v2), "--run", str(run), "--sanity-steps", "2", "--switch-steps", "1",
                    "--set", "scene_fusion=true", "batch_size=8"])
    res = json.loads((run / "sanity.json").read_text())
    assert res["switches_off"]["pass"] and res["switches_off"]["text_sens"] == 0.0
    assert "scene_fusion" in res["switches_off"]
    assert LangConfig.load(run / "overfit" / "lang_config.json").scene_fusion
    assert not LangConfig.load(run / "switches_off" / "lang_config.json").scene_fusion


# ---------------------------------------------------------------------------
# offline evaluation
# ---------------------------------------------------------------------------


@pytest.fixture
def fused_ckpt(tmp_path, v2):
    tr = fusing(tmp_path / "run", v2)
    tr.grad_step()
    tr.save_ckpt()
    return tmp_path / "run"


def evaluator(run, root, **kw):
    return Evaluator(run, "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText(), **kw)


def test_eval_fuses_each_state_with_its_sentence(fused_ckpt, v2):
    ev = evaluator(fused_ckpt, v2, fusion=StubFusion())
    assert ev.cfg.scene_fusion and ev.fusion is not None
    s = ev.sources[0]
    keys = [(0, i) for i in range(0, 40, 7)]
    sents = [f"don't move the {c} block" for c in ("red", "blue", "red", "green", "red", "blue")]
    words, wmask = ev.text.lookup(sents)
    st, sp, wf = ev.fused_scene(keys, sents, words, wmask)
    reads = [s.read(i) for _, i in keys]
    assert st.shape == (6, 3, 12, 256) and sp.shape == (6, 3, 12, 3) and wf.shape == words.shape
    for n, r in enumerate(reads):
        assert torch.equal(sp[n, :, 0], torch.as_tensor(r["scene_pos"][:, 0]))      # frames t, t-k, 0
        assert torch.equal(sp[n, :, 1], codes(words)[n].expand(3, -1))
        assert torch.equal(wf[n], (words[n] + float(r["scene_pos"][0, 0, 0])) * wmask[n, :, None].float())
    seen = {}
    spy(ev.pe, seen, "pe")
    ev.q_batch(reads, sents, keys, [None] * 6)
    assert torch.equal(seen["pe"]["scene_pos"], sp) and torch.equal(seen["pe"]["words"], wf)   # fused words
    with pytest.raises(ValueError, match="state key"):
        ev.q_batch(reads, sents)


def test_eval_cache_keys_and_lifetimes(fused_ckpt, v2):
    """Cache key (source, ring row, sentence): frame 0 of another state of the episode under the same sentence is a
    hit and survives a new chunk; frames t / t-k are dropped at the next chunk; another sentence is a miss."""
    stub = StubFusion()
    ev = evaluator(fused_ckpt, v2, fusion=stub)
    s = ev.sources[0]
    i, j = (int(x) for x in np.flatnonzero(s.eid == s.eid[0])[[5, 20]])   # two states of one episode
    sent = ["don't knock over the red block"]
    words, wmask = ev.text.lookup(sent)
    a = ev.fused_scene([(0, i)], sent, words, wmask)
    assert ev.fuse_stats["fused"] == 3 and len(stub.frames) == 3
    ev._fz_chunk.clear()                                   # what evaluate() does at a new chunk
    b = ev.fused_scene([(0, j)], sent, words, wmask)
    assert ev.fuse_stats["hit_first"] == 1 and ev.fuse_stats["fused"] == 5   # frame 0 reused, t / t-k fused
    assert torch.equal(a[0][0, 2], b[0][0, 2]) and torch.equal(a[1][0, 2], b[1][0, 2])
    c = ev.fused_scene([(0, j)], sent, words, wmask)
    assert ev.fuse_stats["hit_chunk"] == 2 and ev.fuse_stats["fused"] == 5 and all(torch.equal(x, y) for x, y in zip(b, c))
    w2, m2 = ev.text.lookup(["keep the blue block still"])
    ev.fused_scene([(0, j)], ["keep the blue block still"], w2, m2)
    assert ev.fuse_stats["fused"] == 8 and len(set(stub.frames)) == len(stub.frames) == 8
    small = evaluator(fused_ckpt, v2, fusion=StubFusion(), fuse_cache=1)   # LRU bound
    small.fused_scene([(0, i)], sent, words, wmask)
    assert len(small._fz_chunk) == 1 and len(small._fz_first) == 1


def test_eval_cached_scores_equal_uncached(fused_ckpt, v2):
    """A cached evaluation scores exactly what an uncached one does, fusing fewer frames."""
    stub, stub0 = StubFusion(), StubFusion()
    ev, ev0 = evaluator(fused_ckpt, v2, fusion=stub), evaluator(fused_ckpt, v2, fusion=stub0, fuse_cache=0)
    M, rows = ev.evaluate(per_task=40, n_para=3, chunk_states=8)
    M0, rows0 = ev0.evaluate(per_task=40, n_para=3, chunk_states=8)
    fz, fz0 = ev.fuse_stats, ev0.fuse_stats
    assert fz["frames"] == fz0["frames"] and fz0["hit_first"] == fz0["hit_chunk"] == 0
    assert fz["fused"] == len(stub.frames) <= fz0["fused"] == len(stub0.frames) < fz["frames"]
    assert fz["fused"] + fz["hit_first"] + fz["hit_chunk"] <= fz["frames"]
    assert [r["q"] for r in rows] == [r["q"] for r in rows0]
    p, p0 = M["train/symmetric/pooled"], M0["train/symmetric/pooled"]
    assert p["n_pairs"] == p0["n_pairs"] and p["rank_acc"] == p0["rank_acc"]
    d = data_info(ev, rows, [v2])
    assert d["scene_fusion"] and d["fused_words"] and d["fusion_frames"]["fused"] == fz["fused"]


def test_eval_version1_shard_refused(fused_ckpt, tmp_path):
    v1 = make_root(tmp_path / "v1", episodes=2, capacity=200)
    with pytest.raises(ValueError, match="version-1 shard"):
        evaluator(fused_ckpt, v1, fusion=StubFusion())


def test_eval_axes_train_only(tmp_path):
    """--axes train: the same states and train-axis sentences as a full run, no other axis, no base / paraphrase
    sentences; summarize leaves the base cells out. Default: every axis, base and paraphrases."""
    root = make_root(tmp_path / "val", episodes=4, capacity=400)
    tr = make_trainer(tmp_path / "run", root)
    tr.grad_step()
    tr.save_ckpt()
    M, rows = evaluator(tmp_path / "run", root).evaluate(per_task=12, n_para=3, chunk_states=8)
    ev = evaluator(tmp_path / "run", root)
    Mt, rt = ev.evaluate(per_task=12, n_para=3, chunk_states=8, axes=("train",))
    assert ev.axes_scored == ("train",) and not ev.extras_scored
    assert any("base" in r["q"] for r in rows) and any("para" in r["q"] for r in rows)
    assert all(set(r["q"]) == {"train"} and set(r["sentences"]) == {"train"} for r in rt)
    full = {(r["src"], r["i"], r["rule"]): r for r in rows if r["eligible"]["train"]}
    assert set(full) == {(r["src"], r["i"], r["rule"]) for r in rt}
    for r in rt:
        f = full[(r["src"], r["i"], r["rule"])]
        assert r["sentences"]["train"] == f["sentences"]["train"]
        assert abs(r["q"]["train"] - f["q"]["train"]) < 1e-5
    assert Mt["train/symmetric/pooled"]["n_pairs"] == M["train/symmetric/pooled"]["n_pairs"]
    assert not any(k.split("/")[0] != "train" for k in Mt if not k.startswith("aux/"))
    assert "auroc" in Mt["train/state/pooled"] and "auroc_base" not in Mt["train/state/pooled"]
    # rows of a train-only run summarize over every axis without a KeyError (pairs of unscored axes left out)
    Ma = summarize(rt)
    assert "heldout_template/symmetric/pooled" not in Ma and "shuffled/state/pooled" not in Ma
    for key in ("n_pairs", "rank_acc"):
        assert Ma["train/symmetric/pooled"][key] == Mt["train/symmetric/pooled"][key]
    assert data_info(ev, rt, [root])["axes"] == ["train"]
    with pytest.raises(ValueError, match="subset"):
        ev.evaluate(per_task=4, axes=("nope",))


def test_eval_records_the_new_settings(fused_ckpt, v2):
    ev = evaluator(fused_ckpt, v2, fusion=StubFusion())
    mi = model_info(ev.cfg)
    assert mi["scene_fusion"] and mi["fusion_words"] and mi["fusion_dtype"] == "fp32"
    assert mi["critic_readout"] == LangConfig().critic_readout and mi["train_episodes_per_task"] == 0
    assert "scene fusion fp32" in model_line(ev.cfg)
    old = {k: v for k, v in LangConfig().to_dict().items() if k not in ("scene_fusion", "critic_readout")}
    mo = model_info(LangConfig.from_dict(old))
    assert not mo["scene_fusion"] and not mo["fused_words"] and mo["critic_readout"] == "softmin"


# ---------------------------------------------------------------------------
# closed loop: the controller gets the checkpoint's stage
# ---------------------------------------------------------------------------


def test_controller_fusion_uses_the_rollout_encoders_spatial_enhancer(monkeypatch):
    from main.lang.eval_closed import controller_fusion

    calls = []
    monkeypatch.setattr(train_mod, "make_fusion",
                        lambda cfg, device="cuda", spatial_enhancer=None: calls.append((device, spatial_enhancer)) or "stage")
    enc = SimpleNamespace(spatial_enhancer=object())
    assert controller_fusion(LangConfig(), enc) is None and not calls
    assert controller_fusion(dataclasses.replace(LangConfig(), scene_fusion=True), enc) == "stage"
    assert calls == [("cuda", enc.spatial_enhancer)]


# ---------------------------------------------------------------------------
# the real stage (checkpoint weights): fused() against per-frame official passes
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not (torch.cuda.is_available() or os.environ.get("LANG_FUSION_TEST_CPU")),
                    reason="needs CUDA (LANG_FUSION_TEST_CPU=1 runs it on the CPU) and the HoloBrain checkpoint")
def test_fused_with_the_real_stage_matches_single_frame_passes():
    from main.lang.fusion import FusionStage, neck_to_row, rows_to_neck, synthetic_neck
    from main.train.collect import CKPT_DIR

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stage = FusionStage(str(CKPT_DIR), device=dev, chunk=2)
    neck = synthetic_neck(3, seed=1)
    rows = [neck_to_row({k: v[f:f + 1] for k, v in neck.items()}) for f in range(3)]
    table = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    g = torch.Generator().manual_seed(0)
    words = torch.randn(2, 24, 256, generator=g).to(dev)
    wmask = torch.zeros(2, 24, dtype=torch.bool, device=dev)
    wmask[0, :6], wmask[1, :9] = True, True
    b = {"neck": table, "neck_idx": np.array([[0, 1, 2, 1, 2], [2, 0, 1, 1, 0]]), "sentence": ["a", "b"]}
    host = SimpleNamespace(fusion=stage, dev=dev, last_fused_frames=0)
    fb = LangTrainer.fused(host, b, words, wmask)
    assert host.last_fused_frames == 6
    ref = rows_to_neck(table, dev)
    for i in range(2):
        for k, cols in (("", (0, 1, 2)), ("_next", (3, 4, 2))):
            for slot, c in enumerate(cols):
                f = int(b["neck_idx"][i, c])
                t, p, w = stage.fuse({kk: v[f:f + 1] for kk, v in ref.items()}, words[i:i + 1], wmask[i:i + 1])
                assert (fb["scene_tokens" + k][i, slot] - t[0]).abs().max() < 2e-5
                assert (fb["scene_pos" + k][i, slot] - p[0]).abs().max() < 1e-5
                if slot == 0:
                    assert (fb["w_fused" + k][i] - w[0]).abs().max() < 2e-5
