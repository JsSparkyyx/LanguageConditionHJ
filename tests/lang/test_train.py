"""LangTrainer on CPU with fake data and a stub text encoder: one update,
target coverage, checkpoint round trip, exact resume."""

import dataclasses

import numpy as np
import pytest
import torch

from main.lang.config import LangConfig
from main.lang.data import stable_hash
from main.lang.train import LangTrainer, load_policy, target_rank_acc

from .fake_data import make_root


class StubText:
    def embed(self, sentences):
        e = torch.stack([torch.randn(24, 256, generator=torch.Generator().manual_seed(stable_hash(s) % 2**31))
                         for s in sentences])
        m = torch.zeros(len(sentences), 24, dtype=torch.bool)
        m[:, :10] = True
        return e, m


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path / "data", episodes=4, capacity=400)


def trainer(run, root, prefetch=False, **kw):
    cfg = dataclasses.replace(LangConfig(), batch_size=8, **kw)
    return LangTrainer(cfg, run, [root], (), (), use_wandb=False, prefetch=prefetch, device="cpu",
                       text_encoder=StubText(), seed=0)


def test_grad_step_and_targets(tmp_path, root):
    torch.set_num_threads(4)
    tr = trainer(tmp_path / "run", root)
    out = tr.grad_step(log=True)
    for k in ("loss/q", "loss/pi", "q/mean", "lang/pair_rank_acc", "grad/encoder", "grad/text_adapter", "actor/action_std"):
        assert k in out
    assert np.isfinite(out["loss/q"]) and np.isfinite(out["loss/pi"])
    live = list(tr.policy_enc.parameters()) + list(tr.critics.parameters())
    targ = list(tr.policy_enc_targ.parameters()) + list(tr.critics_targ.parameters())
    assert len(live) == len(targ) == len(tr.target_pairs)
    assert all(a.shape == b.shape and not a.requires_grad for a, b in zip(targ, live))
    assert all(p.requires_grad for p in live)   # restored after the actor step


def test_checkpoint_roundtrip(tmp_path, root):
    tr = trainer(tmp_path / "run", root)
    tr.grad_step()
    p = tr.save_ckpt()
    tr2 = trainer(tmp_path / "run2", root)
    tr2.load_weights(p)
    for a, b in zip(tr.policy_enc.state_dict().values(), tr2.policy_enc.state_dict().values()):
        assert torch.equal(a, b)
    off = trainer(tmp_path / "run3", root, lang_film=False)
    with pytest.raises(ValueError):
        off.load_weights(p)


@pytest.mark.parametrize("prefetch", [False, True])
def test_exact_resume(tmp_path, root, prefetch):
    torch.set_num_threads(4)
    a = trainer(tmp_path / "a", root, prefetch=prefetch)
    for _ in range(4):
        a.grad_step()
    b = trainer(tmp_path / "b", root, prefetch=prefetch)
    for _ in range(2):
        b.grad_step()
    b.save_state()
    if b.prefetch is not None:
        b.prefetch.stop()
    c = trainer(tmp_path / "b", root, prefetch=prefetch)
    c.load_state()
    assert c.step == 2
    for _ in range(2):
        c.grad_step()
    for x, y in zip(a.policy_enc.state_dict().values(), c.policy_enc.state_dict().values()):
        assert torch.allclose(x, y, atol=1e-6)
    for x, y in zip(a.actor.state_dict().values(), c.actor.state_dict().values()):
        assert torch.allclose(x, y, atol=1e-6)
    for t in (a, b, c):
        if t.prefetch is not None:
            t.prefetch.stop()


def test_target_rank_acc():
    q = np.array([0.1, 0.5, 0.3, 0.2, 0.9, 0.9])
    y = np.array([0.0, 0.6, 0.5, 0.1, 0.4, 0.42])
    pair = np.array([0, 0, 1, 1, 2, 2])
    assert target_rank_acc(q, y, pair) == 1.0   # pair 2 below the 0.05 threshold


def test_probe_sentences_ignore_the_run_text_control(tmp_path, root):
    """M3: in a --shuffle-text / --relabel fixed run the probes still pair each state with its own rule's
    sentence and draw rules at random."""
    from main.item_envs import language

    cfg = dataclasses.replace(LangConfig(), batch_size=8, shuffle_text=True, relabel="fixed")
    tr = LangTrainer(cfg, tmp_path / "run", [root], [root], (), use_wandb=False, prefetch=False, device="cpu",
                     text_encoder=StubText(), seed=0)
    assert tr.sampler.cfg.shuffle_text and not tr.val_sampler.cfg.shuffle_text
    assert tr.val_sampler.cfg.relabel == "random"
    b = tr.val_sampler.sample(32)
    bank = language.load_bank()
    for k in range(32):
        src = tr.val_sampler.sources[b["source"][k]] if isinstance(b["source"][k], (int, np.integer)) else \
            next(s for s in tr.val_sampler.sources if s.name == b["source"][k])
        rec = src.rs.records[int(b["episode_id"][k])]
        rule = next(r for r in rec["rules"] if r["rule_id"] == b["rule_id"][k])
        assert b["sentence"][k] in language.sentences(rule, rec, "train", bank)


def test_resume_without_state_uses_the_flags(tmp_path):
    """O4: a lang_config.json from a start that never saved state does not override corrected flags."""
    from types import SimpleNamespace

    from main.lang.train import config_from_args

    run = tmp_path / "run"
    run.mkdir()
    dataclasses.replace(LangConfig(), lang_film=False).save(run / "lang_config.json")
    flags = dict(run=str(run), resume=True, relabel=None, shuffle_text=False, text_backend=None,
                 activation_checkpointing=False, tasks="all", embodiments="all", set=[],
                 **{f"no_{k}": False for k in ("scene", "body", "film", "queries", "rtoken")})
    assert config_from_args(SimpleNamespace(**flags)).lang_film is True     # no state.pt: the flags win
    (run / "state.pt").write_bytes(b"x")
    assert config_from_args(SimpleNamespace(**flags)).lang_film is False    # a started run keeps its config


def test_training_defaults_to_one_embodiment_and_resume_keeps_the_saved_one(tmp_path):
    """Training uses ur5-wsg only unless --embodiments says otherwise; the trained subset comes from the
    run's config, so a resumed run keeps what it started with whatever this launch's flags say."""
    import argparse
    from types import SimpleNamespace

    from main.lang.config import CANON_EMBODIMENTS
    from main.lang.train import add_common, config_from_args, trained_subsets

    p = argparse.ArgumentParser()
    add_common(p)
    args = p.parse_args(["--data", "d", "--run", str(tmp_path / "run")])
    assert args.embodiments == "ur5-wsg"
    cfg = config_from_args(args)
    assert cfg.embodiment_choices == ("ur5-wsg",)
    assert trained_subsets(cfg) == (None, ("ur5-wsg",))
    cfg_all = config_from_args(p.parse_args(["--data", "d", "--run", str(tmp_path / "run"), "--embodiments", "all"]))
    assert cfg_all.embodiment_choices == CANON_EMBODIMENTS and trained_subsets(cfg_all) == (None, None)
    # a run started on both embodiments resumes on both, though the default flag is now ur5-wsg
    run = tmp_path / "old"
    run.mkdir()
    LangConfig().save(run / "lang_config.json")
    (run / "state.pt").write_bytes(b"x")
    old = config_from_args(SimpleNamespace(**{**vars(args), "run": str(run), "resume": True}))
    assert trained_subsets(old) == (None, None)
