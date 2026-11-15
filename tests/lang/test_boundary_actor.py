"""The boundary actor (docs/lang_filter/T8_BOUNDARY_PLAN.md): off by default, trained beside the argmax actor without
touching the encoder, critics or argmax actor updates, saved and resumed, drawn per episode from the seed."""

import dataclasses

import numpy as np
import pytest
import torch

from main.lang.config import LangConfig, episode_actor
from main.lang.metrics import LangRoundMetrics

from .fake_data import make_root
from .test_train import trainer


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path / "data", episodes=4, capacity=400)


def _same(m1, m2, atol=0.0):
    return all(torch.allclose(a, b, atol=atol, rtol=0) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))


def test_off_by_default_and_old_configs_load_off(tmp_path, root):
    tr = trainer(tmp_path / "run", root)
    assert tr.actor_b is None and tr.opt_b is None
    out = tr.grad_step(log=True)
    assert "loss/pi_b" not in out
    assert "actor_boundary" not in tr.ckpt_dict()
    d = LangConfig().to_dict()
    for k in ("boundary_actor", "boundary_lo", "boundary_hi", "boundary_actor_frac"):
        d.pop(k)
    assert LangConfig.from_dict(d).boundary_actor is False


def test_beta_leaves_the_other_updates_unchanged(tmp_path, root):
    torch.set_num_threads(4)
    # each trainer reseeds the global torch RNG when built: build and step one at a time, so both steps draw the
    # same noise (built together, the second build's reseed shifts the first trainer's draws)
    a = trainer(tmp_path / "a", root)
    out_a = a.grad_step()
    b = trainer(tmp_path / "b", root, boundary_actor=True)
    out_b = b.grad_step()
    # one step: the encoder, critic and argmax-actor updates come before beta's, and beta's initial weights come
    # from a forked RNG, so they are identical
    assert all(_same(x, y) for x, y in ((a.policy_enc, b.policy_enc), (a.critics, b.critics), (a.actor, b.actor)))
    assert out_a["loss/q"] == out_b["loss/q"] and out_a["loss/pi"] == out_b["loss/pi"]
    for k in ("loss/pi_b", "actor_b/q_mean", "actor_b/in_band_frac", "actor_b/hinge", "grad/actor_b"):
        assert np.isfinite(out_b[k]), k
    # the critics and the encoder are trainable again after the step
    assert all(p.requires_grad for p in b.critics.parameters())


def test_hinge_is_zero_inside_the_band(tmp_path, root):
    tr = trainer(tmp_path / "run", root, boundary_actor=True, boundary_lo=-100.0, boundary_hi=100.0)
    out = tr.grad_step()
    assert out["actor_b/hinge"] == 0.0 and out["actor_b/in_band_frac"] == 1.0
    tr = trainer(tmp_path / "run2", root, boundary_actor=True, boundary_lo=50.0, boundary_hi=60.0)
    out = tr.grad_step()   # every Q is far below the band: hinge = lo - Q
    assert out["actor_b/in_band_frac"] == 0.0
    assert out["actor_b/hinge"] == pytest.approx(50.0 - out["actor_b/q_mean"], rel=1e-4)


def test_hinge_gradient_moves_q_toward_the_band(tmp_path, root):
    """With the critics fixed, steps on the hinge alone (no entropy) raise Q when the band is above it."""
    torch.manual_seed(0)
    tr = trainer(tmp_path / "run", root, boundary_actor=True, boundary_lo=5.0, boundary_hi=6.0,
                 alpha_start=0.0, alpha_final=0.0)
    for g in tr.opt_e.param_groups + tr.opt_c.param_groups + tr.opt_a.param_groups:
        g["lr"] = 0.0
    for g in tr.opt_b.param_groups:
        g["lr"] = 1e-2
    b = tr.sampler.sample(8)
    q0 = tr.grad_step(b)["actor_b/q_mean"]
    for _ in range(15):
        q = tr.grad_step(b)["actor_b/q_mean"]
    assert q > q0


def test_checkpoint_and_exact_resume_with_beta(tmp_path, root):
    torch.set_num_threads(4)
    a = trainer(tmp_path / "a", root, boundary_actor=True)
    for _ in range(4):
        a.grad_step()
    b = trainer(tmp_path / "b", root, boundary_actor=True)
    for _ in range(2):
        b.grad_step()
    b.save_state()
    c = trainer(tmp_path / "b", root, boundary_actor=True)
    c.load_state()
    for _ in range(2):
        c.grad_step()
    for x, y in zip(a.actor_b.state_dict().values(), c.actor_b.state_dict().values()):
        assert torch.allclose(x, y, atol=1e-6)
    p = a.save_ckpt()
    d = trainer(tmp_path / "d", root, boundary_actor=True)
    d.load_weights(p)
    assert _same(a.actor_b, d.actor_b)
    plain = trainer(tmp_path / "e", root)   # an evaluation-style trainer ignores the boundary actor
    plain.load_weights(p)
    assert _same(a.actor, plain.actor)


def test_episode_actor_is_drawn_from_the_seed():
    off = LangConfig()
    assert {episode_actor(off, s) for s in range(50)} == {"argmax"}
    on = dataclasses.replace(LangConfig(), boundary_actor=True, boundary_actor_frac=0.5)
    kinds = [episode_actor(on, 81_000_000 + s) for s in range(2000)]
    assert kinds == [episode_actor(on, 81_000_000 + s) for s in range(2000)]
    assert 0.45 < kinds.count("boundary") / len(kinds) < 0.55
    all_b = dataclasses.replace(on, boundary_actor_frac=1.0)
    assert {episode_actor(all_b, s) for s in range(50)} == {"boundary"}


def test_round_summary_splits_by_actor(tmp_path):
    rm = LangRoundMetrics(tmp_path)
    for i, (kind, harm) in enumerate([("argmax", False), ("boundary", True), ("boundary", False)]):
        line = {"task": "tray_slide", "embodiment": "ur5-wsg", "named_consequence": "upright", "designation": "harm",
                "episode_id": i, "status": "ok", "success": True, "n_ticks": 10, "actor": kind}
        row = LangRoundMetrics.episode_row(1, line, {"h_named": [0.5, -0.2 if harm else 0.4]})
        assert row["actor"] == kind
        rm.add(row)
    out = rm.round_summary(1)
    assert out["round_overview/actor_boundary/n"] == 2 and out["round_overview/actor_argmax/n"] == 1
    assert out["round_overview/actor_boundary/harm_named"] == 0.5
    assert out["round_overview/actor_argmax/harm_named"] == 0.0
