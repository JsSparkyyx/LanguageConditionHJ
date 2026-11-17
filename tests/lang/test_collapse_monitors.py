"""The collapse monitors (main/lang/metrics.py: critic_collapse_stats, actor_collapse_stats, hook_collapse_stats):
is the critic action-agnostic, is the actor observation-agnostic; logged by the trainer and per round."""

import dataclasses

import numpy as np
import pytest
import torch

from main.lang.metrics import (HOOK_COLLAPSE_KEYS, SCALES, LangRoundMetrics, actor_collapse_stats,
                               critic_collapse_stats, hook_collapse_stats)

from .fake_data import make_root
from .test_train import trainer


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path / "data", episodes=4, capacity=400)


def test_critic_stats_are_zero_for_an_action_agnostic_critic():
    rng = np.random.default_rng(0)
    B = 32
    q_state = rng.normal(0, 0.5, B)                       # Q depends on the state only
    bd = np.arange(B) < 8
    out = critic_collapse_stats(np.tile(q_state, (8, 1)), np.tile(q_state, (len(SCALES), 1)), q_state, bd)
    for k in ("critic_rand_range", "critic_rand_over_state", "critic_scale_range", "critic_stop_minus_data",
              "critic_rand_range_boundary", "critic_scale_range_boundary", "critic_pi_minus_data_boundary"):
        assert out[f"collapse/{k}"] == pytest.approx(0.0, abs=1e-12), k
    assert out["collapse/boundary_rows"] == 8


def test_critic_stats_see_an_action_dependent_critic():
    B = 16
    scale = np.asarray(SCALES)[:, None]
    q_scale = 0.2 - 0.6 * scale * np.ones((1, B))         # the faster the stored decision, the lower Q
    q_rand = np.linspace(-0.3, 0.3, 8)[:, None] * np.ones((1, B))
    bd = np.zeros(B, bool)
    bd[:4] = True
    out = critic_collapse_stats(q_rand, q_scale, q_scale[0], bd)
    assert out["collapse/critic_rand_range"] == pytest.approx(0.6)
    assert out["collapse/critic_scale_range_boundary"] == pytest.approx(0.9)
    assert out["collapse/critic_stop_minus_data_boundary"] == pytest.approx(0.6)     # stopping scores higher
    assert out["collapse/critic_half_minus_data"] == pytest.approx(0.3)
    assert out["collapse/critic_pi_minus_data"] == pytest.approx(0.6)                # the actor's (a stop here)
    no_bd = critic_collapse_stats(q_rand, q_scale, q_scale[0], np.zeros(B, bool))
    assert np.isnan(no_bd["collapse/critic_scale_range_boundary"]) and no_bd["collapse/boundary_rows"] == 0


def test_actor_stats_tell_a_constant_actor_from_one_that_follows_the_state():
    rng = np.random.default_rng(1)
    B, A = 32, 6
    valid = np.ones((B, A), bool)
    valid[:, 4:] = False                                  # two padded joints
    data = rng.uniform(-0.5, 0.5, (B, A))
    pair = np.full(B, -1)
    pair[:8] = np.repeat(np.arange(4), 2)                 # 4 pairs: the same transition, two sentences
    data[1:8:2] = data[0:8:2]
    bd = np.arange(B) % 2 == 0
    stop = actor_collapse_stats(np.zeros((B, A)), valid, data, pair, bd)
    assert stop["collapse/actor_obs_std"] == 0.0 and stop["collapse/actor_abs"] == 0.0
    assert stop["collapse/actor_pair_diff"] == 0.0 and stop["collapse/actor_norm_ratio_data"] == 0.0
    follow = actor_collapse_stats(0.8 * data, valid, data, pair, bd)   # slows the data's motion to 80%
    assert follow["collapse/actor_obs_std"] > 0.1
    assert follow["collapse/actor_cos_data"] == pytest.approx(1.0)
    assert follow["collapse/actor_norm_ratio_data"] == pytest.approx(0.8)
    assert follow["collapse/actor_cos_data_boundary"] == pytest.approx(1.0)
    assert follow["collapse/actor_pair_diff"] == pytest.approx(0.0)   # same transition, same action: ignores the sentence
    by_sentence = 0.8 * data.copy()
    by_sentence[1:8:2] *= -1                              # the other sentence of each pair reverses the action
    assert actor_collapse_stats(by_sentence, valid, data, pair, bd)["collapse/actor_pair_diff"] > 0.1
    named = actor_collapse_stats(np.zeros((B, A)), valid, data, pair, bd, name="actor_b")
    assert "collapse/actor_b_obs_std" in named


def test_hook_stats_from_a_rollout_trace():
    assert all(np.isnan(v) for v in hook_collapse_stats({"V": [0.5, 0.4]}).values())   # a one-tick run: none recorded
    n = 6
    trace = {"V": [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], "Q_stop": [0.9, 0.6, 0.4, 0.3, -0.2, -0.9],
             "Q_pi": [0.9, 0.5, 0.2, 0.0, -0.5, -0.9],
             "a_pi": [np.full(4, 0.1 * k, np.float32) for k in range(n)],
             "a_nom": [np.full(4, 0.3, np.float32)] * n}
    out = hook_collapse_stats(trace)
    assert set(out) == set(HOOK_COLLAPSE_KEYS)
    assert out["n_boundary_hooks"] == 2                                       # |Q(s, a_nom)| < 0.25: 0.1 and -0.1
    assert out["q_stop_minus_nom_boundary"] == pytest.approx((0.3 + 0.4) / 2)
    assert out["q_decision_range_boundary"] == pytest.approx((0.3 + 0.4) / 2)
    assert out["q_pi_minus_nom"] == pytest.approx((0.1 + 0.1) / 6)
    assert out["actor_obs_std"] == pytest.approx(np.std(0.1 * np.arange(n)))
    assert out["actor_cos_nom"] == pytest.approx(5 / 6)       # the first hook's action is a stop: cosine 0 there
    flat = dict(trace, Q_stop=trace["V"], Q_pi=trace["V"], a_pi=[np.zeros(4, np.float32)] * n)
    out = hook_collapse_stats(flat)
    assert out["q_decision_range"] == 0.0 and out["actor_obs_std"] == 0.0 and out["actor_abs"] == 0.0


def test_round_summary_carries_the_hook_stats(tmp_path):
    rm = LangRoundMetrics(tmp_path)
    trace = {"h_named": [0.5, 0.4], "V": [0.2, -0.1], "Q_stop": [0.4, 0.1], "Q_pi": [0.2, -0.1],
             "a_pi": [np.zeros(4, np.float32), np.full(4, 0.2, np.float32)], "a_nom": [np.full(4, 0.3, np.float32)] * 2}
    for i, tr in enumerate((trace, {"h_named": [0.5, 0.4]})):   # the second episode recorded none
        line = {"task": "tray_slide", "embodiment": "ur5-wsg", "named_consequence": "upright", "designation": "harm",
                "episode_id": i, "status": "ok", "success": True, "n_ticks": 10}
        rm.add(LangRoundMetrics.episode_row(1, line, tr))
    line = {"task": "tray_slide", "embodiment": "ur5-wsg", "named_consequence": "upright", "designation": "harm",
            "episode_id": 9, "status": "ok", "success": True, "n_ticks": 10, "actor": "boundary"}
    rm.add(LangRoundMetrics.episode_row(1, line, dict(trace, Q_stop=[0.9, 0.6])))   # a boundary-actor episode
    out = rm.round_summary(1)
    assert out["collapse_round/actor_boundary/q_stop_minus_nom"] == pytest.approx(0.7)   # kept apart
    assert out["collapse_round/q_stop_minus_nom"] == pytest.approx(0.2)
    assert out["collapse_round/n_boundary_hooks"] == 2
    assert out["collapse_round/actor_obs_std"] == pytest.approx(0.1)


@pytest.mark.parametrize("kw", [{}, {"decision_ticks": 3, "critic_injection": "revised", "critic_readout": "mlp",
                                     "boundary_actor": True}])
def test_trainer_logs_the_monitors_without_touching_the_training_stream(tmp_path, root, kw):
    torch.set_num_threads(4)
    a = trainer(tmp_path / "a", root, **kw)
    out = a.grad_step(log=True)
    keys = ["collapse/critic_rand_range", "collapse/critic_rand_over_state", "collapse/critic_scale_range",
            "collapse/critic_stop_minus_data", "collapse/boundary_rows", "collapse/actor_obs_std",
            "collapse/actor_abs", "collapse/actor_cos_data", "critic/action_sens_rel"]
    if kw:
        keys += ["collapse/actor_b_obs_std", "collapse/actor_b_abs"]
    for k in keys:
        assert k in out and np.isfinite(out[k]), k
    assert out["collapse/critic_rand_range"] >= 0 and out["collapse/actor_obs_std"] >= 0
    # logging draws nothing from the global random stream: the next step is the same with and without it (each
    # trainer reseeds the global stream when built, so they run one after the other)
    second = a.grad_step()["loss/q"]
    b = trainer(tmp_path / "b", root, **kw)
    b.grad_step(log=False)
    assert second == b.grad_step()["loss/q"]
