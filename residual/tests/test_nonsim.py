"""Non-simulator checks of the residual package (run from the repo root):

    OMP_NUM_THREADS=1 python -m pytest -q residual/tests/test_nonsim.py
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from residual.config import N_ACT, ResidualConfig, apply_sets
from residual.policy import build_model
from residual.ppo import episode_advantages, make_batch, ppo_update


def test_config_roundtrip_and_sets():
    cfg = apply_sets(ResidualConfig(), ['replan_trigger="fixed"', "replan_every=3", "completion=always"])
    assert cfg.replan_trigger == "fixed" and cfg.replan_every == 3 and cfg.completion == "always"
    back = ResidualConfig.from_dict(cfg.to_dict() | {"unknown_key": 1})
    assert back == cfg
    assert abs(cfg.completion_value - 100.0) < 1e-9
    with pytest.raises(ValueError):
        apply_sets(ResidualConfig(), ["replan_threshold=0.3"])   # must be below the delta bound (0.2)


def test_gae_terminal_finished_truncated():
    g, lam = 0.99, 0.95
    ep = {"rew": np.array([1.0, 1.0]), "disc": np.array([g, g]), "value": np.array([0.0, 0.0]),
          "end": "violation", "boot": 0.0, "last_value": 0.0}
    adv, ret = episode_advantages(ep, lam)
    assert np.allclose(ret, [1 + g * lam * 1.0, 1.0])
    # finished: the completion value enters as the next value of the last transition
    ep2 = dict(ep, end="finished", boot=100.0, value=np.array([100.0, 100.0]))
    adv2, ret2 = episode_advantages(ep2, lam)
    assert np.isclose(adv2[-1], 1.0 + g * 100.0 - 100.0)
    # truncated: V(s_T)
    ep3 = dict(ep, end="truncated", last_value=50.0, value=np.array([10.0, 20.0]))
    adv3, _ = episode_advantages(ep3, lam)
    assert np.isclose(adv3[-1], 1.0 + g * 50.0 - 20.0)
    # an SMDP transition (3 ticks inside one transition) uses its own discount
    ep4 = dict(ep, end="finished", boot=100.0, disc=np.array([g ** 3, g]), value=np.array([0.0, 0.0]))
    a4, _ = episode_advantages(ep4, lam)
    d1 = 1.0 + g * 100.0
    assert np.isclose(a4[-1], d1) and np.isclose(a4[0], 1.0 + g ** 3 * 0.0 + g ** 3 * lam * d1)


def test_constant_reward_return_is_completion_value():
    """r = 1 every tick then the closed-form completion value: the return is 1 / (1 - gamma) at every tick, so
    finishing early or late never changes it."""
    g = 0.99
    for T in (5, 50):
        ep = {"rew": np.ones(T), "disc": np.full(T, g), "value": np.zeros(T), "end": "finished", "boot": 1 / (1 - g)}
        _, ret = episode_advantages(ep, lam=1.0)
        assert np.allclose(ret, 1 / (1 - g))


def test_policy_and_update_shapes():
    cfg = ResidualConfig(minibatch=64, epochs=2)
    model = build_model(cfg, 20, N_ACT)
    obs = np.random.default_rng(0).normal(size=20).astype(np.float32)
    u, logp, v = model.act(obs)
    assert u.shape == (N_ACT,) and np.isfinite(logp) and np.isfinite(v)
    std0 = float(model.std().mean())
    assert abs(std0 - cfg.init_std) < 1e-6
    um, _, _ = model.act(obs, deterministic=True)
    assert np.max(np.abs(um)) < 0.2   # small output layer: the mean residual starts near 0
    rng = np.random.default_rng(1)
    eps = []
    for _ in range(4):
        T = 40
        o = rng.normal(size=(T, 20)).astype(np.float32)
        us, lps, vs = zip(*[model.act(x) for x in o])
        eps.append({"obs": o, "u": np.array(us, dtype=np.float32), "logp": np.array(lps, dtype=np.float32),
                    "value": np.array(vs, dtype=np.float32), "rew": np.ones(T, dtype=np.float32),
                    "disc": np.full(T, 0.99, dtype=np.float32), "end": "finished", "boot": 100.0, "last_value": 0.0})
    batch = make_batch(eps, 0.95)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = ppo_update(model, opt, batch, cfg, np.random.default_rng(2))
    assert batch["n"] == 160 and np.isfinite(stats["loss_v"]) and np.isfinite(stats["kl"])
    model.norm.update(batch["obs"])
    assert torch.all(model.norm.var > 0)


def test_obs_dim_matches_build_obs():
    from residual.state import build_obs, obs_dim

    cfg = ResidualConfig()
    n_links = 26
    m = {"T": np.eye(4), "link_pos": np.zeros((n_links, 3)), "link_d": np.ones(n_links), "link_dir": np.zeros((n_links, 3)),
         "held": [(0.1, np.array([1.0, 0, 0]), np.array([0.3, 0, 1.0]), np.eye(3), np.ones(3) * 0.02)],
         "vel": np.zeros(6), "tilt": 0.0}
    rob = {"q_left": np.zeros(6), "q_right": np.zeros(6), "qd_left": np.zeros(6), "qd_right": np.zeros(6),
           "grip": np.ones(2), "ee_left": np.array([0.3, 0, 1.0, 1, 0, 0, 0]), "ee_right": np.array([-0.3, 0, 1, 1, 0, 0, 0])}
    view = {"a_nom": np.zeros(12), "future": np.zeros((len(cfg.future_ticks), 12)), "active": np.ones(2)}
    probe = types.SimpleNamespace(spec=types.SimpleNamespace(half_extents=np.ones(3) * 0.03), start_pos=np.zeros(3),
                                  named=types.SimpleNamespace(kind="stay"))
    obs = build_obs(cfg, probe, m, rob, view, 1.0)
    assert obs.shape == (obs_dim(cfg, n_links),) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    # the held object went to the left hand (nearest), the right slot is empty
    assert obs[24 + 2 + 78] == 1.0 and obs[24 + 2 + 78 + 13] == 0.0


def test_balancer_keeps_the_share(monkeypatch):
    import residual.scene as scene
    from residual.episode import Job
    from residual.worker import SideBalancer

    cfg = ResidualConfig()
    layouts = []

    def fake_options(runner):
        return runner

    def fake_pick(runner, in_way, kind, rng):
        return types.SimpleNamespace(in_way=in_way, kind=kind)

    monkeypatch.setattr(scene, "named_options", fake_options)
    monkeypatch.setattr(scene, "pick_named", fake_pick)
    bal = SideBalancer(cfg)
    rng = np.random.default_rng(0)
    rebuilt = 0
    got = []
    for k in range(400):
        while True:
            has_out = rng.random() < 0.35            # most scenes have only in-the-way objects
            runner = {True: ["a"], False: ["b"] if has_out else []}
            named = bal(runner, rng, Job())
            if named is not None:
                break
            rebuilt += 1
        got.append(named.in_way)
    share = np.mean(got)
    assert 0.45 <= share <= 0.55, share
    assert rebuilt < 400   # rebuilds stay below one per episode


def test_select_eval_scenes_balanced():
    from residual.train import select_eval_scenes

    hosts = ["h0", "h1", "h2", "h3", "h4"]
    scans = []
    for s in range(60):
        roles = {"obj_a": "on_path", "obj_b": "near_path"}
        if s % 3 == 0:
            roles["obj_c"] = "out_of_way"
        scans.append({"seed": s, "host": hosts[s % 5], "roles": roles, "n_candidates": len(roles),
                      "categories": {k: "apple" for k in roles}})
    sel = select_eval_scenes(scans, 10, 6)
    ins = [c for c in sel if c["in_way"]]
    outs = [c for c in sel if not c["in_way"]]
    assert len(ins) == 10 and len(outs) == 6
    assert all(c["item_id"] == "obj_a" for c in ins)                 # the most in-the-way object
    assert {c["rule_kind"] for c in ins} == {"avoid", "stay"}
    assert len({c["seed"] for c in sel}) == len(sel)                 # one case per seed
