"""One decision = decision_ticks control ticks (LangConfig.decision_ticks; docs/lang_filter/T8_3TICK_PLAN.md): the
target over the decision, the sampler's window, the trainer with the revised injection, the filter's held block."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.lang.config import LangConfig
from main.lang.data import LangSampler
from main.lang.network import decision_target, policy_decision, stored_decision

from .fake_data import make_root
from .test_train import trainer

H3 = dict(decision_ticks=3, critic_injection="revised", critic_readout="mlp")


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path / "data", episodes=4, capacity=400)


def test_target_one_tick_is_crosssafes_and_three_ticks_recurse():
    g = 0.95
    h = torch.tensor([0.3, -0.2, 1.0])
    q = torch.tensor([0.5, 0.4, -0.7])
    assert torch.allclose(decision_target(h[:, None], q, g), (1 - g) * h + g * torch.minimum(h, q))
    # the docstring's example: h = (1, 1, -1), q_end = 1 -> y_2 = -1, y_1 = -0.9, y_0 = -0.805
    y = decision_target(torch.tensor([[1.0, 1.0, -1.0]]), torch.tensor([1.0]), g)
    assert float(y) == pytest.approx(-0.805)
    # a violation inside the decision is not missed: the in-between tick's h bounds the target
    y = decision_target(torch.tensor([[1.0, -0.5, 1.0]]), torch.tensor([1.0]), g)
    assert float(y) < 0


def test_actions_of_a_decision():
    cfg1, cfg3 = LangConfig(), dataclasses.replace(LangConfig(), **H3)
    a = torch.randn(2, 6)
    seq = torch.randn(2, 3, 6)
    assert torch.equal(stored_decision(cfg1, a, seq), a) and torch.equal(policy_decision(cfg1, a), a)
    assert torch.allclose(stored_decision(cfg3, a, seq), seq.sum(1))
    assert torch.allclose(policy_decision(cfg3, a), 3 * a)
    with pytest.raises(ValueError):
        stored_decision(cfg3, a, None)


def test_config_checks():
    dataclasses.replace(LangConfig(), **H3)
    with pytest.raises(AssertionError):   # one summed action only
        dataclasses.replace(LangConfig(), decision_ticks=3, critic_chunk=12, critic_readout="mlp")
    with pytest.raises(AssertionError):   # the filter block must cover the decision
        dataclasses.replace(LangConfig(), decision_ticks=3, hj_hold_ticks=2)
    d = LangConfig().to_dict()
    d.pop("decision_ticks")
    assert LangConfig.from_dict(d).decision_ticks == 1   # saved before the field: one tick


def test_sampler_window(root):
    cfg3 = dataclasses.replace(LangConfig(), batch_size=8, **H3)
    s1 = LangSampler([root], LangConfig(), seed=0)
    s3 = LangSampler([root], cfg3, seed=0)
    src1, src3 = s1.sources[0], s3.sources[0]
    A = src3.rs.buf.arrays
    # rows: those whose t+3 is in the episode; the last 3 rows of each episode drop out (1-tick: the last one)
    assert src3.rows.size == src1.rows.size - 2 * len(set(src1.eid.tolist()))
    assert (src3.next == src3.rows + 3).all()
    assert (np.asarray(A["episode_id"])[src3.next] == src3.eid).all()
    b = s3.sample(16)
    S = src3.step
    for k in range(16):
        r = int(b["row"][k])
        assert int(b["next_row"][k]) == r + 3
        assert np.allclose(b["dtheta_seq"][k], np.asarray(S["dtheta"][r:r + 3]))
        assert np.allclose(b["dtheta"][k], b["dtheta_seq"][k, 0])
        col = int(b["rule_col"][k])
        assert np.allclose(b["h_seq"][k], np.asarray(A["h_rules"][r:r + 3, col]))
        assert b["h_seq"][k, 0] == b["h"][k]
        assert np.allclose(b["body_tokens_next"][k], np.asarray(S["body_tokens"][r + 3]))
        t = int(b["t_in_ep"][k])
        assert int(b["next_prev_row"][k]) == r + 3 - min(2, t + 3)
        assert np.allclose(b["scene_tokens_next"][k, 0], np.asarray(S["scene_tokens"][r + 3]))
    assert "dtheta_seq" not in s1.sample(4) and "h_seq" not in s1.sample(4)


def test_trainer_three_ticks_revised(tmp_path, root):
    torch.set_num_threads(4)
    tr = trainer(tmp_path / "run", root, **H3)
    ref = json.loads((tmp_path / "run" / "action_ref.json").read_text())
    assert ref["decision_ticks"] == 3 and ref["s_p_decision"] == pytest.approx(3 * ref["s_p"])
    assert bool(tr.critics.c1.ref_set) and bool(tr.critics_targ.c1.ref_set)
    out = tr.grad_step(log=True)
    assert np.isfinite(out["loss/q"]) and np.isfinite(out["loss/pi"]) and np.isfinite(out["critic/action_sens_rel"])
    p = tr.save_ckpt()
    tr2 = trainer(tmp_path / "run2", root, **H3)
    tr2.load_weights(p)
    assert torch.equal(tr2.critics.c1.act_ref, tr.critics.c1.act_ref)


def test_trainer_one_tick_unchanged(tmp_path, root):
    """decision_ticks 1 (the default) reads no window and writes no action_ref."""
    tr = trainer(tmp_path / "run", root)
    assert tr.action_ref is None and not (tmp_path / "run" / "action_ref.json").exists()
    assert np.isfinite(tr.grad_step()["loss/q"])


class _Flt:
    def __init__(self, q):
        self.q, self.margin, self.release_margin, self.hold_ticks, self._was_engaged = q, 0.0, 0.0, 3, False
        self.n_actor = 0
        self.scored = []

    def q_nom(self, enc, a, Jl, Ja):
        self.scored.append(a.cpu().numpy().copy())
        return torch.tensor(self.q)

    def actor_action(self, enc):
        self.n_actor += 1
        return torch.full((1, 4), 0.01 * self.n_actor)

    def track(self, engaged):
        self._was_engaged = engaged


def _controller(q, monkeypatch):
    from main.lang import rollout as R

    ro = R.LangRolloutController.__new__(R.LangRolloutController)
    steps = []
    ro.cfg = dataclasses.replace(LangConfig(), **H3)
    ro.flt, ro.cf_text_out, ro.filter_active, ro.mode = _Flt(q), None, True, "filtered"
    ro.trace = {"V": [], "V_t": [], "intervened": []}
    ro._n_physics, ro._n_act = 0, 4
    ro.env = SimpleNamespace(PHYSICS_FREQ=250, control_dt=0.04, step_dtheta=lambda a: steps.append(np.array(a)))
    ro.extractor = SimpleNamespace(delta_theta_max=np.ones(4))
    ro._policy_inputs_cached = lambda: ({"Jlin": torch.zeros(1), "Jang": torch.zeros(1)}, None)
    ro._anom = lambda ctx: torch.full((1, 4), 0.1)
    monkeypatch.setattr(R.LangRolloutController, "_anom_decision", lambda self, ctx: torch.full((1, 4), 0.3))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    return ro, steps


def test_filter_holds_the_actor_for_the_decision(monkeypatch):
    ro, steps = _controller(-0.2, monkeypatch)
    assert ro._tick_hook({"i": 0}) is True
    assert ro.flt.n_actor == 1 and len(steps) == 3              # one query per hook, held for 3 ticks
    assert all(np.allclose(s, 0.01) for s in steps)
    assert np.allclose(ro.flt.scored[0], 0.3)                   # scored the nominal's 3-tick displacement
    assert ro.trace["intervened"] == [True]


def test_filter_free_when_q_is_safe(monkeypatch):
    ro, steps = _controller(0.4, monkeypatch)
    assert ro._tick_hook({"i": 0}) is False
    assert ro.flt.n_actor == 1 and not steps                    # the actor is only scored (collapse monitors)
    # per hook: Q of the nominal's decision, of a stop and of the actor's decision, and both per-tick actions / bound
    assert len(ro.trace["V"]) == len(ro.trace["Q_stop"]) == len(ro.trace["Q_pi"]) == 1
    assert np.allclose(ro.flt.scored[1], 3 * 0.01) and np.allclose(ro.flt.scored[2], 0.0)
    assert np.allclose(ro.trace["a_pi"][0], 0.01) and np.allclose(ro.trace["a_nom"][0], 0.1)
    assert np.allclose(ro._tick_action, 0.1)                    # the stored command stays the per-tick one


def test_nominal_decision_is_the_sum_of_the_per_tick_commands():
    """A planner arm that tracks its plan with a constant lag: the filter's nominal decision equals the sum of the
    H per-tick commands the buffer would store (each = the target at the tick's end minus the live joint state)."""
    from main.lang.rollout import nominal_decision

    spt, H, v, lag = 10, 3, np.array([0.006, -0.002]), np.array([0.02, -0.01])
    pos = np.arange(200)[:, None] * v[None]                       # plan rows, constant speed
    q_at = lambda row: pos[row - 1] - lag                         # live state at the tick that starts at `row`
    for i in (10, 40, 150):
        cmds = [pos[min(i + k * spt + spt - 1, len(pos) - 1)] - q_at(i + k * spt) for k in range(H)]
        assert np.allclose(nominal_decision(pos, q_at(i), i, spt, H), np.sum(cmds, 0))
        assert np.allclose(nominal_decision(pos, q_at(i), i, spt, 1), cmds[0])        # H = 1: CrossSafe's a_nom
        # the bare endpoint misses (H - 1) lags
        assert np.allclose(pos[i + H * spt - 1] - q_at(i), np.sum(cmds, 0) - (H - 1) * lag)
    # a new plan from the live state (row 0, no lag yet): the endpoint itself
    assert np.allclose(nominal_decision(pos, pos[0], 0, spt, H), pos[H * spt - 1] - pos[0])
    # the plan ends inside the decision: the later ticks command the remaining lag to its last row
    T = len(pos)
    i = T - 1 - spt - 3                                           # tick 0 and tick 1 start inside the plan, tick 2 after it
    q0 = q_at(i)
    cmds = [pos[i + spt - 1] - q0, pos[T - 1] - q_at(i + spt), lag]
    assert np.allclose(nominal_decision(pos, q0, i, spt, H), np.sum(cmds, 0))
    # the hook also runs after the plan's last row (a gripper still moving): rows clip to the plan's end
    assert np.allclose(nominal_decision(pos, pos[T - 1] - lag, T + 5, spt, H), H * lag)


def test_release_test_counts_the_offset_from_the_stale_plan_once():
    from main.lang.rollout import nominal_decision

    pos = np.arange(200)[:, None] * np.array([0.006, -0.002])[None]
    q = pos[30] - np.array([0.05, -0.02])            # the actor held the arm back while the cursor stayed at row 40
    assert np.allclose(nominal_decision(pos, q, 40, 10, 3, stale=True), pos[69] - q)
    assert not np.allclose(nominal_decision(pos, q, 40, 10, 3), pos[69] - q)
