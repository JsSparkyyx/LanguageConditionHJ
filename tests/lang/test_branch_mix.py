"""Branch rows in the batch (LangConfig.branch_cont_frac / branch_decision_frac; T8_BRANCH_PLAN.md): context rows
and held-out episodes never train, row groups, the fixed shares of a batch."""

import dataclasses
import json

import numpy as np
import pytest

from main.lang.config import LangConfig
from main.lang.data import LangSampler

from .fake_data import make_root

N_CTX = 3


@pytest.fixture
def roots(tmp_path):
    """One normal root (4 episodes) and one branch root: 6 episodes of 20 rows, each a branch with 3 context rows;
    the last one held out."""
    normal = make_root(tmp_path / "normal", episodes=4, capacity=400)
    branch = make_root(tmp_path / "branch", episodes=6, capacity=400, lengths=[20] * 6, eid0=5000)
    p = branch / "lang_episodes.jsonl"
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    for k, ln in enumerate(lines):
        ln["n_ctx"] = N_CTX
        ln["branch"] = {"state_id": f"s/{k // 2}", "decision": "plan" if k % 2 == 0 else "stop", "held_out": k == 5}
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return normal, branch


def test_context_rows_and_held_out_episodes_never_train(roots):
    normal, branch = roots
    s = LangSampler([branch], LangConfig(), seed=0).sources[0]
    assert s.branch.all() and (s.t >= N_CTX).all() and (s.t_branch == s.t - N_CTX).all()
    assert 5005 not in set(s.eid.tolist()) and set(s.eid.tolist()) == {5000, 5001, 5002, 5003, 5004}
    # one-tick decision: the branch state row is the decision row; 3 ticks: the first 3 rows from the state
    assert (s.group[s.t_branch == 0] == 2).all() and (s.group[s.t_branch > 0] == 1).all()
    s3 = LangSampler([branch], dataclasses.replace(LangConfig(), decision_ticks=3, critic_injection="revised",
                                                   critic_readout="mlp"), seed=0).sources[0]
    assert (s3.group[s3.t_branch < 3] == 2).all() and (s3.group[s3.t_branch >= 3] == 1).all()
    n = LangSampler([normal], LangConfig(), seed=0).sources[0]
    assert not n.branch.any() and (n.group == 0).all() and (n.t_branch == -1).all()


def test_batch_shares(roots):
    cfg = dataclasses.replace(LangConfig(), batch_size=64, branch_cont_frac=0.25, branch_decision_frac=0.25)
    smp = LangSampler(list(roots), cfg, seed=0)
    for _ in range(5):
        b = smp.sample(64)
        g = np.asarray(b["group"])
        assert [(g == k).sum() for k in (0, 1, 2)] == [32, 16, 16]
        # pairs stay inside one group: both members are the same transition
        for pid in np.unique(b["pair"][b["pair"] >= 0]):
            rows = np.nonzero(b["pair"] == pid)[0]
            assert rows.size == 2 and g[rows[0]] == g[rows[1]] and b["row"][rows[0]] == b["row"][rows[1]]
        assert (np.asarray(b["t_in_ep"])[g > 0] >= N_CTX).all()


def test_no_shares_is_the_old_draw(roots):
    normal, _ = roots
    a = LangSampler([normal], LangConfig(), seed=3).plan(64)
    b = LangSampler([normal], dataclasses.replace(LangConfig(), branch_cont_frac=0.0), seed=3).plan(64)
    assert a == b
    assert "group" in LangSampler([normal], LangConfig(), seed=3).sample(8)


def test_shares_without_branch_data_fail(roots):
    normal, _ = roots
    cfg = dataclasses.replace(LangConfig(), branch_decision_frac=0.25)
    with pytest.raises(RuntimeError):
        LangSampler([normal], cfg, seed=0).sample(16)
    with pytest.raises(AssertionError):
        dataclasses.replace(LangConfig(), branch_cont_frac=0.6, branch_decision_frac=0.5)
