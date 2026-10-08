"""Offline probe on CPU: fake data, a 2-step checkpoint, the stub text encoder."""

import numpy as np

from main.lang.eval_offline import AXES, Evaluator, auroc, summarize, tables_md

from .fake_data import make_root
from .test_train import StubText, trainer


def test_auroc():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5


def test_evaluator_end_to_end(tmp_path):
    import torch

    torch.set_num_threads(4)
    root = make_root(tmp_path / "val", episodes=4, capacity=400)
    tr = trainer(tmp_path / "run", root)
    tr.grad_step()
    tr.save_ckpt()
    ev = Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText())
    M, rows = ev.evaluate(per_task=12, n_para=3, chunk_states=8)
    assert rows and all("train" in r["q"] for r in rows if r["eligible"]["train"])
    assert "train/symmetric/pooled" in M and 0.0 <= M["train/symmetric/pooled"]["rank_acc"] <= 1.0
    assert "shuffled/state/pooled" in M
    md = tables_md(M, "test")
    assert "| train |" in md


def test_hj_return_matches_the_closed_form():
    """E1 / D1: h = 1 until a violation k ticks ahead, then -1: V_k = (1 - g^k) + g^k (-1)."""
    from main.lang.eval_offline import hj_return, next_violation

    g, k = 0.9, 10
    h = np.array([1.0] * k + [-1.0] * 5)[:, None]
    y = hj_return(h, g)
    assert abs(y[0, 0] - (1 - 2 * g ** k)) < 1e-9 and y[k, 0] == -1.0
    nv = next_violation(np.array([[0.5], [0.2], [-0.1], [0.3]]))
    assert nv[:, 0].tolist() == [2, 1, 0, -1]


def _r(state, ep, task, cons, ref, fs, q, h, y=None, ttv=None, prior=None, sent="keep the block still",
       censored=False):
    el = {ax: ax == "train" for ax in AXES}
    return {"src": 0, "i": state, "episode_id": ep, "task": task, "consequence": cons, "referent": ref,
            "fails_soon": fs, "q": {"train": q, "base": q}, "h": h, "y": y, "ttv": ttv,
            "prior": {} if prior is None else {"train": prior}, "censored": censored,
            "eligible": el, "recall": fs, "sentences": {"train": sent}}


def test_summarize_label_checks_strata_and_intervals():
    """E1 / E2 / E4: rank by current h, agreement with the HJ return, text prior, time-to-violation and T5
    size-word strata, episode-bootstrap intervals."""
    rows = []
    for ep in range(6):
        for st in range(4):
            sid = ep * 10 + st
            rows.append(_r(sid, ep, "tray_transport", "upright", "tall", True, -0.2, 0.1, y=-0.3, ttv=st,
                           prior=0.6, sent="don't tip the tall block"))
            rows.append(_r(sid, ep, "tray_transport", "upright", "short", False, 0.4 if st else -0.5, 0.9, y=0.8,
                           prior=0.2, sent="don't tip the red block"))
    M = summarize(rows)
    p = M["train/symmetric/pooled"]
    assert p["n_pairs"] == 24 and p["n_episodes"] == 6
    assert abs(p["rank_acc"] - 18 / 24) < 1e-9 and p["rank_acc_ci"][0] <= p["rank_acc"] <= p["rank_acc_ci"][1]
    assert p["rank_acc_h"] == 1.0 and p["rank_acc_text"] == 1.0
    assert abs(p["rank_acc_mc"] - 18 / 24) < 1e-9          # Q agrees with y's order where Q ranks right
    assert M["train/symmetric/ttv/violated"]["rank_acc"] == 0.0    # st = 0: the benign Q was lower
    assert M["train/symmetric/ttv/1-7"]["rank_acc"] == 1.0
    assert M["train/symmetric/t5/size_word"]["n_pairs"] == 24
    assert M["train/state/pooled"]["auroc_ci"][0] <= M["train/state/pooled"]["auroc"]
    # C1: per-task headline cells; C2: y ranks every pair like the label here; C4: paired model - prior
    assert M["train/symmetric/task_state/tray_transport/violated"]["rank_acc"] == 0.0
    assert M["train/symmetric/task_state/tray_transport/not_yet"]["rank_acc"] == 1.0
    assert p["rank_acc_y"] == 1.0 and abs(p["model_minus_text"] - (18 / 24 - 1.0)) < 1e-9
    assert "train/symmetric/rc_pool" not in M          # T5 has no referent / consequence decisions pool
    assert "Label checks" in tables_md(M, "t")


def test_summarize_ties_censoring_and_rc_pool():
    """C5: a tie scores 0.5; C8: censored states leave the pairs and are counted; C6: T1 / T3 / T7 pool."""
    rows = [_r(0, 0, "gap_place", "stay", "a", True, 0.3, 0.2), _r(0, 0, "gap_place", "stay", "b", False, 0.3, 0.9),
            _r(1, 1, "gap_place", "stay", "a", True, 0.1, 0.2, censored=True),
            _r(1, 1, "gap_place", "stay", "b", False, 0.5, 0.9, censored=True)]
    M = summarize(rows)
    p = M["train/symmetric/pooled"]
    assert p["n_pairs"] == 1 and p["rank_acc"] == 0.5 and M["train/_censored_pairs"]["n"] == 1
    assert M["train/symmetric/rc_pool"]["n_pairs"] == 1


def test_evaluator_prior_and_same_state_base(tmp_path):
    """E2: the text prior fits on training roots; E3: the base sentence names another rule of the same state."""
    import torch

    from main.item_envs import language
    from main.lang.eval_offline import TextPrior

    torch.set_num_threads(4)
    root = make_root(tmp_path / "val", episodes=4, capacity=400)
    tr = trainer(tmp_path / "run", root)
    tr.grad_step()
    tr.save_ckpt()
    ev = Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText())
    ev.prior = TextPrior([root])
    M, rows = ev.evaluate(per_task=12, n_para=3, chunk_states=8)
    assert all(r.get("prior") is not None and r.get("y") is not None for r in rows)
    bank = language.load_bank()
    for r in rows[:20]:
        rec = ev.sources[r["src"]].rs.records[r["episode_id"]]
        others = {o["rule_id"] for o in rec["rules"] if o.get("sampleable") and o["rule_id"] != r["rule"]}
        if others:
            base = r["sentences"]["base"]
            assert any(base in language.sentences(o, rec, "train", bank) for o in rec["rules"] if o["rule_id"] in others)


def test_evaluator_scores_the_trained_embodiments(tmp_path):
    """By default the probe scores the validation episodes of the embodiments the run trained on."""
    import torch

    from main.lang.config import CANON_EMBODIMENTS

    torch.set_num_threads(4)
    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    emb = None
    for embs in (("ur5-wsg",), ("franka-panda",)):
        run = tmp_path / f"run_{embs[0]}"
        tr = trainer(run, root, embodiment_choices=embs)
        tr.save_ckpt()
        ev = Evaluator(run, "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText())
        assert ev.embodiments == embs
        emb = ev.sources[0].emb[0]
        assert bool(ev.states(per_task=50)) == (emb in embs)
    ev_all = Evaluator(run, "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText(), embodiments="all")
    assert ev_all.embodiments is None and ev_all.states(per_task=50) and emb in CANON_EMBODIMENTS
