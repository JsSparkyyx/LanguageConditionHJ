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
