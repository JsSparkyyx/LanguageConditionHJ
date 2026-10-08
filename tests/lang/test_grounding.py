"""Review round 3, Part II: the grounding switches (V1 / V2 / V3), the seeing-vs-selecting analysis (E0b) and the
word-attention probe (E0c), on CPU with fake data and the stub text encoder."""

import dataclasses

import numpy as np
import pytest
import torch

from main.lang.config import LangConfig

from .fake_data import make_root
from .test_train import StubText, trainer


def test_drop_special_masks_first_and_last_real_token():
    from main.lang.network import drop_special

    wm = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    out = drop_special(wm)
    assert out[0].tolist() == [False, True, True, False, False, False]
    assert out[1].tolist() == [True, True, False, False, False, False]   # nothing left: the mask is kept


def test_oracle_tag_weights_follow_the_named_object():
    from main.lang.network import oracle_tag_weights

    pos = torch.zeros(1, 2 * 3, 3)
    pos[0, :3, 0] = torch.tensor([0.0, 0.04, 0.5])         # frame 0: tokens at 0, 4 cm, 50 cm from the centre
    pos[0, 3:, 0] = torch.tensor([0.0, 0.04, 0.5])
    w = oracle_tag_weights(pos, torch.zeros(1, 2, 3), torch.tensor([[True, False]]), 0.04)
    assert w[0, 0] == pytest.approx(1.0) and w[0, 1] == pytest.approx(np.exp(-0.5), rel=1e-4) and w[0, 2] < 1e-10
    assert torch.all(w[0, 3:] == 0)                          # frame without a known centre: no tag


@pytest.mark.parametrize("switch", ["text_out_norm", "mask_special_tokens", "oracle_ref_tag"])
def test_switches_train_and_default_off(tmp_path, switch):
    """Each switch trains (one grad step on fake data); all default off, so the official runs are unchanged."""
    assert getattr(LangConfig(), switch) is False
    root = make_root(tmp_path / "d", episodes=3, capacity=300)
    tr = trainer(tmp_path / "run", root, **{switch: True})
    if switch == "oracle_ref_tag":
        b = tr.sampler.sample(8)
        assert b["ref_center"].shape == (8, 3, 3) and b["ref_center_valid"].shape == (8, 3)
        assert b["ref_center_next"].shape == (8, 3, 3)
        assert hasattr(tr.policy_enc, "e_tag") and float(tr.policy_enc.e_tag.abs().sum()) == 0.0
    out = tr.grad_step()
    assert np.isfinite(out["loss/q"])
    if switch == "text_out_norm":
        assert hasattr(tr.policy_enc.text_adapter, "norm_w")


def test_v3_checkpoint_scored_with_each_members_tag(tmp_path):
    from main.lang.eval_offline import Evaluator

    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    tr = trainer(tmp_path / "run", root, oracle_ref_tag=True)
    tr.grad_step()
    tr.save_ckpt()
    ev = Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText())
    M, rows = ev.evaluate(per_task=6, n_para=2, chunk_states=4)
    assert rows and all(np.isfinite(r["q"]["train"]) for r in rows if "train" in r["q"])
    k = ev.states(4)[0]
    with pytest.raises(ValueError, match="state key and rule column"):
        ev.q_batch([ev.sources[k[0]].read(k[1])], ["x"])


def test_rollout_refuses_the_oracle():
    from main.lang.rollout import LangRolloutController

    with pytest.raises(SystemExit, match="offline"):
        LangRolloutController(dataclasses.replace(LangConfig(), oracle_ref_tag=True), 0)


def test_seeing_vs_selecting_reading_rule():
    """A model whose Q follows the scene's most endangered object, whichever is named, has rho_any > rho_named."""
    from main.lang.eval_offline import seeing_vs_selecting

    rng = np.random.default_rng(0)
    rows = []
    for ep in range(6):
        for st in range(20):
            h_obj = rng.uniform(-1, 1, size=3)                   # three objects of one consequence
            for o in range(3):
                rows.append({"task": "selective_avoid", "consequence": "avoid", "eligible": {"train": True},
                             "q": {"train": float(h_obj.min() + rng.normal(0, 0.05))}, "h": float(h_obj[o]),
                             "h_any": float(h_obj.min()), "fs_any": bool(h_obj.min() < 0.2),
                             "src": 0, "episode_id": ep, "i": st})
    S = seeing_vs_selecting(rows, B=50)
    m = S["selective_avoid/all"]
    assert m["rho_any"] > 0.9 and m["rho_any"] - m["rho_named"] > 0.2 and m["any_minus_named_ci"][0] > 0


def test_word_attention_runs_on_cpu(tmp_path, monkeypatch):
    """E0c: every word-reading layer is probed, the critic blocks included, with per-head entropies."""
    import importlib.util
    import json

    from main.lang import REPO

    spec = importlib.util.spec_from_file_location("word_attention", REPO / "scripts/lang/word_attention.py")
    wa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wa)
    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    tr = trainer(tmp_path / "run", root)
    tr.save_ckpt()
    wa._TEXT_ENCODER = StubText()
    try:
        tok = __import__("transformers").AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased tokenizer not available offline")
    del tok
    assert wa.main(["--run", str(tmp_path / "run"), "--ckpt", "checkpoint.pt", "--val", str(root),
                    "--states-per-task", "4", "--batch", "8", "--device", "cpu", "--no-special-check"]) == 0
    out = json.loads((tmp_path / "run" / "eval_offline" / "checkpoint" / "word_attention.json").read_text())
    names = list(out["layers"]["pooled"])
    assert any("critic 2 block" in n for n in names) and any("encoder block" in n for n in names)
    row = out["layers"]["pooled"][names[0]]
    assert row["entropy_head_query"] <= row["entropy_averaged"] + 1e-6 and {"cls", "sep"} <= set(row)
