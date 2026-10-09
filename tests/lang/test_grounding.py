"""Review round 3, Part II: the grounding switches (V1 / V2 / V3), the seeing-vs-selecting analysis (E0b) and the
word-attention probe (E0c), on CPU with fake data and the stub text encoder."""

import dataclasses

import numpy as np
import pytest
import torch

from main.lang.config import LangConfig

from .fake_data import make_root
from .test_train import StubText, root, trainer  # noqa: F401  (root: a fixture)


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


@pytest.mark.parametrize("switch", ["text_out_norm", "mask_special_tokens", "oracle_ref_tag", "aux_target_loss"])
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
    if switch == "aux_target_loss":
        b = tr.sampler.sample(8)
        assert b["aux_target"].shape == (8, 2, 3) and b["aux_valid"].shape == (8, 2)
        assert hasattr(tr.policy_enc, "aux_head")
    out = tr.grad_step(log=True)
    assert np.isfinite(out["loss/q"])
    if switch == "aux_target_loss" and "aux/loss" in out:
        assert np.isfinite(out["aux/loss"]) and out["aux/center_err_cm_t"] >= 0
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


def test_grounding_summary_averages_seeds_and_decides(tmp_path):
    """E2 / E4 summary on synthetic rows: V1 ranks every referent pair right, V0 half: the text fix."""
    import gzip
    import importlib.util
    import json

    from main.lang import REPO

    spec = importlib.util.spec_from_file_location("gs", REPO / "scripts/lang/grounding_summary.py")
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)

    def rows_for(good: float, seed: int):
        rng = np.random.default_rng(seed)
        rows = []
        for ep in range(6):
            for st in range(10):
                right = rng.random() < good
                for harmful, obj in ((True, "a"), (False, "b")):
                    q = (-0.5 if harmful else 0.5) * (1 if right else -1)
                    rows.append({"src": 0, "i": ep * 100 + st, "episode_id": ep, "task": "selective_avoid",
                                 "rule": f"avoid|{obj}", "consequence": "avoid", "referent": obj,
                                 "fails_soon": harmful, "ttv": 0 if harmful else None, "censored": False,
                                 "eligible": {"train": True}, "q": {"train": q},
                                 "prior": {"train": 0.6 if harmful else 0.4}})
        return rows

    for v, good in (("v0", 0.5), ("v1", 1.0)):
        for seed in (0, 1):
            _fake_run(tmp_path / f"g1008_{v}_s{seed}", rows_for(good, seed),
                      cfg={"text_out_norm": True} if v == "v1" else None)
    assert gs.main(["--runs", str(tmp_path / "g1008_v*"), "--md", str(tmp_path / "s.md")]) == 0
    res = json.loads((tmp_path / "s.json").read_text())
    assert res["v1"]["referent/already_harmed/T1 + T7"]["mean"][0] == 1.0
    assert 0.3 < res["v0"]["referent/already_harmed/T1 + T7"]["mean"][0] < 0.7
    assert res["v0"]["referent/already_harmed/T1 + T7"]["prior"][0] == 1.0
    assert res["_e4"]["outcome"].startswith("a fixes it")


def test_v4_head_error_in_probe_and_eval(tmp_path):
    """V4: the head's centre error is reported on validation (training probe) and per row (eval_offline)."""
    from main.lang.eval_offline import Evaluator

    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    tr = trainer(tmp_path / "run", root, aux_target_loss=True)
    tr.val_sampler = tr.sampler
    pr = tr.aux_probe(n=8)
    assert all(np.isfinite(v) for v in pr.values())
    tr.grad_step()
    tr.save_ckpt()
    ev = Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText())
    M, rows = ev.evaluate(per_task=6, n_para=2, chunk_states=4)
    assert any(r.get("aux_err_cm") is not None for r in rows)


def test_v3_and_v4_parts_are_checkpointed(tmp_path, root):
    """The V3 tag and the V4 head travel with checkpoint.pt; a checkpoint without them is refused, not loaded
    with a zero tag."""
    from main.lang.network import LangPolicyEncoder
    from main.lang.train import load_policy

    tr = trainer(tmp_path / "run", root, oracle_ref_tag=True, aux_target_loss=True)
    with torch.no_grad():
        tr.policy_enc.e_tag.fill_(0.5)
    ck = tr.ckpt_dict()
    assert "e_tag" in ck and "aux_head" in ck
    pe = LangPolicyEncoder(tr.cfg)
    load_policy(pe, ck)
    assert torch.allclose(pe.e_tag, torch.full_like(pe.e_tag, 0.5))
    del ck["e_tag"]
    with pytest.raises(ValueError, match="re-export"):
        load_policy(LangPolicyEncoder(tr.cfg), ck)


def test_v4_loss_in_units_and_logged_ratio(tmp_path, root):
    """Round 4, item 3: the V4 loss is a Huber on (prediction, target) / aux_unit with delta 1 in those units, and
    aux/ratio = lambda * L_aux / L_critic. A 5 cm error on one axis is one unit: 0.5 per element, so 1/6 per
    (frame, xyz) mean."""
    tr = trainer(tmp_path / "run", root, aux_target_loss=True, aux_loss_weight=0.5)
    b = tr.sampler.sample(8)
    assert np.asarray(b["aux_valid"]).any()
    off = torch.tensor([0.05, 0.0, 0.0])
    tr.policy_enc.aux_predict = lambda enc: torch.as_tensor(b["aux_target"]) + off
    out = tr.grad_step(b, log=True)
    assert out["aux/loss"] == pytest.approx(0.5 / 3, rel=1e-4)
    assert out["aux/ratio"] == pytest.approx(0.5 * out["aux/loss"] / out["loss/q"], rel=1e-4)
    assert out["aux/center_err_cm_t"] == pytest.approx(5.0, rel=1e-4)


def test_v4b_attention_loss_reaches_the_query_attention(tmp_path, root):
    """V4b (round 4, item 3): q1's last-block scene attention, renormalized per frame (t and 0), weighted by the
    squared distance to the named item's centre in aux_unit; it trains the query attention and is logged."""
    tr = trainer(tmp_path / "run", root, aux_attention_loss=True, aux_attention_weight=0.3)
    assert LangConfig().aux_attention_loss is False
    assert tr.policy_enc.blocks[-1].keep_q_attn and not getattr(tr.policy_enc.blocks[0], "keep_q_attn", False)
    b = tr.sampler.sample(8)
    w0 = tr.policy_enc.blocks[-1].attn_qs.in_proj_weight.detach().clone()
    out = tr.grad_step(b, log=True)
    assert np.isfinite(out["aux/att_loss"]) and out["aux/att_loss"] >= 0
    assert out["aux/att_ratio"] == pytest.approx(0.3 * out["aux/att_loss"] / out["loss/q"], rel=1e-4)
    att = tr.policy_enc.blocks[-1].q_attn
    assert att.shape[-1] == np.asarray(b["scene_pos"]).shape[1] * np.asarray(b["scene_pos"]).shape[2]
    # the loss alone has a gradient on the query attention of the last block
    enc, *_ = tr.encode(b, *tr.text.lookup(b["sentence"], splits=["train"] * 8))
    a = tr.policy_enc.blocks[-1].q_attn[:, 0].reshape(8, 3, -1)
    a = a[:, 0] / a[:, 0].sum(-1, keepdim=True)
    d2 = (torch.as_tensor(np.asarray(b["scene_pos"]))[:, 0] ** 2).sum(-1)
    (a * d2).sum().backward()
    g = tr.policy_enc.blocks[-1].attn_qs.in_proj_weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    assert not torch.equal(w0, tr.policy_enc.blocks[-1].attn_qs.in_proj_weight.detach())


def test_v4b_refuses_without_queries_or_with_checkpointing():
    from main.lang.network import LangPolicyEncoder

    with pytest.raises(ValueError, match="readout queries"):
        LangPolicyEncoder(dataclasses.replace(LangConfig(), aux_attention_loss=True, lang_queries=False))


def _summary_module():
    import importlib.util

    from main.lang import REPO

    spec = importlib.util.spec_from_file_location("gs", REPO / "scripts/lang/grounding_summary.py")
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    return gs


def _fake_run(d, rows, cfg=None, ckpt_keys=("blocks",), step=6000, log=None):
    import gzip
    import json

    e = d / "eval_offline" / "grounding"
    e.mkdir(parents=True)
    with gzip.open(e / "rows.json.gz", "wt") as f:
        json.dump(rows, f)
    (e / "metrics.json").write_text(json.dumps({"step": step, "data": {}}))
    base = {"gamma": 0.95, "tf32_train": True, "task_choices": ["selective_avoid", "gap_place"],
            "embodiment_choices": ["ur5-wsg"]}
    (d / "lang_config.json").write_text(json.dumps({**base, **(cfg or {})}))
    torch.save({k: {} for k in ckpt_keys} | {"identity": {"git_head": "abc1234def"}}, d / "checkpoint_final.pt")
    if log:
        (d / "train_log.jsonl").write_text("\n".join(json.dumps(x) for x in log) + "\n")


def test_grounding_summary_refuses_incomparable_runs_and_untagged_v3(tmp_path):
    """Round 4, items 1-3: a run whose gamma / actor head / steps differ from V0's is left out, a V3 checkpoint
    without its tag is left out, the settings table is written, and a V4 whose aux ratio was below 0.05 does not
    count as tried."""
    import json

    gs = _summary_module()
    rows = [{"src": 0, "i": st, "episode_id": st % 4, "task": "selective_avoid", "rule": f"avoid|{o}",
             "consequence": "avoid", "referent": o, "fails_soon": o == "a", "ttv": 0 if o == "a" else None,
             "censored": False, "eligible": {"train": True}, "q": {"train": -0.5 if o == "a" else 0.5},
             "prior": {"train": 0.5}} for st in range(12) for o in ("a", "b")]
    _fake_run(tmp_path / "g_v0_s0", rows)
    _fake_run(tmp_path / "g_v1_s0", rows, cfg={"gamma": 0.9, "gamma_final": 0.9})
    _fake_run(tmp_path / "g_v2_s0", rows, cfg={"actor_hidden": 256})
    _fake_run(tmp_path / "g_v3_s0", rows, cfg={"oracle_ref_tag": True})                    # no e_tag
    _fake_run(tmp_path / "g_v4_s0", [dict(r, q={"train": 0.0}) for r in rows], cfg={"aux_target_loss": True},
              ckpt_keys=("blocks", "aux_head"), log=[{"step": 480, "aux/ratio": 0.01}, {"step": 6000, "aux/ratio": 0.02}])
    _fake_run(tmp_path / "g_v4b_s0", rows, cfg={"aux_attention_loss": True}, step=3000)
    assert gs.main(["--runs", str(tmp_path / "g_v*"), "--md", str(tmp_path / "s.md")]) == 0
    res = json.loads((tmp_path / "s.json").read_text())
    assert set(res["_refused"]) == {"g_v1_s0", "g_v2_s0", "g_v3_s0", "g_v4b_s0"}
    assert "e_tag" in " ".join(res["_refused"]["g_v3_s0"]) and "gamma" in " ".join(res["_refused"]["g_v1_s0"])
    assert "v1" not in res and "v3" not in res and "v0" in res and "v4" in res
    assert res["_settings"]["g_v4_s0"]["aux_ratio_500"] == pytest.approx(0.01)
    assert res["_e4"]["weak_aux"]["v4"] and "does not count" in res["_e4"]["outcome"]
    md = (tmp_path / "s.md").read_text()
    assert "## Run settings" in md and "abc1234def" in md


def test_eval_offline_records_parts_and_model_settings(tmp_path):
    """Round 4, items 1 and 7: the scored metrics name the checkpoint's parts and the model settings."""
    from main.lang.eval_offline import Evaluator, model_info, model_line

    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    tr = trainer(tmp_path / "run", root, oracle_ref_tag=True)
    tr.save_ckpt()
    ev = Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText())
    assert "e_tag" in ev.ckpt_parts
    mi = model_info(ev.cfg)
    assert mi["actor_hidden"] == LangConfig().actor_hidden and mi["gamma"] == pytest.approx(0.95)
    assert mi["recall_ticks"] == 14 and "actor_hidden" in model_line(ev.cfg)


def test_grounding_summary_flags_an_unused_oracle_and_the_train_gap(tmp_path):
    """Audit 2026-10-08: V3 whose zeroed tag changes no primary outcome is no ceiling (status inconclusive_oracle,
    not 'the bottleneck is not grounding'); training-episode scores and the in-batch accuracy are tabulated."""
    import gzip
    import json

    gs = _summary_module()

    def rows(right: float, seed: int):
        rng = np.random.default_rng(seed)
        out = []
        for st in range(40):
            ok_ = rng.random() < right
            for o in ("a", "b"):
                q = (-0.5 if o == "a" else 0.5) * (1 if ok_ else -1)
                out.append({"src": 0, "i": st, "episode_id": st % 5, "task": "gap_place", "rule": f"stay|{o}",
                            "consequence": "stay", "referent": o, "fails_soon": o == "a", "ttv": 0 if o == "a" else None,
                            "censored": False, "eligible": {"train": True}, "q": {"train": q}, "prior": {"train": 0.5}})
        return out

    log = [{"step": 500, "aux/ratio": 0.2, "lang/pair_rank_acc": 0.95}, {"step": 6000, "lang/pair_rank_acc": 0.96}]
    _fake_run(tmp_path / "g_v0_s0", rows(0.55, 0), log=log)
    _fake_run(tmp_path / "g_v4_s0", rows(0.6, 1), cfg={"aux_target_loss": True}, ckpt_keys=("blocks", "aux_head"), log=log)
    _fake_run(tmp_path / "g_v3_s0", rows(0.6, 2), cfg={"oracle_ref_tag": True}, ckpt_keys=("blocks", "e_tag"))
    for name, src in (("g_v3_s0/eval_offline/grounding_notag", rows(0.6, 2)),
                      ("g_v0_s0/eval_offline/grounding_train", rows(1.0, 3))):
        (tmp_path / name).mkdir(parents=True)
        with gzip.open(tmp_path / name / "rows.json.gz", "wt") as f:
            json.dump(src, f)
    assert gs.main(["--runs", str(tmp_path / "g_v*"), "--md", str(tmp_path / "s.md")]) == 0
    res = json.loads((tmp_path / "s.json").read_text())
    assert res["_extra"]["g_v3_s0"]["tag_use"]["flip_frac"] == 0.0
    assert res["_e4"]["status"] == "inconclusive_oracle", res["_e4"]
    assert res["_extra"]["g_v0_s0"]["train"][0] == 1.0 and res["_settings"]["g_v0_s0"]["train_pair_acc"] == pytest.approx(0.96)
    md = (tmp_path / "s.md").read_text()
    assert "Training versus validation" in md and "Does V3 read its tag" in md


def test_zone_rules_target_the_zone_not_the_pushed_box():
    """Audit 2026-10-08: a zone rule's subject is the pushed box but the sentence names the zone: V3 / V4 centres
    are the mat's (zone_exclude) or none (zone_include: the table frame sits at z = 0)."""
    from main.lang.data import named_centers, ref_centers

    T = np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1))
    T[0, 1, :3, 3] = [0.3, 0.1, 0.75]
    A = {"item_valid": np.ones((1, 1), bool), "item_pose": np.array([[[0.1, 0, 0.8, 1, 0, 0, 0]]], np.float32),
         "item_obb": np.zeros((1, 1, 9), np.float32), "region_T": T}
    rec = {"objects": [{"id": 0}]}
    ex = {"consequence": "zone_exclude", "subject": {"object": 0}, "reference": {"region": 1}}
    inc = {"consequence": "zone_include", "subject": {"object": 0}, "reference": {"region": 0}}
    stay = {"consequence": "stay", "subject": {"object": 0}}
    c, v = named_centers(A, rec, ex, [0])
    assert v[0] and np.allclose(c[0], [0.3, 0.1, 0.75])
    assert not named_centers(A, rec, inc, [0])[1].any() and not ref_centers(A, rec, ex, [0])[1].any()
    c, v = named_centers(A, rec, stay, [0])
    assert v[0] and np.allclose(c[0], [0.1, 0, 0.8])


def test_eval_zero_tag(tmp_path):
    from main.lang.eval_offline import Evaluator

    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    tr = trainer(tmp_path / "run", root, oracle_ref_tag=True)
    with torch.no_grad():
        tr.policy_enc.e_tag.fill_(0.3)
    tr.save_ckpt()
    ev = Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText(), zero_tag=True)
    assert float(ev.pe.e_tag.abs().sum()) == 0.0
    tr2 = trainer(tmp_path / "run2", root)
    tr2.save_ckpt()
    with pytest.raises(ValueError, match="zero-tag"):
        Evaluator(tmp_path / "run2", "checkpoint.pt", [root], device="cpu", batch=16, text_encoder=StubText(), zero_tag=True)
