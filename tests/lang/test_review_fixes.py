"""Review fixes of the V5 implementation (commit 533d5c7, plan-compliance F4 / F5 / F7), on CPU with fake shards, the
stub text encoder and the stub fusion stage of test_train_fusion.

- F4: word_attention.py reads a scene_fusion checkpoint's scene and words through the evaluator's scene_inputs (the
  fused frames, the fused words), not the stored tokens and cached words.
- F5: eval_offline refuses to write over an eval_offline/<out-name> that holds another set's scoring (other
  validation roots, other scored tasks / embodiments) unless --force; the same set is rewritten as before.
- F7: bench_fusion labels step_ms_320 as the fusion share only, and --train-steps times LangTrainer.grad_step with
  scene_fusion off and on.
"""

import dataclasses
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from main.lang import REPO
from main.lang import eval_offline as eo
from main.lang.config import LangConfig

from .fake_data import make_root
from .test_train import StubText, trainer
from .test_train_fusion import StubFusion, fusing

torch.set_num_threads(4)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "lang" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def need_tokenizer():
    try:
        __import__("transformers").AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased tokenizer not available offline")


# ---------------------------------------------------------------------------
# F4: word_attention on a scene_fusion checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fusion_words", [True, False])
def test_word_attention_reads_the_fused_scene_and_words(tmp_path, monkeypatch, fusion_words):
    """Every encoder forward of word_attention gets exactly the fused frames of its items (fused_scene's output) and,
    with fused words, frame t's fused words (else the cached words, as training read them)."""
    need_tokenizer()
    v2 = make_root(tmp_path / "v2", episodes=3, capacity=300, neck=True)
    tr = fusing(tmp_path / "run", v2, fusion_words=fusion_words)
    tr.grad_step()
    tr.save_ckpt()
    wa = load_script("word_attention")
    stub = StubFusion()
    monkeypatch.setattr(wa, "_TEXT_ENCODER", StubText())
    monkeypatch.setattr(wa, "_FUSION", stub)
    seen = []    # per encoder forward: what it read
    fused = []   # per fused_scene call: its input words and its output

    class Spy(eo.Evaluator):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)

            def hook(_m, args, kwargs):
                seen.append({"scene": args[5].clone(), "pos": args[6].clone(), "words": kwargs["words"].clone()})
            self.pe.register_forward_pre_hook(hook, with_kwargs=True)

        def fused_scene(self, keys, sentences, words, wmask):
            out = super().fused_scene(keys, sentences, words, wmask)
            fused.append({"words": words.clone(), "out": out})
            return out

    monkeypatch.setattr(eo, "Evaluator", Spy)
    assert wa.main(["--run", str(tmp_path / "run"), "--ckpt", "checkpoint.pt", "--val", str(v2),
                    "--states-per-task", "4", "--batch", "8", "--device", "cpu", "--no-special-check"]) == 0
    assert stub.frames and seen and len(seen) == len(fused)
    for s, f in zip(seen, fused):
        st, sp, wf = f["out"]
        assert torch.equal(s["scene"], st) and torch.equal(s["pos"], sp)
        assert torch.equal(s["words"], wf if fusion_words else f["words"])
        assert not torch.equal(wf, f["words"])   # the stub's fused words differ from the cached ones
    out = json.loads((tmp_path / "run" / "eval_offline" / "checkpoint" / "word_attention.json").read_text())
    assert out["scene_fusion"]["fused_words"] is fusion_words and out["scene_fusion"]["fusion_frames"]["fused"] > 0
    assert out["layers"]["pooled"]
    md = (tmp_path / "run" / "eval_offline" / "checkpoint" / "word_attention.md").read_text()
    assert "Scene fusion" in md


def test_scene_inputs_unfused_reads_the_stored_frames(tmp_path):
    """Without scene_fusion, scene_inputs returns the batch's stored tokens / positions and the words unchanged
    (q_batch's old path)."""
    root = make_root(tmp_path / "val", episodes=2, capacity=200)
    tr = trainer(tmp_path / "run", root)
    tr.save_ckpt()
    ev = eo.Evaluator(tmp_path / "run", "checkpoint.pt", [root], device="cpu", batch=8, text_encoder=StubText())
    keys = ev.states(4)[:3]
    reads = [ev.sources[j].read(i) for j, i in keys]
    b = {k: np.stack([r[k] for r in reads]) for k in reads[0]}
    words, wmask = ev.text.lookup(["keep the blue block still"] * 3)
    sc, sp, w = ev.scene_inputs(b, keys, ["keep the blue block still"] * 3, words, wmask)
    assert torch.equal(sc, torch.as_tensor(b["scene_tokens"]).float()) and w is words
    assert torch.equal(sp, torch.as_tensor(b["scene_pos"]).float())


# ---------------------------------------------------------------------------
# F5: eval_offline never replaces another set's scoring in place
# ---------------------------------------------------------------------------


def write_metrics(out, **data):
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"ckpt": "c.pt", "step": 1, "data": data, "metrics": {}}))


def test_out_dir_clash_rules(tmp_path, monkeypatch):
    a, b = tmp_path / "val" / "shard_g0", tmp_path / "val" / "shard_g1"
    a.mkdir(parents=True), b.mkdir(parents=True)
    out = tmp_path / "run" / "eval_offline" / "grounding"
    scored = {"tasks": ["gap_place", "selective_avoid"], "embodiments": ["ur5-wsg"]}
    assert eo.out_dir_clash(out, [a], scored) is None                       # nothing there yet
    out.mkdir(parents=True)
    assert eo.out_dir_clash(out, [a], scored) is None                       # a directory without metrics.json
    write_metrics(out, val_roots=[str(a), str(b)], **scored)
    monkeypatch.chdir(tmp_path)
    # the same set: other spelling (relative, other order) of the same roots, the same tasks / embodiments
    assert eo.out_dir_clash(out, ["val/shard_g1", "val/shard_g0"], dict(scored, tasks=scored["tasks"][::-1])) is None
    why = eo.out_dir_clash(out, [a], scored)
    assert "other validation roots" in why and "shard_g1" in why
    assert "other scored tasks" in eo.out_dir_clash(out, [a, b], dict(scored, tasks=["gap_place"]))
    assert "other scored embodiments" in eo.out_dir_clash(out, [a, b], dict(scored, embodiments=["franka-panda"]))
    write_metrics(out)                                                      # a record from before the fields
    assert eo.out_dir_clash(out, [a], scored) is None
    (out / "metrics.json").write_text("{trunc")
    assert "unreadable" in eo.out_dir_clash(out, [a], scored)


def test_scored_set_is_the_scored_rows_set(tmp_path):
    """scored_set (computed before scoring) is exactly data_info's tasks / embodiments of the rows evaluate writes,
    for every task / embodiment filter and axis choice."""
    ra = make_root(tmp_path / "a", episodes=2, capacity=200)                                # gap_place, ur5-wsg
    rb = make_root(tmp_path / "b", episodes=2, capacity=200, task="selective_avoid", eid0=2000)  # franka-panda
    tr = trainer(tmp_path / "run", ra)
    tr.save_ckpt()
    cases = (({}, eo.AXES), ({"tasks": ["selective_avoid"]}, eo.AXES), ({"embodiments": ["ur5-wsg"]}, ("train",)))
    for kw, axes in cases:
        ev = eo.Evaluator(tmp_path / "run", "checkpoint.pt", [ra, rb], device="cpu", batch=16, text_encoder=StubText(),
                          **kw)
        pred = ev.scored_set(6, axes)
        _, rows = ev.evaluate(per_task=6, n_para=1, chunk_states=8, axes=axes)
        d = eo.data_info(ev, rows, [ra, rb])
        assert pred == {"tasks": d["tasks"], "embodiments": d["embodiments"]} and rows, (kw, axes)


class CPUEvaluator(eo.Evaluator):
    def __init__(self, *a, **k):
        super().__init__(*a, device="cpu", text_encoder=StubText(), **k)


def test_eval_offline_main_refuses_another_set_unless_forced(tmp_path, monkeypatch):
    ra = make_root(tmp_path / "a", episodes=2, capacity=200)
    rb = make_root(tmp_path / "b", episodes=2, capacity=200, task="selective_avoid", eid0=2000)
    tr = trainer(tmp_path / "run", ra)
    tr.save_ckpt()
    monkeypatch.setattr(eo, "Evaluator", CPUEvaluator)
    evaluated = []
    real_evaluate = eo.Evaluator.evaluate
    monkeypatch.setattr(eo.Evaluator, "evaluate",
                        lambda self, *a, **k: evaluated.append(1) or real_evaluate(self, *a, **k))
    out = tmp_path / "run" / "eval_offline" / "grounding"

    def run(*val, extra=()):
        return eo.main(["--run", str(tmp_path / "run"), "--ckpt", "checkpoint.pt", "--val", *map(str, val),
                        "--states-per-task", "6", "--axes", "train", "--batch", "16", "--out-name", "grounding",
                        *extra])

    def recorded():
        d = json.loads((out / "metrics.json").read_text())["data"]
        return d["val_roots"], d["tasks"], d["embodiments"]

    assert run(ra, rb) == 0
    first = recorded()
    assert first[1] == ["gap_place", "selective_avoid"] and first[2] == ["franka-panda", "ur5-wsg"]
    assert run(ra, rb) == 0 and recorded() == first                        # the same set: rewritten, as before
    n = len(evaluated)
    with pytest.raises(SystemExit, match="another set: other validation roots"):
        run(ra)                                                             # another validation set
    with pytest.raises(SystemExit, match="other scored tasks"):
        run(ra, rb, extra=("--tasks", "gap_place"))                        # same roots, another task set
    with pytest.raises(SystemExit, match="other scored embodiments"):
        run(ra, rb, extra=("--embodiments", "ur5-wsg"))
    assert len(evaluated) == n and recorded() == first                     # refused before scoring, nothing written
    assert run(ra, extra=("--force",)) == 0
    assert recorded()[0] == [str(ra)] and recorded()[1] == ["gap_place"]


# ---------------------------------------------------------------------------
# F7: bench_fusion
# ---------------------------------------------------------------------------


@pytest.fixture
def bench():
    return load_script("bench_fusion")


def test_bench_labels_step_ms_320_as_the_fusion_share(bench):
    res = {"env": {"gpu": "g", "source": "synthetic", "frames": 4},
           "speed": [{"lowmem": True, "dtype": "fp32", "chunk": 8, "lengths": "all 12", "ms_per_frame": 1.0,
                      "step_ms_320": 320.0, "peak_mb": 10.0, "peak_mb_per_frame": 2.5}]}
    md = bench.markdown(res)
    assert "fusion share only (upper bound, before dedup" in md and "fusion ms / step (x 320) *" in md
    assert "fusion share only (upper bound, before dedup" in bench.STEP_MS_320_NOTE
    assert "--train-steps" in bench.__doc__ and "perf/step_ms" in bench.__doc__


def test_bench_train_steps_times_fusion_off_and_on(tmp_path, monkeypatch, bench):
    v2 = make_root(tmp_path / "v2", episodes=3, capacity=300, neck=True)
    stubs = []
    monkeypatch.setattr(bench, "DEVICE", "cpu")
    monkeypatch.setattr(bench, "TEXT_ENCODER", StubText())
    monkeypatch.setattr(bench.fz, "FusionStage", lambda *a, **k: stubs.append(StubFusion()) or stubs[-1])
    out = tmp_path / "bench" / "train.json"
    assert bench.main(["--data", str(v2), "--train-steps", "2", "--train-warmup", "1", "--train-set", "batch_size=8",
                       "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    t = res["train"]
    off, on = t["variants"]["V0' (scene_fusion off)"], t["variants"]["V5 (scene_fusion on)"]
    assert off["fusion_ms"] is None and off["fusion_frames"] == 0 and off["step_ms"] > 0
    assert on["fusion_ms"] > 0 and on["step_ms"] >= on["fusion_ms"] and on["fusion_frames"] > 0
    assert len(stubs) == 1 and stubs[0].frames and on["fusion_frames"] <= 5 * 8   # one stage; <= 5 frames x batch
    assert stubs[0].dtype == LangConfig().fusion_dtype and stubs[0].chunk == LangConfig().fusion_chunk
    assert t["batch_size"] == 8 and np.isclose(t["step_ratio"], on["step_ms"] / off["step_ms"])
    assert t["more_than_doubles"] == (t["step_ratio"] > 2) and "speed" not in res
    md = out.with_suffix(".md").read_text()
    assert "Training step, scene_fusion off vs on" in md and "V5 (scene_fusion on)" in md
    assert not list(out.parent.glob("_bench_train_*"))                    # the trainers' run dirs are removed


def test_bench_train_steps_argument_checks(tmp_path, monkeypatch, bench):
    monkeypatch.setattr(bench, "DEVICE", "cpu")
    with pytest.raises(SystemExit):
        bench.main(["--synthetic", "--train-steps", "2", "--out", str(tmp_path / "x.json")])   # needs --data
    with pytest.raises(SystemExit, match="times both"):
        bench.main(["--data", str(tmp_path), "--train-steps", "2", "--train-set", "scene_fusion=true",
                    "--out", str(tmp_path / "x.json")])
    assert dataclasses.replace(LangConfig(), embodiment_choices=("ur5-wsg",)) == bench.train_config([])
    assert bench.train_config(['fusion_dtype="bf16"']).fusion_dtype == "bf16"
    assert not os.path.exists(tmp_path / "x.json")
