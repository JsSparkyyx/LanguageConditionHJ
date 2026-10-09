"""V5 tools on CPU (fusion_impl.md section 6, package F): check_shards' version-2 checks and fusion parity (a stub
stage), fusion_probe's metrics and a stub-stage run on a fake version-2 shard, grounding_summary's E5 tables (run
names, reference, exemptions, the added held-equal fields, the decision) and its unchanged E2 output, scene_curve on
fake runs, and the grounding page's V5 section."""

import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from main.lang import REPO
from main.lang.data import stable_hash

from .fake_data import N_TOK_CAM, make_root, neck_ident

SCRIPTS = REPO / "scripts" / "lang"
E2 = REPO / "runs" / "lang" / "grounding_1008"
E2_COMMIT = "bca6bc5"   # the commit fusion_impl.md is anchored at: grounding_summary.py before the V5 changes
REAL_LEVELS = ((32, 40), (16, 20), (8, 10), (4, 5))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_t", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- check_shards --


def _finish(root: Path, version: int | None) -> Path:
    """A fake root as collection leaves it: episode lines with rows (and lang_version), shard.json."""
    ep = np.load(root / "item_buffer" / "episode_id.npy", mmap_mode="r")
    n = json.loads((root / "item_buffer" / "header.json").read_text())["n"]
    lines, r0 = [], 0
    for d in [json.loads(x) for x in (root / "lang_episodes.jsonl").read_text().splitlines()]:
        r1 = r0 + int((np.asarray(ep[:n]) == d["episode_id"]).sum())
        lines.append(dict(d, rows=[r0, r1], embodiment="ur5-wsg", **({"lang_version": version} if version else {})))
        r0 = r1
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (root / "shard.json").write_text(json.dumps({"kind": "validation", **({"lang_version": version} if version else {})}))
    return root


@pytest.fixture
def v2_root(tmp_path):
    return _finish(make_root(tmp_path / "v2" / "shard_g0", episodes=3, capacity=200, neck=True), 2)


def test_check_shards_version2_root_passes_and_each_fault_is_caught(tmp_path, v2_root):
    cs = _script("check_shards")
    assert cs.check(v2_root) == []

    def broken(name, edit):
        r = tmp_path / name / "shard_g0"
        shutil.copytree(v2_root, r)
        edit(r)
        return " | ".join(cs.check(r))

    def neck_row_off(r):                        # one neck row of another tick: a rollback on one side only
        a = np.load(r / "neck_buffer" / "t_in_ep.npy", mmap_mode="r+")
        a[17] += 1
        a.flush()

    def neck_header_off(r):
        h = json.loads((r / "neck_buffer" / "header.json").read_text())
        (r / "neck_buffer" / "header.json").write_text(json.dumps(dict(h, n=h["n"] - 1, head=h["head"] - 1)))

    def meta_v1(r):
        (r / "shard.json").write_text(json.dumps({"kind": "validation"}))

    def line_v1(r):
        lines = (r / "lang_episodes.jsonl").read_text().splitlines()
        d = json.loads(lines[1])
        del d["lang_version"]
        lines[1] = json.dumps(d)
        (r / "lang_episodes.jsonl").write_text("\n".join(lines) + "\n")

    def field_missing(r):
        (r / "neck_buffer" / "neck_3d.npy").unlink()

    assert "out of lockstep" in broken("a", neck_row_off) and "first row 17" in broken("a2", neck_row_off)
    assert "capacity / n / head" in broken("b", neck_header_off)
    assert "shard.json lang_version 1" in broken("c", meta_v1)
    assert "lack lang_version 2" in broken("d", line_v1)
    assert "neck_buffer:" in broken("e", field_missing)


def test_check_shards_version1_root_and_the_command_line(tmp_path):
    """A version-1 root passes as before (and may not claim version 2); `check_shards.py <roots>` keeps working,
    without torch, and exits 1 on a failed shard."""
    cs = _script("check_shards")
    d = tmp_path / "v1"
    roots = [_finish(make_root(d / f"shard_g{g}", episodes=2, capacity=120, eid0=1000 + 10 * g), None) for g in (0, 1)]
    assert cs.check(roots[0]) == []
    probe = ("import runpy, sys; sys.argv = ['check_shards.py'] + sys.argv[1:]\n"
             "try:\n    runpy.run_path(%r, run_name='__main__')\n"
             "except SystemExit as e:\n    assert 'torch' not in sys.modules, 'torch imported'\n    raise\n")
    run = lambda *a: subprocess.run([sys.executable, "-c", probe % str(SCRIPTS / "check_shards.py"), *map(str, a)],  # noqa: E731
                                    capture_output=True, text=True, cwd=REPO)
    p = run(d)
    assert p.returncode == 0 and "2 shards checked, 0 failed" in p.stdout, p.stdout + p.stderr
    p = run(roots[0], roots[1])
    assert p.returncode == 0 and "2 shards checked, 0 failed" in p.stdout
    (roots[1] / "shard.json").write_text(json.dumps({"lang_version": 2}))
    p = run(d, tmp_path / "nothing_here")
    assert p.returncode == 1 and "1 failed" in p.stdout and "no neck_buffer" in p.stdout
    assert "note" in p.stdout and "nothing_here" in p.stdout


def test_check_shards_replay_shard(tmp_path, v2_root):
    """A --replay shard: line i must hold slot i's recorded (task, seed), and a done shard's summary.json must
    record no replay_mismatch; shards without replay_of are not checked for it."""
    cs = _script("check_shards")
    lines = [json.loads(x) for x in (v2_root / "lang_episodes.jsonl").read_text().splitlines()]
    meta = json.loads((v2_root / "shard.json").read_text())
    slots = [[d["task"], "ur5-wsg", None, d["episode_id"]] for d in lines]
    (v2_root / "shard.json").write_text(json.dumps(dict(meta, replay_of="/src/shard_g0", slots=slots)))
    (v2_root / "summary.json").write_text(json.dumps({"complete": True, "replay_mismatch": []}))
    assert cs.check(v2_root) == []
    slots[1][3] += 7
    (v2_root / "shard.json").write_text(json.dumps(dict(meta, replay_of="/src/shard_g0", slots=slots)))
    assert any("replay slot 1" in b for b in cs.check(v2_root))
    slots[1][3] -= 7
    (v2_root / "shard.json").write_text(json.dumps(dict(meta, replay_of="/src/shard_g0", slots=slots)))
    (v2_root / "summary.json").write_text(json.dumps({"complete": True, "replay_mismatch": ["slot 2: skipped"]}))
    assert cs.check(v2_root) == ["summary.json replay_mismatch (1): slot 2: skipped"]
    (v2_root / "shard.json").write_text(json.dumps(meta))   # not a replay: replay_mismatch is not read
    assert cs.check(v2_root) == []


class ParityStage:
    """unfused(neck) of a fake shard: rebuilds the StepBuffer row the neck row was written with from its identity
    (fake_data.neck_row: (write index, t, episode) in neck_maps[:, 0, :3])."""

    device = torch.device("cpu")

    def unfused(self, neck):
        F = neck["neck_maps"].shape[0]
        tok = torch.zeros(F, 3 * N_TOK_CAM, 256)
        pos = torch.zeros(F, 3 * N_TOK_CAM, 3)
        for f in range(F):
            w, t, e = neck_ident(neck["neck_maps"][f].numpy())
            tok[f] = (w % 1000) / 1000.0
            pos[f, 0] = torch.tensor([w, t, e], dtype=torch.float32)
        return tok, pos


def test_check_shards_fusion_parity_with_a_stub_stage(tmp_path, v2_root, monkeypatch):
    cs = _script("check_shards")
    monkeypatch.setattr(cs, "_STAGE_FACTORY", lambda args: ParityStage())
    out = tmp_path / "parity.json"
    assert cs.main([str(v2_root), "--fusion-parity", "500", "--device", "cpu", "--json", str(out)]) == 0
    par = json.loads(out.read_text())["shards"][str(v2_root)]["parity"]
    assert par["rows"] == par["filled"] == 120 and par["tokens"]["row_rel_max"] < 1e-2
    m = np.load(v2_root / "neck_buffer" / "neck_maps.npy", mmap_mode="r+")   # a neck row of another transition
    m[40, 0, 0, 0] = 90
    m.flush()
    assert cs.main([str(v2_root), "--fusion-parity", "500", "--device", "cpu", "--json", str(out)]) == 1
    rep = json.loads(out.read_text())["shards"][str(v2_root)]
    assert rep["parity"]["tokens"]["worst_row"] == 40 and any("fusion parity: row 40" in b for b in rep["bad"])
    assert cs.check(v2_root) == []   # the alignment fields still agree: only the parity sees it


# ------------------------------------------------------------------------------------------- fusion_probe --


def test_fusion_probe_metric_functions():
    fp = _script("fusion_probe")
    delta, mask = np.ones(12), np.zeros(12, bool)
    delta[[2, 3]], mask[[2, 3]] = 4.0, True
    assert fp.enrichment(delta, mask) == pytest.approx(4.0)
    assert np.isnan(fp.enrichment(delta, np.zeros(12, bool))) and np.isnan(fp.enrichment(delta, np.ones(12, bool)))
    own, other = np.array([3.0, 1.0, 2.0, 5.0, 9.0]), np.array([1.0, 2.0, 1.0, 1.0, 0.0])
    assert fp.contrast(own, other, np.array([1, 1, 1, 1, 0], bool)) == pytest.approx(0.75)
    # the stride-32 mask projected to the 4 x 5 stride-64 grid by a 2 x 2 max, camera-major
    tm = np.zeros((3, 400), np.uint8)
    tm[2, 320 + 5 * 10 + 7] = 255                        # camera 2, stride 32, row 5, column 7 -> cell (2, 3)
    tm[2, 320 + 4 * 10 + 6] = 30                         # same 2 x 2 block
    tm[0, 15] = 255                                      # a stride-16 token only: no coarse cell from it
    cells = fp.cells_of(tm, REAL_LEVELS)
    assert cells.shape == (60,) and np.flatnonzero(cells).tolist() == [2 * 20 + 2 * 5 + 3]
    tm[1, 320 + 9] = 1                                   # camera 1, row 0, column 9 -> cell (0, 4)
    assert np.flatnonzero(fp.cells_of(tm, REAL_LEVELS)).tolist() == [20 + 4, 53]
    cell = np.full((2, 5, 60), 0.4 / 57)                 # heads, words, cells
    cell[:, :, [1, 2, 3]] = 0.2                          # 0.6 of every word's mass on cells 1-3
    on = np.zeros(60, bool)
    on[[1, 2, 3]] = True
    share, uni = fp.attention_share(cell, [2, 3, 9], on)  # word 9 is past the sentence: ignored
    assert share == pytest.approx(0.6) and uni == pytest.approx(3 / 60)
    assert np.isnan(fp.attention_share(cell, [], on)[0]) and np.isnan(fp.attention_share(cell, [1], ~on & False)[0])
    assert fp.bare_phrase("Don't touch the red mug.", ["the mug", "the red mug"]) == "the red mug"
    assert fp.bare_phrase("Never bump the cupboard", ["the cup"]) is None
    assert fp.bare_phrase("The Red Mug must stay put", ["the red mug"]) == "the red mug"


def _items_and_D(fp, own_gain=5.0):
    """Two avoid rules on objects a and b (disjoint masks) and one stay rule on a; each sentence lights its own
    object's tokens by own_gain over a background of 1."""
    N = 12
    ma, mb = np.zeros(N, bool), np.zeros(N, bool)
    ma[[0, 1]], mb[[6, 7]] = True, True
    items = [{"rule": "avoid|a", "consequence": "avoid", "object": "a", "sentence": "don't touch a", "phrase": "a",
              "mask": ma, "cells": np.array([1, 0, 0], bool), "obj_s": [3], "obj_p": [1]},
             {"rule": "avoid|b", "consequence": "avoid", "object": "b", "sentence": "don't touch b", "phrase": "b",
              "mask": mb, "cells": np.zeros(3, bool), "obj_s": [3], "obj_p": [1]},
             {"rule": "stay|a", "consequence": "stay", "object": "a", "sentence": "keep a still", "phrase": "a",
              "mask": ma, "cells": np.array([1, 0, 0], bool), "obj_s": [2], "obj_p": [1]}]
    texts = ["", "don't touch a", "a", "don't touch b", "b", "keep a still"]
    delta = np.ones((len(texts), N))
    delta[0] = 0.0
    for f, m in ((1, ma), (2, ma), (3, mb), (4, mb), (5, ma)):
        delta[f, m] = own_gain
    move = np.full((len(texts), 24), np.nan)
    move[:, :5] = 0.1
    move[:, 3] = 0.5
    cell = [np.full((2, 5, 3), 1 / 3) for _ in texts]
    for c in cell:
        c[:, :, 0] = 0.8
        c[:, :, 1:] = 0.1
    D = {"index": {s: i for i, s in enumerate(texts)}, "delta": delta, "delta_u": delta + 1.0, "shift": delta * 0.1,
         "shift_u": delta * 0.2, "move": move, "cell": {0: cell, 5: cell}}
    return items, D


def test_fusion_probe_state_metrics_summary_and_gate():
    fp = _script("fusion_probe")
    items, D = _items_and_D(fp)
    obs, pairs = fp.state_metrics(items, D)
    assert [o["rule"] for o in obs] == ["avoid|a", "avoid|b", "stay|a"]
    o = obs[0]
    assert o["enrich"] == pytest.approx(5.0) and o["enrich_phrase"] == pytest.approx(5.0)
    assert o["enrich_unfused"] == pytest.approx(6.0 / 2.0)
    assert o["word_move"] == pytest.approx((0.1 * 4 + 0.5) / 5) and o["word_move_obj"] == pytest.approx(0.5)
    assert o["att_share_l1"] == pytest.approx(0.8) and o["att_ratio_l1"] == pytest.approx(0.8 * 3)
    assert np.isnan(obs[1]["att_share_l1"])            # no coarse cell covers b
    assert o["pos_shift_named_cm"] == pytest.approx(0.5) and o["delta_empty_unfused_mean"] == pytest.approx(1.0)
    # pairs: same consequence, other object, in both orders (the stay rule has no partner)
    assert sorted((p["rule"], p["other"]) for p in pairs) == [("avoid|a", "avoid|b"), ("avoid|b", "avoid|a")]
    assert all(p["contrast"] == 1.0 and p["contrast_phrase"] == 1.0 for p in pairs)

    def episode(task, ep, gain):
        it, d = _items_and_D(fp, gain)
        o_, p_ = fp.state_metrics(it, d)
        where = {"task": task, "src": 0, "episode_id": ep, "t": 0}
        return [dict(where, **x) for x in o_], [dict(where, **x) for x in p_]

    obs, pairs = [], []
    for ep in range(4):
        for task, gain in (("selective_avoid", 3.0 + ep), ("gap_place", 1.2)):
            o_, p_ = episode(task, ep, gain)
            obs += o_
            pairs += p_
    S = fp.summarize(obs, pairs, B=200)
    assert set(S) == {"selective_avoid", "gap_place", "pooled"}
    assert S["selective_avoid"]["enrich"]["mean"] == pytest.approx(np.mean([3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]))
    assert S["selective_avoid"]["enrich"]["n_episodes"] == 4 and S["pooled"]["n_rules"] == 24
    assert S["selective_avoid"]["n_states"] == 4 and S["selective_avoid"]["n_pairs"] == 8
    g = fp.gate(S)
    assert g["task"] == "selective_avoid" and g["passed"] is True and g["clearly"] is True
    assert fp.gate(S, "gap_place")["passed"] is False            # enrichment 1.2 < 2
    S["selective_avoid"]["contrast"]["mean"] = 0.6
    assert fp.gate(S)["passed"] is False
    assert fp.gate({}, "selective_avoid")["passed"] is None
    picks = fp.pick_overlays(obs)
    assert len(picks) == 6 and obs[picks[0]]["task"] == "selective_avoid" and obs[picks[0]]["enrich"] == 6.0
    md = fp.markdown({"summary": S, "gate": fp.gate(S), "overlays": [],
                      "config": {"val": ["x"], "every": 25, "dtype": "fp32", "ckpt": "c"},
                      "n": {"states": 8, "rules": 24, "pairs": 16}})
    assert "PASSED" in md and "enrichment, bare phrase" in md


class StubText:
    """Words for the probe: zeros for the content-free sentence, seeded noise otherwise; 10 valid tokens."""

    def embed(self, sentences):
        e = torch.stack([torch.zeros(24, 256) if s == "" else
                         torch.randn(24, 256, generator=torch.Generator().manual_seed(stable_hash(s) % 2**31))
                         for s in sentences])
        m = torch.zeros(len(sentences), 24, dtype=torch.bool)
        m[:, :10] = True
        return e, m


class ProbeStage:
    """A deterministic stand-in for FusionStage on a fake shard: fuse adds c * (bump + 0.01) to the tokens, c = the
    sentence's mean |word| (0 for ""), bump = 1 on token (camera 0, t % 4), the fake item's mask; the fused words
    are the words + 0.1; the bi-attention is uniform."""

    device = torch.device("cpu")

    def __init__(self):
        self._caps = []

    def unfused(self, neck):
        F = neck["neck_maps"].shape[0]
        return torch.zeros(F, 3 * N_TOK_CAM, 256), torch.zeros(F, 3 * N_TOK_CAM, 3)

    def fuse(self, neck, words, wmask):
        F = words.shape[0]
        tok, pos = self.unfused(neck)
        for f in range(F):
            t = neck_ident(neck["neck_maps"][f].numpy())[1]
            c = float(words[f].abs().mean())
            tok[f] += 0.01 * c
            tok[f, t % N_TOK_CAM] += c
            pos[f, :, 2] = 0.001 * c
        for cap in self._caps:
            cap.cells = [torch.full((2, int(m.sum()), 3), 1 / 3) for m in wmask]
        return tok, pos, (words + 0.1) * wmask[..., None]

    @contextmanager
    def capture_bi_attention(self, layers=(0, 5)):
        stage = self

        class Cap:
            cells = []

            def per_frame(self, layer, fuse_call=None):
                return [None] * len(self.cells), self.cells
        cap = Cap()
        stage._caps.append(cap)
        try:
            yield cap
        finally:
            stage._caps.remove(cap)


def test_fusion_probe_runs_on_a_fake_version2_shard(tmp_path, monkeypatch):
    fp = _script("fusion_probe")
    try:
        tok = __import__("transformers").AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased tokenizer not available offline")
    root = _finish(make_root(tmp_path / "val" / "shard_g0", episodes=4, capacity=200, neck=True), 2)
    monkeypatch.setattr(fp, "_TEXT_ENCODER", StubText())
    monkeypatch.setattr(fp, "_TOKENIZER", tok)
    monkeypatch.setattr(fp, "_STAGE_FACTORY", lambda args: ProbeStage())
    out = tmp_path / "e4"
    assert fp.main(["--val", str(tmp_path / "val"), "--text-cache", "--out", str(out), "--tasks", "gap_place",
                    "--device", "cpu", "--boot", "100"]) == 0
    res = json.loads((out / "fusion_probe.json").read_text())
    S = res["summary"]["gap_place"]
    assert res["n"]["states"] == 8 and res["n"]["rules"] == 32   # ticks 0 and 25 of 4 episodes, 4 rules each
    assert S["enrich"]["n"] == 16 and S["enrich"]["mean"] == pytest.approx(101.0, rel=1e-3)   # block_0's rules
    assert S["contrast"]["n"] == 16 and 0.0 <= S["contrast"]["mean"] <= 1.0
    assert S["word_move"]["n"] == 32 and S["pos_shift_cm"]["mean"] > 0
    assert S["enrich_phrase"]["n"] == 16   # "the yellow block" etc. found in every sentence
    assert res["gate"]["passed"] is None   # no T1 here
    assert len(res["overlays"]) == 6 and all((out / o["png"]).exists() for o in res["overlays"])
    with gzip.open(out / "fusion_probe_rows.json.gz", "rt") as f:
        rows = json.load(f)
    assert {o["object"] for o in rows["obs"]} == {"block_0", "block_1"} and rows["pairs"]
    assert "token grid" in (out / "fusion_probe.md").read_text()


def test_fusion_probe_refuses_a_version1_shard(tmp_path, monkeypatch):
    fp = _script("fusion_probe")
    _finish(make_root(tmp_path / "val" / "shard_g0", episodes=1, capacity=60), None)
    monkeypatch.setattr(fp, "_STAGE_FACTORY", lambda args: ProbeStage())
    with pytest.raises(SystemExit, match="version-1"):
        fp.main(["--val", str(tmp_path / "val"), "--text-cache", "--out", str(tmp_path / "e4"), "--device", "cpu"])


# ------------------------------------------------------------------------------------- grounding_summary --


def _rows(acc: float, episodes=range(6), states: int = 20, src: int = 0) -> list[dict]:
    """Referent pairs (selective_avoid, already harmed): the first round(acc * states) states of every episode are
    ranked right, the rest wrong, so every episode has the same accuracy (degenerate bootstrap intervals)."""
    out = []
    for ep in episodes:
        for st in range(states):
            right = st < round(acc * states)
            for harmful, obj in ((True, "a"), (False, "b")):
                q = (-0.5 if harmful else 0.5) * (1 if right else -1)
                out.append({"src": src, "i": ep * 100 + st, "episode_id": ep, "task": "selective_avoid",
                            "rule": f"avoid|{obj}", "consequence": "avoid", "referent": obj, "fails_soon": harmful,
                            "ttv": 0 if harmful else None, "censored": False, "eligible": {"train": True},
                            "q": {"train": q}, "prior": {"train": 0.5}})
    return out


def _data_root(d: Path, version: int, n_ep: int = 10, embodiments=("ur5-wsg",), eid0: int = 1000) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    lines, r0 = [], 0
    for emb in embodiments:
        for k in range(n_ep):
            for task in ("selective_avoid", "gap_place"):
                lines.append({"episode_id": eid0 + len(lines), "task": task, "embodiment": emb, "rows": [r0, r0 + 50]})
                r0 += 50
    lines.append({"episode_id": eid0 + len(lines), "task": "gap_place", "embodiment": embodiments[0],
                  "rows": [r0, r0 + 1]})   # a one-row episode: never drawn
    (d / "lang_episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (d / "shard.json").write_text(json.dumps({"kind": "warmup", **({"lang_version": version} if version != 1 else {})}))
    return d


def _run(d: Path, rows, data_roots=(), cfg=None, step=6000, train_rows=None, train_roots=None, val_roots=("/v/a",)):
    e = d / "eval_offline" / "grounding"
    e.mkdir(parents=True)
    with gzip.open(e / "rows.json.gz", "wt") as f:
        json.dump(rows, f)
    data = {"val_roots": list(val_roots), "tasks": ["gap_place", "selective_avoid"], "embodiments": ["ur5-wsg"]}
    (e / "metrics.json").write_text(json.dumps({"step": step, "ckpt": "checkpoint_final.pt", "data": data}))
    base = {"gamma": 0.95, "tf32_train": True, "task_choices": ["selective_avoid", "gap_place"],
            "embodiment_choices": ["ur5-wsg"], "batch_size": 64}
    (d / "lang_config.json").write_text(json.dumps({**base, **(cfg or {})}))
    torch.save({"blocks": {}, "identity": {"git_head": "abc1234def", "data_roots": [str(x) for x in data_roots]}},
               d / "checkpoint_final.pt")
    if train_rows is not None:
        t = d / "eval_offline" / "grounding_train"
        t.mkdir(parents=True)
        with gzip.open(t / "rows.json.gz", "wt") as f:
            json.dump(train_rows, f)
        (t / "metrics.json").write_text(json.dumps({"step": step, "data": {"val_roots": [str(x) for x in train_roots]}}))


def test_grounding_summary_run_names_and_variant_codes():
    gs = _script("grounding_summary")
    parse = lambda n: gs.RUN_RE.search(n).groups()  # noqa: E731
    assert parse("g1008_v4b_s0") == ("v4b", "0") and parse("g1008_v0_s1") == ("v0", "1")
    assert parse("e5_v0p_s0") == ("v0p", "0") and parse("e5_v0ps_s1") == ("v0ps", "1")
    assert parse("e5_v5w_s0") == ("v5w", "0") and parse("e5_v5d2_s1") == ("v5d2", "1")
    assert parse("e5_v5_d2_s0") == ("v5", "0")      # an underscore makes "d2" a tag: refused as a duplicate
    assert gs.RUN_RE.search("g1008_cal2_v4") is None and gs.RUN_RE.search("g1008_d2_s0") is None
    assert [gs.variant_code(x) for x in ("V0'", "V0'-softmin", "V5-d2", "v5w", "V5")] == ["v0p", "v0ps", "v5d2", "v5w",
                                                                                         "v5"]
    assert gs.exemptions(["V5-d2:embodiments", "v0:train_data_version"]) == {"v5d2": {"embodiments"},
                                                                             "v0": {"train_data_version"}}
    with pytest.raises(SystemExit, match="variant"):
        gs.exemptions(["v5d2:scene_fusion"])        # differs by design: never compared, nothing to exempt


def test_grounding_summary_e5_table_reference_exemptions_and_decision(tmp_path):
    gs = _script("grounding_summary")
    v2 = [_data_root(tmp_path / "data" / f"v2_g{g}", 2, eid0=1000 * (g + 1)) for g in (0, 1)]
    v2d = [_data_root(tmp_path / "data" / "v2_d2", 2, embodiments=("ur5-wsg", "franka-panda"), eid0=9000)]
    v1 = [_data_root(tmp_path / "data" / "v1_g0", 1)]
    runs = tmp_path / "runs"
    mlp = {"critic_readout": "mlp"}
    for s in (0, 1):
        _run(runs / f"e5_v0p_s{s}", _rows(0.55), v2, mlp)
        _run(runs / f"e5_v0ps_s{s}", _rows(0.55), v2, {"critic_readout": "softmin"})
        _run(runs / f"e5_v5_s{s}", _rows(0.75), v2, {**mlp, "scene_fusion": True, "fusion_words": True})
        _run(runs / f"e5_v5w_s{s}", _rows(0.7), v2, {**mlp, "scene_fusion": True, "fusion_words": False})
        _run(runs / f"e5_v5d2_s{s}", _rows(0.8), v2d, {**mlp, "scene_fusion": True,
                                                       "embodiment_choices": ["ur5-wsg", "franka-panda"]})
    _run(runs / "e5_v0_s0", _rows(0.55), v1)                       # the report's V0, trained on version-1 data
    _run(runs / "e5_v5_tag_s0", _rows(0.75), v2, {**mlp, "scene_fusion": True})   # a second v5 seed 0
    _run(runs / "e5_v5_s3", _rows(0.75, episodes=range(5)), v2, {**mlp, "scene_fusion": True})   # other episodes
    md = tmp_path / "s.md"
    args = ["--runs", str(runs / "e5_*"), "--md", str(md), "--reference", "V0'"]
    assert gs.main(args) == 0
    res = json.loads(md.with_suffix(".json").read_text())
    ref = res["_refused"]
    assert set(ref) == {"e5_v0_s0", "e5_v5_tag_s0", "e5_v5_s3", "e5_v5d2_s0", "e5_v5d2_s1"}, ref
    assert "train_data_version 1 != the reference (V0')'s 2" in " ".join(ref["e5_v0_s0"])
    assert "same variant and seed as e5_v5_s0" in " ".join(ref["e5_v5_tag_s0"])
    assert "val_episodes" in " ".join(ref["e5_v5_s3"]) and "embodiments" in " ".join(ref["e5_v5d2_s0"])
    # scene_fusion / fusion_words / critic_readout differ by design: shown, never refused
    sw = res["_settings"]["e5_v5w_s0"]["switches"]
    assert "scene_fusion" in sw and "fusion_words off" in sw and "critic_readout mlp" in sw
    assert "critic_readout softmin" in res["_settings"]["e5_v0ps_s0"]["switches"]
    st = res["_settings"]["e5_v0p_s0"]
    assert st["train_data_version"] == "2" and st["train_episodes"] == 40
    assert st["samples_per_episode"] == pytest.approx(64 * 6000 / 40) and res["_reference"] == "e5_v0p_s0"
    assert res["_e5"]["status"] == "pending"    # V5-d2 refused: does V5 rise with scenes?
    assert res["_e5"]["v5_minus_v0p"][:3] == pytest.approx([0.2, 0.2, 0.2])
    text = md.read_text()
    assert "## E5: the fusion runs" in text and "| data version | val episodes |" in text and "## Decision (decision 5)" not in text
    # V5-d2 exempted on its embodiments (the label form), and the report's V0 on its data version
    assert gs.main(args + ["--exempt", "V5-d2:embodiments", "--exempt", "v0:train_data_version"]) == 0
    res = json.loads(md.with_suffix(".json").read_text())
    assert "e5_v5d2_s0" not in res["_refused"] and "e5_v0_s0" not in res["_refused"]
    assert res["v5d2"]["seeds"] == [0, 1] and res["_settings"]["e5_v5d2_s0"]["train_episodes"] == 40
    d = res["_e5"]
    assert d["status"] == "more_scenes", d
    assert d["paired_vs_v0p"]["v5d2"][0] == pytest.approx(0.25) and d["paired_vs_v0p"]["v0"][0] == pytest.approx(0.0)
    assert "e5_v5d2_s0 (step 6000)" in md.read_text() and "V5-d2 may differ in embodiments" in md.read_text()


@pytest.mark.parametrize("v5_acc, e4_passed, status", [(0.9, None, "adopt"), (0.55, True, "data_lever"),
                                                       (0.55, False, "e4_failed"), (0.55, None, "pending"),
                                                       (0.4, None, "below")])
def test_grounding_summary_e5_decision_rule(tmp_path, v5_acc, e4_passed, status):
    gs = _script("grounding_summary")
    runs = tmp_path / "runs"
    for s in (0, 1):
        _run(runs / f"e5_v0p_s{s}", _rows(0.55))
        _run(runs / f"e5_v5_s{s}", _rows(v5_acc), cfg={"scene_fusion": True})
    args = ["--runs", str(runs / "e5_*"), "--md", str(tmp_path / "s.md"), "--reference", "v0p"]
    if e4_passed is not None:
        (tmp_path / "e4.json").write_text(json.dumps({"gate": {"passed": e4_passed}}))
        args += ["--e4", str(tmp_path / "e4.json")]
    assert gs.main(args) == 0
    d = json.loads((tmp_path / "s.json").read_text())["_e5"]
    assert d["status"] == status, d
    assert d["e4_passed"] is e4_passed


def test_grounding_summary_e2_layout_without_e5_runs(tmp_path):
    """Without V0' / V5 runs the tables keep the E2 layout and JSON keys (the new fields are still checked)."""
    gs = _script("grounding_summary")
    _run(tmp_path / "g_v0_s0", _rows(0.55), [_data_root(tmp_path / "d2", 2)])
    _run(tmp_path / "g_v1_s0", _rows(0.6), [_data_root(tmp_path / "d1", 1)], {"text_out_norm": True})
    assert gs.main(["--runs", str(tmp_path / "g_v*"), "--md", str(tmp_path / "s.md")]) == 0
    res = json.loads((tmp_path / "s.json").read_text())
    assert "train_data_version 1 != V0's 2" in " ".join(res["_refused"]["g_v1_s0"])
    assert not any(k in res["_settings"]["g_v0_s0"] for k in gs.V2_KEYS) and "_e5" not in res and "_reference" not in res
    md = (tmp_path / "s.md").read_text()
    assert md.startswith("# Grounding runs: summary (review round 3, Part II E2 / E4)") and "data version" not in md
    assert "## Decision (decision 5)" in md


@pytest.mark.skipif(not (E2 / "summary.md").exists(), reason="the E2 runs are not in this checkout")
def test_grounding_summary_e2_output_is_unchanged(tmp_path):
    """The queued E2 summary lines run this script: on the E2 runs it writes what it wrote before the V5 changes
    (runs/lang/grounding_1008/summary.md / .json; if newer rows have landed since, the pre-V5 script's output on the
    same rows)."""
    md = tmp_path / "summary.md"
    cmd = ["--runs", str(E2 / "g1008_v*_s*"), "--out-name", "grounding", "--md"]
    assert _script("grounding_summary").main(cmd + [str(md)]) == 0
    same = lambda a, b: a.read_bytes() == b.read_bytes()  # noqa: E731
    if same(md, E2 / "summary.md") and same(md.with_suffix(".json"), E2 / "summary.json"):
        return
    old = subprocess.run(["git", "show", f"{E2_COMMIT}:scripts/lang/grounding_summary.py"], cwd=REPO,
                         capture_output=True, text=True)
    if old.returncode:
        pytest.skip(f"git show {E2_COMMIT} failed: {old.stderr.strip()}")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "gs_old.py").write_text(old.stdout.replace(
        "REPO = Path(__file__).resolve().parents[2]", f"REPO = Path({str(REPO)!r})"))
    spec = importlib.util.spec_from_file_location("gs_old", tmp_path / "old" / "gs_old.py")
    gs_old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs_old)
    ref = tmp_path / "old" / "summary.md"
    assert gs_old.main(cmd + [str(ref)]) == 0
    assert same(md, ref) and same(md.with_suffix(".json"), ref.with_suffix(".json"))


def test_training_episodes_matches_the_sampler_subset_order(tmp_path):
    """The training scenes of a run follow LangSampler._episode_subset: per (task, embodiment) the first N episodes
    by stable_hash(("e6", task, embodiment, eid)); one-row episodes never count; nested in N."""
    gs = _script("grounding_summary")
    roots = (str(_data_root(tmp_path / "a", 1, n_ep=6)), str(_data_root(tmp_path / "b", 1, n_ep=6, eid0=5000)))
    full = gs.training_episodes(roots, ("selective_avoid", "gap_place"), ("ur5-wsg",), 0)
    assert len(full["episodes"]) == 24 and full["data_version"] == "1"
    prev = set()
    for n in (1, 3, 5):
        sub = gs.training_episodes(roots, ("selective_avoid", "gap_place"), ("ur5-wsg",), n)
        assert sub["per_group"] == {"gap_place/ur5-wsg": n, "selective_avoid/ur5-wsg": n} and prev <= set(sub["episodes"])
        prev = set(sub["episodes"])
    lines = [json.loads(x) for r in roots for x in (Path(r) / "lang_episodes.jsonl").read_text().splitlines()]
    t1 = sorted((d["episode_id"] for d in lines if d["task"] == "selective_avoid"),
                key=lambda e: stable_hash(("e6", "selective_avoid", "ur5-wsg", e)))[:3]
    sub = gs.training_episodes(roots, ("selective_avoid", "gap_place"), ("ur5-wsg",), 3)
    assert sorted(e for _, e in sub["episodes"] if e in {d["episode_id"] for d in lines
                                                          if d["task"] == "selective_avoid"}) == sorted(t1)
    assert gs.training_episodes((str(tmp_path / "missing"),)) is None


# ------------------------------------------------------------------------------------------- scene_curve --


def test_scene_curve_points_training_scenes_and_refusals(tmp_path):
    sc = _script("scene_curve")
    gs = sc.gs
    ur5 = [_data_root(tmp_path / "data" / f"g{g}", 1, n_ep=10, eid0=1000 * (g + 1)) for g in (0, 1)]
    both = ur5 + [_data_root(tmp_path / "data" / "fr", 1, n_ep=20, embodiments=("franka-panda",), eid0=7000)]
    tasks = ("selective_avoid", "gap_place")
    runs = tmp_path / "runs"

    def train_rows(data_roots, embs, per_task, acc_own, acc_other):
        """Rows on every ur5 episode of the roots: right on the run's own training episodes with acc_own, else
        acc_other; src indexes the roots."""
        own = set(gs.training_episodes(tuple(str(x) for x in data_roots), tasks, embs, per_task)["episodes"])
        out = []
        for j, r in enumerate(ur5):
            for d in [json.loads(x) for x in (r / "lang_episodes.jsonl").read_text().splitlines()]:
                if d["rows"][1] - d["rows"][0] > 1:
                    acc = acc_own if (str(r), d["episode_id"]) in own else acc_other
                    out += _rows(acc, episodes=[d["episode_id"]], states=4, src=j)
        return out

    soft = {"critic_readout": "softmin", "recall_horizon_ticks": 7}
    for s in (0, 1):
        for n, acc in ((5, 0.55), (0, 0.6)):
            name = f"e6_n{n}_s{s}" if n else f"g_v0_s{s}"
            _run(runs / name, _rows(acc), ur5, {**soft, "train_episodes_per_task": n},
                 train_rows=train_rows(ur5, ("ur5-wsg",), n, 1.0, 0.25), train_roots=ur5)
        _run(runs / f"g_d2_s{s}", _rows(0.65), both, {"embodiment_choices": ["ur5-wsg", "franka-panda"]})
    _run(runs / "e6_n5_s7", _rows(0.5), ur5, {"train_episodes_per_task": 5, "critic_readout": "mlp"})
    out = tmp_path / "e6" / "curve"
    assert sc.main(["--runs", str(runs / "*"), "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["reference"] == "g_v0_s0" and set(res["refused"]) == {"e6_n5_s7"}
    assert "critic_readout mlp != g_v0_s0's softmin" in " ".join(res["refused"]["e6_n5_s7"])
    pts = res["points"]
    assert [p["train_episodes"] for p in pts] == [10, 40, 80] and [p["label"] for p in pts] == [
        "5 / task (ur5-wsg)", "all (ur5-wsg)", "all (ur5-wsg, franka-panda)"]
    assert pts[0]["new"]["mean"][0] == pytest.approx(0.55) and pts[0]["seeds"] == ["0", "1"]
    # training scenes: only the run's own episodes (10 of the 40 ur5 ones for the subsample), all right
    assert pts[0]["train"]["mean"][0] == pytest.approx(1.0) and pts[0]["train"]["n_episodes"] == 10
    assert pts[1]["train"]["n_episodes"] == 40 and pts[2]["train"]["n_pairs"] == 0
    assert pts[0]["samples_per_episode"] == pytest.approx(64 * 6000 / 10)
    assert out.with_suffix(".png").exists() and "| 10 | 5 / task (ur5-wsg) |" in out.with_suffix(".md").read_text()


# --------------------------------------------------------------------------------------------- the page --


def test_grounding_page_v5_section_only_from_existing_outputs(tmp_path):
    gp = _script("grounding_page")
    assert gp.section_v5(None, None, None) == ""
    fp = _script("fusion_probe")
    obs, pairs = [], []
    for ep in range(3):
        items, D = _items_and_D(fp, 3.0)
        o_, p_ = fp.state_metrics(items, D)
        where = {"task": "selective_avoid", "src": 0, "episode_id": ep, "t": 0}
        obs += [dict(where, **x) for x in o_]
        pairs += [dict(where, **x) for x in p_]
    S = fp.summarize(obs, pairs, B=50)
    e4 = {"summary": S, "gate": fp.gate(S), "config": {"every": 25}, "n": {"states": 3, "rules": 9}, "overlays": []}
    html = gp.section_v5(e4, None, None, n=8)
    assert '<h2 id="v5-h">8.' in html and "Agreed test" in html and "passed" in html and "3.00" in html
    e6 = {"bar": 0.85, "points": [
        {"label": "5 / task (ur5-wsg)", "train_episodes": 10, "train_episodes_runs": [10], "embodiments": ["ur5-wsg"],
         "samples_per_episode": 38400.0, "new": {"mean": [0.55, 0.5, 0.6], "per_seed": {"0": 0.55}, "n_pairs": 9},
         "train": {"mean": [0.97, 0.95, 0.99], "per_seed": {"0": 0.97}, "n_pairs": 9}},
        {"label": "all (ur5-wsg, franka-panda)", "train_episodes": 119, "train_episodes_runs": [119],
         "embodiments": ["ur5-wsg", "franka-panda"], "samples_per_episode": None,
         "new": {"mean": [0.6, 0.55, 0.66], "per_seed": {"0": 0.6}, "n_pairs": 9},
         "train": {"mean": [float("nan")] * 3, "per_seed": {}, "n_pairs": 0}}]}
    html = gp.section_v5(None, None, e6)
    assert html.count("<circle") == 3 + 1 and 'class="h1"' in html and "38,400" in html and "<svg" in html
    e5 = {"v0p": {"referent/already_harmed/T1 + T7": {"mean": [0.55, 0.5, 0.6], "per_seed": {"0": 0.55},
                                                      "n_pairs": 120, "n_episodes": 6}},
          "v5": {"referent/already_harmed/T1 + T7": {"mean": [0.75, 0.7, 0.8], "per_seed": {"0": 0.75}}},
          "_settings": {"e5_v5_s0": {"samples_per_episode": 9600.0, "steps": 6000},
                        "e5_v0p_s0": {"samples_per_episode": 9600.0, "steps": 6000}},
          "_e5": {"paired_vs_v0p": {"v5": [0.2, 0.15, 0.25, 120]}, "bar": 0.85, "e4_passed": True,
                  "status": "more_scenes", "outcome": "collect more scenes"}}
    html = gp.section_v5(None, e5, None)
    assert "+0.20 [+0.15, +0.25]" in html and "9,600" in html and "collect more scenes" in html and "0.85" in html
