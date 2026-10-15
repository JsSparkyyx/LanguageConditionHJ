"""V6 collection (qwen_impl.md 6), CPU, no simulator: run_episodes with the kv / frames sidecars (lockstep,
rollback, flush, lines), the argument checks, open_extras / resume_point with the extra rings, the replay rules
(free fields, the pre-10-12 exemption) and the layout check."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from main.item_envs.item_buffer import ItemSidecarBuffer, LockstepItemWriter
from main.lang import collect as C
from main.lang.config import LangConfig
from main.lang.frames_buffer import FramesBuffer
from main.lang.kv_buffer import KvBuffer
from main.lang.legacy_pool import PRE_1012_BANK_HASH
from main.train.buffer import StepBuffer

from .fake_data import N_TOK_CAM, h_pattern, item_row, step_entry, template
from .test_collect_v2 import FakeLock, fake_env  # noqa: F401  (the fixture)
from .test_v6_storage import FR_DIMS, KV_DIMS, frames_row, kv_ident, kv_row

REPO = Path(__file__).resolve().parents[2]


def qcfg(tasks=("gap_place",), **kw):
    return dataclasses.replace(LangConfig(), task_choices=tasks, encoder="holobrain_qwen",
                               text_backend="holobrain_qwen", **kw)


class QwenFake:
    """The base rollout's row protocol (one capture late, the last with done) with the V6 rows in each entry."""

    PLAN: dict = {}
    built: list = []
    w = 0

    def __init__(self, cfg, seed, mode="collect", **kw):
        self.plan = QwenFake.PLAN.get(seed, {})
        QwenFake.built.append((seed, kw))
        self.cfg, self.seed, self.mode = cfg, seed, mode
        self.buf, self.episode_id = kw["buf"], kw["episode_id"]
        self.keep_kv, self.keep_frames = kw.get("keep_kv"), kw.get("keep_frames")
        self.filter_active, self.video, self.h_fn, self.writer = kw.get("filter_active"), None, None, None
        runner = NS(host=cfg.item_host, active_rule_id=None, _primary_rule_id=lambda task: None)
        self.env = NS(obstacle=object(), _item_runner=runner, task=NS(), close=lambda: None)

    def run(self):
        L = self.plan.get("ticks", 6)
        rec = template("gap_place")
        H = h_pattern(len(rec["rules"]), L, 0)
        rng = np.random.default_rng(self.seed)
        pending, status = None, "ok"
        for t in range(L):
            self.h_fn.row_sink.offer(item_row(rec, H[t], t, self.seed))
            if t == self.plan.get("fail_at"):
                status = "exception"
                break
            entry = step_entry(rng, QwenFake.w, t, self.seed % 1000, self.seed, float(H[t].min()), False)
            if self.keep_kv:
                entry["_kv"] = kv_row(QwenFake.w)
            if self.keep_frames:
                entry["_frames"] = frames_row(QwenFake.w)
            QwenFake.w += 1
            if pending is not None:
                self.buf.append(pending)
            pending = entry
        if pending is not None:
            pending["done"] = np.bool_(True)
            self.buf.append(pending)
        return {"status": status, "success": status == "ok", "n_ticks": L}


@pytest.fixture
def qwen_env(fake_env, monkeypatch):  # noqa: F811
    import main.lang.qwen_masks as qm
    import main.lang.rollout as rollout

    QwenFake.PLAN, QwenFake.built, QwenFake.w = {}, [], 1
    monkeypatch.setattr(rollout, "LangRolloutController", QwenFake)
    grid = []
    monkeypatch.setattr(qm, "use_qwen_grid", lambda ro: grid.append(ro.seed))
    monkeypatch.setattr(C, "item_n_tok", lambda cfg: N_TOK_CAM)    # the fake item rows' 4 cells per camera
    return QwenFake, grid


def rings(root, cfg, capacity=200, frames=True):
    buf, side = C.make_pair(root, cfg, capacity)
    kb = KvBuffer.for_step_buffer(buf, ckpt_id="c0", attn="sdpa", **KV_DIMS)
    fb = FramesBuffer.for_step_buffer(buf, **FR_DIMS) if frames else None
    return buf, side, kb, fb


def test_item_n_tok_follows_the_encoder():
    assert C.item_n_tok(qcfg()) == 99 and C.item_n_tok(LangConfig()) == 400


def test_run_episodes_qwen_rings_in_lockstep(tmp_path, qwen_env):
    fake, grid = qwen_env
    cfg, root = qcfg(), tmp_path / "shard"
    buf, side, kb, fb = rings(root, cfg)
    fake.PLAN = {5001: {"ticks": 9, "fail_at": 4}}
    lock = FakeLock(root)
    C.run_episodes(cfg, root, [("gap_place", "ur5-wsg", None)] * 3, 5000, buf, side, encoder="stub",
                   kins={"ur5-wsg": None}, with_masks=False, writer=lock, qwen=object(), kv_buf=kb, frames_buf=fb)
    lines = C.read_lines(root / "lang_episodes.jsonl")
    assert [d["status"] for d in lines] == ["ok", "exception", "ok"]
    assert [d["rows"] for d in lines] == [[0, 6], [6, 6], [6, 12]]     # the tick error rolled back all four rings
    assert all(d["encoder"] == "holobrain_qwen" for d in lines)
    assert [(kw["keep_kv"], kw["keep_frames"], kw["qwen"] is not None) for _, kw in fake.built] == [(True, True, True)] * 3
    assert grid == [5000, 5001, 5002]
    assert kb._guarded_by is lock and fb._guarded_by is lock
    k, f = KvBuffer(root / "kv_buffer"), FramesBuffer(root / "frames")
    assert k.n == f.n == 12 == json.loads((root / "buffer" / "header.json").read_text())["n"]
    pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")
    got = k.read(np.arange(12))
    assert [kv_ident(x) for x in got["k"]] == pos[:12, 0, 0].astype(int).tolist()
    assert f.read(np.arange(12))["rgb"][:, 0, 0, 0, 0].tolist() == pos[:12, 0, 0].astype(int).tolist()
    t_s = np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")
    assert np.array_equal(got["t_in_ep"], t_s[:12]) and got["episode_id"].tolist() == [5000] * 6 + [5002] * 6


def test_run_episodes_qwen_argument_checks(tmp_path, qwen_env):
    cfg, root = qcfg(), tmp_path / "shard"
    buf, side, kb, fb = rings(root, cfg, frames=False)
    with pytest.raises(ValueError, match="exactly under its config"):
        C.run_episodes(cfg, root, [], 5000, buf, side, encoder="stub", kv_buf=kb)
    with pytest.raises(ValueError, match="exactly under its config"):
        C.run_episodes(dataclasses.replace(LangConfig(), task_choices=("gap_place",)), root, [], 5000, buf, side,
                       encoder="stub", qwen=object(), kv_buf=kb)
    with pytest.raises(ValueError, match="kv_buf is required"):
        C.run_episodes(cfg, root, [], 5000, buf, side, encoder="stub", qwen=object())
    kb.append(dict(kv_row(0), episode_id=1, t_in_ep=0))                # a kv row the pair does not have
    with pytest.raises(RuntimeError, match="out of lockstep"):
        C.run_episodes(cfg, root, [("gap_place", "ur5-wsg", None)], 5000, buf, side, encoder="stub",
                       kins={"ur5-wsg": None}, qwen=object(), kv_buf=kb)
    assert len(buf) == len(side) == 0


def test_open_extras_and_resume(tmp_path, qwen_env):
    cfg, root = qcfg(), tmp_path / "shard"
    buf, side = C.make_pair(root, cfg, 20)
    ex = C.open_extras(root, buf, kv_dims={"ckpt_id": "c0", "attn": "sdpa"}, frames=True)
    assert set(ex) == {"_kv", "_frames"} and ex["_kv"].header["ckpt_id"] == "c0" and ex["_kv"].capacity == 20
    ex2 = C.open_extras(root, buf, kv_dims={"ckpt_id": "c0", "attn": "sdpa"}, frames=True)   # reopened r+
    assert ex2["_kv"].writable and ex2["_frames"].writable
    with pytest.raises(SystemExit, match="ckpt_id"):
        C.open_extras(root, buf, kv_dims={"ckpt_id": "other", "attn": "sdpa"})
    other = tmp_path / "other"
    b2, _ = C.make_pair(other, cfg, 20)
    b2.n = 3                                                              # rows but no kv_buffer/
    with pytest.raises(SystemExit, match="refusing to resume"):
        C.open_extras(other, b2, kv_dims={"ckpt_id": "c0", "attn": "sdpa"})


def test_resume_point_rolls_back_the_extra_rings(tmp_path, qwen_env):
    fake, _ = qwen_env
    cfg, root = qcfg(), tmp_path / "shard"
    buf, side, kb, fb = rings(root, cfg)
    C.run_episodes(cfg, root, [("gap_place", "ur5-wsg", None)] * 2, 5000, buf, side, encoder="stub",
                   kins={"ur5-wsg": None}, qwen=object(), kv_buf=kb, frames_buf=fb)
    lines = C.read_lines(root / "lang_episodes.jsonl")
    (root / "lang_episodes.jsonl").write_text(json.dumps(lines[0]) + "\n")   # the second line never written
    buf, side = StepBuffer(root / "buffer"), ItemSidecarBuffer(root / "item_buffer")
    kb2, fb2 = KvBuffer.reopen(root / "kv_buffer"), FramesBuffer.reopen(root / "frames")
    assert C.resume_point(root, buf, side, None, kb2, fb2) == (1, 2, 6)
    assert len(KvBuffer(root / "kv_buffer")) == len(FramesBuffer(root / "frames")) == 6
    kb2.arrays["t_in_ep"][5] = 9
    with pytest.raises(RuntimeError, match="do not hold that episode"):
        C.resume_point(root, buf, side, None, kb2, fb2)


def test_replay_free_fields_and_the_pre_1012_exemption():
    old = dataclasses.replace(LangConfig(), bank_hash=PRE_1012_BANK_HASH)
    new = qcfg(spawn_pool="pre_1012", qwen_attn="eager", tasks=LangConfig().task_choices)
    drift = C.config_drift(old, new)
    assert "encoder" not in drift and "qwen_attn" not in drift
    assert set(drift) == {"bank_hash", "spawn_pool"}
    assert C.legacy_replay(old, new)
    assert not C.legacy_replay(old, dataclasses.replace(new, spawn_pool="current"))
    assert not C.legacy_replay(LangConfig(), new)                        # a source from today's pool


def write_shard(root, lines, items):
    root.mkdir(parents=True)
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))
    (root / "item_buffer").mkdir()
    (root / "item_buffer" / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_id": e, "items": it}) + "\n" for e, it in items))


def test_replay_layout_mismatch(tmp_path):
    item = {"item_id": "obj_0", "category": "drink_box", "color": None, "kind": "rigid", "role": "candidate",
            "half_extents": [0.03, 0.03, 0.06], "obb_center_local": [0, 0, 0], "mass": 0.1569361611227606}
    pl = {"item_id": "obj_0", "cls": "on_path", "xyz": [0.1, 0.2, 0.8], "degraded_from": None, "lateral": None}
    t1 = {"episode_id": 7, "task": "selective_avoid", "scene_meta": {"placements": [pl], "corridors": []}}
    t5 = {"episode_id": 8, "task": "tray_transport",
          "scene_meta": {"episode_info": {"referent": "block_0", "items": [{"h": 0.1648, "color": "blue"}]}}}
    src = tmp_path / "src"
    write_shard(src, [t1, t5, {"skipped": True}], [(7, [item]), (8, [])])
    out = tmp_path / "out"
    pl2 = dict(pl, xyz=[0.1 + 3e-6, 0.2, 0.8], asset="068_boxdrink", model_id=0)   # within 1e-5; more keys
    write_shard(out, [dict(t1, scene_meta={"placements": [pl2]}), t5],
                [(7, [dict(item, color="red")]), (7, [item]), (8, [])])               # a stale line, then the real one
    assert C.replay_layout_mismatch(out, src) == []
    bad = tmp_path / "bad"
    t5b = dict(t5, scene_meta={"episode_info": {"referent": "block_0", "items": [{"h": 0.17, "color": "blue"}]}})
    write_shard(bad, [dict(t1, scene_meta={"placements": [dict(pl, xyz=[0.1, 0.21, 0.8])]}), t5b],
                [(7, [dict(item, category="bottle")]), (8, [])])
    diffs = C.replay_layout_mismatch(bad, src)
    assert any("placement obj_0 differs in ['xyz']" in d for d in diffs)
    assert any("item obj_0 differs in ['category']" in d for d in diffs)
    assert any("episode 8: episode_info differs" in d for d in diffs)


@pytest.mark.skipif(not (REPO / "data/lang/warmup/shard_g0/shard.json").exists(), reason="no recorded shard here")
def test_plan_shard_accepts_a_pre_1012_replay():
    args = NS(tasks="selective_avoid,push_zone,tray_transport,gap_place", replay=str(REPO / "data/lang/warmup/shard_g0"),
              encoder="holobrain_qwen", spawn_pool="pre_1012", qwen_attn="sdpa", frames=None, shard=None,
              n_shards=None, episodes_per_task=None, no_masks=False, embodiment="ur5-wsg", lang_version=1)
    cfg, meta, slots, _, _ = C.plan_shard(args, "warmup")
    assert cfg.qwen and cfg.spawn_pool == "pre_1012" and meta["frames"] and meta["encoder"] == "holobrain_qwen"
    assert meta["replay_bank_hash"] == PRE_1012_BANK_HASH and meta["kv"]["prefix_len"] == 325
    args.spawn_pool = "current"
    with pytest.raises(SystemExit, match="bank_hash"):
        C.plan_shard(args, "warmup")
