"""The pre-10-12 vocabulary switch (legacy_pool.install_vocab, LangConfig.vocab; V6 pre-run review 2026-10-15):
every reader of objects.yaml / paraphrases.yaml switches together and back, the old file gives the old T1 pool and
the old words, the bank hash follows the vocabulary, collect hashes of today's configs do not move, and a stored
record named with today's words is re-named to the vanilla runs' words at load (refresh_record). CPU only."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from main.lang import legacy_pool as lp
from main.lang.collect import collect_fields
from main.lang.config import REPO, LangConfig, bank_hash

DATA = REPO / "data"


@pytest.fixture
def old_vocab():
    lp.install_vocab("pre_1012")
    try:
        yield
    finally:
        lp.install_vocab("current")


def test_install_switches_every_reader_and_back(old_vocab):
    from main.item_envs import language, physics, rules

    old = yaml.safe_load((REPO / "tests/lang/fixtures/objects_pre_1012.yaml").read_text())
    cfg = physics.load_objects_cfg()
    assert cfg == old and language._objects_cfg() is cfg and lp.installed_vocab() == "pre_1012"
    assert cfg["assets"]["068_boxdrink"] == {"category": "drink_box", "ids": [0, 1, 2, 3]}
    assert "flask" in language.load_bank()["heldout_words"] and "flask" in rules._HELDOUT_WORDS
    assert language.model_words(cfg, "068_boxdrink", 1) == ("drink_box", ["drink box", "juice box", "carton"], True)
    assert language.model_words(cfg, "086_woodenblock", 2)[1] == ["wooden block", "block", "cube"]
    assert physics.load_objects_cfg(str(REPO / "envs/main/item_envs/configs/objects.yaml"))["assets"][
        "068_boxdrink"]["category"] == "bottle"                       # an explicit path still reads that file
    assert getattr(language.refresh_record, "vocab", None) == "pre_1012"
    rec = {"task": "clean_pour", "objects": [], "rules": [],
           "regions": [{"id": 0, "region_id": "bin_blue", "kind": "bin", "names": ["bin", "box"]}]}
    assert language.refresh_record(rec)["regions"][0]["names"] == ["bin", "box", "tub"]   # the old region words
    lp.install_vocab("pre_1012")                                     # idempotent
    lp.install_vocab("current")
    assert not hasattr(language.refresh_record, "vocab")
    cur = physics.load_objects_cfg()
    assert cur["assets"]["068_boxdrink"]["category"] == "bottle" and lp.installed_vocab() == "current"
    assert language._objects_cfg()["assets"]["068_boxdrink"]["category"] == "bottle"
    assert "plastic bottle" in language.load_bank()["heldout_words"] and rules._HELDOUT_WORDS is None
    with pytest.raises(ValueError, match="vocab"):
        lp.install_vocab("old")


def test_old_vocab_gives_the_legacy_pool(old_vocab):
    """Under the old objects.yaml today's spawn code draws from the pre-10-12 pool, the same pool the spawn_pool
    switch rebuilds from today's file (and the switch is then a no-op)."""
    from main.item_envs import physics, spawn

    old_pool = spawn.t1_pool_entries(physics.load_objects_cfg())
    lp.install_vocab("current")
    rebuilt = lp.legacy_entries(spawn.t1_pool_entries(physics.load_objects_cfg()))
    assert [e["asset"] for e in old_pool] == [e["asset"] for e in rebuilt]
    for a, b in zip(old_pool, rebuilt):
        assert (a.get("ids"), a.get("category")) == (b.get("ids"), b.get("category")), a["asset"]
        for k in ("mass_range", "side_xy", "height", "radius", "colored"):
            assert a.get(k) == b.get(k), (a["asset"], k)
    assert lp.legacy_entries(old_pool) == old_pool


def test_physics_fields_of_both_files_are_equal():
    """The 10-12 change of objects.yaml touched names only: the physics read from it is the same under both."""
    cur = yaml.safe_load(lp.vocab_files("current")[0].read_text())
    old = yaml.safe_load(lp.vocab_files("pre_1012")[0].read_text())
    for cat, e in old["categories"].items():
        c = cur["categories"].get(cat) or {}
        for k in ("mass_range", "friction_range", "dynamic_ratio", "colored"):
            assert e.get(k) == c.get(k), (cat, k)
    assert old.get("colors") == cur.get("colors") and old.get("t1_pool") == cur.get("t1_pool")


def test_bank_hash_follows_the_vocab():
    def md5_of(objects, bank):
        h = hashlib.md5()
        for p in (bank, REPO / "envs/main/item_envs/configs/splits.yaml", objects):
            h.update(Path(p).read_bytes())
        return h.hexdigest()[:12]

    assert bank_hash() == md5_of(*lp.vocab_files("current")) == LangConfig().bank_hash
    old = md5_of(*lp.vocab_files("pre_1012"))
    assert bank_hash("pre_1012") == old != bank_hash()
    assert LangConfig(vocab="pre_1012").bank_hash == old
    assert dataclasses.replace(LangConfig(), vocab="pre_1012").bank_hash == old         # no stale inherited hash
    assert dataclasses.replace(LangConfig(vocab="pre_1012"), vocab="current").bank_hash == bank_hash()
    saved = dict(LangConfig().to_dict(), bank_hash="693f7ef522ed")
    assert LangConfig.from_dict(saved).bank_hash == "693f7ef522ed"                     # a saved config keeps its own
    assert LangConfig.from_dict({k: v for k, v in LangConfig().to_dict().items() if k != "vocab"}).vocab == "current"
    with pytest.raises(AssertionError):
        LangConfig(vocab="old")


def test_collect_hash_of_todays_configs_does_not_move():
    assert "vocab" not in collect_fields(LangConfig())
    d = collect_fields(LangConfig(vocab="pre_1012"))
    assert d["vocab"] == "pre_1012" and d["bank_hash"] == bank_hash("pre_1012")


REPLAY = DATA / "lang_qwen/warmup/shard_g3/item_buffer/episodes/4000008.json"
SOURCE = DATA / "lang/warmup/shard_g3/item_buffer/episodes/4000008.json"


@pytest.mark.skipif(not (REPLAY.exists() and SOURCE.exists()), reason="the replayed T1 shard is not on this machine")
def test_a_replayed_record_gets_the_vanilla_words_back(old_vocab):
    """Episode 4000008: today's words call 001_bottle and 068_boxdrink id 3 both 'bottle' (4 of 11 rules sampleable);
    the old vocabulary names them as the vanilla runs' record does (8 of 11, the same rules)."""
    from main.item_envs import language

    src = json.loads(SOURCE.read_text())
    rec = language.refresh_record(json.loads(REPLAY.read_text()))
    words = lambda r: {o["item_id"]: [o["category"]] + list(o.get("synonyms") or []) for o in r["objects"]}  # noqa: E731
    assert words(rec) == words(src)
    flags = lambda r: {x["rule_id"]: bool(x.get("sampleable")) for x in r["rules"]}  # noqa: E731
    assert flags(rec) == flags(src) and sum(flags(rec).values()) == 8
    lp.install_vocab("current")
    today = language.refresh_record(json.loads(REPLAY.read_text()))
    assert sum(flags(today).values()) == 4


def test_checkpoint_vocab_reads_old_configs():
    assert lp.checkpoint_vocab({"vocab": "pre_1012"}) == "pre_1012"
    assert lp.checkpoint_vocab({"bank_hash": "693f7ef522ed"}) == "pre_1012"          # both vanilla runs' configs
    assert lp.checkpoint_vocab({"bank_hash": bank_hash()}) == "current"
    assert lp.checkpoint_vocab({}) == "current"
    for run in ("rounds_ur5_g095_mlp", "rounds_ur5_g098"):
        p = REPO / "runs" / "lang" / run / "lang_config.json"
        if p.exists():
            assert lp.checkpoint_vocab(json.loads(p.read_text())) == "pre_1012", run
