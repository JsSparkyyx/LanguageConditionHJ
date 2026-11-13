"""T8 tray_slide in the language filter's pipeline (2026-11-13): the appended task index, the per-episode behaviour
mix, the behaviour-driven designations, the sentences, and the collector's per-episode risk override."""

import dataclasses
from collections import Counter
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from main.lang import collect as C
from main.lang import config as CF
from main.lang import scene_roles as SR
from main.lang.config import LangConfig

SEVEN = ("selective_avoid", "granular_pour", "push_zone", "granular_carry", "tray_transport", "clean_pour", "gap_place")


# ------------------------------------------------------------------------------------------ task lists --
def test_canon_tasks_appended_only():
    assert CF.CANON_TASKS[:7] == SEVEN == CF.CANON_TASKS_7      # stored task ids and eval seeds keep their index
    assert CF.CANON_TASKS[7] == "tray_slide" and CF.TASK_SHORT["tray_slide"] == "T8"
    assert CF.TRAIN_TASKS == ("selective_avoid", "push_zone", "tray_transport", "clean_pour", "gap_place")
    assert LangConfig().task_choices == CF.TRAIN_TASKS                 # the default is unchanged
    # the old tasks' eval seeds are the same numbers as before; T8's come after T7's
    assert CF.eval_seed("gap_place", "ur5-wsg", 3) == 1_500_000_000 + 6 * 10_000_000 + 3
    assert CF.eval_seed("tray_slide", "ur5-wsg", 999, 9) == 1_570_009_999 <= CF.INT32_MAX


def test_task_list_specs():
    assert CF.task_list("all") == CF.TRAIN_TASKS
    assert CF.task_list("tray_slide") == ("tray_slide",)
    assert CF.task_list("all,tray_slide") == CF.TRAIN_TASKS + ("tray_slide",)       # the five plus T8
    assert CF.task_list("gap_place,all") == ("gap_place",) + tuple(t for t in CF.TRAIN_TASKS if t != "gap_place")
    with pytest.raises(ValueError, match="unknown task"):
        CF.task_list("tray_slid")


def test_train_flags_pick_the_task_lists(tmp_path):
    import argparse

    from main.lang.train import add_common, config_from_args, trained_subsets

    p = argparse.ArgumentParser()
    add_common(p)
    for spec, want in (("all", CF.TRAIN_TASKS), ("tray_slide", ("tray_slide",)),
                       ("all,tray_slide", CF.TRAIN_TASKS + ("tray_slide",))):
        cfg = config_from_args(p.parse_args(["--data", "d", "--run", str(tmp_path / spec), "--tasks", spec]))
        assert cfg.task_choices == want and trained_subsets(cfg)[0] == want


def test_heldout_pool_refused_until_the_task_has_one():
    with pytest.raises(NotImplementedError, match="held-out"):
        CF.episode_risk(dataclasses.replace(LangConfig(), spawn_pool="heldout"), "tray_slide", 5)
    assert CF.episode_risk(dataclasses.replace(LangConfig(), spawn_pool="heldout"), "gap_place", 5) is None


def test_task_runs_on_registry():
    assert CF.task_runs_on("tray_slide", "ur5-wsg") and not CF.task_runs_on("tray_slide", "franka-panda")
    assert all(CF.task_runs_on(t, e) for t in SEVEN for e in CF.CANON_EMBODIMENTS)


def test_schedules_skip_tray_slide_on_franka():
    from main.lang.rounds import RoundSchedule

    s = RoundSchedule(("gap_place", "tray_slide"), CF.CANON_EMBODIMENTS, CF.CANON_T1_HOSTS)
    assert s.combos == [("gap_place", "ur5-wsg"), ("gap_place", "franka-panda"), ("tray_slide", "ur5-wsg")]
    with pytest.raises(ValueError, match="no task"):
        RoundSchedule(("tray_slide",), ("franka-panda",), CF.CANON_T1_HOSTS)
    assert {t for t, _, _ in C.shard_schedule(("gap_place", "tray_slide"), "franka-panda", 0, 1, 2, 0)} == {"gap_place"}
    assert Counter(t for t, _, _ in C.shard_schedule(("gap_place", "tray_slide"), "ur5-wsg", 0, 1, 2, 0)) == {
        "gap_place": 2, "tray_slide": 2}


def test_h_code_of_old_states_unchanged():
    """code_identity's first hash stays over the seven tasks (states that recorded only it still resume), and a
    seven-task run's h_code_v2 is the hash of its own list, whatever CANON_TASKS holds."""
    from main.lang.train import code_identity, trained_subsets

    ident = code_identity(SEVEN, [])
    assert ident["h_code"] == C.h_code_hash(SEVEN)
    assert ident["h_code_v2"] == C.h_code_hash_v2(SEVEN)
    old = dataclasses.replace(LangConfig(), task_choices=SEVEN)       # a run saved with the seven tasks
    tasks, _ = trained_subsets(old)
    assert code_identity(tasks or old.task_choices, [])["h_code_v2"] == C.h_code_hash_v2(SEVEN)
    assert C.h_code_hash_v2(SEVEN + ("tray_slide",)) != C.h_code_hash_v2(SEVEN)   # T8's own files are hashed


# ------------------------------------------------------------------------------------ behaviour mix --
def test_episode_risk_mix():
    cfg = LangConfig()
    assert cfg.behaviour_mix == {"tray_slide": {"medium": 0.5, "fast": 0.25, "slow": 0.25}}
    draws = [CF.episode_risk(cfg, "tray_slide", s)["behaviour"] for s in range(100_000_000, 100_003_000)]
    c = Counter(draws)
    assert set(c) == {"medium", "fast", "slow"}
    assert abs(c["medium"] / 3000 - 0.5) < 0.04 and abs(c["fast"] / 3000 - 0.25) < 0.04
    # a seed always gets the same behaviour (a replay, a resume or an evaluation of it)
    assert [CF.episode_risk(cfg, "tray_slide", s)["behaviour"] for s in range(100_000_000, 100_003_000)] == draws
    assert CF.episode_risk(cfg, "gap_place", 5) is None                             # other tasks: the yaml's draw
    assert CF.episode_risk(dataclasses.replace(cfg, item_risk={"time_scale": 1.0}), "tray_slide", 5) == {
        "time_scale": 1.0}                                                           # an explicit override wins
    assert CF.episode_risk(dataclasses.replace(cfg, behaviour_mix={}), "tray_slide", 5) is None
    one = dataclasses.replace(cfg, behaviour_mix={"tray_slide": {"recover_can": 1.0}})
    assert CF.episode_risk(one, "tray_slide", 5) == {"behaviour": "recover_can"}


def test_behaviour_mix_hash_only_with_tray_slide(tmp_path):
    """A config without tray_slide keeps its collect_hash (old shards resume); with it, the mix is in the hash."""
    cfg = LangConfig()
    assert C.collect_hash(cfg) == C.collect_hash(dataclasses.replace(cfg, behaviour_mix={}))
    old = {k: v for k, v in cfg.to_dict().items() if k != "behaviour_mix"}       # saved before the field
    assert C.collect_fields(LangConfig.from_dict(old)) == {k: v for k, v in C.collect_fields(cfg).items()}
    assert "behaviour_mix" not in C.collect_fields(cfg)
    t8 = dataclasses.replace(cfg, task_choices=("tray_slide",))
    assert C.collect_fields(t8)["behaviour_mix"] == {"tray_slide": {"medium": 0.5, "fast": 0.25, "slow": 0.25}}
    other = dataclasses.replace(t8, behaviour_mix={"tray_slide": {"fast": 1.0}})
    assert C.collect_hash(t8) != C.collect_hash(other)
    back = LangConfig.load(t8.save(tmp_path / "c.json"))
    assert back == t8 and C.collect_hash(back) == C.collect_hash(t8)


# ------------------------------------------------------------------------------------- designations --
def _t8_record():
    """A tray_slide episode record built from the task's real rule table (yaml + splits + objects.yaml), no sim."""
    from main.item_envs import language, physics
    from main.item_envs.items import ItemSpec
    from main.item_envs.registry import ITEM_TASKS
    from main.item_envs.rules import enumerate_rules, resolve_splits

    entry = ITEM_TASKS["tray_slide"]
    task_cfg = yaml.safe_load(entry.config.read_text())
    splits = yaml.safe_load((entry.config.parent.parent / "splits.yaml").read_text())
    items = [ItemSpec(item_id=i, entity=None, category=c, role=r)
             for i, c, r in (("tray", "tray", "task_object"), ("glass", "wine_glass", "candidate"),
                             ("can", "can", "candidate"))]
    items[1].size, items[2].size = "tall", "short"   # register_item's size words
    rules = enumerate_rules({i.item_id: i for i in items}, {}, task_cfg, objects_cfg=physics.load_objects_cfg(),
                            splits_cfg=resolve_splits(splits, "tray_slide"))
    runner = NS(task_name="tray_slide", host=None, embodiment="ur5-wsg", seed=1, items=items, regions=[],
                rules=rules, active_rule_id=None)
    return language.episode_record(runner, 1)


@pytest.mark.parametrize("behaviour,harm", [("slow", set()), ("medium", {"glass"}), ("fast", {"glass", "can"}),
                                            ("recover_glass", set()), ("recover_can", {"glass"})])
def test_in_way_follows_the_behaviour(behaviour, harm):
    rec = _t8_record()
    env = NS(task=NS(risk_knobs={"behaviour": behaviour}, episode_info={"referent": "can"}))
    assert SR.in_way_reasons(env, rec) == {i: behaviour for i in harm}
    desig = SR.all_designations(rec, SR.in_way(env, rec))
    assert desig == {f"upright|{i}|-": ("harm" if i in harm else "benign") for i in ("can", "glass")}


def test_in_way_needs_a_known_behaviour():
    rec = _t8_record()
    # recorded after the drag (episode_info.drag_profile) when the knob is gone
    env = NS(task=NS(risk_knobs={}, episode_info={"drag_profile": {"behaviour": "medium"}}))
    assert SR.in_way_reasons(env, rec) == {"glass": "medium"}
    with pytest.raises(KeyError, match="behaviour"):
        SR.in_way_reasons(NS(task=NS(risk_knobs={"behaviour": "warp"})), rec)
    with pytest.raises(KeyError, match="behaviour"):
        SR.in_way_reasons(NS(task=NS()), rec)


def test_sentences_name_each_item_uniquely():
    from main.item_envs import language

    rec = _t8_record()
    assert [r["rule_id"] for r in rec["rules"]] == ["upright|can|-", "upright|glass|-"]
    assert all(r["sampleable"] and r["split"] == "train" for r in rec["rules"])   # no splits.yaml section needed
    sent = {r["rule_id"]: language.sentences(r, rec, "train") for r in rec["rules"]}
    glass = [o for o in rec["objects"] if o["item_id"] == "glass"][0]
    can = [o for o in rec["objects"] if o["item_id"] == "can"][0]
    assert "the wine glass" in language.describe(glass, rec) and "the glass" in language.describe(glass, rec)
    assert "the can" in language.describe(can, rec)
    assert "keep the wine glass upright" in sent["upright|glass|-"] and "don't knock over the can" in sent["upright|can|-"]
    # no sentence of one rule names the other item, and the two rules share no sentence
    assert not set(sent["upright|glass|-"]) & set(sent["upright|can|-"])
    assert not any("can" in s.split() for s in sent["upright|glass|-"])
    assert not any("glass" in s for s in sent["upright|can|-"])
    held = {r["rule_id"]: language.sentences(r, rec, "heldout_template") for r in rec["rules"]}
    assert "make sure the wine glass stays standing" in held["upright|glass|-"]
    assert "make sure the goblet stays standing" not in held["upright|glass|-"]   # goblet = held-out word
    assert any("goblet" in s for s in language.sentences(rec["rules"][1], rec, "heldout_word"))


# --------------------------------------------------------------------------------- collector override --
def test_run_episodes_sets_the_behaviour_per_seed(tmp_path, fake_env, monkeypatch):
    """run_episodes hands each tray_slide build its seed's behaviour (cfg.item_risk) and writes it on the line;
    other tasks build with item_risk None, as before."""
    import main.lang.rollout as rollout

    seen = []

    class Rollout(fake_env):
        def __init__(self, cfg, seed, mode="collect", **kw):
            super().__init__(cfg, seed, mode, **kw)
            seen.append((cfg.task, seed, cfg.item_risk))
            if cfg.item_risk:
                self.env.task.risk_knobs = dict(cfg.item_risk)

    monkeypatch.setattr(rollout, "LangRolloutController", Rollout)
    cfg = dataclasses.replace(LangConfig(), task_choices=("tray_slide", "gap_place"))
    root = tmp_path / "shard"
    buf, side = C.make_pair(root, cfg, 400)
    slots = [("tray_slide", "ur5-wsg", None)] * 4 + [("gap_place", "ur5-wsg", None)]
    C.run_episodes(cfg, root, slots, 5000, buf, side, encoder="stub", kins={"ur5-wsg": None}, with_masks=False)
    assert [(t, s) for t, s, _ in seen] == [("tray_slide", 5000 + i) for i in range(4)] + [("gap_place", 5004)]
    assert [r for _, _, r in seen] == [CF.episode_risk(cfg, "tray_slide", 5000 + i) for i in range(4)] + [None]
    lines = C.read_lines(root / "lang_episodes.jsonl")
    assert [d.get("behaviour") for d in lines] == [r["behaviour"] for _, _, r in seen[:4]] + [None]
    assert "behaviour" not in lines[4]


from .test_collect_v2 import fake_env  # noqa: E402,F401  (the fake controller fixture)
