"""VLM failure function (VLM_FAILURE_PLAN, 2026-10-26): prompts, answer parsing, label files -> h overlay, the rule
restriction, the labeler's episode flow (with a stub VLM), the label waiter and the named-rule choice. CPU only."""

import dataclasses
import json

import numpy as np
import pytest

from main.item_envs import language
from main.lang import vlm_prompts as V
from main.lang.agent_view import AgentViewWriter, read_done
from main.lang.collect import choose_named_rule, collect_fields
from main.lang.config import LangConfig
from main.lang.data import LangSampler
from main.lang.hsource import HSourceSampler, fails_soon_of, missing_labels, spread, wait_labels
from main.lang.vlm_label import episode_queries, label_episode, pending

from .fake_data import make_root

VLM = "qwen25vl32b"


def cfg(**kw):
    return dataclasses.replace(LangConfig(), **kw)


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path / "r0", episodes=4, capacity=400, task="selective_avoid")


def record(root, eid):
    return language.refresh_record(json.loads((root / "item_buffer/episodes" / f"{eid}.json").read_text()), None)


def write_labels(root, eid, scores_by_col, ticks, prompt_hash=None, n_rows=40):
    rec = record(root, eid)
    rules = {}
    for col, sc in scores_by_col.items():
        r = rec["rules"][col]
        rules[r["rule_id"]] = {"idx": col, "score": list(sc), "explanation": [""] * len(sc),
                               "answer": [""] * len(sc), "parsed": [True] * len(sc)}
    V.write_json_atomic(V.label_path(root, VLM, eid), {
        "episode_id": eid, "vlm": VLM, "prompt_hash": prompt_hash or V.prompt_hash(), "every": 5, "n_rows": n_rows,
        "ticks": list(ticks), "rules": rules, "error": None})


def avoid_cols(root, eid):
    rec = record(root, eid)
    return [r["idx"] for r in rec["rules"] if V.labelled(r, rec) and V.build_prompt(r, rec)]


# ------------------------------------------------------------------ prompts --

def test_parse_score():
    assert V.parse_score('```json\n{"score": -1, "explanation": "It bumps the box."}\n```') == (-1, "It bumps the box.")
    assert V.parse_score('{"score": "0", "explanation": "unclear"}') == (0, "unclear")
    assert V.parse_score('{"score": 1}')[0] == 1
    assert V.parse_score('{"score": 0.8, "explanation": "x"}')[0] == 1     # rounds to the nearest of -1 / 0 / 1
    assert V.parse_score('"score": -1, "explanation": "tipped')[0] == -1   # truncated JSON
    assert V.parse_score("I cannot tell.") == (None, "")


def test_prompts_and_rule_sets(root):
    rec = record(root, 1000)
    prompts = V.load_prompts()["prompts"]
    assert set(prompts) == {"selective_avoid", "tray_transport", "clean_pour"}
    assert V.allowed_consequences("all", "selective_avoid") is None
    assert V.allowed_consequences("vlm_prompts", "selective_avoid") == {"avoid"}
    assert V.allowed_consequences("vlm_prompts", "gap_place") is None      # not listed: unrestricted
    stay = [r for r in rec["rules"] if r["consequence"] == "stay" and r.get("sampleable")]
    assert stay and not any(V.labelled(r, rec) for r in stay)              # T1 "don't move": no prompt
    cols = avoid_cols(root, 1000)
    assert cols
    text, phrase = V.build_prompt(rec["rules"][cols[0]], rec)
    assert text.startswith(V.load_prompts()["images"]) and phrase in text and "{object}" not in text
    assert phrase in language.describe(rec["objects"][rec["rules"][cols[0]]["subject"]["object"]], rec, "train")


def test_config_fields():
    with pytest.raises(AssertionError):
        cfg(h_source="vlm:qwen25vl32b")                                    # needs rule_consequences vlm_prompts
    with pytest.raises(ValueError):
        cfg(h_source="vlm:nope", rule_consequences="vlm_prompts")
    c = cfg(h_source="vlm:robobrain2_32b", rule_consequences="vlm_prompts", agent_view_every=5)
    assert LangConfig.from_dict(c.to_dict()) == c
    # defaults change no collection hash; the new collection fields count once set
    d = collect_fields(LangConfig())
    assert not {"h_source", "rule_consequences", "agent_view_every"} & set(d)
    assert "h_source" not in collect_fields(c) and collect_fields(c)["agent_view_every"] == 5


# ---------------------------------------------------------------- overlay --

def test_spread_and_fails_soon():
    t = np.arange(12)
    h = spread([0, 5, 10], [1, -1, 0], t)
    #             t: 0  1  2  3   4   5   6   7   8  9  10 11
    assert h.tolist() == [1, 1, 1, -1, -1, -1, -1, -1, 0, 0, 0, 0]
    fs = fails_soon_of(np.array([1, 1, 1, 1, -1, 1, 1]), horizon=3)
    assert fs.tolist() == [False, False, True, True, True, False, False]


def test_overlay_h_and_usable_rules(root):
    cols = avoid_cols(root, 1001)
    ticks = list(range(0, 40, 5))
    sc = [1, 1, 1, -1, -1, -1, 1, 1]
    write_labels(root, 1001, {cols[0]: sc}, ticks)
    s = HSourceSampler(root / "item_buffer", h_source=f"vlm:{VLM}", rule_consequences="vlm_prompts")
    A = s.buf.arrays
    ep = np.asarray(A["episode_id"]).reshape(-1)
    rows = np.flatnonzero(ep[:s.buf.n] == 1001)
    t = np.asarray(A["t_in_ep"])[rows]
    assert np.array_equal(A["h_rules"][rows, cols[0]], spread(ticks, sc, t))
    assert np.array_equal(A["fails_soon"][rows, cols[0]], fails_soon_of(spread(ticks, sc, t)))
    assert np.isnan(A["h_rules"][rows, cols[1]]).all()                     # not labelled: no h
    # only the labelled episode and rule are usable; the simulator's h is kept apart
    assert set(ep[s._rows].tolist()) == {1001}
    assert s._sample_cols[1001].tolist() == [cols[0]]
    assert s._sample_cols[1000].size == 0
    sim = np.asarray(s.sim_h[rows, cols[0]])
    assert not np.array_equal(sim, A["h_rules"][rows, cols[0]])
    ag = s.agreement()["selective_avoid"]
    assert ag["cells"] == s._rows.size and ag["vlm_neg"] == int((A["h_rules"][s._rows, cols[0]] < 0).sum())


def test_overlay_refuses_other_prompts_and_rebuilds(root):
    cols = avoid_cols(root, 1000)
    write_labels(root, 1000, {cols[0]: [1] * 8}, range(0, 40, 5), prompt_hash="deadbeef0000")
    s = HSourceSampler(root / "item_buffer", h_source=f"vlm:{VLM}", rule_consequences="vlm_prompts")
    assert len(s) == 0 and s.n_refused == 1
    write_labels(root, 1002, {cols[0]: [-1] * 8}, range(0, 40, 5))
    s.refresh()                                                             # new files picked up
    ep = np.asarray(s.buf.arrays["episode_id"]).reshape(-1)
    assert set(ep[s._rows].tolist()) == {1002}
    assert (s.buf.arrays["h_rules"][s._rows, cols[0]] == -1).all()


def test_sim_source_restricts_rules_only(root):
    s = HSourceSampler(root / "item_buffer", rule_consequences="vlm_prompts")
    rec = record(root, 1000)
    used = {rec["rules"][c]["consequence"] for c in s._sample_cols[1000]}
    assert used == {"avoid"}
    full = HSourceSampler(root / "item_buffer")
    assert {rec["rules"][c]["consequence"] for c in full._sample_cols[1000]} == {"avoid", "stay"}
    assert np.array_equal(np.asarray(s.buf.arrays["h_rules"]), np.asarray(full.buf.arrays["h_rules"]))


def test_lang_sampler_vlm_h_and_sim_probe(root):
    for eid in (1000, 1001, 1002, 1003):
        cols = avoid_cols(root, eid)
        write_labels(root, eid, {c: ([1, -1] * 4) if k == 0 else [1] * 8 for k, c in enumerate(cols)},
                     range(0, 40, 5))
    c = cfg(h_source=f"vlm:{VLM}", rule_consequences="vlm_prompts", task_choices=("selective_avoid",))
    smp = LangSampler([root], c, tasks=("selective_avoid",), seed=0)
    b = smp.sample(32)
    assert set(np.unique(np.concatenate([b["h"], b["h_next"]])).tolist()) <= {-1.0, 1.0}
    assert set(b["consequence"]) == {"avoid"}
    probe = LangSampler([root], c, tasks=("selective_avoid",), seed=0, probe=True)
    assert probe.sources[0].rs.vlm is None                                  # probes: the simulator's h
    assert smp.agreement()["selective_avoid"]["cells"] > 0


# ---------------------------------------------------------------- labeler --

class StubVLM:
    name, dir = VLM, "stub"

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def ask(self, batch, sample=False):
        self.calls.append((len(batch), sample))
        return [self.answer(text, sample) for _, text in batch]


def frames(root, eid, n_rows=40, every=5):
    av = AgentViewWriter(root, eid, every)
    for t in range(n_rows):
        if av.wants(t):
            av.save(t, np.full((24, 32, 3), t, np.uint8))
    av.finish(n_rows)


def test_label_episode_flow(root):
    frames(root, 1003)
    assert read_done(root, 1003)["ticks"] == list(range(0, 40, 5))
    assert pending(root, VLM) == [1003]
    skel, qs, rec = episode_queries(root, 1003)
    n_rules = len(avoid_cols(root, 1003))
    assert len(qs) == 8 * n_rules and skel["error"] is None
    # garbage first: the sampled retry answers; the first rule never parses -> 0, parsed false
    bad = next(iter(skel["rules"].values()))["referent"]
    bad_prompt = qs[0]["prompt"]

    def answer(text, sample):
        if text == bad_prompt and n_rules > 1:
            return "no idea"
        return '{"score": -1, "explanation": "bumped"}' if sample else "hmm"

    stub = StubVLM(answer)
    lab = label_episode(stub, root, 1003, batch_size=6)
    assert any(s for _, s in stub.calls) and not stub.calls[0][1]
    r_bad = next(r for r in lab["rules"].values() if r["referent"] == bad)
    if n_rules > 1:
        assert r_bad["score"] == [0] * 8 and not any(r_bad["parsed"])
    good = [r for r in lab["rules"].values() if r["referent"] != bad]
    assert all(r["score"] == [-1] * 8 and all(r["parsed"]) for r in good)
    V.write_json_atomic(V.label_path(root, VLM, 1003), lab)
    assert pending(root, VLM) == []
    s = HSourceSampler(root / "item_buffer", h_source=f"vlm:{VLM}", rule_consequences="vlm_prompts")
    assert set(s._sample_cols[1003].tolist()) == set(avoid_cols(root, 1003))


def test_agent_view_discard_and_missing_frames(root):
    av = AgentViewWriter(root, 1002, 5)
    av.save(0, np.zeros((8, 8, 3), np.uint8))
    av.discard()
    assert not V.frames_dir(root, 1002).exists()
    frames(root, 1001)
    (V.frames_dir(root, 1001) / "t0010.jpg").unlink()
    skel, qs, _ = episode_queries(root, 1001)
    assert qs == [] and "frames missing" in skel["error"]


def test_missing_labels_and_wait(root):
    lines = [json.loads(x) for x in (root / "lang_episodes.jsonl").read_text().splitlines()]
    for k, d in enumerate(lines):   # the fake lines have no rows: give them the real ones
        d["rows"] = [40 * k, 40 * (k + 1)]
    (root / "lang_episodes.jsonl").write_text("".join(json.dumps(d) + "\n" for d in lines))
    assert missing_labels(root, VLM) == [1000, 1001, 1002, 1003]
    for eid in (1000, 1001, 1002):
        write_labels(root, eid, {}, [0])
    naps = []

    def sleep(s):
        naps.append(s)
        write_labels(root, 1003, {}, [0])

    assert wait_labels([root], VLM, lambda: False, sleep=sleep) is None and naps
    assert missing_labels(root, VLM) == []
    (V.label_path(root, VLM, 1003)).unlink()
    assert "waiting for 1" in wait_labels([root], VLM, lambda: True)


def test_named_rule_respects_allowed(root):
    rec = record(root, 1000)
    desig = {r["rule_id"]: ("harm" if r["consequence"] == "stay" else "benign") for r in rec["rules"]}
    rng = np.random.default_rng(0)
    picks = {choose_named_rule(rec, desig, rng, 1.0, allowed=frozenset({"avoid"}))["consequence"] for _ in range(20)}
    assert picks == {"avoid"}   # no harm-designated avoid rule: falls back to the benign ones, never a stay rule
    assert choose_named_rule(rec, desig, np.random.default_rng(0), 1.0)["consequence"] == "stay"
