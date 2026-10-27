"""The item demo collector (vla_finetuning/items/collect_items.py, make_queue.py) without a simulator: the settings
pin, the resume bookkeeping, the CLI, the workqueue lines, and whole collection runs on the fake runner of
_items_fakes.py (files, attempts rows, keep reasons, renderer-error retries, exit codes):

    python -m pytest -q vla_finetuning/tests/test_items_collect.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

import h5py
import numpy as np
import pytest
from _items_fakes import FakeRunner, install

from vla_finetuning import common as C
from vla_finetuning.items import collect_items as CI
from vla_finetuning.items import make_queue as Q


def test_settings_pin(tmp_path):
    s = {"task": "gap_place", "max_steps": 7500, "keep_force_n": 20.0, "host": None}
    CI.pin_settings(tmp_path, s)
    assert json.loads((tmp_path / "settings.json").read_text()) == s
    CI.pin_settings(tmp_path, dict(s))                                   # the same settings: fine
    with pytest.raises(SystemExit, match="max_steps"):
        CI.pin_settings(tmp_path, {**s, "max_steps": 9000})


def test_tried_counts_retry_renderer_errors_up_to_three_times():
    rows = [{"seed": 1, "started": "x"}, {"seed": 1, "reason": "kept"},
            {"seed": 2, "started": "x"}, {"seed": 2, "reason": "render_error"},
            {"seed": 3, "started": "x"},                                 # the process died: tried
            {"seed": 4, "started": "x"}, {"seed": 4, "reason": "planner_error"},
            {"seed": 4, "started": "x"}, {"seed": 4, "reason": "render_error"}]
    rows += [r for _ in range(3) for r in ({"seed": 5, "started": "x"}, {"seed": 5, "reason": "render_error"})]
    rows += [{"seed": 7, "started": "x"}, {"seed": 7, "reason": "code_error"}]
    c = CI.tried_counts(rows)
    assert (c[1], c[2], c[3], c[4], c[5], c[6], c[7]) == (1, 0, 1, 0, 1, 0, 0)


def test_resume_state(tmp_path):
    d = tmp_path
    rows = [{"seed": 7, "started": "x", "attempt": 0}, {"seed": 7, "reason": "kept", "kept": True, "episode": 0},
            {"seed": 9, "started": "x", "attempt": 1}, {"seed": 9, "reason": "task_failed", "kept": False,
                                                        "video": "rejected_seed9_task_failed.mp4"},
            {"seed": 11, "started": "x", "attempt": 2}]                  # died after writing its files
    for p in ("data/episode0.hdf5", "data/episode1.hdf5", "data/episode1.hdf5.123.tmp",
              "instructions/episode0.json", "instructions/episode1.json"):
        (d / p).parent.mkdir(parents=True, exist_ok=True)
        (d / p).write_text("x")
    (d / "seed.txt").write_text("7 11")
    st = CI.resume_state(d, rows)
    assert [r["seed"] for r in st["kept"]] == [7] and st["seeds"] == [7]
    assert sorted(p.name for p in (d / "data").iterdir()) == ["episode0.hdf5"]
    assert sorted(p.name for p in (d / "instructions").iterdir()) == ["episode0.json"]
    assert (d / "seed.txt").read_text() == "7"
    assert st["counts"] == Counter({7: 1, 9: 1, 11: 1}) and st["n_started"] == 3 and st["n_rej_vid"] == 1
    with pytest.raises(SystemExit):
        CI.resume_state(d, rows + [{"seed": 13, "reason": "kept", "kept": True, "episode": 2}])


def test_diag_overrides_refused(monkeypatch):
    for k in [k for k in os.environ if k.startswith("ITEM_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("ITEM_VIDEO_STYLE", "demo")
    assert CI.diag_overrides() == []
    monkeypatch.setenv("ITEM_T6_REFERENT", "hit")
    assert CI.diag_overrides() == ["ITEM_T6_REFERENT"]


def test_cli():
    a = CI.parse_args(["--task", "gap_place", "--root", "/r", "--config-name", "s0"])
    assert a.max_steps == 7500 and a.host is None and a.max_attempts == 40 and not a.video
    a = CI.parse_args(["--task", "selective_avoid", "--host", "stack_bowls_two", "--root", "/r", "--config-name", "x"])
    assert a.max_steps == 10000 and a.host == "stack_bowls_two"
    assert CI.parse_args(["--task", "push_zone", "--root", "/r", "--config-name", "s0"]).max_steps == 15000
    assert CI.parse_args(["--task", "clean_pour", "--root", "/r", "--config-name", "s0",
                          "--max-steps", "123"]).max_steps == 123
    for bad in (["--task", "selective_avoid"],                         # T1 needs its host
                ["--task", "selective_avoid", "--host", "place_shoe"],
                ["--task", "gap_place", "--host", "stack_blocks_two"],
                ["--task", "granular_pour"],                           # not a demo task (2026-10-23)
                ["--task", "gap_place", "--seed-start", "999999", "--seed-stop", "1000001"]):
        with pytest.raises(SystemExit):
            CI.parse_args(bad + ["--root", "/r", "--config-name", "s0"])


def test_workqueue_shards():
    sh = Q.shards()
    assert len(sh) == 5 + 4 * 4
    per_task = Counter()
    for s in sh:
        per_task[s["task"]] += s["target"]
    assert per_task == {t: 100 for t in Q.ITEM_TASKS}
    t1 = [s for s in sh if s["task"] == "selective_avoid"]
    assert [s["host"] for s in t1] == list(Q.t1_hosts()) and all(s["config"] == s["host"] for s in t1)
    walks = {}
    for s in sh:
        walks[(s["task"], s["config"])] = set(range(s["seed_start"], s["seed_stop"], s["seed_step"]))
        assert 300000 <= s["seed_start"] < s["seed_stop"] <= 350000 < CI.SEED_STOP
    seen = set()
    for k, w in walks.items():                                          # no seed in two shards
        assert not (w & seen), k
        seen |= w
    # the collision-avoidance evaluation grid of stack_blocks_two (eval_ckpt.episode_seed + redraws of 1000)
    ca_eval = {70000 + 2 * 100000 + 3 * 10000 + ep + 1000 * r for ep in range(50) for r in range(5)}
    assert not (walks[("selective_avoid", "stack_blocks_two")] & ca_eval)
    for s in sh:                                                       # every line is a valid collector call
        argv = Q.command(s, "/abs/root").split()
        assert argv[:2] == ["bash", Q.LOOP]
        a = CI.parse_args(argv[2:])
        assert (a.task, a.host, a.config_name, a.seed_start, a.target) == (s["task"], s["host"], s["config"],
                                                                         s["seed_start"], s["target"])
        assert a.video and a.max_attempts == 40 and a.root == "/abs/root"


def test_make_queue_refuses_the_demo_root(tmp_path):
    root = tmp_path / "demos"
    with pytest.raises(SystemExit):
        Q.main(["--root", str(root), "--out", str(root / "q.cmds")])
    out = tmp_path / "q.cmds"
    assert Q.main(["--root", str(root), "--out", str(out), "--tasks", "gap_place"]) == 0
    assert len([ln for ln in out.read_text().splitlines() if ln.startswith("bash ")]) == 4


def test_make_queue_only_tasks_with_a_safe_mode(tmp_path, monkeypatch):
    """Before the T3 safe mode is merged push_zone's yaml has no safe_mode section: left out by default, refused by
    name (its jobs would only end in code errors and the workqueue drop them)."""
    for task in ("selective_avoid", "tray_transport", "clean_pour", "gap_place"):
        assert Q.has_safe_mode(task), task
    monkeypatch.setattr(Q, "has_safe_mode", lambda task: task != "push_zone")
    root, out = tmp_path / "demos", tmp_path / "q.cmds"
    assert Q.main(["--root", str(root), "--out", str(out)]) == 0
    lines = [ln for ln in out.read_text().splitlines() if ln.startswith("bash ")]
    assert len(lines) == 5 + 3 * 4 and not any("push_zone" in ln for ln in lines)
    with pytest.raises(SystemExit, match="push_zone"):
        Q.main(["--root", str(root), "--out", str(out), "--tasks", "gap_place", "push_zone"])
    monkeypatch.setattr(Q, "has_safe_mode", lambda task: True)            # after the merge: all 21 shards
    assert Q.main(["--root", str(root), "--out", str(out)]) == 0
    assert len([ln for ln in out.read_text().splitlines() if ln.startswith("bash ")]) == 21


def _rows(d):
    return C.read_jsonl(d / "attempts.jsonl")


def _done(d):
    return [r for r in _rows(d) if "reason" in r]


def test_collect_gap_place(tmp_path, monkeypatch):
    scripts = {340004: {"dip": ("stay|block_1|-", 5, -0.001)},          # the right block moved 2.1 cm at capture 5
               340008: {"safe_fail": "no_route"},
               340012: {"render_error_at": 1}}                           # the renderer breaks at the first frame
    install(monkeypatch, scripts)
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s0", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340100", "--target", "3"]
    assert CI.main(argv) == 75 and CI._hard_exit                       # a renderer error ends the process
    d = tmp_path / "gap_place" / "s0"
    assert [r["reason"] for r in _done(d)] == ["kept", "rule:stay|block_1|-", "safe_fail:no_route", "render_error"]
    assert all(r.closed for r in FakeRunner.BUILT)
    install(monkeypatch, {})                                             # the next process: the renderer works
    assert CI.main(argv) == 0 and not CI._hard_exit
    done = _done(d)
    assert [(r["seed"], r["reason"]) for r in done[4:]] == [(340012, "kept"), (340016, "kept")]   # retried first
    assert (d / "seed.txt").read_text() == "340000 340012 340016"
    assert [r["attempt"] for r in _rows(d) if "started" in r] == list(range(6))
    for k in range(3):
        with h5py.File(d / "data" / f"episode{k}.hdf5", "r") as f:
            n = f["joint_action/vector"].shape[0]
            assert f["joint_action/vector"].shape == (n, 14) and f["endpose"].shape == (n, 14)
            assert set(f["observation"]) == set(C.CAMS)
            vec = f["joint_action/vector"][:]
            assert not any(np.array_equal(a, b) for a, b in zip(vec, vec[1:]))   # pure waits left out
        instr = json.loads((d / "instructions" / f"episode{k}.json").read_text())
        assert len(instr["seen"]) == len(instr["unseen"]) == 100
        assert all("the wooden block" in s for s in instr["seen"] + instr["unseen"])
    kept = [r for r in done if r["kept"]]
    assert [r["episode"] for r in kept] == [0, 1, 2]
    r0 = kept[0]
    # the start frame + one every 15 steps of the 360-step play_once (24), minus the 4 in the 60-step wait; the end
    # frame repeats the last one (skipped too)
    assert r0["n_frames"] == len(r0["frame_steps"]) == 21 and r0["n_skipped_frames"] == 5
    assert r0["frame_steps"][:3] == [1, 16, 31] and r0["settle_steps"] == 250 and r0["n_steps"] == 1 + 360 + 250
    assert set(r0["protected"]) == {"upright|block_0|-", "stay|block_0|-", "upright|block_1|-", "stay|block_1|-"}
    assert r0["min_h"]["stay|block_0|-"] == pytest.approx(1.0) and r0["first_violation_t"] == {}
    assert r0["info"]["{A}"] == "the wooden block"
    assert r0["instruction_example"] == json.loads((d / "instructions" / "episode0.json").read_text())["seen"][0]
    bad = done[1]
    assert bad["broken"] == ["stay|block_1|-"] and bad["first_violation_t"]["stay|block_1|-"] == pytest.approx(0.2)
    assert bad["first_violation_tick"] == {"stay|block_1|-": 5}
    assert bad["min_h"]["stay|block_1|-"] == pytest.approx(-0.05) and not (d / "data" / "episode3.hdf5").exists()
    sf = done[2]
    assert sf["safe_fail"] == "no_route" and sf["plan_failures"] == 1 and sf["settle_steps"] == 0
    re_ = done[3]
    assert re_["error"] == "RuntimeError: cannot create buffer" and re_["n_frames"] == 0
    assert (d / "runner" / "0" / "gap_place" / "ur5-wsg" / "seed_340000" / "trace.npz").exists()
    assert done[0]["runner_dir"] == "runner/0/gap_place/ur5-wsg/seed_340000"
    assert json.loads((d / "settings.json").read_text())["instruction"] == "plain"
    with pytest.raises(SystemExit):                                     # other settings in the same directory
        CI.main(argv[:-2] + ["--target", "4", "--max-steps", "9000"])


def test_collect_t1_contact_and_ambiguity(tmp_path, monkeypatch):
    install(monkeypatch, {302100: {"box_color": "red"},                # "stack the green block on the red block"
                          302101: {"touch_at": 50},                     # the hand touches obj_0 at step 50
                          302102: {"setup_error": "UnStableError: objects might fall"}})
    argv = ["--task", "selective_avoid", "--host", "stack_blocks_two", "--root", str(tmp_path), "--config-name",
            "stack_blocks_two", "--seed-start", "302100", "--seed-step", "1", "--seed-stop", "303000", "--target", "1"]
    assert CI.main(argv) == 0
    d = tmp_path / "selective_avoid" / "stack_blocks_two"
    done = _done(d)
    assert [r["reason"] for r in done] == ["ambiguous_instruction", "contact", "no_scene", "kept"]
    assert done[0]["detail"] == ["obj_0"] and done[0]["scene"]["placements"][0]["cls"] == "on_path"
    c = done[1]
    assert c["contact"] and c["first_touch_step"] == 50 and c["contact_partners"] == ["finger_left"]
    assert c["error"].startswith("ContactAbort") and c["n_steps"] == 50
    k = done[3]
    assert k["protected"] == {"stay|obj_0|-": 0.0, "stay|obj_1|-": 0.0} and k["contact"] is False
    assert k["host"] == "stack_blocks_two" and k["force_max"] == 0.0
    instr = json.loads((d / "instructions" / "episode0.json").read_text())
    assert all("red block" in s and "green block" in s for s in instr["seen"])
    assert instr == C.make_instructions("stack_blocks_two", {"{A}": "red block", "{B}": "green block"}, 302103)


def test_collect_t1_extra_objects_touching_each_other(tmp_path, monkeypatch):
    """Any contact of an extra object but with the table rejects (another extra object too); the table does not.
    The T1 safe expert's diagnostics reach the row's top level."""
    levels = {"0": 6, "2": 1, "fail": 1}
    install(monkeypatch, {302200: {"contacts": {40: [("obj_1", "obj_0")]}},
                          302201: {"contacts": {30: [("obj_0", "table"), ("obj_1", "table")]},
                                   "episode_info": {"safe_plan_levels": levels, "safe_last_fail": "ik_fail",
                                                    "safe_plan_log": [{"ok": False, "kind": "ik_fail"}]}}})
    argv = ["--task", "selective_avoid", "--host", "stack_blocks_two", "--root", str(tmp_path), "--config-name",
            "stack_blocks_two", "--seed-start", "302200", "--seed-step", "1", "--seed-stop", "302300", "--target", "1"]
    assert CI.main(argv) == 0
    done = _done(tmp_path / "selective_avoid" / "stack_blocks_two")
    assert [r["reason"] for r in done] == ["contact", "kept"]
    assert done[0]["contact_partners"] == ["obj_0", "obj_1"] and done[0]["first_touch_step"] == 40
    k = done[1]
    assert k["contact"] is False and k["safe_plan_levels"] == levels and k["safe_last_fail"] == "ik_fail"
    assert "push" not in k and "route" not in k and k["diag_task"]["safe_plan_log"][0]["kind"] == "ik_fail"


def test_plan_failed_only_on_the_end_flag(tmp_path, monkeypatch):
    """A failed plan the task recovered from (meta.plan_failures > 0, plan_success True at the end: T5 / T7's step
    chain after a failed via path, the T6 pour) is kept; plan_success False at the end is plan_failed, even when
    check_success holds, and comes before task_failed."""
    install(monkeypatch, {340000: {"plan_failures": 2},
                          340004: {"plan_success": False},
                          340008: {"plan_success": False, "success": False}})
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s0", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340100", "--target", "2"]
    assert CI.main(argv) == 0
    done = _done(tmp_path / "gap_place" / "s0")
    assert [r["reason"] for r in done] == ["kept", "plan_failed", "plan_failed", "kept"]
    assert (done[0]["plan_failures"], done[0]["plan_success"], done[0]["settle_steps"]) == (2, True, 250)
    assert (done[1]["plan_failures"], done[1]["plan_success"], done[1]["settle_steps"]) == (1, False, 0)
    assert done[1]["task_success"] is True and done[3]["plan_failures"] == 0


def test_t3_route_and_push_summaries():
    """push_zone's safe expert writes episode_info route (plan_route's answer) and push (push_via's log): their
    summaries go to the row's top level, the bulky parts stay in diag_task."""
    diag = {"safe_fail": None, "plan_fail_step": 7, "replans": [{"ok": True}],
            "route": {"ok": True, "via": [[0.1, 0.0], [0.2, 0.05], [0.3, 0.1]], "sides": {"zone_red": "near"},
                      "hand": "wide", "turns_deg": [20.5], "first_deg": 3.0, "length": 0.31,
                      "min_clear": {"zone_red": 0.018}, "reason": None, "detail": {"lanes": list(range(50))}},
            "push": {"steps": [{"kind": "approach"}, {"kind": "push"}, {"kind": "push"}, {"kind": "retreat"}],
                     "replans": [{"at": [0.2, 0.0], "why": "lane", "ok": True}], "yaw_jump_max_deg": 4.5,
                     "ee_miss_max": 0.006, "ee_miss_flag": False, "box_tilt_max_deg": 1.2, "fail": None}}
    out = CI.expert_diag(diag)
    assert out["route"] == {"n_via": 3, "ok": True, "reason": None, "length": 0.31, "hand": "wide",
                            "sides": {"zone_red": "near"}, "turns_deg": [20.5], "first_deg": 3.0,
                            "min_clear": {"zone_red": 0.018}}
    assert out["push"] == {"steps": {"approach": 1, "push": 2, "retreat": 1}, "n_replans": 1, "yaw_jump_max_deg": 4.5,
                           "ee_miss_max": 0.006, "ee_miss_flag": False, "box_tilt_max_deg": 1.2, "fail": None}
    assert out["plan_fail_step"] == 7 and set(out) == {"route", "push", "plan_fail_step"}
    assert CI.expert_diag({}) == {} and CI.expert_diag(None) == {}
    assert CI.expert_diag({"push": {"fail": "lane"}}) == {"push": {"steps": {}, "n_replans": 0, "fail": "lane"}}


def test_errors_before_the_scene_and_in_the_trace_are_code_errors(tmp_path, monkeypatch):
    """The seeding and the runner's constructor run inside the guarded block, a trace-scoring failure keeps its type
    first: a deterministic bug in any of them is a code_error (the seed kept for later, 3 in a row stop with 3)."""
    install(monkeypatch, {340004: {"ctor_error": AttributeError("'RiskParams' object has no attribute 'task'")},
                          340008: {"bad_trace": True}})

    def seeds(seed):
        if seed == 340000:
            raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(C, "set_global_seeds", seeds)
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s0", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340100", "--target", "3"]
    assert CI.main(argv) == 3
    d = tmp_path / "gap_place" / "s0"
    done = _done(d)
    assert [r["reason"] for r in done] == ["code_error"] * 3
    assert done[0]["error"].startswith("ModuleNotFoundError: No module named 'torch'")
    assert all(r["detail"] == "seeding / runner constructor" for r in done[:2])     # run_attempt's own guard
    assert done[1]["error"].startswith("AttributeError: ")
    assert done[2]["error"].startswith("TraceMismatch: trace: ")
    assert all(r.closed for r in FakeRunner.BUILT) and len(FakeRunner.BUILT) == 1    # the constructor never finished
    install(monkeypatch, {})
    assert CI.main(argv) == 0
    assert [r["seed"] for r in _done(d)[3:]] == [340000, 340004, 340008]


def test_an_exception_past_the_attempt_still_gets_its_row(tmp_path, monkeypatch):
    """Anything run_attempt itself lets through (here a bug in the row bookkeeping) is classified by the loop: the
    "started" line gets its row and the process goes on."""
    install(monkeypatch, {})
    orig, calls = CI.expert_diag, []

    def flaky(task_diag):
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("push")
        return orig(task_diag)

    monkeypatch.setattr(CI, "expert_diag", flaky)
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s0", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340100", "--target", "1"]
    assert CI.main(argv) == 0
    rows = _rows(tmp_path / "gap_place" / "s0")
    assert [r.get("reason") for r in rows] == [None, "code_error", None, "kept"]
    assert rows[1]["error"] == "KeyError: 'push'" and rows[1]["detail"] == "run_attempt" and rows[1]["seed"] == 340000
    assert all(r.closed for r in FakeRunner.BUILT)


def test_videos(tmp_path, monkeypatch):
    install(monkeypatch, {340004: {"dip": ("stay|block_1|-", 5, -0.001)}, 340008: {"safe_fail": "no_route"},
                          340012: {"setup_error": "UnStableError: x"}, 340016: {"success": False}})
    made = []
    monkeypatch.setattr(CI, "write_video", lambda path, frames: made.append((path.name, len(frames))))
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s0", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340100", "--target", "2", "--video", "--rejected-videos", "2"]
    assert CI.main(argv) == 0
    # kept episode 0 (every 10th), then the first two rejected attempts with frames; the no_scene one has none
    assert [m[0] for m in made] == ["episode0_seed340000.mp4", "rejected_seed340004_rule_stay_block_1_-.mp4",
                                    "rejected_seed340008_safe_fail_no_route.mp4"]
    assert all(n > 0 for _, n in made)
    rows = _done(tmp_path / "gap_place" / "s0")
    assert [r.get("video") for r in rows] == [m[0] for m in made] + [None, None, None]
    assert [r["reason"] for r in rows] == ["kept", "rule:stay|block_1|-", "safe_fail:no_route", "no_scene",
                                           "task_failed", "kept"]


def test_code_errors_stop_the_process_and_keep_their_seeds(tmp_path, monkeypatch):
    """A broken checkout or code (here: a safe mode not merged yet, then a bug inside play_once) must not burn the
    seeds as no_scene / error: three code errors in a row end the process with 3, and the fixed code tries them."""
    seeds = [340000, 340004, 340008]
    install(monkeypatch, {seeds[0]: {"setup_error": NotImplementedError("gap_place: no safe nominal mode yet")},
                          seeds[1]: {"play_error": AttributeError("'NoneType' object has no attribute 'p'")},
                          seeds[2]: {"setup_error": ImportError("cannot import name 'push_via'")}})
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s0", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340100", "--target", "3"]
    assert CI.main(argv) == 3
    d = tmp_path / "gap_place" / "s0"
    assert [r["reason"] for r in _done(d)] == ["code_error"] * 3
    install(monkeypatch, {})
    assert CI.main(argv) == 0
    assert [r["seed"] for r in _done(d)[3:]] == seeds and (d / "seed.txt").read_text().split() == list(map(str, seeds))


def test_max_attempts_recycles_the_process(tmp_path, monkeypatch):
    install(monkeypatch, {s: {"success": False} for s in range(340000, 340100, 4)})
    argv = ["--task", "gap_place", "--root", str(tmp_path), "--config-name", "s1", "--seed-start", "340000",
            "--seed-step", "4", "--seed-stop", "340012", "--target", "2", "--max-attempts", "2"]
    assert CI.main(argv) == 75
    assert CI.main(argv) == 0                                           # one seed left, then the walk ends
    assert [r["reason"] for r in _done(tmp_path / "gap_place" / "s1")] == ["task_failed"] * 3
