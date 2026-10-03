"""probe sweep resume: finished episodes are skipped, corrupt metas recomputed."""

import json
from types import SimpleNamespace

from main.item_envs import probe


def _args(out):
    return SimpleNamespace(
        out=str(out),
        task="push_zone",
        embodiment="ur5-wsg",
        risk=[],
        host=None,
        active_rule=None,
        render="none",
        video=False,
        max_steps=0,
        task_cfg=None,
    )


def _meta_dir(root, tag="cellA", seed=3):
    d = root / tag / "push_zone" / "ur5-wsg" / f"seed_{seed}"
    d.mkdir(parents=True)
    return d


def test_run_one_resumes_existing_meta(tmp_path):
    meta = {
        "task": "push_zone",
        "embodiment": "ur5-wsg",
        "seed": 3,
        "task_success": True,
        "plan_failures": 0,
        "rules": [{"rule_id": "inside|push_0|table_top"}],
        "violations": {"inside|push_0|table_top": False},
    }
    (_meta_dir(tmp_path) / "meta.json").write_text(json.dumps(meta))
    m, error, runner = probe._run_one(_args(tmp_path), {"overshoot": 0.0}, "cellA", 3)
    assert error is None and runner is None
    assert m.task_success is True
    # resumed metas feed the adaptive escalation check
    assert probe._primary_violation_rate([m]) == 0.0


def test_run_one_recomputes_corrupt_meta(tmp_path, monkeypatch):
    (_meta_dir(tmp_path) / "meta.json").write_text("{not json")
    called = {}

    class FakeRunner:
        def __init__(self, *a, **kw):
            called["built"] = True

        def run(self):
            meta = SimpleNamespace(task_success=False, plan_failures=0, rules=[], violations={})
            return meta, None

    monkeypatch.setattr("main.item_envs.runner.ItemEpisodeRunner", FakeRunner)
    m, error, runner = probe._run_one(_args(tmp_path), {}, "cellA", 3)
    assert called.get("built") is True
    assert error is None and runner is not None
    assert m.task_success is False


# ---------------------------------------------------------------------------
# relative --out paths: Env() chdirs to the RoboTwin root, so every output
# path must be absolute before the first env exists (calib sweeps 22634591..
# died writing trace.npz under RoboTwin/runs/...).
# ---------------------------------------------------------------------------


def test_main_resolves_relative_out_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def _capture(args):
        seen["out"] = args.out
        return 0

    parser = probe.build_parser()
    monkeypatch.setattr(probe, "build_parser", lambda: parser)
    for name in ("episode", "sweep"):
        sub = parser._subparsers._group_actions[0].choices[name]
        sub.set_defaults(func=_capture)
        assert probe.main([name, "--task", "push_zone", "--embodiment", "ur5-wsg",
                           "--out", "runs/rel"]) == 0
        assert seen["out"] == str(tmp_path / "runs" / "rel")


def test_runner_out_dir_is_absolute(tmp_path, monkeypatch):
    from main.item_envs.runner import ItemEpisodeRunner

    monkeypatch.chdir(tmp_path)
    runner = ItemEpisodeRunner("push_zone", embodiment="ur5-wsg", seed=2,
                               render="none", out_root="runs/rel")
    assert runner.out_dir.is_absolute()
    assert runner.out_dir == tmp_path / "runs" / "rel" / "push_zone" / "ur5-wsg" / "seed_2"


def test_sweep_shard_partitions_cells(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(probe, "_run_one", lambda args, cell, tag, seed: (
        ran.append((tag, seed)) or (SimpleNamespace(task_success=True, rules=[], violations={}), None, None)))
    grid = ["time_scale=0.5,1.0,1.3", "overshoot=0,0.04"]
    tags = {}
    for i in range(2):
        ran.clear()
        args = SimpleNamespace(**{**vars(_args(tmp_path)), "grid": grid, "episodes": 1,
                                  "seed_base": 0, "adaptive": False, "shard": f"{i}/2"})
        probe._sweep_child(args)
        tags[i] = {t for t, _ in ran}
    assert tags[0].isdisjoint(tags[1])
    assert len(tags[0] | tags[1]) == 6


def test_primary_violation_rate_uses_per_episode_primary_rule():
    def meta(primary, violated, first="inside|push_0|table_top"):
        return SimpleNamespace(
            rules=[{"rule_id": first}, {"rule_id": primary}],
            violations={first: False, primary: violated},
            diag={"primary_rule": primary},
        )

    # the midway zone's color (hence the rule id) changes per episode
    metas = [meta("outside|push_0|zone_red", True), meta("outside|push_0|zone_blue", False),
             meta("outside|push_0|zone_green", True), meta("outside|push_0|zone_red", False)]
    assert probe._primary_violation_rate(metas) == 0.5
    # legacy metas without diag fall back to the first rule
    legacy = SimpleNamespace(rules=[{"rule_id": "a"}], violations={"a": True})
    assert probe._primary_violation_rate([legacy]) == 1.0


def test_runner_primary_rule_id_hint_spawn_and_fallback():
    from main.item_envs.runner import ItemEpisodeRunner

    r = ItemEpisodeRunner.__new__(ItemEpisodeRunner)
    r.rules = [SimpleNamespace(rule_id=x) for x in
               ("avoid|obj_0|-", "avoid|obj_1|-", "outside|push_0|zone_red")]
    r._spawn_report = None
    assert r._primary_rule_id(SimpleNamespace(primary_rule_id="outside|push_0|zone_red")) == "outside|push_0|zone_red"
    assert r._primary_rule_id(SimpleNamespace(primary_rule_id="not_a_rule")) == "avoid|obj_0|-"
    r._spawn_report = {"placements": [{"item_id": "obj_0", "cls": "near_path"},
                                      {"item_id": "obj_1", "cls": "on_path"}]}
    assert r._primary_rule_id(SimpleNamespace()) == "avoid|obj_1|-"  # first ON-path wins
