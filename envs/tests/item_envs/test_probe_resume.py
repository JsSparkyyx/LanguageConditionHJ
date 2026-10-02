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
