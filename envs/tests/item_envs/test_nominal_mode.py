"""The nominal controller's "safe" mode (Yuxuan 2026-10-22, the expert for safe demonstrations): the task yaml's
safe_mode knobs pinned over the sampled risk knobs; "careless" (the default) leaves them as sampled."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs.risk import sample_risk
from main.item_envs.runner import NOMINAL_MODES, ItemEpisodeRunner, apply_nominal_mode

TASKS = Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs" / "tasks"


def _cfg(name):
    return yaml.safe_load((TASKS / name).read_text())


def _sampled(cfg, seed=3):
    return sample_risk(cfg, np.random.RandomState(seed))


def test_careless_is_the_sampled_knobs():
    cfg = _cfg("T7_gap_place.yaml")
    a, b = _sampled(cfg), _sampled(cfg)
    apply_nominal_mode(b, cfg, "careless", "gap_place")
    assert a.as_dict() == b.as_dict()


def test_safe_t5_drags_slowly_and_t7_slides_centred():
    t5 = _cfg("T5_tray_transport.yaml")
    p = _sampled(t5)
    apply_nominal_mode(p, t5, "safe", "tray_transport")
    assert p.time_scale == 0.5
    t7 = _cfg("T7_gap_place.yaml")
    for seed in range(5):
        q, ref = _sampled(t7, seed), _sampled(t7, seed)
        apply_nominal_mode(q, t7, "safe", "gap_place")
        assert q.task["place_offset"] == 0.0
        assert q.time_scale == 0.5                     # the slide speed pinned too (2026-10-27)
        assert q.safe_time_scale == ref.safe_time_scale


def test_safe_t1_plans_around_the_named_object():
    t1 = _cfg("T1_selective_avoid.yaml")
    assert t1["safe_mode"]["plan"] == "obstacle_aware" and t1["safe_mode"]["protect"] == "named"
    assert t1["safe_mode"]["margin"] == 0.01
    p = _sampled(t1)
    apply_nominal_mode(p, t1, "safe", "selective_avoid")
    assert p.time_scale == 0.6


def test_tasks_without_a_safe_expert_refuse():
    for name, task in (("T2_granular_pour.yaml", "granular_pour"), ("T3_push_zone.yaml", "push_zone")):
        cfg = _cfg(name)
        with pytest.raises(NotImplementedError, match="no safe nominal mode"):
            apply_nominal_mode(_sampled(cfg), cfg, "safe", task)


def test_runner_checks_the_mode():
    assert NOMINAL_MODES == ("careless", "safe")
    assert ItemEpisodeRunner("gap_place", seed=0, render="none").nominal_mode == "careless"
    with pytest.raises(ValueError, match="nominal_mode"):
        ItemEpisodeRunner("gap_place", seed=0, render="none", nominal_mode="slow")
