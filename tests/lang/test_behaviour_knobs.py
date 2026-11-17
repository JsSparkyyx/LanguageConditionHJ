"""Behaviour-mix keys with drag knobs (config.behaviour_risk; the T8 recovery data, 2026-11-16)."""

import dataclasses

from main.lang.config import LangConfig, behaviour_risk, episode_risk


def test_plain_names_are_unchanged():
    assert behaviour_risk("medium") == {"behaviour": "medium"}
    cfg = LangConfig()
    assert {episode_risk(cfg, "tray_slide", s)["behaviour"] for s in range(200)} == {"medium", "fast", "slow"}
    assert all(len(episode_risk(cfg, "tray_slide", s)) == 1 for s in range(50))


def test_knobs_parse_as_floats():
    assert behaviour_risk("recover_glass|slow_at=0.6|change_dist=0.05") == {
        "behaviour": "recover_glass", "slow_at": 0.6, "change_dist": 0.05}


def test_mix_with_knobs_draws_them_per_seed():
    late = "recover_glass|slow_at=0.6|change_dist=0.05"
    cfg = dataclasses.replace(LangConfig(), behaviour_mix={"tray_slide": {"recover_glass": 1.0, late: 1.0}})
    draws = [episode_risk(cfg, "tray_slide", 41_000_000 + s) for s in range(400)]
    assert draws == [episode_risk(cfg, "tray_slide", 41_000_000 + s) for s in range(400)]
    n_late = sum(d == behaviour_risk(late) for d in draws)
    assert 150 < n_late < 250
    assert all(d["behaviour"] == "recover_glass" for d in draws)
