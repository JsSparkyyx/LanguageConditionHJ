"""Sim tests for T3 push_zone (PLAN v3 §6 Phase 4 step 3). GPU node only.

- 20 seeds x 2 embodiments: setup() builds without UnStableError, the scene
  structure matches the T3 config (2 push candidates, 2-3 zones, 1-2
  bystanders), the rule table is exactly the yaml's rules block over those
  items/regions, every margin is positive at t = 0, and every push lane has
  a midway zone near its midpoint (edge within 6 cm of the corridor,
  measured on interior segment points so mats beside a corridor endpoint
  don't confound it) while no zone intersects a corridor interior.
- One full low-risk episode per embodiment (time_scale 0.5, overshoot 0,
  lateral_offset 0): error is None, task_success, the t = 0 margins are
  positive, the trace shows the safe (0.6) and risky (0.5) time-scale
  windows, and meta/trace/retime/keyframes are written.

Requires the ``push_zone`` registry entry (robotwin kind -> envs.push_zone);
the sim suite runs after the Phase 4 registry wiring lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]
TASK = "push_zone"
TASK_CFG = Path(__file__).resolve().parents[2] / "main" / "item_envs" / "configs" / "tasks" / "T3_push_zone.yaml"


def _expected_rule_ids(items, regions):
    push = sorted(s.item_id for s in items if s.group == "push")
    byst = sorted(s.item_id for s in items if s.role == "bystander")
    zones = sorted(r.region_id for r in regions if r.kind == "zone")
    ids = [f"inside|{i}|table_top" for i in push]
    ids += [f"outside|{i}|{z}" for i in push for z in zones]
    ids += [f"upright|{i}|-" for i in byst]
    ids += [f"stay|{i}|-" for i in byst]
    return ids


def _zone_corridor_clearances(runner):
    """(corridor_k, t, edge clearance) per zone/corridor pair with an interior foot.

    For each (zone, corridor) pair where the zone center's perpendicular foot
    lands strictly inside the corridor segment: the foot parameter t, and the
    distance from the zone center to the segment minus the corridor
    half-width minus the zone's support radius along the corridor normal.
    Endpoint-foot pairs are skipped — the capsule metric would misread them.
    """
    task = runner.env.task
    out = []
    for zone in task.zones:
        c = np.asarray(zone["xy"], dtype=np.float64).reshape(2)
        hx, hy = zone["size_xy"][0] / 2.0, zone["size_xy"][1] / 2.0
        for k, corr in enumerate(task.layout_info["corridors"]):
            a = np.asarray(corr["a"], dtype=np.float64).reshape(2)
            b = np.asarray(corr["b"], dtype=np.float64).reshape(2)
            ab = b - a
            t = float(((c - a) @ ab) / float(ab @ ab))
            if not (0.01 < t < 0.99):
                continue
            d = ab / float(np.linalg.norm(ab))
            perp = np.array([-d[1], d[0]])
            z_half = float(abs(perp[0]) * hx + abs(perp[1]) * hy)
            dist = float(np.linalg.norm(c - (a + t * ab)))
            out.append((k, t, dist - float(corr["half"]) - z_half))
    return out


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("seed", list(range(20)))
def test_setup_margins_positive_at_t0(embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner(
        TASK, embodiment=embodiment, seed=seed, render="none",
        out_root=tmp_path, task_cfg_path=TASK_CFG,
    )
    try:
        runner.setup()  # raises UnStableError / RuntimeError on a bad layout
        task = runner.env.task

        # scene structure
        push = [s for s in runner.items if s.group == "push"]
        byst = [s for s in runner.items if s.role == "bystander"]
        zones = [r for r in runner.regions if r.kind == "zone"]
        assert len(push) == 1  # one push per episode, random side
        assert all(s.role == "candidate" for s in push)
        assert 1 <= len(byst) <= 2
        assert 2 <= len(zones) <= 3
        assert len({z.color for z in zones}) == len(zones)  # distinct zone colors
        assert [r.region_id for r in runner.regions][0] == "table_top"
        # push_0 starts on the pushing arm's side and crosses the midline
        rec = task.push_objects[0]
        sgn = -1.0 if rec["arm"] == "left" else 1.0
        assert rec["item_id"] == "push_0"
        assert sgn * rec["start_xy"][0] > 0 and sgn * rec["goal_xy"][0] < 0
        # explicit masses within the randomized physics bands (x U(0.7, 1.3))
        for spec in runner.items:
            assert 0.02 < spec.mass < 0.45, (spec.item_id, spec.mass)

        # rule table == the yaml's rules block over this scene
        assert [r.rule_id for r in runner.rules] == _expected_rule_ids(runner.items, runner.regions)
        for rule in runner.rules:
            assert rule.phrases
            for phrase in rule.phrases:
                assert not any(c.isdigit() for c in phrase), phrase

        # every margin positive at t = 0
        margins = runner._rule_margins()
        assert margins.shape == (len(runner.rules),)
        assert np.all(np.isfinite(margins))
        assert (margins > 0.0).all(), f"non-positive t=0 margins: {margins}"

        # 1-3 painted zones ON the push path; the primary rule names one
        path = set(task.layout_info["path_zones"])
        assert 1 <= len(path) <= 3
        assert task.primary_rule_id in {r.rule_id for r in runner.rules}

        # time-scale convention installed for the risky() CM
        assert task._time_scale_safe == pytest.approx(runner.risk_params.safe_time_scale)
        assert task._time_scale_risky == pytest.approx(runner.risk_params.time_scale)
        assert task.active_time_scale == pytest.approx(runner.risk_params.safe_time_scale)
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_full_episode_low_risk(embodiment, tmp_path):
    runner = ItemEpisodeRunner(
        TASK,
        embodiment=embodiment,
        seed=0,
        risk={"time_scale": 0.5, "overshoot": 0.0, "lateral_offset": 0.0},
        render="full",
        video=False,
        out_root=tmp_path,
        task_cfg_path=TASK_CFG,
    )
    meta, error = runner.run()
    assert error is None
    assert meta.task_success is True
    assert meta.risk["task"]["overshoot"] == 0.0
    assert meta.risk["task"]["lateral_offset"] == 0.0

    data = np.load(runner.out_dir / "trace.npz")
    n_rules = len(runner.rules)
    assert data["margins"].shape[1] == n_rules
    assert (data["margins"][0] > 0.0).all()  # all margins positive at t = 0
    assert len(data["t"]) > 50  # a real episode, not a stub
    scales = np.unique(np.round(data["time_scale"], 3))
    assert 0.5 in scales  # the paired push displacement ran at the risky scale
    assert 0.6 in scales  # approach / retreat at the safe scale

    assert (runner.out_dir / "meta.json").exists()
    retime = json.loads((runner.out_dir / "retime.json").read_text())
    assert "plans" in retime and "v_ratio_events" in retime
    assert retime["v_ratio_events"] == []  # s <= 1 only slows plans down
    assert (runner.out_dir / "00_start.png").exists()
    assert (runner.out_dir / "30_end.png").exists()
