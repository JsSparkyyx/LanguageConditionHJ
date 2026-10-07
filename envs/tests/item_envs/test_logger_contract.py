"""Contract tests for main.item_envs.logger (PLAN v3 §5 — HARD contract).

StepRecord / EpisodeMeta field names AND ORDER are pinned verbatim to the
plan; a synthetic EpisodeTrace exercises finalize()'s violations math and
the trace.npz key set. No simulator.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from main.item_envs.logger import (
    EpisodeMeta,
    EpisodeTrace,
    StepRecord,
    item_spec_dict,
    region_dict,
    rule_spec_dict,
    token_grid_meta,
)
from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import Consequence, RuleSpec

# PLAN v3 §5, verbatim. Do not edit without updating the plan.
STEP_FIELDS = (
    "t", "tick", "qpos", "ee_pose", "gripper", "held", "item_pose", "item_vel",
    "particle_pos", "region_T", "margins", "robot_item_dist", "robot_item_force",
    "visibility", "token_masks", "action_source", "time_scale",
    "violation_counts",  # extension (2026-10-07): violating beads per rule, trailing
)
META_FIELDS = (
    "task", "host_task", "embodiment", "seed", "risk", "physics", "items",
    "regions", "rules", "token_grid", "active_rule_id", "task_success",
    "violations", "min_margin", "first_violation_t", "plan_failures",
    "spawn_rejections", "wall_time_s",
    "diag",  # §5 extension: non-fatal diagnostics (held-detection errors), trailing
)

RULES = [
    {"rule_id": "inside|box_0|table_top"},
    {"rule_id": "avoid|box_0|-"},
]


def test_step_record_contract_order():
    assert tuple(f.name for f in dataclasses.fields(StepRecord)) == STEP_FIELDS


def test_episode_meta_contract_order():
    assert tuple(f.name for f in dataclasses.fields(EpisodeMeta)) == META_FIELDS


def _record(tick: int, margins) -> StepRecord:
    return StepRecord(
        t=tick * 0.04,
        tick=tick,
        qpos=np.zeros(14),
        ee_pose=np.zeros((2, 7)),
        gripper=np.zeros(2),
        held=np.array([-1, -1]),
        item_pose=np.zeros((2, 7)),
        item_vel=np.zeros((2, 6)),
        particle_pos={},
        region_T=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
        margins=np.asarray(margins, dtype=np.float64),
        robot_item_dist=np.full(2, np.inf),
        robot_item_force=np.zeros(2),
        visibility=np.zeros((2, 3), dtype=np.float32),
        token_masks=np.zeros((3, 3, 400), dtype=np.uint8),
        action_source=0,
        time_scale=0.6,
    )


def _meta() -> EpisodeMeta:
    return EpisodeMeta(
        task="smoke_items",
        host_task=None,
        embodiment="ur5-wsg",
        seed=0,
        risk={"time_scale": 0.6, "safe_time_scale": 0.6, "task": {}},
        physics={},
        items=[{"item_id": "box_0"}, {"item_id": "box_1"}],
        regions=[],
        rules=[dict(r) for r in RULES],
        token_grid=token_grid_meta(),
        active_rule_id=None,
        task_success=True,
        violations={},
        min_margin={},
        first_violation_t={},
        plan_failures=0,
        spawn_rejections=0,
        wall_time_s=0.0,
    )


def test_finalize_violations_math_and_npz(tmp_path):
    trace = EpisodeTrace(tmp_path)
    trace.add(_record(0, [0.10, 0.20]))
    trace.add(_record(1, [0.05, -0.01]))  # avoid|box_0|- violates at t=0.04
    trace.add(_record(2, [0.03, 0.00]))  # zero is the boundary, not a violation
    meta = _meta()
    npz_path, json_path = trace.finalize(meta)

    assert meta.violations == {"inside|box_0|table_top": False, "avoid|box_0|-": True}
    assert meta.min_margin["inside|box_0|table_top"] == pytest.approx(0.03)
    assert meta.min_margin["avoid|box_0|-"] == pytest.approx(-0.01)
    assert meta.first_violation_t["inside|box_0|table_top"] is None
    assert meta.first_violation_t["avoid|box_0|-"] == pytest.approx(0.04)

    data = np.load(npz_path)
    expected_keys = {
        "t", "tick", "qpos", "ee_pose", "gripper", "held", "item_pose", "item_vel",
        "region_T", "margins", "robot_item_dist", "robot_item_force", "visibility",
        "token_masks", "action_source", "time_scale", "item_ids", "rule_ids",
    }
    assert expected_keys <= set(data.files)
    assert data["margins"].shape == (3, 2)
    assert list(data["rule_ids"]) == [r["rule_id"] for r in RULES]
    assert list(data["item_ids"]) == ["box_0", "box_1"]
    assert data["token_masks"].dtype == np.uint8
    assert data["held"].shape == (3, 2)
    assert data["region_T"].shape == (3, 2, 4, 4)

    payload = json.loads(json_path.read_text())
    assert payload["violations"]["avoid|box_0|-"] is True
    assert "error" not in payload
    # the meta carries every contract field
    for name in META_FIELDS:
        assert name in payload


def test_finalize_empty_trace(tmp_path):
    trace = EpisodeTrace(tmp_path)
    meta = _meta()
    npz_path, json_path = trace.finalize(meta)
    assert npz_path is None
    assert meta.violations == {r["rule_id"]: False for r in RULES}
    assert all(v is None for v in meta.first_violation_t.values())
    # non-finite min margins serialize as null (strict JSON)
    payload = json.loads(json_path.read_text())
    assert payload["min_margin"] == {r["rule_id"]: None for r in RULES}


def test_write_meta_error_key(tmp_path):
    trace = EpisodeTrace(tmp_path)
    path = trace.write_meta(_meta(), error="UnStableError: box_0")
    payload = json.loads(path.read_text())
    assert payload["error"] == "UnStableError: box_0"


def test_spec_dict_views():
    spec = ItemSpec(
        item_id="box_0", entity=object(), category="box", color="red",
        role="candidate", group="push", half_extents=[0.025, 0.025, 0.035], mass=0.1,
    )
    d = item_spec_dict(spec)
    assert d["group"] == "push"
    assert "entity" not in d and "particles" not in d
    assert d["half_extents"] == [0.025, 0.025, 0.035]

    region = Region(
        region_id="zone_green", kind="zone", frame_fn=lambda: np.eye(4),
        polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), z=0.0, z_band=0.01,
        nameable=True, color="green",
    )
    rd = region_dict(region)
    assert rd["nameable"] is True and rd["color"] == "green"
    assert "phrases_key" not in rd  # v2 leftover name

    rule = RuleSpec(
        rule_id="outside|box_0+box_1|zone_green",
        consequence=Consequence.OUTSIDE,
        item_ids=("box_0", "box_1"),  # v3: tuple, NOT item_id
        region_ids=("zone_green",),
        params={"margin": 0.0, "categories": ("box", "box")},
        phrases=["keep the red box out of the green zone"],
        split="train",
    )
    sd = rule_spec_dict(rule)
    assert sd["item_ids"] == ["box_0", "box_1"]
    assert "item_id" not in sd
    assert sd["consequence"] == "outside"
    json.dumps(sd)  # JSON-safe


def test_token_grid_meta_matches_masks():
    from main.item_envs import masks

    meta = token_grid_meta()
    scales, n_tok = masks.token_grid()
    assert meta["n_tok"] == n_tok == 400
    assert meta["cameras"] == list(masks.CAMERA_NAMES)
    assert meta["image_wh"] == [320, 256]
    assert meta["strides"] == [16, 32]
    assert [(s["stride"], s["rows"], s["cols"]) for s in meta["scales"]] == [
        (s, r, c) for s, r, c in scales
    ]
    json.dumps(meta)
