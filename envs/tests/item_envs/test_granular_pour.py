"""Non-sim tests: the T2 granular_pour rule table and task-file/config contract.

Mirrors the robotwin_tasks/granular_pour.py scene (one ``beads`` particle_set
candidate, bowl_src task_object, container interiors bowl_src_interior /
target_interior, zone mats zone_red / zone_blue) with mock items/regions and
enumerates rules from the REAL configs/tasks/T2_granular_pour.yaml +
objects.yaml + splits.yaml. Also pins the task-file module constants against
the yaml (they are two copies of the same numbers by design — the task file
owns the layout, the yaml documents it — and this test is the drift alarm).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from main.item_envs.items import ItemSpec
from main.item_envs.regions import Region
from main.item_envs.rules import enumerate_rules, resolve_splits

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "main" / "item_envs" / "configs"
TASK_SRC = ROOT / "robotwin_tasks" / "granular_pour.py"

EXPECTED_RULE_IDS = [
    "particles_inside|beads|bowl_src_interior+target_interior",
]


def _load(name):
    with open(CONFIGS / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mock_pour_scene():
    """beads particle_set candidate + the four T2 regions (two containers, two mats)."""
    items = {
        "beads": ItemSpec(
            item_id="beads", entity=object(), category="beads", color="",
            kind="particle_set", particles=[object() for _ in range(4)], role="candidate",
        ),
        "bowl_src": ItemSpec(
            item_id="bowl_src", entity=object(), category="bowl", color="", role="task_object",
        ),
    }
    disk = np.array([[0.05 * np.cos(t), 0.05 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
    rect = np.array([[-0.07, -0.05], [0.07, -0.05], [0.07, 0.05], [-0.07, 0.05]])
    regions = {
        "bowl_src_interior": Region(
            region_id="bowl_src_interior", kind="container", frame_fn=lambda: np.eye(4),
            polygon=disk, z=0.0, z_band=0.01, nameable=True,
        ),
        "target_interior": Region(
            region_id="target_interior", kind="container", frame_fn=lambda: np.eye(4),
            polygon=disk, z=0.0, z_band=0.01, nameable=True,
        ),
        "zone_red": Region(
            region_id="zone_red", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=rect, z=0.0, z_band=0.01, nameable=True, color="red",
        ),
        "zone_blue": Region(
            region_id="zone_blue", kind="zone", frame_fn=lambda: np.eye(4),
            polygon=rect, z=0.0, z_band=0.01, nameable=True, color="blue",
        ),
    }
    return items, regions


def _enumerate():
    items, regions = mock_pour_scene()
    task_cfg = _load("tasks/T2_granular_pour.yaml")
    return enumerate_rules(
        items,
        regions,
        task_cfg,
        objects_cfg=_load("objects.yaml"),
        splits_cfg=resolve_splits(_load("splits.yaml"), "granular_pour"),
    )


def test_t2_cfg_shape():
    task_cfg = _load("tasks/T2_granular_pour.yaml")
    for key in (
        "task", "robotwin_task", "embodiments", "bimanual", "bowl_model",
        "bowl_ids", "beads", "target", "risk", "rules", "risk_collect",
    ):
        assert key in task_cfg, f"T2 yaml missing {key}"
    assert task_cfg["task"] == "granular_pour"
    assert task_cfg["robotwin_task"] == "granular_pour"
    assert task_cfg["embodiments"][0] == "ur5-wsg"
    assert "franka-panda" in task_cfg["embodiments"]
    assert task_cfg["bimanual"] is False
    assert task_cfg["risk"]["safe_time_scale"] == 0.6
    assert task_cfg["risk"]["time_scale"]["dist"] == "choice"
    assert task_cfg["risk"]["task"]["tilt_rate"]["range"] == [1.0, 4.0]
    assert task_cfg["risk"]["task"]["pour_height"]["range"] == [0.05, 0.20]
    assert task_cfg["risk"]["task"]["pour_offset"]["range"] == [0.0, 0.05]


def test_t2_rule_table_order_and_ids():
    rules = _enumerate()
    assert [r.rule_id for r in rules] == EXPECTED_RULE_IDS
    # deterministic: a second enumeration is identical
    assert [r.rule_id for r in _enumerate()] == EXPECTED_RULE_IDS


def test_t2_rule_semantics():
    rules = _enumerate()
    by_id = {r.rule_id: r for r in rules}
    for rid, rule in by_id.items():
        assert rule.consequence.value == rid.split("|")[0]
        assert rule.item_ids == ("beads",), rid
        assert rule.phrases, f"{rid} rendered no phrases"
        for phrase in rule.phrases:
            assert not any(c.isdigit() for c in phrase), phrase
        assert rule.split in ("train", "heldout_item", "heldout_region", "heldout_phrase")
    # the joint particles_inside rule spans BOTH container interiors
    assert by_id["particles_inside|beads|bowl_src_interior+target_interior"].region_ids == (
        "bowl_src_interior", "target_interior",
    )
    # no forbidden-mat rules (user decision 2026-10-03)
    assert not any(r.consequence.value == "particles_outside" for r in rules)
    # softmin temperature: the 0.005 default over 60 beads pulled the t=0 rule
    # margin negative with every bead inside the bowl (p4-accept failures) —
    # the yaml pins T = 0.001 on all three entries
    for rid in EXPECTED_RULE_IDS:
        assert by_id[rid].params["T"] == 0.001, rid


def test_t2_task_file_matches_yaml():
    """The task-file module constants and the yaml must tell the same story."""
    src = TASK_SRC.read_text(encoding="utf-8")
    cfg = _load("tasks/T2_granular_pour.yaml")

    beads = cfg["beads"]
    assert f"N_BEADS = {beads['n']}" in src
    assert f"BEAD_R = {beads['radius']}" in src
    lo, hi = beads["mass"]
    assert f"BEAD_MASS_RANGE = ({lo}, {hi})" in src

    ids = ", ".join(str(i) for i in cfg["bowl_ids"])
    assert f"BOWL_IDS = ({ids})" in src
    assert cfg["bowl_model"] == "002_bowl"

    assert f"BIMANUAL = {cfg['bimanual']}" in src  # False vs False

    # pour knob defaults + tilt range
    assert "POUR_HEIGHT_DEFAULT = 0.10" in src
    assert "POUR_OFFSET_DEFAULT = 0.0" in src
    assert "TILT_RATE_DEFAULT = 2.0" in src
    assert "TILT_ANGLE_DEG = (100.0, 120.0)" in src

    assert "ZONE_DEFS" not in src  # no forbidden mats (user decision 2026-10-03)


# ---------------------------------------------------------------------------
# p4-accept regression seams (pure numpy, login-node safe)
# ---------------------------------------------------------------------------

_ITEM_UTILS_PATH = ROOT / "robotwin_tasks" / "_item_utils.py"
ASSET_OBJ_ROOT = Path("/work/nvme/bifq/project/RoboTwin/assets/objects")


def _item_utils():
    """Load robotwin_tasks/_item_utils.py by file path (no package import)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_item_utils", _ITEM_UTILS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not ASSET_OBJ_ROOT.is_dir(), reason="RoboTwin assets not on this host")
def test_container_region_margin_contract():
    """Resting beads sit comfortably inside the registered container polygon.

    p4-accept regression: with the rim-OPENING radius the belly (up to 5.3 mm
    wider) plus the T-softmin over the bead set put the joint particles_inside
    margin at -1.6..-5.1 mm at t=0. The registration now uses the OUTER
    footprint radius + 2 mm (container_region_radius) and T = 0.001: the
    softmin over the spawn grid must clear +8 mm per bowl id, and survive a
    simulated 1.5 mm settle creep of the outer ring. Runs at the current
    default N=100 (fills the bowl to ~one layer below the rim — the top ring
    projects wider than the 60-bead grid, so the gate is tighter there).
    """
    from main.item_envs.geometry import circle_polygon
    from main.item_envs.margins import m_particles_inside

    iu = _item_utils()
    for bid in (1, 2, 3, 5):
        geom = iu.bowl_geometry("002_bowl", bid, obj_root=str(ASSET_OBJ_ROOT))
        radius = iu.container_region_radius(geom, pad=0.002)
        poly = circle_polygon(radius, 16)
        region = Region(
            region_id=f"bowl_{bid}", kind="container", frame_fn=lambda: np.eye(4),
            polygon=poly, z=0.0, z_band=0.01, nameable=True,
        )
        pos = np.asarray(
            iu.bead_grid_positions(100, 0.006, geom["inner_radius"], geom["floor_y"],
                                   bands=geom["bands"], rng=np.random)
        )
        margin = m_particles_inside([pos[:, [0, 2]]], [region], T=0.001)
        assert margin > 0.008, (bid, margin)
        # settle creep: the outer ring rolls up to 1.5 mm outward against the wall
        rho = np.hypot(pos[:, 0], pos[:, 2])
        crept = pos.copy()
        outer = rho > 0.7 * rho.max()
        crept[outer, 0] *= (rho[outer] + 0.0015) / rho[outer]
        crept[outer, 2] *= (rho[outer] + 0.0015) / rho[outer]
        margin_crept = m_particles_inside([crept[:, [0, 2]]], [region], T=0.001)
        assert margin_crept > 0.004, (bid, margin_crept)


def test_zone_keepaway_helper_semantics():
    """rect_support_radius is the exact rect support; zone_clear_ok enforces it."""
    iu = _item_utils()
    size = (0.16, 0.14)
    assert iu.rect_support_radius(size, (1.0, 0.0)) == pytest.approx(0.08)
    assert iu.rect_support_radius(size, (0.0, 1.0)) == pytest.approx(0.07)
    diag = 2 ** -0.5
    assert iu.rect_support_radius(size, (diag, diag)) == pytest.approx(0.5 * (diag * 0.16 + diag * 0.14))
    # a mat 3 cm (edge) from a bowl of outer radius 6 cm passes; 1 cm fails
    bowl = [(np.array([0.0, 0.0]), 0.06)]
    assert iu.zone_clear_ok(np.array([0.06 + 0.08 + 0.03, 0.0]), size, bowl, 0.02)
    assert not iu.zone_clear_ok(np.array([0.06 + 0.08 + 0.01, 0.0]), size, bowl, 0.02)
    # approach from the diagonal needs the diagonal support
    d = diag * (0.06 + 0.5 * (diag * 0.16 + diag * 0.14) + 0.025)
    assert iu.zone_clear_ok(np.array([d, d]), size, bowl, 0.02)
    d2 = diag * (0.06 + 0.5 * (diag * 0.16 + diag * 0.14) + 0.005)
    assert not iu.zone_clear_ok(np.array([d2, d2]), size, bowl, 0.02)


def test_pour_height_floor_and_tilt_guards():
    """The rim-ring interlink floor and the tilt abort guard are in place."""
    iu = _item_utils()
    src = TASK_SRC.read_text(encoding="utf-8")
    # floor: src outer radius + 1.5 cm clearance, inside the yaml knob range
    if ASSET_OBJ_ROOT.is_dir():
        for bid in (1, 2, 3, 5):
            geom = iu.bowl_geometry("002_bowl", bid, obj_root=str(ASSET_OBJ_ROOT))
            floor = iu.min_pour_height(geom, clearance=0.015)
            assert 0.05 < floor <= 0.20, (bid, floor)
    # play_once clamps the knob; the tilt aborts on a lost pin; quat continuity
    assert "min_pour_height(self._src_geom, clearance=POUR_HEIGHT_CLEAR)" in src
    assert "POUR_HEIGHT_CLEAR = 0.015" in src
    assert "POUR_PIN_ABORT = 0.03" in src
    assert "np.dot(q_new, ee[3:])) < 0.0" in src
    # registration uses the footprint-column radius, not the rim opening
    assert "rim_radius=container_region_radius(self._src_geom" in src
    assert "rim_radius=container_region_radius(self._target_geom" in src
    # no forbidden mats any more (user decision 2026-10-03)
    assert "zone_mat(" not in src
