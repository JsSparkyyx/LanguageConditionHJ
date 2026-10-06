"""Tests for robotwin_tasks/_item_utils.py (loaded BY FILE PATH — the
robotwin_tasks/ dir is not a python package; RoboTwin imports these files as
envs.* through the install symlinks).

Non-sim: metadata dict schemas (register_item / register_zone), the risky()
context manager, and the hard rule that robotwin_tasks files never import
CEHJ ``main.*`` (PLAN v3 §2) and carry no module-level sapien import in
_item_utils.py (so non-sim tests can load it headless).

Sim: zone_mat + register_item mass application on a headless scene.
"""

from __future__ import annotations

import ast
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_TASKS = REPO_ROOT / "robotwin_tasks"
ITEM_UTILS_PATH = ROBOTWIN_TASKS / "_item_utils.py"


def load_item_utils():
    """Load robotwin_tasks/_item_utils.py by file path (no package context)."""
    spec = importlib.util.spec_from_file_location("_item_utils_under_test", ITEM_UTILS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# import hygiene (PLAN v3 §2 hard rules)
# ---------------------------------------------------------------------------


def _imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.col_offset, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.col_offset, node.module


@pytest.mark.parametrize("py_file", sorted(ROBOTWIN_TASKS.glob("*.py")), ids=lambda p: p.name)
def test_no_main_import(py_file):
    """robotwin_tasks files must NEVER import CEHJ main.* (PLAN v3 §2)."""
    offenders = [mod for _col, mod in _imports_of(py_file) if mod == "main" or mod.startswith("main.")]
    assert not offenders, f"{py_file.name} imports {offenders}"


def test_item_utils_has_no_module_level_sapien():
    """_item_utils.py stays loadable headless: sapien imports are function-local."""
    top_level = [mod for col, mod in _imports_of(ITEM_UTILS_PATH) if col == 0]
    assert not [m for m in top_level if m.split(".")[0] == "sapien"], top_level


def test_loads_by_file_path_headless():
    mod = load_item_utils()
    for name in ("zone_mat", "risky", "register_item", "register_zone"):
        assert callable(getattr(mod, name))


# ---------------------------------------------------------------------------
# metadata dict schemas (the ONLY RoboTwin -> CEHJ interface)
# ---------------------------------------------------------------------------


def test_register_item_schema():
    iu = load_item_utils()
    task = types.SimpleNamespace()
    entity = object()
    meta = iu.register_item(
        task, "box_0", entity,
        category="box", color="red", kind="rigid", role="candidate",
        group=None, particles=None, half_extents=[0.025, 0.025, 0.035], mass=None,
    )
    assert task.item_meta == [meta]
    assert set(meta) == {
        "item_id", "entity", "category", "color", "kind", "role", "group",
        "particles", "half_extents", "obb_center_local", "up_axis_local", "mass", "size",
    }
    assert meta["size"] is None  # optional size word ("tall" / "short", T5 / T7)
    assert meta["item_id"] == "box_0"
    assert meta["entity"] is entity
    assert meta["half_extents"] == [0.025, 0.025, 0.035]
    assert meta["mass"] is None  # no live PhysX write when mass is None
    # lazily created, then appended
    iu.register_item(task, "box_1", entity, category="box", color="blue",
                     kind="rigid", role="candidate")
    assert [d["item_id"] for d in task.item_meta] == ["box_0", "box_1"]


def test_register_zone_schema():
    iu = load_item_utils()
    task = types.SimpleNamespace()
    entity = object()
    meta = iu.register_zone(task, "zone_green", entity, color="green", size_xy=(0.14, 0.12))
    assert task.region_meta == [meta]
    for key in ("region_id", "entity", "kind", "color", "size_xy", "z", "z_band", "nameable"):
        assert key in meta
    assert meta["kind"] == "zone"
    assert meta["nameable"] is True
    assert meta["size_xy"] == [0.14, 0.12]


# ---------------------------------------------------------------------------
# risky() context manager
# ---------------------------------------------------------------------------


def _task_with_scales():
    return types.SimpleNamespace(
        _time_scale_risky=1.3, _time_scale_safe=0.6, active_time_scale=0.6
    )


def test_risky_flips_and_restores():
    iu = load_item_utils()
    task = _task_with_scales()
    with iu.risky(task):
        assert task.active_time_scale == pytest.approx(1.3)
    assert task.active_time_scale == pytest.approx(0.6)


def test_risky_nesting_restores_previous():
    iu = load_item_utils()
    task = _task_with_scales()
    with iu.risky(task):
        assert task.active_time_scale == pytest.approx(1.3)
        with iu.risky(task, risk=0.9):
            assert task.active_time_scale == pytest.approx(0.9)
        assert task.active_time_scale == pytest.approx(1.3)
    assert task.active_time_scale == pytest.approx(0.6)


def test_risky_explicit_risk_object():
    iu = load_item_utils()
    task = _task_with_scales()
    risk = types.SimpleNamespace(time_scale=1.1)
    with iu.risky(task, risk):
        assert task.active_time_scale == pytest.approx(1.1)
    assert task.active_time_scale == pytest.approx(0.6)


def test_risky_falls_back_without_runner_attrs():
    """Vanilla RoboTwin collection (no runner-installed attrs) leaves scale at 1.0."""
    iu = load_item_utils()
    task = types.SimpleNamespace()
    with iu.risky(task):
        assert task.active_time_scale == pytest.approx(1.0)
    assert task.active_time_scale == pytest.approx(1.0)


def test_risky_restores_on_exception():
    iu = load_item_utils()
    task = _task_with_scales()
    with pytest.raises(RuntimeError):
        with iu.risky(task):
            raise RuntimeError("boom")
    assert task.active_time_scale == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# sim: zone_mat + register_item mass application on a headless scene
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_zone_mat_and_register_item_mass_headless():
    sapien = pytest.importorskip("sapien")
    import sapien.physx as physx

    iu = load_item_utils()
    scene = sapien.Scene([physx.PhysxCpuSystem()])
    scene.set_timestep(1.0 / 250.0)

    mat = iu.zone_mat(
        scene, sapien.Pose([0.1, 0.2, 0.741]), size_xy=(0.14, 0.12), color=(0.0, 0.8, 0.0),
        name="zone_green",
    )
    static = mat.find_component_by_type(physx.PhysxRigidStaticComponent)
    assert static is not None
    shapes = static.get_collision_shapes()
    assert len(shapes) == 1
    half = np.asarray(shapes[0].half_size, dtype=np.float64)
    assert half == pytest.approx([0.07, 0.06, 0.001], abs=1e-6)

    # CPU-only scene: the render-system query must come back false, so NO
    # RenderBodyComponent is attached (attaching one would make add_entity
    # raise "no system with name [render] is added to scene" on GPU nodes)
    import sapien.render

    assert mat.find_component_by_type(sapien.render.RenderBodyComponent) is None

    # pybind11 Entities reject Python attributes, so zone_region must derive
    # the footprint from the collision-shape AABB — and get the builder's size
    from main.item_envs.regions import zone_region

    region = zone_region(mat, "green", region_id="zone_green")
    assert region.polygon[:, 0].max() == pytest.approx(0.07, abs=1e-6)
    assert region.polygon[:, 1].max() == pytest.approx(0.06, abs=1e-6)

    # dynamic box: register_item(..., mass=...) writes through PhysX immediately
    entity = sapien.Entity()
    dyn = physx.PhysxRigidDynamicComponent()
    dyn.attach(
        physx.PhysxCollisionShapeBox(
            half_size=[0.01, 0.01, 0.01],
            material=scene.create_physical_material(0.5, 0.5, 0.0),
        )
    )
    entity.add_component(dyn)
    scene.add_entity(entity)
    task = types.SimpleNamespace()
    meta = iu.register_item(
        task, "box_0", entity, category="box", color="red", kind="rigid",
        role="candidate", half_extents=[0.01, 0.01, 0.01], mass=0.123,
    )
    assert float(dyn.mass) == pytest.approx(0.123, rel=1e-5)
    assert meta["mass"] == pytest.approx(0.123)
