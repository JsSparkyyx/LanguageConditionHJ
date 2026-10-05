"""Tests for main.item_envs.registry (PLAN v3 §4): TaskEntry kinds + ITEM_TASKS."""

from __future__ import annotations

import pytest

from main.item_envs.registry import (
    ITEM_TASKS,
    TaskEntry,
    default_embodiment,
    get_task_entry,
    production_tasks,
)

T1_HOSTS = (
    "place_container_plate",
    "place_burger_fries",
    "stack_blocks_two",
    "stack_bowls_two",
    "place_bread_basket",
)


def test_registered_tasks_phase4():
    assert set(ITEM_TASKS) == {"smoke_items", "selective_avoid", "granular_pour", "push_zone", "granular_carry", "tray_transport"}


def test_smoke_entry_shape():
    entry = get_task_entry("smoke_items")
    assert entry.kind == "robotwin"
    assert entry.robotwin_task == "smoke_items"
    assert entry.host_tasks == ()
    assert entry.embodiments == ("ur5-wsg", "franka-panda")
    assert entry.default_embodiment == "ur5-wsg"
    assert entry.test_only is True
    assert entry.config is not None and entry.config.exists(), entry.config
    assert entry.config.name == "smoke_items.yaml"


def test_selective_avoid_entry_is_host_kind():
    entry = get_task_entry("selective_avoid")
    assert entry.kind == "host"
    assert entry.robotwin_task is None
    assert entry.host_tasks == T1_HOSTS
    assert entry.embodiments == ("ur5-wsg", "franka-panda")
    assert entry.test_only is False
    assert entry.config is not None and entry.config.exists(), entry.config
    assert entry.config.name == "T1_selective_avoid.yaml"


@pytest.mark.parametrize(
    "key,robotwin_task,yaml_name",
    [
        ("granular_pour", "granular_pour", "T2_granular_pour.yaml"),
        ("push_zone", "push_zone", "T3_push_zone.yaml"),
    ],
)
def test_robotwin_phase4_entries(key, robotwin_task, yaml_name):
    entry = get_task_entry(key)
    assert entry.kind == "robotwin"
    assert entry.robotwin_task == robotwin_task
    assert entry.host_tasks == ()
    assert entry.test_only is False
    assert entry.config is not None and entry.config.exists(), entry.config
    assert entry.config.name == yaml_name


def test_default_embodiment_helper():
    assert default_embodiment("smoke_items") == "ur5-wsg"
    assert default_embodiment("selective_avoid") == "ur5-wsg"
    assert default_embodiment("granular_pour") == "ur5-wsg"
    assert default_embodiment("push_zone") == "ur5-wsg"


def test_get_task_entry_unknown_key():
    with pytest.raises(KeyError, match="not a registered item task"):
        get_task_entry("place_empty_cup")  # a CrossSafe host, not an item task


def test_production_tasks_excludes_test_only():
    assert production_tasks() == ["selective_avoid", "granular_pour", "push_zone", "granular_carry", "tray_transport"]


def test_entry_validation():
    with pytest.raises(ValueError, match="unknown task kind"):
        TaskEntry(key="x", kind="weird")
    with pytest.raises(ValueError, match="needs robotwin_task"):
        TaskEntry(key="x", kind="robotwin")
    with pytest.raises(ValueError, match="needs host_tasks"):
        TaskEntry(key="x", kind="host")
    host = TaskEntry(key="t1_like", kind="host", host_tasks=("place_empty_cup",))
    assert host.robotwin_task is None
