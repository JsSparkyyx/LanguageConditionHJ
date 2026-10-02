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


def test_only_smoke_registered_in_phase2():
    assert set(ITEM_TASKS) == {"smoke_items"}


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


def test_default_embodiment_helper():
    assert default_embodiment("smoke_items") == "ur5-wsg"


def test_get_task_entry_unknown_key():
    with pytest.raises(KeyError, match="not a registered item task"):
        get_task_entry("push_zone")  # lands in Phase 4


def test_production_tasks_excludes_test_only():
    assert production_tasks() == []


def test_entry_validation():
    with pytest.raises(ValueError, match="unknown task kind"):
        TaskEntry(key="x", kind="weird")
    with pytest.raises(ValueError, match="needs robotwin_task"):
        TaskEntry(key="x", kind="robotwin")
    with pytest.raises(ValueError, match="needs host_tasks"):
        TaskEntry(key="x", kind="host")
    host = TaskEntry(key="t1_like", kind="host", host_tasks=("place_empty_cup",))
    assert host.robotwin_task is None
