"""ITEM_TASKS registry (PLAN v3 §4, §2.7): task key -> TaskEntry.

kind is "robotwin" for tasks with their own RoboTwin task file
(``robotwin_task`` = the name ``Env`` imports as ``envs.<robotwin_task>``;
T2 granular_pour, T3 push_zone, and the test-only smoke_items) and "host"
for T1 selective_avoid, which spawns candidates on top of an unmodified
CrossSafe host task (``host_tasks`` = the allowed hosts, chosen per episode
via the runner's ``host`` argument or, when omitted, round-robin by seed). Registered: smoke_items (Phase 2,
test-only) plus the three Phase 4 deliverable tasks T1-T3.

Importing the registry never imports a task module, never imports sapien,
and never triggers the CEHJ bootstrap. ``config`` is the absolute path to
the task YAML under ``main/item_envs/configs/tasks/`` — it may not exist
yet for unregistered Phase 4 tasks; callers check ``.exists()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CONFIG_TASKS = Path(__file__).resolve().parent / "configs" / "tasks"


@dataclass(frozen=True)
class TaskEntry:
    """One registered item-safety task."""

    key: str  # registry key, e.g. "smoke_items" (== config stem)
    kind: str  # "robotwin" | "host"
    robotwin_task: str | None = None  # envs.<robotwin_task> for kind == "robotwin"
    host_tasks: tuple[str, ...] = ()  # allowed CrossSafe hosts for kind == "host"
    embodiments: tuple[str, ...] = ("ur5-wsg", "franka-panda")  # [0] = default
    config: Path | None = None  # absolute path to configs/tasks/<key>.yaml
    test_only: bool = False  # throwaway/debug tasks; excluded from production sweeps

    def __post_init__(self):
        if self.kind not in ("robotwin", "host"):
            raise ValueError(f"unknown task kind: {self.kind!r}")
        if self.kind == "robotwin" and not self.robotwin_task:
            raise ValueError(f"robotwin task {self.key!r} needs robotwin_task")
        if self.kind == "host" and not self.host_tasks:
            raise ValueError(f"host task {self.key!r} needs host_tasks")

    @property
    def default_embodiment(self) -> str:
        return self.embodiments[0]


ITEM_TASKS: dict[str, TaskEntry] = {
    # TEST-ONLY Phase 2 acceptance task (PLAN v3 §6 Phase 2): one zone mat,
    # two candidate boxes. Not a deliverable task — do not sweep in production.
    "smoke_items": TaskEntry(
        key="smoke_items",
        kind="robotwin",
        robotwin_task="smoke_items",
        config=_CONFIG_TASKS / "smoke_items.yaml",
        test_only=True,
    ),
    # Phase 4 deliverable tasks (PLAN v3 §7). T1 rides on the five unmodified
    # CrossSafe hosts (no RoboTwin file); T2/T3 are new RoboTwin task files
    # installed from robotwin_tasks/ (siblings land them in Phase 4).
    "selective_avoid": TaskEntry(
        key="selective_avoid",
        kind="host",
        host_tasks=(
            "place_container_plate",
            "place_burger_fries",
            "stack_blocks_two",
            "stack_bowls_two",
            "place_bread_basket",
        ),
        config=_CONFIG_TASKS / "T1_selective_avoid.yaml",
    ),
    "granular_pour": TaskEntry(
        key="granular_pour",
        kind="robotwin",
        robotwin_task="granular_pour",
        config=_CONFIG_TASKS / "T2_granular_pour.yaml",
    ),
    "push_zone": TaskEntry(
        key="push_zone",
        kind="robotwin",
        robotwin_task="push_zone",
        config=_CONFIG_TASKS / "T3_push_zone.yaml",
    ),
}


def get_task_entry(key: str) -> TaskEntry:
    """The registry entry for a task key (KeyError with a useful message)."""
    try:
        return ITEM_TASKS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a registered item task; registered: {sorted(ITEM_TASKS)}"
        ) from None


def default_embodiment(key: str) -> str:
    """The registry default embodiment for a task (embodiments[0])."""
    return get_task_entry(key).default_embodiment


def production_tasks() -> list[str]:
    """Registered non-test task keys, in registration order."""
    return [key for key, entry in ITEM_TASKS.items() if not entry.test_only]
