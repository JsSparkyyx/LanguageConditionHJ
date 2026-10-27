"""The item demos' step / play_once hooks (vla_finetuning/items/monitor.py) on a fake scene and task, driven the way
the item runner drives them (one physics step, then its own scene.step hook around ours, play_once, check_success):

    python -m pytest -q vla_finetuning/tests/test_items_monitor.py
"""

from __future__ import annotations

import numpy as np
import pytest
from _items_fakes import DEFAULT, Contact, Task

import main.lang  # noqa: F401 - main.item_envs on main.__path__
from main.item_envs.safe_plan import contact_with
from vla_finetuning.items.monitor import DemoMonitor, joint_vector


class Rec:
    """FrameRecorder's surface: called with the physics step count, keeps .frames and .steps."""

    def __init__(self, task):
        self.task, self.frames, self.steps = task, [], []

    def __call__(self, n=0):
        self.frames.append({"joint": joint_vector(self.task.robot)})
        self.steps.append(int(n))


def drive(task, mon):
    """The runner's _play: a first step, a capture hook around the installed step (one tick per 10 steps),
    play_once, then check_success. -> (ticks captured after t = 0, error or None)."""
    task.scene.step()
    inner, ticks = task.scene.step, []

    def hooked():
        inner()
        if task.scene.n % 10 == 0:
            ticks.append(task.scene.n)

    task.scene.step = hooked
    try:
        task.play_once()
        task.check_success()
        return ticks, None
    except Exception as exc:  # noqa: BLE001 - as the runner records it
        return ticks, f"{type(exc).__name__}: {exc}"
    finally:
        task.scene.step = inner


def _setup(contact_fn=None, **script):
    task = Task(dict(DEFAULT, **script))
    mon = DemoMonitor(task, recorder=Rec(task), contact_fn=contact_fn).install()
    return task, mon


def test_frames_every_15_steps_and_pure_waits_skipped():
    task, mon = _setup(move1=30, hold=45, move2=30)
    ticks, err = drive(task, mon)
    assert err is None
    # start frame at step 1 (after the runner's first step), then n0 + 15 k: 16 and 31 move the left arm, 46 / 61 /
    # 76 fall in the wait (the same drive targets: skipped), 91 and 106 move the right arm; the end frame (step 106,
    # nothing moved since) is skipped too
    assert mon.recorder.steps == [1, 16, 31, 91, 106]
    assert mon.n_skipped == 4
    assert mon.play_steps == 105
    joints = [f["joint"] for f in mon.frames]
    assert all(not np.array_equal(a, b) for a, b in zip(joints, joints[1:]))
    # the settle: 250 more steps after play_once, no frames, through the runner's hook (its captures go on)
    assert mon.settled == 250 and mon.n == 1 + 105 + 250 and task.scene.n == mon.n
    assert ticks[-1] == 350 and len([k for k in ticks if k > 106]) == 25


def test_end_frame_kept_when_the_targets_moved():
    task, mon = _setup(move1=30, hold=0, move2=7)      # play ends at step 38, 7 steps after the frame at 31
    drive(task, mon)
    assert mon.recorder.steps == [1, 16, 31, 38]


def test_joint_vector_is_the_obs_vector():
    task = Task(dict(DEFAULT))
    task.robot.q[:] = np.arange(14) * 0.1
    assert np.array_equal(joint_vector(task.robot), task.get_obs()["joint_action"]["vector"])


def test_no_settle_after_a_failed_attempt():
    task, mon = _setup(move1=30, hold=15, safe_fail="no_route")
    ticks, err = drive(task, mon)
    assert err is None and task.episode_info["safe_fail"] == "no_route"
    assert mon.settled == 0 and mon.n == 1 + 45


def test_contact_aborts_while_the_expert_runs():
    task, mon = _setup(move1=100, hold=0, move2=100)
    task.scene.contacts[57] = [Contact("obj_0", "finger_left")]
    mon.contact_fn = lambda: contact_with(type("E", (), {"task": task})(), [type("X", (), {"name": "obj_0"})()])
    ticks, err = drive(task, mon)
    assert err is not None and err.startswith("ContactAbort") and "finger_left" in err
    assert mon.touched and mon.first_touch == 57 and mon.partners == {"finger_left"}
    assert mon.n == 57 and mon.settled == 0                # stopped at the touching step
    assert mon.force_max == pytest.approx(0.01 * 250.0)    # |impulse| / dt


def test_contact_with_the_table_or_between_extra_objects_is_no_touch():
    task, mon = _setup(move1=40, hold=0, move2=0)
    task.scene.contacts[10] = [Contact("obj_0", "table"), Contact("obj_0", "obj_1")]
    task.scene.contacts[20] = [Contact("obj_1", "finger_left", separation=0.004)]     # speculative: not a touch
    ents = [type("X", (), {"name": "obj_0"})(), type("X", (), {"name": "obj_1"})()]
    mon.contact_fn = lambda: contact_with(type("E", (), {"task": task})(), ents)
    ticks, err = drive(task, mon)
    assert err is None and not mon.touched and mon.settled == 250


def test_a_touch_before_play_once_aborts_at_its_start():
    task, mon = _setup(move1=30)
    calls = []

    def contact_fn():
        calls.append(task.scene.n)
        return (5.0, True, {"block1"}) if task.scene.n == 1 else (0.0, False, set())

    mon.contact_fn = contact_fn
    ticks, err = drive(task, mon)
    assert err is not None and err.startswith("ContactAbort")
    assert mon.first_touch == 1 and calls == [1] and mon.frames == []    # nothing ran after the runner's first step


def test_abort_can_be_switched_off():
    task = Task(dict(DEFAULT, move1=20, hold=0, move2=0))
    mon = DemoMonitor(task, recorder=Rec(task), contact_fn=lambda: (1.0, True, {"x"}), abort_on_touch=False).install()
    ticks, err = drive(task, mon)
    assert err is None and mon.touched and mon.first_touch == 1 and mon.settled == 250


def test_errors_inside_the_expert_propagate():
    task, mon = _setup(move1=20, render_error_at=1)
    mon.recorder = type("R", (), {"frames": [], "steps": [], "__call__": lambda self, n=0: task.get_obs()})()
    ticks, err = drive(task, mon)
    assert err == "RuntimeError: cannot create buffer" and not mon.armed and not mon.recording
