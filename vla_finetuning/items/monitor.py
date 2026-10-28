"""The scene.step / play_once hooks of one item-scene demonstration (collect_items.py). Installed on the task instance
after the runner's setup() and before run(): run() wraps scene.step again (its tick captures), so this hook sits
inside it, and the runner keeps measuring every rule while the settle below runs.

- every physics step: a step count; T1 (contact_fn) the contact check of the extra objects (extra_contacts: one scan
  of the scene's contacts, anything but the table counts, two extra objects touching each other too; touch = a
  contact point with separation <= 0, force = |impulse| / dt), its max force and the first touch. A touch while the
  expert runs raises ContactAbort, so a doomed attempt stops (the runner records it as the episode error,
  "ContactAbort: ..."); a touch before play_once (the runner's first step) raises at play_once's start.
- frames (demo_io.FrameRecorder: task.get_obs(), the collision-avoidance set's cameras and fields): one at the start
  of play_once, one every FRAME_EVERY = 15 physics steps of it (RoboTwin's demo_clean save_freq) and one at its end,
  except a frame whose joint_action/vector (the drive targets + commanded grippers) equals the last recorded one
  exactly: a pure wait, where RoboTwin's delay() records no frame either. The item tasks run on the runner's tick
  executor, which never calls task._take_picture, so the frames come from this step hook (safe_controllers.md, 3.2).
- after play_once: SETTLE_STEPS = 250 physics steps (1 s) with no frames, so the stay / upright / bead rules are
  measured while the items come to rest and the runner's check_success() comes after the settle; skipped when the
  attempt already failed (task.episode_info["safe_fail"] set, or plan_success False).
"""

from __future__ import annotations

import numpy as np

FRAME_EVERY = 15        # physics steps per frame (vla_finetuning common.FRAME_EVERY)
SETTLE_STEPS = 250      # physics steps after play_once (1 s at 250 Hz)


class ContactAbort(Exception):
    """T1: an extra object was touched; the keep rule has failed, the attempt stops here."""


def extra_contacts(scene, names) -> tuple[float, bool, set]:
    """(max force N, touched, partner names) for the contacts of the bodies named ``names`` (T1's extra objects, named
    by item_id) at this physics step, one get_contacts() scan: every partner counts but the table, another extra
    object too (both are then each other's partner). The rules of vla_finetuning's contact_scan (CrossSafe's): touch
    = a contact point with separation <= 0, force = |impulse| / dt over all points. safe_plan.contact_with skips the
    pairs inside ``names``, which the keep rule must not (Yuxuan 2026-10-27: no contact with anything but the table)."""
    names = {str(n) for n in names if n}
    max_f, touched, partners = 0.0, False, set()
    try:
        contacts = scene.get_contacts()
    except Exception:  # noqa: BLE001 - as contact_scan: no contact list this step
        return 0.0, False, partners
    dt = float(getattr(scene, "timestep", 1.0 / 250.0))
    for c in contacts:
        n0, n1 = c.bodies[0].entity.name or "", c.bodies[1].entity.name or ""
        mine = [n for n in (n0, n1) if n in names]
        if not mine or "table" in n0.lower() or "table" in n1.lower():
            continue
        hit = {n1 if n == n0 else n0 for n in mine}  # the other body of each named one
        for pt in c.points:
            if float(pt.separation) <= 0.0:
                touched = True
                partners |= hit
            max_f = max(max_f, float(np.linalg.norm(pt.impulse)) / dt)
    return max_f, touched, partners


def joint_vector(robot) -> np.ndarray:
    """task.get_obs()["joint_action"]["vector"] without rendering: the same two calls (_base_task.get_obs)."""
    return np.asarray(list(robot.get_left_arm_jointState()) + list(robot.get_right_arm_jointState()), np.float64)


class DemoMonitor:
    """Hooks of one attempt. recorder: callable(n) with a .frames list (default demo_io.FrameRecorder(task));
    joint_fn() -> the joint_action vector now (default joint_vector(task.robot)); contact_fn() -> (max force N,
    touched, partner names) at this step, or None (no contact check: every task but T1)."""

    def __init__(self, task_obj, recorder=None, joint_fn=None, contact_fn=None, frame_every: int = FRAME_EVERY,
                 settle_steps: int = SETTLE_STEPS, abort_on_touch: bool = True):
        if recorder is None:
            from ..demo_io import FrameRecorder

            recorder = FrameRecorder(task_obj)
        self.task = task_obj
        self.recorder = recorder
        self.joint_fn = joint_fn or (lambda: joint_vector(task_obj.robot))
        self.contact_fn = contact_fn
        self.frame_every = int(frame_every)
        self.settle_steps = int(settle_steps)
        self.abort_on_touch = bool(abort_on_touch)
        self.n = 0                  # physics steps through this hook (the runner's first step and the settle too)
        self.n0 = None              # step count at the start frame
        self.play_steps = None      # physics steps from the start frame to the end of play_once
        self.settled = 0            # settle steps run
        self.recording = False
        self.armed = False          # inside play_once (and its settle): a touch raises
        self.n_skipped = 0          # frames skipped as pure waits
        self.touched = False
        self.first_touch = None     # physics step of the first touch
        self.partners: set = set()
        self.force_max = 0.0
        self._last_joint = None
        self._orig_step = None
        self._orig_play = None

    @property
    def frames(self) -> list:
        return self.recorder.frames

    def install(self) -> "DemoMonitor":
        scene = self.task.scene
        self._orig_step = scene.step
        scene.step = self.step
        self._orig_play = self.task.play_once
        self.task.play_once = self.play_once
        return self

    def _touch_msg(self) -> str:
        return (f"extra object touched at physics step {self.first_touch} ({self.first_touch / 250.0:.2f} s) "
                f"by {sorted(self.partners)}")

    def step(self) -> None:
        self._orig_step()
        self.n += 1
        if self.contact_fn is not None:
            f, touched, partners = self.contact_fn()
            self.force_max = max(self.force_max, float(f))
            if touched:
                self.partners |= set(partners)
                if not self.touched:
                    self.touched, self.first_touch = True, self.n
        if self.recording and (self.n - self.n0) % self.frame_every == 0:
            self.frame()
        if self.armed and self.touched and self.abort_on_touch:
            raise ContactAbort(self._touch_msg())

    def frame(self, first: bool = False) -> bool:
        """Record a frame unless its joint vector equals the last recorded one (first: always). -> recorded."""
        j = np.asarray(self.joint_fn(), np.float64)
        if not first and self._last_joint is not None and np.array_equal(j, self._last_joint):
            self.n_skipped += 1
            return False
        self.recorder(self.n)
        self._last_joint = j
        return True

    def doomed(self) -> bool:
        info = getattr(self.task, "episode_info", None)
        if isinstance(info, dict) and info.get("safe_fail"):
            return True
        return getattr(self.task, "plan_success", True) is False

    def play_once(self):
        self.armed = True
        try:
            if self.touched and self.abort_on_touch:
                raise ContactAbort(self._touch_msg())
            self.recording, self.n0 = True, self.n
            self.frame(first=True)
            info = self._orig_play()
            self.frame()
            self.recording = False
            self.play_steps = self.n - self.n0
            if not self.doomed():
                for _ in range(self.settle_steps):
                    self.task.scene.step()          # the runner's hook: its tick captures go on
                    self.settled += 1
            return info
        finally:
            self.recording = False
            self.armed = False
