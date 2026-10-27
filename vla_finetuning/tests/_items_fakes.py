"""Simulator-free stand-ins for ItemEpisodeRunner, its Env and task, which drive vla_finetuning/items on CPU: the
surface collect_items uses (setup / run / close, env.task with scene.step + play_once + get_obs + get_arm_pose,
items, rules, out_dir, _spawn_report) and run()'s contract (one physics step, a t = 0 capture, its own scene.step hook
around whatever is installed, play_once then check_success, every exception recorded as "<type>: <message>", a
trace.npz with a capture every 10 physics steps). Every outcome is scripted per seed (FakeRunner.SCRIPTS)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import main.lang  # noqa: F401 - main.item_envs on main.__path__
from main.item_envs.items import ItemSpec
from main.item_envs.rules import RuleSpec
from vla_finetuning.common import CAMS

HW = (24, 32)
DEG = np.pi / 180.0
DEFAULT = {"move1": 150, "hold": 60, "move2": 150, "success": True, "setup_error": None, "render_error_at": None,
           "safe_fail": None, "plan_success": True, "dip": None, "touch_at": None, "box_color": "blue"}


class ItemTimeout(Exception):
    """The runner's max_steps exception (same name: keep.py reads it by name)."""


class Entity:
    def __init__(self, name):
        self.name = name


class _Body:
    def __init__(self, name):
        self.entity = Entity(name)


class _Point:
    def __init__(self, separation, impulse):
        self.separation, self.impulse = separation, np.asarray(impulse, np.float64)


class Contact:
    """A PhysX contact pair as safe_plan.contact_with reads it (bodies[i].entity.name, points[k].separation/impulse)."""

    def __init__(self, a, b, separation=-0.001, impulse=(0.0, 0.0, 0.01)):
        self.bodies = [_Body(a), _Body(b)]
        self.points = [_Point(separation, impulse)]


class Scene:
    timestep = 1.0 / 250.0

    def __init__(self):
        self.n = 0
        self.contacts: dict = {}        # physics step -> the contacts at that step

    def step(self):
        self.n += 1

    def get_contacts(self):
        return list(self.contacts.get(self.n, []))


class Robot:
    def __init__(self):
        self.q = np.zeros(14)

    def get_left_arm_jointState(self):
        return [float(v) for v in self.q[:7]]

    def get_right_arm_jointState(self):
        return [float(v) for v in self.q[7:]]


class Task:
    """An expert that moves the left arm (a new drive target every physics step), waits (no target change, as
    RoboTwin's delay()), then moves the right arm."""

    def __init__(self, script):
        self.s = script
        self.robot = Robot()
        self.scene = Scene()
        self.episode_info = {}
        self.plan_success = True

    def play_once(self):
        for _ in range(self.s["move1"]):
            self.robot.q[0] += 0.01
            self.scene.step()
        for _ in range(self.s["hold"]):
            self.scene.step()
        if self.s["safe_fail"]:                 # the safe controller found no safe motion: it stops here
            self.episode_info["safe_fail"] = self.s["safe_fail"]
            self.plan_success = False
            return {"info": {}}
        for _ in range(self.s["move2"]):
            self.robot.q[7] += 0.01
            self.scene.step()
        self.plan_success = self.s["plan_success"]
        return {"info": {}}

    def check_success(self):
        return bool(self.s["success"])

    def get_obs(self):
        if self.s["render_error_at"] is not None and self.scene.n >= self.s["render_error_at"]:
            raise RuntimeError("cannot create buffer")
        obs = {"observation": {}, "joint_action": {"vector": self.robot.q.copy()}}
        for i, cam in enumerate(CAMS):
            obs["observation"][cam] = {"rgb": np.full(HW + (3,), 20 * i, np.uint8), "depth": np.full(HW, 500.0 + i),
                                       "intrinsic_cv": np.eye(3),
                                       "extrinsic_cv": np.hstack([np.eye(3), np.zeros((3, 1))])}
        return obs

    def get_arm_pose(self, arm):
        return [0.1 if arm == "left" else -0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]


class FakeRunner:
    """ItemEpisodeRunner for selective_avoid (host stack_blocks_two: a box and an apple as extra objects) and
    gap_place (two flanking blocks). Scripts (DEFAULT keys) per seed in SCRIPTS; every runner built is in BUILT."""

    SCRIPTS: dict = {}
    BUILT: list = []

    def __init__(self, task, embodiment=None, seed=0, host=None, nominal_mode="careless", active_rule_id=None,
                 render="full", out_root=".", max_steps=7500, **kw):
        assert (embodiment, nominal_mode, active_rule_id, render) == ("ur5-wsg", "safe", None, "none")
        self.task_name, self.seed, self.host, self.max_steps = task, int(seed), host, int(max_steps)
        self.out_dir = Path(out_root).resolve() / task / "ur5-wsg" / f"seed_{self.seed}"
        self.s = dict(DEFAULT, **self.SCRIPTS.get(self.seed, {}))
        self.env, self.items, self.rules, self._spawn_report = None, [], [], None
        self.closed = False
        FakeRunner.BUILT.append(self)

    def setup(self):
        task = Task(self.s)
        self.env = SimpleNamespace(task=task, robot=task.robot)
        if self.s["setup_error"]:
            raise RuntimeError(self.s["setup_error"])
        if self.task_name == "selective_avoid":
            self.items = [ItemSpec(item_id="block1", entity=Entity("block1"), category="wooden_block",
                                   role="task_object"),
                          ItemSpec(item_id="obj_0", entity=Entity("obj_0"), category="box", color=self.s["box_color"]),
                          ItemSpec(item_id="obj_1", entity=Entity("obj_1"), category="apple")]
            ids = ("obj_0", "obj_1")
            self.rules = [RuleSpec(f"avoid|{i}|-", "avoid", (i,), (), {"margin": 0.01}) for i in ids]
            self.rules += [RuleSpec(f"stay|{i}|-", "stay", (i,), (), {"d_tol": 0.03, "theta_tol": 20 * DEG})
                           for i in ids]
            self._spawn_report = {"placements": [{"item_id": "obj_0", "cls": "on_path", "category": "box",
                                                  "color": self.s["box_color"], "model_id": None, "xyz": [0, 0, 0]}]}
            if self.s["touch_at"] is not None:
                task.scene.contacts[self.s["touch_at"]] = [Contact("obj_0", "finger_left")]
        elif self.task_name == "gap_place":
            task.standing = [{"item_id": "block_0", "color": "red", "role_in_slot": "flanker"},
                             {"item_id": "block_1", "color": "blue", "role_in_slot": "flanker"}]
            self.items = [ItemSpec(item_id=b["item_id"], entity=Entity(b["item_id"]), category="block",
                                   color=b["color"]) for b in task.standing]
            for i in ("block_0", "block_1"):
                self.rules.append(RuleSpec(f"upright|{i}|-", "upright", (i,), (), {"theta_max": 45 * DEG}))
                self.rules.append(RuleSpec(f"stay|{i}|-", "stay", (i,), (), {"d_tol": 0.02, "theta_tol": 45 * DEG}))
        else:
            raise NotImplementedError(self.task_name)

    def _margins(self, tick: int) -> np.ndarray:
        m = np.array([0.5 if r.consequence.value == "upright" else 0.02 for r in self.rules])   # all safe
        dip = self.s["dip"]                      # (rule_id, from capture tick, margin)
        if dip is not None and tick >= dip[1]:
            m[[r.rule_id for r in self.rules].index(dip[0])] = dip[2]
        return m

    def run(self):
        task = self.env.task
        self.out_dir.mkdir(parents=True, exist_ok=True)
        task.scene.step()                        # the runner's first step, through any installed hook
        t, rows, n = [], [], 0

        def capture():
            t.append(n / 250.0)
            rows.append(self._margins(len(rows)))

        capture()
        orig = task.scene.step

        def hooked():
            nonlocal n
            orig()
            n += 1
            if n > self.max_steps:
                raise ItemTimeout(f"rollout exceeded {self.max_steps} physics steps")
            if n % 10 == 0:
                capture()

        task.scene.step = hooked
        meta = SimpleNamespace(task_success=False, plan_failures=0, diag={}, risk={"time_scale": 0.6})
        err = None
        try:
            task.play_once()
            meta.task_success = bool(task.check_success())
        except Exception as exc:  # noqa: BLE001 - the runner records every episode exception
            err = f"{type(exc).__name__}: {exc}"
        finally:
            task.scene.step = orig
        if task.plan_success is False:
            meta.plan_failures = 1
        meta.diag["task"] = dict(task.episode_info)
        m = np.stack(rows)
        np.savez_compressed(self.out_dir / "trace.npz", t=np.asarray(t), margins=m,
                            violation_counts=np.full(m.shape, -1, np.int64),
                            rule_ids=np.array([r.rule_id for r in self.rules], dtype=str))
        self.close()
        return meta, err

    def close(self):
        self.closed = True
        self.env = None


def install(monkeypatch, scripts: dict) -> None:
    """Point collect_items at FakeRunner with these per-seed scripts (no seeding of torch, no planner check)."""
    from vla_finetuning import common as C
    from vla_finetuning.items import collect_items as CI

    FakeRunner.SCRIPTS = dict(scripts)
    FakeRunner.BUILT = []
    monkeypatch.setattr(CI, "_runner_class", lambda: FakeRunner)
    monkeypatch.setattr(CI, "planner_problem", lambda runner, task: None)
    monkeypatch.setattr(C, "set_global_seeds", lambda seed: None)
    monkeypatch.delenv("STAGE_DEADLINE", raising=False)
    for k in list(os.environ):
        if k.startswith("ITEM_"):
            monkeypatch.delenv(k, raising=False)
