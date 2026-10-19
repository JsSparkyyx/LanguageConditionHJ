"""Shared pieces of the safe-demonstration experiment (PLAN.md): the CrossSafe scene, the obstacle-aware
cuRobo world, a physics-step monitor (collision metrics every 10 steps, frames every 15), the instruction
text, and the episode metrics. Every scene, demonstrator and policy row goes through these functions, so
the rows differ only in who drives the robot.

Run everything from the repository root inside the container (scripts/lang/in_container.sh); Env's
prepare() chdirs to the RoboTwin root, so every output path here is absolute.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

VF = Path(__file__).resolve().parent
REPO = VF.parent
RL = Path(os.environ.get("ROBO_ORCHARD_LAB", "/work/nvme/bifq/project/RoboOrchardLab"))
CK = Path(os.environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD"))
for _p in (str(REPO), str(RL)):
    if _p not in sys.path:
        sys.path.append(_p)

# eval_ckpt.ALL_TASKS order: the task index of the evaluation seed grid
TASKS = ["place_container_plate", "place_burger_fries", "stack_blocks_two", "stack_bowls_two",
         "place_bread_basket"]
EMBODIMENT = "ur5-wsg"
EMB_IDX = 3                 # eval_ckpt.ALL_EMBODIMENTS.index("ur5-wsg")
OBSTACLE_MODEL = "068_boxdrink"
PHYS_HZ = 250
TICK_EVERY = 10             # physics steps per metric sample (CrossSafe's 25 Hz control tick)
FRAME_EVERY = 15            # physics steps per recorded frame (RoboTwin demo_clean save_freq)
FORCE_N = 30.0              # eval collision: contact force >= 30 N (eval_ckpt FORCE_VIOL_N)
KEEP_FORCE_N = 20.0         # demo keep rule: contact force < 20 N (CrossSafe contact_force_threshold)
CAMS = ("left_camera", "right_camera", "head_camera")   # HoloBrain robotwin2_0_ur5_wsg cam_names
SEED_BASE = 70000           # eval_ckpt default --seed-base
INFERENCE_PREFIX = "robotwin2_0_ur5_wsg"


class SimTimeout(Exception):
    """The physics cap of an episode was reached."""


class NoScene(Exception):
    """No valid scene for this seed (unstable setup or no obstacle spawn)."""


def set_global_seeds(seed: int) -> None:
    from main.train.eval_ckpt import set_global_seeds as _set

    _set(int(seed))


def episode_seed(task: str, ep: int, seed_base: int = SEED_BASE) -> int:
    from main.train.eval_ckpt import episode_seed as _seed

    return _seed(int(seed_base), TASKS.index(task), EMB_IDX, int(ep))


# ---------------------------------------------------------------- scene

def build_scene(task: str, seed: int):
    """CrossSafe's evaluation scene for (task, seed): Env with ur5-wsg 0.8 m apart, the pick objects
    stretched, the on-path box spawned, cuRobo's world reset to RoboTwin's table (make_training_env).
    Raises NoScene when the seed does not give one."""
    from main.train.collect import make_training_env
    from main.train.config import FrozenConfig

    cfg = FrozenConfig(task=task, embodiment=EMBODIMENT, obstacle_mode="on_path",
                       obstacle_model=OBSTACLE_MODEL)
    try:
        env = make_training_env(cfg, int(seed))
    except Exception as exc:          # UnStableError and friends: the seed has no valid layout
        if type(exc).__name__ in ("UnStableError",):
            raise NoScene(f"seed {seed}: {type(exc).__name__}: {exc}") from exc
        raise
    if env.obstacle is None:
        env.close()
        raise NoScene(f"seed {seed}: no obstacle spawn")
    return env


def eval_scene(task: str, ep: int, seed_base: int = SEED_BASE):
    """Scene ep of the evaluation grid, with eval_ckpt's redraw offsets (5 retries of +1000 inside 10
    attempts of +10M) when a seed has no valid scene. -> (env, used_seed)."""
    seed = episode_seed(task, ep, seed_base)
    for attempt in range(10):
        s0 = seed + attempt * 10_000_000
        for retry in range(5):
            s = s0 + retry * 1000
            set_global_seeds(s)
            try:
                return build_scene(task, s), s
            except NoScene as exc:
                print(f"[scene] {task} ep{ep}: {exc}; re-drawing")
    raise NoScene(f"{task} ep{ep}: no valid scene after 50 seeds")


def standard_scene(task: str, ep: int, seed_base: int = SEED_BASE):
    """A plain RoboTwin layout (no stretching of the pick objects, no box) on the same Env (ur5-wsg 0.8 m
    apart, demo_clean cameras): the pipeline sanity check, where the base model should do what it does on
    RoboTwin. Same seed grid and redraws as eval_scene. -> (env, used_seed)."""
    import main.envs.obstacle as obstacle_mod
    from main.envs.controller import CuroboIKController
    from main.envs.env import Env

    from main.train.collect import ARM_DISTANCE

    seed = episode_seed(task, ep, seed_base)
    for attempt in range(10):
        for retry in range(5):
            s = seed + attempt * 10_000_000 + retry * 1000
            set_global_seeds(s)
            CuroboIKController.install()
            keep = obstacle_mod.stretch_task_spawns
            obstacle_mod.stretch_task_spawns = lambda env: None      # Env.__init__ imports it at call time
            try:
                env = Env(task, EMBODIMENT, s, arm_distance=ARM_DISTANCE[EMBODIMENT])
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ != "UnStableError":
                    raise
                print(f"[scene] {task} ep{ep}: seed {s} unstable; re-drawing")
                continue
            finally:
                obstacle_mod.stretch_task_spawns = keep
            env.obstacle = env.obstacle_xyz = env.obstacle_half = None
            return env, s
    raise NoScene(f"{task} ep{ep}: no stable standard scene")


def obstacle_pose(env):
    """(centre xyz, quat wxyz, half extents in the model frame) of the spawned box, as h uses it."""
    ent = getattr(env.obstacle, "actor", env.obstacle)
    return (np.asarray(env.obstacle_xyz, np.float64), np.asarray(ent.get_pose().q, np.float64),
            np.asarray(env.obstacle_half, np.float64))


def step_limit(task: str) -> int:
    """RoboTwin's evaluation action limit for the task (env_cfg/task_config/_eval_step_limit.yml)."""
    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    data = yaml.safe_load((Path(CONFIGS_PATH) / "_eval_step_limit.yml").read_text())
    return int(data.get(task, 1000))


# ---------------------------------------------------------------- cuRobo world

def require_curobo(robot) -> None:
    """Both arms must plan with cuRobo in this process: the mplib fallback (cuRobo init failure) builds
    its world before the box exists and would plan straight through it."""
    from envs.robot.planner import CuroboPlanner

    if getattr(robot, "communication_flag", False):
        raise RuntimeError("cuRobo planners run in subprocesses: the world cannot be updated")
    for side in ("left", "right"):
        p = getattr(robot, f"{side}_planner", None)
        if not (isinstance(p, CuroboPlanner) and hasattr(p, "motion_gen") and hasattr(p, "motion_gen_batch")):
            raise RuntimeError(f"{side} planner is {type(p).__name__}, not cuRobo: the box would be ignored")


def inflated_box(xyz, quat, half, margin: float, top_extra: float):
    """The box grown by `margin` on every side and by `top_extra` more on top (in world z), for the
    planner only. The standing box's model axes are not world axes, so the extra height goes into the
    world-frame centre and the half extent along whichever model axis is vertical."""
    import transforms3d as t3d

    R = t3d.quaternions.quat2mat(np.asarray(quat, np.float64))
    half = np.asarray(half, np.float64) + float(margin)
    up = int(np.argmax(np.abs(R[2, :])))        # model axis closest to world z
    half = half.copy()
    half[up] += 0.5 * float(top_extra)
    centre = np.asarray(xyz, np.float64) + np.array([0.0, 0.0, 0.5 * float(top_extra)])
    return centre, np.asarray(quat, np.float64), half


def push_world(robot, box=None, margin: float = 0.0, top_extra: float = 0.0, table: bool = True) -> None:
    """Write RoboTwin's table (table=True) plus (optionally) the inflated box into both MotionGens of both
    arms, each in its arm's base frame (CrossSafe's _obstacle_in_base). box = obstacle_pose(env) or None
    (table only: the careless nominal, and the per-episode reset of the cached MotionGens). Raises on any
    failure, and checks the box slot is enabled in every collision checker it was written to.

    table=False is the safe demonstrator's world (the box only). RoboTwin's table cuboid is placed with a
    base-frame formula that misses the real table: for the left ur5 arm it lies beside the table, for the
    right arm it covers x in [-0.6, 0.1] with its top 2 cm ABOVE the real top (z 0.76). The stock model
    never notices (it ends at ee_link); with the hand and the held object modelled, every grasp and every
    object resting on the table starts "in collision" there (smoke B2, 10-15: INVALID_START_STATE_WORLD_COLLISION)."""
    from curobo.geom.types import WorldConfig
    from main.envs.obstacle import OBSTACLE_NAME, _obstacle_in_base, _table_world_dict

    require_curobo(robot)
    for planner in (robot.left_planner, robot.right_planner):
        world = _table_world_dict(planner) if table else {"cuboid": {}}
        if box is not None:
            c, q, h = inflated_box(*box, margin=margin, top_extra=top_extra)
            world["cuboid"][OBSTACLE_NAME] = _obstacle_in_base(planner, c, q, h)
        cfg = WorldConfig.from_dict(world)
        for mg in (planner.motion_gen, planner.motion_gen_batch):
            mg.update_world(cfg)
            if box is not None:
                _check_enabled(mg, OBSTACLE_NAME)


def _check_enabled(mg, name: str) -> None:
    wcc = mg.world_coll_checker
    names = wcc._env_obbs_names[0]
    if name not in names:
        raise RuntimeError(f"cuRobo world has no cuboid {name!r} after update_world")
    idx = names.index(name)
    n = int(wcc._env_n_obbs[0])
    enabled = int(wcc._cube_tensor_list[2][0, idx])
    if not (idx < n and enabled == 1):
        raise RuntimeError(f"cuRobo cuboid {name!r} slot {idx} not enabled (n={n}, flag={enabled})")


# ---------------------------------------------------------------- physics monitor

def contact_scan(env) -> tuple[float, bool, set]:
    """(max force N, touched, names touching) for contacts of the box with anything but the table at
    this physics step: CrossSafe's _robot_obstacle_contact_force (main/envs/distance.py), same rules
    (touch = a contact point with separation <= 0; force = |impulse| / dt), plus the partner names."""
    from main.envs.obstacle import OBSTACLE_NAME

    max_f, touched, names = 0.0, False, set()
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:  # noqa: BLE001
        return 0.0, False, names
    dt = float(getattr(env.task.scene, "timestep", 1.0 / PHYS_HZ))
    for c in contacts:
        n0, n1 = c.bodies[0].entity.name or "", c.bodies[1].entity.name or ""
        if OBSTACLE_NAME not in (n0, n1):
            continue
        other = n0 if n1 == OBSTACLE_NAME else n1
        if other == OBSTACLE_NAME or "table" in other.lower():
            continue
        for p in c.points:
            if float(p.separation) <= 0.0:
                touched = True
                names.add(other)
            f = float(np.linalg.norm(p.impulse)) / dt
            if f > max_f:
                max_f = f
    return max_f, touched, names


class PhysicsMonitor:
    """Wraps task.scene.step for one episode.

    Every physics step: the box contact (contact_scan), kept as a running any-touch / max-force and the
    set of touching bodies (a graze shorter than one 10-step tick is not missed; review 10-15, planner F3).
    Every TICK_EVERY steps, a sample as CrossSafe's eval records it: h (metres; robot collision spheres +
    held object vs the box, main.train.hfunc.compute_h), that step's force and touch, whether an object is
    held, and the first time the task succeeds (RoboTwin's latched eval_success; with poll_success, also
    check_success() itself, for the planner rows, whose play_once never sets the latch).
    Every FRAME_EVERY steps it calls on_frame(n) (test videos). Past cap_steps it raises SimTimeout from
    inside the step, as CrossSafe's rollout does."""

    def __init__(self, env, cap_steps: int, on_frame=None, poll_success: bool = False):
        self.env = env
        self.cap = int(cap_steps)
        self.on_frame = on_frame
        self.poll_success = bool(poll_success)
        self.n = 0
        self.t_success = None
        self.touch_any = False
        self.force_max = 0.0
        self.partners: set = set()
        self.trace = {"t": [], "h": [], "force": [], "touch": [], "holding": []}
        self._orig = None

    def __enter__(self):
        scene = self.env.task.scene
        self._orig = scene.step

        def step():
            self._orig()
            self.n += 1
            f, touch, names = contact_scan(self.env)
            if touch:
                self.touch_any = True
                self.partners |= names
            if f > self.force_max:
                self.force_max = f
            if self.n % TICK_EVERY == 0:
                self.sample(f, touch)
            if self.on_frame is not None and self.n % FRAME_EVERY == 0:
                self.on_frame(self.n)
            if self.n >= self.cap:
                raise SimTimeout(f"physics cap {self.cap} steps ({self.cap / PHYS_HZ:.0f} s)")

        scene.step = step
        return self

    def __exit__(self, *exc):
        self.env.task.scene.step = self._orig
        return False

    def sample(self, force=None, touch=None) -> None:
        from main.train.hfunc import compute_h

        h, diag = compute_h(self.env)
        if force is None:
            force, touch, names = contact_scan(self.env)
            self.partners |= names if touch else set()
            self.touch_any |= bool(touch)
            self.force_max = max(self.force_max, float(force))
        self.trace["t"].append(self.n / PHYS_HZ)
        self.trace["h"].append(float(h))
        self.trace["force"].append(float(force))
        self.trace["touch"].append(bool(touch))
        self.trace["holding"].append(bool((diag or {}).get("holding")))
        if self.t_success is None:
            ok = bool(getattr(self.env.task, "eval_success", False))
            if not ok and self.poll_success:
                try:
                    ok = bool(self.env.task.check_success())
                except Exception:  # noqa: BLE001
                    ok = False
            if ok:
                self.t_success = self.n / PHYS_HZ


def robot_link_names(env) -> set:
    from main.envs.distance import _robot_link_names

    return set(_robot_link_names(env.robot))


def episode_metrics(mon: PhysicsMonitor) -> dict:
    """Per-episode collision numbers. `collision` / `contact` / `force*` use every physics step;
    `*_tick` are CrossSafe's eval definitions on the 10-step samples (comparable with its tables).
    contact_robot / contact_object split the touching bodies into the robot (and anything it holds is
    reported in `holding`) vs other scene objects pushed into the box."""
    h = np.asarray(mon.trace["h"], np.float64)
    f = np.asarray(mon.trace["force"], np.float64)
    touch = np.asarray(mon.trace["touch"], bool)
    n = len(h)
    if n == 0:
        return {"n_ticks": 0}
    links = robot_link_names(mon.env)
    partners = sorted(mon.partners)
    return {
        "n_ticks": n,
        "duration_s": mon.n / PHYS_HZ,
        "collision": bool(mon.touch_any or mon.force_max >= FORCE_N),
        "contact": bool(mon.touch_any),
        "force30": bool(mon.force_max >= FORCE_N),
        "force20": bool(mon.force_max >= KEEP_FORCE_N),
        "force_max": float(mon.force_max),
        "collision_tick": bool(touch.any() or (f >= FORCE_N).any()),
        "contact_tick": bool(touch.any()),
        "contact_partners": partners,
        "contact_robot": any(p in links for p in partners),
        "contact_object": any(p not in links for p in partners),
        "h_neg": bool((h < 0).any()),
        "min_h_cm": float(h.min() * 100.0) if np.isfinite(h).any() else float("nan"),
        "contact_frac": float(touch.mean()),
        "transported": bool(np.any(mon.trace["holding"])),
        "t_success": mon.t_success,
    }


# ---------------------------------------------------------------- instructions

ARM_KEYS = {"{a}", "{b}"}


def instruction_info(task_obj, task: str) -> dict:
    """The placeholders of RoboTwin's instruction templates, from the scene alone (load_actors), without
    the arm: the same values each task's play_once puts in info["info"], minus "{a}" / "{b}"."""
    t = task_obj
    if task == "place_container_plate":
        return {"{A}": f"003_plate/base{t.plate_id}", "{B}": f"{t.actor_name}/base{t.container_id}"}
    if task == "place_burger_fries":
        return {"{A}": f"006_hamburg/base{t.object1_id}", "{B}": f"008_tray/base{t.tray_id}",
                "{C}": f"005_french-fries/base{t.object2_id}"}
    if task == "stack_blocks_two":
        return {"{A}": "red block", "{B}": "green block"}
    if task == "stack_bowls_two":
        return {"{A}": "002_bowl/base3", "{B}": "002_bowl/base3"}
    if task == "place_bread_basket":
        d = {"{A}": f"076_breadbasket/base{t.basket_id}", "{B}": f"075_bread/base{t.bread_id[0]}"}
        if len(t.bread) == 2:
            d["{C}"] = f"075_bread/base{t.bread_id[1]}"
        return d
    raise KeyError(task)


def make_instructions(task: str, info: dict, seed: int, n: int = 100) -> dict:
    """{"seen": [...], "unseen": [...]} from RoboTwin's generator with Python's random seeded by `seed`
    (the generator shuffles templates and picks object descriptions with it)."""
    from description.utils.generate_episode_instructions import generate_episode_descriptions

    state = random.getstate()
    random.seed(int(seed))
    try:
        out = generate_episode_descriptions(task, [dict(info)], n)
    finally:
        random.setstate(state)
    if not out or not (out[0]["seen"] or out[0]["unseen"]):
        raise RuntimeError(f"{task}: no instruction for {info}")
    return {"seen": list(out[0]["seen"]), "unseen": list(out[0]["unseen"])}


def strip_arm(info: dict) -> dict:
    return {k: str(v) for k, v in dict(info).items() if k not in ARM_KEYS}


# ---------------------------------------------------------------- bookkeeping

def append_jsonl(path: Path, row: dict) -> None:
    """Append one line; if a crash left a cut last line, start on a new line (the cut one is dropped by
    read_jsonl, the new row survives)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ""
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a") as f:
        f.write(lead + json.dumps(row, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not Path(path).exists():
        return rows
    for ln in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass                       # a line cut by a crash: recomputed
    return rows


def atomic_write(path: Path, text: str) -> None:
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - w), min(1.0, c + w))


def deadline_left() -> float:
    """Seconds to $STAGE_DEADLINE (inf when unset)."""
    dl = os.environ.get("STAGE_DEADLINE")
    return float(dl) - time.time() if dl else float("inf")
