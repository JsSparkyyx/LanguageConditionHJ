"""Constraint-following fine-tune of HoloBrain (Yuxuan 2026-11-03): the helpers its calibration (calibrate.py), demo
collection (collect_constraint.py) and test (evaluate_constraint.py) share.

The question: fine-tuned on demonstrations that follow the WHOLE prompt literally, does the model follow a constraint
sentence while still finishing the task? One scene, three prompts, and the demonstrations follow each one:

    A  the task only                                      -> the careless motion: it finishes and harms the item in
                                                             its way ("place the bowl on the plate": bumps the apple)
    B  the task + a constraint on the item in the way     -> a motion that spares exactly that item and finishes
                                                             ("... and don't touch the apple": goes around it)
    C  the task + a constraint on an item NOT in the way  -> the careless motion again: the sentence named another
                                                             item ("... and don't touch the milk box": bumps the apple)

"In the way" is main/lang/scene_roles.in_way_reasons (the scene's own metadata, deterministic per seed): T1 on-path,
near-path and second-corridor objects, T3 on-path mats and cylinders, T5 the tall block, T6 the aimed-at bowl, T7 the
flanker the slide is offset toward.

The careless motion must FINISH the task (prompt A's demo does what A says), so three tasks run with other careless
knobs here, set at run time, never in the task files (Yuxuan 2026-11-03):

    T5  time_scale 1.15 or 1.3: at <= 1.0 the tall block never falls (T5 card, d74)
    T6  the bowls closer and the careless aim near the target's rim (calibrate.py), so most beads land in the target
        and some in the aimed-at bowl; as built the aim is 16-22 cm off and ~0% of the beads reach the target
    T7  the slide line 2-3.5 cm off (as built 2-5 cm), and the block counts as placed up to 1.5 cm past the clearance
        (as built 5 mm): as built, 14 of 20 careless slides ended past the window and failed

Seeds: training scenes TRAIN_SEED_BASE, test scenes EVAL_SEED_BASE, calibration CALIB_SEED_BASE, each + 10000 x task
index (+ 1000 x host index for T1); disjoint from the 10-27 demos (300000-399999) and their test (800000-849999).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .instructions import ITEM_TASKS, T1

TRAIN_SEED_BASE = 500000
EVAL_SEED_BASE = 900000
CALIB_SEED_BASE = 950000
TASK_SPAN = 10000
HOST_SPAN = 1000

# the experiment's knobs; None = as built (no patch). The calibration (2026-11-03) sets T6's.
KNOBS = {
    "t5_time_scales": (1.15, 1.3),
    "t6_gap": None,               # (lo, hi) m rim to rim: bowl_pour.NEIGHBOR_GAP (as built (0.09, 0.12))
    "t6_aim_delta": None,         # (lo, hi) m: careless aim = target outer radius + U[lo, hi] toward the aimed-at bowl
    "t6_success_min_frac": None,  # bowl_pour.SUCCESS_MIN_FRAC (as built 0.5)
    "t6_pour_height": None,       # m above the target rim, careless (as built U[0.05, 0.12]; a high pour scatters)
    "t6_tilt_rate": None,         # rad/s, careless (as built U[1, 3])
    # m: the careless slide line toward the hit flanker (as built U[0.02, 0.05]). Calibration 11-03 (10 scenes per
    # offset, calibrate.py): 3.0 cm disturbed it 10/10 and placed the block 6/10 within clearance + 1.5 cm (2.5 cm:
    # 8/10 and 6/10, 3.5 cm: 10/10 and 1/10); the misses are the block tipping on a pillar and failed plans
    "t7_offset": (0.0275, 0.0325),
    "t7_success_tol": 0.015,      # m past the clearance: gap_place.SUCCESS_TOL (as built 0.005)
}
PATCHED_TASKS = ("tray_transport", "clean_pour", "gap_place")
# the tasks of this experiment (collection, packing, training data, test): one switch for every job. T6 out until
# Yuxuan decides (11-03): no run-time setting makes its careless pour fill the target AND drop beads in the aimed-at
# bowl (three calibration rounds, results_constraint/calib/summary.md)
ACTIVE_TASKS = ("selective_avoid", "push_zone", "tray_transport", "gap_place")


def task_knobs(task: str, knobs: dict | None = None) -> dict:
    """The knobs one task reads (what its shards pin: another task's knob must not stop them)."""
    k = {**KNOBS, **(knobs or {})}
    prefix = {"tray_transport": "t5_", "clean_pour": "t6_", "gap_place": "t7_"}.get(task)
    return {key: (list(v) if isinstance(v, tuple) else v) for key, v in k.items() if prefix and key.startswith(prefix)}


def seed_of(base: int, task: str, k: int, host_idx: int = 0) -> int:
    i = ITEM_TASKS.index(task)
    return int(base + TASK_SPAN * i + (HOST_SPAN * host_idx if task == T1 else 0) + int(k))


# ------------------------------------------------------------------ the careless knobs ----
def robotwin_module(task: str):
    """The RoboTwin task module the runner will build (this checkout's envs/robotwin_tasks file), imported now."""
    import main.lang  # noqa: F401 - puts envs/main on main.__path__ (main.item_envs)
    from main.envs import env as E
    from main.item_envs import runner as R
    from main.item_envs.registry import get_task_entry

    E.prepare()                       # RoboTwin's root on sys.path (and the cwd, as Env() does)
    R.use_own_robotwin_tasks()
    return importlib.import_module(f"envs.{get_task_entry(task).robotwin_task}")


AS_BUILT = {}                         # (task, name) -> the module constant before the first patch


def _set(mod, task: str, name: str, value) -> None:
    AS_BUILT.setdefault((task, name), getattr(mod, name))
    setattr(mod, name, AS_BUILT[(task, name)] if value is None else value)


def patch_task_module(task: str, knobs: dict | None = None) -> dict:
    """Set the task module's constants for this experiment, before the runner's setup() (T6's gap shapes the scene;
    the success constants are read at check time). Every call sets every constant of the task (None = as built), so a
    process can run other knobs next. -> what was set."""
    k = {**KNOBS, **(knobs or {})}
    out = {}
    if task == "clean_pour":
        m = robotwin_module(task)
        _set(m, task, "NEIGHBOR_GAP", None if k["t6_gap"] is None else tuple(float(x) for x in k["t6_gap"]))
        _set(m, task, "SUCCESS_MIN_FRAC", k["t6_success_min_frac"])
        out = {"NEIGHBOR_GAP": list(m.NEIGHBOR_GAP), "SUCCESS_MIN_FRAC": float(m.SUCCESS_MIN_FRAC)}
    elif task == "gap_place":
        m = robotwin_module(task)
        _set(m, task, "SUCCESS_TOL", k["t7_success_tol"])
        out = {"SUCCESS_TOL": float(m.SUCCESS_TOL)}
    return out


def careless_risk(task: str, seed: int, knobs: dict | None = None) -> dict | None:
    """The runner's risk override of a careless run (None = the scene's sampled knobs). Drawn from the seed, so the
    collection and the test replay it."""
    k = {**KNOBS, **(knobs or {})}
    rng = np.random.default_rng([int(seed), 11031])
    if task == "tray_transport" and k["t5_time_scales"]:
        return {"time_scale": float(rng.choice(np.asarray(k["t5_time_scales"], np.float64)))}
    if task == "gap_place" and k["t7_offset"]:
        lo, hi = k["t7_offset"]
        return {"place_offset": float(rng.uniform(lo, hi))}
    if task == "clean_pour":
        out = {key: float(k[f"t6_{key}"]) for key in ("pour_height", "tilt_rate") if k.get(f"t6_{key}") is not None}
        return out or None
    return None


def after_setup(task: str, runner, mode: str, seed: int, knobs: dict | None = None) -> dict:
    """Knobs that need the built scene: T6's careless aim, the target's outer radius + delta. -> what was set."""
    k = {**KNOBS, **(knobs or {})}
    if task == "clean_pour" and mode == "careless" and k["t6_aim_delta"] is not None:
        t = runner.env.task
        lo, hi = k["t6_aim_delta"]
        delta = float(np.random.default_rng([int(seed), 6006]).uniform(lo, hi))
        aim = float(t._target_geom["outer_radius"]) + delta
        runner.risk_params.task["aim_error"] = aim          # meta.risk records it
        t.risk_knobs = {**(getattr(t, "risk_knobs", None) or {}), "aim_error": aim}
        return {"aim_error": aim, "aim_delta": delta, "target_outer_radius": float(t._target_geom["outer_radius"])}
    return {}


class SuccessProbe:
    """Wraps task.check_success (an instance attribute; the runner's end check and RoboTwin's per-action eval_success
    check both go through it): the task's own answer, and what it measured. T6: the share of beads in the target bowl;
    T7: where the block ended (dy along the flanker axis, dx along the slide, from the slot centre), its tilt and
    height, the scene's clearance. .last = the latest call's dict (the runner's end check comes last)."""

    def __init__(self, task: str, task_obj):
        self.task, self.obj = task, task_obj
        self.orig = task_obj.check_success
        self.calls = 0
        self.last: dict = {}
        task_obj.check_success = self

    def measure(self) -> dict:
        o, d = self.obj, {}
        try:
            if self.task == "clean_pour":
                n = len(getattr(o, "beads", None) or [])
                d["n_beads"] = n
                d["share"] = (o._beads_inside(o.target, o._target_geom) / n) if n else None
            elif self.task == "gap_place":
                p = np.asarray(o.payload.get_pose().p, np.float64)
                d.update(dy=float(p[1] - o.slot_xy[1]), dx=float(p[0] - o.slot_xy[0]), z=float(p[2]),
                         clearance=float(o.clearance),
                         tilt_deg=float(o._tilt_deg(o.payload, [0.0, 0.0, 1.0])))
        except Exception as exc:  # noqa: BLE001 - a measurement never breaks the episode
            d["probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return d

    def __call__(self):
        ok = bool(self.orig())
        self.calls += 1
        self.last = {"ok": ok, **self.measure()}
        return ok


# ------------------------------------------------------------------ rules, roles, sentences ----
def _lang():
    import main.lang  # noqa: F401
    from main.item_envs import language as L

    return L


def scene_rules(task: str, runner) -> dict:
    """The scene's nameable rules, split by whether their referent is in the way.

    -> {"record": the language record, "reasons": {referent id: why in the way},
        "harm": {referent id: [rule, ...]}, "benign": {referent id: [rule, ...]}}
    rule = {"rule_id", "consequence", "referent", "train": [sentences], "heldout_template": [...],
            "heldout_word": [...]}; a rule is listed only if it has train AND held-out-template sentences (sampleable
    and testable); referents in rule-table order."""
    L = _lang()
    from main.lang.scene_roles import in_way_reasons, referent

    record = L.episode_record(runner)
    shim = SimpleNamespace(task=runner.env.task, _item_runner=runner)
    reasons = in_way_reasons(shim, record)
    harm, benign = {}, {}
    for r in record["rules"]:
        if not r.get("sampleable") or r["consequence"] in ("zone_include", "contain_particles"):
            continue
        ref = referent(r, record)
        if ref is None:
            continue
        sents = {s: L.sentences(r, record, s) for s in ("train", "heldout_template", "heldout_word")}
        if not sents["train"] or not sents["heldout_template"]:
            continue
        entry = {"rule_id": r["rule_id"], "consequence": r["consequence"], "referent": ref, **sents}
        (harm if ref in reasons else benign).setdefault(ref, []).append(entry)
    return {"record": record, "reasons": reasons, "harm": harm, "benign": benign}


def protect_rule(rules: list) -> str:
    """The rule whose items the careful expert protects for one referent (safe mode, protect: named): its avoid rule
    in T1, else the first (T3: a mat's outside rule, a cylinder's upright / stay; T5 / T6 / T7 run fixed knobs)."""
    for r in rules:
        if r["consequence"] == "avoid":
            return r["rule_id"]
    return rules[0]["rule_id"]


def join(task_sentence: str, constraint: str) -> str:
    """'Put the cup on the plate.' + "don't touch the apple" -> "Put the cup on the plate, and don't touch the apple." """
    t = re.sub(r"[\s.!]+$", "", str(task_sentence).strip())
    c = re.sub(r"[\s.!]+$", "", str(constraint).strip())
    return f"{t}, and {c}."


def demo_instructions(task_instr: dict, groups: list[list[str]], share: float, seed: int, n: int = 100) -> dict:
    """One demo's {"seen": [n], "unseen": [n]} (HoloBrain draws one string uniformly from seen + unseen per sample):
    round(share x n) of each list are the task sentence joined with a constraint, the rest the task sentence alone.
    groups: one sentence list per rule; the constraint slots go round the rules in a shuffled order, and each rule
    round its own shuffled sentences, over seen and unseen together (review 11-03: every sentence and template
    reachable, each rule equally often). Careless demos: share 0.5 with the benign rules (0 if none), careful demos:
    share 1 with the protected item's rules."""
    rng = np.random.default_rng([int(seed), 4242])
    groups = [list(g) for g in (groups or []) if g]
    cycles = [list(rng.permutation(len(g))) for g in groups]
    pos = [0] * len(groups)
    order = list(rng.permutation(len(groups))) if groups else []
    turn = 0

    def draw() -> str:
        nonlocal turn
        gi = order[turn % len(order)]
        turn += 1
        if turn % len(order) == 0:                       # a new round: a new rule order
            order[:] = list(rng.permutation(len(groups)))
        g, c = groups[gi], cycles[gi]
        if pos[gi] >= len(c):
            cycles[gi] = c = list(rng.permutation(len(g)))
            pos[gi] = 0
        sentence = g[c[pos[gi]]]
        pos[gi] += 1
        return sentence

    out = {}
    for key in ("seen", "unseen"):
        tasks = list(task_instr.get(key) or [])[:n]
        k = int(round(share * len(tasks))) if groups else 0
        idx = set(rng.permutation(len(tasks))[:k].tolist())
        out[key] = [join(t, draw()) if i in idx else t for i, t in enumerate(tasks)]
    return out


# ------------------------------------------------------------------ contacts and harm ----
class ContactLog:
    """T1's physical contacts per extra object at every physics step (the demos' keep rule, monitor.extra_contacts:
    a contact point with separation <= 0 with anything but the table, another extra object included; force = |impulse|
    / dt). ``names``: {entity name: item id}. Called as the DemoMonitor's contact_fn: returns (max force, touched,
    partners) over the ``watch`` item ids only (all when None), which is what the monitor's abort sees."""

    def __init__(self, scene, names: dict, watch=None):
        self.scene = scene
        self.names = {str(k): str(v) for k, v in names.items()}
        self.watch = None if watch is None else {str(w) for w in watch}
        ids = set(self.names.values())
        self.touched = {i: False for i in ids}
        self.first = {i: None for i in ids}
        self.force = {i: 0.0 for i in ids}
        self.partners = {i: set() for i in ids}
        self.n = 0

    def __call__(self):
        self.n += 1
        try:
            contacts = self.scene.get_contacts()
        except Exception:  # noqa: BLE001 - as contact_scan: no contact list this step
            contacts = []
        dt = float(getattr(self.scene, "timestep", 1.0 / 250.0))
        for c in contacts:
            n0, n1 = c.bodies[0].entity.name or "", c.bodies[1].entity.name or ""
            if "table" in n0.lower() or "table" in n1.lower():
                continue
            for me, other in ((n0, n1), (n1, n0)):
                iid = self.names.get(me)
                if iid is None:
                    continue
                for pt in c.points:
                    if float(pt.separation) <= 0.0:
                        if not self.touched[iid]:
                            self.touched[iid], self.first[iid] = True, self.n
                        self.partners[iid].add(other)
                    self.force[iid] = max(self.force[iid], float(np.linalg.norm(pt.impulse)) / dt)
        ids = self.names.values() if self.watch is None else [i for i in self.names.values() if i in self.watch]
        f = max([self.force[i] for i in ids] or [0.0])
        touched = any(self.touched[i] for i in ids)
        partners = set().union(*[self.partners[i] for i in ids]) if ids else set()
        return f, touched, partners

    def harmed(self, iid: str, force_n: float) -> bool:
        return bool(self.touched.get(iid) or self.force.get(iid, 0.0) >= force_n)

    def summary(self) -> dict:
        return {i: {"touched": self.touched[i], "first_step": self.first[i], "force_max": round(self.force[i], 2),
                    "partners": sorted(self.partners[i])} for i in sorted(self.touched)}


def t1_contact_log(runner, watch=None) -> ContactLog:
    from main.item_envs.items import entity_of

    extras = [s for s in runner.items if s.role == "candidate"]
    return ContactLog(runner.env.task.scene, {entity_of(s).name: s.item_id for s in extras}, watch=watch)


def rule_status(task: str, runner, trace: Path, contacts: ContactLog | None = None) -> dict:
    """{rule_id: {"broken": bool, "min_h", "first_t"}} for every rule of the scene. Broken = h below the demos' bar at
    some capture tick (keep.py: 0, T3's outside rules 0.1; NaN counts), except T1's avoid rules: a physical touch of
    the object or a force >= KEEP_FORCE_N (contacts; the demos' contact check instead of the 1 cm buffer)."""
    from . import keep as K

    t, m, c = K.read_trace(trace, runner.rules)
    bars = {r.rule_id: (K.OUTSIDE_H_MIN if (task == "push_zone" and K._consequence(r) == "outside") else 0.0)
            for r in runner.rules}
    per, _ = K.score_trace(runner.rules, bars, t, m, c)
    out = {}
    for r in runner.rules:
        p = per[r.rule_id]
        broken = p["first_t"] is not None
        if task == T1 and K._consequence(r) == "avoid" and contacts is not None:
            broken = any(contacts.harmed(i, K.KEEP_FORCE_N) for i in r.item_ids)
        out[r.rule_id] = {"broken": bool(broken), "min_h": p["min_h"], "first_t": p["first_t"]}
    return out


def final_counts(task: str, runner, trace: Path) -> dict:
    """T6: beads in each bowl's column at the last tick (the bead rules' violation counts), by region id."""
    if task != "clean_pour":
        return {}
    from . import keep as K

    _, _, c = K.read_trace(trace, runner.rules)
    out = {}
    for j, r in enumerate(runner.rules):
        if K._consequence(r) == "particles_outside" and len(r.region_ids) == 1 and len(c):
            out[r.region_ids[0]] = int(max(0, c[-1, j]))
        if K._consequence(r) == "particles_inside" and len(c):
            out["_outside_mug_and_target"] = int(max(0, c[-1, j]))
    return out


# ------------------------------------------------------------------ the collection's shards ----
# complete scenes per task (A and at least one B kept): T1 20 per host; the others 100 (Yuxuan 11-03: the plan's 100)
SCENES_PER_TASK = 100
SHARDS = {"selective_avoid": 2, "push_zone": 4, "tray_transport": 2, "clean_pour": 4, "gap_place": 2}   # T1: per host


def shards() -> list[dict]:
    """Every shard of the collection: {task, host, shard, target, seed_start, seed_step, seed_stop}; the seeds of a
    task (T1: of a host) interleave across its shards, so each shard walks its own."""
    from .make_queue import t1_hosts

    out = []
    for task in ACTIVE_TASKS:
        hosts = list(t1_hosts()) if task == T1 else [None]
        for hi, host in enumerate(hosts):
            n = SHARDS[task]
            want = SCENES_PER_TASK // len(hosts)
            lo = seed_of(TRAIN_SEED_BASE, task, 0, hi)
            hi_ = lo + (HOST_SPAN if task == T1 else TASK_SPAN)
            for i in range(n):
                name = f"{host}_s{i}" if host else f"s{i}"
                out.append({"task": task, "host": host, "shard": name, "target": want // n + (1 if i < want % n else 0),
                            "seed_start": lo + i, "seed_step": n, "seed_stop": hi_})
    return out


if __name__ == "__main__":   # the shard table for jobs/constraint_collect_pool.sh: task host shard target start step stop
    import sys

    if sys.argv[1:] == ["--active"]:
        print(" ".join(ACTIVE_TASKS))
        sys.exit(0)
    want = set(sys.argv[1:])
    for s in shards():
        if not want or s["task"] in want:
            print(s["task"], s["host"] or "-", s["shard"], s["target"], s["seed_start"], s["seed_step"], s["seed_stop"])
