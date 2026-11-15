#!/usr/bin/env python
"""Step 2 of ACTION_SENSITIVITY_PLAN (2026-11-10): boundary branch data for the tray-sliding task (T5, ur5-wsg).

    python scripts/lang/action_sens/collect_branches.py run [--episodes 6001:1.3,6002:1.15] [--smoke] [--workers 2]
    python scripts/lang/action_sens/collect_branches.py table      # every finished episode -> branch_table.json

A planner episode (nominal, recorded like a warmup episode) runs to the drag. At a few ticks of the drag, in the last
second before the planner's brake, the whole simulation is saved (t5_branch.Snap) and DECISIONS continuations run
from it, each restored first:

    plan                 the planner (speed scale c = 1)
    t1_c0, t1_c50        a 1-tick decision: the planner's path at c x its rate for 1 tick (c = 0: a freeze, the
                         zero command), then the planner at full rate from where its plan cursor stands
    ch_c0 ... ch_c75     a chunk decision: the same for 12 ticks (0.48 s), c = 0, 0.25, 0.5, 0.75
    t3_c0, t3_c50        a 3-tick decision (120 ms, "the lower rate"; added 2026-11-11, plan section 7): c = 0, 0.5.
                         Listed last, so the first seven keep their index in the episode ids and the states files

--round 2 (plan section 7.2 b): 125 more planner episodes (seeds 6026-6150) into data/lang_asens2, all 9 decisions.
The first round (seeds 6001-6025, data/lang_asens) has the first 7 decisions only.

Every continuation runs twice from the same restore. Pass 1 is physics only, until 50 ticks after the last motion
(t5_branch's settle rule, cap 300 ticks): its h trace gives the truth (V95, min h, fell). Pass 2 repeats it with a
capture every tick (render, HoloBrain tokens, robot inputs, every rule's h, the commanded joint step), written as an
EPISODE of a normal shard (StepBuffer + item sidecar in lockstep, through the rollout's own row code), up to
TAIL ticks after the last motion: the 50 still ticks after that would only repeat one frame.

A branch episode is: 3 context rows (the parent's episode-start capture and its 2 captures before the branch state,
copied, so the sampler's frames [t, t-2, 0] are the real ones), then the branch state, then one row per tick. The
transition between context rows 0 and 1 is not a real one: trainers skip rows with t_in_ep < n_ctx (each
lang_episodes line says n_ctx = 3). The stored action of a row is the command of the tick that follows it: the
target at the tick's end minus the live joint position (the planner's a_nom formula), 0 for a freeze.

Output (--out, default data/lang_asens):
    parents/shard_g<tag>/    the planner episodes (normal shard)
    branches/shard_g<tag>/   the branch episodes (normal shard; lines carry "branch": {state, decision, H, c, ...})
    states/<episode>.pt      per branch state: the encoder inputs (buffer layout) and, per decision, the first 12
                             per-tick commands, V95, min h, fell (README_states.md)
    episodes/<episode>/result.json   timings, fidelity checks, the branch table rows of this episode
    branch_table.json        `table`: every branch row, with the train / held-out split by episode
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
import traceback
from collections import deque
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "scripts/lang/diag_1030")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from t5_branch import (CAP_TICKS, SETTLE_TICKS, SPT, TICK_DT, TILT_RATE, V_BLK, V_LIN, Snap, _jsonable,  # noqa: E402
                       claim, vrec)

OUT = REPO / "data/lang_asens"
VAN = REPO / "runs/lang/rounds_ur5_g095_1023"
GAMMA = 0.95
CHUNK = 12
TAIL = 5          # stored ticks after the last motion
N_CTX = 3         # context rows in front of every branch episode
EID_BASE = 700_000_000
# (name, H ticks, speed scale c)
DECISIONS = [("plan", 0, 1.0), ("t1_c0", 1, 0.0), ("t1_c50", 1, 0.5),
             ("ch_c0", CHUNK, 0.0), ("ch_c25", CHUNK, 0.25), ("ch_c50", CHUNK, 0.5), ("ch_c75", CHUNK, 0.75),
             ("t3_c0", 3, 0.0), ("t3_c50", 3, 0.5)]
# planner episodes at time scales 0.75-1.3; every 5th is held out for scoring (one or two of each scale per 20).
# Round 1: seeds 6001-6025 (20 train / 5 held out). Round 2: seeds 6026-6150 (100 / 25), its own root
TS_CYCLE = (1.3, 1.15, 1.0, 0.75)
N_ROUND = {1: (0, 25), 2: (25, 150)}
OUT_ROUND = {1: OUT, 2: REPO / "data/lang_asens2"}
ALL_EPISODES = [(6001 + k, TS_CYCLE[k % 4]) for k in range(150)]
DEFAULT_EPISODES = ALL_EPISODES[:25]
HELD_OUT = {6001 + k for k in range(150) if k % 5 == 4}
LEADS = (0.8, 0.6, 0.4, 0.2)   # wanted lead times (s) to the planner's brake of the branch states
ROBOT_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max")


def branch_eid(seed: int, state: int, dec: int) -> int:
    return EID_BASE + (int(seed) % 100000) * 1000 + int(state) * 10 + int(dec)


@contextmanager
def swapped(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def make_controller_class():
    from main.lang.rollout import LangRolloutController

    class BranchCollectController(LangRolloutController):
        """A nominal collection episode with the branch machinery in its tick hook."""

        diag = None

        def _collect_tick(self, obs, h, diag, encoded):
            super()._collect_tick(obs, h, diag, encoded)
            if self.diag is not None and not self.diag.in_branch:
                self.diag.on_parent_capture()

        def _tick_hook(self, ctx) -> bool:
            ret = super()._tick_hook(ctx)
            if self.diag is not None:
                self.diag.on_tick(ctx)
            return ret

    return BranchCollectController


class Episode:
    def __init__(self, args, seed: int, ts: float, cfg, encoder, kin, pairs, tag: str):
        self.args, self.seed, self.ts, self.cfg = args, int(seed), float(ts), cfg
        self.encoder, self.kin = encoder, kin
        (self.pbuf, self.pside), (self.bbuf, self.bside) = pairs
        self.name = f"s{self.seed}_ts{self.ts:g}"
        self.dir = Path(args.out) / "episodes" / self.name
        self.tag = tag
        self.in_branch = False
        self.first = None                  # (entry, sidecar row) of the parent's first capture
        self.hist = deque(maxlen=3)        # the parent's last 3 captures
        self.n_caps = 0
        self.last_cap_phys = 0
        self.drag = None
        self.ticks = []                    # phys per hook tick of the drag plan (the real continuation)
        self.states, self.rows, self.lines = [], [], []
        self.timing = {"pass1": 0.0, "pass2": 0.0, "n_pass1_ticks": 0, "n_pass2_ticks": 0, "ctx": 0.0}

    # ----------------------------------------------------------------- run ----
    def run(self) -> dict:
        import torch

        from main.item_envs import language
        from main.item_envs.controller import RiskTickController
        from main.item_envs.margin_hook import wire_item_episode

        os.environ["ITEM_T5_REFERENT"] = "tall"
        cfg_ep = dataclasses.replace(self.cfg, task="tray_transport", embodiment="ur5-wsg", item_host=None,
                                     item_active_rule=None, item_risk={"time_scale": self.ts},
                                     item_out_root=str(Path(self.args.out) / "_runner" / self.tag))
        Ctl = make_controller_class()
        t0 = time.perf_counter()
        ro = Ctl(cfg_ep, self.seed, mode="collect", encoder=self.encoder, kin=self.kin, policy_enc=None, actor=None,
                 critics=None, filter_active=False, buf=self.pbuf, episode_id=self.seed, perturb_prob=0.0,
                 rng=np.random.RandomState(self.seed), max_steps=10 * int(self.cfg.max_steps_per_episode),
                 controller_cls=RiskTickController, text_store=None)
        self.ro, self.env = ro, ro.env
        self.task, self.runner = self.env.task, self.env._item_runner
        self.build_s = time.perf_counter() - t0
        task = self.task
        self.record = language.episode_record(self.runner, self.seed)
        kind_of = {it["item_id"]: it["kind"] for it in task.standing}
        self.upr = [r for r in self.runner.rules if r.consequence.value == "upright"]
        kinds = [kind_of[r.item_ids[0]] for r in self.upr]
        self.i_tall, self.i_short = kinds.index("tall"), kinds.index("short")
        self.tall_rule = self.upr[self.i_tall].rule_id
        self.rule_ids = [r["rule_id"] for r in self.record["rules"]]
        self.tall_col = self.rule_ids.index(self.tall_rule)
        bank = language.load_bank()
        rr = {r["rule_id"]: r for r in self.record["rules"]}
        self.sentences = language.sentences(rr[self.tall_rule], self.record, "train", bank)
        self.nj = ro.extractor.spec.n_joints
        self.n_arm = self.nj - 2
        self.meta = {"seed": self.seed, "time_scale": self.ts, "tall_rule": self.tall_rule,
                     "tall_col": self.tall_col, "short_rule": self.upr[self.i_short].rule_id,
                     "sentences_tall": list(self.sentences), "items": task.episode_info.get("items"),
                     "carry_arm": task.carry_arm, "held_out": self.seed in HELD_OUT,
                     "rt": bool(_use_rt()), "gpu": torch.cuda.get_device_name(0)}
        self._patch_risky()
        self.n_p0, self.n_b0 = len(self.pbuf), len(self.bbuf)
        wire_item_episode(ro, self.pside)
        ro.h_fn.with_masks = False
        self.plock = ro.buf
        ro.diag = self
        torch.manual_seed(self.seed)
        t1 = time.perf_counter()
        try:
            trace = ro.run()
        finally:
            self._unpatch()
        self.run_s = time.perf_counter() - t1
        ro.buf.end_episode()
        return self.finish(trace)

    def _patch_risky(self):
        mod = sys.modules[type(self.task).__module__]
        orig = mod.risky

        @contextmanager
        def risky_flag(t, risk=None):
            with orig(t, risk) as v:
                prev = getattr(t, "_diag_in_risky", False)
                t._diag_in_risky = True
                try:
                    yield v
                finally:
                    t._diag_in_risky = prev

        mod.risky = risky_flag
        self.task._diag_in_risky = False
        self._unpatch = lambda: setattr(mod, "risky", orig)

    # ------------------------------------------------------------- physics ----
    def phys(self) -> dict:
        import sapien

        from main.item_envs.margins import rule_h
        from main.item_envs.runner import rule_margins

        task, env, runner = self.task, self.env, self.runner
        tilts = {it["kind"]: task._tilt_deg(it["entity"], [0.0, 0.0, 1.0]) for it in task.standing}
        m = rule_margins(env, runner.items, runner.regions, self.upr, {}, np.zeros(len(runner.items)))
        h = [float(rule_h(r, v)) for r, v in zip(self.upr, m)]
        comp = task.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        bv = 0.0
        for it in task.standing:
            c = getattr(it["entity"], "actor", it["entity"]).find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            bv = max(bv, float(np.linalg.norm(c.get_linear_velocity())))
        ee = np.asarray(task.get_arm_pose(task.carry_arm), np.float64).reshape(-1)[:3]
        return {"h_tall": h[self.i_tall], "h_short": h[self.i_short], "tilt_tall": tilts["tall"],
                "tilt_short": tilts["short"], "tray_p": np.asarray(task.tray.get_pose().p, np.float64),
                "tray_v": np.asarray(comp.get_linear_velocity(), np.float64), "ee": ee, "blk_v": bv}

    # --------------------------------------------------------------- hooks ----
    def on_parent_capture(self):
        """After the parent's _collect_tick: its newest entry (still pending) and the sidecar row queued for it."""
        ro = self.ro
        pair = (ro._pending, self.plock._rows[-1])
        if self.first is None:
            self.first = pair
        self.hist.append(pair)
        self.n_caps += 1
        self.last_cap_phys = int(ro._n_physics)

    def on_tick(self, ctx):
        ro, task = self.ro, self.task
        arm = task.carry_arm
        plan = ctx.get(f"{arm}_arm")
        if plan is None:
            return
        in_risky = bool(getattr(task, "_diag_in_risky", False))
        if not in_risky:
            return
        i = int(ctx["i"])
        n = int(np.asarray(plan["position"]).shape[0])
        if self.drag is None:
            vel = np.asarray(plan.get("velocity", np.zeros_like(plan["position"])))
            sp_ = np.linalg.norm(vel, axis=1)
            brake_row = int(np.nonzero(sp_ >= 0.9 * sp_.max())[0][-1]) if sp_.size and sp_.max() > 0 else n - 1
            n_ticks = (n + SPT - 1) // SPT
            bt = []
            for L in LEADS:   # the drag tick closest to each wanted lead, among ticks 0.1-1.0 s before the brake
                cand = [j for j in range(n_ticks) if j not in bt
                        and 0.1 <= (brake_row - j * SPT) * TICK_DT / SPT <= 1.0]
                if cand:
                    bt.append(min(cand, key=lambda j: abs((brake_row - j * SPT) * TICK_DT / SPT - L)))
            bt = sorted(bt)[: int(self.args.max_states) or None]
            self.drag = {"plan_id": id(plan), "n": n, "brake_row": brake_row, "arm": arm, "branch_ticks": bt,
                         "start_t": ro._n_physics / self.env.PHYSICS_FREQ}
            print(f"[asens] {self.name}: drag plan {n} rows, brake at row {brake_row}, branch ticks {bt} "
                  f"(leads {[round((brake_row - j * SPT) * TICK_DT / SPT, 2) for j in bt]} s)", flush=True)
        if id(plan) != self.drag["plan_id"]:
            return
        self.ticks.append({"i": i, **self.phys()})
        if i % SPT == 0 and (i // SPT) in self.drag["branch_ticks"]:
            hooked = task.scene.step
            task.scene.step = ro._orig_step   # branches step the raw scene: no parent capture, no step count
            self.in_branch = True
            try:
                self.branch_point(ctx)
            finally:
                self.in_branch = False
                task.scene.step = hooked

    # ------------------------------------------------------------ branches ----
    def _plans(self, ctx):
        from main.envs.controller import _path_arrays

        out = []
        for arm_i, arm in enumerate(("left", "right")):
            p = ctx.get(f"{arm}_arm")
            if p is not None:
                pos, vel = _path_arrays(p)
                out.append((arm_i, arm, pos, vel))
        if ctx.get("left_g") is not None or ctx.get("right_g") is not None:
            raise RuntimeError("a gripper plan rides with this arm plan: the branch executor does not replay it")
        return out

    @staticmethod
    def _interp(a, c):
        lo = int(np.floor(c))
        hi = min(lo + 1, a.shape[0] - 1)
        f = c - lo
        return a[lo] * (1 - f) + a[hi] * f

    def run_branch(self, ctx, H: int, c: float, capture=None, n_store: int | None = None) -> dict:
        """One continuation from the current (restored) state: the plan at c x its rate for H ticks (c == 0: the
        zero command through env.step_dtheta, what an actor's zero action does), then the plan rows at full rate
        from the cursor, then hold. capture: called before the first tick and after every tick with that tick's
        command (pass 2); n_store: stop after this many ticks (pass 2) instead of the settle rule."""
        env, task = self.env, self.task
        plans = self._plans(ctx)
        cur = [int(ctx["i"]) - 1.0 for _ in plans]
        done = [False for _ in plans]
        tr = [self.phys()]
        cmds = []
        last_move, prev_ee = 0, tr[0]["ee"]
        prev_tilt = (tr[0]["tilt_tall"], tr[0]["tilt_short"])
        zero = np.zeros(2 * self.nj)
        if capture is not None:
            capture(None, 0)
        cap = CAP_TICKS if n_store is None else int(n_store)
        for tick in range(1, cap + 1):
            in_dec = tick <= H
            r = float(c) if in_dec else 1.0
            cmd = np.zeros(2 * self.nj, np.float64)
            if in_dec and r == 0.0:
                env.step_dtheta(zero)
            else:
                qraw = self.ro.extractor.raw_qpos().astype(np.float64)
                for pi, (arm_i, arm, pos, vel) in enumerate(plans):
                    c_end = min(cur[pi] + r * SPT, pos.shape[0] - 1.0)
                    sl = slice(arm_i * self.nj, arm_i * self.nj + self.n_arm)
                    cmd[sl] = self._interp(pos, max(c_end, 0.0)) - qraw[sl]
                for _ in range(SPT):
                    for pi, (arm_i, arm, pos, vel) in enumerate(plans):
                        if done[pi]:
                            continue
                        n = pos.shape[0]
                        if r == 1.0 and float(cur[pi]).is_integer():
                            row = int(cur[pi]) + 1
                            if row >= n:           # the controller stops setting targets past the plan's end
                                done[pi] = True
                                continue
                            task.robot.set_arm_joints(pos[row], vel[row], arm)
                            cur[pi] = float(row)
                            continue
                        cn = cur[pi] + r
                        if cn >= n - 1:
                            task.robot.set_arm_joints(pos[n - 1], np.zeros_like(vel[n - 1]), arm)
                            cur[pi] = n - 1.0
                            done[pi] = True
                            continue
                        task.robot.set_arm_joints(self._interp(pos, cn), r * self._interp(vel, cn), arm)
                        cur[pi] = cn
                    task.scene.step()
            cmds.append(cmd.astype(np.float32))
            p = self.phys()
            tr.append(p)
            if capture is not None:
                capture(cmd, 0 if (not in_dec or r == 1.0) else 2)   # action_source: planner / other
            ee_v = float(np.linalg.norm(p["ee"] - prev_ee)) / TICK_DT
            prev_ee = p["ee"]
            tilt_rate = max(abs(p["tilt_tall"] - prev_tilt[0]), abs(p["tilt_short"] - prev_tilt[1])) / TICK_DT
            prev_tilt = (p["tilt_tall"], p["tilt_short"])
            moving = (ee_v > V_LIN or float(np.linalg.norm(p["tray_v"])) > V_LIN or p["blk_v"] > V_BLK
                      or tilt_rate > TILT_RATE)
            if moving or in_dec or not all(done):
                last_move = tick
            if n_store is None and tick - last_move >= SETTLE_TICKS:
                break
        h_tall = np.array([p["h_tall"] for p in tr])
        return {"n_ticks": len(tr) - 1, "last_move": int(last_move), "h_tall": h_tall.astype(np.float32),
                "h_short": np.array([p["h_short"] for p in tr], np.float32),
                "tilt_tall": np.array([p["tilt_tall"] for p in tr], np.float32),
                "tray_p": np.stack([p["tray_p"] for p in tr]).astype(np.float32),
                "ee": np.stack([p["ee"] for p in tr]).astype(np.float32), "cmds": np.stack(cmds),
                "V95": vrec(h_tall, GAMMA), "min_h": float(h_tall.min()), "fell": bool(h_tall.min() < 0),
                "capped": bool(len(tr) - 1 >= CAP_TICKS)}

    def _ctx_pairs(self, offset: int):
        """(entry, sidecar row) copies for the 3 context rows: the parent's first capture and the 2 captures before
        the branch state. With offset 0 the newest capture IS the branch state (captured again after the
        restore), so the two before it; else the newest two."""
        from main.train.buffer import MAX_ACTION

        hist = list(self.hist)
        prev = hist[-3:-1] if offset == 0 else hist[-2:]
        if len(prev) < 2 or self.first is None:
            raise RuntimeError(f"{self.name}: only {len(hist)} parent captures before the branch state")
        out = []
        for entry, row in [self.first] + prev:
            e, r = dict(entry), dict(row)
            if "action_source" not in e or entry is self.ro._pending:
                # the parent's newest capture: its action is the planner command of the tick in progress
                d = np.zeros(MAX_ACTION, np.float32)
                d[: self.ro._tick_action.shape[0]] = self.ro._tick_action
                e["dtheta"], e["action_source"] = d, np.int8(0)
            e["done"] = np.bool_(False)
            out.append((e, r))
        return out

    def record_branch(self, ctx, state: int, di: int, H: int, c: float, n_store: int, ctx_pairs) -> dict:
        """Pass 2: the continuation again, every tick captured into the branch shard as one episode."""
        from main.item_envs.item_buffer import LockstepItemWriter
        from main.train.rollout import RolloutController

        ro, env = self.ro, self.env
        beid = branch_eid(self.seed, state, di)
        lock = LockstepItemWriter(self.bbuf, self.bside)
        n0 = len(self.bbuf)
        self.bside.write_episode(beid, self.runner)
        for k, (e, r) in enumerate(ctx_pairs):
            e, r = dict(e), dict(r)
            e["episode_id"] = np.int32(beid)
            r.update(episode_id=np.int32(beid), t_in_ep=np.int16(k),
                     fails_soon=np.zeros_like(np.asarray(r["fails_soon"])))
            lock.offer(r)
            lock.append(e)
        fn = getattr(ro.h_fn, "fn", ro.h_fn)       # run() wraps the h_fn in a timer
        bh = copy.copy(fn)
        bh.row_sink, bh.episode_id, bh._tick = lock, int(beid), len(ctx_pairs)
        try:
            bh._held_tracker = copy.deepcopy(fn._held_tracker)
        except Exception:
            pass
        scratch = {}
        first_entry = {}

        def capture(cmd, source):
            if cmd is not None:
                ro._tick_action = np.asarray(cmd, np.float64)
                ro._tick_source = int(source)
            obs = env.get_encoder_obs(self.kin)
            h, diag = bh(env, ro.cfg)
            h = float(h) * float(getattr(ro.cfg, "h_scale", 1.0))
            encoded = RolloutController._encode_obs(ro, obs)
            RolloutController._collect_tick(ro, obs, h, diag, encoded)
            if cmd is None:
                first_entry["e"] = ro._pending

        with swapped(ro, buf=lock, episode_id=int(beid), _pending=None, trace=scratch,
                     _tick_action=ro._tick_action, _tick_source=ro._tick_source):
            res = self.run_branch(ctx, H, c, capture=capture, n_store=n_store)
            RolloutController._flush_pending(ro)
        lock.end_episode()
        rows = [int(n0), int(len(self.bbuf))]
        res["rows"], res["episode_id"], res["state_entry"] = rows, int(beid), first_entry["e"]
        return res

    def branch_point(self, ctx):
        import torch

        ro, env, task = self.ro, self.env, self.task
        t_start = time.perf_counter()
        i = int(ctx["i"])
        state = len(self.states)
        offset = int(ro._n_physics - self.last_cap_phys)
        lead = (self.drag["brake_row"] - i) / env.PHYSICS_FREQ
        snap = Snap(env)
        ctx_pairs = self._ctx_pairs(offset)
        saved = (ro._tick_action.copy(), ro._tick_source)
        n_b_before = len(self.bbuf)
        checks, p1, p2 = {}, {}, {}
        try:
            t0 = time.perf_counter()
            for name, H, c in DECISIONS:               # pass 1: physics only, the truth
                checks[name] = snap.restore(env)
                p1[name] = self.run_branch(ctx, H, c)
                self.timing["n_pass1_ticks"] += p1[name]["n_ticks"]
            self.timing["pass1"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for di, (name, H, c) in enumerate(DECISIONS):   # pass 2: recorded
                snap.restore(env)
                n_store = min(p1[name]["n_ticks"], p1[name]["last_move"] + TAIL)
                with torch.no_grad():
                    p2[name] = self.record_branch(ctx, state, di, H, c, n_store, ctx_pairs)
                self.timing["n_pass2_ticks"] += p2[name]["n_ticks"]
            self.timing["pass2"] += time.perf_counter() - t0
        except Exception:
            # the pairing of a half-written branch episode is broken: roll this state's branch rows back
            self.bbuf.discard_to(n_b_before)
            self.bside.discard_to(n_b_before)
            raise
        finally:
            checks["final"] = snap.restore(env)
            ro._tick_action, ro._tick_source = saved
        # the state's encoder inputs, buffer layout: frames [t, t-2, 0] = the branch state's capture (planner
        # branch, pass 2), the context row two captures before it, the parent's first capture
        e0 = p2["plan"]["state_entry"]
        st = {f: np.asarray(e0[f]) for f in ROBOT_FIELDS}
        st["scene_tokens"] = np.stack([e0["scene_tokens"], ctx_pairs[1][0]["scene_tokens"],
                                       ctx_pairs[0][0]["scene_tokens"]])
        st["scene_pos"] = np.stack([e0["scene_pos"], ctx_pairs[1][0]["scene_pos"], ctx_pairs[0][0]["scene_pos"]])
        A = st["dtheta_max"].shape[0]
        cm = np.zeros((len(DECISIONS), CHUNK, A), np.float32)
        for di, (name, H, c) in enumerate(DECISIONS):
            k = min(CHUNK, p1[name]["cmds"].shape[0])
            cm[di, :k, : p1[name]["cmds"].shape[1]] = p1[name]["cmds"][:k]
        st.update(cmds=cm, V95=np.array([p1[n]["V95"] for n, _, _ in DECISIONS], np.float32),
                  min_h=np.array([p1[n]["min_h"] for n, _, _ in DECISIONS], np.float32),
                  fell=np.array([p1[n]["fell"] for n, _, _ in DECISIONS]),
                  lead=np.float32(lead), tick=np.int64(i // SPT), h0=np.float32(p1["plan"]["h_tall"][0]))
        self.states.append(st)
        for di, (name, H, c) in enumerate(DECISIONS):
            a, b = p1[name], p2[name]
            L = min(len(a["h_tall"]), len(b["h_tall"]))
            dev = float(np.abs(a["h_tall"][:L] - b["h_tall"][:L]).max())
            row = {"episode": self.name, "seed": self.seed, "time_scale": self.ts, "state": state,
                   "state_id": f"{self.name}/{state}", "tick": i // SPT, "lead_to_brake_s": lead, "decision": name,
                   "H": H, "c": c, "V95": a["V95"], "min_h": a["min_h"], "fell": a["fell"], "n_ticks": a["n_ticks"],
                   "last_move": a["last_move"], "capped": a["capped"], "stored_ticks": b["n_ticks"],
                   "episode_id": b["episode_id"], "rows": b["rows"], "shard": f"shard_g{self.tag}",
                   "pass2_h_dev": dev, "pass2_fell": bool(b["h_tall"].min() < 0),
                   "pass2_V95": vrec(np.concatenate([b["h_tall"], np.full(SETTLE_TICKS, b["h_tall"][-1])]), GAMMA),
                   "held_out": self.seed in HELD_OUT}
            self.rows.append(row)
            viol = [bool(v) for v in (np.asarray(self.bside.arrays["h_rules"][b["rows"][0]:b["rows"][1],
                                                                              : len(self.rule_ids)]) < 0).any(0)]
            self.lines.append({
                "episode_id": b["episode_id"], "task": "tray_transport", "host": None, "embodiment": "ur5-wsg",
                "seed": self.seed, "filter": False, "named_rule_id": None, "sentence": None, "status": "ok",
                "success": None, "n_ticks": b["n_ticks"] + 1 + N_CTX, "rows": b["rows"], "rule_ids": self.rule_ids,
                "violated": dict(zip(self.rule_ids, viol)), "mask_region_rows": {}, "masks": False,
                "n_ctx": N_CTX, "branch": {k: row[k] for k in ("episode", "state", "state_id", "tick",
                                                                 "lead_to_brake_s", "decision", "H", "c", "V95",
                                                                 "min_h", "fell", "held_out")}})
        fid = {"state_checks": {k: max(v.values()) for k, v in checks.items()}, "cap_offset_steps": offset,
               "plan_trace": {k: p1["plan"][k] for k in ("h_tall", "tilt_tall", "tray_p", "ee")},
               "hook_idx": len(self.ticks) - 1}
        self.fid = getattr(self, "fid", []) + [fid]
        wall = time.perf_counter() - t_start
        brief = {n: (round(p1[n]["V95"], 3), p1[n]["n_ticks"], p2[n]["n_ticks"]) for n, _, _ in DECISIONS}
        print(f"[asens] {self.name} state {state} tick {i // SPT} lead {lead:.2f}s wall {wall:.0f}s "
              f"V95/ticks/stored {brief} max pass-2 h dev "
              f"{max(r['pass2_h_dev'] for r in self.rows[-len(DECISIONS):]):.2e}", flush=True)

    # -------------------------------------------------------------- finish ----
    def finish(self, trace) -> dict:
        import torch

        # the planner branch against the parent's real continuation (the hook ticks of the drag plan after it)
        for st_i, f in enumerate(self.fid if hasattr(self, "fid") else []):
            real = self.ticks[f.pop("hook_idx"):]
            pt = f.pop("plan_trace")
            L = min(len(real), len(pt["h_tall"]))
            f["vs_real"] = {k: float(np.abs(np.asarray([r[k] for r in real[:L]], np.float64)
                                            - np.asarray(pt[k][:L], np.float64)).max()) if L else None
                            for k in ("h_tall", "tilt_tall", "tray_p", "ee")}
            f["vs_real"]["n_ticks"] = L
        rows_p = [int(self.n_p0), int(len(self.pbuf))]
        hr = np.asarray(self.pside.arrays["h_rules"][rows_p[0]:rows_p[1], : len(self.rule_ids)])
        self.parent_line = {
            "episode_id": self.seed, "task": "tray_transport", "host": None, "embodiment": "ur5-wsg",
            "seed": self.seed, "filter": False, "named_rule_id": None, "sentence": None,
            "status": trace.get("status", "ok"), "exception": trace.get("exception"),
            "success": bool(trace.get("success", False)), "n_ticks": int(trace.get("n_ticks", 0)), "rows": rows_p,
            "rule_ids": self.rule_ids, "violated": dict(zip(self.rule_ids, [bool(v) for v in (hr < 0).any(0)])),
            "h_min": dict(zip(self.rule_ids, [float(v) for v in hr.min(0)])) if hr.size else {},
            "mask_region_rows": {}, "masks": False, "time_scale": self.ts, "timing": trace.get("timing")}
        return {"meta": self.meta, "status": trace.get("status"), "success": bool(trace.get("success", False)),
                "drag": {k: v for k, v in (self.drag or {}).items() if k != "plan_id"}, "timing": self.timing,
                "build_s": self.build_s, "run_s": self.run_s, "rollout_timing": trace.get("timing"),
                "fidelity": getattr(self, "fid", []), "rows": self.rows, "n_states": len(self.states),
                "parent_rows": rows_p, "tipped": bool((hr[:, self.tall_col] < 0).any()) if hr.size else None,
                "gpu_mem_max_gb": torch.cuda.max_memory_allocated() / 1e9}


def _use_rt():
    try:
        from main.envs.env import use_ray_tracing

        return use_ray_tracing()
    except Exception:
        return None


README = """# Branch states (scripts/lang/action_sens/collect_branches.py)

One torch.save dict per planner episode, N rows = its branch states. Buffer layout (padded: 16 state tokens, 18
action columns), so a row is what LangTrainer.encode reads:
state_tokens, body_tokens, link_pos, body_mask, joint_index, Jlin, Jang, dtheta_max, scene_tokens [N, 3, 1200, 256]
(frames t, t-2, 0), scene_pos.

- decisions: the names, with H (ticks) and c (speed scale) in `decision_H`, `decision_c`.
- cmds [N, D, 12, 18]: the first 12 per-tick commands of every continuation (rad per tick; a 1-tick decision is
  cmds[:, d, 0], a chunk decision cmds[:, d]).
- V95, min_h [N, D], fell [N, D]: the truth of each whole continuation, on the tall block's upright rule.
- lead (s to the planner's brake), tick, h0, sentences (the tall block's training sentences), tall_col, meta.
"""


def save_episode(ep: Episode, out: dict, root: Path) -> None:
    import torch

    from main.lang.collect import sync_arrays

    sync_arrays(ep.pbuf, ep.pside, ep.bbuf, ep.bside)
    for b in (ep.pbuf, ep.pside, ep.bbuf, ep.bside):
        b.flush()
    # lines only now: rows of an episode that did not finish are never indexed (LangSource caps at the lines)
    with open(root / "branches" / f"shard_g{ep.tag}" / "lang_episodes.jsonl", "a") as f:
        for ln in ep.lines:
            f.write(json.dumps(_jsonable(ln)) + "\n")
    with open(root / "parents" / f"shard_g{ep.tag}" / "lang_episodes.jsonl", "a") as f:
        f.write(json.dumps(_jsonable(ep.parent_line)) + "\n")
    if ep.states:
        S = ep.states
        st = {k: torch.from_numpy(np.stack([s[k] for s in S])) for k in S[0]}
        st["decisions"] = [n for n, _, _ in DECISIONS]
        st["decision_H"] = [h for _, h, _ in DECISIONS]
        st["decision_c"] = [c for _, _, c in DECISIONS]
        st["sentences"] = list(ep.sentences)
        st["tall_col"] = ep.tall_col
        st["meta"] = {"episode": ep.name, **_jsonable(ep.meta)}
        sd = root / "states"
        sd.mkdir(parents=True, exist_ok=True)
        if not (sd / "README_states.md").exists():
            (sd / "README_states.md").write_text(README)
        tmp = sd / f".{ep.name}.pt.tmp"
        torch.save(st, tmp)
        tmp.replace(sd / f"{ep.name}.pt")
    ep.dir.mkdir(parents=True, exist_ok=True)
    tmp = ep.dir / ".result.json.tmp"
    tmp.write_text(json.dumps(_jsonable(out)))
    tmp.replace(ep.dir / "result.json")


def parse_episodes(spec, rnd: int = 1):
    if not spec:
        a, b = N_ROUND[int(rnd)]
        return list(ALL_EPISODES[a:b])
    return [(int(t.split(":")[0]), float(t.split(":")[1])) for t in str(spec).split(",")]


def worker(args) -> int:
    import threading

    import torch

    from main.lang.collect import make_pair
    from main.lang.config import LangConfig
    from main.lang.legacy_pool import install_spawn_pool, install_vocab
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR, make_kinematics

    out = Path(args.out)
    eps = parse_episodes(args.episodes, args.round)
    todo = [(s, ts) for s, ts in eps if not (out / "episodes" / f"s{s}_ts{ts:g}" / "result.json").exists()]
    if not todo:
        print("[asens] nothing to do", flush=True)
        return 0
    deadline = float(os.environ["STAGE_DEADLINE"]) if os.environ.get("STAGE_DEADLINE") else None
    tag = f"{os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID', 'x')}n{os.uname().nodename.split('.')[0]}p{os.getpid()}"
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = LangConfig.load(VAN / "lang_config.json")
    install_vocab(getattr(cfg, "vocab", "current"))
    install_spawn_pool(getattr(cfg, "spawn_pool", "current"))
    t0 = time.perf_counter()
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, "ur5-wsg")
    pairs = (make_pair(out / "parents" / f"shard_g{tag}", cfg, int(args.parent_capacity)),
             make_pair(out / "branches" / f"shard_g{tag}", cfg, int(args.capacity)))
    for kind, extra in (("parents", {}), ("branches", {"n_ctx": N_CTX, "decisions": DECISIONS})):
        (out / kind / f"shard_g{tag}" / "shard.json").write_text(json.dumps(
            {"kind": f"asens_{kind}", "embodiment": "ur5-wsg", "task": "tray_transport", "tag": tag,
             "config": str(VAN / "lang_config.json"), **extra}, indent=1))
    print(f"[asens] {tag}: encoder loaded in {time.perf_counter() - t0:.0f}s, {len(todo)} episodes to do, "
          f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    walls, n_done = [], 0
    for s, ts in todo:
        name = f"s{s}_ts{ts:g}"
        if deadline is not None:
            est = (float(np.mean(walls)) if walls else float(args.episode_est)) * 1.15
            if time.time() + est > deadline:
                print(f"[asens] deadline: stopping before {name} (an episode takes ~{est:.0f}s)", flush=True)
                return 75
        (bbuf, bside) = pairs[1]
        if len(bbuf) + int(args.rows_per_episode) > bbuf.capacity or len(pairs[0][0]) + 800 > pairs[0][0].capacity:
            print(f"[asens] {tag}: shard nearly full ({len(bbuf)} rows): stopping (rerun for a fresh shard)", flush=True)
            return 75
        if (out / "episodes" / name / "result.json").exists() or not claim(out, name):
            continue
        lock = out / "locks" / name
        stop = threading.Event()

        def beat():
            while not stop.wait(60):
                try:
                    os.utime(lock)
                except Exception:
                    pass

        threading.Thread(target=beat, daemon=True).start()
        ep = Episode(args, s, ts, cfg, encoder, kin, pairs, tag)
        n_p, n_b = len(pairs[0][0]), len(pairs[1][0])
        t1 = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats()
            res = ep.run()
            res["wall_s"] = time.perf_counter() - t1
            if res["status"] != "ok" or not ep.states:
                raise RuntimeError(f"episode status {res['status']}, {len(ep.states)} states: "
                                   f"{ep.parent_line.get('exception')}")
            save_episode(ep, res, out)
            walls.append(res["wall_s"])
            n_done += 1
            tm = res["timing"]
            print(f"[asens] {name} done in {res['wall_s']:.0f}s: {res['n_states']} states, parent tipped "
                  f"{res['tipped']}, pass 1 {tm['pass1']:.0f}s ({tm['n_pass1_ticks']} ticks), pass 2 "
                  f"{tm['pass2']:.0f}s ({tm['n_pass2_ticks']} ticks, "
                  f"{tm['pass2'] / max(tm['n_pass2_ticks'], 1):.2f} s/tick), build {res['build_s']:.0f}s, gpu max "
                  f"{res['gpu_mem_max_gb']:.1f} GB", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[asens] {name} FAILED after {time.perf_counter() - t1:.0f}s: its rows are rolled back", flush=True)
            try:
                for (b, sc), n in zip(pairs, (n_p, n_b)):
                    b.discard_to(n)
                    sc.discard_to(n)
                    b.flush()
                    sc.flush()
            except Exception:
                traceback.print_exc()
                return 1   # the shard is in an unknown state: this process must not write to it again
        finally:
            stop.set()
            try:
                ep.env.close()
            except Exception:
                pass
            try:
                for f in lock.iterdir():
                    f.unlink()
                lock.rmdir()
            except Exception:
                pass
        if args.max_episodes and n_done >= int(args.max_episodes):
            break
    left = [1 for s, ts in eps if not (out / "episodes" / f"s{s}_ts{ts:g}" / "result.json").exists()]
    return 75 if left and not args.max_episodes else 0


def run_cmd(args) -> int:
    if int(args.workers) <= 1:
        return worker(args)
    import subprocess

    argv = list(sys.argv[1:])
    if "--workers" in argv:
        k = argv.index("--workers")
        del argv[k:k + 2]
    logdir = Path(args.out) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for w in range(int(args.workers)):
        f = open(logdir / f"worker_{os.environ.get('SLURM_JOB_ID') or os.environ.get('LSB_JOBID', 'x')}_{os.uname().nodename.split('.')[0]}_"
                          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'g').replace(',', '-')}_{w}.log", "a")
        procs.append(subprocess.Popen([sys.executable, __file__] + argv + ["--workers", "1"], stdout=f,
                                      stderr=subprocess.STDOUT))
        time.sleep(float(args.stagger))
    rcs = [p.wait() for p in procs]
    print(f"[asens] worker exit codes {rcs}", flush=True)
    return 0 if all(r == 0 for r in rcs) else (75 if all(r in (0, 75) for r in rcs) else 1)


def table(args) -> int:
    out = Path(args.out)
    rows, eps = [], []
    for p in sorted((out / "episodes").glob("*/result.json")):
        d = json.loads(p.read_text())
        rows += d["rows"]
        tm = d["timing"]
        eps.append({"episode": p.parent.name, "time_scale": d["meta"]["time_scale"], "held_out": d["meta"]["held_out"],
                    "n_states": d["n_states"], "tipped": d["tipped"], "wall_s": d.get("wall_s"),
                    "s_per_stored_tick": tm["pass2"] / max(tm["n_pass2_ticks"], 1),
                    "stored_ticks": tm["n_pass2_ticks"],
                    "vs_real_h": [f["vs_real"]["h_tall"] for f in d["fidelity"]],
                    "vs_real_ee": [f["vs_real"]["ee"] for f in d["fidelity"]],
                    "restore_err": max(max(f["state_checks"].values()) for f in d["fidelity"]) if d["fidelity"] else None})
    R = {"episodes": eps, "rows": rows, "decisions": DECISIONS, "held_out_seeds": sorted(HELD_OUT)}
    tmp = out / ".branch_table.json.tmp"
    tmp.write_text(json.dumps(_jsonable(R), indent=1))
    tmp.replace(out / "branch_table.json")
    print(f"{len(eps)} episodes, {len({r['state_id'] for r in rows})} states, {len(rows)} branches, "
          f"{sum(r['stored_ticks'] + 1 + N_CTX for r in rows)} stored rows")
    for name, H, c in DECISIONS:
        rr = [r for r in rows if r["decision"] == name]
        if rr:
            print(f"  {name:7s} n {len(rr):3d}  fell {np.mean([r['fell'] for r in rr]):.2f}  V95 mean "
                  f"{np.mean([r['V95'] for r in rr]):+.3f}  ticks {np.mean([r['n_ticks'] for r in rr]):.0f} "
                  f"stored {np.mean([r['stored_ticks'] for r in rr]):.0f}  pass-2 outcome differs "
                  f"{sum(r['fell'] != r['pass2_fell'] for r in rr)}")
    by = {}
    for r in rows:
        by.setdefault(r["state_id"], []).append(r["V95"])
    sp = [max(v) - min(v) for v in by.values()]
    if sp:
        print(f"  V95 spread over a state's {len(rows) // max(len(by), 1)} decisions: mean {np.mean(sp):.3f}, median "
              f"{np.median(sp):.3f}; states with spread > 0.05: {sum(s > 0.05 for s in sp)} of {len(sp)}")
    for e in eps:
        print(f"  {e['episode']}: states {e['n_states']} tipped {e['tipped']} wall {e['wall_s'] or 0:.0f}s "
              f"{e['s_per_stored_tick']:.2f} s/stored tick; planner branch vs the real continuation: h "
              f"{max(e['vs_real_h'] or [0]):.2e} ee {max(e['vs_real_ee'] or [0]):.2e} m; restore error "
              f"{e['restore_err']:.1e}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episodes", default="", help="seed:time_scale,... (default: the 25 planned episodes)")
    r.add_argument("--out", default="", help="default: data/lang_asens (round 1), data/lang_asens2 (round 2)")
    r.add_argument("--round", type=int, default=1, choices=sorted(N_ROUND))
    r.add_argument("--max-states", type=int, default=0, help="smoke: this many branch states per episode")
    r.add_argument("--max-episodes", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--stagger", type=float, default=30.0)
    r.add_argument("--capacity", type=int, default=24000, help="rows of this process's branch shard")
    r.add_argument("--parent-capacity", type=int, default=6000)
    r.add_argument("--rows-per-episode", type=int, default=4500, help="room a new episode needs in the shard")
    r.add_argument("--episode-est", type=float, default=1500.0, help="seconds per episode before one is measured")
    t = sub.add_parser("table")
    t.add_argument("--out", default="")
    t.add_argument("--round", type=int, default=1, choices=sorted(N_ROUND))
    args = ap.parse_args(argv)
    args.out = args.out or str(OUT_ROUND[int(args.round)])
    args.out = str(Path(args.out).resolve())   # the env build chdirs into RoboTwin: every path must be absolute
    if args.cmd == "run":
        import main.lang  # noqa: F401  (main.item_envs on the path)

        return run_cmd(args)
    return table(args)


if __name__ == "__main__":
    sys.exit(main())
