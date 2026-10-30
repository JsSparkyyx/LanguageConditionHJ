#!/usr/bin/env python
"""Checks 2 and 3 of the 2026-10-30 diagnosis: T5 ground truth by branching the simulation, and the critic's ranking
of the same branches.

    # one worker loop over an episode list (resumable: an episode whose result.json exists is skipped)
    python scripts/lang/diag_1030/t5_branch.py run --episodes 5001:1.3,5002:1.0 [--smoke] [--workers 4]
    # aggregate every finished episode into results.json (light numpy, no torch)
    python scripts/lang/diag_1030/t5_branch.py aggregate

A T5 (tray_transport, ur5-wsg) episode is built the way main/lang/eval_closed.py filter_run builds a filtered one
(LangRolloutController + RiskTickController, the vanilla final checkpoint loaded, the named rule = the TALL block's
upright rule with a training sentence, the short block's sentence as the counterfactual), in monitor mode: the
PLANNER drives every tick (the filter never engages). Every tick the four vanilla checkpoints score Q(s, a_nom)
and their actor's mu / sigma on the captured frames.

At each branch tick (inside the controller's tick hook, i.e. at a plan-tick boundary, during the risky drag every
--branch-every ticks, plus the last two ticks of the pre-drag descent as controls) the whole simulation is
snapshotted (physx pack + every articulation's qpos / qvel / qf / drive targets, the robot's gripper values, the
env's step counters and tick accumulator, the env's held-object latch), and the branches run from it, each restored
before it starts (the first planner branch runs from the live state):

    plan    the remaining plan rows exactly as TickChunkedController plays them (set_arm_joints(pos[row], vel[row])
            per physics step), then hold
    stop    env.step_dtheta(0) every tick (what a zero actor action does: an abrupt freeze)
    half    the plan with a fractional row cursor advancing 0.5 rows per physics step (positions interpolated,
            velocities x 0.5), then hold
    slow    the rate falls linearly 1 -> 0 over 25 ticks along the plan, then hold
    pause3  step_dtheta(0) for 3 ticks (one filter block), then the plan rows from where it paused
    plan2   the planner branch again (restore fidelity: compared with plan, and with the real continuation)

and at --actor-leads branch ticks (lead times to the planner's brake) each first action X in {plan, stop, half,
slow} is also continued by the final checkpoint's actor: X's first tick, then the DETERMINISTIC actor every tick
(X_det) and X's first tick then the STOCHASTIC actor (X_sto0..3, 4 seeds). Branches run until 50 ticks (2 s) after
the last motion of the arm, the tray and the blocks (ee / tray / block speed < 0.01 m/s, block spin < 0.1 rad/s),
capped at 300 ticks (stochastic-actor continuations never settle: capped at --stoch-cap ticks).

Per branch: per-tick h of both upright rules, tilts, tray position / velocity, EE position; min h_tall (the ground
truth), min h_short, max tilt, peak tray deceleration, success (tray within 4 cm of the pad), V_g by the backward
recursion y(t) = (1-g) h_t + g min(h_t, y(t+1)), y(T) = h_T on the h_tall trace (g = 0.95, 0.98, 0.99).
Critic side at each branch tick: Q(s, a) of every checkpoint, with the tall and the short sentence, for the
candidate first actions [a_nom, 0, 0.5 a_nom, half's first-tick command, slow's first-tick command, pause3's
(= 0), the actor mean, 8 actor samples, 16 uniform random actions], on a frame rendered at the snapshot state.
The raw encoder inputs at every branch tick go to <out>/states/<episode>.pt (README_states.md).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
from collections import deque
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

OUT = REPO / "runs/lang/diag_1030/t5_branch"
VAN = REPO / "runs/lang/rounds_ur5_g095_1023"
CKPTS = ["checkpoint_round10.pt", "checkpoint_round20.pt", "checkpoint_round30.pt", "checkpoint.pt"]
CACHES = [REPO / "data/lang_1023/warmup/text_cache.pt", REPO / "data/lang_1023/val/text_cache.pt",
          REPO / "data/lang_1023/val20/text_cache.pt"]
# production episodes: (seed, time_scale); a mix of cruise speeds, 0.5 never tips (calibration), 0.75 sometimes
DEFAULT_EPISODES = [(5001, 1.3), (5002, 1.15), (5003, 1.0), (5004, 1.3), (5005, 1.15), (5006, 1.0), (5007, 0.75),
                    (5008, 1.3), (5009, 1.15), (5010, 1.0), (5011, 0.75), (5012, 0.5)]
SPT = 10
TICK_DT = 0.04
SETTLE_TICKS = 50
CAP_TICKS = 300
# "moving" for the settle rule: EE or tray faster than 1 cm/s, a block faster than 2 cm/s, a block's tilt changing
# faster than 1 deg/s (the smoke: resting blocks jitter above 0.1 rad/s of spin while their tilt stays constant)
V_LIN, V_BLK, TILT_RATE = 0.01, 0.02, 1.0
PERT_W, PERT_V = 0.05, 0.001   # rad/s, m/s: the perturbed repeats' kick to the tall block
N_PERT = 3                     # perturbed repeats per physical branch (plan, stop, half, slow)
GAMMAS = (0.95, 0.98, 0.99)
N_SAMPLES, N_RAND = 8, 16
FIRST_ACTIONS = ("plan", "stop", "half", "slow")


def vrec(h, g: float) -> float:
    """y(T) = h_T; y(t) = (1 - g) h_t + g min(h_t, y(t + 1)); returns y(0)."""
    h = np.asarray(h, np.float64)
    if h.size == 0:
        return float("nan")
    y = h[-1]
    for t in range(h.size - 2, -1, -1):
        y = (1 - g) * h[t] + g * min(h[t], y)
    return float(y)


def _copy(v):
    try:
        return copy.deepcopy(v)
    except Exception:
        return copy.copy(v)


# ---------------------------------------------------------------------------
# models (eval_closed.Models without writing into the run dir)
# ---------------------------------------------------------------------------


class Model:
    def __init__(self, path: Path, text_store=None, extra_path: Path | None = None):
        import torch

        from main.lang.config import LangConfig
        from main.lang.legacy_pool import checkpoint_vocab, install_vocab
        from main.lang.network import LangPolicyEncoder, LangTwinCritic, make_actor
        from main.lang.test_words import install as no_banned_test_words
        from main.lang.text import TextStore
        from main.lang.train import load_policy

        ck = torch.load(path, map_location="cuda", weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        vocab = checkpoint_vocab(ck["lang_config"])
        install_vocab(vocab)
        no_banned_test_words()
        if vocab != self.cfg.vocab:
            self.cfg = dataclasses.replace(self.cfg, vocab=vocab, bank_hash=self.cfg.bank_hash)
        self.name = Path(path).stem
        self.step = int(ck.get("step", 0))
        self.pe = LangPolicyEncoder(self.cfg).cuda().eval()
        load_policy(self.pe, ck)
        self.actor = make_actor(self.cfg).cuda().eval()
        self.actor.load_state_dict(ck["actor"])
        self.critics = LangTwinCritic(self.cfg).cuda().eval()
        self.critics.load_state_dict(ck["critics"])
        if text_store is None:
            run = Path(path).parent
            extra = [run / "text_cache_extra.pt"] if (run / "text_cache_extra.pt").exists() else []
            text_store = TextStore([c for c in CACHES if c.exists()] + extra, self.cfg.text_backend, "cuda",
                                   extra_path=extra_path, backend_paths=self.cfg.backend_paths)
        self.text = text_store
        self.text_out = {}
        del ck

    def set_text(self, key: str, sentence: str):
        import torch

        with torch.no_grad():
            w, m = self.text.lookup([sentence])
            self.text_out[key] = self.pe.encode_text(w, m)


def enc_rep(enc, B: int):
    from main.lang.network import LangEncoded

    return LangEncoded(**{f.name: getattr(enc, f.name).expand(B, *getattr(enc, f.name).shape[1:])
                          for f in dataclasses.fields(enc)})


# ---------------------------------------------------------------------------
# simulation snapshot
# ---------------------------------------------------------------------------


class Snap:
    """Everything a branch can change: physx state, articulation qpos / qvel / qf / drive targets, the robot's
    gripper values (joint_state reads them), the env's counters / tick accumulator / held latch, the task's flags."""

    def __init__(self, env):
        task = env.task
        px = task.scene.physx_system
        self.data = px.pack()
        self.arts = []
        for art in task.scene.get_all_articulations():
            js = art.get_active_joints()
            self.arts.append((art, np.array(art.get_qpos()).copy(), np.array(art.get_qvel()).copy(),
                              np.array(art.get_qf()).copy(),
                              [(j, np.array(j.get_drive_target()).copy(), np.array(j.get_drive_velocity_target()).copy())
                               for j in js]))
        self.rigid = [(c, np.concatenate([c.entity.get_pose().p, c.entity.get_pose().q]), np.array(c.get_linear_velocity()).copy(),
                       np.array(c.get_angular_velocity()).copy()) for c in px.get_rigid_dynamic_components()]
        r = task.robot
        self.grip = (r.left_gripper_val, r.right_gripper_val)
        self.env_cnt = (env.n_control_steps, env.n_physics_steps, env._substep_acc)
        self.env_py = {k: _copy(v) for k, v in vars(env).items() if k.startswith("_cehj_")}
        self.task_py = {k: _copy(getattr(task, k)) for k in ("plan_success", "active_time_scale", "_diag_in_risky")
                        if hasattr(task, k)}

    def restore(self, env) -> dict:
        task = env.task
        px = task.scene.physx_system
        px.unpack(self.data)
        for art, q, qv, qf, drives in self.arts:
            art.set_qpos(q)
            art.set_qvel(qv)
            art.set_qf(qf)
            for j, t, v in drives:
                j.set_drive_target(t)
                j.set_drive_velocity_target(v)
        r = task.robot
        r.left_gripper_val, r.right_gripper_val = self.grip
        env.n_control_steps, env.n_physics_steps, env._substep_acc = self.env_cnt
        for k in [k for k in vars(env) if k.startswith("_cehj_") and k not in self.env_py]:
            delattr(env, k)
        for k, v in self.env_py.items():
            setattr(env, k, _copy(v))
        for k, v in self.task_py.items():
            setattr(task, k, _copy(v))
        task._update_render()
        return self.check(env)

    def check(self, env) -> dict:
        """Max deviation of the live state from the snapshot (after a restore it should be ~0)."""
        e_pose = e_vel = e_q = e_dt = 0.0
        for c, p, v, w in self.rigid:
            pc = np.concatenate([c.entity.get_pose().p, c.entity.get_pose().q])
            dq = min(np.abs(pc[3:] - p[3:]).max(), np.abs(pc[3:] + p[3:]).max())
            e_pose = max(e_pose, float(np.abs(pc[:3] - p[:3]).max()), float(dq))
            e_vel = max(e_vel, float(np.abs(np.array(c.get_linear_velocity()) - v).max()),
                        float(np.abs(np.array(c.get_angular_velocity()) - w).max()))
        for art, q, qv, qf, drives in self.arts:
            e_q = max(e_q, float(np.abs(np.array(art.get_qpos()) - q).max()) if q.size else 0.0,
                      float(np.abs(np.array(art.get_qvel()) - qv).max()) if qv.size else 0.0)
            for j, t, v in drives:
                e_dt = max(e_dt, float(np.abs(np.array(j.get_drive_target()) - t).max()),
                           float(np.abs(np.array(j.get_drive_velocity_target()) - v).max()))
        return {"rigid_pose": e_pose, "rigid_vel": e_vel, "art_qpos_qvel": e_q, "drive_targets": e_dt}


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------


def make_controller_class():
    from main.lang.rollout import LangRolloutController
    from main.train.rollout import RolloutController

    class BranchController(LangRolloutController):
        """Monitor-mode language controller (the planner always drives) with the branch machinery in its tick hook."""

        diag = None   # the Episode, set before run()

        def _encode_obs(self, obs):
            out = super()._encode_obs(obs)
            if self._in_capture:
                self._raw_cap = out
            return out

        def _capture_tick(self):
            obs = super()._capture_tick()
            if self.diag is not None:
                self.diag.on_capture()
            return obs

        def _tick_hook(self, ctx) -> bool:
            ret = super()._tick_hook(ctx)
            if ret:
                raise RuntimeError("monitor mode: the filter engaged")
            if self.diag is not None:
                self.diag.on_tick(ctx)
            return False

        def raw_encode(self, obs):
            return RolloutController._encode_obs(self, obs)

    return BranchController


class Episode:
    def __init__(self, args, seed: int, ts: float, models: list, encoder, kin, worker_tag: str, actor: bool = True):
        self.args, self.seed, self.ts = args, int(seed), float(ts)
        self.actor_leads = list(args.actor_leads) if actor else []   # no actor continuations when False
        self.models = models            # Model list; the last one is the final checkpoint (drives the continuations)
        self.final = models[-1]
        self.encoder, self.kin = encoder, kin
        self.name = f"s{self.seed}_ts{self.ts:g}"
        self.dir = Path(args.out) / "episodes" / self.name
        self.worker_tag = worker_tag
        self.ticks = []                 # per hook tick of the nominal episode
        self.caps = []                  # per capture
        self.branch_pts = []            # per branch tick
        self.states = []                # raw encoder inputs per branch tick
        self.drag = None                # {"plan_id", "n", "start_hook", "start_t", "brake_row", "arm"}
        self.drag_plans_seen = set()
        self.ctrl_plan_seen = {}
        self.timing = {"branch_phys": 0.0, "actor_roll": 0.0, "score": 0.0, "n_actor_ticks": 0, "n_phys_ticks": 0}
        self.rng = np.random.RandomState(self.seed + 7)

    # ----------------------------------------------------------- build / run --
    def run(self):
        import torch

        from main.item_envs import language
        from main.item_envs.controller import RiskTickController
        from main.lang.eval_closed import hsel, set_global_seeds
        from main.lang.rollout import wire_item_eval

        os.environ["ITEM_T5_REFERENT"] = "tall"
        cfg = self.final.cfg
        cfg_ep = dataclasses.replace(cfg, task="tray_transport", embodiment="ur5-wsg", item_host=None,
                                     item_active_rule=None, item_risk={"time_scale": self.ts},
                                     item_out_root=str(Path(self.args.out) / "_runner" / self.worker_tag))
        set_global_seeds(self.seed)
        Ctl = make_controller_class()
        t0 = time.perf_counter()
        ro = Ctl(cfg_ep, self.seed, mode="filtered", encoder=self.encoder, kin=self.kin, policy_enc=self.final.pe,
                 actor=self.final.actor, critics=self.final.critics, filter_active=True, monitor_only=True,
                 episode_id=self.seed, rng=np.random.RandomState(self.seed),
                 max_steps=10 * int(cfg.max_steps_per_episode), controller_cls=RiskTickController,
                 text_store=self.final.text)
        self.ro, self.env = ro, ro.env
        task = self.env.task
        self.task, self.runner = task, self.env._item_runner
        self.build_s = time.perf_counter() - t0
        # rules, sentences
        rec = language.episode_record(self.runner, self.seed)
        bank = language.load_bank()
        kind_of = {it["item_id"]: it["kind"] for it in task.standing}
        self.upr = [r for r in self.runner.rules if r.consequence.value == "upright"]
        self.rule_kind = [kind_of[r.item_ids[0]] for r in self.upr]
        tall = next(r for r, k in zip(self.upr, self.rule_kind) if k == "tall")
        short = next(r for r, k in zip(self.upr, self.rule_kind) if k == "short")
        self.i_tall, self.i_short = self.upr.index(tall), self.upr.index(short)
        rr = {r["rule_id"]: r for r in rec["rules"]}
        self.sent_tall = hsel(language.sentences(rr[tall.rule_id], rec, "train", bank), (self.seed, "train"))
        self.sent_short = hsel(language.sentences(rr[short.rule_id], rec, "train", bank), (self.seed, "train", "cf"))
        for m in self.models:
            m.set_text("tall", self.sent_tall)
            m.set_text("short", self.sent_short)
        wire_item_eval(ro, tall.rule_id)
        ro.set_instruction(self.sent_tall, self.sent_short, rule_id=tall.rule_id, cf_rule_id=short.rule_id)
        self.meta = {"seed": self.seed, "time_scale": self.ts, "cruise_speed": 0.30 * self.ts,
                     "tall_rule": tall.rule_id, "short_rule": short.rule_id, "sentence_tall": self.sent_tall,
                     "sentence_short": self.sent_short, "items": task.episode_info.get("items"),
                     "carry_arm": task.carry_arm, "start_xy": task.start_xy.tolist(),
                     "target_xy": task.target_xy.tolist(), "safe_time_scale": float(task._time_scale_safe),
                     "ckpts": [(m.name, m.step) for m in self.models], "rt": bool(_use_rt())}
        self._patch_risky()
        # the joint layout of the action vector
        ex = ro.extractor
        self.nj = ex.spec.n_joints
        self.n_arm = self.nj - 2
        bt = ex.extract()
        self.joint_index = bt["joint_index"].cuda()                                  # [M]
        self.dmax = torch.from_numpy(ex.delta_theta_max.astype(np.float32)).cuda()   # [A]
        ji = bt["joint_index"].numpy()
        self.valid_cols = np.unique(ji[ji >= 0])
        ro.diag = self
        torch.manual_seed(self.seed)
        t1 = time.perf_counter()
        with torch.no_grad():
            trace = ro.run()
        self.run_s = time.perf_counter() - t1
        return self.finish(trace)

    def _patch_risky(self):
        mod = sys.modules[type(self.task).__module__]
        orig = mod.risky
        task = self.task

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
        task._diag_in_risky = False
        self._unpatch = lambda: setattr(mod, "risky", orig)

    # ------------------------------------------------------------ physics ----
    def phys(self) -> dict:
        import sapien

        from main.item_envs.margins import rule_h
        from main.item_envs.runner import rule_margins

        task, env, runner = self.task, self.env, self.runner
        tilts = {it["kind"]: task._tilt_deg(it["entity"], [0.0, 0.0, 1.0]) for it in task.standing}
        m = rule_margins(env, runner.items, runner.regions, self.upr, {}, np.zeros(len(runner.items)))
        h = [float(rule_h(r, v)) for r, v in zip(self.upr, m)]
        comp = task.tray.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        bv, bw = 0.0, 0.0
        for it in task.standing:
            c = getattr(it["entity"], "actor", it["entity"]).find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            bv = max(bv, float(np.linalg.norm(c.get_linear_velocity())))
            bw = max(bw, float(np.linalg.norm(c.get_angular_velocity())))
        ee = np.asarray(task.get_arm_pose(task.carry_arm), np.float64).reshape(-1)[:3]
        return {"h_tall": h[self.i_tall], "h_short": h[self.i_short], "tilt_tall": tilts["tall"],
                "tilt_short": tilts["short"], "tray_p": np.asarray(task.tray.get_pose().p, np.float64),
                "tray_v": np.asarray(comp.get_linear_velocity(), np.float64), "ee": ee, "blk_v": bv, "blk_w": bw}

    # ------------------------------------------------------------ captures ----
    def on_capture(self):
        ro = self.ro
        p = self.phys()
        p["t"] = ro._n_physics / self.env.PHYSICS_FREQ
        p["in_drag"] = bool(getattr(self.task, "_diag_in_risky", False))
        self.caps.append(p)

    # ---------------------------------------------------------------- hook ----
    def on_tick(self, ctx):
        import torch

        ro, task = self.ro, self.task
        arm = task.carry_arm
        plan = ctx.get(f"{arm}_arm")
        n = 0 if plan is None else int(np.asarray(plan["position"]).shape[0])
        i = int(ctx["i"])
        in_risky = bool(getattr(task, "_diag_in_risky", False))
        a_nom = ro._anom(ctx)                                    # [1, A] cuda, live qpos
        rec = {"t": ro._n_physics / self.env.PHYSICS_FREQ, "i": i, "n": n, "in_drag": in_risky, "plan_id": id(plan),
               "plan_local": bool(getattr(task, "plan_local", False)), "cap": len(self.caps) - 1}
        rec.update({k: v for k, v in self.phys().items()})
        # every tick: each checkpoint's Q(s, a_nom) and actor stats on the capture's frames (what the filter sees)
        t0 = time.perf_counter()
        raw = getattr(ro, "_raw_cap", None)
        shim = ro.policy_enc
        if raw is not None and shim.hist:
            tok, pos, state, body = raw
            st, sp = shim._stack(shim.hist, shim.frame0, shim.hist[-1])
            rec["score"] = self.score_tick(state, body, st, sp, a_nom)
        self.timing["score"] += time.perf_counter() - t0
        # drag bookkeeping
        if in_risky and plan is not None and self.drag is None:
            pos, vel = np.asarray(plan["position"]), np.asarray(plan.get("velocity", np.zeros_like(plan["position"])))
            sp_ = np.linalg.norm(vel, axis=1)
            brake_row = int(np.nonzero(sp_ >= 0.9 * sp_.max())[0][-1]) if sp_.size and sp_.max() > 0 else n - 1
            self.drag = {"plan_id": id(plan), "n": n, "start_hook": len(self.ticks), "start_t": rec["t"],
                         "start_cap": len(self.caps) - 1, "brake_row": brake_row, "arm": arm,
                         "speed_profile_max": float(sp_.max()) if sp_.size else 0.0}
            n_ticks = (n + SPT - 1) // SPT
            bt = list(range(0, n_ticks, int(self.args.branch_every)))
            if self.args.max_branch_ticks:
                # smoke: spread the few branch ticks over the drag (start, before the brake, at the brake)
                brake_tick = brake_row // SPT
                want = [0, max(brake_tick - 12, 0), brake_tick][: int(self.args.max_branch_ticks)]
                bt = sorted({min(bt, key=lambda b: abs(b - w)) for w in want})
            self.drag["branch_ticks"] = bt
            brake_tick = brake_row / SPT
            act = set()
            for L in self.actor_leads:
                cand = [b for b in bt if b not in act]
                if cand:
                    act.add(min(cand, key=lambda b: abs((brake_tick - b) * TICK_DT - L)))
            self.drag["actor_ticks"] = sorted(act)
        elif in_risky and plan is not None and id(plan) != self.drag["plan_id"] and id(plan) not in self.drag_plans_seen:
            self.drag_plans_seen.add(id(plan))
            print(f"[t5_branch] {self.name}: a second risky plan (n={n}); branching only on the first", flush=True)
        kind = None
        if self.drag is not None and in_risky and plan is not None and id(plan) == self.drag["plan_id"]:
            j = i // SPT
            if i % SPT == 0 and j in self.drag["branch_ticks"]:
                kind = "drag"
        elif self.drag is None and not in_risky and rec["plan_local"] and plan is not None and n > 2 * SPT:
            if (n - 1 - i) // SPT < 2:
                kind = "control"
        self.ticks.append(rec)
        if kind is not None:
            # branches step the RAW scene (no capture, no step count, no trace / history push of the main episode)
            hooked = task.scene.step
            task.scene.step = ro._orig_step
            try:
                self.branch_point(ctx, kind, a_nom)
            except Exception as exc:
                traceback.print_exc()
                self.branch_pts.append({"error": f"{type(exc).__name__}: {exc}"[:500], "t": rec["t"], "i": i,
                                        "kind": kind})
                if getattr(self, "_snap", None) is not None:
                    self._snap.restore(self.env)
                    self._snap = None
            finally:
                task.scene.step = hooked

    # -------------------------------------------------------------- scoring --
    def _enc(self, m, key, state, body, st, sp):
        return m.pe(state, body["body_tokens"][None].cuda(), body["link_pos"][None].cuda(),
                    body["body_mask"][None].cuda(), body["joint_index"][None].cuda(), st, sp,
                    text_out=m.text_out[key])

    def actor_stats(self, m, enc):
        M = self.joint_index.shape[0]
        mu, log_std = m.actor.head(enc.body[:, :M]).unbind(-1)
        valid = (self.joint_index >= 0)
        mu, ls = mu[0][valid], log_std.clamp(-5, 0)[0][valid]
        return {"mu": mu.cpu().numpy().astype(np.float32), "sigma": ls.exp().cpu().numpy().astype(np.float32),
                "abs_tanh_mu": float(torch_tanh(mu).abs().mean())}

    def score_tick(self, state, body, st, sp, a_nom) -> dict:
        import torch

        out = {}
        Jl, Ja = body["Jlin"][None].cuda(), body["Jang"][None].cuda()
        for m in self.models:
            for key in ("tall", "short"):
                enc = self._enc(m, key, state, body, st, sp)
                a_det = m.actor(enc.body, self.joint_index, self.dmax, deterministic=True)[0]
                A = torch.cat([a_nom, a_det, torch.zeros_like(a_nom)])
                q = m.critics.qmin(enc_rep(enc, 3), A, Jl.expand(3, -1, -1, -1), Ja.expand(3, -1, -1, -1),
                                   self.joint_index).cpu().numpy()
                d = {"q_nom": float(q[0]), "q_det": float(q[1]), "q_zero": float(q[2])}
                if key == "tall":
                    d.update(self.actor_stats(m, enc))
                    d["a_det_absmax_frac"] = float((a_det.abs() / self.dmax.clamp_min(1e-9)).max())
                out[f"{m.name}|{key}"] = d
        return out

    def score_branch(self, state, body, st, sp, cands: dict) -> dict:
        """Q of every candidate first action, per checkpoint and sentence; plus the actor's mean and 8 samples
        (each checkpoint's own) and 16 uniform random actions (shared)."""
        import torch

        names = list(cands)
        A0 = torch.stack([cands[k] for k in names])                                   # [K, A]
        U = np.zeros((N_RAND, self.dmax.shape[0]), np.float32)
        U[:, self.valid_cols] = self.rng.uniform(-1, 1, (N_RAND, self.valid_cols.size))
        R = torch.from_numpy(U).cuda() * self.dmax
        Jl, Ja = body["Jlin"][None].cuda(), body["Jang"][None].cuda()
        out = {"names": names, "rand": R.cpu().numpy()}
        for m in self.models:
            for key in ("tall", "short"):
                enc = self._enc(m, key, state, body, st, sp)
                a_det = m.actor(enc.body, self.joint_index, self.dmax, deterministic=True)[0]
                a_smp = m.actor(enc_rep(enc, N_SAMPLES).body, self.joint_index, self.dmax)[0]
                A = torch.cat([A0, a_det, a_smp, R])
                B = A.shape[0]
                e = enc_rep(enc, B)
                q1, q2, _, _ = m.critics(e, A, Jl.expand(B, -1, -1, -1), Ja.expand(B, -1, -1, -1), self.joint_index)
                q1, q2 = q1.cpu().numpy(), q2.cpu().numpy()
                K = len(names)
                qm = np.minimum(q1, q2)
                d = {"q1": q1[:K], "q2": q2[:K], "qmin": qm[:K], "q_det": float(qm[K]),
                     "q_samples": qm[K + 1:K + 1 + N_SAMPLES], "q_rand": qm[K + 1 + N_SAMPLES:],
                     "a_det": a_det[0].cpu().numpy(), "a_samples": a_smp.cpu().numpy()}
                if key == "tall":
                    d.update(self.actor_stats(m, enc))
                out[f"{m.name}|{key}"] = d
        return out

    # ------------------------------------------------------------- branches --
    def _plans(self, ctx):
        out = []
        for arm_i, arm in enumerate(("left", "right")):
            p = ctx.get(f"{arm}_arm")
            if p is not None:
                from main.envs.controller import _path_arrays

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

    def _rate(self, kind, k):
        """Rows per physics step at branch physics step k (0-based)."""
        if kind in ("plan", "plan2"):
            return 1.0
        if kind == "half":
            return 0.5
        if kind == "slow":
            return max(0.0, 1.0 - (k + 1) / (25.0 * SPT))
        if kind == "pause3":
            return 0.0 if k < 3 * SPT else 1.0
        raise ValueError(kind)

    def first_cmd(self, ctx, kind):
        """The commanded one-tick displacement of a branch's first tick (target at the tick's end - live qpos)."""
        import torch

        qraw = self.ro.extractor.raw_qpos().astype(np.float64)
        a = np.zeros(2 * self.nj, np.float64)
        if kind == "stop" or kind == "pause3":
            return torch.from_numpy(a.astype(np.float32)).cuda()
        i = int(ctx["i"])
        for arm_i, arm, pos, vel in self._plans(ctx):
            n = pos.shape[0]
            c = i - 1.0
            for k in range(SPT):
                c = min(c + self._rate(kind, k), n - 1.0)
            a[arm_i * self.nj: arm_i * self.nj + self.n_arm] = self._interp(pos, max(c, 0.0)) - \
                qraw[arm_i * self.nj: arm_i * self.nj + self.n_arm]
        return torch.from_numpy(a.astype(np.float32)).cuda()

    def perturb(self, seed: int):
        """A tiny kick to the tall block (angular velocity N(0, PERT_W) rad/s and linear N(0, PERT_V) m/s per axis):
        the repeats measure how sensitive a branch's outcome is near the tipping threshold."""
        import sapien

        rng = np.random.RandomState(int(seed))
        it = next(it for it in self.task.standing if it["kind"] == "tall")
        c = getattr(it["entity"], "actor", it["entity"]).find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        c.set_angular_velocity(np.asarray(c.get_angular_velocity(), np.float64) + rng.normal(0, PERT_W, 3))
        c.set_linear_velocity(np.asarray(c.get_linear_velocity(), np.float64) + rng.normal(0, PERT_V, 3))

    def run_branch(self, ctx, kind, actor=None, cap=CAP_TICKS, seed=None, hist=None, perturb=None):
        """One branch from the current (snapshot) state. kind in plan / plan2 / stop / half / slow / pause3, or
        '<X>_det' / '<X>_sto' (X's first tick, then the final checkpoint's actor); '~p<r>' suffix = a perturbed
        repeat (perturb = its seed). Returns the per-tick trace summary."""
        import torch

        env, task = self.env, self.task
        base = kind.split("~")[0]
        if base == "plan_live":
            base = "plan"
        first, cont = (base.split("_") + [None])[:2] if "_" in base else (base, None)
        if perturb is not None:
            self.perturb(perturb)
        plans = self._plans(ctx) if first not in ("stop",) else []
        i0 = int(ctx["i"])
        cur = [i0 - 1.0 for _ in plans]
        done_plan = [False for _ in plans]
        k_glob = 0
        tr = [self.phys()]
        last_move, prev_ee, prev_tilt = 0, tr[0]["ee"], (tr[0]["tilt_tall"], tr[0]["tilt_short"])
        if seed is not None:
            torch.manual_seed(int(seed))
        a_log, q_log = [], []
        zero = np.zeros(2 * self.nj)
        for tick in range(1, cap + 1):
            if cont is not None and tick > 1:
                t0 = time.perf_counter()
                a, q = self.actor_step(hist, deterministic=(cont == "det"))
                env.step_dtheta(a)
                a_log.append(a.astype(np.float32))
                q_log.append(q)
                self.timing["n_actor_ticks"] += 1
                self.timing["actor_roll"] += time.perf_counter() - t0
            elif first == "stop" or (first == "pause3" and tick <= 3):
                env.step_dtheta(zero)
                k_glob += SPT
            else:
                for s in range(SPT):
                    r = self._rate(first, k_glob)
                    k_glob += 1
                    for pi, (arm_i, arm, pos, vel) in enumerate(plans):
                        if done_plan[pi]:
                            continue
                        n = pos.shape[0]
                        c = cur[pi] + r
                        if first in ("plan", "plan2", "pause3") and r == 1.0:
                            row = int(round(c))
                            if row >= n:           # the controller stops setting targets past the plan's end
                                done_plan[pi] = True
                                continue
                            task.robot.set_arm_joints(pos[row], vel[row], arm)
                            cur[pi] = float(row)
                            continue
                        if c >= n - 1:
                            task.robot.set_arm_joints(pos[n - 1], np.zeros_like(vel[n - 1]), arm)
                            cur[pi] = n - 1.0
                            done_plan[pi] = True
                            continue
                        if r > 0:
                            task.robot.set_arm_joints(self._interp(pos, c), r * self._interp(vel, c), arm)
                        elif k_glob == 25 * SPT:   # slow: the rate reached 0: hold where it is
                            task.robot.set_arm_joints(self._interp(pos, c), np.zeros_like(vel[0]), arm)
                        cur[pi] = c
                    task.scene.step()
                if cont is not None:
                    task._update_render()
            p = self.phys()
            tr.append(p)
            self.timing["n_phys_ticks"] += 1
            ee_v = float(np.linalg.norm(p["ee"] - prev_ee)) / TICK_DT
            prev_ee = p["ee"]
            p["ee_v"] = ee_v
            tilt_rate = max(abs(p["tilt_tall"] - prev_tilt[0]), abs(p["tilt_short"] - prev_tilt[1])) / TICK_DT
            prev_tilt = (p["tilt_tall"], p["tilt_short"])
            p["tilt_rate"] = tilt_rate
            moving = (ee_v > V_LIN or float(np.linalg.norm(p["tray_v"])) > V_LIN or p["blk_v"] > V_BLK
                      or tilt_rate > TILT_RATE)
            scripted = (cont is None and first in ("plan", "plan2", "half", "slow", "pause3") and not all(done_plan)
                        and (first != "slow" or k_glob < 25 * SPT))
            if moving or scripted:
                last_move = tick
            if tick - last_move >= SETTLE_TICKS:
                break
        return self.summarize(tr, kind, a_log, q_log)

    def actor_step(self, hist, deterministic: bool):
        """Render + encode the current state, push it into the continuation's own frame history, and return the
        final checkpoint's action (deterministic mean or a sample) and its Q(s, a)."""
        import torch

        ro, m = self.ro, self.final
        obs = self.env.get_encoder_obs(self.kin)
        tokens, pos_mean, state, body = ro.raw_encode(obs)
        tok, ps = tokens.flatten(1, 2), pos_mean.flatten(1, 2)
        hist["h"].append((tok, ps))
        st, sp = ro.policy_enc._stack(hist["h"], hist["f0"], hist["h"][-1])
        enc = self._enc(m, "tall", state, body, st, sp)
        a = m.actor(enc.body, self.joint_index, self.dmax, deterministic=deterministic)[0]
        q = float(m.critics.qmin(enc, a, body["Jlin"][None].cuda(), body["Jang"][None].cuda(), self.joint_index))
        return a[0].cpu().numpy().astype(np.float64), q

    def summarize(self, tr, kind, a_log=(), q_log=()) -> dict:
        h_tall = np.array([p["h_tall"] for p in tr])
        h_short = np.array([p["h_short"] for p in tr])
        tray_p = np.stack([p["tray_p"] for p in tr])
        tray_v = np.stack([p["tray_v"] for p in tr])
        ee = np.stack([p["ee"] for p in tr])
        acc = np.diff(tray_v[:, :2], axis=0) / TICK_DT
        d = np.asarray(self.task.target_xy) - np.asarray(self.task.start_xy)
        u = d / max(np.linalg.norm(d), 1e-9)
        decel = -(acc @ u) if acc.size else np.zeros(0)
        neg = np.nonzero(h_tall < 0)[0]
        out = {"kind": kind, "n_ticks": int(len(tr) - 1), "min_h_tall": float(h_tall.min()),
               "min_h_short": float(h_short.min()), "max_tilt_tall": float(max(p["tilt_tall"] for p in tr)),
               "max_tilt_short": float(max(p["tilt_short"] for p in tr)),
               "peak_decel": float(decel.max()) if decel.size else 0.0,
               "peak_acc_abs": float(np.linalg.norm(acc, axis=1).max()) if acc.size else 0.0,
               "first_neg_tick": int(neg[0]) if neg.size else None,
               "success": bool(np.linalg.norm(tray_p[-1, :2] - np.asarray(self.task.target_xy)) <= 0.04),
               "final_tray_dist": float(np.linalg.norm(tray_p[-1, :2] - np.asarray(self.task.target_xy))),
               "capped": bool(len(tr) - 1 >= CAP_TICKS),
               "last_tilt_tall": float(tr[-1]["tilt_tall"])}
        for g in GAMMAS:
            out[f"V{int(round(g * 100))}"] = vrec(h_tall, g)
        out["trace"] = {"h_tall": h_tall.astype(np.float32), "h_short": h_short.astype(np.float32),
                        "tilt_tall": np.array([p["tilt_tall"] for p in tr], np.float32),
                        "tilt_short": np.array([p["tilt_short"] for p in tr], np.float32),
                        "tray_p": tray_p.astype(np.float32), "tray_v": tray_v.astype(np.float32),
                        "ee": ee.astype(np.float32), "blk_v": np.array([p["blk_v"] for p in tr], np.float32),
                        "blk_w": np.array([p["blk_w"] for p in tr], np.float32)}
        if len(a_log):
            out["trace"]["actor_a"] = np.stack(a_log)
            out["trace"]["actor_q"] = np.asarray(q_log, np.float32)
            out["actor_abs_frac_mean"] = float(np.mean(np.abs(np.stack(a_log))[:, self.valid_cols]
                                                       / self.dmax.cpu().numpy()[self.valid_cols]))
        return out

    def branch_point(self, ctx, kind, a_nom):
        """Snapshot, score, run every branch (restoring before each), restore, check the restore, return."""
        import torch

        ro, env, task = self.ro, self.env, self.task
        t_start = time.perf_counter()
        i = int(ctx["i"])
        n_plan = int(np.asarray(ctx[f"{task.carry_arm}_arm"]["position"]).shape[0])
        bp = {"kind": kind, "t": ro._n_physics / env.PHYSICS_FREQ, "i": i, "j": i // SPT, "n_plan": n_plan,
              "hook_idx": len(self.ticks) - 1, "cap_idx": len(self.caps) - 1,
              "cap_offset_steps": int(ro._n_physics - round(self.caps[-1]["t"] * env.PHYSICS_FREQ)) if self.caps else None}
        if kind == "drag":
            bp["lead_to_brake_s"] = (self.drag["brake_row"] - i) / env.PHYSICS_FREQ
            bp["t_since_drag_start"] = bp["t"] - self.drag["start_t"]
        snap = Snap(env)
        self._snap = snap
        # a frame rendered at the snapshot state (the hook can sit a few physics steps after the last capture),
        # twice (the renderer's own run-to-run noise); scored as frame t with the shim's history for t-2 and 0
        obs_a = env.get_encoder_obs(self.kin)
        obs_b = env.get_encoder_obs(self.kin)
        raw_a = ro.raw_encode(obs_a)
        tok, pos_mean, state, body = raw_a
        tok_f, pos_f = tok.flatten(1, 2), pos_mean.flatten(1, 2)
        shim = ro.policy_enc
        st, sp = shim._stack(shim.hist, shim.frame0, (tok_f, pos_f))
        cands = {"a_nom": a_nom[0], "zero": torch.zeros_like(a_nom[0]), "half_lit": 0.5 * a_nom[0],
                 "half_cmd": self.first_cmd(ctx, "half"), "slow_cmd": self.first_cmd(ctx, "slow")}
        t0 = time.perf_counter()
        bp["score"] = self.score_branch(state, body, st, sp, cands)
        self.timing["score"] += time.perf_counter() - t0
        bp["cands"] = {k: v.cpu().numpy() for k, v in cands.items()}
        # the raw encoder inputs (for retrained critics), kept once the branches finished
        state_row = ({
            "state_tokens": state[0].cpu().float().numpy(), "body_tokens": body["body_tokens"].numpy(),
            "link_pos": body["link_pos"].numpy(), "body_mask": body["body_mask"].numpy(),
            "joint_index": body["joint_index"].numpy().astype(np.int64), "Jlin": body["Jlin"].numpy(),
            "Jang": body["Jang"].numpy(), "dtheta_max": self.dmax.cpu().numpy(),
            "scene_tokens": st[0].cpu().half().numpy(), "scene_pos": sp[0].cpu().float().numpy(),
            "cands": np.stack([bp["cands"][k] for k in cands]), "t": bp["t"], "kind": kind,
            "j": bp["j"], "lead_to_brake_s": bp.get("lead_to_brake_s", float("nan"))})
        # branches (physics only)
        t0 = time.perf_counter()
        # plan_live: from the live state (it still holds PhysX's contact warm start, which pack() does not carry);
        # every other branch from a restored state (restore -> run is deterministic: plan2 == plan == the real
        # continuation, which also resumes from a restore)
        order = ["plan_live", "plan", "stop", "half", "slow", "pause3", "plan2"]
        res, checks = {}, {}
        for k, b in enumerate(order):
            if k > 0:
                checks[b] = snap.restore(env)
            res[b] = self.run_branch(ctx, b)
        for b in FIRST_ACTIONS:
            for r in range(1, N_PERT + 1):
                checks[f"{b}~p{r}"] = snap.restore(env)
                res[f"{b}~p{r}"] = self.run_branch(ctx, f"{b}~p{r}", perturb=100000 * r + 1000 * bp["j"] + self.seed % 1000)
        self.timing["branch_phys"] += time.perf_counter() - t0
        # actor continuations at the actor ticks
        if kind == "drag" and bp["j"] in self.drag.get("actor_ticks", []):
            for X in FIRST_ACTIONS:
                for cont in ["det"] + [f"sto{r}" for r in range(int(self.args.n_stoch))]:
                    checks[f"{X}_{cont}"] = snap.restore(env)
                    hist = {"h": deque(list(shim.hist), maxlen=shim.k + 1), "f0": shim.frame0}
                    det = cont == "det"
                    res[f"{X}_{cont}"] = self.run_branch(
                        ctx, f"{X}_{'det' if det else 'sto'}", cap=int(self.args.det_cap) if det else int(self.args.stoch_cap),
                        seed=None if det else 1000 * self.seed + 10 * bp["j"] + int(cont[3:]), hist=hist)
                    res[f"{X}_{cont}"]["kind"] = f"{X}_{cont}"
        checks["final"] = snap.restore(env)
        self._snap = None
        # restore fidelity: plan vs plan2 (max over the overlapping ticks of the trace fields)
        fid = {}
        for tag, (x, y) in {"": ("plan", "plan2"), "live_": ("plan_live", "plan")}.items():
            a, b = res[x]["trace"], res[y]["trace"]
            L = min(len(a["h_tall"]), len(b["h_tall"]))
            for f in ("h_tall", "h_short", "tilt_tall", "tilt_short", "tray_p", "ee"):
                dd = np.abs(np.asarray(a[f][:L], np.float64) - np.asarray(b[f][:L], np.float64))
                fid[f"{tag}{f}"] = float(dd.max())
                fid[f"{tag}{f}_tick1"] = float(dd[1].max()) if L > 1 else None
        fid["n_ticks"] = [res["plan"]["n_ticks"], res["plan2"]["n_ticks"], res["plan_live"]["n_ticks"]]
        # the first rendered frame after the final restore vs the frame at the snapshot (and the render noise)
        obs_c = env.get_encoder_obs(self.kin)
        tok_c = ro.raw_encode(obs_c)[0]
        fid["img_ab_maxabs"] = float((obs_a["imgs"] - obs_b["imgs"]).abs().max())
        fid["img_ac_maxabs"] = float((obs_a["imgs"] - obs_c["imgs"]).abs().max())
        fid["img_ac_meanabs"] = float((obs_a["imgs"] - obs_c["imgs"]).abs().mean())
        fid["depth_ac_maxabs"] = float((obs_a["depths"] - obs_c["depths"]).abs().max())
        # one more render after one more scene sync with no state change: separates the renderer's dependence on
        # its update history from state that the restore missed
        task._update_render()
        obs_d = env.get_encoder_obs(self.kin)
        fid["img_cd_maxabs"] = float((obs_c["imgs"] - obs_d["imgs"]).abs().max())
        fid["img_ad_maxabs"] = float((obs_a["imgs"] - obs_d["imgs"]).abs().max())
        fid["img_ac_frac_px_gt_0.01"] = float(((obs_a["imgs"] - obs_c["imgs"]).abs() > 0.01).float().mean())
        tok_b = ro.raw_encode(obs_b)[0]
        fid["tok_ab_maxabs"] = float((tok - tok_b).abs().max())
        fid["tok_ac_maxabs"] = float((tok - tok_c).abs().max())
        fid["tok_scale"] = float(tok.abs().mean())
        fid["state_checks"] = checks
        bp["fidelity"] = fid
        bp["branches"] = res
        bp["wall_s"] = time.perf_counter() - t_start
        self.branch_pts.append(bp)
        self.states.append(state_row)
        brief = {b: (round(r["min_h_tall"], 3), round(r["V95"], 3), r["n_ticks"]) for b, r in res.items()}
        qf = bp["score"][f"{self.final.name}|tall"]["qmin"]
        print(f"[t5_branch] {self.name} {kind} j={bp['j']} t={bp['t']:.2f} lead={bp.get('lead_to_brake_s')} "
              f"wall {bp['wall_s']:.1f}s min_h/V95/ticks {brief} Q_final {np.round(qf, 4).tolist()} "
              f"plan-vs-plan2 h {fid['h_tall']:.2e} tilt {fid['tilt_tall']:.2e} img {fid['img_ac_maxabs']:.3g} "
              f"(noise {fid['img_ab_maxabs']:.3g})", flush=True)

    # --------------------------------------------------------------- finish --
    def finish(self, trace) -> dict:
        import torch

        self._unpatch()
        out = {"meta": self.meta, "status": trace.get("status"), "success": bool(trace.get("success", False)),
               "n_caps": len(self.caps), "build_s": self.build_s, "run_s": self.run_s, "timing": self.timing,
               "rollout_timing": trace.get("timing"), "drag": self.drag,
               "gpu_mem_max_gb": torch.cuda.max_memory_allocated() / 1e9}
        caps_h = np.array([c["h_tall"] for c in self.caps])
        out["nominal"] = {"min_h_tall": float(caps_h.min()) if caps_h.size else None,
                          "min_h_short": float(min(c["h_short"] for c in self.caps)) if self.caps else None,
                          "first_neg_t": (float(self.caps[int(np.argmax(caps_h < 0))]["t"]) if (caps_h < 0).any()
                                          else None),
                          "tipped": bool((caps_h < 0).any()) if caps_h.size else None}
        # real continuation vs plan / plan2 at each drag branch tick: the hook ticks of the same plan after it
        for bp in self.branch_pts:
            if bp.get("kind") != "drag" or "branches" not in bp:
                continue
            h0 = bp["hook_idx"]
            pid = self.ticks[h0]["plan_id"]
            real = [r for r in self.ticks[h0:] if r["plan_id"] == pid]
            L = min(len(real), len(bp["branches"]["plan"]["trace"]["h_tall"]))
            cmp = {}
            for f in ("h_tall", "tilt_tall", "tray_p", "ee"):
                rv = np.asarray([r[f] for r in real[:L]], np.float64)
                for b in ("plan", "plan2"):
                    bv = np.asarray(bp["branches"][b]["trace"][f][:L], np.float64)
                    cmp[f"{f}_{b}"] = float(np.abs(rv - bv).max()) if L else None
            cmp["n_ticks"] = L
            bp["fidelity"]["vs_real"] = cmp
        out["branch_points"] = self.branch_pts
        out["ticks"] = self.ticks
        out["caps"] = self.caps
        out["main_trace"] = {k: trace.get(k) for k in ("t", "V", "V_t", "Q_cf", "Q_cf_t", "h_named", "action_source")}
        return out


def torch_tanh(x):
    import torch

    return torch.tanh(x)


def _use_rt():
    try:
        from main.envs.env import use_ray_tracing

        return use_ray_tracing()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_episode(ep: Episode, out: dict):
    import torch

    d = ep.dir
    d.mkdir(parents=True, exist_ok=True)
    # states file: one row per branch tick (score_states.py / LangTrainer.encode format)
    if ep.states:
        S = ep.states
        names = ["a_nom", "zero", "half_lit", "half_cmd", "slow_cmd"]
        truth_of = {"a_nom": "plan", "zero": "stop", "half_lit": "half", "half_cmd": "half", "slow_cmd": "slow"}
        bps = [bp for bp in ep.branch_pts if "branches" in bp]
        min_h = np.array([[bp["branches"][truth_of[n]]["min_h_tall"] for n in names] for bp in bps], np.float32)
        v95 = np.array([[bp["branches"][truth_of[n]]["V95"] for n in names] for bp in bps], np.float32)
        st = {k: torch.from_numpy(np.stack([s[k] for s in S])) for k in
              ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max",
               "scene_tokens", "scene_pos")}
        st["actions"] = torch.from_numpy(np.stack([s["cands"] for s in S]))
        st["action_names"] = names
        st["sentences"] = [ep.sent_tall] * len(S)
        st["cf_sentences"] = [ep.sent_short] * len(S)
        st["min_h"] = torch.from_numpy(min_h)
        st["V95"] = torch.from_numpy(v95)
        st["t"] = torch.tensor([s["t"] for s in S])
        st["j"] = torch.tensor([s["j"] for s in S])
        st["lead_to_brake_s"] = torch.tensor([s["lead_to_brake_s"] for s in S])
        st["is_drag"] = torch.tensor([s["kind"] == "drag" for s in S])
        st["meta"] = {"episode": ep.name, **_jsonable(ep.meta)}
        sd = Path(ep.args.out) / "states"
        sd.mkdir(parents=True, exist_ok=True)
        tmp = sd / f".{ep.name}.pt.tmp"
        torch.save(st, tmp)
        tmp.replace(sd / f"{ep.name}.pt")
    # traces to npz, the rest to json
    arrays = {}
    for bi, bp in enumerate(out["branch_points"]):
        for b, r in (bp.get("branches") or {}).items():
            for f, v in r.pop("trace", {}).items():
                arrays[f"bp{bi}/{b}/{f}"] = np.asarray(v)
        sc = bp.get("score") or {}
        for key, v in sc.items():
            if isinstance(v, dict):
                for f, a in v.items():
                    if isinstance(a, np.ndarray):
                        arrays[f"bp{bi}/score/{key}/{f}"] = a
                bp["score"][key] = {f: a for f, a in v.items() if not isinstance(a, np.ndarray) or a.size <= 32}
        if "rand" in sc:
            arrays[f"bp{bi}/score/rand"] = sc.pop("rand")
    np.savez_compressed(d / "traces.npz", **arrays)
    tmp = d / ".result.json.tmp"
    tmp.write_text(json.dumps(_jsonable(out)))
    tmp.replace(d / "result.json")


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def claim(out: Path, name: str) -> bool:
    lock = out / "locks" / name
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            age = time.time() - lock.stat().st_mtime
            info = json.loads((lock / "owner.json").read_text()) if (lock / "owner.json").exists() else {}
        except Exception:
            age, info = 0, {}
        # a stale lock (a killed job): its owner's heartbeat stopped
        if age < 600 or info.get("host") == os.uname().nodename and _pid_alive(info.get("pid")):
            return False
        try:
            for f in lock.iterdir():
                f.unlink()
            lock.rmdir()
        except Exception:
            return False
    try:
        lock.mkdir()
    except FileExistsError:
        return False
    (lock / "owner.json").write_text(json.dumps({"host": os.uname().nodename, "pid": os.getpid(),
                                                 "job": os.environ.get("SLURM_JOB_ID"), "t": time.time()}))
    return True


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def worker(args) -> int:
    import threading

    import torch

    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR, make_kinematics

    out = Path(args.out)
    tag = f"{os.uname().nodename}_{os.getpid()}"
    eps = parse_episodes(args.episodes)
    todo = [(s, ts, a) for s, ts, a in eps if not (out / "episodes" / f"s{s}_ts{ts:g}" / "result.json").exists()]
    if not todo:
        print("[t5_branch] nothing to do", flush=True)
        return 0
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, "ur5-wsg")
    ck = [Path(c) if Path(c).is_absolute() else VAN / c for c in args.ckpts]
    extra = out / "_text" / f"extra_{tag}.pt"
    extra.parent.mkdir(parents=True, exist_ok=True)
    m0 = Model(ck[-1], extra_path=extra)
    models = [Model(c, text_store=m0.text) for c in ck[:-1]] + [m0]
    print(f"[t5_branch] {tag}: models {[m.name for m in models]} loaded in {time.perf_counter() - t0:.1f}s; "
          f"GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    n_done = 0
    for s, ts, act in todo:
        name = f"s{s}_ts{ts:g}"
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

        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            torch.cuda.reset_peak_memory_stats()
            ep = Episode(args, s, ts, models, encoder, kin, tag, actor=act)
            t1 = time.perf_counter()
            res = ep.run()
            res["wall_s"] = time.perf_counter() - t1
            res["settings"] = {"actor_leads": ep.actor_leads, "det_cap": int(args.det_cap),
                               "stoch_cap": int(args.stoch_cap), "n_stoch": int(args.n_stoch),
                               "branch_every": int(args.branch_every), "n_pert": N_PERT}
            save_episode(ep, res)
            n_done += 1
            print(f"[t5_branch] {name} done in {res['wall_s']:.0f}s: status {res['status']} nominal tipped "
                  f"{res['nominal']['tipped']} min_h_tall {res['nominal']['min_h_tall']} branch pts "
                  f"{len(res['branch_points'])} timing {json.dumps(_jsonable(res['timing']))} gpu max "
                  f"{res['gpu_mem_max_gb']:.2f} GB", flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            stop.set()
            try:
                for f in lock.iterdir():
                    f.unlink()
                lock.rmdir()
            except Exception:
                pass
        if args.max_episodes and n_done >= int(args.max_episodes):
            break
    return 0


def parse_episodes(spec):
    """seed:time_scale[:n] (n = no actor continuations), comma separated."""
    if not spec:
        return [(s, ts, True) for s, ts in DEFAULT_EPISODES]
    out = []
    for tok in str(spec).split(","):
        f = tok.split(":")
        out.append((int(f[0]), float(f[1]), not (len(f) > 2 and f[2] == "n")))
    return out


def run_cmd(args) -> int:
    if int(args.workers) <= 1:
        return worker(args)
    import subprocess

    procs = []
    argv = [a for a in sys.argv[1:]]
    if "--workers" in argv:
        k = argv.index("--workers")
        del argv[k:k + 2]
    logdir = Path(args.out) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    for w in range(int(args.workers)):
        f = open(logdir / f"worker_{os.uname().nodename}_{os.environ.get('SLURM_JOB_ID', 'x')}_{w}.log", "a")
        procs.append(subprocess.Popen([sys.executable, __file__] + argv + ["--workers", "1"], stdout=f,
                                      stderr=subprocess.STDOUT))
        time.sleep(float(args.stagger))
    rc = 0
    for p in procs:
        rc |= p.wait()
    return rc


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def kendall(a, b) -> tuple[float, int, int]:
    """(tau-b-like pairwise agreement over pairs with a strict order in the truth b, n agree, n pairs)."""
    n_ok = n = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            db = b[i] - b[j]
            if abs(db) <= 1e-3:
                continue
            n += 1
            n_ok += int(np.sign(a[i] - a[j]) == np.sign(db))
    return (n_ok / n if n else float("nan")), n_ok, n


def aggregate(args) -> int:
    out = Path(args.out)
    eps = sorted((out / "episodes").glob("*/result.json"))
    print(f"{len(eps)} episodes")
    R = {"episodes": [], "branch_ticks": [], "summary": {}}
    rows = []
    for p in eps:
        d = json.loads(p.read_text())
        meta = d["meta"]
        R["episodes"].append({"name": p.parent.name, **{k: meta[k] for k in ("seed", "time_scale", "cruise_speed")},
                              "nominal": d["nominal"], "status": d["status"], "success": d["success"],
                              "wall_s": d.get("wall_s"), "gpu_mem_max_gb": d.get("gpu_mem_max_gb"),
                              "timing": d.get("timing"), "n_branch_points": len(d["branch_points"]),
                              "items": meta.get("items"), "drag": {k: v for k, v in (d.get("drag") or {}).items()
                                                                   if k != "plan_id"}})
        for bp in d["branch_points"]:
            if "branches" not in bp:
                continue
            row = {"episode": p.parent.name, "time_scale": meta["time_scale"], "kind": bp["kind"], "j": bp["j"],
                   "t": bp["t"], "lead_to_brake_s": bp.get("lead_to_brake_s"),
                   "nominal_tipped": d["nominal"]["tipped"]}
            for b, r in bp["branches"].items():
                row[f"{b}|min_h"] = r["min_h_tall"]
                row[f"{b}|V95"] = r["V95"]
                row[f"{b}|V98"] = r["V98"]
                row[f"{b}|V99"] = r["V99"]
                row[f"{b}|ticks"] = r["n_ticks"]
                row[f"{b}|peak_decel"] = r["peak_decel"]
                row[f"{b}|success"] = r["success"]
                row[f"{b}|max_tilt"] = r["max_tilt_tall"]
                row[f"{b}|min_h_short"] = r["min_h_short"]
            fp = bp["branches"]["plan"].get("first_neg_tick")
            row["lead_to_plan_harm_s"] = fp * TICK_DT if fp is not None else None
            sc = bp["score"]
            for key, v in sc.items():
                if not isinstance(v, dict) or "qmin" not in v:
                    continue
                row[f"Q|{key}"] = v["qmin"]
                row[f"Qdet|{key}"] = v["q_det"]
            row["fidelity"] = {k: v for k, v in bp["fidelity"].items() if k != "state_checks"}
            row["state_check_max"] = max(max(c.values()) for c in bp["fidelity"]["state_checks"].values())
            row["score_names"] = sc.get("names")
            rows.append(row)
    R["branch_ticks"] = rows
    # ranking agreement over the four first actions, truth from the full branches
    first = {"a_nom": "plan", "zero": "stop", "half_cmd": "half", "slow_cmd": "slow"}
    ck_keys = sorted({k.split("|", 1)[1] for r in rows for k in r if k.startswith("Q|")})
    summ = {}
    for key in ck_keys:
        for truth in ("min_h", "V95"):
            ok = n = 0
            per = []
            for r in rows:
                if r["kind"] != "drag" or f"Q|{key}" not in r:
                    continue
                names = r["score_names"]
                q = [r[f"Q|{key}"][names.index(c)] for c in first]
                g = [r[f"{b}|{truth}"] for b in first.values()]
                _, a_ok, a_n = kendall(q, g)
                ok += a_ok
                n += a_n
                per.append(float(np.max(q) - np.min(q)))
            summ[f"{key}|{truth}"] = {"pair_agree": ok / n if n else None, "n_pairs": n,
                                      "mean_Q_spread_4": float(np.mean(per)) if per else None}
    R["summary"]["ranking"] = summ
    # truths vs Q at the actor points
    fit = {}
    for key in ck_keys:
        for truth in ("full", "det", "sto"):
            err = []
            for r in rows:
                if f"Q|{key}" not in r or "plan_det|V95" not in r:
                    continue
                names = r["score_names"]
                for c, X in first.items():
                    q = r[f"Q|{key}"][names.index(c)]
                    if truth == "full":
                        v = r[f"{X}|V95"]
                    elif truth == "det":
                        v = r[f"{X}_det|V95"]
                    else:
                        vs = [r[k] for k in r if k.startswith(f"{X}_sto") and k.endswith("|V95")]
                        v = float(np.mean(vs)) if vs else float("nan")
                    err.append(q - v)
            err = np.asarray(err)
            fit[f"{key}|{truth}"] = {"n": int(err.size), "mae": float(np.abs(err).mean()) if err.size else None,
                                     "signed": float(err.mean()) if err.size else None}
    R["summary"]["q_vs_truth_V95"] = fit
    R["summary"]["n_branch_ticks"] = len(rows)
    tmp = out / ".results.json.tmp"
    tmp.write_text(json.dumps(_jsonable(R), indent=1))
    tmp.replace(out / "results.json")
    print(json.dumps(_jsonable(R["summary"]), indent=1)[:4000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episodes", default="", help="seed:time_scale,... (default: the 12 production episodes)")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--ckpts", nargs="+", default=CKPTS, help="vanilla checkpoints; the LAST drives the continuations")
    r.add_argument("--branch-every", type=int, default=3)
    r.add_argument("--max-branch-ticks", type=int, default=0, help="smoke: this many drag branch ticks only")
    r.add_argument("--actor-leads", type=float, nargs="*", default=[0.8, 0.5, 0.25],
                   help="lead times (s) to the planner's brake of the actor-continuation branch ticks")
    r.add_argument("--n-stoch", type=int, default=4)
    r.add_argument("--stoch-cap", type=int, default=80)
    r.add_argument("--det-cap", type=int, default=125)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--stagger", type=float, default=20.0)
    r.add_argument("--max-episodes", type=int, default=0)
    a = sub.add_parser("aggregate")
    a.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    args.out = str(Path(args.out).resolve())   # the env build chdirs into RoboTwin: every path must be absolute
    if args.cmd == "run":
        import main.lang  # noqa: F401  (main.item_envs on the path)

        return run_cmd(args)
    return aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
