"""The planner as a 25 Hz policy plus a residual (residual/PLAN.md sections 1 and 3).

``ResidualTickController`` replaces RoboTwin's per-physics-step plan playback. Every 40 ms tick:

1. per arm, a_nom = pos[min(i + 9, n - 1)] - q_live (the plan row at the end of the tick minus the live joints).
   An arm with no plan in the current motion holds its anchor, the last row of its last plan (its start pose before
   the first one): a_nom = anchor - q_live, so the residual never adds up on it (Yuxuan 11-09);
2. the agent returns the residual delta (12 joints, |delta| <= delta_bound);
3. the robot executes clip(a_nom + noise + delta, +-dtheta_max) over the tick's 10 physics steps, the joint target
   moving linearly with velocity feed-forward cmd / 0.04 s (``Env.step_dtheta`` semantics); the gripper rows play in
   the same physics steps as RoboTwin plays them; noise ~ N(0, nom_noise_std) only in training;
4. each arm's plan cursor advances 10 rows;
5. replan an arm whose plan still plays, on the configured trigger: ``fixed`` every N ticks, ``deviation`` when any
   joint is more than the threshold off the plan row it should be at now. The new plan goes from the live joints to
   the same target and constraint (``_replan_arm``) and its cursor restarts at row 0. A failed replan ends the
   episode as a truncation.

Gripper-only segments (close / open, 300 physics steps = 30 ticks) also run through this loop, so the policy acts
every tick (Yuxuan 11-09). Untagged plans (not from this controller's ``plan``) still play through RoboTwin's own
``take_dense_action`` without ticks; their physics steps are counted (``stats["untagged_steps"]``).

The agent interface: ``agent.policy_tick(view) -> delta [12]`` with ``view = {"a_nom": [12], "future": [F, 12],
"active": [2]}``, then ``agent.tick_done()`` after the tick's physics (it may raise to end the episode).
"""

from __future__ import annotations

import time

import numpy as np

from . import ensure_paths
from .config import N_ARM, STEPS_PER_TICK, TICK_DT

ensure_paths()

from main.envs.controller import _ee_err, _has_tag, _n_wp, _path_arrays, _plan_ok  # noqa: E402
from main.item_envs.controller import RiskTickController  # noqa: E402

ARMS = ("left", "right")


class ReplanFailed(Exception):
    pass


class ResidualTickController(RiskTickController):
    controller_name = "residual_tick"

    def __init__(self, env, cfg, agent=None, noise_rng: np.random.Generator | None = None, **kw):
        super().__init__(env, **kw)
        self.cfg = cfg
        self.agent = agent
        self.noise_rng = noise_rng          # None = no executed noise on a_nom (evaluation, delta = 0 baseline)
        self.anchor: dict[str, np.ndarray | None] = {a: None for a in ARMS}
        self.last_target: dict[str, list | None] = {a: None for a in ARMS}
        self.in_loop = False
        self._loop = None                   # (plans, cursor) while the loop runs, for view_now()
        self.grip_events: list[dict] = []
        self.replan_log: list[dict] = []
        self.stats = {
            "policy_ticks": 0, "gripper_ticks": 0, "untagged_steps": 0, "untagged_calls": 0,
            "replans": 0, "replans_fixed": 0, "replans_deviation": 0, "replan_failures": 0, "replan_time": 0.0,
            "max_dev": 0.0, "clip_frac": 0.0, "clip_ticks": 0,
        }
        self._dmax = {a: None for a in ARMS}

    # ------------------------------------------------------------------ helpers ----
    def dtheta_max(self, arm: str) -> np.ndarray:
        if self._dmax[arm] is None:
            self._dmax[arm] = self._v_limit(arm) * TICK_DT * float(self.cfg.speed_kappa)
        return self._dmax[arm]

    def _q(self) -> dict[str, np.ndarray]:
        q = self._arm_qpos()
        for a in ARMS:
            if self.anchor[a] is None:
                self.anchor[a] = q[a].copy()
        return q

    def _view(self, plans: dict, cursor: dict, q: dict) -> dict:
        spt = STEPS_PER_TICK
        fut = list(self.cfg.future_ticks)
        a_nom = np.zeros(2 * N_ARM)
        future = np.zeros((len(fut), 2 * N_ARM))
        active = np.zeros(2)
        for k, a in enumerate(ARMS):
            sl = slice(k * N_ARM, (k + 1) * N_ARM)
            plan = plans.get(a)
            if plan is not None:
                pos, _ = _path_arrays(plan)
                n = pos.shape[0]
                i = cursor[a]
                a_nom[sl] = pos[min(i + spt - 1, n - 1)] - q[a]
                for f, t in enumerate(fut):
                    future[f, sl] = pos[min(i + spt * t + spt - 1, n - 1)] - q[a]
                active[k] = 1.0 if i < n else 0.0
            else:
                a_nom[sl] = self.anchor[a] - q[a]
                future[:, sl] = (self.anchor[a] - q[a])[None]
        return {"a_nom": a_nom, "future": future, "active": active}

    def view_now(self) -> dict:
        """The planner part of the state right now (stale plans while the loop runs; anchors outside it)."""
        q = self._q()
        if self._loop is not None:
            plans, cursor = self._loop
            return self._view(plans, cursor, q)
        return self._view({}, {}, q)

    def _record_grip_event(self, arm: str, g: dict) -> None:
        """Hand offset from the plan when this gripper starts to close or open (joint and EE distance)."""
        try:
            cur = float(self.robot.get_left_gripper_val() if arm == "left" else self.robot.get_right_gripper_val())
            tgt = float(np.asarray(g["result"])[-1])
            if abs(tgt - cur) < 0.05:   # a gripper call that does not open or close it
                return
            q = self._arm_qpos()[arm]
            ee = _ee_err(self.robot, arm, self.last_target[arm]) if self.last_target[arm] is not None else None
            self.grip_events.append({
                "arm": arm, "kind": "close" if tgt < cur else "open",
                "joint_off": float(np.linalg.norm(q - self.anchor[arm])) if self.anchor[arm] is not None else None,
                "ee_off": None if ee is None else float(ee),
                "tick": int(self.stats["policy_ticks"]),
            })
        except Exception as exc:  # diagnostics never break control
            print(f"[residual] grip event failed: {exc}")

    # --------------------------------------------------------------- the loop ----
    def _tick_take_dense_action(self, orig, control_seq, save_freq=-1):
        task = self.env.task
        arms = {a: control_seq.get(f"{a}_arm") for a in ARMS}
        grips = {a: control_seq.get(f"{a}_gripper") for a in ARMS}
        tagged = any(_has_tag(arms[a]) for a in ARMS)
        has_arm = any(arms[a] is not None for a in ARMS)
        has_grip = any(grips[a] is not None for a in ARMS)
        if not tagged and (has_arm or not has_grip or not self.cfg.act_in_gripper_segments):
            n = 0
            for a in ARMS:
                if arms[a] is not None and arms[a].get("position") is not None:
                    n = max(n, int(np.asarray(arms[a]["position"]).shape[0]))
                if grips[a] is not None:
                    n = max(n, int(grips[a]["num_step"]))
            self.stats["untagged_steps"] += n
            self.stats["untagged_calls"] += 1
            return orig(control_seq, save_freq)

        q = self._q()
        plans = {a: (arms[a] if _plan_ok(arms[a]) else None) for a in ARMS}
        for a in ARMS:
            if plans[a] is not None:
                self.anchor[a] = np.asarray(_path_arrays(plans[a])[0][-1], dtype=np.float64).copy()
                self.last_target[a] = plans[a].get("_cehj_target_pose")
        for a in ARMS:
            if grips[a] is not None:
                self._record_grip_event(a, grips[a])
        gripper_only = all(plans[a] is None for a in ARMS)

        spt = STEPS_PER_TICK
        cursor = {a: 0 for a in ARMS}
        since_replan = {a: 0 for a in ARMS}
        g_i = 0
        self.in_loop = True
        self._loop = (plans, cursor)
        normal_exit = False
        try:
            while task.plan_success:
                n = {a: _n_wp(plans[a]) for a in ARMS}
                n_g = max([0] + [int(grips[a]["num_step"]) - g_i for a in ARMS if grips[a] is not None])
                if all(cursor[a] >= n[a] for a in ARMS) and n_g <= 0:
                    break
                q = self._q()
                view = self._view(plans, cursor, q)
                delta = np.zeros(2 * N_ARM)
                if self.agent is not None:
                    delta = np.asarray(self.agent.policy_tick(view), dtype=np.float64).reshape(2 * N_ARM)
                noise = np.zeros(2 * N_ARM)
                if self.noise_rng is not None and self.cfg.nom_noise_std > 0:
                    noise = self.noise_rng.normal(0.0, float(self.cfg.nom_noise_std), 2 * N_ARM)
                cmd, q0 = {}, {}
                clipped = 0
                for k, a in enumerate(ARMS):
                    sl = slice(k * N_ARM, (k + 1) * N_ARM)
                    dmax = self.dtheta_max(a)
                    raw = view["a_nom"][sl] + noise[sl] + delta[sl]
                    cmd[a] = np.clip(raw, -dmax, dmax)
                    clipped += int(np.any(np.abs(raw) > dmax + 1e-12))
                    q0[a] = q[a]
                self.stats["clip_ticks"] += int(clipped > 0)
                for k in range(spt):
                    frac = (k + 1) / spt
                    for a in ARMS:
                        task.robot.set_arm_joints(q0[a] + cmd[a] * frac, cmd[a] / TICK_DT, a)
                    for a in ARMS:
                        g = grips[a]
                        if g is not None and g_i + k < int(g["num_step"]):
                            task.robot.set_gripper(g["result"][g_i + k], a, g["per_step"])
                    task.scene.step()
                    if task.render_freq and (g_i + k) % task.render_freq == 0:
                        task._update_render()
                for a in ARMS:
                    if plans[a] is not None:
                        cursor[a] += spt
                        since_replan[a] += 1
                g_i += spt
                self.stats["policy_ticks"] += 1
                self.stats["gripper_ticks"] += int(gripper_only)
                if self.agent is not None:
                    self.agent.tick_done()
                self._maybe_replan(plans, cursor, since_replan)
            normal_exit = True
        finally:
            self.in_loop = False
            # an exception (replan failure, time limit) keeps the last plans and cursor: the truncation bootstrap
            # V(s_T) then sees the state the policy was in, not the between-motions anchor view
            if normal_exit:
                self._loop = None
        return True

    def _maybe_replan(self, plans: dict, cursor: dict, since_replan: dict) -> None:
        trig = self.cfg.replan_trigger
        if trig == "none":
            return
        q = None
        for a in ARMS:
            plan = plans[a]
            if plan is None:
                continue
            n = _n_wp(plan)
            if cursor[a] >= n:      # finished: the arm holds the last row, a_nom pulls it back
                continue
            if q is None:
                q = self._arm_qpos()
            pos, _ = _path_arrays(plan)
            dev = float(np.max(np.abs(q[a] - pos[min(cursor[a] - 1, n - 1)])))
            self.stats["max_dev"] = max(self.stats["max_dev"], dev)
            fire = (trig == "fixed" and since_replan[a] >= int(self.cfg.replan_every)) or (
                trig == "deviation" and dev > float(self.cfg.replan_threshold))
            if not fire:
                continue
            # does the new path start from rest? speed of the old plan at the cursor vs the new plan's first tick
            # (RoboTwin hands cuRobo the joint positions only, JointState.from_position: zero start velocity)
            _p_old, v_old = _path_arrays(plan)
            v_plan_old = float(np.linalg.norm(v_old[min(cursor[a] - 1, n - 1)]))
            t0 = time.perf_counter()
            new = self._replan_arm(a, plan)
            self.stats["replan_time"] += time.perf_counter() - t0
            if new is None or not _plan_ok(new):
                self.stats["replan_failures"] += 1
                self.env.task.plan_success = False
                raise ReplanFailed(f"{a} replan failed (trigger {trig}, deviation {dev:.3f} rad)")
            p_new, v_new = _path_arrays(new)
            self.replan_log.append({
                "arm": a, "tick": int(self.stats["policy_ticks"]), "dev": dev, "v_plan_old": v_plan_old,
                "v_new_row0": float(np.linalg.norm(v_new[0])),
                "v_new_tick1": float(np.linalg.norm(v_new[min(STEPS_PER_TICK - 1, len(v_new) - 1)])),
                "rows_new": int(len(p_new)), "rows_left_old": int(n - cursor[a]),
            })
            self.stats["replans"] += 1
            self.stats[f"replans_{trig}"] += 1
            plans[a] = new
            cursor[a] = 0
            since_replan[a] = 0
            self.anchor[a] = np.asarray(_path_arrays(new)[0][-1], dtype=np.float64).copy()
