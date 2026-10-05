"""RiskTickController (PLAN v3 §2.5): TickChunkedController + plan retiming.

Every successful plan is retimed with the task's ``active_time_scale`` via
``retime.retime``. Velocities are NEVER clipped: retime reports
``info["v_ratio"]`` (max |v'| / URDF joint velocity limit) which lands in
``retime_log`` / ``v_ratio_events`` and on stdout.

Time-scale ownership (T1 mechanism, do not combine paths):

- T2/T3 (and smoke_items): the task's ``risky(task)`` context manager
  flips ``task.active_time_scale`` around a motion primitive; plan() only
  READS the attribute for the retime factor.
- T1: host tasks run unmodified, so risky segments are chosen by skill
  type. When ``risky_skill_labels`` is non-empty, plan() first calls
  ``_sync_time_scale_from_skill()``: the current skill label (the CEHJ
  skill tracker stamps ``task._cehj_skill`` with the label of the action
  being planned) is substring-matched against ``risky_skill_labels`` and
  ``task.active_time_scale`` is set to the risky or safe scale
  accordingly. With an empty ``risky_skill_labels`` this path is inert.

Realized-vs-commanded tracking: ``attach()`` chains this controller's
``tick_hook`` around any pre-existing hook (saved as ``_user_tick_hook``
and always honored). At each tick boundary the upcoming chunk's commanded
joint displacement (``pos[min(i+spt, n-1)] - qpos_live``, per arm, in plan
joint order) is recorded; the next boundary measures the achieved
displacement. Pairs accumulate only after the first v_ratio > 1 event
(the only regime where tracking matters); ``mean_realized_ratio`` is the
rollout.py definition — summed achieved norms / summed commanded norms
over chunks with nonzero command. All tracking is wrapped in try/except:
diagnostics never break control.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.controller lives in the sibling CEHJ copy

from main.envs.controller import TickChunkedController, current_skill  # noqa: E402

from .retime import retime  # noqa: E402
from .via_path import path_fractions, profiled_via_path  # noqa: E402

DEFAULT_JOINT_VEL = 2.0  # rad/s (or m/s prismatic) fallback, body_features.py policy
PHYSICS_DT = 1.0 / 250.0  # one plan row per physics step (cuRobo interpolation_dt, mplib time_step)
LOCAL_JUMP_MAX = 1.0  # rad: a plan_local step whose joints leave the start by more is a branch flip
BRANCH_FLIP_TRAVEL = 2.0  # rad: any-joint travel above this triggers a local (screw) retry
VIA_IK_STEP_MAX = 0.35  # rad: a via waypoint's IK fallback must stay this close (every joint) to the previous waypoint
BRANCH_FLIP_REPLANS = 2  # cuRobo re-plans when no local path exists (least travel wins)


class RiskTickController(TickChunkedController):
    """Vanilla tick controller whose plans are uniformly time-scaled."""

    controller_name = "risk_tick"

    def __init__(self, env, risky_skill_labels: tuple[str, ...] = (), **kw):
        super().__init__(env, **kw)
        self.risky_skill_labels = tuple(str(s) for s in risky_skill_labels)
        self._v_limit_cache: dict[str, np.ndarray] = {}
        self.retime_log: list[dict] = []  # one entry per retimed plan
        self.v_ratio_events: list[dict] = []  # subset with v_ratio > 1 (never clipped)
        self.n_plan_failures: int = 0
        # realized-vs-commanded tracking (see module docstring)
        self._user_tick_hook = None
        self._tracking_enabled = False
        self._pending_chunk = None  # (plan ids, qpos per arm, commanded norms)
        self._realized_pairs: list[tuple[float, float]] = []

    # ----------------------------------------------------------------- plan ----
    def plan(self, arm: str, target_pose, **kw):
        self._sync_time_scale_from_skill()
        result = super().plan(arm, target_pose, **kw)
        if not result or result.get("status") != "Success":
            self.n_plan_failures += 1
            return result
        result = self._unwrap_revolutions(arm, result)
        pos = np.asarray(result.get("position"), dtype=np.float64)
        vel_raw = result.get("velocity")
        vel = np.zeros_like(pos) if vel_raw is None else np.asarray(vel_raw, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2 or vel.shape != pos.shape:
            # degenerate plan (0/1 rows or missing velocity): leave untouched —
            # retime needs >= 2 rows; never crash a sweep on a short plan.
            return result
        task = getattr(self.env, "task", None)
        if result.get("_item_profiled"):
            # a via path already carries its own time law (speed set by the
            # task, risky time_scale included): never rescale it again
            self.retime_log.append({"arm": arm, "s": 1.0, "T_in": int(pos.shape[0]), "T_out": int(pos.shape[0]),
                                    "via": True})
            return result
        s = float(getattr(task, "active_time_scale", 1.0))
        # duration-targeted retime: a task that knows how long a step should
        # take (T2 tilt: |step angle| / tilt_rate) sets plan_duration_target;
        # s then follows from the plan's own duration, whichever planner made
        # it (mplib screw plans run ~10x slower than cuRobo's — the old fixed
        # tilt_rate / 2 rad/s factor turned a 1 rad/s pour into ~0.1 rad/s
        # and timed out the episode, diag job 22636290)
        target = getattr(task, "plan_duration_target", None)
        if target is not None and float(target) > 0.0:
            t_in = (pos.shape[0] - 1) * PHYSICS_DT
            s = max(1e-3, t_in / float(target))
        pos_out, vel_out, info = retime(pos, vel, s, v_limit=self._v_limit(arm))
        result["position"], result["velocity"] = pos_out, vel_out
        result["_item_retime"] = info
        self.retime_log.append({"arm": arm, **info})
        v_ratio = info.get("v_ratio")
        if v_ratio is not None and v_ratio > 1.0:
            event = {
                "arm": arm,
                "s": info["s"],
                "v_ratio": float(v_ratio),
                "T_in": info["T_in"],
                "T_out": info["T_out"],
            }
            self.v_ratio_events.append(event)
            self._tracking_enabled = True  # start realized-vs-commanded tracking
            print(
                f"[risk_tick] v_ratio={v_ratio:.3f} > 1 (arm={arm} s={info['s']} "
                f"T {info['T_in']}->{info['T_out']}) — NOT clipped; tracking "
                f"realized-vs-commanded displacement"
            )
        return result

    def _plan_body(self, arm: str, target_pose, **kw):
        """Local-planning mode for small reorientation steps (T2 tilt).

        cuRobo's goal-pose IK may pick ANOTHER IK branch for a 6-deg wrist
        step: diag job 22636048 (granular_pour ur5 seed 0) swung joint 0
        -2.02 -> +0.58 rad and joint 4 +1.86 -> -1.02 rad on one tilt step,
        sweeping the bowl through (-0.21, -0.55, 1.64) and flinging the beads.
        When the task sets ``plan_local`` (duck-typed, RoboTwin task files
        cannot import main.*), try the mplib SCREW path first — a straight
        Cartesian interpolation solved by IK from the live joints, so it stays
        on the current branch — and reject any fallback plan whose joints
        travel more than LOCAL_JUMP_MAX from the start (a failed step is
        recorded as a plan failure; flinging the payload is not).
        """
        task = getattr(self.env, "task", None)
        if not getattr(task, "plan_local", False):
            result = super()._plan_body(arm, target_pose, **kw)
            return self._avoid_branch_flip(arm, target_pose, result, **kw)
        self._n_plan += 1
        plan_kw = {k: v for k, v in kw.items() if not str(k).startswith("_cehj")}
        if getattr(task, "plan_via", None) is not None:  # [] = a one-segment via path
            return self._plan_via(arm, target_pose, task, plan_kw)
        try:
            result = self._plan_screw(arm, target_pose, **plan_kw)
        except Exception as exc:  # mplib unavailable / IK error: fall through
            print(f"[risk_tick] plan #{self._n_plan} {arm} local screw error: {exc}")
            result = None
        if result and result.get("status") == "Success" and _max_joint_travel(result) <= LOCAL_JUMP_MAX:
            # keep the screw's END joints (IK on the live branch) but replace
            # its path with a minimum-jerk joint interpolation: mplib's screw
            # timing has velocity spikes, and duration-targeted retimes of 7-11x
            # (T2 tilt at 1 rad/s) hit v_ratio 3 and shook the bowl out of the
            # grasp (diag job 22636542). For a 6-11 deg wrist step the joint-
            # space path tracks the screw path; peak speed = 1.875 x average.
            result = _min_jerk_resample(result)
            print(f"[risk_tick] plan #{self._n_plan} {arm} local screw ok "
                  f"n_wp={len(result['position'])} travel={_max_joint_travel(result):.2f}rad (min-jerk)")
            return result
        self._n_plan -= 1  # super()._plan_body counts its own attempt
        result = super()._plan_body(arm, target_pose, **kw)
        if result and result.get("status") == "Success":
            travel = _max_joint_travel(result)
            if travel > LOCAL_JUMP_MAX:
                print(f"[risk_tick] plan #{self._n_plan} {arm} REJECTED: local step "
                      f"joint travel {travel:.2f} rad > {LOCAL_JUMP_MAX} (IK branch flip)")
                return {"status": "Fail_branch_flip"}
        return result

    def _plan_via(self, arm: str, target_pose, task, plan_kw):
        """ONE continuous local trajectory through ``task.plan_via`` (the
        intermediate EE poses, then ``target_pose``): chained mplib screws
        from the live joints give the IK of every waypoint on the current
        branch; via_path joins them with a cubic spline over the Cartesian
        path fraction and times it with ``task.plan_profile`` (minjerk /
        hard_stop). Replaces a chain of rest-to-rest steps, which stopped
        the payload at every boundary (user review 7). Any segment that
        fails or travels > LOCAL_JUMP_MAX fails the whole move."""
        poses = [list(p) for p in task.plan_via] + [list(target_pose)]
        mplib = self.robot.left_mplib_planner if arm == "left" else self.robot.right_mplib_planner
        entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
        if mplib is None:
            return {"status": "Fail_via_no_mplib"}
        q = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
        idx = np.asarray(mplib.planner.move_group_joint_indices, dtype=np.int64)
        qs = [q[idx].copy()]
        for k, pose in enumerate(poses):
            try:
                res = self._plan_screw(arm, pose, last_qpos=q, **plan_kw)
            except Exception as exc:
                print(f"[risk_tick] plan #{self._n_plan} {arm} via {k}/{len(poses)} screw error: {exc}")
                return {"status": "Fail_via"}
            if res and res.get("status") == "Success" and _max_joint_travel(res) <= LOCAL_JUMP_MAX:
                end = np.asarray(res["position"], dtype=np.float64)
                if end.ndim == 2 and len(end):
                    q[idx] = end[-1]
            else:
                # the straight screw between two dense waypoints can fail where
                # the step chain (re-planned from the LIVE joints) passed (ur5
                # carry 2/6, d60): IK of the waypoint itself, nearest to the
                # previous one, keeps the trajectory continuous
                q_ik = self._ik_near(mplib, arm, pose, q, idx)
                if q_ik is None:
                    print(f"[risk_tick] plan #{self._n_plan} {arm} via {k}/{len(poses)} failed "
                          f"(screw {None if not res else res.get('status')}, no IK near the previous waypoint)")
                    return {"status": "Fail_via"}
                q[idx] = q_ik
            qs.append(q[idx].copy())
        W = np.stack(qs)
        ee0 = np.asarray(task.get_arm_pose(arm), dtype=np.float64).reshape(7)
        pts = np.stack([ee0[:3]] + [np.asarray(p[:3], dtype=np.float64) for p in poses])
        quats = [ee0[3:]] + [np.asarray(p[3:], dtype=np.float64) for p in poses]
        angs = [2.0 * np.arccos(np.clip(abs(float(np.dot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                                        0.0, 1.0)) for a, b in zip(quats[:-1], quats[1:])]
        profile = dict(getattr(task, "plan_profile", None) or {"kind": "minjerk", "duration": 1.0})
        pos, vel = profiled_via_path(W, path_fractions(pts, np.asarray(angs)), profile, dt=PHYSICS_DT)
        print(f"[risk_tick] plan #{self._n_plan} {arm} via path ok: {len(poses)} waypoints, "
              f"{profile.get('kind')} {len(pos) * PHYSICS_DT:.2f}s travel={float(np.max(np.abs(W - W[0]))):.2f}rad")
        return {"status": "Success", "position": pos, "velocity": vel, "_item_profiled": True}

    def _ik_near(self, mplib, arm: str, pose, q_full, idx, max_step: float = VIA_IK_STEP_MAX):
        """Move-group joints of an IK solution of ``pose`` closest to
        ``q_full`` (mplib IK, collision-checked), or None if none lies within
        ``max_step`` rad of it on every joint."""
        try:
            goal = self.robot._trans_from_gripper_to_endlink(list(pose), arm_tag=arm)
            status, sol = mplib.planner.IK(goal, np.asarray(q_full, dtype=np.float64), return_closest=True)
        except Exception as exc:
            print(f"[risk_tick] via IK error: {exc}")
            return None
        if status != "Success" or sol is None:
            # mplib's IK collision-checks every sample against its planning
            # world (the closed fingers on a held handle can register there);
            # a dense waypoint between two feasible ones only needs the
            # kinematics: one CLIK solve seeded at the previous waypoint,
            # joint limits enforced (T5 ur5 drag, d68)
            try:
                pl = mplib.planner
                link = pl.link_name_2_idx[pl.move_group]
                q_ik, ok, _ = pl.pinocchio_model.compute_IK_CLIK(link, goal, np.asarray(q_full, dtype=np.float64), [])
                if not (ok and pl.wrap_joint_limit(q_ik)):
                    return None
                sol = q_ik
            except Exception as exc:
                print(f"[risk_tick] via CLIK error: {exc}")
                return None
        sol = np.asarray(sol, dtype=np.float64).reshape(-1)
        q_new = sol[idx] if len(sol) == len(q_full) else sol[: len(idx)]
        if float(np.max(np.abs(q_new - np.asarray(q_full)[idx]))) > max_step:
            return None
        return q_new

    def _avoid_branch_flip(self, arm: str, target_pose, result, **kw):
        """cuRobo's goal IK sometimes returns ANOTHER IK branch: the ur5 arm
        swung its shoulder pan 187 deg and wrist 2 167 deg around for a
        modest EE move (T1 place_burger_fries seed 2, 6.4-7.4 s — a branch
        flip, not a 2*pi wrap). When any joint travels more than
        BRANCH_FLIP_TRAVEL, retry as an mplib screw from the live joints
        (stays on the current branch) and keep it if it succeeds with less
        travel; otherwise the original plan stands."""
        if not result or result.get("status") != "Success":
            return result
        travel = _max_joint_travel(result)
        if travel <= BRANCH_FLIP_TRAVEL:
            return result
        plan_kw = {k: v for k, v in kw.items() if not str(k).startswith("_cehj")}
        try:
            screw = self._plan_screw(arm, target_pose, **plan_kw)
        except Exception:
            screw = None
        if screw and screw.get("status") == "Success" and _max_joint_travel(screw) < travel:
            print(f"[risk_tick] plan #{self._n_plan} {arm} branch flip avoided: "
                  f"{travel:.2f} rad -> screw {_max_joint_travel(screw):.2f} rad")
            return screw
        # no local path: cuRobo's goal IK is seeded randomly — re-plan and keep
        # the least-travel success
        best, best_travel = result, travel
        for _ in range(BRANCH_FLIP_REPLANS):
            self._n_plan -= 1  # keep plan numbering per request
            again = super()._plan_body(arm, target_pose, **kw)
            if again and again.get("status") == "Success" and _max_joint_travel(again) < best_travel:
                best, best_travel = again, _max_joint_travel(again)
            if best_travel <= BRANCH_FLIP_TRAVEL:
                break
        print(f"[risk_tick] plan #{self._n_plan} {arm} large joint travel {travel:.2f} rad -> "
              f"{best_travel:.2f} rad after re-plans")
        return best

    def _unwrap_revolutions(self, arm: str, result) -> dict:
        """Shortest-way joint ends: a revolute joint at b and at b +/- 2*pi is
        the SAME configuration, but cuRobo's goal IK sometimes returns the far
        one — the ur5 wrist then spins ~350 deg instead of ~10 deg (T1
        place_burger_fries seed 2, 5-7 s). Any joint whose net travel exceeds
        pi gets its end wrapped to the nearest equivalent inside its limits;
        the column keeps the plan's own time profile (normalized progress
        s(t) = (q(t) - q0) / (q1 - q0)), every other joint is untouched."""
        pos = np.asarray(result.get("position"), dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2:
            return result
        try:
            entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
            limits = np.asarray(entity.get_qlimits(), dtype=np.float64)
        except Exception:  # no joint-limit source (mock robots): leave the plan alone
            return result
        q0, q1 = pos[0], pos[-1]
        new_pos = pos.copy()
        changed = []
        for j in range(min(pos.shape[1], limits.shape[0])):
            d = float(q1[j] - q0[j])
            if abs(d) <= np.pi:
                continue
            lo, hi = float(limits[j, 0]), float(limits[j, 1])
            target = float(q1[j]) - 2.0 * np.pi * round(d / (2.0 * np.pi))
            if not (lo - 1e-6 <= target <= hi + 1e-6) or abs(target - q0[j]) >= abs(d) - 1e-6:
                continue
            s_prog = (pos[:, j] - q0[j]) / d
            new_pos[:, j] = q0[j] + s_prog * (target - q0[j])
            changed.append((j, d, target - float(q0[j])))
        if not changed:
            return result
        out = dict(result)
        out["position"] = new_pos
        out["velocity"] = np.gradient(new_pos, PHYSICS_DT, axis=0)
        print(f"[risk_tick] {arm} joint unwrap " + ", ".join(
            f"j{j}: {np.degrees(a):+.0f}deg -> {np.degrees(b):+.0f}deg" for j, a, b in changed))
        return out

    def _sync_time_scale_from_skill(self) -> None:
        """T1 path: set task.active_time_scale from the current skill label.

        Inert when risky_skill_labels is empty (T2/T3 use the risky() CM).
        The skill tracker stamps task._cehj_skill with the label of the
        action being planned BEFORE the planner runs, so the label read here
        is the one this plan belongs to.
        """
        if not self.risky_skill_labels:
            return
        task = getattr(self.env, "task", None)
        if task is None:
            return
        label = current_skill(task)
        risky = any(tag in label for tag in self.risky_skill_labels)
        scale = getattr(task, "_time_scale_risky", 1.0) if risky else getattr(
            task, "_time_scale_safe", 1.0
        )
        task.active_time_scale = float(scale)

    # ------------------------------------------------------------- tracking ----
    def attach(self) -> None:
        super().attach()  # plan() routing + skill tracker + tick chunking
        self._user_tick_hook = self.tick_hook  # RolloutController's, if any
        self.tick_hook = self._tracking_tick_hook
        # planning dry-run capability for RoboTwin-side task code (which
        # cannot import main.*): granular_pour picks a tilt direction the arm
        # can actually execute (ur5-wsg wrist 1 is limited to +/-1.75 rad)
        task = getattr(self.env, "task", None)
        if task is not None:
            task.screw_plan_ok = self.screw_plan_ok
            task.screw_path_ok = self.screw_path_ok

    def screw_path_ok(self, arm: str, poses) -> bool:
        """Dry run of a whole pose SEQUENCE (nothing executes): chained mplib
        screws, each from the previous plan's end joints (``last_qpos``) —
        the way the task will actually execute it. A single start->final
        screw was a weak proxy: it passed ur5's far-lip pour, whose stepwise
        tilt then dead-ended after 1-2 steps (diag job 22639694)."""
        mplib = self.robot.left_mplib_planner if arm == "left" else self.robot.right_mplib_planner
        if mplib is None:
            return True  # no screw planner: nothing to dry-run against
        entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
        try:
            q = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
            idx = np.asarray(mplib.planner.move_group_joint_indices, dtype=np.int64)
        except Exception:
            return False
        for pose in poses:
            try:
                result = self._plan_screw(arm, list(pose), last_qpos=q)
            except Exception:
                return False
            if not result or result.get("status") != "Success":
                return False
            pos = np.asarray(result.get("position"), dtype=np.float64)
            if pos.ndim == 2 and pos.shape[0] > 0:
                q[idx] = pos[-1]
        return True

    def screw_plan_ok(self, arm: str, target_pose) -> bool:
        """Dry run (nothing executes, no plan counters move): is a screw path
        from the LIVE joints to ``target_pose`` feasible on the current IK
        branch (joint travel <= LOCAL_JUMP_MAX per local step is NOT checked
        here — a full tilt legitimately travels further)?"""
        try:
            result = self._plan_screw(arm, list(target_pose))
        except Exception:
            return False
        return bool(result) and result.get("status") == "Success"

    @property
    def mean_realized_ratio(self) -> float | None:
        """Summed achieved / summed commanded joint displacement (cmd > 0 chunks)."""
        num = sum(a for a, c in self._realized_pairs if c > 0)
        den = sum(c for a, c in self._realized_pairs if c > 0)
        if den <= 0:
            return None
        return float(num / den)

    def _tracking_tick_hook(self, ctx):
        try:
            self._observe_tick(ctx)
        except Exception:
            pass  # diagnostics never break control
        if self._user_tick_hook is not None:
            return self._user_tick_hook(ctx)
        return False

    def _observe_tick(self, ctx) -> None:
        qpos = self._arm_qpos()  # {"left": [n_l], "right": [n_r]} plan-order arm joints
        if self._pending_chunk is not None and self._tracking_enabled:
            plan_ids, start_qpos, cmd_norms = self._pending_chunk
            # a replan between boundaries discards the pending chunk: its
            # commanded rows never played (plan object identity changed)
            if plan_ids == (id(ctx.get("left_arm")), id(ctx.get("right_arm"))):
                achieved = 0.0
                for arm in ("left", "right"):
                    if arm in start_qpos:
                        n = start_qpos[arm].shape[0]
                        achieved += float(np.linalg.norm(qpos[arm][:n] - start_qpos[arm]))
                self._realized_pairs.append((achieved, sum(cmd_norms.values())))
        spt = self.steps_per_tick
        i = int(ctx.get("i", 0))
        start_qpos, cmd_norms = {}, {}
        for arm, key in (("left", "left_arm"), ("right", "right_arm")):
            plan = ctx.get(key)
            pos = None if plan is None else plan.get("position")
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            n_rows, n_cols = pos.shape[0], pos.shape[1]
            if n_rows < 2 or i >= n_rows or qpos[arm].shape[0] < n_cols:
                continue
            q_now = qpos[arm][:n_cols]
            target_row = pos[min(i + spt, n_rows - 1)]
            start_qpos[arm] = q_now.copy()
            cmd_norms[arm] = float(np.linalg.norm(target_row - q_now))
        self._pending_chunk = (
            (id(ctx.get("left_arm")), id(ctx.get("right_arm"))),
            start_qpos,
            cmd_norms,
        )

    def _arm_qpos(self) -> dict[str, np.ndarray]:
        """Live arm-joint qpos per arm, plan joint order (gripper excluded)."""
        robot = self.env.robot
        return {
            "left": np.asarray(robot.get_left_arm_real_jointState(), dtype=np.float64)[:-1],
            "right": np.asarray(robot.get_right_arm_real_jointState(), dtype=np.float64)[:-1],
        }

    # ------------------------------------------------------------- v_limit ----
    def _v_limit(self, arm: str) -> np.ndarray:
        """URDF <limit velocity> per plan column, in plan joint order.

        Parses the embodiment URDF directly — the same <limit velocity>
        pattern as main.network.body_features._parse_urdf_joints — WITHOUT
        instantiating BodyTokenExtractor (that needs torch, pytorch_kinematics
        and a live sim). Missing/implausible limits fall back to
        DEFAULT_JOINT_VEL and are logged, mirroring body_features policy.
        """
        if arm in self._v_limit_cache:
            return self._v_limit_cache[arm]
        if arm == "left":
            urdf_path = self.robot.left_urdf_path
            joint_names = list(self.robot.left_arm_joints_name)
        else:
            urdf_path = self.robot.right_urdf_path
            joint_names = list(self.robot.right_arm_joints_name)
        limits = _urdf_velocity_limits(urdf_path)
        vels, fallbacks = [], []
        for name in joint_names:
            v = limits.get(name)
            if v is None or not (0.1 < v < 10.0):
                v = DEFAULT_JOINT_VEL
                fallbacks.append(name)
            vels.append(float(v))
        if fallbacks:
            print(
                f"[risk_tick] URDF {Path(str(urdf_path)).name}: missing/implausible "
                f"<limit velocity> on {fallbacks}; fallback {DEFAULT_JOINT_VEL}"
            )
        out = np.asarray(vels, dtype=np.float64)
        self._v_limit_cache[arm] = out
        return out


def _urdf_velocity_limits(urdf_path) -> dict[str, float]:
    """joint name -> <limit velocity> (revolute rad/s, prismatic m/s)."""
    root = ET.parse(str(urdf_path)).getroot()
    out: dict[str, float] = {}
    for joint in root.iter("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("velocity") is None:
            continue
        try:
            out[joint.get("name")] = float(limit.get("velocity"))
        except ValueError:
            continue
    return out


def _max_joint_travel(result) -> float:
    """max over path rows and joints of |q(t) - q(0)| (rad) — 0 for short plans."""
    pos = np.asarray(result.get("position"), dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(pos - pos[0])))


def _min_jerk_resample(result) -> dict:
    """Same rows / endpoints, minimum-jerk profile q0 -> q1 (velocity rows =
    analytic derivative at PHYSICS_DT spacing). Leaves degenerate plans alone."""
    pos = np.asarray(result.get("position"), dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 3:
        return result
    n = pos.shape[0]
    tau = np.linspace(0.0, 1.0, n)[:, None]
    shape = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    dshape = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / ((n - 1) * PHYSICS_DT)
    q0, q1 = pos[0], pos[-1]
    out = dict(result)
    out["position"] = q0 + shape * (q1 - q0)
    out["velocity"] = dshape * (q1 - q0)
    return out
