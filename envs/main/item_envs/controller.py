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

DEFAULT_JOINT_VEL = 2.0  # rad/s (or m/s prismatic) fallback, body_features.py policy
PHYSICS_DT = 1.0 / 250.0  # one plan row per physics step (cuRobo interpolation_dt, mplib time_step)
LOCAL_JUMP_MAX = 1.0  # rad: a plan_local step whose joints leave the start by more is a branch flip


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
        pos = np.asarray(result.get("position"), dtype=np.float64)
        vel_raw = result.get("velocity")
        vel = np.zeros_like(pos) if vel_raw is None else np.asarray(vel_raw, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2 or vel.shape != pos.shape:
            # degenerate plan (0/1 rows or missing velocity): leave untouched —
            # retime needs >= 2 rows; never crash a sweep on a short plan.
            return result
        task = getattr(self.env, "task", None)
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
            return super()._plan_body(arm, target_pose, **kw)
        self._n_plan += 1
        plan_kw = {k: v for k, v in kw.items() if not str(k).startswith("_cehj")}
        try:
            result = self._plan_screw(arm, target_pose, **plan_kw)
        except Exception as exc:  # mplib unavailable / IK error: fall through
            print(f"[risk_tick] plan #{self._n_plan} {arm} local screw error: {exc}")
            result = None
        if result and result.get("status") == "Success" and _max_joint_travel(result) <= LOCAL_JUMP_MAX:
            print(f"[risk_tick] plan #{self._n_plan} {arm} local screw ok "
                  f"n_wp={len(result['position'])} travel={_max_joint_travel(result):.2f}rad")
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
