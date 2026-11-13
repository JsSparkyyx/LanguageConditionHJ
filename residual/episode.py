"""One episode of the residual policy: scene, named rule, the 25 Hz loop, termination, settling, transitions.

How an episode ends (residual/PLAN.md section 1):

- **violation** (terminal, nothing after): the named rule's margin goes below 0 (training only; evaluation keeps
  going to measure task success and records the first break);
- **finished** (terminal plus a completion value): the host script returned, the arms hold still and everything
  settled (both hands below 1 cm/s, every item below 2 cm/s, the named object's tilt below 1 deg/s, for 50 ticks).
  The completion value is 1 / (1 - gamma) = 100, paid only on task success (``completion = "success"``) or always
  (``"always"``). The settling ticks belong to the last transition and pay nothing themselves;
- **truncated** (bootstrap V(s_T)): a failed replan, a failed plan in the host script, the time limit, or an error.

Reward per policy tick: r = 1 / (1 + w * |delta / delta_bound|^2 / 12), w ~ U(0, 2) per episode (a state input).
Ticks that run without the policy (an untagged plan played by RoboTwin) pay 1 each inside the open transition, with
the discount: a transition's ``rew`` sums gamma^k r_k and ``disc`` = gamma^(its ticks).

What differs per task (the scene, the named rule, the probe and the state) comes from ``tasks.task_hooks(cfg.task)``;
for ``tray_slide`` the named rule is one item's upright rule and the other item is only logged (PLAN section 8).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np

from . import ensure_paths
from .config import N_ACT, STEPS_PER_TICK, TICK_DT

ensure_paths()

from main.item_envs.runner import ItemTimeout, _velocity6  # noqa: E402

from .controller import ReplanFailed, ResidualTickController  # noqa: E402
from .state import robot_reading  # noqa: E402
from .tasks import task_hooks  # noqa: E402

AVOID_CONTACT = 0.01   # m: the avoid rule's buffer (an arm sphere or the held object within 1 cm breaks it)


class RuleBroken(ItemTimeout):
    """The named rule broke (training episodes end here)."""


class TimeLimit(ItemTimeout):
    """Physics-step budget used up (truncation)."""


@dataclass
class Job:
    kind: str = "train"            # train | eval | scan | throughput
    seed: int | None = None
    in_way: bool | None = None     # wanted side of the named object (None: the worker's balancer decides)
    rule_kind: str | None = None   # avoid | stay (None: drawn with cfg.stay_frac)
    item_id: str | None = None     # a fixed named object (evaluation scenes)
    w: float | None = None         # None: drawn from U(w_min, w_max)
    policy: str = "model"          # model | zero (the delta = 0 baseline)
    deterministic: bool = False
    noise: bool = True             # executed noise on a_nom (training only)
    terminate: bool = True         # end at the first break of the named rule
    controller: str = "residual"   # residual | playback (RoboTwin's own 250 Hz playback, no policy)
    trigger: str | None = None     # overrides cfg.replan_trigger
    video: str | None = None       # mp4 path (absolute)
    weights: str | None = None     # evaluation: a fixed weights snapshot
    tag: str = ""
    job_id: int = -1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class _Transitions:
    def __init__(self, gamma: float):
        self.g = float(gamma)
        self.obs, self.u, self.logp, self.value, self.rew, self.disc = [], [], [], [], [], []

    def open(self, obs, u, logp, value, r) -> None:
        self.obs.append(obs); self.u.append(u); self.logp.append(logp); self.value.append(value)
        self.rew.append(float(r)); self.disc.append(self.g)

    def extend(self) -> None:
        """One tick without the policy inside the open transition: reward 1 (no correction), discounted."""
        if not self.rew:
            return
        self.rew[-1] += self.disc[-1] * 1.0
        self.disc[-1] *= self.g

    def __len__(self) -> int:
        return len(self.rew)

    def arrays(self) -> dict:
        n = len(self)
        return {
            # an episode can end before its first policy tick (host plan failure, an error): reshape(0, -1) raises
            "obs": (np.asarray(self.obs, dtype=np.float32).reshape(n, -1) if n else np.zeros((0, 0), np.float32)),
            "u": np.asarray(self.u, dtype=np.float32).reshape(n, N_ACT),
            "logp": np.asarray(self.logp, dtype=np.float32),
            "value": np.asarray(self.value, dtype=np.float32),
            "rew": np.asarray(self.rew, dtype=np.float32),
            "disc": np.asarray(self.disc, dtype=np.float32),
        }


class Episode:
    def __init__(self, cfg, job: Job, runner, named, model, rng: np.random.Generator):
        self.cfg, self.job, self.runner, self.named, self.model, self.rng = cfg, job, runner, named, model, rng
        self.env = runner.env
        self.task = self.env.task
        self.w = float(job.w) if job.w is not None else float(rng.uniform(cfg.w_min, cfg.w_max))
        self.hooks = task_hooks(cfg.task)
        self.n_phys = 0
        self.probe = self.hooks.make_probe(runner, named, cfg, clock=lambda: self.n_phys)
        self.trans = _Transitions(cfg.gamma)
        self.m = None
        self.m_phys = -1
        self.tick = 0                 # every measured tick (policy, off-loop, settling)
        self.off_ticks = 0            # ticks run without the policy before settling
        self.settle_ticks = 0
        self.settling = False
        self.min_margin = float("inf")
        self.first_violation = None
        self.contact_part = None      # "arm" | "held": which part first came within 1 cm of the named object
        self.touch_part = None        # the same at 0 cm
        self.named_held_ever = False
        self.corr_sq = []             # |delta / bound|^2 / 12 per policy tick
        self.rews = []
        self.video = None
        self.margin_hist, self.corr_hist = [], []
        self.ctrl = None
        self._last_corr = 0.0

    # -------------------------------------------------------------- measuring ----
    def measure(self) -> dict:
        if self.m is None or self.m_phys != self.n_phys:
            self.m = self.probe.measure()
            self.m_phys = self.n_phys
            if self.check_consistency:
                self._consistency(self.m)
        return self.m

    check_consistency = False   # smoke test: compare with the runner's own distance and margin every tick
    consistency = None

    def _consistency(self, m: dict) -> None:
        """The official numbers for the named rule (robot_item_distances over every item, rule_margins)."""
        from main.item_envs import robot_distance as rd

        try:
            if self.named.kind in ("avoid", "stay"):
                dists = rd.robot_item_distances(self.env, self.runner.items)
            else:   # upright (tray_slide): no distance in the rule; compare the margin, the distances stay inf
                dists = np.full(len(self.runner.items), np.inf)
            margins = self.runner._rule_margins(dists)
            d_off = float(dists[self.named.item_index])
            m_off = float(margins[self.named.rule_index])
            # beyond item_shapes.NEAR (2 cm) both use hull bounds, which may differ a little between one call over
            # every sphere and one call per link; the rule's sign (1 cm) is decided below it, where both are exact
            c = self.consistency or {"n": 0, "n_near": 0, "max_dist_err": 0.0, "max_margin_err": 0.0,
                                     "max_margin_err_near": 0.0, "sign_flips": 0, "worst": None}
            c["n"] += 1
            upright = self.named.kind == "upright"
            if np.isfinite(d_off) or np.isfinite(m["dist"]) or upright:
                err_d = abs(d_off - m["dist"]) if np.isfinite(d_off) and np.isfinite(m["dist"]) else (
                    0.0 if upright else float("inf"))
                err_m = abs(m_off - m["margin"]) if np.isfinite(m_off) and np.isfinite(m["margin"]) else float("inf")
                if err_m > c["max_margin_err"]:
                    c["worst"] = {"tick": self.tick, "dist_official": d_off, "dist_mine": m["dist"],
                                  "margin_official": m_off, "margin_mine": m["margin"]}
                c["max_dist_err"] = max(c["max_dist_err"], err_d)
                c["max_margin_err"] = max(c["max_margin_err"], err_m)
                if min(d_off, m["dist"]) < 0.03 or self.named.kind != "avoid":
                    c["n_near"] += 1
                    c["max_margin_err_near"] = max(c["max_margin_err_near"], err_m)
                c["sign_flips"] += int((m_off < 0) != (m["margin"] < 0))
            self.consistency = c
        except Exception as exc:
            print(f"[residual] consistency check failed: {exc}")

    def _check(self, m: dict, initial: bool = False) -> None:
        if not initial:   # t = 0 is tick 0: tick k is k x 40 ms into the episode
            self.tick += 1
        margin = float(m["margin"])
        self.min_margin = min(self.min_margin, margin)
        self.named_held_ever |= bool(m["named_held"])
        if self.contact_part is None and min(m["d_arm"], m["d_held"]) < AVOID_CONTACT:
            self.contact_part = "held" if m["d_held"] < m["d_arm"] else "arm"
        if self.touch_part is None and min(m["d_arm"], m["d_held"]) <= 0.0:
            self.touch_part = "held" if m["d_held"] < m["d_arm"] else "arm"
        self._video_frame(m)
        if margin < 0.0 and self.first_violation is None:
            self.first_violation = self.tick
            if self.job.terminate:
                raise RuleBroken(f"{self.named.kind} {self.named.item_id} broken at tick {self.tick}")

    # ---------------------------------------------------------- agent hooks ----
    def policy_tick(self, view: dict) -> np.ndarray:
        cfg = self.cfg
        m = self.measure()
        if self.job.policy == "zero" or self.model is None:
            delta = np.zeros(N_ACT)
            u, logp, v = np.zeros(N_ACT), 0.0, 0.0
            obs = None
        else:
            obs = self.hooks.build_obs(cfg, self.probe, m, robot_reading(self.env), view, self.w)
            u, logp, v = self.model.act(obs, deterministic=self.job.deterministic)
            delta = float(cfg.delta_bound) * np.clip(u, -1.0, 1.0)
        c = float(np.sum((delta / float(cfg.delta_bound)) ** 2) / N_ACT)
        r = 1.0 / (1.0 + self.w * c)
        self.corr_sq.append(c)
        self.rews.append(r)
        self._last_corr = float(np.sqrt(c))
        if obs is not None and self.job.kind == "train":
            self.trans.open(obs, u, logp, v, r)
        return delta

    def tick_done(self) -> None:
        self._check(self.measure())

    def _off_loop_tick(self) -> None:
        m = self.measure()
        if self.settling:
            self.settle_ticks += 1
        else:
            self.off_ticks += 1
            self.trans.extend()
        self._check(m)

    # ---------------------------------------------------------------- video ----
    def _video_frame(self, m: dict) -> None:
        if self.video is None:
            return
        try:
            from main.envs.record import _draw_cube, _project, observer_rgb, to_record_size

            frame = observer_rgb(self.task)
            origin, rot, half = m["obb"]
            corners = np.array([origin + rot @ (half * np.array([1 - 2 * ((k >> 2) & 1), 1 - 2 * ((k >> 1) & 1),
                                                                 1 - 2 * (k & 1)])) for k in range(8)])
            uv = _project(self.task.cameras.observer_camera, corners)
            broken = self.first_violation is not None
            frame = _draw_cube(frame, uv, (255, 60, 60) if broken else (255, 200, 40))
            frame = to_record_size(frame)
            unit = self.hooks.margin_unit(self.named)
            self.margin_hist.append(float(np.clip(m["margin"] / unit, -1, 1)))
            self.corr_hist.append(self._last_corr)
            st = self.ctrl.stats if self.ctrl is not None and hasattr(self.ctrl, "stats") else {}
            lines = [
                f"{self.runner.host} seed {self.runner.seed}",
                *self.hooks.describe(self.named),
                f"{'delta = 0 (planner)' if self.job.policy == 'zero' else 'residual policy'}  w {self.w:.2f}",
                f"t {self.tick * TICK_DT:5.2f} s  {self.hooks.margin_text(m['margin'])}",
                f"|delta|/bound {self._last_corr:.2f}  replans {st.get('replans', 0)}"
                + (f"  resyncs {st['resyncs']}" if st.get("resyncs") else ""),
            ]
            if broken:
                lines.append(f"!rule broken at t {self.first_violation * TICK_DT:.2f} s")
            plot = {"series": [(list(self.margin_hist), (255, 170, 40), "h named"),
                               (list(self.corr_hist), (80, 220, 255), "|d|/b")],
                    "n": max(len(self.margin_hist), 400)}
            self.video.add(frame, lines, plot=plot)
            if broken and self.first_violation == self.tick:
                self.video.flash()
        except Exception as exc:  # video never breaks an episode
            print(f"[residual] video frame failed: {exc}")
            self.video = None

    # ----------------------------------------------------------------- play ----
    def _settle(self) -> bool:
        """Hold both arms and step until everything rests; returns True when the cap was hit."""
        cfg = self.cfg
        self.settling = True
        self._off_acc = 0   # settle ticks line up with the off-loop measurement
        robot = self.env.robot
        q = {"left": np.asarray(robot.get_left_arm_real_jointState()[:-1], dtype=np.float64),
             "right": np.asarray(robot.get_right_arm_real_jointState()[:-1], dtype=np.float64)}
        for a in ("left", "right"):
            robot.set_arm_joints(q[a], np.zeros_like(q[a]), a)
        dyn = [s for s in self.runner.items if s.kind != "particle_set"]
        prev_ee = None
        prev_tilt = None
        quiet = 0
        for _ in range(int(cfg.settle_cap_ticks)):
            for _k in range(STEPS_PER_TICK):
                self.task.scene.step()
            ee = np.concatenate([np.asarray(robot.get_left_ee_pose(), dtype=np.float64)[:3],
                                 np.asarray(robot.get_right_ee_pose(), dtype=np.float64)[:3]])
            m = self.measure()
            moving = False
            if prev_ee is not None:
                v_ee = max(np.linalg.norm(ee[:3] - prev_ee[:3]), np.linalg.norm(ee[3:] - prev_ee[3:])) / TICK_DT
                moving |= v_ee > cfg.settle_v_ee
                moving |= abs(np.degrees(m["tilt"] - prev_tilt)) / TICK_DT > cfg.settle_tilt_rate
            else:
                moving = True
            for s in dyn:
                try:
                    if float(np.linalg.norm(_velocity6(s)[:3])) > cfg.settle_v_item:
                        moving = True
                        break
                except Exception:
                    continue
            prev_ee, prev_tilt = ee, m["tilt"]
            quiet = 0 if moving else quiet + 1
            if quiet >= int(cfg.settle_ticks):
                return False
        return True

    def play(self) -> dict:
        cfg, job, task = self.cfg, self.job, self.task
        t0 = time.perf_counter()
        risky = tuple(self.runner.task_cfg.get("risky_skill_labels", ()) or ())
        if job.controller == "playback":
            from main.item_envs.controller import RiskTickController

            ctrl = RiskTickController(self.env, risky_skill_labels=risky)
        else:
            if job.trigger is not None and job.trigger != cfg.replan_trigger:
                from dataclasses import replace

                ctrl_cfg = replace(cfg, replan_trigger=job.trigger)
            else:
                ctrl_cfg = cfg
            ctrl = ResidualTickController(self.env, ctrl_cfg, agent=self,
                                          noise_rng=self.rng if job.noise else None, risky_skill_labels=risky)
        ctrl.attach()
        self.runner.ctrl = ctrl
        self.ctrl = ctrl
        if job.video:
            from main.item_envs.logger import OverlayVideoWriter

            self.video = OverlayVideoWriter(job.video, fps=25)

        task.scene.step()   # the runner's flush step: PhysX's contact list still holds removed bodies
        end, reason, success, err = None, None, False, None
        orig_step = task.scene.step
        self._off_acc = 0

        def hooked_step():
            orig_step()
            self.n_phys += 1
            if self.n_phys > cfg.max_physics_steps:
                raise TimeLimit(f"over {cfg.max_physics_steps} physics steps")
            if not getattr(ctrl, "in_loop", False):
                self._off_acc += 1
                if self._off_acc >= STEPS_PER_TICK:
                    self._off_acc = 0
                    self._off_loop_tick()

        task.scene.step = hooked_step
        settle_capped = False
        t_play = time.perf_counter()
        t_settle = None
        try:
            self._check(self.measure(), initial=True)   # t = 0
            task.play_once()
            if task.plan_success is False:
                end, reason = "truncated", "host_plan_fail"
            else:
                t_settle = time.perf_counter()
                settle_capped = self._settle()
                success = bool(task.check_success())
                end, reason = "finished", "settled"
        except RuleBroken:
            end, reason = "violation", "rule_broken"
        except ReplanFailed as exc:
            end, reason = "truncated", "replan_fail"
            print(f"[residual] {exc}")
        except TimeLimit:
            end, reason = "truncated", "timeout"
        except Exception as exc:  # never crash a worker on one bad episode
            end, reason = "truncated", "error"
            err = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, AssertionError) and "target_pose cannot be None" in str(exc):
                reason = "host_plan_fail"    # RoboTwin found no reachable grasp pose (grasp_actor), a host failure
            else:
                traceback.print_exc()
        finally:
            task.scene.step = orig_step
        t_end = time.perf_counter()
        if end != "finished":
            try:
                success = bool(task.check_success()) if end != "violation" or not job.terminate else False
            except Exception:
                success = False

        boot, last_value = 0.0, 0.0
        if end == "finished":
            boot = cfg.completion_value if (cfg.completion == "always" or success) else 0.0
        if job.kind == "train" and end == "truncated" and self.model is not None and len(self.trans):
            try:
                view = ctrl.view_now()
                obs = self.hooks.build_obs(cfg, self.probe, self.measure(), robot_reading(self.env), view, self.w)
                _u, _lp, last_value = self.model.act(obs, deterministic=True)
            except Exception as exc:
                print(f"[residual] final value failed ({exc}): bootstrap 0")
                last_value = 0.0
        video_path = None
        if self.video is not None:
            try:
                video_path = str(self.video.close())
            except Exception as exc:
                print(f"[residual] video close failed: {exc}")

        st = dict(getattr(ctrl, "stats", {}) or {})
        corr = np.asarray(self.corr_sq) if self.corr_sq else np.zeros(1)
        n_pol = int(st.get("policy_ticks", 0))
        out = {
            "kind": job.kind, "tag": job.tag, "job_id": job.job_id, "seed": int(self.runner.seed),
            "host": self.runner.host, "item_id": self.named.item_id, "category": self.named.category,
            "rule_kind": self.named.kind, "in_way": bool(self.named.in_way), "reason": self.named.reason,
            "w": self.w, "time_scale": float(self.runner.risk_params.time_scale),
            "policy": job.policy, "controller": job.controller, "trigger": job.trigger or cfg.replan_trigger,
            "end": end, "end_reason": reason, "error": err, "success": bool(success),
            "violated": self.first_violation is not None,
            "first_violation_s": None if self.first_violation is None else self.first_violation * TICK_DT,
            "min_margin": float(self.min_margin), "contact_part": self.contact_part, "touch_part": self.touch_part,
            "named_held": bool(self.named_held_ever),
            "policy_ticks": n_pol, "off_ticks": int(self.off_ticks), "settle_ticks": int(self.settle_ticks),
            "settle_capped": bool(settle_capped), "sim_ticks": int(self.tick), "phys_steps": int(self.n_phys),
            "replans": int(st.get("replans", 0)), "replan_failures": int(st.get("replan_failures", 0)),
            "replan_time": float(st.get("replan_time", 0.0)), "max_dev": float(st.get("max_dev", 0.0)),
            "clip_frac": float(st.get("clip_ticks", 0)) / max(1, n_pol),
            "gripper_ticks": int(st.get("gripper_ticks", 0)), "untagged_steps": int(st.get("untagged_steps", 0)),
            "corr_rms": float(np.sqrt(corr.mean())), "corr_max": float(np.sqrt(corr.max())),
            "rew_mean": float(np.mean(self.rews)) if self.rews else 1.0,
            "grip_events": list(getattr(ctrl, "grip_events", []) or []),
            "replan_log": list(getattr(ctrl, "replan_log", []) or [])[:50],
            "plan_failures": int(getattr(ctrl, "n_plan_failures", 0)),
            "wall_play": (t_settle or t_end) - t_play, "wall_settle": (t_end - t_settle) if t_settle else 0.0,
            "wall_episode": t_end - t0, "video": video_path,
            "boot": float(boot), "last_value": float(last_value),
        }
        if self.consistency is not None:
            out["consistency"] = self.consistency
        try:
            out.update(self.hooks.episode_extra(self))   # T1: nothing; tray: the other item, the drag, resyncs
        except Exception as exc:
            print(f"[residual] episode extras failed: {exc}")
        if job.kind == "train":
            out["traj"] = self.trans.arrays()
        return out


def run_episode(cfg, job: Job, model, rng: np.random.Generator, out_root: str, choose, next_seed=None,
                max_builds: int = 12) -> dict:
    """Build scenes until ``choose(runner, rng)`` names a rule (or the job's seed is fixed), then play.

    ``choose`` returns a ``scene.Named`` or None (reject the scene). Returns the episode dict with ``builds``
    (Env builds used), ``wall_build`` and ``rejected`` (rejected seeds); ``skipped`` when no scene fit.
    """
    builds, rejected, wall_build = 0, [], 0.0
    runner, named = None, None
    hooks = task_hooks(cfg.task)
    hooks.prepare_job(cfg, job, rng)   # tray_slide: the episode's drag behaviour (T1: nothing)
    while builds < max_builds:
        seed = int(job.seed) if job.seed is not None else int(next_seed())
        t0 = time.perf_counter()
        try:
            runner = hooks.build_runner(cfg, seed, out_root, job)
        except Exception as exc:
            builds += 1
            wall_build += time.perf_counter() - t0
            print(f"[residual] scene build failed (seed {seed}): {type(exc).__name__}: {exc}")
            rejected.append({"seed": seed, "why": "build_error"})
            runner = None
            if job.seed is not None:
                break
            continue
        builds += 1
        wall_build += time.perf_counter() - t0
        try:
            named = choose(runner, rng)
        except Exception as exc:
            print(f"[residual] choose failed (seed {seed}): {exc}")
            named = None
        if named is not None:
            break
        rejected.append({"seed": seed, "why": "no_fit"})
        runner.close()
        runner = None
        if job.seed is not None:
            break
    if runner is None or named is None:
        return {"kind": job.kind, "tag": job.tag, "job_id": job.job_id, "skipped": True, "builds": builds,
                "rejected": rejected, "wall_build": wall_build}
    try:
        ep = Episode(cfg, job, runner, named, model, rng)
        out = ep.play()
    finally:
        runner.close()
    out.update({"builds": builds, "rejected": rejected, "wall_build": wall_build, "skipped": False})
    return out
