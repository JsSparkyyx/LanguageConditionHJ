"""ItemEpisodeRunner (PLAN v3 §2.6): nominal item-safety episode + per-tick log.

Builds a CEHJ Env through the registry (robotwin-kind tasks via
``Env(entry.robotwin_task, ...)``; host-kind T1 lands with spawn.py in Phase 4
and raises NotImplementedError here), refreshes the cuRobo world with the
TABLE ONLY (the nominal never plans around items), attaches a
RiskTickController, and wraps ``scene.step`` with the same tick accumulator
as ``RolloutController.run()`` (1e-9 epsilon kept: 10 x float(0.1) < 1.0).

Scene interface: the RoboTwin task file registers plain metadata dicts
(``task.item_meta`` / ``task.region_meta``, written by
``robotwin_tasks/_item_utils.py`` — the ONLY RoboTwin->CEHJ channel). The
runner re-reads them into ItemSpec / Region objects, samples physics onto
them, enumerates the rule table from the task YAML, and evaluates every
rule's margin each tick.

Determinism (PLAN v3 §2): risk is sampled FIRST from
``np.random.RandomState(seed + 7919)``, then the Env is built (RoboTwin
seeds its own global RNG from the episode seed), then physics is sampled
from the SAME runner stream, in that fixed order.

KNOWN DEVIATION (documented, accepted for Phase 2): ``physics.apply_physics``
runs AFTER ``Env()`` construction, i.e. after RoboTwin's ``check_stable`` —
its docstring asks for pre-settle application, but RoboTwin-side task files
cannot import ``main.*``, so the earliest CEHJ-side hook is post-setup. The
sampled values are contact-affecting, so the first ticks may differ from
the check_stable configuration; items re-settle under the nominal's first
moves. Revisit if Phase 3 probes show layout drift.

Output: ``runs/item_envs/<task>/<embodiment>/seed_<k>/`` with meta.json,
trace.npz, retime.json, keyframe PNGs (render="full"), and an optional
overlay video. Plan failures, UnStableError, timeouts, and any episode
exception land in meta.json's "error" — a failed episode never crashes a
sweep.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from ._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.* lives in the sibling CEHJ copy

from main.envs.controller import CuroboIKController  # noqa: E402
from main.envs.env import Env  # noqa: E402
from main.envs.obstacle import update_curobo_world  # noqa: E402
from main.envs.record import observer_rgb  # noqa: E402

from . import geometry, margins as margin_fns, masks, physics, robot_distance  # noqa: E402
from .controller import RiskTickController  # noqa: E402
from .items import ItemSpec, entity_of  # noqa: E402
from .logger import (  # noqa: E402
    EpisodeMeta,
    EpisodeTrace,
    OverlayVideoWriter,
    StepRecord,
    draw_overlay,
    item_spec_dict,
    region_dict,
    rule_spec_dict,
    token_grid_meta,
)
from .regions import Region, table_region, zone_region  # noqa: E402
from .registry import get_task_entry  # noqa: E402
from .risk import RiskParams, sample_risk  # noqa: E402
from .rules import Consequence, enumerate_rules, resolve_splits  # noqa: E402

PHYSICS_FREQ = 250.0  # RoboTwin scene timestep is 1/250 s
# collect.py:123 parity: the ckpt dual-ur5 URDF mounts the arms 0.8 m apart
ARM_DISTANCE = {"ur5-wsg": 0.8}


class ItemTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""


class ItemEpisodeRunner:
    """One nominal item-safety episode with full per-tick logging."""

    def __init__(
        self,
        task: str,
        embodiment: str | None = None,
        seed: int = 0,
        host: str | None = None,
        risk: dict | None = None,
        active_rule_id: str | None = None,
        render: str = "full",
        video: bool = False,
        out_root: str | Path = "runs/item_envs",
        max_steps: int = 7500,
        arm_distance: float | None = None,
        task_cfg_path: str | None = None,
        video_fps: float = 25.0,
    ):
        self.entry = get_task_entry(task)
        if self.entry.kind == "host":
            raise NotImplementedError(
                f"{task!r} is host-kind (T1): spawn.py lands in Phase 4; "
                "the runner only supports robotwin-kind tasks in Phase 2"
            )
        if render not in ("full", "none"):
            raise ValueError(f"render must be 'full' or 'none', got {render!r}")
        self.task_name = task
        self.embodiment = embodiment or self.entry.default_embodiment
        self.seed = int(seed)
        self.host = host
        self.risk_override = risk
        self.active_rule_id = active_rule_id
        self.render = render
        self.video_enabled = bool(video) and render == "full"
        self.out_dir = Path(out_root) / task / self.embodiment / f"seed_{self.seed}"
        self.max_steps = int(max_steps)
        self.arm_distance = arm_distance
        self.video_fps = float(video_fps)
        self.task_cfg_path = Path(task_cfg_path) if task_cfg_path else self.entry.config

        # setup() products (sim tests call setup() directly)
        self.env = None
        self.risk_params: RiskParams | None = None
        self.task_cfg: dict = {}
        self.physics_sample: dict = {}
        self.items: list[ItemSpec] = []
        self.regions: list[Region] = []
        self.rules: list = []
        self.ctrl: RiskTickController | None = None
        self._stay_ref: dict[str, np.ndarray] = {}
        self._item_psids: dict[int, str] = {}
        self._side_psids: dict[int, str] | None = None
        self._held_latch: dict[str, str | None] = {"left": None, "right": None}
        self._held_warned = False
        self._held_error: str | None = None

    # ----------------------------------------------------------------- setup ----
    def setup(self) -> None:
        """Build env + items + regions + rules + controller. May raise (recorded by run())."""
        rng = np.random.RandomState(self.seed + 7919)
        self.task_cfg = _load_yaml(self.task_cfg_path)
        self.risk_params = sample_risk(self.task_cfg, rng)
        if self.risk_override:
            _merge_risk_override(self.risk_params, self.risk_override)

        CuroboIKController.install()
        env = Env(
            self.entry.robotwin_task,
            self.embodiment,
            self.seed,
            arm_distance=(
                float(self.arm_distance)
                if self.arm_distance is not None
                else ARM_DISTANCE.get(self.embodiment, 0.6)
            ),
            control_freq=25.0,
        )
        # Cached MotionGen may still hold cuboids from a previous episode;
        # the nominal's world model is the TABLE ONLY — it never plans around items.
        update_curobo_world(env.robot)
        self.env = env
        task = env.task

        self.items = [_item_from_meta(d) for d in getattr(task, "item_meta", [])]
        item_map = {s.item_id: s for s in self.items}
        task.items = item_map  # physics.apply_physics reads task.items
        self.regions = [table_region(task)]
        for d in getattr(task, "region_meta", []) or []:
            self.regions.append(zone_region(d["entity"], d.get("color"), region_id=d["region_id"]))
        region_map = {r.region_id: r for r in self.regions}

        # physics: same rng stream, fixed order after risk sampling
        self.physics_sample = physics.sample_physics(item_map, physics.load_physics_cfg(), rng)
        physics.apply_physics(task, self.physics_sample)

        splits_yaml = _load_yaml(self.task_cfg_path.parent.parent / "splits.yaml")
        self.rules = enumerate_rules(
            item_map,
            region_map,
            self.task_cfg,
            objects_cfg=physics.load_objects_cfg(),
            splits_cfg=resolve_splits(splits_yaml, self.task_name),
        )
        # stay rules reference the item pose at rule activation (t = 0)
        self._stay_ref = {
            r.rule_id: _item_position(item_map[r.item_ids[0]])
            for r in self.rules
            if r.consequence is Consequence.STAY
        }

        # held detection keys on per_scene_id (C++-assigned, wrapper-stable)
        self._item_psids = {int(entity_of(s).per_scene_id): s.item_id for s in self.items}
        self._side_psids = None
        self._held_latch = {"left": None, "right": None}

        self.ctrl = RiskTickController(
            env, risky_skill_labels=tuple(self.task_cfg.get("risky_skill_labels", ()) or ())
        )
        self.ctrl.attach()
        # time-scale convention (see robotwin_tasks/_item_utils.py): the runner
        # installs the scales; risky(task) / the T1 skill path switch between them
        task._time_scale_safe = float(self.risk_params.safe_time_scale)
        task._time_scale_risky = float(self.risk_params.time_scale)
        task.active_time_scale = float(self.risk_params.safe_time_scale)

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    # ------------------------------------------------------------------ run ----
    def run(self) -> tuple[EpisodeMeta, str | None]:
        """Run the episode; returns (meta, error). error is None on a clean run."""
        t_start = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace = EpisodeTrace(self.out_dir)
        meta = EpisodeMeta(
            task=self.task_name,
            host_task=self.host if self.entry.kind == "host" else None,
            embodiment=self.embodiment,
            seed=self.seed,
            risk=dict(self.risk_override or {}),
            physics={},
            items=[],
            regions=[],
            rules=[],
            token_grid=token_grid_meta(),
            active_rule_id=self.active_rule_id,
            task_success=False,
            violations={},
            min_margin={},
            first_violation_t={},
            plan_failures=0,
            spawn_rejections=0,
            wall_time_s=0.0,
        )
        try:
            if self.env is None:
                self.setup()
        except Exception as exc:  # UnStableError and friends: record, never crash
            error = f"{type(exc).__name__}: {exc}"
            print(
                f"[runner] setup failed ({self.task_name}/{self.embodiment}/"
                f"seed_{self.seed}): {error}"
            )
            traceback.print_exc()
            meta.wall_time_s = time.perf_counter() - t_start
            trace.write_meta(meta, error=error)
            self.close()
            return meta, error

        error = self._play(trace, meta)
        meta.wall_time_s = time.perf_counter() - t_start
        trace.finalize(meta)
        self._write_retime_sidecar()
        if error is not None:
            trace.write_meta(meta, error=error)
        self.close()
        return meta, error

    def _play(self, trace: EpisodeTrace, meta: EpisodeMeta) -> str | None:
        """Body of run() once the Env exists; returns an error string or None."""
        env, task, ctrl = self.env, self.env.task, self.ctrl
        video = OverlayVideoWriter(self.out_dir / "video.mp4", fps=self.video_fps) if self.video_enabled else None
        state = _TickState()

        # initial tick (t = 0), before play_once moves anything
        self._capture_tick(env, trace, video, state, n_physics=0)

        n_physics = 0
        tick_acc = 0.0
        orig_step = task.scene.step
        control_freq = float(env.control_freq)

        def hooked_step():
            nonlocal n_physics, tick_acc
            orig_step()
            n_physics += 1
            if n_physics > self.max_steps:
                raise ItemTimeout(f"rollout exceeded {self.max_steps} physics steps")
            tick_acc += control_freq / PHYSICS_FREQ
            # epsilon: 10 x float(0.1) sums to 0.9999999999999999 — without it
            # the capture drifts one physics step late (rollout.py keeps 1e-9)
            if tick_acc >= 1.0 - 1e-9:
                tick_acc -= 1.0
                self._capture_tick(env, trace, video, state, n_physics)

        task.scene.step = hooked_step
        error = None
        try:
            task.play_once()
            meta.task_success = bool(task.check_success())
        except ItemTimeout as exc:
            error = f"ItemTimeout: {exc}"
            print(f"[runner] {exc}")
        except Exception as exc:  # never crash a sweep mid-episode
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            task.scene.step = orig_step
            if self.render == "full":
                try:
                    state.keyframes["30_end"] = self._render_frame(task, state)
                except Exception:
                    pass  # end-of-episode diagnostics never mask the real error
            _write_keyframes(self.out_dir, state.keyframes)
            if video is not None:
                video.close()

        meta.risk = self.risk_params.as_dict()
        meta.physics = _jsonable_physics(self.physics_sample)
        meta.items = [item_spec_dict(s) for s in self.items]
        meta.regions = [region_dict(r) for r in self.regions]
        meta.rules = [rule_spec_dict(r) for r in self.rules]
        meta.plan_failures = int(ctrl.n_plan_failures)
        if getattr(task, "plan_success", True) is False and meta.plan_failures == 0:
            meta.plan_failures = 1  # plan_success flag tripped outside plan()
        if self._held_error is not None:
            meta.diag["held_error"] = self._held_error
        return error

    # -------------------------------------------------------------- capture ----
    def _capture_tick(self, env, trace, video, state: "_TickState", n_physics: int) -> None:
        task = env.task
        tick = len(trace)
        t_now = n_physics / PHYSICS_FREQ
        dists = robot_distance.robot_item_distances(env, self.items)
        forces = robot_distance.robot_item_forces(env, self.items)
        margins = self._rule_margins(dists)
        held = self._held_indices(env)

        if self.render == "full" or state.mask_cache is None:
            # collection cadence: masks every tick in render="full"; in
            # render="none" they are computed once (t = 0) and forward-filled
            try:
                visibility, token_masks = masks.item_masks(task, self.items, self.regions)
            except Exception as exc:
                if not state.mask_warned:
                    print(f"[runner] item_masks failed (zeros/NaN forward-filled): {exc}")
                    state.mask_warned = True
                n_cam = len(masks.CAMERA_NAMES)
                _, n_tok = masks.token_grid()
                n_rows = len(self.items) + sum(
                    1 for r in self.regions if r.nameable and r.entity is not None
                )
                visibility = np.full((len(self.items), n_cam), np.nan, dtype=np.float32)
                token_masks = np.zeros((n_rows, n_cam, n_tok), dtype=np.uint8)
            state.mask_cache = (visibility, token_masks)
        visibility, token_masks = state.mask_cache

        robot = env.robot
        qpos = np.asarray(
            robot.get_left_arm_real_jointState() + robot.get_right_arm_real_jointState(),
            dtype=np.float64,
        )
        ee_pose = np.stack(
            [
                np.asarray(robot.get_left_ee_pose(), dtype=np.float64).reshape(-1)[:7],
                np.asarray(robot.get_right_ee_pose(), dtype=np.float64).reshape(-1)[:7],
            ]
        )
        gripper = np.array(
            [robot.get_left_gripper_val(), robot.get_right_gripper_val()],
            dtype=np.float64,
        )
        rec = StepRecord(
            t=t_now,
            tick=tick,
            qpos=qpos,
            ee_pose=ee_pose,
            gripper=gripper,
            held=held,
            item_pose=(
                np.stack([_pose7(s) for s in self.items])
                if self.items
                else np.zeros((0, 7), dtype=np.float64)
            ),
            item_vel=(
                np.stack([_velocity6(s) for s in self.items])
                if self.items
                else np.zeros((0, 6), dtype=np.float64)
            ),
            particle_pos={
                s.item_id: _particle_positions(s) for s in self.items if s.kind == "particle_set"
            },
            region_T=(
                np.stack([r.world_T() for r in self.regions])
                if self.regions
                else np.zeros((0, 4, 4), dtype=np.float64)
            ),
            margins=margins,
            robot_item_dist=dists,
            robot_item_force=forces,
            visibility=np.asarray(visibility, dtype=np.float64).copy(),
            token_masks=np.asarray(token_masks, dtype=np.uint8).copy(),
            action_source=0,  # planner-only in Phases 2-4
            time_scale=float(getattr(task, "active_time_scale", 1.0)),
        )
        trace.add(rec)

        if self.render == "full":
            self._update_visuals(task, rec, dists, forces, video, state)

    # -------------------------------------------------------------- visuals ----
    def _update_visuals(self, task, rec, dists, forces, video, state: "_TickState") -> None:
        """Video frame every tick; keyframes on start/contact/violation events."""
        violated_now = bool(rec.margins.size and (rec.margins < 0.0).any())
        new_contacts = [
            self.items[k].item_id
            for k in range(len(self.items))
            if self.items[k].item_id not in state.contacted
            and (dists[k] <= 0.0 or forces[k] > 0.0)
        ]
        want_start = "00_start" not in state.keyframes
        first_violation = violated_now and not state.violated_once
        if video is None and not (want_start or new_contacts or first_violation):
            return

        frame = self._observer_frame(task, state)
        lines = _overlay_lines(self, rec.margins, rec.tick)
        if want_start:
            state.keyframes["00_start"] = draw_overlay(frame, lines)
        for item_id in new_contacts:
            state.contacted.add(item_id)
            state.keyframes[f"10_contact_{item_id}"] = draw_overlay(frame, lines)
        if first_violation:
            state.violated_once = True
            state.keyframes["20_violation"] = draw_overlay(frame, lines, flash=True)
            if video is not None:
                video.flash()
        if video is not None:
            video.add(frame, lines)

    def _observer_frame(self, task, state: "_TickState") -> np.ndarray:
        """Raw observer RGB frame (blank + warn-once when rendering fails)."""
        try:
            return observer_rgb(task)  # does its own _update_render
        except Exception as exc:
            if not state.render_warned:
                print(f"[runner] observer frame failed (blank): {exc}")
                state.render_warned = True
            return np.zeros((240, 320, 3), dtype=np.uint8)

    def _render_frame(self, task, state: "_TickState", rec=None) -> np.ndarray:
        """Observer frame with the overlay already drawn (keyframe-ready)."""
        frame = self._observer_frame(task, state)
        margins = rec.margins if rec is not None else self._rule_margins()
        tick = rec.tick if rec is not None else 0
        return draw_overlay(frame, _overlay_lines(self, margins, tick))

    # --------------------------------------------------------------- margins ----
    def _rule_margins(self, dists: np.ndarray | None = None) -> np.ndarray:
        """[R] margins in rule-table order, raw SI, positive = safe."""
        if dists is None:
            dists = robot_distance.robot_item_distances(self.env, self.items)
        item_map = {s.item_id: (k, s) for k, s in enumerate(self.items)}
        region_map = {r.region_id: r for r in self.regions}
        values = []
        for rule in self.rules:
            indexed = [item_map[iid] for iid in rule.item_ids]
            regs = [region_map[rid] for rid in rule.region_ids]
            values.append(self._one_margin(rule, indexed, regs, dists))
        return np.asarray(values, dtype=np.float64)

    def _one_margin(self, rule, indexed, regs, dists) -> float:
        consequence = rule.consequence
        params = rule.params
        if consequence is Consequence.AVOID:
            per_item = [
                margin_fns.m_avoid(dists[k], margin=params.get("margin", 0.0))
                for k, _spec in indexed
            ]
            return margin_fns.set_rule_margin(per_item)
        if consequence in (Consequence.INSIDE, Consequence.OUTSIDE):
            region = regs[0]
            world_T_region = region.world_T()
            per_item = []
            for _k, spec in indexed:
                pts_world = geometry.footprint(
                    _world_T(spec), spec.obb_center_local, spec.half_extents, spec.up_axis_local
                )
                pts_region = geometry.to_frame(pts_world, world_T_region)
                if consequence is Consequence.INSIDE:
                    per_item.append(
                        margin_fns.m_inside(
                            pts_region, region, require_on_surface=region.kind == "surface"
                        )
                    )
                else:
                    per_item.append(margin_fns.m_outside(pts_region, region))
            return margin_fns.set_rule_margin(per_item)
        if consequence is Consequence.UPRIGHT:
            per_item = [
                margin_fns.m_upright(
                    geometry.tilt_angle(_world_T(spec), spec.up_axis_local), params["theta_max"]
                )
                for _k, spec in indexed
            ]
            return margin_fns.set_rule_margin(per_item)
        if consequence is Consequence.STAY:
            per_item = []
            for _k, spec in indexed:
                p_ref = self._stay_ref.get(rule.rule_id)
                if p_ref is None:
                    p_ref = _item_position(spec)
                    self._stay_ref[rule.rule_id] = p_ref
                per_item.append(
                    margin_fns.m_stay(
                        _item_position(spec),
                        p_ref,
                        geometry.tilt_angle(_world_T(spec), spec.up_axis_local),
                        params["d_tol"],
                        params["theta_tol"],
                    )
                )
            return margin_fns.set_rule_margin(per_item)
        if consequence in (Consequence.PARTICLES_INSIDE, Consequence.PARTICLES_OUTSIDE):
            spec = indexed[0][1]
            pos = _particle_positions(spec)
            xy_per_region = [
                geometry.to_frame(pos, region.world_T())[:, :2] for region in regs
            ]
            if consequence is Consequence.PARTICLES_INSIDE:
                return margin_fns.m_particles_inside(xy_per_region, regs, T=params.get("T", 0.005))
            return margin_fns.m_particles_outside(xy_per_region, regs, T=params.get("T", 0.005))
        raise ValueError(f"unknown consequence: {consequence!r}")

    def _robot_side_psids(self, env) -> dict[int, str]:
        """per_scene_id -> left/right for every robot link entity (cached per env).

        per_scene_id is a C++-assigned int, stable across pybind wrappers.
        distance.py's ``_entity_side_lookup`` keys on ``id()`` of temporary
        link wrappers that are garbage-collected, so contact-time entities
        never match it in this process (held stayed -1 despite closed-gripper
        contact with box_0 for ~90 ticks) — hence the local lookup.
        """
        if self._side_psids is None:
            m: dict[int, str] = {}
            for side, art in (("left", env.robot.left_entity), ("right", env.robot.right_entity)):
                for link in art.get_links():
                    ent = getattr(link, "entity", None)
                    if ent is not None:
                        m[int(ent.per_scene_id)] = side
            self._side_psids = m
        return self._side_psids

    def _contact_held_sides(self, env, cehj_distance) -> dict[str, str]:
        """side -> item_id for items touched by a CLOSED gripper link this tick."""
        item_psids = self._item_psids
        side_psids = self._robot_side_psids(env)
        found: dict[str, str] = {}
        try:
            contacts = env.task.scene.get_contacts()
        except Exception:
            contacts = []  # transient (e.g. before the first step), same guard as distance.py
        for contact in contacts:
            bodies = contact.bodies
            for i in range(2):
                item_id = item_psids.get(int(bodies[1 - i].entity.per_scene_id))
                if item_id is None:
                    continue
                grip_ent = bodies[i].entity
                if not cehj_distance._is_gripper_entity(env, grip_ent):
                    continue
                side = side_psids.get(int(grip_ent.per_scene_id))
                if side is None or side in found or not cehj_distance._arm_is_holding(env, side):
                    continue
                found[side] = item_id
        return found

    def _held_indices(self, env) -> np.ndarray:
        """[2] item index held by left/right arm, -1 if none (contact latch).

        CEHJ latch semantics: a closed-gripper contact STARTS the latch; it
        holds while the arm stays closed and the item remains near the
        TCP/fingers (``_hold_metrics``), and releases otherwise. Failures
        warn ONCE and land in meta.diag["held_error"] — never silently -1.
        """
        held = np.array([-1, -1], dtype=np.int64)
        if not self._item_psids:
            return held
        from main.envs import distance as cehj_distance

        try:
            found = self._contact_held_sides(env, cehj_distance)
            index = {s.item_id: k for k, s in enumerate(self.items)}
            by_id = {s.item_id: s for s in self.items}
            for side, slot in (("left", 0), ("right", 1)):
                item_id = found.get(side)
                if item_id is None:
                    prev = self._held_latch.get(side)
                    if prev is not None and cehj_distance._arm_is_holding(env, side):
                        if cehj_distance._hold_metrics(env, by_id[prev].entity, side)["near"]:
                            item_id = prev
                self._held_latch[side] = item_id
                if item_id is not None:
                    held[slot] = index[item_id]
        except Exception as exc:
            if not self._held_warned:
                self._held_warned = True
                self._held_error = f"{type(exc).__name__}: {exc}"
                print(f"[runner] held detection failed (held stays -1): {self._held_error}")
                traceback.print_exc()
        return held

    # --------------------------------------------------------------- sidecar ----
    def _write_retime_sidecar(self) -> None:
        """retime.json: per-plan retime entries, v_ratio events, realized ratio.

        Kept OUT of meta.json (the §5 contract stays exact); v_ratio > 1 is
        never clipped, so this sidecar + the qpos trace are the record.
        """
        if self.ctrl is None:
            return
        payload = {
            "plans": self.ctrl.retime_log,
            "v_ratio_events": self.ctrl.v_ratio_events,
            "mean_realized_ratio": self.ctrl.mean_realized_ratio,
        }
        from .logger import _jsonable

        (self.out_dir / "retime.json").write_text(
            json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# per-tick state + helpers
# ---------------------------------------------------------------------------


class _TickState:
    """Mutable per-episode capture state."""

    def __init__(self):
        self.mask_cache = None
        self.mask_warned = False
        self.render_warned = False
        self.contacted: set[str] = set()
        self.violated_once = False
        self.keyframes: dict[str, np.ndarray] = {}


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_risk_override(params: RiskParams, override: dict) -> None:
    """CLI risk dict onto sampled RiskParams: time_scale/safe_time_scale are
    top-level knobs; every other key is a task-specific knob."""
    for key, value in override.items():
        if key == "time_scale":
            params.time_scale = float(value)
        elif key == "safe_time_scale":
            params.safe_time_scale = float(value)
        else:
            params.task[key] = value


def _item_from_meta(d: dict) -> ItemSpec:
    """ItemSpec from a robotwin_tasks/_item_utils.register_item dict."""
    return ItemSpec(
        item_id=d["item_id"],
        entity=d["entity"],
        category=d["category"],
        color=d.get("color"),
        kind=d.get("kind", "rigid"),
        particles=d.get("particles"),
        half_extents=np.asarray(d.get("half_extents") or [0.0, 0.0, 0.0], dtype=np.float64),
        obb_center_local=np.asarray(d.get("obb_center_local") or [0.0, 0.0, 0.0], dtype=np.float64),
        up_axis_local=np.asarray(d.get("up_axis_local") or [0.0, 0.0, 1.0], dtype=np.float64),
        mass=float(d.get("mass") or 0.0),
        role=d.get("role", "candidate"),
        group=d.get("group"),
    )


def _world_T(spec: ItemSpec) -> np.ndarray:
    return np.asarray(entity_of(spec).get_pose().to_transformation_matrix(), dtype=np.float64)


def _item_position(spec: ItemSpec) -> np.ndarray:
    return np.asarray(entity_of(spec).get_pose().p, dtype=np.float64).reshape(3).copy()


def _pose7(spec: ItemSpec) -> np.ndarray:
    pose = entity_of(spec).get_pose()
    return np.concatenate(
        [
            np.asarray(pose.p, dtype=np.float64).reshape(3),
            np.asarray(pose.q, dtype=np.float64).reshape(4),  # sapien quat is wxyz
        ]
    )


def _velocity6(spec: ItemSpec) -> np.ndarray:
    import sapien.physx as physx  # lazy: runner is sim-only, but keep module import light

    ent = entity_of(spec)
    comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
    if comp is None:
        return np.zeros(6, dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(comp.linear_velocity, dtype=np.float64).reshape(3),
            np.asarray(comp.angular_velocity, dtype=np.float64).reshape(3),
        ]
    )


def _particle_positions(spec: ItemSpec) -> np.ndarray:
    return np.stack(
        [np.asarray(entity_of(p).get_pose().p, dtype=np.float64).reshape(3) for p in (spec.particles or [])]
    ) if spec.particles else np.zeros((0, 3), dtype=np.float64)


def _overlay_lines(runner: ItemEpisodeRunner, margins: np.ndarray, tick: int) -> list[str]:
    """Active rule phrase, its margin, min margin over rules, time scale."""
    rules = list(runner.rules)
    lines = []
    active = next((r for r in rules if r.rule_id == runner.active_rule_id), None)
    if active is not None:
        j = rules.index(active)
        phrase = active.phrases[0] if active.phrases else active.rule_id
        lines.append(f"rule: {phrase}")
        lines.append(f"m(active)={_fmt_margin(active, margins[j])}")
    else:
        lines.append("rule: (none active)")
    if margins.size and rules:
        j_min = int(np.argmin(margins))
        lines.append(f"min={_fmt_margin(rules[j_min], margins[j_min])} [{rules[j_min].rule_id}]")
    s = float(getattr(runner.env.task, "active_time_scale", 1.0)) if runner.env is not None else 1.0
    lines.append(f"s={s:.2f} tick={tick}")
    return lines


def _fmt_margin(rule, value: float) -> str:
    """Human-facing margin: cm for length rules, deg for upright (raw SI in traces)."""
    consequence = getattr(rule, "consequence", None)
    name = getattr(consequence, "value", str(consequence))
    if name == "upright":
        return f"{float(np.rad2deg(value)):+.1f}deg"
    return f"{float(value) * 100.0:+.1f}cm"


def _write_keyframes(out_dir: Path, keyframes: dict) -> None:
    """keyframe PNGs (RGB), in insertion order; no-op when empty."""
    import imageio.v2 as imageio

    for name, frame in keyframes.items():
        if frame is None:
            continue
        imageio.imwrite(Path(out_dir) / f"{name}.png", np.asarray(frame, dtype=np.uint8))


def _jsonable_physics(sampled: dict) -> dict:
    from .logger import _jsonable

    return {str(k): _jsonable(v) for k, v in (sampled or {}).items()}
