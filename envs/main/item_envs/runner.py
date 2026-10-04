"""ItemEpisodeRunner (PLAN v3 §2.6): nominal item-safety episode + per-tick log.

Builds a CEHJ Env through the registry (robotwin-kind tasks via
``Env(entry.robotwin_task, ...)``; host-kind T1 via ``Env(host, ...)`` plus
``spawn.spawn_candidates`` — the host's own actors are registered as
task_object/support items from the task YAML's ``host_objects`` table, K
candidates are spawned on the host's corridors with
``np.random.RandomState(seed + 4241)``, and a layout that yields zero
candidates is recorded as infeasible and runs on with host items only,
never crashing), refreshes the cuRobo world with the
TABLE ONLY (the nominal never plans around items), attaches a
RiskTickController, and wraps ``scene.step`` with the same tick accumulator
as ``RolloutController.run()`` (1e-9 epsilon kept: 10 x float(0.1) < 1.0).

Scene interface: the RoboTwin task file registers plain metadata dicts
(``task.item_meta`` / ``task.region_meta``, written by
``robotwin_tasks/_item_utils.py`` — the ONLY RoboTwin->CEHJ channel; the T1
spawner appends to ``task.item_meta`` from the CEHJ side in the same
schema). The runner re-reads them into ItemSpec / Region objects, samples
physics onto them (STATIC items — no dynamic rigid component, e.g. T1's
plate/tray supports — are skipped: mass/CoM writes are meaningless there
and ``set_mass_props`` would raise), enumerates the rule table from the
task YAML, and evaluates every rule's margin each tick.

Determinism (PLAN v3 §2): risk is sampled FIRST from
``np.random.RandomState(seed + 7919)``, then the Env is built (RoboTwin
seeds its own global RNG from the episode seed), then physics is sampled
from the SAME runner stream, in that fixed order. T1's post-init spawns use
the separate ``seed + 4241`` stream (§2); the sampled
``risk.task.on_path_count`` is INJECTED into the spawn config so the count
meta.risk reports is the count that was placed.

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
from .regions import Region, container_region, table_region, zone_region  # noqa: E402
from .registry import get_task_entry  # noqa: E402
from .risk import RiskParams, sample_risk  # noqa: E402
from .rules import Consequence, enumerate_rules, resolve_splits  # noqa: E402
from .spawn import spawn_candidates  # noqa: E402

PHYSICS_FREQ = 250.0  # RoboTwin scene timestep is 1/250 s
# collect.py:123 parity: the ckpt dual-ur5 URDF mounts the arms 0.8 m apart
ARM_DISTANCE = {"ur5-wsg": 0.8}


class ItemTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""


def resolve_host(entry, host: str | None, seed: int) -> str | None:
    """T1 host resolution: an explicit ``host``, else round-robin over
    ``entry.host_tasks`` by episode seed (the task rotates through the 5
    CrossSafe hosts so collection and sweeps cover them uniformly)."""
    if entry.kind != "host":
        return None
    resolved = host or entry.host_tasks[int(seed) % len(entry.host_tasks)]
    if resolved not in entry.host_tasks:
        raise ValueError(
            f"{resolved!r} is not an allowed host for the task; "
            f"entry.host_tasks = {list(entry.host_tasks)}"
        )
    return resolved


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
        if render not in ("full", "none"):
            raise ValueError(f"render must be 'full' or 'none', got {render!r}")
        self.task_name = task
        self.embodiment = embodiment or self.entry.default_embodiment
        self.seed = int(seed)
        self.host = resolve_host(self.entry, host, self.seed)
        self.risk_override = risk
        self.active_rule_id = active_rule_id
        self.render = render
        self.video_enabled = bool(video) and render == "full"
        # absolute NOW: Env() chdirs to the RoboTwin root (env.py prepare()),
        # so a relative out_root would resolve under RoboTwin/ at write time
        self.out_dir = Path(out_root).resolve() / task / self.embodiment / f"seed_{self.seed}"
        self.max_steps = int(max_steps)
        self.arm_distance = arm_distance
        self.video_fps = float(video_fps)
        self.task_cfg_path = Path(task_cfg_path).resolve() if task_cfg_path else self.entry.config

        # setup() products (sim tests call setup() directly)
        self.env = None
        self.risk_params: RiskParams | None = None
        self.task_cfg: dict = {}
        self.physics_sample: dict = {}
        self.items: list[ItemSpec] = []
        self.regions: list[Region] = []
        self.rules: list = []
        self.ctrl: RiskTickController | None = None
        self._spawn_report: dict | None = None  # T1: spawn_candidates' info dict
        self._stay_ref: dict[str, np.ndarray] = {}
        self._item_psids: dict[int, str] = {}
        self._held_tracker = None  # robot_distance.HeldTracker, built on first use
        self._held_warned = False
        self._held_error: str | None = None

    # ----------------------------------------------------------------- setup ----
    def setup(self, build_controller: bool = True) -> None:
        """Build env + items + regions + rules + controller. May raise (recorded by run()).

        build_controller=False (Phase 5 collection): skip the
        RiskTickController — RolloutController attaches its own. The task
        time-scale attributes and risk_knobs are still installed (plan()
        reads them whoever owns the controller).
        """
        rng = np.random.RandomState(self.seed + 7919)
        self.task_cfg = _load_yaml(self.task_cfg_path)
        self.risk_params = sample_risk(self.task_cfg, rng)
        if self.risk_override:
            _merge_risk_override(self.risk_params, self.risk_override)

        CuroboIKController.install()
        task_name = self.host if self.entry.kind == "host" else self.entry.robotwin_task
        env = Env(
            task_name,
            self.embodiment,
            self.seed,
            arm_distance=(
                float(self.arm_distance)
                if self.arm_distance is not None
                else ARM_DISTANCE.get(self.embodiment, 0.6)
            ),
            control_freq=25.0,
        )
        self.env = env
        task = env.task

        self._spawn_report = None
        if self.entry.kind == "host":
            # T1 (PLAN v3 §7): the host's own actors register FIRST (self.items
            # = host items + spawned candidates), then K candidates spawn on
            # the host's corridors (spawn.py settles 250 physics steps and
            # rejects bad layouts internally). Zero candidates = layout-
            # infeasible: the report is recorded (meta.diag["spawn"]) and the
            # episode continues with host items only — never a crash.
            if not hasattr(task, "item_meta"):
                task.item_meta = []
            task.item_meta.extend(_host_item_meta(task, self.host, self.task_cfg))
            spawn_cfg = dict(self.task_cfg)
            objects = dict(self.task_cfg.get("objects") or {})
            if "on_path_count" in self.risk_params.task:
                # the runner's risk sampler owns this knob (seed + 7919 stream);
                # inject it so the placed count matches meta.risk
                objects["on_path_count"] = int(self.risk_params.task["on_path_count"])
            spawn_cfg["objects"] = objects
            specs, self._spawn_report = spawn_candidates(
                env, spawn_cfg, np.random.RandomState(self.seed + 4241)
            )
            if not specs:
                print(
                    f"[runner] spawn layout-infeasible on {self.host} "
                    f"(layout_retries={self._spawn_report.get('layout_retries')}, "
                    f"settle_rejections={self._spawn_report.get('settle_rejections')}, "
                    f"max_layouts_hit={self._spawn_report.get('max_layouts_hit')}) — "
                    "continuing with host items only"
                )
        # Cached MotionGen may still hold cuboids from a previous episode;
        # the nominal's world model is the TABLE ONLY — it never plans around
        # items. T1: refreshed AFTER spawning, so candidates never enter it.
        update_curobo_world(env.robot)

        self.items = [_item_from_meta(d) for d in getattr(task, "item_meta", [])]
        item_map = {s.item_id: s for s in self.items}
        task.items = item_map  # physics.apply_physics reads task.items
        self.regions = [table_region(task)]
        for d in getattr(task, "region_meta", []) or []:
            kind = d.get("kind", "zone")
            if kind == "zone":
                self.regions.append(zone_region(d["entity"], d.get("color"), region_id=d["region_id"],
                                                size_xy=d.get("size_xy")))
            elif kind == "container":
                self.regions.append(
                    container_region(
                        d["entity"],
                        d["rim_radius"],
                        d["region_id"],
                        up_axis_local=d.get("up_axis_local"),
                        rim_height=d.get("rim_height", 0.0),
                    )
                )
            else:
                raise ValueError(
                    f"unknown region_meta kind {kind!r} (region_id {d.get('region_id')!r})"
                )
        region_map = {r.region_id: r for r in self.regions}

        # physics: same rng stream, fixed order after risk sampling. STATIC
        # items (no dynamic rigid component — T1's plate/tray supports, T2's
        # static target) are skipped: mass/CoM writes are meaningless there
        # and set_mass_props would raise TypeError.
        physics_items = {iid: s for iid, s in item_map.items() if _is_dynamic_item(s)}
        self.physics_sample = physics.sample_physics(physics_items, physics.load_physics_cfg(), rng)
        physics.apply_physics(task, self.physics_sample)

        splits_yaml = _load_yaml(self.task_cfg_path.parent.parent / "splits.yaml")
        self.rules = enumerate_rules(
            item_map,
            region_map,
            self.task_cfg,
            objects_cfg=physics.load_objects_cfg(),
            splits_cfg=resolve_splits(splits_yaml, self.task_name),
        )
        if self.active_rule_id == "primary":  # CLI shorthand: the task's at-risk rule
            self.active_rule_id = self._primary_rule_id(task)
        # stay rules reference the item pose at rule activation (t = 0)
        self._stay_ref = {
            stay_ref_key(r.rule_id, iid): _item_position(item_map[iid])
            for r in self.rules
            if r.consequence is Consequence.STAY
            for iid in r.item_ids
        }

        # held detection keys on per_scene_id (C++-assigned, wrapper-stable)
        self._item_psids = {int(entity_of(s).per_scene_id): s.item_id for s in self.items}

        if build_controller:
            self.ctrl = RiskTickController(
                env, risky_skill_labels=tuple(self.task_cfg.get("risky_skill_labels", ()) or ())
            )
            self.ctrl.attach()
        # time-scale convention (see robotwin_tasks/_item_utils.py): the runner
        # installs the scales; risky(task) / the T1 skill path switch between them
        task._time_scale_safe = float(self.risk_params.safe_time_scale)
        task._time_scale_risky = float(self.risk_params.time_scale)
        task.active_time_scale = float(self.risk_params.safe_time_scale)
        # task-specific sampled knobs (T2/T3 read them inside play_once via
        # getattr(self, "risk_knobs", {}); T1's on_path_count is consumed by
        # spawn_candidates above)
        task.risk_knobs = dict(self.risk_params.task)
        # per-embodiment static config (PLAN §7: "embodiment differences live
        # in data"): the task yaml's embodiment_overrides[<embodiment>]
        overrides = (self.task_cfg.get("embodiment_overrides") or {}).get(self.embodiment) or {}
        task.embodiment_params = dict(overrides)

        # optional task pre-roll BEFORE t = 0 (never recorded): T4's held mode
        # grasps + levels the brim-full bowl and deletes the beads it spilled.
        # Item poses move, so the stay references are re-latched afterwards.
        prepare = getattr(task, "prepare_episode", None)
        if callable(prepare):
            prepare()
            # the pre-roll may delete particles (spilled beads): a particle
            # set's handle entity must be a live member, and the held-detection
            # per_scene_id map is rebuilt from the live entities
            for s in self.items:
                if s.kind == "particle_set" and s.particles:
                    s.entity = s.particles[0]
            self._item_psids = {int(entity_of(s).per_scene_id): s.item_id for s in self.items}
            self._stay_ref = {
                stay_ref_key(r.rule_id, iid): _item_position(item_map[iid])
                for r in self.rules
                if r.consequence is Consequence.STAY
                for iid in r.item_ids
            }

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
        if self._spawn_report is not None:  # T1: spawn_candidates' info dict
            meta.spawn_rejections = int(self._spawn_report.get("spawn_rejections", 0))
            meta.diag["spawn"] = self._spawn_report
            if int(self._spawn_report.get("k", 0)) == 0:
                meta.diag["spawn_layout_infeasible"] = True
        if self._held_error is not None:
            meta.diag["held_error"] = self._held_error
        meta.diag["primary_rule"] = self._primary_rule_id(task)
        layout = getattr(task, "layout_info", None)
        if isinstance(layout, dict) and "scenario" in layout:
            # T3 rev. 3: scenario / roles / referent for per-scenario calibration
            meta.diag["layout"] = {k: layout[k] for k in ("scenario", "mode", "referent", "elements") if k in layout}
        meta.active_rule_id = self.active_rule_id
        return error

    def _primary_rule_id(self, task) -> str | None:
        """The episode's at-risk rule for calibration (PLAN §8: adaptive cells
        and the 0.2-0.5 primary violation band).

        Task hint first (``task.primary_rule_id``: T3 names the midway zone's
        ``outside`` rule — the first table rule, ``inside|push_0|table_top``,
        cannot be violated now that goals sit inside the table); T1 takes the
        ``avoid`` rule of the first on-path (else near-path) candidate from
        the spawn report; otherwise the first rule (T2: the spill rule).
        """
        ids = [r.rule_id for r in self.rules]
        hint = getattr(task, "primary_rule_id", None)
        if hint in ids:
            return hint
        if hint is not None:
            print(f"[runner] WARNING: task primary_rule_id {hint!r} is not in the rule table — falling back")
        if self._spawn_report:
            placements = self._spawn_report.get("placements") or []
            for cls in ("on_path", "near_path"):
                for pl in placements:
                    rid = f"avoid|{pl.get('item_id')}|-"
                    if pl.get("cls") == cls and rid in ids:
                        return rid
        return ids[0] if ids else None

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
        plot = _h_plot(self, rec.margins)
        if want_start:
            state.keyframes["00_start"] = draw_overlay(frame, lines, plot=plot)
        for item_id in new_contacts:
            state.contacted.add(item_id)
            state.keyframes[f"10_contact_{item_id}"] = draw_overlay(frame, lines, plot=plot)
        if first_violation:
            state.violated_once = True
            state.keyframes["20_violation"] = draw_overlay(frame, lines, flash=True, plot=plot)
            if video is not None:
                video.flash()
        if video is not None:
            video.add(frame, lines, plot=plot)

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
        return rule_margins(
            self.env, self.items, self.regions, self.rules, self._stay_ref, dists
        )

    def _one_margin(self, rule, indexed, regs, dists) -> float:
        return one_margin(rule, indexed, regs, dists, self._stay_ref)

    def _held_indices(self, env) -> np.ndarray:
        """[2] item index held by left/right arm, -1 if none (contact latch).

        CEHJ latch semantics (robot_distance.HeldTracker): a closed-gripper
        contact STARTS the latch; it holds while the arm stays closed and the
        item remains near the TCP/fingers (``_hold_metrics``), and releases
        otherwise. Failures warn ONCE and land in meta.diag["held_error"] —
        never silently -1.
        """
        if not self._item_psids:
            return np.array([-1, -1], dtype=np.int64)
        try:
            if self._held_tracker is None:
                self._held_tracker = robot_distance.HeldTracker(self.items)
            return self._held_tracker.held_indices(env)
        except Exception as exc:
            if not self._held_warned:
                self._held_warned = True
                self._held_error = f"{type(exc).__name__}: {exc}"
                print(f"[runner] held detection failed (held stays -1): {self._held_error}")
                traceback.print_exc()
        return np.array([-1, -1], dtype=np.int64)

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


# ---------------------------------------------------------------------------
# margin evaluation (ItemEpisodeRunner._rule_margins' core, shared with
# margin_hook.ItemMarginHFn — Phase 5. stay_ref is mutated for STAY rules:
# the reference pose is latched at first evaluation, runner semantics)
# ---------------------------------------------------------------------------


def stay_ref_key(rule_id: str, item_id: str) -> str:
    """stay_ref key: one reference pose per (rule, member item)."""
    return f"{rule_id}#{item_id}"


def rule_margins(env, items, regions, rules, stay_ref, dists=None) -> np.ndarray:
    """[R] margins in rule-table order, raw SI, positive = safe."""
    if dists is None:
        dists = robot_distance.robot_item_distances(env, items)
    item_map = {s.item_id: (k, s) for k, s in enumerate(items)}
    region_map = {r.region_id: r for r in regions}
    values = []
    for rule in rules:
        indexed = [item_map[iid] for iid in rule.item_ids]
        regs = [region_map[rid] for rid in rule.region_ids]
        values.append(one_margin(rule, indexed, regs, dists, stay_ref))
    return np.asarray(values, dtype=np.float64)


def one_margin(rule, indexed, regs, dists, stay_ref) -> float:
    consequence = rule.consequence
    params = rule.params
    if consequence is Consequence.AVOID:
        per_item = [
            margin_fns.m_avoid(dists[k], margin=params.get("margin", 0.0))
            for k, _spec in indexed
        ]
        return margin_fns.set_rule_margin(per_item)
    if consequence in (Consequence.INSIDE, Consequence.OUTSIDE):
        # several regions (region_set: pairs): outside every one of them =
        # min over regions; inside any of them = max over regions
        per_region = []
        for region in regs:
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
            per_region.append(margin_fns.set_rule_margin(per_item))
        return float(max(per_region) if consequence is Consequence.INSIDE else min(per_region))
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
            key = stay_ref_key(rule.rule_id, spec.item_id)  # per member: set rules hold several refs
            p_ref = stay_ref.get(key)
            if p_ref is None:
                p_ref = _item_position(spec)
                stay_ref[key] = p_ref
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


def _host_item_meta(task, host: str, task_cfg: dict) -> list[dict]:
    """item_meta dicts for a T1 host's own actors (PLAN v3 §7 T1 Items).

    The task YAML's ``host_objects`` table maps the host to its actor
    attributes: pick objects (role "task_object") and targets (role
    "support"). List-valued attributes (place_bread_basket's bread) expand
    to "<attr><i>". The OBB mirrors ``spawn._spec_from`` for host actors:
    the collision-model AABB from the actor's own config via CEHJ's
    ``distance._obb_from_actor`` — the same convention distance.py uses for
    the CrossSafe obstacle, so held-payload distances through
    robot_distance stay bit-consistent with ``_actor_obb_distance`` (its
    docstring's parity argument). Mass is read back from the rigid body
    (0.0 for static supports, which physics sampling then skips); color is
    None (textured assets).
    """
    from main.envs import distance as cehj_distance

    table = (task_cfg.get("host_objects") or {}).get(host)
    if not table:
        raise KeyError(f"task yaml host_objects has no entry for host {host!r}")
    metas: list[dict] = []
    for entry in table:
        attr = str(entry["attr"])
        if not hasattr(task, attr):
            raise ValueError(
                f"host {host!r} has no actor attribute {attr!r} "
                "(fix the task yaml host_objects table)"
            )
        value = getattr(task, attr)
        actors = list(value) if isinstance(value, (list, tuple)) else [value]
        for i, actor in enumerate(actors):
            if actor is None:
                continue
            item_id = attr if not isinstance(value, (list, tuple)) else f"{attr}{i}"
            origin, rot, half = cehj_distance._obb_from_actor(actor)
            ent = entity_of(actor)
            pose = ent.get_pose()
            center_local = rot.T @ (origin - np.asarray(pose.p, dtype=np.float64).reshape(3))
            # RoboTwin glb assets are Y-up (regions.container_region's default);
            # world-aligned prims (create_box) override via the yaml entry
            up = np.asarray(entry.get("up_axis_local") or [0.0, 1.0, 0.0], dtype=np.float64)
            metas.append(
                {
                    "item_id": item_id,
                    "entity": actor,
                    "category": _host_category(task, entry),
                    "color": None,
                    "kind": "rigid",
                    "role": str(entry["role"]),
                    "group": None,
                    "particles": None,
                    "half_extents": [float(v) for v in np.asarray(half, dtype=np.float64).reshape(3)],
                    "obb_center_local": [float(v) for v in center_local.reshape(3)],
                    "up_axis_local": [float(v) for v in up.reshape(3)],
                    "mass": _dynamic_mass(ent),
                }
            )
    return metas


def _host_category(task, entry: dict) -> str:
    """Category for a host_objects entry; category_from resolves it at runtime."""
    src = entry.get("category_from")
    if src:
        raw = str(getattr(task, src["attr"], ""))
        mapped = (src.get("map") or {}).get(raw)
        if mapped is not None:
            return str(mapped)
        if src.get("default") is not None:
            return str(src["default"])
    return str(entry.get("category", "object"))


def _dynamic_mass(entity) -> float:
    """Mass of the entity's dynamic rigid component; 0.0 for static bodies."""
    import sapien.physx as physx  # lazy: runner imports stay light

    comp = entity_of(entity).find_component_by_type(physx.PhysxRigidDynamicComponent)
    return float(comp.mass) if comp is not None else 0.0


def _is_dynamic_item(spec: ItemSpec) -> bool:
    """True when the item (or any particle of a particle_set) has a dynamic body.

    Static items skip physics sampling: ``set_mass_props`` needs a
    PhysxRigidDynamicComponent and raises TypeError on static supports
    (T1's plate/tray, T2's static target container).
    """
    import sapien.physx as physx  # lazy: runner imports stay light

    entities = [spec.entity]
    if spec.kind == "particle_set":
        entities = list(spec.particles or [])
    for ent in entities:
        if entity_of(ent).find_component_by_type(physx.PhysxRigidDynamicComponent) is not None:
            return True
    return False


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
    """Active rule phrase + margin, min margin, time scale, then every rule margin
    and the task's optional ``overlay_lines()`` diagnostics."""
    rules = list(runner.rules)
    lines = []
    active = next((r for r in rules if r.rule_id == runner.active_rule_id), None)
    if active is not None:
        j = rules.index(active)
        phrase = active.phrases[0] if active.phrases else active.rule_id
        k = _kappa(active)
        lines.append(f"rule: {phrase}")
        lines.append(f"  = {active.rule_id}")
        lines.append(f"m(active)={_fmt_margin(active, margins[j])}")
        lines.append(f"h = k*m = {k * float(margins[j]):+.3f}  (k={k:g}/{'rad' if k < 10 else 'm'})")
    else:
        lines.append("rule: (none active)")
    if margins.size and rules:
        j_min = int(np.argmin(margins))
        lines.append(f"min={_fmt_margin(rules[j_min], margins[j_min])} [{rules[j_min].rule_id}]")
    s = float(getattr(runner.env.task, "active_time_scale", 1.0)) if runner.env is not None else 1.0
    lines.append(f"s={s:.2f} tick={tick} t={tick / 25.0:.2f}s")
    # every rule's margin (human units), violations prefixed "!" (drawn red),
    # so a video can be checked number-by-number against the trace
    for rule, value in zip(rules, margins):
        flag = "!" if float(value) < 0.0 else ""
        lines.append(f"{flag}{rule.rule_id}: {_fmt_margin(rule, value)}")
    # optional task-side diagnostics (RoboTwin task files cannot import
    # main.*, so this is a duck-typed hook returning plain strings)
    task = getattr(runner.env, "task", None) if runner.env is not None else None
    extra = getattr(task, "overlay_lines", None)
    if callable(extra):
        try:
            lines.extend(str(x) for x in extra())
        except Exception as exc:  # diagnostics must never kill an episode
            lines.append(f"overlay_lines error: {type(exc).__name__}")
    return lines


_KAPPA_CACHE: dict = {}


def _kappa(rule) -> float:
    """physics.yaml downstream kappa for a rule's consequence (20/m distance-
    type, 2/rad upright) — what the buffer h is: h = kappa * margin."""
    if not _KAPPA_CACHE:
        try:
            cfg = _load_yaml(Path(__file__).resolve().parent / "configs" / "physics.yaml")
            _KAPPA_CACHE.update({str(k): float(v) for k, v in (cfg.get("kappa") or {}).items()})
        except Exception:
            _KAPPA_CACHE.update({"upright": 2.0})
    name = getattr(getattr(rule, "consequence", None), "value", "")
    if name.startswith("particles"):
        name = "particles"
    return float(_KAPPA_CACHE.get(name, 20.0))


def _h_plot(runner, margins: np.ndarray) -> dict | None:
    """h(t) history for the video side panel: the active rule's h (orange)
    and the min over all rules of kappa_j * m_j (cyan); h < 0 = violation."""
    rules = list(runner.rules)
    if not rules or margins.size == 0:
        return None
    hist = runner.__dict__.setdefault("_h_hist", {"active": [], "all": []})
    hs = np.array([_kappa(r) * float(m) for r, m in zip(rules, margins)], dtype=np.float64)
    hist["all"].append(float(np.min(hs)))
    series = []
    j = next((i for i, r in enumerate(rules) if r.rule_id == runner.active_rule_id), None)
    if j is not None:
        hist["active"].append(float(hs[j]))
        series.append((list(hist["active"]), (255, 170, 40), "h active"))
    series.append((list(hist["all"]), (80, 220, 255), "h min(all)"))
    return {"series": series, "n": max(int(runner.max_steps // 10), len(hist["all"]))}


def _fmt_margin(rule, value: float) -> str:
    """Human-facing margin: cm for length rules, deg for upright (raw SI in traces)."""
    consequence = getattr(rule, "consequence", None)
    name = getattr(consequence, "value", str(consequence))
    if name == "upright":
        return f"{float(np.rad2deg(value)):+.1f}deg"
    if name == "stay":
        # m_stay = min(d_tol - displacement [m], theta_tol - tilt [rad]): the
        # binding term decides the unit, so show the raw value, not "cm"
        return f"{float(value):+.3f}(m|rad)"
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
