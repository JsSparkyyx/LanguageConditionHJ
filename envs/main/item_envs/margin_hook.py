"""Phase 5: item-margin h_fn for RolloutController + collection wiring.

``ItemMarginHFn(active_rule_id, kappa)`` is the h_fn item tasks inject into
RolloutController: ``h = kappa * margin_active`` (raw SI margins from the
runner's ``rule_margins`` machinery; positive = safe). rollout.py then
multiplies the return by ``cfg.h_scale``, so the collection wiring passes
``kappa = κ_consequence / h_scale`` (physics.yaml's downstream κ table:
20/m distance-type, 2/rad upright) and the buffer h lands at exactly
κ·margin — for distance rules the same h×20 units CEHJ trains on. With no
active rule, h is the min over ALL rules of κ_j·margin_j (collection
protects every rule; the language referent is chosen at training time).

``diag`` carries all rule margins + per-tick item state (the sidecar row
source) plus the compat keys ``main/train`` reads from compute_h's diag:

- ``per_link``      rollout._collect_tick -> buffer per_link_h (dict name->m)
- ``d_arm_arm``     rollout._collect_tick -> buffer field (inf here: N/A)
- ``d_left`` / ``d_right`` / ``d_left_held`` / ``d_right_held``,
  ``true_argmin``, ``holding``, ``holding_left``, ``holding_right``,
  ``hold_debug``, ``contact``
                    rollout._capture_tick video extras -> eval_utils panel

inf or empty where they do not apply. For a T1 single-item ``avoid`` rule
(the active one when set, else the min-margin single-item avoid rule)
``per_link`` is filled from the real per-link distances to that item —
hfunc._arm_spheres + point_obb_signed_distance, the same construction as
compute_h — so CrossSafe's diagnostics keep working.

``build_item_training_env`` is make_training_env's item-task path:
``ItemEpisodeRunner.setup(build_controller=False)`` (RolloutController
attaches its own RiskTickController), the runner stashed on
``env._item_runner`` for ItemMarginHFn.bind, and ``env.obstacle`` set as
collect.py's totality marker: an episode with zero rules (e.g. a
layout-infeasible T1 seed) gets ``None`` and is re-drawn — h = +inf would
NaN the Bellman target. The marker is never distance-queried (the h_fn
replaces compute_h; obstacle_contact keys on entity names, not this
attribute).

``wire_item_episode`` is collect.py's ONE per-episode call for item tasks:
bind the h_fn with the lockstep row sink, hand the RiskTickController the
task's risky_skill_labels, and write the episode's item/region/rule tables
into the sidecar's episodes.jsonl.
"""

from __future__ import annotations

import numpy as np

from ._bootstrap import ensure_main_path

ensure_main_path()  # main.envs.* / main.train.* live in the sibling CEHJ copy

from . import masks, robot_distance  # noqa: E402
from .items import entity_of  # noqa: E402
from .rules import Consequence  # noqa: E402

_PARTICLE_CONSEQUENCES = (Consequence.PARTICLES_INSIDE, Consequence.PARTICLES_OUTSIDE)


class _TickData:
    """One tick of raw item state (the sim seam — tests stub _compute_tick)."""

    __slots__ = (
        "margins", "dists", "held", "item_pose", "item_vel",
        "region_T", "visibility", "token_masks",
    )


class ItemMarginHFn:
    """(env, cfg) -> (kappa * margin_active, diag) for an item-task episode.

    active_rule_id: the rule h tracks (None = min over all rules of
    κ_j·margin_j). kappa: scalar, or mapping consequence name -> scale
    ("particles" covers both particles_* consequences); the collection
    wiring passes physics.yaml's κ table divided by cfg.h_scale (rollout.py
    applies h_scale to the return). ctx: the episode's ItemEpisodeRunner (or
    anything with .items/.regions/.rules/._stay_ref/.active_rule_id), bound
    explicitly or via bind(env) (reads env._item_runner). row_sink: optional
    object with offer(row_dict) — the lockstep sidecar writer; one row per
    call, unpadded (the sidecar pads to its k_max/r_max/g_max).
    """

    def __init__(self, active_rule_id: str | None = None, kappa=1.0,
                 ctx=None, row_sink=None, with_masks: bool = True):
        self.active_rule_id = active_rule_id
        self.kappa = kappa
        self.ctx = ctx
        self.row_sink = row_sink
        self.with_masks = bool(with_masks)
        self._held_tracker = None
        self._held_warned = False

    def bind(self, env):
        """Resolve the context from env._item_runner (build_item_training_env)."""
        ctx = getattr(env, "_item_runner", None)
        if ctx is None:
            raise ValueError(
                "ItemMarginHFn: env has no _item_runner — not built by "
                "build_item_training_env (make_training_env's item-task path)"
            )
        self.ctx = ctx
        return self

    # -------------------------------------------------------------- sim seam ----
    def _compute_tick(self, env) -> _TickData:
        """Raw margins + item state for the current scene state (sim-only)."""
        from .runner import _pose7, _velocity6, rule_margins

        ctx = self.ctx
        data = _TickData()
        data.dists = robot_distance.robot_item_distances(env, ctx.items)
        data.margins = rule_margins(
            env, ctx.items, ctx.regions, ctx.rules, ctx._stay_ref, data.dists
        )
        try:
            if self._held_tracker is None:
                self._held_tracker = robot_distance.HeldTracker(ctx.items)
            data.held = self._held_tracker.held_indices(env)
        except Exception as exc:  # runner parity: warn once, held stays -1
            if not self._held_warned:
                self._held_warned = True
                print(f"[margin_hook] held detection failed (held stays -1): {exc}")
            data.held = np.array([-1, -1], dtype=np.int64)
        items = list(ctx.items)
        data.item_pose = (
            np.stack([_pose7(s) for s in items])
            if items else np.zeros((0, 7), dtype=np.float64)
        )
        data.item_vel = (
            np.stack([_velocity6(s) for s in items])
            if items else np.zeros((0, 6), dtype=np.float64)
        )
        data.region_T = (
            np.stack([r.world_T() for r in ctx.regions])
            if ctx.regions else np.zeros((0, 4, 4), dtype=np.float64)
        )
        if self.with_masks:
            # runner's render="none" fallback: zeros/NaN on render failure —
            # collection recomputes EVERY tick (T1's candidates are dynamic,
            # a forward-filled mask would go stale)
            try:
                vis, tok = masks.item_masks(env.task, ctx.items, ctx.regions)
                data.visibility, data.token_masks = vis, tok
            except Exception as exc:
                print(f"[margin_hook] item_masks failed once (NaN/zeros): {exc}")
                n_cam = len(masks.CAMERA_NAMES)
                _, n_tok = masks.token_grid()
                n_rows = len(items) + sum(
                    1 for r in ctx.regions if r.nameable and r.entity is not None
                )
                data.visibility = np.full((len(items), n_cam), np.nan, dtype=np.float32)
                data.token_masks = np.zeros((n_rows, n_cam, n_tok), dtype=np.uint8)
        else:
            n_cam = len(masks.CAMERA_NAMES)
            _, n_tok = masks.token_grid()
            data.visibility = np.zeros((len(items), n_cam), dtype=np.float32)
            data.token_masks = np.zeros(
                (len(items) + sum(1 for r in ctx.regions if r.nameable and r.entity is not None),
                 n_cam, n_tok),
                dtype=np.uint8,
            )
        return data

    def _per_link(self, env, spec) -> dict:
        """Per-link signed distances to ONE item — compute_h's construction
        (hfunc._arm_spheres + point_obb_signed_distance, min per link)."""
        from main.train.hfunc import _arm_spheres

        from main.envs.distance import point_obb_signed_distance

        origin, rot, half = robot_distance._spec_obb(spec)
        per_link: dict[str, float] = {}
        for label, center, radius in _arm_spheres(env):
            d = float(point_obb_signed_distance(center, origin, rot, half) - radius)
            per_link[label] = min(per_link.get(label, float("inf")), d)
        return per_link

    # ----------------------------------------------------------------- pure ----
    def _kappa_for(self, rule) -> float:
        if isinstance(self.kappa, dict):
            name = rule.consequence.value
            if rule.consequence in _PARTICLE_CONSEQUENCES:
                name = "particles"
            return float(self.kappa.get(name, 1.0))
        return float(self.kappa)

    def _active_index(self, rules) -> int | None:
        rule_id = self.active_rule_id or getattr(self.ctx, "active_rule_id", None)
        if rule_id is None:
            return None
        for j, rule in enumerate(rules):
            if rule.rule_id == rule_id:
                return j
        raise KeyError(f"active_rule_id {rule_id!r} not in the episode's rule table")

    @staticmethod
    def _per_link_target(rules, margins, active_j):
        """The single-item avoid rule per_link describes: the active one when
        it is single-item avoid, else the min-raw-margin single-item avoid
        rule; None when no such rule exists."""
        singles = [
            j for j, r in enumerate(rules)
            if r.consequence is Consequence.AVOID and len(r.item_ids) == 1
        ]
        if not singles:
            return None
        if active_j in singles:
            return active_j
        return min(singles, key=lambda j: margins[j])

    def __call__(self, env, cfg) -> tuple[float, dict]:
        ctx = self.ctx if self.ctx is not None else self.bind(env).ctx
        data = self._compute_tick(env)
        rules = list(ctx.rules)
        margins = np.asarray(data.margins, dtype=np.float64).reshape(-1)
        scaled = np.array(
            [self._kappa_for(r) * m for r, m in zip(rules, margins)], dtype=np.float64
        )
        active_j = self._active_index(rules)
        if not rules:
            h = float("inf")
        elif active_j is not None:
            h = float(scaled[active_j])
        else:
            h = float(scaled.min())

        # per_link compat: real per-link distances for a T1 single-item avoid
        # target, empty (=> inf in the buffer) otherwise
        target_j = self._per_link_target(rules, margins, active_j)
        per_link: dict = {}
        true_argmin = ""
        if target_j is not None:
            spec = next(s for s in ctx.items if s.item_id == rules[target_j].item_ids[0])
            try:
                per_link = self._per_link(env, spec)
                true_argmin = min(per_link, key=per_link.get) if per_link else ""
            except Exception as exc:
                print(f"[margin_hook] per_link failed once (empty): {exc}")
                per_link = {}

        latched = (
            self._held_tracker.latched
            if self._held_tracker is not None else {"left": None, "right": None}
        )
        margin_active = (
            float(margins[active_j])
            if active_j is not None and rules
            else (float(margins.min()) if rules else float("inf"))
        )
        diag = {
            # item state + all margins (the sidecar row source)
            "margins": margins,
            "margin_active": margin_active,
            "active_rule_id": rules[active_j].rule_id if active_j is not None else None,
            "rule_ids": [r.rule_id for r in rules],
            "robot_item_dist": np.asarray(data.dists, dtype=np.float64),
            "held": np.asarray(data.held, dtype=np.int64),
            "item_pose": data.item_pose,
            "item_vel": data.item_vel,
            "region_T": data.region_T,
            "visibility": data.visibility,
            "token_masks": data.token_masks,
            # compute_h compat keys (main/train consumers — module docstring)
            "h": h,
            "d_block": margin_active,
            "d_held": float("inf"),
            "d_left": float("inf"),
            "d_right": float("inf"),
            "d_left_held": float("inf"),
            "d_right_held": float("inf"),
            "d_arm_arm": float("inf"),
            "per_link": per_link,
            "true_argmin": true_argmin,
            "holding": " ".join(sorted(x for x in latched.values() if x)),
            "holding_left": latched.get("left") or "",
            "holding_right": latched.get("right") or "",
            "hold_debug": {},
            "contact": bool(data.dists.size and (data.dists <= 0.0).any()),
            "n_below_table": 0,
        }
        if self.row_sink is not None:
            self.row_sink.offer(self._row(ctx, data, margins))
        return h, diag

    @staticmethod
    def _row(ctx, data: _TickData, margins) -> dict:
        """One sidecar row, UNPADDED (ItemSidecarBuffer pads to its maxima).
        item_obb packs the item-frame OBB: [obb_center_local, half_extents,
        up_axis_local]."""
        items = list(ctx.items)
        return {
            "item_pose": np.asarray(data.item_pose, dtype=np.float32),
            "item_vel": np.asarray(data.item_vel, dtype=np.float32),
            "item_obb": np.stack(
                [
                    np.concatenate(
                        [s.obb_center_local, s.half_extents, s.up_axis_local]
                    )
                    for s in items
                ]
            ).astype(np.float32)
            if items
            else np.zeros((0, 9), dtype=np.float32),
            "item_valid": np.ones(len(items), dtype=np.bool_),
            "held": np.asarray(data.held, dtype=np.int8),
            "robot_item_dist": np.asarray(data.dists, dtype=np.float32),
            "margins": np.asarray(margins, dtype=np.float32),
            "rule_valid": np.ones(len(margins), dtype=np.bool_),
            "region_T": np.asarray(data.region_T, dtype=np.float32),
            "token_masks": np.asarray(data.token_masks, dtype=np.uint8),
            "visibility": np.asarray(data.visibility, dtype=np.float16),
        }


# ---------------------------------------------------------------------------
# make_training_env's item-task path (Phase 5)
# ---------------------------------------------------------------------------


def build_item_training_env(cfg, seed: int, control_freq: float = 25.0):
    """Env + episode context for an item task, built with the runner's
    machinery (registry / host spawn / physics / rules) but NO controller —
    RolloutController attaches its own RiskTickController (controller_cls).

    Per-episode overrides ride on cfg as plain attributes (FrozenConfig is
    not actually frozen): item_host (T1 host, default the registry's first),
    item_risk (risk-knob override dict), item_active_rule.
    """
    if float(control_freq) != 25.0:
        raise ValueError(
            f"item tasks run at control_freq=25.0 (runner parity), got {control_freq}"
        )
    from .runner import ItemEpisodeRunner

    runner = ItemEpisodeRunner(
        cfg.task,
        embodiment=cfg.embodiment,
        seed=int(seed),
        host=getattr(cfg, "item_host", None),
        risk=getattr(cfg, "item_risk", None),
        active_rule_id=getattr(cfg, "item_active_rule", None),
        render="none",
        out_root=getattr(cfg, "item_out_root", "runs/item_envs"),
    )
    runner.setup(build_controller=False)
    env = runner.env
    env._item_runner = runner
    # collect.py's totality gate (env.obstacle is not None): an item episode
    # is total iff it has rules — a layout-infeasible T1 seed (zero
    # candidates) has none and is re-drawn. Never distance-queried (module
    # docstring).
    env.obstacle = entity_of(runner.items[0]) if runner.rules else None
    return env


def _kappa_over_h_scale(h_scale: float) -> dict:
    """physics.yaml's downstream κ table divided by h_scale — rollout.py
    multiplies the h_fn's return by cfg.h_scale, so the buffer h lands at
    exactly κ·margin (h×20 units for distance rules)."""
    from . import physics

    table = dict(physics.load_physics_cfg().get("kappa") or {})
    return {name: float(v) / float(h_scale) for name, v in table.items()}


def wire_item_episode(ro, sidecar) -> None:
    """collect.py's ONE per-episode call for item tasks, after the
    RolloutController is constructed (controller_cls=RiskTickController) and
    before ro.run():

    - ro.h_fn = ItemMarginHFn bound to the episode's runner, with the
      lockstep sidecar row sink (one row per capture; ro.buf is swapped for
      the LockstepItemWriter so every StepBuffer append pairs a sidecar row)
    - the RiskTickController gets the task's risky_skill_labels (T1's
      skill-type risk path; its constructor got the rollout-default args)
    - the episode's item/region/rule tables land in episodes.jsonl
    """
    from .item_buffer import LockstepItemWriter

    runner = getattr(ro.env, "_item_runner", None)
    if runner is None:
        raise ValueError("wire_item_episode: env was not built by build_item_training_env")
    lockstep = LockstepItemWriter(ro.buf, sidecar)
    h_scale = float(getattr(ro.cfg, "h_scale", 1.0))
    ro.h_fn = ItemMarginHFn(
        runner.active_rule_id,
        _kappa_over_h_scale(h_scale),
        ctx=runner,
        row_sink=lockstep,
    )
    ro.buf = lockstep
    labels = tuple(runner.task_cfg.get("risky_skill_labels", ()) or ())
    if labels and hasattr(ro.ctrl, "risky_skill_labels"):
        ro.ctrl.risky_skill_labels = labels
    sidecar.write_episode(ro.episode_id, runner)
