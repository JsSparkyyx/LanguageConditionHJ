# Phase 2 report — simulation infrastructure (plan v3)

Date: 2026-10-01. Scope: PLAN v3 §6 Phase 2 (items 2.1–2.7). Supersedes the v2
report of the same filename (that document described deleted v2 modules:
`primitives.py`, `visibility.py`, `base_task.py`, `tasks/`).

Acceptance (PLAN §6 Phase 2): the test-only `smoke_items` task (one zone mat,
two boxes) runs end to end on both embodiments with visibly correct margins —
**met**: full episode (grasp → risky move → place → retreat) on `ur5-wsg` and
`franka-panda`, `task_success: True`, all margins > 0 at t = 0, held latch
contiguous across grasp→carry→place, keyframes + optional overlay video
written to `runs/item_envs/smoke_items/<embodiment>/seed_<k>/`.

## Test evidence

| suite | where | result | log |
|---|---|---|---|
| `pytest tests/item_envs -m "not sim and not gpu" -q` | login node | **143 passed, 34 deselected** | — |
| `pytest tests/item_envs -m sim -q` | GPU (A40, srun+apptainer) | **34 passed** in 460 s | `tmp/logs/robot_dist_held_194427.log` |
| `test_physics_sim.py` (headless CPU-PhysX) | login node | 11 passed | — |

Sim highlights (both embodiments):

- `robot_item_distances(env, [obstacle])` == `distance_info(env)["d_system"]`
  to ≤ 5.6e-10 m (requirement 1e-6), pre-grasp and during carry.
- Held rule (v3 §2.2): cup reads **+inf** during carry and joins the payload
  set — verified end-to-end with a scripted grasp (log
  `robot_dist_held_evidence_195303.log`); obstacle distance then equals the
  held-cup-to-obstacle OBB distance, matching `d_system` exactly.
- `masks.item_masks`: obstacle visibility > 0 in ≥ 1 camera at t = 0, token
  mask nonzero in the right camera, shapes `(K,3)` / `(K+G,3,400)`; raw
  `Segmentation[..., 1]` == `per_scene_id` runtime-confirmed (closes recon
  Q8's pending GPU item).
- `smoke_items`: 5 seeds × 2 embodiments build without `UnStableError`;
  margins positive at t = 0; low-risk episode completes with
  `time_scale == 0.6` throughout; fast-risk (1.3) episode shows the retime
  window and `retime.json` sidecar.

## Files

**CEHJ-side** (`main/item_envs/`): `physics.py` (v3 §2.1; CoM offset x/y
only), `robot_distance.py` (§2.2; psid-keyed held scan), `regions.py`
(contract dataclass + `table_region` / `zone_region` / `container_region`,
§2.3), `masks.py` (§2.4 visibility + token-grid uint8 masks; pure-numpy
`pool_mask_to_tokens` core), `controller.py` (`RiskTickController` §2.5 +
skill-label time-scale path for T1), `runner.py` (`ItemEpisodeRunner` §2.6),
`logger.py` (§5 `StepRecord`/`EpisodeMeta` verbatim), `registry.py`
(`TaskEntry`, only `smoke_items` registered), `probe.py` (§2.7 CLI;
`physics` subcommand is a Phase-3 stub), `_bootstrap.py` (`main` package
path merge).

**RoboTwin-side** (`robotwin_tasks/`, symlinked into `RoboTwin/envs/` by
`scripts/item_envs/install_robotwin_tasks.sh`, idempotent, verified no
tracked RoboTwin file is modified): `_item_utils.py` (zone-mat builder,
`risky(task)` context manager, `register_item`/`register_zone` metadata —
the ONLY task→CEHJ channel; no `main.*` imports, AST-tested),
`smoke_items.py` (test-only task).

**Configs/tests**: `configs/tasks/smoke_items.yaml` (v3 style; yields
`inside/outside/avoid` rules over box_0/box_1/table_top/zone_green); tests
`test_regions_masks.py`, `test_robot_distance_sim.py`, `test_masks_sim.py`,
`test_smoke_sim.py`, `test_item_utils_meta.py`, `test_registry.py`,
`test_smoke_rules.py`, `test_logger_contract.py`, `test_controller.py`,
`test_physics_sim.py`, `sim_utils.py`.

## Bugs found by GPU acceptance (all fixed; lessons recorded)

1. **pybind11 attribute write** — `zone_mat` set `entity.size_xy` on a raw
   `sapien.Entity` (pybind11 rejects arbitrary attributes) → killed 16/32
   tests. Fix: metadata flows through `register_zone`'s dict;
   `zone_region`'s collision-AABB fallback derives the footprint.
2. **`masks.py` NameError** — region branch of a comprehension referenced an
   undefined variable; only exercised with entity-backed regions, which no
   local test did. Fix + mock-camera test that executes the branch.
3. **Env-export trap** — `export TMP=... XDG_CACHE_HOME=$TMP/...` on one
   line expands `$TMP` before assignment → `/xdg-cache` → cuRobo "init
   failed, mplib fallback". Fix: `env VAR=literal` one-liner (now the
   documented pattern in `.kimi-code/skills/delta-hpc/SKILL.md`).
4. **Render body on a render-less scene** — on GPU nodes, `RenderMaterial`
   construction succeeds, so a try/except "best-effort" render body got
   attached and `add_entity` then threw `no system with name [render]`;
   worse, `Scene.__del__` re-threw during GC → SIGABRT killed pytest. Fix:
   gate on `scene.get_system("render")` (same lookup `add_entity` performs).
5. **Held detection silently off** — CEHJ's `_entity_side_lookup` keys on
   `id()` of pybind11 link wrappers that get garbage-collected; contact-time
   entities are fresh wrappers → side lookup never matches. (Masked in CEHJ
   tasks by the near/latch pass; **CEHJ `distance.py` itself has this latent
   bug** — flagged, not fixed, it is read-only for us.) Fix in
   `runner.py::_contact_held_sides` and `robot_distance.py::_held_info`:
   `per_scene_id`-keyed scans (stable C++ ints), CEHJ's
   `_is_gripper_entity`/`_arm_is_holding` gates reused.

## Contract notes / documented choices (downstream-relevant)

- `EpisodeMeta` gained a **trailing** `diag: dict = {}` field (additive
  extension of §5; holds e.g. `held_error` instead of silent swallowing).
  `retime.json` (per-plan retime entries, v_ratio events,
  mean_realized_ratio) is a **sidecar**, deliberately kept out of
  `meta.json` to keep the §5 contract exact.
- `token_masks` shape is `(K + G_nameable, n_cam, 400)` per camera;
  `.reshape(K+G, 1200)` yields the encoder's flattened camera-major
  scene-token order (buffer layout). The NEAREST 240→256 resize is
  reimplemented integer-exact in pure numpy (`resize_segmentation_nearest`)
  because cv2 silently retypes int64→int32 and its row mapping is
  `floor(v*240/256)` (recon Q15's native-pooling claim was only
  approximate — corrected).
- `visibility` = seg pixels / projected-OBB bounding-rect area (v2
  definition kept): a fraction-of-image definition would make P4's ≥ 0.3
  gate unreachable for small items.
- `physics.apply_physics` runs after `Env()`/`check_stable` (RoboTwin-side
  files cannot import `main.*`); revisit if Phase 3 shows layout drift.
- `test_smoke_sim` `avoid|box_0|-` min-margin assertion is `< 0.05` (not
  0.02): the WSG gripper's collision spheres do not cover the finger contact
  patch, so sphere-OBB distance bottoms at 4.26 cm during grasp.
- `container_region` default up axis `[0,1,0]` (RoboTwin glb Y-up, recon
  Q13); `zone_region` footprint prefers a `size_xy` attr (Python wrappers)
  else box-collision AABB.
- aloha-agilex: the `per_scene_id` side map is ambiguous on a single
  articulation (same as CEHJ); only ur5-wsg/franka-panda are exercised.

## Open items for Phase 3+

- `probe.py physics` subcommand is a stub (Phase 3 probes P1–P5).
- Token-cell ↔ encoder `pos_mean` alignment is probe P5 (`gpu`).
- T1 spawner (`spawn.py`) and the T1/T2/T3 task files are Phase 4.
- `registry.py` host-kind entries raise `NotImplementedError` until Phase 4.
