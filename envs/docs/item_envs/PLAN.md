> **Local adaptations (agreed with the human; updated 2026-10-01 for plan v3):**
> - Project root is `/work/nvme/bifq/project/LanguageConditionHJ/envs/` (self-contained;
>   this file's "repo root" means that directory). "CEHJ_safe (this repo, write here)" is the
>   `LanguageConditionHJ` checkout; the Phase 5 integration edits target its `main/` copy of CEHJ.
>   The original `CEHJ_safe` repo is never edited.
> - Cluster: NCSA Delta; GPU work goes through `srun --account=bifq-delta-gpu
>   --partition=gpuA40x4-interactive ... apptainer exec --nv .../robot-sim_cu128.sif`
>   (see `.kimi-code/skills/delta-hpc/SKILL.md` in LanguageConditionHJ).
> - This document is plan **v3** (3 tasks: T1 selective_avoid, T2 granular_pour, T3 push_zone).
>   It supersedes the 7-task v2 plan entirely.

---

# Item-Safety Environments on RoboTwin: Implementation Plan for a Coding Agent

Repos: `CEHJ_safe` (this repo, write here) and `RoboTwin` (sibling checkout at `../RoboTwin`).
Simulator: SAPIEN 3.0.0b1 (PhysX 5, CPU physics at 250 Hz) in the `RoboTwin` conda env.
Robots: dual-arm setups only. Default `ur5-wsg` pair, also `franka-panda` pair.

---

## 0. How to use this document

You are building three RoboTwin task environments plus the data infrastructure around them. You are
**not** building or training the safety filter.

Work phase by phase (Section 6). Each phase ends with passing tests, a short report under
`docs/item_envs/`, and a commit with prefix `item_envs(phaseN):`. Phases with a **STOP gate** require a
human review if the gate fails. Do not tune physics until something "looks right" without reporting it.

An API marked **(verify)** must be confirmed in SAPIEN 3.0.0b1 or RoboTwin before use, with the
finding recorded in `docs/item_envs/recon.md`.

Judge physics with **numbers** (distances, tilts, counts), not by watching videos. For every probe
and every task, also save 4 to 8 PNG keyframes, which you can inspect directly.

---

## 1. Mission and context

### 1.1 Research context (read once)

We train a Hamilton-Jacobi (HJ) safety filter on frozen HoloBrain-0 encoders. HoloBrain is a
**bimanual** model, so every environment keeps the RoboTwin dual-arm setup with both arms in the
action space, exactly as in the previous project (CrossSafe, this repo).

The new filter is **item-centric and promptable**. A rule names a consequence type (for example
"don't touch", "don't spill", "keep out of a region") and a **referent**: the item it is about, plus
optionally a region. Downstream, the referent reaches the filter as a **mask** over the scene tokens,
produced from a text prompt (GroundingDINO + SAM), a click, or the ground-truth simulator mask. You do
not handle language. You generate rule specifications, log ground-truth masks and states, and keep
everything relabelable: the same transition must be usable with a different referent.

### 1.2 The three tasks

| ID | Name | Built on | Main failure | Referent matters because |
|---|---|---|---|---|
| T1 | `selective_avoid` | the 5 CrossSafe host tasks, unmodified | robot or held payload touches a **specified** object | several objects sit on or near the path, and only the specified ones must be avoided. Touching the others is allowed |
| T2 | `granular_pour` | new RoboTwin task derived from `place_container_plate` (bowl variant) | beads leave the bowls, or land in a forbidden region | several regions exist, and only the specified one is forbidden |
| T3 | `push_zone` | new RoboTwin task | pushed object leaves the table or enters a forbidden zone, or a bystander tips or moves | several zones and bystanders exist, and only the specified ones are constrained |

### 1.3 Deliverables

1. Pure-numpy margin functions for the consequence types in Section 5, with unit tests.
2. T1 as a multi-object spawner on top of the existing hosts, and T2 and T3 as RoboTwin task files.
3. A risk knob on the nominal planner (plan retiming plus task-specific knobs).
4. Physical-property randomization.
5. A per-tick logger with privileged item state, all candidate-rule margins, contacts, visibility,
   and **token-grid item masks**, plus keyframes and optional videos.
6. Feasibility probes and base-rate calibration per task and embodiment.
7. A minimal hook so CEHJ's rollout and collection code can use item margins, and a sidecar buffer
   for item state and masks. Training code stays untouched.

---

## 2. Ground rules

- **RoboTwin task files.** T2 and T3 are new RoboTwin tasks. Write them in this repo under
  `robotwin_tasks/` and install them into `RoboTwin/envs/` with
  `scripts/item_envs/install_robotwin_tasks.sh` (symlinks, idempotent). **Never modify existing
  RoboTwin files.** Subclass existing tasks where possible. Modified assets go into
  `assets_overlay/` and are loaded by absolute path.
- **New RoboTwin task files must not import CEHJ code** (`main.*`). Anything that needs CEHJ helpers
  (corridors, robot collision spheres, distances) runs on the CEHJ side after the task is built.
- **CEHJ semantics stay identical.** `main/train/config.py`, `buffer.py` `FIELDS`, `hfunc.py`,
  `obstacle.py`, `distance.py`, and the controller behavior for existing tasks must not change.
  Section 6 lists every allowed edit to an existing CEHJ file.
- **Regression.** Before any change and after Phase 5, run the CrossSafe smoke path for
  `place_empty_cup`, seed 0, on `ur5-wsg` and `franka-panda`. The `h` traces must be identical.
- **Dual-arm always.** Both arms are present and in the action space in every episode. A task may use
  one arm or both for its nominal. Never hardcode action widths. UR5 has 6 arm joints and Franka has 7.
  Read them the way `Env.step_dtheta` does. CEHJ's buffer pads to 18.
- **SI units everywhere.** Degrees appear only in reports and YAML fields named `*_deg`.
- **Determinism.** RoboTwin seeds the global numpy RNG in `_init_task_env_`, and its utilities
  (`rand_pose`) use it. Use the global RNG for layout inside `load_actors`. Use a separate
  `np.random.RandomState(seed + 7919)` for physics randomization and risk sampling, and
  `RandomState(seed + 4241)` for T1's post-init spawns, so changing one range never changes the
  layout of another stage.
- **Explicit masses.** `Actor.__init__(actor, data, mass=0.01)` sets every wrapped object to 10 g on
  construction (`envs/utils/actor_utils.py`). Set realistic masses explicitly on every item. The beads
  in RoboTwin's `dump_bin_bigbin` are 0.1 g, which is also unrealistic.
- **Testing.** Pure functions are tested without SAPIEN. Simulator tests are marked
  `@pytest.mark.sim`, GPU tests (cuRobo, HoloBrain) `@pytest.mark.gpu`. Simulator and GPU tests are
  parametrized over `["ur5-wsg", "franka-panda"]`.
- **Headless.** `render_freq=0`. Calibration runs in **no-render mode** (Section 8).

---

## 3. Verified facts about the codebases (read these files first)

### 3.1 RoboTwin

| Fact | Where |
|---|---|
| Lifecycle: `setup_demo(**kw)` calls `_init_task_env_`, which runs `setup_scene` (timestep 1/250, default material static 0.5, dynamic 0.5, restitution 0), `create_table_and_wall`, `load_robot`, `load_camera`, `load_actors()`, then `check_stable()` | `envs/_base_task.py` lines 43 to 160, 201 to 240 |
| `check_stable()` runs 2500 physics steps and raises `UnStableError` if any actor rotates more than 3° within the last 200 steps | `envs/_base_task.py` line 162 |
| Table: 1.2 m (x) × 0.7 m (y), top at z = 0.74 + `table_z_bias`, centered at `table_xy_bias` | `create_table_and_wall` |
| `Actor(entity, data, mass=0.01)` sets the mass on construction | `envs/utils/actor_utils.py` |
| `create_actor(..., convex=False)` builds a triangle-mesh collision, which PhysX supports only for static bodies. `convex=True` calls `add_multiple_convex_collisions_from_file` (one hull per sub-mesh in the collision file) | `envs/utils/create_actor.py` line 501 |
| `create_box` returns an `Actor` with top-down `contact_points_pose`, so `grasp_actor` works. `create_sphere` and `create_cylinder` return raw entities. Raw entity mass: `entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = m` | `envs/utils/create_actor.py`, `envs/dump_bin_bigbin.py` |
| Skills (`grasp_actor`, `place_actor`, `move_by_displacement(arm_tag, x, y, z, quat, move_axis)`, `move_to_pose`, `open_gripper`, `close_gripper`, `back_to_origin`) return `(arm_tag, [Action])`, executed by `self.move(a1, a2=None)`. Two-arm moves run through `together_move_to_pose` | `envs/_base_task.py` lines 864 to 1470 |
| cuRobo plans: `interpolation_dt = 1/250`, `MotionGenPlanConfig(max_attempts=10)`, no time dilation. Result `{"status", "position": [T, n], "velocity": [T, n]}`, one row per physics step | `envs/robot/planner.py` lines 60 to 155 |
| Segmentation: `camera.get_picture("Segmentation")`, channel 1 is the actor-level id. RoboTwin's helper casts to `uint8`, truncating ids above 255, so read the raw channel | `envs/camera/camera.py` line 377 |
| `place_container_plate`: container is `002_bowl` (model ids 1, 2, 3, 5) or `021_cup`, created with `convex=True`, x ∈ ±[0.2, 0.28], y ∈ [−0.1, 0.05]. Plate `003_plate` (scale 0.025, static) near x = ±0.05, y ∈ [−0.15, −0.1]. Arm chosen by container x. Nominal: `grasp_actor` (contact point 0 or 2), lift 0.1 m along the arm axis, `place_actor` onto the plate's functional point, retreat | `envs/place_container_plate.py` |

### 3.2 CEHJ_safe

| Fact | Where |
|---|---|
| `Env(task, embodiment, seed, arm_distance, ...)` imports `envs.{task}`, calls `setup_demo(**_task_args(...))`, then `stretch_task_spawns(self)` (a no-op for tasks missing from `TASK_SPECS`, because it uses `.get`), then `settle_arms()` | `main/envs/env.py` lines 338 to 385, `main/envs/obstacle.py` line 753 |
| `make_training_env(cfg, seed)`: installs the cuRobo controller, builds `Env`, spawns **one static** obstacle with `choose_and_spawn` (corridor parameter t from `_corridor_t(seed)`), then `update_curobo_world(env.robot)` with the **table only**. The nominal never sees the obstacle, which is where unsafe transitions come from | `main/train/collect.py` line 126 |
| Corridor geometry and placement: `longest_corridor`, `iter_corridors`, `keepaways_from_task`, `keepaways_from_robot`, `geometric_pose`, `spawn_obstacle`, `TASK_SPECS` | `main/envs/obstacle.py`, `main/envs/tasks.py` |
| Robot-to-obstacle distance: robot collision spheres per embodiment (`load_collision_spheres`), sphere-to-OBB distances, held-object detection (`detect_held_by_arm`), held-payload OBB distance. All written for a single `env.obstacle` | `main/envs/distance.py`, `main/train/hfunc.py` |
| Host tasks used by CrossSafe: `place_container_plate`, `place_burger_fries`, `stack_blocks_two`, `stack_bowls_two`, `place_bread_basket` (`FrozenConfig.task_choices`) | `main/train/config.py` |
| `TickChunkedController` runs every plan in 10-step ticks with `tick_hook(ctx)`, and routes `together_move_to_pose` through the same path. Plans come from `CuroboIKController.plan(arm, target_pose, **kw)`. A skill tracker (`_install_skill_tracker`) labels each move with its skill | `main/envs/controller.py` lines 260 to 330, 547 to 680, 1054 to 1240 |
| `RolloutController.run()` wraps `scene.step`, calls `_capture_tick()` every tick (float accumulator with a 1e-9 epsilon), and `_capture_tick` calls `compute_h` | `main/train/rollout.py` lines 213 to 290, 433 to 492 |
| `StepBuffer` is a flat memmap ring buffer with module-level `FIELDS` | `main/train/buffer.py` |
| Gotchas: `_update_render()` before `take_picture`, `TMPDIR=/tmp` for cuRobo, one child process per embodiment for long runs | `AGENTS.md` |

---

## 4. Target layout

```
robotwin_tasks/                     # installed into RoboTwin/envs/ by symlink
  granular_pour.py                  # T2: class granular_pour(place_container_plate)
  push_zone.py                      # T3: class push_zone(Base_Task)
  _item_utils.py                    # RoboTwin-side helpers only (zone mats, push primitive, item metadata)
main/item_envs/
  __init__.py
  registry.py        # ITEM_TASKS: name -> TaskEntry(robotwin_task | host_tasks, embodiments, config)
  geometry.py        # OBB, footprints, polygons, signed distances (pure numpy)
  margins.py         # consequence margins (pure numpy)
  robot_distance.py  # multi-item generalization of distance.py: robot + payload to any item
  regions.py         # Region dataclass and constructors
  items.py           # ItemSpec, item registry, metadata read from the task
  rules.py           # RuleSpec, rule enumeration, phrase generation, splits
  risk.py            # RiskParams and samplers
  retime.py          # plan retiming (pure numpy)
  physics.py         # friction, mass, CoM randomization with readback
  masks.py           # per-item visibility and token-grid masks from segmentation
  spawn.py           # T1: multi-object spawner on the host tasks
  controller.py      # RiskTickController(TickChunkedController)
  runner.py          # ItemEpisodeRunner: env + controller + tick hook + logger, no learning
  logger.py          # trace.npz + meta.json + keyframes + optional video
  margin_hook.py     # Phase 5
  item_buffer.py     # Phase 5
  probe.py           # CLI for probes, episodes, sweeps, summaries
  configs/
    physics.yaml  objects.yaml  splits.yaml
    tasks/T1_selective_avoid.yaml  T2_granular_pour.yaml  T3_push_zone.yaml
assets_overlay/objects/<asset>/collision/...
scripts/item_envs/
  install_robotwin_tasks.sh  audit_collisions.py  decompose_assets.py  run_probes.sh  run_calibration.sh
tests/item_envs/
docs/item_envs/
  recon.md  feasibility.md  calibration.md  task_cards/T1.md T2.md T3.md  final_report.md
```

---

## 5. Data contracts

Field names are a contract with the downstream training code. Do not rename them without updating this
file.

```python
# items.py
@dataclass
class ItemSpec:
    item_id: str                  # unique in episode, e.g. "obj_2", "beads", "bystander_0"
    entity: object                # sapien Entity or RoboTwin Actor (use entity_of())
    category: str                 # key into objects.yaml, e.g. "cup", "can", "beads", "box"
    color: str | None             # set for primitives and zone mats, None for textured assets
    kind: str                     # "rigid" | "particle_set"
    particles: list | None        # entities when kind == "particle_set"
    half_extents: np.ndarray      # [3] OBB half extents in the item frame (m)
    obb_center_local: np.ndarray  # [3]
    up_axis_local: np.ndarray     # [3]
    mass: float                   # kg actually set
    role: str                     # "candidate" | "task_object" | "support" | "bystander"
    group: str | None             # task-specific grouping, e.g. "push"

# regions.py
@dataclass
class Region:
    region_id: str                # "table_top", "zone_red", "bowl_src_interior", "target_interior"
    kind: str                     # "surface" | "zone" | "container"
    frame_fn: Callable[[], np.ndarray]   # 4x4 world_T_region, evaluated every tick (bowls move)
    polygon: np.ndarray           # [M, 2] CCW polygon in the region frame xy plane (m)
    z: float
    z_band: float
    nameable: bool                # True if a rule may name it (zones, containers), False for internal regions
    color: str | None
    entity: object | None         # the mat or container entity, for masks

# rules.py
class Consequence(str, Enum):
    AVOID = "avoid"                           # robot links and held payload keep distance >= margin from the item
    INSIDE = "inside"                         # item footprint stays inside a region (on its surface for kind == "surface")
    OUTSIDE = "outside"                       # item footprint stays outside a region
    UPRIGHT = "upright"                       # tilt(item) <= theta_max
    STAY = "stay"                             # item stays within d_tol (and theta_tol) of its pose at rule activation
    PARTICLES_INSIDE = "particles_inside"     # every particle stays in the column above one of the regions
    PARTICLES_OUTSIDE = "particles_outside"   # no particle enters the column above any of the regions

@dataclass
class RuleSpec:
    rule_id: str                  # stable string, e.g. "avoid|obj_2|-", "outside|box_0|zone_red"
    consequence: Consequence
    item_ids: tuple[str, ...]     # usually one item. Several items = set rule ("avoid every cup"), margin = min
    region_ids: tuple[str, ...]   # empty for AVOID, UPRIGHT, STAY
    params: dict                  # {"margin": m} | {"theta_max": rad} | {"d_tol": m, "theta_tol": rad} | {"T": m}
    phrases: list[str]            # generated text prompts (never contain digits)
    split: str                    # "train" | "heldout_item" | "heldout_region" | "heldout_phrase"

# logger.py, one record per control tick (25 Hz)
@dataclass
class StepRecord:
    t: float; tick: int
    qpos: np.ndarray              # raw joints, both arms
    ee_pose: np.ndarray           # [2, 7] TCP poses, wxyz
    gripper: np.ndarray           # [2]
    held: np.ndarray              # [2] item index held by left/right arm, -1 if none
    item_pose: np.ndarray         # [K, 7]
    item_vel: np.ndarray          # [K, 6]
    particle_pos: dict            # item_id -> [N, 3]
    region_T: np.ndarray          # [G, 4, 4] world_T_region for every region this tick
    margins: np.ndarray           # [R] raw SI, in rule-table order
    robot_item_dist: np.ndarray   # [K] min signed distance robot links + payload to item (m)
    robot_item_force: np.ndarray  # [K] robot-item contact force (N)
    visibility: np.ndarray        # [K, n_cam] visible pixel fraction
    token_masks: np.ndarray       # [K + G_nameable, n_cam, n_tok] uint8, fraction of each scene-token cell covered (0..255)
    action_source: int            # 0 planner, 1 actor, 2 perturbation
    time_scale: float

@dataclass
class EpisodeMeta:
    task: str; host_task: str | None; embodiment: str; seed: int
    risk: dict; physics: dict
    items: list[dict]; regions: list[dict]; rules: list[dict]
    token_grid: dict              # per camera: image size, strides, n_tok, token order (must match the encoder)
    active_rule_id: str | None
    task_success: bool
    violations: dict              # rule_id -> bool (min margin < 0)
    min_margin: dict
    first_violation_t: dict
    plan_failures: int
    spawn_rejections: int
    wall_time_s: float
```

**Margin convention:** positive is safe, zero is the boundary, raw SI, never clipped in the
environment. Downstream scaling for reference (in `physics.yaml`): κ = 20 per metre for distance-type
margins (matches CrossSafe's `h_scale`), κ = 2 per radian for `upright`.

**Token-grid masks:** the cells must match the encoder's scene tokens exactly. Read the image
resolution, any crop or resize, the strides (16 and 32), and the token order (scale by scale, finest
first, row-major) from `Env.get_encoder_obs` and `HoloBrainEncoder.encode_visual_tokens`. Write a test
that projects one known item and checks that the mask cells line up with the encoder's token positions
`pos_mean` (`gpu` test, probe P5).

---

## 6. Phases

Time estimates are for a coding agent. "Agent" is active work, "compute" is simulation wall time,
"review" is whether a human must look before you continue.

| Phase | Agent | Compute | Review |
|---|---|---|---|
| 0 Recon | 1 to 2 h | minutes | no |
| 1 Pure-numpy core | 1 to 2 h | none | no |
| 2 Simulation infrastructure | 3 to 5 h | short runs | no |
| 3 Feasibility probes | 2 to 3 h | 1 to 2 h | STOP gate |
| 4 Tasks T1, T3, T2 | 2 to 4 h each | calibration, 1 to 3 h per task and embodiment in no-render mode, parallel | short check per task |
| 5 Integration | 2 to 4 h | under 1 h (GPU) | no |

### Phase 0: Reconnaissance

1. Activate the `RoboTwin` env, confirm the assets are downloaded, and run the regression reference
   (Section 2). Save the `h` traces under `docs/item_envs/regression/`.
2. Answer in `recon.md`, each with a snippet you actually ran:
   - (verify) Setting a physical material on existing collision shapes, for example
     `component.get_collision_shapes()` and `shape.set_physical_material(mat)`. Otherwise, how to pass
     `material=` at build time.
   - (verify) Linear and angular damping, center-of-mass pose, CCD, kinematic bodies with
     `set_kinematic_target`, and solver iteration counts.
   - (verify) Contact impulses between two specific entities from `scene.get_contacts()` (the existing
     pattern is in `distance.py` `_robot_obstacle_contact_force`).
   - (verify) Actor-level segmentation ids and the entity-to-`per_scene_id` mapping. Which cameras
     exist, their resolution, and the exact image preprocessing in `Env.get_encoder_obs`.
   - For the `ur5-wsg` and `franka-panda` pairs: arm base poses (`ARM_DISTANCE` in `collect.py`), arm
     DoF, URDF velocity limits, gripper maximum opening and finger length, and the reachable tabletop
     workspace from an IK sweep (2 cm grid, heights table + 0.02, + 0.10, + 0.20 m, top-down gripper),
     saved as `configs/workspace_<embodiment>.npz`.
   - Collision audit for `002_bowl` ids 1, 2, 3, 5, `003_plate`, `062_plasticbox`, `015_laptop`, and
     every object in T1's candidate pool: number of convex pieces, hull-to-mesh volume ratio, and
     whether `model_data*.json` has `contact_points_pose`. Script: `audit_collisions.py`.
   - Confirm that symlinked files in `RoboTwin/envs/` import correctly through `Env` (`envs.{task}`),
     and that nothing in RoboTwin rejects unknown task names when `eval_mode=False`.

### Phase 1: Pure-numpy core

Implement `geometry.py`, `margins.py`, `retime.py`, `rules.py`, `risk.py`. No SAPIEN imports.

```python
# geometry.py
def quat_wxyz_to_mat(q); def pose_to_mat(p, q); def to_frame(points_world, world_T_frame)
def obb_corners(world_T_item, center_local, half) -> np.ndarray          # [8, 3]
def footprint(world_T_item, center_local, half, up_local) -> np.ndarray  # 4 corners of the face facing -z
def signed_dist_inside_polygon(pts_xy, poly) -> np.ndarray               # > 0 inside, metres
def circle_polygon(radius, n=16) -> np.ndarray
def tilt_angle(world_T_item, up_local) -> float

# margins.py (all positive = safe)
def m_avoid(robot_item_dist, margin=0.0) -> float                        # distance comes from robot_distance.py
def m_inside(item_pts_region, region, require_on_surface) -> float       # continuous also in z (no big constants)
def m_outside(item_pts_region, region) -> float
def m_upright(tilt, theta_max) -> float
def m_stay(p_now, p_ref, tilt_now, d_tol, theta_tol) -> float
def m_particles_inside(particle_xy_per_region, regions, T=0.005) -> float   # per particle max over regions, soft-min over particles
def m_particles_outside(particle_xy_per_region, regions, T=0.005) -> float
def softmin(x, T) -> float
def set_rule_margin(per_item_margins) -> float                           # min over the items of a set rule

# retime.py
def retime(position, velocity, s, dt=1/250, v_limit=None) -> (position, velocity, info)
    # uniform time scaling, s > 1 faster. Exact endpoints. velocity scaled by s.
    # info = {"s", "T_in", "T_out", "v_ratio"}. Never clip. Callers log v_ratio.

# risk.py
@dataclass
class RiskParams:
    time_scale: float = 1.0          # risky segments
    safe_time_scale: float = 0.6     # approach, grasp, retreat
    task: dict = field(default_factory=dict)
def sample_risk(task_cfg, rng) -> RiskParams

# rules.py
def enumerate_rules(items, regions, task_cfg) -> list[RuleSpec]
def render_phrases(rule, objects_cfg, rng, n=3) -> list[str]   # templates per consequence, no digits
def assign_split(rule, splits_cfg) -> str
```

Phrase templates (store in `rules.py`):

- avoid: "don't touch the {item}", "avoid the {item}", "stay away from the {item}". Set rules:
  "don't touch any {category}", "avoid all the {category}s".
- inside: "keep the {item} on the {region}", "don't push the {item} off the {region}".
- outside: "keep the {item} out of the {region}", "don't push the {item} into the {region}".
- upright: "don't knock over the {item}", "keep the {item} upright".
- stay: "don't disturb the {item}", "leave the {item} where it is".
- particles_inside: "don't spill the {item}".
- particles_outside: "keep the {item} off the {region}", "don't get any {item} in the {region}".

`{item}` and `{region}` come from `objects.yaml` names and synonyms, with a color prefix when the item
or region has a color.

Tests (`not sim`):

- polygon signed distance on a unit square (center 0.5, edge 0, outside −0.2);
- the footprint of an upright box is its bottom face, and at 90° tilt it is the side that now faces
  down;
- `m_inside` and `m_outside` are continuous across the polygon edge and across the `z_band` boundary
  (finite differences at ±1e-4 m);
- `m_upright` is zero exactly at `theta_max`;
- the particle soft-min is at most the hard min, and within `T * log(N)` of it;
- `retime` with s = 1 is the identity, endpoints are exact for s ∈ {0.3, 0.7, 1.3, 2.0}, duration
  scales as 1/s within one row, and velocity scales by s;
- `enumerate_rules` produces stable `rule_id`s, and `render_phrases` never emits digits.

### Phase 2: Simulation infrastructure

**2.1 `physics.py`.** `apply_material`, `set_mass_props(entity, mass, com_offset_local)`,
`sample_physics(items, cfg, rng)`, `apply_physics(task, sampled)`, and `read_back(entity)`. Draw
static friction from U(0.3, 0.9), then dynamic friction from U(0.2, static), restitution from
U(0, 0.1), mass scale from U(0.7, 1.3) around the category's nominal mass, and a CoM offset of
U(−0.2, 0.2) × half extents in x and y. Sim test: apply, then read back.

**2.2 `robot_distance.py`.** Generalize `distance.py` from one `env.obstacle` to any list of items,
**without changing `distance.py`**. Import its helpers (`load_collision_spheres`,
`_arm_collision_spheres`, `point_obb_signed_distance`, `detect_held_by_arm`, `_actor_obb_distance`).

```python
def robot_item_distances(env, items: list[ItemSpec]) -> np.ndarray   # [K], robot links + held payload
def robot_item_forces(env, items) -> np.ndarray                      # [K], contact force in N
```

Rule: an item held by a gripper is part of the robot body for every **other** item's distance, and its
own distance is +inf while it is held. Sim test: with the single CrossSafe obstacle in
`place_empty_cup`, the result equals `distance_info(env)["d_system"]` to 1e-6 m.

**2.3 `regions.py`.** `table_region(task)`, `zone_region(mat_actor, color)`, and
`container_region(actor, rim_radius, region_id)`. A container region follows its actor every tick and
uses a 16-gon at the inner rim radius in the actor's up-axis frame.

**2.4 `masks.py`.**

```python
def item_masks(task, items, regions, cameras) -> tuple[np.ndarray, np.ndarray]
    # returns visibility [K, n_cam] and token_masks [K + G_nameable, n_cam, n_tok] (uint8)
```

Read the raw actor-level segmentation id (no uint8 cast), apply the encoder's preprocessing, then
average-pool each item's binary mask into each token cell at strides 16 and 32. A particle set's mask
is the union of its particles. Nameable regions get masks too (zone mats, containers). Call
`task._update_render()` first. Computed every tick in collection mode, and only at t = 0 in no-render
calibration mode.

**2.5 `controller.py`.**

```python
class RiskTickController(TickChunkedController):
    def plan(self, arm, target_pose, **kw):
        result = super().plan(arm, target_pose, **kw)
        if result and result.get("status") == "Success":
            s = float(getattr(self.env.task, "active_time_scale", 1.0))
            pos, vel, info = retime(result["position"], result["velocity"], s, v_limit=self._v_limit(arm))
            result["position"], result["velocity"], result["_item_retime"] = pos, vel, info
        return result
```

T2 and T3 set `active_time_scale` around risky segments with a context manager `risky(task, risk)`.
Approach, grasp, and retreat stay at `safe_time_scale`. T1 runs unmodified host tasks, so its risky
segments are chosen by skill type: hook the skill tracker so that moves labeled as transport or place
use `time_scale` and everything else uses `safe_time_scale`. If `v_ratio > 1`, log it together with
CEHJ's realized-versus-commanded ratio. Do not clip silently.

**2.6 `runner.py` and `logger.py`.** `ItemEpisodeRunner(task_key, embodiment, seed, host=None,
risk=None, active_rule_id=None, render="full" | "none", video=False)`:

- builds the environment through the registry (T1 through `spawn.py` on a host, T2 and T3 through
  `Env` with their RoboTwin task names), calls `update_curobo_world(env.robot)` with the table only,
  attaches `RiskTickController`, and wraps `scene.step` with the tick accumulator from
  `RolloutController.run()` (keep the 1e-9 epsilon);
- each tick, writes a `StepRecord`;
- at the end, writes `meta.json`, `trace.npz`, 4 to 8 keyframes (start, first contact with each
  candidate, first violation, end), and an optional observer video with an overlay of the active
  rule's phrase and margin, the min margin over all rules, and the time scale;
- catches plan failures and `UnStableError` and records them, never crashing a sweep.

Output: `runs/item_envs/<task>/<embodiment>/seed_<k>/`.

**2.7 `probe.py`.**

```
python -m main.item_envs.probe episode   --task selective_avoid --host place_container_plate --seed 0 --embodiment ur5-wsg --video
python -m main.item_envs.probe sweep     --task push_zone --grid time_scale=0.5,0.75,1.0,1.25 overshoot=0,0.04,0.08 \
                                          --episodes 5 --adaptive --render none --out runs/item_envs/calib/T3
python -m main.item_envs.probe physics   --probe P2 --out runs/item_envs/feas/P2
python -m main.item_envs.probe summarize runs/item_envs/calib/T3
```

Sweeps spawn one child process per (task, embodiment, shard), as in `AGENTS.md`.

**Phase 2 acceptance:** a test-only `smoke_items` task (one zone mat, two boxes) runs end to end on
both embodiments, and the margins in the keyframe overlays are correct.

### Phase 3: Feasibility probes (STOP gate)

Use kinematic movers where noted, so physics is isolated from the planner. Results, plots, and
keyframes go in `feasibility.md`.

| Probe | Setup | Sweep | Gate |
|---|---|---|---|
| P1 multi-object path crossing (T1) | for each host task, 50 seeds, spawn 3 to 5 candidates with `spawn.py`, run the nominal with no filter, robot unaware of objects | both embodiments | in ≥ 50% of episodes the nominal comes within 2 cm of at least 2 candidates. In ≥ 30% it touches at least one. With all candidates removed, the host task succeeds in ≥ 70% of episodes |
| P2 bowl containment and beads (T2) | `002_bowl` ids 1, 2, 3, 5 with N ∈ {30, 60, 100} beads, r ∈ {5, 6, 8} mm. Kinematic bowl: (a) translate with peak accel {1, 2, 4, 8} m/s², (b) tilt to 110° at ω ∈ {1, 3} rad/s | as left | beads rest **inside** the bowl at rest (the bowl's collision is not filled in). Zero beads tunnel through the bowl or the table. Spill increases with accel and with ω. Physics ≤ 20 ms per control tick at N = 60 (report the CPU). If a bowl id fails containment, use a CoACD overlay or drop that id |
| P3 push coast (T3) | kinematic pusher moves a box at v ∈ {0.1, 0.3, 0.6} m/s for 0.15 m, then stops | friction {0.3, 0.6, 0.9} | coasting distance > 0, increases with v, decreases with friction |
| P3-robot push primitive | the push primitive (Section 7, T3) on both embodiments, 20 seeds, single push | time_scale {0.5, 1.0, 1.3} | at time_scale 0.5 the box ends within 3 cm of the commanded push distance in ≥ 80% of seeds |
| P4 visibility | initial layout of each task, 20 seeds | | every candidate item and every nameable region has visible fraction ≥ 0.3 in at least one camera at t = 0 in ≥ 95% of seeds |
| P5 token alignment (`gpu`) | one item at a known pose | | the token-mask cells with the highest coverage contain the encoder token positions closest to the item centroid |

If a gate fails, try at most two documented fixes (solver iterations, CCD, decomposition, layout
change). If it still fails, stop and report.

### Phase 4: Tasks

Order: **T1, T3, T2**. T1 reuses the most existing code, and T2 depends on the P2 outcome. For each
task:

1. implement it (Section 7) with its YAML,
2. write the task card `docs/item_envs/task_cards/Tn.md`,
3. add `sim` tests on both embodiments: 20 seeds build without `UnStableError`, rules enumerate, all
   margins are positive at t = 0, one nominal episode completes at low risk,
4. calibrate (Section 8).

### Phase 5: Integration with CEHJ collection

Allowed edits to existing CEHJ files:

1. `main/train/rollout.py`: optional constructor arguments `h_fn=None` and
   `controller_cls=TickChunkedController`. `_capture_tick` calls
   `(self.h_fn or default_h_fn)(self.env, self.cfg)`, where `default_h_fn` wraps the current
   `compute_h` call byte for byte.
2. `main/train/collect.py` `make_training_env`: if `cfg.task` is an item task, build the environment
   through `main.item_envs.registry` instead of the single-obstacle path.

New code:

- `margin_hook.py`: `ItemMarginHFn(active_rule_id, kappa)` returns `(kappa * margin_active, diag)`.
  `diag` contains all margins and item state, plus the keys the existing video and metric code reads
  (`d_left`, `per_link`, `holding`, and so on), filled with `inf` or empty values where they do not
  apply. For T1 with a single-item `avoid` rule, `per_link` is filled from the per-link distances to
  that item, so CrossSafe's diagnostics keep working.
- `item_buffer.py`: `ItemSidecarBuffer(root, capacity, k_max=8, r_max=64, g_max=8, n_cam=3, n_tok)`,
  same ring semantics as `StepBuffer`, written in lockstep (same index):

  | field | dtype | shape |
  |---|---|---|
  | `item_pose` | float32 | [k_max, 7] |
  | `item_vel` | float32 | [k_max, 6] |
  | `item_obb` | float32 | [k_max, 9] |
  | `item_valid` | bool | [k_max] |
  | `held` | int8 | [2] |
  | `robot_item_dist` | float32 | [k_max] |
  | `margins` | float32 | [r_max] |
  | `rule_valid` | bool | [r_max] |
  | `region_T` | float32 | [g_max, 4, 4] |
  | `token_masks` | uint8 | [k_max + g_max, n_cam, n_tok] (items, then nameable regions) |
  | `visibility` | float16 | [k_max, n_cam] |

  Each episode's item, region, and rule tables go to a JSON-lines file keyed by `episode_id`. No text
  is stored. Phrases are regenerated from the rule table at training time.
- Tests: lockstep indices after a ring wrap. A `gpu` dry run of 3 nominal episodes per task and
  embodiment into a fresh buffer, with buffer margins matching `trace.npz` exactly. The Phase 0
  regression traces are unchanged.

Do not touch `train.py`, the networks, or the buffer `FIELDS`.

---

## 7. Task specifications

Common to all three:

- **Dual-arm setup** on `ur5-wsg` (default) and `franka-panda`. Layouts are sampled inside the
  intersection of both embodiments' workspaces (Phase 0), so a seed gives the same scene on both. Reject
  and resample layouts where a key grasp, place, or push pose fails IK on either embodiment, and log
  `spawn_rejections`.
- **Embodiment differences live in data** (`embodiment_overrides:` in the task YAML), never in
  `if embodiment == ...` branches.
- **Task success** is the nominal's goal only. Safety outcomes are recorded per rule.
- **The referent must matter.** Every scene has at least two candidates of the same rule type that are
  exposed to the nominal (on or near its path, or adjacent to its target), so protecting the wrong one
  changes the outcome.
- **Nominal masses** live in `objects.yaml`: bottle 0.05 to 0.30 kg, cup or mug 0.05 to 0.25, can
  0.05 to 0.35, wooden block 0.05 to 0.15, boxed items 0.05 to 0.30, bead 0.0005 to 0.002.

### T1 `selective_avoid`

**Goal.** CrossSafe's collision avoidance, but with several objects on and near the path, where only
the **specified** objects must not be touched.

**Hosts.** The five CrossSafe tasks, unmodified: `place_container_plate`, `place_burger_fries`,
`stack_blocks_two`, `stack_bowls_two`, `place_bread_basket`. No new RoboTwin file. Placement needs the
robot and the task corridors, so spawning happens on the CEHJ side after `Env` is built, exactly like
`make_training_env`.

**Spawner (`spawn.py`).** Generalize `choose_and_spawn` to K objects without changing `obstacle.py`
(reuse `longest_corridor`, `iter_corridors`, `keepaways_from_task`, `keepaways_from_robot`,
`geometric_pose`):

- K ∈ [3, 5]. For each object, choose a placement class: **on-path** (corridor parameter
  t ∈ [0.22, 0.48] as in CrossSafe, at most 2 objects), **near-path** (2 to 6 cm lateral offset from the
  corridor), or **off-path** (elsewhere on the table, at least 10 cm away). At least two objects must be
  on- or near-path.
- Objects keep at least 3 cm from each other and from the host's keepaways.
- **Objects are dynamic** with realistic masses (CrossSafe's were static). The robot may push
  non-specified objects aside, which is allowed. Add a config flag `static_objects: true` for a
  parity run with CrossSafe.
- After spawning, settle 250 physics steps and reject the layout if any object moved more than 5 mm or
  tilted more than 3°.
- `update_curobo_world(env.robot)` with the **table only**. The nominal never plans around any object.

**Object pool (`objects.yaml`).** Upright-stable RoboTwin objects 5 to 20 cm tall, for example
`068_boxdrink`, `086_woodenblock`, `001_bottle`, `114_bottle`, `021_cup`, `039_mug`, `071_can`,
`080_pillbottle`, `105_sauce-can`, `112_tea-box`, `113_coffee-box`. The agent extends the pool after the
Phase 0 audit. Also add colored primitive boxes and cylinders (6 training colors, 3 held-out), so that
two objects of the same category can be told apart only by color.

**Items.** Spawned objects are `candidate`. The host's pick objects are `task_object`, and the host's
targets (plate, basket, other bowl) are `support`.

**Rules.**

- `avoid(obj_k, margin=0)` for every candidate.
- Set rules `avoid(all candidates of category c)` whenever a category appears at least twice.
- `avoid(all candidates)` is CrossSafe's setting, kept as a baseline rule.
- Optional `stay(obj_k, d_tol=0.03)`, which catches chained contact: a non-specified object pushed into
  a specified one.
- Never an `avoid` rule on `task_object` or `support` items.

**Risk knobs.** `time_scale` ∈ [0.6, 1.3] on the transport segments. The main source of risk is
placement, not speed. `task.on_path_count` ∈ {1, 2}.

**Success.** The host's own `check_success()`.

### T2 `granular_pour`

**Goal.** Carry a bowl of beads and pour them into a target container without spilling, and keep beads
out of a specified forbidden region.

**RoboTwin file.** `robotwin_tasks/granular_pour.py`, defining
`class granular_pour(place_container_plate)`. Override `load_actors`:

1. Force the container to `002_bowl` with ids that passed P2. Keep the host's container pose sampling.
2. Spawn N beads (default from P2, for example 60 at radius 6 mm) in a grid inside the bowl, with
   masses 0.5 to 2 g. Name them `bead_<i>`. They form one `particle_set` item, category `beads`, with
   synonyms "beans", "rice", "grains".
3. Replace the plate with a **target container**: a second `002_bowl` (static) or `062_plasticbox`
   (static, if it passes the audit), placed where the host puts the plate.
4. Spawn **two zone mats** of different colors near the target (one may be replaced by a static
   `015_laptop`). Only one of them is forbidden in any given rule.
5. Beads may jitter during `check_stable`, so keep a set of bead entity names and override
   `check_stable` to skip them (same algorithm otherwise).

Override `play_once`:

1. grasp the bowl as the host does (contact point 0 or 2, `safe_time_scale`),
2. lift 0.1 m along the arm axis,
3. `with risky()`: move above the target at height `task.pour_height` above its rim, with lateral
   offset `task.pour_offset` toward one of the zones,
4. `with risky()`: tilt by 100° to 120° about the horizontal axis perpendicular to the arm at rate
   `task.tilt_rate` (an EE rotation through `move_to_pose`, timed by the retimer), hold 1 s, tilt back,
5. place the empty bowl beside the target and retreat (`safe_time_scale`).

**Bimanual variant** (`bimanual: true`): the second arm grasps the target container (a dynamic
`002_bowl` in this variant) and holds it 5 to 10 cm above the table during the pour, then sets it down.
This uses both arms, and the target moves, so its region follows it every tick.

**Regions.** `bowl_src_interior`, `target_interior` (both `container`, following their actors), and
`zone_<color>` (nameable).

**Rules.**

- `particles_inside(beads, [bowl_src_interior, target_interior])`: "don't spill the beans".
- `particles_outside(beads, [zone_c])` for each zone: "keep the beans off the red mat".

**Risk knobs.** `time_scale` ∈ [0.5, 1.3], `task.tilt_rate` ∈ [1, 4] rad/s, `task.pour_height` ∈
[0.05, 0.20] m, `task.pour_offset` ∈ [0, 0.05] m.

**Extra logs.** Spill count (beads below table + 2r outside every container footprint), and physics
milliseconds per tick.

**Success.** At least 80% of the beads are inside the target.

### T3 `push_zone`

**Goal.** Push objects to goals near the table edge and near zones, without pushing them off the table
or into the specified zone, and without disturbing specified bystanders.

**RoboTwin file.** `robotwin_tasks/push_zone.py`, defining `class push_zone(Base_Task)`.

`load_actors`:

1. Two push objects P_L and P_R, one per side of the table (x < 0 and x > 0): primitive boxes of
   random size (side 4 to 8 cm, height 3 to 8 cm) and color, or `113_coffee-box` or `112_tea-box`.
2. A goal G for each, 5 to 15 cm from the table's front or side edge, or adjacent to a zone mat.
3. Two or three zone mats of different colors, at least one within 6 cm of some push line.
4. One or two tippable bystanders (`001_bottle`, `021_cup`, tall colored cylinders) 2 to 8 cm off the
   push lines.

`play_once`: **both arms push at the same time**, each pushing the object on its side, which keeps both
arms active as HoloBrain expects. The push primitive goes in `_item_utils.py`, implemented with
RoboTwin skills only:

1. close the gripper,
2. move to a pre-push pose 5 cm behind P along the push direction, at P's CoM height (at least 1 cm
   above the table), fingers vertical,
3. `with risky()`: `move_by_displacement` along the push direction by `dist(P, G) + task.overshoot`,
   with the contact point shifted sideways by `task.lateral_offset`,
4. retreat 5 cm and lift.

Use `self.move(push_left, push_right)` so both arms run through `together_move_to_pose`. The planner's
world model holds only the table, so plans that pass through P are expected. That is how a push works.

**Single-push variant** (`n_push: 1`): one push object, and the other arm moves to a pre-push pose and
holds. Use it for debugging and for probe P3-robot.

**Regions.** `table_top` and `zone_<color>` (nameable).

**Rules.**

- `inside(P_k, table_top)`: "don't push the blue box off the table".
- `outside(P_k, zone_c)` for every zone: "keep the blue box out of the red zone".
- `upright(Y_j, θ_max = 45°)` and `stay(Y_j, d_tol = 0.03)`.

**Risk knobs.** `time_scale` ∈ [0.5, 1.3], `task.overshoot` ∈ [0, 0.08] m, `task.lateral_offset` ∈
[−0.02, 0.02] m.

**Success.** Each push object within 3 cm of its goal.

---

## 8. Calibration protocol (per task and embodiment)

**No-render mode.** Calibration needs margins, not images. Run with rendering and cameras off except
one frame at t = 0 for the P4 visibility check. Keyframes are written only for 5 episodes per grid cell.

1. **Adaptive grid.** Start with 5 episodes per cell over the task's knob grid. Add 15 more episodes
   only in cells whose primary-rule violation rate falls between 0.3 and 0.95.
2. For each rule template, record the violation rate, the min-margin distribution, and the time of
   first violation.
3. Pick a **collection risk distribution** (a mixture over the grid cells) such that:
   - the nominal is **usually unsafe** (user decision 2026-10-04, replaces the original 0.2–0.5
     band): the primary rule's violation rate is ≥ 0.5 (target 0.6–0.9) over the scenes where the
     conditioned referent is exposed — every T2 / T4 episode, T1 with the referent on the path,
     T3 A-type layouts. The **benign** scenes (T1 referent off the path, T3 B / B-B layouts: every
     blocking obstacle is a non-referent) stay in the mixture and stay ~0, so the filter must also
     learn when NOT to intervene,
   - in the same scenes, **non-specified** candidates are also violated at a comparable rate, so a
     filter that protects everything pays a visible task-success cost (T1: at least one non-specified
     object touched in ≥ 30% of episodes),
   - nominal task success is ≥ 0.7 at the lowest-risk cell.
4. Write the distribution into the task YAML under `risk_collect` and the evidence into
   `calibration.md`, with one table per embodiment.

If a task misses these numbers after documented redesign attempts, report it rather than lowering the
bar.

---

## 9. Configuration examples

`configs/tasks/T3_push_zone.yaml`

```yaml
task: push_zone
robotwin_task: push_zone
embodiments: [ur5-wsg, franka-panda]     # first is the default
n_push: 2
push_objects: {pool: [prim_box, 113_coffee-box, 112_tea-box], side: [0.04, 0.08], height: [0.03, 0.08]}
goal: {edge_dist: [0.05, 0.15]}
zones: {n: [2, 3], size_xy: [[0.08, 0.15], [0.08, 0.15]], near_push_line: 0.06}
bystanders: {n: [1, 2], pool: [001_bottle, 021_cup, prim_tall_cylinder], line_offset: [0.02, 0.08]}
risk:
  time_scale: {dist: choice, values: [0.5, 0.75, 1.0, 1.15, 1.3]}
  safe_time_scale: 0.6
  task:
    overshoot: {dist: uniform, range: [0.0, 0.08]}
    lateral_offset: {dist: uniform, range: [-0.02, 0.02]}
rules:
  - {consequence: inside, items: {role: candidate, group: push}, regions: [table_top]}
  - {consequence: outside, items: {role: candidate, group: push}, regions: {kind: zone}}
  - {consequence: upright, items: {role: bystander}, theta_max_deg: 45}
  - {consequence: stay, items: {role: bystander}, d_tol: 0.03, theta_tol_deg: 20}
embodiment_overrides:
  ur5-wsg:      {prepush_dist: 0.05, push_height_offset: 0.0}
  franka-panda: {prepush_dist: 0.05, push_height_offset: 0.0}
risk_collect: null                       # filled by calibration
```

`configs/tasks/T1_selective_avoid.yaml`

```yaml
task: selective_avoid
host_tasks: [place_container_plate, place_burger_fries, stack_blocks_two, stack_bowls_two, place_bread_basket]
embodiments: [ur5-wsg, franka-panda]
objects: {k: [3, 5], max_on_path: 2, min_on_or_near_path: 2, near_path_offset: [0.02, 0.06],
          off_path_min_dist: 0.10, min_separation: 0.03, static_objects: false}
corridor_t: [0.22, 0.48]
risk:
  time_scale: {dist: choice, values: [0.6, 0.8, 1.0, 1.15, 1.3]}
  safe_time_scale: 0.6
  task:
    on_path_count: {dist: choice, values: [1, 2]}
rules:
  - {consequence: avoid, items: {role: candidate}, margin: 0.0}
  - {consequence: avoid, items: {role: candidate, same_category_min: 2}, set_rule: true, margin: 0.0}
  - {consequence: avoid, items: {role: candidate}, set_rule: all, margin: 0.0}     # CrossSafe baseline
  - {consequence: stay, items: {role: candidate}, d_tol: 0.03, theta_tol_deg: 20}
risk_collect: null
```

`configs/splits.yaml` (excerpt)

```yaml
selective_avoid:
  heldout_item_categories: [pill_bottle, sauce_can, tea_box]
  heldout_colors: [purple, orange, cyan]
granular_pour:
  heldout_regions: [laptop]
  heldout_colors: [purple, orange, cyan]
push_zone:
  heldout_colors: [purple, orange, cyan]
  heldout_item_categories: [tea_box]
all:
  heldout_phrase_templates: ["stay away from the {item}", "don't get any {item} in the {region}"]
```

---

## 10. Final report

`docs/item_envs/final_report.md`: per task and embodiment, the build status, probe gate results,
calibration numbers (violation rate per rule, non-specified violation rate, nominal success, wall time
per episode, physics milliseconds per tick), keyframe sets for a safe, a near-miss, and a violation
episode, known issues, and every deviation from this plan with its reason.

---

## 11. Out of scope, and decisions reserved for the human

Out of scope: grounding models, language, encoder changes, critic and actor changes, training
scripts, hardware.

Ask before changing any of these:

1. The embodiment set (`ur5-wsg` default, `franka-panda` supported).
2. The held-out lists in `splits.yaml`.
3. Dropping a task, a host task, or a bowl id after a failed gate.
4. The consequence types or the margin sign convention.
5. Making T1 objects static by default.

