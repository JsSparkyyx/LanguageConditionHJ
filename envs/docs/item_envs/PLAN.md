# Item-Safety Environments: Implementation Plan for a Coding Agent

Target repos: `CEHJ_safe` (this repo, write here) and `RoboTwin` (sibling checkout, **read-only**).
Simulator: SAPIEN 3.0.0b1 (PhysX 5, CPU physics at 250 Hz) inside the `RoboTwin` conda env.

> **Local adaptations (agreed with the human, 2026-09-30):**
> - Project root is `/work/nvme/bifq/project/LanguageConditionHJ/envs/` (self-contained;
>   this file's "repo root" means that directory).
> - Phase 2.7 / Phase 5 integration edits target the `LanguageConditionHJ/main/` copy of CEHJ.
> - This session covers Phase 0 + Phase 1 only.
> - Cluster: NCSA Delta; GPU work goes through `srun --account=bifq-delta-gpu
>   --partition=gpuA40x4-interactive ... apptainer exec --nv .../robot-sim_cu128.sif`
>   (see `.kimi-code/skills/delta-hpc/SKILL.md` in LanguageConditionHJ).

---

## 0. How to use this document

You are building simulation environments and data infrastructure. You are **not** building or training
the safety filter. Work phase by phase. Each phase ends with:

1. passing tests listed in that phase,
2. a short report file under `docs/item_envs/`,
3. a commit with message prefix `item_envs(phaseN):`.

Some phases have **STOP gates**. If a gate fails, stop, write the report with numbers, videos, and your
diagnosis, and wait for a human. Do not tune physics until a probe "looks right" without reporting it.

When this plan names an API as **(verify)**, confirm the exact name in SAPIEN 3.0.0b1 or RoboTwin
before using it, and record what you found in `docs/item_envs/recon.md`.

---

## 1. Mission and context

### 1.1 What the downstream project needs

We train a Hamilton-Jacobi (HJ) safety filter that blocks unsafe actions of a nominal planner. The
previous version (`main/train/hfunc.py`) used a failure margin `h = min distance(robot ∪ held object,
obstacle)`. That margin depends only on robot state, because the obstacle is static. Such constraints
are better handled by analytic CBFs, so they are not interesting for a learned filter.

The new project targets **item-centric safety**. The failure is defined on the state of an item whose
state is **not** a function of the robot's joint state. The item moves on its own through contact,
friction, rolling, or flow. Examples are "don't let the glass fall off the tray", "don't let the ball
roll off the table", and "don't spill the beans outside the bowl".

The safety rule names an item and optionally a region in language. You do not handle language
understanding. You only generate the rule specifications and phrases, and you log everything needed to
train a language-conditioned filter later.

### 1.2 What you deliver

1. A package `main/item_envs/` that defines seven RoboTwin-compatible tasks (T1 to T7, Section 7).
2. Pure-numpy failure-margin functions for four rigid consequence types and two particle-set
   variants, with unit tests.
3. A risk knob on the nominal planner (plan retiming plus task-specific knobs).
4. Physical-property randomization and asset collision fixes.
5. A per-tick episode logger with privileged item state, all candidate-rule margins, contacts,
   visibility, and debug videos.
6. Feasibility probes and base-rate calibration reports per task.
7. A minimal hook so the existing CEHJ rollout/collection code can use item margins instead of
   obstacle distance, plus a sidecar buffer for item state. Training code stays untouched.

---

## 2. Ground rules

- **Never edit files under `RoboTwin/`.** Import and subclass only. Modified assets go into
  `assets_overlay/` in this repo, and are loaded by absolute path.
- **Do not change existing CEHJ semantics.** `main/train/config.py`, `buffer.py` `FIELDS`,
  `hfunc.py`, and the controller behavior for existing tasks must stay identical. Section 6, Phase 5
  lists the only allowed edits to existing files. Before Phase 5, existing code is read-only.
- **Regression check.** Run `scripts/run_smoke.sh` (or the closest smoke script, see Phase 0) before
  your first change and after Phase 5. Old tasks must produce the same `h` trace for a fixed seed.
- **SI units everywhere.** Metres, kilograms, seconds, radians. Degrees appear only in human phrases
  and in reports.
- **Determinism.** All randomness flows from the episode seed. RoboTwin seeds the global numpy RNG in
  `_init_task_env_` (`np.random.seed(seed)`), and its utilities (`rand_pose`, `rand_create_actor`) use
  the global RNG. Use the global RNG inside `load_actors`, and use a separate
  `np.random.RandomState(seed + 7919)` for physics randomization and risk sampling, so that changing a
  physics range never changes the scene layout.
- **Explicit masses.** RoboTwin's `Actor.__init__(actor, data, mass=0.01)` sets every wrapped object
  to 10 g on construction (`envs/utils/actor_utils.py`). Every item in our tasks must have its mass set
  explicitly after creation.
- **Pure functions are tested without the simulator.** Tests that need SAPIEN are marked
  `@pytest.mark.sim`, and tests that need a GPU (HoloBrain encoder, cuRobo) are marked
  `@pytest.mark.gpu`.
- **Keep the run scripts headless.** No viewer. `render_freq=0`, as `_task_args` already sets.

---

## 3. Verified facts about the codebases (read these files first)

### 3.1 RoboTwin

| Fact | Where |
|---|---|
| Task lifecycle: `setup_demo(**kw)` calls `_init_task_env_`, which calls `setup_scene` (timestep 1/250, default material static 0.5, dynamic 0.5, restitution 0), then `create_table_and_wall`, `load_robot`, `load_camera`, `load_actors()`, optional clutter, then `check_stable()` | `envs/_base_task.py` lines 43 to 160, 201 to 240 |
| `check_stable()` steps 2000 + 500 physics steps over `scene.get_all_actors()`, and raises `UnStableError` if any actor's orientation changes more than 3° within the last 200 steps | `envs/_base_task.py` line 162 |
| Table: 1.2 m (x) by 0.7 m (y), top surface at z = 0.74 + `table_z_bias`, centered at `table_xy_bias` | `create_table_and_wall`, `create_table` |
| `Actor(entity, data, mass=0.01)` sets the mass on construction. `ArticulationActor` has `get_qpos`, `set_qpos`, `set_properties(damping, stiffness, friction, force_limit)`, `get_link_dict()` | `envs/utils/actor_utils.py` |
| `create_actor(..., convex=False)` builds a **non-convex triangle-mesh** collision, which PhysX supports only for static bodies. `convex=True` calls `add_multiple_convex_collisions_from_file` and creates one convex hull per sub-mesh in the collision file | `envs/utils/create_actor.py` line 501 |
| `create_box` returns an `Actor` whose default data includes top-down `contact_points_pose`, so `grasp_actor` works on it. `create_sphere` and `create_cylinder` return raw `sapien.Entity` objects built with `sapien.physx.PhysxRigidDynamicComponent` and `scene.default_physical_material` | `envs/utils/create_actor.py` |
| Mass on a raw entity: `entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = m` | used in `envs/dump_bin_bigbin.py` |
| Skill API: `grasp_actor`, `place_actor`, `move_by_displacement(arm_tag, x, y, z, quat, move_axis)`, `move_to_pose`, `open_gripper`, `close_gripper`, `back_to_origin` return `(arm_tag, [Action])` and are executed by `self.move(a1, a2=None)` | `envs/_base_task.py` lines 957 to 1470 |
| cuRobo plans use `interpolation_dt = 1/250` and `MotionGenPlanConfig(max_attempts=10)` with no time dilation. Results are `{"status", "position": [T, n], "velocity": [T, n]}` at one row per physics step | `envs/robot/planner.py` lines 60 to 155 |
| Segmentation: `camera.get_picture("Segmentation")` returns [H, W, 4]. Channel 1 is the actor-level id. RoboTwin's own helper casts to `uint8`, which truncates ids above 255, so read the raw channel yourself | `envs/camera/camera.py` line 377 |
| Host tasks to reuse: `put_object_cabinet` (cabinet `036_cabinet`, model id 46653, `fix_root_link=True`, drawer pulled by 4 × `move_by_displacement(y=-0.04)`), `dump_bin_bigbin` (handheld `063_tabletrashbin` ids [0, 3, 7, 8, 9, 10], 5 spheres of radius 8 mm, static `011_dustbin`), `lift_pot`, `place_empty_cup` (`021_cup` on `019_coaster`) | `envs/*.py` |

Object assets referenced by RoboTwin tasks include `001_bottle`, `002_bowl`, `003_plate`,
`008_tray`, `011_dustbin`, `015_laptop`, `019_coaster`, `021_cup`, `036_cabinet`, `039_mug`,
`060_kitchenpot`, `062_plasticbox`, `063_tabletrashbin`, `071_can`, `075_bread`,
`076_breadbasket`, `080_pill_bottle`, `086_woodenblock`, `102_roller`, `105_sauce-can`, `106_skillet`,
`110_basket`, `114_bottle`. The asset files themselves are downloaded separately
(`assets/_download.py`), so check their collision files on disk in Phase 0.

### 3.2 CEHJ_safe

| Fact | Where |
|---|---|
| `Env(task, embodiment, seed, ...)` imports `envs.{task}` from RoboTwin, calls `setup_demo(**_task_args(...))`, then `stretch_task_spawns(self)` (CEHJ obstacle logic), then `settle_arms()` | `main/envs/env.py` lines 338 to 385 |
| `Env.step(action)` runs one control tick (10 physics steps at 25 Hz) with position interpolation plus velocity feedforward. `step_dtheta` wraps it with joint displacements | `main/envs/env.py` lines 586 to 700 |
| `TickChunkedController` replaces `take_dense_action`, `left/right_move_to_pose`, and `together_move_to_pose` so every plan runs in 10-step ticks, with `tick_hook(ctx)` at each tick boundary. Plans come from `CuroboIKController.plan(arm, target_pose, **kw)`, which calls cuRobo and then screw fallbacks | `main/envs/controller.py` lines 260 to 330, 1054 to 1240 |
| `RolloutController.run()` wraps `scene.step` to count physics steps, and calls `_capture_tick()` every control tick using a float accumulator with a `1e-9` epsilon. `_capture_tick` calls `compute_h(env, ...)` | `main/train/rollout.py` lines 213 to 290, 433 to 492 |
| `StepBuffer` is a flat memmap ring buffer with `FIELDS` defined at module level | `main/train/buffer.py` |
| `AGENTS.md` documents the semantics and the gotchas (ray tracing, `_update_render` before `take_picture`, `TMPDIR` for cuRobo, child process per embodiment for long runs) | repo root |

---

## 4. Target layout

```
main/item_envs/
  __init__.py
  registry.py          # ITEM_TASKS: name -> TaskEntry(module, cls, embodiments, single_arm_ok); embodiments[0] is the default
  geometry.py          # OBB, footprints, polygons, signed distances (pure numpy)
  margins.py           # rigid consequence margins + particle-set variants (pure numpy)
  regions.py           # Region dataclass, constructors from actors/links/table (thin sim layer)
  items.py             # ItemSpec, ItemRegistry
  rules.py             # RuleSpec, candidate rule enumeration, phrase generation, split assignment
  physics.py           # friction/mass/CoM/damping randomization with readback
  primitives.py        # rimmed tray, handled board, open box, square cup, zone mat (multi-box actors)
  visibility.py        # per-item visible-pixel fraction per camera
  retime.py            # plan retiming (risk knob), pure numpy
  risk.py              # RiskParams dataclass + samplers
  controller.py        # RiskTickController(TickChunkedController): retimes every plan
  base_task.py         # ItemSafetyTask(Base_Task)
  runner.py            # ItemEpisodeRunner: env + controller + tick hook + logger, no learning models
  logger.py            # EpisodeTrace writer (npz + json) and video overlay
  margin_hook.py       # Phase 5: adapter so RolloutController can use item margins
  item_buffer.py       # Phase 5: sidecar memmap for item state, lockstep with StepBuffer
  probe.py             # CLI for feasibility probes and calibration sweeps
  tasks/
    __init__.py
    tray_transport.py      # T1
    drawer_contents.py     # T2
    support_removal.py     # T3
    rolling_edge.py        # T4
    granular_pour.py       # T5
    push_zone.py           # T6
    dump_bin.py            # T7
  configs/
    physics.yaml           # randomization ranges
    phrases.yaml           # names, synonyms, colors per asset or primitive
    splits.yaml            # held-out categories, colors, regions, phrasings per task
    tasks/T1_tray_transport.yaml ... T7_dump_bin.yaml
assets_overlay/
  objects/<asset>/collision/...      # decomposed collision meshes
scripts/item_envs/
  audit_collisions.py
  decompose_assets.py
  run_probes.sh
  run_calibration.sh
tests/item_envs/
  test_geometry.py  test_margins.py  test_retime.py  test_rules.py  test_risk.py
  test_primitives_sim.py  test_physics_sim.py  test_tasks_sim.py  test_item_buffer.py
docs/item_envs/
  recon.md  feasibility.md  calibration.md  task_cards/T1.md ... T7.md  final_report.md
```

---

## 5. Data contracts

Implement these as dataclasses in the listed modules. Field names are part of the contract with the
downstream training code, so do not rename them without updating this file.

```python
# items.py
@dataclass
class ItemSpec:
    item_id: str                 # unique within episode, e.g. "bottle_0"
    entity: object              # sapien Entity (raw) or RoboTwin Actor; use .entity_of() helper
    category: str               # e.g. "bottle"; key into phrases.yaml
    color: str | None           # for primitives and zone mats; None for textured assets
    kind: str                   # "rigid" | "particle_set"
    particles: list | None      # list of entities when kind == "particle_set"
    half_extents: np.ndarray    # [3] collision OBB half extents in the item frame (m)
    obb_center_local: np.ndarray  # [3] OBB center in the item frame
    up_axis_local: np.ndarray   # [3] unit vector, the item's "up" in its own frame
    mass: float                 # kg, the value actually set
    role: str                   # "constrained_candidate" | "task_object" | "support" | "bystander"

# regions.py
@dataclass
class Region:
    region_id: str              # e.g. "tray_top", "zone_red", "table_top", "drawer_floor"
    kind: str                   # "surface" | "zone" | "container"
    frame_fn: Callable[[], np.ndarray]   # returns 4x4 world_T_region, evaluated every tick
    polygon: np.ndarray         # [M, 2] CCW polygon in the region frame's xy plane (m)
    z: float                    # surface height in the region frame (usually 0)
    z_band: float               # vertical tolerance for "on surface" checks (m)
    phrases_key: str | None     # key into phrases.yaml, None if not nameable

# rules.py
class Consequence(str, Enum):
    INSIDE = "inside"           # item footprint stays inside region polygon (and on its surface if kind=="surface")
    OUTSIDE = "outside"         # item footprint stays outside region polygon
    UPRIGHT = "upright"         # tilt(item) <= theta_max
    STAY = "stay"               # item stays within d_tol of its pose at rule activation
    PARTICLES_INSIDE = "particles_inside"   # every particle stays in the column above one of the regions
    PARTICLES_OUTSIDE = "particles_outside" # no particle enters the column above any of the regions

@dataclass
class RuleSpec:
    rule_id: str                # stable string, e.g. "upright|bottle_0|-"
    consequence: Consequence
    item_id: str
    region_ids: tuple[str, ...] # empty for UPRIGHT and STAY
    params: dict                # {"theta_max": rad} | {"d_tol": m, "theta_tol": rad (default 0.35)} | {"T": m}
    phrases: list[str]          # natural-language variants, generated from templates
    split: str                  # "train" | "heldout_item" | "heldout_region" | "heldout_phrase"

# logger.py, one per control tick (25 Hz)
@dataclass
class StepRecord:
    t: float                    # sim seconds
    tick: int
    qpos: np.ndarray            # raw joint positions, both arms
    ee_pose: np.ndarray         # [2, 7] left/right TCP poses, wxyz
    gripper: np.ndarray         # [2]
    item_pose: np.ndarray       # [K, 7] world, wxyz
    item_vel: np.ndarray        # [K, 6] linear then angular, world
    particle_pos: dict          # item_id -> [N, 3] for particle sets
    margins: np.ndarray         # [R] raw SI margins, order = rule table order
    robot_item_contact: np.ndarray   # [K] bool, robot link or gripper touching item
    robot_item_force: np.ndarray     # [K] summed normal impulse / dt (N)
    visibility: np.ndarray      # [K, n_cam] visible pixel fraction of the item's mask
    action_source: int          # 0 planner, 1 actor, 2 perturbation (always 0 in Phases 2 to 4)
    time_scale: float           # active retime factor this tick

@dataclass
class EpisodeMeta:
    task: str; embodiment: str; seed: int
    risk: dict                  # RiskParams.as_dict()
    physics: dict               # sampled friction, mass, CoM offset, damping per item
    items: list[dict]           # ItemSpec without entity handles
    regions: list[dict]
    rules: list[dict]           # RuleSpec table; margins columns follow this order
    active_rule_id: str | None  # the rule shown to the filter in filtered runs
    task_success: bool
    violations: dict            # rule_id -> bool (min margin over episode < 0)
    min_margin: dict            # rule_id -> float
    first_violation_t: dict     # rule_id -> float | None
    plan_failures: int
    wall_time_s: float
```

Margin sign convention: **positive means safe**, zero is the failure boundary, values are raw SI and
**never clipped** in the environment. Scaling (κ per consequence) is done downstream. Record default κ
values in `configs/physics.yaml` for reference only: κ_inside = κ_outside = κ_stay = 20 (1/m),
κ_upright = 2 (1/rad), κ_particles = 20.

---

## 6. Phases

### Phase 0: Reconnaissance (about half a day)

Tasks:

1. Set up the env. Activate `RoboTwin`, check that the RoboTwin assets are downloaded, and run the
   existing CEHJ smoke script (look under `scripts/` for `run_smoke.sh`). Record the command, the
   runtime, and the final `h` trace for seed 0 on `place_empty_cup` with embodiment `ur5-wsg`, and
   again with `franka-panda`. These are the regression references.
2. Answer each question below in `docs/item_envs/recon.md` with a code snippet that you actually ran.
   - (verify) How to set a physical material on the existing collision shapes of an entity in SAPIEN
     3.0.0b1. Candidates are `component.get_collision_shapes()` combined with
     `shape.set_physical_material(mat)` or a `shape.physical_material` property. If neither works,
     document how to pass `material=` at build time (`add_box_collision(..., material=)`,
     `add_multiple_convex_collisions_from_file(..., material=)`).
   - (verify) Linear and angular damping on `PhysxRigidDynamicComponent`, for example the
     `linear_damping` and `angular_damping` properties.
   - (verify) Setting the center-of-mass local pose, for example `cmass_local_pose`.
   - (verify) Continuous collision detection on a dynamic body, if exposed.
   - (verify) Kinematic bodies, for example `component.kinematic = True` plus
     `set_kinematic_target(pose)`. The physics probes in Phase 3 use kinematic supports.
   - (verify) Solver iteration counts (scene-level or per-body).
   - (verify) Reading contact impulses between two specific entities from `scene.get_contacts()`.
     CEHJ's `main/envs/distance.py` (`_robot_obstacle_contact_force`) shows the existing pattern.
   - (verify) Actor-level segmentation ids. How to map an entity to its `per_scene_id`, and which
     cameras exist (head plus wrist cameras, at which resolution).
   - Does `Env._task_args` or RoboTwin use `task_name` for anything that breaks with unknown task
     names (instruction files, step-limit files, data paths)?
   - For `ur5-wsg` and `franka-panda`: the arm base poses in the paired setup (`arm_distance`, default
     0.6 m in `Env`), arm DoF, URDF joint velocity limits, the gripper's maximum opening and finger
     length, and the reachable tabletop workspace. Compute the workspace by an IK sweep over a
     2 cm grid on the table at three heights (table + 0.02, + 0.10, + 0.20 m) with a top-down gripper,
     and save it as `configs/workspace_<embodiment>.npz`. Task layouts are sampled inside the
     intersection of both embodiments' workspaces (Section 7).
   - What is the current working directory during `setup_demo`? RoboTwin uses relative
     `assets/objects` paths, and `AGENTS.md` notes that RoboTwin changes directory during setup.
   - For every asset in Section 7's candidate pools: number of convex collision pieces, collision
     file path, and whether `model_data*.json` provides `contact_points_pose`.
3. Write `docs/item_envs/recon.md`.

Acceptance: `recon.md` answers every question, and the regression reference is saved under
`docs/item_envs/regression/`.

### Phase 1: Pure-numpy core (about 1 day)

Implement `geometry.py`, `margins.py`, `retime.py`, `rules.py`, `risk.py` and their tests. None of
these may import SAPIEN.

**geometry.py**

```python
def quat_wxyz_to_mat(q) -> np.ndarray                    # [3, 3]
def pose_to_mat(p, q) -> np.ndarray                      # [4, 4]
def obb_corners(world_T_item, center_local, half) -> np.ndarray   # [8, 3]
def footprint(world_T_item, center_local, half, up_local) -> np.ndarray
    # the 4 corners of the OBB face most aligned with -world_z (the face the item stands on);
    # when the item is tilted, still return the lowest face's 4 corners
def signed_dist_inside_polygon(pts_xy, poly) -> np.ndarray   # >0 inside, <0 outside, metres to nearest edge
def tilt_angle(world_T_item, up_local) -> float          # angle between item up and world +z, radians
def to_frame(points_world, world_T_frame) -> np.ndarray
```

**margins.py**, with every function returning a float and positive meaning safe:

```python
def m_inside(item_pts_region_frame, region, require_on_surface: bool) -> float
    # min over footprint corners of signed_dist_inside_polygon.
    # if require_on_surface and min corner z deviates from region.z by more than region.z_band:
    #     return min(xy_margin, -(|dz| - z_band))   (continuous, not a big constant)
def m_outside(item_pts_region_frame, region) -> float    # -max over corners of signed_dist_inside
def m_upright(tilt, theta_max) -> float                  # theta_max - tilt
def m_stay(p_now, p_ref, tilt_now, d_tol, theta_tol) -> float
    # min(d_tol - ||p_now_xy - p_ref_xy||, theta_tol - tilt_now)
def m_particles_inside(particle_xy_by_region_frame: list[np.ndarray], regions, T=0.005) -> float
    # per particle: max over regions of signed_dist_inside_polygon (column semantics)
    # then soft-min over particles: -T * logsumexp(-m_b / T)
def m_particles_outside(particle_xy_by_region_frame: list[np.ndarray], regions, T=0.005) -> float
    # per particle: min over regions of -signed_dist_inside_polygon, then the same soft-min
def softmin(x, T) -> float
```

**retime.py**

```python
def retime(position: np.ndarray, velocity: np.ndarray, s: float, dt: float = 1/250,
           v_limit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Uniform time scaling. s > 1 is faster.
    New sample times t_k = k * s * dt for k = 0..K, with K = floor((T-1)/s), plus the exact final row.
    position' = per-joint linear interpolation of position at t_k / dt.
    velocity' = s * interpolated velocity.
    If v_limit is given and max |velocity'| / v_limit > 1, do NOT clip. Return info["v_ratio"]
    so callers can log or reject. info also holds {"s": s, "T_in": T, "T_out": K + 1}."""
```

**risk.py**

```python
@dataclass
class RiskParams:
    time_scale: float = 1.0               # applied to every plan in "risky" segments
    safe_time_scale: float = 0.6          # applied to approach / grasp / retreat segments
    task: dict = field(default_factory=dict)   # task-specific knobs, see Section 7
    def as_dict(self) -> dict: ...
def sample_risk(task_cfg: dict, rng: np.random.RandomState) -> RiskParams
    # task_cfg["risk"] lists each knob as {"dist": "uniform"|"choice", "range"/"values": ...}
```

**rules.py**

```python
def enumerate_rules(items, regions, task_cfg) -> list[RuleSpec]
    # task_cfg["rules"] lists templates such as
    #   {"consequence": "upright", "items": {"role": "constrained_candidate", "tippable": true}, "theta_max_deg": 45}
    #   {"consequence": "inside",  "items": {"role": "constrained_candidate"}, "regions": ["tray_top"]}
    # Produce one RuleSpec per (template, matching item, matching region).
def render_phrases(rule, phrases_cfg, rng, n=3) -> list[str]
    # Templates per consequence, for example
    #   inside:  "keep the {item} on the {region}", "don't let the {item} fall off the {region}"
    #   outside: "keep the {item} out of the {region}", "don't let the {item} get into the {region}"
    #   upright: "don't knock over the {item}", "keep the {item} upright"
    #   stay:    "don't disturb the {item}", "leave the {item} where it is"
    #   particles_inside: "don't spill the {item}", "keep the {item} in the {region}"
    #   particles_outside: "keep the {item} off the {region}", "don't get any {item} on the {region}"
    # {item} is drawn from the item's names in phrases.yaml (category, synonyms, optional color prefix).
    # Never put numbers into phrases. theta_max and d_tol stay in RuleSpec.params.
def assign_split(rule, splits_cfg) -> str
```

**Tests** (`tests/item_envs/test_*.py`):

- polygon signed distance on a unit square: center 0.5, edge 0, outside point −0.2.
- footprint of an upright box equals its bottom face. At 90° tilt, it is the side face now facing
  down.
- `m_inside` is continuous across the polygon edge (finite differences at ±1e-4 m), and continuous
  when the item lifts off the surface by `z_band ± 1e-4`.
- `m_upright` equals 0 exactly at `tilt == theta_max`.
- `m_particles_inside` is at most the min of the per-particle margins (soft-min is conservative), and
  within `T * log(N)` of it.
- `retime` with s = 1 returns the input exactly. Endpoints are preserved for s in
  {0.3, 0.7, 1.3, 2.0}. Duration scales as 1/s within one row. Velocity scales by s.
- `enumerate_rules` on a toy scene gives the expected count and stable `rule_id` strings.
- `render_phrases` never emits digits.

Acceptance: `pytest tests/item_envs -m "not sim and not gpu"` passes.

### Phase 2: Simulation infrastructure (about 2 days)

**2.1 `physics.py`**

```python
def apply_material(entity, static_f, dynamic_f, restitution, scene) -> None
def set_mass_props(entity, mass, com_offset_local=None) -> None
def set_damping(entity, linear=None, angular=None) -> None
def sample_physics(item_specs, physics_cfg, rng) -> dict      # per item_id
def apply_physics(task, sampled: dict) -> None
def read_back(entity) -> dict          # actual values from SAPIEN, for tests and logging
```

Rules for sampling: draw static friction first, then dynamic friction ≤ static. Default ranges in
`physics.yaml`: static U(0.3, 0.9), dynamic U(0.2, static), restitution U(0, 0.1), mass scale
U(0.7, 1.3) around the per-category nominal mass, CoM offset U(−0.2, 0.2) × half extents in x and y
and U(−0.1, 0.3) × half height in z (higher CoM makes tipping more likely). Angular damping only for
items with `rolls: true`.

Test (`sim`): apply, then read back, for a box and a sphere.

**2.2 `primitives.py`**

All multi-part containers are built from **box collision shapes on one entity**. Every part is convex,
so there are no decomposition issues. Each builder returns a RoboTwin `Actor` with a hand-written
`data` dict containing `contact_points_pose` for its handle (copy the format from `create_box`'s
default data), so `grasp_actor` works.

```python
def rimmed_tray(scene, pose, size_xy=(0.30, 0.22), thickness=0.012, rim_h=0.02, rim_t=0.006,
                handles="both_short_sides" | "one_short_side", color=..., mass=0.25) -> Actor
def handled_board(scene, pose, size_xy=(0.20, 0.14), thickness=0.02, handle="short_side", color=..., mass=0.2) -> Actor
def open_box(scene, pose, inner=(0.10, 0.10, 0.08), wall_t=0.005, color=..., mass=0.1, is_static=False) -> Actor
def square_cup(scene, pose, inner=(0.06, 0.06, 0.08), wall_t=0.004, color=..., mass=0.06) -> Actor
def zone_mat(scene, pose, size_xy, color_name, thickness=0.002) -> Actor   # static, visual + thin collision
```

Expose `interior_polygon()` and `top_surface_z()` for each container, for region construction.

Test (`sim`): each primitive survives `check_stable`, a sphere dropped into each container stays
inside after 2 s, and `grasp_actor` produces a successful plan on both `ur5-wsg` and `franka-panda`
(`gpu`).

**2.3 `regions.py`**

```python
def region_from_actor_top(actor, region_id, polygon_local, z_local, z_band, kind, phrases_key) -> Region
def region_from_link(articulation_actor, link_name, ...) -> Region      # drawer floor follows the link
def table_region(task) -> Region     # 1.2 x 0.7 polygon at the table top, uses table_xy_bias and table_z_bias
def zone_region(mat_actor, color_name) -> Region
```

**2.4 `visibility.py`**

```python
def visible_fraction(task, item_specs, cameras=("head_camera", "left_camera", "right_camera")) -> np.ndarray  # [K, n_cam]
```

Use the raw actor-level segmentation id (no `uint8` cast). The fraction is visible pixels divided by
pixels of the item's OBB projected into the image (a cheap upper bound on its area). Call
`task._update_render()` before taking pictures. It is too expensive to run every tick, so compute it
every N ticks (default 5) and forward-fill.

**2.5 `base_task.py`**

```python
class ItemSafetyTask(Base_Task):
    TASK_KEY: str                         # registry key, also the config file stem
    def setup_demo(self, risk: dict | None = None, physics_seed: int | None = None,
                   active_rule_id: str | None = None, task_cfg_path: str | None = None, **kw):
        self.task_cfg = load_task_cfg(self.TASK_KEY, task_cfg_path)
        self.items: dict[str, ItemSpec] = {}
        self.regions: dict[str, Region] = {}
        self.unstable_ok: set[str] = set()     # entity names excluded from check_stable
        self._rng_phys = np.random.RandomState((kw.get("seed", 0) + 7919) if physics_seed is None else physics_seed)
        self.risk = RiskParams(**risk) if risk else sample_risk(self.task_cfg, self._rng_phys)
        self.active_time_scale = self.risk.safe_time_scale
        super()._init_task_env_(**kw)          # calls load_actors and check_stable
        self._after_settle()                   # randomized damping, record reference poses, enumerate rules

    # subclass hooks
    def build_scene(self) -> None: ...           # create actors, call self.add_item / self.add_region
    def nominal(self) -> None: ...               # the stage machine using self.move(...)
    def task_success(self) -> bool: ...

    # provided
    def load_actors(self):
        self.build_scene()
        self.add_region(table_region(self))
        self.physics_sample = sample_physics(self.items, self.phys_cfg, self._rng_phys)
        apply_physics(self, self.physics_sample, stage="pre_settle")   # friction, mass, CoM
    def check_stable(self):
        # same algorithm as Base_Task.check_stable, but skip entities whose name is in self.unstable_ok.
        # Rolling items keep high damping during settle (see T4) and are not exempt by default.
    def _after_settle(self):
        apply_physics(self, self.physics_sample, stage="post_settle")  # rolling damping
        self.ref_pose = {k: pose(item) for k, item in self.items.items()}
        self.rules = enumerate_rules(self.items, self.regions, self.task_cfg)
    def risky(self):            # context manager, sets active_time_scale = risk.time_scale
    def play_once(self):
        self.nominal(); return self.info
    def check_success(self):
        return self.task_success()
    def privileged_state(self) -> dict
    def margins(self) -> np.ndarray       # [R] in self.rules order
```

`nominal()` wraps the segments the risk knob should affect in `with self.risky():`. Approach, grasp,
and retreat stay at `safe_time_scale`, so plan failures and grasp failures do not pollute the risk
signal.

**2.6 `controller.py`**

```python
class RiskTickController(TickChunkedController):
    def plan(self, arm, target_pose, **kw):
        result = super().plan(arm, target_pose, **kw)
        if result and result.get("status") == "Success":
            s = float(getattr(self.env.task, "active_time_scale", 1.0))
            pos, vel, info = retime(result["position"], result["velocity"], s,
                                    v_limit=self._v_limit(arm))
            result["position"], result["velocity"] = pos, vel
            result["_item_retime"] = info
        return result
```

`_v_limit` reads the URDF joint velocity limits (reuse CEHJ's `BodyTokenExtractor` or the
embodiment's URDF). Log `info["v_ratio"]`. If it exceeds 1.0, record the tick-level realized vs
commanded joint displacement, as `rollout.py` already does (`mean_realized_ratio`). Do not clip
silently.

**2.7 Env integration (the only edits to existing files before Phase 5)**

In `main/envs/env.py`, add a guarded branch in `Env.__init__`:

```python
from main.item_envs.registry import ITEM_TASKS
if task in ITEM_TASKS:
    entry = ITEM_TASKS[task]
    module = importlib.import_module(entry.module)
    self.task = getattr(module, entry.cls)()
    self.task.setup_demo(**_task_args(task, self.embodiment, seed, arm_distance, cluttered=False),
                         **(item_kwargs or {}))
    # no stretch_task_spawns for item tasks
else:
    ...existing code unchanged...
```

Add an `item_kwargs: dict | None = None` parameter to `Env.__init__` with default `None`. Existing call
sites must not change.

**2.8 `runner.py` and `logger.py`**

`ItemEpisodeRunner(task, embodiment, seed, risk=None, active_rule_id=None, video=False)`:

- builds `Env` with `item_kwargs`, attaches `RiskTickController`, and wraps `scene.step` with the
  same tick accumulator as `RolloutController.run()` (keep the `1e-9` epsilon);
- at every tick, writes a `StepRecord`, with margins from `task.margins()`, contacts, visibility
  every N ticks, and the active time scale;
- at the end, writes `EpisodeMeta`, `trace.npz`, `meta.json`, and an optional mp4 from the observer
  camera with an overlay showing the active rule's phrase, its current margin, the min margin over
  all rules, the time scale, and a red flash on the first violation;
- catches plan failures and `UnStableError`, records them in `meta.json`, and never crashes the
  sweep.

Output layout: `runs/item_envs/<task>/<embodiment>/seed_<k>/{meta.json, trace.npz, video.mp4}`.

**2.9 `probe.py` CLI**

```
python -m main.item_envs.probe episode  --task tray_transport --seed 0 --risk time_scale=1.0 --video
python -m main.item_envs.probe sweep    --task tray_transport --grid time_scale=0.5,0.75,1.0,1.25 \
                                         --episodes 20 --out runs/item_envs/calib/T1
python -m main.item_envs.probe physics  --probe P1 --out runs/item_envs/feas/P1
python -m main.item_envs.probe summarize runs/item_envs/calib/T1     # writes CSV + plots
```

Long sweeps follow the `AGENTS.md` pattern: one child process per (task, embodiment) so that
repeated env creation does not exhaust the pybind11 TLS keys.

**Phase 2 acceptance:** a throwaway `smoke_items` task (registered only in tests) with one tray, two
items on it, and one zone mat runs end to end with `ItemEpisodeRunner`, produces `trace.npz`,
`meta.json`, and a video, and its margins are visibly correct in the overlay. All `sim` tests pass.

### Phase 3: Feasibility probes (1 to 2 days). STOP gate.

Probes P1 to P6 use **kinematic** supports or pushers (no robot), so physics is isolated from the
planner. Then P1 and P4 are repeated with the robot. Write all results to
`docs/item_envs/feasibility.md`, with plots and three example videos per probe.

| Probe | Setup | Sweep | Measure | Gate |
|---|---|---|---|---|
| P1 tray | kinematic rimmed tray carrying (a) block 6 cm cube, (b) cylinder "bottle" r 3.5 cm, h 20 cm, (c) `021_cup` | trapezoid velocity profile, v_max 0.5 m/s, peak accel a in {0.5, 1, 2, 4, 8} m/s², 3 seeds, friction fixed at 0.5 | slip distance in tray frame, max tilt, margins | block sliding onset within ±30% of μ_s·g. Bottle tipping onset within ±30% of g·r/h_com. Same params repeat within ±10% across seeds |
| P1-robot | robot carries the empty tray at time_scale s in {0.5, 0.75, 1.0, 1.25}, on `ur5-wsg` and on `franka-panda` | as left | peak tray acceleration achieved, tray rotation in gripper | there exists s with a_peak ≥ bottle tipping threshold and a_peak < block sliding threshold. Tray rotates less than 5° in the grip |
| P2 support removal | kinematic board under a mug and under a cylinder | pull speed v in {0.05, 0.2, 0.5, 1, 2} m/s, accel 20 m/s² | item displacement, tilt | displacement varies with v by at least 3 cm across the grid. Report whether it is non-monotonic (no gate on that) |
| P3 rolling | sphere r 2 cm and cylinder lying on its side, initial speed 0.3 m/s | angular damping in {0, 0.05, 0.1, 0.2, 0.5, 1} | stopping distance, straightness | some damping range gives stopping distance in 0.1 to 0.5 m. Deliver a damping-to-distance lookup table |
| P4 drawer | `036_cabinet` drawer with a sphere, a lying cylinder, and an upright cylinder inside | drawer opened by the robot at s in {0.3, 0.6, 1.0, 1.3}, on both embodiments | items inside, tilt | drawer interior is hollow (items rest on the drawer floor, not on top of the cabinet). At s = 0.3, items stay inside in ≥ 90% of seeds. Failures appear at higher s |
| P5 beads | `square_cup` with N in {30, 60, 100, 150} spheres, r in {5, 8} mm | kinematic tilt to 120° at ω in {1, 3} rad/s | wall time of physics per control tick (10 substeps), beads tunneled through walls or table, spill count | at N = 60, physics ≤ 20 ms per tick (report CPU model). Zero tunneling. Spill count increases with ω |
| P6 push | kinematic pusher pushes a box at v in {0.1, 0.3, 0.6} m/s for 0.15 m, then stops | friction in {0.3, 0.6, 0.9} | coasting distance after the pusher stops, box rotation | coasting distance > 0, increases with v, decreases with friction |
| P7 container audit | `scripts/item_envs/audit_collisions.py` over every asset in the Section 7 pools | | number of convex pieces, hull-to-mesh volume ratio | every concave container used as a container either has ≥ 4 pieces, or a CoACD-decomposed overlay in `assets_overlay/` that passes P4 or P5 style containment tests |
| P8 visibility | initial layout of each task, 20 seeds | | visible fraction per constrained item per camera | every constrained candidate item has visible fraction ≥ 0.3 in at least one camera at t = 0 in ≥ 95% of seeds |

If a gate fails, try at most two documented fixes (for example raising solver iterations, enabling
CCD, or decomposing a mesh). If the gate still fails, **stop and report**.

### Phase 4: Tasks T1 to T7 (about 1 to 1.5 days per task)

Build in this order: T1, T3, T4, T2, T6, T7, T5. For each task:

1. implement `tasks/<file>.py` and `configs/tasks/<file>.yaml`,
2. write the task card `docs/item_envs/task_cards/Tn.md` (layout figure, item pools, rules, knobs,
   known issues),
3. add `sim` tests: the scene builds for 20 seeds without `UnStableError`, rules enumerate, the
   margins at t = 0 are all positive, and one nominal episode runs to completion at low risk,
4. run calibration (Section 8) and add the results to `calibration.md`.

Task-specific requirements are in Section 7.

### Phase 5: Integration with CEHJ collection (1 to 2 days)

Allowed edits to existing files:

1. `main/train/rollout.py`: add an optional constructor argument `h_fn=None`. In `_capture_tick`,
   replace the direct call with `h, diag = (self.h_fn or default_h_fn)(self.env, self.cfg)`, where
   `default_h_fn` wraps the existing `compute_h` call byte for byte. Also add an optional
   `controller_cls` argument (default `TickChunkedController`) so item tasks can use
   `RiskTickController`.
2. `main/train/collect.py`, `make_training_env`: pass `item_kwargs` through when `cfg.task` is an item
   task.

New code:

- `margin_hook.py`: `ItemMarginHFn(active_rule_id, kappa)` callable returning
  `(kappa * margin_active, diag)`, where `diag` contains all margins, item state, and fields that
  keep the existing video and metrics code from crashing (fill `d_left`, `per_link`, etc. with
  `inf` or empty values).
- `item_buffer.py`: `ItemSidecarBuffer(root, capacity, k_max=8, r_max=48, n_cam=3)`, a memmap with
  the same ring semantics as `StepBuffer`, written in lockstep (same index) with fields

  | field | dtype | shape |
  |---|---|---|
  | `item_pose` | float32 | [k_max, 7] |
  | `item_vel` | float32 | [k_max, 6] |
  | `item_obb` | float32 | [k_max, 9] (center_local 3, half 3, up_local 3) |
  | `item_valid` | bool | [k_max] |
  | `margins` | float32 | [r_max] |
  | `rule_valid` | bool | [r_max] |
  | `visibility` | float16 | [k_max, n_cam] |
  | `region_T` | float32 | [r_max_regions = 8, 4, 4] |

  The rule table, item table, and region polygons of each episode go in a JSON-lines file keyed by
  `episode_id`. The buffer stores no language. Phrases are regenerated from the rule table at
  training time.
- Test: after a ring wrap, `ItemSidecarBuffer[i]` and `StepBuffer[i]` refer to the same transition
  (same `episode_id`, same tick).
- Dry run (`gpu`): 3 nominal-only episodes per task into a fresh buffer with the HoloBrain encoder
  running. A readback script compares buffer margins with `trace.npz` from the logger for the same
  seed. The difference must be 0.
- Regression: the old-task `h` trace from Phase 0 must be identical.

Do not touch `train.py`, the networks, or the buffer `FIELDS`.

---

## 7. Task specifications

Common to all tasks:

- **Embodiments.** Default `ur5-wsg` (it worked best in the previous CrossSafe experiments, and
  HoloBrain-GD was trained on RoboTwin data that includes `robotwin2_0_ur5_wsg`). Every task must also
  run on `franka-panda`, the most widely used arm. Both are single arms that RoboTwin instantiates as
  a pair (`dual_arm: False` in their config, so `_task_args` passes `[e, e, arm_distance]`). Rules
  that follow from this:
  - **One code path for both.** No `if embodiment == ...` branches inside task logic. Embodiment
    differences live in data: per-embodiment entries in the task YAML (`embodiment_overrides:`) for
    grasp offsets, pre-grasp distances, and layout shifts, and the workspace files from Phase 0.
  - **Joint count differs.** UR5 has 6 arm joints and Franka has 7. Never hardcode action widths.
    Read them from the robot (`get_left_arm_real_jointState()` length minus the gripper) the way
    `Env.step_dtheta` does. CEHJ's buffer already pads to 18.
  - **Grippers differ.** Every handle, rim, and graspable primitive must fit the smaller maximum
    opening and the shorter finger length of the two grippers (from Phase 0). Grasp contact points
    in primitive `data` dicts must work for both, verified by the Phase 2 grasp test.
  - **Shared workspace.** Sample every layout inside the intersection of the two embodiments'
    reachable workspaces, so the same seed produces the same scene on both arms. Reject and resample
    any layout where a key grasp or place pose fails IK on either embodiment, and log the rejection
    rate.
  - **Single-arm variant.** Every task has a config flag `single_arm: true` that uses only one arm of
    the pair (the other stays at home). The later hardware target is a small single arm, so the
    single-arm variant is the one calibrated first. Bimanual variants (T1 tray with two handles) are
    optional extras.
  - **Tests and reports per embodiment.** Every `sim` and `gpu` test in Phase 4 is parametrized over
    `["ur5-wsg", "franka-panda"]`, and every report table has one row per embodiment.
- **Candidate items.** Every scene has at least two constrained-candidate items with **different
  vulnerability** to the task's main failure mode (for example a tall bottle and a low block on the
  same tray). This is what makes naming the item matter.
- **Task success** is defined only by the nominal's goal. Safety outcomes are recorded separately per
  rule. Do not fold safety into `check_success`.
- **Nominal masses** per category go in `phrases.yaml` alongside the names. Defaults: bottle
  0.05 to 0.30 kg, cup or mug 0.05 to 0.25, can 0.05 to 0.35, wooden block 0.05 to 0.15, bread 0.03 to
  0.08, ping-pong ball 0.0027, bead 0.0005 to 0.002.

### T1 `tray_transport` (support friction)

- **Scene.** `rimmed_tray` with a handle on one short side (single-arm, default, 0.20 × 0.15 m tray)
  or handles on both short sides (bimanual variant, both arms of the pair, 0.30 × 0.22 m tray). 2 to 4 items on the tray, at least 1.5 cm from the rims and from
  each other. Item pools:
  - tippable: `001_bottle`, `114_bottle`, `021_cup`, `039_mug`, primitive cylinder with h/r ≥ 3;
  - stable: `086_woodenblock`, `075_bread`, low primitive box.
  Use `asset_overlay` collision for any asset whose audit (P7) failed.
- **Nominal.** Grasp the handle(s), lift 0.10 m, then `with self.risky():` move to a goal offset
  (dx ∈ [0.15, 0.35] m, dy ∈ [−0.10, 0.10] m), lower, release, and retreat.
- **Risk knobs.** `time_scale` ∈ [0.5, 1.3]. `task.stop_abruptly` ∈ {0, 1}: if 1, split the transport
  into two plans with a full stop between them, which adds a braking phase.
- **Regions.** `tray_top` (interior polygon in the tray frame, `kind="surface"`, `z_band=0.01`),
  `table_top`.
- **Rules.** `inside(item, tray_top)` for every item. `upright(item, θ_max=45°)` for tippable items.
- **Success.** Tray center within 3 cm of the goal, and the tray is released.
- **Design check.** Pick item pairs whose analytic thresholds differ, using a_slide = μ_s·g and
  a_tip = g·(half-width / CoM height), so that one item fails and the other survives over a sizable
  band of `time_scale`. Log both thresholds in `meta.json`.

### T2 `drawer_contents` (articulation)

- **Scene.** Reuse `put_object_cabinet`'s cabinet (`036_cabinet`, model id 46653, `fix_root_link=True`).
  The drawer starts **open 6 to 10 cm** (set the joint qpos) so that contents are visible. Place 1 to 3
  items on the drawer floor: an upright primitive cylinder (tippable), a can lying on its side
  (`rolls: true`), a wooden block (stable), and a sphere.
- **Fallback.** If P4 showed the drawer interior is not hollow, build an `open_box` on a prismatic
  articulation instead (document the builder in the task card).
- **Nominal.** Grasp the drawer handle as in `put_object_cabinet`, then `with self.risky():` pull
  open to a total extension of 0.15 to 0.25 m. Optionally, push it closed again (knob).
- **Risk knobs.** `time_scale` ∈ [0.3, 1.3]. `task.n_pull_steps` ∈ {1, 2, 4} (fewer, larger steps
  are jerkier). `task.close_after` ∈ {0, 1}. Randomize drawer joint damping and friction via
  `ArticulationActor.set_properties`.
- **Regions.** `drawer_floor`, using `region_from_link`, so the region follows the drawer link every
  tick.
- **Rules.** `inside(item, drawer_floor)`, `upright(item)`.
- **Success.** Drawer extension within 2 cm of the target.

### T3 `support_removal` (chained support)

- **Variant a, `drag_with`.** A `handled_board` with a mug, a cup, or a tall cylinder on top. The
  nominal grasps the handle and drags the board 0.15 to 0.25 m. Failure means the item falls off the
  board. Rules: `inside(item, board_top)`, `upright(item)`.
- **Variant b, `extract_from_under`.** The same board, but the goal is to pull the board out from under
  the item, leaving the item standing on the table. Rules: `upright(item)` and
  `stay(item, d_tol=0.05)`.
- **Variant c, `tower`.** Three stacked blocks (`086_woodenblock` or primitive boxes). The nominal
  pulls the middle block sideways out of the stack (as in Jenga). Rule: `stay(top_block, d_tol=0.03)`.
- **Risk knobs.** `time_scale` ∈ [0.3, 2.0] with a wide grid, because the effect may be
  non-monotonic. `task.single_yank` ∈ {0, 1} (one long plan instead of several short ones).
- **Success.** Board displaced by at least 90% of the commanded distance.

### T4 `rolling_edge` (rolling, point of no return)

- **Scene.** A rolling item R (primitive sphere r ∈ [0.02, 0.03] m, primitive cylinder on its side,
  or `102_roller`) placed 0.05 to 0.15 m from the table edge or from a `zone_mat`, and 0.03 to 0.10 m
  from the nominal's path. A task object X (a box) that the nominal must place or push near R, and one
  stable bystander.
- **Settle.** During `check_stable`, give R angular damping 50 and linear damping 5. Switch to the
  sampled damping (from the P3 lookup table) in `_after_settle`. Do not add R to `unstable_ok`.
- **Nominal.** Two sub-variants. (a) Pick X and place it at a target offset toward R by
  `task.target_offset` ∈ [0, 0.06] m. (b) Push X with the push primitive from T6 toward R.
- **Risk knobs.** `time_scale` ∈ [0.5, 1.3], `task.target_offset`.
- **Regions.** `table_top`, an optional `zone_<color>`.
- **Rules.** `inside(R, table_top)`, `outside(R, zone)`. Also `upright(bystander)` for chained contact.
- **Success.** X within 2 cm of its target.

### T5 `granular_pour` (granular flow, liquid stand-in)

- **Scene.** A `square_cup` (or `021_cup` with an overlay collision that passed P5) holding N beads
  (N from P5, default 60, radius 6 mm), a target `open_box` or decomposed `002_bowl`, and a forbidden
  region (`zone_mat` or a static `015_laptop`).
- **Particle item.** The beads are a single `ItemSpec` with `kind="particle_set"` and category
  `"beans"` (synonyms "beads", "rice", "grains"). Its OBB is the axis-aligned bounding box of the
  particles, recomputed every tick for logging.
- **Nominal.** Grasp the cup, lift, then `with self.risky():` move above the target at height
  `task.pour_height` ∈ [0.05, 0.20] m above the rim with lateral offset `task.pour_offset` ∈
  [0, 0.05] m, tilt by 100° to 120° at rate `task.tilt_rate` ∈ [1, 4] rad/s (implement as an EE
  rotation via `move_to_pose` with retiming), hold 1 s, return upright, place the cup back.
- **Rules.** `particles_inside(beans, [cup_interior, target_interior])` ("don't spill the beans"),
  and `particles_outside(beans, [forbidden_region])` ("keep the beans off the laptop").
- **Extra log fields.** Spill count (beads below table height + 2r that lie outside every container
  footprint), and per-tick physics wall time.
- **Success.** Fraction of beads inside the target ≥ 0.8.

### T6 `push_zone` (planar sliding and chained contact)

- **Scene.** A push object P (primitive boxes of varied size and color, `048_stapler`, `113_coffee-box`),
  a goal G 0.05 to 0.15 m from the table edge or next to a `zone_mat`, and a tippable bystander Y
  0.02 to 0.08 m off the push line.
- **Push primitive** (put it in `base_task.py` for reuse in T4): close the gripper, move to a pre-push
  pose 0.05 m behind P at P's CoM height (at least 1 cm above the table), then `with self.risky():`
  `move_by_displacement` along the push direction by `dist(P, G) + task.overshoot`, and retreat. The
  planner's world model contains only the table, so plans through P are expected. That is the point of
  a push.
- **Risk knobs.** `time_scale` ∈ [0.5, 1.3], `task.overshoot` ∈ [0, 0.08] m,
  `task.lateral_offset` ∈ [−0.02, 0.02] m (off-center contact induces rotation).
- **Rules.** `inside(P, table_top)`, `outside(P, zone)`, `upright(Y)`, `stay(Y, d_tol=0.03)`.
- **Success.** P within 3 cm of G.

### T7 `dump_bin` (many rigid items)

- **Scene.** Follow `dump_bin_bigbin`. A handheld `063_tabletrashbin` (ids [0, 3, 7, 8, 9, 10], use
  overlay collisions if needed) or an `open_box`, with 3 to 6 small contents (spheres of radius 8 mm,
  small blocks, short cylinders) with **realistic masses** (5 to 30 g, not the host's 1e-4 kg). A
  static target `011_dustbin` or a static `open_box`, and a `zone_mat` or `015_laptop` next to it.
- **Nominal.** Same structure as the host (grasp, lift, move over the target, tilt to dump, return).
  Risky segment: the move and the tilt.
- **Risk knobs.** `time_scale` ∈ [0.5, 1.3], `task.target_miss` ∈ [0, 0.08] m,
  `task.tilt_rate` ∈ [1, 4] rad/s.
- **Rules.** `particles_inside(contents, [handheld_bin_interior, target_interior])`, and
  `outside(content_k, zone)` per content item (each item is also a separate rigid `ItemSpec` so it
  can be named individually, for example "the red ball").
- **Success.** All contents inside the target bin.

### Deferred, do not build

Swinging items (hanging bag or mug) and deformables (cloth, bags). Note them in `final_report.md`
under future work.

---

## 8. Calibration protocol (per task)

1. Grid: the task's `time_scale` values (5 points) × its main task knob (3 points) × 20 episodes, with
   physics randomization on. That is 300 nominal-only episodes per task and embodiment, run for
   `ur5-wsg` and `franka-panda`. The gates in step 3 must hold on both. If they hold only on one,
   report it per embodiment.
2. For each rule template, compute the violation rate (min margin over the episode < 0), the
   distribution of min margins, and the time of first violation.
3. Pick a **collection risk distribution** (a mixture over the grid) such that
   - the primary rule's violation rate is between 0.2 and 0.5,
   - at least two candidate items in the same scene distribution differ in violation rate by ≥ 0.15
     (differential vulnerability),
   - nominal task success is ≥ 0.7 at the lowest risk setting (the planner itself works).
4. Write the chosen distribution into `configs/tasks/<task>.yaml` under `risk_collect`, and the
   evidence (tables, plots) into `calibration.md`.

If a task cannot meet these numbers after reasonable redesign of layout or item pools (document what
you tried), report it rather than lowering the bar.

---

## 9. Configuration examples

`configs/tasks/T1_tray_transport.yaml`

```yaml
task: tray_transport
embodiments: [ur5-wsg, franka-panda]   # first entry is the default
single_arm: true
tray: {size_xy: [0.20, 0.15], rim_h: 0.02, handles: one_short_side, mass: [0.15, 0.30]}
bimanual_tray: {size_xy: [0.30, 0.22], rim_h: 0.02, handles: both_short_sides, mass: [0.20, 0.35]}
embodiment_overrides:                  # data only, no branches in task code
  ur5-wsg:      {pre_grasp_dis: 0.08, grasp_z_offset: 0.0}
  franka-panda: {pre_grasp_dis: 0.08, grasp_z_offset: 0.0}
items:
  n: [2, 4]
  pools:
    tippable: [001_bottle, 114_bottle, 021_cup, 039_mug, prim_tall_cylinder]
    stable: [086_woodenblock, 075_bread, prim_low_box]
  min_tippable: 1
  min_stable: 1
  edge_clearance: 0.015
goal: {dx: [0.15, 0.35], dy: [-0.10, 0.10]}
risk:
  time_scale: {dist: choice, values: [0.5, 0.75, 1.0, 1.15, 1.3]}
  safe_time_scale: 0.6
  task:
    stop_abruptly: {dist: choice, values: [0, 1]}
rules:
  - {consequence: inside, items: {role: constrained_candidate}, regions: [tray_top]}
  - {consequence: upright, items: {role: constrained_candidate, tippable: true}, theta_max_deg: 45}
risk_collect: null        # filled by calibration
```

`configs/splits.yaml` (excerpt)

```yaml
tray_transport:
  heldout_item_categories: [mug, sauce_can, pill_bottle]
  heldout_colors: [purple, orange, cyan]
  heldout_regions: []
  heldout_phrase_templates: ["don't let the {item} slide off the {region}"]
```

---

## 10. Final report

`docs/item_envs/final_report.md` contains:

- a table per task with build status, P-gate results, calibration numbers (violation rates per rule,
  differential vulnerability, nominal success, mean wall time per episode, physics ms per tick), and
  links to three videos (safe, near-miss, violation);
- known issues and their workarounds;
- every deviation from this plan, with the reason.

---

## 11. Out of scope, and decisions reserved for the human

Out of scope: grounding labels (only log OBBs and region transforms, labels are computed downstream
from scene-token positions), language models, encoder changes, critic or actor changes, training
scripts, SO-101 hardware.

Reserved decisions. Ask before changing any of these:

1. The embodiment set (`ur5-wsg` default, `franka-panda` supported). Adding or dropping an embodiment
   needs approval.
2. Final held-out category, color, and phrase lists in `splits.yaml`.
3. Whether any task is dropped after a failed STOP gate.
4. Any change to the consequence types or the margin sign convention.
