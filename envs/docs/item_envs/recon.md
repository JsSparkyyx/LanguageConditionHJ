# Phase 0 recon — item-safety environments

Date: 2026-10-01. Node: Delta login node (NO GPU, NO Vulkan). Python:
`/work/nvme/bifq/conda/envs/RoboTwin/bin/python` (sapien 3.0.0b1). Scratch scripts:
`envs/scripts/item_envs/recon_physics_api.py`, `recon_contacts.py`, `recon_urdf.py`,
`recon_assets.py` (all run; outputs pasted below). RoboTwin checkout read-only at
`/work/nvme/bifq/project/RoboTwin`; CEHJ copy at `/work/nvme/bifq/project/LanguageConditionHJ/main`.

**GPU follow-up job (2026-10-01): items 1-2 DONE, item 3 still PENDING.**
1. DONE — regression `h` traces: `docs/item_envs/regression/{ur5-wsg,franka-panda}/{trace.npz,meta.json}`,
   produced by `envs/scripts/item_envs/regression_trace.py` (see "Regression references" below).
   NOTE: `scripts/run_smoke.sh` is STALE — its `ROOT=` points at a dead autodl path; do not use it.
2. DONE — workspace sweeps `envs/main/item_envs/configs/workspace_{ur5-wsg,franka-panda}.npz` +
   measured embodiment facts `facts_{ur5-wsg,franka-panda}.json` (merged into Q10).
3. STILL PENDING-GPU — runtime confirmation that `Segmentation[..., 1] == entity.per_scene_id`
   (the GPU jobs ran without an RT render pass, so this remains unverified; static evidence in Q8).

**CPU-node gotcha that shapes everything below:** even `sapien.Engine().create_scene()`
constructs a `RenderSystem` and dies with `vk::PhysicalDevice::createDeviceUnique:
ErrorExtensionNotPresent`. Physics-only scenes work ONLY via
`sapien.Scene([sapien.physx.PhysxCpuSystem()])`, and any `add_*_visual` /
`scene.add_ground` call also needs Vulkan. All probes below are collision-only.

```python
>>> import sapien; sapien.__version__
'3.0.0b1'
>>> scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])   # works headless
>>> sapien.Scene()                                          # needs Vulkan
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent
```

---

## Regression references (Phase-0 item 1) — GPU job, 2026-10-01

`envs/scripts/item_envs/regression_trace.py` replaces the stale `scripts/run_smoke.sh`
(whose `ROOT=` points at a dead autodl path). It reproduces the `--nominal` path of
`main/train/eval_ckpt.py` for one (task, embodiment) pair: FrozenConfig defaults,
`obstacle_mode="on_path"`, `obstacle_model=068_boxdrink`, HoloBrain encoder via
`build_models`, `RolloutController(mode="filtered", max_steps=7500)`. Command
(cwd `/work/nvme/bifq/project/LanguageConditionHJ`):

```
/work/nvme/bifq/conda/envs/RoboTwin/bin/python envs/scripts/item_envs/regression_trace.py \
    --embodiment <emb> --task place_empty_cup --seed 0 --out envs/docs/item_envs/regression
```

Seed derivation: `episode_seed(seed_base=0, task_idx=0, emb_idx, ep=0)`
(eval_ckpt.py:97-99); `place_empty_cup` is NOT in `eval_ckpt.ALL_TASKS` so the recon uses
a single-task list (task_idx=0); `emb_idx = ALL_EMBODIMENTS.index(emb)` → scene seed
**10000 for franka-panda, 30000 for ur5-wsg** (used_seed identical, no spawn retries
changed it). `h` is stored in `trace.npz` both raw (`h_scaled`, ×20 per `h_scale=20`)
and in metres (`h_m`); meta.json reports metres. Results (meta.json):

| embodiment | seed | ticks | min_h (m) | mean_h (m) | max_h (m) | success | wall (s) |
|---|---|---|---|---|---|---|---|
| ur5-wsg | 30000 | 143 | -0.0277 | 0.0730 | 0.1689 | false | 88.6 (setup 20.6, rollout 68.0) |
| franka-panda | 10000 | 149 | -0.0388 | 0.0011 | 0.0417 | false | 94.2 (setup 20.5, rollout 73.6) |

Both traces go h<0 (nominal planner clips the on-path obstacle — expected, this is the
failure the filter exists for; `success=false` under the eval budget). Planner was
`CuroboPlanner` in both runs (no mplib fallback). These are the fixed-seed references
that must be IDENTICAL after the Phase 5 edits.

---

## Q1 — physical material on EXISTING collision shapes: WORKS (property setter + method)

Snippet (`recon_physics_api.py`, abridged):

```python
import inspect, sapien, sapien.physx as physx
scene = sapien.Scene([physx.PhysxCpuSystem()])
b = scene.create_actor_builder()
m_build = scene.create_physical_material(0.9, 0.8, 0.05)   # (static, dynamic, restitution)
b.add_box_collision(half_size=[0.05]*3, material=m_build)
box = b.build(name="box_q1")
comp = box.find_component_by_type(physx.PhysxRigidDynamicComponent)
sh = comp.get_collision_shapes()[0]
print(sh.physical_material.static_friction, sh.physical_material.dynamic_friction,
      sh.physical_material.restitution)
m_new = scene.create_physical_material(0.11, 0.12, 0.13)
sh.physical_material = m_new                               # property SETTER
print(sh.physical_material.static_friction, sh.physical_material.dynamic_friction,
      sh.physical_material.restitution)
print(hasattr(sh, "set_physical_material"))
```

Output:

```
0.8999999761581421 0.800000011920929 0.05000000074505806   # build-time material sticks
0.10999999940395355 0.11999999731779099 0.12999999523162842 # post-build setter sticks
True                                                        # set_physical_material also exists
```

Findings:

- `comp.get_collision_shapes()` → `list[PhysxCollisionShape*]`; `shape.physical_material`
  is a **read/write property** and `shape.set_physical_material(mat)` also exists. Setting the
  material on existing shapes of an already-built entity works. (Contact-behaviour effect of the
  swap is exercised in Phase 2's `test_physics_sim.py`, not here.)
- Materials come from **`scene.create_physical_material(static, dynamic, restitution)`** — the
  `PhysxCpuSystem` object does NOT have `create_physical_material` in this build
  (`AttributeError` verified).
- Build-time `material=` works on all collision-add methods. Signatures (via `inspect.signature`):

```
add_box_collision(pose=Pose([0,0,0],[1,0,0,0]), half_size=(1,1,1),
                  material: Optional[PhysxMaterial] = None, density=1000,
                  patch_radius=0, min_patch_radius=0, is_trigger=False)
add_multiple_convex_collisions_from_file(filename, pose=..., scale=(1,1,1),
                  material=None, density=1000, patch_radius=0, min_patch_radius=0,
                  is_trigger=False, decomposition: Literal['none','coacd']='none',
                  decomposition_params={})
add_convex_collision_from_file(filename, pose=..., scale=(1,1,1),
                  material=None, density=1000, patch_radius=0, min_patch_radius=0,
                  is_trigger=False)
```

  Note the built-in **`decomposition='coacd'`** option — asset collision fixes (P7) may not
  need external CoACD tooling.
- Default material on a bare physics-only scene is **0.3 / 0.3 / 0.1**; RoboTwin's
  `setup_scene` overrides it to **0.5 / 0.5 / 0.0** via
  `self.scene.default_physical_material = self.scene.create_physical_material(...)`
  (`RoboTwin/envs/_base_task.py:227-231`, old-style `sapien.core` scene; the new-style
  `sapien.Scene` wrapper has NO `default_physical_material` attribute — verified
  `AttributeError`, so per-shape setters are the portable path).
- Mass via density: 0.1 m cube at `density=1000` → `comp.mass == 1.0`. `comp.mass = x`
  assignment sticks even though `auto_compute_mass` defaults to True (readback 0.5), and
  `auto_compute_mass` itself is read-only (`AttributeError: can't set attribute`).

## Q2 — linear/angular damping: WORKS

```python
comp.linear_damping = 0.35
comp.angular_damping = 0.45
print(comp.linear_damping, comp.angular_damping)
# 0.3499999940395355 0.44999998807907104
# defaults on a fresh body: 0.0 0.05000000074505806
```

`PhysxRigidDynamicComponent.linear_damping` / `.angular_damping` are read/write properties.
Defaults: linear 0.0, angular 0.05. (T4's "angular damping 50 during settle" is just a big
value through the same property.)

## Q3 — center-of-mass local pose: WORKS

```python
print(comp.cmass_local_pose)                 # Pose([0, 0, 0], [1, 0, 0, 0])
comp.cmass_local_pose = Pose(p=[0.01, -0.02, 0.03], q=[1, 0, 0, 0])
print(comp.cmass_local_pose.p, comp.cmass_local_pose.q)
# [ 0.01 -0.02  0.03] [1. 0. 0. 0.]
comp.mass = 0.5                              # cmass survives an explicit mass change
print(comp.cmass_local_pose.p)               # [0.02 0. 0.] in the auto_compute probe
```

`cmass_local_pose` is a read/write `Pose` property (also `get/set_cmass_local_pose`).
`mass` is a read/write float property. CoM offset and explicit mass are independent and
both stick.

## Q4 — CCD: NOT exposed per-body; process-global only, must be set BEFORE scene creation

```python
[a for a in dir(physx.PhysxRigidDynamicComponent) if "ccd" in a.lower()]   # []
[a for a in dir(physx) if "ccd" in a.lower()]                              # []
[a for a in dir(scene) if "ccd" in a.lower()]                              # []
# PhysxSceneConfig HAS enable_ccd, but:
cfg = physx.PhysxSceneConfig(); cfg.enable_ccd = True
physx.PhysxCpuSystem(cfg)          # TypeError: PhysxCpuSystem() takes no arguments
sys_cfg = scene.get_physx_system().get_config()
sys_cfg.enable_ccd = True          # get_config() returns a COPY — does not stick
scene.get_physx_system().get_config().enable_ccd   # still False
# the only working path:
physx.set_scene_config(enable_ccd=True)
scene2 = sapien.Scene([physx.PhysxCpuSystem()])
scene2.get_physx_system().get_config().enable_ccd  # True
physx.set_scene_config(enable_ccd=False)           # restore
```

No per-body CCD flag exists anywhere in sapien.physx (component, Scene, physx system).
CCD is a **process-global scene default** set via `physx.set_scene_config(enable_ccd=True)`
(also `physx.set_scene_config(config: PhysxSceneConfig)`); it applies to scenes created
AFTER the call. If P5 (bead tunneling) needs CCD, the probe process must call it before
`Env` construction — and it affects every scene in that process.

## Q5 — kinematic bodies: WORKS, pose follows exactly

```python
kb = scene.create_actor_builder()
kb.add_box_collision(half_size=[0.05]*3)
kin_box = kb.build_kinematic(name="kin")                     # path A
kcomp = kin_box.find_component_by_type(physx.PhysxRigidDynamicComponent)
other = scene.create_actor_builder(); ...build(name="kin2")  # path B
kcomp2 = other.find_component_by_type(physx.PhysxRigidDynamicComponent)
kcomp2.kinematic = True                                      # post-build flag, sticks
for i in range(10):
    kcomp.set_kinematic_target(Pose(p=[0.1*(i+1), 0, 0.5]))
    scene.step()
    poses.append(kin_box.get_pose().p)
# max |x - target| over 10 steps: 2.38e-08 ; z drift: 0.0
```

Both `ActorBuilder.build_kinematic()` and post-build `component.kinematic = True` work;
`set_kinematic_target(Pose)` teleports the body to the target at the next `scene.step()`
(10/10 steps exact to 2.4e-8 m). This is the mechanism for the P1–P6 kinematic
trays/boards/pushers.

## Q6 — solver iteration counts: PER-BODY only (defaults 10 / 1)

```python
[a for a in dir(physx.PhysxSceneConfig()) if not a.startswith("_")]
# ['bounce_threshold','enable_ccd','enable_enhanced_determinism',
#  'enable_friction_every_iteration','enable_pcm','enable_tgs','gravity']   <- no iters
comp.solver_position_iterations, comp.solver_velocity_iterations   # defaults: 10 1
comp.solver_position_iterations = 8; comp.solver_velocity_iterations = 4
comp.solver_position_iterations, comp.solver_velocity_iterations   # 8 4
[a for a in dir(physx.PhysxBodyConfig()) if not a.startswith("_")]
# ['sleep_threshold', 'solver_position_iterations', 'solver_velocity_iterations']
```

- **Scene-level: not settable** — `PhysxSceneConfig` has no iteration fields.
- **Per-body**: `PhysxRigidDynamicComponent.solver_position_iterations` /
  `.solver_velocity_iterations`, read/write, defaults **10 / 1**.
- **Global default** for new bodies: `physx.set_body_config(PhysxBodyConfig)` (same two
  fields, defaults 10/1).
- PLAN P3/P5 gate fix "raise solver iterations" must therefore be per-item-body (or the
  process-global body config before Env construction).

## Q7 — contact impulses between two specific entities: WORKS, force = impulse/dt

Existing pattern (`LanguageConditionHJ/main/envs/distance.py:858-886`,
`_robot_obstacle_contact_force`): iterate `env.task.scene.get_contacts()`, match by
`contact.bodies[i].entity.name` (skip obstacle↔table), per point compute
`f = ||p.impulse|| / dt` (dt = scene timestep) and take the max; `touched = any(
p.separation <= 0)`. The docstring notes PhysX reports SPECULATIVE contacts at 1–2 cm
with zero impulse, so real touches need the separation ≤ 0 check; hard pushes measured
≥ 79 N. Same shape/contact-list pattern is reused for held-object detection
(`_held_obstacle_contact`, `:828-844`).

Runtime probe (`recon_contacts.py`): 1 kg box resting on a static floor, 50 steps:

```python
contacts = scene.get_contacts()
c0 = contacts[0]
[a for a in dir(c0) if not a.startswith("_")]        # ['bodies', 'points', 'shapes']
[b.entity.name for b in c0.bodies]                   # ['box_q7', 'floor_q7']
[a for a in dir(c0.points[0]) if not a.startswith("_")]
# ['impulse', 'normal', 'position', 'separation']
sum(float(np.linalg.norm(p.impulse)) for p in c0.points) / dt   # dt = 1/250
# 9.811 N   (== m*g = 1.0 * 9.81 exactly)
min(float(p.separation) for p in c0.points)          # -1.9e-07
# filter to a specific entity pair:
[c for c in scene.get_contacts()
 if {c.bodies[0].entity.per_scene_id, c.bodies[1].entity.per_scene_id}
 == {floor.per_scene_id, box.per_scene_id}]          # 1 contact
```

Exact API: `scene.get_contacts()` → `list[PhysxContact]`; each contact has `bodies`
(2 rigid components, `.entity` back-reference; for articulations the bodies are
`PhysxArticulationLinkComponent`s whose `.entity` is the LINK entity), `points`
(`position`, `normal`, `separation`, `impulse` — impulse is a 3-vector, N·s), `shapes`.
Pair filtering works by `entity.per_scene_id` or `.name`. Summed |impulse|/dt over the
points of a resting contact equals m·g to 4 digits, so `robot_item_force = Σ|impulse|/dt`
(StepRecord contract) is sound.

## Q8 — actor-level segmentation ids and cameras

Source (`RoboTwin/envs/camera/camera.py:377-388`):

```python
seg_labels = camera.get_picture("Segmentation")   # [H, W, 4]
if level == "mesh":  label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
if level == "actor": label0_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
```

- Channel 0 = mesh(visual-shape)-level id, channel 1 = actor-level id. RoboTwin's own
  helper casts to **uint8, truncating ids > 255** — read the raw channel (PLAN already
  warns; confirmed in source).
- Entity → id: `entity.per_scene_id` (`sapien/pysapien/__init__.pyi:136`; returned real
  ints 1, 2 for rigid entities in the Q7 CPU probe). So `mask = seg[..., 1] ==
  item.entity.per_scene_id`. **STILL PENDING-GPU**: the runtime confirmation that the
  rendered actor-level channel equals `per_scene_id` — the 2026-10-01 GPU jobs ran
  without an RT render pass, so this check was not run; static evidence stands
  (camera.py:377-388 channel assignment; shader plumbing
  `gbuffer.vert: outSegmentation = objectDataBuffer.segmentation`).
- Cameras a RoboTwin task creates (`_base_task.py:402-415` `load_camera` →
  `camera.py:76-151` `load_camera`, config `env_cfg/task_config/_camera_config.yml`):
  - **head_camera** — static, from the embodiment config's `static_camera_list`
    (both ur5-wsg and franka-panda: position (-0.032, -0.45, 1.35), forward (0,0.6,-0.8));
    type from `camera.head_camera_type` default **D435 = 320×240, fovy 37** (franka's
    config also has a commented-out `front_camera` entry — not active).
  - **left_camera / right_camera** — wrist cameras, also D435 320×240, re-posed every
    `_update_render()` from robot links named `left_camera` / `right_camera`
    (`_base_task.py:432`, `envs/robot/robot.py:198-210`; falls back to a link named
    `camera`, then link 0).
  - CEHJ adds its own **observer_camera** (patched to 640×480 default,
    `main/envs/env.py:214-247`).
  - `_task_args` (env.py:317-325) forces head+wrist collection on, types D435.

## Q9 — task-name-keyed resources: nothing in setup_demo breaks on unknown names

Every use found:

| Where | Keyed by task_name? | Unknown name behaviour |
|---|---|---|
| `RoboTwin/envs/_base_task.py:63` | stored only | — |
| `_base_task.py:142-148` `_eval_step_limit.yml` | yes | only read when `eval_mode=True`; the `data[self.task_name]` lookup sits inside try/except → unknown name falls back to `step_lim = 1000` (the `open()` itself is outside the try, but the file exists on disk). CEHJ `_task_args` passes `eval_mode: False` (env.py:303) — dead code path |
| `_base_task.py:118, 571-575` instruction | no | `self.instruction = None`; only set by policy-eval scripts via `set_instruction` |
| `RoboTwin/description/` (task_instruction/, objects_description/) | yes | used by `scripts/collect_data.py` / `eval_policy*.py`, NOT by `setup_demo` |
| `LanguageConditionHJ/main/envs/env.py:375-376` | yes | `importlib.import_module(f"envs.{task}")` + `getattr(module, task)` → **ModuleNotFoundError / AttributeError on unknown names**. This is the real gate: PLAN §2.7's `ITEM_TASKS` registry branch must run before this line |
| `main/envs/tasks.py:47` `TASK_SPECS` (16 known tasks) | yes | `distance.py:334` and `obstacle.py:755` (`stretch_task_spawns`) use `.get(name, {})` → safe no-op; **`obstacle.py:817` `TASK_SPECS[env.task_name]` (in `choose_and_spawn`) raises KeyError** — item tasks must collect with `obstacle_mode="none"` (PLAN §2.7 skips `stretch_task_spawns`, but `choose_and_spawn` is called at collection/rollout time; flag for Phase 5) |

`demo_clean.yml` is loaded as a template by `_task_args`; `task_config: "demo_clean"` is
passed through but never re-read by RoboTwin. No `.json` asset lookup is keyed by task name
(asset lookups are keyed by modelname/model_id).

## Q10 — ur5-wsg and franka-panda

Parsed from the URDFs directly (`recon_urdf.py`):

**ur5-wsg** (`RoboTwin/assets/embodiments/ur5-wsg/ur5_wsg_gripper.urdf`) — **6 arm DoF**,
8 non-fixed joints total:

| joint | type | vel limit (rad/s) | effort (N·m) | range (rad) |
|---|---|---|---|---|
| shoulder_pan_joint | revolute | 3.15 | 150 | ±3.142 |
| shoulder_lift_joint | revolute | 3.15 | 150 | [-3.142, 0] |
| elbow_joint | revolute | 3.15 | 150 | ±3.142 |
| wrist_1_joint | revolute | 3.2 | 100 | ±1.75 |
| wrist_2_joint | revolute | 3.2 | 100 | ±π |
| wrist_3_joint | revolute | 3.2 | 100 | ±π |
| base_joint_gripper_left | prismatic | 1.0 | — | [-0.055, -0.0027] m |
| base_joint_gripper_right | prismatic | 1.0 | — | [0.0027, 0.055] m |

Gripper (WSG-50): both finger prismatic joints sit at origin 0 with axes ±x → finger-base
separation range **0.0054–0.11 m; max opening ≈ 0.11 m** (matches config `gripper_bias:
0.11`). Finger collision mesh `meshes/gripper/WSG-FMF.stl` spans z 0.0575–0.1305 →
**finger length ≈ 73 mm**, closing-axis (x) thickness ≈ 23 mm (trimesh bounds).

**franka-panda** (`RoboTwin/assets/embodiments/franka-panda/panda.urdf`) — **7 arm DoF**,
9 non-fixed joints total:

| joint | type | vel limit (rad/s) | effort | range (rad) |
|---|---|---|---|---|
| panda_joint1–4 | revolute | 2.175 | 87 | [-2.8973,2.8973] / [-1.7628,1.7628] / [-2.8973,2.8973] / [-3.0718,-0.0698] |
| panda_joint5–7 | revolute | 2.61 | 12 | [-2.8973,2.8973] / [-0.0175,3.7525] / [-2.8973,2.8973] |
| panda_finger_joint1/2 | prismatic | 0.2 | — | [0, 0.04] m |

Gripper: finger joints at origin z=0.0584, axes ±y → separation 0–0.08 m;
**max opening 0.08 m** (matches `gripper_bias: 0.08`). Finger mesh
`franka_description/meshes/collision/finger.stl` z-span 0–0.0538 → **finger length ≈ 54 mm**.

So the binding constraints for graspable primitives: **max opening 0.08 m, finger length
≈ 54 mm (franka is the smaller of both)**.

**Paired placement / arm_distance** (`RoboTwin/envs/robot/robot.py:63-122`): both configs
have `dual_arm: False` → `_task_args` passes `embodiment=[e, e, arm_distance]`
(env.py:307) and Robot takes the non-dual branch: TWO separate URDF loads,
`fix_root_link=True`, left base = `robot_pose[0] + (-arm_distance/2, 0, 0)`, right base =
`robot_pose[1] + (+arm_distance/2, 0, 0)`. Config `robot_pose` entries:

- ur5-wsg: `[[0,-0.65,0.65, 1,0,0,0], [0,-0.65,0.75, 1,0,0,0]]` — **asymmetric base
  heights (left z=0.65, right z=0.75)** and identity quats.
- franka-panda: both `[0,-0.65,0.75, 0.707,0,0,0.707]` (yaw +90°).

Default `arm_distance` in LanguageConditionHJ `Env.__init__` = **0.6** (env.py:348), but
**collection overrides it per embodiment**: `main/train/collect.py:123`
`ARM_DISTANCE = {"ur5-wsg": 0.8}` ("ckpt dual-ur5 URDF mounts the arms 0.8 m apart"),
`0.6` otherwise. So training-time bases: ur5-wsg at x = ∓0.4 / ±0.4, franka-panda at
x = ∓0.3 / ±0.3, both y = −0.65.

**Measured arm base poses** (GPU job, `envs/main/item_envs/configs/facts_*.json`,
`arm_base_pose_scene`, collected with the collect.py distances above; scene values match
the configured ones to float noise):

- ur5-wsg (arm_distance 0.8): left p=(-0.400, -0.650, **0.650**), right p=(+0.400,
  -0.650, **0.750**), both q_wxyz=(1,0,0,0) — the **asymmetric base heights are real at
  runtime**, not a config typo.
- franka-panda (arm_distance 0.6): left p=(-0.300, -0.650, 0.750), right p=(+0.300,
  -0.650, 0.750), both q_wxyz≈(0.7071,0,0,0.7071) (yaw +90°; sim-normalized from the
  configured 0.707).

DoF, velocity limits, and gripper openings from the URDF parse above are **confirmed at
runtime** by the same facts files (UR5 6 DoF, 3.15×3 + 3.2×3 rad/s; Franka 7 DoF,
2.175×4 + 2.61×3 rad/s; WSG opening [0.0054, 0.11] m; Panda [0, 0.08] m). Additional
measured facts: table top z = 0.74 m, size 1.2 × 0.7 m, xy_bias (0,0);
`approach_to_tcp_m = 0.12` (the gripper-pose origin → fingertip offset used by the
planner); `place_empty_cup` seed 0 grasps with the **RIGHT** arm; planner
`CuroboPlanner` in both jobs (no mplib fallback); cuRobo max_acceleration 15.0,
max_jerk 500.0.

**Reachable tabletop workspace** (GPU job,
`envs/main/item_envs/configs/workspace_<embodiment>.npz`): LEFT-arm IK sweep, grid x 61 ×
y 36 at 2 cm over the table (x ∈ [-0.6, 0.6], y ∈ [-0.35, 0.35] m), z = table + {0.02,
0.10, 0.20} m (z = {0.76, 0.84, 0.94}), top-down gripper quat (-0.5, 0.5, -0.5, -0.5)
wxyz, 2196 cells/layer, planner CuroboPlanner with **0 planner exceptions** on both.
npz keys: `x, y, z, heights_above_table, feasible[3,36,61] bool, planner_errors,
table_xy_bias, table_top_z, quat_wxyz, embodiment, swept_arm, z_mode,
approach_to_tcp_m`. Per-layer feasible fractions:

| embodiment | +0.02 m | +0.10 m | +0.20 m |
|---|---|---|---|
| ur5-wsg | 0.603 | 0.587 | 0.542 |
| franka-panda | 0.440 | 0.473 | 0.500 |

Caveats: (a) the sweep used the LEFT arm; `dual_arm: False` means the right arm's map is
the mirror about x=0 (relevant because `place_empty_cup` seed 0 grasps RIGHT). (b)
**z-mode caveat**: `z_mode="gripper"` — z heights are gripper-pose origins whose
fingertips sit `approach_to_tcp=0.12` m lower, so the +0.02 layer is mostly a
hand-inside-table probe (fingertips at table −0.10 m); a `--z-mode tcp` variant of the
sweep script exists if TCP-referenced layers are needed later. (c) ur5-wsg's feasible
fraction DECREASES with height while franka's INCREASES — plausible given the ur5 base
z asymmetry, but keep it in mind when intersecting workspaces for layout sampling.

## Q11 — cwd during setup_demo: MUST be the RoboTwin root; CEHJ chdirs for you

`os.chdir` appears exactly once in the involved code: **`main/envs/env.py:142`
(`prepare()`)** — it inserts `ROBOTWIN_ROOT` into `sys.path` and `os.chdir(ROBOTWIN_ROOT)`
before `envs.{task}` is imported. RoboTwin itself never chdirs.

Relative paths that break under any other cwd (grep `"./assets` / `"assets/objects`):

- `envs/utils/create_actor.py:405, 451, 511, 566` — `Path("assets/objects") / modelname`
  (EVERY actor load, incl. collision + model_data json).
- `envs/utils/rand_create_cluttered_actor.py:17, 36, 84` — cluttered-table object lists.
- `envs/_base_task.py:278`, `envs/utils/create_actor.py:52, 225, 358` — texture files.
- Embodiment configs themselves: `file_path: "./assets/embodiments/ur5-wsg"` (relative,
  resolved by `_task_args` → `_load_yaml(Path(robot_file) / "config.yml")`, env.py:290-291),
  and inside them `urdf_path: "./ur5_wsg_gripper.urdf"` (joined to robot_file,
  robot.py:40). URDF-internal mesh refs are relative to the URDF dir (loader resolves).
- Absolute-path exceptions: `envs/_GLOBAL_CONFIGS.py` derives `ASSETS_PATH` /
  `CONFIGS_PATH` from `__file__` (camera config load at camera.py:82 is cwd-independent).

Conclusion: the process cwd must equal the RoboTwin checkout root from `prepare()` onward;
any item-task code that creates actors outside an `Env` lifetime must chdir itself or run
after `prepare()`.

## Q12 — asset audit (Section 7 candidate pools)

`recon_assets.py` mirrors `create_actor.py:487-523` resolution
(`collision/base{id}.glb` → `collision/textured{id}.obj` → top-level) and counts convex
pieces as trimesh scene-geometry nodes of the first valid model id's collision file
(= hulls sapien builds via `add_multiple_convex_collisions_from_file`). `cpp` = fraction
of `model_data*.json` containing `contact_points_pose`.

| asset | on disk | model_data jsons | cpp | collision file (first id) | pieces (first id) | notes |
|---|---|---|---|---|---|---|
| 001_bottle | yes | 23 (ids 0-22) | 23/23 | collision/base0.glb | 3 | |
| 002_bowl | yes | 7 (ids **1-7**) | 7/7 | collision/base1.glb | 28 | **id 0 MISSING — `create_actor(model_id=0)` returns None**; per-id pieces 21-36 |
| 008_tray | yes | 4 (ids 0-3) | 4/4 | collision/base0.glb | 16 | per-id 13-17 (real rim geometry) |
| 011_dustbin | yes | 1 (id 0) | 1/1 | collision/base0.glb | 86 | |
| 015_laptop | yes | 0 top-level | n/a (see notes) | articulation | — | **SAPIEN-articulation asset**: 11 numeric subdirs (9748, 9912, 9960, 9968, 9992, 9996, 10040, 10098, 10101, 10125, 10211) each with `mobility.urdf` + own `model_data.json` (schema: `contact_points`, NOT `contact_points_pose`; scale 0.15 for 10098). 10098: link_0 (body) 83 collision meshes, link_1 (lid) 3, revolute lid joint [-1.583, 1.118] rad |
| 019_coaster | yes | 1 (id 0) | 1/1 | collision/base0.glb | 4 | |
| 021_cup | yes | 13 (ids 0-12) | 13/13 | collision/base0.glb | 30 | hollow cup, ≥4 pieces ✓ (P5) |
| 036_cabinet | yes | 0 top-level | n/a (see notes) | articulation | — | single model subdir `46653/` (`mobility.urdf`); base + frame link_0 (11 coll meshes) + 3 prismatic drawer links (12 coll meshes each, travel 0-0.66 URDF-units × scale 0.27 ≈ 0.178 m); its model_data.json has `contact_points`, `init_qpos`, `target_points`, scale 0.27 — **no `contact_points_pose`**; also has top-level `visual/base0.glb` + `points_info.json` |
| 039_mug | yes | 13 (ids 0-12) | 13/13 | collision/base0.glb | 27 | |
| 063_tabletrashbin | yes | 11 (ids 0-10) | **7/11** | collision/base0.glb | 39 | **ids 1,2,4,5 lack `contact_points_pose`** (reduced schema center/extents/stable); PLAN T7 ids [0,3,7,8,9,10] all have it; per-id pieces 2-70 (id0: 39, id3: 51, id7: 43, id8: 35, id9: 70, id10: 45) |
| 071_can | yes | 6 (ids 0-5) | 6/6 | collision/base0.glb | 26 | |
| 075_bread | yes | 7 (ids 0-6) | 7/7 | collision/base0.glb | 9 | |
| 080_pillbottle | yes | 5 (ids **1-5**) | 5/5 | collision/base1.glb | 2 | **id 0 MISSING**; PLAN §3.1 spells it `080_pill_bottle` — on disk it is `080_pillbottle` |
| 086_woodenblock | yes | 5 (ids 0-4) | 5/5 | collision/base0.glb | 2 | scale [1,1,1] (mesh already metric) |
| 102_roller | yes | 3 (ids 0-2) | 3/3 | collision/base0.glb | 1 | single convex hull (convex cylinder — fine) |
| 105_sauce-can | yes | 5 (ids 0-4) | 5/5 | collision/base0.glb | 4 | |
| 110_basket | yes | 4 (ids 0-3) | 4/4 | collision/base0.glb | 51 | |
| 114_bottle | yes | 4 (ids **1-4**) | 4/4 | collision/base1.glb | 5 | **id 0 MISSING** — T1's tippable pool must pass model_id≥1 |
| 048_stapler | yes | 7 (ids 0-6) | 7/7 | collision/base0.glb | 16 | |
| 113_coffee-box | yes | 7 (ids 0-6) | 7/7 | collision/base0.glb | 4 | |

Articulation assets (`015_laptop`, `036_cabinet`) load via
`create_sapien_urdf_obj` / `rand_create_sapien_urdf_obj`
(`envs/utils/create_actor.py:587-653`, `rand_create_actor.py:150-180`): `modelid` is
matched against the subdir NAME (or used as a sorted index if smaller than the subdir
count), `loader.scale = model_data["scale"]` (a plain float there — 0.27 / 0.15),
`fix_root_link` configurable, `init_qpos` from the json is applied, per-link masses come
from `model_data["mass"]` when present, and the json's `transform_matrix` is composed
into the root pose.

Piece counts per model id for container-type assets (hollowness evidence for P4/P5/P7):

- `063_tabletrashbin`: id0=39, id1=7, id2=7, id3=51, id4=2, id5=3, id6=20, id7=43,
  id8=35, id9=70, id10=45.
- `002_bowl`: ids 1-7 → 28, 31, 24, 24, 36, 24, 21.
- `008_tray`: ids 0-3 → 16, 13, 17, 13.
- `011_dustbin`: id0=86.
- `036_cabinet` drawers: 12 collision meshes per drawer link (from mobility.urdf).

All 20 assets exist on disk. Every rigid asset that has model_data jsons provides
`contact_points_pose` for at least its first id. P7's "≥4 pieces or CoACD overlay" gate
fails as-is only for: `080_pillbottle` (2 pieces — but it is not a container, only a
tippable item, so single-hull is fine), `086_woodenblock` (2), `102_roller` (1), and any
`063_tabletrashbin` ids 4/5 (2-3 pieces) — none of these is used as a container in
Section 7, so no overlay is strictly required by the audit; confirm in P7.

## Most surprising findings (plan-affecting)

1. `PhysxSceneConfig` has **no solver-iteration fields** — "raise solver iterations" (P3/P5
   gate fixes) is per-body (`comp.solver_{position,velocity}_iterations`, defaults 10/1) or
   process-global via `physx.set_body_config` before scene creation.
2. CCD is **not a per-body flag**; only `physx.set_scene_config(enable_ccd=True)` before
   scene creation, process-global. PLAN's "enabling CCD" fix touches the whole process.
3. `002_bowl`, `080_pillbottle`, `114_bottle` have **no model_id 0** —
   `create_actor(model_id=0)` (the default) prints "is not exist model file!" and returns
   None. Item samplers must enumerate existing ids.
4. `015_laptop` and `036_cabinet` are **articulation assets** (numeric subdir +
   `mobility.urdf`), not glb — and their `model_data.json` uses `contact_points`, not
   `contact_points_pose`. "Static laptop as forbidden region" = a fix_root_link
   articulation, not a static actor.
5. `063_tabletrashbin` ids 1,2,4,5 silently use a reduced json schema (no
   `contact_points_pose`); the PLAN's id list [0,3,7,8,9,10] happens to be safe.
6. ur5-wsg's `robot_pose` config gives the **left base z=0.65 and the right z=0.75** —
   the pair is asymmetric in height (franka is symmetric). **Confirmed at runtime by the
   GPU job** (`facts_ur5-wsg.json`, scene pose == configured). Related correction to the
   first-pass doc: collection does NOT use `Env`'s 0.6 default `arm_distance` for ur5 —
   `collect.py:123` overrides it to **0.8** (bases at x=∓0.4); franka stays 0.6.
7. `add_multiple_convex_collisions_from_file` has **built-in `decomposition='coacd'`** —
   asset collision fixes may not need external decomposition tooling.
8. Material/damping/CoM/kinematic/contacts all work exactly as the PLAN hopes, verified
   headless; but no RoboTwin task can EVER run CPU-only (renderer is unconditional), so
   even the pure-physics probes P1-P6 could in principle run headless — but anything that
   builds a `Base_Task` cannot.
