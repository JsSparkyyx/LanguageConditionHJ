# Residual safety policy on the don't-touch task: plan and how-to

Written 2026-11-09 for a new session that works in `residual/`. Section 1 is what to build, as agreed with
Yuxuan. Sections 2 to 5 say how to use what already exists: the modified RoboTwin tasks, the cuRobo planner,
logging, and the cluster. Work out the remaining design details with Yuxuan.

The root `CLAUDE.md` working rules apply. Don't edit `main/network`, `main/train`, `main/envs` or anything under
`envs/` without asking. Compose and subclass in `residual/`.

## 1. Goal and agreed design

Train a residual policy on privileged simulator state with PPO, on top of the cuRobo planner, on the don't-touch
task (`selective_avoid`), ur5-wsg only. Evaluate it periodically against the planner alone, and log training
numbers, evaluation numbers and evaluation videos to wandb. No images, no language and no distillation in this step.

Why: the learned HJ filter's critic ignores the action (`docs/lang_filter/DIAG_ACTOR_2026-10-30.md`), and PPO
needs only a state value V(s). The recipe follows the ResSafe-style filter in arXiv 2610.37652
(https://arxiv.org/html/2610.37652v1).

Agreed on 2026-11-09:

- **Rate: 25 Hz, our previous setting.** cuRobo's plan has one row per physics step (250 Hz), and the runner's own
  playback sets a new joint target at every physics step. A normal policy runs at 10 to 25 Hz, so here the planner
  and the policy both act once per 40 ms tick, as in CrossSafe. The planner is read once per tick as a_nom, the
  displacement from the live joints to the plan row at the end of the tick. The policy outputs one δ per tick.
- **Command.** Each tick the robot executes clip(a_nom + δ), interpolated over the tick's 10 physics steps with a
  velocity feed-forward (`step_dtheta` semantics, section 3). δ is the policy's residual per arm joint (12 joints,
  grippers untouched). The plan cursor advances every tick. With δ = 0 this is the planner running as a 25 Hz
  policy, and that is the evaluation baseline, not the 250 Hz playback.
- **Replan when the arm is far off the plan** (Yuxuan, 2026-11-09). a_nom aims from the live joints at the plan
  row, so the planner pulls the arm back onto its old path every tick, and δ does not add up over ticks: the arm
  sits at most |δ| away from the plan. Example on one joint with δ bounded at 0.06 rad: the plan row is 1.01, the
  arm ends a tick at 1.07, next tick a_nom = 1.02 − 1.07 = −0.05 and δ = +0.06 puts it at 1.08, still 0.06 off.
  0.06 rad is about 3 cm at the hand, and getting around an object can take over 10 cm. So replan the arm from
  its live joints to the same goal (`TickChunkedController._replan_arm`, which keeps the target and constraint) and
  restart its cursor at row 0. The detour then becomes the new path, the policy no longer has to hold it, and
  there is no return trip to the old path. Implement both triggers as a config switch and compare them later
  (Yuxuan, 2026-11-09):
  - **fixed rate**: replan every N ticks (start with N = 5, i.e. 5 Hz). The planner becomes a closed-loop nominal,
    but the δ = 0 baseline is then no longer the calibrated planner;
  - **deviation**: replan when any joint of the arm is more than a threshold off its plan row (start with 0.1 rad,
    about half the δ bound; it must be below the bound or it never fires). With δ = 0 it never fires, so the
    baseline stays the calibrated 25 Hz planner, and it costs a plan only during detours.

  Things to measure for the comparison: replans per episode and ticks per second (a cuRobo plan costs about 100 ms
  of wall time), whether a replanned path starts from rest (check whether the current joint velocity is passed to
  cuRobo; if not, every replan slows the arm down and up again, which makes the fixed-rate nominal stutter and
  hit fewer objects), and the δ = 0 hit rate and task success under each trigger. Starting δ bound: 0.2 rad per
  joint. The δ = 0 baseline always uses the same trigger as the policy it is compared with. A failed replan ends
  the episode as a truncation (bootstrap V(s)).
- **Reward.** r = 1 / (1 + w‖δ‖²/n) per tick, with δ divided by its bound and n = 12. The correction weight w is
  drawn from U(0, 2) per episode and is a policy input.
- **Termination.** The episode ends when the named object's rule breaks. Don't touch (`avoid`): an arm collision
  sphere or the held object comes within 1 cm of it. Don't move (`stay`): it moved more than 3 cm, or tilted, with
  20° counting as the full 3 cm.
- **Task finished.** Once the plan has finished and everything has settled, end the episode and bootstrap with the
  value of surviving forever with no correction, 1/(1 − γ) = 100 at γ = 0.99. Finishing early or late then never
  changes the return. Use this closed form rather than the critic, because no transition ever starts from a
  finished, settled state, so V is never trained there. Fallback if it proves unstable: hold the arm until a fixed
  horizon, keep paying the reward, and treat the horizon as a time limit with a V(s) bootstrap.
- **Missing the grasp on purpose** (discussed with Yuxuan, 2026-11-09). The held item makes the robot bigger, so
  a policy can protect the named object by nudging the hand off the plan just before the gripper closes: nothing
  is picked up, and in scenes where the held item is what would hit the object, the empty carry passes safely.
  The residual must still act all the time, including when nothing is held (Yuxuan), so implement two variants
  as a switch and compare them:
  - **completion value only on success**: at a finished, settled episode, bootstrap with 1/(1 − γ) only if
    `task.check_success()` is true, otherwise 0. Note that the careless planner itself fails the task in about
    30% of episodes, so some of those zeros are not the policy's doing;
  - **no change**: the completion value is paid whatever the task outcome, to see how severe the problem really is.

  For both, log task success next to the violation rate in training and evaluation (success falling while
  violations fall is the sign of this problem). In the throughput step's δ = 0 runs, also record which part makes
  the first contact with the in-the-way object, the held item or an arm sphere (`robot_item_distances` takes the
  minimum over both). That says how often the incentive exists at all.
- **State.**
  - robot: both arms' joint positions and velocities, gripper openings, link (collision-sphere) positions, the
    held object's pose;
  - planner: a_nom and the plan's rows 5, 10, 15, 20 and 25 ticks ahead, as offsets from the current joints;
  - the named object only: pose, size, velocity, displacement from its start pose, an avoid / stay flag, and the
    distances and closest-point directions from each link and the held object to it;
  - w.

  Other objects are left out on purpose.
- **Episodes.** The named object is in the planner's way in half of the episodes and out of its way in the other
  half. Small noise on a_nom during training.
- **Network.** Plain MLPs for actor and critic, with input normalization.
- **Small initial exploration noise** (Yuxuan, 2026-11-09). PPO adds fresh noise to δ every tick. With replanning,
  noise that pushes an arm past the replan point is locked into the new path (every replan with the fixed-rate
  trigger), so early in training the arm can drift and miss grasps or placements by accident. Start the policy's
  standard deviation small, well below the replan threshold (for example 0.1 of the δ bound, about 0.02 rad), and
  log it. Also log, per episode, the hand's offset from the plan at the moment each gripper closes and opens
  (joint and end-effector distance) and task success, so an accidental miss can be told apart from a learned one.
- **First step: throughput.** Ticks per second of one process without rendering, scene building and cuRobo
  planning included, and how many processes fit on a node. That number sizes the PPO budget.
- **Evaluation.** The baseline is the same controller with δ = 0.
  - Scenes with the named object in the way: how often its rule breaks.
  - Scenes with it out of the way: the mean correction, which should stay near 0.
  - Both: task success.
  - Videos of a few fixed scenes at every evaluation.

Two notes for the session:

- γ = 0.99 is PPO's discount for this reward. The root `CLAUDE.md` line "γ = 0.95 only" is about the HJ filter's
  Bellman target and does not apply here. The avoid / stay flag in the state is fine too. The ban on rule-type
  inputs is for the deployed language model, and this policy is a privileged teacher.
- The final clip on clip(a_nom + δ) is the speed limit, dθmax per joint (0.13 rad per tick for a 3.14 rad/s joint
  limit). The δ bound is separate and larger: it is how far from the plan the policy may put the arm.

## 2. The modified RoboTwin tasks (item environments)

**Where things are.**

- `envs/main/item_envs/`: the item environments, imported as `main.item_envs` (runner, rules, margins,
  distances, spawning, the controller with plan retiming).
- `envs/robotwin_tasks/`: our RoboTwin task files for the new tasks. The runner puts this folder first on
  RoboTwin's `envs.__path__` (`runner.use_own_robotwin_tasks`).
- The don't-touch task has no task file. It runs five unmodified RoboTwin pick-and-place hosts
  (`place_container_plate`, `place_burger_fries`, `stack_blocks_two`, `stack_bowls_two`, `place_bread_basket`,
  picked by seed % 5) and spawns 2 or 3 extra objects (wine glass, milk box, apple, bottle, coloured box, coloured
  cylinder) on, near or off the carry path (`spawn.py`). Config:
  `envs/main/item_envs/configs/tasks/T1_selective_avoid.yaml`.
- Docs: `envs/docs/item_envs/tasks_overview.md`, `envs/docs/item_envs/task_cards/T1.md` (its object pool is the
  older one, the yaml is current), `envs/docs/item_envs/calibration.md`, `docs/lang_filter/DESIGN_CONTEXT.md`
  sections 2 and 3.

**Importing.** From the repo root, `import main.lang` first. It appends `<repo>/envs/main` to `main.__path__`, so
`main.item_envs` imports next to CrossSafe's `main.envs`. Code run from `envs/` imports `main.item_envs`
directly. One episode with a video, to look at the task:

```bash
scripts/lang/in_container.sh bash -c "cd envs && python -m main.item_envs.probe episode \
    --task selective_avoid --embodiment ur5-wsg --seed 0 --video --out /abs/out"
```

**Running an episode from Python.**

```python
import main.lang  # noqa: F401  (path setup for main.item_envs)
from main.item_envs.runner import ItemEpisodeRunner

r = ItemEpisodeRunner("selective_avoid", embodiment="ur5-wsg", seed=seed, render="none",
                      out_root="/abs/out", max_steps=7500)   # Env() chdirs into RoboTwin: absolute paths only
r.setup(build_controller=False)   # Env, spawned objects, physics randomization, rules, no controller yet
# build and attach your controller here (section 3), then r.ctrl = ctrl
meta, err = r.run()               # runs task.play_once(), writes trace.npz and meta.json to r.out_dir
```

`render="none"` skips all camera rendering. `meta` holds per-rule `violations`, `min_margin` and
`first_violation_t`, plus `task_success`, `plan_failures` and the spawn report in `meta.diag["spawn"]`. `run()`
writes the trace files for every episode, so point training episodes at node-local scratch or skip
`trace.finalize` in a subclass.

**What the runner gives you.**

- `r.items`: ItemSpecs, first the host's own objects (role `task_object` or `support`), then the spawned
  candidates. Each has `item_id`, `half_extents`, `obb_center_local`.
- `r.rules`: one `avoid` and one `stay` rule per candidate, plus set rules that you skip. Fields `rule_id`
  (`avoid|<item_id>|-` for don't touch), `consequence`, `item_ids`, `params`.
- `r._spawn_report["placements"]`: per candidate `item_id`, `cls` (`on_path`, `near_path` or `off_path`),
  `category`, `color`, `xyz`.
- Every tick `_capture_tick` appends a StepRecord with `qpos`, `ee_pose`, `gripper`, `held` (item index per arm,
  -1 for none), `item_pose` [K, 7], `item_vel` [K, 6], `margins` [R] (per rule, metres, positive = safe),
  `robot_item_dist`, `robot_item_force`. To end the episode on a broken rule, subclass, call
  `super()._capture_tick(...)`, read the named rule's margin and raise. A subclass of the runner's `ItemTimeout`
  unwinds `play_once` without a traceback.
- Margins (`envs/main/item_envs/margins.py`): `m_avoid` = distance − 1 cm, `m_stay` = min(3 cm − horizontal
  displacement, 3 cm × (20° − tilt) / 20°), with the reference pose taken at t = 0. The rule is broken when the
  margin is below 0. h = clip(margin / unit, −1, 1) (`rule_h`), with a unit of 5 cm for avoid and 3 cm for stay.
- Distances (`envs/main/item_envs/robot_distance.py`): `robot_item_distances(env, items)` gives the exact signed
  distance from the arm collision spheres and the held object to each item's collision shapes. Sphere centres and
  radii in the world come from `_arm_spheres_world(env, spheres, cehj)`, with the spheres of
  `configs/collision_spheres/ur5-wsg.yml` (66 per arm over 13 links). The distance of each sphere to one item:
  `item_shapes.sphere_distances(centers, radii, shape)`.
- In the planner's way or not: `main/lang/scene_roles.py` `_t1_reasons(env, record)`. Set `env._item_runner = r`
  first (the record argument is not used for this task). On-path and near-path objects, and off-path objects
  within 10 cm of a second carry corridor, count as in the way, the rest as out of the way. This is fixed per seed.
  The spawner puts 2 objects on or near the path whenever the layout allows, so only some scenes have an
  out-of-the-way object. Draw another seed when an episode needs one.

**Known behaviour.**

- cuRobo is nondeterministic run to run. The same seed gives the same layout but not the same motion, so compare
  rates over many scenes.
- Some layouts place no object (`meta.diag["spawn_layout_infeasible"]`). Skip them.
- Every Env build leaks pybind11 thread-local keys, and a process aborts after about 750 builds. Build at most 200
  per process (`main/lang/config.py` `max_inits_per_process`), then restart it. Existing code exits with 75 for
  that.
- The careless planner breaks the in-the-way object's rule in about 81% of ur5 episodes and finishes the task in
  about 70% (`envs/docs/item_envs/calibration.md`). These numbers come from the 250 Hz playback, so measure the
  25 Hz planner's own rates in the throughput step.

## 3. The cuRobo planner and the per-tick controller

**Planner.** `CuroboIKController.install()` (`main/envs/controller.py`, called inside `setup()`) patches RoboTwin's
planner so each process builds cuRobo's MotionGen once and reuses it across episodes.
`update_curobo_world(env.robot)` (`main/envs/obstacle.py`) resets cuRobo's world to the table only, so the planner
never sees the extra objects and drives through them. A "safe" planner mode also exists (`nominal_mode="safe"`,
`safe_plan.py`, plans around protected objects). You don't need it.

**Speed.** `RiskTickController` (`envs/main/item_envs/controller.py`) retimes every plan. On the carry segments
(skill labels `pre_place`, `place`, `lift`) it uses the episode's time scale, one of 0.5, 0.75, 1.0, 1.15 and 1.3
with equal weight per seed (the yaml's `risk_collect` cells, which `risk.sample_risk` prefers over `risk`). Other
segments run at 0.6. Higher is faster.

**How a tick runs.** `TickChunkedController._tick_take_dense_action` (`main/envs/controller.py`, about line 1164)
replaces RoboTwin's `take_dense_action`. Physics runs at 250 Hz and the plan has one row per physics step. This loop
plays the rows one per physics step (a 250 Hz command), and only its hook runs at 25 Hz, once every 10 rows. At each
tick boundary it calls `self.tick_hook(ctx)` with
`ctx = {left_arm, right_arm, i, left_g, right_g, g_i}`: the two arm plans (dicts with `position` and `velocity`
rows, read with `_path_arrays(plan)`), the row cursor `i`, the gripper plans and their cursor `g_i`. The hook runs
only while an arm plan is playing. If it returns True, the cursor stays put and the arms are replanned when control
is handed back. That is how the HJ filter takes over.

The residual needs a different loop that sends one arm command per tick (25 Hz): a subclass of `RiskTickController`
that overrides `_tick_take_dense_action` and, every tick, computes a_nom, asks the policy for δ, executes
clip(a_nom + δ) over the 10 physics steps, plays the gripper rows as the original does, and advances `i` by 10. It
replans an arm on the configured trigger of section 1 (every N ticks, or when the arm is more than the threshold
off its plan), with `_replan_arm(arm, plan)` (returns None on failure), then sets that arm's cursor back to row 0.

**a_nom** (as in `main/train/rollout.py` `_anom`): per arm, a_nom = pos[min(i + 9, n − 1)] − q_live, the plan row at
the end of this tick minus the live joints (`robot.get_left_arm_real_jointState()[:-1]` and the right one, in plan
joint order). An arm without a plan in the current motion holds still.

**Executing a joint displacement.** `Env.step` and `Env.step_dtheta` (`main/envs/env.py`, about line 645) move the
joint target from q0 to q0 + a in 10 equal steps with velocity feed-forward a / 0.04 s, the same (position,
velocity) semantics as the planner's own playback. Write that loop inside your controller so the gripper rows play
in the same physics steps. `Env.step` also calls `_update_render()` every tick, which only video needs.

**Speed limit.** `RiskTickController._v_limit(arm)` reads the URDF joint velocity limits. dθmax = limit × 0.04 s.

**Wiring.** After `r.setup(build_controller=False)`, build the controller as the runner does (`runner.py`, about
line 330), then attach it:

```python
ctrl = ResidualTickController(r.env, risky_skill_labels=tuple(r.task_cfg.get("risky_skill_labels", ()) or ()))
ctrl.attach()
r.ctrl = ctrl
meta, err = r.run()
```

**Watch out.**

- Plans that do not come from the controller's `plan()` (untagged, see `_has_tag`) play through RoboTwin's original
  `take_dense_action` without ticks. Count those ticks.
- `main/envs/controller.py` also has a class named `ResidualController`. It adds a whole-path offset at plan time,
  is zero by default, and the tick controllers inherit from it. It is not this residual.
- Settling after the plan: the branch tool's rule (`scripts/lang/diag_1030/t5_branch.py`, `SETTLE_TICKS = 50`,
  speeds below about 1 to 2 cm/s for 2 s). One way is to wrap `task.play_once` so it keeps stepping physics until
  everything has settled. The runner's per-tick capture keeps running during those steps.

## 4. Logging

- **wandb.** A new project, for example `cehj-residual` (the language filter uses `cehj-lang`). Wrap every log call
  like `_safe_wandb_log` in `main/lang/train.py` (catch, print, continue). A CommError once killed a run. On a
  resumed run use `wandb.init(id=<saved id>, resume="allow")`.
- **Online mode on Delta.** In the job script, on the host side before the container starts, run
  `source scripts/lang/wandb_env.sh`. It reads WANDB_API_KEY from `/work/nvme/bifq/conda/activate_RoboTwin.sh` and
  sets WANDB_MODE=online, and the container inherits both. Never print the key or put it on a command line. Without
  it, `in_container.sh` runs wandb offline (`wandb sync <run dir>` later).
- **Videos.**
  - `main.envs.record.observer_rgb(task)` returns the observer camera frame and syncs the ray-traced scene itself.
    It does not depend on the runner's render mode, but it needs an RTX GPU (A40). `to_record_size` downsamples
    to 320 × 240.
  - `envs/main/item_envs/logger.py` `OverlayVideoWriter(path, fps=25)` has `.add(frame, lines, plot)`, `.flash()`
    (a red border, for the first violation) and `.close()` (writes the mp4).
  - To mark the named object, project its box corners with `main/envs/record.py`
    `_project(task.cameras.observer_camera, corners)` and draw them with `_draw_cube`.
  - Log the videos of one evaluation as a list under one key, `[wandb.Video(p, format="mp4", caption=...)]`, as
    `main/lang/rounds.py` does for `round_videos`. Select the files by modification time, not by name, because
    names repeat across restarts.
- **Checkpoints.** Write to a temporary file, then `os.replace`, so a crash never leaves half a file.

## 5. Cluster (NCSA Delta)

- **Code.** `/work/nvme/bifq/project/LanguageConditionHJ`, this repo. `main/envs/env.py` finds RoboTwin at
  `<repo>/RoboTwin` or `../RoboTwin`. Write outputs to `runs/residual/<name>` (`runs/` is gitignored).
- **Container.** Run every simulator or GPU command as `scripts/lang/in_container.sh <command>`. It uses the
  apptainer image `/work/nvme/bifq/robot-sim_cu128.sif` and the RoboTwin conda env's python, with cwd = repo root
  and the caches and TMPDIR set. cuRobo needs a writable TMPDIR, otherwise it silently falls back to the mplib
  planner. Grep logs for "CuRobo init failed".
- **GPUs.** Account `bifq-delta-gpu`. Use A40s only (46 GB, with RT cores for the video frames). cuRobo fails on
  Delta's A100s (GPU architecture), and they have no RT cores.
  - `gpuA40x4-interactive`: 1-hour jobs. `scripts/lang/stage.sbatch` (1 GPU, 16 CPUs, 64 GB), e.g.
    `sbatch --export=ALL,STAGENAME=<name>,STAGECMD='python -m residual.<module> ...' scripts/lang/stage.sbatch`.
    `scripts/lang/gpu_stage.sbatch` takes 1 to 4 GPUs, sets STAGE_DEADLINE, and treats exit code 75 as "stopped in
    time, resubmit".
  - `gpuA40x4`: up to 48 hours. `scripts/lang/long.sbatch` requeues itself on USR1 300 s before the end and
    relaunches the command on exit code 75, so the command must resume from its own checkpoint.
  - A node has 4 A40s and 64 CPU cores (check with `sinfo -p gpuA40x4 -o "%n %c %m %G"`). To run several
    processes over a job's GPUs with CUDA_VISIBLE_DEVICES pinned, use
    `scripts/lang/par_gpu.sh <cmds> <log dir> <n_gpus> <per_gpu>`.
- **Processes per GPU.** Physics is CPU PhysX, one core per process. cuRobo shares the GPU, and about 8 planners
  fit on an A40 (`envs/scripts/item_envs/par_episodes.sh`). The language collector takes 14 to 15 GB per process
  because of HoloBrain and rendering. A physics-only worker should need much less, so measure it. Set OMP, MKL and
  torch threads to 1 per worker.
- **Queue.** The cluster is busy. On 11-09 one-hour A40 jobs waited 4 to 8 hours, and the don't-touch data test is
  using the A40 interactive slots (`docs/lang_filter/TODO.md`). Never cancel other jobs (`squeue -u $USER` lists
  them).
- **Storage.** `/work/nvme` has a project quota on its SSD pool: `lfs quota -p 22016 --pool ddn_ssd /work/nvme`
  (soft 2,000 GiB, and plain `quota` does not show it). Put large output folders on the HDD pool with
  `lfs setstripe -p ddn_hdd <dir>` (files created inside afterwards land there). Don't keep per-episode traces of
  training episodes.
- **Login nodes** (dt-login01 to dt-login04). Light watchers and feeders run there under nohup. Heavy work goes
  through Slurm.

## 6. Other context

- `AGENTS.md`: CrossSafe's semantics and gotchas. Ray tracing needs `task._update_render()` before a picture, the
  tick accumulator needs its 1e-9 epsilon, and PhysX contact points are speculative (a real touch is a separation
  ≤ 0).
- `docs/lang_filter/DIAG_ACTOR_2026-10-30.md`: why the HJ filter's actor learned nothing.
- `main/lang/eval_closed.py`: `trace_violations(runner)` (per-rule violation and its first time, from a trace) and
  `wilson_str` (confidence intervals for rates).
- `docs/lang_filter/TODO.md`: what is running on the cluster now, and the storage notes.

## 7. Decisions of 2026-11-09 (session start) and the implementation

Yuxuan's answers before the code was written:

- **The policy acts on every tick, also while a gripper closes or opens.** A gripper segment has no arm plan, so
  before this it played outside the per-tick loop (300 physics steps = 30 ticks, 2 to 4 per episode). An arm with no
  plan in the current motion holds its **anchor**, the last row of its last plan (its start pose before the first
  one): a_nom = anchor − q_live. Example: the arm ends the grasp descent 0.05 rad off the grasp row on one joint;
  while the gripper closes, a_nom = −0.05 pulls it back, so δ never adds up on an arm that has no plan.
- **Training noise goes on the executed motion**: the arm runs clip(a_nom + noise + δ) with noise ~ N(0, 0.01 rad)
  per joint per tick. The a_nom in the state stays exact. No noise in evaluation or in the δ = 0 baselines.
- **This session stops after build, test and the throughput step** and reports before any long training. The first
  PPO run, when started, is the deviation trigger with the completion value paid only on task success.

Defaults chosen in the session (change in `residual/config.py`):

- Each episode names one extra object, in the planner's way in half of the episodes, and its `avoid` or `stay` rule,
  half each. A per-worker counter keeps the in-the-way share at 50 %: a scene with only in-the-way objects is rebuilt
  with the next seed once the out-of-the-way episodes lag by 2.
- How an episode ends: the named rule breaks (terminal, nothing after); the host script returns, the arms hold still
  and everything rests (both hands below 1 cm/s, every item below 2 cm/s, the named object's tilt below 1 deg/s for
  50 ticks, at most 300 ticks of waiting): terminal plus the completion value, the waiting ticks pay nothing; a
  failed replan, a failed plan in the host script, the time limit (50 s) or an error: truncation with V(s).
- Found while reading the code: **a replanned path starts from rest.** RoboTwin hands cuRobo the joint positions
  only (`JointState.from_position`), so every replan slows the arm down and up again. The throughput step measures
  how much (the old plan's speed at the cursor vs the new plan's first-tick speed, per replan).
- Smoke test (2026-11-11, A40): confirmed, the old plan moved at 0.9 rad/s at the cursor and the new plan's first row
  at 0.01. Also: a new cuRobo plan is at least about 77 rows (8 ticks) even for a few-mm move, so the fixed trigger
  with N = 5 never reached a plan's end (250 replans, then the time limit). The fixed trigger now skips a replan when
  the current plan ends within N ticks anyway.
- Throughput step (2026-11-12, one A40, 16 cores, 6 minutes per setting, delta = 0, training noise on,
  `runs/residual/check_1109/throughput/summary.json`):

  | setting | sim ticks/s (all workers) | per worker | replans per episode | success | in the way: broken | out of the way: broken |
  |---|---|---|---|---|---|---|
  | planner playback, 8 workers | 267 | 34 | 0 | 0.64 | 0.70 | 0.01 |
  | 25 Hz loop, deviation, 1 worker | 53 | 53 | 6.9 | 0.58 | 0.59 | 0.00 |
  | 25 Hz loop, deviation, 8 workers | 208 | 28 | 4.7 | 0.53 | 0.66 | 0.00 |
  | 25 Hz loop, deviation, 12 workers | 233 | 21 | 2.9 | 0.57 | 0.76 | 0.02 |
  | 25 Hz loop, fixed 5 Hz, 8 workers | 42 | 6 | 169 (median 42) | 0.29 | 0.50 | 0.00 |

  About 65% of the sim ticks are policy ticks (the rest is settling), so 8 workers give about 135 training
  transitions per second. 12 workers add only 12%: the 16 cores are the limit (load 16 to 22). Each worker uses
  2.2 GB of GPU memory. A scene build costs about 3.5 s of a 13 s episode. About 14% of the episodes end in a host
  planning failure in every setting, the 25 Hz loop included.
  Replans start from rest: the old plan's median speed at the cursor 0.80 rad/s, the new plan's first row 0.01 rad/s.
  Under the fixed trigger the arm therefore crawls (finished episodes 306 policy ticks against 222, a third of
  them reach the time limit) and each replan costs 0.44 s: the fixed trigger is 5 times slower in wall time.
  The 25 Hz loop lowers success a little against playback (0.53 to 0.58 against 0.64). At a grasp's close, the hand
  is 1.9 cm (median) off the plan's last row, probably because the move segment ends when the cursor passes the plan's end
  and the arm still lags.
- Training check (2 updates, 4 workers): runs end to end; 0 of 24 episodes succeed with the initial random
  residual (std 0.02 rad per joint per tick), against 0.53 at delta = 0. With the completion value paid only on
  success, success is then rare at the start.
- First PPO run (Yuxuan, 2026-11-12): 12 hours, 350 updates, one A40 with 8 workers, deviation trigger, completion
  value only on success, start noise init_std 0.025 of the bound (0.005 rad per tick, not 0.02, so successes stay
  common early). Run `runs/residual/residual_dev_success_s025_1112`, job 23824247 (gpuA40x4).
- A link's position in the state is the centre of its collision spheres (13 links per arm); its distance to the
  named object is the exact minimum over its spheres (the rule's own code); its direction points to the closest
  point of the object's box.
- Evaluation: 20 in-the-way and 20 out-of-the-way scenes from seeds training never uses (9,000,000 and up), hosts
  balanced, avoid and stay alternating, w = 1, deterministic policy, no noise; the episode runs to its end to measure
  task success, and the first break is recorded. The δ = 0 baseline runs once on the same scenes with the same
  trigger. Videos of the first 2 + 2 scenes at every evaluation.

Code (all in `residual/`, nothing outside it changed):

| file | what |
|---|---|
| `config.py` | every knob with today's defaults |
| `controller.py` | `ResidualTickController`: the 25 Hz loop, anchors, both replan triggers, gripper offsets |
| `state.py` | the named object's distances (equal to `robot_item_distances`, checked every tick in the smoke test) and the state vector (340 numbers) |
| `scene.py` | build a scene, which objects are in the way, name a rule |
| `episode.py` | one episode: reward, termination, settling, transitions; videos |
| `policy.py`, `ppo.py` | MLP actor and critic with input normalization; GAE and clipped PPO |
| `worker.py`, `pool.py` | worker processes (several per GPU) and their restarts after 200 Env builds |
| `train.py` | the learner: PPO updates, evaluation rounds, wandb (`cehj-residual`), checkpoints, resume, exit 75 |
| `throughput.py` | the first step: δ = 0 episodes per trigger and process count |
| `smoke.py`, `tests/` | simulator smoke test; non-simulator unit tests |
| `jobs/check.sbatch` | smoke test → two-update training → throughput on one A40 |

How to run (inside the container, from the repo root):

```bash
python -m pytest -q residual/tests/test_nonsim.py                      # no simulator
python -m residual.smoke --out /abs/runs/residual/smoke                  # one A40
python -m residual.throughput --out /abs/runs/residual/throughput \
    --settings deviation:8,fixed:8,playback:8 --minutes 8
python -m residual.train --run /abs/runs/residual/<name> --updates 2000 \
    --set replan_trigger='"deviation"' completion='"success"'           # wandb online: source scripts/lang/wandb_env.sh first
```
