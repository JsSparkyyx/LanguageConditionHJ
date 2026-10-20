# Does a VLA fine-tuned on safe demonstrations become safe?

Question (Yuxuan, 2026-10-15): if HoloBrain-0 is fine-tuned on demonstrations that avoid the obstacle, does
the fine-tuned policy also avoid it? We collect safe demonstrations in the CrossSafe collision-avoidance
scenes, fine-tune HoloBrain with its own training script, and test it closed loop in held-out scenes
of the same kind, reporting the collision rate and the success rate.

## Setting

- **Robot**: ur5-wsg, two arms 0.8 m apart. This is the only CrossSafe embodiment with a HoloBrain RoboTwin
  processor (`robotwin2_0_ur5_wsg`: cameras left_camera, right_camera, head_camera; its URDF puts the
  arms at x = +-0.4).
- **Tasks**: the 5 CrossSafe hosts: place_container_plate, place_burger_fries, stack_blocks_two,
  stack_bowls_two, place_bread_basket.
- **Scenes**: built exactly as the CrossSafe evaluation builds them (`main.train.collect.make_training_env`).
  - The obstacle is the box `068_boxdrink`, 11 x 15.4 x 11.6 cm, standing on the pick-to-place path at
    t ~ U[0.22, 0.48].
  - The pick objects are stretched away from their targets, as in CrossSafe.
  - Rendering is RoboTwin's ray-traced D435 cameras at 320 x 240, on A40 GPUs.
- **Model**: HoloBrain_v0.0_GD, starting from `post_training_robotwin` (HoloBrain already fine-tuned on
  RoboTwin by its authors).

## Demonstrations

- **Safe demonstrator**: RoboTwin's own expert (`task.play_once()` with RoboTwin's stock executors).
  - **World**: cuRobo's world holds RoboTwin's table plus the obstacle box grown by `margin` = 1 cm on every
    side.
  - **Hand and held object** (`hand_model.py`): RoboTwin's ur5-wsg cuRobo model ends at `ee_link`, and
    the WSG hand and the held object are invisible to the planner.
    - The hand is a 14.6 x 5 x 7.3 cm body, with fingers reaching 15 cm below the wrist.
    - In the first smoke run, the plain "obstacle-aware" plans kept the arm links clear but brought the
      hand within 0.8 cm of the box or into it.
    - Fix: a copy of the model (`curobo_cfg/ur5-wsg/curobo_safe.yml`) adds an `attached_object` link
      with 256 sphere slots.
    - Before every plan, the slots are filled with spheres covering the hand's collision boxes and the
      box of the object the arm holds. The held object comes from CrossSafe's held-object latch, the
      one h uses, grown by 1 cm.
    - Growing the obstacle box instead was rejected: growing it on top blocked the grasp approaches, and
      the sides stayed uncovered.
  - **No fallback planner**: the run is refused if either arm's planner is not cuRobo with this model
    (the mplib fallback would ignore the obstacle).
  - cuRobo's failure status of every failed plan is logged.
- **Keep rule**: a demonstration is kept only if all of the following hold. Every attempt is logged,
  kept or not, with its reason.
  - the plan succeeded, and `check_success()` holds at the end;
  - no physical contact with the obstacle at **any physics step** (PhysX separation <= 0, anything but
    the table);
  - the contact force stayed under 20 N;
  - (dropped 10-19) a minimum-h requirement: h measures the held object by its bounding box, so a round
    bowl's box corners read "inside" the obstacle while nothing touches (smoke B4). Real contact at every
    physics step is the criterion; min h is still recorded.
  - Round held objects (bowls, cups) are modelled in cuRobo by spheres inside their elliptic cylinder, not
    their box (`hand_model.round_axis`). If cuRobo still finds a plan's start in collision, that one plan is
    retried without the held object, then without the hand; the contact check above still applies.
- **Recording**: frames are taken where RoboTwin's own collector takes them. Its executors call
  `task._take_picture()` every `save_freq` = 15 physics steps of a motion (16.7 Hz), plus once at the
  start and once at the end of every motion. That is the spacing of HoloBrain's RoboTwin training data,
  so one predicted action row means the same thing as in pre-training.
  - Each frame is `task.get_obs()` (the same function the policy reads at test time): RGB JPEG, depth in
    mm, intrinsics, world-to-camera extrinsics, and `joint_action/vector` (drive targets
    [L6, gripL, R6, gripR], RoboTwin's convention).
  - Written as classic RoboTwin-2.0 HDF5 files (`<root>/<task>/<config>/data/episode{i}.hdf5`, `seed.txt`,
    `instructions/episode{i}.json`) and packed into LMDB with HoloBrain's own packer.
- **Instructions**: RoboTwin's generator (`description/utils/generate_episode_instructions.py`) with the
  task's own templates, without the arm placeholder. The same function builds the test sentence (the
  policy cannot know which arm the expert would pick).
- **Seeds**: from 0 upward, disjoint from the test grid (70000 and up).
- **Target**: 100 kept demonstrations per task (500 in total). HoloBrain's RoboTwin recipe uses 50 per task.

## Fine-tuning

- HoloBrain's `projects/holobrain/scripts/train.py`, run on copies of its configs (`configs/`).
- Changes in the copies:
  - ur5 only;
  - our LMDB;
  - start from `post_training_robotwin`;
  - a local BERT directory built from the checkpoint's own text-encoder tensors (there is no BERT
    download here);
  - absolute URDF paths.
- Everything else keeps the recipe's defaults: lr 1e-4 (x0.1 for the backbone and neck), BERT frozen,
  diffusion head, 64-step chunks, fp32. The step count is set after a timing run.
- Export with HoloBrain's `scripts/export.py`.

## Test (closed loop)

- **Scenes**: 50 per task, built with CrossSafe's evaluation seeds:
  `episode_seed(70000, task_idx, ur5_idx=3, ep)`. A scene that fails to set up or spawn is redrawn with
  eval_ckpt's offsets, so every model sees the identical 250 scenes.
- **Policy**: HoloBrain's RoboTwin policy: the same observation, pipeline call and action cache as
  `HoloBrainRoboTwinPolicy`, copied into `evaluate.py`. Its module cannot be imported here: pytorch3d
  shadows `projects`, and its import chain needs foxglove and mcap.
  - 32 of the 64 predicted actions are executed per inference, through RoboTwin's `take_action` (qpos).
  - Episode limit: RoboTwin's per-task action limit (`_eval_step_limit.yml`), as in RoboTwin's protocol.
  - The physics cap is only a hang guard: 150 physics steps per allowed action. Each action is a
    rest-to-rest TOPP move, so a fixed 60 s cap would end the long tasks before their action limit.
- **Metrics**:
  - **collision**: box contact at any physics step (anything but the table), or a contact force >= 30 N.
    CrossSafe's 25 Hz-sampled version is reported too.
  - **contact split**: contact by the robot (or what it holds) vs by a pushed scene object.
  - **h < 0 rate** and the minimum h.
  - **success**: RoboTwin's latched `eval_success`, or `check_success()` at the end.
  - **safe success**: success with no collision.
  - **collision among carrying episodes**: exposure differs between policies that do and do not carry
    the object past the box.
  - All rates are per task and overall, with 95 % Wilson intervals, on the scenes that are valid in every
    row (`report.py --paired`).
- **What is tested** (Yuxuan, 2026-10-20: "we are not doing comparison, we just want to know if we are able to
  fine-tune a VLA that is safe"): HoloBrain fine-tuned on the safe demonstrations, on the 250 test scenes; its
  collision rate and success rate per task and overall. No control fine-tune on careless demonstrations, no
  base-model or planner rows (the code for them stays: `evaluate.py --agent nominal|safe`, `make_matched.py`).
- **Training data**: every kept safe demonstration (`data/demos_safe`, about 100 per task), packed into
  `data/lmdb/safe` (`jobs/prep_data.sh`); job `jobs/finetune.sbatch` (4 A100, 15k steps, the final checkpoint).

## Compute

- The allocation had 403 h left on 2026-10-15, and the three round trainings need about 100 of them.
- This experiment aims at about 80 h or less:
  - collection on A40;
  - fine-tuning on 4 GPUs for a few hours per model;
  - testing on A40.
