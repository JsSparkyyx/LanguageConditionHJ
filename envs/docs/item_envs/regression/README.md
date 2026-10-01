# Regression references — DONE (GPU job, 2026-10-01), one item PENDING

Fixed-seed nominal references for `place_empty_cup`, produced by
`envs/scripts/item_envs/regression_trace.py` (replaces the stale
`scripts/run_smoke.sh`, whose `ROOT=` points at a dead autodl path). Re-run
after the Phase 5 edits: the old-task `h` traces must be IDENTICAL.

## 1. h traces — `ur5-wsg/`, `franka-panda/`

Command (cwd `/work/nvme/bifq/project/LanguageConditionHJ`):

```
/work/nvme/bifq/conda/envs/RoboTwin/bin/python envs/scripts/item_envs/regression_trace.py \
    --embodiment <emb> --task place_empty_cup --seed 0 --out envs/docs/item_envs/regression
```

Seed derivation: `episode_seed(seed_base=0, task_idx=0, emb_idx, ep=0)`;
`place_empty_cup` is not in `eval_ckpt.ALL_TASKS` so task_idx=0; scene seeds
10000 (franka-panda) / 30000 (ur5-wsg). `trace.npz` stores `h_scaled` (h × 20)
and `h_m` (metres); meta.json reports metres.

| embodiment | seed | ticks | min_h (m) | mean_h (m) | max_h (m) | success | wall (s) |
|---|---|---|---|---|---|---|---|
| ur5-wsg | 30000 | 143 | -0.0277 | 0.0730 | 0.1689 | false | 88.6 |
| franka-panda | 10000 | 149 | -0.0388 | 0.0011 | 0.0417 | false | 94.2 |

Planner `CuroboPlanner` both runs. Full numbers: `<emb>/meta.json`.

## 2. Workspace sweeps + measured embodiment facts — `../../../main/item_envs/configs/`

`workspace_{ur5-wsg,franka-panda}.npz` (left-arm IK sweep, 2 cm grid x61 × y36,
z = table + {0.02, 0.10, 0.20} m, top-down quat (-0.5,0.5,-0.5,-0.5) wxyz,
CuroboPlanner, 0 planner exceptions). Per-layer feasible fractions: ur5-wsg
0.603 / 0.587 / 0.542; franka-panda 0.440 / 0.473 / 0.500. Right arm = mirror
about x=0 (dual_arm: False). z-mode caveat: layers are gripper-pose origins,
fingertips 0.12 m lower (`--z-mode tcp` variant exists).

`facts_{ur5-wsg,franka-panda}.json`: measured arm base poses (ur5 arm_distance
0.8 from collect.py:123 — left (-0.4,-0.65,0.65) / right (0.4,-0.65,0.75),
identity quats; franka arm_distance 0.6 — x=∓0.3, y=-0.65, z=0.75, yaw-90 quat),
DoF/velocity limits/gripper openings (all confirm the URDF parse in
`../recon.md` Q10), approach_to_tcp 0.12 m, table geometry.

## 3. Segmentation runtime check — STILL PENDING

The GPU jobs ran without an RT render pass, so
`Segmentation[..., 1] == entity.per_scene_id` remains unverified at runtime
(static evidence in `../recon.md` Q8). Any future GPU job that renders can
close this with a one-frame probe: spawn an item, take the head-camera
Segmentation picture, compare the raw channel-1 mask against the item's
`per_scene_id`.
