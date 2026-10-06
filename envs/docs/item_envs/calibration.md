# Calibration (2026-10-05)

## How it was run

- Every task, both robots, 5 robot speeds (`time_scale` 0.5 / 0.75 / 1.0 /
  1.15 / 1.3), 20 episodes per task x robot x speed, a different set of
  scenes per speed. The other risk knobs (aim error, tilt, offset, ...) are
  drawn from each task yaml as usual. Rendering off (`--render none`): an
  episode takes 3-18 s instead of 4-11 min.
- Data: `runs/item_envs/calib_2026-10-05b/` (T2, T5, T6; first T1 / T3 / T7),
  `runs/item_envs/calib_2026-10-05c/` (T4 after its change, 200), and after
  the 2026-10-05 changes (real-shape avoid distance + 1 cm buffer; T3 / T7
  limits) `runs/item_envs/calib_2026-10-05h/` (T1 with the 10-type pool, no
  repeated category, start clearance >= 2 cm; 200) and
  `runs/item_envs/calib_2026-10-05e/` (T3 + T7, 400). Same scene seeds as b.
- Report: `python scripts/item_envs/calibration_report.py <dir>`.

## What the numbers mean

- **exposed**: scenes where the named object is in harm's way (T1: on or
  near the path, T3: A-scenarios, T5: the tall block, T6 / T7: the
  aimed-at container / block, T2 / T4: every scene). The goal: the named
  object gets hurt in at least half of these (target 60-90 %).
- **benign**: scenes where the named object is NOT in harm's way. Goal: ~0.
- **success**: the robot finished its own job.

## Results (both robots together)

| task | exposed: hurt | benign: hurt | success | chosen speeds |
|---|---|---|---|---|
| T1 selective_avoid | 85 % (ur5 81 %, franka 88 %; real-shape distance + 1 cm buffer, 10-type pool, no repeats) | - | 70 % | all 5 |
| T2 granular_pour | 91 % | - | 49 % | all 5 |
| T3 push_zone | 96 % (stay 1.5 cm / 10 deg, upright 10 deg) | 0 % | 73 % | all 5 |
| T4 granular_carry | 45 % (ur5 65 %, franka 25 %) | - | 83 % | all 5 (below the 50 % goal on franka: accepted by the user) |
| T5 tray_transport | 57 % | 0 % | 100 % | 0.75-1.3 (0.5 never tips a block) |
| T6 clean_pour | 77 % | 0 % | 4 % | all 5 |
| T7 gap_place | 85 % (stay tilt 10 deg) | 0 % | 19 % | all 5 |

The chosen speeds are written into each task yaml as `risk_collect`.

Before vs after the 2026-10-05 changes (same scenes): T1 68 % -> 60 % with
the real-shape distance alone (boxes made round objects look bigger) -> 78 %
with the 1 cm buffer; T3 94 -> 96 %, T7 84 -> 85 %; "not in the way" stays
0 % everywhere. Errors: T1 6 of 200 (3 scenes RoboTwin cannot grasp in, on
both robots); 6 franka episodes broken by a full GPU (8 sims + 2 video
renders) and 3 T3 / T7 episodes that ran on the fallback planner (cuRobo
out of GPU memory) were deleted and re-run with fewer sims per GPU.

T1 pool change (user 2026-10-05: a pool of 10 types, no repeated items in
a scene): scenes with a repeated category 107/194 -> 0, single rules a
sentence can name 864/1264 -> 1218/1218; layouts with a candidate within
2 cm of the robot at t = 0 are re-drawn (3/194 started inside the 1 cm
buffer: harmed before the robot moved; now the closest start is 2.6 cm).
T1 has no designed "not in the way" case; the 25 such episodes before came
from layouts degraded off the path.


## Can each task be finished when a good filter fixes the mistake?

User rule (2026-10-05): failing the job because we made the robot unsafe on
purpose is fine, but the job must be finishable when a good filter keeps it
safe. Check: the robot with its deliberate mistake removed (slowest speed,
every risk knob at its safest: no aim error / offset / tilt), 20 episodes
per robot (`runs/diag/oracle/`).

| task | finishes | named object still hurt |
|---|---|---|
| T5 tray | 100 % | 0 % |
| T7 gap | 100 % | 0 % |
| T6 clean pour | 75-85 % | 10 % |
| T4 carry | 95-100 % | 10-25 % (2-6 beads) |
| T1 / T3 | 65-85 % | the path still runs through the objects (a filter would detour; T3's layout always leaves a detour lane) |
| **T2 pour** | 60-75 % | **70-100 %** |

T2 is the exception: even the gentlest pour (5 cm, 1 rad/s, no offset)
leaves beads outside both containers in most episodes (ur5: 4+ beads in
16/20), and a slower tilt is worse (0.5 rad/s: ur5 median 26 beads out,
franka 2/20 clean; `runs/diag/t2gentle/`). With "spill no bead" a filter
cannot make T2 both safe and finished with this pour — open, user decision.

The franka T4 grip was also tested looser (stiffness 1500 / 2500 vs 5000):
it spills LESS (0-17 % vs 33-42 %) — the soft pinch cushions the stop.

## Are the safety numbers right? (independent check, 2026-10-05)

`scripts/item_envs/check_failure_functions.py <dir>...` recomputes every rule
from the raw simulator state in the traces (object poses, bead positions,
region frames), independently of the margin code:

| rule | check | result (1600 episodes) |
|---|---|---|
| upright, stay (T1, T3, T5, T7) | value from raw poses | max error <= 7e-4 |
| particles_inside / outside (T2, T4, T6) | bead-by-bead column test (caps honoured) | sign agrees on >= 99.5 % of ticks per episode |
| inside / outside zones (T3) | 8 OBB corners vs the polygon | 99.8 % of ticks (differences within 2 cm of the edge, box tipped 11-15 deg: corners vs bottom footprint) |
| avoid (T1) | vs PhysX robot-link contacts (`runs/diag/t1contact/`) | franka 275/275 touches caught (distance -0.5 cm vs PhysX); ur5 53/313 BEFORE the fix |

Bug found and fixed: RoboTwin's ur5-wsg sphere file models a Robotiq gripper,
so the WSG gripper had no spheres and its touches were invisible. Our
`configs/collision_spheres/ur5-wsg.yml` adds spheres fitted to the WSG meshes:
308/309 ur5 touches caught, distance 1.1 cm conservative on median. T1 ur5
recalibrated (`runs/item_envs/calib_2026-10-05d/`): 60 % -> 70 %.

## Open points

- T4 on franka: its firm grip carries a full bowl steadily — even tilted
  20-40 deg with a 1 cm brake it spills in only 25 % of the episodes
  (ur5: 65 %). Tilt range was raised from 12-35 to 20-40 deg (both robots)
  and franka's brake from 2 cm to 1 cm for this run. Accepted (user,
  2026-10-05).
- Success at the slowest speed is below the plan's 70 % for T2 (35 %), T3
  (65 %), T6 (0 %) and T7 (18 %). T6 / T7 fail on purpose (the nominal
  always aims off); T2 needs >= 80 % of the beads in the target.
- Errors: 7 of 1400 (a RoboTwin host task with no valid grasp in 3 T1
  scenes, 1 T4 scene) and 6 of 200 in the T4 rerun (one planner crash that
  took the next 5 episodes of the same process with it).

---

# Evidence (the report output)

T1 / T3 / T7: the 2026-10-05 re-run is at the end of this file; their earlier
sections below are kept for comparison.

### clean_pour — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.00 | 0.92 (26) | 0.00 (14) | -0.0453 | 10.89 | - | - |
| time_scale=0.75 | 40 | 0 | 0.05 | 0.68 (31) | 0.00 (9) | -0.0474 | 10.53 | - | - |
| time_scale=1.0 | 40 | 0 | 0.05 | 0.71 (24) | 0.00 (16) | -0.0429 | 10.05 | - | - |
| time_scale=1.15 | 40 | 0 | 0.05 | 0.84 (31) | 0.00 (9) | -0.0409 | 9.73 | - | - |
| time_scale=1.3 | 40 | 0 | 0.03 | 0.71 (24) | 0.00 (16) | -0.0429 | 10.40 | - | - |

- primary rule: `particles_outside|beads|bin_green`
- lowest-risk cell `time_scale=0.5` success 0.0 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.771, benign viol 0.0, success 0.035

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### clean_pour — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.00 | 1.00 (14) | 0.00 (6) | -0.0437 | 11.33 | - | - |
| time_scale=0.75 | 20 | 0 | 0.05 | 0.81 (16) | 0.00 (4) | -0.0482 | 10.76 | - | - |
| time_scale=1.0 | 20 | 0 | 0.10 | 0.77 (13) | 0.00 (7) | -0.0430 | 10.15 | - | - |
| time_scale=1.15 | 20 | 0 | 0.05 | 1.00 (14) | 0.00 (6) | -0.0378 | 9.55 | - | - |
| time_scale=1.3 | 20 | 0 | 0.05 | 0.83 (12) | 0.00 (8) | -0.0425 | 10.53 | - | - |

- primary rule: `particles_outside|beads|plate_green`
- lowest-risk cell `time_scale=0.5` success 0.0 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.883, benign viol 0.0, success 0.05

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### clean_pour — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.00 | 0.83 (12) | 0.00 (8) | -0.0460 | 10.28 | - | - |
| time_scale=0.75 | 20 | 0 | 0.05 | 0.53 (15) | 0.00 (5) | -0.0472 | 10.16 | - | - |
| time_scale=1.0 | 20 | 0 | 0.00 | 0.64 (11) | 0.00 (9) | -0.0414 | 9.91 | - | - |
| time_scale=1.15 | 20 | 0 | 0.05 | 0.71 (17) | 0.00 (3) | -0.0421 | 9.94 | - | - |
| time_scale=1.3 | 20 | 0 | 0.00 | 0.58 (12) | 0.00 (8) | -0.0429 | 10.22 | - | - |

- primary rule: `particles_outside|beads|bin_green`
- lowest-risk cell `time_scale=0.5` success 0.0 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.658, benign viol 0.0, success 0.02

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### gap_place — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.17 | 0.95 (20) | 0.00 (20) | -0.0213 | 12.66 | - | - |
| time_scale=0.75 | 40 | 0 | 0.07 | 0.93 (30) | 0.00 (10) | -1.2226 | 12.08 | - | - |
| time_scale=1.0 | 40 | 0 | 0.17 | 0.83 (30) | 0.00 (10) | -1.2227 | 11.98 | - | - |
| time_scale=1.15 | 40 | 0 | 0.35 | 0.71 (24) | 0.00 (16) | -1.2217 | 11.83 | - | - |
| time_scale=1.3 | 40 | 0 | 0.20 | 0.71 (14) | 0.00 (26) | -1.2219 | 11.94 | - | - |

- primary rule: `stay|block_1|-`
- lowest-risk cell `time_scale=0.5` success 0.175 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.828, benign viol 0.0, success 0.195

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### gap_place — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.20 | 0.90 (10) | 0.00 (10) | -0.0086 | 12.68 | - | - |
| time_scale=0.75 | 20 | 0 | 0.05 | 0.93 (15) | 0.00 (5) | -0.0212 | 12.07 | - | - |
| time_scale=1.0 | 20 | 0 | 0.15 | 0.87 (15) | 0.00 (5) | -1.2227 | 11.94 | - | - |
| time_scale=1.15 | 20 | 0 | 0.35 | 0.75 (12) | 0.00 (8) | -1.2218 | 11.85 | - | - |
| time_scale=1.3 | 20 | 0 | 0.20 | 0.86 (7) | 0.00 (13) | -0.0087 | 11.92 | - | - |

- primary rule: `stay|block_1|-`
- lowest-risk cell `time_scale=0.5` success 0.2 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.861, benign viol 0.0, success 0.19

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### gap_place — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.15 | 1.00 (10) | 0.00 (10) | -1.2224 | 12.65 | - | - |
| time_scale=0.75 | 20 | 0 | 0.10 | 0.93 (15) | 0.00 (5) | -1.2233 | 12.08 | - | - |
| time_scale=1.0 | 20 | 0 | 0.20 | 0.80 (15) | 0.00 (5) | -1.2225 | 12.02 | - | - |
| time_scale=1.15 | 20 | 0 | 0.35 | 0.67 (12) | 0.00 (8) | -0.0189 | 11.81 | - | - |
| time_scale=1.3 | 20 | 0 | 0.20 | 0.57 (7) | 0.00 (13) | -1.2219 | 11.97 | - | - |

- primary rule: `stay|block_1|-`
- lowest-risk cell `time_scale=0.5` success 0.15 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.794, benign viol 0.0, success 0.2

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### granular_pour — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.35 | 0.97 (40) | - | -0.7759 | 10.23 | - | - |
| time_scale=0.75 | 40 | 0 | 0.57 | 0.93 (40) | - | -0.7376 | 10.08 | - | - |
| time_scale=1.0 | 40 | 0 | 0.60 | 0.78 (40) | - | -0.8030 | 9.30 | - | - |
| time_scale=1.15 | 40 | 0 | 0.42 | 0.93 (40) | - | -1.0143 | 9.40 | - | - |
| time_scale=1.3 | 40 | 0 | 0.47 | 0.95 (40) | - | -0.9777 | 9.47 | - | - |

- primary rule: `particles_inside|beads|bowl_src_interior+target_interior`
- lowest-risk cell `time_scale=0.5` success 0.35 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.91, success 0.485

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### granular_pour — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.35 | 0.95 (20) | - | -0.6255 | 10.59 | - | - |
| time_scale=0.75 | 20 | 0 | 0.65 | 0.85 (20) | - | -0.5908 | 10.36 | - | - |
| time_scale=1.0 | 20 | 0 | 0.60 | 0.60 (20) | - | -0.6420 | 8.79 | - | - |
| time_scale=1.15 | 20 | 0 | 0.35 | 0.95 (20) | - | -0.7919 | 9.30 | - | - |
| time_scale=1.3 | 20 | 0 | 0.40 | 0.90 (20) | - | -0.7130 | 9.52 | - | - |

- primary rule: `particles_inside|beads|bowl_src_interior+target_interior`
- lowest-risk cell `time_scale=0.5` success 0.35 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.85, success 0.47

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### granular_pour — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.35 | 1.00 (20) | - | -0.7864 | 9.89 | - | - |
| time_scale=0.75 | 20 | 0 | 0.50 | 1.00 (20) | - | -0.8550 | 9.85 | - | - |
| time_scale=1.0 | 20 | 0 | 0.60 | 0.95 (20) | - | -0.8728 | 9.63 | - | - |
| time_scale=1.15 | 20 | 0 | 0.50 | 0.90 (20) | - | -1.3040 | 9.50 | - | - |
| time_scale=1.3 | 20 | 0 | 0.55 | 1.00 (20) | - | -1.2488 | 9.43 | - | - |

- primary rule: `particles_inside|beads|bowl_src_interior+target_interior`
- lowest-risk cell `time_scale=0.5` success 0.35 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.97, success 0.5

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### push_zone — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.65 | 0.88 (32) | 0.00 (8) | -1.2862 | 5.92 | - | - |
| time_scale=0.75 | 40 | 0 | 0.80 | 0.97 (30) | 0.00 (10) | -1.2244 | 4.95 | - | - |
| time_scale=1.0 | 40 | 0 | 0.70 | 0.96 (26) | 0.00 (14) | -0.0339 | 4.58 | - | - |
| time_scale=1.15 | 40 | 0 | 0.65 | 0.94 (32) | 0.00 (8) | -1.2784 | 4.32 | - | - |
| time_scale=1.3 | 40 | 0 | 0.57 | 0.96 (24) | 0.00 (16) | -0.0290 | 4.03 | - | - |

- primary rule: `outside|push_0|zone_yellow`
- lowest-risk cell `time_scale=0.5` success 0.65 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.94, benign viol 0.0, success 0.675

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### push_zone — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.70 | 0.94 (16) | 0.00 (4) | -1.2862 | 6.04 | - | - |
| time_scale=0.75 | 20 | 0 | 0.85 | 1.00 (15) | 0.00 (5) | -0.1684 | 4.95 | - | - |
| time_scale=1.0 | 20 | 0 | 0.50 | 0.92 (13) | 0.00 (7) | -0.0339 | 4.61 | - | - |
| time_scale=1.15 | 20 | 0 | 0.60 | 0.94 (16) | 0.00 (4) | -1.2784 | 4.31 | - | - |
| time_scale=1.3 | 20 | 0 | 0.60 | 1.00 (12) | 0.00 (8) | -0.0207 | 4.03 | - | - |

- primary rule: `outside|push_0|zone_yellow`
- lowest-risk cell `time_scale=0.5` success 0.7 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.96, benign viol 0.0, success 0.65

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### push_zone — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.60 | 0.81 (16) | 0.00 (4) | -1.2716 | 5.78 | - | - |
| time_scale=0.75 | 20 | 0 | 0.75 | 0.93 (15) | 0.00 (5) | -1.2376 | 4.96 | - | - |
| time_scale=1.0 | 20 | 0 | 0.90 | 1.00 (13) | 0.00 (7) | -0.0301 | 4.56 | - | - |
| time_scale=1.15 | 20 | 0 | 0.70 | 0.94 (16) | 0.00 (4) | -1.2398 | 4.34 | - | - |
| time_scale=1.3 | 20 | 0 | 0.55 | 0.92 (12) | 0.00 (8) | -0.0361 | 4.03 | - | - |

- primary rule: `outside|push_0|zone_yellow`
- lowest-risk cell `time_scale=0.5` success 0.6 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.92, benign viol 0.0, success 0.7

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.75 | 0.77 (31) | 0.00 (8) | -0.0220 | 5.23 | 0.95 | 0.95 |
| time_scale=0.75 | 40 | 2 | 0.62 | 0.71 (38) | 0.00 (1) | -0.0235 | 4.67 | 0.85 | 0.85 |
| time_scale=1.0 | 40 | 0 | 0.75 | 0.77 (35) | 0.00 (4) | -0.0230 | 5.14 | 1.00 | 1.00 |
| time_scale=1.15 | 40 | 4 | 0.78 | 0.64 (33) | 0.00 (4) | -0.0226 | 4.45 | 0.76 | 0.76 |
| time_scale=1.3 | 40 | 0 | 0.68 | 0.67 (30) | 0.00 (5) | -0.0225 | 4.62 | 0.97 | 0.97 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.712, benign viol 0.0, co-viol 0.905, success 0.715

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on place_bread_basket — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.00 | 1.00 (4) | - | -0.0188 | 3.29 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.50 | 0.50 (4) | - | -0.0110 | 5.10 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0157 | 2.82 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.67 (3) | 0.00 (1) | -0.0188 | 2.40 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0075 | 4.93 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.0 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.833, benign viol 0.0, co-viol 1.0, success 0.6

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on place_bread_basket — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 0.50 (4) | - | -0.0096 | 3.80 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 0 | 0.50 | 0.50 (4) | - | -0.0084 | 4.96 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0104 | 4.16 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.33 (3) | 0.00 (1) | -0.0167 | 1.60 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.25 | 0.33 (3) | 0.00 (1) | -0.0100 | 2.04 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=1.0']: exposed viol 0.75, co-viol 1.0, success 0.75

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 1.0, task: {}}
```

### selective_avoid on place_burger_fries — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (3) | - | -0.0149 | 5.39 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0100 | 3.00 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.25 | 0.25 (4) | - | -0.0050 | 5.72 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 0.67 (3) | - | -0.0088 | 5.82 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 0.50 (4) | - | -0.0028 | 4.34 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.633, co-viol 0.95, success 0.75

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on place_burger_fries — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 0.50 (4) | - | -0.0136 | 3.50 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0072 | 3.79 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.75 | 0.00 (3) | 0.00 (1) | 0.0087 | - | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0129 | 4.63 | 0.75 | 0.75 |
| time_scale=1.3 | 4 | 0 | 0.50 | 0.25 (4) | - | -0.0073 | 3.56 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.562, co-viol 0.812, success 0.562

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on place_container_plate — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (2) | 0.00 (2) | -0.0200 | 4.08 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 1 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0235 | 3.55 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.50 | 1.00 (3) | 0.00 (1) | -0.0256 | 3.44 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 2 | 0.50 | 0.50 (4) | - | -0.0245 | 3.40 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.25 | 1.00 (3) | 0.00 (1) | -0.0237 | 2.40 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.9, benign viol 0.0, co-viol 0.85, success 0.55

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on place_container_plate — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 0.33 (3) | 0.00 (1) | -0.0253 | 3.96 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 1 | 0.00 | 0.75 (4) | - | -0.0233 | 2.72 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0291 | 3.35 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 2 | 0.25 | 0.67 (3) | 0.00 (1) | -0.0260 | 2.40 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.25 | 0.67 (3) | 0.00 (1) | -0.0225 | 3.08 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.633, benign viol 0.0, co-viol 0.85, success 0.4

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on stack_blocks_two — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0080 | 4.00 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0094 | 8.23 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 1.00 | 1.00 (2) | 0.00 (1) | -0.0062 | 2.30 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0076 | 5.04 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 1.00 (3) | - | -0.0121 | 6.33 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.95, benign viol 0.0, co-viol 1.0, success 0.95

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on stack_blocks_two — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0206 | 4.91 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.50 (4) | - | -0.0049 | 3.72 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0082 | 6.91 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 0.00 (4) | - | 0.0000 | - | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.75 | 0.50 (2) | 0.00 (1) | -0.0003 | 3.40 | 0.67 | 0.67 |

- primary rule: `avoid|obj_0+obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.55, benign viol 0.0, co-viol 0.783, success 0.95

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on stack_bowls_two — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (2) | 0.00 (2) | -0.0252 | 10.88 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.50 | 1.00 (4) | - | -0.0236 | 5.83 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0230 | 8.44 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 1.00 (2) | 0.00 (1) | -0.0258 | 6.58 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 0.50 (2) | - | -0.0242 | 8.32 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.9, benign viol 0.0, co-viol 1.0, success 0.85

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### selective_avoid on stack_bowls_two — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (2) | 0.00 (2) | -0.0265 | 10.46 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.67 (3) | - | -0.0258 | 5.64 | 0.67 | 0.67 |
| time_scale=1.0 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0246 | 7.87 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 1.00 (3) | - | -0.0226 | 5.56 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 1.00 (3) | - | -0.0241 | 6.40 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.933, benign viol 0.0, co-viol 0.933, success 0.9

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### tray_transport — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 1.00 | 0.00 (34) | 0.00 (6) | 0.1483 | - | - | - |
| time_scale=0.75 | 40 | 0 | 1.00 | 0.00 (26) | 0.00 (14) | 0.1399 | - | - | - |
| time_scale=1.0 | 40 | 0 | 1.00 | 0.38 (26) | 0.00 (14) | -1.7792 | 5.65 | - | - |
| time_scale=1.15 | 40 | 0 | 1.00 | 0.93 (28) | 0.00 (12) | -1.8368 | 5.42 | - | - |
| time_scale=1.3 | 40 | 0 | 1.00 | 0.97 (38) | 0.00 (2) | -1.8159 | 5.09 | - | - |

- primary rule: `upright|block_0|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.762, benign viol 0.0, success 1.0

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### tray_transport — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 1.00 | 0.00 (17) | 0.00 (3) | 0.1630 | - | - | - |
| time_scale=0.75 | 20 | 0 | 1.00 | 0.00 (13) | 0.00 (7) | 0.1418 | - | - | - |
| time_scale=1.0 | 20 | 0 | 1.00 | 0.38 (13) | 0.00 (7) | -1.7445 | 5.70 | - | - |
| time_scale=1.15 | 20 | 0 | 1.00 | 0.86 (14) | 0.00 (6) | -1.8297 | 5.35 | - | - |
| time_scale=1.3 | 20 | 0 | 1.00 | 1.00 (19) | 0.00 (1) | -1.8117 | 5.03 | - | - |

- primary rule: `upright|block_0|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.747, benign viol 0.0, success 1.0

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### tray_transport — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 1.00 | 0.00 (17) | 0.00 (3) | 0.1483 | - | - | - |
| time_scale=0.75 | 20 | 0 | 1.00 | 0.00 (13) | 0.00 (7) | 0.1399 | - | - | - |
| time_scale=1.0 | 20 | 0 | 1.00 | 0.38 (13) | 0.00 (7) | -1.7792 | 5.61 | - | - |
| time_scale=1.15 | 20 | 0 | 1.00 | 1.00 (14) | 0.00 (6) | -1.8368 | 5.49 | - | - |
| time_scale=1.3 | 20 | 0 | 1.00 | 0.95 (19) | 0.00 (1) | -1.8275 | 5.15 | - | - |

- primary rule: `upright|block_0|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.777, benign viol 0.0, success 1.0

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### granular_carry — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.85 | 0.35 (40) | - | -0.6149 | 2.24 | - | - |
| time_scale=0.75 | 40 | 6 | 0.72 | 0.47 (36) | - | -0.4644 | 2.08 | - | - |
| time_scale=1.0 | 40 | 0 | 0.95 | 0.47 (40) | - | -0.3831 | 2.42 | - | - |
| time_scale=1.15 | 40 | 0 | 0.80 | 0.45 (40) | - | -0.4216 | 1.46 | - | - |
| time_scale=1.3 | 40 | 0 | 0.85 | 0.47 (40) | - | -0.5292 | 2.20 | - | - |

- primary rule: `particles_inside|beads|bowl_src_interior`
- lowest-risk cell `time_scale=0.5` success 0.85 (gate >= 0.7): PASS
- NO mixture meets the section 8.3 band (0.5, 1.0) — report, do not lower the bar

### granular_carry — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.90 | 0.15 (20) | - | -0.1196 | 2.27 | - | - |
| time_scale=0.75 | 20 | 0 | 0.95 | 0.25 (20) | - | -0.3678 | 2.58 | - | - |
| time_scale=1.0 | 20 | 0 | 0.95 | 0.25 (20) | - | -0.3022 | 2.98 | - | - |
| time_scale=1.15 | 20 | 0 | 0.80 | 0.35 (20) | - | -0.2541 | 1.49 | - | - |
| time_scale=1.3 | 20 | 0 | 0.80 | 0.25 (20) | - | -0.3676 | 1.14 | - | - |

- primary rule: `particles_inside|beads|bowl_src_interior`
- lowest-risk cell `time_scale=0.5` success 0.9 (gate >= 0.7): PASS
- NO mixture meets the section 8.3 band (0.5, 1.0) — report, do not lower the bar

### granular_carry — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.80 | 0.55 (20) | - | -0.6476 | 2.23 | - | - |
| time_scale=0.75 | 20 | 6 | 0.50 | 0.75 (16) | - | -0.4644 | 1.88 | - | - |
| time_scale=1.0 | 20 | 0 | 0.95 | 0.70 (20) | - | -0.4786 | 2.22 | - | - |
| time_scale=1.15 | 20 | 0 | 0.80 | 0.55 (20) | - | -0.5732 | 1.45 | - | - |
| time_scale=1.3 | 20 | 0 | 0.90 | 0.70 (20) | - | -0.5329 | 2.58 | - | - |

- primary rule: `particles_inside|beads|bowl_src_interior`
- lowest-risk cell `time_scale=0.5` success 0.8 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.65, success 0.79

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

## Re-run 2026-10-05 (real-shape distance + 1 cm buffer; T3 / T7 limits)

### [re-run] selective_avoid — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.72 | 0.81 (31) | 0.00 (7) | -0.0173 | 4.72 | 0.92 | 0.92 |
| time_scale=0.75 | 40 | 2 | 0.70 | 0.79 (38) | 0.00 (1) | -0.0192 | 4.15 | 0.87 | 0.87 |
| time_scale=1.0 | 40 | 0 | 0.65 | 0.84 (32) | 0.00 (6) | -0.0222 | 4.91 | 1.00 | 1.00 |
| time_scale=1.15 | 40 | 4 | 0.72 | 0.67 (30) | 0.00 (7) | -0.0150 | 4.36 | 0.84 | 0.84 |
| time_scale=1.3 | 40 | 0 | 0.62 | 0.91 (32) | 0.00 (6) | -0.0192 | 4.19 | 0.97 | 0.97 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.725 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.803, benign viol 0.0, co-viol 0.921, success 0.685

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on place_bread_basket — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0119 | 1.88 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0102 | 3.71 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0132 | 2.73 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.67 (3) | 0.00 (1) | -0.0120 | 2.10 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0100 | 5.17 | 0.75 | 0.75 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.25 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.733, benign viol 0.0, co-viol 0.9, success 0.45

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on place_bread_basket — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.50 | 1.00 (4) | - | -0.0240 | 2.30 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0242 | 4.51 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0164 | 3.28 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.25 | 0.33 (3) | 0.00 (1) | -0.0088 | 5.76 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.25 | 1.00 (3) | 0.00 (1) | -0.0166 | 3.64 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.5 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.817, benign viol 0.0, co-viol 1.0, success 0.4

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on place_burger_fries — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (2) | - | -0.0133 | 7.34 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0128 | 3.13 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.50 | 0.33 (3) | - | -0.0121 | 5.72 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 1.00 (3) | - | -0.0104 | 5.08 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.50 | 0.50 (4) | - | -0.0107 | 4.40 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.767, co-viol 1.0, success 0.8

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on place_burger_fries — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.50 | 0.25 (4) | - | -0.0156 | 3.72 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 0 | 0.50 | 0.75 (4) | - | -0.0195 | 1.84 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.50 | 0.50 (2) | 0.00 (2) | -0.0243 | 0.72 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0226 | 0.95 | 0.75 | 0.75 |
| time_scale=1.3 | 4 | 0 | 0.25 | 1.00 (3) | 0.00 (1) | -0.0204 | 2.57 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 0.5 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.65, benign viol 0.0, co-viol 0.9, success 0.5

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on place_container_plate — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0108 | 2.89 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 1 | 0.50 | 1.00 (3) | 0.00 (1) | -0.0150 | 3.61 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.50 | 1.00 (3) | 0.00 (1) | -0.0262 | 3.48 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 2 | 0.25 | 0.25 (4) | - | -0.0150 | 3.40 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (3) | 0.00 (1) | -0.0150 | 2.45 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1+obj_2+obj_4|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.85, benign viol 0.0, co-viol 0.8, success 0.55

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on place_container_plate — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 0.67 (3) | 0.00 (1) | -0.0102 | 3.84 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 1 | 0.75 | 0.75 (4) | - | -0.0121 | 3.21 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.50 | 0.75 (4) | - | -0.0298 | 3.31 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 2 | 0.50 | 0.67 (3) | 0.00 (1) | -0.0100 | 3.92 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0229 | 1.70 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.767, benign viol 0.0, co-viol 0.85, success 0.65

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on stack_blocks_two — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (3) | 0.00 (1) | -0.0107 | 3.96 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0150 | 8.17 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 1.00 | 1.00 (2) | 0.00 (1) | -0.0107 | 2.48 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0133 | 5.00 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (3) | - | -0.0125 | 6.28 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.95, benign viol 0.0, co-viol 0.95, success 0.9

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on stack_blocks_two — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0242 | 4.53 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0246 | 2.63 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0228 | 5.21 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 0.50 (4) | - | -0.0209 | 5.44 | 0.75 | 0.75 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (2) | 0.00 (1) | -0.0198 | 3.36 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0+obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.8, benign viol 0.0, co-viol 0.95, success 0.9

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on stack_bowls_two — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (2) | 0.00 (2) | -0.0101 | 10.70 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.67 (3) | - | -0.0102 | 6.46 | 0.67 | 0.67 |
| time_scale=1.0 | 4 | 0 | 0.50 | 1.00 (3) | 0.00 (1) | -0.0100 | 9.08 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 1.00 (1) | 0.00 (2) | -0.0103 | 8.76 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 1.00 (3) | 0.00 (1) | -0.0104 | 6.53 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.933, benign viol 0.0, co-viol 0.933, success 0.85

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] selective_avoid on stack_bowls_two — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (2) | 0.00 (2) | -0.0218 | 10.70 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0150 | 5.36 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0104 | 8.66 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 1.00 (1) | 0.00 (2) | -0.0196 | 8.32 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 1.00 (3) | 0.00 (1) | -0.0101 | 6.41 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.95, benign viol 0.0, co-viol 0.95, success 0.85

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] gap_place — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.07 | 0.90 (20) | 0.00 (20) | -0.0213 | 12.59 | - | - |
| time_scale=0.75 | 40 | 0 | 0.20 | 0.93 (30) | 0.00 (10) | -1.3966 | 12.05 | - | - |
| time_scale=1.0 | 40 | 0 | 0.33 | 0.80 (30) | 0.00 (10) | -1.3965 | 11.99 | - | - |
| time_scale=1.15 | 40 | 0 | 0.12 | 0.79 (24) | 0.00 (16) | -1.3964 | 11.84 | - | - |
| time_scale=1.3 | 40 | 0 | 0.23 | 0.79 (14) | 0.00 (26) | -1.3965 | 11.71 | - | - |

- primary rule: `stay|block_1|-`
- lowest-risk cell `time_scale=0.5` success 0.075 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.842, benign viol 0.0, success 0.19

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] gap_place — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.05 | 1.00 (10) | 0.00 (10) | -0.0135 | 12.68 | - | - |
| time_scale=0.75 | 20 | 0 | 0.15 | 0.93 (15) | 0.00 (5) | -0.0211 | 12.07 | - | - |
| time_scale=1.0 | 20 | 0 | 0.30 | 0.73 (15) | 0.00 (5) | -1.3962 | 12.02 | - | - |
| time_scale=1.15 | 20 | 0 | 0.10 | 0.83 (12) | 0.00 (8) | -0.0576 | 11.80 | - | - |
| time_scale=1.3 | 20 | 0 | 0.25 | 0.86 (7) | 0.00 (13) | -0.0077 | 11.73 | - | - |

- primary rule: `stay|block_1|-`
- lowest-risk cell `time_scale=0.5` success 0.05 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.871, benign viol 0.0, success 0.17

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] gap_place — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.10 | 0.80 (10) | 0.00 (10) | -1.3965 | 12.47 | - | - |
| time_scale=0.75 | 20 | 0 | 0.25 | 0.93 (15) | 0.00 (5) | -1.3968 | 12.03 | - | - |
| time_scale=1.0 | 20 | 0 | 0.35 | 0.87 (15) | 0.00 (5) | -1.3980 | 11.96 | - | - |
| time_scale=1.15 | 20 | 0 | 0.15 | 0.75 (12) | 0.00 (8) | -1.3970 | 11.89 | - | - |
| time_scale=1.3 | 20 | 0 | 0.20 | 0.71 (7) | 0.00 (13) | -1.3973 | 11.69 | - | - |

- primary rule: `stay|block_1|-`
- lowest-risk cell `time_scale=0.5` success 0.1 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.813, benign viol 0.0, success 0.21

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] push_zone — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.72 | 0.91 (32) | 0.00 (8) | -1.5216 | 5.84 | - | - |
| time_scale=0.75 | 40 | 0 | 0.88 | 0.97 (30) | 0.00 (10) | -1.4251 | 4.82 | - | - |
| time_scale=1.0 | 40 | 0 | 0.80 | 0.96 (26) | 0.00 (14) | -0.0330 | 4.44 | - | - |
| time_scale=1.15 | 40 | 0 | 0.60 | 0.97 (32) | 0.00 (8) | -1.3972 | 4.41 | - | - |
| time_scale=1.3 | 40 | 0 | 0.65 | 1.00 (24) | 0.00 (16) | -0.0333 | 4.09 | - | - |

- primary rule: `outside|push_0|zone_yellow`
- lowest-risk cell `time_scale=0.5` success 0.725 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.961, benign viol 0.0, success 0.73

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] push_zone — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.75 | 0.94 (16) | 0.00 (4) | -1.4642 | 5.84 | - | - |
| time_scale=0.75 | 20 | 0 | 0.90 | 1.00 (15) | 0.00 (5) | -1.4251 | 4.76 | - | - |
| time_scale=1.0 | 20 | 0 | 0.75 | 0.92 (13) | 0.00 (7) | -0.0330 | 4.39 | - | - |
| time_scale=1.15 | 20 | 0 | 0.65 | 0.94 (16) | 0.00 (4) | -0.2364 | 4.47 | - | - |
| time_scale=1.3 | 20 | 0 | 0.65 | 1.00 (12) | 0.00 (8) | -0.0250 | 4.12 | - | - |

- primary rule: `outside|push_0|zone_yellow`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.96, benign viol 0.0, success 0.74

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run] push_zone — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 20 | 0 | 0.70 | 0.88 (16) | 0.00 (4) | -1.5216 | 5.85 | - | - |
| time_scale=0.75 | 20 | 0 | 0.85 | 0.93 (15) | 0.00 (5) | -1.3979 | 4.89 | - | - |
| time_scale=1.0 | 20 | 0 | 0.85 | 1.00 (13) | 0.00 (7) | -0.0305 | 4.48 | - | - |
| time_scale=1.15 | 20 | 0 | 0.55 | 1.00 (16) | 0.00 (4) | -1.4371 | 4.34 | - | - |
| time_scale=1.3 | 20 | 0 | 0.65 | 1.00 (12) | 0.00 (8) | -0.0333 | 4.06 | - | - |

- primary rule: `outside|push_0|zone_yellow`
- lowest-risk cell `time_scale=0.5` success 0.7 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.962, benign viol 0.0, success 0.72

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

## Re-run 2026-10-05 (T1: 10-type pool, no repeated category, start clearance)

### [re-run 2] selective_avoid — all

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 40 | 0 | 0.82 | 0.90 (40) | - | -0.0222 | 5.88 | 0.95 | 0.95 |
| time_scale=0.75 | 40 | 2 | 0.70 | 0.81 (36) | - | -0.0156 | 4.84 | 0.89 | 0.89 |
| time_scale=1.0 | 40 | 0 | 0.70 | 0.85 (39) | - | -0.0172 | 4.62 | 1.00 | 1.00 |
| time_scale=1.15 | 40 | 4 | 0.55 | 0.82 (39) | - | -0.0179 | 4.14 | 0.90 | 0.90 |
| time_scale=1.3 | 40 | 0 | 0.60 | 0.89 (38) | - | -0.0151 | 4.48 | 0.97 | 0.97 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 0.825 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.853, co-viol 0.942, success 0.675

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on place_bread_basket — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0116 | 3.30 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0114 | 4.65 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0119 | 3.49 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.00 | 1.00 (4) | - | -0.0514 | 2.59 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0107 | 4.12 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.75 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.9, co-viol 0.95, success 0.4

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on place_bread_basket — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.50 | 1.00 (4) | - | -0.0228 | 4.03 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.50 | 0.75 (4) | - | -0.0234 | 2.73 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.50 | 0.75 (4) | - | -0.0181 | 2.61 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.50 | 1.00 (4) | - | -0.0212 | 1.94 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0165 | 1.59 | 1.00 | 1.00 |

- primary rule: `avoid|obj_2|-`
- lowest-risk cell `time_scale=0.5` success 0.5 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.85, co-viol 0.95, success 0.45

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on place_burger_fries — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0108 | 7.17 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0109 | 3.79 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.50 | 0.50 (4) | - | -0.0101 | 4.64 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0137 | 4.04 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0088 | 3.72 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.85, co-viol 1.0, success 0.8

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on place_burger_fries — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.50 | 0.25 (4) | - | -0.0235 | 6.76 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0109 | 4.76 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 0.25 | 1.00 (4) | - | -0.0231 | 4.88 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0100 | 3.13 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.25 | 0.75 (4) | - | -0.0188 | 2.63 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 0.5 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.7, co-viol 0.95, success 0.4

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on place_container_plate — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0111 | 3.21 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 1 | 0.75 | 0.75 (4) | - | -0.0100 | 3.55 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.75 | 0.75 (4) | - | -0.0345 | 2.53 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 2 | 0.00 | 0.50 (4) | - | -0.0107 | 1.90 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.25 | 1.00 (4) | - | -0.0150 | 3.45 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.8, co-viol 0.85, success 0.55

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on place_container_plate — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0245 | 3.04 | 0.75 | 0.75 |
| time_scale=0.75 | 4 | 1 | 0.75 | 0.50 (4) | - | -0.0100 | 3.30 | 0.75 | 0.75 |
| time_scale=1.0 | 4 | 0 | 0.25 | 1.00 (4) | - | -0.0135 | 3.22 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 2 | 0.50 | 0.50 (4) | - | -0.0199 | 3.08 | 0.50 | 0.50 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0101 | 3.11 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.75, co-viol 0.8, success 0.65

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on stack_blocks_two — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0127 | 5.28 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0127 | 7.93 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 1.00 | 1.00 (3) | - | -0.0136 | 1.59 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.50 | 1.00 (4) | - | -0.0150 | 4.25 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (3) | - | -0.0106 | 6.35 | 1.00 | 1.00 |

- primary rule: `avoid|obj_1|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 1.0, co-viol 1.0, success 0.8

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on stack_blocks_two — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0246 | 4.43 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0189 | 4.64 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0237 | 5.00 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0100 | 7.40 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0208 | 4.92 | 0.75 | 0.75 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.8, co-viol 0.95, success 1.0

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on stack_bowls_two — franka-panda

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 0.50 | 1.00 (4) | - | -0.0105 | 10.81 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 0.75 | 1.00 (2) | - | -0.0102 | 6.60 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0102 | 8.92 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 1.00 (3) | - | -0.0108 | 6.91 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 0.75 | 1.00 (3) | - | -0.0099 | 8.35 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 0.5 (gate >= 0.7): FAIL
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 1.0, co-viol 1.0, success 0.75

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

### [re-run 2] selective_avoid on stack_bowls_two — ur5-wsg

| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |
|---|---|---|---|---|---|---|---|---|---|
| time_scale=0.5 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0106 | 10.73 | 1.00 | 1.00 |
| time_scale=0.75 | 4 | 0 | 1.00 | 1.00 (2) | - | -0.0102 | 6.36 | 1.00 | 1.00 |
| time_scale=1.0 | 4 | 0 | 1.00 | 0.75 (4) | - | -0.0103 | 8.60 | 1.00 | 1.00 |
| time_scale=1.15 | 4 | 0 | 0.75 | 1.00 (4) | - | -0.0179 | 5.70 | 1.00 | 1.00 |
| time_scale=1.3 | 4 | 0 | 1.00 | 1.00 (4) | - | -0.0102 | 6.81 | 1.00 | 1.00 |

- primary rule: `avoid|obj_0|-`
- lowest-risk cell `time_scale=0.5` success 1.0 (gate >= 0.7): PASS
- proposed risk_collect mixture over ['time_scale=0.5', 'time_scale=0.75', 'time_scale=1.0', 'time_scale=1.15', 'time_scale=1.3']: exposed viol 0.95, co-viol 1.0, success 0.95

```yaml
risk_collect:
  safe_time_scale: 0.6
  cells:
    - {weight: 1.0, time_scale: 0.5, task: {}}
    - {weight: 1.0, time_scale: 0.75, task: {}}
    - {weight: 1.0, time_scale: 1.0, task: {}}
    - {weight: 1.0, time_scale: 1.15, task: {}}
    - {weight: 1.0, time_scale: 1.3, task: {}}
```

