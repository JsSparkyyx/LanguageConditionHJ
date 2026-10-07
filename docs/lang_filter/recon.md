# Phase 0 recon (VANILLA_PLAN Section 3.4)

Date: 2026-10-06 (overnight run). Raw outputs: `runs/lang_recon/` (gitignored):
`check_parity.json`, `check_text.json`, `check_crosssafe.json`, `roles/*.jsonl`, `filtered/*.json`,
`tests/*.log`, `collect/<embodiment>_<task>/` (28 recon episodes), `collect_report_recon.md`.

## Summary

| check | result |
|---|---|
| 1. env tests | CPU suite: **417 passed** (`pytest tests/item_envs -m "not sim and not gpu"`). GPU/sim suites: see the table in check 1 (run in the first A40 collection job) |
| 2. lockstep parity on 28 nominal episodes (2 per task and embodiment) | **pass**: `StepBuffer.h == h_rules[active]` (min over valid rules when active = -1) with max abs difference **0.0**; `len(buffer) == len(sidecar)` in all 14 roots; `t_in_ep` runs 0, 1, 2... per episode; done flag only on each episode's last row; `episodes/{id}.json` exists for every episode; `fails_soon` equals a recomputation from `h_rules` (25 ticks) |
| 3. episode length and timing | max 703 ticks over 4340 calibration episodes, no timeouts at 750 ticks: **`max_steps_per_episode = 750`** kept (FrozenConfig's 450 would cut T2 / T6, mean ~430 ticks). Wall time ~0.9 s per tick with 7 simulations per A40 |
| 4. HoloBrainText vs the cache | **pass in fp16 precision, fail as literally specified**: max abs difference 0.028, max relative difference **4.8e-4** (= fp16 half step 2^-11). The word features reach \|x\| = 150 (RMS 9), so fp16 storage alone rounds by up to 0.06 absolute; the check now reports both and passes on relative < 1e-3. Masks identical |
| 5. filtered base controller + `wire_item_eval` | see check 5 (run in the first A40 collection job) |
| 6. in-the-way validation, 170 nominal render-none episodes | **pass** (no benign-designated consequence violated in > 20% of runs; no harm-designated one below 20%). Findings for the human below |
| 7. `token_masks` row order and `ref_rows` | **pass**: rows = record objects in order, then regions that are nameable and have an entity (`masks.item_masks`); every rule's referent rows are non-empty at t = 0 in all 13 (task, consequence) groups |
| Section 2 regression: `main/train/train.py` 20 grad steps | **pass** (`scripts/lang/crosssafe_regression.py`: 20 steps, loss_q 0.092 -> 0.014, all finite). No CrossSafe dataset exists on Delta, so a recon StepBuffer (same format) was used |

STOP gate (2, 4 or 5 fail, or benign-designated violation > 20%): 2 passes, 4 passes in fp16 precision,
6 passes; 5 pending at the time of writing (see below).

## 1. Env tests

CPU: `cd envs && pytest tests/item_envs -m "not sim and not gpu"` -> 417 passed, 399 deselected (45 s).

GPU / sim (`test_phase5_dryrun.py`, `test_phase5_regression.py`, the 7 `*_sim.py`): results in
`runs/lang_recon/tests/<name>.rc` / `.log`, filled in below when the job ran.

<!-- GPU_TESTS -->

## 2. Lockstep parity

`python -m main.lang.recon parity runs/lang_recon/collect/*` (episodes collected with the Section 7.1 loop,
`python -m main.lang.collect recon`). All 14 roots pass every item; see `check_parity.json`.

## 3. Episode length and timing

Calibration episodes (`envs/runs/item_envs/calib_2026-10-05*`, render none, 750-tick cap):

| task | ur5 mean / p95 / max | franka mean / p95 / max |
|---|---|---|
| T1 selective_avoid | 226 / 357 / 427 | 240 / 385 / 703 |
| T2 granular_pour | 432 / 476 / 491 | 424 / 498 / 533 |
| T3 push_zone | 206 / 321 / 582 | 219 / 374 / 561 |
| T4 granular_carry | 238 / 384 / 502 | 244 / 394 / 438 |
| T5 tray_transport | 214 / 236 / 244 | 210 / 234 / 236 |
| T6 clean_pour | 437 / 478 / 497 | 432 / 509 / 561 |
| T7 gap_place | 376 / 406 / 414 | 379 / 405 / 421 |

No episode reached the cap. `LangConfig.max_steps_per_episode = 750` (30 s).

Wall time with rendering and encoding every tick, 7 processes per A40 (the recon collection): per tick
rendering 315 ms, encoding 238 ms, h including the per-tick segmentation masks 296 ms, physics 8 ms,
planning / control / python 18 ms; 43 s to build an episode (env + cuRobo warmup); ~300 s per episode.
GPU memory: 4-6 GB per simulation process, up to 9.4 GB with rendering and HoloBrain.

Consequences for Phase 1 (deviations recorded in `collect_report.md` too):
- warmup shards are collected **without per-tick segmentation masks** (`--no-masks`, ~30% faster). Only
  validation data feeds `ref_mask` (probes and the offline probe use validation only); validation keeps them;
- shards are **resumable at episode boundaries**: a shard of 14 episodes takes about 70 min, longer than
  the 1-hour interactive slot (the batch partition was estimated to start in two weeks). An interrupted
  episode is rolled back (`discard_to` on both rings) and collected again with the same seed, i.e. the same
  scene. Nothing else changes (ids, schedule, records);
- 5 shards per A40 (memory).

Training throughput (A100 40 GB, fp32, B = 64, recon data): 0.75 steps/s, 22.1 GB peak. See `phase2.md`.

## 4. Text encoder vs cache

`build_text_cache.py` over the 28 recon episodes: 6111 sentences. Five random sentences re-embedded:
max abs difference 0.028, max relative 4.8e-4, masks identical; padding to a longer batch changes the
features by 5e-5. The cached features are large (RMS 9, outlier dimensions with RMS 76, max 150), so the
plan's absolute 1e-3 cannot hold in fp16. A modeling note for the human: the text adapter's input
projection is the identity for `holobrain_bert` (plan 5.3), so the adapted words enter the body-reads-words
and scene-reads-words attentions as keys and values at that scale, while every other attention input is
RMS-normalized to ~1.

## 5. Filtered base controller

`python -m main.lang.recon filtered --task <t> --embodiment <e>`: a random-init CrossSafe model
(PolicyEncoder + TokenActor + TwinCritic), `wire_item_eval(ro, primary)`, one filtered episode per task and
embodiment; the check is that the h_fn runs every tick and `diag["margins"]` has one entry per rule.

<!-- FILTERED -->

## 6. In the way (`main/lang/scene_roles.py`)

`python -m main.lang.recon roles`: 10 seeds per task and embodiment, 25 for T1 (5 per host), render none,
nominal runs of the standalone runner. Violation rate of each designated rule in its nominal run
(rate, number of rules):

| task | embodiment | consequence | harm | benign | unclear |
|---|---|---|---|---|---|
| T1 | franka | avoid | 0.73 (11) | 0.00 (47) | 0.79 (29) |
| T1 | franka | stay | 0.55 (11) | 0.00 (47) | 0.69 (29) |
| T1 | ur5 | avoid | 0.85 (13) | 0.00 (46) | 0.68 (41) |
| T1 | ur5 | stay | 0.69 (13) | 0.00 (46) | 0.59 (41) |
| T2 | franka / ur5 | contain_particles | 0.90 / 1.00 (10) | – | – |
| T3 | franka | stay / upright / zone_exclude | 0.80 / 0.80 / 1.00 | 0.00 / 0.00 / 0.00 | – |
| T3 | franka | zone_include | – | 0.00 (10) | – |
| T3 | ur5 | stay / upright / zone_exclude | 1.00 / 1.00 / 1.00 | 0.20 / 0.20 / 0.00 | – |
| T3 | ur5 | zone_include | – | 0.20 (10) | – |
| T4 | franka / ur5 | contain_particles | 0.30 / 0.50 (10) | – | – |
| T5 | franka / ur5 | upright | 0.50 / 0.70 (10) | 0.06 (17) / 0.00 (15) | – |
| T6 | franka / ur5 | contain_particles | 0.90 / 1.00 (10) | – | – |
| T6 | franka / ur5 | exclude_particles | 0.70 / 0.60 (10) | 0.00 / 0.00 | 0.00 / 0.00 (8) |
| T7 | franka / ur5 | stay | 0.90 / 1.00 (10) | 0.00 (14) / 0.00 (13) | – |
| T7 | franka / ur5 | upright | 0.40 / 0.60 (10) | 0.00 | – |

T1 per host (avoid, harm / unclear): place_bread_basket 0.75-1.00 / 0.44-0.50, place_burger_fries 1.00 /
0.71-1.00, place_container_plate 0.50 / 0.14-0.67, stack_blocks_two – / 1.00, stack_bowls_two 1.00 (1) /
1.00. The two stacking hosts almost never have an `on_path` candidate, so their candidates are `near_path`
(unclear by the plan) and are hit in 78-100% of runs.

Primary rule sampleable: T1 0.68 (franka) / 0.88 (ur5), T3 0.80 / 0.50 (pair rules), all others 1.00.
Errors: 3 of 50 T1 runs (all `place_container_plate`): RoboTwin's grasp planner returns no pose
(`AssertionError: target_pose cannot be None for move action`, `_base_task.py:1212`); the recon collection hit
the same failure once. About 30% of `place_container_plate` episodes end this way.

Findings for the human:
- **T1 near-path candidates** are designated unclear (never named) but are violated in 59-79% of nominal
  runs, and in 78-100% on the stacking hosts, where on-path candidates are rare. Promoting `near_path` to
  harm would give T1 many more harm runs; keeping the plan's rule leaves the stacking hosts with almost
  no harm-designated rules.
- **T3 ur5 benign** stay / upright / zone_include are violated in 1 of 5 / 1 of 5 / 2 of 10 runs, exactly
  at the 20% line.
- **T4** (0.30 / 0.50) and **T7 upright** (0.40 / 0.60) are the weakest harm designations; all are above
  the 20% floor.
- **`place_container_plate` grasp failures** (~30% of that host's episodes): an env-side issue, not
  changed (envs/ is read-only for this work).

## 7. Token masks

`ref_rows(record, rule, region_rows)` (`scene_roles.py`) maps a rule to its `token_masks` rows: the subject
object's record index (avoid / upright / stay), the bead set's (contain_particles), and the region's row
`K + j` among nameable regions with an entity (zone_exclude, exclude_particles, zone_include).
`mask_region_rows(runner)` is stored per episode in `lang_episodes.jsonl` because the record does not say
which regions have an entity. Every referent is visible at t = 0 (13 of 13 groups, 1.00).

## Delay audit (preview for Phase 1)

From the same 170 traces (`main/lang/collect_report.py`, time from the start of the risky segment to the
first violation): medians 0.60 s (T7 upright) to 3.4 s (T3 upright); every task is above the 0.4 s flag,
roughly the 10-tick horizon of gamma = 0.9. Full table in `collect_report.md`.
