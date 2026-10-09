# Phase 0 recon (VANILLA_PLAN Section 3.4)

Date: 2026-10-06 (overnight run). Raw outputs: `runs/lang_recon/` (gitignored):
`check_parity.json`, `check_text.json`, `check_crosssafe.json`, `roles/*.jsonl`, `filtered/*.json`,
`tests/*.log`, `collect/<embodiment>_<task>/` (28 recon episodes), `collect_report_recon.md`.

## Summary

| check | result |
|---|---|
| 1. env tests | CPU suite: **417 passed**. GPU/sim: Phase-5 regression 2 passed; T3, T4, T5, T7 sim suites pass; T2 and T6 each have 1 failing single-episode test (env-side, below); the dryrun and T1 suites rerun in the collection jobs |
| 2. lockstep parity on 28 nominal episodes (2 per task and embodiment) | **pass**: `StepBuffer.h == h_rules[active]` (min over valid rules when active = -1) with max abs difference **0.0**; `len(buffer) == len(sidecar)` in all 14 roots; `t_in_ep` runs 0, 1, 2... per episode; done flag only on each episode's last row; `episodes/{id}.json` exists for every episode; `fails_soon` equals a recomputation from `h_rules` (25 ticks) |
| 3. episode length and timing | max 703 ticks over 4340 calibration episodes, no timeouts at 750 ticks: **`max_steps_per_episode = 750`** kept (FrozenConfig's 450 would cut T2 / T6, mean ~430 ticks). Wall time ~0.9 s per tick with 7 simulations per A40 |
| 4. HoloBrainText vs the cache | **pass in fp16 precision, fail as literally specified**: max abs difference 0.028, max relative difference **4.8e-4** (= fp16 half step 2^-11). The word features reach \|x\| = 150 (RMS 9), so fp16 storage alone rounds by up to 0.06 absolute; the check now reports both and passes on relative < 1e-3. Masks identical |
| 5. filtered base controller + `wire_item_eval` | **pass**: h_fn on every tick, one margin per rule, Q evaluated at every hook, in all 15 episodes that ran (one T1 run hit the RoboTwin grasp failure at tick 1) |
| 6. in-the-way validation, 170 nominal render-none episodes | **pass** (no benign-designated consequence violated in > 20% of runs; no harm-designated one below 20%). Findings for the human below |
| 7. `token_masks` row order and `ref_rows` | **pass**: rows = record objects in order, then regions that are nameable and have an entity (`masks.item_masks`); every rule's referent rows are non-empty at t = 0 in all 13 (task, consequence) groups |
| Section 2 regression: `main/train/train.py` 20 grad steps | **pass** (`scripts/lang/crosssafe_regression.py`: 20 steps, loss_q 0.092 -> 0.014, all finite). No CrossSafe dataset exists on Delta, so a recon StepBuffer (same format) was used |

STOP gate (2, 4 or 5 fail, or benign-designated violation > 20%): 2 passes, 4 passes in fp16 precision,
5 and 6 pass: no STOP condition.

## 1. Env tests

CPU: `cd envs && pytest tests/item_envs -m "not sim and not gpu"` -> 417 passed, 399 deselected (45 s).

GPU / sim (`runs/lang_recon/tests/<name>.log`):

| suite | result |
|---|---|
| `test_phase5_regression.py` (gpu) | **2 passed** (the statistical contract, both embodiments) |
| `test_phase5_dryrun.py` (gpu) | cut off by the first job's deadline; rerun in the collection jobs (`runs/lang_recon/tests/`) |
| `test_gap_place_sim.py` | 8 passed |
| `test_granular_carry_sim.py` | 12 passed |
| `test_push_zone_sim.py` | 42 passed |
| `test_tray_transport_sim.py` | 8 passed |
| `test_clean_pour_sim.py` | 7 passed, **1 failed**: `test_zero_aim_error_keeps_neighbours_clean[ur5-wsg]`. With zero aim error, beads still reached `bin_green` in that single episode |
| `test_granular_pour_sim.py` | 41 passed, **1 failed**: `test_full_episode_low_risk[ur5-wsg]`. The low-risk ur5 pour did not succeed (seed 0) |
| `test_selective_avoid_sim.py` | cut off by the first job's deadline; rerun in the collection jobs |

Both failures are single nominal episodes of the env's own tests under cuRobo's run-to-run randomness
(the ur5 pour was already the fragile one in the env calibration). They are env-side and were not
touched (envs/ is read-only for this work). For the human: rerun them, or loosen them to rates over
several seeds.

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

**Pass.** 16 episodes (7 tasks x 2 embodiments, plus T1 repeated on `place_burger_fries`): in every
episode that ran, the h_fn was called on all 751 captured ticks with one margin per rule (1 to 9 rules),
and the filter evaluated Q(s, a_nom) at every hook (241-251 times). A random-init critic gives Q < 0
everywhere, so the filter drives the arms the whole episode (intervention rate 1.0) and every episode
hits the 750-tick cap (status `timeout`). This is expected at random init. The franka T1 run on
`place_container_plate` ended at tick 1 with the RoboTwin grasp failure (`target_pose cannot be None`,
check 6); its repeat on `place_burger_fries` ran normally.

| run | status | rules | h calls | margins per rule ok | Q evaluations | intervention rate |
|---|---|---|---|---|---|---|
| T6 clean_pour franka / ur5 | timeout | 3 | 751 | yes | 241 | 1.0 |
| T7 gap_place franka / ur5 | timeout | 4 | 751 | yes | 241 | 1.0 |
| T4 granular_carry franka / ur5 | timeout | 1 | 751 | yes | 251 | 1.0 |
| T2 granular_pour franka / ur5 | timeout | 1 | 751 | yes | 241 | 1.0 |
| T3 push_zone franka / ur5 | timeout | 4 | 751 | yes | 241 | 1.0 |
| T1 place_burger_fries franka / ur5 | timeout | 9 | 751 | yes | 251 | 1.0 |
| T1 place_container_plate ur5 | timeout | 9 | 751 | yes | 251 | 1.0 |
| T1 place_container_plate franka | exception at tick 1 (grasp planner) | 9 | 1 | yes | 0 | – |
| T5 tray_transport franka / ur5 | timeout | 2 | 751 | yes | 241 | 1.0 |


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

## V5 (2026-10-09)

FUSION_PLAN step 1 (recon) and R2, for the frozen fusion stage. Checked on the login node on CPU: the
safetensors header, a CPU build of the module, numpy cosines. No GPU, no simulator. Line anchors for every
fact: `fusion_impl.md` section 1.

| check | result |
|---|---|
| `feature_enhancer.*` in `pretrain/model.safetensors` | **277 keys, all F32**, 21,906,688 parameters (1,226 keys in the file). Per layer (6 layers): 18 `text_img_attn_blocks`, 16 `img_attn_blocks`, 12 `text_attn_blocks`; plus `level_embed` [4, 256] |
| strict load | **pass**: `build(cfg["feature_enhancer"])` (`TextImageDeformable2DEnhancer`, from `model.config.json`) loads the 277 tensors with `strict=True`, no missing or unexpected keys. `pre_spatial_enhancer = False`: HoloBrain runs the fusion before the spatial enhancer |
| layer scales γ_v, γ_l (init 1e-4) | trained but small: mean \|γ_v\| 0.013-0.030 per layer (max 0.078), mean \|γ_l\| 0.011-0.039 (max 0.109) |
| D1 head-noun fix in `language._answers` | 0 of 2,806 stored `sampleable` flags flip on the 840 on-disk records (warmup 600, val 100, val20 100, val_grounding 40); env CPU suite 424 passed |

### Provenance of the fusion weights (R2)

Cosine similarity per tensor (a random pair of matrices this size gives about 0). Key sets, shapes and the
`feature_enhancer` config entry are identical in the three HoloBrain / BIP3D checkpoints.

| pair | cosine |
|---|---|
| `pretrain` vs `post_training_robotwin` (HoloBrain's RoboTwin stage) | median 0.986 over the 277 tensors, 0.95-0.98 on the matrices; none identical (min 0.65) |
| `pretrain` vs `BIP3D_Tiny_Det` (Hugging Face, a detection model) | median 0.903 (min 0.488); 2-D weights median 0.838 (min 0.561), ‖Δ‖/‖b‖ median 0.57 |
| `post_training_robotwin` vs `BIP3D_Tiny_Det` | median 0.886 |
| `pretrain` vs `IDEA-Research/grounding-dino-tiny` | fusion `v_proj` / `l_proj` 0.885-0.918, `sampling_offsets` 0.89-0.94, image FFN 0.88-0.92, text FFN 0.73-0.83, `level_embed` 0.992; `text_feat_map` vs `text_projection` 0.919 |

The lowest similarities are in the text self-attention blocks. BERT (197 tensors) is identical in `pretrain`
and `post_training_robotwin`; `text_feat_map` differs (weight cosine 0.977). Reading: Grounding-DINO-family
weights retrained through HoloBrain's stages, not a frozen copy of BIP3D, so E4 carries the decision. Our
neck, spatial enhancer, BERT and `text_feat_map` come from `pretrain/`; the fusion is taken from the same file
and checkpoints are never mixed. Open: the comparison against the `HorizonRobotics/BIP3D` grounding `.pth`
files (needs torch).

### Deformable attention: pure PyTorch, no kernel

`bip3d/layers.py:29-39` imports `MultiScaleDeformableAttentionFunction` and `load_cuda_kernels` from
transformers. 4.57.6 has neither, so the code falls back to transformers' `MultiScaleDeformableAttention`
(`load_cuda_kernels = None`, checked): `grid_sample`, bilinear, zero padding, no parameters, `im2col_step`
ignored. No `kernels` package, mmcv or `ms_deform_attn` build is installed, so nothing replaces it on the GPU.

Memory. The op samples one tensor per level, [3F·8, 32, 1700, 4] fp32 for F frames (3 cameras × 8 heads,
32 channels per head, 1700 queries, 4 points), stacks the 4 (a copy) and multiplies by the weights (a third
copy): about 250 MB per frame at peak, about 16 GB at 64 frames, against 2.7 GB for the FFN's hidden layer
(from the shapes; `scripts/lang/bench_fusion.py` measures it). So memory, not the FFN, sets the chunk size.
Fix: `fusion.msda_lowmem`, a drop-in for each image block's `self_attn.func` that sums per level instead of
stacking (exact up to summation order), and chunks of 32 frames (`fusion_chunk`).

### Two masking defects of the official forward, and the fix

1. Padded words are not masked in the text-image attention (`layers.py:662-681`). The mask is bool, and
   `mask.masked_fill(mask == 0, -9e15)` casts −9e15 to True (checked: `[[True, False]]` → `[[True, True]]`),
   so every logit gets +1 and nothing is masked. Image cells then read the pad rows, which are not zero:
   BERT does not zero them under the 3-D mask, and the cache holds `text_feat_map(BERT([PAD]))` up to the
   longest sentence of each build chunk, zeros after it. A padded sentence's fusion would depend on how the
   cache was built.
2. The text self-attention mask is tiled head-major (`feature_enhancer.py:220-225`,
   `~masks.repeat(num_heads, 1, 1)`), while `nn.MultiheadAttention` reads a [B·H, L, L] mask batch-major
   (row b·H + h). With B > 1 and sentences of different lengths, one sample's heads get other samples'
   sub-sentence masks.

HoloBrain's inference runs one unpadded sentence, so it hits neither. Fix (`FusionStage.fuse`): group the
frames by sentence length n (`wmask.sum(1)`), slice the words to n, and run the official forward unchanged.
Within a group there is no padding and every mask is the same, so neither defect can act, and the result is
HoloBrain's single-sentence inference. `position_ids` ([CLS] = 0, w₁ = 0, ..., [SEP] = n − 2) and the
sub-sentence mask (one block over tokens 1..n − 1, plus the identity) are a closed form of n, derived from the
cached `mask`: no text-cache change. The fused words are re-padded with zeros to 24.

### Sentence-dependent positions

The spatial enhancer predicts each cell's depth distribution from the 2-D features it is given
(`spatial_enhancer.py:211-248`); in HoloBrain's order those are the fused maps. So the tokens' 3D positions
depend on the sentence too, not only their content: `fuse` returns tokens and positions, and the trainer, the
shim, the probes and V4b carry the fused positions. Also, each fusion layer writes `LN(x) + γ·Δ`
(`layers.py:807-817`) and then post-norm image and text layers, so even the content-free sentence
("" → [CLS][SEP]) changes every cell. E4 measures token changes against that sentence and the pair's other
sentence, and reports the change against the unfused tokens alongside.
