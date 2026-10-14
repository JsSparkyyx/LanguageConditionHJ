# Plan: vanilla Qwen (V6), the vanilla model on HoloBrain's Qwen2.5-VL front end

Written 2026-10-14 (Yuxuan's decision, same day). Replaces the earlier draft of this file (6b7e184).
Everything in `VANILLA_PLAN.md` and `DESIGN_CONTEXT.md` stays in force unless this file says otherwise.
`FUSION_PLAN.md` (V5) is independent: V6 does not use `scene_fusion`.

## 0. Why we are adding this

**What we found.** The vanilla model, trained with HJ-SAC and paired relabeling, does read the constraint
sentence: on the scenes it trained on it separates "don't touch the red mug" from "don't touch the blue
bottle" at the same state (object pairs, already harmed: 0.93-0.97). On new scenes it falls to the level of a
reader that sees only the words (0.56-0.63, `grounding_report_2026-10-09.md`; confirmed again by the recent
vanilla training). The report's evidence says the name → object map is **memorized per training scene**,
not learned as a general rule: the gap between training and new scenes, foresight on training scenes that
only memory explains, a flat learning curve on new scenes, an auxiliary "where is the named object" head that
is also memorized (V4), and an oracle location tag the model ignores (V3).

**The hypothesis V6 tests.** Our frozen front end gives the model nothing to generalize from on the
language side. The GD scene tokens come from a Swin trained for HoloBrain's action decoder, and the words come
from BERT and never saw an image. Nothing in the frozen path relates "mug" to the mug's pixels, so the
trainable model has to build that relation from about 60 scenes, and memorizing the layouts is the easier
fit. If the frozen front end already supplies a pretrained word-image relation, the model only has to *use*
it, which should transfer to new scenes.

**Why HoloBrain's Qwen variant is the right test.** It is the same family of front end we use today (same
authors, same RoboTwin training data, same spatial enhancer for 3D positions, same robot-state encoder
design), so it changes one thing: the vision-language part. Its vision tower is Qwen2.5-VL-3B's, pretrained
on web-scale image-text data, and its words come out of a Qwen language layer that has attended over the
image tokens. "A better VLM" is therefore tested without changing the robot side, the geometry, or the data.

**What it is not.** It is one transformer layer of a 3B model (HoloBrain cut the language model to layer 0
of 36). That is a pretrained joint image-text space plus one round of cross-modal attention, not deep VLM
reasoning. If V6 helps, a deeper cut of the base Qwen is the next step (section 6). If it does not, the
cheap probe of section 4 tells us whether the words point at the named object at all.

## 1. How HoloBrain's Qwen variant is built (verified in the code and the published config)

Sources: `robo_orchard_lab/models/holobrain/structure.py` (class `HoloBrain_Qwen2_5_VL`, lines cited),
`projects/holobrain/configs/config_holobrain_qwen_common.py`, `config_robotwin_dataset.py`, and
`HoloBrain_v0.0_Qwen/pretrain/model.config.json` on Hugging Face.

1. **Inputs.** Three camera RGB images at 308 × 252 (`dst_wh`), the three depth maps at the same size, the
   robot joint state, the instruction string.
2. **Prompt** (`TextTemplate`, 51-81), per sample, images before the instruction:
   ```
   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
   <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> ×3 (one per camera)
   \nYou are a robot. {instruction}<|im_end|>\n<|im_start|>assistant\n
   ```
3. **Vision tower, frozen** (`freeze_vision = true`). Qwen2.5-VL-3B's ViT: 32 blocks, width 1280, patch
   14 px, windowed attention with full attention at blocks 7, 15, 23, 31. 308 × 252 gives 22 × 18 patches;
   a 2 × 2 merger makes one 2048-d token per 28 × 28 px cell, 11 × 9 = 99 per camera, 297 per frame. These
   fill the `<|image_pad|>` slots of the token embeddings (`_forward_vlm`, 337-370).
4. **Language model cut to one layer** (`num_vlm_layers = 1`, 131-138): layer 0 of 36, causal
   self-attention + FFN at width 2048, 3D rotary positions (time, row, column for image tokens).
   Layer 0 and the token embeddings are fine-tuned at 0.1 × lr (`freeze_vlm = false`) and the checkpoint
   stores them (`use_state_dict_with_vlm = true`, 2.2 GB `model.safetensors`). The Qwen base itself is
   downloaded separately (`vlm_pretrain = ./ckpt/Qwen2.5-VL-3B-Instruct`). Two hidden states are kept, the
   input embeddings and the output of layer 0.
5. **Feature mapping** (144-164, 236-250): each hidden state through its own Linear(2048, 256), summed with
   weights softmax(w), w a learned 2-vector (starts near 0.46 / 0.54).
6. **Split** (252-292): image tokens = the 297 image-slot states, one 256 × 9 × 11 map per camera; text
   tokens = every other non-pad position (system prompt, vision markers, "You are a robot.", the
   instruction, the assistant header).
7. **Geometry.** A depth Swin (patch 7, strides 7 / 2 / 2, so one cell per 28 px) gives a 128-d depth
   feature per cell; the `DepthFusionSpatialEnhancer` predicts a distribution over 128 depth bins
   (0.01-1.2 m) per cell from the image and depth features, takes the expected 3D point along the cell's
   ray, embeds it and fuses it into the token. It does not read the text.
8. **Action decoder** (not used by us): 6 diffusion layers with cross-attention to the image tokens and to
   the text tokens.

**The property the design rests on.** Layer 0 is causal and the images come first, so:
- the image tokens **never see the instruction**: they are identical for every sentence and can be
  computed once per frame and stored, as the GD tokens are today;
- the instruction tokens **attend over all 297 image tokens**: the words come out image-conditioned.

In HoloBrain-Qwen the grounding therefore lives in the words, not in the patches.

## 2. The model (V6)

Only the frozen front end changes. Two inputs of the trainable model change, the scene tokens and the
words. Everything from the text adapter down is the vanilla model as it is.

### 2.1 Scene side (per frame, sentence-free, stored)

For each frame (t, t−k, 0): the three 308 × 252 images through the Qwen vision tower; layer 0 over
HoloBrain's prompt with an **empty instruction**; `feat_mapping`; the image-slot states with their 3D
positions from the depth Swin and the spatial enhancer. Result per frame: `scene_tokens` [297, 256],
`scene_pos` [297, 3] (world metres, the same convention as `encode_visual_tokens` with `out_extrinsic`).
The trunk sees 3 × 297 = 891 scene tokens instead of 3600.

Also kept per frame: layer 0's **keys and values of the prefix** (every position up to and including
"You are a robot. ", about 325 positions × 2 key-value heads × 128, bf16, about 330 KB) and the rotary
position counter at the end of the prefix. The text pass needs both.

The empty instruction is exact, not an approximation: causal attention means no image position ever reads
the instruction's positions.

### 2.2 Text side (per frame and sentence)

The constraint sentence s (the sample's paraphrase, never a task description) is tokenized as the template's
tail, `"{s}<|im_end|>\n<|im_start|>assistant\n"`, embedded with the fine-tuned token embeddings, and run
through layer 0 attending causally to its own tokens and to the cached prefix of **frame t** (frame t+1 for
the target). `feat_mapping` gives 256-d states. Keep **only the instruction's own tokens** (L ≤ 24, mask
padded): the template's tail and the prefix's text tokens are the same for every sentence, the failure we
measured with BERT's constant [CLS] (`grounding_report` 4.3).

These words replace the `TextStore` words as the text adapter's input. Concretely, the state of "mug" is
its embedding plus what it read from the image tokens it attended to; if that attention lands on the mug's
cells, the word carries the mug's appearance and the trunk can match word to patch by content instead of
by memorized layout.

### 2.3 Unchanged

`SceneSet` (CrossSafe's scene adapter plus τ_f per frame role, now on 297-token frames; `n_tok` is read from
the shard), scene reads words, the 3 trunk blocks (a)-(e) with the geometric bias on the Qwen positions,
`RobotInjection`, the actor, the twin critic with the MLP readout, HJ-SAC with γ = 0.95, paired relabeling,
the five text switches.

**Robot path: keep the GD robot-state encoder** (`HoloBrainEncoder.encode_joint_angles`, as today), so the
comparison changes the vision-language side only. HoloBrain-Qwen has its own copy of that encoder; swapping
it is a later, separate change.

### 2.4 Cost

- Training: one layer-0 pass per sample over about 15 tokens against a cached prefix (paired samples share
  the prefix, not the sentence), roughly 1 TFLOP per batch of 64 in all. No online fusion stage.
- Deployment, per tick: the vision tower on 3 images (order of 1 TFLOP, about 10-20 ms in bf16 on an A100),
  depth Swin and enhancer (small), the text pass for the episode's sentence (two in monitor mode), then the
  vanilla model. The text adapter runs every tick, because the words change with frame t (the per-tick
  `text_out` path V5 already built for `fusion_words`).
- Storage: tokens 152 KB + positions 4 KB + prefix 330 KB per step, about 0.5 MB with the robot fields,
  against 0.63 MB today.

## 3. Implementation (for the code agent)

Do not edit `main/network/encoder.py`, `main/envs/env.py`, `main/train/`, or anything under `envs/`.
Compose in `main/lang/`.

1. **Recon** (`recon.md`). Download `HoloBrain_v0.0_Qwen/pretrain` and `Qwen/Qwen2.5-VL-3B-Instruct` into the
   HF cache on Delta. Build `HoloBrain_Qwen2_5_VL` from `model.config.json` with `vlm_pretrain` pointed at
   the local base; load `model.safetensors` through the class's own `load_state_dict`. List the `vlm.*`
   keys (expect the embeddings, layer 0 and the vision tower). HoloBrain asks for `flash_attention_2`; if the
   container lacks it, use `attn_implementation = "sdpa"` and record which ran. Time and memory of one
   frame and one text pass. **Confirm the causal claim**: image-slot states identical (max abs diff 0 in
   bf16) for two different instructions.
2. **`main/lang/qwen_encoder.py`**: `HoloBrainQwenEncoder` with
   - `encode_frame(rgb, depth, image_wh, projection_mat, out_extrinsic) -> tokens [3, 99, 256], pos
     [3, 99, 3], prefix_kv`;
   - `encode_text(prefix_kv, sentences) -> words [B, 24, 256], mask [B, 24]` (instruction tokens only,
     zero-padded);
   - all `@torch.no_grad()`, the VLM in bf16 as HoloBrain runs it, the depth Swin and the enhancer in fp32.

   Tests: (a) **parity**, the instruction tokens from `encode_text` equal the same tokens from one pass over
   the full prompt (within bf16 summation order); (b) the token grid is 297, positions in world metres,
   ordering cameras then rows then columns as our flattening expects; (c) a batch of 2 equals two batches
   of 1, with sentences of different lengths (padding masked).
3. **Observation path** (`main/lang/obs.py`): RGB and depth at 308 × 252 with the projection matrices
   scaled to that size, raw 0-255 RGB (the Qwen processor does its own normalisation; do **not** apply the
   GD bundle's mean / std), BGR/RGB as HoloBrain's preprocessor expects, depth with the same alpha masking
   as `get_encoder_obs`.
4. **Storage.** A V6 shard is a normal shard whose `scene_tokens` / `scene_pos` are the Qwen ones (297 wide)
   plus a `kv_buffer/` sidecar (the prefix per step) in lockstep, modelled on `neck_buffer/`. The shard meta
   records `encoder = "holobrain_qwen"`; the sampler and the trainer refuse a shard whose encoder differs
   from the run's.
   **Also store the renders** in a `frames/` sidecar (3 × RGB uint8 at 308 × 252, 3 × depth fp16, projection
   matrices, `T_ego2world`; about 1.2 MB per step). Every front-end change so far forced a re-collection
   because shards hold encoder outputs. With the frames, the GD reference on the same episodes (section 4,
   V0'') and any later front end are offline computations, and every variant is scored on identical pairs.
5. **Config** (`LangConfig`): `encoder` ("holobrain_gd" default and for saved configs, "holobrain_qwen"),
   `text_backend = "holobrain_qwen"` (allowed only with that encoder; `in_proj` identity), and
   `qwen_words_prefix` ("frame", default; "blank" = the text pass against a prefix built from black images,
   the control of section 4). `scene_fusion` is refused with the Qwen encoder.
6. **Sampler and trainer.** `LangSampler` loads `prefix_kv` for frames t and t+1. `_grad_step` calls
   `encode_text` under `no_grad`, chunked, before `encode`, and feeds the words to the text adapter in place
   of `text.lookup`; the target branch uses frame t+1's prefix with the same sentence. Paired samples share
   the prefix and differ in the sentence.
7. **Rollout and collection.** Override `_encode_obs` in `LangRolloutController` to run `encode_frame` and
   the GD robot-state encoder, return the Qwen tokens and positions to the base capture (so the StepBuffer
   row is written as usual), keep `prefix_kv` for the sidecar and the shim. The shim runs `encode_text` for
   the episode's sentence (and the counterfactual one in monitor mode) every tick, then `encode_text` of the
   text adapter, then the model. Nominal collection runs no text pass.
8. **Evaluation.** `eval_offline.py`: words per (state, sentence) through `encode_text` with the state's
   prefix, cached per (state, sentence) inside a chunk. `grounding_summary.FIXED` gains `encoder`,
   `text_backend`, `qwen_words_prefix`.
9. **Docs**: `DESIGN_CONTEXT.md` 5.1 (the encoder switch, what each front end provides) and 8 (V6 in the
   decision order), `CLAUDE.md` (the Qwen base location; the frames sidecar as the collection default).

## 4. Experiments

### E7. Do the Qwen words point at the named object? (no training, first)

On a V6 validation collection (T1 and T7, 20 episodes each, ur5-wsg, token masks on, the `val_grounding`
seed stream), every 25th tick, every sampleable rule with a training-template sentence:

- **attention enrichment**: in layer 0, the attention from the referent phrase's tokens (for example "red
  mug") to the 297 image tokens, averaged over the 16 heads; the share landing on the named item's cells
  (its mask projected to the 11 × 9 grids) divided by the item's share of all cells. 3 means the referent
  looks at the mug 3× more than at a random cell;
- **contrast**: for two rules of the same consequence in one scene (an object pair of the report), the share
  of pairs in which each sentence's referent puts more attention on its own item than on the other one;
- **image dependence of the words**: cosine between a referent token's state under the real frame and under
  the blank prefix; near 1 means the image barely changes the word.

Per task with episode-cluster bootstrap intervals, and 6 overlay images per task (attention on the camera
images, named item outlined) for the coauthor page. **Gate**: enrichment ≥ 2 and contrast ≥ 0.7 on T1. If
it fails, the frozen words do not ground the referent and training V6 is not worth the collection; report and
stop.

### E8. V6 short runs (the E2 / E5 protocol)

Collect T1 + T7 ur5-wsg training data with the Qwen encoder and the frames sidecar (the 60 layouts of
`data/lang/warmup`, same seed streams), then 6,000 steps, batch 64, lr 3e-4, γ 0.95, `actor_hidden 0`, MLP
readout, seeds 0 and 1, scored on the V6 validation pairs:

| run | setting | what it isolates |
|---|---|---|
| V0'' | the GD front end computed offline from the stored frames, BERT words | the reference on exactly these episodes and pairs |
| V6 | Qwen front end, words against frame t's prefix | the whole change |
| V6-blank | Qwen front end, words against the blank prefix | better visual tokens alone, without image-conditioned words |

Report as E5: object pairs already harmed (main), not yet, rule-kind pairs, the training-scene score, the
learning curve on new scenes (1,000-step checkpoints), in-batch pair accuracy, and the per-episode repeat
count. **Decision**: V6 ≥ 0.85 on the main score, adopt the Qwen front end (full collection, retrain the
official model). Clearly above V0'' but below 0.85, check V6-blank: if the gain is in V6 only, the
image-conditioned words carry it and a deeper Qwen cut is the next step (section 6); if V6-blank has it too,
the visual tokens carry it. At V0'' with E7 passed, the model ignores grounded words and memorization
dominates; the lever is then the data (the scene-count curve E6), not the front end.

## 5. Things to watch

- **Resolution.** One token per 28 px cell against GD's 16 and 32 px cells. A T7 neighbour block or a thin
  bottle may cover one or two cells. Report T1 and T7 separately.
- **What layer 0 was tuned on.** HoloBrain fine-tuned it on RoboTwin task instructions ("place the cup on
  the coaster"), not on constraints ("don't touch …"). E7 checks whether the attention still lands on the
  named object under our phrasing.
- **Token count.** 891 scene tokens instead of 3600 make the trunk about 4× cheaper in scene attention;
  the geometric bias now works on coarser positions.

## 6. Later, only if V6 moves the number

A deeper cut of the base Qwen2.5-VL-3B (HoloBrain fine-tuned only layer 0, so deeper layers are the base
model's): read the words at a middle layer, about 18, with images still first (deep words, sentence-free
patches, cost about 120 GFLOP per (frame, sentence)); or put the text before the images so the patches see
the sentence (no longer precomputable per frame, so one fixed paraphrase per (episode, rule) at collection, a
change to the relabeling design). Both need a trainable 2048 → 256 projection and positions from the depth
image directly, since the spatial enhancer was trained on layer-0 features. Not part of V6.
