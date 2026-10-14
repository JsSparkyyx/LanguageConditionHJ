# Plan: HoloBrain's Qwen variant as the frozen encoder of the vanilla model (V6)

Written 2026-10-14. Context: the vanilla pipeline reads the sentence and separates objects on the scenes it
trained on, but not on new scenes (`grounding_report_2026-10-09.md`). The question is whether a stronger
vision-language model as the frozen front end changes that. HoloBrain ships a Qwen2.5-VL variant
(`HoloBrain_v0.0_Qwen`). This file says what that variant is, what it would give the vanilla model, how to
plug it in, and how to test it. `FUSION_PLAN.md` (V5) stays as written; V6 is a second candidate on the same
protocol. Everything in `VANILLA_PLAN.md` stays in force unless this file says otherwise.

## 0. What the Qwen variant is (verified in the code and the published config)

Sources: `robo_orchard_lab/models/holobrain/structure.py` (class `HoloBrain_Qwen2_5_VL`),
`projects/holobrain/configs/config_holobrain_qwen_common.py`, `config_robotwin_dataset.py`, and
`HoloBrain_v0.0_Qwen/pretrain/model.config.json` on Hugging Face.

| fact | value | where |
|---|---|---|
| base VLM | Qwen2.5-VL-3B-Instruct: vision tower 32 blocks of width 1280, patch 14 px, 2 × 2 merge, so one visual token per 28 × 28 px; language model 36 layers, width 2048, 16 heads, 2 key-value heads, FFN 11008 | Qwen config.json |
| how much of the language model runs | **one layer**. `num_vlm_layers = 1` truncates `language_model.layers` to layer 0. The usable hidden states are the input embeddings and the output of that one layer | `structure.py` 131-138, `model.config.json` |
| what is trained | the vision tower is frozen (`freeze_vision = true`), layer 0 and the token embeddings are fine-tuned at 0.1 × lr (`freeze_vlm = false`), and the checkpoint carries them (`use_state_dict_with_vlm = true`, `model.safetensors` is 2.2 GB). The Qwen base must be downloaded separately (`vlm_pretrain = ./ckpt/Qwen2.5-VL-3B-Instruct`) | `structure.py` 103-129, 413-420; model card |
| the prompt | `<\|im_start\|>system\nYou are a helpful assistant.<\|im_end\|>\n<\|im_start\|>user\n<\|vision_start\|><\|image_pad\|><\|vision_end\|> × 3 cameras \nYou are a robot. {instruction}<\|im_end\|>\n<\|im_start\|>assistant\n` | `structure.py` 55-59 |
| input size | 308 × 252 (`dst_wh`), so each camera gives 11 × 9 = 99 visual tokens, 297 per frame (today's GD path gives 1200) | `config_robotwin_dataset.py` 173 |
| feature mapping | the 2 hidden states (embeddings, layer 0) each go through their own Linear(2048, 256), and are summed with learned softmax weights (`feat_mapping`, `weight`) | `structure.py` 144-164, 236-250 |
| scene tokens | the mapped states at the `<\|image_pad\|>` positions, reshaped to one 11 × 9 map per camera | `structure.py` 268-277 |
| text tokens | the mapped states at every non-image, non-pad position (system prompt, "You are a robot.", the instruction, the assistant header) | `structure.py` 279-291 |
| 3D positions | the same `DepthFusionSpatialEnhancer` as the GD variant, on the one 28-px map, with a depth Swin (`backbone_3d`, patch 7, strides 7 / 2 / 2, so stride 28) giving a 128-d depth feature per cell | config 160-195 |
| robot state encoder | the same `HoloBrainRobotStateEncoder` as GD, its own weights | config |

**The one fact that decides how to use it.** The language model is causal, and the images come before the
instruction in the prompt. So the hidden states at the image positions never see the instruction: **the
scene tokens of the Qwen variant are sentence-independent**, exactly like today's GD tokens. The
instruction's tokens, on the other hand, attend over all 297 image tokens in layer 0, so **the text states
are image-conditioned**: each word's state carries what the image tokens it attended to contained. Grounding
in HoloBrain-Qwen therefore lives in the words, not in the patches, and the action decoder reads it through
its text cross-attention. This is the mirror image of the GD fusion stage (V5), where image cells read the
words.

Consequences for us:
- the sentence-independent scene tokens can be precomputed and stored as today (no online fusion stage, no
  25 TFLOP per step);
- the words must be computed per (frame, sentence), but that is one language-model layer over about 15
  tokens with the frame's prefix cached, a negligible cost;
- "a better VLM" in the sense of deep multimodal reasoning is **not** what HoloBrain-Qwen provides. One
  layer of a 3B model is a learned cross-modal attention in a pretrained joint embedding space, nothing
  more. That is still a different and much larger-pretrained alignment than ours (section 1), and it is
  cheap to test. The deep version is section 3.

## 1. What it offers against today's GD path, and why it could help

Today: Swin-tiny + neck at 320 × 256 (1200 tokens per frame), with no text contact at all in the frozen
stage, and BERT words that never saw an image. The model has to learn the name → patch map from our scenes,
and it memorizes them.

With HoloBrain-Qwen:
- **Visual tokens from a VLM's vision tower** (0.7 B parameters, pretrained on web-scale image-text data,
  then fine-tuned for Qwen2.5-VL's joint space). Object identity and colour are what such towers encode
  best. 99 tokens per camera at 28 px against our 400 at 16 and 32 px: coarser, which matters for thin
  parts (a 2 cm block edge is under one cell), and less for identity.
- **Words that have read the image.** After layer 0, the token "mug" has attended over the three images'
  tokens with attention weights trained on image-text data and fine-tuned by HoloBrain on RoboTwin
  instructions. If the attention lands on the mug, the word state carries the mug's visual content and
  position-correlated features, and our trunk can match it to the scene tokens by content rather than by
  memorized layout.
- **The same sidecar and sampler as today**: tokens and positions per frame, sentence-free. The only new
  online stage is the per-sentence text pass.

What it does not change: the trainable model, the loss, relabeling, the readout, the frame history, the
deployment logic. The scene geometry (`scene_pos` from the spatial enhancer's depth distribution) is
HoloBrain-parity for this variant, as it is for GD today.

## 2. Integration A (the drop-in, V6)

### 2.1 Frozen encoder (`main/lang/qwen_encoder.py`, new; no edit to `main/network/encoder.py`)

`HoloBrainQwenEncoder(ckpt_dir, qwen_dir)` builds the HoloBrain class from `model.config.json`
(`HoloBrain_Qwen2_5_VLConfig`, with `vlm_pretrain` pointed at the local Qwen base) and loads
`model.safetensors` through the class's own `load_state_dict` (it carries the fine-tuned layer 0 and
embeddings). Attention backend: HoloBrain asks for `flash_attention_2`; if the container lacks it, build
with `attn_implementation = "sdpa"` (Qwen2.5-VL supports it in transformers 4.57). Record which backend
ran in `recon.md`; the two differ only numerically.

Three methods, all `@torch.no_grad()`, bf16 for the VLM as HoloBrain runs it, fp32 for the spatial enhancer:

```python
def encode_frame(imgs_rgb_308x252_uint8, depths, image_wh, projection_mat, out_extrinsic):
    # 1. vision tower on the 3 images -> 3 x 99 merged embeddings (2048-d)
    # 2. the prompt with an EMPTY instruction, through layer 0 -> hidden states (embeddings, layer 0)
    # 3. feat_mapping -> 256-d; image positions -> one 11 x 9 map per camera
    # 4. depth Swin + spatial enhancer -> tokens [n_cam, 99, 256], positions [n_cam, 99, 3] (world metres,
    #    via out_extrinsic as encode_visual_tokens does today)
    # 5. layer-0 keys and values of the prefix (everything up to and including "You are a robot. "),
    #    plus the rope position state at the end of the prefix
    return tokens, pos, prefix_kv            # prefix_kv is what the text pass needs

def encode_text(prefix_kv, sentences):      # per (frame, sentence), the only online stage
    # tokenize "{sentence}<|im_end|>\n<|im_start|>assistant\n" (the template's tail), embed, run layer 0
    # with attention over [prefix_kv, own causal tokens], feat_mapping -> words [L, 256], L <= 24
    # (the instruction tokens only; the assistant header is dropped). Returns words and a mask.

def encode_joint_angles(...)                # HoloBrain's robot state encoder, Qwen checkpoint weights
```

Why the empty instruction in step 2 is exact: the image positions precede the instruction, and attention is
causal, so their states are the same for every instruction. The prefix keys and values are likewise
instruction-independent. `encode_text` reproduces HoloBrain's full forward for the instruction tokens
bit-for-bit up to bf16 summation order (test: run the whole prompt in one pass and compare the instruction
tokens' states).

Observation bundle. HoloBrain-Qwen takes RGB at 308 × 252 (the Qwen processor does its own rescaling and
normalisation; `BaseDataPreprocessor` only flips BGR to RGB) and depth at the same size for the depth Swin.
Add a lang-side observation path that renders or resizes to 308 × 252 and keeps the raw 0-255 RGB (the GD
bundle's mean / std normalisation must not be applied), with the projection matrices scaled to the new
size. Do not edit `main/envs/env.py`: compose in `main/lang/`.

### 2.2 Storage (sidecars, as `neck_buffer/` in V5)

| field | shape, dtype | bytes per step |
|---|---|---|
| `scene_tokens` | [3, 99, 256] fp16 (the StepBuffer field, now 297 wide) | 152 KB |
| `scene_pos` | [3, 99, 3] fp32 | 4 KB |
| `prefix_kv` | layer 0 keys and values, about 325 tokens × 2 kv-heads × 128, bf16, both | 330 KB |
| robot fields, h, item sidecar | as today | ~20 KB |

About 0.5 MB per step, against 0.63 MB today and 3.5 MB for V5. The StepBuffer's token width is a
per-shard constant (`n_tok` is read from the shard), so a V6 shard is a normal shard with 297-wide tokens
plus a `kv_buffer/` sidecar in lockstep.

**Store the renders this time.** Every encoder change so far has forced a re-collection because the shards
hold encoder outputs, not images. Add a `frames/` sidecar with the raw renders per step (3 × RGB uint8 at
308 × 252 = 700 KB, 3 × depth fp16 = 465 KB, projection matrices and `T_ego2world`), about 1.2 MB per step.
With it, GD tokens, V5 neck maps, V6 tokens and any later layer choice (section 3) are offline computations
over the same episodes, and every variant is scored on identical pairs. 1.7 MB per step in total for a V6
shard with frames, 130 GB for the T1 / T7 training and validation set.

### 2.3 Training step

Unchanged except for the words. `LangSampler` loads `prefix_kv` for frame t (and, if `words_per_frame`
asks for them, for frames t−k and 0; default frame t only, as V5's `fusion_words`). `_grad_step` calls
`encode_text(prefix_kv_t, sentences)` under `no_grad` and hands the words to the text adapter in place of the
`TextStore` words. The target branch uses the words of frame t+1 with the same sentence. Paired samples
share the prefix and differ in the sentence, so the text pass runs once per sample (320 small passes per
step, about 1 TFLOP in all).

`text_backend = "holobrain_qwen"` (words = the 256-d mapped states, `in_proj` identity, default) and
`"qwen_raw"` (the 2048-d layer-0 states with a trainable Linear(2048, 256), the Q3 analogue of `bert_raw`).

### 2.4 Deployment and collection

Per tick: `encode_frame` on the fresh frame (vision tower on 3 images: 3 × 396 patches through 32 blocks,
on the order of 1 TFLOP, 10-20 ms in bf16 on an A100; the depth Swin and the enhancer are small), then
`encode_text` with the episode's sentence (and the counterfactual sentence in monitor mode), then the
vanilla model. The frame history keeps tokens and positions as today, and the words of frame t are
recomputed each tick. `set_instruction` keeps the sentence; `text_out` is per tick, as V5 already does.
Nominal collection stores tokens, positions, `prefix_kv` and the frames, and runs no text pass.

### 2.5 Switches

| field | default | meaning |
|---|---|---|
| `encoder` | "holobrain_gd" (saved configs), "holobrain_qwen" for V6 runs | which frozen front end a run and its shards use; a shard records it in its meta and a run refuses a mismatch |
| `text_backend` | "holobrain_qwen" under the Qwen encoder | section 2.3 |
| `words_per_frame` | "t" | which frames' words the adapter sees ("t" or "all": t, t−k, 0 concatenated, 3 L tokens) |

`scene_fusion` (V5) is GD-only and is refused under the Qwen encoder. `critic_readout = "mlp"` as decided.

## 3. Variant B: the deep version (only if A moves the needle or you want the hypothesis tested properly)

HoloBrain-Qwen uses one layer because the action decoder needs speed, not understanding. To test "a better
VLM" as such, take the **base** Qwen2.5-VL-3B (not HoloBrain's weights, which exist for layer 0 only), run
ℓ layers (HoloBrain's own weighting peaks at the middle layer, so ℓ ≈ 18), and read both the image and the
text states at layer ℓ. Two prompt orders give two different models:

- **B1, images then text** (HoloBrain's order). Image states at layer ℓ are still sentence-independent and
  are precomputed. The instruction's states need layers 1..ℓ over about 15 tokens against the prefix's
  cached keys and values of ℓ layers: 330 KB × ℓ per frame (6 MB at ℓ = 18), about 120 GFLOP per (frame,
  sentence), roughly 40 TFLOP per step. Deep words, shallow patches.
- **B2, text then images.** The image states see the sentence (deep, text-conditioned patches, the closest
  thing to a VLM "looking for the named object"). Nothing is sentence-independent, so the frozen pass is
  about 1 TFLOP per (frame, sentence) and cannot run online per sample. Precompute at collection, per
  tick, for each sampleable rule of the episode with **one fixed paraphrase per (episode, rule)**: about 6
  passes per tick, 6 TFLOP, 0.1 s on an A100, and 297 × 2048 bf16 = 1.2 MB per (tick, rule), about 7 MB per
  step. The text adapter still draws a random paraphrase; the scene conditioning uses the fixed one of the
  same rule. Held-out templates then test the adapter side only.

Both need a trainable projection (2048 → 256) in place of HoloBrain's `feat_mapping`, and positions from
the depth image itself (the mean 3D point of each 28-px cell's depth pixels through the camera matrices),
because the spatial enhancer was trained on HoloBrain's layer-0 features, not on layer ℓ. At deployment a
depth camera gives the same. This is a departure from HoloBrain parity on the geometry side, and it is why
B comes after A.

## 4. Experiments

### E7. Does layer 0 ground the named object? (no training, first)

On the re-collected validation episodes (T1 and T7, 20 each, ur5-wsg, token masks on the 11 × 9 grid), every
25th tick, every sampleable rule with a training-template sentence:

- **attention mass**: in layer 0, the attention from the named object's instruction tokens (the referent
  phrase's tokens) to the image tokens, averaged over the 16 heads: the share landing on the named item's
  mask cells against the item's share of all cells (enrichment), and against the other named objects'
  cells (contrast, as E4 defines it);
- **image conditioning of the words**: cosine between the referent tokens' layer-0 states under the frame
  and under (a) a blank frame, (b) another episode's frame: how much the image moves the words at all;
- **word contrast**: for object pairs of the report's kind (same consequence, different object), the
  distance between the two sentences' referent-token states under the same frame, against the distance
  between two paraphrases of the same rule.

Per task, episode-cluster bootstrap intervals, 6 overlay images per task for the coauthor page. **Gate**:
enrichment ≥ 2 and contrast ≥ 0.7 on T1. If layer 0 does not point the words at the named object, A cannot
work and B2 is the only Qwen test left.

### E8. V6 short runs (the E5 protocol)

Collect T1 + T7 ur5-wsg with the Qwen encoder and the frames sidecar (same seed streams as `data/lang/warmup`
and `val_grounding`), then 6,000 steps, batch 64, lr 3e-4, γ 0.95, `actor_hidden 0`, MLP readout, seeds 0
and 1:

| run | setting |
|---|---|
| V0'' | the GD encoder computed offline from the stored frames: the reference on exactly these episodes |
| V6 | `encoder = holobrain_qwen`, `text_backend = holobrain_qwen`, `words_per_frame = t` |
| V6-raw | `text_backend = qwen_raw` |
| V6-noimg | the words computed with a blank prefix (image-free words): isolates what the image-conditioned words add |

Scored as E5: object pairs already harmed (main), not yet, rule-kind pairs, training-scene score, learning
curve, in-batch accuracy, and the per-episode repeat count. `grounding_summary.FIXED` gains `encoder`,
`text_backend` and `words_per_frame`. Decision rule as E5: ≥ 0.85 adopt; clearly above V0'' but below, the
scene count is the next lever; at V0'' with E7 passed, the grounded words are ignored and B2 is the remaining
Qwen test.

## 5. Implementation steps for the code agent

1. **Recon** (`recon.md`): download `HoloBrain_v0.0_Qwen/pretrain` and the Qwen base into the HF cache on
   Delta; list the `vlm.*` keys of `model.safetensors` (expect layer 0, embeddings, and the vision tower);
   build the class on the GPU; check the attention backend; time `encode_frame` and `encode_text`; memory.
   Confirm the causal claim empirically: the image-position states are identical for two different
   instructions (max abs diff 0 in bf16).
2. `main/lang/qwen_encoder.py`: the three methods of 2.1, with the parity test (whole-prompt pass against
   the prefix-plus-text pass) and the token-grid test (297 tokens, positions in world metres, same
   convention as `encode_visual_tokens`).
3. The observation path at 308 × 252 (`main/lang/obs.py`), the `frames/` and `kv_buffer/` sidecars
   (lockstep, as `neck_buffer/`), the shard meta `encoder` field, and the StepBuffer at `n_tok = 297`.
4. `data.py`: `prefix_kv` loading; `train.py`: the per-sample text pass (chunked), `text_backend` routing;
   `rollout.py`: the per-tick text pass and the counterfactual words. Nothing else in the model changes.
5. E7 script `scripts/lang/qwen_probe.py` (hook layer 0's attention weights through `output_attentions`
   on the truncated model, or a forward hook on the attention module) with the overlays.
6. `eval_offline.py`: words per (state, sentence) through `encode_text` with the state's `prefix_kv`;
   `grounding_summary.py` fields.
7. Docs: `DESIGN_CONTEXT.md` section 5.1 (the encoder switch and what each frozen front end provides),
   section 8 (V6 in the decision order after V5), `CLAUDE.md` (the Qwen base location and the frames
   sidecar as the collection default from now on).

Do not edit `main/network/encoder.py`, `main/envs/env.py`, or anything under `envs/`. Compose in
`main/lang/`.

## 6. Open points for Yuxuan

1. **Order against V5.** V5 (the GD fusion) is implemented and not yet run on a GPU. V6 is cheaper to
   train and store, but its grounding lives in the words rather than the patches. Both are testable on one
   re-collection if the frames sidecar is adopted now (2.2). Recommendation: collect once with frames, run
   E4 and E7 (no training) on the same episodes, and let those two numbers order E5 and E8.
2. **The depth input.** HoloBrain-Qwen's depth Swin sees the depth at 308 × 252 with the same alpha
   masking as GD. Keep the GD conventions (`get_encoder_obs`) for the depth side.
3. **B2 and paraphrase diversity.** If B2 is ever run, the fixed paraphrase per (episode, rule) is a real
   change to the relabeling design (the frozen side never sees paraphrases). Decide then.
4. **Disk.** 1.7 MB per step with frames. The full 7-task two-robot warmup would be about 600 GB; T1 / T7
   with validation about 130 GB.
