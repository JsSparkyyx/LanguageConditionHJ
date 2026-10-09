# Plan: HoloBrain's own text-image fusion as the grounding stage (V5)

Addendum to `VANILLA_PLAN.md`, written 2026-10-09 after `grounding_report_2026-10-09.md`. Read that report
first. Everything in `VANILLA_PLAN.md` stays in force unless this file says otherwise.

## 0. Decisions (Yuxuan, 2026-10-09)

1. **γ = 0.95 everywhere, from now on.** No run at 0.9 is started, continued or reported as a baseline.
   - Stop `rounds_ur5_g090_tf32`. `rounds_ur5_g095_tf32` is the current official run.
   - Remove the `GAMMA` variable from the job scripts (hard-code 0.95) and drop 0.9 from the experiment
     matrix (`VANILLA_PLAN.md` 9.3, `scripts/lang/matrix.sh`). `LangConfig.gamma` keeps its assert and the
     saved-config rule, so old checkpoints still load.
   - The official model of the grounding report (`lang_full_paired_off_1007`, γ 0.9) is a historical
     reference only. New comparisons use models trained at 0.95.
2. **Slots plus a pointer stay on the shelf.** The report's evidence (V3 ignored, V4 memorized) does not
   support them. The next architecture step is the one below, which adds a *pretrained* grounding stage
   instead of learning grounding from our 60 scenes.
3. **The fusion stage sees the constraint sentence only.** Never the task description. HoloBrain used it for
   the task instruction, we use it for the rule. One sentence per sample, the same string that the text
   adapter sees.
4. **Keep the grounding gate as it is**: ≥ 0.85 rank accuracy on object pairs, already harmed, T1 + T7
   pooled, on the 1,735 validation pairs of the report. "Not yet" pairs are reported alongside but do not
   gate (the critic bootstraps through the actor, so they do not cleanly test foresight).
5. **Run the scene-count curve** (report section 8) in parallel with V5. It is cheap (V0 settings, subsampled
   data) and separates "too few scenes" from "the architecture cannot ground".
6. **The critic's readout is an MLP, not the softmin** (Yuxuan, 2026-10-09). New runs use it by default
   (`critic_readout = "mlp"`, section 3.6); a config saved before the field keeps the softmin. E5 isolates the
   readout's own effect with one extra row.

## 1. Why this, and why now

The report's firm findings. The sentence reaches the decision (object words attended 2.1× their share, 0.97
on training scenes), but the name → object map is learned per training scene and does not carry to new
scenes (0.56-0.63). Even direct 3D-centre supervision (V4) is memorized. The oracle location tag (V3) is
ignored because the training loss is already near zero through memorization.

What we skip today. HoloBrain-0 GD solves exactly this name → patches problem for its task instruction, with
a stage that was trained on far more scenes than we have. Its pipeline is

```
RGB  -> backbone (Swin) -> neck (4 maps, strides 8 / 16 / 32 / 64, 256-d)
text -> BERT -> text_feat_map (768 -> 256)                      [we have this: HoloBrainText]
     -> feature_enhancer (6 layers of text-image fusion)        [we SKIP this]
     -> spatial_enhancer (depth, 3D positions)                  [we run this, on unfused maps]
     -> decoder (also cross-attends to the fused words)          [not used]
```

`main/network/encoder.py` loads `backbone`, `neck`, `backbone_3d`, `neck_3d`, `spatial_enhancer` and the
robot-state encoder, and nothing else (the change list `figures/make_change_list_pdf.py`, row
"feature_enhancer / text branch", already flagged this). Two consequences:

- our scene tokens carry no pretrained word-to-patch alignment, so the model has to learn grounding from
  our data, and it memorizes instead;
- the frozen `spatial_enhancer` runs on inputs it was never trained on (it always followed the fusion stage
  in HoloBrain training).

V5 fixes both by running the checkpoint's `feature_enhancer`, frozen, with the constraint sentence, in
HoloBrain's own order. Nothing is trained in the frozen path. The trainable model is unchanged except
for what it receives.

Honest risk. The fusion rewrites the scene tokens with the sentence, so the named object's patches differ
in content, which is a much stronger signal than V3's zero-initialized additive tag. But if 60 scenes seen
6,400 times each remain the easiest fit, the model can still memorize. That is why section 4 starts with a
measurement that needs no training (E4), and why the scene-count curve runs alongside.

## 2. What the fusion stage does (HoloBrain `TextImageDeformable2DEnhancer`)

Definitions. A **map** is one neck output per camera, a grid of 256-d cells. At HoloBrain's input size
320 × 256 the four maps have 40 × 32, 20 × 16, 10 × 8 and 5 × 4 cells (strides 8, 16, 32, 64), so 1280 +
320 + 80 + 20 = 1700 cells per camera and 5100 over our 3 cameras. Our current 1200 scene tokens are the
stride-16 and stride-32 cells (400 per camera). **Words** are the BERT tokens of the sentence after
`text_feat_map`, L ≤ 24 of them, 256-d each, with the [CLS] / [SEP] sub-sentence masks HoloBrain's BERT
produces.

One fusion layer, repeated 6 times with separate weights (`robo_orchard_lab/models/bip3d/feature_enhancer.py`
lines 187-231, `layers.py` `SingleScaleBiAttentionBlock`, `BiMultiHeadAttention`):

```
# a. bi-directional text-image attention, on the coarsest map only (stride 64: 20 cells per camera,
#    60 cells over the 3 cameras attend jointly, the cameras are flattened into one sequence)
v = LayerNorm(cells_64);  l = LayerNorm(words)
A = (W_q v) (W_k l)^T                       # one logit grid, 4 heads, inner width 1024
cells_64 += g_v * softmax_over_words(A)  W_vl l    # each cell reads the words
words    += g_l * softmax_over_cells(A^T) W_vv v   # each word reads the cells
# g_v, g_l: learned per-channel layer scales (init 1e-4, trained)
# b. deformable self-attention over ALL 4 maps (8 heads, 4 levels, 4 sample points per head and level)
#    plus an FFN (256 -> 2048 -> 256). This spreads the text signal from the stride-64 cells to the
#    finer cells, including the stride-16 / 32 cells we use.
# c. text self-attention (4 heads) plus FFN (256 -> 1024 -> 256), with sinusoidal position embedding
#    of the token index and HoloBrain's sub-sentence attention mask.
```

Outputs. All 4 maps, now sentence-conditioned, and the fused words `w_fused` [L, 256] (the words after 6
rounds of reading the image). The spatial enhancer then runs on the fused maps of strides 16 and 32 exactly
as it does now (`encode_visual_tokens`, `feature_level=(1, 2)`), giving 1200 tokens and their 3D positions.

Why it should ground. The checkpoint's fusion weights come from BIP3D grounding pretraining (referring
expressions over indoor scenes) and HoloBrain's RoboTwin training with instructions that name objects. The
word-to-cell logits `A` are a trained matching between a phrase and the cells it refers to. That is the
alignment our model has to invent from 60 scenes today.

What the sentence does to the frozen stage. The task instruction HoloBrain saw ("place the cup on the
coaster") and our rule ("don't touch the red mug") are both short imperative sentences with an object
phrase. The fusion stage never sees a verb-specific head, so there is no reason it needs the task form. E4
measures whether it lights up our named objects.

## 3. Design

### 3.1 Frozen fusion module (`main/lang/fusion.py`, new)

`FusedSceneEncoder(ckpt_dir)`, a wrapper that owns the HoloBrain pieces the fusion needs. It reuses the
existing `HoloBrainEncoder` for backbone / neck / backbone_3d / neck_3d / spatial_enhancer (composition, no
edit to `main/network/encoder.py`) and adds

- `feature_enhancer`: built from the checkpoint's `model.config.json` (`feature_enhancer` entry,
  `TextImageDeformable2DEnhancer`, 6 layers) with weights `feature_enhancer.*` from `model.safetensors`.
  **Phase-0 check**: list the keys with that prefix and assert the load is strict (no missing, no
  unexpected). Record the count in `docs/lang_filter/recon.md`;
- the text side: extend `HoloBrainText.embed` (or add `embed_dict`) to return HoloBrain's full `text_dict`
  (`embedded` after `text_feat_map`, `text_token_mask`, `position_ids`, `masks`), because the enhancer's
  text self-attention needs `position_ids` and `masks`. The existing 24-token cache stays for the adapter's
  fallback; `position_ids` and `masks` are recomputed from the tokenizer (deterministic, cheap).

Two methods, both `@torch.no_grad()`, fp32 (HoloBrain's GD path has no autocast; measure first, then decide
whether a `fusion_dtype = "bf16"` switch is worth it):

```python
def neck_features(imgs, depths) -> dict           # per frame, sentence-independent, what we STORE
    # feature_maps: 4 maps fp16 [n_cam, 256, h, w]; feature_3d: strides 16 / 32 only, fp16 [n_cam, 128, h, w]
    # (the spatial enhancer uses feature_3d of the levels it enhances), projection_mat, image_wh

def fuse(neck: dict, text_dict) -> (scene_tokens [B, 1200, 256], scene_pos [B, 1200, 3], w_fused [B, L, 256])
    # feature_enhancer on the 4 maps with the sentence -> spatial_enhancer on fused strides 16 / 32 ->
    # tokens and positions exactly as encode_visual_tokens does today
```

`fuse` with `text_dict=None` must reproduce today's tokens bit-for-bit (fusion skipped, spatial enhancer on
the raw maps). That is the V0 path, so one stored shard serves both V0 and V5. Test it against
`encode_visual_tokens` on a recorded frame.

### 3.2 What the buffer stores (a lockstep sidecar, `lang_version = 2`)

Scene tokens are now sentence-dependent, and relabeling draws a different sentence per sample, so the
tokens cannot be precomputed. Store the sentence-independent neck features per step and fuse online.

Do this as a second sidecar, `neck_buffer/`, next to `item_buffer/`, written in lockstep with the
`StepBuffer` exactly as `ItemSidecarBuffer` is (same capacity and row indices, rollback and flush mirrored,
`guard_writes` and `sync_arrays` cover it). The CrossSafe `StepBuffer` is not touched and keeps writing the
unfused `scene_tokens` / `scene_pos` as today, so a version-2 shard is a version-1 shard plus a sidecar:
every existing reader (V0, E6, the probes) works on it unchanged, and `fuse(neck, None)` has the stored
tokens to be tested against.

| field | shape, dtype | bytes |
|---|---|---|
| `neck_maps` | 4 maps, 3 cams, 1700 cells × 256, fp16 | 2.6 MB |
| `neck_3d` | strides 16 / 32, 3 cams, 400 × 128, fp16 | 0.3 MB |
| `projection_mat`, `image_wh`, `T_ego2world` | [3, 4, 4] fp32, [3, 2], [4, 4] | ~0.3 KB |
| StepBuffer and item sidecar | as today | 0.63 MB |

About 3.5 MB per step against 0.63 MB today. `LangSampler` loads the 5 frames a sample needs (t, t−k, 0,
t+1, t+1−k) as neck dicts when `scene_fusion` is on, and the tokens as today when it is off. Frame 0's
neck dict is cached per episode in RAM as the tokens are now (2.9 MB per episode).

Budget for the first experiment: T1 + T7 ur5-wsg, 30 + 30 training episodes plus 20 + 20 validation
episodes, about 75k steps, about 260 GB on local NVMe. The full 7-task two-robot warmup would be about
1.2 TB and is not collected until V5 passes.

**Validation must be re-collected too.** The existing validation shards hold tokens, not images, so the
neck features cannot be recovered. Re-collect T1 and T7 (20 each, ur5-wsg, masks on, the same seed stream
from 91M) in version 2. Same seeds give the same layouts but cuRobo gives different motions, so the pairs
will not be the report's 1,735 pairs. Every run of E5, V0' included, is scored on the new set, and the
existing V0 checkpoints are scored on it as well to link the two sets.

### 3.3 Where fusion runs

- **Training step** (`train.py _grad_step`): before `encode`, for every sample fuse its 5 frames with its
  sentence, under `no_grad`. Paired samples share frames but not sentences, so nothing is shared. 5 × 64 =
  320 frame-passes per step, each 6 layers over 5100 cells (the FFN, 256 → 2048 → 256, is about 11 GFLOP
  per layer per frame and dominates) plus the spatial enhancer. About 25 TFLOP per step, so roughly
  0.3-0.5 s on an A100 in TF32, and more than the current step. Run the passes in chunks of about 64
  frames (the FFN's hidden activation of 320 frames at once is 13 GB in fp32). **Bench it first**
  (`scripts/lang/bench_fusion.py`: ms per frame-pass at chunk 64, and the end-to-end step time against the
  current step time logged as `perf/steps_per_sec`). If the step time more than doubles, overlap the fusion
  with the backward pass (a second CUDA stream, or fuse the next batch in the prefetch thread on the GPU),
  or switch the frozen stage to bf16 if the parity test passes within 1e-2 on the tokens. Upload the neck
  arrays from pinned memory (930 MB per batch in fp16).
- **Collection and deployment** (`LangRolloutController`, `LangPolicyShim`): override `_encode_obs` in
  `LangRolloutController` (it is already wrapped there by a timer) to run `neck_features` on the fresh
  frame, keep the dict on the controller, and return the unfused tokens so the base `_capture_tick` writes
  the `StepBuffer` row as today. The shim then fuses from the kept dict with the episode's sentence instead
  of using the tokens it is handed. The frame history keeps the *fused* tokens and positions of frames t−k
  and 0 (the sentence is fixed within an episode), so one fusion pass per tick, two in monitor mode with
  the counterfactual sentence (the counterfactual history is kept separately). After the base capture, the
  controller appends the neck dict to `neck_buffer/` at the same row. Nominal collection fuses nothing.
- **Offline evaluation** (`eval_offline.py`): pairs fuse each sentence of the pair separately. The 1,735
  main pairs stay the scored set. Cache fused tokens per (state, sentence) within one evaluation run only.

### 3.4 What the trainable model receives

Unchanged modules. `SceneSet`, `SceneReadsWords`, `LangTrunkBlock`, actor, the critic blocks, the loss, the
five text switches and V1-V4 all stay as they are (the critic's readout changes, section 3.6). Two inputs
change:

- `scene_tokens`, `scene_pos`: the fused tokens (sentence-conditioned) instead of the sentence-free ones.
- `words`: with `fusion_words = True` (default) the text adapter receives `w_fused` of frame t (the words
  after reading the current image) instead of the cached BERT words. `in_proj` stays identity (256-d).
  With `fusion_words = False` the adapter gets the cached words as today, so the fusion changes only the
  scene side. This is the one new ablation.

Relabel consistency. The sentence given to `fuse` is the sample's sentence, the same string the adapter
sees. The shuffled-sentence control shuffles before fusion, so it stays a true control. Paired samples fuse
the same frames twice, once per sentence.

Deployment cost note. The text adapter can no longer run once per episode when `fusion_words` is on, because
`w_fused` depends on the frame. It is 1.6 M parameters on 29 tokens, negligible next to the fusion pass.
`text_out` is recomputed every tick (`set_instruction` keeps the sentence, the shim calls `encode_text`).

Text inputs of the fusion. The enhancer needs HoloBrain's full `text_dict`: `embedded` (the cached 24 × 256
words, exactly what `TextStore` holds), `text_token_mask`, `position_ids` and the sub-sentence attention
`masks`. Extend the text cache with `position_ids` [N, 24] int16 and `masks` [N, 24, 24] bool (cache
version bump, rebuilt by `build_text_cache.py`), rather than tokenizing at training time.

### 3.5 Switches (`LangConfig`)

| field | default | meaning |
|---|---|---|
| `scene_fusion` | True for new runs (`lang_version = 2`), False for configs saved before | run the frozen `feature_enhancer` with the sentence before the spatial enhancer |
| `fusion_words` | True | the text adapter reads `w_fused` instead of the cached words (needs `scene_fusion`) |
| `fusion_dtype` | "fp32" | "bf16" allowed after the parity test |
| `critic_readout` | "mlp" for new runs, "softmin" for configs saved before the field | section 3.6; `softmin_T_h` is unused under "mlp" |

A version-1 shard with `scene_fusion = True` is a hard error (no neck features stored). A version-2 shard
with `scene_fusion = False` is V0 on the same episodes.

### 3.6 Critic readout: an MLP instead of the softmin (decision 6)

Today (`LangCriticHead`). After the two critic blocks, a value head gives one number per body token
(V_i, 22 of them, padding filled with 10⁴) and one per readout query (V_k, 4), and Q = −T·log Σ exp(−V/T)
with T = 0.05 in h units, a soft minimum over the 26 values. The intent was CrossSafe's: for a contact
rule the link closest to the item holds the minimum, for spill, tilt, displacement and zones a query does.

Why replace it. The gradient of Q with respect to V_i is the softmin weight w_i = exp(−V_i/T) / Σ_j
exp(−V_j/T). With T = 0.05 and values spread over a few tenths, one token takes nearly all of each sample's
gradient and the other 25 heads learn only on samples where they hold the minimum. The readout is therefore
a hard routing decision the critic has to get right before anything else, and for the rules whose minimum
should sit in a query, that query must first undercut all 22 link values. An MLP over a fixed set of pooled
tokens gives every sample a dense gradient into r, the queries and the body, and makes no assumption about
which token "owns" the rule. What is lost: the per-link "which part is in danger" readout and the readout's
parity with CrossSafe. Both become an ablation axis (the switch), not a default.

The readout:

```
# after the critic's 2 blocks: b [B, 22, 256] with mask m [B, 22], r [B, 1, 256], q [B, K, 256]
b_mean = (b * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)    # masked mean, [B, 256]
parts  = [r[:, 0]] + [q[:, k] for k in range(K)] + [b_mean]                 # K + 2 vectors of 256
z      = cat([RMSNorm(x) for x in parts], -1)                               # [B, (K + 2) * 256], 1536 at K = 4
Q      = Linear(512, 1)(SiLU(Linear((K + 2) * 256, 512)(z)))               # one scalar
```

- One RMSNorm(256) shared by the parts, then the concatenation; hidden width 512, SiLU; about 0.8 M
  parameters per critic head, each twin has its own.
- `b_mean` is in the set because the action effects are injected into the body tokens (`act_proj` on the
  8 effect numbers, post-action positions), so it is the most direct route from the action to Q; r and the
  queries see the action only through the self-attention sub-layer d. Padding tokens are excluded by the
  mask. With `lang_queries` off (K = 0) the input is 2 × 256; with `lang_rtoken` off, r is the static token
  after the blocks, still included.
- `forward` returns `(Q, None)` under "mlp": nothing reads the per-token values (`SafetyFilter` and the
  trainer call `qmin` only, checked). `LangTwinCritic` is unchanged.
- Checkpoints: `critic_readout` is saved in `lang_config`; `load_policy` and `Evaluator` build the head
  the config names, so the running `rounds_ur5_g095_tf32` (softmin) and the E2 checkpoints still load and
  score. `grounding_summary.FIXED` gains `critic_readout`.
- Tests: Q is invariant to the order of padded body tokens and to padding count (two embodiments in one
  batch); K = 0 builds; a saved config without the field loads as "softmin" and reproduces an old
  checkpoint's Q bit-for-bit.

## 4. Experiments, in order

### E4. Does the frozen fusion light up the named object? (no training, first)

On the 38 usable validation episodes of the report (sidecar token masks are on), every 25th tick, every
sampleable rule with a training-template sentence:

- `delta_j = || fuse(neck, sentence)_j − fuse(neck, None)_j ||` per scene token j (1200, frame t), and the
  same for `w_fused` versus the cached words (how much the words moved);
- **enrichment** = mean `delta_j` over the named item's mask tokens ÷ mean over the other tokens;
- **contrast** = for pairs of rules of the same consequence in the same scene (the report's object pairs),
  the share of the named item's mask tokens where `delta` under its own sentence exceeds `delta` under the
  other sentence;
- the raw bi-attention map `A` of layer 1 and layer 6 (hook `BiMultiHeadAttention`): the share of the
  object words' attention mass that lands on the stride-64 cells covering the named item (project the mask
  to the 5 × 4 grid).

Report per task (T1, T7) with episode-cluster bootstrap intervals, plus 6 overlay images (delta heat map on
the camera image, named item outlined) for the coauthor page. **Gate**: enrichment ≥ 2 and contrast ≥ 0.7 on
T1. If the frozen stage does not single out the named object on our scenes, stop here and report; the
fallback is the referent phrase alone as the fusion text (an ablation, not the default), then the
scene-count curve decides.

### E5. V5 short runs (the E2 setup, re-collected in version 2)

Collect T1 + T7 ur5-wsg warmup in version 2 (same seeds as `data/lang/warmup`, so the same 60 layouts)
and the version-2 validation set of 3.2, then 6,000 steps, batch 64, lr 3e-4, γ 0.95, `actor_hidden 0`
(pinned to V0's settings), seeds 0 and 1, scored on the new validation pairs with `grounding_summary.py`
(its `FIXED` settings gain `scene_fusion`, `fusion_words` and `lang_version`; the reference run of a table
is V0', not the report's V0):

| run | setting |
|---|---|
| V0' | `scene_fusion` off on the version-2 shards (the report's V0 checkpoints scored on the new validation set link the two sets; V0' should land where they do) |
| V5 | `scene_fusion` on, `fusion_words` on |
| V5w | `scene_fusion` on, `fusion_words` off |
| V5-d2 | V5 on both robots' T1 / T7 (120 scenes), scored on the same ur5 pairs |
| V0'-softmin | V0' with `critic_readout = "softmin"`: the readout change on its own (decision 6) |

All other rows use the MLP readout. V0' versus V0'-softmin is the readout's effect, V5 versus V0' is the
fusion's, each on the same data and pairs.

Also report, for each: rule-kind pairs, not-yet pairs, the training-scene score (`grounding_train`), the
learning curve at 1,000-step checkpoints, and the in-batch pair accuracy. The training-scene score and the
learning curve are the memorization diagnostics.

**Decision rule.** V5 ≥ 0.85 on the main score: adopt (collect the full warmup in version 2, retrain the
official model at γ 0.95). V5 clearly above V0' but below 0.85 and still rising with scenes (V5-d2 > V5):
collect more T1 / T7 scenes and re-test. V5 at the V0 level with E4 passed: the model ignores a working
grounding signal, memorization dominates, and the next lever is the data (scene count, scene diversity,
fewer repeats per scene), not the architecture.

### E6. Scene-count curve (parallel, V0 settings)

Report section 8 as written: 15, 30, 60 and 120 scenes, same pairs. Runs on the existing version-1 data,
no new collection.

## 5. Implementation steps for the code agent

1. **Recon** (`recon.md`): checkpoint keys under `feature_enhancer.*` (count, strict load), the
   `model.config.json` enhancer entry, memory of one fusion pass at a chunk of 64 frames, time per
   frame-pass. Also the provenance check of section 7, item R2.
2. `main/lang/fusion.py`: `FusedSceneEncoder`, `neck_features`, `fuse`; `HoloBrainText.embed_dict`; the
   text cache extension of 3.4. Tests: `fuse(neck, None)` equals `encode_visual_tokens` (tokens and
   positions, atol 1e-5 fp32); a sentence changes the tokens; two sentences give different tokens; a batch
   of 2 equals two batches of 1; the enhancer's output on one recorded frame equals the full BIP3D
   `_forward` path run on the same inputs (the integration check that the order and the inputs match).
3. `neck_buffer/` sidecar (`main/lang/neck_buffer.py`, modelled on `ItemSidecarBuffer`): the fields of
   3.2, lockstep with the `StepBuffer`, `lang_version = 2` in `lang_episodes.jsonl` lines and the shard
   meta, the hard error of 3.5. `LangSampler` loads neck dicts for the 5 frames when `scene_fusion` is on.
   The version-1 path stays as it is (E6 uses it).
4. `train.py`: fusion before `encode` (no grad, chunked), `fusion_words` routing into `text.lookup` vs
   `w_fused`. `scripts/lang/bench_fusion.py`. Log `perf/fusion_ms` and `perf/step_ms`.
4b. `network.py`: `critic_readout` in `LangCriticHead` (section 3.6), the config field with its
   saved-config default, the tests of 3.6. This step is independent of the fusion and can land first.
5. `rollout.py`: the `_encode_obs` override, fused frame history, counterfactual history, the sidecar
   append after each capture. `collect warmup --stream v2` and `collect validation --stream v2` for T1 /
   T7 ur5 (same seed streams as the existing warmup and `val_grounding`).
6. `eval_offline.py`: per-sentence fusion with an in-run cache keyed by (state, sentence).
   `grounding_summary.py` records `scene_fusion`, `fusion_words`, `lang_version` and the validation set in
   the comparability check, with V0' as the reference of a version-2 table.
7. E4 script `scripts/lang/fusion_probe.py` with the overlays. Then E5, E6.
8. Docs: `DESIGN_CONTEXT.md` section 5.1 (add the fusion stage as part of the frozen path and the two
   switches; the twin-critic paragraph: MLP readout by default, softmin as the ablation), section 8 (γ 0.95 only, decisions 1-5 above), `CLAUDE.md` working rules (γ line), and the
   coauthor page generator (a "V5" section fed from E4 / E5 outputs, no hard-coded numbers).

Do not edit `main/network/encoder.py` or anything under `envs/` for this; compose in `main/lang/`. If
`HoloBrainText` (under `envs/`) must change, make the extension a new method and ask before touching the
existing `embed`.

## 6. Open points for Yuxuan

1. Storage: 3.5 MB per step is acceptable for T1 / T7 only (260 GB with validation). Before a full warmup
   (1.2 TB) we need
   either more disk or the cheaper variant (cache fused tokens per episode, rule and one fixed paraphrase,
   losing paraphrase diversity on the frozen side). Decide after E5.
2. Fusion text: the whole sentence (decided). The referent phrase alone is the fallback ablation only if E4
   fails on the whole sentence.
3. Frozen or not: the fusion stage stays frozen. Unfreezing its layer scales `g_v`, `g_l` (a few hundred
   parameters) is a possible later ablation, not part of V5.

## 7. Review of the current code (round 5, at 3e833f8) and what it means for this plan

Read in full: `network.py`, `train.py`, `data.py`, `eval_offline.py` (Evaluator, pairs, strata),
`rollout.py`, the episode loop of `collect.py`, `config.py`, the round-4 response and the two grounding
reports. The round-4 items are done as described. Nothing below invalidates the report's numbers. Items are
"R" (bears on the V5 implementation), "C" (code), "D" (needs a decision from Yuxuan).

**R1. The report's pairs cannot be reused for V5.** Validation shards store tokens, not images, and the
fusion needs the neck maps. The report's 1,735 pairs are therefore the last numbers on that set. Section
3.2 now says to re-collect validation in version 2 and to score the old V0 checkpoints on the new set so
the two sets are linked. The agent must not present a V5 number next to the report's V0 number as if they
were on the same pairs.

**R2. Provenance of the fusion weights is an assumption until checked.** The plan's argument rests on the
enhancer having learned phrase-to-region alignment on far more data than ours. `HoloBrain_v0.0_GD` is
trained from a "pretrain" checkpoint whose origin the code does not state. Check: compare the
`feature_enhancer.*` tensors of `HoloBrain_v0.0_GD` with those of `hf://HorizonRobotics/BIP3D_Tiny_Det`
(cosine per tensor). Identical or near-identical means the fusion is BIP3D's grounding-trained stage, kept
frozen or lightly tuned during HoloBrain training. Far apart means it was retrained on RoboTwin
instructions only, which name a few objects per task, and the pretrained-alignment argument is weaker (E4
then carries the whole decision). Record the result in `recon.md`.

**R3. The `StepBuffer` must keep writing unfused tokens.** CrossSafe's `_capture_tick` hands the same
`encoded` tuple to the buffer write and to the policy encoder. If `_encode_obs` returned fused tokens, the
buffer would store sentence-dependent tokens on filter episodes and sentence-free ones on nominal episodes,
and relabeling would silently train on a mix. Section 3.3 now fixes the split: the override returns unfused
tokens for the buffer, the shim fuses from the kept neck dict.

**R4. The shim's frame history must be per sentence.** `LangPolicyShim.frames` keeps one deque of
(tokens, positions). With fusion, a deque of fused frames is valid for one sentence only. Monitor mode
needs two histories (named and counterfactual), and `set_instruction` called mid-episode (it is not today,
but nothing prevents it) must clear them. Add the assert.

**R5. The target encoder sees the same fused frames.** `encode(..., target=True)` reuses the batch's
`scene_tokens_next`; with fusion, the t+1 frames are fused once with the sample's sentence and fed to both
the live and the Polyak encoder. Do not fuse twice (cost) and do not fuse t+1 with a different paraphrase
(it would change the target's sentence). The V3 `ref_center_next` handling in `encode` is the pattern.

**R6. `fuse(neck, None)` parity is the gate for the sidecar, not only a unit test.** The stored
`scene_tokens` of every version-2 row must equal `fuse(neck_row, None)` within fp16 rounding. Run it on
100 random rows of the first shard before training on it (`scripts/lang/check_shards.py --fusion-parity`).
A mismatch means the sidecar and the StepBuffer rows are misaligned (a rollback mirrored on one side only),
the exact bug the item sidecar's lockstep guards against.

**R7. Chunk the fusion, and keep it out of the autograd graph.** `_grad_step` must call the frozen stage
under `torch.no_grad()` in chunks of about 64 frames and keep only the 1200-token outputs (fp32, 5 frames ×
64 × 1200 × 256 × 4 B = 390 MB). Keeping the 5100-cell maps of all frames alive at once is 13 GB in the FFN
alone (3.3). Activation checkpointing does not apply to a frozen no-grad stage.

**C1. Memorization has an obvious handle the report does not name.** Frame 0 (and any frame) is an
episode fingerprint, and each training episode is seen about 6,400 times in a 6,000-step run (report
conclusion 6). The sampler offers nothing against it: no scene-level augmentation is possible on frozen
tokens, and the paired batch halves the distinct transitions per batch. The plan's data lever (E6, more
scenes) is the right one, and the fusion is meant to make the name → object map cheaper to read than the
fingerprint table. Report the per-episode repeat count (`samples / episodes`) in every E5 table so this
stays visible.

**C2. V3 was a weak oracle by construction.** A zero-initialized additive vector on the named object's tokens,
placed before scene-reads-words, is a signal the trunk has to learn to use, under the same memorization
pressure as everything else. "The model ignores the oracle" (report 4.6) is consistent with "the oracle was
too easy to ignore". The fusion is a different kind of oracle test because it rewrites the tokens' content.
If V5 also fails and a true ceiling is still wanted, the hard version is masking: scene tokens outside the
named item's mask (plus the robot) zeroed, which the model cannot ignore. Not part of V5.

**C3. The referent-phrase uniqueness defect (round-4 audit, open).** `language._answers` lets "the box"
pass next to a "coffee box" and "the bottle" next to a "pill bottle". 153 of 1,279 T1 already-harmed pairs
use such a phrase. The audit showed they score like the clean pairs, so the conclusions stand, but a
re-collected validation set should not carry the defect into V5. **D1: approve the head-noun match in
`_answers` (a change under `envs/`), then regenerate sampleable flags and the text caches before the
version-2 collection.** Recommended.

**C4. `recall_horizon_ticks` pinned to 7 in the E2 runs at γ 0.95** (`e2.cmds`, to match V0). It only
affects `fails_soon_recall` and the `q/recall_soon` metric, not training or the pair score. For E5 use the
derived value (14 at 0.95) in every run, V0' included, so the diagnostics are read on one horizon.

**C5. Deployment `_capture_tick` ordering.** `run()` captures once before the base run in non-collect
modes so frame 0 is s₀. With fusion, that first capture must happen after `set_instruction`, or frame 0 is
fused with no sentence (the V0 path) while later frames are fused with the sentence. Assert in the shim
that a sentence is set before the first fused push when `scene_fusion` is on.

**C6. The text cache's `extra_path` grows per process** (`text_cache_extra.pt` in the run, the evaluator's
own `_eval` file). With `position_ids` and `masks` added (3.4), bump the cache version and refuse a mixed
load rather than filling missing fields with defaults; a wrong sub-sentence mask changes the fusion
silently.

**C7. `grounding_summary.FIXED` is the only guard against mixed tables.** It compares γ, actor head, TF32,
steps, tasks, embodiments and the scored set. Add `scene_fusion`, `fusion_words`, `lang_version`, the
sidecar's `collect_hash` and the validation set's episode ids, and make the reference run explicit
(`--reference V0'`) instead of "the row named V0".

**C8. `eval_offline` caches state reads per chunk of 192 states, and each state is scored under several
sentences** (train, held-out axes, base, paraphrases). With fusion each (state, sentence) is one fusion pass
of 3 frames. The 1,735-pair set has about 2,600 (state, rule) rows and roughly 5 sentences each, so about
40k frame-passes per run, a few minutes. Fine, but cache by (state, sentence) inside the chunk so the
"base" sentence of one rule, which is another rule's train sentence, is not fused twice.

**C9. The nominal-run validation labels are one cuRobo draw.** Already in the report's limits. For V5 the
re-collection makes a second draw of the same layouts, which is an opportunity: scoring the old V0
checkpoints on the new set measures how much the validation score itself moves between draws of the same
layouts. Report that number (R1); it is the floor on any difference worth discussing.

**D2. Whether E4 should also test the referent phrase alone.** The decision says the whole sentence. E4 is
cheap (no training), so measuring both the sentence and the bare referent phrase ("the red mug") in the
same probe costs one more pass per rule and tells us at once whether the verb frame hurts the frozen
grounding. Recommended as a reported column, with the whole sentence staying the default for V5.

## 8. Implementation judgement (code agent, 2026-10-09)

The plan is sound and is implemented as written, with the changes below. The contract the code follows is
`docs/lang_filter/fusion_impl.md` (facts with anchors, interfaces, data format, tests).

**Order of work.** E4 cannot use the report's validation shards (they hold tokens, R1), so the order is: code
→ version-2 *validation* collection (T1 / T7, 40 episodes, replaying the recorded seeds) → E4 → only if E4
passes, the version-2 warmup (the 60 recorded layouts, replayed) → E5. E6 runs now on version-1 data.

**What the code showed that the plan did not know.**
- HoloBrain's own enhancer forward does not mask padded words (`bool.masked_fill(-9e15)` stays True) and tiles
  the text mask head-major, so a padded or mixed-length batch differs from HoloBrain's single-sentence
  inference. The fusion therefore runs **unpadded, grouped by sentence length** (identical masks within a
  group), using the official forward unchanged.
- `position_ids` and the sub-sentence `masks` are a closed form of the sentence length: no text-cache change.
- The deformable attention is the pure-PyTorch fallback (no kernel installed): memory, not the FFN, sets the
  chunk size; a per-level drop-in (`msda_lowmem`, exact up to summation order) replaces its stacking.
- The spatial enhancer predicts depth from the fused maps, so **token positions also depend on the sentence**.
  The trainer, the shim and the probes carry fused positions (V4b included).
- The fusion layers write `LN(x) + γ·Δ` (not `x += ...`) and then post-norm image layers: they change every
  cell even for an empty sentence. E4 measures deltas against the **content-free sentence** ("" → [CLS][SEP])
  and the pair's other sentence; the delta against the unfused tokens is reported alongside.
- Stored neck maps are fp16, so stored-row parity (R6) uses a measured tolerance plus a CPU alignment check
  (`episode_id`, `t_in_ep` in the neck rows); the 1e-5 test runs on live fp32 neck dicts.
- Sizes: T1 + T7 ur5 is ~30k rows (18,611 warmup + 11,664 validation), ~108 GB with the neck sidecar (+67 GB
  for V5-d2), not 260 GB. `/work/nvme` is the shared Lustre allocation (~1.3 TB headroom).
- R2: the `feature_enhancer` of `pretrain/` (the file our neck, spatial enhancer and `text_feat_map` come from)
  has median cosine 0.90 to BIP3D_Tiny_Det (a detection model), 0.88-0.92 to grounding-dino-tiny on the
  attention matrices, 0.986 to `post_training_robotwin`, none identical: Grounding-DINO-family weights,
  retrained through HoloBrain's stages, not a frozen copy. E4 carries the decision.

**Decisions taken in the code.**
1. `scene_fusion` defaults to False (not True) until V5 is adopted: every shard today is version 1, and a True
   default would make every V0-style command a hard error. V5 runs pass `--set scene_fusion=true`.
2. `critic_readout` defaults to "mlp" (decision 6); saved configs without it load "softmin".
3. E6 pins `critic_readout = softmin` and V0's other settings, so its 60- and 120-scene points are the
   report's V0 and the queued d2 run. New sizes: 8 and 15 episodes per task.
4. `lang_version` is the shard's data version (2 only with `--lang-version 2`), kept in `collect_hash`.
5. One `fusion_dtype` for training, evaluation and the rollout (fp32 default; tf32 / bf16 after the bench).
6. Rounds training refuses `scene_fusion` for now (no neck ring); E5 is offline.
7. D1 approved (head-noun match in `_answers`): 0 sampleable flags flip on the 840 existing records.

**Decision 1, live runs.** The γ 0.9 run that is live is `rounds_ur5_g090_mlp` (the `_tf32` runs were retired
on 10-08): chain stage `rounds_mlp_g090` in `runs/lang_pipeline_ur5` and the pending general job for the same
run. Stopping it is left to Yuxuan (this session may not cancel jobs). `rounds_ur5_g095_mlp` (MLP actor,
softmin critic) is the official run.
