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

### 3.2 What the buffer stores (new shard format, `lang_version = 2`)

Scene tokens are now sentence-dependent, and relabeling draws a different sentence per sample, so the
tokens cannot be precomputed. Store the sentence-independent neck features per step instead and fuse
online:

| field | shape, dtype | bytes |
|---|---|---|
| `neck_maps` | 4 maps, 3 cams, 1700 cells × 256, fp16 | 2.6 MB |
| `neck_3d` | strides 16 / 32, 3 cams, 400 × 128, fp16 | 0.3 MB |
| `projection_mat`, `image_wh` | [3, 4, 4] fp32, [3, 2] | ~0.2 KB |
| everything else | as today (robot fields, h, sidecar) | ~0.1 MB |

About 2.9 MB per step against 0.63 MB today. The `scene_tokens` / `scene_pos` fields are dropped from
version-2 shards (they are reproducible from `fuse(neck, None)`). Sampling and the ring work as before.
`LangSampler` loads the 5 frames a sample needs (t, t−k, 0, t+1, t+1−k) as neck dicts.

Budget for the first experiment: T1 + T7 ur5-wsg, 30 + 30 episodes, about 45k steps, about 130 GB on local
NVMe. The full 7-task two-robot warmup would be about 1 TB and is not collected until V5 passes.

### 3.3 Where fusion runs

- **Training step** (`train.py _grad_step`): before `encode`, for every sample fuse its 5 frames with its
  sentence, under `no_grad`. Paired samples share frames but not sentences, so nothing is shared. 5 × 64 =
  320 frame-passes per step, each 6 deformable layers over 5100 cells plus the spatial enhancer. Rough
  estimate 10-15 TFLOP per step, so about 0.2-0.4 s on an A100 in fp32 / TF32. **Bench it first**
  (`scripts/lang/bench_fusion.py`: ms per frame-pass at batch 320, and the end-to-end step time against the
  current step time). If it more than doubles the step time, put the fusion in the `PrefetchSampler` thread
  on the GPU, or switch the frozen stage to bf16 if the parity test passes within 1e-2 on the tokens.
- **Collection and deployment** (`LangRolloutController`, `LangPolicyShim`): per tick, `neck_features` on
  the fresh frame, then `fuse` with the episode's sentence. The frame history keeps the *fused* tokens and
  positions of frames t−k and 0 (the sentence is fixed within an episode), so one fusion pass per tick, two
  in monitor mode with the counterfactual sentence (the counterfactual history is kept separately). Nominal
  collection stores `neck_features` only.
- **Offline evaluation** (`eval_offline.py`): pairs fuse each sentence of the pair separately. The 1,735
  main pairs stay the scored set. Cache fused tokens per (state, sentence) within one evaluation run only.

### 3.4 What the trainable model receives

Unchanged modules. `SceneSet`, `SceneReadsWords`, `LangTrunkBlock`, actor, critics, the loss, the five text
switches and V1-V4 all stay as they are. Two inputs change:

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
`text_out` is recomputed every tick.

### 3.5 Switches (`LangConfig`)

| field | default | meaning |
|---|---|---|
| `scene_fusion` | True for new runs (`lang_version = 2`), False for configs saved before | run the frozen `feature_enhancer` with the sentence before the spatial enhancer |
| `fusion_words` | True | the text adapter reads `w_fused` instead of the cached words (needs `scene_fusion`) |
| `fusion_dtype` | "fp32" | "bf16" allowed after the parity test |

A version-1 shard with `scene_fusion = True` is a hard error (no neck features stored). A version-2 shard
with `scene_fusion = False` is V0 on the same episodes.

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

### E5. V5 short runs (the E2 setup, so the numbers are directly comparable)

Collect T1 + T7 ur5-wsg warmup in the version-2 format (same seeds as `data/lang/warmup`, so the same 60
scenes), then 6,000 steps, batch 64, lr 3e-4, γ 0.95, `actor_hidden 0` (pinned to V0's settings), seeds 0
and 1, scored on the report's 1,735 main pairs with `grounding_summary.py`:

| run | setting |
|---|---|
| V0' | `scene_fusion` off on the version-2 shards (must reproduce V0 within seed noise, 0.56-0.61) |
| V5 | `scene_fusion` on, `fusion_words` on |
| V5w | `scene_fusion` on, `fusion_words` off |
| V5-d2 | V5 on both robots' T1 / T7 (120 scenes), scored on the same ur5 pairs |

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
   `model.config.json` enhancer entry, memory of one fusion pass at batch 320 frames, time per frame-pass.
2. `main/lang/fusion.py`: `FusedSceneEncoder`, `neck_features`, `fuse`; `HoloBrainText.embed_dict`.
   Tests: `fuse(neck, None)` equals `encode_visual_tokens` (tokens and positions, atol 1e-5 fp32); a
   sentence changes the tokens; two sentences give different tokens; batch of 2 equals two batches of 1.
3. Buffer version 2 (`main/lang/buffer_v2.py` or a version switch in the existing writer): the fields of
   3.2, `lang_version = 2` in the header, the hard error of 3.5. `LangSampler` loads neck dicts for the 5
   frames. Keep the version-1 path working (E6 uses it).
4. `train.py`: fusion before `encode` (no grad), `fusion_words` routing into `text.lookup` vs `w_fused`.
   `scripts/lang/bench_fusion.py`. Log `time/fusion_ms` and `time/step_ms`.
5. `rollout.py`: fused frame history, counterfactual history, one `neck_features` per tick. Nominal
   collection writes neck features. `collect warmup --stream v2` for T1 / T7 ur5 (same seeds as the
   existing warmup).
6. `eval_offline.py`: per-sentence fusion with an in-run cache. `grounding_summary.py` accepts version-2
   runs and records `scene_fusion` / `fusion_words` in the comparability check.
7. E4 script `scripts/lang/fusion_probe.py` with the overlays. Then E5, E6.
8. Docs: `DESIGN_CONTEXT.md` section 5.1 (add the fusion stage as part of the frozen path and the two
   switches), section 8 (γ 0.95 only, decisions 1-5 above), `CLAUDE.md` working rules (γ line), and the
   coauthor page generator (a "V5" section fed from E4 / E5 outputs, no hard-coded numbers).

Do not edit `main/network/encoder.py` or anything under `envs/` for this; compose in `main/lang/`. If
`HoloBrainText` (under `envs/`) must change, make the extension a new method and ask before touching the
existing `embed`.

## 6. Open points for Yuxuan

1. Storage: 2.9 MB per step is acceptable for T1 / T7 only (130 GB). Before a full warmup (1 TB) we need
   either more disk or the cheaper variant (cache fused tokens per episode, rule and one fixed paraphrase,
   losing paraphrase diversity on the frozen side). Decide after E5.
2. Fusion text: the whole sentence (decided). The referent phrase alone is the fallback ablation only if E4
   fails on the whole sentence.
3. Frozen or not: the fusion stage stays frozen. Unfreezing its layer scales `g_v`, `g_l` (a few hundred
   parameters) is a possible later ablation, not part of V5.
