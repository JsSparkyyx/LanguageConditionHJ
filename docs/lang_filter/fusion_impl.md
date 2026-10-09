# V5 implementation notes (FUSION_PLAN, 2026-10-09)

What the code agent decided while implementing `FUSION_PLAN.md`, and the interfaces every module builds
against. `FUSION_PLAN.md` section 8 summarises the judgement; this file is the contract. Recon reports
(read-only, 2026-10-09) back every statement here; anchors are file:line at commit bca6bc5.

## 1. Facts that changed the plan

| plan says | code says | consequence |
|---|---|---|
| E4 on the report's 38 validation episodes | v1 shards hold tokens, no images or neck maps | E4 runs on the **v2 validation set**; collect it first, warmup only after E4 |
| text_token_mask masks the padding (2a) | `layers.py:662-681`: `bool.masked_fill(mask==0, -9e15)` stays all True, so pads are NOT masked | fuse **unpadded**: group frames by sentence length n, slice the words to n (HoloBrain's own inference is one unpadded sentence) |
| `masks` tiled per head | `feature_enhancer.py:220-225` uses `.repeat(4,1,1)` (head-major); `nn.MultiheadAttention` reads batch-major | same fix: within a group all masks are identical, so the tiling order cannot matter |
| store `position_ids` / `masks` in the text cache (3.4, C6) | both are a closed form of the true length n (`[CLS]`=0, w1=0, ..., `[SEP]`=n-2; block mask over 1..n-1 plus the identity) | no cache change, no version bump; derived from the cached `mask` |
| FFN dominates memory (R7) | the deformable attention is pure PyTorch (transformers 4.57.6 has no kernel; no mmcv): ~250 MB per frame at peak, ~16 GB at 64 frames | a low-memory drop-in for `self_attn.func` (per-level sum, exact up to summation order), chunk 32 |
| only the tokens change (3.4) | the spatial enhancer predicts depth from the fused maps, so `scene_pos` changes with the sentence too | `fuse` returns tokens AND positions; the batch carries both; V4b reads the fused positions |
| `cells += ...` (2a) | `x = LN(x) + gamma * delta` per layer, then post-norm image and text layers | E4 compares against a **content-free sentence** ("" -> [CLS][SEP]) and the pair's other sentence, not only against the unfused tokens |
| bit-for-bit `fuse(neck, None)` | stored neck maps are fp16, the stored tokens came from fp32 maps | 1e-5 parity on live fp32 neck dicts; stored-row parity (R6) with a measured tolerance, plus a CPU alignment check on `episode_id` / `t_in_ep` |
| 75k steps, 260 GB | T1+T7 ur5: 18,611 warmup + 11,664 validation rows | ~30k rows x 3.6 MB = ~108 GB (+67 GB for franka, V5-d2) |
| `collect warmup --stream v2` | `--stream` picks the seed stream (validation only); warmup seeds are positions in 7-task schedules | `--lang-version 2` plus `--replay <v1 shard>`: replays the recorded (task, host, episode_id) |
| append the neck after the capture (3.3) | StepBuffer rows are written one capture late (`_pending`) through `LockstepItemWriter` | the neck rides in the pending entry (`_neck`) and is popped by a lockstep writer inside the same `append` |
| R2: is the fusion BIP3D's? | cosine pretrain vs BIP3D_Tiny_Det (a detection model) 0.90 median, vs post_training_robotwin 0.986; none identical; grounding-dino-tiny 0.88-0.92 on the attention matrices | descends from Grounding DINO / BIP3D weights, retrained; not a frozen copy. We use `pretrain/`, the same file as our neck, spatial enhancer and text_feat_map; never mix checkpoints |
| stop `rounds_ur5_g090_tf32` | the `_tf32` runs were retired on 10-08; the live γ 0.9 run is `rounds_ur5_g090_mlp` (job 22772155 running, 22764785 pending, chain stage `rounds_mlp_g090`) | the user stops it (hold file + scancel); this session may not cancel jobs |

## 2. Decisions taken here (judgement)

1. `scene_fusion` defaults to **False** (not True as in plan 3.5) until V5 passes its gate. Every existing
   shard is version 1, so a True default would turn every V0-style command (E6, V0', probes, the official
   rounds) into a hard error. V5 commands pass `--set scene_fusion=true`. Flip the default when the full
   warmup is re-collected in version 2 (the plan's "adopt" branch).
2. `critic_readout` defaults to "mlp" (decision 6); a saved config without it loads "softmin".
3. **E6 pins `critic_readout = softmin`** (and V0's other settings: γ 0.95, actor_hidden 0, recall 7), so
   its 60-scene point is the report's V0 and its 120-scene point the queued d2 run. Sizes: 8 and 15 episodes
   per task (new) + 30 (V0, 29 usable T1) + d2.
4. `lang_version` (config) is the **data** version of a shard: 2 only for shards collected with
   `--lang-version 2` (it stays in `collect_hash`). The class default stays 1.
5. Fusion precision is one config field for every place the stage runs (train, eval, rollout), so training
   and deployment never differ. Default fp32; tf32 / bf16 after the bench and the parity numbers.
6. The fusion stage is never a submodule of `LangPolicyEncoder` (deep-copied for the target, optimised,
   saved). It is built from the checkpoint in `eval()`, `requires_grad_(False)`, under `no_grad`.
7. E4's gate stays as written (enrichment ≥ 2, contrast ≥ 0.7 on T1), computed against the content-free
   sentence; the delta against the unfused tokens is reported alongside. D2: the bare referent phrase is a
   reported column (cheap).
8. Rounds mode refuses `scene_fusion` (no neck ring for the replay buffer yet); E5 is offline training.
9. D1 approved by Yuxuan (2026-10-09): the head-noun fix in `language._answers`. Measured on all 840 records:
   0 sampleable flags flip; v2 collection picks it up at collection time; no cache rebuild needed.

## 3. Data format: `neck_buffer/` (version-2 shards)

Next to `buffer/` and `item_buffer/`, same capacity, ring and row indices as the StepBuffer.

| field | shape per row | dtype | content |
|---|---|---|---|
| `neck_maps` | [3, 1700, 256] | fp16 | the 4 neck maps per camera, cells level-major (stride 8: 32x40, 16: 16x20, 32: 8x10, 64: 4x5), row-major within a level |
| `neck_3d` | [3, 400, 128] | fp16 | neck_3d levels 1 and 2 (stride 16 then 32), row-major |
| `projection_mat` | [3, 4, 4] | fp32 | ego-frame projection (obs, cast to fp32 as every current use does) |
| `image_wh` | [3, 2] | fp32 | (320, 256) per camera |
| `T_ego2world` | [4, 4] | fp32 | the positions' out_extrinsic |
| `episode_id` | [] | int64 | alignment check against the item sidecar |
| `t_in_ep` | [] | int32 | alignment check |

`header.json`: `capacity`, `n`, `head`, `fields` {name: [shape, dtype]}, `level_shapes`, `neck_version` 1.
Writes: memmaps flushed, then the header atomically (tmp + replace), exactly like `ItemSidecarBuffer.flush`.
Readers open `np.load(mmap_mode="r")` and never create files.

## 4. Interfaces

### 4.1 `main/lang/fusion.py`

```python
LEVEL_SHAPES = ((32, 40), (16, 20), (8, 10), (4, 5))   # (h, w) per neck level at 320 x 256
N_CELLS = 1700                                          # per camera
TOKEN_LEVELS = (1, 2)                                   # 320 + 80 = 400 tokens per camera

def text_dict_for_length(n: int, L: int, device) -> dict
    # position_ids [L] long, masks [L, L] bool, text_token_mask [L] bool, for one sentence of n tokens
    # ([CLS] w1..wk [SEP], n = k + 2) truncated/padded to L (L == n inside fuse)

def msda_lowmem(value, spatial_shapes, spatial_shapes_list, level_start_index, sampling_locations,
                attention_weights, im2col_step) -> Tensor
    # same signature and result as transformers' pure-PyTorch MultiScaleDeformableAttention.forward,
    # summing per level instead of stacking

class FusionStage:
    """HoloBrain's frozen feature_enhancer + spatial_enhancer, from the pretrain/ checkpoint."""
    def __init__(self, ckpt_dir: str, device="cuda", dtype: str = "fp32", chunk: int = 32,
                 spatial_enhancer=None, lowmem: bool = True)
        # builds feature_enhancer from model.config.json, loads feature_enhancer.* STRICTLY (277 keys) from the
        # same model.safetensors; spatial_enhancer: reuse the HoloBrainEncoder's (rollout) or build + load it
        # (training / eval); eval(), requires_grad_(False); patches every img_attn block's self_attn.func
        self.n_enhancer_keys: int
    @torch.no_grad()
    def unfused(self, neck) -> tuple[Tensor, Tensor]
        # neck: dict of torch tensors with a leading frame dim F (section 3 fields, any float dtype, any device)
        # -> tokens [F, 1200, 256] fp32, pos [F, 1200, 3] fp32: the spatial enhancer on the raw levels 1, 2,
        #    flattened camera-major exactly as rollout.py:350 stores scene_tokens
    @torch.no_grad()
    def fuse(self, neck, words: Tensor, wmask: Tensor) -> tuple[Tensor, Tensor, Tensor]
        # words [F, 24, 256] (TextStore.lookup output), wmask [F, 24] bool; frame f is fused with sentence f
        # -> tokens [F, 1200, 256] fp32, pos [F, 1200, 3] fp32, w_fused [F, 24, 256] fp32 (zeros past n)
        # groups frames by n = wmask.sum(1), slices words to n, builds text_dict_for_length(n, n), runs the
        # OFFICIAL enhancer forward on [F_chunk, 3, 256, h, w] maps per group in chunks of `chunk`, then the
        # spatial enhancer on fused levels 1, 2 with neck_3d; precision context from dtype; restores order
    def capture_bi_attention(self, layers=(0, 5))  # context manager for E4: records, per call, the word
        # softmax of each stride-64 cell [F, heads, 3 * 20, n] and the cell softmax of each word [F, heads, n, 60]

def encode_obs_with_neck(encoder: HoloBrainEncoder, obs: dict) -> tuple[tuple, dict]
    # ONE backbone/neck pass: returns ((tokens [1,3,400,256], pos_mean [1,3,400,3]) bit-identical to
    # encoder.encode_visual_tokens(imgs, depths, image_wh, projection_mat, out_extrinsic=T_ego2world),
    # neck dict with leading F = 1 (section 3 fields; maps fp32 on the encoder's device))

def neck_to_row(neck: dict) -> dict      # numpy row for the buffer (fp16 maps, fp32 rest), F = 1 squeezed
def rows_to_neck(arrays: dict, device) -> dict   # numpy [F, ...] -> torch dict on device (maps .float())
```

### 4.2 `main/lang/neck_buffer.py`

```python
NECK_FIELDS: dict[str, tuple[tuple, np.dtype]]     # section 3
class NeckBuffer:
    def __init__(self, root: Path, capacity: int | None = None)   # capacity: create/open r+; None: read-only
    @classmethod
    def for_step_buffer(cls, step_buf, root=None) -> "NeckBuffer"  # root default step_buf.root.parent/"neck_buffer";
        # existing: capacity, n, head must match the StepBuffer; fresh: only when step_buf.n == 0
    n: int; head: int; capacity: int; arrays: dict
    def append(self, row: dict) -> None
    def discard_to(self, n: int) -> None      # pre-wrap only (head == n asserted), mirrors ItemSidecarBuffer
    def flush(self) -> None
    def refresh(self) -> None
    def read(self, rows: np.ndarray) -> dict  # {field: array [len(rows), ...]}

class NeckLockstepWriter:
    """Wraps ro.buf (a LockstepItemWriter) in v2 collection: append(entry) pops entry["_neck"] (a neck_to_row
    dict; KeyError if missing), checks shapes, calls inner.append(entry), then neck.append(row + episode_id,
    t_in_ep of the entry). Everything else (end_episode, _rows, ...) is passed through to inner."""
```

### 4.3 `main/lang/data.py` (sampler)

- `LangSource.neck` (lazy, read-only `NeckBuffer`) and `LangSource.has_neck`.
- `LangSampler(..., neck: bool = False, episodes_per_task: int = 0)`.
  - `neck=True` and a source without `neck_buffer/`: `ValueError` naming the root ("version-1 shard").
  - With `neck=True` the batch gains `neck` = {field: np array [U, ...]} over the unique (source, ring row)
    frames of the batch, and `neck_idx` int64 [B, 5] indexing it for frames (t, t-k, 0, t+1, t+1-k). Frame 0
    neck rows are cached per (source, episode) in RAM like `first_frame`.
  - `episodes_per_task = N > 0`: only the first N episodes of each (task, embodiment) in the fixed order
    `sorted(eids, key=lambda e: stable_hash(("e6", task, embodiment, e)))` are drawn (nested in N); the
    shuffled-sentence pool is restricted the same way. `sampler.episode_counts()` reports {(task, emb): n}.

### 4.4 `main/lang/train.py`

- `cfg.scene_fusion`: the trainer builds `FusionStage(CKPT_DIR, dtype=cfg.fusion_dtype, chunk=cfg.fusion_chunk)`
  and the training sampler with `neck=True`; the probe/validation sampler too.
- `grad_step`: after the text lookup, `fuse` the 5 frames of every sample with its sentence (frame 0 once per
  sample, shared by the live and the target stacks), write `scene_tokens` / `scene_pos` ([B,3,1200,*], frames
  t, t-k, 0) and `scene_tokens_next` / `scene_pos_next` (t+1, t+1-k, 0) into a shallow copy of the batch;
  with `cfg.fused_words` the live encoder gets `w_fused(t)` and the target `w_fused(t+1)` (mask unchanged).
  Time it with CUDA events: `perf/fusion_ms`, `perf/step_ms` (per step, averaged over the log window).
- The same fusion in `probe_forward`, `aux_probe`, `run_sanity` (its "switches off => sentence cannot affect
  Q" check also needs fusion off, or it is reported as not applicable).
- `cfg.train_episodes_per_task` reaches the training sampler only (the probe config resets it to 0); the
  run logs `data/samples_per_episode`.
- `--init-weights` from a checkpoint whose `scene_fusion` / `fusion_words` differ: refused.
- `rounds` with `scene_fusion`: refused (ValueError).

### 4.5 `main/lang/network.py` (decision 6)

`LangCriticHead` with `cfg.critic_readout`: "softmin" unchanged; "mlp" as FUSION_PLAN 3.6 (shared RMSNorm
over [r, q_1..q_K, masked body mean], Linear((K+2)·256, critic_mlp_hidden) → SiLU → Linear(·, 1)), forward
returns `(Q, None)`. No other module changes.

### 4.6 Rollout / collection / eval

- `LangRolloutController(..., fusion: FusionStage | None = None, keep_neck: bool = False)`. `_encode_obs`
  runs `encode_obs_with_neck` when either is set (else the base path, unchanged) and keeps `self._neck_now`.
  `_collect_tick` attaches `neck_to_row(self._neck_now)` to `self._pending["_neck"]` when `keep_neck`.
- `LangPolicyShim` with fusion: ignores the tokens it is handed; fuses `ro._neck_now` with the named sentence
  (and, in monitor mode, the counterfactual) into separate histories; with `fused_words` recomputes
  `text_out` from `w_fused(t)` every call; asserts a sentence is set before the first fused push, and
  `set_instruction` after a push raises.
- `collect warmup|validation --lang-version 2 [--replay <v1 shard root>]`: neck buffer created / resumed /
  rolled back / flushed with the pair; episode lines and shard.json carry `lang_version: 2`.
- `eval_offline`: with `cfg.scene_fusion`, per-item fusion in `q_batch`, cache keyed (source, ring row,
  sentence) within a chunk and frame 0 per (source, episode, sentence) across chunks; `--axes` selects the
  sentence axes to score (default: all, as today).

## 5. Tests (CPU unless marked; login node: `nice`, `OMP_NUM_THREADS=4`, no simulator)

- `msda_lowmem` equals the HF function (random inputs, 1e-5).
- `fuse` on a randomly initialised enhancer (built from the real config, small F): grouping by length equals
  per-frame official forward on unpadded words; a frame's result does not depend on its batch mates or on
  the cached padding content; `text_dict_for_length` equals `generate_masks_with_special_tokens_and_transfer_map`
  on tokenizer output (skip if the tokenizer is not cached).
- [gpu] strict load of the 277 keys; `unfused(neck)` vs `encode_visual_tokens` (live fp32, 1e-5) on synthetic
  images; fused vs the BIP3D order run module by module (text encoder → text_feat_map → official enhancer →
  spatial enhancer) on the same inputs.
- `NeckBuffer` / `NeckLockstepWriter`: lockstep with a fake StepBuffer + item sidecar through append,
  rollback, resume, flush; a neck key never reaches the StepBuffer.
- Sampler: `neck_idx` points at the right rows (fake shards encode (row, t, ep) in the neck maps), frame-0
  cache, v1 source with `neck=True` raises, `episodes_per_task` nested and deterministic.
- Trainer with a stub `FusionStage` (deterministic function of (neck, sentence)): live / target receive the
  right frames and words; paired members fuse with their own sentences.
- Critic readout: padding order / count invariance, K = 0 builds, a config without the field loads softmin
  and reproduces an old checkpoint's Q bit for bit.
- Config: legacy `from_dict`, `collect_hash(LangConfig())` unchanged by the new fields.

## 6. Ownership (parallel implementation, one owner per file)

| package | files |
|---|---|
| A readout | `main/lang/network.py`, `tests/lang/test_critic_readout.py` |
| B fusion core | `main/lang/fusion.py`, `tests/lang/test_fusion.py`, `scripts/lang/bench_fusion.py` |
| C neck data | `main/lang/neck_buffer.py`, `main/lang/data.py`, `tests/lang/fake_data.py`, `tests/lang/test_neck_buffer.py`, `tests/lang/test_data_neck.py` |
| D collection | `main/lang/rollout.py`, `main/lang/collect.py`, `main/lang/rounds.py`, `tests/lang/test_collect_v2.py` |
| E training / eval | `main/lang/train.py`, `main/lang/eval_offline.py`, `main/lang/eval_closed.py`, `scripts/lang/grounding_traces.py`, `tests/lang/test_train_fusion.py` |
| F tools / docs | `scripts/lang/check_shards.py`, `scripts/lang/fusion_probe.py`, `scripts/lang/grounding_summary.py`, `scripts/lang/scene_curve.py`, `scripts/lang/grounding_page.py`, `scripts/lang/bridges/*`, `envs/main/item_envs/language/__init__.py` (D1 only) + its test, docs |

Never edit: `main/network`, `main/train`, `main/envs`, anything else under `envs/`, the shared
`/work/nvme/bifq/project/RoboOrchardLab` checkout (every job imports it), and the files live jobs read from
this checkout (`scripts/lang/chain6.sh`, `gpu_stage*.sbatch`, `in_container.sh`, `wq_gpus.sh`,
`main/lang/workqueue.py`).

## 7. As built (integration, 2026-10-09)

Additions to section 4 that the seven packages made, each kept because a caller needs it. Nothing in section 4 was
removed or renamed.

**Fusion core (`fusion.py`).** `FusionStage` is a plain class (no `nn.Module`, so no `.train()` or `state_dict` can
reach it). `dtype` and `chunk` are read at call time. The spatial enhancer runs 8 frames at a time inside a chunk
(its depth-bin features are 79 MB per frame). `fuse` refuses a `wmask` that is not a non-empty prefix mask.
`capture_bi_attention` records one entry per enhancer call, not per `fuse` call, because frames of different
sentence lengths cannot share one tensor: `calls[i] = {fuse_call, frames, n}`, `word[layer][i]`
[g, heads, 60, n], `cell[layer][i]` [g, heads, n, 60] (CPU fp32, camera-major, row-major 4 x 5 per camera), and
`per_frame(layer)` returns them in the fuse call's frame order. `LEVEL_SHAPES`, `TOKEN_LEVELS` and the row dtypes
are defined once, in `neck_buffer.py` (torch-free, read by `check_shards` on the login node), and imported by
`fusion.py`.

**Neck data (`neck_buffer.py`, `data.py`).**
- `NeckBuffer(root, capacity=None, *, level_shapes, n_cam)`: the keywords exist only for tiny test shards.
- `NeckBuffer.validate_row`, `ROW_FIELDS` (the five fields of a `neck_to_row` row).
- `NeckLockstepWriter` takes `t_in_ep` from the inner writer's oldest queued item row, because CrossSafe's entries
  have no `t_in_ep`. It raises if that row belongs to another episode.
- `LangSampler(neck=...)` stores the flag as `.with_neck`. `LangSource.neck_rows(rows, check=True)` and
  `check_neck` compare the neck rows' `episode_id` / `t_in_ep` with the item sidecar's. Every fused batch is checked
  this way.

**Training (`train.py`).**
- `LangTrainer.fused(b, words, wmask, sentences=None, target=True)` makes one pass per unique (neck row, sentence).
- Functions `make_fusion(cfg, device, spatial_enhancer=None)` and `fuse_neck_rows`.
- `SharedNeck`, a workaround for `metrics.Prober`, which builds its sub-batches as `[v[k]] * n`.
- Log keys: `perf/step_ms`, `perf/fusion_ms`, `perf/fusion_frames`, and `data/samples_per_episode` (step × batch /
  episodes drawn).
- `--init-weights` refuses a checkpoint whose effective (scene_fusion, fused words) differ from the run's, and one
  of the other `critic_readout` (a config saved before the field counts as softmin).
- The rounds refusal lives in `rounds.run_rounds`.

**Collection (`collect.py`, `rollout.py`).**
- `collect warmup|validation --lang-version 2 [--replay SHARD]`. With `--replay`, the shard, `--n-shards`,
  `--episodes-per-task`, the masks setting and the task list come from the source. `--tasks` only filters the
  recorded episodes, and `--stream` / `--g-base` are ignored.
- Slots are 4-tuples (task, emb, host, seed): one build each, no new seed on failure. Collection-config drift from
  the source is refused.
- `shard.json` gains `lang_version`, `replay_of` and `replay_slots`. `summary.json` gains `replay_mismatch`; a
  non-empty list exits 1, also on a rerun.
- `check_shards.py` checks a replay shard's lines against its slots and its `replay_mismatch`.
- The controller keeps `_neck_capture` apart from `_neck_now`, so a cache-miss encoding can never reach a row.
- The shim adds `ro.timing["fuse"]`. With `fused_words`, `ro.text_out` / `cf_text_out` are None until the first
  frame and then hold the last frame's text output.

**Evaluation.**
- `eval_offline --axes train` scores only the train axis and skips the base and paraphrase sentences.
  `--fuse-cache N` sets how many fused frames each of its two caches keeps.
- `eval_closed.controller_fusion(cfg, encoder)` builds the stage on the rollout encoder's spatial enhancer.
- The `_meta` of `eval_closed` and the `data` block of `metrics.json` record `scene_fusion`, `fused_words`,
  `fusion_dtype` and `critic_readout`.

**Commands.**
- `--set` values are JSON. Booleans and numbers go bare (`--set scene_fusion=true train_episodes_per_task=8`).
  Strings need JSON quotes: `--set 'critic_readout="softmin"' 'fusion_dtype="bf16"'`. A bare
  `critic_readout=softmin` fails in `json.loads`.
- E6 (V0 settings): `--set 'critic_readout="softmin"' gamma=0.95 gamma_final=0.95 actor_hidden=0
  recall_horizon_ticks=7 train_episodes_per_task=N`.
- E5 run names for `grounding_summary.py` are `<prefix>_<variant>_s<seed>`, with variant v0p, v0ps, v5, v5w or
  v5d2 (no underscore inside the variant). Use `--reference v0p --exempt v5d2:embodiments`.
- The page reads the V5 outputs from `runs/lang/fusion_1009/` by default.
- `check_shards.py --fusion-parity N --parity-tol T`: T = 2e-2 is a placeholder; set it from the first measurement.

**D1 side effect.** After the head-noun fix, 98 sampleable T1 rules (22 in `val_grounding`) lose their bare
head-noun sentences. `eval_offline.axis_sentence` picks `ss[hash % len(ss)]`, so re-scoring an old checkpoint on the
old validation shards with this code picks other sentences for those rules. The report's numbers then do not
reproduce exactly.

**Not covered.**
- `scripts/lang/word_attention.py` builds its own `ev.pe(...)` call from the stored `scene_tokens`, so it would score
  a scene_fusion checkpoint on unfused tokens without an error. It has no owner in section 6; until it calls
  `ev.fused_scene`, do not run it on V5 checkpoints.
- `bench_latency.py` and `profile_step.py` copy `grad_step`. On a scene_fusion config they stop with an error: the
  sampler refuses version-1 shards, and on version-2 shards `encode` refuses a batch that was not fused.
- Nothing has run on a GPU: bf16 and tf32, memory per chunk, speed, the real `encode_obs_with_neck` inside a
  capture, and stored-row parity on real version-2 data.
