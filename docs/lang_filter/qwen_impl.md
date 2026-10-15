# V6 implementation contract (vanilla Qwen front end)

Written 2026-10-14 from `QWEN_PLAN.md` and a read-only map of the code (seven readers, every fact below was
checked in the code or on the real checkpoint files). This file is the contract the code follows; where it and
`QWEN_PLAN.md` differ, the reason is given.

## 0. What is being built, and the decisions

**The run** (Yuxuan, 2026-10-14): one V6 round-training run next to the two vanilla ones, "the same round
training": 100 rounds × 4 episodes × 2048 steps, probe every 10 rounds, ur5-wsg only, all 7 tasks, γ 0.95,
softmin critic readout (pinned: both vanilla runs train it), MLP actor (`actor_hidden` 256), `tf32_train`,
seed 0, `filter_episode_frac` 0.8, on the general A40 partition. E7 (the no-training gate) and E8 (short runs)
are not run now: the user asked for the round training first and the verification afterwards.

**Its data.** Qwen replays, with the same seeds and slots, of the ur5 part of `data/lang/warmup` (the main
shards' T1 / T3 / T5 / T7 episodes plus the `_bead` shards' T2 / T4 / T6 episodes: 210 episodes, 30 per task,
exactly the rows the vanilla runs train on) and of `data/lang/val` (35 episodes, masks on, for the probe).
Into `data/lang_qwen/{warmup,val}/<same shard names>`.

**Same layouts as the vanilla runs.** The T1 candidate pool changed on 2026-10-12 (7c2476d), so today's code
draws other T1 objects for the same seeds. `spawn_pool = "pre_1012"` rebuilds the old pool from objects.yaml's
`legacy_ids` / `legacy_category` (verified field by field against `7c2476d^`: identical ids, categories,
mass ranges, colours, pool order). The V6 run uses it for the replays and for its rounds, so its training scenes
are the vanilla runs' scenes (rounds seeds are paired by `--seed 0`). Names come from today's vocabulary, as in
every run of today's code. Replays check every T1 placement against the source line.

**Front end built in `main/lang`, not through HoloBrain's class.** `HoloBrain_Qwen2_5_VL` hard-codes
`flash_attention_2`, loads the full 36-layer 7.5 GB base before cutting it, resolves `vlm_pretrain` relative to
the checkpoint folder and builds the action decoder. Everything V6 needs is in
`HoloBrain_v0.0_Qwen/pretrain/model.safetensors` (vision tower bit-identical to the base, layer 0 lightly tuned,
the final norm, `feat_mapping`, the mixing weights, the depth Swin, the spatial enhancer); the Qwen base folder
supplies only config, tokenizer and processor files.

**Not in this step:** Qwen support in `eval_offline.py`, `eval_closed.py`, `grounding_traces.py`,
`word_attention.py`, `grounding_summary.py` and the E7 script. They refuse a Qwen checkpoint with a clear error
until the verification step ports them. The trainer's own probe (`Prober` through `probe_forward`) is ported
now, because round training runs it every 10 rounds.

## 1. Config (`main/lang/config.py`, done in the commit that adds this file)

| field | values | kind | meaning |
|---|---|---|---|
| `encoder` | `holobrain_gd` (default, and for saved configs) / `holobrain_qwen` | collection | the frozen front end |
| `text_backend` | adds `holobrain_qwen` | training | must equal "qwen" exactly when `encoder` is |
| `qwen_words_prefix` | `frame` (default) / `blank` | training | which prefix the words read |
| `qwen_attn` | `sdpa` (default) / `flash_attention_2` / `eager` | collection (Qwen only) | attention kernel |
| `spawn_pool` | `current` (default) / `pre_1012` | collection | the T1 candidate pool |

Refused: Qwen with `scene_fusion`, Qwen with `mask_special_tokens` (no [CLS] / [SEP] in the Qwen words), a
text backend and encoder that do not match. Properties `cfg.qwen` and `cfg.scene_tokens_per_frame` (297 / 1200).
`collect.collect_fields` drops `encoder` / `spawn_pool` at their neutral values and `qwen_attn` under GD, so
every existing shard keeps its `collect_hash` (checked on the 33 shards that record one).
`qwen_words_prefix` is in `TRAIN_ONLY_FIELDS`.

## 2. The frozen front end: `main/lang/qwen_encoder.py`

```python
QWEN_CKPT = env HOLOBRAIN_QWEN_CKPT or
  /work/nvme/bifq/project/tmp/hf-cache/hub/models--HorizonRobotics--HoloBrain_v0.0_Qwen/snapshots/b82becd9b5f0cd72ef22796d894de292ac2b0333/pretrain
QWEN_BASE = env QWEN_BASE_DIR or
  /work/nvme/bifq/project/tmp/hf-cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

class HoloBrainQwenEncoder:                      # a plain object, never an nn.Module of the trained model
    IMAGE_WH = (308, 252); N_CAM = 3; GRID = (9, 11); N_TOK_CAM = 99; N_TOK = 297
    PREFIX_LEN = 325; ROPE_NEXT = 61; N_KV_HEADS = 2; HEAD_DIM = 128; MAX_WORDS = 24
    def __init__(self, ckpt_dir=QWEN_CKPT, base_dir=QWEN_BASE, device="cuda", attn="sdpa", vision=True)
    ckpt_id: str          # sha256[:16] of the safetensors header bytes + file size: recorded in kv_buffer and checkpoints
    def encode_frame(self, rgb, depth, projection_mat, T_ego2world)
        # rgb uint8 [F,3,252,308,3] RGB (np or torch); depth float32 [F,3,252,308] metres;
        # projection_mat [F,3,4,4] ego frame at 308x252; T_ego2world [F,4,4]
        # -> tokens fp32 [F,3,99,256], pos fp32 [F,3,99,3] world metres, (K, V) bf16 [F,2,325,128] each
    def encode_text(self, kv, sentences, kv_index=None)
        # kv = (K, V) bf16 [U,2,325,128] on the device; sentences list[str] (N); kv_index LongTensor [N] into U
        # (default: arange(N), or zeros when U == 1) -> words fp32 [N,24,256], mask bool [N,24]
    def blank_kv(self)    # (K, V) [1,2,325,128] of the prefix with black images (RGB 0) and zero depth, cached
    def instruction_ids(self, sentence) -> list[int]
```

Build (composed, no RoboOrchardLab class):
- `Qwen2_5_VLConfig.from_pretrained(base_dir)`, `text_config.num_hidden_layers = 1`,
  `Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation=attn, dtype=torch.bfloat16)`; load the
  checkpoint's `vlm.*` tensors with `vlm.` stripped, `strict=False`; assert the only missing key is
  `model.language_model.embed_tokens.weight` and that it is the same Parameter as `lm_head.weight` (tied; the
  checkpoint stores the fine-tuned embedding once, as `vlm.lm_head.weight`); nothing unexpected. eval, no grad.
  With `vision=False` (the trainer's text pass) the vision tower is dropped after loading.
- `feat_mapping`: two bf16 `Linear(2048, 256)` from `feat_mapping.{0,1}.*` and the bf16 `weight` [2]; computed as
  HoloBrain does (`foward_feat_mapping`, structure.py:236-250): `softmax(weight)` in bf16, sum of the two mapped
  hidden states, cast to fp32.
- `backbone_3d` and `spatial_enhancer` built from `model.config.json` with `main.network.build`, strict-loaded
  from the prefixes `backbone_3d.` / `spatial_enhancer.`, fp32 (the FusionStage pattern, fusion.py:288-330).
- `AutoProcessor.from_pretrained(base_dir, use_fast=True)` for pixel values (CLIP normalisation; no resize at
  308x252) and the tokenizer.

The prompt and its prefix (verified with the real tokenizer and `get_rope_index`):
- template `'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' +
  ' '.join(['<|vision_start|><|image_pad|><|vision_end|>'] * 3) + '\nYou are a robot. {s}<|im_end|>\n<|im_start|>assistant\n'`;
- the **prefix ends at the "." of "You are a robot."**: 325 positions after the 3 × 99 pad expansion. The space
  before the sentence belongs to the sentence's first token (" Don"); an empty instruction would instead leave
  a lone " " token at index 325, so never cache the empty-instruction prompt uncropped;
- the tail is `' ' + s + '<|im_end|>\n<|im_start|>assistant\n'`; the instruction tokens are `tail[:-5]`
  (the fixed 5-token template end [151645, 198, 151644, 77091, 198]); 3-15 tokens for every cached sentence;
  more than 24 is an error;
- mrope positions of the prefix from `vlm.model.get_rope_index` once (constant: they depend on token positions,
  not content); the tail's positions are `61 + j` on all three axes, **passed explicitly** (`position_ids
  [3,N,T]`; without them the model would use `cache_position` 325 + j);
- image slots: sequence indices 15-113, 117-215, 219-317 for cameras left wrist, right wrist, head, each a 9 × 11
  raster (row-major), which is the order the 297 scene tokens are stored in.

`encode_frame`: pixel values (bf16) → vision tower → scatter into the pad slots of the prefix embeddings →
`language_model(inputs_embeds, position_ids, use_cache=True, output_hidden_states=True)` on the 325-token prefix
alone (the image slots are causal and never see what follows) → `K, V = past_key_values.layers[0].keys/values`
(after RoPE) → `feat_mapping` of the image slots (the second hidden state is `norm(layer0(x))`: transformers
appends the final-normed state) → [F,3,256,9,11] → `backbone_3d(depth)` → spatial enhancer → tokens [F,3,99,256]
and `depth_prob` → positions with `fusion.token_positions` (cell top-left convention, world metres).

`encode_text`: tail embeddings with the fine-tuned table, right-padded with id 151643; a **fresh**
`DynamicCache(ddp_cache_data=[(K[kv_index], V[kv_index])])` per call (update mutates it);
`language_model(inputs_embeds, position_ids=61 + arange(T), past_key_values=cache, cache_position=arange(325,
325 + T), attention_mask=None, use_cache=True, output_hidden_states=True)`; never call the decoder layer directly
(sdpa with a cache and no mask aligns causality top-left); `feat_mapping` of the two hidden states; keep each
sentence's first L_i rows; zero-pad to 24. Chunked over N.

Tests (`tests/lang/test_qwen_encoder.py`, CPU, a tiny random Qwen2.5-VL with the same layout so nothing is
downloaded): (a) parity: `encode_text` rows equal the instruction rows of one full-prompt pass; (b) causality:
the image-slot states are identical for two sentences; (c) a batch of two sentences of different lengths equals
two batches of one; (d) the prefix ids equal the full prompt's first 325 ids for several sentences (the
trailing-space trap); (e) grid order 297 = 3 × 9 × 11 raster, cameras left, right, head.

GPU recon (`scripts/lang/qwen_recon.py`, A100 or H200, real weights; writes `runs/lang/qwen_recon/recon.json`):
the same five checks on the real model with `sdpa`, and with `flash_attention_2` if it runs; bf16 numbers per
check; time and memory of one `encode_frame` (3 images) and one `encode_text` batch; word norms against the BERT
words (about 113 per token); and, if HoloBrain's own class can be built there, its image and text features on one
frame against ours.

## 3. Observation path: `main/lang/obs.py`

`qwen_obs(env, kin=None) -> dict`, a copy of `Env.get_encoder_obs`'s camera loop (env.py:463-584, which may not
be edited) at 308 × 252: `task._update_render()` first; cameras [left, right, head]; Color → uint8 RGB; depth =
−Position.z × clip(alpha, 0, 1) × (Position.w < 1); cv2 bilinear resize for RGB **and depth** (HoloBrain-Qwen's
Resize, which its weights were trained with; the GD path used nearest for depth); K rows scaled by 308/320 and
252/240; ego projection `K_c @ ext_c @ inv(ext_head)`; `T_ego2world = inv(ext_head)`; `T_base2ego` as env.py;
measured `joint_state` torch fp32 [1,1,n] as env.py. No mean / std. Keys: `qwen` (True, a marker),
`rgb` uint8 [3,252,308,3], `depth` fp32 [3,252,308], `projection_mat` [3,4,4], `image_wh` [[308,252]]×3,
`T_ego2world`, `T_base2ego`, `joint_state`, `joint_cmd_gap`, `camera_names`, and `native` = {`rgb` uint8
[3,240,320,3], `depth` fp32 [3,240,320], `K` [3,3,3], `ext` [3,4,4] world→camera} for the frames sidecar.

## 4. Storage

All three rings of a shard share capacity and row index (`buffer/`, `item_buffer/`, plus the new sidecars);
rows ride in the pending StepBuffer entry and are popped before `StepBuffer.append` (an unknown key there raises
only after other fields were written).

- **`buffer/`**: unchanged code. A Qwen `_encode_obs` returning [1,3,99,256] / [1,3,99,3] makes 297-wide
  `scene_tokens` fp16 / `scene_pos` fp32 rows (the memmaps take the first row's shape; header.json shows the
  FIELDS 1200 until the first flush, so widths are read from the arrays). Tokens are checked finite before the
  fp16 cast.
- **`kv_buffer/`** (`main/lang/kv_buffer.py`, cloned from `neck_buffer.py`): fields `k`, `v` int16 [2,325,128]
  (bf16 bit patterns: numpy has no bf16; `torch.from_numpy(a).view(torch.bfloat16)` on load, never a numeric
  cast), `episode_id` int64, `t_in_ep` int32. Header: `kv_version` 1, `prefix_len` 325, `rope_next` 61,
  `n_kv_heads` 2, `head_dim` 128, `value_dtype` "bfloat16 bits in int16", `key_rope` "applied", `ckpt_id`, `attn`,
  plus capacity / n / head / fields. API as NeckBuffer: `KvBuffer(root, capacity=None, **dims)`,
  `for_step_buffer(step_buf, root=None, **dims)`, `validate_row`, `append`, `discard_to`, `flush`, `refresh`,
  `read(rows)`, `len`. Helpers `kv_to_row(K, V) -> {"k", "v"}` and `rows_to_kv(k, v, device) -> (K, V)`. A reader
  refuses a different `ckpt_id` when one is given. 332,812 B per row.
- **`frames/`** (`main/lang/frames_buffer.py`): native renders, so both front ends can be recomputed exactly
  offline: `rgb` uint8 [3,240,320,3], `depth` fp16 [3,240,320], `K` f64 [3,3,3], `ext` f64 [3,4,4],
  `T_base2ego` f64 [4,4], `episode_id`, `t_in_ep`. Header `frames_version` 1, channel order, camera names,
  native size. On in the replayed warmup / val shards, off in rounds rings (about 260 GB by round 100).
- **`main/lang/sidecar.py`** (the contract first named it `lockstep.py`): `RingSidecar`, NeckBuffer's ring with
  the fields given by a subclass (`KvBuffer` and `FramesBuffer` are thin subclasses; `reopen(root)` reopens r+ with
  the header's own fields), and `SidecarLockstepWriter(inner, {"_kv": kv_buf, "_frames": frames_buf})`, the
  NeckLockstepWriter pattern generalised (pop all keys, align `episode_id` / `t_in_ep` from the queued item row,
  validate every row, `inner.append`, then each sidecar; restore the keys on failure; `end_episode` checks each
  sidecar's (n, head); attribute passthrough).
- **Masks on the Qwen grid** (`main/lang/qwen_masks.py`): `QwenGridMarginHFn(ItemMarginHFn)` overrides
  `_compute_tick`: it runs the parent, then replaces `token_masks` with the same segmentation pictures pooled on
  the 11 × 9 grid (`masks.resize_segmentation_nearest(ids, (308, 252))`, `pool_mask_to_tokens(...,
  image_wh=(308, 252), strides=(28,))`, cameras [left, right, head]); zeros [K+G, 3, 99] when masks are off.
  `use_qwen_grid(ro)` switches a wired controller's h_fn to it. Qwen shards create the item sidecar with
  `n_tok = 99`, so the probe's `ref_mask` is [3, 297] per frame and data.py:644/681 works unchanged.
- **`main/lang/legacy_pool.py`**: `PRE_1012_BANK_HASH = "693f7ef522ed"` (the hash every pre-10-12 shard records:
  `bank_hash()` then covered paraphrases.yaml and splits.yaml only, equal at ae5b02c and 7c2476d^, and objects.yaml
  did not change between them); `install_spawn_pool(name)` wraps
  `main.item_envs.spawn.t1_pool_entries` (looked up by module global at spawn.py:1145) so each entry gets
  `ids = sorted(ids + legacy_ids)` and `category = legacy_category or category`; "current" restores the original;
  idempotent. A test compares the rebuilt pool with `tests/lang/fixtures/objects_pre_1012.yaml` (the file at
  `7c2476d^`).
- **Shard meta**: `shard.json` gains `encoder` (only when not GD), `kv` {prefix_len, rope_next, ckpt_id, attn},
  `frames` (bool), `spawn_pool` (only when not current) and, for replays, `replay_bank_hash`. Lines carry
  `encoder`. `lang_version` stays 1 (version 2 means `neck_buffer/` everywhere).

## 5. Rollout (`main/lang/rollout.py`)

`LangRolloutController(..., qwen=None, keep_kv=False, keep_frames=False)`; with `qwen`:
- `cfg.qwen` required; `fusion`, `keep_neck` refused;
- after `super().__init__`: `env.get_encoder_obs = _Timer(partial(qwen_obs, env), timing, "render")` (the base
  capture and the cache-miss path both render through it);
- `_encode_obs(obs)` (asserts `obs["qwen"]`): `encode_frame` on the bundle, GD `encode_joint_angles(...).squeeze(2)`
  for the robot state, `extractor.extract()`; returns `(tokens[None], pos[None], state, body)`; keeps
  `_kv_now` always and `_kv_capture` / `_frames_capture` only inside a capture;
- `_collect_tick` builds the kv and frames rows before `super()._collect_tick` and stores them in
  `_pending["_kv"]` / `_pending["_frames"]` after it (the V5 neck order);
- `LangPolicyShim`: one frame history (the scene tokens do not depend on the sentence); each call runs
  `encode_text(_kv_now or blank_kv, [sentence] (+ [cf_sentence]))` as one batch, then `pe.encode_text`, and sets
  `text_out` / `cf_text_out`; time in `timing["text"]`;
- `set_instruction` stores the sentences only (no TextStore).

Nominal episodes run no text pass (the shim is called only in filter episodes).

## 6. Collection (`main/lang/collect.py`, `scripts/lang/check_shards.py`)

- `run_episodes(..., qwen=None, kv_buf=None, frames_buf=None)`: `qwen` is required exactly when `cfg.qwen`, and
  `kv_buf` with it; `install_spawn_pool(cfg.spawn_pool)` before the first env; the controller gets
  `qwen` / `keep_kv` / `keep_frames`; after `wire_item_episode`, `use_qwen_grid(ro)` under Qwen and
  `ro.buf = SidecarLockstepWriter(ro.buf, ...)`; the extras join `guard_writes`, the per-episode lockstep check,
  the rollback, `sync_arrays` and the flushes; lines carry `encoder`.
- `make_pair(out, cfg, capacity, *, neck=False)` (item sidecar `n_tok` = `item_n_tok(cfg)`, 99 under Qwen),
  `open_extras(out, buf, *, kv_dims, frames)` -> {"_kv", "_frames"} (the `open_neck` pattern: fresh next to an empty
  StepBuffer, else reopened r+ from its header; refuses kv rows of another checkpoint or attention kernel),
  `resume_point(out, buf, sidecar, *extras)` (None entries ignored).
- CLI for `warmup` / `validation`: `--encoder {holobrain_gd,holobrain_qwen}`, `--frames / --no-frames` (default on
  for Qwen), `--spawn-pool {current,pre_1012}`, `--qwen-attn`. The Qwen encoder is built once per process.
- Replays: `config_drift` ignores `encoder` and `qwen_attn` (like `lang_version`); `bank_hash` and `spawn_pool`
  drift is allowed only when the source's bank hash is `PRE_1012_BANK_HASH` and the replay runs
  `spawn_pool = pre_1012` (the shard records `replay_bank_hash`). **Layout check**: every replayed T1 line's
  placements (item ids, classes, xyz within 1e-5 m, asset and model id) and every T3-T7 line's `layout` /
  `episode_info` must equal the source line; a mismatch is written to `replay_mismatch` (the command then exits 1,
  as today). As built (`replay_layout_mismatch`): per episode id, the item tables of `item_buffer/episodes.jsonl`
  (the last line per id: ids, categories, colours, kinds, roles, half extents, OBB centres within 1e-5, masses within
  1e-9), T1's placements (item id, class, xyz within 1e-5; asset and model id only where the source recorded them,
  which the pre-10-12 lines do not: the item tables carry the draw instead) and the other tasks' `layout` /
  `episode_info` (the source's keys, floats within 1e-5). It runs for every Qwen replay and every legacy replay.
- `check_shards.py`: `kv_buffer/` and `frames/` present exactly as `shard.json` says; equal (capacity, n, head)
  with buffer and item sidecar; per-row (episode_id, t_in_ep) equality; the kv rows' checkpoint; scene width 297
  under Qwen and never 297 under GD (fake test shards are narrower); under Qwen the item `n_cam × n_tok` equals the
  scene width; Qwen lines name the encoder.

## 7. Training (`network.py`, `text.py`, `data.py`, `train.py`)

- `network.py`: `TEXT_DIMS["holobrain_qwen"] = 256`; `in_proj` is the identity for it (the plan's choice; the
  word scale is measured in recon).
- `text.py`: `text_dim("holobrain_qwen") = 256`; no TextStore is built under Qwen (`trainer.text = None`; every
  `self.text` use is guarded). The auto-found `text_cache.pt` next to `--data` / `--val` is ignored under Qwen.
- `data.py`: `LangSource` reads the encoder from `buffer/header.json` (absent = GD) and checks the scene width
  against it; `has_kv` / `kv` (lazy read-only `KvBuffer`, capacity checked) / `kv_rows(rows, check=True)` with the
  lockstep check against the item sidecar. `LangSampler(..., kv=False)`: the batch gains
  `kv = {"k", "v": int16 [U,2,325,128], "episode_id", "t_in_ep"}` and `kv_idx` int64 [B,2] (frame t `row`, frame
  t+1 `next_row`), one entry per unique (source, row), no random draws. Sources with another encoder or width are
  refused with a clear message; with `probe=True` a Qwen source's item sidecar width must equal the scene width.
- `train.py`: under Qwen the trainer builds `HoloBrainQwenEncoder(vision=False)` unless one is passed (`qwen=`,
  injectable for CPU stubs, shared with round collection); `qwen_words(b, sentences=None, target=True)`, modelled
  on `fused()`: one `encode_text` per unique (kv entry, sentence) over columns t and t+1, chunked, under no_grad,
  outside the TF32 block, timed as span `qwen_text`; writes `w_qwen` / `wm_qwen` (+ `_next`) and a `_qwen`
  marker; `blank` uses `blank_kv()` for every row. `encode()` takes `g("w_qwen")`, `g("wm_qwen")` under Qwen, so
  the target encode reads frame t+1's words with the same sentence. `probe_forward` / `aux_probe` use frame t only;
  `run_probe` trims the probe batch's kv to the live column; `run_sanity` shares one text pass. `load_weights`
  refuses a checkpoint whose encoder differs (a GD checkpoint loads strictly into a Qwen model otherwise: no
  parameter depends on the width); checkpoints record `encoder` and the Qwen `ckpt_id`. `n_tok` comes from the
  shard, with the encoder's width as the fallback.
- `eval_offline.py`, `eval_closed.py`: refuse a Qwen config with a clear "not ported yet" error.

## 8. Rounds (`main/lang/rounds.py`)

Under Qwen: the ring is buffer + item sidecar (`n_tok` 99) + `kv_buffer/` (no frames); resume rolls all three back
together; one `HoloBrainQwenEncoder(vision=True)` per process, shared with the trainer; collection passes `qwen` and
`kv_buf`; the GD `HoloBrainEncoder` stays for the robot-state encoder and kinematics.

## 9. Jobs

- Replays on the A40 interactive chain (a workqueue over 30 warmup + 10 val replay commands, 4-5 per A40), into
  `data/lang_qwen/`. Then `check_shards.py` and the layout check.
- The run on the general partition, from a frozen worktree `LanguageConditionHJ_runQ` at the V6 commit:
  `sbatch --export=ALL -A bifq-delta-gpu -p gpuA40x4 --gpus-per-node=1 --mem=64g --time=48:00:00 -J lang-rounds-qwen
  scripts/lang/bridges/rounds.sbatch` with `RUN=.../runs/lang/rounds_ur5_qwen_g095`, `DATA=.../data/lang_qwen/warmup`,
  `VAL=.../data/lang_qwen/val`, `LANG_REPO=.../LanguageConditionHJ_runQ`, `EXTRA_ARGS='--embodiments ur5-wsg'`,
  `SET='encoder="holobrain_qwen" text_backend="holobrain_qwen" critic_readout="softmin" spawn_pool="pre_1012"'`, and
  no `CHAIN_PIPE` / `CHAIN_STAGE` / `GAMMA` in the environment (they would pause the chain). The config is pinned at
  the first state save, so the first launch must be right.

## 10. Order of work and checks

Status (2026-10-14): step 1 done (commits 8fa60a0 to 5a85771; `tests/lang` 340 passed on the login node, the
memory-heavy `test_shim_with_the_real_stage` left out). Steps 2 and 3 are queued from the frozen worktree
`LanguageConditionHJ_qwenT` at 5a85771: `qwen_recon` on the A100 chain (`runs/lang_pipeline_e2`), `qwen_smoke` on
the A40 chain (`runs/lang/qwen_smoke/smoke.sh`).

Recon (A100-40GB, real weights, `runs/lang/qwen_recon/recon.json`), sdpa / flash_attention_2: (a) text pass vs the
full prompt: relative mean difference 0.54-0.71 % / 0.57-0.70 %, max 0.04 / 0.03 (bf16 rounding; word norm about
15.8); (b) image slots for two sentences: identical, and equal to the prefix pass; (c) batch vs single: 0.04-0.27 % /
0.17-0.30 %; (d) the prefix ids equal the processor's for every sentence and for ""; (e) slots and token order as
specified. encode_frame (3 images) 74 / 51 ms, 2.06 GB peak (weights 1.99 GB); encode_text 128 sentences 8.5 /
7.6 ms. Qwen words have norm 15.8 +- 1.5 per token (BERT + text_feat_map: about 113); in_proj stays the identity
as planned. Prefix keys reach |K| 94.5, values 3.5. The blank prefix changes the words by 7 % (relative mean). The
run keeps sdpa.


1. CPU: this contract's unit tests, the whole `tests/lang` suite.
2. GPU recon (A100 / H200, no simulator): `qwen_recon.py`.
3. A40 smoke: one replay shard (T1 and one bead task) with kv and frames; `check_shards.py`; the layout check; then
   a rounds smoke (1 round, few steps, probe at round 0) to exercise filter collection, the text pass per tick,
   training on kv rows and the probe.
4. The full replays, then the general job.
