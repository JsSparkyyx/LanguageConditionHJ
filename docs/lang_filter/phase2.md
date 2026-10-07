# Phase 2: the vanilla model (`main/lang/network.py`)

Implemented as specified in VANILLA_PLAN Section 5, reusing `RobotInjection`, `GeoCrossAttention` and
`TokenActor` unchanged (imported, never copied or edited); `action_effects` is CriticHead's step 1 copied
with a parity test.

| part | what it does |
|---|---|
| `TextAdapter` | `[r, q_1..q_4, words + position]` through 2 custom pre-norm blocks (RMSNorm, MHA `batch_first=True`, FFN 256-1024-256 SiLU, no dropout); returns `r0`, the 4 queries and the adapted words `w'`. `in_proj` is the identity for `holobrain_bert`, `Linear(768, 256)` for `bert_raw` / `sentence_generic` |
| `SceneSet` | CrossSafe's scene adapter (shared over frames) + a learned time embedding per frame role; asserts `[B, 3, 1200, 256]` (frames first, cameras flattened within a frame) |
| `SceneReadsWords` | `s = s + MHA(RMSNorm(s), w', w')`, once (switch `lang_scene`) |
| `LangTrunkBlock` | a. body reads scene (CrossSafe geometry), b. body reads words (`lang_body`), c. queries read scene (`lang_queries`), d. self-attention over [body, r, queries], e. FiLM from `r0` (`lang_film`, zero-init) and separate FFNs for body / queries / r (`update_r=False` in the last critic block) |
| `LangPolicyEncoder` -> `LangEncoded` | injection, scene set, text adapter (or a precomputed `text_out` broadcast over the batch at deployment), scene-reads-words, 3 trunk blocks; `r` = `r0` or a learned `r_static` (`lang_rtoken` off) |
| `LangCriticHead`, `LangTwinCritic` | action injection `Linear(8, 256)`, post-action geometry, 2 blocks, value heads on 22 body tokens and the queries, `Q = -T logsumexp(-V / T)` with T = 0.05 in h units (not scaled by `h_scale`); `forward -> (q1, q2, v1, v2)` and `qmin`, so `SafetyFilter` is reused unchanged |
| probe mode | `set_probe(module, True)`: blocks keep their attention weights and the geometric attention's inputs (`metrics.geo_attention_weights` recomputes them from `q_proj`, `k_proj`, `bias_mlp`) |

## Tests (`tests/lang/test_network.py`, 19 passed on CPU)

- shapes at B = 2 for both embodiments' padding (UR5 14 tokens, Franka 16);
- padding: changing padded body tokens, padded words, or masked scene tokens leaves Q unchanged (1e-5);
- batch permutation permutes every output (Q, V, body, scene, queries, r0);
- all five path switches off: Q identical for two sentences; defaults: Q differs;
- each switch off alone: its parameters do not exist (and with `lang_scene` off, the scene is identical for
  two sentences); each switch on alone (the others off, FiLM not zero-initialized): the sentence reaches Q;
- FiLM zero-init is the identity at initialization;
- the critic loss gives non-zero gradients to every parameter of the encoder and both critics; the frozen
  inputs (scene / state tokens, words) carry no gradient;
- `action_effects` matches CrossSafe's `CriticHead` (effects and post-action positions, < 1e-6);
- softmin of n equal values v is v - T log n; padded body values (1e4) lose the min;
- `bert_raw` / `sentence_generic` adapters project 768 -> 256; deployment `text_out` broadcast equals per-sample words.

## Size

| group | parameters |
|---|---|
| encoder / injection (CrossSafe `RobotInjection`) | 202,752 |
| encoder / scene set (time embedding) | 768 |
| encoder / text adapter | 1,585,920 |
| encoder / scene-reads-words | 263,424 |
| encoder / trunk (3 blocks of 2.77 M) | 8,307,960 |
| critics (2 heads, 5.15 M each) | 10,294,436 |
| actor (`TokenActor`) | 514 |
| total trainable | 20.66 M (CrossSafe: 7.75 M) |

Per trunk block: geometric attention 270 k, three MHAs 263 k each, FiLM 132 k, three FFNs 526 k each.
Switches off: `lang_queries` removes 4.6 M (query attention, query FFN, query value heads), `lang_body`
1.8 M, `lang_film` 0.9 M, `lang_scene` 0.26 M; `lang_rtoken` off adds the 256-parameter `r_static`.

## Memory and speed (one training step, B = 64, fp32, the full Section 6.1 update)

Measured in `runs/lang_recon/train_smoke` (300 steps on the 28 recon episodes) on an A100 40 GB:
**peak 22.1 GB** (fits an A40 48 GB without activation checkpointing), **0.75 steps/s**, data wait 0.0
(GPU-bound). That is 7.4 h for 20k steps on one A100. The batch upload widened fp16 scene tokens to
fp32 on the CPU; it now casts on the GPU (identical values). `scripts/lang/profile_step.py` breaks the
step down by part. B and precision are unchanged: both are the human's decision. A TF32 timing is
measured for information only.

## Note for the human: scale of the HoloBrain word features

The cached `text_feat_map` outputs have RMS 9 per token, with outlier dimensions of RMS 76 and values up
to 150 (`recon.md`, check 4). The plan's identity `in_proj` passes them into the adapter's residual
stream, and the adapted words then serve as keys and values in body-reads-words and scene-reads-words
without normalization, while every other attention input is RMS-normalized (~1). The position embedding
(N(0, 0.02)) is ~450x smaller than the words. The overfit check passes regardless (target-rank accuracy
1.00 on recon data), so this was left as the plan specifies. A fixed input scale or a final RMSNorm on
`w'` would be a small, testable change if the long runs show unstable text attention.

## Regression

CrossSafe `main/train/train.py`: 20 grad steps run (`scripts/lang/crosssafe_regression.py`). The env Phase-5
regression ran in the first A40 collection job (`recon.md`, check 1). No CrossSafe file was edited.
