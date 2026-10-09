# Grounding experiments, 2026-10-08 (review round 3, Part II; Yuxuan's decisions of the same day)

Which object does the sentence name? Small experiments on T1 (`selective_avoid`) and T7 (`gap_place`),
ur5-wsg. Code at commit 9b5057f / e9f1d98 (frozen checkouts `LanguageConditionHJ_exp`, `_exp2`); outputs under
`runs/lang/grounding_1008/` and `runs/lang/lang_full_paired_off_1007/eval_offline/grounding_e0/`.

Decisions in force (Yuxuan, 2026-10-08): gamma 0.95 for new runs; "unclear" referents count as in the way;
20 ur5-wsg validation episodes per task; the reading fixes in the order a (V1 / V2), b (V4, an auxiliary target
loss), with the oracle tag V3 as the ceiling; decision rule below.

## E1. More validation for T1 and T7

`python -m main.lang.collect validation --stream grounding --tasks selective_avoid,gap_place --embodiment
ur5-wsg --shard i --n-shards 4 --episodes-per-task 20` (i = 0-3; ids 91M-95M, `config.valg_offset`), masks on,
text cache `data/lang/val_grounding/text_cache.pt`. 40 episodes: T7 20 ok, T1 18 ok and 2 exceptions (the host
task `place_container_plate` found no grasp pose and threw on its first tick; recorded as exceptions, no usable
rows). T1 hosts 4 each. `check_shards.py`: 4 shards, 0 failed. All scoring below uses these plus the existing
ur5-wsg T1 / T7 validation episodes: 23 T1 and 25 T7 episodes.

## E0. The existing checkpoint (official step 20,000), no training

Scored on the 23 + 25 ur5-wsg episodes (`eval_offline ... --tasks selective_avoid,gap_place --embodiments
ur5-wsg --seeing-vs-selecting --out-name grounding_e0`).

### E0b. Seeing versus selecting

Per (state, rule) of avoid / stay / upright rules with their training sentence: rho_named = Spearman(Q, the
named rule's h now), rho_any = Spearman(Q, the lowest h now over the scene's rules of the same consequence,
named or not). [ ] = episode-cluster bootstrap 95% interval; "any - named" is the paired difference.

| task | state | rows (episodes) | rho(Q, h named) | rho(Q, h any) | any - named |
|---|---|---|---|---|---|
| T1 | all | 7202 (23) | 0.21 [0.14, 0.28] | 0.33 [0.23, 0.41] | +0.12 [0.02, 0.20] |
| T1 | already harmed | 1889 (23) | 0.18 [0.04, 0.35] | 0.03 [-0.11, 0.19] | -0.15 [-0.34, 0.02] |
| T1 | not yet | 969 (23) | 0.13 [0.04, 0.22] | 0.29 [0.18, 0.42] | +0.17 [0.03, 0.29] |
| T7 | all | 8206 (25) | 0.26 [0.16, 0.35] | 0.33 [0.21, 0.47] | +0.07 [0.02, 0.14] |
| T7 | already harmed | 827 (18) | 0.18 [-0.11, 0.51] | 0.20 [0.00, 0.41] | +0.02 [-0.27, 0.39] |
| T7 | not yet | 282 (18) | 0.11 [-0.02, 0.29] | 0.01 [-0.14, 0.18] | -0.10 [-0.23, 0.05] |

Reading (review round 4, item 4: judged by the state-paired difference any - named and its interval, not by
whether the two rho intervals overlap):
- T1: Q follows the scene's most endangered object more closely than the named one, +0.12 [0.02, 0.20]
  overall, +0.17 [0.03, 0.29] before contact (not yet) and +0.13 [0.04, 0.21] when no harm is near (4344 of
  the 7202 rows); at already-harmed moments -0.15 [-0.34, 0.02], inconclusive.
- T7: +0.07 [0.02, 0.14] overall; inconclusive in each stratum (+0.02 [-0.27, 0.39] already harmed, -0.10
  [-0.23, 0.05] not yet).
- Every correlation is weak (at most 0.33): the model sees the danger only weakly, and ties it to the named
  object even less.

### E0c. Word attention, cleaned up

`scripts/lang/word_attention.py` (300 states per task; `grounding_e0/word_attention.md`):
- **The HoloBrain text encoder uses its sub-sentence mask** (`use_sub_sentence_represent: true` in the
  checkpoint config). Measured on 400 training sentences, the frozen [CLS] feature is the same vector for
  every sentence (cosine to the mean 1.0000, relative spread 0.0000). [SEP] is NOT constant (relative spread
  0.28, cosine 0.994; a first real word 0.71 / 0.87): the sub-sentence mask isolates only the token at column
  0, and [SEP] attends to the whole sentence, so it is a sentence summary that can carry the name (audit
  2026-10-08; RoboOrchardLab bip3d/bert.py:78-90). V2 therefore hides one content-free token and one summary
  token. The adapter's own self-attention mixes the sentence into every position, so the adapted [CLS] (w')
  can still carry sentence content; only its input is constant.
- **Attention per head is sharp, not diffuse:** entropy per (head, query) 0.03-0.11 nats against 1.3-1.8 for
  the head- and query-averaged attention and 2.2 for uniform. Each head reads essentially one word.
- **[CLS] / [SEP] have larger keys** (RMS 11 vs 6 for words; w' RMS 7.0 vs 3.8), and [CLS] gets 24% of encoder
  block 1 (2.2x its share), 17% of the first critic's last block (1.5x) and 48% of the second critic's (4.3x).
  Large keys do not always mean more attention: [SEP] gets exactly its share in encoder block 1 (1.0x), and
  where the scene reads the words [CLS] has its largest key but gets 0.9x its share.
- **The named item's words are preferred where the body first reads words:** encoder block 1 2.1x (T1 2.5x),
  the critics' first blocks 1.4-1.8x; scene-reads-words 1.5x.

## E2. Short runs V0-V3 (6,000 steps, 2 seeds each)

Runs `runs/lang/grounding_1008/g1008_v{0..3}_s{0,1}`, trained at 9b5057f / e9f1d98 (V3 s1 finished on a5b6055,
which differs only in checkpoint export), all with gamma 0.95, the linear actor head (actor_hidden 0),
tf32_train on, 6,000 steps, T1 + T7, ur5-wsg, recall horizon 7; scored on the same 1735 primary pairs (41
episodes with pairs). Full tables and the per-run settings: `runs/lang/grounding_1008/summary.md`.

| primary: referent, already harmed, T1 + T7 | seed mean [95%] | seed 0 / seed 1 |
|---|---|---|
| V0 baseline | 0.59 [0.48, 0.70] | 0.557 / 0.613 |
| V1 words normalized | 0.59 [0.44, 0.75] | 0.575 / 0.612 |
| V2 V1 + [CLS] / [SEP] hidden | 0.59 [0.50, 0.71] | 0.560 / 0.628 |
| V3 oracle tag | 0.58 [0.47, 0.71] | 0.580 / 0.588 |
| sentence-only prior | 0.55 [0.47, 0.64] | |

Neither text fix helps (decision step a fails). V3 is no ceiling: zeroing its learned tag (eval_offline
`--zero-tag`, `grounding_notag`) changes 1.7% of seed 0's primary outcomes (0.580 with the tag, 0.569 without).

## Why: the model memorizes its training scenes (audit 2026-10-08)

The same primary score on the training episodes (T1 / T7 ur5 of `data/lang/warmup`, 30 + 30 episodes; out-name
`grounding_train`, 2498 pairs) and on the new ones:

| model | training scenes | new scenes |
|---|---|---|
| official (20,000 steps, 7 tasks, both robots; 120 T1 / T7 scenes) | 0.93 [0.90, 0.96] | 0.57 [0.45, 0.71] |
| V0 s0 | 0.971 | 0.557 |
| V0 s1 | 0.974 | 0.613 |
| V3 s0 | 0.962 | 0.580 |

- On training scenes the model also ranks the "not yet" pairs above the simulator's present h (V0 s0: T1 0.84
  vs 0.73, T7 0.96 vs 0.61): it has learned which object each training episode will hit, which only seeing
  the episode can tell.
- V0 s0 on new scenes is flat from the first checkpoint: 0.567 (1,000 steps), 0.563 (2,000), 0.561 (3,000),
  0.557 (6,000).
- So the text path works on scenes it has seen (it picks the named object at 0.97), but it does not carry to
  new scenes. The oracle tag changes nothing because the training loss is already near its floor without it.
- Training data: 30 T1 + 30 T7 ur5 episodes (212 + 150 sampleable rules); 6,000 steps x 64 = 384k samples, so
  every episode is seen about 6,400 times.

## V4 (decision step b): the auxiliary target loss

Two seeds from `_exp3` (0e97772), V0's settings pinned, weight 0.0082 from the two-pass calibration (median
aux/ratio over steps 400-600: 0.26 for s0, 0.48 for s1, both counting as tried):

| primary | seed mean [95%] | seed 0 / seed 1 |
|---|---|---|
| V4 | 0.60 [0.50, 0.72] | 0.627 / 0.581 |
| V0 (same seeds) | 0.59 [0.48, 0.70] | 0.557 / 0.613 |

Step b fails too. Where the V4 head points on new scenes (median centre error, frame t): T1 19 cm, T7 6 cm; the
middle of the scene's named objects is 18 cm (T1) and 5 cm (T7) from the named one, the nearest other named
object 13 cm and 10 cm. On new scenes the head is no closer than the middle of the scene, so even a direct
"where is it" signal is learned only for the training scenes (training batches: 6 cm).

V0 s0's validation learning curve, complete: 0.567, 0.563, 0.561, 0.560, 0.542, 0.557 at 1,000-6,000 steps.
V3 s1 with its tag zeroed: 0.588 -> 0.581 (0.9% of outcomes change).

Running (queue `runs/lang_grounding_1008/e2.cmds`, frozen checkout `_exp3` at 0e97772, pinned to V0's
settings): V4 and V4b x 2 seeds, V3 s1 with its tag zeroed, V0 trained on both robots' T1 / T7 (120 scenes,
`g1008_d2_s*`), V0 s0's training-scene score at 1,000 / 3,000 steps, and the official model on val20 (T3 / T5).
V4 weight: two-pass calibration (`calib/v4_weight2` = 0.0082; median aux/ratio over steps 400-600: 0.26 for s0,
0.48 for s1); V4b weight 0.0039 (att ratio 1.1-1.2: stronger than the 0.2 aimed at; at weight 1 the attention
loss pushed the critic loss from about 0.04 to 0.3).
