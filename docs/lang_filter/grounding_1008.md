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

Reading: Q follows the scene's most endangered object somewhat better than the named one on T1 (paired
+0.12 [0.02, 0.20], from the not-yet moments, +0.17 [0.03, 0.29]) and slightly on T7 (+0.07). But every
correlation is weak (at most 0.33), and the two rho intervals overlap, so the reviewer's reading rule
("non-overlapping intervals") is not met: the model sees the danger only weakly, and ties it to the named
object even less.

### E0c. Word attention, cleaned up

`scripts/lang/word_attention.py` (300 states per task; `grounding_e0/word_attention.md`):
- **The HoloBrain text encoder uses its sub-sentence mask** (`use_sub_sentence_represent: true` in the
  checkpoint config). Measured on 400 training sentences, the frozen [CLS] feature is the same vector for
  every sentence (cosine to the mean 1.0000, relative spread 0.0000); [SEP] nearly so (0.994); a first real
  word 0.87. The adapter's own self-attention mixes the sentence into every position, so the adapted [CLS]
  (w') can still carry sentence content; only its input is constant.
- **Attention per head is sharp, not diffuse:** entropy per (head, query) 0.03-0.11 nats against 1.3-1.8 for
  the head- and query-averaged attention and 2.2 for uniform. Each head reads essentially one word.
- **[CLS] / [SEP] draw attention through larger keys:** their keys have RMS 11 vs 6 for words, w' RMS 7.0 vs
  3.8. [CLS] gets 24% of encoder block 1 (2.2x its share) and 48% of the second critic's last block (4.3x).
- **The named item's words are preferred where the body first reads words:** encoder block 1 2.1x (T1 2.5x),
  the critics' first blocks 1.4-1.8x; scene-reads-words 1.5x.

(E0a, E2, V4, E3 and the decision follow.)
