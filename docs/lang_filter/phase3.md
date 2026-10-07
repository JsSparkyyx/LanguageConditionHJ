# Phase 3: text store, sampler, trainer, probes, offline probe

## Built

| piece | file | notes |
|---|---|---|
| `TextStore` | `main/lang/text.py` | loads `build_text_cache.py` caches (fp16 [N, 24, 256]); misses (heldout_item sentences, deployment sentences) encoded in one batch, added, persisted to `<run>/text_cache_extra.pt` every 1000 entries and at checkpoints; miss fraction by split |
| other backends (Q3) | `main/lang/text.py` | `bert_raw` (HoloBrain BERT without `text_feat_map`, 768-d) and `sentence_generic` (`all-mpnet-base-v2`, downloaded to `/work/nvme/bifq/project/tmp/hf-cache/all-mpnet-base-v2`), each with its own cache (`python -m main.lang.text build --backend ...`). Both ran end to end on the recon data: 6111 sentences cached, 60 training steps (`runs/lang_recon/smoke_bert_raw`, `smoke_generic`) |
| `LangSampler`, `PrefetchSampler` | `main/lang/data.py` | Section 4.4. One `RelabelSampler(<root>/item_buffer, split="train", k_prev=2)` per source for the usable-row index and rule columns; terminal rows excluded; task balance; task-balanced pairs from mixed rows; `paired` / `random` / `fixed`; shuffled-sentence control; frames copied once from the memmap into preallocated batch arrays (paired transitions read once); frame-0 cache per (source, episode); 16 read threads; one prefetch thread (depth 2) whose batches carry the RNG state after them, for exact resume |
| `LangTrainer` | `main/lang/train.py` | Section 6.1 update. Polyak targets of the whole encoder and both critics. Encoder, text-adapter, critic and actor gradient norms. Section 8.1 scalars every 20 steps (`train_log.jsonl` + wandb offline, project `cehj-lang`). Checkpoints with the config, bank hash, backend and switches (atomic). `state.pt`. `--resume`. The stage deadline gives a clean exit 75 |
| probes | `main/lang/metrics.py` | Section 8.2 on 512 fixed validation samples with `ref_mask`; geometric attention weights recomputed from `GeoCrossAttention`'s own projections (a test rebuilds its output from them) |
| offline probe | `main/lang/eval_offline.py` | Section 9.1: five axes, symmetric and mixed pairs, referent vs consequence decisions, AUROC vs an other-episode baseline, precision / recall at 0, `v_le_h`, paraphrase spread; `metrics.json`, `tables.md`, figures, examples |

Tests (CPU): `test_data.py` (ring seam and physical array end, terminal and truncated rows, pairs always
mixed, `fixed` never changes an episode's rule, no held-out sentence in 10k training samples,
`next_prev_row` at t = 0, 1, 2, task balance, recall labels, `ref_mask`), `test_train.py` (one update,
target coverage, checkpoint round trip, switch mismatch refused, exact resume with and without prefetch),
`test_metrics.py`, `test_eval_offline.py` (end to end on fake data).

## Throughput (one grad step, B = 64, fp32)

| GPU | step | TF32 step (information only) | peak memory |
|---|---|---|---|
| A100 40 GB | 1.22 s | 0.87 s | 22.2 GB |
| H200 | **0.46 s** | 0.39 s | 22.3 GB |

Sampling a batch: 0.085 s with the data in the page cache, ~0.5 s cold (login node). 20k steps take
~2.6 h on an H200 and ~6.8 h on an A100; billing is 6 and 2 units per GPU-hour, so the H200 costs about
the same per step and is used for the long run.

## Sanity checks (Section 6.4)

Preview on the 28 recon episodes (`runs/lang_recon/sanity`, 600 steps each; the real gate run uses the
warmup data, below):

| check | result | required |
|---|---|---|
| overfit, 32 pairs, 600 steps | target-rank accuracy **1.000** from step 100; loss_q 3.16 -> 3e-8 | >= 0.95, loss down 10x |
| shuffled sentences, same batch | **0.571** (`sanity2`, scored on 4 fresh other-episode draws per scoring); the first run's 0.690 was scored on the sentences the model had just been updated on, which measures one step of fitting | <= 0.6 |
| all five switches off | `text_sens` = **0.0** exactly | 0 |

Why the shuffled score is not exactly 0.5: the target `y = (1 - g) h + g min(h, Q_targ(s', a' | sentence))`
depends on the sentence through the target critic, and the live and target critics are correlated. So
"sign(Q1 - Q2) = sign(y1 - y2)" can exceed chance with no information about which rule is harmful. The
sanity output now also reports the label-based score P[Q(harmful) < Q(benign)], which is exactly chance
in expectation under shuffling.

Gate run on the warmup data (`runs/lang/sanity_warmup_partial`, 1000 steps each, fixed batch of 32 pairs
from mixed rows of T1, T3, T5, T6, T7; run while collection was at ~110 of 420 warmup episodes, which
changes nothing for a fixed-batch test):

| check | result | required |
|---|---|---|
| overfit | target-rank accuracy **1.000**; loss_q down **98,000x** | >= 0.95, 10x |
| shuffled sentences | target-rank **0.547** (mean of the last 4 scorings; the last single scoring 0.603), label-based **0.506** | <= 0.6 |
| switches off | `text_sens` **0.0** | 0 |

**The gate passes.** No label leaks into the inputs: the label-based score is at chance.

## Offline run (`full`, 20k steps, probes every 2k)

### Preview run (36% of the warmup, frozen), offline probe at step 8000

`runs/lang/lang_full_paired_off_preview`: the 151 warmup episodes present at 02:47 (46,863 rows),
`--freeze-data`, otherwise the `full` defaults; probes every 2k steps. Offline probe
(`eval_offline/checkpoint_step8000/`), validation data only:

rank_acc = P[Q(harmful rule) < Q(benign rule)] over symmetric pairs at the same state (chance 0.5).

| axis | pooled | T1 | T2 | T3 | T4 | T5 | T6 | T7 | referent | consequence |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 0.668 (4421) | 0.66 (2043) | – | 0.56 (154) | – | 0.94 (178) | 0.81 (1261) | 0.42 (785) | 0.569 (1842) | 0.617 (329) |
| heldout_template | 0.649 (4421) | 0.63 (2043) | – | 0.49 (154) | – | 0.88 (178) | 0.79 (1261) | 0.44 (785) | 0.553 (1842) | 0.590 (329) |
| heldout_word | 0.599 (2193) | 0.57 (687) | – | 0.28 (18) | – | 0.78 (178) | 0.86 (525) | 0.42 (785) | 0.515 (934) | 0.646 (226) |
| heldout_item | 0.105 (76) | 0.16 (25) | – | 0.08 (51) | – | – | – | – | 0.261 (23) | 0.333 (3) |
| shuffled | 0.501 (4421) | 0.49 (2043) | – | 0.55 (154) | – | 0.51 (178) | 0.50 (1261) | 0.50 (785) | 0.488 (1842) | 0.541 (329) |

Mixed pairs (only one member axis-eligible): rank_acc - rank_acc_base (r1's sentence from another episode).

| axis | n pairs | rank_acc | rank_acc_base | difference |
|---|---|---|---|---|
| heldout_word | 1724 | 0.722 | 0.577 | +0.145 |
| heldout_item | 810 | 0.606 | 0.588 | +0.019 |

Per (state, rule): AUROC of -Q for fails_soon (base: other-episode sentences), precision / recall of Q < 0 against fails_soon_recall (7 ticks), v_le_h = P[Q <= h + 0.05].

| axis | n | auroc | auroc_base | prec@0 | rec@0 | v_le_h | para_std | para_std_ratio |
|---|---|---|---|---|---|---|---|---|
| train | 13213 | 0.832 | 0.688 | 0.472 | 0.691 | 0.852 | 0.090 | 0.338 |
| heldout_template | 13213 | 0.828 | 0.688 | 0.436 | 0.690 | 0.841 | nan | nan |
| heldout_word | 10385 | 0.836 | 0.695 | 0.448 | 0.736 | 0.871 | nan | nan |
| heldout_item | 1094 | 0.400 | 0.517 | 0.011 | 0.024 | 0.885 | nan | nan |
| shuffled | 13213 | 0.683 | 0.688 | 0.261 | 0.515 | 0.837 | nan | nan |

Per task (train axis):

| task | n | auroc | auroc_base | rank_acc | v_le_h |
|---|---|---|---|---|---|
| selective_avoid | 3268 | 0.612 | 0.516 | 0.661 | 0.895 |
| granular_pour | 873 | 0.974 | 0.887 | nan | 0.942 |
| push_zone | 1039 | 0.684 | 0.563 | 0.558 | 0.845 |
| granular_carry | 514 | 0.873 | 0.730 | nan | 0.932 |
| tray_transport | 967 | 0.864 | 0.668 | 0.938 | 0.827 |
| clean_pour | 3164 | 0.902 | 0.772 | 0.807 | 0.876 |
| gap_place | 3388 | 0.728 | 0.635 | 0.424 | 0.761 |

Reading:
- **Q1**: the sentence matters. On training templates, ranking of the harmful vs benign rule at the
  same state is 0.668 (4,421 pairs). The shuffled control is exactly chance (0.501), and the sentence
  lifts the imminent-harm AUROC from 0.688 (other-episode sentence) to 0.832.
- **Per task**: strong on T5 (0.94) and T6 (0.81), weak on T3 (0.56), **below chance on T7 (0.42)**:
  for the gap-place neighbours (upright vs stay of the same or the other block) the ordering is
  systematically wrong.
- **Q2**: referent decisions (same consequence, different item) 0.569 < consequence decisions (same
  item, different consequence) 0.617. Which item is meant is the harder part.
- **Q4**: held-out templates cost little (0.649), held-out nouns more (0.599; mixed pairs +0.145 over
  their baseline). **Held-out items fail first and badly**: 0.105 on symmetric pairs (inverted, 76
  pairs), AUROC 0.40 vs 0.52 baseline, no gain on mixed pairs (+0.019).

**Caveat on `rank_acc` (for the human).** Pairs are labelled by `fails_soon` (h < 0 within 25 ticks), but
Q is a value. Some benign rules sit permanently at low h: T3's `zone_include` (box stays on the table)
reads h ~0.2 at every tick. A harmful rule can read h = 1.0 until shortly before its violation. For such
pairs a correct critic gives the benign rule the lower Q, and the pair counts as wrong. The inverted
held-out-item score (mostly T3 rules about a held-out box, where `zone_include` is one member) and part of
T7 (stay vs upright of one block) may come from this, not from misreading the sentence. A version
restricted to pairs whose current h are within 0.1 of each other, or the target-rank form used in the
sanity check, would separate the two.

Training probes (step 2000 / 4000 / 6000): `text_sens` 0.81 / 0.77 / 0.69; body-to-scene attention on the
named referent 2.2x / 1.7x / 3.8x its token share (queries 1.9x / 1.5x / 2.6x); paraphrase spread 0.38-0.40
of the rule-to-rule spread.

<!-- OFFLINE_RUN -->
