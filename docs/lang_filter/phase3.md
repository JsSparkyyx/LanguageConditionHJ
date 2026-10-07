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

<!-- OFFLINE_RUN -->
