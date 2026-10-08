# Vanilla language-conditioned filter: overnight implementation report (2026-10-06/07)

Status per phase, what was built, every deviation from `VANILLA_PLAN.md` and why, and the decisions
left for the human. Detailed reports: `recon.md` (Phase 0), `collect_report_warmup.md` /
`collect_report_val.md` (Phase 1), `phase2.md` (model), `phase3.md` (sampler, trainer, sanity, offline
run), `phase4.md` (rounds), `phase5.md` (closed loop).

## 0. Status at the end of the night

| phase | state | evidence |
|---|---|---|
| 0 recon | **done**, no STOP condition | `recon.md`: lockstep parity exact (28 episodes), text cache equal in fp16 precision, filtered controller ok, in-the-way designations within bounds, masks ok, CrossSafe and env Phase-5 regressions pass; 2 env single-episode sim tests fail (env-side) |
| 1 collection | **done** 06:03: warmup 420 episodes (30 per task and embodiment, 6 per T1 host; 419 ok, 1 env exception; 82 GB), validation 70 (13.5 GB); all labels within the 2-80% band; delay audit flags every task (gamma decision) | `collect_report_warmup.md`, `collect_report_val.md` |
| 2 model | **done** | `phase2.md`: 19 model tests, 20.7 M parameters, 22 GB at B = 64 |
| 3 trainer, probes, offline probe | **sanity gate passes** on warmup data (overfit 1.000, shuffled 0.547 / label 0.506, switches off 0.0). Official 20k-step run on the full warmup (`runs/lang/lang_full_paired_off_1007`) is running on the A100x4 chain (~2.5k steps per hour; moved there at 08:40 when the A100x8 queue stalled; done ~15:30); a preview run on a frozen 36% of the warmup is further along, and its offline probe at step 8000 is in `phase3.md` | `phase3.md` |
| 4 rollout + rounds | **checks pass**: 14/14 filter episodes in lockstep with videos, deployment frame 0 == collection s0 (diff 0.0), one capture before physics in both modes, forced timeout / exception statuses; rounds smoke resumed correctly after a job kill (crash path); the exit-75 path is being exercised by `runs/lang_phase4/rounds_exit75` | `phase4.md` |
| 5 closed loop | small grid (2 scenes per combo, axes train + heldout_template) running on the preview checkpoint at step 8000, 8 children over A40 nodes, resuming across jobs; partial tables in `phase5.md` | `phase5.md` |
| 6 matrix + report | `scripts/lang/matrix.sh` prints the runs; nothing beyond `full` was launched | this file |


## 1. What was built

| file | content |
|---|---|
| `main/lang/__init__.py` | `main.__path__` bootstrap for `main.item_envs` |
| `main/lang/config.py` | `LangConfig(FrozenConfig)` (Section 11 defaults; the four per-episode env overrides declared), canonical index tuples, the Section 4.2 seed / id helpers with bound asserts, bank hash |
| `main/lang/scene_roles.py` | `referent`, `in_way` (T1 placements + second-corridor rule, T3 elements, T5 tall block, T6 hit container + beads + unclear diagonal, T7 hit flanker), `designation`, `ref_rows`, `mask_region_rows` |
| `main/lang/collect.py` | the one episode loop (`run_episodes`, Section 7.1) for warmup, validation, recon and rounds; shard schedule (Section 7.2); warmup / validation / recon CLIs with episode-boundary resume; per-episode scene metadata |
| `main/lang/rollout.py` | `LangRolloutController` (status, s0 capture outside collect mode, timing, rule trace, Q_cf), `LangPolicyShim`, `set_instruction`, `wire_item_eval` |
| `main/lang/network.py` | the model of Section 5 (`phase2.md`) |
| `main/lang/text.py` | `TextStore`, the `bert_raw` / `sentence_generic` backends and their cache builder |
| `main/lang/data.py` | `LangSampler` (Section 4.4), `PrefetchSampler` |
| `main/lang/train.py` | `LangTrainer` (Section 6.1), offline mode, checkpoints, `--resume`, sanity checks (Section 6.4), CLI |
| `main/lang/rounds.py` | rounds mode (Sections 6.2, 7.3, 7.4) |
| `main/lang/metrics.py` | Section 8.1 scalars, Section 8.2 probes (attention recomputed from `GeoCrossAttention`'s own projections), `LangRoundMetrics` |
| `main/lang/video.py` | `LangPanelVideoWriter` (Section 8.4) |
| `main/lang/eval_offline.py` | offline language probe (Section 9.1) |
| `main/lang/eval_closed.py` | closed-loop grid (Section 9.2), sharded parallel children |
| `main/lang/report.py` | one table per research question |
| `main/lang/recon.py`, `phase4_checks.py`, `collect_report.py` | Phase 0 checks, Phase 4 simulator checks, the Phase 1 data report incl. the delay audit |
| `main/lang/workqueue.py` | shared work queue for chained jobs of any size (atomic claims, heartbeats, stale takeover, slot weights) |
| `scripts/lang/` | `in_container.sh`; chains `chain.sh` -> `chain5.sh` (chained interactive jobs; `chain5.sh` with multi-node stages and configurable job length is the current one); `gpu_stage.sbatch`, `gpu_stage_mn.sbatch`, `par_gpu.sh`, `par_cmds.sh`; `stage.sbatch`, `long.sbatch`; `matrix.sh` (prints only); `run_rounds.sh`; `crosssafe_regression.py`; `profile_step.py` |
| `tests/lang/` | 74 CPU tests (seed scheme, scene roles on records of all 7 tasks, schedule, shard resume, model, sampler, trainer resume, metrics, offline probe, closed-loop aggregates, shim history) |

## 2. Deviations from the plan, and why

1. **Compute path.** The batch partition (`gpuA40x4`) was estimated to start in two weeks, so everything
   runs as chained 1-hour interactive jobs (one per partition at a time). Interactive jobs may take a
   whole 4-GPU node, so collection runs 5 shards per A40. Training-only stages run on the A100
   interactive partition, which has its own one-job limit. A100s have no ray-tracing cores, so all
   rendering stays on A40.
2. **Resumable shards.** A shard (14 episodes, ~70 min) does not fit one hour. Shards resume at episode
   boundaries: the interrupted episode's rows are rolled back on both rings (`discard_to`) and the slot
   is collected again with the same seed (same scene). The plan's rule "a crashed shard is deleted and
   collected again" exists to avoid id collisions and overwritten records. Neither can happen this way,
   because the ids and the records are the same scene's.
3. **No per-tick segmentation masks in warmup** (`--no-masks`): ~30% of the per-tick cost; `ref_mask`
   only comes from validation data (probes, offline probe), which keeps the masks.
4. **Check 4 tolerance**: read as relative 1e-3 (fp16 storage of features up to |150|; `recon.md`).
5. **Closed-loop nominal runs** use the env's standalone runner without rendering (same controller
   and physics, no model reads the cameras), ~5x faster than a rendered controller run.
6. **The CrossSafe 20-step regression** uses a recon StepBuffer (no CrossSafe dataset on Delta).
7. **`eval_closed`** shards each embodiment's work list and runs children in parallel; the full grid is
   ~1120 filtered episodes per checkpoint (~90 h of simulation), which one child per embodiment
   cannot finish.

8. **A shared work queue instead of per-GPU command lists** (`main/lang/workqueue.py`,
   `gpu_stage_mn.sbatch`). Free A40 GPUs came as single GPUs on nodes whose memory was mostly taken,
   so a job now asks for `--nodes=1-4 --gpus-per-node=1` with 24 GB per node and starts at once with
   whatever exists (2-3 nodes per job). Each node's worker claims commands atomically.
9. **Training on A100 (fp32), not H200.** The H200 queue filled up (estimated start: the next
   afternoon). A100 runs the step in 1.22-1.27 s, so 20k steps take ~7 h across 1-hour jobs.
10. **A preview offline run on a frozen 36% of the warmup** (`runs/lang/lang_full_paired_off_preview`,
    `--freeze-data`: the 151 episodes, 46,863 rows, present at 02:47). It gives curves and probes by
    morning. The plan's Phase-3 run on the full warmup starts when collection ends, as
    `runs/lang/lang_full_paired_off_1007`.
11. **Sanity check details**: the shuffled control is scored on fresh sentence draws and judged on the
    mean of its last 4 scorings. A label-based score (Q(harmful) < Q(benign), exactly chance under
    shuffling) is reported next to it, because the target-rank score depends on the sentence through
    the target critic.
12. **Collection is not success-only, and episodes with an env exception are kept** with
    `status: exception` (as the plan says). The `place_container_plate` grasp failure produces 1-row
    episodes that never enter training (terminal rows are excluded).

<!-- DEVIATIONS_MORE -->

## 3. Decisions reserved for the human

Each item names the evidence and a recommendation; nothing below was decided for you.

1. **gamma (0.9 now).** Delay audit (170 nominal traces): the median time from the start of the risky
   motion to the first violation is 0.6 s (T7) to 3.4 s (T3), against the ~0.4 s (10-tick) horizon of
   gamma = 0.9. The T3 numbers are inflated: its risky segment is the whole push. T5 tipping (0.80 s)
   and T7 scraping (0.60-0.64 s) are the clean cases. Recommendation: try 0.95 (horizon ~20 ticks) as
   one extra offline run against 0.9; with 0.9 the critic can only warn ~7 ticks ahead.
2. **Warmup size (30 episodes per task and embodiment = 420, collected).** Harmful labels are 5.9% (T7)
   to 55% (T4) of sampleable rule-rows, mixed rows 20-58% in the multi-rule tasks
   (`collect_report_warmup.md`). T7 is the thinnest: its scraped block tips in only 33% of runs (stay
   72%). More data is ~1 episode per minute per A40 GPU.
3. **T1 near-path candidates.** Designated *unclear* by the plan (never named), but violated in 59-79% of
   nominal runs (78-100% on the two stacking hosts, which rarely have an on-path candidate). Promoting
   `near_path` to harm-designated would give T1's stacking hosts harm runs; keeping it leaves them
   benign-only. Review R2 (2026-10-07) adds: nominal runs break the unclear (near-path) T1 rules as often
   as the on-path ones (avoid 0.85 vs 0.83, stay 0.62 vs 0.67, `collect_report_warmup.md`), the warmup
   has 24 harm vs 97 unclear T1 rules, and the closed-loop grid had no T1 harm run at all. The reviewer
   recommends promoting them.
4. **Text feature scale.** HoloBrain word features have RMS 9 (outliers to 150) and pass the identity
   `in_proj` unnormalized into keys / values (`phase2.md`). Training is stable so far; a fixed scale or an
   RMSNorm on `w'` would be a one-line change, to try only if probes show the text attention saturating.
5. **Batch size / precision.** Unchanged (B = 64, fp32). TF32 would cut the step 15% on H200, 29% on
   A100: small enough to leave as is.
6. **Evaluation scene counts.** The full grid (20 scenes, 2 axes) is ~90 h of simulation per checkpoint
   (~4.5 h on a 4-GPU node). Recommendation: 10 scenes per combo on `train` + `heldout_template` for the
   first comparison.
7. **Matrix runs to launch.** `bash scripts/lang/matrix.sh` prints all of them; nothing beyond `full` was
   launched. An offline run of 20k steps is ~8 h on A100 (~16 units) or ~2.6 h on H200 when that queue
   is open (~16 units).
8. **Env-side issues (envs/ is read-only for this work).** `place_container_plate` sometimes fails its
   grasp (`target_pose cannot be None`, RoboTwin `_base_task.py:1212`): 4 of ~15 recon runs, 1 of 12 in
   the warmup (kept as `status: exception`). Two env sim tests fail on single episodes
   (`test_full_episode_low_risk[ur5-wsg]` T2, `test_zero_aim_error_keeps_neighbours_clean[ur5-wsg]` T6).
   T5's benign-designated (short) blocks tip in 17% of warmup runs.
9. **The offline rank metric.** `rank_acc` labels pairs by `fails_soon`, but some benign rules have a
   permanently low h (T3 `zone_include` ~0.2), so a correct critic can rank them "wrong"
   (`phase3.md`, caveat). Recommendation: also report pairs with matched current h, before reading the
   held-out-item and T7 numbers as failures.


## 5. Results so far (08:00)

- **Data** (`collect_report_warmup.md`): 420 warmup + 70 validation episodes, balanced, no episode
  at the cap; harmful labels 6-55% of sampleable rule-rows; benign-designated rules violated in at
  most 17% of runs (T5).
- **Sanity gate** (Section 6.4): passes on warmup data.
- **Offline language probe, preview run at step 8000** (`phase3.md`):
  - rank accuracy 0.668 on training templates, shuffled control 0.501;
  - imminent-harm AUROC 0.832 vs 0.688 with other-episode sentences;
  - held-out templates 0.649, held-out nouns 0.599, held-out items 0.105 (inverted, partly a metric
    artifact, see the caveat);
  - referent decisions 0.569 < consequence decisions 0.617;
  - T7 below chance (0.42).
- **Training probes**: attention on the named referent rises to ~4x its token share in both runs, with
  no supervision on grounding. The text-sensitivity and paraphrase-spread probes look healthy.
- **Closed loop, preview at step 8000, partial** (`phase5.md`): protection 0.10 on the train axis and
  0.30 on held-out templates, from 47 filtered runs of a partial grid (25 missing). Each task cell has 1-4
  harm runs; with 2 scenes per combo T1 covers hosts 0 and 1 only and has no harm run at all (see
  decision 3). That is too few runs to claim a protection level (no intervals were computed then; the
  grid now reports scene-bootstrap intervals). Benign runs intervene on 1.2% of filter DECISIONS, which is
  about 3.5% of ticks (one engaged decision holds 3 ticks: 3r / (1 + 2r)); 20% of them intervene at
  least once. Interventions cost the task. (Corrected 2026-10-07, review F / M7.)
- **Still running (resumable chains)**:
  - the official 20k-step run on the full warmup (A100x4 chain, step 4642 at 08:40, ~2.5k steps per hour, done ~15:30), followed by
    its offline probe;
  - the preview run, stopped near step 17.5k to free the A100 for the official run; its offline probe on the step-16000 checkpoint follows the official run;
  - the closed-loop grid children, the rounds smoke (rounds 0-3 done once, finishing) and the exit-75
    rounds check (A40).

  `runs/lang_pipeline*/chain.log` show progress; `bash scripts/lang/matrix.sh` lists what to launch next.

## 5b. Afternoon update (2026-10-07, 16:00)

Everything in the "still running" list above finished; no job was left in the queue.

**Official run, step 20000** (`runs/lang/lang_full_paired_off_1007/eval_offline/checkpoint_final/tables.md`),
next to the preview run (36% of the warmup) at step 16000:

| measure | official, step 20000 | preview, step 16000 |
|---|---|---|
| rank accuracy, training templates | 0.650 | 0.678 |
| held-out templates / nouns / items | 0.617 / 0.647 / 0.408 | 0.663 / 0.660 / 0.316 |
| shuffled-sentence control | 0.501 | 0.497 |
| **referent** decisions (same consequence, other item) | **0.507** | 0.593 |
| **consequence** decisions (same item, other consequence) | **0.696** | 0.629 |
| imminent-harm AUROC (base: other-episode sentences) | 0.856 (0.742) | – |
| V <= h + 0.05 | 0.925 | – |

Per task (training templates): T5 0.92, T3 0.77, T6 0.72, T1 0.61, T7 0.54.

All T2 / T4 / T6 numbers in this section (including T6's 0.72 and its referent 0.25 below) were computed
with the OLD bead h (before the bead-count h). The offline probe now leaves those rows out until the bead
data is recollected, so current tables have no T2 / T4 / T6 columns.

The pooled referent score (0.507) is a mix that differs by task (pairs: same consequence, other item):

| task | referent pairs | official | preview |
|---|---|---|---|
| T1 selective_avoid (avoid / stay, other object) | 884 | 0.51 | 0.64 |
| T6 clean_pour (keep beads out of the other container) | 386 | **0.25** (reversed) | 0.43 |
| T7 gap_place (other neighbour) | 362 | 0.53 | 0.51 |
| T5 tray_transport (other block) | 178 | 0.92 | 0.92 |
| T3 push_zone (other mat / object) | 32 | 0.88 | 0.59 |

So: at chance on T1 and T7, reversed on T6, good on T5 and T3. Two cautions before reading this as
"cannot ground the referent":
- the rank metric caveat above (benign rules with a low but safe h). It can make T6 look reversed: a
  container next to the stream reads low h without ever being hit;
- T5 has a shortcut: the block at risk is always the tall one, and sentences may say "the tall block".

What is at stake is the label, not the attribution: both members of a pair share the state and the action,
so a Q difference always comes from the sentence. The question is whether `fails_soon` (h < 0 within 25
ticks) is what a gamma = 0.9 critic should rank by. A violation k ticks ahead is worth about 1 - 2 * 0.9^k
(0.30 at k = 10), above a benign rule that sits at a low but safe h, so a correct critic can lose such a
pair (review E1). The offline probe now reports, on the same pairs, the ranking by current h (no model),
the agreement with the discounted HJ return of the recorded h, a scene-free text prior, strata by time to
the violation, and episode-bootstrap intervals (validation has 10 episodes per task, so the official vs
preview swings above, T1 0.51 vs 0.64 and T3 0.88 vs 0.59, fit episode-level noise).

**Re-scored (round-2 review, 2026-10-08; official step-20000 checkpoint, which was trained on the OLD bead h;
validation now includes the 30 new bead-count episodes, so T6 is back; 41 episodes for the pair cells; [ ] =
95% episode bootstrap, optimistic with 7-10 episodes per task; `eval_offline/checkpoint_final/tables.md`;
earlier versions in `checkpoint_final_pre_review/` and `checkpoint_final_round1/`):**

| training templates | model | current h, no model | text prior (the pair's sentences only) | model - prior |
|---|---|---|---|---|
| pooled | 0.66 [0.58, 0.74] | 0.89 | 0.70 [0.61, 0.78] | -0.04 [-0.14, 0.07] |
| T1 selective_avoid | 0.61 [0.53, 0.69] | 0.89 | 0.64 | -0.02 [-0.17, 0.17] |
| T3 push_zone | 0.77 [0.63, 0.89] | 0.60 | 0.56 | +0.21 [0.12, 0.33] |
| T5 tray_transport | 0.92 [0.84, 0.99] | 0.90 | 0.82 | +0.10 [-0.02, 0.22] |
| T6 clean_pour | 0.78 [0.66, 0.90] | 0.93 | 0.85 | -0.08 [-0.18, 0.03] |
| T7 gap_place | 0.54 [0.36, 0.75] | 0.91 | 0.68 | -0.14 [-0.36, 0.16] |

The harmful rule already broken at that tick, or not yet (model / current h):

| task | already broken | not yet broken |
|---|---|---|
| T1 | 0.69 / 1.00 (1272 pairs) | 0.48 / 0.71 (771) |
| T3 | 0.87 / 1.00 (75) | 0.68 / 0.23 (79) |
| T5 | 0.91 / 1.00 (130) | 0.94 / 0.61 (48) |
| T6 | 0.80 / 1.00 (898) | 0.64 / 0.53 (147) |
| T7 | 0.54 / 1.00 (602) | 0.55 / 0.59 (183) |

What this does and does not show (corrected after review round 2, C1 / C2 / C4):
- **T1 and T7: the model misreads the named item's PRESENT state.** Where the harmful rule is already broken,
  the current h ranks every pair right by construction, and the model gets 0.69 (T1) and 0.54 (T7). Most
  of the misranked pairs are of this kind. (The earlier "little foresight" reading was wrong: those pairs
  need no foresight.)
- **T3 and T5 show foresight:** on not-yet-broken pairs the model beats the current h (T3 0.68 vs 0.23, T5
  0.94 vs 0.61).
- **The label check is uninformative here:** the return of the NOMINAL continuation ranks 99% of the pairs
  the same way as `fails_soon`, so agreeing with it says nothing new; and the critic learns the value of its
  ACTOR's continuation, not the nominal's, so neither the earlier "the label is not the cause" nor any
  foresight claim follows from it. Foresight proper is judged in closed loop.
- **The text prior does not beat the model:** keyed on the words of the actual sentences (it used to key on
  the record's category / colour / size, more than a sentence may say), the paired difference is not
  significant on T1 or T7 and favours the model on T3. (The earlier "the prior beats the model on T1 / T7"
  is withdrawn.)
- Referent vs consequence over T1 + T3 + T7 only: 0.53 [0.45, 0.61] vs 0.70 [0.54, 0.83].
- AUROC of -Q for harm within 25 ticks: 0.86 [0.83, 0.90]; for harm not yet happening only: 0.69 [0.64, 0.74].
- T5 is partly a shortcut: 0.94 with "tall" / "short" in a sentence, 0.77 [0.44, 0.93] (26 pairs) without.
- Word attention (review D1, `word_attention.md`): the first body block puts 45% of its word attention on
  [CLS] / [SEP] (2.1x their share); every layer prefers the named item's words 1.1-1.7x; entropy 1.5-1.8
  nats vs 2.24 uniform. Consistent with the unnormalized-keys hypothesis for that block, not decisive. A
  final RMSNorm on the adapter outputs is a model change, left to the human.

**Bead rules: h now counts beads** (user request, `margins.particle_h`). Once n >= 1 beads break a bead
rule, h = -(0.5 + 0.5 (min(n, 20) - 1) / 19): one bead -0.5, 2 beads -0.53, 10 beads -0.74, 20 or more
-1. Before, one bead 1 cm out already read -1, the same as a whole spill. The safe side is depth of the
worst bead / unit, clipped to [0, 1], over the hard min of the beads. (Correction, review B1: the first
bead-count commit 58b048b still took a 1 mm soft-min, because every bead rule set T: 0.001 in its task yaml;
it read up to 0.46 low on the safe side and could put the margin below 0 with no bead out. Since commit
deb43e7 the yaml sets no T: margin < 0 exactly when a bead is out. The bead shards collected
in between were moved to `data/lang_stale/bead_softmin_1007/`; the recollection finished on 2026-10-07,
180 warmup + 30 validation episodes, every bead cell consistent.) Per-tick counts are stored (`n_violating` in the sidecar, `violation_counts` in
`trace.npz`), so this kind of change can be relabeled offline next time.

Consequence for the data: the warmup and validation shards hold the old bead h and no counts, so their
T2 / T4 / T6 rows cannot be relabeled. `LangSampler` and the offline probe now skip the bead-task rows of
any shard without `item_buffer/n_violating.npy` (`LangConfig.drop_stale_bead_rows`, default on); the
other four tasks' rows are unchanged and stay in use. The bead tasks need recollecting (bead-only shards
`shard_g<g>_bead`, about 180 warmup + 30 validation episodes), then a retrain.

## 4. How to run (all inside `scripts/lang/in_container.sh`, cwd = repo root)

On Bridges-2 or any other Slurm cluster without Delta's container: `scripts/lang/bridges/README.md`
(copy list, setup check, bead data, smoke test, the self-resubmitting rounds job, monitoring).

```
# collection (resumable shards; any number of 1-hour jobs)
python -m main.lang.collect warmup --out data/lang/warmup/shard_g<g> --embodiment ur5-wsg --shard <i> --n-shards 15 --no-masks
python -m main.lang.collect validation --out data/lang/val/shard_g<g> --embodiment ur5-wsg --shard <i> --n-shards 5
python -m main.lang.collect_report data/lang/warmup --out docs/lang_filter/collect_report_warmup.md
# text caches
cd envs && python scripts/item_envs/build_text_cache.py --episodes ../data/lang/warmup/shard_g*/item_buffer/episodes --out ../data/lang/warmup/text_cache.pt --device cuda
# sanity gate, offline training, offline probe
python -m main.lang.train sanity --data data/lang/warmup --run runs/lang/sanity_full
python -m main.lang.train offline --data data/lang/warmup --val data/lang/val --run runs/lang/<name> --grad-steps 20000 --probe-every 2000 --resume
python -m main.lang.eval_offline --run runs/lang/<name> --ckpt checkpoint_final.pt --val data/lang/val
# rounds, closed loop, tables
bash scripts/lang/run_rounds.sh --data data/lang/warmup --val data/lang/val --run runs/lang/<name> --collect-rounds 100
python -m main.lang.eval_closed --run runs/lang/<name> --ckpt checkpoint_final.pt --val data/lang/val --scenes-per-combo 20 --n-shards 5 --parallel 10 --gpus 4
python -m main.lang.report --runs runs/lang/* --out docs/lang_filter/results/
bash scripts/lang/matrix.sh            # prints the experiment matrix; --submit queues it
```

Job orchestration used tonight: `scripts/lang/chain4.sh` (A40, multi-node work queue over
`runs/lang_pipeline/work.cmds`) and `scripts/lang/chain2.sh` / `chain3.sh` (A100, one stage per line of
`runs/lang_pipeline_a100*/queue.txt`). Each chain's log is `runs/lang_pipeline*/chain.log`; a file named
`state/hold` pauses a chain.
