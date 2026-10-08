# Response to the code review of 2026-10-07 (at commit 5b09d75)

> Superseded in part by `review_response_round2_2026-10-08.md`: the T1 avoided-object scenario (R2) was a misreading and is reverted, and three conclusions of the re-scored probe below are withdrawn there ("little foresight", "the label is not the cause", "the prior beats the model").

Every item was checked against the code before changing anything; every one held. Below: what was done,
where, the test that covers it, and comments where the fix differs from the suggestion or has limits.
The two decisions (D1, R2) are Yuxuan's and are not discussed here (D1: the rounds job takes `GAMMA`,
default 0.9, and a 0.95 run is planned; R2: implemented in d021e04, its safety check is running).

Commits: deb43e7 (B1), 58af629 (B2-B4, M1-M5, M7-M9, E6, E7, O8), dfab289 (R1, O1-O7, M6, E1-E5),
b8f2404 (F), c63c33f + 847b70d (re-scored offline probe in the report). CPU suites: tests/lang 94 passed,
envs/tests/item_envs 424 passed (`-m "not sim and not gpu"`).

## Data consequences (read first)

- **B1 invalidated today's bead recollection.** The 23 partial `shard_g*_bead` warmup shards collected
  between 17:42 and 19:35 with the 1 mm soft-min were moved to `data/lang_stale/bead_softmin_1007/` (not
  deleted) and the recollection restarted with fresh queue state. In the new shards (11 checked), the
  bead margin is < 0 exactly when n_violating > 0 in all 34,568 bead (row, rule) cells.
- **O1 one-off check:** `scripts/lang/check_shards.py` (rows of every episode line carry its id in both
  buffers, t_in_ep counts 0..n-1, no overlap, header n = last end) passes on all 61 existing shards and
  rings: the race never corrupted data.
- **M4:** the new hash covers everything that shapes a nominal run, so the closed-loop nominal cache
  rebuilds on the next grid (it would anyway after B1).

## A. Bead h

| item | what was done | test |
|---|---|---|
| B1 | removed the four `T: 0.001` entries and the `params.setdefault("T", 0.005)` default; rewrote the explanation in T2_granular_pour.yaml; `T` stays an explicit opt-in | `test_particle_margin_sign_is_the_bead_count` (random bead clouds, inside and outside, 1-2 regions); T2/T4/T6 rule tables have no `T` |
| B2 | `eval_closed.trace_violations`: nominal `violated` / `first_violation_t` from trace.npz with `margins.rule_h` (bead rules by count); `lead_time` tests `is not None` | `test_trace_violations_use_bead_counts`, `test_run_metrics_ticks_and_nominal_duration` (a t = 0.0 first violation counts) |
| B3 | `collect_report` skips bead-task episodes of shards without `n_violating.npy` and prints / reports the count | covered by the same rule as `LangSource` (`test_stale_bead_rows_dropped`) |
| B4 | `resume_point` refuses lines whose rows do not hold that episode (ids at both ends in both buffers, `t_in_ep` 0 and n - 1); README copies only shards whose `summary.json` says complete, with `--sparse` | `test_resume_point_refuses_rows_of_another_episode` |

Comments:
- B1: the docs also had the old temperature wrong (relabeling.md said 5 mm); corrected there and in the
  final report.
- B2: `meta.json`'s margin-based `violations` (envs logger) were left as they are. With B1 they agree
  with h < 0 for every rule (margin < 0 iff h < 0), and evaluation no longer reads them.

## B. Reported numbers

| item | what was done | test |
|---|---|---|
| M1 | round metrics read Q from `trace["V"]` | `test_episode_row_reads_q_and_ticks` |
| M2 | `rollout.single_rule` follows `language.episode_record` (set = several items, or a zone inside/outside rule over several regions); used for `h_other_min` and the panel | `test_spill_rule_counts_as_another_rule` (T6 exclude named, the spill rule enters the minimum) |
| M3 | probes use `dataclasses.replace(cfg, shuffle_text=False, relabel="random")` | `test_probe_sentences_ignore_the_run_text_control` |
| M4 | hash over `envs/main/item_envs/**/*.py`, `**/*.yaml`, `envs/robotwin_tasks/*.py`, `main/envs/*.py` and `main/lang/scene_roles.py`; every row carries `code_hash`; a resume refuses rows under another hash | `test_merged_rows_dedupe_and_hash` |
| M5 | `pick_rules`: one order per (scene, kind) by hash of (seed, kind, rule_id); each axis takes the first eligible | `test_pick_rules_same_rule_on_every_axis` |
| M6 | `eval_offline` writes `data` into metrics.json (tasks scored, stale bead rows dropped, bead h: count / old / none, val roots); `report.py --ckpt` (no mtime), one set of tables per task set | ran on the existing runs: last night's file lands in "task set not recorded" |
| M7 | `intervened_tick_frac` (ticks with action_source == actor) next to `intervention_rate` in round metrics and closed loop | `test_episode_row_reads_q_and_ticks`, `test_run_metrics_ticks_and_nominal_duration` |
| M8 | `para_std` and `rule_std` over the same states, in the training probe and in `eval_offline` | – |
| M9 | `LangTrainer.reset_perf()` after probes and after rounds collection; text misses logged by split (`train` / `probe`) | – |

Comments:
- M4 adds `scene_roles.py` to the suggested globs: the nominal cache stores designations, which R1 (and
  now R2) change.

## C. Designations

| item | what was done | test |
|---|---|---|
| R1 | unclear iff the neighbour's `slot == "far_across"`; records without slot names fall back to a 67.5 deg cut; plan 3.5 row updated | `test_in_way_other_tasks` (an outer container 81 deg away is benign, the diagonal unclear) |

Confirmed in the data before the fix: the first T6 warmup episode had both other containers unclear.

## D. Data integrity, resume, jobs

| item | what was done | test |
|---|---|---|
| O1 | `main/lang/locks.py`: a claim is a directory `<name>.<g>`; taking over a stale one creates g + 1. Workqueue claims use it (a fresh legacy `<id>.lock` still counts). `collect.cmd_shard` and `rounds.run_rounds` hold a `WriterLock` on their data root for the process lifetime (a second process waits out a crashed writer's lock, up to 200 s, then exits 75 / refuses) | `test_claims_have_one_winner`, `test_writer_lock`, `test_workqueue` |
| O2 | `make_ring` reopens StepBuffer and sidecar directly; parity asserted after the roll-back | `test_make_ring_reopens_out_of_lockstep_and_checks_capacity` |
| O3 | reopening refuses when the ring cannot hold the requested rounds | same test |
| O4 | `--resume` reuses `lang_config.json` only when `state.pt` exists; otherwise the flags win (a note lists the differences) | `test_resume_without_state_uses_the_flags` |
| O5 | `LangRolloutController._h_unit_margins` sets the filter's margins from LangConfig in h units after the base class built them | `test_filter_margins_in_h_units` |
| O6 | `rounds.sbatch` relaunches after a crash when `state.pt` advanced (at most 3 per job); the smoke test already treated a timeout as a note (96f37bf) | – |
| O7 | the text store counts loaded extras as extras, and `save_extra` merges entries another process wrote to the file; evaluators read the trainer's extras and write `text_cache_extra_eval.pt` | `test_extra_file_keeps_loaded_and_concurrent_entries` |
| O8 | scenes whose redraws all fail write `missing` rows (reason "no scene"); the merge keeps one row per (embodiment, key), the last written, under the current hash | `test_merged_rows_dedupe_and_hash` |

Comments:
- **O1 differs from the suggested fix.** `os.rename(lock, stale)` then `mkdir` still has a gap: a worker
  that saw the claim as stale can rename away a claim another worker took a moment earlier. A new
  generation directory is never confused with the stale one. `fcntl.flock` was not used: Lustre mounted
  with `localflock` keeps such locks node-local.
- The lock change went live in the running Delta bead collection: claims show as `<id>.lock.0`, every
  shard holds `.writer.0`, no errors.
- The rounds stop/resume test was extended (kill mid-episode, stop between episodes, stop mid-training)
  and still matches an uninterrupted run exactly.

## E. Evaluation methodology

| item | what was done | test |
|---|---|---|
| E1 | per pair: `rank_acc_h` (current h, no model); `rank_acc_mc` (agreement with the discounted HJ return `hj_return` of the recorded h, pairs with \|y1 - y2\| > 0.05); per state Spearman(Q, y) (>= 3 rules); strata by the harmful rule's ticks to its first violation (violated / 1-7 / 8-25) | `test_hj_return_matches_the_closed_form` (V_k = 1 - 2 g^k), `test_summarize_label_checks_strata_and_intervals` |
| E2 | rows keep their sentences; scene-free text prior `TextPrior` (`--prior-data`); T5 pairs split by "tall" / "short" in either sentence; the shuffled axis is labelled a code check in the tables | same tests, `test_evaluator_prior_and_same_state_base` |
| E3 | `auroc_base` / `rank_acc_base` use another sampleable rule's sentence of the same state (another episode's only when none) | `test_evaluator_prior_and_same_state_base` |
| E4 | episode-cluster bootstrap 95% intervals on every rank and AUROC cell, `n_episodes`, task-macro means; AUROC ranks vectorized | `test_summarize_label_checks_strata_and_intervals` |
| E5 | a table of referent vs consequence within T1 / T3 / T7, with intervals | – |
| E6 | `harm_named_nomdur`: harm within the longer nominal run's tick count | `test_run_metrics_ticks_and_nominal_duration` |
| E7 | paired scene bootstrap for `protect_rate` and `selectivity`; `protect_rate_pos` on scenes where a nominal run harmed; pooled selectivity over scenes with both kinds only | `test_aggregate_intervals_and_selectivity_scenes` |

Comments:
- E2: the text prior keys on the consequence plus the referent's attributes (object category, colour,
  size; region kind, colour), not a logistic regression on sentence features. It knows at least as much
  as any one sentence says, so it is a conservative (strong) baseline.
- E4's request for ~30 validation episodes per task is a compute call and was not made.
- E6 / E7: the closed-loop grid was not re-run (GPU hours; the nominal cache has to rebuild after B1 and
  M4 anyway).

**Re-scored offline probe** (official step-20000 checkpoint; T1 / T3 / T5 / T7; 31 episodes; full table
in `runs/lang/lang_full_paired_off_1007/eval_offline/checkpoint_final/tables.md`, report 5b):

| | model | current h, no model | vs HJ return | text prior |
|---|---|---|---|---|
| pooled | 0.62 [0.53, 0.70] | 0.88 | 0.62 | 0.68 |
| T1 | 0.61 | 0.89 | 0.61 | 0.67 |
| T3 | 0.77 | 0.60 | 0.83 | 0.75 |
| T5 | 0.92 | 0.90 | 0.92 | 0.89 |
| T7 | 0.54 [0.36, 0.75] | 0.91 | 0.54 | 0.62 |

- E1's hypothesis does not hold on this checkpoint. Scored against the HJ return, the pooled score is
  unchanged; only T3 moves (up).
- The low T1 / T7 scores come from little foresight: 2079 of the 3160 pairs are already-violated rules,
  ranked 0.67; violations 8-25 ticks ahead are ranked 0.51.
- The scene-free prior beats the model on T1 and T7.
- T5 drops from 0.94 to 0.77 (26 pairs) without a size word.

## F. Report corrections

All done in `final_report.md` (b8f2404):
- the bead soft-min history (B1);
- the old-h note next to every T2 / T4 / T6 number;
- the closed-loop caveats: 47 runs, 25 missing, 1-4 harm runs per cell, T1 only hosts 0 and 1 and no harm
  run, no protection level claimed;
- "1.2% of decisions, about 3.5% of ticks" (M7);
- decision 3 with the near-path evidence;
- the matched-h wording ("fixes the label, not the attribution").

## G. Notes

Not acted on (they were addressed to Yuxuan):
- T6's nominal success of 3-7%: success cost means nothing on T6 until the aim error or the success test
  is recalibrated.
- The attention / ranking mismatch. The re-scored probe is consistent with it: the model attends to the
  named item but does not foresee contact on T1 and T7.
