# Response to the code review, round 2 (at commit 8b48be6)

Every item was checked against the code before changing anything; all held. Commits: 7d8c57e (A, revert),
b60008d (B1), 0c5a875 (ungrouped stats, user request), 45787e5 (a bug of round 1's B1, below), aa4341d
(B2), f052cbb (C1-C8, D1 probe), 82956d0 (launch scripts), then the independent-review fixes and this
file. CPU suites at the end: tests/lang and envs/tests/item_envs pass (run on a compute node; the login node
was at load ~3,370).

## A. T1 avoided-object scenario: withdrawn, not fixed

The scenario of d021e04 came from a misreading of the R2 decision: the user never asked for the nominal to
plan around any object. 7d8c57e restores spawn.py, runner.py, the T1 yaml, scene_roles.py (near_path unclear
again), collect.py's scene_meta, plan 3.5 and DESIGN_CONTEXT 2 to their state before d021e04 and removes
scripts/lang/t1_avoid_check.py with its test. The queued check runs were stopped (partial output in
runs/lang_verify/t1_avoid_reverted/). A1-A3 therefore have nothing left to fix. The designations
(harm / benign / unclear) stay as before; at the user's request every grouped statistic now also has an
ungrouped "all runs" version (0c5a875: round metrics, heatmaps, trend, closed-loop tables, collect report).

## B. Locks

| item | what was done | test |
|---|---|---|
| B1 | heartbeat counter in `<name>.<g>/beat`; stale = the same count for a full window on the TAKER's monotonic clock (claims without a beat file keep the mtime rule); `WriterLock.check()` before every flush, episode line, state and checkpoint save, and about once a second inside an episode (`LangRolloutController._capture_tick`); `LockLost` is a BaseException so the base rollout's `except Exception` cannot finish the episode; collect exits 75, rounds returns 75 without writing; the heartbeat reports a vanished claim | `test_claims_have_one_winner`, `test_legacy_claims_by_mtime`, `test_writer_lock_lost_after_a_stall`, `test_heartbeat_reports_a_missing_claim` |
| B2 | rounds.sbatch: a crash with progress near the deadline resubmits; make_ring pairs a missing sidecar; resume_point's comment states what it checks; shard.json records `h_code` (margins, rules, runner, margin_hook, item_buffer, the shard's task yaml), a resume refuses another one, a finished shard is recognised first, backfilled into the 40 new bead shards; `TextStore.save_extra` merges under a short lock; the g -> g + 1 takeover is tested through `workqueue.main` | `test_make_ring_pairs_a_missing_sidecar`, `test_h_code_hash_depends_on_the_tasks`, `test_save_waits_for_another_saver`, `test_workqueue_takes_over_a_dead_claim` |

## C. Evaluation and the re-scored numbers

| item | what was done | test |
|---|---|---|
| C1 | (task x stratum) cells; per task "already broken" / "not yet" is the headline table | `test_summarize_label_checks_strata_and_intervals` |
| C2 | y renamed "the return of the nominal continuation"; `rank_acc_y` = P[y(harmful) < y(benign)] per group; the report's claims rewritten (below) | same |
| C3 | `protect_rate_pos` cross-fitted (select on one nominal run, baseline from the other, both ways). Simulated do-nothing filter at p = 0.3: 0.48 before, -0.06 now | `test_protect_pos_is_cross_fitted` |
| C4 | the prior is a logistic regression on the words of the pair's actual sentences (fit on training-template sentences of warmup rules); paired episode bootstrap of model - prior; prior roots in metrics.json, refused when they overlap `--val` | `test_evaluator_prior_and_same_state_base` |
| C5 | ties count 0.5 for the model too | `test_summarize_ties_censoring_and_rc_pool` |
| C6 | `rc_pool` = T1 + T3 + T7; report.py Q2 per task and pooled over those three; closed-loop summary.json `_meta` (tasks, env and filter code hashes, ckpt), report.py groups on it | same |
| C7 | `filter_code_hash` (env hash + the code a filtered run executes) on every row; a resume refuses filter rows under another one; the merge drops them | `test_merged_rows_drop_filter_rows_of_other_filter_code` |
| C8 | pick_rules names the first rule eligible on every axis when one exists; intervals called optimistic; AUROC over harm not yet happening; the "8-24" stratum; states whose 25-tick window runs past a time-capped episode's end leave the pairs and are counted | `test_pick_rules_prefers_a_rule_eligible_everywhere`, `test_summarize_ties_censoring_and_rc_pool` |

Re-scored offline probe (official step-20000 checkpoint; validation now has the 30 bead-count episodes;
full tables in `runs/lang/lang_full_paired_off_1007/eval_offline/checkpoint_final/tables.md`, report 5b):

| | model | current h | text prior | model - prior | already broken: model / h | not yet: model / h |
|---|---|---|---|---|---|---|
| pooled | 0.66 [0.58, 0.74] | 0.89 | 0.70 | -0.04 [-0.14, 0.07] | | |
| T1 | 0.61 | 0.89 | 0.64 | -0.02 [-0.17, 0.17] | 0.69 / 1.00 | 0.48 / 0.71 |
| T3 | 0.77 | 0.60 | 0.56 | +0.21 [0.12, 0.33] | 0.87 / 1.00 | 0.68 / 0.23 |
| T5 | 0.92 | 0.90 | 0.82 | +0.10 [-0.02, 0.22] | 0.91 / 1.00 | 0.94 / 0.61 |
| T6 | 0.78 | 0.93 | 0.85 | -0.08 [-0.18, 0.03] | 0.80 / 1.00 | 0.64 / 0.53 |
| T7 | 0.54 | 0.91 | 0.68 | -0.14 [-0.36, 0.16] | 0.54 / 1.00 | 0.55 / 0.59 |

Corrections to the round-1 response and the report (the reviewer was right on each):
- "Little foresight on T1 / T7" is withdrawn: the T1 / T7 errors sit mostly in already-broken pairs, which
  need no foresight. The model misreads the named item's present state there.
- "The label is not what keeps the score low" / "E1's hypothesis does not hold" are withdrawn: y ranks 99%
  of the pairs like the label, so agreement with it is not evidence; and y is the nominal's continuation, not
  the actor's.
- "The text prior beats the model on T1 / T7" is withdrawn: with a prior that sees only the sentence, the
  paired differences are not significant there (and favour the model on T3).

## D. Word attention (D1)

`scripts/lang/word_attention.py` on the step-20000 checkpoint (300 states per task,
`eval_offline/checkpoint_final/word_attention.md`):
- first body block: 45% of the word attention on [CLS] / [SEP] (2.1x their share);
- every layer prefers the named item's words 1.1-1.7x; punctuation gets ~0;
- entropy 1.5-1.8 nats against 2.24 for uniform.

Consistent with the unnormalized-keys hypothesis for the first block, not decisive. The RMSNorm run is a model
change and left to Yuxuan.

## A bug of my own found on the way (round 1's B1)

The hard min over beads raised on an empty bead set (numpy min of a zero-size array); the soft-min it
replaced returned +inf. One of the 210 new bead episodes (a T4 validation carry whose beads did not exist yet
at its first tick) died there with no rows, and its line said "ok" because the base rollout catches errors in
the first capture, outside play_once. Fixed: `_bead_min` returns +inf for no beads, an episode without a
captured tick is recorded as an exception, `check_shards.py` lists rowless episodes as a note. All 40 new bead
shards pass; in 146,973 bead cells the margin is < 0 exactly when a bead is out.

## Independent review of the rounds-run code (a fresh-context agent, before the launch)

Nothing silently corrupting on the normal stop / resume paths. Its eight robustness findings, all confirmed,
and what was done:

| # | finding | done |
|---|---|---|
| 1 | an error inside a tick, between the sidecar row and the StepBuffer append, left a queued row; `end_episode` then raised and ended the process | `run_episodes` rolls both rings back to the episode start and records it as an exception |
| 2 | rounds.sbatch ended the chain on the first crash before a launch had saved anything | one retry without progress per job; the Delta chain's retry limit raised |
| 3 | round metrics counted exception episodes as clean filter runs (biasing harm and false alarms down) | `LangRoundMetrics.ran()`: exception episodes left out of the summary, heatmaps and trend, counted in `n_not_run`; the closed-loop aggregate likewise |
| 4 | stale claim directories were never removed (every later launch waited ~3 min); a stalled holder could regain the lock; a beat in flight could survive release | a takeover removes the dead lower claims; claims carry an owner token, `holds()` and the heartbeat check it (generation numbers restart at 0 once every claim is gone: the token, not the number, identifies a claim); release stops and joins the heartbeat |
| 5 | a shard with a rowless episode could not resume | `resume_point` accepts rows [n, n] |
| 6 | the resume rewrote lang_episodes.jsonl in place | temp file + replace |
| 7 | nothing pinned the h definition for the ring | `state.pt` records `h_code` and the git commit; a resume under another h_code refuses; the overnight run executes from a frozen git worktree |
| 8 | wandb drops replayed steps after a hard kill | accepted (wrapped, harmless); the run gets a fresh name |
