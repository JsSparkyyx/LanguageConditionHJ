# Response to the code review, round 3 (at commit 65d417a)

Part I of the round-3 file. Every item was checked against the code first; all held. Part II (the grounding
experiments) is reported in `grounding_1008.md` and on the coauthor page.

Commits: 5bfe4ca (C1-C8, B1, B2, B5, B6, the A tables), f248f60 (the grounding validation stream), ae23060 /
e6082ce (Part II code), b5ba5c9 (report 5b), and this file. CPU suite: tests/lang 128 passed (on a compute
node).

Yuxuan's decisions on this round: new validation episodes for T1 and T7 only (B3 = Part II E1); the E4
decision rule as written; the two running gamma runs keep the frozen code they started with (worktree at
65d417a), so the robustness fixes reach new runs and the experiments only.

## Check now (live run)

Both rounds runs (`rounds_ur5_g090_tf32`, `rounds_ur5_g095_tf32`) execute from the frozen worktree. Their
interactive jobs pass absolute `--data` / `--val` / `--run`, and each start logs "rows 64607 from 60 sources"
(the 60 warmup shards, ur5-wsg rows). The general jobs were submitted with absolute RUN / DATA / VAL and
LANG_REPO exported (the chain's jobs inherit it, so `gpu_stage` runs the worktree's code).

## A. Conclusions (report 5b and the round-2 response)

The official checkpoint was re-scored with the new tables (`eval_offline/checkpoint_final_round3/tables.md`:
the prior and model - prior per task x state, referent / consequence per task x state with the prior and
paired differences, T3 by the benign rule's consequence, T5 size words x state, AUROC per task with -h now
and the prior). Report 5b and the round-2 response were corrected with the reviewer's wording.

| item | what was done | numbers that changed the text |
|---|---|---|
| A1 | "T3 and T5 show foresight" replaced: foresight is not established | T5 not yet: model 0.94, prior 0.81, model - prior +0.12 [0.02, 0.25] (with a size word +0.09 [0.00, 0.21]); tall-block rules: AUROC of -Q for harm not yet happening 0.68 vs 0.77 for the current h. T3 not yet by the benign consequence: zone_include 0.34 (= the prior), stay 1.00, upright 0.90, zone_exclude 0.73 |
| A2 | "most misranked pairs are already broken" replaced; the referent / consequence split per stratum tabulated | T1: about 394 of 795 misranked pairs already broken (half); T7 about 277 of 359 (three quarters). T1 referent 0.56 already broken / 0.44 not yet; consequence 0.78 / 0.60 |
| A3 | "the prior does not beat the model" replaced by "indistinguishable pooled and on T1 / T6 / T7, better on T3" | pooled -0.04 [-0.14, 0.07]; T6 carries the old-h caveat |
| A4 | AUROC per task, -h now and the prior on the same rows, pooled without the bead tasks | T1 0.64 / 0.88 / 0.60; T7 0.83 / 0.91 / 0.63; pooled without T2 / T4 / T6 0.75 / 0.89 / 0.58 |
| A5 | `rank_acc_referent_text`, `rank_acc_consequence_text` and the paired differences per group | T1 + T3 + T7 referent: model 0.53 vs prior 0.67 (-0.14 [-0.27, 0.05]); consequence 0.70 vs 0.55 (+0.15 [-0.04, 0.33]); T7 consequence 0.57 = prior 0.57 |

Comment: A1's T3 hypothesis is half right. The current h is no reference on T3's not-yet pairs, but the
zone_include shortcut does not produce the score: the model is at the prior's 0.34 on those pairs, and its
0.68 comes from the other benign consequences.

## B. Metric and reporting defects

| item | what was done | test |
|---|---|---|
| B1 | checkpoints record the training data's bead h (`bead_h_train`) and the run identity; `metrics.json` records it, the scored embodiments, an eval-code hash and the checkpoint sha; `report.py` groups on (tasks, training bead h, embodiments, eval code). Closed loop: rows carry `ckpt_sha` / `ckpt_step`, `_meta` the embodiments and checkpoint, a resume refuses filter rows of another checkpoint file | re-scored official checkpoint: `bead_h_train: not recorded` (trained before the field; it is grouped apart from bead-count runs) |
| B2 | only runs with no tick are left out; a run that threw mid-episode keeps its observed harm and interventions and counts as a task failure; `not_run` per (task, kind) and `exceptions_scored` (closed loop and round metrics) | `test_exception_runs_are_scored_and_the_control_summarized`, `test_round_summary_leaves_out_episodes_that_did_not_run` |
| B3 | DECISION (Yuxuan): T1 / T7 only, = Part II E1 (20 + 20 ur5-wsg episodes, own id stream 91M-100M); the other tasks stay at 5 per task, stated in the tables | `test_streams_disjoint_and_int32` |
| B4 | = Part II E0c: [CLS] / [SEP] apart, entropy per head and query, the critic blocks probed, referent words from `describe` / `describe_region` / `materials`, RMS of w' and of the keys per kind, the [CLS] / [SEP] spread across sentences. Checkpoint config: `use_sub_sentence_represent: true` | `test_word_attention_runs_on_cpu` |
| B5 | `filter_code_hash` also covers `main/train/{collect,config,hfunc}.py` and `main/lang/{config,train}.py`; `eval_closed --control-scenes N` adds a filter-off run through `LangRolloutController` per scene, compared with the standalone runner's nominal runs (`--axes none`: nominal + control only; the control on 3 + 3 T1 / T7 scenes is queued) | `test_exception_runs_are_scored_and_the_control_summarized` |
| B6 | report 180-182 corrected; the round-2 response's T6 / pooled rows carry the old-h note; `--prior-data` with no shards or no sentences refused | – |

## C. Robustness and the launch path

| item | what was done | test |
|---|---|---|
| C1 | `collect.guard_writes`: the writer lock is checked right before every row write of both buffers (`WriterLock.check_fast`: the heartbeat's in-memory flag, and the lock files when the last look is over 2 s old, which is always true after a stall long enough for a takeover); the capture checks every tick in memory. An interrupted episode is collected again under the next seed (`resume_point`, the rounds roll-back), so rows a stale writer might still add carry another episode id | `test_writes_are_checked_and_a_woken_writer_stops`, `test_resume_point_rolls_back_partial_episode` |
| C2 | arrays synced first, the lock checked, then the headers written; the metrics row after the episode line; `LangSource` counts rows only up to the last episode line and refuses a header with fewer rows than the lines (`resume_point` already refused shorter rings) | `test_rows_capped_at_the_last_episode_line` |
| C3 | saves go through `_publish`: a temp name unique to the process, the writer lock checked right before `os.replace`; offline training holds a writer lock (exit 75 on a lost lock) | the suite's save / resume tests |
| C4 | `rounds.sbatch` makes RUN / DATA / VAL absolute and refuses relative ones with LANG_REPO set; an empty `--data` or `--val` glob is refused; `state.pt` records `h_code_v2`, the resolved data roots and `git_head` (+ dirty flag), and a resume refuses another h, other data roots, or another commit unless `--accept-code-change`; the RoboTwin task links are hashed through their resolved targets (C6) | `test_resume_refuses_other_data_or_code`, the sbatch block run with stubs |
| C5 | shards compare `collect_hash` (the config without the training-only fields); for older shards it is recomputed from their `config.json` (all 80 existing shards match today's) | `test_h_code_v2_and_collect_hash` |
| C6 | `h_code_v2` hashes exactly what defines h: margins, rules, geometry, robot_distance, item_shapes, regions, objects.yaml, the collision spheres, the h functions of `runner.py` / `margin_hook.py` by source (parsed, not imported), the shard's task yamls, the resolved RoboTwin task links, ITEM_DISTANCE. The first hash stays for older shards / states, which are compared on it | same, `test_resume_refuses_another_h_definition` |
| C7 | `play_wrapped` marks that play_once started; an episode where it never did is an exception whatever the first capture appended | covered by the simulator runs (the first E1 episode, a failed host grasp after one tick, is recorded as an exception) |
| C8 | videos: per-process partial names, discarded (never published) after a lost lock; `read_lines` rewrites through a temp file; workqueue claims carry owner tokens (beat and release only one's own); sentences are looked up before the TF32 step; `gpu_stage` / `gpu_stage_mn` count rc 124 / 137 as failures; `eval_closed` merges only the evaluated embodiments | `test_tf32_only_inside_the_training_step` (the lookup runs in fp32) |

Comments:
- C1's second fix changes the resume semantics slightly: after a kill mid-episode the slot is collected under
  the next seed, not the same one. Since cuRobo is nondeterministic run to run, the same seed never gave the
  same rows anyway. The rounds stop / resume tests pass unchanged.
- C4: refusing another commit means a deliberate code update of a running run needs `--accept-code-change`.
  The live gamma runs do not have this check (they run the frozen 65d417a).

## D. Notes for Yuxuan

Answered by Part II (`grounding_1008.md`).
