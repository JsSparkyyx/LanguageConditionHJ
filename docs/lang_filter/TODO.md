# Work left to do (started 2026-10-22)

Keep this list current: tick items off with the date, add new ones at the end of their section.

## A. Revision round (Yuxuan guides it step by step)

The three round jobs of 2026-10-20 (23067248 vanilla, 23067249 Qwen, 23067250 T1-only, all gamma 0.95) keep
running until Yuxuan says to cancel and resubmit. Nothing of this round is committed until the revisions are
checked.

- [x] 10-22 Drop T4 granular_carry and T6 clean_pour from training, collection and evaluation
  (`main/lang/config.py` `TRAIN_TASKS`; `CANON_TASKS` keeps seven for the stored ids and the seeds).
- [x] 10-22 T1: 2 or 3 extra objects (was 3-5). T5: always two blocks (no middle block).
- [ ] 10-22 Hand-picked object corpus (in progress):
  - [x] colours: train red, green, blue; held out purple, orange;
  - [x] T1 pool: wine glass, milk box, apple, bottle, coloured box, coloured tall cylinder (book and kettle removed
    10-22: never fit on the path); 3 training models + 1 held-out model each (apple 1 + 1); category drawn uniformly;
  - [x] held-out models never in training scenes; spawn pool "heldout" (`cfg.spawn_pool`) builds the OOD scenes;
    rules naming them are in the heldout_item split;
  - [x] T3: only the coloured box is pushed, only coloured cylinders stand; T5 / T7 blocks in the 3 colours;
  - [x] T1 on-path size limit 3.5 cm -> 6 cm radius (wine glasses 1 / 2, bottles);
  - [x] 10-22 the three models wider than 6 cm shrunk (wine glass 0 x0.87, milk box 1 x0.88, milk box 2 x0.90;
    `objects.yaml` scale_by_id, spawn `_create_actor_scaled`); checked in sim: real half-width 5.5 cm, bottom on the
    table, on / near the path in about half of 22 placements;
  - [x] 10-22 tests pass (41 lang, 231 env non-sim); fixed on the way: the per-category draw and T3 giving two
    mats the same colour once 3 colours run out;
  - [x] 10-22 renders of the new models, corpus page republished (apple 0 and notebook 1 render untextured in the
    matplotlib viewer: check them in a simulator frame);
  - [x] 10-22 Yuxuan checked the words: fine, except the block's held-out word is now "cube" (was "rectangular
    block", dropped).
- [x] 10-22 T1 spawn sweep (5 hosts x 20 seeds, ur5-wsg; runs/lang/corpus_render/t1_sweep*): on-path size limit
  0.035 / 0.06 / 0.16 m left 9 / 8 / 14 of 100 layouts empty (mostly stack_bowls_two, stack_blocks_two); books and
  kettles never fit on or near the path at any limit. Set to 0.06 (wine glasses and bottles can stand on the path).
  Held-out pool: 10 / 10 scenes built, every candidate rule in heldout_item.
- [x] 10-22 Yuxuan: remove books and kettles; skip empty T1 layouts (they have no rules: collect retries with a new
  seed, eval_closed redraws).
  Sweep after the removal (100 layouts): 6 empty (5 stack_bowls_two, 1 stack_blocks_two), each with 0 rules;
  every placed layout has >= 5 rules. On or near the path: cylinder 54, box 40, bottle 33, apple 28, milk box 14,
  wine glass 13.
- [ ] OOD scenes with the held-out colours for T3 / T5 / T7 (T1 has the "heldout" pool; the other tasks draw their
  colours in their task files and have no switch yet).
- [x] 10-22 revision round checked (trial collection runs/lang/smoke_1022: 5 tasks + 2 OOD T1 scenes, records,
  sentences, splits; 361 lang + 231 env tests) and committed as f1d2488; pushed 10-22 with the new token (with 4c6a158).
- [ ] 10-22 Yuxuan: submit the new runs WITHOUT cancelling the 10-20 ones. Running:
  - frozen worktree LanguageConditionHJ_r1022 at 22b5b3a (superseded on 10-23: see A2);
  - (10-23) RoboTwin task links: the shared RoboTwin/envs/<task>.py links point at the 10-20 runs' own copies
    (LanguageConditionHJ_r1020) again, so their resume check (h_code_v2 hashes the linked files: c6e909dda672 /
    9301411cc7c5, checked) passes and their scenes are their own; only bowl_pour.py links to main (they never load
    it). Newer code no longer uses the links: runner.use_own_robotwin_tasks puts the checkout's envs/robotwin_tasks
    first on RoboTwin's envs.__path__, and h_code_hash_v2 hashes those files. (From 10-22 18:00 to 10-23 13:45 the
    links pointed at main / r1022, so the 10-20 runs' round collections in that window used the 10-22 T3 / T5 / T7
    scenes, and a restart would have been refused.)
  - chain stages d1022_vanilla / d1022_qwen in runs/lang_pipeline_vf/queue.txt (orchestrator pid 2997937 on
    dt-login01): data/lang_1022/{warmup (15 shards, 150 eps), val (25), val20 (T2/T3/T5, 60), val_grounding (T1/T7,
    40)} + text caches + check_shards -> READY; then data/lang_qwen_1022 (Qwen replays of the same shards,
    flash_attention_2) -> READY. Commands runs/lang_1022_data/*.cmds, stage scripts there;
  - login watcher runs/lang/rounds_1022/submit_when_ready.sh (log watcher.log) submits rounds_ur5_g095_1022,
    rounds_t1_ur5_g095_1022, then rounds_ur5_qwen_g095_1022 (runs/lang/rounds_1022/launch.sh) when READY;
  - the chain holds the one gpuA40x4-interactive slot: my own sim tests wait or use another partition.
- [ ] (old item) re-collect the warmup and validation data of T1, T3, T5, T7
  (old shards hold the old objects, colours and layouts; T2 is unchanged); cancel the three round jobs and
  resubmit (`runs/lang/rounds_1020/launch.sh` as the template, new run names).

## A2. 2026-10-23 changes (Yuxuan), before the collection resumes

The 10-22 collection (chain stages d1022_*) is ON HOLD (runs/lang_pipeline_vf/state/hold.d1022_*): its data has T2
and the old tilt limits. The watcher process had died (no run was submitted).
- [x] Tilt limits 45 deg in T3 / T5 / T7: upright theta_max and stay theta_tol (tilt counted as movement).
- [x] T2 replaced by a redesigned T6 in TRAIN_TASKS (T1, T3, T5, T6, T7).
- [x] New T6 in envs/robotwin_tasks/bowl_pour.py (key clean_pour; old clean_pour.py untouched for the 10-20 runs):
  three bowls 002_bowl 5 white / 3 blue / 7 green (base 1 at first: black inside, Yuxuan swapped it the same day),
  colours shuffled, target at T2's target spot, others at the old T6 far / across spots; careless aims at one of the two (50/50); named bowl one of the two (50/50); only the
  bowl rules are named (the spill rule recorded, never named); safe mode = aim 0, low, slow; colors.fixed: [white].
- [x] Calibrate T6 (runs/lang/t6_1023, ur5-wsg; record in envs/docs/item_envs/task_cards/T6.md): the first check's
  "splash into the other bowl" was beads lying on the table against the bowl (fixed: Region.z_floor); the plain
  bowls keep every bead that lands in them, the safe pour's losses were misses (T2's sideways aim lead, fit to its
  deep bowl, put the stream 3.4 cm past the centre). Chosen: gap 9-12 cm, ur5 roll_aim_side 0.075.
- [x] Yuxuan 10-23: base 1 is black inside -> base 5 is the white bowl (taller: rim 6.9 cm, inner radius 6.0 cm).
  Re-check on base 5 (24 + 24 runs): a 13-20 cm careless aim landed in the gap too often (8/14 named-aimed runs
  harmed) -> aim error 16-22 cm: 11/14 harmed, 0/10 benign broken, no plan failures; safe 21/24 success, named bowl
  1/24. Sim tests on A40: T6 8/8, T5 8/8, T7 8/8, phase-5 regression 2/2, nominal mode 5/5, non-sim 75/75;
  test_push_zone_sim was stale since 10-20 (it asked keep-on-the-table for phrases): fixed, 42/42; phase-5
  dryrun 6/6 (the A100 failures were cuRobo's GPU arch).
- [x] T3 / T5 / T7 tilt check under 45 deg (check_tilt.py, 6 careless runs each): every knocked item ends at 90 deg
  or more, standing ones stay under 10 deg; the cards say so.
- [x] Independent review of the diff (findings 0-16, fixed).
- [x] Docs: task cards, tasks_overview, relabeling, VANILLA_PLAN, safe_controllers, DESIGN_CONTEXT; corpus page v8
  (T6 card and scene, examples re-made from today's scenes: template_review/examples_v3.json by
  corpus_page/page_examples.py).
- [x] Lang suite 361/361 (A40, 10-23).
- [x] Committed 7c77532 (pushed), frozen worktree LanguageConditionHJ_r1023, trial collection runs/lang/smoke_1023
  passed (5 warmup + 3 val20 episodes, text caches, check_shards).
- [x] 10-25 Yuxuan's go: the three 10-20 runs cancelled (they had hit the 48 h limit on 10-24 and sat in the queue;
  stopped at round 39 vanilla / 40 T1 / 83 Qwen, run dirs kept; the Qwen filter had drifted to driving 97 % of the
  ticks, every episode timing out); data/lang_1022 + data/lang_qwen_1022 and the core.* dumps deleted.
- [x] 10-26 Re-collection done: chain stages d1023_vanilla -> d1023_qwen (10-25 20:08 -> 10-26 00:13),
  data/lang_1023/READY, data/lang_qwen_1023/READY; checked (150 warmup episodes, every 10-23 change in the data).
- [x] (running since 10-27 ~16:40; 10-29 11:00: vanilla and T1-only round 34 / 100, Qwen round 76 / 100; their
  48 h jobs end 10-29 ~16:40-17:00 and resubmit themselves) The three runs were submitted EARLY (Yuxuan 10-25: the batch queue waits days), so they queue while the data
  is collected: rounds_ur5_g095_1023 (job 23205090), rounds_t1_ur5_g095_1023 (23205091), rounds_ur5_qwen_g095_1023
  (23205092), via runs/lang/rounds_1023/presubmit.sh + wait_then_rounds.sbatch: a job that starts before its READY
  file waits up to 6 h, else resubmits itself. The READY watcher (submit_when_ready.sh) was stopped (no double
  submission). Scheduler estimate on 10-25: start ~10-31.
- [ ] 10-26 Yuxuan: a fourth run, collision avoidance (T1) only on the Qwen front end, like the GD T1 run:
  rounds_t1_ur5_qwen_g095_1023 (job 23229109, `runs/lang/rounds_1023/launch.sh t1qwen`, data/lang_qwen_1023).
  Submitted, pending (estimate 10-30 13:21).
- [x] 10-30 Yuxuan: every round training STOPPED (the results are bad; issues found, the model is analyzed and
  adjusted later). Cancelled while pending: the 10-23 resubmits 23323331 / 23323630 / 23323656 (vanilla, T1-only,
  Qwen), 23229109 (T1 Qwen), 23230275 (sim-h reference), 23319949 (8-bit Qwen VLM failure function). Do not
  resubmit any round run until Yuxuan says so; the run dirs and checkpoints under runs/lang/rounds_* stay.
- [x] 10-30 Yuxuan: confirm the ACTOR issue before changing the model (object identification set aside). The four
  checks of the analysis pasted 10-30; code scripts/lang/diag_1030/, results runs/lang/diag_1030/<check>/: (1) actor
  statistics on T5 states (|mu| / dtheta_max, sigma); (2) T5 ground truth: branch mid-slide states (physics snapshot)
  into planner / freeze / half speed / smooth stop over 1 s, min h of each (A40 production); (3) the critic's ranking of
  those branches' first actions; (4) critic-only retrains with the target through the deterministic actor or a fixed
  continuation (stored next action, zero) vs the stochastic control. Lead from the logs: in harmed T5 episodes the
  filter first engaged only 0.2-0.4 s before h < 0 (gamma 0.95 at 25 Hz looks about 0.54 s ahead).
  10-30 18:20 checks 1-3 done, check 4 interim at ~2k steps: docs/lang_filter/DIAG_ACTOR_2026-10-30.md (commit 4423eb2).
  Actor at the max-entropy point (sigma 0.864, mu 0.018 everywhere = a freeze); Q ranks nothing (spread ~1e-5 vs a
  true 0.49); every target variant loses the action within 250 steps (not the target); on T5 only slowing down
  saves late, freezing tips the block by itself.
  10-31 08:10 check 4 final (20k steps, rescored on the T5 branch states): verdict by the rule fixed in advance "not
  the target; data and the one-tick time step". No variant near a clear spread (largest 5.5e-4 vs 0.02; det / data /
  zero within 1.6x of the stoch control, stoch_norm 2.1x), the T5 sign test decides 0 of 311 pairs for every final
  checkpoint, the actor stays at zero (|tanh mu| <= 0.024). Check 4 cannot separate the data from the softmin readout
  (shared by all variants; the min sits on a query head in 100 % of states). Report section 5; next-step test proposed
  in its section 7 point 6 (branch-tool counterfactual episodes, retrain softmin vs MLP readout). Feeders and watchers
  stopped; the fine-tune resumed (C).
- [ ] 10-31 STORAGE (decide by ~11-04 00:15): /work/nvme/bifq is over the SSD-pool soft quota (2,196 GiB used at 07:20
  vs 2,000 GiB soft; the hard limit of 2,200 GiB was hit on 10-31 05:20-07:15 and saves failed). Check: `lfs quota -p
  22016 --pool ddn_ssd /work/nvme` (`quota` does not show it). After the grace ends every write to /work/nvme/bifq
  fails until usage is below 2,000 GiB. Already done (no data deleted): new files of runs/lang/diag_1030, the
  fine-tune dirs (vla_finetuning/{ws,results_items,models,logs}, runs/lang_pipeline_ft) and project/tmp/logs go to
  the HDD pool (`lfs setstripe -p ddn_hdd`). Proposed, waiting for Yuxuan: `lfs migrate -p ddn_hdd` (same paths,
  nothing deleted, reversible) the rings of the four round runs stopped before 10-25: rounds_ur5_g098 (147 GB),
  rounds_ur5_g095_mlp (143), rounds_ur5_qwen_g095 (93), rounds_ur5_qwen_g095_1020 (95) = 478 GB. Not
  rounds_ur5_g095_1023 (check 4 read its ring).

## B. Evaluation of the new round trainings

- [ ] Distance probe: port branch probe-1013 to main with the new sentence bank, plus Qwen support (Qwen needs
  replays of data/lang_probe with frames / kv).
- [ ] `eval_closed` for Qwen (it still refuses Qwen).
- [ ] Report success, collision rate (CR: the named rule broken; T1 a touch) and safe success (success with the
  named rule never broken) for vanilla gamma 0.95, Qwen gamma 0.95 and T1-only gamma 0.95.

## C. VLA fine-tuned on safe demonstrations (collision avoidance)

- [x] Cancelled 10-29 (Yuxuan: "no need"): job 23051382 never ran; the 500 demos stay in vla_finetuning/data/demos_safe.

## D. VLMs for later use (`VLM/`)

- [x] 10-22 Qwen2.5-VL-32B-Instruct downloaded (64 GB). [ ] RoboBrain2.0-32B: gated on Hugging Face, waits for
  Yuxuan's token (accept the terms on the model page, then `hf auth login`).
- [x] 10-22 Qwen2.5-VL-32B load test passed on 2 x A40 (`VLM/test_vlm.py`); `VLM/README.md` written.
- [x] 10-22 RoboBrain2.0-32B downloaded (63 GB), load test passed on 2 x A40.

## E. Safe nominal controllers for every task (for safe-demo VLA fine-tuning on all tasks)

- [x] 10-22 Investigation: `docs/lang_filter/safe_controllers.md`. Safest-knob runs of 2026-10-05 (20 seeds per task
  and robot, rechecked from envs/runs/diag/oracle/): T5 (slow drag, time_scale 0.5) and T7 (centred slide,
  place_offset 0) clean and finished 40/40; T3 0/40 clean (needs a route planner around the elements, ~16-24 h);
  T2 0/20 clean on ur5, 6/20 franka (no safe pour known); T1 harmed in 11-14/20 even at the slowest speed (needs
  the vla_finetuning obstacle-aware cuRobo with every object in its world, ~8-12 h). Shared demo collector for the
  item scenes ~10-14 h (frames every 15 physics steps, instructions for the item tasks). Counts predate the 10-22
  scene changes (T1 / T3 to recheck).
- [x] 10-22 Yuxuan's answers (recorded in safe_controllers.md, "Decisions"): ur5 only; T1 keep = no contact with
  any object (no 1 cm buffer); T2 keep = no bead out at all; T3 = 80% named rule on an element in the way + the safe
  (detour) trajectory, 20% named rule on an element not in the way + the plain nominal trajectory; the safe
  controllers go into envs/ as an additional mode of the nominal controller. The same 80 / 20 split for T1, T5, T7.
- [ ] Implement (after the revision round is committed), in this order:
  - [x] 10-22 runner plumbing: `ItemEpisodeRunner(nominal_mode="safe")` (default "careless" unchanged), the task
    yaml's `safe_mode:` section; T5 safe = time_scale 0.5, T7 safe = place_offset 0; tests/item_envs/test_nominal_mode.py.
    10-22 sim check on today's scenes (runs/lang/safe_mode_1022): T5 8/8 and T7 8/8 finished with no rule broken;
  - [x] 10-22 T1 safe (runs/lang/safe_mode_1022/check_t1.py): `safe_plan.py` puts the
    protected objects (named rule's, else all extra objects; `protect=` overrides) in cuRobo's world, +2 cm, with the
    WSG hand + held object as attached spheres (copied from vla_finetuning/hand_model.py, which could import it now);
    RiskTickController.world_aware_only: no mplib fallback, branch-flip re-plans with cuRobo only; risky speed 0.6;
    keep = no contact with a protected object at any physics step (`safe_plan.contact_with`);
    First check (8 scenes, the object on the path protected): careless touched it 5/8, finished 2/8; safe touched it
    0/8 but finished 1/8 (4 plan failures, 3 "no grasp pose"). Fixes: cuRobo status log for the start-in-collision
    fallback, margin 2 -> 1 cm. Rerun (check_t1_v2): safe finished 4/8 with 0/8 touches (careless finished without
    touching it 1/8); the failures are "no grasp pose" (2, one also fails careless) and IK_FAIL at the target (2);
  - [x] 10-27/28 T1 safe rebuilt (hull meshes, the real table, margin ladder, wide grasp search; merged) and
    10-28 the table-free retry of a straight grasp / place descent (burger_fries 0/12 and bread_basket 1/12 -> 1/3
    and 2/3 in the GPU debug run);
  - [x] 10-27/28 T3 safe (route planner _push_route.py + push_via, merged) and 10-28 min_push 1.5 cm toward the
    goal (14 % -> 43 % kept; the box had stuck 5-12 mm short);
  - [x] (was: T3 safe: route around the in-the-way element (waypoints + a pusher that follows them, lifting the hand at
    corners);
  - [x] (dropped 10-23: T2 left the task set) T2 safe;
  - [x] 10-23 T6 safe mode (the pour aimed at the target, low and slow; ur5 aim lead fit to the plain bowls):
    final check (base 5) 21/24 finished, 1/24 put beads in the named bowl (T6 card);
  - [x] 10-27/28 the demo collector for item scenes (vla_finetuning/items, merged): Yuxuan 10-27 replaced the
    80 / 20 split by "protect every item, plain task instruction", 100 kept per task (T1 20 per host), keep = no
    rule broken; GPU pilot 10-28;
  - [x] collected 10-29 22:06: 500 / 500 (T1 20 per host, T3 / T5 / T6 / T7 100), every kept demo rescanned (no
    item above 1.4 m). 12 reused from the GPU pilot and the T1 debug run (same collector and settings). Fixes on the
    way: T3 min_push (14 -> 43 % kept), T1 grasp descent raised 5 mm above the table / table-free retry, the
    flown:<item> keep rule (2 burger_fries demos with the fries shot to 1.7 m dropped), T1 collectors weighted 2
    slots (14.5 GB GPU each); burger_fries stayed hard (~6 % kept: the fries carton tips in the grasp);
  - [ ] fine-tune HoloBrain: ONE model for all five tasks (decided 10-28), ONE GPU (Yuxuan), GRAD_ACCUM 4 (batch 32),
    15k steps; running as chain runs/lang_pipeline_ft on A40 interactive 1-hour jobs (ft_chain_step.sh: packs one
    LMDB per task, then trains, resuming from the newest checkpoint); batch backup job 23327423 held;
    10-30 ~15:15 PAUSED (Yuxuan: the actor-issue checks first; the training needs no A40, only its sim evaluation
    does) via runs/lang_pipeline_ft/state/hold.ft_items, at step ~8000 of 15000 (25 GB, fits an A100). Resume: delete
    the hold file (the chain6 orchestrator keeps running); check the base-eval job 23329871 first (a test running at
    the same time as the pipeline's own base test fails on the evaluator lock);
    Yuxuan 10-30 18:45: resume it as soon as the actor-issue verification is done (check 4's 20k verdict, rescored,
    reported): stop the A40 retrain feeder first (scripts/lang/diag_1030/a40_feeder.sh on dt-login01), then delete
    the hold file;
    10-31 07:56 RESUMED (feeders stopped, hold file removed; chain6 451528 submitted 23374798). The separate
    base-eval job 23329871 is HELD (`scontrol release 23329871` to undo): the pipeline runs the same base test itself
    and the two must not overlap. New checkpoints and results land on the HDD pool (see the STORAGE entry);

  - [ ] an evaluator for the item scenes (evaluate.py knows only the collision-avoidance tasks), then evaluate.

## F. VLM as the failure function (Yuxuan 2026-10-26, docs/lang_filter/VLM_FAILURE_PLAN.md)

- [x] 10-26 Code (24b9444, pushed): agent-view frames during collection, the labeler (`main/lang/vlm_label.py`), the
  h switch (`h_source`, `rule_consequences`, `hsource.py`), rounds wait for labels; CPU tests
  tests/lang/test_vlm_failure.py. Frozen worktree LanguageConditionHJ_rvlm.
- [x] 10-26 Smoke (runs/lang/vlm_1026/smoke): 6 warmup episodes replayed with frames (frames line up with the rows);
  RoboBrain labelled them (852 questions in 9 min on two A100s, every answer parsed). Its agreement with the
  simulator: T6 catches 69 % of the harmed ticks (precision 1.0), T5 35 % (0.74), T1 touches 0 % (it calls every T1
  tick safe). Every Delta A100 has 40 GB: a 32B model runs on two (gpuA100x4 / gpuA100x8 interactive).
- [x] 10-26 Data: chain stage vlm1026_data (16:33-17:00) -> data/lang_vlm_1026/READY; 15 shards, frames check and
  check_shards clean.
- [x] Warmup labelled: LABELS_qwen25vl32b 10-26 23:19, LABELS_robobrain2_32b 10-27 02:16 (90 / 90 episodes each; one
  Qwen job lost to a GPU fault, "GPU requires reset"). Chains runs/lang_pipeline_vlm_{qwen25vl32b,robobrain2_32b}
  (orchestrators and labeler_watch.sh on dt-login02) now held until the runs collect. Agreement with the simulator
  on the warmup (runs/lang/vlm_1026/agree_warmup_<vlm>.json), share of the simulator's harmed ticks caught / share of
  the VLM's harmed answers that are right: Qwen T6 0.43 / 0.90, T5 0.52 / 0.98, T1 0.28 / 0.12; RoboBrain T6
  0.54 / 0.77, T5 0.85 / 0.86, T1 0.15 / 0.15. Uncertain answers: Qwen T6 34 %, T1 11 %; RoboBrain T6 13 %.
- [x] 10-27 the two 32B VLM runs cancelled (Yuxuan); 10-27/28 compared 7 labelers incl. 3 small ones and 4-bit / 8-bit
  32B (VLM_FAILURE_PLAN.md); 10-29 the saved 8-bit copies decided it: Qwen 0.851 vs RoboBrain 0.764.
- [ ] (10-30 both cancelled with every round training, see A; not started) Runs: rounds_ur5_simh_g095_1026 (job 23230275, the sim-h reference, pending, estimate 10-30 13:21) and
  rounds_ur5_vlm_qwen25vl32b_fp8saved_g095_1028 (job 23319949, one 2-A40 job: GPU 0 trains, GPU 1 labels; pending);
  25 rounds x 16 episodes x 8192.
- [ ] Report: VLM vs simulator agreement on the warmup (`vlm_label agree`), then the closed-loop results.

