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
  sentences, splits; 361 lang + 231 env tests) and committed as f1d2488. [ ] Push: the GitHub token in GH_TOKEN is
  rejected (401, expired since the 10-20 push); needs a new token from Yuxuan.
- [ ] When Yuxuan says go: re-collect the warmup and validation data of T1, T3, T5, T7
  (old shards hold the old objects, colours and layouts; T2 is unchanged); cancel the three round jobs and
  resubmit (`runs/lang/rounds_1020/launch.sh` as the template, new run names).

## B. Evaluation of the new round trainings

- [ ] Distance probe: port branch probe-1013 to main with the new sentence bank, plus Qwen support (Qwen needs
  replays of data/lang_probe with frames / kv).
- [ ] `eval_closed` for Qwen (it still refuses Qwen).
- [ ] Report success, collision rate (CR: the named rule broken; T1 a touch) and safe success (success with the
  named rule never broken) for vanilla gamma 0.95, Qwen gamma 0.95 and T1-only gamma 0.95.

## C. VLA fine-tuned on safe demonstrations (collision avoidance)

- [ ] Fine-tune job 23051382 (vf-ft-safe, gpuA100x4, pending) -> export (`vla_finetuning/jobs/export.sh`) ->
  `vla_finetuning/evaluate.py --agent vla` on the 250 test scenes -> report collision rate and success rate.

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
  - [ ] T1 safe: the named object (in the way) in cuRobo's world with the WSG hand + held-object model (from
    vla_finetuning/hand_model.py), 2 cm margin, stock executor; keep = no contact with it at any physics step;
  - [ ] T3 safe: route around the in-the-way element (waypoints + a pusher that follows them, lifting the hand at
    corners);
  - [ ] T2 safe: find where beads leave in a gentle pour (saved runs first), then a pour with zero beads out;
  - [ ] the demo collector for item scenes (vla_finetuning): frames every 15 physics steps, task instructions +
    the constraint sentence, the 80 / 20 split (in-the-way named + safe trajectory / not-in-the-way named +
    nominal trajectory), ur5-wsg, the HDF5 writer;
  - [ ] collect, fine-tune HoloBrain, evaluate as in C.
