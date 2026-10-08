# Vanilla language-conditioned item safety filter: implementation plan for a coding agent

Repo: **`LanguageConditionHJ`** (GitHub `JsSparkyyx/LanguageConditionHJ`, on Delta at
`/work/nvme/bifq/project/LanguageConditionHJ`). Write here.

| part of the repo | what it is |
|---|---|
| `main/` | CrossSafe (HJ-SAC filter on frozen HoloBrain), with the Phase-5 item hooks in `main/train/collect.py` and `main/train/rollout.py` |
| `envs/main/item_envs/` | the 7 item tasks' Python side, imported as `main.item_envs` through a path bootstrap (Section 2) |
| `envs/robotwin_tasks/` | RoboTwin task files, symlinked into `../RoboTwin/envs/` |
| `envs/docs/item_envs/` | environment docs: `relabeling.md`, `tasks_overview.md`, `task_cards/`, `final_report.md` |
| `../RoboTwin` | RoboTwin with assets (`/work/nvme/bifq/project/RoboTwin`) |

HoloBrain checkpoint: `HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD`.
Container: `/work/nvme/bifq/robot-sim_cu128.sif` (sapien 3.0.0b1, cuRobo, torch). Cluster: NCSA Delta,
account `bifq-delta-gpu`, A40 GPUs (48 GB).

---

## 0. How to use this document

You are implementing the **vanilla** text-conditioned safety filter on top of the item environments:

- the model (CrossSafe plus text), in new files, without editing the CrossSafe network files;
- an extension of the existing relabeling sampler (multi-source, 3 frames, controls), plus a text store;
- one episode loop for warmup, validation and round collection (modeled on the existing item-task
  collection), round-based training with a filter that reads a sentence, logging, videos;
- an offline language probe and a closed-loop evaluation grid.

You are **not** implementing the advanced version (object slots and an entity pointer), new tasks, or
changes to environment physics, margins or rule tables.

### 0.1 What already exists (reuse it, do not rebuild it)

| piece | where | notes |
|---|---|---|
| 7 item tasks, registry, runner | `envs/main/item_envs/registry.py`, `runner.py`, `envs/robotwin_tasks/` | T1 hosts round-robin by seed |
| item-task collection with a lockstep sidecar | `main/train/collect.py` (`is_item_task`, `make_training_env` → `margin_hook.build_item_training_env`, `margin_hook.wire_item_episode`) | `env.obstacle` is the "has rules" marker. The reference for our episode loop (Section 7.1) |
| per-tick labels of **every** rule | `ItemSidecarBuffer` in `<data>/item_buffer/` (`envs/main/item_envs/item_buffer.py`): `h_rules`, `fails_soon` (25 ticks), `margins`, `rule_valid`, `active_rule`, `t_in_ep`, `episode_id`, `token_masks`, item state | same row index as `<data>/buffer/` (StepBuffer) |
| per-episode relabeling records | `<data>/item_buffer/episodes/{episode_id}.json` (`language.episode_record`) | objects, regions, rules with `sampleable` |
| h on one scale | `margins.rule_h = clip(margin / h_unit, −1, 1)`, 1.0 = comfortably safe | no per-rule gains needed |
| paraphrase bank, referent phrases, splits | `envs/main/item_envs/language/` (`paraphrases.yaml`, `describe`, `sentences`, `sample_sentence`), `configs/splits.yaml` | splits `train`, `heldout_template`, `heldout_word`, plus `heldout_item` rules |
| frozen HoloBrain text encoder | `language/text_encoder.HoloBrainText` | checkpoint's own BERT + `text_feat_map`, parity checked |
| text cache builder | `envs/scripts/item_envs/build_text_cache.py` | `{"index", "emb" fp16 [N, 24, 256], "mask" [N, 24], "split"}` |
| relabeling sampler | `envs/main/item_envs/relabel.py` (`RelabelSampler`) | harmful/benign balance, pairs from mixed rows, rows t, t+1, t−k, first |
| docs | `envs/docs/item_envs/relabeling.md`, `tasks_overview.md`, `task_cards/T1..T7.md` | read before Phase 0 |

### 0.2 What this plan adds

`main/lang/` (model, sampler extension, text store, scene roles, rollout, collection, trainer, metrics,
video, evaluation), `scripts/lang/` (container and Slurm wrappers, matrix), `tests/lang/`, `docs/lang_filter/`,
and `CLAUDE.md` at the repo root.

Read first: `AGENTS.md`, `docs/lang_filter/DESIGN_CONTEXT.md`, `envs/docs/item_envs/relabeling.md`,
`envs/docs/item_envs/tasks_overview.md`.

Work phase by phase (Section 10). Each phase ends with passing tests, a short report in
`docs/lang_filter/`, and a commit with prefix `lang(phaseN):`. Phases with a **STOP gate** need a human
review before you continue. Judge with numbers. Videos are for humans.

### 0.3 Glossary

| term | meaning |
|---|---|
| tick | one control step, 0.04 s (25 Hz), exactly 10 physics steps |
| data root | a directory with `buffer/` (StepBuffer) and `item_buffer/` (sidecar + `episodes/`) written in lockstep |
| rule | one row of an episode's rule table, for example `upright|block_1|-`. Its column index j addresses `h_rules[:, j]` |
| consequence | the rule type in the records: `avoid`, `upright`, `stay`, `zone_exclude`, `zone_include`, `contain_particles`, `exclude_particles` |
| sampleable rule | a record rule with `sampleable: true`: training split, single referent, at least one training sentence in that scene. Set, pair and ambiguous rules are recorded but never sampled |
| `h` | `h_rules[t, j] = clip(margin / h_unit, −1, 1)`: 1 = comfortably safe, 0 = harm, −1 = badly harmed. Q is learned in these units |
| harmful / benign at tick t | rule j is harmful at t if `fails_soon[t, j]` (h_j < 0 somewhere in ticks t … t+24 of the episode), else benign |
| mixed row | a row with at least one harmful and one benign sampleable rule |
| in the way | an item the careless nominal is aimed at or will run into, read from the task's own scene metadata (Section 3.5). Deterministic per seed |
| harm-designated / benign-designated rule | a rule (not a set or ambiguous rule) whose referent is / is not in the way. Referents that the nominal hits in some runs are `unclear` and never named. Used to choose named rules in rounds (sampleable rules only) and in evaluation (rules eligible for the axis) |
| primary rule | the environment's **scenario referent** (`active_rule_id="primary"`), drawn by each task from a calibrated mix of in-the-way and benign scenarios (for example T6: hit container 70%, other 30%), and in T3 sometimes a pair rule. Stored as a label only, never used to decide harm or benign |
| named rule | the rule a sentence refers to, in a training sample or a filter run |
| frames t, t−k, 0 | scene tokens at the current tick, k ticks earlier (k = 2), and at the episode's first tick |
| combo | (task, embodiment). T1 hosts are balanced inside T1's share |

---

## 1. What we are building

### 1.1 Problem

Each episode has several sampleable rules, and a sentence names one. The filter must keep the named
rule's h above 0 and must leave the nominal controller alone when only unnamed items are at risk. It
sees the 3 cameras, the joint state and the sentence, changes arm joint motion, and never drives the
grippers.

### 1.2 The vanilla model in one paragraph

CrossSafe (per-link tokens, geometric link-to-scene attention, Jacobian action injection, post-action
geometry, softmin readout, token actor, HJ-SAC) plus text. The scene is three frames (t, t−k, 0) of
frozen HoloBrain tokens, 3600 tokens with their own world positions and a learned time embedding per
frame. The sentence goes through frozen HoloBrain BERT plus `text_feat_map` and a small trainable text
adapter. Text reaches the model through **five paths**, each with a switch:
- scene tokens read the words once;
- link tokens read the words in every block;
- FiLM from the sentence summary `r₀` modulates every block;
- 4 readout queries carry scene-located danger into the readout;
- the rule token `r` joins self-attention.

Training uses only the HJ-SAC losses. Text conditioning is learned from **relabeling**: the same
transition is trained with different rules of the same scene, whose h differ. There are no rule types,
task ids, slots, pointers or auxiliary losses anywhere in the model.

### 1.3 Research questions the code must support

| id | question | what the code must provide |
|---|---|---|
| Q1 | Does HJ loss plus counterfactual relabeling teach the filter to read the sentence? | `relabel ∈ {paired, random, fixed}` and a shuffled-sentence control |
| Q2 | Where must language enter? | five path switches, and metrics that separate referent decisions from consequence decisions |
| Q3 | Which text space? | a pluggable text backend: HoloBrain BERT + `text_feat_map` (default), raw BERT with a trainable projection, one generic sentence encoder |
| Q4 | Which generalization axis fails first? | evaluation on `heldout_template`, `heldout_word` and `heldout_item` |

---

## 2. Ground rules

- **New code lives in `main/lang/`** at the repo root. Do not edit:
  - the CrossSafe files `main/network/*`, `main/train/*` (the Phase-5 hooks are already there), `main/envs/*`;
  - anything under `envs/` (item environments, language, sampler, docs), RoboTwin files.

  Subclass and compose. If an env-side change is truly needed, stop and ask. The one exception is the
  `DESIGN_CONTEXT.md` pointer of Section 2.1.
- **Imports.** Run language-filter code from the repo root (`python -m main.lang...`). There, `main` is
  `<repo>/main`. `main/lang/__init__.py` must append `<repo>/envs/main` to `main.__path__` before anything
  imports `main.item_envs` (the idiom of `collect.is_item_task` and `envs/main/item_envs/_bootstrap.py`).
  Env tests still run from `envs/` (`pytest tests/item_envs ...`).
- **Container.** Every GPU or simulator command runs inside the apptainer, through
  `scripts/lang/in_container.sh` (write it by copying the `apptainer exec` line of
  `envs/scripts/item_envs/stage_runner.sbatch`, with cwd = repo root instead of `envs/`).
  - Slurm templates go in `scripts/lang/`: one for 1-hour interactive stages (`gpuA40x4-interactive`)
    and one for long jobs (`gpuA40x4`, the partition's maximum walltime, resume on requeue).
  - Compute nodes run wandb offline (`WANDB_MODE=offline`). Sync afterwards with `wandb sync`.
- **Config.** `FrozenConfig` defaults stay unchanged. Add `LangConfig(FrozenConfig)` in
  `main/lang/config.py` (Section 11). Dict fields use `field(default_factory=...)`.
  - Save it with every data root, run and checkpoint, and log its hash.
  - Assert `gamma_final == gamma`, because `Trainer.gamma()` interpolates between them.
  - `build_item_training_env` reads per-episode overrides from the cfg: `item_host`, `item_risk`,
    `item_active_rule`, `item_out_root`. Declare all four as `LangConfig` fields, because
    `dataclasses.replace` drops undeclared attributes. Defaults: None for the first three, and
    `item_out_root = "runs/item_envs"`. A None out root makes the runner call `Path(None)`, every build
    fails, and the loop silently skips every slot.
- **Regression.** Before any change and at the end of Phases 2 and 4:
  - run the environment's Phase-5 regression (`envs/tests/item_envs/test_phase5_regression.py`, a
    statistical contract, because cuRobo is nondeterministic run to run);
  - check that `main/train/train.py` still imports and runs 20 grad steps on a CrossSafe buffer.
- **One writer per memmap.** Each collector process writes its own data root. Training reads several
  data roots as sources (Section 4.4), so shards are never merged or copied.
- **Child processes.** Long simulation loops (collection, evaluation) run one child per (embodiment,
  shard).
  - A child exits after 200 environment inits, and the parent restarts it until its work list is done.
    Repeated `Env` creation leaks pybind11 TLS keys (`AGENTS.md`).
  - After 3 crashes on the same work item, write an error row and move on.
  - Rounds training follows the same limit (Section 7.4).
- **wandb.** Project `cehj-lang`. Every call goes through `_safe_wandb_log`.
- **Units.** h is already on one scale (Section 0.3). Q is compared only with h. Panels show h and Q
  (unitless), never centimetres. `StepBuffer.h` and the base controller's `trace["h"]` are not read by
  `main/lang` except in the Phase-0 parity check.
- **cuRobo is nondeterministic run to run** (`envs/docs/item_envs/final_report.md`). The same seed gives
  the same scene layout, but not exactly the same nominal motion. The evaluation design accounts for
  this (Section 9.2).
- **Tests.** Pure functions are tested without SAPIEN or a GPU. Register the markers `sim` and `gpu` in
  `tests/lang/conftest.py`. Simulator and GPU tests are parametrized over `["ur5-wsg", "franka-panda"]`.
- **Batch layout.** Every attention layer in `main/lang` uses `batch_first=True`. A test permutes the
  batch and checks that every module's outputs permute the same way.
- **Randomness.** `np.random.default_rng(seed)` Generators (the type `RelabelSampler` and `language`
  expect), owned by the caller. The base `RolloutController` takes a legacy `np.random.RandomState` for
  its perturbations. Pass `np.random.RandomState(seed)`.

### 2.1 Target layout

```
CLAUDE.md                    # imports AGENTS.md and docs/lang_filter/DESIGN_CONTEXT.md
main/lang/
  __init__.py                # main.__path__ bootstrap for main.item_envs
  config.py                  # LangConfig(FrozenConfig), seed/id helpers (Section 4.2)
  text.py                    # TextStore (cache + HoloBrainText fallback), alternative backends
  data.py                    # LangSampler (multi-source, built on RelabelSampler), StepBuffer row loading, PrefetchSampler
  network.py                 # TextAdapter, SceneSet, LangTrunkBlock, LangPolicyEncoder, LangEncoded,
                             # action_effects(), LangCriticHead, LangTwinCritic
  rollout.py                 # LangRolloutController, LangPolicyShim, wire_item_eval
  scene_roles.py             # in_way, designation (Section 3.5)
  collect.py                 # run_episodes (warmup, validation, rounds), warmup/validation CLI, report (CLI)
  train.py                   # LangTrainer + CLI (offline and rounds modes, resume)
  metrics.py                 # training metrics, probes, attention recomputation, LangRoundMetrics
  video.py                   # LangPanelVideoWriter
  eval_offline.py            # offline language probe (CLI)
  eval_closed.py             # closed-loop evaluation grid (CLI)
  report.py                  # tables and figures (CLI)
scripts/lang/                # in_container.sh, sbatch templates, run_rounds.sh, matrix.sh
tests/lang/
docs/lang_filter/            # DESIGN_CONTEXT.md, VANILLA_PLAN.md, recon.md, collect_report_*.md, phase reports, final_report.md
```

Replace `envs/docs/item_envs/DESIGN_CONTEXT.md` with a one-line pointer to
`docs/lang_filter/DESIGN_CONTEXT.md` (the old copy predates the current design). This is the one
allowed edit under `envs/`.

---

## 3. Using the item environments

### 3.1 Tasks and rules (from `relabeling.md` and `tasks_overview.md`)

| task | rules in an episode | consequences | h |
|---|---|---|---|
| T1 `selective_avoid` (5 CrossSafe hosts, 3 to 5 extra objects of different types) | per extra object: `avoid`, `stay`. Set rules (recorded, never sampled) | `avoid`, `stay` | avoid: (real-shape gap − 1 cm) / 5 cm |
| T2 `granular_pour` | one | `contain_particles` | safe: depth of the worst bead inside the columns / 1 cm. Violated: −0.5 for one bead out, linearly to −1 at 20 beads (2026-10-07) |
| T3 `push_zone` | table, per mat (+ pair rules), per standing object `upright` + `stay` (+ pair) | `zone_include`, `zone_exclude`, `upright` (10°), `stay` (1.5 cm, 10°) | zones / 5 cm. upright (10° − tilt) / 10°. stay (1.5 cm − moved, tilt as movement) / 1.5 cm |
| T4 `granular_carry` | one | `contain_particles` | as T2 |
| T5 `tray_transport` | per block (2 to 3) | `upright` (10°) | (10° − tilt) / 10° |
| T6 `clean_pour` | per bin or plate, plus the spill rule | `exclude_particles`, `contain_particles` | safe: distance of the closest bead from entering the column / 5 cm. Violated: by bead count, as T2 |
| T7 `gap_place` | per standing block: `upright` + `stay` | `upright` (10°), `stay` (2 cm, 10°) | as T3 |

Only sampleable rules are ever named. T1, T3 and T7 have **two consequences on the same item** (avoid
vs stay, upright vs stay). These are the consequence decisions Q2 measures.

### 3.2 Interfaces (exist; confirm behavior in Phase 0)

| need | interface |
|---|---|
| is a task an item task | `main.train.collect.is_item_task(task)` (also performs the path bootstrap) |
| build an item env | `make_training_env(cfg, seed)` → `build_item_training_env`. Per-episode cfg attributes: `item_host` (T1 host; when None the runner picks `host_tasks[seed % 5]`), `item_risk`, `item_active_rule` (a rule id, `"primary"`, or None = h is the min over all rules), `item_out_root`. **Declare all four as `LangConfig` fields** (defaults None, None, None, `"runs/item_envs"`): `dataclasses.replace`, used by `collect.sample_scene` and by eval loops, silently drops undeclared attributes |
| scene metadata | T1 `runner._spawn_report["placements"]` (`cls` = `on_path` / `near_path` / `off_path`), T3 `env.task.layout_info["elements"]` (`on_path`, `role`), T5 / T6 / T7 `env.task.episode_info` (`referent_kind`, `hit`) |
| primary rule | `runner._primary_rule_id(env.task)`, a label only (Section 3.3 item 7) |
| the episode context | `env._item_runner` (an `ItemEpisodeRunner`: `.items`, `.regions`, `.rules`, `.active_rule_id`, `.task_cfg`) |
| controller | `RolloutController(cfg, seed, mode, encoder, kin, policy_enc, actor, critics, filter_active, buf, episode_id, perturb_prob, rng, video, max_steps, h_fn, controller_cls)`. `max_steps` counts physics steps (ticks × 10). Item tasks use `controller_cls=RiskTickController`. `run()` closes the env |
| collection wiring | `wire_item_episode(ro, sidecar)`. It sets `ro.h_fn = ItemMarginHFn(runner.active_rule_id, 1/h_scale, row_sink=LockstepItemWriter)`, swaps `ro.buf` for the lockstep writer, hands T1's `risky_skill_labels` to the controller, and writes `episodes.jsonl` plus `episodes/{id}.json` |
| per-tick h of all rules | `ItemMarginHFn.__call__` returns `(h, diag)`. `diag["margins"]` holds every rule's raw margin, and `rule_h(rule, margin)` converts. `StepBuffer.h` = the active rule's h, or the min over all rules |
| lockstep teardown | `LockstepItemWriter.end_episode()`. `collect()` never calls it, so our loop must (Section 7.1) |
| sidecar | `ItemSidecarBuffer(<data>/item_buffer)`; `for_step_buffer(buf, n_tok=masks.token_grid()[1])` for a fresh pair |
| language | `language.episode_record`, `describe`, `describe_region`, `sentences(rule, rec, split)`, `sample_sentence(rule, rec, rng, split)`, `load_bank` |
| text | `HoloBrainText(device=...).embed(sentences) → (emb [B, L, 256], valid [B, L])` |
| sampler | `RelabelSampler(<data>/item_buffer, split, harmful_frac, pair_frac, k_prev, text_cache)` |
| standalone debugging | from `envs/`: `python -m main.item_envs.probe episode --task push_zone --seed 0 --embodiment ur5-wsg --video` |

### 3.3 Known facts that change the implementation

1. **Terminal rows.** `RelabelSampler` includes an episode's last row (`done=True`, next = itself).
   CrossSafe never trains on done rows. Exclude them (Section 4.4).
2. **`k_prev`** defaults to 1 in `RelabelSampler`. We use k = 2.
3. **Text cache coverage.** `build_text_cache.py` embeds train sentences only for `sampleable` rules.
   The `heldout_item` axis uses training templates on rules with env split `heldout_item`, and those
   sentences are not in the cache. The text store must encode misses (Section 4.3).
4. **Episode length.** `collect()` caps episodes at `cfg.max_steps_per_episode × 10` physics steps
   (FrozenConfig: 450 ticks = 18 s). The standalone runner uses 7500 steps (30 s). Phase 0 measures
   episode lengths per task and sets `LangConfig.max_steps_per_episode`.
5. **Rendering cost.** HoloBrain needs camera images every tick, so collection renders every tick.
   The environment report measured 3 to 10 min per rendered episode. Phase 0 measures wall time per
   episode with encoding. Collection runs in parallel shards.
6. **Determinism.** Layout is deterministic per seed, motion is not (cuRobo). See Section 9.2.
7. **"Primary" is not "at risk".** Each task draws its primary referent from a scenario mix:
   - T3: roles A (violated), B (crossed, allowed), off-path, and pair rules in A-A scenes;
   - T5: tall 0.75 / short 0.25;
   - T6: hit 0.7 / other 0.3;
   - T7: hit 0.6 / other 0.25 / bystander 0.15.

   So harm and benign come from Section 3.5, not from `primary`.
8. **`main.train.collect.collect()` is the reference, not the tool.** It is item-aware and verified,
   but:
   - it records no per-episode status or success;
   - every shard gets the same shuffled schedule (seed = `round_index` = 0);
   - an episode without rules is retried 5 times with the **same** seed;
   - it builds a plain `RolloutController`.

   All collection (warmup, validation, rounds) therefore runs through one loop in `main/lang/collect.py`
   that mirrors `collect()` (Section 7).

### 3.4 Phase-0 checks (record in `docs/lang_filter/recon.md`)

1. From `envs/`, run `pytest tests/item_envs -m "not sim and not gpu"` (this includes `test_relabel.py`,
   `test_language.py`, `test_item_buffer.py`, `test_margin_hook.py`). Then, in the container on a GPU
   node, run `test_phase5_dryrun.py`, `test_phase5_regression.py` (both marked `gpu`) and the `*_sim.py`
   tests of the 7 tasks.
2. Collect 2 nominal episodes per task and embodiment with the Section 7.1 loop into a scratch root.
   Check:
   - `StepBuffer.h[row] == h_rules[row, active_rule]`, or the min over valid rules when
     `active_rule = −1`;
   - `len(buffer) == len(sidecar)`;
   - `t_in_ep` runs 0, 1, 2… per episode;
   - `episodes/{id}.json` exists for each episode.
3. Episode length per task (mean, max), timeouts at the current cap, wall time per tick split into
   rendering + encoding, physics, and the rest.
4. `HoloBrainText` embeds 5 sentences on the GPU. Compare with `build_text_cache.py` output for the same
   sentences (max abs difference 1e-3, fp16).
5. Build a filtered base `RolloutController` without a sidecar (`wire_item_eval`, Section 7.5) and run
   one episode per task with a random-init CrossSafe model (the language model comes in Phase 2). Check
   that the h_fn runs and `diag["margins"]` has one entry per rule.
6. Implement `in_way(env, record)` and `designation` (Section 3.5). Over 10 seeds per task and
   embodiment (25 for T1, 5 per host), record:
   - the in-the-way and unclear items, and the scenario (`referent_kind`, T3 `scenario`);
   - the primary rule and whether it is sampleable;
   - per rule with a designation, whether it was violated in the nominal run.

   Harm-designated rules should be violated often, and benign-designated ones rarely. Report both
   rates per (task, embodiment, consequence), per host for T1, and for unclear items separately. A
   harm-designated consequence violated in fewer than 20% of runs (for example T7 `upright` on Franka,
   where the scrape shoves rather than tips) is reported for the human to keep or drop from harm runs.
7. The `token_masks` row order (items in record order, then nameable regions that have an entity), and
   the map from a record rule's subject and reference to those rows. Write `ref_rows(record, rule)`.

**STOP** if 2, 4 or 5 fail, or if any task's benign-designated rules are violated in more than 20% of
nominal runs (then the in-the-way classification is wrong for that task).

### 3.5 In the way (`main/lang/scene_roles.py`)

`in_way(env, record) -> (set[str], set[str])`: the ids of the items and regions the careless nominal is
aimed at or will run into, and the ids that are unclear. All of them come from the task's own scene
metadata, which is deterministic per seed:

| task | in the way (and unclear) |
|---|---|
| T1 | candidates with `cls == "on_path"` in `runner._spawn_report["placements"]`. Unclear: `near_path` candidates (2 to 6 cm from the corridor, hit in some runs), and `off_path` candidates within `off_path_min_dist` (10 cm) of any other corridor of the host (`iter_corridors(env.task, TASK_SPECS[host])`), because `off_path` is measured from the longest corridor only and two-carry hosts have a second path |
| T2, T4 | the bead set (one rule. The nominal spills with some probability) |
| T3 | elements with `on_path == True` in `env.task.layout_info["elements"]` (roles A and B) |
| T5 | the tall block (`size == "tall"` in the record) |
| T6 | the hit container (`env.task.episode_info["hit"]`) and the bead set (beads that reach a neighbour are also outside the target, so the spill rule is at risk too). Unclear: the container in the diagonal `far_across` slot (`episode_info["neighbors"][k]["slot"]`, 45° from both outer slots, which splashes reach). The other outer slot is the task's own benign referent, 90° ± 20° away (each slot is jittered ±10°). Changed 2026-10-07 (review R1): the earlier cut at 90° of azimuth marked that benign container unclear in half the scenes. Records without slot names use a 67.5° cut |
| T7 | the hit flanker (`env.task.episode_info["hit"]`) |

The **referent** of a rule, for this purpose:
- `avoid`, `upright`, `stay`: the subject item;
- `zone_exclude`, `exclude_particles`: the reference region (the mat, the bin or plate);
- `contain_particles`: the bead set;
- `zone_include`: the table, which is never in the way (goals sit inside the table).

Ids are the env's string ids (`item_id`, `region_id`). A record rule points to its referent by index
(`subject {"object": k}`, `reference {"region": k}`), so convert with `record["objects"][k]["item_id"]`
and `record["regions"][k]["region_id"]` (the bead set: its particle object's `item_id`). A test on one
saved record per task checks the conversion.

`designation(rule, record, roles) -> "harm" | "benign" | "unclear"`, for every rule that is not a set or
ambiguous rule (evaluation names held-out rules too, which are never sampleable): **harm-designated** if
its referent is in the way, **unclear** if its referent is unclear, else **benign-designated**. Rounds
name only sampleable rules (Section 7.1). Unclear rules are never named. Store every rule's designation
and the environment's `primary` / `referent_kind` in `lang_episodes.jsonl`.

---

## 4. Data

### 4.1 Data roots

```
data/lang/warmup/shard_g<g>/      # one per collector process
  buffer/  item_buffer/{*.npy, header.json, episodes/, episodes.jsonl}  config.json  videos/
data/lang/warmup/text_cache.pt    # build_text_cache.py over all warmup shards
data/lang/val/shard_g<g>/ ...     # validation, never trained on
data/lang/val/text_cache.pt
runs/lang/<name>/ring/{buffer, item_buffer}   # rounds mode: a fresh pair per run
```

`data/` and `runs/` are gitignored.

### 4.2 Seeds and episode ids

Our loop keeps `collect()`'s convention `seed = episode_id = episode_offset + attempt`, where `attempt`
counts every environment build of the stream, retries included, so ids never repeat within a stream:

| stream | `episode_offset` | ids and seeds |
|---|---|---|
| warmup shard g | `(1 + g) · 1_000_000` | 1M … 51M (g < 50) |
| validation shard g | `(60 + g) · 1_000_000` | 60M … 90M (g < 30) |
| rounds, round r | `(100 + r) · 1_000_000` | ≥ 100M |
| evaluation scene | not an episode id. Seed `1_500_000_000 + task_idx·10M + host_idx·1M + emb_idx·100k + attempt·1000 + scene_idx` (scene_idx < 1000, attempt < 10) | 1.5B … 1.57B |

- `g = emb_idx · n_shards + shard`. `emb_idx`, `task_idx` and `host_idx` are indices in fixed canonical
  tuples in `main/lang/config.py` (`CANON_TASKS`, `CANON_EMBODIMENTS`, `CANON_T1_HOSTS`), never in the
  filtered `task_choices` or `embodiment_choices`. A single-task or single-embodiment run then evaluates
  on the same seeds as `full`.
- Assert g and attempt bounds at startup. A test checks that all ranges are disjoint and fit int32.

### 4.3 Text store (`main/lang/text.py`)

`TextStore(cache_paths, backend="holobrain_bert", device="cuda", extra_path=<run>/text_cache_extra.pt)`:
- loads one or more caches built by `build_text_cache.py` (`index`, `emb` fp16 [N, 24, 256], `mask`,
  `split`);
- `lookup(sentences) -> (words [B, 24, 256], mask [B, 24])` on GPU;
- encodes misses with `HoloBrainText` in one batch, adds them, and persists them to `extra_path` when it
  grows by 1000 entries and at checkpoints. Logs `data/text_miss_frac` by split.

Other backends (Q3):
- `bert_raw`: the same BERT without `text_feat_map`, 768-d, with the adapter's input projection
  `Linear(768, 256)` trainable;
- `sentence_generic`: one frozen general-purpose encoder with token outputs (for example
  `sentence-transformers/all-mpnet-base-v2`) from a local path, plus a trainable `Linear(d, 256)`.

Both get their own cache files, built by a variant of `build_text_cache.py` in `main/lang/text.py`
(enumerate sentences with the same `collect_sentences` logic).

### 4.4 `LangSampler` (`main/lang/data.py`)

Built on `RelabelSampler`, by composition or subclassing, without editing `envs/`. **Sources** = a list
of data roots: all warmup shards, plus the run's ring in rounds mode. Each source has its
`RelabelSampler(<root>/item_buffer, split="train", k_prev=2)` for the index and its StepBuffer (opened
read-only with `np.load(..., mmap_mode="r")`, except the ring the run itself writes).

Additions over `RelabelSampler`:

1. **Global index** over (source, row), with task, host and embodiment per row from the records.
   **Terminal rows are excluded.** Filters: `--tasks`, `--embodiments`.
2. **Task balance** (default on): draw a task uniformly, then a row of that task uniformly. T1 hosts
   share T1's draw.
3. **Pairs from mixed rows, task-balanced.** `n_pair = round(pair_frac · B / 2)` (16 at B = 64,
   `pair_frac` 0.5). Each pair is one transition with one harmful and one benign sampleable rule. Tasks
   without mixed rows (T2 and T4 have one rule) contribute no pairs. Log `data/pair_mixed_frac`, which
   should be 1.0.
4. **Single samples**: rule harmful with probability 0.5 when both kinds exist, as `RelabelSampler`.
5. **Relabel modes**:
   - `paired` (default): items 3 and 4.
   - `random`: no pairs.
   - `fixed`: one rule per episode, never changing. In a filter episode that is the named rule (the
     record's `active_rule`, Section 7.1). Otherwise a sampleable rule chosen by a hash of `episode_id`.
     No pairs. This is the Q1 control: language is informative, but no state is ever seen with two
     rules.
6. **Shuffled control** (`shuffle_text`): each sample's sentence is replaced by a training sentence of a
   sampleable rule from a **different episode**, redrawn every step.
7. **Rows returned**: `row` (t), `next_row` (t+1), `prev_row` (t−k), `first_row`, and
   `next_prev_row = next_row − min(k, t_in_ep + 1)`. All ring-aware (modulo capacity, as
   `RelabelSampler` does).
8. **`fails_soon_recall`** (7 ticks, matched to γ = 0.9), computed at refresh from `h_rules` per
   episode with a reverse sliding minimum. Used only for recall metrics.
9. **Loading**: StepBuffer fields at `row` and `next_row` (robot fields, `dtheta`, Jacobians,
   `dtheta_max`), scene tokens and positions at [row, prev_row, first_row] and [next_row,
   next_prev_row, first_row].
10. **`ref_mask`** (probe batches only): the named rule's referent rows of the sidecar `token_masks`,
    per frame. Never a model input.

Output dict (B samples, frames in the order [t, t−k, 0]):

| key | shape | dtype | note |
|---|---|---|---|
| `scene_tokens`, `scene_tokens_next` | [B, 3, 1200, 256] | fp16 | each frame = one buffer row (cameras already flattened) |
| `scene_pos`, `scene_pos_next` | [B, 3, 1200, 3] | fp32 | world metres |
| robot fields at t and t+1 | as `StepBuffer` | | frame t only |
| `dtheta` | [B, 18] | fp32 | commanded action at t |
| `h`, `h_next` | [B] | fp32 | `h_rules` of the chosen rule |
| `sentence` | list[str] | | for the text store |
| `rule_id`, `consequence`, `task`, `host`, `embodiment`, `source` | lists | | logging only, never model input |
| `fails_soon`, `fails_soon_recall`, `pair`, `episode_id` | [B] | | logging and pair metrics |
| `ref_mask` (probe batches) | [B, 3, 1200] | fp16 | measuring tool only |

Throughput:
- read rows t−k … t+1 as one contiguous slice per sample when they do not cross the array end;
- cache frame-0 tokens per (source, episode_id) in RAM;
- read each paired transition once;
- sample in a background thread (`PrefetchSampler`, queue depth 2);
- in rounds mode, stop and drain the thread before collecting, then refresh all sources and rebuild the
  index;
- log `perf/data_wait_frac`, and report it above 0.3.

Tests:
- rows across the ring seam and the physical array end;
- terminal rows excluded;
- truncated episodes excluded after a wrap;
- pairs always mixed;
- `fixed` mode never changes an episode's rule;
- no `heldout_*` sentence in 10k training samples;
- `next_prev_row` correct at t_in_ep 0, 1 and 2.

---

## 5. Model (`main/lang/network.py`)

### 5.1 Overview

| part | status | source |
|---|---|---|
| `RobotInjection` (link tokens, 2 payload tokens, `scene_adapter`) | reused unchanged | `main/network/encoder.py` |
| `GeoCrossAttention` | reused unchanged | `main/network/trunk.py` |
| `TokenActor` | reused unchanged | `main/network/heads.py` |
| `SafetyFilter` | reused unchanged | `main/train/filter.py` |
| action effects (`dp = Jlin·dθ`, `dω = Jang·dθ`, the 8-number effect `[own dθ, dp, dω, valid]`, zero rows for the 2 payload tokens, post-action positions) | copied into `action_effects()` with a parity test against `CriticHead` | `main/network/heads.py` |
| `TextAdapter`, `SceneSet`, scene-reads-words, `LangTrunkBlock`, `LangPolicyEncoder`, `LangCriticHead`, `LangTwinCritic` | new | `main/lang/network.py` |

Width 256, 8 heads, FFN 1024 with SiLU, RMSNorm, pre-norm residual sub-layers everywhere, no dropout,
`batch_first=True` everywhere. All trainable weights start from scratch. HoloBrain vision, the robot
state encoder, BERT and `text_feat_map` are frozen.

### 5.2 Inputs

| name | shape | meaning |
|---|---|---|
| `scene_tokens` | [B, 3, 1200, 256] | frozen tokens for frames [t, t−k, 0]. Within a frame, 1200 = 3 cameras × 400 (left wrist, right wrist, head), already flattened |
| `scene_pos` | [B, 3, 1200, 3] | world positions, each frame from its own depth |
| robot fields | as CrossSafe | frame t only |
| `words`, `word_mask` | [B, 24, d], [B, 24] | from the text store (d = 256 for `holobrain_bert`) |

### 5.3 Text adapter

Two custom pre-norm transformer blocks, **not** `nn.TransformerEncoderLayer`. Its defaults (post-norm,
dropout 0.1, ReLU, FFN 2048, `batch_first=False`) would mix samples across the batch.

```python
class TextAdapter(nn.Module):
    # learned: r (1x256), q (K x 256), word position embedding (24 x 256), all N(0, 0.02)
    # K = n_queries if lang_queries else 0 (no query parameters at all when queries are off)
    # in_proj: identity for holobrain_bert, Linear(d, 256) otherwise
    def forward(self, words, word_mask):                       # [B, L, d], [B, L]
        w = self.in_proj(words) + self.pos_emb[:L]
        x = cat([r.expand(B, -1, -1), q.expand(B, -1, -1), w], dim=1)    # [B, 1+K+L, 256]
        pad = cat([zeros(B, 1+K, bool), ~word_mask], dim=1)
        for blk in self.blocks:                                 # 2 blocks
            y = RMSNorm(x); x = x + MHA(y, y, y, key_padding_mask=pad)   # MHA(256, 8, batch_first=True)
            x = x + FFN(RMSNorm(x))                             # 256-1024-256, SiLU
        return x[:, 0], x[:, 1:1+K], x[:, 1+K:]                 # r0 [B,256], q [B,K,256], w' [B,L,256]
```

The words are updated too, and `w'` replaces the frozen words everywhere downstream. The adapter does
not depend on the observation, so at deployment it runs once per episode. Its outputs `text_out = (r0,
q, w', word_mask)` are reused every tick.

### 5.4 Scene set and scene-reads-words

```python
class SceneSet(nn.Module):
    # tau: learned [3, 256], N(0, 0.02), one per frame role (now, past, start)
    def forward(self, scene_tokens, scene_pos, injection):
        assert scene_tokens.shape[1:] == (3, 1200, 256)        # frames first, never cameras
        s = injection.scene_adapter(scene_tokens) + tau[None, :, None, :]   # [B, 3, 1200, 256]
        return s.flatten(1, 2), scene_pos.flatten(1, 2)                      # [B, 3600, 256], [B, 3600, 3]

# scene reads words, once                                    (switch: lang_scene)
s = s + MHA(RMSNorm(s), w', w', key_padding_mask=~word_mask)
# from here on the 3600 scene tokens are read-only
```

The encoder returns `[B, n_cam, 400, 256]` per frame. Every caller flattens cameras **within** a frame
first, then stacks frames. The assert catches a tensor whose second axis is cameras.

### 5.5 `LangTrunkBlock`

Updated tokens: body `b` [B, 22, 256] with positions `x` and `body_mask`, the rule token `r` [B, 1, 256],
queries `q` [B, K, 256]. Read-only: scene `s`, `p`, `scene_mask`, words `w'`, `word_mask`, and `r0`.

```python
# a. body reads scene, CrossSafe geometry (unchanged module)
b = b + self.geo(RMSNorm_a(b), x, s, p, scene_mask)
# b. body reads words                                      (switch: lang_body)
y = RMSNorm_b(b); b = b + MHA_bw(y, w', w', key_padding_mask=~word_mask)
# c. queries read scene, no geometric bias                 (switch: lang_queries)
y = RMSNorm_c(q); q = q + MHA_qs(y, s, s, key_padding_mask=~scene_mask)
# d. self-attention over [body, r, queries]
K = q.shape[1]                                             # 0 when queries are off
z = cat([b, r, q], 1); y = RMSNorm_d(z)
z = z + MHA_self(y, y, y, key_padding_mask=cat([~body_mask, zeros(B, 1+K, bool)], 1))
b, r, q = z[:, :22], z[:, 22:23], z[:, 23:]
# e. FiLM from r0, then FFN                                (switch: lang_film)
gamma, beta = self.film(r0).unsqueeze(1).chunk(2, dim=-1)  # one Linear(256, 512) per block, shared by b and q
b = b + FFN_b((1 + gamma) * RMSNorm_eb(b) + beta)
q = q + FFN_q((1 + gamma) * RMSNorm_eq(q) + beta)
if self.update_r:
    r = r + FFN_r(RMSNorm_er(r))                           # r gets a plain FFN
return b, r, q
```

- `FFN_b`, `FFN_q`, `FFN_r` are separate (256-1024-256, SiLU).
- The FiLM linear is zero-initialized by default (`film_zero_init`), so FiLM starts as the identity. A
  stability choice, not a gate.
- `update_r=False` in the **last critic block**, whose `r` output is never used.
- With `lang_queries` off, skip sub-layer c and `FFN_q` entirely. With `lang_body` off, skip sub-layer
  b. With `lang_film` off, `self.film` does not exist and `gamma = beta = 0`.
- Padded body queries attend normally, and their outputs are discarded downstream (CrossSafe).
- `need_weights=False` everywhere except in probe mode (Section 8.2).

### 5.6 `LangPolicyEncoder` and `LangEncoded`

```python
@dataclass
class LangEncoded:
    body: Tensor; pos: Tensor; mask: Tensor                  # [B, 22, 256], [B, 22, 3], [B, 22]
    scene: Tensor; scene_pos: Tensor; scene_mask: Tensor     # [B, 3600, 256], [B, 3600, 3], [B, 3600]
    words: Tensor; word_mask: Tensor                         # [B, L, 256], [B, L]
    r0: Tensor; r: Tensor; queries: Tensor                   # [B, 256], [B, 1, 256], [B, K, 256]
    def detach(self) -> "LangEncoded": ...                   # detaches every tensor field

class LangPolicyEncoder(nn.Module):
    def forward(self, state_tokens, body_tokens, link_pos, body_mask, joint_index,
                scene_tokens, scene_pos, words=None, word_mask=None, text_out=None) -> LangEncoded:
        b, x, m = self.injection(state_tokens, body_tokens, link_pos, body_mask, joint_index)
        s, p = self.scene_set(scene_tokens, scene_pos, self.injection)
        if text_out is None:
            text_out = (*self.text_adapter(words, word_mask), word_mask)
        r0, q, w, wm = text_out
        if cfg.lang_scene: s = self.scene_reads_words(s, w, wm)
        r = r0[:, None] if cfg.lang_rtoken else self.r_static.expand(B, 1, -1)   # r_static exists only when lang_rtoken is off
        for blk in self.blocks:                                                    # 3 LangTrunkBlocks
            b, r, q = blk(b, x, m, r, q, s, p, scene_mask, w, wm, r0)
        return LangEncoded(b, x, m, s, p, scene_mask, w, wm, r0, r, q)
```

`SafetyFilter` only touches `enc.body` and calls `critics.qmin(enc, a, Jlin, Jang, joint_index)`, so
`LangTwinCritic` keeps that signature and `SafetyFilter` is reused unchanged.

### 5.7 Actor

`TokenActor` unchanged, on `enc.body` (from a detached `LangEncoded` in the actor update).

### 5.8 `LangCriticHead` and readout

```python
eff, dp = action_effects(dtheta, Jlin, Jang, joint_index)   # [B, 22, 8], [B, 22, 3]
b = enc.body + self.act_proj(eff)                            # Linear(8, 256)
x = enc.pos + dp                                             # post-action geometry
r, q = enc.r, enc.queries
for blk in self.blocks:                                      # 2 LangTrunkBlocks, own weights, last has update_r=False
    b, r, q = blk(b, x, enc.mask, r, q, enc.scene, enc.scene_pos, enc.scene_mask,
                  enc.words, enc.word_mask, enc.r0)
V_body = self.value_head(b).squeeze(-1)                      # [B, 22], MLP 256-256-1, SiLU
V_q    = self.value_head_q(q).squeeze(-1)                    # [B, K]; value_head_q does not exist when queries are off
V = cat([V_body.masked_fill(~enc.mask, 1e4), V_q], -1)       # padding must lose the min
Q = -T * logsumexp(-V / T, dim=-1)                           # T = cfg.softmin_T_h = 0.05 (h units)
return Q, V
```

`T` is in h units and is **not** multiplied by `h_scale`. CrossSafe's `softmin_T` is in metres and gets
scaled, but h here is unitless in [−1, 1]. `LangTwinCritic` exposes `forward -> (q1, q2, v1, v2)` and
`qmin`, like `TwinCritic`.

### 5.9 Switches

| flag (`LangConfig`) | default | effect when off |
|---|---|---|
| `lang_scene` | on | no scene-reads-words layer |
| `lang_body` | on | sub-layer b removed in all trunk and critic blocks |
| `lang_film` | on | plain FFNs, no FiLM parameters |
| `lang_queries` | on | no queries: no query parameters, sub-layer c removed, readout over the 22 body values only |
| `lang_rtoken` | on | `r` is a learned sentence-independent token (`r_static`). `r0` still drives FiLM if `lang_film` is on |
| `shuffle_text` | off | sampler control (Section 4.4) |
| `relabel` | `paired` | `random` or `fixed` (Section 4.4) |
| `text_backend` | `holobrain_bert` | `bert_raw` or `sentence_generic` |

With all five path switches off, the sentence cannot affect Q (required test).

### 5.10 Tests (`tests/lang/test_network.py`)

- shapes for B = 2 with both embodiments' padding (UR5 14 state tokens, Franka 16);
- padding: changing padded body tokens, padded words, or masked scene tokens does not change Q;
- batch permutation: permuting the batch permutes every output;
- all five path switches off: Q identical for two sentences. Defaults with random init: Q differs;
- each switch off alone: its parameters do not exist and its path is gone;
- the critic loss gives non-zero gradients to every existing new parameter, and zero gradients to
  frozen modules;
- `action_effects` matches `CriticHead` on random inputs (max abs difference 1e-6);
- softmin with all n valid values equal to v returns `v − T·log(n)`;
- report parameter counts per module group and peak GPU memory of one training step at B = 64 (A40,
  48 GB) in `docs/lang_filter/phase2.md`.
  - The geometric attention builds a [B, 22, 3600] pair grid per call, 7 calls in the critic-loss graph,
    about 3× CrossSafe.
  - If memory does not fit, wrap trunk and critic blocks in `torch.utils.checkpoint` (a wrapper, no edit
    to `trunk.py`).
  - Do not change B or precision without human approval.

---

## 6. Training (`main/lang/train.py`)

### 6.1 Update (one grad step)

```python
b = sampler.next()                                           # Section 4.4
words, wmask = text_store.lookup(b["sentence"])
enc = policy_enc(..., b["scene_tokens"], b["scene_pos"], words, wmask)
with no_grad():
    enc_n = policy_enc_targ(..._next, words, wmask)          # same sentence at t+1
    a_n, _, _, _ = actor(enc_n.body, ji_n, dtheta_max_n)
    q1_t, q2_t, _, _ = critics_targ(enc_n, a_n, Jlin_n, Jang_n, ji_n)
    y = (1 - gamma) * h + gamma * min(h, 0.5 * (q1_t + q2_t))   # gamma = 0.9, mean of target twins
q1, q2, _, _ = critics(enc, dtheta, Jlin, Jang, ji)
loss_q = mse(q1, y) + mse(q2, y)                             # trains critics + the whole policy encoder
# actor step: encoder and critic params requires_grad False for the step, every field detached
enc_d = enc.detach()
a_pi, logp, _, pre = actor(enc_d.body, ji, dtheta_max)
loss_pi = (-min(critics(enc_d, a_pi)) + alpha * logp + 1e-3 * mean_valid(pre**2)).mean()
polyak(targets, live, tau=0.005)
```

- **Encoder parameters** (`opt_e`, trained by `loss_q`): `injection`, `scene_set` (τ), `text_adapter`,
  `r_static` (only when `lang_rtoken` is off), scene-reads-words, the 3 trunk blocks.
- **Critic parameters** (`opt_c`): both critic heads. **Actor** (`opt_a`): `TokenActor`.
- Adam, lr 3e-4. Clip gradient norms at 10 per group and log each group's norm.
- **Targets**: Polyak copies of the whole `LangPolicyEncoder` and of both critic heads. A test checks
  that every trainable encoder and critic parameter has a target copy.
- **The same sentence** is used at t and t+1, and for `a'`.
- Schedules as `FrozenConfig`: γ fixed 0.9, α 0.2 → 0.02 over 20k steps, hard Bellman target.

### 6.2 Modes and commands (always through `scripts/lang/in_container.sh`)

**Offline** (warmup shards read in place, no simulation). The fast loop for the language questions and
for debugging:

```
python -m main.lang.train offline --data data/lang/warmup --val data/lang/val --run runs/lang/<name> \
    --grad-steps 50000 --probe-every 5000
```

**Rounds** (as CrossSafe: round 0 trains on the warmup, each later round collects with the current
weights, 80% filter episodes and 20% nominal only, then trains):

```
python -m main.lang.train rounds --data data/lang/warmup --val data/lang/val --run runs/lang/<name> \
    --collect-rounds 100 --episodes-per-round 4 --grad-steps-per-round 2048 --probe-every-rounds 10
```

- Rounds mode creates a fresh ring pair in `runs/lang/<name>/ring/`: `StepBuffer` with the LangConfig
  header, then `ItemSidecarBuffer.for_step_buffer(buf, n_tok=masks.token_grid()[1])`. Capacity =
  rounds × episodes per round × `max_steps_per_episode` × 1.1.
- The sampler's sources are the warmup shards plus the ring. Rounds runs always start from scratch,
  never from an offline checkpoint.
- Common flags: `--tasks all|<task,...>`, `--embodiments` (default `ur5-wsg`: training and rounds
  collection use one embodiment; `all` for both; a resumed run keeps the choice in its `lang_config.json`),
  the switches of Section 5.9, `--no-wandb`. The evaluators score the run's trained embodiments by
  default (`--embodiments trained`; `all` adds the other one as a cross-embodiment test).
- `--resume <run_dir>` restores everything: weights, targets, optimizers, step, round index, schedule
  cursors, sampler and env RNG states, and the ring position (Section 7.4).
- `--init-weights <ckpt>` (offline mode only) loads weights only and re-syncs targets.
- Run names encode variant and mode, for example `lang_full_paired_off_1007-1530`.

### 6.3 Run directory and checkpoints

```
runs/lang/<name>/
  lang_config.json  checkpoint.pt  checkpoint_round<N>.pt (every 10 rounds)  checkpoint_final.pt
  state.pt          (resume state)
  text_cache_extra.pt
  ring/             (rounds mode)
  train_log.jsonl   (every 20 steps, Section 8.1)
  probes/  eval/  eval_offline/  eval_closed/
```

- A checkpoint holds `injection`, `scene_set`, `text_adapter`, `r_static` (when present),
  `scene_reads_words` (when present), `trunk`, `actor`, `critics`, plus the `LangConfig` dict, the
  paraphrase-bank hash (of `envs/main/item_envs/language/paraphrases.yaml` and `configs/splits.yaml`),
  and the text backend name.
- Loading checks that the switches match. Publish atomically (tmp file, then `os.replace`).

### 6.4 Sanity checks before any long run (Phase 3 gate)

1. **Overfit.** Fix one batch of 64 built with `pair_frac = 1` from mixed rows (32 pairs). Train 1000
   steps on it.
   - Measure **target-rank accuracy**: over pairs with |y₁ − y₂| > 0.05, the fraction where
     sign(Q₁ − Q₂) = sign(y₁ − y₂), with y the critic target.
   - Required: ≥ 0.95, and `loss_q` falls by at least 10×.
2. **Shuffled control.** The same batch with `shuffle_text` redrawn every step. Target-rank accuracy
   must stay ≤ 0.6. Higher means labels leak into inputs: stop and find the leak.
3. **Switches off.** With all five path switches off, `lang/text_sens` (Section 8.2) is exactly 0.

---

## 7. Collection (`main/lang/collect.py`)

### 7.1 One episode loop for everything

`run_episodes(cfg, out_root, plan, models=None, ...)` collects warmup, validation and round episodes
alike. It mirrors `main.train.collect.collect()`, the verified reference for item tasks, and differs
where Section 3.3 item 8 says it must.

Per episode:

1. **Slot**: the next (task, embodiment) from a schedule with a persistent cursor (Sections 7.2 and
   7.3). Seed and `episode_id` from Section 4.2. For T1, set `cfg.item_host` explicitly from a
   per-embodiment round-robin cursor over the 5 hosts.
2. **Build**: `cfg_ep = dataclasses.replace(cfg, task=..., embodiment=..., item_host=...,
   item_active_rule=None)`, then `ro = LangRolloutController(cfg_ep, seed, mode="collect", encoder=...,
   kin=..., <models>, filter_active=..., buf=<StepBuffer>, episode_id=..., perturb_prob=...,
   rng=np.random.RandomState(seed), max_steps=10 * max_steps_per_episode,
   controller_cls=RiskTickController)`. The controller builds the env. Filter episodes pass the live
   weights in eval mode and run under `torch.no_grad()`. Nominal episodes pass no models.
   - If the build raises, or `ro.env.obstacle is None` (no rules, close the env), **increment
     `attempt`** (new seed) and retry, at most 5 times. Then skip the slot and count it.
3. **Scene roles** (every episode): `runner = ro.env._item_runner`,
   `record = language.episode_record(runner, episode_id)`, `roles = in_way(ro.env, record)`, and every
   rule's designation (Section 3.5).
4. **Named rule and sentence** (filter episodes only. Nominal episodes keep `active_rule_id = None`, so
   their StepBuffer h is the min over all rules):
   - `rng_ep = np.random.default_rng(seed)` for this step. `sample_sentence` calls `rng.integers`, so
     it needs a Generator, not the controller's RandomState;
   - pick a **sampleable** harm-designated rule with probability `named_harm_frac` = 0.5, else a
     sampleable benign-designated one. If the wanted kind does not exist, take the other. If neither
     exists, set `ro.filter_active = False` (the base reads it every tick) and run the episode nominal
     only;
   - set `runner.active_rule_id = rule["rule_id"]`;
   - `sentence = language.sample_sentence(rule, record, rng_ep, "train")`, then
     `ro.set_instruction(sentence)` (TextStore lookup, then the text adapter, no grad).
5. **Wire**: `wire_item_episode(ro, sidecar)`. Because `runner.active_rule_id` is already set, the h_fn,
   the sidecar's `active_rule` column and `episodes/{id}.json` all name the same rule.
6. **Run**: `trace = ro.run()` (it closes the env), then `ro.buf.end_episode()` (the lockstep teardown
   check), then flush buffer and sidecar. If `end_episode()` raises, stop the process. That is a lockstep
   bug, and every label after it would be wrong.
7. **Record**: append one line to `<out_root>/lang_episodes.jsonl` with:
   - `episode_id`, task, host, embodiment, seed;
   - `filter`, `named_rule_id`, its `designation`, every rule's designation, the in-the-way and unclear
     ids;
   - `primary` (`runner._primary_rule_id(ro.env.task)`), the task's `referent_kind`, T3's `scenario`;
   - `sentence`, `status`, `success`, `n_ticks`;
   - per-rule `violated`.

   The environment's records do not store sentences or outcomes.
8. **Init limit**: count environment inits. The loop's caller decides what happens at 200 (Sections
   7.2, 7.4).

Videos use `LangPanelVideoWriter` (h units, not CrossSafe's centimetre panel). Create it after the
controller and assign `ro.video`, as `collect()` does.

### 7.2 Warmup and validation (nominal only, collected once, reused by every run)

```
# one process per (embodiment, shard), g = emb_idx * n_shards + shard
python -m main.lang.collect warmup --out data/lang/warmup/shard_g<g> --embodiment ur5-wsg \
    --shard <i> --n-shards <n> --episodes-per-task 30 [--video-every 10]
python envs/scripts/item_envs/build_text_cache.py --episodes data/lang/warmup/shard_g*/item_buffer/episodes \
    --out data/lang/warmup/text_cache.pt --device cuda
python -m main.lang.collect report data/lang/warmup
```

- **Schedule**: `n_shards` must divide `episodes_per_task`. Each shard then runs `cycles =
  episodes_per_task / n_shards` whole cycles over the 7 tasks (order within a cycle shuffled with seed
  g), so every shard and the union are exactly balanced. T1's host in cycle c of shard i is
  `t1_hosts[(i · cycles + c) mod 5]`, balanced over the union when 5 divides `episodes_per_task` (30
  does).
- **Fresh directory only**: the driver refuses a non-empty `--out`. Shards are sized to stay under 200
  environment inits including retries. A shard that reaches 200 inits aborts with an error. Delete it
  and collect again with more shards. A shard that crashes is deleted and collected again (episode ids
  in a reused directory would collide, and `write_episode` would overwrite `episodes/{id}.json` with a
  different scene).
- **Pair creation**: `StepBuffer(<out>/buffer, capacity, header=LangConfig.to_dict())`, then
  `ItemSidecarBuffer.for_step_buffer(buf, n_tok=masks.token_grid()[1])`. Capacity = the shard's episodes ×
  `max_steps_per_episode` × 1.1, so a shard never wraps.
- **Not success-only.** In some tasks "success" correlates with harm (a pour that spills may also count
  as failed), and dropping those episodes would remove exactly the unsafe transitions training needs.
  Every episode is kept, with its `status`.
- **Size**: 30 episodes per (task, embodiment) = 420 episodes. Phase 0's wall time per episode sets the
  number of shards. Report total disk use (about 0.63 MB per step plus the sidecar).
- **Validation**: the same command with `validation`, `--out data/lang/val/shard_g<g>`,
  `--episodes-per-task 5`, `episode_offset = (60+g)·1e6`, and its own cache `data/lang/val/text_cache.pt`.
  Used only by the offline probe and the training probes.

**`collect_report.md`** (and `.json`, copied to `docs/lang_filter/`), the input to the Phase 1 STOP gate:
1. episodes per combo (and per T1 host), skipped slots, `status` counts, success rate, episode length per
   task (mean, p95, max vs the cap);
2. per task and consequence:
   - fraction of episodes in which a rule is violated, split by harm- and benign-designated;
   - fraction of rows with `fails_soon` set, and fraction of mixed rows;
   - sampleable rules per episode.

   Flag any task where harmful labels are below 2% or above 80% of its sampleable rule-rows;
3. per consequence: h quantiles at t = 0 and over all ticks;
4. **delay audit**:
   - Source: 20 standalone probe episodes per task (from `envs/`: `python -m main.item_envs.probe
     episode ... --render none`), whose `trace.npz` has `time_scale` and margins per tick.
   - Measure the time from the start of the risky segment containing a rule's first violation to that
     violation.
   - Flag it if the median exceeds 0.4 s (10 ticks, roughly the horizon of γ = 0.9). γ may need 0.95;
5. violation rates among non-`ok` episodes (timeouts can correlate with harm too);
6. text cache: sentences per split, 20 random (rule id, sentence) examples per task;
7. wall time per episode, rendering + encoding vs physics.

### 7.3 Round collection (inside `train.py rounds`)

- `episodes_per_round` (default 4) per round from a global (task, embodiment) schedule whose cursor
  persists across rounds, with a T1 host cursor per embodiment. Each is a filter episode with
  probability `filter_episode_frac` = 0.8, else nominal only.
- Seed and id: `(100 + round)·1e6 + attempt`.
- Episodes go into the run's ring (`runs/lang/<name>/ring`), keeping every episode. The default ring
  never wraps (Section 6.2).
- Before collecting, stop and drain the prefetch thread. After collecting, refresh all sources and
  rebuild the index.
- Filter episodes record videos and round metrics (Sections 8.3, 8.4).

### 7.4 Resume and the init limit (rounds)

Changed 2026-10-07: a round (2048 steps plus 4 episodes, about 85 min on an A40) is longer than the
1-hour jobs, so the run also stops and resumes **inside** a round.

- `state.pt` holds weights, targets, optimizers, step, α, round index, schedule cursors, RNG states, the
  ring's `n` and `head`, and where in the round the run is:
  - `phase = "collect"`: the round's slots, how many are done, the next attempt counter. Written at the
    round start and after **every** collected episode;
  - `phase = "train"`: the round's step target. Written when collection ends, every 500 steps, and when
    the run stops.
- **Clean stop** (state saved, exit code 75): at the time budget (`--time-budget` or `$STAGE_DEADLINE`;
  no episode starts that would not finish in time), on SIGTERM / SIGUSR1, and at the init limit (when
  inits so far + 5 × `episodes_per_round` > 200 before a round, or 200 inside one).
  `scripts/lang/run_rounds.sh` and `scripts/lang/bridges/rounds.sbatch` relaunch with `--resume`.
- **Resume**, also after a hard kill: load `state.pt`, `discard_to(n)` on the StepBuffer and the sidecar
  (valid because the ring has not wrapped), drop the lines of later episodes from `lang_episodes.jsonl`
  and `eval/rounds.jsonl`, add the ring to the sampler (after the roll-back), and continue in the saved
  phase with the remaining slots and their seeds. A hard kill loses at most the episode in progress or
  500 steps. Re-collection overwrites those ids' `episodes/{id}.json`, which is consistent because their
  old rows were discarded.

### 7.5 `LangRolloutController(RolloutController)` (`main/lang/rollout.py`)

The base class owns capture, buffer writes, the trace and the video. Change it as little as possible by
injecting small objects, rather than copying its long methods.

- **`run()` override.**
  - Call `self._capture_tick()` before `super().run()` **only when `self.mode != "collect"`** (collect
    mode already captures s₀). Without it, frame 0 at deployment would be s₁ while training uses s₀.
  - Set `trace["status"]` ∈ {`ok`, `timeout`, `exception`} plus the exception text by wrapping
    `env.task.play_once`. The base catches these and only prints them.
- **`set_instruction(sentence, cf_sentence=None)`**: TextStore lookup and text adapter (no grad), stored
  as `self.text_out` (and `self.cf_text_out`). Called after construction, because the rule table only
  exists once the env is built.
- **`wire_item_eval(ro, rule_id)`** for evaluation (no buffer):
  - `ro.h_fn = ItemMarginHFn(rule_id, 1/h_scale, ctx=runner, row_sink=None, with_masks=False)` (masks
    render segmentation from 3 cameras every tick, and evaluation does not read them);
  - set `ro.ctrl.risky_skill_labels` from `runner.task_cfg` (as `wire_item_episode` does).
- **`LangPolicyShim`** (assigned to `self.policy_enc`) has CrossSafe's 7-argument call signature
  `(state, body_tokens, link_pos, body_mask, joint_index, tokens, pos_mean)`. It:
  - flattens cameras per frame (`[1, 3, 400, 256] → [1, 1200, 256]`);
  - **pushes the frame into the history only during a capture**. A thin `_capture_tick` override sets
    `self._in_capture = True` around `super()._capture_tick()`. The first pushed frame is frame 0, and a
    deque holds the last k+1 captured frames;
  - assembles [t, t−k, 0], with t−k = frame 0 until k+1 frames exist (matches the sampler's
    `max(t − k, 0)`);
  - calls `LangPolicyEncoder(..., text_out=self.text_out)` and returns the `LangEncoded`;
  - with a counterfactual sentence, also computes and stores `enc_cf` with `cf_text_out`.

  On the base cache-miss path (`_policy_inputs_cached → _model_inputs`), the shim uses the fresh frame as
  t without pushing it, so the history never gets extra frames.
- **`_tick_hook` override** (only with a counterfactual sentence): compute `a = self._anom(ctx)` and
  `Q_cf = critics.qmin(shim.enc_cf, a, Jlin, Jang, ji)`, append it to the trace, then call `super()`.
- **Trace** per tick, from `self._last_diag` (set by the base class at each capture):
  - `t`;
  - `h_all` (`rule_h(rule, m)` for every rule of `runner.rules` and `diag["margins"]`);
  - `h_named`;
  - `Q` (base trace `V`, at hook times), `Q_cf`;
  - `intervened`, `action_source`;
  - the named item's `robot_item_dist`;
  - `held`.
- **Contact**: the base `obstacle_contact` keys on CrossSafe's obstacle name and returns 0 here. Report
  contact from `robot_item_dist ≤ 0` of the named rule's subject, for `avoid`, `upright` and `stay`
  rules only. Otherwise report NaN: a zone rule's subject is the pushed box, which the hand touches the
  whole time.

---

## 8. Logging

### 8.1 Training scalars (every 20 steps, wandb and `train_log.jsonl`)

| key | definition |
|---|---|
| `loss/q`, `loss/pi`, `alpha`, `gamma` | as CrossSafe |
| `q/mean`, `h/mean`, `h/min`, `gap/mean` | batch means. gap = h − min(Q1, Q2) |
| `q/precision`, `q/recall` | predicted unsafe = Q < 0, true unsafe = `h_next < 0` |
| `q/precision_soon`, `q/recall_soon` | true unsafe = `fails_soon_recall` (7 ticks). With γ = 0.9 the ideal V crosses 0 only about 7 ticks before an abrupt violation, so recall against the 25-tick label would be capped near 7/25 |
| `lang/pair_rank_acc` | over in-batch pairs: fraction where Q(harmful rule) < Q(benign rule) |
| `lang/pair_target_rank_acc` | over in-batch pairs with \|y₁ − y₂\| > 0.05: fraction where sign(Q₁ − Q₂) = sign(y₁ − y₂) |
| `lang/pair_q_gap` | mean Q(benign) − Q(harmful) over pairs |
| `by_task/<task>/{q,h,gap,rank_acc}`, `by_cons/<consequence>/{q,h,gap}` | splits (logging only) |
| `grad/{encoder,text_adapter,critic,actor}` | gradient norms per group, before clipping |
| `actor/action_std`, `actor/sat_frac`, `data/action_std`, `entropy`, `critic/action_sens_rel` | as CrossSafe |
| `data/harm_frac`, `data/pair_mixed_frac`, `data/text_miss_frac` | sampling health |
| `perf/steps_per_sec`, `perf/data_wait_frac`, `perf/gpu_mem_gb` | throughput |

### 8.2 Probes (every `probe_every` steps, on a fixed probe set of 512 validation samples with `ref_mask`)

| key | definition |
|---|---|
| `lang/text_sens` | mean \|Q(s, a, sentence) − Q(s, a, shuffled sentence)\| / mean \|Q\|, the shuffled sentence from a different episode |
| `lang/para_std` | std of Q over 3 training paraphrases of the same rule |
| `lang/para_std_ratio` | `para_std` / std of Q over the different sampleable rules at the same state (multi-rule tasks only) |
| `attn/body_ref_enrich` | attention mass of body-to-scene attention (last trunk block, averaged over heads and valid links) on the named referent's tokens, divided by those tokens' share of all tokens |
| `attn/query_ref_enrich` | same for the queries' scene attention |
| `attn/scene_delta_ratio` | mean ‖Δs_j‖ added by scene-reads-words on named-referent tokens / mean on all other tokens |
| `attn/word_entropy` | entropy of body-to-word attention |
| `film/gamma_rms_<blk>`, `film/beta_rms_<blk>` | FiLM modulation size per block |

The ground-truth token masks are only a measuring tool. Enrichment above 1 means attention concentrates
on the named referent: direct evidence of whether grounding emerges from the HJ loss alone.

`GeoCrossAttention` returns no attention weights, and `trunk.py` must not be edited. In probe mode,
recompute them from the module's own `q_proj`, `k_proj` and `bias_mlp` (same logits and masking as its
`forward`). Test that these weights times its values, with the weighted directions, through `out_proj`,
reproduce its output.

Also save `probes/step<N>_examples.md`: for 8 fixed validation states, every sampleable rule's sentence
with its Q, h and true `fails_soon`.

### 8.3 Round metrics (filter episodes of each round)

`LangRoundMetrics` records per filter episode, keyed by (round, task, host, embodiment, consequence,
`designation`):
- `status`, `task_success`;
- `harm_named` (named h < 0 at any tick), `harm_named_frac`, `harm_other_any`;
- `intervention_rate`, `any_intervention`, `mode_switches`;
- `v_le_h_frac` (Q ≤ h_named + 0.05), `hq_corr`;
- `first_harm_t`, `first_intervene_t`, `realized_cmd_ratio`.

Heatmaps: tasks × (embodiment × designation) columns, one image per metric plus a trend plot over
rounds (CrossSafe layout). wandb keys `round_overview/*`, `heatmap/*`, `trend/*`. `harm_named` on
harm-designated episodes next to `any_intervention` on benign-designated ones is the round-level view of
selectivity.

### 8.4 Videos

`LangPanelVideoWriter(PanelVideoWriter)` keeps a reference to the controller and draws from its current
state, ignoring the base class's centimetre `h` argument. Observer frame plus a panel with:
- the instruction (at most 2 lines) and the named rule id;
- `h_named` and Q, plus Q_cf when present;
- FILTER or NOMINAL, |a| ratio, intervened ticks, block counter;
- the most at-risk **unnamed** rule and its h;
- a scrolling plot of `h_named`, Q and the unnamed minimum;
- a red frame border on ticks where `h_named < 0`.

In nominal episodes (warmup, validation, nominal round episodes) the panel shows the minimum h over all
rules and the rule attaining it, with no sentence and no Q.

File names: `<task>[_<host>]_<embodiment>_s<seed>_<kind>[_<axis>]_<outcome>.mp4`, with kind `nominal`,
`harm` or `benign` (the named rule's designation), the axis in evaluation only, and outcome `ok`,
`harmed` (the named rule violated, or any rule in a nominal episode) or `fail` (task failure without
harm). wandb: one carousel per round, filtered by mtime.

### 8.5 Console

Every 20 steps: step, losses, q, h, gap, `lang/pair_rank_acc`, steps/s, data wait. Every collected
episode: task, host, embodiment, seed, FILTER or nominal, named rule and its designation, sentence,
status, success, `harm_named`, interventions.

---

## 9. Evaluation

### 9.1 Offline language probe (`main/lang/eval_offline.py`, built in Phase 3)

```
python -m main.lang.eval_offline --run runs/lang/<name> --ckpt checkpoint.pt --val data/lang/val \
    --states-per-task 2000 --paraphrases 3
```

Uses only the validation set. States: every 5th tick of every episode (terminal rows excluded), at most
2000 per task. For each state and rule, compute Q(s, a_logged | sentence) per **axis**:

| axis | sentences | eligible rules |
|---|---|---|
| `train` | `sentences(rule, rec, "train")` | sampleable rules |
| `heldout_template` | `sentences(rule, rec, "heldout_template")` | env split `train`, not set or ambiguous |
| `heldout_word` | `sentences(rule, rec, "heldout_word")` | env split `train`, not set or ambiguous, with a held-out noun |
| `heldout_item` | training templates | env split `heldout_item` |
| `shuffled` | a training sentence of a sampleable rule of a different episode | sampleable rules (control) |

`eval_offline.py` checks the eligibility column itself, from the record's `split`, `set` and
`ambiguous` fields. `RelabelSampler(split="heldout_template")` does not check the env split, so it is
not used here.

**Pairs**: two rules at the same state whose `fails_soon` bits differ, treated alike:
- `train`, `heldout_template`: both members get the axis sentence;
- `shuffled`: each member gets its own sentence from a different episode, so chance is exactly 0.5;
- `heldout_word`, `heldout_item`, **symmetric pairs**: both members are axis-eligible and get the axis
  sentence. Report the number of pairs, as they can be rare;
- `heldout_word`, `heldout_item`, **mixed pairs** (only r₁ eligible): r₁ gets the axis sentence, r₂ its
  `train` sentence. Also compute `rank_acc_base` on the same pairs with r₁'s sentence replaced by
  another episode's sentence. Report `rank_acc − rank_acc_base`, because r₂'s sentence alone already
  orders many mixed pairs.

Metrics, per axis, task and consequence, and pooled:
- `rank_acc`: fraction of pairs where Q(harmful) < Q(benign). Chance 0.5 for symmetric pairs. **The
  headline language metric.**
- `rank_acc_referent` and `rank_acc_consequence`: pairs with the same consequence on different items (a
  referent decision), or different consequences on the same item (a consequence decision, for example
  T7's `upright` vs `stay` of one block). This is how Q2 separates the two.
- `auroc` of −Q for `fails_soon`, per (state, rule), with each eligible rule given the axis sentence.
  Plus `auroc_base` with every sentence replaced by one from a different episode.
- `prec_at_0`, `rec_at_0`: Q < 0 against `fails_soon_recall`.
- `v_le_h`: P[Q ≤ h + 0.05]. `para_std`, `para_std_ratio` (Section 8.2, 3 paraphrases).
- the enrichment diagnostics of Section 8.2.

Outputs in `eval_offline/<ckpt>/`: `metrics.json`, `tables.md` (axis × task), `rank_acc_bars.png`,
`q_pairs_scatter.png`, `examples.md`. It runs every 10 rounds in rounds training, every `probe_every`
steps in offline training, and at the end.

### 9.2 Closed-loop evaluation grid (`main/lang/eval_closed.py`)

Same structure as `main/train/eval_ckpt.py`: deterministic seeds, pair-level resume from shard jsonl, a
merging parent. Children run one embodiment each, exit after 200 inits, and the parent restarts them
(3 crashes on one item → error row).

```
python -m main.lang.eval_closed --run runs/lang/<name> --ckpt checkpoint_final.pt \
    --scenes-per-combo 20 --axes train,heldout_template [--axes ...,heldout_word,heldout_item] [--video-every 1]
```

**Why the design looks like this**: cuRobo makes the nominal motion differ between runs of the same
seed. So "would the named item have been harmed" cannot be read from a separate nominal run. Instead,
the named rule follows the **in-the-way designation** (Section 3.5), which comes from the scene layout
and is deterministic per seed:
- `harm` runs name a harm-designated rule;
- `benign` runs name a benign-designated rule;

and baseline rates come from repeated nominal runs on the same seeds. The environment's `primary` rule
is only a label (Section 3.3 item 7).

**Protocol per scene** (task, embodiment, scene index, seed from Section 4.2). T1 cycles hosts with
`host_idx = scene_idx mod 5`, passed explicitly as `item_host = t1_hosts[host_idx]`. Left as None, the
runner would pick `host_tasks[seed % 5]`, which disagrees with the seed formula.
1. **Nominal runs.** 2 runs without a filter, `set_global_seeds(seed)` before each. Record every rule's
   outcome, every rule's designation, and the seed actually used. Cache under
   `runs/lang/_nominal_cache/<key>/` (key = task, host, embodiment, seed, `max_steps_per_episode`, env
   code hash, bank hash), shared by every checkpoint. Redraw a failed scene build at most 7 times (Section 4.2).
2. **Named rules per axis**: `harm` = a harm-designated rule eligible for the axis (Section 9.1 table),
   `benign` = a benign-designated eligible rule, each chosen by a hash of (seed, axis). Each runs
   independently, and a missing one is counted (T2 and T4 have no benign rule).
3. **Filter runs**: for each axis and available named rule, one sentence (chosen by a hash of seed and
   axis), one filtered episode on the nominal runs' used seed. The rule table and the designations must
   match the nominal runs' (else an error row). When both harm and benign exist, record `Q_cf` with the
   other named rule's sentence.

**Per-run metrics**:
- `status`, `harm_named`, `harm_named_frac`, `task_success`, `intervention_rate`, `any_intervention`;
- `lead_time` (harm runs): first violation time of the named rule in the nominal runs (mean over the
  nominal runs that violate it) minus the filter run's first intervention time. Positive = the filter
  acted before the harm would have happened;
- `harm_other_any`, `v_le_h_frac`, `hq_corr`;
- `q_cf_gap` (mean Q_cf − Q before the first intervention).

**Aggregates** (per task, embodiment, axis, pooled):

| metric | definition | good |
|---|---|---|
| `harm_rate_nominal` | P[named rule violated] over the nominal runs, for harm-designated named rules | context (how often "in the way" means harmed) |
| `benign_rate_nominal` | the same for benign-designated named rules | near 0 (checks the designation) |
| `harm_rate_filter` | P[`harm_named`] over harm runs | low |
| `protect_rate` | 1 − `harm_rate_filter` / `harm_rate_nominal` | high |
| `benign_intervention` | mean `intervention_rate` over benign runs | low |
| `benign_false_alarm` | P[`any_intervention`] over benign runs | low |
| `success_cost` | P[success \| nominal] − P[success \| filter], for harm and benign runs | low |
| `selectivity` | `protect_rate` − `benign_false_alarm` | high |

- Report the nominal-vs-nominal disagreement rate of the named rules' outcomes (nondeterminism) next
  to every table.
- Rates come with Wilson 95% intervals. Checkpoint differences are paired over identical scenes, with a
  bootstrap over scenes.

Outputs in `eval_closed/<ckpt>/`: `episodes[_emb-<e>].jsonl`, `summary.json`, `tables.md`, `videos/`,
`figures/`.

Cost: 20 scenes per (task, embodiment) and axes `train,heldout_template` give up to 4 filtered runs per
scene, up to 1120 filtered episodes per checkpoint, plus 560 nominal runs shared by all checkpoints.
Report wall time per episode in Phase 5.

### 9.3 Experiment matrix (scripts only. The human decides what to launch)

`scripts/lang/matrix.sh` prints (and with `--submit` launches) these runs on the same warmup and
validation data:

| run | flags | answers |
|---|---|---|
| `full` | defaults | reference |
| `fixedrule` | `--relabel fixed` | Q1 |
| `randrelabel` | `--relabel random` | Q1 (value of pairing) |
| `shuffled` | `--shuffle-text` | Q1 control, lower bound |
| `noscene`, `nobody`, `nofilm`, `noqueries`, `nortoken` | one path switch off each | Q2 |
| `bertraw`, `generic` | `--text-backend bert_raw` / `sentence_generic` | Q3 |
| `pertask_<task>` | `--tasks <task>` | per-task vs co-training |

- Every run trains **offline** first and gets the offline probe.
- Closed-loop grids compare runs trained the same way.
- Rounds training is for `full` and the variants the human picks, compared among rounds runs only.

`python -m main.lang.report --runs runs/lang/* --out docs/lang_filter/results/` builds one table per
question and the figures.

---

## 10. Phases

| phase | agent time | compute | STOP gate |
|---|---|---|---|
| 0 Recon (Section 3.4) | 1 to 2 h | about 1 GPU-hour | yes |
| 1 Warmup + validation collection, text caches, report | 2 to 3 h | collection (from Phase 0 timing, parallel shards) | yes |
| 2 Model | 3 to 5 h | minutes | no |
| 3 Sampler extension, text store, offline trainer, probes, offline probe, sanity checks | 5 to 6 h | 1 to 2 h | yes |
| 4 Rollout integration and rounds | 3 to 5 h | a 3-round test | no |
| 5 Closed-loop evaluation and report tooling | 4 to 6 h | a small grid | no |
| 6 Matrix scripts, first runs, final report | 2 h | as the human approves | yes |

- **Phase 0**: Section 3.4. Also write `scripts/lang/in_container.sh`, the sbatch templates,
  `main/lang/__init__.py` (bootstrap), `main/lang/config.py`, `main/lang/scene_roles.py` (check 6),
  `wire_item_eval` (check 5), and the Section 7.1 loop for nominal episodes, with a first
  `LangRolloutController` that only has the `run()` override (check 2).
- **Phase 1**: collect warmup and validation (Sections 7.1, 7.2), build both text caches, write
  `collect_report.md`. **STOP**: the human reviews label balance, h distributions, the delay audit (γ),
  and the sentence examples.
- **Phase 2**: Section 5 with its tests. Re-run the regression (Section 2).
- **Phase 3**:
  - implement Sections 4.3, 4.4, 6, 8.1, 8.2 and 9.1;
  - run the sanity checks of Section 6.4, then one offline run of `full` for 20k steps with probes every
    2k steps and the offline probe at the end;
  - **STOP**: the human reviews the curves and probe tables.
- **Phase 4**: Sections 7.3 to 7.5, 8.3 and 8.4. Tests:
  - one filter episode per task and embodiment with a random-init model runs to completion, with buffer,
    sidecar and `lang_episodes.jsonl` in lockstep and a video;
  - deployment frame 0 equals the s₀ row collection writes for the same seed;
  - exactly one capture happens before the first physics step in both modes;
  - the history never grows on cache misses;
  - `trace["status"]` is right for a forced timeout and a forced exception;
  - resume after the exit-75 path continues step, α, round index, schedule cursor and RNG states
    exactly.

  Then a 3-round smoke run with 256 grad steps per round, and the regression again.
- **Phase 5**: Section 9.2 and `report.py`. Run the grid with 2 scenes per (task, embodiment) on the
  Phase 3 checkpoint, including the nominal cache. Test resume by killing a child midway.
- **Phase 6**: `scripts/lang/matrix.sh`. Launch only what the human approves. Write
  `docs/lang_filter/final_report.md`.

---

## 11. `LangConfig` defaults and decisions reserved for the human

| field | default | note |
|---|---|---|
| `task_choices` | the 7 item tasks | `--tasks` filters |
| `t1_hosts` | the 5 CrossSafe hosts | |
| `embodiment_choices` | `ur5-wsg`, `franka-panda` | |
| `item_host`, `item_risk`, `item_active_rule`, `item_out_root` | None, None, None, `"runs/item_envs"` | per-episode env overrides (Section 2), declared so `dataclasses.replace` keeps them. `item_out_root` must never be None |
| `max_steps_per_episode` | from Phase 0 (ticks) | `collect()` and every rollout use it |
| `k_prev` | 2 | frame t−k |
| `n_queries` | 4 | |
| `max_text_tokens` | 24 | the cache's padded length |
| `text_backend`, backend paths | `holobrain_bert` | |
| `trunk_depth`, `critic_depth` | 3, 2 | |
| `lang_scene`, `lang_body`, `lang_film`, `lang_queries`, `lang_rtoken` | all on | path switches |
| `film_zero_init` | true | |
| `shuffle_text`, `relabel` | false, `paired` | |
| `softmin_T_h` | 0.05 | h units, not scaled by `h_scale` |
| `harmful_frac`, `pair_frac`, `task_balance` | 0.5, 0.5, true | |
| `recall_horizon_ticks` | 7 | `fails_soon` itself is the env's 25 ticks |
| `gamma`, `gamma_final` | 0.9, 0.9 | **human decides** after the delay audit. Must be equal |
| `batch_size`, `lr`, `tau` | 64, 3e-4, 0.005 | B fixed unless the human approves |
| `alpha_start`, `alpha_final`, `alpha_anneal_steps` | 0.2, 0.02, 20000 | |
| `filter_margin`, `filter_release_margin`, `hj_hold_ticks` | 0, 0, 3 | |
| `filter_episode_frac`, `episodes_per_round`, `grad_steps_per_round`, `collect_rounds` | 0.8, 4, 2048, 100 | |
| `named_harm_frac` | 0.5 | share of filter episodes whose named rule is harm-designated (Section 7.1) |
| `max_inits_per_process` | 200 | child restart and exit-75 limit (Sections 2, 7.4) |
| `perturb_prob` | 0.05 | |
| `warmup episodes per task per embodiment`, `validation` | 30, 5 | |
| seed bases | Section 4.2 | |
| `activation_checkpointing` | false | only if memory requires (Section 5.10) |
| `bank_hash` | hash of `paraphrases.yaml` + `splits.yaml` | written automatically |

Reserved for the human:
- γ;
- warmup size;
- evaluation scene counts;
- which matrix runs to launch;
- any change of batch size or precision;
- any edit under `envs/` beyond the `DESIGN_CONTEXT.md` pointer.

---

## 12. Final report (`docs/lang_filter/final_report.md`)

1. What was built, with file paths, and every deviation from this plan and why.
2. Phase 0 findings (tests, h parity, episode lengths, timing, in-the-way validation).
3. Data report summary (label balance per task and consequence, delay audit, γ decision).
4. Sanity checks of Section 6.4 with numbers.
5. For each launched run: offline probe tables per axis, closed-loop tables with the nondeterminism
   rate, and the per-question tables from `report.py`.
6. Where the vanilla model fails (by axis, task, consequence, referent vs consequence decisions), with
   example videos. This is the input for deciding on the advanced version (slots and an entity pointer).
