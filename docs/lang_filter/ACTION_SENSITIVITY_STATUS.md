# Action-sensitivity experiment: status and how to continue

Written 2026-11-11 17:40 (Delta time) for whoever continues the experiment on another machine (WashU RIS).
The plan is `docs/lang_filter/ACTION_SENSITIVITY_PLAN.md`; section 7 of it (the second round) is what to run now.
Code state: commit `d5d1a9d` or later.

## 1. Where things stand

- The first round is finished. Every critic fails, and the action injection is confirmed as one real cause.
- The second round is implemented and has barely started: 12 of its 125 planner episodes are collected, and none
  of its five trainings has run.
- **Nothing of this experiment runs on Delta any more.** On 11-11 17:33 the feeder was stopped and the queued jobs
  (B, D2, G, the collection) were cancelled, so that the work can move to a machine without queue waits. Do not
  restart the collection on Delta: two machines must not collect the same episodes.

## 2. Terms

- **Tick**: one control step, 40 ms. **h**: the safety margin of the rule "keep the tall block upright" (1 safe,
  0 at the tilt limit, below 0 tipped).
- **V95**: the true value of a continuation, from its h trace: at the last tick y = h, and before that
  y_t = 0.05 · h_t + 0.95 · min(h_t, y_{t+1}). Example: a branch that stays upright with h = 0.9 throughout has
  V95 = 0.9; one that tips later has a V95 that falls toward the tipped h the sooner the tip comes.
- **Decision**: what the critic scores at a state. `plan` = follow the planner. `t1_c0`, `t1_c50` = play the
  planner's path at 0 or 0.5 times its rate for 1 tick, then the planner again. `t3_c0`, `t3_c50` = the same for 3
  ticks. `ch_c0`, `ch_c25`, `ch_c50`, `ch_c75` = the same for 12 ticks (a 0.48 s chunk).
- **Rank agreement**: over pairs of decisions at one state whose V95 differ by more than 0.05, the share Q orders as
  the truth does. 0.5 is chance.
- **Spread ratio**: (highest − lowest Q over a state's decisions) ÷ (highest − lowest V95), each averaged over the
  states first. Example: true values 0.10 to 0.95 (spread 0.85), Q 0.29 to 0.33 (spread 0.04), ratio 0.05. A ratio
  of 1 means Q changes as much as the truth; 0 means Q gives every action the same score.
- **Pass**: rank agreement ≥ 0.70 **and** spread ratio ≥ 0.25 on held-out states, at step 6,000. Fixed on 11-10 in
  `runs/lang/action_sens/criteria.json`. Do not change these numbers.

## 3. What has been run, and the results

### Step 1: fit Q directly to V95 on 117 old branch states (done)

`fit_check.py`, 3 folds by episode, 1,500 steps, three first actions per state.

| injection | training states: rank / spread ratio | held-out states: rank / spread ratio |
|---|---|---|
| current (one Linear on the raw action effects) | 0.62 / 0.00 | 0.38 / 0.00 |
| revised (plan section 3.4) | 0.94 / 0.93 | 0.52 / 0.88 |

The current injection cannot represent the action's effect even when given the answer. The revised one can, but
with 8 training episodes it memorizes: new states are ranked at chance.
Outputs: `runs/lang/action_sens/step1/` (300 steps), `step1_long_linear/`, `step1_long_revised/`.

### Step 2: first branch collection (done)

`data/lang_asens/`: 25 planner episodes (seeds 6001-6025), 100 states, 7 decisions each = 700 branches, 27,177
stored rows, 21 GB. 20 episodes for training, 5 held out (every fifth seed). Two one-hour A40 jobs.

| decision | block falls | mean V95 |
|---|---|---|
| plan | 64% | +0.36 |
| t1_c0 | 87% | +0.17 |
| t1_c50 | 71% | +0.30 |
| ch_c0 | 96% | −0.01 |
| ch_c25 | 93% | +0.11 |
| ch_c50 | 72% | +0.44 |
| ch_c75 | 54% | +0.53 |

- The V95 spread over a state's decisions is 0.83 on average over all 7, 0.80 over the chunk decisions and 0.32
  over the 1-tick ones (median 0.19).
- A hard stop is the worst choice (it tips the block); slowing to 75% is the best.
- Every branch is recorded in a second pass; the outcome was the same in 699 of 700.

### Step 3, first round: critics trained with the normal HJ target (done, all fail)

6,000 steps, from scratch, MLP readout, branch rows a quarter of each batch ("old" mix). Probes on 20 held-out and
40 training states.

| run | decision | injection | branch data | held-out: rank / spread ratio | training states: rank / spread ratio | state |
|---|---|---|---|---|---|---|
| A | 1 tick | current | yes | 0.75 / 0.00002 | 0.62 / 0.00002 | done, fails |
| F | 1 tick | revised | no | 0.78 / 0.004 | 0.67 / 0.002 | done, fails |
| C | chunk | current | yes | 0.63 / 0.0001 | 0.61 / 0.0001 | done, fails |
| D | chunk | revised | yes | 0.68 / 0.04 (peak 0.10 near step 4,000-5,000) | 0.47 / 0.09 | done, fails |
| B | 1 tick | revised | yes | 0.81 / 0.005 at step 4,250 | 0.62 / 0.004 | stopped at step 4,250 of 6,000 |
| E | chunk | current | no | | | dropped by Yuxuan on 11-11 (stopped at step 2,000) |

The 1-tick runs are ranked on 36 held-out pairs, the chunk runs on 173.

What it says:
- With the current injection Q does not move with the action, whatever the data or the decision length (A, C).
- A, F and B rank above 0.70 only by accident: their Q differences are in the third to fifth decimal, and 36 pairs
  are few.
- The revised injection alone is not enough (F), and B, which adds the branch data, looks the same so far.
- D, with all three changes, is the best and still far too flat. It is low on its own training states too, so it
  never learned the action's effect (it did not memorize).

The numbers are the last probe of each run's `runs/lang/action_sens/runs/<run>/diag.jsonl`. `train_critic.py score`
(which writes `scores.json`) has **not** been run yet.

### Second round (plan section 7): what exists

- **Code, all pushed.** Collector: two 3-tick decisions and `--round 2` (seeds 6026-6150 into `data/lang_asens2/`).
  Trainer: the 3-tick critic, the new batch mix, runs D2, G, B2, H, D3, probes and scores on training and held-out
  states side by side.
- **Collection, round 2: 12 of 125 planner episodes** (seeds 6026-6037; 48 states, 432 branches, 11 GB; held out
  among them: 6030, 6035). One one-hour A40 job, no errors, second-pass outcome the same in 432 of 432.

  | decision | block falls | mean V95 |
  |---|---|---|
  | plan | 67% | +0.24 |
  | t1_c0 / t1_c50 | 83% / 77% | +0.14 / +0.18 |
  | t3_c0 / t3_c50 | 98% / 92% | +0.00 / +0.09 |
  | ch_c0 / ch_c25 / ch_c50 / ch_c75 | 96% / 96% / 83% / 65% | −0.00 / +0.08 / +0.29 / +0.38 |

  V95 spread per state: 0.27 over the 1-tick decisions, 0.42 over the 3-tick ones, 0.67 over the chunk ones. So a
  3-tick decision changes the outcome clearly more than a 1-tick one in these 48 states.
- **Trainings of the round: none has started.**

## 4. What is not done

| item | needs | state |
|---|---|---|
| collection, round 2 | simulator, A40 | 113 planner episodes left (seeds 6038-6150) |
| D2: chunk, small data, new mix | the copied data | not started |
| G: 3 ticks, normal rows only | the copied data | not started |
| B: 1 tick, revised, small data, old mix | the copied data | resumes at step 4,250 |
| B2: 1 tick, large data, new mix | the finished collection | not started |
| H: 3 ticks, large data, new mix | the finished collection | not started |
| D3: chunk, large data, new mix | the finished collection | not started |
| scores (`train_critic.py score`) | all runs done | not started |
| `docs/lang_filter/ACTION_SENSITIVITY_REPORT.md` | the scores | not written |

"Small" data is the first round's 100 states, "large" adds the round-2 states. The "new" mix is 32 normal, 16 branch
continuation and 16 decision rows per batch of 64 (plan section 7.2 a).

**Not yet tested on a GPU** (a CPU check of the batches passed): the new batch mix (D2, B2, H, D3), the 3-tick
critic (G, H), the new `score` command, and everything in `scripts/lang/action_sens/portable/`. Tested on a GPU:
the 9-decision collector (the 12 episodes above) and the old-mix trainer path under the new code (run B).

## 5. How to continue on the new machine

### 5.1 Copy the data (about 110 GB)

The code reads only paths inside the repo, so the data must sit at the same relative paths.

```bash
cd <repo> && git pull
ssh -M -S ~/.ssh/cm-delta -fN <user>@login.delta.ncsa.illinois.edu       # one connection: one password + Duo
RSYNC_RSH="ssh -S ~/.ssh/cm-delta" scripts/lang/action_sens/portable/transfer.sh \
  <user>@login.delta.ncsa.illinois.edu:/work/nvme/bifq/project/LanguageConditionHJ .
ssh -S ~/.ssh/cm-delta -O exit <user>@login.delta.ncsa.illinois.edu
```

| what it copies | real size | read by |
|---|---|---|
| `data/lang_1023/warmup`, text caches of `val` and `val20` | 29 GB | trainings (normal data) |
| `runs/lang/rounds_ur5_g095_1023/ring`, `lang_config.json` | 53 GB | trainings (normal data, settings); the collector reads the config |
| `runs/lang/diag_1030/retrain/action_norm.json` | small | the revised injection's bounds |
| `data/lang_asens` (branches, states, episodes) | 17 GB | trainings, probes |
| `data/lang_asens2` (the 12 episodes) | 11 GB | trainings, probes; the collector skips these episodes |
| `runs/lang/action_sens` | 5 GB | pass numbers, finished runs, run B's state |

- The buffer files are sparse (400 GB as a plain copy). The script uses `rsync --sparse`; Globus would fill them.
- If `data/` or `runs/` are symlinks to another storage, the script follows them.
- Interrupted: run the same command again. `DRY=1` in front only lists.
- Also needed, not copied by the script: the HoloBrain GD checkpoint, `bert-base-uncased`, and for the collection
  RoboTwin with its assets and the simulator environment (`docs/lang_filter/NEW_MACHINE_SETUP.md`).

### 5.2 Variables

```bash
export HOLOBRAIN_CKPT=<dir>/HoloBrain_v0.0_GD     # required
export PY=<python of the simulator environment>   # default: python
# with Slurm: export SBATCH_FLAGS="-A <account> -p <partition> --gpus=<type>:1"
```

Everything here is the GD version (HoloBrain-0 GD encoder with its BERT), γ = 0.95. Every job uses **one GPU**
(Yuxuan's rule: one GPU per job unless he says otherwise). Several jobs at once are fine.

### 5.3 Order of work

Run each line from the repo root inside a one-GPU job (with Slurm, `scripts/lang/action_sens/portable/submit.sh`
submits them; with LSF, wrap them in a `bsub` job).

1. **Start the collection** (A40, the long part):
   ```bash
   ROUND=2 bash scripts/lang/action_sens/portable/collect.sbatch
   ```
   - About 7.5 min per planner episode per worker, 2 workers per GPU: about 7 hours with one job, about 3.5 with
     two jobs side by side, about 2 with four.
   - It refuses a GPU without ray tracing (V100, A100, H100): the images would differ from all other data.
   - Done when `data/lang_asens2/COLLECT_DONE` exists (125 `episodes/*/result.json`).
   - In its log, check for `cuRobo` falling back to mplib (a path or TMPDIR problem) and that every state prints
     all 9 decisions.
2. **At the same time, the three trainings that need no new data** (any CUDA GPU, about 80 min each):
   ```bash
   NAME=D2 bash scripts/lang/action_sens/portable/train.sbatch
   NAME=G  bash scripts/lang/action_sens/portable/train.sbatch
   NAME=B  bash scripts/lang/action_sens/portable/train.sbatch
   ```
   The first lines of D2's log are the test of the new code. Expect:
   - `branch data (small, mix new): 700 branch episodes ... training rows: 15657 continuation, 4080 decision; batch parts [32, 16, 16]`
   - `run D2: decision 12 tick(s), injection revised ... probe states {'held': 20, 'train': 40}`
   - a `[probe] step 0` line, then one every 250 steps.

   G prints "no branch states with this decision length yet" unless `data/lang_asens2/states` already has files
   (the copied 12 episodes give it 8 held-out and some training states). That is expected: only round 2 has 3-tick
   decisions.
3. **After the collection:** `NAME=B2`, `NAME=H`, `NAME=D3` the same way. Expect `mix new` and `large` in the first
   lines, and three probe sets: `held`, `train`, `held1` (the first round's held-out states alone, to compare D3
   with D2).
4. **Scores:**
   ```bash
   $PY scripts/lang/action_sens/train_critic.py score --names A B C D F D2 G B2 H D3
   ```
   writes `runs/lang/action_sens/scores.json` (held-out, training states and `held1` per run, and `passes`).
5. **Report:** write `docs/lang_filter/ACTION_SENSITIVITY_REPORT.md` in plain words with the tables of plan section
   4 (step 4) and section 7.4, training and held-out side by side, the number of pairs behind every rank.

A run is finished when `runs/lang/action_sens/runs/<run>.DONE` exists. A killed job loses at most 250 steps (or the
planner episode in progress): start the same line again and it resumes.

### 5.4 How to read the results (plan section 7.4)

| result | conclusion |
|---|---|
| D2 well above D (spread ratio 0.04) | the batch mix was the limit |
| D3 well above D2 | more boundary states help, keep collecting |
| G passes | a lower rate (about 8 Hz) alone is enough |
| H passes, B2 fails | about 8 Hz with branch data is enough, no chunks needed |
| only D3 passes | the decision has to be about 0.5 s long |
| nothing passes, high on training states | still too few states |
| nothing passes, low on training states too | the model still cannot carry the action (plan section 7.5 lists what to try) |

### 5.5 Bringing results back

The report and the repo's `docs/` go through git. The run outputs are not in git (`runs/` is ignored): copy
`runs/lang/action_sens/` (and `data/lang_asens2/` if Delta should hold the full data) back with `transfer.sh`,
source and destination swapped, or keep them on the new machine and say so in the report.

## 6. Things that are easy to get wrong

- **Context rows.** Every branch episode starts with 3 context rows (`n_ctx` in its `lang_episodes.jsonl` line) so
  that the frames t, t−2 and 0 are real. They are never training rows; `train_critic.py` drops them.
- **Decision rows** are the first max(H, 1) rows of a branch from its state row (H = the branch's decision length).
  They are drawn one branch at a time (each branch equally often, then one of its decision rows), so about 47% of
  them are state rows, where the 7 or 9 branches show the same state with different actions. The plan fixes only
  their 25% share of the batch; drawing per branch is an implementation choice.
- **The 3-tick critic** takes the sum of 3 stored commands as one action. The model is the one-tick model
  (`critic_chunk` = 1); only the bounds of the revised injection are 3 times one tick's. A 3-tick critic scores
  `plan`, `t3_c0`, `t3_c50`, which exist only in round 2.
- **Probe sets differ between small-data and large-data runs** (20 held-out states against all held-out states).
  Compare D3 with D2 on `held1`, not on `held`.
- **Do not edit** the CrossSafe files (`main/network`, `main/train`, `main/envs`) or `envs/`; the experiment's
  model code is `main/lang/network.py` `ActionCriticHead`.
- **Run E** is dropped; its folder (`runs/E`, step 2,000) can be ignored. `runs/lang/action_sens/smoke/` and
  `data/lang_asens_smoke/` are test leftovers.

## 7. Files

| file | what |
|---|---|
| `scripts/lang/action_sens/fit_check.py` | step 1 |
| `scripts/lang/action_sens/collect_branches.py` | the branch collector (`run --round 2`, `table`) |
| `scripts/lang/action_sens/train_critic.py` | `train --name <run>`, `score`, `criteria`; the runs are in its `RUNS` |
| `scripts/lang/action_sens/portable/` | `transfer.sh`, `train.sbatch`, `collect.sbatch`, `submit.sh` (any machine) |
| `scripts/lang/action_sens/{feeder.sh, train.sbatch, collect.sbatch}` | Delta only (its queues and container) |
| `runs/lang/action_sens/runs/<run>/diag.jsonl` | every probe of a run |
| `runs/lang/action_sens/criteria.json` | the pass numbers |
| `data/lang_asens*/branch_table.json`, `episodes/*/result.json` | every branch with its truth |
