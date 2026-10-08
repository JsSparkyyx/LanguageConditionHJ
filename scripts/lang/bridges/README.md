# Running the language filter on Bridges-2 (or any Slurm GPU cluster)

The goal is the rounds training run (VANILLA_PLAN 6.2): round 0 trains on the warmup data, and every
later round collects 4 episodes with the current weights (80% with the filter on), then trains 2048
steps. The scripts here are plain bash plus one Slurm file. They never need Delta's container.

| script | what it does | where |
|---|---|---|
| `check.sh` | environment, RoboTwin links, checkpoints, data (incl. bead shards), CPU tests | login node, ~5 min |
| `collect_bead.sh` | the T2 / T4 / T6 shards under the new bead h (only if not copied from Delta) | GPU node, ~3.5 GPU-h |
| `smoke.sh` | a tiny rounds run with a forced stop and resume, then checks every output | GPU node, ~30 min |
| `rounds.sbatch` | the real run, self-resubmitting until done | Slurm |
| `status.sh <run>` | progress: round, phase, losses, per-round numbers, newest videos | login node |

## 1. What to copy

| what | Delta path | size | where on Bridges |
|---|---|---|---|
| code | `git clone https://github.com/JsSparkyyx/LanguageConditionHJ` (latest `main`) | small | `<repo>` |
| RoboTwin with assets | `/work/nvme/bifq/project/RoboTwin/` | ~20 GB (skip the core dumps) | `<repo>/../RoboTwin` (the code looks there or in `<repo>/RoboTwin`; it ignores `ROBOTWIN_ROOT`) |
| HoloBrain checkpoint | `/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD` | 2.1 GB | anywhere, `HOLOBRAIN_CKPT` |
| BERT tokenizer + config | `/work/nvme/bifq/project/tmp/hf-cache/hub/models--bert-base-uncased` | small | `$HF_HOME/hub/` (only if compute nodes have no internet) |
| warmup data | `data/lang/warmup/` (30 original shards + 30 `shard_g*_bead`, `text_cache.pt`) | 89 GB + ~47 GB bead | `<repo>/data/lang/warmup` |
| validation data | `data/lang/val/` (10 original + 10 `_bead`, `text_cache.pt`) | 14 GB + ~7 GB bead | `<repo>/data/lang/val` |

```bash
rsync -a --exclude '.git' --exclude 'core.*' delta:/work/nvme/bifq/project/RoboTwin/ ../RoboTwin/
```

Data: copy with `--sparse` (the buffer files are created at full capacity; without it rsync writes the
unused part too, ~365 GB instead of ~160 GB). Copy only FINISHED shards while Delta is still collecting:
rsync copies the arrays before `header.json`, so a shard that gets a new episode mid-copy arrives with a
header that claims rows its arrays lack (resuming such a shard now refuses, but the copy is wasted).

```bash
# list the finished shards on Delta (summary.json says complete), then copy them and the text caches
ssh delta 'cd /work/nvme/bifq/project/LanguageConditionHJ/data/lang && for d in */shard_g*; do
  grep -q "\"complete\": true" $d/summary.json 2>/dev/null && echo $d; done; ls */text_cache.pt' > shards.txt
rsync -a -r --sparse --files-from=shards.txt delta:/work/nvme/bifq/project/LanguageConditionHJ/data/lang/ data/lang/
```

Rerun both commands until `check.sh` reports every bead shard complete (finished shards are skipped).

The RoboTwin directory on Delta also holds ~100 GB of `core.python.*` crash dumps: the `--exclude`
skips them. The scripts here turn core dumps off (`ulimit -c 0`).

Python environment: the same one as for the env verification (`envs/scripts/bridges/README.md`,
sections 0 and 1: sapien 3.0.0b1, cuRobo, CUDA torch, the four editable installs).

**GPU:** rendering uses ray tracing, which needs an RTX-class GPU (A40, L40S, RTX 6000 ...). V100,
A100 and H100 have no RT cores. The trainer peaks at 27 GB in rounds mode plus the simulator, so use a
48 GB card. Keep one GPU model for the whole run (cuRobo and PhysX differ at the cm level across
hardware).

## 2. Setup check

```bash
cd <repo>
export HOLOBRAIN_CKPT=<dir>/HoloBrain_v0.0_GD PY=<python of the RoboTwin env>
# export HF_HOME=<dir>      # if you copied the BERT files
bash scripts/lang/bridges/check.sh
```

It installs the RoboTwin task links (the ones copied from Delta point at Delta paths), and ends with
`ALL OK` or a list of FAIL lines. It fails while any bead shard is incomplete: that is on purpose (step 3).

## 3. Bead data (T2 / T4 / T6)

The bead h changed on 2026-10-07, so these tasks were collected again on Delta into `shard_g*_bead`
folders (old bead rows are skipped automatically). Either:

- copy the finished folders from Delta (rsync again when Delta is done), or
- finish them on Bridges: `bash scripts/lang/bridges/collect_bead.sh 4` on a GPU node. Same seeds as
  on Delta; copied folders are reused (complete ones skipped, partial ones resumed). Rerun after a stop.
  Do not run it while Delta's queue is still writing the same folders.

**Start the rounds run only after `check.sh` reports all bead shards complete:** the trainer indexes the
data folders once at startup, and a folder that is empty then is left out of the whole run.

## 4. Smoke test (once, on a GPU node)

```bash
bash scripts/lang/bridges/smoke.sh
```

It runs 2 rounds of 2 episodes and 64 steps, stops itself after 8 minutes, resumes, and finishes. It
ends with `[smoke] ALL OK`, or names what failed (log in `runs/bridges_logs/`). A `cuRobo fell back to
mplib` warning means a broken TMPDIR or GPU setup: fix that before the real run.

## 5. The run

```bash
mkdir -p runs/bridges_logs                       # Slurm does not create the log directory
export RUN=runs/lang/lang_full_paired_rounds_<date> HOLOBRAIN_CKPT=<dir> PY=<python>
export SBATCH_FLAGS="-A <account> -p GPU-shared --gpus=<rtx type>:1"
# export WANDB_API_KEY=<key>                      # online wandb (project cehj-lang); else offline files
sbatch $SBATCH_FLAGS scripts/lang/bridges/rounds.sbatch
```

- gamma: the default is 0.9 (the plan's). The 0.95 run is a second run in its own directory:
  `RUN=runs/lang/rounds_g095 GAMMA=0.95 sbatch $SBATCH_FLAGS scripts/lang/bridges/rounds.sbatch`.
  Other config overrides: `SET="key=value ..."` (not inside EXTRA_ARGS). A run keeps the config of its
  first start; resubmissions inherit the variables through `--export=ALL`.
- 100 rounds take ~85 min each on an A40 (67 min of training at ~0.5 steps/s, 15-20 min of collection):
  ~140 GPU-hours, ~3 jobs of 48 h. Smaller: `export STEPS=1024` (~50 min per round) before the first
  `sbatch`, and keep it for every resubmission.
- Each job stops the trainer 10 minutes before its time limit, inside the round if needed (state saved),
  and submits the next job itself. A process restart for the environment-init limit happens inside a
  job. If a job dies without that (node failure), submit the same command again: it resumes from the last
  saved point (at most one episode or 500 steps are lost).
- Disk: the run directory grows to ~80-90 GB (400 episodes in `ring/`). The ring file is created at its
  full size (~205 GB apparent) but only the written part takes space.
- The trainer reads ~80 MB per step at random from `data/lang/`: keep it on the fastest storage you
  have. If `status.sh` shows steps/s well below 0.5 on an A40-class GPU, check `perf/data_wait_frac` in
  `train_log.jsonl` (the fraction of time spent waiting for data).
- Stop for good: `scancel <job>`, then `RESUBMIT=0` before any later `sbatch`.

## 6. Watching it

```bash
bash scripts/lang/bridges/status.sh $RUN
```

Files (all under `$RUN`):

| file | what |
|---|---|
| `eval/trend.png` | per round: harmed on harm runs (should fall), intervened on benign runs (should stay low), task success, Q <= h |
| `eval/<metric>/roundNNN.png` | heatmaps task x (robot x harm / benign) for harm_named, any_intervention, intervention_rate, task_success |
| `eval/rounds.jsonl` | one row per filter episode |
| `ring/videos/*.mp4` | every collected episode: `<task>[_<T1 host>]_<robot>_s<seed>_<kind>_<outcome>.mp4`, kind = harm / benign (filter on, the named rule's item in the way or not) or nominal (filter off), outcome = ok / harmed / fail. The panel shows the sentence, h_named, Q, FILTER / NOMINAL; a red border while the named rule is broken |
| `train_log.jsonl` | losses and diagnostics every 20 steps, round summaries |
| `probes/` | offline language probe every 10 rounds |
| `checkpoint.pt`, `checkpoint_round<N>.pt`, `checkpoint_final.pt` | weights |

**What the first rounds look like.** The critic starts pessimistic: in the offline run, mean Q stayed
below 0 (the filter's trigger) until about step 4000-8000, while the true h averaged +0.3 to +0.4. So
rounds 1-3 will probably show the filter stepping in on most ticks, benign runs flagged as false
alarms, and filter episodes running to the 30 s cap (`_fail` videos, `status: timeout`). That is
expected; it should fall once Q rises above 0 for safe moments. The smoke test on Delta showed the
same thing after 64 steps (intervention rate 1.0, timeout).

With `WANDB_API_KEY` set, the same numbers, the heatmaps, the trend plot and each round's videos also go
to wandb (`round_videos`).
