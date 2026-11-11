# Running the safety filter on a new machine

This guide covers what to copy, what to set and what to check before you run the language-conditioned safety filter
(`main/lang/`) on a cluster other than Delta. The scripts are in `scripts/lang/bridges/`. They were written for PSC
Bridges-2 and work on any Slurm GPU cluster; their README has more detail. You do not need to edit any code: every
path comes from where the repo sits, or from the variables below.

Use commit `06731e0` or later. Older commits have Delta's paths hard-coded in the safe planner.

## 1. What to copy

| what | size | where to put it | how the code finds it |
|---|---|---|---|
| this repo | | anywhere: `git clone https://github.com/JsSparkyyx/LanguageConditionHJ` | |
| RoboTwin, with its `assets/` | ~20 GB | **next to the repo** (`<parent>/RoboTwin`) or inside it (`<repo>/RoboTwin`) | its location only (section 2) |
| HoloBrain GD checkpoint, the whole `HoloBrain_v0.0_GD` folder | 2.1 GB | anywhere | `HOLOBRAIN_CKPT` |
| `bert-base-uncased` (the Hugging Face cache folder `models--bert-base-uncased`) | < 1 MB | `$HF_HOME/hub/` | `HF_HOME`; skip it if the compute nodes have internet |
| Qwen runs only: `models--HorizonRobotics--HoloBrain_v0.0_Qwen` and `models--Qwen--Qwen2.5-VL-3B-Instruct` | 2.1 + 7.1 GB | `$HF_HOME/hub/` | `HF_HOME`, or `HOLOBRAIN_QWEN_CKPT` and `QWEN_BASE_DIR` |
| data (optional: copy it or collect it again, section 6) | ~52 GB; Qwen replays ~128 GB | anywhere, on the fastest disk you have | script variables (section 4) |

Where these are on Delta:

```bash
rsync -a --exclude '.git' --exclude 'core.*' delta:/work/nvme/bifq/project/RoboTwin/ <parent>/RoboTwin/   # skip ~100 GB of crash dumps
rsync -a delta:/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD/ <ckpt>/HoloBrain_v0.0_GD/
rsync -a delta:/work/nvme/bifq/project/tmp/hf-cache/hub/models--bert-base-uncased <hf_home>/hub/
# Qwen runs only: the same for models--HorizonRobotics--HoloBrain_v0.0_Qwen and models--Qwen--Qwen2.5-VL-3B-Instruct
# data: --sparse, because the buffer files are created at full size and are mostly empty
rsync -a --sparse delta:/work/nvme/bifq/project/LanguageConditionHJ/data/lang_1023/ <data>/lang_1023/
rsync -a --sparse delta:/work/nvme/bifq/project/LanguageConditionHJ/data/lang_qwen_1023/ <data>/lang_qwen_1023/
```

**Python environment** (Delta's `RoboTwin` conda env):
- Python 3.10, torch 2.7.1+cu128, sapien 3.0.0b1, transformers 4.57.6, mplib 0.2.1, warp-lang 1.12.0,
  pytorch_kinematics 0.10.0, wandb.
- Installed in editable mode: cuRobo (`RoboTwin/envs/curobo`), XPolicyLab (`RoboTwin/XPolicyLab`), `robo_orchard_core`
  at commit `6aa3eaa`, `RoboOrchardLab` at commit `241bbc6`.
- Qwen runs also need flash-attn 2.8.3.

The full package list is `envs/scripts/bridges/pip-freeze.txt`. The install commands are in
`envs/scripts/bridges/README.md`, section 0.

## 2. Folder layout

```
<parent>/
  LanguageConditionHJ/    the repo
  RoboTwin/               RoboTwin with assets/   (or LanguageConditionHJ/RoboTwin)
```

The simulator code (`main/envs/env.py`) looks for RoboTwin only in these two places. Setting `ROBOTWIN_ROOT` does
**not** move it.

## 3. One-time fixes

1. **RoboTwin's robot files hold absolute paths.** Each `RoboTwin/assets/embodiments/*/curobo.yml` names its robot
   model files (URDF and collision spheres) by full path. A RoboTwin copied from Delta still names Delta's files. The
   cuRobo motion planner then fails to load, and every plan silently falls back to the slower mplib planner. Fix it
   once after copying:
   ```bash
   cd <parent>/RoboTwin && $PY scripts/update_embodiment_config_path.py
   ```
2. **Task links.** The setup check `check.sh` (section 5) adds one symlink per item task,
   `RoboTwin/envs/<task>.py` → `<repo>/envs/robotwin_tasks/<task>.py`. It changes nothing else in RoboTwin.
3. **The safe planner's robot model** needs nothing. `envs/main/item_envs/configs/curobo_safe/ur5-wsg.yml` names
   Delta's files; at run time the code writes a copy that names this machine's RoboTwin into `$TMPDIR`.

## 4. Variables

Put these in a file and source it in every shell before you submit. The Slurm scripts pass the environment on to
their jobs:

```bash
export PY=python                          # the RoboTwin env's python (activate the env first), or its full path,
                                          # or a command such as "apptainer exec --nv <image.sif> python"
export HOLOBRAIN_CKPT=<ckpt>/HoloBrain_v0.0_GD
export HF_HOME=<hf_home>                  # holds hub/models--bert-base-uncased (and the Qwen folders)
export LANG_TMP=<fast scratch>/lang_tmp   # caches and TMPDIR; must be writable
export SBATCH_FLAGS="-A <account> -p <partition> --gpus=<rt gpu type>:1"
export WANDB_API_KEY=<key>                # optional; without it wandb writes offline files in the run
```

| variable | needed | what it is |
|---|---|---|
| `HOLOBRAIN_CKPT` | yes | the `HoloBrain_v0.0_GD` folder (holds `model.config.json`); every script stops without it |
| `PY` | yes | the python to run; default `python` |
| `SBATCH_FLAGS` | yes, for jobs | account, partition, GPU. The self-resubmitting jobs reuse it, so put every site flag here, not only on the first `sbatch` line |
| `HF_HOME` | recommended | Hugging Face cache; default `<repo>/runs/bridges_tmp/hf-cache` |
| `LANG_TMP` | recommended | scratch for caches and `TMPDIR`; default `<repo>/runs/bridges_tmp`. cuRobo needs a writable `TMPDIR`, or it falls back to mplib |
| `WANDB_API_KEY`, or `WANDB_KEY_FILE` | optional | online wandb (project `cehj-lang`). `WANDB_KEY_FILE` names a file with an `export WANDB_API_KEY=...` line, read by `rounds.sbatch` and never printed |
| `HOLOBRAIN_QWEN_CKPT`, `QWEN_BASE_DIR` | Qwen runs, only if the models are not in `$HF_HOME/hub` | the HoloBrain-Qwen `pretrain/` folder (revision `b82becd9…`) and the Qwen2.5-VL-3B-Instruct folder (revision `66285546…`) |
| `SENTENCE_GENERIC_DIR` | only for the `text_backend=sentence_generic` ablation | the `all-mpnet-base-v2` folder (else `$HF_HOME/all-mpnet-base-v2`) |
| `VLM_ROOT`, `VLM_FP8_ROOT` | only for `h_source=vlm:<name>` runs | the VLM checkpoint folders |

**Leave these unset:**
- `TMPDIR`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`: the scripts point them into `LANG_TMP`.
- `SAPIEN_DISABLE_RAY_TRACING`: chosen from the GPU automatically.
- `STAGE_DEADLINE`: `collect.sbatch` sets it.
- `ROBOTWIN_ROOT`: ignored by the safety filter.
- `GAMMA`: the discount is fixed at 0.95.
- `CHAIN_PIPE`, `CHAIN_STAGE`, `LANG_REPO`: Delta only.

**Path variables of each script.** Relative paths are taken from the repo root.

| script | variables | defaults |
|---|---|---|
| `check.sh` | `DATA_DIRS` (data folders to check), `QWEN=1` (also the Qwen models and flash-attn), `SKIP_LINKS=1` | the `data/lang/{warmup,val,val20,val_grounding}` folders that exist |
| `smoke.sh` | `DATA`, `VAL`; smaller test: `SMOKE_EPISODES`, `SMOKE_STEPS`, `SMOKE_BUDGET` | `data/lang/warmup`, `data/lang/val` |
| `collect.sbatch` | `ROOT`, `QROOT`, `WHAT` (`vanilla`, `qwen` or `all`), `PAR` (simulators per GPU), `ALLOW_RASTER=1` | `data/lang`, `data/lang_qwen`, `vanilla`, 4 |
| `launch.sh` | `DATA_ROOT`, `QDATA_ROOT`, `RUNS_DIR`, `TAG` | `data/lang`, `data/lang_qwen`, `runs/lang`, today's MMDD |
| `rounds.sbatch` (direct use) | `RUN`, `DATA`, `VAL`, `EXTRA_ARGS`, `SET`, `ROUNDS`, `EPISODES`, `STEPS`, `PROBE`, `RESUBMIT=0` | see its header |

## 5. The GPU and Slurm

- **The GPU must have ray-tracing (RT) cores:** A40, L40S, RTX A6000 or RTX 6000 Ada, for example. V100, A100 and H100
  have none; on them the simulator switches to plain rasterization by itself. The camera images then differ from
  Delta's, and so do the HoloBrain tokens the filter learns from.
  - `collect.sh` refuses such a GPU; `ALLOW_RASTER=1` overrides.
  - Do not train on Delta's data on such a GPU: the new episodes collected during training would be rendered
    differently from the copied ones.
  - On Bridges-2, the V100 and H100 nodes have no RT cores. `sinfo -o "%P %G"` lists the GPU types; pick an RT-capable
    one.
- **48 GB of GPU memory:** the trainer peaks at 27 GB, next to the simulator and cuRobo.
- **One GPU model for a whole run:** cuRobo and the physics engine differ at the centimetre level across GPU models.
- **Time and CPUs:**
  - The jobs ask for `--cpus-per-task=16`; the time limit is `--time=48:00:00` for training and 24 h for collection.
  - If your partition caps these, add `--cpus-per-task=…` and `--time=…` to `SBATCH_FLAGS`: flags on the command line
    override the script's own `#SBATCH` lines.
  - Delta's runs also used `--mem=64g`.
- **Job logs** go to `<repo>/runs/bridges_logs/`. The scripts create this folder. If you call `sbatch` on a job script
  before ever running one of them, run `mkdir -p runs/bridges_logs` first: Slurm does not create the folder, and the
  job fails without writing a log.

## 6. Check it, in this order

1. **Setup check** (~13 minutes; the CPU tests take ~11 of them):
   ```bash
   cd <repo> && source <your variables file>
   bash scripts/lang/bridges/check.sh            # QWEN=1 bash ... for the Qwen runs too
   ```
   It ends with `ALL OK`, or with `FAIL:` lines that say what to do. Run it in an interactive session if the cluster
   limits CPU time on login nodes.
   - It checks every path in sections 1 to 3, and the data folders when given (`DATA_DIRS`).
   - "note: no data yet" is not a failure.
   - After copying data, also run `$PY scripts/lang/check_shards.py <data folders>` once, on a compute node. It reads
     every row, ~1 minute per 10 GB.
2. **Data**: copied (section 1), or collected from scratch with Delta's seeds (~5 GPU-hours):
   ```bash
   ROOT=<data>/lang sbatch $SBATCH_FLAGS scripts/lang/bridges/collect.sbatch                       # -> <data>/lang/READY
   WHAT=all ROOT=<data>/lang QROOT=<data>/lang_qwen sbatch $SBATCH_FLAGS scripts/lang/bridges/collect.sbatch   # + Qwen
   ```
   It resubmits itself until `READY` appears. If a shard fails 3 times it stops; the logs are in
   `runs/lang_collect/<root name>/logs` and `<ROOT>/logs`.
3. **Smoke test** on a GPU node (~30 minutes): a tiny training run that is stopped, resumed and checked.
   ```bash
   DATA=<data>/lang_1023/warmup VAL=<data>/lang_1023/val bash scripts/lang/bridges/smoke.sh
   ```
   It must end with `[smoke] ALL OK`. In its log (`runs/bridges_logs/smoke_*.log`), the line `[env] <GPU name>: ray
   tracing` must appear and `CuRobo init failed` must not.
4. **The runs** (Delta's 10-23 settings: ur5-wsg, discount 0.95, 100 rounds of 4 episodes and 2048 steps):
   ```bash
   DATA_ROOT=<data>/lang_1023 bash scripts/lang/bridges/launch.sh vanilla       # or t1
   QDATA_ROOT=<data>/lang_qwen_1023 bash scripts/lang/bridges/launch.sh qwen    # or t1qwen
   bash scripts/lang/bridges/status.sh runs/lang/<run name>                     # progress
   ```
   A run takes ~140 GPU-hours (about 3 jobs of 48 h) and grows to ~80-90 GB.

## 7. Things that go wrong quietly

- **cuRobo falls back to mplib** without stopping anything. Causes: robot files that still name Delta's paths (section
  3.1), an unwritable `TMPDIR`, or a GPU problem. To spot it, grep the logs for `CuRobo init failed`.
- **A run cannot move between machines.** A run folder records its data folders (absolute paths) and the code commit
  when it starts, and resuming refuses a different data location. Start a new run on the new machine.
- **One folder per run.** Two trainers writing one run folder corrupt its buffer. `launch.sh` refuses a run folder
  that already exists.
- **Slow disk.** The trainer reads ~80 MB per step at random from the data. On slow storage the GPU waits:
  `perf/data_wait_frac` in the run's `train_log.jsonl` shows how much.
- **Core dumps.** A simulator crash writes multi-GB core files into the RoboTwin folder. The scripts turn core files
  off (`ulimit -c 0`); do the same in your own shells.

## Tested

On 2026-11-09, on Delta, from a fresh clone and a RoboTwin in another location (not the original paths):
- the setup check ended `ALL OK`;
- RoboTwin was found next to the moved repo, and both robots' cuRobo files pointed into it;
- the safe planner's copy named the moved files;
- the checkpoints and the Qwen models loaded;
- tests: 366 passed in `tests/lang`, 517 passed in `envs/tests/item_envs`.

The GPU smoke test from that clone was still waiting in the queue.

## Moving the action-sensitivity experiment (added 2026-11-11, not yet run on another machine)

The critic experiment of `ACTION_SENSITIVITY_PLAN.md` reads only paths inside the repo, so it moves with one copy
script and three job scripts in `scripts/lang/action_sens/portable/`.

1. **Copy.** Clone the repo on the new machine, then copy the data to the same relative paths (about 110 GB real
   size on 11-11; the files are sparse, so use rsync):
   ```bash
   # on Delta (push), or the mirror image on the new machine (pull)
   scripts/lang/action_sens/portable/transfer.sh . me@other:/path/LanguageConditionHJ          # DRY=1 lists only
   ```
   Also copy the HoloBrain GD checkpoint and `bert-base-uncased` (section 1). With Globus, which fills sparse files
   (400 GB here), pack first: `tar -S -cf asens.tar <the paths transfer.sh lists>` and unpack with `tar -S -xf`.
2. **Trainings** (no simulator, any CUDA GPU, about 80 min per run):
   ```bash
   export HOLOBRAIN_CKPT=<dir>/HoloBrain_v0.0_GD PY=<python> SBATCH_FLAGS="-A <account> -p <partition> --gpus=<type>:1"
   scripts/lang/action_sens/portable/submit.sh train D2 G B
   ```
3. **Collection** (the simulator: RoboTwin next to the repo, the environment of section 1, a ray-tracing GPU):
   `scripts/lang/action_sens/portable/submit.sh collect 4` runs 4 one-GPU jobs side by side; the 113 planner
   episodes left on 11-11 take about 7.5 min each per worker, 2 workers per GPU. Episodes already collected on Delta
   are skipped if `data/lang_asens2` was copied. Stop Delta's collection first (two machines must not collect the
   same episodes).
4. Then `submit.sh train B2 H D3`, and `submit.sh score`.
