# Running the safety filter on Bridges-2 (or any Slurm GPU cluster)

The scripts here run the language-conditioned safety filter (`main/lang/`) without Delta's container: the rounds
training (VANILLA_PLAN 6.2) with the settings of Delta's 2026-10-23 runs, the data collection behind them, and the
checks. Every path comes from the repo's location or from the variables below; nothing points at Delta.

| script | what it does | where |
|---|---|---|
| `check.sh` | packages, every path (RoboTwin and its robot files, task links, checkpoints, safe planner model, data), CPU tests | login node, ~5 min |
| `smoke.sh` | a tiny rounds run on existing data, stopped by a time budget, resumed, checked | GPU node, ~30 min |
| `collect.sbatch` | the data from scratch (`collect.sh`: Delta's 10-23 collection, same seeds), self-resubmitting until READY | Slurm |
| `launch.sh` | starts one of the four 10-23 runs (vanilla, t1, qwen, t1qwen) as `rounds.sbatch` | login node |
| `rounds.sbatch` | the rounds run, self-resubmitting until done | Slurm |
| `status.sh <run>` | progress: round, phase, losses, per-round numbers, newest videos | login node |

## 1. Where things go

| what | where on Bridges | how the code finds it |
|---|---|---|
| this repo | `<repo>` = `git clone https://github.com/JsSparkyyx/LanguageConditionHJ` | the scripts' own location |
| RoboTwin with its assets | `<repo>/../RoboTwin` (or `<repo>/RoboTwin`) | `main/envs/env.py` looks only there; `ROBOTWIN_ROOT` is not read (neither by it nor by the safe planner) |
| HoloBrain GD checkpoint | anywhere (Delta: `/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD`, 2.1 GB) | `HOLOBRAIN_CKPT` |
| `bert-base-uncased` | `$HF_HOME/hub/models--bert-base-uncased` (only if compute nodes have no internet) | `HF_HOME` |
| Qwen runs only: HoloBrain-Qwen + Qwen2.5-VL-3B | `$HF_HOME/hub/models--HorizonRobotics--HoloBrain_v0.0_Qwen` (revision `b82becd9…`) and `models--Qwen--Qwen2.5-VL-3B-Instruct` (`66285546…`), or any directories | `HF_HOME`, or `HOLOBRAIN_QWEN_CKPT` (the `pretrain/` folder) and `QWEN_BASE_DIR` |
| data | anywhere: one folder holding `warmup/ val/ val20/ val_grounding/` (Qwen: a second one, same layout) | `DATA_ROOT` / `QDATA_ROOT` for `launch.sh`, `DATA` / `VAL` for `smoke.sh`, `DATA_DIRS` for `check.sh` |
| runs | `<repo>/runs/lang/<name>` by default | `RUNS_DIR` for `launch.sh` |

**RoboTwin's robot files hold absolute paths.** `assets/embodiments/*/curobo.yml` name their URDF and sphere
files by full path. RoboTwin's asset download writes them for its own location; a RoboTwin copied from Delta still
names Delta's files, cuRobo then fails to load and every plan quietly falls back to mplib. Fix once after copying:

```bash
cd <repo>/../RoboTwin && python scripts/update_embodiment_config_path.py
```

`check.sh` tests this. The safe planner's own robot model (`envs/main/item_envs/configs/curobo_safe/ur5-wsg.yml`,
Delta's paths) needs nothing: `safe_plan.safe_yml_for` writes a copy naming this machine's RoboTwin files.

When copying RoboTwin from Delta, skip the crash dumps (~100 GB):
`rsync -a --exclude '.git' --exclude 'core.*' delta:/work/nvme/bifq/project/RoboTwin/ ../RoboTwin/`

Variables every script reads (put them in your job environment or a file you source):

```bash
export HOLOBRAIN_CKPT=<dir>/HoloBrain_v0.0_GD
export PY=<python of the RoboTwin env>      # a command is fine, e.g. "apptainer exec --nv <sif> python"
export HF_HOME=<dir>                         # optional; default <repo>/runs/bridges_tmp/hf-cache
export LANG_TMP=<fast scratch>               # optional; caches + TMPDIR (cuRobo needs a writable TMPDIR)
export SBATCH_FLAGS="-A <account> -p GPU-shared --gpus=<rtx type>:1"
export WANDB_API_KEY=<key>                   # optional: online wandb (project cehj-lang); else offline files
```

Python environment: the one of `envs/scripts/bridges/README.md` (sections 0 and 1: sapien 3.0.0b1, cuRobo, CUDA
torch, the editable installs); Qwen runs also need `flash-attn`.

**GPU:** rendering uses ray tracing, which needs RT cores (A40, L40S, RTX 6000 ...). On V100 / A100 / H100 the
simulator switches to rasterization by itself, so images and HoloBrain's tokens differ from Delta's;
`collect.sh` refuses such a GPU (`ALLOW_RASTER=1` overrides). The trainer peaks at 27 GB in rounds mode next to the
simulator: use a 48 GB card, and one GPU model for a whole run (cuRobo and PhysX differ at the cm level across
hardware).

## 2. Setup check

```bash
cd <repo>
bash scripts/lang/bridges/check.sh                                   # QWEN=1 for the Qwen checkpoints too
DATA_DIRS="/ocean/.../lang/warmup /ocean/.../lang/val" bash scripts/lang/bridges/check.sh   # data elsewhere
```

It installs the RoboTwin task links (only adds symlinks), checks every path above, and ends with `ALL OK` or FAIL
lines that say what to do. A data folder needs complete shards and its `text_cache.pt` (Qwen replay shards: none).
After copying data, also run `$PY scripts/lang/check_shards.py <dirs>` once (every row; ~1 min per 10 GB).

## 3. Data

Copy it (a data root folder per kind; `--sparse`, the buffer files are created at full size and mostly empty):

```bash
rsync -a --sparse delta:/work/nvme/bifq/project/LanguageConditionHJ/data/lang_1023/ <data>/lang_1023/   # ~52 GB
rsync -a --sparse delta:/work/nvme/bifq/project/LanguageConditionHJ/data/lang_qwen_1023/ <data>/lang_qwen_1023/   # Qwen, ~128 GB
```

The shards hold no absolute paths the code reads, so any location works. Or collect it from scratch (same 29
shards and seeds as Delta: 150 warmup + 25 val + 60 val20 + 40 val_grounding episodes, ~5 GPU-hours):

```bash
mkdir -p runs/bridges_logs
ROOT=<data>/lang sbatch $SBATCH_FLAGS scripts/lang/bridges/collect.sbatch              # -> <data>/lang/READY
WHAT=all ROOT=<data>/lang QROOT=<data>/lang_qwen sbatch $SBATCH_FLAGS scripts/lang/bridges/collect.sbatch   # + Qwen
```

More GPUs in one job (`--gpus=<type>:4`) or several jobs at once both work (one shared queue). The job resubmits
itself until READY; if a shard fails 3 times it stops: see `runs/lang_collect/<root name>/logs` and `<ROOT>/logs`.

## 4. Smoke test (once, on a GPU node)

```bash
DATA=<data>/lang_1023/warmup VAL=<data>/lang_1023/val bash scripts/lang/bridges/smoke.sh
```

2 rounds of 2 episodes and 64 steps, stopped after 8 minutes, resumed, finished; ends with `[smoke] ALL OK` or names
what failed (log in `runs/bridges_logs/`). A `cuRobo fell back to mplib` warning means a robot-file path, TMPDIR or
GPU problem: fix it before the real run.

## 5. The runs

```bash
DATA_ROOT=<data>/lang_1023 bash scripts/lang/bridges/launch.sh vanilla      # or t1
QDATA_ROOT=<data>/lang_qwen_1023 bash scripts/lang/bridges/launch.sh qwen   # or t1qwen
```

Delta's 10-23 settings: ur5-wsg, gamma 0.95, softmin critic readout, MLP actor, 100 rounds x 4 episodes x 2048
steps, probes every 10 rounds. Run names `rounds_[t1_]ur5_[qwen_]g095_<TAG>` (TAG = MMDD by default).
Other settings: `rounds.sbatch` directly (`RUN`, `DATA`, `VAL`, `EXTRA_ARGS`, `SET="key=value ..."`, `ROUNDS`,
`EPISODES`, `STEPS`, `PROBE`; see its header).

- 100 rounds take ~85 min each on an A40 (67 min of training at ~0.5 steps/s, 15-20 min of collection): ~140
  GPU-hours, ~3 jobs of 48 h.
- Each job stops the trainer 10 minutes before its time limit (state saved, inside the round if needed) and submits
  the next job itself. A job that dies without that (node failure): submit `rounds.sbatch` again with the same `RUN`
  and variables; it resumes from the last saved point.
- Disk: a run grows to ~80-90 GB (`ring/`, sparse). The trainer reads ~80 MB per step at random from the data: keep
  the data on the fastest storage you have (`perf/data_wait_frac` in `train_log.jsonl` shows the waiting).
- Stop for good: `scancel <job>`, then `RESUBMIT=0` before any later `sbatch`.

A run directory records its data folders (absolute) and code commit when it starts; resuming refuses another data
location or commit (`--accept-code-change` admits a new commit). So a run continues on the machine and paths it
started with; a new machine means a new run.

## 6. Watching it

```bash
bash scripts/lang/bridges/status.sh runs/lang/<name>
```

| file (under the run) | what |
|---|---|
| `eval/trend.png` | per round: harmed on harm runs (should fall), intervened on benign runs (should stay low), task success, Q <= h |
| `eval/<metric>/roundNNN.png` | heatmaps task x (robot x harm / benign) |
| `eval/rounds.jsonl` | one row per filter episode |
| `ring/videos/*.mp4` | filter episodes: `<task>[_<T1 host>]_<robot>_s<seed>_<kind>_<outcome>.mp4` |
| `train_log.jsonl` | losses and diagnostics every 20 steps, round summaries |
| `probes/` | offline language probe every 10 rounds |
| `checkpoint.pt`, `checkpoint_round<N>.pt`, `checkpoint_final.pt` | weights |

Closed-loop evaluation of a checkpoint: `$PY -m main.lang.eval_closed --run <run> --ckpt checkpoint_final.pt
--scenes-per-combo 20 --axes train,heldout_template` (header of `main/lang/eval_closed.py`).
