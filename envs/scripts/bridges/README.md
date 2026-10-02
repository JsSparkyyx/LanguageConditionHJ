# Bridges runbook — item_envs verification + calibration

All remaining GPU work for `docs/item_envs/PLAN.md` runs here. The scripts are
plain bash (no Slurm); run them on a GPU node in a terminal (or adapt to
Bridges Slurm). Everything writes under `envs/runs/` (gitignored) — no code
changes are needed on Bridges, so the Delta repo stays the single source of
truth.

## 0. What you need

- The repo at the handoff commit (recorded at the bottom of this file).
- **RoboTwin checkout WITH assets** at `<repo>/../RoboTwin` (or set
  `ROBOTWIN_ROOT`). The Delta tree is only **~20 GB** (the old 1.3 TB was
  core dumps, now deleted), so a plain rsync is fine:
  `rsync -a --exclude '.git' <delta>:/work/nvme/bifq/project/RoboTwin/ ./RoboTwin/`
  If you already have a working RoboTwin with assets on Bridges, just point
  `ROBOTWIN_ROOT` at it; the item tasks need the standard RoboTwin asset set
  (objects, robots ur5-wsg + franka-panda, backgrounds). Do NOT let our
  symlink installer touch anything but `RoboTwin/envs/*.py` (it only adds
  symlinks).
- A python env matching Delta's `RoboTwin` conda env: sapien **3.0.0b1**,
  cuRobo, CUDA torch. Reference package list: `pip-freeze.txt` next to this
  file. RT-capable NVIDIA GPU (A40/A100 class; V100 untested).
- Four **editable** installs (Delta paths won't exist on Bridges — recreate):
  ```bash
  pip install -e $ROBOTWIN_ROOT/envs/curobo        # nvidia_curobo 0.7.8 (vendored)
  pip install -e $ROBOTWIN_ROOT/XPolicyLab         # xpolicylab 0.0.1 (vendored)
  git clone https://github.com/HorizonRobotics/robo_orchard_core && cd robo_orchard_core && git checkout 6aa3eaa && pip install -e . && cd ..
  git clone https://github.com/HorizonRobotics/RoboOrchardLab && cd RoboOrchardLab && git checkout 241bbc6 && pip install -e . && cd ..
  ```
  (the robo_orchard pair is only needed by the HoloBrain-encoder paths:
  phase5_verify and later collection — but install them anyway).
- The HoloBrain checkpoint (~2.1 GB): rsync from Delta
  `/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD` → set
  `HOLOBRAIN_CKPT` to the local dir (must contain `model.config.json`).

## 1. Setup

```bash
git clone <repo> LanguageConditionHJ && cd LanguageConditionHJ
git checkout <HANDOFF_COMMIT>
export PY=<your python> HOLOBRAIN_CKPT=<path>/HoloBrain_v0.0_GD
# export ROBOTWIN_ROOT=<path>/RoboTwin   # only if not <repo>/../RoboTwin
bash envs/scripts/bridges/setup_check.sh   # installs task symlinks too
```

## 2. Run order

```bash
bash envs/scripts/bridges/run_accept.sh          # V2: Phase-4 acceptance   (~40-60 min)
bash envs/scripts/bridges/run_phase5_verify.sh   # V3: Phase-5 verify       (~30 min)
bash envs/scripts/bridges/run_f5b_validate.sh    # V1: F5b visibility gate  (~15 min)
bash envs/scripts/bridges/run_calib_sweeps.sh    # V4: 14 calib sweeps      (~10-20 GPU-h)
```

- Logs land in `envs/runs/bridges_logs/`.
- `run_calib_sweeps.sh` supports sharding: `SHARD=i/n` per terminal/GPU
  (e.g. `SHARD=0/4`..`SHARD=3/4` with distinct `CUDA_VISIBLE_DEVICES`).
  **All sweeps must run on the same GPU model** — cuRobo/PhysX results differ
  at the cm level across hardware (measured on Delta; the regression suite is
  nondeterminism-hardened for exactly this reason).
- `run_phase5_verify.sh` can flake with a *chaos signature* (pair-count/onset
  stats in the assert) — rerun once; only structural failures (ALL pairs
  diverge at tick 0/1, lockstep errors) are real bugs.

## 3. Sync back to Delta

Copy these back (all small, meta/logs only — no videos/traces needed):

- `envs/runs/bridges_logs/*.log`
- `envs/main/item_envs/configs/validate/visibility_rows_*.jsonl` (F5b rows)
- `envs/runs/item_envs/calib/{T1,T2,T3}/**/meta.json` (preserve the tree)

The Delta side then runs `calibration_report.py`, writes `risk_collect` +
`calibration.md`, renders the P4-gate verdict, and closes with
`final_report.md`.

## Handoff commit

Clone/checkout **latest `main`** (the handoff was made at the commit that
added this kit; the Delta side will tell you the exact hash). The code
state being handed off is **`d69c1e0`** — `item_envs(phase3): F5b rework
after ur5 A/B regression` (fused covered-preference spawn pass; validate
rows instrumented with placement fields). Non-sim suite state at handoff:
317 passed (`pytest tests/item_envs -m "not sim and not gpu"` from `envs/`).

F5b gate expectation on Bridges (per-class frac(vis_max >= 0.3) >= 0.95):
off_path PASS (proven 0.995); near_path / on_path are coin-flips near the
gate — sibling occlusion of tall thin items, not mask coverage, is the
suspected residual mechanism (see `docs/item_envs/task_cards/T1.md`).
