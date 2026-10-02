#!/bin/bash
#SBATCH --account=bifq-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=32g --time=00:45:00
#SBATCH --job-name=p5-determ
#SBATCH -o /work/nvme/bifq/project/tmp/logs/p5determ_%A.out

# Same-node determinism check — decides the Phase-5 regression fix design:
#   reruns BIT-IDENTICAL -> fix (a): regenerate the reference traces on a
#     designated node and pin the regression sbatch to it (#SBATCH --nodelist);
#   reruns DIVERGE       -> fix (b): redesign the regression comparison to be
#     robust (tolerance / structural invariants), justified by the measured
#     divergence scale.
#
# Runs regression_trace.py TWICE per embodiment in ONE job (same node, same
# GPU allocation, separate processes) on a pristine worktree at 14c6180
# (pre-Phase-5 — the exact code that generated the Phase-0 references),
# saves both traces under $OUT/run{1,2}/, then diffs element-wise at full
# precision: run1 vs run2 (the determinism verdict) and run1 vs the stored
# reference (the exact first-divergence indices for the test docstring).

set -e
WT=/work/nvme/bifq/project/wt_pre5_determ_${SLURM_JOB_ID}
OUT=/work/nvme/bifq/project/tmp/p5determ_${SLURM_JOB_ID}
cd /work/nvme/bifq/project/LanguageConditionHJ
git worktree add --detach "$WT" 14c6180
trap 'git worktree remove --force "$WT" || true' EXIT
mkdir -p "$OUT"
hostname | tee "$OUT/node.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt" || true

cat > "$OUT/diff.py" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
REF = Path("/work/nvme/bifq/project/LanguageConditionHJ/envs/docs/item_envs/regression")


def report(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        print(f"{name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    eq = a == b
    if eq.all():
        print(f"{name}: IDENTICAL ({a.size} elements)")
        return True
    idx = np.nonzero(~eq)[0]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    regions = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    rtxt = ", ".join(f"[{r[0]}..{r[-1]}]" for r in regions[:10])
    print(f"{name}: first divergence index {idx[0]}, {idx.size}/{a.size} "
          f"mismatched, max abs diff {d.max():.6f}\n  regions: {rtxt}")
    return False


verdict = True
for emb in ["ur5-wsg", "franka-panda"]:
    r1 = np.load(OUT / "run1" / emb / "trace.npz")
    r2 = np.load(OUT / "run2" / emb / "trace.npz")
    rf = np.load(REF / emb / "trace.npz")
    print(f"===== {emb} =====")
    verdict &= report("run1 vs run2: t       ", r1["t"], r2["t"])
    verdict &= report("run1 vs run2: h_scaled", r1["h_scaled"], r2["h_scaled"])
    report("run1 vs reference: t       ", r1["t"], rf["t"])
    report("run1 vs reference: h_scaled", r1["h_scaled"], rf["h_scaled"])

print("VERDICT:", "bit-identical same-node reruns -> fix (a) node-pinned reference"
      if verdict else "same-node reruns DIVERGE -> fix (b) robust comparison")
PYEOF

apptainer exec --nv --bind /work/hdd/bifq,/work/nvme/bifq,/projects/bifq \
  /work/nvme/bifq/robot-sim_cu128.sif env \
  TMP=/work/nvme/bifq/project/tmp TMPDIR=/work/nvme/bifq/project/tmp/tmp \
  HOLOBRAIN_CKPT=/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD \
  HF_HOME=/work/nvme/bifq/project/tmp/hf-cache XDG_CACHE_HOME=/work/nvme/bifq/project/tmp/xdg-cache \
  TRITON_CACHE_DIR=/work/nvme/bifq/project/tmp/triton-cache \
  TORCH_EXTENSIONS_DIR=/work/nvme/bifq/project/tmp/torch-ext WANDB_MODE=offline \
  bash -c "
    set -e
    cd $WT
    for run in 1 2; do
      for emb in ur5-wsg franka-panda; do
        /work/nvme/bifq/conda/envs/RoboTwin/bin/python \
          envs/scripts/item_envs/regression_trace.py \
          --embodiment \$emb --task place_empty_cup --seed 0 --out $OUT/run\$run
      done
    done
    /work/nvme/bifq/conda/envs/RoboTwin/bin/python $OUT/diff.py $OUT
  "
