#!/bin/bash
# Smoke test T (inside the container, one GPU, under 1 h): pack the smoke nominal demos, 30 training steps
# of HoloBrain's train.py from post_training_robotwin, export, and reload the exported model directory.
#   scripts/lang/in_container.sh bash vla_finetuning/jobs/smoke_t.sh
set -uo pipefail
VF=/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning
S=$VF/data/smoke
cd /work/nvme/bifq/project/LanguageConditionHJ
step() { echo; echo "=== $1 $(date +%T)"; }
step pack
rm -rf "$S/lmdb_nominal"
python -m vla_finetuning.pack --root $S/demos_nominal --out $S/lmdb_nominal || exit 1
step train
rm -rf "$VF/ws/smoke_t" "$VF/models/smoke_t"
bash vla_finetuning/jobs/train.sh smoke_t $S/lmdb_nominal 30 30 1 || exit 1
ls -R "$VF/ws/smoke_t/checkpoints" | head -20
step export
CK=$(ls -d $VF/ws/smoke_t/checkpoints/checkpoint_* | sort -V | tail -1)
bash vla_finetuning/jobs/export.sh smoke_t "$CK/model/model.safetensors" || exit 1
step reload
python - <<'EOF'
import sys
from pathlib import Path
from vla_finetuning import setup_models
setup_models.check_load(Path("/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/models/smoke_t/model"))
import torch
from safetensors.torch import load_file
a = load_file("/work/nvme/bifq/project/LanguageConditionHJ/vla_finetuning/models/smoke_t/model/model.safetensors")
b = load_file(str(setup_models.SRC / "model.safetensors"))
same = [k for k in a if k in b and torch.equal(a[k], b[k])]
txt = [k for k in a if k.startswith("text_encoder.")]
print(f"[smoke_t] {len(a)} tensors, {len(same)} unchanged vs post_training_robotwin; text_encoder "
      f"{sum(k in same for k in txt)}/{len(txt)} unchanged; missing in export {len(set(b) - set(a))}")
EOF
step done
