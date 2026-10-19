"""One-time setup (needs torch: run in a GPU job inside the container).

1. vla_finetuning/ckpt/bert-base-uncased/: the BERT that HoloBrain's text encoder loads by name. There is
   no internet download here, so it is assembled from the checkpoint's own tensors
   (text_encoder.language_backbone.body.model.*, identical in pretrain/ and post_training_robotwin/)
   plus the bert-base-uncased config and tokenizer files in the local HF cache, then verified by
   loading it with transformers and comparing every tensor.
2. vla_finetuning/models/base_post_training_robotwin/: a directory load_pipeline can open for the
   base model (HoloBrain resolves ./urdf and ./ckpt relative to the model directory, and the shipped
   post_training_robotwin/ has neither): symlinks to the weights, configs, URDFs and the BERT above.

    scripts/lang/in_container.sh python -m vla_finetuning.setup_models
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import common as C

HF_SNAPSHOT = Path("/work/nvme/bifq/project/tmp/hf-cache/hub/models--bert-base-uncased/snapshots")
PREFIX = "text_encoder.language_backbone.body.model."
BERT_DIR = C.VF / "ckpt" / "bert-base-uncased"
BASE_DIR = C.VF / "models" / "base_post_training_robotwin"
SRC = C.CK / "post_training_robotwin"


def build_bert() -> None:
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import BertModel

    snaps = sorted(p for p in HF_SNAPSHOT.iterdir() if p.is_dir())
    if not snaps:
        raise SystemExit(f"no bert-base-uncased snapshot under {HF_SNAPSHOT}")
    snap = snaps[-1]
    BERT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        shutil.copyfile(snap / name, BERT_DIR / name)          # follows the blob symlinks
    sd = load_file(str(SRC / "model.safetensors"))
    bert = {k[len(PREFIX):]: v.contiguous() for k, v in sd.items() if k.startswith(PREFIX)}
    if len(bert) < 190:
        raise SystemExit(f"only {len(bert)} BERT tensors under {PREFIX}")
    save_file(bert, str(BERT_DIR / "model.safetensors"), metadata={"format": "pt"})
    m = BertModel.from_pretrained(str(BERT_DIR), add_pooling_layer=False)
    got = m.state_dict()
    missing = [k for k in bert if k not in got]
    unused = [k for k in got if k not in bert and not k.endswith("position_ids")]
    diff = [k for k in bert if k in got and not torch.equal(got[k].cpu(), bert[k].cpu())]
    if missing or unused or diff:
        raise SystemExit(f"BERT check failed: missing {missing[:5]}, not loaded {unused[:5]}, differ {diff[:5]}")
    print(f"[setup] {BERT_DIR}: {len(bert)} tensors from {SRC.name}, loaded and equal")


def _link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        if dst.is_symlink():
            dst.unlink()
        else:
            raise SystemExit(f"{dst} exists and is not a symlink")
    dst.symlink_to(src)


def assemble_base() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "model.config.json", f"{C.INFERENCE_PREFIX}.config.json"):
        _link(BASE_DIR / name, SRC / name)
    _link(BASE_DIR / "urdf", C.CK / "urdf")
    (BASE_DIR / "ckpt").mkdir(exist_ok=True)
    _link(BASE_DIR / "ckpt" / "bert-base-uncased", BERT_DIR)
    print(f"[setup] {BASE_DIR}: base model directory")


def check_load(model_dir: Path) -> None:
    from robo_orchard_lab.models.holobrain.pipeline import HoloBrainInferencePipeline

    pipe = HoloBrainInferencePipeline.load_pipeline(directory=str(model_dir), inference_prefix=C.INFERENCE_PREFIX,
                                                    device="cuda", load_weights=True, load_impl="native")
    n = sum(p.numel() for p in pipe.model.parameters())
    print(f"[setup] load_pipeline({model_dir.name}) ok: {n / 1e6:.1f} M parameters")


def main() -> int:
    build_bert()
    assemble_base()
    check_load(BASE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
