"""Save a Qwen2.5-VL-architecture 32B model with 8-bit (FP8) language-model weights, for a vLLM labeler on ONE A40
(Yuxuan 2026-10-28: the 2-A40 training job, one GPU trains, the other labels).

    python scripts/lang/vlm/to_fp8.py <src dir> <out dir> [--device cuda] [--check-shard N]

Why a saved copy: vLLM's own FP8 (quantization="fp8", the robobrain2_32b_fp8 entry) loads the bf16 model first
(~64 GB) and only then converts it, which fits on two 40 GB A100s but not on one 48 GB A40. This writes what
llm-compressor's FP8_DYNAMIC scheme writes (compressed-tensors "float-quantized"), one shard at a time (a few GB of
memory, no whole model): every language-model Linear weight (q/k/v/o, gate/up/down of model.layers.*) becomes
float8_e4m3fn with one bf16 scale per output channel,

    scale[o] = max_i |W[o, i]| / 448          (448 = the largest float8_e4m3fn)
    Wq[o, i] = float8(W[o, i] / scale[o])     (the stored scale, bf16, is the one divided by)

and the activations are quantized per token at run time (dynamic). On an A40 / A100 (no FP8 matmul) vLLM runs the
weight-only W8A16 kernels. Kept in bf16: the vision tower (visual.*, its merger too), the embeddings, lm_head and
every norm and bias. RoboBrain2.0-32B is a Qwen2.5-VL-32B fine-tune: the same tensor names.
``--check-shard N`` converts shard N only into <out dir> (a format check, nothing else written).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)   # 448
QUANTIZED = re.compile(r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
QUANT_CONFIG = {
    "config_groups": {"group_0": {
        "input_activations": {"actorder": None, "block_structure": None, "dynamic": True, "group_size": None,
                              "num_bits": 8, "observer": None, "observer_kwargs": {}, "strategy": "token",
                              "symmetric": True, "type": "float"},
        "output_activations": None,
        "targets": ["Linear"],
        "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": None, "num_bits": 8,
                    "observer": "minmax", "observer_kwargs": {}, "strategy": "channel", "symmetric": True,
                    "type": "float"}}},
    "format": "float-quantized",
    "global_compression_ratio": None,
    "ignore": ["lm_head", "re:visual.*"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def quantize(w: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(float8 weight, bf16 scale [out, 1]) of one [out, in] weight, on ``device``, returned on the CPU."""
    x = w.to(device=device, dtype=torch.float32)
    scale = (x.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=torch.finfo(torch.float32).eps)
    scale = scale.to(torch.bfloat16)
    q = (x / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.cpu(), scale.cpu()


def convert_shard(src: Path, dst: Path, device: str) -> tuple[dict, int, int]:
    """(tensor name -> shard file name, bytes written, tensors quantized)."""
    out, n_q = {}, 0
    with safe_open(str(src), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if QUANTIZED.match(k):
                q, s = quantize(t, device)
                out[k], out[k + "_scale"] = q, s
                n_q += 1
            else:
                out[k] = t
    save_file(out, str(dst), metadata={"format": "pt"})
    size = sum(t.numel() * t.element_size() for t in out.values())
    return {k: dst.name for k in out}, size, n_q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check-shard", type=int, default=None)
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    out.mkdir(parents=True, exist_ok=True)
    if args.check_shard is not None:
        name = shards[args.check_shard]
        _, size, n_q = convert_shard(src / name, out / name, args.device)
        print(f"[to_fp8] check: {name} -> {out / name}: {n_q} weights quantized, {size / 1e9:.2f} GB")
        return 0
    if (out / "config.json").exists():
        print(f"[to_fp8] {out} is complete (config.json written last): nothing to do")
        return 0
    weight_map, total, n_all = {}, 0, 0
    t0 = time.time()
    for i, name in enumerate(shards):
        wm, size, n_q = convert_shard(src / name, out / name, args.device)
        weight_map.update(wm)
        total += size
        n_all += n_q
        print(f"[to_fp8] {i + 1}/{len(shards)} {name}: {n_q} quantized, {size / 1e9:.2f} GB ({time.time() - t0:.0f}s)",
              flush=True)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    for p in src.iterdir():   # tokenizer, processor, chat template, generation config, README
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in ("model.safetensors.index.json",
                                                                                   "config.json"):
            shutil.copy2(p, out / p.name)
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = QUANT_CONFIG
    (out / "config.json").write_text(json.dumps(cfg, indent=2))   # last: marks the copy complete
    print(f"[to_fp8] {src.name} -> {out}: {n_all} weights quantized, {total / 1e9:.1f} GB, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
