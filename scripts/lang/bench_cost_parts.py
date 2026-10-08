"""Where the language model's training step cost comes from: the grad step with its extra parts removed one
at a time, against CrossSafe (timing only; the variants are never trained).

    python scripts/lang/bench_cost_parts.py --data data/lang/warmup/shard_g0 \
        --text-cache data/lang/warmup/text_cache.pt --out runs/lang_bench/<gpu>_parts.json

Variants of the language model (B = 64, fp32, one fixed batch, grad_step without the every-20th-step
diagnostics): as trained; without the readout queries (the scene is then read once per block, not twice);
frame t only (1200 scene tokens instead of 3600; built with network.N_FRAMES = 1 for this timing); both;
both and every text path off. CrossSafe: its grad step without its per-step diagnostics (the replica in
bench_latency.py). Median ms over --iters steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import bench_latency as BL  # noqa: E402

OFF = dict(lang_scene=False, lang_body=False, lang_film=False, lang_rtoken=False)
VARIANTS = [   # (label, frames, LangConfig overrides)
    ("language model, as trained (3 frames = 3600 scene tokens, 4 readout queries, all text paths)", 3, {}),
    ("- readout queries (the scene is read once per block)", 3, {"lang_queries": False}),
    ("- frames t-2 and 0 (frame t only = 1200 scene tokens)", 1, {}),
    ("- both", 1, {"lang_queries": False}),
    ("- both, - every text path (scene/body read words, FiLM, rule token)", 1, {"lang_queries": False, **OFF}),
]


def one_frame(b: dict) -> dict:
    b = dict(b)
    for k in ("scene_tokens", "scene_pos", "scene_tokens_next", "scene_pos_next"):
        if k in b:
            b[k] = np.ascontiguousarray(np.asarray(b[k])[:, :1])
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-cache", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    import main.lang  # noqa: F401
    from main.lang import network
    from main.lang.config import LangConfig
    from main.lang.train import LangTrainer
    from main.train.config import FrozenConfig
    from main.train.train import Trainer

    torch.backends.cuda.matmul.allow_tf32 = False   # PyTorch's default, as the trainers run
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / f"_{out.stem}_scratch"
    shard = Path(args.data).resolve()
    caches = [Path(c) for c in args.text_cache]
    res = {"gpu": torch.cuda.get_device_name(0), "iters": args.iters, "rows": []}
    batch = None
    for label, frames, kw in VARIANTS:
        network.N_FRAMES = frames
        cfg = dataclasses.replace(LangConfig(), embodiment_choices=("ur5-wsg",), **kw)
        tr = LangTrainer(cfg, scratch / f"lang_{len(res['rows'])}", [shard], (), caches, embodiments=("ur5-wsg",),
                         use_wandb=False, prefetch=False, seed=0)
        if batch is None:
            batch = tr.sampler.sample(cfg.batch_size)
        b = batch if frames == 3 else one_frame(batch)
        for m in (tr.policy_enc, tr.critics, tr.actor):
            m.train()
        torch.cuda.reset_peak_memory_stats()
        ms = BL.bench(lambda: tr.grad_step(b, log=False), args.iters, args.warmup)
        res["rows"].append({"variant": label, "step_ms": ms, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                            "params_m": BL.n_params(tr.policy_enc, tr.critics, tr.actor) / 1e6})
        print(res["rows"][-1], flush=True)
        del tr
        torch.cuda.empty_cache()
    network.N_FRAMES = 3
    ccfg = FrozenConfig()
    ctr = Trainer(ccfg, shard, scratch / "crosssafe", buffer_dir=shard / "buffer")
    cb = ctr.buf.sample(ccfg.batch_size, ctr.rng)
    torch.cuda.reset_peak_memory_stats()
    ms = BL.bench(lambda: BL.crosssafe_step_replica(ctr, cb, BL.Phases()), args.iters, args.warmup)
    res["rows"].append({"variant": "CrossSafe (1200 scene tokens; its grad step without its per-step diagnostics)",
                        "step_ms": ms, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                        "params_m": BL.n_params(ctr.injection, ctr.trunk, ctr.critics, ctr.actor) / 1e6})
    print(res["rows"][-1], flush=True)
    cs = res["rows"][-1]["step_ms"]
    L = [f"# Training step cost by part ({res['gpu']}, fp32, B = 64)", "",
         "| variant | step (ms) | x CrossSafe | peak memory (GB) | parameters |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['variant']} | {r['step_ms']:.0f} | {r['step_ms'] / cs:.2f}x | {r['peak_gb']:.1f} | "
                 f"{r['params_m']:.1f} M |")
    md = "\n".join(L) + "\n"
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
