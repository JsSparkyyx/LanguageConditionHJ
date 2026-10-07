"""Time one LangTrainer grad step by part (GPU), and the same step with TF32
matmuls (information only: precision changes are the human's decision).

    python scripts/lang/profile_step.py --data <roots...> --text-cache <cache> --out <json>
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from main.lang.config import LangConfig  # noqa: E402
from main.lang.train import LangTrainer, _t, roots_of  # noqa: E402


def timed(fn, n=1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--text-cache", nargs="+", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    cfg = LangConfig()
    tr = LangTrainer(cfg, Path(a.run), roots_of(a.data), (), [Path(c) for c in a.text_cache], use_wandb=False,
                     prefetch=False)
    res = {"gpu": torch.cuda.get_device_name(0)}
    b = tr.sampler.sample(cfg.batch_size)
    res["sample_s"], _ = timed(lambda: tr.sampler.sample(cfg.batch_size), 3)
    words, wmask = tr.text.lookup(b["sentence"])
    res["upload_s"], _ = timed(lambda: (_t(b["scene_tokens"]), _t(b["scene_tokens_next"])), 3)
    for _ in range(3):
        tr.grad_step(b)
    res["step_fp32_s"], _ = timed(lambda: tr.grad_step(b), a.steps)
    with torch.no_grad():
        res["encode_fwd_s"], enc = timed(lambda: tr.encode(b, words, wmask), 3)
        e, J, A, ji = enc
        res["critic_fwd_s"], _ = timed(lambda: tr.critics(e, _t(b["dtheta"]), J, A, ji), 3)
    def fwd_bwd():
        e, J, A, ji = tr.encode(b, words, wmask)
        q1, q2, _, _ = tr.critics(e, _t(b["dtheta"]), J, A, ji)
        (q1.square().mean() + q2.square().mean()).backward()
    res["critic_loss_fwd_bwd_s"], _ = timed(fwd_bwd, 3)
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for _ in range(2):
        tr.grad_step(b)
    res["step_tf32_s_info_only"], _ = timed(lambda: tr.grad_step(b), a.steps)
    torch.backends.cuda.matmul.allow_tf32 = False
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
