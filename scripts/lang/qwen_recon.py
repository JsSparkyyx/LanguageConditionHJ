"""V6 GPU recon (qwen_impl.md 2, 10 step 2): HoloBrainQwenEncoder on the real HoloBrain-Qwen weights, no simulator.

For each attention kernel (sdpa, and flash_attention_2 if it builds), in the encoder's bf16:
  (a) parity: encode_text rows vs the instruction rows of one full-prompt pass (prefix + sentence + template end);
  (b) causality: the image-slot features of two full prompts with different sentences;
  (c) a batch of sentences of different lengths vs one batch per sentence;
  (d) the prefix ids vs the processor's own expansion of the full prompt, for several sentences;
  (e) the grid: 297 image slots at 15-113, 117-215, 219-317, the scene tokens in that order;
plus time and peak memory of encode_frame (3 images) and of an encode_text batch, the Qwen word norms against the
cached BERT words (data/lang/warmup/text_cache.pt; about 113 per token), and the K / V value range.

    scripts/lang/in_container.sh python scripts/lang/qwen_recon.py --out runs/lang/qwen_recon

Writes <out>/recon.json. Inputs: random and constant 308 x 252 frames, random depth in 0.3-1.0 m, an ego projection
at the D435 intrinsics scaled to 308 x 252.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SENTENCES = ["Don't touch the red mug.", "keep the beads in the bowl", "Keep the tall blue block upright.",
             "Make sure the wooden block stays where it is.", "3 cups must stay put", "do not knock over the can"]


def frames(F=1, seed=0):
    g = np.random.default_rng(seed)
    rgb = torch.from_numpy(g.integers(0, 256, (F, 3, 252, 308, 3), dtype=np.uint8))
    depth = torch.from_numpy(g.uniform(0.3, 1.0, (F, 3, 252, 308)).astype(np.float32))
    K = torch.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 288.0 * 308 / 320, 288.0 * 252 / 240, 154.0, 126.0
    return rgb, depth, K.expand(F, 3, 4, 4).clone(), torch.eye(4).expand(F, 4, 4).clone()


def rel(a, b):
    a, b = a.float(), b.float()
    return {"max_abs": float((a - b).abs().max()), "rel_mean": float((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12))}


@torch.no_grad()
def full_pass(enc, rgb, sentence):
    from main.lang import qwen_encoder as qe

    ids = torch.tensor(enc.prefix_ids.tolist() + enc.instruction_ids(sentence) + qe.TEMPLATE_END_IDS,
                       device=enc.device)[None]
    pv = enc.pixel_values(rgb.flatten(0, 1)).to(enc.vlm.model.visual.dtype)
    feats = enc.vlm.model.visual(pv, grid_thw=enc.grid_thw)
    emb = enc.embed_tokens(ids).clone()
    emb[0, ids[0] == qe.IMAGE_PAD_ID] = feats.to(emb.dtype)
    pos, _ = enc.vlm.model.get_rope_index(ids, image_grid_thw=enc.grid_thw, attention_mask=torch.ones_like(ids))
    out = enc.language_model(inputs_embeds=emb, position_ids=pos, output_hidden_states=True)
    return enc.feat_mapping(out.hidden_states)[0], ids[0]


def timed(fn, n=5):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return {"ms": (time.perf_counter() - t0) / n * 1000, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}


def checks(attn: str) -> dict:
    from main.lang import qwen_encoder as qe

    t0 = time.perf_counter()
    enc = qe.HoloBrainQwenEncoder(device="cuda", attn=attn)
    out = {"build_s": time.perf_counter() - t0, "ckpt_id": enc.ckpt_id,
           "weights_gb": torch.cuda.memory_allocated() / 2**30}
    rgb, depth, proj, T = frames(1, 0)
    rgb, depth, proj, T = (x.cuda() for x in (rgb, depth, proj, T))
    _, K, V = enc.prefix_pass(rgb)
    words, mask = enc.encode_text((K, V), SENTENCES)
    a = []
    for i, s in enumerate(SENTENCES):
        ref, _ = full_pass(enc, rgb, s)
        L = len(enc.instruction_ids(s))
        a.append({"sentence": s, "tokens": L, **rel(words[i, :L], ref[325:325 + L])})
    out["a_parity"] = a
    f1, ids = full_pass(enc, rgb, SENTENCES[0])
    f2, _ = full_pass(enc, rgb, SENTENCES[2])
    img = ids[:325] == qe.IMAGE_PAD_ID
    hs, _, _ = enc.prefix_pass(rgb)
    pre = enc.feat_mapping(hs)[0]
    out["b_causality"] = {"two_sentences": rel(f1[:325][img], f2[:325][img]),
                          "prefix_pass_vs_full": rel(pre[img], f1[:325][img])}
    c = []
    for i, s in enumerate(SENTENCES):
        w1, m1 = enc.encode_text((K, V), [s])
        c.append({"sentence": s, "mask_equal": bool(torch.equal(m1[0], mask[i])), **rel(w1[0], words[i])})
    out["c_batch"] = c
    black = [np.zeros((252, 308, 3), np.uint8)] * 3
    d = []
    for s in SENTENCES + [""]:
        got = enc.processor(text=[qe.full_prompt(s)], images=black, return_tensors="pt")["input_ids"][0].tolist()
        d.append({"sentence": s, "prefix_equal": got[:325] == enc.prefix_ids.tolist(),
                  "tail_equal": (got[325:] == enc.instruction_ids(s) + qe.TEMPLATE_END_IDS) if s else None})
    out["d_prefix_ids"] = d
    slots = torch.nonzero(enc.image_mask).flatten().tolist()
    seq = torch.arange(325, dtype=torch.float32, device="cuda").view(1, 325, 1)
    toks = enc.image_maps(seq).flatten(3).transpose(2, 3)[0, :, :, 0]
    out["e_grid"] = {"slots_ok": slots == [i for c_ in range(3) for i in range(15 + 102 * c_, 114 + 102 * c_)],
                     "token_order_ok": toks.flatten().long().tolist() == slots}
    tokens, pos, (K1, V1) = enc.encode_frame(rgb, depth, proj, T)
    out["frame"] = {"tokens_norm_mean": float(tokens.norm(dim=-1).mean()), "pos_mean": pos.mean((0, 1, 2)).tolist(),
                    "pos_std": pos.std((0, 1, 2)).tolist(), "finite": bool(torch.isfinite(tokens).all()),
                    "fp16_safe": bool(tokens.abs().max() < 6e4), "k_absmax": float(K1.float().abs().max()),
                    "v_absmax": float(V1.float().abs().max())}
    out["time_encode_frame"] = timed(lambda: enc.encode_frame(rgb, depth, proj, T))
    sents = [SENTENCES[i % len(SENTENCES)] for i in range(128)]
    K128, V128 = K.expand(128, -1, -1, -1).contiguous(), V.expand(128, -1, -1, -1).contiguous()
    out["time_encode_text_128"] = timed(lambda: enc.encode_text((K128, V128), sents))
    out["time_encode_text_2"] = timed(lambda: enc.encode_text((K, V), SENTENCES[:2]))
    wn = words.norm(dim=-1)[mask]
    out["word_norm"] = {"qwen_mean": float(wn.mean()), "qwen_std": float(wn.std())}
    cache = REPO / "data" / "lang" / "warmup" / "text_cache.pt"
    if cache.exists():
        tc = torch.load(cache, map_location="cpu", weights_only=False)
        bn = tc["emb"][:2000].float().norm(dim=-1)[tc["mask"][:2000]]
        out["word_norm"]["bert_mean"] = float(bn.mean())
    blank = enc.blank_kv()
    wb, mb = enc.encode_text(blank, SENTENCES[:2])
    out["blank_vs_frame_words"] = rel(wb[0][mb[0]], words[0][mask[0]])
    del enc
    torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "runs" / "lang" / "qwen_recon"))
    ap.add_argument("--attn", nargs="*", default=["sdpa", "flash_attention_2"])
    args = ap.parse_args(argv)
    import main.lang  # noqa: F401

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    import transformers

    res["transformers"] = transformers.__version__
    for attn in args.attn:
        try:
            res[attn] = checks(attn)
        except Exception as exc:   # flash-attn may not build here: record why, keep the other kernel's numbers
            import traceback

            res[attn] = {"error": f"{type(exc).__name__}: {exc}"[:2000], "trace": traceback.format_exc()[-3000:]}
        (out_dir / "recon.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({attn: {k: v for k, v in res[attn].items() if k not in ("trace",)}}, indent=1)[:6000],
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
