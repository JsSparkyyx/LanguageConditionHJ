"""Cost and precision of the frozen fusion stage (FUSION_PLAN 3.3, fusion_impl.md 4.1) on one GPU.

    python scripts/lang/bench_fusion.py --synthetic --out runs/lang_bench/fusion_<gpu>.json
    python scripts/lang/bench_fusion.py --data data/lang/val_v2/shard_g0 \
        --text-cache data/lang/val_grounding/text_cache.pt --out runs/lang_bench/fusion_<gpu>_data.json
    python scripts/lang/bench_fusion.py --data data/lang_v2/warmup/shard_g0 --train-steps 50 \
        --text-cache data/lang/warmup/text_cache.pt --out runs/lang_bench/fusion_<gpu>_train.json

- Speed and memory: ms per frame-pass of ``FusionStage.fuse`` at chunks 8 / 16 / 32 / 64, dtypes fp32 / tf32 / bf16,
  the low-memory deformable attention (``msda_lowmem``) on and off. Every frame gets the same sentence length here,
  so the chunk is the only variable (one enhancer call per chunk); one more row per (lowmem, dtype) fuses the batch's
  mixed lengths at chunk 32, as training does. ``unfused`` (the V0 path) for reference. Median over --iters after
  --warmup, synchronized. Peak = max allocated during the call minus allocated before (outputs included).
  ``step_ms_320`` = ms per frame x 320, the 5 x 64 frame-passes of one training step: the fusion share only, an
  upper bound (before the trainer's dedup of repeated (frame, sentence) pairs), not the step time.
- Precision: tf32 / bf16 (and lowmem off) against fp32 lowmem on the same frames and sentences: max / mean abs and
  relative differences of the tokens, positions (cm) and fused words (valid positions only), next to the scale of the
  sentence's own effect (the same frames fused with other sentences).
- With --data (a version-2 shard: buffer/, item_buffer/, neck_buffer/): stored-row parity, ``unfused`` on neck rows
  vs the StepBuffer's stored scene_tokens / scene_pos on --parity-rows random rows (R6; the neck rows are fp16, so the
  bound is measured, not 1e-5), the neck rows' episode_id / t_in_ep against the item sidecar, and the time to read
  and upload a training step's frames.
- With --train-steps N (needs --data, a version-2 shard; only this phase runs): the end-to-end training step,
  the input to FUSION_PLAN 3.3's "more than doubles" decision. LangTrainer is built on the shard twice, with
  scene_fusion off (V0') and on (V5), the same config otherwise (LangConfig defaults, embodiment ur5-wsg, plus
  --train-set key=value overrides with JSON values, as train.py --set), and times N grad_steps after
  --train-warmup: the trainer's own CUDA-timed perf/step_ms and perf/fusion_ms, the frames fused per step, the
  synchronous batch draw (sample_ms: the host gather, neck rows included, that the PrefetchSampler hides behind
  the previous step) and the wall time of draw + step. step_ratio = step_ms on / off; > 2 means the step more
  than doubles. A cross-check from two short training runs, perf/step_ms and perf/fusion_ms of their
  <run>/train_log.jsonl (every 20 steps):
      python -m main.lang.train offline --data data/lang_v2/warmup --run runs/lang_bench/smoke_v0p \
          --tasks selective_avoid,gap_place --grad-steps 200 --probe-every 0 --ckpt-every 0 --no-wandb
      python -m main.lang.train offline --data data/lang_v2/warmup --run runs/lang_bench/smoke_v5 \
          --tasks selective_avoid,gap_place --grad-steps 200 --probe-every 0 --ckpt-every 0 --no-wandb \
          --set scene_fusion=true

Words come from --text-cache files (rows drawn at random) or, without one, random N(0, 1) words with lengths 5-18.
Writes <out>.json and the tables as <out>.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import main.lang  # noqa: E402,F401  (main.__path__ bootstrap)
from main.lang import fusion as fz  # noqa: E402

DEFAULT_CKPT = os.environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD")
STEP_FRAMES = 320  # 5 frames (t, t-k, 0, t+1, t+1-k) x batch 64
TIMING_LEN = 12    # sentence length of the speed sweep (the cache's lengths run 5-18)
DEVICE = "cuda"
TEXT_ENCODER = None  # tests inject a stub text encoder (--train-steps; None: the trainer's HoloBrain BERT)
STEP_MS_320_NOTE = ("ms per frame x 320 (5 frames x batch 64): the fusion share only (upper bound, before dedup of "
                    "repeated (frame, sentence) pairs), not the training step time; the step time on / off is "
                    "--train-steps")


def timed(fn, iters: int, warmup: int) -> float:
    """Median wall ms of fn(), CUDA-synchronized."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))


def peak_mb(fn) -> float:
    """Peak memory allocated by one fn() call above what was allocated before it, MB."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def guarded(fn) -> dict:
    """fn() -> its dict, or {"error": ...} (an OOM at one configuration must not end the sweep)."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        return {"error": "oom"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        torch.cuda.empty_cache()


def diff(a, b, scale: float = 1.0, mask=None) -> dict:
    """a vs reference b: max / mean abs (x scale), max abs / max |b|, ||a - b|| / ||b||."""
    if mask is not None:
        a, b = a[mask], b[mask]
    d = (a - b).abs()
    return {
        "max_abs": scale * float(d.max()), "mean_abs": scale * float(d.mean()),
        "rel_max": float(d.max() / b.abs().max().clamp_min(1e-12)),
        "rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
    }


# ----------------------------------------------------------------- inputs --


def load_words(paths, n: int, rng, device):
    """n rows of the text caches -> (words [n, 24, 256] fp32, mask [n, 24] bool, sentences); without caches random
    words with lengths 5-18."""
    if paths:
        embs, masks, sents = [], [], []
        for p in paths:
            c = torch.load(p, map_location="cpu", weights_only=False)
            inv = {i: s for s, i in c["index"].items()}
            embs.append(c["emb"])
            masks.append(c["mask"])
            sents += [inv.get(i, "") for i in range(c["emb"].shape[0])]
        emb, mask = torch.cat(embs), torch.cat(masks)
        rows = rng.choice(emb.shape[0], n, replace=emb.shape[0] < n)
        return emb[rows].float().to(device), mask[rows].bool().to(device), [sents[i] for i in rows]
    lengths = rng.integers(5, 19, n)
    g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
    w = torch.randn(n, 24, 256, generator=g)
    m = torch.arange(24)[None] < torch.as_tensor(lengths)[:, None]
    return (w * m[..., None]).to(device), m.to(device), [f"<random {k}>" for k in lengths]


def fixed_length(mask, n: int):
    """The same prefix mask of length n for every frame (timing only: the words keep their values)."""
    return (torch.arange(mask.shape[1], device=mask.device)[None] < n).expand_as(mask).contiguous()


def open_neck(root: Path):
    from main.lang.neck_buffer import NeckBuffer  # package C; only --data needs it

    nb = NeckBuffer(root / "neck_buffer")
    n_valid = min(int(nb.n), int(nb.capacity))
    if n_valid == 0:
        raise SystemExit(f"{root}/neck_buffer is empty")
    return nb, n_valid


def neck_frames(args, n: int, rng, device):
    if args.synthetic:
        return fz.synthetic_neck(n, seed=int(rng.integers(1 << 31)), device=device), None
    nb, n_valid = open_neck(Path(args.data))
    rows = np.sort(rng.choice(n_valid, n, replace=n_valid < n))
    return fz.rows_to_neck(nb.read(rows), device), rows


# ----------------------------------------------------------------- phases --


def speed(stages, neck, words, mask, args) -> list[dict]:
    F = int(words.shape[0])
    m_fixed = fixed_length(mask, TIMING_LEN)
    out = []
    for lowmem, st in stages.items():
        for dt in args.dtypes:
            st.dtype = dt
            for chunk in args.chunks:
                st.chunk = chunk

                def run(m=m_fixed):
                    return st.fuse(neck, words, m)

                def one(chunk=chunk):
                    ms = timed(run, args.iters, args.warmup)
                    pk = peak_mb(run)
                    return {"ms_per_frame": ms / F, "step_ms_320": ms / F * STEP_FRAMES, "peak_mb": pk,
                            "peak_mb_per_frame": pk / min(chunk, F)}

                row = {"lowmem": lowmem, "dtype": dt, "chunk": chunk, "lengths": f"all {TIMING_LEN}", **guarded(one)}
                out.append(row)
                print(json.dumps(row), flush=True)
            st.chunk = 32

            def mixed():
                ms = timed(lambda: st.fuse(neck, words, mask), args.iters, args.warmup)
                return {"ms_per_frame": ms / F, "step_ms_320": ms / F * STEP_FRAMES,
                        "peak_mb": peak_mb(lambda: st.fuse(neck, words, mask))}

            n_len = len(set(mask.sum(1).tolist()))
            row = {"lowmem": lowmem, "dtype": dt, "chunk": 32, "lengths": f"mixed ({n_len} lengths)", **guarded(mixed)}
            out.append(row)
            print(json.dumps(row), flush=True)
            st.dtype = "fp32"
    return out


def speed_unfused(st, neck, args) -> list[dict]:
    F = int(neck["neck_maps"].shape[0])
    out = []
    for dt in args.dtypes:
        st.dtype, st.chunk = dt, 32

        def one():
            ms = timed(lambda: st.unfused(neck), args.iters, args.warmup)
            return {"ms_per_frame": ms / F, "peak_mb": peak_mb(lambda: st.unfused(neck))}

        out.append({"dtype": dt, "chunk": 32, **guarded(one)})
        print(json.dumps(out[-1]), flush=True)
    st.dtype = "fp32"
    return out


def precision(stages, neck, words, mask, words2, mask2) -> dict:
    """Everything against fp32 with the low-memory attention; the sentence effect as the scale to read them by."""
    ref_st = stages[True]
    ref_st.dtype, ref_st.chunk = "fp32", 32
    t0, p0, w0 = ref_st.fuse(neck, words, mask)
    wm = mask[..., None].expand_as(w0)
    res = {}
    for name, st, dt in [("tf32", stages[True], "tf32"), ("bf16", stages[True], "bf16"),
                         ("fp32 lowmem off", stages.get(False), "fp32")]:
        if st is None:
            continue

        def one(st=st, dt=dt):
            st.dtype = dt
            t, p, w = st.fuse(neck, words, mask)
            st.dtype = "fp32"
            return {"tokens": diff(t, t0), "pos_cm": diff(p, p0, 100.0), "w_fused": diff(w, w0, mask=wm)}

        res[name] = guarded(one)
    t2, p2, _ = ref_st.fuse(neck, words2, mask2)
    tu, pu = ref_st.unfused(neck)
    res["scale: other sentence (fp32)"] = {"tokens": diff(t2, t0), "pos_cm": diff(p2, p0, 100.0)}
    res["scale: unfused vs fused (fp32)"] = {"tokens": diff(tu, t0), "pos_cm": diff(pu, p0, 100.0)}
    ref_st.dtype = "tf32"
    tu32, pu32 = ref_st.unfused(neck)
    ref_st.dtype = "fp32"
    res["unfused tf32"] = {"tokens": diff(tu32, tu), "pos_cm": diff(pu32, pu, 100.0)}
    return res


def parity(st, args, rng) -> dict:
    """R6 on stored rows: unfused(neck row) vs the StepBuffer row it was written with, plus the lockstep check."""
    root = Path(args.data)
    nb, n_valid = open_neck(root)
    step_n = json.loads((root / "buffer" / "header.json").read_text()).get("n")
    rows = np.sort(rng.choice(n_valid, min(args.parity_rows, n_valid), replace=False))
    arrs = nb.read(rows)
    st.dtype, st.chunk = "fp32", 32
    tok, pos = st.unfused(fz.rows_to_neck(arrs, DEVICE))
    s_tok = torch.from_numpy(np.load(root / "buffer" / "scene_tokens.npy", mmap_mode="r")[rows]).to(DEVICE).float()
    s_pos = torch.from_numpy(np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")[rows]).to(DEVICE).float()
    per_row = (tok - s_tok).abs().flatten(1).max(1).values.cpu().numpy()
    per_row_pos = 100.0 * (pos - s_pos).abs().flatten(1).max(1).values.cpu().numpy()
    out = {
        "rows": int(len(rows)), "neck_n": int(nb.n), "step_n": step_n,
        "tokens": diff(tok, s_tok), "pos_cm": diff(pos, s_pos, 100.0),
        "row_max_abs_tokens_q50_q99_max": [float(np.quantile(per_row, q)) for q in (0.5, 0.99, 1.0)],
        "row_max_abs_pos_cm_q50_q99_max": [float(np.quantile(per_row_pos, q)) for q in (0.5, 0.99, 1.0)],
        "rows_tokens_over_0.05": int((per_row > 0.05).sum()),
    }
    side = root / "item_buffer"
    for k in ("episode_id", "t_in_ep"):
        if k in arrs and (side / f"{k}.npy").exists():
            ref = np.load(side / f"{k}.npy", mmap_mode="r")[rows]
            out[f"{k}_mismatch"] = int((np.asarray(arrs[k]) != ref).sum())
        else:
            out[f"{k}_mismatch"] = None
    return out


def io_cost(args, rng) -> dict:
    """Reading a training step's distinct frames from the neck memmaps and uploading them, per frame."""
    nb, n_valid = open_neck(Path(args.data))
    n = min(STEP_FRAMES, n_valid)
    rows = np.sort(rng.choice(n_valid, n, replace=False))
    t0 = time.perf_counter()
    arrs = nb.read(rows)
    t_read = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fz.rows_to_neck(arrs, DEVICE)
    torch.cuda.synchronize()
    t_up = time.perf_counter() - t0
    mb = sum(np.asarray(a).nbytes for a in arrs.values()) / 2**20
    return {"frames": n, "read_ms_per_frame": 1000 * t_read / n, "upload_ms_per_frame": 1000 * t_up / n,
            "mb_per_frame": mb / n, "note": "cold or warm page cache: run twice to see both"}


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def train_config(sets):
    """LangConfig defaults on one embodiment (ur5-wsg, train.py's default) plus key=value overrides (JSON values)."""
    import dataclasses

    from main.lang.config import LangConfig

    kw = {"embodiment_choices": ("ur5-wsg",)}
    for item in sets:
        k, v = item.split("=", 1)
        kw[k] = json.loads(v)
    if "scene_fusion" in kw:
        raise SystemExit("--train-set scene_fusion: the bench times both settings")
    return dataclasses.replace(LangConfig(), **kw)


def train_steps(args, stage, device, text_encoder=None) -> dict:
    """End-to-end training steps (FUSION_PLAN 3.3): LangTrainer on the --data shard with scene_fusion off, then on (the
    stage passed in, at the config's fusion_dtype / chunk), the same config and seed otherwise. Per variant, over
    --train-steps steps after --train-warmup: the medians of the trainer's own spans (perf/step_ms: text lookup,
    fusion, updates, Polyak; perf/fusion_ms), of the synchronous batch draw (sample_ms) and of draw + step wall time
    (synchronized), the mean frames fused per step, and the peak memory."""
    import dataclasses
    import gc
    import tempfile

    from main.lang.train import LangTrainer, trained_subsets

    base = train_config(args.train_set)
    tasks, embs = trained_subsets(base)
    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"steps": int(args.train_steps), "warmup": int(args.train_warmup), "batch_size": int(base.batch_size),
           "config_hash": base.hash(), "overrides": list(args.train_set), "variants": {}}
    for on in (False, True):
        cfg = dataclasses.replace(base, scene_fusion=on)
        if on:
            stage.dtype, stage.chunk = cfg.fusion_dtype, int(cfg.fusion_chunk)
        name = "V5 (scene_fusion on)" if on else "V0' (scene_fusion off)"
        with tempfile.TemporaryDirectory(prefix="_bench_train_", dir=out_dir) as tmp:
            tr = LangTrainer(cfg, Path(tmp) / "run", [Path(args.data)], (), [Path(c) for c in args.text_cache],
                             tasks=tasks, embodiments=embs, use_wandb=False, prefetch=False, device=device,
                             text_encoder=text_encoder, seed=args.seed, fusion=stage if on else None)
            tr.policy_enc.train(), tr.critics.train(), tr.actor.train()

            def one():
                t0 = time.perf_counter()
                b = tr.sampler.sample(cfg.batch_size)
                t1 = time.perf_counter()
                tr.grad_step(b)
                _sync(device)
                t2 = time.perf_counter()
                spans = tr.timer.read()   # this step's spans (read per step: the timer keeps a bounded window)
                return {"sample_ms": 1000 * (t1 - t0), "wall_ms": 1000 * (t2 - t0), "step_ms": spans["step"][0],
                        "fusion_ms": spans["fusion"][0] if "fusion" in spans else None,
                        "fusion_frames": tr.last_fused_frames if on else 0}

            for _ in range(args.train_warmup):
                one()
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            rows = [one() for _ in range(args.train_steps)]
            row = {k: (float(np.median([r[k] for r in rows])) if rows[0][k] is not None else None)
                   for k in ("sample_ms", "wall_ms", "step_ms", "fusion_ms")}
            row["step_ms_mean"] = float(np.mean([r["step_ms"] for r in rows]))
            row["fusion_frames"] = float(np.mean([r["fusion_frames"] for r in rows]))
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if str(device).startswith("cuda") else None
            row["text_miss_frac"] = tr.text.miss_frac().get("all", 0.0)
            res["variants"][name] = row
            print(json.dumps({name: row}), flush=True)
            del tr
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    off, on_ = res["variants"]["V0' (scene_fusion off)"], res["variants"]["V5 (scene_fusion on)"]
    res["step_ratio"] = on_["step_ms"] / max(off["step_ms"], 1e-9)
    res["wall_ratio"] = on_["wall_ms"] / max(off["wall_ms"], 1e-9)
    res["more_than_doubles"] = bool(res["step_ratio"] > 2.0)
    return res


# ----------------------------------------------------------------- report --


def markdown(res: dict) -> str:
    def f(x, nd=2):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    env = res["env"]
    lines = [f"# Fusion bench: training step ({env['gpu']}, {env['source']})" if res.get("train") else
             f"# Fusion bench ({env['gpu']}, {env['source']}, {env['frames']} frames)", ""]
    if res.get("train"):
        t = res["train"]
        lines += ["## Training step, scene_fusion off vs on (LangTrainer.grad_step)", "",
                  f"{t['steps']} steps after {t['warmup']}, batch {t['batch_size']}, config {t['config_hash']} "
                  f"(overrides: {', '.join(t['overrides']) or 'none'}). Medians per step. step ms = the trainer's "
                  "perf/step_ms (text lookup, fusion, updates, Polyak); sample ms = the synchronous batch draw "
                  "(hidden by the PrefetchSampler in a real run); wall = draw + step, synchronized.", "",
                  "| variant | step ms | fusion ms | frames fused / step | sample ms | wall ms | peak GB | text miss |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in t["variants"].items():
            lines.append(f"| {name} | {f(r['step_ms'], 0)} | {f(r['fusion_ms'], 0)} | {f(r['fusion_frames'], 0)} | "
                         f"{f(r['sample_ms'], 0)} | {f(r['wall_ms'], 0)} | {f(r['peak_gb'], 1)} | "
                         f"{f(r['text_miss_frac'], 3)} |")
        lines += ["", f"- step time on / off: {t['step_ratio']:.2f}x (wall {t['wall_ratio']:.2f}x): "
                      f"{'more than doubles' if t['more_than_doubles'] else 'does not double'} (FUSION_PLAN 3.3)", ""]
    if res.get("speed"):
        lines += ["## Speed and memory (fuse)", "",
                  "| lowmem | dtype | chunk | lengths | ms / frame | fusion ms / step (x 320) * | peak MB | "
                  "peak MB / frame |", "|---|---|---|---|---|---|---|---|"]
        for r in res["speed"]:
            if "error" in r:
                lines.append(f"| {r['lowmem']} | {r['dtype']} | {r['chunk']} | {r['lengths']} | {r['error']} | | | |")
            else:
                lines.append(f"| {r['lowmem']} | {r['dtype']} | {r['chunk']} | {r['lengths']} | "
                             f"{f(r['ms_per_frame'])} | {f(r['step_ms_320'], 0)} | {f(r['peak_mb'], 0)} | "
                             f"{f(r.get('peak_mb_per_frame'), 1)} |")
        lines += ["", f"\\* step_ms_320: {STEP_MS_320_NOTE}"]
    if res.get("unfused"):
        lines += ["", "## unfused (V0 path)", "", "| dtype | ms / frame | peak MB |", "|---|---|---|"]
        for r in res["unfused"]:
            lines.append(f"| {r['dtype']} | {r.get('error') or f(r['ms_per_frame'])} | {f(r.get('peak_mb'), 0)} |")
    if res.get("precision"):
        lines += ["", "## Precision against fp32 (lowmem)", "",
                  "| run | field | max abs | mean abs | max / max ref | rel L2 |", "|---|---|---|---|---|---|"]
        for name, r in res["precision"].items():
            if "error" in r:
                lines.append(f"| {name} | {r['error']} | | | | |")
                continue
            for field, d in r.items():
                lines.append(f"| {name} | {field} | {d['max_abs']:.3g} | {d['mean_abs']:.3g} | {d['rel_max']:.3g} | "
                             f"{d['rel_l2']:.3g} |")
    if res.get("parity"):
        p = res["parity"]
        lines += ["", "## Stored-row parity (unfused on fp16 neck rows vs stored tokens)", "",
                  f"- rows {p['rows']} (neck n {p['neck_n']}, StepBuffer n {p['step_n']})",
                  f"- tokens: max {p['tokens']['max_abs']:.3g}, mean {p['tokens']['mean_abs']:.3g}; per-row max "
                  f"q50 / q99 / max {', '.join(f'{x:.3g}' for x in p['row_max_abs_tokens_q50_q99_max'])}; "
                  f"rows over 0.05: {p['rows_tokens_over_0.05']}",
                  f"- positions (cm): max {p['pos_cm']['max_abs']:.3g}, mean {p['pos_cm']['mean_abs']:.3g}",
                  f"- lockstep: episode_id mismatches {p['episode_id_mismatch']}, t_in_ep mismatches "
                  f"{p['t_in_ep_mismatch']}"]
    if res.get("io"):
        io = res["io"]
        lines += ["", "## Neck read and upload", "",
                  f"- {io['frames']} frames, {io['mb_per_frame']:.2f} MB / frame: read {io['read_ms_per_frame']:.2f} "
                  f"ms / frame, upload {io['upload_ms_per_frame']:.2f} ms / frame ({io['note']})"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="random neck dicts with valid camera geometry")
    src.add_argument("--data", help="a version-2 shard root (buffer/, item_buffer/, neck_buffer/)")
    ap.add_argument("--text-cache", nargs="*", default=[], help="text_cache.pt files (else random words)")
    ap.add_argument("--out", required=True, help="<name>.json; the tables go next to it as <name>.md")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--frames", type=int, default=64, help="frames per timed fuse call")
    ap.add_argument("--chunks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--dtypes", nargs="+", default=list(fz.FUSION_DTYPES), choices=fz.FUSION_DTYPES)
    ap.add_argument("--no-hf", action="store_true", help="skip transformers' stacking attention (lowmem off)")
    ap.add_argument("--precision-frames", type=int, default=16, help=">= 2 (the sentence scale pairs frames)")
    ap.add_argument("--parity-rows", type=int, default=100)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=0,
                    help="N > 0: only time N LangTrainer grad_steps with scene_fusion off and on (needs --data, a "
                         "version-2 shard); the input to FUSION_PLAN 3.3's 'more than doubles' decision")
    ap.add_argument("--train-warmup", type=int, default=3, help="--train-steps: untimed steps first, per variant")
    ap.add_argument("--train-set", nargs="*", default=[],
                    help="--train-steps: LangConfig overrides key=value, JSON values (as train.py --set), e.g. "
                         "fusion_dtype='\"bf16\"' batch_size=64; scene_fusion is the bench's own variable")
    args = ap.parse_args(argv)
    if args.precision_frames < 2:
        ap.error("--precision-frames must be >= 2")
    if args.train_steps < 0 or args.train_warmup < 0:
        ap.error("--train-steps / --train-warmup must be >= 0")
    if args.train_steps and not args.data:
        ap.error("--train-steps needs --data (a version-2 shard: the trainer reads its neck rows)")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("bench_fusion needs a GPU")

    rng = np.random.default_rng(args.seed)
    if args.train_steps:   # the training-step comparison alone (it builds its own trainers; no sweep)
        train_config(args.train_set)   # a bad override fails before the stage loads
        t0 = time.time()
        stage = fz.FusionStage(args.ckpt, device=DEVICE)
        res = {"env": {"gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE,
                       "torch": torch.__version__, "source": str(args.data),
                       "text": args.text_cache or "encoded on a cache miss", "build_s": time.time() - t0,
                       "args": vars(args)}}
        res["train"] = train_steps(args, stage, DEVICE, TEXT_ENCODER)
        return write(res, Path(args.out))
    t0 = time.time()
    stages = {True: fz.FusionStage(args.ckpt, device=DEVICE)}
    if not args.no_hf:
        stages[False] = fz.FusionStage(args.ckpt, device=DEVICE, lowmem=False,
                                       spatial_enhancer=stages[True].spatial_enhancer)
    build_s = time.time() - t0
    res = {"env": {"gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE, "torch": torch.__version__,
                   "source": "synthetic" if args.synthetic else str(args.data), "frames": args.frames,
                   "text": args.text_cache or "random words", "build_s": build_s,
                   "n_enhancer_keys": stages[True].n_enhancer_keys, "n_lowmem_layers": stages[True].n_lowmem,
                   "args": vars(args)},
           "notes": {"step_ms_320": STEP_MS_320_NOTE}}

    neck, _ = neck_frames(args, args.frames, rng, DEVICE)
    words, mask, _ = load_words(args.text_cache, args.frames, rng, DEVICE)
    res["speed"] = speed(stages, neck, words, mask, args)
    res["unfused"] = speed_unfused(stages[True], neck, args)
    del neck

    P = args.precision_frames
    pneck, _ = neck_frames(args, P, rng, DEVICE)
    w1, m1, s1 = load_words(args.text_cache, P, rng, DEVICE)
    # the other sentence of frame f is frame f-1's (a second draw can repeat frame f's own sentence)
    w2, m2 = w1.roll(1, 0), m1.roll(1, 0)
    res["precision"] = precision(stages, pneck, w1, m1, w2, m2)
    res["precision_sentences"] = s1[:8]
    del pneck

    if args.data:
        res["parity"] = parity(stages[True], args, rng)
        res["io"] = io_cost(args, rng)

    return write(res, Path(args.out))


def write(res: dict, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    out.with_suffix(".md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
