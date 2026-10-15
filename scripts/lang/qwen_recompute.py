"""Recompute a V6 shard's HoloBrain-Qwen outputs from its frames/ sidecar (pre-run review 2026-10-15).

The shards of 2026-10-14 were collected with front-end version 1, which rounded the rotary inv_freq buffers to bf16
(image features 6-9 % off HoloBrain-Qwen's own). The simulation itself never read those outputs (nominal episodes),
and every replayed shard keeps the native renders (frames/), so the encoder outputs can be rewritten exactly as
obs.qwen_obs + HoloBrainQwenEncoder.encode_frame compute them at collection: buffer/scene_tokens (fp16),
buffer/scene_pos and kv_buffer/{k, v} (bf16 bits), for every filled row. Nothing else changes (robot state, h,
actions, item sidecar). The only input difference is depth stored as fp16 (about 0.5 mm at 1 m).

    scripts/lang/in_container.sh python scripts/lang/qwen_recompute.py <shard> [--attn flash_attention_2]

One frame per encode_frame call by default, as collection, the rounds ring and deployment encode: the bf16 front end
is not batch-invariant (a cuBLAS bias GEMM in the vision MLP picks another algorithm for one frame than for several,
and the 32 vision blocks grow the 1e-3 difference to 4-9 %; second review 2026-10-15). Resumable (<shard>/
recompute.json records the rows done); a different encoder version, kernel or frames per call starts over. When every row is done it stamps kv_buffer/header.json (encoder_version,
attn, recomputed) and shard.json (kv.encoder_version, kv.attn, recomputed). Run it on the GPU type the rounds run
collects on (A40): the bf16 front end drifts a few percent across GPU types.
Exit 0 = done, 75 = the shard is not finished collecting yet, or the stage deadline came (resume later).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def frame_inputs(fr: dict, f: int):
    """FramesBuffer rows -> one frame's encode_frame inputs, as obs.qwen_obs builds them (cv2 bilinear 320x240 ->
    308x252 for RGB and depth, K rows scaled, ego projections, T_ego2world = inv(ext_head))."""
    from main.lang.obs import ego_projections, resize_frame

    rgb, depth, Ks = [], [], []
    for c in range(fr["rgb"].shape[1]):
        r, d, k = resize_frame(np.ascontiguousarray(fr["rgb"][f, c]), np.asarray(fr["depth"][f, c], np.float32),
                               fr["K"][f, c])
        rgb.append(r), depth.append(d), Ks.append(k)
    proj, T = ego_projections(Ks, [np.asarray(e, np.float64) for e in fr["ext"][f]])
    return np.stack(rgb), np.stack(depth).astype(np.float32), proj, T


def check_alignment(root: Path, n: int) -> None:
    """buffer/, item_buffer/, kv_buffer/ and frames/ hold the same (episode, tick) in every filled row."""
    ep_b = np.asarray(np.load(root / "buffer" / "episode_id.npy", mmap_mode="r")[:n], np.int64)
    ep_i = np.asarray(np.load(root / "item_buffer" / "episode_id.npy", mmap_mode="r")[:n], np.int64)
    t_i = np.asarray(np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")[:n], np.int64)
    if not np.array_equal(ep_b, ep_i):
        raise RuntimeError(f"{root}: buffer/ and item_buffer/ episode ids differ")
    for side in ("kv_buffer", "frames"):
        e = np.asarray(np.load(root / side / "episode_id.npy", mmap_mode="r")[:n], np.int64)
        t = np.asarray(np.load(root / side / "t_in_ep.npy", mmap_mode="r")[:n], np.int64)
        if not (np.array_equal(e, ep_i) and np.array_equal(t, t_i)):
            raise RuntimeError(f"{root}: {side}/ is out of lockstep with item_buffer/")


def recompute_shard(root, enc, *, batch: int = 1, deadline: float | None = None, flush_every: int = 200,
                    log=print) -> int:
    """Rewrite the shard's encoder outputs with ``enc`` (encode_frame, ENCODER_VERSION, attn, ckpt_id). -> 0 done,
    75 not finished collecting / stopped at the deadline."""
    import torch

    from main.lang.frames_buffer import FramesBuffer
    from main.lang.kv_buffer import KvBuffer

    root = Path(root).resolve()
    summ = root / "summary.json"
    if not (summ.exists() and json.loads(summ.read_text()).get("complete")):
        log(f"[recompute] {root}: still collecting (no complete summary.json): later")
        return 75
    meta_p = root / "shard.json"
    meta = json.loads(meta_p.read_text())
    if meta.get("encoder") != "holobrain_qwen" or not meta.get("frames"):
        raise SystemExit(f"{root}: not a V6 shard with frames/ (encoder {meta.get('encoder')}, frames {meta.get('frames')})")
    version, attn, batch = int(enc.ENCODER_VERSION), str(enc.attn), int(batch)
    if meta["kv"].get("ckpt_id") != enc.ckpt_id:
        raise SystemExit(f"{root}: kv rows of checkpoint {meta['kv'].get('ckpt_id')}, the encoder is {enc.ckpt_id}")
    prog_p = root / "recompute.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {}
    same = (prog.get("encoder_version"), prog.get("attn"), prog.get("frames_per_call")) == (version, attn, batch)
    if same and prog.get("complete"):
        log(f"[recompute] {root}: done already (version {version}, {attn}, {batch} frame(s) per call)")
        return 0
    n = int(json.loads((root / "buffer" / "header.json").read_text())["n"])
    kb, fb = KvBuffer.reopen(root / "kv_buffer"), FramesBuffer(root / "frames")
    if not (kb.n == fb.n == n and kb.head == fb.head == n % kb.capacity):
        raise RuntimeError(f"{root}: rings at n = buffer {n}, kv {kb.n}, frames {fb.n}")
    check_alignment(root, n)
    tok = np.load(root / "buffer" / "scene_tokens.npy", mmap_mode="r+")
    pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r+")
    if tok.shape[1:] != (enc.N_TOK, 256) or pos.shape[1:] != (enc.N_TOK, 3):
        raise RuntimeError(f"{root}: scene rows {tok.shape} / {pos.shape}")
    start = int(prog.get("rows_done", 0)) if same else 0
    prog = {"encoder_version": version, "attn": attn, "frames_per_call": batch, "ckpt_id": enc.ckpt_id, "n": n,
            "rows_done": start,
            "complete": False, "started": prog.get("started") if same else time.strftime("%F %T"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "commit": _commit()}
    _atomic_json(prog_p, prog)
    t0, k = time.time(), 0
    for r0 in range(start, n, batch):
        if deadline is not None and time.time() > deadline - 120:
            _flush(tok, pos, kb)
            prog["rows_done"] = r0
            _atomic_json(prog_p, prog)
            log(f"[recompute] {root}: deadline at row {r0}/{n}: later")
            return 75
        rows = np.arange(r0, min(n, r0 + batch))
        fr = fb.read(rows)
        ins = [frame_inputs(fr, f) for f in range(len(rows))]
        with torch.no_grad():
            tokens, p, (K, V) = enc.encode_frame(np.stack([x[0] for x in ins]), np.stack([x[1] for x in ins]),
                                                 np.stack([x[2] for x in ins]), np.stack([x[3] for x in ins]))
        t = tokens.flatten(1, 2).float().cpu().numpy()
        if not np.isfinite(t).all() or float(np.abs(t).max()) > 6.0e4:
            raise RuntimeError(f"{root}: rows {r0}..: scene tokens outside the fp16 range")
        tok[rows] = t.astype(np.float16)
        pos[rows] = p.flatten(1, 2).float().cpu().numpy()
        kb.arrays["k"][rows] = K.contiguous().view(torch.int16).cpu().numpy()
        kb.arrays["v"][rows] = V.contiguous().view(torch.int16).cpu().numpy()
        k += 1
        if k % int(flush_every) == 0:
            _flush(tok, pos, kb)
            prog["rows_done"] = int(rows[-1]) + 1
            _atomic_json(prog_p, prog)
            log(f"[recompute] {root.name}: {prog['rows_done']}/{n} rows, {(time.time() - t0) / k:.2f} s/batch")
    _flush(tok, pos, kb)
    stamp = {"encoder_version": version, "attn": attn, "frames_per_call": batch, "at": time.strftime("%F %T"),
             "gpu": prog["gpu"], "commit": prog["commit"], "rows": n}
    kb.header.update({"encoder_version": version, "attn": attn, "frames_per_call": batch, "recomputed": stamp})
    kb._write_header()
    meta["kv"].update({"encoder_version": version, "attn": attn, "frames_per_call": batch})
    meta["recomputed"] = stamp
    _atomic_json(meta_p, meta)
    prog.update(rows_done=n, complete=True, finished=stamp["at"])
    _atomic_json(prog_p, prog)
    log(f"[recompute] {root}: {n} rows rewritten (version {version}, {attn}, {batch} frame(s) per call) in "
        f"{time.time() - t0:.0f} s")
    return 0


def _flush(tok, pos, kb) -> None:
    tok.flush(), pos.flush()
    for a in kb.arrays.values():
        a.flush()


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--attn", default="flash_attention_2", choices=("sdpa", "flash_attention_2", "eager"))
    ap.add_argument("--batch", type=int, default=1,
                    help="frames per encode_frame call (3 images each); 1 = as collection and deployment encode")
    args = ap.parse_args(argv)
    import main.lang  # noqa: F401
    from main.lang.collect import stage_deadline
    from main.lang.qwen_encoder import HoloBrainQwenEncoder

    ready = [s for s in args.shards
             if (Path(s) / "summary.json").exists() and json.loads((Path(s) / "summary.json").read_text()).get("complete")]
    if not ready:   # nothing finished collecting yet: do not load the front end (the workqueue retries later)
        print(f"[recompute] none of {args.shards} has finished collecting: later")
        return 75
    enc = HoloBrainQwenEncoder(device="cuda", attn=args.attn)
    deadline = stage_deadline(None)
    rc = 0 if len(ready) == len(args.shards) else 75
    for s in ready:
        r = recompute_shard(s, enc, batch=args.batch, deadline=deadline)
        rc = max(rc, r)
        if r == 75 and deadline is not None and time.time() > deadline - 120:
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
