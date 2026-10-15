"""Integrity check of collected shards / rings (review O1): every episode line's rows [r0, r1) carry that
episode's id in BOTH buffers, t_in_ep counts 0..n-1 within them, lines do not overlap, and the header n is
the end of the last line. Two writers on one shard break at least one of these.

    python scripts/lang/check_shards.py data/lang/warmup data/lang/val [runs/lang/<run>/ring ...]
    python scripts/lang/check_shards.py --fusion-parity 100 data/lang/val_grounding_v2    # GPU, in the container

Version-2 shards (FUSION_PLAN 3.2: a neck_buffer/ next to buffer/ and item_buffer/), still torch-free: the neck
header's capacity, n and head equal the StepBuffer's and the item sidecar's; every filled neck row holds the item
sidecar's episode_id and t_in_ep of that row; shard.json and every episode line say lang_version 2. A shard
without neck_buffer/ must not claim version 2. A --replay shard: line i holds slot i's recorded episode, with rows,
and its summary.json records no replay_mismatch.

V6 shards (qwen_impl.md 6; shard.json "encoder": "holobrain_qwen"): kv_buffer/ and frames/ exist exactly as
shard.json says (kv always, frames when "frames" is true), each with the StepBuffer's capacity, n and head and the
item sidecar's episode_id / t_in_ep in every filled row, the kv rows of the recorded checkpoint; the scene width is
297 tokens per frame under Qwen (and never 297 under GD), and the item sidecar's n_cam x n_tok equals it; every episode line
names the encoder.

--fusion-parity N (FUSION_PLAN R6; torch and a GPU): N random filled rows of every version-2 shard,
FusionStage.unfused(neck rows) (the spatial enhancer on the stored fp16 neck maps) against the StepBuffer's
stored scene_tokens / scene_pos of the same rows. A row whose relative mean error exceeds --parity-tol fails the
shard: a rollback or resume mirrored on one side only gives O(1) errors there.

Reads only (memmaps opened read-only). Exit 1 if any shard fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_CKPT = os.environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD")
PARITY_BATCH = 16      # neck rows per unfused call (2.9 MB each on the host)
_STAGE_FACTORY = None  # tests inject a stand-in for FusionStage here: f(args) -> object with .unfused(neck)


def roots(args):
    for a in args:
        p = Path(a)
        if (p / "lang_episodes.jsonl").exists():
            yield p
        else:
            yield from sorted(q for q in p.glob("shard_g*") if (q / "lang_episodes.jsonl").exists())


def check(root: Path) -> list[str]:
    bad = []
    hdr = json.loads((root / "buffer" / "header.json").read_text())
    side_hdr = json.loads((root / "item_buffer" / "header.json").read_text())
    n_buf, n_side = int(hdr.get("n", 0)), int(side_hdr.get("n", 0))
    ep_buf = np.load(root / "buffer" / "episode_id.npy", mmap_mode="r")
    ep_side = np.load(root / "item_buffer" / "episode_id.npy", mmap_mode="r")
    t_side = np.load(root / "item_buffer" / "t_in_ep.npy", mmap_mode="r")
    lines = []
    for ln in (root / "lang_episodes.jsonl").read_text().splitlines():
        try:
            lines.append(json.loads(ln))
        except json.JSONDecodeError:
            bad.append("a cut line in lang_episodes.jsonl")
    end = 0
    ids = set()
    for d in lines:
        if d.get("skipped"):
            continue
        eid = int(d["episode_id"])
        if eid in ids:
            bad.append(f"episode {eid} listed twice")
        ids.add(eid)
        r0, r1 = (int(x) for x in d["rows"])
        if r0 != end:
            bad.append(f"episode {eid}: rows start at {r0}, previous episode ended at {end}")
        if r1 <= r0:   # an episode that wrote no rows (failed before its first tick): nothing to corrupt
            print(f"note {root}: episode {eid} has no rows (status {d.get('status')}, n_ticks {d.get('n_ticks')})")
            continue
        if (np.asarray(ep_buf[r0:r1]) != eid).any() or (np.asarray(ep_side[r0:r1]) != eid).any():
            bad.append(f"episode {eid}: rows [{r0}, {r1}) hold other episode ids")
        if not np.array_equal(np.asarray(t_side[r0:r1]), np.arange(r1 - r0)):
            bad.append(f"episode {eid}: t_in_ep is not 0..{r1 - r0 - 1}")
        end = r1
    if (n_buf, n_side) != (end, end) and lines:
        bad.append(f"headers n = ({n_buf}, {n_side}) but the last episode ends at {end}")
    bad += check_neck(root, hdr, side_hdr, lines, ep_side, t_side)
    bad += check_v6(root, hdr, side_hdr, lines, ep_side, t_side)
    bad += check_replay(root, lines)
    return bad


def _lockstep(name: str, sc, hdr: dict, side_hdr: dict, ep_side, t_side) -> list[str]:
    """A ring sidecar (sidecar.RingSidecar) against the StepBuffer and item sidecar headers and rows."""
    bad = []
    for other, h in (("StepBuffer", hdr), ("item sidecar", side_hdr)):
        if (sc.capacity, sc.n, sc.head) != _ring(h):
            bad.append(f"{name} capacity / n / head = ({sc.capacity}, {sc.n}, {sc.head}) but the {other}'s are {_ring(h)}")
    filled = min(sc.n, sc.capacity, *_ring(side_hdr)[:2])
    ep_n, t_n = (np.asarray(sc.arrays[k][:filled], np.int64) for k in ("episode_id", "t_in_ep"))
    ep_i, t_i = np.asarray(ep_side[:filled], np.int64), np.asarray(t_side[:filled], np.int64)
    off = np.flatnonzero((ep_n != ep_i) | (t_n != t_i))
    if off.size:
        r = int(off[0])
        bad.append(f"{name} out of lockstep with item_buffer: {off.size} of {filled} filled rows differ, first row {r} "
                   f"({name} episode {ep_n[r]} tick {t_n[r]}, item episode {ep_i[r]} tick {t_i[r]})")
    return bad


def check_v6(root: Path, hdr: dict, side_hdr: dict, lines: list[dict], ep_side, t_side) -> list[str]:
    """The V6 checks (numpy only): the kv / frames sidecars as shard.json says, in lockstep; the scene width of the
    encoder and the item masks' width."""
    from main.lang.frames_buffer import FramesBuffer
    from main.lang.kv_buffer import KvBuffer

    bad = []
    meta_p = root / "shard.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
    if meta is not None:
        qwen = meta.get("encoder", "holobrain_gd") == "holobrain_qwen"
        want = {"kv_buffer": qwen, "frames": qwen and bool(meta.get("frames"))}
    else:   # a rounds ring: the StepBuffer's header carries the config
        qwen = hdr.get("encoder", "holobrain_gd") == "holobrain_qwen"
        want = {"kv_buffer": qwen, "frames": (root / "frames" / "header.json").exists()}
    for d, cls in (("kv_buffer", KvBuffer), ("frames", FramesBuffer)):
        present = (root / d / "header.json").exists()
        if present != want[d]:
            bad.append(f"{d}/ {'present' if present else 'missing'}, shard.json says "
                       f"{'yes' if want[d] else 'no'} (encoder {meta.get('encoder', 'holobrain_gd') if meta else '?'})")
            continue
        if not present:
            continue
        try:
            sc = cls(root / d)
        except (OSError, ValueError, KeyError) as e:
            bad.append(f"{d}: {e}")
            continue
        bad += _lockstep(d, sc, hdr, side_hdr, ep_side, t_side)
        if d == "kv_buffer" and meta is not None and sc.header.get("ckpt_id") != (meta.get("kv") or {}).get("ckpt_id"):
            bad.append(f"kv_buffer rows of checkpoint {sc.header.get('ckpt_id')}, shard.json says "
                       f"{(meta.get('kv') or {}).get('ckpt_id')}")
    tok_p = root / "buffer" / "scene_tokens.npy"
    if tok_p.exists():
        width = int(np.load(tok_p, mmap_mode="r").shape[1])
        if qwen and width != 297:
            bad.append(f"scene width {width}, the Qwen encoder makes 297")
        elif not qwen and width == 297:
            bad.append("scene width 297 (the Qwen grid) in a shard that does not say encoder holobrain_qwen")
        cells = int(side_hdr.get("n_cam", 3)) * int(side_hdr.get("n_tok", -1))
        if qwen and cells != width:   # the probe's ref_mask needs masks on the scene grid (GD shards: 3 x 400 = 1200)
            bad.append(f"item sidecar n_cam x n_tok = {cells}, the scene width is {width}")
    if qwen:
        off = [d for d in lines if not d.get("skipped") and d.get("encoder") != "holobrain_qwen"]
        if off:
            bad.append(f"{len(off)} episode lines do not name the Qwen encoder (first: episode {off[0].get('episode_id')})")
    return bad


def check_replay(root: Path, lines: list[dict]) -> list[str]:
    """A --replay shard (shard.json has replay_of): episode line i holds the recorded episode of slot i (task and
    the replayed seed, the slot's 4th field) with rows (an episode rolled back to zero rows lost its layout, as in
    collect.replay_mismatch; its status is not compared), and a finished shard's summary.json records no
    replay_mismatch (a slot whose one build failed, or other ids than the source's)."""
    meta_p = root / "shard.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if "replay_of" not in meta:
        return []
    bad = []
    slots = meta.get("slots") or []
    if len(lines) > len(slots):
        bad.append(f"replay: {len(lines)} episode lines for {len(slots)} replayed slots")
    for i, (d, s) in enumerate(zip(lines, slots)):
        if d.get("skipped"):
            continue   # no episode in the rings; summary.json's replay_mismatch lists it once the shard is done
        if len(s) < 4 or (d.get("task"), int(d["episode_id"])) != (s[0], int(s[3])):
            bad.append(f"replay slot {i}: episode {d.get('task')} {d.get('episode_id')}, the replayed slot is {s}")
        rows = d.get("rows")
        if not rows or int(rows[1]) <= int(rows[0]):
            bad.append(f"replay slot {i}: episode {d.get('task')} {d.get('episode_id')} has no rows (rows {rows}, "
                       f"status {d.get('status')}: {(d.get('exception') or '')[:120]})")
    summ = root / "summary.json"
    mism = json.loads(summ.read_text()).get("replay_mismatch") if summ.exists() else None
    if mism:
        bad.append(f"summary.json replay_mismatch ({len(mism)}): {'; '.join(mism[:3])}")
    return bad


def _ring(h: dict) -> tuple[int, int, int]:
    """(capacity, n, head) of a ring header (head defaults to n, as the buffers read it)."""
    n = int(h.get("n", 0))
    return int(h.get("capacity", -1)), n, int(h.get("head", n))


def check_neck(root: Path, hdr: dict, side_hdr: dict, lines: list[dict], ep_side, t_side) -> list[str]:
    """The version-2 checks (numpy only): the data version in shard.json and the lines, and the neck sidecar in
    lockstep with the item sidecar (FUSION_PLAN 3.2, R6's CPU half)."""
    bad = []
    meta_p = root / "shard.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
    v_lines = [d.get("lang_version", 1) for d in lines]
    if not (root / "neck_buffer" / "header.json").exists():   # version 1: nothing may say otherwise
        if (root / "neck_buffer").exists():
            bad.append("neck_buffer/ without header.json (a partial copy?)")
        if meta is not None and meta.get("lang_version", 1) == 2:
            bad.append("shard.json says lang_version 2 but there is no neck_buffer/")
        if 2 in v_lines:
            bad.append(f"{v_lines.count(2)} episode lines say lang_version 2 but there is no neck_buffer/")
        return bad
    from main.lang.neck_buffer import NeckBuffer   # numpy only

    if meta is None:
        print(f"note {root}: no shard.json (a ring?): its data version is not recorded")
    elif meta.get("lang_version", 1) != 2:
        bad.append(f"shard.json lang_version {meta.get('lang_version', 1)} next to a neck_buffer/ (expected 2)")
    off = [d for d in lines if d.get("lang_version", 1) != 2]
    if off:
        bad.append(f"{len(off)} of {len(lines)} episode lines lack lang_version 2 (first: episode "
                   f"{off[0].get('episode_id')}, {off[0].get('lang_version', 'none')})")
    try:
        nb = NeckBuffer(root / "neck_buffer")   # read-only; raises on a bad header or a missing / misshapen field
    except (OSError, ValueError, KeyError) as e:
        return bad + [f"neck_buffer: {e}"]
    for name, h in (("StepBuffer", hdr), ("item sidecar", side_hdr)):
        if (nb.capacity, nb.n, nb.head) != _ring(h):
            bad.append(f"neck_buffer capacity / n / head = ({nb.capacity}, {nb.n}, {nb.head}) but the {name}'s are "
                       f"{_ring(h)}")
    filled = min(nb.n, nb.capacity, *_ring(side_hdr)[:2])    # rows written in both rings (all of them once wrapped)
    ep_n, t_n = (np.asarray(nb.arrays[k][:filled], np.int64) for k in ("episode_id", "t_in_ep"))
    ep_i, t_i = np.asarray(ep_side[:filled], np.int64), np.asarray(t_side[:filled], np.int64)
    off = np.flatnonzero((ep_n != ep_i) | (t_n != t_i))
    if off.size:
        r = int(off[0])
        bad.append(f"neck_buffer out of lockstep with item_buffer: {off.size} of {filled} filled rows differ, first "
                   f"row {r} (neck episode {ep_n[r]} tick {t_n[r]}, item episode {ep_i[r]} tick {t_i[r]})")
    return bad


# ------------------------------------------------------------- fusion parity --


def make_stage(args):
    if _STAGE_FACTORY is not None:
        return _STAGE_FACTORY(args)
    import main.lang  # noqa: F401  (main.__path__ bootstrap)
    from main.lang.fusion import FusionStage

    return FusionStage(args.ckpt, device=args.device, dtype=args.dtype, chunk=args.chunk)


def fusion_parity(root: Path, stage, n_rows: int, rng, device: str) -> dict:
    """R6 on stored rows: unfused(neck row) vs the StepBuffer row it was written with, n_rows random filled rows.
    Per row: the relative mean error mean|d| / mean|stored| of the tokens and of the positions."""
    import torch

    from main.lang.fusion import rows_to_neck
    from main.lang.neck_buffer import NeckBuffer

    nb = NeckBuffer(root / "neck_buffer")
    step_n = int(json.loads((root / "buffer" / "header.json").read_text()).get("n", 0))
    filled = min(nb.n, nb.capacity, step_n)
    rows = np.sort(rng.choice(filled, min(int(n_rows), filled), replace=False)) if filled else np.zeros(0, np.int64)
    s_tok = np.load(root / "buffer" / "scene_tokens.npy", mmap_mode="r")
    s_pos = np.load(root / "buffer" / "scene_pos.npy", mmap_mode="r")
    acc = {k: [] for k in ("tok_rel", "pos_rel", "tok_max", "pos_max", "tok_abs", "pos_abs", "tok_ref", "pos_ref")}
    for a in range(0, len(rows), PARITY_BATCH):
        sel = rows[a:a + PARITY_BATCH]
        tok, pos = stage.unfused(rows_to_neck(nb.read(sel), device))
        ref_t = torch.from_numpy(np.asarray(s_tok[sel], np.float32)).to(tok.device)
        ref_p = torch.from_numpy(np.asarray(s_pos[sel], np.float32)).to(pos.device)
        dt, dp = (tok.float() - ref_t).abs().flatten(1), (pos.float() - ref_p).abs().flatten(1)
        for k, d, ref in (("tok", dt, ref_t), ("pos", dp, ref_p)):
            mref = ref.abs().flatten(1).mean(1)
            acc[f"{k}_rel"] += (d.mean(1) / mref.clamp_min(1e-12)).tolist()
            acc[f"{k}_max"] += d.max(1).values.tolist()
            acc[f"{k}_abs"] += d.mean(1).tolist()
            acc[f"{k}_ref"] += mref.tolist()
    out = {"rows": int(len(rows)), "filled": int(filled)}
    if not len(rows):
        return out
    for k, scale, name in (("tok", 1.0, "tokens"), ("pos", 100.0, "pos_cm")):
        rel = np.asarray(acc[f"{k}_rel"])
        out[name] = {"max_abs": scale * float(np.max(acc[f"{k}_max"])), "mean_abs": scale * float(np.mean(acc[f"{k}_abs"])),
                     "rel_mean": float(np.mean(acc[f"{k}_abs"]) / max(np.mean(acc[f"{k}_ref"]), 1e-12)),
                     "row_rel_q50": float(np.median(rel)), "row_rel_max": float(rel.max()),
                     "worst_row": int(rows[int(rel.argmax())])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="data roots, or directories of shard_g* roots")
    ap.add_argument("--fusion-parity", type=int, default=0, metavar="N",
                    help="also check N random filled rows of every version-2 shard on the fusion stage (torch, GPU)")
    ap.add_argument("--parity-tol", type=float, default=2e-2,
                    help="largest relative mean error per row, mean|d| / mean|stored| of its tokens (and of its "
                         "positions). The neck rows are fp16, so parity is not 1e-5: 2e-2 is a placeholder, set it "
                         "from the first measurement (this check's row_rel_q50 / max, or bench_fusion.py --data)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="HoloBrain checkpoint root (or its pretrain/)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", help="fusion precision (LangConfig.fusion_dtype)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="row draw of --fusion-parity")
    ap.add_argument("--json", default=None, help="write the per-shard failures and parity numbers here")
    args = ap.parse_args(argv)
    failed = 0
    n = 0
    stage = None
    report = {}
    for a in args.roots:
        if not any(True for _ in roots([a])):
            print(f"note {a}: no shard (no lang_episodes.jsonl there or in a shard_g* below it)")
    for root in roots(args.roots):
        n += 1
        bad = check(root)
        par = None
        if args.fusion_parity > 0 and (root / "neck_buffer" / "header.json").exists():
            if stage is None:
                stage = make_stage(args)
            par = fusion_parity(root, stage, args.fusion_parity, np.random.default_rng(args.seed), args.device)
            if par.get("tokens"):
                t, p = par["tokens"], par["pos_cm"]
                print(f"parity {root}: {par['rows']} rows; tokens max {t['max_abs']:.3g} mean {t['mean_abs']:.3g} "
                      f"rel {t['rel_mean']:.3g} (per row q50 {t['row_rel_q50']:.3g}, max {t['row_rel_max']:.3g} at "
                      f"row {t['worst_row']}); positions max {p['max_abs']:.3g} cm mean {p['mean_abs']:.3g} cm rel "
                      f"{p['rel_mean']:.3g} (per row max {p['row_rel_max']:.3g})")
                for name, d in (("tokens", t), ("positions", p)):
                    if d["row_rel_max"] > args.parity_tol:
                        bad.append(f"fusion parity: row {d['worst_row']} {name} relative mean error "
                                   f"{d['row_rel_max']:.3g} > {args.parity_tol:g} (misaligned neck row?)")
        report[str(root)] = {"bad": bad, "parity": par}
        if bad:
            failed += 1
            print(f"FAIL {root}:")
            for b in bad[:8]:
                print(f"   {b}")
    print(f"{n} shards checked, {failed} failed")
    if args.json:
        Path(args.json).write_text(json.dumps({"tol": args.parity_tol, "shards": report}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
