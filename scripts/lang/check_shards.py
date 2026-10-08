"""Integrity check of collected shards / rings (review O1): every episode line's rows [r0, r1) carry that
episode's id in BOTH buffers, t_in_ep counts 0..n-1 within them, lines do not overlap, and the header n is
the end of the last line. Two writers on one shard break at least one of these.

    python scripts/lang/check_shards.py data/lang/warmup data/lang/val [runs/lang/<run>/ring ...]

Reads only (memmaps opened read-only). Exit 1 if any shard fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


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
        if r1 <= r0:
            bad.append(f"episode {eid}: empty rows [{r0}, {r1})")
            continue
        if (np.asarray(ep_buf[r0:r1]) != eid).any() or (np.asarray(ep_side[r0:r1]) != eid).any():
            bad.append(f"episode {eid}: rows [{r0}, {r1}) hold other episode ids")
        if not np.array_equal(np.asarray(t_side[r0:r1]), np.arange(r1 - r0)):
            bad.append(f"episode {eid}: t_in_ep is not 0..{r1 - r0 - 1}")
        end = r1
    if (n_buf, n_side) != (end, end) and lines:
        bad.append(f"headers n = ({n_buf}, {n_side}) but the last episode ends at {end}")
    return bad


def main(argv) -> int:
    failed = 0
    n = 0
    for root in roots(argv):
        n += 1
        bad = check(root)
        if bad:
            failed += 1
            print(f"FAIL {root}:")
            for b in bad[:8]:
                print(f"   {b}")
    print(f"{n} shards checked, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
