#!/usr/bin/env python
"""Build the relabeling text-token cache (docs/item_envs/relabeling.md).

Enumerates every sentence the RelabelSampler can draw — (template x unique
referent phrase) for each rule of each episode, in the train, held-out
template and held-out word splits — and embeds each ONCE with the frozen
HoloBrain BERT + text_feat_map (fp16 [L, 256], L <= --max-len, padded, with
a mask). A sentence missing from the cache is an error at train time, so
the split stays controlled.

The text encoder is main/item_envs/language/text_encoder.HoloBrainText
(the checkpoint's own BERT + text_feat_map).

    python scripts/item_envs/build_text_cache.py --episodes <sidecar>/episodes \\
        --out <sidecar>/text_cache.pt [--device cuda]
    python scripts/item_envs/build_text_cache.py --metas 'runs/item_envs/calib_*/*/*/*/seed_*/meta.json' \\
        --out runs/item_envs/text_cache.pt          # from saved probe / calibration runs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ENVS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENVS))

from main.item_envs import language  # noqa: E402


def collect_sentences(records: list[dict]) -> dict[str, str]:
    """sentence -> split, over every usable rule of every record."""
    out: dict[str, str] = {}
    for rec in records:
        for rule in rec["rules"]:
            if rule.get("set") or rule.get("ambiguous"):
                continue
            for split in language.SPLITS:
                if split == "train" and not rule.get("sampleable"):
                    continue
                for s in language.sentences(rule, rec, split):
                    out.setdefault(s, split)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="*", default=[], help="directories of episodes/{id}.json records")
    ap.add_argument("--metas", nargs="*", default=[], help="globs of runner meta.json files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--list-only", action="store_true", help="count the sentences, embed nothing")
    args = ap.parse_args(argv)

    records = []
    for d in args.episodes:
        records += [json.loads(f.read_text()) for f in sorted(Path(d).glob("*.json"))]
    for g in args.metas:
        for i, f in enumerate(sorted(glob.glob(g))):
            meta = json.loads(Path(f).read_text())
            if not meta.get("error"):
                records.append(language.record_from_meta(meta, i))
    sents = collect_sentences(records)
    counts = {s: sum(1 for v in sents.values() if v == s) for s in language.SPLITS}
    print(f"[text-cache] {len(records)} episodes -> {len(sents)} sentences {counts}")
    if args.list_only:
        return 0

    import torch

    from main.item_envs.language.text_encoder import HoloBrainText

    pipe = HoloBrainText(device=args.device)
    keys = sorted(sents)
    L = int(args.max_len)
    emb = torch.zeros(len(keys), L, 256, dtype=torch.float16)
    mask = torch.zeros(len(keys), L, dtype=torch.bool)
    too_long = 0
    for a in range(0, len(keys), args.batch):
        chunk = keys[a:a + args.batch]
        with torch.no_grad():
            e, v = pipe.embed(chunk)
        e, v = e.float().cpu(), v.cpu().bool()
        n = min(L, e.shape[1])
        too_long += int((v.sum(1) > L).sum())
        emb[a:a + len(chunk), :n] = e[:, :n].half()
        mask[a:a + len(chunk), :n] = v[:, :n]
        print(f"[text-cache] {a + len(chunk)}/{len(keys)}", flush=True)
    if too_long:
        print(f"[text-cache] WARNING: {too_long} sentences longer than {L} tokens were truncated")
    out = {"index": {s: i for i, s in enumerate(keys)}, "split": [sents[s] for s in keys],
           "emb": emb, "mask": mask, "max_len": L, "ckpt": str(pipe.ckpt)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.out)
    print(f"[text-cache] wrote {args.out}: {len(keys)} sentences, emb {tuple(emb.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
