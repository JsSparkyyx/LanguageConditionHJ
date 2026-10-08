"""Review round 2 D1 check: where the model's attention over the sentence's words lands.

    python scripts/lang/word_attention.py --run runs/lang/<name> --ckpt checkpoint_final.pt --val data/lang/val

For validation states (every 5th tick, up to --states-per-task per task) and every sampleable rule's
training sentence: the attention weights of body-reads-words (each trunk block, averaged over the valid
body tokens) and of scene-reads-words (averaged over the scene tokens), split by what each word token is:

- referent: a word of the named item (object category, synonyms, colour, size; region names and colour;
  "beads" for bead rules);
- special: [CLS] / [SEP];
- punct: punctuation;
- other: every other word (the consequence: touch, tip, spill, move ...);
- pad: padding (should get none: the keys are masked).

Reported per task: attention mass on each kind, its share of the valid tokens, enrichment = mass / share,
and the attention entropy over the valid tokens (nats; uniform would be ln of the token count). Writes
<run>/eval_offline/<ckpt>/word_attention.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

KINDS = ("referent", "other", "special", "punct")


def referent_words(rule: dict, rec: dict) -> set[str]:
    subj, ref = rule.get("subject") or {}, rule.get("reference") or {}
    words = []
    if "object" in subj:
        o = rec["objects"][subj["object"]]
        words += [o.get("category") or ""] + list(o.get("synonyms") or []) + [o.get("color") or "", o.get("size") or ""]
    if "particles" in subj:
        words.append("beads")
    for k in ([ref["region"]] if "region" in ref else list(ref.get("regions") or [])):
        g = rec["regions"][k]
        words += list(g.get("names") or []) + [g.get("color") or ""]
    out = set()
    for w in words:
        out.update(x for x in str(w).lower().replace("_", " ").split() if x)
    return out


def token_kinds(tokenizer, sentence: str, ref_words: set[str], L: int) -> list[str]:
    """Kind of each of the L positions the text features use (same tokenization: bert-base-uncased,
    max_length L, truncation, padding to L)."""
    enc = tokenizer(sentence, max_length=L, truncation=True, padding="max_length")
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    ref_pieces = set()
    for w in ref_words:
        ref_pieces.update(p.lstrip("#") for p in tokenizer.tokenize(w))
    kinds = []
    for t in toks:
        if t == "[PAD]":
            kinds.append("pad")
        elif t in ("[CLS]", "[SEP]"):
            kinds.append("special")
        elif not any(c.isalnum() for c in t):
            kinds.append("punct")
        elif t.lstrip("#") in ref_pieces:
            kinds.append("referent")
        else:
            kinds.append("other")
    return kinds


_TEXT_ENCODER = None   # tests inject a stub text encoder here


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer

    import main.lang  # noqa: F401
    from main.item_envs import language
    from main.lang.eval_offline import Evaluator, axis_sentence
    from main.lang.network import set_probe
    from main.lang.train import _t, caches_of, roots_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--states-per-task", type=int, default=200)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    roots = roots_of(args.val)
    ev = Evaluator(Path(args.run), args.ckpt, roots, device=args.device, text_caches=caches_of(args.val),
                   batch=args.batch, text_encoder=_TEXT_ENCODER)
    L = int(ev.cfg.max_text_tokens)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bank = language.load_bank()
    states = ev.states(args.states_per_task)
    # per task: per layer: sum of mass per kind, sum of share per kind, entropy sum, count
    acc = defaultdict(lambda: defaultdict(lambda: {"mass": defaultdict(float), "share": defaultdict(float),
                                                    "ent": 0.0, "ent_uniform": 0.0, "n": 0}))
    items = []
    for j, i in states:
        s = ev.sources[j]
        eid = int(s.eid[i])
        rec = s.rs.records[eid]
        for rule in rec["rules"]:
            if not rule.get("sampleable"):
                continue
            sent = axis_sentence(rule, rec, "train", (eid, int(s.t[i]), rule["idx"]), bank)
            if sent:
                items.append(((j, i), rec["task"], sent, token_kinds(tok, sent, referent_words(rule, rec), L)))
    print(f"[word_attention] {len(states)} states, {len(items)} (state, sentence) items", flush=True)
    d = ev.dev
    for a in range(0, len(items), args.batch):
        chunk = items[a:a + args.batch]
        reads = [ev.sources[k[0]].read(k[1]) for k, _, _, _ in chunk]
        b = {key: np.stack([r[key] for r in reads]) for key in reads[0]}
        words, wmask = ev.text.lookup([c[2] for c in chunk])
        stores = set_probe(ev.pe, True)
        with torch.no_grad():
            enc = ev.pe(_t(b["state_tokens"], dev=d), _t(b["body_tokens"], dev=d), _t(b["link_pos"], dev=d),
                  _t(b["body_mask"], torch.bool, d), _t(b["joint_index"], torch.int64, d),
                  _t(b["scene_tokens"], dev=d), _t(b["scene_pos"], dev=d), words=words, word_mask=wmask)
        set_probe(ev.pe, False)
        bm = enc.mask.bool()   # [B, 22]: the 20 links and 2 payload tokens (the data's body_mask has the links only)
        layers = []
        for st in stores:
            if "scene_word_attn" in st:
                layers.append(("scene reads words", st["scene_word_attn"].float().mean(1)))   # [B, L]
            if "body_word_attn" in st:
                w = st["body_word_attn"].float()                                              # [B, 22, L]
                m = bm.float().unsqueeze(-1)
                layers.append((f"body reads words, block {sum(1 for n, _ in layers if n.startswith('body')) + 1}",
                               (w * m).sum(1) / m.sum(1).clamp_min(1)))
        wm = wmask.bool().cpu().numpy()
        for name, att in layers:
            att = att.cpu().numpy()
            for k, (_, task, _, kinds) in enumerate(chunk):
                valid = wm[k][:len(kinds)]
                p = att[k][:len(kinds)]
                kk = np.array(kinds)
                nv = int(valid.sum())
                if nv == 0:
                    continue
                for grp in (task, "pooled"):
                    cell = acc[grp][name]
                    for kind in KINDS:
                        sel = (kk == kind) & valid
                        cell["mass"][kind] += float(p[sel].sum())
                        cell["share"][kind] += float(sel.sum()) / nv
                    cell["mass"]["pad"] += float(p[~valid].sum())
                    pv = np.clip(p[valid] / max(p[valid].sum(), 1e-12), 1e-12, 1)
                    cell["ent"] += float(-(pv * np.log(pv)).sum())
                    cell["ent_uniform"] += float(np.log(nv))
                    cell["n"] += 1
    out = {}
    Lmd = [f"# Word attention: {Path(args.run).name} / {args.ckpt} (step {ev.ckpt_step})", "",
           "Mass of the attention over the sentence's tokens, by token kind, and enrichment = mass / the kind's "
           "share of the valid tokens (1 = no preference). Entropy in nats over the valid tokens (uniform in "
           "brackets). Review round 2 D1.", ""]
    for grp in ["pooled"] + sorted(k for k in acc if k != "pooled"):
        Lmd += [f"## {grp}", "", "| layer | n | " + " | ".join(f"{k} mass (enrich.)" for k in KINDS) +
                " | pad mass | entropy (uniform) |", "|---|---|" + "---|" * len(KINDS) + "---|---|"]
        out[grp] = {}
        for name, cell in acc[grp].items():
            n = max(cell["n"], 1)
            row = {k: {"mass": cell["mass"][k] / n, "share": cell["share"][k] / n} for k in KINDS}
            for k in KINDS:
                row[k]["enrichment"] = row[k]["mass"] / max(row[k]["share"], 1e-9)
            row["pad_mass"], row["entropy"], row["entropy_uniform"], row["n"] = (
                cell["mass"]["pad"] / n, cell["ent"] / n, cell["ent_uniform"] / n, cell["n"])
            out[grp][name] = row
            Lmd.append(f"| {name} | {cell['n']} | " + " | ".join(
                f"{row[k]['mass']:.2f} ({row[k]['enrichment']:.1f}x)" for k in KINDS) +
                f" | {row['pad_mass']:.3f} | {row['entropy']:.2f} ({row['entropy_uniform']:.2f}) |")
        Lmd.append("")
    od = Path(args.run) / "eval_offline" / Path(args.ckpt).stem
    od.mkdir(parents=True, exist_ok=True)
    (od / "word_attention.json").write_text(json.dumps(out, indent=1))
    (od / "word_attention.md").write_text("\n".join(Lmd) + "\n")
    print("\n".join(Lmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
