"""Where the model's attention over the sentence's words lands (review round 2 D1; cleaned up in round 3, B4 /
Part II E0c).

    python scripts/lang/word_attention.py --run runs/lang/<name> --ckpt checkpoint_final.pt --val data/lang/val \
        [--tasks selective_avoid,gap_place] [--out-name <dir>]

For validation states (every 5th tick, up to --states-per-task per task) and every sampleable rule's training
sentence, the attention over the word tokens of every layer that reads words: scene-reads-words (averaged over
the scene tokens), body-reads-words in each encoder block AND each critic block (both twins: the blocks that
produce Q), averaged over the valid body tokens. Each word token is one kind:

- referent: a word of the named item's phrases (language.describe for objects, describe_region for zones /
  bins, materials for bead sets: rice, beans, ...; a zone rule's pushed box is not its referent);
- cls / sep: [CLS] / [SEP], reported apart;
- punct: punctuation;
- other: every other word (the consequence: touch, tip, spill, move ...);
- pad: padding (should get none: the keys are masked).

Reported per task and layer: attention mass per kind, its share of the valid tokens, enrichment = mass / share;
the entropy per (head, query) averaged (nats; the entropy of the head- and query-averaged attention, an upper
bound, in brackets; uniform = ln of the token count); and the RMS of the adapted word features w' and of each
layer's keys per token kind (do large keys draw the attention?). Also: how much [CLS] and [SEP] vary across
sentences (the HoloBrain text encoder's sub-sentence mask makes special tokens attend only to themselves).
Writes <run>/eval_offline/<out-name>/word_attention.{md,json}.
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

KINDS = ("referent", "other", "cls", "sep", "punct")


def referent_words(rule: dict, rec: dict, bank=None) -> set[str]:
    """Words of the named item's phrases, from the same phrase builders the sentences use."""
    from main.item_envs import language

    cons = rule["consequence"]
    subj, ref = rule.get("subject") or {}, rule.get("reference") or {}
    phrases = []
    if cons in ("avoid", "upright", "stay") and "object" in subj:
        phrases += language.describe(rec["objects"][subj["object"]], rec, "train", bank)
    if cons in ("zone_exclude", "zone_include", "exclude_particles"):
        for k in ([ref["region"]] if "region" in ref else list(ref.get("regions") or [])):
            phrases += language.describe_region(rec["regions"][k], rec, "train", bank)
    if "particles" in subj:
        phrases += language.materials(rec["objects"][subj["particles"]], "train", bank)
    out = set()
    for ph in phrases:
        out.update(x for x in str(ph).lower().replace("_", " ").split() if x and x not in ("the", "a", "an"))
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
        elif t == "[CLS]":
            kinds.append("cls")
        elif t == "[SEP]":
            kinds.append("sep")
        elif not any(c.isalnum() for c in t):
            kinds.append("punct")
        elif t.lstrip("#") in ref_pieces:
            kinds.append("referent")
        else:
            kinds.append("other")
    return kinds


def entropy(p: np.ndarray) -> float:
    p = np.clip(p / max(p.sum(), 1e-12), 1e-12, 1)
    return float(-(p * np.log(p)).sum())


def key_rms(mha, w):
    """RMS over channels of the keys mha builds from w [B, L, D] (its in_proj k slice)."""
    import torch.nn.functional as F

    D = w.shape[-1]
    k = F.linear(w, mha.in_proj_weight[D:2 * D], mha.in_proj_bias[D:2 * D] if mha.in_proj_bias is not None else None)
    return k.pow(2).mean(-1).sqrt()


_TEXT_ENCODER = None   # tests inject a stub text encoder here


def special_token_spread(sentences, device) -> dict:
    """How much the frozen features of [CLS] / [SEP] (and of real words, for scale) vary across sentences: the
    std over sentences relative to the mean feature norm, and the mean cosine to the mean feature."""
    from main.item_envs.language.text_encoder import HoloBrainText

    enc = HoloBrainText(device=device)
    emb, valid = enc.embed(list(sentences))
    emb, valid = emb.cpu().numpy(), valid.cpu().numpy()
    n = valid.sum(1)
    cls = emb[:, 0]
    sep = emb[np.arange(len(n)), n - 1]
    word = emb[:, 1]

    def spread(x):
        mu = x.mean(0)
        cos = (x @ mu) / (np.linalg.norm(x, axis=1) * np.linalg.norm(mu) + 1e-12)
        return {"rel_std": float(x.std(0).mean() / (np.abs(x).mean() + 1e-12)), "mean_cos_to_mean": float(cos.mean()),
                "rms": float(np.sqrt((x ** 2).mean()))}

    return {"n_sentences": len(n), "sub_sentence_mask": bool(enc.sub_sentence), "cls": spread(cls), "sep": spread(sep),
            "first_word": spread(word)}


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
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--states-per-task", type=int, default=200)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--no-special-check", action="store_true", help="skip the [CLS] / [SEP] spread (needs BERT)")
    args = ap.parse_args(argv)
    roots = roots_of(args.val)
    ev = Evaluator(Path(args.run), args.ckpt, roots, device=args.device, text_caches=caches_of(args.val),
                   batch=args.batch, text_encoder=_TEXT_ENCODER, tasks=None if args.tasks == "all" else args.tasks.split(","))
    L = int(ev.cfg.max_text_tokens)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bank = language.load_bank()
    states = ev.states(args.states_per_task)
    acc = defaultdict(lambda: defaultdict(lambda: {"mass": defaultdict(float), "share": defaultdict(float),
                                                    "ent_hq": 0.0, "ent_avg": 0.0, "ent_uniform": 0.0, "n": 0,
                                                    "krms": defaultdict(float), "kcnt": defaultdict(int)}))
    wrms = defaultdict(lambda: {"sum": defaultdict(float), "cnt": defaultdict(int)})
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
                items.append(((j, i), rec["task"], sent, token_kinds(tok, sent, referent_words(rule, rec, bank), L),
                              rule["idx"]))
    print(f"[word_attention] {len(states)} states, {len(items)} (state, sentence) items", flush=True)
    d = ev.dev
    for a in range(0, len(items), args.batch):
        chunk = items[a:a + args.batch]
        reads = [ev.sources[k[0]].read(k[1]) for k, _, _, _, _ in chunk]
        b = {key: np.stack([r[key] for r in reads]) for key in reads[0]}
        words, wmask = ev.text.lookup([c[2] for c in chunk])
        kw = {}
        if getattr(ev.cfg, "oracle_ref_tag", False):
            c_, v_ = ev.ref_tag([c[0] for c in chunk], [c[4] for c in chunk])
            kw = {"ref_center": _t(c_, dev=d), "ref_valid": _t(v_, torch.bool, d)}
        enc_stores = set_probe(ev.pe, True, heads=True)
        crit_stores = set_probe(ev.critics, True, heads=True)
        with torch.no_grad():
            enc = ev.pe(_t(b["state_tokens"], dev=d), _t(b["body_tokens"], dev=d), _t(b["link_pos"], dev=d),
                        _t(b["body_mask"], torch.bool, d), _t(b["joint_index"], torch.int64, d),
                        _t(b["scene_tokens"], dev=d), _t(b["scene_pos"], dev=d), words=words, word_mask=wmask, **kw)
            ev.critics.qmin(enc, _t(b["dtheta"], dev=d), _t(b["Jlin"], dev=d), _t(b["Jang"], dev=d),
                            _t(b["joint_index"], torch.int64, d))
            # keys of every word-reading attention, in the order the stores below follow
            mhas = ([("scene reads words", ev.pe.scene_reads_words.attn)] if hasattr(ev.pe, "scene_reads_words") else [])
            mhas += [(f"body reads words, encoder block {n + 1}", blk.attn_bw) for n, blk in enumerate(ev.pe.blocks)
                     if hasattr(blk, "attn_bw")]
            for c_i, head in enumerate((ev.critics.c1, ev.critics.c2)):
                mhas += [(f"body reads words, critic {c_i + 1} block {n + 1}", blk.attn_bw)
                         for n, blk in enumerate(head.blocks) if hasattr(blk, "attn_bw")]
            krms = {name: key_rms(m, enc.words).float().cpu().numpy() for name, m in mhas}
            w_rms = enc.words.float().pow(2).mean(-1).sqrt().cpu().numpy()
        set_probe(ev.pe, False), set_probe(ev.critics, False)
        bm = enc.mask.bool()   # [B, 22]: links and payload tokens
        layers = []
        names = iter(n for n, _ in mhas)
        for st in enc_stores + crit_stores:
            if "scene_word_attn_heads" in st:
                layers.append((next(names), st["scene_word_attn_heads"].float(), None))      # [B, H, 3600, L]
            if "body_word_attn_heads" in st:
                layers.append((next(names), st["body_word_attn_heads"].float(), bm))         # [B, H, 22, L]
        keym = enc.word_mask.bool()                 # the keys the word reads saw (V2 drops [CLS] / [SEP])
        wm = keym.cpu().numpy()
        for name, att, qmask in layers:
            # head- and query-averaged attention [B, L], and the mean entropy over (head, valid query) [B]
            if qmask is None:
                avg = att.mean((1, 2))
            else:
                qmf = qmask.float()[:, None, :, None]
                avg = (att * qmf).sum((1, 2)) / (qmf.sum((1, 2)) * att.shape[1]).clamp_min(1)
            a_ = att * keym[:, None, None, :].float()
            a_ = a_ / a_.sum(-1, keepdim=True).clamp_min(1e-12)
            e_ = -(a_ * a_.clamp_min(1e-12).log()).sum(-1)                                   # [B, H, Nq]
            if qmask is None:
                ent_hq = e_.mean((1, 2))
            else:
                qm_ = qmask.float()[:, None, :]
                ent_hq = (e_ * qm_).sum((1, 2)) / (qm_.sum((1, 2)) * e_.shape[1]).clamp_min(1)
            avg, ent_hq = avg.cpu().numpy(), ent_hq.cpu().numpy()
            for k, (_, task, _, kinds, _) in enumerate(chunk):
                valid = wm[k][:len(kinds)]
                nv = int(valid.sum())
                if nv == 0:
                    continue
                kk = np.array(kinds)
                p = avg[k][:len(kinds)]
                for grp in (task, "pooled"):
                    cell = acc[grp][name]
                    for kind in KINDS:
                        sel = (kk == kind) & valid
                        cell["mass"][kind] += float(p[sel].sum())
                        cell["share"][kind] += float(sel.sum()) / nv
                        if sel.any():
                            cell["krms"][kind] += float(krms[name][k][:len(kinds)][sel].mean())
                            cell["kcnt"][kind] += 1
                    cell["mass"]["pad"] += float(p[~valid].sum())
                    cell["ent_hq"] += float(ent_hq[k])
                    cell["ent_avg"] += entropy(p[valid])
                    cell["ent_uniform"] += float(np.log(nv))
                    cell["n"] += 1
        for k, (_, task, _, kinds, _) in enumerate(chunk):
            valid = wm[k][:len(kinds)]
            kk = np.array(kinds)
            for grp in (task, "pooled"):
                for kind in KINDS:
                    sel = (kk == kind) & valid
                    if sel.any():
                        wrms[grp]["sum"][kind] += float(w_rms[k][:len(kinds)][sel].mean())
                        wrms[grp]["cnt"][kind] += 1
    out = {"layers": {}, "w_rms": {}}
    Lmd = [f"# Word attention: {Path(args.run).name} / {args.ckpt} (step {ev.ckpt_step})", "",
           "Attention over the sentence's tokens by token kind: mass, and enrichment = mass / the kind's share of the "
           "valid tokens (1 = no preference). Entropy in nats over the valid tokens: the mean over heads and queries "
           "(the head- and query-averaged attention's, an upper bound, then uniform, in brackets). key RMS: the RMS of "
           "the layer's keys per token kind. Review round 2 D1, round 3 B4 / E0c.", ""]
    for grp in ["pooled"] + sorted(k for k in acc if k != "pooled"):
        Lmd += [f"## {grp}", "", "| layer | n | " + " | ".join(f"{k} mass (enrich.)" for k in KINDS) +
                " | pad mass | entropy per head, query (averaged, uniform) | key RMS: " + " / ".join(KINDS) + " |",
                "|---|---|" + "---|" * len(KINDS) + "---|---|---|"]
        out["layers"][grp] = {}
        for name, cell in acc[grp].items():
            n = max(cell["n"], 1)
            row = {k: {"mass": cell["mass"][k] / n, "share": cell["share"][k] / n,
                       "key_rms": cell["krms"][k] / max(cell["kcnt"][k], 1)} for k in KINDS}
            for k in KINDS:
                row[k]["enrichment"] = row[k]["mass"] / max(row[k]["share"], 1e-9)
            row.update({"pad_mass": cell["mass"]["pad"] / n, "entropy_head_query": cell["ent_hq"] / n,
                        "entropy_averaged": cell["ent_avg"] / n, "entropy_uniform": cell["ent_uniform"] / n,
                        "n": cell["n"]})
            out["layers"][grp][name] = row
            Lmd.append(f"| {name} | {cell['n']} | " + " | ".join(
                f"{row[k]['mass']:.2f} ({row[k]['enrichment']:.1f}x)" for k in KINDS) +
                f" | {row['pad_mass']:.3f} | {row['entropy_head_query']:.2f} ({row['entropy_averaged']:.2f}, "
                f"{row['entropy_uniform']:.2f}) | " + " / ".join(f"{row[k]['key_rms']:.1f}" for k in KINDS) + " |")
        w = wrms[grp]
        out["w_rms"][grp] = {k: w["sum"][k] / max(w["cnt"][k], 1) for k in KINDS}
        Lmd += ["", "RMS of the adapted word features w' per kind: " + ", ".join(
            f"{k} {out['w_rms'][grp][k]:.2f}" for k in KINDS), ""]
    if not args.no_special_check:
        sents = sorted({c[2] for c in items})[:400]
        sp = special_token_spread(sents, args.device)
        out["special_tokens"] = sp
        Lmd += ["## [CLS] / [SEP] across sentences (frozen text features)", "",
                f"The HoloBrain text encoder uses its sub-sentence mask: {sp['sub_sentence_mask']}. Over "
                f"{sp['n_sentences']} training sentences: relative std (std over sentences / mean |feature|) and the mean "
                "cosine to the mean feature (1.0 = the same vector for every sentence). The first real word is shown "
                "for scale.", "", "| token | relative std | mean cosine to the mean | RMS |", "|---|---|---|---|"]
        for k in ("cls", "sep", "first_word"):
            Lmd.append(f"| {k} | {sp[k]['rel_std']:.4f} | {sp[k]['mean_cos_to_mean']:.4f} | {sp[k]['rms']:.2f} |")
        Lmd.append("")
    od = Path(args.run) / "eval_offline" / (args.out_name or Path(args.ckpt).stem)
    od.mkdir(parents=True, exist_ok=True)
    (od / "word_attention.json").write_text(json.dumps(out, indent=1))
    (od / "word_attention.md").write_text("\n".join(Lmd) + "\n")
    print("\n".join(Lmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
