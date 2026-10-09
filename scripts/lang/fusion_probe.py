"""E4 (FUSION_PLAN section 4, as amended by fusion_impl.md decision 7): does HoloBrain's frozen text-image fusion
single out the object a sentence names, on our scenes? No training.

    python scripts/lang/fusion_probe.py --val data/lang/val_grounding_v2 --out runs/lang/fusion_1009/e4

On version-2 validation shards (neck_buffer/ next to the item sidecar, token masks on), every --every-th tick (25)
of every T1 / T7 episode, every sampleable avoid / stay / upright rule on one object, with its training-template
sentence (eval_offline.axis_sentence: the sentence the offline score gives that state and rule). Frame t's neck
row is fused (FusionStage.fuse) with
- s: the rule's sentence;
- p: the bare referent phrase in it ("the red mug"; D2, a reported column);
- "": the content-free sentence ([CLS] [SEP]), the baseline. The fusion layers write LN(x) + gamma * delta and
  post-norm every cell, so the stage changes every token even without words; deltas against the unfused tokens
  (FusionStage.unfused, today's V0 tokens) are reported alongside.

Per (state, rule), with delta_j = ||fuse(s)_j - fuse("")_j|| over the scene tokens j of frame t (1200: 3 cameras x
(stride 16 + stride 32)):
- enrichment = mean delta_j over the named item's mask tokens (item sidecar token_masks > 0) / mean over the others;
  also against the unfused tokens, and for the bare phrase;
- contrast, for pairs of same-consequence rules on different objects in the same state: the share of the named
  item's mask tokens where delta under its own sentence exceeds delta under the other rule's sentence;
- word movement: ||w_fused_k - words_k|| / ||words_k|| over the sentence's tokens k (words = the cached words the
  text adapter reads), and over the object's words alone;
- position shift: ||pos(s)_j - pos("")_j|| in cm, over all tokens and the named item's (the spatial enhancer predicts
  depth from the fused maps);
- bi-attention, fusion layers 1 and 6 (FusionStage.capture_bi_attention): the share of the object words' attention
  over the 60 stride-64 cells (3 cameras x 4 x 5) that lands on the cells covering the named item (its stride-32
  token mask projected to the 4 x 5 grid by a 2 x 2 max), and that share over the uniform one (cells / 60).

Reported per task (and pooled) with episode-cluster bootstrap 95% intervals (eval_offline.boot_mean_ci). Gate, on
T1: enrichment >= 2 and contrast >= 0.7 (whole sentence, against ""). Writes <out>/fusion_probe.{json,md},
<out>/fusion_probe_rows.json.gz (every observation) and 6 overlay PNGs (<out>/overlays/): the shards hold no camera
images, so each draws the delta on the token grid of every camera (stride 16 and stride 32) with the named item's
mask outlined. Needs a GPU (the fusion stage; BERT for the sentences missing from the text caches).
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_CKPT = os.environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD")
CONSEQUENCES = ("avoid", "stay", "upright")          # the rules on one object (decision 7)
GATE = {"task": "selective_avoid", "enrichment": 2.0, "contrast": 0.7}
LAYERS = (0, 5)                                      # fusion layers 1 and 6
CAMS = ("left wrist", "right wrist", "head")
N_OVERLAYS = 6
NAN = float("nan")
# observation metrics (one per state and rule) and pair metrics (one per ordered same-consequence pair)
OBS_METRICS = (
    "enrich", "enrich_unfused", "enrich_phrase", "enrich_phrase_unfused",
    "delta_mean", "delta_named", "delta_unfused_mean", "delta_empty_unfused_mean",
    "word_move", "word_move_obj", "word_move_phrase",
    "pos_shift_cm", "pos_shift_named_cm", "pos_shift_unfused_cm",
) + tuple(f"att_{k}{p}_l{i + 1}" for i in LAYERS for p in ("", "_phrase") for k in ("share", "ratio"))
PAIR_METRICS = ("contrast", "contrast_unfused", "contrast_phrase")
_TEXT_ENCODER = None   # tests inject a stub text encoder, tokenizer and stage factory here
_TOKENIZER = None
_STAGE_FACTORY = None


def _sibling(name: str):
    """A script next to this one, as a module (scripts/lang is not a package)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- metrics (numpy) --


def enrichment(delta, mask) -> float:
    """Mean delta on the mask tokens / mean on the others (nan when either set is empty or the others' mean is 0)."""
    delta, mask = np.asarray(delta, np.float64).reshape(-1), np.asarray(mask, bool).reshape(-1)
    if not mask.any() or mask.all():
        return NAN
    off = float(delta[~mask].mean())
    return float(delta[mask].mean()) / off if off > 0 else NAN


def contrast(own, other, mask) -> float:
    """The share of the mask tokens where the delta under the item's own sentence exceeds the delta under the other
    sentence (nan for an empty mask)."""
    own, other = np.asarray(own, np.float64).reshape(-1), np.asarray(other, np.float64).reshape(-1)
    mask = np.asarray(mask, bool).reshape(-1)
    return float((own[mask] > other[mask]).mean()) if mask.any() else NAN


def cells_of(token_mask, level_shapes) -> np.ndarray:
    """token_mask [n_cam, n_tok] (levels 1 then 2 per camera, row-major; > 0 = the item) -> [n_cam * h3 * w3] bool:
    the stride-64 cells (level 3) covering the item, from its stride-32 (level 2) mask, each fine cell mapped to the
    coarse cell that holds it (a 2 x 2 max at 320 x 256: 8 x 10 -> 4 x 5). Camera-major, row-major, the order of
    the bi-attention's 60 cells."""
    (h1, w1), (h2, w2), (h3, w3) = (tuple(int(x) for x in level_shapes[i]) for i in (1, 2, 3))
    tm = np.asarray(token_mask)
    n_cam = tm.shape[0]
    l2 = tm.reshape(n_cam, -1)[:, h1 * w1:h1 * w1 + h2 * w2].reshape(n_cam, h2, w2) > 0
    rr = np.repeat((np.arange(h2) * h3 // h2)[:, None], w2, 1)
    cc = np.repeat((np.arange(w2) * w3 // w2)[None, :], h2, 0)
    out = np.zeros((n_cam, h3, w3), bool)
    for c in range(n_cam):
        np.logical_or.at(out[c], (rr, cc), l2[c])
    return out.reshape(-1)


def attention_share(cell, word_idx, cells) -> tuple[float, float]:
    """cell [heads, n, n_cells]: each word's softmax over the stride-64 cells (BiAttentionCapture.per_frame) ->
    (the object words' attention mass on ``cells``, averaged over heads and object words; the uniform share
    cells / n_cells). nan without object words or covered cells."""
    cell = np.asarray(cell, np.float64)
    cells = np.asarray(cells, bool).reshape(-1)
    idx = [int(k) for k in word_idx if 0 <= int(k) < cell.shape[1]]
    if not idx or not cells.any():
        return NAN, NAN
    return float(cell[:, idx][:, :, cells].sum(-1).mean()), float(cells.mean())


def bare_phrase(sentence: str, phrases) -> str | None:
    """The longest of the item's referent phrases (language.describe) that the sentence contains, as a whole phrase."""
    s = (sentence or "").lower()
    hits = [p for p in phrases if re.search(r"\b" + re.escape(str(p).lower()) + r"\b", s)]
    return max(hits, key=len) if hits else None


def _mean(x) -> float:
    x = np.asarray(x, np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else NAN


def state_metrics(items: list[dict], D: dict) -> tuple[list[dict], list[dict]]:
    """The metrics of one state from its fused frames.

    items: one per rule, {"rule", "consequence", "object", "sentence", "phrase", "mask" [N] bool, "cells" [n_cells]
    bool, "obj_s" / "obj_p" (object-word positions in s / p)}. D (fuse_state's output): "index" {text: frame},
    "delta" / "delta_u" [F, N] (||fuse(text) - fuse("")||, ||fuse(text) - unfused|| per token), "shift" / "shift_u"
    [F, N] (position change, cm), "move" [F, L] (word movement, nan past the sentence), "cell" {layer: [F] lists of
    [heads, n, n_cells]}. -> (observations, ordered pairs)."""
    ix, e = D["index"], D["index"][""]
    obs, pairs = [], []
    for it in items:
        m = np.asarray(it["mask"], bool)
        fs, fp = ix[it["sentence"]], ix.get(it["phrase"]) if it["phrase"] else None
        o = {"rule": it["rule"], "consequence": it["consequence"], "object": it["object"], "sentence": it["sentence"],
             "phrase": it["phrase"], "n_mask": int(m.sum()), "n_cells": int(np.asarray(it["cells"]).sum()),
             "enrich": enrichment(D["delta"][fs], m), "enrich_unfused": enrichment(D["delta_u"][fs], m),
             "enrich_phrase": enrichment(D["delta"][fp], m) if fp is not None else NAN,
             "enrich_phrase_unfused": enrichment(D["delta_u"][fp], m) if fp is not None else NAN,
             "delta_mean": _mean(D["delta"][fs]), "delta_named": _mean(D["delta"][fs][m]) if m.any() else NAN,
             "delta_unfused_mean": _mean(D["delta_u"][fs]), "delta_empty_unfused_mean": _mean(D["delta_u"][e]),
             "word_move": _mean(D["move"][fs]),
             "word_move_obj": _mean(D["move"][fs][[k for k in it["obj_s"] if k < D["move"].shape[1]]]),
             "word_move_phrase": _mean(D["move"][fp]) if fp is not None else NAN,
             "pos_shift_cm": _mean(D["shift"][fs]), "pos_shift_named_cm": _mean(D["shift"][fs][m]) if m.any() else NAN,
             "pos_shift_unfused_cm": _mean(D["shift_u"][fs])}
        for layer in LAYERS:
            cells = D["cell"].get(layer)
            for p, f, idx in (("", fs, it["obj_s"]), ("_phrase", fp, it["obj_p"])):
                sh, uni = attention_share(cells[f], idx, it["cells"]) if cells is not None and f is not None else (NAN, NAN)
                o[f"att_share{p}_l{layer + 1}"] = sh
                o[f"att_ratio{p}_l{layer + 1}"] = sh / uni if uni == uni and uni > 0 else NAN
        obs.append(o)
    for a in items:
        for b in items:
            if a is b or a["consequence"] != b["consequence"] or a["object"] == b["object"]:
                continue
            m = np.asarray(a["mask"], bool)
            if not m.any():
                continue
            fa, fb = ix[a["sentence"]], ix[b["sentence"]]
            pa, pb = ix.get(a["phrase"]) if a["phrase"] else None, ix.get(b["phrase"]) if b["phrase"] else None
            pairs.append({"rule": a["rule"], "other": b["rule"], "consequence": a["consequence"],
                          "contrast": contrast(D["delta"][fa], D["delta"][fb], m),
                          "contrast_unfused": contrast(D["delta_u"][fa], D["delta_u"][fb], m),
                          "contrast_phrase": (contrast(D["delta"][pa], D["delta"][pb], m)
                                              if pa is not None and pb is not None else NAN)})
    return obs, pairs


def _cluster(o: dict) -> str:
    return f"{o['src']}:{o['episode_id']}"   # eval_offline._ep: the pairs of one episode are correlated


def _stat(vals, B: int, seed: int) -> dict:
    from main.lang.eval_offline import boot_mean_ci

    vals = [(c, float(v)) for c, v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": NAN, "lo": NAN, "hi": NAN, "n": 0, "n_episodes": 0}
    lo, hi = boot_mean_ci(vals, B=B, seed=seed)
    return {"mean": float(np.mean([v for _, v in vals])), "lo": lo, "hi": hi, "n": len(vals),
            "n_episodes": len({c for c, _ in vals})}


def summarize(obs: list[dict], pairs: list[dict], B: int = 1000, seed: int = 0) -> dict:
    """{task or "pooled": {metric: {mean, lo, hi, n, n_episodes}, "n_states", "n_rules", "n_pairs"}}."""
    out = {}
    for task in sorted({o["task"] for o in obs}) + ["pooled"]:
        so = [o for o in obs if task == "pooled" or o["task"] == task]
        sp = [p for p in pairs if task == "pooled" or p["task"] == task]
        cell = {k: _stat([(_cluster(o), o.get(k)) for o in so], B, seed) for k in OBS_METRICS}
        cell.update({k: _stat([(_cluster(p), p.get(k)) for p in sp], B, seed) for k in PAIR_METRICS})
        cell.update({"n_states": len({(o["src"], o["episode_id"], o["t"]) for o in so}), "n_rules": len(so),
                     "n_pairs": len(sp), "n_episodes": len({_cluster(o) for o in so})})
        out[task] = cell
    return out


def gate(summary: dict, task: str = GATE["task"]) -> dict:
    """E4's gate (FUSION_PLAN E4, decision 7): on `task`, enrichment >= 2 and contrast >= 0.7 (the whole sentence,
    against the content-free one). passed None when either was not measured; "clearly" = both interval bounds pass."""
    c = summary.get(task) or {}
    en, co = c.get("enrich") or {}, c.get("contrast") or {}
    out = {"task": task, "thresholds": {"enrichment": GATE["enrichment"], "contrast": GATE["contrast"]},
           "enrichment": [en.get("mean", NAN), en.get("lo", NAN), en.get("hi", NAN)],
           "contrast": [co.get("mean", NAN), co.get("lo", NAN), co.get("hi", NAN)], "passed": None, "clearly": None}
    if en.get("n") and co.get("n"):
        out["passed"] = bool(en["mean"] >= GATE["enrichment"] and co["mean"] >= GATE["contrast"])
        out["clearly"] = bool(en["lo"] >= GATE["enrichment"] and co["lo"] >= GATE["contrast"])
    return out


def pick_overlays(obs: list[dict], n: int = N_OVERLAYS) -> list[int]:
    """Observation indices to draw: per task (T1, then T7, then any other) the highest, median and lowest
    enrichment among those with a mask, then the rest by enrichment until n."""
    ok = [i for i, o in enumerate(obs) if o["n_mask"] and np.isfinite(o["enrich"])]
    tasks = sorted({obs[i]["task"] for i in ok}, key=lambda t: ({"selective_avoid": 0, "gap_place": 1}.get(t, 2), t))
    out = []
    per = max(1, n // max(len(tasks), 1))
    for t in tasks:
        s = sorted((i for i in ok if obs[i]["task"] == t), key=lambda i: -obs[i]["enrich"])
        for i in [s[0], s[len(s) // 2], s[-1]][:per]:
            if i not in out:
                out.append(i)
    for i in sorted(ok, key=lambda i: -obs[i]["enrich"]):
        if len(out) >= n:
            break
        if i not in out:
            out.append(i)
    return out[:n]


def overlay_png(path: Path, delta, mask, level_shapes, title: str) -> None:
    """The delta of every scene token on each camera's token grid (rows: stride 16, stride 32), one colour scale,
    the named item's mask tokens outlined. No camera images: the shards store none."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = [tuple(int(x) for x in level_shapes[i]) for i in (1, 2)]
    n_tok = sum(h * w for h, w in shapes)
    delta = np.asarray(delta, np.float64).reshape(-1, n_tok)
    mask = np.asarray(mask, bool).reshape(-1, n_tok)
    n_cam = delta.shape[0]
    vmax = float(delta.max()) if delta.size and delta.max() > 0 else 1.0
    fig, axes = plt.subplots(2, n_cam, figsize=(3.4 * n_cam, 5.2), squeeze=False)
    im = None
    for c in range(n_cam):
        start = 0
        for lv, (h, w) in enumerate(shapes):
            ax = axes[lv, c]
            d = delta[c, start:start + h * w].reshape(h, w)
            m = mask[c, start:start + h * w].reshape(h, w)
            start += h * w
            im = ax.imshow(d, cmap="Blues", vmin=0.0, vmax=vmax, interpolation="nearest")
            for r, q in zip(*np.nonzero(m)):   # outline the mask: the cell edges that border a non-mask cell
                for dr, dq, xs, ys in ((-1, 0, (q - .5, q + .5), (r - .5, r - .5)), (1, 0, (q - .5, q + .5), (r + .5, r + .5)),
                                       (0, -1, (q - .5, q - .5), (r - .5, r + .5)), (0, 1, (q + .5, q + .5), (r - .5, r + .5))):
                    rr, qq = r + dr, q + dq
                    if not (0 <= rr < h and 0 <= qq < w) or not m[rr, qq]:
                        ax.plot(xs, ys, color="#eb6834", lw=2)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{CAMS[c] if c < len(CAMS) else f'camera {c}'}, stride {16 * 2 ** lv}", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label='||fuse(s) - fuse("")|| per token')
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------- fusion (torch) --


def fuse_state(stage, text, neck: dict, texts: list[str], layers=LAYERS) -> dict:
    """Frame t (neck: one row, leading dim 1) fused with each of `texts` ("" included; the neck expanded to one copy
    per text, so one fuse call groups them by length) and unfused -> state_metrics' D (numpy)."""
    import torch

    words, wmask = text.lookup(texts)
    F = len(texts)
    nk = {k: v.expand(F, *v.shape[1:]) for k, v in neck.items()}
    with stage.capture_bi_attention(layers) as cap:
        tok, pos, wf = stage.fuse(nk, words, wmask)
    tok_u, pos_u = stage.unfused(neck)
    e = texts.index("")
    with torch.no_grad():
        wm = wmask.to(wf.device).bool()
        w = words.to(wf.device).float()
        move = (wf - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
        move = torch.where(wm, move, torch.full_like(move, NAN))
        D = {"index": {s: i for i, s in enumerate(texts)},
             "delta": (tok - tok[e:e + 1]).norm(dim=-1), "delta_u": (tok - tok_u).norm(dim=-1),
             "shift": 100.0 * (pos - pos[e:e + 1]).norm(dim=-1), "shift_u": 100.0 * (pos - pos_u).norm(dim=-1),
             "move": move}
        D = {k: (v if k == "index" else v.float().cpu().numpy()) for k, v in D.items()}
    D["cell"] = {layer: [np.asarray(c.float().cpu().numpy() if hasattr(c, "cpu") else c) for c in cap.per_frame(layer)[1]]
                 for layer in layers}
    return D


def probe(sources, stage, text, tokenizer, every: int = 25, tasks=None, embodiments=None, bank=None,
          log_every: int = 50) -> tuple[list[dict], list[dict], dict]:
    """Every every-th tick of the sources' episodes -> (observations, pairs, {obs index: (delta float16 [N], mask
    [N] bool, level shapes)} for the overlays)."""
    from main.item_envs import language
    from main.lang.config import CANON_TASKS
    from main.lang.eval_offline import axis_sentence
    from main.lang.fusion import rows_to_neck
    from main.lang.scene_roles import ref_rows

    wa = _sibling("word_attention")
    bank = bank or language.load_bank()
    L = int(text.max_len)
    obs, pairs, store = [], [], {}
    t0, n_states = time.time(), 0
    for j, s in enumerate(sources):
        level_shapes = s.neck.level_shapes
        sel = s.t % every == 0
        if tasks:
            sel &= np.isin(s.task, [CANON_TASKS.index(x) for x in tasks])
        if embodiments:
            sel &= np.isin(s.emb, list(embodiments))
        tm_all = s.rs.buf.arrays["token_masks"]
        for i in np.flatnonzero(sel):
            eid, slot, t = int(s.eid[i]), int(s.rows[i]), int(s.t[i])
            rec = s.rs.records[eid]
            region_rows = (s.lines.get(eid) or {}).get("mask_region_rows") or {}
            items = []
            for rule in rec["rules"]:
                subj = rule.get("subject") or {}
                if not rule.get("sampleable") or rule["consequence"] not in CONSEQUENCES or "object" not in subj:
                    continue
                sent = axis_sentence(rule, rec, "train", (eid, t, rule["idx"]), bank)
                if not sent:
                    continue
                obj = rec["objects"][subj["object"]]
                phrase = bare_phrase(sent, language.describe(obj, rec, "train", bank))
                rr = ref_rows(rec, rule, region_rows)
                tm = np.asarray(tm_all[slot, rr]).max(0) if rr else np.zeros(tm_all.shape[2:], np.uint8)  # [n_cam, n_tok]
                ref_w = wa.referent_words(rule, rec, bank)

                def obj_idx(x):
                    return [k for k, kind in enumerate(wa.token_kinds(tokenizer, x, ref_w, L)) if kind == "referent"]
                items.append({"rule": rule["rule_id"], "consequence": rule["consequence"], "object": obj["item_id"],
                              "sentence": sent, "phrase": phrase, "mask": tm.reshape(-1) > 0,
                              "cells": cells_of(tm, level_shapes), "obj_s": obj_idx(sent),
                              "obj_p": obj_idx(phrase) if phrase else []})
            if not items:
                continue
            texts = [""] + list(dict.fromkeys(x for it in items for x in (it["sentence"], it["phrase"]) if x))
            neck = rows_to_neck(s.neck_rows(np.array([slot])), stage.device)
            D = fuse_state(stage, text, neck, texts)
            if D["delta"].shape[1] != items[0]["mask"].size:
                raise ValueError(f"{s.root}: {D['delta'].shape[1]} fused tokens but token masks of "
                                 f"{items[0]['mask'].size}")
            o_s, p_s = state_metrics(items, D)
            where = {"task": rec["task"], "src": j, "source": s.name, "episode_id": eid, "t": t}
            for it, o in zip(items, o_s):
                store[len(obs)] = (D["delta"][D["index"][it["sentence"]]].astype(np.float16), it["mask"], level_shapes)
                obs.append(dict(where, **o))
            pairs += [dict(where, **p) for p in p_s]
            n_states += 1
            if n_states % log_every == 0:
                print(f"[fusion_probe] {n_states} states, {len(obs)} rules, {len(pairs)} pairs, {time.time() - t0:.0f}s",
                      flush=True)
    return obs, pairs, store


# ---------------------------------------------------------------------------- report --


def _cell(c: dict, nd: int = 2) -> str:
    if not c or not c.get("n"):
        return "–"
    ci = f" [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]" if c["lo"] == c["lo"] else ""
    return f"{c['mean']:.{nd}f}{ci} ({c['n']})"


ROWS_MD = [
    ("enrich", "enrichment, sentence vs \"\" (gate)"), ("contrast", "contrast, sentence vs the other rule's (gate)"),
    ("enrich_unfused", "enrichment, sentence vs unfused"), ("enrich_phrase", "enrichment, bare phrase vs \"\" (D2)"),
    ("enrich_phrase_unfused", "enrichment, bare phrase vs unfused"), ("contrast_phrase", "contrast, bare phrases"),
    ("contrast_unfused", "contrast, against unfused"),
    ("delta_mean", "delta, sentence vs \"\", all tokens"), ("delta_named", "delta, sentence vs \"\", named item"),
    ("delta_unfused_mean", "delta, sentence vs unfused, all tokens"),
    ("delta_empty_unfused_mean", "delta, \"\" vs unfused, all tokens"),
    ("word_move", "word movement, all words"), ("word_move_obj", "word movement, object words"),
    ("word_move_phrase", "word movement, bare phrase"),
    ("pos_shift_cm", "position shift (cm), all tokens"), ("pos_shift_named_cm", "position shift (cm), named item"),
    ("pos_shift_unfused_cm", "position shift vs unfused (cm)"),
] + [(f"att_{k}{p}_l{i + 1}", f"bi-attention layer {i + 1}, object words on the item's cells: "
      f"{'share' if k == 'share' else 'share / uniform'}{' (bare phrase)' if p else ''}")
     for i in LAYERS for p in ("", "_phrase") for k in ("share", "ratio")]


def markdown(res: dict) -> str:
    S, g = res["summary"], res["gate"]
    tasks = [t for t in S if t != "pooled"] + ["pooled"]
    verdict = {True: "PASSED", False: "NOT PASSED", None: "not measured"}[g["passed"]]
    L = ["# E4: does the frozen fusion light up the named object? (FUSION_PLAN section 4)", "",
         f"Shards: {', '.join(res['config']['val'])}; every {res['config']['every']}th tick; {res['n']['states']} states, "
         f"{res['n']['rules']} (state, rule) rows, {res['n']['pairs']} ordered pairs. Fusion {res['config']['dtype']}, "
         f"checkpoint {res['config']['ckpt']}. Cells: mean [episode-bootstrap 95% interval] (observations). delta = "
         "||fuse(text) - fuse(\"\")|| per scene token of frame t unless it says unfused; enrichment = mean delta on "
         "the named item's mask tokens / mean on the others; contrast = share of the item's mask tokens where its own "
         "sentence's delta exceeds the other same-consequence rule's; word movement = ||w_fused - words|| / ||words||.",
         "", f"**Gate** ({g['task']}): enrichment >= {g['thresholds']['enrichment']} and contrast >= "
         f"{g['thresholds']['contrast']}: enrichment {g['enrichment'][0]:.2f} [{g['enrichment'][1]:.2f}, "
         f"{g['enrichment'][2]:.2f}], contrast {g['contrast'][0]:.2f} [{g['contrast'][1]:.2f}, {g['contrast'][2]:.2f}]: "
         f"**{verdict}**" + (" (both interval bounds pass)" if g.get("clearly") else "") + ".", "",
         "| metric | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for k, name in ROWS_MD:
        L.append(f"| {name} | " + " | ".join(_cell(S[t].get(k, {})) for t in tasks) + " |")
    L.append("| states / rules / pairs / episodes | " + " | ".join(
        f"{S[t]['n_states']} / {S[t]['n_rules']} / {S[t]['n_pairs']} / {S[t]['n_episodes']}" for t in tasks) + " |")
    L += ["", "## Overlays", "",
          "The shards store no camera images (only the neck maps), so each figure draws the delta on the token grid of "
          "every camera (rows: stride 16, stride 32; columns: left wrist, right wrist, head), the named item's mask "
          "tokens outlined. Per task the rules with the highest, median and lowest enrichment.", ""]
    for ov in res["overlays"]:
        L.append(f"- `{ov['png']}`: {ov['task']} episode {ov['episode_id']} t {ov['t']}, \"{ov['sentence']}\", "
                 f"enrichment {ov['enrich']:.2f}, {ov['n_mask']} mask tokens")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", nargs="+", required=True, help="version-2 validation roots (or directories / globs)")
    ap.add_argument("--text-cache", nargs="*", default=None, help="text caches (default: text_cache.pt next to --val)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tasks", default="selective_avoid,gap_place")
    ap.add_argument("--embodiments", default="ur5-wsg", help="comma list, or 'all'")
    ap.add_argument("--every", type=int, default=25, help="every n-th tick of an episode")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", help="fusion precision (LangConfig.fusion_dtype)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import main.lang  # noqa: F401  (main.__path__ bootstrap)
    from main.lang.data import LangSource
    from main.lang.text import TextStore
    from main.lang.train import caches_of, roots_of

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for r in roots_of(args.val):
        meta = json.loads((r / "shard.json").read_text()) if (r / "shard.json").exists() else {}
        if meta.get("masks") is False:
            print(f"[fusion_probe] skipping {r}: collected without token masks", flush=True)
            continue
        s = LangSource(r)
        if not s.has_neck:
            raise SystemExit(f"{r}: no neck_buffer/ (a version-1 shard): E4 needs shards collected with --lang-version 2")
        sources.append(s)
    if not sources:
        raise SystemExit("no version-2 validation shard with token masks")
    caches = caches_of(args.val) if args.text_cache is None else [Path(p) for p in args.text_cache]
    text = TextStore(caches, "holobrain_bert", args.device, extra_path=out / "text_cache_extra.pt", encoder=_TEXT_ENCODER)
    if _TOKENIZER is not None:
        tokenizer = _TOKENIZER
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if _STAGE_FACTORY is not None:
        stage = _STAGE_FACTORY(args)
    else:
        from main.lang.fusion import FusionStage

        stage = FusionStage(args.ckpt, device=args.device, dtype=args.dtype, chunk=args.chunk)
    t0 = time.time()
    tasks = tuple(args.tasks.split(",")) if args.tasks != "all" else None
    embs = tuple(args.embodiments.split(",")) if args.embodiments != "all" else None
    obs, pairs, store = probe(sources, stage, text, tokenizer, args.every, tasks, embs)
    if text.extra_path is not None and len(text) > text.n_cached:
        text.save_extra()
    summary = summarize(obs, pairs, B=args.boot, seed=args.seed)
    res = {"config": {"val": [str(s.root) for s in sources], "every": args.every, "tasks": list(tasks or []),
                      "embodiments": list(embs or []), "ckpt": str(args.ckpt), "dtype": args.dtype, "layers": list(LAYERS),
                      "boot": args.boot},
           "n": {"states": len({(o["src"], o["episode_id"], o["t"]) for o in obs}), "rules": len(obs),
                 "pairs": len(pairs)},
           "summary": summary, "gate": gate(summary), "wall_s": time.time() - t0, "overlays": []}
    (out / "overlays").mkdir(exist_ok=True)
    for i in pick_overlays(obs):
        o = obs[i]
        delta, mask, shapes = store[i]
        name = f"overlays/{o['task']}_{o['episode_id']}_t{o['t']}_{re.sub(r'[^a-z0-9]+', '-', o['rule'].lower())}.png"
        overlay_png(out / name, delta, mask, shapes,
                    f"{o['task']} episode {o['episode_id']} t {o['t']}: \"{o['sentence']}\" (enrichment {o['enrich']:.2f})")
        res["overlays"].append({"png": name, "task": o["task"], "episode_id": o["episode_id"], "t": o["t"],
                                "rule": o["rule"], "sentence": o["sentence"], "enrich": o["enrich"], "n_mask": o["n_mask"]})
    with gzip.open(out / "fusion_probe_rows.json.gz", "wt") as f:
        json.dump({"obs": obs, "pairs": pairs}, f, default=float)
    (out / "fusion_probe.json").write_text(json.dumps(res, indent=1, default=float))
    md = markdown(res)
    (out / "fusion_probe.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
