"""TextStore (VANILLA_PLAN 4.3) and the text backends of Q3.

``TextStore(cache_paths, backend, device, extra_path)`` holds word tokens for
sentences: ``lookup(sentences) -> (words [B, L, d] float32, mask [B, L] bool)``
on the device. Caches come from ``envs/scripts/item_envs/build_text_cache.py``
(holobrain_bert) or ``python -m main.lang.text build`` (any backend):
``{"index": {sentence: i}, "emb": fp16 [N, L, d], "mask": [N, L], "split": [N]}``.
Misses (heldout_item sentences, which use training templates on rules the cache
builder skips; deployment sentences) are encoded with the backend in one batch,
added, and persisted to ``extra_path`` every 1000 new entries and at
checkpoints (``save_extra``).

Backends (``LangConfig.text_backend``):
- ``holobrain_bert``: frozen HoloBrain BERT + text_feat_map, d = 256 (default);
- ``bert_raw``: the same BERT without text_feat_map, d = 768 (the adapter's
  in_proj Linear(768, 256) is trainable);
- ``sentence_generic``: a frozen general-purpose encoder with token outputs
  (all-mpnet-base-v2 from a local path), d = 768, trainable Linear(d, 256).

    python -m main.lang.text build --backend bert_raw \\
        --episodes data/lang/warmup/shard_g*/item_buffer/episodes --out data/lang/warmup/text_cache_bert_raw.pt
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

from . import REPO, ensure_item_path

ensure_item_path()

MAX_LEN = 24
SAVE_EVERY = 1000


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class RawBertText:
    """HoloBrain's BERT without text_feat_map: [B, L, 768] (same tokenizer,
    masks and layer averaging as HoloBrainText.embed)."""

    def __init__(self, device: str = "cuda"):
        from main.item_envs.language.text_encoder import HoloBrainText

        self.base = HoloBrainText(device=device)
        self.device = device

    def embed(self, sentences):
        import torch
        from robo_orchard_lab.models.bip3d.bert import generate_masks_with_special_tokens_and_transfer_map
        from robo_orchard_lab.utils.transformers_compat import batch_encode_tokenizer, prepare_hf_attention_mask

        b = self.base
        tok = batch_encode_tokenizer(
            b.tokenizer, list(sentences), max_length=MAX_LEN, padding="max_length",
            return_special_tokens_mask=True, return_tensors="pt", truncation=True,
        ).to(self.device)
        if b.sub_sentence:
            mask, position_ids = generate_masks_with_special_tokens_and_transfer_map(tok, b.special_ids)
            token_type_ids = tok["token_type_ids"]
        else:
            mask, position_ids, token_type_ids = tok.attention_mask, None, None
        with torch.no_grad():
            out = b.bert(input_ids=tok.input_ids, attention_mask=prepare_hf_attention_mask(b.bert, mask, tok.input_ids.shape),
                         position_ids=position_ids, token_type_ids=token_type_ids, output_hidden_states=True)
            layers = out.hidden_states[1:]
            feats = torch.stack(layers[-b.n_layers:], 1).mean(1) / b.n_layers
            if mask.dim() == 2:
                feats = feats * mask.unsqueeze(-1).float()
        return feats.float(), tok.attention_mask.bool()


class GenericText:
    """A frozen sentence encoder's token outputs (last hidden state), [B, L, d]."""

    def __init__(self, path: str, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval().requires_grad_(False)
        self.device = device

    def embed(self, sentences):
        import torch

        tok = self.tokenizer(list(sentences), padding="max_length", max_length=MAX_LEN, truncation=True,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**tok).last_hidden_state
        m = tok["attention_mask"].bool()
        return (out * m.unsqueeze(-1)).float(), m


def make_encoder(backend: str, device: str = "cuda", backend_paths: dict | None = None):
    if backend == "holobrain_bert":
        from main.item_envs.language.text_encoder import HoloBrainText

        return HoloBrainText(device=device)
    if backend == "bert_raw":
        return RawBertText(device=device)
    if backend == "sentence_generic":
        # the configured path (Delta's cache), else $SENTENCE_GENERIC_DIR, else $HF_HOME/all-mpnet-base-v2 (another
        # machine, 2026-11-09: the config default names Delta's path and runs keep their saved config)
        tries = [(backend_paths or {}).get("sentence_generic"), os.environ.get("SENTENCE_GENERIC_DIR"),
                 os.path.join(os.environ["HF_HOME"], "all-mpnet-base-v2") if os.environ.get("HF_HOME") else None]
        path = next((p for p in tries if p and Path(p).exists()), None)
        if path is None:
            raise FileNotFoundError(f"sentence_generic: no local model at {[p for p in tries if p]}")
        return GenericText(path, device=device)
    raise ValueError(f"unknown text backend {backend!r}")


def text_dim(backend: str) -> int:
    return 256 if backend in ("holobrain_bert", "holobrain_qwen") else 768


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TextStore:
    def __init__(self, cache_paths=(), backend: str = "holobrain_bert", device: str = "cuda",
                 extra_path: str | Path | None = None, backend_paths: dict | None = None,
                 max_len: int = MAX_LEN, encoder=None):
        import torch

        self.backend, self.device, self.max_len = backend, device, int(max_len)
        self.backend_paths = backend_paths or {}
        self.extra_path = Path(extra_path) if extra_path else None
        self._encoder = encoder
        self.d = text_dim(backend)
        self.index: dict[str, int] = {}
        self.split: list[str] = []
        embs, masks = [], []
        paths = [Path(p) for p in cache_paths]
        if self.extra_path is not None and self.extra_path.exists():
            paths.append(self.extra_path)
        n_base = None
        for p in paths:
            if self.extra_path is not None and p == self.extra_path:
                n_base = len(self.split)    # entries from here on are extras: save_extra keeps them (review O7)
            c = torch.load(p, map_location="cpu", weights_only=False)
            cb = c.get("backend", "holobrain_bert")
            if cb != backend:
                raise ValueError(f"{p}: cache backend {cb!r} != store backend {backend!r}")
            e, m = c["emb"], c["mask"].bool()
            if e.shape[1] != self.max_len or e.shape[2] != self.d:
                raise ValueError(f"{p}: emb {tuple(e.shape)} does not match [N, {self.max_len}, {self.d}]")
            keys = sorted(c["index"], key=c["index"].get)
            keep = [k for k in keys if k not in self.index]
            if not keep:
                continue
            rows = torch.tensor([c["index"][k] for k in keep], dtype=torch.long)
            base = len(self.split)
            for j, k in enumerate(keep):
                self.index[k] = base + j
            split = c.get("split") or ["?"] * len(keys)
            self.split += [split[c["index"][k]] for k in keep]
            embs.append(e[rows].half())
            masks.append(m[rows])
        self.emb = torch.cat(embs) if embs else torch.zeros(0, self.max_len, self.d, dtype=torch.float16)
        self.mask = torch.cat(masks) if masks else torch.zeros(0, self.max_len, dtype=torch.bool)
        self.emb_dev, self.mask_dev = self.emb.to(device), self.mask.to(device)
        self.n_cached = len(self.split) if n_base is None else n_base   # base caches only; extras follow
        self._extra_since_save = 0
        self.stats = Counter()      # lookups / misses by split

    def __len__(self) -> int:
        return len(self.split)

    def __contains__(self, s: str) -> bool:
        return s in self.index

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = make_encoder(self.backend, self.device, self.backend_paths)
        return self._encoder

    def add(self, sentences, split: str = "extra") -> int:
        """Encode new sentences (one batch), append them; returns how many were new."""
        import torch

        new = sorted({s for s in sentences if s not in self.index})
        if not new:
            return 0
        e, v = self.encoder.embed(new)
        e, v = e.float(), v.bool()
        n = min(self.max_len, e.shape[1])
        emb = torch.zeros(len(new), self.max_len, self.d, dtype=torch.float16, device=self.device)
        msk = torch.zeros(len(new), self.max_len, dtype=torch.bool, device=self.device)
        emb[:, :n] = e[:, :n].to(self.device).half()
        msk[:, :n] = v[:, :n].to(self.device)
        base = len(self.split)
        for j, s in enumerate(new):
            self.index[s] = base + j
        self.split += [split] * len(new)
        self.emb_dev = torch.cat([self.emb_dev, emb])
        self.mask_dev = torch.cat([self.mask_dev, msk])
        self._extra_since_save += len(new)
        if self.extra_path is not None and self._extra_since_save >= SAVE_EVERY:
            self.save_extra()
        return len(new)

    def lookup(self, sentences, splits=None):
        """-> (words [B, L, d] float32, mask [B, L] bool) on the device."""
        import torch

        sentences = list(sentences)
        miss = [s for s in sentences if s not in self.index]
        for i, s in enumerate(sentences):
            sp = splits[i] if splits is not None else "?"
            self.stats[f"n/{sp}"] += 1
            if s not in self.index:
                self.stats[f"miss/{sp}"] += 1
        if miss:
            self.add(miss)
        idx = torch.tensor([self.index[s] for s in sentences], dtype=torch.long, device=self.emb_dev.device)
        return self.emb_dev[idx].float(), self.mask_dev[idx]

    def miss_frac(self, reset: bool = True) -> dict:
        out = {}
        for k, v in list(self.stats.items()):
            if k.startswith("n/"):
                sp = k[2:]
                out[sp] = self.stats.get(f"miss/{sp}", 0) / max(v, 1)
        tot = sum(v for k, v in self.stats.items() if k.startswith("n/"))
        out["all"] = sum(v for k, v in self.stats.items() if k.startswith("miss/")) / max(tot, 1)
        if reset:
            self.stats.clear()
        return out

    def save_extra(self) -> None:
        """Persist every extra sentence (the ones loaded from extra_path and the ones added since), merged
        with entries another process wrote to the same file meanwhile; atomic (review O7: the file used to be
        overwritten with only the sentences added since the last load)."""
        import torch

        if self.extra_path is None or self._extra_since_save == 0:
            return
        self.extra_path.parent.mkdir(parents=True, exist_ok=True)
        lock = Path(f"{self.extra_path}.lock")   # merge + replace under a short lock (round 2 B2)
        held = False
        for _ in range(60):
            try:
                lock.mkdir()
                held = True
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 120:   # a saver that died mid-save
                        lock.rmdir()
                        continue
                except OSError:
                    pass
                time.sleep(0.5)
        try:
            self._save_extra_locked()
        finally:
            if held:
                try:
                    lock.rmdir()
                except OSError:
                    pass

    def _save_extra_locked(self) -> None:
        import torch

        keys = list(range(self.n_cached, len(self.split)))
        inv = {i: s for s, i in self.index.items() if i >= self.n_cached}
        index = {inv[i]: j for j, i in enumerate(keys)}
        split = [self.split[i] for i in keys]
        emb, mask = [self.emb_dev[self.n_cached:].cpu()], [self.mask_dev[self.n_cached:].cpu()]
        if self.extra_path.exists():   # entries only the file has (another process): carry them over
            try:
                c = torch.load(self.extra_path, map_location="cpu", weights_only=False)
                theirs = [k for k in sorted(c["index"], key=c["index"].get) if k not in self.index]
                if theirs and c.get("backend", self.backend) == self.backend:
                    rows = torch.tensor([c["index"][k] for k in theirs], dtype=torch.long)
                    for k in theirs:
                        index[k] = len(split)
                        split.append((c.get("split") or ["?"] * len(c["index"]))[c["index"][k]])
                    emb.append(c["emb"][rows].half())
                    mask.append(c["mask"][rows].bool())
            except Exception as exc:
                print(f"[text] {self.extra_path} unreadable ({exc}); writing this process's extras only")
        out = {"index": index, "split": split, "emb": torch.cat(emb), "mask": torch.cat(mask),
               "max_len": self.max_len, "backend": self.backend}
        tmp = f"{self.extra_path}.{os.getpid()}.tmp"
        torch.save(out, tmp)
        os.replace(tmp, self.extra_path)
        self._extra_since_save = 0


# ---------------------------------------------------------------------------
# cache builder for any backend (same sentence enumeration as build_text_cache.py)
# ---------------------------------------------------------------------------


def _collect_sentences():
    spec = importlib.util.spec_from_file_location(
        "_build_text_cache", REPO / "envs/scripts/item_envs/build_text_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_sentences


def load_records(episode_dirs) -> list[dict]:
    recs = []
    for d in episode_dirs:
        for d2 in sorted(glob.glob(str(d))):
            recs += [json.loads(f.read_text()) for f in sorted(Path(d2).glob("*.json"))]
    return recs


def build_cache(records, backend: str, out: str | Path, device: str = "cuda", backend_paths=None,
                batch: int = 64, extra_sentences: dict | None = None) -> dict:
    import torch

    sents = _collect_sentences()(records)
    for s, sp in (extra_sentences or {}).items():
        sents.setdefault(s, sp)
    enc = make_encoder(backend, device, backend_paths)
    keys = sorted(sents)
    d = text_dim(backend)
    emb = torch.zeros(len(keys), MAX_LEN, d, dtype=torch.float16)
    mask = torch.zeros(len(keys), MAX_LEN, dtype=torch.bool)
    for a in range(0, len(keys), batch):
        e, v = enc.embed(keys[a:a + batch])
        n = min(MAX_LEN, e.shape[1])
        emb[a:a + len(e), :n] = e[:, :n].float().cpu().half()
        mask[a:a + len(e), :n] = v[:, :n].cpu()
    res = {"index": {s: i for i, s in enumerate(keys)}, "split": [sents[s] for s in keys], "emb": emb,
           "mask": mask, "max_len": MAX_LEN, "backend": backend}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    torch.save(res, tmp)
    os.replace(tmp, out)
    return {"n": len(keys), "splits": dict(Counter(res["split"]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--backend", required=True, choices=("holobrain_bert", "bert_raw", "sentence_generic"))
    p.add_argument("--episodes", nargs="+", required=True, help="episodes/ directories (globs ok)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--with-heldout-item", action="store_true",
                   help="also embed training templates of heldout_item rules (offline probe axis)")
    args = ap.parse_args(argv)
    from .config import LangConfig

    recs = load_records(args.episodes)
    extra = heldout_item_sentences(recs) if args.with_heldout_item else None
    info = build_cache(recs, args.backend, args.out, args.device, LangConfig().backend_paths, extra_sentences=extra)
    print(f"[text] {len(recs)} records -> {args.out}: {info}")
    return 0


def heldout_item_sentences(records) -> dict:
    """Training templates on rules whose env split is heldout_item (Section 9.1)."""
    from main.item_envs import language

    out = {}
    for rec in records:
        for rule in rec["rules"]:
            if rule.get("set") or rule.get("ambiguous") or rule.get("split") != "heldout_item":
                continue
            for s in language.sentences(rule, rec, "train"):
                out.setdefault(s, "heldout_item")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
