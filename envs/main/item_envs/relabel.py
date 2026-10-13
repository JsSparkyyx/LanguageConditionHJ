"""Relabeling sampler for the text-conditioned filter (docs/item_envs/relabeling.md).

Reads an ItemSidecarBuffer (the per-tick ``h_rules`` / ``fails_soon`` /
``t_in_ep`` / ``episode_id`` written in lockstep with the StepBuffer — slot i
of either ring is the same transition) and its ``episodes/{id}.json``
records, and draws minibatches of (transition, rule, sentence):

1. a transition: row t of an episode, with its next row t + 1 (or terminal);
2. a rule among the episode's sampleable rules, harmful
   (``fails_soon[t, rule]``) and benign about ``harmful_frac`` / rest when
   both exist;
3. a sentence: a template of the split + unique referent phrases
   (language.sample_sentence), optionally looked up in a text cache (a miss
   raises — the split stays controlled);
4. targets h_rules[t, rule] and h_rules[t + 1, rule] (times ``h_gain`` of the
   consequence, clipped to [-1, 1]; the env h is already on one scale, so
   the default gain is 1);
5. paired samples: ``pair_frac`` of the batch is the same transition twice,
   with a harmful and a benign rule.

Also returned: rows t - k (clamped to the episode start) and the episode's
first row, for the previous-frame and first-frame lookups.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import language
from .item_buffer import ItemSidecarBuffer

__all__ = ["RelabelSampler"]


class RelabelSampler:
    def __init__(self, root: str | Path, *, split: str = "train", harmful_frac: float = 0.5,  # or heldout_template / heldout_word / heldout_item
                 pair_frac: float = 0.5, h_gain: dict | None = None, k_prev: int = 1,
                 text_cache: dict | None = None, bank: dict | None = None):
        self.root = Path(root)
        self.buf = ItemSidecarBuffer(self.root)
        self.split = split
        self.harmful_frac = float(harmful_frac)
        self.pair_frac = float(pair_frac)
        self.h_gain = dict(h_gain or {})
        self.k_prev = int(k_prev)
        self.text_cache = text_cache
        self.bank = bank or language.load_bank()
        self.records: dict[int, dict] = {}
        self._mids_stamp = None
        self.refresh()

    # -------------------------------------------------------------- index ----
    def refresh(self) -> int:
        """Re-read the ring's n / head and new episode records; rebuild the
        index of usable rows. Returns the number of usable rows."""
        self.buf.refresh()
        ep_dir = self.root / "episodes"
        # records keep the words they were collected with: re-name them with the current vocabulary and the
        # recovered T1 model ids (language.refresh_record; names checked 2026-10-12), again when the id file changes
        mpath = self.root / "model_ids.json"
        stamp = mpath.stat().st_mtime if mpath.exists() else None
        mids = language.load_model_ids(self.root)
        if stamp != self._mids_stamp:
            for eid, rec in self.records.items():
                language.refresh_record(rec, mids.get(eid), bank=self.bank)
            self._mids_stamp = stamp
        for f in sorted(ep_dir.glob("*.json")) if ep_dir.exists() else []:
            eid = int(f.stem)
            if eid not in self.records:
                self.records[eid] = language.refresh_record(json.loads(f.read_text()), mids.get(eid), bank=self.bank)
        self._sample_cols = {
            eid: np.array([r["idx"] for r in rec["rules"] if self._usable(r, rec)], dtype=np.int64)
            for eid, rec in self.records.items()
        }
        C, n, head = self.buf.capacity, self.buf.n, self.buf.head
        slots = np.arange(C) if n == C else np.arange(n)
        ep = np.asarray(self.buf.arrays["episode_id"])
        t = np.asarray(self.buf.arrays["t_in_ep"]).astype(np.int64)
        nxt = (slots + 1) % C
        has_next = nxt != head  # the newest row's successor is not written yet
        if n < C:
            has_next &= nxt < n
        cont = has_next & (ep[nxt] == ep[slots]) & (t[nxt] == t[slots] + 1)
        terminal = has_next & ~cont  # the episode ended at this row
        first = (slots - t[slots]) % C
        first_ok = (ep[first] == ep[slots]) & (t[first] == 0) & (t[slots] < max(n, 1))
        known = np.array([int(e) in self._sample_cols and self._sample_cols[int(e)].size > 0 for e in ep[slots]],
                         dtype=bool)
        ok = (cont | terminal) & first_ok & known
        self._rows = slots[ok]
        self._next = np.where(cont[ok], nxt[ok], slots[ok])
        self._done = terminal[ok]
        self._first = first[ok]
        self._mixed = self._mixed_rows()
        return int(self._rows.size)

    @property
    def _text_split(self) -> str:
        """The sentence split: "heldout_item" (unseen colors / categories,
        splits.yaml) uses the TRAIN templates and words."""
        return "train" if self.split in ("train", "heldout_item") else self.split

    def _usable(self, rule: dict, rec: dict) -> bool:
        if self.split == "train":
            return bool(rule.get("sampleable"))
        if rule.get("set") or rule.get("ambiguous"):
            return False
        if self.split == "heldout_item" and rule.get("split") != "heldout_item":
            return False
        return bool(language.sentences(rule, rec, self._text_split, self.bank))

    def __len__(self) -> int:
        return int(self._rows.size)

    # ------------------------------------------------------------- sample ----
    def _h(self, cons: str, value: float) -> float:
        return float(np.clip(self.h_gain.get(cons, 1.0) * float(value), -1.0, 1.0))

    def _pick_rule(self, slot: int, cols: np.ndarray, rng, want: bool | None = None) -> int:
        fs = np.asarray(self.buf.arrays["fails_soon"][slot])[cols]
        harmful, benign = cols[fs], cols[~fs]
        if want is None:
            if harmful.size and benign.size:
                want = bool(rng.random() < self.harmful_frac)
            else:
                want = bool(harmful.size)
        pool = harmful if want else benign
        return int(pool[int(rng.integers(pool.size))])

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        """A minibatch (module docstring). Keys: row, next_row, prev_row,
        first_row, done, rule, consequence, sentence, h, h_next, fails_soon,
        pair (pair id, -1 = unpaired), episode_id, text_index (with a cache)."""
        if not len(self):
            raise ValueError(f"{self.root}: no usable rows")
        n_pairs = int(round(self.pair_frac * batch_size / 2.0))
        picks: list[tuple[int, int, int]] = []  # (index into self._rows, rule col, pair id)
        if n_pairs:
            mixed = self._mixed
            if mixed.size:
                for p, i in enumerate(rng.choice(mixed, size=n_pairs)):
                    slot = int(self._rows[i]); cols = self._cols(slot)
                    picks.append((int(i), self._pick_rule(slot, cols, rng, True), p))
                    picks.append((int(i), self._pick_rule(slot, cols, rng, False), p))
        while len(picks) < batch_size:
            i = int(rng.integers(len(self)))
            slot = int(self._rows[i])
            picks.append((i, self._pick_rule(slot, self._cols(slot), rng), -1))
        picks = picks[:batch_size]
        A = self.buf.arrays
        out = {key: [] for key in ("row", "next_row", "prev_row", "first_row", "done", "rule", "consequence",
                                   "sentence", "h", "h_next", "fails_soon", "pair", "episode_id")}
        C = self.buf.capacity
        for i, col, pair in picks:
            slot, nxt = int(self._rows[i]), int(self._next[i])
            eid = int(A["episode_id"][slot]); rec = self.records[eid]; rule = rec["rules"][col]
            t = int(A["t_in_ep"][slot])
            out["row"].append(slot); out["next_row"].append(nxt)
            out["prev_row"].append((slot - min(self.k_prev, t)) % C); out["first_row"].append(int(self._first[i]))
            out["done"].append(bool(self._done[i])); out["rule"].append(col)
            out["consequence"].append(rule["consequence"])
            out["sentence"].append(language.sample_sentence(rule, rec, rng, self._text_split, self.bank))
            out["h"].append(self._h(rule["consequence"], A["h_rules"][slot, col]))
            out["h_next"].append(self._h(rule["consequence"], A["h_rules"][nxt, col]))
            out["fails_soon"].append(bool(A["fails_soon"][slot, col])); out["pair"].append(pair)
            out["episode_id"].append(eid)
        batch = {k: (v if k in ("consequence", "sentence") else np.asarray(v)) for k, v in out.items()}
        if self.text_cache is not None:
            index = self.text_cache["index"]
            missing = [s for s in batch["sentence"] if s not in index]
            if missing:
                raise KeyError(f"text cache miss ({len(missing)}): {missing[:3]}")
            batch["text_index"] = np.asarray([index[s] for s in batch["sentence"]])
        return batch

    def _cols(self, slot: int) -> np.ndarray:
        return self._sample_cols[int(self.buf.arrays["episode_id"][slot])]

    def _mixed_rows(self) -> np.ndarray:
        """Indices (into the usable rows) with both a harmful and a benign rule."""
        fs = np.asarray(self.buf.arrays["fails_soon"])
        ep = np.asarray(self.buf.arrays["episode_id"])
        out = []
        for i, slot in enumerate(self._rows):
            cols = self._sample_cols[int(ep[slot])]
            f = fs[slot, cols]
            if f.any() and not f.all():
                out.append(i)
        return np.asarray(out, dtype=np.int64)

    def benign_rules(self, episode_id: int) -> list[int]:
        """Columns of an episode's usable rules whose fails_soon is never set
        (the benign-scene intervention-rate evaluation: a filter run with
        such a rule should not intervene)."""
        ep = np.asarray(self.buf.arrays["episode_id"])
        n = self.buf.n if self.buf.n < self.buf.capacity else self.buf.capacity
        rows = np.flatnonzero(ep[:n] == int(episode_id))
        cols = self._sample_cols.get(int(episode_id), np.zeros(0, dtype=np.int64))
        if not rows.size:
            return []
        fs = np.asarray(self.buf.arrays["fails_soon"])[rows][:, cols]
        return [int(c) for c, f in zip(cols, fs.any(axis=0)) if not f]
