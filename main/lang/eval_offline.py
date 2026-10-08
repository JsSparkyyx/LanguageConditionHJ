"""Offline language probe (VANILLA_PLAN 9.1), validation data only.

    python -m main.lang.eval_offline --run runs/lang/<name> --ckpt checkpoint.pt --val data/lang/val \\
        --states-per-task 2000 --paraphrases 3

States: every 5th tick of every validation episode (terminal rows excluded),
at most --states-per-task per task. For each state and rule, Q(s, a_logged |
sentence) per axis:

| axis | sentences | eligible rules |
|---|---|---|
| train | sentences(rule, rec, "train") | sampleable rules |
| heldout_template | sentences(rule, rec, "heldout_template") | env split train, not set / ambiguous |
| heldout_word | sentences(rule, rec, "heldout_word") | env split train, not set / ambiguous, with a held-out noun |
| heldout_item | training templates | env split heldout_item |
| shuffled | a training sentence of a sampleable rule of another episode | sampleable rules (control) |

Pairs: two rules at one state whose fails_soon bits differ. Symmetric pairs:
both members axis-eligible, both get the axis sentence (shuffled: each its
own other-episode sentence). Mixed pairs (heldout_word / heldout_item, only r1
eligible): r1 gets the axis sentence, r2 its train sentence; rank_acc_base
replaces r1's sentence by another episode's, and rank_acc - rank_acc_base is
reported. Metrics per axis, task and consequence and pooled: rank_acc (Q(harmful)
< Q(benign)), rank_acc_referent (same consequence, different item),
rank_acc_consequence (same item, different consequence), auroc of -Q for
fails_soon (+ auroc_base with other-episode sentences), prec / rec at Q < 0
against fails_soon_recall, v_le_h = P[Q <= h + 0.05], para_std(_ratio).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import ensure_item_path

ensure_item_path()

from main.item_envs import language  # noqa: E402
from main.item_envs.item_buffer import FAILS_SOON_H  # noqa: E402

from .config import CANON_TASKS, LangConfig  # noqa: E402
from .data import BEAD_TASKS, LangSource, stable_hash  # noqa: E402
from .scene_roles import referent  # noqa: E402

AXES = ("train", "heldout_template", "heldout_word", "heldout_item", "shuffled")
V_LE_H_EPS = 0.05


def eligible(rule: dict, rec: dict, axis: str, bank) -> bool:
    if axis in ("train", "shuffled"):
        return bool(rule.get("sampleable"))
    if rule.get("set") or rule.get("ambiguous"):
        return False
    if axis == "heldout_item":
        return rule.get("split") == "heldout_item" and bool(language.sentences(rule, rec, "train", bank))
    if rule.get("split") != "train":
        return False
    return bool(language.sentences(rule, rec, axis, bank))


def axis_sentence(rule, rec, axis, key, bank) -> str | None:
    split = "train" if axis in ("train", "heldout_item") else axis
    ss = language.sentences(rule, rec, split, bank)
    return ss[stable_hash((key, axis)) % len(ss)] if ss else None


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged (vectorized)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, start, counts = np.unique(xs, return_index=True, return_counts=True)
    r = np.empty(len(x))
    r[order] = np.repeat(start + (counts - 1) / 2.0 + 1, counts)
    return r


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of score for label (rank-based, ties averaged)."""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _avg_ranks(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b) -> float:
    a, b = _avg_ranks(a), _avg_ranks(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def hj_return(h: np.ndarray, gamma: float) -> np.ndarray:
    """[L, R] discounted HJ return of every rule's recorded h along one episode, the value a gamma critic
    learns (review E1): y_{L-1} = h_{L-1}, y_t = (1 - gamma) h_t + gamma min(h_t, y_{t+1})."""
    y = np.array(h, dtype=np.float64, copy=True)
    for t in range(len(y) - 2, -1, -1):
        y[t] = (1 - gamma) * h[t] + gamma * np.minimum(h[t], y[t + 1])
    return y


def next_violation(h: np.ndarray) -> np.ndarray:
    """[L, R] ticks from t to the first h < 0 at or after t (0 = already violated), -1 = none in the episode."""
    nv = np.full(h.shape, -1, dtype=np.int64)
    if not len(h):
        return nv
    nv[-1] = np.where(h[-1] < 0, 0, -1)
    for t in range(len(h) - 2, -1, -1):
        nv[t] = np.where(h[t] < 0, 0, np.where(nv[t + 1] >= 0, nv[t + 1] + 1, -1))
    return nv


def rule_words(rule: dict, rec: dict) -> tuple:
    """What a sentence can say about a rule without the scene: the consequence and the referent's words
    (object category / colour / size, or the region kind / colour a bead rule refers to). The key of the
    scene-free text prior (review E2); it is at least as informative as any one sentence."""
    subj, ref = rule.get("subject") or {}, rule.get("reference") or {}
    words = []
    if "object" in subj:
        o = rec["objects"][subj["object"]]
        words += [o.get("category"), o.get("color"), o.get("size")]
    elif "particles" in subj:
        words.append("beads")
    regs = [ref["region"]] if "region" in ref else list(ref.get("regions") or [])
    for k in regs:
        g = rec["regions"][k]
        words += [g.get("kind"), g.get("color")]
    return (rule["consequence"],) + tuple(str(w) for w in words if w)


_WORD = __import__("re").compile(r"[a-z]+")


def sentence_words(sentence: str) -> list[str]:
    return _WORD.findall((sentence or "").lower())


class TextPrior:
    """P(fails_soon | the sentence's words): an L2 logistic regression on word presence, fit on training
    data with training-template sentences of every sampleable rule (weights: the rule's fails_soon counts
    over its episode). A reader of the ACTUAL sentence (the words the model gets) that never looks at the
    scene (review E2; round 2 C4: it used to key on the record's category / colour / size, which a sentence
    may leave out). ``roots`` are recorded so main() can check they are disjoint from the validation data."""

    def __init__(self, roots, k_prev: int = 2, n_sent: int = 4, seed: int = 0, l2: float = 1e-2, iters: int = 25):
        self.roots = [str(Path(r).resolve()) for r in roots]
        bank = language.load_bank()
        rng = np.random.default_rng(seed)
        docs, ks, ns = [], [], []
        for r in roots:
            src = LangSource(r, k_prev, 7)
            if not src.rows.size:
                continue
            fs_all = src.rs.buf.arrays["fails_soon"]
            for e in np.unique(src.eid):
                idx = src.rows[src.eid == e]
                rec = src.rs.records[int(e)]
                for rule in rec["rules"]:
                    if not rule.get("sampleable"):
                        continue
                    fs = np.asarray(fs_all[idx, int(rule["idx"])], dtype=bool)
                    for _ in range(n_sent):
                        docs.append(set(sentence_words(language.sample_sentence(rule, rec, rng, "train", bank))))
                        ks.append(fs.sum() / n_sent)
                        ns.append(fs.size / n_sent)
        df = {}
        for d in docs:
            for w in d:
                df[w] = df.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
        V = len(self.vocab)
        X = np.zeros((len(docs), V + 1))
        X[:, V] = 1.0
        for i, d in enumerate(docs):
            for w in d:
                if w in self.vocab:
                    X[i, self.vocab[w]] = 1.0
        k, n = np.asarray(ks, dtype=np.float64), np.asarray(ns, dtype=np.float64)
        w = np.zeros(V + 1)
        N = max(float(n.sum()), 1.0)
        reg = 2 * l2 * np.eye(V + 1)
        reg[V, V] = 0.0
        for _ in range(iters):   # Newton steps on the weighted binomial log-likelihood
            p_ = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
            g = X.T @ (n * p_ - k) / N + reg @ w
            H = (X * (n * p_ * (1 - p_))[:, None]).T @ X / N + reg + 1e-9 * np.eye(V + 1)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        self.n_docs = len(docs)

    def __call__(self, sentence: str) -> float:
        x = np.zeros(len(self.vocab) + 1)
        x[-1] = 1.0
        for word in set(sentence_words(sentence)):
            if word in self.vocab:
                x[self.vocab[word]] = 1.0
        return float(1 / (1 + np.exp(-np.clip(x @ self.w, -30, 30))))


class Evaluator:
    def __init__(self, run: Path, ckpt: str, val_roots, device="cuda", text_caches=(), batch: int = 96,
                 text_encoder=None, embodiments="trained", tasks=None):
        from .network import LangPolicyEncoder, LangTwinCritic
        from .text import TextStore
        from .train import load_policy

        self.run = Path(run)
        ck = torch.load(self.run / ckpt if not Path(ckpt).is_absolute() else ckpt, map_location=device,
                        weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        # provenance (review round 3, B1): what the weights were trained on, and which file they are
        self.bead_h_train = ck.get("bead_h_train") or "not recorded"
        self.ckpt_sha = file_sha(self.run / ckpt if not Path(ckpt).is_absolute() else Path(ckpt))
        # validation episodes scored: of the embodiments the run trained on ("trained"), all ("all"), or a list
        self.embodiments = (tuple(self.cfg.embodiment_choices) if embodiments == "trained"
                            else None if embodiments in (None, "all") else tuple(embodiments))
        self.dev = device
        self.tasks = None if tasks in (None, "all") else tuple(tasks)
        self.sources = [LangSource(r, self.cfg.k_prev, self.cfg.recall_horizon_ticks,
                                   drop_stale_bead=getattr(self.cfg, "drop_stale_bead_rows", True)) for r in val_roots]
        n_tok = self.sources[0].step["scene_tokens"].shape[1]
        self.pe = LangPolicyEncoder(self.cfg, n_tok=n_tok).to(device).eval()
        load_policy(self.pe, ck)
        self.critics = LangTwinCritic(self.cfg).to(device).eval()
        self.critics.load_state_dict(ck["critics"])
        # the trainer's extras are read only; the evaluator writes its own file (review O7)
        train_extra = [self.run / "text_cache_extra.pt"] if (self.run / "text_cache_extra.pt").exists() else []
        self.text = TextStore(list(text_caches) + train_extra, self.cfg.text_backend, device,
                              extra_path=self.run / "text_cache_extra_eval.pt", backend_paths=self.cfg.backend_paths,
                              encoder=text_encoder)
        self.bank = language.load_bank()
        self.batch = int(batch)
        self.ckpt_step = int(ck.get("step", 0))
        self.gamma = float(self.cfg.gamma)
        self._ep = {}
        self.prior = None   # TextPrior, set by main when --prior-data is given

    # ------------------------------------------------------------- states --
    def states(self, per_task: int, every: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        by_task = defaultdict(list)
        for j, s in enumerate(self.sources):
            sel = s.t % every == 0
            if self.tasks is not None:
                sel &= np.isin(s.task, [CANON_TASKS.index(t) for t in self.tasks])
            if self.embodiments is not None:
                sel &= np.isin(s.emb, list(self.embodiments))
            for i in np.flatnonzero(sel):
                by_task[int(s.task[i])].append((j, int(i)))
        out = []
        for t, lst in sorted(by_task.items()):
            if len(lst) > per_task:
                lst = [lst[k] for k in sorted(rng.choice(len(lst), size=per_task, replace=False))]
            out += lst
        return out

    def _episode(self, j: int, eid: int):
        """(t_in_ep -> episode row, HJ return y [L, R], ticks to the next violation [L, R]) of one episode."""
        key = (j, eid)
        if key not in self._ep:
            A, n = self.sources[j].rs.buf.arrays, self.sources[j].rs.buf.n
            rows = np.flatnonzero(np.asarray(A["episode_id"][:n]) == eid)
            t_in = np.asarray(A["t_in_ep"])[rows]
            rows, t_in = rows[np.argsort(t_in, kind="stable")], np.sort(t_in, kind="stable")
            h = np.asarray(A["h_rules"][rows], dtype=np.float64)
            self._ep[key] = ({int(t): k for k, t in enumerate(t_in)}, hj_return(h, self.gamma), next_violation(h),
                             len(rows))
        return self._ep[key]

    def _base_sentence(self, rule, rec, j, eid, rng) -> tuple[str, int | None]:
        """The base for auroc_base / rank_acc_base: a sentence naming ANOTHER sampleable rule of the same
        state (same task and scene; review E3); another episode's only when the state has no other rule.
        -> (sentence, the rule's column in this state, or None for another episode's sentence)."""
        others = [r for r in rec["rules"] if r.get("sampleable") and r["rule_id"] != rule["rule_id"]]
        if others:
            o = others[int(rng.integers(len(others)))]
            return language.sample_sentence(o, rec, rng, "train", self.bank), int(o["idx"])
        return self._other_sentence(j, eid, rng), None

    def _other_sentence(self, j, eid, rng) -> str:
        eps = [(jj, int(e)) for jj, s in enumerate(self.sources) for e in np.unique(s.eid)]
        for _ in range(200):
            j2, e2 = eps[int(rng.integers(len(eps)))]
            if (j2, e2) == (j, eid):
                continue
            s2 = self.sources[j2]
            cols = s2.rs._sample_cols.get(e2)
            if cols is None or not cols.size:
                continue
            rec = s2.rs.records[e2]
            return language.sample_sentence(rec["rules"][int(cols[rng.integers(cols.size)])], rec, rng, "train", self.bank)
        raise RuntimeError("no other-episode sentence")

    # ------------------------------------------------------------ forward --
    def ref_tag(self, keys, cols) -> tuple[np.ndarray, np.ndarray]:
        """V3 checkpoints (oracle_ref_tag): the centre of each item's named object in frames [t, t-k, 0];
        the tag follows each pair member's own named object (review round 3, E2)."""
        from .data import ref_centers

        c, v = np.zeros((len(keys), 3, 3), np.float32), np.zeros((len(keys), 3), bool)
        for n, ((j, i), col) in enumerate(zip(keys, cols)):
            if col is None:
                continue
            s = self.sources[j]
            rec, rows = s.rs.records[int(s.eid[i])], s.rows_of(i)
            c[n], v[n] = ref_centers(s.rs.buf.arrays, rec, rec["rules"][int(col)],
                                     (rows["row"], rows["prev_row"], rows["first_row"]))
        return c, v

    @torch.no_grad()
    def q_batch(self, reads: list[dict], sentences: list[str], keys=None, cols=None) -> np.ndarray:
        from .train import _t

        d = self.dev
        b = {k: np.stack([r[k] for r in reads]) for k in reads[0]}
        words, wmask = self.text.lookup(sentences)
        kw = {}
        if getattr(self.cfg, "oracle_ref_tag", False):
            if keys is None or cols is None:
                raise ValueError("a V3 (oracle_ref_tag) checkpoint needs each item's state key and rule column")
            c, v = self.ref_tag(keys, cols)
            kw = {"ref_center": _t(c, dev=d), "ref_valid": _t(v, torch.bool, d)}
        enc = self.pe(_t(b["state_tokens"], dev=d), _t(b["body_tokens"], dev=d), _t(b["link_pos"], dev=d),
                      _t(b["body_mask"], torch.bool, d), _t(b["joint_index"], torch.int64, d),
                      _t(b["scene_tokens"], dev=d), _t(b["scene_pos"], dev=d), words=words, word_mask=wmask, **kw)
        q = self.critics.qmin(enc, _t(b["dtheta"], dev=d), _t(b["Jlin"], dev=d), _t(b["Jang"], dev=d),
                              _t(b["joint_index"], torch.int64, d))
        return q.float().cpu().numpy()

    def run_items(self, items, cache) -> np.ndarray:
        """items: (state key, sentence[, rule column of the sentence's rule in that state]) -> Q, batched;
        state reads cached per chunk. The column feeds the V3 oracle tag (None: no tag)."""
        out = np.zeros(len(items), np.float32)
        for a in range(0, len(items), self.batch):
            chunk = items[a:a + self.batch]
            reads = [cache[it[0]] for it in chunk]
            out[a:a + len(chunk)] = self.q_batch(reads, [it[1] for it in chunk], [it[0] for it in chunk],
                                                 [it[2] if len(it) > 2 else None for it in chunk])
        return out

    # ---------------------------------------------------------------- run --
    def evaluate(self, per_task: int = 2000, n_para: int = 3, para_states: int = 200, chunk_states: int = 192,
                 seed: int = 0) -> tuple[dict, list[dict]]:
        rng = np.random.default_rng(seed)
        states = self.states(per_task, seed=seed)
        rows = []          # one per (state, rule)
        para_rows = []
        t0 = time.time()
        for c0 in range(0, len(states), chunk_states):
            chunk = states[c0:c0 + chunk_states]
            cache = {k: self.sources[k[0]].read(k[1]) for k in chunk}
            items, meta = [], []
            for k in chunk:
                j, i = k
                s = self.sources[j]
                eid, slot = int(s.eid[i]), int(s.rows[i])
                rec = s.rs.records[eid]
                A = s.rs.buf.arrays
                tmap, y_ep, nv_ep, L_ep = self._episode(j, eid)
                k_ep = tmap.get(int(s.t[i]))
                # an episode that ran to the tick cap was cut: its last FAILS_SOON_H - 1 rows' fails_soon and
                # return look past the end (censored; round 2 C8)
                censored = bool(L_ep >= int(self.cfg.max_steps_per_episode) - 1
                                and L_ep - 1 - int(s.t[i]) < FAILS_SOON_H - 1)
                for rule in rec["rules"]:
                    el = {ax: eligible(rule, rec, ax, self.bank) for ax in AXES}
                    if not any(el.values()):
                        continue
                    col = rule["idx"]
                    # E0b (round 3): the most endangered object of this consequence in the scene, named or not
                    same = [r["idx"] for r in rec["rules"] if r["consequence"] == rule["consequence"]
                            and "object" in (r.get("subject") or {}) and not r.get("set")]
                    base = {"src": j, "i": i, "task": rec["task"], "emb": rec["embodiment"], "episode_id": eid,
                            "t": int(s.t[i]), "rule": rule["rule_id"], "col": col, "consequence": rule["consequence"],
                            "referent": referent(rule, rec), "split": rule.get("split"),
                            "fails_soon": bool(A["fails_soon"][slot, col]), "recall": bool(s.recall[slot, col]),
                            "h": float(A["h_rules"][slot, col]), "eligible": el, "q": {},
                            "y": float(y_ep[k_ep, col]) if k_ep is not None else None,
                            "ttv": int(nv_ep[k_ep, col]) if k_ep is not None and nv_ep[k_ep, col] >= 0 else None,
                            "words": list(rule_words(rule, rec)), "sentences": {}, "prior": {}, "censored": censored,
                            "size": subject_size(rule, rec),
                            "h_any": float(np.min(A["h_rules"][slot, same])) if same else None,
                            "fs_any": bool(np.any(A["fails_soon"][slot, same])) if same else None}
                    r_idx = len(rows)
                    rows.append(base)
                    for ax in AXES:
                        if not el[ax]:
                            continue
                        sent = (self._other_sentence(j, eid, rng) if ax == "shuffled"
                                else axis_sentence(rule, rec, ax, (eid, int(s.t[i]), col), self.bank))
                        base["sentences"][ax] = sent
                        if self.prior is not None:
                            base["prior"][ax] = float(self.prior(sent))
                        items.append((k, sent, None if ax == "shuffled" else col))
                        meta.append((r_idx, ax))
                    # base sentence for auroc_base / rank_acc_base: another rule of the same state (E3)
                    base["sentences"]["base"], base_col = self._base_sentence(rule, rec, j, eid, rng)
                    items.append((k, base["sentences"]["base"], base_col))
                    meta.append((r_idx, "base"))
                    if rule.get("sampleable") and len(para_rows) < para_states * len(CANON_TASKS) and \
                            stable_hash((eid, int(s.t[i]), col, "para")) % 4 == 0:
                        ss = language.sentences(rule, rec, "train", self.bank)
                        if len(ss) >= 2:
                            pick = [ss[q] for q in rng.choice(len(ss), size=min(n_para, len(ss)), replace=False)]
                            for sent in pick:
                                items.append((k, sent, col))
                                meta.append((r_idx, "para"))
            qs = self.run_items(items, cache)
            for (r_idx, ax), qv in zip(meta, qs):
                if ax == "para":
                    rows[r_idx]["q"].setdefault("para", []).append(float(qv))
                else:
                    rows[r_idx]["q"][ax] = float(qv)
            print(f"[eval_offline] {min(c0 + chunk_states, len(states))}/{len(states)} states, {len(rows)} (state, rule) "
                  f"rows, {time.time() - t0:.0f}s", flush=True)
        return summarize(rows), rows


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def subject_size(rule: dict, rec: dict) -> str | None:
    """The named object's size word ("tall" / "short" / None)."""
    subj = rule.get("subject") or {}
    return (rec["objects"][subj["object"]].get("size") or None) if "object" in subj else None


def _pairs(rows: list[dict]):
    by_state = defaultdict(list)
    for r in rows:
        by_state[(r["src"], r["i"])].append(r)
    for rs in by_state.values():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                if rs[a]["fails_soon"] != rs[b]["fails_soon"]:
                    yield (rs[a], rs[b]) if rs[a]["fails_soon"] else (rs[b], rs[a])   # (harmful, benign)


BOOT = 1000          # episode-cluster bootstrap resamples for rank cells (review E4)
BOOT_AUROC = 200     # ... for AUROC cells
MC_EPS = 0.05        # rank_acc_mc counts pairs whose HJ returns differ by more than this
TTV_BINS = (("violated", 0, 0), ("1-7", 1, 7), ("8-24", 8, FAILS_SOON_H - 1))   # fails_soon looks 25 ticks
RC_TASKS = ("selective_avoid", "push_zone", "gap_place")   # the tasks with both referent and consequence pairs
SIZE_WORDS = ("tall", "short")


def _ttv_bin(k):
    for name, lo, hi in TTV_BINS:
        if k is not None and lo <= k <= hi:
            return name
    return None


def boot_mean_ci(obs, B: int = BOOT, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of (cluster, value) observations, resampling whole clusters (episodes: the
    pairs of one episode, every 5th tick, are correlated)."""
    if not obs:
        return float("nan"), float("nan")
    keys = {}
    sums, cnts = defaultdict(float), defaultdict(int)
    for c, v in obs:
        keys.setdefault(c, len(keys))
        sums[c] += v
        cnts[c] += 1
    if len(keys) < 2:
        return float("nan"), float("nan")
    S = np.array([sums[c] for c in keys])
    N = np.array([cnts[c] for c in keys], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(len(keys), size=(B, len(keys)))
    vals = S[idx].sum(1) / N[idx].sum(1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_auroc_ci(score, label, clusters, B: int = BOOT_AUROC, seed: int = 0) -> tuple[float, float]:
    score, label, clusters = np.asarray(score), np.asarray(label, bool), np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    if len(uniq) < 2:
        return float("nan"), float("nan")
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        sel = np.concatenate([members[k] for k in rng.integers(len(uniq), size=len(uniq))])
        a = auroc(score[sel], label[sel])
        if a == a:
            vals.append(a)
    if len(vals) < B // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _rank(a: float, b: float) -> float:
    """1 if the harmful member's score a is below the benign member's b, 0.5 on a tie, else 0."""
    return 0.5 if a == b else float(a < b)


def _ep(r) -> str:
    return f"{r['src']}:{r['episode_id']}"


def summarize(rows: list[dict], seed: int = 0) -> dict:
    """Pair metrics per axis and group, each with an episode-bootstrap 95% interval (`<metric>_ci`) and the
    number of episodes behind it:

    - rank_acc: P[Q(harmful) < Q(benign)] (labels: fails_soon); rank_acc_base: the harmful rule's sentence
      replaced by another rule's of the same state (mixed pairs); rank_acc_referent / _consequence;
    - label checks (review E1): rank_acc_h ranks by the rules' CURRENT h (no model: a perfect reader with
      no foresight); rank_acc_mc = agreement of Q's order with the order of the discounted HJ return y
      (what a gamma critic learns), over pairs with |y1 - y2| > MC_EPS; rank_acc_text ranks by a scene-free
      text prior P(fails_soon | rule words) fit on training data (review E2; with --prior-data);
    - groups: pooled, task, consequence, the harmful rule's time to its first violation (violated / 1-7 /
      8-25 ticks, review E1c), and T5 pairs split by whether either sentence has a size word (review E2).
    Per (state, rule): AUROC of -Q for fails_soon with intervals, and the per-state Spearman correlation of
    Q with y across the state's rules (states with >= 3 rules).
    """
    out = {}

    def groups(hr, bn, ax):
        g = [("pooled",), ("task", hr["task"]), ("cons", hr["consequence"])]
        if hr["task"] in RC_TASKS:
            g.append(("rc_pool",))   # T1 / T3 / T7: referent vs consequence over the same task set (C6)
        b = _ttv_bin(hr.get("ttv"))
        state = None
        if b:
            state = "violated" if b == "violated" else "not_yet"
            g.append(("ttv", b))
            g.append(("task_ttv", hr["task"], b))
            # headline per task (C1): the harmful rule already broken, or not yet
            g.append(("task_state", hr["task"], state))
            if hr["task"] in RC_TASKS:
                g.append(("rc_state", state))   # T1 + T3 + T7 per stratum (round 3, A2)
        if hr["task"] == "push_zone" and state:
            g.append(("t3_benign", state, bn["consequence"]))   # T3 by the benign rule's consequence (round 3, A1)
        if hr["task"] == "tray_transport":
            sents = [(x.get("sentences") or {}).get(ax) or "" for x in (hr, bn)]
            sw = "size_word" if any(w in sent.lower().split() for sent in sents for w in SIZE_WORDS) else "no_size_word"
            g.append(("t5", sw))
            if state:
                g.append(("t5_state", sw, state))   # the size-word split crossed with the strata (round 3, A1)
        return g

    for ax in AXES:
        acc = defaultdict(lambda: defaultdict(list))
        n_censored = 0
        for hr, bn in _pairs(rows):
            e1, e2 = hr["eligible"][ax], bn["eligible"][ax]
            if hr.get("censored") and (e1 or e2):   # the state's 25-tick window runs past a capped end (C8)
                n_censored += 1
                continue
            if not (e1 or e2):
                continue
            if e1 and e2:
                kind = "symmetric"
                qh, qb = hr["q"][ax], bn["q"][ax]
                qh0 = qb0 = None
            elif ax in ("heldout_word", "heldout_item") and (e1 or e2) and hr["eligible"]["train"] | bn["eligible"]["train"]:
                kind = "mixed"
                elig, other = (hr, bn) if e1 else (bn, hr)
                if "train" not in other["q"]:
                    continue
                q_el, q_ot = elig["q"][ax], other["q"]["train"]
                q_el0 = elig["q"]["base"]
                qh, qb = (q_el, q_ot) if e1 else (q_ot, q_el)
                qh0, qb0 = (q_el0, q_ot) if e1 else (q_ot, q_el0)
            else:
                continue
            same_cons = hr["consequence"] == bn["consequence"]
            same_ref = hr["referent"] == bn["referent"] and hr["referent"] is not None
            ep = _ep(hr)
            ok = _rank(qh, qb)                      # ties count 0.5 everywhere (C5)
            ph, pb = (hr.get("prior") or {}).get(ax), (bn.get("prior") or {}).get(ax)
            if kind == "mixed":                     # the other member's train-sentence prior
                ph = ph if ph is not None else (hr.get("prior") or {}).get("train")
                pb = pb if pb is not None else (bn.get("prior") or {}).get("train")
            for g in groups(hr, bn, ax):
                d = acc[(kind,) + g]
                d["rank"].append((ep, ok))
                if qh0 is not None:
                    d["rank_base"].append((ep, _rank(qh0, qb0)))
                if same_cons and not same_ref:
                    d["rank_ref"].append((ep, ok))
                if same_ref and not same_cons:
                    d["rank_cons"].append((ep, ok))
                d["rank_h"].append((ep, _rank(hr["h"], bn["h"])))
                if hr.get("y") is not None and bn.get("y") is not None:
                    # the label vs the return of the NOMINAL continuation: does y rank the pair like fails_soon? (C2)
                    d["rank_y"].append((ep, _rank(hr["y"], bn["y"])))
                    if abs(hr["y"] - bn["y"]) > MC_EPS:
                        d["rank_mc"].append((ep, float(ok == float(hr["y"] < bn["y"]))))
                if ph is not None and pb is not None:
                    t_ok = _rank(-ph, -pb)               # the prior ranks by a HIGHER P(fails_soon)
                    d["rank_text"].append((ep, t_ok))
                    d["model_minus_text"].append((ep, ok - t_ok))   # paired, same pairs (C4)
                    if same_cons and not same_ref:   # the prior on the referent / consequence pairs (round 3, A5)
                        d["rank_ref_text"].append((ep, t_ok))
                        d["ref_minus_text"].append((ep, ok - t_ok))
                    if same_ref and not same_cons:
                        d["rank_cons_text"].append((ep, t_ok))
                        d["cons_minus_text"].append((ep, ok - t_ok))
        for key, d in acc.items():
            name = "/".join([ax] + list(key))
            m = {"n_pairs": len(d["rank"]), "n_episodes": len({c for c, _ in d["rank"]}),
                 "rank_acc": float(np.mean([v for _, v in d["rank"]])), "rank_acc_ci": boot_mean_ci(d["rank"], seed=seed)}
            for src, dst, n_key in (("rank_base", "rank_acc_base", None), ("rank_ref", "rank_acc_referent", "n_referent"),
                                    ("rank_cons", "rank_acc_consequence", "n_consequence"), ("rank_h", "rank_acc_h", None),
                                    ("rank_mc", "rank_acc_mc", "n_mc"), ("rank_text", "rank_acc_text", None),
                                    ("rank_y", "rank_acc_y", None), ("model_minus_text", "model_minus_text", None),
                                    ("rank_ref_text", "rank_acc_referent_text", None),
                                    ("ref_minus_text", "referent_minus_text", None),
                                    ("rank_cons_text", "rank_acc_consequence_text", None),
                                    ("cons_minus_text", "consequence_minus_text", None)):
                if d[src]:
                    m[dst] = float(np.mean([v for _, v in d[src]]))
                    m[dst + "_ci"] = boot_mean_ci(d[src], seed=seed)
                    if n_key:
                        m[n_key] = len(d[src])
            if "rank_acc_base" in m:
                m["rank_acc_minus_base"] = m["rank_acc"] - m["rank_acc_base"]
            out[name] = m
        out[f"{ax}/_censored_pairs"] = {"n": n_censored}
        for kind in ("symmetric", "mixed"):   # task-macro mean (every task weighs the same; review E4)
            tm = [out[f"{ax}/{kind}/task/{t}"]["rank_acc"] for t in CANON_TASKS if f"{ax}/{kind}/task/{t}" in out]
            if tm and f"{ax}/{kind}/pooled" in out:
                out[f"{ax}/{kind}/pooled"]["rank_acc_task_macro"] = float(np.mean(tm))
        # per (state, rule)
        sel = [r for r in rows if r["eligible"][ax] and ax in r["q"]]
        for gname, f in (("pooled", lambda r: True), ("pooled_no_bead", lambda r: r["task"] not in BEAD_TASKS),
                         ("t5_tall", lambda r: r["task"] == "tray_transport" and r.get("size") == "tall")) + tuple(
                ((f"task/{t}"), (lambda r, t=t: r["task"] == t)) for t in CANON_TASKS) + tuple(
                ((f"cons/{c}"), (lambda r, c=c: r["consequence"] == c)) for c in
                ("avoid", "stay", "upright", "zone_exclude", "zone_include", "contain_particles", "exclude_particles")):
            rr = [r for r in sel if f(r)]
            if not rr:
                continue
            q = np.array([r["q"][ax] for r in rr])
            qb = np.array([r["q"]["base"] for r in rr])
            fs = np.array([r["fails_soon"] for r in rr])
            rec = np.array([r["recall"] for r in rr])
            h = np.array([r["h"] for r in rr])
            cl = np.array([_ep(r) for r in rr])
            pred = q < 0
            tp = float((pred & rec).sum())
            # AUROC over harm that is NOT yet happening: already-broken rows left out of the positives (C8)
            nyet = np.array([not (r["fails_soon"] and r.get("ttv") == 0) for r in rr])
            m = out.setdefault(f"{ax}/state/{gname}", {})
            # the same rows ranked by the current h and by the sentence prior (round 3, A4)
            m["auroc_h"], m["auroc_h_ci"] = auroc(-h, fs), boot_auroc_ci(-h, fs, cl, seed=seed)
            m["auroc_h_not_yet"] = auroc(-h[nyet], fs[nyet])
            pr = [(r.get("prior") or {}).get(ax) for r in rr]
            if all(x is not None for x in pr):
                pr = np.array(pr, dtype=np.float64)
                m["auroc_text"], m["auroc_text_ci"] = auroc(pr, fs), boot_auroc_ci(pr, fs, cl, seed=seed)
            m.update({"n": len(rr), "n_episodes": int(len(set(cl))), "auroc": auroc(-q, fs),
                      "auroc_not_yet": auroc(-q[nyet], fs[nyet]),
                      "auroc_not_yet_ci": boot_auroc_ci(-q[nyet], fs[nyet], cl[nyet], seed=seed),
                      "auroc_ci": boot_auroc_ci(-q, fs, cl, seed=seed), "auroc_base": auroc(-qb, fs),
                      "auroc_base_ci": boot_auroc_ci(-qb, fs, cl, seed=seed),
                      "prec_at_0": tp / max(float(pred.sum()), 1e-9), "rec_at_0": tp / max(float(rec.sum()), 1e-9),
                      "v_le_h": float((q <= h + V_LE_H_EPS).mean()), "q_mean": float(q.mean()), "h_mean": float(h.mean())})
            by_state = defaultdict(list)
            for r in rr:
                if r.get("y") is not None:
                    by_state[(r["src"], r["i"])].append((r["q"][ax], r["y"]))
            rho = [spearman([a for a, _ in v], [b for _, b in v]) for v in by_state.values() if len(v) >= 3]
            rho = [x for x in rho if x == x]
            if rho:
                m["spearman_q_y"], m["n_spearman_states"] = float(np.mean(rho)), len(rho)
            if ax == "train":   # paraphrase vs rule spread over the SAME states (review M8)
                by_state = defaultdict(list)
                for r in rr:
                    by_state[(r["src"], r["i"])].append(r)
                para, across = [], []
                for v in by_state.values():
                    p_ = [np.std(r["q"]["para"]) for r in v if len(r["q"].get("para", [])) >= 2]
                    if p_ and len(v) >= 2:
                        para += p_
                        across.append(np.std([r["q"]["train"] for r in v]))
                if para:
                    m["para_std"], m["rule_std"] = float(np.mean(para)), float(np.mean(across))
                    m["para_std_ratio"] = m["para_std"] / max(m["rule_std"], 1e-9)
    return out


SVS_CONS = ("avoid", "stay", "upright")


def svs_state(r: dict) -> str | None:
    """E0b stratum of a state: an object of this consequence is harmed now (h_any < 0), or one will be within
    25 ticks (not yet), or neither."""
    if r.get("h_any") is None:
        return None
    return "already_harmed" if r["h_any"] < 0 else ("not_yet" if r.get("fs_any") else "neither")


def seeing_vs_selecting(rows: list[dict], B: int = 200, seed: int = 0) -> dict:
    """E0b (review round 3, Part II): over (state, rule) rows of avoid / stay / upright rules with their training
    sentence, Spearman rho(Q, h_named) (the named rule's current h) and rho(Q, h_any) (the most endangered object
    of the same consequence in the scene, named or not), per task x state stratum, with episode-cluster
    bootstrap intervals. rho(Q, h_any) clearly above rho(Q, h_named): the model sees the danger but does not tie
    it to the named object. h_named vs h_any themselves (how far the two references differ) is reported too."""
    rr = [r for r in rows if r["consequence"] in SVS_CONS and r["eligible"].get("train") and "train" in r["q"]
          and r.get("h_any") is not None]
    out = {}
    groups = defaultdict(list)
    for r in rr:
        st = svs_state(r)
        for g in ((r["task"], "all"), (r["task"], st)):
            groups[g].append(r)
    rng = np.random.default_rng(seed)
    for (task, st), g in sorted(groups.items()):
        q = np.array([r["q"]["train"] for r in g])
        hn = np.array([r["h"] for r in g])
        ha = np.array([r["h_any"] for r in g])
        cl = np.array([_ep(r) for r in g])
        uniq, inv = np.unique(cl, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        m = {"n": len(g), "n_episodes": int(len(uniq)), "rho_named": spearman(q, hn), "rho_any": spearman(q, ha),
             "rho_named_any": spearman(hn, ha), "frac_named_is_min": float(np.mean(hn <= ha + 1e-9))}
        boots = {"rho_named": [], "rho_any": [], "any_minus_named": []}
        if len(uniq) >= 2:
            for _ in range(B):
                sel = np.concatenate([members[k] for k in rng.integers(len(uniq), size=len(uniq))])
                a, b = spearman(q[sel], hn[sel]), spearman(q[sel], ha[sel])
                if a == a and b == b:
                    boots["rho_named"].append(a), boots["rho_any"].append(b), boots["any_minus_named"].append(b - a)
        for k, v in boots.items():
            m[k + "_ci"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= B // 2 \
                else (float("nan"), float("nan"))
        m["any_minus_named"] = m["rho_any"] - m["rho_named"]
        out[f"{task}/{st}"] = m
    return out


def svs_md(S: dict, title: str) -> str:
    L = [f"# Seeing versus selecting: {title}", "",
         "Per (state, rule) of avoid / stay / upright rules, the training sentence. rho_named = Spearman(Q, h of the "
         "named rule now); rho_any = Spearman(Q, the lowest h now over the scene's rules of the same consequence, "
         "named or not). States: already_harmed (an object of that consequence is harmed now), not_yet (one will be "
         "within 25 ticks), neither. [ ] = episode-cluster bootstrap 95% interval. Reading rule: rho_any above "
         "rho_named with non-overlapping intervals = the model sees the danger but does not select the named "
         "object. named = min: share of rows whose named rule is itself the most endangered.", "",
         "| task | state | rows (episodes) | rho(Q, h named) | rho(Q, h any) | any - named | rho(h named, h any) | "
         "named = min |", "|---|---|---|---|---|---|---|---|"]
    for k, m in S.items():
        task, st = k.split("/")
        c = lambda key: (f"{m[key]:.2f} [{m[key + '_ci'][0]:.2f}, {m[key + '_ci'][1]:.2f}]"  # noqa: E731
                         if m[key + "_ci"][0] == m[key + "_ci"][0] else f"{m[key]:.2f}")
        L.append(f"| {task} | {st} | {m['n']} ({m['n_episodes']}) | {c('rho_named')} | {c('rho_any')} | "
                 f"{c('any_minus_named')} | {m['rho_named_any']:.2f} | {m['frac_named_is_min']:.2f} |")
    return "\n".join(L) + "\n"


def _ci(m: dict, key: str, nd: int = 2) -> str:
    v, c = m.get(key), m.get(key + "_ci")
    if v is None or v != v:
        return "–"
    return f"{v:.{nd}f} [{c[0]:.{nd}f}, {c[1]:.{nd}f}]" if c and c[0] == c[0] else f"{v:.{nd}f}"


def tables_md(M: dict, title: str) -> str:
    sym = lambda ax, g: M.get(f"{ax}/symmetric/{g}", {})  # noqa: E731
    L = [f"# Offline language probe: {title}", "",
         "rank_acc = P[Q(harmful rule) < Q(benign rule)] over symmetric pairs at the same state (chance 0.5), "
         "labels from fails_soon (h < 0 within 25 ticks); ties count 0.5. [ ] = episode-bootstrap 95% interval "
         "(optimistic: 7-10 validation episodes per task, percentile intervals under-cover); cells give "
         "(pairs / episodes). The shuffled axis is 0.5 by construction (both members get other-episode "
         "sentences): a code check, not a result. "
         f"States whose 25-tick window runs past a time-capped episode's end are left out "
         f"({M.get('train/_censored_pairs', {}).get('n', 0)} training-axis pairs).", "",
         "| axis | pooled | task macro | " + " | ".join(f"T{i + 1}" for i in range(len(CANON_TASKS))) + " | referent | consequence |",
         "|---|---|---|" + "---|" * len(CANON_TASKS) + "---|---|"]
    for ax in AXES:
        p = sym(ax, "pooled")
        cells = [f"{_ci(p, 'rank_acc', 3)} ({p['n_pairs']} / {p['n_episodes']})" if p else "–",
                 f"{p['rank_acc_task_macro']:.3f}" if p.get("rank_acc_task_macro") is not None else "–"]
        for t in CANON_TASKS:
            m = sym(ax, f"task/{t}")
            cells.append(f"{m['rank_acc']:.2f} ({m['n_pairs']})" if m else "–")
        cells.append(f"{_ci(p, 'rank_acc_referent')} ({p['n_referent']})" if p.get("rank_acc_referent") is not None else "–")
        cells.append(f"{_ci(p, 'rank_acc_consequence')} ({p['n_consequence']})" if p.get("rank_acc_consequence") is not None else "–")
        L.append(f"| {ax} | " + " | ".join(cells) + " |")
    L += ["", "**Per task: the harmful rule already broken, or not yet** (training templates; rounds 2 C1, 3 A1). "
          "Already broken: the current h ranks every pair right by construction, so these cells test whether the "
          "model reads the named item's PRESENT state. Not yet: the current h is no reference (it can rank below "
          "chance, e.g. a benign rule that sits below 1 all episode), and the sentence prior (words only, no scene) "
          "shows how much of the order the words alone give; model - prior is paired, on the same pairs.", "",
          "| task | state | model (pairs / episodes) | h now | sentence prior | model - prior |", "|---|---|---|---|---|---|"]
    for t in CANON_TASKS:
        for st_, name in (("violated", "already broken"), ("not_yet", "not yet")):
            m = sym("train", f"task_state/{t}/{st_}")
            if m:
                L.append(f"| {t} | {name} | {_ci(m, 'rank_acc')} ({m['n_pairs']} / {m['n_episodes']}) | "
                         f"{_ci(m, 'rank_acc_h')} | {_ci(m, 'rank_acc_text')} | {_ci(m, 'model_minus_text')} |")
    L += ["", "**Referent vs consequence, per task and state (round 3, A2 / A5).** Referent pairs: the same "
          "consequence on different items (which item does the sentence name?). Consequence pairs: the same item "
          "under different consequences. prior: the sentence-only reader on the same pairs; model - prior paired.", "",
          "| task | state | referent: model (pairs) | prior | model - prior | consequence: model (pairs) | prior | "
          "model - prior |", "|---|---|---|---|---|---|---|---|"]
    rc = [(t, f"task/{t}", f"task_state/{t}/") for t in RC_TASKS] + [("T1 + T3 + T7", "rc_pool", "rc_state/")]
    for t, k_all, k_st in rc:
        for st_, name, key in (("all", "all", k_all), ("violated", "already broken", k_st + "violated"),
                               ("not_yet", "not yet", k_st + "not_yet")):
            m = sym("train", key)
            if not m or (m.get("rank_acc_referent") is None and m.get("rank_acc_consequence") is None):
                continue
            L.append(f"| {t} | {name} | {_ci(m, 'rank_acc_referent')} ({m.get('n_referent', 0)}) | "
                     f"{_ci(m, 'rank_acc_referent_text')} | {_ci(m, 'referent_minus_text')} | "
                     f"{_ci(m, 'rank_acc_consequence')} ({m.get('n_consequence', 0)}) | "
                     f"{_ci(m, 'rank_acc_consequence_text')} | {_ci(m, 'consequence_minus_text')} |")
    t3 = sorted(k.split("/", 3)[3] for k in M if k.startswith("train/symmetric/t3_benign/"))
    if t3:
        L += ["", "**T3 by the benign rule's consequence (round 3, A1).** zone_include (keep the box on the table) "
              "rarely breaks, so a sentence naming it is benign almost always.", "",
              "| state / benign consequence | model (pairs) | h now | sentence prior | model - prior |",
              "|---|---|---|---|---|"]
        for k in t3:
            m = sym("train", f"t3_benign/{k}")
            L.append(f"| {k.replace('/', ' / ')} | {_ci(m, 'rank_acc')} ({m['n_pairs']}) | {_ci(m, 'rank_acc_h')} | "
                     f"{_ci(m, 'rank_acc_text')} | {_ci(m, 'model_minus_text')} |")
    L += ["", "**Label checks (review E1, E2; round 2 C2, C4), training templates.** h now: rank by each rule's "
          "current h, no model; ties (mostly h = 1.0 on both) count 0.5, so it is not a perfect-reader ceiling. "
          "y = the return of the NOMINAL continuation, (1 - gamma) h_t + gamma min(h_t, y_t+1) over the recorded "
          "h: not what the critic learns (it bootstraps through its actor). y ranks like the label: P[y(harmful) < "
          "y(benign)]. Q vs y: agreement of Q's order with y's (|y1 - y2| > 0.05). text prior: a logistic regression "
          "on the words of the pair's sentences, fit on training data, never looking at the scene; model - text: "
          "paired difference with an episode bootstrap. Spearman: per state, Q vs y across its rules (>= 3).", "",
          "| group | model (pairs) | h now | y ranks like the label | Q vs y (pairs) | text prior | model - text | "
          "Spearman Q, y (states) |", "|---|---|---|---|---|---|---|---|"]
    for g in ["pooled"] + [f"task/{t}" for t in CANON_TASKS]:
        m = sym("train", g)
        st = M.get(f"train/state/{g}", {})
        if not m:
            continue
        L.append(f"| {g} | {_ci(m, 'rank_acc')} ({m['n_pairs']}) | {_ci(m, 'rank_acc_h')} | {_ci(m, 'rank_acc_y')} | "
                 f"{_ci(m, 'rank_acc_mc')} ({m.get('n_mc', 0)}) | {_ci(m, 'rank_acc_text')} | {_ci(m, 'model_minus_text')} | "
                 f"{st.get('spearman_q_y', float('nan')):.2f} ({st.get('n_spearman_states', 0)}) |")
    L += ["", "**By the harmful rule's time to its first violation (review E1c)**, training templates. 1-7 ticks is "
          "the closest offline test of foresight; the nominal-continuation value of a violation k ticks ahead is "
          "1 - 2 * 0.9^k (0.30 at k = 10), but the critic values its actor's continuation, so a violation the "
          "actor can still avoid may rightly stay a near-tie.", "",
          "| harmful rule | rank_acc (pairs / episodes) | h now | vs HJ return |", "|---|---|---|---|"]
    for name, _, _ in TTV_BINS:
        m = sym("train", f"ttv/{name}")
        if m:
            L.append(f"| {name} | {_ci(m, 'rank_acc')} ({m['n_pairs']} / {m['n_episodes']}) | {_ci(m, 'rank_acc_h')} | "
                     f"{_ci(m, 'rank_acc_mc')} |")
    L += ["", "**T5 pairs by size words (review E2; crossed with the state, round 3 A1):** the at-risk block is "
          "almost always the tall one, so a size word alone can give the order.", "",
          "| T5 pairs | state | rank_acc (pairs) | h now | sentence prior | model - prior |", "|---|---|---|---|---|---|"]
    for k in ("size_word", "no_size_word"):
        for st_, name, key in (("all", "all", f"t5/{k}"), ("violated", "already broken", f"t5_state/{k}/violated"),
                               ("not_yet", "not yet", f"t5_state/{k}/not_yet")):
            m = sym("train", key)
            if m:
                L.append(f"| {k.replace('_', ' ')} | {name} | {_ci(m, 'rank_acc')} ({m['n_pairs']}) | "
                         f"{_ci(m, 'rank_acc_h')} | {_ci(m, 'rank_acc_text')} | {_ci(m, 'model_minus_text')} |")
    L += ["", "Mixed pairs (only one member axis-eligible): rank_acc - rank_acc_base (r1's sentence replaced by "
          "another rule's of the same state).", "",
          "| axis | n pairs | rank_acc | rank_acc_base | difference |", "|---|---|---|---|---|"]
    for ax in ("heldout_word", "heldout_item"):
        m = M.get(f"{ax}/mixed/pooled")
        if m:
            L.append(f"| {ax} | {m['n_pairs']} | {_ci(m, 'rank_acc', 3)} | {_ci(m, 'rank_acc_base', 3)} | "
                     f"{m.get('rank_acc_minus_base', float('nan')):+.3f} |")
    L += ["", "Per (state, rule): AUROC of -Q for fails_soon (base: another rule's sentence of the same state), "
          "precision / recall of Q < 0 against fails_soon_recall (7 ticks), v_le_h = P[Q <= h + 0.05].", "",
          "| axis | n (episodes) | auroc | auroc, harm not yet happening | auroc_base | prec@0 | rec@0 | v_le_h | "
          "para_std | para_std_ratio |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ax in AXES:
        m = M.get(f"{ax}/state/pooled")
        if m:
            L.append(f"| {ax} | {m['n']} ({m.get('n_episodes', '?')}) | {_ci(m, 'auroc', 3)} | "
                     f"{_ci(m, 'auroc_not_yet', 3)} | {_ci(m, 'auroc_base', 3)} | "
                     f"{m['prec_at_0']:.3f} | {m['rec_at_0']:.3f} | {m['v_le_h']:.3f} | {m.get('para_std', float('nan')):.3f} | "
                     f"{m.get('para_std_ratio', float('nan')):.3f} |")
    L += ["", "**AUROC per task (train axis; round 3, A4).** Per (state, rule): AUROC of -Q for fails_soon, and "
          "of the current h (-h now) and the sentence prior on the same rows. Pooled AUROC rewards differences in "
          "base rates between tasks: read the task rows. The bead tasks (T2 / T4 / T6) compare only for a "
          "checkpoint trained on the bead-count h (Data: bead_h_train).", "",
          "| rows | n (episodes) | -Q | -h now | prior | not yet: -Q | not yet: -h now | v_le_h |",
          "|---|---|---|---|---|---|---|---|"]
    for g, name in ([("pooled", "pooled"), ("pooled_no_bead", "pooled without T2 / T4 / T6")]
                    + [(f"task/{t}", t) for t in CANON_TASKS] + [("t5_tall", "T5, tall-block rules only")]):
        m = M.get(f"train/state/{g}")
        if m:
            L.append(f"| {name} | {m['n']} ({m.get('n_episodes', '?')}) | {_ci(m, 'auroc', 3)} | {_ci(m, 'auroc_h', 3)} | "
                     f"{_ci(m, 'auroc_text', 3)} | {_ci(m, 'auroc_not_yet', 3)} | "
                     f"{m.get('auroc_h_not_yet', float('nan')):.3f} | {m['v_le_h']:.3f} |")
    return "\n".join(L) + "\n"


def figures(M: dict, rows: list[dict], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    w = 0.15
    for k, a in enumerate(AXES):
        ys = [M.get(f"{a}/symmetric/task/{t}", {}).get("rank_acc", np.nan) for t in CANON_TASKS]
        ax.bar(np.arange(len(CANON_TASKS)) + (k - 2) * w, ys, w, label=a)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(CANON_TASKS)), [f"T{i + 1}" for i in range(len(CANON_TASKS))])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rank_acc")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(out / "rank_acc_bars.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    ph = [(h["q"]["train"], b["q"]["train"]) for h, b in _pairs(rows) if "train" in h["q"] and "train" in b["q"]]
    if ph:
        a = np.array(ph)
        ax.scatter(a[:, 1], a[:, 0], s=3, alpha=0.4)
        lo, hi = float(a.min()), float(a.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("Q(benign rule)")
    ax.set_ylabel("Q(harmful rule)")
    fig.tight_layout()
    fig.savefig(out / "q_pairs_scatter.png", dpi=110)
    plt.close(fig)


def examples_md(ev: Evaluator, rows: list[dict], path: Path, n: int = 8) -> None:
    by_state = defaultdict(list)
    for r in rows:
        by_state[(r["src"], r["i"])].append(r)
    keys = [k for k, v in by_state.items() if len({x["fails_soon"] for x in v}) == 2][:n]
    L = ["# Offline probe examples", ""]
    for k in keys:
        rs = by_state[k]
        L += [f"## {rs[0]['task']} {rs[0]['emb']} episode {rs[0]['episode_id']} t={rs[0]['t']}", "",
              "| rule | Q train | Q heldout_template | h | fails_soon |", "|---|---|---|---|---|"]
        for r in rs:
            qt = r["q"].get("train")
            qh = r["q"].get("heldout_template")
            L.append(f"| {r['rule']} | {'' if qt is None else f'{qt:+.3f}'} | {'' if qh is None else f'{qh:+.3f}'} | "
                     f"{r['h']:+.3f} | {r['fails_soon']} |")
        L.append("")
    path.write_text("\n".join(L))


def file_sha(path: Path, n: int = 12) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


EVAL_CODE = ("main/lang/eval_offline.py", "main/lang/data.py", "main/lang/network.py", "main/lang/text.py",
             "main/lang/config.py", "main/lang/train.py", "main/network/heads.py", "main/network/trunk.py",
             "main/network/encoder.py", "envs/main/item_envs/language/__init__.py",
             "envs/main/item_envs/language/paraphrases.yaml")


def eval_code_hash() -> str:
    """Hash of the code an offline score depends on (review round 3, B1)."""
    import hashlib

    from . import REPO

    h = hashlib.md5()
    for f in EVAL_CODE:
        p = REPO / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:12]


def data_info(ev, rows, roots) -> dict:
    """What the numbers cover (review M6): tasks scored, bead-task rows left out for their old bead h, and
    which bead h the scored bead rows carry ("count" = n_violating stored, "old", or "none" scored)."""
    from .data import BEAD_TASKS

    bead_srcs = [s for s in ev.sources if s.rows.size and any(CANON_TASKS[int(t)] in BEAD_TASKS for t in np.unique(s.task))]
    bead_h = ("none" if not bead_srcs else
              "count" if all((s.root / "item_buffer" / "n_violating.npy").exists() for s in bead_srcs) else "old")
    return {"tasks": sorted({r["task"] for r in rows}), "embodiments": sorted({r["emb"] for r in rows}),
            "stale_bead_rows_dropped": int(sum(s.n_stale_dropped for s in ev.sources)),
            "bead_h": bead_h, "val_roots": [str(r) for r in roots],
            "prior_roots": list(ev.prior.roots) if ev.prior is not None else None,
            "prior_docs": int(ev.prior.n_docs) if ev.prior is not None else None,
            "bead_h_train": ev.bead_h_train, "ckpt_sha": ev.ckpt_sha, "ckpt_step": ev.ckpt_step,
            "eval_code_hash": eval_code_hash()}


def main(argv=None) -> int:
    from .train import caches_of, roots_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoint.pt")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--text-cache", nargs="*", default=None)
    ap.add_argument("--states-per-task", type=int, default=2000)
    ap.add_argument("--paraphrases", type=int, default=3)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--embodiments", default="trained",
                    help="validation embodiments scored: 'trained' (the run's, default), 'all', or a comma list")
    ap.add_argument("--tasks", default="all", help="validation tasks scored (comma list)")
    ap.add_argument("--seeing-vs-selecting", action="store_true",
                    help="also write seeing_vs_selecting.{json,md} (review round 3, Part II E0b)")
    ap.add_argument("--out-name", default=None, help="output directory name under eval_offline/ (default: ckpt)")
    ap.add_argument("--prior-data", nargs="*", default=None,
                    help="training data roots for the scene-free text prior (rank_acc_text), e.g. data/lang/warmup")
    args = ap.parse_args(argv)
    roots = roots_of(args.val)
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.val)
    embs = args.embodiments if args.embodiments in ("trained", "all") else tuple(args.embodiments.split(","))
    ev = Evaluator(Path(args.run), args.ckpt, roots, text_caches=caches, batch=args.batch, embodiments=embs,
                   tasks=None if args.tasks == "all" else args.tasks.split(","))
    if args.prior_data:
        prior_roots = roots_of(args.prior_data)
        if not prior_roots:   # else the prior is a constant 0.5 without a word (round 3, B6)
            raise SystemExit(f"--prior-data {args.prior_data}: no data shards found")
        overlap = {str(Path(r).resolve()) for r in prior_roots} & {str(Path(r).resolve()) for r in roots}
        if overlap:   # the prior must not have seen the validation episodes (round 2 C4)
            raise SystemExit(f"--prior-data overlaps --val: {sorted(overlap)}")
        ev.prior = TextPrior(prior_roots, ev.cfg.k_prev)
        if not ev.prior.n_docs:
            raise SystemExit(f"--prior-data {args.prior_data}: the text prior found no training sentences")
    M, rows = ev.evaluate(args.states_per_task, args.paraphrases)
    out = Path(args.run) / "eval_offline" / (args.out_name or Path(args.ckpt).stem)
    out.mkdir(parents=True, exist_ok=True)
    import gzip

    with gzip.open(out / "rows.json.gz", "wt") as f:   # every (state, rule) row: later checks need no GPU
        json.dump(rows, f, default=float)
    if args.seeing_vs_selecting:
        S = seeing_vs_selecting(rows)
        (out / "seeing_vs_selecting.json").write_text(json.dumps(S, indent=1))
        (out / "seeing_vs_selecting.md").write_text(svs_md(S, f"{Path(args.run).name} / {args.ckpt} (step {ev.ckpt_step})"))
    data = data_info(ev, rows, roots)
    (out / "metrics.json").write_text(json.dumps({"ckpt": args.ckpt, "step": ev.ckpt_step, "data": data, "metrics": M},
                                                 indent=1))
    (out / "tables.md").write_text(tables_md(M, f"{Path(args.run).name} / {args.ckpt} (step {ev.ckpt_step})")
                                   + "\nData: " + json.dumps({k: (f"{len(v)} roots" if isinstance(v, list) and k.endswith("roots")
                                                               else v) for k, v in data.items()}) + "\n")
    figures(M, rows, out)
    examples_md(ev, rows, out / "examples.md")
    ev.text.save_extra()
    print((out / "tables.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
