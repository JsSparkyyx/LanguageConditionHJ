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

from .config import CANON_TASKS, LangConfig  # noqa: E402
from .data import LangSource, stable_hash  # noqa: E402
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


class TextPrior:
    """P(fails_soon | rule words) from training data, Laplace-smoothed, backing off to the consequence: a
    reader of the sentence that never looks at the scene (review E2)."""

    def __init__(self, roots, k_prev: int = 2):
        from collections import Counter

        self.k, self.n = Counter(), Counter()
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
                    for key in (rule_words(rule, rec), (rule["consequence"],)):
                        self.k[key] += int(fs.sum())
                        self.n[key] += int(fs.size)

    def __call__(self, words: tuple) -> float:
        key = words if self.n[words] else (words[0],)
        return (self.k[key] + 1.0) / (self.n[key] + 2.0)


class Evaluator:
    def __init__(self, run: Path, ckpt: str, val_roots, device="cuda", text_caches=(), batch: int = 96,
                 text_encoder=None):
        from .network import LangPolicyEncoder, LangTwinCritic
        from .text import TextStore
        from .train import load_policy

        self.run = Path(run)
        ck = torch.load(self.run / ckpt if not Path(ckpt).is_absolute() else ckpt, map_location=device,
                        weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        self.dev = device
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
            for i in np.flatnonzero(s.t % every == 0):
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
            self._ep[key] = ({int(t): k for k, t in enumerate(t_in)}, hj_return(h, self.gamma), next_violation(h))
        return self._ep[key]

    def _base_sentence(self, rule, rec, j, eid, rng) -> str:
        """The base for auroc_base / rank_acc_base: a sentence naming ANOTHER sampleable rule of the same
        state (same task and scene; review E3); another episode's only when the state has no other rule."""
        others = [r for r in rec["rules"] if r.get("sampleable") and r["rule_id"] != rule["rule_id"]]
        if others:
            return language.sample_sentence(others[int(rng.integers(len(others)))], rec, rng, "train", self.bank)
        return self._other_sentence(j, eid, rng)

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
    @torch.no_grad()
    def q_batch(self, reads: list[dict], sentences: list[str]) -> np.ndarray:
        from .train import _t

        d = self.dev
        b = {k: np.stack([r[k] for r in reads]) for k in reads[0]}
        words, wmask = self.text.lookup(sentences)
        enc = self.pe(_t(b["state_tokens"], dev=d), _t(b["body_tokens"], dev=d), _t(b["link_pos"], dev=d),
                      _t(b["body_mask"], torch.bool, d), _t(b["joint_index"], torch.int64, d),
                      _t(b["scene_tokens"], dev=d), _t(b["scene_pos"], dev=d), words=words, word_mask=wmask)
        q = self.critics.qmin(enc, _t(b["dtheta"], dev=d), _t(b["Jlin"], dev=d), _t(b["Jang"], dev=d),
                              _t(b["joint_index"], torch.int64, d))
        return q.float().cpu().numpy()

    def run_items(self, items, cache) -> np.ndarray:
        """items: (state key, sentence) -> Q, batched; state reads cached per chunk."""
        out = np.zeros(len(items), np.float32)
        for a in range(0, len(items), self.batch):
            chunk = items[a:a + self.batch]
            reads = [cache[k] for k, _ in chunk]
            out[a:a + len(chunk)] = self.q_batch(reads, [s for _, s in chunk])
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
                tmap, y_ep, nv_ep = self._episode(j, eid)
                k_ep = tmap.get(int(s.t[i]))
                for rule in rec["rules"]:
                    el = {ax: eligible(rule, rec, ax, self.bank) for ax in AXES}
                    if not any(el.values()):
                        continue
                    col = rule["idx"]
                    base = {"src": j, "i": i, "task": rec["task"], "emb": rec["embodiment"], "episode_id": eid,
                            "t": int(s.t[i]), "rule": rule["rule_id"], "col": col, "consequence": rule["consequence"],
                            "referent": referent(rule, rec), "split": rule.get("split"),
                            "fails_soon": bool(A["fails_soon"][slot, col]), "recall": bool(s.recall[slot, col]),
                            "h": float(A["h_rules"][slot, col]), "eligible": el, "q": {},
                            "y": float(y_ep[k_ep, col]) if k_ep is not None else None,
                            "ttv": int(nv_ep[k_ep, col]) if k_ep is not None and nv_ep[k_ep, col] >= 0 else None,
                            "words": list(rule_words(rule, rec)), "sentences": {}}
                    if self.prior is not None:
                        base["prior"] = float(self.prior(tuple(base["words"])))
                    r_idx = len(rows)
                    rows.append(base)
                    for ax in AXES:
                        if not el[ax]:
                            continue
                        sent = (self._other_sentence(j, eid, rng) if ax == "shuffled"
                                else axis_sentence(rule, rec, ax, (eid, int(s.t[i]), col), self.bank))
                        base["sentences"][ax] = sent
                        items.append((k, sent))
                        meta.append((r_idx, ax))
                    # base sentence for auroc_base / rank_acc_base: another rule of the same state (E3)
                    base["sentences"]["base"] = self._base_sentence(rule, rec, j, eid, rng)
                    items.append((k, base["sentences"]["base"]))
                    meta.append((r_idx, "base"))
                    if rule.get("sampleable") and len(para_rows) < para_states * len(CANON_TASKS) and \
                            stable_hash((eid, int(s.t[i]), col, "para")) % 4 == 0:
                        ss = language.sentences(rule, rec, "train", self.bank)
                        if len(ss) >= 2:
                            pick = [ss[q] for q in rng.choice(len(ss), size=min(n_para, len(ss)), replace=False)]
                            for sent in pick:
                                items.append((k, sent))
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
TTV_BINS = (("violated", 0, 0), ("1-7", 1, 7), ("8-25", 8, 25))
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
        b = _ttv_bin(hr.get("ttv"))
        if b:
            g.append(("ttv", b))
        if hr["task"] == "tray_transport":
            sents = [(x.get("sentences") or {}).get(ax) or "" for x in (hr, bn)]
            sw = any(w in sent.lower().split() for sent in sents for w in SIZE_WORDS)
            g.append(("t5", "size_word" if sw else "no_size_word"))
        return g

    for ax in AXES:
        acc = defaultdict(lambda: defaultdict(list))
        for hr, bn in _pairs(rows):
            e1, e2 = hr["eligible"][ax], bn["eligible"][ax]
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
            ok = float(qh < qb)
            for g in groups(hr, bn, ax):
                d = acc[(kind,) + g]
                d["rank"].append((ep, ok))
                if qh0 is not None:
                    d["rank_base"].append((ep, float(qh0 < qb0)))
                if same_cons and not same_ref:
                    d["rank_ref"].append((ep, ok))
                if same_ref and not same_cons:
                    d["rank_cons"].append((ep, ok))
                d["rank_h"].append((ep, 0.5 if hr["h"] == bn["h"] else float(hr["h"] < bn["h"])))
                if hr.get("y") is not None and bn.get("y") is not None and abs(hr["y"] - bn["y"]) > MC_EPS:
                    d["rank_mc"].append((ep, float(ok == float(hr["y"] < bn["y"]))))
                if hr.get("prior") is not None and bn.get("prior") is not None:
                    d["rank_text"].append((ep, 0.5 if hr["prior"] == bn["prior"] else float(hr["prior"] > bn["prior"])))
        for key, d in acc.items():
            name = "/".join([ax] + list(key))
            m = {"n_pairs": len(d["rank"]), "n_episodes": len({c for c, _ in d["rank"]}),
                 "rank_acc": float(np.mean([v for _, v in d["rank"]])), "rank_acc_ci": boot_mean_ci(d["rank"], seed=seed)}
            for src, dst, n_key in (("rank_base", "rank_acc_base", None), ("rank_ref", "rank_acc_referent", "n_referent"),
                                    ("rank_cons", "rank_acc_consequence", "n_consequence"), ("rank_h", "rank_acc_h", None),
                                    ("rank_mc", "rank_acc_mc", "n_mc"), ("rank_text", "rank_acc_text", None)):
                if d[src]:
                    m[dst] = float(np.mean([v for _, v in d[src]]))
                    m[dst + "_ci"] = boot_mean_ci(d[src], seed=seed)
                    if n_key:
                        m[n_key] = len(d[src])
            if "rank_acc_base" in m:
                m["rank_acc_minus_base"] = m["rank_acc"] - m["rank_acc_base"]
            out[name] = m
        for kind in ("symmetric", "mixed"):   # task-macro mean (every task weighs the same; review E4)
            tm = [out[f"{ax}/{kind}/task/{t}"]["rank_acc"] for t in CANON_TASKS if f"{ax}/{kind}/task/{t}" in out]
            if tm and f"{ax}/{kind}/pooled" in out:
                out[f"{ax}/{kind}/pooled"]["rank_acc_task_macro"] = float(np.mean(tm))
        # per (state, rule)
        sel = [r for r in rows if r["eligible"][ax] and ax in r["q"]]
        for gname, f in (("pooled", lambda r: True),) + tuple(
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
            m = out.setdefault(f"{ax}/state/{gname}", {})
            m.update({"n": len(rr), "n_episodes": int(len(set(cl))), "auroc": auroc(-q, fs),
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


def _ci(m: dict, key: str, nd: int = 2) -> str:
    v, c = m.get(key), m.get(key + "_ci")
    if v is None or v != v:
        return "–"
    return f"{v:.{nd}f} [{c[0]:.{nd}f}, {c[1]:.{nd}f}]" if c and c[0] == c[0] else f"{v:.{nd}f}"


def tables_md(M: dict, title: str) -> str:
    sym = lambda ax, g: M.get(f"{ax}/symmetric/{g}", {})  # noqa: E731
    L = [f"# Offline language probe: {title}", "",
         "rank_acc = P[Q(harmful rule) < Q(benign rule)] over symmetric pairs at the same state (chance 0.5), "
         "labels from fails_soon (h < 0 within 25 ticks). [ ] = episode-bootstrap 95% interval; cells give "
         "(pairs / episodes). The shuffled axis is 0.5 by construction (both members get other-episode "
         "sentences): a code check, not a result.", "",
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
    L += ["", "**Label checks (review E1, E2), training templates.** h now: rank by each rule's current h, no model "
          "(a perfect reader with no foresight). vs HJ return: agreement of Q's order with the discounted HJ return "
          "y_t = (1 - gamma) h_t + gamma min(h_t, y_t+1) of the recorded h (what the critic learns), pairs with "
          "|y1 - y2| > 0.05. text prior: rank by P(fails_soon | consequence + referent words) from training data, "
          "never looking at the scene. Spearman: per state, Q vs y across its rules (>= 3 rules).", "",
          "| group | rank_acc (pairs) | h now | vs HJ return (pairs) | text prior | Spearman Q, y (states) |",
          "|---|---|---|---|---|---|"]
    for g in ["pooled"] + [f"task/{t}" for t in CANON_TASKS]:
        m = sym("train", g)
        st = M.get(f"train/state/{g}", {})
        if not m:
            continue
        L.append(f"| {g} | {_ci(m, 'rank_acc')} ({m['n_pairs']}) | {_ci(m, 'rank_acc_h')} | "
                 f"{_ci(m, 'rank_acc_mc')} ({m.get('n_mc', 0)}) | {_ci(m, 'rank_acc_text')} | "
                 f"{st.get('spearman_q_y', float('nan')):.2f} ({st.get('n_spearman_states', 0)}) |")
    L += ["", "**By the harmful rule's time to its first violation (review E1c)**, training templates. A gamma = 0.9 "
          "critic with h = 1 before and -1 after a violation k ticks ahead values it 1 - 2 * 0.9^k (0.30 at k = 10).", "",
          "| harmful rule | rank_acc (pairs / episodes) | h now | vs HJ return |", "|---|---|---|---|"]
    for name, _, _ in TTV_BINS:
        m = sym("train", f"ttv/{name}")
        if m:
            L.append(f"| {name} | {_ci(m, 'rank_acc')} ({m['n_pairs']} / {m['n_episodes']}) | {_ci(m, 'rank_acc_h')} | "
                     f"{_ci(m, 'rank_acc_mc')} |")
    L += ["", "**T5 pairs by size words (review E2):** the at-risk block is always the tall one.", "",
          "| T5 pairs | rank_acc (pairs) |", "|---|---|"]
    for k in ("size_word", "no_size_word"):
        m = sym("train", f"t5/{k}")
        if m:
            L.append(f"| {k.replace('_', ' ')} | {_ci(m, 'rank_acc')} ({m['n_pairs']}) |")
    L += ["", "**Referent vs consequence within the tasks that have both (review E5)**, training templates.", "",
          "| task | referent (pairs) | consequence (pairs) |", "|---|---|---|"]
    for t in ("selective_avoid", "push_zone", "gap_place"):
        m = sym("train", f"task/{t}")
        if m:
            L.append(f"| {t} | {_ci(m, 'rank_acc_referent')} ({m.get('n_referent', 0)}) | "
                     f"{_ci(m, 'rank_acc_consequence')} ({m.get('n_consequence', 0)}) |")
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
          "| axis | n (episodes) | auroc | auroc_base | prec@0 | rec@0 | v_le_h | para_std | para_std_ratio |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ax in AXES:
        m = M.get(f"{ax}/state/pooled")
        if m:
            L.append(f"| {ax} | {m['n']} ({m.get('n_episodes', '?')}) | {_ci(m, 'auroc', 3)} | {_ci(m, 'auroc_base', 3)} | "
                     f"{m['prec_at_0']:.3f} | {m['rec_at_0']:.3f} | {m['v_le_h']:.3f} | {m.get('para_std', float('nan')):.3f} | "
                     f"{m.get('para_std_ratio', float('nan')):.3f} |")
    L += ["", "Per task (train axis):", "", "| task | n | auroc | auroc_base | rank_acc | v_le_h |", "|---|---|---|---|---|---|"]
    for t in CANON_TASKS:
        m = M.get(f"train/state/task/{t}")
        p = sym("train", f"task/{t}")
        if m:
            L.append(f"| {t} | {m['n']} | {_ci(m, 'auroc', 3)} | {_ci(m, 'auroc_base', 3)} | "
                     f"{_ci(p, 'rank_acc', 3)} | {m['v_le_h']:.3f} |")
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


def data_info(ev, rows, roots) -> dict:
    """What the numbers cover (review M6): tasks scored, bead-task rows left out for their old bead h, and
    which bead h the scored bead rows carry ("count" = n_violating stored, "old", or "none" scored)."""
    from .data import BEAD_TASKS

    bead_srcs = [s for s in ev.sources if s.rows.size and any(CANON_TASKS[int(t)] in BEAD_TASKS for t in np.unique(s.task))]
    bead_h = ("none" if not bead_srcs else
              "count" if all((s.root / "item_buffer" / "n_violating.npy").exists() for s in bead_srcs) else "old")
    return {"tasks": sorted({r["task"] for r in rows}), "stale_bead_rows_dropped": int(sum(s.n_stale_dropped for s in ev.sources)),
            "bead_h": bead_h, "val_roots": [str(r) for r in roots]}


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
    ap.add_argument("--prior-data", nargs="*", default=None,
                    help="training data roots for the scene-free text prior (rank_acc_text), e.g. data/lang/warmup")
    args = ap.parse_args(argv)
    roots = roots_of(args.val)
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.val)
    ev = Evaluator(Path(args.run), args.ckpt, roots, text_caches=caches, batch=args.batch)
    if args.prior_data:
        ev.prior = TextPrior(roots_of(args.prior_data), ev.cfg.k_prev)
    M, rows = ev.evaluate(args.states_per_task, args.paraphrases)
    out = Path(args.run) / "eval_offline" / Path(args.ckpt).stem
    out.mkdir(parents=True, exist_ok=True)
    data = data_info(ev, rows, roots)
    (out / "metrics.json").write_text(json.dumps({"ckpt": args.ckpt, "step": ev.ckpt_step, "data": data, "metrics": M},
                                                 indent=1))
    (out / "tables.md").write_text(tables_md(M, f"{Path(args.run).name} / {args.ckpt} (step {ev.ckpt_step})")
                                   + f"\nData: {json.dumps(data)}\n")
    figures(M, rows, out)
    examples_md(ev, rows, out / "examples.md")
    ev.text.save_extra()
    print((out / "tables.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
