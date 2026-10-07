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


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of score for label (rank-based, ties averaged)."""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
        self.sources = [LangSource(r, self.cfg.k_prev, self.cfg.recall_horizon_ticks) for r in val_roots]
        n_tok = self.sources[0].step["scene_tokens"].shape[1]
        self.pe = LangPolicyEncoder(self.cfg, n_tok=n_tok).to(device).eval()
        load_policy(self.pe, ck)
        self.critics = LangTwinCritic(self.cfg).to(device).eval()
        self.critics.load_state_dict(ck["critics"])
        self.text = TextStore(text_caches, self.cfg.text_backend, device,
                              extra_path=self.run / "text_cache_extra.pt", backend_paths=self.cfg.backend_paths,
                              encoder=text_encoder)
        self.bank = language.load_bank()
        self.batch = int(batch)
        self.ckpt_step = int(ck.get("step", 0))

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
                for rule in rec["rules"]:
                    el = {ax: eligible(rule, rec, ax, self.bank) for ax in AXES}
                    if not any(el.values()):
                        continue
                    col = rule["idx"]
                    base = {"src": j, "i": i, "task": rec["task"], "emb": rec["embodiment"], "episode_id": eid,
                            "t": int(s.t[i]), "rule": rule["rule_id"], "col": col, "consequence": rule["consequence"],
                            "referent": referent(rule, rec), "split": rule.get("split"),
                            "fails_soon": bool(A["fails_soon"][slot, col]), "recall": bool(s.recall[slot, col]),
                            "h": float(A["h_rules"][slot, col]), "eligible": el, "q": {}}
                    r_idx = len(rows)
                    rows.append(base)
                    for ax in AXES:
                        if not el[ax]:
                            continue
                        sent = (self._other_sentence(j, eid, rng) if ax == "shuffled"
                                else axis_sentence(rule, rec, ax, (eid, int(s.t[i]), col), self.bank))
                        items.append((k, sent))
                        meta.append((r_idx, ax))
                    # base sentence for auroc_base / rank_acc_base (another episode's)
                    items.append((k, self._other_sentence(j, eid, rng)))
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


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = lambda r: [("pooled",), ("task", r["task"]), ("cons", r["consequence"])]  # noqa: E731
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
            for g in groups(hr):
                d = acc[(kind,) + g]
                d["rank"].append(qh < qb)
                if qh0 is not None:
                    d["rank_base"].append(qh0 < qb0)
                if same_cons and not same_ref:
                    d["rank_ref"].append(qh < qb)
                if same_ref and not same_cons:
                    d["rank_cons"].append(qh < qb)
        for key, d in acc.items():
            name = "/".join([ax] + list(key))
            m = {"n_pairs": len(d["rank"]), "rank_acc": float(np.mean(d["rank"]))}
            if d["rank_base"]:
                m["rank_acc_base"] = float(np.mean(d["rank_base"]))
                m["rank_acc_minus_base"] = m["rank_acc"] - m["rank_acc_base"]
            if d["rank_ref"]:
                m["rank_acc_referent"], m["n_referent"] = float(np.mean(d["rank_ref"])), len(d["rank_ref"])
            if d["rank_cons"]:
                m["rank_acc_consequence"], m["n_consequence"] = float(np.mean(d["rank_cons"])), len(d["rank_cons"])
            out[name] = m
        # per (state, rule) metrics
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
            pred = q < 0
            tp = float((pred & rec).sum())
            m = out.setdefault(f"{ax}/state/{gname}", {})
            m.update({"n": len(rr), "auroc": auroc(-q, fs), "auroc_base": auroc(-qb, fs),
                      "prec_at_0": tp / max(float(pred.sum()), 1e-9), "rec_at_0": tp / max(float(rec.sum()), 1e-9),
                      "v_le_h": float((q <= h + V_LE_H_EPS).mean()), "q_mean": float(q.mean()), "h_mean": float(h.mean())})
            if ax == "train":
                para = [np.std(r["q"]["para"]) for r in rr if len(r["q"].get("para", [])) >= 2]
                if para:
                    m["para_std"] = float(np.mean(para))
                    across = []
                    by_state = defaultdict(list)
                    for r in rr:
                        by_state[(r["src"], r["i"])].append(r["q"]["train"])
                    across = [np.std(v) for v in by_state.values() if len(v) >= 2]
                    if across:
                        m["rule_std"] = float(np.mean(across))
                        m["para_std_ratio"] = m["para_std"] / max(m["rule_std"], 1e-9)
    return out


def tables_md(M: dict, title: str) -> str:
    L = [f"# Offline language probe: {title}", "",
         "rank_acc = P[Q(harmful rule) < Q(benign rule)] over symmetric pairs at the same state (chance 0.5).", "",
         "| axis | pooled | " + " | ".join(f"T{i + 1}" for i in range(len(CANON_TASKS))) + " | referent | consequence |",
         "|---|---|" + "---|" * len(CANON_TASKS) + "---|---|"]
    for ax in AXES:
        p = M.get(f"{ax}/symmetric/pooled", {})
        cells = [f"{p['rank_acc']:.3f} ({p['n_pairs']})" if p else "–"]
        for t in CANON_TASKS:
            m = M.get(f"{ax}/symmetric/task/{t}", {})
            cells.append(f"{m['rank_acc']:.2f} ({m['n_pairs']})" if m else "–")
        cells.append(f"{p['rank_acc_referent']:.3f} ({p['n_referent']})" if p.get("rank_acc_referent") is not None else "–")
        cells.append(f"{p['rank_acc_consequence']:.3f} ({p['n_consequence']})" if p.get("rank_acc_consequence") is not None else "–")
        L.append(f"| {ax} | " + " | ".join(cells) + " |")
    L += ["", "Mixed pairs (only one member axis-eligible): rank_acc - rank_acc_base (r1's sentence from another episode).", "",
          "| axis | n pairs | rank_acc | rank_acc_base | difference |", "|---|---|---|---|---|"]
    for ax in ("heldout_word", "heldout_item"):
        m = M.get(f"{ax}/mixed/pooled")
        if m:
            L.append(f"| {ax} | {m['n_pairs']} | {m['rank_acc']:.3f} | {m.get('rank_acc_base', float('nan')):.3f} | "
                     f"{m.get('rank_acc_minus_base', float('nan')):+.3f} |")
    L += ["", "Per (state, rule): AUROC of -Q for fails_soon (base: other-episode sentences), precision / recall of Q < 0 "
          "against fails_soon_recall (7 ticks), v_le_h = P[Q <= h + 0.05].", "",
          "| axis | n | auroc | auroc_base | prec@0 | rec@0 | v_le_h | para_std | para_std_ratio |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ax in AXES:
        m = M.get(f"{ax}/state/pooled")
        if m:
            L.append(f"| {ax} | {m['n']} | {m['auroc']:.3f} | {m['auroc_base']:.3f} | {m['prec_at_0']:.3f} | "
                     f"{m['rec_at_0']:.3f} | {m['v_le_h']:.3f} | {m.get('para_std', float('nan')):.3f} | "
                     f"{m.get('para_std_ratio', float('nan')):.3f} |")
    L += ["", "Per task (train axis):", "", "| task | n | auroc | auroc_base | rank_acc | v_le_h |", "|---|---|---|---|---|---|"]
    for t in CANON_TASKS:
        m = M.get(f"train/state/task/{t}")
        p = M.get(f"train/symmetric/task/{t}", {})
        if m:
            L.append(f"| {t} | {m['n']} | {m['auroc']:.3f} | {m['auroc_base']:.3f} | "
                     f"{p.get('rank_acc', float('nan')):.3f} | {m['v_le_h']:.3f} |")
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
    args = ap.parse_args(argv)
    roots = roots_of(args.val)
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.val)
    ev = Evaluator(Path(args.run), args.ckpt, roots, text_caches=caches, batch=args.batch)
    M, rows = ev.evaluate(args.states_per_task, args.paraphrases)
    out = Path(args.run) / "eval_offline" / Path(args.ckpt).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"ckpt": args.ckpt, "step": ev.ckpt_step, "metrics": M}, indent=1))
    (out / "tables.md").write_text(tables_md(M, f"{Path(args.run).name} / {args.ckpt} (step {ev.ckpt_step})"))
    figures(M, rows, out)
    examples_md(ev, rows, out / "examples.md")
    ev.text.save_extra()
    print((out / "tables.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
