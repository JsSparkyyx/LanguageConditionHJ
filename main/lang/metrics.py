"""Training metrics (VANILLA_PLAN 8.1), probes (8.2) and round metrics (8.3).

``batch_metrics``: the per-step scalars of a training batch (pair rank
accuracy, precision / recall against h_next and fails_soon_recall, splits per
task and consequence).

``geo_attention_weights``: GeoCrossAttention returns no weights and trunk.py
must not be edited, so probe mode recomputes them from the module's own
q_proj, k_proj and bias_mlp with the same logits and masking as its forward
(tests check that these weights reproduce its output).

``Prober``: a fixed probe set of validation samples with ref_mask; text
sensitivity, paraphrase spread, attention enrichment on the named referent,
scene-delta ratio, word-attention entropy, FiLM sizes, and an examples page.

``LangRoundMetrics``: per filter episode of a round (Phase 4).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PAIR_TARGET_EPS = 0.05


# ---------------------------------------------------------------------------
# per-step training metrics (8.1)
# ---------------------------------------------------------------------------


def _f(x) -> float:
    return float(x.detach().float().mean()) if torch.is_tensor(x) else float(np.mean(x))


def pair_stats(qmin: np.ndarray, y: np.ndarray, fails_soon: np.ndarray, pair: np.ndarray) -> dict:
    """In-batch pairs (same transition, one harmful + one benign rule)."""
    rank, tgt, gap = [], [], []
    for p in np.unique(pair[pair >= 0]):
        k = np.flatnonzero(pair == p)
        if k.size != 2:
            continue
        fs = fails_soon[k]
        if fs[0] != fs[1]:
            harm, ben = (k[0], k[1]) if fs[0] else (k[1], k[0])
            rank.append(qmin[harm] < qmin[ben])
            gap.append(qmin[ben] - qmin[harm])
        if abs(y[k[0]] - y[k[1]]) > PAIR_TARGET_EPS:
            tgt.append(np.sign(qmin[k[0]] - qmin[k[1]]) == np.sign(y[k[0]] - y[k[1]]))
    nan = float("nan")
    return {"lang/pair_rank_acc": float(np.mean(rank)) if rank else nan,
            "lang/pair_target_rank_acc": float(np.mean(tgt)) if tgt else nan,
            "lang/pair_q_gap": float(np.mean(gap)) if gap else nan,
            "lang/n_pairs": len(rank), "lang/n_target_pairs": len(tgt)}


def prec_rec(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    tp = float((pred & true).sum())
    return tp / max(float(pred.sum()), 1e-9), tp / max(float(true.sum()), 1e-9)


def batch_metrics(b: dict, q1, q2, y, h) -> dict:
    """Scalars of one batch: q / h / gap, precision / recall, pair metrics, splits."""
    qmin = torch.minimum(q1, q2).detach().float().cpu().numpy()
    yv = y.detach().float().cpu().numpy()
    hv = h.detach().float().cpu().numpy()
    hn = np.asarray(b["h_next"], dtype=np.float32)
    out = {"q/mean": float(qmin.mean()), "h/mean": float(hv.mean()), "h/min": float(hv.min()),
           "gap/mean": float((hv - qmin).mean()), "y/mean": float(yv.mean())}
    pred = qmin < 0
    out["q/precision"], out["q/recall"] = prec_rec(pred, hn < 0)
    out["q/precision_soon"], out["q/recall_soon"] = prec_rec(pred, np.asarray(b["fails_soon_recall"], bool))
    out.update(pair_stats(qmin, yv, np.asarray(b["fails_soon"], bool), np.asarray(b["pair"])))
    tasks = np.asarray(b["task"])
    cons = np.asarray(b["consequence"])
    for t in np.unique(tasks):
        m = tasks == t
        out[f"by_task/{t}/q"] = float(qmin[m].mean())
        out[f"by_task/{t}/h"] = float(hv[m].mean())
        out[f"by_task/{t}/gap"] = float((hv[m] - qmin[m]).mean())
        ps = pair_stats(qmin[m], yv[m], np.asarray(b["fails_soon"], bool)[m], np.asarray(b["pair"])[m])
        out[f"by_task/{t}/rank_acc"] = ps["lang/pair_rank_acc"]
    for c in np.unique(cons):
        m = cons == c
        out[f"by_cons/{c}/q"] = float(qmin[m].mean())
        out[f"by_cons/{c}/h"] = float(hv[m].mean())
        out[f"by_cons/{c}/gap"] = float((hv[m] - qmin[m]).mean())
    return out


def grad_norm(params) -> float:
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.norm(torch.stack(gs))) if gs else 0.0


# ---------------------------------------------------------------------------
# attention recomputation (8.2)
# ---------------------------------------------------------------------------


def geo_attention_weights(geo, body, pos, scene, scene_pos, scene_mask):
    """A [B, H, N, M] of a GeoCrossAttention for its inputs (same logits and
    masking as GeoCrossAttention.forward)."""
    B, N, C = body.shape
    M, H = scene.shape[1], geo.heads
    q = geo.q_proj(body).view(B, N, H, geo.dh).transpose(1, 2)
    k = geo.k_proj(scene).view(B, M, H, geo.dh).transpose(1, 2)
    logits = (q @ k.transpose(-1, -2)) * geo.dh ** -0.5
    if geo.geo:
        rel = scene_pos[:, None, :, :] - pos[:, :, None, :]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        bias_in = torch.cat([dist.log(), rel / dist], dim=-1)
        logits = logits + geo.bias_mlp(bias_in).permute(0, 3, 1, 2)
    if scene_mask is not None:
        logits = logits.masked_fill(~scene_mask[:, None, None, :], torch.finfo(logits.dtype).min)
    return logits.softmax(dim=-1)


def geo_output_from_weights(geo, A, scene, pos, scene_pos):
    """GeoCrossAttention's output rebuilt from A (the test of the recomputation)."""
    B, H, N, M = A.shape
    v = geo.v_proj(scene).view(B, M, H, geo.dh).transpose(1, 2)
    val = (A @ v).transpose(1, 2).reshape(B, N, H * geo.dh)
    if not geo.geo:
        return geo.out_proj(val)
    rel = scene_pos[:, None, :, :] - pos[:, :, None, :]
    dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    direction = torch.einsum("bhnm,bnmc->bhnc", A, rel / dist)
    direction = direction.transpose(1, 2).reshape(B, N, H * 3) if geo.dir_per_head else direction.mean(1)
    return geo.out_proj(torch.cat([val, direction], dim=-1))


def enrichment(A_tok: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """A_tok [B, M] attention over scene tokens (sums to 1), ref [B, M] in
    [0, 1] referent coverage -> [B] mass on the referent / its token share
    (NaN where the referent is not visible)."""
    share = ref.sum(-1) / ref.shape[-1]
    mass = (A_tok * ref).sum(-1)
    out = mass / share.clamp_min(1e-9)
    return torch.where(share > 0, out, torch.full_like(out, float("nan")))


def entropy(p: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    e = -(p.clamp_min(1e-12).log() * p)
    if mask is not None:
        e = e * mask
    return e.sum(-1)


# ---------------------------------------------------------------------------
# probes (8.2)
# ---------------------------------------------------------------------------


class Prober:
    """Fixed probe set (n samples, ref_mask) from the validation sources.

    q_fn(batch, sentences, probe) -> (qmin [B], enc, stores) evaluates the
    live model at the logged action for the given sentences (the trainer's
    ``probe_forward``)."""

    def __init__(self, sampler, n: int = 512, seed: int = 1234, chunk: int = 32):
        from main.item_envs import language

        self.language = language
        self.sampler = sampler
        self.chunk = int(chunk)
        rng = np.random.default_rng(seed)
        picks = sampler.plan(int(n), rng, pair_frac=0.0)
        self.batch = sampler.sample(int(n), rng, picks=picks)
        self.rng_seed = seed
        self.n = int(n)

    def _chunks(self):
        for a in range(0, self.n, self.chunk):
            yield a, min(self.n, a + self.chunk)

    def _sub(self, a, z):
        return {k: (v[a:z] if isinstance(v, (np.ndarray, list)) else v) for k, v in self.batch.items()}

    def _record(self, k):
        b = self.batch
        s = next(x for x in self.sampler.sources if x.name == b["source"][k])
        return s.rs.records[int(b["episode_id"][k])]

    def _other_sentence(self, k, rng):
        b = self.batch
        j = next(i for i, x in enumerate(self.sampler.sources) if x.name == b["source"][k])
        return self.sampler._shuffled_sentence(j, int(b["episode_id"][k]), rng)

    @torch.no_grad()
    def run(self, q_fn, out_dir: Path | None = None, step: int = 0, n_examples: int = 8) -> dict:
        lang = self.language
        rng = np.random.default_rng(self.rng_seed + 1)
        b = self.batch
        res = defaultdict(list)
        film = defaultdict(list)
        for a, z in self._chunks():
            sub = self._sub(a, z)
            sents = list(sub["sentence"])
            q, enc, stores = q_fn(sub, sents, True)
            shuf = [self._other_sentence(a + i, rng) for i in range(z - a)]
            q_s, _, _ = q_fn(sub, shuf, False)
            res["dq"].append((q - q_s).abs().cpu())
            res["absq"].append(q.abs().cpu())
            ref = torch.as_tensor(np.asarray(sub["ref_mask"], dtype=np.float32)).flatten(1).to(q.device)  # [B, 3600]
            # stores: the POLICY ENCODER's probe dicts in module order (scene_reads_words
            # when present, then the trunk blocks); the last one is the last trunk
            # block, whose "geo_A" probe_forward adds (geo_attention_weights)
            last = stores[-1] if stores else {}
            if "geo_A" in last:
                A = last["geo_A"].mean(1)                                     # [B, N, M], heads averaged
                m = enc.mask.float()
                A_tok = (A * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
                res["body_enrich"].append(enrichment(A_tok, ref).cpu())
            if "query_scene_attn" in last:
                res["query_enrich"].append(enrichment(last["query_scene_attn"].mean(1), ref).cpu())
            if "body_word_attn" in last:
                wm = enc.word_mask.float()
                Aw = last["body_word_attn"]                                    # [B, 22, L]
                ent = entropy(Aw, wm[:, None, :])
                m = enc.mask.float()
                res["word_entropy"].append(((ent * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
            srw = next((st for st in stores if "scene_delta_norm" in st), None)
            if srw is not None:
                dn = srw["scene_delta_norm"]                                   # [B, 3600]
                w_ref = (ref > 0).float()
                on = (dn * w_ref).sum(1) / w_ref.sum(1).clamp_min(1e-9)
                off = (dn * (1 - w_ref)).sum(1) / (1 - w_ref).sum(1).clamp_min(1e-9)
                r = on / off.clamp_min(1e-12)
                res["scene_delta_ratio"].append(torch.where(w_ref.sum(1) > 0, r, torch.full_like(r, float("nan"))).cpu())
            for bi, st in enumerate(x for x in stores if "film_gamma_rms" in x):
                film[f"film/gamma_rms_{bi}"].append(float(st["film_gamma_rms"]))
                film[f"film/beta_rms_{bi}"].append(float(st["film_beta_rms"]))
        out = {"lang/text_sens": float(torch.cat(res["dq"]).mean() / torch.cat(res["absq"]).mean().clamp_min(1e-9))}
        for key, name in (("body_enrich", "attn/body_ref_enrich"), ("query_enrich", "attn/query_ref_enrich"),
                          ("word_entropy", "attn/word_entropy"), ("scene_delta_ratio", "attn/scene_delta_ratio")):
            if res[key]:
                v = torch.cat(res[key])
                out[name] = float(v[torch.isfinite(v)].mean()) if torch.isfinite(v).any() else float("nan")
        out.update({k: float(np.mean(v)) for k, v in film.items()})
        out.update(self._paraphrases(q_fn, rng))
        if out_dir is not None:
            self.examples(q_fn, Path(out_dir) / f"step{step}_examples.md", n_examples)
        return out

    def _paraphrases(self, q_fn, rng, n_states: int = 64, n_para: int = 3) -> dict:
        """para_std: std of Q over 3 training paraphrases of the same rule;
        para_std_ratio: para_std / std over the different sampleable rules of
        the same state. Both average over the SAME states: those with >= 2
        sampleable rules and >= 2 paraphrases (review M8)."""
        lang = self.language
        b = self.batch
        para, across = [], []
        for k in range(min(n_states, self.n)):
            rec = self._record(k)
            rule = rec["rules"][int(b["rule_col"][k])]
            sents = lang.sentences(rule, rec, "train")
            rules = [r for r in rec["rules"] if r.get("sampleable")]
            if len(sents) >= 2 and len(rules) >= 2:
                pick = [sents[i] for i in rng.choice(len(sents), size=min(n_para, len(sents)), replace=False)]
                sub = {kk: (v[k:k + 1].repeat(len(pick), 0) if isinstance(v, np.ndarray) else [v[k]] * len(pick))
                       for kk, v in b.items()}
                q, _, _ = q_fn(sub, pick, False)
                para.append(float(q.std(unbiased=False)))
                ss = [lang.sample_sentence(r, rec, rng, "train") for r in rules]
                sub = {kk: (v[k:k + 1].repeat(len(ss), 0) if isinstance(v, np.ndarray) else [v[k]] * len(ss))
                       for kk, v in b.items()}
                q, _, _ = q_fn(sub, ss, False)
                across.append(float(q.std(unbiased=False)))
        ps = float(np.mean(para)) if para else float("nan")
        ac = float(np.mean(across)) if across else float("nan")
        return {"lang/para_std": ps, "lang/rule_std": ac, "lang/para_std_ratio": ps / ac if ac and ac == ac else float("nan")}

    def examples(self, q_fn, path: Path, n: int = 8) -> None:
        """For n fixed validation states: every sampleable rule's sentence with Q, h and fails_soon."""
        lang = self.language
        b = self.batch
        rng = np.random.default_rng(7)
        lines = [f"# Probe examples", ""]
        for k in range(min(n, self.n)):
            rec = self._record(k)
            s = next(x for x in self.sampler.sources if x.name == b["source"][k])
            A = s.rs.buf.arrays
            row = int(b["row"][k])
            rules = [r for r in rec["rules"] if r.get("sampleable")]
            ss = [lang.sample_sentence(r, rec, rng, "train") for r in rules]
            sub = {kk: (v[k:k + 1].repeat(len(ss), 0) if isinstance(v, np.ndarray) else [v[k]] * len(ss))
                   for kk, v in b.items()}
            q, _, _ = q_fn(sub, ss, False)
            lines += [f"## {b['task'][k]} {b['host'][k] or ''} {b['embodiment'][k]} episode {b['episode_id'][k]} "
                      f"t={b['t_in_ep'][k]}", "", "| rule | sentence | Q | h | fails_soon |", "|---|---|---|---|---|"]
            for r, sent, qv in zip(rules, ss, q.cpu().numpy()):
                lines.append(f"| {r['rule_id']} | {sent} | {qv:+.3f} | {float(A['h_rules'][row, r['idx']]):+.3f} | "
                             f"{bool(A['fails_soon'][row, r['idx']])} |")
            lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# round metrics (8.3)
# ---------------------------------------------------------------------------


class LangRoundMetrics:
    """Per filter episode of a round, keyed by (round, task, host, embodiment,
    consequence, designation). Writes <run>/eval/rounds.jsonl and heatmaps."""

    KEYS = ("status", "task_success", "harm_named", "harm_named_frac", "harm_other_any", "intervention_rate",
            "intervened_tick_frac", "any_intervention", "mode_switches", "v_le_h_frac", "hq_corr", "first_harm_t",
            "first_intervene_t", "realized_cmd_ratio")

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "eval"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        p = self.dir / "rounds.jsonl"
        if p.exists():   # a resumed run: heatmaps and trends cover every round, not only this process's
            for ln in p.read_text().splitlines():
                try:
                    self.rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue   # a line cut by a kill

    def keep(self, pred) -> None:
        """Drop rows of episodes a resume collects again (rewrites rounds.jsonl atomically)."""
        self.rows = [r for r in self.rows if pred(r)]
        p, tmp = self.dir / "rounds.jsonl", self.dir / "rounds.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in self.rows))
        tmp.replace(p)

    @staticmethod
    def episode_row(round_idx: int, line: dict, trace: dict) -> dict:
        h = np.asarray(trace.get("h_named", []), dtype=np.float64)
        Q = np.asarray(trace.get("V", []), dtype=np.float64)   # the controller stores Q(s, a_nom) as "V" (review M1)
        other = np.asarray(trace.get("h_other_min", []), dtype=np.float64)
        inter = np.asarray(trace.get("intervened", []), dtype=bool)   # one entry per DECISION (3-tick holds)
        src = np.asarray(trace.get("action_source", []), dtype=np.int64)  # one entry per tick (1 = actor)
        t = np.asarray(trace.get("t", []), dtype=np.float64)
        tq = np.asarray(trace.get("V_t", []), dtype=np.float64)
        harm = h.size > 0 and bool((h < 0).any())
        # Q is sampled at hook times; align to the nearest earlier capture
        hq = np.interp(tq, t, h) if (tq.size and t.size == h.size and h.size) else np.zeros(0)
        corr = float(np.corrcoef(hq, Q)[0, 1]) if (Q.size > 2 and np.std(Q) > 0 and np.std(hq) > 0) else float("nan")
        row = {
            "round": int(round_idx), "task": line["task"], "host": line.get("host"), "embodiment": line["embodiment"],
            "consequence": line.get("named_consequence"), "designation": line.get("designation"),
            "episode_id": line["episode_id"], "status": line.get("status"), "task_success": bool(line.get("success")),
            "harm_named": harm, "harm_named_frac": float((h < 0).mean()) if h.size else float("nan"),
            "harm_other_any": bool(other.size and (other < 0).any()),
            "intervention_rate": float(inter.mean()) if inter.size else 0.0,
            "intervened_tick_frac": float((src == 1).mean()) if src.size else float("nan"),   # review M7
            "any_intervention": bool(inter.any()), "mode_switches": int(trace.get("mode_switches", 0) or 0),
            "v_le_h_frac": float((Q <= hq + 0.05).mean()) if Q.size and hq.size else float("nan"),
            "hq_corr": corr,
            "first_harm_t": float(t[np.argmax(h < 0)]) if harm and t.size == h.size else float("nan"),
            "first_intervene_t": float(tq[np.argmax(inter)]) if inter.any() and tq.size == inter.size else float("nan"),
            "realized_cmd_ratio": float(trace.get("mean_realized_ratio", float("nan"))),
        }
        return row

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.dir / "rounds.jsonl", "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    @staticmethod
    def ran(r: dict) -> bool:
        """The episode ran: ok, or cut at the tick cap. An exception episode (a failed grasp, an error at the
        first tick) has no outcome and would read as a clean, quiet filter run (independent review)."""
        return r.get("status") in (None, "ok", "timeout")

    def round_summary(self, round_idx: int) -> dict:
        all_rows = [r for r in self.rows if r["round"] == round_idx]
        rows = [r for r in all_rows if self.ran(r)]
        out = {}
        for d in ("harm", "benign", "all"):   # all = every filter episode, whatever its designation
            sel = rows if d == "all" else [r for r in rows if r["designation"] == d]
            if not sel:
                continue
            for k in ("harm_named", "any_intervention", "intervention_rate", "intervened_tick_frac", "task_success",
                      "v_le_h_frac"):
                vals = [float(r[k]) for r in sel if r.get(k) is not None and r[k] == r[k]]   # older rows lack new keys
                out[f"round_overview/{d}/{k}"] = float(np.mean(vals)) if vals else float("nan")
        out["round_overview/n_filter_episodes"] = len(rows)
        out["round_overview/n_not_run"] = len(all_rows) - len(rows)
        return out

    def heatmaps(self, round_idx: int) -> list[Path]:
        """tasks x (embodiment x designation) heatmap per metric + trend plot."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return []
        from .config import CANON_EMBODIMENTS, CANON_TASKS

        # columns for the embodiments the run collects on (a one-embodiment run has no empty columns)
        embs = [e for e in CANON_EMBODIMENTS if any(r.get("embodiment") == e for r in self.rows)] or list(CANON_EMBODIMENTS)
        cols = [(e, d) for e in embs for d in ("harm", "benign", "all")]
        paths = []
        for metric in ("harm_named", "any_intervention", "intervention_rate", "task_success"):
            M = np.full((len(CANON_TASKS), len(cols)), np.nan)
            for i, t in enumerate(CANON_TASKS):
                for j, (e, d) in enumerate(cols):
                    v = [float(r[metric]) for r in self.rows if self.ran(r)
                         and r["task"] == t and r["embodiment"] == e and (d == "all" or r["designation"] == d)]
                    if v:
                        M[i, j] = np.mean(v)
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(cols)), [f"{e[:5]}\n{d}" for e, d in cols], fontsize=7)
            ax.set_yticks(range(len(CANON_TASKS)), CANON_TASKS, fontsize=7)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if M[i, j] == M[i, j]:
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
            ax.set_title(f"{metric} (all rounds to {round_idx})", fontsize=8)
            fig.colorbar(im, ax=ax)
            d = self.dir / metric
            d.mkdir(exist_ok=True)
            p = d / f"round{round_idx:03d}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=110)
            plt.close(fig)
            paths.append(p)
        paths.append(self.trend(plt))
        return paths

    def trend(self, plt) -> Path:
        """eval/trend.png: per round, harm_named on harm-designated runs (should fall), any_intervention
        on benign-designated runs (should stay low), the same two over ALL filter runs (dashed, no
        grouping), task_success and v_le_h_frac on all filter runs."""
        rounds = sorted({int(r["round"]) for r in self.rows})
        series = (("harm_named", "harm", "harmed (harm runs)"), ("any_intervention", "benign", "intervened (benign runs)"),
                  ("harm_named", None, "harmed (all runs)"), ("any_intervention", None, "intervened (all runs)"),
                  ("task_success", None, "task success"), ("v_le_h_frac", None, "Q <= h + 0.05"))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for key, des, label in series:
            ys = []
            for rd in rounds:
                v = [float(r[key]) for r in self.rows if self.ran(r) and int(r["round"]) == rd
                     and (des is None or r["designation"] == des) and r.get(key) is not None and r[key] == r[key]]
                ys.append(np.mean(v) if v else np.nan)
            ax.plot(rounds, ys, marker=".", label=label, ls="--" if "(all runs)" in label else "-")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("round")
        ax.set_title(f"filter episodes per round ({len(self.rows)} so far)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = self.dir / "trend.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        return p


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Rate with a Wilson 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - hw), min(1.0, c + hw)
