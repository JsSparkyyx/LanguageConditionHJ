"""LangSampler (VANILLA_PLAN 4.4): multi-source relabeling batches with 3 frames.

Sources are data roots (``<root>/buffer`` StepBuffer + ``<root>/item_buffer``
sidecar, written in lockstep): every warmup shard, plus the run's ring in
rounds mode. Each source keeps its env-side ``RelabelSampler(<root>/item_buffer,
split="train", k_prev=2)`` for the usable-row index and the rule columns,
and opens its StepBuffer arrays read-only (``np.load(mmap_mode="r")``).

Additions over RelabelSampler: a global (source, row) index without terminal
rows; task balance (task uniform, then a row of that task; T1 hosts share T1's
draw); task-balanced pairs from mixed rows; relabel modes paired / random /
fixed; the shuffled-sentence control; rows t, t+1, t-k, 0 and
next_prev_row = next_row - min(k, t_in_ep + 1); fails_soon_recall (7 ticks);
StepBuffer loading (frames [t, t-k, 0] and [t+1, t+1-k, 0]); ref_mask for
probe batches (a measuring tool, never a model input).

Throughput: rows t-k ... t+1 of a sample are one contiguous slice when they do
not cross the array end; frame-0 tokens are cached per (source, episode); a
paired transition is read once; reads run on a thread pool; PrefetchSampler
samples in a background thread (queue depth 2).
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import ensure_item_path

ensure_item_path()

from main.item_envs import language  # noqa: E402
from main.item_envs.relabel import RelabelSampler  # noqa: E402

from .config import CANON_TASKS  # noqa: E402
from .scene_roles import ref_rows  # noqa: E402

ROBOT_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max")
STEP_FIELDS = ROBOT_FIELDS + ("scene_tokens", "scene_pos", "dtheta", "episode_id")
# tasks whose rules are bead rules; their h changed on 2026-10-07 (bead count once violated)
BEAD_TASKS = ("granular_pour", "granular_carry", "clean_pour")


def stable_hash(x) -> int:
    return int(hashlib.md5(str(x).encode()).hexdigest()[:12], 16)


def recall_labels(h_rules: np.ndarray, ep: np.ndarray, t: np.ndarray, n: int, horizon: int) -> np.ndarray:
    """fails_soon_recall[s, j]: h_rules[s + w, j] < 0 for some 0 <= w < horizon,
    with row s + w in the same episode (same episode_id, t_in_ep + w; ring
    modulo). Rows never written read False. -> [C, R] bool."""
    C = h_rules.shape[0]
    written = np.zeros(C, dtype=bool)
    written[: min(n, C)] = True
    neg = (h_rules < 0) & written[:, None]
    slots = np.flatnonzero(written)
    out = np.zeros_like(neg)
    out[slots] = neg[slots]
    for w in range(1, int(horizon)):
        nxt = (slots + w) % C
        ok = written[nxt] & (ep[nxt] == ep[slots]) & (t[nxt] == t[slots] + w)
        out[slots] |= neg[nxt] & ok[:, None]
    return out


class LangSource:
    """One data root: index of usable non-terminal rows + StepBuffer arrays."""

    def __init__(self, root, k_prev: int = 2, recall_horizon: int = 7, name: str | None = None,
                 max_n: int | None = None, drop_stale_bead: bool = True):
        self.root = Path(root).resolve()
        # written before the bead-count h: its bead-task rows hold the old bead h, never train on them
        self.stale_bead = bool(drop_stale_bead) and not (self.root / "item_buffer" / "n_violating.npy").exists()
        self.name = name or self.root.name
        self.k = int(k_prev)
        self.recall_horizon = int(recall_horizon)
        self.rs = RelabelSampler(self.root / "item_buffer", split="train", k_prev=self.k)
        self.max_n = max_n
        if max_n is not None:   # a frozen dataset: rows beyond max_n (collected later) are invisible
            orig = self.rs.buf.refresh

            def capped():
                orig()
                if self.rs.buf.n > max_n:
                    self.rs.buf.n = int(max_n)
                    self.rs.buf.head = int(max_n)
                return self.rs.buf.n

            self.rs.buf.refresh = capped
        self.step: dict[str, np.ndarray] = {}
        self.lines: dict[int, dict] = {}
        self.first_cache: dict[int, tuple] = {}
        self.refresh()

    @property
    def capacity(self) -> int:
        return self.rs.buf.capacity

    def _open_step(self):
        for f in STEP_FIELDS:
            p = self.root / "buffer" / f"{f}.npy"
            if f not in self.step and p.exists():
                self.step[f] = np.load(p, mmap_mode="r")

    def refresh(self) -> int:
        self.rs.refresh()
        self._open_step()
        lj = self.root / "lang_episodes.jsonl"
        if lj.exists():
            for ln in lj.read_text().splitlines():
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if not d.get("skipped"):
                    self.lines[int(d["episode_id"])] = d
        rs, A = self.rs, self.rs.buf.arrays
        mixed = np.zeros(rs._rows.size, dtype=bool)
        mixed[rs._mixed] = True
        keep = ~rs._done                                   # terminal rows: CrossSafe never trains on them
        self.rows = rs._rows[keep].astype(np.int64)
        self.next = rs._next[keep].astype(np.int64)
        self.first = rs._first[keep].astype(np.int64)
        self.mixed = mixed[keep]
        self.eid = np.asarray(A["episode_id"])[self.rows].astype(np.int64)
        self.t = np.asarray(A["t_in_ep"])[self.rows].astype(np.int64)
        tasks, hosts, embs = [], [], []
        if not self.rows.size:
            self.task, self.host, self.emb = np.zeros(0, np.int64), np.zeros(0, object), np.zeros(0, object)
            self.recall = np.zeros((rs.buf.capacity, rs.buf.r_max), dtype=bool)
            return 0
        for e in self.eid:
            rec = rs.records[int(e)]
            tasks.append(rec["task"]), hosts.append(rec.get("host_task")), embs.append(rec["embodiment"])
        self.task = np.array([CANON_TASKS.index(x) for x in tasks], dtype=np.int64) if tasks else np.zeros(0, np.int64)
        self.host = np.array(hosts, dtype=object)
        self.emb = np.array(embs, dtype=object)
        if self.stale_bead:
            fresh = ~np.isin(self.task, [CANON_TASKS.index(x) for x in BEAD_TASKS])
            for f in ("rows", "next", "first", "mixed", "eid", "t", "task", "host", "emb"):
                setattr(self, f, getattr(self, f)[fresh])
        self.recall = recall_labels(np.asarray(A["h_rules"]), np.asarray(A["episode_id"]),
                                    np.asarray(A["t_in_ep"]).astype(np.int64), rs.buf.n, self.recall_horizon)
        # drop cached frame 0 of episodes no longer in the ring
        live = set(int(e) for e in np.unique(self.eid))
        self.first_cache = {e: v for e, v in self.first_cache.items() if e in live}
        return int(self.rows.size)

    # -------------------------------------------------------------- reads ----
    def first_frame(self, eid: int, first_slot: int):
        v = self.first_cache.get(eid)
        if v is None:
            v = (np.array(self.step["scene_tokens"][first_slot]), np.array(self.step["scene_pos"][first_slot]))
            self.first_cache[eid] = v
        return v

    def read(self, i: int) -> dict:
        """Frames and robot fields of index row i (rows t-k ... t+1 in one slice when possible)."""
        S, C, k = self.step, self.capacity, self.k
        slot, nxt, t = int(self.rows[i]), int(self.next[i]), int(self.t[i])
        kk, kn = min(k, t), min(k, t + 1)
        lo = slot - kk
        if lo >= 0 and slot + 1 < C:
            tok = np.array(S["scene_tokens"][lo:slot + 2])
            pos = np.array(S["scene_pos"][lo:slot + 2])
            it, ip, inx, inp = kk, 0, kk + 1, kk + 1 - kn
        else:  # crosses the physical array end: per-row (ring modulo)
            idx = [(slot) % C, (slot - kk) % C, nxt % C, (nxt - kn) % C]
            tok = np.stack([np.array(S["scene_tokens"][j]) for j in idx])
            pos = np.stack([np.array(S["scene_pos"][j]) for j in idx])
            it, ip, inx, inp = 0, 1, 2, 3
        f_tok, f_pos = self.first_frame(int(self.eid[i]), int(self.first[i]))
        out = {
            "scene_tokens": np.stack([tok[it], tok[ip], f_tok]),
            "scene_pos": np.stack([pos[it], pos[ip], f_pos]),
            "scene_tokens_next": np.stack([tok[inx], tok[inp], f_tok]),
            "scene_pos_next": np.stack([pos[inx], pos[inp], f_pos]),
            "dtheta": np.array(S["dtheta"][slot]),
        }
        for f in ROBOT_FIELDS:
            out[f] = np.array(S[f][slot])
            out[f + "_next"] = np.array(S[f][nxt])
        return out

    def shapes(self) -> dict:
        S = self.step
        sh = {"scene_tokens": ((3,) + S["scene_tokens"].shape[1:], np.float16),
              "scene_pos": ((3,) + S["scene_pos"].shape[1:], np.float32),
              "dtheta": (S["dtheta"].shape[1:], np.float32)}
        sh["scene_tokens_next"], sh["scene_pos_next"] = sh["scene_tokens"], sh["scene_pos"]
        for f in ROBOT_FIELDS:
            sh[f] = sh[f + "_next"] = (S[f].shape[1:], S[f].dtype)
        return sh

    def read_into(self, i: int, out: dict, bs: list[int]) -> None:
        """read(i), written straight into batch positions bs of preallocated arrays: one copy per
        frame from the memmap (rows t, t-k, t+1, t+1-k), frame 0 from the RAM cache."""
        S, C, k = self.step, self.capacity, self.k
        slot, nxt, t = int(self.rows[i]), int(self.next[i]), int(self.t[i])
        prev, nprev = (slot - min(k, t)) % C, (nxt - min(k, t + 1)) % C
        f_tok, f_pos = self.first_frame(int(self.eid[i]), int(self.first[i]))
        b0 = bs[0]
        T, P, Tn, Pn = out["scene_tokens"], out["scene_pos"], out["scene_tokens_next"], out["scene_pos_next"]
        T[b0, 0], T[b0, 1], T[b0, 2] = S["scene_tokens"][slot], S["scene_tokens"][prev], f_tok
        P[b0, 0], P[b0, 1], P[b0, 2] = S["scene_pos"][slot], S["scene_pos"][prev], f_pos
        Tn[b0, 0], Tn[b0, 1], Tn[b0, 2] = S["scene_tokens"][nxt], S["scene_tokens"][nprev], f_tok
        Pn[b0, 0], Pn[b0, 1], Pn[b0, 2] = S["scene_pos"][nxt], S["scene_pos"][nprev], f_pos
        out["dtheta"][b0] = S["dtheta"][slot]
        for f in ROBOT_FIELDS:
            out[f][b0] = S[f][slot]
            out[f + "_next"][b0] = S[f][nxt]
        for b in bs[1:]:   # the other member of a pair: same transition, copied once
            for key, arr in out.items():
                arr[b] = arr[b0]

    def rows_of(self, i: int) -> dict:
        slot, t = int(self.rows[i]), int(self.t[i])
        C, k = self.capacity, self.k
        nxt = int(self.next[i])
        return {"row": slot, "next_row": nxt, "prev_row": (slot - min(k, t)) % C, "first_row": int(self.first[i]),
                "next_prev_row": (nxt - min(k, t + 1)) % C}


class LangSampler:
    """sample(B) -> batch dict (Section 4.4 output table)."""

    def __init__(self, roots, cfg, tasks=None, embodiments=None, seed: int = 0, probe: bool = False,
                 n_threads: int = 8, split: str = "train", max_rows: dict | None = None):
        self.cfg = cfg
        self.k = int(cfg.k_prev)
        self.sources = []
        self.max_rows = {str(Path(k).resolve()): int(v) for k, v in (max_rows or {}).items()}
        for r in roots:   # shards still being collected may have no rows (or no arrays) yet
            if self.max_rows and str(Path(r).resolve()) not in self.max_rows:
                continue   # frozen dataset: shards started after the freeze are not part of it
            src = LangSource(r, self.k, int(cfg.recall_horizon_ticks),
                             max_n=self.max_rows.get(str(Path(r).resolve())),
                             drop_stale_bead=getattr(cfg, "drop_stale_bead_rows", True))
            if src.rows.size and "scene_tokens" in src.step:
                self.sources.append(src)
            else:
                print(f"[data] skipping {r}: no usable rows yet")
        self.tasks = None if tasks in (None, "all") else tuple(tasks)
        self.embodiments = None if embodiments in (None, "all") else tuple(embodiments)
        self.rng = np.random.default_rng(seed)
        self.probe = bool(probe)
        self.split = split
        self.bank = language.load_bank()
        self.pool = ThreadPoolExecutor(max_workers=int(n_threads))
        self.stats = {"mixed": 0, "pairs": 0, "harm": 0, "n": 0}
        self.refresh()

    def add_source(self, root) -> None:
        src = LangSource(root, self.k, int(self.cfg.recall_horizon_ticks),
                         drop_stale_bead=getattr(self.cfg, "drop_stale_bead_rows", True))
        if src.rows.size and "scene_tokens" in src.step:
            self.sources.append(src)
            self.refresh(sources=False)
        else:
            print(f"[data] add_source {root}: no usable rows yet (added at the next refresh)")
            self._pending = getattr(self, "_pending", []) + [root]

    def refresh_pending(self) -> None:
        for root in list(getattr(self, "_pending", [])):
            self._pending.remove(root)
            self.add_source(root)

    def refresh(self, sources: bool = True) -> int:
        if sources:
            for s in self.sources:
                s.refresh()
        sid, loc, task, emb, mixed = [], [], [], [], []
        for j, s in enumerate(self.sources):
            n = s.rows.size
            sid.append(np.full(n, j, dtype=np.int64)), loc.append(np.arange(n, dtype=np.int64))
            task.append(s.task), emb.append(s.emb), mixed.append(s.mixed)
        cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
        self.sid, self.loc = cat(sid, np.int64), cat(loc, np.int64)
        self.task_of, self.emb_of, self.mixed_of = cat(task, np.int64), cat(emb, object), cat(mixed, bool)
        ok = np.ones(self.sid.size, dtype=bool)
        if self.tasks is not None:
            ok &= np.isin(self.task_of, [CANON_TASKS.index(t) for t in self.tasks])
        if self.embodiments is not None:
            ok &= np.isin(self.emb_of, list(self.embodiments))
        self.ok = ok
        self.by_task = {int(t): np.flatnonzero(ok & (self.task_of == t)) for t in np.unique(self.task_of[ok])}
        self.mixed_by_task = {t: idx[self.mixed_of[idx]] for t, idx in self.by_task.items()}
        self.mixed_by_task = {t: v for t, v in self.mixed_by_task.items() if v.size}
        self.all_ok = np.flatnonzero(ok)
        # episodes per (source) for the shuffled control
        self.episodes = [(j, int(e)) for j, s in enumerate(self.sources) for e in np.unique(s.eid)]
        return int(self.all_ok.size)

    def __len__(self) -> int:
        return int(self.all_ok.size)

    # ------------------------------------------------------------- choice ----
    def _draw(self, n: int, pool: dict, rng) -> np.ndarray:
        if not pool:
            return np.zeros(0, dtype=np.int64)
        if self.cfg.task_balance:
            keys = sorted(pool)
            tk = rng.integers(len(keys), size=n)
            return np.array([pool[keys[k]][rng.integers(pool[keys[k]].size)] for k in tk], dtype=np.int64)
        allidx = np.concatenate([pool[t] for t in sorted(pool)])
        return allidx[rng.integers(allidx.size, size=n)]

    def _cols(self, s: LangSource, i: int) -> np.ndarray:
        return s.rs._sample_cols[int(s.eid[i])]

    def _fixed_col(self, s: LangSource, i: int) -> int:
        eid = int(s.eid[i])
        cols = self._cols(s, i)
        rec = s.rs.records[eid]
        a = int(rec.get("active_rule", -1))
        if a >= 0 and a in set(cols.tolist()):
            return a
        return int(cols[stable_hash(eid) % cols.size])

    def _pick(self, s: LangSource, i: int, rng, want=None) -> int:
        cols = self._cols(s, i)
        fs = np.asarray(s.rs.buf.arrays["fails_soon"][int(s.rows[i])])[cols]
        harmful, benign = cols[fs], cols[~fs]
        if want is None:
            want = bool(rng.random() < self.cfg.harmful_frac) if (harmful.size and benign.size) else bool(harmful.size)
        pool = harmful if want else benign
        return int(pool[rng.integers(pool.size)])

    def _shuffled_sentence(self, j: int, eid: int, rng) -> str:
        for _ in range(100):
            j2, e2 = self.episodes[int(rng.integers(len(self.episodes)))]
            if (j2, e2) == (j, eid):
                continue
            s2 = self.sources[j2]
            cols = s2.rs._sample_cols.get(e2)
            if cols is None or not cols.size:
                continue
            rec = s2.rs.records[e2]
            rule = rec["rules"][int(cols[rng.integers(cols.size)])]
            return language.sample_sentence(rule, rec, rng, "train", self.bank)
        raise RuntimeError("shuffled control: no other episode with a sampleable rule")

    def plan(self, B: int, rng=None, pair_frac: float | None = None):
        """-> list of (global index, rule col, pair id) of length B."""
        rng = rng or self.rng
        cfg = self.cfg
        pf = cfg.pair_frac if pair_frac is None else pair_frac
        picks = []
        if cfg.relabel == "paired":
            n_pair = int(round(pf * B / 2.0))
            for p, gi in enumerate(self._draw(n_pair, self.mixed_by_task, rng)):
                s, i = self.sources[self.sid[gi]], int(self.loc[gi])
                picks.append((int(gi), self._pick(s, i, rng, True), p))
                picks.append((int(gi), self._pick(s, i, rng, False), p))
        for gi in self._draw(B - len(picks), self.by_task, rng):
            s, i = self.sources[self.sid[gi]], int(self.loc[gi])
            col = self._fixed_col(s, i) if cfg.relabel == "fixed" else self._pick(s, i, rng)
            picks.append((int(gi), col, -1))
        return picks[:B]

    # ------------------------------------------------------------- sample ----
    def sample(self, B: int, rng=None, picks=None, pair_frac: float | None = None) -> dict:
        rng = rng or self.rng
        picks = picks if picks is not None else self.plan(B, rng, pair_frac)
        where = {}
        for b, (gi, _, _) in enumerate(picks):
            where.setdefault(gi, []).append(b)
        src0 = self.sources[self.sid[picks[0][0]]]
        out = {k: np.empty((len(picks),) + shp, dtype=dt) for k, (shp, dt) in src0.shapes().items()}
        list(self.pool.map(lambda kv: self.sources[self.sid[kv[0]]].read_into(int(self.loc[kv[0]]), out, kv[1]),
                           where.items()))
        meta = {k: [] for k in ("sentence", "rule_id", "consequence", "task", "host", "embodiment", "source",
                                "episode_id", "rule_col", "row", "next_row", "prev_row", "first_row", "next_prev_row",
                                "t_in_ep")}
        h, hn, fs, fsr, pair = [], [], [], [], []
        ref = np.zeros((len(picks), 3, out["scene_tokens"].shape[2]), dtype=np.float16) if self.probe else None
        for b, (gi, col, p) in enumerate(picks):
            j, i = int(self.sid[gi]), int(self.loc[gi])
            s = self.sources[j]
            eid = int(s.eid[i])
            rec = s.rs.records[eid]
            rule = rec["rules"][col]
            rows = s.rows_of(i)
            A = s.rs.buf.arrays
            if self.cfg.shuffle_text:
                sent = self._shuffled_sentence(j, eid, rng)
            else:
                sent = language.sample_sentence(rule, rec, rng, "train", self.bank)
            meta["sentence"].append(sent)
            meta["rule_id"].append(rule["rule_id"]), meta["consequence"].append(rule["consequence"])
            meta["task"].append(rec["task"]), meta["host"].append(rec.get("host_task"))
            meta["embodiment"].append(rec["embodiment"]), meta["source"].append(s.name)
            meta["episode_id"].append(eid), meta["rule_col"].append(int(col)), meta["t_in_ep"].append(int(s.t[i]))
            for k2, v in rows.items():
                meta[k2].append(v)
            h.append(float(A["h_rules"][rows["row"], col])), hn.append(float(A["h_rules"][rows["next_row"], col]))
            fs.append(bool(A["fails_soon"][rows["row"], col])), fsr.append(bool(s.recall[rows["row"], col]))
            pair.append(int(p))
            if ref is not None:
                region_rows = (s.lines.get(eid) or {}).get("mask_region_rows") or {}
                rr = ref_rows(rec, rule, region_rows)
                if rr:
                    tm = A["token_masks"]
                    for f, slot in enumerate((rows["row"], rows["prev_row"], rows["first_row"])):
                        ref[b, f] = np.asarray(tm[slot, rr]).max(0).reshape(-1).astype(np.float16) / 255.0
        out.update({"h": np.asarray(h, np.float32), "h_next": np.asarray(hn, np.float32),
                    "fails_soon": np.asarray(fs), "fails_soon_recall": np.asarray(fsr), "pair": np.asarray(pair)})
        for k2, v in meta.items():
            out[k2] = np.asarray(v) if k2 in ("episode_id", "rule_col", "row", "next_row", "prev_row", "first_row",
                                              "next_prev_row", "t_in_ep") else v
        if ref is not None:
            out["ref_mask"] = ref
        n_pairs = int((out["pair"] >= 0).sum() // 2)
        self.stats["pairs"] += n_pairs
        self.stats["mixed"] += sum(1 for p in range(n_pairs)
                                   if out["fails_soon"][out["pair"] == p].any() and not out["fails_soon"][out["pair"] == p].all())
        self.stats["harm"] += int(out["fails_soon"].sum())
        self.stats["n"] += len(picks)
        return out

    def health(self, reset: bool = True) -> dict:
        st = self.stats
        out = {"data/harm_frac": st["harm"] / max(st["n"], 1),
               "data/pair_mixed_frac": st["mixed"] / max(st["pairs"], 1) if st["pairs"] else float("nan")}
        if reset:
            self.stats = {"mixed": 0, "pairs": 0, "harm": 0, "n": 0}
        return out


class PrefetchSampler:
    """Background thread filling a queue (depth 2) with sampler.sample(B)."""

    def __init__(self, sampler: LangSampler, batch_size: int, depth: int = 2, seed: int = 0):
        self.sampler, self.B = sampler, int(batch_size)
        self.depth = int(depth)
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue | None = None
        self.thread = None
        self._stop = threading.Event()
        self.wait_s = 0.0
        self.error = None

    def start(self):
        if self.thread is not None:
            return
        self.q = queue.Queue(maxsize=self.depth)
        self._stop.clear()

        def work():
            try:
                while not self._stop.is_set():
                    b = self.sampler.sample(self.B, self.rng)
                    # the RNG state right after this batch: a resume from it draws
                    # exactly the batch an uninterrupted run would draw next
                    b["_rng_after"] = self.rng.bit_generator.state
                    while not self._stop.is_set():
                        try:
                            self.q.put(b, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:   # surfaced on the next get()
                self.error = exc

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def next(self) -> dict:
        if self.thread is None:
            self.start()
        t0 = time.perf_counter()
        while True:
            if self.error is not None:
                raise self.error
            try:
                b = self.q.get(timeout=1.0)
                break
            except queue.Empty:
                continue
        self.wait_s += time.perf_counter() - t0
        return b

    def stop(self, rewind_to=None):
        """Stop and drain (before collection in rounds mode). rewind_to: the RNG
        state after the last CONSUMED batch, so the drained batches are drawn
        again (an interrupted and an uninterrupted run then match)."""
        if self.thread is None:
            if rewind_to is not None:
                self.set_rng_state(rewind_to)
            return
        self._stop.set()
        while self.thread.is_alive():
            try:
                self.q.get(timeout=0.1)
            except queue.Empty:
                pass
        self.thread.join()
        self.thread, self.q = None, None
        if rewind_to is not None:
            self.set_rng_state(rewind_to)

    def rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, st):
        self.rng.bit_generator.state = st
