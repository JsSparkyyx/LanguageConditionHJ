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

Version-2 shards (FUSION_PLAN 3.2) add ``<root>/neck_buffer`` (NeckBuffer, read
only). With ``neck=True`` a batch also carries the neck rows of its 5 frames (t,
t-k, 0, t+1, t+1-k): each unique (source, ring row) once in ``neck``, indexed by
``neck_idx`` [B, 5]; frame 0 from a per-(source, episode) RAM cache. The neck path
draws no random numbers. ``episodes_per_task = N`` (E6 scene-count curve) keeps
the first N episodes of each (task, embodiment) in a fixed hash order.
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
from .kv_buffer import KvBuffer  # noqa: E402
from .neck_buffer import NeckBuffer  # noqa: E402
from .scene_roles import ref_rows  # noqa: E402

ROBOT_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max")
STEP_FIELDS = ROBOT_FIELDS + ("scene_tokens", "scene_pos", "dtheta", "episode_id")
# the 5 frames of a sample in neck_idx column order (t, t-k, 0, t+1, t+1-k): keys of LangSource.rows_of
NECK_FRAMES = ("row", "prev_row", "first_row", "next_row", "next_prev_row")
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


def _quat_rotate(q, v) -> np.ndarray:
    """Rotate v by the unit quaternion q = [w, x, y, z]."""
    w, u = float(q[0]), np.asarray(q[1:4], np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


ZONE_RULES = ("zone_exclude", "zone_include")


def ref_centers(A, rec: dict, rule: dict, slots) -> tuple[np.ndarray, np.ndarray]:
    """World centre of the rule's subject object at each of the frame rows `slots` (V3 oracle tag, review round 3
    E2): item_pose position + the rotated item_obb centre offset. valid is False for rules without a subject
    object (beads), for zone rules (their subject is the pushed box, but the sentence names the zone: audit
    2026-10-08), and for rows where the item is not valid."""
    c, v = np.zeros((len(slots), 3), np.float32), np.zeros(len(slots), bool)
    subj = rule.get("subject") or {}
    if "object" not in subj or rule.get("consequence") in ZONE_RULES:
        return c, v
    k = int(rec["objects"][subj["object"]]["id"])   # record object id = sidecar item row (checked on the data)
    for f, slot in enumerate(slots):
        if bool(A["item_valid"][slot, k]):
            pose, obb = np.asarray(A["item_pose"][slot, k], np.float64), np.asarray(A["item_obb"][slot, k], np.float64)
            c[f], v[f] = pose[:3] + _quat_rotate(pose[3:7], obb[:3]), True
    return c, v


def named_centers(A, rec: dict, rule: dict, slots) -> tuple[np.ndarray, np.ndarray]:
    """V4 targets (decision 5b): the named item's world centre at each frame row: the subject object's centre
    (ref_centers), or for a rule on ONE region (zone_exclude / zone_include / exclude_particles) the region's
    centre, the translation of its region_T (region rows follow the record's regions list). Invalid for bead-set
    and multi-region rules, and for zone_include: its region is the table, whose frame sits at z = 0 (the top is
    at 0.74 m), and the whole table is no place to localise (audit 2026-10-08)."""
    c, v = ref_centers(A, rec, rule, slots)
    if rule.get("consequence") not in ZONE_RULES and (v.any() or "object" in (rule.get("subject") or {})):
        return c, v
    ref = rule.get("reference") or {}
    if rule["consequence"] in ("zone_exclude", "exclude_particles") and "region" in ref:
        k = int(ref["region"])
        for f, slot in enumerate(slots):
            T = np.asarray(A["region_T"][slot, k], np.float64)
            if np.isfinite(T).all() and abs(T[3, 3] - 1.0) < 1e-6:
                c[f], v[f] = T[:3, 3], True
    return c, v


def bead_h_of(sources) -> str:
    """Which bead h the bead-task rows of these sources carry: "count" (n_violating stored, 2026-10-07 on),
    "old" (soft-min / plain clip, only when drop_stale_bead_rows is off), "mixed", or "none" (no bead rows)."""
    kinds = set()
    for s in sources:
        if s.rows.size and any(CANON_TASKS[int(t)] in BEAD_TASKS for t in np.unique(s.task)):
            kinds.add("old" if s.old_bead_h else "count")
    return "none" if not kinds else kinds.pop() if len(kinds) == 1 else "mixed"


def unique_source_names(sources) -> None:
    """Batches name their source (meta["source"]) and the probes look it up by that name: two roots with one
    folder name (val/shard_g3 and val_grounding/shard_g3) would resolve to the first. A clashing name becomes
    parent/name; names that do not clash stay as they are (the T1 rounds run's prober hit this, 2026-10-13)."""
    for _ in range(4):
        count: dict[str, int] = {}
        for s in sources:
            count[s.name] = count.get(s.name, 0) + 1
        clash = [s for s in sources if count[s.name] > 1]
        if not clash:
            return
        for s in clash:
            depth = s.name.count("/") + 2
            s.name = "/".join(s.root.parts[-depth:])
    raise ValueError(f"source names still clash: {[s.name for s in sources]}")


class LangSource:
    """One data root: index of usable non-terminal rows + StepBuffer arrays (+ the neck sidecar of a version-2
    shard, opened on first use)."""

    def __init__(self, root, k_prev: int = 2, recall_horizon: int = 7, name: str | None = None,
                 max_n: int | None = None, drop_stale_bead: bool = True):
        self.root = Path(root).resolve()
        # written before the bead-count h: its bead-task rows hold the old bead h, never train on them
        self.old_bead_h = not (self.root / "item_buffer" / "n_violating.npy").exists()
        self.stale_bead = bool(drop_stale_bead) and self.old_bead_h
        self.n_stale_dropped = 0
        self.name = name or self.root.name
        self.k = int(k_prev)
        self.recall_horizon = int(recall_horizon)
        self.rs = RelabelSampler(self.root / "item_buffer", split="train", k_prev=self.k)
        self.max_n = max_n
        self.step: dict[str, np.ndarray] = {}
        self.lines: dict[int, dict] = {}
        orig = self.rs.buf.refresh

        def capped():   # applied before RelabelSampler indexes the rows
            orig()
            cap = self._row_cap()
            if cap is not None and self.rs.buf.n > cap:
                self.rs.buf.n = int(cap)
                self.rs.buf.head = int(cap)
            return self.rs.buf.n

        self.rs.buf.refresh = capped
        self.first_cache: dict[int, tuple] = {}
        self.neck_first_cache: dict[int, dict] = {}
        self._neck: NeckBuffer | None = None
        self._kv: KvBuffer | None = None
        hdr = self.root / "buffer" / "header.json"
        # the front end the scene tokens came from (V6): the StepBuffer header holds the collection config
        self.encoder = (json.loads(hdr.read_text()).get("encoder") if hdr.exists() else None) or "holobrain_gd"
        self.refresh()

    @property
    def capacity(self) -> int:
        return self.rs.buf.capacity

    @property
    def has_neck(self) -> bool:
        """A version-2 shard: neck_buffer/ next to buffer/ and item_buffer/."""
        return (self.root / "neck_buffer" / "header.json").exists()

    @property
    def neck(self) -> NeckBuffer:
        """The neck sidecar, opened read-only on first use (never creates a file)."""
        if self._neck is None:
            if not self.has_neck:
                raise ValueError(f"{self.root}: no neck_buffer/ (a version-1 shard); the neck maps exist only in "
                                 "shards collected with --lang-version 2")
            nb = NeckBuffer(self.root / "neck_buffer")
            if nb.capacity != self.capacity:
                raise ValueError(f"{self.root}: neck_buffer capacity {nb.capacity} != item_buffer capacity "
                                 f"{self.capacity}")
            self._neck = nb
        return self._neck

    @property
    def has_kv(self) -> bool:
        """A V6 shard or ring: kv_buffer/ (each frame's HoloBrain-Qwen prefix keys / values)."""
        return (self.root / "kv_buffer" / "header.json").exists()

    @property
    def kv(self) -> KvBuffer:
        """The kv sidecar, opened read-only on first use (never creates a file)."""
        if self._kv is None:
            if not self.has_kv:
                raise ValueError(f"{self.root}: no kv_buffer/ (the HoloBrain-Qwen words read each frame's prefix; "
                                 "collect with --encoder holobrain_qwen)")
            kb = KvBuffer(self.root / "kv_buffer")
            if kb.capacity != self.capacity:
                raise ValueError(f"{self.root}: kv_buffer capacity {kb.capacity} != item_buffer capacity "
                                 f"{self.capacity}")
            self._kv = kb
        return self._kv

    def check_kv(self, rows, ep, t) -> None:
        """The kv rows at these slots must hold the item sidecar's (episode_id, t_in_ep) there (as check_neck)."""
        self._check_side("kv_buffer", rows, ep, t)

    def kv_rows(self, rows, check: bool = True) -> dict:
        """kv rows of these ring slots -> {field: array [len(rows), ...]}; check: aligned with the item sidecar."""
        rows = np.asarray(rows, np.int64).reshape(-1)
        out = self.kv.read(rows)
        if check:
            self.check_kv(rows, out["episode_id"], out["t_in_ep"])
        return out

    def _open_step(self):
        for f in STEP_FIELDS:
            p = self.root / "buffer" / f"{f}.npy"
            if f not in self.step and p.exists():
                self.step[f] = np.load(p, mmap_mode="r")

    def _row_cap(self) -> int | None:
        """The rows this source may use: up to the end of the last recorded episode (review round 3, C2) and
        max_n (a frozen dataset). Rows past the last episode line are an episode in progress or a stale
        writer's; a header with FEWER rows than the lines record was rewritten by a stale writer or copied
        mid-write, and is refused."""
        ends = [int(d["rows"][1]) for d in self.lines.values() if d.get("rows")]
        if self.max_n is not None:
            ends = [e for e in ends if e <= self.max_n]
        caps = [c for c in ((max(ends) if ends else None), self.max_n) if c is not None]
        if ends and self.rs.buf.n < max(ends):
            raise RuntimeError(f"{self.root}: header.json says {self.rs.buf.n} rows, the episode lines end at "
                               f"{max(ends)} (a stale writer's header or a partial copy): check with "
                               "scripts/lang/check_shards.py")
        return min(caps) if caps else None

    def refresh(self) -> int:
        lj = self.root / "lang_episodes.jsonl"   # the lines first: they cap the rows RelabelSampler indexes
        if lj.exists():
            for ln in lj.read_text().splitlines():
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if not d.get("skipped"):
                    self.lines[int(d["episode_id"])] = d
        self.rs.refresh()
        self._open_step()
        if self._neck is not None:   # after the lines: the writer publishes every header before the line
            self._neck.refresh()
        if self._kv is not None:
            self._kv.refresh()
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
            self.n_stale_dropped = int((~fresh).sum())
            for f in ("rows", "next", "first", "mixed", "eid", "t", "task", "host", "emb"):
                setattr(self, f, getattr(self, f)[fresh])
        self.recall = recall_labels(np.asarray(A["h_rules"]), np.asarray(A["episode_id"]),
                                    np.asarray(A["t_in_ep"]).astype(np.int64), rs.buf.n, self.recall_horizon)
        # drop cached frame 0 of episodes no longer in the ring
        live = set(int(e) for e in np.unique(self.eid))
        self.first_cache = {e: v for e, v in self.first_cache.items() if e in live}
        self.neck_first_cache = {e: v for e, v in self.neck_first_cache.items() if e in live}
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

    # --------------------------------------------------------- neck rows ----
    def neck_first(self, eid: int, first_slot: int) -> dict:
        """Frame 0's neck row {field: array}, cached per episode like first_frame (2.9 MB each)."""
        v = self.neck_first_cache.get(eid)
        if v is None:
            v = {f: np.array(a[first_slot]) for f, a in self.neck.arrays.items()}
            self.neck_first_cache[eid] = v
        return v

    def check_neck(self, rows: np.ndarray, ep: np.ndarray, t: np.ndarray) -> None:
        """The neck rows at these ring slots must hold the item sidecar's (episode_id, t_in_ep) there: a rollback
        or resume mirrored on one side only shows here, on the CPU (FUSION_PLAN R6)."""
        self._check_side("neck_buffer", rows, ep, t)

    def _check_side(self, name: str, rows: np.ndarray, ep: np.ndarray, t: np.ndarray) -> None:
        A = self.rs.buf.arrays
        rows = np.asarray(rows, np.int64)
        want_e, want_t = np.asarray(A["episode_id"][rows], np.int64), np.asarray(A["t_in_ep"][rows], np.int64)
        bad = np.flatnonzero((np.asarray(ep, np.int64) != want_e) | (np.asarray(t, np.int64) != want_t))
        if bad.size:
            b = int(bad[0])
            raise RuntimeError(f"{self.root}: {name} row {int(rows[b])} holds episode {int(ep[b])} tick "
                               f"{int(t[b])}, item_buffer episode {int(want_e[b])} tick {int(want_t[b])} ({bad.size} "
                               "of the rows read): the sidecars are out of lockstep; check with "
                               "scripts/lang/check_shards.py")

    def neck_rows(self, rows, check: bool = True) -> dict:
        """Neck rows of these ring slots -> {field: array [len(rows), ...]} (eval_offline, check_shards); check:
        their alignment with the item sidecar (check_neck)."""
        rows = np.asarray(rows, np.int64).reshape(-1)
        out = self.neck.read(rows)
        if check:
            self.check_neck(rows, out["episode_id"], out["t_in_ep"])
        return out


class LangSampler:
    """sample(B) -> batch dict (Section 4.4 output table).

    neck: also gather the version-2 neck rows of the 5 frames (``neck`` / ``neck_idx``, module docstring); a
    source without neck_buffer/ is a ValueError. episodes_per_task = N > 0: draw only from the first N episodes
    of each (task, embodiment) (``_episode_subset``), for the training sampler of the scene-count curve.
    kv (V6): also gather the prefix keys / values of frames t and t+1 (``kv`` = {"k", "v", "episode_id",
    "t_in_ep"} with one entry per unique (source, ring row), ``kv_idx`` int64 [B, 2]); a source without kv_buffer/
    is a ValueError. Every source must come from the config's encoder, with one scene width; a probe sampler's
    item token masks must be as wide as the scene tokens (ref_mask).
    """

    def __init__(self, roots, cfg, tasks=None, embodiments=None, seed: int = 0, probe: bool = False,
                 n_threads: int = 8, split: str = "train", max_rows: dict | None = None, neck: bool = False,
                 episodes_per_task: int = 0, kv: bool = False):
        self.cfg = cfg
        self.k = int(cfg.k_prev)
        self.with_neck = bool(neck)
        self.with_kv = bool(kv)
        self.probe = bool(probe)
        self.episodes_per_task = int(episodes_per_task)
        assert self.episodes_per_task >= 0, self.episodes_per_task
        self.sources = []
        self.max_rows = {str(Path(k).resolve()): int(v) for k, v in (max_rows or {}).items()}
        for r in roots:   # shards still being collected may have no rows (or no arrays) yet
            if self.max_rows and str(Path(r).resolve()) not in self.max_rows:
                continue   # frozen dataset: shards started after the freeze are not part of it
            src = LangSource(r, self.k, int(cfg.recall_ticks),
                             max_n=self.max_rows.get(str(Path(r).resolve())),
                             drop_stale_bead=getattr(cfg, "drop_stale_bead_rows", True))
            if src.rows.size and "scene_tokens" in src.step:
                self._check_neck_source(src)
                self._check_encoder_source(src)
                self.sources.append(src)
            else:
                print(f"[data] skipping {r}: no usable rows yet")
        unique_source_names(self.sources)
        self.tasks = None if tasks in (None, "all") else tuple(tasks)
        self.embodiments = None if embodiments in (None, "all") else tuple(embodiments)
        self.rng = np.random.default_rng(seed)
        self.split = split
        self.bank = language.load_bank()
        self.pool = ThreadPoolExecutor(max_workers=int(n_threads))
        self.stats = {"mixed": 0, "pairs": 0, "harm": 0, "n": 0}
        self.refresh()

    def add_source(self, root) -> None:
        src = LangSource(root, self.k, int(self.cfg.recall_ticks),
                         drop_stale_bead=getattr(self.cfg, "drop_stale_bead_rows", True))
        if src.rows.size and "scene_tokens" in src.step:
            self._check_neck_source(src)
            self._check_encoder_source(src)
            self.sources.append(src)
            unique_source_names(self.sources)
            self.refresh(sources=False)
        else:
            print(f"[data] add_source {root}: no usable rows yet (added at the next refresh)")
            self._pending = getattr(self, "_pending", []) + [root]

    def refresh_pending(self) -> None:
        for root in list(getattr(self, "_pending", [])):
            self._pending.remove(root)
            self.add_source(root)

    def _check_neck_source(self, src: LangSource) -> None:
        """With neck on: the source must be a version-2 shard whose neck fields match the other sources'."""
        if not self.with_neck:
            return
        if not src.has_neck:
            raise ValueError(f"{src.root}: no neck_buffer/ (a version-1 shard): scene_fusion needs version-2 "
                             "shards (collect --lang-version 2)")
        fields = src.neck.fields   # opens it (read-only) and checks its capacity
        if self.sources and fields != self.sources[0].neck.fields:
            raise ValueError(f"{src.root}: neck fields {fields} differ from {self.sources[0].root}'s "
                             f"{self.sources[0].neck.fields}")

    def _check_encoder_source(self, src: LangSource) -> None:
        """V6: the source's front end is the config's, its scene width the other sources', its item masks as wide
        as its scene tokens (probe), and with kv on it has kv_buffer/."""
        want = getattr(self.cfg, "encoder", "holobrain_gd")
        if src.encoder != want:
            raise ValueError(f"{src.root}: scene tokens of the {src.encoder} front end, this config's encoder is {want}")
        width = int(src.step["scene_tokens"].shape[1])
        if self.sources and width != int(self.sources[0].step["scene_tokens"].shape[1]):
            raise ValueError(f"{src.root}: {width} scene tokens per frame, {self.sources[0].root} has "
                             f"{int(self.sources[0].step['scene_tokens'].shape[1])}")
        if self.probe and src.encoder == "holobrain_qwen":   # GD shards: 3 x 400 masks for 1200 tokens
            buf = src.rs.buf
            cells = int(getattr(buf, "n_cam", 3)) * int(getattr(buf, "n_tok", width // 3))
            if cells != width:
                raise ValueError(f"{src.root}: item token masks of {cells} cells per frame, {width} scene tokens "
                                 "(the probe's ref_mask needs masks on the scene token grid)")
        if self.with_kv:
            src.kv   # opens it (read-only) and checks its capacity

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
        keep = self._episode_subset() if self.episodes_per_task > 0 else None
        self.episode_keep = keep
        if keep is not None:
            ok &= cat([np.isin(s.eid, [e for jj, e in keep if jj == j]) for j, s in enumerate(self.sources)], bool)
        self.ok = ok
        self.by_task = {int(t): np.flatnonzero(ok & (self.task_of == t)) for t in np.unique(self.task_of[ok])}
        self.mixed_by_task = {t: idx[self.mixed_of[idx]] for t, idx in self.by_task.items()}
        self.mixed_by_task = {t: v for t, v in self.mixed_by_task.items() if v.size}
        self.all_ok = np.flatnonzero(ok)
        # episodes per (source) for the shuffled control (the same subset as the drawn rows when one is set)
        self.episodes = [(j, int(e)) for j, s in enumerate(self.sources) for e in np.unique(s.eid)
                         if keep is None or (j, int(e)) in keep]
        return int(self.all_ok.size)

    def _episode_subset(self) -> set[tuple[int, int]]:
        """(source index, episode id) of the first episodes_per_task episodes of each (task, embodiment), in the
        order sorted(eids, key=stable_hash(("e6", task, embodiment, eid))): fixed, so the subset for N is part of
        the subset for N + 1 and every sampler over the same data draws the same episodes. Keyed by (source,
        eid): episode ids repeat across sources (a _bead shard reuses its shard's ids; a version-2 replay its v1
        shard's), and a repeated id sorts next to its twin (the same layout), ties broken by source name."""
        groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for j, s in enumerate(self.sources):
            eids, first = np.unique(s.eid, return_index=True)
            for e, i0 in zip(eids.tolist(), first.tolist()):
                groups.setdefault((CANON_TASKS[int(s.task[i0])], str(s.emb[i0])), []).append((j, int(e)))
        keep = set()
        for (task, emb), eps in groups.items():
            eps.sort(key=lambda je: (stable_hash(("e6", task, emb, je[1])), self.sources[je[0]].name, je[0]))
            keep.update(eps[: self.episodes_per_task])
        return keep

    def episode_counts(self) -> dict[tuple[str, str], int]:
        """{(task, embodiment): episodes the sampler draws from} (after the task / embodiment filters and the
        episode subset): with the sample count, the repeats per episode of a run (FUSION_PLAN C1)."""
        out: dict[tuple[str, str], int] = {}
        for j, s in enumerate(self.sources):
            m = self.ok[self.sid == j]
            task, emb = s.task[m], s.emb[m]
            for i0 in np.unique(s.eid[m], return_index=True)[1].tolist():
                key = (CANON_TASKS[int(task[i0])], str(emb[i0]))
                out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

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
    def _gather_neck(self, where: dict, B: int) -> tuple[dict, np.ndarray]:
        """-> (neck {field: [U, ...]}, neck_idx int64 [B, 5]): every unique (source, ring row) among the batch's
        frames (t, t-k, 0, t+1, t+1-k), read once (rows_of: the scene tokens' row arithmetic); a pair's members
        and frames shared by samples (t = 0, t + 1 of one sample = t of another) point at one entry. Frame-0 rows
        come from the per-episode cache, the others from the memmaps on the thread pool. No random numbers."""
        idx = np.empty((B, len(NECK_FRAMES)), dtype=np.int64)
        uniq: dict[tuple[int, int], int] = {}
        todo: list[list] = []   # [u, source, ring row, eid if frame 0 of a sample else None]
        for gi, bs in where.items():
            j, i = int(self.sid[gi]), int(self.loc[gi])
            rows = self.sources[j].rows_of(i)
            for f, name in enumerate(NECK_FRAMES):
                u = uniq.get((j, rows[name]))
                if u is None:
                    u = uniq[(j, rows[name])] = len(todo)
                    todo.append([u, j, rows[name], None])
                if name == "first_row":
                    todo[u][3] = int(self.sources[j].eid[i])
                idx[bs, f] = u
        fields = self.sources[todo[0][1]].neck.fields
        out = {f: np.empty((len(todo),) + tuple(shp), dtype=dt) for f, (shp, dt) in fields.items()}

        def fill(item):
            u, j, row, eid = item
            s = self.sources[j]
            if eid is not None:
                for f, v in s.neck_first(eid, row).items():
                    out[f][u] = v
            else:
                for f, a in s.neck.arrays.items():
                    out[f][u] = a[row]

        list(self.pool.map(fill, todo))
        js, rws = (np.array([it[c] for it in todo], dtype=np.int64) for c in (1, 2))   # entry u = todo[u]
        for j in np.unique(js).tolist():   # lockstep check against the item sidecar (episode_id, t_in_ep)
            m = js == j
            self.sources[j].check_neck(rws[m], out["episode_id"][m], out["t_in_ep"][m])
        return out, idx

    def _gather_kv(self, where: dict, B: int) -> tuple[dict, np.ndarray]:
        """-> (kv {"k", "v", "episode_id", "t_in_ep": [U, ...]}, kv_idx int64 [B, 2]): the prefix rows of frames t
        and t+1 (rows_of "row" / "next_row"), each unique (source, ring row) read once on the thread pool and
        checked against the item sidecar. No random numbers."""
        idx = np.empty((B, 2), dtype=np.int64)
        uniq: dict[tuple[int, int], int] = {}
        todo: list[tuple[int, int]] = []
        for gi, bs in where.items():
            j, i = int(self.sid[gi]), int(self.loc[gi])
            rows = self.sources[j].rows_of(i)
            for f, name in enumerate(("row", "next_row")):
                u = uniq.get((j, rows[name]))
                if u is None:
                    u = uniq[(j, rows[name])] = len(todo)
                    todo.append((j, rows[name]))
                idx[bs, f] = u
        fields = self.sources[todo[0][0]].kv.fields
        out = {f: np.empty((len(todo),) + tuple(shp), dtype=dt) for f, (shp, dt) in fields.items()}

        def fill(u):
            j, row = todo[u]
            for f, a in self.sources[j].kv.arrays.items():
                out[f][u] = a[row]

        list(self.pool.map(fill, range(len(todo))))
        js, rws = (np.array([it[c] for it in todo], dtype=np.int64) for c in (0, 1))
        for j in np.unique(js).tolist():
            m = js == j
            self.sources[j].check_kv(rws[m], out["episode_id"][m], out["t_in_ep"][m])
        return out, idx

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
        if self.with_neck:
            out["neck"], out["neck_idx"] = self._gather_neck(where, len(picks))
        if self.with_kv:
            out["kv"], out["kv_idx"] = self._gather_kv(where, len(picks))
        meta = {k: [] for k in ("sentence", "rule_id", "consequence", "task", "host", "embodiment", "source",
                                "episode_id", "rule_col", "row", "next_row", "prev_row", "first_row", "next_prev_row",
                                "t_in_ep")}
        h, hn, fs, fsr, pair = [], [], [], [], []
        tag = {k: [] for k in ("ref_center", "ref_center_valid", "ref_center_next", "ref_center_valid_next")} \
            if getattr(self.cfg, "oracle_ref_tag", False) else None
        aux = ({"aux_target": [], "aux_valid": []} if (getattr(self.cfg, "aux_target_loss", False)
                                                       or getattr(self.cfg, "aux_attention_loss", False)) else None)
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
            if aux is not None:   # V4: the named item's centre at frames t and 0
                c, v = named_centers(A, rec, rule, (rows["row"], rows["first_row"]))
                aux["aux_target"].append(c), aux["aux_valid"].append(v)
            if tag is not None:   # V3: the named object's centre in frames [t, t-k, 0] and [t+1, t+1-k, 0]
                c, v = ref_centers(A, rec, rule, (rows["row"], rows["prev_row"], rows["first_row"]))
                cn, vn = ref_centers(A, rec, rule, (rows["next_row"], rows["next_prev_row"], rows["first_row"]))
                for k2, x in zip(tag, (c, v, cn, vn)):
                    tag[k2].append(x)
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
        if tag is not None:
            out.update({k2: np.stack(v) for k2, v in tag.items()})
        if aux is not None:
            out.update({k2: np.stack(v) for k2, v in aux.items()})
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
