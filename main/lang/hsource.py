"""Where a training sample's h comes from (VLM_FAILURE_PLAN, Yuxuan 2026-10-26).

``HSourceSampler`` is the env-side RelabelSampler of one data root with two additions, both off at the defaults
(h_source "sim", rule_consequences "all", where it is RelabelSampler exactly):

- rule_consequences "vlm_prompts": a rule is usable only if its task's prompts in vlm_prompts.yaml list its
  consequence (T1 avoid, T5 upright, T6 exclude_particles; a task the file does not list keeps every rule);
- h_source "vlm:<name>": ``h_rules`` and ``fails_soon`` in ``buf.arrays`` are replaced, in this process's reader only
  (the files are never written), by the VLM's labels (``<root>/vlm_labels/<name>/<episode_id>.json``). The score at
  labelled tick k is the h of every row whose nearest labelled tick is k (ties go to the earlier one; rows after the
  last labelled tick take the last score), and fails_soon is recomputed from it as the sidecar does: some h < 0 in
  rows t .. t + FAILS_SOON_H - 1 of the episode. A rule is usable only where its episode's label file has it; an
  episode without a label file, or with labels of another prompt hash, has no usable rule (never trained on).
  The simulator's arrays stay as ``sim_h`` / ``sim_fs`` (``agreement()``, the run's log).

The overlay is rebuilt from the label files at every refresh, so rows a resumed ring rolled back and wrote again (a
new episode id) never keep an old episode's labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ensure_item_path

ensure_item_path()

from main.item_envs.item_buffer import FAILS_SOON_H  # noqa: E402
from main.item_envs.relabel import RelabelSampler  # noqa: E402

from .vlm_prompts import allowed_consequences, label_dir, prompt_hash, vlm_name  # noqa: E402


def spread(ticks, scores, t: np.ndarray) -> np.ndarray:
    """h at ticks t from labels (ticks, scores): the score of the nearest labelled tick (ties: the earlier one)."""
    L = np.asarray(ticks, np.int64)
    S = np.asarray(scores, np.float32)
    t = np.asarray(t, np.int64)
    j = np.clip(np.searchsorted(L, t, side="left"), 0, L.size - 1)    # first label tick >= t
    prev = np.clip(j - 1, 0, L.size - 1)
    take_prev = (j > 0) & ((t - L[prev]) <= (L[j] - t))
    return S[np.where(take_prev, prev, j)]


def fails_soon_of(h: np.ndarray, horizon: int = FAILS_SOON_H) -> np.ndarray:
    """fails_soon over one episode's rows (in tick order) of one rule: h < 0 within rows t .. t + horizon - 1."""
    neg = np.concatenate([[0], np.cumsum(np.asarray(h) < 0)])
    n = len(h)
    end = np.minimum(np.arange(n) + horizon, n)
    return (neg[end] - neg[np.arange(n)]) > 0


class HSourceSampler(RelabelSampler):
    def __init__(self, root, *, h_source: str = "sim", rule_consequences: str = "all", **kw):
        # set before RelabelSampler.__init__: it calls refresh()
        self.h_source, self.rule_consequences = str(h_source), str(rule_consequences)
        self.vlm = vlm_name(self.h_source)
        self.want_hash = prompt_hash() if self.vlm else None
        self.labels: dict[int, dict] = {}       # episode_id -> parsed label file (cached; written once)
        self.label_cols: dict[int, set] = {}    # episode_id -> rule columns with labels
        self.n_refused = 0
        self.sim_h = self.sim_fs = None
        super().__init__(root, **kw)

    # ------------------------------------------------------------ overlay --
    def _read_labels(self) -> None:
        d = label_dir(Path(self.root).parent, self.vlm)
        if not d.exists():
            return
        for f in d.glob("*.json"):
            try:
                eid = int(f.stem)
            except ValueError:
                continue
            if eid in self.labels:
                continue
            try:
                lab = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue   # being written (the labeler renames into place: should not happen)
            if lab.get("prompt_hash") != self.want_hash:
                self.n_refused += 1
                lab = {"rules": {}, "refused": f"prompt_hash {lab.get('prompt_hash')} != {self.want_hash}"}
            self.labels[eid] = lab

    def _overlay(self) -> None:
        A = self.buf.arrays
        if self.sim_h is None:
            self.sim_h, self.sim_fs = A["h_rules"], A["fails_soon"]
            A["h_rules"] = np.full(self.sim_h.shape, np.nan, np.float32)
            A["fails_soon"] = np.zeros(self.sim_fs.shape, bool)
        H, F = A["h_rules"], A["fails_soon"]
        H[:] = np.nan
        F[:] = False
        self._read_labels()
        C, n = self.buf.capacity, self.buf.n
        m = C if n == C else n
        ep = np.asarray(A["episode_id"][:m]).reshape(-1).astype(np.int64)
        t = np.asarray(A["t_in_ep"][:m]).reshape(-1).astype(np.int64)
        self.label_cols = {}
        if not m:
            return
        order = np.argsort(ep, kind="stable")
        eids, starts = np.unique(ep[order], return_index=True)
        bounds = list(starts) + [order.size]
        for k, eid in enumerate(eids):
            lab = self.labels.get(int(eid))
            if lab is None or not lab.get("rules"):
                continue
            rows = order[bounds[k]:bounds[k + 1]]
            rows = rows[np.argsort(t[rows], kind="stable")]
            tt = t[rows]
            if lab.get("n_rows") is not None and (rows.size != int(lab["n_rows"]) or tt[0] != 0
                                                   or not np.array_equal(tt, np.arange(rows.size))):
                continue   # another episode's rows under this id (should not happen) or a partial one
            cols = set()
            for rule in lab["rules"].values():
                if not rule.get("score"):
                    continue
                j = int(rule["idx"])
                h = spread(lab["ticks"], rule["score"], tt)
                H[rows, j] = h
                F[rows, j] = fails_soon_of(h)
                cols.add(j)
            self.label_cols[int(eid)] = cols

    def refresh(self) -> int:
        if self.vlm is not None:
            self.buf.refresh()   # n / head first (LangSource caps them): the overlay covers exactly these rows
            self._overlay()
        return super().refresh()

    def _usable(self, rule: dict, rec: dict) -> bool:
        allowed = allowed_consequences(self.rule_consequences, rec["task"])
        if allowed is not None and rule["consequence"] not in allowed:
            return False
        if self.vlm is not None and int(rule["idx"]) not in self.label_cols.get(int(rec["episode_id"]), ()):
            return False
        return super()._usable(rule, rec)

    # -------------------------------------------------------------- stats --
    def agreement(self) -> dict:
        """VLM vs simulator on the labelled cells (usable rows x labelled rules), per task: how often each says
        h < 0 and how often they agree; plus fails_soon. Empty for h_source sim."""
        if self.vlm is None or not self._rows.size:
            return {}
        A = self.buf.arrays
        ep = np.asarray(A["episode_id"])[self._rows].astype(np.int64)
        out: dict = {}
        for eid in np.unique(ep):
            cols = sorted(self.label_cols.get(int(eid), ()))
            rec = self.records.get(int(eid))
            if not cols or rec is None:
                continue
            r = self._rows[ep == eid]
            v, s = A["h_rules"][r][:, cols] < 0, np.asarray(self.sim_h[r][:, cols]) < 0
            fv, fs = A["fails_soon"][r][:, cols], np.asarray(self.sim_fs[r][:, cols])
            c = out.setdefault(rec["task"], {"cells": 0, "vlm_neg": 0, "sim_neg": 0, "agree": 0, "fs_agree": 0,
                                             "both_neg": 0})
            c["cells"] += int(v.size)
            c["vlm_neg"] += int(v.sum())
            c["sim_neg"] += int(s.sum())
            c["both_neg"] += int((v & s).sum())
            c["agree"] += int((v == s).sum())
            c["fs_agree"] += int((fv == fs).sum())
        return out


def missing_labels(root, vlm: str) -> list[int]:
    """Episode ids with rows in this data root (lang_episodes.jsonl), of a task with VLM prompts, but no label file
    of this VLM yet. (Episodes of other tasks never train under a VLM h: nothing to wait for.)"""
    from .vlm_prompts import label_path, load_prompts

    root = Path(root)
    tasks = set(load_prompts()["prompts"])
    out = []
    lj = root / "lang_episodes.jsonl"
    for ln in lj.read_text().splitlines() if lj.exists() else []:   # read only (a shared data root)
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue   # the line being written
        rows = d.get("rows") or [0, 0]
        if d.get("skipped") or int(rows[1]) <= int(rows[0]) or d.get("task") not in tasks:
            continue
        if not label_path(root, vlm, int(d["episode_id"])).exists():
            out.append(int(d["episode_id"]))
    return out


def labeler_age(roots, vlm: str) -> float | None:
    """Seconds since the newest labeler heartbeat in these roots' label dirs (None: no labeler seen)."""
    import time

    ages = []
    for r in roots:
        for p in label_dir(r, vlm).glob("heartbeat*.json") if label_dir(r, vlm).exists() else []:
            try:
                ages.append(time.time() - p.stat().st_mtime)
            except OSError:
                pass
    return min(ages) if ages else None


def wait_labels(roots, vlm: str, out_of_time, prefix: str = "", poll_s: float = 30.0,
                log_every_s: float = 300.0, sleep=None, want: bool = False) -> str | None:
    """Block until every episode with rows in these roots has a label file of this VLM. Returns None when they all
    have, or the reason to stop (out_of_time() true; the caller saves state and exits 75). want: keep the roots'
    WANTED markers fresh while waiting and remove them at the end (want_labels)."""
    import time

    sleep = sleep or time.sleep
    last = 0.0
    t0 = time.time()
    while True:
        missing = {str(r): missing_labels(r, vlm) for r in roots}
        n = sum(len(v) for v in missing.values())
        if want:
            for r in roots:
                want_labels(r, vlm, on=bool(n))
        if not n:
            if time.time() - t0 > 1:
                print(f"{prefix}[vlm] every episode labelled by {vlm} after {time.time() - t0:.0f}s", flush=True)
            return None
        if out_of_time():
            return f"waiting for {n} {vlm} label files"
        if time.time() - last >= log_every_s:
            age = labeler_age(roots, vlm)
            where = ", ".join(f"{Path(k).name}: {len(v)}" for k, v in missing.items() if v)
            print(f"{prefix}[vlm] waiting for {n} episodes' {vlm} labels ({where}); labeler heartbeat "
                  f"{'none seen' if age is None else f'{age:.0f}s ago'}", flush=True)
            last = time.time()
        sleep(poll_s)


WANTED_FRESH_S = 1800.0   # a WANTED marker older than this is a run that stopped: no labeler is kept for it


def want_labels(root, vlm: str, on: bool = True) -> None:
    """The run's ring asks for a labeler (``<root>/vlm_labels/<vlm>/WANTED``, touched while it collects and waits;
    removed once every episode is labelled). The labeler stays up while a fresh marker exists
    (vlm_label serve --idle-exit) and scripts/lang/vlm/labeler_watch.sh starts one for it."""
    import time

    p = label_dir(root, vlm) / "WANTED"
    if on:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"t": time.time()}))
    else:
        p.unlink(missing_ok=True)


def wanted(roots, vlm: str) -> bool:
    import time

    for r in roots:
        p = label_dir(r, vlm) / "WANTED"
        try:
            if time.time() - p.stat().st_mtime < WANTED_FRESH_S:
                return True
        except OSError:
            pass
    return False
