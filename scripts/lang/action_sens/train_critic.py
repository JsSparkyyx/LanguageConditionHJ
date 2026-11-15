#!/usr/bin/env python
"""Steps 3 and 4 of ACTION_SENSITIVITY_PLAN (2026-11-10): train a critic only, score it on the held-out branch states.

    python scripts/lang/action_sens/train_critic.py criteria                      # fix the pass numbers (once)
    python scripts/lang/action_sens/train_critic.py train --name A [--grad-steps 6000] [--time-budget 3000] [--resume]
    python scripts/lang/action_sens/train_critic.py score [--names A B ...]       # -> runs/lang/action_sens/scores.json

The runs (RUNS): the decision a critic scores (1 tick, 3 ticks, or a 12-tick chunk), how the action enters it (current
Linear injection, or the revised one), and the data (the vanilla run's warmup + ring restricted to the tray-sliding
task, with or without the boundary branch data as a quarter of each batch). All use the MLP readout. Otherwise the
vanilla run's settings (its lang_config: gamma 0.95, batch 64, lr 3e-4, TF32 train step), from scratch.

Critic only: no actor loss; the action in the target is the stored one.
    1 tick:  y = (1 - g) h_t + g min(h_t, mean of the target twins at (s_t+1, the stored command of row t+1))
             (critic_retrain.py's "data" target)
    chunk:   a row t starts a chunk A = the stored commands of rows t .. t+11 (zeros past the episode's end).
             With row t+12 in the episode and not its last: y_t+12 = mean of the target twins at (s_t+12, the
             stored chunk of row t+12), then y_t+k = (1 - g) h_t+k + g min(h_t+k, y_t+k+1) for k = 11 .. 0.
             Else the episode ends inside the chunk: the recursion starts at its last row L with y_L = h_L.

Second round (plan section 7, 2026-11-11):
    3 ticks: "the lower rate". The action is the sum of the 3 stored commands of rows t .. t+2 and enters the model
             as a one-tick action does (critic_chunk stays 1; the revised injection's bounds are 3 x one tick's).
             Target: as the chunk's with 3 in place of 12.
    batch mix "new": 50% normal rows, 25% branch continuation rows, 25% decision rows. A decision row is one of
             the first max(H, 1) rows of a branch from its state row (H = the branch's decision length; the
             planner branch counts its state row): the rows whose action is part of the decision. They are drawn
             one branch at a time (every branch equally often, then one of its decision rows), so about 45% of them
             are state rows, where the 7 or 9 branches show the same state with different actions.
             Mix "old" (first round): 75% normal, 25% any branch row (decision rows about 2%).
    data:    "small" = the first round's 100 states (data/lang_asens), "large" = small + data/lang_asens2.

Third round (plan section 8, 2026-11-12):
    target "measured" (runs M1, M3, M12): y_t = V95 of the sample's rule column from its row t to the end of its
             episode or branch (y_last = h_last, y_t = (1 - g) h_t + g min(h_t, y_t+1)), only recorded margins, no
             critic and no target network. The same number for a 1-tick, 3-tick or chunk critic. Normal rows only
             from episodes stored completely and driven by the planner on every tick (planner_episodes); large
             branch data, the new mix, MSE on both twins as in rounds 1 and 2 (M3_huber: Huber, delta 0.5).
    Resuming D3 / H to 20,000 steps: train --name D3 --run <a copy of runs/D3> --resume --grad-steps 20000.

Every --probe-every steps the fixed probe states (the held-out branch states and as many training ones) are scored:
rank agreement and spread ratio over each state's decisions (section 4 of the plan), and the spread of Q over 16
random actions / chunks -> <run>/diag.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import queue
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from main.lang.config import LangConfig  # noqa: E402
from main.lang.train import EXIT_RESUME, LOG_EVERY, LangTrainer, _t, caches_of, matmul_tf32, roots_of  # noqa: E402

VANILLA = REPO / "runs/lang/rounds_ur5_g095_1023"
NORMAL = [REPO / "data/lang_1023/warmup", VANILLA / "ring"]
BRANCH = REPO / "data/lang_asens"        # round 1: 100 states, 7 decisions ("small")
BRANCH2 = REPO / "data/lang_asens2"      # round 2: about 500 states, 9 decisions ("large" = both)
OUT = REPO / "runs/lang/action_sens"
NORM = REPO / "runs/lang/diag_1030/retrain/action_norm.json"
CRITERIA = OUT / "criteria.json"
T5 = "tray_transport"
CHUNK = 12
BRANCH_FRAC = 0.25
N_RAND = 16
# shares of a batch: (normal, branch continuation, decision); None = any branch row
MIX = {"old": (0.75, None, 0.25), "new": (0.50, 0.25, 0.25)}


def _run(ticks, injection, branch, mix="old", target="bootstrap", loss="mse"):
    r = {"decision_ticks": ticks, "injection": injection, "branch_data": branch, "mix": mix}
    if target != "bootstrap":
        r["target"] = target
    if loss != "mse":
        r["loss"] = loss
    return r


# branch_data: None (normal rows only), "small" or "large". E was dropped on 2026-11-11 (C failed with more help).
# target "measured" (third round, plan section 8.4, 2026-11-12): y_t = V95 of the row's own recorded h from tick t
# to the end of its episode or branch, no critic in the target; normal rows only from complete planner episodes.
RUNS = {"A": _run(1, "linear", "small"), "B": _run(1, "revised", "small"), "C": _run(CHUNK, "linear", "small"),
        "D": _run(CHUNK, "revised", "small"), "E": _run(CHUNK, "linear", None), "F": _run(1, "revised", None),
        "D2": _run(CHUNK, "revised", "small", "new"), "G": _run(3, "revised", None),
        "B2": _run(1, "revised", "large", "new"), "H": _run(3, "revised", "large", "new"),
        "D3": _run(CHUNK, "revised", "large", "new"),
        "M1": _run(1, "revised", "large", "new", "measured"), "M3": _run(3, "revised", "large", "new", "measured"),
        "M12": _run(CHUNK, "revised", "large", "new", "measured"),
        # the same as M3 with a Huber loss (delta HUBER_DELTA, as fit_check.py) in place of MSE (Yuxuan 2026-11-12)
        "M3_huber": _run(3, "revised", "large", "new", "measured", "huber"),
        "H_huber": _run(3, "revised", "large", "new", "bootstrap", "huber")}   # H with the Huber loss
HUBER_DELTA = 0.5
ENC_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max",
              "scene_tokens", "scene_pos")
PASS = {"rank_agree_min": 0.70, "spread_ratio_min": 0.25, "gt_tol": 0.05}


def run_cfg(ticks: int, injection: str) -> LangConfig:
    """The model takes a chunk only for the 12-tick decision; a 3-tick decision is one summed action."""
    cfg = LangConfig.load(VANILLA / "lang_config.json")
    return dataclasses.replace(cfg, critic_readout="mlp", critic_injection=injection,
                               critic_chunk=CHUNK if int(ticks) == CHUNK else 1)


def branch_dirs(scope, small_root: Path = BRANCH) -> list:
    """The data roots of a run's branch scope (None -> [])."""
    return [] if not scope else ([Path(small_root)] + ([BRANCH2] if scope == "large" else []))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def restrict(sampler, keep: np.ndarray) -> None:
    """Drop rows from a LangSampler's draw (LangSampler.refresh's bookkeeping after `ok`)."""
    ok = sampler.ok & keep
    sampler.ok = ok
    sampler.by_task = {int(t): np.flatnonzero(ok & (sampler.task_of == t)) for t in np.unique(sampler.task_of[ok])}
    sampler.mixed_by_task = {t: idx[sampler.mixed_of[idx]] for t, idx in sampler.by_task.items()}
    sampler.mixed_by_task = {t: v for t, v in sampler.mixed_by_task.items() if v.size}
    sampler.all_ok = np.flatnonzero(ok)


def branch_keep(sampler, held_out: bool = False) -> np.ndarray:
    """Rows of a branch sampler a trainer may draw: not a context row (t_in_ep < n_ctx: the transition between
    context rows 0 and 1 is not a real one, the others repeat the parent's), and of a training episode."""
    keep = np.zeros(sampler.sid.size, bool)
    o = 0
    for s in sampler.sources:
        n = s.rows.size
        k = np.zeros(n, bool)
        for e in np.unique(s.eid):
            ln = s.lines.get(int(e)) or {}
            br = ln.get("branch")
            if br is None or bool(br.get("held_out")) != bool(held_out):
                continue
            m = s.eid == e
            k[m] = s.t[m] >= int(ln.get("n_ctx", 3))
        keep[o:o + n] = k
        o += n
    return keep


def branch_masks(sampler, held_out: bool = False):
    """-> (keep, dec, w) over a branch sampler's rows. keep: branch_keep. dec: the decision rows, the first
    max(H, 1) rows of a branch from its state row (H = the branch's decision length). w: 12 / max(H, 1), the times a
    decision row is repeated in the draw so that every branch is drawn equally often."""
    N = sampler.sid.size
    keep, dec, w = np.zeros(N, bool), np.zeros(N, bool), np.ones(N, np.int64)
    o = 0
    for s in sampler.sources:
        n = s.rows.size
        for e in np.unique(s.eid):
            ln = s.lines.get(int(e)) or {}
            br = ln.get("branch")
            if br is None or bool(br.get("held_out")) != bool(held_out):
                continue
            m = np.flatnonzero(s.eid == e)
            nc, hb = int(ln.get("n_ctx", 3)), max(int(br.get("H", 0)), 1)
            t = s.t[m]
            keep[o + m] = t >= nc
            dec[o + m] = (t >= nc) & (t < nc + hb)
            w[o + m] = max(CHUNK // hb, 1)
        o += n
    return keep, dec, w


def reweight(sampler, w: np.ndarray) -> None:
    """Repeat every row w times in the sampler's draw pools."""
    sampler.by_task = {t: np.repeat(i, w[i]) for t, i in sampler.by_task.items()}
    sampler.mixed_by_task = {t: np.repeat(i, w[i]) for t, i in sampler.mixed_by_task.items()}


class EpIndex:
    """Per source: for every ring slot, how many more rows its episode has (rem), and slot -> index row."""

    def __init__(self, s):
        A = s.rs.buf.arrays
        n, C = int(s.rs.buf.n), int(s.capacity)
        if n >= C:
            raise RuntimeError(f"{s.root}: the ring wrapped; the chunk reader assumes it did not")
        ep = np.asarray(A["episode_id"][:n]).astype(np.int64)
        t = np.asarray(A["t_in_ep"][:n]).astype(np.int64)
        cont = np.zeros(n, bool)
        cont[:-1] = (ep[1:] == ep[:-1]) & (t[1:] == t[:-1] + 1)
        end = np.where(~cont, np.arange(n), n)                     # index of each episode's last row
        end = np.minimum.accumulate(end[::-1])[::-1]
        self.rem = (end - np.arange(n)).astype(np.int64)           # rows after this one in its episode
        self.inv = np.full(n, -1, np.int64)
        self.inv[s.rows] = np.arange(s.rows.size)
        self.dtheta = s.step["dtheta"]
        self.h = A["h_rules"]


def planner_episodes(s) -> tuple[set, dict]:
    """Episodes of a normal source that are stored completely and driven by the planner on every tick: a line with
    status ok, not a filter episode, rows [r0, r1) holding exactly n_ticks rows of that episode id with t_in_ep
    0 .. n-1, and action_source 0 (planner) on every row. -> (episode ids, counts of what was left out and why)."""
    A = s.rs.buf.arrays
    n = int(s.rs.buf.n)
    p = s.root / "buffer" / "action_source.npy"
    src = np.asarray(np.load(p, mmap_mode="r")[:n]) if p.exists() else None
    ok, why = set(), {"filter": 0, "status": 0, "incomplete": 0, "not_planner": 0}
    for e, ln in s.lines.items():
        if (s.rs.records.get(int(e)) or {}).get("task") != T5:
            continue
        if ln.get("filter"):
            why["filter"] += 1
            continue
        if ln.get("status") != "ok":
            why["status"] += 1
            continue
        r0, r1 = (int(x) for x in ln["rows"])
        if r1 > n or r1 - r0 != int(ln.get("n_ticks", -1)) \
                or not (np.asarray(A["episode_id"][r0:r1]) == int(e)).all() \
                or not (np.asarray(A["t_in_ep"][r0:r1]) == np.arange(r1 - r0)).all():
            why["incomplete"] += 1
            continue
        if src is None or (np.asarray(src[r0:r1]) != 0).any():
            why["not_planner"] += 1
            continue
        ok.add(int(e))
    return ok, why


def measured_table(s, gamma: float, eids: set | None) -> np.ndarray:
    """V [n, R] (float32, NaN where not defined): for every stored row t of the episodes eids (None = every episode
    of the source) and every rule column, V95 of that column's recorded h from t to the episode's last stored row:
    V_last = h_last, V_t = (1 - g) h_t + g min(h_t, V_t+1). A branch is stored to 5 ticks after its last motion and
    nothing moves afterwards, so its h stays at h_last and this equals the V95 of the whole continuation."""
    A = s.rs.buf.arrays
    n = int(s.rs.buf.n)
    ep = np.asarray(A["episode_id"][:n]).astype(np.int64)
    t = np.asarray(A["t_in_ep"][:n]).astype(np.int64)
    h = np.asarray(A["h_rules"][:n], np.float64)
    cont = np.zeros(n, bool)                              # row i+1 continues row i's episode
    cont[:-1] = (ep[1:] == ep[:-1]) & (t[1:] == t[:-1] + 1)
    V = np.empty_like(h)
    g = float(gamma)
    V[n - 1] = h[n - 1]
    for i in range(n - 2, -1, -1):
        V[i] = (1 - g) * h[i] + g * np.minimum(h[i], V[i + 1]) if cont[i] else h[i]
    if eids is not None:
        V[~np.isin(ep, np.fromiter(eids, np.int64, len(eids)))] = np.nan
    return V.astype(np.float32)


class Batcher:
    """Batches of 64 from parts [(sampler, rows, kind)], kind 0 normal / 1 branch (continuation, or any branch row
    under the old mix) / 2 decision. Critics of 3 or 12 ticks also get, per sample, the stored commands of the
    decision window, the h of its rows, and the state and stored window at its end (a second read)."""

    def __init__(self, tr: "CriticTrainer", parts: list, chunk: int, seed: int, need_end: bool = True):
        self.tr, self.chunk, self.need_end = tr, int(chunk), bool(need_end)
        self.parts = [p for p in parts if p[1] > 0]
        self.rng = np.random.default_rng(seed)
        self.idx = {}
        if self.chunk > 1:
            for smp, _, _ in self.parts:
                for s in smp.sources:
                    if id(s) not in self.idx:
                        self.idx[id(s)] = EpIndex(s)
        self.q: queue.Queue = queue.Queue(maxsize=3)
        self.stop_flag = False
        self.err = None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        try:
            while not self.stop_flag:
                b = self.make()
                while not self.stop_flag:
                    try:
                        self.q.put(b, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:   # surfaced by next()
            import traceback

            traceback.print_exc()
            self.err = exc

    def next(self) -> dict:
        while True:
            if self.err is not None:
                raise RuntimeError(f"batcher failed: {self.err}")
            try:
                return self.q.get(timeout=1.0)
            except queue.Empty:
                continue

    def stop(self):
        self.stop_flag = True

    def make(self) -> dict:
        outs = []
        for smp, n, kind in self.parts:
            picks = smp.plan(n, self.rng)
            b = smp.sample(len(picks), rng=self.rng, picks=picks)
            b["_branch"] = np.full(len(picks), kind > 0)
            b["_kind"] = np.full(len(picks), kind, np.int64)
            if self.chunk > 1:
                self._chunk_part(smp, picks, b)
            outs.append(b)
        return concat(outs)

    def _chunk_part(self, smp, picks, b) -> None:
        chunk_fields(smp, picks, b, self.chunk, self.idx, self.rng, need_end=self.need_end)


def chunk_fields(smp, picks, b, H: int, idx: dict, rng, need_end: bool = True) -> None:
    """The decision window of every pick (H rows from its row): the stored commands, the h of its rows, where the
    recursion ends and whether it bootstraps there, the stored window at that end, and (need_end) the state at that
    end, a second read. idx: {id(source): EpIndex}. A measured-outcome target needs only the commands (need_end
    False skips the second read)."""
    B = len(picks)
    A = np.asarray(b["dtheta"]).shape[-1]
    cmds = np.zeros((B, H, A), np.float32)
    cmds2 = np.zeros((B, H, A), np.float32)
    hseq = np.zeros((B, H + 1), np.float32)
    L = np.zeros(B, np.int64)             # last row of the recursion (H with a bootstrap)
    boot = np.zeros(B, bool)
    picks2, where2 = [], []
    off = {id(s): o for s, o in zip(smp.sources, np.cumsum([0] + [s.rows.size for s in smp.sources[:-1]]))}
    for k, (gi, col, _) in enumerate(picks):
        s = smp.sources[int(smp.sid[gi])]
        ix = idx[id(s)]
        slot = int(s.rows[int(smp.loc[gi])])
        rem = int(ix.rem[slot])
        n = min(rem, H)                                   # rows after t that the chunk reaches
        cmds[k, : n + (1 if n < H else 0)] = np.asarray(ix.dtheta[slot: slot + n + (1 if n < H else 0)])
        hs = np.asarray(ix.h[slot: slot + n + 1, col], np.float32)
        hseq[k, : n + 1] = hs
        hseq[k, n + 1:] = hs[-1]
        j = int(ix.inv[slot + H]) if rem > H else -1      # row t+H in the episode and not its last
        if j >= 0:
            boot[k], L[k] = True, H
            m = min(int(ix.rem[slot + H]) + 1, H)
            cmds2[k, :m] = np.asarray(ix.dtheta[slot + H: slot + H + m])
            picks2.append((int(off[id(s)] + j), int(col), -1))
            where2.append(k)
        else:
            L[k] = n
    b["chunk_cmds"], b["chunk_h"], b["chunk_L"], b["chunk_boot"] = cmds, hseq, L, boot
    b["chunk_cmds_next"] = cmds2
    if not need_end:
        return
    if picks2:
        b2 = smp.sample(len(picks2), rng=rng, picks=picks2)
        for f in ENC_FIELDS:
            full = np.zeros((B,) + np.asarray(b2[f]).shape[1:], np.asarray(b2[f]).dtype)
            full[where2] = b2[f]
            b[f + "_end"] = full
    else:
        for f in ENC_FIELDS:
            b[f + "_end"] = np.zeros_like(np.asarray(b[f]))


def concat(bs: list) -> dict:
    if len(bs) == 1:
        return bs[0]
    out = {}
    for k in bs[0]:
        v = [b[k] for b in bs]
        if isinstance(v[0], np.ndarray):
            if k == "pair":   # pair ids are per part
                v = [np.where(x >= 0, x + 1000 * i, x) for i, x in enumerate(v)]
            out[k] = np.concatenate(v)
        elif isinstance(v[0], list):
            out[k] = sum(v, [])
        else:
            out[k] = v[0]
    return out


# ---------------------------------------------------------------------------
# probe states and the scores of section 4
# ---------------------------------------------------------------------------


class States:
    """The branch states of some episodes (collect_branches.py's states files), stacked, with the decisions a
    critic of `ticks` ticks scores: the planner's and the decisions of that length (1-tick: plan, t1_*; 3-tick:
    plan, t3_*; chunk: plan, ch_*). Files without a decision of that length are left out (the first round has no
    3-tick ones)."""

    def __init__(self, files, ticks: int):
        self.ticks = int(ticks)
        parts, self.episode, self.sentence, self.names = [], [], [], None
        for p in files:
            d = torch.load(p, map_location="cpu", weights_only=False)
            idx = [i for i, h in enumerate(d["decision_H"]) if h == 0 or h == self.ticks]
            names = [d["decisions"][i] for i in idx]
            if len(idx) < 2:
                continue
            if self.names is None:
                self.names = names
            elif names != self.names:
                raise RuntimeError(f"{p}: decisions {names}, the other files have {self.names}")
            q = {k: np.asarray(d[k]) for k in ENC_FIELDS + ("lead",)}
            q.update({k: np.asarray(d[k])[:, idx] for k in ("cmds", "V95", "min_h", "fell")})
            parts.append(q)
            n = q["V95"].shape[0]
            self.episode += [Path(p).stem] * n
            self.sentence += [d["sentences"][0]] * n
        self.n = 0
        if parts:
            self.arr = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
            self.n = self.arr["V95"].shape[0]

    def actions(self) -> np.ndarray:
        """[N, D, A] for a 1-tick critic, the summed [N, D, A] for a 3-tick one, [N, D, 12, A] for a chunk."""
        c = self.arr["cmds"]                                            # [N, D, 12, A]
        if self.ticks == 1:
            return c[:, :, 0]
        return c[:, :, :CHUNK] if self.ticks == CHUNK else c[:, :, : self.ticks].sum(2)


def section4(Q: np.ndarray, V: np.ndarray, tol: float) -> dict:
    """Q, V [N, D]: rank agreement over pairs of one state's decisions whose truth differs by more than tol, and
    the spread ratio (spread of Q over a state's decisions / spread of their truth, as means over the states)."""
    ok = n = 0
    D = V.shape[1]
    for i in range(D):
        for j in range(i + 1, D):
            m = np.abs(V[:, i] - V[:, j]) > tol
            n += int(m.sum())
            ok += int((np.sign(Q[m, i] - Q[m, j]) == np.sign(V[m, i] - V[m, j])).sum())
    qs, vs = Q.max(1) - Q.min(1), V.max(1) - V.min(1)
    return {"n_states": int(V.shape[0]), "n_pairs": n, "rank_agree": (ok / n) if n else None,
            "q_spread": float(qs.mean()), "true_spread": float(vs.mean()),
            "spread_ratio": float(qs.mean() / max(vs.mean(), 1e-9)), "mae": float(np.abs(Q - V).mean())}


@torch.no_grad()
def score_states(tr, S: States, rand_u: np.ndarray | None = None, bs: int = 16) -> dict:
    """Q (min of the twins) of every scored decision at every state [N, D], and at N_RAND random actions."""
    was = tr.policy_enc.training
    tr.policy_enc.eval(), tr.critics.eval()
    acts = S.actions()
    D = acts.shape[1]
    chunk = tr.chunk
    Q = np.zeros((S.n, D), np.float32)
    QR = np.zeros((S.n, N_RAND), np.float32) if rand_u is not None else None
    for c0 in range(0, S.n, bs):
        sl = slice(c0, min(S.n, c0 + bs))
        sub = {k: S.arr[k][sl] for k in ENC_FIELDS}
        n = sub["dtheta_max"].shape[0]
        words, wmask = tr.text.lookup(S.sentence[sl], splits=["probe"] * n)
        enc, Jlin, Jang, ji = tr.encode(sub, words, wmask)
        groups = [_t(acts[sl][:, d], dev=tr.dev) for d in range(D)]
        if rand_u is not None:
            dm = _t(sub["dtheta_max"], dev=tr.dev)
            for r in range(N_RAND):
                u = _t(rand_u[sl, r], dev=tr.dev)                        # [n, T, A] in [-1, 1]
                groups.append(u[:, 0] * dm * tr.H if chunk == 1 else u * dm[:, None])
        for g0 in range(0, len(groups), 4):
            grp = groups[g0:g0 + 4]
            m = len(grp)
            idx = torch.arange(n, device=tr.dev).repeat(m)
            rep = lambda x: x.repeat(m, *([1] * (x.dim() - 1)))  # noqa: E731
            q1, q2, _, _ = tr.critics(enc.index(idx), torch.cat(grp, 0), rep(Jlin), rep(Jang), rep(ji))
            qm = torch.minimum(q1, q2).view(m, n).T.float().cpu().numpy()
            for a in range(m):
                k = g0 + a
                if k < D:
                    Q[sl, k] = qm[:, a]
                else:
                    QR[sl, k - D] = qm[:, a]
    tr.policy_enc.train(was), tr.critics.train(was)
    return {"Q": Q, "QR": QR, "V": S.arr["V95"].astype(np.float32), "names": list(S.names)}


def rand_actions(n: int, chunk: int, A: int, seed: int = 20261110) -> np.ndarray:
    """[n, N_RAND, T, A] in [-1, 1] (x dtheta_max = commands). One tick: uniform in the box. A chunk: the first half
    hold one uniform command for all its ticks (the widest cumulative steps), the rest draw every tick afresh."""
    rng = np.random.default_rng(seed)
    T = CHUNK if chunk == CHUNK else 1     # a 3-tick action is one summed command
    u = rng.uniform(-1, 1, size=(n, N_RAND, T, A)).astype(np.float32)
    u[:, : N_RAND // 2] = u[:, : N_RAND // 2, :1]
    return u


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


class CriticTrainer(LangTrainer):
    def __init__(self, cfg, run, roots, caches, seed: int = 0, ticks: int = 0):
        super().__init__(cfg, run, roots, (), caches, tasks=(T5,), embodiments=("ur5-wsg",), use_wandb=False,
                         seed=seed, prefetch=False)
        self.chunk = int(cfg.critic_chunk)       # the model's: 12 (per-tick commands) or 1 (one action)
        self.H = int(ticks) if ticks else self.chunk   # ticks one decision covers: 1, 3 or 12
        assert self.H == self.chunk or self.chunk == 1, (self.H, self.chunk)
        if cfg.critic_injection == "revised":
            sc = json.loads(NORM.read_text())
            k = float(self.H) if self.chunk == 1 else 1.0   # a summed action's bounds: H x one tick's
            self.critics.set_action_ref([k * float(v) for v in np.asarray(sc["dtheta_max"]).reshape(-1)],
                                        k * sc["s_p"], k * sc["s_w"])
            self.critics_targ.load_state_dict(self.critics.state_dict())
        self.batcher = None
        self._src = {s.name: s for s in self.sampler.sources}
        self._dth = {}
        self.measured = None          # {source name: V [n, R]} for a measured-outcome target (measured_table)
        self.loss_kind = "mse"        # or "huber" (HUBER_DELTA)

    def add_source_index(self, sampler):
        for s in sampler.sources:
            self._src[s.name] = s

    def stored_next(self, b) -> np.ndarray:
        """The stored command of row t+1 (the 1-tick target's action)."""
        a = np.empty_like(np.asarray(b["dtheta"]))
        for k, (name, row) in enumerate(zip(b["source"], b["next_row"])):
            arr = self._dth.get(name)
            if arr is None:
                s = self._src[name]
                arr = self._dth[name] = np.asarray(s.step["dtheta"][: int(s.rs.buf.n)], np.float32)
            a[k] = arr[int(row)]
        return a

    def grad_step(self, b=None, log: bool = False) -> dict:
        if b is None:
            b = self.batcher.next()
        t0 = self.timer.mark()
        words, wmask = self.text.lookup(b["sentence"], splits=["train"] * len(b["sentence"]))
        self._step_t0 = t0
        with matmul_tf32(getattr(self.cfg, "tf32_train", False)):
            return self._grad_step(b, words, wmask, log)

    def act_of(self, cmds):
        """Per-tick commands [B, H, A] -> the model's action: the chunk itself, or the summed command (3 ticks)."""
        return cmds if self.chunk > 1 else cmds.sum(1)

    def measured_y(self, b) -> np.ndarray:
        """The measured-outcome target of every sample: V95 of its rule column from its row to its episode's end."""
        y = np.array([self.measured[name][int(r), int(c)] for name, r, c in zip(b["source"], b["row"], b["rule_col"])],
                     np.float32)
        if not np.isfinite(y).all():
            raise RuntimeError(f"{int((~np.isfinite(y)).sum())} samples without a measured value (rows of episodes "
                               "left out of the table were drawn)")
        return y

    def _target(self, b, words, wmask, h):
        g, d = self.gamma(), self.dev
        if self.measured is not None:
            return _t(self.measured_y(b), dev=d)
        if self.H == 1:
            enc_n, Jl, Ja, ji = self.encode(b, words, wmask, "_next", target=True)
            a_n = _t(self.stored_next(b), dev=d)
            q1, q2, _, _ = self.critics_targ(enc_n, a_n, Jl, Ja, ji)
            return (1 - g) * h + g * torch.minimum(h, 0.5 * (q1 + q2))
        hs = _t(b["chunk_h"], dev=d)                                   # [B, H + 1]
        L = _t(b["chunk_L"], torch.int64, d)
        boot = _t(b["chunk_boot"], torch.bool, d)
        y = hs.gather(1, L[:, None]).squeeze(1)                        # h at the recursion's last row
        if bool(boot.any()):
            idx = torch.nonzero(boot).squeeze(1)
            sub = {f: np.asarray(b[f + "_end"])[boot.cpu().numpy()] for f in ENC_FIELDS}
            enc_e, Jl, Ja, ji = self.encode(sub, words[idx], wmask[idx], target=True)
            q1, q2, _, _ = self.critics_targ(enc_e, self.act_of(_t(b["chunk_cmds_next"], dev=d)[idx]), Jl, Ja, ji)
            y = y.clone()
            y[idx] = 0.5 * (q1 + q2).to(y.dtype)
        for k in range(self.H - 1, -1, -1):
            yk = (1 - g) * hs[:, k] + g * torch.minimum(hs[:, k], y)
            y = torch.where(k < L, yk, y)
        return y

    def _grad_step(self, b: dict, words, wmask, log: bool = False) -> dict:
        d = self.dev
        enc, Jlin, Jang, ji = self.encode(b, words, wmask)
        h = _t(b["h"], dev=d)
        act = _t(b["dtheta"], dev=d) if self.H == 1 else self.act_of(_t(b["chunk_cmds"], dev=d))
        with torch.no_grad():
            y = self._target(b, words, wmask, h)
        q1, q2, _, _ = self.critics(enc, act, Jlin, Jang, ji)
        if self.loss_kind == "huber":
            loss = F.huber_loss(q1, y, delta=HUBER_DELTA) + F.huber_loss(q2, y, delta=HUBER_DELTA)
        else:
            loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_e.zero_grad(set_to_none=True)
        self.opt_c.zero_grad(set_to_none=True)
        loss.backward()
        ge = float(torch.nn.utils.clip_grad_norm_(self.policy_enc.parameters(), 10.0))
        gc = float(torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0))
        self.opt_e.step()
        self.opt_c.step()
        with torch.no_grad():
            tau = self.cfg.tau
            for tp, sp in self.target_pairs:
                tp.mul_(1 - tau).add_(tau * sp)
        if self._step_t0 is not None:
            self.timer.add("step", self._step_t0, self.timer.mark())
            self._step_t0 = None
        self.step += 1
        out = {"loss/q": float(loss), "grad/encoder": ge, "grad/critic": gc}
        if log:
            with torch.no_grad():   # the same states with the batch's actions shuffled: how much Q follows the action
                perm = torch.randperm(act.shape[0], device=d)
                qp = self.critics.qmin(enc.detach(), act[perm], Jlin, Jang, ji)
                qm = torch.minimum(q1, q2)
                br = _t(b["_branch"], torch.bool, d)
                dc = _t(b["_kind"], torch.int64, d) == 2
                out.update({"q/mean": float(qm.mean()), "y/mean": float(y.mean()), "h/mean": float(h.mean()),
                            "critic/action_sens": float((qm - qp).abs().mean()),
                            "q/abs_err": float((qm - y).abs().mean()),
                            "q/abs_err_branch": float((qm - y).abs()[br].mean()) if bool(br.any()) else float("nan"),
                            "q/abs_err_decision": float((qm - y).abs()[dc].mean()) if bool(dc.any()) else float("nan"),
                            "data/branch_frac": float(br.float().mean()),
                            "data/decision_frac": float(dc.float().mean())})
                if self.H > 1:
                    out["target/boot_frac"] = float(np.mean(b["chunk_boot"]))
        return out

    def ckpt_dict(self) -> dict:
        dct = super().ckpt_dict()
        dct["asens"] = {"chunk": self.chunk, "ticks": self.H, "injection": self.cfg.critic_injection}
        return dct


def make_trainer(name: str, run: Path, seed: int, with_batcher: bool = True, branch_root: Path = BRANCH):
    import copy

    from main.lang.data import LangSampler

    R = RUNS[name]
    ticks = int(R["decision_ticks"])
    cfg = run_cfg(ticks, R["injection"])
    roots = roots_of([str(p) for p in NORMAL])
    caches = caches_of([str(p) for p in (REPO / "data/lang_1023/warmup", REPO / "data/lang_1023/val",
                                         REPO / "data/lang_1023/val20")])
    tr = CriticTrainer(cfg, run, roots, caches, seed=seed, ticks=ticks)
    tr.loss_kind = R.get("loss", "mse")
    B = int(cfg.batch_size)
    measured = R.get("target") == "measured"
    if measured:      # plan section 8.4: normal rows only from complete planner episodes, V95 of their own h trace
        tr.measured, keep, n_ep, why_all = {}, [], 0, {}
        for s in tr.sampler.sources:
            ok, why = planner_episodes(s)
            tr.measured[s.name] = measured_table(s, tr.gamma(), ok)
            keep.append(np.isfinite(tr.measured[s.name][s.rows, 0]))
            n_ep += len(ok)
            for k, v in why.items():
                why_all[k] = why_all.get(k, 0) + v
        keep = np.concatenate(keep)
        n_before = int(tr.sampler.ok.sum())
        restrict(tr.sampler, keep)
        print(f"[asens] measured target: normal data keeps {n_ep} complete planner episodes of {T5}, "
              f"{int(tr.sampler.ok.sum())} of {n_before} trainable normal rows; left out: {why_all}", flush=True)
    parts = [(tr.sampler, B, 0)]
    if R["branch_data"]:
        dirs = branch_dirs(R["branch_data"], branch_root)
        broots = roots_of([str(d / "branches") for d in dirs])
        if not broots:
            raise SystemExit(f"no branch shards under {[str(d) for d in dirs]}")
        bs = LangSampler(broots, cfg, tasks=(T5,), embodiments=("ur5-wsg",), seed=seed + 5, n_threads=8)
        keep, dec, w = branch_masks(bs)
        tr.add_source_index(bs)
        if measured:   # every branch is stored to its end: V95 from every row
            for s in bs.sources:
                if s.name in tr.measured:
                    raise RuntimeError(f"two data roots named {s.name}")
                tr.measured[s.name] = measured_table(s, tr.gamma(), None)
        n_ep = len({(j, int(e)) for j, s in enumerate(bs.sources) for e in np.unique(s.eid)})
        f_norm, f_cont, f_dec = MIX[R["mix"]]
        n_dec = int(round(f_dec * B))
        if f_cont is None:          # first round: a quarter of the batch from any branch row
            restrict(bs, keep)
            parts = [(tr.sampler, B - n_dec, 0), (bs, n_dec, 1)]
            n_rows = (len(bs), int((bs.ok & dec).sum()))
        else:
            cont, dcs = copy.copy(bs), copy.copy(bs)
            restrict(cont, keep & ~dec)
            restrict(dcs, keep & dec)
            n_rows = (len(cont), len(dcs))
            reweight(dcs, w)
            n_cont = int(round(f_cont * B))
            parts = [(tr.sampler, B - n_cont - n_dec, 0), (cont, n_cont, 1), (dcs, n_dec, 2)]
        print(f"[asens] branch data ({R['branch_data']}, mix {R['mix']}): {n_ep} branch episodes in "
              f"{len(bs.sources)} shards (with the held-out ones); training rows: "
              + (f"{n_rows[0]} branch rows, {n_rows[1]} of them decision rows" if f_cont is None else
                 f"{n_rows[0]} continuation, {n_rows[1]} decision") + f"; batch parts {[p[1] for p in parts]}",
              flush=True)
        if not n_rows[0] or (f_cont is not None and not n_rows[1]):
            raise SystemExit("the branch shards have no training rows")
    if with_batcher:
        tr.batcher = Batcher(tr, parts, tr.H, seed + 9, need_end=not measured)
    return tr


def probe_files(dirs) -> tuple[list, list]:
    """(held-out, training) states files of some data roots."""
    held, train = [], []
    for d in dirs:
        for p in sorted((Path(d) / "states").glob("s*.pt")):
            meta = torch.load(p, map_location="cpu", weights_only=False)["meta"]
            (held if meta.get("held_out") else train).append(p)
    return held, train


def probe_dirs(name: str, small_root: Path = BRANCH) -> list:
    """Where a run's probe states come from: its branch data's roots; both roots for a 3-tick critic (the first
    round has no 3-tick decisions) and for a run without branch data, the small root."""
    R = RUNS[name]
    if int(R["decision_ticks"]) == 3 or R["branch_data"] == "large":
        return [Path(small_root), BRANCH2]
    return [Path(small_root)]


def probe_sets(name: str, tr, small_root: Path = BRANCH, train_cap: int | None = None) -> dict:
    """{key: (States, random actions)}. held: the held-out states of the run's roots. train: training states (as
    many files as 2 x the held-out ones, evenly spread, or train_cap). held1: with two roots, the first round's
    held-out states alone, the set the small-data runs are scored on."""
    dirs = probe_dirs(name, small_root)
    held, trn = probe_files(dirs)
    n_tr = train_cap if train_cap is not None else max(len(held), 1) * 2
    if len(trn) > n_tr:
        trn = [trn[i] for i in np.linspace(0, len(trn) - 1, n_tr).round().astype(int)] if len(dirs) > 1 else trn[:n_tr]
    groups = [("held", held), ("train", trn)]
    if len(dirs) > 1:
        groups.append(("held1", probe_files(dirs[:1])[0]))
    sets = {}
    for k, (key, files) in enumerate(groups):
        S = States(files, tr.H) if files else None
        if S is not None and S.n:
            sets[key] = (S, rand_actions(S.n, tr.chunk, S.arr["dtheta_max"].shape[1], seed=20261110 + k))
    return sets


def probe_metrics(tr, S: States, U, tol: float) -> tuple[dict, dict]:
    r = score_states(tr, S, U)
    m = section4(r["Q"].astype(np.float64), r["V"].astype(np.float64), tol)
    m["spread16"] = float((r["QR"].max(1) - r["QR"].min(1)).mean())
    m["q_mean"] = r["Q"].mean(0).tolist()
    return m, r


def write_probe(tr, sets: dict, path: Path, step: int, tol: float) -> dict:
    t0 = time.time()
    out = {"step": int(step)}
    for key, (S, U) in sets.items():
        m, _ = probe_metrics(tr, S, U, tol)
        out.update({f"{key}/{k}": v for k, v in m.items()})
    out["probe_s"] = time.time() - t0
    with open(path, "a") as f:
        f.write(json.dumps(out) + "\n")
    msg = " | ".join(f"{k}: rank {out[f'{k}/rank_agree'] if out[f'{k}/rank_agree'] is not None else float('nan'):.2f} "
                     f"({out[f'{k}/n_pairs']} pairs, {out[f'{k}/n_states']} states) spread ratio "
                     f"{out[f'{k}/spread_ratio']:.3f} spread16 {out[f'{k}/spread16']:.4f} mae {out[f'{k}/mae']:.3f}"
                     for k in sets)
    print(f"[probe] step {step} ({out['probe_s']:.0f} s) {msg}", flush=True)
    return out


def train(args) -> int:
    name = args.name
    run = (Path(args.run) if args.run else OUT / "runs" / name).resolve()
    if not CRITERIA.exists():
        raise SystemExit(f"{CRITERIA} missing: run `train_critic.py criteria` first (the plan fixes the pass numbers "
                         "before step 3)")
    crit = json.loads(CRITERIA.read_text())
    if not args.resume and (run / "state.pt").exists():
        raise SystemExit(f"{run} has a state.pt: pass --resume or use a new run directory")
    t_start = time.time()
    deadline = t_start + args.time_budget if args.time_budget else None
    tr = make_trainer(name, run, args.seed, branch_root=Path(args.branch_root))
    if args.resume and (run / "state.pt").exists():
        tr.load_state()
        print(f"[asens] resumed at step {tr.step}", flush=True)
    sets = probe_sets(name, tr, Path(args.branch_root))
    if not sets:
        print("[asens] no branch states with this decision length yet: probes are skipped (a later job of the run "
              "picks them up)", flush=True)
    diag = run / "diag.jsonl"
    done = set()
    if diag.exists():
        for ln in diag.read_text().splitlines():
            try:
                done.add(int(json.loads(ln)["step"]))
            except Exception:
                pass
    tr.install_signal()
    R = RUNS[name]
    print(f"[asens] run {name}: decision {tr.H} tick(s), injection {tr.cfg.critic_injection}, branch data "
          f"{R['branch_data']}, mix {R['mix']}, target {R.get('target', 'bootstrap')}, loss {tr.loss_kind}; probe states "
          f"{ {k: v[0].n for k, v in sets.items()} }; setup {time.time() - t_start:.0f} s, step {tr.step} -> "
          f"{args.grad_steps}", flush=True)
    tol = float(crit["gt_tol"])
    if tr.step == 0 and 0 not in done and sets:
        write_probe(tr, sets, diag, 0, tol)
    tr.policy_enc.train(), tr.critics.train()
    tr.reset_perf()
    t_loop, s_loop, t_probe = time.time(), tr.step, 0.0
    while tr.step < args.grad_steps:
        do_log = (tr.step + 1) % LOG_EVERY == 0
        out = tr.grad_step(log=do_log)
        if do_log:
            out["perf/wall_s_per_step"] = (time.time() - t_loop - t_probe) / max(tr.step - s_loop, 1)
            tr.log(out)
            if tr.step % 100 == 0:
                print(f"[asens] {name} step {tr.step} loss {out['loss/q']:.4f} q {out.get('q/mean', 0):+.3f} y "
                      f"{out.get('y/mean', 0):+.3f} action sens {out.get('critic/action_sens', 0):.2e} "
                      f"{out['perf/wall_s_per_step']:.2f} s/step", flush=True)
        if tr.step % args.probe_every == 0 and tr.step not in done and sets:
            t0 = time.time()
            write_probe(tr, sets, diag, tr.step, tol)
            done.add(tr.step)
            t_probe += time.time() - t0
        if tr.step % args.state_every == 0:
            tr.save_state()
        if tr._stop or (deadline is not None and time.time() > deadline):
            tr.save_state()
            tr.save_ckpt()
            tr.batcher.stop()
            print(f"[asens] stopping at step {tr.step} ({'signal' if tr._stop else 'time budget'}), "
                  f"{(time.time() - t_loop - t_probe) / max(tr.step - s_loop, 1):.2f} s/step; resume with --resume",
                  flush=True)
            return EXIT_RESUME
    tr.save_state()
    tr.save_ckpt()
    tr.save_ckpt("checkpoint_final.pt")
    tr.batcher.stop()
    print(f"[asens] {name} done at step {tr.step}", flush=True)
    return 0


def score(args) -> int:
    """Every run's final checkpoint on its held-out states (the pass / fail), its training states and, for a run
    with both data roots, the first round's held-out states alone -> scores.json, scores_<run>.npz."""
    from main.lang.train import load_policy

    crit = json.loads(CRITERIA.read_text())
    tol = float(crit["gt_tol"])
    res = {}
    path = OUT / "scores.json"
    if path.exists():
        res = json.loads(path.read_text())
    for name in args.names:
        run = OUT / "runs" / name
        ck = run / "checkpoint_final.pt"
        if not ck.exists():
            print(f"[score] {name}: no checkpoint_final.pt yet", flush=True)
            continue
        tr = make_trainer(name, OUT / "score_tmp" / name, 0, with_batcher=False, branch_root=Path(args.branch_root))
        st = torch.load(ck, map_location="cuda", weights_only=False)
        load_policy(tr.policy_enc, st)
        tr.critics.load_state_dict(st["critics"])
        sets = probe_sets(name, tr, Path(args.branch_root), train_cap=int(args.train_cap))
        if "held" not in sets:
            print(f"[score] {name}: no held-out states with a {tr.H}-tick decision", flush=True)
            continue
        out, npz = {}, {}
        for key, (S, U) in sets.items():
            m, r = probe_metrics(tr, S, U, tol)
            m["decisions"] = r["names"]
            m["q_mean_per_decision"] = m.pop("q_mean")
            m["v_mean_per_decision"] = r["V"].mean(0).tolist()
            out[key] = m
            npz.update({f"{key}_Q": r["Q"], f"{key}_QR": r["QR"], f"{key}_V": r["V"],
                        f"{key}_episode": np.array(S.episode), f"{key}_lead": S.arr["lead"]})
            print(f"[score] {name} {key}: rank agreement {m['rank_agree']} over {m['n_pairs']} pairs "
                  f"({m['n_states']} states), spread ratio {m['spread_ratio']:.3f} (Q {m['q_spread']:.3f} / truth "
                  f"{m['true_spread']:.3f}), spread over 16 random {m['spread16']:.4f}, MAE {m['mae']:.3f}", flush=True)
        m = dict(out["held"])
        m["train"], m["held1"] = out.get("train"), out.get("held1")
        m["step"] = int(st.get("step", -1))
        m["passes"] = bool(m["rank_agree"] is not None and m["rank_agree"] >= crit["rank_agree_min"]
                           and m["spread_ratio"] >= crit["spread_ratio_min"])
        m["run"] = dict(RUNS[name])
        np.savez_compressed(OUT / f"scores_{name}.npz", names=np.array(out["held"]["decisions"]), **npz)
        res[name] = m
        print(f"[score] {name} -> {'PASS' if m['passes'] else 'fail'}", flush=True)
        del tr
        torch.cuda.empty_cache()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(path)
    return 0


def criteria(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if CRITERIA.exists():
        doc = json.loads(CRITERIA.read_text())
        if "round2" not in doc:   # the same numbers; only the list of runs and the 3-tick decisions are new
            doc["round2"] = {"added": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "note": "plan section 7 (2026-11-11): the pass numbers are unchanged. 3-tick critics "
                                     "score plan, t3_c0, t3_c50. Runs with the large branch data are scored on all "
                                     "held-out states (both collections); their score on the first round's held-out "
                                     "states is reported next to it.",
                             "runs": {k: v for k, v in RUNS.items() if k not in doc.get("runs", {})}}
            CRITERIA.write_text(json.dumps(doc, indent=1))
        print(f"{CRITERIA} exists (fixed before step 3):\n{CRITERIA.read_text()}")
        return 0
    doc = {"fixed": time.strftime("%Y-%m-%d %H:%M:%S"), **PASS,
           "rule": "a critic reads the action if, on the held-out branch states, rank agreement >= rank_agree_min "
                   "(pairs of one state's scored decisions whose V95 differ by more than gt_tol; the share Q orders "
                   "as the truth does) AND the spread ratio >= spread_ratio_min (mean over states of max - min of Q "
                   "over the scored decisions, divided by the same for V95). One-tick critics score plan, t1_c0, "
                   "t1_c50; 3-tick critics plan, t3_c0, t3_c50; chunk critics plan, ch_c0, ch_c25, ch_c50, ch_c75. "
                   "Scored at the final checkpoint (6,000 steps).",
           "runs": dict(RUNS),
           "source": "docs/lang_filter/ACTION_SENSITIVITY_PLAN.md sections 4 and 7"}
    CRITERIA.write_text(json.dumps(doc, indent=1))
    print(CRITERIA.read_text())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("criteria")
    p = sub.add_parser("train")
    p.add_argument("--name", choices=sorted(RUNS), required=True)
    p.add_argument("--run", default="")
    p.add_argument("--grad-steps", type=int, default=6000)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--state-every", type=int, default=250)
    p.add_argument("--branch-root", default=str(BRANCH), help="the first round's (small) branch data root")
    p = sub.add_parser("score")
    p.add_argument("--names", nargs="+", default=sorted(RUNS))
    p.add_argument("--branch-root", default=str(BRANCH))
    p.add_argument("--train-cap", type=int, default=60, help="training states files scored per run")
    args = ap.parse_args(argv)
    if args.mode == "criteria":
        return criteria(args)
    if args.mode == "train":
        return train(args)
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
