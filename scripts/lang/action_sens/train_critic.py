#!/usr/bin/env python
"""Steps 3 and 4 of ACTION_SENSITIVITY_PLAN (2026-11-10): train a critic only, score it on the held-out branch states.

    python scripts/lang/action_sens/train_critic.py criteria                      # fix the pass numbers (once)
    python scripts/lang/action_sens/train_critic.py train --name A [--grad-steps 6000] [--time-budget 3000] [--resume]
    python scripts/lang/action_sens/train_critic.py score [--names A B ...]       # -> runs/lang/action_sens/scores.json

The six runs (RUNS): the decision a critic scores (1 tick, or a 12-tick chunk), how the action enters it (current
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
BRANCH = REPO / "data/lang_asens"
OUT = REPO / "runs/lang/action_sens"
NORM = REPO / "runs/lang/diag_1030/retrain/action_norm.json"
CRITERIA = OUT / "criteria.json"
T5 = "tray_transport"
CHUNK = 12
BRANCH_FRAC = 0.25
N_RAND = 16
# name: (decision ticks, injection, branch data)
RUNS = {"A": (1, "linear", True), "B": (1, "revised", True), "C": (CHUNK, "linear", True),
        "D": (CHUNK, "revised", True), "E": (CHUNK, "linear", False), "F": (1, "revised", False)}
ENC_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max",
              "scene_tokens", "scene_pos")
PASS = {"rank_agree_min": 0.70, "spread_ratio_min": 0.25, "gt_tol": 0.05}


def run_cfg(chunk: int, injection: str) -> LangConfig:
    cfg = LangConfig.load(VANILLA / "lang_config.json")
    return dataclasses.replace(cfg, critic_readout="mlp", critic_injection=injection, critic_chunk=int(chunk))


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


class Batcher:
    """Batches of 64: normal rows, plus branch rows as BRANCH_FRAC of the batch. Chunk critics also get, per sample,
    the stored commands of the chunk, the h of its rows, and the state and stored chunk at its end (a second read)."""

    def __init__(self, tr: "CriticTrainer", bsampler, chunk: int, seed: int):
        self.tr, self.chunk = tr, int(chunk)
        self.parts = [(tr.sampler, tr.cfg.batch_size)]
        if bsampler is not None:
            nb = int(round(BRANCH_FRAC * tr.cfg.batch_size))
            self.parts = [(tr.sampler, tr.cfg.batch_size - nb), (bsampler, nb)]
        self.rng = np.random.default_rng(seed)
        self.idx = {}
        if self.chunk > 1:
            for smp, _ in self.parts:
                for s in smp.sources:
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
        for smp, n in self.parts:
            picks = smp.plan(n, self.rng)
            b = smp.sample(len(picks), rng=self.rng, picks=picks)
            b["_branch"] = np.full(len(picks), smp is not self.tr.sampler)
            if self.chunk > 1:
                self._chunk_part(smp, picks, b)
            outs.append(b)
        return concat(outs)

    def _chunk_part(self, smp, picks, b) -> None:
        H = self.chunk
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
            ix = self.idx[id(s)]
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
        if picks2:
            b2 = smp.sample(len(picks2), rng=self.rng, picks=picks2)
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
    """The branch states of some episodes (collect_branches.py's states files), stacked."""

    def __init__(self, files):
        parts, self.episode, self.sentence = [], [], []
        for p in files:
            d = torch.load(p, map_location="cpu", weights_only=False)
            parts.append(d)
            n = d["V95"].shape[0]
            self.episode += [Path(p).stem] * n
            self.sentence += [d["sentences"][0]] * n
        d0 = parts[0]
        self.names, self.H = list(d0["decisions"]), list(d0["decision_H"])
        self.arr = {k: np.concatenate([np.asarray(d[k]) for d in parts]) for k in ENC_FIELDS + ("cmds", "V95", "min_h",
                                                                                               "fell", "lead")}
        self.n = self.arr["V95"].shape[0]

    def decisions(self, chunk: int) -> list[int]:
        """A one-tick critic scores the planner's and the 1-tick decisions, a chunk critic the planner's and the
        chunk decisions."""
        return [i for i, h in enumerate(self.H) if h == 0 or (h == 1) == (chunk == 1)]

    def actions(self, chunk: int) -> np.ndarray:
        c = self.arr["cmds"][:, self.decisions(chunk)]                 # [N, D, 12, A]
        return c[:, :, 0] if chunk == 1 else c[:, :, :chunk]


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
def score_states(tr, S: States, chunk: int, rand_u: np.ndarray | None = None, bs: int = 16) -> dict:
    """Q (min of the twins) of every scored decision at every state [N, D], and at N_RAND random actions."""
    was = tr.policy_enc.training
    tr.policy_enc.eval(), tr.critics.eval()
    acts = S.actions(chunk)
    D = acts.shape[1]
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
                groups.append(u[:, 0] * dm if chunk == 1 else u * dm[:, None])
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
    return {"Q": Q, "QR": QR, "V": S.arr["V95"][:, S.decisions(chunk)].astype(np.float32),
            "names": [S.names[i] for i in S.decisions(chunk)]}


def rand_actions(n: int, chunk: int, A: int, seed: int = 20261110) -> np.ndarray:
    """[n, N_RAND, T, A] in [-1, 1] (x dtheta_max = commands). One tick: uniform in the box. A chunk: the first half
    hold one uniform command for all its ticks (the widest cumulative steps), the rest draw every tick afresh."""
    rng = np.random.default_rng(seed)
    T = max(chunk, 1)
    u = rng.uniform(-1, 1, size=(n, N_RAND, T, A)).astype(np.float32)
    u[:, : N_RAND // 2] = u[:, : N_RAND // 2, :1]
    return u


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


class CriticTrainer(LangTrainer):
    def __init__(self, cfg, run, roots, caches, seed: int = 0):
        super().__init__(cfg, run, roots, (), caches, tasks=(T5,), embodiments=("ur5-wsg",), use_wandb=False,
                         seed=seed, prefetch=False)
        self.chunk = int(cfg.critic_chunk)
        if cfg.critic_injection == "revised":
            sc = json.loads(NORM.read_text())
            self.critics.set_action_ref(sc["dtheta_max"], sc["s_p"], sc["s_w"])
            self.critics_targ.load_state_dict(self.critics.state_dict())
        self.batcher = None
        self._src = {s.name: s for s in self.sampler.sources}
        self._dth = {}

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

    def _target(self, b, words, wmask, h):
        g, d = self.gamma(), self.dev
        if self.chunk == 1:
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
            q1, q2, _, _ = self.critics_targ(enc_e, _t(b["chunk_cmds_next"], dev=d)[idx], Jl, Ja, ji)
            y = y.clone()
            y[idx] = 0.5 * (q1 + q2).to(y.dtype)
        for k in range(self.chunk - 1, -1, -1):
            yk = (1 - g) * hs[:, k] + g * torch.minimum(hs[:, k], y)
            y = torch.where(k < L, yk, y)
        return y

    def _grad_step(self, b: dict, words, wmask, log: bool = False) -> dict:
        d = self.dev
        enc, Jlin, Jang, ji = self.encode(b, words, wmask)
        h = _t(b["h"], dev=d)
        act = _t(b["dtheta"] if self.chunk == 1 else b["chunk_cmds"], dev=d)
        with torch.no_grad():
            y = self._target(b, words, wmask, h)
        q1, q2, _, _ = self.critics(enc, act, Jlin, Jang, ji)
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
                out.update({"q/mean": float(qm.mean()), "y/mean": float(y.mean()), "h/mean": float(h.mean()),
                            "critic/action_sens": float((qm - qp).abs().mean()),
                            "q/abs_err": float((qm - y).abs().mean()),
                            "q/abs_err_branch": float((qm - y).abs()[br].mean()) if bool(br.any()) else float("nan"),
                            "data/branch_frac": float(br.float().mean())})
                if self.chunk > 1:
                    out["target/boot_frac"] = float(np.mean(b["chunk_boot"]))
        return out

    def ckpt_dict(self) -> dict:
        dct = super().ckpt_dict()
        dct["asens"] = {"chunk": self.chunk, "injection": self.cfg.critic_injection}
        return dct


def make_trainer(name: str, run: Path, seed: int, with_batcher: bool = True, branch_root: Path = BRANCH):
    from main.lang.data import LangSampler

    chunk, inj, use_branch = RUNS[name]
    cfg = run_cfg(chunk, inj)
    roots = roots_of([str(p) for p in NORMAL])
    caches = caches_of([str(p) for p in (REPO / "data/lang_1023/warmup", REPO / "data/lang_1023/val",
                                         REPO / "data/lang_1023/val20")])
    tr = CriticTrainer(cfg, run, roots, caches, seed=seed)
    bs = None
    if use_branch:
        broots = roots_of([str(branch_root / "branches")])
        if not broots:
            raise SystemExit(f"no branch shards under {branch_root}/branches")
        bs = LangSampler(broots, cfg, tasks=(T5,), embodiments=("ur5-wsg",), seed=seed + 5, n_threads=8)
        restrict(bs, branch_keep(bs))
        tr.add_source_index(bs)
        n_ep = len({(j, int(e)) for j, s in enumerate(bs.sources) for e in np.unique(s.eid)})
        print(f"[asens] branch data: {len(bs)} training rows of {bs.sid.size} in {len(bs.sources)} shards "
              f"({n_ep} branch episodes with the held-out ones)", flush=True)
        if not len(bs):
            raise SystemExit("the branch shards have no training rows")
    if with_batcher:
        tr.batcher = Batcher(tr, bs, chunk, seed + 9)
    return tr


def probe_sets(branch_root: Path = BRANCH):
    files = sorted((branch_root / "states").glob("s*.pt"))
    held, train = [], []
    for p in files:
        meta = torch.load(p, map_location="cpu", weights_only=False)["meta"]
        (held if meta.get("held_out") else train).append(p)
    return held, train


def write_probe(tr, sets: dict, path: Path, step: int, tol: float) -> dict:
    t0 = time.time()
    out = {"step": int(step)}
    for key, (S, U) in sets.items():
        r = score_states(tr, S, tr.chunk, U)
        m = section4(r["Q"].astype(np.float64), r["V"].astype(np.float64), tol)
        m["spread16"] = float((r["QR"].max(1) - r["QR"].min(1)).mean())
        m["q_mean"] = r["Q"].mean(0).tolist()
        out.update({f"{key}/{k}": v for k, v in m.items()})
    out["probe_s"] = time.time() - t0
    with open(path, "a") as f:
        f.write(json.dumps(out) + "\n")
    msg = " | ".join(f"{k}: rank {out[f'{k}/rank_agree'] if out[f'{k}/rank_agree'] is not None else float('nan'):.2f} "
                     f"({out[f'{k}/n_pairs']} pairs) spread ratio {out[f'{k}/spread_ratio']:.3f} spread16 "
                     f"{out[f'{k}/spread16']:.4f} mae {out[f'{k}/mae']:.3f}" for k in sets)
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
    held, trn = probe_sets(Path(args.branch_root))
    sets = {}
    A = None
    for key, files in (("held", held), ("train", trn[: max(len(held), 1) * 2])):
        if files:
            S = States(files)
            A = S.arr["dtheta_max"].shape[1]
            sets[key] = (S, rand_actions(S.n, tr.chunk, A, seed=20261110 + (key == "train")))
    if not sets:
        print("[asens] no branch states yet: probes are skipped", flush=True)
    diag = run / "diag.jsonl"
    done = set()
    if diag.exists():
        for ln in diag.read_text().splitlines():
            try:
                done.add(int(json.loads(ln)["step"]))
            except Exception:
                pass
    tr.install_signal()
    print(f"[asens] run {name}: decision {tr.chunk} tick(s), injection {tr.cfg.critic_injection}, branch data "
          f"{RUNS[name][2]}; setup {time.time() - t_start:.0f} s, step {tr.step} -> {args.grad_steps}", flush=True)
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
    crit = json.loads(CRITERIA.read_text())
    tol = float(crit["gt_tol"])
    held, _ = probe_sets(Path(args.branch_root))
    if not held:
        raise SystemExit("no held-out branch states")
    S = States(held)
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
        from main.lang.train import load_policy

        load_policy(tr.policy_enc, st)
        tr.critics.load_state_dict(st["critics"])
        U = rand_actions(S.n, tr.chunk, S.arr["dtheta_max"].shape[1])
        r = score_states(tr, S, tr.chunk, U)
        m = section4(r["Q"].astype(np.float64), r["V"].astype(np.float64), tol)
        m["spread16"] = float((r["QR"].max(1) - r["QR"].min(1)).mean())
        m["decisions"] = r["names"]
        m["q_mean_per_decision"] = r["Q"].mean(0).tolist()
        m["v_mean_per_decision"] = r["V"].mean(0).tolist()
        m["step"] = int(st.get("step", -1))
        m["passes"] = bool(m["rank_agree"] is not None and m["rank_agree"] >= crit["rank_agree_min"]
                           and m["spread_ratio"] >= crit["spread_ratio_min"])
        m["run"] = dict(zip(("decision_ticks", "injection", "branch_data"), RUNS[name]))
        np.savez_compressed(OUT / f"scores_{name}.npz", Q=r["Q"], QR=r["QR"], V=r["V"], names=np.array(r["names"]),
                            episode=np.array(S.episode), lead=S.arr["lead"])
        res[name] = m
        print(f"[score] {name}: rank agreement {m['rank_agree']} over {m['n_pairs']} pairs, spread ratio "
              f"{m['spread_ratio']:.3f} (Q {m['q_spread']:.3f} / truth {m['true_spread']:.3f}), spread over 16 random "
              f"{m['spread16']:.4f}, MAE {m['mae']:.3f} -> {'PASS' if m['passes'] else 'fail'}", flush=True)
        del tr
        torch.cuda.empty_cache()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(path)
    return 0


def criteria(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if CRITERIA.exists():
        print(f"{CRITERIA} exists (fixed before step 3):\n{CRITERIA.read_text()}")
        return 0
    doc = {"fixed": time.strftime("%Y-%m-%d %H:%M:%S"), **PASS,
           "rule": "a critic reads the action if, on the held-out branch states, rank agreement >= rank_agree_min "
                   "(pairs of one state's scored decisions whose V95 differ by more than gt_tol; the share Q orders "
                   "as the truth does) AND the spread ratio >= spread_ratio_min (mean over states of max - min of Q "
                   "over the scored decisions, divided by the same for V95). One-tick critics score plan, t1_c0, "
                   "t1_c50; chunk critics plan, ch_c0, ch_c25, ch_c50, ch_c75. Scored at the final checkpoint "
                   "(6,000 steps).",
           "runs": {k: dict(zip(("decision_ticks", "injection", "branch_data"), v)) for k, v in RUNS.items()},
           "source": "docs/lang_filter/ACTION_SENSITIVITY_PLAN.md section 4"}
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
    p.add_argument("--branch-root", default=str(BRANCH))
    p = sub.add_parser("score")
    p.add_argument("--names", nargs="+", default=sorted(RUNS))
    p.add_argument("--branch-root", default=str(BRANCH))
    args = ap.parse_args(argv)
    if args.mode == "criteria":
        return criteria(args)
    if args.mode == "train":
        return train(args)
    return score(args)


if __name__ == "__main__":
    raise SystemExit(main())
