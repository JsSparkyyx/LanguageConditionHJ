#!/usr/bin/env python
"""Check 4 (diagnosis 2026-10-30): retrain the critic with another continuation in the HJ-SAC target.

The vanilla run (runs/lang/rounds_ur5_g095_1023) bootstraps its target through a SAMPLE of the actor at s':

    y = (1 - g) h(s) + g min(h(s), 0.5 (Q1' + Q2')(s', a')),   a' ~ pi(. | s')   (main/lang/train.py _grad_step)

and its critic stayed action-blind from step 20 on (critic/action_sens_rel ~5e-5). This script trains the same model,
from scratch, on the same data (data/lang_1023/warmup + the vanilla run's ring, read only) with the vanilla run's own
lang_config (gamma 0.95, softmin readout, MLP actor, batch 64, lr 3e-4, TF32 train step), changing ONLY a':

    stoch  a' ~ pi(. | s')            as trained (the control)
    det    a' = the actor's mean at s' (tanh(mu) * dtheta_max); the actor is still trained as usual
    data   a' = the stored action of row t+1 (the behaviour policy's own next command: planner, actor or perturbation)
    zero   a' = 0 (freeze)
    stoch_norm  a' ~ pi(. | s') as stoch, but the critics read NORMALIZED action effects (user review point 4): the own
           joint's dtheta / dtheta_max, dp / s_p and dw / s_w, with s_p, s_w the 95th percentiles of the per-link |dp|,
           |dw| at a saturated action a = dtheta_max * sign(u) over the probe states (fixed once: action_norm.json);
           the post-action geometry offset (p + dp, metres) is unchanged. Implemented by wrapping
           main.lang.network.action_effects inside this process only (ActionNorm); the checkpoint carries the scale
           ("diag_action_norm") and score_states.py applies it.

Every --probe-every steps it scores a FIXED probe set (probe_index.json, built once by `build-probe`: 512 validation
samples from data/lang_1023/val + val20, i.e. other scenes, 192 of them T5 drag states; 256 training samples, 64 of
them T5 drag states) and appends one line to <run>/diag.jsonl (metric names: module `probe_metrics`).

    python scripts/lang/diag_1030/critic_retrain.py build-probe
    python scripts/lang/diag_1030/critic_retrain.py train --target det --run runs/lang/diag_1030/retrain/det \\
        --grad-steps 6000 --time-budget 3000 [--resume]
    python scripts/lang/diag_1030/critic_retrain.py score-ckpt --ckpt runs/lang/rounds_ur5_g095_1023/checkpoint_round10.pt \\
        --run runs/lang/diag_1030/retrain/orig_scoring

train exits 75 after --time-budget seconds (state saved; rerun with --resume). Checkpoints are LangTrainer.ckpt_dict
(the round runs' checkpoint.pt format) plus "diag_target".
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from main.lang.config import CANON_TASKS, LangConfig  # noqa: E402
from main.lang.train import (EXIT_RESUME, LOG_EVERY, LangTrainer, _t, caches_of, code_identity,  # noqa: E402
                             roots_of, trained_subsets)

VANILLA = REPO / "runs/lang/rounds_ur5_g095_1023"
DATA = [REPO / "data/lang_1023/warmup", VANILLA / "ring"]
VAL = [REPO / "data/lang_1023/val", REPO / "data/lang_1023/val20"]
OUT = REPO / "runs/lang/diag_1030/retrain"
PROBE = OUT / "probe_index.json"
TARGETS = ("stoch", "det", "data", "zero", "stoch_norm", "stoch_mlp")
# stoch_mlp (exploratory, added 16:15 after the smoke; not part of the pre-registered verdict): the stochastic target with
# the critic's MLP readout (critic_readout "mlp", decision 6) instead of the vanilla run's softmin readout. Reason: in
# every smoke checkpoint the softmin minimum sat on a QUERY head in 100% of probe states, whose value hardly depends on
# the action, while the body heads (which receive the action injection) did vary with it.
NORM_FILE = OUT / "action_norm.json"
PROBE_FIELDS = ("state_tokens", "body_tokens", "link_pos", "body_mask", "joint_index", "Jlin", "Jang", "dtheta_max",
                "scene_tokens", "scene_pos", "dtheta")
N_RAND = 16
# training-batch metrics averaged over the logged steps (every 20) since the last probe -> diag line "batch/<key>"
BATCH_KEYS = ("loss/q", "loss/pi", "critic/action_sens_rel", "entropy", "actor/action_std", "actor/sat_frac",
              "data/action_std", "q/mean", "h/mean", "y/mean", "lang/pair_rank_acc", "grad/actor", "alpha",
              "target/a_next_frac", "target/next_actor", "target/next_pert", "perf/step_ms")
DT = 0.04
T5 = "tray_transport"


def vanilla_cfg() -> LangConfig:
    return LangConfig.load(VANILLA / "lang_config.json")


def data_args():
    roots = roots_of([str(p) for p in DATA])
    vals = roots_of([str(p) for p in VAL])
    caches = caches_of([str(p) for p in DATA]) + caches_of([str(p) for p in VAL])
    return roots, vals, caches


# ---------------------------------------------------------------------------
# stoch_norm: normalized action inputs of the critics (this process only)
# ---------------------------------------------------------------------------


class ActionNorm:
    """Wraps main.lang.network.action_effects (the critic head's step 1, looked up as a module global at call time):
    eff [B, M + n_res, 8] = [own dtheta, dp (3), dw (3), valid] -> [own dtheta / dtheta_max(own joint), dp / s_p,
    dw / s_w, valid]; the residual tokens' zeros and dp_full (the post-action geometry offset) unchanged."""

    scale: dict | None = None
    _orig = None
    _dev_cache: dict = {}

    @classmethod
    def install(cls, scale: dict | None) -> None:
        import main.lang.network as net

        if cls._orig is None:
            cls._orig = net.action_effects
            net.action_effects = cls.effects
        cls.scale = None if scale is None else {"dtheta_max": [float(x) for x in scale["dtheta_max"]],
                                                "s_p": float(scale["s_p"]), "s_w": float(scale["s_w"])}
        cls._dev_cache = {}

    @classmethod
    def effects(cls, dtheta, Jlin, Jang, joint_index, n_res: int = 2, dtype=None):
        eff, dp_full = cls._orig(dtheta, Jlin, Jang, joint_index, n_res, dtype)
        S = cls.scale
        if S is None:
            return eff, dp_full
        key = (str(eff.device), eff.dtype)
        c = cls._dev_cache.get(key)
        if c is None:
            dm = torch.tensor(S["dtheta_max"], device=eff.device, dtype=eff.dtype)
            dm = torch.where(dm > 0, dm, torch.ones_like(dm))
            c = cls._dev_cache[key] = (dm, 1.0 / float(S["s_p"]), 1.0 / float(S["s_w"]))
        dm, ip, iw = c
        B = eff.shape[0]
        ji = joint_index if joint_index.dim() == 2 else joint_index.unsqueeze(0).expand(B, -1)
        ji = ji.to(eff.device).long()
        M = ji.shape[1]
        own = dm[ji.clamp_min(0)]                                                  # [B, M]
        head = torch.cat([eff[:, :M, :1] / own.unsqueeze(-1), eff[:, :M, 1:4] * ip, eff[:, :M, 4:7] * iw,
                          eff[:, :M, 7:]], dim=-1)
        return torch.cat([head, eff[:, M:]], dim=1), dp_full


def compute_norm(P) -> dict:
    """s_p, s_w: 95th percentiles of per-link |J_lin a|, |J_ang a| over the probe states' valid body links (body_mask,
    the 20 link tokens), a = dtheta_max * sign(u) with u the probe's first fixed random action."""
    dmax = P.arr["dtheta_max"].astype(np.float64)                      # [n, A]
    if np.abs(dmax - dmax[:1]).max() > 1e-6:
        print(f"[norm] WARNING dtheta_max differs across probe states (max dev {np.abs(dmax - dmax[:1]).max():.2e}); "
              "using the per-joint max", flush=True)
    dm = dmax.max(0)
    a = dmax * np.sign(P.U[:, 0, :dmax.shape[1]])
    Jl, Ja = P.arr["Jlin"].astype(np.float32), P.arr["Jang"].astype(np.float32)   # [n, L, 3, A]
    dp = np.linalg.norm(np.einsum("nlca,na->nlc", Jl, a.astype(np.float32)), axis=-1)
    dw = np.linalg.norm(np.einsum("nlca,na->nlc", Ja, a.astype(np.float32)), axis=-1)
    L = dp.shape[1]
    m = np.asarray(P.arr["body_mask"])[:, :L].astype(bool) & (dp > 0)
    s_p, s_w = float(np.percentile(dp[m], 95)), float(np.percentile(dw[m], 95))
    return {"dtheta_max": dm.tolist(), "s_p": s_p, "s_w": s_w, "n_states": int(P.n), "n_links": int(m.sum()),
            "dp_median": float(np.median(dp[m])), "dw_median": float(np.median(dw[m])),
            "rule": "own dtheta / dtheta_max; dp / s_p; dw / s_w; s_* = 95th pct of per-link |dp|, |dw| at "
                    "a = dtheta_max * sign(u_0) over the probe states (body_mask links)"}


def load_or_make_norm(P, run: Path) -> dict:
    """The scale is fixed once (NORM_FILE) and copied into the run dir; a resumed run reads its own copy."""
    mine = run / "action_norm.json"
    if mine.exists():
        return json.loads(mine.read_text())
    if NORM_FILE.exists():
        sc = json.loads(NORM_FILE.read_text())
    else:
        sc = compute_norm(P)
        tmp = NORM_FILE.with_name(f"{NORM_FILE.name}.{time.monotonic_ns()}.tmp")
        tmp.write_text(json.dumps(sc, indent=1))
        tmp.replace(NORM_FILE)
        sc = json.loads(NORM_FILE.read_text())
    mine.write_text(json.dumps(sc, indent=1))
    return sc


# ---------------------------------------------------------------------------
# the trainer: LangTrainer with the target continuation as a switch
# ---------------------------------------------------------------------------


class RetrainTrainer(LangTrainer):
    def __init__(self, *a, target: str = "stoch", **kw):
        super().__init__(*a, **kw)
        if target not in TARGETS:
            raise ValueError(f"--target must be one of {TARGETS}, got {target}")
        c = self.cfg
        if c.aux_target_loss or c.aux_attention_loss or c.scene_fusion or c.qwen or c.oracle_ref_tag:
            raise ValueError("this diagnostic copies the vanilla _grad_step: no aux losses, fusion, Qwen or oracle tag")
        self.target = target
        self._src = {s.name: s for s in self.sampler.sources}
        self._dtheta_ram: dict[str, np.ndarray] = {}
        self._asrc_ram: dict[str, np.ndarray] = {}
        self._tstat = collections.defaultdict(float)
        self.action_norm: dict | None = None

    # the stored command of row t+1 (data target)
    def _stored(self, name: str, kind: str) -> np.ndarray:
        s = self._src[name]
        if kind == "dtheta":
            arr = self._dtheta_ram.get(name)
            if arr is None:
                arr = self._dtheta_ram[name] = np.asarray(s.step["dtheta"][: s.capacity], np.float32)
            return arr
        arr = self._asrc_ram.get(name)
        if arr is None:
            arr = self._asrc_ram[name] = np.asarray(
                np.load(s.root / "buffer" / "action_source.npy", mmap_mode="r")[: s.capacity]).astype(np.int64)
        return arr

    def stored_next(self, b) -> tuple[np.ndarray, np.ndarray]:
        A = np.asarray(b["dtheta"]).shape[-1]
        a = np.empty((len(b["source"]), A), np.float32)
        src = np.empty(len(b["source"]), np.int64)
        for k, (name, row) in enumerate(zip(b["source"], b["next_row"])):
            a[k] = self._stored(name, "dtheta")[int(row)]
            src[k] = self._stored(name, "asrc")[int(row)]
        return a, src

    def target_action(self, b, enc_n, ji_n, dmax_n):
        if self.target in ("stoch", "stoch_norm", "stoch_mlp"):
            return self.actor(enc_n.body, ji_n, dmax_n)[0]
        if self.target == "det":
            return self.actor(enc_n.body, ji_n, dmax_n, deterministic=True)[0]
        if self.target == "data":
            a, src = self.stored_next(b)
            self._tstat["next_actor"] += float((src == 1).mean())
            self._tstat["next_pert"] += float((src == 2).mean())
            return _t(a, dev=self.dev)
        return torch.zeros_like(dmax_n)

    def _grad_step(self, b: dict, words, wmask, log: bool = False) -> dict:
        """main.lang.train.LangTrainer._grad_step (no aux branches: off in this config), a' from target_action."""
        cfg = self.cfg
        enc, Jlin, Jang, ji = self.encode(b, words, wmask)
        d = self.dev
        h = _t(b["h"], dev=d)
        dtheta, dmax = _t(b["dtheta"], dev=d), _t(b["dtheta_max"], dev=d)
        gamma, alpha = self.gamma(), self.alpha()
        with torch.no_grad():
            enc_n, Jlin_n, Jang_n, ji_n = self.encode(b, words, wmask, "_next", target=True)
            dmax_n = _t(b["dtheta_max_next"], dev=d)
            a_n = self.target_action(b, enc_n, ji_n, dmax_n)
            q1_t, q2_t, _, _ = self.critics_targ(enc_n, a_n, Jlin_n, Jang_n, ji_n)
            y = (1 - gamma) * h + gamma * torch.minimum(h, 0.5 * (q1_t + q2_t))
            jm_n = (dmax_n > 0).float()
            self._tstat["a_next_frac"] += float(((a_n.abs() / dmax_n.clamp_min(1e-9)) * jm_n).sum()
                                                / jm_n.sum().clamp_min(1))
            self._tstat["n"] += 1
        q1, q2, _, _ = self.critics(enc, dtheta, Jlin, Jang, ji)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        loss = loss_q
        self.opt_e.zero_grad(set_to_none=True)
        self.opt_c.zero_grad(set_to_none=True)
        loss.backward()
        gn = {}
        if log:
            from main.lang.metrics import grad_norm

            gn["grad/text_adapter"] = grad_norm(self.policy_enc.text_adapter.parameters())
        gn["grad/encoder"] = float(torch.nn.utils.clip_grad_norm_(self.policy_enc.parameters(), 10.0))
        gn["grad/critic"] = float(torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0))
        self.opt_e.step()
        self.opt_c.step()
        frozen = list(self.policy_enc.parameters()) + list(self.critics.parameters())
        for p in frozen:
            p.requires_grad_(False)
        enc_d = enc.detach()
        a_pi, logp, n_act, pre = self.actor(enc_d.body, ji, dmax)
        q1p, q2p, _, _ = self.critics(enc_d, a_pi, Jlin, Jang, ji)
        jv = (ji >= 0)
        pen = (pre.pow(2) * jv).sum(-1) / jv.sum(-1).clamp_min(1)
        loss_pi = (-torch.minimum(q1p, q2p) + alpha * logp + 1e-3 * pen).mean()
        self.opt_a.zero_grad(set_to_none=True)
        loss_pi.backward()
        gn["grad/actor"] = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0))
        self.opt_a.step()
        for p in frozen:
            p.requires_grad_(True)
        with torch.no_grad():
            tau = cfg.tau
            for tp, sp in self.target_pairs:
                tp.mul_(1 - tau).add_(tau * sp)
        if self._step_t0 is not None:
            self.timer.add("step", self._step_t0, self.timer.mark())
            self._step_t0 = None
        self.step += 1
        out = {"loss/q": float(loss_q), "loss/pi": float(loss_pi), "alpha": alpha, "gamma": gamma, **gn}
        if log:
            out.update(self._diagnostics(b, enc_d, q1, q2, y, h, logp, dtheta, dmax, Jlin, Jang, ji))
            n = max(self._tstat.pop("n", 0.0), 1.0)
            for k in list(self._tstat):
                out[f"target/{k}"] = self._tstat.pop(k) / n
        return out

    def ckpt_dict(self) -> dict:
        d = super().ckpt_dict()
        d["diag_target"] = self.target
        if self.action_norm is not None:
            d["diag_action_norm"] = dict(self.action_norm)
        return d


# ---------------------------------------------------------------------------
# the fixed probe set
# ---------------------------------------------------------------------------


def _offsets(sampler) -> list[int]:
    off, o = [], 0
    for s in sampler.sources:
        off.append(o)
        o += int(s.rows.size)
    return off


def _tray_speed(s, i: int) -> float:
    """Horizontal speed (m/s) of the T5 tray between rows t and t+1 (item_pose of the record's tray object)."""
    rec = s.rs.records[int(s.eid[i])]
    k = next((int(o["id"]) for o in rec["objects"] if o.get("category_key") == "tray" or o.get("item_id") == "tray"),
             None)
    if k is None:
        return 0.0
    A = s.rs.buf.arrays
    slot, nxt = int(s.rows[i]), int(s.next[i])
    if not (bool(A["item_valid"][slot, k]) and bool(A["item_valid"][nxt, k])):
        return 0.0
    p0, p1 = np.asarray(A["item_pose"][slot, k, :2], np.float64), np.asarray(A["item_pose"][nxt, k, :2], np.float64)
    return float(np.linalg.norm(p1 - p0) / DT)


def _size_cols(s, i: int) -> dict:
    """{"tall": col, "short": col} of the T5 upright rules (sampleable) of row i's episode."""
    eid = int(s.eid[i])
    rec = s.rs.records[eid]
    out = {}
    for c in s.rs._sample_cols[eid].tolist():
        r = rec["rules"][int(c)]
        o = (r.get("subject") or {}).get("object")
        if r["consequence"] == "upright" and o is not None:
            size = rec["objects"][int(o)].get("size")
            if size in ("tall", "short"):
                out[size] = int(c)
    return out


def _asrc(s) -> np.ndarray:
    return np.asarray(np.load(s.root / "buffer" / "action_source.npy", mmap_mode="r")[: s.capacity]).astype(np.int64)


def _t5_drag_picks(sampler, n_states: int, rng, planner_only: bool = True, sources=None) -> list[tuple]:
    """(gi, col, pair) for n_states T5 drag states (tray moving > 5 cm/s, planner tick), each with its tall- and
    short-block upright rule (a pair), spread over episodes (round robin)."""
    off = _offsets(sampler)
    t5 = CANON_TASKS.index(T5)
    by_ep = collections.defaultdict(list)
    for j, s in enumerate(sampler.sources):
        if sources is not None and s.name not in sources:
            continue
        asrc = _asrc(s)
        for i in np.flatnonzero(s.task == t5).tolist():
            gi = off[j] + i
            if not sampler.ok[gi]:
                continue
            if planner_only and asrc[int(s.rows[i])] != 0:
                continue
            if _tray_speed(s, i) > 0.05:
                cols = _size_cols(s, i)
                if "tall" in cols and "short" in cols:
                    by_ep[(j, int(s.eid[i]))].append((gi, cols))
    eps = sorted(by_ep)
    rng.shuffle(eps)
    for e in eps:
        rng.shuffle(by_ep[e])
    chosen, k = [], 0
    while len(chosen) < n_states and any(by_ep[e] for e in eps):
        e = eps[k % len(eps)]
        if by_ep[e]:
            chosen.append(by_ep[e].pop())
        k += 1
    picks = []
    for p, (gi, cols) in enumerate(chosen):
        picks.append((gi, cols["tall"], p))
        picks.append((gi, cols["short"], p))
    print(f"[probe] T5 drag: {len(chosen)} states from {len(eps)} episodes", flush=True)
    return picks


def _entries(sampler, picks, set_name: str, t5drag: bool, pair_base: int, rng) -> list[dict]:
    from main.item_envs import language

    out = []
    srcs = sampler.sources
    asrc = {s.name: _asrc(s) for s in srcs}
    for gi, col, p in picks:
        j, i = int(sampler.sid[gi]), int(sampler.loc[gi])
        s = srcs[j]
        eid = int(s.eid[i])
        rec = s.rs.records[eid]
        rule = rec["rules"][int(col)]
        slot = int(s.rows[i])
        o = (rule.get("subject") or {}).get("object")
        out.append({
            "set": set_name, "root": str(s.root), "source": s.name, "slot": slot, "local": i, "col": int(col),
            "pair": (pair_base + int(p)) if p >= 0 else -1, "t5drag": bool(t5drag),
            "sentence": language.sample_sentence(rule, rec, rng, "train", sampler.bank),
            "task": rec["task"], "rule_id": rule["rule_id"], "consequence": rule["consequence"],
            "size": rec["objects"][int(o)].get("size") if (o is not None and rec["task"] == T5) else None,
            "episode_id": eid, "t_in_ep": int(s.t[i]), "action_source": int(asrc[s.name][slot]),
            "fails_soon": bool(s.rs.buf.arrays["fails_soon"][slot, int(col)]),
            "h": float(s.rs.buf.arrays["h_rules"][slot, int(col)]),
            "tray_speed": _tray_speed(s, i) if rec["task"] == T5 else None,
        })
    return out


def build_probe(args) -> int:
    from main.lang.data import LangSampler
    import dataclasses

    from main.lang.legacy_pool import install_vocab

    cfg = vanilla_cfg()
    install_vocab(cfg.vocab)
    tasks, embs = trained_subsets(cfg)
    roots, vals, _ = data_args()
    pcfg = dataclasses.replace(cfg, shuffle_text=False, relabel="paired", train_episodes_per_task=0)
    rng = np.random.default_rng(20261030)
    val_s = LangSampler(vals, pcfg, tasks=tasks, embodiments=embs, seed=11)
    tr_s = LangSampler(roots, cfg, tasks=tasks, embodiments=embs, seed=12)
    ent = []
    # validation (other scenes): 96 T5 drag states x (tall, short) + 320 task-balanced samples, half of them paired
    p = _t5_drag_picks(val_s, 96, rng)
    ent += _entries(val_s, p, "val", True, 0, rng)
    p = val_s.plan(320, rng, pair_frac=0.5)
    ent += _entries(val_s, p, "val", False, 1000, rng)
    # training: 32 T5 drag states (warmup, planner ticks) x 2 + 192 task-balanced samples, half paired
    warm = {s.name for s in tr_s.sources if s.name != "ring"}
    p = _t5_drag_picks(tr_s, 32, rng, sources=warm)
    ent += _entries(tr_s, p, "train", True, 2000, rng)
    p = tr_s.plan(192, rng, pair_frac=0.5)
    ent += _entries(tr_s, p, "train", False, 3000, rng)
    U = rng.uniform(-1.0, 1.0, size=(len(ent), N_RAND, 64)).astype(np.float32)   # first A columns used
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "probe_rand_u.npy", U)
    out = Path(args.out)
    out.write_text(json.dumps({"built": time.ctime(), "n": len(ent), "rand_u": str(OUT / "probe_rand_u.npy"),
                               "entries": ent}, indent=0))
    c = collections.Counter((e["set"], e["task"], e["t5drag"], e["action_source"]) for e in ent)
    print(f"[probe] {len(ent)} samples -> {out}")
    for k, v in sorted(c.items()):
        print("   ", k, v)
    return 0


class ProbeSet:
    """The probe samples' encoder inputs (frames t, t-k, 0 only), sentences, labels and fixed random actions."""

    def __init__(self, path: Path, train_sampler, val_sampler):
        doc = json.loads(Path(path).read_text())
        self.ent = doc["entries"]
        self.n = len(self.ent)
        U = np.load(doc["rand_u"])
        arrs = {k: [] for k in PROBE_FIELDS}
        for set_name, smp in (("val", val_sampler), ("train", train_sampler)):
            idx = [k for k, e in enumerate(self.ent) if e["set"] == set_name]
            if not idx:
                continue
            off = _offsets(smp)
            jmap = {str(s.root): j for j, s in enumerate(smp.sources)}
            picks = []
            for k in idx:
                e = self.ent[k]
                j = jmap[e["root"]]
                s = smp.sources[j]
                i = int(e["local"])
                if int(s.rows[i]) != int(e["slot"]):
                    raise RuntimeError(f"probe entry {k}: {e['root']} local {i} is row {int(s.rows[i])}, not {e['slot']}")
                gi = off[j] + i
                assert int(smp.sid[gi]) == j and int(smp.loc[gi]) == i
                picks.append((gi, int(e["col"]), -1))
            for c0 in range(0, len(picks), 64):
                b = smp.sample(len(picks[c0:c0 + 64]), rng=np.random.default_rng(0), picks=picks[c0:c0 + 64])
                for k in PROBE_FIELDS:
                    arrs[k].append((idx[c0:c0 + 64], np.asarray(b[k])))
            smp.health(reset=True)
        self.arr = {}
        for k, parts in arrs.items():
            first = parts[0][1]
            a = np.empty((self.n,) + first.shape[1:], first.dtype)
            for ii, v in parts:
                a[ii] = v
            self.arr[k] = a
        A = self.arr["dtheta"].shape[-1]
        self.U = U[:, :, :A].copy()
        self.sent = [e["sentence"] for e in self.ent]
        g = lambda f: np.array([f(e) for e in self.ent])  # noqa: E731
        self.set = g(lambda e: e["set"])
        self.task = g(lambda e: e["task"])
        self.t5drag = g(lambda e: bool(e["t5drag"]))
        self.asrc = g(lambda e: int(e["action_source"]))
        self.pair = g(lambda e: int(e["pair"]))
        self.fs = g(lambda e: bool(e["fails_soon"]))
        self.h = g(lambda e: float(e["h"]))
        self.size = g(lambda e: e["size"] or "")
        print(f"[probe] loaded {self.n} samples ({int((self.set == 'val').sum())} val, "
              f"{int((self.set == 'train').sum())} train, {int(self.t5drag.sum())} T5 drag)", flush=True)

    def groups(self) -> dict[str, np.ndarray]:
        v, t = self.set == "val", self.set == "train"
        pl = self.asrc == 0
        g = {"all": np.ones(self.n, bool), "val": v, "train": t, "val_planner": v & pl, "train_planner": t & pl,
             "train_actor": t & (self.asrc == 1), "val_t5drag": v & self.t5drag, "train_t5drag": t & self.t5drag,
             "t5drag_tall": self.t5drag & (self.size == "tall"), "t5drag_short": self.t5drag & (self.size == "short"),
             "t5drag_tall_harm": self.t5drag & (self.size == "tall") & self.fs}
        for task in sorted(set(self.task.tolist())):
            g[f"val_{task}"] = v & (self.task == task) & ~self.t5drag
        return g


def _rep_enc(enc, m: int):
    """LangEncoded with each sample repeated m times, sample-major blocks [m x n] (action k of sample i at k*n + i)."""
    n = enc.body.shape[0]
    idx = torch.arange(n, device=enc.body.device).repeat(m)
    return enc.index(idx)


@torch.no_grad()
def probe_q(tr: LangTrainer, P: ProbeSet, chunk: int = 64, per_pass: int = 4) -> dict:
    """Per-sample arrays: Q at the named actions and the 16 random ones, dQ/da at the data action, actor stats."""
    was = tr.policy_enc.training
    tr.policy_enc.eval(), tr.critics.eval(), tr.actor.eval()
    names = ["data", "zero", "half", "det"] + [f"r{k}" for k in range(N_RAND)]
    Q = np.zeros((P.n, len(names)), np.float32)
    Q1 = np.zeros(P.n, np.float32)
    Q2 = np.zeros(P.n, np.float32)
    gnorm, slope, gnorm_raw = (np.zeros(P.n, np.float32) for _ in range(3))
    mu_abs, u_abs, sig, cap, ls = (np.zeros(P.n, np.float32) for _ in range(5))
    a_det_all = np.zeros(P.arr["dtheta"].shape, np.float32)
    # softmin readout (twin 1): per-head values at the 16 random actions, and where the minimum sits
    VR = None
    d = tr.dev
    for c0 in range(0, P.n, chunk):
        sl = slice(c0, min(P.n, c0 + chunk))
        sub = {k: P.arr[k][sl] for k in PROBE_FIELDS}
        sents = P.sent[sl]
        n = len(sents)
        words, wmask = tr.text.lookup(sents, splits=["probe"] * n)
        enc, Jlin, Jang, ji = tr.encode(sub, words, wmask)
        dmax = _t(sub["dtheta_max"], dev=d)
        a_data = _t(sub["dtheta"], dev=d)
        a_det = tr.actor(enc.body, ji, dmax, deterministic=True)[0]
        acts = [a_data, torch.zeros_like(a_data), 0.5 * a_data, a_det]
        U = _t(P.U[sl], dev=d)
        acts += [U[:, k] * dmax for k in range(N_RAND)]
        for p0 in range(0, len(acts), per_pass):
            grp = acts[p0:p0 + per_pass]
            m = len(grp)
            e_m = _rep_enc(enc, m)
            rep = lambda x: x.repeat(m, *([1] * (x.dim() - 1)))  # noqa: E731
            q1, q2, v1, _ = tr.critics(e_m, torch.cat(grp, 0), rep(Jlin), rep(Jang), rep(ji))
            qm = torch.minimum(q1, q2).view(m, n).T.float().cpu().numpy()
            Q[sl, p0:p0 + m] = qm
            if v1 is not None:
                if VR is None:
                    VR = np.zeros((P.n, len(names), v1.shape[-1]), np.float32)
                VR[sl, p0:p0 + m] = v1.view(m, n, -1).permute(1, 0, 2).float().cpu().numpy()
            if p0 == 0:
                Q1[sl] = q1[:n].float().cpu().numpy()
                Q2[sl] = q2[:n].float().cpu().numpy()
        with torch.enable_grad():
            a = a_data.clone().requires_grad_(True)
            qa = tr.critics.qmin(enc, a, Jlin, Jang, ji)
            (g,) = torch.autograd.grad(qa.sum(), a)
        gnorm[sl] = ((g * dmax) ** 2).sum(-1).sqrt().cpu().numpy()
        gnorm_raw[sl] = g.norm(dim=-1).cpu().numpy()
        slope[sl] = (g * a_data).sum(-1).cpu().numpy()
        M = ji.shape[1]
        mu, log_std = tr.actor.head(enc.body[:, :M]).unbind(-1)
        v = (ji >= 0).float()
        nv = v.sum(-1).clamp_min(1)
        mu_abs[sl] = ((mu.abs() * v).sum(-1) / nv).cpu().numpy()
        u_abs[sl] = ((torch.tanh(mu).abs() * v).sum(-1) / nv).cpu().numpy()
        lsc = log_std.clamp(-5, 0)
        sig[sl] = ((lsc.exp() * v).sum(-1) / nv).cpu().numpy()
        ls[sl] = ((log_std * v).sum(-1) / nv).cpu().numpy()
        cap[sl] = (((log_std >= 0).float() * v).sum(-1) / nv).cpu().numpy()
        a_det_all[sl] = a_det.float().cpu().numpy()
    tr.policy_enc.train(was), tr.critics.train(was), tr.actor.train(was)
    extra = {}
    if VR is not None:
        K = int(getattr(tr.critics.c1, "K", 0))
        extra = {"V1": VR, "K": np.array(K)}
    return {**extra, "names": names, "Q": Q, "Q1": Q1, "Q2": Q2, "gnorm": gnorm, "gnorm_raw": gnorm_raw, "slope": slope,
            "mu_abs": mu_abs, "u_abs": u_abs, "sigma": sig, "log_std_raw": ls, "cap_frac": cap, "a_det": a_det_all}


class tr_cfg_T:   # the softmin temperature in h units (the vanilla config's softmin_T_h)
    T = 0.05


def probe_metrics(P: ProbeSet, r: dict) -> dict:
    from main.lang.metrics import pair_stats

    Q = r["Q"]
    qd, q0, qh, qdet = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    R = Q[:, 4:]
    dmax = P.arr["dtheta_max"]
    jm = dmax > 0
    out = {}
    for name, m in P.groups().items():
        if not m.any():
            continue
        pre = f"{name}/"
        qa = np.abs(qd[m]).mean()
        o = {"n": int(m.sum()), "q_mean": float(qd[m].mean()), "h_mean": float(P.h[m].mean()),
             "gap_mean": float((P.h[m] - qd[m]).mean()),
             "sens_rel": float(np.abs(R[m, 0] - R[m, 1]).mean() / max(qa, 1e-9)),
             "sens_abs": float(np.abs(R[m, 0] - R[m, 1]).mean()),
             "spread16": float((R[m].max(1) - R[m].min(1)).mean()),
             "spread16_rel": float((R[m].max(1) - R[m].min(1)).mean() / max(qa, 1e-9)),
             "std16": float(R[m].std(1).mean()),
             "gnorm": float(r["gnorm"][m].mean()), "gnorm_rel": float(r["gnorm"][m].mean() / max(qa, 1e-9)),
             "slope_c": float(r["slope"][m].mean()), "slope_c_abs": float(np.abs(r["slope"][m]).mean()),
             "q_det_minus_data": float((qdet[m] - qd[m]).mean()),
             "q_det_minus_zero": float((qdet[m] - q0[m]).mean()),
             "actor_mu_abs": float(r["mu_abs"][m].mean()), "actor_u_abs": float(r["u_abs"][m].mean()),
             "actor_sigma": float(r["sigma"][m].mean()), "actor_log_std_raw": float(r["log_std_raw"][m].mean()),
             "actor_cap_frac": float(r["cap_frac"][m].mean()),
             "actor_a_frac": float((np.abs(r["a_det"][m]) / np.where(jm[m], dmax[m], 1.0) * jm[m]).sum()
                                   / max(jm[m].sum(), 1)),
             "actor_action_std": float((r["a_det"][m].std(0) * jm[m][0]).sum() / max(jm[m][0].sum(), 1)),
             "data_action_std": float((P.arr["dtheta"][m].std(0) * jm[m][0]).sum() / max(jm[m][0].sum(), 1))}
        pl = m & (P.asrc == 0)
        if pl.any():   # a_nom = the stored planner command
            o.update({"n_planner": int(pl.sum()),
                      "q0_minus_qnom": float((q0[pl] - qd[pl]).mean()),
                      "q0_minus_qnom_abs": float(np.abs(q0[pl] - qd[pl]).mean()),
                      "q0_gt_qnom_frac": float((q0[pl] > qd[pl]).mean()),
                      "qhalf_minus_qnom": float((qh[pl] - qd[pl]).mean()),
                      "qhalf_minus_qnom_abs": float(np.abs(qh[pl] - qd[pl]).mean()),
                      "qhalf_gt_qnom_frac": float((qh[pl] > qd[pl]).mean()),
                      "qhalf_gt_q0_frac": float((qh[pl] > q0[pl]).mean()),
                      # dQ/dc of Q(s, c a_nom) at c = 1 and ||dQ/da * dtheta_max|| on planner ticks
                      "slope_c_planner": float(r["slope"][pl].mean()),
                      "slope_c_planner_abs": float(np.abs(r["slope"][pl]).mean()),
                      "gnorm_planner": float(r["gnorm"][pl].mean())})
        if "V1" in r:   # where twin 1's softmin minimum sits, and the per-head action spread (16 random actions)
            V = r["V1"][m]                                   # [n, 20 actions, H]; padded body heads hold 1e4
            K = int(r["K"])
            Hb = V.shape[-1] - K
            bm = np.asarray(P.arr["body_mask"])[m].astype(bool)
            if bm.shape[-1] != Hb:   # fall back to the padding fill value (masked_fill 1e4)
                bm = V[:, 0, :Hb] != 1e4
            valid = np.concatenate([bm, np.ones((len(V), K), bool)], -1)
            am = np.where(valid, V[:, 0, :], np.inf).argmin(-1)
            T = float(getattr(tr_cfg_T, "T", 0.05))
            w = np.exp(-(V[:, 0, :] - V[:, 0, :].min(-1, keepdims=True)) / T) * valid
            w = w / w.sum(-1, keepdims=True)
            sp = V[:, 4:, :].max(1) - V[:, 4:, :].min(1)     # [n, H]
            vb = valid[:, :Hb]
            o.update({"min_on_query_frac": float((am >= Hb).mean()), "w_query": float(w[:, Hb:].sum(-1).mean()),
                      "head_spread16_body": float((sp[:, :Hb] * vb).sum() / max(vb.sum(), 1)),
                      "head_spread16_body_max": float(np.where(vb, sp[:, :Hb], 0).max(-1).mean()),
                      "head_spread16_query": float(sp[:, Hb:].mean()) if K else float("nan"),
                      "head_spread16_argmin": float(sp[np.arange(len(am)), am].mean()),
                      # how far above the readout's minimum the lowest valid body head sits (h units), at a_data
                      "body_min_minus_qmin": float(np.nanmean(np.where(bm, V[:, 0, :Hb], np.inf).min(-1)
                                                              - V[:, 0, :].min(-1, where=valid, initial=np.inf)))
                      if bm.any() else float("nan"),
                      "body_val_median": float(np.median(V[:, 0, :Hb][bm])) if bm.any() else float("nan")})
        ps = pair_stats(qd[m], qd[m], P.fs[m], P.pair[m])
        o["pair_rank_acc"] = ps["lang/pair_rank_acc"]
        o["n_pairs"] = ps["lang/n_pairs"]
        out.update({pre + k: v for k, v in o.items()})
    return out


def write_probe(tr, P, path: Path, step: int, extra: dict | None = None, chunk: int = 64, per_pass: int = 4) -> dict:
    t0 = time.time()
    r = probe_q(tr, P, chunk=chunk, per_pass=per_pass)
    m = probe_metrics(P, r)
    m.update({"step": int(step), "probe_s": time.time() - t0, **(extra or {})})
    with open(path, "a") as f:
        f.write(json.dumps({k: (None if isinstance(v, float) and v != v else v) for k, v in m.items()}) + "\n")
    qd = Path(path).parent / "probe_q"
    qd.mkdir(exist_ok=True)
    np.savez_compressed(qd / f"step{int(step):06d}.npz", **{k: v for k, v in r.items() if k != "names"},
                        names=np.array(r["names"]))
    print(f"[probe] step {step} ({m['probe_s']:.0f} s): all sens_rel {m['all/sens_rel']:.2e} spread16 "
          f"{m['all/spread16']:.4f} | val_planner q0-qnom {m.get('val_planner/q0_minus_qnom', float('nan')):+.4f} "
          f"qhalf-qnom {m.get('val_planner/qhalf_minus_qnom', float('nan')):+.4f} | T5 drag gnorm "
          f"{m.get('val_t5drag/gnorm', float('nan')):.4f} | q {m['all/q_mean']:+.3f} h {m['all/h_mean']:+.3f} | "
          f"rank val {m.get('val/pair_rank_acc', float('nan')):.2f} | sigma {m['all/actor_sigma']:.3f} "
          f"|u| {m['all/actor_u_abs']:.4f}", flush=True)
    return m


def _logged_steps(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for ln in path.read_text().splitlines():
        try:
            out.add(int(json.loads(ln)["step"]))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_trainer(run: Path, target: str, seed: int) -> RetrainTrainer:
    cfg = vanilla_cfg()
    if target == "stoch_mlp":
        import dataclasses

        cfg = dataclasses.replace(cfg, critic_readout="mlp")
    tasks, embs = trained_subsets(cfg)
    roots, vals, caches = data_args()
    if not roots or len(roots) != 16:
        raise SystemExit(f"expected 15 warmup shards + the ring, got {len(roots)}: {roots}")
    tr = RetrainTrainer(cfg, run, roots, vals, caches, tasks=tasks, embodiments=embs, use_wandb=False, seed=seed,
                        target=target)
    tr.identity = code_identity(tasks or cfg.task_choices, roots)
    return tr


def train(args) -> int:
    run = Path(args.run).resolve()
    if not args.resume and (run / "state.pt").exists():
        raise SystemExit(f"{run} has a state.pt: pass --resume or use a new run directory")
    t_start = time.time()
    deadline = t_start + args.time_budget if args.time_budget else None
    tr = make_trainer(run, args.target, args.seed)
    if args.resume and (run / "state.pt").exists():
        tr.load_state()
        print(f"[retrain] resumed at step {tr.step}", flush=True)
    P = ProbeSet(Path(args.probe), tr.sampler, tr.val_sampler)
    if args.target == "stoch_norm":
        tr.action_norm = load_or_make_norm(P, run)
        ActionNorm.install(tr.action_norm)
        print(f"[retrain] stoch_norm: own dtheta / dtheta_max, dp / {tr.action_norm['s_p']:.4f} m, "
              f"dw / {tr.action_norm['s_w']:.4f} rad (fixed: {run / 'action_norm.json'})", flush=True)
    diag = run / "diag.jsonl"
    done = _logged_steps(diag)
    tr.install_signal()
    print(f"[retrain] target {args.target}: setup {time.time() - t_start:.0f} s, step {tr.step} -> {args.grad_steps}",
          flush=True)
    if tr.step == 0 and 0 not in done:
        write_probe(tr, P, diag, 0, {"target": args.target})
    tr.policy_enc.train(), tr.critics.train(), tr.actor.train()
    tr.reset_perf()
    t_loop, s_loop = time.time(), tr.step
    acc = collections.defaultdict(list)   # batch metrics of the logged steps since the last probe
    t_probe_total = 0.0
    while tr.step < args.grad_steps:
        do_log = (tr.step + 1) % LOG_EVERY == 0
        out = tr.grad_step(log=do_log)
        if do_log:
            out["perf/wall_s_per_step"] = (time.time() - t_loop - t_probe_total) / max(tr.step - s_loop, 1)
            tr.log(out)
            tr.console(out)
            for k in BATCH_KEYS:
                v = out.get(k)
                if isinstance(v, (int, float)) and v == v:
                    acc[k].append(float(v))
        if tr.step % args.probe_every == 0 and tr.step not in done:
            t0p = time.time()
            ex = {"target": args.target, **{f"batch/{k}": float(np.mean(v)) for k, v in acc.items() if v},
                  "train_s_per_step": (time.time() - t_loop - t_probe_total) / max(tr.step - s_loop, 1)}
            write_probe(tr, P, diag, tr.step, ex)
            acc.clear()
            done.add(tr.step)
            t_probe_total += time.time() - t0p
            tr.reset_perf()
        if tr.step % args.state_every == 0:
            tr.save_state()
        if args.ckpt_every and tr.step % args.ckpt_every == 0:
            tr.save_ckpt(f"checkpoint_step{tr.step}.pt")
        over = deadline is not None and time.time() > deadline
        if tr._stop or over:
            tr.save_state()
            tr.save_ckpt()
            print(f"[retrain] stopping at step {tr.step} ({'signal' if tr._stop else 'time budget'}), "
                  f"{(time.time() - t_loop) / max(tr.step - s_loop, 1):.3f} s/step this job; resume with --resume",
                  flush=True)
            if tr.prefetch is not None:
                tr.prefetch.stop()
            return EXIT_RESUME
    tr.save_state()
    tr.save_ckpt()
    tr.save_ckpt("checkpoint_final.pt")
    if tr.prefetch is not None:
        tr.prefetch.stop()
    print(f"[retrain] done at step {tr.step}", flush=True)
    return 0


def score_ckpt(args) -> int:
    """Score existing checkpoints (e.g. the vanilla run's) on the same probe set -> <run>/diag_<name>.jsonl."""
    run = Path(args.run).resolve()
    ro = {(torch.load(ck, map_location="cpu", weights_only=False).get("lang_config") or {}).get("critic_readout",
                                                                                                "softmin")
          for ck in args.ckpt}
    if len(ro) != 1:
        raise SystemExit(f"score-ckpt: one readout per call, got {ro}")
    tr = make_trainer(run, "stoch_mlp" if ro == {"mlp"} else "stoch", 0)
    P = ProbeSet(Path(args.probe), tr.sampler, tr.val_sampler)
    for ck in args.ckpt:
        tr.load_weights(ck)
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        step = int(meta.get("step", -1))
        sc = meta.get("diag_action_norm")
        if sc is not None or ActionNorm._orig is not None:
            ActionNorm.install(sc)   # a stoch_norm checkpoint is scored with its own action scaling
        name = Path(ck).parent.name + "__" + Path(ck).stem
        write_probe(tr, P, run / f"diag_{name}.jsonl", step, {"ckpt": str(ck), "diag_target": meta.get("diag_target"),
                                                             "action_norm": sc is not None},
                    chunk=args.chunk, per_pass=args.per_pass)
        del meta
    return 0


def export_states(args) -> int:
    """The probe's T5 drag states as a score_states.py states file (a smoke test of the scorer and a fallback set
    until check 2 writes its own): candidate first actions planner (the stored a_nom), 0.5 / 0.25 a_nom, zero."""
    cfg = vanilla_cfg()
    from main.lang.data import LangSampler
    from main.lang.legacy_pool import install_vocab

    install_vocab(cfg.vocab)
    tasks, embs = trained_subsets(cfg)
    roots, vals, _ = data_args()
    import dataclasses

    pcfg = dataclasses.replace(cfg, shuffle_text=False, relabel="random", train_episodes_per_task=0)
    tr_s = LangSampler(roots, cfg, tasks=tasks, embodiments=embs, seed=12)
    val_s = LangSampler(vals, pcfg, tasks=tasks, embodiments=embs, seed=11, probe=True)
    P = ProbeSet(Path(args.probe), tr_s, val_s)
    m = np.flatnonzero(P.t5drag)
    enc = {k: P.arr[k][m] for k in PROBE_FIELDS if k != "dtheta"}
    a = P.arr["dtheta"][m].astype(np.float32)
    acts = {"planner": a, "half": 0.5 * a, "quarter": 0.25 * a, "zero": np.zeros_like(a)}
    meta = [{k: P.ent[i][k] for k in ("set", "source", "slot", "col", "pair", "task", "rule_id", "size", "episode_id",
                                       "t_in_ep", "fails_soon", "h", "tray_speed")} for i in m.tolist()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": enc, "actions": acts, "sentences": [P.sent[i] for i in m.tolist()], "meta": meta,
                "h": P.h[m], "fails_soon": P.fs[m], "is_tall": P.size[m] == "tall", "probe_index": m},
               out)
    print(f"[export] {len(m)} T5 drag states -> {out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("build-probe")
    p.add_argument("--out", default=str(PROBE))
    p = sub.add_parser("train")
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--grad-steps", type=int, default=6000)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", default=str(PROBE))
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--state-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p = sub.add_parser("export-states")
    p.add_argument("--out", default=str(OUT / "states_probe_t5drag.pt"))
    p.add_argument("--probe", default=str(PROBE))
    p = sub.add_parser("score-ckpt")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--probe", default=str(PROBE))
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--per-pass", type=int, default=4)
    args = ap.parse_args(argv)
    if args.mode == "build-probe":
        return build_probe(args)
    if args.mode == "train":
        return train(args)
    if args.mode == "export-states":
        return export_states(args)
    return score_ckpt(args)


if __name__ == "__main__":
    raise SystemExit(main())
