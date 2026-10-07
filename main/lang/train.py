"""LangTrainer + CLI (VANILLA_PLAN Section 6): HJ-SAC on relabeled batches.

    python -m main.lang.train offline --data data/lang/warmup --val data/lang/val --run runs/lang/<name> \\
        --grad-steps 50000 --probe-every 5000
    python -m main.lang.train rounds --data data/lang/warmup --val data/lang/val --run runs/lang/<name> \\
        --collect-rounds 100 --episodes-per-round 4 --grad-steps-per-round 2048 --probe-every-rounds 10
    python -m main.lang.train sanity --data data/lang/warmup --run runs/lang/sanity_<name>   # Section 6.4

Update (one grad step, Section 6.1): the critic loss trains the whole
LangPolicyEncoder (injection, scene set, text adapter, scene-reads-words,
trunk; r_static when lang_rtoken is off) and both critic heads; the actor loss
trains TokenActor only (encoder and critics frozen for the step, every
LangEncoded field detached). Target y = (1 - g) h + g min(h, mean of the
target twins at t+1), the same sentence at t and t+1 and for a'. Polyak
copies of the whole encoder and both critics (tau 0.005).

Resume: ``--resume`` restores weights, targets, optimizers, step, the sampler
and torch RNG states (the next batch is exactly the one an uninterrupted run
would draw). ``--time-budget`` stops cleanly (state saved, exit 75) before a
job's walltime; SIGTERM does the same.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import ensure_item_path

ensure_item_path()

from main.network.heads import TokenActor  # noqa: E402

from .config import CANON_TASKS, LangConfig  # noqa: E402
from .network import LangPolicyEncoder, LangTwinCritic, n_params, param_groups, set_probe  # noqa: E402

EXIT_RESUME = 75
LOG_EVERY = 20
STATE_EVERY = 500
CKPT_PARTS = ("injection", "scene_set", "text_adapter", "scene_reads_words", "r_static", "blocks")


def _safe_wandb_log(run, logs: dict, step: int) -> None:
    """main.train.train._safe_wandb_log (re-declared: importing train.py pulls in its CLI globals)."""
    if run is None:
        return
    try:
        run.log(logs, step=step)
    except Exception as exc:
        print(f"[lang-train] wandb log failed at step {step} ({exc}); continuing")


def _t(x, dtype=torch.float32, dev="cuda"):
    """numpy -> device tensor. Uploads in the array's own dtype and casts on the
    device: the fp16 scene tokens (~470 MB per batch as fp32) are never widened
    on the CPU (same values: fp16 -> fp32 is exact)."""
    if torch.is_tensor(x):
        return x.to(dev).to(dtype)
    a = np.asarray(x)
    if not a.flags.writeable:
        a = a.copy()
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev, non_blocking=False).to(dtype)


def roots_of(spec) -> list[Path]:
    """--data accepts a data root, a directory of shard_g* roots, or globs."""
    out = []
    for s in ([spec] if isinstance(spec, (str, Path)) else spec):
        for p in sorted(glob.glob(str(s))):
            p = Path(p)
            if (p / "item_buffer").is_dir() and (p / "buffer").is_dir():
                out.append(p)
            else:
                out += sorted(q for q in p.glob("shard_g*") if (q / "item_buffer").is_dir())
    return out


def caches_of(spec) -> list[Path]:
    out = []
    for s in ([spec] if isinstance(spec, (str, Path)) else spec):
        p = Path(s)
        c = p / "text_cache.pt" if p.is_dir() else p
        if c.exists():
            out.append(c)
    return out


class LangTrainer:
    def __init__(self, cfg: LangConfig, run_dir: Path, data_roots, val_roots=(), text_caches=(),
                 tasks=None, embodiments=None, use_wandb: bool = True, seed: int = 0, probe_n: int = 512,
                 name: str | None = None, prefetch: bool = True, device: str = "cuda", text_encoder=None):
        self.cfg, self.run = cfg, Path(run_dir).resolve()
        self.dev = device
        self.run.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run / "lang_config.json")
        torch.manual_seed(seed)
        np.random.seed(seed)
        from .data import LangSampler, PrefetchSampler
        from .text import TextStore

        self.sampler = LangSampler(data_roots, cfg, tasks=tasks, embodiments=embodiments, seed=seed, n_threads=16)
        n_tok = self.sampler.sources[0].step["scene_tokens"].shape[1] if self.sampler.sources else 1200
        self.policy_enc = LangPolicyEncoder(cfg, n_tok=n_tok).to(device)
        self.actor = TokenActor().to(device)
        self.critics = LangTwinCritic(cfg).to(device)
        self.policy_enc_targ = copy.deepcopy(self.policy_enc)
        self.critics_targ = copy.deepcopy(self.critics)
        self.target_pairs = list(zip(list(self.policy_enc_targ.parameters()) + list(self.critics_targ.parameters()),
                                     list(self.policy_enc.parameters()) + list(self.critics.parameters())))
        for p in list(self.policy_enc_targ.parameters()) + list(self.critics_targ.parameters()):
            p.requires_grad_(False)
        self.opt_e = torch.optim.Adam(self.policy_enc.parameters(), lr=cfg.lr)
        self.opt_c = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.text = TextStore(text_caches, cfg.text_backend, device, extra_path=self.run / "text_cache_extra.pt",
                              backend_paths=cfg.backend_paths, encoder=text_encoder)
        self.prefetch = PrefetchSampler(self.sampler, cfg.batch_size, seed=seed + 1) if prefetch else None
        self.val_sampler = LangSampler(val_roots, cfg, tasks=tasks, embodiments=embodiments, seed=seed + 2,
                                       probe=True) if val_roots else None
        self.prober = None
        self.probe_n = int(probe_n)
        self.step = 0
        self.round = 0
        self._stop = False
        self._last_rng_after = None
        self.t_last = time.perf_counter()
        self.wait_last = 0.0
        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(project="cehj-lang", name=name or self.run.name, dir=str(self.run),
                                        config={**cfg.to_dict(), "config_hash": cfg.hash()}, resume="allow",
                                        id=(name or self.run.name).replace("/", "_"))
            except Exception as exc:
                print(f"[lang-train] wandb disabled ({exc})")
        print(f"[lang-train] run {self.run} config {cfg.hash()} bank {cfg.bank_hash} | rows {len(self.sampler)} "
              f"from {len(self.sampler.sources)} sources | text cache {len(self.text)} | params "
              f"enc {n_params(self.policy_enc.parameters())} critics {n_params(self.critics.parameters())} "
              f"actor {n_params(self.actor.parameters())}")

    # ------------------------------------------------------------ schedules --
    def gamma(self) -> float:
        return float(self.cfg.gamma)          # fixed (asserted == gamma_final)

    def alpha(self) -> float:
        c = self.cfg
        f = min(1.0, self.step / max(c.alpha_anneal_steps, 1))
        return c.alpha_start + f * (c.alpha_final - c.alpha_start)

    # --------------------------------------------------------------- encode --
    def encode(self, b, words, wmask, suffix: str = "", target: bool = False):
        g = lambda k: b[k + suffix]  # noqa: E731
        d = self.dev
        mod = self.policy_enc_targ if target else self.policy_enc
        enc = mod(_t(g("state_tokens"), dev=d), _t(g("body_tokens"), dev=d), _t(g("link_pos"), dev=d),
                  _t(g("body_mask"), torch.bool, d), _t(g("joint_index"), torch.int64, d),
                  _t(g("scene_tokens"), dev=d), _t(g("scene_pos"), dev=d), words=words, word_mask=wmask)
        return enc, _t(g("Jlin"), dev=d), _t(g("Jang"), dev=d), _t(g("joint_index"), torch.int64, d)

    # --------------------------------------------------------------- update --
    def grad_step(self, b: dict | None = None, log: bool = False) -> dict:
        cfg = self.cfg
        if b is None:
            b = self.prefetch.next() if self.prefetch is not None else self.sampler.sample(cfg.batch_size)
            self._last_rng_after = b.get("_rng_after")
        words, wmask = self.text.lookup(b["sentence"])
        enc, Jlin, Jang, ji = self.encode(b, words, wmask)
        d = self.dev
        h = _t(b["h"], dev=d)
        dtheta, dmax = _t(b["dtheta"], dev=d), _t(b["dtheta_max"], dev=d)
        gamma, alpha = self.gamma(), self.alpha()
        with torch.no_grad():
            enc_n, Jlin_n, Jang_n, ji_n = self.encode(b, words, wmask, "_next", target=True)
            a_n, _, _, _ = self.actor(enc_n.body, ji_n, _t(b["dtheta_max_next"], dev=d))
            q1_t, q2_t, _, _ = self.critics_targ(enc_n, a_n, Jlin_n, Jang_n, ji_n)
            y = (1 - gamma) * h + gamma * torch.minimum(h, 0.5 * (q1_t + q2_t))
        q1, q2, _, _ = self.critics(enc, dtheta, Jlin, Jang, ji)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_e.zero_grad(set_to_none=True)
        self.opt_c.zero_grad(set_to_none=True)
        loss_q.backward()
        gn = {}
        if log:
            from .metrics import grad_norm

            gn["grad/text_adapter"] = grad_norm(self.policy_enc.text_adapter.parameters())
        gn["grad/encoder"] = float(torch.nn.utils.clip_grad_norm_(self.policy_enc.parameters(), 10.0))
        gn["grad/critic"] = float(torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0))
        self.opt_e.step()
        self.opt_c.step()
        # actor: encoder and critics frozen for this step, every field detached
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
        self.step += 1
        out = {"loss/q": float(loss_q), "loss/pi": float(loss_pi), "alpha": alpha, "gamma": gamma, **gn}
        if log:
            out.update(self._diagnostics(b, enc_d, q1, q2, y, h, logp, dtheta, dmax, Jlin, Jang, ji))
        return out

    @torch.no_grad()
    def _diagnostics(self, b, enc_d, q1, q2, y, h, logp, dtheta, dmax, Jlin, Jang, ji) -> dict:
        from .metrics import batch_metrics

        out = batch_metrics(b, q1, q2, y, h)
        out["entropy"] = float(-logp.mean())
        a_det, _, _, _ = self.actor(enc_d.body, ji, dmax, deterministic=True)
        jm = (dmax > 0).float()
        out["actor/action_std"] = float((a_det.std(0) * jm).sum() / jm.sum().clamp_min(1))
        out["actor/sat_frac"] = float(((a_det.abs() / dmax.clamp_min(1e-9)) > 0.95).float().mul(jm).sum()
                                      / jm.sum().clamp_min(1))
        out["data/action_std"] = float((dtheta.std(0) * jm).sum() / jm.sum().clamp_min(1))
        r1 = (torch.rand_like(dtheta) * 2 - 1) * dmax
        r2 = (torch.rand_like(dtheta) * 2 - 1) * dmax
        sens = (self.critics.qmin(enc_d, r1, Jlin, Jang, ji) - self.critics.qmin(enc_d, r2, Jlin, Jang, ji)).abs().mean()
        out["critic/action_sens_rel"] = float(sens / torch.minimum(q1, q2).abs().mean().clamp_min(1e-9))
        out.update(self.sampler.health())
        out["data/text_miss_frac"] = self.text.miss_frac().get("all", 0.0)
        now = time.perf_counter()
        dt = now - self.t_last
        wait = self.prefetch.wait_s if self.prefetch is not None else 0.0
        out["perf/steps_per_sec"] = LOG_EVERY / max(dt, 1e-9)
        out["perf/data_wait_frac"] = (wait - self.wait_last) / max(dt, 1e-9)
        out["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if self.dev.startswith("cuda") else 0.0
        self.t_last, self.wait_last = now, wait
        return out

    def pause_prefetch(self) -> None:
        """Stop the prefetch thread and rewind its RNG to the last consumed batch."""
        if self.prefetch is not None:
            self.prefetch.stop(rewind_to=self._last_rng_after)

    # ---------------------------------------------------------------- probe --
    @torch.no_grad()
    def probe_forward(self, sub: dict, sentences, probe: bool):
        """qmin at the logged action for the given sentences (+ the encoder's probe stores)."""
        from .metrics import geo_attention_weights

        was = self.policy_enc.training
        self.policy_enc.eval(), self.critics.eval()
        words, wmask = self.text.lookup(sentences)
        stores = set_probe(self.policy_enc, True) if probe else []
        enc, Jlin, Jang, ji = self.encode(sub, words, wmask)
        if probe and stores:
            last = stores[-1]
            if "geo_inputs" in last:
                y, x = last["geo_inputs"]
                blk = self.policy_enc.blocks[-1]
                last["geo_A"] = geo_attention_weights(blk.geo, y, x, enc.scene, enc.scene_pos, enc.scene_mask)
        set_probe(self.policy_enc, False)
        q = self.critics.qmin(enc, _t(sub["dtheta"], dev=self.dev), Jlin, Jang, ji)
        self.policy_enc.train(was), self.critics.train(was)
        return q, enc, stores

    def run_probe(self) -> dict:
        from .metrics import Prober

        if self.val_sampler is None or not len(self.val_sampler):
            return {}
        if self.prober is None:
            self.prober = Prober(self.val_sampler, n=self.probe_n)
        t0 = time.time()
        out = self.prober.run(self.probe_forward, self.run / "probes", self.step)
        out["perf/probe_s"] = time.time() - t0
        return out

    # ------------------------------------------------------------ checkpoint --
    def ckpt_dict(self) -> dict:
        pe = self.policy_enc
        d = {k: getattr(pe, k).state_dict() if isinstance(getattr(pe, k), torch.nn.Module) else getattr(pe, k).detach().cpu()
             for k in CKPT_PARTS if hasattr(pe, k)}
        d["trunk"] = d.pop("blocks")
        d.update({"actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
                  "lang_config": self.cfg.to_dict(), "bank_hash": self.cfg.bank_hash,
                  "text_backend": self.cfg.text_backend, "step": self.step, "round": self.round,
                  "switches": self.cfg.path_switches()})
        return d

    def save_ckpt(self, name: str = "checkpoint.pt") -> Path:
        p = self.run / name
        tmp = p.with_suffix(".tmp")
        torch.save(self.ckpt_dict(), tmp)
        os.replace(tmp, p)
        self.text.save_extra()
        return p

    def load_weights(self, path, strict_switches: bool = True) -> None:
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        if strict_switches and ck.get("switches") != self.cfg.path_switches():
            raise ValueError(f"{path}: switches {ck.get('switches')} != {self.cfg.path_switches()}")
        if ck.get("text_backend") != self.cfg.text_backend:
            raise ValueError(f"{path}: text backend {ck.get('text_backend')} != {self.cfg.text_backend}")
        load_policy(self.policy_enc, ck)
        self.actor.load_state_dict(ck["actor"])
        self.critics.load_state_dict(ck["critics"])
        self.policy_enc_targ.load_state_dict(self.policy_enc.state_dict())
        self.critics_targ.load_state_dict(self.critics.state_dict())

    def save_state(self, extra: dict | None = None) -> Path:
        st = {"policy_enc": self.policy_enc.state_dict(), "policy_enc_targ": self.policy_enc_targ.state_dict(),
              "actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
              "critics_targ": self.critics_targ.state_dict(), "opt_e": self.opt_e.state_dict(),
              "opt_c": self.opt_c.state_dict(), "opt_a": self.opt_a.state_dict(), "step": self.step,
              "round": self.round, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state() if self.dev.startswith("cuda") else None,
              "sampler_rng": self.sampler.rng.bit_generator.state,
              "prefetch_rng": self._last_rng_after if self._last_rng_after is not None
              else (self.prefetch.rng_state() if self.prefetch is not None else None),
              "lang_config": self.cfg.to_dict(), **(extra or {})}
        p = self.run / "state.pt"
        tmp = p.with_suffix(".tmp")
        torch.save(st, tmp)
        os.replace(tmp, p)
        self.text.save_extra()
        return p

    def load_state(self, path=None) -> dict:
        st = torch.load(path or self.run / "state.pt", map_location=self.dev, weights_only=False)
        for k in ("policy_enc", "policy_enc_targ", "actor", "critics", "critics_targ"):
            getattr(self, k).load_state_dict(st[k])
        for k in ("opt_e", "opt_c", "opt_a"):
            getattr(self, k).load_state_dict(st[k])
        self.step, self.round = int(st["step"]), int(st.get("round", 0))
        torch.set_rng_state(st["torch_rng"].cpu())
        if st.get("cuda_rng") is not None and self.dev.startswith("cuda"):
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        self.sampler.rng.bit_generator.state = st["sampler_rng"]
        if self.prefetch is not None and st.get("prefetch_rng") is not None:
            self.prefetch.stop()
            self.prefetch.set_rng_state(st["prefetch_rng"])
        return st

    # --------------------------------------------------------------- logging --
    def log(self, logs: dict) -> None:
        logs = {k: v for k, v in logs.items() if isinstance(v, (int, float, np.floating, np.integer))}
        logs["step"] = self.step
        with open(self.run / "train_log.jsonl", "a") as f:
            f.write(json.dumps({k: (float(v) if v == v else None) for k, v in logs.items()}) + "\n")
        _safe_wandb_log(self.wandb, logs, self.step)

    def console(self, o: dict) -> None:
        print(f"[lang-train] step {self.step} loss_q {o.get('loss/q', 0):.4f} loss_pi {o.get('loss/pi', 0):+.3f} "
              f"q {o.get('q/mean', 0):+.3f} h {o.get('h/mean', 0):+.3f} gap {o.get('gap/mean', 0):+.3f} "
              f"rank {o.get('lang/pair_rank_acc', float('nan')):.2f} sps {o.get('perf/steps_per_sec', 0):.2f} "
              f"wait {o.get('perf/data_wait_frac', 0):.2f} mem {o.get('perf/gpu_mem_gb', 0):.1f}G", flush=True)

    def install_signal(self):
        def handler(sig, frame):
            print(f"[lang-train] signal {sig}: stopping after this step", flush=True)
            self._stop = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGUSR1, handler)

    # ----------------------------------------------------------- offline run --
    def train_offline(self, grad_steps: int, probe_every: int, time_budget: float | None = None,
                      ckpt_every: int = 5000) -> int:
        from .collect import stage_deadline

        deadline = stage_deadline(time_budget)
        self.install_signal()
        self.policy_enc.train(), self.critics.train(), self.actor.train()
        while self.step < grad_steps:
            do_log = (self.step + 1) % LOG_EVERY == 0
            out = self.grad_step(log=do_log)
            if do_log:
                self.log(out)
                self.console(out)
            if probe_every and self.step % probe_every == 0:
                pr = self.run_probe()
                if pr:
                    self.log(pr)
                    print(f"[lang-train] probe step {self.step}: " + " ".join(f"{k}={v:.3f}" for k, v in pr.items()
                                                                              if isinstance(v, float)), flush=True)
            if ckpt_every and self.step % ckpt_every == 0:
                self.save_ckpt()
                self.save_ckpt(f"checkpoint_step{self.step}.pt")
            if self.step % STATE_EVERY == 0:
                self.save_state()
            over = deadline is not None and time.time() > deadline
            if self._stop or over:
                self.save_state()
                self.save_ckpt()
                print(f"[lang-train] stopping at step {self.step} ({'signal' if self._stop else 'time budget'}); "
                      f"resume with --resume", flush=True)
                return EXIT_RESUME
        self.save_state()
        self.save_ckpt()
        self.save_ckpt("checkpoint_final.pt")
        if self.prefetch is not None:
            self.prefetch.stop()
        return 0


def load_policy(policy_enc: LangPolicyEncoder, ck: dict) -> None:
    for k in CKPT_PARTS:
        key = "trunk" if k == "blocks" else k
        if not hasattr(policy_enc, k):
            if key in ck:
                raise ValueError(f"checkpoint has {key} but the model has no such part (switch mismatch)")
            continue
        part = getattr(policy_enc, k)
        if isinstance(part, torch.nn.Module):
            part.load_state_dict(ck[key])
        else:
            with torch.no_grad():
                part.copy_(ck[key].to(part.device))


# ---------------------------------------------------------------------------
# sanity checks (Section 6.4)
# ---------------------------------------------------------------------------


def target_rank_acc(q: np.ndarray, y: np.ndarray, pair: np.ndarray) -> float:
    ok = []
    for p in np.unique(pair[pair >= 0]):
        k = np.flatnonzero(pair == p)
        if k.size == 2 and abs(y[k[0]] - y[k[1]]) > 0.05:
            ok.append(np.sign(q[k[0]] - q[k[1]]) == np.sign(y[k[0]] - y[k[1]]))
    return float(np.mean(ok)) if ok else float("nan")


def run_sanity(args) -> int:
    cfg0 = config_from_args(args)
    roots = roots_of(args.data)
    out = {"data": [str(r) for r in roots]}
    res_dir = Path(args.run).resolve()
    res_dir.mkdir(parents=True, exist_ok=True)
    caches = ([Path(c) for c in args.text_cache] if args.text_cache
              else caches_of(args.data) + caches_of(args.val or []))
    from .data import LangSampler

    base = LangSampler(roots, cfg0, seed=0)
    rng = np.random.default_rng(0)
    picks = base.plan(cfg0.batch_size, rng, pair_frac=1.0)       # 32 pairs from mixed rows
    batch = base.sample(cfg0.batch_size, rng, picks=picks)
    print(f"[sanity] fixed batch: {int((batch['pair'] >= 0).sum())} paired samples, tasks "
          f"{sorted(set(batch['task']))}", flush=True)

    def shuffled_sentences(tr, r):
        return [tr.sampler._shuffled_sentence(next(i for i, x in enumerate(tr.sampler.sources) if x.name == src),
                                              int(e), r) for src, e in zip(batch["source"], batch["episode_id"])]

    def rank_acc(tr, cfg, sentences):
        """target-rank accuracy of the live critics on the fixed batch with these sentences."""
        b = dict(batch)
        b["sentence"] = sentences
        with torch.no_grad():
            words, wmask = tr.text.lookup(b["sentence"])
            enc, Jlin, Jang, ji = tr.encode(b, words, wmask)
            q1, q2, _, _ = tr.critics(enc, _t(b["dtheta"], dev=tr.dev), Jlin, Jang, ji)
            h = _t(b["h"], dev=tr.dev)
            enc_n, Jn, An, jn = tr.encode(b, words, wmask, "_next", target=True)
            a_n, _, _, _ = tr.actor(enc_n.body, jn, _t(b["dtheta_max_next"], dev=tr.dev))
            t1, t2, _, _ = tr.critics_targ(enc_n, a_n, Jn, An, jn)
            y = (1 - cfg.gamma) * h + cfg.gamma * torch.minimum(h, 0.5 * (t1 + t2))
        qm = torch.minimum(q1, q2).cpu().numpy()
        # label-based: P[Q(harmful) < Q(benign)] over the pairs; unlike the target-rank score it does
        # not depend on the sentence through the target critic, so it is exactly chance when shuffled
        from .metrics import pair_stats

        lab = pair_stats(qm, y.cpu().numpy(), np.asarray(batch["fails_soon"], bool), np.asarray(batch["pair"]))
        return target_rank_acc(qm, y.cpu().numpy(), batch["pair"]), lab["lang/pair_rank_acc"]

    def overfit(tag, cfg, shuffle: bool, steps: int):
        tr = LangTrainer(cfg, res_dir / tag, roots, (), caches, use_wandb=False, prefetch=False, seed=0)
        eval_rng = np.random.default_rng(12345)
        hist = []
        for s in range(steps):
            b = dict(batch)
            if shuffle:
                b["sentence"] = shuffled_sentences(tr, rng)
            o = tr.grad_step(b, log=False)
            if s % 50 == 0 or s == steps - 1:
                # shuffled control: scored on FRESH other-episode sentences (4 draws), never on the
                # draw the model was just updated on (that would measure one step of fitting)
                accs = ([rank_acc(tr, cfg, shuffled_sentences(tr, eval_rng)) for _ in range(4)]
                        if shuffle else [rank_acc(tr, cfg, list(batch["sentence"]))])
                acc = float(np.mean([a for a, _ in accs]))
                lab = float(np.nanmean([b for _, b in accs]))
                hist.append({"step": s, "loss_q": o["loss/q"], "target_rank_acc": acc, "label_rank_acc": lab})
                print(f"[sanity {tag}] step {s} loss_q {o['loss/q']:.5f} target_rank_acc {acc:.3f} "
                      f"label_rank_acc {lab:.3f}", flush=True)
        return tr, hist

    steps = int(args.sanity_steps)
    tr, hist = overfit("overfit", cfg0, False, steps)
    l0 = np.mean([h["loss_q"] for h in hist[:1]])
    l1 = hist[-1]["loss_q"]
    out["overfit"] = {"history": hist, "loss_ratio": float(l0 / max(l1, 1e-12)),
                      "target_rank_acc": hist[-1]["target_rank_acc"],
                      "pass": bool(hist[-1]["target_rank_acc"] >= 0.95 and l0 / max(l1, 1e-12) >= 10)}
    del tr
    torch.cuda.empty_cache()
    tr, hist = overfit("shuffled", dataclasses.replace(cfg0, shuffle_text=True), True, steps)
    tail = hist[-4:]   # the last 4 scorings (150 steps), each over 4 fresh sentence draws
    out["shuffled"] = {"history": hist, "target_rank_acc": hist[-1]["target_rank_acc"],
                       "target_rank_acc_tail_mean": float(np.mean([h["target_rank_acc"] for h in tail])),
                       "label_rank_acc_tail_mean": float(np.nanmean([h["label_rank_acc"] for h in tail])),
                       # "stays <= 0.6": judged on the mean of the last 4 scorings (a single scoring of
                       # ~30 pairs has a standard error of ~0.05)
                       "pass": bool(np.mean([h["target_rank_acc"] for h in tail]) <= 0.6)}
    del tr
    torch.cuda.empty_cache()
    # switches off: the sentence cannot change Q (text_sens exactly 0)
    off = dataclasses.replace(cfg0, **{k: False for k in cfg0.path_switches()})
    tr = LangTrainer(off, res_dir / "switches_off", roots, (), caches, use_wandb=False, prefetch=False, seed=0)
    for _ in range(int(args.switch_steps)):
        tr.grad_step(log=False)
    with torch.no_grad():
        b = dict(batch)
        q_a, _, _ = tr.probe_forward(b, list(b["sentence"]), False)
        shuf = [tr.sampler._shuffled_sentence(next(i for i, x in enumerate(tr.sampler.sources) if x.name == src),
                                              int(e), rng) for src, e in zip(b["source"], b["episode_id"])]
        q_b, _, _ = tr.probe_forward(b, shuf, False)
    sens = float((q_a - q_b).abs().mean() / q_a.abs().mean().clamp_min(1e-9))
    out["switches_off"] = {"text_sens": sens, "pass": sens == 0.0}
    out["pass"] = all(out[k]["pass"] for k in ("overfit", "shuffled", "switches_off"))
    (res_dir / "sanity.json").write_text(json.dumps(out, indent=1))
    print(f"[sanity] overfit {out['overfit']['pass']} (acc {out['overfit']['target_rank_acc']:.3f}, loss x"
          f"{out['overfit']['loss_ratio']:.1f}) | shuffled {out['shuffled']['pass']} "
          f"(acc {out['shuffled']['target_rank_acc']:.3f}, tail mean {out['shuffled']['target_rank_acc_tail_mean']:.3f}, "
          f"label {out['shuffled']['label_rank_acc_tail_mean']:.3f}) | switches off {out['switches_off']['pass']} "
          f"(text_sens {sens:.2e})", flush=True)
    return 0 if out["pass"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common(p):
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", default="all")
    p.add_argument("--embodiments", default="all")
    p.add_argument("--relabel", choices=("paired", "random", "fixed"), default=None)
    p.add_argument("--shuffle-text", action="store_true")
    p.add_argument("--text-backend", choices=("holobrain_bert", "bert_raw", "sentence_generic"), default=None)
    for s in ("scene", "body", "film", "queries", "rtoken"):
        p.add_argument(f"--no-{s}", action="store_true", help=f"lang_{s} off")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--time-budget", type=float, default=None, help="seconds; then save state and exit 75")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="LangConfig overrides key=value (JSON values)")
    p.add_argument("--text-cache", nargs="*", default=None,
                   help="text caches (default: text_cache.pt next to --data / --val directories)")


def config_from_args(args) -> LangConfig:
    run = Path(args.run)
    if getattr(args, "resume", False) and (run / "lang_config.json").exists():
        return LangConfig.load(run / "lang_config.json")
    kw = {}
    if args.relabel:
        kw["relabel"] = args.relabel
    if args.shuffle_text:
        kw["shuffle_text"] = True
    if args.text_backend:
        kw["text_backend"] = args.text_backend
    for s in ("scene", "body", "film", "queries", "rtoken"):
        if getattr(args, f"no_{s}"):
            kw[f"lang_{s}"] = False
    if args.activation_checkpointing:
        kw["activation_checkpointing"] = True
    if args.tasks != "all":
        kw["task_choices"] = tuple(args.tasks.split(","))
    if args.embodiments != "all":
        kw["embodiment_choices"] = tuple(args.embodiments.split(","))
    for item in args.set:
        k, v = item.split("=", 1)
        kw[k] = json.loads(v)
    return dataclasses.replace(LangConfig(), **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("offline")
    add_common(p)
    p.add_argument("--grad-steps", type=int, default=50000)
    p.add_argument("--probe-every", type=int, default=5000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--init-weights", default=None)
    p = sub.add_parser("rounds")
    add_common(p)
    p.add_argument("--collect-rounds", type=int, default=100)
    p.add_argument("--episodes-per-round", type=int, default=4)
    p.add_argument("--grad-steps-per-round", type=int, default=2048)
    p.add_argument("--probe-every-rounds", type=int, default=10)
    p.add_argument("--video-every", type=int, default=1, help="filter-episode videos every N rounds (0 = none)")
    p = sub.add_parser("sanity")
    add_common(p)
    p.add_argument("--sanity-steps", type=int, default=1000)
    p.add_argument("--switch-steps", type=int, default=50)
    args = ap.parse_args(argv)
    if args.mode == "sanity":
        return run_sanity(args)
    cfg = config_from_args(args)
    tasks = None if args.tasks == "all" else tuple(args.tasks.split(","))
    embs = None if args.embodiments == "all" else tuple(args.embodiments.split(","))
    roots = roots_of(args.data)
    vals = roots_of(args.val) if args.val else []
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.data) + caches_of(args.val)
    if args.mode == "rounds":
        from .rounds import run_rounds

        return run_rounds(cfg, args, roots, vals, caches, tasks, embs)
    if not args.resume and (Path(args.run) / "state.pt").exists():
        raise SystemExit(f"{args.run} has a state.pt: pass --resume or use a new run directory")
    tr = LangTrainer(cfg, Path(args.run), roots, vals, caches, tasks=tasks, embodiments=embs,
                     use_wandb=not args.no_wandb, seed=args.seed)
    if args.resume and (tr.run / "state.pt").exists():
        tr.load_state()
        print(f"[lang-train] resumed at step {tr.step}", flush=True)
    elif args.init_weights:
        tr.load_weights(args.init_weights)
    return tr.train_offline(args.grad_steps, args.probe_every, args.time_budget, args.ckpt_every)


if __name__ == "__main__":
    raise SystemExit(main())
