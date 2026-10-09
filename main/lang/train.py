"""LangTrainer + CLI (VANILLA_PLAN Section 6): HJ-SAC on relabeled batches.

    python -m main.lang.train offline --data data/lang/warmup --val data/lang/val --run runs/lang/<name> \\
        --grad-steps 50000 --probe-every 5000
    python -m main.lang.train rounds --data data/lang/warmup --val data/lang/val --run runs/lang/<name> \\
        --collect-rounds 100 --episodes-per-round 4 --grad-steps-per-round 2048 --probe-every-rounds 10
    python -m main.lang.train sanity --data data/lang/warmup --run runs/lang/sanity_<name>   # Section 6.4

Update (one grad step, Section 6.1): the critic loss trains the whole
LangPolicyEncoder (injection, scene set, text adapter, scene-reads-words,
trunk; r_static when lang_rtoken is off) and both critic heads; the actor loss
trains the actor only (encoder and critics frozen for the step, every
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
import contextlib
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


from .config import CANON_EMBODIMENTS, CANON_TASKS, LangConfig  # noqa: E402
from .network import LangPolicyEncoder, LangTwinCritic, make_actor, n_params, param_groups, set_probe  # noqa: E402

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


@contextlib.contextmanager
def matmul_tf32(on: bool):
    """TF32 matmuls inside the block only (cfg.tf32_train): the flag is process-wide, and the rest of the
    process (rounds collection, cuRobo, probes, evaluation) keeps PyTorch's fp32 default."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


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
                 name: str | None = None, prefetch: bool = True, device: str = "cuda", text_encoder=None,
                 max_rows: dict | None = None):
        self.cfg, self.run = cfg, Path(run_dir).resolve()
        self.dev = device
        self.run.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run / "lang_config.json")
        torch.manual_seed(seed)
        np.random.seed(seed)
        from .data import LangSampler, PrefetchSampler
        from .text import TextStore

        self.sampler = LangSampler(data_roots, cfg, tasks=tasks, embodiments=embodiments, seed=seed, n_threads=16,
                                   max_rows=max_rows)
        n_tok = self.sampler.sources[0].step["scene_tokens"].shape[1] if self.sampler.sources else 1200
        self.policy_enc = LangPolicyEncoder(cfg, n_tok=n_tok).to(device)
        self.actor = make_actor(cfg).to(device)
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
        # probes measure the Section 8.2 quantities whatever the run's own sentence control is: real sentences
        # (no --shuffle-text) and a random rule per sample (no --relabel fixed) (review M3)
        probe_cfg = dataclasses.replace(cfg, shuffle_text=False, relabel="random")
        self.val_sampler = LangSampler(val_roots, probe_cfg, tasks=tasks, embodiments=embodiments, seed=seed + 2,
                                       probe=True) if val_roots else None
        self.prober = None
        self.identity: dict = {}   # code_identity(): saved with every state (set by the rounds / offline drivers)
        self.writer = None         # WriterLock of this run: checked right before a save replaces a file
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
        kw = ({"ref_center": _t(g("ref_center"), dev=d), "ref_valid": _t(g("ref_center_valid"), torch.bool, d)}
              if getattr(self.cfg, "oracle_ref_tag", False) else {})   # V3 (round 3 E2)
        enc = mod(_t(g("state_tokens"), dev=d), _t(g("body_tokens"), dev=d), _t(g("link_pos"), dev=d),
                  _t(g("body_mask"), torch.bool, d), _t(g("joint_index"), torch.int64, d),
                  _t(g("scene_tokens"), dev=d), _t(g("scene_pos"), dev=d), words=words, word_mask=wmask, **kw)
        return enc, _t(g("Jlin"), dev=d), _t(g("Jang"), dev=d), _t(g("joint_index"), torch.int64, d)

    # --------------------------------------------------------------- update --
    def grad_step(self, b: dict | None = None, log: bool = False) -> dict:
        if b is None:
            b = self.prefetch.next() if self.prefetch is not None else self.sampler.sample(self.cfg.batch_size)
            self._last_rng_after = b.get("_rng_after")
        # outside TF32: a sentence missing from the cache is encoded here and cached for collection and probes
        # too, so it must be encoded in fp32 (review round 3, C8)
        words, wmask = self.text.lookup(b["sentence"], splits=["train"] * len(b["sentence"]))
        with matmul_tf32(getattr(self.cfg, "tf32_train", False)):
            return self._grad_step(b, words, wmask, log)

    def _grad_step(self, b: dict, words, wmask, log: bool = False) -> dict:
        cfg = self.cfg
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
        loss = loss_q
        aux_out = {}
        if getattr(cfg, "aux_target_loss", False):   # V4 (decision 5b): trains the encoder and the head
            pred = self.policy_enc.aux_predict(enc)
            tgt, val = _t(b["aux_target"], dev=d), _t(b["aux_valid"], torch.bool, d)
            if bool(val.any()):
                loss_aux = F.huber_loss(pred[val], tgt[val], delta=float(cfg.aux_huber_delta))
                loss = loss_q + float(cfg.aux_loss_weight) * loss_aux
                err = (pred - tgt).norm(dim=-1).detach()
                aux_out = {"aux/loss": float(loss_aux),
                           "aux/center_err_cm_t": 100.0 * float(err[:, 0][val[:, 0]].mean()) if val[:, 0].any() else float("nan"),
                           "aux/center_err_cm_0": 100.0 * float(err[:, 1][val[:, 1]].mean()) if val[:, 1].any() else float("nan")}
        self.opt_e.zero_grad(set_to_none=True)
        self.opt_c.zero_grad(set_to_none=True)
        loss.backward()
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
        out = {"loss/q": float(loss_q), "loss/pi": float(loss_pi), "alpha": alpha, "gamma": gamma, **gn, **aux_out}
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
        miss = self.text.miss_frac()
        out["data/text_miss_frac"] = miss.get("all", 0.0)
        out.update({f"data/text_miss_frac/{sp}": v for sp, v in miss.items() if sp != "all"})
        now = time.perf_counter()
        dt = now - self.t_last
        wait = self.prefetch.wait_s if self.prefetch is not None else 0.0
        out["perf/steps_per_sec"] = LOG_EVERY / max(dt, 1e-9)
        out["perf/data_wait_frac"] = (wait - self.wait_last) / max(dt, 1e-9)
        out["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if self.dev.startswith("cuda") else 0.0
        self.t_last, self.wait_last = now, wait
        return out

    def reset_perf(self) -> None:
        """Restart the throughput timer and the text-miss counts after non-training work (probes, rounds
        collection), so perf/* and data/text_miss_frac describe training steps only (review M9)."""
        self.t_last = time.perf_counter()
        self.wait_last = self.prefetch.wait_s if self.prefetch is not None else 0.0
        self.text.stats.clear()

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
        words, wmask = self.text.lookup(sentences, splits=["probe"] * len(sentences))
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
        if getattr(self.cfg, "aux_target_loss", False):
            out.update(self.aux_probe())
        out["perf/probe_s"] = time.time() - t0
        return out

    @torch.no_grad()
    def aux_probe(self, n: int = 256) -> dict:
        """V4: the head's centre error on validation (cm; mean and median, frames t and 0): does the model locate
        the named item, whether or not Q uses it (decision 5b)."""
        sub = self.val_sampler.sample(n)
        was = self.policy_enc.training
        self.policy_enc.eval()
        words, wmask = self.text.lookup(sub["sentence"], splits=["probe"] * len(sub["sentence"]))
        enc, _, _, _ = self.encode(sub, words, wmask)
        self.policy_enc.train(was)
        err = (self.policy_enc.aux_predict(enc) - _t(sub["aux_target"], dev=self.dev)).norm(dim=-1).cpu().numpy() * 100
        val = np.asarray(sub["aux_valid"], bool)
        out = {}
        for f, name in ((0, "t"), (1, "0")):
            e = err[:, f][val[:, f]]
            if e.size:
                out[f"probe/aux_center_err_cm_{name}"] = float(e.mean())
                out[f"probe/aux_center_err_cm_{name}_median"] = float(np.median(e))
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
                  "switches": self.cfg.path_switches(),
                  # what the weights were trained on (review round 3, B1): evaluations group on it
                  "bead_h_train": _bead_h_of(self.sampler.sources), "identity": dict(self.identity)})
        return d

    def _publish(self, obj, p: Path) -> Path:
        """torch.save to a temp file unique to this process, then replace p, checking the run's writer lock
        right before: a stale process that lost the run never replaces the new owner's file, and two savers
        never share a temp file (review round 3, C3)."""
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        try:
            torch.save(obj, tmp)
            if self.writer is not None:
                self.writer.check()
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        return p

    def save_ckpt(self, name: str = "checkpoint.pt") -> Path:
        p = self._publish(self.ckpt_dict(), self.run / name)
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
              "lang_config": self.cfg.to_dict(), **self.identity, **(extra or {})}
        p = self._publish(st, self.run / "state.pt")
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
                self.reset_perf()
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
    p.add_argument("--embodiments", default="ur5-wsg",
                   help="comma list or 'all' (default ur5-wsg: one embodiment); a resumed run keeps the saved choice")
    p.add_argument("--relabel", choices=("paired", "random", "fixed"), default=None)
    p.add_argument("--shuffle-text", action="store_true")
    p.add_argument("--text-backend", choices=("holobrain_bert", "bert_raw", "sentence_generic"), default=None)
    for s in ("scene", "body", "film", "queries", "rtoken"):
        p.add_argument(f"--no-{s}", action="store_true", help=f"lang_{s} off")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--accept-code-change", action="store_true",
                   help="resume although the code is at another git commit than the run's last save")
    p.add_argument("--time-budget", type=float, default=None, help="seconds; then save state and exit 75")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="LangConfig overrides key=value (JSON values)")
    p.add_argument("--text-cache", nargs="*", default=None,
                   help="text caches (default: text_cache.pt next to --data / --val directories)")


def config_from_args(args) -> LangConfig:
    """--resume continues with the run's saved config only once the run has state (state.pt). Before that
    nothing was trained, and the trainer writes lang_config.json at start: a first job that died early must
    not pin its flags on a corrected relaunch (review O4)."""
    run = Path(args.run)
    saved = run / "lang_config.json"
    if getattr(args, "resume", False) and saved.exists() and (run / "state.pt").exists():
        return LangConfig.load(saved)
    cfg = _config_from_flags(args)
    if getattr(args, "resume", False) and saved.exists():
        old = LangConfig.load(saved).to_dict()
        diff = {k: (old.get(k), v) for k, v in cfg.to_dict().items()
                if k not in ("collect_rounds", "episodes_per_round", "grad_steps_per_round") and old.get(k) != v}
        if diff:
            print(f"[lang-train] {saved} has no state.pt yet: using these flags, not the saved ones {diff}")
    return cfg


def _bead_h_of(sources) -> str:
    from .data import bead_h_of

    return bead_h_of(sources)


def code_identity(tasks, roots) -> dict:
    """What a run's stored rows and weights depend on, saved in state.pt (review rounds 2 and 3, C4): the h
    definition (h_code_v2 over the run's tasks; h_code, the first hash, for older states), the resolved
    training data roots and the git commit of the code that runs (plus whether its checkout is dirty)."""
    import subprocess

    from .collect import h_code_hash, h_code_hash_v2

    repo = str(Path(__file__).resolve().parents[2])
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        head, dirty = "", None
    return {"h_code": h_code_hash(CANON_TASKS), "h_code_v2": h_code_hash_v2(tasks or CANON_TASKS),
            "git_head": head, "git_dirty": dirty, "data_roots": sorted(str(Path(r).resolve()) for r in roots)}


def check_resume_identity(st: dict, ident: dict, run, accept_code_change: bool = False) -> None:
    """Refuse a resume whose h definition or training data differ from the run's start, or whose code is at
    another git commit (unless accept_code_change: a deliberate, logged update)."""
    run = Path(run)
    key = "h_code_v2" if st.get("h_code_v2") else "h_code"
    if st.get(key) and st[key] != ident[key]:
        # the stored rows were labelled under another h definition: mixing them silently would corrupt the
        # training targets. Run from the code the run started with.
        raise SystemExit(f"{run}: the h definition changed since this run started ({key} {st[key]}, now "
                         f"{ident[key]}; started at git {str(st.get('git_head') or '?')[:10]}): resume from that "
                         "code (a worktree at that commit) or start a new run")
    if st.get("data_roots") is not None and list(st["data_roots"]) != list(ident["data_roots"]):
        a, b = set(st["data_roots"]), set(ident["data_roots"])
        raise SystemExit(f"{run}: the training data roots changed since this run started (missing "
                         f"{sorted(a - b)[:3]}, new {sorted(b - a)[:3]}, {len(a)} -> {len(b)} roots): check --data")
    if st.get("git_head") and ident.get("git_head") and st["git_head"] != ident["git_head"]:
        if not accept_code_change:
            raise SystemExit(f"{run}: this code is at git {ident['git_head'][:10]}, the run's last save at "
                             f"{st['git_head'][:10]}: resume from that commit, or pass --accept-code-change")
        print(f"[lang-train] code changed {st['git_head'][:10]} -> {ident['git_head'][:10]} (--accept-code-change)",
              flush=True)


def trained_subsets(cfg) -> tuple:
    """(tasks, embodiments) a run trains and collects on, from its config (fixed at the first start, in
    lang_config.json), not from this launch's flags, so a resume cannot drift; None = all."""
    tasks, embs = tuple(cfg.task_choices), tuple(cfg.embodiment_choices)
    return (None if tasks == CANON_TASKS else tasks), (None if embs == CANON_EMBODIMENTS else embs)


def _config_from_flags(args) -> LangConfig:
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
    p.add_argument("--freeze-data", action="store_true",
                   help="train on the rows present at the first start (data_manifest.json), even if shards grow")
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
    tasks, embs = trained_subsets(cfg)
    roots = roots_of(args.data)
    vals = roots_of(args.val) if args.val else []
    # an empty glob must not train on the ring alone or skip every probe without a word (review round 3, C4)
    if not roots:
        raise SystemExit(f"--data {args.data}: no data shards found")
    if args.val and not vals:
        raise SystemExit(f"--val {args.val}: no validation shards found")
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.data) + caches_of(args.val)
    if args.mode == "rounds":
        from .rounds import run_rounds

        return run_rounds(cfg, args, roots, vals, caches, tasks, embs)
    if not args.resume and (Path(args.run) / "state.pt").exists():
        raise SystemExit(f"{args.run} has a state.pt: pass --resume or use a new run directory")
    max_rows = None
    if args.freeze_data:   # the dataset is the rows present at the first start (shards may still grow)
        man = Path(args.run) / "data_manifest.json"
        if not man.exists():
            Path(args.run).mkdir(parents=True, exist_ok=True)
            rows = {str(r.resolve()): int(json.loads((r / "item_buffer" / "header.json").read_text()).get("n", 0))
                    for r in roots if (r / "item_buffer" / "header.json").exists()}
            man.write_text(json.dumps({"frozen_at": time.ctime(), "rows": rows,
                                       "total": sum(rows.values())}, indent=1))
        max_rows = json.loads(man.read_text())["rows"]
        print(f"[lang-train] frozen dataset: {sum(max_rows.values())} rows in {len(max_rows)} roots ({man})")
    from .locks import WriterLock

    # one writer per run (review round 3, C3): a second process waits out a crashed one's lock, then refuses
    Path(args.run).mkdir(parents=True, exist_ok=True)
    try:
        writer = WriterLock(Path(args.run), wait_s=200.0)
    except WriterLock.LockHeld as exc:
        raise SystemExit(f"[lang-train] {exc}: is another job training this run?")
    try:
        tr = LangTrainer(cfg, Path(args.run), roots, vals, caches, tasks=tasks, embodiments=embs,
                         use_wandb=not args.no_wandb, seed=args.seed, max_rows=max_rows)
        tr.identity, tr.writer = code_identity(tasks or cfg.task_choices, roots), writer
        if args.resume and (tr.run / "state.pt").exists():
            st = tr.load_state()
            check_resume_identity(st, tr.identity, tr.run, args.accept_code_change)
            print(f"[lang-train] resumed at step {tr.step}", flush=True)
        elif args.init_weights:
            tr.load_weights(args.init_weights)
        return tr.train_offline(args.grad_steps, args.probe_every, args.time_budget, args.ckpt_every)
    except WriterLock.LockLost as exc:   # another process owns the run now: write nothing more
        print(f"[lang-train] {exc} (exit 75)", flush=True)
        return 75
    finally:
        writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
