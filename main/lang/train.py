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

V5 (``--set scene_fusion=true``, FUSION_PLAN; contract fusion_impl.md 4.4):
version-2 shards only. Each step fuses the 5 frames of every sample (t, t-k, 0,
t+1, t+1-k) with the sample's own sentence in HoloBrain's frozen fusion stage
(``FusionStage``, built once from the checkpoint, never part of the optimised,
copied or saved modules), outside the TF32 block (the stage sets its own
precision, cfg.fusion_dtype). The live encoder gets the fused frames t, t-k, 0,
the target encoder t+1, t+1-k, 0 (frame 0 fused once per sample); with
fusion_words the text adapter reads the fused words of frame t (t+1 for the
target). Probes and sanity checks fuse the same way. Rounds mode refuses it (no
neck ring yet: rounds.run_rounds).
"""

from __future__ import annotations

import argparse
import collections
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
from .neck_buffer import ROW_FIELDS  # noqa: E402
from .network import LangPolicyEncoder, LangTwinCritic, make_actor, n_params, param_groups, set_probe  # noqa: E402

EXIT_RESUME = 75
LOG_EVERY = 20
STATE_EVERY = 500
# every policy-encoder part a switch can add must be listed, or a checkpoint silently drops it (the V3 tag and the V4
# head were missing until 2026-10-08: V3 checkpoints held no tag and were scored with a zero one). V5 adds no part:
# the fusion stage is frozen and rebuilt from the HoloBrain checkpoint, so consumers read cfg.scene_fusion
CKPT_PARTS = ("injection", "scene_set", "text_adapter", "scene_reads_words", "r_static", "blocks", "e_tag", "aux_head")
# neck_idx columns (data.NECK_FRAMES: t, t-k, 0, t+1, t+1-k) of the live stack [t, t-k, 0] and the target
# stack [t+1, t+1-k, 0]: frame 0 is the same entry in both
LIVE_COLS, TARGET_COLS = (0, 1, 2), (3, 4, 2)


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


# ---------------------------------------------------------------------------
# V5: the frozen fusion stage on stored neck rows (fusion_impl.md 4.1, 4.4)
# ---------------------------------------------------------------------------


def make_fusion(cfg, device="cuda", spatial_enhancer=None):
    """The frozen fusion stage of a scene_fusion config, None when it is off: HoloBrain's pretrain/ checkpoint (the
    file of our neck, spatial enhancer and text_feat_map; $HOLOBRAIN_CKPT via main.train.collect.CKPT_DIR) at the
    config's precision and chunk, the same in training, evaluation and the rollout. spatial_enhancer: the rollout's
    HoloBrainEncoder one (no second copy); training and offline evaluation build their own."""
    if not getattr(cfg, "scene_fusion", False):
        return None
    from main.train.collect import CKPT_DIR

    from .fusion import FusionStage

    return FusionStage(str(CKPT_DIR), device=device, dtype=cfg.fusion_dtype, chunk=int(cfg.fusion_chunk),
                       spatial_enhancer=spatial_enhancer)


def fusion_flags(lang_config: dict) -> tuple[bool, bool]:
    """(scene_fusion, fused words) of a saved lang_config: what the scene tokens and the words the weights were
    trained on are. A config saved before V5 has neither (unfused tokens, cached words)."""
    sf = bool((lang_config or {}).get("scene_fusion", False))
    return sf, sf and bool(lang_config.get("fusion_words", True))


class SharedNeck(dict):
    """A batch's neck rows {field: [U, ...]}, one table its samples share (neck_idx points into it).

    metrics.Prober builds per-state sub-batches as ``[v[k]] * n`` for values that are not arrays; an integer index
    returns the table itself, so such a sub-batch keeps the whole table next to its own neck_idx rows (neck_table
    undoes the list). Field names index as in a dict."""

    def __getitem__(self, k):
        return self if isinstance(k, (int, np.integer)) else super().__getitem__(k)


def neck_table(x) -> dict:
    """b["neck"] as the sampler gives it, or the per-state list of one SharedNeck (Prober sub-batches)."""
    if isinstance(x, (list, tuple)):
        if not x or any(t is not x[0] for t in x):
            raise ValueError("a batch's neck rows must be one table shared by its samples")
        x = x[0]
    return x


def probe_neck(batch: dict) -> dict:
    """A probe set's neck rows for frames t, t-k, 0 only (probes never encode the t+1 stack: about 3 GB of host
    memory for 512 samples instead of 5), as a SharedNeck. Columns 3, 4 of neck_idx become -1 (fused() refuses
    them)."""
    nidx = np.asarray(batch["neck_idx"], np.int64)
    used, inv = np.unique(nidx[:, list(LIVE_COLS)].ravel(), return_inverse=True)
    new = np.full_like(nidx, -1)
    new[:, list(LIVE_COLS)] = inv.reshape(-1, len(LIVE_COLS))
    table = neck_table(batch["neck"])
    return {"neck": SharedNeck({k: np.ascontiguousarray(v[used]) for k, v in table.items()}), "neck_idx": new}


def probe_kv(batch: dict) -> dict:
    """A probe set's prefix keys / values for frame t only (probes encode no t+1 stack), as a SharedNeck the
    Prober's per-state sub-batches share; column 1 of kv_idx becomes -1 (qwen_words refuses it)."""
    kidx = np.asarray(batch["kv_idx"], np.int64)
    used, inv = np.unique(kidx[:, 0], return_inverse=True)
    new = np.full_like(kidx, -1)
    new[:, 0] = inv
    table = neck_table(batch["kv"])
    return {"kv": SharedNeck({k: np.ascontiguousarray(np.asarray(v)[used]) for k, v in table.items()}), "kv_idx": new}


def make_qwen(cfg, device="cuda"):
    """The trainer's frozen HoloBrain-Qwen text pass (V6): layer 0 and feat_mapping, without the vision tower."""
    from .qwen_encoder import HoloBrainQwenEncoder

    return HoloBrainQwenEncoder(device=device, attn=cfg.qwen_attn, vision=False)


def neck_upload(rows: dict, device) -> dict:
    """numpy neck rows -> device tensors of the five fusion fields, each in its stored dtype (fp16 maps stay fp16:
    the stage casts each chunk to fp32, so a batch's ~250 frames take 0.7 GB, not 1.5), staged through pinned
    memory on CUDA."""
    cuda = str(device).startswith("cuda")
    out = {}
    for k in ROW_FIELDS:
        a = np.asarray(rows[k])
        if not a.flags.writeable:
            a = a.copy()
        t = torch.from_numpy(np.ascontiguousarray(a))
        out[k] = t.pin_memory().to(device, non_blocking=True) if cuda else t.to(device)
    return out


def fuse_neck_rows(fusion, rows: dict, entry, words, wmask, device):
    """The fusion of F frames: frame f is neck row entry[f] of rows ({field: np [U, ...]}) fused with words[f] /
    wmask[f] -> fusion.fuse = (tokens [F, n, 256], positions [F, n, 3], fused words [F, L, 256]). The rows the
    frames use are uploaded once and gathered on the device."""
    entry = np.asarray(entry, np.int64)
    n_rows = int(np.asarray(rows["neck_maps"]).shape[0])
    used, inv = np.unique(entry, return_inverse=True)
    if used.size != n_rows:   # some rows are not needed (probe sub-batches): upload the used ones only
        rows = {k: np.asarray(rows[k])[used] for k in ROW_FIELDS}
    dev = neck_upload(rows, device)
    gather = torch.as_tensor(inv.reshape(-1), dtype=torch.long, device=device)
    return fusion.fuse({k: t[gather] for k, t in dev.items()}, words, wmask)


class StepTimer:
    """Durations of named spans (perf/fusion_ms, perf/step_ms), read at log time: CUDA events on a GPU (wall time
    without a sync is wrong there: a step's last kernels are still queued when the next step starts),
    perf_counter on the CPU. A span is (mark(), mark()); at most the last `keep` spans per name are kept."""

    def __init__(self, cuda: bool, keep: int = 4 * LOG_EVERY):
        self.cuda = bool(cuda)
        self.keep = int(keep)
        self.spans: dict[str, collections.deque] = {}

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def add(self, name: str, t0, t1) -> None:
        self.spans.setdefault(name, collections.deque(maxlen=self.keep)).append((t0, t1))

    def read(self) -> dict:
        """{name: (mean ms per span, number of spans)} since the last read or clear; then cleared."""
        out = {}
        for name, spans in self.spans.items():
            if not spans:
                continue
            if self.cuda:
                spans[-1][1].synchronize()   # spans of one name are recorded in order on one stream
                ms = [a.elapsed_time(b) for a, b in spans]
            else:
                ms = [1000.0 * (b - a) for a, b in spans]
            out[name] = (float(np.mean(ms)), len(ms))
        self.spans.clear()
        return out

    def clear(self) -> None:
        self.spans.clear()


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
                 max_rows: dict | None = None, fusion=None, qwen=None):
        """fusion: a FusionStage-like object (fuse(neck, words, wmask)) to use with cfg.scene_fusion instead of
        building one from the checkpoint (shared by run_sanity's trainers; the CPU tests' stub). qwen (V6): a
        HoloBrainQwenEncoder-like object (encode_text(kv, sentences, kv_index), blank_kv(), ckpt_id) to use with
        encoder holobrain_qwen instead of building the text pass (shared with rounds collection; the CPU stub)."""
        self.cfg, self.run = cfg, Path(run_dir).resolve()
        self.dev = device
        self.run.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run / "lang_config.json")
        torch.manual_seed(seed)
        np.random.seed(seed)
        from .data import LangSampler, PrefetchSampler
        from .legacy_pool import install_vocab
        from .text import TextStore

        install_vocab(cfg.vocab)   # before the samplers name anything (RelabelSampler re-names records at load)

        # V5: version-2 shards' neck rows (a version-1 shard is a ValueError); E6: the training sampler alone draws
        # from the first train_episodes_per_task episodes per (task, embodiment)
        self.sampler = LangSampler(data_roots, cfg, tasks=tasks, embodiments=embodiments, seed=seed, n_threads=16,
                                   max_rows=max_rows, neck=bool(cfg.scene_fusion),
                                   episodes_per_task=int(cfg.train_episodes_per_task), kv=cfg.qwen)
        n_tok = (self.sampler.sources[0].step["scene_tokens"].shape[1] if self.sampler.sources
                 else cfg.scene_tokens_per_frame)
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
        # V6: the words come from HoloBrain-Qwen's layer 0 against each frame's stored prefix (qwen_words), no store
        self.text = None if cfg.qwen else TextStore(text_caches, cfg.text_backend, device,
                                                     extra_path=self.run / "text_cache_extra.pt",
                                                     backend_paths=cfg.backend_paths, encoder=text_encoder)
        self.prefetch = PrefetchSampler(self.sampler, cfg.batch_size, seed=seed + 1) if prefetch else None
        # probes measure the Section 8.2 quantities whatever the run's own sentence control is: real sentences
        # (no --shuffle-text) and a random rule per sample (no --relabel fixed) (review M3), on every validation
        # episode (no E6 subset)
        probe_cfg = dataclasses.replace(cfg, shuffle_text=False, relabel="random", train_episodes_per_task=0)
        self.val_sampler = LangSampler(val_roots, probe_cfg, tasks=tasks, embodiments=embodiments, seed=seed + 2,
                                       probe=True, neck=bool(cfg.scene_fusion), kv=cfg.qwen) if val_roots else None
        self.qwen = None
        if cfg.qwen:
            self.qwen = qwen if qwen is not None else make_qwen(cfg, device)
            for smp in (self.sampler, self.val_sampler):
                for src in (smp.sources if smp is not None else ()):
                    got = src.kv.header.get("ckpt_id")
                    if got is not None and got != self.qwen.ckpt_id:
                        raise ValueError(f"{src.root}: kv rows of HoloBrain-Qwen checkpoint {got}, the text pass is "
                                         f"{self.qwen.ckpt_id} (never mix checkpoints)")
                    # the front-end version and kernel the rows were computed with: the text pass reads them with its
                    # own (version 1 rounded the RoPE buffers to bf16; scripts/lang/qwen_recompute.py rewrites rows)
                    want_v = getattr(self.qwen, "ENCODER_VERSION", None)
                    have_v = int(src.kv.header.get("encoder_version", 1))
                    if want_v is not None and have_v != int(want_v):
                        raise ValueError(f"{src.root}: kv rows of HoloBrain-Qwen front-end version {have_v}, this code is "
                                         f"version {want_v}: recompute them (scripts/lang/qwen_recompute.py)")
                    have_a = src.kv.header.get("attn")
                    if want_v is not None and have_a is not None and have_a != cfg.qwen_attn:
                        raise ValueError(f"{src.root}: kv rows computed with attention {have_a}, this run uses "
                                         f"{cfg.qwen_attn}")
        self._qwen_passes = 0       # sentence passes of the training steps since the last log (V6)
        self.last_qwen_passes = 0
        # V5: HoloBrain's frozen fusion stage, a plain object next to the model (never a submodule of what is
        # optimised, deep-copied for the target or saved); its weights come from the HoloBrain checkpoint
        self.fusion = None
        if cfg.scene_fusion:
            self.fusion = fusion if fusion is not None else make_fusion(cfg, device)
        self.timer = StepTimer(self.dev.startswith("cuda"))
        self._step_t0 = None
        self._fusion_frames = 0     # frames fused by the training steps since the last log
        self.last_fused_frames = 0  # frame passes of the last fused() call
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
              f"from {len(self.sampler.sources)} sources | text cache {len(self.text) if self.text is not None else '-'} | params "
              f"enc {n_params(self.policy_enc.parameters())} critics {n_params(self.critics.parameters())} "
              f"actor {n_params(self.actor.parameters())}")
        if self.fusion is not None:
            print(f"[lang-train] scene_fusion: frozen fusion stage {cfg.fusion_dtype}, chunk {cfg.fusion_chunk}, "
                  f"fused words {cfg.fused_words}")
        if self.qwen is not None:
            print(f"[lang-train] encoder holobrain_qwen: {n_tok} scene tokens per frame, words from layer 0 against "
                  f"the {cfg.qwen_words_prefix} prefix, checkpoint {self.qwen.ckpt_id}, attn {cfg.qwen_attn}")
        if cfg.train_episodes_per_task:
            print(f"[lang-train] train_episodes_per_task {cfg.train_episodes_per_task}: episodes "
                  + ", ".join(f"{t}/{e} {n}" for (t, e), n in self.sampler.episode_counts().items()))

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
        if self.fusion is not None:
            # V5: the stored scene tokens are the unfused ones; a batch must pass through fused() first
            if suffix not in b.get("_fused", ()):
                raise RuntimeError(f"scene_fusion: the batch's frames{suffix or ' (live)'} are not fused "
                                   "(LangTrainer.fused)")
            if self.cfg.fused_words:
                words = g("w_fused")   # frame t's fused words; t+1's for the target, never t's (review R5)
        if self.qwen is not None:
            # V6: each frame's own words (layer 0 reads that frame's prefix); the target reads frame t+1's
            if suffix not in b.get("_qwen", ()):
                raise RuntimeError(f"encoder holobrain_qwen: the batch has no words for frames{suffix or ' (live)'} "
                                   "(LangTrainer.qwen_words)")
            words, wmask = g("w_qwen"), g("wm_qwen")
        kw = ({"ref_center": _t(g("ref_center"), dev=d), "ref_valid": _t(g("ref_center_valid"), torch.bool, d)}
              if getattr(self.cfg, "oracle_ref_tag", False) else {})   # V3 (round 3 E2)
        enc = mod(_t(g("state_tokens"), dev=d), _t(g("body_tokens"), dev=d), _t(g("link_pos"), dev=d),
                  _t(g("body_mask"), torch.bool, d), _t(g("joint_index"), torch.int64, d),
                  _t(g("scene_tokens"), dev=d), _t(g("scene_pos"), dev=d), words=words, word_mask=wmask, **kw)
        return enc, _t(g("Jlin"), dev=d), _t(g("Jang"), dev=d), _t(g("joint_index"), torch.int64, d)

    def fused(self, b: dict, words, wmask, sentences=None, target: bool = True) -> dict:
        """V5 (scene_fusion): a shallow copy of b whose scene frames are fused with each sample's sentence.

        words / wmask: the samples' TextStore words (the sentences' order; ``sentences`` default b["sentence"]).
        One fusion pass per unique (neck row, sentence): the two members of a pair share frames, not sentences;
        frame 0, and frames shared by samples with one sentence (t = 0, repeated draws), are fused once. Writes
        scene_tokens / scene_pos [B, 3, n, *] (frames t, t-k, 0) and w_fused (frame t's fused words) and, with
        target, scene_tokens_next / scene_pos_next (t+1, t+1-k, 0: the same frame-0 pass) and w_fused_next (t+1's).
        No-op when scene_fusion is off."""
        if self.fusion is None:
            return b
        if "neck" not in b or "neck_idx" not in b:
            raise ValueError("scene_fusion: the batch has no neck rows (a sampler built without neck=True)")
        table = neck_table(b["neck"])
        nidx = np.asarray(b["neck_idx"], np.int64)
        sents = list(b["sentence"] if sentences is None else sentences)
        cols = LIVE_COLS + TARGET_COLS[:2] if target else LIVE_COLS
        if bool((nidx[:, list(cols)] < 0).any()):
            raise ValueError("scene_fusion: the batch has no neck rows for some frames (a probe set: live frames only)")
        first, passes, entry, who = {}, {}, [], []
        p = np.full(nidx.shape, -1, np.int64)
        for i, s in enumerate(sents):
            w = first.setdefault(s, i)          # the words of a sentence: its first sample's
            for c in cols:
                key = (int(nidx[i, c]), s)
                if key not in passes:
                    passes[key] = len(entry)
                    entry.append(key[0])
                    who.append(w)
                p[i, c] = passes[key]
        sel = torch.as_tensor(who, dtype=torch.long, device=words.device)
        tok, pos, wf = fuse_neck_rows(self.fusion, table, entry, words[sel], wmask[sel], self.dev)
        self.last_fused_frames = len(entry)
        pt = torch.as_tensor(p, dtype=torch.long, device=tok.device)
        out = dict(b)
        live = pt[:, list(LIVE_COLS)]
        out["scene_tokens"], out["scene_pos"], out["w_fused"] = tok[live], pos[live], wf[pt[:, 0]]
        if target:
            targ = pt[:, list(TARGET_COLS)]
            out["scene_tokens_next"], out["scene_pos_next"] = tok[targ], pos[targ]
            out["w_fused_next"] = wf[pt[:, TARGET_COLS[0]]]
        out["_fused"] = ("", "_next") if target else ("",)
        return out

    def qwen_words(self, b: dict, sentences=None, target: bool = True) -> dict:
        """V6 (encoder holobrain_qwen): a shallow copy of b with each sample's words from HoloBrain-Qwen's layer 0,
        the sentence read against the stored prefix keys / values of its frame (qwen_words_prefix "frame") or of
        the black-image prefix ("blank").

        One text pass per unique (prefix row, sentence) over frame t (kv_idx column 0) and, with target, frame t+1
        (column 1): the two members of a pair share the row, not the sentence. Writes w_qwen / wm_qwen (frame t) and
        w_qwen_next / wm_qwen_next (t+1) and the "_qwen" marker. Frozen, no grad, in the encoder's precision (bf16),
        outside the TF32 block (the rollout runs the same pass). No-op under GD."""
        if self.qwen is None:
            return b
        from .kv_buffer import rows_to_kv

        sents = list(b["sentence"] if sentences is None else sentences)
        cols = (0, 1) if target else (0,)
        blank = getattr(self.cfg, "qwen_words_prefix", "frame") == "blank"
        if not blank:
            if "kv" not in b or "kv_idx" not in b:
                raise ValueError("encoder holobrain_qwen: the batch has no prefix rows (a sampler built without kv)")
            kidx = np.asarray(b["kv_idx"], np.int64)
            if bool((kidx[:, list(cols)] < 0).any()):
                raise ValueError("encoder holobrain_qwen: no prefix rows for some frames (a probe set: frame t only)")
        passes, entry, who = {}, [], []
        p = np.full((len(sents), 2), -1, np.int64)
        for i, s in enumerate(sents):
            for c in cols:
                key = (-1 if blank else int(kidx[i, c]), s)
                if key not in passes:
                    passes[key] = len(entry)
                    entry.append(key[0])
                    who.append(s)
                p[i, c] = passes[key]
        with torch.no_grad():
            if blank:
                kv, index = self.qwen.blank_kv(), torch.zeros(len(entry), dtype=torch.long)
            else:
                table = neck_table(b["kv"])
                used, inv = np.unique(np.asarray(entry, np.int64), return_inverse=True)
                cuda = str(self.dev).startswith("cuda")
                arrs = []
                for k in ("k", "v"):
                    t = torch.from_numpy(np.ascontiguousarray(np.asarray(table[k])[used]))
                    arrs.append(t.pin_memory() if cuda else t)
                kv, index = rows_to_kv(arrs[0], arrs[1], device=self.dev), torch.as_tensor(inv.reshape(-1))
            words, wmask = self.qwen.encode_text(kv, who, kv_index=index.to(self.dev))
        self.last_qwen_passes = len(entry)
        pt = torch.as_tensor(p, dtype=torch.long, device=words.device)
        out = dict(b)
        out["w_qwen"], out["wm_qwen"] = words[pt[:, 0]], wmask[pt[:, 0]]
        if target:
            out["w_qwen_next"], out["wm_qwen_next"] = words[pt[:, 1]], wmask[pt[:, 1]]
        out["_qwen"] = ("", "_next") if target else ("",)
        return out

    # --------------------------------------------------------------- update --
    def grad_step(self, b: dict | None = None, log: bool = False) -> dict:
        if b is None:
            b = self.prefetch.next() if self.prefetch is not None else self.sampler.sample(self.cfg.batch_size)
            self._last_rng_after = b.get("_rng_after")
        t0 = self.timer.mark()
        if self.qwen is not None:
            # V6, outside the TF32 block: the frozen text pass in the encoder's own precision, as the rollout runs it
            q0 = self.timer.mark()
            b = self.qwen_words(b)
            self.timer.add("qwen_text", q0, self.timer.mark())
            self._qwen_passes += self.last_qwen_passes
            words, wmask = b["w_qwen"], b["wm_qwen"]
        else:
            # outside TF32: a sentence missing from the cache is encoded here and cached for collection and probes
            # too, so it must be encoded in fp32 (review round 3, C8)
            words, wmask = self.text.lookup(b["sentence"], splits=["train"] * len(b["sentence"]))
        if self.fusion is not None:
            # V5, also outside the TF32 block: the stage sets its own precision (cfg.fusion_dtype), the one the
            # evaluation and the rollout use, so training never sees other fused tokens than deployment
            f0 = self.timer.mark()
            b = self.fused(b, words, wmask)
            self.timer.add("fusion", f0, self.timer.mark())
            self._fusion_frames += self.last_fused_frames
        self._step_t0 = t0
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
        use_aux, use_att = getattr(cfg, "aux_target_loss", False), getattr(cfg, "aux_attention_loss", False)
        if use_aux or use_att:
            tgt, val = _t(b["aux_target"], dev=d), _t(b["aux_valid"], torch.bool, d)   # [B, 2, 3]: frames t, 0
            unit = float(getattr(cfg, "aux_unit", 0.05))
            lq = float(loss_q.detach().clamp_min(1e-12))
        if use_aux and bool(val.any()):
            # V4 (decision 5b; round 4, item 3): in units of aux_unit, Huber delta in those units
            pred = self.policy_enc.aux_predict(enc)
            loss_aux = F.huber_loss(pred[val] / unit, tgt[val] / unit, delta=float(cfg.aux_huber_delta))
            loss = loss + float(cfg.aux_loss_weight) * loss_aux
            err = (pred - tgt).norm(dim=-1).detach()
            aux_out.update({"aux/loss": float(loss_aux), "aux/ratio": float(cfg.aux_loss_weight) * float(loss_aux) / lq,
                            "aux/center_err_cm_t": 100.0 * float(err[:, 0][val[:, 0]].mean()) if val[:, 0].any() else float("nan"),
                            "aux/center_err_cm_0": 100.0 * float(err[:, 1][val[:, 1]].mean()) if val[:, 1].any() else float("nan")})
        if use_att and bool(val.any()):
            # V4b: q1's head-averaged scene attention of the last encoder block, renormalized within frame f in
            # {t, 0} (scene frames 0 and 2), times the squared distance of each token to the named item's centre
            att = self.policy_enc.blocks[-1].q_attn[:, 0]                              # [B, 3 * n_tok]
            Bn = att.shape[0]
            att = att.reshape(Bn, 3, -1)
            # the positions the encoder attended over: with scene_fusion the FUSED ones (the spatial enhancer
            # predicts depth from the fused maps, so they move with the sentence); else the stored ones, same values
            pos = enc.scene_pos.reshape(Bn, 3, att.shape[-1], 3)
            terms = []
            for k_f, f in ((0, 0), (1, 2)):
                a_f = att[:, f] / att[:, f].sum(-1, keepdim=True).clamp_min(1e-12)
                d2 = ((pos[:, f] - tgt[:, k_f, None, :]) ** 2).sum(-1) / unit ** 2       # [B, n_tok]
                terms.append(torch.where(val[:, k_f], (a_f * d2).sum(-1), torch.zeros_like(d2[:, 0])))
            loss_att = (terms[0] + terms[1]).sum() / val.sum().clamp_min(1)
            loss = loss + float(cfg.aux_attention_weight) * loss_att
            aux_out.update({"aux/att_loss": float(loss_att),
                            "aux/att_ratio": float(cfg.aux_attention_weight) * float(loss_att) / lq,
                            # attention-weighted distance of q1's look to the named item (cm), frame t
                            "aux/att_dist_cm_t": 100.0 * unit * float(torch.sqrt(terms[0][val[:, 0]].clamp_min(0)).mean())
                            if val[:, 0].any() else float("nan")})
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
        if self._step_t0 is not None:   # the whole step: text lookup, fusion, updates, Polyak (perf/step_ms)
            self.timer.add("step", self._step_t0, self.timer.mark())
            self._step_t0 = None
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
        if self.text is not None:
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
        # per step, averaged over the steps since the last log: the whole step and (V5) its fusion, CUDA-timed
        spans = self.timer.read()
        out.update({f"perf/{k}_ms": ms for k, (ms, _) in spans.items()})
        if "fusion" in spans:   # frames per step: fusion_ms / fusion_frames is the cost of one frame-pass
            out["perf/fusion_frames"] = self._fusion_frames / max(spans["fusion"][1], 1)
        self._fusion_frames = 0
        if "qwen_text" in spans:   # sentence passes per step (V6): qwen_text_ms / qwen_passes per pass
            out["perf/qwen_passes"] = self._qwen_passes / max(spans["qwen_text"][1], 1)
        self._qwen_passes = 0
        # repeats per training episode so far (FUSION_PLAN C1: every E5 / E6 table reports it)
        n_ep = sum(self.sampler.episode_counts().values())
        out["data/samples_per_episode"] = self.step * int(self.cfg.batch_size) / max(n_ep, 1)
        return out

    def reset_perf(self) -> None:
        """Restart the throughput timer and the text-miss counts after non-training work (probes, rounds
        collection), so perf/* and data/text_miss_frac describe training steps only (review M9)."""
        self.t_last = time.perf_counter()
        self.wait_last = self.prefetch.wait_s if self.prefetch is not None else 0.0
        if self.text is not None:
            self.text.stats.clear()
        self.timer.clear()
        self._fusion_frames = 0
        self._qwen_passes = 0

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
        if self.qwen is not None:   # V6: THESE sentences against frame t's prefix (the probes vary the sentence)
            sub = self.qwen_words(sub, sentences=list(sentences), target=False)
            words, wmask = sub["w_qwen"], sub["wm_qwen"]
        else:
            words, wmask = self.text.lookup(sentences, splits=["probe"] * len(sentences))
        # V5: the frames fused with THESE sentences (the shuffled / paraphrase probes change the fused tokens too)
        sub = self.fused(sub, words, wmask, sentences=list(sentences), target=False)
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
            if "neck" in self.prober.batch:   # V5: frames t, t-k, 0 only, one table the per-state sub-batches share
                self.prober.batch.update(probe_neck(self.prober.batch))
            if "kv" in self.prober.batch:     # V6: frame t's prefix rows only, one shared table
                self.prober.batch.update(probe_kv(self.prober.batch))
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
        if self.qwen is not None:
            sub = self.qwen_words(sub, target=False)
            words, wmask = sub["w_qwen"], sub["wm_qwen"]
        else:
            words, wmask = self.text.lookup(sub["sentence"], splits=["probe"] * len(sub["sentence"]))
        sub = self.fused(sub, words, wmask, target=False)
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
                  "bead_h_train": _bead_h_of(self.sampler.sources), "identity": dict(self.identity),
                  "encoder": self.cfg.encoder,
                  **({"qwen_ckpt_id": self.qwen.ckpt_id} if self.qwen is not None else {})})
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
        if self.text is not None:
            self.text.save_extra()
        return p

    def load_weights(self, path, strict_switches: bool = True) -> None:
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        if strict_switches and ck.get("switches") != self.cfg.path_switches():
            raise ValueError(f"{path}: switches {ck.get('switches')} != {self.cfg.path_switches()}")
        if ck.get("text_backend") != self.cfg.text_backend:
            raise ValueError(f"{path}: text backend {ck.get('text_backend')} != {self.cfg.text_backend}")
        # V6: no parameter depends on the scene width, so a GD checkpoint would load strictly into a Qwen model
        theirs_enc = (ck.get("lang_config") or {}).get("encoder", "holobrain_gd")
        if theirs_enc != self.cfg.encoder:
            raise ValueError(f"{path}: trained on the {theirs_enc} front end, this run has {self.cfg.encoder}")
        # V5 adds no parameter, so the parts cannot tell: weights trained on unfused tokens (or cached words) are no
        # start for fused ones, and back
        theirs, mine = fusion_flags(ck.get("lang_config") or {}), fusion_flags(self.cfg.to_dict())
        if theirs != mine:
            raise ValueError(f"{path}: trained with (scene_fusion, fused words) = {theirs}, this run has {mine}")
        # decision 6: the readouts have different parameters (value heads vs readout_*); say so before the strict
        # load does. A config saved before the field trained the softmin
        readout = (ck.get("lang_config") or {}).get("critic_readout", "softmin")
        if readout != self.cfg.critic_readout:
            raise ValueError(f"{path}: critic readout {readout}, this run has {self.cfg.critic_readout} "
                             "(--set 'critic_readout=\"softmin\"' to continue a softmin model)")
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
        if self.text is not None:
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
        if key not in ck:
            raise ValueError(f"the model has {key} but the checkpoint does not (saved before {key} was checkpointed? "
                             "re-export it from the run's state.pt)")
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

    base = LangSampler(roots, cfg0, seed=0, neck=bool(cfg0.scene_fusion), kv=cfg0.qwen)
    stage = make_fusion(cfg0)   # V5: one frozen stage for the fusing trainers below (None when off)
    qwen = make_qwen(cfg0) if cfg0.qwen else None   # V6: one frozen text pass for every trainer below
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
            if tr.qwen is not None:
                b = tr.qwen_words(b)
                words, wmask = b["w_qwen"], b["wm_qwen"]
            else:
                words, wmask = tr.text.lookup(b["sentence"])
            b = tr.fused(b, words, wmask)
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
        tr = LangTrainer(cfg, res_dir / tag, roots, (), caches, use_wandb=False, prefetch=False, seed=0, fusion=stage,
                         qwen=qwen)
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
    # switches off: the sentence cannot change Q (text_sens exactly 0). With scene_fusion the fused tokens carry the
    # sentence whatever the switches, so the check runs with the fusion off too (the shards' stored tokens)
    off = dataclasses.replace(cfg0, scene_fusion=False, **{k: False for k in cfg0.path_switches()})
    tr = LangTrainer(off, res_dir / "switches_off", roots, (), caches, use_wandb=False, prefetch=False, seed=0,
                     qwen=qwen)
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
    if cfg0.scene_fusion:
        out["switches_off"]["scene_fusion"] = "off for this check (the fused tokens carry the sentence)"
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
    p.add_argument("--tasks", default="all",
                   help="comma list or 'all' (the five training tasks, TRAIN_TASKS); a resumed run keeps the saved choice")
    p.add_argument("--embodiments", default="ur5-wsg",
                   help="comma list or 'all' (default ur5-wsg: one embodiment); a resumed run keeps the saved choice")
    p.add_argument("--relabel", choices=("paired", "random", "fixed"), default=None)
    p.add_argument("--shuffle-text", action="store_true")
    p.add_argument("--text-backend", choices=("holobrain_bert", "bert_raw", "sentence_generic", "holobrain_qwen"),
                   default=None)
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
    p.add_argument("--ckpt-every-rounds", type=int, default=10, help="keep checkpoint_round<r>.pt every N rounds")
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
    if args.mode == "rounds":   # refuses scene_fusion (run_rounds: no neck ring yet, fusion_impl.md decision 8)
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
