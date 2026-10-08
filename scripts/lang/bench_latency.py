"""Latency of the language-conditioned filter against CrossSafe on one GPU (both fp32, as they train).

    python scripts/lang/bench_latency.py --data data/lang/warmup/shard_g0 \
        --text-cache data/lang/warmup/text_cache.pt --out runs/lang_bench/<gpu>.json

Both models are built the way they train and deploy, and read the same warmup shard (ur5-wsg rows):
CrossSafe sees frame t only (1200 scene tokens), the language model frames t, t-2 and 0 (3600) plus the
sentence.

- Inference (B = 1, no grad, eval mode): one deployment tick. The frozen HoloBrain encoders (vision over
  the 3 cameras, robot state) are shared and identical for both models: the language model keeps frames
  t-2 and 0 from earlier ticks, so it encodes one new frame per tick like CrossSafe. Then the trainable
  encoder, the twin critic Q(s, a_nom) (a free tick) and the deterministic actor (an engaged tick). The
  sentence (BERT + the text adapter) is encoded once per episode. Not timed: the analytic body features
  (Jacobians read from the simulator, the same for both) and rendering.
- Training (B = 64, each trainer's own batch size): each trainer's grad_step on one fixed batch (sampling
  excluded; the language trainer computes its diagnostics every 20th step, CrossSafe every step), and a
  phase breakdown from a replica of the step, CUDA-synchronized between phases.

Median milliseconds over --iters after --warmup, torch.cuda.synchronize() around every call. --tf32 times
the same with TF32 matmuls (the trainers keep PyTorch's defaults: fp32 matmuls, TF32 convolutions).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(ts))


def n_params(*mods) -> int:
    return int(sum(p.numel() for m in mods for p in m.parameters()))


class Phases:
    """Wall time per phase of one replicated step (synchronized), summed over the measured steps."""

    def __init__(self):
        self.acc, self.n, self.t = {}, 0, None

    def start(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def mark(self, name):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + now - self.t
        self.t = now

    def result(self) -> dict:
        return {k: 1000.0 * v / max(self.n, 1) for k, v in self.acc.items()}


# ----------------------------------------------------------------- CrossSafe --
def crosssafe_step_replica(tr, b, ph: Phases):
    """main/train/train.py Trainer.grad_step without its diagnostics, timed by phase."""
    from main.network.heads import Encoded
    from main.train.train import _t

    cfg = tr.cfg
    ph.start()
    enc, Jlin, Jang, ji = tr._encode(b)
    ph.mark("encode s (with grad)")
    enc_n, Jlin_n, Jang_n, ji_n = tr._encode(b, "_next", target=True)
    h, dtheta, dmax, dmax_n = _t(b["h"]), _t(b["dtheta"]), _t(b["dtheta_max"]), _t(b["dtheta_max_next"])
    gamma, alpha = tr.gamma(), tr.alpha()
    with torch.no_grad():
        a_n, _, _, _ = tr.actor(enc_n.body, ji_n, dmax_n)
        q1_t, q2_t, _, _ = tr.critics_targ(enc_n, a_n, Jlin_n, Jang_n, ji_n)
        target = (1 - gamma) * h + gamma * torch.minimum(h, 0.5 * (q1_t + q2_t))
    ph.mark("target pass s' (no grad)")
    q1, q2, _, _ = tr.critics(enc, dtheta, Jlin, Jang, ji)
    loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    ph.mark("critic forward")
    tr.opt_c.zero_grad(set_to_none=True)
    tr.opt_e.zero_grad(set_to_none=True)
    loss_q.backward()
    ph.mark("critic loss backward (critic + encoder)")
    torch.nn.utils.clip_grad_norm_(tr.critics.parameters(), 10.0)
    torch.nn.utils.clip_grad_norm_(list(tr.injection.parameters()) + list(tr.trunk.parameters()), 10.0)
    tr.opt_c.step()
    tr.opt_e.step()
    ph.mark("clip + optimizer (encoder, critic)")
    frozen = list(tr.injection.parameters()) + list(tr.trunk.parameters()) + list(tr.critics.parameters())
    for p in frozen:
        p.requires_grad_(False)
    enc_d = Encoded(enc.body.detach(), enc.pos.detach(), enc.mask, enc.scene.detach(), enc.scene_pos.detach(),
                    enc.scene_mask)
    a_pi, logp, _, pre = tr.actor(enc_d.body, ji, dmax)
    q1p, q2p, _, _ = tr.critics(enc_d, a_pi, Jlin, Jang, ji)
    jv = ji >= 0
    pen = (pre.pow(2) * jv).sum(-1) / jv.sum(-1).clamp_min(1)
    loss_pi = (-torch.minimum(q1p, q2p) + alpha * logp + 1e-3 * pen).mean()
    tr.opt_a.zero_grad(set_to_none=True)
    loss_pi.backward()
    torch.nn.utils.clip_grad_norm_(tr.actor.parameters(), 10.0)
    tr.opt_a.step()
    for p in frozen:
        p.requires_grad_(True)
    ph.mark("actor update (forward + backward + optimizer)")
    with torch.no_grad():
        for tp, sp in zip(tr.target_params, tr.live_params):
            tp.mul_(1 - cfg.tau).add_(cfg.tau * sp)
    ph.mark("Polyak target update")
    ph.n += 1


def bench_crosssafe(shard: Path, scratch: Path, args, res: dict, holo: dict):
    from main.train.config import FrozenConfig
    from main.train.train import Trainer, _t

    cfg = FrozenConfig()
    tr = Trainer(cfg, shard, scratch / "crosssafe_run", buffer_dir=shard / "buffer")
    b = tr.buf.sample(cfg.batch_size, tr.rng)
    tr.buf.sample = lambda B, rng: b          # the fixed batch: sampling is not timed
    out = {"batch": int(cfg.batch_size), "scene_tokens": int(np.asarray(b["scene_tokens"]).shape[1]),
           "params": {"encoder": n_params(tr.injection, tr.trunk),
                      "critic (twin)": n_params(tr.critics), "actor": n_params(tr.actor)}}
    # ---- training ----
    for m in (tr.injection, tr.trunk, tr.critics, tr.actor):
        m.train()
    torch.cuda.reset_peak_memory_stats()
    out["train_step_ms"] = bench(tr.grad_step, args.iters_train, args.warmup)
    out["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    ph = Phases()
    for _ in range(args.warmup):
        crosssafe_step_replica(tr, b, Phases())
    for _ in range(args.iters_train):
        crosssafe_step_replica(tr, b, ph)
    out["train_phases_ms"] = ph.result()
    out["train_phases_ms"]["diagnostics (every step)"] = out["train_step_ms"] - sum(out["train_phases_ms"].values())
    # ---- inference, B = 1 ----
    for m in (tr.injection, tr.trunk, tr.critics, tr.actor):
        m.eval()
    g = {k: _t(np.asarray(b[k])[:1], dt) for k, dt in (
        ("state_tokens", torch.float32), ("body_tokens", torch.float32), ("link_pos", torch.float32),
        ("body_mask", torch.bool), ("joint_index", torch.int64), ("scene_tokens", torch.float32),
        ("scene_pos", torch.float32), ("Jlin", torch.float32), ("Jang", torch.float32),
        ("dtheta", torch.float32), ("dtheta_max", torch.float32))}
    with torch.no_grad():
        enc_fn = lambda: tr.policy_enc(g["state_tokens"], g["body_tokens"], g["link_pos"], g["body_mask"],  # noqa: E731
                                       g["joint_index"], g["scene_tokens"], g["scene_pos"])
        enc = enc_fn()
        inf = {"encoder (trainable)": bench(enc_fn, args.iters, args.warmup),
               "critic Q(s, a_nom)": bench(lambda: tr.critics.qmin(enc, g["dtheta"], g["Jlin"], g["Jang"],
                                                                   g["joint_index"]), args.iters, args.warmup),
               "actor (deterministic)": bench(lambda: tr.actor(enc.body, g["joint_index"], g["dtheta_max"],
                                                               deterministic=True), args.iters, args.warmup)}
    out["infer_ms"] = inf
    # ---- the frozen HoloBrain encoders (shared by both models) ----
    holo.update(bench_holobrain(tr._eval_encoder, args))
    res["crosssafe"] = out
    del tr
    torch.cuda.empty_cache()


def bench_holobrain(enc, args) -> dict:
    from main.train.collect import CKPT_DIR, make_kinematics

    H, W, n_cam = 256, 320, 3
    dev = "cuda"
    imgs = torch.rand(1, n_cam, 3, H, W, device=dev)
    depths = torch.full((1, n_cam, 1, H, W), 1.0, device=dev)
    image_wh = torch.tensor([[W, H]] * n_cam)
    K = torch.tensor([[300.0, 0, W / 2, 0], [0, 300.0, H / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    proj = K.expand(1, n_cam, 4, 4).clone().to(dev)
    eye = torch.eye(4)
    with torch.no_grad():
        vis = lambda: enc.encode_visual_tokens(imgs, depths, image_wh, proj, out_extrinsic=eye.to(dev))  # noqa: E731
        tok, _, _ = vis()
        out = {"HoloBrain vision (3 cameras, one frame)": bench(vis, args.iters, args.warmup),
               "_vision_tokens": int(tok.shape[1] * tok.shape[2])}
        kin = make_kinematics(CKPT_DIR, "ur5-wsg")
        keys = getattr(kin, "arm_link_keys", None) or [kin.left_arm_link_keys, kin.right_arm_link_keys]
        n = 2 * (len(keys[0]) + 1)                 # [arm joints, gripper] x 2
        js = torch.zeros(1, n)
        out["HoloBrain robot state"] = bench(lambda: enc.encode_joint_angles(js, kin, embodiedment_mat=eye),
                                             args.iters, args.warmup)
    return out


# -------------------------------------------------------------- language --
def lang_step_replica(tr, b, ph: Phases):
    """main/lang/train.py LangTrainer.grad_step(log=False), timed by phase."""
    from main.lang.train import _t

    cfg, d = tr.cfg, tr.dev
    ph.start()
    words, wmask = tr.text.lookup(b["sentence"], splits=["train"] * len(b["sentence"]))
    ph.mark("text lookup (cached features)")
    enc, Jlin, Jang, ji = tr.encode(b, words, wmask)
    ph.mark("encode s (with grad)")
    h = _t(b["h"], dev=d)
    dtheta, dmax = _t(b["dtheta"], dev=d), _t(b["dtheta_max"], dev=d)
    gamma, alpha = tr.gamma(), tr.alpha()
    with torch.no_grad():
        enc_n, Jlin_n, Jang_n, ji_n = tr.encode(b, words, wmask, "_next", target=True)
        a_n, _, _, _ = tr.actor(enc_n.body, ji_n, _t(b["dtheta_max_next"], dev=d))
        q1_t, q2_t, _, _ = tr.critics_targ(enc_n, a_n, Jlin_n, Jang_n, ji_n)
        y = (1 - gamma) * h + gamma * torch.minimum(h, 0.5 * (q1_t + q2_t))
    ph.mark("target pass s' (no grad)")
    q1, q2, _, _ = tr.critics(enc, dtheta, Jlin, Jang, ji)
    loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    ph.mark("critic forward")
    tr.opt_e.zero_grad(set_to_none=True)
    tr.opt_c.zero_grad(set_to_none=True)
    loss_q.backward()
    ph.mark("critic loss backward (critic + encoder)")
    torch.nn.utils.clip_grad_norm_(tr.policy_enc.parameters(), 10.0)
    torch.nn.utils.clip_grad_norm_(tr.critics.parameters(), 10.0)
    tr.opt_e.step()
    tr.opt_c.step()
    ph.mark("clip + optimizer (encoder, critic)")
    frozen = list(tr.policy_enc.parameters()) + list(tr.critics.parameters())
    for p in frozen:
        p.requires_grad_(False)
    enc_d = enc.detach()
    a_pi, logp, _, pre = tr.actor(enc_d.body, ji, dmax)
    q1p, q2p, _, _ = tr.critics(enc_d, a_pi, Jlin, Jang, ji)
    jv = ji >= 0
    pen = (pre.pow(2) * jv).sum(-1) / jv.sum(-1).clamp_min(1)
    loss_pi = (-torch.minimum(q1p, q2p) + alpha * logp + 1e-3 * pen).mean()
    tr.opt_a.zero_grad(set_to_none=True)
    loss_pi.backward()
    torch.nn.utils.clip_grad_norm_(tr.actor.parameters(), 10.0)
    tr.opt_a.step()
    for p in frozen:
        p.requires_grad_(True)
    ph.mark("actor update (forward + backward + optimizer)")
    with torch.no_grad():
        for tp, sp in tr.target_pairs:
            tp.mul_(1 - cfg.tau).add_(cfg.tau * sp)
    ph.mark("Polyak target update")
    ph.n += 1


def bench_lang(shard: Path, caches, scratch: Path, args, res: dict):
    from main.item_envs.language.text_encoder import HoloBrainText
    from main.lang.config import LangConfig
    from main.lang.train import LangTrainer, _t

    # the trainer's own TF32 switch (cfg.tf32_train) follows --tf32, so the fp32 timing stays fp32
    cfg = dataclasses.replace(LangConfig(), embodiment_choices=("ur5-wsg",),
                              tf32_train=bool(torch.backends.cuda.matmul.allow_tf32))
    tr = LangTrainer(cfg, scratch / "lang_run", [shard], (), caches, embodiments=("ur5-wsg",), use_wandb=False,
                     prefetch=False, seed=0)
    b = tr.sampler.sample(cfg.batch_size)
    st = np.asarray(b["scene_tokens"])
    pe = tr.policy_enc
    out = {"batch": int(cfg.batch_size), "scene_tokens": int(np.prod(st.shape[1:-1])),
           "params": {"encoder": n_params(pe),
                      "critic (twin)": n_params(tr.critics), "actor": n_params(tr.actor)}}
    # ---- training ----
    for m in (pe, tr.critics, tr.actor):
        m.train()
    torch.cuda.reset_peak_memory_stats()
    t_plain = bench(lambda: tr.grad_step(b, log=False), args.iters_train, args.warmup)
    t_log = bench(lambda: tr.grad_step(b, log=True), max(3, args.iters_train // 4), 1)
    out["train_step_ms"] = (19 * t_plain + t_log) / 20          # diagnostics on every 20th step (LOG_EVERY)
    out["train_step_no_diag_ms"], out["train_step_with_diag_ms"] = t_plain, t_log
    out["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    ph = Phases()
    for _ in range(args.warmup):
        lang_step_replica(tr, b, Phases())
    for _ in range(args.iters_train):
        lang_step_replica(tr, b, ph)
    out["train_phases_ms"] = ph.result()
    out["train_phases_ms"]["diagnostics (every 20th step, averaged)"] = (t_log - t_plain) / 20
    # ---- inference, B = 1 ----
    for m in (pe, tr.critics, tr.actor):
        m.eval()
    g = {k: _t(np.asarray(b[k])[:1], dt, tr.dev) for k, dt in (
        ("state_tokens", torch.float32), ("body_tokens", torch.float32), ("link_pos", torch.float32),
        ("body_mask", torch.bool), ("joint_index", torch.int64), ("scene_tokens", torch.float32),
        ("scene_pos", torch.float32), ("Jlin", torch.float32), ("Jang", torch.float32),
        ("dtheta", torch.float32), ("dtheta_max", torch.float32))}
    sent = b["sentence"][0]
    bert = HoloBrainText(device=tr.dev)
    with torch.no_grad():
        w, m = tr.text.lookup([sent])
        text_out = pe.encode_text(w, m)
        enc_fn = lambda: pe(g["state_tokens"], g["body_tokens"], g["link_pos"], g["body_mask"],  # noqa: E731
                            g["joint_index"], g["scene_tokens"], g["scene_pos"], text_out=text_out)
        enc = enc_fn()
        inf = {"encoder (trainable)": bench(enc_fn, args.iters, args.warmup),
               "critic Q(s, a_nom)": bench(lambda: tr.critics.qmin(enc, g["dtheta"], g["Jlin"], g["Jang"],
                                                                   g["joint_index"]), args.iters, args.warmup),
               "actor (deterministic)": bench(lambda: tr.actor(enc.body, g["joint_index"], g["dtheta_max"],
                                                               deterministic=True), args.iters, args.warmup)}
        per_ep = {"BERT (HoloBrain text encoder)": bench(lambda: bert.embed([sent]), args.iters, args.warmup),
                  "text adapter": bench(lambda: pe.encode_text(w, m), args.iters, args.warmup)}
    out["infer_ms"], out["per_episode_ms"], out["sentence"] = inf, per_ep, sent
    res["lang"] = out
    del tr
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ report --
def tables(res: dict) -> str:
    cs, ln, holo = res["crosssafe"], res["lang"], res["holobrain"]
    f = lambda v: f"{v:.2f}"  # noqa: E731
    rat = lambda a, b: f"{b / a:.2f}x" if a > 0 else "-"  # noqa: E731
    L = [f"# Latency: language filter vs CrossSafe ({res['gpu']}, {res.get('precision', 'fp32')})", "",
         f"Scene tokens per sample: CrossSafe {cs['scene_tokens']}, language {ln['scene_tokens']} "
         f"(frames t, t-2, 0). Median ms over {res['iters']} calls (training {res['iters_train']} steps).", "",
         "## Inference, one deployment tick (B = 1)", "",
         "| part | CrossSafe | language | language / CrossSafe |", "|---|---|---|---|"]
    hv, hs = holo["HoloBrain vision (3 cameras, one frame)"], holo["HoloBrain robot state"]
    L += [f"| HoloBrain vision, frozen (3 cameras, one frame; shared) | {f(hv)} | {f(hv)} | 1.00x |",
          f"| HoloBrain robot state, frozen (shared) | {f(hs)} | {f(hs)} | 1.00x |"]
    for k in ("encoder (trainable)", "critic Q(s, a_nom)", "actor (deterministic)"):
        L.append(f"| {k} | {f(cs['infer_ms'][k])} | {f(ln['infer_ms'][k])} | {rat(cs['infer_ms'][k], ln['infer_ms'][k])} |")
    for name, last in (("free tick: encoders + critic", "critic Q(s, a_nom)"),
                       ("engaged tick: encoders + actor", "actor (deterministic)")):
        a = hv + hs + cs["infer_ms"]["encoder (trainable)"] + cs["infer_ms"][last]
        b2 = hv + hs + ln["infer_ms"]["encoder (trainable)"] + ln["infer_ms"][last]
        L.append(f"| **{name}** | **{f(a)}** | **{f(b2)}** | {rat(a, b2)} |")
    pe = ln["per_episode_ms"]
    L += ["", f"Once per episode (language only): BERT {f(pe['BERT (HoloBrain text encoder)'])} ms, text adapter "
          f"{f(pe['text adapter'])} ms. Not timed: rendering and the analytic body features (simulator, same for both).",
          "", f"## Training, one gradient step (B = {cs['batch']} CrossSafe, {ln['batch']} language)", "",
          "| phase | CrossSafe | language | language / CrossSafe |", "|---|---|---|---|"]
    keys = list(dict.fromkeys([k for k in ln["train_phases_ms"]] + [k for k in cs["train_phases_ms"]]))
    for k in keys:
        a, b2 = cs["train_phases_ms"].get(k), ln["train_phases_ms"].get(k)
        L.append(f"| {k} | {f(a) if a is not None else '-'} | {f(b2) if b2 is not None else '-'} | "
                 f"{rat(a, b2) if a is not None and b2 is not None else '-'} |")
    L += [f"| **grad step (each trainer's own code)** | **{f(cs['train_step_ms'])}** | **{f(ln['train_step_ms'])}** | "
          f"{rat(cs['train_step_ms'], ln['train_step_ms'])} |",
          f"| steps per second | {1000 / cs['train_step_ms']:.2f} | {1000 / ln['train_step_ms']:.2f} | |",
          f"| peak GPU memory (GB) | {cs['train_peak_mem_gb']:.1f} | {ln['train_peak_mem_gb']:.1f} | |", "",
          "## Parameters", "", "| module | CrossSafe | language |", "|---|---|---|"]
    for k in cs["params"]:
        fmt = lambda n: f"{n / 1e6:.2f} M" if n >= 1e5 else f"{n:,}"  # noqa: E731
        L.append(f"| {k} | {fmt(cs['params'][k])} | {fmt(ln['params'][k])} |")
    L += ["", "Encoder: CrossSafe = robot injection + geometric trunk; language = the same injection + the scene "
          "adapter over 3 frames + the text adapter + the 3-block trunk with the word paths."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="one warmup shard (StepBuffer + item sidecar)")
    ap.add_argument("--text-cache", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="<name>.json; the tables go next to it as <name>.md")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--iters-train", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls for everything timed")
    args = ap.parse_args()
    import main.lang  # noqa: F401  (main.item_envs on the path)

    # default: PyTorch's settings, which both trainers keep (fp32 matmuls, TF32 convolutions); --tf32: TF32 matmuls
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / f"_{out.stem}_scratch"
    shard = Path(args.data).resolve()
    res = {"gpu": torch.cuda.get_device_name(0), "precision": "TF32 matmuls" if args.tf32 else "fp32 matmuls",
           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
           "iters": args.iters, "iters_train": args.iters_train,
           "data": str(shard), "holobrain": {}}
    bench_crosssafe(shard, scratch, args, res, res["holobrain"])
    bench_lang(shard, [Path(c) for c in args.text_cache], scratch, args, res)
    out.write_text(json.dumps(res, indent=1))
    md = tables(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
