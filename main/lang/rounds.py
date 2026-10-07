"""Rounds mode (VANILLA_PLAN 6.2, 7.3, 7.4): CrossSafe's train -> collect -> train.

Round 0 trains on the warmup shards; every later round first collects
``episodes_per_round`` episodes into the run's own ring
(``runs/lang/<name>/ring``) with the current weights (each a filter episode
with probability ``filter_episode_frac``, the named rule chosen by
designation, Section 7.1), then trains ``grad_steps_per_round`` steps on the
warmup shards plus the ring.

Resume (Section 7.4): ``state.pt`` is written at the START of every round
(weights, targets, optimizers, step, round index, schedule cursors, RNG
states, the ring's n / head). ``--resume`` loads it, rolls both rings back to
that n (``discard_to``, valid because the ring never wraps), drops that
round's lines from ``lang_episodes.jsonl`` and collects the round again with
the same ids. Init limit: between rounds, when the inits of this process plus
5 x episodes_per_round would pass ``max_inits_per_process``, the trainer
writes the checkpoint and the next round's start state and exits 75
(scripts/lang/run_rounds.sh relaunches with --resume).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import ensure_item_path

ensure_item_path()

from .collect import read_lines, run_episodes  # noqa: E402
from .config import CANON_EMBODIMENTS, CANON_TASKS, round_offset  # noqa: E402
from .train import EXIT_RESUME, LangTrainer  # noqa: E402


class RoundSchedule:
    """Global (task, embodiment) cursor + a T1 host cursor per embodiment."""

    def __init__(self, tasks, embodiments, t1_hosts):
        self.combos = [(t, e) for t in tasks for e in embodiments]
        self.t1_hosts = tuple(t1_hosts)
        self.cursor = 0
        self.host_cursor = {e: 0 for e in embodiments}

    def next(self):
        t, e = self.combos[self.cursor % len(self.combos)]
        self.cursor += 1
        host = None
        if t == "selective_avoid":
            host = self.t1_hosts[self.host_cursor[e] % len(self.t1_hosts)]
            self.host_cursor[e] += 1
        return (t, e, host)

    def state(self):
        return {"cursor": self.cursor, "host_cursor": dict(self.host_cursor)}

    def load(self, st):
        self.cursor = int(st["cursor"])
        self.host_cursor = {k: int(v) for k, v in st["host_cursor"].items()}


def make_ring(run: Path, cfg, rounds: int, episodes_per_round: int):
    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.item_envs.masks import token_grid
    from main.train.buffer import StepBuffer

    ring = run / "ring"
    if (ring / "buffer" / "header.json").exists():
        buf = StepBuffer(ring / "buffer")
        side = ItemSidecarBuffer.for_step_buffer(buf)
        return buf, side
    cap = int(rounds * episodes_per_round * cfg.max_steps_per_episode * 1.1) + 16
    buf = StepBuffer(ring / "buffer", capacity=cap, header=cfg.to_dict())
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=token_grid()[1])
    return buf, side


def run_rounds(cfg, args, roots, vals, caches, tasks, embs) -> int:
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR

    from .metrics import LangRoundMetrics

    run = Path(args.run).resolve()
    if not args.resume and (run / "state.pt").exists():
        raise SystemExit(f"{run} has a state.pt: pass --resume or use a new run directory")
    cfg = type(cfg).from_dict({**cfg.to_dict(), "collect_rounds": args.collect_rounds,
                               "episodes_per_round": args.episodes_per_round,
                               "grad_steps_per_round": args.grad_steps_per_round})
    ring_buf, ring_side = make_ring(run, cfg, args.collect_rounds, args.episodes_per_round)
    tr = LangTrainer(cfg, run, roots, vals, caches, tasks=tasks, embodiments=embs, use_wandb=not args.no_wandb,
                     seed=args.seed)
    ring_added = False
    if len(ring_buf) > 0:
        tr.sampler.add_source(run / "ring")
        ring_added = True
    sched = RoundSchedule(tasks or cfg.task_choices, embs or cfg.embodiment_choices, cfg.t1_hosts)
    rng_policy = np.random.default_rng(args.seed + 17)
    start_round = 0
    if args.resume and (run / "state.pt").exists():
        st = tr.load_state()
        start_round = int(st["round"])
        sched.load(st["schedule"])
        rng_policy.bit_generator.state = st["rng_policy"]
        n = int(st["ring_n"])
        if (len(ring_buf), len(ring_side)) != (n, n):
            print(f"[rounds] resume: ring {len(ring_buf)} -> {n} rows (round {start_round} is collected again)")
            ring_buf.discard_to(n)
            ring_side.discard_to(n)
            ring_buf.flush(), ring_side.flush()
        lj = run / "ring" / "lang_episodes.jsonl"
        keep = [d for d in read_lines(lj) if int(d.get("round", -1)) < start_round]
        lj.write_text("".join(json.dumps(d) + "\n" for d in keep))
        print(f"[rounds] resumed: round {start_round}, step {tr.step}, ring {n} rows", flush=True)
        tr.round = start_round
        post = resume_fingerprint(tr, sched, rng_policy, ring_buf)
        (run / "resume_check_post.json").write_text(json.dumps(post))
        pre_p = run / "resume_check_pre.json"
        if pre_p.exists():
            pre = json.loads(pre_p.read_text())
            diff = {k: (pre.get(k), post.get(k)) for k in post if pre.get(k) != post.get(k)}
            print(f"[rounds] resume check: {'exact' if not diff else diff}", flush=True)
    encoder = None
    kins = {}
    inits = {"n": 0}
    metrics = LangRoundMetrics(run)
    for r in range(start_round, int(args.collect_rounds) + 1):
        tr.round = r
        if r > start_round and inits["n"] + 5 * int(args.episodes_per_round) > int(cfg.max_inits_per_process):
            tr.pause_prefetch()
            tr.save_ckpt()
            save_round_state(tr, sched, rng_policy, ring_buf)
            (run / "resume_check_pre.json").write_text(json.dumps(resume_fingerprint(tr, sched, rng_policy, ring_buf)))
            print(f"[rounds] init limit ({inits['n']} inits): exit 75 before round {r}", flush=True)
            return EXIT_RESUME
        tr.pause_prefetch()
        save_round_state(tr, sched, rng_policy, ring_buf)
        if r > 0:
            tr.pause_prefetch()          # drain before collecting
            if encoder is None:
                encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
            slots = [sched.next() for _ in range(int(args.episodes_per_round))]
            for m in (tr.policy_enc, tr.actor, tr.critics):
                m.eval()
            t0 = time.time()
            summ = run_episodes(cfg, run / "ring", slots, round_offset(r), ring_buf, ring_side, encoder=encoder,
                                kins=kins, models=(tr.policy_enc, tr.actor, tr.critics),
                                filter_frac=float(cfg.filter_episode_frac), rng_policy=rng_policy, inits=inits,
                                video_every=1 if (args.video_every and r % args.video_every == 0) else 0,
                                text_store=tr.text, log_prefix=f"[round {r}] ", extra_line={"round": r},
                                round_metrics=(metrics, r))
            for m in (tr.policy_enc, tr.actor, tr.critics):
                m.train()
            if not ring_added:
                tr.sampler.add_source(run / "ring")
                ring_added = True
            else:
                tr.sampler.refresh()
            logs = {"round/collect_s": time.time() - t0, "round/episodes": summ["episodes"], "round/index": r,
                    **metrics.round_summary(r)}
            tr.log(logs)
            metrics.heatmaps(r)
        t0 = time.time()
        steps_target = tr.step + int(args.grad_steps_per_round)
        while tr.step < steps_target:
            do_log = (tr.step + 1) % 20 == 0
            out = tr.grad_step(log=do_log)
            if do_log:
                tr.log(out)
                tr.console(out)
        tr.log({"round/train_s": time.time() - t0})
        tr.save_ckpt()
        if r % 10 == 0 and r > 0:
            tr.save_ckpt(f"checkpoint_round{r}.pt")
        if args.probe_every_rounds and r % int(args.probe_every_rounds) == 0:
            pr = tr.run_probe()
            if pr:
                tr.log(pr)
    tr.round = int(args.collect_rounds) + 1
    save_round_state(tr, sched, rng_policy, ring_buf)
    tr.save_ckpt("checkpoint_final.pt")
    if tr.prefetch is not None:
        tr.prefetch.stop()
    return 0


def resume_fingerprint(tr: LangTrainer, sched: RoundSchedule, rng_policy, ring_buf) -> dict:
    """What --resume must restore exactly (Phase 4 test): step, alpha, round, schedule cursors, RNG
    states (hashed), ring n."""
    import hashlib

    def hsh(x):
        return hashlib.md5(repr(x).encode()).hexdigest()[:12]

    return {"step": tr.step, "alpha": tr.alpha(), "round": tr.round, "schedule": sched.state(),
            "rng_policy": hsh(rng_policy.bit_generator.state), "sampler_rng": hsh(tr.sampler.rng.bit_generator.state),
            "prefetch_rng": hsh(tr._last_rng_after if tr._last_rng_after is not None
                                else (tr.prefetch.rng_state() if tr.prefetch is not None else None)),
            "torch_rng": hsh(torch.get_rng_state().tolist()), "ring_n": int(len(ring_buf))}


def save_round_state(tr: LangTrainer, sched: RoundSchedule, rng_policy, ring_buf) -> None:
    tr.save_state({"schedule": sched.state(), "rng_policy": rng_policy.bit_generator.state,
                   "ring_n": int(len(ring_buf)), "ring_head": int(ring_buf.head)})
