"""Rounds mode (VANILLA_PLAN 6.2, 7.3, 7.4): CrossSafe's train -> collect -> train.

Round 0 trains on the warmup shards; every later round first collects
``episodes_per_round`` episodes into the run's own ring
(``runs/lang/<name>/ring``) with the current weights (each a filter episode
with probability ``filter_episode_frac``, the named rule chosen by
designation, Section 7.1), then trains ``grad_steps_per_round`` steps on the
warmup shards plus the ring.

Resume (Section 7.4, extended 2026-10-07 to stop and resume INSIDE a round,
so a run fits jobs shorter than a round): ``state.pt`` (weights, targets,
optimizers, step, round index, schedule cursors, RNG states, the ring's n /
head) also records where in the round the run is:

- ``phase = "collect"``: the round's slots, how many are done and the next
  attempt counter. Written at the round start and after EVERY collected
  episode, so a hard kill loses at most the episode in progress;
- ``phase = "train"``: the round's step target. Written when collection ends,
  every ``STATE_EVERY`` steps and when the run stops.

``--resume`` loads it, rolls both rings back to its n (``discard_to``, valid
because the ring never wraps), drops the lines of episodes past that point from
``lang_episodes.jsonl`` and ``eval/rounds.jsonl``, and continues in the
recorded phase (the remaining slots keep their seeds). The run stops cleanly
(state saved, exit 75) at the time budget (``--time-budget`` or
``$STAGE_DEADLINE``), on SIGTERM / SIGUSR1, and at the init limit (when this
process's inits plus 5 x episodes_per_round would pass
``max_inits_per_process``). scripts/lang/run_rounds.sh relaunches on 75.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import ensure_item_path

ensure_item_path()

from .collect import InitLimit, item_n_tok, open_extras, read_lines, run_episodes, stage_deadline  # noqa: E402
from .config import CANON_EMBODIMENTS, CANON_TASKS, round_offset  # noqa: E402
from .train import EXIT_RESUME, LOG_EVERY, STATE_EVERY, LangTrainer, check_resume_identity, code_identity  # noqa: E402




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
    """The run's ring: StepBuffer + item sidecar (token masks on the config's encoder grid). V6's kv_buffer/ is
    opened next to it by collect.open_extras (_run_rounds)."""
    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.train.buffer import StepBuffer

    ring = run / "ring"
    cap = int(rounds * episodes_per_round * cfg.max_steps_per_episode * 1.1) + 16
    if (ring / "buffer" / "header.json").exists():
        # opened directly: for_step_buffer refuses a pair whose n differ, which a kill between the two
        # flushes leaves, before the resume can roll both back (review O2)
        buf = StepBuffer(ring / "buffer")
        if not (ring / "item_buffer" / "header.json").exists() and len(buf) == 0:
            # the first launch died between the two creations (round 2 B2): pair a fresh sidecar
            side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=item_n_tok(cfg))
        else:
            side = ItemSidecarBuffer(ring / "item_buffer")
        if side.capacity != buf.capacity:
            raise SystemExit(f"{ring}: buffer capacity {buf.capacity} != sidecar capacity {side.capacity}")
        if buf.capacity < cap:   # the ring never wraps: it must hold every round asked for (review O3)
            raise SystemExit(f"{ring}: capacity {buf.capacity} rows < {cap} needed for {rounds} rounds x "
                             f"{episodes_per_round} episodes; resume with the values of the first launch "
                             f"(lang_config.json) or start a new run")
        return buf, side
    buf = StepBuffer(ring / "buffer", capacity=cap, header=cfg.to_dict())
    side = ItemSidecarBuffer.for_step_buffer(buf, n_tok=item_n_tok(cfg))
    return buf, side


def run_rounds(cfg, args, roots, vals, caches, tasks, embs, *, trainer=None, collect_fn=None) -> int:
    """trainer / collect_fn: injected by the CPU tests (default: a LangTrainer and run_episodes).

    One process per run (review O1): the ring's writer lock is held for the whole call; a second process
    refuses (after waiting out a crashed one's lock)."""
    from .locks import WriterLock

    if getattr(cfg, "scene_fusion", False):
        # fusion_impl.md decision 8: the ring is a version-1 pair (no neck_buffer/), so the rounds' own episodes
        # could not be fused at training time, and the rollout would need the stage for its filter episodes
        raise ValueError("rounds mode does not support scene_fusion yet (the run's ring has no neck_buffer/): "
                         "train V5 offline on version-2 shards (E5)")
    run = Path(args.run).resolve()
    try:
        writer = WriterLock(run / "ring", wait_s=200.0)
    except WriterLock.LockHeld as exc:
        raise SystemExit(f"[rounds] {exc}: is another job running this run?")
    try:
        return _run_rounds(cfg, args, roots, vals, caches, tasks, embs, trainer=trainer, collect_fn=collect_fn,
                           writer=writer)
    except WriterLock.LockLost as exc:   # another process owns the run now: write nothing more
        print(f"[rounds] {exc} (exit 75)", flush=True)
        return EXIT_RESUME
    finally:
        writer.release()


def _run_rounds(cfg, args, roots, vals, caches, tasks, embs, *, trainer=None, collect_fn=None, writer=None) -> int:
    from .metrics import LangRoundMetrics

    run = Path(args.run).resolve()
    if not args.resume and (run / "state.pt").exists():
        raise SystemExit(f"{run} has a state.pt: pass --resume or use a new run directory")
    cfg = type(cfg).from_dict({**cfg.to_dict(), "collect_rounds": args.collect_rounds,
                               "episodes_per_round": args.episodes_per_round,
                               "grad_steps_per_round": args.grad_steps_per_round})
    n_rounds, n_eps, n_steps = int(args.collect_rounds), int(args.episodes_per_round), int(args.grad_steps_per_round)
    ring_buf, ring_side = make_ring(run, cfg, n_rounds, n_eps)
    # V6 (qwen_impl.md 8): one HoloBrain-Qwen front end per process, the trainer's text pass and the collection's
    # frame encoder; the ring gains kv_buffer/ (no frames/), created, resumed and rolled back with the pair
    qwen = getattr(trainer, "qwen", None) if trainer is not None else None
    if cfg.qwen and qwen is None and trainer is None:
        from .qwen_encoder import HoloBrainQwenEncoder

        qwen = HoloBrainQwenEncoder(device="cuda", attn=cfg.qwen_attn, vision=True)
    if cfg.qwen and qwen is None:
        raise ValueError("encoder holobrain_qwen: the injected trainer has no qwen front end")
    extras = (open_extras(run / "ring", ring_buf, kv_dims={"ckpt_id": qwen.ckpt_id, "attn": cfg.qwen_attn,
                                                           "encoder_version": getattr(qwen, "ENCODER_VERSION", None)})
              if cfg.qwen else {})
    tr = trainer if trainer is not None else LangTrainer(
        cfg, run, roots, vals, caches, tasks=tasks, embodiments=embs, use_wandb=not args.no_wandb, seed=args.seed,
        qwen=qwen)
    ring_rings = (ring_buf, ring_side, *extras.values())
    collect_fn = collect_fn or run_episodes
    sched = RoundSchedule(tasks or cfg.task_choices, embs or cfg.embodiment_choices, cfg.t1_hosts)
    rng_policy = np.random.default_rng(args.seed + 17)
    metrics = LangRoundMetrics(run)
    lj = run / "ring" / "lang_episodes.jsonl"
    # where in a round the run is (module docstring); None = the start of round `start_round`
    start_round, pos = 0, None
    # h definition, data roots and git commit, saved with every state; a resume refuses a change (rounds 2-3)
    tr.identity = code_identity(tasks or cfg.task_choices, roots)
    if writer is not None:
        tr.writer = writer
    if args.resume and (run / "state.pt").exists():
        st = tr.load_state()
        check_resume_identity(st, tr.identity, run, bool(getattr(args, "accept_code_change", False)))
        start_round = int(st["round"])
        sched.load(st["schedule"])
        rng_policy.bit_generator.state = st["rng_policy"]
        if st.get("phase") in ("collect", "train"):
            pos = {"phase": st["phase"], "slots": [tuple(x) for x in st.get("slots") or []],
                   "done": int(st.get("slots_done", 0)), "attempt": int(st.get("attempt_next", 0)),
                   "steps_target": st.get("steps_target")}
        n = int(st["ring_n"])
        if any(len(b) != n for b in ring_rings):
            # the episode in progress is collected again under the next seed (attempt + 1): rows a stalled
            # former writer might still add then carry another episode id (review round 3, C1)
            print(f"[rounds] resume: ring {[len(b) for b in ring_rings]} -> {n} rows (the episode in progress is "
                  "collected again, next seed)")
            for b in ring_rings:
                b.discard_to(n)
            for b in ring_rings:
                b.flush()
            if pos is not None and pos["phase"] == "collect":
                pos["attempt"] += 1
        if any(len(b) != n for b in ring_rings):
            raise SystemExit(f"{run}: ring at {[len(b) for b in ring_rings]} rows after the roll-back to {n}")

        def kept(d):   # episodes the restored state already contains
            rd = int(d.get("round", -1))
            if rd != start_round:
                return rd < start_round
            return pos is not None and (pos["phase"] == "train" or int(d.get("slot", 1 << 30)) < pos["done"])

        keep = [d for d in read_lines(lj) if kept(d)]
        tmp = lj.with_suffix(".jsonl.tmp")   # atomic: a kill mid-rewrite must not cut the record (review)
        tmp.write_text("".join(json.dumps(d) + "\n" for d in keep))
        tmp.replace(lj)
        ids = {int(d["episode_id"]) for d in keep if "episode_id" in d}   # skipped slots have no episode
        metrics.keep(lambda row: int(row["round"]) < start_round or int(row["episode_id"]) in ids)
        print(f"[rounds] resumed: round {start_round} ({pos['phase'] if pos else 'start'}), step {tr.step}, "
              f"ring {n} rows", flush=True)
        tr.round = start_round
        post = resume_fingerprint(tr, sched, rng_policy, ring_buf)
        (run / "resume_check_post.json").write_text(json.dumps(post))
        pre_p = run / "resume_check_pre.json"
        if pre_p.exists():
            pre = json.loads(pre_p.read_text())
            diff = {k: (pre.get(k), post.get(k)) for k in post if pre.get(k) != post.get(k)}
            print(f"[rounds] resume check: {'exact' if not diff else diff}", flush=True)
            pre_p.unlink()
    ring_added = False
    if len(ring_buf) > 0:   # after the roll-back: the sampler never indexes discarded rows
        tr.sampler.add_source(run / "ring")
        ring_added = True
    deadline = stage_deadline(getattr(args, "time_budget", None))
    tr.install_signal()
    encoder = None
    kins = {}
    inits = {"n": 0}

    def check():   # before every write (round 2 B1)
        if writer is not None:
            writer.check()

    def save(r_pos):
        check()
        save_round_state(tr, sched, rng_policy, ring_buf, r_pos)

    def stop(r_pos, why):
        tr.pause_prefetch()
        save(r_pos)
        check()
        tr.save_ckpt()
        (run / "resume_check_pre.json").write_text(json.dumps(resume_fingerprint(tr, sched, rng_policy, ring_buf)))
        print(f"[rounds] {why}: state saved in round {tr.round} ({r_pos['phase']}), step {tr.step}; exit 75",
              flush=True)
        return EXIT_RESUME

    def out_of_time(need_s: float = 0.0) -> bool:
        return bool(tr._stop) or (deadline is not None and time.time() + need_s > deadline)

    for r in range(start_round, n_rounds + 1):
        tr.round = r
        if r == start_round and pos is not None:
            r_pos = pos                    # resumed inside this round
        else:
            r_pos = ({"phase": "collect", "slots": [sched.next() for _ in range(n_eps)], "done": 0, "attempt": 0,
                      "steps_target": None} if r > 0 else
                     {"phase": "train", "slots": [], "done": 0, "attempt": 0, "steps_target": None})
            if r > start_round and inits["n"] + 5 * n_eps > int(cfg.max_inits_per_process):
                return stop(r_pos, f"init limit ({inits['n']} inits)")
            tr.pause_prefetch()
            save(r_pos)
        if r_pos["phase"] == "collect":
            tr.pause_prefetch()            # drain before collecting
            if encoder is None and collect_fn is run_episodes:
                from main.network.encoder import HoloBrainEncoder
                from main.train.collect import CKPT_DIR

                encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
            for m in (tr.policy_enc, tr.actor, tr.critics):
                m.eval()
            t0, walls = time.time(), []
            slots = r_pos["slots"]
            while r_pos["done"] < len(slots):
                est = float(np.mean(walls)) if walls else 420.0
                if out_of_time(1.2 * est):
                    return stop(r_pos, "signal" if tr._stop else f"time budget (next episode ~{est:.0f}s)")
                k = r_pos["done"]
                t_ep = time.time()
                try:
                    summ = collect_fn(cfg, run / "ring", [slots[k]], round_offset(r), ring_buf, ring_side,
                                      encoder=encoder, kins=kins, models=(tr.policy_enc, tr.actor, tr.critics),
                                      filter_frac=float(cfg.filter_episode_frac), rng_policy=rng_policy,
                                      attempt0=r_pos["attempt"], inits=inits,
                                      video_every=1 if (args.video_every and r % args.video_every == 0) else 0,
                                      video_filter_only=True, text_store=tr.text, log_prefix=f"[round {r}] ", slot_offset=k,
                                      extra_line={"round": r}, round_metrics=(metrics, r), writer=writer,
                                      **({"qwen": qwen, "kv_buf": extras["_kv"]} if cfg.qwen else {}))
                except InitLimit:
                    return stop(r_pos, f"init limit inside round ({inits['n']} inits)")
                walls.append(time.time() - t_ep)
                r_pos["attempt"] = int(summ["attempt"])
                r_pos["done"] = k + 1
                save(r_pos)                # a kill now loses at most the next episode
            for m in (tr.policy_enc, tr.actor, tr.critics):
                m.train()
            if not ring_added:
                tr.sampler.add_source(run / "ring")
                ring_added = True
            else:
                tr.sampler.refresh()
            tr.sampler.refresh_pending()
            tr.log({"round/collect_s": time.time() - t0, "round/episodes": len(slots), "round/index": r,
                    **metrics.round_summary(r)})
            log_round_media(tr, r, lj, metrics.heatmaps(r))
            tr.reset_perf()
            r_pos = {**r_pos, "phase": "train", "steps_target": None}
        if r_pos["steps_target"] is None:
            r_pos["steps_target"] = tr.step + n_steps
            save(r_pos)
        t0 = time.time()
        for m in (tr.policy_enc, tr.actor, tr.critics):
            m.train()
        while tr.step < int(r_pos["steps_target"]):
            if out_of_time():
                return stop(r_pos, "signal" if tr._stop else "time budget")
            do_log = (tr.step + 1) % LOG_EVERY == 0
            out = tr.grad_step(log=do_log)
            if do_log:
                tr.log(out)
                tr.console(out)
            if tr.step % STATE_EVERY == 0 and tr.step < int(r_pos["steps_target"]):
                tr.pause_prefetch()
                save(r_pos)
        tr.log({"round/train_s": time.time() - t0})
        check()
        tr.save_ckpt()
        if r % 10 == 0 and r > 0:
            tr.save_ckpt(f"checkpoint_round{r}.pt")
        if args.probe_every_rounds and r % int(args.probe_every_rounds) == 0:
            pr = tr.run_probe()
            if pr:
                tr.log(pr)
            tr.reset_perf()
    tr.round = n_rounds + 1
    tr.pause_prefetch()
    save({"phase": None, "slots": [], "done": 0, "attempt": 0, "steps_target": None})
    check()
    tr.save_ckpt("checkpoint_final.pt")
    if tr.prefetch is not None:
        tr.prefetch.stop()
    return 0


def log_round_media(tr: LangTrainer, r: int, lj: Path, images) -> None:
    """wandb (when on): this round's filter-episode videos as one carousel, the heatmaps and the trend plot.
    The files stay on disk either way (ring/videos/, eval/)."""
    if tr.wandb is None:
        return
    try:
        import wandb

        from .train import _safe_wandb_log

        # filter episodes only (Yuxuan, 2026-10-11): nominal videos written before the change stay on disk, unlogged
        vids = [d["video"] for d in read_lines(lj) if d.get("round") == r and d.get("filter") and d.get("video")
                and Path(d["video"]).exists()]
        logs = {f"heatmap/{p.parent.name}" if p.name != "trend.png" else "trend/overview": wandb.Image(str(p))
                for p in images if p and Path(p).exists()}
        if vids:
            logs["round_videos"] = [wandb.Video(v, format="mp4", caption=Path(v).stem) for v in vids]
        if logs:
            _safe_wandb_log(tr.wandb, logs, tr.step)
    except Exception as exc:   # media logging must never stop a run
        print(f"[rounds] wandb media log failed ({exc}); continuing", flush=True)


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


def save_round_state(tr: LangTrainer, sched: RoundSchedule, rng_policy, ring_buf, r_pos: dict | None = None) -> None:
    r_pos = r_pos or {}
    tr.save_state({"schedule": sched.state(), "rng_policy": rng_policy.bit_generator.state,
                   "ring_n": int(len(ring_buf)), "ring_head": int(ring_buf.head),
                   "phase": r_pos.get("phase"), "slots": [list(x) for x in r_pos.get("slots") or []],
                   "slots_done": int(r_pos.get("done", 0)), "attempt_next": int(r_pos.get("attempt", 0)),
                   "steps_target": r_pos.get("steps_target")})
