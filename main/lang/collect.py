"""One episode loop for warmup, validation and rounds (VANILLA_PLAN 7.1, 7.2).

``run_episodes`` mirrors ``main.train.collect.collect()`` (the verified
reference for item tasks) and differs where Section 3.3 item 8 says it must:
every episode gets a ``lang_episodes.jsonl`` line (status, success, scene
roles, per-rule outcome), retries draw a NEW seed (attempt + 1), the schedule
is given per shard, ``LockstepItemWriter.end_episode()`` is checked, and the
controller is a ``LangRolloutController``.

CLI (always through scripts/lang/in_container.sh, cwd = repo root):

    python -m main.lang.collect warmup --out data/lang/warmup/shard_g<g> \\
        --embodiment ur5-wsg --shard <i> --n-shards <n> --episodes-per-task 30
    python -m main.lang.collect validation --out data/lang/val/shard_g<g> ...
    python -m main.lang.collect recon --out <scratch> --embodiment ur5-wsg --per-task 2
    python -m main.lang.collect report data/lang/warmup

Version 2 (V5, fusion_impl.md 4.6): ``--lang-version 2`` adds the neck_buffer/
sidecar (created, resumed, rolled back and flushed with the pair). ``--replay
<shard>`` collects that shard's recorded episodes again, each with its own seed:

    python -m main.lang.collect validation --lang-version 2 --embodiment ur5-wsg \\
        --replay data/lang/val_grounding/shard_g0 --out data/lang_v2/val_grounding/shard_g0
    python -m main.lang.collect warmup --lang-version 2 --embodiment ur5-wsg \\
        --tasks selective_avoid,gap_place --replay data/lang/warmup/shard_g0 --out data/lang_v2/warmup/shard_g0

T8 tray_slide (2026-11-13; ur5-wsg only, schedules skip it elsewhere): each episode's nominal behaviour is drawn from
its seed (config.episode_risk, LangConfig.behaviour_mix). Its own episode ids through --g-base (warmup g 0-30 and
validation g 0-25 are taken on disk): the validation set is 20 ur5-wsg episodes at g 26 (ids 86M + attempt), the
warmup at g 40 (ids 41M + attempt), e.g.

    python -m main.lang.collect validation --embodiment ur5-wsg --tasks tray_slide --shard 0 --n-shards 1 \\
        --episodes-per-task 20 --g-base 26 --out data/lang/val_tray/shard_g26
    python -m main.lang.collect warmup --embodiment ur5-wsg --tasks tray_slide --shard <i> --n-shards <n> \\
        --episodes-per-task 30 --g-base 40 --out data/lang/warmup_tray/shard_g<40 + i>

V6 (qwen_impl.md 6): ``--encoder holobrain_qwen`` collects with HoloBrain-Qwen's front end (297 scene tokens per
frame, the kv_buffer/ sidecar with each frame's prefix keys / values, and by default the frames/ sidecar with the
native renders); ``--spawn-pool pre_1012`` draws T1 objects from the pool before 2026-10-12, so a replay of a
pre-10-12 shard places its recorded objects again (checked per episode: the layout check):

    python -m main.lang.collect warmup --embodiment ur5-wsg --encoder holobrain_qwen --spawn-pool pre_1012 \\
        --tasks selective_avoid,push_zone,tray_transport,gap_place --replay data/lang/warmup/shard_g0 \\
        --out data/lang_qwen/warmup/shard_g0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import ensure_item_path

ensure_item_path()

from .config import (  # noqa: E402
    CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS, TRAIN_TASKS, LangConfig, check_stream, episode_actor, episode_risk,
    shard_g,
    task_list, task_runs_on, val_offset, valg_offset, warmup_offset,
)


class InitLimit(Exception):
    """The process reached max_inits_per_process environment builds."""


# ---------------------------------------------------------------------------
# schedules (Section 7.2)
# ---------------------------------------------------------------------------


def shard_schedule(tasks, emb: str, shard: int, n_shards: int, episodes_per_task: int,
                   g: int, t1_hosts=CANON_T1_HOSTS) -> list[tuple]:
    """Whole cycles over the tasks for one shard: cycles = episodes_per_task /
    n_shards, order within a cycle shuffled with seed g. T1's host in cycle c
    of shard i is t1_hosts[(i * cycles + c) mod 5] (balanced over the union
    when 5 divides episodes_per_task). Tasks not registered for emb are left out (tray_slide: ur5-wsg only).
    -> [(task, emb, host)]."""
    if episodes_per_task % n_shards:
        raise ValueError(f"n_shards={n_shards} must divide episodes_per_task={episodes_per_task}")
    cycles = episodes_per_task // n_shards
    rng = np.random.default_rng(g)
    tasks = [t for t in tasks if task_runs_on(t, emb)]
    out = []
    for c in range(cycles):
        order = list(tasks)
        rng.shuffle(order)
        for t in order:
            host = t1_hosts[(shard * cycles + c) % len(t1_hosts)] if t == "selective_avoid" else None
            out.append((t, emb, host))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def item_n_tok(cfg) -> int:
    """Token-mask cells per camera in the item sidecar: the scene token grid of the config's encoder."""
    if getattr(cfg, "qwen", False):
        from .qwen_masks import QWEN_N_TOK

        return QWEN_N_TOK
    from main.item_envs.masks import token_grid

    return token_grid()[1]


def make_pair(out_root: Path, cfg: LangConfig, capacity: int, *, neck: bool = False):
    """A fresh StepBuffer + lockstep sidecar under out_root (never reused). neck=True (version 2): also the
    neck_buffer/ sidecar (same capacity and rows), returned third. Under the Qwen encoder the sidecar's token masks
    are on the 9 x 11 grid (99 cells per camera)."""
    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.train.buffer import StepBuffer

    buf = StepBuffer(out_root / "buffer", capacity=int(capacity), header=cfg.to_dict())
    sidecar = ItemSidecarBuffer.for_step_buffer(buf, n_tok=item_n_tok(cfg))
    if not neck:
        return buf, sidecar
    from .neck_buffer import NeckBuffer

    return buf, sidecar, NeckBuffer.for_step_buffer(buf)


def open_neck(out_root: Path, buf):
    """The neck_buffer/ of a version-2 shard at resume, opened directly (r+, the shapes of its header): for_step_buffer
    refuses a pair whose n differ, which a kill between the flushes leaves, before resume_point rolls the rings back
    (rounds.make_ring parity). A neck buffer never created (the first launch died between the creations) is paired
    fresh while the StepBuffer is still empty."""
    from .neck_buffer import NeckBuffer

    root = Path(out_root) / "neck_buffer"
    if (root / "header.json").exists():
        hdr = json.loads((root / "header.json").read_text())
        return NeckBuffer(root, capacity=buf.capacity, level_shapes=[tuple(x) for x in hdr["level_shapes"]])
    if len(buf) == 0:
        return NeckBuffer.for_step_buffer(buf)
    raise SystemExit(f"{out_root}: a version-2 shard with {len(buf)} rows and no neck_buffer/; refusing to resume")


def open_extras(out_root: Path, buf, *, kv_dims: dict | None = None, frames: bool = False) -> dict:
    """The V6 sidecars of a shard or ring, keyed as their rows ride in the StepBuffer entries: {"_kv": KvBuffer,
    "_frames": FramesBuffer} (each only when asked for). Fresh next to an empty StepBuffer; otherwise reopened r+
    with their own header (resume_point then rolls them back with the pair, as open_neck), refused when the kv rows
    come from another HoloBrain-Qwen checkpoint or a sidecar is missing."""
    from .frames_buffer import FramesBuffer
    from .kv_buffer import KvBuffer

    out = {}
    for key, cls, want in (("_kv", KvBuffer, kv_dims is not None), ("_frames", FramesBuffer, frames)):
        if not want:
            continue
        root = Path(out_root) / cls.DIRNAME
        if (root / "header.json").exists():
            sc = cls.reopen(root)
            if sc.capacity != buf.capacity:
                raise SystemExit(f"{root}: capacity {sc.capacity}, the StepBuffer's is {buf.capacity}")
            if key == "_kv":
                for k in ("ckpt_id", "attn", "encoder_version"):
                    have = sc.header.get(k, 1 if k == "encoder_version" else None)
                    if kv_dims.get(k) is not None and have != kv_dims[k]:
                        raise SystemExit(f"{root}: kv rows of {k} {have}, this process uses {kv_dims[k]}")
        elif len(buf) == 0:
            sc = cls.for_step_buffer(buf, root=root, **(kv_dims if key == "_kv" else {}))
        else:
            raise SystemExit(f"{out_root}: {len(buf)} rows and no {cls.DIRNAME}/; refusing to resume")
        out[key] = sc
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in (sorted(x) if isinstance(x, set) else x)]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _rule_outcomes(sidecar, rows: np.ndarray, n_rules: int) -> tuple[list, list]:
    """(violated per rule, min h per rule) over the episode's sidecar rows."""
    if rows.size == 0 or n_rules == 0:
        return [False] * n_rules, [None] * n_rules
    H = np.asarray(sidecar.arrays["h_rules"][rows, :n_rules], dtype=np.float32)
    return [bool(v) for v in (H < 0).any(0)], [float(v) for v in H.min(0)]


def scene_meta(env, runner) -> dict:
    """The task metadata in_way reads (so designations can be recomputed
    offline): T1 spawn placements + corridors, T3 layout elements, T5-T7
    episode_info."""
    out = {}
    rep = getattr(runner, "_spawn_report", None) or {}
    if rep:
        out["placements"] = [{k: p.get(k) for k in ("item_id", "cls", "xyz", "degraded_from", "lateral", "asset",
                                                    "model_id", "category")}
                             for p in rep.get("placements") or []]
        try:
            from main.envs.tasks import TASK_SPECS, iter_corridors

            out["corridors"] = [[np.asarray(a)[:2].tolist(), np.asarray(b)[:2].tolist(), str(la), str(tg)]
                                for a, b, la, tg in iter_corridors(env.task, TASK_SPECS[runner.host])]
        except Exception as exc:
            out["corridors_error"] = str(exc)[:200]
    layout = getattr(env.task, "layout_info", None)
    if isinstance(layout, dict):
        out["layout"] = {k: layout.get(k) for k in ("scenario", "mode", "referent", "elements") if k in layout}
    info = getattr(env.task, "episode_info", None)
    if isinstance(info, dict):
        out["episode_info"] = info
    return _jsonable(out)


def choose_named_rule(record: dict, desig: dict, rng: np.random.Generator,
                      named_harm_frac: float = 0.5, allowed=None) -> dict | None:
    """Section 7.1 item 4: a SAMPLEABLE harm-designated rule with probability
    named_harm_frac, else a sampleable benign-designated one; the other kind
    when the wanted one does not exist; None when neither exists. allowed: the
    consequences the run names (cfg.rule_consequences), None = all."""
    pool = {"harm": [], "benign": []}
    for r in record["rules"]:
        d = desig.get(r["rule_id"])
        if r.get("sampleable") and d in pool and (allowed is None or r["consequence"] in allowed):
            pool[d].append(r)
    want = "harm" if rng.random() < named_harm_frac else "benign"
    if not pool[want]:
        want = "benign" if want == "harm" else "harm"
    if not pool[want]:
        return None
    return pool[want][int(rng.integers(len(pool[want])))]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def run_episodes(cfg: LangConfig, out_root: Path, slots: list[tuple], episode_offset: int,
                 buf, sidecar, *, encoder=None, kins=None, models=None, filter_frac: float = 0.0,
                 rng_policy: np.random.Generator | None = None, attempt0: int = 0,
                 inits: dict | None = None, video_every: int = 0, text_store=None,
                 max_retries: int = 5, log_prefix: str = "", slot_offset: int = 0,
                 with_masks: bool = True, deadline: float | None = None, extra_line: dict | None = None,
                 round_metrics=None, writer=None, neck_buf=None, video_filter_only: bool = False,
                 qwen=None, kv_buf=None, frames_buf=None) -> dict:
    """Run one episode per slot (task, emb, host) into (buf, sidecar).

    Episode i of the stream has seed = episode_id = episode_offset + attempt
    (attempt counts every env build, retries included). A slot (task, emb,
    host, seed) is a replayed episode (--replay): that seed, one build, no new
    seed on failure (a skipped line). Returns a summary
    dict; appends one line per slot to out_root/lang_episodes.jsonl.
    models: (policy_enc, actor, critics) for filter episodes (Phase 4), plus the boundary actor as a fourth entry
    when the run trains one (each filter episode then draws its actor, config.episode_actor),
    None = nominal only. slot_offset: index of slots[0] in the full schedule
    (resume). writer: the data root's WriterLock, checked about once a second during an episode and before
    every flush and line (round 2 B1). with_masks=False skips the per-tick segmentation masks (zeros in
    the sidecar; only validation data feeds ref_mask). deadline (unix time):
    no new slot starts when the mean episode wall time would pass it
    (summary["stopped"] = True; the caller exits 75 and resumes later).
    neck_buf: the NeckBuffer of a version-2 shard (cfg.lang_version 2): the controller keeps each capture's neck
    and ro.buf becomes a NeckLockstepWriter, so slot i of the three rings is the same capture; guarded, synced,
    flushed and rolled back with the pair, and every line carries lang_version 2.
    video_filter_only: record videos of filter episodes only (rounds training; Yuxuan, 2026-10-11: the
    nominal-episode videos are not wanted in the training log). Warmup / validation keep every video.
    qwen (V6): the HoloBrainQwenEncoder, exactly when cfg.encoder is holobrain_qwen, with kv_buf (its KvBuffer) and
    optionally frames_buf (FramesBuffer): the controller keeps each capture's rows, ro.buf becomes a
    SidecarLockstepWriter, the item masks are on the Qwen grid; the rings are guarded, synced, flushed and rolled
    back with the pair. cfg.spawn_pool is installed before the first env.
    """
    import torch

    from main.item_envs import language
    from main.item_envs.controller import RiskTickController
    from main.item_envs.margin_hook import wire_item_episode
    from main.train.collect import CKPT_DIR, make_kinematics

    from .rollout import LangRolloutController
    from .scene_roles import all_designations, harm_reasons, in_way, in_way_reasons, mask_region_rows

    if (neck_buf is not None) != (int(getattr(cfg, "lang_version", 1)) >= 2):
        raise ValueError(f"lang_version {getattr(cfg, 'lang_version', 1)} with neck_buf "
                         f"{'given' if neck_buf is not None else 'None'}: a version-2 shard is written with its neck "
                         "buffer, a version-1 shard without")
    if neck_buf is not None:
        from .neck_buffer import NeckLockstepWriter

        extra_line = {**(extra_line or {}), "lang_version": int(cfg.lang_version)}
    if (qwen is not None) != bool(getattr(cfg, "qwen", False)):
        raise ValueError(f"encoder {getattr(cfg, 'encoder', 'holobrain_gd')} with qwen "
                         f"{'given' if qwen is not None else 'None'}: the Qwen encoder runs exactly under its config")
    if qwen is not None and kv_buf is None:
        raise ValueError("the Qwen encoder writes its prefix keys / values: kv_buf is required")
    if qwen is None and (kv_buf is not None or frames_buf is not None):
        raise ValueError("kv_buf / frames_buf are the V6 sidecars: they need qwen")
    extras = {k: b for k, b in (("_kv", kv_buf), ("_frames", frames_buf)) if b is not None}
    if qwen is not None:
        extra_line = {**(extra_line or {}), "encoder": cfg.encoder}
    from .legacy_pool import install_spawn_pool, install_vocab

    install_vocab(getattr(cfg, "vocab", "current"))   # before anything names an object (episode_record, rules)
    install_spawn_pool(getattr(cfg, "spawn_pool", "current"))
    necks = ((neck_buf,) if neck_buf is not None else ()) + tuple(extras.values())
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl = out_root / "lang_episodes.jsonl"
    if encoder is None:
        from main.network.encoder import HoloBrainEncoder

        encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kins = {} if kins is None else kins
    inits = {"n": 0} if inits is None else inits
    rng_policy = rng_policy if rng_policy is not None else np.random.default_rng(episode_offset)
    attempt = int(attempt0)
    summary = {"episodes": 0, "skipped": 0, "status": {}, "inits": 0, "wall": 0.0, "stopped": False}
    if writer is not None:
        guard_writes(writer, buf, sidecar, *necks)
    for slot_i, slot in enumerate(slots):
        task, emb, host = slot[:3]
        fixed = int(slot[3]) if len(slot) > 3 else None   # a replayed episode's recorded seed
        if deadline is not None:
            est = summary["wall"] / summary["episodes"] if summary["episodes"] else 420.0
            if time.time() + 1.2 * est > deadline:
                summary["stopped"] = True
                print(f"{log_prefix}[collect] deadline: stopping before slot {slot_offset + slot_i} "
                      f"(mean episode {est:.0f}s)", flush=True)
                break
        if emb not in kins:
            kins[emb] = make_kinematics(CKPT_DIR, emb)
        use_filter = models is not None and rng_policy.random() < filter_frac
        ro, seed, errors = None, None, []
        for _ in range(max_retries if fixed is None else 1):
            if inits["n"] >= cfg.max_inits_per_process:
                raise InitLimit(f"{inits['n']} environment inits")
            seed = check_stream(episode_offset, attempt) if fixed is None else fixed
            attempt += 1
            inits["n"] += 1
            # item_risk: a speed-risk task's behaviour for this seed (config.episode_risk; None for the others)
            cfg_ep = dataclasses.replace(
                cfg, task=task, embodiment=emb, item_host=host, item_active_rule=None,
                item_out_root=str(out_root / "_runner"), item_risk=episode_risk(cfg, task, seed),
            )
            # the actor of this episode's interventions: models[3] is the boundary actor when the run trains one
            # (T8_BOUNDARY_PLAN.md section 5; drawn from the seed, so a resume repeats it)
            actor_kind = episode_actor(cfg, seed) if (models and use_filter and len(models) > 3) else "argmax"
            try:
                cand = LangRolloutController(
                    cfg_ep, seed, mode="collect", encoder=encoder, kin=kins[emb],
                    policy_enc=models[0] if (models and use_filter) else None,
                    actor=(models[3] if actor_kind == "boundary" else models[1]) if (models and use_filter) else None,
                    critics=models[2] if (models and use_filter) else None,
                    filter_active=use_filter, buf=buf, episode_id=seed,
                    perturb_prob=cfg.perturb_prob, rng=np.random.RandomState(seed),
                    max_steps=10 * int(cfg.max_steps_per_episode),
                    controller_cls=RiskTickController, text_store=text_store, keep_neck=neck_buf is not None,
                    qwen=qwen, keep_kv=kv_buf is not None, keep_frames=frames_buf is not None,
                )
            except Exception as exc:
                errors.append(f"seed {seed}: build {type(exc).__name__}: {exc}"[:300])
                print(f"{log_prefix}[collect] {task}/{emb} seed {seed}: build failed ({exc}); "
                      f"{'new seed' if fixed is None else 'replayed seed: no retry'}")
                continue
            if cand.env.obstacle is None:  # no rules (layout-infeasible T1 seed)
                errors.append(f"seed {seed}: no rules")
                try:
                    cand.env.close()
                except Exception:
                    pass
                continue
            ro = cand
            break
        if ro is None:
            summary["skipped"] += 1
            if writer is not None:
                writer.check()
            with open(jsonl, "a") as f:
                f.write(json.dumps({"skipped": True, "task": task, "embodiment": emb, "host": host,
                                    "errors": errors, "slot": slot_offset + slot_i,
                                    "attempt_next": attempt, **(extra_line or {})}) + "\n")
            continue

        t_ep = time.time()
        env, runner = ro.env, ro.env._item_runner
        record = language.episode_record(runner, seed)
        roles = in_way(env, record)
        desig = all_designations(record, roles)
        reasons = harm_reasons(record, in_way_reasons(env, record))   # why each harm rule is (decision 3)
        primary = runner._primary_rule_id(env.task)
        info = getattr(env.task, "episode_info", None) or {}
        layout = getattr(env.task, "layout_info", None) or {}
        region_rows = mask_region_rows(runner)
        smeta = scene_meta(env, runner)   # before run(): run() closes the env (actor poses gone)
        # the nominal's named behaviour (tray_slide's risk knob; its designations follow it), only where a task has one
        behaviour = (getattr(env.task, "risk_knobs", None) or {}).get("behaviour")
        # its drag-profile overrides, if the mix gave any (config.behaviour_risk, e.g. a late, sharp recovery)
        b_knobs = {k: v for k, v in (getattr(env.task, "risk_knobs", None) or {}).items() if k != "behaviour"} \
            if behaviour is not None else {}
        named, sentence, cf_sentence, cf_rule = None, None, None, None
        from .vlm_prompts import allowed_consequences

        allowed = allowed_consequences(getattr(cfg, "rule_consequences", "all"), task)
        if use_filter:
            rng_ep = np.random.default_rng(seed)
            named = choose_named_rule(record, desig, rng_ep, cfg.named_harm_frac, allowed)
            if named is None:
                ro.filter_active = False
            else:
                runner.active_rule_id = named["rule_id"]
                sentence = language.sample_sentence(named, record, rng_ep, "train")
                # counterfactual: a sampleable rule of the other designation (Q_cf on the panel)
                other = "benign" if desig.get(named["rule_id"]) == "harm" else "harm"
                cands = [r for r in record["rules"] if r.get("sampleable") and desig.get(r["rule_id"]) == other
                         and (allowed is None or r["consequence"] in allowed)]
                if cands:
                    cf_rule = cands[int(rng_ep.integers(len(cands)))]
                    cf_sentence = language.sample_sentence(cf_rule, record, rng_ep, "train")
                ro.set_instruction(sentence, cf_sentence, rule_id=named["rule_id"])
        n_before = len(buf)
        if necks and any(len(b) != n_before for b in (sidecar, *necks)):
            ro.env.close()
            raise RuntimeError(f"{out_root}: rings ({n_before}, {len(sidecar)}, {[len(b) for b in necks]}) out of "
                               f"lockstep before episode {seed}")
        ro.writer = writer
        wire_item_episode(ro, sidecar)
        if qwen is not None:
            from .qwen_masks import use_qwen_grid

            use_qwen_grid(ro)   # the item sidecar's token masks on the 9 x 11 grid of the scene tokens
        if neck_buf is not None:
            ro.buf = NeckLockstepWriter(ro.buf, neck_buf)   # pops each entry's _neck inside the same append
        if extras:
            from .sidecar import SidecarLockstepWriter

            ro.buf = SidecarLockstepWriter(ro.buf, extras)  # pops _kv / _frames inside the same append
        if not with_masks:
            ro.h_fn.with_masks = False
        av = None
        if int(getattr(cfg, "agent_view_every", 0) or 0) > 0:   # the VLM labeler's frames (agent_view.py)
            from .agent_view import AgentViewWriter

            av = ro.agent_view = AgentViewWriter(out_root, seed, int(cfg.agent_view_every))
        if video_every and summary["episodes"] % int(video_every) == 0 and (not video_filter_only or named is not None):
            from .video import LangPanelVideoWriter, video_name

            vdir = out_root / "videos"
            vdir.mkdir(parents=True, exist_ok=True)
            kind = desig.get(named["rule_id"]) if named else "nominal"
            ro.video = LangPanelVideoWriter(vdir / video_name(task, runner.host if task == "selective_avoid" else None,
                                                              emb, seed, kind), ro, named_rule=named, sentence=sentence,
                                            record=record, alt_rule=cf_rule, alt_sentence=cf_sentence)
        if use_filter:
            ctx = torch.no_grad()
            ctx.__enter__()
        lost = False
        try:
            trace = ro.run()
        except BaseException as exc:
            lost = writer is not None and isinstance(exc, type(writer).LockLost)
            raise
        finally:
            if use_filter:
                ctx.__exit__(None, None, None)
            if ro.video is not None:
                # after a lost lock the new owner records the same seed: never publish over its video
                ro.video.discard() if lost else ro.video.close()
        if writer is not None:
            writer.check()
        try:
            ro.buf.end_episode()  # lockstep teardown: every queued sidecar row was paired
        except RuntimeError as exc:
            # a tick raised between queuing its sidecar row and the StepBuffer append (an encode / CUDA error;
            # the base run caught it as an episode error): the pairing of this episode is broken. Roll both
            # rings back to its start and record it as an exception episode, instead of ending the process
            # (independent review, round 2)
            print(f"{log_prefix}[collect] {task}/{emb} seed {seed}: {exc}; episode rolled back", flush=True)
            getattr(ro.buf, "_rows", []).clear()
            buf.discard_to(n_before)
            sidecar.discard_to(n_before)
            for nb in necks:
                nb.discard_to(n_before)
            trace["status"] = "exception"
            trace["exception"] = f"lockstep after a tick error ({exc})"[:800]
        # arrays first (the heavy msync), then the lock check, then the small header writes: a writer that
        # stalls in the sync and wakes after a takeover never writes headers (review round 3, C2)
        sync_arrays(buf, sidecar, *necks)
        if writer is not None:
            writer.check()
        buf.flush()
        sidecar.flush()
        for nb in necks:
            nb.flush()
        if av is not None:   # after the rows are on disk: done.json lets the labeler take the episode
            av.finish(len(buf) - n_before) if len(buf) > n_before else av.discard()
        rows = np.arange(n_before, len(buf)) % buf.capacity
        violated, hmin = _rule_outcomes(sidecar, rows, len(record["rules"]))
        line = {
            **({"behaviour": behaviour} if behaviour is not None else {}),
            **({"behaviour_knobs": b_knobs} if b_knobs else {}),
            "episode_id": seed, "task": task, "host": runner.host if task == "selective_avoid" else None,
            "embodiment": emb, "seed": seed, "filter": bool(use_filter and named is not None),
            **({"actor": actor_kind} if use_filter and named is not None else {}),
            "named_rule_id": named["rule_id"] if named else None,
            "named_consequence": named["consequence"] if named else None,
            "designation": desig.get(named["rule_id"]) if named else None,
            "sentence": sentence, "cf_sentence": cf_sentence, "designations": desig,
            "in_way": sorted(roles[0]), "unclear": sorted(roles[1]), "harm_reasons": reasons,
            "primary": primary, "referent_kind": info.get("referent_kind"),
            "scenario": layout.get("scenario"),
            "status": trace.get("status", "ok"), "exception": trace.get("exception"),
            "success": bool(trace.get("success", False)), "n_ticks": int(trace.get("n_ticks", 0)),
            "rows": [int(n_before), int(len(buf))],
            "rule_ids": [r["rule_id"] for r in record["rules"]],
            "violated": dict(zip([r["rule_id"] for r in record["rules"]], violated)),
            "h_min": dict(zip([r["rule_id"] for r in record["rules"]], hmin)),
            "mask_region_rows": region_rows, "scene_meta": smeta,
            "intervention_rate": trace.get("intervention_rate"),
            "timing": trace.get("timing"), "wall_s": round(time.time() - t_ep, 2),
            "retries": errors, "slot": slot_offset + slot_i, "attempt_next": attempt, "masks": bool(with_masks),
            "video": str(ro.video.path) if getattr(ro.video, "path", None) else None, **(extra_line or {}),
        }
        if writer is not None:
            writer.check()
        with open(jsonl, "a") as f:
            f.write(json.dumps(_jsonable(line)) + "\n")
        if round_metrics is not None and line["filter"]:   # after the line (round 3, C2): never a row without it
            from .metrics import LangRoundMetrics

            rm, r_idx = round_metrics
            rm.add(LangRoundMetrics.episode_row(r_idx, line, trace))
        summary["episodes"] += 1
        summary["status"][line["status"]] = summary["status"].get(line["status"], 0) + 1
        summary["wall"] += line["wall_s"]
        print(f"{log_prefix}[collect] {task}{'/' + str(line['host']) if line['host'] else ''}/{emb} "
              f"seed {seed}: {('FILTER/' + line['actor']) if line['filter'] else 'nominal'} "
              f"{line['named_rule_id'] or ''} {line['designation'] or ''} {sentence or ''} | "
              f"status {line['status']} success {line['success']} ticks {line['n_ticks']} "
              f"violated {sum(violated)}/{len(violated)} wall {line['wall_s']:.0f}s", flush=True)
    summary["inits"] = inits["n"]
    summary["attempt"] = attempt
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def stage_deadline(budget: float | None) -> float | None:
    """min(now + budget, $STAGE_DEADLINE): gpu_stage.sbatch exports the job's
    absolute deadline, so a process started late in a job still stops in time."""
    import os

    dl = time.time() + float(budget) if budget else None
    env = os.environ.get("STAGE_DEADLINE")
    if env:
        dl = min(dl, float(env)) if dl is not None else float(env)
    return dl


def _check_fresh(out: Path) -> None:
    if out.exists() and any(p for p in out.iterdir() if not p.name.startswith(".writer.")):
        raise SystemExit(f"{out} is not empty: shards are collected into fresh directories only")


def guard_writes(writer, *buffers) -> None:
    """Check the writer lock right before every row write of these buffers (review round 3, C1): the
    in-memory heartbeat flag, and the lock files whenever the last look is over 2 s old (always after a
    stall). The rollout's per-capture check alone missed a stall inside a capture (render / encode)."""
    for b in buffers:
        if getattr(b, "_guarded_by", None) is writer:
            continue
        orig = b.__dict__.get("_unguarded_append", b.append)

        def append(row, _orig=orig):
            writer.check_fast()
            return _orig(row)

        b._unguarded_append, b.append, b._guarded_by = orig, append, writer


def sync_arrays(*buffers) -> None:
    """msync the memory-mapped arrays of these buffers (their flush() does this and then writes header.json)."""
    for b in buffers:
        for arr in getattr(b, "arrays", {}).values():
            f = getattr(arr, "flush", None)
            if f is not None:
                f()


def read_lines(jsonl: Path) -> list[dict]:
    """lang_episodes.jsonl lines; a crash-truncated last line is dropped (and the file rewritten)."""
    if not jsonl.exists():
        return []
    raw = jsonl.read_text().splitlines()
    out, bad = [], False
    for ln in raw:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            bad = True
    if bad:   # temp file + replace: a kill mid-rewrite must not cut the record (round 3, C8)
        tmp = jsonl.with_suffix(f".jsonl.{__import__('os').getpid()}.tmp")
        tmp.write_text("".join(json.dumps(d) + "\n" for d in out))
        tmp.replace(jsonl)
    return out


def resume_point(out: Path, buf, sidecar, *extras) -> tuple[int, int, int]:
    """(slots done, attempt, rows) of an interrupted shard; both rings are
    rolled back to the end of the last recorded episode (an episode killed
    mid-run, or flushed without its line, is discarded and collected again
    with the NEXT seed (attempt + 1): rows a stalled former writer might still add then carry another
    episode id than the new rows at the same positions, so check_shards sees them; review round 3, C1).
    extras: a version-2 shard's NeckBuffer, a V6 shard's KvBuffer / FramesBuffer (None entries are ignored), each
    checked and rolled back exactly like the item sidecar (their rows hold episode_id / t_in_ep too)."""
    lines = read_lines(out / "lang_episodes.jsonl")
    done = len(lines)
    attempt = int(lines[-1]["attempt_next"]) if lines else 0
    rows = 0
    for d in lines:
        if not d.get("skipped"):
            rows = int(d["rows"][1])
    rings = (buf, sidecar) + tuple(x for x in extras if x is not None)
    if any(len(b) < rows for b in rings):
        raise RuntimeError(f"{out}: rings {tuple(len(b) for b in rings)} shorter than the recorded rows {rows}")
    # every recorded episode's END rows must hold that episode (both buffers) with ticks 0 and n-1: this
    # catches two writers and most copies made mid-write (review B4), but not every torn array (only the end
    # rows are read). The real guard is copying finished shards only (scripts/lang/bridges/README.md)
    eb = buf.arrays.get("episode_id")
    sides = [(b.arrays["episode_id"], b.arrays["t_in_ep"]) for b in rings[1:]]   # item sidecar (+ neck buffer)
    for d in lines:
        if d.get("skipped"):
            continue
        r0, r1 = (int(x) for x in d["rows"])
        eid = int(d["episode_id"])
        if r1 == r0:   # an episode that failed before its first tick wrote no rows (review, round 2)
            continue
        ends_ok = (r1 > r0 and all(int(es[r0]) == eid and int(es[r1 - 1]) == eid and int(ts[r0]) == 0
                                   and int(ts[r1 - 1]) == r1 - r0 - 1 for es, ts in sides)
                   and (eb is None or (int(eb[r0]) == eid and int(eb[r1 - 1]) == eid)))
        if not ends_ok:
            raise RuntimeError(f"{out}: episode {eid} rows [{r0}, {r1}) do not hold that episode (a partial copy or "
                               f"two writers?); refusing to resume: check with scripts/lang/check_shards.py")
    if any(len(b) != rows for b in rings):
        print(f"[collect] resume: discarding rows {rows}..{max(len(b) for b in rings)} of an interrupted episode")
        for b in rings:
            b.discard_to(rows)
        for b in rings:
            b.flush()
        attempt += 1
    return done, attempt, rows


# Config fields that shape only training, evaluation or bookkeeping, never what a collection episode does.
# collect_hash leaves them out, so adding or changing one (e.g. tf32_train) does not stop a shard from
# resuming (review round 3, C5).
TRAIN_ONLY_FIELDS = frozenset({
    # CrossSafe FrozenConfig
    "gamma", "gamma_final", "gamma_anneal_steps", "batch_size", "lr", "tau", "alpha_start", "alpha_final",
    "alpha_anneal_steps", "grad_steps", "eval_every", "eval_sweep_every_epochs", "eval_ema", "checkpoint_every",
    "save_every_epochs", "eval_seeds", "ablate_geometry", "ablate_injection", "video_bbox", "n_episodes",
    # LangConfig: model, sampling, precision, rounds bookkeeping
    "k_prev", "n_queries", "max_text_tokens", "text_backend", "backend_paths", "trunk_depth", "critic_depth",
    "lang_scene", "lang_body", "lang_film", "lang_queries", "lang_rtoken", "film_zero_init", "softmin_T_h",
    "activation_checkpointing", "tf32_train", "actor_hidden", "text_out_norm", "mask_special_tokens", "oracle_ref_tag",
    "oracle_tag_sigma", "aux_target_loss", "aux_loss_weight", "aux_huber_delta", "aux_unit", "aux_attention_loss",
    "aux_attention_weight", "shuffle_text", "relabel", "harmful_frac", "pair_frac", "task_balance",
    "recall_horizon_ticks", "drop_stale_bead_rows", "episodes_per_round", "grad_steps_per_round", "collect_rounds",
    "named_harm_frac", "filter_episode_frac", "warmup_episodes_per_task", "val_episodes_per_task",
    # V5 / decision 6 / E6 (2026-10-09): the frozen fusion runs on stored neck maps, the readout and the episode
    # subset are training-side only. lang_version stays IN the hash: a version-2 shard is a different collection
    "scene_fusion", "fusion_words", "fusion_dtype", "fusion_chunk", "critic_readout", "critic_mlp_hidden",
    "train_episodes_per_task",
    # 2026-11-10 (ACTION_SENSITIVITY_PLAN): how the action enters the critic; model only (review 2026-11-17: they
    # were left in the hash, which changed it for every shard collected before them)
    "critic_injection", "critic_chunk",
    # V6 (2026-10-14): which prefix the words read is decided at training time (the shard stores every frame's)
    "qwen_words_prefix",
    # VLM failure function (2026-10-26): the labels are read at training time, next to the rows
    "h_source",
})

# Collection fields added after shards were written (V6, 2026-10-14). At these values they change nothing a
# collection does, so collect_hash leaves them out and every shard written before them keeps its hash
# (qwen_attn matters only with the Qwen encoder)
NEUTRAL_COLLECT_FIELDS = {"encoder": "holobrain_gd", "spawn_pool": "current", "vocab": "current",
                          "rule_consequences": "all", "agent_view_every": 0,
                          # 2026-11-15 / 17 (the boundary actor, the 3-tick decision): rounds-only settings. The T8
                          # shards collected on 11-15 / 16 stored a hash WITH the boundary fields, so resuming one of
                          # them reports collect_hash (they are complete; collect into a new shard instead)
                          "boundary_actor": False, "boundary_lo": 0.0, "boundary_hi": 0.1,
                          "boundary_actor_frac": 0.5, "decision_ticks": 1}


def collect_fields(cfg) -> dict:
    """The config fields a collection episode depends on: everything but TRAIN_ONLY_FIELDS, without the
    post-hoc fields at their neutral values."""
    d = {k: v for k, v in cfg.to_dict().items() if k not in TRAIN_ONLY_FIELDS}
    for k, neutral in NEUTRAL_COLLECT_FIELDS.items():
        if d.get(k, neutral) == neutral:
            d.pop(k, None)
    if d.get("encoder", "holobrain_gd") == "holobrain_gd":
        d.pop("qwen_attn", None)
    # the behaviour mix of the config's own tasks only (2026-11-13): a shard without tray_slide keeps its hash
    mix = {t: m for t, m in (d.pop("behaviour_mix", None) or {}).items() if t in (d.get("task_choices") or ())}
    if mix:
        d["behaviour_mix"] = mix
    return d


def collect_hash(cfg) -> str:
    """Hash of the config fields a collection episode depends on (collect_fields)."""
    import hashlib

    return hashlib.md5(json.dumps(collect_fields(cfg), sort_keys=True, default=str).encode()).hexdigest()[:12]


# what defines h (review round 3, C6): whole files, plus the h functions of files that mostly hold other code
H_FILES = ("margins.py", "rules.py", "geometry.py", "robot_distance.py", "item_shapes.py", "regions.py",
           "configs/objects.yaml")
H_FUNCS = {"runner.py": ("rule_margins", "rule_violation_counts", "_pose7", "_velocity6"),
           "margin_hook.py": ("ItemMarginHFn",)}


def _source_of(path: Path, names) -> str:
    """The source of the named top-level functions / classes (parsed, not imported)."""
    import ast

    src = path.read_text()
    nodes = {n.name: n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    missing = sorted(set(names) - set(nodes))
    if missing:
        raise RuntimeError(f"h_code: {path} has no {missing} (update collect.H_FUNCS)")
    return "\n".join(ast.get_source_segment(src, nodes[n]) for n in names)


def h_code_hash_v2(tasks) -> str:
    """Hash of exactly what defines h for these tasks (review round 3, C6): margins, rules, the distance and
    shape code, regions, the object table, the robot's collision spheres, the h functions of runner.py and
    margin_hook.py, the tasks' yaml files, the RoboTwin scene builders this checkout loads (its own
    envs/robotwin_tasks, since 2026-10-23; before, the shared RoboTwin/envs links), and the distance mode (ITEM_DISTANCE). Unrelated edits to runner / item_buffer / margin_hook
    no longer refuse a resume. Recorded as `h_code_v2`; `h_code` (h_code_hash) stays for older shards / runs."""
    import hashlib
    import os

    from . import REPO

    root = REPO / "envs/main/item_envs"
    h = hashlib.md5()

    def add(name: str, data: bytes) -> None:
        h.update(name.encode())
        h.update(data)

    for f in H_FILES:
        add(f, (root / f).read_bytes())
    for f in sorted((root / "configs/collision_spheres").glob("*.yml")):
        add(f"collision_spheres/{f.name}", f.read_bytes())
    for f, names in H_FUNCS.items():
        add(f, _source_of(root / f, names).encode())
    # the scene builders this checkout's runner loads (2026-10-23: runner.use_own_robotwin_tasks imports the
    # checkout's envs/robotwin_tasks first, not the shared RoboTwin/envs links): each task's registered RoboTwin
    # module (T6 clean_pour -> bowl_pour) and, followed through their relative imports, every local file they build
    # on (granular_pour, _item_utils, push_zone, ...). Host tasks (T1) are RoboTwin's own files and add nothing.
    import re

    from main.item_envs.registry import ITEM_TASKS

    own = REPO / "envs/robotwin_tasks"
    seen: set = set()

    def add_module(name: str) -> None:
        f = own / f"{name}.py"
        if name in seen or not f.exists():
            return
        seen.add(name)
        src = f.read_bytes()
        add(f"robotwin_tasks/{name}.py", src)
        for dep in re.findall(rb"^from \.(\w+) import", src, re.M):
            add_module(dep.decode())

    for t in sorted(tasks):
        for y in sorted((root / "configs/tasks").glob(f"*_{t}.yaml")):
            add(y.name, y.read_bytes())
        entry = ITEM_TASKS.get(t)
        if entry is not None and entry.kind == "robotwin":
            add_module(str(entry.robotwin_task))
    add_module("_item_utils")
    add("ITEM_DISTANCE", (os.environ.get("ITEM_DISTANCE") or "shape").encode())   # robot_distance's default
    return h.hexdigest()[:12]


def h_code_hash(tasks) -> str:
    """The first h hash (kept to compare shards and runs that recorded only it): margins.py, rules.py,
    runner.py, margin_hook.py, item_buffer.py and the tasks' yaml files. Recorded in shard.json: a shard
    written under another h definition (e.g. the 1 mm soft-min bead margin of 2026-10-07) refuses to resume
    (round 2 B2)."""
    import hashlib

    from . import REPO

    root = REPO / "envs/main/item_envs"
    # margins and rules define h; runner / margin_hook compute margins and bead counts per tick; item_buffer
    # turns h into the fails_soon labels
    files = [root / f for f in ("margins.py", "rules.py", "runner.py", "margin_hook.py", "item_buffer.py")]
    for t in sorted(tasks):
        files += sorted((root / "configs/tasks").glob(f"*_{t}.yaml"))
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def shard_meta_diff(old: dict, meta: dict, out: Path) -> list:
    """The shard.json fields that stop a resume. config_hash is recorded only: the collection fields are
    compared through collect_hash, recomputed from the shard's config.json for shards written before it
    existed (review round 3, C5). h is compared on h_code_v2 when the old shard has it, else on h_code."""
    # lang_version: absent = 1 (every shard before V5); replay_of: where the replayed shard was read (its slots,
    # seeds included, are compared, so a moved source still resumes)
    skip = {"config_hash", "collect_hash", "h_code", "h_code_v2", "lang_version", "replay_of"}
    diff = [k for k in meta if k not in skip and old.get(k) != meta[k]]
    if old.get("lang_version", 1) != meta.get("lang_version", 1):
        diff.append("lang_version")
    old_c = old.get("collect_hash")
    if old_c is None and (out / "config.json").exists():
        old_c = collect_hash(LangConfig.load(out / "config.json"))
    if old_c != meta["collect_hash"]:
        diff.append("collect_hash")
    key = "h_code_v2" if "h_code_v2" in old else "h_code"
    if old.get(key) != meta[key]:
        diff.append(key)
    return sorted(diff)


def replay_plan(src: Path, kind: str, emb: str, tasks=None) -> dict:
    """What --replay collects again: the recorded episodes of the FINISHED shard src for these tasks (None = all)
    and this embodiment, in slot order, each with its own seed -> {"meta": src's shard.json, "cfg": its config,
    "slots": [(task, emb, host, episode_id)], "src_slots": their slot indices in src}.

    Every recorded episode counts, whatever its status (an exception episode is replayed too); a skipped slot (no
    build succeeded, so no layout was recorded) is left out. Read-only: src is never rewritten (read_lines repairs
    a truncated line in place; here a bad line refuses)."""
    src = Path(src).resolve()
    if not (src / "shard.json").exists():
        raise SystemExit(f"--replay {src}: no shard.json")
    meta = json.loads((src / "shard.json").read_text())
    summ = src / "summary.json"
    if not (summ.exists() and json.loads(summ.read_text()).get("complete")):
        raise SystemExit(f"--replay {src}: not a finished shard (no complete summary.json)")
    if (meta.get("kind"), meta.get("embodiment")) != (kind, emb):
        raise SystemExit(f"--replay {src}: a {meta.get('kind')} shard of {meta.get('embodiment')}; this command "
                         f"collects {kind} for {emb}")
    slots, src_slots = [], []
    for i, ln in enumerate((src / "lang_episodes.jsonl").read_text().splitlines()):
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            raise SystemExit(f"--replay {src}: line {i + 1} of lang_episodes.jsonl is not JSON (a partial copy?)")
        if d.get("skipped") or d.get("embodiment") != emb or (tasks is not None and d.get("task") not in tasks):
            continue
        slots.append((d["task"], emb, d.get("host"), int(d["episode_id"])))
        src_slots.append(int(d["slot"]))
    if not slots:
        raise SystemExit(f"--replay {src}: no recorded {emb} episode of tasks {tasks or 'all'}")
    return {"meta": meta, "cfg": LangConfig.load(src / "config.json"), "slots": slots, "src_slots": src_slots}


# collection fields a replay may change: what is stored per tick, never what happens in the episode
REPLAY_FREE_FIELDS = frozenset({"lang_version", "encoder", "qwen_attn", "agent_view_every"})


def config_drift(old, new) -> list:
    """The collection fields (outside TRAIN_ONLY_FIELDS and REPLAY_FREE_FIELDS) on which two configs differ."""
    a, b = (json.loads(json.dumps(c.to_dict(), default=str)) for c in (old, new))
    return sorted(k for k in set(a) | set(b)
                  if k not in TRAIN_ONLY_FIELDS and k not in REPLAY_FREE_FIELDS and a.get(k) != b.get(k))


def legacy_replay(src_cfg, cfg) -> bool:
    """A replay of a pre-10-12 shard with the pre-10-12 T1 pool (V6): its bank hash and spawn pool differ from the
    source's by design (today's names, the old draws), and the layout check verifies the placements."""
    from .legacy_pool import PRE_1012_BANK_HASH

    return src_cfg.bank_hash == PRE_1012_BANK_HASH and getattr(cfg, "spawn_pool", "current") == "pre_1012"


def _same(a, b, tol: float) -> bool:
    """a's recorded content equals b's: dict keys of a only (b may record more), lists element-wise, floats within
    tol, everything else exactly."""
    if isinstance(a, dict):
        return isinstance(b, dict) and all(k in b and _same(v, b[k], tol) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _episode_items(root: Path) -> dict:
    """episode_id -> the item table item_buffer/episodes.jsonl recorded LAST for it (a rolled-back episode leaves a
    stale line before its replacement)."""
    out = {}
    p = Path(root) / "item_buffer" / "episodes.jsonl"
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out[int(d["episode_id"])] = d.get("items") or []
    return out


ITEM_KEYS = ("item_id", "category", "color", "kind", "role", "half_extents", "obb_center_local", "mass")
# episode_info fields that are OUTCOMES of the pre-roll before t = 0, not layout draws: T4's prepare_episode settles
# the beads (physics) and grasps / levels / tilts with cuRobo plans (nondeterministic run to run), then records how
# many beads it removed or trimmed, whether it re-grasped, fell back to the table grasp, aborted a levelling step,
# and the tilt it reached (granular_carry.py). The layout fields next to them (carry_mode, n_beads_spawned,
# interior_depth) are still compared. Measured on the V6 replays (2026-10-14): n_beads 129 / 133, 134 / 133, 133 / 115
# for the same seeds.
OUTCOME_INFO_KEYS = frozenset({"n_beads", "n_beads_removed", "n_beads_trimmed", "level_aborted", "held_regrasp",
                               "held_fallback", "carry_tilt_eff_deg"})


def replay_layout_mismatch(out: Path, src: Path, tol: float = 1e-5) -> list:
    """How the replayed episodes' scenes differ from the recorded ones (V6 layout check), per episode id: the item
    tables (ids, categories, colours, sizes, masses), T1's placements (item id, class, xyz within tol, asset and
    model id where the source has them), the other tasks' layout / episode_info (the source's keys, without the
    pre-roll outcomes OUTCOME_INFO_KEYS). [] = every replayed episode stands on its recorded scene."""
    src_lines = {int(d["episode_id"]): d for d in read_lines(Path(src) / "lang_episodes.jsonl") if not d.get("skipped")}
    src_items, new_items = _episode_items(src), _episode_items(out)
    diffs = []
    for d in read_lines(Path(out) / "lang_episodes.jsonl"):
        if d.get("skipped"):
            continue
        eid = int(d["episode_id"])
        s = src_lines.get(eid)
        if s is None:
            diffs.append(f"episode {eid}: not in the replayed shard")
            continue
        a, b = src_items.get(eid), new_items.get(eid)
        if a is None or b is None or len(a) != len(b):
            diffs.append(f"episode {eid}: item tables {None if a is None else len(a)} vs {None if b is None else len(b)}")
        else:
            for x, y in zip(a, b):
                bad = [k for k in ITEM_KEYS if k in x and not _same(x[k], y.get(k), tol if k != "mass" else 1e-9)]
                if bad:
                    diffs.append(f"episode {eid}: item {x.get('item_id')} differs in {bad}")
        sm, nm = s.get("scene_meta") or {}, d.get("scene_meta") or {}
        if d.get("task") == "selective_avoid":
            pa, pb = sm.get("placements") or [], nm.get("placements") or []
            if len(pa) != len(pb):
                diffs.append(f"episode {eid}: {len(pa)} recorded placements, {len(pb)} replayed")
            for x, y in zip(pa, pb):
                keys = ["item_id", "cls", "xyz"] + [k for k in ("asset", "model_id") if x.get(k) is not None]
                bad = [k for k in keys if not _same(x.get(k), y.get(k), tol)]
                if bad:
                    diffs.append(f"episode {eid}: placement {x.get('item_id')} differs in {bad}")
        else:
            for key in ("layout", "episode_info"):
                want = sm.get(key)
                if key == "episode_info" and isinstance(want, dict):
                    want = {k: v for k, v in want.items() if k not in OUTCOME_INFO_KEYS}
                if key in sm and not _same(want, nm.get(key), tol):
                    diffs.append(f"episode {eid}: {key} differs")
    return diffs


def replay_rows_missing(d: dict) -> bool:
    """An episode line without rows in the rings: rows [r0, r1) with r1 <= r0 (or no rows field). run_episodes
    writes it when a tick error came between the h_fn's sidecar row and the StepBuffer append (neck_to_row's fp16
    overflow, an encode error) and the episode was rolled back, or when the first capture failed before any row."""
    rows = d.get("rows")
    return not rows or int(rows[1]) <= int(rows[0])


def replay_mismatch(lines: list, slots) -> list:
    """How a finished replay shard differs from its plan: slot i must hold the recorded episode slots[i] (task,
    embodiment, host, episode_id) WITH rows. A skipped slot (its one build failed) is a difference, and so is an
    episode rolled back to zero rows (replay_rows_missing): its layout is missing from the rings either way.

    The status is not compared: cuRobo redraws the motion on every run, so a replayed episode may end ok where the
    source raised, or raise where the source was ok. An exception episode that kept its rows holds the recorded
    layout up to the error and is not a difference (replay_plan replays those too)."""
    out = [] if len(lines) == len(slots) else [f"{len(lines)} episode lines for {len(slots)} replayed slots"]
    for i, (d, s) in enumerate(zip(lines, slots)):
        got = (d.get("task"), d.get("embodiment"), d.get("host"), d.get("episode_id"))
        if d.get("skipped"):
            out.append(f"slot {i}: skipped ({'; '.join(d.get('errors') or [])}) instead of {tuple(s)}")
            continue
        if got != tuple(s):
            out.append(f"slot {i}: {got} instead of {tuple(s)}")
        if replay_rows_missing(d):
            out.append(f"slot {i}: no rows for {tuple(s)} (rows {d.get('rows')}, status {d.get('status')}: "
                       f"{(d.get('exception') or '')[:120]})")
    return out


def plan_shard(args, kind: str):
    """What cmd_shard collects, before anything is written -> (cfg, shard.json meta, slots, episode offset, masks).

    Version 2 (--lang-version 2): cfg.lang_version = 2 (in collect_hash) and meta["lang_version"] = 2; a version-1
    meta is exactly the one before V5. --replay <shard>: the shard's recorded episodes (replay_plan) with its shard
    / g / offset / masks / task list, refused when today's collection config differs from its config."""
    tasks = None if args.tasks == "all" else task_list(args.tasks)
    replay = getattr(args, "replay", None)
    enc = getattr(args, "encoder", None) or "holobrain_gd"
    cfg = LangConfig()
    if enc != "holobrain_gd" or getattr(args, "spawn_pool", None) not in (None, "current") \
            or getattr(args, "qwen_attn", None) not in (None, "sdpa"):
        cfg = dataclasses.replace(cfg, encoder=enc, text_backend="holobrain_qwen" if enc == "holobrain_qwen"
                                  else cfg.text_backend, spawn_pool=getattr(args, "spawn_pool", None) or "current",
                                  qwen_attn=getattr(args, "qwen_attn", None) or "sdpa")
    if getattr(args, "perturb_prob", None) is not None:
        # the nominal's random joint kicks (CrossSafe perturb_prob, 0.05 by default; per tick p = perturb_prob x
        # 10/60). T8 data uses 0.001 (Yuxuan, 2026-11-15: at 0.05 a kick while the arm holds the tray tipped both
        # items in slow drags). In collect_hash, so the shard records it
        cfg = dataclasses.replace(cfg, perturb_prob=float(args.perturb_prob))
    if getattr(args, "behaviour_mix", None):
        # one task's nominal-behaviour mix (config.behaviour_risk keys may carry drag knobs), e.g. the T8 recovery
        # warmup (2026-11-16). In collect_hash, so the shard records it
        tmix = json.loads(args.behaviour_mix)
        cfg = dataclasses.replace(cfg, behaviour_mix={**cfg.behaviour_mix, **tmix})
    frames = getattr(args, "frames", None)
    frames = cfg.qwen if frames is None else bool(frames)
    if frames and not cfg.qwen:
        raise SystemExit("--frames is a V6 (--encoder holobrain_qwen) sidecar")
    rp = None
    legacy = False
    if replay:
        rp = replay_plan(Path(replay), kind, args.embodiment, tasks)
        src = rp["meta"]
        for name, got in (("shard", args.shard), ("n_shards", args.n_shards),
                          ("episodes_per_task", args.episodes_per_task)):
            if got is not None and got != src.get(name):
                raise SystemExit(f"--replay: --{name.replace('_', '-')} {got}, the replayed shard has {src.get(name)}")
        if args.no_masks and src.get("masks"):
            raise SystemExit("--replay: --no-masks, but the replayed shard has masks (a replay keeps its setting)")
        # its task list too: the StepBuffer's task_id is the task's index in it
        cfg = dataclasses.replace(cfg, task_choices=tuple(rp["cfg"].task_choices))
        drift = config_drift(rp["cfg"], cfg)
        legacy = legacy_replay(rp["cfg"], cfg)
        if legacy:
            drift = [k for k in drift if k not in ("bank_hash", "spawn_pool")]
        if drift:
            raise SystemExit(f"--replay: today's collection config differs from {Path(replay).resolve()}'s in {drift}: "
                             "the episodes would not be the recorded ones")
        shard, n_shards, g = src["shard"], src["n_shards"], src["g"]
        offset, ept, masks = src["episode_offset"], src["episodes_per_task"], bool(src.get("masks"))
        slots = rp["slots"]
    else:
        if args.shard is None or args.n_shards is None:
            raise SystemExit("--shard and --n-shards are required (unless --replay)")
        if tasks is not None:
            cfg = dataclasses.replace(cfg, task_choices=tasks)
        shard, n_shards, masks = args.shard, args.n_shards, not args.no_masks
        ept = args.episodes_per_task if args.episodes_per_task is not None else (30 if kind == "warmup" else 5)
        g = shard_g(args.embodiment, shard, n_shards) + int(getattr(args, "g_base", 0) or 0)
        stream = getattr(args, "stream", "default")
        offset = warmup_offset(g) if kind == "warmup" else valg_offset(g) if stream == "grounding" else val_offset(g)
        slots = shard_schedule(cfg.task_choices, args.embodiment, shard, n_shards, ept, g)
    v2 = int(getattr(args, "lang_version", 1) or 1) >= 2
    if v2:
        cfg = dataclasses.replace(cfg, lang_version=2)
    if int(getattr(args, "agent_view", 0) or 0) > 0:   # the VLM labeler's frames (in collect_hash, replay-free)
        cfg = dataclasses.replace(cfg, agent_view_every=int(args.agent_view))
    meta = {"kind": kind, "embodiment": args.embodiment, "shard": shard, "n_shards": n_shards,
            "g": g, "episode_offset": offset, "episodes_per_task": ept,
            "slots": [list(x) for x in slots], "config_hash": cfg.hash(), "collect_hash": collect_hash(cfg),
            "bank_hash": cfg.bank_hash, "masks": masks, "h_code": h_code_hash(cfg.task_choices),
            "h_code_v2": h_code_hash_v2(cfg.task_choices)}
    if v2:
        meta["lang_version"] = 2
    if cfg.qwen:   # V6 (qwen_impl.md 4): what the extra sidecars hold
        from .qwen_encoder import QWEN_CKPT, HoloBrainQwenEncoder as Q, checkpoint_id

        meta["encoder"] = cfg.encoder
        meta["kv"] = {"prefix_len": Q.PREFIX_LEN, "rope_next": Q.ROPE_NEXT,
                      "ckpt_id": checkpoint_id(Path(QWEN_CKPT) / "model.safetensors"), "attn": cfg.qwen_attn,
                      "encoder_version": Q.ENCODER_VERSION}
        meta["frames"] = bool(frames)
    if cfg.spawn_pool != "current":
        meta["spawn_pool"] = cfg.spawn_pool
    if legacy:
        meta["replay_bank_hash"] = rp["cfg"].bank_hash
    if rp is not None:
        meta["replay_of"] = str(Path(replay).resolve())
        meta["replay_slots"] = rp["src_slots"]
    return cfg, meta, slots, offset, masks


def cmd_shard(args, kind: str) -> int:
    cfg, meta, slots, offset, with_masks = plan_shard(args, kind)
    out = Path(args.out).resolve()
    if meta.get("replay_of") == str(out):   # before the writer lock, which writes a file into --out
        raise SystemExit(f"--out {out} is the replayed shard itself: a replay is written to a new root")
    g = meta["g"]
    v2 = int(meta.get("lang_version", 1)) >= 2
    kv_dims = ({"ckpt_id": meta["kv"]["ckpt_id"], "attn": meta["kv"]["attn"],
                "encoder_version": int(meta["kv"].get("encoder_version", 1))} if cfg.qwen else None)
    capacity = int(len(slots) * cfg.max_steps_per_episode * 1.1) + 16
    # one writer per shard for this process's lifetime (review O1): a second process (a duplicated claim)
    # waits out a crashed writer's lock, then leaves the shard to the live one (exit 75: resumable)
    from .locks import WriterLock

    try:
        writer = WriterLock(out, wait_s=200.0)
    except WriterLock.LockHeld as exc:
        print(f"[collect] {exc}: leaving this shard to that process (exit 75)", flush=True)
        return 75
    if (out / "shard.json").exists():
        # a finished shard is done whatever code wrote it (checked first: older shard.json lack h_code)
        done_summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        if done_summary.get("complete"):
            if done_summary.get("replay_mismatch"):
                print(f"[collect] ERROR: {out} is complete but differs from the replayed episodes: "
                      f"{done_summary['replay_mismatch']}", flush=True)
                return 1
            print(f"[collect] {out} is complete")
            return 0
        old = json.loads((out / "shard.json").read_text())
        diff = shard_meta_diff(old, meta, out)
        if diff:
            raise SystemExit(f"{out}: shard.json differs from this command in {diff}"
                             + (" (written under another h definition)" if {"h_code", "h_code_v2"} & set(diff) else "")
                             + "; refusing to resume")
        from main.item_envs.item_buffer import ItemSidecarBuffer
        from main.train.buffer import StepBuffer

        buf = StepBuffer(out / "buffer")
        sidecar = ItemSidecarBuffer(out / "item_buffer")
        neck = open_neck(out, buf) if v2 else None
        extras = open_extras(out, buf, kv_dims=kv_dims, frames=bool(meta.get("frames"))) if cfg.qwen else {}
        done, attempt, rows = resume_point(out, buf, sidecar, neck, *extras.values())
        print(f"[collect] resume {out.name}: {done}/{len(slots)} slots done, attempt {attempt}, rows {rows}")
    else:
        _check_fresh(out)
        out.mkdir(parents=True, exist_ok=True)
        if len(slots) * 5 + 5 > cfg.max_inits_per_process:
            print(f"[collect] WARNING: {len(slots)} slots may exceed {cfg.max_inits_per_process} inits with retries")
        cfg.save(out / "config.json")
        (out / "shard.json").write_text(json.dumps(meta, indent=1))
        if v2:
            buf, sidecar, neck = make_pair(out, cfg, capacity, neck=True)
        else:
            (buf, sidecar), neck = make_pair(out, cfg, capacity), None
        extras = open_extras(out, buf, kv_dims=kv_dims, frames=bool(meta.get("frames"))) if cfg.qwen else {}
        done, attempt = 0, 0
    qwen = None
    if cfg.qwen:   # built once per process
        from .qwen_encoder import HoloBrainQwenEncoder

        qwen = HoloBrainQwenEncoder(device="cuda", attn=cfg.qwen_attn)
        if qwen.ckpt_id != kv_dims["ckpt_id"]:
            raise SystemExit(f"HoloBrain-Qwen checkpoint {qwen.ckpt_id}, the shard records {kv_dims['ckpt_id']}")
    deadline = stage_deadline(args.time_budget)
    t0 = time.time()
    try:
        summary = run_episodes(cfg, out, slots[done:], offset, buf, sidecar, video_every=args.video_every,
                               log_prefix=f"[g{g}] ", slot_offset=done, attempt0=attempt,
                               with_masks=with_masks, deadline=deadline, writer=writer, neck_buf=neck,
                               qwen=qwen, kv_buf=extras.get("_kv"), frames_buf=extras.get("_frames"))
        writer.check()
    except WriterLock.LockLost as exc:
        print(f"[collect] {exc} (exit 75)", flush=True)
        return 75
    summary["wall_total"] = time.time() - t0
    lines = read_lines(out / "lang_episodes.jsonl")
    summary["complete"] = len(lines) >= len(slots)
    if "replay_of" in meta and summary["complete"]:
        summary["replay_mismatch"] = replay_mismatch(lines, slots)   # [] = every slot holds its recorded episode
        if "replay_bank_hash" in meta or cfg.qwen:   # V6: the scenes too (the old pool rebuilt, another encoder)
            summary["replay_mismatch"] += replay_layout_mismatch(out, Path(meta["replay_of"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[collect] {'done' if summary['complete'] else 'paused'}: {summary}")
    if summary["inits"] >= cfg.max_inits_per_process:
        print("[collect] ERROR: shard reached the init limit — delete it and collect with more shards")
        return 3
    if summary.get("replay_mismatch"):
        print(f"[collect] ERROR: the replay differs from the recorded episodes: {summary['replay_mismatch']}",
              flush=True)
        return 1
    return 0 if summary["complete"] else 75


def cmd_recon(args) -> int:
    """Phase 0: `per_task` nominal episodes per task for one embodiment into a
    scratch root (round-robin T1 hosts), for the parity / timing checks."""
    cfg = LangConfig()
    tasks = task_list(args.tasks)
    out = Path(args.out).resolve()
    _check_fresh(out)
    out.mkdir(parents=True, exist_ok=True)
    slots = []
    for k in range(args.per_task):
        for t in tasks:
            slots.append((t, args.embodiment, CANON_T1_HOSTS[k % 5] if t == "selective_avoid" else None))
    cfg.save(out / "config.json")
    buf, sidecar = make_pair(out, cfg, int(len(slots) * cfg.max_steps_per_episode * 1.1) + 16)
    # scratch stream (disjoint from Section 4.2); one process per task gets its own range
    offset = 900_000_000 + CANON_EMBODIMENTS.index(args.embodiment) * 1_000_000
    if len(tasks) == 1:
        offset += CANON_TASKS.index(tasks[0]) * 10_000
    summary = run_episodes(cfg, out, slots, offset, buf, sidecar, video_every=args.video_every)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[recon] done: {summary}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for kind in ("warmup", "validation"):
        p = sub.add_parser(kind)
        p.add_argument("--out", required=True)
        p.add_argument("--embodiment", required=True, choices=CANON_EMBODIMENTS)
        p.add_argument("--shard", type=int, default=None, help="required unless --replay (then the replayed shard's)")
        p.add_argument("--n-shards", type=int, default=None, help="required unless --replay")
        p.add_argument("--episodes-per-task", type=int, default=None,
                       help=f"default {30 if kind == 'warmup' else 5} (with --replay: the replayed shard's)")
        p.add_argument("--tasks", default="all", help="comma list (all = TRAIN_TASKS, e.g. all,tray_slide); with --replay, "
                       "the recorded episodes to replay")
        p.add_argument("--video-every", type=int, default=0)
        p.add_argument("--agent-view", type=int, default=0, metavar="N",
                       help="save the agent-view frame every N ticks for the VLM labeler (0 = off)")
        p.add_argument("--no-masks", action="store_true", help="skip per-tick segmentation masks (warmup)")
        p.add_argument("--time-budget", type=float, default=None, help="seconds; then pause (exit 75)")
        p.add_argument("--lang-version", type=int, choices=(1, 2), default=1,
                       help="2: a version-2 shard, with the neck_buffer/ sidecar the frozen fusion reads (V5)")
        p.add_argument("--encoder", choices=("holobrain_gd", "holobrain_qwen"), default="holobrain_gd",
                       help="the frozen front end (V6: holobrain_qwen adds kv_buffer/ and, by default, frames/)")
        p.add_argument("--frames", action=argparse.BooleanOptionalAction, default=None,
                       help="the frames/ sidecar with the native renders (default: on under holobrain_qwen)")
        p.add_argument("--spawn-pool", choices=("current", "pre_1012", "heldout"), default="current",
                       help="pre_1012: the T1 object pool before 2026-10-12 (replays of shards collected before it); "
                            "heldout: the OOD scenes (held-out models and colours only, 2026-10-22)")
        p.add_argument("--qwen-attn", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
        p.add_argument("--replay", default=None, metavar="SHARD",
                       help="collect a finished shard's recorded episodes again (task, host, episode id; all "
                            "statuses), one build each, with its masks setting; the new shard's slots and ids must "
                            "equal the recorded ones and every slot must keep rows (else exit 1)")
        # warmup too since 2026-11-13: a tray_slide-only shard gets ids no shard on disk uses (warmup g 0-30 and
        # validation g 0-25 are taken; e.g. warmup --g-base 40, validation --g-base 26)
        p.add_argument("--g-base", type=int, default=0, help="added to the shard's g (its episode-id offset)")
        p.add_argument("--perturb-prob", type=float, default=None,
                       help="the nominal's random-kick probability (default: the config's, 0.05)")
        p.add_argument("--behaviour-mix", default=None, metavar="JSON",
                       help='{task: {behaviour[|knob=value...]: weight}} replacing those tasks\' mix, e.g. '
                            '\'{"tray_slide": {"recover_glass": 1, "recover_glass|slow_at=0.6|change_dist=0.05": 1}}\'')
        if kind == "validation":
            p.add_argument("--stream", choices=("default", "grounding"), default="default",
                           help="grounding: the 20-per-task ur5-wsg validation ids (config.valg_offset; T1 / T7 at "
                                "g 0-3, round 3 E1; T2-T6 at g 4-8, decision 4)")
    p = sub.add_parser("recon")
    p.add_argument("--out", required=True)
    p.add_argument("--embodiment", required=True, choices=CANON_EMBODIMENTS)
    p.add_argument("--per-task", type=int, default=2)
    p.add_argument("--tasks", default="all")
    p.add_argument("--video-every", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("root")
    p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd in ("warmup", "validation"):
        return cmd_shard(args, args.cmd)
    if args.cmd == "recon":
        return cmd_recon(args)
    if args.cmd == "report":
        from .collect_report import main as report_main

        return report_main([args.root] + (["--out", args.out] if args.out else []))
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InitLimit as exc:
        print(f"[collect] init limit: {exc}")
        raise SystemExit(75)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
