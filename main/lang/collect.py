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
    CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS, LangConfig, check_stream, shard_g,
    val_offset, warmup_offset,
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
    when 5 divides episodes_per_task). -> [(task, emb, host)]."""
    if episodes_per_task % n_shards:
        raise ValueError(f"n_shards={n_shards} must divide episodes_per_task={episodes_per_task}")
    cycles = episodes_per_task // n_shards
    rng = np.random.default_rng(g)
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


def make_pair(out_root: Path, cfg: LangConfig, capacity: int):
    """A fresh StepBuffer + lockstep sidecar under out_root (never reused)."""
    from main.item_envs.item_buffer import ItemSidecarBuffer
    from main.item_envs.masks import token_grid
    from main.train.buffer import StepBuffer

    buf = StepBuffer(out_root / "buffer", capacity=int(capacity), header=cfg.to_dict())
    sidecar = ItemSidecarBuffer.for_step_buffer(buf, n_tok=token_grid()[1])
    return buf, sidecar


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
        out["placements"] = [{k: p.get(k) for k in ("item_id", "cls", "xyz", "degraded_from", "lateral")}
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
                      named_harm_frac: float = 0.5) -> dict | None:
    """Section 7.1 item 4: a SAMPLEABLE harm-designated rule with probability
    named_harm_frac, else a sampleable benign-designated one; the other kind
    when the wanted one does not exist; None when neither exists."""
    pool = {"harm": [], "benign": []}
    for r in record["rules"]:
        d = desig.get(r["rule_id"])
        if r.get("sampleable") and d in pool:
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
                 round_metrics=None, writer=None) -> dict:
    """Run one episode per slot (task, emb, host) into (buf, sidecar).

    Episode i of the stream has seed = episode_id = episode_offset + attempt
    (attempt counts every env build, retries included). Returns a summary
    dict; appends one line per slot to out_root/lang_episodes.jsonl.
    models: (policy_enc, actor, critics) for filter episodes (Phase 4),
    None = nominal only. slot_offset: index of slots[0] in the full schedule
    (resume). writer: the data root's WriterLock, checked about once a second during an episode and before
    every flush and line (round 2 B1). with_masks=False skips the per-tick segmentation masks (zeros in
    the sidecar; only validation data feeds ref_mask). deadline (unix time):
    no new slot starts when the mean episode wall time would pass it
    (summary["stopped"] = True; the caller exits 75 and resumes later).
    """
    import torch

    from main.item_envs import language
    from main.item_envs.controller import RiskTickController
    from main.item_envs.margin_hook import wire_item_episode
    from main.train.collect import CKPT_DIR, make_kinematics

    from .rollout import LangRolloutController
    from .scene_roles import all_designations, in_way, mask_region_rows

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
    for slot_i, (task, emb, host) in enumerate(slots):
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
        for _ in range(max_retries):
            if inits["n"] >= cfg.max_inits_per_process:
                raise InitLimit(f"{inits['n']} environment inits")
            seed = check_stream(episode_offset, attempt)
            attempt += 1
            inits["n"] += 1
            cfg_ep = dataclasses.replace(
                cfg, task=task, embodiment=emb, item_host=host, item_active_rule=None,
                item_out_root=str(out_root / "_runner"),
            )
            try:
                cand = LangRolloutController(
                    cfg_ep, seed, mode="collect", encoder=encoder, kin=kins[emb],
                    policy_enc=models[0] if (models and use_filter) else None,
                    actor=models[1] if (models and use_filter) else None,
                    critics=models[2] if (models and use_filter) else None,
                    filter_active=use_filter, buf=buf, episode_id=seed,
                    perturb_prob=cfg.perturb_prob, rng=np.random.RandomState(seed),
                    max_steps=10 * int(cfg.max_steps_per_episode),
                    controller_cls=RiskTickController, text_store=text_store,
                )
            except Exception as exc:
                errors.append(f"seed {seed}: build {type(exc).__name__}: {exc}"[:300])
                print(f"{log_prefix}[collect] {task}/{emb} seed {seed}: build failed ({exc}); new seed")
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
        primary = runner._primary_rule_id(env.task)
        info = getattr(env.task, "episode_info", None) or {}
        layout = getattr(env.task, "layout_info", None) or {}
        region_rows = mask_region_rows(runner)
        smeta = scene_meta(env, runner)   # before run(): run() closes the env (actor poses gone)
        named, sentence, cf_sentence = None, None, None
        if use_filter:
            rng_ep = np.random.default_rng(seed)
            named = choose_named_rule(record, desig, rng_ep, cfg.named_harm_frac)
            if named is None:
                ro.filter_active = False
            else:
                runner.active_rule_id = named["rule_id"]
                sentence = language.sample_sentence(named, record, rng_ep, "train")
                # counterfactual: a sampleable rule of the other designation (Q_cf on the panel)
                other = "benign" if desig.get(named["rule_id"]) == "harm" else "harm"
                cands = [r for r in record["rules"] if r.get("sampleable") and desig.get(r["rule_id"]) == other]
                if cands:
                    cf_sentence = language.sample_sentence(cands[int(rng_ep.integers(len(cands)))], record, rng_ep, "train")
                ro.set_instruction(sentence, cf_sentence, rule_id=named["rule_id"])
        n_before = len(buf)
        ro.writer = writer
        wire_item_episode(ro, sidecar)
        if not with_masks:
            ro.h_fn.with_masks = False
        if video_every and summary["episodes"] % int(video_every) == 0:
            from .video import LangPanelVideoWriter, video_name

            vdir = out_root / "videos"
            vdir.mkdir(parents=True, exist_ok=True)
            kind = desig.get(named["rule_id"]) if named else "nominal"
            ro.video = LangPanelVideoWriter(vdir / video_name(task, runner.host if task == "selective_avoid" else None,
                                                              emb, seed, kind), ro, named_rule=named, sentence=sentence)
        if use_filter:
            ctx = torch.no_grad()
            ctx.__enter__()
        try:
            trace = ro.run()
        finally:
            if use_filter:
                ctx.__exit__(None, None, None)
            if ro.video is not None:
                ro.video.close()
        if writer is not None:
            writer.check()
        ro.buf.end_episode()  # lockstep teardown: raises on a pairing bug (stops the process)
        buf.flush()
        sidecar.flush()
        rows = np.arange(n_before, len(buf)) % buf.capacity
        violated, hmin = _rule_outcomes(sidecar, rows, len(record["rules"]))
        line = {
            "episode_id": seed, "task": task, "host": runner.host if task == "selective_avoid" else None,
            "embodiment": emb, "seed": seed, "filter": bool(use_filter and named is not None),
            "named_rule_id": named["rule_id"] if named else None,
            "named_consequence": named["consequence"] if named else None,
            "designation": desig.get(named["rule_id"]) if named else None,
            "sentence": sentence, "cf_sentence": cf_sentence, "designations": desig,
            "in_way": sorted(roles[0]), "unclear": sorted(roles[1]),
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
        if round_metrics is not None and line["filter"]:
            from .metrics import LangRoundMetrics

            rm, r_idx = round_metrics
            rm.add(LangRoundMetrics.episode_row(r_idx, line, trace))
        if writer is not None:
            writer.check()
        with open(jsonl, "a") as f:
            f.write(json.dumps(_jsonable(line)) + "\n")
        summary["episodes"] += 1
        summary["status"][line["status"]] = summary["status"].get(line["status"], 0) + 1
        summary["wall"] += line["wall_s"]
        print(f"{log_prefix}[collect] {task}{'/' + str(line['host']) if line['host'] else ''}/{emb} "
              f"seed {seed}: {'FILTER' if line['filter'] else 'nominal'} "
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
    if bad:
        jsonl.write_text("".join(json.dumps(d) + "\n" for d in out))
    return out


def resume_point(out: Path, buf, sidecar) -> tuple[int, int, int]:
    """(slots done, attempt, rows) of an interrupted shard; both rings are
    rolled back to the end of the last recorded episode (an episode killed
    mid-run, or flushed without its line, is discarded and collected again
    with the same seed, i.e. the same scene)."""
    lines = read_lines(out / "lang_episodes.jsonl")
    done = len(lines)
    attempt = int(lines[-1]["attempt_next"]) if lines else 0
    rows = 0
    for d in lines:
        if not d.get("skipped"):
            rows = int(d["rows"][1])
    if len(buf) < rows or len(sidecar) < rows:
        raise RuntimeError(f"{out}: rings ({len(buf)}, {len(sidecar)}) shorter than the recorded rows {rows}")
    # every recorded episode's END rows must hold that episode (both buffers) with ticks 0 and n-1: this
    # catches two writers and most copies made mid-write (review B4), but not every torn array (only the end
    # rows are read). The real guard is copying finished shards only (scripts/lang/bridges/README.md)
    eb, es, ts = buf.arrays.get("episode_id"), sidecar.arrays["episode_id"], sidecar.arrays["t_in_ep"]
    for d in lines:
        if d.get("skipped"):
            continue
        r0, r1 = (int(x) for x in d["rows"])
        eid = int(d["episode_id"])
        ends_ok = (r1 > r0 and int(es[r0]) == eid and int(es[r1 - 1]) == eid and int(ts[r0]) == 0
                   and int(ts[r1 - 1]) == r1 - r0 - 1 and (eb is None or (int(eb[r0]) == eid and int(eb[r1 - 1]) == eid)))
        if not ends_ok:
            raise RuntimeError(f"{out}: episode {eid} rows [{r0}, {r1}) do not hold that episode (a partial copy or "
                               f"two writers?); refusing to resume: check with scripts/lang/check_shards.py")
    if (len(buf), len(sidecar)) != (rows, rows):
        print(f"[collect] resume: discarding rows {rows}..{max(len(buf), len(sidecar))} of an interrupted episode")
        buf.discard_to(rows)
        sidecar.discard_to(rows)
        buf.flush()
        sidecar.flush()
    return done, attempt, rows


def h_code_hash(tasks) -> str:
    """Hash of what defines h for these tasks: margins.py, rules.py and the tasks' yaml files. Recorded in
    shard.json: a shard written under another h definition (e.g. the 1 mm soft-min bead margin of
    2026-10-07) refuses to resume (round 2 B2)."""
    import hashlib

    from . import REPO

    root = REPO / "envs/main/item_envs"
    files = [root / "margins.py", root / "rules.py"]
    for t in sorted(tasks):
        files += sorted((root / "configs/tasks").glob(f"*_{t}.yaml"))
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def cmd_shard(args, kind: str) -> int:
    cfg = LangConfig()
    if args.tasks != "all":
        cfg = dataclasses.replace(cfg, task_choices=tuple(args.tasks.split(",")))
    out = Path(args.out).resolve()
    g = shard_g(args.embodiment, args.shard, args.n_shards)
    offset = warmup_offset(g) if kind == "warmup" else val_offset(g)
    slots = shard_schedule(cfg.task_choices, args.embodiment, args.shard, args.n_shards, args.episodes_per_task, g)
    meta = {"kind": kind, "embodiment": args.embodiment, "shard": args.shard, "n_shards": args.n_shards,
            "g": g, "episode_offset": offset, "episodes_per_task": args.episodes_per_task,
            "slots": [list(x) for x in slots], "config_hash": cfg.hash(), "bank_hash": cfg.bank_hash,
            "masks": not args.no_masks, "h_code": h_code_hash(cfg.task_choices)}
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
        if (out / "summary.json").exists() and json.loads((out / "summary.json").read_text()).get("complete"):
            print(f"[collect] {out} is complete")
            return 0
        old = json.loads((out / "shard.json").read_text())
        diff = sorted(k for k in meta if old.get(k) != meta[k])
        if diff:
            raise SystemExit(f"{out}: shard.json differs from this command in {diff}"
                             + (" (written under another h definition)" if "h_code" in diff else "")
                             + "; refusing to resume")
        from main.item_envs.item_buffer import ItemSidecarBuffer
        from main.train.buffer import StepBuffer

        buf = StepBuffer(out / "buffer")
        sidecar = ItemSidecarBuffer(out / "item_buffer")
        done, attempt, rows = resume_point(out, buf, sidecar)
        print(f"[collect] resume {out.name}: {done}/{len(slots)} slots done, attempt {attempt}, rows {rows}")
    else:
        _check_fresh(out)
        out.mkdir(parents=True, exist_ok=True)
        if len(slots) * 5 + 5 > cfg.max_inits_per_process:
            print(f"[collect] WARNING: {len(slots)} slots may exceed {cfg.max_inits_per_process} inits with retries")
        cfg.save(out / "config.json")
        (out / "shard.json").write_text(json.dumps(meta, indent=1))
        buf, sidecar = make_pair(out, cfg, capacity)
        done, attempt = 0, 0
    deadline = stage_deadline(args.time_budget)
    t0 = time.time()
    try:
        summary = run_episodes(cfg, out, slots[done:], offset, buf, sidecar, video_every=args.video_every,
                               log_prefix=f"[g{g}] ", slot_offset=done, attempt0=attempt,
                               with_masks=not args.no_masks, deadline=deadline, writer=writer)
        writer.check()
    except WriterLock.LockLost as exc:
        print(f"[collect] {exc} (exit 75)", flush=True)
        return 75
    summary["wall_total"] = time.time() - t0
    lines = read_lines(out / "lang_episodes.jsonl")
    summary["complete"] = len(lines) >= len(slots)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[collect] {'done' if summary['complete'] else 'paused'}: {summary}")
    if summary["inits"] >= cfg.max_inits_per_process:
        print("[collect] ERROR: shard reached the init limit — delete it and collect with more shards")
        return 3
    return 0 if summary["complete"] else 75


def cmd_recon(args) -> int:
    """Phase 0: `per_task` nominal episodes per task for one embodiment into a
    scratch root (round-robin T1 hosts), for the parity / timing checks."""
    cfg = LangConfig()
    tasks = CANON_TASKS if args.tasks == "all" else tuple(args.tasks.split(","))
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
        p.add_argument("--shard", type=int, required=True)
        p.add_argument("--n-shards", type=int, required=True)
        p.add_argument("--episodes-per-task", type=int, default=30 if kind == "warmup" else 5)
        p.add_argument("--tasks", default="all")
        p.add_argument("--video-every", type=int, default=0)
        p.add_argument("--no-masks", action="store_true", help="skip per-tick segmentation masks (warmup)")
        p.add_argument("--time-budget", type=float, default=None, help="seconds; then pause (exit 75)")
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
