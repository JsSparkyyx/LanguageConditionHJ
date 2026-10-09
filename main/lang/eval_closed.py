"""Closed-loop evaluation grid (VANILLA_PLAN 9.2).

    python -m main.lang.eval_closed --run runs/lang/<name> --ckpt checkpoint_final.pt \\
        --scenes-per-combo 20 --axes train,heldout_template [--video-every 1]

cuRobo makes the nominal motion differ between runs of one seed, so whether
the named item "would have been harmed" cannot be read from one separate
nominal run. The named rule follows the in-the-way designation (Section 3.5,
deterministic per seed); baseline rates come from repeated nominal runs.

Per scene (task, embodiment, scene index; seed = eval_seed(...); T1 host =
t1_hosts[scene_idx mod 5], passed explicitly):
1. nominal: 2 runs without a filter (``set_global_seeds(seed)`` before each),
   every rule's outcome and designation, the seed actually used (a failed
   build is redrawn with attempt + 1, at most 7 times). Cached under
   ``runs/lang/_nominal_cache/<key>.json`` (task, host, embodiment, seed,
   max_steps_per_episode, env code hash, bank hash), shared by every
   checkpoint. The nominal runs use the env's standalone runner without
   rendering (same controller and physics; no model reads the cameras);
2. named rules per axis: a harm-designated and a benign-designated rule
   eligible for the axis (Section 9.1), each chosen by a hash of (seed, axis);
3. one filtered episode per (axis, named rule) on the used seed, one sentence
   chosen by a hash of (seed, axis); Q_cf with the other named rule's
   sentence when both exist. The rule table and the designations must match
   the nominal runs' (else an error row).

Children run one embodiment each (``--child``), exit after
max_inits_per_process env inits (exit 75) and are restarted by the parent;
three crashes on one item -> an error row. Rows resume at (task, scene, axis,
kind) level from ``episodes_emb-<e>.jsonl``; the parent merges into
``summary.json`` and ``tables.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import REPO, ensure_item_path

ensure_item_path()

from .config import CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS, LangConfig, eval_seed  # noqa: E402

NOMINAL_RUNS = 2
MAX_REDRAWS = 7
EXIT_INIT = 75


def set_global_seeds(seed: int) -> None:
    """main.train.eval_ckpt.set_global_seeds."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


HASHED_CODE = ("envs/main/item_envs/**/*.py", "envs/main/item_envs/**/*.yaml", "envs/robotwin_tasks/*.py",
               "main/envs/*.py", "main/lang/scene_roles.py")
# what a FILTERED run also depends on (round 2 C7): the language controller, the metrics, the model code
FILTER_CODE = ("main/lang/rollout.py", "main/lang/eval_closed.py", "main/lang/network.py", "main/lang/text.py",
               "main/lang/collect.py", "main/train/rollout.py", "main/train/filter.py", "main/network/*.py",
               # round 3, B5: the environment / config code a filtered run also executes
               "main/train/collect.py", "main/train/config.py", "main/train/hfunc.py", "main/lang/config.py",
               "main/lang/train.py")


def _hash_files(patterns) -> str:
    h = hashlib.md5()
    for pat in patterns:
        for p in sorted(REPO.glob(pat)):
            if "__pycache__" not in p.parts:
                h.update(str(p.relative_to(REPO)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:12]


def filter_code_hash() -> str:
    """The env hash plus the code a filtered run executes: filter rows carry it and a resume refuses rows
    written under another one (round 2 C7: rows from before fixes such as O5 or M2 would otherwise stay)."""
    return _hash_files(HASHED_CODE + FILTER_CODE)


def env_code_hash() -> str:
    """Everything that shapes a nominal run, its h and its designations (the nominal cache key; every result
    row carries it, and a resume refuses rows written under another hash; review M4)."""
    h = hashlib.md5()
    for pat in HASHED_CODE:
        for p in sorted(REPO.glob(pat)):
            if "__pycache__" not in p.parts:
                h.update(str(p.relative_to(REPO)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:12]


def hsel(items, key):
    """Deterministic choice by a hash of key."""
    if not items:
        return None
    return items[int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) % len(items)]


def pick_rules(rec: dict, designations: dict, seed: int, axes, is_eligible) -> dict:
    """{axis: {kind: rule or None}}: one rule order per (scene, kind). The first rule eligible on EVERY
    requested axis is named on all of them, so those axes differ in the sentence only (round 2 C8). When no
    rule is eligible everywhere (heldout_word's set is a subset of train's), each axis falls back to its own
    first eligible rule, and then the axes do differ in the rule."""
    out = {ax: {} for ax in axes}
    for kind in ("harm", "benign"):
        order = sorted((r for r in rec["rules"] if designations.get(r["rule_id"]) == kind),
                       key=lambda r: hashlib.md5(str((seed, kind, r["rule_id"])).encode()).hexdigest())
        common = next((r for r in order if all(is_eligible(r, ax) for ax in axes)), None)
        for ax in axes:
            out[ax][kind] = common if common is not None else next((r for r in order if is_eligible(r, ax)), None)
    return out


# ---------------------------------------------------------------------------
# nominal runs (cached, shared by every checkpoint)
# ---------------------------------------------------------------------------


def trace_violations(runner) -> tuple[dict, dict]:
    """(violated, first_violation_t) per rule from the run's trace.npz with h = margins.rule_h (bead rules
    by their violating-bead count): the same h < 0 test the filtered runs use (review B2). meta.json's
    violations judge margin < 0 instead."""
    from main.item_envs.margins import rule_h

    p = Path(runner.out_dir) / "trace.npz"
    if not p.exists():
        return {}, {}
    z = np.load(p)
    t, m = np.asarray(z["t"], float), np.asarray(z["margins"], float)
    c = np.asarray(z["violation_counts"]) if "violation_counts" in z.files else np.full(m.shape, -1)
    viol, first = {}, {}
    for j, rule in enumerate(runner.rules):
        if j >= m.shape[1]:
            continue
        h = np.array([rule_h(rule, m[k, j], int(c[k, j])) for k in range(m.shape[0])])
        neg = h < 0
        viol[rule.rule_id] = bool(neg.any())
        first[rule.rule_id] = float(t[int(np.argmax(neg))]) if neg.any() else None
    return viol, first



def nominal_scene(task: str, emb: str, scene_idx: int, cfg: LangConfig, cache_dir: Path, inits: dict,
                  code_hash: str) -> dict:
    from main.item_envs import language
    from main.item_envs.runner import ItemEpisodeRunner

    from .scene_roles import all_designations, harm_reasons, in_way, in_way_reasons

    host = CANON_T1_HOSTS[scene_idx % 5] if task == "selective_avoid" else None
    seed0 = eval_seed(task, emb, scene_idx, 0, host)
    key = f"{task}_{host or '-'}_{emb}_{seed0}_{cfg.max_steps_per_episode}_{code_hash}_{cfg.bank_hash}"
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {"task": task, "host": host, "embodiment": emb, "scene_idx": scene_idx, "seed0": seed0, "key": key,
           "runs": [], "errors": []}
    for attempt in range(MAX_REDRAWS):
        seed = eval_seed(task, emb, scene_idx, attempt, host)
        runs, record, roles, desig, ok = [], None, None, None, True
        for k in range(NOMINAL_RUNS):
            set_global_seeds(seed)
            inits["n"] += 1
            runner = ItemEpisodeRunner(task, embodiment=emb, seed=seed, host=host, render="none",
                                       out_root=str(cache_dir / "runs" / f"r{k}"),
                                       max_steps=10 * int(cfg.max_steps_per_episode))
            try:
                runner.setup()
                if not runner.rules:
                    raise RuntimeError("no rules")
                runner.env._item_runner = runner
                rec = language.episode_record(runner, seed)
                rl = in_way(runner.env, rec)
                ds = all_designations(rec, rl)
                rs = harm_reasons(rec, in_way_reasons(runner.env, rec))
            except Exception as exc:
                out["errors"].append(f"seed {seed} run {k}: {type(exc).__name__}: {exc}"[:300])
                runner.close()
                ok = False
                break
            if record is None:
                record, roles, desig, reasons = rec, rl, ds, rs
            elif [r["rule_id"] for r in rec["rules"]] != [r["rule_id"] for r in record["rules"]] or ds != desig:
                out["errors"].append(f"seed {seed}: rule table / designations differ between nominal runs")
            meta, error = runner.run()
            viol, first = trace_violations(runner)
            runs.append({"violated": viol, "first_violation_t": first, "success": bool(meta.task_success),
                         "error": error, "wall_s": meta.wall_time_s,
                         "n_ticks": int(len(np.load(Path(runner.out_dir) / "trace.npz")["t"]))
                         if (Path(runner.out_dir) / "trace.npz").exists() else None})
        if ok:
            out.update({"seed": seed, "attempt": attempt, "runs": runs, "record": record,
                        "in_way": sorted(roles[0]), "unclear": sorted(roles[1]), "designations": desig,
                        "harm_reasons": reasons})
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=str))
    tmp.replace(path)
    return out


# ---------------------------------------------------------------------------
# filtered runs
# ---------------------------------------------------------------------------


class Models:
    def __init__(self, run: Path, ckpt: str, caches):
        import torch

        from .network import LangPolicyEncoder, LangTwinCritic, make_actor
        from .text import TextStore
        from .train import load_policy

        path = run / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
        ck = torch.load(path, map_location="cuda", weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        self.step, self.sha = int(ck.get("step", 0)), ckpt_sha(path)   # round 3, B1
        self.pe = LangPolicyEncoder(self.cfg).cuda().eval()
        load_policy(self.pe, ck)
        self.actor = make_actor(self.cfg).cuda().eval()
        self.actor.load_state_dict(ck["actor"])
        self.critics = LangTwinCritic(self.cfg).cuda().eval()
        self.critics.load_state_dict(ck["critics"])
        # the trainer's extras are read only; the evaluator writes its own file (review O7)
        train_extra = [run / "text_cache_extra.pt"] if (run / "text_cache_extra.pt").exists() else []
        self.text = TextStore(list(caches) + train_extra, self.cfg.text_backend, "cuda",
                              extra_path=run / "text_cache_extra_eval.pt", backend_paths=self.cfg.backend_paths)
        self.step = int(ck.get("step", 0))


def run_metrics(trace: dict, nom: dict, named_id: str, kind: str) -> dict:
    h = np.asarray(trace.get("h_named", []), float)
    t = np.asarray(trace.get("t", []), float)
    Q = np.asarray(trace.get("V", []), float)
    tq = np.asarray(trace.get("V_t", []), float)
    inter = np.asarray(trace.get("intervened", []), bool)
    other = np.asarray(trace.get("h_other_min", []), float)
    qcf = np.asarray(trace.get("Q_cf", []), float)
    hq = np.interp(tq, t, h) if (tq.size and t.size == h.size and h.size) else np.zeros(0)
    first_int = float(tq[np.argmax(inter)]) if inter.any() and tq.size == inter.size else None
    src = np.asarray(trace.get("action_source", []), int)
    nom_ticks = [int(r["n_ticks"]) for r in nom["runs"] if r.get("n_ticks")]
    L = max(nom_ticks) if nom_ticks else None
    m = {
        "status": trace.get("status"), "task_success": bool(trace.get("success", False)),
        "harm_named": bool(h.size and (h < 0).any()), "harm_named_frac": float((h < 0).mean()) if h.size else None,
        "intervention_rate": float(trace.get("intervention_rate", 0.0) or 0.0), "any_intervention": bool(inter.any()),
        "harm_other_any": bool(other.size and (other[np.isfinite(other)] < 0).any()),
        "v_le_h_frac": float((Q <= hq + 0.05).mean()) if Q.size and hq.size == Q.size else None,
        "hq_corr": float(np.corrcoef(hq, Q)[0, 1]) if Q.size > 2 and np.std(Q) > 0 and np.std(hq) > 0 else None,
        "first_intervention_t": first_int, "n_ticks": int(t.size),
        # intervention_rate counts DECISIONS (one engaged decision holds 3 ticks); this counts ticks (review M7)
        "intervened_tick_frac": float((src == 1).mean()) if src.size else None,
        # harm within the nominal runs' duration: a stalled filtered run is not exposed longer (review E6)
        "harm_named_nomdur": bool(h.size and (h[:L] < 0).any()) if L else None,
    }
    fvs = [r["first_violation_t"][named_id] for r in nom["runs"]
           if (r.get("first_violation_t") or {}).get(named_id) is not None]   # a first violation at t = 0.0 counts
    if kind == "harm" and fvs and first_int is not None:
        m["lead_time"] = float(np.mean(fvs)) - first_int
    if qcf.size and Q.size:
        n = int(np.argmax(inter)) if inter.any() else Q.size
        k = min(n, qcf.size, Q.size)
        m["q_cf_gap"] = float(np.mean(qcf[:k] - Q[:k])) if k else None
    return m


def ckpt_sha(path: Path) -> str:
    from .eval_offline import file_sha

    return file_sha(Path(path))


def filter_run(models: Models, cfg: LangConfig, nom: dict, rule: dict, kind: str, axis: str, sentence: str,
               cf_sentence: str | None, inits: dict, encoder, kins, video_dir: Path | None,
               filter_active: bool = True) -> dict:
    """One closed-loop run naming `rule`. filter_active=False: the control (round 3, B5) — the same language
    controller with the filter never engaging, to compare with the standalone runner's nominal runs."""
    import dataclasses

    import torch

    from main.item_envs import language
    from main.item_envs.controller import RiskTickController
    from main.train.collect import CKPT_DIR, make_kinematics

    from .rollout import LangRolloutController, wire_item_eval
    from .scene_roles import all_designations, in_way

    task, emb, host, seed = nom["task"], nom["embodiment"], nom["host"], int(nom["seed"])
    if emb not in kins:
        kins[emb] = make_kinematics(CKPT_DIR, emb)
    cfg_ep = dataclasses.replace(models.cfg, task=task, embodiment=emb, item_host=host, item_active_rule=None,
                                 item_out_root=str(REPO / "runs/lang/_eval_runner"))
    set_global_seeds(seed)
    inits["n"] += 1
    ro = LangRolloutController(cfg_ep, seed, mode="filtered", encoder=encoder, kin=kins[emb],
                               policy_enc=models.pe, actor=models.actor, critics=models.critics,
                               filter_active=bool(filter_active),
                               episode_id=seed, rng=np.random.RandomState(seed),
                               max_steps=10 * int(cfg.max_steps_per_episode), controller_cls=RiskTickController,
                               text_store=models.text)
    runner = ro.env._item_runner
    rec = language.episode_record(runner, seed)
    desig = all_designations(rec, in_way(ro.env, rec))
    if [r["rule_id"] for r in rec["rules"]] != [r["rule_id"] for r in nom["record"]["rules"]] or desig != nom["designations"]:
        ro.env.close()
        return {"error": "rule table or designations differ from the nominal runs"}
    wire_item_eval(ro, rule["rule_id"])
    ro.set_instruction(sentence, cf_sentence, rule_id=rule["rule_id"])
    if video_dir is not None:
        from .video import LangPanelVideoWriter, video_name

        video_dir.mkdir(parents=True, exist_ok=True)
        ro.video = LangPanelVideoWriter(video_dir / video_name(task, host, emb, seed, kind, axis), ro,
                                        named_rule=rule, sentence=sentence)
    with torch.no_grad():
        try:
            trace = ro.run()
        finally:
            if ro.video is not None:
                ro.video.close()
    m = run_metrics(trace, nom, rule["rule_id"], kind)
    m["video"] = str(ro.video.path) if ro.video is not None else None
    m["timing"] = trace.get("timing")
    return m


def child(args) -> int:
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR

    from .eval_offline import eligible
    from .train import caches_of

    from main.item_envs import language

    run = Path(args.run).resolve()
    out_dir = run / "eval_closed" / Path(args.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"emb-{args.embodiment}" + (f"_s{args.shard}" if args.n_shards > 1 else "")
    jsonl = out_dir / f"episodes_{tag}.jsonl"
    code_hash, filter_hash = env_code_hash(), filter_code_hash()
    sha = ckpt_sha(run / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt))
    done = set()
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("code_hash") != code_hash:
                raise SystemExit(f"{jsonl}: rows written under env code hash {d.get('code_hash')}, now {code_hash}: "
                                 f"the env changed; evaluate into a fresh directory (move {out_dir} away)")
            if _is_filter_row(d) and d.get("filter_hash") != filter_hash:
                raise SystemExit(f"{jsonl}: filter rows written under filter code hash {d.get('filter_hash')}, now "
                                 f"{filter_hash}: evaluate into a fresh directory (move {out_dir} away)")
            if _is_filter_row(d) and d.get("ckpt_sha") not in (None, sha):   # a rewritten checkpoint (round 3, B1)
                raise SystemExit(f"{jsonl}: filter rows of checkpoint {d.get('ckpt_sha')}, {args.ckpt} is now {sha}: "
                                 f"the weights changed; evaluate into a fresh directory (move {out_dir} away)")
            done.add(tuple(d["key"]))
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.val or [])
    models = Models(run, args.ckpt, caches)
    cfg = models.cfg
    bank = language.load_bank()
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kins, inits = {}, {"n": 0}
    cache_dir = REPO / "runs/lang/_nominal_cache"
    axes = [] if args.axes in ("", "none") else args.axes.split(",")   # none: nominal + control runs only
    tasks = CANON_TASKS if args.tasks == "all" else tuple(args.tasks.split(","))
    crashes = defaultdict(int)
    cr_path = out_dir / f"crashes_{tag}.json"
    if cr_path.exists():
        crashes.update({tuple(json.loads(k)): v for k, v in json.loads(cr_path.read_text()).items()})

    def write(row):
        with open(jsonl, "a") as f:
            f.write(json.dumps({**row, "code_hash": code_hash, "filter_hash": filter_hash, "ckpt_sha": sha,
                                "ckpt_step": models.step}, default=str) + "\n")

    work = [(task, si) for task in tasks for si in range(int(args.scenes_per_combo))]
    work = [w for k, w in enumerate(work) if k % int(args.n_shards) == int(args.shard)]
    for task, si in work:
        if True:
            nkey = (task, si, "nominal", "-")
            if nkey not in done:
                if inits["n"] + NOMINAL_RUNS + 2 * len(axes) > cfg.max_inits_per_process:
                    return EXIT_INIT
                nom = nominal_scene(task, args.embodiment, si, cfg, cache_dir, inits, code_hash)
                write({"key": list(nkey), "type": "nominal", "task": task, "embodiment": args.embodiment,
                       "scene_idx": si, "host": nom["host"], "seed": nom.get("seed"), "errors": nom["errors"],
                       "runs": nom["runs"], "designations": nom.get("designations")})
                done.add(nkey)
            else:
                nom = nominal_scene(task, args.embodiment, si, cfg, cache_dir, inits, code_hash)
            if "record" not in nom:   # every redraw failed: the scene's items count as missing (review O8)
                for axis in axes:
                    for kind in ("harm", "benign"):
                        key = (task, si, axis, kind)
                        if key not in done:
                            write({"key": list(key), "type": "filter", "task": task, "embodiment": args.embodiment,
                                   "scene_idx": si, "axis": axis, "kind": kind, "missing": True, "reason": "no scene"})
                            done.add(key)
                continue
            rec = nom["record"]
            all_picks = pick_rules(rec, nom["designations"], nom["seed"], axes or ["train"],
                                   lambda r, ax: eligible(r, rec, ax, bank))
            for axis in axes:
                picks = all_picks[axis]
                for kind in ("harm", "benign"):
                    key = (task, si, axis, kind)
                    if key in done:
                        continue
                    rule = picks[kind]
                    base = {"key": list(key), "type": "filter", "task": task, "embodiment": args.embodiment,
                            "scene_idx": si, "host": nom["host"], "seed": nom["seed"], "axis": axis, "kind": kind}
                    if rule is None:
                        write({**base, "missing": True})
                        done.add(key)
                        continue
                    if crashes[key] >= 3:
                        write({**base, "error": "3 crashes"})
                        done.add(key)
                        continue
                    if inits["n"] + 1 > cfg.max_inits_per_process:
                        return EXIT_INIT
                    split = "train" if axis in ("train", "heldout_item", "shuffled") else axis
                    sent = hsel(language.sentences(rule, rec, split, bank), (nom["seed"], axis))
                    other = picks["benign" if kind == "harm" else "harm"]
                    cf = hsel(language.sentences(other, rec, split, bank), (nom["seed"], axis, "cf")) if other else None
                    crashes[key] += 1
                    cr_path.write_text(json.dumps({json.dumps(list(k)): v for k, v in crashes.items()}))
                    vdir = out_dir / "videos" if (args.video_every and si % int(args.video_every) == 0) else None
                    t0 = time.time()
                    try:
                        m = filter_run(models, cfg, nom, rule, kind, axis, sent, cf, inits, encoder, kins, vdir)
                    except Exception as exc:
                        print(f"[eval_closed] {key} crashed: {exc}", flush=True)
                        raise
                    crashes[key] -= 1
                    nom_harm = [bool(r["violated"].get(rule["rule_id"], False)) for r in nom["runs"]]
                    nom_succ = [bool(r["success"]) for r in nom["runs"]]
                    write({**base, "rule_id": rule["rule_id"], "consequence": rule["consequence"],
                           "harm_reason": (nom.get("harm_reasons") or {}).get(rule["rule_id"]),
                           "split": rule.get("split"), "sentence": sent, "cf_sentence": cf,
                           "nominal_violated": nom_harm, "nominal_success": nom_succ, "wall_s": time.time() - t0, **m})
                    done.add(key)
                    print(f"[eval_closed] {task} {args.embodiment} scene {si} {axis}/{kind} {rule['rule_id']} "
                          f"'{sent}': harm {m.get('harm_named')} (nominal {nom_harm}) interv {m.get('intervention_rate')} "
                          f"success {m.get('task_success')} {m.get('error') or ''}", flush=True)
            # control (round 3, B5): the language controller with the filter off on the first scenes; it should
            # reproduce the standalone runner's nominal outcomes up to cuRobo's run-to-run variation
            ckey = (task, si, "control", "harm")
            ax0 = axes[0] if axes else "train"
            rule = all_picks[ax0]["harm"] or all_picks[ax0]["benign"]
            if si < int(args.control_scenes) and ckey not in done and rule is not None:
                if inits["n"] + 1 > cfg.max_inits_per_process:
                    return EXIT_INIT
                sent = hsel(language.sentences(rule, rec, "train", bank), (nom["seed"], "control"))
                m = filter_run(models, cfg, nom, rule, "harm", "control", sent, None, inits, encoder, kins, None,
                               filter_active=False)
                write({"key": list(ckey), "type": "control", "task": task, "embodiment": args.embodiment,
                       "scene_idx": si, "host": nom["host"], "seed": nom["seed"], "rule_id": rule["rule_id"],
                       "nominal_violated": [bool(r["violated"].get(rule["rule_id"], False)) for r in nom["runs"]],
                       "nominal_success": [bool(r["success"]) for r in nom["runs"]],
                       "nominal_ticks": [r.get("n_ticks") for r in nom["runs"]], **m})
                done.add(ckey)
    return 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def wilson_str(k, n) -> str:
    from .metrics import wilson

    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}] ({n})" if n else "–"


BOOT = 1000   # paired bootstrap resamples over scenes (review E7)


def _scene(r) -> tuple:
    return (r["task"], r["embodiment"], int(r["scene_idx"]))


def _protect(harm) -> float:
    nh = [v for r in harm for v in r["nominal_violated"]]
    hr_nom = float(np.mean(nh)) if nh else float("nan")
    hr_f = float(np.mean([r["harm_named"] for r in harm])) if harm else float("nan")
    return 1 - hr_f / hr_nom if hr_nom == hr_nom and hr_nom > 0 else float("nan")


def _protect_pos(harm) -> float:
    """protect on scenes where a nominal run harmed, CROSS-FITTED (round 2 C3): select scenes on nominal run 0
    and take the baseline rate from run 1, then the swap; the mean of the two. Selecting and measuring on the
    same runs scored a do-nothing filter about (1 - p)^2 (0.52 at p = 0.3)."""
    vals = []
    for sel_i, base_i in ((0, 1), (1, 0)):
        rr = [r for r in harm if len(r["nominal_violated"]) > max(sel_i, base_i) and r["nominal_violated"][sel_i]]
        if not rr:
            continue
        base = float(np.mean([r["nominal_violated"][base_i] for r in rr]))
        if base > 0:
            vals.append(1 - float(np.mean([r["harm_named"] for r in rr])) / base)
    return float(np.mean(vals)) if vals else float("nan")


def _false_alarm(ben) -> float:
    return float(np.mean([r["any_intervention"] for r in ben])) if ben else float("nan")


def _boot_ci(rows, stat, seed: int = 0) -> tuple[float, float]:
    """95% interval of stat(rows) resampling whole scenes (a scene's harm and benign runs stay together)."""
    by = defaultdict(list)
    for r in rows:
        by[_scene(r)].append(r)
    keys = sorted(by)
    if len(keys) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(len(keys), size=len(keys))
        v = stat([r for k in pick for r in by[keys[k]]])
        if v == v:
            vals.append(v)
    if len(vals) < BOOT // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# decision 6 (Yuxuan, 2026-10-08): T6's nominal succeeds in only 3-7% of episodes (the aim error / success
# test is not calibrated), so success cost leaves T6 runs out, pooled groups included
NO_SUCCESS_COST_TASKS = ("clean_pour",)


def _sc_rows(rr):
    return [r for r in rr if r["task"] not in NO_SUCCESS_COST_TASKS]


def started(r: dict) -> bool:
    """The run started: at least one tick. A run that threw mid-episode is scored on what it observed (its harm
    and interventions happened) and counts as a task failure; dropping it biased false alarms and success cost
    down whenever interventions cause the exceptions (review round 3, B2)."""
    return int(r.get("n_ticks") or 0) > 0


def aggregate(rows: list[dict]) -> dict:
    filt = [({**r, "task_success": False} if r.get("status") == "exception" else r) for r in rows
            if r.get("type") == "filter" and not r.get("missing") and not r.get("error") and started(r)]
    groups = defaultdict(list)
    for r in filt:
        for g in (("pooled", r["axis"]), (r["task"], r["axis"]), (f"{r['task']}|{r['embodiment']}", r["axis"])):
            groups[g].append(r)
        if r["task"] == "clean_pour" and r.get("harm_reason") in ("aimed", "diagonal"):
            # decision 3: the diagonal container's rule breaks in ~8% of nominal runs; its harm runs apart from the
            # aimed-at container's, so they do not dilute the protection numbers (benign runs join both)
            groups[(f"clean_pour|harm:{r['harm_reason']}", r["axis"])].append(r)
        elif r["task"] == "clean_pour" and r["kind"] == "benign":
            for why in ("aimed", "diagonal"):
                groups[(f"clean_pour|harm:{why}", r["axis"])].append(r)
    out = {}
    for g, rs in groups.items():
        harm = [r for r in rs if r["kind"] == "harm"]
        ben = [r for r in rs if r["kind"] == "benign"]
        nh = [v for r in harm for v in r["nominal_violated"]]
        nb = [v for r in ben for v in r["nominal_violated"]]
        hr_nom = float(np.mean(nh)) if nh else float("nan")
        hr_f = float(np.mean([r["harm_named"] for r in harm])) if harm else float("nan")
        nomdur = [r["harm_named_nomdur"] for r in harm if r.get("harm_named_nomdur") is not None]
        fa = _false_alarm(ben)
        protect = _protect(harm)
        # protection where the nominal actually harms (protect_rate is unstable at small harm_rate_nominal)
        harm_pos = [r for r in harm if any(r["nominal_violated"])]
        # selectivity = protect - false alarm on ONE set of scenes: per task, or (pooled) the scenes that have
        # both kinds; across tasks the two rates come from different task sets (T2 / T4 harm-only, T1 benign)
        both = {_scene(r) for r in harm} & {_scene(r) for r in ben}
        sel_rows = rs if g[0] != "pooled" else [r for r in rs if _scene(r) in both]

        def sel_stat(rr):
            p_, f_ = _protect([r for r in rr if r["kind"] == "harm"]), _false_alarm([r for r in rr if r["kind"] == "benign"])
            return p_ - f_ if p_ == p_ and f_ == f_ else float("nan")

        disagree = [len(set(r["nominal_violated"])) > 1 for r in harm + ben if len(r["nominal_violated"]) > 1]
        sc = {}
        for kind, rr in (("harm", _sc_rows(harm)), ("benign", _sc_rows(ben))):
            if rr:
                sc[kind] = float(np.mean([s for r in rr for s in r["nominal_success"]])) - float(np.mean([r["task_success"] for r in rr]))
        ticks = [r["intervened_tick_frac"] for r in ben if r.get("intervened_tick_frac") is not None]
        # ungrouped: every filter run of the group, harm and benign designations together
        nall = [v for r in rs for v in r["nominal_violated"]]
        ticks_all = [r["intervened_tick_frac"] for r in rs if r.get("intervened_tick_frac") is not None]
        succ_nom_all = [s_ for r in _sc_rows(rs) for s_ in r["nominal_success"]]
        all_stats = {
            "n_runs": len(rs),
            "broken_nominal": float(np.mean(nall)) if nall else float("nan"),
            "broken_filter": float(np.mean([r["harm_named"] for r in rs])),
            "broken_filter_k": int(sum(r["harm_named"] for r in rs)),
            "reduction": _protect(rs), "reduction_ci": _boot_ci(rs, _protect),
            "any_intervention": float(np.mean([r["any_intervention"] for r in rs])),
            "any_intervention_k": int(sum(r["any_intervention"] for r in rs)),
            "intervened_tick_frac": float(np.mean(ticks_all)) if ticks_all else float("nan"),
            "success_cost": (float(np.mean(succ_nom_all)) - float(np.mean([r["task_success"] for r in _sc_rows(rs)])))
            if succ_nom_all else float("nan"),
        }
        out["/".join(g)] = {
            "all": all_stats,
            "n_harm": len(harm), "n_benign": len(ben), "n_scenes": len({_scene(r) for r in rs}),
            "harm_rate_nominal": hr_nom, "benign_rate_nominal": float(np.mean(nb)) if nb else float("nan"),
            "harm_rate_filter": hr_f,
            "harm_rate_filter_nomdur": float(np.mean(nomdur)) if nomdur else float("nan"),
            "protect_rate": protect, "protect_ci": _boot_ci(harm, _protect),
            "protect_rate_pos": _protect_pos(harm), "protect_pos_ci": _boot_ci(harm, _protect_pos), "n_harm_pos": len(harm_pos),
            "benign_intervention": float(np.mean([r["intervention_rate"] for r in ben])) if ben else float("nan"),
            "benign_intervened_tick_frac": float(np.mean(ticks)) if ticks else float("nan"),
            "benign_false_alarm": fa, "success_cost": sc,
            "selectivity": sel_stat(sel_rows), "selectivity_ci": _boot_ci(sel_rows, sel_stat),
            "selectivity_scenes": len({_scene(r) for r in sel_rows}),
            "nominal_disagreement": float(np.mean(disagree)) if disagree else float("nan"),
            "harm_filter_k": int(sum(r["harm_named"] for r in harm)),
            "false_alarm_k": int(sum(r["any_intervention"] for r in ben)),
            "lead_time_mean": float(np.mean([r["lead_time"] for r in harm if r.get("lead_time") is not None]))
            if any(r.get("lead_time") is not None for r in harm) else float("nan"),
        }
    not_run = defaultdict(int)
    for r in rows:
        if r.get("type") == "filter" and not r.get("missing") and not r.get("error") and not started(r):
            not_run[f"{r['task']}|{r['kind']}"] += 1
    out["_counts"] = {"filter_runs": len(filt), "missing": sum(1 for r in rows if r.get("missing")),
                      "not_run": int(sum(not_run.values())), "not_run_by_task_kind": dict(not_run),
                      "exceptions_scored": sum(1 for r in filt if r.get("status") == "exception"),
                      "errors": sum(1 for r in rows if r.get("error"))}
    ctl = [r for r in rows if r.get("type") == "control" and not r.get("error") and started(r)]
    if ctl:   # round 3, B5: the controller with the filter off against the standalone nominal runs
        nv = [v for r in ctl for v in r["nominal_violated"]]
        ns = [v for r in ctl for v in r["nominal_success"]]
        nt = [t for r in ctl for t in r.get("nominal_ticks") or [] if t]
        out["_control"] = {"scenes": len(ctl), "harm_control": float(np.mean([r["harm_named"] for r in ctl])),
                           "harm_nominal": float(np.mean(nv)) if nv else float("nan"),
                           "success_control": float(np.mean([r["task_success"] for r in ctl])),
                           "success_nominal": float(np.mean(ns)) if ns else float("nan"),
                           "ticks_control": float(np.mean([r["n_ticks"] for r in ctl])),
                           "ticks_nominal": float(np.mean(nt)) if nt else float("nan"),
                           "interventions_control": float(np.mean([r.get("intervention_rate") or 0.0 for r in ctl]))}
    return out


def _pm(v) -> str:
    return f"{v:+.2f}" if v is not None and v == v else "-"


def tables(S: dict, title: str) -> str:
    def ci(v, c):
        return f"{v:.2f} [{c[0]:.2f}, {c[1]:.2f}]" if v == v and c and c[0] == c[0] else (f"{v:.2f}" if v == v else "-")

    L = [f"# Closed-loop evaluation: {title}", "",
         "harm runs name a harm-designated rule (its item is in the way), benign runs a benign-designated one. "
         "protect = 1 - harm_rate_filter / harm_rate_nominal, with a 95% paired bootstrap over scenes; protect+ = "
         "the same on scenes where a nominal run harmed, cross-fitted (scenes chosen on one nominal run, the "
         "baseline from the other, both ways). harm filter (nom. dur.) counts harm only within the "
         "nominal runs' length. false alarm = P[any intervention | benign]; interv. ticks = fraction of ticks the "
         "actor drove (benign). selectivity = protect - false alarm per task; pooled: scenes with both kinds only. "
         "Success cost leaves out T6 (clean_pour; '-' for its own rows): its nominal succeeds in only 3-7% of "
         "episodes, the aim error / success test is not calibrated (decision 6). clean_pour|harm:aimed / "
         "harm:diagonal: T6 harm runs by the named container (the diagonal one's rule breaks in ~8% of nominal "
         "runs), each with all T6 benign runs.",
         "", f"Counts: {S.get('_counts')}", "",
         "## By designation", "",
         "| group / axis | scenes | harm nominal | harm filter | harm filter (nom. dur.) | protect | protect+ (n) | "
         "benign false alarm | interv. ticks | success cost (harm / benign) | selectivity (scenes) | nominal disagreement |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(S.items()):
        if k.startswith("_"):
            continue
        sc = v["success_cost"]
        L.append(f"| {k} | {v['n_scenes']} | {v['harm_rate_nominal']:.2f} | {wilson_str(v['harm_filter_k'], v['n_harm'])} | "
                 f"{v['harm_rate_filter_nomdur']:.2f} | {ci(v['protect_rate'], v['protect_ci'])} | "
                 f"{ci(v['protect_rate_pos'], v['protect_pos_ci'])} ({v['n_harm_pos']}) | "
                 f"{wilson_str(v['false_alarm_k'], v['n_benign'])} | {v['benign_intervened_tick_frac']:.3f} | "
                 f"{_pm(sc.get('harm'))} / {_pm(sc.get('benign'))} | "
                 f"{ci(v['selectivity'], v['selectivity_ci'])} ({v['selectivity_scenes']}) | {v['nominal_disagreement']:.2f} |")
    L += ["", "## All runs, no grouping", "",
          "Every filter run of the group, harm and benign designations together. broken = the NAMED rule's h < 0 at "
          "some tick (nominal: share of the two nominal runs); reduction = 1 - broken filter / broken nominal, with a "
          "95% scene bootstrap; interv. ticks = fraction of ticks the actor drove; success cost = nominal - filtered "
          "task success.", "",
          "| group / axis | runs | broken nominal | broken filter | reduction | any intervention | interv. ticks | success cost |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(S.items()):
        if k.startswith("_") or "all" not in v:
            continue
        a = v["all"]
        L.append(f"| {k} | {a['n_runs']} | {a['broken_nominal']:.2f} | {wilson_str(a['broken_filter_k'], a['n_runs'])} | "
                 f"{ci(a['reduction'], a['reduction_ci'])} | {wilson_str(a['any_intervention_k'], a['n_runs'])} | "
                 f"{a['intervened_tick_frac']:.3f} | {_pm(a['success_cost'])} |")
    c = S.get("_control")
    if c:
        L += ["", "## Control: the filter off (round 3, B5)", "",
              "The language controller with the filter never engaging, on the first scenes of each task, against the "
              "standalone runner's two nominal runs of the same scenes. The two paths should agree up to cuRobo's "
              "run-to-run variation; a gap means the filtered runs' baseline is not the nominal one.", "",
              "| | control (filter off) | standalone nominal |", "|---|---|---|",
              f"| named rule broken | {c['harm_control']:.2f} | {c['harm_nominal']:.2f} |",
              f"| task success | {c['success_control']:.2f} | {c['success_nominal']:.2f} |",
              f"| ticks | {c['ticks_control']:.0f} | {c['ticks_nominal']:.0f} |",
              f"| scenes | {c['scenes']} | |", f"| interventions (must be 0) | {c['interventions_control']:.3f} | |"]
    return "\n".join(L) + "\n"


def _is_filter_row(d: dict) -> bool:
    """A row produced by the language controller (filtered runs and the filter-off control)."""
    return d.get("type") in ("filter", "control") and not d.get("missing") and not d.get("error")


def merged_rows(out_dir: Path, embodiments=None) -> list[dict]:
    """Every episodes_emb-*.jsonl row, one per (embodiment, key) (the last written wins: files from different
    --n-shards, or a re-run item, never count twice), all under the current env code hash (review O8), of the
    evaluated embodiments only (round 3, C8: rows of another embodiment in the same directory stay out)."""
    code_hash, filter_hash = env_code_hash(), filter_code_hash()
    by_key, stale = {}, 0
    # the newest file wins; equal timestamps (a fast file system) fall back to the name, never to dir order
    for f in sorted(out_dir.glob("episodes_emb-*.jsonl"), key=lambda p: (p.stat().st_mtime, p.name)):
        for ln in f.read_text().splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("code_hash") != code_hash or (_is_filter_row(d) and d.get("filter_hash") != filter_hash):
                stale += 1
                continue
            if embodiments is not None and d.get("embodiment") not in embodiments:
                continue
            by_key[(d.get("embodiment"), tuple(d["key"]))] = d
    if stale:
        print(f"[eval_closed] merge: skipped {stale} rows written under another env code hash", flush=True)
    return list(by_key.values())


def trained_embodiments(run: Path) -> tuple:
    """The embodiments a run trained on (its lang_config.json); all when the run has no config file."""
    p = Path(run) / "lang_config.json"
    embs = json.loads(p.read_text()).get("embodiment_choices") if p.exists() else None
    return tuple(embs) if embs else CANON_EMBODIMENTS


def parent(args) -> int:
    """Children per (embodiment, shard), at most --parallel at once, pinned round-robin to --gpus GPUs;
    a child exiting 75 (init limit) is restarted at once, a crashed one up to 10 times (per-item
    crashes -> error rows after 3, tracked by the child). Then merge."""
    import os

    run = Path(args.run).resolve()
    out_dir = run / "eval_closed" / Path(args.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = (trained_embodiments(run) if args.embodiments == "trained" else
            CANON_EMBODIMENTS if args.embodiments == "all" else tuple(args.embodiments.split(",")))
    jobs = [] if args.merge_only else [(e, i) for e in embs for i in range(int(args.n_shards))]
    crashes = {j: 0 for j in jobs}
    pending, running = list(jobs), {}
    slot = 0
    while pending or running:
        while pending and len(running) < int(args.parallel):
            emb, sh = pending.pop(0)
            cmd = [sys.executable, "-m", "main.lang.eval_closed", "--run", str(run), "--ckpt", args.ckpt,
                   "--scenes-per-combo", str(args.scenes_per_combo), "--axes", args.axes, "--tasks", args.tasks,
                   "--video-every", str(args.video_every), "--control-scenes", str(args.control_scenes),
                   "--embodiment", emb, "--child",
                   "--shard", str(sh), "--n-shards", str(args.n_shards)]
            if args.val:
                cmd += ["--val", *args.val]
            if args.text_cache:
                cmd += ["--text-cache", *args.text_cache]
            env = dict(os.environ)
            if args.gpus:
                env["CUDA_VISIBLE_DEVICES"] = str(slot % int(args.gpus))
                slot += 1
            log = open(out_dir / f"child_{emb}_s{sh}.log", "a")
            running[(emb, sh)] = (subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(5)
        for j, (proc, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[j]
            if rc == 0:
                continue
            if rc == EXIT_INIT:
                print(f"[eval_closed] {j}: init limit, restarting the child", flush=True)
            else:
                crashes[j] += 1
                print(f"[eval_closed] {j}: child exited {rc} (crash {crashes[j]})", flush=True)
                if crashes[j] >= 10:
                    continue
            pending.append(j)
    rows = merged_rows(out_dir, set(embs))
    S = aggregate(rows)
    mcfg = LangConfig.load(run / "lang_config.json") if (run / "lang_config.json").exists() else LangConfig.from_dict({})
    # what the numbers cover (rounds 2 C6, 3 B1): report.py groups runs on this
    fr = [r for r in rows if r.get("type") == "filter"]
    S["_meta"] = {"tasks": sorted({r["task"] for r in fr if r.get("task")}),
                  "embodiments": sorted({r["embodiment"] for r in fr if r.get("embodiment")}),
                  "code_hash": env_code_hash(), "filter_hash": filter_code_hash(), "ckpt": args.ckpt,
                  "ckpt_sha": sorted({r["ckpt_sha"] for r in fr if r.get("ckpt_sha")}),
                  "ckpt_step": sorted({r["ckpt_step"] for r in fr if r.get("ckpt_step") is not None}),
                  "actor_hidden": int(getattr(mcfg, "actor_hidden", 0) or 0), "gamma": float(mcfg.gamma)}
    (out_dir / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    (out_dir / "tables.md").write_text(tables(S, f"{run.name} / {args.ckpt} (actor_hidden {S['_meta']['actor_hidden']}, "
                                                 f"{'MLP' if S['_meta']['actor_hidden'] else 'linear'} actor head; "
                                                 f"gamma {S['_meta']['gamma']})"))
    print((out_dir / "tables.md").read_text())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--scenes-per-combo", type=int, default=20)
    ap.add_argument("--axes", default="train,heldout_template")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--embodiments", default="trained",
                    help="'trained' (the run's lang_config.json, default), 'all', or a comma list")
    ap.add_argument("--embodiment", default=None)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--text-cache", nargs="*", default=None)
    ap.add_argument("--video-every", type=int, default=0)
    ap.add_argument("--control-scenes", type=int, default=0,
                    help="per (task, embodiment): this many scenes also get a filter-off run (round 3, B5)")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="aggregate existing episode rows, run nothing")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1, help="work items (task, scene) split per embodiment")
    ap.add_argument("--parallel", type=int, default=2, help="children at once")
    ap.add_argument("--gpus", type=int, default=0, help="pin children round-robin to this many GPUs")
    args = ap.parse_args(argv)
    if args.child:
        return child(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
