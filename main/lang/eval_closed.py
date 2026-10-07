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


def env_code_hash() -> str:
    h = hashlib.md5()
    for pat in ("envs/main/item_envs/*.py", "envs/main/item_envs/configs/*.yaml", "envs/robotwin_tasks/*.py"):
        for p in sorted(REPO.glob(pat)):
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def hsel(items, key):
    """Deterministic choice by a hash of key."""
    if not items:
        return None
    return items[int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) % len(items)]


# ---------------------------------------------------------------------------
# nominal runs (cached, shared by every checkpoint)
# ---------------------------------------------------------------------------


def nominal_scene(task: str, emb: str, scene_idx: int, cfg: LangConfig, cache_dir: Path, inits: dict,
                  code_hash: str) -> dict:
    from main.item_envs import language
    from main.item_envs.runner import ItemEpisodeRunner

    from .scene_roles import all_designations, in_way

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
            except Exception as exc:
                out["errors"].append(f"seed {seed} run {k}: {type(exc).__name__}: {exc}"[:300])
                runner.close()
                ok = False
                break
            if record is None:
                record, roles, desig = rec, rl, ds
            elif [r["rule_id"] for r in rec["rules"]] != [r["rule_id"] for r in record["rules"]] or ds != desig:
                out["errors"].append(f"seed {seed}: rule table / designations differ between nominal runs")
            meta, error = runner.run()
            runs.append({"violated": {r: bool(v) for r, v in (meta.violations or {}).items()},
                         "first_violation_t": meta.first_violation_t, "success": bool(meta.task_success),
                         "error": error, "wall_s": meta.wall_time_s})
        if ok:
            out.update({"seed": seed, "attempt": attempt, "runs": runs, "record": record,
                        "in_way": sorted(roles[0]), "unclear": sorted(roles[1]), "designations": desig})
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

        from main.network.heads import TokenActor

        from .network import LangPolicyEncoder, LangTwinCritic
        from .text import TextStore
        from .train import load_policy

        ck = torch.load(run / ckpt if not Path(ckpt).is_absolute() else ckpt, map_location="cuda", weights_only=False)
        self.cfg = LangConfig.from_dict(ck["lang_config"])
        self.pe = LangPolicyEncoder(self.cfg).cuda().eval()
        load_policy(self.pe, ck)
        self.actor = TokenActor().cuda().eval()
        self.actor.load_state_dict(ck["actor"])
        self.critics = LangTwinCritic(self.cfg).cuda().eval()
        self.critics.load_state_dict(ck["critics"])
        self.text = TextStore(caches, self.cfg.text_backend, "cuda", extra_path=run / "text_cache_extra.pt",
                              backend_paths=self.cfg.backend_paths)
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
    m = {
        "status": trace.get("status"), "task_success": bool(trace.get("success", False)),
        "harm_named": bool(h.size and (h < 0).any()), "harm_named_frac": float((h < 0).mean()) if h.size else None,
        "intervention_rate": float(trace.get("intervention_rate", 0.0) or 0.0), "any_intervention": bool(inter.any()),
        "harm_other_any": bool(other.size and (other[np.isfinite(other)] < 0).any()),
        "v_le_h_frac": float((Q <= hq + 0.05).mean()) if Q.size and hq.size == Q.size else None,
        "hq_corr": float(np.corrcoef(hq, Q)[0, 1]) if Q.size > 2 and np.std(Q) > 0 and np.std(hq) > 0 else None,
        "first_intervention_t": first_int, "n_ticks": int(t.size),
    }
    fvs = [r["first_violation_t"].get(named_id) for r in nom["runs"] if (r.get("first_violation_t") or {}).get(named_id)]
    if kind == "harm" and fvs and first_int is not None:
        m["lead_time"] = float(np.mean(fvs)) - first_int
    if qcf.size and Q.size:
        n = int(np.argmax(inter)) if inter.any() else Q.size
        k = min(n, qcf.size, Q.size)
        m["q_cf_gap"] = float(np.mean(qcf[:k] - Q[:k])) if k else None
    return m


def filter_run(models: Models, cfg: LangConfig, nom: dict, rule: dict, kind: str, axis: str, sentence: str,
               cf_sentence: str | None, inits: dict, encoder, kins, video_dir: Path | None) -> dict:
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
                               policy_enc=models.pe, actor=models.actor, critics=models.critics, filter_active=True,
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
    done = set()
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            try:
                d = json.loads(ln)
                done.add(tuple(d["key"]))
            except Exception:
                pass
    caches = [Path(c) for c in args.text_cache] if args.text_cache else caches_of(args.val or [])
    models = Models(run, args.ckpt, caches)
    cfg = models.cfg
    bank = language.load_bank()
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kins, inits = {}, {"n": 0}
    code_hash = env_code_hash()
    cache_dir = REPO / "runs/lang/_nominal_cache"
    axes = args.axes.split(",")
    tasks = CANON_TASKS if args.tasks == "all" else tuple(args.tasks.split(","))
    crashes = defaultdict(int)
    cr_path = out_dir / f"crashes_{tag}.json"
    if cr_path.exists():
        crashes.update({tuple(json.loads(k)): v for k, v in json.loads(cr_path.read_text()).items()})

    def write(row):
        with open(jsonl, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

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
            if "record" not in nom:
                continue
            rec = nom["record"]
            for axis in axes:
                picks = {}
                for kind in ("harm", "benign"):
                    el = [r for r in rec["rules"] if nom["designations"].get(r["rule_id"]) == kind
                          and eligible(r, rec, axis, bank)]
                    picks[kind] = hsel(el, (nom["seed"], axis, kind))
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
                           "split": rule.get("split"), "sentence": sent, "cf_sentence": cf,
                           "nominal_violated": nom_harm, "nominal_success": nom_succ, "wall_s": time.time() - t0, **m})
                    done.add(key)
                    print(f"[eval_closed] {task} {args.embodiment} scene {si} {axis}/{kind} {rule['rule_id']} "
                          f"'{sent}': harm {m.get('harm_named')} (nominal {nom_harm}) interv {m.get('intervention_rate')} "
                          f"success {m.get('task_success')} {m.get('error') or ''}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def wilson_str(k, n) -> str:
    from .metrics import wilson

    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}] ({n})" if n else "–"


def aggregate(rows: list[dict]) -> dict:
    filt = [r for r in rows if r.get("type") == "filter" and not r.get("missing") and not r.get("error")]
    groups = defaultdict(list)
    for r in filt:
        for g in (("pooled", r["axis"]), (r["task"], r["axis"]), (f"{r['task']}|{r['embodiment']}", r["axis"])):
            groups[g].append(r)
    out = {}
    for g, rs in groups.items():
        harm = [r for r in rs if r["kind"] == "harm"]
        ben = [r for r in rs if r["kind"] == "benign"]
        nh = [v for r in harm for v in r["nominal_violated"]]
        nb = [v for r in ben for v in r["nominal_violated"]]
        hr_nom = float(np.mean(nh)) if nh else float("nan")
        hr_f = float(np.mean([r["harm_named"] for r in harm])) if harm else float("nan")
        fa = float(np.mean([r["any_intervention"] for r in ben])) if ben else float("nan")
        protect = 1 - hr_f / hr_nom if hr_nom and hr_nom == hr_nom and hr_nom > 0 else float("nan")
        disagree = [len(set(r["nominal_violated"])) > 1 for r in harm + ben if len(r["nominal_violated"]) > 1]
        sc = {}
        for kind, rr in (("harm", harm), ("benign", ben)):
            if rr:
                sc[kind] = float(np.mean([s for r in rr for s in r["nominal_success"]])) - float(np.mean([r["task_success"] for r in rr]))
        out["/".join(g)] = {
            "n_harm": len(harm), "n_benign": len(ben),
            "harm_rate_nominal": hr_nom, "benign_rate_nominal": float(np.mean(nb)) if nb else float("nan"),
            "harm_rate_filter": hr_f, "protect_rate": protect,
            "benign_intervention": float(np.mean([r["intervention_rate"] for r in ben])) if ben else float("nan"),
            "benign_false_alarm": fa, "success_cost": sc,
            "selectivity": protect - fa if protect == protect and fa == fa else float("nan"),
            "nominal_disagreement": float(np.mean(disagree)) if disagree else float("nan"),
            "harm_filter_k": int(sum(r["harm_named"] for r in harm)),
            "false_alarm_k": int(sum(r["any_intervention"] for r in ben)),
            "lead_time_mean": float(np.mean([r["lead_time"] for r in harm if r.get("lead_time") is not None]))
            if any(r.get("lead_time") is not None for r in harm) else float("nan"),
        }
    out["_counts"] = {"filter_runs": len(filt), "missing": sum(1 for r in rows if r.get("missing")),
                      "errors": sum(1 for r in rows if r.get("error"))}
    return out


def tables(S: dict, title: str) -> str:
    L = [f"# Closed-loop evaluation: {title}", "",
         "harm runs name a harm-designated rule (its item is in the way), benign runs a benign-designated one. "
         "protect_rate = 1 - harm_rate_filter / harm_rate_nominal; benign_false_alarm = P[any intervention | benign].",
         "", f"Counts: {S.get('_counts')}", "",
         "| group / axis | harm nominal | harm filter | protect | benign nominal | benign false alarm | benign interv. | "
         "success cost (harm / benign) | selectivity | nominal disagreement |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(S.items()):
        if k.startswith("_"):
            continue
        sc = v["success_cost"]
        L.append(f"| {k} | {v['harm_rate_nominal']:.2f} | {wilson_str(v['harm_filter_k'], v['n_harm'])} | "
                 f"{v['protect_rate']:.2f} | {v['benign_rate_nominal']:.2f} | {wilson_str(v['false_alarm_k'], v['n_benign'])} | "
                 f"{v['benign_intervention']:.3f} | {sc.get('harm', float('nan')):+.2f} / {sc.get('benign', float('nan')):+.2f} | "
                 f"{v['selectivity']:.2f} | {v['nominal_disagreement']:.2f} |")
    return "\n".join(L) + "\n"


def parent(args) -> int:
    """Children per (embodiment, shard), at most --parallel at once, pinned round-robin to --gpus GPUs;
    a child exiting 75 (init limit) is restarted at once, a crashed one up to 10 times (per-item
    crashes -> error rows after 3, tracked by the child). Then merge."""
    import os

    run = Path(args.run).resolve()
    out_dir = run / "eval_closed" / Path(args.ckpt).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = CANON_EMBODIMENTS if args.embodiments == "all" else tuple(args.embodiments.split(","))
    jobs = [] if args.merge_only else [(e, i) for e in embs for i in range(int(args.n_shards))]
    crashes = {j: 0 for j in jobs}
    pending, running = list(jobs), {}
    slot = 0
    while pending or running:
        while pending and len(running) < int(args.parallel):
            emb, sh = pending.pop(0)
            cmd = [sys.executable, "-m", "main.lang.eval_closed", "--run", str(run), "--ckpt", args.ckpt,
                   "--scenes-per-combo", str(args.scenes_per_combo), "--axes", args.axes, "--tasks", args.tasks,
                   "--video-every", str(args.video_every), "--embodiment", emb, "--child",
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
    rows = []
    for f in sorted(out_dir.glob("episodes_emb-*.jsonl")):
        for ln in f.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    S = aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    (out_dir / "tables.md").write_text(tables(S, f"{run.name} / {args.ckpt}"))
    print((out_dir / "tables.md").read_text())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--scenes-per-combo", type=int, default=20)
    ap.add_argument("--axes", default="train,heldout_template")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--embodiments", default="all")
    ap.add_argument("--embodiment", default=None)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--text-cache", nargs="*", default=None)
    ap.add_argument("--video-every", type=int, default=0)
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
