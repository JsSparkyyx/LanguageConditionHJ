"""Q traces of two sentences on the same nominal run (review round 3, Part II E0a / E3).

    python scripts/lang/grounding_traces.py --run <run> --ckpt checkpoint_final.pt \
        --val data/lang/val_grounding --variant v0 --out runs/lang/grounding_1008/e0a [--select <selection.json>]

Scenes: --scenes-per-task per task from the validation episodes, each with a harm-designated and a
benign-designated rule of the same consequence (avoid / stay / upright): sentence A names the harm rule (its
item is in the way of the careless motion), sentence B the benign one. The selection is written to
<out>/selection.json; --select reuses one (E3: the same seeds and sentences for another checkpoint).

Each scene runs once with LangRolloutController in monitor_only mode: the filter never engages (nominal
motion) while Q(s, a_nom) of A and of B (the counterfactual path) is recorded every tick. Writes
<out>/<variant>_<task>_s<seed>.json (t, h(A), h(B), Q(A), Q(B), sentences, outcome) for the page's figures,
and videos/grounding_<variant>_<t1|t7>_s<seed>.mp4 with the two-sentence panel.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONS = ("avoid", "stay", "upright")
SHORT = {"selective_avoid": "t1", "gap_place": "t7"}


def single_object(rule: dict) -> bool:
    return rule["consequence"] in CONS and "object" in (rule.get("subject") or {}) and not rule.get("set")


def select_scenes(roots, tasks, per_task: int) -> list[dict]:
    """Episodes (sorted by seed) with a harm- and a benign-designated sampleable rule of one consequence."""
    from main.item_envs import language
    from main.lang.eval_closed import hsel

    bank = language.load_bank()
    out = []
    for task in tasks:
        cand = []
        for root in roots:
            for ln in (Path(root) / "lang_episodes.jsonl").read_text().splitlines():
                d = json.loads(ln)
                if d.get("skipped") or d["task"] != task or d.get("status") not in ("ok", "timeout"):
                    continue
                rec = json.loads((Path(root) / "item_buffer" / "episodes" / f"{d['episode_id']}.json").read_text())
                des = d.get("designations") or {}
                rules = [r for r in rec["rules"] if r.get("sampleable") and single_object(r)]
                for cons in CONS:
                    harm = [r for r in rules if r["consequence"] == cons and des.get(r["rule_id"]) == "harm"]
                    ben = [r for r in rules if r["consequence"] == cons and des.get(r["rule_id"]) == "benign"]
                    if harm and ben:
                        a, b = hsel(harm, (d["seed"], "A")), hsel(ben, (d["seed"], "B"))
                        cand.append({"task": task, "seed": int(d["seed"]), "host": d.get("host"),
                                     "root": str(root), "rule_a": a["rule_id"], "rule_b": b["rule_id"],
                                     "consequence": cons,
                                     "sentence_a": hsel(language.sentences(a, rec, "train", bank), (d["seed"], "sA")),
                                     "sentence_b": hsel(language.sentences(b, rec, "train", bank), (d["seed"], "sB"))})
                        break
        cand.sort(key=lambda c: c["seed"])
        out += cand[:per_task]
    return out


def run_scene(models, sc: dict, variant: str, out: Path, vdir: Path, encoder, kins) -> dict:
    import torch

    from main.item_envs import language
    from main.item_envs.controller import RiskTickController
    from main.lang.eval_closed import set_global_seeds
    from main.lang.rollout import LangRolloutController, wire_item_eval
    from main.lang.video import LangPanelVideoWriter
    from main.train.collect import CKPT_DIR, make_kinematics

    emb, seed, task = "ur5-wsg", int(sc["seed"]), sc["task"]
    if emb not in kins:
        kins[emb] = make_kinematics(CKPT_DIR, emb)
    cfg = models.cfg
    cfg_ep = dataclasses.replace(cfg, task=task, embodiment=emb, item_host=sc.get("host"), item_active_rule=None,
                                 item_out_root=str(REPO / "runs/lang/_grounding_runner"))
    set_global_seeds(seed)
    ro = LangRolloutController(cfg_ep, seed, mode="filtered", encoder=encoder, kin=kins[emb], policy_enc=models.pe,
                               actor=models.actor, critics=models.critics, filter_active=True, monitor_only=True,
                               episode_id=seed, rng=np.random.RandomState(seed),
                               max_steps=10 * int(cfg.max_steps_per_episode), controller_cls=RiskTickController,
                               text_store=models.text)
    runner = ro.env._item_runner
    rec = language.episode_record(runner, seed)
    ids = [r["rule_id"] for r in rec["rules"]]
    if sc["rule_a"] not in ids or sc["rule_b"] not in ids:
        ro.env.close()
        return {"error": "the scene's rule table differs from the recorded episode"}
    rule_a, rule_b = (rec["rules"][ids.index(r)] for r in (sc["rule_a"], sc["rule_b"]))
    wire_item_eval(ro, sc["rule_a"])
    ro.set_instruction(sc["sentence_a"], sc["sentence_b"], rule_id=sc["rule_a"], cf_rule_id=sc["rule_b"])
    vdir.mkdir(parents=True, exist_ok=True)
    stem = vdir / f"grounding_{variant}_{SHORT.get(task, task)}_s{seed}"
    ro.video = LangPanelVideoWriter(stem, ro, named_rule=rule_a, sentence=sc["sentence_a"], cf_rule=rule_b,
                                    cf_sentence=sc["sentence_b"])
    with torch.no_grad():
        trace = ro.run()
    path = ro.video.close()
    final = stem.with_suffix(".mp4")
    if path is not None and Path(path).exists():
        os.replace(path, final)
    tr = ro.trace
    j_b = ids.index(sc["rule_b"])
    h_all = np.asarray(tr.get("h_all", []), dtype=np.float64)
    res = {**sc, "variant": variant, "status": trace.get("status"), "success": bool(trace.get("success", False)),
           "t": [float(x) for x in tr.get("t", [])], "h_a": [float(x) for x in tr.get("h_named", [])],
           "h_b": [float(x) for x in (h_all[:, j_b] if h_all.size else [])],
           "q_t": [float(x) for x in tr.get("V_t", [])], "q_a": [float(x) for x in tr.get("V", [])],
           "q_b_t": [float(x) for x in tr.get("Q_cf_t", [])], "q_b": [float(x) for x in tr.get("Q_cf", [])],
           "intervened": int(np.sum(tr.get("intervened", []))), "video": str(final)}
    (out / f"{variant}_{task}_s{seed}.json").write_text(json.dumps(res))
    return res


def main(argv=None) -> int:
    import main.lang  # noqa: F401
    from main.lang.eval_closed import Models
    from main.lang.train import caches_of, roots_of
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--tasks", default="selective_avoid,gap_place")
    ap.add_argument("--scenes-per-task", type=int, default=3)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default=None, help="video directory (default: <out>/../videos)")
    ap.add_argument("--select", default=None, help="reuse this selection.json (same seeds and sentences)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = roots_of(args.val)
    if not roots:
        raise SystemExit(f"--val {args.val}: no shards")
    if args.select:
        scenes = json.loads(Path(args.select).read_text())
    else:
        scenes = select_scenes(roots, args.tasks.split(","), args.scenes_per_task)
        (out / "selection.json").write_text(json.dumps(scenes, indent=1))
    print(f"[grounding_traces] {len(scenes)} scenes: " + ", ".join(f"{s['task']} s{s['seed']}" for s in scenes), flush=True)
    models = Models(Path(args.run), args.ckpt, caches_of(args.val))
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kins = {}
    vdir = Path(args.videos) if args.videos else out.parent / "videos"
    for sc in scenes:
        done = out / f"{args.variant}_{sc['task']}_s{sc['seed']}.json"
        if done.exists():
            continue
        r = run_scene(models, sc, args.variant, out, vdir, encoder, kins)
        print(f"[grounding_traces] {args.variant} {sc['task']} s{sc['seed']}: {r.get('status', r.get('error'))} "
              f"A '{sc['sentence_a']}' / B '{sc['sentence_b']}'", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
