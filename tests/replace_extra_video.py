#!/usr/bin/env python
"""One-off: replace the extra loo_franka presentation video for ONE
embodiment with a scene where the filter ACTUALLY intervenes.

Candidates are harvested straight from <run>/eval_full/checkpoint_rm0/t18
(success, no collision, intervention_rate > 0, smallest min_h first) and
verified on REPLAY — including trace["interventions"] > 0, since GPU
nondeterminism can make the replay differ from the eval row. The nominal
on the same seed is recorded too; a colliding nominal is preferred.

    python tests/replace_extra_video.py --emb ur5-wsg \
        --out $TMP/cehj/videos_presentation
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "tests"

from tests.record_presentation import (  # noqa: E402
    FORCE_VIOL_N, RUNS_DIR, record_one, trace_collision)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--run", default="loo_franka")
    p.add_argument("--task", default=None, help="restrict to one task")
    p.add_argument("--budget", type=int, default=18,
                   help="eval budget dir to harvest from (t<budget>)")
    p.add_argument("--exclude-json", type=str, default=None,
                   help="comma-separated selections jsons — skip scenes "
                        "already picked")
    p.add_argument("--allow-contact", action="store_true",
                   help="pool and replay acceptance are success-only "
                        "(contact allowed)")
    p.add_argument("--min-interventions", type=int, default=1,
                   help="replay must show at least this many interventions")
    p.add_argument("--selections-out", type=Path, default=None,
                   help="merge the chosen candidate into this json (per emb)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-trials", type=int, default=8)
    args = p.parse_args()

    t_dir = (RUNS_DIR / args.run
             / f"eval_full/checkpoint_rm0/t{args.budget}")
    rows = []
    for f in sorted(t_dir.glob("episodes*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("embodiment") != args.emb or "error" in r:
                continue
            if args.task is not None and r.get("task") != args.task:
                continue
            rows.append(r)
    if args.allow_contact:
        cands = [r for r in rows if r.get("success")]
    else:
        cands = [r for r in rows
                 if r.get("success") and not r.get("collision")
                 and r.get("intervention_rate", 0) > 0]
    cands.sort(key=lambda r: (r.get("intervention_rate", 0) == 0,
                              r.get("min_h_cm", 1e9)))
    taken = set()
    for pth in (args.exclude_json.split(",") if args.exclude_json else []):
        pth = Path(pth)
        if not pth.exists():
            continue
        sel = json.loads(pth.read_text())
        for group in sel.values():
            if isinstance(group, dict):
                for u in group.values():
                    if isinstance(u, dict) and "episode" in u:
                        taken.add((u.get("embodiment"), u.get("task"),
                                   u.get("episode")))
    before = len(cands)
    cands = [c for c in cands
             if (c.get("embodiment"), c.get("task"), c.get("episode"))
             not in taken]
    print(f"[cand] {len(cands)} success+clean+IR>0 rows for "
          f"{args.emb}/{args.task or '*'} in {args.run} "
          f"(t{args.budget}; {before - len(cands)} excluded as taken)")

    from main.train.config import FrozenConfig
    from main.train.eval_ckpt import build_models

    def load_cfg():
        cfg = FrozenConfig.load(RUNS_DIR / args.run / "config.json")
        cfg.obstacle_mode = "on_path"
        cfg.filter_release_margin = 0.0
        return cfg

    cfg = load_cfg()
    kind, encoder, kin_fn, model_kw = build_models(
        cfg, RUNS_DIR / args.run / "checkpoint.pt")
    assert kind == "hb", kind
    nom_kind, nom_encoder, nom_kin_fn, _ = build_models(load_cfg(), None)
    assert nom_kind == "nominal"

    out_dir = args.out.resolve()
    fallback = None
    chosen, nom_collides = None, False
    for i, cand in enumerate(cands[: args.max_trials]):
        task, ep, seed = cand["task"], cand["episode"], cand["seed"]
        loo_vp = out_dir / f"{args.run}_{args.emb}_{task}_ep{ep:03d}.mp4"
        nom_vp = out_dir / f"nominal_{args.emb}_{task}_ep{ep:03d}.mp4"
        if loo_vp.exists() or nom_vp.exists():
            print(f"[skip] {task}/{args.emb} ep{ep:03d}: output exists")
            continue
        trace = record_one(cfg, seed, task, args.emb, args.budget, encoder,
                           kin_fn(args.emb), model_kw, loo_vp, show_q=True)
        touch = any(trace["contact_touch"])
        fmax = max(trace["contact_force"]) if trace["contact_force"] else 0.0
        n_iv = trace.get("interventions", 0)
        cm = 100.0 / float(cfg.h_scale)
        min_h = min(trace["h"]) * cm if trace["h"] else float("nan")
        ok = (bool(trace["success"])
              and (args.allow_contact
                   or (not touch and fmax < FORCE_VIOL_N))
              and n_iv >= args.min_interventions)
        print(f"[loo] {task}/{args.emb} cand{i} ep{ep:03d} seed={seed}: "
              f"success={trace['success']} contact={touch} "
              f"fmax={fmax:.1f}N min_h={min_h:.2f}cm interventions={n_iv} "
              f"{'' if ok else '— REJECT'}")
        if not ok:
            loo_vp.unlink(missing_ok=True)
            continue
        trace_nom = record_one(load_cfg(), seed, task, args.emb,
                               args.budget, nom_encoder,
                               nom_kin_fn(args.emb), {},
                               nom_vp, show_q=False)
        coll = trace_collision(trace_nom)
        print(f"[nom] {task}/{args.emb} cand{i} seed={seed}: "
              f"success={trace_nom['success']} collision={coll}")
        if coll:
            chosen, nom_collides = cand, True
            if fallback is not None:
                fallback[1].unlink(missing_ok=True)
                fallback[2].unlink(missing_ok=True)
            break
        if fallback is None:
            fallback = (cand, loo_vp, nom_vp)
        else:
            loo_vp.unlink(missing_ok=True)
            nom_vp.unlink(missing_ok=True)
    if chosen is None:
        if fallback is None:
            print(f"[sel] {args.emb}: no intervening clean loo replay — "
                  f"NOTHING RECORDED")
            return
        chosen = fallback[0]
        print(f"[sel] {args.emb}: nominal never collided — keeping the "
              f"first intervening pair")
    info = {"run": args.run, "task": chosen["task"],
            "embodiment": chosen.get("embodiment", args.emb),
            "episode": chosen["episode"], "seed": chosen["seed"],
            "min_h_cm": chosen.get("min_h_cm"),
            "intervention_rate": chosen.get("intervention_rate"),
            "budget": args.budget, "nominal_collision": nom_collides}
    print(json.dumps(info, indent=1))
    if args.selections_out is not None:
        d = (json.loads(args.selections_out.read_text())
             if args.selections_out.exists() else {})
        d[args.emb] = info
        args.selections_out.write_text(json.dumps(d, indent=1))
        print(f"[sel] merged into {args.selections_out}")
    print("REPLACE DONE")


if __name__ == "__main__":
    main()
