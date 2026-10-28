#!/usr/bin/env python
"""T3 push_zone safe mode: route feasibility off-GPU (no simulator, Yuxuan 2026-10-27 demo set).

Replays the layouts of many seeds exactly (tests/item_envs/test_push_zone.py's loader: the real task module with
sapien and RoboTwin stubbed, the pre-load RNG stream reproduced) and runs the safe mode's route planner
(robotwin_tasks/_push_route.py, the same call push_zone._play_route makes) on each layout.

Cases per layout:
  all        every element protected (the demo collector: no rule named)
  named_on   each on-path element named alone
  named_off  the off-path element named (nothing blocks the straight route: it must come back straight)
  referent   the layout's primary rule (the A elements, else the off-path one)
Variants: every combination of --hands (wide, narrow) and --reaches (npz, linear, either, both, none); --cfg
overrides route knobs (JSON, e.g. '{"mat_clear": 0.025}'); the yaml's safe_mode.route is the base.

Output (--out): routes.jsonl (one row per seed x case x variant), summary.md (yield by scenario and by the
on-path element kinds, per case and variant; failure reasons; planning time), printed too; with --png N, top-down
plots of the first N layouts (the "all" case of the first variant: route, box and hand footprints, clearances).

    cd envs && python scripts/item_envs/t3_route_feasibility.py --seeds 0-59 --out runs/diag/t3_routes
The full sweep (about 1000 layouts, a few variants) belongs on a compute node (srun), not the login node.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]   # envs/
sys.path.insert(0, str(ROOT))

TASK_YAML = ROOT / "main" / "item_envs" / "configs" / "tasks" / "T3_push_zone.yaml"
NPZ = ROOT / "main" / "item_envs" / "configs" / "workspace_{}.npz"
CASES = ("all", "named_on", "named_off", "referent")


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _kinds(elements, on_path=True) -> str:
    on = sorted((e for e in elements if e["on_path"] == on_path), key=lambda e: (e["order"] is None, e["order"]))
    return "+".join("mat" if e["kind"] == "zone" else "cyl" for e in on) or "-"


def _protect_sets(task, cases):
    """(case, protected ids or None = all, named element id or None) for one layout."""
    li = task.layout_info
    out = []
    for case in cases:
        if case == "all":
            out.append((case, None, None))
        elif case == "named_on":
            out += [(case, [e["id"]], e["id"]) for e in li["elements"] if e["on_path"]]
        elif case == "named_off":
            out += [(case, [e["id"]], e["id"]) for e in li["elements"] if not e["on_path"]]
        elif case == "referent":
            out.append((case, list(li["referent"]), "+".join(li["referent"])))
    return out


def _samples(pr, ctx, pts, yaw0):
    """The planner's box / hand samples of a route, for drawing."""
    acc = pr._Acc()
    lo = hi = float(yaw0)
    prev = None
    for k in range(len(pts) - 1):
        a, b = np.asarray(pts[k], dtype=np.float64), np.asarray(pts[k + 1], dtype=np.float64)
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        prev, lo, hi = pr._leg(ctx, a, b, prev, lo, hi, k, acc)
    return acc.arrays()


def _draw(path, pr, task, inp, route, cfg, hand, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    start, goal, bhalf, yaw, box_h, elems = inp
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.add_patch(Rectangle((-0.6, -0.35), 1.2, 0.7, fill=False, lw=1.5, color="k"))
    for e in elems:
        if e.kind == "zone":
            ax.add_patch(Rectangle((e.xy[0] - e.half[0], e.xy[1] - e.half[1]), 2 * e.half[0], 2 * e.half[1],
                                   color="tab:orange", alpha=0.5))
        else:
            ax.add_patch(Circle(e.xy, e.half[0], color="tab:blue", alpha=0.6))
        ax.text(e.xy[0], e.xy[1] + 0.05, e.id, ha="center", fontsize=7)
    g = task.push_objects[0]["goal_xy"]
    ax.add_patch(Rectangle((g[0] - 0.06, g[1] - 0.06), 0.12, 0.12, fill=False, lw=1.0, color="0.2"))
    pts = [np.asarray(start)] + [np.asarray(v) for v in (route.via if route.ok else [goal])]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color="tab:green" if route.ok else "tab:red", ms=3)
    ctx = pr._Ctx(cfg, hand, bhalf, box_h, None, elems, None, None)
    box, hnd, _ = _samples(pr, ctx, pts, yaw)
    for row in box[:: max(1, len(box) // 40)]:
        kx, ky = pr._rect_corners(row[0], row[1], row[2], ctx.bx, ctx.by)
        ax.add_patch(Polygon(np.stack([kx, ky], axis=1), fill=False, lw=0.4, color="tab:green"))
    base = hand.layer("base")
    for row in hnd[:: max(1, len(hnd) // 25)]:
        kx, ky = pr._rect_corners(row[0], row[1], row[2], base.a + row[3], base.b + row[4])
        ax.add_patch(Polygon(np.stack([kx, ky], axis=1), fill=False, lw=0.4,
                             color="tab:purple" if row[5] <= ctx.ee_h + 1e-6 else "tab:cyan"))
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.37, 0.37)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="0-59", help="e.g. 0-999 or 0-9,100-109")
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    ap.add_argument("--hands", nargs="+", default=["wide"], choices=["wide", "narrow"])
    ap.add_argument("--reaches", nargs="+", default=["npz"], choices=["npz", "linear", "either", "both", "none"])
    ap.add_argument("--cfg", default="{}", help="route knob overrides (JSON) on top of the yaml's safe_mode.route")
    ap.add_argument("--embodiment", default="ur5-wsg")
    ap.add_argument("--out", default=None, help="output dir (routes.jsonl, summary.md, PNGs)")
    ap.add_argument("--png", type=int, default=0, help="top-down plots of the first N layouts")
    args = ap.parse_args()

    from tests.item_envs.test_push_zone import _load_task_module, _replay_layout

    task_cfg = yaml.safe_load(TASK_YAML.read_text())
    base = dict(((task_cfg.get("safe_mode") or {}).get("route") or {}), **json.loads(args.cfg))
    overrides = (task_cfg.get("embodiment_overrides") or {}).get(args.embodiment) or {}
    tip_len = float(overrides.get("push_tip_len", 0.164))
    npz = NPZ.with_name(NPZ.name.format(args.embodiment))
    out = Path(args.out) if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    pz, restore = _load_task_module()
    try:
        pr = importlib.import_module("pz_replay._push_route")
        for n_seed, seed in enumerate(_seeds(args.seeds)):
            task = _replay_layout(pz, seed)
            inp = task._route_inputs()
            start, goal, bhalf, yaw, box_h, elems = inp
            li = task.layout_info
            arm = task.push_objects[0]["arm"]
            for hand_mode, reach_kind in itertools.product(args.hands, args.reaches):
                cfg = pr.RouteCfg.from_dict({**base, "hand": hand_mode, "reach": reach_kind})
                hand = pr.wsg_hand(hand_mode, tip_len=tip_len)
                reach = pr.make_reach(reach_kind, arm, npz if npz.is_file() else None, erode=cfg.reach_erode)
                for case, protect, named in _protect_sets(task, args.cases):
                    t0 = time.perf_counter()
                    r = pr.plan_route(start, goal, bhalf, yaw, elems, protect=protect, cfg=cfg, reach_ok=reach,
                                      box_h=box_h, hand=hand, ee_height=cfg.tip_clear + tip_len)
                    ms = 1000.0 * (time.perf_counter() - t0)
                    d = r.as_dict()
                    kind = None if named is None else "+".join(
                        sorted({"mat" if e["kind"] == "zone" else "cyl" for e in li["elements"] if e["id"] in protect}))
                    rows.append({
                        "seed": seed, "scenario": li["scenario"], "mode": li["mode"], "arm": arm, "case": case,
                        "named": named, "named_kind": kind, "on_kinds": _kinds(li["elements"]),
                        "off_kinds": _kinds(li["elements"], on_path=False), "hand": hand_mode, "reach": reach_kind,
                        "ok": d["ok"], "reason": d["reason"], "n_via": len(d["via"]), "turns_deg": d["turns_deg"],
                        "first_deg": d["first_deg"], "length": d["length"], "min_clear": d["min_clear"],
                        "sides": d["sides"], "detail": d["detail"], "ms": round(ms, 1),
                    })
                    if out is not None and n_seed < args.png and case == "all" and \
                            (hand_mode, reach_kind) == (args.hands[0], args.reaches[0]):
                        _draw(out / f"seed{seed:04d}_{case}.png", pr, task, inp, r, cfg, hand,
                              f"seed {seed} {li['scenario']}/{li['mode']} {arm} arm, {case}, {hand_mode}/{reach_kind}: "
                              f"{'ok' if r.ok else r.reason} turns {d['turns_deg']} clear {d['min_clear']}")
    finally:
        restore()
    text = summarize(rows)
    print(text)
    if out is not None:
        with open(out / "routes.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        (out / "summary.md").write_text(text + "\n")
    return 0


def _rate(rows) -> str:
    n = len(rows)
    k = sum(1 for r in rows if r["ok"])
    return f"{k}/{n} ({100.0 * k / n:.0f}%)" if n else "-"


def summarize(rows) -> str:
    lines = []
    by_var = defaultdict(list)
    for r in rows:
        by_var[(r["hand"], r["reach"])].append(r)
    for (hand_mode, reach_kind), vrows in by_var.items():
        lines.append(f"## hand {hand_mode}, reach {reach_kind}: {len({r['seed'] for r in vrows})} layouts")
        lines.append("")
        cases = [c for c in CASES if any(r["case"] == c for r in vrows)]
        scen = sorted({r["scenario"] for r in vrows})
        lines.append("| case | all | " + " | ".join(scen) + " |")
        lines.append("|---|---|" + "---|" * len(scen))
        for c in cases:
            crows = [r for r in vrows if r["case"] == c]
            lines.append(f"| {c} | {_rate(crows)} | " + " | ".join(_rate([r for r in crows if r["scenario"] == s])
                                                                  for s in scen) + " |")
        lines.append("")
        kinds = sorted({r["on_kinds"] for r in vrows})
        lines.append("| case | " + " | ".join(f"on-path {k}" for k in kinds) + " | named mat | named cyl |")
        lines.append("|---|" + "---|" * (len(kinds) + 2))
        for c in cases:
            crows = [r for r in vrows if r["case"] == c]
            lines.append(f"| {c} | " + " | ".join(_rate([r for r in crows if r["on_kinds"] == k]) for k in kinds)
                         + f" | {_rate([r for r in crows if r['named_kind'] == 'mat'])}"
                         + f" | {_rate([r for r in crows if r['named_kind'] == 'cyl'])} |")
        lines.append("")
        for c in cases:
            bad = [r for r in vrows if r["case"] == c and not r["ok"]]
            if not bad:
                continue
            why = Counter(r["reason"] for r in bad)
            first = Counter()
            for r in bad:
                for k, v in (r["detail"].get("fails") or {}).items():
                    first[k] += v
                for k, v in (r["detail"].get("lanes") or {}).items():
                    if v != "ok":
                        first["lane " + v] += 1
            lines.append(f"- {c}, no route: {dict(why)}; failed checks: {dict(first.most_common(6))}")
        ms = [r["ms"] for r in vrows]
        lines.append(f"- planning time: mean {np.mean(ms):.0f} ms, max {np.max(ms):.0f} ms, "
                     f"total {np.sum(ms) / 1000:.1f} s")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
