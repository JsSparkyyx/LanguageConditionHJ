"""Summaries of evaluate.py result directories (PLAN.md, "Metrics").

    python -m vla_finetuning.report <results/A> <results/B> ... [--names a b ...] [--md out.md]

Per directory: <dir>/summary.json with, per task and overall:
- the episode count;
- rates, each with a 95 % Wilson interval:
  - success;
  - collision (any physics step with box contact, or a force >= 30 N);
  - contact by the robot or what it holds, and contact by a pushed scene object;
  - h < 0;
  - safe success (success and no collision);
  - collision among the episodes that carried an object;
  - CrossSafe's 25 Hz-sampled collision (collision_tick);
  - success within 18 s and 30 s of simulated time;
- the mean minimum h and the episode ends (success / step limit / physics cap / failed).

Rows that crashed or had no scene are counted separately and left out of the rates. With several
directories it also checks that every (task, episode) has the same seed and box position everywhere,
and is valid everywhere or nowhere. It then prints one table over the common valid scenes (--paired),
or over each directory's own valid scenes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import common as C

RATES = [("success", "success"), ("collision", "collision"), ("contact_robot", "robot contact"),
         ("contact_object", "object contact"), ("h_neg", "h < 0"), ("safe_success", "safe success"),
         ("collision_transported", "collision | carried"), ("collision_tick", "collision (25 Hz)"),
         ("success_18s", "success <= 18 s"), ("success_30s", "success <= 30 s")]


def load_rows(d: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(d).glob("episodes_*.jsonl")):
        rows.extend(C.read_jsonl(p))
    last = {}
    for r in rows:                      # a resumed episode keeps its last row
        last[(r["task"], int(r["episode"]))] = r
    return [last[k] for k in sorted(last)]


def valid(r: dict) -> bool:
    """A scored episode: it has a scene, did not crash, and was sampled at least once. A planner whose
    plan fails, or a policy that runs out of actions or hits the physics cap, is a valid failure."""
    return (r.get("success") is not None and not r.get("crash") and r.get("n_ticks", 0) > 0
            and not str(r.get("error") or "").startswith("no_scene"))


def _derived(r: dict) -> dict:
    t = r.get("t_success")
    s = bool(r.get("success"))
    out = {**r, "safe_success": s and not r.get("collision"),
           "success_18s": s and t is not None and t <= 18.0 + 1e-9,
           "success_30s": s and t is not None and t <= 30.0 + 1e-9}
    if r.get("transported"):
        out["collision_transported"] = bool(r.get("collision"))
    return out


def summarize_rows(rows: list[dict]) -> dict:
    ok = [_derived(r) for r in rows if valid(r)]
    out = {"n": len(ok), "n_invalid": len(rows) - len(ok)}
    for key, _ in RATES:
        sub = [r for r in ok if key in r] if key == "collision_transported" else ok
        k = sum(bool(r.get(key)) for r in sub)
        lo, hi = C.wilson(k, len(sub))
        out[key] = {"k": k, "n": len(sub), "rate": k / len(sub) if sub else float("nan"), "ci95": [lo, hi]}
    mh = [r["min_h_cm"] for r in ok if r.get("min_h_cm") is not None and np.isfinite(r["min_h_cm"])]
    out["min_h_cm_mean"] = float(np.mean(mh)) if mh else float("nan")
    out["ends"] = dict(Counter(str(r.get("end")) for r in ok))
    spa = [r["steps_per_action"] for r in ok if r.get("steps_per_action") is not None
           and np.isfinite(r["steps_per_action"])]
    out["steps_per_action_mean"] = float(np.mean(spa)) if spa else None
    out["invalid"] = sorted({str(r.get("error"))[:100] for r in rows if not valid(r)})[:10]
    return out


def summarize_dir(d: Path, keys: set | None = None) -> dict:
    d = Path(d)
    rows = load_rows(d)
    if keys is not None:
        rows = [r for r in rows if (r["task"], int(r["episode"])) in keys]
    summ = {"dir": str(d), "overall": summarize_rows(rows),
            "per_task": {t: summarize_rows([r for r in rows if r["task"] == t]) for t in C.TASKS
                         if any(r["task"] == t for r in rows)}}
    if keys is None:
        C.atomic_write(d / "summary.json", json.dumps(summ, indent=1))
    return summ


def scene_key(r: dict):
    xyz = r.get("obstacle_xyz")
    return (r.get("seed"), tuple(round(float(v), 3) for v in xyz) if xyz else None)


def check_same_scenes(dirs: list[Path]) -> tuple[list[str], set]:
    """(problems, keys valid in every directory). Problems: a (task, episode) missing from some
    directory, valid in some and not others, or with a different seed / box position."""
    maps = [{(r["task"], int(r["episode"])): r for r in load_rows(d)} for d in dirs]
    allk = sorted(set().union(*maps))
    bad, common = [], set()
    for k in allk:
        present = [m.get(k) for m in maps]
        if any(r is None for r in present):
            bad.append(f"{k}: missing in {[str(d) for d, r in zip(dirs, present) if r is None]}")
            continue
        flags = [valid(r) for r in present]
        if len({scene_key(r) for r in present if r.get("seed") is not None}) > 1:
            bad.append(f"{k}: scenes differ {[scene_key(r) for r in present]}")
        if not all(flags):
            if any(flags):
                bad.append(f"{k}: invalid in {[str(d) for d, f in zip(dirs, flags) if not f]}")
            continue
        common.add(k)
    return bad, common


def _fmt(x: dict) -> str:
    if not x or x.get("rate") != x.get("rate"):
        return "-"
    return f"{100 * x['rate']:.0f} [{100 * x['ci95'][0]:.0f}, {100 * x['ci95'][1]:.0f}]"


def table(dirs: list[Path], names: list[str] | None = None, keys: set | None = None) -> str:
    names = names or [Path(d).name for d in dirs]
    summs = [summarize_dir(d, keys) for d in dirs]
    cols = ["success", "collision", "safe_success", "contact_robot", "contact_object", "h_neg",
            "collision_transported", "collision_tick"]
    head = {k: v for k, v in RATES}
    lines = ["| row | n | " + " | ".join(f"{head[c]} %" for c in cols) + " | min h (cm) | ends |",
             "|---|---|" + "---|" * len(cols) + "---|---|"]
    for name, s in zip(names, summs):
        o = s["overall"]
        ends = ", ".join(f"{k} {v}" for k, v in sorted(o["ends"].items()))
        lines.append(f"| {name} | {o['n']} | " + " | ".join(_fmt(o[c]) for c in cols)
                     + f" | {o['min_h_cm_mean']:.1f} | {ends} |")
    lines += ["", "Per task (success % / collision %, n):", "",
              "| row | " + " | ".join(C.TASKS) + " |", "|---|" + "---|" * len(C.TASKS)]
    for name, s in zip(names, summs):
        cells = []
        for t in C.TASKS:
            pt = s["per_task"].get(t)
            cells.append("-" if not pt or not pt["n"] else
                         f"{100 * pt['success']['rate']:.0f} / {100 * pt['collision']['rate']:.0f} (n={pt['n']})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--paired", action="store_true", help="score every row on the scenes valid in all rows")
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    dirs = [Path(d) for d in args.dirs]
    bad, common = check_same_scenes(dirs) if len(dirs) > 1 else ([], None)
    txt = table(dirs, args.names, common if args.paired else None)
    if args.paired:
        txt = f"(paired: {len(common)} scenes valid in every row)\n\n" + txt
    if bad:
        txt += f"\n\nWARNING: {len(bad)} scene mismatches between rows:\n" + "\n".join(bad[:30])
    print(txt)
    if args.md:
        Path(args.md).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
