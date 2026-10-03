"""probe.py — item-safety probe / calibration CLI (PLAN v3 §2.7).

Run from the project root (envs/):

    python -m main.item_envs.probe episode --task smoke_items --seed 0 \
        --embodiment ur5-wsg --risk time_scale=1.0 --video
    python -m main.item_envs.probe sweep --task smoke_items \
        --grid time_scale=0.5,0.75,1.0,1.25 --episodes 5 --adaptive \
        --render none --out runs/item_envs/calib/smoke
    python -m main.item_envs.probe physics --probe P2 --out runs/item_envs/feas/P2
    python -m main.item_envs.probe summarize runs/item_envs/calib/smoke

Sweeps spawn ONE CHILD PROCESS PER (task, embodiment) — the AGENTS.md
pattern: repeated Env creation leaks pybind11 TLS keys, so the parent
re-execs ``sweep --child --embodiment <emb>`` and respawns a failed child
up to 3 times. ``--adaptive`` adds --adaptive-episodes extra episodes to
cells whose primary-rule (first rules entry) violation rate falls in
[0.1, 0.6] (PLAN v3 §8.1).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ._bootstrap import ensure_main_path

ensure_main_path()  # runner/controller pull main.envs.* from the sibling CEHJ copy

from .registry import ITEM_TASKS, default_embodiment, get_task_entry

PROBE_IDS = ["P1", "P2", "P3", "P3-robot", "P5"]
MAX_CHILD_RETRIES = 3


def _parse_value(text: str):
    """'1.0' -> float, '1' -> int, else str."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_kv(pairs: list[str] | None) -> dict:
    """['time_scale=1.0', 'overshoot=0.04'] -> {'time_scale': 1.0, ...}."""
    out: dict = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=value, got {pair!r}")
        out[key.strip()] = _parse_value(value.strip())
    return out


def _parse_grid(pairs: list[str]) -> dict[str, list]:
    """['time_scale=0.5,0.75,1.0'] -> {'time_scale': [0.5, 0.75, 1.0]}."""
    out: dict[str, list] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=v1,v2,..., got {pair!r}")
        out[key.strip()] = [_parse_value(v.strip()) for v in value.split(",") if v.strip()]
    return out


def _grid_cells(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _cell_tag(cell: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in cell.items()).replace("/", "-")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_episode(args) -> int:
    from .runner import ItemEpisodeRunner

    risk = _parse_kv(args.risk) or None
    runner = ItemEpisodeRunner(
        args.task,
        embodiment=args.embodiment,
        seed=args.seed,
        host=args.host,
        risk=risk,
        active_rule_id=args.active_rule,
        render=args.render,
        video=args.video,
        out_root=args.out,
        max_steps=args.max_steps,
        task_cfg_path=args.task_cfg,
    )
    meta, error = runner.run()
    summary = {
        "out": str(runner.out_dir),
        "task_success": meta.task_success,
        "plan_failures": meta.plan_failures,
        "violations": meta.violations,
        "min_margin": meta.min_margin,
        "wall_time_s": round(meta.wall_time_s, 2),
        "error": error,
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sweep(args) -> int:
    if not args.child:
        return _sweep_parent(args)
    return _sweep_child(args)


def _sweep_parent(args) -> int:
    """One child process per (task, embodiment); respawn failures up to 3x."""
    entry = get_task_entry(args.task)
    embodiments = [args.embodiment] if args.embodiment else list(entry.embodiments)
    rc = 0
    for emb in embodiments:
        cmd = [
            sys.executable,
            "-m",
            "main.item_envs.probe",
            "sweep",
            "--child",
            "--task",
            args.task,
            "--embodiment",
            emb,
            "--episodes",
            str(args.episodes),
            "--seed-base",
            str(args.seed_base),
            "--render",
            args.render,
            "--out",
            args.out,
            "--max-steps",
            str(args.max_steps),
        ]
        if args.grid:
            cmd += ["--grid", *args.grid]
        if args.risk:
            cmd += ["--risk", *args.risk]
        if args.adaptive:
            cmd += ["--adaptive", "--adaptive-episodes", str(args.adaptive_episodes)]
        if args.video:
            cmd += ["--video"]
        if args.active_rule:
            cmd += ["--active-rule", args.active_rule]
        if args.host:
            cmd += ["--host", args.host]
        if args.task_cfg:
            cmd += ["--task-cfg", args.task_cfg]
        if getattr(args, "shard", None):
            cmd += ["--shard", args.shard]
        print(f"[sweep] child for embodiment {emb}: {' '.join(cmd)}")
        attempt = 0
        while True:
            attempt += 1
            proc = subprocess.run(cmd)
            if proc.returncode == 0:
                break
            if attempt >= MAX_CHILD_RETRIES:
                print(f"[sweep] child for {emb} FAILED {attempt}x (rc={proc.returncode})")
                rc = 1
                break
            print(f"[sweep] child for {emb} exited rc={proc.returncode}; respawn {attempt + 1}/{MAX_CHILD_RETRIES}")
    return rc


def _sweep_child(args) -> int:
    from .runner import ItemEpisodeRunner

    grid = _parse_grid(args.grid)
    base_risk = _parse_kv(args.risk)
    cells = _grid_cells(grid) if grid else [{}]
    if getattr(args, "shard", None):
        # cells[i::n] — n processes split one sweep (same --out, disjoint cell
        # dirs, so shards never write the same meta.json)
        i, n = (int(v) for v in args.shard.split("/"))
        cells = cells[i::n]
    total = len(cells) * args.episodes
    print(
        f"[sweep] {args.task}/{args.embodiment}: {len(cells)} cell(s) x "
        f"{args.episodes} episode(s) = {total} (render={args.render})"
    )
    n_fail = 0
    for cell in cells:
        tag = _cell_tag(cell) if cell else "nominal"
        cell_metas = []
        for ep in range(args.episodes):
            seed = args.seed_base + ep
            meta, error, runner = _run_one(args, cell, tag, seed)
            cell_metas.append(meta)
            if error is not None or not meta.task_success:
                n_fail += 1
        if args.adaptive:
            rate = _primary_violation_rate(cell_metas)
            if rate is not None and 0.1 <= rate <= 0.6:
                print(f"[sweep] {tag}: primary violation rate {rate:.2f} in [0.1, 0.6] "
                      f"-> +{args.adaptive_episodes} adaptive episodes")
                for ep in range(args.episodes, args.episodes + args.adaptive_episodes):
                    seed = args.seed_base + ep
                    meta, error, _ = _run_one(args, cell, tag, seed)
                    if error is not None or not meta.task_success:
                        n_fail += 1
    print(f"[sweep] done: {args.task}/{args.embodiment} under {args.out} ({n_fail} failed/unsuccessful)")
    return 0


def _run_one(args, cell: dict, tag: str, seed: int):
    from .runner import ItemEpisodeRunner

    # Resume: reuse a finished episode so chained/partitioned runs can rerun
    # the same command; corrupt or partial metas (killed mid-write) fall
    # through and are recomputed.
    meta_path = Path(args.out) / tag / args.task / args.embodiment / f"seed_{seed}" / "meta.json"
    if meta_path.exists():
        try:
            meta = SimpleNamespace(**json.loads(meta_path.read_text()))
            print(f"[sweep] {tag} seed={seed} resume: meta.json exists, skipping")
            return meta, None, None
        except Exception:
            pass

    base_risk = _parse_kv(args.risk)
    runner = ItemEpisodeRunner(
        args.task,
        embodiment=args.embodiment,
        seed=seed,
        host=args.host,
        risk={**base_risk, **cell} or None,
        active_rule_id=args.active_rule,
        render=args.render,
        video=args.video,
        out_root=Path(args.out) / tag,
        max_steps=args.max_steps,
        task_cfg_path=args.task_cfg,
    )
    meta, error = runner.run()
    status = "OK" if error is None else f"ERROR({error})"
    print(
        f"[sweep] {tag} seed={seed} success={meta.task_success} "
        f"plan_failures={meta.plan_failures} {status}"
    )
    return meta, error, runner


def _primary_violation_rate(metas) -> float | None:
    """Violation rate of the primary rule (first rules entry) over cell episodes."""
    if not metas:
        return None
    first = metas[0]
    if not first.rules:
        return None
    primary_id = first.rules[0]["rule_id"]
    return sum(1 for m in metas if (m.violations or {}).get(primary_id)) / len(metas)


def _parse_seeds(text: str) -> list[int]:
    """"0-49" -> range; "0,1,2" -> explicit list."""
    text = text.strip()
    if "-" in text and "," not in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",") if s.strip()]


def cmd_physics(args) -> int:
    from .probes import get_probe

    mod = get_probe(args.probe.lower().replace("-", "_"))
    out = Path(args.out or f"runs/item_envs/feas/{args.probe.upper()}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    kw: dict = {}
    for kv in args.kw or []:
        key, eq, val = kv.partition("=")
        if not eq:
            print(f"[physics] bad --kw {kv!r}, want key=value")
            return 2
        try:
            kw[key] = json.loads(val)
        except ValueError:
            kw[key] = val
    if args.embodiments:
        kw["embodiments"] = [e.strip() for e in args.embodiments.split(",") if e.strip()]
    if args.seeds:
        kw["seeds"] = _parse_seeds(args.seeds)
    if args.shard:
        kw["shard"] = args.shard
    result = mod.summarize(out) if args.summarize_only else mod.run(out, **kw)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_summarize(args) -> int:
    root = Path(args.dir)
    metas = sorted(root.rglob("meta.json"))
    if not metas:
        print(f"[summarize] no meta.json under {root}")
        return 1
    rows = []
    for path in metas:
        try:
            d = json.loads(path.read_text())
        except Exception as exc:
            print(f"[summarize] skipping unreadable {path}: {exc}")
            continue
        rel = path.relative_to(root)
        # layout: [cell/]<task>/<embodiment>/seed_k/meta.json
        cell = rel.parts[0] if len(rel.parts) == 5 else ""
        row = {
            "task": d.get("task"),
            "embodiment": d.get("embodiment"),
            "seed": d.get("seed"),
            "cell": cell,
            "task_success": d.get("task_success"),
            "plan_failures": d.get("plan_failures"),
            "spawn_rejections": d.get("spawn_rejections"),
            "wall_time_s": d.get("wall_time_s"),
            "error": d.get("error"),
        }
        risk = d.get("risk") or {}
        row["time_scale"] = risk.get("time_scale")
        row["safe_time_scale"] = risk.get("safe_time_scale")
        for key, value in (risk.get("task") or {}).items():
            row[f"risk.{key}"] = value
        for rid, value in (d.get("min_margin") or {}).items():
            row[f"min_margin|{rid}"] = value
        for rid, value in (d.get("violations") or {}).items():
            row[f"violated|{rid}"] = value
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summarize] wrote {csv_path} ({len(rows)} episodes)")

    # console aggregate: per (task, embodiment) success + violation rates
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["task"], row["embodiment"]), []).append(row)
    violation_cols = [c for c in fieldnames if c.startswith("violated|")]
    for (task, emb), group in sorted(groups.items()):
        n = len(group)
        n_success = sum(1 for r in group if r["task_success"])
        n_error = sum(1 for r in group if r["error"])
        rates = {
            c.split("|", 1)[1]: round(sum(1 for r in group if r.get(c)) / n, 3)
            for c in violation_cols
        }
        print(
            f"[summarize] {task}/{emb}: n={n} success={n_success / n:.2f} "
            f"errors={n_error} violation_rate={rates}"
        )

    try:
        _plot_summary(rows, root)
    except Exception as exc:
        print(f"[summarize] plots skipped ({exc})")
    return 0


def _plot_summary(rows, root: Path) -> None:
    """Violation rate and task success vs time_scale (best effort)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        r
        for r in rows
        if isinstance(r.get("time_scale"), (int, float)) and r.get("time_scale") is not None
    ]
    if not points:
        return
    scales = sorted({float(r["time_scale"]) for r in points})
    violation_cols = sorted({c for r in points for c in r if c.startswith("violated|")})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col in violation_cols:
        rates = [
            sum(1 for r in points if float(r["time_scale"]) == s and r.get(col))
            / max(1, sum(1 for r in points if float(r["time_scale"]) == s))
            for s in scales
        ]
        axes[0].plot(scales, rates, marker="o", label=col.split("|", 1)[1])
    axes[0].set_xlabel("time_scale")
    axes[0].set_ylabel("violation rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=6)
    success = [
        sum(1 for r in points if float(r["time_scale"]) == s and r.get("task_success"))
        / max(1, sum(1 for r in points if float(r["time_scale"]) == s))
        for s in scales
    ]
    axes[1].plot(scales, success, marker="o", color="tab:green")
    axes[1].set_xlabel("time_scale")
    axes[1].set_ylabel("task success")
    axes[1].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out = root / "summary.png"
    fig.savefig(out, dpi=120)
    print(f"[summarize] wrote {out}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.item_envs.probe", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--task", default="smoke_items", choices=sorted(ITEM_TASKS))
        p.add_argument("--embodiment", default=None,
                       help="default: episode = registry default; sweep parent = every registered embodiment")
        p.add_argument("--host", default=None, help="T1 host task (Phase 4)")
        p.add_argument("--risk", nargs="*", default=None,
                       help="key=value risk overrides, e.g. time_scale=1.0")
        p.add_argument("--active-rule", default=None, dest="active_rule")
        p.add_argument("--render", choices=["full", "none"], default="full")
        p.add_argument("--video", action="store_true")
        p.add_argument("--task-cfg", default=None, dest="task_cfg",
                       help="explicit task yaml path (overrides configs/tasks lookup)")
        p.add_argument("--max-steps", type=int, default=7500, dest="max_steps")

    p_ep = sub.add_parser("episode", help="run one episode")
    add_common(p_ep)
    p_ep.add_argument("--seed", type=int, default=0)
    p_ep.add_argument("--out", default="runs/item_envs")
    p_ep.set_defaults(func=cmd_episode)

    p_sw = sub.add_parser("sweep", help="risk-grid calibration sweep (one child per embodiment)")
    add_common(p_sw)
    p_sw.add_argument("--grid", nargs="*", default=None,
                      help="key=v1,v2,... grid dimensions, e.g. time_scale=0.5,0.75,1.0")
    p_sw.add_argument("--episodes", type=int, default=20)
    p_sw.add_argument("--adaptive", action="store_true",
                      help="add --adaptive-episodes to cells with primary violation rate in [0.1, 0.6]")
    p_sw.add_argument("--adaptive-episodes", type=int, default=15, dest="adaptive_episodes")
    p_sw.add_argument("--seed-base", type=int, default=0, dest="seed_base")
    p_sw.add_argument("--out", default="runs/item_envs/calib")
    p_sw.add_argument("--shard", default=None,
                      help="'i/n' — run grid cells[i::n] (parallel sweep processes)")
    p_sw.add_argument("--child", action="store_true",
                      help=argparse.SUPPRESS)  # internal: the parent re-execs this
    p_sw.set_defaults(func=cmd_sweep)

    p_ph = sub.add_parser("physics", help="physics feasibility probes (Phase 3)")
    p_ph.add_argument("--probe", required=True, choices=PROBE_IDS)
    p_ph.add_argument("--out", default=None, help="default runs/item_envs/feas/<PROBE>")
    p_ph.add_argument("--embodiments", default=None, help="comma list, e.g. ur5-wsg,franka-panda")
    p_ph.add_argument("--seeds", default=None, help="'0-49' or '0,1,2'")
    p_ph.add_argument("--shard", default=None, help="'i/n' — run cells[i::n]")
    p_ph.add_argument("--kw", nargs="*", default=None,
                      help="extra probe kwargs as key=value (JSON-parsed when possible)")
    p_ph.add_argument("--summarize-only", action="store_true", dest="summarize_only")
    p_ph.set_defaults(func=cmd_physics)

    p_su = sub.add_parser("summarize", help="aggregate meta.json tree -> CSV + plots")
    p_su.add_argument("dir")
    p_su.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Env() chdirs to the RoboTwin root: resolve --out against the launch cwd
    # before any env exists (sweep resume checks + child re-exec read it too)
    if getattr(args, "out", None):
        args.out = str(Path(args.out).resolve())
    if args.cmd == "episode" and getattr(args, "embodiment", None) is None:
        args.embodiment = default_embodiment(args.task)
    if args.cmd == "sweep" and getattr(args, "child", False) and getattr(args, "embodiment", None) is None:
        args.embodiment = default_embodiment(args.task)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
