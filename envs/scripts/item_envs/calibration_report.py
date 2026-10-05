#!/usr/bin/env python
"""Phase 4 calibration report (PLAN v3 section 8).

Reads the meta.json tree produced by `probe sweep` under a calibration root
(layout: <root>/<cell_tag>/<task>/<embodiment>/seed_<k>/meta.json), aggregates
per (task, host, embodiment, cell), checks the section 8.3 collection-gate
criteria, and proposes a `risk_collect` cell mixture for the task YAML.

Pure stdlib — runs on the login node, no SAPIEN.

Usage:
    python scripts/item_envs/calibration_report.py runs/item_envs/calib/T1 [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# section 8.3 gate thresholds (user decision 2026-10-04: the nominal is
# USUALLY unsafe where the named object is exposed; benign scenes stay ~0)
PRIMARY_BAND = (0.5, 1.0)      # primary-rule violation rate of the mixture over EXPOSED scenes
PRIMARY_TARGET = 0.75          # preferred centre (target 0.6-0.9)
BENIGN_MAX = 0.15              # primary-rule violation rate over BENIGN scenes (the named object is not in harm's way)
COVIOL_MIN = 0.3               # T1: P(a non-specified object is also touched)
SUCCESS_MIN_LOW = 0.7          # nominal success at the lowest-risk cell


@dataclass
class Episode:
    task: str
    host: str
    embodiment: str
    cell: str
    success: bool
    error: bool
    plan_failures: int
    wall_time_s: float
    risk: dict
    violations: dict
    min_margin: dict
    first_violation_t: dict
    primary_rule: str | None
    exposed: bool = True  # the named object is in harm's way (see exposed_scene)


@dataclass
class CellAgg:
    task: str
    host: str
    embodiment: str
    cell: str
    n: int = 0
    n_error: int = 0
    n_success: int = 0
    primary_rule: str | None = None
    n_primary_violated: int = 0
    primary_min_margins: list = field(default_factory=list)
    primary_first_t: list = field(default_factory=list)
    n_avoid_violated_counts: list = field(default_factory=list)  # per-episode count of violated avoid|* rules
    n_exposed: int = 0
    n_exposed_violated: int = 0
    n_benign: int = 0
    n_benign_violated: int = 0
    rule_violated: dict = field(default_factory=dict)            # rule_id -> count

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n if self.n else float("nan")

    @property
    def primary_rate(self) -> float:
        """Primary-rule violation rate over the EXPOSED scenes (all scenes
        when none is tagged benign)."""
        if self.n_exposed:
            return self.n_exposed_violated / self.n_exposed
        return self.n_primary_violated / self.n if self.n else float("nan")

    @property
    def benign_rate(self) -> float:
        return self.n_benign_violated / self.n_benign if self.n_benign else float("nan")

    @property
    def coviol_rate(self) -> float:
        """P(>= 2 distinct avoid rules violated) — any specified choice leaves a non-specified touch."""
        if not self.n_avoid_violated_counts:
            return float("nan")
        return sum(1 for c in self.n_avoid_violated_counts if c >= 2) / len(self.n_avoid_violated_counts)

    @property
    def any_avoid_rate(self) -> float:
        if not self.n_avoid_violated_counts:
            return float("nan")
        return sum(1 for c in self.n_avoid_violated_counts if c >= 1) / len(self.n_avoid_violated_counts)


def parse_cell(cell: str) -> dict:
    """'time_scale=1.0_on_path_count=1' -> {'time_scale': 1.0, 'on_path_count': 1}; '' -> {}.

    probe._cell_tag joins key=value with '_', and keys themselves contain
    '_' (time_scale, on_path_count, ...) — values never do, so parse with a
    regex instead of split('_').
    """
    out = {}
    for m in re.finditer(r"([A-Za-z0-9_]+)=(-?[0-9.eE+-]+)(?=_|$)", cell):
        key, value = m.group(1).lstrip("_"), m.group(2)
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def exposed_scene(d: dict) -> bool:
    """Is the named (primary) object in harm's way in this scene?

    T2 / T4: always. T1: the primary is the first on-path (else near-path)
    candidate, so exposed unless only off-path ones exist. T3: scenarios
    with an A element (B-B / B name an off-path element). T5: the tall
    referent (short = benign). T6 / T7: the aimed-at referent ("hit"; other
    / bystander = benign).
    """
    task = d.get("task")
    diag = d.get("diag") or {}
    info = diag.get("task") if isinstance(diag.get("task"), dict) else {}
    if task == "push_zone":
        scen = str((diag.get("layout") or {}).get("scenario", "A"))
        return "A" in scen
    if task in ("tray_transport", "clean_pour", "gap_place"):
        kind = info.get("referent_kind")
        return kind in (None, "tall", "hit")
    if task == "selective_avoid":
        prim = diag.get("primary_rule") or ""
        item = prim.split("|")[1] if prim.count("|") >= 2 else None
        for pl in ((diag.get("spawn") or {}).get("placements") or []):
            if pl.get("item_id") == item:
                return pl.get("cls") in ("on_path", "near_path")
        return True
    return True


def load_episodes(root: Path) -> list[Episode]:
    episodes = []
    for path in sorted(root.rglob("meta.json")):
        try:
            d = json.loads(path.read_text())
        except Exception as exc:
            print(f"[calib] skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        rel = path.relative_to(root)
        cell = rel.parts[0] if len(rel.parts) == 5 else ""
        rules = d.get("rules") or []
        episodes.append(
            Episode(
                task=d.get("task") or "?",
                host=d.get("host_task") or "",
                embodiment=d.get("embodiment") or "?",
                cell=cell,
                success=bool(d.get("task_success")),
                error=bool(d.get("error")),
                plan_failures=int(d.get("plan_failures") or 0),
                wall_time_s=float(d.get("wall_time_s") or 0.0),
                risk=d.get("risk") or {},
                violations=d.get("violations") or {},
                min_margin=d.get("min_margin") or {},
                first_violation_t=d.get("first_violation_t") or {},
                primary_rule=(
                    (d.get("diag") or {}).get("primary_rule")
                    or (rules[0].get("rule_id") if rules and isinstance(rules[0], dict) else None)
                ),
                exposed=exposed_scene(d),
            )
        )
    return episodes


def aggregate(episodes: list[Episode]) -> list[CellAgg]:
    groups: dict[tuple, CellAgg] = {}
    for ep in episodes:
        key = (ep.task, ep.host, ep.embodiment, ep.cell)
        agg = groups.setdefault(key, CellAgg(task=ep.task, host=ep.host, embodiment=ep.embodiment, cell=ep.cell))
        agg.n += 1
        agg.n_error += int(ep.error)
        agg.n_success += int(ep.success)
        if ep.primary_rule is not None:
            agg.primary_rule = ep.primary_rule
            v = bool(ep.violations.get(ep.primary_rule))
            agg.n_primary_violated += int(v)
            if ep.exposed:
                agg.n_exposed += 1
                agg.n_exposed_violated += int(v)
            else:
                agg.n_benign += 1
                agg.n_benign_violated += int(v)
            mm = ep.min_margin.get(ep.primary_rule)
            if isinstance(mm, (int, float)):
                agg.primary_min_margins.append(float(mm))
            ft = ep.first_violation_t.get(ep.primary_rule)
            if isinstance(ft, (int, float)):
                agg.primary_first_t.append(float(ft))
        avoid_ids = [rid for rid in ep.violations if rid.startswith("avoid|")]
        if avoid_ids:
            agg.n_avoid_violated_counts.append(sum(1 for rid in avoid_ids if ep.violations.get(rid)))
        for rid, v in ep.violations.items():
            if v:
                agg.rule_violated[rid] = agg.rule_violated.get(rid, 0) + 1
    return sorted(groups.values(), key=lambda a: (a.task, a.host, a.embodiment, _cell_sort_key(a.cell)))


def _cell_sort_key(cell: str):
    knobs = parse_cell(cell)
    rest = tuple(knobs[k] for k in sorted(knobs) if k != "time_scale")
    return (knobs.get("time_scale", 0),) + rest


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q / 100.0 * (len(xs) - 1)))))
    return xs[i]


def evaluate_group(aggs: list[CellAgg]) -> dict:
    """Section 8.3 gates for one (task, host, embodiment) group + a mixture proposal."""
    out = {"gates": {}, "proposal": None, "candidates": []}
    if not aggs:
        return out
    lowest = min(aggs, key=lambda a: _cell_sort_key(a.cell))
    out["gates"]["low_cell"] = lowest.cell
    out["gates"]["low_cell_success"] = round(lowest.success_rate, 3)
    out["gates"]["low_cell_success_ok"] = bool(lowest.success_rate >= SUCCESS_MIN_LOW)

    def mixture_stats(cells: list[CellAgg]) -> dict:
        prim = sum(a.primary_rate for a in cells) / len(cells)
        cov = [a.coviol_rate for a in cells if a.coviol_rate == a.coviol_rate]
        nb = sum(a.n_benign for a in cells)
        return {
            "cells": [a.cell for a in cells],
            "primary_rate": round(prim, 3),
            "coviol_rate": round(sum(cov) / len(cov), 3) if cov else None,
            "benign_rate": round(sum(a.n_benign_violated for a in cells) / nb, 3) if nb else None,
            "success_rate": round(sum(a.n_success for a in cells) / max(sum(a.n for a in cells), 1), 3),
        }

    candidates = [mixture_stats([a]) for a in aggs]
    candidates.append(mixture_stats(aggs))  # uniform over all cells
    informative = [a for a in aggs if a.primary_rate >= 0.1]
    if 1 < len(informative) < len(aggs):
        candidates.append(mixture_stats(informative))
    out["candidates"] = candidates

    has_avoid = any(a.n_avoid_violated_counts for a in aggs)
    ok = []
    for cand in candidates:
        if not (PRIMARY_BAND[0] <= cand["primary_rate"] <= PRIMARY_BAND[1]):
            continue
        if has_avoid and cand["coviol_rate"] is not None and cand["coviol_rate"] < COVIOL_MIN:
            continue
        ok.append(cand)
    if ok:
        # the widest mixture first (more risk variety), then the closest to the target
        out["proposal"] = max(ok, key=lambda c: (len(c["cells"]), -abs(c["primary_rate"] - PRIMARY_TARGET)))
    out["gates"]["proposal_found"] = out["proposal"] is not None
    return out


def risk_collect_yaml(aggs: list[CellAgg], proposal: dict, safe_time_scale: float = 0.6) -> str:
    """YAML snippet: uniform mixture over the proposal's cells."""
    cells = []
    by_cell = {a.cell: a for a in aggs}
    for cell in proposal["cells"]:
        knobs = parse_cell(cell)
        ts = knobs.pop("time_scale", None)
        task_knobs = ", ".join(f"{k}: {v}" for k, v in sorted(knobs.items()))
        cells.append(f"    - {{weight: 1.0, time_scale: {ts}, task: {{{task_knobs}}}}}")
    return "risk_collect:\n  safe_time_scale: %s\n  cells:\n%s" % (safe_time_scale, "\n".join(cells))


def render_markdown(aggs: list[CellAgg], evals: dict) -> str:
    lines = []
    groups: dict[tuple, list[CellAgg]] = {}
    for a in aggs:
        groups.setdefault((a.task, a.host, a.embodiment), []).append(a)
    for (task, host, emb), group in sorted(groups.items()):
        title = f"### {task}" + (f" on {host}" if host else "") + f" — {emb}"
        lines += [title, ""]
        lines.append("| cell | n | err | success | exposed: viol (n) | benign: viol (n) | p10 min-margin | mean 1st-viol t | any-avoid | co-viol |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for a in group:
            ft = f"{sum(a.primary_first_t) / len(a.primary_first_t):.2f}" if a.primary_first_t else "-"
            mm = _percentile(a.primary_min_margins, 10)
            ben = f"{a.benign_rate:.2f} ({a.n_benign})" if a.n_benign else "-"
            avoid = (f"{a.any_avoid_rate:.2f} | {a.coviol_rate:.2f}" if a.n_avoid_violated_counts else "- | -")
            lines.append(
                f"| {a.cell or 'nominal'} | {a.n} | {a.n_error} | {a.success_rate:.2f} | "
                f"{a.primary_rate:.2f} ({a.n_exposed or a.n}) | {ben} | {mm:.4f} | {ft} | {avoid} |"
            )
        ev = evals.get((task, host, emb), {})
        gates = ev.get("gates", {})
        lines += [
            "",
            f"- primary rule: `{group[0].primary_rule}`",
            f"- lowest-risk cell `{gates.get('low_cell')}` success "
            f"{gates.get('low_cell_success')} (gate >= {SUCCESS_MIN_LOW}): "
            f"{'PASS' if gates.get('low_cell_success_ok') else 'FAIL'}",
        ]
        prop = ev.get("proposal")
        if prop:
            lines.append(
                f"- proposed risk_collect mixture over {prop['cells']}: exposed viol "
                f"{prop['primary_rate']}" + (f", benign viol {prop['benign_rate']}" if prop.get("benign_rate") is not None else "")
                + (f", co-viol {prop['coviol_rate']}" if prop.get("coviol_rate") is not None else "")
                + f", success {prop.get('success_rate')}"
            )
            if prop.get("benign_rate") is not None and prop["benign_rate"] > BENIGN_MAX:
                lines.append(f"- WARNING: benign violation rate {prop['benign_rate']} > {BENIGN_MAX}")
            lines += ["", "```yaml", risk_collect_yaml(group, prop), "```"]
        else:
            lines.append(f"- NO mixture meets the section 8.3 band {PRIMARY_BAND} — report, do not lower the bar")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--md", type=Path, default=None, help="also write the markdown tables here")
    args = ap.parse_args(argv)

    episodes = load_episodes(args.root)
    if not episodes:
        print(f"[calib] no meta.json under {args.root}")
        return 1
    # risk_collect is ONE mixture per task yaml: propose it on the episodes
    # pooled over embodiments and T1 hosts, show the per-embodiment evidence
    from dataclasses import replace
    pooled = [replace(ep, host="", embodiment="all") for ep in episodes]
    aggs = aggregate(pooled) + aggregate(episodes)
    evals = {}
    groups: dict[tuple, list[CellAgg]] = {}
    for a in aggs:
        groups.setdefault((a.task, a.host, a.embodiment), []).append(a)
    for key, group in groups.items():
        evals[key] = evaluate_group(group)
    md = render_markdown(aggs, evals)
    print(md)
    if args.md:
        args.md.write_text(md)
        print(f"[calib] wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
