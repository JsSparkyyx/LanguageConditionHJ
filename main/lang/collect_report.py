"""collect_report.md / .json (VANILLA_PLAN 7.2): the Phase 1 STOP-gate input.

    python -m main.lang.collect_report data/lang/warmup [--delay-runs runs/lang_recon/roles/_roles_runs]
        [--out docs/lang_filter/collect_report_warmup.md]

1. episodes per combo (and T1 host), skipped slots, status counts, success,
   episode length per task (mean, p95, max vs the cap);
2. per task and consequence: episodes with a violated rule (harm- vs
   benign-designated), rows with fails_soon set, mixed rows, sampleable rules
   per episode; flags tasks with harmful labels < 2 % or > 80 % of their
   sampleable rule-rows;
3. per consequence: h quantiles at t = 0 and over all ticks;
4. delay audit: time from the start of the risky segment containing (or
   preceding) a rule's first violation to that violation, from standalone
   probe traces (trace.npz time_scale + meta first_violation_t); flags medians
   above 0.4 s (10 ticks, the horizon of gamma = 0.9);
5. violation rates among non-ok episodes;
6. text cache: sentences per split, 20 random (rule, sentence) examples per task;
7. wall time per episode: rendering + encoding vs physics; disk use.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import REPO, ensure_item_path

ensure_item_path()

from .config import CANON_TASKS, TASK_SHORT, LangConfig  # noqa: E402

DELAY_FLAG_S = 0.4
QS = (1, 5, 25, 50, 75, 95, 99)


def _shards(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "item_buffer").is_dir():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "item_buffer").is_dir())


def _stale_bead(shard: Path, d: dict) -> bool:
    """A bead-task episode of a shard written before the bead-count h (no n_violating.npy): its labels hold
    the old bead h. Skipped like data.LangSource does (review B3)."""
    from .data import BEAD_TASKS

    return d.get("task") in BEAD_TASKS and not (shard / "item_buffer" / "n_violating.npy").exists()


def _lines(shard: Path, stale: list | None = None) -> list[dict]:
    """The shard's episode lines without stale bead episodes (counted into `stale` when given)."""
    p = shard / "lang_episodes.jsonl"
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if _stale_bead(shard, d):
                if stale is not None:
                    stale.append(d.get("episode_id"))
                continue
            out.append(d)
    return out


def _du(path: Path) -> float:
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.stat(os.path.join(dp, f)).st_blocks * 512
            except OSError:
                pass
    return tot / 2**30


def delay_audit(runs_dir: Path, max_per_task: int = 20) -> dict:
    """Per (task, consequence): delays (s) from the risky-segment start to the first violation."""
    from main.item_envs import language

    out = defaultdict(list)
    no_seg = Counter()
    per_task = Counter()
    for meta_p in sorted(glob.glob(str(Path(runs_dir) / "*" / "*" / "seed_*" / "meta.json"))):
        try:
            m = json.loads(Path(meta_p).read_text())
            z = np.load(meta_p.replace("meta.json", "trace.npz"))
        except Exception:
            continue
        task = m["task"]
        if per_task[(task, m["embodiment"])] >= max_per_task:
            continue
        per_task[(task, m["embodiment"])] += 1
        ts, t = np.asarray(z["time_scale"]), np.asarray(z["t"])
        risky = float((m.get("risk") or {}).get("time_scale", 1.0))
        safe = float((m.get("risk") or {}).get("safe_time_scale", 0.6))
        is_r = np.isclose(ts, risky) & (not np.isclose(risky, safe))
        starts = t[np.flatnonzero(is_r & ~np.r_[False, is_r[:-1]])]
        cons = {r["rule_id"]: language.SPEC_CONSEQUENCE.get(r["consequence"], r["consequence"]) for r in m["rules"]}
        # set / pair rules (several items, or inside / outside of several regions) are never named
        is_set = {r["rule_id"]: len(r.get("item_ids") or []) > 1
                  or (r["consequence"] in ("inside", "outside") and len(r.get("region_ids") or []) > 1)
                  for r in m["rules"]}
        for rid, fv in (m.get("first_violation_t") or {}).items():
            if fv is None or is_set.get(rid, True):
                continue
            prev = starts[starts <= fv + 1e-9]
            if prev.size == 0:
                no_seg[(task, cons.get(rid, "?"))] += 1
                continue
            out[(task, cons.get(rid, "?"))].append(float(fv - prev[-1]))
    res = {}
    for k in sorted(set(out) | set(no_seg), key=lambda k: (CANON_TASKS.index(k[0]) if k[0] in CANON_TASKS else 99, k[1])):
        v = np.asarray(out.get(k, []))
        res[f"{k[0]}|{k[1]}"] = {
            "n": int(v.size), "median_s": float(np.median(v)) if v.size else None,
            "p25_s": float(np.percentile(v, 25)) if v.size else None, "p75_s": float(np.percentile(v, 75)) if v.size else None,
            "no_risky_segment": int(no_seg.get(k, 0)),
            "flag": bool(v.size and np.median(v) > DELAY_FLAG_S),
        }
    return res


def build(root: Path, delay_runs: Path | None = None, text_cache: Path | None = None) -> tuple[dict, str]:
    from main.item_envs.item_buffer import ItemSidecarBuffer

    from .scene_roles import all_designations

    cfg = LangConfig()
    shards = _shards(root)
    stale = []
    lines = [(s, d) for s in shards for d in _lines(s, stale)]
    eps = [(s, d) for s, d in lines if not d.get("skipped")]
    R = {"root": str(root), "shards": len(shards), "episodes": len(eps),
         "skipped_slots": sum(1 for _, d in lines if d.get("skipped")),
         "stale_bead_episodes_skipped": len(stale)}
    if stale:
        print(f"[collect_report] skipped {len(stale)} bead-task episodes of shards without n_violating.npy (old bead h)")
    combo = Counter((d["task"], d["embodiment"]) for _, d in eps)
    hosts = Counter((d.get("host"), d["embodiment"]) for _, d in eps if d["task"] == "selective_avoid")
    status = Counter(d.get("status") for _, d in eps)
    R["status"] = dict(status)
    R["per_combo"] = {f"{t}|{e}": n for (t, e), n in sorted(combo.items())}
    R["per_t1_host"] = {f"{h}|{e}": n for (h, e), n in sorted(hosts.items(), key=str)}
    # 1. lengths and success
    by_task = defaultdict(list)
    succ = defaultdict(list)
    for _, d in eps:
        by_task[d["task"]].append(d["n_ticks"])
        succ[(d["task"], d["embodiment"])].append(bool(d.get("success")))
    R["length"] = {t: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": int(np.max(v)),
                       "cap": int(cfg.max_steps_per_episode), "at_cap": int(sum(x >= cfg.max_steps_per_episode for x in v))}
                   for t, v in by_task.items()}
    R["success"] = {f"{t}|{e}": float(np.mean(v)) for (t, e), v in succ.items()}
    # 2. violation by designation (episode level) + row-level labels from the sidecars
    viol = defaultdict(lambda: defaultdict(list))
    nonok = defaultdict(list)
    sampleable_per_ep = defaultdict(list)
    rows = defaultdict(lambda: {"rule_rows": 0, "harm_rule_rows": 0, "rows": 0, "mixed_rows": 0})
    hq0 = defaultdict(list)
    hq = defaultdict(list)
    for s in shards:
        try:
            side = ItemSidecarBuffer(s / "item_buffer")
        except Exception:
            continue
        A = side.arrays
        n = side.n
        ep_ids = np.asarray(A["episode_id"][:n])
        H = np.asarray(A["h_rules"][:n])
        FS = np.asarray(A["fails_soon"][:n])
        T0 = np.asarray(A["t_in_ep"][:n])
        recs = {}
        for f in (s / "item_buffer" / "episodes").glob("*.json"):
            recs[int(f.stem)] = json.loads(f.read_text())
        for d in _lines(s):
            if d.get("skipped"):
                continue
            rec = recs.get(int(d["episode_id"]))
            if rec is None:
                continue
            desig = d.get("designations") or {}
            cons = {r["rule_id"]: r["consequence"] for r in rec["rules"]}
            for rid, v in (d.get("violated") or {}).items():
                ds = desig.get(rid)
                if ds is None:
                    continue
                viol[(d["task"], cons.get(rid, "?"))][ds].append(bool(v))
                viol[(d["task"], cons.get(rid, "?"))]["all"].append(bool(v))   # ungrouped
                if d.get("status") != "ok":
                    nonok[(d["task"], ds)].append(bool(v))
            cols = [r["idx"] for r in rec["rules"] if r.get("sampleable")]
            sampleable_per_ep[d["task"]].append(len(cols))
            a, b = d["rows"]
            if b <= a or not cols:
                continue
            fs = FS[a:b][:, cols]
            st = rows[d["task"]]
            st["rows"] += b - a
            st["rule_rows"] += fs.size
            st["harm_rule_rows"] += int(fs.sum())
            st["mixed_rows"] += int((fs.any(1) & ~fs.all(1)).sum())
            for j in cols:
                c = rec["rules"][j]["consequence"]
                hq0[c].append(float(H[a, j]))
                hq[c].append(H[a:b, j])
    R["violation_by_designation"] = {
        f"{t}|{c}": {ds: {"rate": float(np.mean(v)), "n": len(v)} for ds, v in dd.items()}
        for (t, c), dd in sorted(viol.items(), key=lambda kv: (CANON_TASKS.index(kv[0][0]), kv[0][1]))}
    R["labels"] = {}
    flags = []
    for t, st in rows.items():
        hf = st["harm_rule_rows"] / max(st["rule_rows"], 1)
        R["labels"][t] = {"harmful_rule_row_frac": hf, "mixed_row_frac": st["mixed_rows"] / max(st["rows"], 1),
                          "rows": st["rows"], "sampleable_rules_per_episode": float(np.mean(sampleable_per_ep[t]))}
        if hf < 0.02 or hf > 0.8:
            flags.append(f"{t}: harmful labels are {100 * hf:.1f}% of sampleable rule-rows (outside 2-80%)")
    R["h_quantiles"] = {c: {"t0": [float(np.percentile(hq0[c], q)) for q in QS],
                            "all": [float(np.percentile(np.concatenate(hq[c]), q)) for q in QS]}
                        for c in sorted(hq)}
    R["violation_non_ok"] = {f"{t}|{ds}": {"rate": float(np.mean(v)), "n": len(v)} for (t, ds), v in sorted(nonok.items())}
    # 4. delay audit
    if delay_runs is not None and Path(delay_runs).exists():
        R["delay_audit"] = delay_audit(Path(delay_runs))
        flags += [f"delay {k}: median {v['median_s']:.2f} s > {DELAY_FLAG_S} s" for k, v in R["delay_audit"].items() if v["flag"]]
    # 6. text cache
    if text_cache is not None and Path(text_cache).exists():
        import torch

        c = torch.load(text_cache, map_location="cpu", weights_only=False)
        R["text_cache"] = {"sentences": len(c["index"]), "per_split": dict(Counter(c["split"]))}
    R["examples"] = sentence_examples(shards)
    # 7. wall time
    tim = defaultdict(lambda: defaultdict(list))
    for _, d in eps:
        ti = d.get("timing") or {}
        n = max(int(d.get("n_ticks", 0)), 1)
        for k in ("render", "encode", "h", "physics", "capture", "run", "build"):
            if k in ti:
                tim[d["task"]][k].append(ti[k] / (1 if k == "build" else n))
        tim[d["task"]]["wall_s"].append(d.get("wall_s", 0.0))
    R["timing_per_tick_ms"] = {t: {k: (float(np.mean(v)) * (1 if k in ("wall_s", "build") else 1000)) for k, v in dd.items()}
                               for t, dd in tim.items()}
    R["disk_gb"] = _du(Path(root))
    R["flags"] = flags
    return R, to_markdown(R)


def sentence_examples(shards, per_task: int = 20, seed: int = 0) -> dict:
    from main.item_envs import language

    rng = np.random.default_rng(seed)
    recs = defaultdict(list)
    for s in shards:
        mids = language.load_model_ids(s / "item_buffer")
        for f in sorted((s / "item_buffer" / "episodes").glob("*.json")):
            # re-named with today's words and bank, as training reads it: a rule whose sentences were removed
            # since (the spill rule, 2026-10-23) is no longer sampleable
            r = language.refresh_record(json.loads(f.read_text()), mids.get(int(f.stem)))
            recs[r["task"]].append(r)
    out = {}
    for t, rs in recs.items():
        ex = []
        for _ in range(per_task * 3):
            if len(ex) >= per_task:
                break
            rec = rs[int(rng.integers(len(rs)))]
            rules = [r for r in rec["rules"] if r.get("sampleable")]
            if not rules:
                continue
            rule = rules[int(rng.integers(len(rules)))]
            ex.append([rule["rule_id"], language.sample_sentence(rule, rec, rng, "train")])
        out[t] = ex
    return out


def to_markdown(R: dict) -> str:
    L = [f"# Collection report: `{R['root']}`", "",
         f"{R['episodes']} episodes in {R['shards']} shards, {R['skipped_slots']} skipped slots, "
         f"{R.get('stale_bead_episodes_skipped', 0)} bead-task episodes with the old bead h left out, "
         f"status {R['status']}, disk {R['disk_gb']:.1f} GB.", ""]
    if R["flags"]:
        L += ["**Flags**", ""] + [f"- {f}" for f in R["flags"]] + [""]
    L += ["## 1. Episodes, success, length", "", "| task | embodiment | episodes | success |", "|---|---|---|---|"]
    for k, n in R["per_combo"].items():
        t, e = k.split("|")
        L.append(f"| {TASK_SHORT.get(t, t)} {t} | {e} | {n} | {R['success'].get(k, float('nan')):.2f} |")
    L += ["", "T1 hosts: " + ", ".join(f"{k} {v}" for k, v in R["per_t1_host"].items()), "",
          "| task | mean ticks | p95 | max | cap | at cap |", "|---|---|---|---|---|---|"]
    for t, v in sorted(R["length"].items(), key=lambda kv: CANON_TASKS.index(kv[0])):
        L.append(f"| {t} | {v['mean']:.0f} | {v['p95']:.0f} | {v['max']} | {v['cap']} | {v['at_cap']} |")
    L += ["", "## 2. Labels", "", "Episodes with the rule violated, by designation and over all designated rules "
          "(no grouping) (rate, n):", "",
          "| task | consequence | harm | benign | unclear | all |", "|---|---|---|---|---|---|"]
    for k, dd in R["violation_by_designation"].items():
        t, c = k.split("|")
        f = lambda ds: f"{dd[ds]['rate']:.2f} ({dd[ds]['n']})" if ds in dd else "–"  # noqa: E731
        L.append(f"| {t} | {c} | {f('harm')} | {f('benign')} | {f('unclear')} | {f('all')} |")
    L += ["", "| task | harmful rule-rows | mixed rows | rows | sampleable rules / episode |", "|---|---|---|---|---|"]
    for t, v in sorted(R["labels"].items(), key=lambda kv: CANON_TASKS.index(kv[0])):
        L.append(f"| {t} | {100 * v['harmful_rule_row_frac']:.1f}% | {100 * v['mixed_row_frac']:.1f}% | {v['rows']} | "
                 f"{v['sampleable_rules_per_episode']:.1f} |")
    L += ["", "## 3. h quantiles (sampleable rules)", "", "| consequence | at t=0 (1/5/25/50/75/95/99 %) | all ticks |",
          "|---|---|---|"]
    for c, v in R["h_quantiles"].items():
        L.append(f"| {c} | {' '.join(f'{x:+.2f}' for x in v['t0'])} | {' '.join(f'{x:+.2f}' for x in v['all'])} |")
    if "delay_audit" in R:
        L += ["", "## 4. Delay audit (risky-segment start -> first violation)", "",
              "| task / consequence | n | median s | p25 | p75 | no risky segment before | flag (> 0.4 s) |",
              "|---|---|---|---|---|---|---|"]
        for k, v in R["delay_audit"].items():
            fm = lambda x: f"{x:.2f}" if x is not None else "–"  # noqa: E731
            L.append(f"| {k} | {v['n']} | {fm(v['median_s'])} | {fm(v['p25_s'])} | {fm(v['p75_s'])} | "
                     f"{v['no_risky_segment']} | {'**yes**' if v['flag'] else ''} |")
    if R["violation_non_ok"]:
        L += ["", "## 5. Violation among non-ok episodes", ""]
        L += [f"- {k}: {v['rate']:.2f} of {v['n']}" for k, v in R["violation_non_ok"].items()]
    if "text_cache" in R:
        L += ["", f"## 6. Text cache: {R['text_cache']['sentences']} sentences, {R['text_cache']['per_split']}"]
    L += ["", "Examples (rule id: sentence):", ""]
    for t, ex in R["examples"].items():
        L += [f"**{t}**", ""] + [f"- `{r}`: {s}" for r, s in ex] + [""]
    L += ["## 7. Wall time per tick (ms; build and wall in s per episode)", "",
          "| task | render | encode | h | physics | capture | build s | wall s |", "|---|---|---|---|---|---|---|---|"]
    for t, v in sorted(R["timing_per_tick_ms"].items(), key=lambda kv: CANON_TASKS.index(kv[0])):
        g = lambda k: f"{v[k]:.0f}" if k in v else "–"  # noqa: E731
        L.append(f"| {t} | {g('render')} | {g('encode')} | {g('h')} | {g('physics')} | {g('capture')} | "
                 f"{g('build')} | {g('wall_s')} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--delay-runs", default=str(REPO / "runs/lang_recon/roles/_roles_runs"))
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--out", default=None, help="markdown path (default <root>/collect_report.md)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    tc = Path(args.text_cache) if args.text_cache else root / "text_cache.pt"
    R, md = build(root, Path(args.delay_runs) if args.delay_runs else None, tc)
    out = Path(args.out) if args.out else root / "collect_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(R, indent=1, default=str))
    (root / "collect_report.json").write_text(json.dumps(R, indent=1, default=str))
    print(md[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
