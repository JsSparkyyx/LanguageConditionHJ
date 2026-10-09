"""Section 5 table and the E4 decision for the grounding runs (review round 3, Part II E2 / E4).

    python scripts/lang/grounding_summary.py --runs runs/lang/grounding_1008/g1008_v*_s* \
        --out-name grounding --md runs/lang/grounding_1008/summary.md

Reads each run's eval_offline/<out-name>/rows.json.gz (every (state, rule) row, scored on the same validation
states: the pairs of all runs are the same pairs). Per variant and pair the seeds' outcomes (1 right, 0.5 tie,
0 wrong) are averaged, then rank accuracy per group with an episode-cluster bootstrap 95% interval of that
seed mean; each seed's own value is listed too. Groups: referent pairs (same consequence, other object) and
consequence pairs (same object, other consequence), T1 + T7 pooled and per task, already harmed / not yet;
the sentence-only prior on the same pairs. Decision (Yuxuan, 2026-10-08, decision 5; replaces E4): primary =
referent, already harmed, T1 + T7 pooled. a (V1 / V2) or b (V4, the auxiliary target loss) >= 0.85: keep the
vanilla model with that change. Only if neither does but V3 (the oracle tag) does: propose slots plus a pointer.
V4 is run only if a does not fix it (on the better of V1 / V2 if either helped, else on V0).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LABELS = {"v0": "V0 baseline", "v1": "V1 words normalized", "v2": "V2 V1 + [CLS] / [SEP] hidden",
          "v3": "V3 oracle tag (ceiling)", "v4": "V4 auxiliary target loss",
          "v4b": "V4b auxiliary attention loss"}
MIN_RATIO = 0.05   # an auxiliary loss below 5% of the critic loss at ~500 steps was not really tried (round 4, item 3)


def pair_outcomes(rows: list[dict], ax: str = "train") -> dict:
    """{pair key: (episode cluster, task, state, kind, model ok, prior ok)} over the symmetric pairs of `ax`."""
    from main.lang.eval_offline import _ep, _pairs, _rank, _ttv_bin

    out = {}
    for hr, bn in _pairs(rows):
        if not (hr["eligible"][ax] and bn["eligible"][ax]) or hr.get("censored"):
            continue
        b = _ttv_bin(hr.get("ttv"))
        if b is None:
            continue
        same_cons = hr["consequence"] == bn["consequence"]
        same_ref = hr["referent"] == bn["referent"] and hr["referent"] is not None
        kind = "referent" if (same_cons and not same_ref) else "consequence" if (same_ref and not same_cons) else None
        if kind is None:
            continue
        ph, pb = (hr.get("prior") or {}).get(ax), (bn.get("prior") or {}).get(ax)
        key = (hr["src"], hr["i"], hr["rule"], bn["rule"])
        out[key] = (_ep(hr), hr["task"], "already_harmed" if b == "violated" else "not_yet", kind,
                    _rank(hr["q"][ax], bn["q"][ax]), _rank(-ph, -pb) if ph is not None and pb is not None else None)
    return out


FIXED = ("gamma", "actor_hidden", "tf32_train", "steps", "tasks", "embodiments")   # held equal to V0's (round 4, 2)
SCORED = ("val_roots", "scored_tasks", "scored_embodiments")   # ... and the scored set (audit 2026-10-08)
PARTS = {"oracle_ref_tag": "e_tag", "aux_target_loss": "aux_head"}                   # what a switch's checkpoint needs
RATIO_WINDOW = (400, 600)   # the same steps as scripts/lang/aux_calibrate.py
PRIMARY_TASKS = ("selective_avoid", "gap_place")
TAG_FLIP_MIN = 0.05   # V3 counts as a ceiling only if zeroing its tag changes >= 5% of the primary outcomes


def _ckpt_keys(f: Path) -> tuple[set, dict]:
    """(top-level keys, identity) of a checkpoint file (memory-mapped: no full read)."""
    import torch

    if not f.exists():
        return set(), {}
    ck = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    return set(ck), dict(ck.get("identity") or {})


def _log(run: Path) -> list[dict]:
    out = []
    f = run / "train_log.jsonl"
    if f.exists():
        for ln in f.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def train_ratio(run: Path, key: str, window=RATIO_WINDOW) -> tuple[float | None, int, float | None]:
    """(median of `key` over the logged steps in `window`, how many logs, the last logged value)."""
    sel, last = [], None
    for d in _log(run):
        if d.get(key) is None:
            continue
        last = float(d[key])
        if window[0] <= int(d.get("step", -1)) <= window[1]:
            sel.append(float(d[key]))
    return (float(np.median(sel)) if sel else None), len(sel), last


def train_pair_acc(run: Path, last: int = 1000) -> float | None:
    """Median in-batch pair rank accuracy (lang/pair_rank_acc) over the last `last` logged steps: how well the
    model ranks the pairs of its TRAINING batches."""
    L = [d for d in _log(run) if d.get("lang/pair_rank_acc") is not None]
    if not L:
        return None
    top = max(int(d.get("step", 0)) for d in L)
    return float(np.median([d["lang/pair_rank_acc"] for d in L if int(d.get("step", 0)) > top - last]))


def run_settings(run: Path, out_name: str) -> dict:
    """What a comparison holds fixed and what each variant changes (round 4, item 2), from the run's
    lang_config.json, its scored metrics.json and (for the commit and the parts) the scored checkpoint."""
    cfg = json.loads((run / "lang_config.json").read_text())
    mj = run / "eval_offline" / out_name / "metrics.json"
    md = json.loads(mj.read_text()) if mj.exists() else {}
    data = md.get("data") or {}
    keys, ident = _ckpt_keys(run / (md.get("ckpt") or "checkpoint_final.pt"))
    if data.get("ckpt_parts") is not None:   # recorded at scoring time: the parts the scored weights had
        keys = {"blocks" if k == "trunk" else k for k in data["ckpt_parts"]} | {"identity"}
    out = {"gamma": cfg.get("gamma"), "actor_hidden": int(cfg.get("actor_hidden", 0) or 0),
           "tf32_train": bool(cfg.get("tf32_train", False)), "steps": md.get("step"),
           "tasks": tuple(cfg.get("task_choices") or ()), "embodiments": tuple(cfg.get("embodiment_choices") or ()),
           "recall_ticks": cfg.get("recall_horizon_ticks"),
           "val_roots": tuple(data.get("val_roots") or ()), "scored_tasks": tuple(data.get("tasks") or ()),
           "scored_embodiments": tuple(data.get("embodiments") or ()), "ckpt": md.get("ckpt"),
           "switches": [k for k in ("text_out_norm", "mask_special_tokens", "oracle_ref_tag", "aux_target_loss",
                                    "aux_attention_loss") if cfg.get(k)],
           "aux": {k: cfg.get(k) for k in ("aux_loss_weight", "aux_unit", "aux_huber_delta", "aux_attention_weight")
                   if k in cfg}, "git_head": (ident.get("git_head") or "")[:10], "parts": sorted(keys),
           "train_pair_acc": train_pair_acc(run)}
    out["missing_parts"] = [part for sw, part in PARTS.items() if cfg.get(sw) and part not in keys]
    if cfg.get("aux_target_loss"):
        out["aux_ratio_500"], out["aux_ratio_n"], out["aux_ratio_last"] = train_ratio(run, "aux/ratio")
    if cfg.get("aux_attention_loss"):
        out["att_ratio_500"], out["att_ratio_n"], out["att_ratio_last"] = train_ratio(run, "aux/att_ratio")
    return out


def _rows(f: Path) -> list[dict]:
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def primary_of(outcomes: dict) -> tuple[float, int]:
    """(rank accuracy, pairs) on the primary set of one seed's pair outcomes."""
    sel = [o[4] for o in outcomes.values() if o[3] == "referent" and o[2] == "already_harmed" and o[1] in PRIMARY_TASKS]
    return (float(np.mean(sel)) if sel else float("nan")), len(sel)


def tag_use(with_tag: dict, no_tag: dict) -> dict:
    """V3 with its tag vs the same checkpoint with the tag zeroed, on the shared primary pairs."""
    ks = sorted(k for k, o in with_tag.items() if k in no_tag and o[3] == "referent" and o[2] == "already_harmed"
                and o[1] in PRIMARY_TASKS)
    if not ks:
        return {}
    flips = sum(with_tag[k][4] != no_tag[k][4] for k in ks)
    return {"n_pairs": len(ks), "flip_frac": flips / len(ks), "acc_tag": float(np.mean([with_tag[k][4] for k in ks])),
            "acc_no_tag": float(np.mean([no_tag[k][4] for k in ks]))}


def boot(obs, B: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    from main.lang.eval_offline import boot_mean_ci

    if not obs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = boot_mean_ci(obs, B=B, seed=seed)
    return float(np.mean([v for _, v in obs])), lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-name", default="grounding")
    ap.add_argument("--md", required=True)
    args = ap.parse_args(argv)
    runs = sorted({Path(p) for s in args.runs for p in glob.glob(s) if Path(p).is_dir()})
    by_var = defaultdict(dict)   # variant -> seed -> pair outcomes
    meta, settings, refused = {}, {}, {}
    extra = defaultdict(dict)    # run -> {"train": primary on training episodes, "tag_use": ..., "curve": {step: acc}}
    found, pending = [], []
    for r in runs:
        m = re.search(r"_(v\d[a-z]?)(?:_[a-z0-9]+)?_s(\d+)$", r.name)
        if not m:
            continue
        f = r / "eval_offline" / args.out_name / "rows.json.gz"
        if f.exists():
            found.append((r, m.group(1), int(m.group(2)), f))
        else:
            pending.append(r.name)
    ref = next((x for x in found if x[1] == "v0"), None)
    ref_set = run_settings(ref[0], args.out_name) if ref else None
    for r, v, seed, f in found:
        st = run_settings(r, args.out_name)
        settings[r.name] = st
        why = [f"{k} {st[k]} != V0's {ref_set[k]}" for k in FIXED + SCORED if ref_set is not None and st[k] != ref_set[k]]
        if st["missing_parts"]:   # e.g. a V3 checkpoint without its tag was scored with a zero one (round 4, item 1)
            why.append(f"checkpoint lacks {st['missing_parts']}")
        if why:
            refused[r.name] = why
            print(f"[grounding_summary] refused {r.name}: {'; '.join(why)}", flush=True)
            continue
        by_var[v][seed] = pair_outcomes(_rows(f))
        meta[r.name] = st["steps"]
        ev = r / "eval_offline"
        tr = ev / f"{args.out_name}_train" / "rows.json.gz"       # the same scoring on training episodes
        if tr.exists():
            extra[r.name]["train"] = primary_of(pair_outcomes(_rows(tr)))
        nt = ev / f"{args.out_name}_notag" / "rows.json.gz"       # V3 with its tag zeroed
        if nt.exists():
            extra[r.name]["tag_use"] = tag_use(by_var[v][seed], pair_outcomes(_rows(nt)))
        for d in sorted(ev.glob(f"{args.out_name}_step*")):        # validation learning curve
            if (d / "rows.json.gz").exists():
                k = int(re.sub(r"\D", "", d.name.rsplit("_step", 1)[1]) or 0)
                extra[r.name].setdefault("curve", {})[k] = primary_of(pair_outcomes(_rows(d / "rows.json.gz")))
    groups = [("T1 + T7", PRIMARY_TASKS), ("T1", ("selective_avoid",)), ("T7", ("gap_place",))]
    res = {}
    for v, seeds in sorted(by_var.items()):
        keys = sorted(set.intersection(*[set(p) for p in seeds.values()]))   # sorted: the same bootstrap every run
        first = seeds[next(iter(seeds))]
        res[v] = {"seeds": sorted(seeds)}
        for kind in ("referent", "consequence"):
            for state in ("already_harmed", "not_yet"):
                for gname, tasks in groups:
                    sel = [k for k in keys if first[k][3] == kind and first[k][2] == state and first[k][1] in tasks]
                    mean_obs = [(first[k][0], float(np.mean([seeds[s][k][4] for s in seeds]))) for k in sel]
                    prior_obs = [(first[k][0], first[k][5]) for k in sel if first[k][5] is not None]
                    per_seed = {s: float(np.mean([seeds[s][k][4] for k in sel])) if sel else float("nan") for s in seeds}
                    res[v][f"{kind}/{state}/{gname}"] = {"n_pairs": len(sel), "mean": boot(mean_obs),
                                                         "prior": boot(prior_obs), "per_seed": per_seed,
                                                         "n_episodes": len({first[k][0] for k in sel})}
    # E4
    def prim(v):
        return res.get(v, {}).get("referent/already_harmed/T1 + T7", {}).get("mean", (float("nan"),) * 3)

    def runs_of(v):
        return [n for n in settings if n not in refused and re.search(rf"_{v}(?:_[a-z0-9]+)?_s\d+$", n)]

    def weak(v):   # its auxiliary loss was too small to count as a try (any seed's median ratio below MIN_RATIO)
        key = "att_ratio_500" if v == "v4b" else "aux_ratio_500"
        return [settings[n].get(key) for n in runs_of(v) if settings[n].get(key) is None or settings[n][key] < MIN_RATIO]

    v0, v1, v2, v3, v4, v4b = (prim(v) for v in ("v0", "v1", "v2", "v3", "v4", "v4b"))
    nan = lambda x: x[0] != x[0]  # noqa: E731
    ok = lambda x: not nan(x) and x[0] >= 0.85  # noqa: E731
    best_a = max((("v1", v1), ("v2", v2)), key=lambda kv: kv[1][0] if not nan(kv[1]) else -1)
    tried_b = [v for v, x in (("v4", v4), ("v4b", v4b)) if not nan(x) and not weak(v)]
    tags = [extra[n]["tag_use"] for n in runs_of("v3") if extra[n].get("tag_use")]
    tag_flip = float(np.mean([t["flip_frac"] for t in tags])) if tags else None
    v4b_waiting = any("_v4b_" in n for n in pending)
    oracle_used = (tag_flip is None or tag_flip >= TAG_FLIP_MIN) and not nan(v3) and v3[0] > v0[2]
    if ok(v1) or ok(v2):
        status, outcome = "final_a", f"a fixes it: keep the vanilla model with {LABELS[best_a[0]]}"
    elif ok(v4) or ok(v4b):
        bv = "v4" if ok(v4) and (not ok(v4b) or v4[0] >= v4b[0]) else "v4b"
        status, outcome = "final_b", f"b fixes it: keep the vanilla model with {LABELS[bv]}"
    elif "v4" not in tried_b:
        helped = not nan(best_a[1]) and best_a[1][0] > v0[2]   # above V0's interval
        why = (f"V4's auxiliary loss was below {MIN_RATIO:.2f} of the critic loss, so it does not count"
               if not nan(v4) else "V4 not scored yet")
        status = "pending"
        outcome = (f"a does not fix it; {why}: run V4 on {LABELS[best_a[0]] if helped else LABELS['v0']} "
                   f"({'it helped' if helped else 'neither V1 nor V2 is above V0'})")
    elif v4b_waiting:
        status, outcome = "pending", "neither the text fixes nor V4 reach 0.85; V4b still running"
    elif nan(v3):
        status, outcome = "pending", "neither the text fixes nor V4 reach 0.85; V3 (the oracle) not scored: no conclusion on slots yet"
    elif not oracle_used:
        status = "inconclusive_oracle"
        outcome = ("neither the text fixes nor V4 reach 0.85, and the oracle tag did not work as a ceiling "
                   f"(V3 {v3[0]:.2f} vs V0 {v0[0]:.2f} [{v0[1]:.2f}, {v0[2]:.2f}]"
                   + (f"; zeroing the tag changes {tag_flip:.0%} of the primary outcomes" if tag_flip is not None else "")
                   + "): the decision rule cannot choose between slots plus a pointer and other causes")
    elif ok(v3):
        status, outcome = "final_slots", "neither the text fixes nor V4 reach 0.85 but the oracle does: propose slots plus a pointer"
    else:
        status = "final_other"
        outcome = "neither the text fixes, V4 nor the oracle reach 0.85: the bottleneck is not (only) referent grounding"
    res["_e4"] = {"primary": {k: x for k, x in zip(("v0", "v1", "v2", "v3", "v4", "v4b"), (v0, v1, v2, v3, v4, v4b))},
                  "outcome": outcome, "status": status, "best_a": best_a[0], "tag_flip": tag_flip,
                  "weak_aux": {v: weak(v) for v in ("v4", "v4b") if weak(v)}, "pending_runs": pending}
    res["_settings"] = {n: {k: (list(x) if isinstance(x, tuple) else x) for k, x in st.items() if k != "parts"}
                        for n, st in settings.items()}
    res["_refused"] = refused
    res["_extra"] = {n: {k: ({str(s): a for s, a in x.items()} if k == "curve" else x) for k, x in e.items()}
                     for n, e in extra.items()}
    f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if t[0] == t[0] else "–"  # noqa: E731
    L = ["# Grounding runs: summary (review round 3, Part II E2 / E4)", "",
         f"Runs: {', '.join(f'{k} (step {s})' for k, s in sorted(meta.items()))}. Cells: rank accuracy, the seed mean "
         "[episode-bootstrap 95% interval] (each seed in brackets after it); prior = the sentence-only reader on "
         "the same pairs. Referent pairs: same consequence, other object; consequence pairs: same object, other "
         "consequence.", ""]
    for kind in ("referent", "consequence"):
        L += [f"## {kind.capitalize()} pairs", "",
              "| run | " + " | ".join(f"{g} {s.replace('_', ' ')}" for s in ("already_harmed", "not_yet") for g, _ in groups) + " |",
              "|---|" + "---|" * 6]
        for v in sorted(by_var):
            cells = []
            for state in ("already_harmed", "not_yet"):
                for gname, _ in groups:
                    c = res[v][f"{kind}/{state}/{gname}"]
                    seeds_s = " / ".join(f"{x:.2f}" for x in c["per_seed"].values())
                    cells.append(f"{f(c['mean'])} ({seeds_s}; {c['n_pairs']})")
            L.append(f"| {LABELS.get(v, v)} | " + " | ".join(cells) + " |")
        c0 = res[sorted(by_var)[0]] if by_var else {}
        L.append("| sentence prior | " + " | ".join(f(c0.get(f"{kind}/{s}/{g}", {}).get("prior", (float('nan'),) * 3))
                                                   for s in ("already_harmed", "not_yet") for g, _ in groups) + " |")
        L.append("")
    r2 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
    L += ["## Run settings (round 4, item 2)", "",
          "Held equal to V0's: " + ", ".join(FIXED + SCORED) + ". aux ratio = lambda x L_aux / L_critic from "
          f"train_log.jsonl, the median over steps {RATIO_WINDOW[0]}-{RATIO_WINDOW[1]} (and the last one); below "
          f"{MIN_RATIO} the auxiliary loss does not count. recall = recall_horizon_ticks (diagnostics only).", "",
          "| run | commit | gamma | actor_hidden | tf32_train | steps | recall | tasks | embodiments | switches | "
          "aux settings | aux ratio (n) / last | att ratio (n) / last | refused |", "|---|" + "---|" * 13]
    for n, st in sorted(settings.items()):
        aux = ", ".join(f"{k.replace('aux_', '')} {x}" for k, x in st["aux"].items()) if any(
            sw.startswith("aux") for sw in st["switches"]) else "–"
        L.append(f"| {n} | {st['git_head'] or '?'} | {st['gamma']} | {st['actor_hidden']} | {st['tf32_train']} | "
                 f"{st['steps']} | {st['recall_ticks'] if st['recall_ticks'] is not None else 'derived'} | "
                 f"{', '.join(st['tasks'])} | {', '.join(st['embodiments'])} | {', '.join(st['switches']) or '–'} | {aux} | "
                 f"{r2(st.get('aux_ratio_500'))} ({st.get('aux_ratio_n', 0)}) / {r2(st.get('aux_ratio_last'))} | "
                 f"{r2(st.get('att_ratio_500'))} ({st.get('att_ratio_n', 0)}) / {r2(st.get('att_ratio_last'))} | "
                 f"{'; '.join(refused.get(n, [])) or '–'} |")
    L.append("")
    L += ["## Training versus validation (audit 2026-10-08)", "",
          "Primary metric (referent, already harmed, T1 + T7) per run on the validation episodes, on training "
          f"episodes scored the same way (eval_offline/{args.out_name}_train, if scored), and the in-batch pair rank "
          "accuracy of the last 1000 training steps (lang/pair_rank_acc: the training batches' own pairs, mixed "
          "kinds and states). A large gap = the model fits its training scenes without reading the name in a way "
          "that carries to new scenes.", "",
          "| run | validation | training episodes (pairs) | in-batch training pairs |", "|---|---|---|---|"]
    for n in sorted(meta):
        v, s_ = re.search(r"_(v\d[a-z]?)(?:_[a-z0-9]+)?_s(\d+)$", n).groups()
        val_acc = primary_of(by_var[v][int(s_)])[0]
        tr = extra[n].get("train")
        L.append(f"| {n} | {val_acc:.3f} | {f'{tr[0]:.3f} ({tr[1]})' if tr else '–'} | "
                 f"{r2(settings[n].get('train_pair_acc'))} |")
    L.append("")
    if tags:
        L += ["## Does V3 read its tag? (audit 2026-10-08)", "",
              f"The same V3 checkpoint scored with its learned tag zeroed (eval_offline/{args.out_name}_notag). Under "
              f"{TAG_FLIP_MIN:.0%} of primary outcomes changing, the oracle is not read and V3 is no ceiling.", "",
              "| run | primary pairs | outcomes changed | accuracy with tag | with the tag zeroed |", "|---|---|---|---|---|"]
        for n in runs_of("v3"):
            t = extra[n].get("tag_use")
            if t:
                L.append(f"| {n} | {t['n_pairs']} | {t['flip_frac']:.1%} | {t['acc_tag']:.3f} | {t['acc_no_tag']:.3f} |")
        L.append("")
    curves = {n: e["curve"] for n, e in extra.items() if e.get("curve")}
    if curves:
        steps = sorted({k for c in curves.values() for k in c})
        L += ["## Validation learning curve (primary metric by checkpoint)", "",
              "| run | " + " | ".join(str(k) for k in steps) + " | final |", "|---|" + "---|" * (len(steps) + 1)]
        for n, c in sorted(curves.items()):
            v, s_ = re.search(r"_(v\d[a-z]?)(?:_[a-z0-9]+)?_s(\d+)$", n).groups()
            L.append(f"| {n} | " + " | ".join(f"{c[k][0]:.3f}" if k in c else "–" for k in steps) +
                     f" | {primary_of(by_var[v][int(s_)])[0]:.3f} |")
        L.append("")
    L += ["## Decision (decision 5)", "", "Primary: referent, already harmed, T1 + T7 pooled: " +
          ", ".join(f"{k} {f(x)}" for k, x in res["_e4"]["primary"].items()) + f". Outcome ({status}): **{outcome}**."
          + (f" Not scored yet: {', '.join(pending)}." if pending else "")
          + (f" Left out: {', '.join(refused)}." if refused else ""), ""]
    Path(args.md).write_text("\n".join(L) + "\n")
    Path(args.md).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
