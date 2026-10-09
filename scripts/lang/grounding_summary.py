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

E5 (FUSION_PLAN section 4, the V5 short runs), on the version-2 validation set:

    python scripts/lang/grounding_summary.py --runs 'runs/lang/fusion_1009/e5_v*_s*' --reference v0p \
        --exempt v5d2:embodiments --e4 runs/lang/fusion_1009/e4/fusion_probe.json \
        --out-name grounding --md runs/lang/fusion_1009/summary.md

Run names end in _<variant>[_<tag>]_s<seed>, variant = v<digit> plus letters / digits without an underscore:
v0p (V0': scene_fusion off on version-2 shards, MLP readout), v0ps (V0'-softmin), v5 (fusion on), v5w
(fusion_words off), v5d2 (V5 on both robots; "v5_d2" would read as v5 with a tag). The reference (--reference, a
run name or a variant / label such as V0'; default the first v0 run) holds the settings below fixed; --exempt
variant:field lets one variant differ in one of them (V5-d2's embodiments). Besides gamma, the actor head, TF32,
steps, tasks, embodiments and the scored set, every table holds equal the training data version (shard.json
lang_version of the shards the run trained on, via the checkpoint identity's data_roots) and the scored validation
episodes (a hash of the rows' (source, episode_id) set). scene_fusion, fusion_words and critic_readout differ by
design between E5 rows: they show in the switches column and are never refused on. Tables with V0' / V5 rows also
list the training episodes and samples per episode (batch_size x steps / training episodes, FUSION_PLAN C1) and end
in the E5 decision: V5 >= 0.85 adopt; V5 clearly above V0' (paired difference interval above 0) but below 0.85 and
V5-d2 > V5: more scenes; V5 at V0' level with E4 passed (--e4, fusion_probe.py's gate): the data lever.
"""

from __future__ import annotations

import argparse
import functools
import glob
import gzip
import hashlib
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
          "v4b": "V4b auxiliary attention loss",
          "v0p": "V0' fusion off (version-2 shards, MLP readout)", "v0ps": "V0'-softmin (V0' with the softmin readout)",
          "v5": "V5 fusion on", "v5w": "V5w fusion, cached words (fusion_words off)",
          "v5d2": "V5-d2 V5 on both robots"}
SHORT = {"v0": "V0", "v0p": "V0'", "v0ps": "V0'-softmin", "v5": "V5", "v5w": "V5w", "v5d2": "V5-d2"}
E2_VARIANTS = ("v1", "v2", "v3", "v4", "v4b")             # decision 5's runs
E5_VARIANTS = ("v0p", "v0ps", "v5", "v5w", "v5d2")         # FUSION_PLAN E5's runs
RUN_RE = re.compile(r"_(v\d[a-z0-9]*)(?:_[a-z0-9]+)?_s(\d+)$")
ADOPT = 0.85       # the grounding bar (FUSION_PLAN decision 4)
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
# FUSION_PLAN C7: also held equal in every table (they never differ between the E2 runs, so the E2 tables do not list
# them): the data version of the training shards and the scored validation episodes
FIXED_V2 = ("train_data_version",)
SCORED_V2 = ("val_episodes",)
CHECKED = FIXED + SCORED + FIXED_V2 + SCORED_V2
V2_KEYS = FIXED_V2 + SCORED_V2 + ("train_episodes", "train_episodes_per_task", "samples_per_episode")
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


@functools.lru_cache(maxsize=None)
def _episode_lines(root: str) -> tuple | None:
    """The recorded (not skipped) episode lines of a data root; None when the root has no lang_episodes.jsonl."""
    f = Path(root) / "lang_episodes.jsonl"
    if not f.exists():
        return None
    out = []
    for ln in f.read_text().splitlines():
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not d.get("skipped"):
            out.append(d)
    return tuple(out)


def _data_version(root: str) -> int:
    """A shard's data version: shard.json lang_version (version-2 collection writes 2; older shards have none: 1)."""
    f = Path(root) / "shard.json"
    return int(json.loads(f.read_text()).get("lang_version", 1)) if f.exists() else 1


@functools.lru_cache(maxsize=None)
def training_episodes(data_roots: tuple, tasks: tuple = (), embodiments: tuple = (), per_task: int = 0,
                      drop_stale_bead: bool = True) -> dict | None:
    """The episodes a run's training sampler drew from, from the episode lines of its data roots (the checkpoint
    identity's data_roots): the training tasks / embodiments (lang_config task_choices / embodiment_choices; empty =
    all), episodes with at least one non-terminal row (LangSampler never draws a terminal row), bead-task episodes of
    shards older than the bead-count h dropped as the sampler drops them; with per_task = N > 0
    (train_episodes_per_task) the sampler's subset: the first N of each (task, embodiment) in data.py's fixed order
    stable_hash(("e6", task, embodiment, episode_id)), ties by root name.

    -> {"episodes": sorted [(root, episode_id)], "per_group": {"task/embodiment": n}, "data_version": "1", "2" or
    "1+2" (over the roots that hold a training episode)}; None when no data root has episode lines."""
    from main.lang.data import BEAD_TASKS, stable_hash

    groups, versions, found = defaultdict(list), set(), False
    for j, root in enumerate(data_roots):
        lines = _episode_lines(str(root))
        if lines is None:
            continue
        found = True
        stale = drop_stale_bead and not (Path(root) / "item_buffer" / "n_violating.npy").exists()
        n_root = 0
        for d in lines:
            if (tasks and d.get("task") not in tasks) or (embodiments and d.get("embodiment") not in embodiments):
                continue
            r = d.get("rows")
            if (r is not None and int(r[1]) - int(r[0]) < 2) or (stale and d.get("task") in BEAD_TASKS):
                continue
            groups[(d.get("task"), d.get("embodiment"))].append((int(d["episode_id"]), Path(root).name, j))
            n_root += 1
        if n_root:
            versions.add(_data_version(str(root)))
    if not found:
        return None
    keep, per_group = [], {}
    for (task, emb), eps in sorted(groups.items(), key=lambda kv: str(kv[0])):
        eps.sort(key=lambda x: (stable_hash(("e6", task, emb, x[0])), x[1], x[2]))
        sel = eps[:per_task] if per_task > 0 else eps
        keep += [(str(data_roots[j]), e) for e, _, j in sel]
        per_group[f"{task}/{emb}"] = len(sel)
    return {"episodes": sorted(keep), "per_group": per_group,
            "data_version": "+".join(str(v) for v in sorted(versions)) or None}


def val_episodes(rows: list[dict]) -> str:
    """The scored validation episodes of a rows file: a hash of its (source, episode_id) set, and their count."""
    eps = sorted({(int(r.get("src", 0)), int(r["episode_id"])) for r in rows if r.get("episode_id") is not None})
    return f"{hashlib.md5(json.dumps(eps).encode()).hexdigest()[:12]} ({len(eps)} episodes)"


def run_settings(run: Path, out_name: str) -> dict:
    """What a comparison holds fixed and what each variant changes (round 4, item 2), from the run's
    lang_config.json, its scored metrics.json and (for the commit, the parts and the training data) the scored
    checkpoint. The scored validation episodes (val_episodes) come from the rows: main fills them in."""
    cfg = json.loads((run / "lang_config.json").read_text())
    mj = run / "eval_offline" / out_name / "metrics.json"
    md = json.loads(mj.read_text()) if mj.exists() else {}
    data = md.get("data") or {}
    keys, ident = _ckpt_keys(run / (md.get("ckpt") or "checkpoint_final.pt"))
    if data.get("ckpt_parts") is not None:   # recorded at scoring time: the parts the scored weights had
        keys = {"blocks" if k == "trunk" else k for k in data["ckpt_parts"]} | {"identity"}
    switches = [k for k in ("text_out_norm", "mask_special_tokens", "oracle_ref_tag", "aux_target_loss",
                            "aux_attention_loss") if cfg.get(k)]
    # E5 / E6: what differs by design between their rows (absent from configs saved before the fields)
    if cfg.get("scene_fusion"):
        switches += ["scene_fusion", "fusion_words" if cfg.get("fusion_words", True) else "fusion_words off"]
    if "critic_readout" in cfg:
        switches.append(f"critic_readout {cfg['critic_readout']}")
    per_task = int(cfg.get("train_episodes_per_task") or 0)
    if per_task:
        switches.append(f"{per_task} episodes / task")
    out = {"gamma": cfg.get("gamma"), "actor_hidden": int(cfg.get("actor_hidden", 0) or 0),
           "tf32_train": bool(cfg.get("tf32_train", False)), "steps": md.get("step"),
           "tasks": tuple(cfg.get("task_choices") or ()), "embodiments": tuple(cfg.get("embodiment_choices") or ()),
           "recall_ticks": cfg.get("recall_horizon_ticks"),
           "val_roots": tuple(data.get("val_roots") or ()), "scored_tasks": tuple(data.get("tasks") or ()),
           "scored_embodiments": tuple(data.get("embodiments") or ()), "ckpt": md.get("ckpt"),
           "switches": switches,
           "aux": {k: cfg.get(k) for k in ("aux_loss_weight", "aux_unit", "aux_huber_delta", "aux_attention_weight")
                   if k in cfg}, "git_head": (ident.get("git_head") or "")[:10], "parts": sorted(keys),
           "train_pair_acc": train_pair_acc(run)}
    out["missing_parts"] = [part for sw, part in PARTS.items() if cfg.get(sw) and part not in keys]
    tr = training_episodes(tuple(str(x) for x in ident.get("data_roots") or ()), out["tasks"], out["embodiments"],
                           per_task, bool(cfg.get("drop_stale_bead_rows", True)))
    n_ep = len(tr["episodes"]) if tr else None
    out.update({"train_data_version": tr["data_version"] if tr else None, "train_episodes": n_ep,
                "train_episodes_per_task": per_task, "val_episodes": None,
                "samples_per_episode": (int(cfg.get("batch_size", 64)) * int(md["step"]) / n_ep
                                        if n_ep and md.get("step") is not None else None)})
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


def variant_code(s: str) -> str:
    """A variant code from a code or a label: "V0'" -> v0p, "V0'-softmin" -> v0ps, "V5-d2" -> v5d2, "v5w" -> v5w."""
    return s.strip().lower().replace("-softmin", "s").replace("'", "p").replace("-", "").replace("_", "")


def short(v: str) -> str:
    return SHORT.get(v, v.upper())


def exemptions(specs) -> dict:
    """--exempt variant:field (repeatable) -> {variant: {field, ...}}: fields that variant may differ in."""
    out = defaultdict(set)
    for s in specs or ():
        v, _, field = str(s).partition(":")
        if not v or field not in CHECKED:
            raise SystemExit(f"--exempt {s!r}: expected <variant>:<field>, field one of {', '.join(CHECKED)}")
        out[variant_code(v)].add(field)
    return out


def paired_primary(a: dict | None, b: dict | None) -> tuple:
    """Variant a minus variant b on the primary pairs both scored ({seed: pair outcomes} each): per pair, a's seed
    mean minus b's, averaged with an episode-bootstrap 95% interval -> (mean, lo, hi, pairs)."""
    nan = float("nan")
    if not a or not b:
        return nan, nan, nan, 0
    keys = set.intersection(*[set(p) for p in a.values()], *[set(p) for p in b.values()])
    first = a[next(iter(a))]
    sel = sorted(k for k in keys if first[k][3] == "referent" and first[k][2] == "already_harmed"
                 and first[k][1] in PRIMARY_TASKS)
    obs = [(first[k][0], float(np.mean([a[s][k][4] for s in a])) - float(np.mean([b[s][k][4] for s in b])))
           for k in sel]
    return (*boot(obs), len(sel))


def e5_decision(res: dict, by_var: dict, e4: dict | None, e4_path: str | None) -> dict:
    """The E5 decision rule (FUSION_PLAN section 4): V5 >= 0.85: adopt. V5 clearly above V0' (the paired difference's
    interval above 0) but below 0.85 and still rising with scenes (V5-d2 > V5): more scenes. V5 at the level of V0' with
    E4 passed: the model ignores a working grounding signal, the next lever is the data."""
    nan = lambda x: x[0] != x[0]  # noqa: E731

    def prim(v):
        return tuple(res.get(v, {}).get("referent/already_harmed/T1 + T7", {}).get("mean", (float("nan"),) * 3))
    v0p, v5, d2 = prim("v0p"), prim("v5"), prim("v5d2")
    diff = paired_primary(by_var.get("v5"), by_var.get("v0p"))
    d2diff = paired_primary(by_var.get("v5d2"), by_var.get("v5"))
    passed = ((e4 or {}).get("gate") or {}).get("passed")
    if nan(v5) or nan(v0p):
        status = "pending"
        missing = [n for n, x in (("V5", v5), ("V0'", v0p)) if nan(x)]
        outcome = f"{' and '.join(missing)} not scored yet"
    elif v5[0] >= ADOPT:
        status = "adopt"
        outcome = (f"V5 reaches {ADOPT}: adopt it (collect the full warmup in version 2, retrain the official model "
                   "at gamma 0.95)")
    elif nan(diff):
        status, outcome = "pending", "V5 and V0' share no scored primary pairs (scored on different sets?)"
    elif diff[1] > 0:
        if nan(d2):
            status = "pending"
            outcome = f"V5 is above V0' but below {ADOPT}; V5-d2 not scored yet (does V5 rise with scenes?)"
        elif d2[0] > v5[0]:
            status = "more_scenes"
            outcome = (f"V5 is above V0' but below {ADOPT} and rises with scenes (V5-d2 > V5): collect more T1 / T7 "
                       "scenes and re-test")
        else:
            status = "no_rule"
            outcome = (f"V5 is above V0' but below {ADOPT} and does not rise with scenes (V5-d2 <= V5): the decision "
                       "rule names no next step")
    elif diff[2] < 0:
        status = "below"
        outcome = "V5 is below V0' (paired interval below 0): the fusion lowers the score; the rule names no next step"
    elif passed is True:
        status = "data_lever"
        outcome = ("V5 is at the level of V0' although E4 passed: the model ignores a working grounding signal and "
                   "memorization dominates; the next lever is the data (scene count, scene diversity, fewer repeats "
                   "per scene), not the architecture")
    elif passed is False:
        status = "e4_failed"
        outcome = ("V5 is at the level of V0' and E4 did not pass: the frozen stage does not single out the named object on "
                   "our scenes; the fallback is the referent phrase alone as the fusion text, then the scene-count curve")
    else:
        status = "pending"
        outcome = "V5 is at the level of V0'; the rule needs E4's gate (--e4 fusion_probe.json)"
    paired = {v: paired_primary(by_var[v], by_var.get("v0p")) for v in ("v0",) + E5_VARIANTS
              if v in by_var and v != "v0p"}
    return {"primary": {v: prim(v) for v in ("v0",) + E5_VARIANTS if v in res}, "paired_vs_v0p": paired,
            "v5_minus_v0p": diff, "v5d2_minus_v5": d2diff, "e4_passed": passed, "e4": e4_path, "bar": ADOPT,
            "status": status, "outcome": outcome}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-name", default="grounding")
    ap.add_argument("--md", required=True)
    ap.add_argument("--reference", default=None,
                    help="the run the table is held against: a run name, or a variant / label (v0p, V0'); default "
                         "the first v0 run")
    ap.add_argument("--exempt", action="append", default=[], metavar="VARIANT:FIELD",
                    help=f"let a variant differ from the reference in one field (repeatable), one of {', '.join(CHECKED)}")
    ap.add_argument("--e4", default=None, help="fusion_probe.json (E4): its gate enters the E5 decision")
    args = ap.parse_args(argv)
    exempt = exemptions(args.exempt)
    runs = sorted({Path(p) for s in args.runs for p in glob.glob(s) if Path(p).is_dir()})
    by_var = defaultdict(dict)   # variant -> seed -> pair outcomes
    meta, settings, refused = {}, {}, {}
    extra = defaultdict(dict)    # run -> {"train": primary on training episodes, "tag_use": ..., "curve": {step: acc}}
    found, pending = [], []
    for r in runs:
        m = RUN_RE.search(r.name)
        if not m:
            continue
        f = r / "eval_offline" / args.out_name / "rows.json.gz"
        if f.exists():
            found.append((r, m.group(1), int(m.group(2)), f))
        else:
            pending.append(r.name)
    pend_var = [RUN_RE.search(n).group(1) for n in pending]
    e5 = any(v in E5_VARIANTS for v in [x[1] for x in found] + pend_var)   # an E5 table: the V5 columns and blocks
    ref_key = args.reference or "v0"
    ref = (next((x for x in found if x[0].name == ref_key), None)
           or next((x for x in found if x[1] == variant_code(ref_key)), None))
    if args.reference and ref is None:
        print(f"[grounding_summary] reference {args.reference}: no scored run, nothing is refused", flush=True)
    ref_name = short(ref[1] if ref else variant_code(ref_key))
    ref_of = "V0's" if ref_name == "V0" else f"the reference ({ref_name})'s"   # the E2 tables say "V0's"
    if e5 and ref is not None and ref[1] != "v0p":
        print(f"[grounding_summary] note: this E5 table is held against {ref[0].name}, not V0' (--reference v0p)",
              flush=True)
    outcomes = {}
    for r, v, seed, f in found:   # every run's settings first: the reference's include its scored episodes
        st = run_settings(r, args.out_name)
        rows = _rows(f)
        st["val_episodes"] = val_episodes(rows)
        outcomes[r.name] = pair_outcomes(rows)
        settings[r.name] = st
        del rows
    ref_set = settings[ref[0].name] if ref else None
    taken = {}
    for r, v, seed, f in found:
        st = settings[r.name]
        why = [f"{k} {st[k]} != {ref_of} {ref_set[k]}" for k in CHECKED
               if ref_set is not None and k not in exempt.get(v, ()) and st[k] != ref_set[k]]
        if st["missing_parts"]:   # e.g. a V3 checkpoint without its tag was scored with a zero one (round 4, item 1)
            why.append(f"checkpoint lacks {st['missing_parts']}")
        if (v, seed) in taken:    # "v5_d2_s0" parses as v5 with a tag: it would overwrite v5's seed
            why.append(f"same variant and seed as {taken[(v, seed)]} (write variants without '_', e.g. v5d2)")
        if why:
            refused[r.name] = why
            print(f"[grounding_summary] refused {r.name}: {'; '.join(why)}", flush=True)
            continue
        taken[(v, seed)] = r.name
        by_var[v][seed] = outcomes[r.name]
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
    del outcomes
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
        return [n for n in settings if n not in refused and RUN_RE.search(n).group(1) == v]

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
    has_v5 = any(v in by_var for v in ("v5", "v5w", "v5d2"))
    if has_v5:   # E5 (FUSION_PLAN section 4)
        e4 = json.loads(Path(args.e4).read_text()) if args.e4 and Path(args.e4).exists() else None
        if args.e4 and e4 is None:
            print(f"[grounding_summary] --e4 {args.e4}: no such file; the E5 decision goes without E4", flush=True)
        res["_e5"] = e5_decision(res, by_var, e4, args.e4 if e4 else None)
    hidden = () if e5 else V2_KEYS   # the E2 tables keep their layout (the queued E2 summary lines write them)
    res["_settings"] = {n: {k: (list(x) if isinstance(x, tuple) else x) for k, x in st.items()
                            if k != "parts" and k not in hidden} for n, st in settings.items()}
    res["_refused"] = refused
    res["_extra"] = {n: {k: ({str(s): a for s, a in x.items()} if k == "curve" else x) for k, x in e.items()}
                     for n, e in extra.items()}
    if e5:
        res["_reference"] = ref[0].name if ref else None
        res["_exempt"] = {v: sorted(x) for v, x in sorted(exempt.items())}
    f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if t[0] == t[0] else "–"  # noqa: E731
    title = "(FUSION_PLAN E5)" if e5 else "(review round 3, Part II E2 / E4)"
    L = [f"# Grounding runs: summary {title}", "",
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
    r0 = lambda x: "–" if x is None else f"{x:,.0f}"  # noqa: E731
    held = FIXED + SCORED + (FIXED_V2 + SCORED_V2 if e5 else ())
    L += ["## Run settings (round 4, item 2)", "",
          f"Held equal to {ref_of}: " + ", ".join(held)
          + ("".join(f"; {short(v)} may differ in {', '.join(sorted(x))}" for v, x in sorted(exempt.items())) if e5 else "")
          + (". scene_fusion, fusion_words and critic_readout differ by design (switches); samples / episode = "
             "batch_size x steps / training episodes (FUSION_PLAN C1)" if e5 else "")
          + ". aux ratio = lambda x L_aux / L_critic from "
          f"train_log.jsonl, the median over steps {RATIO_WINDOW[0]}-{RATIO_WINDOW[1]} (and the last one); below "
          f"{MIN_RATIO} the auxiliary loss does not count. recall = recall_horizon_ticks (diagnostics only).", "",
          "| run | commit | gamma | actor_hidden | tf32_train | steps | recall | tasks | embodiments | switches | "
          "aux settings | aux ratio (n) / last | att ratio (n) / last | "
          + ("data version | val episodes | training episodes | samples / episode | " if e5 else "") + "refused |",
          "|---|" + "---|" * (17 if e5 else 13)]
    for n, st in sorted(settings.items()):
        aux = ", ".join(f"{k.replace('aux_', '')} {x}" for k, x in st["aux"].items()) if any(
            sw.startswith("aux") for sw in st["switches"]) else "–"
        L.append(f"| {n} | {st['git_head'] or '?'} | {st['gamma']} | {st['actor_hidden']} | {st['tf32_train']} | "
                 f"{st['steps']} | {st['recall_ticks'] if st['recall_ticks'] is not None else 'derived'} | "
                 f"{', '.join(st['tasks'])} | {', '.join(st['embodiments'])} | {', '.join(st['switches']) or '–'} | {aux} | "
                 f"{r2(st.get('aux_ratio_500'))} ({st.get('aux_ratio_n', 0)}) / {r2(st.get('aux_ratio_last'))} | "
                 f"{r2(st.get('att_ratio_500'))} ({st.get('att_ratio_n', 0)}) / {r2(st.get('att_ratio_last'))} | "
                 + (f"{st['train_data_version'] or '?'} | {st['val_episodes']} | {r0(st['train_episodes'])} | "
                    f"{r0(st['samples_per_episode'])} | " if e5 else "")
                 + f"{'; '.join(refused.get(n, [])) or '–'} |")
    L.append("")
    L += ["## Training versus validation (audit 2026-10-08)", "",
          "Primary metric (referent, already harmed, T1 + T7) per run on the validation episodes, on training "
          f"episodes scored the same way (eval_offline/{args.out_name}_train, if scored), and the in-batch pair rank "
          "accuracy of the last 1000 training steps (lang/pair_rank_acc: the training batches' own pairs, mixed "
          "kinds and states). A large gap = the model fits its training scenes without reading the name in a way "
          "that carries to new scenes." + (" Samples / episode: how often the run saw each training episode." if e5 else ""),
          "",
          "| run | validation | training episodes (pairs) | in-batch training pairs |"
          + (" training episodes | samples / episode |" if e5 else ""), "|---|---|---|---|" + ("---|---|" if e5 else "")]
    for n in sorted(meta):
        v, s_ = RUN_RE.search(n).groups()
        val_acc = primary_of(by_var[v][int(s_)])[0]
        tr = extra[n].get("train")
        L.append(f"| {n} | {val_acc:.3f} | {f'{tr[0]:.3f} ({tr[1]})' if tr else '–'} | "
                 f"{r2(settings[n].get('train_pair_acc'))} |"
                 + (f" {r0(settings[n]['train_episodes'])} | {r0(settings[n]['samples_per_episode'])} |" if e5 else ""))
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
            v, s_ = RUN_RE.search(n).groups()
            L.append(f"| {n} | " + " | ".join(f"{c[k][0]:.3f}" if k in c else "–" for k in steps) +
                     f" | {primary_of(by_var[v][int(s_)])[0]:.3f} |")
        L.append("")
    if not e5 or any(v in E2_VARIANTS for v in list(by_var) + pend_var):
        L += ["## Decision (decision 5)", "", "Primary: referent, already harmed, T1 + T7 pooled: " +
              ", ".join(f"{k} {f(x)}" for k, x in res["_e4"]["primary"].items()) + f". Outcome ({status}): **{outcome}**."
              + (f" Not scored yet: {', '.join(pending)}." if pending else "")
              + (f" Left out: {', '.join(refused)}." if refused else ""), ""]
    if has_v5:
        L += e5_md(res, by_var, settings, refused, pending, ref, f)
    Path(args.md).write_text("\n".join(L) + "\n")
    Path(args.md).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))
    return 0


def e5_md(res: dict, by_var: dict, settings: dict, refused: dict, pending: list, ref, f) -> list[str]:
    """The E5 block: per variant the primary score, the paired difference to V0', the training scenes and how
    often each was seen (C1), the switches; then the decision rule and its outcome."""
    d = res["_e5"]
    r0 = lambda x: "–" if x is None else f"{x:,.0f}"  # noqa: E731
    pd = lambda t: "–" if t[0] != t[0] else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] ({t[3]})"  # noqa: E731
    L = ["## E5: the fusion runs (FUSION_PLAN section 4)", "",
         f"Primary: referent pairs, already harmed, T1 + T7 pooled; the seed mean [episode-bootstrap 95% interval]. "
         "Paired: the variant minus V0' on the primary pairs both scored (per pair, the difference of the seed means), "
         "with its interval and the pair count. Training episodes and samples / episode per run (seeds separated by "
         f"/). Reference of the table: {ref[0].name if ref else 'none'}.", "",
         "| variant | runs | primary | paired vs V0' | training episodes | samples / episode | switches |",
         "|---|---|---|---|---|---|---|"]
    for v in ("v0",) + E5_VARIANTS:
        if v not in by_var:
            continue
        names = [n for n in sorted(settings) if n not in refused and RUN_RE.search(n).group(1) == v]
        st = [settings[n] for n in names]
        diff = d["paired_vs_v0p"].get(v, (float("nan"),) * 3 + (0,))
        sw = sorted({s for x in st for s in x["switches"]})
        L.append(f"| {LABELS.get(v, v)} | {len(names)} | {f(d['primary'][v])} | {pd(diff)} | "
                 f"{' / '.join(r0(x['train_episodes']) for x in st)} | {' / '.join(r0(x['samples_per_episode']) for x in st)} | "
                 f"{', '.join(sw) or '–'} |")
    e4 = {True: "passed", False: "did not pass", None: "not given"}[d["e4_passed"]] + (f" ({d['e4']})" if d["e4"] else "")
    L += ["", f"Decision rule (FUSION_PLAN E5): V5 >= {d['bar']} on the primary score: adopt (collect the full warmup "
          "in version 2, retrain the official model at gamma 0.95). V5 clearly above V0' (paired interval above 0) "
          f"but below {d['bar']} and still rising with scenes (V5-d2 > V5): collect more T1 / T7 scenes and re-test. "
          "V5 at the level of V0' with E4 passed: the model ignores a working grounding signal, and the next lever is the "
          f"data, not the architecture. V5 - V0' {pd(d['v5_minus_v0p'])}; V5-d2 - V5 {pd(d['v5d2_minus_v5'])}; E4 "
          f"{e4}. Outcome ({d['status']}): **{d['outcome']}**."
          + (f" Not scored yet: {', '.join(pending)}." if pending else "")
          + (f" Left out: {', '.join(refused)}." if refused else ""), ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
