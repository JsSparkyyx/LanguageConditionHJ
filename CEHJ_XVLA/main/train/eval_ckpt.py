"""Full cross-embodiment evaluation of ONE trained checkpoint (or nominal).

Runs the full 5 embodiments x 5 tasks x N episodes grid with the safety
filter (deterministic policy), all scenes ON-PATH, seeds fully derived from
(task, embodiment, episode) so every checkpoint is evaluated on IDENTICAL
scenes and re-running the same eval reproduces the same numbers. Every
combo logs EXACTLY N episodes: planner no-pose aborts (empty traces) and
spawn failures redraw a fresh deterministic scene (seed + attempt*10M)
until a real episode completes (10 attempts max).

Auto-detects the architecture from <run>/config.json:
  - encoder_variant "HoloBrain_v0.0_GD"  -> joint-space HJ-SAC (this repo)
  - encoder_variant "X-VLA-RoboTwin2"    -> EE6D X-VLA baseline
    (run the CEHJ_XVLA copy of this file — the `main` package name is
    shared, so both architectures cannot load in one process)
  - no config.json / --nominal           -> nominal controller only
    (FrozenConfig defaults; the seed schedule is the same, so scenes match
    the filtered evals exactly)

Outputs under <out>/ (default <run>/eval_full/<tag>/):
  t<BUDGET>/episodes[_shard].jsonl + summary[_shard].json — ONE simulation
  runs to the max of --cutoffs (default 18,30 s); stats are re-scored per
  budget by slicing the trace (success counts only within the budget).
  videos/   shared across budgets; the filename tag follows the longest
            budget's outcome
  eval_meta.json   args + resolved config + ckpt path for reproducibility

Episode keys: task, embodiment, episode, seed, trained (embodiment seen in
training), success, collision booleans per criterion (collision_h:
any h<0 tick; collision_contact: any PhysX touch; collision_force30:
any >= 30 N tick) + aggregate collision (contact OR force, drives the
video tag) + collision_frac (contact-wise tick fraction), duration_s
(wall of the last control tick, sim time), intervention_rate, n_interventions,
mode_switches, viol_h_frac (h<0), viol_contact_frac (PhysX touch),
viol_force30_frac (force >= 30 N), viol_any_frac, force_max,
force_mean_nz, force_std_nz, force_median_nz (nonzero ticks only),
min_h_cm, mean_h_cm, hq_gap_mean_cm, hq_gap_std_cm, hq_corr, hq_corr_p
(Pearson r / p-value between h(t) and Q(t)), video.

Usage:
    python main/train/eval_ckpt.py --run <run_dir> [--ckpt checkpoint.pt]
        [--episodes 50] [--filter-release-margin 0.005]
        [--embodiments piper,ur5-wsg] [--tasks stack_blocks_two]
    python main/train/eval_ckpt.py --run <run_dir> --nominal
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "main.train"
ROOT = Path(__file__).resolve().parents[2]

ALL_EMBODIMENTS = ["piper", "franka-panda", "ARX-X5", "ur5-wsg", "aloha-agilex"]
ALL_TASKS = [
    "place_container_plate", "place_burger_fries", "stack_blocks_two",
    "stack_bowls_two", "place_bread_basket",
]
FORCE_VIOL_N = 30.0          # collision stat threshold requested for eval
SUMMARY_METRICS = [
    "success", "collision", "collision_h", "collision_contact",
    "collision_force30", "collision_frac", "viol_h_frac",
    "viol_contact_frac", "viol_force30_frac", "viol_any_frac",
    "intervention_rate", "force_max", "min_h_cm",
    "hq_gap_mean_cm", "hq_corr", "hq_corr_p",
]


def episode_seed(seed_base: int, task_idx: int, emb_idx: int, ep: int) -> int:
    """Combo-unique, ckpt-independent: identical scenes for every eval."""
    return seed_base + task_idx * 100_000 + emb_idx * 10_000 + ep


def set_global_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_xvla_config(cfg_d: dict) -> bool:
    return "X-VLA" in str(cfg_d.get("encoder_variant", ""))


def build_models(cfg, ckpt_path: Path | None):
    """-> (kind, encoder, kin_fn, rollout_kwargs). kind in {hb, xvla, nominal}."""
    import torch

    from main.train.collect import make_kinematics
    from main.train.rollout import RolloutController  # noqa: F401

    xvla = hasattr(cfg, "xvla_ckpt")
    if xvla:
        from main.network.encoder import XVLAEncoder
        from main.network.heads import EE6DActor, EE6DTwinCritic
        from main.train.collect import KIN_URDF_DIR

        encoder = XVLAEncoder(str(cfg.xvla_ckpt), device="cuda")
        kin_fn = lambda emb: make_kinematics(KIN_URDF_DIR, emb)  # noqa: E731
        if ckpt_path is None:
            return "nominal", encoder, kin_fn, {}
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
        encoder.adapter.load_state_dict(ckpt["adapter"])
        encoder.proprio_enc.load_state_dict(ckpt["proprio_enc"])
        actor = EE6DActor().cuda()
        critics = EE6DTwinCritic(
            temperature=cfg.softmin_T * float(cfg.h_scale)).cuda()
        actor.load_state_dict(ckpt["actor"])
        critics.load_state_dict(ckpt["critics"])
        actor.eval().requires_grad_(False)
        critics.eval().requires_grad_(False)
        print(f"[eval] loaded X-VLA ckpt {ckpt_path}")
        return "xvla", encoder, kin_fn, {"actor": actor, "critics": critics}

    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR

    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin_fn = lambda emb: make_kinematics(CKPT_DIR, emb)  # noqa: E731
    if ckpt_path is None:
        return "nominal", encoder, kin_fn, {}
    from main.network.encoder import RobotInjection
    from main.network.heads import PolicyEncoder, TokenActor, TwinCritic
    from main.network.trunk import GeometricTrunk

    geo = not cfg.ablate_geometry
    injection = RobotInjection(use_body_features=not cfg.ablate_injection)
    trunk = GeometricTrunk(geo=geo)
    policy_enc = PolicyEncoder(injection, trunk).cuda()
    actor = TokenActor().cuda()
    critics = TwinCritic(
        temperature=cfg.softmin_T * float(cfg.h_scale), geo=geo).cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    injection.load_state_dict(ckpt["injection"])
    trunk.load_state_dict(ckpt["trunk"])
    actor.load_state_dict(ckpt["actor"])
    critics.load_state_dict(ckpt["critics"])
    for m in (injection, trunk, actor, critics):
        m.eval().requires_grad_(False)
    print(f"[eval] loaded HoloBrain ckpt {ckpt_path}")
    return "hb", encoder, kin_fn, {
        "policy_enc": policy_enc, "actor": actor, "critics": critics,
    }


def slice_trace(trace: dict, cutoff_s: float) -> dict:
    """Re-score one recorded trace under a shorter time budget.

    Per-tick arrays are sliced at t <= cutoff_s; success becomes
    success AND finished within the budget (an episode still running at
    the cutoff is a timeout = failure, the real semantics of a time cap).
    Episodes ending before the cutoff slice to themselves — identical
    numbers under both budgets."""
    t = np.asarray(trace["t"], dtype=np.float64)
    out = dict(trace)
    m = t <= cutoff_s + 1e-9
    for k in ("t", "h", "contact_force", "contact_touch"):
        arr = np.asarray(trace.get(k, []))
        if len(arr) == len(t):
            out[k] = arr[m].tolist() if k == "t" else (
                np.asarray(arr)[m])
    Vt = np.asarray(trace.get("V_t", []), dtype=np.float64)
    mv = Vt <= cutoff_s + 1e-9
    for k in ("V", "V_t", "intervened"):
        arr = trace.get(k, [])
        if len(arr) == len(Vt):
            out[k] = np.asarray(arr)[mv]
    finished_at = float(t[-1]) if len(t) else 0.0
    out["success"] = bool(trace["success"]) and finished_at <= cutoff_s + 1e-9
    return out


def episode_stats(trace: dict, h_scale: float, meta: dict) -> dict:
    from scipy.stats import pearsonr

    cm = 100.0 / float(h_scale)
    h = np.asarray(trace["h"], dtype=np.float64)
    force = np.asarray(trace.get("contact_force", []), dtype=np.float64)
    touch = np.asarray(trace.get("contact_touch", []), dtype=bool)
    n = len(h)
    if len(force) != n:
        force = np.zeros(n)
    if len(touch) != n:
        touch = np.zeros(n, dtype=bool)

    nz = force[force > 0]
    v_h = h < 0
    v_f = force >= FORCE_VIOL_N
    # intervention stats from the (possibly time-sliced) per-tick flags so
    # truncated budgets report their own rates, not the full-run totals
    interv = np.asarray(trace.get("intervened", []), dtype=bool)
    if len(interv):
        int_rate = float(interv.mean())
        n_int = int(interv.sum())
        n_sw = int((interv[1:] != interv[:-1]).sum())
    else:
        int_rate = float(trace.get("intervention_rate", 0.0))
        n_int = int(trace.get("interventions", 0))
        n_sw = int(trace.get("mode_switches", 0))
    row = {
        **meta,
        "success": bool(trace["success"]),
        "duration_s": float(trace["t"][-1]) if n else 0.0,
        "intervention_rate": int_rate,
        "n_interventions": n_int,
        "mode_switches": n_sw,
        "viol_h_frac": float(v_h.mean()) if n else float("nan"),
        "viol_contact_frac": float(touch.mean()) if n else float("nan"),
        "viol_force30_frac": float(v_f.mean()) if n else float("nan"),
        "viol_any_frac": float((v_h | v_f | touch).mean()) if n else float("nan"),
        "force_max": float(force.max()) if n else 0.0,
        "force_mean_nz": float(nz.mean()) if len(nz) else 0.0,
        "force_std_nz": float(nz.std()) if len(nz) else 0.0,
        "force_median_nz": float(np.median(nz)) if len(nz) else 0.0,
        "min_h_cm": float(h.min()) * cm if n else float("nan"),
        "mean_h_cm": float(h.mean()) * cm if n else float("nan"),
        # episode-level booleans + contact-wise tick fraction
        "collision_h": bool(v_h.any()) if n else False,
        "collision_contact": bool(touch.any()) if n else False,
        "collision_force30": bool(v_f.any()) if n else False,
        # aggregate (contact-wise): touch or >= 30 N — drives the video tag
        "collision": bool(touch.any() or v_f.any()) if n else False,
        "collision_frac": float((touch | v_f).mean()) if n else float("nan"),
    }

    # h vs Q: Q is logged at tick-hook times V_t; interpolate onto the h
    # tick grid, then Pearson r / p over ticks with a finite pair
    V = np.asarray(trace.get("V", []), dtype=np.float64)
    Vt = np.asarray(trace.get("V_t", []), dtype=np.float64)
    t = np.asarray(trace["t"], dtype=np.float64)
    if n >= 3 and len(V) >= 3:
        q_on_t = np.interp(t, Vt, V)
        gap = (h - q_on_t) * cm
        row["hq_gap_mean_cm"] = float(gap.mean())
        row["hq_gap_std_cm"] = float(gap.std())
        if h.std() > 1e-12 and q_on_t.std() > 1e-12:
            r, p = pearsonr(h, q_on_t)
            row["hq_corr"], row["hq_corr_p"] = float(r), float(p)
        else:
            row["hq_corr"] = row["hq_corr_p"] = float("nan")
    else:
        row["hq_gap_mean_cm"] = row["hq_gap_std_cm"] = float("nan")
        row["hq_corr"] = row["hq_corr_p"] = float("nan")
    return row


def summarize(rows: list[dict]) -> dict:
    def agg(sub: list[dict]) -> dict:
        out = {}
        for m in SUMMARY_METRICS:
            vals = np.array([float(r[m]) for r in sub], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                out[m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            else:
                out[m] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n": int(len(vals)),
                }
        out["n_episodes"] = len(sub)
        return out

    trained = [r for r in rows if r["trained"]]
    per_task, per_task_tr = {}, {}
    per_emb = {}
    for r in rows:
        per_task.setdefault(r["task"], []).append(r)
        if r["trained"]:
            per_task_tr.setdefault(r["task"], []).append(r)
    for e in ALL_EMBODIMENTS:
        sub = [r for r in rows if r["embodiment"] == e]
        if sub:
            per_emb[e] = {"trained": bool(sub[0]["trained"]), **agg(sub)}
    return {
        "overall": agg(rows),
        "overall_trained_only": agg(trained),
        "per_task": {t: agg(s) for t, s in per_task.items()},
        "per_task_trained_only": {t: agg(s) for t, s in per_task_tr.items()},
        "per_embodiment": per_emb,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help="run dir holding config.json + checkpoints")
    p.add_argument("--ckpt", type=str, default="checkpoint.pt",
                   help="checkpoint filename inside <run> or an absolute path")
    p.add_argument("--nominal", action="store_true",
                   help="nominal controller only (no filter, no ckpt)")
    p.add_argument("--episodes", type=int, default=50,
                   help="episodes per (task, embodiment)")
    p.add_argument("--embodiments", type=str, default=None,
                   help="comma list (default: all 5) — for sharding")
    p.add_argument("--tasks", type=str, default=None,
                   help="comma list (default: all 5) — for sharding")
    p.add_argument("--seed-base", type=int, default=70_000,
                   help="scene seeds are derived from this; keep fixed across "
                        "all ckpt evals or the scenes will not match")
    p.add_argument("--filter-release-margin", type=float, default=None,
                   help="metres; override the config's release margin")
    p.add_argument("--filter-margin", type=float, default=None,
                   help="metres; override the config's engage margin")
    p.add_argument("--out", type=Path, default=None,
                   help="default <run>/eval_full/<tag>")
    p.add_argument("--cutoffs", type=str, default="18,30",
                   help="comma list of time budgets in seconds; ONE "
                        "simulation runs to the max, stats are re-scored "
                        "per budget by slicing the trace (success within "
                        "budget only). Output: <out>/t<BUDGET>/ per budget")
    args = p.parse_args()

    import torch  # noqa: F401  (CUDA sanity before any sim work)

    run_dir = Path(args.run).resolve()
    cfg_path = run_dir / "config.json"
    from main.train.config import FrozenConfig

    if cfg_path.exists():
        cfg_d = json.loads(cfg_path.read_text())
        if is_xvla_config(cfg_d) != (ROOT.name == "CEHJ_XVLA"):
            which = "CEHJ_XVLA" if is_xvla_config(cfg_d) else "CEHJ_safe"
            sys.exit(
                f"[eval] {run_dir} is a {cfg_d.get('encoder_variant')} run — "
                f"use the {which}/main/train/eval_ckpt.py copy"
            )
        cfg = FrozenConfig.load(cfg_path)
    else:
        if not args.nominal:
            sys.exit(f"[eval] no config.json in {run_dir} — pass --nominal "
                     f"for a nominal-only baseline")
        cfg = FrozenConfig()
        cfg_d = cfg.to_dict()

    trained_pool = set(cfg.embodiment_choices)
    ckpt_path = None
    if not args.nominal:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = run_dir / ckpt_path
        if not ckpt_path.exists():
            sys.exit(f"[eval] checkpoint not found: {ckpt_path}")

    # eval overrides: all scenes ON-PATH; margins from CLI if given
    cfg.obstacle_mode = "on_path"
    if args.filter_release_margin is not None:
        cfg.filter_release_margin = args.filter_release_margin
    if args.filter_margin is not None:
        cfg.filter_margin = args.filter_margin

    kind, encoder, kin_fn, model_kw = build_models(
        cfg, None if args.nominal else ckpt_path)

    tag = "nominal" if args.nominal else Path(args.ckpt).stem
    if args.filter_release_margin is not None:
        tag += f"_rm{args.filter_release_margin:g}"
    out_dir = (Path(args.out).resolve() if args.out
               else run_dir / "eval_full" / tag)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run": str(run_dir), "ckpt": str(ckpt_path), "kind": kind,
        "vars": vars(args), "trained_pool": sorted(trained_pool),
        "config": cfg_d,
    }
    if ckpt_path is not None:
        # checkpoint.pt is republished during training — stamp its identity
        # so eval_full/checkpoint*/ results are attributable to exact weights
        import hashlib

        st = ckpt_path.stat()
        meta["ckpt_mtime"] = st.st_mtime
        meta["ckpt_size"] = st.st_size
        meta["ckpt_md5"] = hashlib.md5(ckpt_path.read_bytes()).hexdigest()
    (out_dir / "eval_meta.json").write_text(json.dumps(
        meta, indent=2, default=str))

    embs = args.embodiments.split(",") if args.embodiments else ALL_EMBODIMENTS
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS
    from main.train.eval_utils import PanelVideoWriter
    from main.train.rollout import RolloutController

    # sharded jobs (--embodiments/--tasks subsets, e.g. from eval_all.py)
    # share out_dir: per-shard jsonl/summary, videos/ shared (names are
    # per-episode unique). Merge with scripts/eval_all.py --merge <run>.
    suffix = ""
    if args.embodiments or args.tasks:
        parts = []
        if args.embodiments:
            parts.append("emb-" + "-".join(embs))
        if args.tasks:
            parts.append("tsk-" + "-".join(tasks))
        suffix = "_" + "_".join(parts)
    jsonl_paths, summary_paths, rows = {}, {}, {}
    cutoffs = sorted(float(c) for c in args.cutoffs.split(","))
    for cut in cutoffs:
        d = out_dir / f"t{cut:g}"
        d.mkdir(parents=True, exist_ok=True)
        jp = d / f"episodes{suffix}.jsonl"
        if jp.exists():
            sys.exit(f"[eval] {jp} already exists — a rerun would mix "
                     f"partial results. Delete it first.")
        jsonl_paths[cut] = jp
        summary_paths[cut] = d / f"summary{suffix}.json"
        rows[cut] = []
    max_steps = int(max(cutoffs) * 250.0)     # physics steps, 250 Hz
    h_scale = float(getattr(cfg, "h_scale", 1.0))
    jfs = {cut: open(jp, "a") for cut, jp in jsonl_paths.items()}
    try:
        for emb in embs:
            kin = kin_fn(emb)
            for task in tasks:
                t_idx, e_idx = ALL_TASKS.index(task), ALL_EMBODIMENTS.index(emb)
                for ep in range(args.episodes):
                    seed = episode_seed(args.seed_base, t_idx, e_idx, ep)
                    cfg_ep = dataclasses.replace(cfg, task=task, embodiment=emb)
                    meta = {
                        "task": task, "embodiment": emb, "episode": ep,
                        "seed": seed, "trained": emb in trained_pool,
                    }
                    tmp_vid = vid_dir / f".tmp_{emb}_{task}_ep{ep:03d}.mp4"
                    ro, trace, used_seed = None, None, seed
                    try:
                        # attempt loop: a no-tick abort (grasp planner found
                        # no pose) or a 5x spawn failure redraws a FRESH
                        # scene (deterministic offset, far outside the
                        # seed grid) until a real episode is logged — the
                        # combo always ends with exactly --episodes rows
                        for attempt in range(10):
                            s0 = seed + attempt * 10_000_000
                            video = PanelVideoWriter(tmp_vid, fps=25.0,
                                                     h_scale=h_scale)
                            for retry in range(5):
                                set_global_seeds(s0 + retry * 1000)
                                cand = RolloutController(
                                    cfg_ep, s0 + retry * 1000,
                                    mode="filtered",
                                    encoder=encoder, kin=kin, video=video,
                                    max_steps=max_steps,
                                    **model_kw,
                                )
                                if cand.env.obstacle is not None:
                                    ro, used_seed = cand, s0 + retry * 1000
                                    break
                                print(f"[eval] spawn failed "
                                      f"(seed {s0 + retry * 1000}); re-drawing")
                                cand.env.close()
                            if ro is None:
                                video.close()
                                tmp_vid.unlink(missing_ok=True)
                                continue
                            trace = ro.run()
                            video.close()
                            if trace["h"]:
                                break
                            print(f"[eval] {emb}/{task}/ep{ep:03d} produced "
                                  f"no ticks (task aborted); redrawing "
                                  f"scene (attempt {attempt + 1})")
                            tmp_vid.unlink(missing_ok=True)
                            ro = None
                        if ro is None or trace is None or not trace["h"]:
                            raise RuntimeError(
                                "no valid episode after 10 attempts")
                    except Exception as exc:
                        import traceback

                        traceback.print_exc()
                        try:
                            video.close()
                        except Exception:
                            pass
                        tmp_vid.unlink(missing_ok=True)
                        for cut in cutoffs:
                            jfs[cut].write(json.dumps(
                                {**meta, "success": False, "collision": False,
                                 "error": f"{type(exc).__name__}: {exc}"}
                            ) + "\n")
                            jfs[cut].flush()
                            rows[cut].append(
                                {**meta, "success": 0.0, "collision": False,
                                 **{m: float("nan") for m in SUMMARY_METRICS},
                                 "trained": meta["trained"]})
                        continue
                    meta["seed"] = used_seed
                    # one simulation, one video; stats re-scored per budget.
                    # The video tag follows the LONGEST budget's outcome.
                    per_cut = {cut: episode_stats(slice_trace(trace, cut),
                                                  h_scale, dict(meta))
                               for cut in cutoffs}
                    main_row = per_cut[max(cutoffs)]
                    vid_name = (f"{emb}_{task}_ep{ep:03d}_"
                                f"{'success' if main_row['success'] else 'fail'}_"
                                f"{'collision' if main_row['collision'] else 'clean'}.mp4")
                    if tmp_vid.exists():
                        os_replace(tmp_vid, vid_dir / vid_name)
                        vid_ref = f"../videos/{vid_name}"
                    else:
                        vid_ref = None
                    for cut in cutoffs:
                        row = per_cut[cut]
                        row["video"] = vid_ref
                        jfs[cut].write(json.dumps(row) + "\n")
                        jfs[cut].flush()
                        rows[cut].append(row)
                    print(f"[eval] {emb}/{task}/ep{ep:03d} "
                          f"success={main_row['success']} "
                          f"min_h={main_row['min_h_cm']:.1f}cm "
                          f"viol_any={main_row['viol_any_frac']:.3f} "
                          f"int={main_row['intervention_rate']:.3f}")
    finally:
        for jf in jfs.values():
            jf.close()

    for cut in cutoffs:
        summary = summarize(rows[cut])
        summary_paths[cut].write_text(json.dumps(summary, indent=2))
        print(f"[eval] t{cut:g}s: wrote {jsonl_paths[cut]} "
              f"({len(rows[cut])} episodes) + {summary_paths[cut]}")
    ov = summarize(rows[max(cutoffs)])["overall"]
    print("[eval] overall: "
          + " ".join(f"{m}={ov[m]['mean']:.3f}" for m in
                     ("success", "viol_any_frac", "intervention_rate")))
    print("EVAL DONE")


def os_replace(src: Path, dst: Path) -> None:
    import os

    os.replace(src, dst)


if __name__ == "__main__":
    main()
