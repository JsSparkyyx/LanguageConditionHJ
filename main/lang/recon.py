"""Phase-0 checks (VANILLA_PLAN 3.4); results go to docs/lang_filter/recon.md.

Every subcommand writes one JSON file the summarizer reads:

    # check 2 + 3 + 7, on roots written by `python -m main.lang.collect recon`
    python -m main.lang.recon parity <root> [<root> ...] --out <json>
    # check 4: HoloBrainText vs the build_text_cache.py output
    python -m main.lang.recon text --cache <cache.pt> --out <json>
    # check 5: filtered base RolloutController (random-init CrossSafe) + wire_item_eval
    python -m main.lang.recon filtered --task <t> --embodiment <e> --out <json>
    # check 6: in-the-way roles vs nominal outcomes, render-none runner
    python -m main.lang.recon roles --task <t> --embodiment <e> --n-seeds 10 --out <jsonl>
    # everything -> one markdown table set
    python -m main.lang.recon summarize <dir> --out docs/lang_filter/recon.md
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import ensure_item_path

ensure_item_path()

from .config import CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS, LangConfig, TASK_SHORT  # noqa: E402

RECON_SEED_BASE = 950_000_000  # scratch stream, disjoint from every Section 4.2 stream


def _dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ---------------------------------------------------------------------------
# check 2 (+3, 7): lockstep parity on a collected root
# ---------------------------------------------------------------------------


def parity(root: Path) -> dict:
    """StepBuffer.h == h_rules[active] (or min over valid rules), lengths,
    t_in_ep, episode records, done rows, referent token masks."""
    from .scene_roles import ref_rows

    root = Path(root)
    bh = json.loads((root / "buffer/header.json").read_text())
    sh = json.loads((root / "item_buffer/header.json").read_text())
    n, n_sc = int(bh.get("n", 0)), int(sh.get("n", 0))
    out = {"root": str(root), "n_buffer": n, "n_sidecar": n_sc, "len_equal": n == n_sc}
    if n == 0:
        out["ok"] = False
        return out
    ld = lambda d, f: np.load(root / d / f"{f}.npy", mmap_mode="r")[:n]  # noqa: E731
    h = np.asarray(ld("buffer", "h"), dtype=np.float64)
    eid_b = np.asarray(ld("buffer", "episode_id"))
    done = np.asarray(ld("buffer", "done"))
    H = np.asarray(ld("item_buffer", "h_rules"), dtype=np.float64)
    valid = np.asarray(ld("item_buffer", "rule_valid"))
    act = np.asarray(ld("item_buffer", "active_rule"), dtype=np.int64)
    tie = np.asarray(ld("item_buffer", "t_in_ep"), dtype=np.int64)
    eid_s = np.asarray(ld("item_buffer", "episode_id"))
    fs = np.asarray(ld("item_buffer", "fails_soon"))
    masks = np.load(root / "item_buffer/token_masks.npy", mmap_mode="r")
    want = np.where(act >= 0, H[np.arange(n), np.clip(act, 0, None)],
                    np.where(valid, H, np.inf).min(1))
    err = np.abs(h - want)
    out["h_parity_max_abs"] = float(err.max())
    out["h_parity_ok"] = bool(err.max() < 1e-4)
    out["episode_id_equal"] = bool((eid_b == eid_s).all())
    # per-episode: contiguous rows, t_in_ep 0..L-1, last row done, record exists
    eps, bad_t, bad_done, missing = [], [], [], []
    starts = np.flatnonzero(np.r_[True, eid_b[1:] != eid_b[:-1]])
    ends = np.r_[starts[1:], n]
    lines = {}
    lj = root / "lang_episodes.jsonl"
    if lj.exists():
        for ln in lj.read_text().splitlines():
            d = json.loads(ln)
            if not d.get("skipped"):
                lines[int(d["episode_id"])] = d
    mask_hits = defaultdict(list)
    fs_check = []
    for a, b in zip(starts, ends):
        e = int(eid_b[a])
        L = b - a
        eps.append(e)
        if not np.array_equal(tie[a:b], np.arange(L)):
            bad_t.append(e)
        if not (done[b - 1] and not done[a:b - 1].any()):
            bad_done.append(e)
        rec_path = root / "item_buffer/episodes" / f"{e}.json"
        if not rec_path.exists():
            missing.append(e)
            continue
        rec = json.loads(rec_path.read_text())
        R = len(rec["rules"])
        # fails_soon recomputed from h_rules (25 ticks, within the episode)
        Hep = H[a:b, :R]
        want_fs = np.zeros_like(Hep, dtype=bool)
        for t in range(L):
            want_fs[t] = (Hep[t:t + 25] < 0).any(0)
        fs_check.append(bool((fs[a:b, :R] == want_fs).all()))
        # check 7: referent rows are non-empty at t = 0 for most rules
        region_rows = (lines.get(e) or {}).get("mask_region_rows") or {}
        for rule in rec["rules"]:
            rows = ref_rows(rec, rule, region_rows)
            if not rows:
                continue
            m0 = np.asarray(masks[a, rows]).astype(np.float64).sum()
            mall = np.asarray(masks[a:b:10, rows]).astype(np.float64).reshape(-1).max() if L else 0.0
            mask_hits[(rec["task"], rule["consequence"])].append((m0 > 0, mall > 0))
    out.update({
        "episodes": len(eps), "t_in_ep_ok": not bad_t, "done_ok": not bad_done,
        "records_ok": not missing, "fails_soon_ok": all(fs_check) if fs_check else None,
        "bad_t": bad_t[:10], "bad_done": bad_done[:10], "missing_records": missing[:10],
        "mask_visible_t0": {f"{k[0]}|{k[1]}": float(np.mean([x[0] for x in v])) for k, v in mask_hits.items()},
        "mask_visible_any": {f"{k[0]}|{k[1]}": float(np.mean([x[1] for x in v])) for k, v in mask_hits.items()},
        "lines": list(lines.values()),
    })
    out["ok"] = bool(out["len_equal"] and out["h_parity_ok"] and out["episode_id_equal"]
                     and out["t_in_ep_ok"] and out["done_ok"] and out["records_ok"])
    return out


def cmd_parity(args) -> int:
    res = [parity(Path(r)) for r in args.roots]
    _dump(res, args.out)
    for r in res:
        print({k: v for k, v in r.items() if k not in ("lines", "mask_visible_t0", "mask_visible_any")})
    return 0 if all(r.get("ok") for r in res) else 1


# ---------------------------------------------------------------------------
# check 4: HoloBrainText vs the cache
# ---------------------------------------------------------------------------


def cmd_text(args) -> int:
    import torch

    from main.item_envs.language.text_encoder import HoloBrainText

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    keys = sorted(cache["index"])
    rng = np.random.default_rng(0)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(5, len(keys)), replace=False)]
    pipe = HoloBrainText(device="cuda")
    t0 = time.time()
    with torch.no_grad():
        e, v = pipe.embed(pick)
    dt = time.time() - t0
    e, v = e.float().cpu(), v.cpu().bool()
    L = cache["emb"].shape[1]
    diffs, rels, mask_eq = [], [], []
    for i, s in enumerate(pick):
        j = cache["index"][s]
        ce, cm = cache["emb"][j].float(), cache["mask"][j].bool()
        n = min(L, e.shape[1])
        fresh = torch.zeros(L, e.shape[2])
        fresh[:n] = e[i, :n]
        fm = torch.zeros(L, dtype=torch.bool)
        fm[:n] = v[i, :n]
        mask_eq.append(bool((fm == cm).all()))
        diffs.append(float(((fresh - ce).abs() * cm[:, None]).max()))
        rels.append(float((((fresh - ce).abs() / fresh.abs().clamp_min(1.0)) * cm[:, None]).max()))
    # the cache is fp16 and |emb| reaches ~140: fp16 storage alone rounds by up to
    # 2^-11 relative (0.06 absolute at that scale), so the plan's absolute 1e-3 is
    # read as a relative 1e-3 (recon.md)
    out = {"sentences": pick, "max_abs_diff": max(diffs), "per_sentence": diffs, "max_rel_diff": max(rels),
           "emb_abs_max": float(cache["emb"].float().abs().max()),
           "emb_rms": float(cache["emb"].float()[cache["mask"].bool()].pow(2).mean().sqrt()),
           "mask_equal": mask_eq, "embed_s": dt, "cache_n": len(keys),
           "ok": bool(max(rels) < 1e-3 and all(mask_eq))}
    _dump(out, args.out)
    print(out)
    return 0 if out["ok"] else 1


# ---------------------------------------------------------------------------
# check 5: filtered base controller with a random-init CrossSafe model
# ---------------------------------------------------------------------------


def cmd_filtered(args) -> int:
    import torch

    from main.item_envs.controller import RiskTickController
    from main.network.encoder import HoloBrainEncoder, RobotInjection
    from main.network.heads import PolicyEncoder, TokenActor, TwinCritic
    from main.network.trunk import GeometricTrunk
    from main.train.collect import CKPT_DIR, make_kinematics

    import dataclasses

    from .rollout import LangRolloutController, wire_item_eval

    cfg = LangConfig()
    torch.manual_seed(0)
    T = float(cfg.softmin_T) * float(cfg.h_scale)
    pe = PolicyEncoder(RobotInjection().cuda().eval(), GeometricTrunk().cuda().eval()).cuda().eval()
    actor, critics = TokenActor().cuda().eval(), TwinCritic(temperature=T).cuda().eval()
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, args.embodiment)
    results = []
    tasks = CANON_TASKS if args.task == "all" else tuple(args.task.split(","))
    for i, task in enumerate(tasks):
        seed = RECON_SEED_BASE + 500_000 + CANON_EMBODIMENTS.index(args.embodiment) * 1000 + i
        host = CANON_T1_HOSTS[i % 5] if task == "selective_avoid" else None
        cfg_ep = dataclasses.replace(cfg, task=task, embodiment=args.embodiment, item_host=host,
                                     item_out_root=str(Path(args.out).resolve().parent / "_runner"))
        rec = {"task": task, "embodiment": args.embodiment, "seed": seed, "host": host}
        t0 = time.time()
        try:
            ro = LangRolloutController(cfg_ep, seed, mode="filtered", encoder=encoder, kin=kin,
                                       policy_enc=pe, actor=actor, critics=critics, filter_active=True,
                                       episode_id=seed, rng=np.random.RandomState(seed),
                                       max_steps=10 * int(cfg.max_steps_per_episode),
                                       controller_cls=RiskTickController)
            runner = ro.env._item_runner
            rid = runner._primary_rule_id(ro.env.task)
            wire_item_eval(ro, rid)
            n_rules = len(runner.rules)
            seen = []
            orig = ro.h_fn

            def spy(env, c, _orig=orig):
                hv, diag = _orig(env, c)
                seen.append(len(np.asarray(diag["margins"]).reshape(-1)))
                return hv, diag

            ro.h_fn = spy
            with torch.no_grad():
                trace = ro.run()
            rec.update({
                "rule": rid, "n_rules": n_rules, "n_h_calls": len(seen),
                "margins_len_ok": bool(seen) and all(s == n_rules for s in seen),
                "status": trace.get("status"), "exception": trace.get("exception"),
                "success": trace.get("success"), "n_ticks": trace.get("n_ticks"),
                "n_q": len(trace.get("V", [])), "intervention_rate": trace.get("intervention_rate"),
                "timing": trace.get("timing"),
            })
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-1500:]})
        rec["wall_s"] = time.time() - t0
        rec["ok"] = bool(rec.get("margins_len_ok") and rec.get("n_q", 0) > 0 and "error" not in rec)
        print(f"[recon filtered] {rec}", flush=True)
        results.append(rec)
    _dump(results, args.out)
    return 0 if all(r["ok"] for r in results) else 1


# ---------------------------------------------------------------------------
# check 6: in-the-way roles vs nominal outcomes (render none)
# ---------------------------------------------------------------------------


def cmd_roles(args) -> int:
    from main.item_envs import language
    from main.item_envs.runner import ItemEpisodeRunner

    from .scene_roles import all_designations, in_way

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume: skip finished seeds
        for ln in out.read_text().splitlines():
            try:
                done.add(int(json.loads(ln)["seed"]))
            except Exception:
                pass
    t_idx, e_idx = CANON_TASKS.index(args.task), CANON_EMBODIMENTS.index(args.embodiment)
    n = args.n_seeds if args.task != "selective_avoid" else args.n_seeds_t1
    for k in range(args.start, min(n, args.stop) if args.stop is not None else n):
        seed = RECON_SEED_BASE + t_idx * 10_000 + e_idx * 1000 + k
        if seed in done:
            continue
        host = CANON_T1_HOSTS[k % 5] if args.task == "selective_avoid" else None
        row = {"task": args.task, "embodiment": args.embodiment, "seed": seed, "host": host}
        t0 = time.time()
        runner = ItemEpisodeRunner(args.task, embodiment=args.embodiment, seed=seed, host=host,
                                   render="none", out_root=str(out.parent / "_roles_runs"),
                                   max_steps=10 * int(args.max_ticks))
        try:
            runner.setup()
            env = runner.env
            env._item_runner = runner
            record = language.episode_record(runner, seed)
            roles = in_way(env, record)
            desig = all_designations(record, roles)
            primary = runner._primary_rule_id(env.task)
            info = getattr(env.task, "episode_info", None) or {}
            layout = getattr(env.task, "layout_info", None) or {}
            row.update({
                "in_way": sorted(roles[0]), "unclear": sorted(roles[1]), "designations": desig,
                "primary": primary,
                "primary_sampleable": bool(next((r["sampleable"] for r in record["rules"] if r["rule_id"] == primary), False)),
                "referent_kind": info.get("referent_kind"), "scenario": layout.get("scenario"),
                "rules": [{"rule_id": r["rule_id"], "consequence": r["consequence"], "sampleable": r.get("sampleable"),
                           "set": r.get("set"), "ambiguous": r.get("ambiguous"), "split": r.get("split")}
                          for r in record["rules"]],
            })
        except Exception as exc:
            row["error"] = f"setup {type(exc).__name__}: {exc}"[:500]
            runner.close()
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            continue
        meta, error = runner.run()
        row.update({
            "error": error, "success": bool(meta.task_success),
            "violated": {k2: bool(v) for k2, v in (meta.violations or {}).items()},
            "first_violation_t": meta.first_violation_t, "wall_s": round(time.time() - t0, 1),
        })
        print(f"[recon roles] {args.task}/{args.embodiment} seed {seed} host {host}: in_way {row['in_way']} "
              f"unclear {row['unclear']} violated {[r for r, v in row['violated'].items() if v]} err {error}",
              flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    return 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def _rate(xs):
    xs = list(xs)
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def roles_tables(rows: list[dict]) -> tuple[str, dict]:
    """Per (task, embodiment, consequence): violation rate of harm / benign /
    unclear designated rules (+ per host for T1). Returns (markdown, flags)."""
    acc = defaultdict(lambda: defaultdict(list))
    host_acc = defaultdict(lambda: defaultdict(list))
    prim = defaultdict(list)
    errors = defaultdict(int)
    for r in rows:
        if r.get("error") and "violated" not in r:
            errors[(r["task"], r["embodiment"])] += 1
            continue
        if r.get("error"):
            errors[(r["task"], r["embodiment"])] += 1
        cons = {x["rule_id"]: x["consequence"] for x in r.get("rules", [])}
        for rid, d in (r.get("designations") or {}).items():
            if d is None:
                continue
            v = bool((r.get("violated") or {}).get(rid, False))
            acc[(r["task"], r["embodiment"], cons.get(rid, "?"))][d].append(v)
            if r["task"] == "selective_avoid":
                host_acc[(r["host"], r["embodiment"], cons.get(rid, "?"))][d].append(v)
        prim[(r["task"], r["embodiment"])].append(bool(r.get("primary_sampleable")))
    lines = ["| task | embodiment | consequence | harm violated | benign violated | unclear violated |",
             "|---|---|---|---|---|---|"]
    flags = {"benign_over_20": [], "harm_under_20": []}

    def fmt(v):
        p, k = _rate(v)
        return f"{p:.2f} ({k})" if k else "–"

    for key in sorted(acc, key=lambda k: (CANON_TASKS.index(k[0]), k[1], k[2])):
        d = acc[key]
        lines.append(f"| {TASK_SHORT[key[0]]} {key[0]} | {key[1]} | {key[2]} | {fmt(d['harm'])} | "
                     f"{fmt(d['benign'])} | {fmt(d['unclear'])} |")
        pb, kb = _rate(d["benign"])
        ph, kh = _rate(d["harm"])
        if kb and pb > 0.2:
            flags["benign_over_20"].append(f"{key[0]}/{key[1]}/{key[2]}: {pb:.2f} of {kb}")
        if kh and ph < 0.2:
            flags["harm_under_20"].append(f"{key[0]}/{key[1]}/{key[2]}: {ph:.2f} of {kh}")
    lines += ["", "T1 per host:", "",
              "| host | embodiment | consequence | harm violated | benign violated | unclear violated |",
              "|---|---|---|---|---|---|"]
    for key in sorted(host_acc):
        d = host_acc[key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(d['harm'])} | {fmt(d['benign'])} | {fmt(d['unclear'])} |")
    lines += ["", "| task | embodiment | episodes | primary sampleable | errors |", "|---|---|---|---|---|"]
    for key in sorted(prim, key=lambda k: (CANON_TASKS.index(k[0]), k[1])):
        p, k = _rate(prim[key])
        lines.append(f"| {key[0]} | {key[1]} | {k} | {p:.2f} | {errors.get(key, 0)} |")
    return "\n".join(lines), flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity")
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("text")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("filtered")
    p.add_argument("--task", default="all")
    p.add_argument("--embodiment", required=True, choices=CANON_EMBODIMENTS)
    p.add_argument("--out", required=True)
    p = sub.add_parser("roles")
    p.add_argument("--task", required=True, choices=CANON_TASKS)
    p.add_argument("--embodiment", required=True, choices=CANON_EMBODIMENTS)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--n-seeds-t1", type=int, default=25)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--max-ticks", type=int, default=750)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"parity": cmd_parity, "text": cmd_text, "filtered": cmd_filtered, "roles": cmd_roles}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
