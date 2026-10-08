"""R2 check (Phase-0 style): do T1 objects the nominal plans around stay safe?

    # 5 nominal T1 episodes of one (embodiment, host), in the container on a GPU node:
    python scripts/lang/t1_avoid_check.py run --embodiment ur5-wsg --host place_container_plate --n 5 \\
        --out runs/lang_verify/t1_avoid
    # after all 10 (embodiment, host) runs: rates per group and host, the STOP verdict
    python scripts/lang/t1_avoid_check.py report runs/lang_verify/t1_avoid

Seeds: a scratch stream 920M + 1M x embodiment + 10k x host (disjoint from training, validation and
recon). Groups per candidate object: avoided (planner_avoids: in the nominal's cuRobo world), on/near-path
not avoided, off-path. A rule counts as violated when its h < 0 at any tick (the sidecar labels).
STOP (exit 1) when avoided objects' rules are violated in more than 20% of cases: then their benign
designation would not be true under the nominal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STOP_RATE = 0.2


def cmd_run(args) -> int:
    import main.lang  # noqa: F401  (main.item_envs path)
    from main.lang.collect import make_pair, run_episodes
    from main.lang.config import CANON_EMBODIMENTS, CANON_T1_HOSTS, LangConfig

    cfg = LangConfig()
    out = Path(args.out).resolve() / f"{args.embodiment}_{args.host}"
    if (out / "lang_episodes.jsonl").exists():
        print(f"[t1_avoid_check] {out} exists: delete it to rerun")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    slots = [("selective_avoid", args.embodiment, args.host)] * int(args.n)
    buf, side = make_pair(out, cfg, int(len(slots) * cfg.max_steps_per_episode * 1.1) + 16)
    offset = 920_000_000 + CANON_EMBODIMENTS.index(args.embodiment) * 1_000_000 + CANON_T1_HOSTS.index(args.host) * 10_000
    summary = run_episodes(cfg, out, slots, offset, buf, side, video_every=args.video_every, with_masks=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[t1_avoid_check] done: {summary}")
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - hw), min(1.0, c + hw)


def cmd_report(args) -> int:
    root = Path(args.root)
    # (group, consequence) and (host, group, consequence) -> [violated?]
    by = defaultdict(list)
    by_host = defaultdict(list)
    eps = n_avoid_eps = 0
    status = defaultdict(int)
    # episode level, by whether the scene has an avoided object: RoboTwin picks grasps by test-planning with
    # cuRobo, so a box in its world can also make grasps fail (status exception at tick 1)
    ep_by = defaultdict(list)
    for f in sorted(root.glob("*/lang_episodes.jsonl")):
        for ln in f.read_text().splitlines():
            d = json.loads(ln)
            if d.get("skipped"):
                continue
            eps += 1
            status[d.get("status")] += 1
            pl = (d.get("scene_meta") or {}).get("placements") or []
            viol = d.get("violated") or {}
            has_avoid = any(p.get("planner_avoids") for p in pl)
            n_avoid_eps += has_avoid
            kind = "with an avoided object" if has_avoid else "without"
            for key in ((kind,), (d.get("host"), kind)):
                ep_by[key].append((d.get("status") == "ok", bool(d.get("success"))))
            if d.get("status") != "ok":   # an episode that stopped early (failed grasp) says nothing about
                continue                   # its objects' safety: it must not pull the avoided rate to 0
            for p in pl:
                grp = ("avoided" if p.get("planner_avoids") else
                       "on/near, not avoided" if p.get("cls") in ("on_path", "near_path") else "off_path")
                vs = {}
                for cons in ("avoid", "stay"):
                    rid = f"{cons}|{p['item_id']}|-"
                    if rid in viol:
                        vs[cons] = bool(viol[rid])
                if vs:
                    vs["either"] = any(vs.values())
                for cons, v in vs.items():
                    by[(grp, cons)].append(v)
                    by_host[(d.get("host"), grp, cons)].append(v)
    L = [f"# R2 check: {eps} T1 nominal episodes ({n_avoid_eps} with an avoided object), status {dict(status)}", "",
         "Rule violated (h < 0 at any tick) per candidate object, episodes that ran (status ok) only; "
         "[95% Wilson interval] (objects).", "",
         "| group | avoid | stay | either |", "|---|---|---|---|"]
    fmt = lambda xs: (lambda p, lo, hi: f"{p:.2f} [{lo:.2f}, {hi:.2f}] ({len(xs)})")(*wilson(sum(xs), len(xs))) if xs else "–"  # noqa: E731
    for grp in ("avoided", "on/near, not avoided", "off_path"):
        L.append(f"| {grp} | {fmt(by[(grp, 'avoid')])} | {fmt(by[(grp, 'stay')])} | {fmt(by[(grp, 'either')])} |")
    L += ["", "Per host (either rule):", "", "| host | avoided | on/near, not avoided | off_path |", "|---|---|---|---|"]
    for h in sorted({k[0] for k in by_host}):
        L.append(f"| {h} | " + " | ".join(fmt(by_host[(h, g, 'either')]) for g in
                                         ("avoided", "on/near, not avoided", "off_path")) + " |")
    L += ["", "Episodes by whether the scene has an avoided object (ran = status ok, i.e. no exception such as a "
          "failed grasp; success = task done):", "", "| scenes | ran | success |", "|---|---|---|"]
    for kind in ("with an avoided object", "without"):
        e = ep_by[(kind,)]
        L.append(f"| {kind} | {fmt([a for a, _ in e])} | {fmt([b for _, b in e])} |")
    L += ["", "| host | ran, with avoided | ran, without |", "|---|---|---|"]
    for h in sorted({k[0] for k in ep_by if len(k) == 2}):
        L.append(f"| {h} | {fmt([a for a, _ in ep_by[(h, 'with an avoided object')]])} | "
                 f"{fmt([a for a, _ in ep_by[(h, 'without')]])} |")
    av = by[("avoided", "either")]
    rate = sum(av) / len(av) if av else float("nan")
    verdict = ("NO DATA (no avoided object)" if not av else
               f"STOP: avoided objects violated in {rate:.2f} > {STOP_RATE}" if rate > STOP_RATE else
               f"OK: avoided objects violated in {rate:.2f} <= {STOP_RATE}")
    L += ["", f"**Verdict: {verdict}**", ""]
    text = "\n".join(L)
    (root / "report.md").write_text(text)
    print(text)
    return 1 if (av and rate > STOP_RATE) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--embodiment", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--video-every", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("root")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
