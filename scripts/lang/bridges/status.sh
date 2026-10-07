#!/usr/bin/env bash
# Progress of a rounds run (login node, reads files only):
#
#   bash scripts/lang/bridges/status.sh runs/lang/<name>
#
# Prints the round / phase / step the saved state is at, the latest losses, the per-round filter-episode
# numbers, and where the newest videos and plots are.
RUN="${1:?usage: status.sh <run dir>}"
PY="${PY:-python}"
$PY - "$RUN" <<'EOF'
import json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
st = run / "state.pt"
if st.exists():
    try:
        import torch
        s = torch.load(st, map_location="cpu", weights_only=False)
        print(f"state: round {s.get('round')} phase {s.get('phase') or 'start'} step {s.get('step')} "
              f"(episodes {s.get('slots_done', 0)}/{len(s.get('slots') or [])} of this round)  "
              f"saved {time.ctime(st.stat().st_mtime)}")
    except Exception as exc:
        print(f"state.pt unreadable ({exc})")
else:
    print("no state.pt yet")
rows = [json.loads(l) for l in open(run / "train_log.jsonl")] if (run / "train_log.jsonl").exists() else []
last = next((r for r in reversed(rows) if "loss/q" in r and "q/mean" in r), None)
if last:
    keys = (("step", "step"), ("loss/q", "critic loss"), ("loss/pi", "actor loss"), ("q/mean", "mean Q"),
            ("h/mean", "mean h"), ("v_le_h_frac", "Q<=h"), ("lang/pair_rank_acc", "pair rank"),
            ("perf/steps_per_sec", "steps/s"))
    print("train: " + "  ".join(f"{name} {last[k]:.4g}" for k, name in keys if last.get(k) is not None))
rnd = [r for r in rows if "round/collect_s" in r]
if rnd:
    print("round  harm runs: harmed / intervened | benign runs: intervened / success | episodes  collect min")
    for r in rnd[-10:]:
        g = lambda k: r.get(k)  # noqa: E731
        f = lambda v: "  -  " if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{int(r['round/index']):5d}  {f(g('round_overview/harm/harm_named'))} / {f(g('round_overview/harm/any_intervention'))}"
              f"          | {f(g('round_overview/benign/any_intervention'))} / {f(g('round_overview/benign/task_success'))}"
              f"          | {int(g('round_overview/n_filter_episodes') or 0):3d}      {r['round/collect_s'] / 60:5.1f}")
vids = sorted((run / "ring" / "videos").glob("*.mp4"), key=lambda p: p.stat().st_mtime) if (run / "ring" / "videos").exists() else []
print(f"videos: {len(vids)} in {run / 'ring' / 'videos'}" + ("; newest: " + ", ".join(v.name for v in vids[-3:]) if vids else ""))
for p in ("eval/trend.png", "eval/harm_named", "eval/any_intervention", "probes"):
    if (run / p).exists():
        print(f"plots: {run / p}")
EOF
