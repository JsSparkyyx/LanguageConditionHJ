#!/usr/bin/env bash
# GPU end-to-end check before the long run (~25-35 min on one GPU): a tiny rounds run (2 rounds x 2
# episodes x 64 steps) that is stopped by a time budget, resumed, and finished; then the outputs are
# checked. Exercises rendering, cuRobo, the HoloBrain encoder, filter episodes, videos, metrics and the
# mid-round stop / resume.
#
#   HOLOBRAIN_CKPT=<dir> PY=<python> bash scripts/lang/bridges/smoke.sh
# Smaller: SMOKE_EPISODES=1 SMOKE_STEPS=32 SMOKE_BUDGET=300 (~20 min).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
EPS=${SMOKE_EPISODES:-2}; STEPS=${SMOKE_STEPS:-64}; BUDGET=${SMOKE_BUDGET:-480}
RUN="${REPO_ROOT}/runs/lang/bridges_smoke_$(date +%m%d_%H%M%S)"
ARGS=(--data data/lang/warmup --val data/lang/val --run "$RUN" --collect-rounds 2 --episodes-per-round "$EPS"
      --grad-steps-per-round "$STEPS" --probe-every-rounds 2 --video-every 1 --no-wandb --resume)
LOG="${LOG_DIR}/smoke_$(basename "$RUN").log"
echo "[smoke] run $RUN, log $LOG"
# 1. stop after BUDGET seconds: lands in round 0's training or round 1's collection
$PY -m main.lang.train rounds "${ARGS[@]}" --time-budget "$BUDGET" >> "$LOG" 2>&1
rc1=$?
echo "[smoke] first part exit $rc1 (want 75: stopped by the budget)"
# 2. resume to the end (relaunching on 75, like run_rounds.sh)
rc2=75; tries=0
while [[ $rc2 -eq 75 && $tries -lt 5 ]]; do
  $PY -m main.lang.train rounds "${ARGS[@]}" >> "$LOG" 2>&1
  rc2=$?; tries=$((tries + 1))
done
echo "[smoke] resumed part exit $rc2 (want 0)"
grep -q "CuRobo init failed" "$LOG" && echo "[smoke] WARNING: cuRobo fell back to mplib (TMPDIR? GPU?): see $LOG"
grep -m3 -E "Traceback|Error" "$LOG" | head -3
$PY - "$RUN" "$rc1" "$rc2" "$EPS" "$STEPS" <<'EOF'
import json, sys
from pathlib import Path
run, rc1, rc2, n_eps, n_steps = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
bad = []
lines = [json.loads(l) for l in (run / "ring" / "lang_episodes.jsonl").read_text().splitlines()] \
    if (run / "ring" / "lang_episodes.jsonl").exists() else []
eps = [d for d in lines if not d.get("skipped")]
print(f"[smoke] episodes {len(eps)} (want {2 * n_eps}): " + ", ".join(
    f"r{d['round']} {d['task']}/{d['embodiment']} {'FILTER' if d['filter'] else 'nominal'} {d['status']}" for d in eps))
if rc1 != 75: bad.append(f"first part exit {rc1}, not 75")
if rc2 != 0: bad.append(f"resumed part exit {rc2}")
if len(eps) != 2 * n_eps: bad.append(f"{len(eps)} episodes, not {2 * n_eps}")
if len({d['episode_id'] for d in eps}) != len(eps): bad.append("duplicate episode ids")
if any(d["status"] != "ok" for d in eps): bad.append("an episode did not finish ok")
for f in ("checkpoint_final.pt", "eval/trend.png", "eval/rounds.jsonl", "train_log.jsonl"):
    if not (run / f).exists(): bad.append(f"missing {f}")
vids = sorted((run / "ring" / "videos").glob("*.mp4")) if (run / "ring" / "videos").exists() else []
print(f"[smoke] videos: {[v.name for v in vids]}")
if any(d["filter"] for d in eps) and not vids: bad.append("filter episodes but no videos")
rows = [json.loads(l) for l in open(run / "train_log.jsonl")] if (run / "train_log.jsonl").exists() else []
steps = max((r.get("step", 0) for r in rows), default=0)
print(f"[smoke] last logged step {steps} (want {3 * n_steps} = 3 rounds x {n_steps})")
if steps != 3 * n_steps: bad.append(f"step {steps}, not {3 * n_steps}")
print("[smoke] " + ("ALL OK" if not bad else "FAILED: " + "; ".join(bad)))
sys.exit(1 if bad else 0)
EOF
