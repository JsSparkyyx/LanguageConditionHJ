#!/usr/bin/env bash
# Setup check (login or GPU node, ~5 min): environment, every path the safety filter reads (RoboTwin and its robot
# files, the item task links, checkpoints, the safe planner's robot model, data), and the CPU test suites. Prints
# FAIL lines and exits 1 if anything a run needs is missing.
#
#   HOLOBRAIN_CKPT=<dir> PY=<python> bash scripts/lang/bridges/check.sh
#
# Optional: DATA_DIRS="<dir> ..." the data splits to check (default: data/lang/{warmup,val,val20,val_grounding}
#           that exist); QWEN=1 also checks the HoloBrain-Qwen checkpoints and flash-attn (the Qwen runs);
#           SKIP_LINKS=1 only verifies the RoboTwin task links instead of (re)installing them.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
bad=0
fail() { echo "FAIL: $*"; bad=1; }
ok() { echo "ok:   $*"; }

echo "== python"
$PY - <<'EOF' || fail "python packages (torch / sapien / curobo / transformers)"
import torch, sapien, transformers
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} sapien {sapien.__version__} transformers {transformers.__version__}")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
import curobo  # noqa: F401
EOF
if [[ "${QWEN:-0}" == 1 ]]; then
  $PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)" \
    && ok "flash-attn (the Qwen runs' kernel)" || fail "flash-attn missing (the Qwen data and runs use flash_attention_2)"
fi

echo "== RoboTwin (must be <repo>/RoboTwin or <repo>/../RoboTwin; main/envs/env.py ignores ROBOTWIN_ROOT)"
RT=""
for c in "${REPO_ROOT}/RoboTwin" "${REPO_ROOT}/../RoboTwin"; do [[ -d "$c/envs" ]] && { RT="$(cd "$c" && pwd)"; break; }; done
if [[ -z "$RT" ]]; then fail "RoboTwin not found next to the repo"; else
  ok "RoboTwin at $RT"
  [[ -d "$RT/assets/objects" ]] && ok "RoboTwin assets" || fail "RoboTwin assets missing ($RT/assets/objects)"
  # RoboTwin's cuRobo robot files name the URDF and sphere files by absolute path. RoboTwin's asset download writes
  # them (scripts/update_embodiment_config_path.py); a RoboTwin copied from Delta still names Delta's files, and
  # cuRobo then fails to load: every plan silently falls back to mplib
  for e in ur5-wsg franka-panda; do
    y="$RT/assets/embodiments/$e/curobo.yml"
    [[ -f "$y" ]] || { fail "$y missing"; continue; }
    paths=$(grep -E '^\s*(urdf_path|collision_spheres):' "$y" | awk '{print $2}')
    badp=$(for p in $paths; do [[ -f "$p" && ( "$p" == "$RT"/* || "$(realpath "$p")" == "$(realpath "$RT")"/* ) ]] || echo "$p"; done)
    if [[ -z "$badp" ]]; then ok "$e cuRobo model names this RoboTwin's files"
    else fail "$y names files outside $RT or missing: $(echo $badp | head -c 200) -> run: cd $RT && $PY scripts/update_embodiment_config_path.py"; fi
  done
  # the item task links (RoboTwin/envs/<task>.py): code since 2026-10-23 imports this checkout's task files itself,
  # the links serve older checkouts and tools. SKIP_LINKS=1 only verifies them (while simulations are running)
  if [[ "${SKIP_LINKS:-0}" != 1 ]]; then
    ROBOTWIN_ROOT="$RT" bash "${REPO_ROOT}/envs/scripts/item_envs/install_robotwin_tasks.sh" >/dev/null \
      || fail "install_robotwin_tasks.sh"
  fi
  nl=0
  for f in "${REPO_ROOT}"/envs/robotwin_tasks/*.py; do
    l="$RT/envs/$(basename "$f")"
    [[ "$(readlink -f "$l")" == "$(readlink -f "$f")" ]] || { fail "$l does not link to $f"; nl=1; }
  done
  [[ $nl -eq 0 ]] && ok "item task links in $RT/envs point at this checkout"
  # the safe planner's robot model (envs/main/item_envs/configs/curobo_safe/ur5-wsg.yml names Delta's RoboTwin
  # files; safe_plan.safe_yml_for writes a copy naming this RoboTwin's): both files must exist
  $PY - <<'EOF' && ok "safe planner robot model resolves to this RoboTwin" || fail "safe planner robot model (safe_plan.safe_yml_for)"
import re, sys
import main.lang  # noqa: F401  (main.item_envs on the path)
from main.item_envs import safe_plan as S
y = S.safe_yml_for(S.robotwin_root())
paths = re.findall(r"(?m)^\s*(?:urdf_path|collision_spheres):\s*(\S+)", y.read_text())
missing = [p for p in paths if not __import__("os").path.isfile(p)]
print(f"      {y}" + (f": MISSING {missing}" if missing else ""))
sys.exit(1 if missing or len(paths) != 2 else 0)
EOF
fi

echo "== checkpoints"
if [[ -f "$HOLOBRAIN_CKPT/model.config.json" || -f "$HOLOBRAIN_CKPT/pretrain/model.config.json" ]]; then
  ok "HoloBrain checkpoint $HOLOBRAIN_CKPT"; else fail "no model.config.json under HOLOBRAIN_CKPT=$HOLOBRAIN_CKPT"; fi
$PY -c "from transformers import AutoTokenizer, BertConfig; AutoTokenizer.from_pretrained('bert-base-uncased'); BertConfig.from_pretrained('bert-base-uncased')" \
  >/dev/null 2>&1 && ok "bert-base-uncased tokenizer + config (HF_HOME=$HF_HOME)" \
  || fail "bert-base-uncased not loadable: copy Delta's hf-cache/hub/models--bert-base-uncased into \$HF_HOME/hub (or allow internet)"
if [[ "${QWEN:-0}" == 1 ]]; then
  $PY - <<'EOF' && ok "HoloBrain-Qwen checkpoints" || fail "HoloBrain-Qwen checkpoints (README section 1: HOLOBRAIN_QWEN_CKPT / QWEN_BASE_DIR or \$HF_HOME/hub)"
import os, sys
from main.lang.qwen_encoder import QWEN_BASE, QWEN_CKPT
bad = 0
for name, d in (("HOLOBRAIN_QWEN_CKPT", QWEN_CKPT), ("QWEN_BASE_DIR", QWEN_BASE)):
    ok = os.path.isdir(d)
    bad |= not ok
    print(f"      {name}: {d}" + ("" if ok else "  MISSING"))
sys.exit(bad)
EOF
fi

echo "== data"
if [[ -z "${DATA_DIRS:-}" ]]; then
  DATA_DIRS=""
  for k in warmup val val20 val_grounding; do [[ -d "data/lang/$k" ]] && DATA_DIRS="$DATA_DIRS data/lang/$k"; done
fi
if [[ -z "${DATA_DIRS// }" ]]; then
  echo "note: no data yet (data/lang/*): copy it or collect it (collect.sbatch); give other folders with DATA_DIRS"
else
  # shellcheck disable=SC2086
  $PY - $DATA_DIRS <<'EOF' || bad=1
import json, sys
from pathlib import Path
bad = False
for root in map(Path, sys.argv[1:]):
    if not root.is_dir():
        print(f"FAIL: {root} missing"); bad = True; continue
    shards = sorted(p for p in root.glob("shard_g*") if (p / "item_buffer").is_dir())
    done = lambda p: (p / "summary.json").exists() and json.loads((p / "summary.json").read_text()).get("complete")
    eps = sum(sum(1 for l in open(p / "lang_episodes.jsonl") if '"skipped"' not in l)
              for p in shards if (p / "lang_episodes.jsonl").exists())
    inc = [p.name for p in shards if not done(p)]
    print(f"ok:   {root}: {len(shards)} shards, {eps} episodes" if shards and not inc else
          f"FAIL: {root}: {len(shards)} shards, incomplete: {inc[:5]}")
    bad |= not shards or bool(inc)
    if (root / "kv_buffer").exists() or any((p / "kv_buffer").is_dir() for p in shards):
        continue   # Qwen replay shards: no BERT text cache
    if not (root / "text_cache.pt").exists():
        print(f"FAIL: {root}/text_cache.pt missing (collect.sh builds it; or copy it with the shards)"); bad = True
sys.exit(1 if bad else 0)
EOF
  echo "      (integrity of every row: $PY scripts/lang/check_shards.py <dirs>, ~1 min per 10 GB)"
fi

echo "== CPU tests"
pytest_ok() {   # the summary line must say passed, with no failed / error
  local out; out=$("$@" 2>&1 | tail -1); echo "      $out"
  [[ "$out" == *passed* && "$out" != *failed* && "$out" != *error* ]]
}
pytest_ok $PY -m pytest tests/lang -q -m "not sim and not gpu" -p no:cacheprovider \
  && ok "tests/lang" || fail "tests/lang (rerun: $PY -m pytest tests/lang -m 'not sim and not gpu')"
(cd envs && pytest_ok $PY -m pytest tests/item_envs -q -m "not sim and not gpu" -p no:cacheprovider) \
  && ok "envs/tests/item_envs" || fail "envs/tests/item_envs (rerun from envs/)"

[[ $bad -eq 0 ]] && echo "ALL OK" || { echo "SOME CHECKS FAILED"; exit 1; }
