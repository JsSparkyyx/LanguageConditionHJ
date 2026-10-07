#!/usr/bin/env bash
# One-time setup check (login or GPU node, ~5 min): environment, RoboTwin task links, checkpoints, data,
# and the CPU test suites. Prints FAIL lines and exits 1 if anything a run needs is missing.
#
#   HOLOBRAIN_CKPT=<dir> PY=<python> bash scripts/lang/bridges/check.sh
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

echo "== RoboTwin (must be <repo>/RoboTwin or <repo>/../RoboTwin; main/envs/env.py ignores ROBOTWIN_ROOT)"
RT=""
for c in "${REPO_ROOT}/RoboTwin" "${REPO_ROOT}/../RoboTwin"; do [[ -d "$c/envs" ]] && { RT="$(cd "$c" && pwd)"; break; }; done
if [[ -z "$RT" ]]; then fail "RoboTwin not found next to the repo"; else
  ok "RoboTwin at $RT"
  [[ -d "$RT/assets/objects" ]] && ok "RoboTwin assets" || fail "RoboTwin assets missing ($RT/assets/objects)"
  # the links copied from Delta point at Delta paths: (re)install them (only adds symlinks).
  # SKIP_LINKS=1 only verifies them (while simulations of this checkout are running)
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
fi

echo "== checkpoints"
if [[ -f "$HOLOBRAIN_CKPT/model.config.json" || -f "$HOLOBRAIN_CKPT/pretrain/model.config.json" ]]; then
  ok "HoloBrain checkpoint $HOLOBRAIN_CKPT"; else fail "no model.config.json under HOLOBRAIN_CKPT=$HOLOBRAIN_CKPT"; fi
$PY -c "from transformers import AutoTokenizer, BertConfig; AutoTokenizer.from_pretrained('bert-base-uncased'); BertConfig.from_pretrained('bert-base-uncased')" \
  >/dev/null 2>&1 && ok "bert-base-uncased tokenizer + config (HF_HOME=$HF_HOME)" \
  || fail "bert-base-uncased not loadable: copy Delta's hf-cache/hub/models--bert-base-uncased into \$HF_HOME/hub (or allow internet)"

echo "== data"
$PY - <<'EOF' || bad=1
import json, sys
from pathlib import Path
bad = False
for name, n_bead in (("warmup", 30), ("val", 10)):
    root = Path("data/lang") / name
    if not root.is_dir():
        print(f"FAIL: {root} missing"); bad = True; continue
    shards = sorted(p for p in root.glob("shard_g*") if (p / "item_buffer").is_dir())
    old = [p for p in shards if not p.name.endswith("_bead")]
    bead = [p for p in shards if p.name.endswith("_bead")]
    done = lambda p: (p / "summary.json").exists() and json.loads((p / "summary.json").read_text()).get("complete")
    eps = sum(sum(1 for l in open(p / "lang_episodes.jsonl") if '"skipped"' not in l)
              for p in shards if (p / "lang_episodes.jsonl").exists())
    print(f"ok:   {root}: {len(old)} original + {len(bead)} bead shards, {eps} episodes")
    if [p.name for p in old if not done(p)]:
        print(f"FAIL: incomplete original shards in {root}"); bad = True
    nb = sum(1 for p in bead if done(p))
    if nb < n_bead:
        print(f"FAIL: {root}: {nb}/{n_bead} bead shards complete (run collect_bead.sh or copy them from Delta); "
              f"start the rounds run only after all are complete")
        bad = True
    if not (root / "text_cache.pt").exists():
        print(f"FAIL: {root}/text_cache.pt missing"); bad = True
sys.exit(1 if bad else 0)
EOF

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
