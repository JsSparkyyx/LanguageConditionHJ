#!/bin/bash
# Copy what the action-sensitivity experiment (ACTION_SENSITIVITY_PLAN) reads to another machine.
#
#   scripts/lang/action_sens/portable/transfer.sh <source repo> <destination repo> [what ...]
#
# One of the two repos may be remote (rsync form user@host:/path/LanguageConditionHJ), so the same script pushes
# from Delta or pulls from the other machine:
#   on Delta:        transfer.sh . me@other:/path/LanguageConditionHJ
#   on the other:    transfer.sh me@login.delta.ncsa.illinois.edu:/work/nvme/bifq/project/LanguageConditionHJ .
# The destination repo is a git clone of this repo; the files land at the same relative paths, which is where the
# code looks for them (every path of train_critic.py / collect_branches.py is relative to the repo).
#
# what (default: train):
#   train    everything the critic trainings and the scoring read (~110 GB real size on 2026-11-11):
#              data/lang_1023/warmup (29 GB) and the text caches of val, val20      the normal data, part 1
#              runs/lang/rounds_ur5_g095_1023/{lang_config.json, ring} (53 GB)      settings + the normal data, part 2
#              runs/lang/diag_1030/retrain/action_norm.json                         the revised injection's bounds
#              data/lang_asens (branches + states, 17 GB)                           round-1 branch data
#              data/lang_asens2 (branches + states + episodes)                      round-2 branch data so far
#              runs/lang/action_sens (criteria, finished runs, scores; 5 GB)
#   asens2   only data/lang_asens2 (run it again after more collection jobs: rsync sends what is new)
#   parents  the planner episodes of both collections (not read by the trainer; 3 GB + )
#   step1    the 117 old branch states of step 1 (runs/lang/diag_1030/t5_branch, 0.3 GB)
# Delta asks for a password and Duo on every connection and this script makes several: open one shared connection
# first and let rsync reuse it (on the other machine, pulling):
#   ssh -M -S ~/.ssh/cm-delta -fN me@login.delta.ncsa.illinois.edu
#   RSYNC_RSH="ssh -S ~/.ssh/cm-delta" transfer.sh me@login.delta.ncsa.illinois.edu:/work/nvme/bifq/project/LanguageConditionHJ .
#   ssh -S ~/.ssh/cm-delta -O exit me@login.delta.ncsa.illinois.edu       # when done
# Interrupted? Run the same command again: finished files are skipped.
# DRY=1 lists what would be sent. The buffer files are sparse (created at full size, mostly empty): rsync --sparse
# keeps them small. Globus does not: there, pack with `tar -S` first (see NEW_MACHINE_SETUP.md).
# Do not copy data/lang_asens2 while a collection job is writing to it: wait for the job to end.
# Not copied here, needed too: the HoloBrain GD checkpoint and bert-base-uncased (NEW_MACHINE_SETUP.md section 1),
# and, for the collection only, RoboTwin with its assets and the simulator environment.
set -eu
SRC=${1:?source repo}; DST=${2:?destination repo}; shift 2
WHAT=${*:-train}
# -K: a destination directory that is a symlink (e.g. data -> another storage) is followed, not replaced
FLAGS=(-a --sparse --partial --info=progress2,stats1 -R -K)
[ "${DRY:-0}" = 1 ] && FLAGS+=(--dry-run)
EXC=(--exclude 'videos/' --exclude '_runner/' --exclude 'locks/' --exclude 'logs/' --exclude 'score_tmp/'
     --exclude 'smoke/' --exclude '*.tmp' --exclude 'wandb/')
send() { local p; local a=(); for p in "$@"; do a+=("$SRC/./$p"); done; rsync "${FLAGS[@]}" "${EXC[@]}" "${a[@]}" "$DST/"; }
for w in $WHAT; do
  case $w in
    train)
      send runs/lang/rounds_ur5_g095_1023/lang_config.json runs/lang/diag_1030/retrain/action_norm.json \
           data/lang_1023/val/text_cache.pt data/lang_1023/val20/text_cache.pt
      send runs/lang/action_sens
      send data/lang_asens/branches data/lang_asens/states data/lang_asens/episodes data/lang_asens/branch_table.json \
           data/lang_asens/COLLECT_DONE
      send data/lang_asens2/branches data/lang_asens2/states data/lang_asens2/episodes
      send data/lang_1023/warmup
      send runs/lang/rounds_ur5_g095_1023/ring ;;
    asens2)  send data/lang_asens2/branches data/lang_asens2/states data/lang_asens2/episodes ;;
    parents) send data/lang_asens/parents data/lang_asens2/parents ;;
    step1)   send runs/lang/diag_1030/t5_branch/states runs/lang/diag_1030/t5_branch/results.json ;;
    *) echo "unknown: $w (train | asens2 | parents | step1)"; exit 2 ;;
  esac
done
