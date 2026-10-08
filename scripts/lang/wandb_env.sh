# Source this (never execute it) to log online to wandb:   source scripts/lang/wandb_env.sh
# Loads WANDB_API_KEY from the user's activate script (WANDB_KEY_FILE, default Delta's
# /work/nvme/bifq/conda/activate_RoboTwin.sh) without printing it, and sets WANDB_MODE=online.
# Without a key line it leaves wandb offline. The key never appears in a command line or a log.
_wandb_file="${WANDB_KEY_FILE:-/work/nvme/bifq/conda/activate_RoboTwin.sh}"
_wandb_line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?WANDB_API_KEY=' "$_wandb_file" 2>/dev/null | tail -1)"
if [[ -n "$_wandb_line" ]]; then
  _wandb_val="${_wandb_line#*WANDB_API_KEY=}"
  _wandb_val="${_wandb_val%\"}"; _wandb_val="${_wandb_val#\"}"; _wandb_val="${_wandb_val%\'}"; _wandb_val="${_wandb_val#\'}"
  export WANDB_API_KEY="$_wandb_val"
  export WANDB_MODE=online
  echo "[wandb_env] WANDB_API_KEY loaded from $_wandb_file (online)"
else
  echo "[wandb_env] no WANDB_API_KEY line in $_wandb_file: wandb stays offline"
fi
unset _wandb_file _wandb_line _wandb_val
