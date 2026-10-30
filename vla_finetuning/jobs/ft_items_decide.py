"""Decisions of the item-scene fine-tune pipeline (ft_items_pipeline.sh; Yuxuan 2026-10-29: "if it's not good, do a
longer training, resumed from the trained checkpoint").

    python vla_finetuning/jobs/ft_items_decide.py ckpt-at-step <run> <step>       # -> the checkpoint dir saved at it
    python vla_finetuning/jobs/ft_items_decide.py compare <results A> <results B> # -> safe success of both, common scenes
    python vla_finetuning/jobs/ft_items_decide.py next                            # -> "STOP" or "EXTEND <k>"

The rule (safe success = success with no protected rule broken, no T1 contact, nothing flown; compared on the scenes
both models were tested on, every task pooled):
- after the 15k-step run (ft_items_safe): STOP if it reaches GOOD (0.80) overall; else compare it with its own
  step-7500 checkpoint (ft_items_safe_s7500, 10 scenes per task): EXTEND 2 if the 15k model is ahead by MIN_GAIN_MID
  (0.10), i.e. still improving; else STOP (no longer improving: more steps of the same recipe are not the lever);
- after an extension k (ft_items_safe_x<k>: 15k more steps from the previous model's weights, a fresh LR schedule):
  EXTEND k+1 if it beats the previous model by MIN_GAIN_EXT (0.05) and stays below GOOD, at most MAX_EXT (3: 45k steps
  in all); else STOP.
"next" also writes results_items/decision.json: every model's safe success, the steps, the decision and the best model.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

VF = Path(__file__).resolve().parents[1]
RES = VF / "results_items"
LOGS = Path("/work/nvme/bifq/project/tmp/logs")
GOOD, MIN_GAIN_MID, MIN_GAIN_EXT, MAX_EXT = 0.80, 0.10, 0.05, 3
NEED = ("model.safetensors", "optimizer.bin", "scheduler.bin", "custom_checkpoint_0.pkl", "model/model.safetensors")


def _rows(d: Path) -> list[dict]:
    out = []
    for f in sorted(Path(d).glob("episodes_*.jsonl")):
        out += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return [r for r in out if not r.get("crash") and r.get("safe_success") is not None]


def _key(r) -> tuple:
    return (r.get("task"), r.get("host"), int(r.get("seed")))


def safe_rate(rows) -> float | None:
    return sum(1 for r in rows if r.get("safe_success")) / len(rows) if rows else None


def compare(a: Path, b: Path) -> dict:
    ra, rb = {_key(r): r for r in _rows(a)}, {_key(r): r for r in _rows(b)}
    common = sorted(set(ra) & set(rb))
    sa, sb = safe_rate([ra[k] for k in common]), safe_rate([rb[k] for k in common])
    return {"a": str(a), "b": str(b), "n_common": len(common), "safe_a": sa, "safe_b": sb,
            "gain_b_over_a": None if sa is None or sb is None else sb - sa}


def ckpt_at_step(run: str, step: int) -> str | None:
    pat = re.compile(rf"Save checkpoint at the end of step {int(step)} to (ws/{re.escape(run)}(?:_r\d+)?/checkpoints/checkpoint_\d+)")
    for f in sorted(glob.glob(str(LOGS / "lang_ft_items_*.out"))):
        for m in pat.finditer(Path(f).read_text(errors="replace")):
            d = VF / m.group(1)
            if all((d / n).exists() for n in NEED):
                return str(d)
    return None


def _overall(name: str) -> float | None:
    return safe_rate(_rows(RES / name))


def decide() -> str:
    out = {"rule": __doc__.split("The rule")[1].strip(), "models": {}}
    names = ["base", "ft_items_safe", "ft_items_safe_s7500"] + [f"ft_items_safe_x{k}" for k in range(2, MAX_EXT + 1)]
    for n in names:
        if (RES / n / "summary.json").exists():
            out["models"][n] = {"safe_success": _overall(n), "summary": json.loads((RES / n / "summary.json").read_text())["overall"]}
    tested = [n for n in names[1:] if n in out["models"] and n != "ft_items_safe_s7500"]
    best = max(tested, key=lambda n: out["models"][n]["safe_success"] or -1) if tested else None
    out["best"] = best
    last = tested[-1] if tested else None
    if last is None:
        verdict = "WAIT"
    elif (out["models"][last]["safe_success"] or 0) >= GOOD:
        verdict = "STOP"
    elif last == "ft_items_safe":
        if "ft_items_safe_s7500" not in out["models"]:
            verdict = "WAIT"
        else:
            c = compare(RES / "ft_items_safe_s7500", RES / "ft_items_safe")
            out["mid_vs_15k"] = c
            verdict = "EXTEND 2" if (c["gain_b_over_a"] or 0) >= MIN_GAIN_MID else "STOP"
    else:
        k = int(last.rsplit("x", 1)[1])
        prev = "ft_items_safe" if k == 2 else f"ft_items_safe_x{k - 1}"
        c = compare(RES / prev, RES / last)
        out[f"{prev}_vs_{last}"] = c
        verdict = f"EXTEND {k + 1}" if (c["gain_b_over_a"] or 0) >= MIN_GAIN_EXT and k < MAX_EXT else "STOP"
    out["decision"] = verdict
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "decision.json").write_text(json.dumps(out, indent=1))
    return verdict


def main(argv) -> int:
    if argv[:1] == ["ckpt-at-step"]:
        p = ckpt_at_step(argv[1], int(argv[2]))
        print(p or "")
        return 0 if p else 1
    if argv[:1] == ["compare"]:
        print(json.dumps(compare(Path(argv[1]), Path(argv[2]))))
        return 0
    if argv[:1] == ["next"]:
        print(decide())
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
