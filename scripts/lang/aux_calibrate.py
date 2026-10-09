"""Pick the auxiliary-loss weight from a short calibration run (review round 4, item 3).

    python scripts/lang/aux_calibrate.py --run <calibration run> --key aux/ratio --weight 1.0 --target 0.2 \
        --out <file>

The calibration run trains with the auxiliary loss at weight `--weight` for ~500 steps. Its train_log.jsonl
logs `--key` = weight x L_aux / L_critic per logged step; the median over steps [lo, hi] (default 400-600)
divided by the weight is L_aux / L_critic, and the chosen weight is target / that. Writes the weight as a
plain number to `--out` (read by the queued V4 / V4b commands with $(cat ...)) and the details to
`--out`.json. Exit 75 (not ready) when the run has not logged a step >= lo yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--key", required=True, help="aux/ratio (V4) or aux/att_ratio (V4b)")
    ap.add_argument("--weight", type=float, required=True, help="the weight the calibration run used")
    ap.add_argument("--target", type=float, default=0.2, help="wanted weight x L_aux / L_critic")
    ap.add_argument("--lo", type=int, default=400)
    ap.add_argument("--hi", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    f = Path(args.run) / "train_log.jsonl"
    rows = []
    if f.exists():
        for ln in f.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    sel = [float(r[args.key]) for r in rows if r.get(args.key) is not None and args.lo <= int(r.get("step", -1)) <= args.hi]
    if not sel:
        print(f"[aux_calibrate] {f}: no {args.key} logged in steps {args.lo}-{args.hi} yet")
        return 75
    ratio = float(np.median(sel)) / args.weight          # L_aux / L_critic
    w = args.target / max(ratio, 1e-12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(f"{w:.6g}\n")
    Path(args.out + ".json").write_text(json.dumps({"run": args.run, "key": args.key, "calib_weight": args.weight,
                                                    "steps": [args.lo, args.hi], "n": len(sel),
                                                    "median_ratio": float(np.median(sel)),
                                                    "aux_over_critic": ratio, "target": args.target,
                                                    "weight": w}, indent=1))
    print(f"[aux_calibrate] {args.key}: L_aux / L_critic = {ratio:.4g} over {len(sel)} logs -> weight {w:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
