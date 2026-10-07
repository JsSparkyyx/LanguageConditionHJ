"""Section 2 regression: main/train/train.py still imports and runs 20 grad
steps on a CrossSafe-format StepBuffer (no CrossSafe dataset exists on Delta,
so a Phase-0 recon StepBuffer is used; its h is the min over the item rules).

    python scripts/lang/crosssafe_regression.py --buffer <root>/buffer --run <tmp run> --out <json>
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main.train.config import FrozenConfig  # noqa: E402
from main.train.train import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    cfg = FrozenConfig()
    t0 = time.time()
    tr = Trainer(cfg, Path(a.run), Path(a.run), buffer_dir=Path(a.buffer))
    outs = [tr.grad_step() for _ in range(a.steps)]
    res = {"steps": len(outs), "loss_q_first": outs[0]["loss_q"], "loss_q_last": outs[-1]["loss_q"],
           "finite": all(all(v == v for v in o.values() if isinstance(v, float)) for o in outs),
           "wall_s": time.time() - t0}
    res["ok"] = bool(res["finite"] and res["steps"] == a.steps)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
