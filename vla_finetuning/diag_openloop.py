"""Open-loop check of the HoloBrain policy pipeline on recorded demos (no simulator; any GPU).

For frames of recorded episodes (classic hdf5 written by collect_demos), build exactly the observation the
evaluator's HoloBrainPolicy builds from task.get_obs() (RGB decoded from the JPEG, depth mm, intrinsics,
extrinsics, joint_action/vector, the episode's instruction), run the model's inference pipeline, and compare
the 64 predicted joint rows with the recorded rows t+1..t+64. Baseline: holding the current joints.
Small error vs the baseline = the observation and action conventions agree with what the model was trained
on; a closed-loop failure is then the model, not the plumbing.

    scripts/lang/in_container.sh python -m vla_finetuning.diag_openloop --model-dir <dir> <episode.hdf5> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C


def frames_from_hdf5(path: Path):
    import cv2
    import h5py

    with h5py.File(path, "r") as f:
        joints = f["joint_action/vector"][:]
        cams = {}
        for cam in C.CAMS:
            g = f["observation"][cam]
            cams[cam] = {"rgb": g["rgb"][:], "depth": g["depth"][:], "K": g["intrinsic_cv"][:],
                         "ext": g["extrinsic_cv"][:]}
    def obs(t):
        o = {"observation": {}, "joint_action": {"vector": joints[t]}}
        for cam, d in cams.items():
            img = cv2.imdecode(np.frombuffer(d["rgb"][t], np.uint8), cv2.IMREAD_UNCHANGED)
            o["observation"][cam] = {"rgb": img, "depth": d["depth"][t].astype(np.float64),
                                     "intrinsic_cv": d["K"][t], "extrinsic_cv": d["ext"][t]}
        return o
    return joints, obs


def main(argv=None) -> int:
    import torch

    from .evaluate import HoloBrainPolicy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--every", type=int, default=10, help="frame stride")
    ap.add_argument("--out", default=None)
    ap.add_argument("episodes", nargs="+")
    args = ap.parse_args(argv)
    pol = HoloBrainPolicy(Path(args.model_dir), 64)
    arm = np.r_[0:6, 7:13]                      # arm joints, grippers excluded
    rows = []
    for ep in args.episodes:
        ep = Path(ep)
        instr_p = ep.parent.parent / "instructions" / ep.name.replace(".hdf5", ".json")
        instr = json.loads(instr_p.read_text())["seen"][0] if instr_p.exists() else ""
        joints, obs = frames_from_hdf5(ep)
        T = len(joints)
        for t in range(0, T - 8, args.every):
            torch.manual_seed(0)
            pred = pol.pipe(pol._input(obs(t), instr)).action.cpu().numpy()      # [64, 14] absolute joints
            fut = np.stack([joints[min(t + 1 + i, T - 1)] for i in range(64)])
            n = min(len(pred), 64)
            for h in (8, 32, 64):
                hh = min(h, n)
                e_model = float(np.abs(pred[:hh][:, arm] - fut[:hh][:, arm]).mean())
                e_hold = float(np.abs(joints[t][arm][None] - fut[:hh][:, arm]).mean())
                rows.append({"ep": ep.name, "t": t, "h": h, "model": e_model, "hold": e_hold})
            print(f"[diag] {ep.parent.parent.parent.name}/{ep.name} t={t}/{T}: h8 model {rows[-3]['model']:.3f} "
                  f"hold {rows[-3]['hold']:.3f} | h32 model {rows[-2]['model']:.3f} hold {rows[-2]['hold']:.3f} | "
                  f"first pred - current {np.abs(pred[0][arm] - joints[t][arm]).max():.3f} rad", flush=True)
    for h in (8, 32, 64):
        m = np.mean([r["model"] for r in rows if r["h"] == h])
        b = np.mean([r["hold"] for r in rows if r["h"] == h])
        print(f"[diag] horizon {h}: mean |pred - demo| {m:.3f} rad, hold-still baseline {b:.3f} rad")
    if args.out:
        Path(args.out).write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
