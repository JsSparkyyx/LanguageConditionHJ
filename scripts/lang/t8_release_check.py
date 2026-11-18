#!/usr/bin/env python
"""What happens after a filter release inside the T8 drag, with a SCRIPTED filter (2026-11-17).

    python scripts/lang/t8_release_check.py run [--out runs/lang/t8_release_check] [--seeds 8301 8302]
    python scripts/lang/t8_release_check.py table [--out ...]

No model: the "filter" is a script. A planner-only T8 episode runs to a point of the drag (the tray --engage of the
way along). There the script takes over as an engaged filter does: actor-style commands through env.step_dtheta,
held in 3-tick blocks, a0 x g with a0 = the arm's joint displacement over the last tick and g from the script's
list. After the list the script releases, and the tick controller replans the drag as it does after any filter
release. Each (seed, behaviour, script) runs with both controllers:

    replay   RiskTickController: the replan runs every original waypoint again on the original time law
    resume   main/lang/controller.py ResumeViaController: the waypoints ahead, the remaining length, from the arm's speed

Scripts:
    gentle_release   g = 0.85, 0.7, 0.55, 0.4: slow down over 4 blocks, release while the tray still moves
    slow_to_stop     g = 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0, 0: slow to rest over 8 blocks, then release
    hard_stop        g = 0, 0: two blocks of the zero command (what the 1115 actors did), then release

Per run: the tray's progress along the drag every tick, each item's h, and the summary: how far the tray went BACK
after the release, its peak speed after the release, whether the glass / the can fell and when (during the script
or after the release), and whether the tray reached the pad.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

OUT = REPO / "runs/lang/t8_release_check"
CFG = REPO / "runs/lang/rounds_t8_3tick_boundary_1117/lang_config.json"
SCRIPTS = {"gentle_release": [0.85, 0.7, 0.55, 0.4],
           "slow_to_stop": [0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0, 0.0],
           "hard_stop": [0.0, 0.0]}
TICK_DT = 0.04


def make_controller():
    from main.lang.rollout import LangRolloutController
    from main.train.rollout import SRC_ACTOR, SRC_PLANNER

    class Scripted(LangRolloutController):
        script = None    # {"g": [...], "engage": fraction}; state below
        rec = None

        def progress(self):
            task = self.env.task
            d = np.asarray(task.target_xy, np.float64) - np.asarray(task.start_xy, np.float64)
            L = float(np.linalg.norm(d))
            p = np.asarray(task.tray.get_pose().p, np.float64)[:2]
            return float(np.dot(p - np.asarray(task.start_xy, np.float64), d / max(L, 1e-9))), L

        def log(self, source):
            s, L = self.progress()
            self.rec["t"].append(self._n_physics / self.env.PHYSICS_FREQ)
            self.rec["s"].append(s)
            self.rec["src"].append(int(source))
            self.rec["L"] = L

        def _tick_hook(self, ctx) -> bool:
            st = self.script
            self._tick_source = SRC_PLANNER
            self._tick_action = self._anom(ctx).cpu().numpy()[0].astype(np.float64)
            q = self.extractor.raw_qpos().astype(np.float64)
            a_last = None if st.get("q") is None else q - st["q"]
            st["q"] = q
            plan = ctx.get(f"{self.env.task.carry_arm}_arm")
            s, L = self.progress()
            if st["block"] is None and not st["done"] and plan is not None and a_last is not None \
                    and s >= st["engage"] * L:
                nj = self.extractor.spec.n_joints
                a0 = a_last.copy()
                for arm_i in range(2):
                    a0[arm_i * nj + nj - 2: (arm_i + 1) * nj] = 0.0      # never the grippers
                st.update(block=0, a0=a0, s_engage=s, t_engage=self._n_physics / self.env.PHYSICS_FREQ)
            if st["block"] is None or st["block"] >= len(st["g"]):
                if st["block"] is not None and not st["done"]:
                    st.update(done=True, block=None, s_release=s, t_release=self._n_physics / self.env.PHYSICS_FREQ)
                self.log(SRC_PLANNER)
                return False
            a = st["g"][st["block"]] * st["a0"]
            st["block"] += 1
            for _ in range(3):
                self._tick_source, self._tick_action = SRC_ACTOR, a
                self.log(SRC_ACTOR)
                self.env.step_dtheta(a)
            st["q"] = None     # the next hook is 3 ticks later: no one-tick displacement
            return True

    return Scripted


def run_one(args, seed, behaviour, script, ctl_name, cfg, encoder, kin) -> dict:
    import torch

    from main.item_envs.controller import RiskTickController
    from main.lang.controller import ResumeViaController
    from main.lang.rollout import wire_item_eval

    cls = {"replay": RiskTickController, "resume": ResumeViaController}[ctl_name]
    cfg_ep = dataclasses.replace(cfg, task="tray_slide", embodiment="ur5-wsg", item_host=None, item_active_rule=None,
                                 item_risk={"behaviour": behaviour}, perturb_prob=0.0,
                                 item_out_root=str(Path(args.out) / "_runner"))
    Ctl = make_controller()
    ro = Ctl(cfg_ep, seed, mode="filtered", encoder=encoder, kin=kin, policy_enc=None, actor=None, critics=None,
             filter_active=False, episode_id=seed, rng=np.random.RandomState(seed),
             max_steps=10 * int(cfg.max_steps_per_episode), controller_cls=cls, text_store=None)
    ro.script = {"g": list(SCRIPTS[script]), "engage": float(args.engage), "block": None, "done": False, "q": None}
    ro.rec = {"t": [], "s": [], "src": []}
    runner = ro.env._item_runner
    rule_ids = [r.rule_id for r in runner.rules]
    wire_item_eval(ro, "upright|glass|-")
    t0 = time.perf_counter()
    with torch.no_grad():
        trace = ro.run()
    st, rec = ro.script, ro.rec
    t, s = np.asarray(rec["t"]), np.asarray(rec["s"])
    th, h_all = np.asarray(trace["t"]), np.asarray(trace["h_all"], np.float64)
    out = {"seed": seed, "behaviour": behaviour, "script": script, "controller": ctl_name, "status": trace.get("status"),
           "success": bool(trace.get("success", False)), "wall_s": time.perf_counter() - t0, "L": rec.get("L"),
           "engaged": st.get("t_engage") is not None, "t_engage": st.get("t_engage"), "s_engage": st.get("s_engage"),
           "t_release": st.get("t_release"), "s_release": st.get("s_release"),
           "resumes": (getattr(ro.env.task, "episode_info", None) or {}).get("via_resumes"),
           "t": t.tolist(), "s": s.tolist(), "src": rec["src"], "th": th.tolist(), "rule_ids": rule_ids,
           "h": h_all.tolist()}
    if st.get("t_release") is not None:
        after = t >= st["t_release"]
        v = np.diff(s) / np.maximum(np.diff(t), 1e-9)
        out["back_m"] = float(st["s_release"] - s[after].min()) if after.any() else 0.0
        out["peak_speed_after"] = float(v[after[1:]].max()) if after[1:].any() else 0.0
        out["final_s"] = float(s[-1])
        for k, rid in enumerate(rule_ids):
            neg = np.nonzero(h_all[:, k] < 0)[0] if h_all.size else np.zeros(0, int)
            out[f"fell|{rid}"] = None if not neg.size else ("in_script" if th[neg[0]] < st["t_release"] else "after_release")
            out[f"t_fell|{rid}"] = None if not neg.size else float(th[neg[0]])
    try:
        ro.env.close()
    except Exception:
        pass
    return out


def run(args) -> int:
    import torch

    import main.lang  # noqa: F401
    from main.lang.config import LangConfig
    from main.lang.legacy_pool import install_spawn_pool, install_vocab
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR, make_kinematics

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = LangConfig.load(CFG)
    install_vocab(getattr(cfg, "vocab", "current"))
    install_spawn_pool(getattr(cfg, "spawn_pool", "current"))
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, "ur5-wsg")
    for seed in args.seeds:
        for behaviour in args.behaviours:
            for script in args.scripts:
                for ctl in ("replay", "resume"):
                    name = f"s{seed}_{behaviour}_{script}_{ctl}"
                    if (out / f"{name}.json").exists():
                        continue
                    try:
                        res = run_one(args, int(seed), behaviour, script, ctl, cfg, encoder, kin)
                    except Exception:
                        traceback.print_exc()
                        continue
                    (out / f"{name}.json").write_text(json.dumps(res))
                    keys = ("status", "success", "s_engage", "s_release", "back_m", "peak_speed_after", "final_s")
                    print(f"[release] {name}: " + " ".join(f"{k} {res.get(k)}" for k in keys)
                          + " " + " ".join(f"{k} {v}" for k, v in res.items() if k.startswith("fell|"))
                          + f" resumes {res.get('resumes')}", flush=True)
    return table(args)


def table(args) -> int:
    rows = [json.loads(p.read_text()) for p in sorted(Path(args.out).glob("s*.json"))]
    print(f"{len(rows)} runs")
    print("behaviour script controller | n | released at (m) | back (m, max) | peak speed after (m/s, mean) | "
          "glass fell in script / after | can fell in script / after | reached the pad")
    keys = sorted({(r["behaviour"], r["script"], r["controller"]) for r in rows})
    for b, sc, c in keys:
        rr = [r for r in rows if (r["behaviour"], r["script"], r["controller"]) == (b, sc, c) and r.get("t_release")]
        if not rr:
            continue
        g = [r.get("fell|upright|glass|-") for r in rr]
        cn = [r.get("fell|upright|can|-") for r in rr]
        print(f"{b:8s} {sc:15s} {c:7s} | {len(rr)} | {np.mean([r['s_release'] for r in rr]):.3f} | "
              f"{np.max([r['back_m'] for r in rr]):.3f} | {np.mean([r['peak_speed_after'] for r in rr]):.2f} | "
              f"{g.count('in_script')} / {g.count('after_release')} | {cn.count('in_script')} / {cn.count('after_release')} | "
              f"{sum(bool(r['success']) for r in rr)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "table"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT))
        p.add_argument("--seeds", type=int, nargs="+", default=[8301, 8302, 8303])
        p.add_argument("--behaviours", nargs="+", default=["medium", "slow"])
        p.add_argument("--scripts", nargs="+", default=list(SCRIPTS), choices=list(SCRIPTS))
        p.add_argument("--engage", type=float, default=0.4, help="the script takes over this far along the drag")
    args = ap.parse_args(argv)
    args.out = str(Path(args.out).resolve())
    return run(args) if args.cmd == "run" else table(args)


if __name__ == "__main__":
    sys.exit(main())
