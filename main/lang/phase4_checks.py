"""Phase 4 checks that need the simulator and a GPU (VANILLA_PLAN Section 10).

    python -m main.lang.phase4_checks filter --task <t> --embodiment <e> --out <dir>
        one filter episode with a random-init language model through run_episodes:
        buffer, sidecar and lang_episodes.jsonl in lockstep, and a video
    python -m main.lang.phase4_checks frame0 --task <t> --embodiment <e> --out <json>
        deployment frame 0 == the s0 row collection writes for the same seed;
        exactly one capture before the first physics step in both modes
    python -m main.lang.phase4_checks status --task <t> --embodiment <e> --out <json>
        trace["status"] for a forced timeout and a forced exception
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from . import ensure_item_path

ensure_item_path()

from .config import CANON_EMBODIMENTS, CANON_T1_HOSTS, CANON_TASKS, LangConfig  # noqa: E402

SEED_BASE = 960_000_000   # scratch stream


def _models(cfg, caches):
    import torch

    from .network import LangPolicyEncoder, LangTwinCritic, make_actor
    from .text import TextStore

    torch.manual_seed(0)
    pe = LangPolicyEncoder(cfg).cuda().eval()
    actor = make_actor(cfg).cuda().eval()
    critics = LangTwinCritic(cfg).cuda().eval()
    text = TextStore(caches, cfg.text_backend, "cuda")
    return pe, actor, critics, text


def cmd_filter(args) -> int:
    from .collect import make_pair, run_episodes
    from .recon import parity

    cfg = LangConfig()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pe, actor, critics, text = _models(cfg, [Path(c) for c in args.text_cache])
    buf, side = make_pair(out, cfg, 4 * int(cfg.max_steps_per_episode * 1.1) + 16)
    host = CANON_T1_HOSTS[1] if args.task == "selective_avoid" else None
    seed0 = SEED_BASE + CANON_TASKS.index(args.task) * 1000 + CANON_EMBODIMENTS.index(args.embodiment) * 100
    # a scene can have no sampleable rule (e.g. T3 with a held-out box): then the loop runs it nominal,
    # correctly; try the next seeds until a filter episode ran
    attempt = 0
    for _ in range(4):
        summ = run_episodes(cfg, out, [(args.task, args.embodiment, host)], seed0, buf, side,
                            models=(pe, actor, critics), filter_frac=1.0, video_every=1, text_store=text,
                            rng_policy=np.random.default_rng(0), attempt0=attempt)
        attempt = int(summ["attempt"])
        line = json.loads((out / "lang_episodes.jsonl").read_text().splitlines()[-1])
        if line.get("filter"):
            break
    p = parity(out)
    vids = sorted((out / "videos").glob("*.mp4")) if (out / "videos").exists() else []
    res = {"summary": summ, "parity_ok": p["ok"], "len_equal": p["len_equal"], "h_parity": p["h_parity_max_abs"],
           "filter": line.get("filter"), "named": line.get("named_rule_id"), "designation": line.get("designation"),
           "sentence": line.get("sentence"), "status": line.get("status"), "n_ticks": line.get("n_ticks"),
           "intervention_rate": line.get("intervention_rate"), "videos": [str(v) for v in vids],
           "rows_match_line": line["rows"][1] == p["n_buffer"]}
    res["ok"] = bool(res["parity_ok"] and res["rows_match_line"] and vids and line.get("filter")
                     and line.get("status") in ("ok", "timeout"))
    (out / "check.json").write_text(json.dumps(res, indent=1, default=str))
    print(res)
    return 0 if res["ok"] else 1


def cmd_frame0(args) -> int:
    import torch

    from main.item_envs.controller import RiskTickController
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR, make_kinematics

    from .collect import make_pair
    from .rollout import LangRolloutController, wire_item_eval

    cfg = LangConfig()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, args.embodiment)
    pe, actor, critics, text = _models(cfg, [Path(c) for c in args.text_cache])
    host = CANON_T1_HOSTS[0] if args.task == "selective_avoid" else None
    seed = SEED_BASE + 500 + CANON_TASKS.index(args.task) * 10 + CANON_EMBODIMENTS.index(args.embodiment)
    cfg_ep = dataclasses.replace(cfg, task=args.task, embodiment=args.embodiment, item_host=host,
                                 item_out_root=str(out.parent / "_runner"))
    res = {}
    # collection: s0 row
    buf, side = make_pair(out.parent / f"frame0_{args.task}_{args.embodiment}", cfg, 100)
    ro = LangRolloutController(cfg_ep, seed, mode="collect", encoder=enc, kin=kin, filter_active=False, buf=buf,
                               episode_id=seed, rng=np.random.RandomState(seed), max_steps=60,
                               controller_cls=RiskTickController)
    from main.item_envs.margin_hook import wire_item_episode

    wire_item_episode(ro, side)
    n_before_phys = {"captures": 0}
    orig_cap = ro._capture_tick

    def cap_count():
        if ro._n_physics == 0:
            n_before_phys["captures"] += 1
        return orig_cap()

    ro._capture_tick = cap_count
    ro.run()
    ro.buf.end_episode()
    buf.flush(), side.flush()
    s0 = np.asarray(np.load(buf.root / "scene_tokens.npy", mmap_mode="r")[0], dtype=np.float32)
    res["collect_captures_before_physics"] = n_before_phys["captures"]
    # deployment on the same seed
    ro2 = LangRolloutController(cfg_ep, seed, mode="filtered", encoder=enc, kin=kin, policy_enc=pe, actor=actor,
                                critics=critics, filter_active=True, episode_id=seed, rng=np.random.RandomState(seed),
                                max_steps=60, controller_cls=RiskTickController, text_store=text)
    wire_item_eval(ro2, None)
    runner = ro2.env._item_runner
    from main.item_envs import language

    rec = language.episode_record(runner, seed)
    rule = next(r for r in rec["rules"] if r.get("sampleable"))
    ro2.set_instruction(language.sample_sentence(rule, rec, np.random.default_rng(0), "train"), rule_id=rule["rule_id"])
    n2 = {"captures": 0}
    orig2 = ro2._capture_tick

    def cap2():
        if ro2._n_physics == 0:
            n2["captures"] += 1
        return orig2()

    ro2._capture_tick = cap2
    with torch.no_grad():
        ro2.run()
    f0 = ro2.policy_enc.frame0[0][0].float().cpu().numpy()
    res["deploy_captures_before_physics"] = n2["captures"]
    res["frame0_max_abs_diff_fp16"] = float(np.abs(f0.astype(np.float16).astype(np.float32) - s0).max())
    res["frame0_scale"] = float(np.abs(s0).max())
    res["ok"] = bool(res["collect_captures_before_physics"] == 1 and res["deploy_captures_before_physics"] == 1
                     and res["frame0_max_abs_diff_fp16"] < 1e-2 * max(1.0, res["frame0_scale"]))
    out.write_text(json.dumps(res, indent=1))
    print(res)
    return 0 if res["ok"] else 1


def cmd_status(args) -> int:
    from main.item_envs.controller import RiskTickController
    from main.network.encoder import HoloBrainEncoder
    from main.train.collect import CKPT_DIR, make_kinematics

    from .rollout import LangRolloutController, wire_item_eval

    cfg = LangConfig()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, args.embodiment)
    host = CANON_T1_HOSTS[0] if args.task == "selective_avoid" else None
    cfg_ep = dataclasses.replace(cfg, task=args.task, embodiment=args.embodiment, item_host=host,
                                 item_out_root=str(out.parent / "_runner"))
    res = {}
    seed = SEED_BASE + 900 + CANON_TASKS.index(args.task)
    ro = LangRolloutController(cfg_ep, seed, mode="filtered", encoder=enc, kin=kin, filter_active=False,
                               episode_id=seed, rng=np.random.RandomState(seed), max_steps=40,
                               controller_cls=RiskTickController)
    wire_item_eval(ro, None)
    res["timeout_status"] = ro.run()["status"]
    ro = LangRolloutController(cfg_ep, seed, mode="filtered", encoder=enc, kin=kin, filter_active=False,
                               episode_id=seed, rng=np.random.RandomState(seed), max_steps=4000,
                               controller_cls=RiskTickController)
    wire_item_eval(ro, None)
    orig_step = ro.env.task.scene.step
    count = {"n": 0}

    def boom():
        count["n"] += 1
        if count["n"] == 25:
            raise RuntimeError("forced failure for the status test")
        return orig_step()

    ro.env.task.scene.step = boom
    tr = ro.run()
    res["exception_status"], res["exception_text"] = tr["status"], tr.get("exception")
    res["ok"] = res["timeout_status"] == "timeout" and res["exception_status"] == "exception"
    out.write_text(json.dumps(res, indent=1))
    print(res)
    return 0 if res["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("filter", "frame0", "status"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True, choices=CANON_TASKS)
        p.add_argument("--embodiment", required=True, choices=CANON_EMBODIMENTS)
        p.add_argument("--out", required=True)
        p.add_argument("--text-cache", nargs="*", default=[])
    args = ap.parse_args(argv)
    t0 = time.time()
    rc = {"filter": cmd_filter, "frame0": cmd_frame0, "status": cmd_status}[args.cmd](args)
    print(f"[phase4] {args.cmd} {args.task}/{args.embodiment} rc {rc} in {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
