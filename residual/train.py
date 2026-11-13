"""PPO learner for the residual policy (residual/PLAN.md).

    scripts/lang/in_container.sh python -m residual.train --run /abs/runs/residual/<name> \
        [--updates 2000] [--set replan_trigger='"deviation"' completion='"success"'] [--deadline-min 50]

The learner owns the model, the optimizer and the observation statistics. Workers (``worker.py``, several per GPU)
run episodes with the newest weights and send them back; each PPO update uses ``steps_per_update`` transitions from
episodes at most ``max_policy_lag`` versions old.

Evaluation (every ``eval_every`` updates, on fixed scenes that training never uses): ``eval_in_way`` scenes whose
named object is in the planner's way and ``eval_out_way`` scenes where it is not, deterministic policy, w = eval_w,
no noise, the episode runs to its end (so task success is measured; the first break of the named rule is recorded).
The delta = 0 baseline (same controller and trigger) runs once on the same scenes. Videos of the first
``eval_video_scenes`` scenes at every evaluation. ``tray_slide`` (``--set task=tray_slide``): ``eval_pairs`` scenes,
each run with the glass and with the can named; the numbers are split by the named item (PLAN section 8).

Resume: rerunning the same command continues from ``<run>/checkpoint.pt`` (same wandb run). SIGUSR1 / SIGTERM or
``--deadline-min`` stop at the next safe point with a checkpoint and exit 75 (``scripts/lang/long.sbatch``
resubmits on 75).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

EXIT_RESUME = 75


def _safe_wandb_log(run, logs: dict, step: int) -> None:
    if run is None:
        return
    try:
        run.log(logs, step=step)
    except Exception as exc:
        print(f"[residual-train] wandb log failed at update {step} ({exc}); continuing", flush=True)


def _atomic_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(tmp, path)


def _atomic_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _rate(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def _med(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- evaluation scenes ----
_REASON_ORDER = {"on_path": 0, "near_path": 1, "second_corridor": 2}


def select_eval_scenes(scans: list[dict], n_in: int, n_out: int) -> list[dict]:
    """Fixed evaluation scenes from scanned seeds: hosts balanced, the rule kind alternating avoid / stay."""
    scans = sorted([s for s in scans if not s.get("error") and s.get("n_candidates", 0) > 0], key=lambda s: s["seed"])
    hosts = sorted({s["host"] for s in scans})
    out, used = [], set()

    def take(n, in_way):
        cap = math.ceil(n / max(1, len(hosts)))
        per_host = {h: 0 for h in hosts}
        chosen = []
        for relax in (False, True):
            for s in scans:
                if len(chosen) >= n:
                    break
                if s["seed"] in used or (not relax and per_host[s["host"]] >= cap):
                    continue
                items = [(i, r) for i, r in s["roles"].items() if (r != "out_of_way") == in_way]
                if not items:
                    continue
                items.sort(key=lambda ir: (_REASON_ORDER.get(ir[1], 9), ir[0]))
                item, reason = items[0]
                kind = ("avoid", "stay")[len(chosen) % 2]
                chosen.append({"seed": s["seed"], "host": s["host"], "item_id": item, "reason": reason,
                               "in_way": in_way, "rule_kind": kind,
                               "category": s.get("categories", {}).get(item, "")})
                used.add(s["seed"])
                per_host[s["host"]] += 1
        return chosen

    out += take(n_in, True)
    out += take(n_out, False)
    return out


# ---------------------------------------------------------------------------------- learner ----
class Learner:
    def __init__(self, args):
        import torch

        from .config import N_ACT, ResidualConfig, apply_sets
        from .policy import build_model
        from .tasks import task_hooks

        self.args = args
        self.run = Path(args.run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.run / "checkpoint.pt"
        cfg_path = self.run / "config.json"
        if cfg_path.exists():
            cfg = ResidualConfig.from_dict(json.loads(cfg_path.read_text()))
            if args.set:
                print(f"[residual-train] resume: --set ignored, the run keeps {cfg_path}", flush=True)
        else:
            cfg = apply_sets(ResidualConfig(), args.set)
            _atomic_json(cfg.to_dict(), cfg_path)
        if args.workers_per_gpu is not None:
            cfg.workers_per_gpu = int(args.workers_per_gpu)
        self.cfg = cfg
        self.hooks = task_hooks(cfg.task)
        torch.set_num_threads(max(1, min(8, int(args.learner_threads))))
        self.obs_dim = self.hooks.obs_dim(cfg)
        self.model = build_model(cfg, self.obs_dim, N_ACT)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.rng = np.random.default_rng(int(args.seed))
        self.update = 0
        self.version = 0
        self.next_block = 0
        self.wandb_id = None
        self.eval_round = 0
        self.baseline: dict | None = None
        self.norm_ready = False
        self.totals = {"episodes": 0, "transitions": 0, "dropped_stale": 0, "skipped": 0}
        self._stop = False
        self.t_start = time.time()
        if self.ckpt_path.exists():
            self._load()
        self.eval_scenes = None
        p = self.run / "eval_scenes.json"
        if p.exists():
            self.eval_scenes = json.loads(p.read_text())

    # ------------------------------------------------------------------ state ----
    def _load(self) -> None:
        import torch

        blob = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(blob["model"])
        self.opt.load_state_dict(blob["opt"])
        self.update = int(blob["update"])
        self.version = int(blob["version"])
        self.next_block = int(blob.get("next_block", 0)) + 200   # blocks spawned after the last checkpoint
        self.wandb_id = blob.get("wandb_id")
        self.eval_round = int(blob.get("eval_round", 0))
        self.baseline = blob.get("baseline")
        self.norm_ready = bool(blob.get("norm_ready", self.update > 0))
        self.totals = blob.get("totals", self.totals)
        if blob.get("rng") is not None:
            self.rng.bit_generator.state = blob["rng"]
        print(f"[residual-train] resumed at update {self.update} (version {self.version})", flush=True)

    def save(self, pool=None) -> None:
        from .policy import state_for_workers

        blob = {
            "model": state_for_workers(self.model), "opt": self.opt.state_dict(), "update": self.update,
            "version": self.version, "next_block": pool.next_block if pool is not None else self.next_block,
            "wandb_id": self.wandb_id, "eval_round": self.eval_round, "baseline": self.baseline,
            "norm_ready": self.norm_ready, "totals": self.totals, "rng": self.rng.bit_generator.state, "cfg": self.cfg.to_dict(),
            "obs_dim": self.obs_dim,
        }
        _atomic_save(blob, self.ckpt_path)

    def publish(self, path: Path | None = None) -> Path:
        from .policy import state_for_workers

        path = path or (self.run / "weights.pt")
        _atomic_save({"state": state_for_workers(self.model), "version": self.version, "cfg": self.cfg.to_dict(),
                      "obs_dim": self.obs_dim}, path)
        return path

    # ------------------------------------------------------------------ wandb ----
    def _log(self, wb, logs: dict) -> None:
        """Every logged number also goes to <run>/metrics.jsonl (the 11-12 run had no wandb for its first piece)."""
        row = {"update": self.update, "time": time.time()}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        try:
            with open(self.run / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as exc:
            print(f"[residual-train] metrics.jsonl write failed ({exc})", flush=True)
        _safe_wandb_log(wb, logs, self.update)

    def init_wandb(self):
        if self.args.no_wandb:
            return None
        try:
            import wandb

            if self.wandb_id is None:
                gen = getattr(getattr(wandb, "util", None), "generate_id", None)
                if gen is None:   # newer wandb moved it
                    import uuid

                    gen = lambda: uuid.uuid4().hex[:8]
                self.wandb_id = gen()
            run = wandb.init(project=self.cfg.wandb_project, name=self.run.name, id=self.wandb_id, resume="allow",
                             dir=str(self.run), config=self.cfg.to_dict())
            return run
        except Exception as exc:
            print(f"[residual-train] wandb init failed ({exc}); logging to files only", flush=True)
            return None

    # ------------------------------------------------------------------- eval ----
    def _scan_eval_scenes(self, pool) -> None:
        cfg = self.cfg
        need = self.hooks.eval_need(cfg)
        scans: list[dict] = []
        next_seed = cfg.seed_base_eval
        batch = max(need * 2, 2 * pool.n)
        t_scan = time.time()
        while True:
            ids = set()
            for s in range(next_seed, next_seed + batch):
                ids.add(pool.submit({"kind": "scan", "seed": s, "tag": "scan", "job_id": -1}))
            next_seed += batch
            while any(i in pool.jobs for i in ids) and not self._stop:   # a job leaves pool.jobs when done / dropped
                for res in pool.poll(timeout=5.0):
                    if res.get("kind") == "scan":
                        scans.append(res)
                if time.time() - t_scan > 60.0 * self.args.eval_timeout_min:
                    print("[residual-train] eval scan: time limit, using the scans so far", flush=True)
                    for i in ids:
                        pool.jobs.pop(i, None)
                    break
            chosen = self.hooks.select_eval_scenes(scans, cfg)
            enough, what = self.hooks.eval_enough(chosen, cfg)
            print(f"[residual-train] eval scan: {len(scans)} seeds -> {what}", flush=True)
            if self._stop:
                return   # a partial list must not become the run's fixed evaluation scenes
            if enough or next_seed - cfg.seed_base_eval > 2000 \
                    or time.time() - t_scan > 60.0 * self.args.eval_timeout_min:
                break
        self.eval_scenes = chosen
        _atomic_json(chosen, self.run / "eval_scenes.json")
        _atomic_json(scans, self.run / "eval_scans.json")

    def _eval_jobs(self, round_name: str, policy: str, weights: str | None, videos: bool) -> list[dict]:
        from .episode import Job

        cfg = self.cfg
        vid_dir = self.run / "eval" / round_name / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        n_vid_side = cfg.eval_video_scenes // 2
        jobs, seen = [], {}
        for i, sc in enumerate(self.eval_scenes or []):
            side = bool(sc["in_way"])
            group = self.hooks.scene_group(sc)   # T1: in_way / out_of_way; tray: glass / can
            vid = None
            k = seen.get(group, 0)
            if videos and k < n_vid_side:
                vid = str(vid_dir / (f"{self.hooks.video_prefix[group]}{k}_{sc['host']}_s{sc['seed']}_"
                                     f"{sc['rule_kind']}_{sc['category']}.mp4"))
            seen[group] = k + 1
            jobs.append(Job(kind="eval", seed=int(sc["seed"]), in_way=side, rule_kind=sc["rule_kind"],
                            item_id=sc["item_id"], w=cfg.eval_w, policy=policy, deterministic=True, noise=False,
                            terminate=False, video=vid, weights=weights, tag=round_name,
                            extra={"scene": i, **self.hooks.scene_job_extra(sc)}).to_dict())
        return jobs

    def eval_summary(self, eps: list[dict]) -> dict:
        eps = [e for e in eps if not e.get("skipped")]
        out = {"n": len(eps)}
        for name, pred in self.hooks.groups:   # T1: in_way / out_of_way; tray: glass / can
            sub = [e for e in eps if pred(e)]
            out[f"{name}/n"] = len(sub)
            out[f"{name}/broken"] = _rate(e["violated"] for e in sub)
            out[f"{name}/success"] = _rate(e["success"] for e in sub)
            out[f"{name}/safe_success"] = _rate(e["success"] and not e["violated"] for e in sub)
            out[f"{name}/correction_rms"] = _rate(e["corr_rms"] for e in sub)
            out[f"{name}/replans"] = _rate(e["replans"] for e in sub)
            out.update({f"{name}/{k}": v for k, v in self.hooks.group_extra(sub, eval_round=True).items()})
            for k in self.hooks.rule_kinds:
                out[f"{name}/broken_{k}"] = _rate(e["violated"] for e in sub if e["rule_kind"] == k)
        out["success"] = _rate(e["success"] for e in eps)
        out["truncated"] = _rate(e["end"] == "truncated" for e in eps)
        closes = [g for e in eps for g in e.get("grip_events", []) if g["kind"] == "close"]
        out["grasp_ee_off_median"] = _med(g["ee_off"] for g in closes)
        out["grasp_joint_off_median"] = _med(g["joint_off"] for g in closes)
        return out

    # ------------------------------------------------------------------- train ----
    def train_stats(self, eps: list[dict]) -> dict:
        cfg = self.cfg
        out = {"episodes": len(eps), "transitions": int(sum(len(e["traj"]["rew"]) for e in eps))}
        for name, pred in self.hooks.groups:
            sub = [e for e in eps if pred(e)]
            out[f"{name}/share"] = len(sub) / max(1, len(eps))
            out[f"{name}/broken"] = _rate(e["end"] == "violation" for e in sub)
            out[f"{name}/success"] = _rate(e["success"] for e in sub)
            out[f"{name}/correction_rms"] = _rate(e["corr_rms"] for e in sub)
            out.update({f"{name}/{k}": v for k, v in self.hooks.group_extra(sub, eval_round=False).items()})
        for end in ("violation", "finished", "truncated"):
            out[f"end/{end}"] = _rate(e["end"] == end for e in eps)
        for r in ("replan_fail", "host_plan_fail", "timeout", "error"):
            out[f"truncated/{r}"] = _rate(e["end_reason"] == r for e in eps)
        out["success"] = _rate(e["success"] for e in eps)
        out["finished_success"] = _rate(e["success"] for e in eps if e["end"] == "finished")
        out["reward_per_tick"] = _rate(e["rew_mean"] for e in eps)
        out["policy_ticks"] = _rate(e["policy_ticks"] for e in eps)
        out["replans_per_episode"] = _rate(e["replans"] for e in eps)
        out["named_held"] = _rate(e["named_held"] for e in eps)
        out["clip_frac"] = _rate(e["clip_frac"] for e in eps)
        out["builds_per_episode"] = _rate(e.get("builds", 1) for e in eps)
        closes = [g for e in eps for g in e.get("grip_events", []) if g["kind"] == "close"]
        opens = [g for e in eps for g in e.get("grip_events", []) if g["kind"] == "open"]
        out["grasp_ee_off_median"] = _med(g["ee_off"] for g in closes)
        out["grasp_joint_off_median"] = _med(g["joint_off"] for g in closes)
        out["release_ee_off_median"] = _med(g["ee_off"] for g in opens)
        out["std_rad"] = float(self.model.std().mean()) * float(cfg.delta_bound)
        out["std_units"] = float(self.model.std().mean())
        ticks = sum(e["sim_ticks"] for e in eps)
        busy = sum(e.get("wall_job", e["wall_episode"]) for e in eps)
        out["ticks_per_s_per_worker"] = ticks / max(busy, 1e-9)
        return out

    def handle_signal(self, sig, frame) -> None:
        print(f"[residual-train] signal {sig}: stopping at the next safe point", flush=True)
        self._stop = True

    def main(self) -> int:
        from .pool import WorkerPool, visible_gpus
        from .ppo import make_batch, ppo_update

        args, cfg = self.args, self.cfg
        signal.signal(signal.SIGUSR1, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        deadline = self.t_start + 60.0 * float(args.deadline_min) if args.deadline_min else None
        wb = self.init_wandb()
        self.publish()
        gpus = visible_gpus()
        n_workers = int(args.workers) if args.workers else cfg.workers_per_gpu * max(1, len(gpus))
        scratch = Path(args.scratch or os.environ.get("TMPDIR", "/tmp")) / f"residual_{self.run.name}_{os.getpid()}"
        pool = WorkerPool(cfg, self.run, n_workers, self.run / "logs", scratch, gpus=gpus, next_block=self.next_block)
        pool.start()
        print(f"[residual-train] {n_workers} workers on GPUs {gpus}; obs dim {self.obs_dim}; "
              f"update {self.update}/{args.updates}", flush=True)
        ep_log = open(self.run / "train_episodes.jsonl", "a")
        exit_code = 0
        try:
            if self.eval_scenes is None:
                self._scan_eval_scenes(pool)
            eval_state = None   # {"name", "ids", "eps", "t0", "policy"}
            pending_rounds = []
            if self.baseline is None:
                pending_rounds.append(("baseline", "zero", None))
            pool.set_training(True)
            buf, n_buf = [], 0
            t_upd = time.time()
            while self.update < args.updates and not self._stop:
                if deadline is not None and time.time() > deadline:
                    print("[residual-train] deadline reached", flush=True)
                    self._stop = True
                    break
                if eval_state is None and pending_rounds:
                    name, policy, weights = pending_rounds.pop(0)
                    jobs = self._eval_jobs(name, policy, weights, videos=True)
                    ids = {pool.submit(j) for j in jobs}
                    eval_state = {"name": name, "ids": ids, "eps": [], "t0": time.time(), "policy": policy,
                                  "update": self.update}
                    print(f"[residual-train] eval round {name}: {len(ids)} episodes", flush=True)
                for res in pool.poll(timeout=2.0):
                    kind = res.get("kind")
                    if kind == "eval" and eval_state is not None and res.get("tag") == eval_state["name"]:
                        res.pop("traj", None)
                        eval_state["eps"].append(res)
                        eval_state["ids"].discard(res.get("job_id"))
                        continue
                    if kind != "train":
                        continue
                    if res.get("skipped"):
                        self.totals["skipped"] += 1
                        continue
                    lag = self.version - int(res.get("version", -10))
                    light = {k: v for k, v in res.items() if k not in ("traj", "replan_log")}
                    light["update"] = self.update
                    light["lag"] = lag
                    ep_log.write(json.dumps(light) + "\n")
                    if lag > cfg.max_policy_lag or lag < 0:
                        self.totals["dropped_stale"] += 1
                        continue
                    buf.append(res)
                    n_buf += len(res["traj"]["rew"])
                ep_log.flush()
                if eval_state is not None:
                    eval_state["ids"] = {i for i in eval_state["ids"] if i in pool.jobs}   # done or dropped
                    if eval_state["ids"] and time.time() - eval_state["t0"] > 60.0 * args.eval_timeout_min:
                        print(f"[residual-train] eval {eval_state['name']}: time limit, {len(eval_state['ids'])} "
                              f"episodes missing", flush=True)
                        for i in eval_state["ids"]:
                            pool.jobs.pop(i, None)
                        eval_state["ids"] = set()
                if eval_state is not None and not eval_state["ids"]:
                    self._finish_eval(eval_state, wb)
                    eval_state = None
                if n_buf >= cfg.steps_per_update and not self.norm_ready:
                    # the first batch only fits the observation statistics: the actor and critic of update 1 then
                    # train on the inputs they will see. Its episodes ran without normalization, so the version jumps
                    # by max_policy_lag + 1 and they (and those still running) are not used for PPO
                    self.model.norm.update(np.concatenate([e["traj"]["obs"] for e in buf if len(e["traj"]["rew"])]))
                    self.norm_ready = True
                    self.version += int(cfg.max_policy_lag) + 1
                    self.publish()
                    print(f"[residual-train] observation statistics fitted on {n_buf} transitions; "
                          f"version {self.version}", flush=True)
                    buf, n_buf = [], 0
                    self.save(pool)
                    continue
                if n_buf >= cfg.steps_per_update:
                    batch = make_batch([e["traj"] | {"end": e["end"], "boot": e["boot"],
                                                     "last_value": e["last_value"]} for e in buf], cfg.gae_lambda)
                    stats = ppo_update(self.model, self.opt, batch, cfg, self.rng)
                    self.model.norm.update(batch["obs"])
                    tstats = self.train_stats(buf)
                    self.update += 1
                    self.version += 1
                    self.publish()
                    self.totals["episodes"] += len(buf)
                    self.totals["transitions"] += n_buf
                    dt = time.time() - t_upd
                    t_upd = time.time()
                    logs = {f"train/{k}": v for k, v in tstats.items()}
                    logs.update({f"ppo/{k}": v for k, v in stats.items()})
                    logs.update({"perf/update_s": dt, "perf/transitions_per_s": n_buf / max(dt, 1e-9),
                                 "perf/workers": n_workers, "perf/restarts_planned": pool.restarts["planned"],
                                 "perf/restarts_crash": pool.restarts["crash"],
                                 "perf/dropped_stale": self.totals["dropped_stale"],
                                 "perf/total_transitions": self.totals["transitions"]})
                    self._log(wb, logs)
                    print(f"[residual-train] update {self.update}: {len(buf)} eps {n_buf} steps in {dt:.0f}s | "
                          f"{self.hooks.update_line(tstats)} "
                          f"std {tstats['std_rad']:.4f} rad | kl {stats['kl']:.4f} ev {stats['explained_var']:.2f}",
                          flush=True)
                    buf, n_buf = [], 0
                    self.save(pool)
                    if cfg.eval_every and self.update % int(cfg.eval_every) == 0:
                        snap = self.publish(self.run / "eval" / f"weights_v{self.version:05d}.pt")
                        pending_rounds.append((f"u{self.update:05d}", "model", str(snap)))
            if not self._stop and (eval_state is not None or pending_rounds):
                # the last updates are done: finish the evaluation rounds still open (the 11-11 check stopped with
                # its baseline round half run), training episodes off, within the eval time limit and the deadline
                pool.set_training(False)
                while (eval_state is not None or pending_rounds) and not self._stop:
                    if deadline is not None and time.time() > deadline:
                        print("[residual-train] deadline reached during the last evaluation", flush=True)
                        break
                    if eval_state is None:
                        name, policy, weights = pending_rounds.pop(0)
                        ids = {pool.submit(j) for j in self._eval_jobs(name, policy, weights, videos=True)}
                        eval_state = {"name": name, "ids": ids, "eps": [], "t0": time.time(), "policy": policy,
                                      "update": self.update}
                        print(f"[residual-train] eval round {name}: {len(ids)} episodes", flush=True)
                    for res in pool.poll(timeout=2.0):
                        if res.get("kind") == "eval" and res.get("tag") == eval_state["name"]:
                            res.pop("traj", None)
                            eval_state["eps"].append(res)
                            eval_state["ids"].discard(res.get("job_id"))
                    eval_state["ids"] = {i for i in eval_state["ids"] if i in pool.jobs}
                    if eval_state["ids"] and time.time() - eval_state["t0"] > 60.0 * args.eval_timeout_min:
                        print(f"[residual-train] eval {eval_state['name']}: time limit, {len(eval_state['ids'])} "
                              f"episodes missing", flush=True)
                        for i in eval_state["ids"]:
                            pool.jobs.pop(i, None)
                        eval_state["ids"] = set()
                    if not eval_state["ids"]:
                        self._finish_eval(eval_state, wb)
                        eval_state = None
            if self._stop and self.update < args.updates:
                exit_code = EXIT_RESUME
        finally:
            self.save(pool)
            ep_log.close()
            pool.close(timeout=30)
            if wb is not None:
                try:
                    wb.finish()
                except Exception:
                    pass
        print(f"[residual-train] stopped at update {self.update} (exit {exit_code})", flush=True)
        return exit_code

    def _finish_eval(self, st: dict, wb) -> None:
        name = st["name"]
        d = self.run / "eval" / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "episodes.jsonl", "w") as f:
            for e in st["eps"]:
                f.write(json.dumps({k: v for k, v in e.items() if k != "replan_log"}) + "\n")
        summ = self.eval_summary(st["eps"])
        summ["wall_s"] = time.time() - st["t0"]
        _atomic_json(summ, d / "summary.json")
        prefix = "baseline" if name == "baseline" else "eval"
        logs = {f"{prefix}/{k}": v for k, v in summ.items() if isinstance(v, (int, float))}
        if name == "baseline":
            self.baseline = summ
        elif self.baseline is not None:
            for k in self.hooks.baseline_keys:
                logs[f"eval/{k}_baseline"] = self.baseline.get(k)
        vids = sorted([p for p in (d / "videos").glob("*.mp4") if p.stat().st_mtime >= st["t0"] - 1],
                      key=lambda p: p.name)
        if wb is not None and vids:
            try:
                import wandb

                logs[f"{prefix}/videos"] = [wandb.Video(str(p), format="mp4", caption=p.stem) for p in vids]
            except Exception as exc:
                print(f"[residual-train] video log failed ({exc})", flush=True)
        self._log(wb, logs)
        self.eval_round += 1
        print(f"[residual-train] eval {name}: {self.hooks.eval_line(summ)} ({summ['n']} eps, {summ['wall_s']:.0f}s)",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=None, help="total workers (default workers_per_gpu x GPUs)")
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--deadline-min", type=float, default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learner-threads", type=int, default=4)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--eval-timeout-min", type=float, default=90.0, help="an evaluation round or the scene scan "
                    "finishes with what it has after this long")
    args = ap.parse_args()
    return Learner(args).main()


if __name__ == "__main__":
    sys.exit(main())
