"""Worker process: runs episodes with a CPU copy of the policy and sends them to the learner.

The host task's ``play_once`` is a blocking script, so the policy runs inside the worker, called by the controller
every tick. Jobs: the learner's job queue first (evaluation, scans, throughput runs), else a training episode with
the newest weights (``<run>/weights.pt``, reloaded when it changes). A worker exits with 75 after
``max_inits_per_process`` Env builds (pybind11 leaks thread-local keys per build); the learner restarts it.

Messages to the learner (``result_q``): ("start", wid, job_id, kind) when a job starts, ("result", wid, dict) when it
ends, ("dead", wid, why) on an unrecoverable error.
"""

from __future__ import annotations

import os
import queue
import sys
import time
import traceback
from pathlib import Path

EXIT_RESTART = 75
SEEDS_PER_BLOCK = 1000


def _send_result(result_q, spool: Path, wid: int, res: dict) -> None:
    """The result goes through a file; the queue carries only its path. A message under 4 KB is one atomic pipe
    write, so a worker that dies (a simulator crash) can never leave half a message for the learner to block on."""
    import pickle

    spool.mkdir(parents=True, exist_ok=True)
    _send_result.n = getattr(_send_result, "n", 0) + 1
    path = spool / f"w{wid:03d}_{os.getpid()}_{_send_result.n:06d}.pkl"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    result_q.put(("result_file", wid, str(path), int(res.get("job_id", -1))))


def _exit(result_q, log, code: int) -> None:
    """Exit without losing the last message: a result (~1 MB) is larger than a pipe, and os._exit while the queue's
    feeder thread is still writing leaves the learner blocked in get() forever (review 11-09, reproduced)."""
    try:
        result_q.close()
        result_q.join_thread()
    except Exception:
        pass
    try:
        log.flush()
    except Exception:
        pass
    os._exit(code)


class SideBalancer:
    """Keeps the in-the-way share at ``in_way_frac`` with as few rebuilt scenes as possible.

    A scene with both kinds of object serves the side that is behind. A scene with only one kind serves that side
    unless the other side is ``out_way_debt_max`` episodes behind; then it is rebuilt with the next seed.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = {True: 0, False: 0}

    def __call__(self, runner, rng, job):
        from .scene import named_options, pick_named

        cfg = self.cfg
        kind = job.rule_kind or ("stay" if rng.random() < cfg.stay_frac else "avoid")
        if job.item_id is not None:
            from .scene import make_named

            return make_named(runner, job.item_id, kind)
        opts = named_options(runner)
        if job.in_way is not None:
            return pick_named(runner, bool(job.in_way), kind, rng)
        has = {True: bool(opts[True]), False: bool(opts[False])}
        if not has[True] and not has[False]:
            return None
        frac = float(cfg.in_way_frac)
        tot = self.n[True] + self.n[False] + 1
        debt = {True: tot * frac - self.n[True], False: tot * (1 - frac) - self.n[False]}
        if has[True] and has[False]:
            if abs(debt[True] - debt[False]) < 1e-9:
                side = bool(rng.random() < frac)
            else:
                side = debt[True] > debt[False]
        else:
            side = has[True]
            if debt[not side] >= cfg.out_way_debt_max:
                return None
        named = pick_named(runner, side, kind, rng)
        if named is not None:
            self.n[side] += 1
        return named


def _load_weights(path: Path, cache: dict):
    import torch

    from .config import N_ACT, ResidualConfig
    from .policy import build_model

    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if cache.get("key") == key:
        return cache["model"], cache["version"]
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ResidualConfig.from_dict(blob["cfg"])
    model = build_model(cfg, int(blob["obs_dim"]), N_ACT)
    model.load_state_dict(blob["state"])
    model.eval()
    cache.update({"key": key, "model": model, "version": int(blob.get("version", 0))})
    return model, cache["version"]


def worker_main(wid: int, gpu: int | None, cfg_dict: dict, run_dir: str, job_q, result_q, stop_ev, train_ev,
                seed_block: int, log_path: str, scratch: str) -> None:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.setdefault("TMPDIR", "/tmp")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", buffering=1)
    # the simulator and cuRobo print from C as well: send file descriptors 1 and 2 to the log too
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    print(f"[worker {wid}] start pid {os.getpid()} gpu {gpu} seed block {seed_block} "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        from .config import ResidualConfig
        from .episode import Job, run_episode
        from .tasks import task_hooks

        cfg = ResidualConfig.from_dict(cfg_dict)
        run = Path(run_dir)
        rng = np.random.default_rng([int(seed_block), int(wid), 7919])
        # SEEDS_PER_BLOCK seeds per worker life (a life builds at most max_inits_per_process scenes); a block never
        # reaches the evaluation seeds
        lo = cfg.seed_base_train + seed_block * SEEDS_PER_BLOCK
        if lo + SEEDS_PER_BLOCK > cfg.seed_base_eval:
            raise RuntimeError(f"seed block {seed_block} would reach the evaluation seeds ({cfg.seed_base_eval})")
        seeds = iter(range(lo, lo + SEEDS_PER_BLOCK))
        hooks = task_hooks(cfg.task)
        balancer = hooks.make_chooser(cfg)   # T1: SideBalancer; tray: glass / can equally often
        cache: dict = {}
        builds = 0
        while not stop_ev.is_set():
            try:
                job = job_q.get(timeout=0.5)
            except queue.Empty:
                job = None
            if job is None:
                if not train_ev.is_set() or not (run / "weights.pt").exists():
                    continue
                job = Job(kind="train")
            elif isinstance(job, dict):
                job = Job(**job)
            result_q.put(("start", wid, job.job_id, job.kind))
            t0 = time.perf_counter()
            if job.kind == "scan":
                try:
                    r = hooks.build_runner(cfg, int(job.seed), scratch, job)
                    builds += 1
                    try:
                        res = {"kind": "scan", "job_id": job.job_id, "seed": int(job.seed), "tag": job.tag,
                               **hooks.scan_record(r)}
                    finally:
                        r.close()
                except Exception as exc:
                    builds += 1
                    res = {"kind": "scan", "job_id": job.job_id, "seed": int(job.seed), "tag": job.tag,
                           "error": f"{type(exc).__name__}: {exc}"}
            else:
                model, version = None, -1
                if job.policy == "model":
                    wpath = Path(job.weights) if job.weights else run / "weights.pt"
                    model, version = _load_weights(wpath, cache)
                res = run_episode(cfg, job, model, rng, scratch,
                                  choose=lambda runner, r_, _j=job: balancer(runner, r_, _j),
                                  next_seed=lambda: next(seeds))
                res["version"] = version
                builds += int(res.get("builds", 0))
            res["wid"] = wid
            res["wall_job"] = time.perf_counter() - t0
            _send_result(result_q, Path(scratch) / "spool", wid, res)
            if builds >= cfg.max_inits_per_process:
                print(f"[worker {wid}] {builds} Env builds: restarting (exit {EXIT_RESTART})", flush=True)
                log.flush()
                _exit(result_q, log, EXIT_RESTART)
    except Exception as exc:
        traceback.print_exc()
        try:
            result_q.put(("dead", wid, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        _exit(result_q, log, 1)
    _exit(result_q, log, 0)
