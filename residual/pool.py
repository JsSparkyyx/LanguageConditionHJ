"""A pool of worker processes on a job's GPUs (``worker.worker_main``), shared by training and the throughput run.

Workers are pinned round-robin to the visible GPUs (``CUDA_VISIBLE_DEVICES`` of the job). A worker that exits
(75 = planned restart after its Env-build budget, anything else = crash) is respawned with a fresh seed block; a
non-training job it was running goes back on the queue (at most twice).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import time
from pathlib import Path

from .worker import worker_main


def visible_gpus() -> list[int | None]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        return [int(x) for x in vis.split(",") if x.strip() != ""]
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout
        ids = [int(x) for x in out.split() if x.strip().isdigit()]
        return ids or [None]
    except Exception:
        return [None]


class WorkerPool:
    def __init__(self, cfg, run_dir: str | Path, n_workers: int, log_dir: str | Path, scratch: str | Path,
                 gpus: list | None = None, next_block: int = 0):
        self.cfg = cfg
        self.run_dir = str(Path(run_dir).resolve())
        self.n = int(n_workers)
        self.log_dir = Path(log_dir).resolve()
        self.scratch = str(Path(scratch).resolve())
        self.gpus = list(gpus) if gpus else visible_gpus()
        self.ctx = mp.get_context("spawn")
        self.job_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.stop_ev = self.ctx.Event()
        self.train_ev = self.ctx.Event()
        self.procs: dict[int, mp.Process] = {}
        self.inflight: dict[int, dict] = {}
        self.jobs: dict[int, dict] = {}
        self.retries: dict[int, int] = {}
        self.next_block = int(next_block)
        self.next_job_id = 0
        self.restarts = {"planned": 0, "crash": 0}
        Path(self.scratch).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    def _spawn(self, wid: int) -> None:
        gpu = self.gpus[wid % len(self.gpus)] if self.gpus else None
        block = self.next_block
        self.next_block += 1
        p = self.ctx.Process(
            target=worker_main,
            args=(wid, gpu, self.cfg.to_dict(), self.run_dir, self.job_q, self.result_q, self.stop_ev, self.train_ev,
                  block, str(self.log_dir / f"worker{wid:02d}.log"), self.scratch),
            daemon=False,
        )
        p.start()
        self.procs[wid] = p

    def start(self) -> None:
        for wid in range(self.n):
            self._spawn(wid)

    def submit(self, job: dict) -> int:
        job = dict(job)
        if job.get("job_id", -1) < 0:
            job["job_id"] = self.next_job_id
            self.next_job_id += 1
        self.jobs[job["job_id"]] = job
        self.job_q.put(job)
        return job["job_id"]

    def set_training(self, on: bool) -> None:
        (self.train_ev.set if on else self.train_ev.clear)()

    # ------------------------------------------------------------------------
    def poll(self, timeout: float = 1.0) -> list[dict]:
        """Results that arrived (blocks up to ``timeout`` for the first one); restarts dead workers."""
        out = []
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            try:
                msg = self.result_q.get(timeout=max(0.01, left)) if not out else self.result_q.get_nowait()
            except queue.Empty:
                break
            kind, wid = msg[0], msg[1]
            if kind == "start":
                _, _, job_id, jkind = msg
                if jkind != "train" and job_id in self.jobs:
                    self.inflight[wid] = self.jobs[job_id]
                else:
                    self.inflight.pop(wid, None)
            elif kind in ("result", "result_file"):
                if kind == "result_file":
                    try:
                        with open(msg[2], "rb") as f:
                            res = pickle.load(f)
                        os.unlink(msg[2])
                    except Exception as exc:
                        print(f"[pool] result file {msg[2]} unreadable ({exc}); job {msg[3]} dropped", flush=True)
                        self.inflight.pop(wid, None)
                        self.jobs.pop(msg[3], None)
                        continue
                else:
                    res = msg[2]
                self.inflight.pop(wid, None)
                self.jobs.pop(res.get("job_id", -1), None)
                out.append(res)
            elif kind == "dead":
                print(f"[pool] worker {wid} reported: {msg[2]}", flush=True)
        self._reap()
        return out

    def _reap(self) -> None:
        if self.stop_ev.is_set():
            return
        for wid, p in list(self.procs.items()):
            if p.is_alive():
                continue
            code = p.exitcode
            p.join(timeout=1)
            planned = code == 75
            self.restarts["planned" if planned else "crash"] += 1
            if not planned:
                print(f"[pool] worker {wid} died (exit {code}); see {self.log_dir}/worker{wid:02d}.log", flush=True)
            job = self.inflight.pop(wid, None)
            if job is not None:
                jid = job["job_id"]
                self.retries[jid] = self.retries.get(jid, 0) + 1
                if self.retries[jid] <= 2:
                    self.job_q.put(job)
                else:
                    print(f"[pool] job {jid} ({job.get('kind')}, seed {job.get('seed')}) failed 3 times: dropped",
                          flush=True)
                    self.jobs.pop(jid, None)
            self._spawn(wid)

    def pending_jobs(self) -> int:
        return len(self.jobs)

    def close(self, timeout: float = 60.0) -> None:
        self.stop_ev.set()
        # jobs nobody will take must not keep this process alive at exit (the queue's feeder thread is joined then)
        self.job_q.cancel_join_thread()
        try:
            while True:
                self.job_q.get_nowait()
        except Exception:
            pass
        t_end = time.time() + timeout
        while time.time() < t_end and any(p.is_alive() for p in self.procs.values()):
            try:   # keep reading: a worker flushing its last result waits for the reader (worker._exit)
                self.result_q.get(timeout=0.5)
            except Exception:
                pass
        for p in self.procs.values():
            if p.is_alive():
                p.terminate()
        for p in self.procs.values():
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

    def pids(self) -> dict[int, int]:
        return {wid: p.pid for wid, p in self.procs.items() if p.is_alive()}
