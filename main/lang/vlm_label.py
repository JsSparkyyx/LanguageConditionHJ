"""The VLM labeler (VLM_FAILURE_PLAN, Yuxuan 2026-10-26): the failure function h of a run without the simulator's h.

For every episode of the given data roots whose agent-view frames are complete (``agent_view/<id>/done.json``,
agent_view.py), every rule the VLM answers for (vlm_prompts.labelled: T1 avoid, T5 upright, T6 exclude_particles)
and every saved tick, the VLM sees the episode's first agent-view frame and the frame of that tick with the rule's
prompt (vlm_prompts.yaml) and answers -1 / 0 / 1. One label file per episode (vlm_prompts module docstring), written
atomically; hsource.py turns it into h at training time.

    python -m main.lang.vlm_label serve --vlm qwen25vl32b --roots <ring> <warmup>/shard_g* [--until <file>]
    python -m main.lang.vlm_label serve ... --once          # label what is there, then exit
    python -m main.lang.vlm_label serve ... --done-file F --done-roots '<warmup>/shard_g*' --done-after <data READY>
        # writes F once the data is READY and every episode of those roots has a label (runs wait for F)
    python -m main.lang.vlm_label agree --vlm qwen25vl32b --roots ...   # VLM vs simulator h (no GPU)

Several labelers may serve the same roots: an episode is claimed first (``<id>.claim``, taken over after 30 min
without a touch). Roots are served in the order given, and the scan starts again from the first root after every
episode, so list a run's ring first: its episodes hold up training. A heartbeat file in every root's label dir tells
the waiting run (hsource.wait_labels) a labeler is alive.

Decoding: greedy, at most 128 new tokens; an answer without a parsable score is asked again once with sampling
(temperature 0.7), then recorded as score 0 with parsed = false.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from . import ensure_item_path

ensure_item_path()

from .agent_view import frame_path, read_done  # noqa: E402
from .vlm_prompts import (QUANT, VLLM_ONLY, VLMS, build_prompt, label_dir, label_path, labelled, load_prompts, parse_score,  # noqa: E402
                          prompt_hash, write_json_atomic)

CLAIM_STALE_S = 600.0   # a claim is touched after every batch (~10 s): 10 min untouched = its labeler died
HEARTBEAT_S = 60.0
MAX_NEW_TOKENS = 128


def expand_roots(patterns) -> list[Path]:
    """Globs and plain paths in order, duplicates dropped; a plain path that does not exist yet (a ring before its
    run starts) is kept and skipped while missing."""
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p), key=lambda s: (len(s), s)) if any(c in p for c in "*?[") else [p]
        for h in hits:
            q = Path(h).resolve()
            if q not in out:
                out.append(q)
    return out


def pending(root: Path, vlm: str) -> list[int]:
    """Episode ids of this root with complete frames and no label file yet, ascending."""
    av = root / "agent_view"
    if not av.exists():
        return []
    out = []
    for d in av.iterdir():
        if d.is_dir() and d.name.isdigit() and (d / "done.json").exists() and not label_path(root, vlm, int(d.name)).exists():
            out.append(int(d.name))
    return sorted(out)


def claim(root: Path, vlm: str, eid: int) -> Path | None:
    p = label_dir(root, vlm) / f"{eid}.claim"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - p.stat().st_mtime < CLAIM_STALE_S:
                return None
            p.unlink()   # a dead labeler's claim
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileNotFoundError, FileExistsError):
            return None
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "t": time.time()}))
    return p


def episode_queries(root: Path, eid: int) -> tuple[dict, list[dict], dict | None]:
    """(label file skeleton, queries, record). A query = {"rule_id", "idx", "tick", "prompt", "referent"}. An
    episode that cannot be labelled gets a skeleton with "error" set and no queries."""
    from main.item_envs import language

    done = read_done(root, eid)
    skel = {"episode_id": int(eid), "prompt_hash": prompt_hash(), "every": (done or {}).get("every"),
            "n_rows": (done or {}).get("n_rows"), "ticks": list((done or {}).get("ticks") or []), "rules": {},
            "error": None}
    rp = root / "item_buffer" / "episodes" / f"{eid}.json"
    if done is None or not rp.exists():
        skel["error"] = "no done.json" if done is None else "no episode record"
        return skel, [], None
    mids = language.load_model_ids(root / "item_buffer")
    rec = language.refresh_record(json.loads(rp.read_text()), mids.get(int(eid)))
    ticks = skel["ticks"]
    if not ticks or ticks[0] != 0 or not all(frame_path(root, eid, t).exists() for t in ticks):
        skel["error"] = "frames missing (tick 0 or a listed tick)"
        return skel, [], rec
    if rec["task"] not in load_prompts()["prompts"]:
        skel["error"] = None
        skel["note"] = f"no prompts for {rec['task']}"
        return skel, [], rec
    qs = []
    for rule in rec["rules"]:
        if not labelled(rule, rec):
            continue
        bp = build_prompt(rule, rec)
        if bp is None:
            continue
        prompt, phrase = bp
        skel["rules"][rule["rule_id"]] = {"idx": int(rule["idx"]), "referent": phrase, "consequence": rule["consequence"],
                                          "score": [], "explanation": [], "answer": [], "parsed": []}
        qs += [{"rule_id": rule["rule_id"], "tick": int(t), "prompt": prompt} for t in ticks]
    return skel, qs, rec


class VLM:
    """One model of VLMS (Qwen2.5-VL or Qwen3-VL architecture; AutoModelForImageTextToText picks the class from its
    config), bf16, batched generation with left padding."""

    def __init__(self, name: str, attn: str = "flash_attention_2"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name, self.dir = name, VLMS[name]
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.dir), torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=attn).eval()
        self.proc = AutoProcessor.from_pretrained(str(self.dir))
        self.proc.tokenizer.padding_side = "left"
        self.torch = torch
        print(f"[vlm_label] {name} loaded from {self.dir} in {time.time() - t0:.0f}s, GPU memory "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB, attention {attn}", flush=True)

    def ask(self, batch: list[tuple[list, str]], sample: bool = False) -> list[str]:
        """batch: [(images, text)] -> answers. images: PIL images in order (here: the first frame, the frame now)."""
        texts, images = [], []
        for imgs, text in batch:
            msg = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]}]
            texts.append(self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            images += imgs
        inputs = self.proc(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        kw = ({"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 2 * MAX_NEW_TOKENS} if sample
              else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                    "max_new_tokens": MAX_NEW_TOKENS})
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class VLLMVLM:
    """The same interface on vLLM (2026-10-27), for the 4-bit models (QUANT) on one A40: vLLM batches the prompts
    itself, so ask() takes a whole episode's questions at once. Greedy decoding, as the transformers backend; the
    prompt is the model's own chat template (AutoProcessor), the images go in as PIL. Runs in the separate env
    /work/nvme/bifq/conda/envs/vlm (vllm 0.11.0), never in the RoboTwin env."""

    def __init__(self, name: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.88):
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.name, self.dir = name, VLMS[name]
        quant = QUANT.get(name)
        t0 = time.time()
        kw = {"quantization": quant} if quant else {}
        if quant == "bitsandbytes":   # quantized at load time from the bf16 checkpoint
            kw["load_format"] = "bitsandbytes"
        # AWQ kernels want fp16 activations; the bitsandbytes and bf16 checkpoints run in bf16
        dtype = "float16" if quant == "awq" else "bfloat16"
        # every visible GPU (tensor parallel): one A40 for the 4-bit / 8-bit models, two A100s for bf16 32B
        tp = max(1, torch.cuda.device_count())
        self.llm = LLM(model=str(self.dir), dtype=dtype, max_model_len=int(max_model_len),
                       limit_mm_per_prompt={"image": 2}, gpu_memory_utilization=float(gpu_memory_utilization),
                       tensor_parallel_size=tp, seed=0, **kw)
        self.proc = AutoProcessor.from_pretrained(str(self.dir))
        self.SamplingParams = SamplingParams
        print(f"[vlm_label] {name} loaded on vLLM from {self.dir} in {time.time() - t0:.0f}s (quantization "
              f"{quant or 'none'}, dtype {dtype}, tensor parallel {tp})", flush=True)

    def ask(self, batch: list[tuple[list, str]], sample: bool = False) -> list[str]:
        reqs = []
        for imgs, text in batch:
            msg = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]}]
            prompt = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": list(imgs)}})
        sp = (self.SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2 * MAX_NEW_TOKENS, seed=0) if sample
              else self.SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
        outs = self.llm.generate(reqs, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def make_vlm(name: str, backend: str, attn: str = "flash_attention_2", gpu_mem: float = 0.88):
    """The labeler's model: a quantized model (QUANT) always runs on vLLM."""
    if backend == "vllm" or name in QUANT or name in VLLM_ONLY:
        return VLLMVLM(name, gpu_memory_utilization=gpu_mem)
    return VLM(name, attn=attn)


def label_episode(vlm: VLM, root: Path, eid: int, batch_size: int, touch=None) -> dict:
    from PIL import Image

    t0 = time.time()
    skel, qs, _ = episode_queries(root, eid)
    skel.update({"vlm": vlm.name, "model_dir": str(vlm.dir), "batch": int(batch_size), "host": socket.gethostname(),
                 "backend": type(vlm).__name__, "quant": QUANT.get(vlm.name)})
    if not qs:
        skel["wall_s"] = round(time.time() - t0, 1)
        return skel
    imgs = {t: Image.open(frame_path(root, eid, t)).convert("RGB") for t in skel["ticks"]}
    answers = [None] * len(qs)
    for i in range(0, len(qs), batch_size):
        part = qs[i:i + batch_size]
        for k, a in enumerate(vlm.ask([([imgs[0], imgs[q["tick"]]], q["prompt"]) for q in part])):
            answers[i + k] = a
        if touch is not None:
            touch()
    retry = [k for k, a in enumerate(answers) if parse_score(a)[0] is None]
    for i in range(0, len(retry), batch_size):
        part = retry[i:i + batch_size]
        for k, a in zip(part, vlm.ask([([imgs[0], imgs[qs[k]["tick"]]], qs[k]["prompt"]) for k in part], sample=True)):
            if parse_score(a)[0] is not None:
                answers[k] = a
    for q, a in zip(qs, answers):
        score, expl = parse_score(a)
        r = skel["rules"][q["rule_id"]]
        r["score"].append(0 if score is None else int(score))
        r["explanation"].append(expl)
        r["answer"].append(a)
        r["parsed"].append(score is not None)
    skel["n_queries"] = len(qs)
    skel["n_unparsed"] = sum(not p for r in skel["rules"].values() for p in r["parsed"])
    skel["wall_s"] = round(time.time() - t0, 1)
    return skel


def heartbeat(roots, vlm: str, state: dict) -> None:
    body = {"host": socket.gethostname(), "pid": os.getpid(), "t": time.time(), **state}
    for r in roots:
        if r.exists():
            try:
                write_json_atomic(label_dir(r, vlm) / f"heartbeat_{socket.gethostname()}_{os.getpid()}.json", body)
            except OSError:
                pass


def all_labelled(args) -> bool:
    """--done-file: the data is complete (--done-after exists) and every episode of --done-roots has a label."""
    from .hsource import missing_labels

    if args.done_after and not Path(args.done_after).exists():
        return False
    roots = expand_roots(args.done_roots)
    return bool(roots) and not any(missing_labels(r, args.vlm) for r in roots)


def cmd_serve(args) -> int:
    roots = expand_roots(args.roots)
    deadline = (time.time() + args.time_budget) if args.time_budget else (
        float(os.environ["STAGE_DEADLINE"]) if os.environ.get("STAGE_DEADLINE") else None)
    from .hsource import wanted

    vlm = make_vlm(args.vlm, args.backend, attn=args.attn, gpu_mem=args.gpu_mem)
    state = {"vlm": args.vlm, "backend": type(vlm).__name__, "episodes": 0, "queries": 0, "started": time.time(), "prompt_hash": prompt_hash()}
    last_hb = 0.0
    walls: list[float] = []
    idle_since = time.time()
    while True:
        if time.time() - last_hb > HEARTBEAT_S:
            heartbeat(roots, args.vlm, state)
            last_hb = time.time()
        if args.until and Path(args.until).exists():
            print(f"[vlm_label] {args.until} exists: done", flush=True)
            return 0
        est = float(np.mean(walls[-20:])) if walls else 240.0
        if deadline is not None and time.time() + 1.2 * est > deadline:
            print(f"[vlm_label] time budget (next episode ~{est:.0f}s): exit 75", flush=True)
            return 75
        if args.max_episodes and state["episodes"] >= args.max_episodes:
            return 0
        took = False
        for root in roots:
            for eid in pending(root, args.vlm):
                cl = claim(root, args.vlm, eid)
                if cl is None:
                    continue

                def touch(cl=cl):
                    nonlocal last_hb
                    os.utime(cl)
                    if time.time() - last_hb > HEARTBEAT_S:
                        heartbeat(roots, args.vlm, state)
                        last_hb = time.time()

                try:
                    lab = label_episode(vlm, root, eid, args.batch, touch=touch)
                    write_json_atomic(label_path(root, args.vlm, eid), lab)
                finally:
                    cl.unlink(missing_ok=True)
                state["episodes"] += 1
                state["queries"] += int(lab.get("n_queries", 0))
                walls.append(float(lab["wall_s"]))
                sc = [s for r in lab["rules"].values() for s in r["score"]]
                print(f"[vlm_label] {root.name}/{eid}: {lab.get('n_queries', 0)} questions in {lab['wall_s']:.0f}s, "
                      f"scores -1/0/1 = {sc.count(-1)}/{sc.count(0)}/{sc.count(1)}, unparsed "
                      f"{lab.get('n_unparsed', 0)}{' ERROR ' + lab['error'] if lab.get('error') else ''}", flush=True)
                took = True
                break
            if took:
                break
        if not took and args.done_file and not Path(args.done_file).exists() and all_labelled(args):
            write_json_atomic(Path(args.done_file), {"vlm": args.vlm, "prompt_hash": prompt_hash(), "t": time.ctime(),
                                                     "roots": [str(r) for r in expand_roots(args.done_roots)]})
            print(f"[vlm_label] every episode of --done-roots labelled: wrote {args.done_file}", flush=True)
        if took:
            idle_since = time.time()
        elif wanted(roots, args.vlm):
            idle_since = time.time()
        elif args.done_file and not Path(args.done_file).exists() and args.done_after and Path(args.done_after).exists():
            idle_since = time.time()   # the done file is still to be written (labels of episodes still arriving)
        elif args.idle_exit and time.time() - idle_since > args.idle_exit:
            print(f"[vlm_label] idle {args.idle_exit:.0f}s, no run wants labels: exit 75", flush=True)
            return 75
        if not took:
            if args.once:
                print(f"[vlm_label] nothing left in {len(roots)} roots: done ({state['episodes']} episodes)", flush=True)
                return 0
            time.sleep(args.poll)


def cmd_agree(args) -> int:
    """VLM vs simulator h on every labelled training cell of these roots, per task (hsource.HSourceSampler)."""
    from .hsource import HSourceSampler

    tot: dict = {}
    for root in expand_roots(args.roots):
        if not (root / "item_buffer" / "header.json").exists():
            continue
        s = HSourceSampler(root / "item_buffer", h_source=f"vlm:{args.vlm}", rule_consequences="vlm_prompts")
        for task, c in s.agreement().items():
            o = tot.setdefault(task, {})
            for k, v in c.items():
                o[k] = o.get(k, 0) + v
    rows = []
    for task, c in sorted(tot.items()):
        n = max(c["cells"], 1)
        rows.append({"task": task, "cells": c["cells"], "agree": round(c["agree"] / n, 3),
                     "vlm_neg": round(c["vlm_neg"] / n, 3), "sim_neg": round(c["sim_neg"] / n, 3),
                     "recall_of_sim_neg": round(c["both_neg"] / max(c["sim_neg"], 1), 3),
                     "precision_of_vlm_neg": round(c["both_neg"] / max(c["vlm_neg"], 1), 3),
                     "fails_soon_agree": round(c["fs_agree"] / n, 3)})
    print(json.dumps(rows, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--vlm", required=True, choices=sorted(VLMS))
    p.add_argument("--roots", nargs="+", required=True, help="data roots or globs, in priority order (ring first)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--attn", default="flash_attention_2", choices=("flash_attention_2", "sdpa", "eager"))
    p.add_argument("--gpu-mem", type=float, default=0.88, help="vLLM: share of each GPU's memory it may use")
    p.add_argument("--backend", default="hf", choices=("hf", "vllm"),
                   help="hf = transformers (bf16 models); vllm = the separate vLLM env (the 4-bit models always use it)")
    p.add_argument("--poll", type=float, default=20.0, help="seconds between scans when idle")
    p.add_argument("--once", action="store_true", help="exit when nothing is left to label")
    p.add_argument("--until", default=None, help="exit when this file exists (e.g. the run's checkpoint_final.pt)")
    p.add_argument("--time-budget", type=float, default=None, help="seconds (default: $STAGE_DEADLINE)")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--idle-exit", type=float, default=0.0,
                   help="exit 75 after this many idle seconds with no fresh WANTED marker (0 = never)")
    p.add_argument("--done-file", default=None, help="written once --done-roots are fully labelled")
    p.add_argument("--done-roots", nargs="*", default=[])
    p.add_argument("--done-after", default=None, help="only after this file exists (the data's READY)")
    p = sub.add_parser("agree")
    p.add_argument("--vlm", required=True, choices=sorted(VLMS))
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return cmd_serve(args) if args.cmd == "serve" else cmd_agree(args)


if __name__ == "__main__":
    raise SystemExit(main())
