"""Recover the model ids of the T1 pool objects of episodes recorded without them (object names check, 2026-10-12).

    cd <a checkout of the code that COLLECTED the data, e.g. /work/nvme/bifq/project/LanguageConditionHJ_exp4>
    python /work/nvme/bifq/project/LanguageConditionHJ/scripts/lang/recover_model_ids.py \
        --roots data/lang/warmup data/lang/val data/lang/val_grounding \
        --copy-to data/lang_v2/warmup data/lang_v2/val data/lang_v2/val_grounding

objects.yaml now names some objects per model (bottle id 19 is a water bottle, 068_boxdrink id 0 is not a bottle,
086_woodenblock ids 1 / 2 are not blocks), but episodes collected before 2026-10-12 recorded the category only.
A T1 layout is a function of the episode seed, host and embodiment, so this rebuilds each recorded T1 scene
(main.item_envs.margin_hook.build_item_training_env, no rollout) with the code of the current directory (it must
be the code and objects.yaml the data was collected with: the import path is the cwd, not this file's checkout)
and reads the spawn report's asset and model id per object. A scene counts only if every recorded object comes
back with the same category and position (|dxyz| <= --tol): otherwise its ids are not trusted and it is listed.

Writes <shard>/item_buffer/model_ids.json ({"episodes": {episode_id: {item_id: {asset, model_id}}}, "code": git
head, "mismatch": [...]}), atomically after every episode, so a stopped job resumes. --copy-to: version-2 replays of
the same episodes (data/lang_v2, the same seeds and layouts) get the entries of their source episode when the
recorded positions agree. language.load_model_ids / refresh_record read the file when records are loaded.
Needs the simulator (an A40 job); exit 75 when $STAGE_DEADLINE is near (rerun to continue).
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())   # the collecting checkout's main/ (see the docstring), not this file's


def shards(roots: list[str]) -> list[Path]:
    out = []
    for r in roots:
        p = Path(r)
        out += [p] if (p / "item_buffer").is_dir() else sorted(Path(x) for x in glob.glob(str(p / "shard_g*")))
    return [s for s in out if not s.name.endswith("_bead")]


def t1_lines(shard: Path) -> list[dict]:
    out = []
    for ln in (shard / "lang_episodes.jsonl").read_text().splitlines():
        d = json.loads(ln)
        if d.get("skipped") or d.get("task") != "selective_avoid":
            continue
        if (shard / "item_buffer" / "episodes" / f"{d['episode_id']}.json").exists():
            out.append(d)
    return out


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _load(path: Path, code: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"episodes": {}, "mismatch": [], "code": code, "source": "recover_model_ids.py"}


def check(line: dict, record: dict, placements: list[dict], tol: float) -> str | None:
    """None when the rebuilt scene is the recorded one, else why not."""
    rec_xyz = {p["item_id"]: p["xyz"] for p in (line.get("scene_meta") or {}).get("placements") or []}
    got = {p["item_id"]: p for p in placements}
    if set(rec_xyz) != set(got):
        return f"items {sorted(rec_xyz)} recorded, {sorted(got)} rebuilt"
    cats = {o["item_id"]: o["category_key"] for o in record["objects"]}
    for i, xyz in rec_xyz.items():
        d = float(np.abs(np.asarray(xyz) - np.asarray(got[i]["xyz"])).max())
        if d > tol:
            return f"{i}: |dxyz| {d:.2e} m"
        if cats.get(i) != got[i].get("category"):
            return f"{i}: category {cats.get(i)} recorded, {got[i].get('category')} rebuilt"
    return None


def recover_shard(shard: Path, code: str, tol: float, deadline: float | None) -> bool:
    """Fill <shard>/item_buffer/model_ids.json. False when stopped by the deadline."""
    from main.item_envs.margin_hook import build_item_training_env
    from main.lang.config import LangConfig

    out = shard / "item_buffer" / "model_ids.json"
    data = _load(out, code)
    base = LangConfig.from_dict(json.loads((shard / "config.json").read_text()))
    todo = [d for d in t1_lines(shard) if str(d["episode_id"]) not in data["episodes"]
            and not any(m.get("episode_id") == d["episode_id"] for m in data["mismatch"])]
    print(f"[recover] {shard}: {len(todo)} T1 episodes to rebuild", flush=True)
    for d in todo:
        if deadline is not None and time.time() > deadline:
            return False
        eid = int(d["episode_id"])
        record = json.loads((shard / "item_buffer" / "episodes" / f"{eid}.json").read_text())
        cfg = dataclasses.replace(base, task="selective_avoid", embodiment=d["embodiment"], item_host=d.get("host"),
                                  item_active_rule=None,
                                  item_out_root=str(Path(os.environ.get("TMPDIR", "/tmp")) / "recover_model_ids"))
        t0 = time.time()
        env = None
        try:
            env = build_item_training_env(cfg, int(d["seed"]))
            rep = getattr(env._item_runner, "_spawn_report", None) or {}
            placements = rep.get("placements") or []
            why = check(d, record, placements, tol)
            if why is None:
                data["episodes"][str(eid)] = {p["item_id"]: {"asset": p["asset"], "model_id": p["model_id"]}
                                              for p in placements}
            else:
                data["mismatch"].append({"episode_id": eid, "why": why})
        except Exception as exc:   # one scene failing must not stop the rest
            why = f"{type(exc).__name__}: {str(exc)[:200]}"
            data["mismatch"].append({"episode_id": eid, "why": why})
        finally:
            try:
                env.close() if env is not None else None
            except Exception:
                pass
        _write(out, data)
        print(f"[recover] {shard.name} {eid} ({d['embodiment']}, {d.get('host')}): "
              f"{'ok' if why is None else why} ({time.time() - t0:.0f}s)", flush=True)
    return True


def copy_to(sources: list[Path], targets: list[Path], tol: float) -> None:
    """Version-2 replays: the source episode's ids when the recorded positions agree."""
    by_ep, xyz = {}, {}
    for s in sources:
        p = s / "item_buffer" / "model_ids.json"
        if not p.exists():
            continue
        ids = json.loads(p.read_text())["episodes"]
        for d in t1_lines(s):
            if str(d["episode_id"]) in ids:
                key = (d["embodiment"], int(d["episode_id"]))
                by_ep[key] = ids[str(d["episode_id"])]
                xyz[key] = {q["item_id"]: q["xyz"] for q in d["scene_meta"]["placements"]}
    for t in targets:
        out = t / "item_buffer" / "model_ids.json"
        data = {"episodes": {}, "mismatch": [], "code": "copied", "source": [str(s) for s in sources]}
        for d in t1_lines(t):
            key = (d["embodiment"], int(d["episode_id"]))
            mine = {q["item_id"]: q["xyz"] for q in (d.get("scene_meta") or {}).get("placements") or []}
            if key not in by_ep:
                data["mismatch"].append({"episode_id": key[1], "why": "no recovered source episode"})
            elif set(mine) != set(xyz[key]) or any(
                    float(np.abs(np.asarray(mine[i]) - np.asarray(xyz[key][i])).max()) > tol for i in mine):
                data["mismatch"].append({"episode_id": key[1], "why": "positions differ from the source episode"})
            else:
                data["episodes"][str(key[1])] = by_ep[key]
        _write(out, data)
        print(f"[recover] {t}: copied {len(data['episodes'])}, mismatched {len(data['mismatch'])}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="data roots or shard dirs whose T1 episodes to rebuild")
    ap.add_argument("--copy-to", nargs="*", default=[], help="version-2 replays of those episodes (data/lang_v2/...)")
    ap.add_argument("--tol", type=float, default=1e-5, help="largest |dxyz| (m) for the rebuilt scene to count")
    args = ap.parse_args(argv)
    code = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
    deadline = float(os.environ["STAGE_DEADLINE"]) - 300 if os.environ.get("STAGE_DEADLINE") else None
    src = shards(args.roots)
    for s in src:
        if not recover_shard(s, code, args.tol, deadline):
            print("[recover] deadline: stopping (rerun to continue)", flush=True)
            return 75
    if args.copy_to:
        copy_to(src, shards(args.copy_to), args.tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
