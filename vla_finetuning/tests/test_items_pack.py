"""An item demo root (collect_items on the fake runner of _items_fakes.py: its runner/, settings.json, attempts.jsonl
and writer lock beside data/ and instructions/) packs with vla_finetuning/pack.py unchanged and loads through
HoloBrain's RoboTwinLmdbDataset: task directory names, uuids, cameras, frame counts, texts. Needs robo_orchard_lab,
h5py, lmdb, cv2 (the RoboTwin conda env has them):

    python -m pytest -q vla_finetuning/tests/test_items_pack.py
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from _items_fakes import install

from vla_finetuning import common as C
from vla_finetuning import pack
from vla_finetuning.items import collect_items as CI


def test_item_root_packs_unchanged(tmp_path, monkeypatch):
    lmdb_ds = pytest.importorskip("robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset")
    root = tmp_path / "demos_items_safe"
    install(monkeypatch, {340004: {"move2": 75}})                       # a shorter episode
    assert CI.main(["--task", "gap_place", "--root", str(root), "--config-name", "s0", "--seed-start", "340000",
                    "--seed-step", "4", "--seed-stop", "340100", "--target", "2"]) == 0
    install(monkeypatch, {})
    assert CI.main(["--task", "selective_avoid", "--host", "stack_blocks_two", "--root", str(root),
                    "--config-name", "stack_blocks_two", "--seed-start", "302100", "--seed-stop", "302200",
                    "--target", "1"]) == 0
    assert sorted(p.name for p in root.iterdir()) == ["gap_place", "selective_avoid"]   # task directories only
    out = tmp_path / "lmdb"
    assert pack.main(["--root", str(root), "--out", str(out), "--no-check"]) == 0
    exp = pack.expected(root)
    eps = {}                                                            # uuid -> (hdf5, instructions)
    for task, cfg in (("gap_place", "s0"), ("selective_avoid", "stack_blocks_two")):
        d = root / task / cfg
        seeds = (d / "seed.txt").read_text().split()
        for k, seed in enumerate(seeds):
            eps[f"{task}_{cfg}_seed{seed}"] = (d / "data" / f"episode{k}.hdf5",
                                               json.loads((d / "instructions" / f"episode{k}.json").read_text()))
    assert len(eps) == 3 and exp["gap_place"][0] == 2 and exp["selective_avoid"][0] == 1
    ds = lmdb_ds.RoboTwinLmdbDataset(paths=[str(out)], lazy_init=False, transforms=None, dataset_name="t",
                                     cam_names=list(C.CAMS))
    assert len(ds) == sum(v[1] for v in exp.values())                   # pack.py --check's comparison
    seen = set()
    for i in range(len(ds)):
        item = ds[i]
        uuid = item["uuid"]
        path, instr = eps[uuid]
        t = int(item["step_index"])
        with h5py.File(path, "r") as f:
            vec = f["joint_action/vector"][:]
        assert np.allclose(np.asarray(item["joint_state"]), vec)
        assert item["text"] in instr["seen"] + instr["unseen"]
        assert len(item["imgs"]) == len(C.CAMS) and len(item["depths"]) == len(C.CAMS)
        seen.add((uuid, t))
    assert {u for u, _ in seen} == set(eps)
