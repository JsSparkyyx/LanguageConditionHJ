"""Fresh example sentences for the corpus page (docs/lang_filter/corpus_page): the five used tasks' scenes built with
today's code (setup only, no episode), their records, and per sampleable rule two different training fills plus the
held-out-template fill. Writes docs/lang_filter/template_review/examples_v3.json in examples_v2.json's format
({consequence: [{task, train: [s1, s2], heldout: [s3]}]}); examples_v2.json (the old scenes) stays for the review page.

    python docs/lang_filter/corpus_page/page_examples.py [seeds_per_task]
"""
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import main.lang  # noqa: F401,E402
from main.item_envs import language  # noqa: E402
from main.item_envs.runner import ItemEpisodeRunner  # noqa: E402
from main.lang.config import TRAIN_TASKS  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = REPO / "docs/lang_filter/template_review/examples_v3.json"
HOSTS = ("place_container_plate", "place_burger_fries", "stack_blocks_two")   # T1: one host per seed
ex = defaultdict(list)

for task in TRAIN_TASKS:
    for k in range(N):
        seed = 7100 + k
        kw = {"host": HOSTS[k % len(HOSTS)]} if task == "selective_avoid" else {}
        with tempfile.TemporaryDirectory() as tmp:
            r = ItemEpisodeRunner(task, embodiment="ur5-wsg", seed=seed, render="none", out_root=tmp, **kw)
            try:
                r.setup(build_controller=False)
                rec = language.episode_record(r, seed)
            except Exception as exc:  # noqa: BLE001  (an unstable layout: next seed)
                print(task, seed, "setup failed:", f"{type(exc).__name__}: {exc}"[:200], flush=True)
                continue
            finally:
                r.close()
        rng = np.random.default_rng(seed)
        for ru in rec["rules"]:
            if not ru.get("sampleable") or ru.get("set") or ru.get("split") != "train":
                continue
            try:
                train = []
                for _ in range(40):
                    s = language.sample_sentence(ru, rec, rng, "train")
                    if s not in train:
                        train.append(s)
                    if len(train) == 2:
                        break
                held = language.sample_sentence(ru, rec, rng, "heldout_template")
            except ValueError as exc:
                print(task, seed, ru["rule_id"], "no sentence:", exc, flush=True)
                continue
            if len(train) < 2:
                train = train * 2
            ex[ru["consequence"]].append({"task": task, "seed": seed, "rule_id": ru["rule_id"], "train": train,
                                          "heldout": [held]})
            print(task, seed, ru["rule_id"], train, held, flush=True)

OUT.write_text(json.dumps(ex, indent=1))
print("wrote", OUT, {c: len(v) for c, v in ex.items()}, flush=True)
