"""The corpus page's T6 picture (docs/lang_filter/corpus_page/img/scene_clean_pour.jpg): the start frame of a new-T6
scene (bowl_pour, 2026-10-23), observer camera, no overlay, 640 x 460 like the other scene_*.jpg (the 640 x 512
observer frame without its bottom 52 rows). ur5-wsg.

    python docs/lang_filter/corpus_page/render_scene.py [seed]
"""
import sys
import tempfile
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import main.lang  # noqa: F401,E402
from main.envs.record import observer_rgb  # noqa: E402
from main.item_envs.runner import ItemEpisodeRunner  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 6300
OUT = REPO / "docs/lang_filter/corpus_page/img/scene_clean_pour.jpg"

with tempfile.TemporaryDirectory() as tmp:
    r = ItemEpisodeRunner("clean_pour", embodiment="ur5-wsg", seed=SEED, render="full", out_root=tmp)
    try:
        r.setup()
        task = r.env.task
        f = observer_rgb(task)
        print("frame", f.shape, "target", task.target_color, "others", [n["color"] for n in task.neighbors], flush=True)
        if f.shape[:2] != (460, 640):
            f = cv2.resize(f, (640, 512), interpolation=cv2.INTER_AREA)[:460]
        cv2.imwrite(str(OUT), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", OUT, flush=True)
    finally:
        r.close()
