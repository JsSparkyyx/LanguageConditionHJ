"""Non-sim tests: physics sampling — the per-category friction_range of
objects.yaml (T2 beads: slippery, user review 8) remaps only that category."""

from __future__ import annotations

import numpy as np

from main.item_envs.items import ItemSpec
from main.item_envs.physics import load_objects_cfg, load_physics_cfg, sample_physics


def _specs():
    return {
        "beads": ItemSpec(item_id="beads", entity=object(), category="beads", color="", kind="particle_set",
                          particles=[object()] * 3, role="candidate", half_extents=[0.006] * 3),
        "bowl": ItemSpec(item_id="bowl", entity=object(), category="bowl", color="", role="task_object",
                         half_extents=[0.06, 0.03, 0.06]),
    }


def test_bead_friction_range_applies_to_beads_only():
    lo, hi = load_objects_cfg()["categories"]["beads"]["friction_range"]
    for seed in range(40):
        out = sample_physics(_specs(), load_physics_cfg(), np.random.RandomState(seed))
        assert lo - 1e-9 <= out["beads"]["static_friction"] <= hi + 1e-9
        assert out["beads"]["dynamic_friction"] <= out["beads"]["static_friction"] + 1e-12
        assert 0.3 - 1e-9 <= out["bowl"]["static_friction"] <= 0.9 + 1e-9  # physics.yaml range unchanged


def test_tray_dynamic_ratio_and_block_friction():
    cats = load_objects_cfg()["categories"]
    specs = {
        "tray": ItemSpec(item_id="tray", entity=object(), category="tray", color="", role="task_object",
                         half_extents=[0.15, 0.1, 0.035]),
        "block": ItemSpec(item_id="block", entity=object(), category="block", color="red", role="candidate",
                          half_extents=[0.02, 0.02, 0.07]),
    }
    for seed in range(30):
        out = sample_physics(specs, load_physics_cfg(), np.random.RandomState(seed))
        tr = out["tray"]
        assert abs(tr["dynamic_friction"] - cats["tray"]["dynamic_ratio"] * tr["static_friction"]) < 1e-9
        lo, hi = cats["block"]["friction_range"]
        assert lo - 1e-9 <= out["block"]["static_friction"] <= hi + 1e-9
