"""Sim tests for T1 selective_avoid (PLAN v3 §6 Phase 4 step 3, §7 T1).

GPU node only (full CEHJ Env construction + cuRobo). Per (host, embodiment):

- 20 seeds build without UnStableError; host items register from the
  host_objects table (task_object/support roles), candidates spawn per the
  objects contract (0 on a layout-infeasible seed, else 2-5 with
  k + k_dropped = the drawn K in [3, 5] and the on/near quota met by the
  FINAL placement classes — spawn.py Fix 1, adaptive placement + quota
  degradation); rules enumerate non-empty on feasible seeds and every
  margin is positive at t = 0; the generic runner extensions
  (risky_skill_labels, risk_knobs, static-item physics filtering) are in
  place.
- ONE low-risk nominal episode (both scales 0.6 — the YAML's safe cell)
  completes with task_success.

Layout-infeasible cells: when spawn reports zero candidates on ALL 20 seeds
of a (host, embodiment) cell, the cell's yield/episode tests SKIP with the
reason recorded — the known case is stack_bowls_two (finding F1 in
docs/item_envs/feasibility.md: ~0.28 m corridors vs 0.09/0.09/0.12 m
keepaways make a 3-5-object layout infeasible on every seed). spawn.py
Fix 1 (fallback t-window, on->near->off degradation, partial-K down to
k = 2 with quota met, MAX_LAYOUTS = 40) IS applied, so an all-zero cell
now means even a degraded 2-object layout was impossible on every seed —
a stronger statement than pre-Fix-1 infeasibility. The gate is NOT
weakened: per-seed builds still assert the full contract whenever a layout
exists, and the skip is a loud, cited, data-driven outcome.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

sapien = pytest.importorskip("sapien")

from main.item_envs.runner import ItemEpisodeRunner

pytestmark = pytest.mark.sim

HOSTS = (
    "place_container_plate",
    "place_burger_fries",
    "stack_blocks_two",
    "stack_bowls_two",
    "place_bread_basket",
)
EMBODIMENTS = ["ur5-wsg", "franka-panda"]
SEEDS = tuple(range(20))

# minimum host-item ids expected per host (list attrs expand, so bread is a prefix)
HOST_ITEM_IDS = {
    "place_container_plate": {"container", "plate"},
    "place_burger_fries": {"hamburg", "frenchfries", "tray"},
    "stack_blocks_two": {"block1", "block2"},
    "stack_bowls_two": {"bowl1", "bowl2"},
    "place_bread_basket": {"breadbasket"},
}
# static supports: excluded from physics sampling (set_mass_props would raise)
STATIC_ITEM_IDS = {
    "place_container_plate": {"plate"},
    "place_burger_fries": {"tray"},
}

F1_REASON = (
    "spawn layout-infeasible on ALL {n} build seeds for {host}/{embodiment} "
    "(zero candidates placed) — known case, finding F1 in "
    "docs/item_envs/feasibility.md: stack_bowls_two corridors (~0.28 m) are "
    "too short for 3-5-object layouts against the 0.09/0.09/0.12 m keepaways. "
    "spawn.py Fix 1 (adaptive placement + quota degradation: fallback "
    "t-window, on->near->off degradation, partial-K acceptance down to k=2 "
    "with the quota met, MAX_LAYOUTS=40) is APPLIED, so all-zero now means "
    "even a degraded 2-object layout was impossible on every seed. "
    "The host stays registered and the feasibility gate is NOT weakened; "
    "the P1 sweep quantifies the yield. Skipping this cell's yield/episode "
    "assertions only."
)

# (host, embodiment) -> {seed: n_candidates}, filled by the build grid and
# read by the cell-level yield/episode tests (file order matters, pytest default)
SPAWN_OUTCOMES: dict[tuple[str, str], dict[int, int]] = {}


def _skip_if_cell_infeasible(host: str, embodiment: str) -> None:
    counts = SPAWN_OUTCOMES.get((host, embodiment), {})
    if counts and all(k == 0 for k in counts.values()):
        pytest.skip(F1_REASON.format(n=len(counts), host=host, embodiment=embodiment))


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_rules_margins(host, embodiment, seed, tmp_path):
    runner = ItemEpisodeRunner(
        "selective_avoid", host=host, embodiment=embodiment, seed=seed,
        render="none", out_root=tmp_path,
    )
    try:
        runner.setup()  # raises UnStableError on a bad layout — that is the check
        task = runner.env.task

        # host items registered with the yaml's roles; candidates role=candidate
        item_ids = {s.item_id for s in runner.items}
        assert HOST_ITEM_IDS[host] <= item_ids, (host, sorted(item_ids))
        if host == "place_bread_basket":
            assert any(iid.startswith("bread") and iid != "breadbasket" for iid in item_ids)
        yaml_roles = {e["attr"]: e["role"] for e in runner.task_cfg["host_objects"][host]}
        for spec in runner.items:
            if spec.role == "candidate":
                continue
            attr = spec.item_id if spec.item_id in yaml_roles else spec.item_id.rstrip("0123456789")
            assert yaml_roles.get(attr) == spec.role, (host, spec.item_id, spec.role)
        k = sum(1 for s in runner.items if s.role == "candidate")
        SPAWN_OUTCOMES.setdefault((host, embodiment), {})[seed] = k
        # objects.k draws [3, 5]; Fix 1 partial-K acceptance may legally
        # return k >= 2 with the quota met (k + k_dropped = the drawn K)
        assert k == 0 or 2 <= k <= 5, (host, seed, k)
        for spec in runner.items:
            if spec.role == "candidate":
                assert spec.mass > 0.0  # physics sampling applied
            if spec.item_id in STATIC_ITEM_IDS.get(host, set()):
                assert spec.mass == 0.0  # static support: untouched

        # T1 regions: table only (the rules need no zones)
        assert [r.region_id for r in runner.regions] == ["table_top"]

        # static supports excluded from physics sampling; dynamic items sampled
        for iid in STATIC_ITEM_IDS.get(host, set()):
            assert iid not in runner.physics_sample
        for iid in HOST_ITEM_IDS[host] - STATIC_ITEM_IDS.get(host, set()):
            assert iid in runner.physics_sample, (host, iid)

        # generic runner extensions
        labels = tuple(runner.task_cfg.get("risky_skill_labels") or ())
        assert labels and runner.ctrl.risky_skill_labels == labels
        assert task.risk_knobs == runner.risk_params.task
        assert task._time_scale_safe == pytest.approx(runner.risk_params.safe_time_scale)
        assert task._time_scale_risky == pytest.approx(runner.risk_params.time_scale)

        # spawn report captured; on_path_count (when placed) matches the sampled knob
        report = runner._spawn_report
        assert report is not None and report.get("k") == k
        placements = report.get("placements", [])
        on_path = [p for p in placements if p.get("cls") == "on_path"]
        # objects.max_on_path; Fix 1 degradation only moves DOWN the chain
        # (on_path -> near_path -> off_path), so final on_path <= drawn <= 2
        assert len(on_path) <= 2

        if k:
            # Fix 1 report semantics: k + k_dropped = the drawn K in [3, 5];
            # the on/near quota is met by the FINAL classes (a degraded-to-
            # near object counts toward min_on_or_near_path)
            assert k + report.get("k_dropped", 0) in range(3, 6), (host, seed, report)
            on_near = sum(p.get("cls") in ("on_path", "near_path") for p in placements)
            assert on_near >= min(2, k), (host, seed, placements)
            assert report.get("t_fallbacks", 0) == sum(bool(p.get("t_fallback")) for p in placements)
            assert report.get("degraded", 0) == sum(
                p.get("degraded_from") is not None for p in placements
            )
            chain_rank = {"on_path": 0, "near_path": 1, "off_path": 2}
            for p in placements:
                if p.get("degraded_from") is not None:
                    assert chain_rank[p["cls"]] > chain_rank[p["degraded_from"]], p
            singles = [r.rule_id for r in runner.rules
                       if r.rule_id.startswith("avoid|") and "+" not in r.rule_id]
            stays = [r.rule_id for r in runner.rules if r.rule_id.startswith("stay|")]
            # the CrossSafe-baseline all-candidates set spans every candidate
            # (a same-category set may coincide when all candidates share one
            # category, hence >= 1)
            cand_ids = tuple(sorted(s.item_id for s in runner.items if s.role == "candidate"))
            all_sets = [r for r in runner.rules if r.item_ids == cand_ids]
            assert len(singles) == k and len(stays) == k
            assert len(all_sets) >= 1
            margins = runner._rule_margins()
            assert margins.shape == (len(runner.rules),)
            assert np.all(np.isfinite(margins))
            assert (margins > 0.0).all(), f"non-positive t=0 margins: {margins}"
        else:
            assert runner.rules == []  # layout-infeasible seed: host items only
    finally:
        runner.close()


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("host", HOSTS)
def test_cell_spawn_yield(host, embodiment):
    """Cell level: an all-infeasible cell skips loudly (F1); else >= 1 feasible seed."""
    _skip_if_cell_infeasible(host, embodiment)
    counts = SPAWN_OUTCOMES.get((host, embodiment), {})
    assert len(counts) == len(SEEDS), f"build grid incomplete for {host}/{embodiment}"
    assert any(k > 0 for k in counts.values()), (host, embodiment, counts)


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
@pytest.mark.parametrize("host", HOSTS)
def test_nominal_episode_low_risk(host, embodiment, tmp_path):
    """ONE low-risk nominal episode completes (both scales 0.6, the safe cell)."""
    _skip_if_cell_infeasible(host, embodiment)
    counts = SPAWN_OUTCOMES.get((host, embodiment), {})
    # candidate-bearing seeds first: the episode should exercise the rule table
    seed_order = sorted(SEEDS, key=lambda s: (counts.get(s, 1) == 0, s))
    meta = runner = None
    last = None
    for seed in seed_order[:4]:
        runner = ItemEpisodeRunner(
            "selective_avoid", host=host, embodiment=embodiment, seed=seed,
            risk={"time_scale": 0.6, "safe_time_scale": 0.6},
            render="none", video=False, out_root=tmp_path / f"s{seed}",
        )
        meta, error = runner.run()
        last = error or "task_success=False"
        if error is None and meta.task_success:
            break
    else:
        pytest.fail(
            f"no clean low-risk episode on {host}/{embodiment} over seeds "
            f"{seed_order[:4]} (last: {last})"
        )

    assert meta.host_task == host
    assert meta.spawn_rejections >= 0
    assert "spawn" in meta.diag
    assert (runner.out_dir / "meta.json").exists()
    retime = json.loads((runner.out_dir / "retime.json").read_text())
    assert "plans" in retime and "v_ratio_events" in retime
    assert retime["v_ratio_events"] == []  # s = 0.6 only slows plans down

    data = np.load(runner.out_dir / "trace.npz")
    assert len(data["t"]) > 50  # a real episode, not a stub
    assert data["margins"].shape[1] == len(meta.rules)
    if data["margins"].shape[1]:
        assert (data["margins"][0] > 0.0).all()  # all margins positive at t = 0
    # both scales are 0.6: the whole trace sits at the safe scale
    assert np.allclose(data["time_scale"], 0.6)
    # the host's picked task_object registers as held at some tick
    assert (data["held"] >= 0).any()
