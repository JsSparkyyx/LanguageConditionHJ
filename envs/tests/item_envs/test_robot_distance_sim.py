"""Sim tests for robot_distance (PLAN v3 §2.2): parity with CEHJ distance_info.

On place_empty_cup with the single CrossSafe obstacle spawned the way
main/train/collect.py::make_training_env does it, robot_item_distances for the
obstacle's ItemSpec must equal distance_info(env)["d_system"] to 1e-6 m, and
robot_item_forces must match _robot_obstacle_contact_force. Needs a full
RoboTwin Env (cuRobo controller) — runs on the GPU node.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sapien")

from main.item_envs._bootstrap import ensure_main_path

ensure_main_path()

from main.item_envs.robot_distance import robot_item_distances, robot_item_forces

from .sim_utils import actor_spec, make_env, obstacle_spec

pytestmark = pytest.mark.sim

EMBODIMENTS = ["ur5-wsg", "franka-panda"]


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_robot_item_distances_matches_d_system(embodiment):
    env = make_env(embodiment)
    spec = obstacle_spec(env)
    from main.envs.distance import distance_info

    d = robot_item_distances(env, [spec])
    assert d.shape == (1,)
    assert d[0] == pytest.approx(distance_info(env)["d_system"], abs=1e-6)


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_robot_item_forces_matches_cehj_pattern(embodiment):
    env = make_env(embodiment)
    spec = obstacle_spec(env)
    from main.envs.distance import _robot_obstacle_contact_force

    f = robot_item_forces(env, [spec])
    f_ref, _touched = _robot_obstacle_contact_force(env)
    assert f.shape == (1,)
    assert f[0] == pytest.approx(f_ref, abs=1e-6)


@pytest.mark.parametrize("embodiment", EMBODIMENTS)
def test_held_item_is_inf_and_payload_for_others(embodiment):
    """End-to-end held rule: a grasped item reads +inf and joins the payload set.

    Script: open the right gripper, teleport the cup so its OBB center lands
    exactly on the right TCP, close the gripper, and step physics until the
    per_scene_id contact scan sees the grasp. Then the cup's own distance
    must be +inf, and the obstacle's distance must equal
    distance_info(env)["d_system"] to 1e-6 m — distance_info includes the
    held cup through _actor_obb_distance, so equality during the carry proves
    both the fixed scan and spec-OBB payload parity.
    """
    import sapien

    from main.envs import distance as cehj_distance
    from main.item_envs.items import entity_of
    from main.item_envs.robot_distance import _contact_held_item_indices

    env = make_env(embodiment)
    cup = env.task.cup  # place_empty_cup grasp actor (TASK_SPECS["object"])
    cup_spec = actor_spec(env, cup, "cup", "cup", role="task_object")  # the task's own pick object
    obs_spec = obstacle_spec(env)
    specs = [cup_spec, obs_spec]

    # Pre-grasp (cup on the table, gripper open): nothing held, both finite.
    d = robot_item_distances(env, specs)
    print(f"[held-test] {embodiment} pre-grasp:  d_cup={d[0]:.6f} d_obstacle={d[1]:.6f}")
    assert np.isfinite(d[0])
    assert np.isfinite(d[1])

    # Scripted grasp.
    env.robot.set_gripper(1, "right")  # 1 = fully open
    for _ in range(50):
        env.task.scene.step()
    ent = entity_of(cup)
    pose = ent.get_pose()
    rot = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)[:3, :3]
    tcp = np.asarray(env.robot.get_right_tcp_pose(), dtype=np.float64).reshape(-1)[:3]
    ent.set_pose(sapien.Pose(tcp - rot @ cup_spec.obb_center_local, pose.q))
    env.robot.set_gripper(0, "right")  # 0 = closed; _arm_is_holding reads the commanded value
    held = set()
    steps = 0
    for steps in range(1, 401):
        env.task.scene.step()
        held = _contact_held_item_indices(env, specs, cehj_distance)
        if 0 in held:
            break
    assert 0 in held, f"{embodiment}: contact scan never saw the grasped cup"

    d = robot_item_distances(env, specs)
    d_sys = cehj_distance.distance_info(env)["d_system"]
    print(
        f"[held-test] {embodiment} grasp@{steps}: d_cup={d[0]} d_obstacle={d[1]:.6f} "
        f"d_system={d_sys:.6f} |d_obstacle-d_system|={abs(d[1] - d_sys):.3e}"
    )
    assert d[0] == np.inf  # the held task object's own distance
    assert np.isfinite(d[1])
    assert d[1] == pytest.approx(d_sys, abs=1e-6)
    # the same grasp on a CANDIDATE (a rule-protected item) reads the contact,
    # never +inf (audit 2026-10-04: a T1 candidate caught by the closing
    # fingers read "safe" while held)
    cand = actor_spec(env, cup, "cup", "cup", role="candidate")
    d_c = robot_item_distances(env, [cand, obs_spec])
    assert np.isfinite(d_c[0]) and d_c[0] < 0.0, d_c  # a held candidate violates "avoid"
