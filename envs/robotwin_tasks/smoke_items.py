"""smoke_items — TEST-ONLY Phase 2 acceptance task (PLAN v3 §6 Phase 2).

One green zone mat and two colored candidate boxes on the table. The
nominal grasps the red box with the left arm, lifts it, moves it 10 cm
in +x inside a ``with risky(self):`` block (the retimed segment), and
places it back; success is the red box landing within 3 cm of the
sampled target offset. The blue box and the zone mat exist so the rule
table has several candidates of the same rule type — an ``avoid`` rule
on the grasped box WILL violate at grasp contact, which is intended:
it exercises the runner's first-violation keyframe and red flash.

This file is installed into ``RoboTwin/envs/`` by symlink
(``scripts/item_envs/install_robotwin_tasks.sh``) and must not import
any CEHJ ``main.*`` module — the metadata dicts written by
``_item_utils.register_item`` / ``register_zone`` (``task.item_meta`` /
``task.region_meta``) are the only interface the CEHJ-side runner
reads. Layout uses the global numpy RNG, seeded by RoboTwin's
``_init_task_env_`` (PLAN v3 §2 determinism rule).
"""

from ._base_task import Base_Task
from .utils import *
import sapien

from ._item_utils import register_item, register_zone, risky, zone_mat

TABLE_TOP = 0.74  # RoboTwin table height; + self.table_z_bias at runtime

BOX0_HALF = [0.025, 0.025, 0.035]  # red box half extents (m)
BOX1_HALF = [0.030, 0.020, 0.025]  # blue box half extents
ZONE_SIZE_XY = (0.14, 0.12)  # green zone mat full extent

MOVE_DX = 0.10  # the risky translation (+x, world frame)
LIFT_DZ = 0.08
RETREAT_DZ = 0.06
SUCCESS_TOL = 0.03  # box_0 xy within 3 cm of start + [MOVE_DX, 0]


class smoke_items(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        z = TABLE_TOP + self.table_z_bias
        # Global RNG (seeded by _init_task_env_): one draw per dof, fixed order.
        x0 = float(np.random.uniform(-0.18, -0.08))
        y0 = float(np.random.uniform(-0.05, 0.05))
        x1 = float(np.random.uniform(0.10, 0.22))
        y1 = float(np.random.uniform(-0.05, 0.05))
        zx = float(np.random.uniform(-0.05, 0.15))
        zy = float(np.random.uniform(0.18, 0.28))

        self.box_0 = create_box(
            scene=self,
            pose=sapien.Pose([x0, y0, z + BOX0_HALF[2]]),
            half_size=BOX0_HALF,
            color=(1.0, 0.0, 0.0),
            name="box_0",
        )
        self.box_1 = create_box(
            scene=self,
            pose=sapien.Pose([x1, y1, z + BOX1_HALF[2]]),
            half_size=BOX1_HALF,
            color=(0.0, 0.0, 1.0),
            name="box_1",
        )
        self.zone = zone_mat(
            self.scene,
            sapien.Pose([zx, zy, z + 0.001]),
            size_xy=ZONE_SIZE_XY,
            color=(0.0, 0.8, 0.0),
            name="zone_green",
        )

        register_item(
            self, "box_0", self.box_0,
            category="box", color="red", kind="rigid", role="candidate",
            half_extents=BOX0_HALF, mass=0.10,
        )
        register_item(
            self, "box_1", self.box_1,
            category="box", color="blue", kind="rigid", role="candidate",
            half_extents=BOX1_HALF, mass=0.08,
        )
        register_zone(
            self, "zone_green", self.zone,
            color="green", size_xy=ZONE_SIZE_XY,
        )

    def play_once(self):
        start_xy = np.asarray(self.box_0.get_pose().p[:2], dtype=np.float64).copy()
        self.target_xy = start_xy + np.array([MOVE_DX, 0.0])

        arm_tag = ArmTag("left")
        self.move(self.grasp_actor(self.box_0, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=LIFT_DZ))
        with risky(self):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=MOVE_DX))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-LIFT_DZ))
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=RETREAT_DZ))

        self.info["info"] = {"{A}": "red box", "{B}": "blue box", "{Z}": "green zone"}
        return self.info

    def check_success(self):
        box_xy = np.asarray(self.box_0.get_pose().p[:2], dtype=np.float64)
        return bool(
            getattr(self, "target_xy", None) is not None
            and np.linalg.norm(box_xy - self.target_xy) < SUCCESS_TOL
        )
