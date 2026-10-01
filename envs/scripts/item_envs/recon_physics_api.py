"""Phase-0 recon: sapien 3.0.0b1 physics API probes (CPU, no GPU).

Covers PLAN Phase-0 questions 1-6:
  Q1 physical material on EXISTING collision shapes + build-time material=
  Q2 linear_damping / angular_damping set + readback
  Q3 center-of-mass local pose set + readback
  Q4 CCD exposure on dynamic bodies
  Q5 kinematic body + set_kinematic_target, pose follows over 10 steps
  Q6 solver iteration counts (scene-level / per-body)
"""
import inspect

import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose

print("sapien version:", sapien.__version__)

def make_scene():
    # physics-only: no RenderSystem (this node has no GPU/Vulkan)
    return sapien.Scene([physx.PhysxCpuSystem()])

# ---------------------------------------------------------------- Q1: material
print("\n=== Q1: physical material ===")
scene = make_scene()
scene.set_timestep(1 / 250.0)

builder = scene.create_actor_builder()
mat_sig = None
try:
    mat_sig = str(inspect.signature(builder.add_box_collision))
except Exception as e:
    print("add_box_collision signature error:", e)
print("add_box_collision sig:", mat_sig)
print("add_multiple_convex_collisions_from_file sig:",
      inspect.signature(builder.add_multiple_convex_collisions_from_file))
print("add_convex_collision_from_file sig:",
      inspect.signature(builder.add_convex_collision_from_file))

# build a dynamic box, material passed at build time
b = scene.create_actor_builder()
physys = scene.get_physx_system()
print("scene has create_physical_material:", hasattr(scene, "create_physical_material"),
      "| physx system has it:", hasattr(physys, "create_physical_material"))
try:
    m_build = physys.create_physical_material(0.9, 0.8, 0.05)
    print("physx_system.create_physical_material(0.9,0.8,0.05) OK")
except Exception as e:
    print("physx system material failed, falling back to scene:", e)
    m_build = scene.create_physical_material(0.9, 0.8, 0.05)
b.add_box_collision(half_size=[0.05, 0.05, 0.05], material=m_build)
box = b.build(name="box_q1")
comp = box.find_component_by_type(physx.PhysxRigidDynamicComponent)
print("rigid dynamic component:", comp is not None)

shapes = comp.get_collision_shapes()
print("get_collision_shapes() ->", type(shapes), len(shapes))
sh = shapes[0]
print("shape type:", type(sh))
print("shape.physical_material readable:",
      sh.physical_material.static_friction, sh.physical_material.dynamic_friction,
      sh.physical_material.restitution, "(expect 0.9 0.8 0.05 from build-time material=)")

# now try to CHANGE the material on the existing shape
m_new = scene.create_physical_material(0.11, 0.12, 0.13)
setter_works = None
try:
    sh.physical_material = m_new
    setter_works = True
except Exception as e:
    setter_works = f"FAILED: {type(e).__name__}: {e}"
print("shape.physical_material = m_new ->", setter_works)
rb = shapes[0].physical_material
print("readback after setter:", rb.static_friction, rb.dynamic_friction, rb.restitution,
      "(expect 0.11 0.12 0.13)")

has_setter_fn = hasattr(sh, "set_physical_material")
print("has set_physical_material method:", has_setter_fn)

# does the change survive / take effect? check the default material of a fresh build
b2 = scene.create_actor_builder()
b2.add_box_collision(half_size=[0.05, 0.05, 0.05])   # no material -> scene default
box2 = b2.build(name="box_q1b")
sh2 = box2.find_component_by_type(physx.PhysxRigidDynamicComponent).get_collision_shapes()[0]
dm = sh2.physical_material
print("default scene material on box2:", dm.static_friction, dm.dynamic_friction, dm.restitution)
print("physx system attrs with 'material':", [a for a in dir(scene.get_physx_system()) if "material" in a.lower()])

# ------------------------------------------------- Q2/Q3: damping + CoM pose
print("\n=== Q2: damping ===")
print("has linear_damping:", hasattr(comp, "linear_damping"),
      "| has angular_damping:", hasattr(comp, "angular_damping"))
comp.linear_damping = 0.35
comp.angular_damping = 0.45
print("readback:", comp.linear_damping, comp.angular_damping, "(expect 0.35 0.45)")
print("defaults on box2:",
      box2.find_component_by_type(physx.PhysxRigidDynamicComponent).linear_damping,
      box2.find_component_by_type(physx.PhysxRigidDynamicComponent).angular_damping)

print("\n=== Q3: center-of-mass local pose ===")
cmass_attrs = [a for a in dir(comp) if "cmass" in a.lower() or "com" in a.lower()]
print("cmass-ish attrs on component:", cmass_attrs)
print("cmass_local_pose (before):", comp.cmass_local_pose)
new_cm = Pose(p=[0.01, -0.02, 0.03], q=[1, 0, 0, 0])
comp.cmass_local_pose = new_cm
rb = comp.cmass_local_pose
print("readback p:", rb.p, "q:", rb.q, "(expect [0.01 -0.02 0.03] [1 0 0 0])")
comp.mass = 0.42
print("mass readback:", comp.mass)

print("\n=== Q4: CCD ===")
ccd_attrs = [a for a in dir(physx.PhysxRigidDynamicComponent) if "ccd" in a.lower()]
print("ccd attrs on PhysxRigidDynamicComponent:", ccd_attrs)
physx_ccd = [a for a in dir(physx) if "ccd" in a.lower()]
print("ccd names in sapien.physx module:", physx_ccd)
scene_ccd = [a for a in dir(scene) if "ccd" in a.lower()]
print("ccd names on Scene:", scene_ccd)
sys_ccd = [a for a in dir(scene.get_physx_system()) if "ccd" in a.lower()]
print("ccd names on physx system:", sys_ccd)
# scene-level CCD: PhysxSceneConfig.enable_ccd exists, but the config returned by
# get_config() is a COPY (in-place set does not stick) and PhysxCpuSystem takes no
# config argument. The only working path is the process-global default:
physx.set_scene_config(enable_ccd=True)
scene4b = make_scene()
print("new scene enable_ccd after physx.set_scene_config:", scene4b.get_physx_system().get_config().enable_ccd)
physx.set_scene_config(enable_ccd=False)  # restore default

print("\n=== Q5: kinematic body ===")
scene5 = make_scene()
scene5.set_timestep(1 / 250.0)
kb = scene5.create_actor_builder()
kb.add_box_collision(half_size=[0.05, 0.05, 0.05])
kb.set_initial_pose(Pose(p=[0, 0, 0.5]))
kin_box = kb.build_kinematic(name="kin_box_q5")   # also test plain build + kinematic flag below
kcomp = kin_box.find_component_by_type(physx.PhysxRigidDynamicComponent)
print("built kinematic:", kcomp.kinematic)

# second body: plain build then component.kinematic = True
kb2 = scene5.create_actor_builder()
kb2.add_box_collision(half_size=[0.05, 0.05, 0.05])
kb2.set_initial_pose(Pose(p=[0.5, 0, 0.5]))
kin_box2 = kb2.build(name="kin_box2_q5")
kcomp2 = kin_box2.find_component_by_type(physx.PhysxRigidDynamicComponent)
kcomp2.kinematic = True
print("post-build kinematic=True readback:", kcomp2.kinematic)

# move the target over 10 steps, +x at 0.1 m per step, and check the pose follows
poses = []
for i in range(10):
    target = Pose(p=[0.1 * (i + 1), 0, 0.5])
    kcomp.set_kinematic_target(target)
    scene5.step()
    poses.append(kin_box.get_pose().p.copy())
poses = np.asarray(poses)
print("kinematic target following (x per step):")
print(np.round(poses, 4))
expected = 0.1 * np.arange(1, 11)
err = np.abs(poses[:, 0] - expected).max()
print("max |x - target| over 10 steps:", err, "| z drift:", np.abs(poses[:, 2] - 0.5).max())

print("\n=== Q6: solver iteration counts ===")
print("Scene constructor sig:", inspect.signature(sapien.Scene.__init__) if hasattr(sapien.Scene, "__init__") else "n/a")
scene_attrs = [a for a in dir(scene) if "iter" in a.lower() or "solver" in a.lower()]
print("scene attrs matching iter/solver:", scene_attrs)
syscfg = scene.get_physx_system().get_config() if hasattr(scene.get_physx_system(), "get_config") else None
print("physx system type:", type(scene.get_physx_system()))
sys_attrs = [a for a in dir(scene.get_physx_system()) if not a.startswith("_")]
print("physx system attrs:", sys_attrs)
comp_attrs = [a for a in dir(comp) if "iter" in a.lower() or "solver" in a.lower()]
print("component attrs matching iter/solver:", comp_attrs)
# PhysxSceneConfig
cfg_cls = getattr(physx, "PhysxSceneConfig", None)
print("PhysxSceneConfig:", cfg_cls)
if cfg_cls is not None:
    cfg = cfg_cls()
    cfg_attrs = [a for a in dir(cfg) if not a.startswith("_")]
    print("PhysxSceneConfig attrs:", cfg_attrs)
    for a in cfg_attrs:
        if "iter" in a.lower():
            print(f"  {a} default:", getattr(cfg, a))
            setattr(cfg, a, 8)
            print(f"  {a} after set 8:", getattr(cfg, a))
# does the cpu system accept a config?
try:
    print("PhysxCpuSystem sig:", inspect.signature(physx.PhysxCpuSystem.__init__))
except Exception as e:
    print("PhysxCpuSystem sig error:", e)
try:
    sys6 = physx.PhysxCpuSystem(physx.PhysxSceneConfig())
    print("PhysxCpuSystem(config) OK")
except Exception as e:
    print("PhysxCpuSystem(config) failed:", type(e).__name__, e)
# per-body solver iterations
print("per-body defaults:", comp.solver_position_iterations, comp.solver_velocity_iterations)
comp.solver_position_iterations = 8
comp.solver_velocity_iterations = 4
print("per-body readback after set (8,4):",
      comp.solver_position_iterations, comp.solver_velocity_iterations)
# global default solver iters via set_body_config
bcfg = physx.PhysxBodyConfig()
print("PhysxBodyConfig defaults:", bcfg.solver_position_iterations, bcfg.solver_velocity_iterations)
