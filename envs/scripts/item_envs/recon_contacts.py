"""Phase-0 recon Q7: scene.get_contacts() between two specific entities."""
import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose

scene = sapien.Scene([physx.PhysxCpuSystem()])
scene.set_timestep(1 / 250.0)

# static floor box
fb = scene.create_actor_builder()
fb.add_box_collision(half_size=[1, 1, 0.02])
floor = fb.build_static(name="floor_q7")

# dynamic box resting on the floor
db = scene.create_actor_builder()
db.add_box_collision(half_size=[0.05, 0.05, 0.05])
db.set_initial_pose(Pose(p=[0, 0, 0.05 + 0.02]))  # bottom face exactly on floor top
box = db.build(name="box_q7")
comp = box.find_component_by_type(physx.PhysxRigidDynamicComponent)
comp.mass = 1.0

for _ in range(50):
    scene.step()

contacts = scene.get_contacts()
print("n contacts:", len(contacts))
c0 = contacts[0]
print("contact attrs:", [a for a in dir(c0) if not a.startswith("_")])
print("contact.bodies:", c0.bodies)
print("bodies[0] type:", type(c0.bodies[0]),
      "| attrs:", [a for a in dir(c0.bodies[0]) if not a.startswith("_")][:12])
ents = [b.entity for b in c0.bodies]
print("entity names:", [e.name for e in ents])
print("entity ids:", [e.per_scene_id for e in ents])
print("n points:", len(c0.points))
p0 = c0.points[0]
print("point attrs:", [a for a in dir(p0) if not a.startswith("_")])
for p in c0.points:
    print("  pos:", np.round(p.position, 4),
          "sep:", round(float(p.separation), 6),
          "|impulse|:", round(float(np.linalg.norm(p.impulse)), 4))

# expected normal force: m*g = 1.0*9.81 N; force = impulse/dt summed over points
dt = 1 / 250.0
tot_f = sum(float(np.linalg.norm(p.impulse)) for p in c0.points) / dt
print("summed |impulse|/dt =", round(tot_f, 3), "N  (expect ~ m*g =", 1.0 * 9.81, "N)")
print("min separation:", min(float(p.separation) for p in c0.points))

# mapping to specific entities: filter by entity identity
floor_e, box_e = floor, box
hit = [c for c in scene.get_contacts()
       if {c.bodies[0].entity.per_scene_id, c.bodies[1].entity.per_scene_id}
       == {floor_e.per_scene_id, box_e.per_scene_id}]
print("contacts between floor and box by per_scene_id:", len(hit))
