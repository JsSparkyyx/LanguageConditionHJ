"""Phase-0 recon Q10: parse URDFs for DoF, velocity limits, gripper geometry."""
import xml.etree.ElementTree as ET

def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()
    print(f"\n### {path}")
    joints = {}
    for j in root.iter("joint"):
        name = j.get("name")
        jtype = j.get("type")
        lim = j.find("limit")
        entry = {"type": jtype}
        if lim is not None:
            for a in ("lower", "upper", "effort", "velocity"):
                v = lim.get(a)
                if v is not None:
                    entry[a] = float(v)
        joints[name] = entry
    return joints, root

# ---- ur5-wsg ----
ur5_path = "/work/nvme/bifq/project/RoboTwin/assets/embodiments/ur5-wsg/ur5_wsg_gripper.urdf"
fr_path = "/work/nvme/bifq/project/RoboTwin/assets/embodiments/franka-panda/panda.urdf"

ARM_UR5 = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_FR = [f"panda_joint{i}" for i in range(1, 8)]

ju, root_u = parse_urdf(ur5_path)
jf, root_f = parse_urdf(fr_path)

print("\n== UR5 arm joints (DoF, vel limits) ==")
for n in ARM_UR5:
    print(f"  {n}: type={ju[n]['type']} vel={ju[n].get('velocity')} effort={ju[n].get('effort')} range=({ju[n].get('lower')},{ju[n].get('upper')})")

print("\n== Franka arm joints ==")
for n in ARM_FR:
    print(f"  {n}: type={jf[n]['type']} vel={jf[n].get('velocity')} effort={jf[n].get('effort')} range=({jf[n].get('lower')},{jf[n].get('upper')})")

print("\n== WSG gripper joints ==")
for n, e in ju.items():
    if "gripper" in n or "finger" in n:
        print(f"  {n}: type={e['type']} vel={e.get('velocity')} range=({e.get('lower')},{e.get('upper')})")

print("\n== Panda gripper joints ==")
for n, e in jf.items():
    if "finger" in n:
        print(f"  {n}: type={e['type']} vel={e.get('velocity')} range=({e.get('lower')},{e.get('upper')})")

# finger links geometry (collision boxes) to estimate finger length
def link_geom(root, link_names):
    for l in root.iter("link"):
        if l.get("name") in link_names:
            print(f"  link {l.get('name')}:")
            for col in l.iter("collision"):
                origin = col.find("origin")
                geo = col.find("geometry")
                o = origin.get("xyz") if origin is not None else "0 0 0"
                g = ""
                if geo is not None:
                    box = geo.find("box")
                    cyl = geo.find("cylinder")
                    sph = geo.find("sphere")
                    mesh = geo.find("mesh")
                    if box is not None:
                        g = f"box size={box.get('size')}"
                    elif cyl is not None:
                        g = f"cylinder r={cyl.get('radius')} len={cyl.get('length')}"
                    elif sph is not None:
                        g = f"sphere r={sph.get('radius')}"
                    elif mesh is not None:
                        g = f"mesh {mesh.get('filename')} scale={mesh.get('scale')}"
                print(f"    collision origin xyz={o} rpy={origin.get('rpy') if origin is not None else ''} geom: {g}")

print("\n== WSG finger link collision geometry ==")
link_geom(root_u, ["finger_left", "finger_right"])
print("\n== Panda finger link collision geometry ==")
link_geom(root_f, ["panda_leftfinger", "panda_rightfinger"])

# finger joint origins (to compute max opening = 2 * upper + offsets)
def joint_origin(root, names):
    for j in root.iter("joint"):
        if j.get("name") in names:
            origin = j.find("origin")
            axis = j.find("axis")
            print(f"  joint {j.get('name')} ({j.get('type')}): child={j.find('child').get('link')} "
                  f"origin xyz={origin.get('xyz') if origin is not None else None} "
                  f"axis={axis.get('xyz') if axis is not None else None}")

print("\n== gripper finger joint origins ==")
joint_origin(root_u, ["base_joint_gripper_left", "base_joint_gripper_right"])
joint_origin(root_f, ["panda_finger_joint1", "panda_finger_joint2"])

# count actuated (non-fixed) joints total
print("\nnon-fixed joints ur5 urdf:", sum(1 for e in ju.values() if e["type"] != "fixed"))
print("non-fixed joints panda urdf:", sum(1 for e in jf.values() if e["type"] != "fixed"))
print("all non-fixed joint names ur5:", [n for n, e in ju.items() if e["type"] != "fixed"])
print("all non-fixed joint names panda:", [n for n, e in jf.items() if e["type"] != "fixed"])
