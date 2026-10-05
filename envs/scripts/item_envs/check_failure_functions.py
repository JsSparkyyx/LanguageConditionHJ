"""check_failure_functions.py <root>... — independent recomputation of EVERY rule type from raw simulator state.

Per rule, from the trace's raw poses / particle positions (never from the margin code):
  upright   theta_max - tilt(up axis)                         -> value + sign agreement
  stay      min(d_tol - |xy - xy(t0)|, d_tol (theta_tol - tilt) / theta_tol)  -> value + sign agreement
  inside /  outside: item OBB corners (8) in the region frame vs the region polygon -> sign agreement
  particles column test per bead (capped columns honoured)    -> sign agreement
  avoid     vs the PhysX contact force (the only independent signal):
            P(margin <= 5 mm | contact force > 0)  and  P(contact within 0.2 s | margin < -1 cm)
"""
import glob, json, sys
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, "/work/nvme/bifq/project/LanguageConditionHJ/envs")
from main.item_envs.geometry import signed_dist_inside_polygon  # pure polygon math

def tilt_of(q_wxyz, up_local):
    up = R.from_quat(q_wxyz[:, [1, 2, 3, 0]]).apply(np.asarray(up_local, float))
    return np.arccos(np.clip(up[:, 2], -1, 1))

def obb_corners(pose, half, center_local):
    """[T, 8, 3] world corners of the item's OBB."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    local = np.asarray(center_local, float)[None] + signs * np.asarray(half, float)[None]
    rot = R.from_quat(pose[:, [4, 5, 6, 3]])
    return np.stack([rot.apply(c) for c in local], axis=1) + pose[:, None, :3]

stats = defaultdict(lambda: {"n": 0, "sign_agree": [], "val_err": [], "eps": 0})
avoid_c = {"contact_ticks": 0, "contact_with_margin_le_5mm": 0, "deep_neg_ticks": 0, "deep_neg_with_contact_near": 0}
bad = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(root + "/**/seed_*/meta.json", recursive=True)):
        M = json.load(open(f))
        if M.get("error"): continue
        t = np.load(f.replace("meta.json", "trace.npz"), allow_pickle=True)
        rule_ids = [str(x) for x in t["rule_ids"]]; item_ids = [str(x) for x in t["item_ids"]]
        m = np.asarray(t["margins"], float); P = np.asarray(t["item_pose"], float)
        rules = {r["rule_id"]: r for r in M["rules"]}; items = {i["item_id"]: i for i in M["items"]}
        regs = [r["region_id"] for r in M["regions"]]; regm = {r["region_id"]: r for r in M["regions"]}
        RT = np.asarray(t["region_T"], float); ts = np.asarray(t["t"], float)
        for k, rid in enumerate(rule_ids):
            r = rules[rid]; cons = r["consequence"]; prm = r["params"]; ind = None; val = None
            if cons == "upright":
                vals = [prm["theta_max"] - tilt_of(P[:, item_ids.index(i), 3:7], items[i]["up_axis_local"]) for i in r["item_ids"]]
                val = np.min(vals, axis=0)
            elif cons == "stay":
                vals = []
                for i in r["item_ids"]:
                    j = item_ids.index(i); d = np.linalg.norm(P[:, j, :2] - P[0, j, :2], axis=1)
                    tilt = tilt_of(P[:, j, 3:7], items[i]["up_axis_local"])
                    vals.append(np.minimum(prm["d_tol"] - d, prm["d_tol"] * (prm["theta_tol"] - tilt) / prm["theta_tol"]))  # metres (2026-10-05)
                val = np.min(vals, axis=0)
            elif cons in ("inside", "outside"):
                per_region = []
                for reg in r["region_ids"]:
                    kr = regs.index(reg); poly = np.asarray(regm[reg]["polygon"], float); Tr = RT[:, kr]
                    per_item = []
                    for i in r["item_ids"]:
                        it = items[i]; C = obb_corners(P[:, item_ids.index(i)], it["half_extents"], it.get("obb_center_local") or [0, 0, 0])
                        loc = np.einsum("tij,tnj->tni", np.transpose(Tr[:, :3, :3], (0, 2, 1)), C - Tr[:, None, :3, 3])
                        sd = np.stack([signed_dist_inside_polygon(loc[q, :, :2], poly) for q in range(len(ts))])
                        per_item.append(sd.min(axis=1) if cons == "inside" else -sd.max(axis=1))
                    per_region.append(np.min(per_item, axis=0))
                val = np.max(per_region, axis=0) if cons == "inside" else np.min(per_region, axis=0)
                ind = val < 0
                val = None  # corners vs footprint: compare the sign only
            elif cons in ("particles_inside", "particles_outside"):
                key = "particle_pos__" + r["item_ids"][0]
                if key not in t.files: continue
                Pp = np.asarray(t[key], float); per = []
                for reg in r["region_ids"]:
                    kr = regs.index(reg); Tr = RT[:, kr]
                    loc = np.einsum("tij,tnj->tni", np.transpose(Tr[:, :3, :3], (0, 2, 1)), Pp - Tr[:, None, :3, 3])
                    sd = np.stack([signed_dist_inside_polygon(loc[q, :, :2], np.asarray(regm[reg]["polygon"], float)) for q in range(len(ts))])
                    zc = regm[reg].get("z_cap")
                    if zc is not None: sd = np.minimum(sd, float(zc) - loc[:, :, 2])
                    per.append(sd)
                per = np.stack(per)
                best = per.max(axis=0) if cons == "particles_inside" else (-per).min(axis=0)
                ind = (best < 0).any(axis=1)
            elif cons == "avoid":
                js = [item_ids.index(i) for i in r["item_ids"]]
                force = np.asarray(t["robot_item_force"], float)[:, js].max(axis=1)
                c = force > 1e-3
                avoid_c["contact_ticks"] += int(c.sum()); avoid_c["contact_with_margin_le_5mm"] += int((c & (m[:, k] <= 0.005)).sum())
                deep = np.where(m[:, k] < -0.01)[0]
                near = np.array([c[max(0, q - 5):q + 6].any() for q in deep], bool)
                avoid_c["deep_neg_ticks"] += len(deep); avoid_c["deep_neg_with_contact_near"] += int(near.sum())
                continue
            else:
                continue
            if ind is None: ind = val < 0
            s = stats[(M["task"], cons)]; s["n"] += len(ts); s["eps"] += 1
            agree = float(((m[:, k] < 0) == ind).mean()); s["sign_agree"].append(agree)
            if val is not None: s["val_err"].append(float(np.abs(m[:, k] - val).max()))
            if agree < 0.99: bad.append((M["task"], M["embodiment"][:5], M["seed"], rid, round(agree, 4)))
print("task             rule type          episodes  sign agreement (mean / worst episode)   max value error")
for (task, cons), s in sorted(stats.items()):
    ve = f"{max(s['val_err']):.2e}" if s["val_err"] else "(sign only)"
    print(f"{task:16s} {cons:18s} {s['eps']:8d}   {np.mean(s['sign_agree']):.4f} / {min(s['sign_agree']):.4f}                      {ve}")
a = avoid_c
print(f"avoid vs PhysX contact: when the robot touches the object (force > 0), margin <= 5 mm on {a['contact_with_margin_le_5mm']}/{a['contact_ticks']} ticks; "
      f"when margin < -1 cm, a contact within 0.2 s on {a['deep_neg_with_contact_near']}/{a['deep_neg_ticks']} ticks")
print("episodes with < 99 % sign agreement on a rule:", len(bad))
for b in bad[:15]: print("  ", b)
