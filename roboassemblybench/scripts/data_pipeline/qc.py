"""QC of full-process AutoMate raw episodes (2026-10-03) against the data contract and the evaluator's requirements
(the evaluator diversity requirements). Usage: python qc.py <dataset root> [--max N] [--profile mixed]

Per episode: required metadata present (camera models / intrinsics, calibrations with world extrinsics for the front
camera, environment record, layout, generation ids, success / target definitions, attachment assist, phases, home
pose, units); trajectory arrays consistent; action = next frame's absolute EE pose; gripper commands in {open,
training opening, closed}; phases in order 0..6; scripted phases not stalled; policy action zero outside the insert phase; first and last frame at
the fixed home pose; plug seated at the end; video frame counts = trajectory length; camera intrinsics = D435 / D405.
Across episodes: unique base_trajectory_id / layout seeds, home pose identical, intrinsics identical."""
import collections as C
import glob
import json
import os
import subprocess
import sys

import numpy as np

root = sys.argv[1]
maxn = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 10 ** 9
profile = sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else "mixed"
eps = sorted(glob.glob(f"{root}/*/{profile}/episode_*_cartesian_raw"))[:maxn]
REQ_META = ["cameras", "camera_calibrations", "environment", "layout", "generation", "success_definition", "target_reached_definition",
            "attachment_assist", "phases", "home_pose_fingertip_env", "conventions", "feature_units", "gripper_definition", "control",
            "T_world_env", "visual_settle_renders", "checkpoint", "software", "batch_id", "robot_type", "main_instruction"]
EXPECT_K = {"observation.images.front": (616.7, 616.7, 320.0, 240.0), "observation.images.wrist": (435.0, 435.0, 320.0, 240.0)}
problems = C.Counter()
examples = C.defaultdict(list)
stats = C.defaultdict(list)
seen_base, seen_seed = C.Counter(), C.Counter()
home_first = []


def bad(kind, ep, msg=""):
    problems[kind] += 1
    if len(examples[kind]) < 3:
        examples[kind].append(f"{os.path.relpath(ep, root)} {msg}")


def nframes(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


for ep in eps:
    try:
        meta = json.load(open(f"{ep}/metadata.json"))
        z = np.load(f"{ep}/trajectory.npz")
    except Exception as e:  # noqa: BLE001
        bad("unreadable", ep, repr(e))
        continue
    md = meta.get("metadata", meta)
    for k in REQ_META:
        if k not in md:
            bad(f"missing_meta:{k}", ep)
    st, act, pa, ph = z["observation_state"], z["action"], z["policy_action"], z["phase_index"]
    T = len(st)
    stats["frames"].append(T)
    if not (act.shape == (T, 8) and pa.shape == (T, 6) and ph.shape == (T,)):
        bad("shapes", ep, f"{st.shape} {act.shape} {pa.shape} {ph.shape}")
    # action[t] = state[t+1] (EE pose), gripper action in {1, training opening, 0}
    d = np.abs(act[:-1, :3] - st[1:, :3]).max()
    if d > 1e-5:
        bad("action_not_next_state", ep, f"max {d:.2e}")
    g = np.unique(np.round(act[:, 7], 3))
    stats["gripper_values"].append(tuple(g.tolist()))
    if len(g) < 2 or g.min() < 0 or g.max() > 1:
        bad("gripper_cmd_degenerate", ep, str(g))
    # phases 0..6 in order, all present
    if np.any(np.diff(ph) < 0) or set(np.unique(ph).tolist()) != set(range(7)):
        bad("phase_order", ep, str(np.unique(ph).tolist()))
    # the gripper must not press on the table to reach the grasp: upward finger contact force (left + right, world z)
    # during approach / grasp <= 3 N (collector FINGER_TABLE_FORCE_N)
    cf = z["contact_force"] if "contact_force" in z.files else None
    if cf is not None and ((ph == 0) | (ph == 1)).any():
        up = (np.maximum(cf[:, 2], 0) + np.maximum(cf[:, 5], 0))[(ph == 0) | (ph == 1)].max()
        stats["finger_up_force_n"].append(float(up))
        if up > 3.0:
            bad("finger_pressed_table", ep, f"{up:.1f} N")
    ins = ph == 4
    if np.abs(pa[~ins]).max() > 0:
        bad("policy_action_outside_insert", ep)
    if ins.any() and np.abs(pa[ins]).max() == 0:
        bad("policy_action_zero_in_insert", ep)
    stats["insert_frames"].append(int(ins.sum()))
    pf = np.bincount(ph, minlength=7)
    stats["phase_frames"].append(pf.tolist())
    # scripted phases run in lock step: a phase far longer than its normal length means the envs stood still waiting
    # (2026-10-04: one unreachable waypoint held a round 20 s per waypoint); normal approach ~230-300, lift ~50-56,
    # carry ~150-200 frames
    if pf[0] > 450 or pf[2] > 150 or pf[3] > 400:
        bad("scripted_phase_too_long", ep, f"approach {pf[0]} lift {pf[2]} carry {pf[3]}")
    # home pose at the first and last frame
    hp = np.array(md.get("home_pose_fingertip_env", {}).get("position_m", [np.nan] * 3))
    e0, e1 = np.linalg.norm(st[0, :3] - hp), np.linalg.norm(st[-1, :3] - hp)
    stats["home_err_first_mm"].append(e0 * 1000)
    stats["home_err_last_mm"].append(e1 * 1000)
    if e0 > 0.005 or e1 > 0.006:
        bad("not_at_home", ep, f"first {e0 * 1000:.1f} last {e1 * 1000:.1f} mm")
    home_first.append(hp)
    if st[-1, 7] < 0.8:
        bad("gripper_not_open_at_end", ep, f"{st[-1, 7]:.2f}")
    kp = z["keypoint_error"]
    mt = meta.get("metrics", md.get("metrics", {}))
    if mt.get("success_rule") == "deeper_ok":
        # deeper-than-goal success (COLLECT_DEEPER_OK, 2026-10-09): at most 3 mm out along the socket axis, any depth in,
        # at most 3 mm lateral, at most 5 deg tilt
        sf = mt.get("seat_final") or {}
        stats["deeper_ok_axial_mm"].append(1000 * sf.get("axial_m", float("nan")))
        if not (sf.get("axial_m", 1) <= 0.003 and sf.get("lateral_m", 1) <= 0.003 and sf.get("tilt_deg", 99) <= 5.0):
            bad("deeper_ok_out_of_bounds", ep, str(sf))
    elif kp[-1] >= 0.003:
        bad("not_seated_at_end", ep, f"{kp[-1] * 1000:.2f} mm")
    stats["kp_end_mm"].append(float(kp[-1] * 1000))
    # videos and depth
    for cam in ("front", "wrist"):
        v = f"{ep}/videos/observation_images_{cam}.mp4"
        nf = nframes(v)
        if nf != T:
            bad(f"video_frames_{cam}", ep, f"{nf} vs {T}")
    # intrinsics
    cams = md.get("cameras", {})
    for key, (fx, fy, cx, cy) in EXPECT_K.items():
        K = np.array(cams.get(key, {}).get("intrinsics", np.zeros((3, 3))))
        if not np.allclose([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], [fx, fy, cx, cy], atol=0.5):
            bad(f"intrinsics_{key.split('.')[-1]}", ep, f"{K[0, 0]:.1f} {K[1, 1]:.1f} {K[0, 2]:.1f} {K[1, 2]:.1f}")
        if "camera_model" not in cams.get(key, {}):
            bad("camera_model_missing", ep)
    cal = md.get("camera_calibrations", {})
    f_cal, w_cal = cal.get("observation.images.front", {}), cal.get("observation.images.wrist", {})
    if "T_world_camera" not in f_cal:
        bad("front_world_extrinsics_missing", ep)
    if w_cal.get("parent_frame") != "panda_fingertip_centered":
        bad("wrist_parent_frame", ep, str(w_cal.get("parent_frame")))
    stats["wrist_rigidity_um"].append(1e6 * float(w_cal.get("max_translation_deviation_over_episode_m", np.nan)))
    env = md.get("environment", {})
    for k in ("lights", "table", "ground", "distractors", "scene_props", "randomization"):
        if k not in env:
            bad(f"environment_missing:{k}", ep)
    if "optical_board" in env:
        bad("environment_unexpected:optical_board", ep)
    gen = md.get("generation", {})
    seen_base[gen.get("base_trajectory_id")] += 1
    seen_seed[(md.get("assembly_id"), md.get("layout_seed"))] += 1
    lay = md.get("layout", {})
    for k in ("socket_pose_env", "plug_pose_env", "plug_pose_socket_frame", "plug_socket_distance_m"):
        if k not in lay:
            bad(f"layout_missing:{k}", ep)
    if "plug_pose_env" in lay and "socket_pose_env" in lay:
        pxy, sxy = np.array(lay["plug_pose_env"][:2]), np.array(lay["socket_pose_env"][:2])
        stats["plug_x"].append(pxy[0]); stats["plug_y"].append(pxy[1]); stats["socket_x"].append(sxy[0]); stats["socket_y"].append(sxy[1])
        stats["plug_socket_dist"].append(float(np.linalg.norm(pxy - sxy)))
        if np.linalg.norm(pxy - sxy) < 0.149 or not (0.399 <= pxy[0] <= 0.751 and abs(pxy[1]) <= 0.321):
            bad("plug_outside_region", ep, f"{pxy} vs socket {sxy}")
    stats["settle_renders"].append(md.get("visual_settle_renders"))
    m = meta.get("metrics", md.get("metrics", {}))
    if m.get("attachment_assist_used") is not False:
        bad("attachment_assist_not_false", ep)

dup_base = sum(1 for v in seen_base.values() if v > 1)
dup_seed = sum(1 for v in seen_seed.values() if v > 1)
hf = np.array(home_first) if home_first else np.zeros((0, 3))
print(f"episodes checked: {len(eps)} | tasks: {len({os.path.relpath(e, root).split('/')[0] for e in eps})}")
print(f"duplicate base_trajectory_id: {dup_base} | duplicate (task, layout_seed): {dup_seed} | home pose spread across episodes: "
      f"{(hf.max(0) - hf.min(0)).max() * 1000 if len(hf) else float('nan'):.3f} mm")
for k in ("frames", "insert_frames", "home_err_first_mm", "home_err_last_mm", "kp_end_mm", "wrist_rigidity_um",
          "plug_x", "plug_y", "socket_x", "socket_y", "plug_socket_dist", "finger_up_force_n", "deeper_ok_axial_mm"):
    v = np.array([x for x in stats[k] if x is not None], dtype=float)
    if len(v):
        print(f"  {k:20s} p50 {np.median(v):9.3f}  min {v.min():9.3f}  max {v.max():9.3f}")
if stats["phase_frames"]:
    pf = np.array(stats["phase_frames"])
    print("  phase frames p50:", dict(zip(("approach", "grasp", "lift", "carry", "insert", "release", "return"), np.median(pf, 0).astype(int).tolist())))
print("  gripper command values seen:", sorted(C.Counter(stats["gripper_values"]).items(), key=lambda x: -x[1])[:3])
print("  settle renders:", sorted(C.Counter(stats["settle_renders"]).items())[:6])
print("problems:" if problems else "problems: none")
for k, v in sorted(problems.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}  e.g. {examples[k]}")
