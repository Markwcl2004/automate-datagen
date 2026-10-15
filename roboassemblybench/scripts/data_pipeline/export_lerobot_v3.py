"""Export full-process AutoMate raw episodes (one task / profile directory) to a LeRobot v3 dataset with the
ourdata-diversity-1.1 sidecars in meta/diversity (2026-10-09).

Usage: export_automate_lerobot_v3.py --raw <dataset root>/<task>/<profile> --out <new dir> --dataset-id <id>
       [--max-episodes N] [--episodes-per-file 50] [--visual-randomization <json>]

Layout written (mirrors the evaluator's RoboTwin 1.1 sample):
  data/chunk-XXX/file-YYY.parquet          one row per frame (state, action, extra features)
  videos/<camera key>/chunk-XXX/file-YYY.mp4  the episodes' raw H.264 streams concatenated without re-encoding
  sensors/depth/episode_XXXXXX/            the raw chunked depth files and their index, per episode
  annotations/episode_XXXXXX.json          the raw annotation file, per episode
  meta/info.json, meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet, meta/stats.json
  meta/diversity/{schema.json, calibrations.jsonl, environments.jsonl, episodes.jsonl, camera_poses/chunk-000/file-000.parquet}

Avoids the failure modes found in the UR5e batch: no streaming encoder (stream copy of complete per-episode files),
nothing appended to an existing output (the output directory must not exist), and a structural check at the end
(rows = sum of lengths, each episode once with contiguous indices, video frames per file = rows, first frame of every
episode in the concatenated video identical to the raw episode's first frame)."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ("observation.images.front", "observation.images.wrist")
CAM_NAMES = {"observation.images.front": "front_d435", "observation.images.wrist": "wrist_d405"}
RAW_VIDEO = {k: f"videos/{k.replace('.', '_')}.mp4" for k in CAMS}

# visual randomization version of an episode, from the sha256 of the collector environment that wrote it (metadata
# software.source_sha256): v1_20261004 = distractor type by slot parity, scene props uniform over assets (everything
# collected 2026-10-04..10-10); v2_20261010 = independent 50 / 50 type per distractor, scene-prop category first
# (code_collect_relobs_par, the 85 x 1200 expansion from 2026-10-10 15:57 CST); v3_20261011 = v2 with the lower lighting range,
# written into the metadata by the collector. Described in schema visual_randomization.
VR_V2_ENV_SHA256 = {"2fb65101e773c8524feb42bc7a30df46019264780467b3031be495d3d2dbc24f"}
VR_ENV_FILE = "roboassemblybench/library/automate/assembly_collect_env.py"


def vr_version(meta) -> str:
    if meta.get("visual_randomization_version"):  # written by the collector from v3_20261011 on
        return meta["visual_randomization_version"]
    sha = ((meta.get("software") or {}).get("source_sha256") or {}).get(VR_ENV_FILE)
    return "v2_20261010" if sha in VR_V2_ENV_SHA256 else "v1_20261004"


CHUNKS_SIZE = 1000
EXTRA_UNITS = {
    "observation.joint_state": "rad", "observation.joint_velocity": "rad/s", "observation.joint_effort": "N*m (measured joint torque)",
    "observation.contact_force": "N, world z up; net contact force on each finger and on the plug (all partners)",
    "observation.plug_pose": "m (xyz, env frame) + dimensionless (quaternion wxyz)", "observation.socket_pose": "m (xyz, env frame) + dimensionless (quaternion wxyz)",
    "observation.keypoint_error": "m (mean distance between the plug keypoints and their seated positions)",
    "policy_action": "dimensionless in [-1, 1]: the insertion policy's action (position / rotation increments in the socket frame, scaled by the controller thresholds); zero outside the insert phase",
    "phase_index": "index into phases.names (0 approach, 1 grasp, 2 lift, 3 carry, 4 insert, 5 release, 6 return)",
}
DEPTH_LAYOUT = {
    "container": "roboassemblybench_chunked_depth_v1: file header struct '<8sI' = magic b'RABDU16\\0' + version 1; then per frame a chunk header struct '<III' = (frame_index, uncompressed_size, compressed_size) followed by compressed_size bytes of one zstd frame",
    "chunks[].offset": "byte offset of the frame's 12-byte chunk header (the zstd frame starts at offset + 12)",
    "filter": "bitplane16_le: the zstd payload holds 16 bit planes (bit 0 first) of the frame's row-major uint16 samples, each plane packed with numpy packbits(bitorder='little'); it is NOT the bitshuffle library's blocked layout",
    "values": "uint16 millimetres, 0 = invalid; depth = distance to the image plane (z in the camera optical frame)",
    "reader": "decode: zstd-decompress, reshape to (16, ceil(N / 8)) bytes, unpackbits(bitorder='little')[:N] per plane, value |= bit << plane",
}
# extra per-frame features: raw npz key -> (LeRobot name, dtype, names)
EXTRA = {
    "joint_state": ("observation.joint_state", "float32", [f"panda_joint{i}" for i in range(1, 8)]),
    "joint_velocity": ("observation.joint_velocity", "float32", [f"panda_joint{i}" for i in range(1, 8)]),
    "joint_effort": ("observation.joint_effort", "float32", [f"panda_joint{i}" for i in range(1, 8)]),
    "contact_force": ("observation.contact_force", "float32", [f"{b}_{a}" for b in ("left_finger_net", "right_finger_net", "plug_net") for a in "xyz"]),
    "plug_pose": ("observation.plug_pose", "float64", ["x", "y", "z", "qw", "qx", "qy", "qz"]),
    "socket_pose": ("observation.socket_pose", "float64", ["x", "y", "z", "qw", "qx", "qy", "qz"]),
    "keypoint_error": ("observation.keypoint_error", "float32", ["mean_keypoint_error_m"]),
    "policy_action": ("policy_action", "float32", ["dx", "dy", "dz", "drx", "dry", "drz"]),
    "phase_index": ("phase_index", "int64", None),
}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose_to_T(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(np.asarray(p[3:7], dtype=np.float64))
    T[:3, 3] = p[:3]
    return T


def ffprobe_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def decode_frames(path, idx, w=640, h=480):
    """RGB frames at the given frame indices (select filter, exact frame numbers)."""
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='{sel}'", "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def num_stats(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(len(a))]}


def img_stats(frames, n):
    f = frames.astype(np.float64) / 255.0
    per = lambda op: [[[float(v)]] for v in op(f, axis=(0, 1, 2))]
    return {"min": per(np.min), "max": per(np.max), "mean": per(np.mean), "std": per(np.std), "count": [int(n)]}


def merge_stats(parts):
    """Aggregate per-episode stats (count-weighted mean / pooled std)."""
    out = {}
    for k in parts[0]:
        mins = np.array([p[k]["min"] for p in parts]); maxs = np.array([p[k]["max"] for p in parts])
        means = np.array([p[k]["mean"] for p in parts]); stds = np.array([p[k]["std"] for p in parts])
        cnt = np.array([p[k]["count"][0] for p in parts], dtype=np.float64)
        w = cnt.reshape((-1,) + (1,) * (means.ndim - 1)) / cnt.sum()
        mean = (w * means).sum(0)
        var = (w * (stds ** 2 + (means - mean) ** 2)).sum(0)
        out[k] = {"min": mins.min(0).tolist(), "max": maxs.max(0).tolist(), "mean": mean.tolist(), "std": np.sqrt(var).tolist(), "count": [int(cnt.sum())]}
    return out


def env_four_fields(env):
    """The evaluator's four environment fields from our episode environment record."""
    ground = env.get("ground") or {}
    gm = ground.get("material") or {}
    background = f"fabrica_textures/{gm['texture']}" if gm.get("texture") else f"default ground material ({gm.get('type', 'asset default')})"
    tm = (env.get("table") or {}).get("material") or {}
    if tm.get("texture") and "white_1x1" not in tm.get("texture", ""):
        table = f"fabrica_textures/{tm['texture']}"
    elif tm.get("diffuse_color_rgb") or tm.get("tint_rgb") or tm.get("color_rgb"):
        c = tm.get("diffuse_color_rgb") or tm.get("tint_rgb") or tm.get("color_rgb")
        table = "fabrica_table_palette_colour rgb(" + ", ".join(f"{float(v):.2f}" for v in c) + ")"
    else:
        table = f"default table material ({tm.get('type', 'asset default')})"
    dis = env.get("distractors") or []
    cats = [d.get("category") or "unknown" for d in dis]  # same strings as distractors[].category
    return background, table, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--episode-names", default="", help="comma-separated raw episode directory names to export (samples)")
    ap.add_argument("--episodes-per-file", type=int, default=50)
    ap.add_argument("--visual-randomization", type=Path, default=None)
    ap.add_argument("--no-depth", action="store_true", help="leave the depth sidecar out (small public samples); the full export keeps it")
    a = ap.parse_args()
    assert not a.out.exists(), f"output exists: {a.out} (never appended to)"
    eps = sorted(glob.glob(str(a.raw / "episode_*_cartesian_raw")))
    eps = [e for e in eps if os.path.exists(e + "/metadata.json")]
    if a.episode_names:
        want = a.episode_names.split(",")
        eps = [e for e in eps if os.path.basename(e) in want]
        assert len(eps) == len(want), f"episodes not found: {set(want) - {os.path.basename(e) for e in eps}}"
    if a.max_episodes:
        eps = eps[: a.max_episodes]
    assert eps, "no episodes"
    tmp = Path(str(a.out) + ".partial")
    assert not tmp.exists(), f"stale partial output: {tmp}"
    tmp.mkdir(parents=True)

    metas = [json.load(open(e + "/metadata.json")) for e in eps]
    m0 = metas[0]
    fps = int(m0["fps"])
    task_text = m0["main_instruction"]
    robot_type = m0["robot_type"]
    state_names, action_names = m0["state_names"], m0["action_names"]

    # calibrations: one row per distinct (camera, mount, K); the measured wrist mount differs between episodes by
    # ~1-2 um (simulation noise of the measured link transform), and an ID must not cover different content
    cal_rows, ep_cal = [], []
    seen = {}
    for m in metas:
        ids = {}
        for k in CAMS:
            c, cam = m["camera_calibrations"][k], m["cameras"][k]
            key = (k, tuple(np.round(np.asarray(c["T_parent_camera"], dtype=np.float64).reshape(-1), 12)),
                   tuple(np.round(np.asarray(cam["intrinsics"], dtype=np.float64).reshape(-1), 9)))
            if key not in seen:
                cid = f"{CAM_NAMES[k]}_mount{sum(1 for kk in seen if kk[0] == k):04d}"
                seen[key] = cid
                cal_rows.append({"calibration_id": cid, "camera_key": k, "calibration_role": "mount",
                                 "parent_frame": "world" if c["parent_frame"] == "env" else c["parent_frame"],
                                 "parent_frame_note": ("world = this dataset's world frame (the episode's env frame, see schema world_frame)"
                                                       if c["parent_frame"] == "env" else "the end-effector link whose pose is observation.state[0:7]"),
                                 "T_parent_camera": np.asarray(c["T_parent_camera"], dtype=np.float64).tolist(),
                                 "T_parent_camera_meaning": "camera optical coordinates (x right, y down, z forward) -> parent frame, m",
                                 "mount_measurement": ("measured in simulation each episode; deviation over the episode "
                                                       f"{c.get('max_translation_deviation_over_episode_m', 0.0):.2e} m") if c["parent_frame"] != "env" else "fixed",
                                 "K": np.asarray(cam["intrinsics"], dtype=np.float64).tolist(), "image_size_wh": [640, 480],
                                 "distortion_model": "none", "distortion_coefficients": [],
                                 "camera_model": cam.get("camera_model"), "intrinsics_source": cam.get("intrinsics_source")})
            ids[k] = seen[key]
        ep_cal.append(ids)

    # per-file grouping
    groups = [list(range(i, min(i + a.episodes_per_file, len(eps)))) for i in range(0, len(eps), a.episodes_per_file)]
    ep_rows, env_rows, epi_rows, pose_rows = [], [], [], []
    ep_stats_all = []
    global_index = 0
    for fi, grp in enumerate(groups):
        chunk = fi // CHUNKS_SIZE
        cols = {n: [] for n in ("episode_index", "frame_index", "index", "task_index", "timestamp", "observation.state", "action")}
        for k in EXTRA.values():
            cols[k[0]] = []
        concat_lists = {k: [] for k in CAMS}
        vid_cursor = {k: 0 for k in CAMS}
        for ei in grp:
            e, m = eps[ei], metas[ei]
            z = np.load(e + "/trajectory.npz")
            n = int(z["observation_state"].shape[0])
            assert z["action"].shape[0] == n and m["frame_count"] == n
            for k in CAMS:
                vf = ffprobe_frames(Path(e) / RAW_VIDEO[k])
                assert vf == n, f"{e} {k}: video {vf} frames vs {n} rows"
                concat_lists[k].append(str(Path(e) / RAW_VIDEO[k]))
            cols["episode_index"] += [ei] * n
            cols["frame_index"] += list(range(n))
            cols["index"] += list(range(global_index, global_index + n))
            cols["task_index"] += [0] * n
            cols["timestamp"] += (np.arange(n) / fps).tolist()
            cols["observation.state"] += z["observation_state"].astype(np.float64).tolist()
            cols["action"] += z["action"].astype(np.float64).tolist()
            st = {"episode_index": num_stats([ei] * n), "frame_index": num_stats(np.arange(n)), "index": num_stats(np.arange(global_index, global_index + n)),
                  "task_index": num_stats([0] * n), "timestamp": num_stats(np.arange(n) / fps),
                  "observation.state": num_stats(z["observation_state"]), "action": num_stats(z["action"])}
            for raw_k, (name, dt, _) in EXTRA.items():
                v = np.asarray(z[raw_k])
                cols[name] += (v.astype(dt).tolist() if v.ndim > 1 else v.astype(dt).reshape(-1, 1).tolist() if name != "phase_index" else v.astype(dt).tolist())
                st[name] = num_stats(v)
            row = {"episode_index": ei, "length": n, "dataset_from_index": global_index, "dataset_to_index": global_index + n, "tasks": [task_text]}
            for k in CAMS:
                row[f"videos/{k}/chunk_index"] = chunk
                row[f"videos/{k}/file_index"] = fi
                row[f"videos/{k}/from_timestamp"] = vid_cursor[k] / fps
                row[f"videos/{k}/to_timestamp"] = (vid_cursor[k] + n) / fps
                vid_cursor[k] += n
                sample = decode_frames(Path(e) / RAW_VIDEO[k], list(range(0, n, 30)))
                st[k] = img_stats(sample, n)
            row.update({"data/chunk_index": chunk, "data/file_index": fi, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
            for sk, sv in st.items():
                for stat, val in sv.items():
                    row[f"stats/{sk}/{stat}"] = val
            ep_rows.append(row)
            ep_stats_all.append(st)
            # camera poses (env frame = this dataset's world), every frame and camera
            for k, raw_k in (("observation.images.front", "camera_pose_front"), ("observation.images.wrist", "camera_pose_wrist")):
                P = z[raw_k]
                for t in range(n):
                    pose_rows.append({"episode_index": ei, "frame_index": t, "timestamp": t / fps, "camera_key": k,
                                      "T_world_camera": pose_to_T(P[t]).reshape(-1).tolist(), "pose_source": "simulator"})
            # environment and episode sidecar rows
            env = m["environment"]
            bg, table, cats = env_four_fields(env)
            env_id = f"env_{ei:06d}"
            env_rows.append({"environment_id": env_id, "background_texture": bg, "table_texture": table, "clutter_count": len(cats),
                             "clutter_object_categories": cats, "scene_asset": env.get("scene_asset"), "scene_props": env.get("scene_props"),
                             "lighting": env.get("lights"), "distractors": env.get("distractors"), "table": env.get("table"), "ground": env.get("ground"),
                             "back_wall": env.get("back_wall"), "actual_randomization_parameters": env.get("randomization"),
                             "clutter_note": ("clutter = the visual-only distractors added on the table (primitive_box / primitive_cylinder); the "
                                              "warehouse props of the background scene (2-6 m from the front camera, outside the workspace) are scene "
                                              "content, listed in scene_props, not clutter")})
            gen, mt = m["generation"], m.get("metrics", {})
            prod = gen["base_trajectory_id"].split("/")[0]  # production batch, e.g. automate_fullproc_20261004
            src = gen["source_episode_id"]
            src = prod + src[len(src.split("/")[0]):] if src.split("/")[0] != prod else src
            epi_rows.append({"episode_index": ei, "metric_episode_id": f"{a.dataset_id}/{ei}", "batch_id": prod,
                             "source_episode_id": src, "base_trajectory_id": gen["base_trajectory_id"],
                             "generation_type": gen["generation_type"], "variant_index": gen.get("variant_index", 0), "layout_id": gen.get("layout_id"),
                             "seed": m.get("seed"), "layout_seed": m.get("layout_seed"), "round": m.get("round"), "env_index": m.get("env_index"),
                             "T_world_env": m.get("T_world_env"), "environment_id": env_id, "camera_calibrations": dict(ep_cal[ei]),
                             "main_instruction": task_text, "target_categories": m.get("target_categories"),
                             "success": bool(mt.get("success", False)), "target_reached": bool(mt.get("target_reached", False)),
                             "attachment_assist": bool(m.get("attachment_assist", {}).get("used", False)) if isinstance(m.get("attachment_assist"), dict) else False,
                             "success_rule": mt.get("success_rule", "keypoint"), "kp_final_m": mt.get("kp_final_m"), "seat_final": mt.get("seat_final"),
                             **({} if mt.get("seat_final") else {"seat_final_missing_reason": "the plug seat pose (axial / lateral / tilt) was recorded only by the collector version with the deeper_ok rule"}),
                             "descend_landing": (m.get("scripted_waypoints") or {}).get("descend_landing"),
                             **({} if (m.get("scripted_waypoints") or {}).get("descend_landing") else {"descend_landing_missing_reason": "collected before the soft-landing descent was introduced (2026-10-09 11:57 CST); the episode passed the finger-table contact check"}),
                             "finger_up_contact_force_max_n": ((mt.get("pick") or {}).get("finger_up_contact_force_max_n")),
                             "grasp_raise_m": m.get("grasp_raise_m", 0.0), "phase_start_frames": mt.get("phase_start_frames"),
                             "visual_randomization_version": vr_version(m)})
            # depth + annotation sidecars
            if not a.no_depth:
                dd = tmp / "sensors" / "depth" / f"episode_{ei:06d}"
                dd.mkdir(parents=True)
                for f in glob.glob(e + "/sensors/depth/*.zst"):
                    shutil.copy2(f, dd / os.path.basename(f))
                idx = json.load(open(e + "/sensors/depth/index.json"))
                for ck, v in idx.items():
                    v["path"] = f"sensors/depth/episode_{ei:06d}/{os.path.basename(v['path'])}"  # relative to the dataset root
                    v["filter"] = "bitplane16_le"
                    v["layout"] = DEPTH_LAYOUT
                    assert int(v["count"]) == int(m["frame_count"]), f"{e}: depth {ck} has {v['count']} frames"
                (dd / "index.json").write_text(json.dumps(idx, ensure_ascii=False))
            ad = tmp / "annotations"; ad.mkdir(exist_ok=True)
            for f in glob.glob(e + "/annotations/*.json"):
                ann = json.load(open(f))
                ann["lerobot_episode_index"] = ei  # episode_idx inside is the raw collection index
                (ad / f"episode_{ei:06d}.json").write_text(json.dumps(ann, ensure_ascii=False))
            global_index += n
        # data file
        fields = [pa.field("episode_index", pa.int64()), pa.field("frame_index", pa.int64()), pa.field("index", pa.int64()),
                  pa.field("task_index", pa.int64()), pa.field("timestamp", pa.float64()),
                  pa.field("observation.state", pa.list_(pa.float64())), pa.field("action", pa.list_(pa.float64()))]
        for raw_k, (name, dt, _) in EXTRA.items():
            fields.append(pa.field(name, pa.int64() if name == "phase_index" else pa.list_(pa.float32() if dt == "float32" else pa.float64())))
        table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in fields})
        dp = tmp / f"data/chunk-{chunk:03d}/file-{fi:03d}.parquet"
        dp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dp)
        # videos: concatenate the raw per-episode streams without re-encoding
        for k in CAMS:
            vp = tmp / f"videos/{k}/chunk-{chunk:03d}/file-{fi:03d}.mp4"
            vp.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=tmp) as lf:
                for p in concat_lists[k]:
                    lf.write(f"file '{p}'\n")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lf.name, "-c", "copy", "-y", str(vp)], check=True)
            os.unlink(lf.name)

    # meta
    total = global_index
    first_vid = m0["video_shapes"] if "video_shapes" in m0 else {}
    features = {
        "observation.state": {"dtype": "float64", "shape": [len(state_names)], "names": state_names},
        "action": {"dtype": "float64", "shape": [len(action_names)], "names": action_names},
        "episode_index": {"dtype": "int64", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]}, "task_index": {"dtype": "int64", "shape": [1]},
    }
    z0 = np.load(eps[0] + "/trajectory.npz")
    for raw_k, (name, dt, names) in EXTRA.items():
        v = np.asarray(z0[raw_k])
        features[name] = {"dtype": dt, "shape": [1] if v.ndim == 1 else [int(v.shape[1])], "names": names}
    for k in CAMS:
        features[k] = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"],
                       "info": {"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    info = {"codebase_version": "v3.0", "robot_type": robot_type, "total_episodes": len(eps), "total_frames": total, "total_tasks": 1, "fps": fps,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "total_videos": len(CAMS), "total_chunks": groups and (len(groups) - 1) // CHUNKS_SIZE + 1, "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "splits": {"train": f"0:{len(eps)}"}, "features": features,
            "roboassemblybench_sidecars": {"depth": "sensors/depth/episode_XXXXXX/ (index.json + per-camera .u16.bshuf.zst, container roboassemblybench_chunked_depth_v1, uint16 mm, 0 = invalid, distance to image plane)",
                                           "annotations": "annotations/episode_XXXXXX.json"}}
    if a.no_depth:
        info["roboassemblybench_sidecars"]["depth"] = "omitted in this export (--no-depth); the full dataset carries it"
    (tmp / "meta").mkdir(exist_ok=True)
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    pandas_meta = {"index_columns": ["__index_level_0__"],
                   "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
                   "columns": [{"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
                               {"name": None, "field_name": "__index_level_0__", "pandas_type": "unicode", "numpy_type": "object", "metadata": None}],
                   "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.2.3"}
    tasks = pa.table({"task_index": pa.array([0], pa.int64()), "__index_level_0__": pa.array([task_text], pa.string())})
    pq.write_table(tasks.replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()}), tmp / "meta/tasks.parquet")
    (tmp / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), tmp / "meta/episodes/chunk-000/file-000.parquet")
    (tmp / "meta/stats.json").write_text(json.dumps(merge_stats(ep_stats_all), indent=2))

    # diversity sidecars
    dv = tmp / "meta/diversity"; (dv / "camera_poses/chunk-000").mkdir(parents=True)
    vr = json.load(open(a.visual_randomization))["visual_randomization"] if a.visual_randomization else None
    thr = float((m0.get("metrics") or {}).get("strict_thresh_m", 0.003))
    dk = m0.get("deeper_ok") or {}
    deeper = {"out_m": dk.get("out_m", 0.003), "lateral_m": dk.get("lateral_m", 0.003), "tilt_deg": dk.get("tilt_deg", 5.0)}
    success_rules = {
        "keypoint": (f"pick, carry and insertion reach the target (mean keypoint error < {thr} m), the gripper opens, the arm returns to the fixed "
                     f"home pose, and the mean keypoint error is still < {thr} m at the last frame"),
        "deeper_ok": {**deeper, "rule": ("for assemblies whose AutoMate goal pose has no physical stop (e.g. a sleeve on a post, a hole deeper "
                                         "than the goal): the target is reached (mean keypoint error < strict threshold) before the release, and at "
                                         f"the last frame the plug base is at most {deeper['out_m']} m out of the goal along the socket axis (any "
                                         f"depth further in), at most {deeper['lateral_m']} m off laterally and tilted at most {deeper['tilt_deg']} deg; "
                                         "the mean keypoint error at the last frame can then exceed the strict threshold")},
    }
    success_def = (success_rules["keypoint"] + "; or, for episodes with success_rule = deeper_ok in episodes.jsonl, the deeper_ok rule in "
                   "success_rules; only successful episodes are kept")
    asset_registry = {"note": "texture ids are paths relative to the upstream Fabrica texture set; sha256 per episode in environments.jsonl"}
    va = os.environ.get("AUTOMATE_VISUAL_ASSETS") or (os.path.join(os.environ["SHARED_ROOT"], "data/visual_assets") if os.environ.get("SHARED_ROOT") else "")
    for w in [va]:  # the visual assets of the collection (AUTOMATE_VISUAL_ASSETS, or <SHARED_ROOT>/data/visual_assets)
        tm = os.path.join(w, "fabrica_textures/manifest.json")
        pm = os.path.join(w, "warehouse_props/manifest.json")
        if os.path.isfile(tm):
            man = json.load(open(tm))
            asset_registry["fabrica_textures"] = {f"fabrica_textures/{x['file']}": {"upstream_relpath": x.get("upstream_relpath"), "sha256": x.get("sha256")}
                                                  for grp in man.values() if isinstance(grp, list) for x in grp}
        if os.path.isfile(pm):
            asset_registry["scene_props"] = {x["id"]: {"url": x.get("url"), "category": x.get("category")} for x in json.load(open(pm))["props"]}
        if os.path.isfile(tm):
            break
    assert "fabrica_textures" in asset_registry and "scene_props" in asset_registry, "asset manifests not found: asset_registry would be incomplete"
    schema = {
        "schema_version": "ourdata-diversity-1.1", "artifact_kind": "complete_dataset", "dataset_id": a.dataset_id,
        "robot_type": robot_type, "task": m0["assembly_id"], "profile": m0["visual_profile"], "role": "synthetic",
        "world_frame": "env", "world_frame_note": m0["conventions"]["world_frame"], "world_axes": "right_handed_z_up",
        "length_unit": "m", "time_unit": "s", "quaternion_order": "wxyz", "camera_axes": "opencv", "view_metric_axes": "opengl",
        "camera_name_map": dict(CAM_NAMES),
        "state_indices": {"left_xyz": [0, 1, 2], "right_xyz": [0, 1, 2], "left_gripper": 7, "right_gripper": 7,
                          "note": ("single-arm Franka: the one arm is duplicated into both slots, so dual-arm sums computed from them "
                                   "(e.g. export_features.py total_ee_path_length = left + right) are 2x the true single-arm value; "
                                   "left_* and right_* displacement / gripper range are identical")},
        "action_semantics": "cartesian_pose_and_gripper_target",
        "action_semantics_detail": m0["action_semantics"] + "; the last frame's action is the pose / gripper target of the terminal frame, which is not itself recorded",
        "eef_pose_frame": m0["eef_pose_frame"], "eef_links": {"arm": m0["eef_link"]},
        "feature_units": {**m0["feature_units"], **EXTRA_UNITS}, "gripper_definition": m0["gripper_definition"],
        "depth_format": DEPTH_LAYOUT,
        "rgb_present": True, "depth_present": not a.no_depth, "pointcloud_present": False,
        **({"depth_note": "depth omitted in this sample export (--no-depth); the full dataset carries the depth sidecar described in depth_format"} if a.no_depth else {}),
        "success_definition": success_def, "success_rules": success_rules,
        "target_reached_definition": m0["target_reached_definition"],
        "grasp_raise": {"note": ("episodes.jsonl grasp_raise_m: the nominal AutoMate grasp (plug_grasps.json) moved back by this distance along "
                                 "the fingertip approach axis, i.e. the fingertips grasp the plug that much higher, in the pick and in the "
                                 "policy's training reset; used on tasks whose original grasp is at or below the table surface"),
                        "values_m_in_this_dataset": sorted({float(r["grasp_raise_m"] or 0.0) for r in epi_rows})},
        "attachment_assist": m0["attachment_assist"], "phases": m0.get("phases"), "control": m0.get("control"),
        "pick_contact_check": next((m.get("pick_contact_check") for m in metas if m.get("pick_contact_check")), None),
        "scripted_waypoints": next((m.get("scripted_waypoints") for m in metas if (m.get("scripted_waypoints") or {}).get("descend_landing")), None),
        "scripted_waypoints_note": ("episodes.jsonl descend_landing: the soft-landing descent (each control step moves at most land_gain x the remaining "
                                    "error, no jump to the goal) used by the episode, null for episodes collected before it was introduced"),
        "metric_definitions": {"kp_final_m": "the larger of the mean keypoint error at the last recorded frame and at the terminal (unrecorded) state",
                               "seat_final": "plug base relative to its goal in the socket frame at the last frame: axial_m (+ = out), lateral_m, tilt_deg",
                               "finger_up_contact_force_max_n": "maximum upward (world +z) net contact force on the two fingers, summed, during approach and grasp"},
        "joint_commands": "not recorded (the controller outputs joint torques); measured joint states are observation.joint_*",
        "planned_combinations": [{"batch_id": epi_rows[0]["batch_id"], "profile": m0["visual_profile"], "base_trajectories": len(eps), "variants_per_base": 1,
                                  "generation_type": "new_motion"}],
        "visual_randomization": vr,
        "visual_randomization_versions_in_this_dataset": dict(sorted(collections.Counter(r["visual_randomization_version"] for r in epi_rows).items())),
        "asset_registry": asset_registry,
        "environment_field_mapping": {"background_texture": "the floor (ground plane) texture of the episode", "table_texture": "the table texture file, or the Fabrica palette colour of a plain table",
                                      "clutter": "visual-only distractors on the table; background warehouse props are scene content (environments.jsonl scene_props)"},
        "coordinate_note": "T_world_camera maps camera optical (OpenCV) coordinates to the episode's env frame; T_world_env in episodes.jsonl maps the env frame to the simulation world (pure translation)",
        "frame_rate_provenance": f"{fps} Hz recording (physics 120 Hz, every {m0.get('frame_stride')} steps); timestamp = frame_index / fps",
    }
    (dv / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    for name, rows in (("calibrations.jsonl", cal_rows), ("environments.jsonl", env_rows), ("episodes.jsonl", epi_rows)):
        (dv / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    pose_schema = pa.schema([("episode_index", pa.int64()), ("frame_index", pa.int64()), ("timestamp", pa.float64()), ("camera_key", pa.string()),
                             ("T_world_camera", pa.list_(pa.float64(), 16)), ("pose_source", pa.string())])
    pq.write_table(pa.Table.from_pylist(pose_rows, schema=pose_schema), dv / "camera_poses/chunk-000/file-000.parquet")

    # structural check
    rows = sum(pq.read_metadata(p).num_rows for p in glob.glob(str(tmp / "data/*/*.parquet")))
    assert rows == total == sum(r["length"] for r in ep_rows), "row count"
    meta_eps = pq.read_table(tmp / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [r["episode_index"] for r in meta_eps] == list(range(len(eps))), "episode membership"
    for fi, grp in enumerate(groups):
        for k in CAMS:
            vp = tmp / f"videos/{k}/chunk-{fi // CHUNKS_SIZE:03d}/file-{fi:03d}.mp4"
            assert ffprobe_frames(vp) == sum(ep_rows[i]["length"] for i in grp), f"video frames {vp}"
            # all episode start frames of this file in one decoding pass (frame order = start order)
            starts = [round(ep_rows[i][f"videos/{k}/from_timestamp"] * fps) for i in grp]
            got = decode_frames(vp, starts)
            assert got.shape[0] == len(starts), f"decoded {got.shape[0]} of {len(starts)} start frames of {vp}"
            for j, i in enumerate(grp):
                a0 = decode_frames(Path(eps[i]) / RAW_VIDEO[k], [0])
                assert a0.shape[0] == 1 and np.array_equal(got[j], a0[0]), f"first frame of episode {i} in {vp} differs from the raw video"
    tmp.rename(a.out)
    print(json.dumps({"out": str(a.out), "episodes": len(eps), "frames": total, "files": len(groups), "structural_check": "pass"}))


if __name__ == "__main__":
    main()
