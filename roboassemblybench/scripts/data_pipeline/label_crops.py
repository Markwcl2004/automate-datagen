"""Close-up images of each task for instruction labelling (4 per task: the API takes at most 4 images in one prompt):
- front_parts:  front camera, first frame, crops around the plug (left) and the socket (right), projected from the recorded
                poses with the camera intrinsics;
- front_mating: front camera, side view of plug and socket half-way through the insertion;
- wrist_grasp:  wrist camera at the start of the grasp (plug centred below the gripper);
- wrist_insert: wrist camera at the start of the insertion (socket below the held plug).
Uses the first kept episode of each task. Needs ffmpeg and numpy.
usage: label_crops.py --dataset-root D --out DIR [--profile mixed] <task>...  (no task = every task)"""
import argparse, json, os, subprocess
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--profile", default="mixed")
ap.add_argument("tasks", nargs="*")
a = ap.parse_args(); DS, O = a.dataset_root, a.out; os.makedirs(O, exist_ok=True)
tasks = a.tasks or sorted(t for t in os.listdir(DS) if t.isdigit() and len(t) == 5)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def frame(video, idx):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{idx})", "-vsync", "0", "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(out, np.uint8).reshape(480, 640, 3)


def save(img, path, size):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{img.shape[1]}x{img.shape[0]}", "-i", "-",
                    "-vf", f"scale={size}:flags=lanczos", path], input=np.ascontiguousarray(img).tobytes(), check=True)


def crop_at(img, K, cam_pose, point, half, dv=0):
    R, t = quat_to_R(cam_pose[3:]), cam_pose[:3]; pc = R.T @ (point - t); uv = (K @ pc)[:2] / pc[2]
    u = min(max(int(round(uv[0])), half), 640 - half); v = min(max(int(round(uv[1])) + dv, half), 480 - half)
    return img[v - half:v + half, u - half:u + half]


for t in tasks:
    tdir = f"{DS}/{t}/{a.profile}"; E = f"{tdir}/" + sorted(d for d in os.listdir(tdir) if d.endswith("_cartesian_raw"))[0]
    m = json.load(open(f"{E}/metadata.json")); z = np.load(f"{E}/trajectory.npz"); ph = m["metrics"]["phase_start_frames"]
    K = np.array(m["cameras"]["observation.images.front"]["intrinsics"]); front = f"{E}/videos/observation_images_front.mp4"
    f0 = frame(front, 0); cp0 = z["camera_pose_front"][0]
    parts = np.concatenate([crop_at(f0, K, cp0, z["plug_pose"][0][:3], 64), crop_at(f0, K, cp0, z["socket_pose"][0][:3], 64)], axis=1)
    save(parts, f"{O}/{t}_front_parts.png", "896:448")
    mid = (ph["insert"] + ph.get("target_reached", ph["insert"] + 30)) // 2
    save(crop_at(frame(front, mid), K, z["camera_pose_front"][mid], z["socket_pose"][mid][:3], 56, dv=-20), f"{O}/{t}_front_mating.png", "448:448")
    for name, idx in (("wrist_grasp", ph["grasp"]), ("wrist_insert", ph["insert"])):
        save(frame(f"{E}/videos/observation_images_wrist.mp4", idx)[:, 80:560], f"{O}/{t}_{name}.png", "448:448")
    print(t, os.path.basename(E), flush=True)
