# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import subprocess
import sys

import math

import torch
import trimesh
import warp as wp

print("Python Executable:", sys.executable)
print("Python Path:", sys.path)

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
sys.path.append(base_dir)

from .asset_root import resolve_asset_file

"""
Util Functions
"""


def parse_cuda_version(version_string):
    """
       Parse CUDA version string into comparable tuple of (major, minor, patch).

       Args:
           version_string: Version string like "12.8.9" or "11.2"

       Returns:
           Tuple of (major, minor, patch) as integers, where patch defaults to 0 iff
    not present.

       Example:
           "12.8.9" -> (12, 8, 9)
           "11.2" -> (11, 2, 0)
    """
    parts = version_string.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)


def get_cuda_version():
    try:
        # Execute nvcc --version command
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        output = result.stdout

        # Use regex to find the CUDA version (e.g., V11.2.67)
        match = re.search(r"V(\d+\.\d+(\.\d+)?)", output)
        if match:
            return parse_cuda_version(match.group(1))
        else:
            print("CUDA version not found in output.")
            return None
    except FileNotFoundError:
        print("nvcc command not found. Is CUDA installed and in your PATH?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvcc: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_gripper_open_width(obj_filepath):
    obj_mesh = trimesh.load_mesh(resolve_asset_file(obj_filepath, download_dir="./"))
    # obj_mesh = trimesh.load_mesh(obj_filepath)
    aabb = obj_mesh.bounds

    return min(0.04, (aabb[1][1] - aabb[0][1]) / 1.25)


"""
Imitation Reward
"""


def get_closest_state_idx(ref_traj, curr_ee_pos):
    """Find the index of the closest state in reference trajectory."""

    # ref_traj.shape = (num_trajs, traj_len, 3)
    traj_len = ref_traj.shape[1]
    num_envs = curr_ee_pos.shape[0]

    # dist_from_all_state.shape = (num_envs, num_trajs, traj_len, 1)
    dist_from_all_state = torch.cdist(ref_traj.unsqueeze(0), curr_ee_pos.reshape(-1, 1, 1, 3), p=2)

    # dist_from_all_state_flatten.shape = (num_envs, num_trajs * traj_len)
    dist_from_all_state_flatten = dist_from_all_state.reshape(num_envs, -1)

    # min_dist_per_env.shape = (num_envs)
    min_dist_per_env = torch.amin(dist_from_all_state_flatten, dim=-1)

    # min_dist_idx.shape = (num_envs)
    min_dist_idx = torch.argmin(dist_from_all_state_flatten, dim=-1)

    # min_dist_traj_idx.shape = (num_envs)
    # min_dist_step_idx.shape = (num_envs)
    min_dist_traj_idx = min_dist_idx // traj_len
    min_dist_step_idx = min_dist_idx % traj_len

    return min_dist_traj_idx, min_dist_step_idx, min_dist_per_env


def get_reward_mask(ref_traj, curr_ee_pos, tolerance):
    _, min_dist_step_idx, _ = get_closest_state_idx(ref_traj, curr_ee_pos)
    selected_steps = torch.index_select(
        ref_traj, dim=1, index=min_dist_step_idx
    )  # selected_steps.shape = (num_trajs, num_envs, 3)

    x_min = torch.amin(selected_steps[:, :, 0], dim=0) - tolerance
    x_max = torch.amax(selected_steps[:, :, 0], dim=0) + tolerance
    y_min = torch.amin(selected_steps[:, :, 1], dim=0) - tolerance
    y_max = torch.amax(selected_steps[:, :, 1], dim=0) + tolerance

    x_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 0], x_max), torch.gt(curr_ee_pos[:, 0], x_min))
    y_in_range = torch.logical_and(torch.lt(curr_ee_pos[:, 1], y_max), torch.gt(curr_ee_pos[:, 1], y_min))
    pos_in_range = torch.logical_and(x_in_range, y_in_range).int()

    return pos_in_range


_DTW_DIAG_CACHE = {}


def _dtw_antidiagonals(n, m, device):
    """Cell indices (i, j), 1-based, of each anti-diagonal i + j = const of an (n x m) DP table."""
    key = (n, m, str(device))
    if key not in _DTW_DIAG_CACHE:
        ii = torch.arange(1, n + 1)
        diags = []
        for p in range(2, n + m + 1):
            j = p - ii
            ok = (j >= 1) & (j <= m)
            if ok.any():
                diags.append((ii[ok].to(device), j[ok].to(device)))
        _DTW_DIAG_CACHE[key] = diags
    return _DTW_DIAG_CACHE[key]


def batched_dtw_mean_cost(x, y, lengths):
    """Mean squared distance along the optimal DTW alignment, x (B, N, 3) vs each prefix y[b, :lengths[b]] (B, M, 3).

    Hard DTW (min over alignments) with squared Euclidean cost, divided by the number of aligned pairs on the
    optimal path, so the value does not depend on how long the reference segment is (a plug starting next to the
    hole gives a short segment). Padding y to a common length is exact because the DP is causal.
    """
    d = torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2).sum(3)
    b, n, m = d.shape
    r = torch.full((b, n + 1, m + 1), math.inf, device=x.device, dtype=d.dtype)
    cnt = torch.zeros((b, n + 1, m + 1), device=x.device, dtype=d.dtype)
    r[:, 0, 0] = 0
    for iv, jv in _dtw_antidiagonals(n, m, x.device):
        prev_r = torch.stack([r[:, iv - 1, jv - 1], r[:, iv - 1, jv], r[:, iv, jv - 1]], dim=-1)
        prev_c = torch.stack([cnt[:, iv - 1, jv - 1], cnt[:, iv - 1, jv], cnt[:, iv, jv - 1]], dim=-1)
        best, arg = prev_r.min(dim=-1)
        r[:, iv, jv] = d[:, iv - 1, jv - 1] + best
        cnt[:, iv, jv] = prev_c.gather(-1, arg.unsqueeze(-1)).squeeze(-1) + 1
    bi = torch.arange(b, device=x.device)
    return r[bi, n, lengths] / cnt[bi, n, lengths]


def get_imitation_reward_from_dtw(ref_traj, curr_ee_pos, prev_ee_traj, criterion, device, sigma=0.002):
    """Imitation reward (all envs in one batch): exp(-C / sigma^2).

    C is the mean squared distance between the robot's last num_point_robot_traj fingertip positions and the matched
    reference segment along their optimal DTW alignment (metres^2), so it reads directly as "how far from the
    demonstrated path, on average": with sigma = 2 mm, 1 mm -> 0.78, 2 mm -> 0.37, 4 mm -> 0.02. Replaces the vendored
    (1 - tanh(soft-DTW)) * (1 - s / L), whose value depended on the segment length and the unit (soft-min entropy
    term dominated in metres), and whose progress weight counted reference time steps (half of which are the random
    lateral moves) rather than distance; progress towards the goal is left to the SDF reward.

    Segment selection (see the project's code guide, section 5.1): disassemble_traj.json is stored in disassembly
    order (index 0 = assembled goal), while the selection assumes assembly order ("larger index -> closer to goal",
    as for the paper's reversed disassembly paths), so it runs on the time-reversed paths. The rest follows the
    vendored per-env loop: closest-state search, the curr_step_idx == 0 -> [p, p] special case. torch.cdist switches
    to its matmul formula above 25 rows; the per-env choice is reproduced so nearest-point ties resolve as before.
    """
    num_envs = curr_ee_pos.shape[0]
    traj_len = ref_traj.shape[1]
    ref_asm = ref_traj.flip(1)  # assembly order: larger index -> closer to goal
    prev_ee_pos = prev_ee_traj[:, 0, :]
    min_dist_traj_idx, min_dist_step_idx, _ = get_closest_state_idx(ref_asm, prev_ee_pos)

    sel = ref_asm[min_dist_traj_idx]  # (B, L, 3)
    q = curr_ee_pos.reshape(-1, 1, 1, 3)
    d_mm = torch.cdist(sel.unsqueeze(1), q, p=2, compute_mode="use_mm_for_euclid_dist").reshape(num_envs, traj_len)
    d_nomm = torch.cdist(sel.unsqueeze(1), q, p=2, compute_mode="donot_use_mm_for_euclid_dist").reshape(num_envs, traj_len)
    dist = torch.where(((traj_len - min_dist_step_idx) > 25).unsqueeze(1), d_mm, d_nomm)
    steps = torch.arange(traj_len, device=ref_traj.device)
    dist = torch.where(steps.unsqueeze(0) < min_dist_step_idx.unsqueeze(1), torch.full_like(dist, math.inf), dist)
    curr_step_idx = torch.argmin(dist, dim=1) - min_dist_step_idx

    # reference segment [step_idx, step_idx + curr_step_idx), or [p, p] when curr_step_idx == 0
    lengths = torch.where(curr_step_idx == 0, torch.full_like(curr_step_idx, 2), curr_step_idx)
    m = int(lengths.max().item())
    offs = torch.minimum(torch.arange(m, device=ref_traj.device).unsqueeze(0), (lengths - 1).unsqueeze(1))
    offs = torch.where((curr_step_idx == 0).unsqueeze(1), torch.zeros_like(offs), offs)
    idx = (min_dist_step_idx.unsqueeze(1) + offs).clamp(max=traj_len - 1)
    seg = sel[torch.arange(num_envs, device=ref_traj.device).unsqueeze(1), idx]

    eef_traj = torch.cat((prev_ee_traj[:, 1:, :], curr_ee_pos.unsqueeze(1)), dim=1)
    mean_sq_dist = batched_dtw_mean_cost(eef_traj, seg, lengths)
    return torch.exp(-mean_sq_dist / sigma**2)


"""
Sampling-Based Curriculum (SBC)
"""


def get_new_max_disp(curr_success, cfg_task, curriculum_height_bound, curriculum_height_step, curr_max_disp):
    """Update max downward displacement of plug at beginning of episode, based on success rate."""

    if curr_success > cfg_task.curriculum_success_thresh:
        # If success rate is above threshold, increase min downward displacement until max value
        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 0] < curriculum_height_bound[:, 1],
            curr_max_disp + curriculum_height_step[:, 0],
            curriculum_height_bound[:, 1],
        )
    elif curr_success < cfg_task.curriculum_failure_thresh:
        # If success rate is below threshold, decrease min downward displacement until min value
        new_max_disp = torch.where(
            curr_max_disp + curriculum_height_step[:, 1] > curriculum_height_bound[:, 0],
            curr_max_disp + curriculum_height_step[:, 1],
            curriculum_height_bound[:, 0],
        )
    else:
        # Maintain current max downward displacement
        new_max_disp = curr_max_disp

    return new_max_disp


"""
Bonus and Success Checking
"""


def check_plug_close_to_socket(keypoints_plug, keypoints_socket, dist_threshold, progress_buf):
    """Check if plug is close to socket."""

    # Compute keypoint distance between plug and socket
    keypoint_dist = torch.norm(keypoints_socket - keypoints_plug, p=2, dim=-1)

    # Check if keypoint distance is below threshold
    is_plug_close_to_socket = torch.where(
        torch.mean(keypoint_dist, dim=-1) < dist_threshold,
        torch.ones_like(progress_buf),
        torch.zeros_like(progress_buf),
    )

    return is_plug_close_to_socket


def check_plug_inserted_in_socket(
    plug_pos, socket_pos, disassembly_dist, keypoints_plug, keypoints_socket, close_error_thresh, progress_buf,
    height_lower_tol=0.0,
):
    """Check if plug is inserted in socket."""

    # Check if plug is within threshold distance of assembled state.
    # NOTE: at full insertion plug_z == socket_z (assembled frames coincide); contact noise puts a bottomed-out plug
    # up to ~1.1 mm below (measured on 19 tasks), which the vendored lower bound (height_lower_tol = 0) counts as failure.
    is_plug_below_insertion_height = plug_pos[:, 2] < socket_pos[:, 2] + disassembly_dist
    is_plug_above_table_height = plug_pos[:, 2] > socket_pos[:, 2] - height_lower_tol

    is_plug_height_success = torch.logical_and(is_plug_below_insertion_height, is_plug_above_table_height)

    # Check if plug is close to socket
    # NOTE: This check addresses edge case where plug is within threshold distance of
    # assembled state, but plug is outside socket
    is_plug_close_to_socket = check_plug_close_to_socket(
        keypoints_plug=keypoints_plug,
        keypoints_socket=keypoints_socket,
        dist_threshold=close_error_thresh,
        progress_buf=progress_buf,
    )

    # Combine both checks
    is_plug_inserted_in_socket = torch.logical_and(is_plug_height_success, is_plug_close_to_socket)

    return is_plug_inserted_in_socket


def get_curriculum_reward_scale(curr_max_disp, curriculum_height_bound):
    """Compute reward scale for SBC."""

    # Compute difference between max downward displacement at beginning of training (easiest condition)
    # and current max downward displacement (based on current curriculum stage)
    # NOTE: This number increases as curriculum gets harder
    curr_stage_diff = curr_max_disp - curriculum_height_bound[:, 0]

    # Compute difference between max downward displacement at beginning of training (easiest condition)
    # and min downward displacement (hardest condition)
    final_stage_diff = curriculum_height_bound[:, 1] - curriculum_height_bound[:, 0]

    # Compute reward scale
    reward_scale = curr_stage_diff / final_stage_diff + 1.0

    return reward_scale.mean()


"""
Warp Kernels
"""


# Transform points from source coordinate frame to destination coordinate frame
@wp.kernel
def transform_points(src: wp.array(dtype=wp.vec3), dest: wp.array(dtype=wp.vec3), xform: wp.transform):
    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m


# Return interpenetration distances between query points (e.g., plug vertices in current pose)
# and mesh surfaces (e.g., of socket mesh in current pose)
@wp.kernel
def get_interpen_dist(
    queries: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    interpen_dists: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    # Declare arguments to wp.mesh_query_point() that will not be modified
    q = queries[tid]  # query point
    max_dist = 1.5  # max distance on mesh from query point

    # Declare arguments to wp.mesh_query_point() that will be modified
    sign = float(
        0.0
    )  # -1 if query point inside mesh; 0 if on mesh; +1 if outside mesh (NOTE: Mesh must be watertight!)
    face_idx = int(0)  # index of closest face
    face_u = float(0.0)  # barycentric u-coordinate of closest point
    face_v = float(0.0)  # barycentric v-coordinate of closest point

    # Get closest point on mesh to query point
    closest_mesh_point_exists = wp.mesh_query_point(mesh, q, max_dist, sign, face_idx, face_u, face_v)

    # If point exists within max_dist
    if closest_mesh_point_exists:
        # Get 3D position of point on mesh given face index and barycentric coordinates
        p = wp.mesh_eval_position(mesh, face_idx, face_u, face_v)

        # Get signed distance between query point and mesh point
        delta = q - p
        signed_dist = sign * wp.length(delta)

        # If signed distance is negative
        if signed_dist < 0.0:
            # Store interpenetration distance
            interpen_dists[tid] = signed_dist
