# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from .asset_root import ASSET_DIR

_ASSEMBLY_ID = os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_ASSEMBLY_ID", "00015")

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class AssemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # palm_to_finger_dist: float = 0.1034
    palm_to_finger_dist: float = 0.1134

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]  # socket stays on the table: xy only
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_threading task only.
    action_penalty_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15

    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    # SDF reward
    sdf_rwd_scale: float = 1.0
    success_rwd_scale: float = 10.0  # per step while inserted; same order as the SDF term (was 1)
    lateral_rwd_scales: list = [5.0, 3.0, 5.0]  # lateral alignment reward weights: coarse, mid, fine
    lateral_sigmas: list = [0.005, 0.001, 0.0003]  # m
    num_mesh_sample_points: int = 1000

    # Imitation reward
    imitation_rwd_scale: float = 1.0  # vendored value; at 3 a plug jammed at the entrance out-earned a successful insertion
    soft_dtw_gamma: float = 0.01  # only used to build the SoftDTW criterion object; the imitation reward now uses hard DTW
    imitation_sigma: float = 0.002  # imitation reward = exp(-mean squared DTW distance / sigma^2): 1 mm -> 0.78, 2 mm -> 0.37, 4 mm -> 0.02
    num_point_robot_traj: int = 10  # number of waypoints included in the end-effector trajectory

    # SBC
    initial_max_disp: float = 0.01  # max initial downward displacement of plug at beginning of curriculum
    curriculum_success_thresh: float = 0.8  # success rate threshold for increasing curriculum difficulty
    curriculum_failure_thresh: float = 0.5  # success rate threshold for decreasing curriculum difficulty
    curriculum_freespace_range: float = 0.01
    num_curriculum_step: int = 10
    curriculum_height_step: list = [
        -0.005,
        0.003,
    ]  # how much to increase max initial downward displacement after hitting success or failure thresh

    if_sbc: bool = True
    # start-state curriculum used when if_sbc (see assembly_env.randomize_held_initial_state)
    start_curriculum_levels: int = 10
    start_curriculum_p_in_max: float = 0.2  # in-hole start share at level 0 (0.5: pushing straight down dominated)
    start_curriculum_replay: float = 0.2  # share of envs that replay a random easier level
    start_above_h_range: list = [0.0005, 0.010]  # above-hole start: height of the plug above the entrance (m)
    start_above_near: list = [0.0005, 0.003]  # half of the above-hole starts are drawn from this band
    start_in_min_disp: float = 0.003  # in-hole start: at least this far from seated

    # Logging evaluation results
    if_logging_eval: bool = os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_LOG_EVAL", "0") == "1"
    num_eval_trials: int = 100
    eval_filename: str = os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_EVAL_FILENAME", f"evaluation_{_ASSEMBLY_ID}.h5")


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = "plug.usd"
    obj_path = "plug.obj"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = "socket.usd"
    obj_path = "socket.obj"
    diameter = 0.0081
    height = 0.050896
    base_height = 0.0


@configclass
class Insertion(AssemblyTask):
    name = "insertion"

    assembly_id = _ASSEMBLY_ID
    assembly_dir = f"{ASSET_DIR}/{assembly_id}/"

    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    plug_grasp_json = f"{ASSET_DIR}/plug_grasps.json"
    disassembly_dist_json = f"{ASSET_DIR}/disassembly_dist.json"
    disassembly_path_json = os.environ.get(
        "ROBOASSEMBLYBENCH_AUTOMATE_DISASSEMBLY_TRAJ", f"{assembly_dir}/disassemble_traj.json"
    )

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    hand_width_max: float = 0.080  # maximum opening width of gripper

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]  # socket stays on the table: xy only
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0
    fixed_asset_z_offset: float = 0.0  # was 0.1435 (socket floated 14-24 cm above the table)

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    engage_height_thresh: float = 0.01
    success_height_thresh: float = 0.003
    close_error_thresh: float = 0.015  # vendored success keypoint threshold (no longer used for success)
    success_keypoint_thresh: float = 0.003  # success: mean keypoint error below this (bounds height, lateral offset, tilt)
    success_height_lower_tol: float = 0.0015  # plug_z may sit this far below socket_z: fully inserted plugs settle up to ~1.1 mm below (19-task probe); vendored 0 failed 94-100% of them on 7 tasks
    grasp_min_time: float = 0.25  # reset: close the gripper at least this long (vendored fixed duration)
    grasp_max_time: float = 1.0  # ... and at most this long
    grasp_settle_tol: float = 1e-5  # fingers count as settled when no finger joint moves more than this per physics step (m)

    fixed_asset: ArticulationCfg = ArticulationCfg(
        # fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assembly_dir}{fixed_asset_cfg.usd_path}",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, -0.003),  # socket base on the table top (table collider top z = -0.003)
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
    # held_asset: ArticulationCfg = ArticulationCfg(
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assembly_dir}{held_asset_cfg.usd_path}",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        # init_state=ArticulationCfg.InitialStateCfg(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            # joint_pos={},
            # joint_vel={}
        ),
        # actuators={}
    )
