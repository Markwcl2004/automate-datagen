# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import math

import numpy as np
import torch
import warp as wp

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .asset_root import TABLE_USD_PATH, resolve_asset_file
from isaaclab.utils.math import axis_angle_from_quat

from . import automate_algo_utils as automate_algo
from . import automate_log_utils as automate_log
from . import factory_control as fc
from . import industreal_algo_utils as industreal_algo
from .assembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, AssemblyEnvCfg
from .soft_dtw_cuda import SoftDTW


class AssemblyEnv(DirectRLEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        self.cfg_task = cfg.tasks[cfg.task_name]
        # grasp perturbation of the reset (2026-10-02): "approach_m,width_m,deg", e.g. "0.004,0.003,8"; unset = off
        gp = os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_GRASP_PERTURB")
        self.grasp_perturb = bool(gp)
        self.grasp_perturb_approach, self.grasp_perturb_width, self.grasp_perturb_deg = (
            [float(v) for v in gp.split(",")] if gp else [0.004, 0.003, 8.0])
        # fixed raise of the nominal grasp along the fingertip approach axis (m, 2026-10-09): the fingertips grasp the
        # plug higher, for tasks whose grasp is so close to the plug's bottom that the fingers reach the table; unset = 0
        self.grasp_raise = float(os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_GRASP_RAISE") or 0.0)

        super().__init__(cfg, render_mode, **kwargs)

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        # Load asset meshes in warp for SDF-based dense reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_mesh_in_warp(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path,
            self.cfg_task.assembly_dir + self.cfg_task.fixed_asset_cfg.obj_path,
            self.cfg_task.num_mesh_sample_points,
            self.wp_device,
        )
        # socket lowest point on the table top (see randomize_fixed_initial_state)
        self.socket_base_lift = max(0.0, -float(self.socket_mesh.points.numpy()[:, 2].min()))
        print(f"[SOCKET_BASE] {self.cfg_task.assembly_id} socket lifted {self.socket_base_lift * 1000:.2f} mm")

        # Get the gripper open width based on plug object bounding box
        self.gripper_open_width = automate_algo.get_gripper_open_width(
            self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path
        )

        # Create criterion for dynamic time warping (later used for imitation reward)
        cuda_version = automate_algo.get_cuda_version()
        if (cuda_version is not None) and (cuda_version < (13, 0, 0)):
            self.soft_dtw_criterion = SoftDTW(use_cuda=True, device=self.device, gamma=self.cfg_task.soft_dtw_gamma)
        else:
            self.soft_dtw_criterion = SoftDTW(use_cuda=False, device=self.device, gamma=self.cfg_task.soft_dtw_gamma)

        # Evaluate
        if self.cfg_task.if_logging_eval:
            self._init_eval_logging()

    def _init_eval_logging(self):
        self.held_asset_pose_log = torch.empty(
            (0, 7), dtype=torch.float32, device=self.device
        )  # (position, quaternion)
        self.fixed_asset_pose_log = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        self.success_log = torch.empty((0, 1), dtype=torch.float32, device=self.device)

        # Turn off SBC during evaluation so all plugs are initialized outside of the socket
        self.cfg_task.if_sbc = False

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        if self.max_tilt > 0.0:
            # roll / pitch action scale for unlocked tasks (rad per step; yaw unchanged). Env override for long plugs in
            # tight holes, where a tilt step about the fingertip swings the plug tip by more than the clearance (00783)
            self.rot_threshold[:, 0:2] = float(os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_TILT_ACTION_SCALE", 0.03))
        print(f"[TILT_UNLOCK] {self.cfg_task.assembly_id} max tilt {math.degrees(self.max_tilt):.1f} deg, "
              f"rot action scale {self.rot_threshold[0].tolist()}")

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        held_base_z_offset = 0.0

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()
        # Tasks whose plug sits tilted in the closed gripper may tilt the gripper (see _apply_action); 0 = upright lock
        self.max_tilt = 0.0
        unlock_path = os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_TILT_UNLOCK")
        if unlock_path:
            with open(unlock_path) as f:
                self.max_tilt = math.radians(float(json.load(f).get(str(self.cfg_task.assembly_id), 0.0)))
        self.curriculum_height_bound, self.curriculum_height_step = self._get_curriculum_info(self.disassembly_dists)
        self._load_disassembly_data()

        # Load grasp pose from json files given assembly ID
        # Grasp pose tensors
        self.palm_to_finger_center = (
            torch.tensor([0.0, 0.0, -self.cfg_task.palm_to_finger_dist], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.robot_to_gripper_quat = (
            torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.plug_grasp_pos_local = self.plug_grasps[: self.num_envs, :3]
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[: self.num_envs, 3:], -1, 1)
        # grasp actually realized after the fingers close (plug frame -> grasp frame), measured at every reset
        self.realized_grasp_pos_local = self.plug_grasp_pos_local.clone()
        self.realized_grasp_quat_local = self.plug_grasp_quat_local.clone()

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
        self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = 0.0

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # SBC
        if self.cfg_task.if_sbc:
            self.curr_max_disp = self.curriculum_height_bound[:, 0]
        else:
            self.curr_max_disp = self.curriculum_height_bound[:, 1]
        # start-state curriculum (training): level, and per env the kind of start of the current episode
        self.curr_level = 0.0
        self.start_above = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.start_replay = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def _load_assembly_info(self):
        """Load grasp pose and disassembly distance for plugs in each environment."""

        plug_grasp_path = resolve_asset_file(self.cfg_task.plug_grasp_json, download_dir="./")
        with open(plug_grasp_path) as f:
            plug_grasp_dict = json.load(f)
        plug_grasps = [plug_grasp_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        disassembly_dist_path = resolve_asset_file(self.cfg_task.disassembly_dist_json, download_dir="./")
        with open(disassembly_dist_path) as f:
            disassembly_dist_dict = json.load(f)
        disassembly_dists = [disassembly_dist_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _get_curriculum_info(self, disassembly_dists):
        """Calculate the ranges and step sizes for Sampling-based Curriculum (SBC) in each environment."""

        curriculum_height_bound = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        curriculum_height_step = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        curriculum_height_bound[:, 1] = disassembly_dists + self.cfg_task.curriculum_freespace_range

        curriculum_height_step[:, 0] = curriculum_height_bound[:, 1] / self.cfg_task.num_curriculum_step
        curriculum_height_step[:, 1] = -curriculum_height_step[:, 0] / 2.0

        return curriculum_height_bound, curriculum_height_step

    def _load_disassembly_data(self):
        """Load pre-collected disassembly trajectories (end-effector position only)."""

        disassembly_path = resolve_asset_file(self.cfg_task.disassembly_path_json, download_dir="./")
        with open(disassembly_path) as f:
            disassembly_traj = json.load(f)

        eef_pos_traj = []

        for i in range(len(disassembly_traj)):
            curr_ee_traj = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))
            curr_ee_goal = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))[0, :]

            # offset each trajectory to be relative to the goal
            eef_pos_traj.append(curr_ee_traj - curr_ee_goal)

        self.eef_pos_traj = torch.tensor(np.array(eef_pos_traj), dtype=torch.float32, device=self.device).squeeze()

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        table_usd_path = TABLE_USD_PATH
        if table_usd_path is None:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            table_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        cfg = sim_utils.UsdFileCfg(usd_path=table_usd_path)
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = RigidObject(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.rigid_objects["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            self.realized_grasp_quat_local,
            self.realized_grasp_pos_local,
        )

        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.gripper_goal_quat,
            self.gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors.
        self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
            self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_held[:, idx] = torch_utils.tf_combine(
                self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                self.target_held_base_quat,
                self.target_held_base_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    @staticmethod
    def _canon_quat(q):
        """Quaternion (w, x, y, z) with w >= 0 (q and -q are the same rotation)."""
        return torch.where(q[:, :1] < 0, -q, q)

    def _relative_obs(self):
        """Plug pose in the socket frame and in the fingertip frame, from the live simulation state of this step."""
        p_rs = torch_utils.quat_rotate_inverse(self.fixed_quat, self.held_pos - self.fixed_pos)
        q_rs = self._canon_quat(torch_utils.quat_mul(torch_utils.quat_conjugate(self.fixed_quat), self.held_quat))
        p_ih = torch_utils.quat_rotate_inverse(self.fingertip_midpoint_quat, self.held_pos - self.fingertip_midpoint_pos)
        q_ih = self._canon_quat(torch_utils.quat_mul(torch_utils.quat_conjugate(self.fingertip_midpoint_quat), self.held_quat))
        return {"plug_rel_socket_pos": p_rs, "plug_rel_socket_quat": q_rs, "plug_in_hand_pos": p_ih, "plug_in_hand_quat": q_ih}

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        rel = self._relative_obs()

        obs_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
            **rel,
        }

        state_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "joint_vel": self.joint_vel[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
            **rel,
        }
        # obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order + ['prev_actions']]
        obs_tensors = [obs_dict[obs_name] for obs_name in self.cfg.obs_order]
        obs_tensors = torch.cat(obs_tensors, dim=-1)

        # state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order + ['prev_actions']]
        state_tensors = [state_dict[state_name] for state_name in self.cfg.state_order]
        state_tensors = torch.cat(state_tensors, dim=-1)

        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

    def move_gripper_in_place(self, ctrl_target_gripper_dof_pos):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )
        if self.grasp_perturb:
            # In-hole starts: the plug is held aligned with the hole, so the perturbed grasp's tilt has to stay on the
            # gripper. Forcing the gripper upright while closing would tilt the plug inside the hole (wedged or pushed
            # out; 2026-10-02, 00783). Above-hole starts keep the upright gripper (plug tilted in the hand, as after
            # the scripted carry).
            in_hole = (self.curriculum_disp <= self.disassembly_dists).unsqueeze(-1)
            self.ctrl_target_fingertip_midpoint_quat = torch.where(in_hole, self.reset_hold_quat, self.ctrl_target_fingertip_midpoint_quat)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold
        # 2026-10-02: the policy acts in the socket frame (its observations are in the socket frame)
        pos_actions = torch_utils.quat_rotate(self.fixed_quat, pos_actions)
        rot_actions = torch_utils.quat_rotate(self.fixed_quat, rot_actions)

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        if self.max_tilt > 0.0:
            # roll / pitch free within max_tilt of upright (roll = pi, pitch = 0)
            roll_dev = torch.atan2(torch.sin(target_euler_xyz[:, 0] - np.pi), torch.cos(target_euler_xyz[:, 0] - np.pi))
            pitch = torch.atan2(torch.sin(target_euler_xyz[:, 1]), torch.cos(target_euler_xyz[:, 1]))
            target_euler_xyz[:, 0] = np.pi + roll_dev.clamp(-self.max_tilt, self.max_tilt)
            target_euler_xyz[:, 1] = pitch.clamp(-self.max_tilt, self.max_tilt)
        else:
            target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
            target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use GYM's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep

        # success = height within the insertion depth (as vendored) AND mean keypoint error < success_keypoint_thresh
        # (3 mm). The vendored keypoint threshold (close_error_thresh, 1.5 cm) let a plug resting at the socket
        # entrance, or tilted by up to ~17 deg, count as success; 3 mm bounds height, lateral offset and tilt
        # (keypoints span 15 cm along the plug axis, ~0.9 mm mean error per degree of tilt).
        curr_successes = automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.success_keypoint_thresh,
            self.episode_length_buf,
            height_lower_tol=self.cfg_task.success_height_lower_tol,
        )

        rew_buf = self._update_rew_buf(curr_successes)
        self.ep_succeeded = torch.logical_or(self.ep_succeeded, curr_successes)

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(self.ep_succeeded) / self.num_envs

            if self.cfg_task.if_sbc:
                K = self.cfg_task.start_curriculum_levels
                rew_buf *= 1.0 + self.curr_level / K
                succ = self.ep_succeeded.bool()
                above = self.start_above & ~self.start_replay
                if bool(above.any()):
                    s_above = (succ & above).sum().float() / above.sum()
                    if s_above > self.cfg_task.curriculum_success_thresh:
                        self.curr_level = min(float(K), self.curr_level + 1.0)
                    elif s_above < self.cfg_task.curriculum_failure_thresh:
                        self.curr_level = max(0.0, self.curr_level - 0.5)
                    self.extras["success_above"] = s_above
                inside = ~self.start_above
                if bool(inside.any()):
                    self.extras["success_in"] = (succ & inside).sum().float() / inside.sum()
                self.extras["curriculum_level"] = torch.tensor(self.curr_level, device=self.device)
            else:
                sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
                    curr_max_disp=self.curr_max_disp,
                    curriculum_height_bound=self.curriculum_height_bound,
                )
                rew_buf *= sbc_rwd_scale

            self.extras["curr_max_disp"] = self.curr_max_disp

            if self.cfg_task.if_logging_eval:
                self.success_log = torch.cat([self.success_log, self.ep_succeeded.reshape((self.num_envs, 1))], dim=0)

                if self.success_log.shape[0] >= self.cfg_task.num_eval_trials:
                    automate_log.write_log_to_hdf5(
                        self.held_asset_pose_log,
                        self.fixed_asset_pose_log,
                        self.success_log,
                        self.cfg_task.eval_filename,
                    )
                    exit(0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        rew_dict = dict({})

        # SDF-based reward.
        rew_dict["sdf"] = industreal_algo.get_sdf_reward(
            self.plug_mesh,
            self.plug_sample_points,
            self.held_pos,
            self.held_quat,
            self.fixed_pos,
            self.fixed_quat,
            self.wp_device,
            self.device,
        )

        rew_dict["curr_successes"] = curr_successes.clone().float()

        # Imitation Reward: Calculate reward
        curr_eef_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).reshape(
            -1, 3
        )  # relative position instead of absolute position
        rew_dict["imitation"] = automate_algo.get_imitation_reward_from_dtw(
            self.eef_pos_traj,
            curr_eef_pos,
            self.prev_fingertip_midpoint_pos,
            self.soft_dtw_criterion,
            self.device,
            sigma=self.cfg_task.imitation_sigma,
        )

        self.prev_fingertip_midpoint_pos = torch.cat(
            (self.prev_fingertip_midpoint_pos[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1
        )

        # lateral alignment: horizontal offset of the plug from its seated pose, in the socket frame
        rel_xy = torch_utils.quat_rotate(torch_utils.quat_conjugate(self.fixed_quat), self.held_pos - self.fixed_pos)[:, :2]
        e2 = rel_xy.norm(dim=-1) ** 2
        for name, sigma in zip(("lateral_coarse", "lateral_mid", "lateral_fine"), self.cfg_task.lateral_sigmas):
            rew_dict[name] = torch.exp(-e2 / sigma**2)

        rew_buf = (
            self.cfg_task.sdf_rwd_scale * rew_dict["sdf"]
            + self.cfg_task.imitation_rwd_scale * rew_dict["imitation"]
            + self.cfg_task.success_rwd_scale * rew_dict["curr_successes"]
            + sum(
                w * rew_dict[name]
                for name, w in zip(("lateral_coarse", "lateral_mid", "lateral_fine"), self.cfg_task.lateral_rwd_scales)
            )
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self.cfg_task.if_logging_eval:
            self.held_asset_pose_log = torch.cat(
                [self.held_asset_pose_log, torch.cat([self.held_pos, self.held_quat], dim=1)], dim=0
            )
            self.fixed_asset_pose_log = torch.cat(
                [self.fixed_asset_pose_log, torch.cat([self.fixed_pos, self.fixed_quat], dim=1)], dim=0
            )

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(
            1
        )  # (num_envs, 1, 3)
        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(
            prev_fingertip_midpoint_pos, self.cfg_task.num_point_robot_traj, dim=1
        )  # (num_envs, num_point_robot_traj, 3)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """Define grasp pose for plug and move gripper to pose."""

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            gripper_goal_quat,
            gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )
        gripper_goal_quat, gripper_goal_pos = self._raise_grasp(gripper_goal_quat, gripper_goal_pos)

        if self.grasp_perturb:
            gripper_goal_quat, gripper_goal_pos = self._perturb_grasp(gripper_goal_quat, gripper_goal_pos)
            self.reset_hold_quat = gripper_goal_quat.clone()

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        self.ctrl_target_fingertip_midpoint_quat = gripper_goal_quat.clone()

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def _raise_grasp(self, g_quat, g_pos):
        """The nominal fingertip grasp pose moved back by grasp_raise along its approach axis (fingertip z)."""
        if not self.grasp_raise:
            return g_quat, g_pos
        n = g_pos.shape[0]
        d = torch.tensor([0.0, 0.0, -self.grasp_raise], device=self.device).expand(n, 3)
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
        return torch_utils.tf_combine(g_quat, g_pos, ident, d)

    def _perturb_grasp(self, g_quat, g_pos):
        """Offset the open gripper from the nominal grasp before the fingers close (2026-10-02 retraining): in the
        fingertip frame, 0 along the closing axis (finger to finger, from the finger bodies), +-approach along the
        approach axis, +-width along the remaining (pad width) axis; rotations +-deg about the closing axis and about
        the approach axis. The realized grasp after closing is measured as before (D12)."""
        n, dev = self.num_envs, self.device
        lf = self._robot.data.body_pos_w[:, self.left_finger_body_idx]
        rf = self._robot.data.body_pos_w[:, self.right_finger_body_idx]
        hand_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]  # live, same instant as the finger bodies
        c = torch_utils.quat_rotate_inverse(hand_quat, lf - rf)
        c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        a = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        w = torch.cross(a, c, dim=-1)
        w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        u = lambda r: (2 * torch.rand(n, device=dev) - 1) * r
        d_pos = a * u(self.grasp_perturb_approach).unsqueeze(-1) + w * u(self.grasp_perturb_width).unsqueeze(-1)
        r = math.radians(self.grasp_perturb_deg)
        d_quat = torch_utils.quat_mul(torch_utils.quat_from_angle_axis(u(r), c), torch_utils.quat_from_angle_axis(u(r), a))
        self.grasp_perturb_sample = {"pos_m": d_pos, "quat": d_quat, "closing_axis_hand": c}
        return torch_utils.tf_combine(g_quat, g_pos, d_quat, d_pos)

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.50:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.gripper_open_width
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):
        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        fixed_state[:, 2] += self.cfg_task.fixed_asset_z_offset
        fixed_state[:, 2] += self.socket_base_lift

        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):
        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp
        if pre_grasp and self.cfg_task.if_sbc:
            n, dev, cfg = self.num_envs, self.device, self.cfg_task
            K = cfg.start_curriculum_levels
            lvl = torch.full((n,), float(self.curr_level), device=dev)
            replay = torch.rand(n, device=dev) < cfg.start_curriculum_replay
            lvl = torch.where(replay, torch.rand(n, device=dev) * self.curr_level, lvl)
            above = torch.rand(n, device=dev) >= cfg.start_curriculum_p_in_max * (1.0 - lvl / K)
            depth = self.disassembly_dists
            lo_n, hi_n = cfg.start_above_near
            lo_a, hi_a = cfg.start_above_h_range
            u = torch.rand(n, device=dev)
            h_above = torch.where(torch.rand(n, device=dev) < 0.5, lo_n + (hi_n - lo_n) * u, hi_n + (hi_a - hi_n) * u)
            # a share lvl / K of the above-hole starts at the eval start height (curriculum off: freespace range above
            # the entrance), so that level K is the eval / collection start distribution
            at_eval = torch.rand(n, device=dev) < lvl / K
            h_above = torch.where(at_eval, torch.full_like(h_above, cfg.curriculum_freespace_range), h_above)
            in_lo = torch.minimum(torch.full_like(depth, cfg.start_in_min_disp), 0.5 * depth)
            u2 = torch.rand(n, device=dev)
            d_in = torch.where(torch.rand(n, device=dev) < 0.5, depth * (2.0 / 3.0) + depth / 3.0 * u2, in_lo + (depth - in_lo) * u2)
            self.curriculum_disp = torch.where(above, depth + h_above, d_in)
            r_xy = torch.tensor(cfg.held_asset_init_pos_noise[:2], dtype=torch.float32, device=dev)[None, :] * (lvl / K)[:, None]
            xy = (2.0 * torch.rand((n, 2), device=dev) - 1.0) * r_xy
            self.held_pos_init_rand = torch.zeros((n, 3), device=dev)
            self.held_pos_init_rand[:, :2] = torch.where(above[:, None], xy, torch.zeros_like(xy))
            self.start_above, self.start_replay = above, replay
        elif pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
            held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            held_asset_init_pos_rand = torch.tensor(
                self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
            )
            self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        held_state[env_ids, 2] += self.curriculum_disp

        plug_in_freespace_idx = torch.argwhere(self.curriculum_disp > self.disassembly_dists)
        held_state[plug_in_freespace_idx, :2] += self.held_pos_init_rand[plug_in_freespace_idx, :2]

        self._held_asset.write_root_state_to_sim(held_state)
        self._held_asset.reset()

        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self.randomize_held_initial_state(env_ids, pre_grasp=True)

        self._move_gripper_to_grasp_pose(env_ids)

        self.randomize_held_initial_state(env_ids, pre_grasp=False)

        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        # Close the gripper until the fingers stop moving (at least grasp_min_time, at most grasp_max_time), with
        # gravity still off. The vendored code closed for a fixed 0.25 s: for wide plugs (fingers opened ~13-26 mm)
        # the fingers were still closing when the episode started, and the plug moved in the hand during the first
        # ~10 policy steps.
        grasp_time = 0.0
        prev_finger_pos = self.joint_pos[:, 7:9].clone()
        while True:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()
            finger_pos = self.joint_pos[:, 7:9]
            settled = bool((finger_pos - prev_finger_pos).abs().max() < self.cfg_task.grasp_settle_tol)
            prev_finger_pos = finger_pos.clone()
            if (grasp_time >= self.cfg_task.grasp_min_time and settled) or grasp_time >= self.cfg_task.grasp_max_time:
                break

        # realized grasp: G = held^-1 * fingertip * (robot_to_gripper, palm_to_finger_center)^-1, so that
        # held * G * (robot_to_gripper, palm_to_finger_center) reproduces the fingertip pose after closing
        held_inv_quat, held_inv_pos = torch_utils.tf_inverse(self.held_quat, self.held_pos)
        r2g_inv_quat, r2g_inv_pos = torch_utils.tf_inverse(self.robot_to_gripper_quat, self.palm_to_finger_center)
        q, t = torch_utils.tf_combine(held_inv_quat, held_inv_pos, self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
        self.realized_grasp_quat_local, self.realized_grasp_pos_local = torch_utils.tf_combine(q, t, r2g_inv_quat, r2g_inv_pos)
        # refresh the goal with the realized grasp (the last _compute_intermediate_values ran before it was measured)
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.realized_grasp_quat_local, self.realized_grasp_pos_local
        )
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.gripper_goal_quat, self.gripper_goal_pos, self.robot_to_gripper_quat, self.palm_to_finger_center
        )

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
