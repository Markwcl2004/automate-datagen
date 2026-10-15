"""AutoMate assembly env with a scripted pick before the insertion policy (feasibility study, 2026-10-01).

After the vendored reset has placed the socket and computed the policy's start pose (plug just above the socket
entrance), the plug is put upright on the table elsewhere, falls and settles under gravity, and the arm picks it with
the task's policy grasp (plug_grasps.json, vertical approach), lifts it, carries it to the recorded start pose and
hands over to the policy. Everything is physical: the plug is held by finger friction only (no attachment, no pose
writes after the plug is released on the table), gravity is on for the plug, the arm moves under the env's own
task-space impedance controller with interpolated targets.

Per-env results of the last reset are in ``self.pick_info`` (tensors)."""
import math

import carb
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils

from . import factory_control as fc
from .assembly_env import AssemblyEnv

TABLE_TOP_Z = -0.003  # env frame: the socket stands on the table (D11), socket origin = table top


def plug_bottom_z(obj_path: str) -> float:
    """Lowest vertex z of the plug mesh (plug frame, m)."""
    zs = [float(line.split()[3]) for line in open(obj_path) if line.startswith("v ")]
    return min(zs)


class PickAssemblyEnv(AssemblyEnv):
    # pick parameters (overridable as attributes before the first reset)
    pick_enabled = True
    place_dist = (0.18, 0.28)  # |y| distance of the plug from the socket (left or right side), m
    place_dx = (-0.08, 0.08)  # x offset from the socket, m
    place_yaw_deg = 45.0  # plug yaw on the table, uniform +-
    drop_height = 0.002  # plug released this high above the table, m
    settle_s = 0.5
    approach_h = 0.08  # pre-grasp / lift height above the grasp pose, m
    max_lin = 0.006  # m per control step (15 Hz policy rate -> 9 cm/s)
    max_ang = 0.05  # rad per control step
    topple_deg = 20.0
    pick_carry = True  # False: stop after the lift (grasp feasibility only, no carry / hand-over)
    # finger opening (per finger, m) while approaching from above. The vendored gripper_open_width (plug y extent /
    # 1.25) only works for its teleported reset: descending onto the plug, the fingers then land on wider parts above
    # the grasp (2026-10-01 batch3: 25 tasks stopped 15-39 mm high, plug untouched). A real pick opens fully.
    approach_open = 0.04
    frame_hook = None  # callable(env) after every scripted physics step (video capture), None = off

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        obj = f"{self.cfg_task.assembly_dir}{self.cfg_task.held_asset_cfg.obj_path}"
        self.plug_bottom_local = plug_bottom_z(obj)
        self.pick_info = {}

    # ---- low-level scripted control (env's own impedance controller, no policy action clipping) ----
    def _physics_steps(self, tgt_pos, tgt_quat, grip, n):
        for _ in range(n):
            self.ctrl_target_fingertip_midpoint_pos = tgt_pos
            self.ctrl_target_fingertip_midpoint_quat = tgt_quat
            self.ctrl_target_gripper_dof_pos = grip
            self.generate_ctrl_signals()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self._compute_intermediate_values(dt=self.physics_dt)
            if self.frame_hook is not None:  # video capture of the scripted phases (pick_video.py)
                self.frame_hook(self)

    def _servo(self, goal_pos, goal_quat, grip, tol_pos=0.002, tol_ang=0.03, max_iters=300, extra_iters=3):
        """Move the fingertip toward (goal_pos, goal_quat) with per-step clipped targets until every env is within
        tolerance (or max_iters control steps), then hold for extra_iters control steps."""
        done_iters = 0
        for it in range(max_iters):
            pos_err, aa_err = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos, fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=goal_pos, ctrl_target_fingertip_midpoint_quat=goal_quat,
                jacobian_type="geometric", rot_error_type="axis_angle")
            d = pos_err.norm(dim=-1, keepdim=True)
            step = pos_err * torch.clamp(self.max_lin / d.clamp(min=1e-9), max=1.0)
            ang = aa_err.norm(dim=-1)
            axis = aa_err / ang.clamp(min=1e-9).unsqueeze(-1)
            rot = torch_utils.quat_from_angle_axis(torch.clamp(ang, max=self.max_ang), axis)
            rot = torch.where(ang.unsqueeze(-1) > 1e-6, rot, torch.tensor([1.0, 0, 0, 0], device=self.device).expand_as(rot))
            tgt_pos = self.fingertip_midpoint_pos + step
            tgt_quat = torch_utils.quat_mul(rot, self.fingertip_midpoint_quat)
            # near the goal command the goal itself (the impedance controller settles the remainder)
            near = (d.squeeze(-1) < 3 * self.max_lin) & (ang < 3 * self.max_ang)
            tgt_pos = torch.where(near.unsqueeze(-1), goal_pos, tgt_pos)
            tgt_quat = torch.where(near.unsqueeze(-1), goal_quat, tgt_quat)
            self._physics_steps(tgt_pos, tgt_quat, grip, self.cfg.decimation)
            if bool(((d.squeeze(-1) < tol_pos) & (ang < tol_ang)).all()):
                done_iters += 1
                if done_iters > extra_iters:
                    return it
        return max_iters

    def _close_gripper(self, max_time=None):
        """Close until the fingers stop moving, the same rule as the vendored reset (grasp_min/max_time)."""
        max_time = self.cfg_task.grasp_max_time if max_time is None else max_time
        t, prev = 0.0, self.joint_pos[:, 7:9].clone()
        hold_pos, hold_quat = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        while True:
            self._physics_steps(hold_pos, hold_quat, 0.0, 1)
            t += self.physics_dt
            settled = bool((self.joint_pos[:, 7:9] - prev).abs().max() < self.cfg_task.grasp_settle_tol)
            prev = self.joint_pos[:, 7:9].clone()
            if (t >= self.cfg_task.grasp_min_time and settled) or t >= max_time:
                return t

    def _grasp_target(self, held_pos, held_quat):
        """Fingertip pose of the policy grasp for a plug at (held_pos, held_quat), as _move_gripper_to_grasp_pose."""
        q, p = torch_utils.tf_combine(held_quat, held_pos, self.plug_grasp_quat_local, self.plug_grasp_pos_local)
        return self._raise_grasp(*torch_utils.tf_combine(q, p, self.robot_to_gripper_quat, self.palm_to_finger_center))

    @staticmethod
    def _tilt_deg(quat):
        w, x, y, z = quat.unbind(-1)
        cos_t = 1 - 2 * (x * x + y * y)  # z component of the body z axis
        return torch.rad2deg(torch.acos(cos_t.clamp(-1.0, 1.0)))

    # ---- reset ----
    def randomize_initial_state(self, env_ids):
        if not self.pick_enabled:
            return super().randomize_initial_state(env_ids)
        # the reference grasp below must be the unperturbed one even if the training perturbation is switched on
        gp, self.grasp_perturb = self.grasp_perturb, False
        super().randomize_initial_state(env_ids)  # socket placed, plug at the policy start pose in the closed hand
        self.grasp_perturb = gp
        n, dev = self.num_envs, self.device
        info = {}
        start_pos, start_quat = self.held_pos.clone(), self.held_quat.clone()  # policy start pose (env frame)
        # the grasp the policy trains with: the vendored reset's realized grasp after closing (D12), not the nominal
        # plug_grasps.json pose (closing squeezes the plug 0-5.75 mm off it, deterministic per task)
        ref_q, ref_p = self.realized_grasp_quat_local.clone(), self.realized_grasp_pos_local.clone()

        # 1. plug upright on the table beside the socket, released 2 mm above it
        side = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        dy = side * (self.place_dist[0] + (self.place_dist[1] - self.place_dist[0]) * torch.rand(n, device=dev))
        dx = self.place_dx[0] + (self.place_dx[1] - self.place_dx[0]) * torch.rand(n, device=dev)
        yaw = torch.deg2rad((2 * torch.rand(n, device=dev) - 1) * self.place_yaw_deg)
        plug_q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        plug_p = torch.stack([self.fixed_pos[:, 0] + dx, self.fixed_pos[:, 1] + dy,
                              torch.full_like(dx, TABLE_TOP_Z - self.plug_bottom_local + self.drop_height)], -1)
        # open the hand first (the plug is still in it), then move the plug away
        open_w = float(self.approach_open)
        hold_pos, hold_quat = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        state = self._held_asset.data.default_root_state.clone()
        state[:, 0:3] = plug_p + self.scene.env_origins
        state[:, 3:7] = plug_q
        state[:, 7:] = 0.0
        self._held_asset.write_root_state_to_sim(state)
        self._held_asset.reset()
        self._physics_steps(hold_pos, hold_quat, open_w, 12)
        # lift the open hand clear
        up = hold_pos.clone()
        up[:, 2] += self.approach_h
        self._servo(up, hold_quat, open_w, tol_pos=0.01, max_iters=60)
        # 2. settle on the table (the plug is no longer touched by anything)
        steps = int(round(self.settle_s / self.physics_dt))
        self._physics_steps(self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone(), open_w, steps)
        info["placed_pos"], info["settled_pos"] = plug_p.clone(), self.held_pos.clone()
        info["settle_shift_m"] = (self.held_pos[:, :2] - plug_p[:, :2]).norm(dim=-1)
        info["tilt_settled_deg"] = self._tilt_deg(self.held_quat)
        info["toppled"] = info["tilt_settled_deg"] > self.topple_deg

        # 3. pick with the policy grasp from the plug's actual pose
        g_quat, g_pos = self._grasp_target(self.held_pos, self.held_quat)
        pre = g_pos.clone()
        pre[:, 2] += self.approach_h
        info["iters_pregrasp"] = self._servo(pre, g_quat, open_w)
        info["iters_descend"] = self._servo(g_pos, g_quat, open_w)
        info["fingertip_err_at_grasp_m"] = (self.fingertip_midpoint_pos - g_pos).norm(dim=-1)
        # at the grasp height first bring the fingers to the training opening (vendored gripper_open_width), so that
        # the close starts from the same gap as in the policy's training reset (2026-10-01 batch4: closing from the
        # full 4 cm opening outlasted grasp_max_time and the lift started before the fingers touched the plug)
        train_w = float(self.gripper_open_width)
        hold_p, hold_q = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        t_pre, prev = 0.0, self.joint_pos[:, 7:9].clone()
        while t_pre < 2.0:
            self._physics_steps(hold_p, hold_q, train_w, 1)
            t_pre += self.physics_dt
            moved = (self.joint_pos[:, 7:9] - prev).abs().max()
            prev = self.joint_pos[:, 7:9].clone()
            if t_pre > 0.1 and bool(moved < self.cfg_task.grasp_settle_tol):
                break
        info["preclose_s"] = t_pre
        info["finger_at_train_open_err_m"] = (self.joint_pos[:, 7:9] - train_w).abs().max(dim=-1).values
        info["plug_shift_before_close_m"] = (self.held_pos[:, :2] - info["settled_pos"][:, :2]).norm(dim=-1)
        info["close_s"] = self._close_gripper(max_time=max(2.0, self.cfg_task.grasp_max_time))
        plug_z0 = self.held_pos[:, 2].clone()
        lift = self.fingertip_midpoint_pos.clone()
        lift[:, 2] += self.approach_h
        self._servo(lift, self.fingertip_midpoint_quat.clone(), 0.0, tol_pos=0.004)
        info["plug_lifted_m"] = self.held_pos[:, 2] - plug_z0
        info["grasp_ok"] = info["plug_lifted_m"] > 0.5 * self.approach_h
        info["finger_gap_m"] = self.joint_pos[:, 7:9].sum(-1)

        # in-hand pose vs the nominal policy grasp
        held_inv_q, held_inv_p = torch_utils.tf_inverse(self.held_quat, self.held_pos)
        r2g_inv_q, r2g_inv_p = torch_utils.tf_inverse(self.robot_to_gripper_quat, self.palm_to_finger_center)
        q, t = torch_utils.tf_combine(held_inv_q, held_inv_p, self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
        rq, rp = torch_utils.tf_combine(q, t, r2g_inv_q, r2g_inv_p)
        info["grasp_pos_err_m"] = (rp - ref_p).norm(dim=-1)  # vs the policy's training grasp (realized, D12)
        dq = torch_utils.quat_mul(torch_utils.quat_conjugate(ref_q), rq)
        info["grasp_rot_err_deg"] = torch.rad2deg(2 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))
        info["grasp_pos_err_nominal_m"] = (rp - self.plug_grasp_pos_local).norm(dim=-1)  # vs plug_grasps.json
        info["train_grasp_off_nominal_m"] = (ref_p - self.plug_grasp_pos_local).norm(dim=-1)
        if not self.pick_carry:
            self.pick_info = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in info.items()}
            return

        # 4. carry: fingertip target that puts the plug at the recorded start pose, with the measured in-hand transform
        ft_inv_q, ft_inv_p = torch_utils.tf_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
        hp_q, hp_p = torch_utils.tf_combine(ft_inv_q, ft_inv_p, self.held_quat, self.held_pos)  # plug in fingertip frame
        hp_inv_q, hp_inv_p = torch_utils.tf_inverse(hp_q, hp_p)
        tgt_q, tgt_p = torch_utils.tf_combine(start_quat, start_pos, hp_inv_q, hp_inv_p)
        cruise = max(float(tgt_p[:, 2].max()), float(self.fingertip_midpoint_pos[:, 2].max())) + 0.05
        up = self.fingertip_midpoint_pos.clone()
        up[:, 2] = cruise
        self._servo(up, self.fingertip_midpoint_quat.clone(), 0.0, tol_pos=0.006)
        over = tgt_p.clone()
        over[:, 2] = cruise
        self._servo(over, tgt_q, 0.0, tol_pos=0.004)
        info["iters_place"] = self._servo(tgt_p, tgt_q, 0.0)
        self._physics_steps(tgt_p, tgt_q, 0.0, 2 * self.cfg.decimation)
        info["place_pos_err_m"] = (self.held_pos - start_pos).norm(dim=-1)
        rel = torch_utils.quat_rotate_inverse(self.fixed_quat, self.held_pos - self.fixed_pos)  # plug in the socket frame
        info["handover_height_above_entrance_m"] = rel[:, 2] - self.disassembly_dists
        info["handover_xy_offset_m"] = rel[:, :2].norm(dim=-1)
        dq = torch_utils.quat_mul(torch_utils.quat_conjugate(self.fixed_quat), self.held_quat)
        info["handover_plug_tilt_deg"] = self._tilt_deg(dq)
        info["handover_fingertip_tilt_deg"] = torch.rad2deg(torch.acos((-torch_utils.quat_rotate(
            self.fingertip_midpoint_quat, torch.tensor([0.0, 0, 1.0], device=dev).expand(n, 3))[:, 2]).clamp(-1, 1)))
        info["handover_ee_linvel"] = self.fingertip_midpoint_linvel.norm(dim=-1)
        info["handover_ee_angvel"] = self.fingertip_midpoint_angvel.norm(dim=-1)
        dq = torch_utils.quat_mul(torch_utils.quat_conjugate(start_quat), self.held_quat)
        info["place_rot_err_deg"] = torch.rad2deg(2 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))

        # 5. hand over: realized grasp and policy goal as the vendored reset computes them, zero actions / velocities
        held_inv_q, held_inv_p = torch_utils.tf_inverse(self.held_quat, self.held_pos)
        q, t = torch_utils.tf_combine(held_inv_q, held_inv_p, self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
        self.realized_grasp_quat_local, self.realized_grasp_pos_local = torch_utils.tf_combine(q, t, r2g_inv_q, r2g_inv_p)
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.realized_grasp_quat_local, self.realized_grasp_pos_local)
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.gripper_goal_quat, self.gripper_goal_pos, self.robot_to_gripper_quat, self.palm_to_finger_center)
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0
        self._set_gains(self.default_gains)
        self.pick_info = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in info.items()}
