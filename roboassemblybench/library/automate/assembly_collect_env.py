# Copyright (c) 2025, RoboAssemblyBench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""AutoMate full-process collection environment (2026-10-03): pick -> carry -> insert -> release -> return.

Every episode starts with the arm at one fixed home pose (same for every task and env) and the plug standing on the
table. Randomized per episode: the socket pose (the policy's training range) and the plug pose on the table (user
2026-10-03: no pick base, plug and socket placed at random on the table). One round = all envs together:

1. reset: vendored AutoMate reset (socket, policy start pose, realized grasp), plug put upright on the table at a
   random pose and left to settle, arm set to the home joint configuration with the gripper open; visuals randomized
   and settled;
2. scripted pick and carry in lock step (approach, grasp, lift, carry; ``PickAssemblyEnv``'s motions), recorded;
3. the insertion policy (relative observations, socket-frame actions, D16) until the plug is seated; then per env a
   scripted release (gripper opens) and return to the home pose, recorded; the episode ends at home.

Cameras (user 2026-10-03): ``observation.images.front`` = Intel RealSense D435 colour at 640 x 480, fixed in the env
frame, seeing the whole workspace (plug, gripper, socket); ``observation.images.wrist`` = RealSense D405 at
640 x 480 on ``panda_hand``. Both RGB + metric depth (``distance_to_image_plane``). Frames every ``frame_stride``
physics steps (30 FPS from 120 Hz); every captured frame is the state and images of one physics step.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable

PROF = defaultdict(float)  # seconds per component, filled when COLLECT_PROFILE=1
PROFILE = bool(os.environ.get("COLLECT_PROFILE"))


def _sync():
    if PROFILE:
        torch.cuda.synchronize()

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from .assembly_env import AssemblyEnv
from .assembly_env_cfg import AssemblyEnvCfg
from . import factory_control as fc
from .pick_env import TABLE_TOP_Z, PickAssemblyEnv

CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")
# episode phases (index = phase_index in the recorded data)
PHASES = ("approach", "grasp", "lift", "carry", "insert", "release", "return")
PH = {name: i for i, name in enumerate(PHASES)}
FINGER_OPEN_M = 0.04  # Franka finger joint travel; gripper fraction = finger joint / 0.04
# fixed home pose (every task and env): Franka "ready" joint configuration, gripper open
HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
# plug placement on the table (env frame): uniform in this region, at least PLUG_SOCKET_MIN_DIST from the socket centre
PLUG_REGION_X = (0.40, 0.75)
PLUG_REGION_Y = (-0.32, 0.32)
PLUG_SOCKET_MIN_DIST = 0.15


def k_matrix(fx, fy, cx, cy) -> list:
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


# Intel RealSense intrinsics at 640 x 480 (user 2026-10-03: global camera D435, wrist camera D405), rendered as an
# ideal pinhole (no distortion) with square pixels and the principal point at the image centre:
# - D435 colour, 640 x 480: public SDK readouts of real units fx 615.3-616.6, fy 615.4-616.8, ppx 313-318, ppy 241-252
#   (ROS answers 363236, librealsense #3986) -> fx = fy = 616.7 (HFOV 54.8 deg, VFOV 42.5 deg)
# - D405 colour: SDK readout at 848 x 480 fx 435.29, fy 434.67, ppx 431.52, ppy 243.55 (realsense-ros #3289); the
#   640 x 480 mode is the centred horizontal crop -> fx = fy = 435.0 (HFOV 72.6 deg, VFOV 57.8 deg). The real D405
#   colour stream has a small inverse Brown-Conrady distortion (k1 ~ -0.05), not modelled.
CAMERA_MODELS = {
    "observation.images.front": {"model": "Intel RealSense D435 (colour)", "K_fx_fy_cx_cy": (616.7, 616.7, 320.0, 240.0),
                                 "source": "public SDK intrinsics of real D435 units at 640x480 (fx 615.3-616.6, fy 615.4-616.8); square pixels, centred principal point"},
    "observation.images.wrist": {"model": "Intel RealSense D405 (colour)", "K_fx_fy_cx_cy": (435.0, 435.0, 320.0, 240.0),
                                 "source": "public SDK intrinsics of a real D405 at 848x480 (fx 435.29, fy 434.67), 640x480 = centred horizontal crop; square pixels, centred principal point; real lens distortion (inverse Brown-Conrady, k1 ~ -0.05) not modelled"},
}


def usd_value(v):
    """A USD attribute value as JSON (asset paths as their path string, vectors as lists)."""
    from pxr import Sdf

    if isinstance(v, Sdf.AssetPath):
        return v.path
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    try:
        return [float(x) for x in v]
    except TypeError:
        return str(v)


def asset_materials(stage, root: str) -> list:
    """The materials bound under root (Gprims and their GeomSubsets, instance proxies included) with every authored
    shader input value, once per material. Call before the collector binds its own materials over the asset."""
    from pxr import Usd, UsdGeom, UsdShade

    out, seen = [], set()
    for q in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not (q.IsA(UsdGeom.Gprim) or q.IsA(UsdGeom.Subset)):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()
        if not mat or str(mat.GetPath()) in seen:
            continue
        seen.add(str(mat.GetPath()))
        shaders = []
        for sp in Usd.PrimRange(mat.GetPrim(), Usd.TraverseInstanceProxies()):
            if sp.IsA(UsdShade.Shader):
                sh = UsdShade.Shader(sp)
                src = sh.GetSourceAsset("mdl")
                shaders.append({"shader": sp.GetName(), "id": sh.GetShaderId(), "mdl_source": src.path if src else None,
                                "inputs": {i.GetBaseName(): usd_value(i.Get()) for i in sh.GetInputs() if i.Get() is not None}})
        path = str(mat.GetPath())
        out.append({"material": path.split("/Table/", 1)[-1] if "/Table/" in path else path, "shaders": shaders})
    return out


def look_at_quat(eye, target, up, convention: str) -> tuple[float, float, float, float]:
    """(w, x, y, z) rotating the camera frame so it looks from ``eye`` at ``target``; ``up`` is the image-up hint.

    ``ros``: x right, y down, z forward. ``world``: x forward, y left, z up.
    """
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    if convention == "ros":
        rot = np.stack([right, -true_up, forward], axis=1)
    elif convention == "world":
        rot = np.stack([forward, -right, true_up], axis=1)
    else:
        raise ValueError(convention)
    w = np.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) / 2.0, rot[2, 1] - rot[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - rot[0, 0] + rot[1, 1] - rot[2, 2])) / 2.0, rot[0, 2] - rot[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - rot[0, 0] - rot[1, 1] + rot[2, 2])) / 2.0, rot[1, 0] - rot[0, 1])
    return (float(w), float(x), float(y), float(z))


# Front (global) camera, env frame, fixed for every task: in front of the table, looking back at the robot over the
# whole workspace (plug region +-0.32 m in y, socket around x = 0.6 m, the arm's home pose ~0.5 m above the table)
FRONT_EYE = (1.40, 0.0, 0.70)
FRONT_TARGET = (0.55, 0.0, 0.10)
# Wrist camera: panda_hand frame (z points from the hand to the fingertips, fingers open along y). Mounted beside the
# hand body, looking at the point between the fingertips.
WRIST_EYE = (0.07, 0.0, 0.03)
WRIST_TARGET = (0.0, 0.0, 0.11)


@configclass
class AssemblyCollectEnvCfg(AssemblyEnvCfg):
    frame_stride: int = 4  # physics steps between dataset frames (120 Hz / 4 = 30 FPS)
    image_width: int = 640
    image_height: int = 480
    camera_warmup_renders: int = 30
    # after a reset (new visuals) render until every env's front / wrist brightness is still. After a light change the
    # RTX image brightens / darkens fast for ~25 renders and then creeps for a few hundred more (measured 2026-10-01 in
    # the collector: mean front 116.6 -> 124.7 at 50 renders -> 127.3 at 200 -> 127.7 at 275; auto exposure is off,
    # turning off sampled direct lighting does not change it). Single renders are noisy (per env up to ~0.5 % per
    # render in bright scenes even when settled), so the mean image value of the last settle_block renders is compared
    # with the block before, per env and camera and for the mean over envs (which catches the slow creep).
    max_settle_renders: int = 300
    # profiles that change the lights every round wait at least this long before the settle test may stop: the
    # slow creep after a light change takes ~275 renders (collector trace 2026-10-01); 0 = no minimum
    settle_min_renders: int = 0
    settle_block: int = 20
    # the systematic creep is caught by the mean over envs; per env the soft shadows of a low key light keep the block
    # means moving by 1-3 % while settled (validated 2026-10-01, mixed 00143), so per env only gross changes block
    settle_rel_mean: float = 0.002  # mean over envs, per camera, changed less than 0.2 % between blocks ...
    settle_rel_p95: float = 0.02  # ... 95 % of the (env, camera) block means less than 2 % ...
    settle_rel_max: float = 0.05  # ... and none more than 5 %
    # visual randomization assets (hidden / nominal unless a visual hook changes them): one table material per env and
    # up to max_distractors visual-only boxes / cylinders per env (no collision, no rigid body)
    visual_assets: bool = True
    max_distractors: int = 4
    # bind the per-env table materials at scene setup (material bindings made after the simulation has started do not
    # reach the renderer; later colour changes do). Off keeps the table's own look (every profile but table_color).
    bind_table_material: bool = False
    # texture / scene / mixed profiles (set by the collector before the env is created): textured table / floor
    # materials and per-env warehouse props are created at scene setup; per round only values / poses change.
    # scene_props: [{"id", "url", "category", "gprims", "lo", "hi"}] (props manifest; gprims = the complete asset)
    table_texture: bool = False
    floor_texture: bool = False
    texture_dir: str = ""
    scene_props: list = []

    front_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/FrontCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=FRONT_EYE, rot=look_at_quat(FRONT_EYE, FRONT_TARGET, (0, 0, 1), "ros"), convention="ros"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            k_matrix(*CAMERA_MODELS["observation.images.front"]["K_fx_fy_cx_cy"]), width=640, height=480,
            clipping_range=(0.01, 10.0), focal_length=24.0),
        width=640,
        height=480,
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=WRIST_EYE, rot=look_at_quat(WRIST_EYE, WRIST_TARGET, (1, 0, 0), "ros"), convention="ros"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            k_matrix(*CAMERA_MODELS["observation.images.wrist"]["K_fx_fy_cx_cy"]), width=640, height=480,
            clipping_range=(0.01, 10.0), focal_length=12.0),
        width=640,
        height=480,
    )

    # net contact force (N, world frame) on each finger and on the plug. Not filtered by partner body: PhysX GPU
    # contact filters need collider prims and the plug's colliders sit below its root; while the plug is held the
    # fingers touch nothing else, so the finger forces are the grasp forces.
    contact_left_finger: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/panda_leftfinger")
    contact_right_finger: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/panda_rightfinger")
    # the plug USD puts its rigid body one level down: HeldAsset/a_<id>_plug
    contact_plug: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/HeldAsset/.*")

    # after insertion (per env, scripted): open the gripper for release_s, lift lift_m, return to the home pose, stay
    # there home_still_s; the episode ends at home
    release_s: float = 0.5
    lift_m: float = 0.08
    home_still_s: float = 0.5
    # transport moves (home -> pre-grasp, carry, return home) run at transport_speedup x PickAssemblyEnv's rates
    # (max_lin 6 mm per control step = 9 cm/s); the moves near the plug and the socket keep the base rates
    transport_speedup: float = 2.0
    # PickAssemblyEnv._servo stall rule (AssemblyCollectEnv._servo): no progress of (m, rad) over stall_window control
    # steps = stalled; stalled farther than stall_fail (m, rad) from the waypoint = the pick fails
    # _servo(land=True): per control step at most land_gain x the remaining error (the descent onto the plug)
    land_gain: float = 0.3
    stall_window: int = 15
    stall_far_rate: float = 0.1
    stall_min_progress: tuple = (0.0005, 0.005)
    stall_fail: tuple = (0.01, 0.1)
    home_tol_pos: float = 0.004
    home_tol_ang: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        # contact reporting only (no effect on the dynamics)
        self.robot.spawn.activate_contact_sensors = True
        task = self.tasks[self.task_name]
        task.held_asset.spawn.activate_contact_sensors = True
        task.fixed_asset.spawn.activate_contact_sensors = True
        # sim.render_interval stays 1 (as in training). Isaac Lab sets the Kit rendering dt to physics_dt x
        # render_interval, and every sim.step(render=True) -- used by the AssemblyEnv reset (move to the grasp pose,
        # close the gripper) -- advances that many physics steps: render_interval = 4 ran the reset 4x too fast and
        # the policy failed (00014: 0/32 strict successes vs 20/32). The camera frames are rendered explicitly in
        # step() instead.
        self.front_camera.width = self.wrist_camera.width = self.image_width
        self.front_camera.height = self.wrist_camera.height = self.image_height
        self.num_rerenders_on_reset = 2  # also lets a new lighting setup settle before the reset frame
        # neighbouring envs beyond the cameras' far clipping plane (10 m); training keeps its own spacing
        self.scene.env_spacing = 12.0
        # a round ends when the collector resets: no time-out resets inside step()
        self.episode_length_s = 600.0
        # the plug falls, stands and is lifted under gravity (the vendored reset turns global gravity off and back on)
        task.held_asset.spawn.rigid_props.disable_gravity = False


class AssemblyCollectEnv(PickAssemblyEnv):
    """``frame_callback(env, frame)`` is called for every dataset frame: once right after each reset (tag
    ``"reset"``: the arm at home, the plug on the table), during the scripted pick / carry inside the reset, and after
    every ``frame_stride`` physics steps of a policy step (tag ``"step"``). ``frame`` holds per-env tensors (see
    :meth:`capture`)."""

    cfg: AssemblyCollectEnvCfg

    def __init__(self, cfg: AssemblyCollectEnvCfg, render_mode: str | None = None, **kwargs):
        self.frame_callback: Callable | None = None
        self.visual_hook: Callable | None = None  # visual_hook(env) sets lights / materials for the next episodes
        self._last_policy_action = None
        self._cameras_warm = False
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.decimation % self.cfg.frame_stride != 0:
            raise ValueError(f"decimation {self.cfg.decimation} must be a multiple of frame_stride {self.cfg.frame_stride}")
        n, dev = self.num_envs, self.device
        self.grasp_perturb = False  # collection: the real pick's own grasp errors only
        self.phase = torch.full((n,), PH["approach"], dtype=torch.long, device=dev)
        self.grip_cmd = torch.ones(n, device=dev)  # commanded finger opening as a fraction of 0.04 m
        # per-env script after the insertion: 0 policy, 1 release, 2 lift, 3 to home, 4 at home, 5 done, 6 frozen
        self.script = torch.zeros(n, dtype=torch.long, device=dev)
        self.script_t = torch.zeros(n, device=dev)
        self.script_pos = torch.zeros((n, 3), device=dev)
        self.script_quat = torch.zeros((n, 4), device=dev)
        self.script_grip = torch.zeros(n, device=dev)
        self.script_ref_p = torch.zeros((n, 3), device=dev)  # moving reference of the scripted envs (advances toward the goal)
        self.script_ref_q = torch.zeros((n, 4), device=dev)
        self.pick_failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.servo_stalled = torch.zeros(n, dtype=torch.bool, device=dev)  # a scripted waypoint not reached (see _servo)
        self.servo_stall_log = []
        self.home_pos = None  # fingertip pose at HOME_JOINTS (env frame), measured at the first reset
        self.home_quat = None
        self._hook_count = 0
        self.frame_hook = self._scripted_frame_hook
        self.layout = {}

    def _setup_scene(self):
        super()._setup_scene()
        self._align_table_visuals()
        self.front_camera_aim = {"eye_env_m": list(FRONT_EYE), "target_env_m": list(FRONT_TARGET), "fixed": True}
        self._front_camera = TiledCamera(self.cfg.front_camera)
        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.scene.sensors["front_camera"] = self._front_camera
        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self._contact = {
            "left_finger": ContactSensor(self.cfg.contact_left_finger),
            "right_finger": ContactSensor(self.cfg.contact_right_finger),
            "plug": ContactSensor(self.cfg.contact_plug),
        }
        for name, sensor in self._contact.items():
            self.scene.sensors[f"contact_{name}"] = sensor
        import omni.usd

        # the asset's own table / ground materials, before the collector binds its materials over them
        stage = omni.usd.get_context().get_stage()
        self.default_materials = {"table": asset_materials(stage, "/World/envs/env_0/Table"),
                                  "ground": asset_materials(stage, "/World/ground")}
        if self.cfg.visual_assets:
            self._create_visual_assets()
        if self.cfg.scene_props:
            self._create_scene_props()
        # extra light for the lighting profile, one per env at an env-local position (off unless a visual randomizer
        # turns it on)
        key = sim_utils.SphereLightCfg(intensity=0.0, radius=0.1, color=(1.0, 1.0, 1.0))
        key.func("/World/envs/env_.*/KeyLight", key, translation=(0.6, 0.0, 1.2))

    # SeattleLabTable as spawned: visual top surface -1.6 mm, collision cube top -3.0 mm (env frame, ray tests)
    TABLE_VISUAL_DROP = -0.0014

    def _align_table_visuals(self):
        """Lower each env's table visuals onto the collider top, so rendered objects stand on the surface they rest on
        (the socket sits on the collider, D11). A translate op first in the op order acts in the parent (table) frame,
        whose only rotation is about z."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for i in range(self.num_envs):
            visuals = [c for c in stage.GetPrimAtPath(f"/World/envs/env_{i}/Table").GetChildren() if c.GetName().lower() == "visuals"]
            if not visuals:
                raise RuntimeError(f"no table visuals under /World/envs/env_{i}/Table")
            xf = UsdGeom.Xformable(visuals[0])
            ops = xf.GetOrderedXformOps()
            if any(o.GetOpName() == "xformOp:translate:alignCollider" for o in ops):
                continue  # the env tables share env_0's table spec: done once for all
            op = xf.AddTranslateOp(opSuffix="alignCollider")
            op.Set(Gf.Vec3d(0.0, 0.0, self.TABLE_VISUAL_DROP))
            xf.SetXformOpOrder([op] + list(ops))

    def _create_visual_assets(self):
        """Per-env table material (bound over the instanced table visuals) and hidden distractor shapes."""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Scope.Define(stage, "/World/Looks")
        table_uv = self._uv_name(stage.GetPrimAtPath("/World/envs/env_0/Table")) if self.cfg.table_texture else None
        if self.cfg.table_texture and table_uv is None:
            raise RuntimeError("texture profile: the table meshes have no texture coordinates")
        for i in range(self.num_envs):
            # the table visuals are instanced: a material bound outside the instance does not reach its meshes.
            # De-instance the visuals only (before the simulation starts; the collisions stay instanced).
            for child in stage.GetPrimAtPath(f"/World/envs/env_{i}/Table").GetChildren():
                if child.GetName().lower() == "visuals" and child.IsInstanceable():
                    child.SetInstanceable(False)
            mat = UsdShade.Material.Define(stage, f"/World/Looks/CollectTable_{i}")
            shader = UsdShade.Shader.Define(stage, f"/World/Looks/CollectTable_{i}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.2, 0.2))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            if self.cfg.table_texture:
                self._texture_network(stage, f"/World/Looks/CollectTable_{i}", shader, table_uv, self.texture_files("table")[0])
            for k in range(self.cfg.max_distractors):
                # one slot = an Xform with a unit cube ("box") and a unit-height cylinder of diameter 1 ("cylinder") under
                # it; the randomizer poses / scales the Xform and shows one of the two (type drawn per distractor,
                # 2026-10-10: types used to be fixed by slot parity, boxes : cylinders = 60 : 40)
                path = f"/World/envs/env_{i}/Distractor_{k}"
                slot = UsdGeom.Xform.Define(stage, path)
                box = UsdGeom.Cube.Define(stage, f"{path}/box")
                box.CreateSizeAttr(1.0)
                cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cylinder")
                cyl.CreateRadiusAttr(0.5)
                cyl.CreateHeightAttr(1.0)
                slot.MakeInvisible()
                dm = UsdShade.Material.Define(stage, f"/World/Looks/CollectDistractor_{i}_{k}")
                ds = UsdShade.Shader.Define(stage, f"/World/Looks/CollectDistractor_{i}_{k}/Shader")
                ds.CreateIdAttr("UsdPreviewSurface")
                ds.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
                ds.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                dm.CreateSurfaceOutput().ConnectToSource(ds.ConnectableAPI(), "surface")
                for g in (box, cyl):
                    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(dm)
        if self.cfg.floor_texture:
            self._create_floor_material(stage)
        self._table_material_bound = False
        if self.cfg.bind_table_material:
            self.bind_table_materials()

    def texture_files(self, group: str) -> list:
        """Absolute paths of the upstream textures of one group (table / wall / floor), in manifest order."""
        import json
        import os

        man = json.load(open(os.path.join(self.cfg.texture_dir, "manifest.json")))
        return [os.path.join(self.cfg.texture_dir, e["file"]) for e in man[group]]

    @staticmethod
    def _uv_name(root_prim):
        """Name of the first texture-coordinate primvar on a mesh under root_prim (instance proxies included)."""
        from pxr import Sdf, Usd, UsdGeom

        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
                    if pv.GetTypeName() in (Sdf.ValueTypeNames.TexCoord2fArray, Sdf.ValueTypeNames.Float2Array):
                        return pv.GetPrimvarName()
        return None

    @staticmethod
    def _texture_network(stage, mat_path, surface, uv, first_file):
        """UsdPrimvarReader (uv) -> UsdTransform2d (tiling) -> UsdUVTexture (file, tint) -> surface diffuseColor."""
        from pxr import Gf, Sdf, UsdShade

        st = UsdShade.Shader.Define(stage, f"{mat_path}/Uv")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv)
        tiling = UsdShade.Shader.Define(stage, f"{mat_path}/Tiling")
        tiling.CreateIdAttr("UsdTransform2d")
        tiling.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        tiling.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
        tex = UsdShade.Shader.Define(stage, f"{mat_path}/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(first_file))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.0, 1.0, 1.0, 1.0))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(tiling.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))

    def _create_floor_material(self, stage):
        """One textured material bound over the (shared) ground plane."""
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        ground = stage.GetPrimAtPath("/World/ground")
        for prim in list(Usd.PrimRange(ground)):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        uv = self._uv_name(ground)
        if uv is None:
            raise RuntimeError("texture profile: the ground plane has no texture coordinates")
        mat = UsdShade.Material.Define(stage, "/World/Looks/CollectFloor")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/CollectFloor/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._texture_network(stage, "/World/Looks/CollectFloor", shader, uv, self.texture_files("floor")[0])
        bound = 0
        for prim in Usd.PrimRange(ground):
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                bound += 1
        if bound == 0:
            raise RuntimeError("texture profile: no ground plane mesh")

    GROUND_Z = -0.4  # spawn_ground_plane translation in AssemblyEnv._setup_scene

    KEEP_OUT_LO, KEEP_OUT_HI = (-0.6, -0.9, -0.35), (1.6, 0.9, 1.4)  # env frame, m: arm, table, both cameras

    def _create_scene_props(self):
        """Per env one hidden, visual-only copy of every warehouse prop of cfg.scene_props (references to the upstream
        Simple_Warehouse props; collisions, rigid bodies and articulations switched off). A visual randomizer shows
        some of them per round and places them (set_scene_props). A prop that does not load completely (a sub-asset
        missing on the content server) stops the run: its Gprim count must match the props manifest."""
        from pxr import Usd, UsdGeom, UsdPhysics

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for i in range(self.num_envs):
            for k, prop in enumerate(self.cfg.scene_props):
                holder = UsdGeom.Xform.Define(stage, f"/World/envs/env_{i}/SceneProp_{k}")
                asset = stage.DefinePrim(f"/World/envs/env_{i}/SceneProp_{k}/Asset", "Xform")
                asset.GetReferences().AddReference(prop["url"])
                while True:  # de-instance so that the physics APIs of every prim can be switched off
                    inst = [q for q in Usd.PrimRange(asset) if q.IsInstanceable()]
                    if not inst:
                        break
                    for q in inst:
                        q.SetInstanceable(False)
                gprims = 0
                for q in Usd.PrimRange(asset):
                    gprims += q.IsA(UsdGeom.Gprim)
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr().Set(False)
                    if q.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(q).CreateRigidBodyEnabledAttr().Set(False)
                    if q.HasAPI(UsdPhysics.ArticulationRootAPI):
                        q.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if gprims != prop["gprims"]:
                    raise RuntimeError(f"[ASSET_LOAD_FAILED] {prop['url']}: {gprims} Gprims loaded, the complete asset has {prop['gprims']}")
                UsdGeom.Imageable(holder.GetPrim()).MakeInvisible()
        print(f"[SCENE_PROPS] {len(self.cfg.scene_props)} props x {self.num_envs} envs loaded completely", flush=True)

    def set_scene_props(self, placements: list):
        """placements[i] = [(prop index, x, y, yaw_deg), ...] in env i's frame: those props are shown standing on the
        ground plane at that pose, every other prop of env i is hidden."""
        from pxr import Gf, UsdGeom

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for i, placed in enumerate(placements):
            shown = {k: (x, y, yaw) for k, x, y, yaw in placed}
            for k, prop in enumerate(self.cfg.scene_props):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/SceneProp_{k}")
                if k in shown:
                    x, y, yaw = shown[k]
                    xf = UsdGeom.XformCommonAPI(prim)
                    xf.SetTranslate(Gf.Vec3d(float(x), float(y), self.GROUND_Z - float(prop["lo"][2])))
                    xf.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw)))
                    UsdGeom.Imageable(prim).MakeVisible()
                else:
                    UsdGeom.Imageable(prim).MakeInvisible()

    def bind_table_materials(self):
        """Bind the per-env table materials (stronger than the table's own bindings); nominal look otherwise."""
        import omni.usd
        from pxr import Usd, UsdGeom, UsdShade

        if self._table_material_bound:
            return
        if not self.cfg.bind_table_material:
            raise RuntimeError("table_color needs env cfg bind_table_material = True (bindings must exist before the simulation starts)")
        stage = omni.usd.get_context().get_stage()
        for i in range(self.num_envs):
            material = UsdShade.Material(stage.GetPrimAtPath(f"/World/Looks/CollectTable_{i}"))
            visuals = [c for c in stage.GetPrimAtPath(f"/World/envs/env_{i}/Table").GetChildren() if c.GetName().lower() == "visuals"]
            # bind every visual mesh and each of its GeomSubsets (the subsets carry the table's own materials, which
            # would otherwise win for their faces)
            bound = 0
            for prim in Usd.PrimRange(visuals[0]) if visuals else []:
                if prim.IsInstanceProxy():
                    raise RuntimeError(f"table visuals of env {i} are still instanced: {prim.GetPath()}")
                if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                    bound += 1
            if bound == 0:
                raise RuntimeError(f"table_color: no table visual mesh found under /World/envs/env_{i}/Table")
        self._table_material_bound = True

    def _fingertip_pos_now(self) -> torch.Tensor:
        return self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins

    # ---- reset: vendored reset, plug on the table, arm home ----
    def _teleport_home(self):
        """Arm to HOME_JOINTS with the gripper open, at rest; controller targets at the resulting fingertip pose."""
        n = self.num_envs
        q = torch.tensor(list(HOME_JOINTS) + [FINGER_OPEN_M, FINGER_OPEN_M], device=self.device).repeat(n, 1)
        self._robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self._robot.set_joint_position_target(q)
        self.ctrl_target_joint_pos[:] = q
        self.step_sim_no_action()
        if self.home_pos is None:
            self.home_pos = self.fingertip_midpoint_pos[0].clone()
            self.home_quat = self.fingertip_midpoint_quat[0].clone()
        # held with the reset gains: with the policy's soft gains the null-space torque pulled the hand ~3 cm off the
        # home pose while the plug settled (smoke 2026-10-03: first frame 30 mm from home)
        self._set_gains(self._reset_gains(), self.cfg.ctrl.reset_rot_deriv_scale)
        self._physics_steps(self.home_pos.expand(n, 3).clone(), self.home_quat.expand(n, 4).clone(), FINGER_OPEN_M, 2 * self.cfg.decimation)

    def randomize_initial_state(self, env_ids):
        n, dev = self.num_envs, self.device
        self._recording = False
        # vendored reset: socket (training range), plug at the policy start pose in the closed hand, realized grasp
        AssemblyEnv.randomize_initial_state(self, env_ids)
        self.start_pos, self.start_quat = self.held_pos.clone(), self.held_quat.clone()
        self.ref_grasp_q, self.ref_grasp_p = self.realized_grasp_quat_local.clone(), self.realized_grasp_pos_local.clone()
        # plug upright on the table at a random pose: uniform in the plug region, at least PLUG_SOCKET_MIN_DIST from
        # the socket centre (rejection sampling per env)
        plug_xy = torch.zeros((n, 2), device=dev)
        todo = torch.ones(n, dtype=torch.bool, device=dev)
        for _ in range(100):
            cand = torch.stack([PLUG_REGION_X[0] + (PLUG_REGION_X[1] - PLUG_REGION_X[0]) * torch.rand(n, device=dev),
                                PLUG_REGION_Y[0] + (PLUG_REGION_Y[1] - PLUG_REGION_Y[0]) * torch.rand(n, device=dev)], -1)
            plug_xy = torch.where(todo.unsqueeze(-1), cand, plug_xy)
            todo = todo & ((plug_xy - self.fixed_pos[:, :2]).norm(dim=-1) < PLUG_SOCKET_MIN_DIST)
            if not bool(todo.any()):
                break
        if bool(todo.any()):
            raise RuntimeError(f"no plug position {PLUG_SOCKET_MIN_DIST} m from the socket for envs {todo.nonzero().flatten().tolist()}")
        yaw = torch.deg2rad((2 * torch.rand(n, device=dev) - 1) * self.place_yaw_deg)
        plug_q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        plug_p = torch.cat([plug_xy, torch.full((n, 1), TABLE_TOP_Z - self.plug_bottom_local + self.drop_height, device=dev)], -1)
        open_w = float(self.approach_open)
        hold_pos, hold_quat = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        state = self._held_asset.data.default_root_state.clone()
        state[:, 0:3] = plug_p + self.scene.env_origins
        state[:, 3:7] = plug_q
        state[:, 7:] = 0.0
        self._held_asset.write_root_state_to_sim(state)
        self._held_asset.reset()
        self._physics_steps(hold_pos, hold_quat, open_w, 12)
        self._teleport_home()
        # settle on the table (nothing touches the plug)
        self._physics_steps(self.home_pos.expand(n, 3).clone(), self.home_quat.expand(n, 4).clone(), open_w,
                            int(round(self.settle_s / self.physics_dt)))
        self._teleport_home()  # exactly at the home joints for the episode's first frame
        self.tilt_settled_deg = self._tilt_deg(self.held_quat)
        self.pick_failed = self.tilt_settled_deg > self.topple_deg
        self.settled_pos = self.held_pos.clone()
        self.phase[:] = PH["approach"]
        self.grip_cmd[:] = 1.0
        self.script[:] = 0
        self.script_t[:] = 0.0
        # layout of the episode (env frame; the plug also in the socket frame)
        sq_inv, sp_inv = torch_utils.tf_inverse(self.fixed_quat, self.fixed_pos)
        pq_s, pp_s = torch_utils.tf_combine(sq_inv, sp_inv, self.held_quat, self.held_pos)
        self.layout = {"socket_pose_env": torch.cat([self.fixed_pos, self.fixed_quat], -1).clone(),
                       "plug_pose_env": torch.cat([self.held_pos, self.held_quat], -1).clone(),
                       "plug_pose_socket_frame": torch.cat([pp_s, pq_s], -1).clone(),
                       "plug_socket_distance_m": (self.held_pos[:, :2] - self.fixed_pos[:, :2]).norm(dim=-1).clone(),
                       "policy_start_pose_env": torch.cat([self.start_pos, self.start_quat], -1).clone(),
                       "plug_tilt_settled_deg": self.tilt_settled_deg.clone()}

    # ---- scripted pick and carry (all envs in lock step, recorded) ----
    def _scripted_frame_hook(self, env):
        """Called by PickAssemblyEnv._physics_steps after every physics step: a dataset frame every frame_stride."""
        if not getattr(self, "_recording", False):
            return
        self._hook_count += 1
        if self._hook_count % self.cfg.frame_stride == 0:
            self.sim.render()
            self.scene.update(dt=0.0)
            self._emit("step")

    def _reset_gains(self) -> torch.Tensor:
        return torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat((self.num_envs, 1))

    def _set_phase(self, name: str, grip_frac: float):
        self.phase[:] = PH[name]
        self.grip_cmd[:] = grip_frac

    def _servo(self, goal_pos, goal_quat, grip, tol_pos=0.002, tol_ang=0.03, max_iters=300, extra_iters=3, land=False):
        """PickAssemblyEnv._servo (same motion law), except that an env which stops making progress no longer holds
        the others. The scripted phases run in lock step, so one env that could not reach a waypoint (an arm near full
        extension, where the task-space controller cannot push along the arm) kept every env waiting max_iters (20 s)
        per waypoint, standing still on camera: 18 of 83 rounds on 2026-10-04. An env outside the tolerance is stalled
        when, over the last stall_window control steps,
        - still moving (not near the waypoint): its error in motion steps, d / max_lin + ang / max_ang, dropped by less
          than stall_far_rate x stall_window (the motion law covers one step per control step; 2026-10-09: an arm stuck
          229 mm away whose angle error still crept down 0.008 rad/s escaped the rule below for 110 control steps);
        - near the waypoint (final settling): its position and angle errors both dropped by less than
          stall_min_progress.
        A stalled env counts as done for this waypoint. Stalled farther than stall_fail from the waypoint: the pick
        fails (servo_stalled, the episode is not kept).
        land=True (the descent onto the plug): the target never jumps to the goal; each control step moves at most
        land_gain x the remaining error (and at most max_lin / max_ang), so the arm slows down as it arrives. With the
        jump (3 x max_lin = 18 mm from the goal, the arm still at ~90 mm/s under the stiff reset gains) the fingertips
        overshot the grasp by 2.6-5 mm and hit the table on tasks whose grasp is 0.3-8 mm above it (2026-10-09:
        86 N for one frame on 00768, then back to 0 N at the grasp pose)."""
        win, min_dp, min_da = self.cfg.stall_window, *self.cfg.stall_min_progress
        hist_d, hist_a, hist_e = [], [], []
        stalled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done_iters, it = 0, max_iters
        for k in range(max_iters):
            pos_err, aa_err = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos, fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=goal_pos, ctrl_target_fingertip_midpoint_quat=goal_quat,
                jacobian_type="geometric", rot_error_type="axis_angle")
            d = pos_err.norm(dim=-1, keepdim=True)
            ang = aa_err.norm(dim=-1)
            lim_lin = torch.clamp(self.cfg.land_gain * d, max=self.max_lin) if land else self.max_lin
            lim_ang = torch.clamp(self.cfg.land_gain * ang, max=self.max_ang) if land else self.max_ang
            step = pos_err * torch.clamp(lim_lin / d.clamp(min=1e-9), max=1.0)
            axis = aa_err / ang.clamp(min=1e-9).unsqueeze(-1)
            rot = torch_utils.quat_from_angle_axis(torch.minimum(ang, torch.as_tensor(lim_ang, device=self.device)), axis)
            rot = torch.where(ang.unsqueeze(-1) > 1e-6, rot, torch.tensor([1.0, 0, 0, 0], device=self.device).expand_as(rot))
            tgt_pos = self.fingertip_midpoint_pos + step
            tgt_quat = torch_utils.quat_mul(rot, self.fingertip_midpoint_quat)
            near = (d.squeeze(-1) < 3 * self.max_lin) & (ang < 3 * self.max_ang)
            jump = near if not land else (d.squeeze(-1) < tol_pos) & (ang < tol_ang)
            tgt_pos = torch.where(jump.unsqueeze(-1), goal_pos, tgt_pos)
            tgt_quat = torch.where(jump.unsqueeze(-1), goal_quat, tgt_quat)
            self._physics_steps(tgt_pos, tgt_quat, grip, self.cfg.decimation)
            d = d.squeeze(-1)
            within = (d < tol_pos) & (ang < tol_ang)
            e = d / self.max_lin + ang / self.max_ang
            hist_d.append(d)
            hist_a.append(ang)
            hist_e.append(e)
            if len(hist_d) > win:
                slow_far = ~near & ((hist_e[-win - 1] - e) < self.cfg.stall_far_rate * win)
                slow_near = near & ((hist_d[-win - 1] - d) < min_dp) & ((hist_a[-win - 1] - ang) < min_da)
                stalled = stalled | (~within & (slow_far | slow_near))
            if bool((within | stalled).all()):
                done_iters += 1
                if done_iters > extra_iters:
                    it = k
                    break
        far = stalled & ((hist_d[-1] > self.cfg.stall_fail[0]) | (hist_a[-1] > self.cfg.stall_fail[1]))
        self.servo_stalled |= far
        self.servo_stall_log.append((int(stalled.sum()), int(far.sum())))
        return it

    def _servo_dbg(self, name, goal_pos, goal_quat, grip, fast=False, **kw):
        """PickAssemblyEnv._servo (transport moves at transport_speedup x its rates) with a log line: control steps
        used, worst remaining position / angle error."""
        import isaaclab.utils.math as math_utils

        lin0, ang0 = self.max_lin, self.max_ang
        if fast:
            self.max_lin, self.max_ang = lin0 * self.cfg.transport_speedup, ang0 * self.cfg.transport_speedup
        try:
            it = self._servo(goal_pos, goal_quat, grip, **kw)
        finally:
            self.max_lin, self.max_ang = lin0, ang0
        dp = (self.fingertip_midpoint_pos - goal_pos).norm(dim=-1)
        dq = torch_utils.quat_mul(goal_quat, torch_utils.quat_conjugate(self.fingertip_midpoint_quat))
        ang = math_utils.axis_angle_from_quat(torch.where(dq[:, :1] < 0, -dq, dq)).norm(dim=-1)
        if os.environ.get("COLLECT_DEBUG"):
            print(f"[SERVO] {name}: {it} control steps, pos err max {float(dp.max()) * 1000:.1f} mm (env {int(dp.argmax())}), "
                  f"ang err max {float(ang.max()):.3f} rad (env {int(ang.argmax())}), stalled / failed {self.servo_stall_log[-1]}", flush=True)
        return it

    def run_pick_and_carry(self):
        """PickAssemblyEnv's pick (policy grasp from the plug's actual pose) and carry (to the policy start pose with
        the measured in-hand transform), from the home pose; then the hand-over to the policy."""
        n, dev = self.num_envs, self.device
        info = {}
        open_w = float(self.approach_open)
        self._recording, self._hook_count = True, 0
        self.servo_stalled[:] = False
        self.servo_stall_log = []
        # scripted motions with the reset gains (the vendored reset's own): with the policy's soft gains a 6 mm target
        # lead could not overcome the null-space torque and the arm stalled 4 cm short of the pre-grasp (smoke
        # 2026-10-03); the hand-over below restores the policy's gains
        self._set_gains(self._reset_gains(), self.cfg.ctrl.reset_rot_deriv_scale)
        g_quat, g_pos = self._grasp_target(self.held_pos, self.held_quat)
        # a plug that fell over is not picked (the episode is not kept): that env hovers above it, gripper upright,
        # instead of twisting the arm toward a grasp on the lying plug
        fell = self.pick_failed.unsqueeze(-1)
        hover_p = torch.cat([self.held_pos[:, :2], (self.held_pos[:, 2:3] + 0.12)], -1)
        g_pos = torch.where(fell, hover_p, g_pos)
        g_quat = torch.where(fell, self.home_quat.expand(n, 4), g_quat)
        pre = g_pos.clone()
        pre[:, 2] += self.approach_h
        self._set_phase("approach", open_w / FINGER_OPEN_M)
        self._servo_dbg("pregrasp", pre, g_quat, open_w, fast=True)
        self._servo_dbg("descend", g_pos, g_quat, open_w, land=True)
        info["fingertip_err_at_grasp_m"] = (self.fingertip_midpoint_pos - g_pos).norm(dim=-1)
        train_w = float(self.gripper_open_width)
        self._set_phase("grasp", train_w / FINGER_OPEN_M)
        hold_p, hold_q = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        t_pre, prev = 0.0, self.joint_pos[:, 7:9].clone()
        while t_pre < 2.0:
            self._physics_steps(hold_p, hold_q, train_w, 1)
            t_pre += self.physics_dt
            moved = (self.joint_pos[:, 7:9] - prev).abs().max()
            prev = self.joint_pos[:, 7:9].clone()
            if t_pre > 0.1 and bool(moved < self.cfg_task.grasp_settle_tol):
                break
        info["plug_shift_before_close_m"] = (self.held_pos[:, :2] - self.settled_pos[:, :2]).norm(dim=-1)
        self.grip_cmd[:] = 0.0
        self._close_gripper(max_time=max(2.0, self.cfg_task.grasp_max_time))
        plug_z0 = self.held_pos[:, 2].clone()
        lift = self.fingertip_midpoint_pos.clone()
        lift[:, 2] += self.approach_h
        self._set_phase("lift", 0.0)
        self._servo_dbg("lift", lift, self.fingertip_midpoint_quat.clone(), 0.0, tol_pos=0.004)
        info["plug_lifted_m"] = self.held_pos[:, 2] - plug_z0
        info["grasp_ok"] = info["plug_lifted_m"] > 0.5 * self.approach_h
        held_inv_q, held_inv_p = torch_utils.tf_inverse(self.held_quat, self.held_pos)
        r2g_inv_q, r2g_inv_p = torch_utils.tf_inverse(self.robot_to_gripper_quat, self.palm_to_finger_center)
        q, t = torch_utils.tf_combine(held_inv_q, held_inv_p, self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
        rq, rp = torch_utils.tf_combine(q, t, r2g_inv_q, r2g_inv_p)
        info["grasp_pos_err_m"] = (rp - self.ref_grasp_p).norm(dim=-1)
        dq = torch_utils.quat_mul(torch_utils.quat_conjugate(self.ref_grasp_q), rq)
        info["grasp_rot_err_deg"] = torch.rad2deg(2 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))
        # carry
        self._set_phase("carry", 0.0)
        ft_inv_q, ft_inv_p = torch_utils.tf_inverse(self.fingertip_midpoint_quat, self.fingertip_midpoint_pos)
        hp_q, hp_p = torch_utils.tf_combine(ft_inv_q, ft_inv_p, self.held_quat, self.held_pos)
        hp_inv_q, hp_inv_p = torch_utils.tf_inverse(hp_q, hp_p)
        tgt_q, tgt_p = torch_utils.tf_combine(self.start_quat, self.start_pos, hp_inv_q, hp_inv_p)
        # an env whose pick already failed (plug fell over or not lifted, or a waypoint stalled) holds where it is: its
        # in-hand transform is meaningless, so its carry target, and with it the cruise height (the maximum over the
        # envs), could lie far up; every env then travelled up there and back (2026-10-09: rounds of 00437 / 00015
        # with carry 400-660 frames instead of ~194)
        hold = self.pick_failed | ~info["grasp_ok"] | self.servo_stalled
        act = ~hold
        here_p, here_q = self.fingertip_midpoint_pos.clone(), self.fingertip_midpoint_quat.clone()
        tgt_p = torch.where(hold.unsqueeze(-1), here_p, tgt_p)
        tgt_q = torch.where(hold.unsqueeze(-1), here_q, tgt_q)
        ref = act if bool(act.any()) else hold
        cruise = max(float(tgt_p[ref, 2].max()), float(here_p[ref, 2].max())) + 0.05
        up = here_p.clone()
        up[act, 2] = cruise
        self._servo_dbg("cruise", up, here_q, 0.0, fast=True, tol_pos=0.006)
        over = tgt_p.clone()
        over[act, 2] = cruise
        self._servo_dbg("over", over, tgt_q, 0.0, fast=True, tol_pos=0.004)
        self._servo_dbg("place", tgt_p, tgt_q, 0.0)
        self._physics_steps(tgt_p, tgt_q, 0.0, 2 * self.cfg.decimation)
        info["place_pos_err_m"] = (self.held_pos - self.start_pos).norm(dim=-1)
        # hand over to the policy: realized grasp and goal as the vendored reset computes them, zero actions
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
        # the policy's gains, exactly as in training (checked against the config every hand-over)
        expect = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat((n, 1))
        if not torch.allclose(self.default_gains, expect):
            raise RuntimeError("default_gains changed since the env was created")
        self._set_gains(self.default_gains.clone())
        info["servo_stalled"] = self.servo_stalled.clone()
        self.pick_failed = self.pick_failed | ~info["grasp_ok"] | self.servo_stalled
        self.phase[:] = PH["insert"]
        self.grip_cmd[:] = 0.0
        self._recording = False
        self.pick_info = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in info.items()}
        self.pick_info["tilt_settled_deg"] = self.tilt_settled_deg.clone()
        self.pick_info["pick_failed"] = self.pick_failed.clone()

    # ---- per-env script after the insertion (or to park a failed env) ----
    def start_release(self, mask: torch.Tensor):
        """Envs in mask (inserted): open the gripper where they are, then lift and return home."""
        m = mask.to(self.device) & (self.script == 0)
        self.script[m], self.script_t[m] = 1, 0.0
        self.script_pos[m] = self._fingertip_pos_now()[m]
        self.script_quat[m] = self._robot.data.body_quat_w[m, self.fingertip_body_idx]
        self.script_ref_p[m], self.script_ref_q[m] = self.script_pos[m].clone(), self.script_quat[m].clone()
        g = self._reset_gains()
        # copies first: _set_gains keeps a reference to the tensor it is given (default_gains after the hand-over), so
        # an in-place edit here changed default_gains itself and every later round's policy ran with the stiff reset
        # gains in the envs released before (trial 2026-10-03: rounds >= 1 mostly failed to insert)
        self.task_prop_gains = self.task_prop_gains.clone()
        self.task_deriv_gains = self.task_deriv_gains.clone()
        self.task_prop_gains[m] = g[m]
        self.task_deriv_gains[m] = 2 * torch.sqrt(g[m])
        self.task_deriv_gains[m, 3:6] /= self.cfg.ctrl.reset_rot_deriv_scale
        self.phase[m] = PH["release"]
        self.grip_cmd[m] = 1.0

    def freeze_envs(self, mask: torch.Tensor):
        """Envs in mask hold their current fingertip pose and gripper (failed episodes, waiting for the round end)."""
        m = mask.to(self.device) & (self.script != 6)
        self.script[m] = 6
        self.script_pos[m] = self._fingertip_pos_now()[m]
        self.script_quat[m] = self._robot.data.body_quat_w[m, self.fingertip_body_idx]
        self.script_grip[m] = self.joint_pos[m, 7:9].mean(-1)

    def _step_toward(self, goal_p, goal_q, idx, fast=False):
        """Advance the moving reference of envs idx toward (goal_p, goal_q) by the scripted rates per physics step
        (max_lin / max_ang per control step, as PickAssemblyEnv._servo); returns the reference and the remaining
        distance / angle of the actual fingertip to the goal. The reference runs ahead of a lagging arm, so the
        impedance force grows until the arm follows."""
        import isaaclab.utils.math as math_utils

        ref_p, ref_q = self.script_ref_p[idx], self.script_ref_q[idx]
        k = self.cfg.transport_speedup if fast else 1.0
        lin, angs = k * self.max_lin / self.cfg.decimation, k * self.max_ang / self.cfg.decimation
        d = goal_p - ref_p
        dist_r = d.norm(dim=-1, keepdim=True)
        ref_p = ref_p + d * torch.clamp(lin / dist_r.clamp(min=1e-9), max=1.0)
        dq = torch_utils.quat_mul(goal_q, torch_utils.quat_conjugate(ref_q))
        dq = torch.where(dq[:, :1] < 0, -dq, dq)
        aa = math_utils.axis_angle_from_quat(dq)
        ang_r = aa.norm(dim=-1, keepdim=True)
        aa = aa * torch.clamp(angs / ang_r.clamp(min=1e-9), max=1.0)
        an = aa.norm(dim=-1)
        rot = torch_utils.quat_from_angle_axis(an, aa / an.unsqueeze(-1).clamp(min=1e-9))
        rot = torch.where(an.unsqueeze(-1) > 1e-9, rot, torch.tensor([1.0, 0, 0, 0], device=self.device).expand_as(rot))
        ref_q = torch_utils.quat_mul(rot, ref_q)
        self.script_ref_p[idx], self.script_ref_q[idx] = ref_p, ref_q
        dist = (goal_p - self.fingertip_midpoint_pos[idx]).norm(dim=-1)
        dqa = torch_utils.quat_mul(goal_q, torch_utils.quat_conjugate(self.fingertip_midpoint_quat[idx]))
        ang = math_utils.axis_angle_from_quat(torch.where(dqa[:, :1] < 0, -dqa, dqa)).norm(dim=-1)
        return ref_p, ref_q, dist, ang

    def _run_script(self):
        """Override the controller targets of the scripted envs (called every physics step after the policy's)."""
        idx = (self.script > 0).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        dt = self.physics_dt
        self.script_t[idx] += dt
        n_s = idx.numel()
        tgt_p, tgt_q = self.script_pos[idx].clone(), self.script_quat[idx].clone()
        grip = torch.zeros(n_s, device=self.device)
        st = self.script[idx]
        home_p, home_q = self.home_pos.expand(n_s, 3), self.home_quat.expand(n_s, 4)
        # 1 release: hold the pose, open
        r = st == 1
        grip = torch.where(r, torch.full_like(grip, FINGER_OPEN_M), grip)
        go = r & (self.script_t[idx] >= self.cfg.release_s)
        if bool(go.any()):
            j = idx[go]
            self.script[j], self.script_t[j] = 2, 0.0
            self.script_pos[j, 2] += self.cfg.lift_m
            self.phase[j] = PH["return"]
        # 2 lift, 3 to home: clipped targets toward the lift point / the home pose
        for code in (2, 3):
            m = self.script[idx] == code
            if not bool(m.any()):
                continue
            j = idx[m]
            goal_p = self.script_pos[j] if code == 2 else home_p[m]
            goal_q = self.script_quat[j] if code == 2 else home_q[m]
            p, q, dist, ang = self._step_toward(goal_p, goal_q, j, fast=code == 3)
            tgt_p[m], tgt_q[m] = p, q
            grip[m] = FINGER_OPEN_M
            ref_at_goal = (p - goal_p).norm(dim=-1) < 1e-6
            done = ref_at_goal & (dist < (0.006 if code == 2 else self.cfg.home_tol_pos)) & (ang < (0.1 if code == 2 else self.cfg.home_tol_ang))
            if bool(done.any()):
                k = j[done]
                self.script[k], self.script_t[k] = code + 1, 0.0
        # 4 at home: hold the home pose until still for home_still_s, then done
        m = self.script[idx] == 4
        if bool(m.any()):
            tgt_p[m], tgt_q[m], grip[m] = home_p[m], home_q[m], FINGER_OPEN_M
            j = idx[m]
            fin = self.script_t[j] >= self.cfg.home_still_s
            self.script[j[fin]] = 5
        m = self.script[idx] == 5
        if bool(m.any()):
            tgt_p[m], tgt_q[m], grip[m] = home_p[m], home_q[m], FINGER_OPEN_M
        m = self.script[idx] == 6
        if bool(m.any()):
            grip[m] = self.script_grip[idx[m]]
        self.ctrl_target_fingertip_midpoint_pos[idx] = tgt_p
        self.ctrl_target_fingertip_midpoint_quat[idx] = tgt_q
        g = torch.zeros((self.num_envs, 1), device=self.device)  # policy envs keep the gripper closed (0)
        g[idx, 0] = grip
        self.ctrl_target_gripper_dof_pos = g
        sc = self.script[idx]
        upd = (sc >= 1) & (sc <= 5)
        self.grip_cmd[idx[upd]] = grip[upd] / FINGER_OPEN_M
        self.generate_ctrl_signals()

    def _apply_action(self):
        super()._apply_action()
        self._run_script()

    @property
    def cameras(self) -> dict[str, TiledCamera]:
        return {CAMERA_KEYS[0]: self._front_camera, CAMERA_KEYS[1]: self._wrist_camera}

    def capture(self, tag: str) -> dict:
        """State and images of every env at the current physics step (the renderer ran at this step).

        Reads the simulation buffers directly and does not call ``_compute_intermediate_values``: that method also
        advances the finite-difference velocity history used by the controller and the observations, so an extra
        call here would change the rollout.
        """
        robot = self._robot.data
        joint_pos, joint_vel = robot.joint_pos, robot.joint_vel
        frame = {
            "tag": tag,
            "sim_step": int(self._sim_step_counter),
            "episode_step": self.episode_length_buf.clone(),
            "eef_pos": robot.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins,  # env frame
            "eef_quat": robot.body_quat_w[:, self.fingertip_body_idx].clone(),  # (w, x, y, z)
            "finger_pos": joint_pos[:, 7:9].clone(),
            "arm_pos": joint_pos[:, 0:7].clone(),
            "arm_vel": joint_vel[:, 0:7].clone(),
            "arm_effort": robot.applied_torque[:, 0:7].clone(),
            "keypoint_error": self._keypoint_error_now(),
            "policy_action": None if self._last_policy_action is None else self._last_policy_action.clone(),
            "phase": self.phase.clone(),
            "gripper_cmd": self.grip_cmd.clone(),
            "script": self.script.clone(),
            "pick_failed": self.pick_failed.clone(),
        }
        for key, cam in self.cameras.items():
            frame[key] = {"rgb": cam.data.output["rgb"], "depth": cam.data.output["distance_to_image_plane"],
                          # optical centre in the env frame, orientation of the ROS optical frame (x right, y down,
                          # z forward) in the env frame, (w, x, y, z)
                          "pose": self._camera_pose_env(key)}
        # root poses of the plug and the socket in the env frame, (x, y, z, w, x, y, z)
        frame["plug_pose"] = torch.cat([self._held_asset.data.root_pos_w - self.scene.env_origins, self._held_asset.data.root_quat_w], dim=-1)
        frame["socket_pose"] = torch.cat([self._fixed_asset.data.root_pos_w - self.scene.env_origins, self._fixed_asset.data.root_quat_w], dim=-1)
        frame["contact_force"] = torch.cat(
            [
                self._contact["left_finger"].data.net_forces_w[:, 0, :],
                self._contact["right_finger"].data.net_forces_w[:, 0, :],
                self._contact["plug"].data.net_forces_w[:, 0, :],
            ],
            dim=-1,
        )  # (num_envs, 9): net contact force on left finger, right finger, plug; world frame, N
        return frame

    def _camera_pose_env(self, key: str) -> torch.Tensor:
        """Camera pose (x, y, z, w, x, y, z of the ROS optical frame) in the env frame at the current physics step,
        from the same source as the recorded state: the fixed front camera from its env-frame mount, the wrist camera
        from the panda_hand body pose composed with its mount offset. (The camera sensor's own pos_w / quat_w_ros lag
        the physics by up to a couple of frames: 1.7-3.5 cm apart from the hand pose while the arm moves.)"""
        from isaaclab.utils.math import quat_apply, quat_mul

        cfg = self.cfg.front_camera if key.endswith("front") else self.cfg.wrist_camera
        off_p = torch.tensor(cfg.offset.pos, dtype=torch.float32, device=self.device).expand(self.num_envs, 3)
        off_q = torch.tensor(cfg.offset.rot, dtype=torch.float32, device=self.device).expand(self.num_envs, 4)
        if key.endswith("front"):
            return torch.cat([off_p, off_q], dim=-1)
        hand = self._robot.body_names.index("panda_hand")
        hp = self._robot.data.body_pos_w[:, hand] - self.scene.env_origins
        hq = self._robot.data.body_quat_w[:, hand]
        return torch.cat([hp + quat_apply(hq, off_p), quat_mul(hq, off_q)], dim=-1)

    def _keypoint_error_now(self) -> torch.Tensor:
        """Mean keypoint distance, same formula as ``_compute_intermediate_values`` (success criterion input),
        computed into temporaries from the current simulation state."""
        held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        held_quat = self._held_asset.data.root_quat_w
        fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        fixed_quat = self._fixed_asset.data.root_quat_w
        held_base_quat, held_base_pos = torch_utils.tf_combine(
            held_quat, held_pos, self.held_base_quat_local, self.held_base_pos_local
        )
        target_quat, target_pos = torch_utils.tf_combine(
            fixed_quat, fixed_pos, self.identity_quat, self.fixed_success_pos_local
        )
        dists = []
        for offset in self.keypoint_offsets:
            offset = offset.repeat(self.num_envs, 1)
            kp_held = torch_utils.tf_combine(held_base_quat, held_base_pos, self.identity_quat, offset)[1]
            kp_fixed = torch_utils.tf_combine(target_quat, target_pos, self.identity_quat, offset)[1]
            dists.append(torch.norm(kp_held - kp_fixed, p=2, dim=-1))
        return torch.stack(dists, dim=-1).mean(-1)

    def _emit(self, tag: str):
        if self.frame_callback is not None:
            self.frame_callback(self, self.capture(tag))

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        # after the reset, like the in-step resets: the hook places the distractors relative to the socket pose of the
        # new episodes (before the reset it would read the previous, not yet randomized pose)
        if self.visual_hook is not None:
            self.visual_hook(self)
            self._render_until_settled()
            self.scene.update(dt=0.0)
        if not self._cameras_warm:
            # the RGB / depth annotators return no data until the renderer has produced a few frames
            for _ in range(self.cfg.camera_warmup_renders):
                self.sim.render()
            self.scene.update(dt=0.0)
            self._cameras_warm = True
            # the first round's visuals: the warm-up renders alone do not settle a strong light change
            self._render_until_settled()  # last_settle_renders counts the settle loop only, as in later rounds
            self.scene.update(dt=0.0)
        self._last_policy_action = None
        self._emit("reset")  # the episode's first frame: arm at home, plug on the table
        self.run_pick_and_carry()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.extras

    def _render_until_settled(self):
        """Re-render the new visuals until the images stop changing (at least num_rerenders_on_reset renders)."""
        self.last_settle_renders, self.last_settle_capped = 0, False
        if not self.sim.has_rtx_sensors():
            return
        if not self._cameras_warm:  # no image data yet: the camera warm-up renders that follow settle the first round
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
            self.last_settle_renders = self.cfg.num_rerenders_on_reset
            return
        hist = []
        t_settle = time.perf_counter()
        nmin = int(os.environ.get("AUTOMATE_SETTLE_MIN", self.cfg.settle_min_renders))
        cap = max(int(os.environ.get("AUTOMATE_MAX_SETTLE", self.cfg.max_settle_renders)), nmin + 2 * self.cfg.settle_block + 100)
        trace = os.environ.get("AUTOMATE_SETTLE_TRACE")
        for n in range(1, cap + 1):
            self.sim.render()
            self.scene.update(dt=0.0)
            cur = torch.stack([cam.data.output["rgb"].float().mean(dim=(1, 2, 3)) for cam in self.cameras.values()], dim=-1)
            hist.append(cur)
            b = self.cfg.settle_block
            if len(hist) >= 2 * b:
                old, new = torch.stack(hist[:b]).mean(0), torch.stack(hist[b:]).mean(0)  # (num_envs, cameras)
                rel = ((new - old).abs() / new.clamp(min=1.0)).flatten()
                p95, mx = float(torch.quantile(rel, 0.95)), float(rel.max())
                mrel = float(((new.mean(0) - old.mean(0)).abs() / new.mean(0).clamp(min=1.0)).max())
                if trace:
                    print(f"[SETTLE] n={n} rel_p95={p95:.4f} rel_max={mx:.4f} rel_mean={mrel:.4f} "
                          f"mean_front={float(cur[:, 0].mean()):.2f} mean_wrist={float(cur[:, 1].mean()):.2f}", flush=True)
                if n >= nmin and p95 < self.cfg.settle_rel_p95 and mx < self.cfg.settle_rel_max and mrel < self.cfg.settle_rel_mean:
                    break
                hist.pop(0)
        else:  # never settled (2026-10-01: one mixed round still brightened 4 % on average over its first 40 frames)
            self.last_settle_capped = True
        self.last_settle_renders = n
        self.last_settle_wall_s = time.perf_counter() - t_settle
        probe = int(os.environ.get("AUTOMATE_SETTLE_PROBE", "0"))
        if probe:  # study only: physics frozen, render `probe` more frames; per env drift (last 10 vs first 10) and noise
            vals = []
            for _i in range(probe):
                self.sim.render()
                self.scene.update(dt=0.0)
                if _i == 0:
                    self._probe_first = self.cameras["observation.images.front"].data.output["rgb"].clone()
                f = self.cameras["observation.images.front"].data.output["rgb"].float()
                w = self.cameras["observation.images.wrist"].data.output["rgb"].float()
                vals.append(torch.stack([f[:, 400:470, 10:110].mean(dim=(1, 2, 3)), f.mean(dim=(1, 2, 3)), w.mean(dim=(1, 2, 3))], -1))
            v = torch.stack(vals).cpu()  # (renders, envs, 3)
            pdir = os.environ.get("AUTOMATE_SETTLE_PROBE_DIR")
            if pdir:  # save first / last frozen front image of the env whose table patch changed most, and their difference
                import numpy as _np
                from PIL import Image as _Image
                k = int(((v[-1, :, 0] - v[0, :, 0]).abs() / v[0, :, 0].clamp(min=1.0)).argmax())
                os.makedirs(pdir, exist_ok=True)
                tag = f"phase{getattr(self, '_probe_phase', 0)}_env{k}"
                self._probe_phase = getattr(self, "_probe_phase", 0) + 1
                a0 = self._probe_first[k].cpu().numpy().astype(_np.uint8); a1 = self.cameras["observation.images.front"].data.output["rgb"][k].cpu().numpy().astype(_np.uint8)
                dif = _np.clip(_np.abs(a1.astype(int) - a0.astype(int)).sum(-1) * 4, 0, 255).astype(_np.uint8)
                _Image.fromarray(_np.concatenate([a0, a1, _np.stack([dif] * 3, -1)], 1)).save(f"{pdir}/{tag}.png")
                # fixed env 0, last frozen frame (A/B comparisons across settings with the same seed)
                _Image.fromarray(self.cameras["observation.images.front"].data.output["rgb"][0].cpu().numpy().astype(_np.uint8)).save(
                    f"{pdir}/phase{self._probe_phase - 1}_env0_front.png")
                _Image.fromarray(self.cameras["observation.images.wrist"].data.output["rgb"][0].cpu().numpy().astype(_np.uint8)).save(
                    f"{pdir}/phase{self._probe_phase - 1}_env0_wrist.png")
            a, b = v[:10].mean(0), v[-10:].mean(0)
            drift = (b - a) / a.clamp(min=1.0) * 100
            noise = torch.cat([v[:10], v[-10:]]).std(0) / a.clamp(min=1.0) * 100
            real = (drift.abs() > 1.0) & (drift.abs() > 3 * noise / 10 ** 0.5)
            names = ["table_patch", "front", "wrist"]
            msg = " | ".join(f"{nm}: drift p50 {drift[:, k].abs().median():.2f} p90 {drift[:, k].abs().quantile(0.9):.2f} max {drift[:, k].abs().max():.2f} "
                             f"signed_mean {drift[:, k].mean():+.2f} noise_p50 {noise[:, k].median():.2f} envs_drifting {int(real[:, k].sum())}/{v.shape[1]}"
                             for k, nm in enumerate(names))
            print(f"[SETTLE_PROBE] stopped at n={n} capped={self.last_settle_capped} settle_wall_s={self.last_settle_wall_s:.1f} "
                  f"over {probe} frozen renders (%): {msg}", flush=True)

    def step(self, action: torch.Tensor):
        """DirectRLEnv.step (Isaac Lab 2.3.2) with a frame capture after every ``frame_stride`` physics steps."""
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)
        self._last_policy_action = action.clone()
        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for sub in range(self.cfg.decimation):
            t0 = time.perf_counter()
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            _sync(); t1 = time.perf_counter(); PROF["physics"] += t1 - t0
            capture = (sub + 1) % self.cfg.frame_stride == 0
            if capture and is_rendering:
                self.sim.render()
            _sync(); t2 = time.perf_counter(); PROF["render"] += t2 - t1
            self.scene.update(dt=self.physics_dt)
            _sync(); t3 = time.perf_counter(); PROF["scene_update"] += t3 - t2
            if capture:
                self._emit("step")
            PROF["capture_and_write"] += time.perf_counter() - t3

        t4 = time.perf_counter()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            raise RuntimeError(f"unexpected in-step reset of envs {reset_env_ids.tolist()} (rounds end by env.reset())")
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self._get_observations()
        _sync(); PROF["dones_rewards_reset_obs"] += time.perf_counter() - t4
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
