"""Collect single-arm raw Cartesian full-process episodes (2026-10-03): pick -> carry -> insert -> release -> return.

Each episode: the arm starts at the fixed home pose, picks the plug standing on the table (scripted), carries
it above the socket (scripted), the frozen insertion specialist (relative observations, socket-frame actions, D16;
deterministic) inserts it, then the gripper opens and the arm returns to the home pose (scripted). Recorded at 30 FPS
(every 4 physics steps at 120 Hz; policy at 15 Hz) with the D435 global and D405 wrist cameras (RGB + metric depth).
An episode is kept only if the plug is seated (mean keypoint error < 3 mm) at its last frame, with the arm back home.
Failed attempts are logged in ``attempts.jsonl`` and their media deleted.

Frame alignment: frame t is the state and the images of one physics step. ``action[t]`` is the absolute end-effector
pose at frame t+1 plus the gripper target at frame t+1; ``policy_action[t]`` is the policy action applied between
frame t and t+1 during the insert phase (zeros in the scripted phases); ``phase_index[t]`` names the phase.

    ./isaaclab.sh -p collect_automate_single_arm.py --assembly-id 00422 --checkpoint <final.pth> \
        --out-dir <dir> --target-successes 1 --num_envs 4 --headless --enable_cameras --device cuda:0
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assembly-id", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--checkpoint-source", default=None, help="Original checkpoint file, if --checkpoint is a CPU-mapped copy of it.")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--target-successes", type=int, default=1)
parser.add_argument("--max-rounds", type=int, default=50, help="Rounds of num_envs parallel episodes at most.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=30000)
parser.add_argument("--visual-profile", default="lighting", choices=["lighting", "table_color", "object_distractors", "texture", "scene", "mixed"],
                    help="Visual randomization profile (README 4.1). Implemented: lighting, table_color, object_distractors.")
parser.add_argument("--max-insert-s", type=float, default=10.0, help="Insertion phase longer than this: the episode fails.")
parser.add_argument("--max-return-s", type=float, default=15.0, help="Release + return longer than this: the episode fails.")
parser.add_argument("--strict-thresh", type=float, default=0.003)
parser.add_argument("--task-text", default="Insert the plug into the socket.")
parser.add_argument("--production-id", default="automate_fullproc_20261003",
                    help="Production written into the base trajectory / layout ids (shared by the mixed and single batches).")
parser.add_argument("--batch-id", default=None, help="Production batch id written into every episode (default: the output root's "
                    "parent directory name, e.g. the collection date).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

os.environ["ROBOASSEMBLYBENCH_AUTOMATE_ASSEMBLY_ID"] = str(args_cli.assembly_id)
TASK = "RoboAssemblyBench-AutoMate-Assembly-Collect-Direct-v0"
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402

# see rollout_video.py: replicator's overscan resize reads unset /rtx/dataWindowNDC settings in this headless setup
carb.settings.get_settings().set("/rtx/dataWindow/fitOutputToDataWindow", True)
if os.environ.get("AUTOMATE_RTX_SAMPLED_LIGHTING") == "0":  # experiment switch (2026-10-01 settle study), off by default
    carb.settings.get_settings().set("/rtx/directLighting/sampledLighting/enabled", False)
if os.environ.get("AUTOMATE_RTX_AA_OP"):  # experiment switch (settle study): anti-aliasing op, 0 = off, 3 = DLSS (default)
    carb.settings.get_settings().set("/rtx/post/aa/op", int(os.environ["AUTOMATE_RTX_AA_OP"]))

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaacsim.core.utils.torch as torch_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import roboassemblybench.library.automate  # noqa: E402,F401
from roboassemblybench.datasets import SINGLE_ARM_CAMERA_KEYS, SingleArmCartesianEpisodeRecorder  # noqa: E402
from roboassemblybench.library.automate.assembly_collect_env import (  # noqa: E402
    CAMERA_KEYS, CAMERA_MODELS, HOME_JOINTS, PH, PHASES, PLUG_REGION_X, PLUG_REGION_Y, PLUG_SOCKET_MIN_DIST)

assert tuple(CAMERA_KEYS) == tuple(SINGLE_ARM_CAMERA_KEYS)
FINGER_OPEN_M = 0.04  # Franka finger joint travel; gripper_open = mean finger joint / 0.04


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_provenance() -> dict:
    root = Path(roboassemblybench.library.automate.__file__).resolve().parents[3]
    out = {"repo_root": str(root)}
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip():
            out["git_head"] = head.stdout.strip()
            diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD"], capture_output=True).stdout
            out["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
            out["git_diff_files"] = subprocess.run(["git", "-C", str(root), "diff", "--stat", "HEAD"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
        else:
            out.update({"git_head": None, "git_diff_sha256": None, "git_diff_files": None,
                        "git_missing_reason": (head.stderr.strip() or "no git checkout") + "; source_sha256 identifies the code"})
    except OSError as e:
        out.update({"git_head": None, "git_diff_sha256": None, "git_diff_files": None, "git_missing_reason": repr(e)})
    # git diff does not see files that are not tracked yet (the collector's own modules): hash the sources that
    # determine the recorded data
    out["source_sha256"] = {}
    for rel in (
        "roboassemblybench/library/automate/assembly_env.py",
        "roboassemblybench/library/automate/assembly_env_cfg.py",
        "roboassemblybench/library/automate/assembly_tasks_cfg.py",
        "roboassemblybench/library/automate/assembly_collect_env.py",
        "roboassemblybench/library/automate/pick_env.py",
        "roboassemblybench/datasets/cartesian_episode.py",
        "roboassemblybench/scripts/collect_automate_single_arm.py",
    ):
        f = root / rel
        out["source_sha256"][rel] = sha256_file(str(f)) if f.is_file() else None
    return out


# every randomizer draws from its own stream: SeedSequence([run seed, stream, round, slot]), slot 0 = the round-wide
# stream (dome light, ground texture), slot i + 1 = env i. Keys always have 4 entries: SeedSequence zero-pads shorter
# keys, so a 3-entry round key would equal env 0's key. No two randomizers, rounds, envs or runs share a random
# sequence (the key is recorded as the visual seed)
STREAMS = {"lighting": 1, "distractors": 2, "texture": 3, "scene": 4, "table_color": 5}


def stream_rng(seed: int, stream: str, rnd: int, i: int | None = None):
    key = [int(seed), STREAMS[stream], int(rnd), 0 if i is None else int(i) + 1]
    return np.random.default_rng(np.random.SeedSequence(key)), key


class LightingRandomizer:
    """README profile ``lighting``: per round the dome light intensity / tint (one stage light shared by all envs), and
    per env its own sphere key light (position, radius, intensity, colour)."""

    def __init__(self, seed: int):
        self.seed, self.round, self.dome, self.keys = int(seed), -1, None, []

    def __call__(self, env):
        from pxr import Gf, UsdGeom, UsdLux
        import omni.usd

        self.round += 1
        rng, key = stream_rng(self.seed, "lighting", self.round)
        tint = np.clip(0.75 + rng.uniform(-0.15, 0.15, 3), 0.3, 1.0)
        self.dome = {"dome_seed": key, "dome_intensity": float(2000.0 * rng.uniform(0.35, 1.8)), "dome_color": [float(v) for v in tint]}
        if os.environ.get("AUTOMATE_FIX_DOME") == "1":  # experiment switch (settle study): dome left at its default
            self.dome.update({"dome_intensity": 2000.0, "dome_color": [0.75, 0.75, 0.75]})
        stage = omni.usd.get_context().get_stage()
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        dome.GetIntensityAttr().Set(self.dome["dome_intensity"])
        dome.GetColorAttr().Set(Gf.Vec3f(*self.dome["dome_color"]))
        self.keys = []
        for i in range(env.num_envs):  # key_position is env-local
            r, k = stream_rng(self.seed, "lighting", self.round, i)
            p = {"visual_seed": k, "key_intensity": float(r.uniform(0.0, 4.0e5)), "key_radius": float(r.uniform(0.05, 0.25)),
                 "key_position": [float(r.uniform(0.2, 1.0)), float(r.uniform(-0.6, 0.6)), float(r.uniform(0.6, 1.5))],
                 "key_color": [float(v) for v in np.clip(r.uniform(0.7, 1.0, 3), 0.0, 1.0)]}
            key_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/KeyLight")
            light = UsdLux.SphereLight(key_prim)
            light.GetIntensityAttr().Set(p["key_intensity"])
            light.GetRadiusAttr().Set(p["key_radius"])
            light.GetColorAttr().Set(Gf.Vec3f(*p["key_color"]))
            # the spawned light carries its own xformOp:translate (with an orient op XformCommonAPI cannot handle: its
            # SetTranslate failed with "incompatible xformable" and left every key light at the spawn position until
            # 2026-10-03); set the op directly and read it back
            op = key_prim.GetAttribute("xformOp:translate")
            op.Set(Gf.Vec3d(*p["key_position"]))
            got = op.Get()
            if got is None or max(abs(float(a) - float(b)) for a, b in zip(got, p["key_position"])) > 1e-6:
                raise RuntimeError(f"key light {i}: translate not applied ({got} vs {p['key_position']})")
            self.keys.append(p)

    def params_for(self, i: int) -> dict:
        return {**self.keys[i], **self.dome}

    def visual_seed_for(self, i: int):
        return self.keys[i]["visual_seed"]


def static_conventions(env_cfg, base) -> dict:
    """Frames, units, action / success semantics and the default environment, written out explicitly."""
    import omni.usd
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
    key0 = stage.GetPrimAtPath("/World/envs/env_0/KeyLight")
    key_light = UsdLux.SphereLight(key0)
    key_t = key0.GetAttribute("xformOp:translate").Get() if key0.HasAttribute("xformOp:translate") else None
    ctrl = env_cfg.ctrl
    return {
        "batch_id": args_cli.batch_id or Path(args_cli.out_dir).resolve().parent.name,
        "robot_type": "franka_panda_single_arm",
        "main_instruction": args_cli.task_text,
        "target_categories": ["plug", "socket"],
        "conventions": {
            "world_frame": "env: per-env origin on the floor plan, z up; state, action, object and camera poses all use it. "
                           "Each env is an independent copy of the scene; the episode's T_world_env (pure translation "
                           "env_origin_world_m, no rotation) maps it to the simulation world",
            "world_axes": "right_handed_z_up", "length_unit": "m", "time_unit": "s", "quaternion_order": "wxyz",
            "camera_axes": "x_right_y_down_z_forward (ROS optical)",
            "timestamp": "frame_index / fps within the episode",
        },
        "eef_link": "panda_fingertip_centered",
        "eef_pose_frame": "env",
        "feature_units": {"observation.state": ["m", "m", "m", "dimensionless", "dimensionless", "dimensionless", "dimensionless", "fraction_open"],
                          "action": ["m", "m", "m", "dimensionless", "dimensionless", "dimensionless", "dimensionless", "fraction_open"]},
        "gripper_definition": ("state gripper_open = mean of the two finger joints / 0.04 m (0 closed, 1 fully open); action gripper = the "
                               "commanded finger opening as the same fraction at the next frame: 1 open while approaching and releasing, "
                               "the training opening (gripper_open_width / 0.04) while pre-closing, 0 closed from the close on through the insertion"),
        "control": {"controller": "task-space impedance on the fingertip in every phase; insert: target = fingertip + scaled policy action "
                                  "(EMA-smoothed), action in the socket frame (D16); scripted phases: clipped targets toward waypoints",
                    "policy_fps": int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))),
                    "pos_action_threshold_m": list(ctrl.pos_action_threshold), "rot_action_threshold_rad": list(ctrl.rot_action_threshold),
                    "pos_action_bounds_m": list(ctrl.pos_action_bounds), "ema_factor": float(ctrl.ema_factor),
                    "joint_commands": "not recorded (the controller outputs joint torques); measured joint states are in joint_state"},
        "target_reached_definition": f"a frame with mean keypoint error < {args_cli.strict_thresh} m (plug at its seated pose)",
        "success_definition": (f"pick, carry and insertion reach the target, then the gripper opens, the arm returns to the fixed home pose and "
                               f"the mean keypoint error is still < {args_cli.strict_thresh} m at the last frame; only successful episodes are "
                               "kept (every attempt is logged in attempts.jsonl)"),
        "attachment_assist": {"used": False, "kind": "none: the plug is held by finger friction only (no attachment, welding or pose "
                              "writes); after the insertion the gripper opens"},
        "environment_defaults": {
            "scene_asset": None, "scene_note": "no background props: default ground plane and sky", "scene_props": [],
            "ground": {"asset": f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd (Isaac Lab GroundPlaneCfg)",
                       "z_m": -0.4, "visible": True, "material": {"type": "asset default", "bound": base.default_materials["ground"]}},
            "table": {"asset": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", "top_z_m": -0.003,
                      "material": {"type": "asset default", "bound": base.default_materials["table"]},
                      "visual_offset_z_m": float(base.TABLE_VISUAL_DROP)},
            "lights": [{"type": "dome", "prim": "/World/Light", "intensity": float(dome.GetIntensityAttr().Get()),
                        "color_rgb": list(dome.GetColorAttr().Get()), "intensity_unit": "Omniverse RTX nits-scaled", "shared_by": "all envs"},
                       {"type": "sphere", "prim": "/World/envs/env_<i>/KeyLight", "intensity": float(key_light.GetIntensityAttr().Get()),
                        "radius_m": float(key_light.GetRadiusAttr().Get()), "color_rgb": [float(v) for v in key_light.GetColorAttr().Get()],
                        "position_env_m": None if key_t is None else [float(v) for v in key_t],
                        "intensity_unit": "Omniverse RTX nits-scaled", "note": "intensity 0 (off) unless the lighting profile turns it on"}],
            "distractors": [],
            "home_pose": {"joints_rad": list(HOME_JOINTS), "gripper": "open (0.04 m per finger)",
                          "note": "same for every task and env: episodes start and end here"},
            "socket_randomization": {"xy_m": list(base.cfg_task.fixed_asset_init_pos_noise[:2]), "yaw_deg": [base.cfg_task.fixed_asset_init_orn_deg,
                                     base.cfg_task.fixed_asset_init_orn_deg + base.cfg_task.fixed_asset_init_orn_range_deg]},
            "plug_placement": {"on": "table (upright)", "x_env_m": list(PLUG_REGION_X), "y_env_m": list(PLUG_REGION_Y),
                               "distribution": "uniform in the region, rejection-sampled until the plug is at least min_distance_from_socket_m from the socket centre (xy)",
                               "min_distance_from_socket_m": PLUG_SOCKET_MIN_DIST, "yaw_deg": [-base.place_yaw_deg, base.place_yaw_deg],
                               "release_height_m": base.drop_height, "settle_s": base.settle_s},
            "handover": {"plug_height_above_entrance_m": base.cfg_task.curriculum_freespace_range, "lateral_noise_m": list(base.cfg_task.held_asset_init_pos_noise[:2])},
        },
    }


def environment_record(defaults: dict, profile: str, params) -> dict:
    """The environment actually used by one episode: the defaults with the profile's sampled values applied."""
    env = {k: v for k, v in defaults.items() if k not in ("socket_randomization", "plug_placement", "handover")}
    env = json.loads(json.dumps(env))
    if profile in ("lighting", "mixed") and params:
        env["lights"][0].update({"intensity": params["dome_intensity"], "color_rgb": params["dome_color"]})
        env["lights"][1].update({"intensity": params["key_intensity"], "radius_m": params["key_radius"],
                                 "position_env_m": params["key_position"], "color_rgb": params["key_color"],
                                 "intensity_unit": "Omniverse RTX nits-scaled", "shared_by": "this env only"})
        env["lights"][1].pop("note", None)
    if profile == "table_color" and params:
        env["table"]["material"] = {"type": "UsdPreviewSurface", "diffuse_color_rgb": params["table_color"], "roughness": params["table_roughness"]}
    if profile in ("object_distractors", "mixed") and params and "distractors" in params:
        env["distractors"] = params["distractors"]
    if profile in ("texture", "mixed") and params:
        if params.get("table_appearance", "texture") == "texture":
            env["table"]["material"] = {"type": "UsdPreviewSurface + UsdUVTexture", "texture": params["table_texture"],
                                        "texture_sha256": params["table_texture_sha256"], "tiling": params["table_tiling"],
                                        "roughness": params["table_roughness"]}
        else:
            env["table"]["material"] = {"type": "UsdPreviewSurface", "diffuse_color_rgb": params["table_color"],
                                        "roughness": params["table_roughness"],
                                        "note": "plain colour: 1x1 white texture tinted by the palette colour"}
        env["ground"]["material"] = {"type": "UsdPreviewSurface + UsdUVTexture", "texture": params["floor_texture"],
                                     "texture_sha256": params["floor_texture_sha256"], "tiling": params["floor_tiling"],
                                     "shared_by": "all envs (one ground plane)"}
        env["back_wall"] = {"exists": False, "note": "this scene has no back wall (upstream texture profile also textures a wall)"}
    if profile in ("scene", "mixed") and params:
        env["scene_asset"] = SCENE_PROPS_SOURCE
        env["scene_note"] = ("per env, upstream Simple_Warehouse props standing on the ground plane 2-6 m in front of the "
                             "front camera, outside the env keep-out box; visual only (collisions, rigid bodies off)")
        env["scene_props"] = params["scene_props"]
    env["randomization"] = {"profile": profile, "actual_parameters": params}
    return env


def visual_assets_dir() -> str:
    """Directory with the visual assets: fabrica_textures/ and warehouse_props/ (each with a manifest.json). Set
    AUTOMATE_VISUAL_ASSETS, or SHARED_ROOT (then <SHARED_ROOT>/data/visual_assets is used)."""
    d = os.environ.get("AUTOMATE_VISUAL_ASSETS") or (os.path.join(os.environ["SHARED_ROOT"], "data/visual_assets") if os.environ.get("SHARED_ROOT") else "")
    if not d or not os.path.isdir(d):
        raise FileNotFoundError("visual assets not found: set AUTOMATE_VISUAL_ASSETS (or SHARED_ROOT with data/visual_assets)")
    return d


def visual_texture_dir() -> str:
    """Read-only copy of the upstream Fabrica textures (manifest.json)."""
    d = os.path.join(visual_assets_dir(), "fabrica_textures")
    if not os.path.isfile(os.path.join(d, "manifest.json")):
        raise FileNotFoundError(f"{d}/manifest.json not found")
    return d


SCENE_PROPS_SOURCE = "Isaac 5.1 Environments/Simple_Warehouse props (the upstream task7rand5 warehouse scenes' props)"


def scene_props() -> list:
    """Props manifest (warehouse_props/manifest.json): id (path under Simple_Warehouse), url, category, gprims of the
    complete asset, bounds (asset frame)."""
    return json.load(open(os.path.join(visual_assets_dir(), "warehouse_props/manifest.json")))["props"]


class TextureRandomizer:
    """Upstream profile ``texture``: per env a table texture (upstream Fabrica table textures) at a random tiling, per
    round one floor texture (shared ground plane). ``either`` (mixed profile): per env the table is plain (a
    table_color palette colour via a 1x1 white texture tinted by it, random roughness) or textured, 50 / 50."""

    TABLE_TILING, FLOOR_TILING = (1.0, 3.0), (1.0, 3.0)

    def __init__(self, seed: int, either: bool = False):
        self.seed, self.round, self.params, self.either = int(seed), -1, [], either

    def __call__(self, env):
        from pxr import Gf, Sdf, UsdShade
        import omni.usd

        env.bind_table_materials()
        self.round += 1
        stage = omni.usd.get_context().get_stage()
        tables, floors = env.texture_files("table"), env.texture_files("floor")
        rel = lambda f: os.path.relpath(f, env.cfg.texture_dir)
        rng, _ = stream_rng(self.seed, "texture", self.round)
        floor = floors[int(rng.integers(len(floors)))]
        ftile = float(rng.uniform(*self.FLOOR_TILING))
        UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/CollectFloor/Tex")).GetInput("file").Set(Sdf.AssetPath(floor))
        UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/CollectFloor/Tiling")).GetInput("scale").Set(Gf.Vec2f(ftile, ftile))
        self.params = []
        for i in range(env.num_envs):
            r, vs = stream_rng(self.seed, "texture", self.round, i)
            plain = self.either and bool(r.random() < 0.5)
            m = f"/World/Looks/CollectTable_{i}"
            if plain:
                f, tile = os.path.join(env.cfg.texture_dir, "white_1x1.png"), 1.0
                tint = [float(v) for v in TABLE_PALETTE[int(r.integers(len(TABLE_PALETTE)))]]
            else:
                f, tile, tint = tables[int(r.integers(len(tables)))], float(r.uniform(*self.TABLE_TILING)), [1.0, 1.0, 1.0]
            rough = float(r.uniform(0.2, 0.9)) if self.either else 0.6
            UsdShade.Shader(stage.GetPrimAtPath(f"{m}/Tex")).GetInput("file").Set(Sdf.AssetPath(f))
            UsdShade.Shader(stage.GetPrimAtPath(f"{m}/Tex")).GetInput("scale").Set(Gf.Vec4f(*tint, 1.0))
            UsdShade.Shader(stage.GetPrimAtPath(f"{m}/Tiling")).GetInput("scale").Set(Gf.Vec2f(tile, tile))
            UsdShade.Shader(stage.GetPrimAtPath(f"{m}/Shader")).GetInput("roughness").Set(rough)
            p = {"visual_seed": vs, "floor_texture": rel(floor), "floor_texture_sha256": sha256_file(floor), "floor_tiling": ftile,
                 "table_roughness": rough}
            if plain:
                p.update({"table_appearance": "color", "table_color": tint})
            else:
                p.update({"table_appearance": "texture", "table_texture": rel(f), "table_texture_sha256": sha256_file(f),
                          "table_tiling": tile})
            self.params.append(p)

    def params_for(self, i: int) -> dict:
        return self.params[i]

    def visual_seed_for(self, i: int):
        return self.params[i]["visual_seed"]


class SceneRandomizer:
    """Profile ``scene`` (replaces the upstream stage-wide warehouse: with 12 m env spacing most envs stood outside its
    24 x 36 m floor and saw nothing): per env and round 3-6 upstream Simple_Warehouse props (shelves, pallets, boxes,
    forklift, ...) standing on the ground 2-6 m from the front camera, centre within its horizontal field of view;
    their oriented footprints clear the env keep-out box and each other. A position inside another env's front view
    within its far plane is rejected, so envs never see each other's props."""

    N_PROPS, DIST_M, VIEW_MARGIN_DEG = (3, 6), (2.0, 6.0), 0.0

    def __init__(self, seed: int):
        self.seed, self.round, self.params = int(seed), -1, []

    def __call__(self, env):
        self.round += 1
        props = env.cfg.scene_props
        eye = np.array(env.cfg.front_camera.offset.pos[:2], dtype=float)
        tgt = np.array(env.front_camera_aim["target_env_m"][:2], dtype=float)
        heading = float(np.arctan2(*(tgt - eye)[::-1]))
        cam = env.cfg.front_camera.spawn
        half = np.arctan(cam.horizontal_aperture / 2 / cam.focal_length) + np.radians(self.VIEW_MARGIN_DEG)
        lo, hi = np.array(env.KEEP_OUT_LO[:2]), np.array(env.KEEP_OUT_HI[:2])
        # footprint radius of each prop: half the diagonal of its xy bounds, about the asset origin
        rad = [float(np.linalg.norm(np.maximum(np.abs(p["lo"][:2]), np.abs(p["hi"][:2])))) for p in props]
        origins = env.scene.env_origins[:, :2].cpu().numpy()
        far = float(cam.clipping_range[1]) + 0.5

        def seen_by_other(i, xy, r):
            """Would the prop (footprint circle) be inside another env's front view within its far plane?"""
            w = origins[i] + xy
            for j in range(env.num_envs):
                if j == i:
                    continue
                d = w - (origins[j] + eye)
                dist = float(np.linalg.norm(d))
                if dist - r >= far:
                    continue
                ang = abs((np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi)
                if ang <= half + np.arcsin(min(1.0, r / max(dist, 1e-6))):
                    return True
            return False

        def footprint(k, xy, yaw, pad=0.1):
            """Corners of prop k's xy bounds (padded) at pose (xy, yaw), env frame."""
            (x0, y0), (x1, y1) = np.array(props[k]["lo"][:2]) - pad, np.array(props[k]["hi"][:2]) + pad
            c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            return np.array([[x, y] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]) @ np.array([[c, s_], [-s_, c]]) + xy

        def overlap(P, Q):
            """Separating-axis test for two convex quadrilaterals."""
            for poly in (P, Q):
                for e in range(4):
                    ax = np.array([-(poly[(e + 1) % 4] - poly[e])[1], (poly[(e + 1) % 4] - poly[e])[0]])
                    if (P @ ax).max() < (Q @ ax).min() or (Q @ ax).max() < (P @ ax).min():
                        return False
            return True

        keep_out = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        self.params, placements = [], []
        for i in range(env.num_envs):
            rng, vs = stream_rng(self.seed, "scene", self.round, i)
            n = int(rng.integers(self.N_PROPS[0], self.N_PROPS[1] + 1))
            placed, rec, feet = [], [], []
            # category first, then an asset within it (2026-10-10; a uniform permutation of the assets made a category's
            # frequency proportional to its number of assets)
            cats = sorted({p["category"] for p in props})
            remaining = {ct: [int(x) for x in rng.permutation([k for k, p in enumerate(props) if p["category"] == ct])] for ct in cats}
            order = []
            while any(remaining.values()):
                avail = [ct for ct in cats if remaining[ct]]
                order.append(remaining[avail[int(rng.integers(len(avail)))]].pop())
            for k in order + [k for k in order]:  # a second pass over the props not placed yet, if the first fell short
                if len(placed) >= n:
                    break
                if any(q[0] == k for q in placed):
                    continue
                for _ in range(200):
                    yaw = float(rng.uniform(0.0, 360.0))
                    r_, a_ = rng.uniform(*self.DIST_M), heading + rng.uniform(-half, half)
                    xy = eye + r_ * np.array([np.cos(a_), np.sin(a_)])
                    foot = footprint(k, xy, yaw)
                    # its oriented footprint clears the keep-out box and the props placed so far
                    if (not overlap(foot, keep_out) and not any(overlap(foot, f) for f in feet)
                            and not seen_by_other(i, xy, rad[k])):
                        break
                else:
                    continue
                placed.append((int(k), float(xy[0]), float(xy[1]), yaw)); feet.append(foot)
                h = np.radians(yaw) / 2
                rec.append({"asset_id": props[k]["id"], "url": props[k]["url"], "category": props[k]["category"],
                            "prim": f"/World/envs/env_<i>/SceneProp_{int(k)}",
                            "position_m": [float(xy[0]), float(xy[1]), env.GROUND_Z - float(props[k]["lo"][2])],
                            "orientation_wxyz": [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))], "scale_xyz": [1.0, 1.0, 1.0],
                            "yaw_deg": yaw, "footprint_radius_m": rad[k], "bounds_asset_frame_m": [props[k]["lo"], props[k]["hi"]],
                            "footprint_env_xy_m": np.round(footprint(k, xy, yaw, pad=0.0), 4).tolist(),
                            "distance_from_front_camera_m": float(np.linalg.norm(xy - eye))})
            if len(placed) < self.N_PROPS[0]:
                raise RuntimeError(f"scene: only {len(placed)} props placed for env {i}")
            placements.append(placed)
            self.params.append({"visual_seed": vs, "scene_props": rec})
        env.set_scene_props(placements)

    def params_for(self, i: int) -> dict:
        return self.params[i]

    def visual_seed_for(self, i: int):
        return self.params[i]["visual_seed"]


class MixedRandomizer:
    """All five visual domains at once: lighting, object_distractors, scene, table (palette colour or texture, 50 / 50).
    The parts draw from separate streams of the same run seed."""

    def __init__(self, seed: int):
        self.parts = [LightingRandomizer(seed), DistractorRandomizer(seed), TextureRandomizer(seed, either=True), SceneRandomizer(seed)]

    def __call__(self, env):
        for part in self.parts:
            part(env)

    def params_for(self, i: int) -> dict:
        out = {}
        for part in self.parts:
            p = dict(part.params_for(i))
            out.setdefault("visual_seeds", {})[type(part).__name__] = p.pop("visual_seed")
            out.update(p)
        return out

    def visual_seed_for(self, i: int):
        return {type(p).__name__: p.visual_seed_for(i) for p in self.parts}


# upstream Fabrica table_surface palette (task7rand5)
TABLE_PALETTE = [[0.14, 0.14, 0.14], [0.28, 0.31, 0.32], [0.22, 0.30, 0.25], [0.33, 0.24, 0.25], [0.25, 0.27, 0.36],
                 [0.40, 0.36, 0.25]]


class TableColorRandomizer:
    """README profile ``table_color``: every env's table gets its own random base colour and roughness per round."""

    def __init__(self, seed: int):
        self.seed, self.round, self.params = int(seed), -1, []

    def __call__(self, env):
        from pxr import Gf, UsdShade
        import omni.usd

        env.bind_table_materials()
        self.round += 1
        stage = omni.usd.get_context().get_stage()
        self.params = []
        for i in range(env.num_envs):
            rng, vs = stream_rng(self.seed, "table_color", self.round, i)
            color = [float(v) for v in TABLE_PALETTE[int(rng.integers(len(TABLE_PALETTE)))]]
            rough = float(rng.uniform(0.2, 0.9))
            shader = UsdShade.Shader(stage.GetPrimAtPath(f"/World/Looks/CollectTable_{i}/Shader"))
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))
            shader.GetInput("roughness").Set(rough)
            self.params.append({"visual_seed": vs, "table_color": color, "table_roughness": rough})

    def params_for(self, i: int) -> dict:
        return self.params[i]

    def visual_seed_for(self, i: int):
        return self.params[i]["visual_seed"]


class DistractorRandomizer:
    """README profile ``object_distractors``: 0..max visual-only boxes / cylinders per env standing on the table top,
    random size, colour and yaw, at least KEEP_OUT m (xy) from the socket and SIGHT_KEEP_OUT m from the front camera's
    line of sight to the socket, so they never cover the insertion. The prims are a unit cube / a unit-height cylinder
    of diameter 1, scaled by size."""

    TABLE_TOP_Z = -0.003  # env frame, table collision top
    # full-process episodes: the arm picks the plug on the table and carries it to the socket, so the distractors keep
    # clear of the socket, the plug, the segment between them and the global camera's lines of sight to both
    X_RANGE, Y_RANGE, KEEP_OUT, SIGHT_KEEP_OUT = (0.3, 0.95), (-0.45, 0.45), 0.10, 0.08

    def __init__(self, seed: int):
        self.seed, self.round, self.params = int(seed), -1, []

    def __call__(self, env):
        from pxr import Gf, UsdGeom, UsdShade
        import omni.usd

        self.round += 1
        stage = omni.usd.get_context().get_stage()
        socket_xy = (env._fixed_asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).cpu().numpy()
        plug_xy = env.layout["plug_pose_env"][:, :2].cpu().numpy()
        cam_xy = np.array(env.cfg.front_camera.offset.pos[:2], dtype=float)

        def seg_dist(p, a, b):  # xy distance from p to the segment a -> b
            d = b - a
            t = float(np.clip(np.dot(p - a, d) / max(float(np.dot(d, d)), 1e-9), 0.0, 1.0))
            return float(np.linalg.norm(p - (a + t * d)))

        def sight_dist(p, s):  # xy distance from p to the segment front camera -> s
            return seg_dist(p, cam_xy, s)

        self.params = []
        for i in range(env.num_envs):
            rng, vs = stream_rng(self.seed, "distractors", self.round, i)
            n = int(rng.integers(0, env.cfg.max_distractors + 1))
            placed = []
            for k in range(env.cfg.max_distractors):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Distractor_{k}")
                img = UsdGeom.Imageable(prim)
                if k >= n:
                    img.MakeInvisible()
                    continue
                for _ in range(50):
                    xy = np.array([rng.uniform(*self.X_RANGE), rng.uniform(*self.Y_RANGE)])
                    if (np.linalg.norm(xy - socket_xy[i]) >= self.KEEP_OUT and np.linalg.norm(xy - plug_xy[i]) >= self.KEEP_OUT
                            and seg_dist(xy, socket_xy[i], plug_xy[i]) >= self.SIGHT_KEEP_OUT
                            and sight_dist(xy, socket_xy[i]) >= self.SIGHT_KEEP_OUT and sight_dist(xy, plug_xy[i]) >= self.SIGHT_KEEP_OUT):
                        break
                else:
                    img.MakeInvisible()
                    continue
                size = rng.uniform(0.02, 0.06, 3)
                yaw = float(rng.uniform(0, 360))
                color = [float(v) for v in rng.uniform(0.05, 0.95, 3)]
                box = bool(rng.random() < 0.5)  # type drawn per distractor (2026-10-10; was fixed by slot parity)
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Distractor_{k}/box")).MakeVisible() if box else \
                    UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Distractor_{k}/box")).MakeInvisible()
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Distractor_{k}/cylinder")).MakeInvisible() if box else \
                    UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Distractor_{k}/cylinder")).MakeVisible()
                pos = [float(xy[0]), float(xy[1]), self.TABLE_TOP_Z + float(size[2]) / 2]
                xf = UsdGeom.XformCommonAPI(prim)
                xf.SetTranslate(Gf.Vec3d(*pos))
                xf.SetRotate(Gf.Vec3f(0.0, 0.0, yaw))
                xf.SetScale(Gf.Vec3f(*[float(v) for v in size]))
                UsdShade.Shader(stage.GetPrimAtPath(f"/World/Looks/CollectDistractor_{i}_{k}/Shader")).GetInput("diffuseColor").Set(Gf.Vec3f(*color))
                img.MakeVisible()
                h = np.radians(yaw) / 2
                placed.append({
                    "asset_id": "usd_geom_cube_unit" if box else "usd_geom_cylinder_unit",
                    "category": "primitive_box" if box else "primitive_cylinder",
                    "prim": f"/World/envs/env_<i>/Distractor_{k}/{'box' if box else 'cylinder'}",
                    "position_m": pos, "orientation_wxyz": [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))],
                    "scale_xyz": [float(v) for v in size],
                    "dimensions_m": ({"x": float(size[0]), "y": float(size[1]), "z": float(size[2])} if box else
                                     {"diameter_x": float(size[0]), "diameter_y": float(size[1]), "height": float(size[2])}),
                    "yaw_deg": yaw, "color_rgb": color, "material": "UsdPreviewSurface, roughness 0.5",
                    "physics": "visual only (no collision, no rigid body)"})
            self.params.append({"visual_seed": vs, "distractors": placed})

    def params_for(self, i: int) -> dict:
        return self.params[i]

    def visual_seed_for(self, i: int):
        return self.params[i]["visual_seed"]


BASE_ID_PREFIX = f"{args_cli.production_id}/automate"  # base trajectory / layout ids: production-scoped, shared by the mixed and single batches

# scene / mixed: an episode whose first front frame shows (almost) none of its warehouse props -- all of them hidden
# behind the robot arm -- is not kept. Prop pixels: front depth 1.5-10 m, > 5 cm above the ground plane, not the table
# top (the release gate uses the same test; in the other profiles it finds 0 such pixels)
MIN_PROP_PIXEL_FRACTION = 0.002
# the gripper must not press on the table to reach the policy grasp (user 2026-10-09: on tasks whose grasp is a few mm
# above the plug's bottom the fingers were pushed into the table). Upward (world +z) net contact force on the two
# fingers summed, during approach and grasp; tasks with the grasp >= 15 mm above the table stay below 1.7 N (clean
# smokes <= 0.55 N), fingers resting on the table give 1.3-7 N (00186, grasp 0.3 mm above the table), pressing 20-140 N
FINGER_TABLE_FORCE_N = 3.0
# deeper-than-goal success (user 2026-10-09, for assemblies whose AutoMate goal pose has no physical stop, e.g. 00007 /
# 00486: after the release the plug moves further in): COLLECT_DEEPER_OK=1 also accepts, at the last frame, a plug whose
# base is at most DEEPER_OUT_M out of the goal along the socket axis (any depth further in), at most DEEPER_LAT_M off
# laterally and tilted at most DEEPER_TILT_DEG; the target must still be reached (mean keypoint error < strict_thresh)
# before the release, which is what starts the release
DEEPER_OK = os.environ.get("COLLECT_DEEPER_OK") == "1"
# several runs of one task writing into the same directory at once (COLLECT_PARALLEL=1, queue SLOTS > 1, 2026-10-09):
# episode indices are taken by an atomic mkdir of <out>/.idx/<idx>; .incomplete directories of other runs are left
# alone; every run logs its attempts to attempts.seed<seed>.jsonl; the success count is re-read from disk before
# every round, so the runs stop together at the target
PARALLEL = os.environ.get("COLLECT_PARALLEL") == "1"
DEEPER_OUT_M, DEEPER_LAT_M, DEEPER_TILT_DEG = 0.003, 0.003, 5.0


def seat_pose(plug_pose, socket_pose, base_local):
    """Plug base relative to its goal (the socket frame, AutoMate fixed_success_pos_local = 0): axial (m, + = out of
    the socket along its z axis), lateral (m) and tilt (deg) of the plug axis from the socket axis."""
    def qmul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

    def rot(q, v):
        u = np.asarray(q[1:]); return v + 2 * np.cross(u, np.cross(u, v) + q[0] * v)
    pp, sp = np.asarray(plug_pose, dtype=np.float64), np.asarray(socket_pose, dtype=np.float64)
    base = pp[:3] + rot(pp[3:7], np.asarray(base_local, dtype=np.float64))
    sq_inv = np.array([sp[3], -sp[4], -sp[5], -sp[6]])
    rel = rot(sq_inv, base - sp[:3])
    q = qmul(sq_inv, pp[3:7])
    tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0))))
    return float(rel[2]), float(np.linalg.norm(rel[:2])), tilt
GROUND_Z_M, TABLE_TOP_Z_M = -0.4, -0.003


def prop_pixel_fraction(depth: np.ndarray, pose, K) -> float:
    """Share of front pixels on something standing on the ground 1.5-10 m away (env frame pose: x y z qw qx qy qz)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    w, x, y, z = np.asarray(pose[3:7], dtype=np.float64) / np.linalg.norm(pose[3:7])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    height = pose[2] + depth * ((u - K[0][2]) / K[0][0] * R[2, 0] + (v - K[1][2]) / K[1][1] * R[2, 1] + R[2, 2])
    far = (depth >= 1.5) & (depth < 9.99)
    return float((far & (height > GROUND_Z_M + 0.05) & (np.abs(height - TABLE_TOP_Z_M) > 0.015)).mean())


def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t


class EpisodeCollector:
    """Turns the env's frame stream into one single-arm recorder per env and episode (full process).

    A round starts with the env reset (frame tag ``reset``: arm at home, plug on the table); the scripted approach /
    grasp / lift / carry frames follow inside the reset, then the policy's insert frames. Per env: insert -> first
    frame with mean keypoint error < strict_thresh -> ``env.start_release`` (scripted release, lift, return home) ->
    the env reports script state 5 (home, still) -> the episode ends and is kept if the plug is still seated. A plug
    that fell over or was not lifted, a scripted waypoint the arm stalled far from (servo_stalled), fingers pressed on the table while reaching the grasp (finger_pressed_table), an insertion longer than max_insert_frames, or a return longer than
    max_return_frames fail the episode (the env is frozen until the round ends). The round ends when every env's
    episode has ended; the main loop then resets.
    """

    def __init__(self, env, out_dir: Path, static_metadata: dict, target: int, max_insert_frames: int, max_return_frames: int, visual=None):
        self.env, self.out_dir, self.static = env, out_dir, static_metadata
        self.n, self.target, self.visual = env.num_envs, target, visual
        self.max_insert_frames, self.max_return_frames = max_insert_frames, max_return_frames
        self.recorders = [None] * self.n
        self.pending = [None] * self.n  # last captured frame of each env, not yet written
        self.insert_frames = [0] * self.n
        self.return_frames = [0] * self.n
        self.finger_up_max = [0.0] * self.n  # max upward finger contact force during approach / grasp (N)
        self.events = [None] * self.n  # frame indices of the phase changes of the current episode
        self.frame_idx = [0] * self.n
        self.episode_ids = [None] * self.n
        self.settle_renders = [None] * self.n
        self.layouts = [None] * self.n
        self.round = -1
        self.stop = False
        self.attempts = 0
        done, partial = [], []
        for d in sorted(out_dir.glob("episode_*_cartesian_raw*")):  # incl. <name>.incomplete from a killed run
            (done if d.name.endswith("_cartesian_raw") and (d / "metadata.json").exists() else partial).append(d)
        # every episode index ever taken keeps its claim <out>/.idx/<idx>, also after the episode is deleted or moved
        # to quarantine, so a later run never reuses the name of an episode that existed before (2026-10-10: the trim
        # to 200 deleted the highest indices of some tasks, which the next run would otherwise have taken again)
        (out_dir / ".idx").mkdir(exist_ok=True)
        claims = [int(c.name) for c in (out_dir / ".idx").iterdir() if c.name.isdigit()]
        if not PARALLEL:
            for d in partial:
                shutil.rmtree(d, ignore_errors=True)
        self.successes = len(done)
        self.next_episode_idx = 1 + max([int(d.name.split("_")[1]) for d in done + partial] + claims, default=-1)
        self.stop = self.successes >= self.target
        log_name = f"attempts.seed{args_cli.seed}.jsonl" if PARALLEL else "attempts.jsonl"
        self.attempt_log = open(out_dir / log_name, "a", encoding="utf-8")
        self.writers = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"writer{i}") for i in range(self.n)]
        self.pending_writes = [[] for _ in range(self.n)]

    def _reserve_idx(self) -> int:
        """Next free episode index, taken by an atomic mkdir of <out>/.idx/<idx>: two runs of the same task never use
        the same episode name, and no index is used twice over the life of the dataset."""
        while True:
            idx = self.next_episode_idx
            self.next_episode_idx += 1
            try:
                os.mkdir(self.out_dir / ".idx" / f"{idx:06d}")
                return idx
            except FileExistsError:
                continue

    def refresh(self):
        """COLLECT_PARALLEL: re-read the number of kept episodes (all runs) before a round; stop at the target."""
        if PARALLEL:
            self.successes = sum(1 for d in self.out_dir.glob("episode_*_cartesian_raw") if (d / "metadata.json").exists())
            self.stop = self.successes >= self.target

    @property
    def round_done(self) -> bool:
        return all(r is None for r in self.recorders)

    def _drain(self, i: int):
        futs, self.pending_writes[i] = self.pending_writes[i], []
        for f in futs:
            f.result()

    @staticmethod
    def _env_slice(frame: dict, i: int) -> dict:
        out = {k: frame[k] for k in ("tag", "sim_step")}
        for k in ("eef_pos", "eef_quat", "finger_pos", "arm_pos", "arm_vel", "arm_effort", "keypoint_error",
                  "policy_action", "contact_force", "plug_pose", "socket_pose", "phase", "gripper_cmd", "script", "pick_failed"):
            out[k] = None if frame.get(k) is None else frame[k][i]
        for key in CAMERA_KEYS:
            out[key] = {"rgb": frame[key]["rgb"][i], "depth": frame[key]["depth"][i, ..., 0], "pose": frame[key]["pose"][i]}
        return out

    def __call__(self, env, frame: dict):
        from roboassemblybench.library.automate.assembly_collect_env import PROF
        t0 = time.perf_counter()
        host = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in frame.items() if k not in CAMERA_KEYS}
        for key in CAMERA_KEYS:
            host[key] = {"rgb": frame[key]["rgb"].detach().cpu().numpy(), "depth": frame[key]["depth"].detach().cpu().numpy(),
                         "pose": frame[key]["pose"].detach().cpu().numpy()}
        PROF["gpu_to_cpu"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        self._handle(host)
        PROF["write"] += time.perf_counter() - t0

    def _mask(self, i: int):
        m = torch.zeros(self.n, dtype=torch.bool, device=self.env.device)
        m[i] = True
        return m

    def _handle(self, host: dict):
        if host["tag"] == "reset":
            self.round += 1
            from roboassemblybench.library.automate.assembly_collect_env import PROF
            if PROF and os.environ.get("COLLECT_PROFILE"):
                print("[PROFILE] round", self.round, " ".join(f"{k}={v:.1f}s" for k, v in sorted(PROF.items())), flush=True)
                PROF.clear()
        for i in range(self.n):
            f = self._env_slice(host, i)
            if f["tag"] == "reset":
                if self.recorders[i] is not None:
                    self._fail(i, f_last=self.pending[i], reason=f"cut_by_round_end ({PHASES[int(self.pending[i]['phase'])]})")
                if not self.stop:
                    self._start(i)
                    if getattr(self.env, "last_settle_capped", False):  # the round's images never settled: not kept
                        self._fail(i, f_last=f, reason="visuals_not_settled")
                    elif args_cli.visual_profile in ("scene", "mixed"):
                        key = "observation.images.front"
                        frac = prop_pixel_fraction(f[key]["depth"], f[key]["pose"], self.static["cameras"][key]["intrinsics"])
                        if frac < MIN_PROP_PIXEL_FRACTION:
                            self._fail(i, f_last=f, reason=f"scene_props_not_visible (prop pixels {frac:.4f})")
                self.pending[i] = f
                continue
            if self.recorders[i] is None:
                continue
            prev = self.pending[i]
            ph_prev = int(prev["phase"])
            pa = f["policy_action"] if (ph_prev == PH["insert"] and f["policy_action"] is not None) else np.zeros(6, dtype=np.float32)
            self._write(i, prev, policy_action=pa, phase=ph_prev, grip=float(prev["gripper_cmd"]))
            self.frame_idx[i] += 1
            ph = int(f["phase"])
            if ph != ph_prev:
                self.events[i][PHASES[ph]] = self.frame_idx[i]
            if ph in (PH["approach"], PH["grasp"]) and f["contact_force"] is not None:
                cf = np.asarray(f["contact_force"], dtype=np.float64).reshape(9)
                self.finger_up_max[i] = max(self.finger_up_max[i], max(cf[2], 0.0) + max(cf[5], 0.0))
            if ph == PH["insert"]:
                if bool(f["pick_failed"]):
                    reason = ("plug_fell_over" if float(self.env.tilt_settled_deg[i]) > self.env.topple_deg else
                              "servo_stalled" if bool(self.env.servo_stalled[i]) else "plug_not_lifted")
                    self._fail(i, f_last=f, reason=reason)
                    self.env.freeze_envs(self._mask(i))
                    continue
                if self.finger_up_max[i] > FINGER_TABLE_FORCE_N:
                    self._fail(i, f_last=f, reason=f"finger_pressed_table ({self.finger_up_max[i]:.1f} N)")
                    self.env.freeze_envs(self._mask(i))
                    continue
                self.insert_frames[i] += 1
                if float(f["keypoint_error"]) < args_cli.strict_thresh:
                    self.events[i]["target_reached"] = self.frame_idx[i]
                    self.env.start_release(self._mask(i))
                elif self.insert_frames[i] > self.max_insert_frames:
                    self._fail(i, f_last=f, reason="not_inserted")
                    self.env.freeze_envs(self._mask(i))
                    continue
            elif ph in (PH["release"], PH["return"]):
                self.return_frames[i] += 1
                if int(f["script"]) == 5:  # back home and still: the episode ends with this frame
                    self._succeed_or_fail(i, terminal=f)
                    continue
                if self.return_frames[i] > self.max_return_frames:
                    e = self.env
                    dp = float((e.fingertip_midpoint_pos[i] - e.home_pos).norm())
                    dq = torch_utils.quat_mul(e.home_quat.unsqueeze(0), torch_utils.quat_conjugate(e.fingertip_midpoint_quat[i:i + 1]))
                    ang = float(2 * torch.acos(dq[0, 0].abs().clamp(max=1.0)))
                    self._fail(i, f_last=f, reason=f"return_timeout (script state {int(f['script'])}, home err {dp * 1000:.1f} mm {ang:.3f} rad)")
                    self.env.freeze_envs(self._mask(i))
                    continue
            self.pending[i] = f

    def freeze_failed(self):
        """Envs without a running episode (failed at the reset / pick, or the target reached) hold still."""
        m = torch.tensor([r is None for r in self.recorders], dtype=torch.bool, device=self.env.device)
        self.env.freeze_envs(m | self.env.pick_failed)

    def _start(self, i: int):
        idx = self._reserve_idx()
        self.episode_ids[i] = (idx, self.round)
        self.settle_renders[i] = int(getattr(self.env, "last_settle_renders", -1))
        self.recorders[i] = SingleArmCartesianEpisodeRecorder(output_dir=self.out_dir, episode_idx=idx)
        self.insert_frames[i] = self.return_frames[i] = self.frame_idx[i] = 0
        self.finger_up_max[i] = 0.0
        self.events[i] = {"approach": 0}
        self.layouts[i] = {k: (_np(v[i]).tolist() if torch.is_tensor(v) else v) for k, v in self.env.layout.items()}

    def _write(self, i: int, f: dict, policy_action, phase: int, grip: float):
        self.pending_writes[i] = [x for x in self.pending_writes[i] if not x.done() or x.exception()]
        while len(self.pending_writes[i]) >= 16:
            self.pending_writes[i].pop(0).result()
        self.pending_writes[i].append(self.writers[i].submit(self._add_frame, self.recorders[i], f, policy_action, phase, grip))

    @staticmethod
    def _add_frame(rec, f: dict, policy_action, phase: int, grip: float):
        rec.add_frame(
            eef_position=f["eef_pos"],
            eef_quaternion_wxyz=f["eef_quat"],
            gripper_open=float(np.mean(f["finger_pos"]) / FINGER_OPEN_M),
            gripper_command=grip,
            policy_action=policy_action,
            simulation_step=f["sim_step"],
            joint_state=f["arm_pos"],
            joint_velocity=f["arm_vel"],
            joint_effort=f["arm_effort"],
            rgb={k: f[k]["rgb"] for k in CAMERA_KEYS},
            depth_m={k: f[k]["depth"] for k in CAMERA_KEYS},
            phase_index=phase,
            keypoint_error=float(f["keypoint_error"]),
            contact_force=f["contact_force"],
            object_poses={"plug": f["plug_pose"], "socket": f["socket_pose"]},
            camera_poses={k: f[k]["pose"] for k in CAMERA_KEYS},
        )

    def _layout_seed(self, i: int, rnd: int) -> int:
        return int(args_cli.seed) * 1_000_000 + rnd * self.n + i

    def _log(self, entry: dict):
        self.attempts += 1
        self.attempt_log.write(json.dumps(entry) + "\n")
        self.attempt_log.flush()

    def _fail(self, i: int, f_last: dict, reason: str):
        rec, (idx, rnd) = self.recorders[i], self.episode_ids[i]
        self.recorders[i] = None
        self._drain(i)
        if os.environ.get("COLLECT_KEEP_FAILED") and rec._frame_counts[CAMERA_KEYS[0]] > 1:  # debugging only
            meta = dict(self.static)
            meta.update({"seed": int(args_cli.seed), "round": rnd, "env_index": i, "failure_reason": reason,
                         "T_world_env": None, "layout": self.layouts[i]})
            rec.finalize(next_eef_position=f_last["eef_pos"], next_eef_quaternion_wxyz=f_last["eef_quat"],
                         next_gripper_command=float(f_last["gripper_cmd"]), task=args_cli.task_text, metadata=meta,
                         metrics={"success": False, "reason": reason, "phase_start_frames": self.events[i]})
            os.rename(rec.episode_dir, str(rec.episode_dir).replace("episode_", "FAILED_episode_"))
        else:
            rec.discard()
            shutil.rmtree(rec.episode_dir, ignore_errors=True)
        self._log({"episode_idx": idx, "round": rnd, "env": i, "layout_seed": self._layout_seed(i, rnd), "success": False,
                   "reason": reason, "kp_last_m": float(f_last["keypoint_error"]), "events": self.events[i], "time": time.time()})

    def _succeed_or_fail(self, i: int, terminal: dict):
        # the last recorded frame (pending[i], already written) and the state of the terminal frame must both be seated
        kp_final = max(float(terminal["keypoint_error"]), float(self.pending[i]["keypoint_error"]))
        base_local = _np(self.env.held_base_pos_local[i])
        seats = [seat_pose(f["plug_pose"], f["socket_pose"], base_local) for f in (terminal, self.pending[i])]
        seat = {"axial_m": max(x[0] for x in seats), "lateral_m": max(x[1] for x in seats), "tilt_deg": max(x[2] for x in seats)}
        seated = kp_final < args_cli.strict_thresh
        rule = "keypoint"
        if not seated and DEEPER_OK and seat["axial_m"] <= DEEPER_OUT_M and seat["lateral_m"] <= DEEPER_LAT_M and seat["tilt_deg"] <= DEEPER_TILT_DEG:
            seated, rule = True, "deeper_ok"
        if not seated:
            self._fail(i, f_last=terminal, reason=f"unseated_after_release (axial {seat['axial_m'] * 1e3:+.1f} mm, lateral "
                                                  f"{seat['lateral_m'] * 1e3:.1f} mm, tilt {seat['tilt_deg']:.1f} deg)")
            return
        rec, (idx, rnd) = self.recorders[i], self.episode_ids[i]
        self.recorders[i] = None
        self._drain(i)
        pick = {k: float(_np(v[i])) for k, v in self.env.pick_info.items() if torch.is_tensor(v)}
        pick["finger_up_contact_force_max_n"] = float(self.finger_up_max[i])
        metrics = {"success": True, "target_reached": True, "attachment_assist_used": False, "kp_final_m": kp_final,
                   "strict_thresh_m": args_cli.strict_thresh, "phase_start_frames": self.events[i], "pick": pick,
                   "seat_final": seat, "success_rule": rule}
        meta = dict(self.static)
        params = None if self.visual is None else self.visual.params_for(i)
        origin = [float(v) for v in self.env.scene.env_origins[i].tolist()]
        meta.update({"seed": int(args_cli.seed), "layout_seed": self._layout_seed(i, rnd), "round": rnd, "env_index": i,
                     "env_origin_world_m": origin,
                     "visual_settle_renders": self.settle_renders[i],
                     "T_world_env": [[1.0, 0.0, 0.0, origin[0]], [0.0, 1.0, 0.0, origin[1]], [0.0, 0.0, 1.0, origin[2]], [0.0, 0.0, 0.0, 1.0]],
                     "visual_seed": None if self.visual is None else self.visual.visual_seed_for(i),
                     "domain_randomization": {"profile": args_cli.visual_profile, "params": params},
                     "home_pose_fingertip_env": {"position_m": [float(v) for v in self.env.home_pos.tolist()],
                                                 "orientation_wxyz": [float(v) for v in self.env.home_quat.tolist()],
                                                 "joints_rad": list(HOME_JOINTS)},
                     "layout": {**self.layouts[i], "frame": "env (plug_pose_socket_frame: socket frame)",
                                "pose_format": "x y z w x y z (m, quaternion wxyz)"},
                     "generation": {"generation_type": "new_motion", "variant_index": 0,
                                    "base_trajectory_id": f"{BASE_ID_PREFIX}/{args_cli.assembly_id}/layout_{self._layout_seed(i, rnd)}",
                                    "layout_id": f"{BASE_ID_PREFIX}/{args_cli.assembly_id}/layout_{self._layout_seed(i, rnd)}",
                                    "source_episode_id": f"{self.static['batch_id']}/automate/{args_cli.assembly_id}/{args_cli.visual_profile}/episode_{idx:06d}"},
                     "environment": environment_record(self.static["environment_defaults"], args_cli.visual_profile, params)})
        rec.finalize(next_eef_position=terminal["eef_pos"], next_eef_quaternion_wxyz=terminal["eef_quat"],
                     next_gripper_command=float(terminal["gripper_cmd"]), task=args_cli.task_text, metadata=meta, metrics=metrics)
        self.successes += 1
        self._log({"episode_idx": idx, "round": rnd, "env": i, "layout_seed": self._layout_seed(i, rnd), "success": True,
                   "kp_final_m": kp_final, "frames": rec._frame_counts[CAMERA_KEYS[0]], "events": self.events[i], "time": time.time()})
        if self.successes >= self.target:
            self.stop = True

    def close(self):
        for i, rec in enumerate(self.recorders):
            if rec is not None:
                self._drain(i)
                rec.discard()
                shutil.rmtree(rec.episode_dir, ignore_errors=True)
                self.recorders[i] = None
        self.attempt_log.close()
        for w in self.writers:
            w.shutdown(wait=True)


@hydra_task_config(TASK, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.tasks[env_cfg.task_name].if_sbc = False
    env_cfg.bind_table_material = args_cli.visual_profile in ("table_color", "texture", "mixed")
    env_cfg.table_texture = env_cfg.floor_texture = args_cli.visual_profile in ("texture", "mixed")
    env_cfg.texture_dir = visual_texture_dir()
    env_cfg.scene_props = scene_props() if args_cli.visual_profile in ("scene", "mixed") else []
    # the lights change every round: wait out the renderer's slow convergence before recording. 300 renders (2026-10-01)
    # cut to 100 on 2026-10-02: frozen-physics probes (mixed, 4 rounds each) left the same residual after 100, 150 or
    # 300 renders (front p90 ~1-2 %, mostly render noise) and clearly more after 0; 100 renders take ~8 s per round
    env_cfg.settle_min_renders = 100 if args_cli.visual_profile in ("lighting", "mixed") else 0

    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(TASK, cfg=env_cfg)
    base = env.unwrapped
    out_dir = Path(args_cli.out_dir) / args_cli.assembly_id / args_cli.visual_profile
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = base.cameras
    static = {
        "assembly_id": args_cli.assembly_id,
        "visual_profile": args_cli.visual_profile,
        "robot": "franka_panda_single_arm",
        "checkpoint": {
            "path": str(Path(args_cli.checkpoint_source or args_cli.checkpoint).resolve()),
            "sha256": sha256_file(args_cli.checkpoint_source or args_cli.checkpoint),
            "loaded_copy_path": str(Path(args_cli.checkpoint).resolve()),
            "loaded_copy_sha256": sha256_file(args_cli.checkpoint),
        },
        "policy": {"deterministic": True, "sbc_curriculum": False, "observation": "21-dim relative (joint angles, plug pose in the socket frame, plug pose in the fingertip frame; D16)",
                   "action_frame": "socket frame (D16)", "phase": "insert only; every other phase is scripted"},
        "phases": {"names": list(PHASES), "control": {"approach": "scripted", "grasp": "scripted", "lift": "scripted", "carry": "scripted",
                                                      "insert": "policy", "release": "scripted", "return": "scripted"},
                   "description": {"approach": "home pose -> 8 cm above the policy grasp on the plug standing on the table -> down to the grasp pose, gripper fully open",
                                   "grasp": "fingers to the training opening, then closed until they stop", "lift": "8 cm up",
                                   "carry": "up to the cruise height, over the socket, down to the policy start pose (plug 10 mm above the entrance, measured in-hand transform)",
                                   "insert": "the insertion policy until the mean keypoint error is < strict_thresh",
                                   "release": "gripper opens, fingertip held", "return": "8 cm up, back to the home pose, still there for 0.5 s"}},
        "success_criterion": f"mean keypoint error < {args_cli.strict_thresh} m at the last frame, after the release and the return home",
        "cameras": {
            CAMERA_KEYS[0]: {"mount": "env frame, fixed (global view of table, plug, gripper, socket)", "prim_path": base.cfg.front_camera.prim_path,
                             "offset_pos": list(base.cfg.front_camera.offset.pos), "offset_rot_wxyz": list(base.cfg.front_camera.offset.rot),
                             "offset_convention": base.cfg.front_camera.offset.convention, "aim": base.front_camera_aim},
            CAMERA_KEYS[1]: {"mount": "panda_hand", "prim_path": env_cfg.wrist_camera.prim_path,
                             "offset_pos": list(env_cfg.wrist_camera.offset.pos), "offset_rot_wxyz": list(env_cfg.wrist_camera.offset.rot),
                             "offset_convention": env_cfg.wrist_camera.offset.convention},
        },
        "depth_type": "distance_to_image_plane",
        "simulation": {"physics_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation, "frame_stride": env_cfg.frame_stride,
                       "episode_length_s": env_cfg.episode_length_s, "env_spacing_m": env_cfg.scene.env_spacing},
        "software": code_provenance(),
        "assets": {"asset_root": os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_ASSET_ROOT", "")},
    }
    for key, cam in cams.items():
        static["cameras"][key]["camera_model"] = CAMERA_MODELS[key]["model"]
        static["cameras"][key]["intrinsics_source"] = CAMERA_MODELS[key]["source"]
        static["cameras"][key]["intrinsics"] = cam.data.intrinsic_matrices[0].cpu().numpy().tolist()
        static["cameras"][key]["resolution_wh"] = [env_cfg.image_width, env_cfg.image_height]
        static["cameras"][key]["distortion_model"] = "none"
    static.update(static_conventions(env_cfg, base))
    tilt = float(getattr(base, "max_tilt", 0.0))
    static["control"]["orientation"] = (
        {"roll_pitch": f"policy-controlled within {math.degrees(tilt):.1f} deg of upright (per-task unlock)",
         "yaw": "policy-controlled", "rot_action_threshold_rad": [float(v) for v in base.rot_threshold[0].tolist()]}
        if tilt > 0 else
        {"roll_pitch": "locked upright every step (roll = pi, pitch = 0)", "yaw": "policy-controlled",
         "rot_action_threshold_rad": [float(v) for v in base.rot_threshold[0].tolist()]})

    visual = {"lighting": LightingRandomizer, "table_color": TableColorRandomizer,
              "object_distractors": DistractorRandomizer, "texture": TextureRandomizer, "scene": SceneRandomizer,
              "mixed": MixedRandomizer}.get(args_cli.visual_profile, lambda seed: None)(args_cli.seed)
    fps = 1.0 / (env_cfg.sim.dt * env_cfg.frame_stride)
    static["release_and_return"] = {"release_s": base.cfg.release_s, "lift_m": base.cfg.lift_m, "home_still_s": base.cfg.home_still_s,
                                    "home_tolerance_m_rad": [base.cfg.home_tol_pos, base.cfg.home_tol_ang],
                                    "max_insert_s": args_cli.max_insert_s, "max_return_s": args_cli.max_return_s}
    static["grasp_raise_m"] = float(os.environ.get("ROBOASSEMBLYBENCH_AUTOMATE_GRASP_RAISE") or 0.0)
    static["parallel_runs"] = PARALLEL
    static["deeper_ok"] = {"enabled": DEEPER_OK, "out_m": DEEPER_OUT_M, "lateral_m": DEEPER_LAT_M, "tilt_deg": DEEPER_TILT_DEG,
                           "rule": ("success also when, at the last frame, the plug base is at most out_m out of the goal along the socket axis "
                                    "(any depth further in), at most lateral_m off laterally and tilted at most tilt_deg; the target "
                                    "(mean keypoint error < strict_thresh) is still required before the release") if DEEPER_OK else None}
    static["grasp_raise_note"] = ("the nominal grasp (AutoMate plug_grasps.json) moved back by grasp_raise_m along the fingertip approach axis "
                                  "(the fingertips grasp the plug higher), in the pick and in the policy's training reset")
    static["pick_contact_check"] = {"finger_up_contact_force_limit_n": FINGER_TABLE_FORCE_N,
                                    "rule": ("during approach and grasp, the upward (world +z) net contact force on the two fingers, summed, must "
                                             "stay <= the limit: a gripper pushed onto the table to reach the grasp fails the episode "
                                             "(finger_pressed_table, not kept); the episode's maximum is in metrics.pick.finger_up_contact_force_max_n")}
    static["scripted_waypoints"] = {"max_control_steps": 300, "stall_window_control_steps": base.cfg.stall_window,
                                    "descend_landing": ({"land_gain": base.cfg.land_gain,
                                                         "rule": "descent onto the plug: each control step moves at most land_gain x the remaining "
                                                                 "error (no jump to the goal), so the arm slows down as it arrives"}
                                                        if hasattr(base.cfg, "land_gain") else None),
                                    "stall_far_rate": base.cfg.stall_far_rate,
                                    "stall_min_progress_m_rad": list(base.cfg.stall_min_progress), "stall_fail_m_rad": list(base.cfg.stall_fail),
                                    "rule": ("approach / grasp / lift / carry run in lock step over the envs; an env outside the waypoint tolerance is "
                                             "stalled when over the window its error in motion steps (d / max_lin + ang / max_ang) dropped by less "
                                             "than stall_far_rate x window while still moving, or its position and angle errors both dropped by less "
                                             "than stall_min_progress near the waypoint; a stalled env is no longer waited for; stalled farther than "
                                             "stall_fail from the waypoint = the pick fails (servo_stalled, not kept)")}
    collector = EpisodeCollector(base, out_dir, static, target=args_cli.target_successes,
                                 max_insert_frames=int(round(args_cli.max_insert_s * fps)),
                                 max_return_frames=int(round(args_cli.max_return_s * fps)), visual=visual)
    base.frame_callback = collector
    base.visual_hook = visual
    if collector.stop:
        print(f"[COLLECT] {out_dir} already has {collector.successes} >= {args_cli.target_successes} episodes", flush=True)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = base.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args_cli.checkpoint)
    agent.reset()

    t0 = time.time()
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    collector.freeze_failed()
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    while simulation_app.is_running() and collector.round < args_cli.max_rounds and not (collector.stop and collector.round_done):
        with torch.inference_mode():
            if collector.round_done:  # every episode of this round has ended: next round
                collector.refresh()
                if collector.stop:
                    break
                obs = env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                collector.freeze_failed()
                if agent.is_rnn:
                    agent.init_rnn()
                continue
            obs = agent.obs_to_torch(obs)
            _tp = time.perf_counter()
            actions = agent.get_action(obs, is_deterministic=True)
            if os.environ.get("COLLECT_PROFILE"):
                torch.cuda.synchronize()
                from roboassemblybench.library.automate.assembly_collect_env import PROF
                PROF["policy"] += time.perf_counter() - _tp
            actions[base.script > 0] = 0.0  # scripted envs (release / return / frozen): the policy output is not used
            obs, _, dones, _ = env.step(actions)
    collector.close()
    summary = {"assembly_id": args_cli.assembly_id, "visual_profile": args_cli.visual_profile, "successes": collector.successes,
               "attempts": collector.attempts, "rounds": collector.round, "num_envs": base.num_envs,
               "wall_s": round(time.time() - t0, 1), "checkpoint_sha256": static["checkpoint"]["sha256"]}
    (out_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[COLLECT] " + json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    import threading
    import traceback

    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # outputs are complete at this point (episodes finalized, summary written); give Kit 120 s to shut down, then leave
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=120)
    if closer.is_alive():
        print("[COLLECT] Kit shutdown did not finish in 120 s, exiting anyway", flush=True)
    os._exit(code)
