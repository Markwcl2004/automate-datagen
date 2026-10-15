"""Roll out a trained AutoMate specialist and log per-step end-effector paths.

Adapted from Isaac Lab v2.3.2 ``scripts/reinforcement_learning/rl_games/play.py``.
Upstream evaluation logging only stores the final plug/socket pose per episode
(and calls ``exit(0)`` once enough trials are collected), so it cannot be used
to compare rollout paths against the reference disassembly trajectories.

Logged frame: ``fingertip_midpoint_pos - gripper_goal_pos``, i.e. the same
goal-relative frame used by the imitation reward (assembly_env.py L602) and by
``_load_disassembly_data`` (reference trajectories minus their index-0 point).

Run from the IsaacLab directory:
    ./isaaclab.sh -p $REPO_ROOT/roboassemblybench/scripts/training/rollout_logger.py \
        --assembly-id 00015 --checkpoint <ckpt.pth> --out <file.npz> \
        --num_envs 16 --episodes 32 --headless --device cuda:1
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assembly-id", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=32, help="Number of completed episodes to record.")
parser.add_argument("--seed", type=int, default=10000)
parser.add_argument("--keep-sbc", action="store_true", help="Keep the training curriculum (default: off, full randomization).")
parser.add_argument("--strict-thresh", type=float, default=0.003, help="Strict success: mean keypoint error (m) at episode end.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Must be set before the task configs are imported (assembly_tasks_cfg.py L13).
os.environ["ROBOASSEMBLYBENCH_AUTOMATE_ASSEMBLY_ID"] = str(args_cli.assembly_id)
TASK = "RoboAssemblyBench-AutoMate-Assembly-Direct-v0"
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import roboassemblybench.library.automate  # noqa: E402,F401  (registers TASK)


@hydra_task_config(TASK, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed
    if not args_cli.keep_sbc:
        # Same as upstream eval (assembly_env.py _init_eval_logging): plugs start outside the socket.
        env_cfg.tasks[env_cfg.task_name].if_sbc = False

    # rl_games: the trainer reads config.device, the player reads config.device_name
    # (run_assembly.py adds +agent.params.config.device_name for the same reason).
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

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    n = base.num_envs
    buffers = [[] for _ in range(n)]
    # Strict insertion check (see the AutoMate success latch): upstream success latches if, at ANY step, the plug is less than
    # the full insertion depth above the socket and the mean keypoint error is < 1.5 cm -- i.e. touching the entrance
    # counts. Also log the per-step mean keypoint distance so the final / best insertion error can be judged strictly.
    kp_buffers = [[] for _ in range(n)]
    episodes = []  # dicts: pos (T,3), success, env_id
    while simulation_app.is_running() and len(episodes) < args_cli.episodes:
        with torch.inference_mode():
            rel = (base.fingertip_midpoint_pos - base.gripper_goal_pos).detach().cpu().numpy()
            kp = torch.norm(base.keypoints_held - base.keypoints_fixed, p=2, dim=-1).mean(-1).detach().cpu().numpy()
            for i in range(n):
                buffers[i].append(rel[i].copy())
                kp_buffers[i].append(float(kp[i]))
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            done_ids = torch.nonzero(dones).flatten().tolist()
            if done_ids:
                succ = base.ep_succeeded.detach().cpu().numpy().astype(bool)
                for i in done_ids:
                    kpd = np.asarray(kp_buffers[i], dtype=np.float32)
                    episodes.append({"pos": np.asarray(buffers[i], dtype=np.float32), "success": bool(succ[i]), "env_id": i,
                                     "kp_final": float(kpd[-1]), "kp_min": float(kpd.min())})
                    buffers[i] = []
                    kp_buffers[i] = []
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

    episodes = episodes[: args_cli.episodes]
    lengths = np.array([len(e["pos"]) for e in episodes], dtype=np.int32)
    flat = np.concatenate([e["pos"] for e in episodes], axis=0) if episodes else np.zeros((0, 3), np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez(
        args_cli.out,
        assembly_id=str(args_cli.assembly_id),
        checkpoint=str(args_cli.checkpoint),
        seed=args_cli.seed,
        if_sbc=bool(args_cli.keep_sbc),
        frame="fingertip_midpoint_pos - gripper_goal_pos (goal-relative, metres)",
        lengths=lengths,
        positions=flat,
        success=np.array([e["success"] for e in episodes], dtype=bool),
        kp_final=np.array([e["kp_final"] for e in episodes], dtype=np.float32),
        kp_min=np.array([e["kp_min"] for e in episodes], dtype=np.float32),
        strict_thresh=args_cli.strict_thresh,
    )
    rate = float(np.mean([e["success"] for e in episodes])) if episodes else float("nan")
    kf = np.array([e["kp_final"] for e in episodes]); km = np.array([e["kp_min"] for e in episodes])
    st = args_cli.strict_thresh
    print(f"[ROLLOUT] assembly={args_cli.assembly_id} episodes={len(episodes)} success_rate={rate:.3f} "
          f"strict_final(<{st*1000:.0f}mm)={np.mean(kf < st):.3f} strict_any={np.mean(km < st):.3f} "
          f"kp_final_median={np.median(kf)*1000:.1f}mm kp_min_median={np.median(km)*1000:.1f}mm -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
