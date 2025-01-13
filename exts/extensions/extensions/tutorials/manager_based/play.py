"""
    This script use for interacting with an articulation for quadruped robot in Isaac Sim.

    #   Usage
    python exts/extensions/extensions/tutorials/manager_based/play.py --num_envs 1 --checkpoint logs/sb3/AnymalDFlatEnv/2024-11-05_19-19-41/model_1004000_steps.zip
"""

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

##############################################################
#
#           Setting argparse arguments
#
##############################################################

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

##############################################################
#
#           Launch Isaac Sim Simulator first.
#
##############################################################

from omni.isaac.lab.app import AppLauncher

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##############################################################
#
#           Import the library for run play
#
##############################################################

import gymnasium as gym
import numpy as np
import os
import torch

# Import the AnymalDFlatEnvCfg from your custom task file
from flat_env_cfg import AnymalDFlatEnvCfg
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

##############################################################
#
#           Main Function
#
##############################################################


def main():
    """Play with a stable-baselines agent."""
    # Configure environment and agent
    env_cfg = AnymalDFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # agent_cfg = {
    #     "gamma": 0.99,
    #     "learning_rate": 3e-4,
    #     "policy": "MlpPolicy",
    #     "normalize_input": True,
    #     "clip_obs": 10.0,
    # }

    agent_cfg = {
        "seed": 42,
        "n_timesteps": 1000000.0,
        "policy": "MlpPolicy",
        "n_steps": 16,
        "batch_size": 4096,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "n_epochs": 20,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "clip_range": 0.2,
        "policy_kwargs": "dict( activation_fn=nn.ELU, net_arch=[32, 32], squash_output=False, )",
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "normalize_input": True,
        "clip_obs": 10.0,
        "device": "cuda:0",
    }

    # Define logging paths and checkpoint paths
    log_root_path = os.path.join("logs", "sb3", "AnymalDFlatEnv")
    log_root_path = os.path.abspath(log_root_path)
    # Check if a checkpoint path is provided
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # Process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    # Create the Isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)  # Wrap for SB3 compatibility

    # Add video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=agent_cfg.pop("normalize_input", False),
            norm_reward=agent_cfg.pop("normalize_value", False),
            clip_obs=agent_cfg.pop("clip_obs", 10.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # Load the agent directly with PPO.load()
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Loading the checkpoint without using torch.load
    agent = PPO.load(checkpoint_path, env=env)

    print(f"Reset environment and start simulation loop")
    # Reset environment and start simulation loop
    obs = env.reset()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # Close environment and simulation app
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
