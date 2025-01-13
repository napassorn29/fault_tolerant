"""
    This script use for interacting with an articulation for quadruped robot in Isaac Sim.

    #   Usage
    python exts/extensions/extensions/tutorials/manager_based/train.py --headless
"""

"""Script to train RL agent with Stable Baselines3 using AnymalDFlatEnvCfg."""

##############################################################
#
#           Setting argparse arguments
#
##############################################################

import argparse

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

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
import random
from datetime import datetime

# Import the AnymalDFlatEnvCfg from your custom task file
from flat_env_cfg import AnymalDFlatEnvCfg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

##############################################################
#
#           Main Function
#
##############################################################


def main():
    """Train with stable-baselines agent using AnymalDFlatEnvCfg."""
    # Set up environment configuration
    env_cfg = AnymalDFlatEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Set random seed
    env_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed

    # Define training steps
    n_timesteps = args_cli.max_iterations * env_cfg.scene.num_envs if args_cli.max_iterations else 1000000

    # Logging directory
    log_dir = os.path.join("logs", "sb3", "AnymalDFlatEnv", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    # Create the environment instance
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap the environment for Stable Baselines
    env = Sb3VecEnvWrapper(env)

    # Wrap for video recording if enabled
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Normalize observations and rewards
    env = VecNormalize(env, training=True, gamma=0.99, clip_reward=np.inf)

    # Configure the PPO agent
    agent = PPO("MlpPolicy", env, verbose=1)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # Set up callback for checkpointing
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)

    # Train the agent
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    agent.save(os.path.join(log_dir, "model"))

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
