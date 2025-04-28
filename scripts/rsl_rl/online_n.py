# scripts/rsl_rl/online.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from datetime import datetime
import os
import csv

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

import extensions.tasks  # noqa: F401

# -----------------------------------------------------------------------------
# New wrapper to print rewards during training
# -----------------------------------------------------------------------------
class RewardPrintWrapper(RslRlVecEnvWrapper):
    """Wraps RslRlVecEnvWrapper to print per-term rewards every step in training."""
    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)
        for name, term in extras.get("log", {}).items():
            if isinstance(term, torch.Tensor):
                v = term.mean().item()
            else:
                try:
                    v = float(term)
                except Exception:
                    v = term
            print(f"[TRAIN STEP] {name:<30s}: {v:.6f}" if isinstance(v, (float, int))
                  else f"[TRAIN STEP] {name:<30s}: {v}")
        return obs, rew, dones, extras


# Hydra / AppLauncher setup
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--log", action="store_true", default=False, help="Record data of the robot.")
parser.add_argument("--log_path", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Record data of the robot.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Prevent TF32 nondeterminism
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # 1) Environment & Agent config
    env_cfg = cli_args.parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # 2) Logging setup
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    os.makedirs(log_dir, exist_ok=True)

    # 3) Create & wrap env
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # standard RSL-RL wrapper
    env = RslRlVecEnvWrapper(env)
    # wrap again to print rewards during training
    env = RewardPrintWrapper(env)

    # 4) Load policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 5) Dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # 6) Get initial observation
    obs, _ = env.reset()

    # 7) Online‐adaptation loop
    time_step = 0
    trigger = 0  # 0=inference, 1=adaptation

    while simulation_app.is_running():
        # (your joint-lock logic unchanged)
        print(f"time: {time_step}, trigger: {trigger}")

        if trigger == 0:
            # inference phase
            with torch.inference_mode():
                actions = policy(obs)
                obs, reward, dones, extras = env.step(actions)

            # handle resets if any sub‐env terminated
            if dones.any():
                obs, _ = env.reset_done(dones)

        else:
            # adaptation phase
            print("Starting adaptation...")
            ppo_runner.learn_iter1(num_learning_iterations=5000, init_at_random_ep_len=False)
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        time_step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()