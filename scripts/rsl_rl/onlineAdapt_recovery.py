# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

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

# Import extensions to set up environment tasks
import extensions.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main():
    # Initialize environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)
    
    # Load pre-trained policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    ppo_runner.add_git_repo_to_log(__file__)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # Fix RNN Warning
    try:
        ppo_runner.alg.actor_critic.flatten_parameters()
    except AttributeError:
        pass  # Ignore if not an RNN model

    # Training setup
    log_dir = os.path.join(log_root_path, args_cli.load_run, "adaptation")
    os.makedirs(log_dir, exist_ok=True)
    ppo_runner.log_dir = log_dir

    # Online Adaptation Loop
    time_step = 0
    trigger = 0  # Initially in inference mode
    trigger1 = 0
    trigger2 = 0
    obs, _ = env.get_observations()

    # Fix: Ensure correct access to the robot object
    robot = robot = env.unwrapped.scene["robot"]
    
    if robot is None:
        print("[ERROR] Robot object not found in the environment.")
        return

    init_pos = [0] * 12
    jointpos_p2 = init_pos
    jointpos_p = init_pos
    jointpos_n = robot._data.joint_pos[0, :].tolist()

    while simulation_app.is_running():
        jointpos_p2 = jointpos_p
        jointpos_p = jointpos_n
        jointpos_n = robot._data.joint_pos[0, :].tolist()

        fault_time = 200
        trigger2 = trigger1
        if time_step == fault_time:
            # trigger = 1
            jointpos = robot._data.joint_pos
            joint_limits = {
                "RH_KFE": (jointpos[:, 11], jointpos[:, 11]),  # Example joint lock
            }

            device = robot._data.joint_limits.device
            for joint_name, limits in joint_limits.items():
                if joint_name in robot.joint_names:
                    joint_index = robot.joint_names.index(joint_name)
                    limit_tensor = torch.tensor([limits], dtype=torch.float32, device=device)
                    robot.write_joint_limits_to_sim(
                        limits=limit_tensor, joint_ids=[joint_index], env_ids=None
                    )
                    print(f"[INFO] Updated joint limits for {joint_name}: {limits}")

        for i in range(len(jointpos_n)):
            if round(jointpos_n[i], 4) == round(jointpos_p[i], 4) == round(jointpos_p2[i], 4) and time_step > 100:
                trigger = 1
                # if jointpos_n[i] == jointpos_p3[i] == jointpos_p5[i]:
                print(f"time: {time_step}, joint: {robot._data.joint_names[i]}, index joint: {i} is locked")  # Return the matching value


        # if trigger1 == trigger2: trigger = 1

        print("time:", time_step, "trigger:", trigger)

        time_step += 1

        if trigger == 0:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
        # if trigger == 1:
        #     # Switch to training mode
        #     actions = ppo_runner.alg.actor_critic.act(obs)

        #     # Ensure gradients are enabled
        #     for param in ppo_runner.alg.actor_critic.parameters():
        #         param.requires_grad = True  

        #     # Train the policy
        #     ppo_runner.learn(num_learning_iterations=5000, init_at_random_ep_len=False)

        #     # Get the updated policy
        #     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        else:
            # ppo_runner.alg.actor_critic.train()  # Ensure training mode is enabled
            # for param in ppo_runner.alg.actor_critic.parameters():
            #     param.requires_grad = True  
            # obs = obs.clone().detach().requires_grad_(True)  # Ensure obs has grads
            # actions = ppo_runner.alg.actor_critic.act(obs)
            ppo_runner.learn(num_learning_iterations=5000, init_at_random_ep_len=False)
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        # else:
        #     # Switch to training mode
        #     actions = ppo_runner.alg.actor_critic.act(obs)
            
        #     # Fix: Ensure loss has gradients before backward pass
        #     for param in ppo_runner.alg.actor_critic.parameters():
        #         param.requires_grad = True  # Ensure all parameters require gradients
            
        #     loss = ppo_runner.learn(num_learning_iterations=5000, init_at_random_ep_len=False)
        #     if loss is not None and not loss.requires_grad:
        #         loss = loss.clone().detach().requires_grad_(True)

        #     loss.backward()  # Compute gradients
        #     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
