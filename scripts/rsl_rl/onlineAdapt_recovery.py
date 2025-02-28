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

# def main():
#     # Initialize environment
#     env_cfg = parse_env_cfg(
#         args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
#     )
#     agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
#     env_cfg.seed = agent_cfg.seed
#     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
#     log_dir = os.path.dirname(resume_path)

#     env = gym.make(args_cli.task, cfg=env_cfg)
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     env = RslRlVecEnvWrapper(env)
    
#     # Load pre-trained policy
#     ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    
#     ppo_runner.add_git_repo_to_log(__file__)
#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     ppo_runner.load(resume_path)
#     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

#     # Training setup
#     log_dir = os.path.join(log_root_path, args_cli.load_run, "adaptation")
#     os.makedirs(log_dir, exist_ok=True)
#     ppo_runner.log_dir = log_dir

#     # Online Adaptation Loop
#     time_step = 0
#     trigger = 0  # Initially in inference mode
#     obs, _ = env.get_observations()

#     # print(env.unwrapped)
#     robot = env.unwrapped.scene["robot"]
#     # robot = env.env.scene["robot"] 
#     init_pos = [0,0,0,0,0,0,0,0,0,0,0,0]
#     jointpos_p2 = init_pos
#     jointpos_p = init_pos
#     jointpos_n = robot._data.joint_pos[0, :].tolist()


#     while simulation_app.is_running():
#         with torch.inference_mode():
#             ##
#             ## ==================== setting joint locked ====================
#             ##

#             jointpos_p2 = jointpos_p
#             jointpos_p = jointpos_n
#             jointpos_n = robot._data.joint_pos[0, :].tolist()
#             # jointpos_p = jointpos_n
#             # jointpos_p2 = jointpos_p
#             # jointpos_p3 = jointpos_p2
#             # jointpos_p4 = jointpos_p3
#             # jointpos_p5 = jointpos_p4

#             fault_time = 200
#             if time_step == fault_time:
#                 jointpos = robot._data.joint_pos
#                 jointname = robot._data.joint_names

#                 # Modify joint limits after loading the robot
#                 # robot = env.env.scene["robot"]  # Ensure the correct access path
#                 if robot is not None:
#                     trigger = 1
#                     # print("======================= in joint limit ===================")
#                     # Define joint limits as a dictionary with joint names and their (min, max) limits
#                     joint_limits = {
#                         # "LF_HAA": (jointpos[:, 0], jointpos[:, 0]),  # Replace with your joint name and limits (min, max)
#                         # "LF_HFE": (jointpos[:, 4], jointpos[:, 4]),  # Add other joints as needed
#                         # "LF_KFE": (jointpos[:, 8], jointpos[:, 8]),
#                         # "RF_KFE": (jointpos[:, 10], jointpos[:, 10])
#                         "RH_KFE": (jointpos[:, 11], jointpos[:, 11]),
#                     }

#                     # Get the device of the robot data
#                     device = robot._data.joint_limits.device  # Ensure this is the correct device reference

#                     # Loop through the joint limits and apply them
#                     for joint_name, limits in joint_limits.items():
#                     # for i in range(1):
#                         # joint_name = random.choice(jointname)
#                         if joint_name in robot.joint_names:  # Check if the joint exists in the robot
#                             joint_index = robot.joint_names.index(joint_name)  # Find the index of the joint
                            
#                             limits = (jointpos[:, joint_index], jointpos[:, joint_index])
#                             # Prepare limits as a tensor or array
#                             limit_tensor = torch.tensor([limits], dtype=torch.float32, device=device)  # Shape (1, 2) for [min, max]
                            
#                             # Write joint limits to the simulation
#                             robot.write_joint_limits_to_sim(
#                                 limits=limit_tensor,  # Provide the limits tensor
#                                 joint_ids=[joint_index],  # Specify the joint index
#                                 env_ids=None  # Apply to all environments
#                             )
#                             print(f"[INFO] Updated joint limits for {joint_name}: {limits}")
#                         else:
#                             print(f"[WARNING] Joint {joint_name} not found in robot.joint_names.")
#                         # jointname.remove(joint_name)
#                 else:
#                     print("[ERROR] Robot object not found in the environment.")
            
#             print("time: ",time_step)

#             # for i in range(len(jointpos_n)):
#             #     if round(jointpos_n[i], 3) == round(jointpos_p[i], 3) == round(jointpos_p2[i], 3):
#             #     # if jointpos_n[i] == jointpos_p3[i] == jointpos_p5[i]:
#             #         trigger = 1
#             #         print(f"time: {time_step}, joint: {robot._data.joint_names[i]}, index joint: {i} is locked")  # Return the matching value

#             time_step += 1
#             if trigger == 0:
#                 with torch.inference_mode():
#                     actions = policy(obs)
#                     # obs, _, _, _ = env.step(actions)

#             # else:
#             #     # Switch to training mode
#             #     actions = ppo_runner.alg.actor_critic.act(obs)
#             #     ppo_runner.learn(num_learning_iterations=500, init_at_random_ep_len=False)
#             #     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
#             #     ppo_runner.save(os.path.join(log_dir, "latest_policy.pth"))  # Save the newest policy
#             #     ppo_runner.load(os.path.join(log_dir, "latest_policy.pth"))  # Load the newest policy
            

#             else:
#                 # Switch to training mode
#                 actions = ppo_runner.alg.actor_critic.act(obs)
#                 ppo_runner.learn(num_learning_iterations=5000, init_at_random_ep_len=False)
#                 policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
            
#             obs, _, _, _ = env.step(actions)

#             # obs, _, _, _ = env.step(actions)

#     # Close simulation
#     env.close()

# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()

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
        with torch.inference_mode():
            jointpos_p2 = jointpos_p
            jointpos_p = jointpos_n
            jointpos_n = robot._data.joint_pos[0, :].tolist()

            fault_time = 200
            if time_step == fault_time:
                trigger = 1
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

            print("time:", time_step)

            time_step += 1

            if trigger == 0:
                with torch.inference_mode():
                    actions = policy(obs)
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

            if trigger == 1:
                # Switch to training mode
                actions = ppo_runner.alg.actor_critic.act(obs)

                # Ensure gradients are enabled
                for param in ppo_runner.alg.actor_critic.parameters():
                    param.requires_grad = True  

                # Train the policy
                ppo_runner.learn(num_learning_iterations=5000, init_at_random_ep_len=False)

                # Get the updated policy
                policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)


if __name__ == "__main__":
    main()
    simulation_app.close()
