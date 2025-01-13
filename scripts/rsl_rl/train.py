# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import random

from omni.isaac.lab.app import AppLauncher

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

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import extensions.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    ##
    ## ==================== setting joint locked ====================
    ##
    robot = env.env.scene["robot"] 

    # jointpos = robot._data.joint_pos
    # jointname = robot._data.joint_names

    # # Modify joint limits after loading the robot
    # robot = env.env.scene["robot"]  # Ensure the correct access path
    # if robot is not None:
    #     # Get the device of the robot data
    #     device = robot._data.joint_limits.device  # Ensure this is the correct device reference

    #     # Randomly select and lock one joint for each agent
    #     num_agents = 4096  # Assuming _num_envs gives the number of agents
    #     joint_names = robot.joint_names
    #     num_joints = len(joint_names)

    #     # Generate random joint indices for locking
    #     locked_joint_indices = torch.randint(0, num_joints, (num_agents,))

    #     for agent_id in range(num_agents):
    #         joint_index = locked_joint_indices[agent_id].item()  # Get the joint index for this agent
    #         joint_name = joint_names[joint_index]  # Get the joint name for this index

    #         # Define the lock position using the joint position of this agent
    #         lock_position = jointpos[agent_id, joint_index]  # Retrieve the agent-specific joint position
    #         limits = (lock_position, lock_position)  # Min and max are the same for locking

    #         # Prepare limits as a tensor or array
    #         limit_tensor = torch.tensor([limits], dtype=torch.float32, device=device)  # Shape (1, 2) for [min, max]

    #         # Convert env_ids to a tensor
    #         env_id_tensor = torch.tensor([agent_id], dtype=torch.int32, device=device)

    #         # Write joint limits to the simulation for the specific joint of this agent
    #         robot.write_joint_limits_to_sim(
    #             limits=limit_tensor,  # Provide the limits tensor
    #             joint_ids=[joint_index],  # Specify the joint index
    #             env_ids=env_id_tensor  # Apply to the specific agent
    #         )

    #         print(f"[INFO] Locked joint {joint_name} (index {joint_index}) for agent {agent_id} at position {lock_position}.")

    # Access joint positions and joint names
    jointpos = robot._data.joint_pos
    jointname = robot._data.joint_names

    # Modify joint limits after loading the robot
    robot = env.env.scene["robot"]  # Ensure the correct access path
    if robot is not None:
        # Get the device of the robot data
        device = robot._data.joint_limits.device  # Ensure this is the correct device reference

        # Randomly select and lock one joint for each agent
        num_agents = 8192  # Assuming _num_envs gives the number of agents
        joint_names = robot.joint_names
        num_joints = len(joint_names)

        # Generate random joint indices for locking (0 to num_joints+1)
        locked_joint_indices = torch.randint(0, num_joints + 3, (num_agents,))

        for agent_id in range(num_agents):
            joint_index = locked_joint_indices[agent_id].item()  # Get the joint index for this agent

            if joint_index < num_joints:  # Only lock joints if the index is valid
                joint_name = joint_names[joint_index]  # Get the joint name for this index

                # Define the lock position using the joint position of this agent
                lock_position = jointpos[agent_id, joint_index]  # Retrieve the agent-specific joint position
                limits = (lock_position, lock_position)  # Min and max are the same for locking

                # Prepare limits as a tensor or array
                limit_tensor = torch.tensor([limits], dtype=torch.float32, device=device)  # Shape (1, 2) for [min, max]

                # Convert env_ids to a tensor
                env_id_tensor = torch.tensor([agent_id], dtype=torch.int32, device=device)

                # Write joint limits to the simulation for the specific joint of this agent
                robot.write_joint_limits_to_sim(
                    limits=limit_tensor,  # Provide the limits tensor
                    joint_ids=[joint_index],  # Specify the joint index
                    env_ids=env_id_tensor  # Apply to the specific agent
                )

                print(f"[INFO] Locked joint {joint_name} (index {joint_index}) for agent {agent_id} at position {lock_position}.")
            else:
                # Log or skip the agent if the random index is out of range
                print(f"[INFO] Skipping agent {agent_id}, random joint index {joint_index} is out of range.")



    # Modify joint limits after loading the robot
    # robot = env.env.scene["robot"]  # Ensure the correct access path
    # if robot is not None:
    #     # print("======================= in joint limit ===================")
    #     # Define joint limits as a dictionary with joint names and their (min, max) limits
    #     joint_limits = {
    #         "LF_HAA": (jointpos[:, 0], jointpos[:, 0]),  # Replace with your joint name and limits (min, max)
    #         "LF_HFE": (jointpos[:, 4], jointpos[:, 4]),  # Add other joints as needed
    #         # "LF_KFE": (jointpos[:, 8], jointpos[:, 8]),
    #     }

    #     # Get the device of the robot data
    #     device = robot._data.joint_limits.device  # Ensure this is the correct device reference

    #     # Loop through the joint limits and apply them
    #     # for joint_name, limits in joint_limits.items():
    #     for i in range(1):
    #         joint_name = random.choice(jointname)
    #         if joint_name in robot.joint_names:  # Check if the joint exists in the robot
    #             joint_index = robot.joint_names.index(joint_name)  # Find the index of the joint
                
    #             limits = (jointpos[:, joint_index], jointpos[:, joint_index])
    #             # Prepare limits as a tensor or array
    #             limit_tensor = torch.tensor([limits], dtype=torch.float32, device=device)  # Shape (1, 2) for [min, max]
                
    #             # Write joint limits to the simulation
    #             robot.write_joint_limits_to_sim(
    #                 limits=limit_tensor,  # Provide the limits tensor
    #                 joint_ids=[joint_index],  # Specify the joint index
    #                 env_ids=None  # Apply to all environments
    #             )
    #             print(f"[INFO] Updated joint limits for {joint_name}: {limits}")
    #         else:
    #             print(f"[WARNING] Joint {joint_name} not found in robot.joint_names.")
            # jointname.remove(joint_name)
        # else:
        #     print("[ERROR] Robot object not found in the environment.")

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
