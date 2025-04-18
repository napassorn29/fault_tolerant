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
from datetime import datetime

# add argparse arguments
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

import csv

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

class DataLogger:
    def __init__(self, log_path):
        self.data = []
        
        # Initialize CSV file for logging contact forces
        self.csv_file = open(log_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Time", "LF_FOOT_X", "LF_FOOT_Y", "LF_FOOT_Z", 
            "LH_FOOT_X", "LH_FOOT_Y", "LH_FOOT_Z",
            "RF_FOOT_X", "RF_FOOT_Y", "RF_FOOT_Z",
            "RH_FOOT_X", "RH_FOOT_Y", "RH_FOOT_Z", 
            "Base_lin_vel_X", "Base_lin_vel_Y", "Base_lin_vel_Z", 
            "joint0_pos", "joint1_pos", "joint2_pos", 
            "joint3_pos", "joint4_pos", "joint5_pos",
            "joint6_pos", "joint7_pos", "joint8_pos",
            "joint9_pos", "joint10_pos", "joint11_pos",
            "base_height",
            "obs"
        ])
        # self.time_step = 0

    def log_data(self, time, contact_forces, base_lin_vel, joint_pos, base_height, observation):
        # Log each time step's data to CSV
        # row = [self.time_step] + contact_forces[1] + contact_forces[2] + contact_forces[3] + contact_forces[4] + base_lin_vel[0] + joint_pos[0] + observation
        row = [time] + contact_forces[1] + contact_forces[2] + contact_forces[3] + contact_forces[4] + base_lin_vel[0] + joint_pos[0] + observation
        self.csv_writer.writerow(row)
        # self.time_step += 1

    def close_logger(self):
        self.csv_file.close()


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
    log_dir = os.path.join(log_root_path, args_cli.load_run, "adaptation5")
    os.makedirs(log_dir, exist_ok=True)
    ppo_runner.log_dir = log_dir

    # Online Adaptation Loop
    time_step = 0
    trigger = 0  # Initially in inference mode
    trigger1 = 0
    trigger2 = 0
    obs, _ = env.get_observations()

    # Fix: Ensure correct access to the robot object
    robot = env.unwrapped.scene["robot"]
    
    if robot is None:
        print("[ERROR] Robot object not found in the environment.")
        return

    init_pos = [0] * 12
    jointpos_p2 = init_pos
    jointpos_p = init_pos
    jointpos_n = robot._data.joint_pos[0, :].tolist()

    name_joint = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    # name_joint = ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
    
    leg_damage = 'RL_calf_joint'
    # leg_damage = 'FL_hip_joint'
    index_leg_damage = name_joint.index(leg_damage)
    # num_shift = 0 
    # num_shift = 0.2618 # 15 degree
    num_shift = -0.17453 # 10 degree
    # num_shift = -0.349 # 20 degree

    if args_cli.log:
        data_log_path = os.path.join(log_root_path, agent_cfg.load_run, "adaptation3", args_cli.log_path + leg_damage + "_" + str(index_leg_damage) + "_" + str(num_shift) + "_data.csv")
        sensor_node = DataLogger(data_log_path)


    while simulation_app.is_running():
        with torch.inference_mode():
            jointpos_p2 = jointpos_p
            jointpos_p = jointpos_n
            jointpos_n = robot._data.joint_pos[0, :].tolist()

            fault_time = 500
            trigger2 = trigger1
            if time_step == fault_time:
                # trigger = 1
                print(robot._data.joint_names)
                jointpos = robot._data.joint_pos
                joint_limits = {
                    leg_damage: (jointpos[:, index_leg_damage]+num_shift, jointpos[:, index_leg_damage]+num_shift),  # Example joint lock
                    # "RL_calf_joint": (jointpos[:, 10], jointpos[:, 10]),  # Example joint lock
                    # "RH_KFE": (jointpos[:, 11], jointpos[:, 11]),  # Example joint lock
                    # "LF_HFE": (jointpos[:, 4], jointpos[:, 4])
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

            print("time:", time_step, "trigger:", trigger, "height:", robot.data.root_link_pos_w[:, 2], "joint", robot._data.joint_pos[:, 11])

            time_step += 1

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


            # if trigger == 0:
            #     with torch.inference_mode():
            #         actions = policy(obs)
            #         obs, _, _, _ = env.step(actions)

            # else:
            #     # ppo_runner.learn_iteration2(num_learning_iterations=5000, init_at_random_ep_len=False, robot=robot, env=env, log_path=data_log_path)
            #     ppo_runner.learn_iteration1(num_learning_iterations=5000, init_at_random_ep_len=False)
            #     policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

            # print(env.env.scene["contact_forces"].body_names)
            LF_foot_index = env.env.scene["contact_forces"].body_names.index("FL_foot")
            LH_foot_index = env.env.scene["contact_forces"].body_names.index("FR_foot")
            RF_foot_index = env.env.scene["contact_forces"].body_names.index("RL_foot")
            RH_foot_index = env.env.scene["contact_forces"].body_names.index("RR_foot")

            # LF_foot_index = env.env.scene["contact_forces"].body_names.index("LF_FOOT")
            # LH_foot_index = env.env.scene["contact_forces"].body_names.index("LH_FOOT")
            # RF_foot_index = env.env.scene["contact_forces"].body_names.index("RF_FOOT")
            # RH_foot_index = env.env.scene["contact_forces"].body_names.index("RH_FOOT")

            # Extract contact forces
            contact_forces = [
                env.env.scene["contact_forces"].data.net_forces_w[0, 0].cpu().tolist(),  # base
                env.env.scene["contact_forces"].data.net_forces_w[0, LF_foot_index].cpu().tolist(),  # LF_foot
                env.env.scene["contact_forces"].data.net_forces_w[0, LH_foot_index].cpu().tolist(),  # LH_foot
                env.env.scene["contact_forces"].data.net_forces_w[0, RF_foot_index].cpu().tolist(),  # RF_foot
                env.env.scene["contact_forces"].data.net_forces_w[0, RH_foot_index].cpu().tolist(),  # RH_foot
            ]
            # Publish data and export log data
            base_lin_vel = [obs[0, :3].tolist()]
            # joint_pos = [obs[0, 12:24].tolist()]
            joint_pos = [robot._data.joint_pos[0, :].tolist()]
            base_height = robot.data.root_link_pos_w[:, 2]
            observation = [obs[0, :].tolist()]

            # camera follow
            env_ids = torch.tensor([0], device=env.unwrapped.device)
            env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
            lookat = [robot.data.root_pos_w[env_ids.item(), i].cpu().item() for i in range(3)]
            # eye_offset = [3, 4, 0.75] # [2, 0, 1] [2, -1.5, 1] [2, 2, 1]
            eye_offset = [1, 4, 0.5] # [2, 0, 1] [2, -1.5, 1] [2, 2, 1]
            pairs = zip(lookat, eye_offset)
            eye = [x + y for x, y in pairs]
            env.unwrapped.viewport_camera_controller.update_view_location(eye, lookat)

            if args_cli.log:
                sensor_node.log_data(time_step, contact_forces, base_lin_vel, joint_pos, base_height, observation)
        

if __name__ == "__main__":
    main()
    simulation_app.close()