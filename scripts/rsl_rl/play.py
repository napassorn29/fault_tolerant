"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher
# from omni.isaac.lab.assets import ArticulationCfg

from datetime import datetime
import random

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--pos", type=list, default=None, help="Config position.")
# parser.add_argument("--vel", type=list, default=None, help="Config velocity.")
parser.add_argument("--pos", type=lambda s: [float(item) for item in s.split(',')], default=None, help="Config position (format: x,y).")
parser.add_argument("--vel", type=lambda s: [float(item) for item in s.split(',')], default=None, help="Config velocity (format: x,y).")
parser.add_argument("--log", action="store_true", default=False, help="Record data of the robot.")
parser.add_argument("--log_path", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Record data of the robot.")
parser.add_argument("--fault", action="store_true", default=False, help="fault tolerant or not.")


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import math
import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp

# import rclpy
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
import csv

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import extensions.tasks  # noqa: F401


class PubSubData(Node):
    def __init__(self, log_path):
        super().__init__("run_quadruped_env")
        timer = 10
        self.pub_contact_base = self.create_publisher(Vector3, "/isaac_contact/base", timer)
        self.pub_contact_LF = self.create_publisher(Vector3, "/isaac_contact/LF_foot", timer)
        self.pub_contact_LH = self.create_publisher(Vector3, "/isaac_contact/LH_foot", timer)
        self.pub_contact_RF = self.create_publisher(Vector3, "/isaac_contact/RF_foot", timer)
        self.pub_contact_RH = self.create_publisher(Vector3, "/isaac_contact/RH_foot", timer)
        self.pub_base_lin_vel = self.create_publisher(Vector3, "/isaac_velocity/Base_lin", timer)

        self.data = []
        
        # Initialize CSV file for logging contact forces
        self.csv_file = open(log_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Time", "LF_FOOT_X", "LF_FOOT_Y", "LF_FOOT_Z", 
                                  "LH_FOOT_X", "LH_FOOT_Y", "LH_FOOT_Z",
                                  "RF_FOOT_X", "RF_FOOT_Y", "RF_FOOT_Z",
                                  "RH_FOOT_X", "RH_FOOT_Y", "RH_FOOT_Z", 
                                  "Base_lin_vel_X", "Base_lin_vel_Y", "Base_lin_vel_Z", 
                                  "joint0_pos", "joint1_pos", "joint2_pos", 
                                  "joint3_pos", "joint4_pos", "joint5_pos",
                                  "joint6_pos", "joint7_pos", "joint8_pos",
                                  "joint9_pos", "joint10_pos", "joint11_pos",
                                  "obs"])
        self.time_step = 0

    def publish_sensor_data(self, contact_forces):
        # Publish each contact force
        for name, data, pub in zip(
            ["base", "LF_foot", "LH_foot", "RF_foot", "RH_foot"],
            contact_forces,
            [
                self.pub_contact_base,
                self.pub_contact_LF,
                self.pub_contact_LH,
                self.pub_contact_RF,
                self.pub_contact_RH,
            ],
        ):
            msg = Vector3()
            msg.x, msg.y, msg.z = data
            pub.publish(msg)
            # self.get_logger().info(f"Published {name} contact force: {data}")
    
    def publish_velocity_data(self, base_lin_vel):
        for name, data, pub in zip(
            ["Base_lin"],
            base_lin_vel,
            [
                self.pub_base_lin_vel,
            ],
        ):
            msg = Vector3()
            msg.x, msg.y, msg.z = data
            pub.publish(msg)
            # self.get_logger().info(f"Published {name} velocity: {data}")

    def log_data(self, contact_forces, base_lin_vel, joint_pos, observation):
        # Log each time step's data to CSV
        row = [self.time_step] + contact_forces[1] + contact_forces[2] + contact_forces[3] + contact_forces[4] + base_lin_vel[0] + joint_pos[0] + observation
        self.csv_writer.writerow(row)
        self.time_step += 1

    def close_logger(self):
        self.csv_file.close()


def main(args=None):
    """Play with RSL-RL agent."""
    rclpy.init(args=args)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.pos is not None:
        env_cfg.commands.pose_command = mdp.UniformPose2dCommandCfg(
            asset_name="robot",
            simple_heading=True,
            resampling_time_range=(8.0, 8.0),
            debug_vis=True,
            ranges=mdp.UniformPose2dCommandCfg.Ranges(
                pos_x=(float(args_cli.pos[0]), float(args_cli.pos[0])),  # Set fixed x position
                pos_y=(float(args_cli.pos[1]), float(args_cli.pos[1])),  # Set fixed y position
                # heading=(math.pi/4, math.pi/4)
                heading=(-math.pi, math.pi)
            ),
        )
        # env_cfg.commands.pose_command = mdp.UniformPoseCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(8.0, 8.0),
        #     debug_vis=True,
        #     ranges=mdp.UniformPoseCommandCfg.Ranges(
        #         pos_x=(args_cli.pos[0], args_cli.pos[0]),
        #         pos_y=(args_cli.pos[1], args_cli.pos[1]),
        #         pos_z=(0.5, 0.5),  # Fixed height if working on a plane
        #         roll=(0.0, 0.0),   # No roll
        #         pitch=(0.0, 0.0),  # No pitch
        #         yaw=(0.0, 0.0)     # No heading rotation
        #     ),
        # )

    if args_cli.vel is not None:
        env_cfg.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02, 
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(float(args_cli.vel[0]), float(args_cli.vel[0])), # Set fixed x velocity
                lin_vel_y=(float(args_cli.vel[1]), float(args_cli.vel[1])), # Set fixed y velocity
                ang_vel_z=(-1.0, 1.0), # Set fixed z velocity
                heading=(-math.pi, math.pi)
            ),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    ##
    ## ==================== video recording ===================
    ##

    # wrap for video recording
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.fault:
        fault_tol = True
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # initialize ROS node
    # sensor_node = SensorRunner()
    # initialize ROS node and SensorRunner with the path for the CSV log
    data_log_path = os.path.join(log_root_path, agent_cfg.load_run, args_cli.log_path + "_data.csv")
    sensor_node = PubSubData(data_log_path)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    robot = env.env.scene["robot"] 
    print("init_state: ",robot.cfg.init_state.rot)


    init_pos = [0,0,0,0,0,0,0,0,0,0,0,0]
    jointpos_p2 = init_pos
    jointpos_p = init_pos
    jointpos_n = robot._data.joint_pos[0, :].tolist()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            ##
            ## ==================== setting joint locked ====================
            ##

            jointpos_p2 = jointpos_p
            jointpos_p = jointpos_n
            jointpos_n = robot._data.joint_pos[0, :].tolist()
            # jointpos_p = jointpos_n
            # jointpos_p2 = jointpos_p
            # jointpos_p3 = jointpos_p2
            # jointpos_p4 = jointpos_p3
            # jointpos_p5 = jointpos_p4

            fault_time = 250
            if sensor_node.time_step == fault_time:
                jointpos = robot._data.joint_pos
                jointname = robot._data.joint_names

                # Modify joint limits after loading the robot
                robot = env.env.scene["robot"]  # Ensure the correct access path
                if robot is not None:
                    # print("======================= in joint limit ===================")
                    # Define joint limits as a dictionary with joint names and their (min, max) limits
                    joint_limits = {
                        # "LF_HAA": (jointpos[:, 0], jointpos[:, 0]),  # Replace with your joint name and limits (min, max)
                        # "LF_HFE": (jointpos[:, 4], jointpos[:, 4]),  # Add other joints as needed
                        "LF_KFE": (jointpos[:, 8], jointpos[:, 8]),
                        # "RF_KFE": (jointpos[:, 10], jointpos[:, 10])
                    }

                    # Get the device of the robot data
                    device = robot._data.joint_limits.device  # Ensure this is the correct device reference

                    # Loop through the joint limits and apply them
                    for joint_name, limits in joint_limits.items():
                    # for i in range(1):
                        # joint_name = random.choice(jointname)
                        if joint_name in robot.joint_names:  # Check if the joint exists in the robot
                            joint_index = robot.joint_names.index(joint_name)  # Find the index of the joint
                            
                            limits = (jointpos[:, joint_index], jointpos[:, joint_index])
                            # Prepare limits as a tensor or array
                            limit_tensor = torch.tensor([limits], dtype=torch.float32, device=device)  # Shape (1, 2) for [min, max]
                            
                            # Write joint limits to the simulation
                            robot.write_joint_limits_to_sim(
                                limits=limit_tensor,  # Provide the limits tensor
                                joint_ids=[joint_index],  # Specify the joint index
                                env_ids=None  # Apply to all environments
                            )
                            print(f"[INFO] Updated joint limits for {joint_name}: {limits}")
                        else:
                            print(f"[WARNING] Joint {joint_name} not found in robot.joint_names.")
                        # jointname.remove(joint_name)
                else:
                    print("[ERROR] Robot object not found in the environment.")
            
            # print("time: ",sensor_node.time_step, "obs:", obs)

            for i in range(len(jointpos_n)):
                if round(jointpos_n[i], 3) == round(jointpos_p[i], 3) == round(jointpos_p2[i], 3):
                # if jointpos_n[i] == jointpos_p3[i] == jointpos_p5[i]:
                    print(f"time: {sensor_node.time_step}, joint: {robot._data.joint_names[i]}, index joint: {i} is locked")  # Return the matching value



            #
            # ========== set action = 0 and set environment = environment at step before ==========
            #

            # if timestep_fault >= 300:
            #     actions[:, 0] = 0.0
            #     actions[:, 4] = 0.0
            #     actions[:, 8] = 0.0

            # env.env.scene["robot"].write_joint_state_to_sim(obs[:, 12], 0.0, joint_ids=0)
            # env.env.scene["robot"].write_joint_state_to_sim(obs[:, 13], 0.0, joint_ids=1)
            # env.env.scene["robot"].write_joint_state_to_sim(obs[:, 14], 0.0, joint_ids=2)

            obs, _, _, _ = env.step(actions)

            LF_foot_index = env.env.scene["contact_forces"].body_names.index("LF_FOOT")
            LH_foot_index = env.env.scene["contact_forces"].body_names.index("LH_FOOT")
            RF_foot_index = env.env.scene["contact_forces"].body_names.index("RF_FOOT")
            RH_foot_index = env.env.scene["contact_forces"].body_names.index("RH_FOOT")

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
            observation = [obs[0, :].tolist()]

            sensor_node.publish_sensor_data(contact_forces)
            sensor_node.publish_velocity_data(base_lin_vel)

            if args_cli.log:
                sensor_node.log_data(contact_forces, base_lin_vel, joint_pos, observation)
            
            print("obs :", obs)
            # height = torch.max(env.env.scene["height_scanner"].data.ray_hits_w[..., -1]).item()
            # print("timestep :", sensor_node.time_step ,"height :", height)
            print("timestep :", sensor_node.time_step , "height :", robot.data.root_link_pos_w[:, 2])
            print("joint limit :", robot.data.default_joint_limits)
            # asset: RigidObject = env.env.scene["robot"]
            # print(asset.data.body_pos_w)
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # Spin the ROS node to handle callbacks
        rclpy.spin_once(sensor_node, timeout_sec=0)

    # close the simulator and ROS node
    env.close()
    sensor_node.close_logger()

    sensor_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()