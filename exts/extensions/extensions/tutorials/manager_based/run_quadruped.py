# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

# import rospy
import rclpy
from geometry_msgs.msg import Vector3
from quadruped_env import LocomotionVelocityRoughEnvCfg
from rclpy.node import Node

from omni.isaac.lab.envs import ManagerBasedRLEnv


class SensorRunner(Node):
    def __init__(self):
        super().__init__("run_quadruped_env")
        self.pub_contact_base = self.create_publisher(Vector3, "/isaac_contact/base", 10)
        self.pub_contact_LF = self.create_publisher(Vector3, "/isaac_contact/LF_foot", 10)
        self.pub_contact_LH = self.create_publisher(Vector3, "/isaac_contact/LH_foot", 10)
        self.pub_contact_RF = self.create_publisher(Vector3, "/isaac_contact/RF_foot", 10)
        self.pub_contact_RH = self.create_publisher(Vector3, "/isaac_contact/RH_foot", 10)

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
            self.get_logger().info(f"Published {name} contact force: {data}")


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    # create environment configuration
    env_cfg = LocomotionVelocityRoughEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # initialize ROS node
    sensor_node = SensorRunner()

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            print("Received max contact force of: ", torch.max(env.scene["contact_forces"].data.net_forces_w).item())

            # Extract contact forces
            contact_forces = [
                env.scene["contact_forces"].data.net_forces_w[0, 0].cpu().tolist(),  # base
                env.scene["contact_forces"].data.net_forces_w[0, 4].cpu().tolist(),  # LF_foot
                env.scene["contact_forces"].data.net_forces_w[0, 8].cpu().tolist(),  # LH_foot
                env.scene["contact_forces"].data.net_forces_w[0, 12].cpu().tolist(),  # RF_foot
                env.scene["contact_forces"].data.net_forces_w[0, 16].cpu().tolist(),  # RH_foot
            ]

            # Publish contact sensor data
            sensor_node.publish_sensor_data(contact_forces)

            # update counter
            count += 1

        # Spin the ROS node to handle callbacks
        rclpy.spin_once(sensor_node, timeout_sec=0)

    # close the environment and ROS node
    env.close()
    sensor_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
