from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


"""
Position-tracking rewards.
"""


def track_pose_xy(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    pose_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-pose_error / std**2)


"""
base height rewards toggle.
"""

def base_height_toggle(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Toggle reward term to 1 if height < 0.35 and 0 otherwise.

    Args:
        env: Manager-based RL environment.
        target_height: Target height for the asset.
        asset_cfg: Configuration for the asset entity (default: robot).
        sensor_cfg: Optional sensor configuration for height adjustment.

    Returns:
        torch.Tensor: Reward toggle (1 or 0).
    """
    # Extract the asset for height calculations
    asset: RigidObject = env.scene[asset_cfg.name]

    # Adjust the target height if a sensor is provided
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + sensor.data.pos_w[:, 2]
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height

    # Get the current height of the asset
    current_height = asset.data.root_link_pos_w[:, 2]

    # Toggle reward term based on height condition
    reward_toggle = (current_height >= target_height).float()

    return reward_toggle





"""
Step reward for get up and walk
"""

def step_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    weight_lin_vel: float = 1.15,
    weight_exp_height: float = 1.0,
    weight_height_toggle: float = 1.0,
) -> torch.Tensor:
    """Combined reward function based on height condition.

    Args:
        env: Manager-based RL environment.
        target_height: Target height for the asset.
        std: Standard deviation for the XY velocity task reward.
        command_name: Name of the command to track.
        asset_cfg: Configuration for the asset entity (default: robot).
        sensor_cfg: Optional sensor configuration for height adjustment.
        weight_lin_vel: Weight for the XY velocity reward.
        weight_height_toggle: Weight for the height toggle reward.

    Returns:
        torch.Tensor: Combined reward value.
    """
    # Extract the asset for height calculations
    asset: RigidObject = env.scene[asset_cfg.name]

    # Adjust the target height if a sensor is provided
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        adjusted_target_height = target_height + sensor.data.pos_w[:, 2]
    else:
        adjusted_target_height = target_height

    # Get the current height of the asset
    current_height = asset.data.root_link_pos_w[:, 2]

    # Calculate velocity reward
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - asset.data.root_com_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    lin_vel_reward = torch.exp(-lin_vel_error / std**2) * weight_lin_vel

    # Calculate exponential height reward
    height_difference = adjusted_target_height - current_height
    exp_height_reward = (1 - torch.exp(-torch.square(height_difference))) * weight_exp_height

    # Calculate height toggle reward
    height_toggle_reward = weight_height_toggle * (current_height < adjusted_target_height).float()

    # Combine rewards based on the height condition
    combined_reward = torch.where(
        current_height >= adjusted_target_height,
        lin_vel_reward * exp_height_reward,  # Multiply rewards when condition is met
        height_toggle_reward
    )

    return combined_reward


def vel_xy_toggle(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward function 1: Uses track_lin_vel_xy_exp if current height >= target height,
    otherwise sets reward to 0.

    Args:
        env: Manager-based RL environment.
        target_height: Target height for the asset.
        std: Standard deviation for the XY velocity task reward.
        command_name: Name of the command to track.
        asset_cfg: Configuration for the asset entity (default: robot).

    Returns:
        torch.Tensor: Reward value.
    """
    # Extract the asset for height calculations
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current height of the asset
    current_height = asset.data.root_link_pos_w[:, 2]

    # Calculate velocity reward
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - asset.data.root_com_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    lin_vel_reward = torch.exp(-lin_vel_error / std**2)

    # Apply condition based on height
    reward = torch.where(current_height >= target_height, lin_vel_reward, torch.zeros_like(lin_vel_reward))

    return reward



def base_height_exp_toggle(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    weight_exp_height: float = 1.0,
) -> torch.Tensor:
    """
    Reward function 2: Uses 1 * exp_height_reward if current height >= target height,
    otherwise sets reward to 0.

    Args:
        env: Manager-based RL environment.
        target_height: Target height for the asset.
        asset_cfg: Configuration for the asset entity (default: robot).
        exp_scale: Scale factor for exponential height reward.

    Returns:
        torch.Tensor: Reward value.
    """
    # Extract the asset for height calculations
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current height of the asset
    current_height = asset.data.root_link_pos_w[:, 2]

    # Calculate exponential height reward
    height_difference = target_height - current_height
    exp_height_reward = (1 - torch.exp(-torch.square(height_difference))) * weight_exp_height

    # Apply condition based on height
    reward = torch.where(current_height >= target_height, 0.4 + exp_height_reward, torch.zeros_like(exp_height_reward))

    return reward


def upright_orientation(
    env: ManagerBasedRLEnv,
    epsilon: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward function for upright orientation based on the equation:
    reward = exp(-(gz + 1)^2 / (2 * epsilon^2))

    Args:
        env: Manager-based RL environment.
        epsilon: Scaling factor for the exponential term.
        asset_cfg: Configuration for the asset entity (default: robot).

    Returns:
        torch.Tensor: Reward for upright orientation.
    """
    # Extract the asset for orientation calculations
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the z-component of the orientation (assumes quaternion orientation is provided)
    # Orientation as a quaternion (x, y, z, w)
    g_z = asset.data.projected_gravity_b[:,2]

    # Calculate the reward using the equation
    reward = torch.exp(-torch.square(g_z + 1) / (2 * epsilon**2))

    return reward


# def joint_limit_proximity_penalty(
#     env: ManagerBasedRLEnv,
#     penalty: float = 1.0,
#     proximity_threshold: float = 0.2,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """
#     Penalizes the robot as joint positions approach their limits (min or max) within a proximity threshold.

#     Args:
#         env: Manager-based RL environment.
#         joint_limits: A tensor of shape (num_joints, 2), where each row is [min_limit, max_limit].
#         penalty_scale: Scaling factor for the penalty (default: 1.0).
#         proximity_threshold: Threshold distance near the joint limits for penalty application (default: 0.2).
#         asset_cfg: Configuration for the asset entity (default: robot).

#     Returns:
#         torch.Tensor: A tensor of penalties for each environment instance.
#     """
#     # Extract the asset for joint position calculations
#     asset: RigidObject = env.scene[asset_cfg.name]

#     # Get the current joint positions of the robot
#     joint_positions = asset.data.joint_positions  # Shape: (batch_size, num_joints)
#     joint_limits = asset.data.joint_limits

#     # Separate joint limits into min and max
#     min_limits, max_limits = joint_limits[:, 0], joint_limits[:, 1]

#     # Calculate proximity to min and max limits
#     proximity_to_min = torch.clamp(min_limits + proximity_threshold - joint_positions, min=0.0)
#     proximity_to_max = torch.clamp(joint_positions - (max_limits - proximity_threshold), min=0.0)

#     # Penalize exponentially based on proximity
#     penalty_min = penalty * torch.exp(-proximity_to_min / proximity_threshold)
#     penalty_max = penalty * torch.exp(-proximity_to_max / proximity_threshold)

#     # Combine penalties for min and max proximity
#     total_penalty = torch.sum(penalty_min + penalty_max, dim=1)

#     return -total_penalty



def joint_limits(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    
    return torch.sum(out_of_limits, dim=1)