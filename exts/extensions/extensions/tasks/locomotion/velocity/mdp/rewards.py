from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

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


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def base_height_l2_notsensor(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    weight_exp_height: float = 1.0,
) -> torch.Tensor:
    """
    Penalizes the agent based on how far the current base height is below the target height.

    Args:
        env: Manager-based RL environment.
        target_height: Target height for the asset.
        asset_cfg: Configuration for the asset entity (default: robot).
        penalty_scale: Scale factor for the penalty.

    Returns:
        torch.Tensor: Penalty value.
    """
    # Extract the asset for height calculations
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the current height of the asset
    current_height = asset.data.root_link_pos_w[:, 2]

    # Compute the L2 squared penalty
    reward = torch.square(current_height - target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward

def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def upward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "mirror_joints_cache") or env.mirror_joints_cache is None:
        env.mirror_joints_cache = [
            asset.find_joints(joint_name) for joint_pair in mirror_joints for joint_name in joint_pair
        ]
    # compute out of limits constraints
    diff1 = torch.sum(
        torch.square(
            asset.data.joint_pos[:, env.mirror_joints_cache[0][0]]
            - asset.data.joint_pos[:, env.mirror_joints_cache[1][0]]
        ),
        dim=-1,
    )
    diff2 = torch.sum(
        torch.square(
            asset.data.joint_pos[:, env.mirror_joints_cache[2][0]]
            - asset.data.joint_pos[:, env.mirror_joints_cache[3][0]]
        ),
        dim=-1,
    )
    reward = 0.5 * (diff1 + diff2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
