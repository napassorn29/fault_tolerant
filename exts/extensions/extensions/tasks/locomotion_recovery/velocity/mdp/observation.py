# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

# def joint_status(
#     env: ManagerBasedEnv, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """
#     Observes the status of joints in the robot.

#     Joint status:
#     - 1: Joint is active (position has changed between time steps).
#     - 0: Joint is damaged (position remains constant between time steps).

#     Args:
#         env: Manager-based RL environment.
#         asset_cfg: Configuration for the asset entity (default: robot).

#     Returns:
#         torch.Tensor: A tensor of size (num_envs, num_joints) representing the joint statuses.
#     """
#     # Extract the articulation asset
#     asset: Articulation = env.scene[asset_cfg.name]

#     # Current and previous joint positions
#     current_joint_pos = asset._data.joint_pos  # Shape: (num_envs, num_joints)
#     prev_joint_pos = getattr(asset.data, "prev_joint_pos", None)  # Retrieve stored previous joint positions

#     # Initialize previous joint positions if not already available
#     if prev_joint_pos is None:
#         # Assume joints are initially active
#         prev_joint_pos = current_joint_pos.clone()
#         asset._data.prev_joint_pos = prev_joint_pos

#     # Compare current and previous joint positions to determine joint activity
#     joint_status = (current_joint_pos != prev_joint_pos).float()  # 1 if active, 0 if damaged

#     # Update previous joint positions for the next observation step
#     asset.data.prev_joint_pos = current_joint_pos.clone()

#     return joint_status

def joint_status(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Observes the status of joints in the robot.

    Joint status:
    - 1: Joint is active (position has changed across the three latest time steps).
    - 0: Joint is damaged (position remains constant across the three latest time steps).

    Args:
        env: Manager-based RL environment.
        asset_cfg: Configuration for the asset entity (default: robot).

    Returns:
        torch.Tensor: A tensor of size (num_envs, num_joints) representing the joint statuses.
    """
    # Extract the articulation asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Current and previous joint positions
    current_joint_pos = asset._data.joint_pos  # Shape: (num_envs, num_joints)
    prev1_joint_pos = getattr(asset._data, "prev1_joint_pos", None)
    prev2_joint_pos = getattr(asset._data, "prev2_joint_pos", None)

    # Initialize previous joint positions if not already available
    if prev1_joint_pos is None or prev2_joint_pos is None:
        # Assume joints are initially active
        prev1_joint_pos = current_joint_pos.clone()
        prev2_joint_pos = current_joint_pos.clone()
        asset._data.prev1_joint_pos = prev1_joint_pos
        asset._data.prev2_joint_pos = prev2_joint_pos

    # Round positions to 3 decimal places for comparison
    rounded_current = torch.round(current_joint_pos * 1000) / 1000
    rounded_prev1 = torch.round(prev1_joint_pos * 1000) / 1000
    rounded_prev2 = torch.round(prev2_joint_pos * 1000) / 1000

    # Check if positions are equal across the three time steps
    joint_status = ~((rounded_current == rounded_prev1) & (rounded_prev1 == rounded_prev2))
    joint_status = joint_status.float()  # Convert to float after the bitwise operation if required.

    # Update previous joint positions for the next observation step
    asset._data.prev2_joint_pos = prev1_joint_pos.clone()
    asset._data.prev1_joint_pos = current_joint_pos.clone()

    return joint_status