from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeRoughRecoveryEnvCfg

from isaaclab.utils.math import quat_from_euler_xyz
import math
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
import extensions.tasks.locomotion.velocity.mdp as mdp


@configclass
class UnitreeFlatRecoveryEnvCfg(UnitreeRoughRecoveryEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.65
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeFlatRecoveryEnvCfg_PLAY(UnitreeFlatRecoveryEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02, 
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(float(0.7), float(0.7)), # Set fixed x velocity
                lin_vel_y=(float(0), float(0)), # Set fixed y velocity
                ang_vel_z=(0, 0), # Set fixed z velocity
                heading=(math.pi, math.pi)
            ),
        )