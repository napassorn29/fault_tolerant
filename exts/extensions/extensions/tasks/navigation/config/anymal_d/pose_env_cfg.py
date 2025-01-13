from omni.isaac.lab.utils import configclass

from extensions.tasks.navigation.config.anymal_d.navigation_env_cfg import NavigationCommandEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip

@configclass
class NavigationEnvCfg(NavigationCommandEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
