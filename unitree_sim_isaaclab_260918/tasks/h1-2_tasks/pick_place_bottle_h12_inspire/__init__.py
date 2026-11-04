# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import gymnasium as gym
import os

from . import pickplace_bottle_h12_inspire_joint_env_cfg

gym.register(
    id="Isaac-PickPlace-Bottle-H12-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_bottle_h12_inspire_joint_env_cfg.PickPlaceH12InspireHandBaseFixEnvCfg,
    },
    disable_env_checker=True,
)
