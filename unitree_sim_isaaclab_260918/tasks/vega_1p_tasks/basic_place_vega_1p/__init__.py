
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import place_vega_1p_env_cfg


gym.register(
    id="Isaac-Place-VEGA-1P-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": place_vega_1p_env_cfg.PlaceVega1pBaseFixEnvCfg,
    },
    disable_env_checker=True,
)
