
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import pickplace_bin_vega_1p_xhand_env_cfg

gym.register(
    id="Isaac-PickPlace-Bin-VEGA-1P-Xhand-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_bin_vega_1p_xhand_env_cfg.PickPlaceVega1pXhandEnvCfg,
    },
    disable_env_checker=True,
)
