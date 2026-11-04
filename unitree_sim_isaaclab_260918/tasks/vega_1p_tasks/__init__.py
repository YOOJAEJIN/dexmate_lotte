
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

# use relative import

from . import pick_place_bin_vega_1p
from . import pick_place_bin_vega_1p_gripper
from . import pick_place_bin_vega_1p_xhand
from . import pick_place_chilsung_vega_1p_gripper
from . import basic_place_vega_1p
from . import pick_place_pantos_vega_1p_xhand
from . import pick_place_pantos_vega_1p_gripper

# export all modules
__all__ = [
        "basic_place_vega_1p",
        "pick_place_bin_vega_1p",
        "pick_place_bin_vega_1p_gripper",
        "pick_place_chilsung_vega_1p_gripper",
        "pick_place_bin_vega_1p_xhand",
        "pick_place_pantos_vega_1p_xhand",
        "pick_place_pantos_vega_1p_gripper",
]