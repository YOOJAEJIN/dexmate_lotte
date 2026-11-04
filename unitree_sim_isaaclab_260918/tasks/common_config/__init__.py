"""
公共配置模块
提供可复用的机器人和相机配置
"""

from .robot_configs import RobotBaseCfg, H12RobotPresets, RobotJointTemplates,G1RobotPresets
from .camera_configs import CameraBaseCfg, CameraPresets
from .world_axes import world_axes_cfgs, WORLD_AXES_ENABLED

__all__ = [
    "RobotBaseCfg",
    "G1RobotPresets",
    "H12RobotPresets",
    "RobotJointTemplates", 
    "CameraBaseCfg",
    "CameraPresets",
    "world_axes_cfgs",
    "WORLD_AXES_ENABLED",
] 