# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os

project_root = os.environ.get("PROJECT_ROOT")

@configclass
class BaseSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    # 0. ground plane configuration
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=GroundPlaneCfg(
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.5,
    #             restitution=0.0,
    #         )
    #     ),
    # )

    # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",  # use simple room model
        ),
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                   pos_offset=(-0.1, 3.6, 1.6),
                                                   rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                   focal_length=16.5)