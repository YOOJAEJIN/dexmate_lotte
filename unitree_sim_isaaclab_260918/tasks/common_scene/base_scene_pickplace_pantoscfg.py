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
from tasks.common_config import CameraBaseCfg, world_axes_cfgs  # isort: skip
import os

project_root = os.environ.get("PROJECT_ROOT")

@configclass
class PantosSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",  # use simple room model
        ),
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_pantos/prop_Table_Sponge.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )

    long_table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/long_table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_pantos/prop_Table_02.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/card_box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -0.2, 0.76726],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_pantos/CardBox.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )




    libiao = RigidObjectCfg(
        prim_path="/World/envs/env_.*/libiao",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.65, -0.78468, 0.67],
            rot=[0, 0, 0, 1],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_pantos/Libiao.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                   pos_offset=(-0.1, 3.6, 1.6),
                                                   rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                   focal_length=16.5)

    # World-frame axis triad at (0, 0, 0): red +X, green +Y, blue +Z, white ball = origin.
    # Visual only (no physics). Hide with env var WORLD_AXES=0. See tasks/common_config/world_axes.py.
    locals().update(world_axes_cfgs())
