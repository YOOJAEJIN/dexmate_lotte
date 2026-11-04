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
class TableSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
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
            pos=[0.0, 0, 0.0],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",    # table model file
            usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/table.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, 0, 0.9],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            # usd_path=f"{project_root}/assets/objects/workspace_with_bin/small_klt.usd",
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                   pos_offset=(-0.1, 3.6, 1.6),
                                                   rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                   focal_length=16.5)