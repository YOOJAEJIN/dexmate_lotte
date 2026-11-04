# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import CameraBaseCfg  # isort: skip
import os

project_root = os.environ.get("PROJECT_ROOT")


@configclass
class LongTableBottleSceneCfg(InteractiveSceneCfg):  # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",  # use simple room model
        ),
    )

    # Object
    # 2. wood wall table
    packing_table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WoodWallTable",  # table in the scene
        init_state=RigidObjectCfg.InitialStateCfg(
            #pos=[-0.19664, 0.42, 0.242],  # initial position [x, y, z]
            pos=[-0.19664, 0.42, 0.280],  # bear's modif part
            rot=[1.0, 0.0, 0.0, 0.0]
        ),  # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bottle/table_with_wall_longleg.usd",
            # table model file
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # big box
    big_box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BigBox",
        init_state=RigidObjectCfg.InitialStateCfg(
            #pos=[-0.3065, 0.364, 1.03],
            pos=[-0.3065, 0.364, 1.08],  # bear's modif part
            rot=[0.7071, 0.0, 0.0, 0.7071],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bottle/big_box.usd",
            scale=(1.0, 1.0, 1.0),  # 크기 조정
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Bottle
    bottle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bottle",
        init_state=RigidObjectCfg.InitialStateCfg(
            #pos=[-0.078, 0.39, 1.01],
            pos=[-0.078, 0.39, 1.11],  # bear's modif part
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bottle/bottle.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # object = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Ball",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.6, 0.40, 0.84],
    #         rot=[1.0, 0.0, 0.0, 0.0],
    #     ),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{project_root}/assets/objects/workspace_with_bottle/tennis_ball.usd",
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #     # RigidBodyAPI가 자동으로 적용되도록 설정
    #         #     kinematic_enabled=False,  # 동적 객체로 설정
    #         # ),
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    #         # collision_props=sim_utils.CollisionPropertiesCfg(
    #         #     # 충돌 감지 활성화
    #         #     collision_enabled=True,
    #         # ),
    #         # # 추가: activate_contact_sensors를 True로 설정
    #         # activate_contact_sensors=True,
    #     ),
    # )

    # Ground plane
    # 3. ground configuration
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",  # ground in the scene
        spawn=GroundPlaneCfg(),  # ground configuration
    )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",  # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),  # light color (white)
                                     intensity=3000.0),  # light intensity
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                   pos_offset=(-0.1, 3.6, 1.6),
                                                   rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                   focal_length=16.5)
