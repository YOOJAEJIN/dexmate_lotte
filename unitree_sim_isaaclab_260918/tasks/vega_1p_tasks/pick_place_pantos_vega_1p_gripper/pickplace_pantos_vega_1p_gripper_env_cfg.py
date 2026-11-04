# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import tempfile
import math
import torch
from dataclasses import MISSING

from pink.tasks import FrameTask

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  H12RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager, BatchObjectEvent

# import public scene configuration
from tasks.common_scene.base_scene_pickplace_bincfg import  ConveyorBinSceneCfg
from ...common_config.robot_configs import Vega1pRobotPresets
from ...common_scene.base_scene_pickplace_pantoscfg import PantosSceneCfg


##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(PantosSceneCfg):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """

    robot: ArticulationCfg = Vega1pRobotPresets.vega_1p_gripper(
        init_pos=(-1.01755, 0.0, 0.05562),
        init_rot=(1.0, 0, 0, 0.0)
    )

    # 6. add camera configuration 
    head_camera = CameraPresets.vega1p_front_camera()
    head_right_camera = CameraPresets.vega1p_front_right_camera()
    left_wrist_camera = CameraPresets.vega1p_left_wrist_camera()
    right_wrist_camera = CameraPresets.vega1p_right_wrist_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    arm_head_torso_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "L_wheel_j1", "R_wheel_j1",
            "torso_j1", "torso_j2", "torso_j3",
            "head_j1", "head_j2", "head_j3",
            "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", "L_arm_j5", "L_arm_j6", "L_arm_j7",
            "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4", "R_arm_j5", "R_arm_j6", "R_arm_j7",
            "L_finger_joint", "R_finger_joint",
        ],
        scale=1.0,
        use_default_offset=True
    )
    chassis_drive_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["L_wheel_j2", "R_wheel_j2"],
        scale=100.0
    )



@configclass
class ObservationsCfg:
    """
    defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states, params={"enable_dds": False})
        # robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)

        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    # check if the object is out of the working range
    success = DoneTerm(func=mdp.reset_object_estimate)# use task completion check function

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)

@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,  # use uniform distribution reset function
        mode="reset",   # set event mode to reset
        params={
            # position range parameter
            "pose_range": {
                "x": [-0.05, 0.05],  # x axis position range: -0.05 to 0.0 meter
                "y": [-0.05, 0.05],   # y axis position range: 0.0 to 0.05 meter
            },
            # speed range parameter (empty dictionary means using default value)
            "velocity_range": {},
            # specify the object to reset
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceVega1pGripperEnvCfg(ManagerBasedRLEnvCfg):
    """
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=2.5, # environment spacing: 2.5 meter
                                                     replicate_physics=True # enable physics replication
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings
        
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # create event manager
        self.event_manager = SimpleEventManager()

        # register "reset object" event
        # object는 랜덤 범위로 리셋하고, plastic_box는 빈 pose_range로 제자리 리셋
        self.event_manager.register("reset_object_self", BatchObjectEvent(
            object_names=["object", "plastic_box"],
            pose_ranges={
                "object": {
                    "x": [-0.05, 0.05],
                    "y": [-0.12, 0.12],
                    # reset_root_state_uniform은 "roll", "pitch", "yaw" 키를
                    # (min, max) 라디안 범위로 읽습니다. z축 회전은 "yaw" 입니다.
                    "yaw": [-math.pi / 2, math.pi / 2],  # -90도 ~ +90도
                },
                "plastic_box": {},  # 빈 딕셔너리 → 기본 위치로 제자리 리셋
            },
            velocity_ranges={}
        ))
        
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))
