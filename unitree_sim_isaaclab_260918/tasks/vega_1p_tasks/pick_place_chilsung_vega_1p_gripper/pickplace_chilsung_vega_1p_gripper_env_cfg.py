# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import tempfile
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
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_event.reset_events import reset_scene_to_default_skip_kinematic

# import public scene configuration
from tasks.common_scene.base_scene_pickplace_bincfg import  ConveyorBinSceneCfg
from ...common_config.robot_configs import Vega1pRobotPresets
from ...common_scene.base_scene_pickplace_chilsungcfg import ChilsungSceneCfg

# card box stack assets in the scene (bottom card first)
CARD_BOX_ASSET_NAMES = ["object", "object1", "object2", "object3", "object4",
                        "object5", "object6", "object7", "object8", "object9"]


##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(ChilsungSceneCfg):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    # robot: ArticulationCfg = H12RobotPresets.h12_27dof_inspire_base_fix(
    #     init_pos=(0.7, -0.6, 1.024),
    #     init_rot=(0.7071, 0, 0, 0.7071)
    # )

    # robot: ArticulationCfg = Vega1pRobotPresets.vega_1p_vertical_gripper(
    #     init_pos=(-0.21, -1.21, 0.04426),
    #     init_rot=(1, 0, 0, 0)
    # )

    robot: ArticulationCfg = Vega1pRobotPresets.vega_1p_vertical_gripper(
        init_pos=(-0.08, -1.21, 0.04426),
        init_rot=(1, 0, 0, 0)
    )



    # 6. add camera configuration 
    head_camera = CameraPresets.vega1p_front_camera()
    head_right_camera = CameraPresets.vega1p_front_right_camera()
    left_wrist_camera = CameraPresets.vega1p_left_wrist_camera()
    right_wrist_camera = CameraPresets.vega1p_right_wrist_camera()

    def __post_init__(self):
        # Keep the conveyor AprilTag at the robot's world x (robot yaw is 0, so base x == world x):
        # the tag then sits straight to the robot's left on the conveyor face, whatever init_pos
        # is set above. sim_main.update_apriltag_status reads this cfg, so apriltag_pose.txt follows.
        self.conveyor_apriltag.init_state.pos[0] = float(self.robot.init_state.pos[0])

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
            "R_Slider_1", "R_Slider_2", "L_Slider_1", "L_Slider_2",
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
        func=mdp.reset_root_state_uniform,  # move the whole stack with one shared random offset
        mode="reset",   # set event mode to reset
        params={
            # position range parameter (applied to all cards together, stack stays aligned)
            "pose_range": {
                "x": [0.0, 0.0],  # x axis position range: -0.05 to 0.05 meter
                "y": [0.0, 0.0],    # y axis: inward only (+y) - -y direction would push the stack
                                     # back over the table edge (table surface ends at y=-0.40)
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
        self.sim.physx.bounce_threshold_velocity = 0.5  # keep default: a low value makes thin objects bounce on contact
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # create event manager
        self.event_manager = SimpleEventManager()

        # register "reset object" event (moves the whole card stack together)
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: mdp.reset_card_stack_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
                asset_names=CARD_BOX_ASSET_NAMES,
            )
        ))
        
        # note: skips kinematic marker bodies -- writing a velocity to them makes PhysX log
        # "PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!" on every reset.
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: reset_scene_to_default_skip_kinematic(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))
