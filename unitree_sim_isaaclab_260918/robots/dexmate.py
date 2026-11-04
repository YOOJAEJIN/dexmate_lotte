import os
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
VEGA_1P_USD_DIR = f"{PROJECT_ROOT}/assets/robots/vega-1p-usd"

DEFAULT_INIT_POS = (0.7, -0.6, 0.04426)
DEFAULT_INIT_ROT = (0.7071, 0.0, 0.0, 0.7071)

WHEEL_JOINTS = [
    "L_wheel_j1",
    "L_wheel_j2",
    "R_wheel_j1",
    "R_wheel_j2",
]

WHEEL_STIFFNESS = {
    "L_wheel_j1": 100.0,
    "L_wheel_j2": 0.0,
    "R_wheel_j1": 100.0,
    "R_wheel_j2": 0.0,
}
WHEEL_DAMPING = {
    "L_wheel_j1": 80.0,
    "L_wheel_j2": 10.0,
    "R_wheel_j1": 80.0,
    "R_wheel_j2": 10.0,
}
WHEEL_ARMATURE = {
    "L_wheel_j1": 0.1,
    "L_wheel_j2": 0.02,
    "R_wheel_j1": 0.1,
    "R_wheel_j2": 0.02,
}

TORSO_STIFFNESS = {
    "torso_j1": 5200.0,
    "torso_j2": 3800.0,
    "torso_j3": 4900.0,
}
TORSO_DAMPING = {
    "torso_j1": 520.0 * 1.2,
    "torso_j2": 380.0 * 1.2,
    "torso_j3": 490.0 * 1.2,
}

ARM_STIFFNESS = {
    ".*_arm_j1": 8000.0,   # 어깨는 자중 하중이 커서 다소 높게
    ".*_arm_j2": 6000.0,
    ".*_arm_j3": 5000.0,
    ".*_arm_j4": 4000.0,
    ".*_arm_j5": 2500.0,   # 손목쪽은 접촉 순응성을 위해 낮게
    ".*_arm_j6": 2000.0,
    ".*_arm_j7": 1500.0,
}
ARM_DAMPING = {
    ".*_arm_j1": 500.0,
    ".*_arm_j2": 400.0,
    ".*_arm_j3": 320.0,
    ".*_arm_j4": 260.0,
    ".*_arm_j5": 170.0,
    ".*_arm_j6": 150.0,
    ".*_arm_j7": 120.0,
}

XHAND_JOINTS = [
    ".*_hand_thumb_bend_joint",
    ".*_hand_thumb_rota_joint1",
    ".*_hand_thumb_rota_joint2",
    ".*_hand_index_bend_joint",
    ".*_hand_index_joint1",
    ".*_hand_index_joint2",
    ".*_hand_mid_joint1",
    ".*_hand_mid_joint2",
    ".*_hand_ring_joint1",
    ".*_hand_ring_joint2",
    ".*_hand_pinky_joint1",
    ".*_hand_pinky_joint2",
]


def _usd_path(filename: str) -> str:
    return f"{VEGA_1P_USD_DIR}/{filename}"


def _init_state_cfg(init_joint_pos: Optional[dict] = None) -> ArticulationCfg.InitialStateCfg:
    # 주의: joint_pos는 strict regex 매칭이라 특정 관절 키와 ".*"를 함께 쓸 수 없다.
    # dict에 없는 관절은 0.0으로 초기화된다.
    if init_joint_pos is None:
        init_joint_pos = {".*": 0.0}
    return ArticulationCfg.InitialStateCfg(
        pos=DEFAULT_INIT_POS,
        rot=DEFAULT_INIT_ROT,
        joint_pos=init_joint_pos,
        joint_vel={".*": 0.0},
    )


def _spawn_cfg(usd_filename: str) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=_usd_path(usd_filename),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )


def _chassis_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=WHEEL_JOINTS,
        effort_limit=200.0,
        velocity_limit=10.0,
        stiffness=WHEEL_STIFFNESS,
        damping=WHEEL_DAMPING,
        armature=WHEEL_ARMATURE,
    )


def _torso_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=["torso_.*"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=TORSO_STIFFNESS,
        damping=TORSO_DAMPING,
        armature=None,
    )


def _head_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=["head_.*"],
        effort_limit=None,
        velocity_limit=None,
        stiffness={"head_.*": 300.0},
        damping={"head_.*": 7.5},
        armature=None,
    )


def _arms_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=[".*_arm_.*"],
        # implicit PD에는 중력 보상(feedforward)이 없어서 실기 스펙(25 N·m) 그대로면
        # 손목 하중을 못 이김 -> 시뮬레이션용 상한 상향
        effort_limit_sim={
            ".*_arm_j1": 200.0,
            ".*_arm_j2": 200.0,
            ".*_arm_j3": 150.0,
            ".*_arm_j4": 150.0,
            ".*_arm_j5": 100.0,
            ".*_arm_j6": 100.0,
            ".*_arm_j7": 100.0,
        },
        velocity_limit=None,
        stiffness=ARM_STIFFNESS,
        damping=ARM_DAMPING,
        armature=None,
    )


def _gripper_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=[
            "R_finger_joint",
            "L_finger_joint",
        ],
        effort_limit=None,
        velocity_limit=None,
        stiffness={
            "R_finger_joint": 30.0,
            "L_finger_joint": 30.0,
        },
        damping={
            "R_finger_joint": 3.0,
            "L_finger_joint": 3.0,
        },
        armature=None,
    )

def _vertical_gripper_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=[
            "R_Slider_1",
            "R_Slider_2",
            "L_Slider_1",
            "L_Slider_2",
        ],
        effort_limit_sim=30.0,
        effort_limit=None,
        velocity_limit=None,
        stiffness={
            "R_Slider_1": 2000.0,
            "R_Slider_2": 2000.0,
            "L_Slider_1": 2000.0,
            "L_Slider_2": 2000.0,
        },
        damping={
            "R_Slider_1": 80.0,
            "R_Slider_2": 80.0,
            "L_Slider_1": 80.0,
            "L_Slider_2": 80.0,
        },
        armature=None,
    )

def _xhand_actuator_cfg() -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=XHAND_JOINTS,
        effort_limit=None,
        velocity_limit=None,
        stiffness={
            "left_hand_.*": 30.0,
            "right_hand_.*": 30.0,
        },
        damping={
            "left_hand_.*": 3.0,
            "right_hand_.*": 3.0,
        },
        armature=None,
    )


def  _base_actuators() -> dict[str, ImplicitActuatorCfg]:
    return {
        "chassis": _chassis_actuator_cfg(),
        "torso": _torso_actuator_cfg(),
        "head": _head_actuator_cfg(),
        "arms": _arms_actuator_cfg(),
    }


def _vega_1p_cfg(
    usd_filename: str,
    hand_actuator: Optional[ImplicitActuatorCfg] = None,
    init_joint_pos: dict = None,
) -> ArticulationCfg:
    actuators = _base_actuators()
    if hand_actuator is not None:
        actuators["hands"] = hand_actuator

    return ArticulationCfg(
        init_state=_init_state_cfg(init_joint_pos),
        spawn=_spawn_cfg(usd_filename),
        actuators=actuators,
    )


# noinspection PyArgumentList
VEGA_1P_CFG_BASE = _vega_1p_cfg("vega_1p_no.usd")

# noinspection PyArgumentList
VEGA_1P_CFG_MITTEN = _vega_1p_cfg("vega_1p_hook.usd")

# noinspection PyArgumentList
VEGA_1P_CFG_GRIPPER = _vega_1p_cfg(
    "vega_1p_2f85_gripper.usd",
    hand_actuator=_gripper_actuator_cfg(),
)

# noinspection PyArgumentList
VEGA_1P_CFG_VERTICAL_GRIPPER = _vega_1p_cfg(
    "vega_1p_vertical_gripper.usd",
    hand_actuator=_vertical_gripper_actuator_cfg(),
    # 실기 초기 자세로 매핑 (명시되지 않은 관절은 0.0 유지)
    # left arm	[0.842, 0.000, -0.000, -1.446, -0.001, 0.003, 0.000]
    # right arm	[-0.832, -0.000, 0.000, -1.445, -0.001, -0.002, -0.000]
    # head	[-0.184, -0.001, -0.376]
    # torso	[0.214, 1.473, 0.644]

    # init_joint_pos={
    #     # left arm
    #     "L_arm_j1": 0.842,
    #     "L_arm_j2": 0.000,
    #     "L_arm_j3": 0.15,
    #     "L_arm_j4": -1.446,
    #     "L_arm_j5": -0.001,
    #     "L_arm_j6": 0.003,
    #     "L_arm_j7": 0.000,
    #     # right arm
    #     "R_arm_j1": -0.832,
    #     "R_arm_j2": 0.000,
    #     "R_arm_j3": -0.15,
    #     "R_arm_j4": -1.445,
    #     "R_arm_j5": -0.001,
    #     "R_arm_j6": -0.002,
    #     "R_arm_j7": 0.000,
    #     # head
    #     "head_j1": -0.184,
    #     "head_j2": -0.001,
    #     "head_j3": -0.376,
    #     # torso
    #     "torso_j1": 0.214,
    #     "torso_j2": 1.473,
    #     "torso_j3": 0.644,
    # },

    init_joint_pos= {
        'L_arm_j1': 1.5745360851287842,
        'L_arm_j2': -4.601889304467477e-05,
        'L_arm_j3': -0.0001406389201292768,
        'L_arm_j4': -1.5620099306106567,
        'L_arm_j5': -0.00018947907665278763,
        'L_arm_j6': 0.004862525034695864,
        'L_arm_j7': 0.0001019211849779822,
        'R_arm_j1': -1.5743721723556519,
        'R_arm_j2': 0.00020777805184479803,
        'R_arm_j3': 0.00020511656475719064,
        'R_arm_j4': -1.5603846311569214,
        'R_arm_j5': -0.00010557365021668375,
        'R_arm_j6': -0.004143048543483019,
        'R_arm_j7': 3.1130890420172364e-05,
        'head_j1': -0.08560948818922043,
        'head_j2': 0.000296128389891237,
        'head_j3': 0.00010670589108485729,
        'torso_j1': 0.083,
        'torso_j2': 0.4230404496192932,
        'torso_j3': 0.11368651688098907
        }
)

# noinspection PyArgumentList
VEGA_1P_CFG_XHAND = _vega_1p_cfg(
    "vega_1p_xhand.usd",
    hand_actuator=_xhand_actuator_cfg(),
)
