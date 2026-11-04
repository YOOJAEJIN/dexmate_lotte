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

# conveyor root (world). Its -Y side face, which looks at the robot, is at y = CONVEYOR_POS[1] - 0.288.
CONVEYOR_POS = (-4.8, -0.4, 0.0)   # 2026-09-26 18:5x: y = -0.47 so that after 8.rotate_base (base square-on at 90 deg) the held block
                                   # sits on the conveyor box's centre line along the belt (measured 12 cm short at y = -0.35).
                                   # conveyor -Y face constants derive from this -- move the conveyor ONLY here.

# card_box stack constants
# note: card_box.usd origin is at the center of the card, so `pos` below sets the card CENTER.
#       Measured from the USD meshes (tools/usd_mesh_extents.py): bbox x +/-0.200, y +/-0.175,
#       z +/-0.0015 -> a flat 40 x 35 cm, 3 mm blank made of BODY (x +/-0.160, y +/-0.135) plus
#       four 4 cm flaps (LEFT/RIGHT at |x| 0.160..0.200 with y +/-0.133; TOP/BOTTOM at |y|
#       0.135..0.175 spanning the full x). The -X edge (x = -0.200) is therefore full width,
#       with 2 mm slits at |y| = 0.133..0.135 between the left flap and the top/bottom flaps.
CARD_BOX_USD_PATH = f"{project_root}/assets/objects/workspace_with_chilsung/card_box.usd"
# table top surface is flat at z=0.7581 and ends at y=-0.40 (nut_pour table).
# the card center must keep +/-0.175 m in y inside the table: center y=-0.185 puts the
# footprint at y=-0.36 ~ -0.01, fully on the table with a 4cm margin to the edge.
CARD_BOX_BASE_POS = (0.8 - 0.005, -1.2 + 0.005)
CARD_BOX_BASE_Z = 0.68          # center height of the first card: table top 0.7581 + half thickness
                                   # 0.0015 + clearance -> the stack spawns resting on the table

#CARD_BOX_THICKNESS = 0.003         # 3mm thickness
CARD_BOX_THICKNESS = 0.006         # 3mm thickness



CARD_BOX_GAP = 0.001               # small air gap so cards never spawn interpenetrating

# AprilTag (tag36h11) on the conveyor side face that looks at the robot.
# Asset from tools/make_apriltag_usd.py: quad normal +Z, +X image right, +Y image up,
# quad side 0.30 m (tag 0.24 m + quiet zone).
# Conveyor bbox is y = CONVEYOR_POS[1] +/- 0.288 (world); its -Y side panel spans z ~0.36..0.73.
# The robot places boxes standing ~0.6 m from that face with the head camera only
# ~0.3 m from it and ~0.8 m above the panel, so a flush vertical tag would be seen at
# ~70 deg incidence. The tag is therefore mounted like a small angled sign: its
# lower edge touches the -Y face and its normal is tilted CONVEYOR_APRILTAG_TILT_DEG above
# horizontal (0 = flush vertical mount). The default x below (-0.2, the box-row tip) is
# overridden by pickplace_chilsung_vega_1p_gripper_env_cfg.ObjectTableSceneCfg.__post_init__,
# which sets it to the robot's init x so the tag is always level with the robot base.
# Rotation is Rx(90 - tilt): quad +Z -> (0, -cos t, sin t), quad +Y -> (0, sin t, cos t).
# Pose is env-local (equals world for a single env). sim_main writes the same pose in
# the robot base frame to apriltag_pose.txt as ground truth for detect_apriltag.py.
import math as _math

CONVEYOR_APRILTAG_USD_PATH = f"{project_root}/assets/objects/apriltag/apriltag_36h11_00000.usda"
CONVEYOR_APRILTAG_ID = 0
CONVEYOR_APRILTAG_SIZE = 0.24                      # black-square side (m)
CONVEYOR_APRILTAG_TILT_DEG = 55.0                  # normal tilt above horizontal
_CONVEYOR_FACE_Y = CONVEYOR_POS[1] - 0.288          # conveyor -Y face (world)
_TAG_HALF = CONVEYOR_APRILTAG_SIZE * 10 / 8 / 2    # quad half side (0.15 m)
_TAG_RX = _math.radians(90.0 - CONVEYOR_APRILTAG_TILT_DEG)
CONVEYOR_APRILTAG_POS = (
    -0.2,
    _CONVEYOR_FACE_Y - 0.009 - _TAG_HALF * _math.cos(_TAG_RX),  # lower edge ~9 mm proud of face
    0.62,                                                        # spans z ~0.53..0.71 on the panel
)
CONVEYOR_APRILTAG_ROT = (_math.cos(_TAG_RX / 2), _math.sin(_TAG_RX / 2), 0.0, 0.0)  # (w, x, y, z)


def _card_box_cfg(prim_name: str, center_z: float, y_pos = CARD_BOX_BASE_POS[1]) -> RigidObjectCfg:
    """create one card_box rigid object at absolute card-center height `center_z`.

    thin flat objects (3mm) explode apart when spawned overlapping or dropped from a height,
    so all cards are spawned pre-stacked with a small gap and thin-object contact settings.
    """
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[CARD_BOX_BASE_POS[0], y_pos, center_z],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=CARD_BOX_USD_PATH,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,  # 유지 (44장 박판 스택 처짐 방지). 성능 부담 시 64부터 시험
                solver_velocity_iteration_count=16,  # 4 -> 16: 팁이 스택을 '마찰로 쥐고' 버티는 힘은 velocity solver가 냄
                max_depenetration_velocity=0.05,  # 유지: 끼어든 팁이 튕겨 나가지 않게
                linear_damping=0.2,  # 0.1 kg 박판의 접촉 진동 억제
                angular_damping=0.5,  # 찌를 때 카드가 빙글 돌아 이탈하는 것 억제
                max_linear_velocity=10.0,  # 접촉 글리치 시 폭발적 가속 방지 캡
                max_angular_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                # contacts are generated just before the thin (3mm) hulls touch,
                # and objects rest exactly in contact (no air gap, no early repulsion)
                contact_offset=0.004,
                rest_offset=0.000085,
            ),
            # note: UsdFileCfg does not support `physics_material` in isaaclab 0.48 (IsaacLab v2.3.2),
            # so the default physics material (restitution 0) is used instead.
        ),
    )


def _card_box_block_cfg(prim_name: str, sheets: int, bottom_z: float, y_pos = CARD_BOX_BASE_POS[1]) -> RigidObjectCfg:
    """create a merged block of `sheets` cards sitting on `bottom_z` (block BOTTOM face).

    스택의 위/중간 구간을 하나의 강체로 통합해 접촉 쌍과 강체 수를 줄이는 성능 최적화용.
    블록은 내부 틈 없는 솔리드(sheets x 3mm)이며, 낱장과 동일한 접촉 설정을 쓴다.
    """
    height = sheets * CARD_BOX_THICKNESS
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[CARD_BOX_BASE_POS[0], y_pos, bottom_z + height / 2],
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=CARD_BOX_USD_PATH,
            scale=(1.0, 1.0, float(sheets)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=16,
                max_depenetration_velocity=0.05,
                linear_damping=0.2,
                angular_damping=0.5,
                max_linear_velocity=10.0,
                max_angular_velocity=100.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1 * sheets),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.000085,
            ),
        ),
    )


# Coloured edge markers on the card stack. Four points on the -X edge (card-local x = -CARD_BOX_HALF_X)
# of the BOTTOM loose card (object) and the MIDDLE card of the stack, at y = +/-CARD_EDGE_MARKER_Y.
# "+0.5 / -0.5" is read as a fraction of the card half-width (0.175 m) -> +/-0.0875 m; set
# CARD_EDGE_MARKER_Y = 0.5 for an absolute 0.5 m offset instead.
# Stack layout (see the class body): 10 loose cards (object..object9), a 30-sheet block, 10 loose
# cards (object10..object19), a 35-sheet block = 85 sheets. The middle sheet (#42) is object12.
# sim_main.update_card_stack_edge_markers() snaps each marker to its card's live pose every step,
# so the spawn poses only need to be roughly right.
CARD_BOX_HALF_X = 0.200                # measured -X/+X extent of card_box.usd (see note at top)
CARD_BOX_HALF_Y = 0.175
CARD_STACK_MIDDLE_KEY = "object12"
CARD_EDGE_MARKER_Y = 0.5 * CARD_BOX_HALF_Y
CARD_EDGE_MARKER_RADIUS = 0.015
CARD_EDGE_MARKERS = {
    # scene key                   : (card scene key,        card-local offset (x, y, z),                     colour)
    "card_mid_edge_py_marker":    (CARD_STACK_MIDDLE_KEY, (-CARD_BOX_HALF_X, +CARD_EDGE_MARKER_Y, 0.0), (1.0, 0.9, 0.0)),  # yellow
    "card_mid_edge_ny_marker":    (CARD_STACK_MIDDLE_KEY, (-CARD_BOX_HALF_X, -CARD_EDGE_MARKER_Y, 0.0), (0.0, 0.9, 1.0)),  # cyan
    "card_bottom_edge_py_marker": ("object",              (-CARD_BOX_HALF_X, +CARD_EDGE_MARKER_Y, 0.0), (1.0, 0.5, 0.0)),  # orange
    "card_bottom_edge_ny_marker": ("object",              (-CARD_BOX_HALF_X, -CARD_EDGE_MARKER_Y, 0.0), (0.1, 0.2, 1.0)),  # blue
}


def _edge_marker_cfg(prim_name: str, card_pos, local: tuple, color: tuple) -> RigidObjectCfg:
    """kinematic, collision-free sphere like the other debug markers; spawn pose = card pose + offset."""
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[card_pos[0] + local[0], card_pos[1] + local[1], card_pos[2] + local[2]],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.SphereCfg(
            radius=CARD_EDGE_MARKER_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0),
        ),
    )


@configclass
class ChilsungSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
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

    conveyor = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Conveyor",  # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(

        pos=list(CONVEYOR_POS),  # initial position [x, y, z] -- single source of truth (tag follows)
        rot=[1.0, 0.0, 0.0, 0.0]),  # initial rotation [x, y, z, w]

        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/conveyor_stop.usd",  # table model file
        ),
    )

    # AprilTag fixed to the conveyor's robot-facing side (see constants above).
    conveyor_apriltag = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ConveyorAprilTag",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(CONVEYOR_APRILTAG_POS),
            rot=list(CONVEYOR_APRILTAG_ROT),
        ),
        spawn=UsdFileCfg(usd_path=CONVEYOR_APRILTAG_USD_PATH),
    )

    # 1000장,
    # weight: 0.1kg * 1000 = 100kg,
    # height: 0.3cm * 1000 = 3.0m
    card_box_conveyor = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/card_box_conveyor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-2.00, CONVEYOR_POS[1], 1.05],  # x: +X face at -0.265 so the block held at 90 deg is at jaw contact + 5 mm
                                                  # (measured: block centre x -0.162, needs face at -0.2645); y = conveyor centre line
            rot=[0.5, 0.5, 0.5, 0.5],
        ),
        spawn=UsdFileCfg(
            usd_path=CARD_BOX_USD_PATH,
            scale=(1.0, 1.0, 1000.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.000085,
            ),
        ),
    )

    ###############################################################
    # Debug marker: +X tip face center of card_box_conveyor.
    # Init pose is approximate; sim_main snaps it to the live box tip
    # (box may settle slightly after spawn, so a fixed Z would float high).
    ###############################################################
    card_box_conveyor_local_px_marker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/card_box_conveyor_local_px_marker",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.3, -0.15, 1.05],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.05, 0.05),
                metallic=0.0,
            ),
        ),
    )
    # Purple ball at the geometric centre of the top 35-sheet merged block ``card_block_35``
    # (sim_main.update_card_box_mass_center_marker snaps it there every step; falls back to the bottom
    # loose card ``object`` if the block is absent). card_box_mass_center_pos.txt holds the same centre.
    card_box_mass_center_marker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/card_box_mass_center_marker",
        init_state=RigidObjectCfg.InitialStateCfg(
            #pos=[1.13, -1.2, 1.0],
            pos=[0.9, -1.2, 1.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.SphereCfg(
            # Block half-height = 35 x 3 mm / 2 = 5.25 cm; 8 cm pokes ~2.75 cm out of the stack's top face.
            radius=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.2, 1.0),
                metallic=0.0,
            ),
        ),
    )
    ###############################################################
    # height: 10cm
    # pallete = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/pallete",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[1.55, -1.5, 0.1],
    #         rot=[0.0, 1.0, 0.0, 0.0],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{project_root}/assets/objects/workspace_with_bin/pallet.usd",
    #         scale=(0.03, 0.043, 0.006),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
    #             solver_velocity_iteration_count=4,
    #             max_depenetration_velocity=0.05,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=40),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             contact_offset=0.004,
    #             rest_offset=0.000085,
    #         ),
    #     ),
    # )

    # 40장,
    # weight: 0.1kg * 40 = 4kg,
    # height: 3mm * 40 = 0.12m
    # object = RigidObjectCfg(
    #     prim_path=f"/World/envs/env_.*/card_box_mass",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[1.13, -1.2, 1.0],  # 0.1 + 0.72/2
    #         rot=[1.0, 0, 0, 0],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=CARD_BOX_USD_PATH,
    #         scale=(1.0, 1.0, 36.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             # solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
    #             # solver_velocity_iteration_count=4,
    #             # max_depenetration_velocity=0.05,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=4),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             contact_offset=0.004,
    #             rest_offset=0.000085,
    #         ),
    #     ),
    # )
    # 215장,
    # weight: 0.1kg * 215 = 21.5kg,
    # height: 3mm * 215 = 0.645m
    card_box_mass = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/card_box_mass11",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.8, -1.2, 0.34],  # 0.01 + 0.645/2
            rot=[1, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=CARD_BOX_USD_PATH,
            scale=(1.0, 1.0, 215.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=21.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.000085,
            ),
        ),
    )

    # 300장,
    # weight: 0.1kg * 300 = 30kg,
    # height: 0.3cm * 300 = 0.9m
    card_box_mass12 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/card_box_mass12",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.21, -1.2, 0.46],  # 0.01 + 0.9/2
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=CARD_BOX_USD_PATH,
            scale=(1.0, 1.0, 300.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=30),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.000085,
            ),
        ),
    )

    # card_box_mass21 = RigidObjectCfg(
    #     prim_path=f"/World/envs/env_.*/card_box_mass21",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.8, -1.555, 0.46],  # 0.1 + 0.72/2
    #         rot=[1.0, 0, 0, 0],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=CARD_BOX_USD_PATH,
    #         scale=(1.0, 1.0, 300.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
    #             solver_velocity_iteration_count=4,
    #             max_depenetration_velocity=0.05,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=30),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             contact_offset=0.004,
    #             rest_offset=0.000085,
    #         ),
    #     ),
    # )

    card_box_mass22 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/card_box_mass22",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.21, -1.555, 0.46],  # 0.1 + 0.72/2
            rot=[1.0, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=CARD_BOX_USD_PATH,
            scale=(1.0, 1.0, 300.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=128,  # 위치(position) 솔버의 반복 횟수. 침투·미끄러짐 같은 위치 오차를 잡아내는 횟수
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=30),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.000085,
            ),
        ),
    )

    # ---- card stack: 아래부터 [낱개 10장] - [30장 블록] - [낱개 10장] - [35장 블록] (총 85장)
    # 성능 최적화: 찌르기/집기 영역(아래·중간 밴드)만 낱개 강체로 두고 나머지를 병합 블록으로.
    # CARD_BOX_ASSET_NAMES(object~object9, 아래 10장)는 기존 이름 그대로 유지된다.
    _z = CARD_BOX_BASE_Z - CARD_BOX_THICKNESS / 2   # 스택 바닥면부터 bottom-up으로 적치
    _card_i = 0


    for _ in range(10):                              # 1) 아래 낱개 10장
        locals()[f"object{_card_i}" if _card_i else "object"] = _card_box_cfg(
            f"card_box{_card_i}" if _card_i else "card_box", _z + CARD_BOX_THICKNESS / 2
        )
        _z += CARD_BOX_THICKNESS + CARD_BOX_GAP
        _card_i += 1
    card_block_30 = _card_box_block_cfg("card_block_30", 30, _z)   # 2) 30장 병합 블록
    _z += 30 * CARD_BOX_THICKNESS + CARD_BOX_GAP



    for _ in range(10):                              # 3) 중간 낱개 10장 (object10~object19)
        locals()[f"object{_card_i}"] = _card_box_cfg(f"card_box{_card_i}", _z + CARD_BOX_THICKNESS / 2)
        _z += CARD_BOX_THICKNESS + CARD_BOX_GAP
        _card_i += 1
    card_block_35 = _card_box_block_cfg("card_block_35", 35, _z)   # 4) 맨 위 35장 병합 블록
    _z += 35 * CARD_BOX_THICKNESS + CARD_BOX_GAP
    del _, _z, _card_i



    # four coloured -X edge markers (bottom + middle card), see CARD_EDGE_MARKERS above
    for _name, (_card, _local, _color) in CARD_EDGE_MARKERS.items():
        locals()[_name] = _edge_marker_cfg(_name, locals()[_card].init_state.pos, _local, _color)
    del _name, _card, _local, _color


    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                   pos_offset=(-0.1, 3.6, 1.6),
                                                   rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                   focal_length=16.5)

    # World-frame axis triad at (0, 0, 0): red +X, green +Y, blue +Z, white ball = origin.
    # Visual only (no physics). Hide with env var WORLD_AXES=0. See tasks/common_config/world_axes.py.
    locals().update(world_axes_cfgs())
