import omni.kit.commands
from pxr import Usd, UsdPhysics, PhysxSchema

# --- 설정값 ---
ROBOT_PRIM_PATH = "/vega_1p_no"  # 로봇이 임포트된 경로 (Stage 창에서 확인 가능)
# TARGET_STIFFNESS = 1000.0  # 적용할 Stiffness (Kp)
# TARGET_DAMPING = 100.0  # 적용할 Damping (Kd)
# --------------

stage = omni.usd.get_context().get_stage()

# 로봇 프림 아래의 모든 자식 프림을 순회
robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
if not robot_prim:
    print(f"Error: {ROBOT_PRIM_PATH} 경로를 찾을 수 없습니다.")
else:
    for prim in Usd.PrimRange(robot_prim):
        # 해당 프림이 조인트인지 확인 (RevoluteJoint, PrismaticJoint 등)
        if prim.IsA(UsdPhysics.Joint):
            joint_path = prim.GetPath()

            # PhysxJointDriveAPI를 사용하여 조인트 드라이브 설정
            # 보통 URDF 임포트 시 'angular' 또는 'linear' 드라이브가 생성됨
            # 여기서는 회전형 조인트(Revolute) 기준으로 'angular' 드라이브를 수정
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive_api:
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")

            # 값 적용
            joint_name = prim.GetName()
            if "torso_j1" in joint_name:
                drive_api.GetStiffnessAttr().Set(5200.0)
                drive_api.GetDampingAttr().Set(520.0)
            if "torso_j2" in joint_name:
                drive_api.GetStiffnessAttr().Set(3800.0)
                drive_api.GetDampingAttr().Set(380.0)
            if "torso_j3" in joint_name:
                drive_api.GetStiffnessAttr().Set(4900.0)
                drive_api.GetDampingAttr().Set(490.0)
            if "head" in joint_name:
                drive_api.GetStiffnessAttr().Set(300.0)
                drive_api.GetDampingAttr().Set(60.0)
            if "arm_j1" in joint_name:
                drive_api.GetStiffnessAttr().Set(1150.0)
                drive_api.GetDampingAttr().Set(115.0)
            if "arm_j2" in joint_name:
                drive_api.GetStiffnessAttr().Set(900.0)
                drive_api.GetDampingAttr().Set(90.0)
            if "arm_j3" in joint_name:
                drive_api.GetStiffnessAttr().Set(740.0)
                drive_api.GetDampingAttr().Set(74.0)
            if "arm_j4" in joint_name:
                drive_api.GetStiffnessAttr().Set(340.0)
                drive_api.GetDampingAttr().Set(34.0)
            if "arm_j5" in joint_name:
                drive_api.GetStiffnessAttr().Set(174.0)
                drive_api.GetDampingAttr().Set(18.0)
            if "arm_j6" in joint_name:
                drive_api.GetStiffnessAttr().Set(120.0)
                drive_api.GetDampingAttr().Set(12.0)
            if "arm_j7" in joint_name:
                drive_api.GetStiffnessAttr().Set(80.0)
                drive_api.GetDampingAttr().Set(8.0)
            if "wheel_j1" in joint_name:
                drive_api.GetStiffnessAttr().Set(1500.0)
                drive_api.GetDampingAttr().Set(150.0)
            if "wheel_j2" in joint_name:
                drive_api.GetStiffnessAttr().Set(0.0)
                drive_api.GetDampingAttr().Set(2.0)

print("모든 조인트 설정이 완료되었습니다.")