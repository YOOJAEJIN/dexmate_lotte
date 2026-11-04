import numpy as np
import omni.usd
from pxr import UsdPhysics
from isaacsim.core.simulation_manager import SimulationManager

sim_view = SimulationManager.get_physics_sim_view()
stage = omni.usd.get_context().get_stage()

def find_rigid_body_path(keyword):
    """stage 전체에서 이름에 keyword가 포함된 프림을 찾아
    rigid body API가 적용된 실제 프림 경로를 반환 (USD 참조로 스폰된 경우
    rigid body 루트가 하위 프림에 있을 수 있어 조상까지 함께 확인)"""
    for prim in stage.Traverse():
        if keyword not in str(prim.GetPath()).lower():
            continue
        p = prim
        while p and p.IsValid():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(p.GetPath())
            p = p.GetParent()
    return None

def live_pose(path):
    view = sim_view.create_rigid_body_view(path)
    if hasattr(view, "get_root_transforms"):   # torch backend
        t = view.get_root_transforms()
    else:                                      # numpy backend (CPU 실행 시)
        t = view.get_transforms()
    return np.asarray(t).reshape(-1, 7)[0]     # x,y,z,qw,qx,qy,qz

def quat_to_rpy(qw, qx, qy, qz):
    """쿼터니언(w,x,y,z) -> 오일러 각 roll,pitch,yaw (라디안)"""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

path = find_rigid_body_path("card_box")
if path is None:
    print("card_box 프림(또는 rigid body)을 찾지 못했습니다. 아래는 현재 stage의 card 관련 프림들입니다:")
    for prim in stage.Traverse():
        if "card" in str(prim.GetPath()).lower():
            print("  ", prim.GetPath(), "| RigidBody:", prim.HasAPI(UsdPhysics.RigidBodyAPI))
else:
    print(f"rigid body 경로 = {path}")
    x, y, z, qw, qx, qy, qz = live_pose(path)
    roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)

    print(f"card_box 위치   x = {x:.4f} m, y = {y:.4f} m, z = {z:.4f} m")
    print(f"card_box 쿼터니언 qw = {qw:.4f}, qx = {qx:.4f}, qy = {qy:.4f}, qz = {qz:.4f}")
    print(f"card_box 오일러 roll = {np.degrees(roll):.2f}°, pitch = {np.degrees(pitch):.2f}°, yaw = {np.degrees(yaw):.2f}°")
