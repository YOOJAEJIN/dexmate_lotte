#!/usr/bin/env python3
"""로봇 팔 끝에 그리퍼를 고정 조인트로 직접 부착하는 수리 스크립트.

Isaac Sim 5.x Robot Assembler의 Finish 단계는 그리퍼 서브트리를 루트 레이어에서
잘라내 별도 파일(variant payload 구조)로 옮기는데, 이 과정에서
  1) variant selection이 'None'으로 저장되고
  2) payload 상대경로가 깨지며
  3) instancing 프로토타입(/Flattened_Prototype_*)이 따라가지 못해
그리퍼가 사라지는 문제가 발생한다.

이 스크립트는 원본 USD를 복사한 뒤 다음을 인라인으로 적용해 동일한 조립 결과를
복원한다 (Robot Assembler가 미리보기에서 계산했던 값과 동일):
  - 그리퍼 Xform을 암 엔드프레임(attach point)의 world 포즈로 이동
  - PhysicsFixedJoint(AssemblerFixedJoint) 생성: body0=암 링크, body1=그리퍼 base_link
  - 그리퍼 base_link의 ArticulationRootAPI 제거 (로봇 articulation에 병합)

사용 예:
  python3 tools/assemble_gripper.py \
      --src assets/robots/vega-1p-usd/vega_1p_vertical.usd \
      --dst assets/robots/vega-1p-usd/vega_1p_vertical_L_gripper_assembled.usd \
      --gripper /vega_1p_no/L_gripper --arm /vega_1p_no/L_arm_l8
"""

import argparse
import os
import shutil
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def set_orient(prim: Usd.Prim, quat: Gf.Quatd) -> None:
    """xformOp:orient 값을 프림의 기존 속성 타입(quatd/quatf)에 맞춰 설정한다."""
    op = None
    for candidate in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op = candidate
            break
    if op is None:
        op = UsdGeom.Xformable(prim).AddQuaternionOp()
    attr = op.GetAttr()
    if attr.GetTypeName() == "quatf":
        q = Gf.Quatf(quat.real, Gf.Vec3f(*quat.imaginary))
    else:
        q = quat
    attr.Set(q)


def set_translate(prim: Usd.Prim, t: Gf.Vec3d) -> None:
    op = None
    for candidate in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op = candidate
            break
    if op is None:
        op = UsdGeom.Xformable(prim).AddTranslateOp()
    attr = op.GetAttr()
    attr.Set(Gf.Vec3f(*t) if attr.GetTypeName() == "float3" else t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="원본(무결한) 로봇 USD")
    ap.add_argument("--dst", required=True, help="조립 결과를 저장할 USD 경로")
    ap.add_argument("--gripper", required=True, help="그리퍼 프림 경로 (예: /vega_1p_no/L_gripper)")
    ap.add_argument("--arm", required=True, help="부착 대상 암 링크 경로 (예: /vega_1p_no/L_arm_l8)")
    ap.add_argument("--mount-link", default="base_link", help="그리퍼 측 마운트 링크 이름 (기본: base_link)")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if src == dst:
        print("--src와 --dst가 같습니다. 원본을 보존하려면 다른 경로를 지정하세요.", file=sys.stderr)
        return 1
    if not os.path.isfile(src):
        print(f"원본 파일이 없습니다: {src}", file=sys.stderr)
        return 1

    # 1) 원본 보존을 위해 복사 후 복사본을 편집한다 (Save가 composition을 flatten하지 않도록).
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    stage = Usd.Stage.Open(dst)

    gripper = stage.GetPrimAtPath(args.gripper)
    arm = stage.GetPrimAtPath(args.arm)
    mount = stage.GetPrimAtPath(f"{args.gripper}/{args.mount_link}")
    for label, prim in (("gripper", gripper), ("arm", arm), ("mount", mount)):
        if not prim.IsValid():
            print(f"{label} 프림을 찾을 수 없습니다.", file=sys.stderr)
            return 1
    if not arm.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"암 링크가 rigid body가 아닙니다: {arm.GetPath()}", file=sys.stderr)
        return 1

    # 2) 그리퍼를 암 엔드프레임 world 포즈로 이동 (부모 로컬 공간으로 환산).
    xfc = UsdGeom.XformCache()
    arm_world = xfc.GetLocalToWorldTransform(arm)
    parent_world = xfc.GetLocalToWorldTransform(gripper.GetParent())
    local = parent_world.GetInverse() * arm_world
    set_translate(gripper, local.ExtractTranslation())
    set_orient(gripper, local.ExtractRotationQuat())

    # 3) 그리퍼 articulation root를 제거해 로봇 articulation에 병합될 수 있게 한다.
    if mount.HasAPI(UsdPhysics.ArticulationRootAPI):
        mount.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print(f"제거: ArticulationRootAPI @ {mount.GetPath()}")
    else:
        print(f"경고: {mount.GetPath()}에 ArticulationRootAPI가 없습니다 (이미 병합됨?).")

    # 4) 고정 조인트 생성 (Robot Assembler가 만들던 것과 동일한 이름/파라미터).
    joint_path = f"{mount.GetPath()}/AssemblerFixedJoint"
    old = stage.GetPrimAtPath(joint_path)
    if old.IsValid():
        stage.RemovePrim(old.GetPath())
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([arm.GetPath()])
    joint.GetBody1Rel().SetTargets([mount.GetPath()])
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(0, -2.7755576e-17, 0))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, -5.551115e-17, 0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    print(f"생성: FixedJoint @ {joint_path} (body0={arm.GetPath()}, body1={mount.GetPath()})")

    stage.GetRootLayer().Save()
    print(f"저장 완료: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
