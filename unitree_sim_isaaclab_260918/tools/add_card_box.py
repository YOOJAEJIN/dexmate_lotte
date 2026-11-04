# add_card_box.py — card_box.usd에 고마찰 물리 재질을 심는 1회 실행 스크립트
# 실행: bash tools/add_card_box.sh  (isaacsim5_1 python.sh에는 pxr 패스가 없어서 래퍼 사용)
import os

from pxr import Usd, UsdShade, UsdPhysics, PhysxSchema

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USD_PATH = f"{PROJECT_ROOT}/assets/objects/workspace_with_chilsung/card_box.usd"
MAT_PATH = "/ChilsungBox/_materials/CARD_PHYSICS"

STATIC_FRICTION = 1.5   # 날카로운 팁으로 스택을 찔러 들어올리기 위한 고마찰
DYNAMIC_FRICTION = 1.0
RESTITUTION = 0.0

stage = Usd.Stage.Open(USD_PATH)
if stage is None:
    raise SystemExit(f"fail: cannot open {USD_PATH}")

mat = UsdShade.Material.Define(stage, MAT_PATH)
mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
mapi.CreateStaticFrictionAttr(STATIC_FRICTION)
mapi.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
mapi.CreateRestitutionAttr(RESTITUTION)
# 그리퍼 쪽이 기본 마찰(~0.5)이어도 접촉에서 카드 값이 채택되도록 combine=max
PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")

bound = 0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        # 이 빌드에는 UsdPhysics.BindingAPI가 없어 UsdShade 바인딩 사용
        # (물리 재질도 동일한 material:binding 릴레이션으로 PhysX가 인식함)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        bound += 1

stage.Save()
print(f"ok: material {MAT_PATH} (static={STATIC_FRICTION}, dynamic={DYNAMIC_FRICTION}, "
      f"restitution={RESTITUTION}, combine=max) bound to {bound} collision prims")
