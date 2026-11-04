# /home/ubuntu/dx/pantos_tele/.venv/bin/python ./src/omniteleop/reach/test_reach_planner.py
# pinocchio 공식 튜토리얼(https: // github.com / stack - of - tasks / pinocchio의 examples) 중 "Forward kinematics" 참고

import sys; sys.path.insert(0, '/home/ubuntu/dex/omniteleop/src/omniteleop/reach')
import reach_planner as rp, pinocchio as pin, numpy as np

robot = rp._get_robot('/home/ubuntu/dex/omniteleop/external/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf')
m, d = robot.model, robot.data

# 실험 1: 관절 하나를 바꾸면 EE가 어디로 가나
print(m.getFrameId('R_ee'))
print(d.oMf[m.getFrameId('R_ee')])
print(d.oMf[m.getFrameId('R_ee')].translation.round(3)) # 초기값은 [0. 0. 0.]

q = rp.build_q(m, {'torso':[0,0,0], 'head':[0,0,0]})
for j1 in (-0.5, 0.0, 0.5):
    q[m.idx_qs[m.getJointId('R_arm_j1')]] = j1
    pin.framesForwardKinematics(m, d, q)   # q로 모든 프레임 포즈 계산
    print(j1, d.oMf[m.getFrameId('R_ee')].translation.round(3))   # 특정 프레임의 포즈 (SE3)

# 실험 2: 관절 목록/프레임 목록 훑어보기
print([m.names[i] for i in range(m.njoints)])
print([m.frames[i].name for i in range(m.nframes)])