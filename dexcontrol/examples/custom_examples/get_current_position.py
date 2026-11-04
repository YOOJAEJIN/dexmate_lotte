import json
from dexcontrol.robot import Robot

with Robot() as robot:
    # 포즈 저장
    left_pose = robot.left_arm.get_joint_pos()
    right_pose = robot.right_arm.get_joint_pos()
    poses_dict = {"left": left_pose.tolist(), "right": right_pose.tolist()}
    
    with open("poses/L_shape.json", "w") as f:
        json.dump(poses_dict, f)
    