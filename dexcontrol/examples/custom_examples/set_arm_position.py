import json
from dexcontrol.robot import Robot

with Robot() as robot:
    with open("poses/Home_position.json", "r") as f:
        poses_dict = json.load(f)
    robot.left_arm.set_joint_pos(poses_dict["left"], wait_time=3.0)
    robot.right_arm.set_joint_pos(poses_dict["right"], wait_time=3.0)
    
    