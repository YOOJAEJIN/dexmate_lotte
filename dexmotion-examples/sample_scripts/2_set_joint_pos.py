# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Basic dexmotion example - vega_1 robot with joint position setting."""

import time

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager with vega_1 robot
    mm = MotionManager("vega_1")

    # Print robot type
    print(f"Robot type: {mm.robot_type}")

    # Print initial joint positions
    joint_pos_dict = mm.get_joint_pos_dict()
    print(f"Number of active joints: {len(joint_pos_dict)}")
    print(f"Initial joint positions: {joint_pos_dict}")

    # Set new joint positions
    new_joint_positions = {
        "L_arm_j1": 0.5,
        "R_arm_j1": -0.5,
        "L_arm_j2": 0.3,
        "R_arm_j2": -0.3,
    }

    print(f"\nSetting new joint positions: {new_joint_positions}")
    mm.set_joint_pos(new_joint_positions)

    # Print updated joint positions
    updated_joint_pos_dict = mm.get_joint_pos_dict()
    print(f"Updated joint positions: {updated_joint_pos_dict}")

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
