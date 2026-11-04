# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Basic dexmotion example - vega_1 robot with joint position setting."""

import time

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager with vega_1 robot
    initial_config = {
        "L_arm_j1": 0.064,
        "L_arm_j7": 0.03,
        "R_arm_j1": -0.064,
        # "Lift": 1.0,
        # "torso_j1": 0.42,
        # "torso_j2": 0.14,
        # "torso_j3": -0.6,
        "head_j1": 0.4,
    }
    mm = MotionManager(
        initial_joint_configuration_dict=initial_config,
        joint_regions_to_lock=[
            "HEAD",
            "BASE",
            "TORSO",
        ],  # which is also the default
    )

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

    # Full robot joint positions setting

    print(f"\nSetting new joint positions: {new_joint_positions}")
    mm.set_joint_pos(new_joint_positions)

    # Print updated joint positions
    updated_joint_pos_dict = mm.get_joint_pos_dict()
    print(f"Updated joint positions: {updated_joint_pos_dict}")

    time.sleep(5)

    # Component level joint positions setting
    mm.left_arm.set_joint_pos([1.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    mm.left_hand.set_joint_pos([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mm.right_arm.set_joint_pos([-2.5, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    mm.torso.set_joint_pos([0.42, 0.14, -0.6])
    mm.head.set_joint_pos([0.4, 0.0, 0.0])

    # Print updated joint positions
    updated_left_arm_pos = mm.left_arm.get_joint_pos()
    updated_left_hand_pos = mm.left_hand.get_joint_pos()
    updated_right_arm_pos = mm.right_arm.get_joint_pos()
    updated_torso_pos = mm.torso.get_joint_pos()
    updated_head_pos = mm.head.get_joint_pos()
    print(f"Updated left arm positions: {updated_left_arm_pos}")
    print(f"Updated left hand positions: {updated_left_hand_pos}")
    print(f"Updated right arm positions: {updated_right_arm_pos}")
    print(f"Updated torso positions: {updated_torso_pos}")
    print(f"Updated head positions: {updated_head_pos}")

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
