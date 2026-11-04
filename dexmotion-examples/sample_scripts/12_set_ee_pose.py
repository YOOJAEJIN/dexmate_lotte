# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Basic dexmotion example - vega_1 robot with end-effector pose setting."""

import time

import numpy as np

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager with vega_1 robot
    mm = MotionManager(
        joint_regions_to_lock=["HEAD", "BASE", "RIGHT_HAND", "LEFT_HAND"]
    )

    # set right end-effector to a specific position and orientation
    right_target_pos = np.array([0.0, 0.0, 0.3])  # x, y, z in meters
    right_target_rot = np.array([0.0, np.pi / 4, 0.0])  # roll, pitch, yaw in radians
    input("Press Enter to set right end-effector to a specific pos and rot")
    mm.right_arm.set_ee_pose(
        right_target_pos, right_target_rot, target_frame="R_arm_j4", relative=True
    )

    input("Press Enter to move left end-effector by 0.5 meters in z direction")
    mm.left_arm.move_ee_xyz(np.array([0.0, 0.0, 0.5]))

    input("Press Enter to rotate right end-effector by 45 degrees around x-axis")
    mm.right_arm.move_ee_rpy(np.array([np.pi / 4, 0.0, 0.0]), target_frame="R_arm_j4")

    input("Press Enter to move torso end-effector by 1.5 meters in z direction")
    mm.torso.move_ee_xz(np.array([0.0, 0.0, 1.5]))

    input("Press Enter to move torso end-effector pitch by 45 degrees")
    mm.torso.move_ee_pitch(np.pi / 4)

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
