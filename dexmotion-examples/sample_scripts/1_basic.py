# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Basic dexmotion example - Initialize MotionManager and print robot information."""

import time

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager
    mm = MotionManager(visualizer_type="sapien")

    # Print robot type
    print(f"Robot type: {mm.robot_type}")

    # Print joint positions
    joint_pos_dict = mm.get_joint_pos_dict()
    print(f"Number of active joints: {len(joint_pos_dict)}")
    print(f"Joint positions: {joint_pos_dict}")

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
