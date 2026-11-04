# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Global IK example - Forward kinematics and global inverse
kinematics with CasADi solver.

This example demonstrates the use of the global IK solver
with CasADi.
"""

import dexmotion.utils.robot_utils as robot_utils
from dexmotion.configs.ik import GlobalCasadiJointIKConfig
from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager
    mm = MotionManager(
        "vega_1", joint_regions_to_lock=["BASE", "HEAD", "LEFT_HAND", "RIGHT_HAND"]
    )

    # Set target positions and orientations
    target_positions = {
        "L_ee": [0.4, 0.2, 1.0],  # Left end-effector position
        "R_ee": [0.4, -0.2, 1.5],  # Right end-effector position
    }

    target_orientations = {
        "L_ee": [0.0, 0.0, 0.0],  # No rotation [roll, pitch, yaw]
        "R_ee": [0.0, 0.0, 0.0],  # No rotation
    }

    # Custom config to prioritize position accuracy
    custom_config = GlobalCasadiJointIKConfig(
        torso_penalty=0.05,
    )

    # Solve Global IK with CasADi solver
    target_configuration_dict, is_in_collision, is_joints_within_limits = mm.global_ik(
        target_pos=target_positions,
        target_rot=target_orientations,  # Works with RPY angles automatically!
        type="casadi",
        custom_config=custom_config,
    )

    if target_configuration_dict is not None and mm.pin_robot is not None:
        # Apply the solution
        solution = robot_utils.get_qpos_from_joint_dict(
            mm.pin_robot, target_configuration_dict
        )
        updated_poses = mm.fk(["L_ee", "R_ee"], qpos=solution)

        print("\nUpdated end-effector positions:")
        for ee_name, pose in updated_poses.items():
            if pose and hasattr(pose, "translation"):
                print(f"  {ee_name}: {pose.translation}")
        mm.set_joint_pos(solution)
        while True:
            pass


if __name__ == "__main__":
    main()
