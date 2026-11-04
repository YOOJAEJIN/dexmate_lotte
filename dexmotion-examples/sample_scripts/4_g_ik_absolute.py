# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Global IK example - Forward kinematics and global
inverse kinematics with CasADi solver.

This example demonstrates the use of the global IK solver
with CasADi.
"""

import dexmotion.utils.robot_utils as robot_utils
from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager
    mm = MotionManager("vega_1")

    # Set target positions and orientations
    target_positions = {
        "L_ee": [0.4, 0.2, 1.0],  # Left end-effector position
        "R_ee": [0.4, -0.2, 1.0],  # Right end-effector position
    }

    target_orientations = {
        "L_ee": [1.0, 0.0, 0.0, 0.0],  # Quaternion [w, x, y, z]
        "R_ee": [1.0, 0.0, 0.0, 0.0],  # Quaternion [w, x, y, z]
    }

    # Solve Global IK with CasADi solver
    target_configuration_dict, is_in_collision, is_joints_within_limits = mm.global_ik(
        target_pos=target_positions,
        target_rot=target_orientations,  # Works with quaternions automatically!
        type="hybrid",  # "pink" or "casadi" works too
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
