# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""IK example - Forward kinematics and inverse kinematics demonstration."""

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager
    mm = MotionManager("vega_1")
    print(f"Robot type: {mm.robot_type}")

    current_poses = mm.fk(["L_ee", "R_ee"])
    print(f"Current left pose: {current_poses['L_ee']}")
    print(f"Current left pose type: {type(current_poses['L_ee'])}")
    print("Current end-effector positions:")
    for ee_name, pose in current_poses.items():
        if pose and hasattr(pose, "translation"):
            print(f"  {ee_name}: {pose.translation}")

    # Set relative movement (small distances)
    relative_movements = {
        "L_ee": [0.05, 0.0, 0.02],  # Move 5cm forward, 2cm up
        "R_ee": [0.05, 0.0, 0.02],  # Move 5cm forward, 2cm up
    }

    print(f"\nRelative movements: {relative_movements}")

    # Use IK with relative movement
    solution, collision_status, limits_status = mm.ik(
        target_pos=relative_movements, relative=True, type="pink"
    )

    print(f"IK solution found: {solution is not None}")
    print(f"Collision status: {collision_status}")
    print(f"Within limits: {limits_status}")

    if solution is not None:
        # Apply the IK solution
        mm.set_joint_pos(solution)
        # Get updated end-effector positions using forward kinematics
        updated_poses = mm.fk(["L_ee", "R_ee"])

        print("\nUpdated end-effector positions:")
        for ee_name, pose in updated_poses.items():
            if pose and hasattr(pose, "translation"):
                print(f"  {ee_name}: {pose.translation}")

    while True:
        pass  # In order to keep the script running


if __name__ == "__main__":
    main()
