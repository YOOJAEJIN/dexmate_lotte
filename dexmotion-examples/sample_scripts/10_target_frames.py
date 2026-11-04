# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""IK example - Forward kinematics and inverse kinematics demonstration."""

from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkIKConfig
from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager
    mm = MotionManager(
        robot_type="vega_1",
        joint_regions_to_lock=["HEAD", "RIGHT_HAND", "LEFT_HAND", "BASE"],
    )

    current_poses = mm.fk(["head_l3"])

    print("Current end-effector positions:")
    for ee_name, pose in current_poses.items():
        if pose and hasattr(pose, "translation"):
            print(f"  {ee_name}: {pose.translation}")

    # Set relative movement (small distances)
    relative_movements = {
        "head_l3": [0.0, 0.0, 0.5],  # Move 5cm forward, 2cm up
    }

    print(f"\nRelative movements: {relative_movements}")

    # Create custom damping configuration to allow torso movement
    custom_damping = IKDampingWeightsConfig(
        default=1e-12,  # Default damping for other joints
        override={
            "torso_j1": 1e-20,  # Much lower damping for torso
            "torso_j2": 1e-20,  # Lower damping for torso_j2
            "torso_j3": 1e-20,  # Lower damping for torso_j3
        },
    )

    # Create custom IK configuration with the damping weights
    custom_ik_config = LocalPinkIKConfig(
        damping_weights=custom_damping,
        target_frames=["head_l3"],
    )

    print("\nCustom damping configuration:")
    print(f"  Torso joints damping: {custom_damping.override['torso_j1']}")

    # Use IK with relative movement and custom configuration
    solution, collision_status, limits_status = mm.ik(
        target_pos=relative_movements,
        relative=True,
        type="pink",
        custom_config=custom_ik_config,
    )

    print(f"IK solution found: {solution is not None}")
    print(f"Collision status: {collision_status}")
    print(f"Within limits: {limits_status}")

    if solution is not None:
        # Apply the IK solution
        mm.set_joint_pos(solution)
        # Get updated end-effector positions using forward kinematics
        updated_poses = mm.fk(["head_l3"])

        print("\nUpdated end-effector positions:")
        for ee_name, pose in updated_poses.items():
            if pose and hasattr(pose, "translation"):
                print(f"  {ee_name}: {pose.translation}")

    while True:
        pass  # In order to keep the script running


if __name__ == "__main__":
    main()
