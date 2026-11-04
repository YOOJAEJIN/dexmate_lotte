# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Advanced motion planning example -
    Plan to target poses with custom OMPL configuration.

This example demonstrates the use of the motion planner with
a custom OMPL configuration, including
"""

from dexmotion.configs.planning.planning_config import OMPLPlannerConfig
from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager with planning and visualization enabled
    mm = MotionManager(
        joint_regions_to_lock=["BASE", "HEAD", "LEFT_HAND", "RIGHT_HAND"],
    )
    # Get current joint configuration as starting point
    start_config = mm.get_joint_pos_dict()
    print(f"Start configuration: {start_config}")

    # Get current end-effector poses
    if mm.pin_robot is None:
        print("Error: Robot not initialized")
        return
    current_poses = mm.fk(["L_ee", "R_ee"])
    print("\nCurrent end-effector poses:")
    for ee_name, pose in current_poses.items():
        if pose and hasattr(pose, "translation"):
            print(f"  {ee_name}: position = {pose.translation}")

    # Define target poses by applying offsets to current poses
    target_poses = {}
    for ee_name, current_pose in current_poses.items():
        if (
            current_pose
            and hasattr(current_pose, "translation")
            and current_pose.translation is not None
        ):
            # Copy current pose and apply offset
            target_pose = current_pose.copy()
            if target_pose.translation is not None:
                if ee_name == "L_ee":
                    # Move left end-effector: forward, outward, and up
                    target_pose.translation[0] += 0.2  # 20cm forward
                    target_pose.translation[1] += 0.1  # 10cm outward (left)
                    target_pose.translation[2] += 0.15  # 15cm up
                elif ee_name == "R_ee":
                    # Move right end-effector: forward, outward, and up
                    target_pose.translation[0] += 0.2  # 20cm forward
                    target_pose.translation[1] -= 0.1  # 10cm outward (right)
                    target_pose.translation[2] += 0.15  # 15cm up
                target_poses[ee_name] = target_pose

    print("\nTarget poses (with offsets):")
    for ee_name, pose in target_poses.items():
        if pose and hasattr(pose, "translation"):
            print(f"  {ee_name}: position = {pose.translation}")

    # Create custom OMPL configuration for more sophisticated planning
    custom_ompl_config = OMPLPlannerConfig(
        planner_name="rrtconnect",
    )

    waypoints = mm.plan(
        goal_pose=target_poses,  # Using SE3 poses directly
        start_config=start_config,
        planner_type="ompl",
        ompl_config=custom_ompl_config,  # Pass custom OMPL configuration
    )

    if waypoints is not None and len(waypoints) > 0:
        print(f"Planning successful! Generated {len(waypoints)} waypoints")

        # Visualize the motion plan with longer duration for complex path
        print("Visualizing motion plan...")
        mm.visualize_motion_plan(waypoints, duration=5.0)
        while True:
            pass

    else:
        print("Planning failed - no valid path found")


if __name__ == "__main__":
    main()
