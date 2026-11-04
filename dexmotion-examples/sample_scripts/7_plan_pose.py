# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Simple motion planning example - Plan to goal poses with visualization."""

import time

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager with planning and visualization enabled
    mm = MotionManager("vega_1")

    # Get current joint configuration as starting point
    start_config = mm.get_joint_pos_dict()
    print(f"Start configuration: {start_config}")

    # Define goal positions and orientations using the new interface
    goal_positions = {
        "L_ee": [0.1, 0.0, 0.1],  # Move left end-effector forward and up (relative)
        "R_ee": [0.1, 0.0, 0.1],  # Move right end-effector forward and up (relative)
    }

    goal_orientations = {"L_ee": [3.14, 0.0, 0.0], "R_ee": [0.0, 0.0, 3.14]}

    print(f"\nGoal positions (relative): {goal_positions}")
    print(f"Goal orientations: {goal_orientations}")

    # Plan motion using the new interface with relative positioning
    print("\nPlanning motion...")
    waypoints = mm.plan(
        goal_pos=goal_positions,
        target_rot=goal_orientations,
        relative=True,  # Apply changes relative to current poses
        planner_type="interpolation",  # 'ompl' or 'interpolation'
    )

    if waypoints is not None and len(waypoints) > 0:
        print(f"Planning successful! Generated {len(waypoints)} waypoints")
        mm.visualize_motion_plan(waypoints, duration=3.0)

        # Keep the script running to view the visualization
        print("\Viz is running. Open your web browser to view the motion plan.")
        print("Press Ctrl+C to exit...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")

    else:
        print("Planning failed - no valid path found")


if __name__ == "__main__":
    main()
