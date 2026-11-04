# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Simple motion planning example - Plan to goal configuration with visualization."""

import time

from dexmotion.motion_manager import MotionManager


def main():
    # Initialize MotionManager with planning and visualization enabled
    mm = MotionManager("vega_1")

    # Get current joint configuration as starting point
    start_config = mm.get_joint_pos_dict()
    print(f"Start configuration: {start_config}")

    # Define a goal configuration - move both arms
    goal_config = start_config.copy()
    goal_config.update(
        {
            "L_arm_j1": 0.5,  # Left arm shoulder pitch
            "L_arm_j2": 0.3,  # Left arm shoulder roll
            "R_arm_j1": -0.5,  # Right arm shoulder pitch
            "R_arm_j2": -0.3,  # Right arm shoulder roll
        }
    )

    print(f"Goal configuration: {goal_config}")

    # Plan motion from start to goal using OMPL planner
    print("\nPlanning motion...")
    waypoints = mm.plan(
        goal_config=goal_config,
        planner_type="ompl",  # 'ompl' or 'interpolation'
    )

    if waypoints is not None and len(waypoints) > 0:
        print(f"Planning successful! Generated {len(waypoints)} waypoints")
        mm.visualize_motion_plan(waypoints, duration=3.0)

        # Keep the script running to view the visualization
        print("\nViz is running. Open your web browser to view the motion plan.")
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
