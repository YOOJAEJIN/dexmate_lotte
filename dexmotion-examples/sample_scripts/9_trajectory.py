# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Trajectory smoothing example - Plan motion and apply Gaussian smoothing."""

import time

import numpy as np

import dexmotion.utils.robot_utils as robot_utils
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

    if isinstance(waypoints[0], dict):
        # Convert dict waypoints to qpos arrays
        if mm.pin_robot is None:
            print("Error: Robot not initialized")
            return
        waypoints_qpos = np.array(
            [
                robot_utils.get_qpos_from_joint_dict(mm.pin_robot, waypoint)
                for waypoint in waypoints
            ]
        )
    else:
        waypoints_qpos = np.array(waypoints)

    print(f"Original waypoints shape: {waypoints_qpos.shape}")

    # Apply Gaussian trajectory smoothing
    print("\nApplying Gaussian trajectory smoothing...")
    method = "gaussian"
    smooth_trajectory = mm.smooth_trajectory(
        waypoints=waypoints_qpos,
        method=method,  # gaussian, toppra, ruckig
        control_frequency=100.0,
    )

    if smooth_trajectory is not None:
        # Extract data from trajectory dictionary
        qs_sample = smooth_trajectory["positions"]
        qds_sample = smooth_trajectory.get("velocities")
        duration = len(qs_sample) / 100.0  # Calculate duration from number of samples

        # Visualize the smoothed trajectory
        print("\nVisualizing smoothed motion plan...")
        mm.visualize_motion_plan(qs_sample, duration=duration)

        # Show trajectory statistics
        if qds_sample is not None:
            max_velocity = np.max(np.abs(qds_sample))
            print(f"  Max velocity: {max_velocity:.3f} rad/s")
    else:
        print("Trajectory smoothing failed, using original waypoints")
        mm.visualize_motion_plan(waypoints, duration=3.0)

    # Keep the script running to view the visualization
    print("\nVisualization is running. Open your web browser to view the motion plan.")
    print(f"The trajectory has been smoothed using {method} for natural motion.")
    print("Press Ctrl+C to exit...")

    while True:
        time.sleep(0.001)


if __name__ == "__main__":
    main()
