# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""Simple motion planning with obstacle avoidance - Plan to goal poses.

This example demonstrates how to use the interpolation planner with obstacle
avoidance by configuring LocalPinkExtIKConfig with external obstacle info.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkExtIKConfig
from dexmotion.configs.planning import InterpolationPlannerConfig
from dexmotion.motion_manager import MotionManager


def main():
    # Configure external obstacles using dictionary format (cleaner than parallel lists).
    # Replace the placeholder path below with the path to your own URDF asset.
    obstacles = [
        {
            "urdf_path": "path/to/your/obstacle_1.urdf",
            "position": [1.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "name": "obstacle_1",
        }
    ]

    damping_weights = IKDampingWeightsConfig(
        default=1e-8,
        override={
            "torso_j1": 5000,
            "torso_j2": 5000,
            "torso_j3": 5000,
        },
    )

    # Create custom pink_ext IK config with dictionary-based obstacles
    pink_ext_config = LocalPinkExtIKConfig(
        external_obstacles=obstacles,
        damping_weights=damping_weights,
    )

    # Create interpolation planner config with pink_ext IK and custom obstacle config
    interp_config = InterpolationPlannerConfig(
        ik_type="pink_ext",  # Use pink_ext for obstacle avoidance
        custom_ik_config=pink_ext_config,  # Pass custom config with obstacle
        max_pos_interval=0.03,
        max_rot_interval=0.05,
    )

    initial_config = {
        "torso_j1": 0.52,
        "torso_j2": 1.05,
        "torso_j3": 0.37,
        "L_arm_j4": -1.37,
        "L_arm_j6": 1.0,
        "R_arm_j4": -1.37,
        "R_arm_j6": -1.0,
    }

    # Initialize MotionManager with planning and visualization enabled
    # Pass the custom pink_ext config with obstacles to the local IK solver
    mm = MotionManager(
        "vega_1",
        joint_regions_to_lock=["HEAD", "BASE"],
        init_visualizer=True,
        visualizer_type="sapien",
        initial_joint_configuration_dict=initial_config,
        custom_local_ik_config=pink_ext_config,  # Pass config with obstacles
        custom_urdf_path=Path("path/to/your/robot.urdf"),
        custom_srdf_path=Path("path/to/your/robot.srdf"),
    )
    visualizer = mm.visualizer
    assert visualizer is not None, "Visualizer should be initialized"

    # Add the obstacle objects to visualizer (loop through all obstacles).
    # Replace with the path to your own visual mesh.
    obstacle_glb_path = "path/to/your/obstacle_1.glb"
    obstacle_objects = []
    for obstacle in obstacles:
        obstacle_obj = visualizer.add_object(
            obstacle_glb_path,
            pos=np.array(obstacle["position"]),
            quat=np.array(obstacle["orientation"]),
            scale=np.array([1, 1, 1]),
        )
        obstacle_objects.append(obstacle_obj)

    # Get current joint configuration as starting point
    start_config = mm.get_joint_pos_dict()
    print(f"Start configuration: {start_config}")

    # Define goal positions and orientations (same as example 13)
    left_ee_target_position = [0.6, 0.35, 1.08]
    right_ee_target_position = [0.8, -0.25, 1.08]
    # Define target orientations using RPY (Roll, Pitch, Yaw) angles
    # Left arm: 90° roll, 180° pitch, 0° yaw
    left_ee_target_orientation = (
        R.from_euler("xyz", [np.pi / 2, np.pi, 0.0]).as_quat().tolist()
    )
    # Right arm: -90° roll, 180° pitch, 0° yaw
    right_ee_target_orientation = (
        R.from_euler("xyz", [-np.pi / 2, np.pi, 0.0]).as_quat().tolist()
    )

    # Rotate ee_targets by 90 degrees around z axis
    left_ee_target_correction = R.from_euler("x", np.pi / 2).as_quat().tolist()
    right_ee_target_correction = R.from_euler("x", np.pi / 2).as_quat().tolist()

    # Apply the correction to the target orientations
    left_ee_target_orientation = R.from_quat(left_ee_target_orientation) * R.from_quat(
        left_ee_target_correction
    )
    left_ee_target_orientation = left_ee_target_orientation.as_quat().tolist()

    right_ee_target_orientation = R.from_quat(
        right_ee_target_orientation
    ) * R.from_quat(right_ee_target_correction)
    right_ee_target_orientation = right_ee_target_orientation.as_quat().tolist()

    # Add a coordinate frame to visualize the obstacle's pose
    left_ee_target = visualizer.add_coordinate_frame(scale=0.1)
    left_ee_target.set_position(left_ee_target_position)
    left_ee_target.set_rotation(left_ee_target_orientation)

    right_ee_target = visualizer.add_coordinate_frame(scale=0.1)
    right_ee_target.set_position(right_ee_target_position)
    right_ee_target.set_rotation(right_ee_target_orientation)

    goal_positions = {
        "L_ee": left_ee_target_position,
        "R_ee": right_ee_target_position,
    }

    goal_orientations = {
        "L_ee": left_ee_target_orientation,
        "R_ee": right_ee_target_orientation,
    }

    print(f"\nGoal positions: {goal_positions}")
    print(f"Goal orientations: {goal_orientations}")

    # Plan motion using interpolation planner with obstacle avoidance
    print("\nPlanning motion with obstacle avoidance...")
    waypoints = mm.plan(
        goal_pos=goal_positions,
        target_rot=goal_orientations,
        relative=False,  # Absolute positions
        planner_type="interpolation",
        interpolation_config=interp_config,  # Pass config with obstacle avoidance
    )

    # Doing FK to check planned waypoints
    if waypoints is not None and len(waypoints) > 0:
        planned_final_pose = mm.fk(
            mm.target_frames, qpos=np.array(list(waypoints[-1].values()))
        )
        print(f"\n{'=' * 80}")
        print("GOAL vs ACHIEVED:")
        print(f"{'=' * 80}")
        for frame_name in mm.target_frames:
            goal_pos = goal_positions[frame_name]
            achieved_pos = planned_final_pose[frame_name].translation
            error = np.linalg.norm(np.array(goal_pos) - np.array(achieved_pos))
            print(f"\n{frame_name}:")
            print(f"  Goal:     {goal_pos}")
            print(f"  Achieved: {achieved_pos}")
            print(f"  Error:    {error:.4f} meters")
        print(f"{'=' * 80}\n")

    # Save waypoints to waypoints1.txt
    if waypoints is not None and len(waypoints) > 0:
        waypoints_file = Path(__file__).parent / "waypoints1.txt"
        # Convert waypoints to JSON-serializable format
        waypoints_serializable = []
        for waypoint in waypoints:
            waypoint_dict = {k: float(v) for k, v in waypoint.items()}
            waypoints_serializable.append(waypoint_dict)

        with open(waypoints_file, "w", encoding="utf-8") as f:
            json.dump(waypoints_serializable, f, indent=2)
        print(f"Saved {len(waypoints)} waypoints to {waypoints_file}")

    if waypoints is not None and len(waypoints) > 0:
        print(f"Planning successful! Generated {len(waypoints)} waypoints")
        print(f"Last waypoint: {waypoints[-1]}")

        mm.visualize_motion_plan(waypoints, duration=3.0)

        # Keep the script running to view the visualization
        print("\nViz is running. Open your web browser to view the motion plan.")
        print("Press Ctrl+C to exit...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")

        # After visualization, move to a neutral position
        print("\nMoving arms to neutral position...")
        mm.left_arm.set_joint_pos([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mm.right_arm.set_joint_pos([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mm.torso.set_joint_pos([0.0, 0.0, 0.0])

        print("Moved to neutral position. Exiting...")

        while True:
            time.sleep(1)

    else:
        print("Planning failed - no valid path found")


if __name__ == "__main__":
    main()
