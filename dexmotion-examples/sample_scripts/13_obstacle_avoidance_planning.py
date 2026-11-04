# Copyright (c) 2026 Dexmate Inc. — MIT License (see LICENSE).

"""
Example: External Obstacle Avoidance with LocalPinkExtIKSolver

This example demonstrates how to:
1. Configure an external obstacle via LocalPinkExtIKConfig
2. Initialize MotionManager with local_ik_solver_type="pink_ext"
3. Solve IK with type="pink_ext" that automatically avoids the configured obstacle
4. Use SE3.Interpolate for proper geodesic pose interpolation
5. Support left-hand or right-hand control modes (auto-selected based on target y)
6. Handle obstacle attachment to end-effector or static positioning

Control Modes (automatically selected based on target_y):
- "left": When target_y > 0
- "right": When target_y <= 0

Passive hand: positioned ±0.4m in y-direction from active hand

Usage:
  python 13_obstacle_avoidance_planning.py --target-x 1.0 --target-y 0.2 --target-z 0.7
  python 13_obstacle_avoidance_planning.py --target-x 1.0 --target-y -0.2 --target-z 0.7
  python 13_obstacle_avoidance_planning.py --target-x 1.0 --target-y 0.2 \
    --target-z 0.7 --enable-perturbation False
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import sapien
import tyro
from scipy.spatial.transform import Rotation as R

from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkExtIKConfig
from dexmotion.motion_manager import MotionManager


def apply_orientation_correction(
    orientation_quat, correction_axis="x", correction_angle=np.pi / 2
):
    """Apply a rotation correction to a quaternion orientation.

    Args:
        orientation_quat: Quaternion as list [x, y, z, w]
        correction_axis: Axis for correction rotation ('x', 'y', or 'z')
        correction_angle: Angle for correction rotation in radians

    Returns:
        Corrected quaternion as list [x, y, z, w]
    """
    correction = R.from_euler(correction_axis, correction_angle).as_quat().tolist()
    corrected_orientation = R.from_quat(orientation_quat) * R.from_quat(correction)
    return corrected_orientation.as_quat().tolist()


@dataclass
class Config:
    """Configuration for obstacle avoidance planning demo."""

    # Initial target position for active end-effector
    target_x: float = 1.0
    """X coordinate of the initial active EE target position"""

    target_y: float = 0.2
    """Y coordinate of the initial active EE target position
    (determines control mode: >0 = left, <=0 = right)"""

    target_z: float = 0.7
    """Z coordinate of the initial active EE target position"""

    # Passive hand parameters
    passive_hand_offset: float = 0.4
    """Y-axis offset for passive hand from active hand"""

    passive_hand_z: float = 1.0
    """Z coordinate for passive hand"""

    # Collision avoidance parameters
    enable_perturbation: bool = True
    """Enable joint perturbation for collision avoidance (default: True)"""


def interpolate_dual_ee_poses(
    current_poses,
    target_positions,
    target_orientations,
    pos_intervals=0.03,
    rot_intervals=0.05,
    ee_names=None,
    interp_mode="both",
):
    """
    Interpolate poses for dual end-effectors using SE3.Interpolate.

    This uses Pinocchio's SE3.Interpolate() which provides proper geodesic
    interpolation on the SE3 manifold - same as the interpolation planner.

    Args:
        current_poses: Dictionary with FK results (SE3 objects) for
            'L_gripper_base' and 'R_gripper_base'
        target_positions: Dictionary with target positions for each EE
        target_orientations: Dictionary with target orientations
            (quaternions [x,y,z,w])
        pos_intervals: Position interval size in meters (default: 0.03m)
        rot_intervals: Rotation interval size in radians (default: 0.05rad)
        ee_names: List of EE names to interpolate
            (default: ['L_gripper_base', 'R_gripper_base'])
        interp_mode: Interpolation mode - "both", "position_only", or "orientation_only"

    Returns:
        List of dictionaries, each containing interpolated poses for both end-effectors
    """
    if ee_names is None:
        ee_names = ["L_gripper_base", "R_gripper_base"]

    interpolated_poses = []
    ee_data = {}

    # Convert target positions/orientations to SE3 objects for each end-effector
    for ee_name in ee_names:
        # Current pose is already SE3
        current_se3 = current_poses[ee_name]

        # Convert target to SE3
        target_pos = np.array(target_positions[ee_name])
        # [x, y, z, w] from scipy
        target_quat_xyzw = np.array(target_orientations[ee_name])
        # 0123 -> 1230
        target_quat_xyzw = np.roll(target_quat_xyzw, -1)

        # Convert quaternion to rotation matrix
        target_rot_matrix = R.from_quat(target_quat_xyzw).as_matrix()
        target_se3 = pin.SE3(target_rot_matrix, target_pos)

        # Calculate distances for determining number of steps
        pos_distance = np.linalg.norm(target_pos - current_se3.translation)

        # Calculate rotation distance (same as interpolation planner)
        R_diff = target_se3.rotation @ current_se3.rotation.T
        val = (np.trace(R_diff) - 1) / 2
        val = np.clip(val, -1, 1)  # Ensure valid arccos input
        rot_distance = np.arccos(val)  # in radians

        # Determine number of steps based on intervals and mode
        pos_steps = max(int(np.ceil(pos_distance / pos_intervals)), 1)
        rot_steps = max(int(np.ceil(rot_distance / rot_intervals)), 1)

        if interp_mode == "position_only":
            steps = pos_steps
        elif interp_mode == "orientation_only":
            steps = rot_steps
        else:  # "both"
            steps = max(pos_steps, rot_steps)

        print(f"current_se3: {current_se3}")
        print(f"target_se3: {target_se3}")

        print(f"pos_distance: {pos_distance}, rot_distance: {rot_distance}")
        print(f"pos_steps: {pos_steps}, rot_steps: {rot_steps}")
        print(f"interp_mode: {interp_mode}, steps: {steps}")

        ee_data[ee_name] = {
            "current_se3": current_se3,
            "target_se3": target_se3,
            "pos_distance": pos_distance,
            "rot_distance": rot_distance,
            "steps": steps,
        }

    # Use maximum steps across all end-effectors
    num_steps = max(ee_data[ee]["steps"] for ee in ee_names)

    # Print interpolation info
    print("\nInterpolation info (using SE3.Interpolate):")
    for ee_name in ee_names:
        data = ee_data[ee_name]
        print(
            f"  {ee_name} - Position distance: {data['pos_distance']:.3f}m, "
            f"Rotation distance: {data['rot_distance']:.3f}rad"
        )
    print(f"  Total interpolation steps: {num_steps}")

    # Create interpolated waypoints using SE3.Interpolate (geodesic interpolation)
    # This is the exact same method used by the interpolation planner
    alphas = np.linspace(0, 1, num_steps + 1)
    for alpha in alphas:
        waypoint = {}
        for ee_name in ee_names:
            data = ee_data[ee_name]

            if interp_mode == "position_only":
                # Interpolate position only, keep target orientation constant
                current_pos = data["current_se3"].translation
                target_pos = data["target_se3"].translation
                interp_pos = current_pos + alpha * (target_pos - current_pos)
                target_quat_xyzw = R.from_matrix(data["target_se3"].rotation).as_quat()
                interp_quat_xyzw = target_quat_xyzw
                # 1230 -> 0123
                interp_quat_xyzw = np.roll(interp_quat_xyzw, 1)

            elif interp_mode == "orientation_only":
                # Keep target position constant, interpolate orientation only
                target_pos = data["target_se3"].translation
                interp_pos = target_pos
                # Use SE3.Interpolate for orientation interpolation
                interp_se3 = pin.SE3.Interpolate(
                    data["current_se3"], data["target_se3"], float(alpha)
                )
                interp_quat_xyzw = R.from_matrix(interp_se3.rotation).as_quat()
                # 1230 -> 0123
                interp_quat_xyzw = np.roll(interp_quat_xyzw, 1)

            else:  # "both"
                # Use SE3.Interpolate - same as interpolation planner (line 216-222)
                interp_se3 = pin.SE3.Interpolate(
                    data["current_se3"], data["target_se3"], float(alpha)
                )
                # Extract position and orientation from the interpolated SE3
                interp_pos = interp_se3.translation
                # Convert rotation matrix to quaternion [x, y, z, w] (scipy format)
                interp_quat_xyzw = R.from_matrix(interp_se3.rotation).as_quat()
                # 1230 -> 0123
                interp_quat_xyzw = np.roll(interp_quat_xyzw, 1)

            waypoint[ee_name] = {
                "position": interp_pos.tolist(),
                "orientation": interp_quat_xyzw.tolist(),  # [x, y, z, w]
            }

        interpolated_poses.append(waypoint)

    print(f"  Generated {len(interpolated_poses)} waypoints")

    return interpolated_poses


def _get_relative_quaternion(q_current_wxyz, q_desired_wxyz):
    """
    Compute relative quaternion (in wxyz) that rotates current → desired.
    All inputs and outputs are in wxyz format.
    """
    q_current_xyzw = np.roll(q_current_wxyz, -1)
    q_desired_xyzw = np.roll(q_desired_wxyz, -1)

    r_rel = R.from_quat(q_desired_xyzw) * R.from_quat(q_current_xyzw).inv()

    q_rel_xyzw = r_rel.as_quat()
    q_rel_wxyz = np.roll(q_rel_xyzw, 1)
    return q_rel_wxyz


def _get_current_orientation(active_ee_orientation_wxyz, q_rel_wxyz):
    """
    Compute the current orientation in wxyz after applying relative rotation.
    All inputs and outputs are in wxyz format.
    """
    r_rel = R.from_quat(np.roll(q_rel_wxyz, -1))
    r_cur = R.from_quat(np.roll(active_ee_orientation_wxyz, -1))
    # check_xyzw = (r_rel * r_cur).as_quat()
    check_xyzw = (r_cur * r_rel).as_quat()
    check_wxyz = np.roll(check_xyzw, 1)

    # Define 180° about z (in wxyz)
    q_z180_wxyz = np.array([0.0, 0.0, 0.0, 1.0])
    # Convert both to scipy's xyzw format
    q_xyzw = np.roll(check_wxyz, -1)
    q_z180_xyzw = np.roll(q_z180_wxyz, -1)
    # Compose (local z flip fix)
    r_fixed = R.from_quat(q_xyzw) * R.from_quat(q_z180_xyzw)
    # Convert back to wxyz
    q_fixed_wxyz = np.roll(r_fixed.as_quat(), 1)

    return q_fixed_wxyz


def _try_collision_avoidance_with_perturbation(
    mm: MotionManager,
    current_q: dict,
    control_mode: str,
    target_positions: dict,
    target_orientations: dict,
    obstacle_attached: bool,
    obstacle_objects: list,
    active_ee_name: str,
    q_rel_calculated_wxyz: np.ndarray,
    max_attempts: int = 8,
):
    """
    Try to find a collision-free solution by perturbing active arm joints 6 and 7.

    Args:
        mm: MotionManager instance
        current_q: Current joint configuration dictionary
        control_mode: "left" or "right" - which hand is active
        target_positions: Target positions for end-effectors
        target_orientations: Target orientations for end-effectors
        obstacle_attached: Whether obstacle is attached to end-effector
        obstacle_objects: List of visualizer obstacle objects
        active_ee_name: Name of the active end-effector
        q_rel_calculated_wxyz: Relative quaternion for attached obstacle
        max_attempts: Maximum number of perturbation attempts

    Returns:
        Tuple of (success, solution) where:
            - success: Boolean indicating if collision-free solution was found
            - solution: The collision-free joint configuration if found, None otherwise
    """
    # Determine active arm joint names based on control mode
    if control_mode == "left":
        joint_6_name = "L_arm_j6"
        joint_7_name = "L_arm_j7"
    else:  # control_mode == "right"
        joint_6_name = "R_arm_j6"
        joint_7_name = "R_arm_j7"

    # Try different perturbations for joints 6 and 7
    perturbation_angles = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]

    for attempt in range(max_attempts):
        # Create perturbed configuration
        perturbed_q = current_q.copy()

        # Apply perturbations to both joints using joint names
        if attempt < len(perturbation_angles):
            perturbed_q[joint_6_name] += perturbation_angles[attempt]
            perturbed_q[joint_7_name] += perturbation_angles[attempt]
        else:
            # Try random perturbations for remaining attempts
            perturbed_q[joint_6_name] += np.random.uniform(-0.5, 0.5)
            perturbed_q[joint_7_name] += np.random.uniform(-0.5, 0.5)

        # Check if perturbed configuration is collision-free
        try:
            mm.set_joint_pos(perturbed_q)

            # Update attached obstacle if it's attached to end-effector
            if obstacle_attached:
                # Get current FK of active end-effector with perturbed configuration
                current_ee_poses = mm.fk([active_ee_name])
                active_ee_pose = current_ee_poses[active_ee_name]

                # Translate in the EE frame's z-direction
                ee_z_axis = active_ee_pose.rotation[:, 2]
                current_obstacle_2_position = (
                    active_ee_pose.translation + ee_z_axis * 0.15
                )

                # Calculate orientation
                active_ee_orientation = R.from_matrix(
                    active_ee_pose.rotation
                ).as_quat()  # xyzw
                active_ee_orientation_wxyz = np.roll(active_ee_orientation, 1)

                current_obstacle_2_orientation = _get_current_orientation(
                    active_ee_orientation_wxyz, q_rel_calculated_wxyz
                )

                # Update visualizer
                obstacle_objects[1].set_pose(
                    sapien.Pose(
                        p=current_obstacle_2_position, q=current_obstacle_2_orientation
                    )
                )

                # Update collision geometry
                mm.local_ik_solver.update_obstacle_position(
                    obstacle_name_prefix="obstacle_2",
                    new_position=current_obstacle_2_position,
                    new_orientation=current_obstacle_2_orientation,
                )

            # Check collision for the perturbed configuration
            perturbed_solution, perturbed_collision, perturbed_limits = mm.ik(
                target_pos=target_positions,
                target_rot=target_orientations,
                type="pink_ext",
            )

            if (
                perturbed_solution is not None
                and not perturbed_collision
                and perturbed_limits
            ):
                # Found a collision-free solution
                return True, perturbed_solution

        except (ValueError, RuntimeError):
            # If perturbation fails, continue to next attempt
            continue

    # No collision-free solution found
    return False, None


def run_phase(
    mm: MotionManager,
    obstacle_objects: list,
    active_ee_target_position: list,
    control_mode: str,
    obstacle_attached: bool,
    config: Config,
    pos_intervals: float = 0.03,
    rot_intervals: float = 0.05,
    interp_mode: str = "both",
    phase_name: str = "Phase",
    q_rel_wxyz: Optional[np.ndarray] = None,
    enable_perturbation: bool = True,
):
    """
    Run a single planning phase with IK solving for dual end-effectors.

    Args:
        mm: MotionManager instance
        obstacle_objects: List of visualizer obstacle objects
        active_ee_target_position: Target position [x, y, z] for active end-effector
        control_mode: "left" or "right" - which hand is active
        obstacle_attached: If True, obstacle follows active EE; if False,
            obstacle stays at initial position
        config: Configuration object with passive hand parameters
        pos_intervals: Position interval size in meters for interpolation
            (default: 0.03)
        rot_intervals: Rotation interval size in radians for interpolation
            (default: 0.05)
        interp_mode: Interpolation mode - "both", "position_only", or
            "orientation_only" (default: "both")
        phase_name: Name of the phase for logging
        q_rel_wxyz: Relative quaternion for attached obstacle
            (required if obstacle_attached=True in subsequent phases)
        enable_perturbation: Enable joint perturbation for collision avoidance

    Returns:
        Tuple of (final_poses, q_rel_wxyz) where:
            - final_poses: FK results after phase completion
            - q_rel_wxyz: Relative quaternion (only meaningful if
              obstacle_attached=True)
    """
    print("\n" + "=" * 80)
    print(f"{phase_name}: Moving to {active_ee_target_position}")
    print(f"Control mode: {control_mode}, Obstacle attached: {obstacle_attached}")
    print(f"Interpolation mode: {interp_mode}")
    print("=" * 80)

    # Get current end-effector poses
    current_poses = mm.fk(["L_gripper_base", "R_gripper_base"])

    # Determine target orientations based on control mode
    if control_mode == "left":
        active_ee_target_orientation = (
            R.from_euler("xyz", [np.pi / 2, np.pi, 0.0]).as_quat().tolist()
        )
        left_ee_target_position = active_ee_target_position
        left_ee_target_orientation = active_ee_target_orientation

        # Right hand: offset in y, keep initial orientation
        right_ee_target_position = [
            active_ee_target_position[0],
            active_ee_target_position[1] - config.passive_hand_offset,
            config.passive_hand_z,
        ]
        right_rot_matrix = current_poses["R_gripper_base"].rotation
        right_ee_target_orientation = R.from_matrix(right_rot_matrix).as_quat().tolist()
        left_ee_target_orientation = apply_orientation_correction(
            left_ee_target_orientation, "x", np.pi / 2
        )
        print(f"right_ee_target_orientation: {right_ee_target_orientation}")

        active_ee_name = "L_gripper_base"
        print(f"  Active hand: L_gripper_base -> {left_ee_target_position}")
        print(f"  Passive hand: R_gripper_base -> {right_ee_target_position}")

    else:  # control_mode == "right"
        active_ee_target_orientation = (
            R.from_euler("xyz", [-np.pi / 2, np.pi, 0.0]).as_quat().tolist()
        )
        right_ee_target_position = active_ee_target_position
        right_ee_target_orientation = active_ee_target_orientation

        # Left hand: offset in y, keep initial orientation
        left_ee_target_position = [
            active_ee_target_position[0],
            active_ee_target_position[1] + config.passive_hand_offset,
            config.passive_hand_z,
        ]
        left_rot_matrix = current_poses["L_gripper_base"].rotation
        left_ee_target_orientation = R.from_matrix(left_rot_matrix).as_quat().tolist()
        right_ee_target_orientation = apply_orientation_correction(
            right_ee_target_orientation, "x", -np.pi / 2
        )
        print(f"left_ee_target_orientation: {left_ee_target_orientation}")

        active_ee_name = "R_gripper_base"
        print(f"  Active hand: R_gripper_base -> {right_ee_target_position}")
        print(f"  Passive hand: L_gripper_base -> {left_ee_target_position}")

    # Interpolate poses
    interpolated_poses = interpolate_dual_ee_poses(
        current_poses=current_poses,
        target_positions={
            "L_gripper_base": left_ee_target_position,
            "R_gripper_base": right_ee_target_position,
        },
        target_orientations={
            "L_gripper_base": left_ee_target_orientation,
            "R_gripper_base": right_ee_target_orientation,
        },
        pos_intervals=pos_intervals,
        rot_intervals=rot_intervals,
        interp_mode=interp_mode,
    )

    # Solve IK for all waypoints
    print(f"\nSolving IK for {len(interpolated_poses)} waypoints...")
    successful_waypoints = 0
    failed_waypoints = 0
    collision_waypoints = 0

    start_time = time.time()
    # Use provided value or None for first waypoint calculation
    q_rel_calculated_wxyz = q_rel_wxyz

    for i, waypoint in enumerate(interpolated_poses):
        # Extract positions and orientations from the waypoint
        target_positions = {
            ee: waypoint[ee]["position"] for ee in ["L_gripper_base", "R_gripper_base"]
        }
        target_orientations = {
            ee: waypoint[ee]["orientation"]
            for ee in ["L_gripper_base", "R_gripper_base"]
        }

        # Update obstacle if attached to end-effector
        if obstacle_attached:
            # Get current FK of active end-effector
            current_ee_poses = mm.fk([active_ee_name])
            active_ee_pose = current_ee_poses[active_ee_name]

            # Translate in the EE frame's z-direction
            ee_z_axis = active_ee_pose.rotation[:, 2]
            current_obstacle_2_position = active_ee_pose.translation + ee_z_axis * 0.15

            # Calculate orientation
            active_ee_orientation = R.from_matrix(
                active_ee_pose.rotation
            ).as_quat()  # xyzw
            active_ee_orientation_wxyz = np.roll(active_ee_orientation, 1)

            if i == 0 and q_rel_calculated_wxyz is None:
                # Calculate relative quaternion on first waypoint
                obstacle_2_desired_first_waypoint = np.array(
                    [0.70710678, 0.0, 0.0, 0.70710678]
                )
                q_rel_calculated_wxyz = _get_relative_quaternion(
                    active_ee_orientation_wxyz, obstacle_2_desired_first_waypoint
                )

            current_obstacle_2_orientation = _get_current_orientation(
                active_ee_orientation_wxyz, q_rel_calculated_wxyz
            )

            # Update visualizer
            obstacle_objects[1].set_pose(
                sapien.Pose(
                    p=current_obstacle_2_position, q=current_obstacle_2_orientation
                )
            )

            # Update collision geometry
            mm.local_ik_solver.update_obstacle_position(
                obstacle_name_prefix="obstacle_2",
                new_position=current_obstacle_2_position,
                new_orientation=current_obstacle_2_orientation,
            )

        # Solve IK for this waypoint
        solution, is_in_collision, is_within_limits = mm.ik(
            target_pos=target_positions,
            target_rot=target_orientations,
            type="pink_ext",
        )

        # Check solution validity
        if solution is not None and not is_in_collision and is_within_limits:
            mm.set_joint_pos(solution)
            successful_waypoints += 1
            if i % 5 == 0:
                print(f"  Waypoint {i}/{len(interpolated_poses)}: Success")
        else:
            # Try collision avoidance by perturbing active arm joints 6 and 7
            collision_avoided = False
            if is_in_collision and solution is not None and enable_perturbation:
                # Get current joint configuration
                current_q = mm.get_joint_pos_dict()

                # Try collision avoidance with perturbation
                success, perturbed_solution = (
                    _try_collision_avoidance_with_perturbation(
                        mm=mm,
                        current_q=current_q,
                        control_mode=control_mode,
                        target_positions=target_positions,
                        target_orientations=target_orientations,
                        obstacle_attached=obstacle_attached,
                        obstacle_objects=obstacle_objects,
                        active_ee_name=active_ee_name,
                        q_rel_calculated_wxyz=q_rel_calculated_wxyz,
                        max_attempts=8,
                    )
                )

                if success:
                    # Found a collision-free solution
                    mm.set_joint_pos(perturbed_solution)
                    collision_avoided = True
                    successful_waypoints += 1
                    if i % 5 == 0:
                        print(
                            f"  Waypoint {i}/{len(interpolated_poses)}: "
                            f"Success (collision avoided)"
                        )
                else:
                    # Restore original configuration if collision avoidance failed
                    mm.set_joint_pos(current_q)

            if not collision_avoided:
                failed_waypoints += 1
                if is_in_collision:
                    collision_waypoints += 1
                if i % 5 == 0:
                    print(
                        f"  Waypoint {i}/{len(interpolated_poses)}: Failed "
                        f"(collision: {is_in_collision}, "
                        f"limits: {not is_within_limits})"
                    )

    end_time = time.time()

    # Print summary
    print(f"\n{phase_name} IK Solution Summary:")
    print(f"  Total waypoints: {len(interpolated_poses)}")
    success_pct = 100 * successful_waypoints / len(interpolated_poses)
    fail_pct = 100 * failed_waypoints / len(interpolated_poses)
    print(f"  Successful: {successful_waypoints} ({success_pct:.1f}%)")
    print(f"  Failed: {failed_waypoints} ({fail_pct:.1f}%)")
    print(f"  Collisions: {collision_waypoints}")
    print(f"  Time taken: {end_time - start_time:.4f} seconds")

    # Get final end-effector positions
    final_poses = mm.fk(["L_gripper_base", "R_gripper_base"])
    print(f"\nFinal end-effector positions after {phase_name}:")
    for ee_name, pose in final_poses.items():
        if pose and hasattr(pose, "translation"):
            print(f"  {ee_name}: {pose.translation}")

    return final_poses, q_rel_calculated_wxyz


def main():
    # Parse configuration from command line
    config = tyro.cli(Config)

    # Determine control mode based on target_y
    if config.target_y > 0:
        control_mode = "left"
    else:
        control_mode = "right"

    print(f"\n{'=' * 80}")
    print("Obstacle Avoidance Planning Demo")
    print(f"{'=' * 80}")
    print(
        f"Initial target position: [{config.target_x}, {config.target_y}, "
        f"{config.target_z}]"
    )
    print(f"Control mode (auto-selected): {control_mode}")
    perturbation_status = "Enabled" if config.enable_perturbation else "Disabled"
    print(f"Collision avoidance perturbation: {perturbation_status}")
    print(f"{'=' * 80}\n")

    # Initial active EE target position from config
    active_ee_target_position = [config.target_x, config.target_y, config.target_z]

    # Configure external obstacles. Replace the placeholder paths below with
    # paths to your own URDF + collision/visual mesh assets.
    obstacle_2_initial_position = active_ee_target_position.copy()
    obstacle_2_initial_position[2] -= 0.05
    obstacle_2_initial_orientation = (
        R.from_euler("xyz", [np.pi / 2, 0.0, 0.0]).as_quat().tolist()
    )

    obstacles = [
        {
            "urdf_path": "path/to/your/obstacle_1.urdf",
            "position": [1.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "name": "obstacle_1",
        },
        {
            "urdf_path": "path/to/your/obstacle_2.urdf",
            "position": active_ee_target_position,
            "orientation": obstacle_2_initial_orientation,
            "name": "obstacle_2",
        },
    ]

    # Configure IK damping weights
    damping_weights = IKDampingWeightsConfig(
        default=1e-8,
        override={
            "torso_j1": 1e-2,
            "torso_j2": 1e-2,
            "torso_j3": 1e-2,
        },
    )

    ext_ik_config = LocalPinkExtIKConfig(
        external_obstacles=obstacles,
        damping_weights=damping_weights,
        target_frames=["L_gripper_base", "R_gripper_base"],
    )

    # Configure initial joint positions based on control mode
    if control_mode == "left":
        initial_config = {
            "torso_j1": 0.52,
            "torso_j2": 1.05,
            "torso_j3": 0.37,
            "L_gripper_j1": 0.8,
            "L_gripper_j2": 0.8,
        }
    else:
        initial_config = {
            "torso_j1": 0.52,
            "torso_j2": 1.05,
            "torso_j3": 0.37,
            "R_gripper_j1": 0.8,
            "R_gripper_j2": 0.8,
        }

    # Initialize MotionManager with custom IK config
    print("Initializing MotionManager with LocalPinkExtIKSolver...")
    mm = MotionManager(
        robot_type="vega_1",
        custom_urdf_path=Path("path/to/your/robot.urdf"),
        custom_srdf_path=Path("path/to/your/robot.srdf"),
        init_local_ik=True,
        init_global_ik=False,
        visualizer_type="sapien",
        custom_local_ik_config=ext_ik_config,
        initial_joint_configuration_dict=initial_config,
        joint_regions_to_lock=["HEAD", "BASE"],
        target_frames=["L_gripper_base", "R_gripper_base"],
    )

    visualizer = mm.visualizer

    # Add obstacle objects to visualizer. Replace with paths to your own
    # visual + collision meshes (same order as the `obstacles` list above).
    obstacle_glb_path_list = [
        "path/to/your/obstacle_1.glb",
        "path/to/your/obstacle_2.glb",
    ]
    obstacle_collision_path_list = [
        "path/to/your/obstacle_1_collision.glb",
        "path/to/your/obstacle_2_collision.glb",
    ]

    obstacle_objects = []
    for i, obstacle in enumerate(obstacles):
        obstacle_obj = visualizer.add_object(
            obstacle_glb_path_list[i],
            pos=np.array(obstacle["position"]),
            quat=np.array(obstacle["orientation"]),
            scale=np.array([1, 1, 1]),
            collision_mesh_path=obstacle_collision_path_list[i],
        )
        obstacle_objects.append(obstacle_obj)

    # Display current end-effector positions
    print("\nCurrent end-effector positions:")
    current_poses = mm.fk(["L_gripper_base", "R_gripper_base"])
    for ee_name, pose in current_poses.items():
        if pose and hasattr(pose, "translation"):
            print(f"  {ee_name}: {pose.translation}")

    # ============================================================================
    # PHASE 1: Move to initial target position (obstacle at initial position)
    # ============================================================================
    pre_grasp = active_ee_target_position.copy()
    pre_grasp[2] += 0.4
    grasp = active_ee_target_position.copy()
    grasp[2] += 0.3
    _, _ = run_phase(
        mm=mm,
        obstacle_objects=obstacle_objects,
        active_ee_target_position=pre_grasp,
        control_mode=control_mode,
        obstacle_attached=False,  # Obstacle stays at initial position
        config=config,
        phase_name="Phase 1",
        enable_perturbation=config.enable_perturbation,
    )

    _, _ = run_phase(
        mm=mm,
        obstacle_objects=obstacle_objects,
        active_ee_target_position=active_ee_target_position,
        control_mode=control_mode,
        obstacle_attached=False,  # Obstacle stays at initial position
        config=config,
        phase_name="Phase 1.1",
        enable_perturbation=config.enable_perturbation,
        # interp_mode="position_only",
    )

    # ============================================================================
    # PHASE 2: Move to final position (obstacle attached to end-effector)
    # ============================================================================
    # Determine phase 2 target based on control mode
    present_x = active_ee_target_position[0]
    if control_mode == "left":
        active_ee_target_position_phase2 = [present_x, 0.2, 1.2]
    else:
        active_ee_target_position_phase2 = [present_x, -0.2, 1.2]

    _, _ = run_phase(
        mm=mm,
        obstacle_objects=obstacle_objects,
        active_ee_target_position=active_ee_target_position_phase2,
        control_mode=control_mode,
        obstacle_attached=True,  # Obstacle follows active end-effector
        config=config,
        phase_name="Phase 2",
    )

    if control_mode == "left":
        active_ee_target_position_phase2 = [0.6, 0.2, 1.2]
    else:
        active_ee_target_position_phase2 = [0.6, -0.2, 1.2]

    _, _ = run_phase(
        mm=mm,
        obstacle_objects=obstacle_objects,
        active_ee_target_position=active_ee_target_position_phase2,
        control_mode=control_mode,
        obstacle_attached=True,  # Obstacle follows active end-effector
        config=config,
        phase_name="Phase 2.1",
        enable_perturbation=config.enable_perturbation,
    )

    # ============================================================================
    # Complete - Keep visualizer open
    # ============================================================================
    print("\n" + "=" * 80)
    print("All phases complete! Keeping visualizer open...")
    print("=" * 80)
    visualizer.render_loop()


if __name__ == "__main__":
    main()
