"""Torso component processor."""

from typing import Dict, Any, List
import numpy as np

from loguru import logger
from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkIKConfig

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import RobotCommand, CommandMode
from omniteleop.common.log_utils import suppress_loguru_module

class TorsoProcessor(BaseComponentProcessor):
    """Processes torso movement commands.

    Supports two modes:
    - Absolute: Direct joint position control
    - Relative: Delta movement using IK to maintain head target frame position
    """

    def __init__(self, side, config, motion_manager, robot_info, teleop_mode):
        """Initialize torso processor.

        Args:
            side: Not used (always None for torso)
            config: Full configuration dictionary
            motion_manager: Shared motion manager instance
            robot_info: Robot information
            teleop_mode: Teleoperation mode identifier
        """
        super().__init__(None, config, motion_manager, robot_info, teleop_mode)

        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints("torso")

        # Setup IK configuration for torso delta movements
        self._setup_torso_ik_config()

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "torso"

    def _setup_torso_ik_config(self) -> None:
        """Configure torso IK parameters for delta movements."""
        # Use higher damping weights to prevent aggressive joint movements
        # and maintain axis alignment during IK solving
        # Prevent j1 from being used to compensate for z-axis limit constraints
        self.torso_ik_damping = IKDampingWeightsConfig(
            default=1e-12,
            override={
                "torso_j1": 1e-6,    # Much higher damping to lock x-axis, prevent compensatory movements (was: 1e-20, then 1e-6)
                "torso_j2": 1e-8,    # Lower damping for y-axis (main height control) (was: 1e-20)
                "torso_j3": 1e-3,    # Higher damping to stabilize z-axis (was: 1e-20, then 1e-6)
            },
        )
        self._torso_ik_target_frame = "head_l1"
        self.torso_ik_config = LocalPinkIKConfig(
            damping_weights=self.torso_ik_damping,
            target_frames=[self._torso_ik_target_frame],
        )

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process torso movement command.

        Routes to absolute or relative processing based on command mode.

        Args:
            input_data: Torso input data with 'pos' and 'mode' fields
            command: RobotCommand to update

        Returns:
            True if processing succeeded
        """
        mode = input_data.get("mode", CommandMode.RELATIVE)
        positions = input_data.get("pos", [])

        if not positions:
            return False

        if mode == CommandMode.RELATIVE:
            if input_data.get("use_joint", False):
                # Handle as joint delta movement
                return self._process_torso_joint_delta(positions, command)
            # Handle as delta movement using IK
            return self._process_torso_delta(positions, command)
        else:
            if input_data.get("smooth", False):
                # If wait_for_arms is set, check if arms are close enough to L-shape target first
                if input_data.get("wait_for_arms", False):
                    l_shape_left  = [1.4103, 0.3000, 0.0000, -1.5560, 0.0000, 0.0000, 0.0000]
                    l_shape_right = [-1.4103, -0.3000, 0.0000, -1.5560, 0.0000, 0.0000, 0.0000]
                    joint_pos = self.motion_manager.get_joint_pos_dict()
                    left_joints  = self.robot_info.get_component_joints("left_arm")
                    right_joints = self.robot_info.get_component_joints("right_arm")
                    left_err  = max(abs(joint_pos.get(n, 0) - l_shape_left[i])  for i, n in enumerate(left_joints))
                    right_err = max(abs(joint_pos.get(n, 0) - l_shape_right[i]) for i, n in enumerate(right_joints))
                    if left_err > 0.15 or right_err > 0.15:  # ~8.6도 이내면 충분히 도달
                        logger.debug(f"[H-KEY] Waiting for arms (L:{left_err:.3f} R:{right_err:.3f} rad)")
                        return False  # 아직 팔이 목표 미도달, 이번 프레임 토르소 skip
                    logger.info(f"[H-KEY] Arms ready (L:{left_err:.3f} R:{right_err:.3f} rad), starting torso")
                # Smooth interpolation towards target (e.g. B-key neutral, H-key home)
                return self._process_smooth_absolute(positions, command)
            else:
                # Direct absolute positions
                # Also sync motion_manager so subsequent RELATIVE/IK starts from this position
                self.motion_manager.torso.set_joint_pos(positions)
                command.output_components["torso"] = {
                    "pos": positions,
                    "mode": CommandMode.ABSOLUTE.value,
                }
                return True

    def _process_torso_joint_delta(self, deltas: List[float], command: RobotCommand) -> bool:
        """Compute torso joint positions from joint delta movement.

        Args:
            deltas: Joint displacement [dj1, dj2, dj3] in radians
            command: RobotCommand to update with joint positions

        Returns:
            True if succeeded
        """
        if len(deltas) < 3:
            return False

        # Read current virtual state from motion_manager
        current_dict = self.motion_manager.get_joint_pos_dict()
        current = [
            current_dict.get("torso_j1", 0.0),
            current_dict.get("torso_j2", 0.0),
            current_dict.get("torso_j3", 0.0),
        ]

        # Apply deltas
        new_pos = [current[i] + deltas[i] for i in range(3)]

        # Apply safety limits
        new_pos[0] = float(np.clip(new_pos[0], 0.0873, 1.4827))
        new_pos[1] = float(np.clip(new_pos[1], 0.0873, 3.0537))
        new_pos[2] = float(np.clip(new_pos[2], -1.4827, 1.4827))

        # Sync motion_manager so next frame starts from here
        self.motion_manager.torso.set_joint_pos(new_pos)
        command.output_components["torso"] = {
            "pos": new_pos,
            "mode": CommandMode.ABSOLUTE.value,
        }
        return True

    def _process_torso_delta(self, deltas: List[float], command: RobotCommand) -> bool:
        """Compute torso joint positions from Cartesian delta movement.

        Uses IK to solve for torso joint positions that achieve the
        desired Cartesian displacement of the head target frame.

        Args:
            deltas: Cartesian displacement [dx, dy, dz] in meters
            command: RobotCommand to update with joint positions

        Returns:
            True if IK succeeded and positions were set
        """
        if len(deltas) < 3:
            return False

        # Get current torso pose
        torso_pose = self.motion_manager.fk([self._torso_ik_target_frame])[
            self._torso_ik_target_frame
        ]

        # Apply delta movement
        delta_translation = np.array(deltas)
        target_torso_pose = torso_pose.np.copy()
        target_torso_pose[:3, 3] += delta_translation

        # Solve IK
        target_pose = {self._torso_ik_target_frame: target_torso_pose}

        try:
            with suppress_loguru_module("dexmotion", enabled=True):
                joint_solution, has_collision, within_limits = self.motion_manager.ik(
                    target_pose=target_pose,
                    type="pink",
                    custom_config=self.torso_ik_config,
                )

                if not has_collision:
                    self.motion_manager.set_joint_pos(joint_solution)

        except Exception as e:
            logger.error(f"Error solving torso IK: {e}")
            return False

        # Extract torso joint positions from solution
        if joint_solution and within_limits and not has_collision:
            torso_joint_positions = [
                joint_solution[f"torso_j{i + 1}"] for i in range(3)
            ]
            
            # Apply safety limits to all torso joints
            # Limits match leader side: torso_j1, torso_j2, torso_j3
            torso_j1, torso_j2, torso_j3 = torso_joint_positions
            
            # Clamp to safe ranges with 5-degree safety margin applied
            torso_j1 = np.clip(torso_j1, 0.0873, 1.4827)   # URDF: [0, 1.570] 0.0873, 1.4827
            torso_j2 = np.clip(torso_j2, 0.0873, 3.0537)   # URDF: [0, 3.141] 0.0873, 3.0537
            torso_j3 = np.clip(torso_j3, -1.4827, 1.4827)  # URDF: [-1.570, 1.570] -1.4827, 1.4827
            
            torso_joint_positions = [torso_j1, torso_j2, torso_j3]
            
            command.output_components["torso"] = {
                "pos": torso_joint_positions,
                "mode": CommandMode.ABSOLUTE.value,
            }
            return True

        return False

    def _process_smooth_absolute(self, target: List[float], command: RobotCommand) -> bool:
        """Smoothly interpolate towards target position at 10% per frame.

        Uses motion_manager's virtual state as current position,
        so this works correctly regardless of how many times it's called.

        Args:
            target: Target joint positions [j1, j2, j3]
            command: RobotCommand to update

        Returns:
            True always
        """
        # Read current virtual state from motion_manager
        current_dict = self.motion_manager.get_joint_pos_dict()
        current = [
            current_dict.get("torso_j1", target[0]),
            current_dict.get("torso_j2", target[1]),
            current_dict.get("torso_j3", target[2]),
        ]

        # 10% interpolation towards target each frame
        new_pos = [
            current[i] + (target[i] - current[i]) * 0.1
            for i in range(3)
        ]

        # Apply safety limits
        new_pos[0] = float(np.clip(new_pos[0], 0.0873, 1.4827))
        new_pos[1] = float(np.clip(new_pos[1], 0.0873, 3.0537))
        new_pos[2] = float(np.clip(new_pos[2], -1.4827, 1.4827))

        logger.debug(f"[SMOOTH] {[round(c,3) for c in current]} → {[round(p,3) for p in new_pos]} (target: {target})")

        # Sync motion_manager so next frame starts from here
        self.motion_manager.torso.set_joint_pos(new_pos)
        command.output_components["torso"] = {
            "pos": new_pos,
            "mode": CommandMode.ABSOLUTE.value,
        }
        return True

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync torso to robot state.

        Args:
            robot_joints: Dictionary of joint positions from robot feedback
        """
        torso_pos = robot_joints.get("torso", [])
        if torso_pos:
            self.motion_manager.torso.set_joint_pos(torso_pos)
