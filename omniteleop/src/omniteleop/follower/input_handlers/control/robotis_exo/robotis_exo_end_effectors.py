#!/usr/bin/env python3
"""Modular end-effector controllers for different hand/gripper types.

This module provides a unified interface for controlling different
types of end-effectors (5-finger hands, grippers, etc) from JoyCon input.
"""

from __future__ import annotations

from typing import Dict, Any
from dataclasses import dataclass

import numpy as np
from loguru import logger

from omniteleop.follower.input_handlers.control.end_effector import (
    AbstractEndEffectorController,
)

@dataclass
class JoyConEndEffectorInput:
    """Parsed JoyCon input for end-effector control."""

    stick_x: float  # -1 to 1
    stick_y: float  # -1 to 1
    buttons: Dict[str, bool]  # Button states
    gripper_position: float = 0.0  # Gripper position feedback from leader (-0.9 to 0.9)
    gripper_open_pressed: bool = False  # Keyboard O key pressed
    gripper_close_pressed: bool = False  # Keyboard P key pressed
    zl_zr_pressed: bool = False  # ZL for left, ZR for right - digital button
    l_r_pressed: bool = False  # L for left, R for right - top button
    fine_adjustment_active: bool = False  # Whether fine adjustment mode is active

class HandF5D6Controller(AbstractEndEffectorController):
    """Controller for 5-finger/6-DoF dexterous hands.

    JoyCon Mapping:
    - Home pose: capture (left) / home (right) button
    - Open hand: left button (left) / Y button (right)
    - Close hand: right button (left) / A button (right)
    - Pinch (thumb+index): up button (left) / X button (right)
    - Pinch (thumb+index+middle): down button (left) / B button (right)
    - Fine adjustment (toggle with L/R buttons):
      - L/R + stick: Control thumb joints
      - L/R + up/down (left) or X/B (right): Control four fingers
    """

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize 5-finger hand controller.

        Args:
            side: "left" or "right"
            config: Hand configuration with joint names
        """
        super().__init__(side, config)

        # Control sensitivity for fine adjustment mode
        self.sensitivity = config.get("sensitivity", 0.05)
        self.stick_deadzone = config.get("stick_deadzone", 0.1)

        self.predefined_poses = config.get("poses", {})
        # Get DOF from first predefined pose, default to 6 for f5d6 hand
        if self.predefined_poses:
            self.dof = len(list(self.predefined_poses.values())[0])
        else:
            self.dof = 6  # Default for f5d6 hand

    def process_input(self, joycon_input: JoyConEndEffectorInput) -> Dict[str, float]:
        """Process JoyCon input for 5-finger hand control.

        Args:
            joycon_input: Parsed JoyCon input data

        Returns:
            Dictionary of joint positions
        """
        # Check if ZL/ZR is pressed - this is for torso control, not hand
        # When ZL/ZR is pressed, hand controls are disabled (torso takes priority)
        if joycon_input.zl_zr_pressed:
            return [], "absolute"  # Return empty when disabled

        # Fine adjustment mode - RELATIVE control
        # Use fine_adjustment_active flag instead of requiring L/R button to be held
        if joycon_input.fine_adjustment_active:
            joint_deltas = np.zeros(self.dof)  # Store deltas for relative control

            # Calculate thumb deltas
            if (
                abs(joycon_input.stick_x) > self.stick_deadzone
                or abs(joycon_input.stick_y) > self.stick_deadzone
            ):
                # Abduction delta
                abd_delta = -1 * joycon_input.stick_x * self.sensitivity
                if self.side == "left":
                    abd_delta = -abd_delta  # Mirror for left hand
                joint_deltas[-1] = abd_delta

                # Flexion delta
                flex_delta = joycon_input.stick_y * self.sensitivity
                joint_deltas[0] = flex_delta

            # Calculate finger deltas from buttons
            finger_delta = 0.0
            if self.side == "left":
                if joycon_input.buttons.get("up", False):
                    finger_delta = self.sensitivity  # Close increment
                elif joycon_input.buttons.get("down", False):
                    finger_delta = -self.sensitivity  # Open increment
            else:
                if joycon_input.buttons.get("x", False):
                    finger_delta = self.sensitivity  # Close increment
                elif joycon_input.buttons.get("b", False):
                    finger_delta = -self.sensitivity  # Open increment

            if finger_delta != 0.0:
                joint_deltas[1:-1] = finger_delta
            return (
                (joint_deltas.tolist(), "relative")
                if joint_deltas.size > 0
                else ([], "relative")
            )

        else:
            joint_positions = []
            # Non-modifier button controls
            if self.side == "left":
                # Home pose
                if joycon_input.buttons.get("capture", False):
                    joint_positions = self.get_predefined_poses("home")

                # Open hand
                elif joycon_input.buttons.get("left", False):
                    joint_positions = self.get_predefined_poses("open")

                # Close hand
                elif joycon_input.buttons.get("right", False):
                    joint_positions = self.get_predefined_poses("close")

                # Pinch with thumb and index
                elif joycon_input.buttons.get("up", False):
                    joint_positions = self.get_predefined_poses("pinch")

                # Pinch with thumb, index, middle
                elif joycon_input.buttons.get("down", False):
                    joint_positions = self.get_predefined_poses("three_finger_pinch")

            else:  # right controller
                # Home pose
                if joycon_input.buttons.get("home", False):
                    joint_positions = self.get_predefined_poses("home")

                # Open hand
                elif joycon_input.buttons.get("y", False):
                    joint_positions = self.get_predefined_poses("open")

                # Close hand
                elif joycon_input.buttons.get("a", False):
                    joint_positions = self.get_predefined_poses("close")

                # Pinch with thumb and index
                elif joycon_input.buttons.get("x", False):
                    joint_positions = self.get_predefined_poses("pinch")

                # Pinch with thumb, index, middle
                elif joycon_input.buttons.get("b", False):
                    joint_positions = self.get_predefined_poses("three_finger_pinch")

        # Return absolute positions for predefined poses
        return (joint_positions, "absolute")

    def get_predefined_poses(self, pose_name: str) -> Dict[str, float]:
        """Get predefined pose by name."""
        return self.predefined_poses[pose_name]

class GripperController(AbstractEndEffectorController):
    """Controller for simple grippers.

    JoyCon Mapping (similar to HandF5D6Controller):
    - Open gripper: left button (left) / Y button (right)
    - Close gripper: right button (left) / A button (right)
    - Fine adjustment (toggle with L/R buttons):
      - L/R + up/down (left) or X/B (right): Increment/decrement gripper position
    """

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize gripper controller.

        Args:
            side: "left" or "right"
            config: Gripper configuration with poses
        """
        super().__init__(side, config)

        # Control sensitivity for fine adjustment mode
        self.sensitivity = config.get("sensitivity", 0.05)

        # Get poses from config (only open and close for gripper)
        common_hands = config.get("common_hands", {})
        side_ee_type = common_hands.get(f"{side}_ee_type", None)

        if side_ee_type is not None:
            self.predefined_poses = common_hands.get(side_ee_type, {}).get(side, {
                "open": [0.0], "close": [1.0],
            })
        else:
            # Fallback to single ee_type if left_ee_type/right_ee_type is not defined
            self.predefined_poses = config.get("poses", {"open": [0.0], "close": [1.0]})
        
        # Get DOF from first predefined pose, default to 1 for simple gripper
        if self.predefined_poses:
            self.dof = len(list(self.predefined_poses.values())[0])
        else:
            self.dof = 1  # Default for simple gripper

    def process_input(self, joycon_input: JoyConEndEffectorInput) -> Dict[str, float]:
        """Process JoyCon input for gripper control.

        Args:
            joycon_input: Parsed JoyCon input data

        Returns:
            Tuple of (joint positions list, mode string)
        """
        # Check if ZL/ZR is pressed - this is for torso control, not hand
        # When ZL/ZR is pressed, gripper controls are disabled (torso takes priority)
        if joycon_input.zl_zr_pressed:
            return [], "absolute"  # Return empty when disabled

        # Priority 1: Check keyboard O/P flags (gripper_open_pressed / gripper_close_pressed)
        if joycon_input.gripper_open_pressed:
            return (self.get_predefined_poses("open"), "absolute")
        elif joycon_input.gripper_close_pressed:
            return (self.get_predefined_poses("close"), "absolute")

        # Priority 2: Fine adjustment mode - RELATIVE control
        if joycon_input.fine_adjustment_active:
            joint_deltas = np.zeros(self.dof)

            # Calculate gripper delta from buttons
            gripper_delta = 0.0
            if self.side == "left":
                if joycon_input.buttons.get("up", False):
                    gripper_delta = self.sensitivity  # Open increment
                elif joycon_input.buttons.get("down", False):
                    gripper_delta = -self.sensitivity  # Close increment
            else:
                if joycon_input.buttons.get("x", False):
                    gripper_delta = self.sensitivity  # Open increment
                elif joycon_input.buttons.get("b", False):
                    gripper_delta = -self.sensitivity  # Close increment

            if gripper_delta != 0.0:
                joint_deltas[:] = gripper_delta

            return (
                (joint_deltas.tolist(), "relative")
                if joint_deltas.size > 0
                else ([], "relative")
            )

        # Priority 3: Leader gripper feedback from robot reader
        # Use the gripper position from the leader arm
        # Convert from leader range [-0.5 (open) to 0.9 (close)] to robot range based on predefined poses (open vs close)
        if joycon_input.gripper_position is not None:
            # Clamp input to valid range [-0.5, 0.9]
            clamped_leader = max(-0.5, min(0.9, joycon_input.gripper_position))
            
            # Map based on predefined open/close pose values dynamically
            open_pose = self.get_predefined_poses("open")
            close_pose = self.get_predefined_poses("close")
            open_val = open_pose[0] if open_pose else 0.0
            close_val = close_pose[0] if close_pose else 1.0

            # Dynamic mapping formula
            robot_gripper_pos = open_val + (clamped_leader + 0.5) * (close_val - open_val) / 1.4
            
            # Clamp to open/close bounds
            min_val = min(open_val, close_val)
            max_val = max(open_val, close_val)
            robot_gripper_pos = max(min_val, min(max_val, robot_gripper_pos))
            return ([robot_gripper_pos], "absolute")

        # Priority 4: Non-modifier button controls (absolute poses)
        else:
            joint_positions = []
            if self.side == "left":
                # Open gripper
                if joycon_input.buttons.get("left", False):
                    joint_positions = self.get_predefined_poses("open")
                # Close gripper
                elif joycon_input.buttons.get("right", False):
                    joint_positions = self.get_predefined_poses("close")

            else:  # right controller
                # Open gripper
                if joycon_input.buttons.get("y", False):
                    joint_positions = self.get_predefined_poses("open")
                # Close gripper
                elif joycon_input.buttons.get("a", False):
                    joint_positions = self.get_predefined_poses("close")

        # Return absolute positions for predefined poses
        return (joint_positions, "absolute")

    def get_predefined_poses(self, pose_name: str) -> list:
        """Get predefined pose by name."""
        return self.predefined_poses.get(pose_name, [])

class NoEndEffector(AbstractEndEffectorController):
    """Placeholder for arms without end-effectors."""

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize no-op end-effector."""
        super().__init__(side, config)

    def process_input(self, joycon_input: JoyConEndEffectorInput) -> Dict[str, float]:
        """No-op - returns empty dict."""
        return self.joint_positions

def create_end_effector(
    side: str, config: Dict[str, Any]
) -> AbstractEndEffectorController:
    """Factory function to create appropriate end-effector controller.

    Args:
        side: "left" or "right"
        config: End-effector configuration from robot config

    Returns:
        Appropriate EndEffectorController instance
    """
    effector_type = config.get("type", "none")

    if effector_type == "hand_f5d6":
        logger.info(f"Created 5-finger hand controller for {side} side")
        return HandF5D6Controller(side, config)
    elif effector_type == "gripper":
        logger.info(f"Created gripper controller for {side} side")
        return GripperController(side, config)
    else:
        logger.info(f"No end-effector configured for {side} side")
        return NoEndEffector(side, config)
