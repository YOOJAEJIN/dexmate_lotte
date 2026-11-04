#!/usr/bin/env python3
"""Modular end-effector controllers for different gripper types.

This module provides a unified interface for controlling gripper
types from VR input (MetaQuest 3).
"""

from __future__ import annotations

from typing import Dict, Any

from loguru import logger

from omniteleop.follower.input_handlers.control.end_effector import (
    AbstractEndEffectorController,
)

class GripperController(AbstractEndEffectorController):
    """Controller for simple grippers.

    VR Mapping (MetaQuest 3):
    - Squeeze button pressed: close gripper
    - Squeeze button released: open gripper
    """

    def __init__(self, side: str, config: Dict[str, Any]):
        """Initialize gripper controller.

        Args:
            side: "left" or "right"
            config: Gripper configuration with poses
        """
        super().__init__(side, config)

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

    def process_input(self, vr_input: Dict[str, Any]) -> Dict[str, float]:
        """Process VR input for gripper control.

        Args:
            vr_input: Simple dict with "squeeze" bool

        Returns:
            Tuple of (joint positions list, mode string)
        """
        squeeze = vr_input.get("squeeze", False)

        # Squeeze button: close gripper
        if squeeze:
            joint_positions = self.get_predefined_poses("close")
        # No squeeze: open gripper
        else:
            joint_positions = self.get_predefined_poses("open")

        # Return absolute positions
        return (joint_positions, "absolute")

    def get_predefined_poses(self, pose_name: str) -> list:
        """Get predefined pose by name."""
        return self.predefined_poses.get(pose_name, [])

def create_end_effector(
    side: str, config: Dict[str, Any]
) -> AbstractEndEffectorController:
    """Factory function to create appropriate end-effector controller.

    Args:
        side: "left" or "right"
        config: End-effector configuration from robot config

    Returns:
        GripperController instance
    """
    effector_type = config.get("type", "gripper")

    if effector_type == "gripper":
        logger.info(f"Created gripper controller for {side} side")
        return GripperController(side, config)
    else:
        logger.warning(f"Unknown end-effector type '{effector_type}', defaulting to gripper")
        return GripperController(side, config)
