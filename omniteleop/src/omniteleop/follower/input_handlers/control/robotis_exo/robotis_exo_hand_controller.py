"""Hand controller for end-effectors (hands/grippers)."""

from typing import Dict, Any
from loguru import logger
from omniteleop.follower.input_handlers.control.hand_controller import (
    AbstractHandController,
)
from omniteleop.follower.input_handlers.control.end_effector import (
    AbstractEndEffectorController,
)
from omniteleop.follower.input_handlers.control.commands import HandCommand
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_end_effectors import (
    create_end_effector,
    JoyConEndEffectorInput,
)

class HandController(AbstractHandController):
    """Interprets JoyCon inputs for hand/gripper control.

    Control schemes (when no ZL/ZR is pressed):
    - Basic gestures:
        - Open hand: Left/Y button
        - Close hand: Right/A button
        - Pinch (thumb+index): Up/X button
        - Pinch (thumb+index+middle): Down/B button
        - Home pose: Capture/Home button
    - Fine adjustment (L/R + controls):
        - L/R + stick: Control thumb joints
        - L/R + up/down or X/B: Control four fingers
    """

    def __init__(
        self,
        left_config: Dict[str, Any] = None,
        right_config: Dict[str, Any] = None,
        stick_deadzone: float = 0.1,
    ):
        """Initialize hand controller.

        Args:
            left_config: Configuration for left end-effector
            right_config: Configuration for right end-effector
            stick_deadzone: Deadzone for thumbstick inputs
        """
        super().__init__(left_config, right_config, stick_deadzone)

        # Create end-effector controllers

        self.left_effector = create_end_effector("left", left_config)
        self.right_effector = create_end_effector("right", right_config)

        logger.info(
            f"Hand controller initialized with {left_config.get('type', 'none')} (left) and {right_config.get('type', 'none')} (right)"
        )

    def process(self, joycon_data: Dict[str, Any]) -> HandCommand:
        """Process JoyCon data for hand control.

        Args:
            joycon_data: Raw JoyCon data with 'left' and 'right' controller info

        Returns:
            HandCommand with joint positions or deltas
        """
        command = HandCommand()

        # Check if any modifiers are pressed (which would disable hand control)
        if self._has_modifiers(joycon_data):
            return command

        # Get fine adjustment state from main controller
        fine_adjustment_active = joycon_data.get("fine_adjustment_active", False)

        # Process left hand
        left_input = self._parse_joycon_input(
            "left", joycon_data, fine_adjustment_active
        )
        left_result = self.left_effector.process_input(left_input)

        # Process right hand
        right_input = self._parse_joycon_input(
            "right", joycon_data, fine_adjustment_active
        )
        right_result = self.right_effector.process_input(right_input)

        # Check if result is a tuple (positions, mode) or just positions
        if isinstance(left_result, tuple):
            left_positions, left_mode = left_result
        else:
            left_positions = left_result
            left_mode = "absolute"

        if isinstance(right_result, tuple):
            right_positions, right_mode = right_result
        else:
            right_positions = right_result
            right_mode = "absolute"

        # Update command if we have positions
        if left_positions:
            command.left_positions = left_positions
            command.left_mode = left_mode
            command.active = True

        if right_positions:
            command.right_positions = right_positions
            command.right_mode = right_mode
            command.active = True

        return command

    def _has_modifiers(self, joycon_data: Dict[str, Any]) -> bool:
        """Check if any modifier buttons are pressed (ZL/ZR).

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            True if modifiers are pressed
        """
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})

        left_buttons = left_data.get("buttons", {})
        right_buttons = right_data.get("buttons", {})

        left_zl = left_buttons.get("zl", False)
        right_zr = right_buttons.get("zr", False)

        return left_zl or right_zr

    def _parse_joycon_input(
        self, side: str, joycon_data: Dict[str, Any], fine_adjustment_active: bool
    ) -> JoyConEndEffectorInput:
        """Parse JoyCon data for a specific side.

        Args:
            side: "left" or "right"
            joycon_data: Raw JoyCon data
            fine_adjustment_active: Whether fine adjustment mode is active

        Returns:
            Parsed JoyConEndEffectorInput
        """
        controller_data = joycon_data.get(side, {})

        stick = controller_data.get("stick", {})
        buttons = controller_data.get("buttons", {})
        gripper = controller_data.get("gripper", {})

        # Get button states
        zl_zr_key = "zl" if side == "left" else "zr"
        l_r_key = "l" if side == "left" else "r"

        zl_zr_pressed = buttons.get(zl_zr_key, False)
        l_r_pressed = buttons.get(l_r_key, False)
        
        # Get gripper position feedback from leader
        gripper_position = gripper.get("position", 0.0)
        
        # Get keyboard gripper state flags (from robotis_exo_joycon_handle)
        gripper_open_pressed = joycon_data.get("keyboard_gripper_open", False)
        gripper_close_pressed = joycon_data.get("keyboard_gripper_close", False)

        return JoyConEndEffectorInput(
            stick_x=stick.get("x", 0.0),
            stick_y=stick.get("y", 0.0),
            gripper_position=gripper_position,
            gripper_open_pressed=gripper_open_pressed,
            gripper_close_pressed=gripper_close_pressed,
            zl_zr_pressed=zl_zr_pressed,
            l_r_pressed=l_r_pressed,
            buttons=buttons,
            fine_adjustment_active=fine_adjustment_active,
        )

    def is_active(self, joycon_data: Dict[str, Any]) -> bool:
        """Check if hand control is active without processing.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            True if hand control should be active
        """
        # Hand control is active when:
        # 1. No modifiers are pressed AND
        # 2. Either gripper data or keyboard gripper flags are present
        
        if self._has_modifiers(joycon_data):
            return False
        
        # Check for keyboard gripper state
        if joycon_data.get("keyboard_gripper_open") or joycon_data.get("keyboard_gripper_close"):
            return True
        
        # Check for gripper data from leader
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})
        
        left_gripper = left_data.get("gripper", {})
        right_gripper = right_data.get("gripper", {})
        
        # Check if gripper has position data (not empty and has "position" key)
        has_gripper_data = (
            ("position" in left_gripper and left_gripper["position"] is not None) or 
            ("position" in right_gripper and right_gripper["position"] is not None)
        )
        
        return has_gripper_data

    def get_left_effector(self) -> AbstractEndEffectorController:
        """Get the left end-effector controller."""
        return self.left_effector

    def get_right_effector(self) -> AbstractEndEffectorController:
        """Get the right end-effector controller."""
        return self.right_effector
