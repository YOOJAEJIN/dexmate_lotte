"""Hand controller for grippers."""

from typing import Dict, Any
from loguru import logger
from omniteleop.follower.input_handlers.control.hand_controller import (
    AbstractHandController,
)
from omniteleop.follower.input_handlers.control.commands import HandCommand
from omniteleop.follower.input_handlers.control.end_effector import (
    AbstractEndEffectorController,
)
from omniteleop.follower.input_handlers.control.meta_con.end_effectors import (
    create_end_effector,
)

class HandController(AbstractHandController):
    """Interprets VR inputs for gripper control.

    Control scheme (MetaQuest 3 VR):
    - Gripper: squeeze button opens/closes
      - squeeze pressed = close gripper
      - squeeze released = open gripper
    """

    def __init__(
        self,
        left_config: Dict[str, Any] = None,
        right_config: Dict[str, Any] = None,
        stick_deadzone: float = 0.1,
    ):
        """Initialize gripper controller.

        Args:
            left_config: Configuration for left gripper
            right_config: Configuration for right gripper
            stick_deadzone: Not used for gripper (kept for compatibility)
        """
        super().__init__(left_config, right_config, stick_deadzone)

        # Create gripper controllers (grippers only, no fingers)
        self.left_gripper = create_end_effector("left", left_config)
        self.right_gripper = create_end_effector("right", right_config)

        # Track previous squeeze state to detect state change
        self._prev_left_squeeze = False
        self._prev_right_squeeze = False

        logger.info(
            f"Gripper controller initialized with {left_config.get('type', 'none')} (left) and {right_config.get('type', 'none')} (right)"
        )

    def process(self, vr_data: Dict[str, Any]) -> HandCommand:
        """Process VR data for gripper control.

        Args:
            vr_data: VR data with 'leftState' and 'rightState' controller info

        Returns:
            HandCommand with gripper position
        """
        command = HandCommand()

        # Extract VR states
        left_state = vr_data.get("leftState", {})
        right_state = vr_data.get("rightState", {})

        left_squeeze = left_state.get("squeeze", False)
        right_squeeze = right_state.get("squeeze", False)

        # Process left gripper: squeeze=close, release=open
        if left_squeeze:
            left_result = self.left_gripper.process_input({"squeeze": True})
        else:
            left_result = self.left_gripper.process_input({"squeeze": False})

        # Process right gripper: squeeze=close, release=open
        if right_squeeze:
            right_result = self.right_gripper.process_input({"squeeze": True})
        else:
            right_result = self.right_gripper.process_input({"squeeze": False})

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



    def is_active(self, vr_data: Dict[str, Any]) -> bool:
        """Check if gripper control is active.

        Args:
            vr_data: VR data

        Returns:
            True - gripper is always active (can use squeeze anytime)
        """
        return True

    def get_left_gripper(self) -> AbstractEndEffectorController:
        """Get the left gripper controller."""
        return self.left_gripper

    def get_right_gripper(self) -> AbstractEndEffectorController:
        """Get the right gripper controller."""
        return self.right_gripper

    def _has_modifiers(self, vr_data: Dict[str, Any]) -> bool:
        """Check if any modifier buttons are pressed.

        For VR (MetaQuest 3), there are no modifiers that disable gripper control.
        Gripper is always available.

        Args:
            vr_data: VR data

        Returns:
            False - no modifiers in VR, gripper always active
        """
        return False
