"""Torso controller for robot waist/lift mechanism."""

from typing import Dict, Any
from omniteleop.follower.input_handlers.control.commands import TorsoCommand
from omniteleop.follower.input_handlers.control.torso_controller import (
    AbstractTorsoController,
)

class TorsoController(AbstractTorsoController):
    """Interprets JoyCon inputs for torso control.

    Control schemes (when activated via ZL/ZR buttons):
    - Left stick: Smooth control
        - Stick X: Forward/backward
        - Stick Y: Up/down
    - Right buttons: Discrete steps
        - X: Move up
        - B: Move down
        - A: Move forward
        - Y: Move backward
    """

    def __init__(self, sensitivity: float = 0.008, deadzone: float = 0.1): # sensitivity: float = 0.015
        """Initialize torso controller.

        Args:
            sensitivity: Movement sensitivity in meters per frame (reduced to 0.008 for smoother motion)
        """
        super().__init__(sensitivity, deadzone)
        self.button_multiplier = 0.05 # 1.0  # Buttons move at same speed as stick for smooth, consistent control

    def process(self, joycon_data: Dict[str, Any]) -> TorsoCommand:
        """Process JoyCon data and keyboard input for torso control.

        Args:
            joycon_data: Raw JoyCon data with 'left' and 'right' controller info
                        Can also contain keyboard state: 'torso_up', 'torso_down'

        Returns:
            TorsoCommand with movement deltas
        """
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})

        # Extract buttons
        right_buttons = right_data.get("buttons", {})

        left_stick = left_data.get("stick", {})

        # Initialize command
        command = TorsoCommand()

        delta_x = 0.0
        delta_z = 0.0

        # Left controller: stick for smooth control
        stick_x = left_stick.get("x", 0.0)
        stick_y = left_stick.get("y", 0.0)

        # Stick X controls forward/backward
        if abs(stick_x) > self.deadzone:
            delta_x = stick_x * self.sensitivity  # Forward/backward

        # Stick Y controls up/down
        if abs(stick_y) > self.deadzone:
            delta_z = stick_y * self.sensitivity  # Up/down

        # Right controller: buttons for discrete control
        button_step = self.sensitivity * self.button_multiplier

        # Up/down controls (Z-axis)
        if right_buttons.get("x", False):
            delta_z = button_step  # Move up
        elif right_buttons.get("b", False):
            delta_z = -button_step  # Move down

        # Forward/backward controls (X-axis)
        if right_buttons.get("a", False):
            delta_x = button_step  # Move forward
        elif right_buttons.get("y", False):
            delta_x = -button_step  # Move backward

        # Keyboard controls (when torso mode is active)
        # A key: move down, C key: move up
        # B key: neutral/center (handled by RobotisExoController with interpolation)
        if joycon_data.get("torso_up", False):
            delta_z = button_step  # Move up
        elif joycon_data.get("torso_down", False):
            delta_z = -button_step  # Move down
        # NOTE: torso_neutral (b key) is NOT handled here
        # It's handled by RobotisExoController.process() with smooth interpolation
        # to ensure gradual movement to neutral position

        command.delta_x = delta_x
        command.delta_z = delta_z
        command.active = True  # Always active when called

        return command

    def is_active(self, joycon_data: Dict[str, Any]) -> bool:
        """Check if torso control is active without processing.

        Note: With the new toggle-based system, this method is called
        only when torso control is already activated in the main controller.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            True if torso control inputs are active (always True when called)
        """
        # In the new toggle-based system, this method is called only when
        # torso control is already activated, so we always return True
        return True
