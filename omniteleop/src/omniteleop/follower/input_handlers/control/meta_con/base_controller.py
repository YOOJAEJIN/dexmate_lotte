"""Base controller for mobile robot platform."""

from typing import Dict, Any
from omniteleop.follower.input_handlers.control.commands import BaseCommand
from omniteleop.follower.input_handlers.control.base_controller import (
    AbstractBaseController,
)

class BaseController(AbstractBaseController):
    """Interprets VR inputs for mobile base control using LEFT controller.

    Control scheme (when activated via B button):
    - Thumbstick up/down: forward/backward (vx)
    - Thumbstick left/right: side move (vy)
    - X button: turn CCW (wz negative)
    - Y button: turn CW (wz positive)
    """

    def __init__(
        self,
        linear_velocity: float = 0.5,
        angular_velocity: float = 1.0,
        deadzone: float = 0.1,
    ):
        """Initialize base controller.

        Args:
            linear_velocity: Max linear velocity in m/s
            angular_velocity: Max angular velocity in rad/s
        """
        super().__init__(linear_velocity, angular_velocity, deadzone)

    def process(self, vr_data: Dict[str, Any]) -> BaseCommand:
        """Process VR data for base control.

        VR Mapping (LEFT controller when B button active):
        - Stick up/down: vx (forward/backward)
        - Stick left/right: vy (side move)
        - X button: wz negative (turn CCW)
        - Y button: wz positive (turn CW)

        Args:
            vr_data: VR input data

        Returns:
            BaseCommand with velocity commands
        """
        # Initialize command
        command = BaseCommand()

        # VR Data Format
        left_state = vr_data.get("leftState", {})
        
        # Get thumbstick values (normalized -1.0 to 1.0)
        thumb_x, thumb_y = left_state.get("thumbstickValue", [0.0, 0.0])
        
        # Apply deadzone and remap to [0, 1]
        if abs(thumb_y) > self.deadzone:
            sign = 1.0 if thumb_y > 0 else -1.0
            normalized = (abs(thumb_y) - self.deadzone) / (1.0 - self.deadzone)
            command.vx = -sign * normalized * self.linear_velocity  # Forward/backward (inverted)

        if abs(thumb_x) > self.deadzone:
            sign = 1.0 if thumb_x > 0 else -1.0
            normalized = (abs(thumb_x) - self.deadzone) / (1.0 - self.deadzone)
            command.vy = -sign * normalized * self.linear_velocity  # Side move (inverted)
        
        # Buttons for rotation (left controller: X = aButton, Y = bButton)
        if left_state.get("aButton", False):
            command.wz = self.angular_velocity  # X: Turn CW (positive)
        elif left_state.get("bButton", False):
            command.wz = -self.angular_velocity  # Y: Turn CCW (negative)
        
        command.active = True
        return command

    def is_active(self, vr_data: Dict[str, Any]) -> bool:
        """Check if base control is active.

        Returns:
            True (always active when called)
        """
        return True
