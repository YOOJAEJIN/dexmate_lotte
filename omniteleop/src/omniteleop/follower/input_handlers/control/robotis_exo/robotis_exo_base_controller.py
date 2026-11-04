"""Base controller for Robotis exoskeleton mobile base."""

from typing import Dict, Any
from omniteleop.follower.input_handlers.control.commands import BaseCommand
from omniteleop.follower.input_handlers.control.base_controller import (
    AbstractBaseController,
)


class RobotisExoBaseController(AbstractBaseController):
    """Robotis exoskeleton base controller with joystick input.
    
    Control scheme:
    - Left stick: Translational velocity (vx=forward/back, vy=left/right)
    - Right stick: Rotational velocity (wz=in-place rotation)
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
            deadzone: Joystick deadzone threshold
        """
        super().__init__(linear_velocity, angular_velocity, deadzone)

    def process(self, joycon_data: Dict[str, Any]) -> BaseCommand:
        """Process Robotis joystick data for base control.
        
        Control scheme:
        - Left stick: Translational velocity (vx=forward/back, vy=left/right)
        - Right stick: Rotational velocity (wz=rotation)

        Args:
            joycon_data: Robotis joystick data with 'left' and 'right' stick info

        Returns:
            BaseCommand with velocity commands
        """
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})

        left_stick = left_data.get("stick", {})
        right_stick = right_data.get("stick", {})

        # Initialize command
        command = BaseCommand()

        # ===== LEFT STICK: Translational Control =====
        stick_x = left_stick.get("x", 0.0)
        stick_y = left_stick.get("y", 0.0)

        # Forward/backward (Y axis)
        if abs(stick_y) > self.deadzone:
            sign = 1.0 if stick_y > 0 else -1.0
            normalized = (abs(stick_y) - self.deadzone) / (1.0 - self.deadzone)
            command.vx = sign * normalized * self.linear_velocity

        # Left/right (X axis)
        if abs(stick_x) > self.deadzone:
            sign = 1.0 if stick_x > 0 else -1.0
            normalized = (abs(stick_x) - self.deadzone) / (1.0 - self.deadzone)
            command.vy = -sign * normalized * self.linear_velocity

        # ===== RIGHT STICK: Rotational Control (In-place rotation) =====
        right_stick_x = right_stick.get("x", 0.0)
        
        if abs(right_stick_x) > self.deadzone:
            sign = 1.0 if right_stick_x > 0 else -1.0
            normalized = (abs(right_stick_x) - self.deadzone) / (1.0 - self.deadzone)
            # Negate for correct rotation direction
            command.wz = -sign * normalized * self.angular_velocity

        command.active = True

        return command

    def is_active(self, joycon_data: Dict[str, Any]) -> bool:
        """Check if base control is active without processing.

        Note: With the new toggle-based system, this method is called
        only when base control is already activated in the main controller.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            True if base control inputs are active (always True when called)
        """
        # In the new toggle-based system, this method is called only when
        # base control is already activated, so we always return True
        return True
