"""Torso controller for robot waist/lift mechanism."""

from typing import Dict, Any
from omniteleop.follower.input_handlers.control.commands import TorsoCommand
from omniteleop.follower.input_handlers.control.torso_controller import (
    AbstractTorsoController,
)

class TorsoController(AbstractTorsoController):
    """Interprets VR inputs for torso control using RIGHT controller.

    Control scheme (when activated via A button):
    - Thumbstick up/down: forward/backward movement (delta_x)
    - Thumbstick left/right: up/down lift movement (delta_z)
    """

    def __init__(self, sensitivity: float = 0.015, deadzone: float = 0.1):
        """Initialize torso controller.

        Args:
            sensitivity: Movement sensitivity in meters per frame
            deadzone: Thumbstick deadzone
        """
        super().__init__(sensitivity, deadzone)

    def process(self, vr_data: Dict[str, Any]) -> TorsoCommand:
        """Process VR data for torso control.

        VR Mapping (RIGHT controller when A button active):
        - Stick up/down: forward/backward movement (delta_x)
        - Stick left/right: up/down lift movement (delta_z)

        Args:
            vr_data: VR input data

        Returns:
            TorsoCommand with movement deltas
        """
        # Initialize command
        command = TorsoCommand()
        
        # VR Data Format
        right_state = vr_data.get("rightState", {})
        
        # Get thumbstick values (normalized -1.0 to 1.0)
        thumb_x, thumb_y = right_state.get("thumbstickValue", [0.0, 0.0])
        
        # Apply deadzone
        # Stick up/down (Y) controls delta_x (forward/backward)
        if abs(thumb_y) > self.deadzone:
            command.delta_x = thumb_y * self.sensitivity
        
        # Stick left/right (X) controls delta_z (up/down lift)
        if abs(thumb_x) > self.deadzone:
            command.delta_z = thumb_x * self.sensitivity
        
        command.active = True
        return command

    def is_active(self, vr_data: Dict[str, Any]) -> bool:
        """Check if torso control is active.

        Returns:
            True (always active when called)
        """
        return True
