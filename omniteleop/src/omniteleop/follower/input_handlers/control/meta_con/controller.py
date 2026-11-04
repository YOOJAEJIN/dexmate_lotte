"""MetaQuest 3 controller for teleoperation via VR."""

from typing import Dict, Any, Optional
from loguru import logger

from omniteleop.follower.input_handlers.control.controller import AbstractController
from omniteleop.follower.input_handlers.control.meta_con.base_controller import (
    BaseController,
)
from omniteleop.follower.input_handlers.control.meta_con.torso_controller import (
    TorsoController,
)
from omniteleop.follower.input_handlers.control.meta_con.hand_controller import (
    HandController,
)
from omniteleop.follower.input_handlers.control.commands import ControllerCommand


class MetaQuestController(AbstractController):
    """Controller for MetaQuest 3 VR controller inputs.

    Processes MetaQuest 3 controller data and routes to component controllers
    for gripper, torso, and base control.

    Control mapping:
    - Gripper: squeeze button opens/closes (both LEFT and RIGHT)
    - Base (LEFT): B button activates
      * Thumbstick up/down: forward/backward (vx)
      * Thumbstick left/right: side move (vy)
      * X button: turn CCW (wz negative)
      * Y button: turn CW (wz positive)
    - Torso (RIGHT): A button activates
      * Thumbstick up/down: forward/backward movement (delta_x)
      * Thumbstick left/right: up/down lift movement (delta_z)
    - Emergency Stop: Both thumbsticks pressed simultaneously

    Priority: All active modes (base, torso, gripper) run simultaneously
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MetaQuest controller.

        Args:
            config: Configuration dictionary with component settings
        """
        super().__init__(config)

        # input_handlers > vr + common
        self.config = config or {}

        # Initialize component controllers
        self._init_component_controllers()

        # Control state (toggle)
        self.base_active = False
        self.torso_active = False
        self.estop_active = False  # Start with estop inactive; toggle with both thumbsticks
        self.exit_requested = False

        # Track previous button states for toggle detection
        self._prev_base_button = False
        self._prev_torso_button = False
        self._prev_both_sticks = False

        # Statistics
        self.stats = {
            "total_updates": 0,
            "base_activations": 0,
            "torso_activations": 0,
            "hand_activations": 0,
            "estop_toggles": 0,
        }

        logger.info("MetaQuestController initialized for VR teleoperation")

    def _init_component_controllers(self) -> None:
        """Initialize all component controllers."""
        # Extract configurations
        base_config = self.config.get("base", {})
        torso_config = self.config.get("torso", {})
        hands_config = self.config.get("hands", {})

        # Get common hands configuration
        common_hands = self.config.get("common", {}).get("hands", {})

        # Determine hand type from ROBOT_CONFIG env var
        import os

        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        if "gripper" in robot_config:
            hand_type = "gripper"
        elif "f5d6" in robot_config:
            hand_type = "hand_f5d6"
        else:
            hand_type = "none"

        # Get left and right configurations directly from hands config (flat structure)
        left_config = hands_config.get("left", {}).copy()
        right_config = hands_config.get("right", {}).copy()

        # Add type and sensitivity to each config for hand controller
        left_config["type"] = hand_type
        right_config["type"] = hand_type
        left_config["sensitivity"] = hands_config.get("sensitivity", 0.05)
        right_config["sensitivity"] = hands_config.get("sensitivity", 0.05)

        # Inject common hands configuration for ee_type and robotiq poses checking
        left_config["common_hands"] = common_hands
        right_config["common_hands"] = common_hands

        # Get deadzones
        stick_deadzone = self.config.get("stick_deadzone", 0.1)

        # Create controllers
        self.base_controller = BaseController(
            linear_velocity=base_config.get("linear_velocity", 0.5),
            angular_velocity=base_config.get("angular_velocity", 1.0),
            deadzone=stick_deadzone,
        )

        self.torso_controller = TorsoController(
            sensitivity=torso_config.get("sensitivity", 0.01), deadzone=stick_deadzone
        )

        self.hand_controller = HandController(
            left_config=left_config,
            right_config=right_config,
            stick_deadzone=stick_deadzone,
        )



    def process(self, vr_data: Dict[str, Any]) -> ControllerCommand:
        """Process VR data and return commands for all components.

        Base, torso, and gripper can be active simultaneously.
        Base and Torso use toggle activation (B and A buttons).

        Args:
            vr_data: VR input data with pose matrices and button states

        Returns:
            RobotCommands with all active component commands
        """
        self.stats["total_updates"] += 1

        # Initialize result
        vr_command = ControllerCommand()

        # Get states
        left_state = vr_data.get("leftState", {})
        right_state = vr_data.get("rightState", {})

        # Emergency stop combo: both sticks pressed simultaneously
        curr_both_sticks = left_state.get("thumbstick", False) and right_state.get("thumbstick", False)
        if curr_both_sticks and not self._prev_both_sticks:  # Combo just triggered
            self.estop_active = not self.estop_active
            self.stats["estop_toggles"] += 1
            state = "ACTIVATED" if self.estop_active else "DEACTIVATED"
            logger.warning(f"Emergency stop {state}")
        self._prev_both_sticks = curr_both_sticks

        # Check for active estop - if estop active, stop all commands
        if self.estop_active:
            vr_command.estop = True
            vr_command.priority = "estop"
            return vr_command

        # Toggle Base using RIGHT B button
        curr_base_button = right_state.get("bButton", False)
        if curr_base_button and not self._prev_base_button:  # Button just pressed
            self.base_active = not self.base_active
            state = "ACTIVATED" if self.base_active else "DEACTIVATED"
            logger.info(f"Base control {state}")
            self.stats["base_activations"] += 1
        self._prev_base_button = curr_base_button

        # Toggle Torso using RIGHT A button
        curr_torso_button = right_state.get("aButton", False)
        if curr_torso_button and not self._prev_torso_button:  # Button just pressed
            self.torso_active = not self.torso_active
            state = "ACTIVATED" if self.torso_active else "DEACTIVATED"
            logger.info(f"Torso control {state}")
            self.stats["torso_activations"] += 1
        self._prev_torso_button = curr_torso_button

        # Process hand (arm + gripper) control using triggers and squeeze
        if self.hand_controller.is_active(vr_data):
            vr_command.hands = self.hand_controller.process(vr_data)
            vr_command.priority = "hands"
            self.stats["hand_activations"] += 1

        # Process base control if active
        if self.base_active:
            vr_command.base = self.base_controller.process(vr_data)
            vr_command.priority = "base"

        # Process torso control if active
        if self.torso_active:
            vr_command.torso = self.torso_controller.process(vr_data)
            vr_command.priority = "torso"

        # If nothing is active
        if not self.base_active and not self.torso_active and not self.hand_controller.is_active(vr_data):
            vr_command.priority = "none"

        return vr_command

    def get_control_mode(self, vr_data: Dict[str, Any]) -> str:
        """Get the current control mode without processing.

        Args:
            vr_data: Raw VR data

        Returns:
            String indicating active control modes (comma-separated)
        """
        if self.estop_active:
            return "estop"
        
        modes = []
        
        if self.hand_controller.is_active(vr_data):
            modes.append("hands")
        
        if self.base_active:
            modes.append("base")
        
        if self.torso_active:
            modes.append("torso")
        
        if not modes:
            return "none"
        
        return ",".join(modes)  # e.g., "base,torso" or "hands,base,torso"

    def get_stats(self) -> Dict[str, int]:
        """Get controller statistics.

        Returns:
            Dictionary with activation counts
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset controller statistics."""
        for key in self.stats:
            self.stats[key] = 0

    def get_hand_controller(self) -> HandController:
        """Get the hand controller for direct access."""
        return self.hand_controller

    def get_torso_controller(self) -> TorsoController:
        """Get the torso controller for direct access."""
        return self.torso_controller

    def get_base_controller(self) -> BaseController:
        """Get the base controller for direct access."""
        return self.base_controller

    def is_exit_requested(self) -> bool:
        """Check if exit has been requested.

        Returns:
            True if exit was requested
        """
        return self.exit_requested

    def reset_toggle_states(self) -> None:
        """Reset all toggle states (base, torso active status, estop)."""
        self.base_active = False
        self.torso_active = False
        self.estop_active = False
        self._prev_base_button = False
        self._prev_torso_button = False
        self._prev_both_sticks = False
        logger.info("Toggle states reset")

    def get_activation_states(self) -> Dict[str, bool]:
        """Get current activation states of all toggle components.

        Returns:
            Dictionary with activation states
        """
        return {
            "base_active": self.base_active,
            "torso_active": self.torso_active,
            "estop_active": self.estop_active,
        }

    def is_estop_active(self) -> bool:
        """Check if emergency stop is currently active.

        Returns:
            True if estop is active
        """
        return self.estop_active

    def reset_estop(self) -> None:
        """Reset emergency stop state."""
        self.estop_active = False
        self._prev_both_sticks = False
        logger.info("Emergency stop reset")


# Backward compatibility alias (if needed by external code)
InputController = MetaQuestController
MetaConController = MetaQuestController
