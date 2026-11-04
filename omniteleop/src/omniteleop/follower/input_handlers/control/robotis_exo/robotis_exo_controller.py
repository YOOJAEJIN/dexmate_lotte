"""Robotis exoskeleton controller with keyboard and joystick input support."""

from typing import Dict, Any, Optional, Callable, List
from loguru import logger

from omniteleop.follower.input_handlers.control.controller import AbstractController
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_base_controller import (
    RobotisExoBaseController,
)
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_torso_controller import (
    TorsoController,
)
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_head_controller import (
    HeadController,
)
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_hand_controller import (
    HandController,
)
from omniteleop.follower.input_handlers.control.commands import ControllerCommand, HandCommand, TorsoCommand
from omniteleop.follower.input_handlers.utils.button_manager import (
    ButtonManager,
    ButtonEvent,
)

class RobotisExoController(AbstractController):
    """Robotis exoskeleton controller with keyboard and joystick support.

    Manages:
    - Joystick input: XY axes for arm control, stick button for gripper
    - Keyboard input: Mode switching, emergency controls
    - Mode toggles: Head, Torso, Base, Fine adjustment (all independent, run simultaneously)
    - Gripper control: Direct motor position control
    - Emergency stop: Keyboard-based (stick button no longer used for estop)
    
    Control Model:
    - ONLY estop (E-Stop) blocks all motion - highest priority
    - All other modes (base, head, torso, gripper) run SIMULTANEOUSLY and INDEPENDENTLY
    - Each mode is toggled on/off via keyboard (H=head, T=torso, B=base, F=fine, O/P=gripper)
    - When estop is active, all modes are disabled
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Robotis exoskeleton controller.

        Args:
            config: Configuration dictionary with component settings
        """
        super().__init__(config)

        self.config = config or {}

        # Initialize component controllers
        self._init_component_controllers()

        # Initialize button manager (minimal - only for joystick stick button)
        self._init_button_manager()

        # Control state
        self.base_active = False
        self.head_active = False
        self.torso_active = False
        self.arm_active = False  # Left and right arm control
        self.fine_adjustment_active = False
        self.estop_active = True  # Start with estop active for safety
        self.exit_requested = False
        self.recording_active = False
        self._last_recording_state = False
        
        # Torso zero position interpolation state (for b key)
        self._torso_zero_current = None  # Current torso position during interpolation
        self._torso_zero_step = 0.1  # 10% per frame towards zero (~1 second to reach)

        # Keyboard callbacks
        self.keyboard_callbacks: Dict[str, List[Callable]] = {}

        # Statistics
        self.stats = {
            "total_updates": 0,
            "arm_activations": 0,
            "base_activations": 0,
            "head_activations": 0,
            "torso_activations": 0,
            "gripper_activations": 0,
            "estop_toggles": 0,
            "keyboard_events": 0,
        }

        logger.info("Robotis exoskeleton controller initialized (keyboard + joystick support)")

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
        
        logger.info(f"JoyConController selected hand_type: {hand_type} based on ROBOT_CONFIG: {robot_config}")
        
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
        self.base_controller = RobotisExoBaseController(
            linear_velocity=base_config.get("linear_velocity", 0.5),
            angular_velocity=base_config.get("angular_velocity", 1.0),
            deadzone=stick_deadzone,
        )

        self.torso_controller = TorsoController(
            sensitivity=torso_config.get("sensitivity", 0.01), deadzone=stick_deadzone
        )

        # Get head configuration
        head_config = self.config.get("head", {})
        self.head_controller = HeadController(
            sensitivity=head_config.get("sensitivity", 0.1), deadzone=stick_deadzone
        )

        self.hand_controller = HandController(
            left_config=left_config,
            right_config=right_config,
            stick_deadzone=stick_deadzone,
        )

    def _init_button_manager(self) -> None:
        """Initialize button manager for joystick input.
        
        Robotis joystick: X, Y axes + stick buttons (left/right)
        - Left stick button (2초 누르기): Emergency stop toggle
        - Right stick button (2초 누르기): Base control toggle
        """
        self.button_manager = ButtonManager()

        # Get button timing configs
        button_config = self.config.get("button_timings", {})
        default_debounce = button_config.get("default_debounce", 0.01)
        stick_debounce = button_config.get("stick_debounce", 0.03)

        # Add individual stick buttons
        self.button_manager.add_button("left_stick", debounce_time=stick_debounce)
        self.button_manager.add_button("right_stick", debounce_time=stick_debounce)

        # Button timing configs
        arm_toggle_duration = button_config.get("arm_toggle_hold_duration", 2.0)
        base_toggle_duration = button_config.get("base_toggle_hold_duration", 2.0)
        combo_grace_period = button_config.get("combo_grace_period", 0.1)

        # Arm control toggle combo: hold left stick for 2 seconds
        self.button_manager.add_combo(
            name="arm_toggle",
            buttons=["left_stick"],
            hold_duration=arm_toggle_duration,
            require_simultaneous=False,
            grace_period=combo_grace_period,
            on_triggered=self._on_arm_toggle_triggered,
        )

        # Base control toggle combo: hold right stick for 2 seconds
        self.button_manager.add_combo(
            name="base_toggle",
            buttons=["right_stick"],
            hold_duration=base_toggle_duration,
            require_simultaneous=False,
            grace_period=combo_grace_period,
            on_triggered=self._on_base_toggle_triggered,
        )
        
        logger.info("Button manager initialized:")
        logger.info("  - Left stick (2s hold) = arm toggle")
        logger.info("  - Right stick (2s hold) = base control toggle")

    def _on_estop_triggered(self) -> None:
        """Handle emergency stop toggle."""
        self.estop_active = not self.estop_active
        self.stats["estop_toggles"] += 1
        state = "ACTIVATED" if self.estop_active else "DEACTIVATED"
        logger.warning(f"Emergency stop {state}")
        
        # When estop is activated, turn off all control modes
        if self.estop_active:
            self.arm_active = False
            self.torso_active = False
            self.head_active = False
            self.base_active = False
            logger.warning("All control modes deactivated (arm, torso, head, base)")

    def _on_arm_toggle_triggered(self) -> None:
        """Handle arm control toggle via left stick button."""
        self._toggle_mode("arm")

    def _on_exit_triggered(self) -> None:
        """Handle exit request."""
        self.estop_active = True
        self.exit_requested = True
        logger.critical("EMERGENCY EXIT ACTIVATED - Shutting down all systems")

    def _on_base_toggle_triggered(self) -> None:
        """Handle base control toggle via right stick button."""
        self._toggle_mode("base")

    def _on_head_toggle_triggered(self) -> None:
        """Handle head control toggle."""
        self._toggle_mode("head")
    
    # ========== Keyboard Integration Methods ==========
    
    def register_keyboard_callback(self, command: str, callback: Callable) -> None:
        """Register a callback for keyboard commands.
        
        Args:
            command: Command name (e.g., "head_toggle", "torso_toggle")
            callback: Function to call when command is triggered
            
        Example:
            controller.register_keyboard_callback("head_toggle", lambda: ...)
        """
        if command not in self.keyboard_callbacks:
            self.keyboard_callbacks[command] = []
        self.keyboard_callbacks[command].append(callback)
        logger.debug(f"Registered keyboard callback for: {command}")
    
    def on_keyboard_event(self, command: str) -> None:
        """Handle keyboard event from keyboard handler.
        
        Args:
            command: Command name from keyboard
            
        Commands:
            - head_toggle: Toggle head control mode
            - torso_toggle: Toggle torso control mode
            - base_toggle: Toggle base control mode
            - fine_toggle: Toggle fine adjustment mode
            - gripper_open: Open grippers
            - gripper_close: Close grippers
            - estop_toggle: Toggle emergency stop
            - exit_request: Request system shutdown
        """
        self.stats["keyboard_events"] += 1
        
        if command == "head_toggle":
            self._toggle_mode("head")
        elif command == "torso_toggle":
            self._toggle_mode("torso")
        elif command == "base_toggle":
            self._toggle_mode("base")
        elif command == "arm_toggle":
            self._toggle_mode("arm")
        elif command == "fine_toggle":
            self._toggle_mode("fine")
        elif command == "gripper_open":
            # Gripper state will be handled in _process_joycon_commands
            # by mapping keyboard O/P directly to joycon data
            logger.info("Gripper OPEN command via keyboard (will be mapped to joycon data)")
        elif command == "gripper_close":
            # Gripper state will be handled in _process_joycon_commands
            # by mapping keyboard O/P directly to joycon data
            logger.info("Gripper CLOSE command via keyboard (will be mapped to joycon data)")
        elif command == "estop_toggle":
            self._on_estop_triggered()
        elif command == "recording_toggle":
            self.recording_active = not self.recording_active
            state = "STARTED" if self.recording_active else "STOPPED"
            logger.info(f"Recording {state}")
        elif command == "gripper_mode_toggle":
            # Gripper mode toggle is handled by keyboard handler
            logger.info("Gripper mode toggled via keyboard handler")
        elif command == "exit_request":
            self._on_exit_triggered()
        else:
            logger.debug(f"Unknown keyboard command: {command}")
        
        # Trigger custom callbacks if registered
        if command in self.keyboard_callbacks:
            for callback in self.keyboard_callbacks[command]:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in keyboard callback for '{command}': {e}")


    def _toggle_mode(self, mode: str) -> None:
        """Toggle a control mode with head/torso/base mutual exclusion (only one active at a time)."""
        # Head, Torso, Base modes are mutually exclusive
        if mode == "base":
            if self.base_active:
                self.base_active = False
                logger.info("Base control DEACTIVATED")
            else:
                # Deactivate head and torso when base is activated
                self.head_active = False
                self.torso_active = False
                self.base_active = True
                self.stats["base_activations"] += 1
                logger.info("Base control ACTIVATED (Head and Torso deactivated)")

        elif mode == "head":
            if self.head_active:
                self.head_active = False
                logger.info("Manual head control DEACTIVATED")
            else:
                # Deactivate torso and base when head is activated
                self.torso_active = False
                self.base_active = False
                self.head_active = True
                self.stats["head_activations"] += 1
                logger.info("Manual head control ACTIVATED (Torso and Base deactivated)")

        elif mode == "torso":
            if self.torso_active:
                self.torso_active = False
                logger.info("Torso control DEACTIVATED")
            else:
                # Deactivate head and base when torso is activated
                self.head_active = False
                self.base_active = False
                self.torso_active = True
                self.stats["torso_activations"] += 1
                logger.info("Torso control ACTIVATED (Head and Base deactivated)")

        elif mode == "arm":
            self.arm_active = not self.arm_active
            if self.arm_active:
                self.stats["arm_activations"] += 1
                logger.info("Arm control ACTIVATED")
            else:
                # When arm is deactivated, also deactivate torso
                self.torso_active = False
                logger.info("Arm control DEACTIVATED (Torso also deactivated)")

        elif mode == "fine":
            self.fine_adjustment_active = not self.fine_adjustment_active
            if self.fine_adjustment_active:
                logger.info("Fine adjustment ACTIVATED")
            else:
                logger.info("Fine adjustment DEACTIVATED")

    def process(self, joycon_data: Dict[str, Any]) -> ControllerCommand:
        """Process Robotis joystick data and return commands.
        
        For Robotis joystick:
        - left/right X, Y axes: arm control (via component controllers)
        - left stick button (2s hold): emergency stop toggle
        - right stick button: customizable (currently unused)
        - Mode switching: via keyboard input (on_keyboard_event)
        - Gripper control: via keyboard or direct command
        
        Priority: ONLY estop blocks everything else. All other modes run simultaneously.

        Args:
            joycon_data: Joystick data with 'left' and 'right' stick info

        Returns:
            RobotCommands with appropriate component commands
        """
        self.stats["total_updates"] += 1

        # Initialize result
        commands = ControllerCommand()

        # Extract button states from joycon_data
        left_data = joycon_data.get("left", {})
        right_data = joycon_data.get("right", {})
        left_buttons = left_data.get("buttons", {})
        right_buttons = right_data.get("buttons", {})

        # Prepare raw button states for button manager
        raw_states = {
            "left_stick": left_buttons.get("stick", False),
            "right_stick": right_buttons.get("stick", False),
        }

        # Update button manager and get events
        events = self.button_manager.update(raw_states)

        # Debug: log button press events
        for button_name, event in events.items():
            if event == ButtonEvent.PRESSED:
                logger.debug(f"Button {button_name} PRESSED")
            elif event == ButtonEvent.RELEASED:
                logger.debug(f"Button {button_name} RELEASED")

        # Check for exit first (highest priority)
        if self.exit_requested:
            commands.estop = True
            commands.exit_requested = True
            commands.priority = "exit"
            return commands

        # Check for active estop (ONLY thing that blocks other modes)
        if self.estop_active:
            commands.estop = True
            commands.priority = "estop"
            return commands

        # Process all active modes SIMULTANEOUSLY (no priority blocking except estop)
        # Each mode can be toggled on/off independently
        
        # Arm control (left and right - if toggled on)
        if self.arm_active:
            # Process both left and right arms through hand controller
            # since arms have hand end-effectors
            commands.hands = self.hand_controller.process(joycon_data)
        
        # Base control (if toggled on)
        if self.base_active:
            commands.base = self.base_controller.process(joycon_data)
        
        # Head control (if toggled on)
        if self.head_active:
            commands.head = self.head_controller.process(joycon_data)
        
        # Torso control (if toggled on)
        if self.torso_active:
            commands.torso = self.torso_controller.process(joycon_data)
        
        # Hand control is ALWAYS AVAILABLE (independent of mode toggles)
        # BUT: gripper only works when estop is NOT active
        # Hand includes gripper control and fine adjustment
        if not self.estop_active and (self.hand_controller.is_active(joycon_data) or self.fine_adjustment_active):
            joycon_data["fine_adjustment_active"] = self.fine_adjustment_active
            commands.hands = self.hand_controller.process(joycon_data)
            self.stats["gripper_activations"] += 1
        
        # Set priority for logging/tracking (list all active modes for display)
        # All modes run independently, so priority lists all active modes
        active_modes = []
        if self.arm_active:
            active_modes.append("arms")
        if self.base_active:
            active_modes.append("base")
        if self.head_active:
            active_modes.append("head")
        if self.torso_active:
            active_modes.append("torso")
        if commands.hands.active:
            active_modes.append("hands")
        
        commands.priority = " + ".join(active_modes) if active_modes else "none"
        
        return commands

    def get_control_mode(self, joycon_data: Dict[str, Any]) -> str:
        """Get the current control mode without processing.

        Args:
            joycon_data: Raw JoyCon data

        Returns:
            String indicating active control mode: "base", "head", "torso", "hands", or "none"
        """
        if self.base_active:
            return "base"
        elif self.head_active:
            return "head"
        elif self.torso_active:
            return "torso"
        elif self.hand_controller.is_active(joycon_data) or self.fine_adjustment_active:
            return "hands"
        else:
            return "none"

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

    def get_base_controller(self):
        """Get the base controller for direct access."""
        return self.base_controller

    def get_head_controller(self) -> HeadController:
        """Get the head controller for direct access."""
        return self.head_controller

    def is_estop_active(self) -> bool:
        """Check if emergency stop is currently active.

        Returns:
            True if estop is active
        """
        return self.estop_active

    def is_exit_requested(self) -> bool:
        """Check if exit has been requested via estop.

        Returns:
            True if exit was requested
        """
        return self.exit_requested

    def reset_estop(self):
        """Reset emergency stop state."""
        self.estop_active = False
        self.exit_requested = False
        logger.info("Emergency stop reset")

    def get_activation_states(self) -> Dict[str, bool]:
        """Get current activation states of all components.

        Returns:
            Dictionary with activation states
        """
        return {
            "base": self.base_active,
            "head": self.head_active,
            "torso": self.torso_active,
            "fine_adjustment": self.fine_adjustment_active,
        }

    def get_recording_command(self) -> Optional[str]:
        """Get the recording command to send based on current state.
        
        Returns:
            "start" if recording should start, "stop" if recording should stop, None otherwise
        """
        # Only return command on state change
        if self._last_recording_state != self.recording_active:
            self._last_recording_state = self.recording_active
            return "start" if self.recording_active else "stop"
        
        return None
