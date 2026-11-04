"""Input handler for exoskeleton arms with JoyCon controllers."""

import threading
import time
from typing import Dict, Any, Optional

from loguru import logger
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

from .base_handler import BaseInputHandler, RobotCommand, CommandMode, ArmCommandType
from omniteleop.common import get_config
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_controller import (
    RobotisExoController,
)
from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_keyboard_handler import (
    RobotisExoKeyboardHandler,
)

class ExoJoyconHandler(BaseInputHandler):
    """Handler for exoskeleton arm inputs combined with JoyCon controllers.

    This handler processes:
    - Joint positions from leader arms (exoskeleton)
    - JoyCon controller inputs for additional controls
    - Combines both into unified robot commands
    """

    def __init__(self, config: Dict[str, Any], namespace: str = ""):
        """Initialize exoskeleton + JoyCon handler.

        Args:
            config: Configuration dictionary
            namespace: Optional namespace for topics
        """
        # Initialize base handler
        BaseInputHandler.__init__(self, config, namespace)

        # Initialize Node for communication
        self.node = Node(name="exo_joycon_handler", namespace=namespace)

        # JoyCon controller for processing joycon inputs
        # Build config for RobotisExoController from component configs
        robotis_controller_config = {
            "base": config.get("base", {}),
            "torso": config.get("torso", {}),
            "hands": config.get("hands", {}),
            "gripper": config.get("gripper", {}),
            # Add any joycon-specific settings (like deadzones)
            **config.get("joycon", {}),
        }
        self.joycon_controller = RobotisExoController(robotis_controller_config)
        self.config = config

        # [추가] Keyboard handler for mode switching and direct commands
        self.keyboard_handler = RobotisExoKeyboardHandler()
        self._setup_keyboard_handler()

        # Latest data (protected by locks)
        self.latest_leader_arm_joints: Optional[Dict[str, Any]] = None
        self.latest_joycon: Optional[Dict[str, Any]] = None
        self._leader_arm_joints_lock = threading.RLock()
        self._joycon_lock = threading.RLock()

        # Subscribers (will be initialized in setup_subscribers)
        self.joints_sub = None
        self.joycon_sub = None

        # Publisher for recorder control
        self.recorder_pub = None

        # Control state
        self._motion_control_started = False
        self.exit_after_publish = False

        logger.info("ExoJoyconHandler initialized")

    def initialize(self) -> bool:
        """Initialize communication and connections.

        Returns:
            True if successful, False otherwise
        """
        if self.initialized:
            return True

        # Set up subscribers
        self.setup_subscribers()

        self.initialized = True
        self.running = True
        logger.info("ExoJoyconHandler initialization complete")
        return True

    def setup_subscribers(self) -> None:
        """Set up Zenoh subscribers for exo joints and JoyCon data."""
        # Subscribe to exoskeleton joints
        joints_topic = self.config.get("topics", {}).get("exo_joints", "exo/joints")
        self.joints_sub = self.node.create_subscriber(
            joints_topic,
            self._on_leader_arm_joints_received,
            decoder=DictDataCodec.decode,
        )

        # Subscribe to JoyCon inputs
        joycon_topic = self.config.get("topics", {}).get("exo_joycon", "exo/joycon")
        self.joycon_sub = self.node.create_subscriber(
            joycon_topic, self._on_joycon_received, decoder=DictDataCodec.decode
        )

        # Create publisher for recorder control
        recorder_topic = get_config().get_topic("recorder_control")
        self.recorder_pub = self.node.create_publisher(recorder_topic, encoder=DictDataCodec.encode)

        resolved_joints = self.node.resolve_topic(joints_topic)
        resolved_joycon = self.node.resolve_topic(joycon_topic)
        resolved_recorder = self.node.resolve_topic(recorder_topic)

        logger.info(f"Subscribed to {resolved_joints} and {resolved_joycon}")
        logger.info(f"Publishing recorder commands to {resolved_recorder}")

    def _setup_keyboard_handler(self) -> None:
        """[추가] Set up keyboard handler callbacks to forward to joycon_controller."""
        # Register callbacks for keyboard commands
        # These will be called when keyboard events occur
        self.keyboard_handler.register_callback(
            "head_toggle",
            lambda: self.joycon_controller.on_keyboard_event("head_toggle")
        )
        self.keyboard_handler.register_callback(
            "torso_toggle",
            lambda: self.joycon_controller.on_keyboard_event("torso_toggle")
        )
        self.keyboard_handler.register_callback(
            "base_toggle",
            lambda: self.joycon_controller.on_keyboard_event("base_toggle")
        )
        self.keyboard_handler.register_callback(
            "arm_toggle",
            lambda: self.joycon_controller.on_keyboard_event("arm_toggle")
        )
        self.keyboard_handler.register_callback(
            "fine_toggle",
            lambda: self.joycon_controller.on_keyboard_event("fine_toggle")
        )
        self.keyboard_handler.register_callback(
            "estop_toggle",
            lambda: self.joycon_controller.on_keyboard_event("estop_toggle")
        )
        self.keyboard_handler.register_callback(
            "gripper_open",
            lambda: self.joycon_controller.on_keyboard_event("gripper_open")
        )
        self.keyboard_handler.register_callback(
            "gripper_close",
            lambda: self.joycon_controller.on_keyboard_event("gripper_close")
        )
        self.keyboard_handler.register_callback(
            "recording_toggle",
            lambda: self._handle_recording_toggle()
        )
        self.keyboard_handler.register_callback(
            "gripper_mode_toggle",
            lambda: self.joycon_controller.on_keyboard_event("gripper_mode_toggle")
        )
        self.keyboard_handler.register_callback(
            "torso_up",
            lambda: None  # Torso up - handled via get_current_state()
        )
        self.keyboard_handler.register_callback(
            "torso_down",
            lambda: None  # Torso down - handled via get_current_state()
        )
        self.keyboard_handler.register_callback(
            "torso_neutral",
            lambda: None  # Torso neutral - handled via get_current_state()
        )
        
        # [추가] Start keyboard handler and pynput listener
        try:
            self.keyboard_handler.start()
            
            # Import and start pynput listener for global keyboard capture
            from omniteleop.follower.input_handlers.control.robotis_exo.robotis_exo_keyboard_handler import (
                create_pynput_listener,
            )
            self.pynput_listener = create_pynput_listener(self.keyboard_handler)
            self.pynput_listener.start()
            logger.info("Pynput keyboard listener started - keyboard input active globally")
        except ImportError:
            logger.warning("pynput not available, keyboard input disabled")
            self.pynput_listener = None
        except Exception as e:
            logger.warning(f"Failed to start keyboard handler: {e}")
            self.pynput_listener = None

    def _on_leader_arm_joints_received(self, joints_data):
        """Handle incoming leader arm joint positions.

        Args:
            joints_data: Deserialized joint data from exoskeleton
        """
        with self._leader_arm_joints_lock:
            self.latest_leader_arm_joints = joints_data

    def _on_joycon_received(self, joycon_data):
        """Handle incoming JoyCon data.

        Args:
            joycon_data: Deserialized JoyCon data
        """
        with self._joycon_lock:
            self.latest_joycon = joycon_data

    def process_inputs(self) -> Optional[RobotCommand]:
        """Process exoskeleton and JoyCon inputs into robot commands.

        Returns:
            RobotCommand if inputs available, None otherwise
        """
        # Get latest data safely with locks
        with self._leader_arm_joints_lock:
            joints = (
                self.latest_leader_arm_joints.copy()
                if self.latest_leader_arm_joints
                else None
            )

        with self._joycon_lock:
            joycon = self.latest_joycon.copy() if self.latest_joycon else None

        if not joints and not joycon:
            return None

        # Initialize new command
        command = RobotCommand(timestamp_ns=time.time_ns())

        # Process JoyCon inputs FIRST (may disable arm_active via H-key)
        if joycon:
            self._process_joycon_commands(joycon, command)

        # Process joint positions from exoskeleton
        # _process_arm_joints internally skips if arm_active is False
        if joints:
            self._process_arm_joints(joints, command)

        # If no JoyCon data, default to emergency stop for safety
        if not joycon and not self._motion_control_started:
            command.safety_flags.emergency_stop = True

        # Update latest command
        self.update_command(command)

        return command

    def _process_arm_joints(self, joints: Dict[str, Any], command: RobotCommand):
        """Process arm joint positions and velocities.

        Args:
            joints: Joint position data (ExoJointData as dict)
            command: Command to update
        """
        # Check if arm control is active from joycon_controller
        if not self.joycon_controller.arm_active:
            # Arm control is disabled, don't add arm commands
            return
        
        # Handle array-based format with velocities
        left_arm_pos = joints.get("left_arm_pos", [])
        left_arm_vel = joints.get("left_arm_vel", [])
        right_arm_pos = joints.get("right_arm_pos", [])
        right_arm_vel = joints.get("right_arm_vel", [])

        # Process left arm positions and velocities using input_components
        # These will be processed by command processor
        if left_arm_pos:
            command.input_components["left_arm"] = {
                "command_type": ArmCommandType.JOINT,
                "mode": CommandMode.ABSOLUTE,
                "pos": left_arm_pos,
            }
            if left_arm_vel:
                command.input_components["left_arm"]["vel"] = left_arm_vel

        # Process right arm positions and velocities
        if right_arm_pos:
            command.input_components["right_arm"] = {
                "command_type": ArmCommandType.JOINT,
                "mode": CommandMode.ABSOLUTE,
                "pos": right_arm_pos,
            }
            if right_arm_vel:
                command.input_components["right_arm"]["vel"] = right_arm_vel

    def _process_joycon_commands(self, joycon: Dict[str, Any], command: RobotCommand):
        """Process JoyCon inputs into robot commands.

        Args:
            joycon: JoyCon input data (includes gripper feedback from leader)
            command: Command to update
        """
        # Add keyboard state to joycon data for HandController and TorsoController
        keyboard_data = self.keyboard_handler.get_current_state()
        joycon["keyboard_gripper_open"] = keyboard_data.get("gripper_open", False)
        joycon["keyboard_gripper_close"] = keyboard_data.get("gripper_close", False)
        joycon["torso_up"] = keyboard_data.get("torso_up", False)
        joycon["torso_down"] = keyboard_data.get("torso_down", False)
        joycon["torso_neutral"] = keyboard_data.get("torso_neutral", False)  # b key for zero position
        joycon["home_shape"] = keyboard_data.get("home_shape", False)  # h key: L-shape position
        
        # Process through centralized controller
        robot_commands = self.joycon_controller.process(joycon)
        # Always update emergency stop state from JoyCon controller
        command.safety_flags.emergency_stop = robot_commands.estop

        # Check for exit request
        if robot_commands.exit_requested:
            command.safety_flags.exit_requested = True
            logger.critical("Exit requested via JoyCon")
            self.exit_after_publish = True

        # Check if motion control should start (when estop is released)
        if not robot_commands.estop and not self._motion_control_started:
            self._motion_control_started = True
            logger.info("Motion control started - estop released")

        # Check for recording commands and publish them
        recording_command = self.joycon_controller.get_recording_command()
        if recording_command and self.recorder_pub:
            recording_msg = {
                "command": recording_command,
                "metadata": {
                    "source": "joycon",
                    "timestamp": time.time_ns(),
                }
            }
            try:
                self.recorder_pub.publish(recording_msg)
                logger.info(f"Published recording command: {recording_command} to recorder/control")
                logger.debug(f"Recording message content: {recording_msg}")
            except Exception as e:
                logger.error(f"Failed to publish recording command: {e}")
        elif recording_command:
            logger.warning(f"Recording command '{recording_command}' generated but no publisher available")

        # Handle all modes independently based on priority string (now contains "mode1 + mode2 + ...")
        # Store priority info for debug display
        command.debug_info = {"active_modes": robot_commands.priority}
        
        # Process base command if active
        if "base" in robot_commands.priority:
            self._handle_base_command(robot_commands.base, command)
        
        # Process head command if active
        if "head" in robot_commands.priority:
            if robot_commands.head.active:
                command.input_components["head"] = {
                    "mode": CommandMode.RELATIVE,
                    "pos": [
                        robot_commands.head.delta_j1,
                        robot_commands.head.delta_j2,
                        robot_commands.head.delta_j3,
                    ],
                }
        
        # Process torso command if active (skip during H-key home_shape - handled below)
        if "torso" in robot_commands.priority and not joycon.get("home_shape", False):
            if robot_commands.torso.active:
                if joycon.get("torso_neutral", False):
                    # B-key: smooth interpolation towards neutral position
                    command.input_components["torso"] = {
                        "mode": CommandMode.ABSOLUTE,
                        "pos": [0.52, 1.05, 0.37],
                        "smooth": True,
                    }
                else:
                    command.input_components["torso"] = {
                        "mode": CommandMode.RELATIVE,
                        "pos": [
                            robot_commands.torso.delta_x,
                            0.0,
                            robot_commands.torso.delta_z,
                        ],
                    }
        
        # H-key: sequential motion - arms first, then torso after arms near L-shape
        # Also disables exo arm control so position is maintained after key release
        if joycon.get("home_shape", False):
            self.joycon_controller.arm_active = False  # Stop exo from overriding arm position
            # Move arms to L-shape
            command.input_components["left_arm"] = {
                "mode": CommandMode.ABSOLUTE,
                "pos": [1.4103, 0.3000, 0.0000, -1.5560, 0.0000, 0.0000, 0.0000],
                "smooth": True,
            }
            command.input_components["right_arm"] = {
                "mode": CommandMode.ABSOLUTE,
                "pos": [-1.4103, -0.3000, 0.0000, -1.5560, 0.0000, 0.0000, 0.0000],
                "smooth": True,
            }
            # Torso: pass home_shape flag - TorsoProcessor decides when to start
            # based on actual arm position proximity to L-shape
            command.input_components["torso"] = {
                "mode": CommandMode.ABSOLUTE,
                "pos": [0.52, 1.05, 0.37],
                "smooth": True,
                "wait_for_arms": True,
            }

        # Process hand commands if active
        if "hands" in robot_commands.priority:
            if robot_commands.hands.active:
                # Process left hand
                if robot_commands.hands.left_positions:
                    command.input_components["left_hand"] = {
                        "mode": CommandMode.ABSOLUTE
                        if robot_commands.hands.left_mode == "absolute"
                        else CommandMode.RELATIVE,
                        "pos": robot_commands.hands.left_positions,
                    }

                # Process right hand
                if robot_commands.hands.right_positions:
                    command.input_components["right_hand"] = {
                        "mode": CommandMode.ABSOLUTE
                        if robot_commands.hands.right_mode == "absolute"
                        else CommandMode.RELATIVE,
                        "pos": robot_commands.hands.right_positions,
                    }

    def _on_recording_toggle(self) -> None:
        """Handle recording toggle command from keyboard.
        
        Sends start/stop commands to MDPRecorder based on current state.
        """
        if not self.recorder_pub:
            logger.warning("Recording publisher not initialized")
            return
        
        # Get current recording state from controller
        recording_active = self.joycon_controller.recording_active
        command = "start" if recording_active else "stop"
        
        # Send recording control command
        try:
            message = {
                "command": command,
                "metadata": {
                    "source": "keyboard",
                    "timestamp": time.time(),
                }
            }
            self.recorder_pub.publish(message)
            logger.info(f"Recording {command.upper()} command sent to MDPRecorder")
        except Exception as e:
            logger.error(f"Failed to send recording command: {e}")

    def _handle_recording_toggle(self) -> None:
        """Handle recording toggle from keyboard handler.
        
        First toggles the controller state, then sends the command to recorder.
        """
        # First toggle the controller state
        self.joycon_controller.on_keyboard_event("recording_toggle")
        # Then send the recording control message
        self._on_recording_toggle()

    def _handle_base_command(self, base_cmd, command: RobotCommand):
        """Handle mobile base velocity commands.

        Args:
            base_cmd: BaseCommand with velocity commands
            command: Command to update
        """
        if not base_cmd.active:
            return

        # Base uses velocity control directly as input component
        # Command processor will pass it through to output
        command.input_components["chassis"] = {
            "vx": base_cmd.vx,  # Forward/backward velocity
            "vy": base_cmd.vy,  # Left/right velocity
            "wz": base_cmd.wz,  # Angular velocity
        }

    def cleanup(self) -> None:
        """Clean up resources and connections."""
        self.running = False

        # [추가] Stop pynput listener first
        if hasattr(self, 'pynput_listener') and self.pynput_listener:
            self.pynput_listener.stop()
            logger.info("Pynput listener stopped")

        # Stop keyboard handler
        if self.keyboard_handler:
            self.keyboard_handler.stop()

        # Node handles communication cleanup
        self.node.shutdown()

        logger.info("ExoJoyconHandler cleaned up")

    def needs_motion_manager(self) -> bool:
        """Check if this handler needs motion manager for processing.

        Returns:
            True since torso IK and hand processing need motion manager
        """
        return True

    def get_exit_requested(self) -> bool:
        """Check if exit has been requested.

        Returns:
            True if exit requested, False otherwise
        """
        return self.exit_after_publish
