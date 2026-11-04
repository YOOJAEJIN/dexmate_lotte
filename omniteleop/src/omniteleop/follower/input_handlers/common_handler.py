"""Common input handler for processing dashboard and other external commands.

This handler processes commands that should be available across all teleoperation
modes (e.g., dashboard controls, higher-level task commands).
"""

import threading
import time
from typing import Dict, Any, Optional

from loguru import logger
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

from .base_handler import BaseInputHandler, RobotCommand, CommandMode, ArmCommandType

class CommonHandler(BaseInputHandler):
    """Common handler for cross-mode teleoperation commands.

    Processes commands from:
    - Dashboard (head, torso, chassis)
    - Other external control sources
    """

    def __init__(self, config: Dict[str, Any], namespace: str = ""):
        """Initialize common handler.

        Args:
            config: Configuration dictionary
            namespace: Optional namespace for topics
        """
        # Initialize base handler
        super().__init__(config, namespace)

        # Initialize Node for communication
        self.node = Node(name="common_handler", namespace=namespace)

        # Latest data (protected by lock)
        self.latest_external_command: Optional[Dict[str, Any]] = None
        self._command_lock = threading.RLock()

        # Subscriber
        self.command_sub = None

        logger.info("CommonHandler initialized")

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
        logger.info("CommonHandler initialization complete")
        return True

    def setup_subscribers(self) -> None:
        """Set up subscribers for external commands."""
        # Dashboard publishes to robot/safe_commands by default
        # We listen to this to get torso, head, and chassis commands
        command_topic = self.config.get("topics", {}).get("common_input", "common/input")
        self.command_sub = self.node.create_subscriber(
            command_topic,
            self._on_command_received,
            decoder=DictDataCodec.decode,
        )

        resolved_topic = self.node.resolve_topic(command_topic)
        logger.info(f"CommonHandler subscribed to {resolved_topic}")

    def _on_command_received(self, command_data: Dict[str, Any]):
        """Handle incoming external command data.

        Args:
            command_data: Deserialized command data
        """
        logger.info(f"Received external command: {command_data}")
        with self._command_lock:
            self.latest_external_command = command_data

    def process_inputs(self) -> Optional[RobotCommand]:
        """Process external inputs into robot commands.

        Returns:
            RobotCommand if inputs available, None otherwise
        """
        # Get latest data safely with lock
        with self._command_lock:
            external_cmd = (
                self.latest_external_command.copy()
                if self.latest_external_command
                else None
            )
            # Clear data after processing to avoid repeated commands from stale data
            # unless we want latched behavior. Dashboard usually sends periodically 
            # while buttons are held, so clearing is safer.
            self.latest_external_command = None

        if not external_cmd:
            return None

        # Initialize new command
        command = RobotCommand(timestamp_ns=time.time_ns())

        # Extract components (from 'components' key used by dashboard/main.py)
        components = external_cmd.get("components", {})
        
        # Process Torso
        if "torso" in components:
            torso_data = components["torso"]
            torso_input = {
                "mode": CommandMode(torso_data.get("mode", "relative")),
                "pos": torso_data.get("pos", [0.0, 0.0, 0.0]),
            }
            # Joint-space jog flag: 'pos' means joint deltas instead of
            # Cartesian head-frame deltas for the torso IK.
            if torso_data.get("use_joint", False):
                torso_input["use_joint"] = True
            command.input_components["torso"] = torso_input

        # Process Head
        if "head" in components:
            head_data = components["head"]
            mode_str = head_data.get("mode", "relative")
            command.input_components["head"] = {
                "mode": CommandMode(mode_str),
                "pos": head_data.get("pos", [0.0, 0.0, 0.0]),
            }

        # Process Chassis
        if "chassis" in components:
            chassis_data = components["chassis"]
            # Chassis uses direct velocity fields
            command.input_components["chassis"] = {
                "vx": chassis_data.get("vx", 0.0),
                "vy": chassis_data.get("vy", 0.0),
                "wz": chassis_data.get("wz", 0.0),
            }

        # Process Left Hand
        if "left_hand" in components:
            left_hand_data = components["left_hand"]
            command.input_components["left_hand"] = {
                "mode": CommandMode(left_hand_data.get("mode", "absolute")),
                "pos": left_hand_data.get("pos", [0.0]),
            }

        # Process Right Hand
        if "right_hand" in components:
            right_hand_data = components["right_hand"]
            command.input_components["right_hand"] = {
                "mode": CommandMode(right_hand_data.get("mode", "absolute")),
                "pos": right_hand_data.get("pos", [0.0]),
            }

        # Process Left Arm
        if "left_arm" in components:
            left_arm_data = components["left_arm"]
            command.input_components["left_arm"] = self._parse_arm_component(left_arm_data)

        # Process Right Arm
        if "right_arm" in components:
            right_arm_data = components["right_arm"]
            command.input_components["right_arm"] = self._parse_arm_component(right_arm_data)

        # Update latest command
        self.update_command(command)

        return command

    @staticmethod
    def _parse_arm_component(arm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse an arm component from a common-input message.

        Joint commands (default) carry mode/pos. Cartesian jog commands from
        the dashboard carry command_type="ee_pose" plus a delta_xyz (and for
        rotation jogs a delta_R) that is passed through so command_processor
        routes them to the dual-arm IK path instead of direct joint control.
        """
        parsed: Dict[str, Any] = {
            "mode": CommandMode(arm_data.get("mode", "relative")),
            "pos": arm_data.get("pos", [0.0] * 7),
        }

        command_type = arm_data.get("command_type")
        if command_type is not None:
            try:
                parsed["command_type"] = ArmCommandType(command_type)
            except ValueError:
                parsed["command_type"] = ArmCommandType.JOINT
            if "delta_xyz" in arm_data:
                parsed["delta_xyz"] = arm_data["delta_xyz"]
            if "delta_R" in arm_data:
                parsed["delta_R"] = arm_data["delta_R"]
                if "rot_frame" in arm_data:
                    parsed["rot_frame"] = arm_data["rot_frame"]

        return parsed

    def cleanup(self) -> None:
        """Clean up resources and connections."""
        self.running = False
        self.node.shutdown()
        logger.info("CommonHandler cleaned up")
