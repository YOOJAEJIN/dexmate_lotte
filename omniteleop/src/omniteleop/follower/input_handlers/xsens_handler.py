"""Xsens input handler."""

from typing import Dict, Any, Optional
import time

from dexcomm.codecs import DictDataCodec
from dexcomm import Node
from loguru import logger

from omniteleop.common.schemas import XsensRosData, XsensArmData
from omniteleop.follower.input_handlers.base_handler import (
    BaseInputHandler,
    RobotCommand,
    ArmCommandType,
)

class XsensHandler(BaseInputHandler):
    """Input handler for Xsens reader data.

    Subscribes to Xsens Zenoh pose data and forwards it to component processors.
    """

    def __init__(self, config: Dict[str, Any], namespace: str = "", motion_manager=None):
        """Initialize Xsens handler.

        Args:
            config: Configuration dictionary
            namespace: Optional namespace
            motion_manager: Optional motion manager for state sync
        """
        super().__init__(config, namespace)
        self.motion_manager = motion_manager

        # Communication node
        self.node = Node(name="xsens_handler", namespace=namespace)

        # Xsens Zenoh topic
        self.topic = config.get("topics", {}).get("xsens_ros", "xsens/ros")

        # Latest received Xsens pose data
        self._latest_pose_data: Optional[XsensRosData] = None

    def initialize(self) -> bool:
        """Initialize subscribers.

        Returns:
            True if initialization successful
        """
        try:
            self.setup_subscribers()
            self.initialized = True
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize XsensHandler: {e}")
            return False

    def setup_subscribers(self) -> None:
        """Set up subscription for Xsens TF data."""
        self.node.create_subscriber(
            self.topic,
            self._on_pose_received,
            decoder=DictDataCodec.decode,
        )
        logger.info(f"XsensHandler subscribed to {self.topic}")

    def _on_pose_received(self, data: Dict[str, Any]) -> None:
        """Callback for Xsens Zenoh pose data."""
        try:
            if not isinstance(data, dict):
                return
            
            left_data = data.get("left", {})
            right_data = data.get("right", {})
            
            left_arm = XsensArmData(
                upper_arm_pos=left_data.get("upper_arm_pos", []),
                forearm_pos=left_data.get("forearm_pos", []),
                hand_pos=left_data.get("hand_pos", []),
                hand_rot=left_data.get("hand_rot", []),
                valid=left_data.get("valid", False)
            )
            
            right_arm = XsensArmData(
                upper_arm_pos=right_data.get("upper_arm_pos", []),
                forearm_pos=right_data.get("forearm_pos", []),
                hand_pos=right_data.get("hand_pos", []),
                hand_rot=right_data.get("hand_rot", []),
                valid=right_data.get("valid", False)
            )
            
            pose_data = XsensRosData(
                timestamp_ns=data.get("timestamp_ns", time.time_ns()),
                left=left_arm,
                right=right_arm
            )
            
            with self._data_lock:
                self._latest_pose_data = pose_data
        except Exception as e:
            logger.error(f"Error in XsensHandler callback: {e}")

    def process_inputs(self) -> Optional[RobotCommand]:
        """Process latest Xsens poses into robot commands."""
        with self._data_lock:
            if self._latest_pose_data is None:
                return None
            pose_data = self._latest_pose_data

        command = RobotCommand(timestamp_ns=time.time_ns())
        
        # Populate input components for each arm with XSENS command type
        if pose_data.left.valid:
            command.input_components["left_arm"] = {
                "command_type": ArmCommandType.XSENS,
                "upper_arm_pos": pose_data.left.upper_arm_pos,
                "forearm_pos": pose_data.left.forearm_pos,
                "hand_pos": pose_data.left.hand_pos,
                "hand_rot": pose_data.left.hand_rot,
            }
            
        if pose_data.right.valid:
            command.input_components["right_arm"] = {
                "command_type": ArmCommandType.XSENS,
                "upper_arm_pos": pose_data.right.upper_arm_pos,
                "forearm_pos": pose_data.right.forearm_pos,
                "hand_pos": pose_data.right.hand_pos,
                "hand_rot": pose_data.right.hand_rot,
            }
            
        if command.input_components:
            return command
        return None

    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False
        self.node.shutdown()

    def needs_motion_manager(self) -> bool:
        """Indicate that this handler can use a motion manager."""
        return True
