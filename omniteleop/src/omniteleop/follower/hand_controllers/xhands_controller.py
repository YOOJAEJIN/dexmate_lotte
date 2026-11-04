import threading
import time
from typing import List, Dict, Any, Optional
from loguru import logger
from omniteleop.follower.hand_controllers.base_hand_controller import BaseHandController
from dexcomm.codecs import JointStateCodec, JointCmdCodec

class XHandsController(BaseHandController):
    """Controller for XHands grippers."""

    def __init__(self, arm, side: str = "unknown"):
        """
        Initialize the XHands controller.

        Args:
            arm: The robot arm object.
            side: 'left' or 'right' (for logging).
        """
        self.arm = arm
        self.side = side
        self.node = getattr(arm, 'node', None)
        if self.node is None:
            # Fallback if arm doesn't have node (though RobotController arms should have it)
            from dexcomm import Node
            self.node = Node(name=f"xhand_{side}")

        self._state_topic = f"state/xhand/{side}"
        self._control_topic = f"control/xhand/{side}"

        self._publisher = None
        self._subscriber = None
        self._last_status: List[float] = []
        self._lock = threading.Lock()

    def activate(self):
        """Initialize Zenoh topics for state and control."""
        logger.info(
            f"{self.side.capitalize()} SimController: Activating on topics {self._state_topic} and {self._control_topic}")

        # Create publisher for control
        self._publisher = self.node.create_publisher(
            self._control_topic,
            encoder=JointCmdCodec.encode
        )

        # Create subscriber for state
        self._subscriber = self.node.create_subscriber(
            self._state_topic,
            self._on_state_received,
            decoder=JointStateCodec.decode
        )

    def _on_state_received(self, data: Any):
        """Callback for received state data."""
        if isinstance(data, dict) and "pos" in data:
            pos = data["pos"]
            if isinstance(pos, list):
                with self._lock:
                    self._last_status = pos
            elif isinstance(pos, (int, float)):
                with self._lock:
                    self._last_status = [float(pos)]
        elif isinstance(data, list):
            with self._lock:
                self._last_status = data

    def set_position(self, pos: List[float]):
        """Publish the received pos to the control topic."""
        if self._publisher is None:
            logger.warning(f"{self.side.capitalize()} SimController: Not activated, cannot set position")
            return

        # Requirement: "set_joint에서는 받은 pos를 /controll/custom_ee에 publish 한다"
        # The issue description says "set_joint", but the base class uses "set_position"
        self._publisher.publish({"pos": pos})

    def open(self):
        """Sends the open command to the gripper."""
        self.set_position([0.0] * 12)

    def close(self):
        """Sends the close command to the gripper."""
        self.set_position([1.0] * 12)

    def get_status(self) -> List[float]:
        """Return the values received from the state topic."""
        with self._lock:
            return list(self._last_status)

    def start_tracking(self, interval: float = 0.05):
        """Tracking is handled by Zenoh subscriber callback, so no extra thread needed here."""
        pass

    def stop_tracking(self):
        """Nothing to stop as tracking uses Zenoh callbacks."""
        pass
