from typing import Optional
from loguru import logger
from omniteleop.follower.hand_controllers.base_hand_controller import BaseHandController
from omniteleop.follower.hand_controllers.robotiq_controller import RobotiqController
from omniteleop.follower.hand_controllers.robotiq_usb_controller import RobotiqUsbController
from omniteleop.follower.hand_controllers.koras_controller import KorasController
from omniteleop.follower.hand_controllers.xhands_controller import XHandsController
from omniteleop.follower.hand_controllers.sim_controller import SimController

class HandControllerFactory:
    """Factory for creating hand controllers."""

    @staticmethod
    def create_hand_controller(hand_type: str, arm, side: str = "unknown") -> Optional[BaseHandController]:
        """
        Create a hand controller based on the type.

        Args:
            hand_type: Type of the hand ('robotiq', 'robotiq_usb', 'koras', 'xhand', 'sim', etc.)
                'robotiq'     -> Robotiq gripper via the robot arm RS485 pass-through bus.
                'robotiq_usb' -> Robotiq gripper via a direct USB-to-RS485 serial adapter.
                'koras'       -> KORAS KR-GCS gripper via the robot arm RS485 pass-through bus.
            arm: The robot arm object.
            side: 'left' or 'right'.

        Returns:
            An instance of BaseHandController or None if hand_type is unknown.
        """
        if hand_type == "robotiq":
            logger.info(f"Creating Robotiq controller for {side} arm")
            return RobotiqController(arm, side=side)
        elif hand_type == "robotiq_usb":
            logger.info(f"Creating Robotiq USB controller for {side} arm")
            return RobotiqUsbController(arm, side=side)
        elif hand_type == "koras":
            logger.info(f"Creating KORAS controller for {side} arm")
            return KorasController(arm, side=side)
        elif hand_type == "xhand":
            logger.info(f"Creating XHands controller for {side} arm")
            return XHandsController(arm, side=side)
        elif hand_type == "sim":
            logger.info(f"Creating Simulation hand controller for {side} arm")
            return SimController(arm, side=side)
        else:
            logger.warning(f"Unknown hand type: {hand_type}")
            return None
