#!/usr/bin/env python3
"""Robotis leader arm reader for teleoperation system.

Reads joint positions from the Robotis leader arms (exoskeleton) and publishes
them via Zenoh for the teleoperation system.

This is specifically configured for Robotis exoskeleton hardware with DUAL PORTS:
- Left arm: Dynamixel IDs 31-37 on /dev/left_leader
- Right arm: Dynamixel IDs 1-7 on /dev/right_leader
"""

import sys
import time
from dataclasses import asdict
from typing import Dict, Optional, List, Tuple, Any
import tyro
import math
import numpy as np
from numpy.typing import NDArray

# Add dynamixel SDK to path
from dynamixel_sdk import PortHandler, PacketHandler, GroupBulkRead

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm import RateLimiter
from omniteleop.common import ExoJointData, JoyConData, get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from loguru import logger

class RobotisArmReader():
    """Reads Robotis leader arm (exoskeleton) positions and publishes via Zenoh.
    
    Uses dual serial ports:
    - Left port: /dev/left_leader (motors 31-37)
    - Right port: /dev/right_leader (motors 1-7)
    """

    # Robotis Motor configuration constants
    ADDR_PRESENT_VELOCITY = 128
    LEN_PRESENT_VELOCITY = 4
    ADDR_PRESENT_POSITION = 132
    LEN_PRESENT_POSITION = 4
    
    # Joystick/Sensor configuration (Sensorxel Model 600)
    # Actual hardware addresses found via scan (range: 0-4095, center: ~2048)
    ADDR_JOYSTICK_X = 42  # X-axis value (changes 2189-4095 when moved right)
    LEN_JOYSTICK_X = 2
    ADDR_JOYSTICK_Y = 40  # Y-axis value (changes 2196-4095 when moved up)
    LEN_JOYSTICK_Y = 2
    ADDR_JOYSTICK_BUTTON = 27  # Button/Switch value (released: 0, pressed: 256)
    LEN_JOYSTICK_BUTTON = 2
    
    PROTOCOL_VERSION = 2.0

    # Conversion constants
    POSITION_SCALE = 4095.0  # Maximum position value
    VELOCITY_SCALE = 0.229  # Velocity unit in rpm
    RPM_TO_RAD_S = 2 * math.pi / 60.0  # Convert rpm to rad/s

    # Robotis hardware configuration - DUAL PORTS
    DEFAULT_BAUD_RATE = 4000000  # Robotis uses 4Mbps
    DEFAULT_LEFT_PORT = "/dev/ttyUSB0"
    DEFAULT_RIGHT_PORT = "/dev/ttyUSB1"
    DEFAULT_PUBLISH_RATE = 40
    DEFAULT_DEBUG_REFRESH_RATE = 10

    # Robotis motor IDs
    # Left arm: 31-37 (7 joints) on /dev/left_leader
    # Right arm: 1-7 (7 joints) on /dev/right_leader
    # Left gripper: 8 (1 joint) on /dev/left_leader
    # Right gripper: 8 (1 joint) on /dev/right_leader (or separate port)
    # Joystick: ID 10 (analog inputs)
    ROBOTIS_LEFT_ARM_IDS = [31, 32, 33, 34, 35, 36, 37]
    ROBOTIS_RIGHT_ARM_IDS = [1, 2, 3, 4, 5, 6, 7]
    ROBOTIS_LEFT_GRIPPER_ID = 38
    ROBOTIS_RIGHT_GRIPPER_ID = 8
    ROBOTIS_JOYSTICK_ID = 10

    # Publishing condition threshold (radians)
    PUBLISH_THRESHOLD = np.array([0.5, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5])

    def __init__(
        self,
        namespace: str = "",
        left_port: Optional[str] = None,
        right_port: Optional[str] = None,
        baud_rate: Optional[int] = None,
        publish_rate: Optional[int] = None,
        topic: Optional[str] = None,
        debug: bool = False,
        read_velocity: bool = False,
    ) -> None:
        """Initialize Robotis motor reader with dual ports.

        Args:
            namespace: Optional namespace prefix (e.g., "robot1")
            left_port: Override left arm serial port (uses /dev/left_leader if None)
            right_port: Override right arm serial port (uses /dev/right_leader if None)
            baud_rate: Override baud rate (uses Robotis default 4Mbps if None)
            publish_rate: Override publish rate (uses config if None)
            topic: Override topic (uses config if None)
            debug: Enable debug output
            read_velocity: Enable reading and publishing velocity data
        """
        # Initialize Node (namespace is handled automatically by Node)
        self.node = Node(name="robotis_arm_reader", namespace=namespace)

        # Load configuration for non-motor settings
        config = get_config()
        self.config = config

        # Use provided values or defaults
        self.publish_rate = publish_rate or config.get_rate(
            "input_rate", self.DEFAULT_PUBLISH_RATE
        )
        self.arm_topic = topic or config.get_topic("exo_joints")
        self.gripper_joycon_topic = config.get_topic("exo_joycon")
        
        # Try to load ports from configuration first
        teleop_mode = config.get("teleop_mode", "exo_joycon")
        leader_arms_config = config.get("input_handlers", {}).get(teleop_mode, {}).get("leader_arms", {})
        
        # Get left and right arm specific ports if available
        left_arm_config = leader_arms_config.get("left_arm", {})
        right_arm_config = leader_arms_config.get("right_arm", {})
        
        self.left_port = left_port or left_arm_config.get("port") or self.DEFAULT_LEFT_PORT
        self.right_port = right_port or right_arm_config.get("port") or self.DEFAULT_RIGHT_PORT
        self.baud_rate = baud_rate or leader_arms_config.get("baud_rate") or self.DEFAULT_BAUD_RATE

        # Hardware interfaces - DUAL PORTS
        self.left_port_handler: Optional[PortHandler] = None
        self.right_port_handler: Optional[PortHandler] = None
        self.packet_handler: Optional[PacketHandler] = None
        # Left port has 3 bulk readers: arm, gripper, joystick (to avoid timeout)
        self.left_bulk_reader: Optional[GroupBulkRead] = None  # For arm only (31-37)
        self.left_bulk_reader_gripper: Optional[GroupBulkRead] = None  # For gripper (38)
        self.left_bulk_reader_joy: Optional[GroupBulkRead] = None  # For joystick (10)
        # Right port has 2 bulk readers: arm+gripper, joystick
        self.right_bulk_reader: Optional[GroupBulkRead] = None  # For arm + gripper (1-7, 8)
        self.right_bulk_reader_joy: Optional[GroupBulkRead] = None  # For joystick (10)

        # Zenoh publishers
        self.arm_publisher: Optional[Any] = None  # For arm joint data
        self.gripper_joycon_publisher: Optional[Any] = None  # For gripper and joystick data

        # Motor management
        self.motors: Dict[int, Dict[str, Any]] = {}
        self.motor_positions: Dict[int, float] = {}
        
        # Gripper state
        self.gripper_data: Dict[str, float] = {"left": 0.0, "right": 0.0}
        
        # Joystick state - both left and right joysticks with X, Y, Button
        # Each has X, Y, Button values
        self.joystick_data: Dict[str, float] = {
            "left_x": 0.0, "left_y": 0.0, "left_button": 0,
            "right_x": 0.0, "right_y": 0.0, "right_button": 0
        }

        # Initial positions for publishing condition
        self.init_pos: Dict[str, NDArray[np.float64]] = {}

        # Rate limiter for precise timing
        self.rate_limiter = RateLimiter(self.publish_rate)

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "RobotisArm",
                self.publish_rate,
                refresh_rate=self.DEFAULT_DEBUG_REFRESH_RATE,
            )

        # Velocity reading option
        self.read_velocity = read_velocity

        # Joint limit constraints (5-degree safety margin from URDF limits)
        # Left arm: 7 joints with individual limits
        self.left_arm_limits = [
            [-2.988, 2.988],    # L_arm_j1  (orig: ±176°)
            [-0.370, 1.470],    # L_arm_j2  (orig: -26°~89°)
            [-2.988, 2.988],    # L_arm_j3  (orig: ±176°)
            [-2.988, 0.161],    # L_arm_j4  (orig: -176°~14°)
            [-2.988, 2.988],    # L_arm_j5  (orig: ±176°)
            [-1.313, 1.313],    # L_arm_j6  (orig: ±80°)
            [-1.295, 1.034],    # L_arm_j7  (orig: -79°~64°)
        ]
        
        # Right arm: 7 joints with mirrored limits
        self.right_arm_limits = [
            [-2.988, 2.988],    # R_arm_j1  (orig: ±176°)
            [-1.470, 0.370],    # R_arm_j2  (orig: -89°~26°, mirrored)
            [-2.988, 2.988],    # R_arm_j3  (orig: ±176°)
            [-2.988, 0.161],    # R_arm_j4  (orig: -176°~14°)
            # [-0.161, 2.988],    # R_arm_j4  (orig: -14°~176°, mirrored) 
            [-2.988, 2.988],    # R_arm_j5  (orig: ±176°)
            [-1.313, 1.313],    # R_arm_j6  (orig: ±80°)
            [-1.034, 1.295],    # R_arm_j7  (orig: -64°~79°, mirrored)
        ]
        
        # Torso: 3 joints (x, y, z movement)
        # URDF limits with 5-degree safety margin applied
        # 5 degrees = 0.0873 radians
        self.torso_limits = [
            [0.0873, 1.4827],   # torso_j1 (URDF: [0, 1.570])
            [0.0873, 3.0537],   # torso_j2 (URDF: [0, 3.141])
            [-1.4827, 1.4827],  # torso_j3 (URDF: [-1.570, 1.570])
        ]

        logger.info(
            f"Robotis arm reader configured (left_port={self.left_port}, "
            f"right_port={self.right_port}, baud={self.baud_rate}) "
            f"(velocity={'enabled' if read_velocity else 'disabled'})"
        )

    def initialize(self) -> bool:
        """Initialize hardware and Zenoh communication.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize Zenoh publishers through Node (namespace handled automatically)
            self.arm_publisher = self.node.create_publisher(
                self.arm_topic, encoder=DictDataCodec.encode
            )
            self.gripper_joycon_publisher = self.node.create_publisher(
                self.gripper_joycon_topic, encoder=DictDataCodec.encode
            )

            # Initialize Dynamixel hardware (dual ports for left and right arms)
            if not self._initialize_hardware():
                return False

            # Connect to Robotis motors on both ports
            self._connect_robotis_motors()
            
            # Connect to gripper sensors
            self._connect_grippers()
            
            # Connect to joystick sensor if available
            self._connect_joystick()

            # Setup bulk readers for both ports
            self._setup_bulk_readers()

            total_motors = len(self.motors)
            if total_motors == 0:
                logger.error("No Robotis motors connected on either port")
                return False

            logger.success(f"Initialized with {total_motors} Robotis motors/sensors total")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _initialize_hardware(self) -> bool:
        """Initialize Dynamixel hardware interfaces on both ports.

        Returns:
            True if at least one port opened successfully, False if both fail
        """
        try:
            success = False

            # Initialize left port
            try:
                self.left_port_handler = PortHandler(self.left_port)
                self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

                if not self.left_port_handler.openPort():
                    logger.error(f"Failed to open left port {self.left_port}")
                    self.left_port_handler = None
                elif not self.left_port_handler.setBaudRate(self.baud_rate):
                    logger.error(f"Failed to set baud rate on left port {self.baud_rate}")
                    self.left_port_handler.closePort()
                    self.left_port_handler = None
                else:
                    logger.info(f"Opened left port {self.left_port} @ {self.baud_rate} baud")
                    success = True
            except Exception as e:
                logger.error(f"Failed to initialize left port: {e}")
                self.left_port_handler = None

            # Initialize right port
            try:
                self.right_port_handler = PortHandler(self.right_port)
                self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

                if not self.right_port_handler.openPort():
                    logger.error(f"Failed to open right port {self.right_port}")
                    self.right_port_handler = None
                elif not self.right_port_handler.setBaudRate(self.baud_rate):
                    logger.error(f"Failed to set baud rate on right port {self.baud_rate}")
                    self.right_port_handler.closePort()
                    self.right_port_handler = None
                else:
                    logger.info(f"Opened right port {self.right_port} @ {self.baud_rate} baud")
                    success = True
            except Exception as e:
                logger.error(f"Failed to initialize right port: {e}")
                self.right_port_handler = None

            if not success:
                logger.error("Failed to initialize any ports!")
                return False

            return True

        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return False

    def _connect_robotis_motors(self) -> None:
        """Connect to all Robotis motors (left and right arms on separate ports)."""
        self.init_pos = {}
        
        # Connect left arm (IDs 31-37) on left port
        if self.left_port_handler:
            logger.info("Connecting left arm (IDs 31-37) on left port...")
            self._connect_arm_motors("left_arm", self.ROBOTIS_LEFT_ARM_IDS, "left")
            self.init_pos["left_arm"] = self.config.get_init_pos("left_arm")
        else:
            logger.warning("Left port not available, skipping left arm")
        
        # Connect right arm (IDs 1-7) on right port
        if self.right_port_handler:
            logger.info("Connecting right arm (IDs 1-7) on right port...")
            self._connect_arm_motors("right_arm", self.ROBOTIS_RIGHT_ARM_IDS, "right")
            self.init_pos["right_arm"] = self.config.get_init_pos("right_arm")
        else:
            logger.warning("Right port not available, skipping right arm")

    def _connect_arm_motors(self, arm_name: str, motor_ids: List[int], port_side: str) -> None:
        """Connect to motors for a specific arm on the appropriate port.

        Args:
            arm_name: Name of the arm
            motor_ids: List of motor IDs for this arm
            port_side: "left" or "right" to select port
        """
        port_handler = (
            self.left_port_handler if port_side == "left" else self.right_port_handler
        )
        
        if not port_handler:
            logger.warning(f"Port handler for {port_side} is None, cannot connect {arm_name}")
            return

        for motor_id in motor_ids:
            try:
                # Ping motor
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    port_handler, motor_id
                )

                if dxl_comm_result == 0:  # COMM_SUCCESS
                    self.motors[motor_id] = {
                        "connected": True,
                        "model": dxl_model_number,
                        "error": "",
                        "arm": arm_name,
                        "port": port_side,
                    }
                    logger.info(
                        f"{arm_name} Motor {motor_id}: Connected (Model: {dxl_model_number})"
                    )
                else:
                    logger.warning(f"{arm_name} Motor {motor_id}: Not found")

            except Exception as e:
                logger.warning(f"{arm_name} Motor {motor_id}: Connection error - {e}")

    def _connect_grippers(self) -> None:
        """Connect to gripper sensors (ID 8) on both ports."""
        # Connect left gripper on left port
        if self.left_port_handler:
            try:
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.left_port_handler, self.ROBOTIS_LEFT_GRIPPER_ID
                )
                if dxl_comm_result == 0:  # COMM_SUCCESS
                    self.motors[self.ROBOTIS_LEFT_GRIPPER_ID] = {
                        "connected": True,
                        "model": dxl_model_number,
                        "error": "",
                        "arm": "left_gripper",
                        "port": "left",
                    }
                    logger.info(
                        f"Left Gripper {self.ROBOTIS_LEFT_GRIPPER_ID}: Connected (Model: {dxl_model_number})"
                    )
                else:
                    logger.debug(f"Left Gripper {self.ROBOTIS_LEFT_GRIPPER_ID}: Not found")
            except Exception as e:
                logger.debug(f"Left Gripper connection error: {e}")

        # Connect right gripper on right port
        if self.right_port_handler:
            try:
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.right_port_handler, self.ROBOTIS_RIGHT_GRIPPER_ID
                )
                if dxl_comm_result == 0:  # COMM_SUCCESS
                    self.motors[self.ROBOTIS_RIGHT_GRIPPER_ID] = {
                        "connected": True,
                        "model": dxl_model_number,
                        "error": "",
                        "arm": "right_gripper",
                        "port": "right",
                    }
                    logger.info(
                        f"Right Gripper {self.ROBOTIS_RIGHT_GRIPPER_ID}: Connected (Model: {dxl_model_number})"
                    )
                else:
                    logger.debug(f"Right Gripper {self.ROBOTIS_RIGHT_GRIPPER_ID}: Not found")
            except Exception as e:
                logger.debug(f"Right Gripper connection error: {e}")

    def _connect_joystick(self) -> None:
        """Connect to joystick sensors (ID 10) on both left and right ports."""
        # Connect left joystick on left port
        if self.left_port_handler:
            try:
                # Ping joystick sensor
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.left_port_handler, self.ROBOTIS_JOYSTICK_ID
                )

                if dxl_comm_result == 0:  # COMM_SUCCESS
                    self.motors[f"{self.ROBOTIS_JOYSTICK_ID}_left"] = {
                        "connected": True,
                        "model": dxl_model_number,
                        "error": "",
                        "arm": "joystick",
                        "port": "left",
                        "motor_id": self.ROBOTIS_JOYSTICK_ID,
                    }
                    logger.info(
                        f"Left Joystick Sensor {self.ROBOTIS_JOYSTICK_ID}: Connected (Model: {dxl_model_number})"
                    )
                else:
                    logger.debug(f"Left Joystick Sensor {self.ROBOTIS_JOYSTICK_ID}: Not found")

            except Exception as e:
                logger.debug(f"Left Joystick Sensor {self.ROBOTIS_JOYSTICK_ID}: Connection error - {e}")

        # Connect right joystick on right port
        if self.right_port_handler:
            try:
                # Ping joystick sensor
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.right_port_handler, self.ROBOTIS_JOYSTICK_ID
                )

                if dxl_comm_result == 0:  # COMM_SUCCESS
                    self.motors[f"{self.ROBOTIS_JOYSTICK_ID}_right"] = {
                        "connected": True,
                        "model": dxl_model_number,
                        "error": "",
                        "arm": "joystick",
                        "port": "right",
                        "motor_id": self.ROBOTIS_JOYSTICK_ID,
                    }
                    logger.info(
                        f"Right Joystick Sensor {self.ROBOTIS_JOYSTICK_ID}: Connected (Model: {dxl_model_number})"
                    )
                else:
                    logger.debug(f"Right Joystick Sensor {self.ROBOTIS_JOYSTICK_ID}: Not found")

            except Exception as e:
                logger.debug(f"Right Joystick Sensor {self.ROBOTIS_JOYSTICK_ID}: Connection error - {e}")

    def _setup_bulk_readers(self) -> None:
        """Setup bulk readers for both left and right ports.
        
        Left port has 3 readers: arm (31-37), gripper (38), joystick (10)
        Right port has 2 readers: arm+gripper (1-7, 8), joystick (10)
        """
        # Setup left bulk reader for ARM ONLY (ID 31-37)
        if self.left_port_handler and any(
            m["port"] == "left" and m["arm"] == "left_arm"
            for m in self.motors.values() if m["connected"]
        ):
            try:
                self.left_bulk_reader = GroupBulkRead(
                    self.left_port_handler, self.packet_handler
                )
                left_motor_count = 0
                for motor_id, info in self.motors.items():
                    if info["connected"] and info["port"] == "left" and info["arm"] == "left_arm":
                        self._add_motor_to_bulk_reader(motor_id, "left_arm")
                        left_motor_count += 1
                logger.info(f"Left bulk reader (arm) configured with {left_motor_count} motors")
            except Exception as e:
                logger.error(f"Left arm bulk reader setup failed: {e}")
                self.left_bulk_reader = None

        # Setup left bulk reader for GRIPPER ONLY (ID 38)
        if self.left_port_handler and any(
            m["port"] == "left" and m["arm"] == "left_gripper"
            for m in self.motors.values() if m["connected"]
        ):
            try:
                self.left_bulk_reader_gripper = GroupBulkRead(
                    self.left_port_handler, self.packet_handler
                )
                gripper_count = 0
                for motor_id, info in self.motors.items():
                    if info["connected"] and info["port"] == "left" and info["arm"] == "left_gripper":
                        self._add_motor_to_bulk_reader(motor_id, "left_gripper")
                        gripper_count += 1
                logger.info(f"Left bulk reader (gripper) configured with {gripper_count} motors")
            except Exception as e:
                logger.error(f"Left gripper bulk reader setup failed: {e}")
                self.left_bulk_reader_gripper = None

        # Note: Joystick is read directly using readTxRx, not via bulk reader
        # This avoids issues with bulk reader configuration for sensor data
        logger.info("Joystick readers will use direct serial read (readTxRx)")

        # Setup right bulk reader for ARM + GRIPPER (ID 1-7, 8)
        if self.right_port_handler and any(
            m["port"] == "right" and m["arm"] in ["right_arm", "right_gripper"]
            for m in self.motors.values() if m["connected"]
        ):
            try:
                self.right_bulk_reader = GroupBulkRead(
                    self.right_port_handler, self.packet_handler
                )
                right_motor_count = 0
                for motor_id, info in self.motors.items():
                    if info["connected"] and info["port"] == "right" and info["arm"] in ["right_arm", "right_gripper"]:
                        self._add_motor_to_bulk_reader(motor_id, "right_arm")
                        right_motor_count += 1
                logger.info(f"Right bulk reader (arm+gripper) configured with {right_motor_count} motors")
            except Exception as e:
                logger.error(f"Right bulk reader setup failed: {e}")
                self.right_bulk_reader = None

        # Note: Joystick is read directly using readTxRx, not via bulk reader
        # This avoids issues with bulk reader configuration for sensor data
        logger.info("Joystick readers will use direct serial read (readTxRx)")

    def _add_motor_to_bulk_reader(self, motor_id, port_side: str) -> None:
        """Add a motor to the appropriate bulk reader.

        Args:
            motor_id: ID of the motor to add (can be string like "10_left" or int)
            port_side: "left_arm", "left_gripper", "right_arm"
        """
        bulk_reader_map = {
            "left_arm": self.left_bulk_reader,
            "left_gripper": self.left_bulk_reader_gripper,
            "right_arm": self.right_bulk_reader,
        }
        
        bulk_reader = bulk_reader_map.get(port_side)
        actual_id = int(str(motor_id).split('_')[0]) if isinstance(motor_id, str) else motor_id
        
        if not bulk_reader:
            return

        # For regular motors, add position register
        success = bulk_reader.addParam(
            actual_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
        )

        if not success:
            logger.warning(
                f"Failed to add position for motor {motor_id} to {port_side} bulk reader"
            )
        else:
            logger.debug(f"Added position for motor {motor_id} to {port_side} bulk reader")

        # Add velocity register if enabled
        if self.read_velocity:
            success = bulk_reader.addParam(
                actual_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY
            )

            if not success:
                logger.warning(
                    f"Failed to add velocity for motor {motor_id} to {port_side} bulk reader"
                )
            else:
                logger.debug(f"Added velocity for motor {motor_id} to {port_side} bulk reader")

    def _read_positions_velocities(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Read all motor positions and velocities from both ports and return arm arrays.

        Returns:
            Tuple of (left_arm_positions, left_arm_velocities, right_arm_positions, right_arm_velocities)
        """
        left_arm_positions: List[float] = []
        left_arm_velocities: List[float] = []
        right_arm_positions: List[float] = []
        right_arm_velocities: List[float] = []
        left_gripper_positions: List[float] = []
        right_gripper_positions: List[float] = []

        # Read from left port (arm only)
        if self.left_bulk_reader:
            comm_result = self.left_bulk_reader.txRxPacket()
            if comm_result == 0:  # COMM_SUCCESS
                left_arm_positions, left_arm_velocities = self._read_arm_pos_vel(
                    "left_arm", self.ROBOTIS_LEFT_ARM_IDS, self.left_bulk_reader
                )
            else:
                logger.debug(f"Left arm bulk read failed: {comm_result}")

        # Read from left port (gripper only)
        if self.left_bulk_reader_gripper:
            comm_result = self.left_bulk_reader_gripper.txRxPacket()
            if comm_result == 0:  # COMM_SUCCESS
                left_gripper_positions = self._read_gripper_pos(
                    self.ROBOTIS_LEFT_GRIPPER_ID, self.left_bulk_reader_gripper
                )
            else:
                logger.debug(f"Left gripper bulk read failed: {comm_result}")

        # Read from left port (joystick only) - using direct serial read
        if self.left_port_handler:
            self._read_joystick_data(self.left_port_handler, "left")

        # Read from right port (arm + gripper)
        if self.right_bulk_reader:
            comm_result = self.right_bulk_reader.txRxPacket()
            if comm_result == 0:  # COMM_SUCCESS
                right_arm_positions, right_arm_velocities = self._read_arm_pos_vel(
                    "right_arm", self.ROBOTIS_RIGHT_ARM_IDS, self.right_bulk_reader
                )
                # Read right gripper data
                right_gripper_positions = self._read_gripper_pos(
                    self.ROBOTIS_RIGHT_GRIPPER_ID, self.right_bulk_reader
                )
            else:
                logger.debug(f"Right arm+gripper bulk read failed: {comm_result}")

        # Read from right port (joystick only) - using direct serial read
        if self.right_port_handler:
            self._read_joystick_data(self.right_port_handler, "right")

        return (
            left_arm_positions,
            left_arm_velocities,
            right_arm_positions,
            right_arm_velocities,
            left_gripper_positions,
            right_gripper_positions,
        )

    def _read_arm_pos_vel(
        self, arm_name: str, motor_ids: List[int], bulk_reader: GroupBulkRead
    ) -> Tuple[List[float], List[float]]:
        """Read positions and velocities for a single arm from its bulk reader.

        Args:
            arm_name: Name of the arm
            motor_ids: List of motor IDs for this arm
            bulk_reader: The bulk reader for this port

        Returns:
            Tuple of (joint positions in radians, joint velocities in rad/s)
        """
        positions: List[float] = []
        velocities: List[float] = []
        
        # Get polarity corrections from config if available
        motor_polarities = [1] * len(motor_ids)
        try:
            arm_configs = self.config.get_leader_arms()
            if arm_name in arm_configs:
                motor_polarities = arm_configs[arm_name].get("motor_polarities", motor_polarities)
        except:
            pass

        for i, motor_id in enumerate(motor_ids):
            position_rad, velocity_rad_s = self._read_motor_data(
                motor_id, i, motor_polarities, bulk_reader
            )
            positions.append(position_rad)
            velocities.append(velocity_rad_s)

        return positions, velocities

    def _read_motor_data(
        self, motor_id: int, motor_index: int, motor_polarities: List[int],
        bulk_reader: GroupBulkRead
    ) -> Tuple[float, float]:
        """Read position and velocity data for a single motor.

        Args:
            motor_id: ID of the motor to read
            motor_index: Index of the motor in the arm
            motor_polarities: List of polarity corrections
            bulk_reader: The bulk reader for this motor's port

        Returns:
            Tuple of (position in radians, velocity in rad/s)
        """
        position_rad = 0.0
        velocity_rad_s = 0.0

        if motor_id not in self.motors or not self.motors[motor_id]["connected"]:
            return position_rad, velocity_rad_s

        try:
            # Read position data
            position_rad = self._read_motor_position(motor_id, bulk_reader)

            # Read velocity data if enabled
            if self.read_velocity:
                velocity_rad_s = self._read_motor_velocity(motor_id, bulk_reader)

            # Apply polarity correction from config
            if motor_index < len(motor_polarities):
                position_rad *= motor_polarities[motor_index]
                velocity_rad_s *= motor_polarities[motor_index]

        except Exception as e:
            logger.warning(f"Error reading motor {motor_id}: {e}")
            # Keep default values (0.0)

        return position_rad, velocity_rad_s

    def _read_motor_position(self, motor_id: int, bulk_reader: GroupBulkRead) -> float:
        """Read position data for a specific motor.

        Args:
            motor_id: ID of the motor to read
            bulk_reader: The bulk reader for this motor's port

        Returns:
            Position in radians
        """
        if not bulk_reader.isAvailable(
            motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
        ):
            return 0.0

        # Get raw position
        raw_pos = bulk_reader.getData(
            motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
        )

        # Convert position to radians
        position_rad = (raw_pos / self.POSITION_SCALE) * 2 * math.pi - math.pi
        
        # Apply -90 degree offset for motor 1 (right arm) and motor 31 (left arm)
        if motor_id == 1:
            # position_rad += 0.6
            position_rad += math.pi / 2
            position_rad = -position_rad
        if motor_id == 3 or motor_id == 5 or motor_id == 6:
            position_rad = -position_rad


        if motor_id == 31:
            # position_rad += 0.6
            position_rad += math.pi / 2
        if motor_id == 33 or motor_id == 35:
            position_rad = -position_rad

        return position_rad

    def _read_motor_velocity(self, motor_id: int, bulk_reader: GroupBulkRead) -> float:
        """Read velocity data for a specific motor.

        Args:
            motor_id: ID of the motor to read
            bulk_reader: The bulk reader for this motor's port

        Returns:
            Velocity in rad/s
        """
        if not bulk_reader.isAvailable(
            motor_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY
        ):
            return 0.0

        raw_vel = bulk_reader.getData(
            motor_id, self.ADDR_PRESENT_VELOCITY, self.LEN_PRESENT_VELOCITY
        )

        # Convert velocity to rad/s
        # Velocity unit is 0.229 rpm, convert to rad/s
        velocity_rad_s = raw_vel * self.VELOCITY_SCALE * self.RPM_TO_RAD_S
        return velocity_rad_s

    def _read_joystick_data(self, port_handler, port_side: str = "left") -> None:
        """Read joystick (ID 10) X, Y, Button values using direct serial read.

        Args:
            port_handler: The port handler for this port
            port_side: "left" or "right" to identify which joystick to read
        """
        # Find the joystick key for this port
        joystick_key = f"{self.ROBOTIS_JOYSTICK_ID}_{port_side}"
        if joystick_key not in self.motors:
            logger.debug(f"Joystick {joystick_key} not found in motors")
            return

        if not self.motors[joystick_key]["connected"]:
            return

        try:
            # Read joystick X value (address 40, 2 bytes, range 0-4095, center ~2048)
            data_list, result_code, error_code = self.packet_handler.readTxRx(
                port_handler, self.ROBOTIS_JOYSTICK_ID, self.ADDR_JOYSTICK_X, self.LEN_JOYSTICK_X
            )
            
            if result_code == 0 and isinstance(data_list, list) and len(data_list) >= 2:
                raw_x = data_list[0] | (data_list[1] << 8)
                # Sensorxel range: 0-4095 (12-bit), center at ~2048
                # Normalize to -1.0 to 1.0
                x_value = (raw_x - 2048.0) / 2048.0
                x_value = max(-1.0, min(1.0, x_value))
            else:
                x_value = 0.0

            # Read joystick Y value (address 42, 2 bytes, range 0-4095, center ~2048)
            data_list, result_code, error_code = self.packet_handler.readTxRx(
                port_handler, self.ROBOTIS_JOYSTICK_ID, self.ADDR_JOYSTICK_Y, self.LEN_JOYSTICK_Y
            )
            
            if result_code == 0 and isinstance(data_list, list) and len(data_list) >= 2:
                raw_y = data_list[0] | (data_list[1] << 8)
                # Sensorxel range: 0-4095 (12-bit), center at ~2048
                # Normalize to -1.0 to 1.0
                y_value = (raw_y - 2048.0) / 2048.0
                y_value = max(-1.0, min(1.0, y_value))
            else:
                y_value = 0.0

            # Read joystick Button value (address 27, 1 byte)
            # Button: 0 when released, 256 when pressed
            data_list, result_code, error_code = self.packet_handler.readTxRx(
                port_handler, self.ROBOTIS_JOYSTICK_ID, self.ADDR_JOYSTICK_BUTTON, self.LEN_JOYSTICK_BUTTON
            )
            
            if result_code == 0 and isinstance(data_list, list) and len(data_list) >= 1:
                raw_button = data_list[0] | (data_list[1] << 8) if len(data_list) >= 2 else data_list[0]
                # Threshold: values > 128 mean button is pressed (0=released, 256=pressed)
                button_value = 1 if raw_button > 128 else 0
            else:
                button_value = 0
            
            # Store with port side identifier
            if port_side == "left":
                self.joystick_data["left_x"] = x_value
                self.joystick_data["left_y"] = y_value
                self.joystick_data["left_button"] = button_value
            else:  # right
                self.joystick_data["right_x"] = x_value
                self.joystick_data["right_y"] = y_value
                self.joystick_data["right_button"] = button_value

        except Exception as e:
            logger.debug(f"Error reading joystick data from {port_side} port: {e}")

    def _read_gripper_pos(self, gripper_id: int, bulk_reader: GroupBulkRead) -> List[float]:
        """Read gripper joint position.

        Args:
            gripper_id: ID of the gripper (8 for both left and right)
            bulk_reader: The bulk reader for this port

        Returns:
            List with gripper position in radians (single joint)
        """
        positions: List[float] = []

        if gripper_id not in self.motors or not self.motors[gripper_id]["connected"]:
            return positions

        try:
            if bulk_reader.isAvailable(
                gripper_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
            ):
                raw_pos = bulk_reader.getData(
                    gripper_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
                )
                # Convert position to radians (same as arm joints)
                position_rad = (raw_pos / self.POSITION_SCALE) * 2 * math.pi - math.pi
                positions.append(position_rad)
                
                # Store in gripper_data dict
                if gripper_id == self.ROBOTIS_LEFT_GRIPPER_ID:
                    self.gripper_data["left"] = position_rad
                elif gripper_id == self.ROBOTIS_RIGHT_GRIPPER_ID:
                    self.gripper_data["right"] = position_rad

        except Exception as e:
            logger.debug(f"Error reading gripper {gripper_id}: {e}")

        return positions

    def _apply_joint_limits(
        self,
        left_arm_pos: List[float],
        right_arm_pos: List[float],
        torso_pos: List[float] = None,
    ) -> tuple[List[float], List[float], List[float]]:
        """Apply joint position limits to arm and torso positions for safety.

        Args:
            left_arm_pos: Left arm joint positions (7 joints)
            right_arm_pos: Right arm joint positions (7 joints)
            torso_pos: Torso joint positions (3 joints), optional

        Returns:
            Tuple of (limited_left_arm_pos, limited_right_arm_pos, limited_torso_pos)
        """
        # Apply limits to left arm
        limited_left = []
        for i, pos in enumerate(left_arm_pos):
            if i < len(self.left_arm_limits):
                lower, upper = self.left_arm_limits[i]
                clipped = np.clip(pos, lower, upper)
                # Convert numpy float64 to Python float for JSON serialization
                limited_left.append(float(clipped))
            else:
                limited_left.append(pos)

        # Apply limits to right arm
        limited_right = []
        for i, pos in enumerate(right_arm_pos):
            if i < len(self.right_arm_limits):
                lower, upper = self.right_arm_limits[i]
                clipped = np.clip(pos, lower, upper)
                # Convert numpy float64 to Python float for JSON serialization
                limited_right.append(float(clipped))
            else:
                limited_right.append(pos)

        # Apply limits to torso
        limited_torso = []
        if torso_pos:
            for i, pos in enumerate(torso_pos):
                if i < len(self.torso_limits):
                    lower, upper = self.torso_limits[i]
                    clipped = np.clip(pos, lower, upper)
                    # Convert numpy float64 to Python float for JSON serialization
                    limited_torso.append(float(clipped))
                else:
                    limited_torso.append(float(pos))
        
        return limited_left, limited_right, limited_torso

    def _create_exo_joint_data(
        self,
        left_arm_pos: List[float],
        left_arm_vel: List[float],
        right_arm_pos: List[float],
        right_arm_vel: List[float],
    ) -> ExoJointData:
        """Create ExoJointData object from arm data only.

        Args:
            left_arm_pos: Left arm joint positions
            left_arm_vel: Left arm joint velocities
            right_arm_pos: Right arm joint positions
            right_arm_vel: Right arm joint velocities

        Returns:
            ExoJointData object with arm joint data
        """
        return ExoJointData(
            timestamp_ns=time.time_ns(),
            left_arm_pos=left_arm_pos,
            left_arm_vel=left_arm_vel if self.read_velocity else [],
            right_arm_pos=right_arm_pos,
            right_arm_vel=right_arm_vel if self.read_velocity else [],
        )

    def _create_gripper_joycon_data(
        self,
        left_gripper_pos: List[float],
        right_gripper_pos: List[float],
    ) -> JoyConData:
        """Create JoyConData with joystick input and gripper feedback.
        
        Unified data structure for Robotis exoskeleton:
        - Joystick: X, Y axes + stick button (per side)
        - Gripper: Current position feedback (motor reading)

        Args:
            left_gripper_pos: Left gripper current position (list with 1 element)
            right_gripper_pos: Right gripper current position (list with 1 element)

        Returns:
            JoyConData with joystick and gripper data
        """
        # Extract gripper positions if available
        left_gripper_value = left_gripper_pos[0] if left_gripper_pos else 0.0
        right_gripper_value = right_gripper_pos[0] if right_gripper_pos else 0.0
        
        return JoyConData(
            timestamp_ns=time.time_ns(),
            left={
                "buttons": {
                    "stick": bool(self.joystick_data["left_button"]),  # Stick button on/off
                },
                "stick": {
                    "x": self.joystick_data["left_x"],  # Joystick X: -1.0 to 1.0
                    "y": self.joystick_data["left_y"],  # Joystick Y: -1.0 to 1.0
                },
                "gripper": {
                    "position": left_gripper_value,  # Current gripper position (radians)
                }
            },
            right={
                "buttons": {
                    "stick": bool(self.joystick_data["right_button"]),  # Stick button on/off
                },
                "stick": {
                    "x": self.joystick_data["right_x"],  # Joystick X: -1.0 to 1.0
                    "y": self.joystick_data["right_y"],  # Joystick Y: -1.0 to 1.0
                },
                "gripper": {
                    "position": right_gripper_value,  # Current gripper position (radians)
                }
            }
        )

    def _update_debug_display(
        self,
        left_arm_pos: List[float],
        right_arm_pos: List[float],
        left_gripper_pos: List[float],
        right_gripper_pos: List[float],
    ) -> None:
        """Update debug display with arm, gripper, and joystick positions.

        Args:
            left_arm_pos: Left arm joint positions
            right_arm_pos: Right arm joint positions
            left_gripper_pos: Left gripper position
            right_gripper_pos: Right gripper position
        """
        if not self._debug_display:
            return

        # Convert to dict for debug display compatibility
        debug_joints: Dict[str, float] = {}

        # Add left arm joints
        left_joint_names = ["L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", 
                           "L_arm_j5", "L_arm_j6", "L_arm_j7"]
        for i, name in enumerate(left_joint_names[: len(left_arm_pos)]):
            debug_joints[name] = left_arm_pos[i]

        # Add left gripper
        if left_gripper_pos:
            debug_joints["L_gripper"] = left_gripper_pos[0]

        # Add right arm joints
        right_joint_names = ["R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
                            "R_arm_j5", "R_arm_j6", "R_arm_j7"]
        for i, name in enumerate(right_joint_names[: len(right_arm_pos)]):
            debug_joints[name] = right_arm_pos[i]

        # Add right gripper
        if right_gripper_pos:
            debug_joints["R_gripper"] = right_gripper_pos[0]

        # Add joystick values (both left and right with X, Y, Button)
        debug_joints["Joy_L_X"] = self.joystick_data["left_x"]
        debug_joints["Joy_L_Y"] = self.joystick_data["left_y"]
        debug_joints["Joy_L_B"] = self.joystick_data["left_button"]
        debug_joints["Joy_R_X"] = self.joystick_data["right_x"]
        debug_joints["Joy_R_Y"] = self.joystick_data["right_y"]
        debug_joints["Joy_R_B"] = self.joystick_data["right_button"]

        # Use print_leader_arm which creates a nice table format
        self._debug_display.print_leader_arm(debug_joints)

    def run(self) -> None:
        """Main reading loop."""
        resolved_arm_topic = self.node.resolve_topic(self.arm_topic)
        resolved_gripper_joycon_topic = self.node.resolve_topic(self.gripper_joycon_topic)
        logger.info(f"Started publishing to {resolved_arm_topic} and {resolved_gripper_joycon_topic} at {self.publish_rate}Hz")

        # Start the live display if debug mode
        if self._debug_display:
            self._debug_display.start()

        start_publishing = False
        try:
            while True:
                # Read motor positions and velocities from both ports
                (
                    left_arm_pos,
                    left_arm_vel,
                    right_arm_pos,
                    right_arm_vel,
                    left_gripper_pos,
                    right_gripper_pos,
                ) = self._read_positions_velocities()

                # Check publishing condition
                if not start_publishing:
                    start_publishing = True
                    logger.info("Publishing condition satisfied")

                if not start_publishing:
                    self.rate_limiter.sleep()
                    continue

                # Apply joint limits for safety
                left_arm_pos, right_arm_pos, _ = self._apply_joint_limits(
                    left_arm_pos, right_arm_pos, torso_pos=None
                )

                # Create and publish arm data if we have at least one arm
                if left_arm_pos or right_arm_pos:
                    arm_data = self._create_exo_joint_data(
                        left_arm_pos,
                        left_arm_vel,
                        right_arm_pos,
                        right_arm_vel,
                    )
                    # Publish arm data via Zenoh
                    self.arm_publisher.publish(asdict(arm_data))

                # Create and publish gripper+joystick data as JoyConData
                gripper_joycon_data = self._create_gripper_joycon_data(
                    left_gripper_pos,
                    right_gripper_pos,
                )
                # Publish as dict (asdict handles the dataclass conversion)
                self.gripper_joycon_publisher.publish(asdict(gripper_joycon_data))

                # Update debug display
                self._update_debug_display(
                    left_arm_pos,
                    right_arm_pos,
                    left_gripper_pos,
                    right_gripper_pos,
                )

                # Use RateLimiter for precise timing
                self.rate_limiter.sleep()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        if self.left_port_handler:
            self.left_port_handler.closePort()
            logger.info(f"Closed left port {self.left_port}")

        if self.right_port_handler:
            self.right_port_handler.closePort()
            logger.info(f"Closed right port {self.right_port}")

        # Node handles cleanup
        self.node.shutdown()

        logger.info("Robotis arm reader stopped")

def main(
    namespace: str = "",
    left_port: Optional[str] = None,
    right_port: Optional[str] = None,
    baud_rate: Optional[int] = None,
    publish_rate: Optional[int] = None,
    topic: Optional[str] = None,
    debug: bool = False,
    read_velocity: bool = False,
):
    """Main entry point for Robotis leader arm reader.

    Reads from Robotis exoskeleton hardware with fixed motor IDs on separate ports:
    - Left arm: IDs 31-37 on /dev/left_leader
    - Right arm: IDs 1-7 on /dev/right_leader

    Args:
        namespace: Optional namespace prefix (e.g., "robot1")
        left_port: Override left arm serial port (uses /dev/left_leader if None)
        right_port: Override right arm serial port (uses /dev/right_leader if None)
        baud_rate: Override baud rate (uses Robotis default 4Mbps if None)
        publish_rate: Override publish rate (uses config if None)
        topic: Override topic (uses config if None)
        debug: Enable debug output
        read_velocity: Enable reading and publishing velocity data
    """
    # Setup logging
    logger_instance = setup_logging(debug)
    logger_instance.info(
        f"Starting Robotis Arm Reader{f' (namespace: {namespace})' if namespace else ''}"
    )

    reader = RobotisArmReader(
        namespace=namespace,
        left_port=left_port,
        right_port=right_port,
        baud_rate=baud_rate,
        publish_rate=publish_rate,
        topic=topic,
        debug=debug,
        read_velocity=read_velocity,
    )

    if not reader.initialize():
        logger_instance.error("Failed to initialize Robotis arm reader")
        sys.exit(1)

    try:
        reader.run()
    except KeyboardInterrupt:
        logger_instance.info("Shutting down...")
    finally:
        reader.cleanup()

if __name__ == "__main__":
    sys.exit(tyro.cli(main))
