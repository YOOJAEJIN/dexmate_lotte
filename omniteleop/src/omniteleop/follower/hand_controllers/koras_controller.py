# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

import threading
import time
from typing import Optional, Dict, Any
from loguru import logger
from omniteleop.follower.hand_controllers.base_hand_controller import BaseHandController

class KorasController(BaseHandController):
    """Controller for KORAS KR-GCS grippers via RS485 pass-through.

    The KORAS gripper exposes a Modbus RTU interface on the arm's EE pass-through
    bus. Unlike the Robotiq gripper (which accepts a continuous target position),
    KORAS is commanded via discrete operation codes (initialize/open/close)
    written to Holding Registers 0 and 1. Status is read from Holding Registers
    10..14 (status bits, motor position/current/velocity, finger position).
    """

    # KORAS Modbus RTU parameters
    KORAS_SLAVE_ID = 0x01
    KORAS_COMMAND_ADDRESS = 0x0000
    KORAS_STATUS_ADDRESS = 0x000A
    KORAS_STATUS_REGISTER_COUNT = 5
    KORAS_BAUD_RATE = 115200

    # Command codes written to Holding Register 0
    KORAS_CMD_INIT = 101
    KORAS_CMD_OPEN = 102
    KORAS_CMD_CLOSE = 103

    # Pre-built Modbus RTU messages (Hex strings converted to bytes)
    # READ_STATUS: Read Holding Registers 10..14 (status, motor pos/current/velocity, finger pos)
    READ_STATUS_HEX = "01 03 00 0A 00 05 A5 CB"
    # INIT: Write Command=101 (initialize) to Holding Registers 0 and 1
    INIT_HEX = "01 10 00 00 00 02 04 00 65 00 00 E3 B0"
    # CLOSE: Write Command=103 (close) to Holding Registers 0 and 1
    CLOSE_HEX = "01 10 00 00 00 02 04 00 67 00 00 42 70"
    # OPEN: Write Command=102 (open) to Holding Registers 0 and 1
    OPEN_HEX = "01 10 00 00 00 02 04 00 66 00 00 13 B0"

    # KORAS finger_position spans 0..1000 (0: closed, 1000: open). Unlike Robotiq
    # whose position convention is 0=open/255=closed, KORAS reports 0=closed.
    FINGER_POSITION_MAX = 1000

    # Position threshold for deciding open vs close from a normalized command.
    # KORAS only accepts discrete open/close commands, so a normalized position
    # of < 0.5 maps to open and >= 0.5 maps to close (Robotiq-style: 1.0 = close).
    POSITION_CLOSE_THRESHOLD = 0.5

    def __init__(self, arm, side: str = "unknown"):
        """
        Initialize the KORAS controller.

        Args:
            arm: The robot arm object (e.g., bot.left_arm) that supports pass-through.
            side: 'left' or 'right' (for logging).
        """
        self.arm = arm
        self.side = side
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._last_status: Dict[str, Any] = {}
        self._last_timestamp = 0

    def _hex_to_bytes(self, hex_str: str) -> bytes:
        return bytes.fromhex(hex_str.replace(" ", ""))

    def activate(self):
        """Activates the gripper.

        Sends the KORAS initialize command (command=101). The arm's EE baud
        rate must already be set to 115200 via config (KORAS_BAUD_RATE); it is
        not verified here so that a temporary EE baud-rate service outage does
        not abort activation.
        """
        if not self.arm.enable_ee_pass_through:
            logger.warning(f"{self.side.capitalize()} arm: EE pass-through not enabled. Cannot activate KORAS.")
            return

        logger.info(f"{self.side.capitalize()} arm: Activating KORAS gripper...")
        init_message = self._hex_to_bytes(self.INIT_HEX)
        self.arm.send_ee_pass_through_message(init_message)
        # Give the gripper time to process initialization before further commands.
        time.sleep(2.0)

    def open(self):
        """Sends the open command to the gripper."""
        if not self.arm.enable_ee_pass_through:
            return
        open_message = self._hex_to_bytes(self.OPEN_HEX)
        self.arm.send_ee_pass_through_message(open_message)

    def close(self):
        """Sends the close command to the gripper."""
        if not self.arm.enable_ee_pass_through:
            return
        close_message = self._hex_to_bytes(self.CLOSE_HEX)
        self.arm.send_ee_pass_through_message(close_message)

    def set_position(self, pos: list[float]):
        """Sends the position command to the gripper.

        KORAS only accepts discrete open/close commands, so the normalized
        position is thresholded: values >= 0.5 close the gripper and values
        < 0.5 open it (Robotiq-style convention where 1.0 means closed).

        Args:
            pos: List of normalized gripper positions (0.0 to 1.0).
                 For KORAS, only the first element is used.
                 0.0 means fully open, 1.0 means fully closed.
        """
        if not self.arm.enable_ee_pass_through or not pos:
            return

        # Use the first value in the list and clip to [0, 1]
        p = max(0.0, min(1.0, pos[0]))

        if p >= self.POSITION_CLOSE_THRESHOLD:
            self.close()
        else:
            self.open()

    def _compute_crc16(self, data: bytes) -> int:
        """Computes the Modbus RTU CRC16."""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    def _build_command_message(self, command: int, value: int = 0) -> bytes:
        """Builds a Modbus RTU Write Multiple Registers (0x10) message.

        Writes Command and Target value to Holding Registers 0 and 1.

        Args:
            command: KORAS command code (e.g. KORAS_CMD_OPEN).
            value: Target value for Holding Register 1. Default 0.
        """
        packet = bytearray([
            self.KORAS_SLAVE_ID,
            0x10,
            (self.KORAS_COMMAND_ADDRESS >> 8) & 0xFF,
            self.KORAS_COMMAND_ADDRESS & 0xFF,
            0x00, 0x02, 0x04,
            (command >> 8) & 0xFF, command & 0xFF,
            (value >> 8) & 0xFF, value & 0xFF,
        ])
        crc = self._compute_crc16(packet)
        packet.append(crc & 0xFF)
        packet.append((crc >> 8) & 0xFF)
        return bytes(packet)

    def _build_read_status_message(self) -> bytes:
        """Builds a Modbus RTU Read Holding Registers (0x03) message for status."""
        packet = bytearray([
            self.KORAS_SLAVE_ID,
            0x03,
            (self.KORAS_STATUS_ADDRESS >> 8) & 0xFF,
            self.KORAS_STATUS_ADDRESS & 0xFF,
            (self.KORAS_STATUS_REGISTER_COUNT >> 8) & 0xFF,
            self.KORAS_STATUS_REGISTER_COUNT & 0xFF,
        ])
        crc = self._compute_crc16(packet)
        packet.append(crc & 0xFF)
        packet.append((crc >> 8) & 0xFF)
        return bytes(packet)

    def send_raw_command(self, hex_str: str):
        """Sends a raw hex command to the gripper."""
        if not self.arm.enable_ee_pass_through:
            return
        message = self._hex_to_bytes(hex_str)
        self.arm.send_ee_pass_through_message(message)

    def parse_status(self, response_data: bytes) -> Optional[Dict[str, Any]]:
        """Parses the KORAS status response from Holding Registers 10..14.

        Expected Modbus RTU read response (function code 0x03) layout:
            [Slave ID][0x03][Byte Count][Reg0 Hi][Reg0 Lo]...[Reg4 Hi][Reg4 Lo][CRC Lo][CRC Hi]
        where Reg0..Reg4 hold status bits, motor position (deg), motor current
        (mA), motor velocity (RPM), and finger position (0: closed, 1000: open).

        Args:
            response_data: Raw bytes of the status response.

        Returns:
            Dict of parsed status fields, or None if the response is not a valid
            read-status frame (e.g. a write-register echo or malformed frame).
        """
        if len(response_data) < 5:  # Slave ID, Func Code, Byte Count, payload, CRC
            return None

        function_code = response_data[1]
        if function_code == 0x10:  # Ignore Write Multiple Registers response
            return None

        if function_code != 0x03:
            return None

        byte_count = response_data[2]
        expected_byte_count = self.KORAS_STATUS_REGISTER_COUNT * 2
        if byte_count != expected_byte_count or len(response_data) < 3 + byte_count + 2:
            return None

        payload = response_data[3 : 3 + byte_count]
        registers = [
            int.from_bytes(payload[index : index + 2], byteorder="big")
            for index in range(0, byte_count, 2)
        ]

        status_bits = registers[0]
        status = {
            "motor_enabled": bool(status_bits & (1 << 0)),
            "initializing": bool(status_bits & (1 << 1)),
            "position_control": bool(status_bits & (1 << 2)),
            "velocity_control": bool(status_bits & (1 << 3)),
            "current_control": bool(status_bits & (1 << 4)),
            "opening": bool(status_bits & (1 << 5)),
            "closing": bool(status_bits & (1 << 6)),
            "motor_fault": bool(status_bits & (1 << 9)),
            "motor_position_deg": registers[1],
            "motor_current_ma": registers[2],
            "motor_velocity_rpm": registers[3],
            "finger_position": registers[4],  # 0: closed, 1000: open
        }
        return status

    def start_tracking(self, interval: float = 0.05):
        """Starts a background thread to monitor gripper status.

        Args:
            interval: Monitoring interval in seconds. Default is 0.05 (20Hz).
        """
        if self._monitor_thread is not None:
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"{self.side.capitalize()} arm: Started KORAS status monitoring (at {1.0/interval:.1f}Hz).")

    def stop_tracking(self):
        """Stops the status monitoring thread."""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
            logger.info(f"{self.side.capitalize()} arm: Stopped KORAS status monitoring.")

    def _monitor_loop(self, interval: float):
        status_message = self._build_read_status_message()
        while not self._stop_event.is_set():
            self.arm.send_ee_pass_through_message(status_message)

            start = time.time()
            while time.time() - start < 0.1:
                response = self.arm.get_ee_pass_through_response()
                if response and response["timestamp_ns"] > self._last_timestamp:
                    self._last_timestamp = response["timestamp_ns"]
                    status = self.parse_status(response['data'])
                    if status:
                        self._last_status = status
                        # logger.debug(f"{self.side.capitalize()} KORAS status: {status}")
                    break
                time.sleep(0.01)

            time.sleep(interval)

    def get_status(self) -> list[float]:
        """Returns the latest normalized gripper position (0.0 to 1.0).

        KORAS reports finger_position in 0..1000 where 0 means closed and 1000
        means open. To match the Robotiq-style convention used across the code
        base (0.0 = open, 1.0 = closed), the value is inverted before returning.

        Returns:
            Single-element list with the normalized gripper position, where
            0.0 means fully open and 1.0 means fully closed.
        """
        finger_position = self._last_status.get("finger_position", 0)
        normalized_open = float(finger_position) / float(self.FINGER_POSITION_MAX)
        # Invert so that 0.0 = open (Robotiq convention) instead of 0.0 = closed.
        return [1.0 - normalized_open]
