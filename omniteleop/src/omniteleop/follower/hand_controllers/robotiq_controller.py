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

class RobotiqController(BaseHandController):
    """Controller for Robotiq grippers via RS485 pass-through."""

    # Modbus RTU commands (Hex strings converted to bytes)
    ACTIVATE_HEX = "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1"
    # RESET (rACT=0) clears the action register so the next ACTIVATE (rACT=1) is a
    # clean rising edge that reliably (re)triggers the gripper activation sequence.
    RESET_HEX = "09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30"
    READ_STATUS_HEX = "09 04 07 D0 00 03 B1 CE"
    CLOSE_HEX = "09 10 03 E8 00 03 06 09 00 00 FF FF FF 42 29"
    OPEN_HEX = "09 10 03 E8 00 03 06 09 00 00 00 FF FF 72 19"

    def __init__(self, arm, side: str = "unknown"):
        """
        Initialize the Robotiq controller.

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

        Sends RESET (rACT=0) first, waits 0.5s, then ACTIVATE (rACT=1). The
        reset guarantees a clean rACT 0->1 rising edge so the gripper reliably
        (re)starts its activation sequence even if rACT was already latched at
        1 from a prior session.
        """
        if not self.arm.enable_ee_pass_through:
            logger.warning(f"{self.side.capitalize()} arm: EE pass-through not enabled. Cannot activate Robotiq.")
            return

        logger.info(f"{self.side.capitalize()} arm: Activating Robotiq gripper...")
        # 1. Reset (falling edge) so the next activate is a clean rising edge.
        reset_message = self._hex_to_bytes(self.RESET_HEX)
        self.arm.send_ee_pass_through_message(reset_message)
        time.sleep(0.5)
        # 2. Activate (rising edge).
        activate_message = self._hex_to_bytes(self.ACTIVATE_HEX)
        self.arm.send_ee_pass_through_message(activate_message)
        # Give the gripper time to process activation before further commands.
        time.sleep(1.0)

    def open(self):
        """Sends the open command to the gripper."""
        self.set_position([0.0])

    def close(self):
        """Sends the close command to the gripper."""
        self.set_position([1.0])

    def set_position(self, pos: list[float]):
        """Sends the position command to the gripper.
        
        Args:
            pos: List of normalized gripper positions (0.0 to 1.0).
                 For Robotiq, only the first element is used.
                 0.0 means fully open, 1.0 means fully closed.
        """
        if not self.arm.enable_ee_pass_through or not pos:
            return
        
        # Use the first value in the list
        p = pos[0]
        
        # Clip position to [0, 1]
        p = max(0.0, min(1.0, p))
        
        # Map 0.0-1.0 to 0-255 (0x00-0xFF)
        # 0.0 -> 0 (Open), 1.0 -> 255 (Close)
        target_pos = int(p * 255)
        
        # Build Modbus RTU packet for Write Multiple Registers (0x10)
        # Slave ID: 0x09
        # Function Code: 0x10
        # Starting Address: 0x03E8 (1000)
        # Number of Registers: 0x0003
        # Number of Data Bytes: 0x06
        # Data Bytes: 
        #   Register 0x03E8: 0x09 (Action Request: gACT=1, gGTO=1)
        #   Register 0x03E9: 0x00 (Reserved)
        #   Register 0x03EA: 0x00 (Reserved)
        #   Register 0x03EB: target_pos (Target Position)
        #   Register 0x03EC: 0xFF (Velocity)
        #   Register 0x03ED: 0xFF (Force)
        
        packet = bytearray([
            0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 
            0x09, 0x00, 0x00, target_pos, 0xFF, 0xFF
        ])
        
        # Append CRC16
        crc = self._compute_crc16(packet)
        packet.append(crc & 0xFF)
        packet.append((crc >> 8) & 0xFF)
        
        # logger.info(f"{self.side.capitalize()} arm: Setting Robotiq gripper position to {pos:.2f} (hex: {target_pos:02X})")
        self.arm.send_ee_pass_through_message(bytes(packet))

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

    def send_raw_command(self, hex_str: str):
        """Sends a raw hex command to the gripper."""
        if not self.arm.enable_ee_pass_through:
            return
        message = self._hex_to_bytes(hex_str)
        self.arm.send_ee_pass_through_message(message)

    def parse_status(self, response_data: bytes) -> Optional[Dict[str, Any]]:
        """Parses the 6-byte status response from the Robotiq gripper."""
        if len(response_data) < 7:
            return None

        header = response_data[:3]  # Slave ID, Func Code, Byte Count
        payload = response_data[3:-2]  # Data bytes

        if header[1] == 0x10:  # Ignore Write Multiple Registers response
            return None

        if header[1] != 0x04 or header[2] != 0x06:
            return None

        status = {
            "gACT": (payload[0] & 0b00000001),  # Gripper Activation Status
            "gGTO": (payload[0] & 0b00001000) >> 3,  # Action Status (go to)
            "gSTA": (payload[0] & 0b00110000) >> 4,  # Gripper Status
            "gOBJ": (payload[0] & 0b11000000) >> 6,  # Object Detection Status
            "gFLT": (payload[2] & 0b00001111),  # Fault Code
            "gPO": payload[3],  # Position Request Echo
            "gPR": payload[4],  # Real-time Position
            "gCU": payload[5],  # Motor Current
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
        logger.info(f"{self.side.capitalize()} arm: Started Robotiq status monitoring (at {1.0/interval:.1f}Hz).")

    def stop_tracking(self):
        """Stops the status monitoring thread."""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
            logger.info(f"{self.side.capitalize()} arm: Stopped Robotiq status monitoring.")

    def _monitor_loop(self, interval: float):
        status_message = self._hex_to_bytes(self.READ_STATUS_HEX)
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
                        # logger.debug(f"{self.side.capitalize()} Robotiq status: {status}")
                    break
                time.sleep(0.01)
            
            time.sleep(interval)

    def get_status(self) -> list[float]:
        """Returns the latest normalized gripper position (0.0 to 1.0).
        0.0 means fully open, 1.0 means fully closed.
        """
        gPR = self._last_status.get("gPR", 0)
        return [float(gPR) / 255.0]
