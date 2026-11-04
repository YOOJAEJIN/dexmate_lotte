# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robotiq 2F-85 gripper controller over a direct USB-to-RS485 serial link.

Unlike :class:`RobotiqController` (which talks through the robot arm's RS485
pass-through bus), this controller talks to the gripper directly over a
USB-to-RS485 serial adapter using ``pyserial``.

Default wiring:
    * left arm  -> /dev/ttyUSB0
    * right arm -> /dev/ttyUSB1

Protocol: Modbus RTU (slave id 0x09), 115200 baud, 8N1.

The USB bus is shared between the status-monitoring thread and the command
thread, so every read/write is serialized with a per-instance ``threading.Lock``
to prevent interleaving/corruption of Modbus frames. Activation always begins
with an explicit reset (rACT=0) followed by activate (rACT=1) so the rising
edge reliably (re)triggers the gripper's calibration sequence.
"""

import threading
import time
from typing import Dict, Optional

from loguru import logger

try:
    import serial as pyserial
except ImportError:
    pyserial = None  # type: ignore[assignment]

from omniteleop.follower.hand_controllers.base_hand_controller import (
    BaseHandController,
)


class RobotiqUsbController(BaseHandController):
    """Robotiq 2F-85 controller communicating over a direct USB serial port.

    Args:
        arm: Unused for communication (kept for a uniform factory signature).
            Only referenced for optional metadata if present.
        side: ``'left'`` or ``'right'``; selects the default serial port.
        left_port: Override the left-arm serial device.
        right_port: Override the right-arm serial device.
        baudrate: Serial baud rate (default 115200, Robotiq standard).
    """

    # Default USB serial ports per arm side.
    DEFAULT_SERIAL_PORTS = {
        "left": "/dev/ttyUSB0",
        "right": "/dev/ttyUSB1",
    }

    # Serial configuration (Robotiq 2F-85).
    BAUDRATE = 115200
    SERIAL_TIMEOUT = 0.05  # seconds (pyserial per-read timeout)
    # Modbus RTU inter-frame gap: 3.5 char times at 115200 8N1 ~= 0.3ms; use a
    # comfortable margin so the gripper can frame its response correctly.
    FRAME_GAP = 0.005  # seconds
    # Read Input Registers (0x04) status response length:
    # slave(1) + func(1) + byte_count(1) + 6 data + CRC(2) = 11 bytes
    STATUS_RESPONSE_LEN = 11
    # Write Multiple Registers (0x10) response length:
    # slave(1) + func(1) + addr(2) + nreg(2) + CRC(2) = 8 bytes
    WRITE_RESPONSE_LEN = 8

    # Modbus RTU commands (slave id 0x09). CRC is already baked into each hex.
    ACTIVATE_HEX = "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1"
    READ_STATUS_HEX = "09 04 07 D0 00 03 B1 CE"
    CLOSE_HEX = "09 10 03 E8 00 03 06 09 00 00 FF FF FF 42 29"
    OPEN_HEX = "09 10 03 E8 00 03 06 09 00 00 00 FF FF 72 19"

    # gSTA codes (Gripper Status)
    GSTA_RESET = 0  # in reset (not activated)
    GSTA_ACTIVATING = 1  # activation in progress
    GSTA_ACTIVE = 3  # gripper is active / operational

    # gFLT (Fault Status) codes from the Robotiq communication interface.
    FAULT_CODES = {
        0x00: "no fault",
        0x05: "action delayed (activation must complete first)",
        0x06: "interaction (finger contacted object during move)",
        0x07: "interaction (fingertip contacted object during move)",
        0x08: "interaction (finger reached limit before target)",
        0x09: "interaction (fingertip reached limit before target)",
        0x0A: "overcurrent on motor 0",
        0x0B: "overcurrent on motor 1",
        0x0D: "motor 0 blocked",
        0x0E: "motor 1 blocked",
        0x0F: "automatic release in progress",
    }

    def __init__(
        self,
        arm,
        side: str = "unknown",
        left_port: str = DEFAULT_SERIAL_PORTS["left"],
        right_port: str = DEFAULT_SERIAL_PORTS["right"],
        baudrate: int = BAUDRATE,
    ):
        self.arm = arm  # unused for comms; kept for factory signature symmetry
        self.side = side
        self._baudrate = baudrate

        if pyserial is None:
            raise ImportError(
                "pyserial is required for USB Robotiq communication. "
                "Install it with `pip install pyserial`."
            )

        port = {"left": left_port, "right": right_port}.get(side)
        if port is None:
            raise ValueError(
                f"Unknown side '{side}' for RobotiqUsbController. "
                f"Expected one of: {list(self.DEFAULT_SERIAL_PORTS.keys())}."
            )
        self.port = port

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_status: Dict[str, int] = {}
        self._last_timestamp = 0

        self._serial = pyserial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=pyserial.EIGHTBITS,
            parity=pyserial.PARITY_NONE,
            stopbits=pyserial.STOPBITS_ONE,
            timeout=self.SERIAL_TIMEOUT,
        )
        logger.info(
            f"{self.side.capitalize()} arm: Opened Robotiq USB serial on "
            f"{port} @ {baudrate} baud."
        )

    # -- low-level helpers ----------------------------------------------------

    @staticmethod
    def _hex_to_bytes(hex_str: str) -> bytes:
        return bytes.fromhex(hex_str.replace(" ", ""))

    @staticmethod
    def _compute_crc16(data: bytes) -> int:
        """Computes the Modbus RTU CRC16 (polynomial 0xA001, init 0xFFFF)."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    @classmethod
    def _build_action_request(
        cls,
        rACT: int,
        rGTO: int = 0,
        target_pos: int = 0,
        velocity: int = 0xFF,
        force: int = 0xFF,
    ) -> bytes:
        """Builds a Write Multiple Registers (0x10) action-request packet.

        Register 0x03E8 action byte layout:
            bit0 rACT - Activation (0 = reset, 1 = activate)
            bit3 rGTO - Go To (1 = execute the requested position)
        The gripper only (re)starts its activation sequence on a rising edge
        of rACT (0 -> 1), so activation must always be preceded by a reset.

        Returns a complete Modbus RTU message including the CRC16 trailer.
        """
        action_byte = (rACT & 0x01) | ((rGTO & 0x01) << 3)
        packet = bytearray(
            [
                0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                action_byte, 0x00, 0x00, target_pos, velocity, force,
            ]
        )
        crc = cls._compute_crc16(packet)
        packet.append(crc & 0xFF)
        packet.append((crc >> 8) & 0xFF)
        return bytes(packet)

    def _transaction(
        self,
        message: bytes,
        timeout: float = 0.1,
        expected_len: int = STATUS_RESPONSE_LEN,
    ) -> bytes:
        """Sends a frame and reads back its response, fully serialized.

        Acquires the port lock, flushes stale input, writes the frame, enforces
        the Modbus inter-frame gap, then reads up to ``expected_len`` bytes
        within ``timeout``. Never raises on timeout; returns what was read.
        """
        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write(message)
            time.sleep(self.FRAME_GAP)  # let the gripper frame its response
            deadline = time.time() + timeout
            buf = bytearray()
            while time.time() < deadline and len(buf) < expected_len:
                chunk = self._serial.read(expected_len - len(buf))
                if chunk:
                    buf.extend(chunk)
            return bytes(buf)

    def _send_command(self, message: bytes) -> None:
        """Sends a command frame and drains its short write-echo response."""
        self._transaction(
            message, timeout=0.05, expected_len=self.WRITE_RESPONSE_LEN
        )

    def _read_status(self, timeout: float = 0.1) -> Optional[Dict[str, int]]:
        """Polls the gripper status once and returns the parsed fields."""
        data = self._transaction(
            self._hex_to_bytes(self.READ_STATUS_HEX),
            timeout=timeout,
            expected_len=self.STATUS_RESPONSE_LEN,
        )
        if not data:
            logger.debug(f"{self.side.capitalize()} arm: no status response")
            return None
        status = self._parse_status(data)
        if status is None:
            logger.debug(
                f"{self.side.capitalize()} arm: unparseable status, raw: {data.hex(' ')}"
            )
        return status

    def _parse_status(self, response_data: bytes) -> Optional[Dict[str, int]]:
        """Parses the 6-byte status response from the Robotiq gripper.

        Returns a dict of decoded status fields, or ``None`` for non-status
        frames (e.g. a Write Multiple Registers echo) or too-short data.
        """
        if len(response_data) < 7:
            return None

        header = response_data[:3]  # Slave ID, Func Code, Byte Count
        payload = response_data[3:-2]  # Data bytes (strip CRC)

        if header[1] == 0x10:  # Write Multiple Registers response -> ignore
            return None

        if header[1] != 0x04 or header[2] != 0x06 or len(payload) < 6:
            return None

        return {
            "gACT": payload[0] & 0b00000001,        # Gripper Activation Status
            "gGTO": (payload[0] & 0b00001000) >> 3,  # Action Status (go to)
            "gSTA": (payload[0] & 0b00110000) >> 4,  # Gripper Status
            "gOBJ": (payload[0] & 0b11000000) >> 6,  # Object Detection Status
            "gFLT": payload[2] & 0b00001111,         # Fault Code
            "gPO": payload[3],                       # Position Request Echo
            "gPR": payload[4],                       # Real-time Position (0~255)
            "gCU": payload[5],                       # Motor Current
        }

    # -- BaseHandController interface ----------------------------------------

    def activate(self) -> bool:
        """Activates the gripper, polling until ``gSTA == 3`` (active).

        The gripper only (re)starts activation on a rising edge of rACT
        (0 -> 1). If rACT is already latched at 1 (e.g. the gripper is powered
        up after a prior session without a power cycle), re-sending rACT=1 is a
        no-op and the gripper stays in reset forever (gSTA=0, gFLT=5). We
        therefore always begin with an explicit reset (rACT=0) before
        activating.

        Returns True once the gripper reports active, False otherwise.
        """
        # velocity=0/force=0 to exactly match the proven ACTIVATE_HEX framing
        # (rGTO=0 during activation, so speed/force are irrelevant).
        reset_msg = self._build_action_request(rACT=0, velocity=0, force=0)
        activate_msg = self._build_action_request(rACT=1, velocity=0, force=0)

        logger.info(f"{self.side.capitalize()} arm: Activating Robotiq gripper...")
        for attempt in range(1, 4):
            logger.info(
                f"{self.side.capitalize()} arm: activation attempt {attempt}/3: "
                f"RESET (rACT=0) -> ACTIVATE (rACT=1)..."
            )
            # 1. Reset (falling edge) so the next activate is a clean rising edge.
            self._send_command(reset_msg)
            time.sleep(0.5)  # let the gripper clear its state
            # 2. Activate (rising edge).
            self._send_command(activate_msg)

            deadline = time.time() + 10.0
            last: Optional[Dict[str, int]] = None
            while time.time() < deadline:
                status = self._read_status(timeout=0.1)
                if status:
                    last = status
                    gsta = status["gSTA"]
                    gflt = status["gFLT"]
                    logger.info(
                        f"{self.side.capitalize()} arm: activating... gSTA={gsta} "
                        f"gACT={status['gACT']} gFLT={gflt} "
                        f"({self.FAULT_CODES.get(gflt, '?')}) gPR={status['gPR']}"
                    )
                    if gsta == self.GSTA_ACTIVE:
                        logger.success(
                            f"{self.side.capitalize()} arm: Robotiq gripper ACTIVE (gSTA=3)."
                        )
                        return True
                    if gflt not in (0, 5):  # 5 = activation not finished (expected)
                        logger.warning(
                            f"{self.side.capitalize()} arm: unexpected fault during "
                            f"activation: gFLT={gflt} "
                            f"({self.FAULT_CODES.get(gflt, '?')})"
                        )
                time.sleep(0.3)

            last_sta = last["gSTA"] if last else "?"
            last_flt = last["gFLT"] if last else "?"
            logger.warning(
                f"{self.side.capitalize()} arm: activation not complete after "
                f"10.0s (gSTA={last_sta}, gFLT={last_flt})."
            )

        logger.error(
            f"{self.side.capitalize()} arm: FAILED to activate Robotiq gripper after "
            f"3 attempts. If gACT=1 but gSTA stays 0, the gripper never started its "
            f"calibration move -> check the 24V motor supply and that the fingers are "
            f"free to move."
        )
        return False

    def open(self):
        """Sends the fully-open command (position 0.0)."""
        self.set_position([0.0])

    def close(self):
        """Sends the fully-close command (position 1.0)."""
        self.set_position([1.0])

    def set_position(self, pos: list[float]):
        """Sends the position command to the gripper.

        Args:
            pos: List of normalized gripper positions (0.0 to 1.0).
                 For Robotiq, only the first element is used.
                 0.0 means fully open, 1.0 means fully closed.
        """
        if not pos:
            return

        p = max(0.0, min(1.0, pos[0]))
        target_pos = int(p * 255)  # 0x00 (open) .. 0xFF (close)
        packet = self._build_action_request(
            rACT=1, rGTO=1, target_pos=target_pos, velocity=0xFF, force=0xFF
        )
        self._send_command(packet)

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
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(
            f"{self.side.capitalize()} arm: Started Robotiq USB status monitoring "
            f"(at {1.0/interval:.1f}Hz)."
        )

    def stop_tracking(self):
        """Stops the status monitoring thread."""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
            logger.info(
                f"{self.side.capitalize()} arm: Stopped Robotiq USB status monitoring."
            )

    def _monitor_loop(self, interval: float):
        status_message = self._hex_to_bytes(self.READ_STATUS_HEX)
        while not self._stop_event.is_set():
            self._send_command(status_message)

            start = time.time()
            while time.time() - start < 0.1:
                status = self._read_status(timeout=0.02)
                if status:
                    now_ns = time.time_ns()
                    if now_ns > self._last_timestamp:
                        self._last_timestamp = now_ns
                        self._last_status = status
                    break
                time.sleep(0.01)

            time.sleep(interval)

    def get_status(self) -> list[float]:
        """Returns the latest normalized gripper position (0.0 to 1.0).

        0.0 means fully open, 1.0 means fully closed.
        """
        gPR = self._last_status.get("gPR", 0)
        return [float(gPR) / 255.0]

    def shutdown(self):
        """Stops monitoring and closes the serial port."""
        self.stop_tracking()
        if self._serial is not None:
            self._serial.close()
            self._serial = None
            logger.info(
                f"{self.side.capitalize()} arm: Closed Robotiq USB serial port."
            )
