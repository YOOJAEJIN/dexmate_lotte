# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating Robotiq 2F-85 gripper control via USB serial.

This script is the USB/serial counterpart of ``control_customized_ee_read.py``.
Where that script sends raw RS485 messages through the robot arm pass-through
bus, this script talks to the gripper directly over a USB-to-RS485 serial
adapter using ``pyserial``.

Default wiring:
    * left arm  -> /dev/ttyUSB0
    * right arm -> /dev/ttyUSB1

Protocol: Modbus RTU (slave id 0x09), 115200 baud, 8N1.

IMPORTANT (vs. the RS485 pass-through version):
    * All serial access (both the status monitor thread and the command
      thread) is serialized with a per-port ``threading.Lock``. Without this,
      concurrent writes/reads corrupt Modbus frames and the gripper drops
      commands (most critically the activation frame, which leaves the gripper
      in the reset state and refuses every move with fault 5).
    * A Modbus inter-frame gap is enforced after each write.
    * Activation is blocking: we poll until ``gSTA == 3`` (gripper active)
      before issuing any move command, instead of blindly sleeping.
"""

import threading
import time
from typing import Dict, Literal, Optional

import tyro
from loguru import logger

try:
    import serial as pyserial
except ImportError:
    pyserial = None  # type: ignore[assignment]

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


def hex_to_bytes(hex_str: str) -> bytes:
    """Converts a space-separated hex string into bytes."""
    return bytes.fromhex(hex_str.replace(" ", ""))


def compute_crc16(data: bytes) -> int:
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


def build_position_packet(normalized_pos: float) -> bytes:
    """Builds a Write Multiple Registers (0x10) packet for a target position.

    Args:
        normalized_pos: 0.0 (fully open) .. 1.0 (fully closed). Clipped to range.

    Returns:
        A complete Modbus RTU message including the CRC16 trailer.
    """
    p = max(0.0, min(1.0, normalized_pos))
    target_pos = int(p * 255)  # 0x00 (open) .. 0xFF (close)
    return build_action_request(rACT=1, rGTO=1, target_pos=target_pos)


def build_action_request(
    rACT: int, rGTO: int = 0, target_pos: int = 0, velocity: int = 0xFF, force: int = 0xFF
) -> bytes:
    """Builds a Write Multiple Registers (0x10) action-request packet.

    Register 0x03E8 action byte layout:
        bit0 rACT  - Activation (0 = reset, 1 = activate)
        bit3 rGTO  - Go To (1 = execute the requested position)
    The gripper only (re)starts its activation sequence on a rising edge of
    rACT (0 -> 1), so activation must always be preceded by a reset (rACT=0).

    Args:
        rACT: 0 to reset / 1 to activate.
        rGTO: 1 to go to the target position.
        target_pos: 0x00 (open) .. 0xFF (close).
        velocity: 0x00..0xFF (max 0xFF).
        force: 0x00..0xFF (max 0xFF).

    Returns:
        A complete Modbus RTU message including the CRC16 trailer.
    """
    action_byte = (rACT & 0x01) | ((rGTO & 0x01) << 3)
    packet = bytearray(
        [
            0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
            action_byte, 0x00, 0x00, target_pos, velocity, force,
        ]
    )
    crc = compute_crc16(packet)
    packet.append(crc & 0xFF)
    packet.append((crc >> 8) & 0xFF)
    return bytes(packet)


def parse_status(response_data: bytes) -> Optional[Dict[str, int]]:
    """Parses the 6-byte status response from the Robotiq gripper.

    Expected layout:
        [Slave ID][Func 0x04][Byte count 0x06][6 data bytes][CRC2]

    Returns a dict of decoded status fields, or ``None`` for non-status frames
    (e.g. a Write Multiple Registers echo) or too-short/garbled data.
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
        "gACT": payload[0] & 0b00000001,       # Gripper Activation Status
        "gGTO": (payload[0] & 0b00001000) >> 3,  # Action Status (go to)
        "gSTA": (payload[0] & 0b00110000) >> 4,  # Gripper Status
        "gOBJ": (payload[0] & 0b11000000) >> 6,  # Object Detection Status
        "gFLT": payload[2] & 0b00001111,         # Fault Code
        "gPO": payload[3],                       # Position Request Echo
        "gPR": payload[4],                       # Real-time Position (0~255)
        "gCU": payload[5],                       # Motor Current
    }


def open_serial_port(port: str, baudrate: int = BAUDRATE):
    """Opens and returns a pyserial Serial handle for the given port."""
    if pyserial is None:
        raise ImportError(
            "pyserial is required for USB Robotiq communication. "
            "Install it with `pip install pyserial`."
        )
    ser = pyserial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=pyserial.EIGHTBITS,
        parity=pyserial.PARITY_NONE,
        stopbits=pyserial.STOPBITS_ONE,
        timeout=SERIAL_TIMEOUT,
    )
    logger.info(f"Opened Robotiq serial on {port} @ {baudrate} baud.")
    return ser


class RobotiqUsb:
    """Thread-safe controller for one Robotiq 2F-85 over a USB serial port.

    A single ``threading.Lock`` guards every read/write so the background
    status monitor and the command issuer never interleave Modbus frames on
    the same bus.
    """

    def __init__(self, port: str, side: str, baudrate: int = BAUDRATE):
        self.port = port
        self.side = side
        self.ser = open_serial_port(port, baudrate)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_status: Dict[str, int] = {}

    # -- low-level transaction ------------------------------------------------

    def _transaction(
        self,
        message: bytes,
        timeout: float = 0.1,
        expected_len: int = STATUS_RESPONSE_LEN,
    ) -> bytes:
        """Sends a frame and reads back its response, fully serialized.

        Acquires the port lock, flushes any stale input, writes the frame,
        enforces the Modbus inter-frame gap, then reads up to ``expected_len``
        bytes within ``timeout``. Never raises on timeout; returns what was read.
        """
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write(message)
            time.sleep(FRAME_GAP)  # let the gripper frame its response
            deadline = time.time() + timeout
            buf = bytearray()
            while time.time() < deadline and len(buf) < expected_len:
                chunk = self.ser.read(expected_len - len(buf))
                if chunk:
                    buf.extend(chunk)
            return bytes(buf)

    def send_command(self, message: bytes) -> None:
        """Sends a command frame and drains its short write-echo response."""
        self._transaction(message, timeout=0.05, expected_len=WRITE_RESPONSE_LEN)

    def read_status(self, timeout: float = 0.1) -> Optional[Dict[str, int]]:
        """Polls the gripper status once and returns the parsed fields."""
        data = self._transaction(
            hex_to_bytes(READ_STATUS_HEX),
            timeout=timeout,
            expected_len=STATUS_RESPONSE_LEN,
        )
        if not data:
            logger.debug(f"[{self.side}] no status response")
            return None
        status = parse_status(data)
        if status:
            self._last_status = status
        else:
            # Log raw bytes when a response arrives but cannot be parsed, so
            # framing / wiring problems are visible without code changes.
            logger.debug(f"[{self.side}] unparseable status, raw: {data.hex(' ')}")
        return status

    # -- high-level operations -----------------------------------------------

    def activate(
        self, per_attempt_timeout: float = 10.0, retries: int = 3
    ) -> bool:
        """Activates the gripper, polling until ``gSTA == 3`` (active).

        The gripper only (re)starts activation on a rising edge of rACT
        (0 -> 1). If rACT is already latched at 1 (e.g. the gripper is powered
        up after a prior session without a power cycle), re-sending rACT=1 is a
        no-op and the gripper stays in reset forever (gSTA=0, gFLT=5). To avoid
        this we always begin with an explicit reset (rACT=0) before activating.
        """
        # velocity=0/force=0 to exactly match the proven ACTIVATE_HEX framing
        # (rGTO=0 during activation, so speed/force are irrelevant).
        reset_msg = build_action_request(rACT=0, velocity=0, force=0)
        activate_msg = build_action_request(rACT=1, velocity=0, force=0)
        for attempt in range(1, retries + 1):
            logger.info(
                f"[{self.side}] activation attempt {attempt}/{retries}: "
                f"RESET (rACT=0) -> ACTIVATE (rACT=1)..."
            )
            # 1. Reset (falling edge) so the next activate is a clean rising edge.
            self.send_command(reset_msg)
            time.sleep(0.5)  # let the gripper clear its state
            # 2. Activate (rising edge).
            self.send_command(activate_msg)

            deadline = time.time() + per_attempt_timeout
            last: Optional[Dict[str, int]] = None
            while time.time() < deadline:
                status = self.read_status(timeout=0.1)
                if status:
                    last = status
                    gsta = status["gSTA"]
                    gflt = status["gFLT"]
                    logger.info(
                        f"[{self.side}] activating... gSTA={gsta} "
                        f"gACT={status['gACT']} gFLT={gflt} "
                        f"({FAULT_CODES.get(gflt, '?')}) gPR={status['gPR']}"
                    )
                    if gsta == GSTA_ACTIVE:
                        logger.success(
                            f"[{self.side}] gripper ACTIVE (gSTA=3)."
                        )
                        return True
                    if gflt not in (0, 5):  # 5 = activation not finished (expected)
                        logger.warning(
                            f"[{self.side}] unexpected fault during "
                            f"activation: gFLT={gflt} "
                            f"({FAULT_CODES.get(gflt, '?')})"
                        )
                time.sleep(0.3)

            last_sta = last["gSTA"] if last else "?"
            last_flt = last["gFLT"] if last else "?"
            logger.warning(
                f"[{self.side}] activation not complete after "
                f"{per_attempt_timeout}s (gSTA={last_sta}, gFLT={last_flt})."
            )

        logger.error(
            f"[{self.side}] FAILED to activate after {retries} attempts. "
            f"gACT={last.get('gACT') if last else '?'}/gSTA={last.get('gSTA') if last else '?'}. "
            f"If gACT=1 but gSTA stays 0, the gripper never started its "
            f"calibration move -> check the 24V motor supply and that the "
            f"fingers are free to move."
        )
        return False

    def set_position(self, normalized_pos: float) -> None:
        """Sends a position command (0.0 = open, 1.0 = close)."""
        self.send_command(build_position_packet(normalized_pos))

    def open(self) -> None:
        """Sends the fully-open command (matches OPEN_HEX)."""
        self.send_command(hex_to_bytes(OPEN_HEX))

    def close(self) -> None:
        """Sends the fully-close command (matches CLOSE_HEX)."""
        self.send_command(hex_to_bytes(CLOSE_HEX))

    # -- status monitoring thread --------------------------------------------

    def start_monitor(self, interval: float = 0.5) -> None:
        """Starts a daemon thread that periodically logs the gripper status."""
        if self._monitor_thread is not None:
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"[{self.side}] started status monitoring ({1.0/interval:.1f}Hz).")

    def stop_monitor(self) -> None:
        """Stops the status monitoring thread."""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
            logger.info(f"[{self.side}] stopped status monitoring.")

    def _monitor_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            status = self.read_status(timeout=0.1)
            if status:
                logger.info(
                    f"[{self.side}] gPR={status['gPR']} "
                    f"STA={status['gSTA']} FLT={status['gFLT']} "
                    f"CU={status['gCU']}"
                )
            time.sleep(interval)

    # -- teardown -------------------------------------------------------------

    def shutdown(self) -> None:
        self.stop_monitor()
        if self.ser is not None:
            self.ser.close()
            logger.info(f"[{self.side}] closed serial port.")


def main(
    side: Literal["left", "right", "both"] = "both",
    left_port: str = DEFAULT_SERIAL_PORTS["left"],
    right_port: str = DEFAULT_SERIAL_PORTS["right"],
    skip_activate: bool = False,
) -> None:
    """Control the Robotiq 2F-85 gripper(s) over USB serial.

    Runs the sequence: activate (blocking, until gSTA==3) -> close -> open,
    while a background thread polls and logs the gripper status.

    Args:
        side: Which arm's gripper to communicate with.
        left_port: Serial device for the left arm (default /dev/ttyUSB0).
        right_port: Serial device for the right arm (default /dev/ttyUSB1).
        skip_activate: Skip the blocking activation step (use only if the
            gripper is already activated).
    """
    port_map: Dict[str, str] = {}
    if side in ("left", "both"):
        port_map["left"] = left_port
    if side in ("right", "both"):
        port_map["right"] = right_port
    if not port_map:
        logger.error("No side selected. Use --side {left,right,both}.")
        return

    grippers: Dict[str, RobotiqUsb] = {}
    try:
        for arm_side, port in port_map.items():
            grippers[arm_side] = RobotiqUsb(port=port, side=arm_side)

        # --- 1. Activate (block until each gripper reports gSTA == 3) -------
        if not skip_activate:
            for arm_side, g in grippers.items():
                if not g.activate():
                    logger.error(
                        f"[{arm_side}] gripper is NOT active; move commands "
                        f"will be ignored (fault 5). Aborting this arm."
                    )
                    # Keep going for the other arm; this one just won't move.

        # --- 2. Start status monitoring (only after activation is clean) ----
        for g in grippers.values():
            g.start_monitor(interval=0.5)

        # --- 3. Close (fully closed) ----------------------------------------
        logger.info("Sending CLOSE ...")
        for arm_side, g in grippers.items():
            logger.info(f"[{arm_side}] close ...")
            g.close()
        time.sleep(5)

        # --- 4. Open (fully open) -------------------------------------------
        logger.info("Sending OPEN ...")
        for arm_side, g in grippers.items():
            logger.info(f"[{arm_side}] open ...")
            g.open()
        time.sleep(5)

    finally:
        for g in grippers.values():
            g.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
