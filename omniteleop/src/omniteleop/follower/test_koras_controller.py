# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating customized end-effector control via pass-through.

This script shows how to send raw RS485 messages to a customized end effector
and receive responses. This is useful for controlling custom end effectors
that are not natively supported by the robot.

NOTE: This only works when the end effector type is UNKNOWN. If a Hand (F5D6)
or DexGripper is detected, EE pass-through is disabled.

Example use case: KORAS KR-GCS gripper control via Modbus RTU.
"""

import threading
import time
from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models

KORAS_SLAVE_ID = 0x01
KORAS_COMMAND_ADDRESS = 0x0000
KORAS_STATUS_ADDRESS = 0x000A
KORAS_STATUS_REGISTER_COUNT = 5
KORAS_BAUD_RATE = 115200

KORAS_CMD_INIT = 101
KORAS_CMD_OPEN = 102
KORAS_CMD_CLOSE = 103

# READ_STATUS: Read Holding Registers 10..14 (status, motor pos/current/velocity, finger pos)
READ_STATUS_HEX = "01 03 00 0A 00 05 A5 CB"
# INIT: Write Command=101 (initialize) to Holding Registers 0 and 1
INIT_HEX = "01 10 00 00 00 02 04 00 65 00 00 E3 B0"
# CLOSE: Write Command=103 (close) to Holding Registers 0 and 1
CLOSE_HEX = "01 10 00 00 00 02 04 00 67 00 00 42 70"
# OPEN: Write Command=102 (open) to Holding Registers 0 and 1
OPEN_HEX = "01 10 00 00 00 02 04 00 66 00 00 13 B0"


def modbus_crc16(data: bytes) -> int:
    """Return Modbus RTU CRC16 for data without CRC bytes."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def with_modbus_crc(data: bytes) -> bytes:
    crc = modbus_crc16(data)
    return data + bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def koras_read_status_message() -> bytes:
    """Read Holding Registers 10..14: status, motor pos/current/velocity, finger pos."""
    payload = bytes(
        (
            KORAS_SLAVE_ID,
            0x03,
            (KORAS_STATUS_ADDRESS >> 8) & 0xFF,
            KORAS_STATUS_ADDRESS & 0xFF,
            (KORAS_STATUS_REGISTER_COUNT >> 8) & 0xFF,
            KORAS_STATUS_REGISTER_COUNT & 0xFF,
        )
    )
    return with_modbus_crc(payload)


def koras_command_message(command: int, value: int = 0) -> bytes:
    """Write Command and Target value 1 to Holding Registers 0 and 1."""
    payload = bytes(
        (
            KORAS_SLAVE_ID,
            0x10,
            (KORAS_COMMAND_ADDRESS >> 8) & 0xFF,
            KORAS_COMMAND_ADDRESS & 0xFF,
            0x00,
            0x02,
            0x04,
            (command >> 8) & 0xFF,
            command & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        )
    )
    return with_modbus_crc(payload)


def parse_and_log_status(response_data: bytes) -> None:
    """Parses the KORAS KR-GCS status response from Holding Registers 10..14."""
    if len(response_data) < 5:  # Slave ID, Func Code, Byte Count, payload, CRC
        logger.warning(f"Status response too short: {response_data.hex(' ')}")
        return

    function_code = response_data[1]
    if function_code == 0x10:  # Ignore Write Multiple Registers response
        return

    if function_code != 0x03:
        logger.warning(f"Unexpected function code: {response_data.hex(' ')}")
        return

    byte_count = response_data[2]
    expected_byte_count = KORAS_STATUS_REGISTER_COUNT * 2
    if byte_count != expected_byte_count or len(response_data) < 3 + byte_count + 2:
        logger.warning(f"Unexpected status response: {response_data.hex(' ')}")
        return

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
    logger.info(f"finger_position: {status['finger_position']}, Status: {status}")


def status_monitor_thread(arm, stop_event):
    """Thread function to continuously monitor gripper status."""
    status_message = bytes.fromhex(READ_STATUS_HEX.replace(" ", ""))
    last_timestamp = 0

    while not stop_event.is_set():
        arm.send_ee_pass_through_message(status_message)
        # logger.info(f"Status Message is sent: ----------------------------------")

        start = time.time()
        while time.time() - start < 0.1:
            response = arm.get_ee_pass_through_response()
            if response and response["timestamp_ns"] > last_timestamp:
                last_timestamp = response["timestamp_ns"]
                logger.info(f"timestamp_ns: {last_timestamp}, response data: {response['data']}")
                parse_and_log_status(response['data'])
                break
            time.sleep(0.01)

        time.sleep(0.5)


def ensure_ee_baud_rate(arm, baud_rate: int) -> bool:
    """Verify/set the arm EE baud rate so RS485 matches the KORAS gripper."""
    result = arm.get_ee_baud_rate()
    if not result.get("success"):
        logger.error(f"Failed to get arm EE baud rate: {result.get('message')}")
        return False

    current_baud_rate = result.get("baud_rate")
    logger.info(f"Arm current EE baud rate: {current_baud_rate}")
    if current_baud_rate == baud_rate:
        return True

    logger.info(f"Setting arm EE baud rate to {baud_rate}...")
    result = arm.set_ee_baud_rate(baud_rate)
    if not result.get("success"):
        logger.error(f"Failed to set arm EE baud rate: {result.get('message')}")
        return False

    result = arm.get_ee_baud_rate()
    logger.info(f"Arm EE baud rate after setting: {result.get('baud_rate')}")
    return result.get("success") and result.get("baud_rate") == baud_rate


@supported_models("vega_1", "vega_1p", "vega_1u")
def main(
    side: Literal["left", "right", "both"] = "left",
    baud_rate: int = KORAS_BAUD_RATE,
    message_hex: str = INIT_HEX,
    wait_response: bool = True,
    timeout: float = 1.0,
) -> None:
    """Send a pass-through message to the customized end effector.

    Args:
        side: Which arm's end effector to communicate with.
        baud_rate: End-effector RS485 baud rate.
        message_hex: Hex string of the message to send (space-separated bytes).
            Default is KORAS initialize command.
        wait_response: Whether to wait for and display the response.
        timeout: Maximum time to wait for response in seconds.
    """
    with Robot() as bot:
        # KORAS ready check is intended for the left arm.
        if side != "left":
            logger.warning("This KORAS controller test is intended for the left arm.")

        arm = bot.left_arm
        if not arm.enable_ee_pass_through:
            logger.error(
                "Left arm: EE pass-through not enabled. "
                "Customized EE must be detected as UNKNOWN before RS485 can work."
            )
            return

        if not ensure_ee_baud_rate(arm, baud_rate):
            logger.error("Cannot verify/set EE baud rate. Stopping before RS485 check.")
            return

        # Start status monitoring thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=status_monitor_thread,
            args=(arm, stop_event),
            daemon=True
        )
        monitor_thread.start()
        logger.info("Started status monitoring thread for left arm")

        # Convert hex strings to bytes
        read_status_message = bytes.fromhex(READ_STATUS_HEX.replace(" ", ""))
        init_message = bytes.fromhex(message_hex.replace(" ", ""))
        close_message = bytes.fromhex(CLOSE_HEX.replace(" ", ""))
        open_message = bytes.fromhex(OPEN_HEX.replace(" ", ""))

        # --- RS485 connection check with status read ----------------------
        logger.info(f"Checking left arm RS485 connection: {read_status_message.hex(' ')}")
        arm.send_ee_pass_through_message(read_status_message)

        if wait_response:
            start = time.time()
            got_response = False
            while time.time() - start < timeout:
                response = arm.get_ee_pass_through_response()
                if response is not None:
                    logger.info(f"Left arm EE response: {response}")
                    parse_and_log_status(response['data'])
                    got_response = True
                    break
                time.sleep(0.01)
            if not got_response:
                logger.warning(
                    "No status response from KORAS. Check RS485 A/B wiring, power, "
                    "slave ID, baud rate, and whether EE pass-through is enabled."
                )

        # --- Initialize ----------------------------------------------------
        logger.info(f"Sending INITIALIZE: {init_message.hex(' ')}")
        logger.info("Sending INITIALIZE to left arm EE...")
        arm.send_ee_pass_through_message(init_message)

        if wait_response:
            start = time.time()
            while time.time() - start < timeout:
                response = arm.get_ee_pass_through_response()
                if response is not None:
                    logger.info(f"Left arm EE response: {response}")
                    break
                time.sleep(0.01)
            else:
                logger.warning("Left arm: No response received within timeout")

        time.sleep(5)

        # --- Close ---------------------------------------------------------
        logger.info(f"Sending CLOSE: {close_message.hex(' ')}")
        logger.info("Sending CLOSE to left arm EE...")
        arm.send_ee_pass_through_message(close_message)

        if wait_response:
            start = time.time()
            while time.time() - start < timeout:
                response = arm.get_ee_pass_through_response()
                if response is not None:
                    logger.info(f"Left arm EE response: {response}")
                    break
                time.sleep(0.01)
            else:
                logger.warning("Left arm: No response received within timeout")

        time.sleep(5)

        # --- Open ----------------------------------------------------------
        logger.info(f"Sending OPEN: {open_message.hex(' ')}")
        logger.info("Sending OPEN to left arm EE...")
        arm.send_ee_pass_through_message(open_message)

        if wait_response:
            start = time.time()
            while time.time() - start < timeout:
                response = arm.get_ee_pass_through_response()
                if response is not None:
                    logger.info(f"Left arm EE response: {response}")
                    break
                time.sleep(0.01)
            else:
                logger.warning("Left arm: No response received within timeout")

        # Stop monitoring thread before exiting
        logger.info("Stopping status monitoring thread...")
        stop_event.set()
        monitor_thread.join(timeout=1.0)
        logger.info("Stopped status monitoring thread for left arm")

if __name__ == "__main__":
    tyro.cli(main)
