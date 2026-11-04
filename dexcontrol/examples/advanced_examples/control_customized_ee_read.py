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

Example use case: Robotiq gripper activation via Modbus RTU.
"""

import threading
import time
from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models

READ_STATUS_HEX = "09 04 07 D0 00 03 B1 CE"

def parse_and_log_status(response_data: bytes) -> None:
    """Parses the 6-byte status response from the Robotiq gripper."""
    if len(response_data) < 7:  # 1 byte for func code + 6 bytes for data
        logger.warning(f"Status response too short: {response_data.hex(' ')}")
        return

    # Expected response format: 09 04 06 [Byte 0] [Byte 1] ... [Byte 5] [CRC1] [CRC2]
    # We are interested in the data payload which starts after the function code and byte count.
    # The actual data we need is in the latter part of the response.
    # Let's find the start of the actual register data.
    # The response to a "Read Input Registers" (0x04) command is:
    # [Slave ID] [Func Code] [Byte Count] [Data 1] [Data 2] ... [CRC]
    # The data we want is from the 4th byte onwards.
    header = response_data[:3]  # Slave ID, Func Code, Byte Count
    payload = response_data[3:-2]  # Data bytes

    # activation, open, close 시 처리
    if header[1] == 0x10:  # Ignore Write Multiple Registers response
        return

    # READ_STATUS_HEX 가 아닌 경우
    if header[1] != 0x04 or header[2] != 0x06:
        logger.warning(f"Unexpected status response header: {header.hex(' ')}")
        return

    status = {
        "gACT": (payload[0] & 0b00000001),  # Gripper Activation Status
        "gGTO": (payload[0] & 0b00001000) >> 3,  # Action Status (go to)
        "gSTA": (payload[0] & 0b00110000) >> 4,  # Gripper Status
        "gOBJ": (payload[0] & 0b11000000) >> 6,  # Object Detection Status
        "gFLT": (payload[2] & 0b00001111),  # Fault Code
        "gPO": payload[3],  # 0 ~ 255, Position Request Echo: action에 해당
        "gPR": payload[4],  # 0 ~ 255, Position 그리퍼 실시간 현재 위치
        "gCU": payload[5],  # 0 ~ 255, Current 모터 전류
    }
    logger.info(f"gPR: {status['gPR']}, Status: {status}")

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


@supported_models("vega_1", "vega_1p", "vega_1u")
def main(
    side: Literal["left", "right", "both"] = "left",
    message_hex: str = "09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30",
    wait_response: bool = True,
    timeout: float = 1.0,
) -> None:
    """Send a pass-through message to the customized end effector.

    Args:
        side: Which arm's end effector to communicate with.
        message_hex: Hex string of the message to send (space-separated bytes).
            Default is Robotiq activation command.
        wait_response: Whether to wait for and display the response.
        timeout: Maximum time to wait for response in seconds.
    """
    with Robot() as bot:
        # Start status monitoring thread
        stop_event = threading.Event()
        monitor_thread = None

        if side in ("left", "both") and bot.left_arm.enable_ee_pass_through:
            monitor_thread = threading.Thread(
                target=status_monitor_thread,
                args=(bot.left_arm, stop_event),
                daemon=True
            )
            monitor_thread.start()
            logger.info("Started status monitoring thread for left arm")

        # Convert hex string to bytes
        message = bytes.fromhex(message_hex.replace(" ", ""))
        logger.info(f"Sending message: {message.hex(' ')}")

        if side in ("left", "both"):
            if bot.left_arm.enable_ee_pass_through:
                logger.info("Sending to left arm EE...")
                bot.left_arm.send_ee_pass_through_message(message)

                # if wait_response:
                #     start = time.time()
                #     while time.time() - start < timeout:
                #         response = bot.left_arm.get_ee_pass_through_response()
                #         if response is not None:
                #             logger.info(f"Left arm EE response: {response}")
                #             break
                #         time.sleep(0.01)
                #     else:
                #         logger.warning("Left arm: No response received within timeout")
            else:
                logger.warning(
                    "Left arm: EE pass-through not enabled (known EE type detected)"
                )

        if side in ("right", "both"):
            if bot.right_arm.enable_ee_pass_through:
                logger.info("Sending to right arm EE...")
                bot.right_arm.send_ee_pass_through_message(message)

                # if wait_response:
                #     start = time.time()
                #     while time.time() - start < timeout:
                #         response = bot.right_arm.get_ee_pass_through_response()
                #         if response is not None:
                #             logger.info(f"Right arm EE response: {response}")
                #             break
                #         time.sleep(0.01)
                #     else:
                #         logger.warning("Right arm: No response received within timeout")
            else:
                logger.warning(
                    "Right arm: EE pass-through not enabled (known EE type detected)"
                )

        time.sleep(5)

        CLOSE_HEX = "09 10 03 E8 00 03 06 09 00 00 FF FF FF 42 29"
        close_message = bytes.fromhex(CLOSE_HEX.replace(" ", ""))

        if side in ("left", "both"):
            if bot.left_arm.enable_ee_pass_through:
                logger.info("Sending to left arm EE...")
                bot.left_arm.send_ee_pass_through_message(close_message)

                # if wait_response:
                #     start = time.time()
                #     while time.time() - start < timeout:
                #         response = bot.left_arm.get_ee_pass_through_response()
                #         if response is not None:
                #             logger.info(f"Left arm EE response: {response}")
                #             break
                #         time.sleep(0.01)
                #     else:
                #         logger.warning("Left arm: No response received within timeout")
            else:
                logger.warning(
                    "Left arm: EE pass-through not enabled (known EE type detected)"
                )

        time.sleep(5)
        OPEN_HEX = "09 10 03 E8 00 03 06 09 00 00 00 FF FF 72 19"
        open_message = bytes.fromhex(OPEN_HEX.replace(" ", ""))

        if side in ("left", "both"):
            if bot.left_arm.enable_ee_pass_through:
                logger.info("Sending to left arm EE...")
                bot.left_arm.send_ee_pass_through_message(open_message)

                # if wait_response:
                #     start = time.time()
                #     while time.time() - start < timeout:
                #         response = bot.left_arm.get_ee_pass_through_response()
                #         if response is not None:
                #             logger.info(f"Left arm EE response: {response}")
                #             break
                #         time.sleep(0.01)
                #     else:
                #         logger.warning("Left arm: No response received within timeout")
            else:
                logger.warning(
                    "Left arm: EE pass-through not enabled (known EE type detected)"
                )

        # Stop monitoring thread before exiting
        if monitor_thread:
            logger.info("Stopping status monitoring thread...")
            stop_event.set()
            monitor_thread.join(timeout=1.0)

if __name__ == "__main__":
    tyro.cli(main)
