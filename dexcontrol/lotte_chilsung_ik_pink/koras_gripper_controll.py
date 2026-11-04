# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Simple KORAS gripper open/close/position control via EE RS485 pass-through.

Supports the grippers mounted on both hands: commands can target the left,
right, or both arms ('both' actuates the two grippers simultaneously from
parallel threads over each arm's independent RS485 channel).
"""

import argparse
import threading
import time

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
KORAS_CMD_SET_FINGER_POSITION = 104


def modbus_crc16(data: bytes) -> int:
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


def koras_command_message(command: int, value: int = 0) -> bytes:
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


def koras_read_status_message() -> bytes:
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


KORAS_INIT_MESSAGE = koras_command_message(KORAS_CMD_INIT)
KORAS_OPEN_MESSAGE = koras_command_message(KORAS_CMD_OPEN)
KORAS_CLOSE_MESSAGE = koras_command_message(KORAS_CMD_CLOSE)
KORAS_READ_STATUS_MESSAGE = koras_read_status_message()


def set_position_value(value: str) -> int:
    try:
        position = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("set-position must be an integer") from exc

    if not 0 <= position <= 1000:
        raise argparse.ArgumentTypeError("set-position must be in range 0~1000")
    return position


def drain_ee_responses(arm, duration: float = 0.05) -> int:
    newest_timestamp = 0
    start = time.time()
    while time.time() - start < duration:
        response = arm.get_ee_pass_through_response()
        if response is not None:
            newest_timestamp = max(newest_timestamp, response["timestamp_ns"])
        time.sleep(0.01)
    return newest_timestamp


def wait_for_ee_response(arm, last_timestamp: int, timeout: float):
    start = time.time()
    while time.time() - start < timeout:
        response = arm.get_ee_pass_through_response()
        if response and response["timestamp_ns"] > last_timestamp:
            return response
        time.sleep(0.01)
    return None


def send_and_wait(arm, label: str, message: bytes, timeout: float):
    logger.info(f"{label}: sending {message.hex(' ')}")
    last_timestamp = drain_ee_responses(arm)
    arm.send_ee_pass_through_message(message)

    response = wait_for_ee_response(arm, last_timestamp, timeout)
    if response is None:
        logger.warning(f"{label}: no RS485 response within {timeout:.2f}s")
        return None

    response_data = response["data"]
    logger.info(
        f"{label}: timestamp_ns={response['timestamp_ns']}, "
        f"response={response_data.hex(' ')}"
    )
    return response_data


def parse_and_log_koras_status(response_data: bytes) -> None:
    if len(response_data) < 5:
        logger.warning(f"Status response too short: {response_data.hex(' ')}")
        return

    function_code = response_data[1]
    if function_code != 0x03:
        logger.warning(f"Unexpected status response: {response_data.hex(' ')}")
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
    logger.info(
        "KORAS status: "
        f"motor_enabled={bool(status_bits & (1 << 0))}, "
        f"opening={bool(status_bits & (1 << 5))}, "
        f"closing={bool(status_bits & (1 << 6))}, "
        f"motor_fault={bool(status_bits & (1 << 9))}, "
        f"finger_position={registers[4]}"
    )


def ensure_ee_baud_rate(arm, baud_rate: int, label: str = "left arm") -> bool:
    result = arm.get_ee_baud_rate()
    if not result.get("success"):
        logger.error(f"Failed to get {label} EE baud rate: {result.get('message')}")
        return False

    current_baud_rate = result.get("baud_rate")
    logger.info(f"{label} current EE baud rate: {current_baud_rate}")
    if current_baud_rate == baud_rate:
        return True

    logger.info(f"Setting {label} EE baud rate to {baud_rate}...")
    result = arm.set_ee_baud_rate(baud_rate)
    if not result.get("success"):
        logger.error(f"Failed to set {label} EE baud rate: {result.get('message')}")
        return False

    result = arm.get_ee_baud_rate()
    logger.info(f"{label} EE baud rate after setting: {result.get('baud_rate')}")
    return result.get("success") and result.get("baud_rate") == baud_rate


def build_command_message(command: str, set_position: int | None = None) -> bytes:
    if command == "set-position":
        if set_position is None:
            raise ValueError("set_position is required for set-position command")
        return koras_command_message(KORAS_CMD_SET_FINGER_POSITION, set_position)

    return {
        "initialize": KORAS_INIT_MESSAGE,
        "open": KORAS_OPEN_MESSAGE,
        "close": KORAS_CLOSE_MESSAGE,
    }[command]


def execute_gripper_command(
    arm,
    label: str,
    command: str,
    baud_rate: int = KORAS_BAUD_RATE,
    timeout: float = 1.0,
    set_position: int | None = None,
) -> bool:
    """Run one KORAS gripper command on a single arm's EE RS485 channel."""
    message = build_command_message(command, set_position)

    if not arm.enable_ee_pass_through:
        logger.error(
            f"{label}: EE pass-through not enabled. "
            "Customized EE must be detected as UNKNOWN before RS485 can work."
        )
        return False

    if not ensure_ee_baud_rate(arm, baud_rate, label):
        return False

    response = send_and_wait(arm, f"{label} {command}", message, timeout)
    if response is None:
        return False

    time.sleep(0.5)
    status_response = send_and_wait(
        arm, f"{label} status", KORAS_READ_STATUS_MESSAGE, timeout
    )
    if status_response is not None:
        parse_and_log_koras_status(status_response)
    return True


def run_on_robot(
    bot,
    side: str,
    command: str,
    set_position: int | None = None,
    baud_rate: int = KORAS_BAUD_RATE,
    timeout: float = 1.0,
) -> bool:
    """Run a gripper command on an already-connected Robot instance.

    'both' commands the two grippers from parallel threads so they open/close
    simultaneously; each arm has its own RS485 channel so they don't conflict.
    """
    if side not in ("left", "right", "both"):
        logger.error(f"Invalid gripper side: {side!r} (expected left/right/both)")
        return False

    arms = {
        "left": [("left arm", bot.left_arm)],
        "right": [("right arm", bot.right_arm)],
        "both": [("left arm", bot.left_arm), ("right arm", bot.right_arm)],
    }[side]

    if len(arms) == 1:
        label, arm = arms[0]
        return execute_gripper_command(
            arm, label, command, baud_rate, timeout, set_position
        )

    results: dict[str, bool] = {}

    def _worker(label: str, arm) -> None:
        results[label] = execute_gripper_command(
            arm, label, command, baud_rate, timeout, set_position
        )

    threads = [
        threading.Thread(target=_worker, args=(label, arm), daemon=True)
        for label, arm in arms
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [label for label, _ in arms if not results.get(label, False)]
    if failed:
        logger.error(f"Gripper command '{command}' failed on: {', '.join(failed)}")
        return False
    return True


@supported_models("vega_1", "vega_1p", "vega_1u")
def run(
    command: str,
    baud_rate: int,
    timeout: float,
    set_position: int | None = None,
    arm: str = "both",
) -> None:
    with Robot() as bot:
        run_on_robot(
            bot,
            side=arm,
            command=command,
            set_position=set_position,
            baud_rate=baud_rate,
            timeout=timeout,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open, close, or position-control the KORAS grippers "
            "(left/right/both hands) via EE RS485 pass-through."
        )
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="both",
        help="Which arm's gripper to command (default: both, simultaneously).",
    )
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--initialize", action="store_true", help="Initialize the KORAS gripper.")
    command.add_argument("--open", action="store_true", help="Open the KORAS gripper.")
    command.add_argument("--close", action="store_true", help="Close the KORAS gripper.")
    command.add_argument(
        "--set-position",
        type=set_position_value,
        metavar="0~1000",
        help="Move the KORAS gripper finger to a position in range 0(closed)~1000(open).",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=KORAS_BAUD_RATE,
        help="End-effector RS485 baud rate.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Maximum time to wait for each RS485 response in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initialize:
        command = "initialize"
    elif args.open:
        command = "open"
    elif args.close:
        command = "close"
    else:
        command = "set-position"

    run(
        command=command,
        baud_rate=args.baud_rate,
        timeout=args.timeout,
        set_position=args.set_position,
        arm=args.arm,
    )


if __name__ == "__main__":
    main()
