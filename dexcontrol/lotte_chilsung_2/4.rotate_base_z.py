# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""In-place chassis yaw: rotate the base about its Z axis by a chosen angle.

Open-loop timing, same command path as the omniteleop UI buttons
(``chassis.set_velocity`` with wz = 1.0 rad/s). Positive yaw follows the
right-hand rule (counter-clockwise when viewed from above = left turn);
negative = clockwise (right turn).

Notes:
  * dexcontrol's timed chassis command drives for ``wait_time − 1 s``; the
    extra second is added back here so the turn lasts the full
    ``|yaw| / angular_speed``.
  * The simulator runs below real time, so a wall-clock turn covers fewer
    sim-degrees. Use ``--time-scale`` (e.g. 1.5) if the base stops short.
  * Arms / torso / head keep their last commanded targets on exit
    (``--no-leave-held`` for a full shutdown that releases them).

Example:
    python rotate_base_z.py            # +90 deg (default)
    python rotate_base_z.py 80         # +80 deg (left)
    python rotate_base_z.py -45        # -45 deg (right)
    python rotate_base_z.py 80 --time-scale 1.5
    python rotate_base_z.py 80 --angular-speed 0.5
"""

import sys
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from tyro.conf import Positional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import connect_bot, disconnect_leave_held  # noqa: E402
from dexcontrol.utils.compat import supported_models

# dexcontrol core/chassis.py::_execute_timed_command runs for wait_time - 1.0 s.
_DEXCONTROL_TIMED_CMD_OFFSET_S = 1.0


@supported_models("vega_1", "vega_1p")
def main(
    yaw_deg: Positional[float] = 90.0,
    angular_speed: float = 1.0,
    time_scale: float = 1.0,
    leave_held: bool = True,
) -> None:
    """Rotate the chassis in place about the base Z axis.

    Args:
        yaw_deg: Rotation angle in degrees about base Z (positional).
            Positive = counter-clockwise (left turn), negative = clockwise.
        angular_speed: Absolute yaw rate in rad/s (sign comes from yaw_deg).
            Default 1.0 = the omniteleop UI rotate buttons.
        time_scale: Multiplier on the drive time, to compensate a simulator
            running below real time (1.5 ≈ 0.67× real time).
        leave_held: Exit without ``stop()`` so arms/torso keep their last
            targets (safe while holding a box). ``--no-leave-held`` does a full
            shutdown, which releases the arms.
    """
    if angular_speed <= 0.0:
        raise ValueError("angular_speed must be positive")
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive")

    yaw_rad = float(np.deg2rad(yaw_deg))
    if abs(yaw_rad) < 1e-6:
        logger.info("yaw_deg = 0; nothing to do.")
        return
    wz = float(np.sign(yaw_rad) * angular_speed)
    drive_time = abs(yaw_rad) / angular_speed * time_scale
    wait_time = drive_time + _DEXCONTROL_TIMED_CMD_OFFSET_S

    logger.info(
        f"Rotating {yaw_deg:+.1f} deg about base Z "
        f"(wz={wz:+.3f} rad/s, drive {drive_time:.2f} s, time_scale {time_scale:g})"
    )

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        chassis = bot.chassis
        if abs(wz) > chassis.max_ang_vel:
            logger.warning(
                f"wz {abs(wz):.2f} exceeds chassis max {chassis.max_ang_vel:.2f} rad/s; it will be clipped."
            )
        logger.info("Rotating in place about base center...")
        chassis.turn(wz, wait_time=wait_time, center="base_center")
        chassis.stop()
        logger.info("Base Z rotation completed.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        try:
            bot.chassis.stop()
        except Exception:
            pass
    finally:
        if do_shutdown:
            logger.info("Shutting down robot (--no-leave-held)")
            bot.shutdown()
        else:
            disconnect_leave_held(bot)
            logger.info("Client disconnected without stop() — arm/torso targets held.")


if __name__ == "__main__":
    tyro.cli(main)
