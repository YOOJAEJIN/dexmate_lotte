# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Box lift up — joint-space move from the current pose to a fixed lift pose.

While holding a box, arm joints often never settle inside the motion-plugin
tolerance, so ``move_to_joint_pos().wait()`` times out on ``left_arm``. This
script instead:

  1. Cosine-blends all joints open-loop (duration sized for torso travel).
  2. Keeps commanding the final pose until the *torso* settles (or timeout).
  3. Exits without ``stop()`` so the last joint targets stay held (same as
     ``rotate_vertical.py --leave-held``).

Only arms and torso are commanded; the head (camera) is left untouched.

Requires a running robot or simulator reachable via Zenoh.

Example:
    python box_lift_up.py
    python box_lift_up.py --max-joint-vel 0.25 --settle-timeout 25
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tyro
from dexcomm import RateLimiter
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import connect_bot, disconnect_leave_held  # noqa: E402
from dexcontrol.robot import Robot

# Recorded lift pose (rad).
BOX_LIFT_UP_TARGET: dict[str, float] = {
    'L_arm_j1': 0.8464568257331848,
 'L_arm_j2': 0.002086389809846878,
 'L_arm_j3': 0.14785179495811462,
 'L_arm_j4': -1.4432071447372437,
 'L_arm_j5': -0.0009538692538626492,
 'L_arm_j6': -0.3743095099925995,
 'L_arm_j7': 0.0016423321794718504,
 'R_arm_j1': -0.837252676486969,
 'R_arm_j2': -0.0018707311246544123,
 'R_arm_j3': -0.14797762036323547,
 'R_arm_j4': -1.4384677410125732,
 'R_arm_j5': -0.0016088859410956502,
 'R_arm_j6': 0.3746691346168518,
 'R_arm_j7': -0.0012871582293882966,
 'head_j1': -0.07311095297336578,
 'head_j2': 0.0001068116253009066,
 'head_j3': -0.00016859735478647053,
 'torso_j1': 0.5668046474456787,
 'torso_j2': 1.0439049005508423,
 'torso_j3': 0.40320807695388794}





LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]
# Head is intentionally NOT commanded: the head camera stays where teleop left it.
JOINT_NAMES = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + TORSO_JOINTS
COMPONENTS = ["left_arm", "right_arm", "torso"]


def _command(bot: Robot, q: np.ndarray) -> None:
    bot.left_arm.set_joint_pos(q[:7])
    bot.right_arm.set_joint_pos(q[7:14])
    bot.torso.set_joint_pos(q[14:17])


def main(
    duration: float | None = None,
    max_joint_vel: float = 0.25,
    control_hz: float = 100.0,
    settle_timeout: float = 25.0,
    torso_tol: float = 0.08,
    leave_held: bool = True,
) -> None:
    """Move from the current joint pose to the box-lift-up pose.

    Args:
        duration: Fixed blend time in seconds. If None, derived from torso
            travel / max_joint_vel (clamped 3–20 s).
        max_joint_vel: Used when duration is None (rad/s). Keep modest so the
            torso can track under box load.
        control_hz: Command streaming frequency in Hz.
        settle_timeout: After the blend, keep holding the final command until
            torso error <= torso_tol, or this many seconds elapse.
        torso_tol: Max torso joint error (rad) to treat as settled.
        leave_held: If True (default), exit without calling ``stop()`` so the
            last joint targets stay and the arms keep holding. Use
            ``--no-leave-held`` for a full shutdown (arms go limp).
    """
    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        current = bot.get_joint_pos_dict(component=COMPONENTS)
        q0 = np.array([current[n] for n in JOINT_NAMES], dtype=float)
        q1 = np.array([BOX_LIFT_UP_TARGET[n] for n in JOINT_NAMES], dtype=float)

        travel = float(np.abs(q1 - q0).max())
        torso_travel = float(np.abs(q1[14:17] - q0[14:17]).max())
        if travel < 1e-3:
            logger.info("Already at box-lift-up pose; nothing to execute.")
            if leave_held:
                # Still re-assert current/target and leave held.
                rate = RateLimiter(control_hz)
                for _ in range(max(int(0.2 * control_hz), 5)):
                    _command(bot, q1)
                    rate.sleep()
                disconnect_leave_held(bot)
                do_shutdown = False
                logger.info("Client disconnected without stop() — pose held.")
            return

        if duration is None:
            # Size time for the torso climb; arms are usually already near target.
            ref = max(torso_travel, travel)
            duration = float(np.clip(ref / max_joint_vel, 3.0, 20.0))

        logger.info(
            f"Box lift up blend: travel {travel:.3f} rad "
            f"(torso {torso_travel:.3f} rad), {duration:.1f}s @ {control_hz:.0f} Hz"
        )

        rate = RateLimiter(control_hz)
        n_ticks = max(int(duration * control_hz), 1)
        for t in range(1, n_ticks + 1):
            a = 0.5 * (1.0 - np.cos(np.pi * t / n_ticks))
            _command(bot, (1.0 - a) * q0 + a * q1)
            rate.sleep()

        # Hold final targets until torso catches up (arms under box load often
        # never hit a tight absolute pose — do not wait on arm motion status).
        logger.info(
            f"Settling torso (tol={torso_tol:.2f} rad, timeout={settle_timeout:.0f}s)..."
        )
        settle_deadline = time.monotonic() + settle_timeout
        torso_err = float("inf")
        while time.monotonic() < settle_deadline:
            _command(bot, q1)
            reached = bot.get_joint_pos_dict(component=COMPONENTS)
            torso_err = max(
                abs(reached[n] - BOX_LIFT_UP_TARGET[n]) for n in TORSO_JOINTS
            )
            if torso_err <= torso_tol:
                break
            rate.sleep()

        # Re-assert final setpoint before disconnect so the hold buffer is firm.
        for _ in range(max(int(0.2 * control_hz), 5)):
            _command(bot, q1)
            rate.sleep()

        reached = bot.get_joint_pos_dict(component=COMPONENTS)
        err = max(abs(reached[n] - BOX_LIFT_UP_TARGET[n]) for n in JOINT_NAMES)
        torso_err = max(abs(reached[n] - BOX_LIFT_UP_TARGET[n]) for n in TORSO_JOINTS)
        if torso_err > torso_tol:
            logger.warning(
                f"Torso did not settle (error {torso_err:.2f} rad). "
                f"Retry with --max-joint-vel 0.2 or --settle-timeout 40."
            )
        else:
            logger.info(
                f"Box lift up completed "
                f"(torso residual {torso_err:.3f} rad, overall {err:.3f} rad)."
            )

        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info(
                "Client disconnected without stop() — pose held. "
                "Next teleop/control commands will take over."
            )
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        do_shutdown = True
    except TimeoutError as exc:
        logger.error(f"Motion timed out: {exc}")
        do_shutdown = True
    finally:
        if do_shutdown:
            logger.info("Shutting down robot")
            bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
