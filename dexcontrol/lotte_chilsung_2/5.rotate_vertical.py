# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Step 3 — Rotate a horizontally-held box to vertical (left up, right down).

Pipeline (lotte_chilsung):
  1. grasp   : teleop / learned policy (assume already holding)
  2. align   : align_horizontal.py — level the box + lift with arms/torso
  3. rotate  : THIS SCRIPT — cooperative two-hand tip to vertical

Assumes the box is already gripped with the long axis roughly horizontal
(after step 2). Rotates about an axis perpendicular to the long axis so the
left hand rises and the right hand lowers until the long axis is vertical.
Rigid-grasp constraint is kept for the whole path. If fully vertical is
unreachable, the largest reachable tilt is used.

By default the torso is unlocked with low damping so IK can freely use
arms + torso. ``hold_pos`` defaults to the current grasp center (no pull).
After the tip the script exits on its own without ``stop()`` (so the pose
stays held) but still closes the Zenoh client so the next teleop/control
session can take over.

Requires a running robot or simulator reachable via Zenoh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tyro
from dexcomm import RateLimiter
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import (  # noqa: E402
    LEFT_ARM_JOINTS,
    MAX_JOINT_JUMP,
    RIGHT_ARM_JOINTS,
    RigidGraspPlanner,
    create_motion_manager,
    disconnect_leave_held,
    execute_trajectory,
    plan_box_motion,
    connect_bot,
)
from dexcontrol.robot import Robot
from dexmotion.motion_manager import MotionManager


class BoxVerticalPlanner(RigidGraspPlanner):
    """Tip the box long axis from horizontal toward world ±Z."""

    def __init__(self, mm: MotionManager, tilt_direction: str = "up"):
        super().__init__(mm)
        z = np.array([0.0, 0.0, 1.0])
        target = z if tilt_direction == "up" else -z
        rot_axis = np.cross(self.box_axis0, target)
        n = np.linalg.norm(rot_axis)
        if n < 1e-8:
            # Already parallel to ±z.
            self.rot_axis = np.array([1.0, 0.0, 0.0])
            self.full_angle = 0.0 if self.box_axis0 @ target > 0 else np.pi
        else:
            self.rot_axis = rot_axis / n
            self.full_angle = float(
                np.arccos(np.clip(self.box_axis0 @ target, -1.0, 1.0))
            )
        logger.info(
            f"Rotate: axis {np.round(self.box_axis0, 3)} -> {target}, "
            f"angle {np.rad2deg(self.full_angle):.1f} deg "
            f"(tilt_direction={tilt_direction})"
        )


def main(
    hold_pos: tuple[float, float, float] | None = None,
    duration: float = 8.0,
    control_hz: float = 100.0,
    torso_damping: float = 1.0,
    lock_torso: bool = False,
    tilt_direction: str = "up",
    leave_held: bool = True,
) -> None:
    """Rotate a bimanually-held box from horizontal to vertical.

    Left hand ends above the right hand when tilt_direction is \"up\".

    Args:
        hold_pos: Box-center target (robot base frame, meters) while rotating.
            ``None`` (default) keeps the current grasp center (no pull).
        duration: Execution time of the rotation in seconds.
        control_hz: Joint command streaming frequency in Hz.
        torso_damping: Local-IK damping for torso joints when unlocked.
            Low (default 1) lets the torso help find solutions; high
            (~100) keeps it nearly still. Ignored when ``lock_torso=True``.
        lock_torso: If True, hard-freeze torso (arm-only IK). Off by default.
        tilt_direction: \"up\" raises left / lowers right; \"down\" reverses.
        leave_held: If True (default), exit without calling ``stop()`` so the
            last joint targets stay and the arms keep holding, but still close
            the Zenoh client so the next teleop/control session can take over.
            Use ``--no-leave-held`` for a full shutdown (arms go limp).
    """
    # auto_shutdown=False: avoid signal/__del__ path calling stop()+shutdown()
    # which would release the final pose after we exit.
    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        # GlobalHybridIK / MotionManager init can peg the CPU for >1s. The sim
        # heartbeat may stall under that load, and dexcontrol then os._exit(1).
        # Pause only around planning; resume before any motion execute.
        hb = getattr(bot, "heartbeat", None)
        if hb is not None:
            hb.pause()
            logger.info("Heartbeat paused during motion planning.")

        try:
            mm, torso_joints, initial_joint_pos = create_motion_manager(
                bot,
                torso_damping=torso_damping,
                lock_torso=lock_torso,
                # Forward track uses local Pink IK; GlobalHybrid only needed if
                # that fails and we fall back to backward planning.
                init_global_ik=False,
            )
            if lock_torso:
                logger.info("Torso locked — planning arm-only vertical tip.")
            else:
                logger.info(
                    f"Torso unlocked with damping={torso_damping:g} "
                    "(torso free to assist IK)."
                )
            planner = BoxVerticalPlanner(mm, tilt_direction=tilt_direction)

            if hold_pos is None:
                target_hold = planner.center0.copy()
                logger.info(
                    f"hold_pos unset — keeping current box center "
                    f"{np.round(target_hold, 3)}"
                )
            else:
                target_hold = np.asarray(hold_pos, dtype=float)
                logger.info(f"hold_pos override {np.round(target_hold, 3)}")

            logger.info(
                "Planning vertical rotation (validated before any movement)..."
            )
            result = plan_box_motion(
                planner,
                hold_pos=target_hold,
                prefer_forward=True,  # already holding after align
                angle_label="vertical tilt",
            )
            if result is None:
                logger.error("Rotation planning failed. Robot was not moved.")
                return

            start_config, configs = result
            exec_names = (
                [f"L_arm_j{i}" for i in range(1, 8)]
                + [f"R_arm_j{i}" for i in range(1, 8)]
                + torso_joints
            )
            mismatch = max(
                abs(start_config[n] - initial_joint_pos[n]) for n in exec_names
            )
            if mismatch > MAX_JOINT_JUMP:
                logger.error(
                    f"Planned start differs from current config by {mismatch:.2f} rad. "
                    "Run align_horizontal.py first, then retry."
                )
                return

            if not configs:
                logger.info("Already vertical; nothing to execute.")
                return
        finally:
            if hb is not None and hb.is_paused():
                hb.resume()
                logger.info("Heartbeat resumed.")

        logger.info("Executing vertical rotation...")
        execute_trajectory(
            bot, torso_joints, start_config, configs, duration, control_hz
        )

        # Re-assert the final setpoint a few times so the last command buffer
        # is firmly the hold pose before we disconnect.
        final_q = configs[-1]
        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        q = np.array([final_q[n] for n in names], dtype=float)
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.left_arm.set_joint_pos(q[:7])
            bot.right_arm.set_joint_pos(q[7:14])
            if torso_joints:
                bot.torso.set_joint_pos(q[14:])
            rate.sleep()

        logger.info("Rotation completed; final pose left held.")
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
