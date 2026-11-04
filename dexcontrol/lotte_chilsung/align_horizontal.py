# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Step 2 — Align both hands so the held box is horizontal to the ground.

Pipeline (lotte_chilsung):
  1. grasp   : teleop / learned policy (assume already holding)
  2. align   : THIS SCRIPT — level the box + lift with arms/torso
  3. rotate  : rotate_vertical.py — tip box to vertical (left up, right down)

Assumes the box is already gripped bimanually after teleop. Keeps the rigid
grasp (relative pose between hands fixed) while:
  - rotating the box long axis (L→R) into the horizontal plane, and
  - optionally snapping that axis to world ±Y (left–right), and
  - pulling/lifting the box center toward ``hold_pos``.

Low torso damping lets the torso help raise the box to a workable height
before the vertical rotation step.

Requires a running robot or simulator reachable via Zenoh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

# Allow `python align_horizontal.py` from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import (  # noqa: E402
    MAX_JOINT_JUMP,
    RigidGraspPlanner,
    create_motion_manager,
    execute_trajectory,
    plan_box_motion,
    connect_bot,
)
from dexcontrol.robot import Robot
from dexmotion.motion_manager import MotionManager


class BoxAlignPlanner(RigidGraspPlanner):
    """Level the held box (and optionally align long axis to ±Y)."""

    def __init__(self, mm: MotionManager, snap_to_y: bool = True):
        super().__init__(mm)

        axis = self.box_axis0.copy()
        if snap_to_y:
            # Left–right along world Y, preserving which hand is on +Y.
            sign = 1.0 if axis[1] >= 0.0 else -1.0
            target = np.array([0.0, sign, 0.0])
        else:
            # Flatten pitch only: keep yaw of the current long axis.
            target = axis.copy()
            target[2] = 0.0
            n = np.linalg.norm(target)
            if n < 1e-6:
                sign = 1.0 if self.left0.translation[1] >= self.right0.translation[1] else -1.0
                target = np.array([0.0, sign, 0.0])
            else:
                target /= n

        rot_axis = np.cross(axis, target)
        n_axis = np.linalg.norm(rot_axis)
        if n_axis < 1e-8:
            self.rot_axis = np.array([1.0, 0.0, 0.0])
            self.full_angle = 0.0
        else:
            self.rot_axis = rot_axis / n_axis
            self.full_angle = float(np.arccos(np.clip(axis @ target, -1.0, 1.0)))

        logger.info(
            f"Align: axis {np.round(axis, 3)} -> {np.round(target, 3)}, "
            f"angle {np.rad2deg(self.full_angle):.1f} deg "
            f"(snap_to_y={snap_to_y})"
        )


def main(
    hold_pos: tuple[float, float, float] = (0.30, 0.0, 0.88),
    duration: float = 6.0,
    control_hz: float = 100.0,
    torso_damping: float = 1.0,
    snap_to_y: bool = True,
) -> None:
    """Level a held box to be horizontal (and lift toward hold_pos).

    Args:
        hold_pos: Target box-center position (robot base frame, meters).
            Slightly higher / closer to the chest so step 3 stays reachable.
        duration: Execution time in seconds.
        control_hz: Joint command streaming frequency in Hz.
        torso_damping: Local-IK damping for torso joints. Keep low so the
            torso can pitch/lift with the arms.
        snap_to_y: If True, also align the box long axis to world ±Y
            (left–right). If False, only remove pitch (same height hands).
    """
    bot = connect_bot()
    try:
        mm, torso_joints, initial_joint_pos = create_motion_manager(
            bot, torso_damping=torso_damping, init_global_ik=False
        )
        planner = BoxAlignPlanner(mm, snap_to_y=snap_to_y)

        logger.info("Planning horizontal alignment (validated before any movement)...")
        result = plan_box_motion(
            planner,
            hold_pos=np.asarray(hold_pos, dtype=float),
            prefer_forward=True,  # already holding: stay on current branch
            angle_label="align",
        )
        if result is None:
            logger.error("Alignment planning failed. Robot was not moved.")
            return

        start_config, configs = result
        exec_names = (
            [f"L_arm_j{i}" for i in range(1, 8)]
            + [f"R_arm_j{i}" for i in range(1, 8)]
            + torso_joints
        )
        mismatch = max(abs(start_config[n] - initial_joint_pos[n]) for n in exec_names)
        if mismatch > MAX_JOINT_JUMP:
            logger.error(
                f"Planned start differs from current config by {mismatch:.2f} rad. "
                "Hold the box in a reachable pose and retry."
            )
            return

        if not configs:
            logger.info("Already aligned; nothing to execute.")
            return

        logger.info("Executing horizontal alignment...")
        execute_trajectory(
            bot, torso_joints, start_config, configs, duration, control_hz
        )
        logger.info(
            "Alignment done. Next: python rotate_vertical.py "
            "(box still gripped)."
        )
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    except TimeoutError as exc:
        logger.error(f"Motion timed out: {exc}")
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
