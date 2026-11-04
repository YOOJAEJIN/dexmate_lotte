# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Push the right gripper a few cm along its own +Z (approach axis), orientation held.

Follow-up to ``6.right_lower_finger_to_cyan.py``: with the lower finger tip parked in front of
the cyan marker, slide the whole gripper along the end-effector's local +Z — the direction the
fingers point (``R_ee`` / ``R_arm_l8`` / ``R_gripper_base`` share this axis), i.e. toward the
card stack — so the blade hooks the box edge before lifting. The move is a straight line in
that direction with the gripper orientation unchanged; only the right arm (plus the torso with
``--use-torso``) is commanded and the stream is anchored at the sim's held setpoints, so there
is no relax step (see ``_pink_right.py``).

Contact with the box is expected at the end of the push, so measured-state corrections are
off by default (``--refine 1`` re-enables them). The log still reports the measured travel.

Example::

    python 7.right_push_ee_z.py                 # 3 cm along EE +Z over 2 s
    python 7.right_push_ee_z.py --cm 2 --duration 1.5
    python 7.right_push_ee_z.py --use-torso
    python 7.right_push_ee_z.py --dry-run
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pink
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pink_right import (  # noqa: E402
    TORSO_MAX_VEL,
    TORSO_POSTURE_COST,
    gripper_pose_b,
    run_point_move,
)


def push_spec(cm: float):
    """(point, goal) for ``run_point_move``: the gripper origin moved ``cm`` along its own +Z."""

    def spec(configuration: pink.Configuration) -> tuple[np.ndarray, np.ndarray]:
        T = gripper_pose_b(configuration)  # base <- R_gripper_base
        z_axis_b = T.rotation[:, 2]
        goal_b = T.translation + (cm / 100.0) * z_axis_b
        logger.info(
            f"EE +Z in robot base = {np.round(z_axis_b, 3)}; pushing {cm:.1f} cm along it "
            f"(gripper origin {np.round(T.translation, 4)} -> {np.round(goal_b, 4)})"
        )
        return np.zeros(3), goal_b

    return spec


def main(
    cm: float = 3.0,
    duration: float = 2.0,
    step_m: float = 0.005,
    leave_held: bool = True,
    dry_run: bool = False,
    use_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    refine: int = 0,
    refine_tol_m: float = 0.005,
    settle_s: float = 0.7,
    anchor_setpoints: bool = True,
    auto_torso: bool = True,
    ignore_foreign_commands: bool = False,
) -> None:
    """Slide the right gripper along its own +Z to hook the box edge.

    Args:
        cm: Push distance along the end-effector +Z axis (centimetres, > 0).
        duration: Stream time (seconds).
        step_m: Cartesian spacing between IK waypoints (metres).
        leave_held: Exit without ``stop()`` so the pose stays held.
        dry_run: Plan from the URDF neutral pose without connecting; report reachability only.
        use_torso: Also move the torso in the IK (normally unnecessary for a few cm).
        torso_posture_cost: Posture cost on torso joints; higher = torso moves less (arms use 1e-3).
        torso_max_vel: Torso joint speed cap (rad/s).
        refine: Measured-state correction moves after the push (default 0: contact is expected).
        refine_tol_m: Stop refining once the measured error is below this (metres).
        settle_s: Wait before measuring so the controller settles (seconds).
        anchor_setpoints: Read the sim's held targets and anchor the stream there (recommended).
        auto_torso: If the arm alone cannot plan the move (fully extended), retry with the torso.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    if cm <= 0:
        raise ValueError("cm must be positive (push direction is the EE +Z axis)")
    run_point_move(
        push_spec(cm), label=f"push {cm:.1f} cm along EE +Z", duration=duration, step_m=step_m,
        use_torso=use_torso, torso_posture_cost=torso_posture_cost, torso_max_vel=torso_max_vel,
        refine=refine, refine_tol_m=refine_tol_m, settle_s=settle_s, anchor_setpoints=anchor_setpoints, auto_torso=auto_torso, ignore_foreign_commands=ignore_foreign_commands,
        dry_run=dry_run, leave_held=leave_held,
    )


if __name__ == "__main__":
    tyro.cli(main)
