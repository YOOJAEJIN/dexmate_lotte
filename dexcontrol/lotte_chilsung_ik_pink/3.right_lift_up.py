# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Lift the right gripper straight up by a few cm, orientation held.

Follow-up to ``7.right_push_ee_z.py``: with the blade hooked on the box edge, raise the whole
gripper along robot-base +Z (= world up, the base is level) by ``--cm`` (default 5) so the box
lifts off the stack. The move is a straight vertical line with the gripper orientation
unchanged; only the right arm (plus the torso with ``--use-torso``, or automatically when the
arm alone cannot) is commanded, streams are anchored at the sim's held setpoints, and the left
arm is never commanded (see ``_pink_right.py``).

After the lift the measured height is checked once by default (``--refine 1``): under the box
load the arm sags a little more, and a short correction from the measured state restores the
requested height. Use ``--refine 0`` to skip it.

Example::

    python 8.right_lift_up.py                 # 5 cm up over 2.5 s
    python 8.right_lift_up.py --cm 8 --duration 4
    python 8.right_lift_up.py --use-torso
    python 8.right_lift_up.py --dry-run
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

UP_B = np.array([0.0, 0.0, 1.0])  # robot base +Z; the base is level, so this is world up


def lift_spec(cm: float):
    """(point, goal) for ``run_point_move``: the gripper origin raised ``cm`` along base +Z."""

    def spec(configuration: pink.Configuration) -> tuple[np.ndarray, np.ndarray]:
        T = gripper_pose_b(configuration)  # base <- R_gripper_base
        goal_b = T.translation + (cm / 100.0) * UP_B
        logger.info(f"Lifting {cm:.1f} cm along base +Z (gripper origin {np.round(T.translation, 4)} -> {np.round(goal_b, 4)})")
        return np.zeros(3), goal_b

    return spec


def main(
    cm: float = 5.0,
    duration: float = 2.5,
    step_m: float = 0.005,
    leave_held: bool = True,
    dry_run: bool = False,
    use_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    refine: int = 1,
    refine_tol_m: float = 0.005,
    settle_s: float = 0.7,
    anchor_setpoints: bool = True,
    auto_torso: bool = True,
    ignore_foreign_commands: bool = False,
) -> None:
    """Raise the right gripper straight up with its orientation held.

    Args:
        cm: Lift height along robot-base +Z (centimetres, > 0).
        duration: Stream time (seconds).
        step_m: Cartesian spacing between IK waypoints (metres).
        leave_held: Exit without ``stop()`` so the pose stays held.
        dry_run: Plan from the URDF neutral pose without connecting; report reachability only.
        use_torso: Also move the torso in the IK from the start.
        torso_posture_cost: Posture cost on torso joints; higher = torso moves less (arms use 1e-3).
        torso_max_vel: Torso joint speed cap (rad/s).
        refine: Measured-state correction moves after the lift (default 1: compensates load sag).
        refine_tol_m: Stop refining once the measured error is below this (metres).
        settle_s: Wait before measuring so the controller settles (seconds).
        anchor_setpoints: Read the sim's held targets and anchor the stream there (recommended).
        auto_torso: If the arm alone cannot plan the lift (fully extended), retry with the torso.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    if cm <= 0:
        raise ValueError("cm must be positive (lift direction is base +Z)")
    run_point_move(
        lift_spec(cm), label=f"lift {cm:.1f} cm", duration=duration, step_m=step_m,
        use_torso=use_torso, torso_posture_cost=torso_posture_cost, torso_max_vel=torso_max_vel,
        refine=refine, refine_tol_m=refine_tol_m, settle_s=settle_s, anchor_setpoints=anchor_setpoints,
        auto_torso=auto_torso, ignore_foreign_commands=ignore_foreign_commands, dry_run=dry_run, leave_held=leave_held,
    )


if __name__ == "__main__":
    tyro.cli(main)
