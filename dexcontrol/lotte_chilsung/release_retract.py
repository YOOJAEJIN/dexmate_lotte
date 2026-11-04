# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Release step after placing the vertically-held box: back both arms off.

Both hands move at the same time, each along its own **world** direction with
its orientation fixed:

  - left  : world +Z (up, toward the head)      — clears the box top edge
  - right : world −Y (back, toward the chest)   — clears the box bottom edge

Run this right after opening both hands (omniteleop). Distances are small by
default; adjust with ``--left-cm`` / ``--right-cm``.

Example::

    python release_retract.py --dry-run
    python release_retract.py                      # L +3 cm up, R −3 cm world Y
    python release_retract.py --left-cm 5 --right-cm 4
    python release_retract.py --right-dir 0 -1 0.3 # right also slightly up
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _world_translate import load_world_frame, run_translation, unit  # noqa: E402


def main(
    left_cm: float = 3.0,
    right_cm: float = 3.0,
    left_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
    right_dir: tuple[float, float, float] = (0.0, -1.0, 0.0),
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    world_yaw_deg: float | None = None,
    step_m: float = 0.005,
    duration: float = 2.0,
    control_hz: float = 100.0,
    torso_damping: float = 5.0,
    lock_torso: bool = False,
    collision_margin: float = 0.015,
    ignore_self_collision: bool = False,
    dry_run: bool = False,
    leave_held: bool = True,
) -> None:
    """Move left EE up (world +Z) and right EE back (world −Y), orientations fixed.

    Args:
        left_cm: Left hand travel (cm) along ``left_dir``. 0 keeps it still.
        right_cm: Right hand travel (cm) along ``right_dir``. 0 keeps it still.
        left_dir: Left hand direction in the world frame. Default +Z.
        right_dir: Right hand direction in the world frame. Default −Y.
        tip_file: Red tip file (base + world lines). Default: sim file.
        purple_file: Purple center file (base + world lines). Default: sim file.
        world_yaw_deg: Override base→world yaw (deg) instead of deriving it.
        step_m: IK waypoint spacing (m).
        duration: Stream time (s).
        control_hz: Command rate.
        torso_damping: Torso IK damping when unlocked (higher = less torso).
        lock_torso: Freeze torso (arms only).
        collision_margin: Pink self-collision barrier (m); falls back to
            smaller values if the current pose is already closer than this.
        ignore_self_collision: If even the smallest margin fails (arms already
            touching), plan without the self-collision barrier. Use only for
            small moves away from the other arm.
        dry_run: Plan and report, do not move.
        leave_held: Exit without ``stop()``.
    """
    l_delta = (float(left_cm) / 100.0) * unit(left_dir) if abs(left_cm) > 1e-6 else np.zeros(3)
    r_delta = (float(right_cm) / 100.0) * unit(right_dir) if abs(right_cm) > 1e-6 else np.zeros(3)
    logger.info(
        f"Release retract: left {left_cm:.1f} cm along world {np.round(unit(left_dir), 3)}, "
        f"right {right_cm:.1f} cm along world {np.round(unit(right_dir), 3)}"
    )
    logger.warning("Both hands should already be OPEN (omniteleop); grippers are not commanded.")
    wf = load_world_frame(tip_file, purple_file, world_yaw_deg)
    run_translation(
        wf=wf,
        l_delta_w=l_delta,
        r_delta_w=r_delta,
        step_m=step_m,
        duration=duration,
        control_hz=control_hz,
        torso_damping=torso_damping,
        lock_torso=lock_torso,
        collision_margin=collision_margin,
        ignore_self_collision=ignore_self_collision,
        dry_run=dry_run,
        leave_held=leave_held,
    )


if __name__ == "__main__":
    tyro.cli(main)
