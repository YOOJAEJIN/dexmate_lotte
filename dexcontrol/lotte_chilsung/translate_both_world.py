# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Translate both end-effectors by a small distance along a **world** axis,
keeping each EE orientation fixed (default: world −Y).

Both hands move by the same vector, so a box held between them is carried
rigidly. Works equally with open hands. For a different vector per hand see
``release_retract.py``.

The base→world transform is derived from the sim marker files (which contain
robot-base and world coordinates of the same points); use ``--world-yaw-deg``
to override.

Example::

    python translate_both_world.py --dry-run          # plan only
    python translate_both_world.py                    # 3 cm along world −Y
    python translate_both_world.py --cm 5
    python translate_both_world.py --cm 4 --dir 0 0 1  # 4 cm up (world +Z)
    python translate_both_world.py --cm 5 --lock-torso
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _world_translate import load_world_frame, run_translation, unit  # noqa: E402


def main(
    cm: float = 3.0,
    dir: tuple[float, float, float] = (0.0, -1.0, 0.0),
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
    """Move both EEs by ``cm`` along world ``dir`` with orientations fixed.

    Args:
        cm: Distance to move (centimeters). Negative reverses ``dir``.
        dir: Direction in the **world** frame (normalized). Default −Y.
        tip_file: Red tip file (base + world lines). Default: sim file.
        purple_file: Purple center file (base + world lines). Default: sim file.
        world_yaw_deg: Override base→world yaw (deg) instead of deriving it
            from the two marker files.
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
        leave_held: Exit without ``stop()`` so the last targets stay held.
    """
    if abs(cm) < 1e-6:
        logger.info("cm ≈ 0; nothing to do.")
        return
    delta = (float(cm) / 100.0) * unit(dir)
    logger.info(f"Translate both EEs {abs(cm):.1f} cm along world {delta / max(abs(cm) / 100.0, 1e-9)}")
    wf = load_world_frame(tip_file, purple_file, world_yaw_deg)
    run_translation(
        wf=wf,
        l_delta_w=delta,
        r_delta_w=delta,
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
