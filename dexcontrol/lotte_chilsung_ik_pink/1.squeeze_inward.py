# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Squeeze both hands a little toward their midpoint (Pink IK version).

Port of ``lotte_chilsung_2/1.squeeze_inward.py`` onto ``ik_pink.py``: both
grippers move straight toward their midpoint with orientation fixed; torso
and head are never commanded. Shared plumbing lives in ``_pink_common.py``.

Example::

    python 1.squeeze_inward.py
    python 1.squeeze_inward.py --cm 2 --left-extra-cm 1
    python 1.squeeze_inward.py --cm 3 --duration 2
    python 1.squeeze_inward.py --no-leave-held   # release (stop) on exit
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pink_common import Targets, ee_poses, run_cartesian_move  # noqa: E402


def squeeze_targets(configuration, move_m: float, left_extra_m: float = 0.0) -> Targets | None:
    """Target gripper positions moved toward the L/R midpoint (arm_center frame)."""
    p = {s: T.translation.copy() for s, T in ee_poses(configuration).items()}
    mid = 0.5 * (p["left"] + p["right"])
    to_mid = {s: mid - p[s] for s in p}
    dist = {s: float(np.linalg.norm(v)) for s, v in to_mid.items()}
    if min(dist.values()) < 1e-4:
        logger.error("Hands nearly coincident; cannot squeeze.")
        return None

    # Cap so hands never cross the mid-plane.
    move = {
        "left": min(move_m + max(left_extra_m, 0.0), 0.9 * dist["left"]),
        "right": min(move_m, 0.9 * dist["right"]),
    }
    tgt = {s: p[s] + move[s] * to_mid[s] / dist[s] for s in p}
    logger.info(
        f"Squeeze inward L={move['left'] * 100:.1f} cm R={move['right'] * 100:.1f} cm "
        f"(gap {np.linalg.norm(p['left'] - p['right']):.3f} → "
        f"{np.linalg.norm(tgt['left'] - tgt['right']):.3f} m, arm_center frame)"
    )
    return tgt


def main(
    cm: float = 2.0,
    left_extra_cm: float = 1.0,
    duration: float = 1.5,
    step_m: float = 0.01,
    leave_held: bool = True,
) -> None:
    """Close both grippers toward their midpoint.

    Args:
        cm: How far **each** hand moves toward the mid-point (centimeters).
        left_extra_cm: Extra inward travel for the **left** hand only (cm).
            Default 1 so ``--cm 2`` moves left ~3 cm and right 2 cm.
        duration: Stream time (seconds).
        step_m: Cartesian spacing between IK waypoints (meters).
        leave_held: Exit without ``stop()`` so the pose stays held.
    """
    if cm <= 0 and left_extra_cm <= 0:
        logger.info(f"cm={cm}, left_extra_cm={left_extra_cm} (<=0); nothing to do.")
        return
    run_cartesian_move(
        lambda cfg: squeeze_targets(cfg, cm / 100.0, left_extra_cm / 100.0),
        duration=duration, step_m=step_m, leave_held=leave_held, label="squeeze",
    )


if __name__ == "__main__":
    tyro.cli(main)
