# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""The card block ``card_block_35`` in the grasp: dimensions, the gripper finger plate, and where the block
first touches the conveyor box. Shared by ``10.approach_purple_to_red_bimanual.py`` (standoff target) and
``13.hands_push_left.py`` (flush stop).

The block is 40 x 35 x 10.5 cm and is carried upright with its 10.5 cm thickness along the box face normal
(``9.rotate_vertical.py``). The lg_gripper jaws close across that thickness, so one finger plate (2.0 cm along the
closing axis: Finger_1.stl / Finger2_1.stl, x = 82..102 mm, 2FPA-I-03-180) sits between the block and the box. With a square block
that plate touches the box first, purple (block centre) to red (tip face centre) = 7.25 cm along the normal
(``JAW_CONTACT_M``); a block yawed against the face leads with its corner, 5.25 cos(yaw) + 20 sin(yaw) cm. The
boxes themselves can never be flush where the finger sits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _sim_markers import quat_axis, quiet_quat_base

BLOCK_HALF_T = 0.0525  # half thickness (along the face normal)
BLOCK_HALF_W = 0.20    # half width (horizontal, along the box face) -- the leading corner when the block is yawed
FINGER_T = 0.020       # finger plate thickness along the closing axis (2FPA-I-03-180; 1st-gen 06.LG was 0.025)
JAW_CONTACT_M = BLOCK_HALF_T + FINGER_T  # 7.25 cm: purple-to-red along the normal at first contact, square block


def first_contact_along(yaw: float) -> tuple[float, str]:
    """Purple-to-red distance along the face normal at which the block first touches the box, and what touches:
    the finger plate between the boxes (square block) or the block's leading corner (yawed block)."""
    c, s_ = float(np.cos(yaw)), abs(float(np.sin(yaw)))
    finger = (BLOCK_HALF_T + FINGER_T) * c
    corner = BLOCK_HALF_T * c + BLOCK_HALF_W * s_
    return (corner, "corner") if corner > finger else (finger, "finger")


def block_yaw_vs_face(tip_path: Path, purple_path: Path) -> float | None:
    """Yaw of the block's thickness axis relative to the box face normal (rad, in [-pi/2, pi/2), horizontal), from the
    ``quat_base`` lines of both marker files. None if unavailable or the block is not upright (thickness axis vertical)."""
    qb, qk = quiet_quat_base(tip_path), quiet_quat_base(purple_path)
    if qb is None or qk is None:
        return None
    n_box = quat_axis(qb, 2)   # tip face normal = box local +z
    n_blk = quat_axis(qk, 2)   # block thickness axis = block local z
    n_box[2] = 0.0
    n_blk[2] = 0.0
    if np.linalg.norm(n_box) < 0.5 or np.linalg.norm(n_blk) < 0.5:
        return None
    d = np.arctan2(n_blk[1], n_blk[0]) - np.arctan2(n_box[1], n_box[0])
    d = (d + np.pi / 2) % np.pi - np.pi / 2  # the thickness axis has no sign
    return float(d)
