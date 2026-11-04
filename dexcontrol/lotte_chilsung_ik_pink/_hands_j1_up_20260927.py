# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""(Kept out of the sequence since 2026-09-27 11:2x; step 12 is now ``12.hands_back.py``.) Raise BOTH arms out of the block: a little back, a little up (orientation held), then shoulder j1 only.

Three stages, one connection (driver: ``_j1_up.py``):

  1. back -- both grippers move ``--back-cm`` (10 cm) straight back along robot -X with their orientation held
             (Pink IK, both arms + torso; the torso only moves a degree or two), so the open fingers slide out
             of the block edge instead of pivoting around it. If IK / joint limits allow less, the reachable part
             is done and the sequence goes on (a stage with no room at all is skipped with a warning).
  2. up   -- both grippers move ``--lift-cm`` (10 cm) straight up (robot +Z), orientation held.
  3. j1   -- pure joint-space move, no IK. Each arm's j1 goes from its current value 45 deg in that arm's
raise direction (left: j1 decreases, right: j1 increases; ``--deg`` changes the angle, ``--target-rad`` sets the same
absolute goal for both, e.g. 0 = vertical from home, ``--reverse`` goes the other way). The other six joints of each
arm keep their measured values, so the hands keep their elbow / wrist posture and simply swing up over the shoulders.
In every stage both arms are streamed together over one min-jerk profile (``_pink_arm.stream``, leash-paced):
stages 1-2 last at least ``--pre-duration`` (2 s each), stage 3 at least ``--duration`` (4.5 s, i.e. 10 deg/s for the
default 45 deg), longer if an arm joint would exceed ``--arm-vel`` (0.3 rad/s) or the torso ``--torso-vel``. Head and
grippers are never touched. The torso is freed for stages 1-2 by default (``--no-use-torso`` = arms only): from the
block-placing pose the right arm sits on its ``R_arm_j2`` / ``j4`` limits and cannot retract at all on its own
(2026-09-27 plan check: back 5 cm -> no progress; with the torso 5 cm for both arms, torso change ~2 deg). The
streams are anchored at the sim's held joint targets so the torso is never dropped to its sagged measured angle. Swinging j1 straight away moved the hands on an arc about the shoulder and tilted the
grippers, which clipped the block (2026-09-27); the two straight moves first take the fingers clear of it. ``--side left`` /
``--side right`` moves one arm only (``14.left_j1_up.py`` is the left-only shell).

From the block-holding pose (arms reaching forward) 45 deg of j1 lifts the hands about half a metre and pulls them
back a little; 90 deg (``--deg 90``) swings them up over and behind the shoulders (2026-09-27 FK preview: up 94 / 81 cm,
back 62 / 77 cm for left / right). Open the grippers first (``11.open_hands.py``) -- a hand still closed on the block drags it along.

Example::

    python 12.hands_j1_up.py --plan-only        # plan the three stages from the measured state, move nothing
    python 12.hands_j1_up.py                    # back 10 cm, up 10 cm, then j1 +/-45 deg (>= 4.5 s, 0.3 rad/s cap)
    python 12.hands_j1_up.py --back-cm 8 --lift-cm 3     # different clearance moves
    python 12.hands_j1_up.py --back-cm 0 --lift-cm 0     # j1 only (old behaviour)
    python 12.hands_j1_up.py --no-use-torso     # arms only in stages 1-2 (the right arm may not retract)
    python 12.hands_j1_up.py --duration 3       # a little faster
    python 12.hands_j1_up.py --deg 90           # raise both by 90 deg (hands end up behind the head)
    python 12.hands_j1_up.py --target-rad 0.0   # absolute target: both j1 = 0 rad (vertical from home)
    python 12.hands_j1_up.py --side right       # right arm only
    python 12.hands_j1_up.py --reverse          # the other way (arms already past vertical)
"""

from __future__ import annotations

import os
import sys

import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _j1_up import raise_j1  # noqa: E402
from _pink_arm import TORSO_MAX_VEL  # noqa: E402

ARM_VEL = 0.3  # rad/s cap for the shared profile (the single-arm scripts use 0.5)
DURATION_S = 4.5  # minimum stream time: 45 deg over 4.5 s = 10 deg/s (was 3 s; slowed on request 2026-09-27)


def main(
    side: str = "both",
    back_cm: float = 10.0,
    lift_cm: float = 10.0,
    pre_duration: float = 2.0,
    step_m: float = 0.01,
    use_torso: bool = True,
    torso_vel: float = TORSO_MAX_VEL,
    anchor_setpoints: bool = True,
    deg: float = 45.0,
    target_rad: float | None = None,
    reverse: bool = False,
    arm_vel: float = ARM_VEL,
    duration: float = DURATION_S,
    tol_deg: float = 1.0,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Move both grippers back, then up (orientation held), then turn ``L_arm_j1`` / ``R_arm_j1`` to raise the arms.

    Args:
        side: "both" (default), "left" or "right".
        back_cm: Stage 1: straight move along robot -X with orientation held (centimetres, default 10; 0 = skip). Only
            the reachable part is done if the arms run out of room.
        lift_cm: Stage 2: straight move along robot +Z with orientation held (centimetres, default 10; 0 = skip).
        pre_duration: Minimum stream time of each of stages 1-2 (seconds).
        step_m: Cartesian spacing of the IK waypoints in stages 1-2 (metres).
        use_torso: Free the torso in stages 1-2 as well (default; it is streamed with the arms, a degree or two).
        torso_vel: Torso joint speed cap in stages 1-2 (rad/s).
        anchor_setpoints: Stream relative to the sim's held joint targets (recommended; avoids the torso sag drop).
        deg: How far to turn each j1 (degrees, > 0, default 45), applied in each arm's raise direction.
        target_rad: Absolute goal for j1 (radians, same for both arms); overrides ``deg`` / ``reverse``.
        reverse: Turn the other way -- for arms already past vertical.
        arm_vel: Joint speed cap (rad/s, default 0.3); the shared stream time is stretched to respect it.
        duration: Minimum stream time (seconds, default 4.5); this sets the pace for the default 45 deg.
        tol_deg: Residual below this counts as reached (degrees).
        plan_only: Connect, print current vs target, move nothing.
        leave_held: Exit without ``stop()`` so the pose stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming arm targets.
    """
    if side not in ("both", "left", "right"):
        raise ValueError(f"--side must be 'both', 'left' or 'right', got {side!r}")
    sides = ("left", "right") if side == "both" else side
    pre_moves = []
    if back_cm > 0:
        pre_moves.append(("back", (-back_cm / 100.0, 0.0, 0.0)))
    if lift_cm > 0:
        pre_moves.append(("up", (0.0, 0.0, lift_cm / 100.0)))
    return raise_j1(
        sides, pre_moves=pre_moves, pre_duration=pre_duration, step_m=step_m, use_torso=use_torso, torso_max_vel=torso_vel,
        anchor_setpoints=anchor_setpoints, deg=deg, target_rad=target_rad, reverse=reverse, arm_vel=arm_vel, duration=duration,
        tol_deg=tol_deg, plan_only=plan_only, leave_held=leave_held, ignore_foreign_commands=ignore_foreign_commands,
    )


if __name__ == "__main__":
    tyro.cli(main)
