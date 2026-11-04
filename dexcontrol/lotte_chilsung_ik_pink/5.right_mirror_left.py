# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Right gripper: move to the mirror image of the LEFT gripper (about the chest plane), so the right
blade is thrust in exactly like the left one.

Right-hand counterpart of step 10: after ``10.left_mirror_and_push_force.py`` the LEFT blade has been
pushed into the box (it stopped on the wrist force or at its maximum travel). This script brings the
RIGHT gripper to the mirror of where the left gripper *now* is, so both blades engage the box
symmetrically. No force gate: the goal is a fixed pose. Only the right arm is commanded (see
``_pink_arm.py``); the left arm, torso and head are never touched.

Default (``--direct``, 2026-09-26): ONE straight move from where the right gripper is to the mirror of the
left gripper (position + orientation interpolated together; ``--extra-cm`` further along the approach
direction if given). It ends inside / against the box, so no measured-state correction by default
(``--advance-refine``).

``--no-direct`` keeps the older two stages (straight lines, streamed one after another):

  1. lower   -- straight down (or up) at the current x, y to the left gripper's height z_L, turning
                to the mirrored orientation on the way.
  2. advance -- one straight horizontal move to the mirror of the left gripper's x, y, orientation held.

Mirror (robot base frame, M = diag(1, -1, 1)): p_goal = M p_L; R_R = M R_L M (``_left_mirror.py``).
Step 10 placed the left blade at the mirror of the right one (``--lower-cm`` lower, then pushed it
forward), so mirroring the left back gives the right blade the same height and the same depth into
the box that the left reached. ``--flip-blade`` turns the goal 180 deg about the approach axis.

Example (sim running, ROBOT_NAME / ZENOH_CONFIG exported, nobody else streaming control/arm/right)::

    python 5.right_mirror_left.py                  # one straight move to the mirror of the left
    python 5.right_mirror_left.py --no-direct      # lower, then advance (two stages)
    python 5.right_mirror_left.py --plan-only     # connect, print the geometry, move nothing
    python 5.right_mirror_left.py --extra-cm 1    # end 1 cm deeper than the left
    python 5.right_mirror_left.py --pause         # SPACE between stages ('q' stops, pose held)
    python 5.right_mirror_left.py --dry-run       # plan only, from the URDF neutral pose
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pink
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _left_mirror import M_MIRROR, MIN_HORIZONTAL_AXIS, ORIGIN, mirror_arm_joints, mirror_rotation  # noqa: E402
from _pink_arm import TORSO_MAX_VEL, TORSO_POSTURE_COST, Stage, rot_err, run_stages  # noqa: E402
from _pink_common import EE, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402
from utils import wait_for_space  # noqa: E402

LARGE_MOVE_WARN_M = 0.25  # the right hand should already be close to its mirror goal after step 10


class RightMirrorPlan:
    """Goals shared by the stages; computed once from the synced poses (``prepare``)."""

    def __init__(self, extra_cm: float, flip_blade: bool, min_hand_gap_m: float, allow_cross: bool, direct: bool = True) -> None:
        self.extra_m = extra_cm / 100.0
        self.direct = direct
        self.flip_blade = flip_blade
        self.min_hand_gap_m = min_hand_gap_m
        self.allow_cross = allow_cross
        self.R_goal: np.ndarray | None = None  # mirrored left orientation
        self.a_h: np.ndarray | None = None  # unit horizontal approach direction (None if ill-defined)
        self.z_goal: float = 0.0  # = z of the left gripper
        self.p_goal: np.ndarray | None = None  # mirror of the left gripper origin (+ extra)

    # -- called by run_stages on the synced state, before anything moves --------------------
    def prepare(self, cfg: pink.Configuration) -> bool:
        T_L = cfg.get_transform_frame_to_world(EE["left"])
        T_R = cfg.get_transform_frame_to_world(EE["right"])
        p_L, p_R = T_L.translation.copy(), T_R.translation.copy()
        self.R_goal = mirror_rotation(T_L.rotation, self.flip_blade)
        a = self.R_goal[:, 2]  # right EE +Z at the goal (blade forward axis)
        a_h = np.array([a[0], a[1], 0.0])
        n_h = float(np.linalg.norm(a_h))
        self.a_h = a_h / n_h if n_h >= MIN_HORIZONTAL_AXIS else None
        p_mirror = M_MIRROR @ p_L
        self.z_goal = float(p_mirror[2])
        extra = np.zeros(3)
        if self.extra_m > 0:
            if self.a_h is None:
                logger.warning(f"Approach axis {np.round(a, 3)} is nearly vertical; --extra-cm is ignored.")
            else:
                extra = self.extra_m * self.a_h
        self.p_goal = p_mirror + extra

        logger.info(
            f"Left gripper (robot base):  p {np.round(p_L, 4)}, EE +Z {np.round(T_L.rotation[:, 2], 3)}, "
            f"EE +X (finger slide axis) {np.round(T_L.rotation[:, 0], 3)}"
        )
        logger.info(f"Right gripper now:          p {np.round(p_R, 4)}, EE +Z {np.round(T_R.rotation[:, 2], 3)}")
        tilt = np.rad2deg(np.arcsin(np.clip(-a[2], -1.0, 1.0)))
        logger.info(
            f"Mirror of the left gripper: p {np.round(p_mirror, 4)}; right goal orientation EE +Z {np.round(a, 3)}, "
            f"EE +X {np.round(self.R_goal[:, 0], 3)} (approach axis tilted {tilt:.1f} deg down"
            f"{', blade flipped' if self.flip_blade else ''}); rotation from now "
            f"{np.rad2deg(rot_err(T_R.rotation, self.R_goal)):.1f} deg"
        )
        d = self.p_goal - p_R
        fwd = float(d[:2] @ self.a_h[:2]) if self.a_h is not None else float("nan")
        side = float(np.linalg.norm(d[:2] - fwd * self.a_h[:2])) if self.a_h is not None else float("nan")
        where = (f"{np.round(self.p_goal, 4)} (from now: forward {fwd * 100:+.1f} cm along the approach direction, "
                 f"sideways {side * 100:.1f} cm, down {(p_R[2] - self.z_goal) * 100:+.1f} cm, |d| = {np.linalg.norm(d) * 100:.1f} cm"
                 f"{f', {self.extra_m * 100:+.1f} cm beyond the mirror' if self.extra_m > 0 else ''})")
        if self.direct:
            logger.info(f"Plan: one straight move to the mirror of the left gripper at {where}, turning to the mirrored orientation on the way.")
        else:
            logger.info(
                f"Plan: lower from z = {p_R[2]:.4f} to z = {self.z_goal:.4f} "
                f"({(p_R[2] - self.z_goal) * 100:+.1f} cm below now) turning to the mirrored orientation; then advance to {where}"
            )
        if np.linalg.norm(d) > LARGE_MOVE_WARN_M:
            logger.warning(
                f"The mirror goal is {np.linalg.norm(d) * 100:.0f} cm from the right gripper. After step 10 it should be "
                "close (a few cm forward and down); check that step 10 ran and that the hands are where you expect."
            )
        gap = float(np.linalg.norm(self.p_goal - p_L))
        if gap < self.min_hand_gap_m:
            logger.error(
                f"Goal is only {gap * 100:.1f} cm from the left gripper (< {self.min_hand_gap_m * 100:.0f} cm): "
                "the left hand is near the chest midline, the grippers would collide. Nothing commanded "
                "(--min-hand-gap-m to change the limit)."
            )
            return False
        if self.p_goal[1] > 0.0 and not self.allow_cross:
            logger.error(
                f"Goal has y = {self.p_goal[1]:.3f} > 0: the left hand is across the midline, so the arms "
                "would cross. Nothing commanded (--allow-cross to override)."
            )
            return False
        return True

    # -- IK posture bias: the right arm mirrors the left arm's joints (the left is already at the mirror pose) --
    def posture_joints(self, cfg: pink.Configuration) -> np.ndarray:
        return mirror_arm_joints(cfg, "left")

    # -- stage specs: (point in R_gripper_base, goal of that point in robot base, goal rotation) --
    def spec_lower(self, cfg: pink.Configuration):
        p = cfg.get_transform_frame_to_world(EE["right"]).translation
        return ORIGIN, np.array([p[0], p[1], self.z_goal]), self.R_goal.copy()

    def spec_advance(self, cfg: pink.Configuration):
        return ORIGIN, self.p_goal.copy(), self.R_goal.copy()


def build_stages(
    plan: RightMirrorPlan, *, lower_duration: float, advance_duration: float, step_m: float, step_rad: float,
    refine: int, advance_refine: int, refine_tol_m: float, refine_tol_rad: float,
) -> list[Stage]:
    common = dict(step_m=step_m, step_rad=step_rad, refine_tol_m=refine_tol_m, refine_tol_rad=refine_tol_rad,
                  posture_joints=plan.posture_joints)
    if plan.direct:
        # one straight move: position and orientation interpolated together, streamed over the summed time
        return [Stage("mirror", plan.spec_advance, duration=lower_duration + advance_duration, refine=advance_refine, **common)]
    return [
        Stage("lower", plan.spec_lower, duration=lower_duration, refine=refine, **common),
        Stage("advance", plan.spec_advance, duration=advance_duration, refine=advance_refine, **common),
    ]


def _pause(next_label: str) -> bool:
    logger.info(f"Next stage: {next_label}.")
    try:
        wait_for_space()
    except KeyboardInterrupt:
        return False
    return True


def _plan_only(plan: RightMirrorPlan) -> bool:
    """Connect, sync the measured state, print the geometry and leave without commanding anything."""
    configuration, _tasks, _solver, idx = build_ik()
    bot = connect()
    try:
        sync_from_robot(bot, configuration, idx)
        ok = plan.prepare(configuration)
    finally:
        disconnect_leave_held(bot)
    logger.info("Plan only: nothing commanded" + ("" if ok else " (the plan checks failed)") + ".")
    return ok


def main(
    extra_cm: float = 0.0,
    flip_blade: bool = False,
    direct: bool = True,
    lower_duration: float = 2.0,
    advance_duration: float = 4.0,
    step_m: float = 0.01,
    step_deg: float = 3.0,
    refine: int = 1,
    advance_refine: int = 0,
    refine_tol_m: float = 0.005,
    refine_tol_deg: float = 2.0,
    settle_s: float = 0.7,
    min_hand_gap_m: float = 0.12,
    allow_cross: bool = False,
    pause: bool = False,
    leave_held: bool = True,
    dry_run: bool = False,
    plan_only: bool = False,
    use_torso: bool = False,
    auto_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    max_joint_vel: float = 0.5,
    anchor_setpoints: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Move the right gripper to the mirror of the left one -- in one straight move (default) or lower-then-advance.

    Args:
        extra_cm: End this much beyond the mirrored spot along the approach direction (centimetres; 0 = exact mirror).
        flip_blade: Rotate the mirrored goal 180 deg about the approach axis (swap which finger faces the box).
        direct: One straight move (position + orientation together) from the current pose to the mirror goal
            (default). False = the older two stages: lower to the left gripper's height, then advance.
        lower_duration: Stream time of the lower stage (seconds); in direct mode it is added to advance_duration.
        advance_duration: Stream time of the advance stage (seconds).
        step_m: Cartesian spacing between IK waypoints (metres).
        step_deg: Rotation spacing between IK waypoints (degrees).
        refine: Measured-state correction moves after the lower stage (free space).
        advance_refine: Measured-state correction moves after the advance (default 0: it ends against the box).
        refine_tol_m: Position tolerance that ends refinement (metres).
        refine_tol_deg: Orientation tolerance that ends refinement (degrees).
        settle_s: Wait before measuring so the controller settles (seconds).
        min_hand_gap_m: Abort if the goal is closer than this to the left gripper (metres).
        allow_cross: Allow a goal on the left half (y > 0) when the left hand is across the midline.
        pause: Wait for SPACE between stages ('q' stops with the pose held).
        leave_held: Exit without ``stop()`` so the pose stays held.
        dry_run: Plan all stages from the URDF neutral pose without connecting; report reachability only.
        plan_only: Connect and print the geometry computed from the measured poses, but command nothing.
        use_torso: Also move the torso in the IK. WARNING: this moves the left hand pushed into the box too.
        auto_torso: Retry a stage with the torso if the arm alone covers less than half of it (same warning).
        torso_posture_cost: Posture cost on torso joints; higher = torso moves less (arms use 1e-3).
        torso_max_vel: Torso joint speed cap (rad/s).
        max_joint_vel: Arm joint speed cap (rad/s); a stage's stream time is stretched to respect it.
        anchor_setpoints: Read the sim's held targets and anchor the streams there (recommended).
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    if extra_cm < 0:
        raise ValueError("extra_cm must be >= 0")
    if use_torso or auto_torso:
        logger.warning("Torso motion also moves the LEFT hand, which is pushed into the box; use with care.")
    plan = RightMirrorPlan(extra_cm, flip_blade, min_hand_gap_m, allow_cross, direct=direct)
    if plan_only:
        return _plan_only(plan)
    stages = build_stages(
        plan, lower_duration=lower_duration, advance_duration=advance_duration, step_m=step_m,
        step_rad=np.deg2rad(step_deg), refine=refine, advance_refine=advance_refine, refine_tol_m=refine_tol_m,
        refine_tol_rad=np.deg2rad(refine_tol_deg),
    )
    logger.info("Stages: " + " -> ".join(s.label for s in stages))
    return run_stages(
        "right", stages, use_torso=use_torso, torso_posture_cost=torso_posture_cost, torso_max_vel=torso_max_vel,
        max_joint_vel=max_joint_vel, settle_s=settle_s, anchor_setpoints=anchor_setpoints, auto_torso=auto_torso,
        ignore_foreign_commands=ignore_foreign_commands, dry_run=dry_run, leave_held=leave_held,
        prepare=plan.prepare, pause=_pause if pause else None,
    )


if __name__ == "__main__":
    tyro.cli(main)
