# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Mirror-image goal for the left gripper, shared by ``9.left_mirror_and_push.py`` and
``10.left_mirror_and_push_force.py``.

Mirror (robot base frame, M = diag(1, -1, 1)): x, y of the goal = M p_R; R_L = M R_R M. The URDF's
``L/R_gripper_base`` frames are exact mirror images (checked at the neutral pose) and the sim's LG
vertical grippers carry identical finger layouts in their wrist frames (long blade ``Finger_1`` on
local -X on both sides, read from the USD), so this frame mirror also mirrors the blades physically.
``flip_blade`` turns the goal 180 deg about the approach axis if the other finger should face the box.

``MirrorPlan.prepare`` (run by ``_pink_arm.run_stages`` on the synced state) computes:

* ``R_goal`` -- mirrored gripper orientation; ``a_h`` -- unit horizontal approach direction (EE +Z
  of the goal projected on the ground plane);
* ``z_lift`` -- lift / final height, ``lower_cm`` below the right gripper;
* ``p_mirror`` -- mirror of the right gripper's x, y at ``z_lift``;
* ``p_final`` -- ``p_mirror`` moved ``forward_cm`` along ``a_h`` (negative = short of the mirror).
"""

from __future__ import annotations

import numpy as np
import pink
import pinocchio as pin
from loguru import logger

import qpsolvers

import config
from _pink_arm import build_arm_ik, plan_se3_path, rot_err
from _pink_common import EE

M_MIRROR = np.diag([1.0, -1.0, 1.0])  # reflection through the robot's sagittal (base XZ) plane
# Joint-space mirror between the arms: q_left = MIRROR_JOINT_SIGNS * q_right (and vice versa). Found by brute
# force over the 2^7 sign patterns on the live pose 2026-09-26 (FK of the mirrored joints lands on the mirrored
# gripper pose to 0.2 mm): only the elbow pitch (j4) keeps its sign.
MIRROR_JOINT_SIGNS = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
ORIGIN = np.zeros(3)  # the gripper-frame point we place: L_gripper_base origin
MIN_HORIZONTAL_AXIS = 0.3  # |horizontal part of EE +Z| below this: "forward" is ill-defined


def mirror_arm_joints(cfg, from_side: str) -> np.ndarray:
    """The 7 joint values of the *other* arm that mirror ``from_side``'s current joints (robot-frame FK lands on
    the mirrored gripper pose). Used as the IK posture target so a mirror move follows the same elbow branch."""
    from _pink_common import ARM_JOINTS  # local import: _pink_common imports nothing from here

    model = cfg.model
    q = np.array([cfg.q[model.idx_qs[model.getJointId(n)]] for n in ARM_JOINTS[from_side]], dtype=float)
    return MIRROR_JOINT_SIGNS * q


def mirror_rotation(R_right: np.ndarray, flip_blade: bool = False) -> np.ndarray:
    """Mirror of a gripper orientation through the chest plane (proper rotation: det(M)^2 = 1)."""
    R = M_MIRROR @ R_right @ M_MIRROR
    if flip_blade:
        R = R @ pin.exp3(np.array([0.0, 0.0, np.pi]))  # 180 deg about the approach axis
    return R


class MirrorPlan:
    """Goals shared by the stages; computed once from the synced poses (``prepare``)."""

    def __init__(
        self, lower_cm: float, forward_cm: float, flip_blade: bool, min_hand_gap_m: float, allow_cross: bool,
        recover_z: float = 0.55,
    ) -> None:
        self.lower_m = lower_cm / 100.0
        self.forward_m = forward_cm / 100.0
        self.recover_z = recover_z  # height the hand goes back down to when the lift cannot be planned from where it is
        self.flip_blade = flip_blade
        self.min_hand_gap_m = min_hand_gap_m
        self.allow_cross = allow_cross
        self.R_goal: np.ndarray | None = None
        self.a_h: np.ndarray | None = None  # unit horizontal approach direction (None if ill-defined)
        self.z_lift: float = 0.0
        self.p_mirror: np.ndarray | None = None  # mirror of the right gripper's x, y at z_lift
        self.p_final: np.ndarray | None = None  # end of the align stage

    # -- called by run_stages on the synced state, before anything moves --------------------
    def prepare(self, cfg: pink.Configuration) -> bool:
        T_R = cfg.get_transform_frame_to_world(EE["right"])
        T_L = cfg.get_transform_frame_to_world(EE["left"])
        p_R, p_now = T_R.translation.copy(), T_L.translation.copy()
        self.R_goal = mirror_rotation(T_R.rotation, self.flip_blade)
        a = self.R_goal[:, 2]  # mirrored approach axis (left EE +Z at the goal)
        a_h = np.array([a[0], a[1], 0.0])
        n_h = float(np.linalg.norm(a_h))
        self.a_h = a_h / n_h if n_h >= MIN_HORIZONTAL_AXIS else None
        logger.info(
            f"Right gripper (robot base): p {np.round(p_R, 4)}, EE +Z {np.round(T_R.rotation[:, 2], 3)}, "
            f"EE +X (finger slide axis) {np.round(T_R.rotation[:, 0], 3)}"
        )
        logger.info(f"Left gripper now:           p {np.round(p_now, 4)}, EE +Z {np.round(T_L.rotation[:, 2], 3)}")
        self.z_lift = float(p_R[2]) - self.lower_m
        p_mirror_xy = M_MIRROR @ p_R
        self.p_mirror = np.array([p_mirror_xy[0], p_mirror_xy[1], self.z_lift])
        extra = np.zeros(3)
        if abs(self.forward_m) > 0:
            if self.a_h is None:
                logger.warning(f"Approach axis {np.round(a, 3)} is nearly vertical; the forward offset is ignored.")
            else:
                extra = self.forward_m * self.a_h
        self.p_final = self.p_mirror + extra
        tilt = np.rad2deg(np.arcsin(np.clip(-a[2], -1.0, 1.0)))
        logger.info(
            f"Mirror of the right gripper: x, y = {np.round(p_mirror_xy[:2], 4)}; left goal orientation EE +Z {np.round(a, 3)}, "
            f"EE +X {np.round(self.R_goal[:, 0], 3)} (approach axis tilted {tilt:.1f} deg down"
            f"{', blade flipped' if self.flip_blade else ''}); rotation from now {np.rad2deg(rot_err(T_L.rotation, self.R_goal)):.1f} deg"
        )
        d_xy = self.p_final[:2] - p_now[:2]
        fwd = float(d_xy @ self.a_h[:2]) if self.a_h is not None else float("nan")
        logger.info(
            f"Plan: lift to z = {self.z_lift:.4f} ({self.lower_m * 100:.1f} cm below the right gripper at z = {p_R[2]:.4f}); "
            f"then one straight move to {np.round(self.p_final, 4)} (forward {fwd * 100:+.1f} cm along the approach direction, "
            f"|dx, dy| = {np.linalg.norm(d_xy) * 100:.1f} cm"
            f"{f', {self.forward_m * 100:+.1f} cm from the mirror spot' if abs(self.forward_m) > 0 else ''}) "
            f"while turning to the mirrored orientation"
        )
        if self.z_lift >= p_R[2]:
            logger.warning("lower_cm <= 0: the hand ends at or above the right gripper.")
        gap = float(np.linalg.norm(self.p_final - p_R))
        if gap < self.min_hand_gap_m:
            logger.error(
                f"Goal is only {gap * 100:.1f} cm from the right gripper (< {self.min_hand_gap_m * 100:.0f} cm): "
                "the right hand is near the chest midline, the grippers would collide. Nothing commanded "
                "(--min-hand-gap-m to change the limit)."
            )
            return False
        if self.p_final[1] < 0.0 and not self.allow_cross:
            logger.error(
                f"Goal has y = {self.p_final[1]:.3f} < 0: the right hand is across the midline, so the arms "
                "would cross. Nothing commanded (--allow-cross to override)."
            )
            return False
        return True

    # -- IK posture bias for every stage: the left arm mirrors the right arm's joints (reachable by construction) --
    def posture_joints(self, cfg: pink.Configuration) -> np.ndarray:
        return mirror_arm_joints(cfg, "right")

    def lift_coverage(self, cfg: pink.Configuration) -> float:
        """Fraction of the lift (straight up to z_lift, orientation held, posture biased) that can be planned from
        ``cfg`` without a joint jump -- a cheap reachability probe."""
        solver = config.PREFERRED_QP_SOLVER if config.PREFERRED_QP_SOLVER in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
        rcfg, tasks, idx_r = build_arm_ik(cfg, "left", False, 0.02)
        pt = rcfg.q.copy()
        pt[idx_r["left"]] = self.posture_joints(cfg)
        T = rcfg.get_transform_frame_to_world(EE["left"])
        goal_p = np.array([T.translation[0], T.translation[1], self.z_lift])
        planned = plan_se3_path(rcfg, tasks, solver, pin.SE3(T.rotation.copy(), goal_p), 0.01, np.deg2rad(3.0), posture_target=pt)
        if planned is None:
            return 0.0
        _q0, knots = planned
        if not knots:
            return 1.0
        rcfg.update(knots[-1])
        p_end = rcfg.get_transform_frame_to_world(EE["left"]).translation
        total = float(np.linalg.norm(goal_p - T.translation))
        return 1.0 if total < 1e-6 else float(np.clip(1.0 - np.linalg.norm(goal_p - p_end) / total, 0.0, 1.0))

    def spec_descend(self, cfg: pink.Configuration):
        """Recovery stage (runs first): if the lift cannot be planned from the current arm pose -- e.g. after an
        earlier run left the elbow folded with the wrist at its limit -- go straight down to ``recover_z``, from
        where the biased lift is known to work; otherwise a zero move (skipped)."""
        p = cfg.get_transform_frame_to_world(EE["left"]).translation.copy()
        cov = self.lift_coverage(cfg)
        if cov >= 0.95:
            logger.info(f"Lift is plannable from here ({cov * 100:.0f}%); no recovery move.")
            return ORIGIN, p, None
        logger.warning(f"Lift covers only {cov * 100:.0f}% from the current arm pose; descending to z = {self.recover_z:.2f} first "
                       "so the arm leaves the folded configuration and re-lifts on the mirrored branch.")
        return ORIGIN, np.array([p[0], p[1], self.recover_z]), None

    # -- stage specs: (point in L_gripper_base, goal of that point in robot base, goal rotation | None) --
    def spec_lift(self, cfg: pink.Configuration):
        p = cfg.get_transform_frame_to_world(EE["left"]).translation
        return ORIGIN, np.array([p[0], p[1], self.z_lift]), None

    def spec_align(self, cfg: pink.Configuration):
        return ORIGIN, self.p_final.copy(), self.R_goal.copy()
