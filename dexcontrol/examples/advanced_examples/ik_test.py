# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Simplest bimanual box motion: pitch a held box up to vertical while
pulling it toward the chest.

Assumes the box is gripped from both sides (one hand per face) with its
long axis pointing forward, horizontally. The box is rotated about the
axis connecting the two hands, so the forward-pointing end tips up (or
down with --tilt-direction down) until the long axis is vertical. The hand
positions only translate with the pull; the wrists do the rotation.

The rigid-grasp constraint is enforced along the whole path: the relative
pose between the two hands never changes, so the box is not twisted or
dropped. If fully vertical is unreachable, the largest reachable tilt is
used instead.

Flow with --setup-pose (default):
1. Plan the rotation backward from a global-IK solution of the final pose,
   which places the whole trajectory on an arm branch with slack.
2. Move the robot to the trajectory's start configuration (horizontal grasp).
3. Wait until the box is gripped (press Enter).
4. Stream the rotation.

With --no-setup-pose (box already held), the rotation is instead tracked
forward from the robot's current configuration, so it starts exactly on the
actual arm branch.

The trajectory is fully planned and validated before the robot moves, and
the grippers are never commanded. For the full rotate/carry/insert-into-
magazine version, see box_to_magazine.py.

Requires a running robot or simulator reachable via Zenoh (e.g. Isaac Sim
with ROBOT_NAME set).
"""

import time

import numpy as np
import pinocchio as pin
import tyro
from dexcomm import RateLimiter
from dexmotion.configs.ik import IKDampingWeightsConfig, LocalPinkIKConfig
from dexmotion.motion_manager import MotionManager
from dexmotion.utils import robot_utils
from loguru import logger
from scipy.spatial.transform import Rotation as R

from dexcontrol.robot import Robot

LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]

POS_TOL = 0.005  # m, per-hand tracking target
ROT_TOL = 2.0  # deg
# The local QP solver can plateau just above the strict tolerance near joint
# limits (measured ~9-13 mm). Accept such plateaus: a small steady offset is
# tolerable for a firmly gripped box, while a global re-seed can jump to a
# distant joint-space branch (dangerous to execute while holding it).
SOFT_POS_TOL = 0.015  # m
SOFT_ROT_TOL = 3.0  # deg
# Max per-joint change between adjacent trajectory configs.
MAX_JOINT_JUMP = 0.35  # rad


def rot_err_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices in degrees."""
    val = (np.trace(rot_a @ rot_b.T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(val, -1.0, 1.0))))


class BoxRotationPlanner:
    """Plans the rigid-grasp rotation as a list of joint configurations."""

    def __init__(self, mm: MotionManager, tilt_direction: str = "up"):
        self.mm = mm
        poses = mm.fk(["L_ee", "R_ee"])
        # fk may return live references into the model data: deep-copy.
        self.left0 = pin.SE3(poses["L_ee"].homogeneous.copy())
        self.right0 = pin.SE3(poses["R_ee"].homogeneous.copy())
        self.center0 = 0.5 * (self.left0.translation + self.right0.translation)
        box0 = pin.SE3(np.eye(3), self.center0.copy())
        # Constant grasp transforms (box frame -> each EE). Keeping these
        # fixed along the whole trajectory is what keeps the box gripped.
        self.grasp_left = box0.inverse() * self.left0
        self.grasp_right = box0.inverse() * self.right0

        z = np.array([0.0, 0.0, 1.0])
        # Grasp axis: the line connecting the two hands (through the box).
        d = self.left0.translation - self.right0.translation
        grasp_axis = d / np.linalg.norm(d)
        # Box long axis: assumed horizontal, perpendicular to the grasp
        # axis, pointing forward (away from the robot).
        forward = np.cross(grasp_axis, z)
        forward /= np.linalg.norm(forward)
        if forward[0] < 0:
            forward = -forward
        # Rotate the forward axis to +z ("up": far end of the box rises) or
        # -z ("down"). The rotation axis is +-grasp_axis, so the hand
        # positions are invariant: only the wrists rotate (plus the pull).
        target = z if tilt_direction == "up" else -z
        rot_axis = np.cross(forward, target)
        self.rot_axis = rot_axis / np.linalg.norm(rot_axis)
        self.full_angle = float(np.arccos(np.clip(forward @ target, -1.0, 1.0)))

        self.initial_q = dict(mm.get_joint_pos_dict())

    def box_pose(self, angle: float, center: np.ndarray) -> pin.SE3:
        rot = R.from_rotvec(angle * self.rot_axis).as_matrix()
        return pin.SE3(rot, np.asarray(center, dtype=float))

    def ee_targets(self, box: pin.SE3) -> tuple[pin.SE3, pin.SE3]:
        return box * self.grasp_left, box * self.grasp_right

    def _reanchor_posture_task(self) -> None:
        """Re-anchor the posture task to the current config so it acts as a
        local regularizer instead of pulling back toward the start pose."""
        try:
            solver = self.mm.local_ik_solver
            posture = solver.task_dict.get("posture")
            if posture is not None:
                posture.set_target_from_configuration(solver.configuration)
        except AttributeError:
            pass  # older dexmotion: keep the fixed posture target

    def _tracking_error(self, tl: pin.SE3, tr: pin.SE3) -> tuple[float, float]:
        cur = self.mm.fk(["L_ee", "R_ee"])
        perr = max(
            float(np.linalg.norm(cur["L_ee"].translation - tl.translation)),
            float(np.linalg.norm(cur["R_ee"].translation - tr.translation)),
        )
        rerr = max(
            rot_err_deg(cur["L_ee"].rotation, tl.rotation),
            rot_err_deg(cur["R_ee"].rotation, tr.rotation),
        )
        return perr, rerr

    def _global_ik_raw(self, tl: pin.SE3, tr: pin.SE3) -> dict | None:
        """Global IK solve. May report success with a best-effort solution
        that is centimeters off, so callers must verify/refine the result."""
        quat_l = np.roll(R.from_matrix(tl.rotation).as_quat(), 1).tolist()
        quat_r = np.roll(R.from_matrix(tr.rotation).as_quat(), 1).tolist()
        try:
            sol, coll, lim = self.mm.global_ik(
                target_pos={
                    "L_ee": tl.translation.tolist(),
                    "R_ee": tr.translation.tolist(),
                },
                target_rot={"L_ee": quat_l, "R_ee": quat_r},
                type="hybrid",
            )
        except RuntimeError:
            return None
        if sol is None or coll or not lim:
            return None
        return sol

    def _refine(self, tl: pin.SE3, tr: pin.SE3, reps: int = 5) -> tuple[float, float]:
        """Polish the current configuration toward the targets with local IK."""
        perr, rerr = self._tracking_error(tl, tr)
        for _ in range(reps):
            if perr < POS_TOL and rerr < ROT_TOL:
                break
            sol, coll, lim = self.mm.ik(
                target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
                target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
                max_iterations=200,
            )
            if sol is None or coll or not lim:
                break
            self.mm.set_joint_pos(sol)
            perr, rerr = self._tracking_error(tl, tr)
        return perr, rerr

    def pose_feasible(self, box: pin.SE3) -> bool:
        """Global solve + local polish + FK check; model state is restored."""
        tl, tr = self.ee_targets(box)
        sol = self._global_ik_raw(tl, tr)
        if sol is None:
            return False
        q_prev = dict(self.mm.get_joint_pos_dict())
        self.mm.set_joint_pos(robot_utils.get_qpos_from_joint_dict(self.mm.pin_robot, sol))
        perr, rerr = self._refine(tl, tr)
        self.mm.set_joint_pos(q_prev)
        return perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL

    def solve_waypoint(self, box: pin.SE3, max_reps: int = 15) -> bool:
        """Drive the internal model to the waypoint with iterated local IK;
        accept small plateaus; re-seed globally only if continuous."""
        tl, tr = self.ee_targets(box)
        q_prev = dict(self.mm.get_joint_pos_dict())
        self._reanchor_posture_task()
        for _ in range(max_reps):
            sol, coll, lim = self.mm.ik(
                target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
                target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
                max_iterations=200,
            )
            if sol is None or coll or not lim:
                return False
            self.mm.set_joint_pos(sol)
            perr, rerr = self._tracking_error(tl, tr)
            if perr < POS_TOL and rerr < ROT_TOL:
                return True
        if perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL:
            return True
        sol = self._global_ik_raw(tl, tr)
        if sol is None:
            return False
        jump = max(abs(sol[n] - q_prev[n]) for n in q_prev if n in sol)
        if jump > MAX_JOINT_JUMP:
            logger.warning(
                f"Global re-seed rejected: {jump:.2f} rad joint jump would "
                "break trajectory continuity while holding the box."
            )
            self.mm.set_joint_pos(q_prev)
            return False
        self.mm.set_joint_pos(robot_utils.get_qpos_from_joint_dict(self.mm.pin_robot, sol))
        perr, rerr = self._refine(tl, tr)
        return perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL

    def track_segment(
        self, box_prev: pin.SE3, box_next: pin.SE3, max_depth: int = 3
    ) -> list[dict] | None:
        """Advance from box_prev (already realized) to box_next. Oversized
        joint steps (e.g. a fast wrist roll) are bisected in Cartesian space
        so adjacent configs stay close."""
        q_prev = dict(self.mm.get_joint_pos_dict())
        if not self.solve_waypoint(box_next):
            return None
        q_next = dict(self.mm.get_joint_pos_dict())
        jump = max(abs(q_next[n] - q_prev[n]) for n in q_prev)
        if jump <= MAX_JOINT_JUMP:
            return [q_next]
        if max_depth == 0:
            logger.warning(f"Could not smooth a {jump:.2f} rad joint step.")
            return None
        self.mm.set_joint_pos(q_prev)
        box_mid = pin.SE3.Interpolate(box_prev, box_next, 0.5)
        first = self.track_segment(box_prev, box_mid, max_depth - 1)
        if first is None:
            return None
        second = self.track_segment(box_mid, box_next, max_depth - 1)
        if second is None:
            return None
        return first + second

    def find_max_tilt(self, hold_pos: np.ndarray, resolution: float = 0.03) -> float:
        """Largest feasible tilt angle along the rotate+pull path."""

        def center_at(angle: float) -> np.ndarray:
            a = angle / self.full_angle if self.full_angle > 0 else 1.0
            return (1.0 - a) * self.center0 + a * hold_pos

        # The global solver is stochastic: give the full angle two draws.
        full_box = self.box_pose(self.full_angle, center_at(self.full_angle))
        if any(self.pose_feasible(full_box) for _ in range(2)):
            return self.full_angle

        logger.warning("Fully vertical pose unreachable; searching max tilt...")
        lo, hi = 0.0, self.full_angle  # lo feasible (current pose), hi not
        while hi - lo > resolution:
            mid = 0.5 * (lo + hi)
            ok = self.pose_feasible(self.box_pose(mid, center_at(mid)))
            logger.info(f"  tilt {np.rad2deg(mid):.1f} deg: {'ok' if ok else 'unreachable'}")
            lo, hi = (mid, hi) if ok else (lo, mid)
        return lo


def rotation_waypoints(
    planner: BoxRotationPlanner, tilt: float, hold_pos: np.ndarray
) -> list[pin.SE3]:
    """Box poses from the current grasp to `tilt` while pulling to hold_pos.
    Index 0 is the start pose (realized by the start configuration)."""
    n = max(int(np.ceil(np.rad2deg(tilt) / 1.5)), 2)
    wps = [planner.box_pose(0.0, planner.center0)]
    for i in range(1, n + 1):
        a = i / n
        center = (1.0 - a) * planner.center0 + a * hold_pos
        wps.append(planner.box_pose(a * tilt, center))
    return wps


def max_adjacent_jump(configs: list[dict]) -> float:
    names = sorted(configs[0])
    q = np.array([[c[n] for n in names] for c in configs])
    return float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.0


def plan_rotation(
    mm: MotionManager,
    hold_pos: np.ndarray,
    prefer_forward: bool,
    tilt_direction: str = "up",
    max_attempts: int = 3,
) -> tuple[dict, list[dict]] | None:
    """Plan the rotate+pull trajectory.

    Returns (start_config, trajectory configs) or None. Backward planning
    (from a global-IK final pose) finds an arm branch that can complete the
    rotation; forward planning starts on the robot's current branch.
    """
    planner = BoxRotationPlanner(mm, tilt_direction=tilt_direction)
    hands_apart = np.linalg.norm(planner.left0.translation - planner.right0.translation)
    logger.info(
        f"Box grasp: hands {hands_apart:.3f} m apart, "
        f"center {np.round(planner.center0, 3)}"
    )

    tilt = planner.find_max_tilt(hold_pos)
    if tilt < planner.full_angle - 1e-6:
        logger.warning(
            f"Proceeding with best-effort tilt {np.rad2deg(tilt):.1f} deg "
            f"(fully vertical = {np.rad2deg(planner.full_angle):.1f} deg)."
        )
    else:
        logger.info(f"Fully vertical rotation ({np.rad2deg(tilt):.1f} deg) is reachable.")

    wps = rotation_waypoints(planner, tilt, hold_pos)
    start = time.time()

    def forward() -> tuple[dict, list[dict]] | None:
        logger.info("Forward tracking from the current configuration...")
        planner.mm.set_joint_pos(planner.initial_q)
        configs: list[dict] = []
        for i in range(1, len(wps)):
            seg = planner.track_segment(wps[i - 1], wps[i])
            if seg is None:
                logger.error(f"Forward tracking failed at waypoint {i}/{len(wps) - 1}")
                return None
            configs.extend(seg)
        return dict(planner.initial_q), configs

    def backward() -> tuple[dict, list[dict]] | None:
        tl, tr = planner.ee_targets(wps[-1])
        sol = planner._global_ik_raw(tl, tr)
        if sol is None:
            return None
        planner.mm.set_joint_pos(
            robot_utils.get_qpos_from_joint_dict(planner.mm.pin_robot, sol)
        )
        # Polish the (possibly sloppy) global seed.
        if not planner.solve_waypoint(wps[-1]):
            return None
        configs = [dict(planner.mm.get_joint_pos_dict())]
        for k in range(len(wps) - 2, -1, -1):
            seg = planner.track_segment(wps[k + 1], wps[k])
            if seg is None:
                logger.error(f"Backward tracking failed at waypoint {k}/{len(wps) - 1}")
                return None
            configs.extend(seg)
        configs.reverse()  # forward-in-time
        return configs[0], configs[1:]

    def validated(res, label):
        if res is None:
            return None
        jump = max_adjacent_jump([res[0]] + res[1])
        if jump > MAX_JOINT_JUMP:
            logger.warning(f"{label} plan has a {jump:.2f} rad joint jump; discarding.")
            return None
        return res

    result = None
    if prefer_forward:
        result = validated(forward(), "Forward")
    if result is None:
        # The global solver draws random seeds: retry on a rough draw.
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Backward planning attempt {attempt}/{max_attempts}...")
            result = validated(backward(), "Backward")
            if result is not None:
                break
    if result is None and not prefer_forward:
        result = validated(forward(), "Forward")
    if result is None:
        return None

    logger.info(
        f"Planned {len(result[1])} configs in {time.time() - start:.1f}s."
    )
    return result


def move_to_config(
    bot: Robot,
    torso_joints: list[str],
    config: dict,
    max_joint_vel: float = 0.3,
    control_hz: float = 100.0,
) -> None:
    """Joint-space move of both arms (and torso) to a configuration.

    Streams interpolated set_joint_pos commands (same mechanism as the
    trajectory execution) instead of relying on the motion plugin's
    move_to_joint_pos handles, whose completion signal is unreliable in
    some simulator setups.
    """
    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
    components = ["left_arm", "right_arm"] + (["torso"] if torso_joints else [])
    current = bot.get_joint_pos_dict(component=components)
    q0 = np.array([current[n] for n in names])
    q1 = np.array([config[n] for n in names])

    travel = float(np.abs(q1 - q0).max())
    duration = float(np.clip(travel / max_joint_vel, 2.0, 15.0))
    logger.info(f"Setup move: {travel:.2f} rad max joint travel, {duration:.1f}s")

    rate = RateLimiter(control_hz)
    n_ticks = max(int(duration * control_hz), 1)
    for t in range(1, n_ticks + 1):
        a = 0.5 * (1.0 - np.cos(np.pi * t / n_ticks))
        q = (1.0 - a) * q0 + a * q1
        bot.left_arm.set_joint_pos(q[:7])
        bot.right_arm.set_joint_pos(q[7:14])
        if torso_joints:
            bot.torso.set_joint_pos(q[14:])
        rate.sleep()

    time.sleep(0.5)
    reached = bot.get_joint_pos_dict(component=components)
    err = max(abs(reached[n] - config[n]) for n in names)
    if err > 0.1:
        logger.warning(f"Setup pose reached with {err:.2f} rad residual error.")


def execute_trajectory(
    bot: Robot,
    torso_joints: list[str],
    start_config: dict,
    configs: list[dict],
    duration: float,
    control_hz: float,
) -> None:
    """Stream the trajectory, densely resampled in joint space with a smooth
    (cosine) time profile."""
    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
    knots = np.array([[c[n] for n in names] for c in [start_config] + configs])
    n_seg = len(knots) - 1

    rate = RateLimiter(control_hz)
    n_ticks = max(int(duration * control_hz), 1)
    for t in range(1, n_ticks + 1):
        s = 0.5 * (1.0 - np.cos(np.pi * t / n_ticks)) * n_seg
        idx = min(int(s), n_seg - 1)
        frac = s - idx
        q = (1.0 - frac) * knots[idx] + frac * knots[idx + 1]
        bot.left_arm.set_joint_pos(q[:7])
        bot.right_arm.set_joint_pos(q[7:14])
        if torso_joints:
            bot.torso.set_joint_pos(q[14:])
        rate.sleep()


def main(
    hold_pos: tuple[float, float, float] = (0.28, 0.0, 0.80),
    duration: float = 8.0,
    control_hz: float = 100.0,
    setup_pose: bool = True,
    torso_damping: float = 1.0,
    tilt_direction: str = "up",
) -> None:
    """Pitch a bimanually-held box up to vertical while pulling it inward.

    The box is rotated about the axis connecting the two hands, tipping its
    forward-pointing long axis to vertical.

    Args:
        hold_pos: Box-center position (robot base frame, meters) to pull
            toward while rotating; keep it close to the chest so the
            vertical pose stays reachable.
        duration: Execution time of the rotation in seconds.
        control_hz: Joint command streaming frequency in Hz.
        setup_pose: If True, first move to the planned grasp configuration
            and wait until the box is gripped. Disable with --no-setup-pose
            when the box is already held.
        torso_damping: Local-IK damping weight for the torso joints (the
            robot default of 30000 freezes the torso, which makes the
            rotation untrackable with arms alone).
        tilt_direction: "up" tips the far end of the box upward, "down"
            tips it toward the floor.
    """
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please make sure the arms have enough space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    bot = Robot()

    try:
        components = ["left_arm", "right_arm", "head"]
        has_torso = bot.has_component("torso")
        if has_torso:
            components.append("torso")
        initial_joint_pos = bot.get_joint_pos_dict(component=components)

        lock_regions = ["HEAD", "BASE", "LEFT_HAND", "RIGHT_HAND"]
        damping_override = {"L_arm_j2": 100.0, "R_arm_j2": 100.0}
        if has_torso:
            damping_override.update({name: torso_damping for name in TORSO_JOINTS})
        else:
            lock_regions.append("TORSO")
        mm = MotionManager(
            initial_joint_configuration_dict=initial_joint_pos,
            init_planner=False,
            init_trajectory_generator=False,
            init_trajectory_smoother=False,
            init_visualizer=False,
            joint_regions_to_lock=lock_regions,
            custom_local_ik_config=LocalPinkIKConfig(
                target_frames=["L_ee", "R_ee"],
                damping_weights=IKDampingWeightsConfig(
                    default=1e-12, override=damping_override
                ),
            ),
        )
        torso_joints = sorted(
            n for n in mm.get_joint_pos_dict() if n.startswith("torso_")
        )

        if setup_pose:
            # Canonical horizontal grasp: define the box grasp from the
            # arms-at-zero pose (the robot moves straight to the planned
            # grasp branch later, without visiting zeros).
            q_canonical = dict(mm.get_joint_pos_dict())
            for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
                q_canonical[name] = 0.0
            mm.set_joint_pos(q_canonical)

        logger.info("Planning the rotation (validated before any movement)...")
        result = plan_rotation(
            mm,
            hold_pos=np.asarray(hold_pos),
            prefer_forward=not setup_pose,
            tilt_direction=tilt_direction,
        )
        if result is None:
            logger.error("Planning failed. Robot was not moved.")
            return
        start_config, configs = result

        if setup_pose:
            logger.info("Moving to the grasp/setup configuration...")
            move_to_config(bot, torso_joints, start_config)
            input("Grip the box with both hands, then press Enter to continue...")
        else:
            exec_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
            mismatch = max(
                abs(start_config[n] - initial_joint_pos[n]) for n in exec_names
            )
            if mismatch > MAX_JOINT_JUMP:
                logger.error(
                    f"Planned start differs from the current configuration by "
                    f"{mismatch:.2f} rad. Re-run with --setup-pose."
                )
                return

        logger.info("Executing rotation...")
        execute_trajectory(
            bot, torso_joints, start_config, configs, duration, control_hz
        )
        logger.info("Rotation completed. The grippers are still holding the box.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    except TimeoutError as exc:
        logger.error(f"Motion timed out: {exc}")
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
