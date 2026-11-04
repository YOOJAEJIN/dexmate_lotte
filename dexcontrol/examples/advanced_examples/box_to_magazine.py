# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Bimanual box reorientation: rotate a held box to vertical and insert it.

Motion (all under the rigid-grasp constraint: the relative pose between the
two hands never changes, so the box is not twisted or dropped):

1. rotate+pull: rotate the box toward vertical (left hand up, right hand
   down) while pulling it closer to the chest to stay inside the reachable
   workspace. If full vertical is infeasible, a binary search finds the
   maximum reachable tilt and the motion proceeds with that tilt.
2. carry: move the box to the magazine entry position.
3. insert: push the box along its long axis into the magazine. If the full
   depth is unreachable, the insertion is truncated with a warning.

Planning strategy depends on whether the box is already gripped:

- --setup-pose (default): the final pose is solved with global IK (which
  picks an arm branch with slack) and the whole Cartesian path is tracked
  BACKWARD with local IK. Reversed, this gives a continuous forward
  trajectory that ends comfortably. Its first configuration is the
  grasp/setup pose (same horizontal grasp, good branch): the robot moves
  there BEFORE the box is gripped, then waits.
- --no-setup-pose (box already held): the path is tracked FORWARD from the
  robot's current configuration, which guarantees the trajectory starts on
  the actual arm branch. The insert phase may be truncated where that
  branch runs out of reach.

Joint-space continuity is validated either way (large steps are bisected in
Cartesian space; discontinuous branch jumps are rejected).

The whole trajectory is planned and validated before the robot moves. The
grippers are never commanded: release the box with your own logic after
insertion.

Requires a running robot or simulator reachable via Zenoh (e.g. Isaac Sim
with ROBOT_NAME set).

Usage:
    python ik_test.py --magazine-pos 0.32 0.0 0.80 --insert-depth 0.05
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
# The local QP solver can plateau just above the strict tolerance (damping /
# posture-task equilibrium; measured ~9-13 mm near joint limits). Accept such
# plateaus up to these bounds instead of re-seeding: a small steady offset is
# tolerable for a firmly gripped box, while a global re-seed can jump to a
# distant joint-space branch (dangerous to execute while holding it).
SOFT_POS_TOL = 0.015  # m
SOFT_ROT_TOL = 3.0  # deg
# Global-IK solutions are only trusted after an FK check: the hybrid solver
# can report "solved" with its best-effort (centimeters-off) solution.
GLOBAL_ACCEPT_POS = 0.02  # m
GLOBAL_ACCEPT_ROT = 5.0  # deg
# Max per-joint change between adjacent waypoints. A global re-seed whose
# configuration jumps further than this is rejected: sweeping between IK
# branches would violate the grasp constraint mid-motion.
MAX_JOINT_JUMP = 0.35  # rad


def rot_err_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices in degrees."""
    val = (np.trace(rot_a @ rot_b.T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(val, -1.0, 1.0))))


class BoxMotionPlanner:
    """Plans a rigid-grasp box trajectory as a list of joint configurations.

    The grasp (hand poses relative to the box) is captured from the motion
    manager's configuration at construction time.
    """

    def __init__(self, mm: MotionManager):
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

        # Box long axis (right hand -> left hand) and the rotation that
        # brings it to vertical: rotate about axis = d x z by angle(d, z),
        # which sends the left hand up and the right hand down.
        d = self.left0.translation - self.right0.translation
        self.box_axis0 = d / np.linalg.norm(d)
        z = np.array([0.0, 0.0, 1.0])
        rot_axis = np.cross(self.box_axis0, z)
        self.rot_axis = rot_axis / np.linalg.norm(rot_axis)
        self.full_angle = float(np.arccos(np.clip(self.box_axis0 @ z, -1.0, 1.0)))

        self.initial_q = dict(mm.get_joint_pos_dict())
        self.ik_calls = 0
        self.global_fallbacks = 0
        self.soft_accepts = 0

    def box_pose(self, angle: float, center: np.ndarray) -> pin.SE3:
        rot = R.from_rotvec(angle * self.rot_axis).as_matrix()
        return pin.SE3(rot, np.asarray(center, dtype=float))

    def ee_targets(self, box: pin.SE3) -> tuple[pin.SE3, pin.SE3]:
        return box * self.grasp_left, box * self.grasp_right

    def _reanchor_posture_task(self) -> None:
        """Re-anchor the local solver's posture task to the current config.

        The posture task pulls toward its (initially fixed) target, which adds
        a steadily growing tracking residual as the trajectory moves away from
        the start pose. Re-anchoring per waypoint removes that pull while
        keeping the task as a local regularizer.
        """
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
        """Global IK solve. NOTE: the hybrid solver may report success with a
        best-effort solution that is centimeters off (e.g. without the CasADi
        backends), so callers must verify/refine the result."""
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
        """Polish the current configuration toward the targets with local IK
        and return the remaining tracking error."""
        perr, rerr = self._tracking_error(tl, tr)
        for _ in range(reps):
            if perr < POS_TOL and rerr < ROT_TOL:
                break
            sol, coll, lim = self.mm.ik(
                target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
                target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
                max_iterations=200,
            )
            self.ik_calls += 1
            if sol is None or coll or not lim:
                break
            self.mm.set_joint_pos(sol)
            perr, rerr = self._tracking_error(tl, tr)
        return perr, rerr

    def pose_feasible(self, box: pin.SE3) -> bool:
        """Check reachability of a box pose: global solve, local polish,
        FK verification. The model state is restored afterwards."""
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
        """Drive the internal model to the waypoint. Local IK converges like a
        low-gain tracker, so iterate until tolerance; accept small plateaus;
        re-seed globally only if the result stays joint-space continuous."""
        tl, tr = self.ee_targets(box)
        q_prev = dict(self.mm.get_joint_pos_dict())
        self._reanchor_posture_task()
        for _ in range(max_reps):
            sol, coll, lim = self.mm.ik(
                target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
                target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
                max_iterations=200,
            )
            self.ik_calls += 1
            if sol is None or coll or not lim:
                return False
            self.mm.set_joint_pos(sol)
            perr, rerr = self._tracking_error(tl, tr)
            if perr < POS_TOL and rerr < ROT_TOL:
                return True
        if perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL:
            # Plateau close to the target: keep it rather than branch-jump.
            self.soft_accepts += 1
            return True
        self.global_fallbacks += 1
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
        """Advance the model from box_prev (already realized by the current
        configuration) to box_next, returning the intermediate configs.

        If the joint-space step is too large (e.g. a wrist rolling quickly
        near the end of the rotation), the Cartesian segment is bisected
        recursively so adjacent configs stay close and linear joint
        interpolation between them keeps the box constraint tight.
        """
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
        """Return the largest feasible tilt angle at the pulled-in hold pose.

        Feasibility of an angle is checked at the box center interpolated
        between the start and hold positions (same path the rotation phase
        uses). Returns self.full_angle when fully vertical is reachable.
        """

        def center_at(angle: float) -> np.ndarray:
            a = angle / self.full_angle if self.full_angle > 0 else 1.0
            return (1.0 - a) * self.center0 + a * hold_pos

        # The global solver is stochastic: give the full angle two draws
        # before concluding it is unreachable.
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


def build_waypoints(
    planner: BoxMotionPlanner,
    tilt: float,
    hold_pos: np.ndarray,
    magazine_pos: np.ndarray,
    insert_depth: float,
) -> list[tuple[str, list[pin.SE3]]]:
    """Build the box-pose waypoint segments for the three phases."""
    tilt_deg = np.rad2deg(tilt)
    rot_final = R.from_rotvec(tilt * planner.rot_axis).as_matrix()
    # Box long axis after rotation; insertion pushes toward its bottom end.
    insert_dir = -(rot_final @ planner.box_axis0)

    n_rot = max(int(np.ceil(tilt_deg / 1.5)), 2)
    rotate_wps = []
    for i in range(1, n_rot + 1):
        a = i / n_rot
        center = (1.0 - a) * planner.center0 + a * hold_pos
        rotate_wps.append(planner.box_pose(a * tilt, center))

    carry_dist = float(np.linalg.norm(magazine_pos - hold_pos))
    n_carry = max(int(np.ceil(carry_dist / 0.01)), 2)
    carry_wps = [
        planner.box_pose(tilt, (1.0 - i / n_carry) * hold_pos + (i / n_carry) * magazine_pos)
        for i in range(1, n_carry + 1)
    ]

    n_ins = max(int(np.ceil(insert_depth / 0.005)), 2)
    insert_wps = [
        planner.box_pose(tilt, magazine_pos + (i / n_ins) * insert_depth * insert_dir)
        for i in range(1, n_ins + 1)
    ]

    return [("rotate+pull", rotate_wps), ("carry", carry_wps), ("insert", insert_wps)]


def plan_backward(
    planner: BoxMotionPlanner,
    segments: list[tuple[str, list[pin.SE3]]],
    insert_depth: float,
) -> tuple[dict, dict[str, list[dict]]] | None:
    """Plan by tracking the Cartesian path backward from a global-IK final
    configuration, so the whole trajectory lives on one comfortable branch.

    Returns (start_config, {phase: [config, ...]}) or None on failure.
    """
    start_wp = planner.box_pose(0.0, planner.center0)
    flat: list[pin.SE3] = [start_wp]
    counts: list[int] = []
    for _, wps in segments:
        flat.extend(wps)
        counts.append(len(wps))
    insert_wps = segments[-1][1]
    carry_end_idx = len(flat) - 1 - len(insert_wps)

    # Find the deepest feasible endpoint: prefer full insertion, otherwise
    # binary-search how far into the magazine the arms can follow.
    if planner.pose_feasible(flat[-1]):
        final_idx = len(flat) - 1
    elif not planner.pose_feasible(flat[carry_end_idx]):
        logger.error("Magazine entry pose is unreachable. Move the magazine.")
        return None
    else:
        lo, hi = carry_end_idx, len(flat) - 1  # lo feasible, hi not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if planner.pose_feasible(flat[mid]):
                lo = mid
            else:
                hi = mid
        final_idx = lo
    achieved = (final_idx - carry_end_idx) / len(insert_wps) * insert_depth
    if final_idx < len(flat) - 1:
        logger.warning(
            f"Insertion depth limited to {achieved * 100:.1f} cm of "
            f"requested {insert_depth * 100:.1f} cm."
        )

    tl_f, tr_f = planner.ee_targets(flat[final_idx])
    sol = planner._global_ik_raw(tl_f, tr_f)
    if sol is None:
        return None
    planner.mm.set_joint_pos(
        robot_utils.get_qpos_from_joint_dict(planner.mm.pin_robot, sol)
    )
    # Polish the (possibly sloppy) global seed with local iterations so the
    # trajectory endpoint is within tolerance.
    if not planner.solve_waypoint(flat[final_idx]):
        logger.warning("Could not refine the final configuration; retrying.")
        return None

    # tagged: (forward-time key, config). Bisected intermediates between
    # waypoints k and k+1 get fractional keys in (k, k+1).
    tagged = [(float(final_idx), dict(planner.mm.get_joint_pos_dict()))]
    for k in range(final_idx - 1, -1, -1):
        seg = planner.track_segment(flat[k + 1], flat[k])
        if seg is None:
            logger.error(
                f"Backward tracking failed at waypoint {k}/{final_idx}. "
                "Try a different hold/magazine position."
            )
            return None
        m = len(seg)
        for i, config in enumerate(seg):  # travel order: toward flat[k]
            key = float(k) if i == m - 1 else k + (m - 1 - i) / m
            tagged.append((key, config))
    tagged.sort(key=lambda t: t[0])  # forward-in-time

    start_config = tagged[0][1]
    phases: dict[str, list[dict]] = {}
    lo = 0.0
    for (name, _), n in zip(segments, counts):
        hi = lo + n
        phases[name] = [c for key, c in tagged if lo < key <= hi]
        lo = hi
    return start_config, phases


def max_adjacent_jump(start_config: dict, phases: dict[str, list[dict]]) -> float:
    """Largest per-joint change between adjacent configs of the trajectory."""
    seq = [start_config] + [c for configs in phases.values() for c in configs]
    names = sorted(start_config)
    q = np.array([[c[n] for n in names] for c in seq])
    return float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.0


def plan_box_motion(
    mm: MotionManager,
    magazine_pos: np.ndarray,
    insert_depth: float,
    hold_pos: np.ndarray,
    max_attempts: int = 3,
    prefer_forward: bool = False,
) -> tuple[dict, dict[str, list[dict]]] | None:
    """Plan all three phases. Returns (start_config, phases) or None.

    With prefer_forward=True (box already held), forward tracking from the
    current configuration is tried first: it is guaranteed to start on the
    robot's actual arm branch. Otherwise backward planning is tried first,
    which picks the branch from the final pose (best for the setup flow).
    """
    planner = BoxMotionPlanner(mm)
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

    segments = build_waypoints(planner, tilt, hold_pos, magazine_pos, insert_depth)

    start = time.time()

    def validated(res, label):
        if res is None:
            return None
        jump = max_adjacent_jump(*res)
        if jump > MAX_JOINT_JUMP:
            logger.warning(
                f"{label} plan has a {jump:.2f} rad joint jump between "
                "waypoints; discarding it."
            )
            return None
        return res

    def try_forward():
        logger.info("Forward tracking from the current configuration...")
        return validated(
            plan_forward_simple(planner, segments, insert_depth), "Forward"
        )

    def try_backward():
        # The global solver draws random seeds, so the chosen branch (and the
        # smoothness of the backward pass) varies per attempt: retry on a
        # rough draw.
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Backward planning attempt {attempt}/{max_attempts}...")
            res = validated(plan_backward(planner, segments, insert_depth), "Backward")
            if res is not None:
                return res
        return None

    if prefer_forward:
        result = try_forward() or try_backward()
    else:
        result = try_backward() or try_forward()
    if result is None:
        return None

    start_config, phases = result
    for name, configs in phases.items():
        logger.info(f"[{name}] planned {len(configs)} waypoints")
    logger.info(
        f"Planning done in {time.time() - start:.1f}s "
        f"({planner.ik_calls} IK calls, {planner.soft_accepts} soft-tolerance "
        f"waypoints, {planner.global_fallbacks} global re-seeds)"
    )
    return start_config, phases


def plan_forward_simple(
    planner: BoxMotionPlanner,
    segments: list[tuple[str, list[pin.SE3]]],
    insert_depth: float,
) -> tuple[dict, dict[str, list[dict]]] | None:
    """Forward tracking from the current configuration.

    Starts on the robot's actual arm branch by construction. The insert
    phase may be truncated when the arms cannot follow the full depth.
    """
    planner.mm.set_joint_pos(planner.initial_q)
    box_prev = planner.box_pose(0.0, planner.center0)
    phases: dict[str, list[dict]] = {}
    for name, wps in segments:
        configs: list[dict] = []
        for i, box in enumerate(wps):
            seg = planner.track_segment(box_prev, box)
            if seg is None:
                if name == "insert" and configs:
                    achieved = i / len(wps) * insert_depth
                    logger.warning(
                        f"Insertion depth limited to {achieved * 100:.1f} cm of "
                        f"requested {insert_depth * 100:.1f} cm."
                    )
                    break
                logger.error(f"[{name}] IK failed at waypoint {i + 1}/{len(wps)}")
                return None
            configs.extend(seg)
            box_prev = box
        phases[name] = configs
    return dict(planner.initial_q), phases


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


def execute_phase(
    bot: Robot,
    torso_joints: list[str],
    prev_config: dict,
    configs: list[dict],
    duration: float,
    control_hz: float,
) -> dict:
    """Stream one phase, densely resampled in joint space with a smooth
    (cosine) time profile. Adjacent waypoints are close together, so linear
    joint interpolation stays on the grasp-constraint manifold."""
    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
    knots = np.array([[c[n] for n in names] for c in [prev_config] + configs])
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
    return configs[-1]


def main(
    magazine_pos: tuple[float, float, float] = (0.32, 0.0, 0.80),
    insert_depth: float = 0.05,
    hold_pos: tuple[float, float, float] = (0.28, 0.0, 0.80),
    rotate_duration: float = 8.0,
    carry_duration: float = 4.0,
    insert_duration: float = 3.0,
    control_hz: float = 100.0,
    setup_pose: bool = True,
    torso_damping: float = 1.0,
) -> None:
    """Rotate a bimanually-held box to vertical and insert it into a magazine.

    Args:
        magazine_pos: Box-center position (robot base frame, meters) at the
            magazine entry, before insertion.
        insert_depth: How far to push the box along its long axis (meters).
        hold_pos: Box-center position to pull toward while rotating; keep it
            close to the chest so the vertical pose stays reachable.
        rotate_duration: Execution time of the rotate+pull phase (seconds).
        carry_duration: Execution time of the carry phase (seconds).
        insert_duration: Execution time of the insert phase (seconds).
        control_hz: Joint command streaming frequency in Hz.
        setup_pose: If True, first move to the planned grasp configuration
            (horizontal box grasp on the branch that can complete the whole
            motion) and wait until the box is gripped. Disable with
            --no-setup-pose when the box is already held.
        torso_damping: Local-IK damping weight for the torso joints. The
            robot's default config effectively freezes the torso (weight
            30000), which makes the motion untrackable with arms alone.
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

        # Torso joints stay active in the IK to enlarge the workspace, with
        # a damping override: some robot configs (e.g. vega_1p) default the
        # torso damping to 30000, freezing it in local IK.
        lock_regions = ["HEAD", "BASE", "LEFT_HAND", "RIGHT_HAND"]
        damping_override = {"L_arm_j2": 100.0, "R_arm_j2": 100.0}
        if has_torso:
            damping_override.update({name: torso_damping for name in TORSO_JOINTS})
        else:
            lock_regions.append("TORSO")
        local_ik_config = LocalPinkIKConfig(
            target_frames=["L_ee", "R_ee"],
            damping_weights=IKDampingWeightsConfig(
                default=1e-12, override=damping_override
            ),
        )
        mm = MotionManager(
            initial_joint_configuration_dict=initial_joint_pos,
            init_planner=False,
            init_trajectory_generator=False,
            init_trajectory_smoother=False,
            init_visualizer=False,
            joint_regions_to_lock=lock_regions,
            custom_local_ik_config=local_ik_config,
        )
        torso_joints = sorted(
            n for n in mm.get_joint_pos_dict() if n.startswith("torso_")
        )

        if setup_pose:
            # Canonical horizontal grasp: define the box grasp from the
            # arms-at-zero pose (the robot itself moves straight to the
            # planned grasp branch later, without visiting zeros).
            q_canonical = dict(mm.get_joint_pos_dict())
            for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
                q_canonical[name] = 0.0
            mm.set_joint_pos(q_canonical)

        logger.info("Planning box motion (validated before any movement)...")
        result = plan_box_motion(
            mm,
            magazine_pos=np.asarray(magazine_pos),
            insert_depth=insert_depth,
            hold_pos=np.asarray(hold_pos),
            # With the box already held, start from the robot's actual branch.
            prefer_forward=not setup_pose,
        )
        if result is None:
            logger.error(
                "Planning failed. Robot was not moved. "
                "Try a different magazine/hold position."
            )
            return
        start_config, phases = result

        exec_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        if setup_pose:
            logger.info("Moving to the grasp/setup configuration...")
            move_to_config(bot, torso_joints, start_config)
            input("Grip the box with both hands, then press Enter to continue...")
        else:
            mismatch = max(
                abs(start_config[n] - initial_joint_pos[n]) for n in exec_names
            )
            if mismatch > MAX_JOINT_JUMP:
                logger.error(
                    f"Planned start differs from the current configuration by "
                    f"{mismatch:.2f} rad (different arm branch). Executing it "
                    "while holding the box is unsafe. Re-run with --setup-pose "
                    "(grip the box after the setup move)."
                )
                return

        current = start_config
        for name, duration in [
            ("rotate+pull", rotate_duration),
            ("carry", carry_duration),
            ("insert", insert_duration),
        ]:
            if not phases.get(name):
                logger.warning(f"Phase '{name}' is empty; skipping.")
                continue
            logger.info(f"Executing phase: {name}")
            current = execute_phase(
                bot, torso_joints, current, phases[name], duration, control_hz
            )
        logger.info(
            "Box motion completed. The grippers are still holding; "
            "release them once the box is secured in the magazine."
        )
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
    except TimeoutError as exc:
        logger.error(f"Motion timed out: {exc}")
    finally:
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
