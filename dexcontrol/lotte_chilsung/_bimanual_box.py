# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Shared rigid-grasp bimanual box planning helpers for lotte_chilsung demos."""

from __future__ import annotations

import time

import numpy as np
import pinocchio as pin
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

POS_TOL = 0.005  # m
ROT_TOL = 2.0  # deg
SOFT_POS_TOL = 0.015  # m
SOFT_ROT_TOL = 3.0  # deg
MAX_JOINT_JUMP = 0.35  # rad

# Single-motion demos: keep local Pink IK lean (GlobalHybrid / OMPL / Viser off).
IK_MAX_ITERS = 50
IK_MAX_REPS = 6


def make_fast_mm(
    hold_q: dict[str, float],
    joint_regions_to_lock: list[str],
    *,
    need_traj_gen: bool = False,
) -> MotionManager:
    """MotionManager tuned for one-shot demos (skip unused heavy init).

    Biggest win: ``init_global_ik=False`` — avoids GlobalHybridIK / Casadi setup
    that otherwise runs on every script launch even when only local IK is used.
    """
    return MotionManager(
        initial_joint_configuration_dict=hold_q,
        init_global_ik=False,
        init_planner=False,
        init_trajectory_generator=need_traj_gen,
        init_trajectory_smoother=False,
        init_visualizer=False,
        joint_regions_to_lock=joint_regions_to_lock,
    )


def connect_bot(auto_shutdown: bool = False) -> Robot:
    """Connect for lotte_chilsung demos without camera/sensor init."""
    return Robot(auto_shutdown=auto_shutdown, init_sensors=False)


def disconnect_leave_held(bot: Robot) -> None:
    """Drop the client without ``stop()``, so last joint targets remain.

    ``Robot.shutdown()`` calls ``stop()`` first, which re-commands current
    joints at zero velocity and is what makes the arms go limp. We still
    close publishers/subscribers and release the shared Zenoh node so the
    next teleop / control client can take over cleanly.
    """
    if bot._shutdown_called:
        return
    bot._shutdown_called = True

    try:
        from dexcontrol.robot import _active_robots

        _active_robots.discard(bot)
    except Exception:
        pass

    if hasattr(bot, "_idle_monitor"):
        try:
            bot._idle_monitor.stop()
        except Exception as exc:
            logger.warning(f"Idle monitor stop failed: {exc}")

    # Intentionally skip component.stop().

    try:
        if hasattr(bot, "sensors") and bot.sensors is not None:
            bot.sensors.shutdown()
    except Exception as exc:
        logger.warning(f"Sensor shutdown failed: {exc}")

    for component in reversed(getattr(bot, "_components", [])):
        if component is None:
            continue
        try:
            component.shutdown()
        except Exception as exc:
            logger.warning(
                f"Component shutdown failed ({component.__class__.__name__}): {exc}"
            )

    try:
        bot._release_shared_node()
    except Exception as exc:
        logger.warning(f"Shared node release failed: {exc}")


def rot_err_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    val = (np.trace(rot_a @ rot_b.T) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(val, -1.0, 1.0))))


class RigidGraspPlanner:
    """Plans rigid-grasp box motions as joint-configuration sequences.

    Subclasses set ``rot_axis`` / ``full_angle`` describing how the box frame
    should rotate from the current grasp.
    """

    def __init__(self, mm: MotionManager):
        self.mm = mm
        poses = mm.fk(["L_ee", "R_ee"])
        self.left0 = pin.SE3(poses["L_ee"].homogeneous.copy())
        self.right0 = pin.SE3(poses["R_ee"].homogeneous.copy())
        self.center0 = 0.5 * (self.left0.translation + self.right0.translation)
        box0 = pin.SE3(np.eye(3), self.center0.copy())
        self.grasp_left = box0.inverse() * self.left0
        self.grasp_right = box0.inverse() * self.right0
        self.box_axis0 = self.left0.translation - self.right0.translation
        self.box_axis0 = self.box_axis0 / np.linalg.norm(self.box_axis0)
        self.rot_axis = np.array([1.0, 0.0, 0.0])
        self.full_angle = 0.0
        self.initial_q = dict(mm.get_joint_pos_dict())

    def box_pose(self, angle: float, center: np.ndarray) -> pin.SE3:
        rot = R.from_rotvec(angle * self.rot_axis).as_matrix()
        return pin.SE3(rot, np.asarray(center, dtype=float))

    def ee_targets(self, box: pin.SE3) -> tuple[pin.SE3, pin.SE3]:
        return box * self.grasp_left, box * self.grasp_right

    def _reanchor_posture_task(self) -> None:
        try:
            solver = self.mm.local_ik_solver
            posture = solver.task_dict.get("posture")
            if posture is not None:
                posture.set_target_from_configuration(solver.configuration)
        except AttributeError:
            pass

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
        perr, rerr = self._tracking_error(tl, tr)
        for _ in range(reps):
            if perr < POS_TOL and rerr < ROT_TOL:
                break
            sol, coll, lim = self.mm.ik(
                target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
                target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
                max_iterations=IK_MAX_ITERS,
            )
            if sol is None or coll or not lim:
                break
            self.mm.set_joint_pos(sol)
            perr, rerr = self._tracking_error(tl, tr)
        return perr, rerr

    def pose_feasible(self, box: pin.SE3) -> bool:
        tl, tr = self.ee_targets(box)
        sol = self._global_ik_raw(tl, tr)
        if sol is None:
            return False
        q_prev = dict(self.mm.get_joint_pos_dict())
        self.mm.set_joint_pos(robot_utils.get_qpos_from_joint_dict(self.mm.pin_robot, sol))
        perr, rerr = self._refine(tl, tr)
        self.mm.set_joint_pos(q_prev)
        return perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL

    def solve_waypoint(self, box: pin.SE3, max_reps: int = IK_MAX_REPS) -> bool:
        tl, tr = self.ee_targets(box)
        q_prev = dict(self.mm.get_joint_pos_dict())
        self._reanchor_posture_task()
        perr, rerr = 1.0, 90.0
        for _ in range(max_reps):
            sol, coll, lim = self.mm.ik(
                target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
                target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
                max_iterations=IK_MAX_ITERS,
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

    def find_max_angle(self, hold_pos: np.ndarray, resolution: float = 0.03) -> float:
        def center_at(angle: float) -> np.ndarray:
            a = angle / self.full_angle if self.full_angle > 0 else 1.0
            return (1.0 - a) * self.center0 + a * hold_pos

        if self.full_angle < 1e-6:
            return 0.0

        full_box = self.box_pose(self.full_angle, center_at(self.full_angle))
        # One GlobalHybrid probe is enough; a second retry used to double wall time
        # (~10–15s each when Casadi is unavailable).
        if self.pose_feasible(full_box):
            return self.full_angle

        logger.warning("Full target pose unreachable; searching max angle...")
        lo, hi = 0.0, self.full_angle
        while hi - lo > resolution:
            mid = 0.5 * (lo + hi)
            ok = self.pose_feasible(self.box_pose(mid, center_at(mid)))
            logger.info(
                f"  angle {np.rad2deg(mid):.1f} deg: {'ok' if ok else 'unreachable'}"
            )
            lo, hi = (mid, hi) if ok else (lo, mid)
        return lo


def box_waypoints(
    planner: RigidGraspPlanner, angle: float, hold_pos: np.ndarray
) -> list[pin.SE3]:
    n = max(int(np.ceil(np.rad2deg(max(angle, 1e-3)) / 1.5)), 2)
    wps = [planner.box_pose(0.0, planner.center0)]
    for i in range(1, n + 1):
        a = i / n
        center = (1.0 - a) * planner.center0 + a * hold_pos
        wps.append(planner.box_pose(a * angle, center))
    return wps


def max_adjacent_jump(configs: list[dict]) -> float:
    names = sorted(configs[0])
    q = np.array([[c[n] for n in names] for c in configs])
    return float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.0


def plan_box_motion(
    planner: RigidGraspPlanner,
    hold_pos: np.ndarray,
    prefer_forward: bool,
    angle_label: str = "target",
    max_attempts: int = 3,
) -> tuple[dict, list[dict]] | None:
    hands_apart = np.linalg.norm(planner.left0.translation - planner.right0.translation)
    logger.info(
        f"Box grasp: hands {hands_apart:.3f} m apart, "
        f"center {np.round(planner.center0, 3)}, "
        f"axis {np.round(planner.box_axis0, 3)}"
    )

    # prefer_forward (rotate/align after grasp): skip GlobalHybrid reachability.
    # Forward local IK + allow_partial already discovers the reachable tilt, and
    # GlobalHybrid×N was dominating wall time (~10–15s/call without Casadi).
    if prefer_forward:
        angle = planner.full_angle
        logger.info(
            f"Skipping GlobalHybrid reachability; "
            f"forward-tracking full {angle_label} {np.rad2deg(angle):.1f} deg"
        )
    else:
        angle = planner.find_max_angle(hold_pos)
        if angle < planner.full_angle - 1e-6:
            logger.warning(
                f"Proceeding with best-effort {angle_label} {np.rad2deg(angle):.1f} deg "
                f"(full = {np.rad2deg(planner.full_angle):.1f} deg)."
            )
        else:
            logger.info(
                f"Full {angle_label} ({np.rad2deg(angle):.1f} deg) is reachable."
            )

    wps = box_waypoints(planner, angle, hold_pos)
    start = time.time()

    def _start_mismatch(start_cfg: dict) -> float:
        names = [
            n
            for n in start_cfg
            if n in planner.initial_q
            and (n.startswith(("L_arm_", "R_arm_", "torso_")))
        ]
        if not names:
            return 0.0
        return max(abs(start_cfg[n] - planner.initial_q[n]) for n in names)

    def forward(*, allow_partial: bool = False) -> tuple[dict, list[dict]] | None:
        logger.info("Forward tracking from the current configuration...")
        planner.mm.set_joint_pos(planner.initial_q)
        configs: list[dict] = []
        n_seg = len(wps) - 1
        for i in range(1, len(wps)):
            seg = planner.track_segment(wps[i - 1], wps[i])
            if seg is None:
                frac = (i - 1) / max(n_seg, 1)
                if allow_partial and configs and frac >= 0.1:
                    logger.warning(
                        f"Forward tracking stopped at waypoint {i}/{n_seg}; "
                        f"using partial path (~{frac * 100:.0f}% of target tilt)."
                    )
                    return dict(planner.initial_q), configs
                logger.error(f"Forward tracking failed at waypoint {i}/{n_seg}")
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
        if not planner.solve_waypoint(wps[-1]):
            return None
        configs = [dict(planner.mm.get_joint_pos_dict())]
        for k in range(len(wps) - 2, -1, -1):
            seg = planner.track_segment(wps[k + 1], wps[k])
            if seg is None:
                logger.error(f"Backward tracking failed at waypoint {k}/{len(wps) - 1}")
                return None
            configs.extend(seg)
        configs.reverse()
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
        # Prefer a path that starts at the current hold. If the full tilt
        # collides mid-way, keep the continuous prefix instead of a backward
        # plan whose start pose jumps away from the current configuration.
        result = validated(forward(allow_partial=True), "Forward")
        if result is None:
            logger.warning(
                "Forward track failed early; trying reduced-angle forward search..."
            )
            lo, hi = 0.0, angle
            best = None
            while hi - lo > 0.03:
                mid = 0.5 * (lo + hi)
                mid_wps = box_waypoints(planner, mid, hold_pos)
                planner.mm.set_joint_pos(planner.initial_q)
                configs: list[dict] = []
                ok = True
                for i in range(1, len(mid_wps)):
                    seg = planner.track_segment(mid_wps[i - 1], mid_wps[i])
                    if seg is None:
                        ok = False
                        break
                    configs.extend(seg)
                if ok and configs:
                    cand = validated(
                        (dict(planner.initial_q), configs),
                        f"Forward@{np.rad2deg(mid):.1f}deg",
                    )
                    if cand is not None:
                        lo = mid
                        best = cand
                        continue
                hi = mid
            result = best
            if result is not None:
                logger.warning(
                    f"Using reduced forward {angle_label} "
                    f"{np.rad2deg(lo):.1f} deg (full={np.rad2deg(angle):.1f})."
                )

    if result is None:
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Backward planning attempt {attempt}/{max_attempts}...")
            cand = validated(backward(), "Backward")
            if cand is None:
                continue
            mismatch = _start_mismatch(cand[0])
            if prefer_forward and mismatch > MAX_JOINT_JUMP:
                logger.warning(
                    f"Backward plan start differs from current hold by "
                    f"{mismatch:.2f} rad; discarding (unsafe while gripping)."
                )
                continue
            result = cand
            break
    if result is None and not prefer_forward:
        result = validated(forward(allow_partial=False), "Forward")
    if result is None:
        return None

    logger.info(f"Planned {len(result[1])} configs in {time.time() - start:.1f}s.")
    return result

def create_motion_manager(
    bot: Robot,
    torso_damping: float = 1.0,
    lock_torso: bool = False,
    init_global_ik: bool = False,
    collision_margin: float = 0.02,
    avoid_self_collisions: bool = True,
) -> tuple[MotionManager, list[str], dict]:
    """Build a MotionManager for bimanual box IK.

    Args:
        bot: Connected robot.
        torso_damping: Local-IK damping for torso joints when unlocked.
            Low values let the torso assist; high values resist torso motion.
        lock_torso: If True, freeze torso at the current configuration (IK
            and execution use arms only). Preferred when the torso must stay put.
        init_global_ik: Enable GlobalHybridIK (slow init). Default False for
            fast single-motion demos; set True for rotate/align feasibility.
        collision_margin: Pink self-collision barrier distance (m). Init fails
            if the *current* pose already has a link pair closer than this
            (e.g. fully folded elbow ~0.0197 m) — pass a smaller value then.
        avoid_self_collisions: Set False to disable Pink's self-collision
            barrier entirely (last resort when arms are already touching).
    """
    components = ["left_arm", "right_arm", "head"]
    has_torso = bot.has_component("torso")
    if has_torso:
        components.append("torso")
    initial_joint_pos = bot.get_joint_pos_dict(component=components)

    lock_regions = ["HEAD", "BASE", "LEFT_HAND", "RIGHT_HAND"]
    damping_override = {"L_arm_j2": 100.0, "R_arm_j2": 100.0}
    if not has_torso or lock_torso:
        lock_regions.append("TORSO")
    elif has_torso:
        damping_override.update({name: torso_damping for name in TORSO_JOINTS})

    mm = MotionManager(
        initial_joint_configuration_dict=initial_joint_pos,
        init_global_ik=init_global_ik,
        init_planner=False,
        init_trajectory_generator=False,
        init_trajectory_smoother=False,
        init_visualizer=False,
        joint_regions_to_lock=lock_regions,
        custom_local_ik_config=LocalPinkIKConfig(
            target_frames=["L_ee", "R_ee"],
            collision_margin=collision_margin,
            avoid_self_collisions=avoid_self_collisions,
            damping_weights=IKDampingWeightsConfig(
                default=1e-12, override=damping_override
            ),
        ),
    )
    if lock_torso or not has_torso:
        # Do not stream torso commands during execute_trajectory.
        torso_joints: list[str] = []
    else:
        torso_joints = sorted(
            n for n in mm.get_joint_pos_dict() if n.startswith("torso_")
        )
    return mm, torso_joints, initial_joint_pos


def _min_jerk(tau: float) -> float:
    """Quintic s-curve: zero velocity and acceleration at both ends."""
    tau = min(max(tau, 0.0), 1.0)
    return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def _measured(bot: Robot, torso_joints: list[str]) -> np.ndarray | None:
    comps = ["left_arm", "right_arm"] + (["torso"] if torso_joints else [])
    try:
        d = bot.get_joint_pos_dict(component=comps)
        return np.array(
            [d[n] for n in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints], dtype=float
        )
    except Exception:
        return None


def execute_trajectory(
    bot: Robot,
    torso_joints: list[str],
    start_config: dict,
    configs: list[dict],
    duration: float,
    control_hz: float,
    *,
    leash_rad: float = 0.05,
    min_rate: float = 0.3,
    settle_tol: float = 0.01,
    settle_timeout: float = 6.0,
    settle_still_ticks: int = 50,
) -> None:
    """Stream joint knots to the robot with a smooth, tracking-aware profile.

    Compared with a plain time-based blend this avoids end-of-motion wobble on
    the simulator (which runs below real time, so a wall-clock schedule sweeps
    the target faster than the PD loop can follow):

      * quintic (minimum-jerk) time profile: zero velocity *and* acceleration
        at both ends, no kink at the final knot;
      * soft pacing: while the measured joints lag the command by more than
        ``leash_rad`` the schedule slows down proportionally (never below
        ``min_rate`` of nominal, never a full stop), so the target does not
        run away from the robot and the motion stays continuous;
      * settle phase: keeps commanding the final knot until the measured
        joints have stopped moving for ``settle_still_ticks`` ticks (or
        ``settle_timeout`` s).

    ``duration`` is the nominal time; the actual wall time grows when the
    robot lags.
    """
    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
    knots = np.array([[c[n] for n in names] for c in [start_config] + configs])
    n_seg = len(knots) - 1
    if n_seg < 1:
        return

    def q_at(s: float) -> np.ndarray:
        idx = min(int(s), n_seg - 1)
        frac = s - idx
        return (1.0 - frac) * knots[idx] + frac * knots[idx + 1]

    def command(q: np.ndarray) -> None:
        bot.left_arm.set_joint_pos(q[:7])
        bot.right_arm.set_joint_pos(q[7:14])
        if torso_joints:
            bot.torso.set_joint_pos(q[14:])

    rate = RateLimiter(control_hz)
    dt = 1.0 / control_hz
    duration = max(float(duration), 0.5)
    min_rate = float(np.clip(min_rate, 0.05, 1.0))
    tau_time = 0.0  # virtual time along the profile (slowed while lagging)
    wall0 = time.time()
    hard_deadline = wall0 + duration / min_rate + 5.0
    q_cmd = knots[0].copy()
    max_lag_seen = 0.0
    slowed_ticks = 0

    while tau_time < duration:
        q_meas = _measured(bot, torso_joints)
        lag = float(np.max(np.abs(q_cmd - q_meas))) if q_meas is not None else 0.0
        max_lag_seen = max(max_lag_seen, lag)
        if lag > leash_rad and time.time() < hard_deadline:
            factor = max(min_rate, leash_rad / lag)
            slowed_ticks += 1
        else:
            factor = 1.0
        tau_time += dt * factor
        s = _min_jerk(tau_time / duration) * n_seg
        q_cmd = q_at(s)
        command(q_cmd)
        rate.sleep()

    q_final = knots[-1].copy()
    command(q_final)
    stream_wall = time.time() - wall0

    # Settle: hold the final target until the robot has stopped moving.
    still = 0
    prev = _measured(bot, torso_joints)
    deadline = time.time() + settle_timeout
    err = float("nan")
    while time.time() < deadline:
        command(q_final)
        rate.sleep()
        cur = _measured(bot, torso_joints)
        if cur is None or prev is None:
            prev = cur
            continue
        err = float(np.max(np.abs(cur - q_final)))
        moving = float(np.max(np.abs(cur - prev)))
        prev = cur
        still = still + 1 if moving < 5e-4 else 0
        if still >= settle_still_ticks and (
            err <= settle_tol or still >= 2 * settle_still_ticks
        ):
            break
    logger.info(
        f"Trajectory streamed in {stream_wall:.1f}s (nominal {duration:.1f}s, "
        f"max lag {max_lag_seen:.3f} rad, slowed {100.0 * slowed_ticks * dt / max(stream_wall, 1e-3):.0f}% of the time); "
        f"settled: residual {err:.4f} rad, still for {still} ticks."
    )
    if still < settle_still_ticks:
        logger.warning(
            f"Robot was still moving after {settle_timeout:.0f}s of settling "
            f"(residual {err:.3f} rad). Holding final target anyway."
        )
