# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Shared helpers for the lotte_chilsung Pink-IK demos (arms only).

Thin layer over ``ik_pink.py``: same URDF, tasks and costs as
``ik_pink.build_ik_context`` but without camera / head / confirmation prompt.
Targets are ``L/R_gripper_base`` expressed in the ``arm_center`` frame, so the
torso is outside the IK and is never commanded. Head is never commanded.

Typical use::

    run_cartesian_move(lambda cfg: my_targets(cfg, ...), duration=2.0,
                       step_m=0.01, leave_held=True, label="lift")
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

import numpy as np
import pinocchio as pin
import pink
import qpsolvers
from loguru import logger
from pink.tasks import PostureTask, RelativeFrameTask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from ik_pink import solve_ik_for_waypoint  # noqa: E402
from dexcontrol.exceptions import ServiceUnavailableError  # noqa: E402
from dexcontrol.robot import Robot  # noqa: E402

# dexcontrol installs a rich RichHandler on import (a 4-column table per line). In a narrow
# terminal the message column collapses to one character per row, and because rich emits hard
# newlines + padding the terminal cannot reflow it on resize. Replace it with a plain sink so
# each record is one unpadded line that the terminal soft-wraps (and re-flows when widened).
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:7}</level> | {message}",
)

ROOT = "arm_center"
EE = {"left": "L_gripper_base", "right": "R_gripper_base"}
ARM_JOINTS = {
    "left": [f"L_arm_j{i}" for i in range(1, 8)],
    "right": [f"R_arm_j{i}" for i in range(1, 8)],
    "torso": [f"torso_j{i}" for i in range(1, 4)],
}
MAX_JOINT_JUMP = 0.35  # rad, between consecutive planned configs
# ik_pink's convergence test (sum of task errors < 1e-4) is strict; a waypoint that
# is within these tolerances is still accepted (same idea as SOFT_POS_TOL in
# lotte_chilsung/_bimanual_box.py).
SOFT_POS_TOL = 0.010  # m
SOFT_ROT_TOL = np.deg2rad(3.0)
# dexcontrol 0.5 pauses a component's state subscriber after DEFAULT_IDLE_TIMEOUT (5 s) without a read
# and resumes it with a data gap; the right arm / torso are only read at stage boundaries here, so a
# multi-second stream made get_joint_pos() raise ServiceUnavailableError (2026-09-23). We keep every
# subscriber alive for the whole run and retry the few reads that may still hit a gap.
IDLE_TIMEOUT_S = 3600.0
STATE_READ_TIMEOUT_S = 3.0

Targets = dict[str, np.ndarray]  # side -> target position in arm_center frame


# ---------------------------------------------------------------------------
# IK setup
# ---------------------------------------------------------------------------
def build_ik() -> tuple[pink.Configuration, dict, str, dict[str, list[int]]]:
    urdf = config.URDF_PATH
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf))))
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf, package_dirs=[os.path.dirname(urdf), pkg_root], root_joint=None
    )
    model = robot.model
    configuration = pink.Configuration(model, robot.data, pin.neutral(model))
    tasks = {
        "left": RelativeFrameTask(
            EE["left"], root=ROOT,
            position_cost=config.LEFT_EE_POSITION_COST,
            orientation_cost=config.LEFT_EE_ORIENTATION_COST,
        ),
        "right": RelativeFrameTask(
            EE["right"], root=ROOT,
            position_cost=config.RIGHT_EE_POSITION_COST,
            orientation_cost=config.RIGHT_EE_ORIENTATION_COST,
        ),
        "posture": PostureTask(cost=config.POSTURE_COST),
    }
    solver = qpsolvers.available_solvers[0]
    if config.PREFERRED_QP_SOLVER in qpsolvers.available_solvers:
        solver = config.PREFERRED_QP_SOLVER
    idx = {
        side: [model.idx_qs[model.getJointId(n)] for n in names]
        for side, names in ARM_JOINTS.items()
    }
    logger.info(f"Pink IK ready ({model.nq} DOF, QP solver: {solver})")
    return configuration, tasks, solver, idx


def _keep_subscriptions_alive(bot: Robot) -> Robot:
    try:
        bot.set_idle_timeout(IDLE_TIMEOUT_S)
    except Exception as exc:  # older dexcontrol without the idle monitor
        logger.debug(f"set_idle_timeout unavailable: {exc}")
    return bot


def connect() -> Robot:
    """Connect without camera/sensor init (same as lotte_chilsung demos); state subscribers never idle-pause."""
    return _keep_subscriptions_alive(Robot(auto_shutdown=False, init_sensors=False))


def connect_camera_robot() -> Robot:
    """Connect with the head camera enabled over zenoh (ik_pink.build_ik_context style)."""
    from dexcontrol.core.config import get_robot_config

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    return _keep_subscriptions_alive(Robot(configs=configs, auto_shutdown=False))


def read_joint_pos(component, name: str, timeout: float = STATE_READ_TIMEOUT_S) -> np.ndarray:
    """``component.get_joint_pos()`` with retries while the state subscriber has no data (resume gap)."""
    deadline = time.time() + timeout
    while True:
        try:
            return np.asarray(component.get_joint_pos(), dtype=float)
        except ServiceUnavailableError:
            if time.time() >= deadline:
                logger.error(f"No {name} joint state for {timeout:.0f}s (subscriber gap or the sim stopped publishing).")
                raise
            time.sleep(0.05)


def sync_from_robot(bot: Robot, configuration: pink.Configuration, idx: dict) -> None:
    """Copy live arm/torso joints into the Pink configuration."""
    model = configuration.model
    q = configuration.q.copy()
    q[idx["left"]] = read_joint_pos(bot.left_arm, "left_arm")
    q[idx["right"]] = read_joint_pos(bot.right_arm, "right_arm")
    q[idx["torso"]] = read_joint_pos(bot.torso, "torso")
    configuration.update(np.clip(q, model.lowerPositionLimit, model.upperPositionLimit))


def ee_poses(configuration: pink.Configuration) -> dict[str, pin.SE3]:
    """Current gripper poses in the arm_center frame."""
    return {s: configuration.get_transform(f, ROOT) for s, f in EE.items()}


def world_up_in_arm_center(configuration: pink.Configuration) -> np.ndarray:
    """Unit vector of base-frame +Z (world up) expressed in arm_center.

    arm_center pitches with the torso (~37 deg forward in the box-hold pose),
    so arm_center +Z is *not* up; use this for lifting/lowering.
    """
    T = configuration.get_transform_frame_to_world(ROOT)
    return T.rotation.T @ np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Planning: straight-line Cartesian move with fixed orientations
# ---------------------------------------------------------------------------
PoseAt = Callable[[float], dict[str, pin.SE3]]  # a in (0, 1] -> side -> target SE3 (arm_center)


def plan_poses(
    configuration: pink.Configuration,
    tasks: dict,
    solver: str,
    pose_at: PoseAt,
    n: int,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Return (q_start, [q_1 .. q_n]) tracking ``pose_at(i/n)`` for both grippers.

    Generic rigid-pose tracker: each step is solved with
    ``ik_pink.solve_ik_for_waypoint`` from the previous solution. Stops early
    (keeps the prefix) if IK does not converge or a joint jump exceeds
    ``MAX_JOINT_JUMP``.
    """
    n = max(int(n), 1)
    q_start = configuration.q.copy()
    q_last = q_start
    knots: list[np.ndarray] = []
    for i in range(1, n + 1):
        tgt = pose_at(i / n)
        waypoint = {
            "type": "ik",
            "left_pose": {"position": tgt["left"].translation, "rpy": pin.rpy.matrixToRpy(tgt["left"].rotation)},
            "right_pose": {"position": tgt["right"].translation, "rpy": pin.rpy.matrixToRpy(tgt["right"].rotation)},
            "duration": 0.0,
        }
        q, ok = solve_ik_for_waypoint(
            configuration, waypoint, tasks["left"], tasks["right"], tasks["posture"], solver
        )
        if not ok:
            # Not converged to 1e-4; accept if the residual is small (soft tolerance).
            errs = [tasks[s].compute_error(configuration) for s in ("left", "right")]
            perr = max(float(np.linalg.norm(e[:3])) for e in errs)
            rerr = max(float(np.linalg.norm(e[3:])) for e in errs)
            ok = perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL
            detail = f"residual {perr * 1000:.1f} mm / {np.rad2deg(rerr):.1f} deg"
        else:
            detail = ""
        jump = float(np.max(np.abs(q - q_last)))
        if not ok or jump > MAX_JOINT_JUMP:
            reason = f"IK did not converge ({detail})" if not ok else f"joint jump {jump:.2f} rad"
            logger.warning(f"{reason} at {100 * i / n:.0f}% of the move; using prefix.")
            configuration.update(q_last)
            break
        knots.append(q.copy())
        q_last = q

    configuration.update(q_start)
    if not knots:
        logger.error("No progress; nothing planned.")
        return None
    logger.info(f"Planned {len(knots)}/{n} waypoints.")
    return q_start, knots


def plan_cartesian(
    configuration: pink.Configuration,
    tasks: dict,
    solver: str,
    targets: Targets,
    step_m: float = 0.01,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Straight-line move of each gripper to ``targets`` with orientation fixed."""
    cur = ee_poses(configuration)
    p = {s: T.translation.copy() for s, T in cur.items()}
    rot = {s: T.rotation.copy() for s, T in cur.items()}
    travel = max(float(np.linalg.norm(targets[s] - p[s])) for s in p)
    if travel < 1e-4:
        logger.info("Requested move is ~0; nothing to do.")
        return configuration.q.copy(), []
    n = max(int(np.ceil(travel / max(step_m, 1e-4))), 2)
    return plan_poses(
        configuration, tasks, solver,
        lambda a: {s: pin.SE3(rot[s], (1 - a) * p[s] + a * targets[s]) for s in p},
        n,
    )


def rigid_rotation_pose_at(
    poses: dict[str, pin.SE3], center: np.ndarray, axis: np.ndarray, angle: float
) -> PoseAt:
    """Both grippers rotated as one rigid body about ``axis`` through ``center``.

    ``poses`` are the current gripper poses (arm_center frame); the hand-to-hand
    transform is preserved for every ``a``, so a held object is never twisted.
    """
    axis = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-9)
    p0 = {s: T.translation.copy() for s, T in poses.items()}
    R0 = {s: T.rotation.copy() for s, T in poses.items()}

    def pose_at(a: float) -> dict[str, pin.SE3]:
        Rm = pin.exp3(axis * (a * angle))
        return {s: pin.SE3(Rm @ R0[s], center + Rm @ (p0[s] - center)) for s in p0}

    return pose_at


# ---------------------------------------------------------------------------
# Execution (ik_pink style: joint-space smoothstep, set_joint_pos loop)
# ---------------------------------------------------------------------------
def execute(bot: Robot, idx: dict, q_start: np.ndarray, knots: list[np.ndarray],
            duration: float, control_dt: float) -> None:
    Q = np.array([q_start] + knots)
    n_seg = len(Q) - 1
    n_steps = max(1, int(max(duration, 0.5) / control_dt))

    def command(q: np.ndarray) -> None:
        bot.set_joint_pos({"left_arm": q[idx["left"]], "right_arm": q[idx["right"]]})

    for step in range(n_steps):
        t = (step + 1) / n_steps
        s = t * t * (3 - 2 * t) * n_seg  # smoothstep over the whole path
        k = min(int(s), n_seg - 1)
        f = s - k
        command((1 - f) * Q[k] + f * Q[k + 1])
        time.sleep(control_dt)

    # Re-assert the final setpoint briefly so it is the last thing received.
    for _ in range(max(int(0.2 / control_dt), 5)):
        command(Q[-1])
        time.sleep(control_dt)


def disconnect_leave_held(bot: Robot) -> None:
    """Close the client without ``stop()`` so the last joint targets stay held."""
    if bot._shutdown_called:
        return
    bot._shutdown_called = True
    try:
        from dexcontrol.robot import _active_robots
        _active_robots.discard(bot)
    except Exception:
        pass
    if hasattr(bot, "_idle_monitor"):
        bot._idle_monitor.stop()
    sensors = getattr(bot, "sensors", None)
    if sensors is not None:
        try:
            sensors.shutdown()
        except Exception as exc:
            logger.warning(f"Sensor shutdown failed: {exc}")
    for component in reversed(getattr(bot, "_components", [])):
        if component is not None:
            try:
                component.shutdown()
            except Exception as exc:
                logger.warning(f"Component shutdown failed: {exc}")
    try:
        bot._release_shared_node()
    except Exception as exc:
        logger.warning(f"Shared node release failed: {exc}")


# ---------------------------------------------------------------------------
# One-shot driver used by the demo scripts
# ---------------------------------------------------------------------------
def run_pose_move(
    make_plan: Callable[[pink.Configuration], tuple[PoseAt, int] | None],
    *,
    duration: float,
    leave_held: bool = True,
    label: str = "move",
    pre_move: Callable[[Robot], None] | None = None,
) -> None:
    """Connect, sync, plan ``pose_at`` over ``n`` steps, stream it, leave held.

    ``make_plan`` receives the synced configuration and returns ``(pose_at, n)``
    (see ``plan_poses``) or None to abort. ``pre_move`` (optional) runs on the
    connected robot *before* the sync, e.g. a blocking torso move; the arm IK
    then starts from the post-move torso pose.
    """
    configuration, tasks, solver, idx = build_ik()
    bot = connect()
    do_shutdown = not leave_held
    try:
        if pre_move is not None:
            pre_move(bot)
        sync_from_robot(bot, configuration, idx)
        plan = make_plan(configuration)
        planned = plan_poses(configuration, tasks, solver, *plan) if plan else None
        if planned is None:
            do_shutdown = True
            return
        q_start, knots = planned
        if knots:
            logger.info(f"Executing {label} over {duration:.1f}s...")
            execute(bot, idx, q_start, knots, duration, config.CONTROL_DT)
            logger.info(f"{label} completed; final pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() — pose held.")
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        do_shutdown = True
    finally:
        if do_shutdown:
            logger.info("Shutting down robot")
            bot.shutdown()


def run_cartesian_move(
    make_targets: Callable[[pink.Configuration], Targets | None],
    *,
    duration: float,
    step_m: float = 0.01,
    leave_held: bool = True,
    label: str = "move",
    pre_move: Callable[[Robot], None] | None = None,
) -> None:
    """Straight-line variant of ``run_pose_move``: ``make_targets(cfg)`` returns
    target gripper positions (arm_center) with orientation kept, or None."""

    def make_plan(configuration: pink.Configuration):
        targets = make_targets(configuration)
        if not targets:
            return None
        cur = ee_poses(configuration)
        p = {s: T.translation.copy() for s, T in cur.items()}
        rot = {s: T.rotation.copy() for s, T in cur.items()}
        travel = max(float(np.linalg.norm(targets[s] - p[s])) for s in p)
        if travel < 1e-4:
            logger.info("Requested move is ~0; nothing to do.")
            return (lambda a: {s: pin.SE3(rot[s], p[s]) for s in p}), 0
        n = max(int(np.ceil(travel / max(step_m, 1e-4))), 2)
        return (lambda a: {s: pin.SE3(rot[s], (1 - a) * p[s] + a * targets[s]) for s in p}), n

    run_pose_move(make_plan, duration=duration, leave_held=leave_held, label=label, pre_move=pre_move)
