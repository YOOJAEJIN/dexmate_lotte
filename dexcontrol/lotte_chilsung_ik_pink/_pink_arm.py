# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Single-arm pose moves for either arm, executed as a sequence of stages over one connection.

Generalises ``_pink_right.py`` (steps 6-8, right arm only):

* ``side`` -- "left" or "right" is commanded; the other arm and the head are never commanded.
* stages -- ``run_stages`` connects once and runs several straight-line moves in turn (e.g. lift,
  traverse, descend, push). Each stage is planned from the state measured after the previous
  one, with the same setpoint anchoring (``/state/target/...``), sag-offset carry-over, tracking
  abort and measured-state refinement as ``_pink_right.run_point_move``.
* orientation -- a stage may also change the gripper orientation: translation is interpolated
  linearly and rotation along the geodesic (``exp3`` of the scaled ``log3``), so the gripper
  follows a straight line while turning smoothly.

IK: reduced model with only the commanded arm (+ torso in torso mode) free, absolute ``FrameTask``
on ``L/R_gripper_base`` -- the model root is the fixed ``base`` link, so "world" is the robot base
frame -- and a ``PostureTask`` whose per-joint cost is higher for the torso. Used by
``9.left_mirror_and_push.py``; ``_pink_right.py`` is unchanged.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pink
import pinocchio as pin
from loguru import logger
from pink.tasks import FrameTask, PostureTask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _pink_common import (  # noqa: E402
    ARM_JOINTS,
    EE,
    MAX_JOINT_JUMP,
    SOFT_POS_TOL,
    SOFT_ROT_TOL,
    build_ik,
    connect,
    disconnect_leave_held,
    sync_from_robot,
)

# ARM_JOINTS group ("left" / "right" / "torso") <-> dexcontrol component name
COMPONENT = {"left": "left_arm", "right": "right_arm", "torso": "torso"}
GROUP = {c: g for g, c in COMPONENT.items()}
POSITION_COST = {"left": config.LEFT_EE_POSITION_COST, "right": config.RIGHT_EE_POSITION_COST}
ORIENTATION_COST = {"left": config.LEFT_EE_ORIENTATION_COST, "right": config.RIGHT_EE_ORIENTATION_COST}
# Held joint targets re-published by the sim (zenoh_manager.target_publishers, 2026-09-20+) and the
# command topics we stream on (a foreign publisher there would silently override us).
SETPOINT_TOPICS = {
    "left_arm": "state/target/arm/left",
    "right_arm": "state/target/arm/right",
    "torso": "state/target/torso",
}
COMMAND_TOPICS = {"left_arm": "control/arm/left", "right_arm": "control/arm/right", "torso": "control/torso"}
TRACKING_ABORT_RAD = 0.10  # commanded joint lags its setpoint by more than this -> not following us
MAX_ANCHOR_OFFSET = 0.15  # rad; a larger setpoint-measurement gap is treated as stale data
TORSO_POSTURE_COST = 0.02
TORSO_MAX_VEL = 0.15
DEFAULT_STEP_RAD = np.deg2rad(3.0)
DEFAULT_REFINE_TOL_RAD = np.deg2rad(2.0)
# Streaming pace (see stream()): dexcontrol 0.5 rejects any set_joint_pos whose target differs from the
# *measured* joints by more than 0.5 rad (robot.py _guard_joint_step). On the sim the arm lags a fast
# schedule, so the schedule is slowed while the lag exceeds LEASH_RAD (as lotte_chilsung/_bimanual_box.
# execute_trajectory does) and every setpoint is clamped to within STEP_CLAMP_RAD of the measurement.
LEASH_RAD = 0.05
MIN_RATE = 0.3
STEP_CLAMP_RAD = 0.40
FINAL_HOLD_S = 3.0
FINAL_HOLD_LAG_RAD = 0.02

# spec(configuration) -> (point in the gripper frame, its goal in robot base,
#                          goal gripper rotation in robot base or None = hold the current one)
StageSpec = Callable[[pink.Configuration], tuple[np.ndarray, np.ndarray, np.ndarray | None]]


@dataclass
class Stage:
    """One straight-line move; ``spec`` is evaluated on the state measured just before it runs."""

    label: str
    spec: StageSpec
    duration: float = 2.0
    step_m: float = 0.01
    step_rad: float = DEFAULT_STEP_RAD
    refine: int = 1  # measured-state correction moves after the main move (0 = none)
    refine_tol_m: float = 0.005
    refine_tol_rad: float = DEFAULT_REFINE_TOL_RAD
    # Optional hooks (ignored in dry runs). ``before(bot, configuration)`` runs on the connected robot
    # right before the stage is planned (e.g. tare a force sensor); False aborts the run. ``stop_check(bot)``
    # is polled every streaming tick; True stops the motion where it is (measured pose held) -- used for
    # force-gated pushes. A stage that stopped early is not refined.
    before: Callable[[Any, pink.Configuration], bool] | None = None
    stop_check: Callable[[Any], bool] | None = None
    stop_label: str = "stop condition"
    # Optional IK posture bias: ``posture_joints(configuration)`` -> the 7 arm joint values the null space should
    # be pulled toward (None = track the current joints step by step, the default). Used by the mirror scripts
    # with the *other* arm's joints mirrored: that configuration is known to be reachable, so the planner follows
    # the same elbow branch instead of folding the elbow until the wrist hits its limit (2026-09-26, 4.left_mirror
    # lift: joint jump at 81% with L_arm_j6 at +1.396).
    posture_joints: Callable[[pink.Configuration], Any] | None = None


def rot_err(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (rad) between two rotation matrices."""
    return float(np.linalg.norm(pin.log3(R_a.T @ R_b)))


# ---------------------------------------------------------------------------
# Probes (own short-lived dexcomm node, run *before* the Robot client exists)
# ---------------------------------------------------------------------------
def read_setpoints(components: list[str], timeout: float = 2.0) -> dict[str, np.ndarray]:
    """Held joint targets for ``components`` from ``/state/target/...``; ``{}`` if unavailable.

    Under gravity the measured joints sit slightly below the controller's setpoint. Anchoring
    the stream at the true setpoint (and carrying that offset along the whole path) avoids the
    small "relax" step that commanding the measured pose would cause.
    """
    topics = {c: SETPOINT_TOPICS[c] for c in components}
    got: dict[str, np.ndarray] = {}
    try:
        from dexcomm import Node
        from dexcomm.codecs import JointStateCodec

        if not os.environ.get("ROBOT_NAME") and not os.environ.get("ROBOT_CONFIG"):
            raise RuntimeError("ROBOT_NAME is not set")
        node = Node(name="lotte_setpoint_probe")
        try:
            def make_cb(part: str):
                def _cb(msg):
                    if isinstance(msg, dict) and msg.get("pos") is not None:
                        got[part] = np.asarray(msg["pos"], dtype=float)
                return _cb

            subs = [
                node.create_subscriber(topic, make_cb(part), decoder=JointStateCodec.decode)
                for part, topic in topics.items()
            ]
            t0 = time.time()
            while time.time() - t0 < timeout and len(got) < len(topics):
                time.sleep(0.05)
            del subs
        finally:
            try:
                node.shutdown()
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"Held setpoints unavailable ({exc}).")
    if not got:
        logger.warning("No /state/target/* from the sim (old sim_main?); the first command anchors at the "
                       "measured pose, so a small sag step is possible.")
    return got


def detect_foreign_commands(components: list[str], window_s: float = 0.7) -> dict[str, int]:
    """Count command messages on our topics (from other processes) during ``window_s`` seconds."""
    topics = {c: COMMAND_TOPICS[c] for c in components}
    counts: dict[str, int] = {c: 0 for c in topics}
    try:
        from dexcomm import Node
        from dexcomm.codecs import JointCmdCodec

        if not os.environ.get("ROBOT_NAME") and not os.environ.get("ROBOT_CONFIG"):
            raise RuntimeError("ROBOT_NAME is not set")
        node = Node(name="lotte_command_probe")
        try:
            def make_cb(part: str):
                def _cb(_msg):
                    counts[part] += 1
                return _cb

            subs = [
                node.create_subscriber(topic, make_cb(part), decoder=JointCmdCodec.decode)
                for part, topic in topics.items()
            ]
            time.sleep(max(window_s, 0.1))
            del subs
        finally:
            try:
                node.shutdown()
            except Exception:
                pass
    except Exception as exc:
        logger.warning(f"Foreign-command check skipped ({exc}).")
    return counts


# ---------------------------------------------------------------------------
# One arm (+ torso) IK on a reduced model, absolute (robot base) frame task
# ---------------------------------------------------------------------------
def build_arm_ik(
    configuration: pink.Configuration, side: str, use_torso: bool, torso_posture_cost: float
) -> tuple[pink.Configuration, dict, dict[str, list[int]]]:
    """Reduced model with only ``side``'s arm (and the torso if ``use_torso``) free.

    Everything else is locked at the synced pose. Returns ``(reduced_configuration, tasks, idx)``
    where ``idx`` maps the free groups ("left" / "right" / "torso") to q-indices of the reduced
    model. Tasks: absolute ``FrameTask`` on ``EE[side]`` (robot base frame) and a ``PostureTask``
    with a higher per-joint cost on the torso.
    """
    model = configuration.model
    groups = [side] + (["torso"] if use_torso else [])
    keep = {n for g in groups for n in ARM_JOINTS[g]}
    lock_ids = [jid for jid in range(1, model.njoints) if model.names[jid] not in keep]
    rmodel = pin.buildReducedModel(model, lock_ids, configuration.q)
    q = pin.neutral(rmodel)
    for jid in range(1, rmodel.njoints):
        fj = model.getJointId(rmodel.names[jid])
        q[rmodel.idx_qs[jid]: rmodel.idx_qs[jid] + rmodel.nqs[jid]] = configuration.q[
            model.idx_qs[fj]: model.idx_qs[fj] + model.nqs[fj]
        ]
    rcfg = pink.Configuration(rmodel, rmodel.createData(), q)
    idx = {g: [rmodel.idx_qs[rmodel.getJointId(n)] for n in ARM_JOINTS[g]] for g in groups}
    ee = FrameTask(EE[side], position_cost=POSITION_COST[side], orientation_cost=ORIENTATION_COST[side])
    cost = np.full(rmodel.nv, config.POSTURE_COST)
    if use_torso:
        for n in ARM_JOINTS["torso"]:
            cost[rmodel.idx_vs[rmodel.getJointId(n)]] = torso_posture_cost
    posture = PostureTask(cost=config.POSTURE_COST)
    posture.cost = cost  # per-joint weights (Pink builds np.diag(cost))
    posture.set_target(q.copy())
    logger.info(
        f"IK ready: {rmodel.nq} free DOF ({side} arm{' + torso' if use_torso else ''})"
        + (f", torso posture cost {torso_posture_cost} vs arm {config.POSTURE_COST}" if use_torso else "")
    )
    return rcfg, {"ee": ee, "posture": posture}, idx


def plan_se3_path(
    rcfg: pink.Configuration, tasks: dict, solver: str, T_goal: pin.SE3, step_m: float, step_rad: float,
    posture_target: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Straight line of the gripper origin to ``T_goal`` (robot base) with a geodesic rotation.

    Same stepping / soft-tolerance / joint-jump rules as ``_pink_right.plan_cartesian_torso``;
    the number of steps is set by the larger of the translation (``step_m``) and rotation
    (``step_rad``) requirements. ``posture_target`` (reduced-model q) fixes the posture task's target
    for the whole path instead of tracking the current joints, biasing the arm's null space toward a
    known-good configuration. Returns ``(q_start, knots)`` (``knots`` may be a prefix), or None if no
    progress at all.
    """
    frame = tasks["ee"].frame
    T0 = rcfg.get_transform_frame_to_world(frame)
    p0, R0 = T0.translation.copy(), T0.rotation.copy()
    dp = T_goal.translation - p0
    w = pin.log3(R0.T @ T_goal.rotation)  # rotation vector, start-gripper frame
    travel, angle = float(np.linalg.norm(dp)), float(np.linalg.norm(w))
    q_start = rcfg.q.copy()
    if travel < 1e-4 and angle < 1e-3:
        logger.info("Requested move is ~0; nothing to do.")
        return q_start, []
    n = max(int(np.ceil(travel / max(step_m, 1e-4))), int(np.ceil(angle / max(step_rad, 1e-3))), 2)
    q_last = q_start
    knots: list[np.ndarray] = []
    for i in range(1, n + 1):
        a = i / n
        tasks["ee"].set_target(pin.SE3(R0 @ pin.exp3(a * w), p0 + a * dp))
        tasks["posture"].set_target(rcfg.q.copy() if posture_target is None else posture_target)
        ok = False
        for _ in range(config.IK_MAX_ITERS):
            v = pink.solve_ik(rcfg, tasks.values(), config.IK_DT, solver=solver)
            rcfg.update(pin.integrate(rcfg.model, rcfg.q, v * config.IK_DT))
            if np.linalg.norm(tasks["ee"].compute_error(rcfg)) < config.IK_CONVERGENCE_THRESHOLD:
                ok = True
                break
        detail = ""
        if not ok:
            e = tasks["ee"].compute_error(rcfg)
            perr, rerr = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            ok = perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL
            detail = f"residual {perr * 1000:.1f} mm / {np.rad2deg(rerr):.1f} deg"
        q = rcfg.q.copy()
        jump = float(np.max(np.abs(q - q_last)))
        if not ok or jump > MAX_JOINT_JUMP:
            reason = f"IK did not converge ({detail})" if not ok else f"joint jump {jump:.2f} rad"
            logger.warning(f"{reason} at {100 * i / n:.0f}% of the move; using prefix.")
            rcfg.update(q_last)
            break
        knots.append(q)
        q_last = q
    rcfg.update(q_start)
    if not knots:
        logger.error("No progress; nothing planned.")
        return None
    logger.info(f"Planned {len(knots)}/{n} waypoints ({travel * 100:.1f} cm, {np.rad2deg(angle):.1f} deg"
                + (", posture biased" if posture_target is not None else "") + ").")
    return q_start, knots


def apply_reduced(configuration: pink.Configuration, idx_full: dict, q_r: np.ndarray, idx_r: dict) -> None:
    """Write a reduced-model joint vector back into the full configuration (dry runs)."""
    q = configuration.q.copy()
    for g, ii in idx_r.items():
        q[idx_full[g]] = q_r[ii]
    configuration.update(q)


# ---------------------------------------------------------------------------
# Two-arm (+ torso) IK in the robot base frame (used by 5.rotate_vertical.py --use-torso)
# ---------------------------------------------------------------------------
DualPoseAt = Callable[[float], dict[str, pin.SE3]]  # a in [0, 1] -> {"left": SE3, "right": SE3} (robot base)


def build_dual_arm_ik(
    configuration: pink.Configuration, use_torso: bool, torso_posture_cost: float
) -> tuple[pink.Configuration, dict, dict[str, list[int]]]:
    """Reduced model with both arms (and the torso if ``use_torso``) free, everything else locked.

    Like ``build_arm_ik`` but with absolute ``FrameTask``s on *both* grippers (robot base frame), so a
    torso motion is compensated by the arms and can extend the reach of a two-handed move. Returns
    ``(reduced_configuration, tasks, idx)`` with ``tasks`` = {"left", "right", "posture"} and ``idx``
    mapping "left" / "right" / "torso" to q-indices of the reduced model.
    """
    model = configuration.model
    groups = ["left", "right"] + (["torso"] if use_torso else [])
    keep = {n for g in groups for n in ARM_JOINTS[g]}
    lock_ids = [jid for jid in range(1, model.njoints) if model.names[jid] not in keep]
    rmodel = pin.buildReducedModel(model, lock_ids, configuration.q)
    q = pin.neutral(rmodel)
    for jid in range(1, rmodel.njoints):
        fj = model.getJointId(rmodel.names[jid])
        q[rmodel.idx_qs[jid]: rmodel.idx_qs[jid] + rmodel.nqs[jid]] = configuration.q[
            model.idx_qs[fj]: model.idx_qs[fj] + model.nqs[fj]
        ]
    rcfg = pink.Configuration(rmodel, rmodel.createData(), q)
    idx = {g: [rmodel.idx_qs[rmodel.getJointId(n)] for n in ARM_JOINTS[g]] for g in groups}
    tasks: dict = {
        s: FrameTask(EE[s], position_cost=POSITION_COST[s], orientation_cost=ORIENTATION_COST[s])
        for s in ("left", "right")
    }
    cost = np.full(rmodel.nv, config.POSTURE_COST)
    if use_torso:
        for n in ARM_JOINTS["torso"]:
            cost[rmodel.idx_vs[rmodel.getJointId(n)]] = torso_posture_cost
    posture = PostureTask(cost=config.POSTURE_COST)
    posture.cost = cost
    posture.set_target(q.copy())
    tasks["posture"] = posture
    logger.info(
        f"IK ready: {rmodel.nq} free DOF (both arms{' + torso' if use_torso else ''})"
        + (f", torso posture cost {torso_posture_cost} vs arm {config.POSTURE_COST}" if use_torso else "")
    )
    return rcfg, tasks, idx


def plan_dual_pose_path(
    rcfg: pink.Configuration, tasks: dict, solver: str, pose_at: DualPoseAt, n: int
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Track ``pose_at(i / n)`` for both grippers (robot base frame) from the current reduced state.

    Same stepping / soft-tolerance / joint-jump rules as ``plan_se3_path``; the largest reachable
    prefix is kept. Returns ``(q_start, knots)`` or None if no progress at all.
    """
    n = max(int(n), 1)
    q_start = rcfg.q.copy()
    q_last = q_start
    knots: list[np.ndarray] = []
    for i in range(1, n + 1):
        target = pose_at(i / n)
        for s in ("left", "right"):
            tasks[s].set_target(target[s])
        tasks["posture"].set_target(rcfg.q.copy())
        ok = False
        for _ in range(config.IK_MAX_ITERS):
            v = pink.solve_ik(rcfg, tasks.values(), config.IK_DT, solver=solver)
            rcfg.update(pin.integrate(rcfg.model, rcfg.q, v * config.IK_DT))
            if max(np.linalg.norm(tasks[s].compute_error(rcfg)) for s in ("left", "right")) < config.IK_CONVERGENCE_THRESHOLD:
                ok = True
                break
        detail = ""
        if not ok:
            errs = [tasks[s].compute_error(rcfg) for s in ("left", "right")]
            perr = max(float(np.linalg.norm(e[:3])) for e in errs)
            rerr = max(float(np.linalg.norm(e[3:])) for e in errs)
            ok = perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL
            detail = f"residual {perr * 1000:.1f} mm / {np.rad2deg(rerr):.1f} deg"
        q = rcfg.q.copy()
        jump = float(np.max(np.abs(q - q_last)))
        if not ok or jump > MAX_JOINT_JUMP:
            reason = f"IK did not converge ({detail})" if not ok else f"joint jump {jump:.2f} rad"
            logger.warning(f"{reason} at {100 * i / n:.0f}% of the move; using prefix.")
            rcfg.update(q_last)
            break
        knots.append(q)
        q_last = q
    rcfg.update(q_start)
    if not knots:
        logger.error("No progress; nothing planned.")
        return None
    logger.info(f"Planned {len(knots)}/{n} waypoints.")
    return q_start, knots


# ---------------------------------------------------------------------------
# Streaming (commanded components only)
# ---------------------------------------------------------------------------
def _measured(bot, components) -> dict[str, np.ndarray]:
    """Current measured joints per dexcontrol component (``bot.left_arm`` etc.)."""
    return {c: np.asarray(getattr(bot, c).get_joint_pos(), dtype=float) for c in components}


def stream(
    bot, groups: dict[str, np.ndarray], duration: float, control_dt: float, *,
    leash_rad: float = LEASH_RAD, min_rate: float = MIN_RATE, step_clamp_rad: float = STEP_CLAMP_RAD,
    final_hold_s: float = FINAL_HOLD_S, stop_check: Callable[[Any], bool] | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """Stream setpoints for the given components together with a tracking-aware min-jerk profile.

    ``groups`` maps a dexcontrol component to an array of shape (n_knots + 1, n_joints) whose
    first row is the anchor setpoint. Only these components are commanded.

    Pacing: a quintic (minimum-jerk) profile over ``duration`` of *virtual* time; while the
    measured joints lag the command by more than ``leash_rad`` the virtual clock slows down
    proportionally (never below ``min_rate``), so the target never runs away from the robot.
    Every setpoint sent is clamped to within ``step_clamp_rad`` of the measurement, which keeps
    it under dexcontrol's 0.5 rad single-command guard even if the robot stalls. After the
    profile the final knot is re-sent until the lag is below ``FINAL_HOLD_LAG_RAD`` (or
    ``final_hold_s``). If ``stop_check(bot)`` returns True on a tick, the motion stops right there:
    the *measured* pose becomes the held setpoint (so a PD controller does not keep pushing).
    Returns ``(final setpoint per component, stopped_early)``.
    """
    components = list(groups)
    n_seg = max(len(next(iter(groups.values()))) - 1, 1)
    duration = max(float(duration), 0.5)
    min_rate = float(np.clip(min_rate, 0.05, 1.0))

    def q_at(sv: float) -> dict[str, np.ndarray]:
        k = min(int(sv), n_seg - 1)
        f = sv - k
        return {g: (1 - f) * Q[k] + f * Q[k + 1] for g, Q in groups.items()}

    def send(cmd: dict[str, np.ndarray], meas: dict[str, np.ndarray]) -> None:
        bot.set_joint_pos({g: meas[g] + np.clip(q - meas[g], -step_clamp_rad, step_clamp_rad) for g, q in cmd.items()})

    def lag_of(cmd: dict[str, np.ndarray], meas: dict[str, np.ndarray]) -> float:
        return max(float(np.abs(cmd[g] - meas[g]).max()) for g in components)

    tau = 0.0  # virtual time along the profile (slowed while lagging)
    wall0 = time.time()
    hard_deadline = wall0 + duration / min_rate + 5.0
    cmd = q_at(0.0)
    max_lag = 0.0
    slowed = 0
    ticks = 0
    stopped = False
    while tau < duration:
        meas = _measured(bot, components)
        if stop_check is not None and stop_check(bot):
            stopped = True
            break
        lag = lag_of(cmd, meas)
        max_lag = max(max_lag, lag)
        factor = 1.0
        if lag > leash_rad and time.time() < hard_deadline:
            factor = max(min_rate, leash_rad / lag)
            slowed += 1
        tau += control_dt * factor
        ticks += 1
        t = min(tau / duration, 1.0)
        cmd = q_at(t * t * t * (10.0 - 15.0 * t + 6.0 * t * t) * n_seg)  # min-jerk
        send(cmd, meas)
        time.sleep(control_dt)
    stream_wall = time.time() - wall0

    # Hold the final setpoint until the robot has (nearly) reached it. After an early stop the
    # measured pose is held instead, so nothing keeps pushing.
    final = {g: meas[g].copy() for g in components} if stopped else q_at(float(n_seg))
    deadline = time.time() + max(final_hold_s, 5 * control_dt)
    lag = float("nan")
    for i in range(10**6):
        meas = _measured(bot, components)
        send(final, meas)
        time.sleep(control_dt)
        lag = lag_of(final, meas)
        if (i >= 4 and lag < FINAL_HOLD_LAG_RAD) or time.time() > deadline:
            break
    logger.info(
        f"Streamed in {stream_wall:.1f}s (nominal {duration:.1f}s, max lag {np.rad2deg(max_lag):.1f} deg, "
        f"slowed {100.0 * slowed / max(ticks, 1):.0f}% of the ticks)"
        + (f"; STOPPED EARLY at {100.0 * min(tau / duration, 1.0):.0f}% of the profile, measured pose held" if stopped else "")
        + f"; final lag {np.rad2deg(lag):.1f} deg."
    )
    return final if stopped else {g: Q[-1].copy() for g, Q in groups.items()}, stopped


def anchor_offsets(
    configuration: pink.Configuration, idx: dict, setpoints: dict[str, np.ndarray], components: list[str]
) -> dict[str, np.ndarray]:
    """Setpoint - measurement offset (gravity sag) per commanded component; zeros when unknown."""
    off: dict[str, np.ndarray] = {}
    for c in components:
        meas_c = configuration.q[idx[GROUP[c]]]
        sp = setpoints.get(c)
        if sp is not None and sp.shape == meas_c.shape and float(np.abs(sp - meas_c).max()) < MAX_ANCHOR_OFFSET:
            off[c] = sp - meas_c
        else:
            off[c] = np.zeros_like(meas_c)
            if sp is not None:
                logger.warning(f"{c}: held setpoint is > {MAX_ANCHOR_OFFSET} rad from the measurement; ignoring it.")
    logger.info("Anchor offsets setpoint-measured (deg): "
                + ", ".join(f"{c} {np.round(np.rad2deg(off[c]), 2)}" for c in components))
    return off


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_stages(
    side: str,
    stages: list[Stage],
    *,
    use_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    max_joint_vel: float = 0.5,
    settle_s: float = 0.7,
    anchor_setpoints: bool = True,
    auto_torso: bool = False,
    ignore_foreign_commands: bool = False,
    dry_run: bool = False,
    leave_held: bool = True,
    prepare: Callable[[pink.Configuration], bool] | None = None,
    pause: Callable[[str], bool] | None = None,
) -> bool:
    """Connect once, sync, then plan and stream every stage in turn; leave the final pose held.

    ``prepare(configuration)`` (optional) runs on the synced state before the first stage, e.g.
    to compute goals shared by several stages and check them; returning False aborts without
    moving. ``pause(next_label)`` (optional) is called between stages; returning False stops.

    A stage's ``spec`` is evaluated on the measured state right before it runs. Its goal is
    fixed for that stage; corrections re-plan the remaining error from the measured state (up to
    ``refine`` times, while it exceeds the tolerances). With ``auto_torso`` an arm-only plan that
    covers less than half of a stage is re-planned with the torso included -- note that moving
    the torso also moves the *other* hand, so this is off by default here.

    ``max_joint_vel`` (rad/s) stretches a stage's stream time so that no arm joint has to move
    faster than this along the planned path (the leash in ``stream`` then only fine-tunes).

    ``dry_run``: no robot; each stage is planned from the model (URDF neutral pose unless
    ``prepare`` changes it) and the model is advanced to the planned end so the next stage is
    planned from there. Returns True if every stage completed (or was empty).
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    configuration, _tasks, solver, idx = build_ik()
    ee_frame = EE[side]
    state = {"use_torso": use_torso, "components": [COMPONENT[side]] + (["torso"] if use_torso else [])}

    if not dry_run:
        probe = list(dict.fromkeys(state["components"] + ["torso"]))
        foreign = detect_foreign_commands(probe)
        busy = {c: n for c, n in foreign.items() if n > 0}
        blocking = {c: n for c, n in busy.items() if c in state["components"]}
        if blocking and not ignore_foreign_commands:
            logger.error(
                "Another controller is streaming joint targets we need to command: "
                + ", ".join(f"{COMMAND_TOPICS[c]} ({n} msgs in 0.7 s)" for c, n in blocking.items())
                + ". The sim keeps the latest command, so our motion would be overridden. Stop the omniteleop "
                "follower (Ctrl+C in its launch.sh terminal, which also runs `pkill -f omniteleop.follower`) and "
                "retry; --ignore-foreign-commands overrides this check."
            )
            return False
        if "torso" in busy and "torso" not in state["components"] and auto_torso:
            logger.warning(f"control/torso is driven by another process ({busy['torso']} msgs in 0.7 s); "
                           "the automatic torso fallback is disabled for this run (arm only).")
            auto_torso = False
        if blocking:
            logger.warning(f"Foreign commands detected but ignored on request: {blocking}")
    setpoints = (
        read_setpoints(list(dict.fromkeys(state["components"] + (["torso"] if auto_torso else []))))
        if (anchor_setpoints and not dry_run) else {}
    )

    def plan_stage(stage: Stage, point_ee: np.ndarray, goal_p: np.ndarray, goal_R: np.ndarray, with_torso: bool):
        """Plan from the current configuration; returns (rcfg, q_start, knots, idx_r, covered) or None."""
        rcfg, tasks, idx_r = build_arm_ik(configuration, side, with_torso, torso_posture_cost)
        T_goal = pin.SE3(goal_R, goal_p - goal_R @ point_ee)  # gripper origin goal
        posture_target = None
        if stage.posture_joints is not None:
            pj = stage.posture_joints(configuration)
            if pj is not None:
                posture_target = rcfg.q.copy()
                posture_target[idx_r[side]] = np.asarray(pj, dtype=float).reshape(-1)
        planned = plan_se3_path(rcfg, tasks, solver, T_goal, stage.step_m, stage.step_rad, posture_target=posture_target)
        if planned is None:
            return None
        q_start, knots = planned
        if not knots:
            return rcfg, q_start, knots, idx_r, 1.0
        rcfg.update(knots[-1])
        T_end = rcfg.get_transform_frame_to_world(ee_frame)
        p_end = T_end.act(point_ee)
        r_end = rot_err(T_end.rotation, goal_R)
        if with_torso:
            d_torso = np.rad2deg(knots[-1][idx_r["torso"]] - q_start[idx_r["torso"]])
            logger.info(f"Torso change planned (j1, j2, j3): {np.round(d_torso, 1)} deg; other arm joints held.")
        rcfg.update(q_start)
        p0 = rcfg.get_transform_frame_to_world(ee_frame).act(point_ee)
        err_plan = p_end - goal_p
        remaining = float(np.linalg.norm(goal_p - p0))
        covered = 1.0 if remaining < 1e-6 else float(np.clip(1.0 - np.linalg.norm(err_plan) / remaining, 0.0, 1.0))
        (logger.info if (np.linalg.norm(err_plan) < 0.01 and r_end < stage.refine_tol_rad) else logger.warning)(
            f"{stage.label}: planned end: point {np.round(p_end, 4)}, residual {np.round(err_plan, 4)} "
            f"|e| = {np.linalg.norm(err_plan) * 1000:.1f} mm, {np.rad2deg(r_end):.1f} deg ({100 * covered:.0f}% of the move)"
        )
        return rcfg, q_start, knots, idx_r, covered

    def run_stage(stage: Stage, bot, off: dict[str, np.ndarray]) -> tuple[bool, dict[str, np.ndarray]]:
        if stage.before is not None and not dry_run:
            if not stage.before(bot, configuration):
                logger.error(f"{stage.label}: pre-stage check failed; stopping here (pose left as is).")
                return False, off
        point_ee, goal_p, goal_R = stage.spec(configuration)
        point_ee = np.asarray(point_ee, dtype=float).reshape(3)
        goal_p = np.asarray(goal_p, dtype=float).reshape(3)
        T_now = configuration.get_transform_frame_to_world(ee_frame)
        goal_R = T_now.rotation.copy() if goal_R is None else np.asarray(goal_R, dtype=float).reshape(3, 3)
        p_now = T_now.act(point_ee)
        travel0 = float(np.linalg.norm(goal_p - p_now))
        ang0 = rot_err(T_now.rotation, goal_R)
        logger.info(
            f"{stage.label}: point now {np.round(p_now, 4)} -> goal {np.round(goal_p, 4)} (robot base); "
            f"delta {np.round(goal_p - p_now, 4)} |d| = {travel0:.3f} m, rotation {np.rad2deg(ang0):.1f} deg"
        )
        for attempt in range(1 + max(int(stage.refine), 0)):
            result = plan_stage(stage, point_ee, goal_p, goal_R, state["use_torso"])
            if not state["use_torso"] and auto_torso and (result is None or result[4] < 0.5):
                why = "no arm-only plan" if result is None else f"arm alone covers only {100 * result[4]:.0f}%"
                logger.warning(f"{why}; retrying with the torso included (this also moves the other hand!).")
                state["use_torso"] = True
                state["components"] = [COMPONENT[side], "torso"]
                off = anchor_offsets(configuration, idx, setpoints, state["components"])
                result = plan_stage(stage, point_ee, goal_p, goal_R, True)
            if result is None:
                logger.error(f"{stage.label}: nothing planned; the robot was not commanded.")
                return False, off
            rcfg, q_start, knots, idx_r, _covered = result
            if not knots:
                break
            if dry_run:
                apply_reduced(configuration, idx, knots[-1], idx_r)
                T_end = configuration.get_transform_frame_to_world(ee_frame)
                logger.info(f"{stage.label}: dry run: model advanced to the planned end "
                            f"(gripper {np.round(T_end.translation, 4)}); nothing sent.")
                return True, off

            components = state["components"]
            groups = {c: np.array([q[idx_r[GROUP[c]]] + off[c] for q in [q_start] + knots]) for c in components}
            err_now = float(np.linalg.norm(goal_p - configuration.get_transform_frame_to_world(ee_frame).act(point_ee)))
            dur = stage.duration if attempt == 0 else max(1.0, stage.duration * min(1.0, err_now / max(travel0, 1e-6)))
            if state["use_torso"]:
                torso_path = float(np.abs(np.diff(groups["torso"], axis=0)).sum(axis=0).max())
                dur = max(dur, torso_path / max(torso_max_vel, 1e-3))
                logger.info(f"Torso path {np.rad2deg(torso_path):.1f} deg, speed cap {torso_max_vel:.2f} rad/s")
            joint_path = float(np.abs(np.diff(groups[COMPONENT[side]], axis=0)).sum(axis=0).max())
            if max_joint_vel > 0 and joint_path / max_joint_vel > dur:
                logger.info(f"{stage.label}: arm joint path {np.rad2deg(joint_path):.0f} deg -> stream time "
                            f"{dur:.1f}s stretched to {joint_path / max_joint_vel:.1f}s (cap {max_joint_vel:.2f} rad/s)")
                dur = joint_path / max_joint_vel
            logger.info(f"{stage.label}: {'executing' if attempt == 0 else f'correction {attempt}:'} "
                        f"{' + '.join(components)} over {dur:.1f}s...")
            try:
                last_cmd, stopped = stream(bot, groups, dur, config.CONTROL_DT, stop_check=stage.stop_check)
            except Exception as exc:
                logger.error(f"{stage.label}: streaming aborted ({exc}). Stopping here; the pose is left as is.")
                return False, off

            # Verify with measured joints; refresh the sag offsets from what was actually sent.
            time.sleep(max(settle_s, 0.0))
            sync_from_robot(bot, configuration, idx)
            track = {c: last_cmd[c] - configuration.q[idx[GROUP[c]]] for c in components}
            worst = max(float(np.abs(v).max()) for v in track.values())
            if worst > TRACKING_ABORT_RAD:
                logger.error(
                    f"Robot did not follow the commanded setpoints (max lag {np.rad2deg(worst):.1f} deg > "
                    f"{np.rad2deg(TRACKING_ABORT_RAD):.0f} deg): "
                    + ", ".join(f"{c} {np.round(np.rad2deg(track[c]), 1)}" for c in components)
                    + ". Another controller (omniteleop follower?) is probably overriding us. Stopping here; "
                    "the pose is left as is."
                )
                return False, off
            off = {c: np.clip(track[c], -MAX_ANCHOR_OFFSET, MAX_ANCHOR_OFFSET) for c in components}
            T_meas = configuration.get_transform_frame_to_world(ee_frame)
            err = goal_p - T_meas.act(point_ee)
            r_err = rot_err(T_meas.rotation, goal_R)
            if stopped:
                travelled = float(np.linalg.norm(T_meas.act(point_ee) - p_now))
                logger.info(
                    f"{stage.label}: stopped by {stage.stop_label} after {travelled * 100:.1f} cm "
                    f"({travel0 * 100:.1f} cm planned); gripper origin {np.round(T_meas.translation, 4)}, "
                    f"remaining to the planned goal {np.linalg.norm(err) * 100:.1f} cm. Pose held; no correction."
                )
                break
            good = np.linalg.norm(err) < stage.refine_tol_m and r_err < stage.refine_tol_rad
            (logger.info if good else logger.warning)(
                f"{stage.label}: measured: point {np.round(T_meas.act(point_ee), 4)}, error {np.round(err, 4)} "
                f"|e| = {np.linalg.norm(err) * 1000:.1f} mm, {np.rad2deg(r_err):.1f} deg; gripper origin "
                f"{np.round(T_meas.translation, 4)}, EE +Z {np.round(T_meas.rotation[:, 2], 3)}; sag offsets (deg) "
                + ", ".join(f"{c} {np.round(np.rad2deg(off[c]), 2)}" for c in components)
            )
            if good:
                break
            if attempt < stage.refine:
                logger.info("Re-planning the remaining error from the measured state...")
        return True, off

    bot = None
    do_shutdown = False
    ok_all = False
    try:
        if not dry_run:
            bot = connect()
            do_shutdown = not leave_held
            sync_from_robot(bot, configuration, idx)
        if prepare is not None and not prepare(configuration):
            logger.error("Preparation failed; the robot was not commanded.")
            return False
        off = anchor_offsets(configuration, idx, setpoints, state["components"])
        ok_all = True
        for k, stage in enumerate(stages):
            ok, off = run_stage(stage, bot, off)
            if not ok:
                ok_all = False
                break
            if pause is not None and not dry_run and k + 1 < len(stages):
                if not pause(stages[k + 1].label):
                    logger.warning("Stopped by the user before the next stage; pose left held.")
                    ok_all = False
                    break
        logger.info(f"{side} arm stages {'finished' if ok_all else 'stopped'}; pose left held.")
        return ok_all
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        do_shutdown = bot is not None
        return False
    finally:
        if bot is not None:
            if do_shutdown:
                logger.info("Shutting down robot")
                bot.shutdown()
            elif leave_held:
                disconnect_leave_held(bot)
                logger.info("Client disconnected without stop() -- pose held.")
