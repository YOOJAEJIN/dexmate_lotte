# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Right-arm point moves shared by steps 6 (finger to cyan), 7 (push along EE +Z) and 8 (lift).

Builds on ``_pink_common`` (same URDF, tasks, costs) and adds:

* ``read_setpoints`` — the sim's *held* joint targets (``/state/target/...``); streams are
  anchored there instead of at the gravity-sagged measurement, so no "relax" step.
* ``build_torso_ik`` / ``plan_cartesian_torso`` — reduced model with only right arm + torso free,
  absolute (robot-base) FrameTask, torso posture cost above the arms'.
* ``stream`` — joint-space smoothstep streaming of ``right_arm`` (+ ``torso``) only. The left
  arm is never commanded.
* ``run_point_move`` — the driver: connect, sync, plan a straight Cartesian move that brings a
  point fixed in the gripper frame to a goal in the robot base frame (orientation held), stream
  it with the sag offset carried along, then measure and optionally refine from the measured
  state. The scripts only supply the point and the goal. If the arm alone cannot plan the
  move (e.g. fully extended along the push direction) the driver retries with the torso
  included (``auto_torso``), logging why.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

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
    ROOT,
    SOFT_POS_TOL,
    SOFT_ROT_TOL,
    build_ik,
    connect,
    disconnect_leave_held,
    ee_poses,
    plan_cartesian,
    sync_from_robot,
)

# Held joint targets re-published by the sim (zenoh_manager.target_publishers, 2026-09-20+).
SETPOINT_TOPICS = {"right_arm": "state/target/arm/right", "torso": "state/target/torso"}
# Command topics we stream on. If *someone else* (e.g. the omniteleop follower) is publishing here,
# the sim keeps their latest target and silently ignores ours, so we check before moving.
COMMAND_TOPICS = {"right_arm": "control/arm/right", "torso": "control/torso"}
# Abort when a commanded joint lags its setpoint by more than this after a move: the robot is not
# following us (foreign controller, e-stop, ...) and carrying the offset would run away.
TRACKING_ABORT_RAD = 0.10
MAX_ANCHOR_OFFSET = 0.15  # rad; a larger setpoint-measurement gap is treated as stale data
COMMANDED = {"right_arm": "right", "torso": "torso"}  # dexcontrol component -> ARM_JOINTS key
# --use-torso: posture cost on torso joints ([cost]/[rad]; arms keep config.POSTURE_COST = 1e-3),
# and the torso joint speed cap used to stretch the stream duration (as in 3.box_lift_up.py).
TORSO_POSTURE_COST = 0.02
TORSO_MAX_VEL = 0.15

# (point in the R_gripper_base frame, its goal in the robot base frame) from the synced configuration
PointSpec = Callable[[pink.Configuration], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Probes (own short-lived dexcomm node, run *before* the Robot client exists)
# ---------------------------------------------------------------------------
def read_setpoints(timeout: float = 2.0) -> dict[str, np.ndarray]:
    """Held joint targets per component from ``/state/target/...``; ``{}`` if unavailable.

    Under gravity the measured joints sit slightly below the controller's setpoint. Anchoring
    the stream at the true setpoint (and carrying that offset along the whole path) avoids the
    small "relax" step that commanding the measured pose would cause.
    """
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
                for part, topic in SETPOINT_TOPICS.items()
            ]
            t0 = time.time()
            while time.time() - t0 < timeout and len(got) < len(SETPOINT_TOPICS):
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


def detect_foreign_commands(window_s: float = 0.7) -> dict[str, int]:
    """Count command messages on our topics (from other processes) during ``window_s`` seconds.

    Returns ``{component: n_messages}``; a non-zero count means another controller — typically
    the omniteleop follower — is streaming setpoints and would override ours.
    """
    counts: dict[str, int] = {c: 0 for c in COMMAND_TOPICS}
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
                for part, topic in COMMAND_TOPICS.items()
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
# Right arm + torso IK on a reduced model, absolute (robot base) frame task
# ---------------------------------------------------------------------------
def build_torso_ik(
    configuration: pink.Configuration, torso_posture_cost: float
) -> tuple[pink.Configuration, dict, dict[str, list[int]]]:
    """Reduced model with only the right arm and torso free (all else locked at the synced pose).

    Returns ``(reduced_configuration, tasks, idx)`` where ``idx`` maps "right" / "torso" to
    q-indices of the reduced model. Tasks: absolute ``FrameTask`` on ``R_gripper_base`` (the
    model root is the fixed ``base`` link, so "world" is the robot base frame) and a
    ``PostureTask`` whose per-joint cost is higher for the torso.
    """
    model = configuration.model
    keep = set(ARM_JOINTS["right"] + ARM_JOINTS["torso"])
    lock_ids = [jid for jid in range(1, model.njoints) if model.names[jid] not in keep]
    rmodel = pin.buildReducedModel(model, lock_ids, configuration.q)
    q = pin.neutral(rmodel)
    for jid in range(1, rmodel.njoints):
        fj = model.getJointId(rmodel.names[jid])
        q[rmodel.idx_qs[jid]: rmodel.idx_qs[jid] + rmodel.nqs[jid]] = configuration.q[
            model.idx_qs[fj]: model.idx_qs[fj] + model.nqs[fj]
        ]
    rcfg = pink.Configuration(rmodel, rmodel.createData(), q)
    idx = {
        side: [rmodel.idx_qs[rmodel.getJointId(n)] for n in ARM_JOINTS[side]] for side in ("right", "torso")
    }
    right = FrameTask(
        EE["right"],
        position_cost=config.RIGHT_EE_POSITION_COST,
        orientation_cost=config.RIGHT_EE_ORIENTATION_COST,
    )
    cost = np.full(rmodel.nv, config.POSTURE_COST)
    for n in ARM_JOINTS["torso"]:
        cost[rmodel.idx_vs[rmodel.getJointId(n)]] = torso_posture_cost
    posture = PostureTask(cost=config.POSTURE_COST)
    posture.cost = cost  # per-joint weights (Pink builds np.diag(cost))
    posture.set_target(q.copy())
    logger.info(
        f"Torso IK ready: {rmodel.nq} free DOF (right arm + torso), torso posture cost "
        f"{torso_posture_cost} vs arm {config.POSTURE_COST}"
    )
    return rcfg, {"right": right, "posture": posture}, idx


def plan_cartesian_torso(
    rcfg: pink.Configuration, tasks: dict, solver: str, goal_gb_b: np.ndarray, step_m: float
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Straight line of ``R_gripper_base`` to ``goal_gb_b`` (robot base frame), orientation held.

    Same stepping / soft-tolerance / joint-jump rules as ``_pink_common.plan_poses``.
    """
    T0 = rcfg.get_transform_frame_to_world(EE["right"])
    p0, R0 = T0.translation.copy(), T0.rotation.copy()
    travel = float(np.linalg.norm(goal_gb_b - p0))
    q_start = rcfg.q.copy()
    if travel < 1e-4:
        logger.info("Requested move is ~0; nothing to do.")
        return q_start, []
    n = max(int(np.ceil(travel / max(step_m, 1e-4))), 2)
    q_last = q_start
    knots: list[np.ndarray] = []
    for i in range(1, n + 1):
        a = i / n
        tasks["right"].set_target(pin.SE3(R0, (1 - a) * p0 + a * goal_gb_b))
        tasks["posture"].set_target(rcfg.q.copy())
        ok = False
        for _ in range(config.IK_MAX_ITERS):
            v = pink.solve_ik(rcfg, tasks.values(), config.IK_DT, solver=solver)
            rcfg.update(pin.integrate(rcfg.model, rcfg.q, v * config.IK_DT))
            if np.linalg.norm(tasks["right"].compute_error(rcfg)) < config.IK_CONVERGENCE_THRESHOLD:
                ok = True
                break
        detail = ""
        if not ok:
            e = tasks["right"].compute_error(rcfg)
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
    logger.info(f"Planned {len(knots)}/{n} waypoints (arm + torso).")
    return q_start, knots


# ---------------------------------------------------------------------------
# Geometry helpers and planning entry point
# ---------------------------------------------------------------------------
def gripper_pose_b(configuration: pink.Configuration) -> pin.SE3:
    """Pose of ``R_gripper_base`` in the robot base frame."""
    return configuration.get_transform_frame_to_world(EE["right"])


def point_in_base(configuration: pink.Configuration, point_gb: np.ndarray) -> np.ndarray:
    """A point given in the gripper frame, expressed in the robot base frame."""
    return gripper_pose_b(configuration).act(point_gb)


def plan_gripper_to(
    configuration: pink.Configuration, tasks: dict, solver: str, goal_gb_b: np.ndarray,
    use_torso: bool, torso_posture_cost: float, step_m: float,
) -> tuple[pink.Configuration, tuple[np.ndarray, list[np.ndarray]] | None, dict[str, list[int]] | None]:
    """Plan ``R_gripper_base`` to ``goal_gb_b`` (base frame, orientation held) from the synced pose.

    Returns ``(fk configuration, plan, torso-mode idx or None)``; the fk configuration is the one
    the plan's joint vectors belong to (reduced model in torso mode).
    """
    if use_torso:
        rcfg, tasks_t, idx_t = build_torso_ik(configuration, torso_posture_cost)
        return rcfg, plan_cartesian_torso(rcfg, tasks_t, solver, goal_gb_b, step_m), idx_t
    T_ac = configuration.get_transform_frame_to_world(ROOT)  # base <- arm_center
    poses = ee_poses(configuration)
    targets = {"left": poses["left"].translation.copy(), "right": T_ac.actInv(goal_gb_b)}
    return configuration, plan_cartesian(configuration, tasks, solver, targets, step_m=step_m), None


# ---------------------------------------------------------------------------
# Streaming (right arm + optional torso only)
# ---------------------------------------------------------------------------
def stream(bot, groups: dict[str, np.ndarray], duration: float, control_dt: float) -> dict[str, np.ndarray]:
    """Stream setpoints for the given components together (smoothstep over the whole path).

    ``groups`` maps a dexcontrol component ("right_arm", "torso") to an array of shape
    (n_knots + 1, n_joints) whose first row is the anchor setpoint. Only these components are
    commanded (never the left arm). Returns the last setpoint sent per component.
    """
    n_seg = max(len(next(iter(groups.values()))) - 1, 1)
    n_steps = max(1, int(max(duration, 0.5) / control_dt))

    def command(sv: float) -> None:
        k = min(int(sv), n_seg - 1)
        f = sv - k
        bot.set_joint_pos({g: (1 - f) * Q[k] + f * Q[k + 1] for g, Q in groups.items()})

    for step in range(n_steps):
        t = (step + 1) / n_steps
        command(t * t * (3 - 2 * t) * n_seg)
        time.sleep(control_dt)
    for _ in range(max(int(0.2 / control_dt), 5)):  # re-assert the final setpoint
        command(float(n_seg))
        time.sleep(control_dt)
    return {g: Q[-1].copy() for g, Q in groups.items()}


def anchor_offsets(
    configuration: pink.Configuration, idx: dict, setpoints: dict[str, np.ndarray], components: list[str]
) -> dict[str, np.ndarray]:
    """Setpoint - measurement offset (gravity sag) per commanded component; zeros when unknown."""
    off: dict[str, np.ndarray] = {}
    for c in components:
        meas_c = configuration.q[idx[COMMANDED[c]]]
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
def run_point_move(
    spec: PointSpec,
    *,
    label: str,
    duration: float,
    step_m: float = 0.01,
    use_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    refine: int = 2,
    refine_tol_m: float = 0.005,
    settle_s: float = 0.7,
    anchor_setpoints: bool = True,
    auto_torso: bool = True,
    ignore_foreign_commands: bool = False,
    dry_run: bool = False,
    leave_held: bool = True,
) -> None:
    """Bring a gripper-fixed point to a base-frame goal with the gripper orientation held.

    ``spec(configuration)`` is called once after the sync and returns ``(point_gb, goal_b)``:
    the point in the ``R_gripper_base`` frame that should end up at ``goal_b`` (robot base
    frame). The goal is fixed for the whole run; corrections re-plan the remaining error from
    the measured state (up to ``refine`` times, while it exceeds ``refine_tol_m``).

    With ``auto_torso`` (default) an arm-only plan that covers less than half of the move —
    typically because the arm is already fully extended in that direction — is discarded and
    the move is re-planned with the torso included, exactly as ``use_torso=True`` would.
    """
    configuration, tasks, solver, idx = build_ik()
    components = ["right_arm"] + (["torso"] if use_torso else [])

    if not dry_run:
        foreign = detect_foreign_commands()
        busy = {c: n for c, n in foreign.items() if n > 0}
        blocking = {c: n for c, n in busy.items() if c in components}
        if blocking and not ignore_foreign_commands:
            hint = (
                "Run without --use-torso to move the arm only, or stop the omniteleop follower "
                "(Ctrl+C in its launch.sh terminal; it also runs `pkill -f omniteleop.follower`)."
                if "right_arm" not in blocking else
                "Stop the omniteleop follower (Ctrl+C in its launch.sh terminal, which also runs "
                "`pkill -f omniteleop.follower`) and retry."
            )
            logger.error(
                "Another controller is streaming joint targets we need to command: "
                + ", ".join(f"{COMMAND_TOPICS[c]} ({n} msgs in 0.7 s)" for c, n in blocking.items())
                + f". The sim keeps the latest command, so our motion would be overridden. {hint} "
                "--ignore-foreign-commands overrides this check."
            )
            return
        if "torso" in busy and "torso" not in components and auto_torso:
            logger.warning(
                f"control/torso is driven by another process ({busy['torso']} msgs in 0.7 s); the automatic "
                "torso fallback is disabled for this run (arm only)."
            )
            auto_torso = False
        if blocking:
            logger.warning(f"Foreign commands detected but ignored on request: {blocking}")
    setpoints = read_setpoints() if (anchor_setpoints and not dry_run) else {}

    def plan_and_report(with_torso: bool):
        """Plan from the current configuration; returns (fk_cfg, q_start, knots, idx_t, covered) or None."""
        T_gb_b = gripper_pose_b(configuration)
        goal_gb_b = goal_b - T_gb_b.rotation @ point_gb
        fk_cfg, planned, idx_t = plan_gripper_to(
            configuration, tasks, solver, goal_gb_b, with_torso, torso_posture_cost, step_m
        )
        if planned is None:
            return None
        q_start, knots = planned
        if not knots:
            return fk_cfg, q_start, knots, idx_t, 1.0
        fk_cfg.update(knots[-1])
        p_end = point_in_base(fk_cfg, point_gb)
        if idx_t is not None:
            d_torso = np.rad2deg(knots[-1][idx_t["torso"]] - q_start[idx_t["torso"]])
            logger.info(
                f"Torso change planned (j1, j2, j3): {np.round(d_torso, 1)} deg "
                f"{'(arm reaches without the torso)' if np.abs(d_torso).max() < 1.0 else ''}; left arm joints held."
            )
        fk_cfg.update(q_start)
        p0 = point_in_base(fk_cfg, point_gb)
        err_plan = p_end - goal_b
        remaining = float(np.linalg.norm(goal_b - p0))
        covered = 1.0 if remaining < 1e-6 else float(np.clip(1.0 - np.linalg.norm(err_plan) / remaining, 0.0, 1.0))
        (logger.info if np.linalg.norm(err_plan) < 0.01 else logger.warning)(
            f"Planned end: point {np.round(p_end, 4)}, residual to goal {np.round(err_plan, 4)}  "
            f"|e| = {np.linalg.norm(err_plan) * 1000:.1f} mm ({100 * covered:.0f}% of the move)"
        )
        return fk_cfg, q_start, knots, idx_t, covered

    bot = None
    do_shutdown = False
    try:
        if not dry_run:
            bot = connect()
            do_shutdown = not leave_held
            sync_from_robot(bot, configuration, idx)
        torso_start_deg = np.rad2deg(configuration.q[idx["torso"]])
        off = anchor_offsets(configuration, idx, setpoints, components)

        point_gb, goal_b = spec(configuration)
        point_gb = np.asarray(point_gb, dtype=float).reshape(3)
        goal_b = np.asarray(goal_b, dtype=float).reshape(3)
        p_now = point_in_base(configuration, point_gb)
        travel0 = float(np.linalg.norm(goal_b - p_now))
        logger.info(
            f"{label}: point now {np.round(p_now, 4)} -> goal {np.round(goal_b, 4)} (robot base); "
            f"delta {np.round(goal_b - p_now, 4)}  |d| = {travel0:.3f} m; torso now {np.round(torso_start_deg, 1)} deg"
        )

        for attempt in range(1 + max(int(refine), 0)):
            result = plan_and_report(use_torso)
            if not use_torso and auto_torso and (result is None or result[4] < 0.5):
                why = "no arm-only plan" if result is None else f"arm alone covers only {100 * result[4]:.0f}%"
                logger.warning(
                    f"{why} (arm probably fully extended in that direction); retrying with the torso "
                    f"included (--no-auto-torso to disable)."
                )
                use_torso = True
                components = ["right_arm", "torso"]
                off = anchor_offsets(configuration, idx, setpoints, components)
                result = plan_and_report(True)
            if result is None:
                logger.error(f"{label}: nothing planned; the robot was not commanded.")
                break
            fk_cfg, q_start, knots, idx_t, _covered = result
            if not knots:
                break
            if dry_run:
                logger.info("Dry run: nothing sent to the robot.")
                return

            # Setpoint path per component = planned joints + sag offset (anchor row first).
            q_idx = {c: (idx_t[COMMANDED[c]] if idx_t is not None else idx[COMMANDED[c]]) for c in components}
            groups = {c: np.array([q[q_idx[c]] + off[c] for q in [q_start] + knots]) for c in components}
            err_now = float(np.linalg.norm(goal_b - point_in_base(configuration, point_gb)))
            dur = duration if attempt == 0 else max(1.0, duration * min(1.0, err_now / max(travel0, 1e-6)))
            if use_torso:
                torso_path = float(np.abs(np.diff(groups["torso"], axis=0)).sum(axis=0).max())
                dur = max(dur, torso_path / max(torso_max_vel, 1e-3))
                logger.info(f"Torso path {np.rad2deg(torso_path):.1f} deg, speed cap {torso_max_vel:.2f} rad/s")
            logger.info(f"{'Executing' if attempt == 0 else f'Correction {attempt}:'} "
                        f"{' + '.join(components)} over {dur:.1f}s...")
            last_cmd = stream(bot, groups, dur, config.CONTROL_DT)

            # Verify with measured joints; refresh the sag offsets from what was actually sent.
            time.sleep(max(settle_s, 0.0))
            sync_from_robot(bot, configuration, idx)
            track = {c: last_cmd[c] - configuration.q[idx[COMMANDED[c]]] for c in components}
            worst = max(float(np.abs(v).max()) for v in track.values())
            if worst > TRACKING_ABORT_RAD:
                logger.error(
                    f"Robot did not follow the commanded setpoints (max lag {np.rad2deg(worst):.1f} deg > "
                    f"{np.rad2deg(TRACKING_ABORT_RAD):.0f} deg): "
                    + ", ".join(f"{c} {np.round(np.rad2deg(track[c]), 1)}" for c in components)
                    + ". Another controller (omniteleop follower?) is probably overriding us. Stopping here; "
                    "the pose is left as is."
                )
                break
            off = {c: np.clip(track[c], -MAX_ANCHOR_OFFSET, MAX_ANCHOR_OFFSET) for c in components}
            p_meas = point_in_base(configuration, point_gb)
            err = goal_b - p_meas
            torso_deg = np.rad2deg(configuration.q[idx["torso"]])
            (logger.info if np.linalg.norm(err) < refine_tol_m else logger.warning)(
                f"Measured after move: point {np.round(p_meas, 4)}, error to goal {np.round(err, 4)} "
                f"|e| = {np.linalg.norm(err) * 1000:.1f} mm; torso {np.round(torso_deg, 1)} deg "
                f"(changed {np.round(torso_deg - torso_start_deg, 1)} deg since start); sag offsets (deg) "
                + ", ".join(f"{c} {np.round(np.rad2deg(off[c]), 2)}" for c in components)
            )
            if np.linalg.norm(err) < refine_tol_m:
                break
            if attempt < refine:
                logger.info("Re-planning the remaining error from the measured state...")

        logger.info(f"{label} finished; pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() — pose held.")
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        do_shutdown = bot is not None
    finally:
        if do_shutdown and bot is not None:
            logger.info("Shutting down robot")
            bot.shutdown()
