# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Shared driver for ``14.left_j1_up.py`` (and ``_hands_j1_up_20260927.py``): turn the shoulder joint j1 of one or both arms
so the arm(s) swing straight up; every other joint keeps its measured value. Optional Cartesian *pre-moves*
(``pre_moves``: straight translations of the gripper(s) with orientation held, e.g. a little back and then up,
arms-only Pink IK in the robot base frame) run first so the open fingers leave the block before the shoulder
swings -- with j1 alone the hand moves on an arc about the shoulder and tilts, and clipped the block (2026-09-27).

The j1 stage is a pure joint-space move, no IK. Only the chosen arm(s) are commanded (``_pink_arm.stream``: leash-paced,
speed cap; several arms share one min-jerk profile); torso, head and grippers are never touched.

Direction: j1 rotates about the shoulder Y axis (URDF axis +Y on the left, -Y on the right) and the
home pose sits near +90 deg (left) / -90 deg (right) with the forearm folded down. Moving j1 toward 0
swings the whole arm up over the shoulder -- FK check 2026-09-26: gripper z 0.51 m -> 1.46 m for
90 deg from home, x / y almost unchanged. ``RAISE_SIGN`` holds that sign per side.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import numpy as np
import pinocchio as pin
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    COMPONENT,
    DEFAULT_STEP_RAD,
    GROUP,
    MAX_ANCHOR_OFFSET,
    TORSO_MAX_VEL,
    TORSO_POSTURE_COST,
    anchor_offsets,
    build_arm_ik,
    build_dual_arm_ik,
    detect_foreign_commands,
    plan_dual_pose_path,
    plan_se3_path,
    read_setpoints,
    stream,
)
from _pink_common import ARM_JOINTS, EE, build_ik, connect, disconnect_leave_held, read_joint_pos, sync_from_robot  # noqa: E402

J1 = 0  # index of *_arm_j1 in an arm's joint vector
J1_LIMIT_RAD = 3.071  # URDF vega_1p_gripper L/R_arm_j1 limit (symmetric)
ARM_MAX_VEL = 0.5  # rad/s, same cap the IK scripts use for the arms
RAISE_SIGN = {"left": -1.0, "right": +1.0}  # sign of the j1 change that swings the arm up
# COMPONENT (group -> dexcontrol component, incl. "torso") comes from _pink_arm


def plan_knots(cur: np.ndarray, target_j1: float, duration: float, arm_vel: float) -> tuple[np.ndarray, float]:
    """Linear j1 ramp sampled at 25 knots/s; the other joints stay at ``cur``. Returns (knots, total_s)."""
    delta = float(target_j1 - cur[J1])
    t_total = max(duration, abs(delta) / max(arm_vel, 1e-3))
    n_knots = max(int(np.ceil(t_total * 25.0)), 2)
    frac = np.linspace(0.0, 1.0, n_knots + 1)
    knots = np.tile(cur, (n_knots + 1, 1))
    knots[:, J1] = cur[J1] + delta * frac
    return knots, t_total


def _set_arms(configuration, idx: dict, cur: dict[str, np.ndarray]) -> None:
    """Write measured arm joints into the model, clipped to the URDF limits like ``sync_from_robot`` (the sim can sit
    a few 1e-6 rad outside a limit, e.g. L_arm_j2 -0.453001 vs -0.453, which pink's limit check rejects)."""
    m = configuration.model
    q = configuration.q.copy()
    for s, v in cur.items():
        q[idx[s]] = v
    configuration.update(np.clip(q, m.lowerPositionLimit, m.upperPositionLimit))


def gripper_positions(configuration, idx: dict, cur: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gripper origins (robot base) for the given arm joints; the other joints stay as in ``configuration``."""
    _set_arms(configuration, idx, cur)
    return {s: configuration.get_transform_frame_to_world(EE[s]).translation.copy() for s in cur}


def log_gripper_preview(configuration, idx: dict, side: str, cur: np.ndarray, target: np.ndarray) -> None:
    """FK (robot base) of the gripper before / after the j1 turn, so the plan can be sanity-checked."""
    out = []
    for q_arm in (cur, target):
        _set_arms(configuration, idx, {side: q_arm})
        T = configuration.get_transform_frame_to_world(EE[side])
        out.append((T.translation.copy(), T.rotation[:, 2].copy()))
    _set_arms(configuration, idx, {side: cur})
    (p0, a0), (p1, a1) = out
    logger.info(
        f"{side.capitalize()} gripper (robot base): {np.round(p0, 3)} -> {np.round(p1, 3)} m "
        f"(up {(p1[2] - p0[2]) * 100:+.1f} cm, forward {(p1[0] - p0[0]) * 100:+.1f} cm); "
        f"blade axis EE +Z {np.round(a0, 2)} -> {np.round(a1, 2)}"
    )


def plan_translation(configuration, solver: str, sides: list[str], offset: np.ndarray, step_m: float, use_torso: bool):
    """Straight translation of the chosen gripper(s) by ``offset`` (robot base) with orientation held, planned from
    ``configuration`` with only those arms (and the torso if ``use_torso``) free. Returns ``(rcfg, q_start, knots,
    ridx)`` -- ``knots`` may be a reachable prefix -- or None if no progress at all."""
    dist = float(np.linalg.norm(offset))
    if len(sides) == 2:
        rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, TORSO_POSTURE_COST)
        T0 = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in sides}
        n = max(int(np.ceil(dist / max(step_m, 1e-4))), 2)
        planned = plan_dual_pose_path(
            rcfg, tasks, solver,
            lambda a: {s: pin.SE3(T0[s].rotation.copy(), T0[s].translation + a * offset) for s in T0}, n,
        )
    else:
        s = sides[0]
        rcfg, tasks, ridx = build_arm_ik(configuration, s, use_torso, TORSO_POSTURE_COST)
        T0s = rcfg.get_transform_frame_to_world(EE[s])
        planned = plan_se3_path(rcfg, tasks, solver, pin.SE3(T0s.rotation.copy(), T0s.translation + offset), step_m, DEFAULT_STEP_RAD)
    if planned is None:
        return None
    q_start, knots = planned
    return rcfg, q_start, knots, ridx


def raise_j1(
    side: str | Sequence[str],
    *,
    deg: float = 90.0,
    target_rad: float | None = None,
    reverse: bool = False,
    arm_vel: float = ARM_MAX_VEL,
    duration: float = 3.0,
    tol_deg: float = 1.0,
    pre_moves: Sequence[tuple[str, Sequence[float]]] = (),
    pre_duration: float = 2.0,
    step_m: float = 0.01,
    use_torso: bool = False,
    torso_max_vel: float = TORSO_MAX_VEL,
    anchor_setpoints: bool = True,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Optionally translate the gripper(s) of ``side`` ("left", "right" or a sequence) along each ``pre_moves``
    offset (robot base, orientation held; label + xyz metres), then turn each arm's j1 by ``deg`` in its raise
    direction (or to the absolute ``target_rad``) while every other joint holds. Several arms share one profile per
    stage. ``use_torso`` frees the torso for the pre-moves (streamed with them, capped at ``torso_max_vel``).
    ``anchor_setpoints`` reads the sim's held joint targets first and streams relative to them, so a component is
    never commanded to its sagged *measured* angle (torso: ~1.3 deg on 2026-09-27, a visible drop)."""
    sides = [side] if isinstance(side, str) else list(dict.fromkeys(side))
    if not sides or any(s not in RAISE_SIGN for s in sides):
        raise ValueError(f"side must be 'left', 'right' or a list of them, got {side!r}")
    comps = [COMPONENT[s] for s in sides]
    moves = [(str(lbl), np.asarray(off, dtype=float).reshape(3)) for lbl, off in pre_moves]
    moves = [(lbl, off) for lbl, off in moves if np.linalg.norm(off) >= 1e-4]
    if target_rad is None and deg <= 0:
        raise ValueError("--deg must be positive (use --reverse to change the direction)")
    if not plan_only:
        probe = comps + (["torso"] if (use_torso and moves) else [])
        foreign = detect_foreign_commands(probe)
        busy = {c: n for c, n in foreign.items() if n > 0}
        if busy and not ignore_foreign_commands:
            logger.error(
                "Another controller is streaming "
                + ", ".join(f"{COMMAND_TOPICS[c]} ({n} msgs in 0.7 s)" for c, n in busy.items())
                + "; the sim keeps the latest command. Stop it (e.g. omniteleop) and retry; --ignore-foreign-commands overrides."
            )
            return False
    anchor_comps = comps + (["torso"] if (use_torso and moves) else [])
    setpoints = read_setpoints(anchor_comps) if (anchor_setpoints and not plan_only) else {}
    configuration, _tasks, solver, idx = build_ik()
    bot = connect()
    do_shutdown = not leave_held
    try:
        sync_from_robot(bot, configuration, idx)
        cur: dict[str, np.ndarray] = {s: read_joint_pos(getattr(bot, COMPONENT[s]), COMPONENT[s]) for s in sides}
        off = anchor_offsets(configuration, idx, setpoints, anchor_comps)  # setpoint - measured per component
        n_pre = len(moves)

        # -- Cartesian pre-moves: orientation held, arms only (or + torso) ---------------------------------------
        for k, (label, offset) in enumerate(moves, start=1):
            _set_arms(configuration, idx, cur)
            p_before = gripper_positions(configuration, idx, cur)
            res = plan_translation(configuration, solver, sides, offset, step_m, use_torso)
            if res is None:
                logger.warning(f"Stage {k}/{n_pre + 1} '{label}': not even the first step of a {np.round(offset * 100, 1)} cm move is reachable "
                               "(IK / joint limits); stage skipped, going on.")
                continue
            rcfg, q_s, knots, ridx = res
            rcfg.update(knots[-1])
            ach = {s: rcfg.get_transform_frame_to_world(EE[s]).translation - rcfg_p0 for s, rcfg_p0 in
                   ((s, p_before[s]) for s in sides)}
            rcfg.update(q_s)
            short = any(np.linalg.norm(ach[s] - offset) > 0.005 for s in sides)
            (logger.warning if short else logger.info)(
                f"Stage {k}/{n_pre + 1} '{label}': {'/'.join(sides)} gripper(s) {np.round(offset * 100, 1)} cm (robot base), orientation held: "
                f"{len(knots)} steps planned" + ("" if not short else " (IK / joint limits cut the move; going on with what was reached)") + "; "
                + ", ".join(f"{s} {np.round(ach[s] * 100, 1)} cm" for s in sides)
                + (f"; torso change {np.round(np.rad2deg(knots[-1][ridx['torso']] - q_s[ridx['torso']]), 1)} deg" if use_torso else "")
            )
            if plan_only:
                for s in sides:
                    cur[s] = knots[-1][ridx[s]].copy()
                continue
            Q = np.array([q_s] + knots)
            path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
            t_pre = max(pre_duration, max(path[g] / (torso_max_vel if g == "torso" else max(arm_vel, 1e-3)) for g in path))
            groups = {COMPONENT[g]: Q[:, ii] + off[COMPONENT[g]] for g, ii in ridx.items()}
            logger.info(f"Stage {k}/{n_pre + 1} '{label}': streaming {' + '.join(groups)} over {t_pre:.1f}s...")
            last_cmd, _stopped = stream(bot, groups, t_pre, config.CONTROL_DT)
            time.sleep(0.3)
            sync_from_robot(bot, configuration, idx)
            for c in groups:  # refresh the sag offsets from what was actually sent (as run_stages does)
                off[c] = np.clip(last_cmd[c] - configuration.q[idx[GROUP[c]]], -MAX_ANCHOR_OFFSET, MAX_ANCHOR_OFFSET)
            cur = {s: read_joint_pos(getattr(bot, COMPONENT[s]), COMPONENT[s]) for s in sides}
            p_after = gripper_positions(configuration, idx, cur)
            logger.info(f"Stage {k}/{n_pre + 1} '{label}' done (measured): "
                        + ", ".join(f"{s} {np.round((p_after[s] - p_before[s]) * 100, 1)} cm" for s in sides))

        # -- j1 stage: pure joint-space, other joints hold -------------------------------------------------------
        _set_arms(configuration, idx, cur)
        stage = f"Stage {n_pre + 1}/{n_pre + 1} 'j1'" if n_pre else "j1"
        target: dict[str, float] = {}
        for s in sides:
            names = ARM_JOINTS[s]
            if target_rad is not None:
                target[s] = float(target_rad)
            else:
                sign = -RAISE_SIGN[s] if reverse else RAISE_SIGN[s]
                target[s] = float(cur[s][J1] + sign * np.deg2rad(deg))
            if abs(target[s]) > J1_LIMIT_RAD:
                logger.error(
                    f"{names[J1]} target {target[s]:+.4f} rad ({np.rad2deg(target[s]):+.1f} deg) is outside the joint limit "
                    f"+/-{J1_LIMIT_RAD} rad; nothing commanded."
                )
                return False
            logger.info(
                f"{stage}: {names[J1]} {cur[s][J1]:+.4f} -> {target[s]:+.4f} rad ({np.rad2deg(cur[s][J1]):+.1f} -> {np.rad2deg(target[s]):+.1f} deg, "
                f"change {np.rad2deg(target[s] - cur[s][J1]):+.1f} deg); "
                + ", ".join(f"{n} {cur[s][i]:+.3f}" for i, n in enumerate(names) if i != J1) + " rad held"
            )
        moving = [s for s in sides if abs(target[s] - cur[s][J1]) >= np.deg2rad(tol_deg)]
        for s in sides:
            if s not in moving:
                logger.info(f"{ARM_JOINTS[s][J1]} already at the target; that arm is not moved.")
        if not moving:
            logger.info("Nothing to do in the j1 stage.")
            return True
        # one profile for all arms: as long as the slowest arm needs under the speed cap
        t_total = max(max(duration, abs(target[s] - cur[s][J1]) / max(arm_vel, 1e-3)) for s in moving)
        knots = {s: plan_knots(cur[s], target[s], t_total, arm_vel)[0] for s in moving}
        for s in moving:
            log_gripper_preview(configuration, idx, s, cur[s], knots[s][-1])
        logger.info(
            f"{stage}: {' + '.join(ARM_JOINTS[s][J1] for s in moving)} 0.0-{t_total:.1f}s together "
            f"(speed cap {arm_vel:.2f} rad/s, {len(next(iter(knots.values()))) - 1} knots)"
        )
        if plan_only:
            logger.info("Plan only: nothing commanded.")
            return True
        logger.info(f"{stage}: streaming {' + '.join(COMPONENT[s] for s in moving)} over {t_total:.1f}s...")
        stream(bot, {COMPONENT[s]: knots[s] + off[COMPONENT[s]] for s in moving}, t_total, config.CONTROL_DT)
        ok = True
        for s in moving:
            comp, names = COMPONENT[s], ARM_JOINTS[s]
            after = read_joint_pos(getattr(bot, comp), comp)
            residual = float(abs(target[s] - after[J1]))
            others = float(np.abs(after - cur[s])[[i for i in range(len(names)) if i != J1]].max())
            (logger.info if residual <= np.deg2rad(tol_deg) else logger.warning)(
                f"{s.capitalize()} arm move done; {names[J1]} residual {np.rad2deg(residual):.2f} deg, other joints drifted "
                f"at most {np.rad2deg(others):.2f} deg."
            )
            ok = ok and residual <= np.deg2rad(tol_deg)
        logger.info("Pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return ok
    except KeyboardInterrupt:
        logger.warning("Interrupted -- leaving the last commanded pose held.")
        do_shutdown = True
        return False
    finally:
        if bot is not None and (do_shutdown or plan_only):
            if plan_only:
                disconnect_leave_held(bot)
            else:
                logger.info("Shutting down robot")
                bot.shutdown()
