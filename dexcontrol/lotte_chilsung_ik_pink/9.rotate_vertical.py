# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Rotate a horizontally-held box to vertical with both arms, optionally helped by the torso (Pink IK).

Port of ``lotte_chilsung_2/5.rotate_vertical.py`` onto ``ik_pink.py``. The box is treated as a rigid
body between the two grippers: the left-to-right hand vector (the box long axis) is tipped toward
world -Z (right up, left down; the default ``--tilt-direction down`` since 2026-09-26) or +Z
(``--tilt-direction up``, left up / right down) about the box centre, and both gripper orientations
rotate with it, so the hand-to-hand transform is preserved throughout.

Two IK modes:

* ``--use-torso`` (default): both arms *and* the torso are free in a reduced model with absolute
  gripper targets in the robot base frame (``_pink_arm.build_dual_arm_ik``). The torso only moves as
  much as the posture cost allows (a few degrees), but that is enough to finish the tilt where the
  arms alone stall (2026-09-25: arms only stopped at 26% with the right shoulder ``R_arm_j2`` on its
  +0.45 rad limit; with the torso the plan reached vertical with 1-3 deg of torso motion). Arms and
  torso are streamed together (``_pink_arm.stream``, leash-paced, per-joint speed caps).
* ``--no-use-torso``: the original arms-only IK in the arm_center frame (torso never commanded). If the
  full tilt is not reachable, the largest reachable prefix of the path is executed.

Head is never commanded. ``--plan-only`` connects, plans from the measured state and reports how far
the tilt gets without moving anything.

Example::

    python 5.rotate_vertical.py                     # to vertical (right up, left down), arms + torso, 8 s
    python 5.rotate_vertical.py --plan-only         # report reachability only
    python 5.rotate_vertical.py --no-use-torso      # arms only (old behaviour)
    python 5.rotate_vertical.py --tilt-direction up # the other way: left up, right down
    python 5.rotate_vertical.py --max-angle-deg 45  # only tip 45 deg
    python 5.rotate_vertical.py --no-leave-held
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pinocchio as pin
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    COMPONENT,
    TORSO_MAX_VEL,
    TORSO_POSTURE_COST,
    build_dual_arm_ik,
    detect_foreign_commands,
    plan_dual_pose_path,
    stream,
)
from _pink_common import (  # noqa: E402
    EE,
    build_ik,
    connect,
    disconnect_leave_held,
    ee_poses,
    rigid_rotation_pose_at,
    run_pose_move,
    sync_from_robot,
    world_up_in_arm_center,
)

UP_BASE = np.array([0.0, 0.0, 1.0])  # world up in the robot base frame (flat floor)


def tilt_geometry(poses: dict[str, pin.SE3], up: np.ndarray, tilt_direction: str, step_deg: float,
                  max_angle_deg: float | None, frame_label: str):
    """(pose_at, n) tipping the box long axis (left - right) toward ``up`` / ``-up`` about the box centre.

    ``poses`` and ``up`` must be expressed in the same frame; ``pose_at`` returns poses in that frame.
    """
    p_l, p_r = poses["left"].translation, poses["right"].translation
    center = 0.5 * (p_l + p_r)
    box_axis = p_l - p_r
    gap = float(np.linalg.norm(box_axis))
    if gap < 1e-4:
        logger.error("Hands nearly coincident; no box axis.")
        return None
    box_axis /= gap
    target = up if tilt_direction == "up" else -up
    rot_axis = np.cross(box_axis, target)
    if np.linalg.norm(rot_axis) < 1e-6:  # already parallel to +/-Z
        rot_axis = np.array([1.0, 0.0, 0.0])
    angle = float(np.arccos(np.clip(box_axis @ target, -1.0, 1.0)))
    if max_angle_deg is not None:
        angle = min(angle, np.deg2rad(max_angle_deg))
    logger.info(
        f"Vertical tilt ({tilt_direction}): box axis {np.round(box_axis, 3)} -> {np.round(target, 3)}, "
        f"{np.rad2deg(angle):.1f} deg about {np.round(rot_axis / np.linalg.norm(rot_axis), 3)}, "
        f"centre {np.round(center, 3)} ({frame_label}), hand gap {gap:.3f} m"
    )
    if angle < np.deg2rad(0.5):
        logger.info("Already vertical; nothing to do.")
        return rigid_rotation_pose_at(poses, center, rot_axis, 0.0), 0
    n = max(int(np.ceil(np.rad2deg(angle) / max(step_deg, 0.1))), 2)
    return rigid_rotation_pose_at(poses, center, rot_axis, angle), n


def vertical_tilt_plan(configuration, tilt_direction: str, step_deg: float, max_angle_deg: float | None):
    """Arms-only plan in the arm_center frame (original behaviour)."""
    return tilt_geometry(ee_poses(configuration), world_up_in_arm_center(configuration), tilt_direction,
                         step_deg, max_angle_deg, "arm_center")


def _box_angle_to_up(rcfg, up: np.ndarray) -> float:
    p = {s: rcfg.get_transform_frame_to_world(EE[s]).translation for s in ("left", "right")}
    ax = p["left"] - p["right"]
    ax /= max(np.linalg.norm(ax), 1e-9)
    return float(np.rad2deg(np.arccos(np.clip(ax @ up, -1.0, 1.0))))


def run_tilt_dual(
    *, tilt_direction: str, duration: float, step_deg: float, max_angle_deg: float | None, leave_held: bool,
    use_torso: bool, torso_posture_cost: float, max_joint_vel: float, torso_max_vel: float, plan_only: bool,
    ignore_foreign_commands: bool,
) -> bool:
    """Both arms (+ torso) in the robot base frame; plan from the measured state and stream the path."""
    components = ["left_arm", "right_arm"] + (["torso"] if use_torso else [])
    if not plan_only:
        foreign = detect_foreign_commands(components)
        blocking = {c: n for c, n in foreign.items() if n > 0}
        if blocking and not ignore_foreign_commands:
            logger.error(
                "Another controller is streaming joint targets we need to command: "
                + ", ".join(f"{COMMAND_TOPICS[c]} ({n} msgs in 0.7 s)" for c, n in blocking.items())
                + ". Stop it (e.g. omniteleop) and retry; --ignore-foreign-commands overrides this check."
            )
            return False
        if blocking:
            logger.warning(f"Foreign commands detected but ignored on request: {blocking}")

    configuration, _tasks, solver, idx = build_ik()
    bot = connect()
    do_shutdown = not leave_held
    try:
        sync_from_robot(bot, configuration, idx)
        rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        poses = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
        plan = tilt_geometry(poses, UP_BASE, tilt_direction, step_deg, max_angle_deg, "robot base")
        if plan is None:
            do_shutdown = True
            return False
        pose_at, n = plan
        if n == 0:
            return True
        planned = plan_dual_pose_path(rcfg, tasks, solver, pose_at, n)
        if planned is None:
            do_shutdown = True
            return False
        q_start, knots = planned
        Q = np.array([q_start] + knots)
        rcfg.update(Q[-1])
        end_angle = _box_angle_to_up(rcfg, UP_BASE if tilt_direction == "up" else -UP_BASE)
        rcfg.update(q_start)
        path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
        torso_note = ""
        if use_torso:
            d_t = np.rad2deg(Q[-1, ridx["torso"]] - Q[0, ridx["torso"]])
            torso_note = f"; torso change (j1, j2, j3) {np.round(d_t, 1)} deg"
        logger.info(
            f"Plan covers {100 * len(knots) / n:.0f}% of the tilt; box axis ends {end_angle:.1f} deg from "
            f"{'up' if tilt_direction == 'up' else 'down'}{torso_note}; joint path (rad) "
            + ", ".join(f"{g} {path[g]:.2f}" for g in path)
        )
        if plan_only:
            logger.info("Plan only: nothing commanded.")
            return len(knots) == n
        t_stream = max(duration, max(path[g] / (torso_max_vel if g == "torso" else max_joint_vel) for g in path))
        if t_stream > duration:
            logger.info(f"Stream time stretched to {t_stream:.1f}s by the joint speed caps "
                        f"(arms {max_joint_vel:.2f}, torso {torso_max_vel:.2f} rad/s)")
        groups = {COMPONENT[g]: Q[:, ii] for g, ii in ridx.items()}
        logger.info(f"Executing vertical tilt ({', '.join(groups)}) over {t_stream:.1f}s...")
        stream(bot, groups, t_stream, config.CONTROL_DT)
        sync_from_robot(bot, configuration, idx)
        rcfg2, _t2, _i2 = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        logger.info(f"Vertical tilt completed; measured box axis {_box_angle_to_up(rcfg2, UP_BASE if tilt_direction == 'up' else -UP_BASE):.1f} deg "
                    f"from {'up' if tilt_direction == 'up' else 'down'}; final pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return len(knots) == n
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        do_shutdown = True
        return False
    finally:
        if bot is not None and (do_shutdown or plan_only):
            if plan_only:
                disconnect_leave_held(bot)
            else:
                logger.info("Shutting down robot")
                bot.shutdown()


def main(
    tilt_direction: str = "down",
    duration: float = 8.0,
    step_deg: float = 2.0,
    max_angle_deg: float | None = None,
    leave_held: bool = True,
    use_torso: bool = True,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    max_joint_vel: float = 0.3,
    torso_max_vel: float = TORSO_MAX_VEL,
    plan_only: bool = False,
    ignore_foreign_commands: bool = False,
) -> None:
    """Tip the held box from horizontal to vertical with both arms (and a little torso by default).

    Args:
        tilt_direction: "down" (default, 2026-09-26) raises the right hand / lowers the left; "up" reverses.
        duration: Nominal stream time for the whole rotation (seconds); stretched by the speed caps if needed.
        step_deg: Rotation per IK waypoint (degrees).
        max_angle_deg: Optional cap on the tilt angle (default: go fully vertical).
        leave_held: Exit without ``stop()`` so the pose stays held.
        use_torso: Free the torso in the IK (robot base frame) so the arms can finish the tilt. False = arms only.
        torso_posture_cost: Posture cost on the torso joints; higher = torso moves less (arms use 1e-3).
        max_joint_vel: Arm joint speed cap (rad/s) for the streamed path (torso mode only).
        torso_max_vel: Torso joint speed cap (rad/s) (torso mode only).
        plan_only: Connect, plan from the measured state, report how far the tilt gets, move nothing (torso mode only).
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets (torso mode only).
    """
    if tilt_direction not in ("up", "down"):
        raise ValueError("tilt_direction must be 'up' or 'down'")
    if use_torso or plan_only:
        run_tilt_dual(
            tilt_direction=tilt_direction, duration=duration, step_deg=step_deg, max_angle_deg=max_angle_deg,
            leave_held=leave_held, use_torso=use_torso, torso_posture_cost=torso_posture_cost,
            max_joint_vel=max_joint_vel, torso_max_vel=torso_max_vel, plan_only=plan_only,
            ignore_foreign_commands=ignore_foreign_commands,
        )
        return
    run_pose_move(
        lambda cfg: vertical_tilt_plan(cfg, tilt_direction, step_deg, max_angle_deg),
        duration=duration, leave_held=leave_held, label="vertical tilt",
    )


if __name__ == "__main__":
    tyro.cli(main)
