# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Box lift up: move torso + both arms to a recorded lift pose, then (optionally) lift straight up.

``lotte_chilsung_2/3.box_lift_up.py`` blends all joints to a recorded pose where most of the
lift comes from the torso standing up. This version does the same in joint space with the
tracking-aware streaming of ``_pink_arm.stream`` and can add a Cartesian lift afterwards:

  1. **Pose** -- torso, left arm and right arm are streamed *together* from the measured joints
     to the recorded lift pose ``LIFT_POSE`` (``{joint name: radians}``, e.g. ``'L_arm_j1': 0.15``;
     ``--pose-file`` loads a dict in the same form (JSON or Python literal) and ``--torso-rad`` /
     ``--torso-deg``, ``--left-arm-rad``, ``--right-arm-rad`` override single components). One min-jerk
     profile for all 17 joints, its duration set by the slowest joint under the speed caps
     (``--max-joint-vel`` for the arms, ``--torso-vel`` for the torso), slowed while the robot
     lags. ``--torso-first`` moves the torso alone first (blocking ``set_joint_pos_vel``; the arm
     joints keep their setpoint so the box rides rigidly with the chest) and streams the arms
     afterwards. ``--no-move-arms`` moves the torso only.
  2. **Arms up** (``--cm`` > 0, default 0) -- both grippers move straight up along world +Z
     (computed from FK, since arm_center pitches with the torso) with orientation fixed.

Head is never commanded. Exits without ``stop()`` so the pose stays held.

Example::

    python 3.box_lift_up.py                       # torso + both arms to the recorded lift pose
    python 3.box_lift_up.py --cm 5 --duration 3   # ... then 5 cm straight up (hand gap kept)
    python 3.box_lift_up.py --torso-first         # torso first, then the arms
    python 3.box_lift_up.py --no-move-arms        # torso only
    python 3.box_lift_up.py --torso-rad 0.1957 0.3561 0.3164 --torso-vel 0.2   # torso target in radians
    python 3.box_lift_up.py --torso-deg 11.2 20.4 18.1                        # the same, in degrees
    python 3.box_lift_up.py --pose-file lift_pose.txt   # {'L_arm_j1': ..., 'torso_j1': ...} dict from a file
    python 3.box_lift_up.py --no-move-pose --cm 5 # skip the pose stage: Cartesian lift only
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _pink_arm import stream as stream_joints  # noqa: E402
from _pink_common import (  # noqa: E402
    ARM_JOINTS,
    Targets,
    ee_poses,
    read_joint_pos,
    run_cartesian_move,
    world_up_in_arm_center,
)
from dexcontrol.robot import Robot  # noqa: E402

# Recorded lift pose: {URDF joint name: radians} (measured on the sim, 2026-09-25). Paste a pose
# printed by the robot / recorder here as is; --pose-file loads a file in the same form.
# The older recorded torso from lotte_chilsung_2/3.box_lift_up.py was (0.5668, 1.0439, 0.4032) rad
# = (32.5, 59.8, 23.1) deg, chest nearly upright (~4 deg pitch).
LIFT_POSE: dict[str, float] = {
    'L_arm_j1': 0.15048722922801971,
    'L_arm_j2': -0.21428821980953217,
    'L_arm_j3': 0.20760338008403778,
    'L_arm_j4': -0.30718883872032166,
    'L_arm_j5': -0.19593815505504608,
    'L_arm_j6': 0.2470530867576599,
    'L_arm_j7': 0.27619779109954834,
    'R_arm_j1': -0.21666952967643738,
    'R_arm_j2': 0.26929357647895813,
    'R_arm_j3': 0.017410503700375557,
    'R_arm_j4': -0.43465209007263184,
    'R_arm_j5': -0.052145637571811676,
    'R_arm_j6': -0.3151007294654846,
    'R_arm_j7': -0.24235761165618896,
    'torso_j1': 0.1957302689552307,
    'torso_j2': 0.356093168258667,
    'torso_j3': 0.3163677752017975,
}

# dexcontrol component -> ordered URDF joint names (from _pink_common.ARM_JOINTS)
COMPONENT_JOINTS: dict[str, list[str]] = {
    "torso": ARM_JOINTS["torso"],
    "left_arm": ARM_JOINTS["left"],
    "right_arm": ARM_JOINTS["right"],
}
COMPONENT_DOF = {c: len(names) for c, names in COMPONENT_JOINTS.items()}


def pose_to_components(pose: dict[str, float]) -> dict[str, np.ndarray]:
    """Split a ``{joint name: rad}`` dict into per-component arrays (torso / left_arm / right_arm).

    A component is included only if *all* its joints are present; a component with some joints
    missing, or an unknown joint name, is an error.
    """
    known = {n for names in COMPONENT_JOINTS.values() for n in names}
    unknown = sorted(set(pose) - known)
    if unknown:
        raise ValueError(f"unknown joint names in the pose: {unknown} (expected e.g. {sorted(known)[:3]} ...)")
    out: dict[str, np.ndarray] = {}
    for comp, names in COMPONENT_JOINTS.items():
        have = [n for n in names if n in pose]
        if not have:
            continue
        if len(have) != len(names):
            raise ValueError(f"{comp}: pose has {len(have)}/{len(names)} joints ({sorted(set(names) - set(have))} missing)")
        out[comp] = np.array([float(pose[n]) for n in names], dtype=float)
    return out


def load_pose(path: Path) -> dict[str, float]:
    """Read a ``{joint name: rad}`` dict from a file (JSON or a Python dict literal as printed)."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dict of joint name -> radians")
    return {str(k): float(v) for k, v in data.items()}


TORSO_LIFT_RAD = tuple(float(LIFT_POSE[n]) for n in COMPONENT_JOINTS["torso"])
TORSO_LIFT_DEG = tuple(float(x) for x in np.rad2deg(TORSO_LIFT_RAD))  # same torso pose in degrees

Vec3 = tuple[float, float, float]
Vec7 = tuple[float, float, float, float, float, float, float]


def move_torso(
    bot: Robot,
    target_rad: tuple[float, float, float],
    max_vel: float = 0.15,
    tol: float = 0.03,
    extra_wait: float = 10.0,
) -> None:
    """Blocking torso move to ``target_rad`` (radians); per-joint speed scaled so all joints finish together."""
    target = np.asarray(target_rad, dtype=float).reshape(3)
    current = np.asarray(bot.torso.get_joint_pos(), dtype=float)
    err = target - current
    travel = float(np.abs(err).max())
    if travel < tol:
        logger.info("Torso already at target; skipping torso move.")
        return
    T = travel / max_vel
    vel = np.clip(np.abs(err) / T, 0.01, 2.0)
    logger.info(
        f"Torso {np.round(current, 4)} → {np.round(target, 4)} rad "
        f"({np.round(np.rad2deg(current), 1)} → {np.round(np.rad2deg(target), 1)} deg), "
        f"~{T:.1f}s at max {max_vel:.2f} rad/s (vel {np.round(vel, 3)})"
    )
    bot.torso.set_joint_pos_vel(
        joint_pos=target,
        joint_vel=vel,
        wait_time=T + extra_wait,
        exit_on_reach=True,
        exit_on_reach_kwargs={"tolerance": tol},
    )
    residual = float(np.abs(target - np.asarray(bot.torso.get_joint_pos(), dtype=float)).max())
    (logger.info if residual <= tol else logger.warning)(
        f"Torso move done (residual {np.rad2deg(residual):.1f} deg, "
        f"pitch {np.rad2deg(bot.torso.pitch_angle):.1f} deg)."
    )


def move_to_pose(
    bot: Robot,
    targets: dict[str, np.ndarray],
    *,
    max_joint_vel: float = 0.3,
    torso_vel: float = 0.15,
    tol: float = 0.03,
    torso_first: bool = False,
) -> None:
    """Blocking joint-space move of the given components (``torso`` / ``left_arm`` / ``right_arm``) to
    ``targets`` (radians). All components are streamed together with ``_pink_arm.stream``; with
    ``torso_first`` the torso goes first (``move_torso``) and the arms follow."""
    targets = {c: np.asarray(v, dtype=float).reshape(-1) for c, v in targets.items()}
    for c, v in targets.items():
        if c not in COMPONENT_DOF or v.shape != (COMPONENT_DOF[c],):
            raise ValueError(f"{c}: expected {COMPONENT_DOF.get(c, '?')} joint values, got {v.shape}")
    if torso_first and "torso" in targets:
        move_torso(bot, tuple(targets["torso"]), max_vel=torso_vel, tol=tol)
        targets = {c: v for c, v in targets.items() if c != "torso"}
        if not targets:
            return
    comps = list(targets)
    meas = {c: read_joint_pos(getattr(bot, c), c) for c in comps}
    err = {c: targets[c] - meas[c] for c in comps}
    travel = max(float(np.abs(e).max()) for e in err.values())
    if travel < tol:
        logger.info(f"Pose already reached (max joint error {np.rad2deg(travel):.1f} deg); skipping.")
        return
    caps = {c: (torso_vel if c == "torso" else max_joint_vel) for c in comps}
    duration = max(max(float(np.abs(err[c]).max()) / max(caps[c], 1e-3) for c in comps), 1.0)
    for c in comps:
        logger.info(
            f"{c}: {np.round(meas[c], 4)} → {np.round(targets[c], 4)} rad "
            f"(max |d| {np.rad2deg(np.abs(err[c]).max()):.1f} deg)"
        )
    logger.info(
        f"Streaming {', '.join(comps)} together to the lift pose: ~{duration:.1f}s "
        f"(arm cap {max_joint_vel:.2f} rad/s, torso cap {torso_vel:.2f} rad/s)"
    )
    groups = {c: np.stack([meas[c], targets[c]]) for c in comps}
    stream_joints(bot, groups, duration, config.CONTROL_DT)
    after = {c: read_joint_pos(getattr(bot, c), c) for c in comps}
    residual = max(float(np.abs(targets[c] - after[c]).max()) for c in comps)
    (logger.info if residual <= tol else logger.warning)(
        f"Pose move done (max residual {np.rad2deg(residual):.1f} deg; per component (deg): "
        + ", ".join(f"{c} {np.round(np.rad2deg(targets[c] - after[c]), 1)}" for c in comps)
        + (f"; torso pitch {np.rad2deg(bot.torso.pitch_angle):.1f} deg" if "torso" in comps else "")
        + ")"
    )


def lift_targets(configuration, up_m: float) -> Targets:
    """Both grippers translated by ``up_m`` along world up (arm_center frame)."""
    up = world_up_in_arm_center(configuration)
    p = {s: T.translation.copy() for s, T in ee_poses(configuration).items()}
    logger.info(
        f"Lift {up_m * 100:.1f} cm along world-up {np.round(up, 3)} (arm_center frame); "
        f"hand gap {np.linalg.norm(p['left'] - p['right']):.3f} m kept."
    )
    return {s: p[s] + up_m * up for s in p}


def main(
    cm: float = 0.0,
    duration: float = 2.0,
    step_m: float = 0.01,
    move_pose: bool = True,
    move_arms: bool = True,
    torso_first: bool = False,
    pose_file: Path | None = None,
    torso_rad: Vec3 | None = None,
    torso_deg: Vec3 | None = None,
    left_arm_rad: Vec7 | None = None,
    right_arm_rad: Vec7 | None = None,
    max_joint_vel: float = 0.3,
    torso_vel: float = 0.15,
    leave_held: bool = True,
) -> None:
    """Move torso and both arms to the recorded lift pose, then optionally lift the box straight up.

    Args:
        cm: Cartesian arm lift after the pose stage (centimeters, world up). 0 = pose stage only.
        duration: Stream time of the Cartesian lift (seconds).
        step_m: Cartesian spacing between IK waypoints (meters).
        move_pose: If False, skip the pose stage (Cartesian lift only).
        move_arms: If False, the pose stage moves the torso only (arms keep their setpoint).
        torso_first: Move the torso first (blocking), then stream the arms; default streams all together.
        pose_file: File with a ``{joint name: radians}`` dict (JSON or Python literal) replacing ``LIFT_POSE``.
        torso_rad: Torso target [j1, j2, j3] in radians (default: from the pose).
        torso_deg: Torso target [j1, j2, j3] in degrees; alternative to ``torso_rad`` (not both).
        left_arm_rad: Left arm target [j1..j7] in radians (default: from the pose).
        right_arm_rad: Right arm target [j1..j7] in radians (default: from the pose).
        max_joint_vel: Arm joint speed cap for the pose stage (rad/s).
        torso_vel: Torso joint speed cap (rad/s). Keep modest under box load.
        leave_held: Exit without ``stop()`` so the pose stays held.
    """
    if torso_rad is not None and torso_deg is not None:
        raise ValueError("give either --torso-rad or --torso-deg, not both")
    pose = load_pose(pose_file) if pose_file is not None else LIFT_POSE
    base = pose_to_components(pose)
    if pose_file is not None:
        logger.info(f"Pose from {pose_file}: {len(pose)} joints -> components {list(base)}")
    targets: dict[str, np.ndarray] = {}
    if torso_deg is not None:
        targets["torso"] = np.deg2rad(np.asarray(torso_deg, dtype=float))
    elif torso_rad is not None:
        targets["torso"] = np.asarray(torso_rad, dtype=float)
    elif "torso" in base:
        targets["torso"] = base["torso"]
    if move_arms:
        for comp, override in (("left_arm", left_arm_rad), ("right_arm", right_arm_rad)):
            if override is not None:
                targets[comp] = np.asarray(override, dtype=float)
            elif comp in base:
                targets[comp] = base[comp]
    if not targets:
        raise ValueError("no target: the pose has no complete component and no --torso-rad/--*-arm-rad was given")
    pre = (
        (lambda bot: move_to_pose(bot, targets, max_joint_vel=max_joint_vel, torso_vel=torso_vel,
                                  torso_first=torso_first))
        if move_pose else None
    )
    run_cartesian_move(
        lambda cfg: lift_targets(cfg, max(cm, 0.0) / 100.0),
        duration=duration, step_m=step_m, leave_held=leave_held,
        label="box lift up", pre_move=pre,
    )


if __name__ == "__main__":
    tyro.cli(main)
