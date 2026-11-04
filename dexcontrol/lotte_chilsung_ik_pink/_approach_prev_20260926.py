# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Bimanual: turn the base just enough, then carry the purple box centre onto the red conveyor tip.

Port of ``lotte_chilsung_2/6.approach_purple_to_red_bimanual.py`` (Pink IK), plus a base-alignment
stage in front of it:

  0. **align base** (``--align-mode``): the torso joints are all pitch and the arms cannot shift a
     two-handed grasp far sideways, so a sideways gap between purple and red is closed with the base.
     ``yaw`` (default) turns in place, closed loop on the live marker files (``_base_turn.turn_in_place``),
     until the target lies on the purple's bearing *minus* ``--yaw-short-deg`` (default 4 deg): turning
     all the way swings the box held 76 cm ahead into the conveyor box (2026-09-25: 7 cm deep), so the
     base stops a few degrees early and the arms close what they can of the remaining sideways gap.
     ``crab`` translates the base sideways instead (no rotation, slower cycle); ``none`` skips the stage.
     Afterwards both markers are re-read.
     ``--standoff`` (default None = automatic flush placement, see above; 0 = purple exactly on red, the
     held block then overlaps the solid conveyor box in the sim). The tip face normal is a known world
     direction (``--face-normal-world``, +X) turned into the base frame with the base yaw fitted from the
     world-fixed markers (red tip + card-stack edge markers), so no extra marker is needed.
     For yaw mode the script also predicts the clearance between the held block and the conveyor face
     (block half extents ``--block-half-fwd`` / ``--block-half-lat``) and warns when it is negative.
     With ``--concurrent`` (default, yaw mode) the arm approach runs *while* the base turns, starting
     once the turn is ``--approach-start`` (50 %) done. By default (``--servo``) the arms are driven
     closed-loop on the live marker files: every tick red and purple are re-read in the current base
     frame and the rigid gripper pair steps toward the remaining delta with a few Pink IK iterations,
     so the motion is one continuous move that follows the turning base and ends when the purple
     centre is within ``--servo-tol`` (5 mm) of the target -- no separate correction. The servo also
     squares the block to the conveyor box (``--align-block``): the gripper pair is rotated about robot
     z until the block's thickness axis is parallel to the tip face normal, and the default standoff is
     the block half thickness + ``--face-margin`` (5 mm), so the block's 40 x 35 cm face ends up flush
     with the 40 x 35 cm tip face, edges aligned. ``--no-servo``
     uses a pre-planned path (planned against the predicted post-turn geometry) plus a correction move.
     The combined base + arm sweep is checked against the conveyor face; the script falls back to
     turn-then-approach only if the concurrent motion would get closer to the face than the final pose.
  1. **approach**: the grasp is kept rigid (relative L/R gripper poses and orientations fixed) and
     both grippers are translated by

    delta = red_tip - purple_centre        (robot base frame)

so the purple marker lands on the red marker. The purple ball follows the centre of ``card_block_35``
(sim_main.update_card_box_mass_center_marker), which moves rigidly with the grasp, so translating the
gripper pair by ``delta`` moves the purple centre onto red.

IK: both arms + torso free in a reduced model with absolute gripper targets in the robot base frame
(``_pink_arm.build_dual_arm_ik``; ``--no-use-torso`` = arms only). Like the original, several
translation paths are tried -- straight line and axis-ordered detours (z then xy, xy then z, x-y-z,
z-y-x) -- each planned as far as IK / joint limits allow, and the one that leaves the smallest
purple-to-red residual is executed (the largest reachable prefix if none completes). Arms and torso
are streamed together with ``_pink_arm.stream`` (leash-paced, per-joint speed caps). There is no
self-collision model here; the tracking leash and the speed caps are the only safety nets.

Sim must be writing (robot base frame, see ``_sim_markers.py``)::

    unitree_sim_isaaclab*/card_box_tip_pos.txt          (red)
    unitree_sim_isaaclab*/card_box_mass_center_pos.txt  (purple)

Example (box held with both hands)::

    python 10.approach_purple_to_red_bimanual.py --plan-only        # base move needed + approach feasibility, no motion
    python 10.approach_purple_to_red_bimanual.py --yaw-short-deg 6  # turn even less (more clearance, larger residual)
    python 10.approach_purple_to_red_bimanual.py --align-mode crab  # translate the base sideways instead
    python 10.approach_purple_to_red_bimanual.py --align-mode none  # skip the base stage (arms + torso only)
    python 10.approach_purple_to_red_bimanual.py --standoff 0       # purple exactly on red (box overlaps the conveyor box)
    python 10.approach_purple_to_red_bimanual.py                    # execute
    python 10.approach_purple_to_red_bimanual.py --stop-short 0.06  # stop 6 cm before red
    python 10.approach_purple_to_red_bimanual.py --offset 0 0.05 0  # red target + robot-base offset
    python 10.approach_purple_to_red_bimanual.py --scale 0.5        # close half the gap
    python 10.approach_purple_to_red_bimanual.py --no-use-torso     # arms only
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pink
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
from _base_turn import WHEEL_GAIN_SIM, measure_rtf, translate_sideways, turn_in_place, wrap  # noqa: E402
from _pink_common import EE, MAX_JOINT_JUMP, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402
from _sim_markers import (  # noqa: E402
    PURPLE_FILE_NAME,
    TIP_FACE_NORMAL_WORLD,
    TIP_FILE_NAME,
    find_sim_dir,
    load_xyz_base,
    load_xyz_pair,
    world_dir_to_base,
)

CLOSE_ENOUGH_M = 0.01  # residual below this counts as reached


def face_normal_base(sim_dir: Path, normal_world, max_age_s: float) -> np.ndarray:
    """Unit normal of the conveyor tip face in the robot base frame, pointing out of the face toward free
    space. The face normal is a known world direction (``normal_world``, default world +X from the scene
    config); it is turned into the base frame with the base yaw fitted from the world-fixed marker pairs.
    Falls back to robot -y (the face is to the robot's left when the base is square to the conveyor) if the
    yaw cannot be measured."""
    try:
        n = world_dir_to_base(sim_dir, normal_world, max_age_s)
        n[2] = 0.0
        if np.linalg.norm(n) > 1e-6:
            return n / np.linalg.norm(n)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"Tip face normal unavailable ({exc}); assuming robot -y.")
    return np.array([0.0, -1.0, 0.0])


def approach_target(red: np.ndarray, purple: np.ndarray, offset, stop_short: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(target, delta) for the purple centre: red + offset (which includes the standoff), shortened by
    ``stop_short``, scaled by ``scale``."""
    target = red + np.asarray(offset, dtype=float)
    full = target - purple
    full_dist = float(np.linalg.norm(full))
    if stop_short > 0.0 and full_dist > 1e-9:
        target = target - min(stop_short, 0.95 * full_dist) * full / full_dist
    target = purple + float(scale) * (target - purple)
    return target, target - purple


def bearing_yaw(purple: np.ndarray, target: np.ndarray) -> float:
    """Signed base yaw (rad, +CCW) that puts ``target`` on the purple's bearing about the base origin."""
    return wrap(float(np.arctan2(target[1], target[0]) - np.arctan2(purple[1], purple[0])))


def rot_z(p: np.ndarray, yaw: float) -> np.ndarray:
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([c * p[0] - s_ * p[1], s_ * p[0] + c * p[1], p[2]])


def predict_face_clearance(
    tip_path: Path, normal_world, purple_path: Path, max_age_s: float, yaw: float,
    half_fwd: float, half_lat: float, face_half_width: float, shift_b: np.ndarray | None = None,
    block_yaw: float = 0.0,
) -> float | None:
    """Smallest distance (m, negative = penetration) between the held block's footprint corners and the
    conveyor tip face plane after the base yaws by ``yaw`` in place, considering only corners that would
    hit the face (within ``face_half_width`` of its centre along the face). Uses the world lines of the
    marker files to place the base in the world; returns None if a file lacks them.

    Block model: a box of half extents ``half_fwd`` (robot x) x ``half_lat`` (robot y) centred on purple
    (+ ``shift_b``, robot frame) and yawed by ``block_yaw`` about robot z (the servo's alignment), rigid
    with the base otherwise. The face is the plane through red with the normal ``normal_world`` (world frame).
    """
    try:
        red_b, red_w = load_xyz_pair(tip_path, max_age_s)
        purple_b, purple_w = load_xyz_pair(purple_path, max_age_s)
    except (FileNotFoundError, ValueError):
        return None
    # base pose in the world plane from the two point pairs
    d_b, d_w = (purple_b - red_b)[:2], (purple_w - red_w)[:2]
    if np.linalg.norm(d_b) < 0.02:
        return None
    base_yaw = float(np.arctan2(d_w[1], d_w[0]) - np.arctan2(d_b[1], d_b[0]))
    R0 = np.array([[np.cos(base_yaw), -np.sin(base_yaw)], [np.sin(base_yaw), np.cos(base_yaw)]])
    t = red_w[:2] - R0 @ red_b[:2]
    n_w = np.asarray(normal_world, dtype=float)[:2]
    if np.linalg.norm(n_w) < 1e-6:
        return None
    n_w = n_w / np.linalg.norm(n_w)
    tang = np.array([-n_w[1], n_w[0]])
    a = base_yaw + yaw
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    centre_b = purple_b[:2] + (np.asarray(shift_b, dtype=float)[:2] if shift_b is not None else 0.0)
    Rb = np.array([[np.cos(block_yaw), -np.sin(block_yaw)], [np.sin(block_yaw), np.cos(block_yaw)]])
    best = None
    for sx in (+1.0, -1.0):
        for sy in (+1.0, -1.0):
            corner = t + R @ (centre_b + Rb @ np.array([sx * half_fwd, sy * half_lat]))
            rel = corner - red_w[:2]
            if abs(float(rel @ tang)) <= face_half_width:
                dist = float(rel @ n_w)  # >0: on the robot side of the face; <0: inside the conveyor box
                best = dist if best is None else min(best, dist)
    return best


def candidate_offset_paths(delta: np.ndarray) -> list[tuple[str, list[np.ndarray]]]:
    """Translation paths for the gripper pair, as waypoints of the offset from the start (0 -> delta).

    Straight line plus axis-ordered detours, which often reach farther (same set as the original).
    """
    dx, dy, dz = (float(v) for v in delta)
    z = np.array([0.0, 0.0, dz])
    xy = np.array([dx, dy, 0.0])
    x = np.array([dx, 0.0, 0.0])
    y = np.array([0.0, dy, 0.0])
    return [
        ("straight", [delta]),
        ("z_then_xy", [z, delta]),
        ("xy_then_z", [xy, delta]),
        ("x_y_z", [x, x + y, delta]),
        ("z_y_x", [z, z + y, delta]),
    ]


def densify(waypoints: list[np.ndarray], step_m: float) -> list[np.ndarray]:
    """Offsets from 0 through ``waypoints`` with consecutive points at most ``step_m`` apart (excludes 0)."""
    out: list[np.ndarray] = []
    cur = np.zeros(3)
    for nxt in waypoints:
        nxt = np.asarray(nxt, dtype=float)
        dist = float(np.linalg.norm(nxt - cur))
        if dist < 1e-9:
            continue
        n = max(int(np.ceil(dist / max(step_m, 1e-4))), 1)
        for i in range(1, n + 1):
            out.append((1.0 - i / n) * cur + (i / n) * nxt)
        cur = nxt
    return out


def _quiet_xyz(path: Path) -> np.ndarray:
    """First 'x y z' line of a marker file without logging (used at the chassis loop rate)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s_ = line.split()
        if s_ and s_[0] != "#" and len(s_) >= 3:
            return np.array([float(s_[0]), float(s_[1]), float(s_[2])])
    raise ValueError(f"{path}: no 'x y z' line")


def plan_approach(rcfg, tasks, solver, delta: np.ndarray, step_m: float):
    """Try the candidate translation paths for the gripper pair from the current reduced state and return
    ``(residual_m, label, Q, n, mid_offsets)`` for the one leaving the smallest purple-to-target residual
    (``Q`` = joint knots incl. the start row, ``mid_offsets[k]`` = gripper-pair midpoint shift at knot k),
    or None if no path makes progress."""
    T0 = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
    mid0 = pair_midpoint(rcfg)
    q0 = rcfg.q.copy()
    best = None
    for label, waypoints in candidate_offset_paths(delta):
        dense = densify(waypoints, step_m)
        if not dense:
            continue
        n = len(dense)

        def pose_at(a: float, _dense=dense, _n=n):
            off = _dense[min(max(int(round(a * _n)) - 1, 0), _n - 1)]
            return {s: pin.SE3(T0[s].rotation.copy(), T0[s].translation + off) for s in T0}

        rcfg.update(q0)
        logger.info(f"Trying approach path '{label}' ({n} steps)...")
        planned = plan_dual_pose_path(rcfg, tasks, solver, pose_at, n)
        if planned is None:
            logger.warning(f"  '{label}': no progress")
            continue
        q_start, knots = planned
        Q = np.array([q_start] + knots)
        offsets = []
        for q in Q:
            rcfg.update(q)
            offsets.append(pair_midpoint(rcfg) - mid0)
        rcfg.update(q0)
        residual = float(np.linalg.norm(delta - offsets[-1]))
        logger.info(f"  '{label}': {len(knots)}/{n} steps, residual purple->target {residual:.4f} m")
        if best is None or residual < best[0] - 1e-4:
            best = (residual, label, Q, n, np.array(offsets))
            if residual <= CLOSE_ENOUGH_M:
                break
    rcfg.update(q0)
    return best


def stream_path(bot, Q: np.ndarray, ridx: dict, duration: float, max_joint_vel: float, torso_max_vel: float, label: str) -> None:
    """Stream joint knots ``Q`` for the components in ``ridx`` with the per-joint speed caps."""
    path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
    t_stream = max(duration, max(path[g] / (torso_max_vel if g == "torso" else max_joint_vel) for g in path))
    if t_stream > duration:
        logger.info(f"Stream time stretched to {t_stream:.1f}s by the joint speed caps "
                    f"(arms {max_joint_vel:.2f}, torso {torso_max_vel:.2f} rad/s)")
    groups = {COMPONENT[g]: Q[:, ii] for g, ii in ridx.items()}
    logger.info(f"Executing '{label}' ({', '.join(groups)}) over {t_stream:.1f}s...")
    stream(bot, groups, t_stream, config.CONTROL_DT)


def reduced_from_full(rcfg, configuration, ridx: dict, idx_full: dict) -> None:
    """Copy the synced full-model joints into the reduced model (inverse of ``_pink_arm.apply_reduced``)."""
    q = rcfg.q.copy()
    for g, ii in ridx.items():
        q[ii] = configuration.q[idx_full[g]]
    rcfg.update(q)


def block_tilt_from_normal(n_b: np.ndarray, block_axis_b: np.ndarray) -> float:
    """Signed yaw (rad, +CCW about robot z) that turns ``block_axis_b`` (the block's thickness axis, robot
    frame, horizontal) parallel to the face normal ``n_b``; the smaller of the two solutions (+/-n)."""
    a_axis = float(np.arctan2(block_axis_b[1], block_axis_b[0]))
    a_n = float(np.arctan2(n_b[1], n_b[0]))
    c1 = wrap(a_n - a_axis)
    c2 = wrap(a_n + np.pi - a_axis)
    return c1 if abs(c1) <= abs(c2) else c2


def servo_to_target(
    bot, configuration, idx_full: dict, rcfg, tasks: dict, ridx: dict, solver: str, *,
    tip_path: Path, purple_path: Path, offset, stop_short: float, speed: float, tol: float,
    hz: float, ik_iters: int, max_joint_vel: float, torso_max_vel: float, stall_s: float, timeout_s: float,
    progress_fn=None, align_block: bool = False, normal_fn=None, normal_b: np.ndarray | None = None,
    align_rate: float = np.deg2rad(10.0), align_tol: float = np.deg2rad(1.0), standoff: float = 0.0,
    align_lead_max: float = np.deg2rad(10.0), diverge_m: float = 0.05,
) -> tuple[bool, float]:
    """Closed-loop approach on the live marker files: every tick re-read red and purple (robot base frame),
    step the rigid gripper pair ``speed * dt`` along the remaining delta with a few Pink IK iterations and
    command the joints. Works while the base is turning (both markers are re-read in the current base
    frame) and ends without a separate correction. Returns ``(reached, final_gap_m)``.

    With ``align_block`` the gripper pair is also rotated as a rigid body about robot z so the block's
    thickness axis (robot y at servo start) becomes parallel to the conveyor tip face normal
    (``normal_fn()`` -> unit vector in the *current* base frame, re-evaluated every tick, e.g. the known
    world normal turned by the live base yaw; ``normal_b`` is the fallback), turning at most ``align_rate``.
    The ``standoff`` (metres) is then applied along the *live* normal so the block face ends up flush with
    the conveyor face. Reached = gap < ``tol`` for 5 ticks and (if aligning) tilt < ``align_tol``.

    Safety: the commanded pair yaw never runs more than ``align_lead_max`` ahead of the yaw the grippers have
    actually achieved (anti-windup), and the servo stops at once if the gap grows more than ``diverge_m``
    above its best value (IK running away).

    Stops on: reached, no progress for ``stall_s`` (IK / reach limit, pose held), divergence, or ``timeout_s``.
    """
    dt = 1.0 / hz
    sync_from_robot(bot, configuration, idx_full)
    reduced_from_full(rcfg, configuration, ridx, idx_full)
    T0 = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
    mid0 = 0.5 * (T0["left"].translation + T0["right"].translation)
    R0 = {s: T0[s].rotation.copy() for s in T0}
    d0 = {s: T0[s].translation - mid0 for s in T0}  # rigid pair geometry, kept constant
    phi = 0.0  # commanded pair yaw about robot z (rad); driven by the *measured* tilt
    block_axis0 = np.array([0.0, 1.0, 0.0])  # block thickness axis at servo start (robot y)
    caps = {g: (torso_max_vel if g == "torso" else max_joint_vel) for g in ridx}
    t0 = time.monotonic()
    best = None
    best_gap = None
    t_progress = t0
    done_ticks = 0
    last_log = 0.0
    gap = float("nan")
    tilt = 0.0
    while True:
        now = time.monotonic()
        try:
            red_b = _quiet_xyz(tip_path)
            pur_b = _quiet_xyz(purple_path)
        except (ValueError, OSError):
            time.sleep(dt)
            continue
        n_live = normal_b
        if (align_block or standoff > 0) and normal_fn is not None:
            try:
                nn = normal_fn()
                if nn is not None and np.linalg.norm(nn) > 1e-6:
                    n_live = np.asarray(nn, dtype=float) / np.linalg.norm(nn)
            except (FileNotFoundError, ValueError, OSError):
                pass
        off_live = np.asarray(offset, dtype=float) + (standoff * n_live if (standoff > 0 and n_live is not None) else 0.0)
        target, delta = approach_target(red_b, pur_b, off_live, stop_short, 1.0)
        gap = float(np.linalg.norm(delta))
        # measured state (also gives the measured block orientation for the alignment loop)
        sync_from_robot(bot, configuration, idx_full)
        reduced_from_full(rcfg, configuration, ridx, idx_full)
        q_meas = rcfg.q.copy()
        T = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
        mid = 0.5 * (T["left"].translation + T["right"].translation)
        aligning = align_block and n_live is not None
        if aligning:
            axis_now = (T["left"].rotation @ R0["left"].T) @ block_axis0  # block thickness axis, measured
            axis_now[2] = 0.0
            tilt = block_tilt_from_normal(n_live, axis_now / max(np.linalg.norm(axis_now), 1e-9))
        if progress_fn is not None:
            progress_fn(gap)
        pos_ok = gap < tol
        rot_ok = (not aligning) or abs(tilt) < align_tol
        if pos_ok and rot_ok:
            done_ticks += 1
            if done_ticks >= 5:
                logger.info(f"Servo: purple on target (gap {gap * 1000:.1f} mm"
                            + (f", block tilt {np.rad2deg(tilt):.1f} deg" if aligning else "") + ") -- done.")
                return True, gap
        else:
            done_ticks = 0
        best_gap = gap if best_gap is None else min(best_gap, gap)
        if gap > best_gap + diverge_m:
            logger.error(f"Servo: gap grew from {best_gap * 100:.1f} to {gap * 100:.1f} cm (IK running away); stopping here, pose held.")
            return False, gap
        score = gap + (0.3 * abs(tilt) if aligning else 0.0)  # metres + rad at a 0.3 m lever
        if best is None or score < best - 0.002:
            best, t_progress = score, now
        elif now - t_progress > stall_s:
            logger.warning(f"Servo: no progress for {stall_s:.0f}s (gap {gap * 100:.1f} cm"
                           + (f", tilt {np.rad2deg(tilt):.1f} deg" if aligning else "") + "); IK / reach limit -- stopping here, pose held.")
            return False, gap
        if now - t0 > timeout_s:
            logger.warning(f"Servo: timeout after {timeout_s:.0f}s (gap {gap * 100:.1f} cm); stopping, pose held.")
            return False, gap
        step = delta / gap * min(speed * dt, gap) if (gap > 1e-9 and not pos_ok) else np.zeros(3)
        if aligning and not rot_ok:
            phi_meas = float(pin.log3(T["left"].rotation @ R0["left"].T)[2])  # yaw the grippers actually achieved
            if abs(phi - phi_meas) < align_lead_max:  # anti-windup: advance the command only while the arms keep up
                phi += float(np.clip(tilt, -align_rate * dt, align_rate * dt))
        Rphi = pin.exp3(np.array([0.0, 0.0, phi]))
        for s in ("left", "right"):
            tasks[s].set_target(pin.SE3(Rphi @ R0[s], mid + step + Rphi @ d0[s]))
        tasks["posture"].set_target(q_meas)
        for _ in range(ik_iters):
            v = pink.solve_ik(rcfg, tasks.values(), config.IK_DT, solver=solver)
            rcfg.update(pin.integrate(rcfg.model, rcfg.q, v * config.IK_DT))
        q_cmd = rcfg.q.copy()
        cmd = {}
        for g, ii in ridx.items():
            dq = np.clip(q_cmd[ii] - q_meas[ii], -caps[g] * dt, caps[g] * dt)
            cmd[COMPONENT[g]] = q_meas[ii] + dq
        bot.set_joint_pos(cmd)
        if now - last_log > 1.0:
            logger.info(f"  servo: gap {gap * 100:.1f} cm, delta {np.round(delta, 3)} (robot base), purple {np.round(pur_b, 3)}"
                        + (f", block tilt {np.rad2deg(tilt):+.1f} deg (pair yaw cmd {np.rad2deg(phi):+.1f})" if aligning else ""))
            last_log = now
        time.sleep(dt)


def pair_midpoint(rcfg) -> np.ndarray:
    p = [rcfg.get_transform_frame_to_world(EE[s]).translation for s in ("left", "right")]
    return 0.5 * (p[0] + p[1])


def main(
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stop_short: float = 0.0,
    scale: float = 1.0,
    step_m: float = 0.01,
    duration: float = 6.0,
    max_age_s: float = 3.0,
    use_torso: bool = True,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    max_joint_vel: float = 0.3,
    torso_max_vel: float = TORSO_MAX_VEL,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
    align_mode: str = "none",
    concurrent: bool = True,
    approach_start: float = 0.5,
    min_clearance: float = 0.01,
    correction: bool = True,
    servo: bool = True,
    servo_speed: float = 0.10,
    servo_tol: float = 0.005,
    servo_hz: float = 20.0,
    servo_ik_iters: int = 15,
    servo_stall_s: float = 4.0,
    align_block: bool = True,
    align_rate_deg_s: float = 10.0,
    align_block_tol_deg: float = 1.0,
    standoff: float | None = None,
    face_margin: float = 0.005,
    yaw_short_deg: float = 2.0,
    block_half_fwd: float = 0.20,
    face_normal_world: tuple[float, float, float] = TIP_FACE_NORMAL_WORLD,
    block_half_lat: float = 0.0525,
    face_half_width: float = 0.30,
    crab_short: float = 0.02,
    align_tol_m: float = 0.015,
    align_tol_deg: float = 2.0,
    kp: float = 2.0,
    kp_v: float = 3.0,
    v_max: float = 0.5,
    v_min: float = 0.38,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    chassis_hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    turn_timeout_sim: float = 12.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
) -> bool:
    """Move the base sideways (or turn) just enough to line red up with purple, then translate the grasp onto red.

    Args:
        tip_file: Red tip XYZ file (robot base). Default: the running sim's ``card_box_tip_pos.txt``.
        purple_file: Purple centre XYZ file (robot base). Default: the running sim's ``card_box_mass_center_pos.txt``.
        offset: Robot-base XYZ added to the red target (metres).
        stop_short: Stop this far before the target along the approach line (metres).
        scale: Fraction of the remaining delta to apply (1.0 = all the way).
        step_m: Waypoint spacing along the candidate paths (metres).
        duration: Nominal stream time for the executed path (seconds); stretched by the speed caps if needed.
        max_age_s: Marker files older than this are rejected (sim not running).
        use_torso: Free the torso in the IK (robot base frame). False = arms only.
        torso_posture_cost: Posture cost on the torso joints; higher = torso moves less (arms use 1e-3).
        max_joint_vel: Arm joint speed cap (rad/s) for the streamed path.
        torso_max_vel: Torso joint speed cap (rad/s).
        plan_only: Connect, plan from the measured state, report the reachable residual, move nothing.
        leave_held: Exit without ``stop()`` so the grasp stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
        align_mode: "none" (default, 2026-09-26 user request: no wheel motion, arms + torso only); "yaw" = turn in
            place toward the target's bearing (minus yaw_short_deg); "crab" = translate the
            base sideways instead (no rotation, slower); "none" = skip the base stage.
        concurrent: Yaw mode: run the arm approach *while* the base turns (arms aim at the predicted post-turn
            target, then a short correction from the re-read markers). Falls back to turn-then-approach when
            the combined sweep would bring the block closer than ``min_clearance`` to the conveyor face.
        approach_start: Concurrent mode: fraction of the base turn completed before the arms start (0 = together,
            0.5 = halfway, 1 = after the turn).
        min_clearance: Concurrent mode: minimum predicted block-to-face clearance (metres) over the combined motion.
        correction: Planned-path mode only: after a concurrent run, re-read the markers and close the remaining gap.
        servo: Approach by closed-loop servoing on the live marker files (one continuous arm motion that ends
            exactly on the target, also while the base is still turning). False = pre-planned path (+ correction).
        servo_speed: Servo mode: Cartesian speed of the gripper pair (m/s).
        servo_tol: Servo mode: purple-to-target gap that counts as reached (metres).
        servo_hz: Servo mode: control rate (Hz); each tick re-reads the markers and runs servo_ik_iters IK iterations.
        servo_ik_iters: Servo mode: Pink IK iterations per tick.
        servo_stall_s: Servo mode: stop when the gap has not shrunk for this long (reach / IK limit).
        align_block: Servo mode: also rotate the gripper pair about robot z so the block's thickness axis is parallel
            to the conveyor face normal (block placed square to the box). False = keep the current orientation.
        align_rate_deg_s: Servo mode: yaw rate cap for that alignment (deg/s).
        align_block_tol_deg: Servo mode: tilt below this counts as aligned (degrees).
        standoff: Distance of the purple centre from the conveyor face along its normal at the end (metres).
            None (default) = automatic "flush" placement: block half thickness + face_margin when align_block is on
            (the block face touches the conveyor face, edges aligned), else block half thickness * cos(theta) +
            block half length * sin(theta) + face_margin for the predicted tilt theta. 0 = purple exactly on red
            (the held block then overlaps the solid conveyor box in the sim).
        face_margin: Extra clearance added by the automatic standoff (metres).
        yaw_short_deg: Yaw mode: stop the turn this many degrees before the target bearing; the arms take the
            remaining sideways gap (a few cm at most). Keeps the swinging box clear of the conveyor face.
        block_half_fwd: Held block half extent along robot x (metres), for the clearance prediction only.
        face_normal_world: Outward normal of the conveyor tip face in the world (default +X, from the scene config);
            turned into the base frame with the base yaw fitted from the world-fixed markers.
        block_half_lat: Held block half extent along robot y (metres), for the clearance prediction only.
        face_half_width: Half width of the conveyor tip face along the face used by the clearance prediction (metres).
            The real face is 0.20; the default 0.30 is deliberately wider so block corners just beside the face edge
            still count (a small misalignment would put them in front of it).
        crab_short: Crab mode: stop the base this far before the sideways target and leave the rest to the arms
            (metres); an overshoot of the base then cannot push the block into the conveyor face.
        align_tol_m: Crab mode: stop when the sideways (robot y) gap to the (shortened) target is below this (metres).
        align_tol_deg: Yaw mode: stop when the bearing error is below this (degrees).
        kp: Yaw mode P gain, wz = kp * error (rad/s per rad, sim time).
        kp_v: Crab mode P gain, vy = kp_v * error (m/s per m, sim time).
        v_max: Crab mode speed cap (m/s, sim time).
        v_min: Crab mode minimum |vy| while moving (sim: keeps the wheel joint target above the drive dead zone).
        wz_max: Base yaw-rate cap (rad/s, sim time).
        wz_min: Minimum |wz| while turning (sim: keeps the wheel joint target above the drive dead zone).
        chassis_hz: Chassis command + feedback rate (wall clock).
        steer_tol_deg: Start driving once both wheels are within this of the in-place steering angles.
        steer_timeout_sim: Max sim seconds to wait for the steering.
        turn_timeout_sim: Give up on the base turn after this many sim seconds of driving.
        stall_s: Raise the wheel command if the heading has not changed for this many wall seconds.
        wheel_gain: Multiplier on the wheel-speed command (sim default undoes the scale=100; real robot 1.0).
    """
    sim_dir = find_sim_dir()
    tip_path = Path(tip_file) if tip_file is not None else sim_dir / TIP_FILE_NAME
    purple_path = Path(purple_file) if purple_file is not None else sim_dir / PURPLE_FILE_NAME
    try:
        red = load_xyz_base(tip_path, "red tip", max_age_s)
        purple = load_xyz_base(purple_path, "purple centre", max_age_s)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False

    if align_mode not in ("crab", "yaw", "none"):
        raise ValueError("align_mode must be 'crab', 'yaw' or 'none'")
    base_offset = np.asarray(offset, dtype=float)
    normal = face_normal_base(sim_dir, face_normal_world, max_age_s) if (standoff is None or standoff > 0) else np.zeros(3)

    def live_normal() -> np.ndarray | None:
        """Face normal in the *current* base frame (the base may be turning)."""
        n = world_dir_to_base(sim_dir, face_normal_world, max_age_s)
        n[2] = 0.0
        return n
    if standoff is None and align_block and servo:
        standoff = float(block_half_lat + face_margin)
        logger.info(f"Automatic flush standoff {standoff * 100:.1f} cm (block half thickness {block_half_lat * 100:.1f} cm + margin "
                    f"{face_margin * 100:.1f} cm); the servo also squares the block to the conveyor face.")
    if standoff is None:
        # Automatic standoff: the block (half_fwd along robot x, half_lat along robot y) ends up tilted by theta
        # relative to the face when the base stops yaw_short_deg short of full alignment (yaw mode). The
        # nearest corner then sits half_lat*cos(theta) + half_fwd*sin(theta) beyond the centre toward the face.
        # The tilt depends on the turn, which depends on the target, so iterate a few times.
        so = block_half_lat + face_margin
        theta = 0.0
        for _ in range(4):
            tgt_full = red + base_offset + so * normal
            yaw_full = bearing_yaw(purple, tgt_full)
            yaw_g = float(np.sign(yaw_full) * max(abs(yaw_full) - np.deg2rad(yaw_short_deg), 0.0)) if align_mode == "yaw" else 0.0
            n_after = rot_z(normal, -yaw_g)  # face normal in the base frame after the turn
            theta = float(np.arctan2(abs(n_after[0]), abs(n_after[1])))  # tilt of the block x-axis vs the face plane
            so = block_half_lat * np.cos(theta) + block_half_fwd * np.sin(theta) + face_margin
        standoff = float(so)
        logger.info(f"Automatic standoff {standoff * 100:.1f} cm: block tilt vs face after the turn {np.rad2deg(theta):.1f} deg "
                    f"(half thickness {block_half_lat * 100:.1f} cm, half length {block_half_fwd * 100:.0f} cm, margin {face_margin * 100:.1f} cm)")
    offset = tuple(base_offset + standoff * normal)
    if standoff > 0:
        logger.info(f"Standoff {standoff * 100:.1f} cm along the tip face normal {np.round(normal, 3)} (robot base) "
                    f"-> effective target offset {np.round(offset, 3)}")
    if float(np.linalg.norm(red + np.asarray(offset, dtype=float) - purple)) < 1e-4:
        logger.info("Purple already on the target (within 0.1 mm); nothing to do.")
        return True

    def describe(tag: str, purple_: np.ndarray, red_: np.ndarray, target_: np.ndarray, delta_: np.ndarray) -> None:
        logger.info(
            f"{tag} (robot base):\n"
            f"  purple  {np.round(purple_, 4)}\n"
            f"  red     {np.round(red_, 4)}"
            + (f"  +offset {np.round(offset, 3)}" if np.any(np.asarray(offset)) else "")
            + (f"  stop_short {stop_short:.3f}" if stop_short > 0 else "")
            + (f"  scale {scale:g}" if scale != 1.0 else "")
            + f"\n  target  {np.round(target_, 4)}\n"
            f"  delta   {np.round(delta_, 4)}  |d| = {np.linalg.norm(delta_):.4f} m"
        )

    target, delta = approach_target(red, purple, offset, stop_short, scale)
    describe("Purple -> red", purple, red, target, delta)

    # Stage 0: close the sideways gap with the base. The torso joints are all pitch and the arms cannot
    # shift a two-handed grasp far sideways, so this is the only way; crab = pure translation along
    # robot y (the held box stays parallel to the conveyor face), yaw = in-place turn by the bearing
    # difference (swings the box; can hit the conveyor box when it is held far ahead).
    red_target_full = red + np.asarray(offset, dtype=float)
    lateral_needed = float(red_target_full[1] - purple[1])
    yaw_needed = bearing_yaw(purple, red_target_full)
    move_base = False
    crab_goal = float(np.sign(lateral_needed) * max(abs(lateral_needed) - crab_short, 0.0))  # base moves this much
    if align_mode == "crab":
        move_base = abs(crab_goal) >= align_tol_m
        logger.info(
            f"Target is {lateral_needed * 100:+.1f} cm sideways (robot y, +left) of purple"
            + (f": base translates {abs(crab_goal) * 100:.1f} cm {'left' if crab_goal > 0 else 'right'} (no rotation), "
               f"stopping {crab_short * 100:.0f} cm short for the arms to finish."
               if move_base else f" (< {(align_tol_m + crab_short) * 100:.1f} cm): no base move.")
        )
        if move_base and plan_only:
            red_pred = red - np.array([0.0, crab_goal, 0.0])
            target, delta = approach_target(red_pred, purple, offset, stop_short, scale)
            describe(f"Predicted after a {crab_goal * 100:+.1f} cm sideways base move", purple, red_pred, target, delta)
    elif align_mode == "yaw":
        yaw_short = np.deg2rad(yaw_short_deg)
        yaw_goal = float(np.sign(yaw_needed) * max(abs(yaw_needed) - yaw_short, 0.0))
        move_base = abs(yaw_goal) >= np.deg2rad(align_tol_deg)
        logger.info(
            f"Target is {lateral_needed * 100:+.1f} cm sideways of purple: full alignment needs a base yaw of "
            f"{np.rad2deg(yaw_needed):+.1f} deg ({'CCW / left' if yaw_needed > 0 else 'CW / right'}); turning "
            f"{np.rad2deg(yaw_goal):+.1f} deg ({yaw_short_deg:.0f} deg short, the arms take the rest)"
            if move_base else
            f"Target is {lateral_needed * 100:+.1f} cm sideways of purple (yaw {np.rad2deg(yaw_needed):+.1f} deg, "
            f"below {yaw_short_deg:.0f} + {align_tol_deg:.0f} deg): no base turn."
        )
        for label_c, yaw_c in (("after the planned turn", yaw_goal), ("if turned all the way", yaw_needed)):
            clr = predict_face_clearance(tip_path, face_normal_world, purple_path, max_age_s, yaw_c,
                                         block_half_fwd, block_half_lat, face_half_width)
            if clr is None:
                logger.info(f"Clearance prediction {label_c}: n/a (block corners miss the face or world lines missing)")
            else:
                (logger.warning if clr < 0.01 else logger.info)(
                    f"Predicted block-to-conveyor-face clearance {label_c} ({np.rad2deg(yaw_c):+.1f} deg): {clr * 100:+.1f} cm"
                    + (" -- CONTACT / penetration, increase --yaw-short-deg or --standoff" if clr < 0.01 else "")
                )
        if move_base and plan_only:
            red_pred = rot_z(red, -yaw_goal)
            target, delta = approach_target(red_pred, purple, offset, stop_short, scale)
            describe(f"Predicted after a {np.rad2deg(yaw_goal):+.1f} deg base turn", purple, red_pred, target, delta)
    else:
        logger.info(f"Base stage skipped (--align-mode none); sideways gap {lateral_needed * 100:+.1f} cm is left to the arms.")

    def bearing_error() -> tuple[float, str]:
        red_now = _quiet_xyz(tip_path)
        purple_now = _quiet_xyz(purple_path)
        full = bearing_yaw(purple_now, red_now + np.asarray(offset, dtype=float))
        e = float(np.sign(full) * max(abs(full) - np.deg2rad(yaw_short_deg), 0.0))  # stop yaw_short before alignment
        return e, f"bearing gap {np.rad2deg(full):+.1f} deg; red@base ({red_now[0]:+.2f}, {red_now[1]:+.2f}), purple@base ({purple_now[0]:+.2f}, {purple_now[1]:+.2f})"

    def lateral_error() -> tuple[float, str]:
        red_now = _quiet_xyz(tip_path)
        purple_now = _quiet_xyz(purple_path)
        full = float(red_now[1] + offset[1] - purple_now[1])
        e = float(np.sign(full) * max(abs(full) - crab_short, 0.0))  # stop crab_short before the target
        return e, f"red@base ({red_now[0]:+.2f}, {red_now[1]:+.2f}), purple@base ({purple_now[0]:+.2f}, {purple_now[1]:+.2f})"

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
    turn_thread = None
    turn_result: dict = {}
    try:
        def run_turn(rtf: float):
            try:
                turn_result["res"] = turn_in_place(
                    bot.chassis, bearing_error, tol_deg=align_tol_deg, kp=kp, wz_max=wz_max, wz_min=wz_min,
                    hz=chassis_hz, steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim,
                    timeout_sim=turn_timeout_sim, stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                )
            except BaseException as exc:  # noqa: BLE001 - report and stop the chassis
                turn_result["exc"] = exc
                try:
                    bot.chassis.stop()
                except Exception:
                    pass

        def finish(purple_ref: np.ndarray, mid_ref: np.ndarray, target_ref: np.ndarray) -> bool:
            sync_from_robot(bot, configuration, idx)
            rcfg2, _t2, _i2 = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
            purple_est = purple_ref + (pair_midpoint(rcfg2) - mid_ref)
            err = float(np.linalg.norm(purple_est - target_ref))
            logger.info(
                f"Done. purple (FK estimate) {np.round(purple_est, 4)}, target {np.round(target_ref, 4)}, "
                f"err {err:.4f} m. Check {purple_path.name} for the sim-measured value. Pose left held."
            )
            return err <= CLOSE_ENOUGH_M

        use_concurrent = bool(concurrent and move_base and align_mode == "yaw")
        if use_concurrent:
            if not 0.0 <= approach_start <= 1.0:
                raise ValueError("--approach-start must be between 0 and 1")
            # Plan the arms against the predicted post-turn geometry (the held box is fixed in the base
            # frame; red rotates by -yaw_goal about the base origin).
            red_pred = rot_z(red, -yaw_goal)
            target_pred, delta_pred = approach_target(red_pred, purple, offset, stop_short, scale)
            describe(f"Concurrent plan: predicted after the {np.rad2deg(yaw_goal):+.1f} deg turn", purple, red_pred, target_pred, delta_pred)
            sync_from_robot(bot, configuration, idx)
            rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
            mid0 = pair_midpoint(rcfg)
            best = plan_approach(rcfg, tasks, solver, delta_pred, step_m)
            if best is None:
                logger.warning("No arm progress toward the predicted target; falling back to turn-then-approach.")
                use_concurrent = False
            else:
                residual, label, Q, n, offs = best
                # Combined sweep: base at yaw_goal * u, arms at path fraction s(u) (start after approach_start);
                # with the servo's block alignment the block yaw ramps to the post-turn tilt correction.
                yaw_align = block_tilt_from_normal(rot_z(normal, -yaw_goal), np.array([0.0, 1.0, 0.0])) if (align_block and servo and standoff > 0) else 0.0
                worst = None
                for u in np.linspace(0.0, 1.0, 41):
                    s_arm = 0.0 if approach_start >= 1.0 else float(np.clip((u - approach_start) / (1.0 - approach_start), 0.0, 1.0))
                    k = int(round(s_arm * (len(offs) - 1)))
                    clr = predict_face_clearance(tip_path, face_normal_world, purple_path, max_age_s, yaw_goal * u,
                                                 block_half_fwd, block_half_lat, face_half_width, shift_b=offs[k],
                                                 block_yaw=yaw_align * s_arm)
                    if clr is not None:
                        worst = clr if worst is None else min(worst, clr)
                covered = 100.0 * max(0.0, 1.0 - residual / max(float(np.linalg.norm(delta_pred)), 1e-6))
                final_clr = predict_face_clearance(tip_path, face_normal_world, purple_path, max_age_s, yaw_goal,
                                                   block_half_fwd, block_half_lat, face_half_width, shift_b=offs[-1],
                                                   block_yaw=yaw_align)
                if yaw_align:
                    logger.info(f"Servo alignment will yaw the block by {np.rad2deg(yaw_align):+.1f} deg to square it to the face.")
                logger.info(
                    f"Concurrent plan '{label}': {len(Q) - 1}/{n} steps, residual {residual:.4f} m ({covered:.0f}% of the predicted gap); "
                    f"arms start at {approach_start * 100:.0f}% of the turn; predicted block-to-face clearance: minimum over the "
                    f"combined motion {'n/a' if worst is None else f'{worst * 100:+.1f} cm'}, at the final pose "
                    f"{'n/a' if final_clr is None else f'{final_clr * 100:+.1f} cm'}"
                )
                if final_clr is not None and final_clr < 0.0:
                    logger.warning(
                        f"The final pose itself overlaps the conveyor box by {-final_clr * 100:.1f} cm (standoff "
                        f"{standoff * 100:.1f} cm); this is what --standoff 0 asks for. Use --standoff None for an automatic clearance."
                    )
                # Fall back only when the *concurrent* sweep is worse than the final pose by more than 1 cm.
                if worst is not None and worst < min_clearance and (final_clr is None or worst < final_clr - 0.01):
                    logger.warning(
                        f"During the combined motion the block would get {worst * 100:+.1f} cm from the face, worse than the "
                        f"final pose ({'n/a' if final_clr is None else f'{final_clr * 100:+.1f} cm'}); falling back to "
                        "turn-then-approach (or raise --approach-start / --yaw-short-deg)."
                    )
                    use_concurrent = False
            if plan_only:
                logger.info(f"Plan only ({'concurrent' if use_concurrent else 'sequential'}): nothing commanded.")
                return best is not None and best[0] <= CLOSE_ENOUGH_M

        if use_concurrent:
            logger.info("Stage 0+1: turning the base and approaching together (measuring sim speed for 2 s)...")
            rtf = measure_rtf(tip_path)
            e0 = abs(bearing_error()[0])
            turn_thread = threading.Thread(target=run_turn, args=(rtf,), name="base-turn", daemon=True)
            turn_thread.start()
            t_wait0 = time.monotonic()
            while turn_thread.is_alive():
                try:
                    e = abs(bearing_error()[0])
                except (ValueError, OSError):
                    e = e0
                progress = 1.0 - e / max(e0, 1e-6)
                if progress >= approach_start or time.monotonic() - t_wait0 > turn_timeout_sim / rtf + 10.0:
                    break
                time.sleep(0.1)
            logger.info(f"Base turn {min(max(progress, 0.0), 1.0) * 100:.0f}% done; starting the arms.")
            if servo:
                if scale != 1.0:
                    logger.warning("Servo mode ignores --scale (it tracks the live target).")
                gap0 = float(np.linalg.norm(delta_pred))
                reached, gap = servo_to_target(
                    bot, configuration, idx, rcfg, tasks, ridx, solver, tip_path=tip_path, purple_path=purple_path,
                    offset=base_offset, stop_short=stop_short, speed=servo_speed, tol=servo_tol, hz=servo_hz,
                    ik_iters=servo_ik_iters, max_joint_vel=max_joint_vel, torso_max_vel=torso_max_vel,
                    stall_s=servo_stall_s, timeout_s=max(30.0, 3.0 * gap0 / max(servo_speed, 1e-3) + turn_timeout_sim / rtf),
                    align_block=align_block, normal_fn=live_normal, normal_b=normal, align_rate=np.deg2rad(align_rate_deg_s),
                    align_tol=np.deg2rad(align_block_tol_deg), standoff=standoff,
                )
                while turn_thread.is_alive():
                    turn_thread.join(timeout=0.5)
                if "exc" in turn_result:
                    logger.error(f"Base turn failed: {turn_result['exc']!r}")
                time.sleep(0.3)
                red = load_xyz_base(tip_path, "red tip", max_age_s)
                purple2 = load_xyz_base(purple_path, "purple centre", max_age_s)
                n_end = face_normal_base(sim_dir, face_normal_world, max_age_s) if standoff > 0 else np.zeros(3)
                target, delta = approach_target(red, purple2, base_offset + standoff * n_end, stop_short, 1.0)
                logger.info(f"Servo finished ({'reached' if reached else 'stopped short'}); measured purple {np.round(purple2, 4)}, "
                            f"target {np.round(target, 4)}, gap {np.linalg.norm(delta) * 1000:.1f} mm. Pose left held.")
                if leave_held:
                    disconnect_leave_held(bot)
                    do_shutdown = False
                    logger.info("Client disconnected without stop() -- pose held.")
                return reached
            stream_path(bot, Q, ridx, duration, max_joint_vel, torso_max_vel, label)
            while turn_thread.is_alive():
                turn_thread.join(timeout=0.5)
            if "exc" in turn_result:
                logger.error(f"Base turn failed: {turn_result['exc']!r}")
            time.sleep(0.5)
            red = load_xyz_base(tip_path, "red tip", max_age_s)
            purple2 = load_xyz_base(purple_path, "purple centre", max_age_s)
            target, delta = approach_target(red, purple2, offset, stop_short, scale)
            describe("After the concurrent stage (re-read markers)", purple2, red, target, delta)
            if correction and float(np.linalg.norm(delta)) > CLOSE_ENOUGH_M:
                logger.info("Correction: closing the remaining gap from the re-read markers...")
                sync_from_robot(bot, configuration, idx)
                rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
                mid_c = pair_midpoint(rcfg)
                best_c = plan_approach(rcfg, tasks, solver, delta, step_m)
                if best_c is None:
                    logger.warning("Correction: no progress; leaving the pose as is.")
                else:
                    stream_path(bot, best_c[2], ridx, max(2.0, duration / 2), max_joint_vel, torso_max_vel, f"correction {best_c[1]}")
                ok = finish(purple2, mid_c, target) if best_c is not None else False
            else:
                sync_from_robot(bot, configuration, idx)
                ok = float(np.linalg.norm(delta)) <= CLOSE_ENOUGH_M
                logger.info(f"Remaining purple->target gap {np.linalg.norm(delta):.4f} m; no correction. Pose left held.")
            if leave_held:
                disconnect_leave_held(bot)
                do_shutdown = False
                logger.info("Client disconnected without stop() -- pose held.")
            return ok

        # ---- sequential: base stage, then approach ----
        if move_base and not plan_only:
            logger.info(f"Stage 0: aligning the base ({align_mode}; measuring sim speed for 2 s)...")
            rtf = measure_rtf(tip_path)
            if align_mode == "crab":
                translate_sideways(
                    bot.chassis, lateral_error, tol_m=align_tol_m, kp=kp_v, v_max=v_max, v_min=v_min,
                    hz=chassis_hz, steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim,
                    timeout_sim=turn_timeout_sim, stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                )
            else:
                turn_in_place(
                    bot.chassis, bearing_error, tol_deg=align_tol_deg, kp=kp, wz_max=wz_max, wz_min=wz_min,
                    hz=chassis_hz, steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim,
                    timeout_sim=turn_timeout_sim, stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                )
            time.sleep(0.5)
            red = load_xyz_base(tip_path, "red tip", max_age_s)
            purple = load_xyz_base(purple_path, "purple centre", max_age_s)
            target, delta = approach_target(red, purple, offset, stop_short, scale)
            describe("After the base move", purple, red, target, delta)
        logger.info("Stage 1: approach (arms" + (" + torso" if use_torso else "") + ")")
        sync_from_robot(bot, configuration, idx)
        rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        mid0 = pair_midpoint(rcfg)
        best = plan_approach(rcfg, tasks, solver, delta, step_m)
        if best is None:
            logger.error("No progress toward the red tip on any path. Not moving.")
            do_shutdown = True
            return False
        residual, label, Q, n, offs = best
        predicted = purple + offs[-1]
        covered = 100.0 * max(0.0, 1.0 - residual / max(float(np.linalg.norm(delta)), 1e-6))
        verdict = "REACHABLE" if residual <= CLOSE_ENOUGH_M else "PARTIAL"
        torso_note = ""
        if use_torso:
            torso_note = f"; torso change (j1, j2, j3) {np.round(np.rad2deg(Q[-1, ridx['torso']] - Q[0, ridx['torso']]), 1)} deg"
        logger.info(
            f"[{verdict}] path '{label}': {len(Q) - 1}/{n} steps; predicted purple after motion "
            f"{np.round(predicted, 4)}, residual to target {residual:.4f} m ({covered:.0f}% of the gap){torso_note}"
        )
        yaw_for_clr = yaw_goal if (move_base and align_mode == "yaw" and plan_only) else 0.0
        yaw_align_seq = block_tilt_from_normal(rot_z(normal, -yaw_for_clr), np.array([0.0, 1.0, 0.0])) if (align_block and servo and standoff > 0) else 0.0
        clr_final = predict_face_clearance(tip_path, face_normal_world, purple_path, max_age_s, yaw_for_clr,
                                           block_half_fwd, block_half_lat, face_half_width, shift_b=offs[-1],
                                           block_yaw=yaw_align_seq)
        if clr_final is not None:
            (logger.warning if clr_final < min_clearance else logger.info)(
                f"Predicted block-to-face clearance at the final approach pose: {clr_final * 100:+.1f} cm"
                + (" -- too close, raise --standoff / --face-margin" if clr_final < min_clearance else ""))
        if plan_only:
            logger.info("Plan only: nothing commanded." + (" (execution will servo on the live markers)" if servo else ""))
            return residual <= CLOSE_ENOUGH_M
        if servo:
            reached, gap = servo_to_target(
                bot, configuration, idx, rcfg, tasks, ridx, solver, tip_path=tip_path, purple_path=purple_path,
                offset=base_offset, stop_short=stop_short, speed=servo_speed, tol=servo_tol, hz=servo_hz,
                ik_iters=servo_ik_iters, max_joint_vel=max_joint_vel, torso_max_vel=torso_max_vel,
                stall_s=servo_stall_s, timeout_s=max(30.0, 3.0 * float(np.linalg.norm(delta)) / max(servo_speed, 1e-3)),
                align_block=align_block, normal_fn=live_normal, normal_b=normal, align_rate=np.deg2rad(align_rate_deg_s),
                align_tol=np.deg2rad(align_block_tol_deg), standoff=standoff,
            )
            logger.info(f"Servo finished ({'reached' if reached else 'stopped short'}), gap {gap * 1000:.1f} mm. Pose left held.")
            ok = reached
        else:
            stream_path(bot, Q, ridx, duration, max_joint_vel, torso_max_vel, label)
            ok = finish(purple, mid0, target)
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return ok
    except KeyboardInterrupt:
        logger.warning("Interrupted -- stopping the chassis, leaving the last commanded pose held.")
        try:
            bot.chassis.stop()
        except Exception:
            pass
        do_shutdown = True
        return False
    finally:
        if turn_thread is not None and turn_thread.is_alive():
            try:
                bot.chassis.stop()
            except Exception:
                pass
            turn_thread.join(timeout=2.0)
        if bot is not None and (do_shutdown or plan_only):
            if plan_only:
                disconnect_leave_held(bot)
            else:
                logger.info("Shutting down robot")
                bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
