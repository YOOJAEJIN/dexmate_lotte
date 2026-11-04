# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Bimanual: carry the purple box centre to its standoff next to the red conveyor tip (Pink IK version).

Port of ``lotte_chilsung_2/6.approach_purple_to_red_bimanual.py``. The box is held with both hands;
the grasp is kept rigid (relative L/R gripper poses and orientations fixed) and both grippers are
translated by

    delta = target - purple_centre        (robot base frame)

so the purple marker lands on the target next to the red marker. The purple ball follows the centre of
``card_block_35`` (sim_main.update_card_box_mass_center_marker), which moves rigidly with the grasp, so
translating the gripper pair by ``delta`` moves the purple centre onto the target.

IK: both arms + torso free in a reduced model with absolute gripper targets in the robot base frame
(``_pink_arm.build_dual_arm_ik``; ``--no-use-torso`` = arms only). Like the original, several
translation paths are tried -- straight line and axis-ordered detours (z then xy, xy then z, x-y-z,
z-y-x) -- each planned as far as IK / joint limits allow, and the one that leaves the smallest
purple-to-target residual is executed (the largest reachable prefix if none completes). Arms and torso
are streamed together with ``_pink_arm.stream`` (leash-paced, per-joint speed caps). There is no
self-collision model here; the tracking leash and the speed caps are the only safety nets.

Sim must be writing (robot base frame, see ``_sim_markers.py``)::

    unitree_sim_isaaclab*/card_box_tip_pos.txt          (red)
    unitree_sim_isaaclab*/card_box_mass_center_pos.txt  (purple)

Target (2026-09-26): red + (first-contact distance + ``--standoff`` air) along the box face normal, so the block
ends right next to the box; the red marker itself lies inside it. The first-contact distance is computed from the
block's yaw relative to the box face (both orientations come from the marker files, ``_block_geom``): square block
-> the gripper finger plate between the boxes touches first (half thickness 5.25 cm + finger 2.5 cm = 7.75 cm from
red to purple); yawed block -> its leading corner touches first (5.25 cos(yaw) + 20 sin(yaw) cm). The boxes
themselves can never be flush where the finger sits; ``--contact-dist`` overrides the computed value.

Stage 0 (added 2026-09-26, ``--align-base``): before the arms move, the base turns in place by the bearing
difference between purple and that target (closed loop on the live marker files, capped at
``--yaw-cap-deg``, minus ``--yaw-short-deg``), so the remaining delta is mostly forward/down -- the torso
joints are all pitch and the arms cannot shift a two-handed grasp far sideways. Nothing else moves during
the turn; the markers are re-read afterwards. ``--no-align-base`` = the plain 2026-09-25 behaviour.
After the wheels stop the sim base keeps unwinding for a few seconds (2026-09-26 20:09: +1.8 deg residual grew
to +3.7 deg, i.e. 2.5 cm sideways at the block), so the script waits until the bearing is still
(``--settle-s``). ``--max-turns 2`` would then turn once more aiming past the target by the miss just measured;
since 2026-09-28 the default is 1 (no re-turn): the controller stops ~tol short and the unwinding pushed the
first turn over the same 2 deg tolerance nearly every run, so the second turn cost ~11 s wall for a few cm the
arms can absorb.

Stage 1 (the 2026-09-25 move, unchanged): arms + torso translate the rigid grasp along the best candidate path
to the standoff target and the pose is left held.

The final push against the box is no longer part of this script (split out 2026-09-27). The sequence continues
with the right hand let go and swung up out of the way (``11.open_hands.py``, ``12.hands_back.py``)
and ``13.hands_push_left.py`` pushing the block left onto the box with both open hands, stopped by the wrist F/T
sensors. (Until 2026-09-26 that push was Stage 2 here, with both hands; see ``_approach_prev_20260927.py``.)

Example (box held with both hands, base already turned toward the conveyor)::

    python 10.approach_purple_to_red_bimanual.py --plan-only        # feasibility + residual, no motion
    python 10.approach_purple_to_red_bimanual.py                    # execute
    python 10.approach_purple_to_red_bimanual.py --stop-short 0.06  # stop 6 cm before the target
    python 10.approach_purple_to_red_bimanual.py --offset 0 0.05 0  # target + robot-base offset
    python 10.approach_purple_to_red_bimanual.py --scale 0.5        # close half the gap
    python 10.approach_purple_to_red_bimanual.py --no-use-torso     # arms only
    python 10.approach_purple_to_red_bimanual.py --no-align-base    # no base turn first
    python 10.approach_purple_to_red_bimanual.py --standoff 0.01    # leave 1 cm of air instead of 5 mm
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _block_geom import block_yaw_vs_face, first_contact_along  # noqa: E402
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
from _base_turn import WHEEL_GAIN_SIM, measure_rtf, turn_in_place, wrap  # noqa: E402
from _pink_common import EE, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402
from _sim_markers import (  # noqa: E402
    PURPLE_FILE_NAME,
    TIP_FILE_NAME,
    base_yaw_world,
    box_face_normal_base,
    find_sim_dir,
    load_xyz_base,
    quiet_xyz,
    world_fixed_pairs,
)

CLOSE_ENOUGH_M = 0.01  # residual below this counts as reached


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


def pair_midpoint(rcfg) -> np.ndarray:
    p = [rcfg.get_transform_frame_to_world(EE[s]).translation for s in ("left", "right")]
    return 0.5 * (p[0] + p[1])


def rot_z(p: np.ndarray, yaw: float) -> np.ndarray:
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([c * p[0] - s_ * p[1], s_ * p[0] + c * p[1], p[2]])


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
    standoff: float = 0.005,
    contact_dist: float | None = None,
    face_normal_world: tuple[float, float, float] = (1.0, 0.0, 0.0),
    align_base: bool = True,
    yaw_tol_deg: float = 2.0,
    yaw_cap_deg: float = 20.0,
    yaw_short_deg: float = 0.0,
    kp: float = 2.0,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    chassis_hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    turn_timeout_sim: float = 12.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
    settle_s: float = 8.0,
    max_turns: int = 1,
) -> bool:
    """Turn the base a little toward the target if needed, then translate the bimanual grasp so the purple box
    centre reaches the standoff next to the red conveyor tip. The pose is left held; the push against the box
    is ``13.hands_push_left.py`` (after ``11.open_hands.py`` and ``12.hands_back.py``).

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
        standoff: Air left between the block and its first contact with the box (metres): target = red + (first-contact
            distance + standoff) along the face normal. The first-contact distance follows the block's yaw relative
            to the box (finger plate 7.75 cm when square, the leading corner when yawed); step 12 closes the air.
        contact_dist: Override the computed first-contact distance (metres, purple-to-red along the normal).
        face_normal_world: Outward normal of the box face in the world (+X from the scene); turned into the base
            frame with the base yaw fitted from the world-fixed markers.
        align_base: Stage 0: turn the base in place by the purple-to-red bearing difference before the arms move
            (closed loop on the live markers). False = arms + torso only.
        yaw_tol_deg: Skip the turn / stop turning when the bearing gap is below this (degrees).
        yaw_cap_deg: Never turn more than this in one run (degrees); the rest is left to the arms.
        yaw_short_deg: Stop the turn this many degrees before the bearing (0 = line purple up with red).
        kp: Turn P gain (rad/s per rad, sim time).
        wz_max: Turn yaw-rate cap (rad/s, sim time).
        wz_min: Minimum |wz| while turning (sim drive dead zone: >= 1.2 or the base does not move).
        chassis_hz: Chassis command rate (Hz).
        steer_tol_deg: Start driving once both steer wheels are within this of the in-place angles.
        steer_timeout_sim: Max sim seconds to wait for the steering.
        turn_timeout_sim: Give up on the turn after this many sim seconds.
        stall_s: Raise the wheel command when the heading has not changed for this long (seconds).
        wheel_gain: Wheel command multiplier (sim: cancels the chassis_drive_vel scale). Real robot: 1.0.
        settle_s: After each turn wait (at most this long) until the bearing stops drifting -- the sim base unwinds
            a degree or two after the wheels stop -- before re-reading the markers.
        max_turns: Turn again (aiming past the target by the miss just measured) while the settled bearing gap is
            still above ``yaw_tol_deg``, up to this many turns in total. Default 1 = never re-turn (2026-09-28);
            the leftover bearing gap is left to the arms.
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

    def read_pair(retries: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Red and purple (robot base); the sim rewrites the files every step, so an empty read is retried."""
        for i in range(retries):
            try:
                return load_xyz_base(tip_path, "red tip", max_age_s), load_xyz_base(purple_path, "purple centre", max_age_s)
            except (ValueError, OSError):
                if i == retries - 1:
                    raise
                time.sleep(0.05)
        raise RuntimeError("unreachable")

    def base_yaw_deg() -> float | None:
        """Base yaw in the world (deg) from the world-fixed markers, for drift diagnostics; None if unavailable."""
        try:
            yaw, _span = base_yaw_world(world_fixed_pairs(sim_dir, max_age_s, include_purple=False))
            return float(np.rad2deg(yaw))
        except (ValueError, FileNotFoundError, OSError):
            return None

    def face_normal() -> np.ndarray:
        """Unit box face normal in the current base frame (horizontal); zero vector if the markers cannot tell
        (``_sim_markers.box_face_normal_base``: box quaternion in the tip file, else the world normal turned by the
        fitted base yaw)."""
        return box_face_normal_base(tip_path, sim_dir, face_normal_world, max_age_s)


    def contact_geom(yaw: float | None) -> tuple[float, str]:
        """(first-contact distance along the normal, what touches) for a block yaw; the override wins."""
        if contact_dist is not None:
            return float(contact_dist), "override"
        return first_contact_along(0.0 if yaw is None else yaw)

    geom = {"yaw": block_yaw_vs_face(tip_path, purple_path)}
    geom["contact"], geom["what"] = contact_geom(geom["yaw"])
    geom["total"] = geom["contact"] + max(float(standoff), 0.0)  # red -> target along the normal

    def refresh_geom() -> None:
        geom["yaw"] = block_yaw_vs_face(tip_path, purple_path)
        geom["contact"], geom["what"] = contact_geom(geom["yaw"])
        geom["total"] = geom["contact"] + max(float(standoff), 0.0)

    def geom_txt() -> str:
        yaw_txt = f"{np.rad2deg(geom['yaw']):+.1f} deg" if geom["yaw"] is not None else "unknown (block not upright?)"
        return (f"block yaw vs the box face {yaw_txt} -> first contact by the {geom['what']} at {geom['contact'] * 100:.2f} cm "
                f"(purple to red along the normal) + {standoff * 100:.1f} cm air = {geom['total'] * 100:.2f} cm")

    def make_target(red_b: np.ndarray, purple_b: np.ndarray, n_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tgt = red_b + np.asarray(offset, dtype=float) + geom["total"] * n_b
        full_ = tgt - purple_b
        dist_ = float(np.linalg.norm(full_))
        if stop_short > 0.0 and dist_ > 1e-6:
            tgt = tgt - min(stop_short, 0.95 * dist_) * full_ / dist_
        tgt = purple_b + float(scale) * (tgt - purple_b)
        return tgt, tgt - purple_b

    n_face = face_normal()
    if not np.any(n_face):
        logger.warning("Face normal unavailable from the markers; the target is the red marker itself (inside the box!).")
    else:
        logger.info("Geometry: " + geom_txt())
    target, delta = make_target(red, purple, n_face)
    if float(np.linalg.norm(delta)) < 1e-4:
        logger.info("Purple already on target (within 0.1 mm); nothing to do.")
        return True
    logger.info(
        "Purple -> red (bimanual rigid translate, robot base):\n"
        f"  purple  {np.round(purple, 4)}\n"
        f"  red     {np.round(red, 4)}"
        + (f"  +offset {np.round(offset, 3)}" if np.any(np.asarray(offset)) else "")
        + (f"  +{geom['total'] * 100:.2f} cm along the face normal {np.round(n_face, 3)}" if np.any(n_face) else "")
        + (f"  stop_short {stop_short:.3f}" if stop_short > 0 else "")
        + (f"  scale {scale:g}" if scale != 1.0 else "")
        + f"\n  target  {np.round(target, 4)}\n"
        f"  delta   {np.round(delta, 4)}  |d| = {np.linalg.norm(delta):.4f} m"
    )

    # Stage 0: the torso joints are all pitch, so a sideways (robot y) gap can only be closed by the arms, which
    # cannot shift a two-handed grasp far to one side. If red sits off the purple's bearing, turn the base in
    # place by that bearing difference first (capped), so the remaining delta is forward/down.
    def bearing_gap(red_b: np.ndarray, purple_b: np.ndarray, n_b: np.ndarray) -> float:
        """Yaw (rad, +CCW) that puts the target (red + offset + contact + air along the face normal) on the purple's bearing."""
        tgt = red_b + np.asarray(offset, dtype=float) + geom["total"] * n_b
        return float(wrap(np.arctan2(tgt[1], tgt[0]) - np.arctan2(purple_b[1], purple_b[0])))

    yaw_needed = bearing_gap(red, purple, n_face)
    yaw_goal = float(np.clip(yaw_needed, -np.deg2rad(yaw_cap_deg), np.deg2rad(yaw_cap_deg)))
    yaw_goal = float(np.sign(yaw_goal) * max(abs(yaw_goal) - np.deg2rad(yaw_short_deg), 0.0))
    turn_base = bool(align_base and abs(yaw_goal) >= np.deg2rad(yaw_tol_deg))
    if turn_base:
        logger.info(
            f"Stage 0: red is {delta[1] * 100:+.1f} cm sideways (robot y) of purple, bearing gap {np.rad2deg(yaw_needed):+.1f} deg -> "
            f"the base turns {np.rad2deg(yaw_goal):+.1f} deg ({'CCW / left' if yaw_goal > 0 else 'CW / right'}) in place first"
            + (f" (capped at {yaw_cap_deg:.0f} deg)" if abs(yaw_needed) > abs(yaw_goal) + 1e-6 else "") + "; the arms take the rest."
        )
        if plan_only:
            red_pred = rot_z(red, -yaw_goal)
            if geom["yaw"] is not None:  # the block turns with the base, the box does not
                geom["yaw"] = float((geom["yaw"] + yaw_goal + np.pi / 2) % np.pi - np.pi / 2)
                geom["contact"], geom["what"] = contact_geom(geom["yaw"])
                geom["total"] = geom["contact"] + max(float(standoff), 0.0)
            target, delta = make_target(red_pred, purple, rot_z(n_face, -yaw_goal))
            logger.info(f"Predicted after the turn: red {np.round(red_pred, 4)}, delta {np.round(delta, 4)} |d| = {np.linalg.norm(delta):.4f} m; "
                        + geom_txt() + " (the arm plan below uses this).")
    elif abs(delta[1]) > 0.05 or abs(yaw_needed) > np.deg2rad(5.0):
        logger.warning(
            f"Red is {delta[1] * 100:+.1f} cm sideways (robot y) from purple: the base would have to yaw "
            f"{np.rad2deg(yaw_needed):+.1f} deg to line them up. The arms alone rarely cover more than a few cm of that; "
            "consider --align-base (or 8.rotate_base_to_conveyor.py) first."
        )

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
        if turn_base and not plan_only:
            leftover = float(wrap(yaw_needed - yaw_goal))  # bearing gap that should remain after the turn
            comp = {"aim": 0.0}  # extra yaw to aim past the target (rad): the miss measured on the previous turn

            def live_gap() -> float:
                """Bearing gap from the live markers without logging (called at the chassis loop rate)."""
                return bearing_gap(quiet_xyz(tip_path), quiet_xyz(purple_path), face_normal())

            def bearing_error() -> tuple[float, str]:
                gap = live_gap()
                return float(wrap(gap - leftover + comp["aim"])), f"bearing gap {np.rad2deg(gap):+.1f} deg"

            def settle(t_max: float, window_s: float = 1.5, still_deg: float = 0.12) -> tuple[float, float]:
                """Wait until the base stops unwinding after the wheels stop. Returns the bearing gap (rad) right
                after the stop and after settling (still = range below ``still_deg`` over ``window_s``)."""
                t0_ = time.monotonic()
                hist: list[tuple[float, float]] = []
                g0 = None
                while True:
                    try:
                        g = live_gap()
                    except (ValueError, OSError):
                        time.sleep(0.05)
                        continue
                    now = time.monotonic()
                    if g0 is None:
                        g0 = g
                    hist.append((now, g))
                    recent = [h[1] for h in hist if now - h[0] <= window_s]
                    spread = max(recent) - min(recent) if recent else float("inf")
                    if now - t0_ >= window_s + 0.5 and len(recent) >= 3 and spread < np.deg2rad(still_deg):
                        return g0, g
                    if now - t0_ > t_max:
                        logger.warning(f"Base still drifting after {t_max:.0f}s (bearing gap moved {np.rad2deg(spread):.2f} deg "
                                       f"in the last {window_s:.0f}s); continuing anyway.")
                        return g0, g
                    time.sleep(0.1)

            logger.info("Stage 0: turning the base in place (arms still; measuring sim speed for 2 s)...")
            rtf = measure_rtf(tip_path)
            for attempt in range(1, max(int(max_turns), 1) + 1):
                turn_in_place(
                    bot.chassis, bearing_error, tol_deg=yaw_tol_deg, kp=kp, wz_max=wz_max, wz_min=wz_min, hz=chassis_hz,
                    steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim, timeout_sim=turn_timeout_sim,
                    stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                )
                t_settle = time.monotonic()
                g_stop, g_settled = settle(settle_s)
                unwound = float(wrap(g_settled - g_stop))
                remaining = float(wrap(g_settled - leftover))
                logger.info(
                    f"Stage 0 turn {attempt}: bearing gap {np.rad2deg(g_stop):+.1f} deg at the wheel stop, {np.rad2deg(g_settled):+.1f} deg "
                    f"after settling {time.monotonic() - t_settle:.1f}s (base unwound {np.rad2deg(unwound):+.1f} deg); "
                    f"remaining {np.rad2deg(remaining):+.1f} deg (tol {yaw_tol_deg:.1f})"
                )
                if abs(remaining) < np.deg2rad(yaw_tol_deg):
                    break
                if attempt >= max(int(max_turns), 1):
                    logger.warning(f"Bearing gap still {np.rad2deg(remaining):+.1f} deg after {attempt} turn(s); the arms take the rest.")
                    break
                # The controller stops on the first tick inside the tolerance (residual ~tol on the approach side) and the
                # base then unwinds by about what it just did: aim past the target by both so the next stop lands on it.
                s_dir = 1.0 if remaining > 0 else -1.0
                unwind_comp = unwound if abs(unwound) < np.deg2rad(6.0) else 0.0
                comp["aim"] = float(np.clip(s_dir * np.deg2rad(max(yaw_tol_deg - 0.3, 0.0)) + unwind_comp, -np.deg2rad(6.0), np.deg2rad(6.0)))
                logger.info(f"Stage 0: turning again, aiming {np.rad2deg(comp['aim']):+.1f} deg past the target "
                            f"(tolerance undershoot {s_dir * max(yaw_tol_deg - 0.3, 0.0):+.1f} + expected unwinding {np.rad2deg(unwind_comp):+.1f})...")
            red, purple = read_pair()
            n_face = face_normal()
            refresh_geom()
            target, delta = make_target(red, purple, n_face)
            logger.info(f"After the base turn: purple {np.round(purple, 4)}, red {np.round(red, 4)}, delta {np.round(delta, 4)} "
                        f"|d| = {np.linalg.norm(delta):.4f} m (sideways {delta[1] * 100:+.1f} cm); " + geom_txt())
        sync_from_robot(bot, configuration, idx)
        rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        T0 = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
        mid0 = pair_midpoint(rcfg)
        q0 = rcfg.q.copy()

        best = None  # (residual, label, q_start, knots, n)
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
            rcfg.update(knots[-1])
            achieved = pair_midpoint(rcfg) - mid0
            rcfg.update(q0)
            residual = float(np.linalg.norm(delta - achieved))
            logger.info(f"  '{label}': {len(knots)}/{n} steps, residual purple->target {residual:.4f} m")
            if best is None or residual < best[0] - 1e-4:
                best = (residual, label, q_start, knots, n)
                if residual <= CLOSE_ENOUGH_M:
                    break
        if best is None:
            logger.error("No progress toward the red tip on any path. Not moving.")
            do_shutdown = True
            return False
        residual, label, q_start, knots, n = best
        Q = np.array([q_start] + knots)
        rcfg.update(Q[-1])
        predicted = purple + (pair_midpoint(rcfg) - mid0)
        rcfg.update(q0)
        covered = 100.0 * max(0.0, 1.0 - residual / max(float(np.linalg.norm(delta)), 1e-6))
        verdict = "REACHABLE" if residual <= CLOSE_ENOUGH_M else "PARTIAL"
        torso_note = ""
        if use_torso:
            torso_note = f"; torso change (j1, j2, j3) {np.round(np.rad2deg(Q[-1, ridx['torso']] - Q[0, ridx['torso']]), 1)} deg"
        logger.info(
            f"[{verdict}] path '{label}': {len(knots)}/{n} steps; predicted purple after motion "
            f"{np.round(predicted, 4)}, residual to target {residual:.4f} m ({covered:.0f}% of the gap){torso_note}"
        )
        if plan_only:
            logger.info("Plan only: nothing commanded.")
            return residual <= CLOSE_ENOUGH_M


        path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
        t_stream = max(duration, max(path[g] / (torso_max_vel if g == "torso" else max_joint_vel) for g in path))
        if t_stream > duration:
            logger.info(f"Stream time stretched to {t_stream:.1f}s by the joint speed caps "
                        f"(arms {max_joint_vel:.2f}, torso {torso_max_vel:.2f} rad/s)")
        groups = {COMPONENT[g]: Q[:, ii] for g, ii in ridx.items()}
        yaw_before = base_yaw_deg()
        logger.info(f"Executing '{label}' ({', '.join(groups)}) over {t_stream:.1f}s...")
        stream(bot, groups, t_stream, config.CONTROL_DT)

        sync_from_robot(bot, configuration, idx)
        rcfg2, _t2, _i2 = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        purple_est = purple + (pair_midpoint(rcfg2) - mid0)
        err = float(np.linalg.norm(purple_est - target))
        yaw_after = base_yaw_deg()
        drift_txt = (f" Base yaw {yaw_before:+.1f} -> {yaw_after:+.1f} deg during the move ({wrap(np.deg2rad(yaw_after - yaw_before)) * 180 / np.pi:+.1f})."
                     if yaw_before is not None and yaw_after is not None else "")
        logger.info(
            f"Done. purple (FK estimate) {np.round(purple_est, 4)}, target {np.round(target, 4)}, "
            f"err {err:.4f} m. Check {purple_path.name} for the sim-measured value. Pose left held.{drift_txt}"
        )
        logger.info("Next: open the hands (11.open_hands.py), pull them back (12.hands_back.py), "
                    "then 13.hands_push_left.py pushes the block left onto the box with both hands.")

        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return err <= CLOSE_ENOUGH_M
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


if __name__ == "__main__":
    tyro.cli(main)
