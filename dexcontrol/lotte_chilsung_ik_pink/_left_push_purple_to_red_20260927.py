# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""(Out of the sequence since 2026-09-27 11:4x; step 13 is now ``13.hands_push_left.py``.) Left arm only: push the held block toward the red conveyor tip until the left wrist F/T sensor feels the box.

Step 13 of the Chilsung sequence (2026-09-27). ``10.approach_purple_to_red_bimanual.py`` has placed the block at
its standoff next to the box, the right hand has let go (``11.open_hands.py --side right``) and moved out of the way. NOTE: this script assumes
the LEFT hand still holds the block; after ``12.hands_back.py`` (both hands withdrawn) it has to be reworked. The block now hangs from the left gripper alone (the
lower hand after ``9.rotate_vertical.py``), so the push is a single-arm move: the right shoulder's joint limit,
which cut the old two-handed push to a few cm, no longer matters. The left arm and, by default, the torso are
commanded (``--no-use-torso`` = arm only; the torso is needed when the left arm is near its limits at the centreline,
and the raised right arm simply rides along with it); the right arm, head and grippers are never touched.

Geometry (robot base frame, from the sim's live marker files, see ``_sim_markers.py``): the push direction is the
horizontal line from the purple block centre to the red tip (``--no-horizontal`` also closes the height gap), so an
along-face residual left by step 10 is closed on the way and the markers end up lined up. The block cannot reach
red itself -- red lies inside the box and one finger plate of the left gripper sits between block and box -- so
the push is planned toward the alignment point (purple onto red, horizontally), capped at ``--push-max``, and the
F/T sensor is what stops it. The geometric *first contact* (``_block_geom.first_contact_along``: 7.75 cm from red
along the normal for a square block, its leading corner when yawed; ``--contact-dist`` overrides) is logged as the
expected stop and bounds the blind push; ``--cap-at-contact`` plans only up to it + ``--push-over`` (the 2026-09-26
two-handed rule, which on 2026-09-27 left a 13 deg-yawed block with a 0.8 cm plan and 'nothing happens'). Contact
detection: the left wrist wrench is tared while the block is still free, low-pass filtered (0.25 s) and compared with
``max(--contact-n, 1.5 x rest noise + 3 N)`` for ``--confirm-ticks`` ticks (``_contact_gate.ContactGate``); on
contact the *measured* pose is held so nothing keeps pressing. The push also stops when the markers show purple
within ``--align-tol`` of red, and the planned path is cut where the IK stops translating (a 5 mm step that needs
a joint change above ``--push-jump-rad``). Without wrench data nothing is pushed unless ``--allow-blind``, which
then stops ``--blind-short`` before the nominal contact. The stream is anchored at the sim's held joint targets
(``read_setpoints`` / ``anchor_offsets``; components the plan does not move are not commanded) and the gate is
blanked for ``--gate-blank-s`` after the start: on 2026-09-27 commanding the torso at its *measured* (1.3 deg sagged)
angles jolted the wrist to 20.8 N before anything moved. Streamed under the joint speed caps at ``--push-speed``
(``_pink_arm.stream``, leash-paced).

Example (block at the standoff, right hand open and raised, omniteleop follower stopped, ROBOT_NAME / ZENOH_CONFIG exported)::

    python 13.left_push_purple_to_red.py --plan-only      # geometry + reachable push, no motion, no taring
    python 13.left_push_purple_to_red.py                  # push until contact (or aligned / out of room)
    python 13.left_push_purple_to_red.py --contact-n 20   # less sensitive contact detection
    python 13.left_push_purple_to_red.py --push-max 0.04  # never more than 4 cm
    python 13.left_push_purple_to_red.py --no-use-torso   # left arm only, torso still
    python 13.left_push_purple_to_red.py --allow-blind    # no F/T data: stop 3 mm short of the nominal contact
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
from _block_geom import BLOCK_HALF_T, block_yaw_vs_face, first_contact_along  # noqa: E402
from _contact_gate import ContactGate  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    COMPONENT,
    DEFAULT_STEP_RAD,
    TORSO_MAX_VEL,
    TORSO_POSTURE_COST,
    build_arm_ik,
    anchor_offsets,
    detect_foreign_commands,
    plan_se3_path,
    read_setpoints,
    stream,
)
from _pink_common import EE, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402
from _sim_markers import (  # noqa: E402
    PURPLE_FILE_NAME,
    TIP_FILE_NAME,
    box_face_normal_base,
    find_sim_dir,
    load_xyz_base,
    quiet_xyz,
)

SIDE = "left"
MIN_PUSH_COS = 0.2  # the push must close the normal gap at least this fast (cos of its angle to -normal)
MARKER_LOG_PERIOD_S = 0.5


def plan_push(rcfg, tasks, solver, push_dir: np.ndarray, push_len: float, step_m: float, push_jump_rad: float):
    """Straight push of the left gripper origin by ``push_len`` along ``push_dir`` (orientation held), planned from
    the current reduced state and cut where the IK stops *translating*: a step that needs a joint change above
    ``push_jump_rad`` or makes no progress along the push means the arm has run out of room and would swing instead.
    Returns ``(q_start, knots, reach_m, n_planned)``; ``rcfg`` is left at its starting state."""
    T0 = rcfg.get_transform_frame_to_world(EE[SIDE])
    p0 = T0.translation.copy()
    q0 = rcfg.q.copy()
    goal = pin.SE3(T0.rotation.copy(), p0 + push_len * push_dir)
    planned = plan_se3_path(rcfg, tasks, solver, goal, step_m, DEFAULT_STEP_RAD)
    if planned is None:
        rcfg.update(q0)
        return q0, [], 0.0, 0
    q_start, knots_raw = planned
    knots: list[np.ndarray] = []
    q_prev, reach = q_start, 0.0
    for k in knots_raw:
        jump = float(np.max(np.abs(k - q_prev)))
        rcfg.update(k)
        prog = float((rcfg.get_transform_frame_to_world(EE[SIDE]).translation - p0) @ push_dir)
        if jump > push_jump_rad or prog < reach + 0.3 * step_m:
            logger.info(f"Push path cut at {len(knots)}/{len(knots_raw)} steps (next step needs a joint change of "
                        f"{jump:.2f} rad for {(prog - reach) * 1000:.1f} mm) -- the arm is out of room there.")
            break
        knots.append(k)
        q_prev, reach = k, prog
    rcfg.update(q0)
    return q_start, knots, reach, len(knots_raw)


def main(
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    max_age_s: float = 3.0,
    horizontal: bool = True,
    contact_dist: float | None = None,
    face_normal_world: tuple[float, float, float] = (1.0, 0.0, 0.0),
    push_max: float = 0.08,
    push_over: float = 0.006,
    push_speed: float = 0.015,
    step_m: float = 0.005,
    push_jump_rad: float = 0.15,
    min_push: float = 0.002,
    align_tol: float = 0.01,
    contact_n: float = 10.0,
    confirm_ticks: int = 5,
    tare_samples: int = 10,
    warmup_s: float = 1.0,
    rest_s: float = 1.5,
    cap_at_contact: bool = False,
    allow_blind: bool = False,
    blind_short: float = 0.003,
    gate_blank_s: float = 0.3,
    anchor_setpoints: bool = True,
    use_torso: bool = True,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    max_joint_vel: float = 0.3,
    torso_max_vel: float = TORSO_MAX_VEL,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Push the block held in the left hand toward the red tip until the left wrist feels the box, the markers line
    up, or the arm runs out of room. Returns True when the push ended on contact or alignment.

    Args:
        tip_file: Red tip XYZ file (robot base). Default: the running sim's ``card_box_tip_pos.txt``.
        purple_file: Purple centre XYZ file (robot base). Default: the running sim's ``card_box_mass_center_pos.txt``.
        max_age_s: Marker files older than this are rejected (sim not running).
        horizontal: Push along the horizontal purple -> red line (height held). False = straight at red in 3D.
        contact_dist: Override the computed first-contact distance (metres, purple-to-red along the normal).
        face_normal_world: Outward normal of the box face in the world (+X from the scene); used only when the tip
            file carries no box orientation.
        push_max: Never push farther than this (metres), whatever the markers say.
        push_over: With ``--cap-at-contact`` the planned push = distance to the nominal first contact + this margin (metres).
        push_speed: Push speed (m/s) along the push direction; joint speed caps may slow it further.
        step_m: Cartesian spacing of the IK waypoints (metres).
        push_jump_rad: Cut the push path at the first step that needs a joint change above this (rad): beyond it the
            IK swings the arm instead of translating the block. 0.35 = the planner's own limit.
        min_push: Below this reachable push (metres) nothing is commanded and the F/T gate is not armed.
        align_tol: Stop as ALIGNED when the markers put purple within this of red (metres, horizontal unless
            ``--no-horizontal``). Also "nothing to do" if already that close.
        contact_n: Minimum contact threshold on the filtered, tared left wrist force (N); raised automatically to
            1.5 x the measured rest noise + 3 N.
        confirm_ticks: Consecutive stream ticks (50 Hz) above the threshold that count as contact.
        tare_samples: Wrench samples averaged for the baseline before the push.
        warmup_s: Poll the wrench for this long before taring so an idle-paused subscriber delivers fresh samples.
        rest_s: Rest-noise measurement time after taring (seconds); sizes the adaptive threshold.
        cap_at_contact: Plan only up to the nominal first contact + ``push_over`` (the 2026-09-26 Stage 2 rule) instead of
            toward the alignment point. Off by default: the geometric contact estimate is only informative (a yawed block's
            corner, an inexact grasp depth) and the F/T sensor is the stop, as requested.
        allow_blind: Without wrench data, push to ``blind_short`` before the nominal first contact instead of aborting.
        blind_short: Blind push stops this far before the nominal first contact (metres).
        gate_blank_s: Ignore the contact gate for this long after the stream starts (seconds): the first commands can
            still produce a short wrench transient (2026-09-27: 20.8 N within 0.2 s, 0.0 cm moved) while the min-jerk
            profile has barely moved. Ticks above threshold during the blanking are counted and logged.
        anchor_setpoints: Read the sim's held joint targets and anchor the stream there (recommended): commanding the
            *measured* joints instead lets the torso relax by its gravity sag (2026-09-27: torso_j2 1.3 deg), a jolt the
            wrist sensor reads as tens of newtons. Components whose planned path is zero are not commanded at all.
        use_torso: Free the torso in the IK too (default, as in step 10): the torso only moves as much as the posture cost
            allows, but at the centreline the left arm alone is nearly out of room (2026-09-27 diagnostic: L_arm_j2 / j4
            within 0.07 / 0.03 rad of their limits, arm-only +y reach 0.5 cm, with the torso 5 cm). The raised right arm
            rides along with the torso. ``--no-use-torso`` = left arm only.
        torso_posture_cost: Posture cost on the torso joints; higher = torso moves less (arms use 1e-3).
        max_joint_vel: Arm joint speed cap (rad/s) for the streamed push.
        torso_max_vel: Torso joint speed cap (rad/s).
        plan_only: Connect, plan from the measured state, report the reachable push, move nothing (no taring).
        leave_held: Exit without ``stop()`` so the pose stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming left-arm (or torso) targets.
    """
    if push_max <= 0.0 or push_speed <= 0.0 or push_over < 0.0 or align_tol < 0.0 or blind_short < 0.0:
        raise ValueError("push_max and push_speed must be > 0; push_over, align_tol and blind_short >= 0")
    sim_dir = find_sim_dir()
    tip_path = Path(tip_file) if tip_file is not None else sim_dir / TIP_FILE_NAME
    purple_path = Path(purple_file) if purple_file is not None else sim_dir / PURPLE_FILE_NAME
    try:
        red = load_xyz_base(tip_path, "red tip", max_age_s)
        purple = load_xyz_base(purple_path, "purple centre", max_age_s)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False
    n_face = box_face_normal_base(tip_path, sim_dir, face_normal_world, max_age_s)
    if not np.any(n_face):
        logger.error("Box face normal unavailable (no quat_base in the tip file and no base yaw from the markers), so "
                     "the contact distance cannot be placed. Nothing commanded.")
        return False
    s_face = np.cross(np.array([0.0, 0.0, 1.0]), n_face)  # horizontal unit vector along the box face

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

    def offsets(red_b: np.ndarray, purple_b: np.ndarray) -> tuple[float, float, float, float]:
        """(distance purple->red, purple outside the face along the normal, along the face, height) in metres."""
        d = red_b - purple_b
        if horizontal:
            d = np.array([d[0], d[1], 0.0])
        rel = purple_b - red_b
        return float(np.linalg.norm(d)), float(rel @ n_face), float(rel @ s_face), float(rel[2])

    # -- geometry of the push -----------------------------------------------------------------------------------
    yaw = block_yaw_vs_face(tip_path, purple_path)
    if contact_dist is not None:
        contact, what = float(contact_dist), "override"
    else:
        contact, what = first_contact_along(0.0 if yaw is None else yaw)
    dist, along, ds, dz = offsets(red, purple)
    d = red - purple
    if horizontal:
        d = np.array([d[0], d[1], 0.0])
    push_dir = d / dist if dist > 1e-6 else -n_face.copy()
    cos = float(push_dir @ -n_face)
    gap_n = along - contact  # air left before the nominal first contact (along the normal)
    if gap_n <= 0.0:
        to_contact = 0.0
    elif cos >= MIN_PUSH_COS:
        to_contact = gap_n / cos
    else:
        to_contact = float("inf")
    yaw_txt = f"{np.rad2deg(yaw):+.1f} deg" if yaw is not None else "unknown (block not upright?)"
    logger.info(
        "Push geometry (robot base):\n"
        f"  red     {np.round(red, 4)}   face normal {np.round(n_face, 3)}, along-face {np.round(s_face, 3)}\n"
        f"  purple  {np.round(purple, 4)}   {along * 100:+.2f} cm outside the face, {ds * 100:+.1f} cm along it, "
        f"{dz * 100:+.1f} cm high; {dist * 100:.1f} cm from red{' horizontally' if horizontal else ''}\n"
        f"  block yaw vs the face {yaw_txt} -> first contact by the {what} at {contact * 100:.2f} cm from red along the "
        f"normal; air before it {gap_n * 100:+.2f} cm (block face to box face {(along - BLOCK_HALF_T) * 100:.2f} cm)\n"
        f"  push direction {np.round(push_dir, 3)} (cos to -normal {cos:.2f})"
    )
    if dist <= align_tol:
        logger.info(f"Purple is already within {align_tol * 100:.1f} cm of red; nothing to do.")
        return True
    if not np.isfinite(to_contact):
        logger.error(f"The push direction is almost parallel to the box face (cos {cos:.2f} < {MIN_PUSH_COS}): the block "
                     "would slide along the face instead of onto it. Fix the along-face offset first (step 10). Nothing commanded.")
        return False
    push_len = min(float(push_max), dist)
    if cap_at_contact:
        push_len = min(push_len, to_contact + float(push_over))
    blind_len = min(float(push_max), max(to_contact - float(blind_short), 0.0), dist)
    logger.info(f"Planned push: {push_len * 100:.1f} cm toward the alignment point ({dist * 100:.1f} cm away, cap {push_max * 100:.0f} cm"
                + (f", capped at the nominal contact + {push_over * 1000:.0f} mm" if cap_at_contact else "")
                + f") at {push_speed * 100:.1f} cm/s; the geometry expects first contact after {to_contact * 100:.1f} cm, "
                "the left wrist F/T is the stop.")
    if push_len < min_push:
        logger.info(f"Nothing to push: the planned push is below {min_push * 1000:.0f} mm.")
        return True

    components = [COMPONENT[SIDE]] + (["torso"] if use_torso else [])
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

    setpoints = read_setpoints(components) if (anchor_setpoints and not plan_only) else {}
    configuration, _tasks, solver, idx = build_ik()
    bot = connect()
    do_shutdown = not leave_held
    try:
        sync_from_robot(bot, configuration, idx)
        rcfg, tasks, ridx = build_arm_ik(configuration, SIDE, use_torso, torso_posture_cost)
        p_left0 = rcfg.get_transform_frame_to_world(EE[SIDE]).translation.copy()
        logger.info(f"Left gripper origin {np.round(p_left0, 4)} (robot base); planning the push...")
        q_start, knots, reach, n_raw = plan_push(rcfg, tasks, solver, push_dir, push_len, step_m, push_jump_rad)
        torso_note = ""
        if use_torso and knots:
            torso_note = f"; torso change (j1, j2, j3) {np.round(np.rad2deg(knots[-1][ridx['torso']] - q_start[ridx['torso']]), 1)} deg"
        verdict = "REACHABLE" if reach + 1e-4 >= push_len else "PARTIAL"
        logger.info(f"[{verdict}] {reach * 100:.1f} of {push_len * 100:.1f} cm reachable ({len(knots)}/{n_raw} steps){torso_note}")
        if plan_only:
            logger.info("Plan only: nothing commanded, F/T gate not armed.")
            return reach >= min_push
        if reach < min_push:
            logger.warning(f"Only {reach * 1000:.1f} mm of push room after the joint-limit cut (< {min_push * 1000:.0f} mm): the left arm "
                           f"cannot move the block toward the box from here (try --use-torso). Pose held, nothing pushed.")
            return False

        gate = ContactGate(contact_n, confirm_ticks, sample_hz=1.0 / config.CONTROL_DT, sides=(SIDE,),
                           tare_samples=tare_samples, warmup_s=warmup_s, rest_s=rest_s)
        logger.info("Arming the left wrist contact gate (the block must not touch the box now)...")
        if not gate.arm(bot):
            if not allow_blind:
                logger.error("No usable left wrist wrench data (state/wrench/left): the push would be blind. Nothing pushed "
                             f"(--allow-blind pushes to {blind_short * 1000:.0f} mm before the nominal contact instead).")
                return False
            logger.warning(f"Blind push: no wrench data, so the push stops {blind_short * 1000:.0f} mm short of the nominal "
                           f"first contact ({blind_len * 100:.1f} cm) and only the markers can stop it early.")
            if blind_len < min_push:
                logger.info("Nothing to push blind (the block already stands at the nominal contact).")
                return True
            q_start, knots, reach, n_raw = plan_push(rcfg, tasks, solver, push_dir, blind_len, step_m, push_jump_rad)
            if reach < min_push:
                logger.warning("The left arm cannot move the block toward the box from here; pose held, nothing pushed.")
                return False

        Q = np.array([q_start] + knots)
        path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
        t_push = max(reach / push_speed, 1.0, max(path[g] / (torso_max_vel if g == "torso" else max_joint_vel) for g in path))
        off = anchor_offsets(configuration, idx, setpoints, components)
        groups = {}
        for g, ii in ridx.items():
            comp = COMPONENT[g]
            if path[g] < 1e-4:
                logger.info(f"{comp} not commanded: the planned push does not move it (its held target stays untouched).")
                continue
            groups[comp] = Q[:, ii] + off[comp]
        if not groups:
            logger.warning("The planned push moves no joint; nothing to stream.")
            return False

        def pushed_m() -> float:
            sync_from_robot(bot, configuration, idx)
            return float((configuration.get_transform_frame_to_world(EE[SIDE]).translation - p_left0) @ push_dir)

        gate.travel_fn = pushed_m
        state: dict = {"stop": None, "last_log": 0.0, "t0": None, "blanked": 0}

        def stop_check(b) -> bool:
            if state["t0"] is None:
                state["t0"] = time.time()
            msg = gate.check(b) if gate.armed else None
            if msg and time.time() - state["t0"] < gate_blank_s:
                state["blanked"] += 1
                gate._above = 0  # confirmation restarts after the blanking
                if state["blanked"] == 1:
                    logger.info(f"  (contact gate suppressed during the first {gate_blank_s:.1f}s: {msg})")
                msg = None
            if msg:
                state["stop"] = "contact"
                logger.info(msg)
                return True
            try:
                dist_now, along_now, _ds, _dz = offsets(quiet_xyz(tip_path), quiet_xyz(purple_path))
            except (ValueError, OSError):
                return False  # half-written marker file; try again next tick
            if dist_now <= align_tol:
                state["stop"] = "aligned"
                logger.info(f"ALIGNED: purple is {dist_now * 100:.1f} cm from red (<= {align_tol * 100:.1f} cm) -- stopping the push.")
                return True
            now = time.time()
            if now - state["last_log"] >= MARKER_LOG_PERIOD_S:
                state["last_log"] = now
                logger.info(f"  markers: purple {dist_now * 100:.1f} cm from red, {along_now * 100:+.2f} cm outside the face "
                            f"({(along_now - contact) * 1000:+.0f} mm before the nominal {what} contact)")
            return False

        logger.info(f"Pushing {reach * 100:.1f} cm ({len(knots)} steps, {' + '.join(groups)}) over {t_push:.1f}s with "
                    + ("left wrist contact detection..." if gate.armed else "marker alignment only (blind)..."))
        _final, stopped = stream(bot, groups, t_push, config.CONTROL_DT, stop_check=stop_check)
        time.sleep(0.3)
        pushed = pushed_m()
        try:
            red_now, purple_now = read_pair()
            dist_now, along_now, ds_now, dz_now = offsets(red_now, purple_now)
            gap_txt = (f"purple {dist_now * 100:.1f} cm from red, {along_now * 100:+.2f} cm outside the face -> block face to box face "
                       f"{(along_now - BLOCK_HALF_T) * 100:.1f} cm ({(along_now - contact) * 1000:+.0f} mm before the nominal {what} contact); "
                       f"along the face {ds_now * 100:+.1f} cm, height {dz_now * 100:+.1f} cm")
        except (ValueError, OSError, FileNotFoundError):
            gap_txt = "marker gap unavailable"
        reason = {"contact": "CONTACT -- stopped by the left wrist F/T", "aligned": "ALIGNED -- purple reached red",
                  None: "no contact within the push range"}[state["stop"]]
        early_txt = (" (this early -- likely the grasp straining near a joint limit, not real contact; consider raising --contact-n)"
                     if state["stop"] == "contact" and pushed < 0.3 * reach and gate.already_loaded_n > gate.threshold_n else "")
        force_txt = (f"; peak filtered |dF| {gate.peak_n:.1f} N" + (f", {state['blanked']} tick(s) above threshold suppressed in the first {gate_blank_s:.1f}s" if state["blanked"] else "")) if gate.armed else ""
        (logger.info if stopped else logger.warning)(
            f"{reason} after {pushed * 100:.1f} cm of push{early_txt}; {gap_txt}{force_txt}. Pose held."
        )
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return state["stop"] is not None
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
