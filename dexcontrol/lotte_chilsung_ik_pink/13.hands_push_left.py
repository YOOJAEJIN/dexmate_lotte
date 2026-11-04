# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Push the block left onto the conveyor box in two stages: (1) both grippers translate left (robot +Y) with their
orientation held, each arm as far as it can; (2) the LEFT shoulder joint ``L_arm_j2`` alone turns on until the block is
home, with a hard force cap (``--j2-force-cap-rad``). The wrist F/T sensors and the markers stop either stage.

Step 13 of the Chilsung sequence (2026-09-27, final form chosen by the user after trying translation-only, wrist and
j2-only variants). After ``10.approach_purple_to_red_bimanual.py`` placed the block, ``11.open_hands.py`` opened the
grippers and ``12.hands_back.py`` pulled the hands back, the open fingers still reach over the block's near edge with one
finger plate on each side of it; moving the plates left pushes the block's face onto the conveyor box tip.

Stage 1 -- **translate (+ wrist where short)** (both arms): per arm a single-arm Pink IK straight line along +Y of
``--push-cm`` (8 cm) with the gripper orientation held (``_j1_up.plan_translation``, torso locked). Where joint limits
cut a path the reachable prefix is used; the two paths are resampled to one profile and streamed together, so the arm
with less room simply moves slower (2026-09-27: right ~4.5 cm on its ``R_arm_j2`` limit, left ~5.7 cm). An arm that
falls short of ``--push-cm`` appends a wrist turn (``--wrist-deg``, 15 deg; the j5 / j6 / j7 joint that carries its finger
tip farthest toward +Y, other joints held) so it keeps pushing with the other arm instead of standing still.

Stage 2 -- **left j2** (pure joint space, no IK): ``L_arm_j2`` ramps in the direction that carries the left finger tip
toward +Y (found by FK) by up to ``--j2-deg`` (20 deg), never past its limit, every other joint holding its measured
value; the whole arm swings about the shoulder and the plate sweeps the block home (j2 moves the tip ~93 cm per rad, so
the sweep is paced by ``--tip-speed``, 2 cm/s). ``--j2-side`` picks the arm (default left; the right j2 sits on its inward
limit and cannot turn left).

The stages run back to back (no hold in between); one final hold (``--hold-s``) keeps the last targets pressing. Streams
are anchored at the sim's held arm targets. Stage 1's push force follows ``--leash-rad`` (how far the commanded target
may run ahead of the measured joints before the profile slows; the sim's PD force follows that lag: 0.05 -> ~140 N,
0.3 -> ~170-230 N on the left wrist). The leash is not a hard limit (the profile still creeps at 30 % when the lag
exceeds it), so stage 2 -- one arm pressing the block against the box -- gets a real cap: ``--j2-force-cap-rad``
(0.015) clamps every stage-2 and final-hold setpoint to that far from the measurement, i.e. j2 torque <= 6000 x 0.015
= 90 Nm, ~100 N at the finger tip (2026-09-27 evening: with the default 0.4 rad clamp the left-arm sweep lifted the
base wheels). ``--j2-leash-rad`` (0.03) slows the stage-2 profile early as well.

Stops, whichever comes first (checked every tick in both stages):

* **F/T** -- both wrists are tared just before stage 1, low-pass filtered (0.25 s) and compared with
  ``max(--contact-n, 1.5 x rest noise + 3 N)`` for ``--confirm-ticks`` ticks on either wrist (``_contact_gate``).
  ``--contact-n`` defaults to 500 N (ceiling only; the arms push with all they have -- a 60 N stop tried on 2026-09-27
  evening kept the two-arm push from moving the block at all); lower it for a real F/T stop.
  On contact the *measured* pose is held. The gate is blanked for ``--gate-blank-s`` after each stream starts.
* **markers** -- the purple block centre is within ``--flush-tol-cm`` of half the block thickness from the red tip along
  the box face normal (block face flush with the box face).
* **stall** -- after contact (block moved >= ``--stall-arm-mm`` 5 mm since the stage began, or a wrist reads
  ``--stall-force-n`` 30 N) the block face gap (or, without markers, the push distance) changed by less than
  ``--stall-mm`` (3) over ``--stall-s`` (2 s): the block is against something and pushing on only builds force. Ends
  the final hold early as well.
* **planned length / j2 budget / joint limits** reached.

Example (hands open and pulled back, omniteleop follower stopped, ROBOT_NAME / ZENOH_CONFIG exported)::

    python 13.hands_push_left.py --plan-only        # stage 1 reach per arm, stage 2 j2 direction / room; no motion
    python 13.hands_push_left.py                    # translate, then left j2 until the markers / F/T say the block is home
    python 13.hands_push_left.py --push-cm 5 --j2-deg 30
    python 13.hands_push_left.py --leash-rad 0.15   # gentler push
    python 13.hands_push_left.py --push-cm 0        # j2 stage only
    python 13.hands_push_left.py --j2-deg 0         # translation only
    python 13.hands_push_left.py --j2-force-cap-rad 0.02   # stage 2 pushes harder (~130 N)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _block_geom import BLOCK_HALF_T, block_yaw_vs_face  # noqa: E402
from _contact_gate import ContactGate  # noqa: E402
from _j1_up import plan_translation  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    COMPONENT,
    GROUP,
    MAX_ANCHOR_OFFSET,
    anchor_offsets,
    detect_foreign_commands,
    read_setpoints,
    rot_err,
    stream,
)
from _pink_common import ARM_JOINTS, EE, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402
from _sim_markers import (  # noqa: E402
    PURPLE_FILE_NAME,
    TIP_FILE_NAME,
    box_face_normal_base,
    find_sim_dir,
    load_xyz_base,
    quiet_xyz,
)

SIDES = ["left", "right"]
PUSH_DIR = np.array([0.0, 1.0, 0.0])  # robot base +Y = left
TIP_AHEAD_M = 0.2206  # long finger blade tip ahead of the gripper base along EE +Z: 2FPA 2nd gen z_max 0.3031 (arm_l8) - 0.0825 (was 0.188, 1st gen; see 4.left_mirror_and_push_force.py)
J2 = 1  # index of *_arm_j2 in an arm's joint vector
WRIST_CANDIDATES = (4, 5, 6)  # *_arm_j5, j6, j7
MARKER_LOG_PERIOD_S = 0.5


def main(
    push_cm: float = 8.0,
    wrist_deg: float = 15.0,
    j2_deg: float = 20.0,
    j2_side: str = "left",
    j2_vel: float = 0.3,
    tip_speed: float = 0.02,
    step_m: float = 0.005,
    push_speed: float = 0.015,
    max_joint_vel: float = 0.3,
    leash_rad: float = 0.3,
    j2_leash_rad: float = 0.03,
    j2_force_cap_rad: float = 0.015,
    hold_s: float = 5.0,
    contact_n: float = 500.0,
    confirm_ticks: int = 5,
    tare_samples: int = 10,
    warmup_s: float = 1.0,
    rest_s: float = 1.5,
    gate_blank_s: float = 0.3,
    allow_blind: bool = False,
    flush_tol_cm: float = 0.5,
    stall_s: float = 2.0,
    stall_mm: float = 3.0,
    stall_arm_mm: float = 5.0,
    stall_force_n: float = 30.0,
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    max_age_s: float = 3.0,
    anchor_setpoints: bool = True,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Stage 1: both grippers translate left with orientation held (each arm as far as it can). Stage 2: the left
    ``L_arm_j2`` alone turns until the block is home. Returns True when a stage ended on contact or on the markers
    showing the block flush.

    Args:
        push_cm: Stage 1 translation per arm (centimetres; 0 = skip stage 1); joint limits may shorten it.
        wrist_deg: Stage 1: an arm whose translation falls short of ``push_cm`` (the right arm, on its j2 limit) adds a
            wrist turn of up to this many degrees right after its translation -- the wrist joint (j5 / j6 / j7) whose
            rotation carries the finger tip farthest toward +Y, other joints held -- so it keeps pushing along with
            the left arm instead of standing still (user request 2026-09-27 12:4x; 0 = off).
        j2_deg: Stage 2 j2 turn (degrees; 0 = skip stage 2), clipped to the joint limit.
        j2_side: Arm for stage 2 ("left" or "right").
        j2_vel: j2 speed cap in stage 2 (rad/s).
        tip_speed: Finger-tip speed cap along +Y in stage 2 (m/s): j2 moves the tip ~93 cm per rad.
        step_m: Cartesian spacing of the stage-1 IK waypoints (metres).
        push_speed: Stage 1 translation speed (m/s); the joint speed cap may slow it further.
        max_joint_vel: Arm joint speed cap (rad/s).
        leash_rad: THE FORCE KNOB: the commanded target may run this far (rad) ahead of the measured joints before the
            profile slows; the sim's PD force follows that lag. Setpoints are still clamped to 0.4 rad of the measurement.
        j2_leash_rad: Stage 2 leash (rad): the j2 profile slows once the target runs this far ahead of the measurement.
        j2_force_cap_rad: Stage 2 FORCE CAP (rad): every stage-2 (and final-hold) setpoint is clamped to within this
            of the measured joints, so the sim's PD torque is bounded: j2 stiffness 6000 Nm/rad over the ~0.93 m tip
            lever gives ~65 N of push per 0.01 rad -> 0.015 default ~100 N (the wheels lifted at ~170-230 N with the
            previous 0.4 rad clamp). Raise for a harder push, lower for a gentler one.
        hold_s: Keep re-sending the final targets for up to this long at the very end (seconds).
        contact_n: Force ceiling on the filtered, tared wrist force (N), per wrist; 500 = push with all the arms have.
        confirm_ticks: Consecutive stream ticks (50 Hz) above the threshold that count as contact.
        tare_samples: Wrench samples averaged for the baseline before stage 1.
        warmup_s: Poll the wrench for this long before taring so an idle-paused subscriber delivers fresh samples.
        rest_s: Rest-noise measurement time after taring (seconds).
        gate_blank_s: Ignore the contact gate for this long after each stream starts (start-up transient).
        allow_blind: Without wrench data, push anyway (markers / planned lengths are the only stops).
        stall_s: STALL stop: end the current stage (and the final hold) when the block face gap to the box -- or, without
            markers, the finger push distance -- has changed by less than ``stall_mm`` for this long (seconds; 0 = off).
            2026-09-27 22:17: the block came to rest 1.0 cm from the box face (above the 0.5 cm flush tolerance) with
            ~110-120 N on both wrists (below the 500 N ceiling), so nothing stopped stage 2 and it ground on for a
            minute at the capped force; the stall stop ends it after 2 s without progress.
        stall_mm: Progress below this over ``stall_s`` counts as stalled (millimetres).
        stall_arm_mm: The stall watch only arms once the push has made contact: the block has moved this far
            (millimetres) since the stage started, or a wrist reads ``stall_force_n``. Before contact the block gap
            naturally does not change (2026-09-27 22:21: without this the watch stopped stage 1 after 2.3 s with the
            blades still 0.4 cm into their approach).
        stall_force_n: Filtered wrist force (N) that also arms the stall watch (contact without visible block motion).
        flush_tol_cm: Markers: stop when the block face is within this of the box face (centimetres).
        tip_file: Red tip XYZ file (robot base). Default: the running sim's ``card_box_tip_pos.txt``.
        purple_file: Purple centre XYZ file (robot base). Default: the running sim's ``card_box_mass_center_pos.txt``.
        max_age_s: Marker files older than this are treated as unavailable.
        anchor_setpoints: Stream relative to the sim's held arm targets (recommended).
        plan_only: Connect, plan / preview both stages from the measured state, move nothing (no taring).
        leave_held: Exit without ``stop()`` so the pose stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming arm targets.
    """
    if j2_side not in ("left", "right"):
        raise ValueError(f"--j2-side must be 'left' or 'right', got {j2_side!r}")
    if push_cm < 0.0 or j2_deg < 0.0 or push_speed <= 0.0 or j2_vel <= 0.0 or tip_speed <= 0.0 or flush_tol_cm < 0.0:
        raise ValueError("push_cm, j2_deg, flush_tol_cm >= 0; push_speed, j2_vel, tip_speed > 0")
    if push_cm <= 0.0 and j2_deg <= 0.0:
        raise ValueError("both stages are disabled (push_cm 0 and j2_deg 0)")
    push_len = push_cm / 100.0

    # -- markers (optional): flush stop and reporting -------------------------------------------------------------
    sim_dir = find_sim_dir()
    tip_path = Path(tip_file) if tip_file is not None else sim_dir / TIP_FILE_NAME
    purple_path = Path(purple_file) if purple_file is not None else sim_dir / PURPLE_FILE_NAME
    n_face = np.zeros(3)
    markers = False
    try:
        red = load_xyz_base(tip_path, "red tip", max_age_s)
        purple = load_xyz_base(purple_path, "purple centre", max_age_s)
        n_face = box_face_normal_base(tip_path, sim_dir, max_age_s=max_age_s)
        markers = bool(np.any(n_face))
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"Markers unavailable ({exc}); the F/T and the planned lengths are the only stops.")

    def face_gap(red_b: np.ndarray, purple_b: np.ndarray) -> float:
        """Block face to box face along the normal (metres, > 0 = air)."""
        return float((purple_b - red_b) @ n_face) - BLOCK_HALF_T

    if markers:
        gap0 = face_gap(red, purple)
        yaw = block_yaw_vs_face(tip_path, purple_path)
        logger.info(f"Markers: red {np.round(red, 4)}, purple {np.round(purple, 4)}, face normal {np.round(n_face, 3)}; block face to box face "
                    f"{gap0 * 100:.1f} cm" + (f", block yaw vs the face {np.rad2deg(yaw):+.1f} deg" if yaw is not None else ""))
        if gap0 <= flush_tol_cm / 100.0:
            logger.info(f"The block face is already within {flush_tol_cm:.1f} cm of the box face; nothing to do.")
            return True

    comps = [COMPONENT[s] for s in SIDES]
    if not plan_only:
        foreign = detect_foreign_commands(comps)
        busy = {c: n for c, n in foreign.items() if n > 0}
        if busy and not ignore_foreign_commands:
            logger.error(
                "Another controller is streaming "
                + ", ".join(f"{COMMAND_TOPICS[c]} ({n} msgs in 0.7 s)" for c, n in busy.items())
                + "; the sim keeps the latest command. Stop it (e.g. omniteleop) and retry; --ignore-foreign-commands overrides."
            )
            return False
    setpoints = read_setpoints(comps) if (anchor_setpoints and not plan_only) else {}
    configuration, _tasks, solver, idx = build_ik()
    bot = connect()
    do_shutdown = not leave_held
    try:
        sync_from_robot(bot, configuration, idx)
        model = configuration.model
        p0 = {s: configuration.get_transform_frame_to_world(EE[s]).translation.copy() for s in SIDES}
        R0 = {s: configuration.get_transform_frame_to_world(EE[s]).rotation.copy() for s in SIDES}
        off: dict = {}
        last_groups: dict = {}
        logger.info("Grippers (robot base): " + ", ".join(f"{s} {np.round(p0[s], 4)}" for s in SIDES))

        def tip(s: str, q_arm: np.ndarray | None = None) -> np.ndarray:
            """Finger tip of ``s`` for the model's current joints, or for ``q_arm`` (restored afterwards)."""
            if q_arm is None:
                T = configuration.get_transform_frame_to_world(EE[s])
                return T.translation + TIP_AHEAD_M * T.rotation[:, 2]
            q_keep = configuration.q.copy()
            q = q_keep.copy()
            q[idx[s]] = q_arm
            configuration.update(np.clip(q, model.lowerPositionLimit, model.upperPositionLimit))
            out = tip(s)
            configuration.update(q_keep)
            return out

        tip_start = {s: tip(s) for s in SIDES}

        # -- stage 1 plan: per-arm orientation-held translation ---------------------------------------------------------
        def plan_translations(label: str) -> dict:
            """{side: (rows (k+1, 7) from the measured joints, reach_m)} for the arms that can move >= 2 mm."""
            plans: dict = {}
            if push_len <= 0.0:
                return plans
            for s in SIDES:
                res = plan_translation(configuration, solver, [s], push_len * PUSH_DIR, step_m, False)
                if res is None:
                    logger.warning(f"{label}: the {s} arm cannot translate left at all (joint limits); it is left alone in stage 1.")
                    continue
                rcfg, q_s, knots, ridx = res
                base = rcfg.get_transform_frame_to_world(EE[s]).translation.copy()
                rcfg.update(knots[-1])
                reach = float((rcfg.get_transform_frame_to_world(EE[s]).translation - base) @ PUSH_DIR)
                drift = np.rad2deg(rot_err(rcfg.get_transform_frame_to_world(EE[s]).rotation, R0[s]))
                rcfg.update(q_s)
                if reach < 0.002:
                    logger.warning(f"{label}: the {s} arm has less than 2 mm of translation room; it is left alone in stage 1.")
                    continue
                (logger.info if reach >= push_len - 0.003 else logger.warning)(
                    f"[{'REACHABLE' if reach >= push_len - 0.003 else 'PARTIAL'}] {label}: {s} arm translates {reach * 100:.1f} of {push_cm:.1f} cm "
                    f"in {len(knots)} steps, orientation drift {drift:.1f} deg"
                )
                plans[s] = (np.array([q_s[ridx[s]]] + [k[ridx[s]] for k in knots]), reach)
            # arms that fall short add a wrist turn after the translation so they keep pushing
            for s in SIDES:
                if wrist_deg <= 0.0:
                    break
                rows, reach = plans.get(s, (np.array([configuration.q[idx[s]].copy()]), 0.0))
                if reach >= push_len - 0.003:
                    continue
                qa = rows[-1].copy()
                t0 = tip(s, qa)
                cand = []
                for j in WRIST_CANDIDATES:
                    qp = qa.copy()
                    qp[j] += 0.01
                    cand.append((j, float((tip(s, qp) - t0)[1] / 0.01)))
                j, dy = max(cand, key=lambda c: abs(c[1]))
                sign = 1.0 if dy > 0 else -1.0
                qi = model.idx_qs[model.getJointId(ARM_JOINTS[s][j])]
                room = (float(model.upperPositionLimit[qi]) - 0.01 - qa[j]) if sign > 0 else (qa[j] - float(model.lowerPositionLimit[qi]) - 0.01)
                turn = sign * min(np.deg2rad(wrist_deg), max(room, 0.0))
                if abs(turn) < np.deg2rad(1.0):
                    logger.warning(f"{label}: {s} arm has no wrist room on {ARM_JOINTS[s][j]}; no wrist turn.")
                    continue
                n_w = max(int(np.ceil(abs(turn) / np.deg2rad(1.0))), 2)
                W = np.tile(qa, (n_w, 1))
                W[:, j] = qa[j] + turn * np.linspace(1.0 / n_w, 1.0, n_w)
                qa_end = W[-1]
                travel = float((tip(s, qa_end) - t0)[1])
                logger.info(f"{label}: {s} arm adds a wrist turn {ARM_JOINTS[s][j]} {np.rad2deg(turn):+.1f} deg after its {reach * 100:.1f} cm translation "
                            f"-> finger tip a further {travel * 100:+.1f} cm in +Y (tip {abs(dy) * 100:.1f} cm per rad; candidates "
                            + ", ".join(f"{ARM_JOINTS[s][c]} {d * 100:+.1f}" for c, d in cand) + " cm/rad)")
                plans[s] = (np.vstack([rows, W]), reach + max(travel, 0.0))
            return plans

        # -- stage 2 plan: j2 of one arm, other joints held -------------------------------------------------------------
        def plan_j2(label: str):
            """(rows (k+1, 7), turn_rad, tip_travel_m) for ``j2_side`` or None if no room / disabled."""
            if j2_deg <= 0.0:
                return None
            s = j2_side
            qa = configuration.q[idx[s]].copy()
            t0 = tip(s)
            qa_p = qa.copy()
            qa_p[J2] += 0.01
            dy = float((tip(s, qa_p) - t0)[1] / 0.01)  # m of tip travel along +Y per rad
            sign = 1.0 if dy > 0 else -1.0
            qi = model.idx_qs[model.getJointId(ARM_JOINTS[s][J2])]
            lo, hi = float(model.lowerPositionLimit[qi]), float(model.upperPositionLimit[qi])
            room = (hi - 0.01 - qa[J2]) if sign > 0 else (qa[J2] - lo - 0.01)
            turn = sign * min(np.deg2rad(j2_deg), max(room, 0.0))
            name = ARM_JOINTS[s][J2]
            if abs(turn) < np.deg2rad(1.0):
                logger.warning(f"{label}: {name} is {np.rad2deg(max(room, 0.0)):.1f} deg from its limit in the +Y direction; stage 2 has nothing to do.")
                return None
            qa_end = qa.copy()
            qa_end[J2] = qa[J2] + turn
            travel = abs(float((tip(s, qa_end) - t0)[1]))
            logger.info(f"{label}: {name} {np.rad2deg(qa[J2]):+.1f} -> {np.rad2deg(qa_end[J2]):+.1f} deg ({np.rad2deg(turn):+.1f}; tip {abs(dy) * 100:.1f} cm per rad, "
                        f"{np.rad2deg(room):.1f} deg to the limit) -> finger tip {np.sign(dy) * travel * 100:+.1f} cm in +Y; other joints held")
            t_j2 = max(1.0, abs(turn) / max(j2_vel, 1e-3), travel / max(tip_speed, 1e-3))
            n_k = max(int(np.ceil(t_j2 * 25.0)), 2)
            rows = np.tile(qa, (n_k + 1, 1))
            rows[:, J2] = qa[J2] + turn * np.linspace(0.0, 1.0, n_k + 1)
            return rows, turn, travel, t_j2

        def stream_rows(per_arm: dict, min_s: float, stop_check=None, hold: float = 0.0, leash: float | None = None,
                        clamp: float | None = None) -> bool:
            """Stream the arms' rows together over one profile (resampled to a common knot count); anchored."""
            nonlocal off, last_groups
            n = max(len(r) - 1 for r in per_arm.values())
            groups: dict = {}
            paths: dict = {}
            for s, rows in per_arm.items():
                m = len(rows) - 1
                src, dst = np.arange(m + 1, dtype=float), np.linspace(0.0, float(m), n + 1)
                Kr = np.stack([np.interp(dst, src, rows[:, j]) for j in range(rows.shape[1])], axis=1)
                comp = COMPONENT[s]
                groups[comp] = Kr + off[comp]
                paths[comp] = float(np.abs(np.diff(Kr, axis=0)).sum(axis=0).max())
            t_s = max(min_s, 1.0, max(paths.values()) / max(max_joint_vel, 1e-3))
            use_leash = leash_rad if leash is None else leash
            extra = {} if clamp is None else {"step_clamp_rad": clamp}
            logger.info(f"Streaming {' + '.join(groups)} over {t_s:.1f}s (leash {use_leash:.3f} rad"
                        + (f", setpoint clamp / force cap {clamp:.3f} rad" if clamp is not None else "") + ")...")
            last_cmd, stopped = stream(bot, groups, t_s, config.CONTROL_DT, stop_check=stop_check, leash_rad=use_leash, final_hold_s=hold, **extra)
            last_groups = {c: last_cmd[c].copy() for c in groups}
            time.sleep(0.05)
            sync_from_robot(bot, configuration, idx)
            for c in groups:  # refresh the sag offsets from what was actually sent
                off[c] = np.clip(last_cmd[c] - configuration.q[idx[GROUP[c]]], -MAX_ANCHOR_OFFSET, MAX_ANCHOR_OFFSET)
            return stopped

        plans1 = plan_translations("stage 1")
        preview2 = plan_j2("stage 2 (preview from the current pose)")
        if not plans1 and preview2 is None:
            logger.error("Nothing to do: no arm can translate left and stage 2 has no room. Nothing commanded.")
            return False
        if plan_only:
            logger.info("Plan only: nothing commanded, F/T gate not armed (stage 2 is re-planned from the measured pose after stage 1).")
            return True

        # -- gate ----------------------------------------------------------------------------------------------------
        gate = ContactGate(contact_n, confirm_ticks, sample_hz=1.0 / config.CONTROL_DT, sides=tuple(SIDES),
                           tare_samples=tare_samples, warmup_s=warmup_s, rest_s=rest_s)
        logger.info("Arming the wrist contact gates (hands still)...")
        if not gate.arm(bot):
            if not allow_blind:
                logger.error("No usable wrist wrench data (state/wrench/*): the push would be blind. Nothing pushed (--allow-blind overrides).")
                return False
            logger.warning("Blind push: no wrench data; only the markers / planned lengths can stop it.")
        off = anchor_offsets(configuration, idx, setpoints, comps)

        def pushed_sides() -> dict:
            sync_from_robot(bot, configuration, idx)
            return {s: float((configuration.get_transform_frame_to_world(EE[s]).translation - p0[s]) @ PUSH_DIR) for s in SIDES}

        gate.travel_fn = lambda: max(pushed_sides().values())
        state: dict = {"stop": None, "last_log": 0.0, "t0": None, "blanked": 0, "hist": [], "last_prog": 0.0,
                       "prog0": None, "stall_armed": False}

        def wrist_force_n() -> float:
            """Largest filtered wrist force the gate currently sees (N); 0 when not armed."""
            try:
                return max((float(np.linalg.norm(v)) for v in gate.filt.values()), default=0.0) if gate.armed else 0.0
            except Exception:
                return 0.0

        def stalled(progress_m: float) -> bool:
            """True when ``progress_m`` (any monotone measure of the push) has moved < stall_mm over the last stall_s.
            Only after contact: the watch arms once progress since the stage start reaches stall_arm_mm or a wrist reads
            stall_force_n; before that the block gap naturally stands still. Sampled every 0.1 s."""
            if stall_s <= 0.0:
                return False
            now = time.time()
            if now - state["last_prog"] < 0.1:
                return False
            state["last_prog"] = now
            if state["prog0"] is None:
                state["prog0"] = progress_m
            if not state["stall_armed"]:
                moved = progress_m - state["prog0"]
                force = wrist_force_n()
                if moved >= stall_arm_mm / 1000.0 or force >= stall_force_n:
                    state["stall_armed"] = True
                    state["hist"] = []
                    logger.info(f"  stall watch armed (block moved {moved * 1000:.0f} mm, wrist force {force:.0f} N)")
                else:
                    return False
            hist = state["hist"]
            hist.append((now, progress_m))
            state["hist"] = hist = [h for h in hist if now - h[0] <= stall_s]
            if len(hist) < 3 or now - hist[0][0] < 0.9 * stall_s:
                return False
            span = max(h[1] for h in hist) - min(h[1] for h in hist)
            if span < stall_mm / 1000.0:
                state["stop"] = "stall"
                logger.info(f"STALL: the push has not progressed ({span * 1000:.1f} mm in the last {stall_s:.1f}s) -- stopping, measured pose held.")
                return True
            return False

        def stop_check(b) -> bool:
            if state["t0"] is None:
                state["t0"] = time.time()
                state["hist"], state["last_prog"], state["prog0"], state["stall_armed"] = [], 0.0, None, False
            msg = gate.check(b) if gate.armed else None
            if msg and time.time() - state["t0"] < gate_blank_s:
                state["blanked"] += 1
                gate._above = 0
                if state["blanked"] == 1:
                    logger.info(f"  (contact gate suppressed during the first {gate_blank_s:.1f}s: {msg})")
                msg = None
            if msg:
                state["stop"] = "contact"
                logger.info(msg)
                return True
            if not markers:
                return stalled(max(pushed_sides().values())) if stall_s > 0.0 and time.time() - state["last_prog"] >= 0.1 else False
            try:
                gap_now = face_gap(quiet_xyz(tip_path), quiet_xyz(purple_path))
            except (ValueError, OSError):
                return False
            if stalled(-gap_now):
                return True
            if gap_now <= flush_tol_cm / 100.0:
                state["stop"] = "flush"
                logger.info(f"FLUSH: block face {gap_now * 100:.1f} cm from the box face (<= {flush_tol_cm:.1f} cm) -- stopping the push.")
                return True
            now = time.time()
            if now - state["last_log"] >= MARKER_LOG_PERIOD_S:
                state["last_log"] = now
                logger.info(f"  markers: block face {gap_now * 100:.1f} cm from the box face")
            return False

        # -- stage 1 --------------------------------------------------------------------------------------------------
        stopped = False
        pushed = {s: 0.0 for s in SIDES}
        if plans1:
            logger.info("Stage 1: " + " + ".join(f"{s} arm {r * 100:.1f} cm" for s, (_rows, r) in plans1.items()) + " (translation with orientation held, plus a wrist turn where the arm falls short)"
                        + (" and wrist contact detection..." if gate.armed else " (blind)..."))
            state["t0"] = None
            stopped = stream_rows({s: rows for s, (rows, _r) in plans1.items()}, max(r for _rows, r in plans1.values()) / min(push_speed, tip_speed), stop_check)
            pushed = pushed_sides()
            logger.info("Stage 1 done (measured): " + ", ".join(f"{s} {pushed[s] * 100:.1f} cm" for s in SIDES) + ("; stopped early" if stopped else ""))

        # -- stage 2 --------------------------------------------------------------------------------------------------
        j2_txt = ""
        hold_clamp: float | None = None
        if not stopped and j2_deg > 0.0:
            plan2 = plan_j2("stage 2")
            if plan2 is not None:
                rows, turn, travel, t_j2 = plan2
                q_before = configuration.q[idx[j2_side]].copy()
                logger.info(f"Stage 2: turning {ARM_JOINTS[j2_side][J2]} {np.rad2deg(turn):+.1f} deg over >= {t_j2:.1f}s (tip cap {tip_speed * 100:.0f} cm/s)"
                            + (" with wrist contact detection..." if gate.armed else " (blind)..."))
                state["t0"] = None
                stopped = stream_rows({j2_side: rows}, t_j2, stop_check, leash=j2_leash_rad, clamp=j2_force_cap_rad)
                hold_clamp = j2_force_cap_rad
                qa = configuration.q[idx[j2_side]]
                others = float(np.abs(np.delete(qa - q_before, J2)).max())
                j2_txt = (f"; stage 2 {ARM_JOINTS[j2_side][J2]} {np.rad2deg(qa[J2] - q_before[J2]):+.1f} deg (other joints drifted <= {np.rad2deg(others):.1f} deg)")

        if not stopped and last_groups and hold_s > 0.0:
            state["t0"] = None
            extra = {"step_clamp_rad": hold_clamp} if hold_clamp is not None else {}
            logger.info(f"Holding the final push targets for up to {hold_s:.0f}s (the block may still creep"
                        + (f"; force cap {hold_clamp:.3f} rad" if hold_clamp is not None else "") + ")...")
            stream(bot, {c: np.array([v, v]) for c, v in last_groups.items()}, 0.5, config.CONTROL_DT,
                   stop_check=stop_check, leash_rad=leash_rad, final_hold_s=hold_s, **extra)
            time.sleep(0.05)
            sync_from_robot(bot, configuration, idx)

        # -- report ---------------------------------------------------------------------------------------------------
        pushed = pushed_sides()
        tips_txt = "; finger tips " + ", ".join(f"{s} {(tip(s) - tip_start[s])[1] * 100:+.1f} cm in +Y" for s in SIDES)
        after_txt = ""
        if markers:
            try:
                red_now, purple_now = quiet_xyz(tip_path), quiet_xyz(purple_path)
                yaw_now = block_yaw_vs_face(tip_path, purple_path)
                after_txt = (f"; block face {face_gap(red_now, purple_now) * 100:.1f} cm from the box face"
                             + (f", block yaw vs the face {np.rad2deg(yaw_now):+.1f} deg" if yaw_now is not None else ""))
            except (ValueError, OSError):
                after_txt = "; markers unavailable afterwards"
        reason = {"contact": f"CONTACT -- stopped by the wrist F/T (>= {contact_n:.0f} N)", "flush": "FLUSH -- markers show the block against the box",
                  "stall": f"STALL -- the block stopped moving (< {stall_mm:.0f} mm in {stall_s:.0f}s), pushing on would only build force",
                  None: "planned lengths reached without a stop condition"}[state["stop"]]
        force_txt = ((f"; peak filtered |dF| {gate.peak_n:.1f} N" + (f" ({gate.stop_side} wrist stopped it)" if gate.stop_side else "")
                      + (f", {state['blanked']} tick(s) suppressed at stream starts" if state["blanked"] else "")) if gate.armed else "")
        (logger.info if stopped else logger.warning)(
            f"{reason}; stage 1 translated " + ", ".join(f"{s} {pushed[s] * 100:.1f} cm" for s in SIDES) + f"{j2_txt}{tips_txt}{after_txt}{force_txt}. Pose held."
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
