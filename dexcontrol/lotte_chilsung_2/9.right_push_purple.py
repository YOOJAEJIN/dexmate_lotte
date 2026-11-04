# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Right hand, two moves: stretch out along world +X, then press the purple
box center with the tool axis facing it head-on. Left EE is held in place;
the torso may assist.

  1. reach : translate the right EE along world +X by ``--reach-cm``
             (orientation is rotated during this move so the EE local +Z ends
             up pointing along world −X, straight at the ball)
  2. push  : straight line from there to (purple_x + push_end, purple_y,
             purple_z + push_height), orientation held — the hand comes in
             from the +X side and presses the box toward the conveyor stack.

If the wrist cannot complete the re-aim during the reach, the reach is done
with the current orientation and the re-aim happens during the push instead.

Example::

    python right_push_purple.py --dry-run
    python right_push_purple.py                 # reach 30 cm, push to +4 cm of purple
    python right_push_purple.py --reach-cm 40
    python right_push_purple.py --push-end 0.06 # gentler press
    python right_push_purple.py --lock-torso    # arms only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import tyro
from dexcomm import RateLimiter
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import (  # noqa: E402
    IK_MAX_ITERS,
    IK_MAX_REPS,
    LEFT_ARM_JOINTS,
    MAX_JOINT_JUMP,
    POS_TOL,
    RIGHT_ARM_JOINTS,
    ROT_TOL,
    SOFT_POS_TOL,
    SOFT_ROT_TOL,
    connect_bot,
    create_motion_manager,
    disconnect_leave_held,
    execute_trajectory,
    max_adjacent_jump,
    rot_err_deg,
)
from _world_translate import load_world_frame  # noqa: E402
from approach_purple_to_red import _DEFAULT_PURPLE_CANDIDATES, _resolve_existing  # noqa: E402
from retract_and_push import _MARGIN_FALLBACKS, _load_base_world  # noqa: E402

# Phase orientation modes
FIXED = "fixed"      # hold the orientation at the start of the phase
INTERP = "interp"    # rotate from the start orientation to the phase's target
RELAXED = "relaxed"  # position-only (wrist free)


def rot_align_z(rot: np.ndarray, z_des: np.ndarray) -> np.ndarray:
    """Smallest rotation applied to ``rot`` so its +Z column points along ``z_des``."""
    z_cur = rot[:, 2] / np.linalg.norm(rot[:, 2])
    z_des = np.asarray(z_des, dtype=float)
    z_des = z_des / np.linalg.norm(z_des)
    v = np.cross(z_cur, z_des)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z_cur, z_des))
    if s < 1e-9:
        if c > 0:
            return rot.copy()
        return pin.exp3(rot[:, 0] * np.pi) @ rot  # 180°: flip about current x
    return pin.exp3((v / s) * np.arctan2(s, c)) @ rot


def _solve(mm, tl: pin.SE3, tr: pin.SE3, fixed_rot: bool) -> bool:
    """IK for both EEs (left fixed, right target). Restores on failure."""
    q_prev = dict(mm.get_joint_pos_dict())
    try:
        posture = mm.local_ik_solver.task_dict.get("posture")
        if posture is not None:
            posture.set_target_from_configuration(mm.local_ik_solver.configuration)
    except AttributeError:
        pass
    target_rot = {"L_ee": tl.rotation}
    if fixed_rot:
        target_rot["R_ee"] = tr.rotation
    perr, rerr = 1e9, 1e9
    for _ in range(IK_MAX_REPS):
        sol, coll, lim = mm.ik(
            target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
            target_rot=target_rot,
            max_iterations=IK_MAX_ITERS,
        )
        if sol is None or coll or not lim:
            mm.set_joint_pos(q_prev)
            return False
        mm.set_joint_pos(sol)
        cur = mm.fk(["L_ee", "R_ee"])
        perr = max(
            float(np.linalg.norm(cur["R_ee"].translation - tr.translation)),
            float(np.linalg.norm(cur["L_ee"].translation - tl.translation)),
        )
        rerr = rot_err_deg(cur["L_ee"].rotation, tl.rotation)
        if fixed_rot:
            rerr = max(rerr, rot_err_deg(cur["R_ee"].rotation, tr.rotation))
        if perr < POS_TOL and rerr < ROT_TOL:
            return True
    if perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL:
        return True
    mm.set_joint_pos(q_prev)
    return False


def _track(mm, phases, step_m: float):
    """Track right-EE (position, orientation-mode) targets; left EE held fixed.

    ``phases``: list of (label, r_pos_target_base, rot_target_base|None, mode).
    """
    poses = mm.fk(["L_ee", "R_ee"])
    tl = pin.SE3(poses["L_ee"].homogeneous.copy())
    rot_r = poses["R_ee"].rotation.copy()
    rp = poses["R_ee"].translation.copy()
    configs: list[dict] = []
    report: list[tuple[str, float]] = []
    for label, r_tgt, rot_tgt, mode in phases:
        rot_start = rot_r.copy()
        log_delta = (
            pin.log3(rot_start.T @ rot_tgt) if (mode == INTERP and rot_tgt is not None) else None
        )
        span_pos = float(np.linalg.norm(r_tgt - rp))
        span_rot = float(np.linalg.norm(log_delta)) if log_delta is not None else 0.0
        # ~1 cm per step, and no more than ~1.5° of rotation per step
        n = max(int(np.ceil(span_pos / max(step_m, 1e-4))), int(np.ceil(np.rad2deg(span_rot) / 1.5)), 1)
        done = 0
        for i in range(1, n + 1):
            a = i / n
            rot_a = rot_start @ pin.exp3(a * log_delta) if log_delta is not None else rot_start
            tr = pin.SE3(rot_a, (1 - a) * rp + a * r_tgt)
            q_before = dict(mm.get_joint_pos_dict())
            if not _solve(mm, tl, tr, fixed_rot=(mode != RELAXED)):
                break
            q_next = dict(mm.get_joint_pos_dict())
            if max(abs(q_next[k] - q_before[k]) for k in q_before) > MAX_JOINT_JUMP:
                mm.set_joint_pos(q_before)
                break
            configs.append(q_next)
            done = i
        frac = done / n
        report.append((label, frac))
        cur = mm.fk(["R_ee"])["R_ee"]
        rp = cur.translation.copy()
        rot_r = cur.rotation.copy()  # continue from the achieved orientation
        if frac < 0.999:
            break
    return configs, report


def main(
    reach_cm: float = 30.0,
    push_end: float = 0.04,
    push_height: float = 0.0,
    face_purple: bool = True,
    relax_push: bool = False,
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    world_yaw_deg: float | None = None,
    step_m: float = 0.01,
    duration: float = 8.0,
    control_hz: float = 100.0,
    torso_damping: float = 5.0,
    lock_torso: bool = False,
    collision_margin: float = 0.015,
    dry_run: bool = False,
    leave_held: bool = True,
) -> None:
    """Right hand: stretch along world +X, then press the purple center.

    Args:
        reach_cm: How far the right EE moves along world +X first (cm).
        push_end: Right EE world-X offset from purple at the end of the push (m).
            Smaller than the box face offset (~0.07) presses the box into the stack.
        push_height: Push height relative to purple Z (m).
        face_purple: Re-aim the EE +Z (tool axis) along world −X so the push is
            head-on (during the reach; falls back to during the push).
        relax_push: Position-only tracking in the push (wrist free).
        tip_file: Red tip file (base + world lines). Default: sim file.
        purple_file: Purple center file (base + world lines). Default: sim file.
        world_yaw_deg: Override base→world yaw instead of deriving it.
        step_m: IK waypoint spacing (m).
        duration: Nominal stream time for the whole path (s).
        control_hz: Command rate.
        torso_damping: Torso IK damping (higher = less torso, more arm).
        lock_torso: Freeze torso (arms only).
        collision_margin: Pink self-collision barrier (m); falls back smaller.
        dry_run: Plan and report only.
        leave_held: Exit without ``stop()``.
    """
    wf = load_world_frame(tip_file, purple_file, world_yaw_deg)
    pur_b, pur_w = _load_base_world(
        _resolve_existing(purple_file, _DEFAULT_PURPLE_CANDIDATES, "purple center"), "purple"
    )
    minus_x_b = wf.vec_to_base([-1.0, 0.0, 0.0])
    logger.warning("Right hand should be OPEN or empty; grippers are not commanded.")

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
        try:
            mm = torso_joints = None
            for margin in (collision_margin, *[m for m in _MARGIN_FALLBACKS if m < collision_margin]):
                try:
                    mm, torso_joints, _ = create_motion_manager(
                        bot, torso_damping=torso_damping, lock_torso=lock_torso, collision_margin=margin
                    )
                    logger.info(f"MotionManager ready (collision_margin={margin}).")
                    break
                except RuntimeError as exc:
                    if "Collision barrier" not in str(exc):
                        raise
                    logger.warning(f"collision_margin={margin} too large for current pose; retrying smaller.")
            if mm is None:
                logger.error("Could not build MotionManager. Not moving.")
                return

            q_init = dict(mm.get_joint_pos_dict())
            poses = mm.fk(["L_ee", "R_ee"])
            r0_w = wf.to_world(poses["R_ee"].translation.copy())
            l0_w = wf.to_world(poses["L_ee"].translation.copy())
            rot_r0 = poses["R_ee"].rotation.copy()
            z0_w = wf.R @ rot_r0[:, 2]
            logger.info(
                f"World: purple {np.round(pur_w, 3)}  R_ee {np.round(r0_w, 3)} "
                f"(rel {np.round(r0_w - pur_w, 3)})  tool +Z {np.round(z0_w, 3)}  "
                f"L_ee {np.round(l0_w, 3)} (held)"
            )
            rot_push = rot_align_z(rot_r0, minus_x_b) if face_purple else None
            if rot_push is not None:
                logger.info(f"Re-aim tool +Z → world −X: wrist rotation {rot_err_deg(rot_push, rot_r0):.1f} deg")

            p1_w = r0_w + np.array([reach_cm / 100.0, 0.0, 0.0])
            p2_w = np.array([pur_w[0] + push_end, pur_w[1], pur_w[2] + push_height])
            p1_b, p2_b = wf.to_base(p1_w), wf.to_base(p2_w)
            push_mode = RELAXED if relax_push else FIXED
            logger.info(
                f"  reach : +X {reach_cm:.0f} cm → R(world) {np.round(p1_w, 3)}\n"
                f"  push  : → R(world) {np.round(p2_w, 3)}  (length {np.linalg.norm(p2_w - p1_w):.3f} m)"
            )

            plans = []
            if rot_push is not None:
                plans.append(("aim during reach", [("reach", p1_b, rot_push, INTERP), ("push", p2_b, None, push_mode)]))
                plans.append(("aim during push", [("reach", p1_b, None, FIXED), ("push", p2_b, rot_push, RELAXED if relax_push else INTERP)]))
            else:
                plans.append(("keep current aim", [("reach", p1_b, None, FIXED), ("push", p2_b, None, push_mode)]))

            best = None
            for label, phases in plans:
                mm.set_joint_pos(q_init)
                cfgs, rep = _track(mm, phases, step_m)
                ok = len(rep) == 2 and all(f >= 0.999 for _, f in rep)
                prog = sum(f for _, f in rep)
                logger.info(f"  plan '{label}': " + "  ".join(f"{l}:{100 * f:.0f}%" for l, f in rep) + ("  ✓" if ok else ""))
                if best is None or ok or prog > best[3]:
                    best = (label, cfgs, rep, prog)
                if ok:
                    break
            label, configs, report, _ = best
            mm.set_joint_pos(q_init)
        finally:
            if hb is not None and hasattr(hb, "resume"):
                try:
                    hb.resume()
                except Exception:
                    pass

        summary = "  ".join(f"{lab}:{100 * frac:.0f}%" for lab, frac in report)
        complete = len(report) == 2 and all(f >= 0.999 for _, f in report)
        if not configs:
            logger.error(f"No feasible motion from the current pose ({summary}). Not moving.")
            return
        q_last = configs[-1]
        jump = max_adjacent_jump([q_init] + configs)
        torso_delta = {k: round(float(q_last[k] - q_init[k]), 2) for k in q_init if k.startswith("torso_")}
        mm.set_joint_pos(q_last)
        fin = mm.fk(["L_ee", "R_ee"])
        fin_r = wf.to_world(fin["R_ee"].translation.copy())
        fin_l = wf.to_world(fin["L_ee"].translation.copy())
        z_fin_w = wf.R @ fin["R_ee"].rotation[:, 2].copy()
        aim_err = float(np.rad2deg(np.arccos(np.clip(np.dot(z_fin_w, [-1.0, 0.0, 0.0]), -1.0, 1.0))))
        mm.set_joint_pos(q_init)
        logger.info(
            f"[{'COMPLETE' if complete else 'PARTIAL'}] plan '{label}': {summary}\n"
            f"  {len(configs)} configs, max adjacent jump {jump:.2f} rad, torso Δ {torso_delta}\n"
            f"  R final (world) {np.round(fin_r, 3)}  rel purple {np.round(fin_r - pur_w, 3)}\n"
            f"  tool +Z final (world) {np.round(z_fin_w, 3)}  angle to world −X {aim_err:.1f} deg\n"
            f"  L final (world) {np.round(fin_l, 3)}  (moved {np.linalg.norm(fin_l - l0_w) * 1000:.0f} mm)"
        )
        if not complete:
            logger.warning("Only a prefix is feasible; motion stops at the last feasible point.")
        if dry_run:
            logger.info("Dry run: not moving. Pose left held.")
            return

        execute_trajectory(bot, torso_joints, q_init, configs, duration, control_hz)
        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        q = np.array([q_last[n] for n in names], dtype=float)
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.left_arm.set_joint_pos(q[:7])
            bot.right_arm.set_joint_pos(q[7:14])
            if torso_joints:
                bot.torso.set_joint_pos(q[14:])
            rate.sleep()
        logger.info("Right push completed; final pose left held.")
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
    except KeyboardInterrupt:
        logger.warning("Interrupted — leaving last commanded pose held.")
    finally:
        if do_shutdown:
            bot.shutdown()
        else:
            disconnect_leave_held(bot)
            logger.info("Client disconnected without stop() — pose held.")


if __name__ == "__main__":
    tyro.cli(main)
