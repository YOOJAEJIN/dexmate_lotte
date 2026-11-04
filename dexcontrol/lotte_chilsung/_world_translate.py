# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Shared driver: translate L/R end-effectors by world-frame vectors with
fixed orientations (plan with local IK, optional dry-run, execute, leave held).

Used by ``translate_both_world.py`` (same vector for both hands) and
``release_retract.py`` (different vector per hand).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dexcomm import RateLimiter
from loguru import logger

from _bimanual_box import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    connect_bot,
    create_motion_manager,
    disconnect_leave_held,
    execute_trajectory,
    max_adjacent_jump,
    rot_err_deg,
)
from approach_purple_to_red import (
    _DEFAULT_PURPLE_CANDIDATES,
    _DEFAULT_TIP_CANDIDATES,
    _resolve_existing,
)
from retract_and_push import _MARGIN_FALLBACKS, WorldFrame, _load_base_world, _track


def load_world_frame(
    tip_file: Path | None,
    purple_file: Path | None,
    world_yaw_deg: float | None,
) -> WorldFrame:
    """Base→world transform from the sim marker files (or an explicit yaw)."""
    tip_path = _resolve_existing(tip_file, _DEFAULT_TIP_CANDIDATES, "red tip")
    purple_path = _resolve_existing(purple_file, _DEFAULT_PURPLE_CANDIDATES, "purple center")
    red_b, red_w = _load_base_world(tip_path, "red tip")
    pur_b, pur_w = _load_base_world(purple_path, "purple center")
    if world_yaw_deg is None:
        return WorldFrame(pur_b, pur_w, red_b, red_w)
    return WorldFrame.from_yaw(world_yaw_deg, pur_b, pur_w)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("direction must be non-zero")
    return v / n


def run_translation(
    *,
    wf: WorldFrame,
    l_delta_w: np.ndarray,
    r_delta_w: np.ndarray,
    step_m: float = 0.005,
    duration: float = 2.0,
    control_hz: float = 100.0,
    torso_damping: float = 5.0,
    lock_torso: bool = False,
    collision_margin: float = 0.015,
    ignore_self_collision: bool = False,
    dry_run: bool = False,
    leave_held: bool = True,
) -> None:
    """Plan (and optionally execute) a straight-line translation of both EEs.

    ``l_delta_w`` / ``r_delta_w`` are world-frame displacement vectors (m).
    Orientations stay fixed. Never calls ``stop()`` unless ``leave_held`` is
    False, so a held object is not dropped on exit.
    """
    l_delta_w = np.asarray(l_delta_w, dtype=float).reshape(3)
    r_delta_w = np.asarray(r_delta_w, dtype=float).reshape(3)
    l_delta_b = wf.vec_to_base(l_delta_w)
    r_delta_b = wf.vec_to_base(r_delta_w)
    logger.info(
        f"base→world yaw {np.rad2deg(wf.yaw):.1f} deg\n"
        f"  L delta world {np.round(l_delta_w, 4)} → base {np.round(l_delta_b, 4)}\n"
        f"  R delta world {np.round(r_delta_w, 4)} → base {np.round(r_delta_b, 4)}"
    )
    if np.linalg.norm(l_delta_w) < 1e-4 and np.linalg.norm(r_delta_w) < 1e-4:
        logger.info("Both deltas ≈ 0; nothing to do.")
        return

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused for IK planning.")

        try:
            mm = torso_joints = None
            margins = (collision_margin, *[m for m in _MARGIN_FALLBACKS if m < collision_margin])
            for margin in margins:
                try:
                    mm, torso_joints, _ = create_motion_manager(
                        bot,
                        torso_damping=torso_damping,
                        lock_torso=lock_torso,
                        collision_margin=margin,
                    )
                    logger.info(f"MotionManager ready (collision_margin={margin}).")
                    break
                except RuntimeError as exc:
                    if "Collision barrier" not in str(exc):
                        raise
                    detail = str(exc).split("failed:", 1)[-1].strip()
                    logger.warning(
                        f"collision_margin={margin} too large for current pose; retrying smaller. ({detail})"
                    )
            if mm is None and ignore_self_collision:
                logger.warning(
                    "Self-collision barrier DISABLED (--ignore-self-collision): "
                    "arms are already closer than any margin. Keep the motion small."
                )
                mm, torso_joints, _ = create_motion_manager(
                    bot,
                    torso_damping=torso_damping,
                    lock_torso=lock_torso,
                    avoid_self_collisions=False,
                )
            if mm is None:
                logger.error(
                    "Could not build MotionManager at any collision margin. Not moving. "
                    "(Arms are touching; retry with --ignore-self-collision for a small move.)"
                )
                return

            q_init = dict(mm.get_joint_pos_dict())
            poses = mm.fk(["L_ee", "R_ee"])
            l0 = poses["L_ee"].translation.copy()
            r0 = poses["R_ee"].translation.copy()
            l0_w, r0_w = wf.to_world(l0), wf.to_world(r0)
            logger.info(
                f"  L_ee world {np.round(l0_w, 3)} → {np.round(l0_w + l_delta_w, 3)}\n"
                f"  R_ee world {np.round(r0_w, 3)} → {np.round(r0_w + r_delta_w, 3)}"
            )
            configs, report = _track(
                mm, [("translate", l0 + l_delta_b, r0 + r_delta_b)], step_m
            )
        finally:
            if hb is not None and hasattr(hb, "resume"):
                try:
                    hb.resume()
                    logger.info("Heartbeat resumed.")
                except Exception:
                    pass

        frac = report[0][1] if report else 0.0
        if not configs:
            logger.error("No feasible motion from the current pose. Not moving.")
            return
        jump = max_adjacent_jump([q_init] + configs)
        q_last = configs[-1]
        torso_delta = {
            k: round(float(q_last[k] - q_init[k]), 3)
            for k in q_init
            if k.startswith("torso_")
        }
        # Orientation drift check: every knot vs. the initial EE rotations.
        rot_l0, rot_r0 = poses["L_ee"].rotation.copy(), poses["R_ee"].rotation.copy()
        max_rot_l = max_rot_r = 0.0
        for cfg in configs:
            mm.set_joint_pos(cfg)
            cur = mm.fk(["L_ee", "R_ee"])
            max_rot_l = max(max_rot_l, rot_err_deg(cur["L_ee"].rotation, rot_l0))
            max_rot_r = max(max_rot_r, rot_err_deg(cur["R_ee"].rotation, rot_r0))
        mm.set_joint_pos(q_last)
        fin = mm.fk(["L_ee", "R_ee"])
        # FK results alias pinocchio buffers — copy before resetting.
        fin_l = wf.to_world(fin["L_ee"].translation.copy())
        fin_r = wf.to_world(fin["R_ee"].translation.copy())
        mm.set_joint_pos(q_init)
        verdict = "COMPLETE" if frac >= 0.999 else f"PARTIAL {100 * frac:.0f}%"
        logger.info(
            f"[{verdict}] {len(configs)} configs, max adjacent jump {jump:.2f} rad, "
            f"torso Δ {torso_delta}\n"
            f"  EE orientation drift along path: L {max_rot_l:.2f} deg, R {max_rot_r:.2f} deg\n"
            f"  final L(world) {np.round(fin_l, 3)}  R(world) {np.round(fin_r, 3)}"
        )
        if frac < 0.999:
            logger.warning("Only a prefix is feasible; motion will stop at the last feasible point.")

        if dry_run:
            logger.info("Dry run: not moving. Pose left held.")
            return

        exec_duration = max(duration * frac, 0.5)
        logger.info(f"Executing {len(configs)} configs over {exec_duration:.1f}s...")
        execute_trajectory(bot, torso_joints, q_init, configs, exec_duration, control_hz)

        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        q = np.array([q_last[n] for n in names], dtype=float)
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.left_arm.set_joint_pos(q[:7])
            bot.right_arm.set_joint_pos(q[7:14])
            if torso_joints:
                bot.torso.set_joint_pos(q[14:])
            rate.sleep()
        logger.info("Translation completed; final pose left held.")
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
    except KeyboardInterrupt:
        logger.warning("Interrupted — leaving last commanded pose held.")
    finally:
        if do_shutdown:
            logger.info("Shutting down robot (--no-leave-held)")
            bot.shutdown()
        else:
            disconnect_leave_held(bot)
            logger.info("Client disconnected without stop() — pose held.")
