# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Squeeze both hands a little toward their midpoint (close the grasp gap).

From the current pose, moves L_ee and R_ee toward the mid-point between them
while keeping each EE orientation fixed. Torso stays locked. Useful for a
slight pinch after a bimanual hold.

Example::

    python squeeze_inward.py
    python squeeze_inward.py --cm 2
    python squeeze_inward.py --cm 2 --left-extra-cm 1
    python squeeze_inward.py --cm 3 --duration 2
    python squeeze_inward.py --goto-snapshot   # optional: go to recorded joints first
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
    create_motion_manager,
    disconnect_leave_held,
    execute_trajectory,
    max_adjacent_jump,
    rot_err_deg,
    connect_bot,
)
from dexcontrol.robot import Robot

# User-recorded hold pose (optional --goto-snapshot).
SNAPSHOT_Q: dict[str, float] = {
    "L_arm_j1": 0.8416649699211121,
    "L_arm_j2": -0.0001622792042326182,
    "L_arm_j3": 0.1495988816022873,
    "L_arm_j4": -1.4473217725753784,
    "L_arm_j5": -0.0011668592924252152,
    "L_arm_j6": 0.0015830519841983914,
    "L_arm_j7": 0.00017672963440418243,
    "R_arm_j1": -0.8335204720497131,
    "R_arm_j2": 0.0008295349543914199,
    "R_arm_j3": -0.15198761224746704,
    "R_arm_j4": -1.444549322128296,
    "R_arm_j5": -0.000389334571082145,
    "R_arm_j6": -0.001880192314274609,
    "R_arm_j7": 0.0014954495709389448,
    "head_j1": -0.6427745223045349,
    "head_j2": 1.2475245966925286e-05,
    "head_j3": -0.00017319197650067508,
    "torso_j1": 0.19074824452400208,
    "torso_j2": 1.4715925455093384,
    "torso_j3": 0.6378125548362732,
}

TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]
JOINT_NAMES = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + TORSO_JOINTS  # head is never commanded


def _solve_dual_ee(
    mm,
    tl: pin.SE3,
    tr: pin.SE3,
    max_reps: int = IK_MAX_REPS,
) -> bool:
    """Local IK to both EE targets; restore on failure."""
    q_prev = dict(mm.get_joint_pos_dict())
    try:
        solver = mm.local_ik_solver
        posture = solver.task_dict.get("posture")
        if posture is not None:
            posture.set_target_from_configuration(solver.configuration)
    except AttributeError:
        pass

    perr, rerr = 1.0, 90.0
    for _ in range(max_reps):
        sol, coll, lim = mm.ik(
            target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
            target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
            max_iterations=IK_MAX_ITERS,
        )
        if sol is None or coll or not lim:
            mm.set_joint_pos(q_prev)
            return False
        mm.set_joint_pos(sol)
        cur = mm.fk(["L_ee", "R_ee"])
        perr = max(
            float(np.linalg.norm(cur["L_ee"].translation - tl.translation)),
            float(np.linalg.norm(cur["R_ee"].translation - tr.translation)),
        )
        rerr = max(
            rot_err_deg(cur["L_ee"].rotation, tl.rotation),
            rot_err_deg(cur["R_ee"].rotation, tr.rotation),
        )
        if perr < POS_TOL and rerr < ROT_TOL:
            return True
    if perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL:
        return True
    mm.set_joint_pos(q_prev)
    return False


def plan_squeeze(
    mm,
    move_m: float,
    step_m: float = 0.005,
    left_extra_m: float = 0.0,
) -> tuple[dict, list[dict]] | None:
    """Plan configs that move each EE toward the L/R midpoint.

    ``left_extra_m`` adds extra inward travel for the left hand only so it can
    close the gap a bit more than the right.
    """
    poses = mm.fk(["L_ee", "R_ee"])
    left0 = pin.SE3(poses["L_ee"].homogeneous.copy())
    right0 = pin.SE3(poses["R_ee"].homogeneous.copy())
    mid = 0.5 * (left0.translation + right0.translation)
    to_mid_l = mid - left0.translation
    to_mid_r = mid - right0.translation
    dist_l = float(np.linalg.norm(to_mid_l))
    dist_r = float(np.linalg.norm(to_mid_r))
    if dist_l < 1e-4 or dist_r < 1e-4:
        logger.error("Hands nearly coincident; cannot squeeze.")
        return None

    # Cap so hands never cross the mid-plane.
    move_r = min(float(move_m), 0.9 * dist_r)
    move_l = min(float(move_m) + max(float(left_extra_m), 0.0), 0.9 * dist_l)
    if move_l < 1e-4 and move_r < 1e-4:
        logger.info("Requested squeeze is ~0; nothing to do.")
        return dict(mm.get_joint_pos_dict()), []

    dir_l = to_mid_l / dist_l
    dir_r = to_mid_r / dist_r
    left_tgt = left0.translation + move_l * dir_l
    right_tgt = right0.translation + move_r * dir_r

    sep0 = float(np.linalg.norm(left0.translation - right0.translation))
    sep1 = float(np.linalg.norm(left_tgt - right_tgt))
    logger.info(
        f"Squeeze inward L={move_l * 100:.1f} cm R={move_r * 100:.1f} cm "
        f"(gap {sep0:.3f} → {sep1:.3f} m):\n"
        f"  L {np.round(left0.translation, 4)} → {np.round(left_tgt, 4)}\n"
        f"  R {np.round(right0.translation, 4)} → {np.round(right_tgt, 4)}"
    )

    n = max(int(np.ceil(max(move_l, move_r) / max(step_m, 1e-4))), 2)
    initial_q = dict(mm.get_joint_pos_dict())
    configs: list[dict] = []
    mm.set_joint_pos(initial_q)

    for i in range(1, n + 1):
        a = i / n
        tl = pin.SE3(
            left0.rotation.copy(),
            (1.0 - a) * left0.translation + a * left_tgt,
        )
        tr = pin.SE3(
            right0.rotation.copy(),
            (1.0 - a) * right0.translation + a * right_tgt,
        )
        q_before = dict(mm.get_joint_pos_dict())
        if not _solve_dual_ee(mm, tl, tr):
            logger.warning(
                f"IK failed at {100.0 * i / n:.0f}% of squeeze; "
                "using collision-free prefix."
            )
            mm.set_joint_pos(q_before)
            break
        q_next = dict(mm.get_joint_pos_dict())
        jump = max(abs(q_next[k] - q_before[k]) for k in q_before)
        if jump > MAX_JOINT_JUMP:
            logger.warning(
                f"Joint jump {jump:.2f} rad at step {i}; stopping prefix."
            )
            mm.set_joint_pos(q_before)
            break
        configs.append(q_next)

    mm.set_joint_pos(initial_q)
    if not configs:
        logger.error("No squeeze progress.")
        return None
    jump = max_adjacent_jump([initial_q] + configs)
    if jump > MAX_JOINT_JUMP:
        logger.error(f"Planned jump {jump:.2f} rad exceeds safety limit.")
        return None
    logger.info(f"Planned {len(configs)}/{n} squeeze waypoints.")
    return initial_q, configs


def main(
    cm: float = 2.0,
    left_extra_cm: float = 1.0,
    duration: float = 1.5,
    control_hz: float = 100.0,
    step_m: float = 0.01,
    goto_snapshot: bool = False,
    leave_held: bool = True,
) -> None:
    """Close both hands toward their midpoint.

    Args:
        cm: How far **each** hand moves toward the mid-point (centimeters).
        left_extra_cm: Extra inward travel for the **left** hand only (cm).
            Default 1 so ``--cm 2`` moves left ~3 cm and right 2 cm.
        duration: Stream time (seconds).
        control_hz: Command rate.
        step_m: Waypoint spacing along each hand's path (meters).
        goto_snapshot: If True, first move to the recorded joint snapshot.
        leave_held: Exit without ``stop()`` so the pose stays held.
    """
    if cm <= 0 and left_extra_cm <= 0:
        logger.info(f"cm={cm}, left_extra_cm={left_extra_cm} (<=0); nothing to do.")
        return
    move_m = float(cm) / 100.0
    left_extra_m = float(left_extra_cm) / 100.0

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        if goto_snapshot:
            logger.info("Moving to recorded snapshot before squeeze...")
            rate = RateLimiter(control_hz)
            q_snap = np.array([SNAPSHOT_Q[n] for n in JOINT_NAMES], dtype=float)
            for _ in range(max(int(2.0 * control_hz), 10)):
                bot.left_arm.set_joint_pos(q_snap[:7])
                bot.right_arm.set_joint_pos(q_snap[7:14])
                bot.torso.set_joint_pos(q_snap[14:17])
                rate.sleep()

        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused during motion planning.")

        try:
            mm, torso_joints, _ = create_motion_manager(
                bot, torso_damping=100.0, lock_torso=True
            )
            planned = plan_squeeze(
                mm,
                move_m=move_m,
                step_m=step_m,
                left_extra_m=left_extra_m,
            )
        finally:
            if hb is not None and hasattr(hb, "is_paused") and hb.is_paused():
                hb.resume()
                logger.info("Heartbeat resumed.")
            elif hb is not None and hasattr(hb, "resume"):
                try:
                    hb.resume()
                except Exception:
                    pass

        if planned is None:
            do_shutdown = True
            return
        start_cfg, configs = planned
        if not configs:
            if leave_held:
                disconnect_leave_held(bot)
                do_shutdown = False
            return

        logger.info(f"Executing squeeze over {duration:.1f}s...")
        execute_trajectory(
            bot, torso_joints, start_cfg, configs, duration, control_hz
        )

        # Re-assert final setpoint (leave-held pattern).
        final_q = configs[-1]
        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        q = np.array([final_q[n] for n in names], dtype=float)
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.left_arm.set_joint_pos(q[:7])
            bot.right_arm.set_joint_pos(q[7:14])
            if torso_joints:
                bot.torso.set_joint_pos(q[14:])
            rate.sleep()

        logger.info("Squeeze completed; final pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info(
                "Client disconnected without stop() — pose held. "
                "Next teleop/control commands will take over."
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        do_shutdown = True
    except TimeoutError as exc:
        logger.error(f"Motion timed out: {exc}")
        do_shutdown = True
    finally:
        if do_shutdown:
            logger.info("Shutting down robot")
            bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
