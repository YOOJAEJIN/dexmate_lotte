# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Lower the right end-effector in Cartesian space, keeping orientation fixed.

Uses MotionManager ``right_arm.move_ee_xyz`` so only the EE position changes
(default: 5 cm down along robot-base −Z). **Only the right arm is commanded**;
left / torso / head are left untouched so they keep their last held setpoints.

Example::

    python lower_right_ee.py
    python lower_right_ee.py --cm 5
    python lower_right_ee.py --cm 15 --duration 4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tyro
from dexcomm import RateLimiter
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import disconnect_leave_held, make_fast_mm, connect_bot  # noqa: E402
from dexcontrol.robot import Robot

# Snapshot the user recorded (optional --goto-snapshot before the Cartesian move).
SNAPSHOT_Q: dict[str, float] = {
    "L_arm_j1": -0.033005088567733765,
    "L_arm_j2": -0.0044876267202198505,
    "L_arm_j3": 0.6473899483680725,
    "L_arm_j4": -1.3070433139801025,
    "L_arm_j5": 1.164272665977478,
    "L_arm_j6": 0.4074169397354126,
    "L_arm_j7": -1.208831548690796,
    "R_arm_j1": -0.5425229668617249,
    "R_arm_j2": 0.004025580361485481,
    "R_arm_j3": -1.0238919258117676,
    "R_arm_j4": -1.3391488790512085,
    "R_arm_j5": 1.4377226829528809,
    "R_arm_j6": 1.1852997541427612,
    "R_arm_j7": -0.8522598147392273,
    "head_j1": -0.14062345027923584,
    "head_j2": 0.00011097480455646291,
    "head_j3": -0.0001715072285151109,
    "torso_j1": 0.6745240092277527,
    "torso_j2": 1.1522504091262817,
    "torso_j3": 0.3366924822330475,
}

LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]


def _normalize_traj_knots(qs_sample) -> np.ndarray:
    """``move_ee_xyz`` may return a list or an (N, D) ndarray of joint vectors."""
    if qs_sample is None:
        return np.zeros((0, 0), dtype=float)
    arr = np.asarray(qs_sample, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected trajectory shape {arr.shape}")
    return arr


def _right_from_knot(arr: np.ndarray) -> np.ndarray:
    """Extract 7 right-arm joints from a full or right-only knot."""
    flat = np.asarray(arr, dtype=float).reshape(-1)
    if flat.size >= 14:
        return flat[7:14].copy()
    if flat.size == 7:
        return flat.copy()
    raise ValueError(f"Unexpected trajectory knot size {flat.size}")


def _command_right_only(bot: Robot, right: np.ndarray) -> None:
    """Command **only** the right arm — do not touch left / torso / head.

    Re-sending left targets every tick can twitch the left arm (and with a
    shared gripped object, fighting the right-arm motion). Leaving left alone
    keeps its last held setpoint.
    """
    bot.right_arm.set_joint_pos(np.asarray(right, dtype=float).reshape(7))


def _stream_right_only(
    bot: Robot,
    traj: np.ndarray,
    duration: float,
    control_hz: float,
) -> np.ndarray:
    """Cosine-blend **right** arm only; left is never commanded."""
    n = traj.shape[0]
    n_seg = max(n - 1, 1)
    rate = RateLimiter(control_hz)
    n_ticks = max(int(duration * control_hz), 1)
    right = _right_from_knot(traj[-1])
    for t in range(1, n_ticks + 1):
        s = 0.5 * (1.0 - np.cos(np.pi * t / n_ticks)) * n_seg
        idx = min(int(s), max(n_seg - 1, 0))
        frac = s - idx
        if n == 1:
            right = _right_from_knot(traj[0])
        else:
            r0 = _right_from_knot(traj[idx])
            r1 = _right_from_knot(traj[min(idx + 1, n - 1)])
            right = (1.0 - frac) * r0 + frac * r1
        _command_right_only(bot, right)
        rate.sleep()
    return right


def _hold_final(
    bot: Robot,
    right: np.ndarray,
    control_hz: float,
    hold_s: float,
) -> None:
    """Re-assert final **right** setpoint only before disconnect."""
    rate = RateLimiter(control_hz)
    for _ in range(max(int(hold_s * control_hz), 5)):
        _command_right_only(bot, right)
        rate.sleep()


def _traj_left_drift(traj: np.ndarray, hold_q: dict[str, float]) -> float:
    """Max |Δq| of left joints in the planned traj vs hold (0 if traj is right-only)."""
    if traj.ndim != 2 or traj.shape[1] < 14:
        return 0.0
    left0 = np.array([hold_q[n] for n in LEFT_ARM_JOINTS], dtype=float)
    return float(np.max(np.abs(traj[:, :7] - left0)))


def main(
    cm: float = 5.0,
    duration: float = 4.0,
    control_hz: float = 100.0,
    hold_s: float = 0.3,
    goto_snapshot: bool = False,
    leave_held: bool = True,
) -> None:
    """Lower right EE along base −Z while keeping orientation.

    Args:
        cm: How many centimeters to lower (positive = down). Default 5.
        duration: Stream time for the planned trajectory (seconds).
        control_hz: Command rate.
        hold_s: Seconds to re-assert the final joints before disconnect.
        goto_snapshot: If True, first move joints to the recorded snapshot.
        leave_held: If True (default), exit without calling ``stop()`` so the
            last joint targets stay and the arms keep holding, but still close
            the Zenoh client so the next teleop/control session can take over.
            Use ``--no-leave-held`` for a full shutdown (arms go limp).
    """
    if cm <= 0:
        logger.info(f"cm={cm} (<=0); nothing to lower.")
        return
    dz = -float(cm) / 100.0  # meters, negative = down in base Z

    # auto_shutdown=False: avoid signal/__del__ path calling stop()+shutdown()
    # which would release the final pose after we exit.
    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    final_right: np.ndarray | None = None
    hold_q: dict[str, float] | None = None
    try:
        if goto_snapshot:
            logger.info("Moving to recorded snapshot before Cartesian lower...")
            rate = RateLimiter(control_hz)
            q_snap = np.array(
                [
                    SNAPSHOT_Q[n]
                    for n in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + TORSO_JOINTS
                ],
                dtype=float,
            )
            for _ in range(max(int(2.0 * control_hz), 10)):
                bot.left_arm.set_joint_pos(q_snap[:7])
                bot.right_arm.set_joint_pos(q_snap[7:14])
                bot.torso.set_joint_pos(q_snap[14:17])
                rate.sleep()

        components = ["left_arm", "right_arm", "head"]
        if bot.has_component("torso"):
            components.append("torso")
        hold_q = bot.get_joint_pos_dict(component=components)
        left0 = np.array([hold_q[n] for n in LEFT_ARM_JOINTS], dtype=float)
        logger.info(
            "Only right EE will be commanded; left arm / torso / head are not touched."
        )

        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused during motion planning.")

        traj = np.zeros((0, 0), dtype=float)
        r0_pos = None
        try:
            # need_traj_gen: move_ee_xyz uses the trajectory generator.
            # init_global_ik off via make_fast_mm (biggest cold-start win).
            mm = make_fast_mm(
                hold_q,
                [
                    "HEAD",
                    "BASE",
                    "LEFT_ARM",
                    "LEFT_HAND",
                    "RIGHT_HAND",
                    "TORSO",
                ],
                need_traj_gen=True,
            )

            r0 = mm.fk(["R_ee"])["R_ee"]
            r0_pos = np.asarray(r0.translation, dtype=float).copy()
            logger.info(
                f"Right EE before: pos={np.round(r0_pos, 4)}, "
                f"lowering {cm:g} cm (dz={dz:+.4f} m, orientation fixed)"
            )

            qs_sample = mm.right_arm.move_ee_xyz(np.array([0.0, 0.0, float(dz)]))
            traj = _normalize_traj_knots(qs_sample)
            if traj.shape[0] == 0:
                logger.error("move_ee_xyz returned no trajectory. Robot was not moved.")
                do_shutdown = True
                return
            left_drift = _traj_left_drift(traj, hold_q)
            if left_drift > 1e-3:
                logger.warning(
                    f"Planner traj still changes left joints by up to {left_drift:.3f} rad; "
                    "those left knots are IGNORED — only right arm is streamed."
                )
            logger.info(f"Trajectory shape={traj.shape} (right-only stream)")
        finally:
            if hb is not None and hasattr(hb, "is_paused") and hb.is_paused():
                hb.resume()
                logger.info("Heartbeat resumed.")
            elif hb is not None and hasattr(hb, "resume"):
                try:
                    hb.resume()
                except Exception:
                    pass

        logger.info(
            f"Planned {traj.shape[0]} knots; streaming right arm only over {duration:.1f}s"
        )
        final_right = _stream_right_only(
            bot, traj, duration=duration, control_hz=control_hz
        )
        _hold_final(bot, final_right, control_hz, hold_s)

        # Report whether left actually drifted during the move (physics / coupling).
        left1 = bot.left_arm.get_joint_pos()
        left_err = float(np.max(np.abs(np.asarray(left1, dtype=float) - left0)))
        logger.info(f"Left arm measured drift during move: {left_err:.4f} rad (max |Δq|)")

        q_end = dict(hold_q)
        for n, v in zip(RIGHT_ARM_JOINTS, final_right):
            q_end[n] = float(v)
        mm.set_joint_pos(q_end)
        r1 = mm.fk(["R_ee"])["R_ee"]
        logger.info(
            f"Right EE after:  pos={np.round(r1.translation, 4)}, "
            f"delta={np.round(r1.translation - r0_pos, 4)}"
        )
        del mm

        logger.info("Lower completed; final pose left held.")
        if leave_held:
            _hold_final(bot, final_right, control_hz, 0.2)
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info(
                "Client disconnected without stop() — pose held. "
                "Next teleop/control commands will take over."
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted — trying to leave last pose held.")
        if leave_held and final_right is not None:
            try:
                _hold_final(bot, final_right, control_hz, 0.2)
                disconnect_leave_held(bot)
                do_shutdown = False
            except Exception:
                do_shutdown = True
        else:
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
