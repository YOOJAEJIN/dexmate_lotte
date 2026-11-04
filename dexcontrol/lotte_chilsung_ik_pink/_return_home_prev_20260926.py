# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Return to the home pose: base yaws back to its start heading while arms, torso and head go to
``init_joint_pos`` -- both at the same time.

Home joints come from the simulator's robot config (``robots/dexmate.py``, the uncommented
``init_joint_pos`` dict of ``VEGA_1P_CFG_VERTICAL_GRIPPER``; ``--home-file`` to point elsewhere, a copy
is kept here as fallback). Grippers are not commanded (they stay open after ``11.open_hands.py``).

Two motions run together:

* **base**: in-place yaw to ``--home-yaw-deg`` (default 0 = the task's spawn heading; step 8 turned to
  ~+90 deg toward the conveyor). Feedback is the base yaw in the world, fitted from every fresh
  world-fixed marker the sim writes (red tip, the four card-stack edge markers and the
  purple block centre once the block is set down; ``_sim_markers.base_yaw_world``), closed loop with
  ``_base_turn.turn_in_place`` in a background thread.
* **joints**: the arms return one after the other -- ``--arm-order right,left`` (default): the right
  arm (together with torso and head) first, then the left arm, which starts when the right arm has
  covered 70 % of its motion (``--arm-overlap`` 0.3, default; 0 = after the right arm has finished,
  0.5 = at its midpoint, 1 = together). The stages are
  merged into one timeline and streamed as a single motion with ``_pink_arm.stream`` (min-jerk,
  leash-paced, per-joint speed caps). If the base turn is expected to take longer
  (``--turn-rate-deg-s``), the timeline is stretched so arms and base finish together.

Example::

    python 13.return_home.py --plan-only          # joint deltas, yaw needed, expected times; no motion
    python 13.return_home.py                      # turn back + go home together
    python 13.return_home.py --no-turn-base       # joints only
    python 13.return_home.py --arm-overlap 0                # left arm only after the right arm has finished
    python 13.return_home.py --arm-order left,right
    python 13.return_home.py --no-move-head
"""

from __future__ import annotations

import ast
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _base_turn import WHEEL_GAIN_SIM, measure_rtf, turn_in_place, wrap  # noqa: E402
from _pink_arm import COMMAND_TOPICS, COMPONENT, TORSO_MAX_VEL, detect_foreign_commands, stream  # noqa: E402
from _pink_common import ARM_JOINTS, connect, disconnect_leave_held, read_joint_pos  # noqa: E402
from _sim_markers import DEX_ROOT, TIP_FILE_NAME, base_yaw_world, find_sim_dir, world_fixed_pairs  # noqa: E402

HEAD_JOINTS = [f"head_j{i}" for i in range(1, 4)]
COMPONENT_JOINTS = {
    "left_arm": ARM_JOINTS["left"],
    "right_arm": ARM_JOINTS["right"],
    "torso": ARM_JOINTS["torso"],
    "head": HEAD_JOINTS,
}
DEFAULT_HOME_FILE = DEX_ROOT / "unitree_sim_isaaclab_260918" / "robots" / "dexmate.py"

# Copy of robots/dexmate.py VEGA_1P_CFG_VERTICAL_GRIPPER init_joint_pos (2026-09-25); used if the file
# cannot be parsed.
HOME_JOINT_POS_FALLBACK: dict[str, float] = {
    'L_arm_j1': 1.5745360851287842, 'L_arm_j2': -4.601889304467477e-05, 'L_arm_j3': -0.0001406389201292768,
    'L_arm_j4': -1.5620099306106567, 'L_arm_j5': -0.00018947907665278763, 'L_arm_j6': 0.004862525034695864,
    'L_arm_j7': 0.0001019211849779822,
    'R_arm_j1': -1.5743721723556519, 'R_arm_j2': 0.00020777805184479803, 'R_arm_j3': 0.00020511656475719064,
    'R_arm_j4': -1.5603846311569214, 'R_arm_j5': -0.00010557365021668375, 'R_arm_j6': -0.004143048543483019,
    'R_arm_j7': 3.1130890420172364e-05,
    'head_j1': -0.08560948818922043, 'head_j2': 0.000296128389891237, 'head_j3': 0.00010670589108485729,
    'torso_j1': 0.18887637555599213, 'torso_j2': 0.4230404496192932, 'torso_j3': 0.11368651688098907,
}


def load_home_joint_pos(path: Path) -> dict[str, float]:
    """Parse the home ``init_joint_pos = {...}`` dict literal from ``robots/dexmate.py``.

    The file has several ``init_joint_pos`` occurrences (a ``{".*": 0.0}`` default in ``_init_state_cfg``,
    a commented-out older pose, the live one); full-line comments are dropped and, among the remaining
    dict literals, the largest one that names arm and torso joints is used.
    """
    text = Path(path).read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    best: dict[str, float] | None = None
    for m in re.finditer(r"init_joint_pos\s*=\s*\{", code):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = ast.literal_eval(code[start:i + 1])
                    except (ValueError, SyntaxError):
                        d = None
                    if isinstance(d, dict) and "L_arm_j1" in d and "torso_j1" in d and (best is None or len(d) > len(best)):
                        best = {str(k): float(v) for k, v in d.items()}
                    break
    if best is None:
        raise ValueError(f"{path}: no init_joint_pos dict with arm and torso joints found")
    return best


def main(
    home_file: Path = DEFAULT_HOME_FILE,
    home_yaw_deg: float = 0.0,
    turn_base: bool = True,
    move_head: bool = True,
    move_torso: bool = True,
    arm_order: str = "right,left",
    arm_overlap: float = 0.3,
    duration: float = 6.0,
    max_joint_vel: float = 0.4,
    torso_max_vel: float = TORSO_MAX_VEL,
    turn_rate_deg_s: float = 5.0,
    max_age_s: float = 3.0,
    yaw_tol_deg: float = 2.0,
    kp: float = 2.0,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    chassis_hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    turn_timeout_sim: float = 20.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Yaw the base back to its home heading while arms / torso / head return to init_joint_pos.

    Args:
        home_file: robots/dexmate.py (or any file) holding the ``init_joint_pos = {...}`` home dict.
        home_yaw_deg: Base heading in the world to return to (degrees; 0 = the task spawn heading).
        turn_base: Also turn the base (False = joints only).
        move_head: Include the head joints.
        move_torso: Include the torso joints.
        arm_order: Which arm returns first: "right,left" (default) or "left,right"; torso and head move with the first.
        arm_overlap: How much of the first arm's motion the second arm overlaps (0.3 default = the left arm starts
            when the right arm is 70 % done; 0 = after it has finished; 1 = together).
        duration: Minimum joint stream time (seconds).
        max_joint_vel: Arm / head joint speed cap (rad/s).
        torso_max_vel: Torso joint speed cap (rad/s).
        turn_rate_deg_s: Expected base yaw rate in wall seconds, used only to pace the joints so both finish together.
        max_age_s: Marker files older than this are rejected (sim not running).
        yaw_tol_deg: Stop the turn when |heading error| is below this (degrees).
        kp: Base turn P gain, wz = kp * error (rad/s per rad, sim time).
        wz_max: Base yaw-rate cap (rad/s, sim time).
        wz_min: Minimum |wz| while turning (sim: keeps the wheel joint target above the drive dead zone).
        chassis_hz: Chassis command + feedback rate (wall clock).
        steer_tol_deg: Start driving once both wheels are within this of the in-place steering angles.
        steer_timeout_sim: Max sim seconds to wait for the steering.
        turn_timeout_sim: Give up on the turn after this many sim seconds of driving.
        stall_s: Raise the wheel command if the heading has not changed for this many wall seconds.
        wheel_gain: Multiplier on the wheel-speed command (sim default undoes the scale=100; real robot 1.0).
        plan_only: Connect, report joint deltas / yaw needed / expected times, move nothing.
        leave_held: Exit without ``stop()`` so the pose stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    try:
        home = load_home_joint_pos(home_file)
        logger.info(f"Home joints: {len(home)} from {home_file}")
    except (OSError, ValueError, SyntaxError) as exc:
        logger.warning(f"Could not parse {home_file} ({exc}); using the built-in copy.")
        home = dict(HOME_JOINT_POS_FALLBACK)
    comps = ["left_arm", "right_arm"] + (["torso"] if move_torso else []) + (["head"] if move_head else [])
    targets: dict[str, np.ndarray] = {}
    for c in comps:
        names = COMPONENT_JOINTS[c]
        missing = [n for n in names if n not in home]
        if missing:
            logger.warning(f"{c}: {missing} not in the home dict; component skipped.")
            continue
        targets[c] = np.array([home[n] for n in names], dtype=float)
    if not targets:
        logger.error("No component to move.")
        return False

    sim_dir = find_sim_dir()

    def yaw_error() -> tuple[float, str]:
        pairs = world_fixed_pairs(sim_dir, max_age_s)
        yaw_wb, span = base_yaw_world(pairs)
        err = wrap(np.deg2rad(home_yaw_deg) - yaw_wb)
        return err, f"base yaw(world) {np.rad2deg(yaw_wb):+.1f} deg from {len(pairs)} markers (span {span * 100:.0f} cm)"

    yaw_err = None
    if turn_base:
        try:
            yaw_err, info = yaw_error()
            logger.info(f"Base heading error {np.rad2deg(yaw_err):+.1f} deg to home {home_yaw_deg:+.0f} deg  [{info}]")
            if abs(yaw_err) < np.deg2rad(yaw_tol_deg):
                logger.info("Base already at the home heading; no turn.")
                yaw_err = None
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"Cannot measure the base heading ({exc}); base turn disabled.")
            yaw_err = None

    if not plan_only:
        probe = [c for c in targets if c in COMMAND_TOPICS]
        foreign = detect_foreign_commands(probe)
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

    bot = connect()
    do_shutdown = not leave_held
    turn_thread = None
    turn_result: dict = {}
    try:
        meas = {c: read_joint_pos(getattr(bot, c), c) for c in targets}
        path = {}
        for c in targets:
            d = targets[c] - meas[c]
            path[c] = float(np.abs(d).max())
            logger.info(f"{c}: max joint change {np.rad2deg(path[c]):.1f} deg  (now {np.round(meas[c], 3)} -> home {np.round(targets[c], 3)})")
        caps = {c: (torso_max_vel if c == "torso" else max_joint_vel) for c in targets}
        # Stages: first arm (+ torso, head), then the second arm; merged into one timeline.
        arms = [a.strip() for a in arm_order.split(",") if a.strip()]
        if sorted(arms) != ["left", "right"]:
            raise ValueError(f"--arm-order must be 'right,left' or 'left,right', got {arm_order!r}")
        if not 0.0 <= arm_overlap <= 1.0:
            raise ValueError("--arm-overlap must be between 0 and 1")
        stage_comps = [[f"{arms[0]}_arm"] + [c for c in ("torso", "head") if c in targets], [f"{arms[1]}_arm"]]
        stage_comps = [[c for c in grp if c in targets] for grp in stage_comps]
        plan: list[tuple[list[str], float, float]] = []  # (components, start_s, duration_s)
        t_prev_start, t_prev_dur = 0.0, 0.0
        for grp in stage_comps:
            grp = [c for c in grp if path[c] > 1e-6]
            if not grp:
                continue
            t_dur = max(duration, max(path[c] / max(caps[c], 1e-3) for c in grp))
            t_start = t_prev_start + (1.0 - arm_overlap) * t_prev_dur if plan else 0.0
            plan.append((grp, t_start, t_dur))
            t_prev_start, t_prev_dur = t_start, t_dur
        t_joints = max((t0 + td for _, t0, td in plan), default=0.0)
        t_turn = abs(np.rad2deg(yaw_err)) / max(turn_rate_deg_s, 0.1) if yaw_err is not None else 0.0
        scale = t_turn / t_joints if (t_joints > 0 and t_turn > t_joints) else 1.0
        plan = [(grp, t0 * scale, td * scale) for grp, t0, td in plan]
        t_stream = max(t_joints * scale, 0.5)
        logger.info(
            "Joint timeline: " + "; ".join(f"{'+'.join(grp)} {t0:.1f}-{t0 + td:.1f}s" for grp, t0, td in plan)
            + f"; total {t_stream:.1f}s (joints alone need {t_joints:.1f}s"
            + (f", base turn of {np.rad2deg(yaw_err):+.1f} deg expected ~{t_turn:.0f}s at {turn_rate_deg_s:.0f} deg/s"
               + (", timeline stretched to match" if scale > 1.0 else "") if yaw_err is not None else "")
            + ")"
        )
        n_knots = max(int(np.ceil(t_stream * 25.0)), 2)
        times = np.linspace(0.0, t_stream, n_knots + 1)
        groups = {c: np.tile(meas[c], (n_knots + 1, 1)) for c in targets}
        for grp, t0, td in plan:
            frac = np.clip((times - t0) / max(td, 1e-6), 0.0, 1.0)[:, None]
            for c in grp:
                groups[c] = meas[c] + (targets[c] - meas[c]) * frac
        if plan_only:
            logger.info("Plan only: nothing commanded.")
            return True

        if yaw_err is not None:
            rtf = measure_rtf(sim_dir / TIP_FILE_NAME)

            def _turn():
                try:
                    turn_result["res"] = turn_in_place(
                        bot.chassis, yaw_error, tol_deg=yaw_tol_deg, kp=kp, wz_max=wz_max, wz_min=wz_min,
                        hz=chassis_hz, steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim,
                        timeout_sim=turn_timeout_sim, stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                    )
                except BaseException as exc:  # noqa: BLE001 - report and stop the chassis
                    turn_result["exc"] = exc
                    try:
                        bot.chassis.stop()
                    except Exception:
                        pass

            turn_thread = threading.Thread(target=_turn, name="base-turn", daemon=True)
            turn_thread.start()
            logger.info("Base turn started in the background; streaming the joints now.")

        stream(bot, groups, t_stream, config.CONTROL_DT)
        after = {c: read_joint_pos(getattr(bot, c), c) for c in targets}
        res = {c: float(np.rad2deg(np.abs(targets[c] - after[c]).max())) for c in targets}
        logger.info("Joints at home; residual (deg): " + ", ".join(f"{c} {r:.1f}" for c, r in res.items()))

        if turn_thread is not None:
            while turn_thread.is_alive():
                turn_thread.join(timeout=0.5)
            if "exc" in turn_result:
                logger.error(f"Base turn failed: {turn_result['exc']!r}")
            else:
                err, info = turn_result.get("res", (None, ""))
                if err is not None:
                    logger.info(f"Base turn finished; heading residual {np.rad2deg(err):+.1f} deg  [{info}]")
        logger.info("Return home done; pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return True
    except KeyboardInterrupt:
        logger.warning("Interrupted -- stopping the chassis, leaving the last commanded joints held.")
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
