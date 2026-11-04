# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Pull both hands straight back from the block with their orientation held: arms first, torso only if needed.

Step 12 of the Chilsung sequence (2026-09-27): after ``10.approach_purple_to_red_bimanual.py`` placed the block and
``11.open_hands.py`` opened the grippers, both grippers move ``--back-cm`` (15 cm) along robot -X with their
orientation held, so the open fingers slide straight out of the block edge instead of pivoting around it (an arm j1
swing did that and clipped the block).

Planning (Pink IK, robot base frame, ``_j1_up.plan_translation``): first with the two arms only, torso locked. If
that path falls short by more than ``--torso-tol-cm`` (joint limits: from the placing pose the right arm sits on its
``R_arm_j2`` / ``j4`` limits and cannot retract on its own, 2026-09-27), the move is re-planned with the torso freed as
well -- the torso posture cost keeps it to the few degrees the arms cannot supply -- and the plan that reaches farther
is executed. Whatever is still unreachable is dropped: the reachable prefix is streamed and the script stops there
("stop where it can"), logging what was reached. ``--no-allow-torso`` = arms only, even if short. The commanded
components are streamed together with ``_pink_arm.stream`` (arm cap ``--max-joint-vel``, torso cap ``--torso-vel``),
anchored at the sim's held joint targets so nothing is first dropped to its sagged measured angle.

Example::

    python 12.hands_back.py --plan-only        # arms-only vs torso-assisted reach, move nothing
    python 12.hands_back.py                    # back 15 cm (or as far as possible), >= 4 s
    python 12.hands_back.py --back-cm 10
    python 12.hands_back.py --no-allow-torso   # arms only
    python 12.hands_back.py --duration 6       # slower
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _j1_up import plan_translation  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    COMPONENT,
    TORSO_MAX_VEL,
    anchor_offsets,
    detect_foreign_commands,
    read_setpoints,
    rot_err,
    stream,
)
from _pink_common import EE, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402

SIDES = ["left", "right"]


def main(
    back_cm: float = 15.0,
    step_m: float = 0.01,
    duration: float = 4.0,
    max_joint_vel: float = 0.3,
    torso_vel: float = TORSO_MAX_VEL,
    allow_torso: bool = True,
    torso_tol_cm: float = 0.5,
    anchor_setpoints: bool = True,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Move both grippers straight back along robot -X with orientation held; arms only if they can, else + torso.

    Args:
        back_cm: How far back the hands should go (centimetres); less if even the torso-assisted plan runs out of room.
        step_m: Cartesian spacing of the IK waypoints (metres).
        duration: Minimum stream time (seconds); stretched if a joint would exceed its speed cap.
        max_joint_vel: Arm joint speed cap (rad/s).
        torso_vel: Torso joint speed cap (rad/s).
        allow_torso: Re-plan with the torso freed when the arms alone fall short (default). False = arms only.
        torso_tol_cm: The arms-only plan counts as good enough if it is short by at most this (centimetres).
        anchor_setpoints: Stream relative to the sim's held joint targets (recommended; avoids the sag drop).
        plan_only: Connect, plan from the measured state, report the reachable distance, move nothing.
        leave_held: Exit without ``stop()`` so the pose stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming arm / torso targets.
    """
    if back_cm <= 0.0:
        raise ValueError("--back-cm must be > 0")
    back_m = back_cm / 100.0
    offset = np.array([-back_m, 0.0, 0.0])
    comps = [COMPONENT[s] for s in SIDES] + (["torso"] if allow_torso else [])
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
        p0 = {s: configuration.get_transform_frame_to_world(EE[s]).translation.copy() for s in SIDES}
        R0 = {s: configuration.get_transform_frame_to_world(EE[s]).rotation.copy() for s in SIDES}
        logger.info("Pull-back (robot base): " + ", ".join(f"{s} gripper {np.round(p0[s], 4)}" for s in SIDES)
                    + f"; goal = {back_cm:.1f} cm along -X with orientation held")

        def try_plan(use_torso: bool):
            """(plan, reach_m, drift_deg, torso_change_deg) for arms-only or arms + torso; plan None if no progress."""
            res = plan_translation(configuration, solver, SIDES, offset, step_m, use_torso)
            if res is None:
                return None, 0.0, 0.0, None
            rcfg, q_s, knots, ridx = res
            rcfg.update(knots[-1])
            reach = float(-max(rcfg.get_transform_frame_to_world(EE[s]).translation[0] - p0[s][0] for s in SIDES))
            drift = max(np.rad2deg(rot_err(rcfg.get_transform_frame_to_world(EE[s]).rotation, R0[s])) for s in SIDES)
            d_torso = np.round(np.rad2deg(knots[-1][ridx["torso"]] - q_s[ridx["torso"]]), 1) if use_torso else None
            rcfg.update(q_s)
            return res, reach, float(drift), d_torso

        plan, reach, drift, d_torso = try_plan(False)
        mode = "arms only"
        logger.info(f"Arms only: {'no progress' if plan is None else f'{len(plan[2])} steps'}, hands go back {reach * 100:.1f} of {back_cm:.1f} cm"
                    + (f" (orientation drift {drift:.1f} deg)" if plan is not None else ""))
        if allow_torso and reach < back_m - torso_tol_cm / 100.0:
            plan_t, reach_t, drift_t, d_torso_t = try_plan(True)
            logger.info(f"Arms + torso: {'no progress' if plan_t is None else f'{len(plan_t[2])} steps'}, hands go back {reach_t * 100:.1f} of {back_cm:.1f} cm"
                        + (f" (orientation drift {drift_t:.1f} deg, torso change (j1, j2, j3) {d_torso_t} deg)" if plan_t is not None else ""))
            if plan_t is not None and reach_t > reach + 0.002:
                plan, reach, drift, d_torso, mode = plan_t, reach_t, drift_t, d_torso_t, "arms + torso"
        if plan is None:
            logger.error("Neither the arms nor the torso can move the hands back from here (joint limits / reach). Nothing commanded.")
            return False
        rcfg, q_start, knots, ridx = plan
        full = reach >= back_m - 0.005
        (logger.info if full else logger.warning)(
            f"[{'REACHABLE' if full else 'PARTIAL'}] {mode}: hands go back {reach * 100:.1f} of {back_cm:.1f} cm in {len(knots)} steps"
            + ("" if full else " -- out of room there; stopping at that point")
            + f"; orientation drift {drift:.1f} deg" + (f"; torso change (j1, j2, j3) {d_torso} deg" if d_torso is not None else "")
        )
        if plan_only:
            logger.info("Plan only: nothing commanded.")
            return full
        used = [COMPONENT[g] for g in ridx]
        off = anchor_offsets(configuration, idx, setpoints, used)
        Q = np.array([q_start] + knots)
        path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
        t_stream = max(duration, max(path[g] / (torso_vel if g == "torso" else max(max_joint_vel, 1e-3)) for g in path))
        if t_stream > duration:
            logger.info(f"Stream time stretched to {t_stream:.1f}s by the joint speed caps (arms {max_joint_vel:.2f}, torso {torso_vel:.2f} rad/s)")
        groups = {COMPONENT[g]: Q[:, ii] + off[COMPONENT[g]] for g, ii in ridx.items()}
        logger.info(f"Streaming {' + '.join(groups)} over {t_stream:.1f}s...")
        stream(bot, groups, t_stream, config.CONTROL_DT)
        time.sleep(0.3)
        sync_from_robot(bot, configuration, idx)
        moved = {s: configuration.get_transform_frame_to_world(EE[s]).translation - p0[s] for s in SIDES}
        logger.info(
            "Done (measured): " + ", ".join(f"{s} gripper moved {np.round(moved[s] * 100, 1)} cm" for s in SIDES)
            + (f"; torso now {np.round(np.rad2deg(configuration.q[idx['torso']]), 1)} deg" if "torso" in ridx else "")
            + ". Pose left held."
        )
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return full
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
