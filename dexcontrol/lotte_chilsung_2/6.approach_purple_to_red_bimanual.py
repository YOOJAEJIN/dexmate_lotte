# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Bimanual: carry the purple small-box center onto the red conveyor tip.

Pipeline (lotte_chilsung), box already held with **both** hands::

    python squeeze_inward.py --cm 2
    (close both hands via omniteleop)
    python box_lift_up.py
    python rotate_vertical.py
    python approach_purple_to_red_bimanual.py --dry-run   # feasibility only
    python approach_purple_to_red_bimanual.py             # execute

Keeps the rigid grasp (relative L/R EE poses fixed, orientation fixed) and
translates both hands by

    delta = red_tip − purple_center        (robot base frame)

so the purple marker lands on the red marker. Because the small box is
rigidly held, the purple center moves exactly with the EE pair, so this is
``approach_tip.py`` with the tracked point replaced by the purple center.

Sim must be writing (robot base frame):

  - ``unitree_sim_isaaclab/card_box_tip_pos.txt``          (red)
  - ``unitree_sim_isaaclab/card_box_mass_center_pos.txt``   (purple)

Options::

    --dry-run              plan + report residual, do not move
    --scale 0.5            close only half the gap
    --stop-short 0.06      stop 6 cm before red along the approach line
    --offset 0 0.05 0      add robot-base XYZ offset to the red target
    --lock-torso           arms only (default: torso assists, damping 1)

Never calls ``stop()`` on exit (the box is held), unless ``--no-leave-held``.
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
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    connect_bot,
    create_motion_manager,
    disconnect_leave_held,
)
from approach_purple_to_red import (  # noqa: E402
    _DEFAULT_PURPLE_CANDIDATES,
    _DEFAULT_TIP_CANDIDATES,
    _load_xyz,
    _resolve_existing,
)
from approach_tip import (  # noqa: E402
    TranslateMidToTipPlanner,
    execute_until_collision,
    green_ee_mid,
    plan_translate_to_tip,
)
from dexmotion.motion_manager import MotionManager


class PurpleToRedPlanner(TranslateMidToTipPlanner):
    """Rigid translate so the purple box center approaches the red tip.

    Anchor point is the plain L/R EE midpoint (``z_offset=0``). Under a rigid
    grasp the purple center is a fixed offset from that anchor, so steering the
    anchor to ``red + (anchor0 − purple0)`` is the same as steering purple to
    ``red``. Residual reporting then works unchanged.
    """

    def __init__(self, mm: MotionManager, purple: np.ndarray, red: np.ndarray):
        poses = mm.fk(["L_ee", "R_ee"])
        anchor0 = 0.5 * (poses["L_ee"].translation + poses["R_ee"].translation)
        self.purple0 = np.asarray(purple, dtype=float).reshape(3)
        self.red = np.asarray(red, dtype=float).reshape(3)
        self.purple_from_anchor = self.purple0 - anchor0
        super().__init__(mm, tip=self.red - self.purple_from_anchor, z_offset=0.0)

    def purple_now(self) -> np.ndarray:
        """Predicted purple center for the MotionManager's current joints."""
        return green_ee_mid(self.mm, 0.0) + self.purple_from_anchor

    def purple_after(self, configs: list[dict]) -> np.ndarray:
        q_prev = dict(self.mm.get_joint_pos_dict())
        self.mm.set_joint_pos(configs[-1] if configs else self.initial_q)
        purple = self.purple_now()
        self.mm.set_joint_pos(q_prev)
        return purple


def _final_hold(bot, torso_joints: list[str], q_final: dict, control_hz: float) -> None:
    names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
    q = np.array([q_final[n] for n in names], dtype=float)
    rate = RateLimiter(control_hz)
    for _ in range(max(int(0.2 * control_hz), 5)):
        bot.left_arm.set_joint_pos(q[:7])
        bot.right_arm.set_joint_pos(q[7:14])
        if torso_joints:
            bot.torso.set_joint_pos(q[14:])
        rate.sleep()


def main(
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stop_short: float = 0.0,
    scale: float = 1.0,
    step_m: float = 0.01,
    duration: float = 6.0,
    control_hz: float = 100.0,
    torso_damping: float = 1.0,
    lock_torso: bool = False,
    dry_run: bool = False,
    leave_held: bool = True,
) -> None:
    """Translate the bimanual grasp so purple (box center) reaches red (tip).

    Args:
        tip_file: Red tip XYZ file (robot base). Default: sim tip file.
        purple_file: Purple center XYZ file (robot base). Default: sim file.
        offset: Robot-base XYZ added to the red target (meters).
        stop_short: Stop this far before the target along the approach line.
        scale: Fraction of the remaining delta to apply (1.0 = all the way).
        step_m: Waypoint spacing along candidate paths (meters).
        duration: Full-path stream time; shortened if only a prefix is safe.
        control_hz: Command rate.
        torso_damping: Torso IK damping when unlocked (low = torso assists).
        lock_torso: Freeze torso; arms-only IK.
        dry_run: Plan and report the reachable residual, then exit. No motion.
        leave_held: Exit without ``stop()`` so the grasp stays held.
    """
    tip_path = _resolve_existing(tip_file, _DEFAULT_TIP_CANDIDATES, "red tip")
    purple_path = _resolve_existing(
        purple_file, _DEFAULT_PURPLE_CANDIDATES, "purple center"
    )
    red = _load_xyz(tip_path, "red tip")
    purple = _load_xyz(purple_path, "purple center")

    target = red + np.asarray(offset, dtype=float)
    full = target - purple
    full_dist = float(np.linalg.norm(full))
    if full_dist < 1e-4:
        logger.info("Purple already on red (within 0.1 mm); nothing to do.")
        return
    if stop_short > 0.0:
        target = target - min(stop_short, 0.95 * full_dist) * full / full_dist
    target = purple + float(scale) * (target - purple)
    delta = target - purple
    logger.info(
        "Purple → red (bimanual rigid translate, robot base):\n"
        f"  purple  {np.round(purple, 4)}\n"
        f"  red     {np.round(red, 4)}"
        + (f"  +offset {np.round(offset, 3)}" if np.any(np.asarray(offset)) else "")
        + (f"  stop_short {stop_short:.3f}" if stop_short > 0 else "")
        + (f"  scale {scale:g}" if scale != 1.0 else "")
        + f"\n  target  {np.round(target, 4)}\n"
        f"  delta   {np.round(delta, 4)}  |d|={np.linalg.norm(delta):.4f} m"
    )

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused for IK planning.")

        try:
            mm, torso_joints, _ = create_motion_manager(
                bot, torso_damping=torso_damping, lock_torso=lock_torso
            )
            logger.info(
                "Torso locked (arms only)."
                if lock_torso
                else f"Torso unlocked, damping={torso_damping:g}."
            )
            planner = PurpleToRedPlanner(mm, purple=purple, red=target)
            planned = plan_translate_to_tip(planner, step_m=step_m)
        finally:
            if hb is not None and hasattr(hb, "resume"):
                try:
                    hb.resume()
                    logger.info("Heartbeat resumed.")
                except Exception:
                    pass

        if planned is None:
            logger.error("Planning failed: no collision-free progress. Not moving.")
            return

        start_cfg, configs = planned
        predicted = planner.purple_after(configs)
        residual = float(np.linalg.norm(target - predicted))
        covered = 100.0 * max(0.0, 1.0 - residual / max(np.linalg.norm(delta), 1e-6))
        verdict = "REACHABLE" if residual <= 0.01 else "PARTIAL"
        logger.info(
            f"[{verdict}] {len(configs)} configs; predicted purple after motion "
            f"{np.round(predicted, 4)}, residual to target {residual:.4f} m "
            f"({covered:.0f}% of gap)."
        )

        if dry_run:
            logger.info("Dry run: not moving. Pose left held.")
            return
        if not configs:
            logger.info("Already at target; nothing to execute.")
            return

        n_safe = execute_until_collision(
            bot, mm, torso_joints, start_cfg, configs,
            duration=duration, control_hz=control_hz,
        )
        if n_safe == 0:
            logger.error("No safe configs executed. Pose left held.")
            return

        q_final = configs[n_safe - 1]
        _final_hold(bot, torso_joints, q_final, control_hz)
        mm.set_joint_pos(q_final)
        purple_est = planner.purple_now()
        logger.info(
            f"Done ({n_safe}/{len(configs)} configs). purple≈{np.round(purple_est, 4)} "
            f"target={np.round(target, 4)} err={np.linalg.norm(purple_est - target):.4f} m. "
            f"Check {purple_path.name} for the sim-measured value."
        )
    except FileNotFoundError as exc:
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


if __name__ == "__main__":
    tyro.cli(main)
