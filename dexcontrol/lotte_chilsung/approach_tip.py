# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Translate both arms so the green EE-mid marker coincides with the red box tip.

Matches the Isaac Lab debug markers in ``unitree_sim_isaaclab``:

  - green: midpoint of L/R EE with each pose pushed along that EE's local +Z
    (computed here from FK; the sim's green marker was removed)
  - red:   live tip of ``card_box_conveyor``, written to
    ``unitree_sim_isaaclab/card_box_tip_pos.txt`` in **robot base frame**
    (same frame as MotionManager FK)

Planning keeps the current rigid grasp (relative L/R EE poses fixed) and
moves both hands so the green mid approaches the tip. If the tip is not fully
reachable (IK / self-collision / joint jump), the planner searches for the
**closest collision-free approach** along candidate paths (straight line and
axis-ordered detours) and executes that best-effort motion.

Example (sim running with tip file updates)::

    python approach_tip.py
    python approach_tip.py --ee-mid-z-offset 0.05
    python approach_tip.py --target 0.1 1.06 0.90   # robot-base XYZ
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import tyro
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
    RigidGraspPlanner,
    SOFT_POS_TOL,
    SOFT_ROT_TOL,
    create_motion_manager,
    disconnect_leave_held,
    execute_trajectory,
    max_adjacent_jump,
    connect_bot,
)
from dexcontrol.robot import Robot
from dexmotion.motion_manager import MotionManager

# Stop refining when remaining tip error is below this (meters).
_CLOSE_ENOUGH_M = 0.01
# Binary-search resolution along a path (meters of mid travel).
_PROGRESS_RES_M = 0.005

# Default tip file written by unitree_sim_isaaclab/sim_main.py (robot base frame).
_DEFAULT_TIP_FILE = (
    Path(__file__).resolve().parents[2]  # <dex>/  (this file lives at <dex>/dexcontrol/lotte_chilsung/)
    / "unitree_sim_isaaclab"
    / "card_box_tip_pos.txt"
)
# Nominal tip in robot base for Chilsung spawn (robot≈(-0.21,-1.21,0.044), tip≈(-0.3,-0.15,0.94)).
_DEFAULT_TARGET = (-0.09, 1.06, 0.90)


def green_ee_mid(mm: MotionManager, z_offset: float) -> np.ndarray:
    """EE-mid in robot base frame (matches sim green marker after base transform)."""
    poses = mm.fk(["L_ee", "R_ee"])
    left = poses["L_ee"]
    right = poses["R_ee"]
    l_pos = left.translation + z_offset * left.rotation[:, 2]
    r_pos = right.translation + z_offset * right.rotation[:, 2]
    return 0.5 * (l_pos + r_pos)


def load_tip_target(
    tip_file: Path | None,
    target: tuple[float, float, float] | None,
) -> np.ndarray:
    """Load tip XYZ in **robot base frame**."""
    if target is not None:
        tip = np.asarray(target, dtype=float)
        logger.info(f"Using CLI tip target (robot base): {np.round(tip, 4)}")
        return tip
    path = tip_file if tip_file is not None else _DEFAULT_TIP_FILE
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) >= 3:
                tip = np.array([float(vals[0]), float(vals[1]), float(vals[2])])
                logger.info(f"Loaded tip (robot base) from {path}: {np.round(tip, 4)}")
                return tip
        logger.warning(f"Tip file {path} malformed; using default {_DEFAULT_TARGET}")
    else:
        logger.warning(f"Tip file not found ({path}); using default {_DEFAULT_TARGET}")
    return np.asarray(_DEFAULT_TARGET, dtype=float)


class TranslateMidToTipPlanner(RigidGraspPlanner):
    """Translate the rigid grasp so green mid approaches tip (best-effort)."""

    def __init__(self, mm: MotionManager, tip: np.ndarray, z_offset: float):
        super().__init__(mm)
        self.z_offset = float(z_offset)
        self.tip = np.asarray(tip, dtype=float).reshape(3)
        self.mid0 = green_ee_mid(mm, self.z_offset)
        self.delta = self.tip - self.mid0
        self.rot_axis = np.array([1.0, 0.0, 0.0])
        self.full_angle = 0.0
        logger.info(
            f"Approach tip (3D, robot base):\n"
            f"  green_mid {np.round(self.mid0, 4)}\n"
            f"  tip       {np.round(self.tip, 4)}\n"
            f"  delta     {np.round(self.delta, 4)}  "
            f"|d|={np.linalg.norm(self.delta):.4f} m  "
            f"(dx={self.delta[0]:+.3f}, dy={self.delta[1]:+.3f}, dz={self.delta[2]:+.3f})\n"
            f"  z_offset  {self.z_offset:.3f}"
        )

    def center_for_mid(self, mid: np.ndarray) -> np.ndarray:
        """Box-frame center that places green mid at ``mid`` (rigid translate)."""
        return self.center0 + (np.asarray(mid, dtype=float) - self.mid0)


def _path_length(mids: list[np.ndarray]) -> float:
    if len(mids) < 2:
        return 0.0
    return float(
        sum(np.linalg.norm(mids[i] - mids[i - 1]) for i in range(1, len(mids)))
    )


def _candidate_mid_paths(mid0: np.ndarray, tip: np.ndarray) -> list[tuple[str, list[np.ndarray]]]:
    """Straight line plus axis-ordered detours (often reach farther)."""
    mx, my, mz = mid0
    tx, ty, tz = tip
    return [
        ("straight", [mid0, tip]),
        ("z_then_xy", [mid0, np.array([mx, my, tz]), tip]),
        ("xy_then_z", [mid0, np.array([tx, ty, mz]), tip]),
        ("x_y_z", [mid0, np.array([tx, my, mz]), np.array([tx, ty, mz]), tip]),
        ("z_y_x", [mid0, np.array([mx, my, tz]), np.array([mx, ty, tz]), tip]),
    ]


def _densify_mids(mids: list[np.ndarray], step_m: float) -> list[np.ndarray]:
    """Insert points so consecutive mids are at most ``step_m`` apart."""
    out = [np.asarray(mids[0], dtype=float)]
    for nxt in mids[1:]:
        nxt = np.asarray(nxt, dtype=float)
        cur = out[-1]
        dist = float(np.linalg.norm(nxt - cur))
        if dist < 1e-9:
            continue
        n = max(int(np.ceil(dist / max(step_m, 1e-4))), 1)
        for i in range(1, n + 1):
            out.append((1.0 - i / n) * cur + (i / n) * nxt)
    return out


def _forward_track_mids(
    planner: TranslateMidToTipPlanner,
    mids: list[np.ndarray],
) -> list[dict] | None:
    """Track dense mid waypoints from ``initial_q`` with local IK; creep on failure."""
    planner.mm.set_joint_pos(planner.initial_q)
    wps = [planner.box_pose(0.0, planner.center_for_mid(m)) for m in mids]
    configs: list[dict] = []
    for i in range(1, len(wps)):
        q_before = dict(planner.mm.get_joint_pos_dict())
        if _solve_local_only(planner, wps[i]):
            q_next = dict(planner.mm.get_joint_pos_dict())
            jump = max(abs(q_next[n] - q_before[n]) for n in q_before)
            if jump <= MAX_JOINT_JUMP:
                configs.append(q_next)
                continue
            # Large jump — restore and creep with smaller steps.
            planner.mm.set_joint_pos(q_before)
        else:
            planner.mm.set_joint_pos(q_before)

        extra, _t = _track_segment_max(
            planner, wps[i - 1], wps[i], min_step_m=_PROGRESS_RES_M
        )
        if extra:
            configs.extend(extra)
        break
    if not configs:
        return None
    return configs


def _solve_local_only(planner: RigidGraspPlanner, box: pin.SE3, max_reps: int = IK_MAX_REPS) -> bool:
    """Local IK only (no global reseed) — fast probe for best-effort creep."""
    tl, tr = planner.ee_targets(box)
    q_prev = dict(planner.mm.get_joint_pos_dict())
    planner._reanchor_posture_task()
    perr, rerr = 1.0, 90.0
    for _ in range(max_reps):
        sol, coll, lim = planner.mm.ik(
            target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
            target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
            max_iterations=IK_MAX_ITERS,
        )
        if sol is None or coll or not lim:
            planner.mm.set_joint_pos(q_prev)
            return False
        planner.mm.set_joint_pos(sol)
        perr, rerr = planner._tracking_error(tl, tr)
        if perr < POS_TOL and rerr < ROT_TOL:
            return True
    if perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL:
        return True
    planner.mm.set_joint_pos(q_prev)
    return False


def _track_segment_max(
    planner: RigidGraspPlanner,
    box_prev: pin.SE3,
    box_next: pin.SE3,
    min_step_m: float,
) -> tuple[list[dict], float]:
    """Advance from ``box_prev`` toward ``box_next`` as far as local IK allows."""
    span = float(np.linalg.norm(box_next.translation - box_prev.translation))
    if span < min_step_m:
        return [], 0.0

    q_at_prev = dict(planner.mm.get_joint_pos_dict())
    lo, hi = 0.0, 1.0
    best_configs: list[dict] = []
    best_t = 0.0

    while span * (hi - lo) > min_step_m:
        mid_t = 0.5 * (lo + hi)
        box_m = pin.SE3.Interpolate(box_prev, box_next, mid_t)
        planner.mm.set_joint_pos(best_configs[-1] if best_configs else q_at_prev)
        if not _solve_local_only(planner, box_m):
            hi = mid_t
            continue
        q_next = dict(planner.mm.get_joint_pos_dict())
        q_from = best_configs[-1] if best_configs else q_at_prev
        jump = max(abs(q_next[n] - q_from[n]) for n in q_from)
        if jump > MAX_JOINT_JUMP:
            # Too large a joint step — treat as unreachable at this mid_t.
            hi = mid_t
            planner.mm.set_joint_pos(q_from)
            continue
        lo = mid_t
        best_t = mid_t
        best_configs.append(q_next)

    if best_configs:
        planner.mm.set_joint_pos(best_configs[-1])
    else:
        planner.mm.set_joint_pos(q_at_prev)
    return best_configs, best_t


def _residual_after(planner: TranslateMidToTipPlanner, configs: list[dict]) -> float:
    if not configs:
        return float(np.linalg.norm(planner.tip - planner.mid0))
    planner.mm.set_joint_pos(configs[-1])
    mid = green_ee_mid(planner.mm, planner.z_offset)
    err = float(np.linalg.norm(planner.tip - mid))
    planner.mm.set_joint_pos(planner.initial_q)
    return err


def plan_translate_to_tip(
    planner: TranslateMidToTipPlanner,
    step_m: float = 0.01,
) -> tuple[dict, list[dict]] | None:
    """Best-effort plan: minimize green→tip distance under IK/collision limits."""
    dist = float(np.linalg.norm(planner.delta))
    if dist < 1e-4:
        logger.info("Already at tip (within 0.1 mm); nothing to plan.")
        return dict(planner.initial_q), []

    candidates = _candidate_mid_paths(planner.mid0, planner.tip)
    best_label = None
    best_configs: list[dict] | None = None
    best_residual = dist

    for label, mids in candidates:
        path_len = _path_length(mids)
        logger.info(f"Trying approach path '{label}' (len={path_len:.3f} m)...")
        dense = _densify_mids(mids, step_m)
        configs = _forward_track_mids(planner, dense)
        planner.mm.set_joint_pos(planner.initial_q)
        if not configs:
            logger.warning(f"  '{label}': no progress")
            continue
        jump = max_adjacent_jump([planner.initial_q] + configs)
        if jump > MAX_JOINT_JUMP:
            logger.warning(f"  '{label}': jump {jump:.2f} rad — skip")
            continue
        residual = _residual_after(planner, configs)
        logger.info(
            f"  '{label}': {len(configs)} configs, residual green→tip {residual:.4f} m"
        )
        if residual < best_residual - 1e-4:
            best_residual = residual
            best_configs = configs
            best_label = label
            if residual <= _CLOSE_ENOUGH_M:
                break

    if not best_configs:
        logger.error("No collision-free progress toward tip on any path.")
        return None

    covered = max(0.0, 1.0 - best_residual / max(dist, 1e-6))
    if best_residual <= _CLOSE_ENOUGH_M:
        logger.info(
            f"Tip reached via '{best_label}' (residual {best_residual:.4f} m)."
        )
    else:
        logger.warning(
            f"Tip not fully reachable; best effort via '{best_label}' "
            f"(residual {best_residual:.4f} m, closed ~{100.0 * covered:.0f}% of "
            "initial gap). Moving green mid as close as physically possible."
        )

    planner.mm.set_joint_pos(planner.initial_q)
    return dict(planner.initial_q), best_configs


def execute_until_collision(
    bot: Robot,
    mm: MotionManager,
    torso_joints: list[str],
    start_config: dict,
    configs: list[dict],
    duration: float,
    control_hz: float,
) -> int:
    """Stream the path; abort if a knot is flagged colliding by local IK.

    Returns the number of configs that were commanded before a stop.
    """
    if not configs:
        return 0

    # Re-validate each knot just before streaming that segment; pink ``coll``
    # means self-collision — freeze at the last safe command.
    safe: list[dict] = []
    q_prev = dict(start_config)
    mm.set_joint_pos(start_config)
    for i, q in enumerate(configs):
        mm.set_joint_pos(q)
        poses = mm.fk(["L_ee", "R_ee"])
        tl = pin.SE3(poses["L_ee"].homogeneous.copy())
        tr = pin.SE3(poses["R_ee"].homogeneous.copy())
        sol, coll, lim = mm.ik(
            target_pos={"L_ee": tl.translation, "R_ee": tr.translation},
            target_rot={"L_ee": tl.rotation, "R_ee": tr.rotation},
            max_iterations=5,
        )
        if coll or not lim or sol is None:
            logger.warning(
                f"Collision detected at config {i + 1}/{len(configs)}; "
                "stopping arm motion."
            )
            mm.set_joint_pos(q_prev)
            break
        # Reject huge jumps mid-stream.
        jump = max(abs(q[n] - q_prev[n]) for n in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS if n in q)
        if jump > MAX_JOINT_JUMP:
            logger.warning(
                f"Unsafe joint jump {jump:.2f} rad at config {i + 1}; stopping."
            )
            break
        safe.append(q)
        q_prev = q

    if not safe:
        logger.error("No safe configs to execute.")
        return 0

    frac = len(safe) / len(configs)
    exec_duration = max(duration * frac, 0.5)
    logger.info(
        f"Executing {len(safe)}/{len(configs)} configs over {exec_duration:.1f}s"
    )
    execute_trajectory(bot, torso_joints, start_config, safe, exec_duration, control_hz)
    return len(safe)


def main(
    target: tuple[float, float, float] | None = None,
    tip_file: Path | None = None,
    ee_mid_z_offset: float = 0.05,
    step_m: float = 0.01,
    duration: float = 2.0,
    control_hz: float = 100.0,
    torso_damping: float = 1.0,
    lock_torso: bool = False,
    leave_held: bool = True,
) -> None:
    """Move EE-mid (green) as close as possible to the box tip (red).

    Args:
        target: Explicit tip XYZ in **robot base frame**. Overrides tip_file.
        tip_file: Text file with robot-base ``x y z`` (from sim tip marker).
        ee_mid_z_offset: Same local +Z offset as sim ``--ee_mid_z_offset``.
        step_m: Path densification step (meters) along candidate mid paths.
        duration: Full-path stream time; shortened if only a prefix runs.
        control_hz: Command rate.
        torso_damping: Torso IK damping when unlocked.
        lock_torso: Freeze torso during the motion.
        leave_held: Exit without ``stop()`` so arms keep the last targets.
    """
    tip = load_tip_target(tip_file, target)
    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        hb = getattr(bot, "heartbeat", None) or getattr(bot, "_heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused for IK planning.")

        try:
            mm, torso_joints, initial_joint_pos = create_motion_manager(
                bot, torso_damping=torso_damping, lock_torso=lock_torso
            )
            planner = TranslateMidToTipPlanner(
                mm, tip=tip, z_offset=ee_mid_z_offset
            )
            planned = plan_translate_to_tip(planner, step_m=step_m)
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
            logger.error("Planning failed.")
            do_shutdown = True
            return

        start_cfg, configs = planned
        if not configs:
            logger.info("Already at tip.")
            if leave_held:
                disconnect_leave_held(bot)
                do_shutdown = False
            return

        n_safe = execute_until_collision(
            bot,
            mm,
            torso_joints,
            start_cfg,
            configs,
            duration=duration,
            control_hz=control_hz,
        )
        if n_safe == 0:
            do_shutdown = True
            return

        from dexcomm import RateLimiter

        q_final = configs[n_safe - 1]
        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        q = np.array([q_final[n] for n in names], dtype=float)
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.left_arm.set_joint_pos(q[:7])
            bot.right_arm.set_joint_pos(q[7:14])
            if torso_joints:
                bot.torso.set_joint_pos(q[14:])
            rate.sleep()

        mm.set_joint_pos(q_final)
        mid = green_ee_mid(mm, ee_mid_z_offset)
        logger.info(
            f"Done. green_mid≈{np.round(mid, 4)}, tip={np.round(tip, 4)}, "
            f"err={np.linalg.norm(mid - tip):.4f} m"
        )

        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() — pose held.")
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        do_shutdown = True
    finally:
        if do_shutdown:
            logger.info("Shutting down robot")
            bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
