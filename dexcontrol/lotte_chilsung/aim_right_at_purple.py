# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Aim right EE +Z at the purple ball from an outward-hug right-arm seed.

Pipeline:
  1. Move the right arm in joint space toward the outward-hug seed
     (elbow tucked / arm wrapped outward — user reference pose).
  2. Keep that hug EE **position**, rotate so local +Z looks at the purple
     center head-on (face ⊥ line-of-sight). Left arm / torso / head held.

No contact with the ball is required. World −X station moves are not used;
the hug posture defines where the hand sits.

Example::

    python aim_right_at_purple.py
    python aim_right_at_purple.py --duration 5
    python aim_right_at_purple.py --neg-z
    python aim_right_at_purple.py --skip-hug   # look-at only from current pose
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import tyro
from dexcomm import RateLimiter
from dexmotion.motion_manager import MotionManager
from loguru import logger
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bimanual_box import (  # noqa: E402
    IK_MAX_ITERS,
    IK_MAX_REPS,
    MAX_JOINT_JUMP,
    POS_TOL,
    RIGHT_ARM_JOINTS,
    ROT_TOL,
    SOFT_POS_TOL,
    SOFT_ROT_TOL,
    disconnect_leave_held,
    make_fast_mm,
    rot_err_deg,
    connect_bot,
)
from dexcontrol.robot import Robot

_DEX_ROOT = Path(__file__).resolve().parents[2]  # <dex>/  (this file lives at <dex>/dexcontrol/lotte_chilsung/)
_DEFAULT_PURPLE_CANDIDATES = [
    _DEX_ROOT / "unitree_sim_isaaclab" / "card_box_mass_center_pos.txt",
    _DEX_ROOT / "unitree_sim_isaaclab_jjy" / "card_box_mass_center_pos.txt",
]

# Outward-hug right-arm seed (user reference). Left/torso/head stay on robot.
HUG_OUTWARD_RIGHT: dict[str, float] = {
    "R_arm_j1": -0.9685400724411011,
    "R_arm_j2": 0.0006378149846568704,
    "R_arm_j3": -0.1236281469464302,
    "R_arm_j4": -1.4418329000473022,
    "R_arm_j5": -0.0005037019727751613,
    "R_arm_j6": -0.4133748412132263,
    "R_arm_j7": 0.8297421932220459,
}


def _resolve_existing(path: Path | None, candidates: list[Path], label: str) -> Path:
    if path is not None:
        return path
    for cand in candidates:
        if cand.is_file():
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"{label} file not found. Tried:\n  {tried}\n"
        "Restart sim_main.py so it writes card_box_mass_center_pos.txt."
    )


def _load_xyz(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = line.split()
        if len(vals) >= 3:
            xyz = np.array([float(vals[0]), float(vals[1]), float(vals[2])])
            logger.info(f"Loaded {label} (robot base) from {path}: {np.round(xyz, 4)}")
            return xyz
    raise ValueError(f"{label} file malformed: {path}")


def _look_at_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    *,
    neg_z: bool = False,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """EE rotation: +Z (or −Z) → target, face ⊥ line-of-sight, upright."""
    eye = np.asarray(eye, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    forward = target - eye
    n = float(np.linalg.norm(forward))
    if n < 1e-8:
        raise ValueError("eye and target coincide")
    forward = forward / n
    z_axis = -forward if neg_z else forward

    up = np.array([0.0, 0.0, 1.0]) if world_up is None else np.asarray(world_up, float)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    if abs(float(z_axis @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(up, z_axis)
    xn = float(np.linalg.norm(x_axis))
    if xn < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0])
        x_axis = x_axis - (x_axis @ z_axis) * z_axis
        xn = float(np.linalg.norm(x_axis))
    x_axis = x_axis / xn
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    return np.column_stack([x_axis, y_axis, z_axis])


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.rad2deg(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _reanchor_posture(mm: MotionManager) -> None:
    try:
        solver = mm.local_ik_solver
        posture = solver.task_dict.get("posture")
        if posture is not None:
            posture.set_target_from_configuration(solver.configuration)
    except AttributeError:
        pass


def _solve_right_ee(mm: MotionManager, tr: pin.SE3, max_reps: int = IK_MAX_REPS) -> bool:
    """Local IK for R_ee only (L_ee held at current FK)."""
    q_prev = dict(mm.get_joint_pos_dict())
    poses = mm.fk(["L_ee", "R_ee"])
    tl = pin.SE3(poses["L_ee"].homogeneous.copy())
    _reanchor_posture(mm)

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
        cur = mm.fk(["R_ee"])["R_ee"]
        perr = float(np.linalg.norm(cur.translation - tr.translation))
        rerr = rot_err_deg(cur.rotation, tr.rotation)
        if perr < POS_TOL and rerr < ROT_TOL:
            return True
    if perr < SOFT_POS_TOL and rerr < SOFT_ROT_TOL:
        return True
    mm.set_joint_pos(q_prev)
    return False


def _apply_right(q: dict, right: dict[str, float]) -> dict:
    out = dict(q)
    for name in RIGHT_ARM_JOINTS:
        out[name] = float(right[name])
    return out


def plan_hug_then_aim(
    mm: MotionManager,
    target_pos: np.ndarray,
    *,
    hug_right: dict[str, float] = HUG_OUTWARD_RIGHT,
    n_hug: int = 20,
    n_aim: int = 24,
    neg_z: bool = False,
    skip_hug: bool = False,
) -> tuple[dict, list[dict]] | None:
    """Joint-space hug seed, then in-place look-at purple with R_ee +Z."""
    purple = np.asarray(target_pos, dtype=float).reshape(3)
    initial_q = dict(mm.get_joint_pos_dict())
    configs: list[dict] = []

    # --- Phase 1: outward-hug right arm (joint-space) ---
    if skip_hug:
        hug_q = dict(initial_q)
        logger.info("Skipping hug seed; aiming from current right-arm pose.")
    else:
        q0 = np.array([initial_q[n] for n in RIGHT_ARM_JOINTS], dtype=float)
        q1 = np.array([hug_right[n] for n in RIGHT_ARM_JOINTS], dtype=float)
        hug_err = float(np.max(np.abs(q1 - q0)))
        logger.info(
            f"Phase 1: outward-hug right arm "
            f"(max |Δq|={hug_err:.3f} rad, {n_hug} steps)"
        )
        mm.set_joint_pos(initial_q)
        n_h = max(int(n_hug), 1)
        if hug_err < 1e-3:
            hug_q = _apply_right(initial_q, hug_right)
            mm.set_joint_pos(hug_q)
            logger.info("Already at hug seed.")
        else:
            hug_q = dict(initial_q)
            for i in range(1, n_h + 1):
                a = i / n_h
                # Cosine ease for a softer wrap.
                a = 0.5 * (1.0 - np.cos(np.pi * a))
                right_i = {
                    n: float((1.0 - a) * q0[k] + a * q1[k])
                    for k, n in enumerate(RIGHT_ARM_JOINTS)
                }
                hug_q = _apply_right(initial_q, right_i)
                # Keep left/torso/head from initial; only right changes.
                configs.append(dict(hug_q))
            mm.set_joint_pos(hug_q)

    # Anchor IK posture at the hug configuration so wrist IK stays nearby.
    _reanchor_posture(mm)

    # --- Phase 2: rotate in place so +Z looks at purple ---
    poses = mm.fk(["R_ee"])
    right_h = pin.SE3(poses["R_ee"].homogeneous.copy())
    eye = right_h.translation.copy()
    look = purple - eye
    dist = float(np.linalg.norm(look))
    if dist < 1e-4:
        logger.error("Purple coincides with R_ee at hug pose; cannot aim.")
        mm.set_joint_pos(initial_q)
        return None

    rot_tgt = _look_at_rotation(eye, purple, neg_z=neg_z)
    approach0 = -right_h.rotation[:, 2] if neg_z else right_h.rotation[:, 2]
    ang0 = _angle_between(approach0, look)
    logger.info(
        f"Phase 2: look-at purple from hug EE (in-place rotate):\n"
        f"  R_ee     {np.round(eye, 4)}\n"
        f"  purple   {np.round(purple, 4)}  dist={dist:.3f} m\n"
        f"  +Z off-axis now {ang0:.1f} deg → 0 at target"
    )

    if ang0 < 0.5:
        logger.info("Already looking at purple from hug pose.")
        mm.set_joint_pos(initial_q)
        return initial_q, configs

    key_rots = R.from_matrix(np.stack([right_h.rotation, rot_tgt], axis=0))
    slerp = Slerp([0.0, 1.0], key_rots)
    n_a = max(int(n_aim), 2)
    mm.set_joint_pos(hug_q)
    aim_prefix_ok = True
    for i in range(1, n_a + 1):
        a = i / n_a
        tr = pin.SE3(slerp([a]).as_matrix()[0], eye.copy())
        q_before = dict(mm.get_joint_pos_dict())
        if not _solve_right_ee(mm, tr):
            logger.warning(
                f"Aim IK failed at {100.0 * i / n_a:.0f}%; using safe prefix."
            )
            mm.set_joint_pos(q_before)
            aim_prefix_ok = False
            break
        q_next = dict(mm.get_joint_pos_dict())
        jump = max(abs(q_next[k] - q_before[k]) for k in q_before)
        if jump > MAX_JOINT_JUMP:
            logger.warning(f"Joint jump {jump:.2f} rad during aim; stopping.")
            mm.set_joint_pos(q_before)
            aim_prefix_ok = False
            break
        configs.append(q_next)

    if not configs:
        logger.error("No hug/aim progress.")
        mm.set_joint_pos(initial_q)
        return None

    mm.set_joint_pos(configs[-1])
    r_end = mm.fk(["R_ee"])["R_ee"]
    app_end = -r_end.rotation[:, 2] if neg_z else r_end.rotation[:, 2]
    look_end = purple - r_end.translation
    resid = _angle_between(app_end, look_end)
    # How close final right arm stayed to hug seed.
    hug_resid = max(
        abs(configs[-1][n] - hug_right[n]) for n in RIGHT_ARM_JOINTS
    )
    mm.set_joint_pos(initial_q)
    logger.info(
        f"Planned {len(configs)} waypoints "
        f"(hug→aim{' partial' if not aim_prefix_ok else ''}); "
        f"look-at residual {resid:.2f} deg, "
        f"max |q−hug|={hug_resid:.3f} rad"
    )
    return initial_q, configs


def _stream_right(
    bot: Robot,
    start_cfg: dict,
    configs: list[dict],
    duration: float,
    control_hz: float,
) -> None:
    """Stream right-arm joints only (left untouched)."""
    knots = np.array(
        [[c[n] for n in RIGHT_ARM_JOINTS] for c in [start_cfg] + configs],
        dtype=float,
    )
    n_seg = max(len(knots) - 1, 1)
    rate = RateLimiter(control_hz)
    n_ticks = max(int(duration * control_hz), 1)
    for t in range(1, n_ticks + 1):
        s = 0.5 * (1.0 - np.cos(np.pi * t / n_ticks)) * n_seg
        idx = min(int(s), max(n_seg - 1, 0))
        frac = s - idx
        if len(knots) == 1:
            right = knots[0]
        else:
            right = (1.0 - frac) * knots[idx] + frac * knots[min(idx + 1, len(knots) - 1)]
        bot.right_arm.set_joint_pos(right)
        rate.sleep()


def main(
    purple_file: Path | None = None,
    duration: float = 2.0,
    control_hz: float = 100.0,
    n_hug: int = 12,
    n_aim: int = 12,
    skip_hug: bool = False,
    neg_z: bool = False,
    leave_held: bool = True,
) -> None:
    """Outward-hug right arm, then look at purple with EE +Z.

    Args:
        purple_file: Robot-base XYZ file for the purple center.
        duration: Stream time for the combined hug+aim path (seconds).
        control_hz: Command rate.
        n_hug: Joint-space steps toward the outward-hug seed.
        n_aim: Orientation IK steps for look-at purple.
        skip_hug: If True, only rotate from the current pose (no hug move).
        neg_z: If True, aim with local −Z (when tool front is −Z).
        leave_held: Exit without ``stop()`` so the pose stays held.
    """
    purple_path = _resolve_existing(
        purple_file, _DEFAULT_PURPLE_CANDIDATES, "purple center"
    )
    purple = _load_xyz(purple_path, "purple center")

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        components = ["left_arm", "right_arm", "head"]
        if bot.has_component("torso"):
            components.append("torso")
        hold_q = bot.get_joint_pos_dict(component=components)

        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused during motion planning.")

        try:
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
            )
            planned = plan_hug_then_aim(
                mm,
                purple,
                hug_right=HUG_OUTWARD_RIGHT,
                n_hug=n_hug,
                n_aim=n_aim,
                neg_z=neg_z,
                skip_hug=skip_hug,
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

        logger.info(
            f"Streaming right-arm hug→aim over {duration:.1f}s "
            "(left arm not commanded)..."
        )
        _stream_right(bot, start_cfg, configs, duration, control_hz)

        final_right = np.array(
            [configs[-1][n] for n in RIGHT_ARM_JOINTS], dtype=float
        )
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.right_arm.set_joint_pos(final_right)
            rate.sleep()

        logger.info("Aim completed; final pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info(
                "Client disconnected without stop() — pose held. "
                "Next teleop/control commands will take over."
            )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        do_shutdown = True
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
