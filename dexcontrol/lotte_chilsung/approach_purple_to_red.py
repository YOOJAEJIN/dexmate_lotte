# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Left-arm only: move the purple box-center onto the red conveyor tip.

Assumes the small card stack (purple marker) is already held in the **left**
hand. Translates ``L_ee`` in Cartesian space by

    delta = tip_red − purple_center

(both in robot base frame) while keeping left EE orientation fixed. Right arm /
torso / head stay frozen. This is the single-arm analogue of ``approach_tip.py``
("magnet" attach purple → red).

Sim must be writing (robot base):

  - ``unitree_sim_isaaclab/card_box_tip_pos.txt``          (red)
  - ``unitree_sim_isaaclab/card_box_mass_center_pos.txt``   (purple)

Example::

    python approach_purple_to_red.py
    python approach_purple_to_red.py --duration 5
    python approach_purple_to_red.py --scale 0.5   # only go halfway
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

_DEX_ROOT = Path(__file__).resolve().parents[2]  # <dex>/  (this file lives at <dex>/dexcontrol/lotte_chilsung/)
_DEFAULT_TIP_CANDIDATES = [
    _DEX_ROOT / "unitree_sim_isaaclab" / "card_box_tip_pos.txt",
    _DEX_ROOT / "unitree_sim_isaaclab_jjy" / "card_box_tip_pos.txt",
]
_DEFAULT_PURPLE_CANDIDATES = [
    _DEX_ROOT / "unitree_sim_isaaclab" / "card_box_mass_center_pos.txt",
    _DEX_ROOT / "unitree_sim_isaaclab_jjy" / "card_box_mass_center_pos.txt",
]


def _resolve_existing(path: Path | None, candidates: list[Path], label: str) -> Path:
    if path is not None:
        return path
    for cand in candidates:
        if cand.is_file():
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"{label} file not found. Tried:\n  {tried}\n"
        "Restart sim_main.py (needs update_card_box_mass_center_marker writing "
        "card_box_mass_center_pos.txt each step)."
    )

LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]


def _load_xyz(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} file not found: {path}. Is the sim running and writing markers?"
        )
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


def _normalize_traj_knots(qs_sample) -> np.ndarray:
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


def _left_from_knot(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=float).reshape(-1)
    if flat.size >= 14:
        return flat[:7].copy()
    if flat.size == 7:
        return flat.copy()
    raise ValueError(f"Unexpected trajectory knot size {flat.size}")


def _command(
    bot: Robot,
    hold_q: dict[str, float],
    left: np.ndarray,
) -> None:
    """Command left arm; freeze right / torso at ``hold_q`` (head untouched)."""
    bot.left_arm.set_joint_pos(np.asarray(left, dtype=float).reshape(7))
    bot.right_arm.set_joint_pos(
        np.array([hold_q[n] for n in RIGHT_ARM_JOINTS], dtype=float)
    )
    if bot.has_component("torso"):
        bot.torso.set_joint_pos(np.array([hold_q[n] for n in TORSO_JOINTS], dtype=float))


def _stream_left_only(
    bot: Robot,
    hold_q: dict[str, float],
    traj: np.ndarray,
    duration: float,
    control_hz: float,
) -> np.ndarray:
    n = traj.shape[0]
    n_seg = max(n - 1, 1)
    rate = RateLimiter(control_hz)
    n_ticks = max(int(duration * control_hz), 1)
    left = _left_from_knot(traj[-1])
    for t in range(1, n_ticks + 1):
        s = 0.5 * (1.0 - np.cos(np.pi * t / n_ticks)) * n_seg
        idx = min(int(s), max(n_seg - 1, 0))
        frac = s - idx
        if n == 1:
            left = _left_from_knot(traj[0])
        else:
            l0 = _left_from_knot(traj[idx])
            l1 = _left_from_knot(traj[min(idx + 1, n - 1)])
            left = (1.0 - frac) * l0 + frac * l1
        _command(bot, hold_q, left)
        rate.sleep()
    return left


def _hold_final(
    bot: Robot,
    hold_q: dict[str, float],
    left: np.ndarray,
    control_hz: float,
    hold_s: float,
) -> None:
    rate = RateLimiter(control_hz)
    for _ in range(max(int(hold_s * control_hz), 5)):
        _command(bot, hold_q, left)
        rate.sleep()


def main(
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    scale: float = 1.0,
    duration: float = 5.0,
    control_hz: float = 100.0,
    hold_s: float = 0.3,
    leave_held: bool = True,
) -> None:
    """Translate left EE so purple box center approaches red tip (left arm only).

    Args:
        tip_file: Red tip XYZ file (robot base). Default: sim tip file.
        purple_file: Purple center XYZ file (robot base). Default: sim center file.
        scale: Fraction of the full delta to apply (1.0 = full magnet attach).
            Use e.g. 0.5 to only close half the gap.
        duration: Trajectory stream time (seconds). Larger = slower. Default 5.
        control_hz: Command rate.
        hold_s: Re-assert final joints before disconnect.
        leave_held: Exit without ``stop()`` so pose stays held.
    """
    tip_path = _resolve_existing(tip_file, _DEFAULT_TIP_CANDIDATES, "red tip")
    purple_path = _resolve_existing(
        purple_file, _DEFAULT_PURPLE_CANDIDATES, "purple center"
    )
    tip = _load_xyz(tip_path, "red tip")
    purple = _load_xyz(purple_path, "purple center")
    delta = (tip - purple) * float(scale)
    dist = float(np.linalg.norm(delta))
    logger.info(
        f"Magnet purple→red (left arm only):\n"
        f"  purple  {np.round(purple, 4)}\n"
        f"  tip     {np.round(tip, 4)}\n"
        f"  delta   {np.round(delta, 4)}  |d|={dist:.4f} m  scale={scale:g}"
    )
    if dist < 1e-4:
        logger.info("Already aligned (within 0.1 mm); nothing to do.")
        return

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    final_left: np.ndarray | None = None
    hold_q: dict[str, float] | None = None
    try:
        components = ["left_arm", "right_arm", "head"]
        if bot.has_component("torso"):
            components.append("torso")
        hold_q = bot.get_joint_pos_dict(component=components)
        logger.info("Right arm / torso / head frozen; only left EE translates.")

        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused during motion planning.")

        traj = np.zeros((0, 0), dtype=float)
        try:
            mm = make_fast_mm(
                hold_q,
                [
                    "HEAD",
                    "BASE",
                    "RIGHT_ARM",
                    "LEFT_HAND",
                    "RIGHT_HAND",
                    "TORSO",
                ],
                need_traj_gen=True,
            )
            # Pure translation of L_ee; orientation unchanged → box center
            # (fixed in left grasp) moves by the same delta.
            qs_sample = mm.left_arm.move_ee_xyz(delta)
            traj = _normalize_traj_knots(qs_sample)
            if traj.shape[0] == 0:
                logger.error("move_ee_xyz returned no trajectory. Robot was not moved.")
                do_shutdown = True
                return
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
            f"Planned {traj.shape[0]} knots; streaming left arm over {duration:.1f}s"
        )
        final_left = _stream_left_only(
            bot, hold_q, traj, duration=duration, control_hz=control_hz
        )
        _hold_final(bot, hold_q, final_left, control_hz, hold_s)
        del mm

        logger.info("Approach completed; final pose left held.")
        if leave_held:
            _hold_final(bot, hold_q, final_left, control_hz, 0.2)
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
        logger.warning("Interrupted — trying to leave last pose held.")
        if leave_held and hold_q is not None and final_left is not None:
            try:
                _hold_final(bot, hold_q, final_left, control_hz, 0.2)
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
