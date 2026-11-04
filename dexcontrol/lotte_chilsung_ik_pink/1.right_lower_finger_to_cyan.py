# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Move the right gripper's LOWER finger tip to just in front of the cyan card-edge marker.

The cyan marker (``card_mid_edge_ny_marker`` in the Chilsung sim scene) sits on the -X edge
of the middle card of the small stack, at y = -0.0875 m from the card centre. ``sim_main``
writes all four edge markers in **robot base frame** to
``unitree_sim_isaaclab*/card_edge_markers_pos.txt``; this script reads the cyan one, adds a
small stand-off toward the robot (base -X, the side the edge faces) and plans a straight
Cartesian move of the right arm with Pink IK so that the lower finger *tip* ends up there.
Gripper orientation is held; the left arm is never commanded (see ``_pink_right.py`` for the
setpoint anchoring, torso mode and measured-state corrections shared with step 7).

Finger geometry comes from the sim's LG vertical gripper USD (the IK URDF still carries the
2F gripper, so the finger offsets are expressed relative to ``R_arm_l8`` / ``R_gripper_base``):
two fingers slide along ``R_arm_l8`` +X, the long blade ``Finger_1`` (R_Slider_1) on the +X
side and the short ``Finger2_1`` (R_Slider_2) on the -X side. Which one is *lower* depends on
the wrist pose, so it is decided from FK at run time (``--finger`` overrides). Current slider
travel is read from ``/state/custom_ee/right`` when available (else assumed fully open).

Example (sim running, ROBOT_NAME / ZENOH_CONFIG exported)::

    python 6.right_lower_finger_to_cyan.py                     # 3 cm in front of cyan, 3 s
    python 6.right_lower_finger_to_cyan.py --standoff-m 0.02 --duration 4
    python 6.right_lower_finger_to_cyan.py --target 0.82 -0.0675 0.7937   # robot-base XYZ
    python 6.right_lower_finger_to_cyan.py --finger finger_1 --slider 0 0
    python 6.right_lower_finger_to_cyan.py --dry-run           # plan only, no robot
    python 6.right_lower_finger_to_cyan.py --use-torso         # arm + torso IK, streamed together
    python 6.right_lower_finger_to_cyan.py --use-torso --torso-posture-cost 0.001   # torso shares the motion
    python 6.right_lower_finger_to_cyan.py --refine 0          # no measured-state correction moves
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pink
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pink_right import (  # noqa: E402
    TORSO_MAX_VEL,
    TORSO_POSTURE_COST,
    gripper_pose_b,
    run_point_move,
)

# ---------------------------------------------------------------------------
# Marker file (written by unitree_sim_isaaclab*/sim_main.py, robot base frame)
# ---------------------------------------------------------------------------
_DEX_ROOT = Path(__file__).resolve().parents[2]  # <dex>/
MARKER_FILE_NAME = "card_edge_markers_pos.txt"
SIM_DIR_CANDIDATES = ("unitree_sim_isaaclab_260918", "unitree_sim_isaaclab")
CYAN_MARKER = "card_mid_edge_ny_marker"
# Nominal cyan marker in robot base for the Chilsung spawn (fallback only): middle card object12
# centre (0.795, -1.195, 0.8190) world, -X edge = x - 0.200, y - 0.0875; robot base at
# (0.13, -1.21, 0.04426) with zero yaw.
DEFAULT_TARGET_B = (0.465, -0.0725, 0.7747)

# ---------------------------------------------------------------------------
# LG vertical gripper fingers (2nd gen 2FPA-I-03-180, pedestal disc removed, 2026-10-30), measured in
# the R_arm_l8 frame at zero slider travel (assets/robots/vega-1p-usd/vega_1p_vertical_gripper.usd, pxr bbox dump):
#   Finger_1  (R_Slider_1, travel [-0.082, 0]): x [+0.082, +0.102]  y +/-0.028  z [0.0942, 0.3031]  (long blade)
#   Finger2_1 (R_Slider_2, travel [0, +0.082]): x [-0.102, -0.082]  y +/-0.028  z [0.0942, 0.1982]  (short)
# 1st gen (06.LG): Finger2_1 was the long blade, x +/-[0.082, 0.107], z [0.1105, 0.3194] / [0.1105, 0.2195].
# Travel adds to x (both close toward x = 0). l8 +Z is the gripper approach axis; the tip is
# the z-max end at mid-thickness. IK URDF: R_gripper_base = R_arm_l8 translated 0.078 along z.
# ---------------------------------------------------------------------------
GB_FROM_L8_Z = 0.078
FINGERS = {
    "finger_1": {"slider_index": 0, "x0": +0.092, "tip_z": 0.3031, "label": "Finger_1 (long blade, R_Slider_1)"},
    "finger2_1": {"slider_index": 1, "x0": -0.092, "tip_z": 0.1982, "label": "Finger2_1 (short finger, R_Slider_2)"},
}


def find_marker_file(explicit: Path | None) -> Path | None:
    """Explicit path, else the newest ``card_edge_markers_pos.txt`` among the sim folders."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    found = [p for p in (_DEX_ROOT / d / MARKER_FILE_NAME for d in SIM_DIR_CANDIDATES) if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def load_target_b(target: tuple[float, float, float] | None, marker_file: Path | None) -> np.ndarray:
    """Cyan marker XYZ in robot base frame (CLI > marker file > nominal spawn value)."""
    if target is not None:
        p = np.asarray(target, dtype=float)
        logger.info(f"Using CLI target (robot base): {np.round(p, 4)}")
        return p
    path = find_marker_file(marker_file)
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            vals = line.split()
            if len(vals) >= 4 and vals[0] == CYAN_MARKER:
                p = np.array([float(v) for v in vals[1:4]])
                logger.info(f"Loaded {CYAN_MARKER} (robot base) from {path}: {np.round(p, 4)}")
                return p
        logger.warning(f"{path} has no '{CYAN_MARKER}' line; using nominal {DEFAULT_TARGET_B}")
    else:
        logger.warning(f"No {MARKER_FILE_NAME} found under {_DEX_ROOT}/{{{','.join(SIM_DIR_CANDIDATES)}}}; "
                       f"using nominal {DEFAULT_TARGET_B}")
    return np.asarray(DEFAULT_TARGET_B, dtype=float)


def read_slider_pos(timeout: float = 2.0) -> np.ndarray | None:
    """[R_Slider_1, R_Slider_2] travel (m) from the sim's ``/state/custom_ee/right``, or None."""
    try:
        from _hands import HandCommander

        hands = HandCommander(sides=("right",), node_name="lotte_finger_probe")
        try:
            if hands.wait_for_state(timeout):
                pos, _ = hands.state("right")
                if pos is not None and pos.size >= 2:
                    return pos[:2].astype(float)
        finally:
            hands.shutdown()
    except Exception as exc:  # no ROBOT_NAME, no zenoh, ...
        logger.warning(f"Slider state unavailable ({exc}).")
    return None


def finger_tips_gb(slider: np.ndarray) -> dict[str, np.ndarray]:
    """Finger tip points in the R_gripper_base frame for the given slider travel."""
    return {
        name: np.array([f["x0"] + float(slider[f["slider_index"]]), 0.0, f["tip_z"] - GB_FROM_L8_Z])
        for name, f in FINGERS.items()
    }


def pick_finger(configuration: pink.Configuration, tips_gb: dict[str, np.ndarray], finger: str) -> str:
    """``finger`` if given, else the finger whose tip is lower in the robot base frame."""
    T_gb = gripper_pose_b(configuration)
    z = {name: float(T_gb.act(tip)[2]) for name, tip in tips_gb.items()}
    for name in FINGERS:
        logger.info(f"  {FINGERS[name]['label']}: tip z = {z[name]:.3f} m (robot base)")
    if finger != "auto":
        return finger
    lower = min(z, key=z.get)
    logger.info(f"Lower finger: {FINGERS[lower]['label']}")
    return lower


def goal_from_marker(target_b: np.ndarray, standoff_m: float) -> np.ndarray:
    """Finger-tip goal: marker moved ``standoff_m`` toward the robot along base -X."""
    return target_b + np.array([-standoff_m, 0.0, 0.0])


def main(
    target: tuple[float, float, float] | None = None,
    marker_file: Path | None = None,
    standoff_m: float = 0.03,
    finger: str = "auto",
    slider: tuple[float, float] | None = None,
    duration: float = 3.0,
    step_m: float = 0.01,
    leave_held: bool = True,
    dry_run: bool = False,
    use_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    refine: int = 2,
    refine_tol_m: float = 0.005,
    settle_s: float = 0.7,
    anchor_setpoints: bool = True,
    auto_torso: bool = True,
    ignore_foreign_commands: bool = False,
) -> None:
    """Put the right gripper's lower finger tip in front of the cyan card-edge marker.

    Args:
        target: Cyan marker XYZ in robot base frame (overrides the marker file).
        marker_file: Path to card_edge_markers_pos.txt (default: newest under the sim folders).
        standoff_m: Gap between finger tip and marker along robot-base -X (toward the robot).
        finger: "auto" (lower tip by FK), "finger_1" (long blade) or "finger2_1" (short finger).
        slider: [R_Slider_1, R_Slider_2] travel in metres (default: read from the sim, else 0 0 = open).
        duration: Stream time for the main move (seconds); corrections are scaled down.
        step_m: Cartesian spacing between IK waypoints (metres).
        leave_held: Exit without ``stop()`` so the pose stays held.
        dry_run: Plan from the URDF neutral pose without connecting; report reachability only.
        use_torso: Also move the torso (torso_j1..j3) in the IK; left arm keeps its joint angles.
        torso_posture_cost: Posture cost on torso joints; higher = torso moves less (arms use 1e-3).
        torso_max_vel: Torso joint speed cap (rad/s); the stream duration is stretched to respect it.
        refine: Max number of measured-state correction moves after the main move (0 = none).
        refine_tol_m: Stop refining once the measured tip error is below this (metres).
        settle_s: Wait before measuring so the controller settles (seconds).
        anchor_setpoints: Read the sim's held targets and anchor the stream there (recommended).
        auto_torso: If the arm alone cannot plan the move (fully extended), retry with the torso.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    if finger not in ("auto", *FINGERS):
        raise ValueError(f"finger must be one of auto, {', '.join(FINGERS)}")
    target_b = load_target_b(target, marker_file)
    goal_b = goal_from_marker(target_b, standoff_m)

    # Slider probe uses its own short-lived dexcomm node, before any Robot client exists.
    slider_arr = np.asarray(slider, dtype=float) if slider is not None else (
        None if dry_run else read_slider_pos()
    )
    if slider_arr is None:
        slider_arr = np.zeros(2)
        logger.warning("Assuming sliders fully open (0, 0); pass --slider to override.")
    logger.info(f"Slider travel [R_Slider_1, R_Slider_2] = {np.round(slider_arr, 4)} m")
    tips = finger_tips_gb(slider_arr)

    def spec(configuration: pink.Configuration) -> tuple[np.ndarray, np.ndarray]:
        name = pick_finger(configuration, tips, finger)
        logger.info(f"Goal = marker {np.round(target_b, 4)} + stand-off {standoff_m:.3f} m along base -X")
        return tips[name], goal_b

    run_point_move(
        spec, label="lower-finger approach", duration=duration, step_m=step_m, use_torso=use_torso,
        torso_posture_cost=torso_posture_cost, torso_max_vel=torso_max_vel, refine=refine,
        refine_tol_m=refine_tol_m, settle_s=settle_s, anchor_setpoints=anchor_setpoints, auto_torso=auto_torso, ignore_foreign_commands=ignore_foreign_commands,
        dry_run=dry_run, leave_held=leave_held,
    )


if __name__ == "__main__":
    tyro.cli(main)
