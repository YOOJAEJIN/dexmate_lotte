# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Bimanual: carry the purple box centre onto the red conveyor tip (Pink IK version).

Port of ``lotte_chilsung_2/6.approach_purple_to_red_bimanual.py``. The box is held with both hands;
the grasp is kept rigid (relative L/R gripper poses and orientations fixed) and both grippers are
translated by

    delta = red_tip - purple_centre        (robot base frame)

so the purple marker lands on the red marker. The purple ball follows the centre of ``card_block_35``
(sim_main.update_card_box_mass_center_marker), which moves rigidly with the grasp, so translating the
gripper pair by ``delta`` moves the purple centre onto red.

IK: both arms + torso free in a reduced model with absolute gripper targets in the robot base frame
(``_pink_arm.build_dual_arm_ik``; ``--no-use-torso`` = arms only). Like the original, several
translation paths are tried -- straight line and axis-ordered detours (z then xy, xy then z, x-y-z,
z-y-x) -- each planned as far as IK / joint limits allow, and the one that leaves the smallest
purple-to-red residual is executed (the largest reachable prefix if none completes). Arms and torso
are streamed together with ``_pink_arm.stream`` (leash-paced, per-joint speed caps). There is no
self-collision model here; the tracking leash and the speed caps are the only safety nets.

Sim must be writing (robot base frame, see ``_sim_markers.py``)::

    unitree_sim_isaaclab*/card_box_tip_pos.txt          (red)
    unitree_sim_isaaclab*/card_box_mass_center_pos.txt  (purple)

Target (2026-09-26): red + (first-contact distance + ``--standoff`` air) along the box face normal, so the block
ends right next to the box; the red marker itself lies inside it. The first-contact distance is computed from the
block's yaw relative to the box face (both orientations come from the marker files): square block -> the gripper
finger plate between the boxes touches first (half thickness 5.25 cm + finger 2.5 cm = 7.75 cm from red to purple);
yawed block -> its leading corner touches first (5.25 cos(yaw) + 20 sin(yaw) cm). The boxes themselves can never be
flush where the finger sits; ``--contact-dist`` overrides the computed value.

Stage 0 (added 2026-09-26, ``--align-base``): before the arms move, the base turns in place by the bearing
difference between purple and that target (closed loop on the live marker files, capped at
``--yaw-cap-deg``, minus ``--yaw-short-deg``), so the remaining delta is mostly forward/down -- the torso
joints are all pitch and the arms cannot shift a two-handed grasp far sideways. Nothing else moves during
the turn; the markers are re-read afterwards. ``--no-align-base`` = the plain 2026-09-25 behaviour.
After the wheels stop the sim base keeps unwinding for a few seconds (2026-09-26 20:09: +1.8 deg residual grew
to +3.7 deg, i.e. 2.5 cm sideways at the block), so the script waits until the bearing is still
(``--settle-s``) and, if the gap is still above the tolerance, turns once more aiming past the target by the
miss just measured (``--max-turns``).

Stage 2 (added 2026-09-26, ``--push-max`` > 0): once the block stands at the standoff, the grasp keeps
pushing horizontally toward red (up to ``--push-max`` at ``--push-speed``) and stops the moment a wrist
F/T sensor feels the box: both wrists are tared while the block is free, low-pass filtered (0.25 s) and
compared with their own threshold ``max(--contact-n, 1.5 x rest noise + 3 N)`` for ``--confirm-ticks``
ticks (``ContactGate``). On contact the *measured* pose is held, so nothing keeps pressing. The push
length is the live gap to the jaw-contact distance plus ``--push-over``; the path is cut where the IK
starts swinging joints instead of translating (``--push-jump-rad`` per 5 mm step) -- the sideways
authority of a two-handed grasp is only a few cm -- and it is streamed under the same joint speed caps
as the approach, because a fast joint swing squeezes the block between the hands and reads as a contact
(38.9 N false stop on 2026-09-26 20:09). Without wrench data the push is limited to the nominal air gap.
``--push-max 0`` = stop at the standoff.

Example (box held with both hands, base already turned toward the conveyor)::

    python 10.approach_purple_to_red_bimanual.py --plan-only        # feasibility + residual, no motion
    python 10.approach_purple_to_red_bimanual.py                    # execute
    python 10.approach_purple_to_red_bimanual.py --stop-short 0.06  # stop 6 cm before red
    python 10.approach_purple_to_red_bimanual.py --offset 0 0.05 0  # red target + robot-base offset
    python 10.approach_purple_to_red_bimanual.py --scale 0.5        # close half the gap
    python 10.approach_purple_to_red_bimanual.py --no-use-torso     # arms only
    python 10.approach_purple_to_red_bimanual.py --no-align-base    # no base turn first
    python 10.approach_purple_to_red_bimanual.py --push-max 0       # place at the standoff, no F/T push
    python 10.approach_purple_to_red_bimanual.py --contact-n 20     # less sensitive contact detection
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    COMPONENT,
    TORSO_MAX_VEL,
    TORSO_POSTURE_COST,
    build_dual_arm_ik,
    detect_foreign_commands,
    plan_dual_pose_path,
    stream,
)
from _base_turn import WHEEL_GAIN_SIM, measure_rtf, turn_in_place, wrap  # noqa: E402
from _pink_common import EE, build_ik, connect, disconnect_leave_held, sync_from_robot  # noqa: E402
from _sim_markers import (  # noqa: E402
    PURPLE_FILE_NAME,
    TIP_FILE_NAME,
    base_yaw_world,
    find_sim_dir,
    load_xyz_base,
    world_dir_to_base,
    world_fixed_pairs,
)
from read_force import _get_arm_row, tare_force  # noqa: E402

CLOSE_ENOUGH_M = 0.01  # residual below this counts as reached
# Block ``card_block_35``: 40 x 35 x 10.5 cm, held upright with the 10.5 cm thickness along the box face normal.
BLOCK_HALF_T = 0.0525  # half thickness (along the face normal)
BLOCK_HALF_W = 0.20    # half width (horizontal, along the box face) -- the leading corner when the block is yawed
# lg_gripper finger plate thickness along its closing axis (Finger_1.stl / Finger2_1.stl: x = 82..107 mm). One finger
# sits between the block and the box, so with a square block that plate touches the box first.
FINGER_T = 0.025
JAW_CONTACT_M = BLOCK_HALF_T + FINGER_T  # 7.75 cm: purple-to-red along the normal at first contact, square block


def candidate_offset_paths(delta: np.ndarray) -> list[tuple[str, list[np.ndarray]]]:
    """Translation paths for the gripper pair, as waypoints of the offset from the start (0 -> delta).

    Straight line plus axis-ordered detours, which often reach farther (same set as the original).
    """
    dx, dy, dz = (float(v) for v in delta)
    z = np.array([0.0, 0.0, dz])
    xy = np.array([dx, dy, 0.0])
    x = np.array([dx, 0.0, 0.0])
    y = np.array([0.0, dy, 0.0])
    return [
        ("straight", [delta]),
        ("z_then_xy", [z, delta]),
        ("xy_then_z", [xy, delta]),
        ("x_y_z", [x, x + y, delta]),
        ("z_y_x", [z, z + y, delta]),
    ]


def densify(waypoints: list[np.ndarray], step_m: float) -> list[np.ndarray]:
    """Offsets from 0 through ``waypoints`` with consecutive points at most ``step_m`` apart (excludes 0)."""
    out: list[np.ndarray] = []
    cur = np.zeros(3)
    for nxt in waypoints:
        nxt = np.asarray(nxt, dtype=float)
        dist = float(np.linalg.norm(nxt - cur))
        if dist < 1e-9:
            continue
        n = max(int(np.ceil(dist / max(step_m, 1e-4))), 1)
        for i in range(1, n + 1):
            out.append((1.0 - i / n) * cur + (i / n) * nxt)
        cur = nxt
    return out


def _quiet_xyz(path: Path) -> np.ndarray:
    """First 'x y z' line of a marker file without logging (used at the chassis loop rate)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s_ = line.split()
        if s_ and s_[0] != "#" and len(s_) >= 3:
            return np.array([float(s_[0]), float(s_[1]), float(s_[2])])
    raise ValueError(f"{path}: no 'x y z' line")


def _quiet_quat_base(path: Path) -> np.ndarray | None:
    """``# quat_base w x y z`` line of a marker file (orientation in the robot base frame), or None."""
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s_ = line.split()
            if len(s_) >= 6 and s_[0] == "#" and s_[1] == "quat_base":
                return np.array([float(v) for v in s_[2:6]])
    except (OSError, ValueError):
        pass
    return None


def _quat_axis(q_wxyz: np.ndarray, col: int) -> np.ndarray:
    """Column ``col`` of the rotation matrix of a (w, x, y, z) quaternion = that local axis in the parent frame."""
    w, x, y, z = (float(v) for v in q_wxyz)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R[:, col]


def block_yaw_vs_face(tip_path: Path, purple_path: Path) -> float | None:
    """Yaw of the block's thickness axis relative to the box face normal (rad, in [-pi/2, pi/2), horizontal), from the
    ``quat_base`` lines of both marker files. None if unavailable or the block is not upright (thickness axis vertical)."""
    qb, qk = _quiet_quat_base(tip_path), _quiet_quat_base(purple_path)
    if qb is None or qk is None:
        return None
    n_box = _quat_axis(qb, 2)   # tip face normal = box local +z
    n_blk = _quat_axis(qk, 2)   # block thickness axis = block local z
    n_box[2] = 0.0
    n_blk[2] = 0.0
    if np.linalg.norm(n_box) < 0.5 or np.linalg.norm(n_blk) < 0.5:
        return None
    d = np.arctan2(n_blk[1], n_blk[0]) - np.arctan2(n_box[1], n_box[0])
    d = (d + np.pi / 2) % np.pi - np.pi / 2  # the thickness axis has no sign
    return float(d)


def first_contact_along(yaw: float) -> tuple[float, str]:
    """Purple-to-red distance along the face normal at which the block first touches the box, and what touches:
    the finger plate between the boxes (square block) or the block's leading corner (yawed block)."""
    c, s_ = float(np.cos(yaw)), abs(float(np.sin(yaw)))
    finger = (BLOCK_HALF_T + FINGER_T) * c
    corner = BLOCK_HALF_T * c + BLOCK_HALF_W * s_
    return (corner, "corner") if corner > finger else (finger, "finger")


def pair_midpoint(rcfg) -> np.ndarray:
    p = [rcfg.get_transform_frame_to_world(EE[s]).translation for s in ("left", "right")]
    return 0.5 * (p[0] + p[1])


def rot_z(p: np.ndarray, yaw: float) -> np.ndarray:
    c, s_ = np.cos(yaw), np.sin(yaw)
    return np.array([c * p[0] - s_ * p[1], s_ * p[0] + c * p[1], p[2]])


class ContactGate:
    """Tared, low-pass filtered wrist-force monitor on both hands (``stream`` stop_check for the push).

    The sim's wrist wrench is noisy even at rest (2026-09-26, holding the block: left |dF| up to 17 N raw,
    8 N after a 0.25 s low-pass), so the raw magnitude cannot be thresholded at a few newtons. Each wrist
    gets an exponential moving average over ``ema_s`` seconds and its *own* threshold
    ``max(threshold_n, noise_gain * filtered rest max + noise_margin_n)`` measured while arming (block not
    touching the box). Contact = filtered |dF| above the wrist's threshold for ``confirm_ticks``
    consecutive checks. ``travel_fn`` (optional) only feeds the log."""

    def __init__(self, threshold_n: float, confirm_ticks: int, *, sample_hz: float, tare_samples: int = 10,
                 warmup_s: float = 1.0, rest_s: float = 1.5, ema_s: float = 0.25, log_period_s: float = 0.5,
                 noise_gain: float = 1.5, noise_margin_n: float = 3.0) -> None:
        self.threshold_n = float(threshold_n)
        self.confirm_ticks = max(int(confirm_ticks), 1)
        self.sample_hz = max(float(sample_hz), 1.0)
        self.tare_samples = max(int(tare_samples), 3)
        self.warmup_s = warmup_s
        self.rest_s = rest_s
        self.ema_k = max(1.0, ema_s * self.sample_hz)  # EMA length in samples at the stream rate
        self.log_period_s = log_period_s
        self.noise_gain = noise_gain
        self.noise_margin_n = noise_margin_n
        self.sides: list[str] = []
        self.thresholds: dict[str, float] = {}
        self.filt: dict[str, np.ndarray] = {}
        self.armed = False
        self.peak_n = 0.0
        self.stop_side: str | None = None
        self.stop_force_n: float | None = None
        self.already_loaded_n = 0.0
        self.travel_fn = None
        self._above = 0
        self._last_log = 0.0

    def _filtered(self, side: str, f: np.ndarray) -> np.ndarray:
        prev = self.filt.get(side)
        cur = f.copy() if prev is None else prev + (f - prev) / self.ema_k
        self.filt[side] = cur
        return cur

    @staticmethod
    def _force(bot, side: str) -> np.ndarray | None:
        row = _get_arm_row(side, bot)
        return None if row is None else np.array([row["fx"], row["fy"], row["fz"]], dtype=float)

    def arm(self, bot) -> bool:
        """Warm the wrench subscribers up (they may be idle-paused), tare both wrists, measure the rest noise."""
        raw: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        t0 = time.time()
        while time.time() - t0 < max(self.warmup_s, 0.3):
            for side in raw:
                f = self._force(bot, side)
                if f is not None:
                    raw[side].append(f)
            time.sleep(0.02)
        self.sides = [s for s, v in raw.items() if len(v) >= 3 and len({tuple(np.round(x, 4)) for x in v}) >= 2]
        if not self.sides:
            logger.warning("No wrist wrench data on state/wrench/left|right (sim zenoh_manager must publish it); "
                           "contact detection disabled.")
            return False
        if not tare_force("both" if len(self.sides) == 2 else self.sides[0], bot, samples=self.tare_samples):
            logger.warning("Taring the wrist wrench failed; contact detection disabled.")
            return False
        self.filt = {}
        rest: dict[str, list[float]] = {s: [] for s in self.sides}
        t1 = time.time()
        while time.time() - t1 < self.rest_s:
            for side in self.sides:
                f = self._force(bot, side)
                if f is not None:
                    rest[side].append(float(np.linalg.norm(self._filtered(side, f))))
            time.sleep(1.0 / self.sample_hz)
        # Skip the EMA start-up (ema_k samples) so the filter has settled, then use the 90th percentile rather
        # than the max: right after a fast bimanual move the wrench can still show a brief settling spike, and
        # sizing the threshold off that one spike (2026-09-26: max 25 N) makes the gate ignore real, smaller
        # contact forces. already_loaded is the median of that same tail -- if it alone is already high, the
        # grasp is under load (near a joint limit or already touching) before any push happens.
        skip = int(self.ema_k) + 3
        self.thresholds = {}
        already_loaded: dict[str, float] = {}
        for side in self.sides:
            tail = rest[side][skip:] or rest[side][-1:] or [0.0]
            p90 = float(np.percentile(tail, 90))
            self.thresholds[side] = max(self.threshold_n, self.noise_gain * p90 + self.noise_margin_n)
            already_loaded[side] = float(np.median(tail))
        self.rest_p90 = {s: max(self.thresholds[s], 0.0) for s in self.sides}  # for the caller's diagnostics
        self.already_loaded_n = max(already_loaded.values(), default=0.0)
        if self.already_loaded_n > self.threshold_n:
            logger.warning(f"Wrist force is already {self.already_loaded_n:.0f} N at rest (median, no push yet) -- the grasp is under "
                           "load (near a joint limit or already touching the box) before any push motion.")
        self.armed = True
        self._above = 0
        self.peak_n = 0.0
        logger.info("Contact gate armed: filtered |dF| rest p90 "
                    + ", ".join(f"{s} {float(np.percentile((rest[s][skip:] or [0.0]), 90)):.1f} N" for s in self.sides)
                    + " -> thresholds " + ", ".join(f"{s} {self.thresholds[s]:.1f} N" for s in self.sides)
                    + f" for {self.confirm_ticks} ticks (EMA {self.ema_k:.0f} samples at {self.sample_hz:.0f} Hz).")
        return True

    def check(self, bot) -> str | None:
        """Stop message on confirmed contact, else None. Call once per stream tick."""
        if not self.armed:
            return None
        vals = {}
        for side in self.sides:
            f = self._force(bot, side)
            if f is not None:
                vals[side] = float(np.linalg.norm(self._filtered(side, f)))
        if not vals:
            return None
        side, v = max(vals.items(), key=lambda kv: kv[1] - self.thresholds.get(kv[0], self.threshold_n))
        thr = self.thresholds.get(side, self.threshold_n)
        self.peak_n = max(self.peak_n, max(vals.values()))
        now = time.time()
        if now - self._last_log >= self.log_period_s:
            self._last_log = now
            travel = f", pushed {self.travel_fn() * 100:.1f} cm" if self.travel_fn is not None else ""
            logger.info("  wrist filtered |dF| " + ", ".join(f"{s} {x:.1f}/{self.thresholds.get(s, self.threshold_n):.0f}" for s, x in vals.items())
                        + f" N (peak {self.peak_n:.1f}){travel}")
        if v >= thr:
            self._above += 1
            if self._above >= self.confirm_ticks:
                self.stop_side, self.stop_force_n = side, v
                travel = f" after {self.travel_fn() * 100:.1f} cm" if self.travel_fn is not None else ""
                return f"CONTACT on the {side} wrist: filtered |dF| = {v:.1f} N >= {thr:.1f} N{travel}"
        else:
            self._above = 0
        return None


def main(
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stop_short: float = 0.0,
    scale: float = 1.0,
    step_m: float = 0.01,
    duration: float = 6.0,
    max_age_s: float = 3.0,
    use_torso: bool = True,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    max_joint_vel: float = 0.3,
    torso_max_vel: float = TORSO_MAX_VEL,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
    standoff: float = 0.005,
    contact_dist: float | None = None,
    face_normal_world: tuple[float, float, float] = (1.0, 0.0, 0.0),
    align_base: bool = True,
    yaw_tol_deg: float = 2.0,
    yaw_cap_deg: float = 20.0,
    yaw_short_deg: float = 0.0,
    kp: float = 2.0,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    chassis_hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    turn_timeout_sim: float = 12.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
    settle_s: float = 8.0,
    max_turns: int = 2,
    push_max: float = 0.06,
    push_over: float = 0.006,
    push_speed: float = 0.015,
    push_jump_rad: float = 0.15,
    min_push: float = 0.002,
    contact_n: float = 10.0,
    confirm_ticks: int = 5,
) -> bool:
    """Turn the base a little toward red if needed, translate the bimanual grasp so the purple box centre reaches
    the standoff next to the red conveyor tip, then keep pushing toward red until the wrist F/T sensors feel the box.

    Args:
        tip_file: Red tip XYZ file (robot base). Default: the running sim's ``card_box_tip_pos.txt``.
        purple_file: Purple centre XYZ file (robot base). Default: the running sim's ``card_box_mass_center_pos.txt``.
        offset: Robot-base XYZ added to the red target (metres).
        stop_short: Stop this far before the target along the approach line (metres).
        scale: Fraction of the remaining delta to apply (1.0 = all the way).
        step_m: Waypoint spacing along the candidate paths (metres).
        duration: Nominal stream time for the executed path (seconds); stretched by the speed caps if needed.
        max_age_s: Marker files older than this are rejected (sim not running).
        use_torso: Free the torso in the IK (robot base frame). False = arms only.
        torso_posture_cost: Posture cost on the torso joints; higher = torso moves less (arms use 1e-3).
        max_joint_vel: Arm joint speed cap (rad/s) for the streamed path.
        torso_max_vel: Torso joint speed cap (rad/s).
        plan_only: Connect, plan from the measured state, report the reachable residual, move nothing.
        leave_held: Exit without ``stop()`` so the grasp stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
        standoff: Air left between the block and its first contact with the box (metres): target = red + (first-contact
            distance + standoff) along the face normal. The first-contact distance follows the block's yaw relative
            to the box (finger plate 7.75 cm when square, the leading corner when yawed); Stage 2 closes the air.
        contact_dist: Override the computed first-contact distance (metres, purple-to-red along the normal).
        face_normal_world: Outward normal of the box face in the world (+X from the scene); turned into the base
            frame with the base yaw fitted from the world-fixed markers.
        align_base: Stage 0: turn the base in place by the purple-to-red bearing difference before the arms move
            (closed loop on the live markers). False = arms + torso only.
        yaw_tol_deg: Skip the turn / stop turning when the bearing gap is below this (degrees).
        yaw_cap_deg: Never turn more than this in one run (degrees); the rest is left to the arms.
        yaw_short_deg: Stop the turn this many degrees before the bearing (0 = line purple up with red).
        kp: Turn P gain (rad/s per rad, sim time).
        wz_max: Turn yaw-rate cap (rad/s, sim time).
        wz_min: Minimum |wz| while turning (sim drive dead zone: >= 1.2 or the base does not move).
        chassis_hz: Chassis command rate (Hz).
        steer_tol_deg: Start driving once both steer wheels are within this of the in-place angles.
        steer_timeout_sim: Max sim seconds to wait for the steering.
        turn_timeout_sim: Give up on the turn after this many sim seconds.
        stall_s: Raise the wheel command when the heading has not changed for this long (seconds).
        wheel_gain: Wheel command multiplier (sim: cancels the chassis_drive_vel scale). Real robot: 1.0.
        settle_s: After each turn wait (at most this long) until the bearing stops drifting -- the sim base unwinds
            a degree or two after the wheels stop -- before re-reading the markers.
        max_turns: Turn again (aiming past the target by the miss just measured) while the settled bearing gap is
            still above ``yaw_tol_deg``, up to this many turns in total.
        push_max: Stage 2: push the grasp horizontally toward red by at most this (metres) after reaching the standoff,
            stopping on wrist contact. 0 = no push. Without wrench data only the nominal air gap is closed.
        push_over: Push length = live gap to the jaw-contact distance + this (metres), capped by ``push_max``.
        push_speed: Push speed (m/s) along the push direction.
        push_jump_rad: Cut the push path at the first 5 mm step that needs a joint change above this (rad): beyond it
            the IK swings the arms instead of translating the block. Pushing toward the box is sideways for the arms
            (the right shoulder j2 sits at its limit with the hands at the centreline), so legitimate steps there cost
            ~0.1 rad; the joint speed caps keep them slow. 0.35 = the planner's own limit.
        min_push: If the reachable push (after the joint-jump cut) is below this, skip the F/T push entirely and
            report the pose as already at/near the box instead of arming the gate on a load that is not a real push.
        contact_n: Minimum contact threshold on the filtered, tared wrist force (N); raised automatically to
            1.5 x the measured rest noise + 3 N per wrist.
        confirm_ticks: Consecutive stream ticks (50 Hz) above threshold that count as contact.
    """
    sim_dir = find_sim_dir()
    tip_path = Path(tip_file) if tip_file is not None else sim_dir / TIP_FILE_NAME
    purple_path = Path(purple_file) if purple_file is not None else sim_dir / PURPLE_FILE_NAME
    try:
        red = load_xyz_base(tip_path, "red tip", max_age_s)
        purple = load_xyz_base(purple_path, "purple centre", max_age_s)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return False

    def read_pair(retries: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Red and purple (robot base); the sim rewrites the files every step, so an empty read is retried."""
        for i in range(retries):
            try:
                return load_xyz_base(tip_path, "red tip", max_age_s), load_xyz_base(purple_path, "purple centre", max_age_s)
            except (ValueError, OSError):
                if i == retries - 1:
                    raise
                time.sleep(0.05)
        raise RuntimeError("unreachable")

    def base_yaw_deg() -> float | None:
        """Base yaw in the world (deg) from the world-fixed markers, for drift diagnostics; None if unavailable."""
        try:
            yaw, _span = base_yaw_world(world_fixed_pairs(sim_dir, max_age_s, include_purple=False))
            return float(np.rad2deg(yaw))
        except (ValueError, FileNotFoundError, OSError):
            return None

    def face_normal() -> np.ndarray:
        """Unit box face normal in the current base frame (horizontal): from the box quaternion in the tip file, else
        from the world normal turned by the fitted base yaw; zero vector if neither is available."""
        qb = _quiet_quat_base(tip_path)
        if qb is not None:
            n_b = _quat_axis(qb, 2)
            n_b[2] = 0.0
            if np.linalg.norm(n_b) > 0.5:
                return n_b / np.linalg.norm(n_b)
        try:
            n_b = np.asarray(world_dir_to_base(sim_dir, np.asarray(face_normal_world, dtype=float), max_age_s), dtype=float)
            n_b[2] = 0.0
            return n_b / np.linalg.norm(n_b) if np.linalg.norm(n_b) > 1e-6 else np.zeros(3)
        except (FileNotFoundError, ValueError, OSError):
            return np.zeros(3)

    def contact_geom(yaw: float | None) -> tuple[float, str]:
        """(first-contact distance along the normal, what touches) for a block yaw; the override wins."""
        if contact_dist is not None:
            return float(contact_dist), "override"
        return first_contact_along(0.0 if yaw is None else yaw)

    geom = {"yaw": block_yaw_vs_face(tip_path, purple_path)}
    geom["contact"], geom["what"] = contact_geom(geom["yaw"])
    geom["total"] = geom["contact"] + max(float(standoff), 0.0)  # red -> target along the normal

    def refresh_geom() -> None:
        geom["yaw"] = block_yaw_vs_face(tip_path, purple_path)
        geom["contact"], geom["what"] = contact_geom(geom["yaw"])
        geom["total"] = geom["contact"] + max(float(standoff), 0.0)

    def geom_txt() -> str:
        yaw_txt = f"{np.rad2deg(geom['yaw']):+.1f} deg" if geom["yaw"] is not None else "unknown (block not upright?)"
        return (f"block yaw vs the box face {yaw_txt} -> first contact by the {geom['what']} at {geom['contact'] * 100:.2f} cm "
                f"(purple to red along the normal) + {standoff * 100:.1f} cm air = {geom['total'] * 100:.2f} cm")

    def make_target(red_b: np.ndarray, purple_b: np.ndarray, n_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tgt = red_b + np.asarray(offset, dtype=float) + geom["total"] * n_b
        full_ = tgt - purple_b
        dist_ = float(np.linalg.norm(full_))
        if stop_short > 0.0 and dist_ > 1e-6:
            tgt = tgt - min(stop_short, 0.95 * dist_) * full_ / dist_
        tgt = purple_b + float(scale) * (tgt - purple_b)
        return tgt, tgt - purple_b

    n_face = face_normal()
    if not np.any(n_face):
        logger.warning("Face normal unavailable from the markers; the target is the red marker itself (inside the box!).")
    else:
        logger.info("Geometry: " + geom_txt())
    target, delta = make_target(red, purple, n_face)
    if float(np.linalg.norm(delta)) < 1e-4:
        logger.info("Purple already on target (within 0.1 mm); nothing to do.")
        return True
    logger.info(
        "Purple -> red (bimanual rigid translate, robot base):\n"
        f"  purple  {np.round(purple, 4)}\n"
        f"  red     {np.round(red, 4)}"
        + (f"  +offset {np.round(offset, 3)}" if np.any(np.asarray(offset)) else "")
        + (f"  +{geom['total'] * 100:.2f} cm along the face normal {np.round(n_face, 3)}" if np.any(n_face) else "")
        + (f"  stop_short {stop_short:.3f}" if stop_short > 0 else "")
        + (f"  scale {scale:g}" if scale != 1.0 else "")
        + f"\n  target  {np.round(target, 4)}\n"
        f"  delta   {np.round(delta, 4)}  |d| = {np.linalg.norm(delta):.4f} m"
    )

    # Stage 0: the torso joints are all pitch, so a sideways (robot y) gap can only be closed by the arms, which
    # cannot shift a two-handed grasp far to one side. If red sits off the purple's bearing, turn the base in
    # place by that bearing difference first (capped), so the remaining delta is forward/down.
    def bearing_gap(red_b: np.ndarray, purple_b: np.ndarray, n_b: np.ndarray) -> float:
        """Yaw (rad, +CCW) that puts the target (red + offset + contact + air along the face normal) on the purple's bearing."""
        tgt = red_b + np.asarray(offset, dtype=float) + geom["total"] * n_b
        return float(wrap(np.arctan2(tgt[1], tgt[0]) - np.arctan2(purple_b[1], purple_b[0])))

    yaw_needed = bearing_gap(red, purple, n_face)
    yaw_goal = float(np.clip(yaw_needed, -np.deg2rad(yaw_cap_deg), np.deg2rad(yaw_cap_deg)))
    yaw_goal = float(np.sign(yaw_goal) * max(abs(yaw_goal) - np.deg2rad(yaw_short_deg), 0.0))
    turn_base = bool(align_base and abs(yaw_goal) >= np.deg2rad(yaw_tol_deg))
    if turn_base:
        logger.info(
            f"Stage 0: red is {delta[1] * 100:+.1f} cm sideways (robot y) of purple, bearing gap {np.rad2deg(yaw_needed):+.1f} deg -> "
            f"the base turns {np.rad2deg(yaw_goal):+.1f} deg ({'CCW / left' if yaw_goal > 0 else 'CW / right'}) in place first"
            + (f" (capped at {yaw_cap_deg:.0f} deg)" if abs(yaw_needed) > abs(yaw_goal) + 1e-6 else "") + "; the arms take the rest."
        )
        if plan_only:
            red_pred = rot_z(red, -yaw_goal)
            if geom["yaw"] is not None:  # the block turns with the base, the box does not
                geom["yaw"] = float((geom["yaw"] + yaw_goal + np.pi / 2) % np.pi - np.pi / 2)
                geom["contact"], geom["what"] = contact_geom(geom["yaw"])
                geom["total"] = geom["contact"] + max(float(standoff), 0.0)
            target, delta = make_target(red_pred, purple, rot_z(n_face, -yaw_goal))
            logger.info(f"Predicted after the turn: red {np.round(red_pred, 4)}, delta {np.round(delta, 4)} |d| = {np.linalg.norm(delta):.4f} m; "
                        + geom_txt() + " (the arm plan below uses this).")
    elif abs(delta[1]) > 0.05 or abs(yaw_needed) > np.deg2rad(5.0):
        logger.warning(
            f"Red is {delta[1] * 100:+.1f} cm sideways (robot y) from purple: the base would have to yaw "
            f"{np.rad2deg(yaw_needed):+.1f} deg to line them up. The arms alone rarely cover more than a few cm of that; "
            "consider --align-base (or 8.rotate_base_to_conveyor.py) first."
        )

    components = ["left_arm", "right_arm"] + (["torso"] if use_torso else [])
    if not plan_only:
        foreign = detect_foreign_commands(components)
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

    configuration, _tasks, solver, idx = build_ik()
    bot = connect()
    do_shutdown = not leave_held
    try:
        if turn_base and not plan_only:
            leftover = float(wrap(yaw_needed - yaw_goal))  # bearing gap that should remain after the turn
            comp = {"aim": 0.0}  # extra yaw to aim past the target (rad): the miss measured on the previous turn

            def live_gap() -> float:
                """Bearing gap from the live markers without logging (called at the chassis loop rate)."""
                return bearing_gap(_quiet_xyz(tip_path), _quiet_xyz(purple_path), face_normal())

            def bearing_error() -> tuple[float, str]:
                gap = live_gap()
                return float(wrap(gap - leftover + comp["aim"])), f"bearing gap {np.rad2deg(gap):+.1f} deg"

            def settle(t_max: float, window_s: float = 1.5, still_deg: float = 0.12) -> tuple[float, float]:
                """Wait until the base stops unwinding after the wheels stop. Returns the bearing gap (rad) right
                after the stop and after settling (still = range below ``still_deg`` over ``window_s``)."""
                t0_ = time.monotonic()
                hist: list[tuple[float, float]] = []
                g0 = None
                while True:
                    try:
                        g = live_gap()
                    except (ValueError, OSError):
                        time.sleep(0.05)
                        continue
                    now = time.monotonic()
                    if g0 is None:
                        g0 = g
                    hist.append((now, g))
                    recent = [h[1] for h in hist if now - h[0] <= window_s]
                    spread = max(recent) - min(recent) if recent else float("inf")
                    if now - t0_ >= window_s + 0.5 and len(recent) >= 3 and spread < np.deg2rad(still_deg):
                        return g0, g
                    if now - t0_ > t_max:
                        logger.warning(f"Base still drifting after {t_max:.0f}s (bearing gap moved {np.rad2deg(spread):.2f} deg "
                                       f"in the last {window_s:.0f}s); continuing anyway.")
                        return g0, g
                    time.sleep(0.1)

            logger.info("Stage 0: turning the base in place (arms still; measuring sim speed for 2 s)...")
            rtf = measure_rtf(tip_path)
            for attempt in range(1, max(int(max_turns), 1) + 1):
                turn_in_place(
                    bot.chassis, bearing_error, tol_deg=yaw_tol_deg, kp=kp, wz_max=wz_max, wz_min=wz_min, hz=chassis_hz,
                    steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim, timeout_sim=turn_timeout_sim,
                    stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                )
                t_settle = time.monotonic()
                g_stop, g_settled = settle(settle_s)
                unwound = float(wrap(g_settled - g_stop))
                remaining = float(wrap(g_settled - leftover))
                logger.info(
                    f"Stage 0 turn {attempt}: bearing gap {np.rad2deg(g_stop):+.1f} deg at the wheel stop, {np.rad2deg(g_settled):+.1f} deg "
                    f"after settling {time.monotonic() - t_settle:.1f}s (base unwound {np.rad2deg(unwound):+.1f} deg); "
                    f"remaining {np.rad2deg(remaining):+.1f} deg (tol {yaw_tol_deg:.1f})"
                )
                if abs(remaining) < np.deg2rad(yaw_tol_deg):
                    break
                if attempt >= max(int(max_turns), 1):
                    logger.warning(f"Bearing gap still {np.rad2deg(remaining):+.1f} deg after {attempt} turn(s); the arms take the rest.")
                    break
                # The controller stops on the first tick inside the tolerance (residual ~tol on the approach side) and the
                # base then unwinds by about what it just did: aim past the target by both so the next stop lands on it.
                s_dir = 1.0 if remaining > 0 else -1.0
                unwind_comp = unwound if abs(unwound) < np.deg2rad(6.0) else 0.0
                comp["aim"] = float(np.clip(s_dir * np.deg2rad(max(yaw_tol_deg - 0.3, 0.0)) + unwind_comp, -np.deg2rad(6.0), np.deg2rad(6.0)))
                logger.info(f"Stage 0: turning again, aiming {np.rad2deg(comp['aim']):+.1f} deg past the target "
                            f"(tolerance undershoot {s_dir * max(yaw_tol_deg - 0.3, 0.0):+.1f} + expected unwinding {np.rad2deg(unwind_comp):+.1f})...")
            red, purple = read_pair()
            n_face = face_normal()
            refresh_geom()
            target, delta = make_target(red, purple, n_face)
            logger.info(f"After the base turn: purple {np.round(purple, 4)}, red {np.round(red, 4)}, delta {np.round(delta, 4)} "
                        f"|d| = {np.linalg.norm(delta):.4f} m (sideways {delta[1] * 100:+.1f} cm); " + geom_txt())
        sync_from_robot(bot, configuration, idx)
        rcfg, tasks, ridx = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        T0 = {s: rcfg.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
        mid0 = pair_midpoint(rcfg)
        q0 = rcfg.q.copy()

        best = None  # (residual, label, q_start, knots, n)
        for label, waypoints in candidate_offset_paths(delta):
            dense = densify(waypoints, step_m)
            if not dense:
                continue
            n = len(dense)

            def pose_at(a: float, _dense=dense, _n=n):
                off = _dense[min(max(int(round(a * _n)) - 1, 0), _n - 1)]
                return {s: pin.SE3(T0[s].rotation.copy(), T0[s].translation + off) for s in T0}

            rcfg.update(q0)
            logger.info(f"Trying approach path '{label}' ({n} steps)...")
            planned = plan_dual_pose_path(rcfg, tasks, solver, pose_at, n)
            if planned is None:
                logger.warning(f"  '{label}': no progress")
                continue
            q_start, knots = planned
            rcfg.update(knots[-1])
            achieved = pair_midpoint(rcfg) - mid0
            rcfg.update(q0)
            residual = float(np.linalg.norm(delta - achieved))
            logger.info(f"  '{label}': {len(knots)}/{n} steps, residual purple->target {residual:.4f} m")
            if best is None or residual < best[0] - 1e-4:
                best = (residual, label, q_start, knots, n)
                if residual <= CLOSE_ENOUGH_M:
                    break
        if best is None:
            logger.error("No progress toward the red tip on any path. Not moving.")
            do_shutdown = True
            return False
        residual, label, q_start, knots, n = best
        Q = np.array([q_start] + knots)
        rcfg.update(Q[-1])
        predicted = purple + (pair_midpoint(rcfg) - mid0)
        rcfg.update(q0)
        covered = 100.0 * max(0.0, 1.0 - residual / max(float(np.linalg.norm(delta)), 1e-6))
        verdict = "REACHABLE" if residual <= CLOSE_ENOUGH_M else "PARTIAL"
        torso_note = ""
        if use_torso:
            torso_note = f"; torso change (j1, j2, j3) {np.round(np.rad2deg(Q[-1, ridx['torso']] - Q[0, ridx['torso']]), 1)} deg"
        logger.info(
            f"[{verdict}] path '{label}': {len(knots)}/{n} steps; predicted purple after motion "
            f"{np.round(predicted, 4)}, residual to target {residual:.4f} m ({covered:.0f}% of the gap){torso_note}"
        )
        if plan_only:
            if push_max > 0.0:
                logger.info(f"Stage 2 (not planned here): after the move, push toward red by the live gap to the first contact "
                            f"({geom['contact'] * 100:.2f} cm, {geom['what']}) + {push_over * 1000:.0f} mm margin (cap {push_max * 100:.0f} cm) at {push_speed * 100:.1f} cm/s, "
                            f"stopping when a tared wrist force exceeds max({contact_n:.0f} N, 1.5 x rest noise + 3 N).")
            logger.info("Plan only: nothing commanded.")
            return residual <= CLOSE_ENOUGH_M

        path = {g: float(np.abs(np.diff(Q[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx.items()}
        t_stream = max(duration, max(path[g] / (torso_max_vel if g == "torso" else max_joint_vel) for g in path))
        if t_stream > duration:
            logger.info(f"Stream time stretched to {t_stream:.1f}s by the joint speed caps "
                        f"(arms {max_joint_vel:.2f}, torso {torso_max_vel:.2f} rad/s)")
        groups = {COMPONENT[g]: Q[:, ii] for g, ii in ridx.items()}
        yaw_before = base_yaw_deg()
        logger.info(f"Executing '{label}' ({', '.join(groups)}) over {t_stream:.1f}s...")
        stream(bot, groups, t_stream, config.CONTROL_DT)

        sync_from_robot(bot, configuration, idx)
        rcfg2, _t2, _i2 = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
        purple_est = purple + (pair_midpoint(rcfg2) - mid0)
        err = float(np.linalg.norm(purple_est - target))
        yaw_after = base_yaw_deg()
        drift_txt = (f" Base yaw {yaw_before:+.1f} -> {yaw_after:+.1f} deg during the move ({wrap(np.deg2rad(yaw_after - yaw_before)) * 180 / np.pi:+.1f})."
                     if yaw_before is not None and yaw_after is not None else "")
        logger.info(
            f"Done. purple (FK estimate) {np.round(purple_est, 4)}, target {np.round(target, 4)}, "
            f"err {err:.4f} m. Check {purple_path.name} for the sim-measured value. Pose left held.{drift_txt}"
        )

        # Stage 2: keep pushing toward red until the wrists feel the box (2026-09-26). Direction = horizontal
        # purple -> red from the live markers (so a residual sideways error is closed on the way), else -normal.
        if push_max > 0.0:
            red_now, purple_now = read_pair()
            n_now = face_normal()
            push_dir = red_now - purple_now
            push_dir[2] = 0.0
            if float(np.linalg.norm(push_dir)) < 1e-3:
                push_dir = -n_now
            along = float((purple_now - red_now) @ n_now) if np.any(n_now) else float("nan")
            refresh_geom()
            contact_now = geom["contact"]
            # push as far as the nominal first contact plus a margin (the grasp depth is not exact); F/T is the real stop
            push_len = min(float(push_max), max(along - contact_now, 0.0) + float(push_over)) if np.isfinite(along) else float(push_max)
            if float(np.linalg.norm(push_dir)) < 1e-6:
                logger.warning("Stage 2 skipped: no push direction (markers and face normal unavailable).")
                push_len = 0.0
            else:
                push_dir = push_dir / float(np.linalg.norm(push_dir))
                logger.info(f"Stage 2: aiming to push toward red along {np.round(push_dir, 3)} (robot base), up to {push_len * 100:.1f} cm; "
                            f"purple is {along * 100:.2f} cm from red along the face normal; " + geom_txt() + ".")
            # Plan and cut the path to what the arms can actually reach *before* touching the F/T gate: arming the
            # gate on a load that is really the arms straining at a joint limit (not contact) reads as noise, not a
            # push (2026-09-26 20:26: rest force already 25 N, one stream tick to "contact", 0.0 cm moved).
            reach3, knots3, q_s3, n3, ridx3 = 0.0, [], None, 0, {}
            if push_len > 1e-4:
                sync_from_robot(bot, configuration, idx)
                rcfg3, tasks3, ridx3 = build_dual_arm_ik(configuration, use_torso, torso_posture_cost)
                T3 = {s: rcfg3.get_transform_frame_to_world(EE[s]) for s in ("left", "right")}
                mid3 = pair_midpoint(rcfg3)
                q3 = rcfg3.q.copy()
                push_step = min(step_m, 0.005)
                dense3 = densify([push_len * push_dir], push_step)
                n3 = len(dense3)

                def pose_push(a: float, _dense=dense3, _n=n3):
                    off = _dense[min(max(int(round(a * _n)) - 1, 0), _n - 1)]
                    return {s: pin.SE3(T3[s].rotation.copy(), T3[s].translation + off) for s in T3}

                planned3 = plan_dual_pose_path(rcfg3, tasks3, solver, pose_push, n3)
                if planned3 is None:
                    logger.warning("Stage 2: the arms cannot push any further from here (IK / joint limits); pose held.")
                else:
                    # keep only the prefix where the IK still *translates* the block: a step that needs a big joint
                    # change (or makes no progress) means the arms have run out of sideways room and would swing
                    q_s3, knots_raw = planned3
                    q_prev, reach3 = q_s3, 0.0
                    for k in knots_raw:
                        jump = float(np.max(np.abs(k - q_prev)))
                        rcfg3.update(k)
                        prog = float((pair_midpoint(rcfg3) - mid3) @ push_dir)
                        if jump > push_jump_rad or prog < reach3 + 0.3 * push_step:
                            logger.info(f"Stage 2: push path cut at {len(knots3)}/{len(knots_raw)} steps (next step needs a joint change of "
                                        f"{jump:.2f} rad for {(prog - reach3) * 1000:.1f} mm) -- the arms are out of sideways room there.")
                            break
                        knots3.append(k)
                        q_prev, reach3 = k, prog
                rcfg3.update(q3)
            needed = max(along - contact_now, 0.0) if np.isfinite(along) else float("nan")
            if reach3 < min_push:
                logger.info(f"Stage 2: only {reach3 * 1000:.1f} mm of push room left after the joint-limit cut (< {min_push * 1000:.0f} mm) -- "
                            "the arms have no more sideways travel here; not arming the F/T gate on that. Treating the current "
                            f"pose as the end of the approach (purple {along * 100:.2f} cm from red, {(along - contact_now) * 1000:+.0f} mm before the "
                            f"{geom['what']} touches; block face to box face {(along - BLOCK_HALF_T) * 100:.1f} cm). Pose held.")
            else:
                if np.isfinite(needed) and reach3 < needed:
                    logger.warning(f"Stage 2: the arms can push {reach3 * 1000:.1f} mm cleanly but the nominal gap to the jaw contact is "
                                   f"{needed * 1000:.1f} mm; pushing what is reachable (F/T decides the real stop).")
                gate = ContactGate(contact_n, confirm_ticks, sample_hz=1.0 / config.CONTROL_DT)
                logger.info("Arming the wrist contact gate...")
                if not gate.arm(bot):
                    logger.warning("No contact detection available; pushing the reachable distance without an F/T stop.")
                Q3 = np.array([q_s3] + knots3)
                path3 = {g: float(np.abs(np.diff(Q3[:, ii], axis=0)).sum(axis=0).max()) for g, ii in ridx3.items()}
                t_push = max(reach3 / max(push_speed, 1e-4), 1.0,
                             max(path3[g] / (torso_max_vel if g == "torso" else max_joint_vel) for g in path3))
                groups3 = {COMPONENT[g]: Q3[:, ii] for g, ii in ridx3.items()}
                p_left0 = configuration.get_transform_frame_to_world(EE["left"]).translation.copy()

                def pushed_m() -> float:
                    sync_from_robot(bot, configuration, idx)
                    return float((configuration.get_transform_frame_to_world(EE["left"]).translation - p_left0) @ push_dir)

                gate.travel_fn = pushed_m

                def stop_on_contact(b) -> bool:
                    msg = gate.check(b)
                    if msg:
                        logger.info(msg)
                        return True
                    return False

                logger.info(f"Stage 2: pushing {reach3 * 100:.1f} cm ({len(knots3)}/{n3} steps) over {t_push:.1f}s"
                            + (" with wrist contact detection..." if gate.armed else " (geometry only)..."))
                _final3, stopped3 = stream(bot, groups3, t_push, config.CONTROL_DT, stop_check=stop_on_contact if gate.armed else None)
                time.sleep(0.3)
                try:
                    red_now, purple_now = read_pair()
                    along = float((purple_now - red_now) @ n_now) if np.any(n_now) else float("nan")
                except (ValueError, OSError):
                    along = float("nan")
                gap_txt = (f"purple {along * 100:.2f} cm from red along the face normal -> {(along - contact_now) * 1000:+.0f} mm before the "
                           f"{geom['what']} touches (block face to box face {(along - BLOCK_HALF_T) * 100:.1f} cm)" if np.isfinite(along) else "marker gap unavailable")
                early_txt = " (this early -- likely the grasp straining near a joint limit, not real contact; consider raising --contact-n)"                     if stopped3 and pushed_m() < 0.3 * reach3 and gate.already_loaded_n > gate.threshold_n else ""
                logger.info(("Stage 2: CONTACT -- stopped" if stopped3 else "Stage 2: no contact within the push range")
                            + f" after {pushed_m() * 100:.1f} cm of push{early_txt}; {gap_txt}. Pose held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return err <= CLOSE_ENOUGH_M
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
