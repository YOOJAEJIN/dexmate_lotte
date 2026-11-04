# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Closed-loop in-place yaw: turn the base to face the conveyor (AprilTag side).

``lotte_chilsung_2/4.rotate_base_z.py`` turns open-loop for ``angle / speed``
seconds, so on a simulator running below real time the base stops short. This
script instead reads the pose the simulator writes every step and keeps turning
until the heading error is below a tolerance, then stops.

Feedback (robot base frame, from ``unitree_sim_isaaclab*/*.txt`` written by sim_main every step; the
folder is ``$LOTTE_SIM_DIR`` if set, else the ``unitree_sim_isaaclab*`` folder whose
``card_box_tip_pos.txt`` is newest -- e.g. ``unitree_sim_isaaclab_260918``):

* ``apriltag_pose.txt`` (preferred): tag position + orientation in base frame.
  - ``--aim conveyor`` (default): face the conveyor squarely, i.e. base +X along
    the horizontal projection of the tag normal reversed (tag faces the robot).
  - ``--aim tag``: point base +X at the tag position (tag centred in the camera).
* Fallback when the tag file is missing (``--aim conveyor`` only): red tip +
  purple centre files give the base yaw in world; target world yaw is
  ``--conveyor-yaw-deg`` (90 = facing +Y, toward the conveyor's -Y face).
* ``--extra-deg`` (default 0 = plain square-on): turn that much further left past square-on, toward the
  conveyor box, so the held block ends closer to the box face for the arm/torso approach (10).

Chassis command: the two steer wheels are set to the in-place-rotation angles
(about +/-44 deg, from the chassis geometry) with ``chassis.set_motion_state``
and the wheel speeds carry the yaw rate (wz > 0 = counter-clockwise). Before
driving, the script waits until the *measured* steering reaches those angles;
dexcontrol's own steer-first pause is effectively zero, so ``set_velocity``
starts driving on straight wheels and the base only creeps.

``--probe`` runs a short diagnostic (steer only, then a brief drive) and prints
steering / wheel-speed / heading telemetry, to tell "steering does not reach
the target" from "wheels spin but the base does not follow".

Arms / torso / head keep their last targets (leave-held) on exit.

Example::

    python 4.rotate_base_to_conveyor.py --dry-run
    python 4.rotate_base_to_conveyor.py --probe
    python 4.rotate_base_to_conveyor.py
    python 4.rotate_base_to_conveyor.py --aim tag --tol-deg 1
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _pink_common import connect, disconnect_leave_held  # noqa: E402

_DEX_ROOT = Path(__file__).resolve().parents[2]
_SIM_FALLBACK = "unitree_sim_isaaclab_260918"


def _find_sim_dir() -> Path:
    """Simulator folder: ``$LOTTE_SIM_DIR`` if set, else the ``unitree_sim_isaaclab*`` folder under
    ``_DEX_ROOT`` whose ``card_box_tip_pos.txt`` was rewritten most recently (the running sim
    rewrites it every step), else ``_SIM_FALLBACK``."""
    env = os.environ.get("LOTTE_SIM_DIR")
    if env:
        return Path(env).expanduser()
    with_file = [d for d in _DEX_ROOT.glob("unitree_sim_isaaclab*") if (d / "card_box_tip_pos.txt").is_file()]
    if with_file:
        return max(with_file, key=lambda d: (d / "card_box_tip_pos.txt").stat().st_mtime)
    return _DEX_ROOT / _SIM_FALLBACK


_SIM = _find_sim_dir()
TAG_FILE = _SIM / "apriltag_pose.txt"
TIP_FILE = _SIM / "card_box_tip_pos.txt"
PURPLE_FILE = _SIM / "card_box_mass_center_pos.txt"

# vega_1p chassis geometry (dexbot_utils vega_1/chassis.py); used only if the
# chassis object does not expose them.
_AXIS_DIST = 0.219   # centre -> steer-wheel axle (m)
_HALF_WHEELS = 0.225  # half wheel track (m)
_WHEEL_RADIUS = 0.0864  # drive wheel radius (m), from the sim USD visual (0.173 m dia)
# The sim's chassis_drive_vel action (env cfg) applies scale=100 to the wheel
# "m/s" dexcontrol sends and uses the result as a joint velocity in rad/s, so a
# command of v m/s spins the wheel at 100*v rad/s = 100*v*r m/s of surface speed
# (8.6x too fast -> wheels slip). Multiplying our command by 1/(100*r) restores
# the physical mapping. On the real robot use --wheel-gain 1.
_SIM_DRIVE_SCALE = 100.0
_WHEEL_GAIN_SIM = 1.0 / (_SIM_DRIVE_SCALE * _WHEEL_RADIUS)
# The sim wheel drive is a pure velocity drive with damping 10 (robots/dexmate.py):
# torque = 10 * (target - actual) rad/s. Below ~3-4 rad/s of target the torque
# cannot move the 88 kg base at all (observed: target 1 rad/s -> wheels idle),
# so the yaw-rate floor must keep the joint target above that. Joint velocity
# limit is 10 rad/s.
_SIM_JOINT_VEL_LIMIT = 10.0
# One sim_main control step (sim.dt 0.005 * decimation 2) advances sim time by:
_SIM_STEP_S = 0.01
LOG_DIR = Path(__file__).resolve().parent / "logs"


def _setup_file_log(tag: str) -> Path:
    """Mirror the console log into logs/rotate_<tag>_<time>.log (plain lines)."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"rotate_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(str(path), level="INFO", format="{time:HH:mm:ss.SSS} | {level:7} | {message}", enqueue=False)
    return path


def _file_update_rate(path: Path, window_s: float = 2.0) -> float:
    """How often the sim rewrites ``path`` (Hz) — a proxy for the sim step rate."""
    t_end = time.monotonic() + window_s
    last = None
    n = 0
    while time.monotonic() < t_end:
        try:
            m = path.stat().st_mtime_ns
        except OSError:
            m = None
        if m is not None and m != last:
            n += 1
            last = m
        time.sleep(0.005)
    return n / window_s


def measure_rtf(path: Path = TAG_FILE, sim_step_s: float = _SIM_STEP_S) -> float:
    """Real-time factor of the simulator: sim seconds per wall second, from the
    status-file rewrite rate (one rewrite per control step). Clipped to [0.05, 1]."""
    rate = _file_update_rate(path)
    if rate <= 0.0:
        logger.warning("Status file is not being rewritten; assuming real time (RTF 1).")
        return 1.0
    rtf = float(np.clip(rate * sim_step_s, 0.05, 1.0))
    logger.info(f"Sim status updates {rate:.0f} Hz x {sim_step_s:.3f} s/step -> RTF ~{rtf:.2f} "
                f"(1 wall s = {rtf:.2f} sim s)")
    return rtf


# ---------------------------------------------------------------------------
# Sim status files
# ---------------------------------------------------------------------------
def _nums(tokens: list[str]) -> list[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            break
    return out


def read_tag_pose(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(pos_b, quat_b wxyz, pos_w, quat_w wxyz) from apriltag_pose.txt."""
    base = world = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "#":
            if len(s) >= 9 and s[1] == "world":
                world = _nums(s[2:9])
        else:
            base = _nums(s[:7])
    if base is None or len(base) < 7 or world is None or len(world) < 7:
        raise ValueError("incomplete tag pose file")
    b, w = np.asarray(base), np.asarray(world)
    return b[:3], b[3:7], w[:3], w[3:7]


def read_point_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(p_base, p_world) from card_box_*_pos.txt."""
    base = world = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "#":
            if len(s) >= 5 and s[1] == "world":
                world = _nums(s[2:5])
        else:
            base = _nums(s[:3])
    if base is None or len(base) < 3 or world is None or len(world) < 3:
        raise ValueError("incomplete marker file")
    return np.asarray(base), np.asarray(world)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rot(q_wxyz: np.ndarray) -> np.ndarray:
    return R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def _fresh(path: Path, max_age_s: float) -> bool:
    """True if the sim rewrote ``path`` within ``max_age_s`` (it writes every step)."""
    try:
        return (time.time() - path.stat().st_mtime) <= max_age_s
    except OSError:
        return False


def heading_error(
    aim: str,
    conveyor_yaw_deg: float,
    extra_deg: float = 0.0,
    tag_file: Path = TAG_FILE,
    tip_file: Path = TIP_FILE,
    purple_file: Path = PURPLE_FILE,
    max_age_s: float = 3.0,
) -> tuple[float, str]:
    """Signed yaw the base must turn (rad, +CCW) and a short description.

    ``extra_deg`` is added to the square-on heading: positive = turn that much further counter-clockwise
    (to the left, toward the conveyor box with the red tip), so the held block ends up closer to the box
    face and the arm/torso approach (10) has less sideways gap to close.

    Only files rewritten within ``max_age_s`` are trusted, so a stale file from
    an earlier simulator session cannot drive the loop.
    """
    extra = np.deg2rad(extra_deg)
    if tag_file.is_file() and not _fresh(tag_file, max_age_s):
        logger.warning(f"{tag_file.name} is older than {max_age_s:.0f}s (sim not writing?); ignoring it.")
    if tag_file.is_file() and _fresh(tag_file, max_age_s):
        p_b, q_b, p_w, q_w = read_tag_pose(tag_file)
        R_b_tag, R_w_tag = _rot(q_b), _rot(q_w)
        R_wb = R_w_tag @ R_b_tag.T
        yaw_wb = float(np.arctan2(R_wb[1, 0], R_wb[0, 0]))
        robot_w = p_w - R_wb @ p_b
        if aim == "tag":
            err = float(np.arctan2(p_b[1], p_b[0]))
        else:
            n_b = R_b_tag @ np.array([0.0, 0.0, 1.0])  # tag normal, points at the robot
            err = float(np.arctan2(-n_b[1], -n_b[0]))
        err = _wrap(err + extra)
        info = (f"tag@base ({p_b[0]:+.2f}, {p_b[1]:+.2f}) m, base yaw(world) {np.rad2deg(yaw_wb):+.1f} deg, "
                f"robot@world ({robot_w[0]:+.2f}, {robot_w[1]:+.2f})"
                + (f", extra {extra_deg:+.1f} deg past square-on" if extra_deg else ""))
        return err, info
    if aim == "tag":
        raise FileNotFoundError(f"{tag_file} missing or stale; --aim tag needs the live AprilTag pose file.")
    for f in (tip_file, purple_file):
        if not _fresh(f, max_age_s):
            raise FileNotFoundError(f"{f} missing or older than {max_age_s:.0f}s; is sim_main.py running?")
    a_b, a_w = read_point_pair(tip_file)
    b_b, b_w = read_point_pair(purple_file)
    db, dw = b_b - a_b, b_w - a_w
    if np.linalg.norm(db[:2]) < 0.02:
        raise ValueError("red/purple markers too close in XY to derive the base yaw.")
    yaw_wb = _wrap(float(np.arctan2(dw[1], dw[0]) - np.arctan2(db[1], db[0])))
    err = _wrap(np.deg2rad(conveyor_yaw_deg) + extra - yaw_wb)
    return err, (f"(red/purple fallback) base yaw(world) {np.rad2deg(yaw_wb):+.1f} deg, target {conveyor_yaw_deg:+.1f}"
                 + (f" {extra_deg:+.1f} extra" if extra_deg else ""))


# ---------------------------------------------------------------------------
# Chassis: in-place rotation command
# ---------------------------------------------------------------------------
class InplaceTurn:
    """Steering angles + wheel speeds for a pure yaw about the base centre.

    Same geometry as ``Chassis.set_velocity(0, 0, wz)``: steer wheels at
    +/-atan2(axle_dist, half_track) (~44 deg for vega_1p), wheel speed
    ``wz * centre_to_wheel_dist`` with the left wheel reversed for CCW.
    """

    def __init__(self, chassis):
        axis = float(getattr(chassis, "_center_to_wheel_axis_dist", _AXIS_DIST))
        half = float(getattr(chassis, "_half_wheels_dist", _HALF_WHEELS))
        self.d = float(np.hypot(axis, half))
        a = float(np.arctan2(axis, half))
        self.steer = np.array([+a, -a])
        # Let dexcontrol's own solver pick the wheel-speed signs when available
        # (guards against a convention change); fall back to the derivation above.
        self._sign = np.array([-1.0, +1.0])
        cwc = getattr(chassis, "_compute_wheel_control", None)
        lvec = getattr(chassis, "_left_wheel_ang_vel_vector", None)
        rvec = getattr(chassis, "_right_wheel_ang_vel_vector", None)
        if cwc is not None and lvec is not None and rvec is not None:
            try:
                sl, vl = cwc(1.0 * self.d * np.asarray(lvec), self.steer[0])
                sr, vr = cwc(1.0 * self.d * np.asarray(rvec), self.steer[1])
                if abs(sl - self.steer[0]) < 1e-3 and abs(sr - self.steer[1]) < 1e-3:
                    self._sign = np.array([np.sign(vl), np.sign(vr)])
            except Exception:
                pass

    def speeds(self, wz: float, gain: float = 1.0) -> np.ndarray:
        """Wheel-speed command for yaw rate ``wz``; ``gain`` undoes the sim scale."""
        return self._sign * wz * self.d * gain

    def sim_joint_target(self, wz: float, gain: float) -> float:
        """|joint velocity target| (rad/s) the sim will apply for this command."""
        return abs(wz) * self.d * gain * _SIM_DRIVE_SCALE


def _telemetry(chassis, err: float | None) -> str:
    st = np.rad2deg(np.asarray(chassis.steering_angle, float))
    try:
        wv = np.asarray(chassis.wheel_velocity, float)
    except Exception:
        wv = np.array([np.nan, np.nan])
    e = f"err {np.rad2deg(err):+6.1f} deg | " if err is not None else ""
    return f"{e}steer L{st[0]:+6.1f} R{st[1]:+6.1f} deg | wheel vel(reported) L{wv[0]:+.2f} R{wv[1]:+.2f}"


def presteer(chassis, turn: InplaceTurn, hz: float, tol_deg: float, timeout: float,
             rtf: float = 1.0, settle_tol_deg: float = 8.0) -> bool:
    """Command the in-place steering with zero drive until measured steering is close.

    Exits when both wheels are within ``tol_deg`` of the target, or when the
    steering has *stopped moving* (less than 0.3 deg change over ~0.7 s of sim
    time) with a residual below ``settle_tol_deg`` — the sim steer PD leaves a
    few degrees of static offset against ground friction, and waiting for it
    would only burn the whole budget.
    """
    period = 1.0 / hz
    t0 = time.monotonic()
    last = 0.0
    settle_window = 0.7 / max(rtf, 0.05)  # wall seconds
    hist: list[tuple[float, np.ndarray]] = []
    while True:
        chassis.set_motion_state(turn.steer, np.zeros(2))
        time.sleep(period)
        meas = np.asarray(chassis.steering_angle, float)
        derr = float(np.max(np.abs(meas - turn.steer)))
        now = time.monotonic()
        hist.append((now, meas))
        hist = [h for h in hist if now - h[0] <= settle_window]
        if now - last > 0.5:
            logger.info(f"  presteer target {np.round(np.rad2deg(turn.steer), 1)} | {_telemetry(chassis, None)}")
            last = now
        if derr < np.deg2rad(tol_deg):
            logger.info(f"Steering aligned in {now - t0:.1f}s (max err {np.rad2deg(derr):.1f} deg).")
            return True
        if now - hist[0][0] >= settle_window * 0.9 and len(hist) > 3:
            moved = float(np.rad2deg(np.max(np.abs(hist[-1][1] - hist[0][1]))))
            if moved < 0.3 and derr < np.deg2rad(settle_tol_deg):
                logger.info(f"Steering settled in {now - t0:.1f}s with a static offset of {np.rad2deg(derr):.1f} deg "
                            f"(moved {moved:.2f} deg in the last {settle_window:.1f}s); good enough, driving.")
                return True
        if now - t0 > timeout:
            logger.warning(
                f"Steering did not reach the in-place angles in {timeout:.0f}s wall "
                f"(max err {np.rad2deg(derr):.1f} deg). Driving anyway; check --probe output."
            )
            return False


def _encoders(chassis) -> np.ndarray | None:
    try:
        return np.asarray(chassis.wheel_encoder_pos, float)
    except Exception:
        return None


def run_probe(chassis, turn: InplaceTurn, aim: str, conveyor_yaw_deg: float, extra_deg: float, hz: float,
              rtf: float, wheel_gain: float, wheel_radius: float = _WHEEL_RADIUS,
              steer_sim_s: float = 4.0, drive_sim_s: float = 1.0, drive_wz: float = 1.2,
              steer_tol_deg: float = 10.0) -> None:
    """Diagnostic: steer only, then drive briefly; print what the chassis really does.

    Durations are in *sim* seconds (converted with ``rtf``). Wheel encoders tell
    whether the wheels rolled the commanded amount; comparing the yaw implied by
    the wheel travel with the measured heading change separates "wheels do not
    spin" from "wheels spin but slip".
    """
    steer_s, drive_s = steer_sim_s / rtf, drive_sim_s / rtf
    logger.info(f"=== PROBE 1/2: steer to in-place angles, zero drive (up to {steer_s:.0f}s wall = {steer_sim_s:.0f}s sim) ===")
    ok = presteer(chassis, turn, hz, tol_deg=steer_tol_deg, timeout=steer_s, rtf=rtf)
    err0, _ = heading_error(aim, conveyor_yaw_deg, extra_deg)
    sign = float(np.sign(err0)) or 1.0
    enc0 = _encoders(chassis)
    cmd = turn.speeds(sign * drive_wz, wheel_gain)
    logger.info(f"=== PROBE 2/2: drive wz={sign * drive_wz:+.2f} rad/s for {drive_s:.1f}s wall = {drive_sim_s:.1f}s sim "
                f"(wheel surface speed {np.round(turn.speeds(sign * drive_wz), 3)} m/s, sent {np.round(cmd, 4)} after gain {wheel_gain:.4f}, "
                f"sim joint target ~{turn.sim_joint_target(drive_wz, wheel_gain):.1f} rad/s) ===")
    period = 1.0 / hz
    t0 = time.monotonic()
    last = 0.0
    steer_log: list[np.ndarray] = []
    while time.monotonic() - t0 < drive_s:
        chassis.set_motion_state(turn.steer, cmd)
        time.sleep(period)
        steer_log.append(np.asarray(chassis.steering_angle, float))
        if time.monotonic() - last > 0.5:
            try:
                e, _ = heading_error(aim, conveyor_yaw_deg, extra_deg)
            except (ValueError, OSError):
                e = None
            logger.info("  " + _telemetry(chassis, e))
            last = time.monotonic()
    chassis.stop()
    time.sleep(max(0.5, 0.3 / rtf))
    enc1 = _encoders(chassis)
    err1, _ = heading_error(aim, conveyor_yaw_deg, extra_deg)
    turned = float(np.rad2deg(_wrap(err0 - err1)))
    expect = float(np.rad2deg(drive_wz * drive_sim_s))
    steer_dev = float(np.rad2deg(np.max(np.abs(np.array(steer_log) - turn.steer)))) if steer_log else float("nan")
    msg = [f"PROBE result (RTF {rtf:.2f}): steering reached={ok} (max deviation while driving {steer_dev:.1f} deg); "
           f"base turned {turned:+.1f} deg in {drive_sim_s:.1f}s sim (commanded {sign * expect:+.1f} deg)."]
    if enc0 is not None and enc1 is not None:
        travel = np.abs(enc1 - enc0)  # rad at the wheel
        rolled = float(np.mean(travel) * wheel_radius)
        implied = float(np.rad2deg(rolled / turn.d))
        expected_roll = drive_wz * turn.d * drive_sim_s
        msg.append(f"  wheel encoders moved L {travel[0]:.2f} R {travel[1]:.2f} rad -> {rolled:.3f} m of rolling "
                   f"(expected {expected_roll:.3f} m) = {implied:+.1f} deg of yaw if no slip (r={wheel_radius:.3f} m).")
        if not ok:
            msg.append("  -> steering did not reach the in-place angles within the sim-time budget: sim steer actuator too slow or command path issue.")
        elif rolled < 0.3 * expected_roll:
            msg.append("  -> wheels barely rolled: drive command not reaching the sim or --wheel-gain too small.")
        elif rolled > 3.0 * expected_roll:
            msg.append("  -> wheels rolled far more than commanded: --wheel-gain too large for this sim (unit scale).")
        elif abs(turned) < 0.5 * abs(implied):
            msg.append("  -> wheels rolled but the base did not follow: wheel slip / ground friction / caster resistance.")
        else:
            msg.append("  -> chassis responds consistently; the full turn should converge.")
    else:
        msg.append("  (wheel encoder readback unavailable; cannot separate slip from no-drive.)")
    logger.info("\n".join(msg))


# ---------------------------------------------------------------------------
def main(
    aim: str = "conveyor",
    conveyor_yaw_deg: float = 90.0,
    extra_deg: float = 0.0,
    tol_deg: float = 2.0,
    kp: float = 2.0,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    timeout_sim: float = 12.0,
    stall_s: float = 3.0,
    wheel_gain: float = _WHEEL_GAIN_SIM,
    sim_step_s: float = _SIM_STEP_S,
    dry_run: bool = False,
    probe: bool = False,
    wheel_radius: float = _WHEEL_RADIUS,
    leave_held: bool = True,
) -> None:
    """Turn the base in place until it faces the conveyor / AprilTag.

    All rates are in *sim* time: the script measures the simulator's real-time
    factor (RTF) from the status-file rewrite rate and stretches the wall-clock
    timeouts accordingly, so a slow simulator only makes the turn take longer.

    Defaults are tuned for the Isaac sim chassis (weak velocity drive: the wheel
    joint target must stay above ~4 rad/s or the base does not move, so wz_min is
    high; with RTF ~0.15 that is still only ~10 deg/s of wall-clock yaw, which
    the 20 Hz loop stops within tol_deg). Real robot: --wz-min 0.15 --wz-max 0.8
    --wheel-gain 1.

    Args:
        aim: "conveyor" = square to the conveyor face; "tag" = point at the tag.
        conveyor_yaw_deg: World yaw to face when only red/purple files exist.
        extra_deg: Turn this much further counter-clockwise (left, toward the conveyor box) past the square-on
            heading (degrees; negative = right). The held block sits ~0.8 m ahead, so each degree moves it
            ~1.4 cm toward the box face, so a few degrees noticeably shorten the approach in 10. Default 0 =
            stop at square-on (the user's choice); 10 turns the base itself when it cannot reach.
        tol_deg: Stop when |heading error| is below this (degrees).
        kp: P gain, wz = kp * error (rad/s per rad, sim time).
        wz_max: Yaw-rate cap (rad/s, sim time).
        wz_min: Minimum |wz| while turning (sim: keeps the wheel joint target
            above the drive's dead zone).
        hz: Command + feedback rate (wall clock).
        steer_tol_deg: Start driving once both wheels are within this of the
            in-place angles (steering keeps converging while driving).
        steer_timeout_sim: Max *sim* seconds to wait for the steering.
        timeout_sim: Give up after this many *sim* seconds of driving.
        stall_s: If the heading has not changed for this many wall seconds while
            driving, raise the wheel command step by step (up to the sim joint
            velocity limit); if it still does not move, stop and report.
        wheel_gain: Multiplier on the wheel-speed command. Default 1/(100*r)
            cancels the sim's chassis_drive_vel scale=100 (m/s -> rad/s joint
            target) so the commanded surface speed is physical. Real robot: 1.0.
        sim_step_s: Sim seconds per sim_main control step (sim.dt * decimation).
        dry_run: Only print the heading error; do not move.
        probe: Run the steering / drive diagnostic instead of the full turn.
        wheel_radius: Drive wheel radius (m) for the probe's encoder-based slip check.
        leave_held: Exit without ``stop()`` on arms/torso (safe while holding).
    """
    if aim not in ("conveyor", "tag"):
        raise ValueError("aim must be 'conveyor' or 'tag'")
    if not dry_run:
        log_path = _setup_file_log("probe" if probe else aim)
        logger.info(f"Logging to {log_path}")
    err, info = heading_error(aim, conveyor_yaw_deg, extra_deg)
    logger.info(f"Heading error {np.rad2deg(err):+.1f} deg  [{info}]")
    if dry_run:
        return
    tol = np.deg2rad(tol_deg)
    if abs(err) < tol and not probe:
        logger.info("Already facing the conveyor; nothing to do.")
        return

    t_connect = time.monotonic()
    bot = connect()
    chassis = bot.chassis
    turn = InplaceTurn(chassis)
    logger.info(f"Robot connected in {time.monotonic() - t_connect:.1f}s; measuring sim speed (2s)...")
    rtf = measure_rtf(TAG_FILE, sim_step_s)
    logger.info(
        f"In-place turn: steer {np.round(np.rad2deg(turn.steer), 1)} deg, "
        f"wheel surface speed = {turn.d:.3f} m * wz, signs {turn._sign.astype(int).tolist()}, "
        f"wheel command gain {wheel_gain:.4f} -> sim joint target {turn.sim_joint_target(wz_min, wheel_gain):.1f}"
        f"..{turn.sim_joint_target(wz_max, wheel_gain):.1f} rad/s for wz {wz_min}..{wz_max}"
    )
    do_shutdown = not leave_held
    period = 1.0 / hz
    try:
        if probe:
            run_probe(chassis, turn, aim, conveyor_yaw_deg, extra_deg, hz, rtf, wheel_gain,
                      wheel_radius=wheel_radius, steer_tol_deg=steer_tol_deg)
            return

        # Phase 1: steer first, with feedback (budget in sim time).
        t_steer = time.monotonic()
        presteer(chassis, turn, hz, steer_tol_deg, steer_timeout_sim / rtf, rtf=rtf)
        logger.info(f"Steering phase took {time.monotonic() - t_steer:.1f}s wall.")

        # Phase 2: P-control on the measured heading error, with stall boost.
        timeout = timeout_sim / rtf
        div_window = max(4.0, 1.5 / rtf)
        boost_cap = _SIM_JOINT_VEL_LIMIT / max(turn.sim_joint_target(wz_max, wheel_gain), 1e-6)
        logger.info(f"Driving: timeout {timeout:.0f}s wall ({timeout_sim:.0f}s sim), wz in [{wz_min}, {wz_max}] rad/s, "
                    f"stall check every {stall_s:.0f}s")
        t0 = time.monotonic()
        last_log = 0.0
        bad_reads = 0
        hist: list[tuple[float, float]] = []  # (t, |err|) for divergence / stall checks
        boost = 1.0
        stall_since = t0
        stalled_at_cap = 0
        while True:
            try:
                err, info = heading_error(aim, conveyor_yaw_deg, extra_deg)
                bad_reads = 0
            except (ValueError, OSError):
                bad_reads += 1  # sim rewrites the file every step; tolerate partial reads
                if bad_reads > 50:
                    raise
            now = time.monotonic()
            if abs(err) < tol:
                break
            if now - t0 > timeout:
                logger.warning(f"Timeout after {timeout:.0f}s wall; residual {np.rad2deg(err):+.1f} deg")
                break
            hist.append((now, abs(err)))
            hist = [h for h in hist if now - h[0] <= max(div_window, stall_s)]
            recent = [h[1] for h in hist if now - h[0] <= div_window]
            if now - t0 > div_window and recent and recent[0] + np.deg2rad(5.0) < abs(err):
                logger.error(
                    f"Heading error is growing ({np.rad2deg(recent[0]):.1f} -> {np.rad2deg(abs(err)):.1f} deg); "
                    "wheel-speed sign or feedback convention is wrong. Stopping."
                )
                break
            # Stall: heading unchanged for stall_s while we are commanding motion.
            window = [h[1] for h in hist if now - h[0] <= stall_s]
            if now - stall_since > stall_s and len(window) > 3 and (max(window) - min(window)) < np.deg2rad(0.3):
                if boost < boost_cap - 1e-6:
                    boost = min(boost * 1.5, boost_cap)
                    logger.warning(f"Base not moving for {stall_s:.0f}s; raising wheel command x{boost:.2f} "
                                   f"(sim joint target ~{turn.sim_joint_target(wz_max, wheel_gain) * boost:.1f} rad/s)")
                else:
                    stalled_at_cap += 1
                    if stalled_at_cap >= 1:
                        logger.error("Base does not move even at the maximum wheel command; stopping. "
                                     "Check the sim wheel drive (damping 10 is too weak) / ground friction.")
                        break
                stall_since = now
            wz = float(np.clip(kp * err, -wz_max, wz_max))
            if abs(wz) < wz_min:
                wz = float(np.sign(err) * wz_min)
            chassis.set_motion_state(turn.steer, turn.speeds(wz, wheel_gain * boost))
            if now - last_log > 1.0:
                logger.info(f"  wz {wz:+.2f}{f' x{boost:.2f}' if boost > 1 else ''} | {_telemetry(chassis, err)}")
                last_log = now
            time.sleep(period)
        chassis.stop()
        time.sleep(max(0.5, 0.3 / rtf))
        try:
            err, info = heading_error(aim, conveyor_yaw_deg, extra_deg)
        except (ValueError, OSError):
            pass
        logger.info(f"Done in {time.monotonic() - t0:.1f}s wall; residual {np.rad2deg(err):+.1f} deg  [{info}]")
    except KeyboardInterrupt:
        logger.warning("Interrupted; stopping chassis")
        try:
            chassis.stop()
        except Exception:
            pass
        do_shutdown = True
    finally:
        if do_shutdown:
            bot.shutdown()
        else:
            disconnect_leave_held(bot)
            logger.info("Client disconnected without stop() — arm/torso targets held.")


if __name__ == "__main__":
    tyro.cli(main)
