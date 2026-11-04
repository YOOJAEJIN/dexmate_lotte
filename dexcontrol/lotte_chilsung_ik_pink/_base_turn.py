# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Closed-loop in-place yaw of the vega_1p chassis (extracted from ``8.rotate_base_to_conveyor.py``).

``turn_in_place(chassis, error_fn, ...)`` steers both wheels to the in-place-rotation angles first
(with feedback), then P-controls the yaw rate on ``error_fn() -> (err_rad, info)`` until the error is
below a tolerance, with the stall boost and divergence guard of the original. ``error_fn`` is read from
the simulator's per-step status files, so the loop keeps turning on a slow simulator until the
*measured* heading is right (open-loop ``turn(angle / speed)`` stops short below real time).

Sim specifics (see ``8.rotate_base_to_conveyor.py`` for the derivation): the wheel command is scaled by
``WHEEL_GAIN_SIM`` to undo the sim's chassis_drive_vel scale=100, the yaw-rate floor ``wz_min`` keeps
the wheel joint target above the weak velocity drive's dead zone, and all timeouts are given in *sim*
seconds and stretched by the measured real-time factor (``measure_rtf``). Real robot: ``wheel_gain=1``,
``wz_min`` ~0.15, ``wz_max`` ~0.8.

``8.rotate_base_to_conveyor.py`` still carries its own copy of this code; new scripts should import
from here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

# vega_1p chassis geometry (dexbot_utils vega_1/chassis.py); used only if the chassis object does not expose them.
AXIS_DIST = 0.219    # centre -> steer-wheel axle (m)
HALF_WHEELS = 0.225  # half wheel track (m)
WHEEL_RADIUS = 0.0864  # drive wheel radius (m), from the sim USD visual (0.173 m dia)
SIM_DRIVE_SCALE = 100.0  # env cfg chassis_drive_vel scale: "m/s" command -> rad/s joint target
WHEEL_GAIN_SIM = 1.0 / (SIM_DRIVE_SCALE * WHEEL_RADIUS)
SIM_JOINT_VEL_LIMIT = 10.0  # wheel joint velocity limit in the sim (rad/s)
SIM_STEP_S = 0.01  # sim seconds per sim_main control step (sim.dt 0.005 * decimation 2)

ErrorFn = Callable[[], tuple[float, str]]  # -> (signed yaw to turn, rad, +CCW; short description)


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def file_update_rate(path: Path, window_s: float = 2.0) -> float:
    """How often the sim rewrites ``path`` (Hz) -- a proxy for the sim step rate."""
    t_end = time.monotonic() + window_s
    last = None
    n = 0
    while time.monotonic() < t_end:
        try:
            m = Path(path).stat().st_mtime_ns
        except OSError:
            m = None
        if m is not None and m != last:
            n += 1
            last = m
        time.sleep(0.005)
    return n / window_s


def measure_rtf(path: Path, sim_step_s: float = SIM_STEP_S) -> float:
    """Real-time factor of the simulator (sim seconds per wall second) from the status-file rewrite
    rate (one rewrite per control step). Clipped to [0.05, 1]."""
    rate = file_update_rate(path)
    if rate <= 0.0:
        logger.warning("Status file is not being rewritten; assuming real time (RTF 1).")
        return 1.0
    rtf = float(np.clip(rate * sim_step_s, 0.05, 1.0))
    logger.info(f"Sim status updates {rate:.0f} Hz x {sim_step_s:.3f} s/step -> RTF ~{rtf:.2f} "
                f"(1 wall s = {rtf:.2f} sim s)")
    return rtf


class InplaceTurn:
    """Steering angles + wheel speeds for a pure yaw about the base centre.

    Same geometry as ``Chassis.set_velocity(0, 0, wz)``: steer wheels at +/-atan2(axle_dist, half_track)
    (~44 deg for vega_1p), wheel speed ``wz * centre_to_wheel_dist`` with the left wheel reversed for CCW.
    """

    def __init__(self, chassis):
        axis = float(getattr(chassis, "_center_to_wheel_axis_dist", AXIS_DIST))
        half = float(getattr(chassis, "_half_wheels_dist", HALF_WHEELS))
        self.d = float(np.hypot(axis, half))
        a = float(np.arctan2(axis, half))
        self.steer = np.array([+a, -a])
        # Let dexcontrol's own solver pick the wheel-speed signs when available (guards against a
        # convention change); fall back to the derivation above.
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
        return abs(wz) * self.d * gain * SIM_DRIVE_SCALE


def telemetry(chassis, err: float | None) -> str:
    st = np.rad2deg(np.asarray(chassis.steering_angle, float))
    try:
        wv = np.asarray(chassis.wheel_velocity, float)
    except Exception:
        wv = np.array([np.nan, np.nan])
    e = f"err {np.rad2deg(err):+6.1f} deg | " if err is not None else ""
    return f"{e}steer L{st[0]:+6.1f} R{st[1]:+6.1f} deg | wheel vel(reported) L{wv[0]:+.2f} R{wv[1]:+.2f}"


def presteer(chassis, turn: InplaceTurn, hz: float, tol_deg: float, timeout: float,
             rtf: float = 1.0, settle_tol_deg: float = 8.0) -> bool:
    """``presteer_angles`` for an in-place turn."""
    return presteer_angles(chassis, turn.steer, hz, tol_deg, timeout, rtf=rtf, settle_tol_deg=settle_tol_deg)


def presteer_angles(chassis, steer: np.ndarray, hz: float, tol_deg: float, timeout: float,
                    rtf: float = 1.0, settle_tol_deg: float = 8.0) -> bool:
    """Command the steering angles ``steer`` with zero drive until the measured steering is close.

    Exits when both wheels are within ``tol_deg`` of the target, or when the steering has stopped
    moving (< 0.3 deg over ~0.7 s of sim time) with a residual below ``settle_tol_deg`` -- the sim
    steer PD leaves a few degrees of static offset against ground friction.
    """
    period = 1.0 / hz
    t0 = time.monotonic()
    last = 0.0
    settle_window = 0.7 / max(rtf, 0.05)  # wall seconds
    hist: list[tuple[float, np.ndarray]] = []
    steer = np.asarray(steer, float)
    while True:
        chassis.set_motion_state(steer, np.zeros(2))
        time.sleep(period)
        meas = np.asarray(chassis.steering_angle, float)
        derr = float(np.max(np.abs(meas - steer)))
        now = time.monotonic()
        hist.append((now, meas))
        hist = [h for h in hist if now - h[0] <= settle_window]
        if now - last > 0.5:
            logger.info(f"  presteer target {np.round(np.rad2deg(steer), 1)} | {telemetry(chassis, None)}")
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
            logger.warning(f"Steering did not reach the target angles in {timeout:.0f}s wall "
                           f"(max err {np.rad2deg(derr):.1f} deg). Driving anyway.")
            return False


class CrabMove:
    """Steering angles + wheel speeds for a pure sideways translation of the base (robot y).

    Both steer wheels point along the body y axis; the wheel speed is the translation speed with
    the sign dexcontrol's own ``_compute_wheel_control`` picks for a ``[0, vy]`` wheel velocity
    (fallback: steering -90 deg, speed = vy, i.e. positive = left).
    """

    def __init__(self, chassis):
        self.steer = np.array([-np.pi / 2, -np.pi / 2])
        self._sign = 1.0
        cwc = getattr(chassis, "_compute_wheel_control", None)
        if cwc is not None:
            try:
                ang, spd = cwc(np.array([0.0, 1.0]), -np.pi / 2)
                if abs(abs(ang) - np.pi / 2) < 1e-3:
                    self.steer = np.array([ang, ang])
                    self._sign = float(np.sign(spd)) or 1.0
            except Exception:
                pass

    def speeds(self, vy: float, gain: float = 1.0) -> np.ndarray:
        """Wheel-speed command for a sideways speed ``vy`` (m/s, +left); ``gain`` undoes the sim scale."""
        return np.array([1.0, 1.0]) * self._sign * vy * gain

    def sim_joint_target(self, vy: float, gain: float) -> float:
        return abs(vy) * gain * SIM_DRIVE_SCALE


def translate_sideways(
    chassis,
    error_fn: ErrorFn,
    *,
    tol_m: float = 0.015,
    kp: float = 3.0,
    v_max: float = 0.5,
    v_min: float = 0.38,
    hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    timeout_sim: float = 12.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
    rtf: float = 1.0,
) -> tuple[float, str]:
    """Translate the base sideways (robot y, no rotation) until ``error_fn()`` -> (err_m, info) is below
    ``tol_m``; positive error = target is to the LEFT. Same structure as ``turn_in_place``: steer both
    wheels to the crab angles first, then ``vy = clip(kp * err, +/-v_max)`` with ``|vy| >= v_min`` (sim
    drive dead zone: 0.38 m/s = 4.4 rad/s joint target), stall boost, divergence guard, timeout.
    The base is stopped on every exit path (KeyboardInterrupt re-raised)."""
    crab = CrabMove(chassis)
    logger.info(
        f"Sideways move: steer {np.round(np.rad2deg(crab.steer), 1)} deg, speed sign {crab._sign:+.0f}, wheel gain "
        f"{wheel_gain:.4f} -> sim joint target {crab.sim_joint_target(v_min, wheel_gain):.1f}..{crab.sim_joint_target(v_max, wheel_gain):.1f} rad/s"
    )
    period = 1.0 / hz
    err, info = error_fn()
    try:
        t_steer = time.monotonic()
        presteer_angles(chassis, crab.steer, hz, steer_tol_deg, steer_timeout_sim / rtf, rtf=rtf)
        logger.info(f"Steering phase took {time.monotonic() - t_steer:.1f}s wall.")
        timeout = timeout_sim / rtf
        div_window = max(4.0, 1.5 / rtf)
        boost_cap = SIM_JOINT_VEL_LIMIT / max(crab.sim_joint_target(v_max, wheel_gain), 1e-6)
        logger.info(f"Driving sideways: timeout {timeout:.0f}s wall ({timeout_sim:.0f}s sim), |vy| in [{v_min}, {v_max}] m/s, "
                    f"stall check every {stall_s:.0f}s")
        t0 = time.monotonic()
        last_log = 0.0
        bad_reads = 0
        hist: list[tuple[float, float]] = []
        boost = 1.0
        stall_since = t0
        while True:
            try:
                err, info = error_fn()
                bad_reads = 0
            except (ValueError, OSError):
                bad_reads += 1
                if bad_reads > 50:
                    raise
            now = time.monotonic()
            if abs(err) < tol_m:
                break
            if now - t0 > timeout:
                logger.warning(f"Timeout after {timeout:.0f}s wall; residual {err * 100:+.1f} cm")
                break
            hist.append((now, abs(err)))
            hist = [h for h in hist if now - h[0] <= max(div_window, stall_s)]
            recent = [h[1] for h in hist if now - h[0] <= div_window]
            if now - t0 > div_window and recent and recent[0] + 0.05 < abs(err):
                logger.error(f"Sideways error is growing ({recent[0] * 100:.1f} -> {abs(err) * 100:.1f} cm); "
                             "wheel-speed sign or feedback convention is wrong. Stopping.")
                break
            window = [h[1] for h in hist if now - h[0] <= stall_s]
            if now - stall_since > stall_s and len(window) > 3 and (max(window) - min(window)) < 0.003:
                if boost < boost_cap - 1e-6:
                    boost = min(boost * 1.5, boost_cap)
                    logger.warning(f"Base not moving for {stall_s:.0f}s; raising wheel command x{boost:.2f}")
                else:
                    logger.error("Base does not move even at the maximum wheel command; stopping.")
                    break
                stall_since = now
            vy = float(np.clip(kp * err, -v_max, v_max))
            if abs(vy) < v_min:
                vy = float(np.sign(err) * v_min)
            chassis.set_motion_state(crab.steer, crab.speeds(vy, wheel_gain * boost))
            if now - last_log > 1.0:
                logger.info(f"  vy {vy:+.2f}{f' x{boost:.2f}' if boost > 1 else ''} | err {err * 100:+.1f} cm | "
                            f"steer {np.round(np.rad2deg(np.asarray(chassis.steering_angle, float)), 1)} deg | {info}")
                last_log = now
            time.sleep(period)
    except KeyboardInterrupt:
        logger.warning("Interrupted; stopping chassis")
        try:
            chassis.set_motion_state(crab.steer, np.zeros(2))
            chassis.stop()
        except Exception:
            pass
        raise
    # brake with the crab steering held, then release
    for _ in range(max(int(0.5 / period), 3)):
        chassis.set_motion_state(crab.steer, np.zeros(2))
        time.sleep(period)
    chassis.stop()
    time.sleep(max(0.5, 0.3 / rtf))
    try:
        err, info = error_fn()
    except (ValueError, OSError):
        pass
    logger.info(f"Sideways move done; residual {err * 100:+.1f} cm  [{info}]")
    return err, info


def turn_in_place(
    chassis,
    error_fn: ErrorFn,
    *,
    tol_deg: float = 2.0,
    kp: float = 2.0,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    timeout_sim: float = 12.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
    rtf: float = 1.0,
    drive_gate: Callable[[], bool] | None = None,
    gate_timeout_s: float = 60.0,
) -> tuple[float, str]:
    """Turn the base in place until ``error_fn()`` is below ``tol_deg``; returns the final ``(err, info)``.

    ``drive_gate`` (optional): after the wheels are steered, driving waits until this returns True (the steering is
    held with zero wheel speed meanwhile; ``gate_timeout_s`` then drives anyway with a warning). Lets a caller steer
    the wheels early -- that alone takes several seconds and moves nothing -- and release the actual rotation on a
    condition of its own, e.g. the arms having cleared an obstacle (14.return_home.py, 2026-09-27).

    Phase 1 steers the wheels to the in-place angles with feedback; phase 2 applies
    ``wz = clip(kp * err, +/-wz_max)`` with ``|wz| >= wz_min`` while turning, raises the wheel command
    step by step if the heading stalls, and stops on timeout or when the error grows (sign error).
    The chassis is stopped on every exit path (including KeyboardInterrupt, which is re-raised).
    """
    turn = InplaceTurn(chassis)
    logger.info(
        f"In-place turn: steer {np.round(np.rad2deg(turn.steer), 1)} deg, wheel surface speed = {turn.d:.3f} m * wz, "
        f"signs {turn._sign.astype(int).tolist()}, wheel gain {wheel_gain:.4f} -> sim joint target "
        f"{turn.sim_joint_target(wz_min, wheel_gain):.1f}..{turn.sim_joint_target(wz_max, wheel_gain):.1f} rad/s"
    )
    tol = np.deg2rad(tol_deg)
    period = 1.0 / hz
    err, info = error_fn()
    try:
        t_steer = time.monotonic()
        presteer(chassis, turn, hz, steer_tol_deg, steer_timeout_sim / rtf, rtf=rtf)
        logger.info(f"Steering phase took {time.monotonic() - t_steer:.1f}s wall.")
        if drive_gate is not None and not drive_gate():
            logger.info("Steering ready; holding the wheels until the caller releases the turn (drive gate)...")
            t_gate = time.monotonic()
            while not drive_gate():
                chassis.set_motion_state(turn.steer, np.zeros(2))
                time.sleep(period)
                if time.monotonic() - t_gate > gate_timeout_s:
                    logger.warning(f"Drive gate not released within {gate_timeout_s:.0f}s; driving anyway.")
                    break
            else:
                logger.info(f"Drive gate released after {time.monotonic() - t_gate:.1f}s; driving.")
        timeout = timeout_sim / rtf
        div_window = max(4.0, 1.5 / rtf)
        boost_cap = SIM_JOINT_VEL_LIMIT / max(turn.sim_joint_target(wz_max, wheel_gain), 1e-6)
        logger.info(f"Driving: timeout {timeout:.0f}s wall ({timeout_sim:.0f}s sim), wz in [{wz_min}, {wz_max}] rad/s, "
                    f"stall check every {stall_s:.0f}s")
        t0 = time.monotonic()
        last_log = 0.0
        bad_reads = 0
        hist: list[tuple[float, float]] = []
        boost = 1.0
        stall_since = t0
        while True:
            try:
                err, info = error_fn()
                bad_reads = 0
            except (ValueError, OSError):
                bad_reads += 1  # the sim rewrites the files every step; tolerate partial reads
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
                logger.error(f"Heading error is growing ({np.rad2deg(recent[0]):.1f} -> {np.rad2deg(abs(err)):.1f} deg); "
                             "wheel-speed sign or feedback convention is wrong. Stopping.")
                break
            window = [h[1] for h in hist if now - h[0] <= stall_s]
            if now - stall_since > stall_s and len(window) > 3 and (max(window) - min(window)) < np.deg2rad(0.3):
                if boost < boost_cap - 1e-6:
                    boost = min(boost * 1.5, boost_cap)
                    logger.warning(f"Base not moving for {stall_s:.0f}s; raising wheel command x{boost:.2f}")
                else:
                    logger.error("Base does not move even at the maximum wheel command; stopping.")
                    break
                stall_since = now
            wz = float(np.clip(kp * err, -wz_max, wz_max))
            if abs(wz) < wz_min:
                wz = float(np.sign(err) * wz_min)
            chassis.set_motion_state(turn.steer, turn.speeds(wz, wheel_gain * boost))
            if now - last_log > 1.0:
                logger.info(f"  wz {wz:+.2f}{f' x{boost:.2f}' if boost > 1 else ''} | {telemetry(chassis, err)}")
                last_log = now
            time.sleep(period)
    except KeyboardInterrupt:
        logger.warning("Interrupted; stopping chassis")
        brake(chassis, turn.steer, period)
        raise
    # Brake: keep re-sending zero wheel speed with the steering held, then stop() several times. A single stop()
    # message could be delayed or dropped while another thread streams arm commands on the same session, and the
    # sim holds the last received wheel target -- the base would then keep spinning (2026-09-27).
    brake(chassis, turn.steer, period)
    time.sleep(max(0.5, 0.3 / rtf))
    try:
        err, info = error_fn()
    except (ValueError, OSError):
        pass
    logger.info(f"Turn done; residual {np.rad2deg(err):+.1f} deg  [{info}]")
    return err, info


def brake(chassis, steer: np.ndarray, period: float = 0.05, hold_s: float = 0.5, n_stop: int = 5) -> None:
    """Zero the wheel speeds with the steering held for ``hold_s`` (re-sent every ``period``), then ``stop()``
    ``n_stop`` times. Never raises."""
    try:
        for _ in range(max(int(hold_s / period), 3)):
            chassis.set_motion_state(np.asarray(steer, float), np.zeros(2))
            time.sleep(period)
        for _ in range(n_stop):
            chassis.stop()
            time.sleep(period)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Brake command failed: {exc!r}")
