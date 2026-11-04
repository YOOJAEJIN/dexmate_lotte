# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Tared, low-pass filtered wrist-force monitor: a ``stop_check`` for ``_pink_arm.stream`` in force-gated pushes.

Written as Stage 2 of ``10.approach_purple_to_red_bimanual.py`` (2026-09-26, both wrists); since 2026-09-27 it
lives here and drives ``13.hands_push_left.py`` (both wrists; ``sides`` selects the hands).

The sim's wrist wrench (``state/wrench/left|right``, published by ``unitree_sim_isaaclab*/zenoh/zenoh_manager.py``
since 2026-09-23) is noisy even at rest -- holding the block on 2026-09-26: left |dF| up to 17 N raw, 8 N after a
0.25 s low-pass -- so the raw magnitude cannot be thresholded at a few newtons. Each wrist gets an exponential
moving average over ``ema_s`` seconds and its *own* threshold ``max(threshold_n, noise_gain x filtered rest p90 +
noise_margin_n)`` measured while arming (nothing may touch the block then). Contact = filtered |dF| above the
wrist's threshold for ``confirm_ticks`` consecutive checks. On contact the caller holds the *measured* pose
(``stream`` does that on an early stop), so nothing keeps pressing.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
from loguru import logger

from read_force import _get_arm_row, tare_force


class ContactGate:
    """Wrist-force contact monitor on one or both hands; ``arm(bot)`` once while free, ``check(bot)`` every tick.

    ``travel_fn`` (optional, set by the caller) returns the distance pushed so far (m) and only feeds the log."""

    def __init__(self, threshold_n: float, confirm_ticks: int, *, sample_hz: float, sides: Iterable[str] = ("left", "right"),
                 tare_samples: int = 10, warmup_s: float = 1.0, rest_s: float = 1.5, ema_s: float = 0.25,
                 log_period_s: float = 0.5, noise_gain: float = 1.5, noise_margin_n: float = 3.0) -> None:
        self.want_sides = [s for s in sides]
        if not self.want_sides or any(s not in ("left", "right") for s in self.want_sides):
            raise ValueError(f"sides must be a non-empty subset of ('left', 'right'), got {self.want_sides}")
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
        self.sides: list[str] = []  # sides that actually deliver wrench data (set by arm)
        self.thresholds: dict[str, float] = {}
        self.filt: dict[str, np.ndarray] = {}
        self.armed = False
        self.peak_n = 0.0
        self.stop_side: str | None = None
        self.stop_force_n: float | None = None
        self.already_loaded_n = 0.0
        self.travel_fn: Callable[[], float] | None = None
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
        """Warm the wrench subscribers up (they may be idle-paused), tare the wrists, measure the rest noise."""
        raw: dict[str, list[np.ndarray]] = {s: [] for s in self.want_sides}
        t0 = time.time()
        while time.time() - t0 < max(self.warmup_s, 0.3):
            for side in raw:
                f = self._force(bot, side)
                if f is not None:
                    raw[side].append(f)
            time.sleep(0.02)
        self.sides = [s for s, v in raw.items() if len(v) >= 3 and len({tuple(np.round(x, 4)) for x in v}) >= 2]
        if not self.sides:
            logger.warning("No wrist wrench data on state/wrench/" + "|".join(self.want_sides)
                           + " (sim zenoh_manager must publish it); contact detection disabled.")
            return False
        if not all([tare_force(s, bot, samples=self.tare_samples) for s in self.sides]):
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
        # than the max: right after a fast move the wrench can still show a brief settling spike, and sizing the
        # threshold off that one spike (2026-09-26: max 25 N) makes the gate ignore real, smaller contact forces.
        # already_loaded is the median of that same tail -- if it alone is already high, the grasp is under load
        # (near a joint limit or already touching) before any push happens.
        skip = int(self.ema_k) + 3
        self.thresholds = {}
        already_loaded: dict[str, float] = {}
        rest_p90: dict[str, float] = {}
        for side in self.sides:
            tail = rest[side][skip:] or rest[side][-1:] or [0.0]
            rest_p90[side] = float(np.percentile(tail, 90))
            self.thresholds[side] = max(self.threshold_n, self.noise_gain * rest_p90[side] + self.noise_margin_n)
            already_loaded[side] = float(np.median(tail))
        self.rest_p90 = rest_p90  # for the caller's diagnostics
        self.already_loaded_n = max(already_loaded.values(), default=0.0)
        if self.already_loaded_n > self.threshold_n:
            logger.warning(f"Wrist force is already {self.already_loaded_n:.0f} N at rest (median, no push yet) -- the grasp is under "
                           "load (near a joint limit or already touching the box) before any push motion.")
        self.armed = True
        self._above = 0
        self.peak_n = 0.0
        logger.info("Contact gate armed: filtered |dF| rest p90 "
                    + ", ".join(f"{s} {rest_p90[s]:.1f} N" for s in self.sides)
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
