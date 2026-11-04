# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Like step 9, but the final push is force-gated: the left gripper keeps moving forward along the
approach direction until the wrist F/T sensor feels the box, then stops and holds.

Stages (straight lines, Pink IK from the measured state, left arm only, see ``_pink_arm.py``):

  1. lift   -- straight up (or down) at the current x, y to z_R - ``--lower-cm`` (just below the
               right gripper), orientation held.
  2. align  -- one straight move at that height to ``--start-back-cm`` short of the mirror of the
               right gripper's x, y, turning to the mirrored orientation on the way (``_left_mirror.py``).
  3. push   -- warm the wrench subscriber up and tare the left wrist wrench while standing still, then
               move forward along the approach direction (horizontal, height held) towards
               mirror + ``--max-push-cm``. Every control tick the tared force is checked (``--metric``:
               ``axial`` = |dFz| along the wrist/approach axis, default; ``mag`` = |dF|); once it exceeds
               ``--force-threshold-n`` for ``--confirm-ticks`` consecutive ticks -- and at least
               ``--min-push-cm`` has been travelled -- the motion stops where it is and the *measured*
               pose is held, so the arm does not keep pressing. If nothing is felt the push ends at the
               planned maximum (logged as such). No measured-state correction. Note the long blade tip
               sits ~19 cm ahead of the gripper origin, so contact starts well before the gripper body
               reaches the box; a light first touch reads ~3 N on the sim, pushing the blade in reads more.
  4. backoff -- optional: retreat ``--backoff-cm`` along the approach direction after contact.

Force source: ``dexcontrol`` ``left_arm.wrench_sensor`` (topic ``state/wrench/left``), read through
``read_force.tare_force`` / ``get_force``. On the simulator this topic is published by
``unitree_sim_isaaclab*/zenoh/zenoh_manager.py`` (wrist joint reaction wrench, added 2026-09-23 --
the sim has to be (re)started with that version). If no wrench data arrives the run stops before
the push unless ``--fallback-cm`` > 0, which then does a fixed push of that length like step 9.

Example (sim running, ROBOT_NAME / ZENOH_CONFIG exported, omniteleop follower stopped)::

    python 10.left_mirror_and_push_force.py                      # push until |dFz| >= 8 N (after >= 1 cm), max 8 cm past the mirror
    python 10.left_mirror_and_push_force.py --force-threshold-n 12 --min-push-cm 2 --max-push-cm 6
    python 10.left_mirror_and_push_force.py --metric mag --force-threshold-n 5
    python 10.left_mirror_and_push_force.py --start-back-cm 4 --backoff-cm 1
    python 10.left_mirror_and_push_force.py --pause               # SPACE between stages ('q' stops, pose held)
    python 10.left_mirror_and_push_force.py --dry-run             # plan only (URDF neutral pose, no force)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pink
import pinocchio as pin
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _left_mirror import ORIGIN, MirrorPlan  # noqa: E402
from _pink_arm import TORSO_MAX_VEL, TORSO_POSTURE_COST, Stage, run_stages  # noqa: E402
from _pink_common import ARM_JOINTS, EE  # noqa: E402
from read_force import _get_arm_row, tare_force  # noqa: E402
from utils import wait_for_space  # noqa: E402


# Long blade (Finger_1, 2FPA-I-03-180 2nd gen, 2026-10-30) tip is ~0.221 m ahead of L_gripper_base along EE +Z
# (sim USD: z_max 0.3031 in L_arm_l8 minus the 0.0825 m l8 -> gripper_base offset of the IK URDF). Logging only.
# Was 0.188 with the 1st-gen 06.LG gripper.
BLADE_TIP_AHEAD_M = 0.2206


def _tared_force(bot) -> np.ndarray | None:
    """Tared left wrist force vector [fx, fy, fz] in the sensor (wrist) frame, or None if unavailable."""
    row = _get_arm_row("left", bot)
    return None if row is None else np.array([row["fx"], row["fy"], row["fz"]], dtype=float)


class ForceGate:
    """Tared left-wrist force monitor used as the push stage's ``before`` / ``stop_check`` hooks.

    Metric (``--metric``): ``axial`` = |dFz| in the wrist frame -- the wrist/gripper +Z is the approach
    axis, so this is the resistance along the push and ignores the gravity-related drift that lives
    mostly on x/y (the gripper weight, ~22 N, sits on the wrist x axis); ``mag`` = |dF| of the whole
    vector. Nothing can stop the push before ``min_push_cm`` of travel (measured by FK of the left
    joints), so a light touch at the very start does not end it.
    """

    def __init__(self, threshold_n: float, confirm_ticks: int, tare_samples: int, log_period_s: float,
                 fallback_cm: float, *, metric: str = "axial", min_push_cm: float = 1.0, warmup_s: float = 1.0) -> None:
        if metric not in ("axial", "mag"):
            raise ValueError("metric must be 'axial' or 'mag'")
        self.threshold_n = threshold_n
        self.confirm_ticks = max(int(confirm_ticks), 1)
        self.tare_samples = max(int(tare_samples), 3)
        self.log_period_s = log_period_s
        self.fallback_cm = fallback_cm
        self.metric = metric
        self.min_push_m = max(min_push_cm, 0.0) / 100.0
        self.warmup_s = warmup_s
        self.a_h: np.ndarray | None = None  # set by ForcePushPlan.prepare (unit horizontal approach direction)
        self.available = False
        self.armed = False
        self._above = 0
        self._last_log = 0.0
        self.peak_n = 0.0
        self.noise_n = 0.0
        self.travel_m = 0.0
        self.stop_force_n: float | None = None
        self.stop_vec: np.ndarray | None = None
        self._cfg = None
        self._data = None
        self._q_idx: list[int] | None = None
        self._frame_id: int | None = None
        self.p_start: np.ndarray | None = None

    def metric_value(self, f: np.ndarray) -> float:
        return abs(float(f[2])) if self.metric == "axial" else float(np.linalg.norm(f))

    # -- before the push: settle, warm the subscriber up, check the sensor, tare ----------------
    def before(self, bot, cfg: pink.Configuration) -> bool:
        time.sleep(0.3)  # let the arm settle after the align stage
        # The wrench subscriber may have been idle-paused by dexcontrol; poll until fresh samples flow.
        raw: list[np.ndarray] = []
        t0 = time.time()
        while time.time() - t0 < max(self.warmup_s, 0.2):
            f = _tared_force(bot)  # no baseline yet -> raw sensor values
            if f is not None:
                raw.append(f)
            time.sleep(0.02)
        self.available = bool(raw)
        if not self.available:
            if self.fallback_cm > 0:
                logger.warning(
                    "No left wrist wrench data (state/wrench/left). Falling back to a fixed push of "
                    f"{self.fallback_cm:.1f} cm past the mirror spot (--fallback-cm)."
                )
                return True
            logger.error(
                "No left wrist wrench data on state/wrench/left. On the simulator this needs zenoh_manager.py with "
                "wrist wrench publishing (2026-09-23) and a sim restart; check with `python read_force.py --side left`. "
                "Nothing pushed (--fallback-cm N does a fixed push instead)."
            )
            return False

        def n_distinct() -> int:
            return len({tuple(np.round(r, 4)) for r in raw})

        if n_distinct() < 3:
            logger.warning(f"Wrench data looks stale ({len(raw)} samples, {n_distinct()} distinct values); waiting up to 2 s for fresh data...")
            t1 = time.time()
            while time.time() - t1 < 2.0 and n_distinct() < 3:
                f = _tared_force(bot)
                if f is not None:
                    raw.append(f)
                time.sleep(0.02)
            if n_distinct() < 3:
                logger.warning("Wrench values still do not change; taring anyway -- the contact check may be unreliable.")
        if not tare_force("left", bot, samples=self.tare_samples):
            logger.error("Taring the left wrist wrench failed; nothing pushed.")
            return False
        arr = np.array(raw[-25:])
        dev = arr - arr.mean(axis=0)
        self.noise_n = max((self.metric_value(d) for d in dev), default=0.0)
        logger.info(f"Rest noise of the '{self.metric}' force metric: up to {self.noise_n:.2f} N over {len(arr)} samples "
                    f"(raw mean {np.round(arr.mean(axis=0), 2)} N)")
        if self.threshold_n < 3.0 * self.noise_n:
            logger.warning(f"--force-threshold-n {self.threshold_n:.1f} N is below 3x the rest noise ({self.noise_n:.2f} N); "
                           "expect false contacts -- raise it or --confirm-ticks.")
        # FK of the left joints for the travelled distance (separate Data so pink's cache is untouched)
        self._cfg = cfg
        model = cfg.model
        self._data = model.createData()
        self._q_idx = [model.idx_qs[model.getJointId(n)] for n in ARM_JOINTS["left"]]
        self._frame_id = model.getFrameId(EE["left"])
        self.p_start = self._fk(bot)
        self.armed = True
        self._above = 0
        self.peak_n = 0.0
        self.travel_m = 0.0
        logger.info(
            f"Force gate armed: stop when {'|dFz| along the approach axis' if self.metric == 'axial' else '|dF|'} >= "
            f"{self.threshold_n:.1f} N for {self.confirm_ticks} ticks, but not before {self.min_push_m * 100:.1f} cm of travel."
        )
        return True

    def _fk(self, bot) -> np.ndarray:
        q = self._cfg.q.copy()
        q[self._q_idx] = np.asarray(bot.left_arm.get_joint_pos(), dtype=float)
        pin.forwardKinematics(self._cfg.model, self._data, q)
        pin.updateFramePlacement(self._cfg.model, self._data, self._frame_id)
        return self._data.oMf[self._frame_id].translation.copy()

    def travel(self, bot) -> float:
        """Distance advanced since arming along the approach direction (metres)."""
        d = self._fk(bot) - self.p_start
        return float(d[:2] @ self.a_h[:2]) if self.a_h is not None else float(np.linalg.norm(d))

    # -- every streaming tick ---------------------------------------------------------------
    def stop_check(self, bot) -> bool:
        if not self.armed:
            return False
        f = _tared_force(bot)
        if f is None:
            return False
        v = self.metric_value(f)
        self.peak_n = max(self.peak_n, v)
        self.travel_m = self.travel(bot)
        now = time.time()
        if now - self._last_log >= self.log_period_s:
            self._last_log = now
            logger.info(f"  wrist {self.metric} dF = {v:.2f} N (peak {self.peak_n:.2f}), dF vector {np.round(f, 2)} N, "
                        f"travelled {self.travel_m * 100:.1f} cm")
        if self.travel_m < self.min_push_m:
            self._above = 0
            return False
        if v >= self.threshold_n:
            self._above += 1
            if self._above >= self.confirm_ticks:
                self.stop_force_n = v
                self.stop_vec = f.copy()
                logger.info(f"CONTACT: {self.metric} dF = {v:.2f} N >= {self.threshold_n:.1f} N after {self.travel_m * 100:.1f} cm "
                            f"(dF vector {np.round(f, 2)} N) -- stopping the push here.")
                return True
        else:
            self._above = 0
        return False


class ForcePushPlan(MirrorPlan):
    """MirrorPlan whose align ends ``start_back_cm`` short of the mirror; adds the push / backoff specs."""

    def __init__(self, *, lower_cm: float, start_back_cm: float, max_push_cm: float, backoff_cm: float,
                 flip_blade: bool, min_hand_gap_m: float, allow_cross: bool, gate: ForceGate, recover_z: float = 0.55) -> None:
        super().__init__(lower_cm, -abs(start_back_cm), flip_blade, min_hand_gap_m, allow_cross, recover_z=recover_z)
        self.max_push_m = max_push_cm / 100.0
        self.backoff_m = backoff_cm / 100.0
        self.gate = gate

    def prepare(self, cfg: pink.Configuration) -> bool:
        if not super().prepare(cfg):
            return False
        if self.a_h is None:
            logger.error("Approach axis is nearly vertical; a forward push is ill-defined. Nothing commanded.")
            return False
        self.gate.a_h = self.a_h
        p_max = self.p_mirror + self.max_push_m * self.a_h
        a = self.R_goal[:, 2]
        logger.info(
            f"Force push: from {np.round(self.p_final, 4)} forward along {np.round(self.a_h[:2], 3)} up to "
            f"{np.round(p_max, 4)} ({(self.max_push_m - self.forward_m) * 100:.1f} cm max travel, "
            f"{self.max_push_m * 100:.1f} cm past the mirror spot), stopping on contact. Long blade tip is "
            f"~{BLADE_TIP_AHEAD_M * 100:.0f} cm ahead of the gripper origin: at the push start ~{np.round(self.p_final + BLADE_TIP_AHEAD_M * a, 3)}, "
            f"at the mirror spot ~{np.round(self.p_mirror + BLADE_TIP_AHEAD_M * a, 3)} (robot base)."
        )
        return True

    def spec_push(self, cfg: pink.Configuration):
        T = cfg.get_transform_frame_to_world(EE["left"])  # measured, after align
        if self.gate.armed:
            goal = self.p_mirror + self.max_push_m * self.a_h
        else:  # fallback: fixed push past the mirror spot
            goal = self.p_mirror + (self.gate.fallback_cm / 100.0) * self.a_h
        logger.info(f"Pushing from gripper origin {np.round(T.translation, 4)} towards {np.round(goal, 4)} "
                    f"({'force-gated' if self.gate.armed else 'fixed length, no force data'})")
        return ORIGIN, goal, self.R_goal.copy()

    def spec_backoff(self, cfg: pink.Configuration):
        T = cfg.get_transform_frame_to_world(EE["left"])
        return ORIGIN, T.translation - self.backoff_m * self.a_h, self.R_goal.copy()


def build_stages(
    plan: ForcePushPlan, *, lift_duration: float, align_duration: float, push_duration: float, step_m: float,
    step_rad: float, refine: int, align_refine: int, refine_tol_m: float, refine_tol_rad: float,
) -> list[Stage]:
    # every stage biases the IK null space toward the mirror of the right arm's joints (see Stage.posture_joints)
    common = dict(step_m=step_m, step_rad=step_rad, refine_tol_m=refine_tol_m, refine_tol_rad=refine_tol_rad,
                  posture_joints=plan.posture_joints)
    stages = [
        Stage("recover (descend if the lift cannot be planned)", plan.spec_descend, duration=3.0, refine=0, **common),
        Stage("lift", plan.spec_lift, duration=lift_duration, refine=refine, **common),
        Stage("align", plan.spec_align, duration=align_duration, refine=align_refine, **common),
        Stage("push until contact", plan.spec_push, duration=push_duration, refine=0,
              step_m=min(step_m, 0.005), step_rad=step_rad, refine_tol_m=refine_tol_m, refine_tol_rad=refine_tol_rad,
              posture_joints=plan.posture_joints,
              before=plan.gate.before, stop_check=plan.gate.stop_check, stop_label="wrist force threshold"),
    ]
    if plan.backoff_m > 0:
        stages.append(Stage("backoff", plan.spec_backoff, duration=1.5, refine=0, **common))
    return stages


def _pause(next_label: str) -> bool:
    logger.info(f"Next stage: {next_label}.")
    try:
        wait_for_space()
    except KeyboardInterrupt:
        return False
    return True


def main(
    lower_cm: float = 2.0,
    recover_z: float = 0.55,
    start_back_cm: float = 2.0,
    max_push_cm: float = 8.0,
    force_threshold_n: float = 8.0,
    metric: str = "axial",
    min_push_cm: float = 1.0,
    warmup_s: float = 1.0,
    confirm_ticks: int = 3,
    tare_samples: int = 15,
    backoff_cm: float = 0.0,
    fallback_cm: float = 0.0,
    flip_blade: bool = False,
    lift_duration: float = 2.0,
    align_duration: float = 4.0,
    push_duration: float = 4.0,
    force_log_period_s: float = 0.5,
    step_m: float = 0.01,
    step_deg: float = 3.0,
    refine: int = 1,
    align_refine: int = 1,
    refine_tol_m: float = 0.005,
    refine_tol_deg: float = 2.0,
    settle_s: float = 0.7,
    min_hand_gap_m: float = 0.12,
    allow_cross: bool = False,
    pause: bool = False,
    leave_held: bool = True,
    dry_run: bool = False,
    use_torso: bool = False,
    auto_torso: bool = False,
    torso_posture_cost: float = TORSO_POSTURE_COST,
    torso_max_vel: float = TORSO_MAX_VEL,
    max_joint_vel: float = 0.5,
    anchor_setpoints: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Lift, align to the mirror of the right gripper, then push forward until the wrist F/T feels the box.

    Args:
        lower_cm: Height below the right gripper for the lift and the push (centimetres).
        recover_z: If the lift cannot be planned from the current arm pose (elbow folded, wrist at its limit after an
            aborted run), the hand first goes straight down to this height (metres, robot base) and lifts again.
        start_back_cm: The align stage stops this far short of the mirror spot; the push starts there (centimetres).
        max_push_cm: The push ends at most this far past the mirror spot if no contact is felt (centimetres).
        force_threshold_n: Stop when the tared wrist force metric reaches this (newtons). 3 N is a first light touch;
            pushing a blade into a card box needs more (default 8).
        metric: "axial" = |dFz| along the wrist/approach axis (rejects gravity drift on x/y); "mag" = |dF| of the vector.
        min_push_cm: The gate cannot stop the push before this much travel (centimetres).
        warmup_s: Poll the wrench for this long before taring so an idle-paused subscriber delivers fresh samples.
        confirm_ticks: Consecutive control ticks (20 ms) above the threshold needed to stop (spike rejection).
        tare_samples: Wrench samples averaged for the baseline F0 before the push.
        backoff_cm: After contact, retreat this far along the approach direction (0 = hold at contact).
        fallback_cm: If no wrench data is available, do a fixed push this far past the mirror instead (0 = abort).
        flip_blade: Rotate the mirrored goal 180 deg about the approach axis (swap which finger faces the box).
        lift_duration: Stream time of the lift stage (seconds).
        align_duration: Stream time of the align stage (seconds).
        push_duration: Nominal stream time of the full push (seconds); it ends early on contact.
        force_log_period_s: How often the wrist force is logged during the push (seconds).
        step_m: Cartesian spacing between IK waypoints (metres).
        step_deg: Rotation spacing between IK waypoints (degrees).
        refine: Measured-state correction moves after the lift.
        align_refine: Measured-state correction moves after the align (it ends short of the box, so 1 is safe).
        refine_tol_m: Position tolerance that ends refinement (metres).
        refine_tol_deg: Orientation tolerance that ends refinement (degrees).
        settle_s: Wait before measuring so the controller settles (seconds).
        min_hand_gap_m: Abort if the push goal is closer than this to the right gripper (metres).
        allow_cross: Allow a goal on the right half (y < 0) when the right hand is across the midline.
        pause: Wait for SPACE between stages ('q' stops with the pose held).
        leave_held: Exit without ``stop()`` so the pose stays held.
        dry_run: Plan all stages from the URDF neutral pose without connecting (no force, full push planned).
        use_torso: Also move the torso in the IK. WARNING: this moves the right hand hooked on the box too.
        auto_torso: Retry a stage with the torso if the arm alone covers less than half of it (same warning).
        torso_posture_cost: Posture cost on torso joints; higher = torso moves less (arms use 1e-3).
        torso_max_vel: Torso joint speed cap (rad/s).
        max_joint_vel: Arm joint speed cap (rad/s); a stage's stream time is stretched to respect it.
        anchor_setpoints: Read the sim's held targets and anchor the streams there (recommended).
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    if max_push_cm <= 0 or force_threshold_n <= 0 or start_back_cm < 0 or backoff_cm < 0 or fallback_cm < 0:
        raise ValueError("max_push_cm and force_threshold_n must be > 0; start_back_cm, backoff_cm, fallback_cm >= 0")
    if use_torso or auto_torso:
        logger.warning("Torso motion also moves the RIGHT hand, which is hooked on the box; use with care.")
    gate = ForceGate(force_threshold_n, confirm_ticks, tare_samples, force_log_period_s, fallback_cm,
                     metric=metric, min_push_cm=min_push_cm, warmup_s=warmup_s)
    plan = ForcePushPlan(
        lower_cm=lower_cm, start_back_cm=start_back_cm, max_push_cm=max_push_cm, backoff_cm=backoff_cm,
        flip_blade=flip_blade, min_hand_gap_m=min_hand_gap_m, allow_cross=allow_cross, gate=gate, recover_z=recover_z,
    )
    stages = build_stages(
        plan, lift_duration=lift_duration, align_duration=align_duration, push_duration=push_duration,
        step_m=step_m, step_rad=np.deg2rad(step_deg), refine=refine, align_refine=align_refine,
        refine_tol_m=refine_tol_m, refine_tol_rad=np.deg2rad(refine_tol_deg),
    )
    logger.info("Stages: " + " -> ".join(s.label for s in stages))
    ok = run_stages(
        "left", stages, use_torso=use_torso, torso_posture_cost=torso_posture_cost, torso_max_vel=torso_max_vel,
        max_joint_vel=max_joint_vel, settle_s=settle_s, anchor_setpoints=anchor_setpoints, auto_torso=auto_torso,
        ignore_foreign_commands=ignore_foreign_commands, dry_run=dry_run, leave_held=leave_held,
        prepare=plan.prepare, pause=_pause if pause else None,
    )
    if gate.stop_force_n is not None:
        logger.info(f"Contact stop: {gate.metric} dF {gate.stop_force_n:.2f} N after {gate.travel_m * 100:.1f} cm "
                    f"(peak {gate.peak_n:.2f} N, rest noise {gate.noise_n:.2f} N).")
    elif gate.armed:
        logger.warning(f"No contact felt (peak {gate.metric} dF {gate.peak_n:.2f} N < {force_threshold_n:.1f} N); "
                       "the push ended at its maximum.")
    return ok


if __name__ == "__main__":
    tyro.cli(main)
