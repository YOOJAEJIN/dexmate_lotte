# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Release step: retract both hands to the chest while the right hand pushes
the placed box (purple center) from world +X toward world −X.

Pipeline (lotte_chilsung), after ``approach_purple_to_red_bimanual.py``::

    (open both hands via omniteleop — the box must be free before this!)
    python retract_and_push.py --dry-run     # plan + report, no motion
    python retract_and_push.py               # execute

Phases (both arms move at the same time; EE orientations stay fixed):

  1. disengage : both hands back toward the chest (robot base −X).
  2. around    : right hand swings to the world +X side of the box, up to
                 purple height, and a little deeper (base +X); left keeps
                 retracting and rises slightly.
  3. push      : right hand moves world −X through ``purple_x + push_end``,
                 pressing the box against the conveyor stack; left keeps
                 retracting.
  4. retract   : both hands finish retracting to ``--retract`` total.

Frames: "chest" motions are in the **robot base** frame; the push is along
the **world** X axis. The base→world transform is derived from the sim
marker files, which contain both robot-base and world coordinates.

The torso is unlocked by default: from the current folded-elbow hold the
left arm cannot retract at all with the torso frozen (IK fails within 5 cm),
while a ~10° torso motion makes the whole sequence feasible.
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
    MAX_JOINT_JUMP,
    RIGHT_ARM_JOINTS,
    connect_bot,
    create_motion_manager,
    disconnect_leave_held,
    execute_trajectory,
    max_adjacent_jump,
)
from approach_purple_to_red import (  # noqa: E402
    _DEFAULT_PURPLE_CANDIDATES,
    _DEFAULT_TIP_CANDIDATES,
    _resolve_existing,
)
from squeeze_inward import _solve_dual_ee  # noqa: E402

# Collision margins to try, in order, when building the MotionManager from a
# pose that is already near a self-collision pair.
_MARGIN_FALLBACKS = (0.01, 0.005)


def _load_base_world(path: Path, label: str) -> tuple[np.ndarray, np.ndarray]:
    base = world = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split()
        if not s:
            continue
        if s[0] == "#" and len(s) >= 5 and s[1] == "world":
            world = np.array([float(x) for x in s[2:5]])
        elif not line.startswith("#") and len(s) >= 3:
            base = np.array([float(x) for x in s[:3]])
    if base is None or world is None:
        raise ValueError(
            f"{label} file {path} lacks base and/or '# world' line; "
            "restart sim_main.py so both frames are written."
        )
    logger.info(f"Loaded {label}: base {np.round(base, 4)}  world {np.round(world, 4)}")
    return base, world


class WorldFrame:
    """Level base→world transform (yaw + translation) from two marker pairs."""

    def __init__(self, a_b, a_w, b_b, b_w):
        db, dw = np.asarray(b_b) - np.asarray(a_b), np.asarray(b_w) - np.asarray(a_w)
        if np.linalg.norm(db[:2]) < 0.02:
            raise ValueError(
                "Purple and red markers are <2 cm apart in XY; cannot derive the "
                "base→world yaw. Use --world-yaw-deg to give it explicitly."
            )
        self.yaw = float(np.arctan2(dw[1], dw[0]) - np.arctan2(db[1], db[0]))
        self._set_rot()
        self.t = np.asarray(a_w) - self.R @ np.asarray(a_b)

    @classmethod
    def from_yaw(cls, yaw_deg: float, a_b, a_w) -> "WorldFrame":
        obj = cls.__new__(cls)
        obj.yaw = float(np.deg2rad(yaw_deg))
        obj._set_rot()
        obj.t = np.asarray(a_w) - obj.R @ np.asarray(a_b)
        return obj

    def _set_rot(self) -> None:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_world(self, p_b) -> np.ndarray:
        return self.R @ np.asarray(p_b, float) + self.t

    def to_base(self, p_w) -> np.ndarray:
        return self.R.T @ (np.asarray(p_w, float) - self.t)

    def vec_to_base(self, v_w) -> np.ndarray:
        return self.R.T @ np.asarray(v_w, float)


def _track(
    mm,
    phases: list[tuple[str, np.ndarray, np.ndarray]],
    step_m: float,
) -> tuple[list[dict], list[tuple[str, float]]]:
    """Track absolute base-frame EE position targets per phase (fixed rotation).

    Returns the config list and per-phase completion fractions. Stops at the
    first phase that cannot be completed (keeping the feasible prefix).
    """
    poses = mm.fk(["L_ee", "R_ee"])
    rot_l = poses["L_ee"].rotation.copy()
    rot_r = poses["R_ee"].rotation.copy()
    lp = poses["L_ee"].translation.copy()
    rp = poses["R_ee"].translation.copy()
    configs: list[dict] = []
    report: list[tuple[str, float]] = []
    for label, l_tgt, r_tgt in phases:
        span = max(np.linalg.norm(l_tgt - lp), np.linalg.norm(r_tgt - rp))
        n = max(int(np.ceil(span / max(step_m, 1e-4))), 1)
        done = 0
        for i in range(1, n + 1):
            a = i / n
            tl = pin.SE3(rot_l, (1 - a) * lp + a * l_tgt)
            tr = pin.SE3(rot_r, (1 - a) * rp + a * r_tgt)
            q_before = dict(mm.get_joint_pos_dict())
            if not _solve_dual_ee(mm, tl, tr):
                break
            q_next = dict(mm.get_joint_pos_dict())
            if max(abs(q_next[k] - q_before[k]) for k in q_before) > MAX_JOINT_JUMP:
                mm.set_joint_pos(q_before)
                break
            configs.append(q_next)
            done = i
        frac = done / n
        report.append((label, frac))
        cur = mm.fk(["L_ee", "R_ee"])
        lp, rp = cur["L_ee"].translation.copy(), cur["R_ee"].translation.copy()
        if frac < 0.999:
            break
    return configs, report


def main(
    tip_file: Path | None = None,
    purple_file: Path | None = None,
    world_yaw_deg: float | None = None,
    retract: float = 0.20,
    disengage: float = 0.05,
    push_start: float = 0.15,
    push_end: float = 0.04,
    push_height: float = 0.0,
    push_reach: float = 0.10,
    left_rise: float = 0.03,
    step_m: float = 0.01,
    duration: float = 8.0,
    control_hz: float = 100.0,
    torso_damping: float = 5.0,
    lock_torso: bool = False,
    collision_margin: float = 0.015,
    dry_run: bool = False,
    leave_held: bool = True,
) -> None:
    """Retract both hands to the chest while the right hand pushes the box −X.

    Args:
        tip_file: Red tip file (base + world lines). Default: sim file.
        purple_file: Purple center file (base + world lines). Default: sim file.
        world_yaw_deg: Override base→world yaw (deg) instead of deriving it.
        retract: Total retraction of each hand toward the chest, base −X (m).
        disengage: Phase-1 retraction before the right hand swings out (m).
        push_start: Right EE world-X start of the push, relative to purple (m).
            Must clear the box +X face (~+0.07) plus hand width.
        push_end: Right EE world-X end of the push, relative to purple (m).
            Below the face offset means pressing into the box (held by the stack).
        push_height: Right EE world-Z during the push, relative to purple (m).
        push_reach: Extra base +X (toward the box) for the right hand during
            the push so the fingers act deeper inside the box's near edge (m).
        left_rise: Left EE rise during phases 2–3 (m).
        step_m: IK waypoint spacing (m).
        duration: Stream time for the full path (s); shortened for a prefix.
        control_hz: Command rate.
        torso_damping: Torso IK damping when unlocked (higher = less torso).
        lock_torso: Freeze torso (arms only). From the folded-elbow hold this
            is expected to fail early — kept for experiments.
        collision_margin: Pink self-collision barrier (m). Falls back to
            smaller values if the current pose is already closer than this.
        dry_run: Plan and report, do not move.
        leave_held: Exit without ``stop()``.
    """
    tip_path = _resolve_existing(tip_file, _DEFAULT_TIP_CANDIDATES, "red tip")
    purple_path = _resolve_existing(purple_file, _DEFAULT_PURPLE_CANDIDATES, "purple center")
    red_b, red_w = _load_base_world(tip_path, "red tip")
    pur_b, pur_w = _load_base_world(purple_path, "purple center")
    if world_yaw_deg is None:
        wf = WorldFrame(pur_b, pur_w, red_b, red_w)
    else:
        wf = WorldFrame.from_yaw(world_yaw_deg, pur_b, pur_w)
    chest_dir_b = np.array([-1.0, 0.0, 0.0])  # robot base −X
    push_dir_b = wf.vec_to_base([-1.0, 0.0, 0.0])  # world −X in base frame
    logger.info(
        f"base→world yaw {np.rad2deg(wf.yaw):.1f} deg; "
        f"world −X (push) in base = {np.round(push_dir_b, 3)}; "
        f"chest dir (base −X) in world = {np.round(wf.R @ chest_dir_b, 3)}"
    )
    logger.warning(
        "Both hands must already be OPEN (omniteleop). "
        "This script does not command the grippers."
    )

    bot = connect_bot(auto_shutdown=False)
    do_shutdown = not leave_held
    try:
        hb = getattr(bot, "heartbeat", None)
        if hb is not None and hasattr(hb, "pause"):
            hb.pause()
            logger.info("Heartbeat paused for IK planning.")

        try:
            mm = torso_joints = None
            for margin in (collision_margin, *[m for m in _MARGIN_FALLBACKS if m < collision_margin]):
                try:
                    mm, torso_joints, _ = create_motion_manager(
                        bot,
                        torso_damping=torso_damping,
                        lock_torso=lock_torso,
                        collision_margin=margin,
                    )
                    logger.info(f"MotionManager ready (collision_margin={margin}).")
                    break
                except RuntimeError as exc:
                    if "Collision barrier" not in str(exc):
                        raise
                    logger.warning(f"collision_margin={margin} too large for current pose; retrying smaller.")
            if mm is None:
                logger.error("Could not build MotionManager at any collision margin. Not moving.")
                return

            q_init = dict(mm.get_joint_pos_dict())
            poses = mm.fk(["L_ee", "R_ee"])
            l0 = poses["L_ee"].translation.copy()
            r0 = poses["R_ee"].translation.copy()
            l0_w, r0_w = wf.to_world(l0), wf.to_world(r0)
            logger.info(
                "Current (world):\n"
                f"  purple {np.round(pur_w, 3)}   red {np.round(red_w, 3)}\n"
                f"  L_ee   {np.round(l0_w, 3)}   rel purple {np.round(l0_w - pur_w, 3)}\n"
                f"  R_ee   {np.round(r0_w, 3)}   rel purple {np.round(r0_w - pur_w, 3)}"
            )

            # --- phase targets (base frame, absolute) ---
            fwd_b = -chest_dir_b  # toward the box
            up = np.array([0.0, 0.0, 1.0])
            # Phase 1: disengage
            l1 = l0 + disengage * chest_dir_b
            r1 = r0 + disengage * chest_dir_b
            # Phase 2: right around to world +X side at purple height, deeper
            r2_w = wf.to_world(r1 + push_reach * fwd_b)
            r2_w[0] = pur_w[0] + push_start
            r2_w[2] = pur_w[2] + push_height
            r2 = wf.to_base(r2_w)
            l2 = l1 + 0.03 * chest_dir_b + left_rise * up
            # Phase 3: push along world −X
            r3_w = r2_w.copy()
            r3_w[0] = pur_w[0] + push_end
            r3 = wf.to_base(r3_w)
            l3 = l2 + 0.04 * chest_dir_b
            # Phase 4: finish retract (total `retract` along chest dir from start)
            l4 = l0 + retract * chest_dir_b + left_rise * up
            r4 = r3.copy()
            r4[0] = r0[0] - retract  # base −X (chest) to the same depth as left
            phases = [
                ("disengage", l1, r1),
                ("around", l2, r2),
                ("push", l3, r3),
                ("retract", l4, r4),
            ]
            for label, lt, rt in phases:
                logger.info(
                    f"  target {label:9s} L(world) {np.round(wf.to_world(lt), 3)}  "
                    f"R(world) {np.round(wf.to_world(rt), 3)}"
                )

            configs, report = _track(mm, phases, step_m)
        finally:
            if hb is not None and hasattr(hb, "resume"):
                try:
                    hb.resume()
                    logger.info("Heartbeat resumed.")
                except Exception:
                    pass

        summary = "  ".join(f"{lab}:{100 * frac:.0f}%" for lab, frac in report)
        complete = bool(report) and all(frac >= 0.999 for _, frac in report) and len(report) == 4
        if not configs:
            logger.error(f"No feasible motion from the current pose ({summary}). Not moving.")
            return
        jump = max_adjacent_jump([q_init] + configs)
        q_last = configs[-1]
        torso_delta = {
            k: round(float(q_last[k] - q_init[k]), 2)
            for k in q_init
            if k.startswith("torso_")
        }
        mm.set_joint_pos(q_last)
        fin = mm.fk(["L_ee", "R_ee"])
        # FK results alias pinocchio's internal buffers — copy before resetting.
        fin_l = wf.to_world(fin["L_ee"].translation.copy())
        fin_r = wf.to_world(fin["R_ee"].translation.copy())
        mm.set_joint_pos(q_init)
        logger.info(
            f"[{'COMPLETE' if complete else 'PARTIAL'}] {summary}\n"
            f"  {len(configs)} configs, max adjacent jump {jump:.2f} rad, torso Δ {torso_delta}\n"
            f"  final L(world) {np.round(fin_l, 3)}  R(world) {np.round(fin_r, 3)}"
        )
        if not complete:
            logger.warning("Only a prefix of the motion is feasible; it will stop at the last feasible point.")

        if dry_run:
            logger.info("Dry run: not moving. Pose left held.")
            return

        logger.info(f"Executing {len(configs)} configs over {duration:.1f}s...")
        execute_trajectory(bot, torso_joints, q_init, configs, duration, control_hz)

        names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + torso_joints
        q = np.array([q_last[n] for n in names], dtype=float)
        rate = RateLimiter(control_hz)
        for _ in range(max(int(0.2 * control_hz), 5)):
            bot.left_arm.set_joint_pos(q[:7])
            bot.right_arm.set_joint_pos(q[7:14])
            if torso_joints:
                bot.torso.set_joint_pos(q[14:])
            rate.sleep()
        logger.info("Retract-and-push completed; final pose left held.")
    except (FileNotFoundError, ValueError) as exc:
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
