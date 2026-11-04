# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Return to the home pose: raise both arms (shoulder ``j1`` by 40 deg, hand posture held), then bring the arms home
right arm first, left arm next, while the base yaws back to its start heading; torso / head follow last.

Step 14 of the Chilsung sequence (2026-09-27, order specified by the user; the 2026-09-26 version -- no raise, right
arm first with the torso, base turning from the start -- is kept as ``_return_home_prev_20260926.py``). After
``13.hands_push_left.py`` both open hands are stretched out toward the block, one above the other, both near the
robot's midline (gripper bases ~14 cm apart, blades pointing forward).

1. **raise** -- ``L_arm_j1`` / ``R_arm_j1`` turn ``--raise-deg`` (40) in the raise direction (left decreases, right
   increases; ``_j1_up.RAISE_SIGN``) while all other arm joints hold, so both arms swing up with the hand posture
   unchanged (their separation is preserved since both shoulders turn about parallel axes). BEFORE that (same
   stream, first) the torso moves the whole upper body straight back by ``--torso-back-cm`` (10; ``--torso-up-cm``
   for an upward component) with no pitch change -- the three torso joints are all pitch, so (x, z, pitch) is solved exactly on the
   URDF FK -- which retreats both grippers from the block with their orientation kept (user request 2026-09-27; a
   Pink-IK pull-back of the arms alone is impossible here: the left elbow sits at its extension limit, a singular
   pose from which IK cannot start the retraction). ``--no-raise-arms`` skips this stage.
2. **base turn** (``--turn-start clear``, default): the wheels steer to the in-place angles from the very start
   (~7 s, the base does not move), and the rotation is released the moment FK of the measured joints shows every blade
   point ``--turn-clear-cm`` (10) above the block top (73 % of the raise on 2026-09-27) -- the arms only have to
   clear the box, not be fully up; ``home`` = rotate only when the raise has finished. Once the arms are up they come
   home: first both arms' ``j2`` / ``j3`` go home while the arms are still up (``--spread-duration`` 2 s; the hands part sideways, blade gap
   8 -> 18+ cm), then the arms swing down one after the other in the order ``--arm-order`` (default ``right,left``),
   the second starting when the first has covered ``1 - --arm-overlap`` (0.5 -> half) of its motion and moving
   faster (``--second-arm-vel`` 0.6 rad/s, ``--second-duration`` 4 s vs 0.4 rad/s / 6 s for the first). Within each
   swing the elbow and wrist fold in during the first half (``--elbow-lead`` 0.5) so the blade stays close to the
   body on the way down. Without the
   spread only left-first clears (right-first blade gap 0.5-8 cm by FK, 2026-09-27). All joints of an arm move
   together (``--elbow-lead`` 1.0; an early elbow fold, 0.6, swings the blade into the other hand). The wheels need
   ~7 s of steering before the base actually turns, so the arms are mostly down by then; the FK check keeps the
   block at its present place (conservative), the tips never pass their present ~0.54 m.
3. **torso / head home with the second arm** (``--torso-with second``, default since 2026-09-28: they start when
   the second arm starts swinging, ~3 s after the first, and finish with it; ``arms`` = spread over the whole arm
   stage, the 2026-09-27 default; ``after`` = only once the arms are home). The torso's home pose leans forward
   and carries the still-raised blade tips up to ~0.6-0.85 m ahead; that is acceptable only because the base is
   already turning the block away, so the FK safety check places the block where it will be at every instant
   (rotation assumed from ``--turn-model-steer-s`` 7 s after the turn thread starts, at ``--turn-model-rate-deg-s``
   6 deg/s) and aborts if a blade would still come within ``--block-margin-cm`` of it.

``--torso-first`` is the alternative order: after the raise the joints wait until the base has turned
``--joints-after-turn-deg`` (45), then torso / head go home, then the arms (the block must be out of the way first).

Home joints come from the simulator's robot config (``robots/dexmate.py``, the uncommented ``init_joint_pos``
dict of ``VEGA_1P_CFG_VERTICAL_GRIPPER``; ``--home-file`` to point elsewhere, a copy is kept here as fallback).
Grippers are not commanded (they stay open after ``11.open_hands.py``).

**Base turn**: in place until the base faces the card stack the box was picked from (``--aim stack``, default: base
+X pointed at the centre of the four card-stack edge markers, read live from the sim each tick like step 8's
``--aim tag``; ``--aim-offset-deg`` to bias it; ``--aim yaw`` = the fixed world heading ``--home-yaw-deg``, 0 = the
spawn heading, also the fallback when the edge-marker file is missing). Step 8 turned ~+90 deg toward the conveyor
and step 10 a little further, so ~-100 deg come back. Closed loop with ``_base_turn.turn_in_place`` in a background
thread started at ``--turn-start`` (``home`` = once the arms are up, default; ``raise`` = right away; ``end`` = after
the joints are home). The wheels need ~7 s of steering before the base actually turns, then ~8 deg/s. After the
wheels stop the base coasts ~1 deg, so the thread waits until the heading is still (``--settle-s``) and turns again
only if the residual then exceeds ``--correct-above-deg`` (4; ``--max-turns`` 2 in total) -- a re-turn costs ~9 s,
5 s of it re-steering the wheels, so residuals of 2-4 deg are accepted (2026-09-27 run: 1.4 -> 2.2 deg after
settling triggered two useless re-turns). ``turn_in_place`` now brakes by re-sending
zero wheel speed for 0.5 s and ``stop()`` five times (2026-09-27: a single stop message next to the 50 Hz arm stream
is suspected of being lost, the sim then holds the last wheel target and keeps spinning). The console log is mirrored
to ``logs/return_home_<time>.log`` (``--no-log-file``).

**Safety check** (``--no-fk-check`` to skip): the whole planned joint timeline is run through the URDF FK; the run
is aborted if the two finger blades come closer than ``--min-hand-gap-cm`` (8 cm) or a blade point would come within
``--block-margin-cm`` (3) of the block, modelled as an oriented box around the purple marker: half thickness 5.25 cm
along the conveyor-box face normal (the block was pushed flush against that face; a residual block yaw widens it),
half width 20 cm along the face, half height 20 cm; only when the markers are fresh -- the purple marker is read
three times and the median taken, since the sim rewrites it every step. In the default order the block is
checked where it is now (the turn only moves it away); in the torso-first order it is placed where it will be once
the base has turned ``--joints-after-turn-deg`` (footprint widened to a 40 cm square).

Streams: ``_pink_arm.stream`` (min-jerk, leash-paced, per-joint speed caps, anchored at the sim's held setpoints so
nothing first drops to its sagged measured angle); the raise is one stream, spread + arms + torso / head one
timeline in a second (torso-first: torso / head and arms as two streams).

Example (block pushed home, hands open, omniteleop follower stopped, ROBOT_NAME / ZENOH_CONFIG exported)::

    python 14.return_home.py --plan-only          # joint deltas, FK check, yaw needed, expected times; no motion
    python 14.return_home.py                      # raise -> [base turn + spread -> right arm -> left arm] -> torso/head
    python 14.return_home.py --aim yaw --home-yaw-deg 0   # fixed world heading instead of aiming at the card stack
    python 14.return_home.py --aim-offset-deg -5  # face 5 deg right of the stack centre
    python 14.return_home.py --arm-order left,right
    python 14.return_home.py --no-turn-base       # joints only
    python 14.return_home.py --turn-start home    # base turn only once the arms are fully up (previous default)
    python 14.return_home.py --turn-clear-cm 20   # wait until the blades are 20 cm above the block top
    python 14.return_home.py --torso-first        # wait 45 deg of turn, torso/head first, then the arms
    python 14.return_home.py --torso-first --joints-after-turn-deg 90
    python 14.return_home.py --no-raise-arms      # arms already up: skip the raise
    python 14.return_home.py --arm-overlap 0      # second arm only after the first has finished
"""

from __future__ import annotations

import ast
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from _base_turn import WHEEL_GAIN_SIM, measure_rtf, turn_in_place, wrap  # noqa: E402
from _block_geom import BLOCK_HALF_T, block_yaw_vs_face  # noqa: E402
from _j1_up import J1_LIMIT_RAD, RAISE_SIGN  # noqa: E402
from _pink_arm import (  # noqa: E402
    COMMAND_TOPICS,
    TORSO_MAX_VEL,
    anchor_offsets,
    detect_foreign_commands,
    read_setpoints,
    stream,
)
from _pink_common import ARM_JOINTS, EE, build_ik, connect, disconnect_leave_held, read_joint_pos  # noqa: E402
from _sim_markers import (  # noqa: E402
    DEX_ROOT,
    EDGE_MARKERS_FILE_NAME,
    PURPLE_FILE_NAME,
    TIP_FILE_NAME,
    base_yaw_world,
    box_face_normal_base,
    find_sim_dir,
    load_named_pairs,
    load_xyz_pair,
    world_fixed_pairs,
)

LOG_DIR = Path(__file__).resolve().parent / "logs"

HEAD_JOINTS = [f"head_j{i}" for i in range(1, 4)]
COMPONENT_JOINTS = {
    "left_arm": ARM_JOINTS["left"],
    "right_arm": ARM_JOINTS["right"],
    "torso": ARM_JOINTS["torso"],
    "head": HEAD_JOINTS,
}
ARM_COMPONENTS = ("left_arm", "right_arm")
SIDE_OF = {"left_arm": "left", "right_arm": "right"}
ARM_OF = {v: k for k, v in SIDE_OF.items()}
J1 = 0
SHOULDER_JOINTS = [0, 1, 2]       # *_arm_j1, j2, j3 -- take the whole arm stage (no spread)
SPREAD_JOINTS = [1, 2]            # *_arm_j2, j3 -- go home first while the arms are still up (spread stage)
ELBOW_WRIST_JOINTS = [3, 4, 5, 6]  # *_arm_j4 .. j7 -- finish within --elbow-lead of the arm stage
TIP_AHEAD_M = 0.2206  # finger blade tip ahead of the gripper base along EE +Z (2FPA 2nd gen, was 0.188 with 1st gen; see 13.hands_push_left.py)
BLOCK_HALF_W = 0.20  # block half width along the box face (robot y when squared up), see _block_geom
BLOCK_HALF_H = 0.20  # block half height (35 cm side upright after step 9, with a little slack)
KNOT_HZ = 25.0
DEFAULT_HOME_FILE = DEX_ROOT / "unitree_sim_isaaclab_260918" / "robots" / "dexmate.py"

# Copy of robots/dexmate.py VEGA_1P_CFG_VERTICAL_GRIPPER init_joint_pos (2026-09-27); used if the file
# cannot be parsed.
HOME_JOINT_POS_FALLBACK: dict[str, float] = {
    'L_arm_j1': 1.5745360851287842, 'L_arm_j2': -4.601889304467477e-05, 'L_arm_j3': -0.0001406389201292768,
    'L_arm_j4': -1.5620099306106567, 'L_arm_j5': -0.00018947907665278763, 'L_arm_j6': 0.004862525034695864,
    'L_arm_j7': 0.0001019211849779822,
    'R_arm_j1': -1.5743721723556519, 'R_arm_j2': 0.00020777805184479803, 'R_arm_j3': 0.00020511656475719064,
    'R_arm_j4': -1.5603846311569214, 'R_arm_j5': -0.00010557365021668375, 'R_arm_j6': -0.004143048543483019,
    'R_arm_j7': 3.1130890420172364e-05,
    'head_j1': -0.08560948818922043, 'head_j2': 0.000296128389891237, 'head_j3': 0.00010670589108485729,
    'torso_j1': 0.083, 'torso_j2': 0.4230404496192932, 'torso_j3': 0.11368651688098907,
}


def load_home_joint_pos(path: Path) -> dict[str, float]:
    """Parse the home ``init_joint_pos = {...}`` dict literal from ``robots/dexmate.py``.

    The file has several ``init_joint_pos`` occurrences (a ``{".*": 0.0}`` default in ``_init_state_cfg``,
    a commented-out older pose, the live one); full-line comments are dropped and, among the remaining
    dict literals, the largest one that names arm and torso joints is used.
    """
    text = Path(path).read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    best: dict[str, float] | None = None
    for m in re.finditer(r"init_joint_pos\s*=\s*\{", code):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = ast.literal_eval(code[start:i + 1])
                    except (ValueError, SyntaxError):
                        d = None
                    if isinstance(d, dict) and "L_arm_j1" in d and "torso_j1" in d and (best is None or len(d) > len(best)):
                        best = {str(k): float(v) for k, v in d.items()}
                    break
    if best is None:
        raise ValueError(f"{path}: no init_joint_pos dict with arm and torso joints found")
    return best


# ---------------------------------------------------------------------------
# Timeline: per component, which joints move in which stage, sampled at KNOT_HZ
# ---------------------------------------------------------------------------
def stage_time(path_rad: float, min_duration: float, cap: float) -> float:
    return float(max(min_duration, path_rad / max(cap, 1e-3)))


def build_timeline(
    start: dict[str, np.ndarray], target: dict[str, np.ndarray], offsets: dict[str, np.ndarray],
    stages: dict[str, tuple[float, float]], joint_stage: dict[str, list[tuple[str, list[int]]]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Knots (n+1, nj) per component. ``stages`` = {name: (t_start, duration)}; ``joint_stage[c]`` lists
    (stage name, joint indices) for component ``c``. Joints not listed stay at ``start`` (+ offset). The anchor
    offset (setpoint - measurement) of a moving joint fades out with that joint's own progress, so its last knot
    is the target value itself."""
    t_end = max(t0 + td for t0, td in stages.values())
    n_knots = max(int(np.ceil(t_end * KNOT_HZ)), 2)
    times = np.linspace(0.0, t_end, n_knots + 1)
    groups: dict[str, np.ndarray] = {}
    for c in start:
        q = np.tile(start[c] + offsets[c], (n_knots + 1, 1))
        for stage, jj in joint_stage.get(c, []):
            t0, td = stages[stage]
            frac = np.clip((times - t0) / max(td, 1e-6), 0.0, 1.0)[:, None]
            q[:, jj] = start[c][jj] + offsets[c][jj] * (1.0 - frac) + (target[c][jj] - start[c][jj]) * frac
        groups[c] = q
    return times, groups


# ---------------------------------------------------------------------------
# FK check of the planned timeline (URDF via Pink / pinocchio; no IK)
# ---------------------------------------------------------------------------
class HandFK:
    """Gripper base + blade tip (robot base frame) for given arm / torso joints."""

    def __init__(self):
        self.cfg, _tasks, _solver, self.idx = build_ik()
        self.model = self.cfg.model

    def set(self, q_by_comp: dict[str, np.ndarray]) -> None:
        q = self.cfg.q.copy()
        for c, v in q_by_comp.items():
            side = SIDE_OF.get(c, c)
            if side in self.idx:
                q[self.idx[side]] = v
        self.cfg.update(np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit))

    def hands(self, q_by_comp: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """side -> (gripper base, blade tip)."""
        self.set(q_by_comp)
        out = {}
        for side in ("left", "right"):
            T = self.cfg.get_transform_frame_to_world(EE[side])
            out[side] = (T.translation.copy(), T.translation + TIP_AHEAD_M * T.rotation[:, 2])
        return out


def solve_torso_translation(fk: HandFK, arms: dict[str, np.ndarray], torso0: np.ndarray, dx: float, dz: float,
                            iters: int = 8) -> tuple[np.ndarray, np.ndarray, bool]:
    """Torso joints that translate ``arm_center`` (the whole upper body, arms rigid with it) by ``(dx, 0, dz)`` in the
    robot base frame with NO pitch change -- the three torso joints are all pitch, so (x, z, pitch) is a square
    system solved by Newton iterations on the URDF FK. Returns ``(q_torso, achieved (dx, dz), within_limits)``;
    the joint limits are enforced by clipping, so ``achieved`` may fall short."""
    m = fk.model
    idx_t = fk.idx["torso"]
    lo, hi = m.lowerPositionLimit[idx_t], m.upperPositionLimit[idx_t]

    def state(q_t: np.ndarray) -> np.ndarray:
        fk.set({**arms, "torso": q_t})
        T = fk.cfg.get_transform_frame_to_world("arm_center")
        pitch = float(np.arctan2(-T.rotation[2, 0], np.hypot(T.rotation[0, 0], T.rotation[1, 0])))
        return np.array([T.translation[0], T.translation[2], pitch])

    s0 = state(torso0)
    goal = s0 + np.array([dx, dz, 0.0])
    q = torso0.copy()
    eps = 1e-4
    for _ in range(iters):
        s_now = state(q)
        res = goal - s_now
        if np.abs(res).max() < 1e-5:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            qq = q.copy()
            qq[j] += eps
            J[:, j] = (state(qq) - s_now) / eps
        try:
            q = np.clip(q + np.linalg.solve(J, res), lo, hi)
        except np.linalg.LinAlgError:
            break
    s_end = state(q)
    inside = bool(np.all(q > lo + 1e-6) and np.all(q < hi - 1e-6))
    return q, s_end[:2] - s0[:2], inside


def segment_distance(a0, a1, b0, b1, n: int = 10) -> float:
    """Approximate closest distance between two segments (sampled)."""
    A = [a0 + t * (a1 - a0) for t in np.linspace(0.0, 1.0, n)]
    B = [b0 + t * (b1 - b0) for t in np.linspace(0.0, 1.0, n)]
    return float(min(np.linalg.norm(a - b) for a in A for b in B))


def box_clearance(p: np.ndarray, centre: np.ndarray, half: np.ndarray, axis: np.ndarray | None = None) -> float:
    """Per-axis clearance of a point from a box: the largest of (|d_i| - half_i) with ``d`` the point relative to the
    box centre expressed in the box frame -- ``axis`` is the horizontal unit vector of the box's first (thickness)
    axis, the second is horizontal and perpendicular, the third is +Z; ``axis`` None = base axes. Negative = inside
    the box, positive = at least that far outside along some axis."""
    d = p - centre
    if axis is not None:
        u = np.asarray(axis, dtype=float)
        w = np.array([-u[1], u[0], 0.0])
        d = np.array([float(d @ u), float(d @ w), float(d[2])])
    return float(np.max(np.abs(d) - half))


class BlockModel:
    """The card block as an oriented box: centre, horizontal thickness axis (the box face normal -- the block was
    pushed flush against the conveyor box face along it), half sizes (thickness, width, height). ``turned(angle)``
    gives the same block after the base has yawed by ``angle`` (objects turn the opposite way in the base frame)."""

    def __init__(self, centre: np.ndarray, axis: np.ndarray | None, half: np.ndarray):
        self.centre = np.asarray(centre, dtype=float)
        self.axis = None if axis is None else np.asarray(axis, dtype=float) / max(np.linalg.norm(axis), 1e-9)
        self.half = np.asarray(half, dtype=float)

    def turned(self, base_yaw_change: float) -> "BlockModel":
        if abs(base_yaw_change) < 1e-9:
            return self
        c = rotate_about_z(self.centre, -base_yaw_change)
        if self.axis is not None:
            return BlockModel(c, rotate_about_z(self.axis, -base_yaw_change), self.half)
        # unknown orientation: keep the base-aligned box but widen it to a square footprint once it has turned
        return BlockModel(c, None, np.array([max(self.half[0], self.half[1])] * 2 + [self.half[2]]))

    def clearance(self, p: np.ndarray) -> float:
        return box_clearance(p, self.centre, self.half, self.axis)

    def describe(self) -> str:
        return (f"centre {np.round(self.centre, 2)}, thickness axis {np.round(self.axis, 2) if self.axis is not None else 'base x (unknown)'}, "
                f"half sizes {np.round(self.half, 3)} m")


def check_path_fk(fk: HandFK, groups: dict[str, np.ndarray], block: "BlockModel | None" = None, every: int = 2,
                  block_track: "list[BlockModel] | None" = None) -> dict[str, float]:
    """Min blade-to-blade gap, max forward reach / height of any blade point over the timeline and, with ``block``,
    the minimum clearance of any blade point from the block box. ``block_track`` (one BlockModel per knot) overrides
    the fixed block: the block is checked where it is at that knot's time (base turning underneath)."""
    n = len(next(iter(groups.values())))
    res = {"gap_min": np.inf, "x_max": -np.inf, "z_max": -np.inf, "x_min": np.inf, "block_clear": np.inf, "block_clear_knot": -1,
           "block_series": []}  # (knot, min clearance at that knot)
    for k in range(0, n, every):
        hands = fk.hands({c: Q[k] for c, Q in groups.items()})
        (lb, lt), (rb, rt) = hands["left"], hands["right"]
        res["gap_min"] = min(res["gap_min"], segment_distance(lb, lt, rb, rt))
        for p in (lb, lt, rb, rt):
            res["x_max"] = max(res["x_max"], float(p[0]))
            res["x_min"] = min(res["x_min"], float(p[0]))
            res["z_max"] = max(res["z_max"], float(p[2]))
        blk = block_track[min(k, len(block_track) - 1)] if block_track is not None else block
        if blk is not None:
            c_k = np.inf
            for a, b in ((lb, lt), (rb, rt)):
                for q in np.linspace(0.0, 1.0, 5):  # a few points along each blade, not just the ends
                    c_k = min(c_k, blk.clearance(a + q * (b - a)))
            res["block_series"].append((k, float(c_k)))
            if c_k < res["block_clear"]:
                res["block_clear"], res["block_clear_knot"] = float(c_k), k
    return res


def block_violation(series: list[tuple[int, float]], margin_m: float, approach_tol_m: float = 0.01) -> tuple[bool, float, int, int]:
    """Judge the per-knot block clearances. The hands may START in contact (they have just pushed the block), so the
    rule is: once the blades have moved ``margin_m`` clear of the block they must never come back within it, and
    before that they must not get closer than the start clearance minus ``approach_tol_m``. Returns
    ``(violated, worst_clearance, worst_knot, first_clear_knot)`` (first_clear_knot -1 if never clear)."""
    if not series:
        return False, np.inf, -1, -1
    c0 = series[0][1]
    escaped = False
    first_clear = -1
    violated, worst, worst_k = False, np.inf, -1
    for k, c in series:
        if not escaped and c >= margin_m:
            escaped, first_clear = True, k
        bad = (c < margin_m) if escaped else (c < c0 - approach_tol_m)
        if bad and c < worst:
            violated, worst, worst_k = True, c, k
    return violated, worst, worst_k, first_clear


def rotate_about_z(p: np.ndarray, angle: float) -> np.ndarray:
    c_, s_ = np.cos(angle), np.sin(angle)
    return np.array([c_ * p[0] - s_ * p[1], s_ * p[0] + c_ * p[1], p[2]])


def read_block_centre(sim_dir: Path, max_age_s: float, n: int = 3) -> np.ndarray:
    """Purple marker (robot base) as the per-axis median of ``n`` reads 0.1 s apart (the sim rewrites the file
    every step). Raises FileNotFoundError / ValueError if no read succeeds."""
    reads = []
    last_exc: Exception | None = None
    for i in range(n):
        try:
            reads.append(load_xyz_pair(sim_dir / PURPLE_FILE_NAME, max_age_s)[0])
        except (FileNotFoundError, ValueError) as exc:
            last_exc = exc
        if i < n - 1:
            time.sleep(0.1)
    if not reads:
        raise last_exc if last_exc is not None else ValueError("no purple marker read")
    return np.median(np.array(reads), axis=0)


def main(
    home_file: Path = DEFAULT_HOME_FILE,
    aim: Literal["stack", "yaw"] = "stack",
    aim_offset_deg: float = 0.0,
    home_yaw_deg: float = 0.0,
    turn_base: bool = True,
    settle_s: float = 1.5,
    correct_above_deg: float = 4.0,
    max_turns: int = 2,
    log_file: bool = True,
    turn_start: Literal["raise", "clear", "home", "end"] = "clear",
    turn_clear_cm: float = 10.0,
    raise_arms: bool = True,
    raise_deg: float = 40.0,
    raise_duration: float = 3.0,
    torso_back_cm: float = 10.0,
    torso_up_cm: float = 0.0,
    retreat_duration: float = 2.0,
    arm_order: str = "right,left",
    arm_overlap: float = 0.5,
    elbow_lead: float = 0.5,
    spread_duration: float = 2.0,
    duration: float = 6.0,
    second_duration: float = 4.0,
    second_arm_vel: float = 0.6,
    torso_first: bool = False,
    joints_after_turn_deg: float = 45.0,
    torso_with: Literal["arms", "second", "after"] = "second",
    turn_model_steer_s: float = 7.0,
    turn_model_rate_deg_s: float = 6.0,
    settle_duration: float = 3.0,
    move_head: bool = True,
    move_torso: bool = True,
    max_joint_vel: float = 0.4,
    torso_max_vel: float = TORSO_MAX_VEL,
    head_max_vel: float = 0.4,
    fk_check: bool = True,
    min_hand_gap_cm: float = 8.0,
    block_margin_cm: float = 3.0,
    turn_rate_deg_s: float = 8.0,
    max_age_s: float = 3.0,
    yaw_tol_deg: float = 2.0,
    kp: float = 2.0,
    wz_max: float = 1.6,
    wz_min: float = 1.2,
    chassis_hz: float = 20.0,
    steer_tol_deg: float = 10.0,
    steer_timeout_sim: float = 4.0,
    turn_timeout_sim: float = 20.0,
    stall_s: float = 3.0,
    wheel_gain: float = WHEEL_GAIN_SIM,
    anchor_setpoints: bool = True,
    plan_only: bool = False,
    leave_held: bool = True,
    ignore_foreign_commands: bool = False,
) -> bool:
    """Raise both arms (j1 + 90 deg, posture held), then go home arm by arm (torso / head last), base turning back.

    Args:
        home_file: robots/dexmate.py (or any file) holding the ``init_joint_pos = {...}`` home dict.
        aim: What the base turns to face: "stack" (default) = point base +X at the centre of the card stack the box
            was picked from (mean of the four card-stack edge markers, read live from the sim each tick, like step 8's
            ``--aim tag``); "yaw" = the fixed world heading ``home_yaw_deg``.
        aim_offset_deg: Added to the aim heading (degrees, +CCW / left).
        home_yaw_deg: With ``--aim yaw``: base heading in the world to return to (degrees; 0 = the task spawn heading).
            Also the fallback when the edge-marker file is unavailable.
        turn_base: Also turn the base (False = joints only).
        settle_s: After the wheels stop, wait until the heading has been still for this long (the sim base coasts a
            little) before judging the residual.
        correct_above_deg: Re-turn only if the settled residual exceeds this (degrees). The sim base coasts ~1 deg after
            the wheels stop and every re-turn costs ~9 s (5 s of it re-steering), so a residual between ``yaw_tol_deg``
            and this is accepted as done (2026-09-27: 1.4 -> 2.2 deg after settling triggered two useless re-turns).
        max_turns: Turn attempts in total (a re-turn happens only above ``correct_above_deg`` and only while it still
            improves the residual).
        log_file: Mirror the console log into logs/return_home_<time>.log.
        turn_start: When the base turns: "clear" (default) = the wheels start steering right away (that takes ~7 s and
            moves nothing) and the rotation itself is released as soon as every blade point of both hands (FK of the
            measured joints, 25 Hz) is ``turn_clear_cm`` above the block's top (purple marker + half height) -- the arms
            need not be fully up, they only need to pass the box; falls back to "home" when the block marker is
            unavailable. "home" = once the arms are fully up, "raise" = right
            away (the wheels steer for several seconds before the base moves anyway), "end" = after all joints are home
            (not compatible with the torso-first order while the block is in front: nothing would clear it).
        turn_clear_cm: Margin above the block top for the "clear" start (centimetres).
        raise_arms: Stage 1: turn both shoulders' j1 to swing the arms up (False = start the home stage from here).
        raise_deg: How far j1 turns in the raise direction (degrees; left decreases, right increases). 40 default
            (FK-clean down to 30 with the default elbow lead); 90 = arms straight up.
        raise_duration: Minimum time of the raise stage (seconds).
        torso_back_cm: Before the arms are raised the torso moves the whole upper body straight back by this much
            (centimetres, robot -X) with no pitch change -- the three torso joints are solved for (x, z, pitch) on
            the URDF FK -- so both grippers keep their orientation while they retreat from the block. 0 = torso
            still. It is undone by the torso / head home stage at the end.
        torso_up_cm: Same, upward (+Z) component of that upper-body translation (centimetres).
        retreat_duration: Minimum time of the torso retreat (seconds); it runs BEFORE the arm raise.
        arm_order: Which arm goes home first: "right,left" (default, user's choice) or "left,right". Safe either way
            once the shoulders have been spread (``spread_duration``); without the spread only left-first clears.
        arm_overlap: How much of the first arm's motion the second arm overlaps (0.5 default = it starts when the first
            arm is half done; 0 = after it has finished; 1 = together).
        elbow_lead: Fraction of each arm's swing in which j4-j7 (elbow, wrist) finish; j1 takes the whole swing. 0.5
            (default) folds the forearm in first, so the blade stays near the body while the arm comes down -- that is
            what keeps it clear of the block passing by on the left as the base turns (FK 2026-09-27: with 1.0 the
            left blade would meet the block if the base turned slower than ~6 deg/s; with 0.5 it is clear at 4-8).
            Safe for the hands because the shoulders are spread first (without the spread an early fold swung the
            blade into the other hand).
        spread_duration: Before the first arm swings down, both arms' j2 / j3 (shoulder yaw, upper-arm roll) go home
            over at least this time, arms still up: the hands part sideways (blade gap 8 -> 24+ cm), so the arm that
            comes down first cannot brush the other. 0 = skip.
        duration: Minimum time of the first arm's home stage (seconds).
        second_duration: Minimum time of the second arm's home stage (seconds; shorter than the first by default).
        second_arm_vel: Joint speed cap of the second arm (rad/s; the default 0.6 vs 0.4 lets it catch up).
        torso_first: False (default): arms right after the raise while the base turns, torso / head after the arms.
            True: torso + head go home first (waiting ``joints_after_turn_deg`` of base turn), then the arms.
        joints_after_turn_deg: Torso-first only: the joints wait until the base has turned this much (degrees), so the
            block is out of the way before the torso's forward-leaning home pose carries the raised hands ~45 cm ahead
            and the arms swing down (0 = no wait; capped at the turn itself). The FK check models the block at that
            turned position.
        torso_with: Default order only: when torso + head go home. "second" (default since 2026-09-28, user request:
            a little later) = together with the second arm (~3 s after the first arm starts, ending with it);
            "arms" = spread over the whole arm stage (from the first arm's start to the second arm's end; the
            2026-09-27 default); "after" = only once both arms are home (later still). Earlier torso motion pushes
            the still-raised blade tips forward (the torso's home pose leans forward), which is fine only because the
            base is turning the block away by then -- the FK check models exactly that (``turn_model_*``).
        turn_model_steer_s: FK block model: the base starts rotating this long after the base-turn thread starts
            (the wheels steer first; ~7 s on 2026-09-27).
        turn_model_rate_deg_s: FK block model: assumed base yaw rate once rotating (deg/s, wall; measured ~8, 6 is
            conservative). The block is placed accordingly at every knot of the planned joint path.
        settle_duration: Minimum time of the torso + head stage (seconds).
        move_head: Include the head joints.
        move_torso: Include the torso joints.
        max_joint_vel: Arm joint speed cap (rad/s).
        torso_max_vel: Torso joint speed cap (rad/s).
        head_max_vel: Head joint speed cap (rad/s).
        fk_check: Run the planned timeline through the URDF FK and abort on a blade clash / block hit.
        min_hand_gap_cm: Abort if the two finger blades come closer than this anywhere on the path (centimetres).
        block_margin_cm: Abort if a blade point would come within this of the block box around the purple marker
            (centimetres; only when the marker is fresh).
        turn_rate_deg_s: Expected base yaw rate in wall seconds (reporting only).
        max_age_s: Marker files older than this are rejected (sim not running).
        yaw_tol_deg: Stop the turn when |heading error| is below this (degrees).
        kp: Base turn P gain, wz = kp * error (rad/s per rad, sim time).
        wz_max: Base yaw-rate cap (rad/s, sim time).
        wz_min: Minimum |wz| while turning (sim: keeps the wheel joint target above the drive dead zone).
        chassis_hz: Chassis command + feedback rate (wall clock).
        steer_tol_deg: Start driving once both wheels are within this of the in-place steering angles.
        steer_timeout_sim: Max sim seconds to wait for the steering.
        turn_timeout_sim: Give up on the turn after this many sim seconds of driving.
        stall_s: Raise the wheel command if the heading has not changed for this many wall seconds.
        wheel_gain: Multiplier on the wheel-speed command (sim default undoes the scale=100; real robot 1.0).
        anchor_setpoints: Stream relative to the sim's held joint targets (recommended; avoids the sag drop).
        plan_only: Connect, report joint deltas / FK check / yaw needed / expected times, move nothing.
        leave_held: Exit without ``stop()`` so the pose stays held.
        ignore_foreign_commands: Proceed even if another controller is streaming joint targets.
    """
    if max_turns < 1:
        raise ValueError("--max-turns must be >= 1")
    if log_file:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / f"return_home_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(str(log_path), level="INFO", format="{time:HH:mm:ss.SSS} | {level:7} | {message}", enqueue=False)
        logger.info(f"Logging to {log_path}")
    if not 0.0 <= arm_overlap <= 1.0:
        raise ValueError("--arm-overlap must be between 0 and 1")
    if not 0.05 <= elbow_lead <= 1.0:
        raise ValueError("--elbow-lead must be between 0.05 and 1")
    if raise_arms and raise_deg <= 0.0:
        raise ValueError("--raise-deg must be > 0 (or use --no-raise-arms)")
    arms = [a.strip() for a in arm_order.split(",") if a.strip()]
    if sorted(arms) != ["left", "right"]:
        raise ValueError(f"--arm-order must be 'left,right' or 'right,left', got {arm_order!r}")
    first_arm, second_arm = ARM_OF[arms[0]], ARM_OF[arms[1]]

    try:
        home = load_home_joint_pos(home_file)
        logger.info(f"Home joints: {len(home)} from {home_file}")
    except (OSError, ValueError, SyntaxError) as exc:
        logger.warning(f"Could not parse {home_file} ({exc}); using the built-in copy.")
        home = dict(HOME_JOINT_POS_FALLBACK)
    comps = list(ARM_COMPONENTS) + (["torso"] if move_torso else []) + (["head"] if move_head else [])
    targets: dict[str, np.ndarray] = {}
    for c in comps:
        names = COMPONENT_JOINTS[c]
        missing = [n for n in names if n not in home]
        if missing:
            logger.warning(f"{c}: {missing} not in the home dict; component skipped.")
            continue
        targets[c] = np.array([home[n] for n in names], dtype=float)
    if not all(c in targets for c in ARM_COMPONENTS):
        logger.error("Both arms are needed for the return; aborting.")
        return False

    sim_dir = find_sim_dir()

    aim_mode = {"mode": aim}

    def yaw_error() -> tuple[float, str]:
        """Signed yaw the base still has to turn (rad, +CCW) and a short description; raises FileNotFoundError /
        ValueError when the marker files are unavailable (the callers retry)."""
        offset = np.deg2rad(aim_offset_deg)
        if aim_mode["mode"] == "stack":
            edges = load_named_pairs(sim_dir / EDGE_MARKERS_FILE_NAME, max_age_s)
            if len(edges) < 2:
                raise ValueError(f"only {len(edges)} card-stack edge markers in {EDGE_MARKERS_FILE_NAME}")
            c_b = np.mean([b[:2] for b, _w in edges.values()], axis=0)
            err = wrap(float(np.arctan2(c_b[1], c_b[0])) + offset)
            return err, (f"card stack centre at ({c_b[0]:+.2f}, {c_b[1]:+.2f}) m in the base frame, {np.linalg.norm(c_b):.2f} m away, "
                         f"bearing {np.rad2deg(np.arctan2(c_b[1], c_b[0])):+.1f} deg" + (f", offset {aim_offset_deg:+.1f}" if aim_offset_deg else ""))
        pairs = world_fixed_pairs(sim_dir, max_age_s, include_purple=False)
        yaw_wb, span = base_yaw_world(pairs)
        err = wrap(np.deg2rad(home_yaw_deg) + offset - yaw_wb)
        return err, f"base yaw(world) {np.rad2deg(yaw_wb):+.1f} deg from {len(pairs)} markers (span {span * 100:.0f} cm), target {home_yaw_deg:+.0f}"

    yaw_err = None
    if turn_base:
        # The sim rewrites the marker files every step, so a single read can catch a half-written file; retry.
        last_exc: Exception | None = None
        for attempt in range(8):
            try:
                yaw_err, info = yaw_error()
                last_exc = None
                break
            except (FileNotFoundError, ValueError, OSError) as exc:
                last_exc = exc
                time.sleep(0.15)
        if last_exc is not None and aim_mode["mode"] == "stack":
            logger.warning(f"Card-stack edge markers unavailable ({last_exc}); falling back to --aim yaw ({home_yaw_deg:+.0f} deg).")
            aim_mode["mode"] = "yaw"
            for attempt in range(8):
                try:
                    yaw_err, info = yaw_error()
                    last_exc = None
                    break
                except (FileNotFoundError, ValueError, OSError) as exc:
                    last_exc = exc
                    time.sleep(0.15)
        if last_exc is not None:
            logger.error(f"Cannot measure the base heading after 8 tries ({last_exc}); base turn disabled.")
            yaw_err = None
        else:
            logger.info(f"Base turn needed {np.rad2deg(yaw_err):+.1f} deg ({'CCW / left' if yaw_err > 0 else 'CW / right'}) "
                        f"to face the {'card stack' if aim_mode['mode'] == 'stack' else 'home heading'}  [{info}]")
            if abs(yaw_err) < np.deg2rad(yaw_tol_deg):
                logger.info("Base already at the home heading; no turn.")
                yaw_err = None

    block_centre: np.ndarray | None = None
    block_model: BlockModel | None = None
    try:
        block_centre = read_block_centre(sim_dir, max_age_s)
        n_face = box_face_normal_base(sim_dir / TIP_FILE_NAME, sim_dir, max_age_s=max_age_s)
        axis = n_face if float(np.linalg.norm(n_face)) > 0.5 else None
        half_t = BLOCK_HALF_T
        try:
            yaw_b = block_yaw_vs_face(sim_dir / TIP_FILE_NAME, sim_dir / PURPLE_FILE_NAME)
        except (FileNotFoundError, ValueError, OSError):
            yaw_b = None
        if yaw_b is not None:
            half_t += BLOCK_HALF_W * abs(np.sin(yaw_b))  # a yawed block's corners stick out along the normal
        block_model = BlockModel(block_centre, axis, np.array([half_t, BLOCK_HALF_W, BLOCK_HALF_H]))
        logger.info(f"Block model (robot base): {block_model.describe()}"
                    + (f"; block yaw vs the box face {np.rad2deg(yaw_b):+.1f} deg folded into the thickness" if yaw_b is not None else ""))
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(f"Purple marker unavailable ({exc}); the FK check skips the block clearance.")

    probe = [c for c in targets if c in COMMAND_TOPICS]
    if not plan_only:
        foreign = detect_foreign_commands(probe)
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
    setpoints = read_setpoints(probe) if (anchor_setpoints and not plan_only) else {}

    fk = HandFK() if (fk_check or setpoints or turn_start == "clear") else None
    bot = connect()
    do_shutdown = not leave_held
    turn_thread = None
    turn_result: dict = {}
    try:
        meas = {c: read_joint_pos(getattr(bot, c), c) for c in targets}
        for c in targets:
            logger.info(f"{c}: now {np.round(np.rad2deg(meas[c]), 1)} -> home {np.round(np.rad2deg(targets[c]), 1)} deg "
                        f"(max change {np.rad2deg(np.abs(targets[c] - meas[c]).max()):.1f} deg)")
        offsets = {c: np.zeros_like(meas[c]) for c in targets}
        if setpoints and fk is not None:
            fk.set({c: meas[c] for c in targets if c in ("left_arm", "right_arm", "torso")})
            offsets.update(anchor_offsets(fk.cfg, fk.idx, setpoints, [c for c in probe if c in setpoints]))

        # --- stage 1: raise (j1 of both arms, everything else held) -------------------------------------------
        up = {c: meas[c].copy() for c in ARM_COMPONENTS}
        t_raise = 0.0
        torso_back: np.ndarray | None = None
        if raise_arms and "torso" in targets and (abs(torso_back_cm) > 1e-3 or abs(torso_up_cm) > 1e-3):
            if fk is None:
                fk = HandFK()
            torso_back, got, inside = solve_torso_translation(fk, {c: meas[c] for c in ARM_COMPONENTS}, meas["torso"],
                                                              -torso_back_cm / 100.0, torso_up_cm / 100.0)
            logger.info(f"Raise: torso {np.round(np.rad2deg(meas['torso']), 1)} -> {np.round(np.rad2deg(torso_back), 1)} deg moves the upper "
                        f"body back {-got[0] * 100:.1f} cm / up {got[1] * 100:.1f} cm with its pitch unchanged (asked {torso_back_cm:.0f} / {torso_up_cm:.0f})"
                        + ("" if inside else " -- CLIPPED at a torso joint limit"))
            if not inside and abs(-got[0] * 100 - torso_back_cm) > 2.0:
                logger.warning("Torso joint limits cut the requested upper-body retreat short; continuing with what is reachable.")
        if raise_arms:
            for c in ARM_COMPONENTS:
                side = SIDE_OF[c]
                up[c][J1] = meas[c][J1] + RAISE_SIGN[side] * np.deg2rad(raise_deg)
                if abs(up[c][J1]) > J1_LIMIT_RAD:
                    logger.error(f"{COMPONENT_JOINTS[c][J1]} raise target {up[c][J1]:+.3f} rad is outside +/-{J1_LIMIT_RAD} rad; aborting.")
                    return False
                logger.info(f"Raise: {COMPONENT_JOINTS[c][J1]} {np.rad2deg(meas[c][J1]):+.1f} -> {np.rad2deg(up[c][J1]):+.1f} deg, "
                            f"other joints held.")
            t_lift = stage_time(np.deg2rad(raise_deg), raise_duration, max_joint_vel)
            start_a = {c: meas[c] for c in ARM_COMPONENTS}
            target_a = dict(up)
            off_a = {c: offsets[c] for c in ARM_COMPONENTS}
            joint_stage_a = {c: [("raise", [J1])] for c in ARM_COMPONENTS}
            t_retreat = 0.0
            stages_a: dict[str, tuple[float, float]] = {}
            if torso_back is not None:  # torso retreat first, arms raise after it
                t_retreat = stage_time(float(np.abs(torso_back - meas["torso"]).max()), retreat_duration, torso_max_vel)
                start_a["torso"], target_a["torso"], off_a["torso"] = meas["torso"], torso_back, offsets["torso"]
                joint_stage_a["torso"] = [("retreat", [0, 1, 2])]
                stages_a["retreat"] = (0.0, t_retreat)
            stages_a["raise"] = (t_retreat, t_lift)
            t_raise = t_retreat + t_lift
            _t, groups_a = build_timeline(start_a, target_a, off_a, stages_a, joint_stage_a)
        else:
            groups_a = None
            logger.info("Raise stage skipped on request (--no-raise-arms).")

        # --- stage 2: home ----------------------------------------------------------------------------------------
        # Arms: first arm, then the second (overlapping). Torso + head go either BEFORE the arms (torso_first,
        # default: they go home while the base turns, arms still up) or after them (the previous behaviour).
        d_arm = {c: targets[c] - up[c] for c in ARM_COMPONENTS}
        do_spread = spread_duration > 0.0
        t_spread = stage_time(max(float(np.abs(d_arm[c][SPREAD_JOINTS]).max()) for c in ARM_COMPONENTS), spread_duration, max_joint_vel) if do_spread else 0.0
        swing_joints = [J1] if do_spread else SHOULDER_JOINTS  # j2 / j3 already home after the spread
        t_arm: dict[str, float] = {}
        for c in ARM_COMPONENTS:
            cap = second_arm_vel if c == second_arm else max_joint_vel
            min_dur = second_duration if c == second_arm else duration
            t_slow = stage_time(float(np.abs(d_arm[c][swing_joints]).max()), min_dur, cap)
            t_lead = stage_time(float(np.abs(d_arm[c][ELBOW_WRIST_JOINTS]).max()), 0.0, cap) / elbow_lead
            t_arm[c] = max(t_slow, t_lead)
        t_first_start = t_spread
        t_second_start = t_spread + (1.0 - arm_overlap) * t_arm[first_arm]
        t_arms = max(t_first_start + t_arm[first_arm], t_second_start + t_arm[second_arm])
        settle_comps = [c for c in ("torso", "head") if c in targets]
        t_settle = settle_duration
        settle_from = {c: meas[c] for c in settle_comps}
        settle_off = {c: offsets[c] for c in settle_comps}
        if torso_back is not None:
            settle_from["torso"], settle_off["torso"] = torso_back, np.zeros_like(torso_back)  # offset already faded in the raise
        for c, cap in (("torso", torso_max_vel), ("head", head_max_vel)):
            if c in targets:
                t_settle = max(t_settle, stage_time(float(np.abs(targets[c] - settle_from[c]).max()), settle_duration, cap))
        arm_stages = {
            first_arm: (t_first_start, t_arm[first_arm]),
            first_arm + "_lead": (t_first_start, t_arm[first_arm] * elbow_lead),
            second_arm: (t_second_start, t_arm[second_arm]),
            second_arm + "_lead": (t_second_start, t_arm[second_arm] * elbow_lead),
        }
        arm_joint_stage = {c: [(c, swing_joints), (c + "_lead", ELBOW_WRIST_JOINTS)] for c in ARM_COMPONENTS}
        if do_spread:
            arm_stages["spread"] = (0.0, t_spread)
            for c in ARM_COMPONENTS:
                arm_joint_stage[c].insert(0, ("spread", SPREAD_JOINTS))
        # Arms start from the raised pose: j1 at its raise target (offset already faded), the rest at the measured
        # value with its anchor offset still to fade; torso / head from the measurement.
        start_arms = {c: up[c].copy() for c in ARM_COMPONENTS}
        off_arms = {c: offsets[c].copy() for c in ARM_COMPONENTS}
        if raise_arms:
            for c in ARM_COMPONENTS:
                off_arms[c][J1] = 0.0
        settle_joint_stage = {c: [("settle", list(range(len(targets[c]))))] for c in settle_comps}
        streams_b: list[tuple[str, dict[str, np.ndarray], float]] = []  # (label, groups, duration)
        if torso_first:
            if settle_comps:
                _t, g_settle = build_timeline(settle_from, {c: targets[c] for c in settle_comps}, settle_off,
                                              {"settle": (0.0, t_settle)}, settle_joint_stage)
                streams_b.append(("torso/head home", g_settle, t_settle))
            _t, g_arms = build_timeline(start_arms, {c: targets[c] for c in ARM_COMPONENTS}, off_arms, arm_stages, arm_joint_stage)
            streams_b.append((f"arms home ({first_arm} first)", g_arms, t_arms))
        else:
            if torso_with == "arms":
                t_settle_start = t_first_start
                t_settle = max(t_settle, t_arms - t_first_start)
            elif torso_with == "second":
                t_settle_start = t_second_start
                t_settle = max(t_settle, t_arms - t_second_start)
            else:
                t_settle_start = t_arms
            stages_b = dict(arm_stages, settle=(t_settle_start, t_settle))
            start_b = dict(start_arms, **settle_from)
            off_b = dict(off_arms, **settle_off)
            _t, g_all = build_timeline(start_b, targets, off_b, stages_b, dict(arm_joint_stage, **settle_joint_stage))
            streams_b.append((f"arms home ({first_arm} first) + torso/head {t_settle_start:.1f}-{t_settle_start + t_settle:.1f}s", g_all,
                              max(t0 + td for t0, td in stages_b.values())))
        n_stages = (1 if raise_arms else 0) + len(streams_b)
        # How far the base must have turned before the joints move (torso-first only; the torso's home pose leans
        # forward and carries the raised hands ~45 cm ahead, so the block must be out of the way first).
        gate_deg = 0.0
        if torso_first and yaw_err is not None and turn_start != "end":
            gate_deg = float(min(max(joints_after_turn_deg, 0.0), abs(np.rad2deg(yaw_err))))
        t_turn = abs(np.rad2deg(yaw_err)) / max(turn_rate_deg_s, 0.1) if yaw_err is not None else 0.0
        logger.info(
            "Timeline: " + ((f"torso retreat {t_retreat:.1f}s -> " if torso_back is not None else "") + f"raise {t_raise - (t_retreat if torso_back is not None else 0.0):.1f}s | " if raise_arms else "")
            + (f"[base turn starts; joints wait until it has covered {gate_deg:.0f} deg] | " if gate_deg > 0 else "")
            + " | ".join(f"{lbl} {t:.1f}s" for lbl, _g, t in streams_b)
            + "; arm stages: " + (f"spread j2/j3 0.0-{t_spread:.1f}s, " if do_spread else "")
            + f"{first_arm} {t_first_start:.1f}-{t_first_start + t_arm[first_arm]:.1f}s, {second_arm} {t_second_start:.1f}-{t_second_start + t_arm[second_arm]:.1f}s"
            + (f" (elbow/wrist within the first {elbow_lead:.0%} of each)" if elbow_lead < 1.0 else "")
            + f"; joints ~{t_raise + sum(t for _l, _g, t in streams_b):.0f}s of streaming"
            + (f"; base turn {np.rad2deg(yaw_err):+.1f} deg starting at '{turn_start}', ~{t_turn:.0f}s at {turn_rate_deg_s:.0f} deg/s "
               "after the steering" if yaw_err is not None else "; no base turn")
        )

        # --- FK check ------------------------------------------------------------------------------------------
        if fk_check and fk is not None:
            # Concatenate the streams; a component not commanded in a stream holds its last value.
            segments = ([groups_a] if groups_a is not None else []) + [g for _l, g, _t in streams_b]
            last = {c: meas[c] + offsets[c] for c in targets}
            full_parts: dict[str, list[np.ndarray]] = {c: [] for c in targets}
            for seg in segments:
                n_seg = len(next(iter(seg.values())))
                for c in targets:
                    rows = seg[c] if c in seg else np.tile(last[c], (n_seg, 1))
                    full_parts[c].append(rows)
                    last[c] = rows[-1]
            full = {c: np.vstack(v) for c, v in full_parts.items()}
            hands_now = fk.hands({c: meas[c] for c in targets})
            hands_up = fk.hands({**{c: meas[c] for c in targets}, **up, **({"torso": torso_back} if torso_back is not None else {})})
            hands_home = fk.hands(targets)
            for side in ("left", "right"):
                logger.info(
                    f"{side.capitalize()} gripper base (robot base): now {np.round(hands_now[side][0], 3)} -> raised "
                    f"{np.round(hands_up[side][0], 3)} -> home {np.round(hands_home[side][0], 3)}; blade tip now "
                    f"{np.round(hands_now[side][1], 3)} -> home {np.round(hands_home[side][1], 3)}"
                )
            block_fk = block_model
            if block_model is not None and gate_deg > 0:
                # The joints only move once the base has turned gate_deg: the block then sits that much further round.
                block_fk = block_model.turned(np.sign(yaw_err) * np.deg2rad(gate_deg))
                logger.info(f"FK block model: after {gate_deg:.0f} deg of base turn the block is at {block_fk.describe()}")
            block_track = None
            if block_model is not None and not torso_first and yaw_err is not None and turn_start != "end":
                # The base-turn thread starts at t=0 ("clear" / "raise") or when the raise ends ("home"); the wheels
                # steer for turn_model_steer_s, then the base turns at turn_model_rate_deg_s until the yaw is done.
                # Objects turn the opposite way in the base frame; use a 40 cm square footprint once it has turned.
                t_thread = 0.0 if turn_start in ("clear", "raise") else t_raise
                t_drive0 = t_thread + turn_model_steer_s
                n_full = len(next(iter(full.values())))
                block_track = []
                for k in range(n_full):
                    t_k = k / KNOT_HZ
                    turned = min(abs(yaw_err), np.deg2rad(turn_model_rate_deg_s) * max(0.0, t_k - t_drive0))
                    block_track.append(block_model.turned(np.sign(yaw_err) * turned))
                logger.info(f"FK block model: base rotation assumed from t={t_drive0:.1f}s at {turn_model_rate_deg_s:.0f} deg/s "
                            f"(thread at t={t_thread:.1f}s + {turn_model_steer_s:.0f}s steering); the block is checked where it is at each knot.")
            chk = check_path_fk(fk, full, block_fk, block_track=block_track)
            msg = (f"FK check over the whole path: min blade-to-blade gap {chk['gap_min'] * 100:.1f} cm; blade points reach x "
                   f"{chk['x_min']:+.2f}..{chk['x_max']:.2f} m, up to z {chk['z_max']:.2f} m")
            ok = True
            violated, worst_c, worst_k, first_clear_k = block_violation(chk["block_series"], block_margin_cm / 100.0)
            c_start = chk["block_series"][0][1] if chk["block_series"] else np.inf
            if chk["gap_min"] < min_hand_gap_cm / 100.0:
                logger.error(msg + f" -- the blades would come within {min_hand_gap_cm:.0f} cm of each other; aborting.")
                ok = False
            elif (block_fk is not None or block_track is not None) and violated:
                logger.error(msg + f" -- a blade would come within {block_margin_cm:.0f} cm of the block "
                             f"(clearance {worst_c * 100:.1f} cm at t~{worst_k / KNOT_HZ:.1f}s of the joint path; start clearance {c_start * 100:.1f} cm"
                             + (f", first clear at t~{first_clear_k / KNOT_HZ:.1f}s" if first_clear_k >= 0 else ", never clear") + "); aborting."
                             + (" The torso-first order needs the base turned away first: keep --turn-base on, "
                                "or raise --joints-after-turn-deg, or use --no-torso-first." if torso_first else ""))
                ok = False
            else:
                logger.info(msg + ((f"; block: starts {c_start * 100:.1f} cm away (blades still at the block after the push), clear of "
                                    f"{block_margin_cm:.0f} cm from t~{first_clear_k / KNOT_HZ:.1f}s and never back within it (closest approach "
                                    f"later on {min((c for k, c in chk['block_series'] if k >= first_clear_k + 2 * KNOT_HZ), default=np.inf) * 100:.0f} cm)")
                                   if (block_fk is not None or block_track is not None) and first_clear_k >= 0 else
                                   (f"; block clearance >= {chk['block_clear'] * 100:.0f} cm" if (block_fk is not None or block_track is not None) else "")))
            if not ok:
                logger.error("Nothing commanded. (--no-fk-check overrides; --min-hand-gap-cm / --block-margin-cm tune the limits.)")
                return False
        if plan_only:
            logger.info("Plan only: nothing commanded.")
            return True

        # --- execute -------------------------------------------------------------------------------------------
        rtf = measure_rtf(sim_dir / TIP_FILE_NAME) if yaw_err is not None else 1.0

        def start_turn(drive_gate=None) -> None:
            nonlocal turn_thread
            if yaw_err is None or turn_thread is not None:
                return

            def settled_error() -> tuple[float, str]:
                """Wait until the heading has been still for settle_s (the base coasts after the wheels stop), max 6 s."""
                t_end = time.time() + 6.0
                hist: list[tuple[float, float]] = []
                err_s, info_s = yaw_error()
                while time.time() < t_end:
                    try:
                        err_s, info_s = yaw_error()
                    except (FileNotFoundError, ValueError, OSError):
                        time.sleep(0.1)
                        continue
                    now = time.time()
                    hist.append((now, err_s))
                    hist = [h for h in hist if now - h[0] <= settle_s]
                    if len(hist) >= 3 and now - hist[0][0] >= settle_s * 0.9 and max(h[1] for h in hist) - min(h[1] for h in hist) < np.deg2rad(0.3):
                        break
                    time.sleep(0.1)
                return err_s, info_s

            def _turn():
                try:
                    for attempt in range(1, max_turns + 1):
                        res = turn_in_place(
                            bot.chassis, yaw_error, tol_deg=yaw_tol_deg, kp=kp, wz_max=wz_max, wz_min=wz_min,
                            hz=chassis_hz, steer_tol_deg=steer_tol_deg, steer_timeout_sim=steer_timeout_sim,
                            timeout_sim=turn_timeout_sim, stall_s=stall_s, wheel_gain=wheel_gain, rtf=rtf,
                            drive_gate=drive_gate if attempt == 1 else None,
                        )
                        err_s, info_s = settled_error()
                        logger.info(f"Base turn {attempt}/{max_turns}: {np.rad2deg(res[0]):+.1f} deg at the wheel stop, "
                                    f"{np.rad2deg(err_s):+.1f} deg after settling  [{info_s}]")
                        prev = turn_result.get("res", (None, ""))[0]
                        turn_result["res"] = (err_s, info_s)
                        if abs(err_s) < np.deg2rad(yaw_tol_deg):
                            break
                        if abs(err_s) <= np.deg2rad(correct_above_deg):
                            logger.info(f"Residual {np.rad2deg(err_s):+.1f} deg is above the {yaw_tol_deg:.1f} deg tolerance but within "
                                        f"--correct-above-deg {correct_above_deg:.1f}; accepted, no re-turn.")
                            break
                        if prev is not None and abs(err_s) > abs(prev) - np.deg2rad(0.3):
                            logger.warning(f"Re-turn did not improve the residual ({np.rad2deg(prev):+.1f} -> {np.rad2deg(err_s):+.1f} deg); stopping here.")
                            break
                        if attempt < max_turns:
                            logger.warning(f"Heading still off by {np.rad2deg(err_s):+.1f} deg (> {correct_above_deg:.1f}); turning again.")
                except BaseException as exc:  # noqa: BLE001 - report and stop the chassis
                    turn_result["exc"] = exc
                    try:
                        for _ in range(5):
                            bot.chassis.stop()
                            time.sleep(0.05)
                    except Exception:
                        pass

            turn_thread = threading.Thread(target=_turn, name="base-turn", daemon=True)
            turn_thread.start()
            logger.info("Base turn started in the background.")

        if turn_start == "raise":
            start_turn()

        clear_state = {"done": False, "ticks": 0, "min_z": None, "raise_done": False}
        block_top = float(block_centre[2] + BLOCK_HALF_H) if block_centre is not None else None
        if turn_start == "clear" and groups_a is not None and block_top is not None:
            # Steer the wheels now (that alone takes ~7 s and moves nothing); the rotation itself waits for the blades
            # to clear the block top, or for the raise to finish.
            start_turn(drive_gate=lambda: clear_state["done"] or clear_state["raise_done"])

        def clear_check(b) -> bool:
            """stream() hook during the raise: start the base turn once both hands' blades are above the block top;
            never stops the stream (always returns False)."""
            if clear_state["done"] or block_top is None or fk is None:
                return False
            clear_state["ticks"] += 1
            if clear_state["ticks"] % 2:  # FK every other tick (25 Hz)
                return False
            try:
                q_now = {c: np.asarray(getattr(b, c).get_joint_pos(), dtype=float) for c in ("left_arm", "right_arm", "torso")}
            except Exception:
                return False
            hands = fk.hands(q_now)
            min_z = min(float(p[2]) for pts in hands.values() for p in pts)
            clear_state["min_z"] = min_z
            if min_z >= block_top + turn_clear_cm / 100.0:
                clear_state["done"] = True
                logger.info(f"Blades clear the block top (lowest blade point z {min_z:.2f} m >= block top {block_top:.2f} + "
                            f"{turn_clear_cm:.0f} cm) -- base rotation released, arms still rising.")
                start_turn()  # no-op when the thread is already steering
            return False
        if groups_a is not None:
            logger.info(f"Stage 1/{n_stages} raise: " + (f"torso retreat (upper body back {torso_back_cm:.0f} cm) {t_retreat:.1f}s, then " if torso_back is not None else "")
                        + f"both arms j1 {raise_deg:+.0f} deg; {t_raise:.1f}s in total...")
            if turn_start == "clear" and block_top is None and yaw_err is not None:
                logger.warning("Block marker unavailable: the base turn starts when the arms are fully up instead (--turn-start home).")
            final_a, _stopped = stream(bot, groups_a, t_raise, config.CONTROL_DT,
                                       stop_check=clear_check if (turn_start == "clear" and block_top is not None) else None)
            clear_state["raise_done"] = True
            if turn_start == "clear" and not clear_state["done"] and yaw_err is not None:
                if block_top is not None:
                    logger.warning(f"Blades never reached {turn_clear_cm:.0f} cm above the block top during the raise (lowest point "
                                   f"{clear_state['min_z'] if clear_state['min_z'] is not None else float('nan'):.2f} m vs {block_top:.2f} m); "
                                   "releasing the base rotation now that the arms are up.")
                start_turn()
            for _lbl, groups, _t in streams_b:  # re-anchor the later stages at what was actually commanded last
                for c in ARM_COMPONENTS:
                    if c in groups:
                        groups[c][:, J1] += final_a[c][J1] - up[c][J1]
                if torso_back is not None and "torso" in groups and "torso" in final_a:
                    groups["torso"] += final_a["torso"] - torso_back
            after_raise = {c: read_joint_pos(getattr(bot, c), c) for c in ARM_COMPONENTS}
            logger.info("Arms up; j1 residual (deg): "
                        + ", ".join(f"{c} {np.rad2deg(abs(up[c][J1] - after_raise[c][J1])):.1f}" for c in ARM_COMPONENTS))
        if turn_start == "home":
            start_turn()

        def wait_for_turn_progress(deg: float) -> bool:
            """Block until the base has turned ``deg`` (yaw error shrunk by that much). False if the turn thread ended
            or the turn timeouts passed before that -- the block is then still in the way of the torso-first order."""
            need = np.deg2rad(deg)
            err0 = abs(yaw_err)
            deadline = time.time() + (steer_timeout_sim + turn_timeout_sim) / max(rtf, 0.05) + 10.0
            last_log = 0.0
            logger.info(f"Joints wait until the base has turned {deg:.0f} of {np.rad2deg(err0):.0f} deg...")

            def turned_so_far() -> float | None:
                try:
                    err_now, _info = yaw_error()
                    return err0 - abs(err_now)
                except (FileNotFoundError, ValueError, OSError):
                    return None

            while True:
                turned = turned_so_far()
                if turned is not None and turned >= need:
                    logger.info(f"Base has turned {np.rad2deg(turned):.0f} deg; the joints go now"
                                + (" (turn continues in the background)." if turn_thread is not None and turn_thread.is_alive() else "."))
                    return True
                if turn_thread is None or not turn_thread.is_alive():
                    logger.error(f"Base turn ended after only {np.rad2deg(turned) if turned is not None else float('nan'):.0f} of the "
                                 f"{deg:.0f} deg the joints wait for" + (f" ({turn_result['exc']!r})" if "exc" in turn_result else "") + ".")
                    return False
                if time.time() > deadline:
                    logger.error(f"Base turn did not cover {deg:.0f} deg within the timeouts.")
                    return False
                if turned is not None and time.time() - last_log > 2.0:
                    logger.info(f"  base turned {np.rad2deg(turned):.0f} / {deg:.0f} deg")
                    last_log = time.time()
                time.sleep(0.2)

        if gate_deg > 0 and turn_thread is not None and not wait_for_turn_progress(gate_deg):
            logger.error("The block is still in front: torso / arms NOT moved (arms left held up). Fix the turn and re-run with "
                         "--no-raise-arms, or use --no-torso-first --no-turn-base for the joints alone.")
            try:
                bot.chassis.stop()
            except Exception:
                pass
            if leave_held:
                disconnect_leave_held(bot)
                do_shutdown = False
            return False
        stage_no = 2 if raise_arms else 1
        for lbl, groups, t in streams_b:
            logger.info(f"Stage {stage_no}/{n_stages} {lbl}: streaming {' + '.join(groups)} over {t:.1f}s...")
            stream(bot, groups, t, config.CONTROL_DT)
            stage_no += 1
        after = {c: read_joint_pos(getattr(bot, c), c) for c in targets}
        res = {c: float(np.rad2deg(np.abs(targets[c] - after[c]).max())) for c in targets}
        logger.info("Joints at home; residual (deg): " + ", ".join(f"{c} {r:.1f}" for c, r in res.items()))
        if turn_start == "end":
            start_turn()

        if turn_thread is not None:
            logger.info("Waiting for the base turn to finish...")
            while turn_thread.is_alive():
                turn_thread.join(timeout=0.5)
            if "exc" in turn_result:
                logger.error(f"Base turn failed: {turn_result['exc']!r}")
            else:
                err, info = turn_result.get("res", (None, ""))
                if err is not None:
                    logger.info(f"Base turn finished; heading residual {np.rad2deg(err):+.1f} deg  [{info}]")
        logger.info("Return home done; pose left held.")
        if leave_held:
            disconnect_leave_held(bot)
            do_shutdown = False
            logger.info("Client disconnected without stop() -- pose held.")
        return True
    except KeyboardInterrupt:
        logger.warning("Interrupted -- stopping the chassis, leaving the last commanded joints held.")
        try:
            bot.chassis.stop()
        except Exception:
            pass
        do_shutdown = True
        return False
    finally:
        if turn_thread is not None and turn_thread.is_alive():
            try:
                bot.chassis.stop()
            except Exception:
                pass
            turn_thread.join(timeout=2.0)
        if bot is not None and (do_shutdown or plan_only):
            if plan_only:
                disconnect_leave_held(bot)
            else:
                logger.info("Shutting down robot")
                bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
