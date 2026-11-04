# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Marker status files written by ``unitree_sim_isaaclab*/sim_main.py`` every step (robot base frame).

* ``card_box_tip_pos.txt``          -- red ball: +X tip face centre of the conveyor box
* ``card_box_mass_center_pos.txt``  -- purple ball: centre of the tracked card block (``# source ...``)
* ``card_edge_markers_pos.txt``     -- yellow / cyan / orange / blue -X edge markers of the card stack

The simulator folder is ``$LOTTE_SIM_DIR`` if set, else the ``dex/unitree_sim_isaaclab*`` folder whose
``card_box_tip_pos.txt`` was rewritten most recently (the running sim rewrites it every step).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

DEX_ROOT = Path(__file__).resolve().parents[2]
SIM_FALLBACK = "unitree_sim_isaaclab_260918"
TIP_FILE_NAME = "card_box_tip_pos.txt"
PURPLE_FILE_NAME = "card_box_mass_center_pos.txt"
TIP_FACE_NORMAL_WORLD = (1.0, 0.0, 0.0)  # the conveyor box tip face (+Z of the box) faces world +X (scene cfg)
EDGE_MARKERS_FILE_NAME = "card_edge_markers_pos.txt"


def find_sim_dir(stamp_file: str = TIP_FILE_NAME) -> Path:
    """Simulator folder (see module docstring)."""
    env = os.environ.get("LOTTE_SIM_DIR")
    if env:
        return Path(env).expanduser()
    with_file = [d for d in DEX_ROOT.glob("unitree_sim_isaaclab*") if (d / stamp_file).is_file()]
    if with_file:
        return max(with_file, key=lambda d: (d / stamp_file).stat().st_mtime)
    return DEX_ROOT / SIM_FALLBACK


def is_fresh(path: Path, max_age_s: float) -> bool:
    """True if the sim rewrote ``path`` within ``max_age_s`` seconds."""
    try:
        return (time.time() - path.stat().st_mtime) <= max_age_s
    except FileNotFoundError:
        return False


def load_xyz_pair(path: Path, max_age_s: float | None = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """(p_base, p_world) from a marker file (``x y z`` line + ``# world x y z`` line), no logging."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"marker file not found: {path}")
    if max_age_s is not None and not is_fresh(path, max_age_s):
        raise FileNotFoundError(f"marker file {path} is older than {max_age_s:.0f}s; is sim_main.py running?")
    base = world = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "#":
            if len(s) >= 5 and s[1] == "world":
                world = np.array([float(v) for v in s[2:5]])
        elif len(s) >= 3 and base is None:
            base = np.array([float(v) for v in s[:3]])
    if base is None or world is None:
        raise ValueError(f"{path}: needs an 'x y z' line and a '# world x y z' line")
    return base, world


def load_xyz_base(path: Path, label: str, max_age_s: float | None = 3.0) -> np.ndarray:
    """First non-comment ``x y z`` line of a marker file (robot base frame).

    Raises FileNotFoundError if the file is missing or, with ``max_age_s``, older than that (the sim
    is not running / not writing).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}. Is sim_main.py running and writing markers?")
    if max_age_s is not None and not is_fresh(path, max_age_s):
        raise FileNotFoundError(f"{label} file {path} is older than {max_age_s:.0f}s; is sim_main.py running?")
    source = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "#":
            if len(s) >= 3 and s[1] == "source":
                source = s[2]
            continue
        if len(s) >= 3:
            xyz = np.array([float(s[0]), float(s[1]), float(s[2])])
            logger.info(f"Loaded {label} (robot base) from {path.name}: {np.round(xyz, 4)}"
                        + (f" (source {source})" if source else ""))
            return xyz
    raise ValueError(f"{label} file {path} has no 'x y z' line")


def load_named_pairs(path: Path, max_age_s: float | None = 3.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{name: (p_base, p_world)}`` from a multi-marker file (``name x y z`` + ``# world name x y z`` lines),
    e.g. ``card_edge_markers_pos.txt``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"marker file not found: {path}")
    if max_age_s is not None and not is_fresh(path, max_age_s):
        raise FileNotFoundError(f"marker file {path} is older than {max_age_s:.0f}s; is sim_main.py running?")
    base: dict[str, np.ndarray] = {}
    world: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "#":
            if len(s) >= 6 and s[1] == "world":
                world[s[2]] = np.array([float(v) for v in s[3:6]])
        elif len(s) >= 4:
            base[s[0]] = np.array([float(v) for v in s[1:4]])
    return {n: (base[n], world[n]) for n in base if n in world}


def world_fixed_pairs(sim_dir: Path, max_age_s: float = 3.0, include_purple: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All fresh (base, world) point pairs of markers that do not move in the world: red tip, the four
    card-stack edge markers and (``include_purple``) the purple block centre -- the latter only when the
    block has been set down."""
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fname in (("red_tip", TIP_FILE_NAME),) + ((("purple_centre", PURPLE_FILE_NAME),) if include_purple else ()):
        try:
            pairs[name] = load_xyz_pair(sim_dir / fname, max_age_s)
        except (FileNotFoundError, ValueError):
            pass
    try:
        pairs.update(load_named_pairs(sim_dir / EDGE_MARKERS_FILE_NAME, max_age_s))
    except (FileNotFoundError, ValueError):
        pass
    return pairs


def base_yaw_world(pairs: dict[str, tuple[np.ndarray, np.ndarray]], min_span_m: float = 0.05) -> tuple[float, float]:
    """Base yaw in the world (rad) from >= 2 world-fixed point pairs by a planar rigid fit
    (world = R(yaw) base + t, least squares). Returns ``(yaw, span)`` where ``span`` is the largest
    pairwise XY distance of the points; raises ValueError if fewer than 2 pairs or ``span < min_span_m``."""
    if len(pairs) < 2:
        raise ValueError(f"need >= 2 world-fixed marker pairs for the base yaw, have {len(pairs)}")
    B = np.array([b[:2] for b, _ in pairs.values()])
    W = np.array([w[:2] for _, w in pairs.values()])
    span = max(float(np.linalg.norm(B[i] - B[j])) for i in range(len(B)) for j in range(i + 1, len(B)))
    if span < min_span_m:
        raise ValueError(f"marker points span only {span * 100:.1f} cm in XY; base yaw would be unreliable")
    Bc, Wc = B - B.mean(axis=0), W - W.mean(axis=0)
    # optimal planar rotation: yaw = atan2(sum(bx*wy - by*wx), sum(bx*wx + by*wy))
    num = float(np.sum(Bc[:, 0] * Wc[:, 1] - Bc[:, 1] * Wc[:, 0]))
    den = float(np.sum(Bc[:, 0] * Wc[:, 0] + Bc[:, 1] * Wc[:, 1]))
    return float(np.arctan2(num, den)), span


def world_dir_to_base(sim_dir: Path, dir_world, max_age_s: float = 3.0, include_purple: bool = False) -> np.ndarray:
    """A horizontal world direction expressed in the robot base frame, using the base yaw fitted from the
    fresh world-fixed marker pairs (``base_yaw_world``). Raises FileNotFoundError / ValueError if the yaw
    cannot be measured."""
    yaw, _span = base_yaw_world(world_fixed_pairs(sim_dir, max_age_s, include_purple=include_purple))
    d = np.asarray(dir_world, dtype=float)
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])


# ---------------------------------------------------------------------------
# Control-rate readers (no logging) and the conveyor box face normal
# ---------------------------------------------------------------------------
def quiet_xyz(path: Path) -> np.ndarray:
    """First ``x y z`` line of a marker file without logging (for polling at the control rate).

    Raises ValueError if the file has no such line (the sim rewrites the files every step, so a
    caller polling fast should catch ValueError / OSError and retry on the next tick)."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s_ = line.split()
        if s_ and s_[0] != "#" and len(s_) >= 3:
            return np.array([float(s_[0]), float(s_[1]), float(s_[2])])
    raise ValueError(f"{path}: no 'x y z' line")


def quiet_quat_base(path: Path) -> np.ndarray | None:
    """``# quat_base w x y z`` line of a marker file (orientation in the robot base frame), or None."""
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            s_ = line.split()
            if len(s_) >= 6 and s_[0] == "#" and s_[1] == "quat_base":
                return np.array([float(v) for v in s_[2:6]])
    except (OSError, ValueError):
        pass
    return None


def quat_axis(q_wxyz: np.ndarray, col: int) -> np.ndarray:
    """Column ``col`` of the rotation matrix of a (w, x, y, z) quaternion = that local axis in the parent frame."""
    w, x, y, z = (float(v) for v in q_wxyz)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R[:, col]


def box_face_normal_base(
    tip_path: Path, sim_dir: Path, face_normal_world=TIP_FACE_NORMAL_WORLD, max_age_s: float = 3.0
) -> np.ndarray:
    """Unit horizontal normal of the conveyor box tip face in the current robot base frame.

    From the box quaternion in the tip file (``# quat_base``, face normal = box local +z, written by
    sim_main since 2026-09-26); else from ``face_normal_world`` turned into the base frame with the
    base yaw fitted from the world-fixed markers (``world_dir_to_base``). The zero vector if neither
    is available. Used by steps 10 and 12."""
    qb = quiet_quat_base(tip_path)
    if qb is not None:
        n_b = quat_axis(qb, 2)
        n_b[2] = 0.0
        if np.linalg.norm(n_b) > 0.5:
            return n_b / np.linalg.norm(n_b)
    try:
        n_b = np.asarray(world_dir_to_base(sim_dir, np.asarray(face_normal_world, dtype=float), max_age_s), dtype=float)
        n_b[2] = 0.0
        return n_b / np.linalg.norm(n_b) if np.linalg.norm(n_b) > 1e-6 else np.zeros(3)
    except (FileNotFoundError, ValueError, OSError):
        return np.zeros(3)
