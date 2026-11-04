# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Open / close the simulated slider grippers exactly the way omniteleop does.

omniteleop (``follower/hand_controllers/sim_controller.py``, ee_type ``sim``)
publishes ``{"pos": [s1, s2]}`` with ``JointCmdCodec`` on
``<ROBOT_NAME>/control/custom_ee/{left,right}``; the simulator maps the two
values onto ``L_Slider_1/L_Slider_2`` (``R_…`` for the right hand) and clips
them to the joint limits (±0.04 m). The open/close targets come from
``omniteleop/src/omniteleop/configs/vega_1p.yaml`` →
``input_handlers.common.hands.sim.<side>.{open,close}`` (currently
open ``[0, 0]``, close ``[1, -1]``), so the scripts here read that file and
stay in sync with teleop.

The simulator keeps the last command, so a short burst of messages is enough;
the hands hold their target after this script exits.

Needs ``ROBOT_NAME`` (and ``ZENOH_CONFIG``) in the environment, same as the
other lotte_chilsung scripts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from dexcomm import Node
from dexcomm.codecs import JointCmdCodec, JointStateCodec
from loguru import logger

SIDES = ("left", "right")
DEFAULT_OPEN = [0.0, 0.0]
DEFAULT_CLOSE = [1.0, -1.0]

_DEX_ROOT = Path(__file__).resolve().parents[2]  # <dex>/  (this file lives at <dex>/dexcontrol/lotte_chilsung/)
DEFAULT_TELEOP_CONFIG = _DEX_ROOT / "omniteleop" / "src" / "omniteleop" / "configs" / "vega_1p.yaml"


def _find_sim_hands(cfg: dict) -> dict | None:
    """Locate the ``hands.sim`` block regardless of exact nesting."""
    try:
        return cfg["input_handlers"]["common"]["hands"]["sim"]
    except (KeyError, TypeError):
        pass
    stack = [cfg]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "sim" in node and isinstance(node["sim"], dict) and "left" in node["sim"]:
                return node["sim"]
            stack.extend(node.values())
    return None


def load_hand_targets(config_path: Path | None = None) -> dict[str, dict[str, list[float]]]:
    """Return ``{side: {"open": [...], "close": [...]}}`` from the teleop config."""
    path = config_path or DEFAULT_TELEOP_CONFIG
    targets = {s: {"open": list(DEFAULT_OPEN), "close": list(DEFAULT_CLOSE)} for s in SIDES}
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        sim = _find_sim_hands(cfg) if isinstance(cfg, dict) else None
        if sim is None:
            raise KeyError("hands.sim block not found")
        for s in SIDES:
            for key in ("open", "close"):
                vals = sim[s][key]
                targets[s][key] = [float(v) for v in vals]
        logger.info(f"Hand targets from {path}: {targets}")
    except Exception as exc:  # missing file / yaml / keys → fall back
        logger.warning(f"Could not read hand targets from {path} ({exc}); using defaults {targets}")
    return targets


class HandCommander:
    """Publish gripper targets and observe slider state via dexcomm/Zenoh."""

    def __init__(self, sides: Iterable[str] = SIDES, node_name: str = "lotte_hands"):
        if not os.environ.get("ROBOT_NAME") and not os.environ.get("ROBOT_CONFIG"):
            raise RuntimeError("ROBOT_NAME is not set; hand topics need the robot namespace.")
        self.sides = tuple(sides)
        self.node = Node(name=node_name)
        self._pubs = {
            s: self.node.create_publisher(f"control/custom_ee/{s}", encoder=JointCmdCodec.encode)
            for s in self.sides
        }
        self._state: dict[str, dict] = {}
        self._subs = {
            s: self.node.create_subscriber(
                f"state/custom_ee/{s}", self._make_cb(s), decoder=JointStateCodec.decode
            )
            for s in self.sides
        }

    def _make_cb(self, side: str):
        def _cb(msg):
            if isinstance(msg, dict) and "pos" in msg:
                self._state[side] = msg
        return _cb

    def wait_for_state(self, timeout: float = 3.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(s in self._state for s in self.sides):
                return True
            time.sleep(0.05)
        return all(s in self._state for s in self.sides)

    def state(self, side: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        m = self._state.get(side)
        if not m:
            return None, None
        pos = np.asarray(m.get("pos", []), dtype=float)
        vel = np.asarray(m.get("vel", []), dtype=float) if m.get("vel") else None
        return pos, vel

    def publish(self, side: str, pos: list[float]) -> None:
        self._pubs[side].publish({"pos": [float(v) for v in pos]})

    def command(self, targets: dict[str, list[float]], hold_s: float = 0.5, hz: float = 20.0) -> None:
        """Send each side's target repeatedly for ``hold_s`` seconds."""
        n = max(int(hold_s * hz), 1)
        for _ in range(n):
            for s, pos in targets.items():
                self.publish(s, pos)
            time.sleep(1.0 / hz)

    def wait_settled(self, timeout: float = 3.0, vel_tol: float = 2e-3, still_s: float = 0.3) -> dict[str, np.ndarray]:
        """Block until slider velocities stay below ``vel_tol`` for ``still_s``."""
        t0 = time.time()
        still_since: float | None = None
        while time.time() - t0 < timeout:
            vels = []
            for s in self.sides:
                _, v = self.state(s)
                if v is None or v.size == 0:
                    vels = None
                    break
                vels.append(float(np.max(np.abs(v))))
            if vels is not None and max(vels) < vel_tol:
                if still_since is None:
                    still_since = time.time()
                elif time.time() - still_since >= still_s:
                    break
            else:
                still_since = None
            time.sleep(0.05)
        return {s: self.state(s)[0] for s in self.sides}

    def shutdown(self) -> None:
        try:
            self.node.shutdown()
        except Exception:
            pass


def run(
    action: str,
    side: str = "both",
    value: float | None = None,
    config: Path | None = None,
    hold_s: float = 0.5,
    wait: bool = True,
    settle_timeout: float = 3.0,
    dry_run: bool = False,
) -> None:
    """Open or close hands.

    Args:
        action: "open" or "close".
        side: "left", "right" or "both".
        value: Optional blend 0..1 between open (0) and close (1); overrides
            ``action``'s endpoint. E.g. 0.5 = half closed.
        config: omniteleop yaml with hands.sim open/close values.
        hold_s: How long to keep re-publishing the target.
        wait: Wait for the sliders to stop moving and report the result.
        settle_timeout: Max seconds to wait for settling.
        dry_run: Only read the current state and print what would be sent.
    """
    if action not in ("open", "close"):
        raise ValueError("action must be 'open' or 'close'")
    sides = SIDES if side == "both" else (side,)
    if any(s not in SIDES for s in sides):
        raise ValueError("side must be left, right or both")
    targets_cfg = load_hand_targets(config)
    blend = (1.0 if action == "close" else 0.0) if value is None else float(np.clip(value, 0.0, 1.0))
    targets = {}
    for s in sides:
        o = np.asarray(targets_cfg[s]["open"], dtype=float)
        c = np.asarray(targets_cfg[s]["close"], dtype=float)
        targets[s] = ((1.0 - blend) * o + blend * c).tolist()

    hc = HandCommander(sides)
    try:
        have_state = hc.wait_for_state(timeout=3.0)
        for s in sides:
            pos, _ = hc.state(s)
            logger.info(
                f"{s:5s} current sliders {np.round(pos, 4) if pos is not None else 'n/a'}  "
                f"→ target {np.round(targets[s], 3)} ({action}, blend={blend:g})"
            )
        if not have_state:
            logger.warning("No state/custom_ee messages received — is the sim running with --enable_zenoh_hand vertical?")
        if dry_run:
            logger.info("Dry run: nothing published.")
            return
        hc.command(targets, hold_s=hold_s)
        if wait:
            final = hc.wait_settled(timeout=settle_timeout)
            for s in sides:
                p = final.get(s)
                logger.info(f"{s:5s} settled sliders {np.round(p, 4) if p is not None else 'n/a'}")
        logger.info(f"Hands {action}: done ({', '.join(sides)}). Sim holds the last target.")
    finally:
        hc.shutdown()
