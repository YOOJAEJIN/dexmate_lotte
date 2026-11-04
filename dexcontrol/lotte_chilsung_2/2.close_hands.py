# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Close both simulated slider grippers (same command path as omniteleop).

Example::

    python close_hands.py                 # both hands, teleop 'close' values
    python close_hands.py --side left
    python close_hands.py --value 0.5     # half closed
    python close_hands.py --dry-run       # show current state / target only
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hands import run  # noqa: E402


def main(
    side: str = "both",
    value: float | None = None,
    config: Path | None = None,
    hold_s: float = 0.5,
    wait: bool = True,
    settle_timeout: float = 3.0,
    dry_run: bool = False,
) -> None:
    """Close hand(s).

    Args:
        side: "left", "right" or "both".
        value: Optional blend 0..1 (0 = open, 1 = fully closed).
        config: omniteleop yaml providing hands.sim open/close values.
        hold_s: Seconds to keep re-publishing the target (20 Hz).
        wait: Wait for the sliders to stop moving and report positions.
        settle_timeout: Max seconds to wait for settling.
        dry_run: Read state and print the target without publishing.
    """
    run(
        "close",
        side=side,
        value=value,
        config=config,
        hold_s=hold_s,
        wait=wait,
        settle_timeout=settle_timeout,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    tyro.cli(main)
