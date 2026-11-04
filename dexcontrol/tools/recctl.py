#!/usr/bin/env python3
"""Terminal control for the sim's keyed action recorder (tools/action_recorder.py).

Writes a one-line command file that the running simulator picks up at its next
step, so no viewport focus is needed. Same commands as the F-keys::

    python tools/recctl.py rec        # F5  start / stop recording a segment
    python tools/recctl.py play       # F6  play all segments (restore first snapshot)
    python tools/recctl.py play-here  # F7  play all from the current state
    python tools/recctl.py next       # F8  play the next segment only
    python tools/recctl.py stop       # F9  stop playback
    python tools/recctl.py delete     # F10 delete last segment (send twice within 2 s)
    python tools/recctl.py reload     # F3  reload segments from disk
    python tools/recctl.py status     #     print record_status.txt and the segment list

Pick segments by file number (seg_003 -> 3), several, or a range::

    python tools/recctl.py play 3            # seg_003 only, its start state restored first
    python tools/recctl.py play 2 4 5        # in this order
    python tools/recctl.py play 2-4
    python tools/recctl.py play-here 3       # from the current state
    python tools/recctl.py restore 3         # only restore seg_003's start state
    python tools/recctl.py goto 3            # F8 / next will play seg_003
    python tools/recctl.py list              # segments with lengths (printed by the sim)
    python tools/recctl.py release           # drop the post-playback hold (all joints follow live commands)
    python tools/recctl.py speed 2           # playback speed x2 (0.25-8), used by F6/F7/F8 and play
    python tools/recctl.py video off|on      # save an mp4 of each playback (recordings/videos/), default on
    python tools/recctl.py yawtrack off|on   # closed-loop base yaw on playback (default on; needs segments
                                             #  recorded on/after 2026-09-28, older ones replay open-loop)

Use ``--record-dir`` if the sim was started with a non-default ``--record_dir``.
The sim is assumed to live next to this repo as ``../unitree_sim_isaaclab``;
set ``UNITREE_SIM_ROOT`` to override.
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The recorder runs inside the simulator (unitree_sim_isaaclab), a sibling of this
# repo; it watches <sim>/recordings/cmd.txt and writes <sim>/record_status.txt.
SIM_ROOT = os.environ.get(
    "UNITREE_SIM_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "unitree_sim_isaaclab")
)
# The only record_dir we use: <dexcontrol>/recordings (sim_main.py defaults to it as well).
# The old <sim>/recordings fallback was removed on purpose so that a stale copy there can
# never be picked up by mistake; pass --record-dir explicitly if you really need another one.
RECORD_DIR_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "recordings"),
)


def default_record_dir() -> str:
    for d in RECORD_DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    return RECORD_DIR_CANDIDATES[0]


def status_path_for(record_dir: str) -> str:
    """record_status.txt sits next to the sim's record_dir; prefer the most recently written one."""
    primary = os.path.join(os.path.dirname(os.path.abspath(record_dir)), "record_status.txt")
    candidates = [primary] + [os.path.join(os.path.dirname(c), "record_status.txt") for c in RECORD_DIR_CANDIDATES]
    existing = [c for c in dict.fromkeys(candidates) if os.path.isfile(c)]
    return max(existing, key=os.path.getmtime) if existing else primary
COMMANDS = ("rec", "play", "play-here", "next", "stop", "delete", "reload", "status", "restore", "goto", "list", "release", "speed", "video", "yawtrack")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("selectors", nargs="*", help="segment numbers / ranges / names (play, restore, goto) or a speed factor (speed)")
    ap.add_argument("--record-dir", default=default_record_dir(),
                    help="the sim's --record_dir (default: first existing of %s)" % " , ".join(RECORD_DIR_CANDIDATES))
    ap.add_argument("--wait", type=float, default=1.5, help="seconds to wait for the sim to consume the command")
    args = ap.parse_args()

    status_path = status_path_for(args.record_dir)
    if args.command == "status":
        try:
            print(open(status_path, encoding="utf-8").read().strip())
        except OSError:
            print("no record_status.txt yet (is the sim running with the recorder?)")
        segs = sorted(f for f in os.listdir(args.record_dir) if f.startswith("seg_") and f.endswith(".npz")) if os.path.isdir(args.record_dir) else []
        print(f"{len(segs)} segment(s) in {args.record_dir}: {', '.join(segs) if segs else '-'}")
        return 0

    # The running sim creates its record_dir itself; if it is missing here, we are
    # pointing at the wrong place. Creating it would make the sim rescan an empty
    # folder and drop every loaded segment ("have []", segments=0).
    if not os.path.isdir(args.record_dir):
        print(f"record dir not found: {args.record_dir}\n"
              "-> is the sim running, and was it started with this --record_dir? "
              "(pass --record-dir or set UNITREE_SIM_ROOT)")
        return 2
    cmd_path = os.path.join(args.record_dir, "cmd.txt")
    with open(cmd_path, "a", encoding="utf-8") as f:
        f.write(" ".join([args.command, *args.selectors]) + "\n")

    deadline = time.time() + args.wait
    while time.time() < deadline and os.path.exists(cmd_path):
        time.sleep(0.05)
    if os.path.exists(cmd_path):
        print(f"'{args.command}' written to {cmd_path} but not picked up within {args.wait:.1f}s.\n"
              "-> is the sim running, and is this its --record_dir? "
              f"(sim_main.py defaults to {RECORD_DIR_CANDIDATES[0]}; older launches used {RECORD_DIR_CANDIDATES[1]})")
        return 1
    time.sleep(0.2)
    try:
        print(open(status_path, encoding="utf-8").read().strip())
    except OSError:
        print(f"'{args.command}' sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
