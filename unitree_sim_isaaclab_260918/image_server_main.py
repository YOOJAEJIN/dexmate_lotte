# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
#!/usr/bin/env python3
"""Standalone image server for the Isaac Sim teleoperation stack.

Runs teleimager's ImageServer in its own process so the simulation main loop
does not compete with the image server threads for the GIL or CPU cores.

The server reads camera frames from the shared memory written by sim_main.py
(tools/shared_memory_utils.py), so it works no matter which side starts first.

Usage (same environment as sim_main.py, from the project root):

    # terminal 1 - optionally pin to cores you want to keep away from physics
    taskset -c 0-2 python image_server_main.py

    # terminal 2
    python sim_main.py --external_image_server ...

Note: only one image server may run at a time (the ZMQ config responser binds
port 60000 and each WebRTC stream binds its own port), so always pair this
script with sim_main.py --external_image_server.

Shutdown: Ctrl+C requests a graceful stop, then force-kills the process group
(mirroring teleimager's own workaround - lingering aiortc / multiprocessing
queue workers can otherwise block interpreter exit forever). Press Ctrl+C
twice to kill immediately.
"""
import functools
import os
import signal
import sys
import threading
import time

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PROJECT_ROOT", project_root)

# IsaacSimCamera imports tools.* from the repo, so the project root must be
# importable even when this script is launched from another directory.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from teleimager.image_server import run_isaacsim_server
except ImportError:
    # teleimager is a src-layout package next to this script; use it directly
    # when it is not installed into the environment (pip install -e teleimager).
    # The local ./teleimager directory would otherwise resolve as an empty
    # namespace package, so drop it before re-importing.
    sys.modules.pop("teleimager", None)
    sys.path.insert(0, os.path.join(project_root, "teleimager", "src"))
    from teleimager.image_server import run_isaacsim_server

# Grace period between the graceful stop request and the forced SIGKILL.
GRACE_SEC = 3.0


def _force_exit():
    """SIGKILL our whole process group (children included).

    This intentionally never returns. It is the same workaround teleimager's
    own main() uses, because lingering non-daemon workers (aiortc/multiprocessing
    log queue) can block a clean interpreter exit.
    """
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except Exception:
        pass
    os._exit(1)


def _delayed_force_exit():
    time.sleep(GRACE_SEC)
    _force_exit()


def main():
    # Own process group so the forced SIGKILL can never reach the parent shell.
    try:
        os.setpgrp()
    except Exception as e:
        print(f"[image_server_main] Failed to set process group: {e}")

    print("=" * 60)
    print("standalone image server started")
    print(f"project root: {project_root}")
    print("run sim_main.py with --external_image_server")
    print("=" * 60)

    server = run_isaacsim_server()

    stopping = threading.Event()

    def request_stop(signum, frame):
        # First Ctrl+C: graceful stop with a force-kill watchdog.
        # Second Ctrl+C: kill immediately.
        if stopping.is_set():
            print(f"\n[image_server_main] Received signal {signum} again, killing.", flush=True)
            _force_exit()
        stopping.set()
        print(f"\n[image_server_main] Received signal {signum}, stopping (Ctrl+C again to force kill)...", flush=True)
        threading.Thread(target=_delayed_force_exit, daemon=True).start()
        try:
            server.stop()
        except Exception as e:
            print(f"[image_server_main] Failed to stop server: {e}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print("[image_server_main] Running... Press Ctrl+C to exit.")
    server.wait()
    print("[image_server_main] Server stopped.")

    # Even after a clean wait() the interpreter may hang on exit; give logs a
    # moment to flush, then force-kill the group like teleimager's main() does.
    time.sleep(0.5)
    _force_exit()


if __name__ == "__main__":
    main()
