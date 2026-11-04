#!/usr/bin/env python3
"""One-off validation: capture head RGB + depth frames from the running sim
via WebRTC and save them exactly the way the reach endpoint does."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/dex/omniteleop/src")
from omniteleop.record.webrtc_receiver import WebRTCReceiver  # noqa: E402

OUT_DIR = Path("/home/ubuntu/dex/omniteleop/src/omniteleop/reach/captures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def grab(ip, port, name, timeout=15.0):
    rx = WebRTCReceiver(url=f"https://{ip}:{port}/offer", name=name)
    rx.start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        f = rx.get_latest_frame()
        if f is not None:
            rx.stop()
            return f
        time.sleep(0.2)
    rx.stop()
    raise RuntimeError(f"no frame from {name} ({ip}:{port})")


if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    rgb = grab(ip, 60001, "head_left_rgb")
    depth = grab(ip, 60003, "head_depth")
    cv2.imwrite(str(OUT_DIR / "reach_rgb.png"), rgb)
    depth_m = depth[..., 0].astype(np.float32) * (1.6 / 255.0)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(OUT_DIR / "reach_depth.png"), depth_mm)
    print(f"rgb frame: {rgb.shape} dtype={rgb.dtype}")
    print(f"depth frame: {depth.shape} min={depth_m.min():.3f}m max={depth_m.max():.3f}m "
          f"center={depth_m[240, 320]:.3f}m")
    print("saved to", OUT_DIR)
