#!/usr/bin/env python3
"""Targeting System (was "HEAD SEG"): SEG mode = box-pile segmentation + base-frame coordinates
+ grasp overlay; EDGE mode = place-target edges once a box is held upright (see TGT_* below).

  relay 60001 RGB tap (zmq 55601) ─┬─ every SCAN_S: FastSAM full frame
  nano metric depth (TCP 55703, mm) ┘   -> box candidates (area/border/depth)
  zenoh sensors/head_camera/imu    ──> gravity in the camera frame
  zenoh robot/joints (fresh only)  ──> pinocchio FK -> T(base<-zed_left_camera)

  per scan : each box -> front face = vertical plane (gravity) -> rectangle
             -> 4 corners, width/height, 6.1 cm layer lines down both edges,
             grasp = centre of the top 20-sheet bundle -> logs/box_coords.json
             (box size and layers need only camera + IMU, so they do not
             depend on the torso/head posture; BASE XYZ needs live joints)
             focus = extreme points of all boxes, padded to the frame aspect
  per frame: bicubic zoom to focus -> FastSAM overlay -> markers/labels
             -> x264 RTP udp 50006 -> relay 60006 -> GUI tile "head seg"

Needs a python with torch(CUDA) + ultralytics + opencv + pyzmq + pinocchio
(`pip install pin`). See README.md for the full setup.
"""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import zmq

RGB_TAP = os.getenv("SEG_TAP", "tcp://127.0.0.1:55601")
# Metric depth (uint16 mm) straight from ZED retrieve_measure on the nano
# (gstsender METRIC_DEPTH_PORT). Not the 60003 view stream: that is a display
# rendering whose scale shifts with the scene.
DEPTH_SRC = os.getenv("SEG_DEPTH_SRC", "192.168.50.22:55703")
DEST = os.getenv("SEG_DEST", "127.0.0.1")
PORT = int(os.getenv("SEG_PORT", "50006"))
MODEL = os.getenv("SEG_MODEL", "FastSAM-s.pt")
CONF = float(os.getenv("SEG_CONF", "0.4"))
SCAN_S = float(os.getenv("SEG_SCAN_S", "2.0"))
# Live robot feeds over zenoh (dexcomm; needs ROBOT_NAME + ZENOH_CONFIG in the env).
IMU_TOPIC = os.getenv("SEG_IMU_TOPIC", "sensors/head_camera/imu")  # nano gstsender, ~140 Hz
JOINTS_TOPIC = os.getenv("SEG_JOINTS_TOPIC", "robot/joints")       # robot_controller, 100 Hz
# GUI "Lotte Reach": REQ here -> scan now with frames newer than the request -> JSON reply
RESCAN_ADDR = os.getenv("SEG_RESCAN", "tcp://127.0.0.1:55610")
FEED_MAX_AGE_S = 1.0  # older than this = dead feed, never reuse (2026-11-12: a cached
                      # pose from a crashed robot_controller put the camera 22 deg off)
_HERE = Path(__file__).resolve().parent
COORDS_JSON = os.getenv("SEG_COORDS_JSON", str(_HERE / "logs" / "box_coords.json"))
URDF = os.getenv(
    "SEG_URDF",
    str(_HERE.parent / "omniteleop/external/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf"),
)
# Targeting System (2026-11-18): SEG mode = the box segmentation below. EDGE mode = a box is
# held AND turned upright (place_edges/targeting.py: gripper sensor + gripper tilt) -> FastSAM
# off, place-target edges (place_edges/edges.py) on HD1200 snapshots instead.
TGT_SIDE = os.getenv("TGT_SIDE", "left")  # right KORAS sends no status replies (RS485, 2026-11-18)
HIRES_SRC = os.getenv("TGT_HIRES_SRC", "192.168.50.22:55704")  # nano gstsender HIRES_PORT
EDGE_S = float(os.getenv("TGT_EDGE_S", "1.5"))  # one HD snapshot + detection per EDGE_S
EDGE_MEDIAN_N = 5  # shown / saved = the result nearest the median K of the last N good ones
PLACE_JSON = os.getenv("TGT_PLACE_JSON", str(_HERE / "logs" / "place_target.json"))
# Base-frame point the edge search is centred on (H5 x V2 corner measured 2026-11-18); it then
# follows the detected corner, so head / torso moves keep the target inside the tuned ROI.
TGT_ANCHOR = [float(x) for x in os.getenv("TGT_ANCHOR", "1.124,0.409,1.055").split(",")]
# vega_1p.yaml reach.camera / camera_mount_rpy_corr
# ZED factory calibration SN50691325 [LEFT_CAM_FHD1200] (1920x1200), scaled to the
# 640x480 stream; gstsender resizes non-uniformly (x 1/3, y 0.4) so fx != fy.
FX, FY, CX, CY = 735.25 / 3, 735.141 * 0.4, 992.571 / 3, 618.88 * 0.4
DEPTH_VALID = (0.30, 1.60)
# Pure URDF FK on the real robot. reach's camera_mount_rpy_corr (-0.315, 0.0377,
# 0.0941) was tuned for the Isaac Lab render: on hardware it tilts vertical box
# edges ~20 deg; with zero they come out ~2 deg (best fit roll +0.025 rad).
MOUNT_RPY = (0.0, 0.0, 0.0)
# Cardboard-brown gate (HSV). Hue is ~16 for box and grey crate alike, so
# saturation separates them: box S~130 (11-10) but only S~89 (11-13, other
# light / exposure), grey crate S~67. S>=80 on >=50% of the inner box: box
# 0.80-0.88, box+wall superset 0.39, grey crate < 0.5 by its median.
# 2026-11-17: the box is now a small bundle straight ahead and only 0.32-0.34
# brown at 0.86 m; position (CENTER_FRAC) is the main gate, colour only keeps
# out the grey wall, white arm and black gripper (0.00-0.03).
BROWN_MIN_FRAC = float(os.getenv("SEG_BROWN_MIN", "0.25"))  # 0 disables
BROWN_H = (int(os.getenv("SEG_BROWN_H0", "5")), int(os.getenv("SEG_BROWN_H1", "30")))
BROWN_S_MIN = int(os.getenv("SEG_BROWN_S", "80"))
BROWN_V_MIN = int(os.getenv("SEG_BROWN_V", "50"))
# Sanity bounds on the measured front face (metres): drops specks and
# wall-sized masks. The box itself ranges from a 15x14 cm bundle (2026-11-17)
# to the 40.5x45.5 cm pile (2026-11-12). 0 disables.
BOX_MIN_W = float(os.getenv("SEG_BOX_MIN_W", "0.05"))
BOX_MIN_H = float(os.getenv("SEG_BOX_MIN_H", "0.03"))
# Smallest candidate bbox as a fraction of the image (the 15x14 cm box at
# 0.86 m is 0.0069; specks below this are noise).
MIN_AREA = float(os.getenv("SEG_MIN_AREA", "0.002"))
BOX_MAX = float(os.getenv("SEG_BOX_MAX", "1.2"))
# The pile is always straight ahead (operator, 2026-11-17): a candidate's centre
# must lie in the middle CENTER_FRAC of the image width. 0 disables.
CENTER_FRAC = float(os.getenv("SEG_CENTER_FRAC", "0.5"))
# Layer pitch along the box's vertical edges: 20 cardboard sheets = 6.1 cm.
PITCH = float(os.getenv("SEG_PITCH", "0.061"))


def project(p_cam):
    """Camera-frame 3D point -> pixel (u, v)."""
    return [int(round(FX * p_cam[0] / p_cam[2] + CX)), int(round(FY * p_cam[1] / p_cam[2] + CY))]


def layer_lines(corners_cam, T):
    """Cut lines every PITCH metres down both vertical edges, from the top.

    Walking along the 3D edge (not pixels) keeps spacing metric under
    perspective / head tilt. Returns (layers, grasp) where grasp is the
    centre of the top bundle (between top edge and the first cut line).
    """
    if len(corners_cam) != 4:
        return [], None
    pts = sorted(corners_cam, key=lambda c: c["px"][1])
    TL, TR = sorted(pts[:2], key=lambda c: c["px"][0])
    BL, BR = sorted(pts[2:], key=lambda c: c["px"][0])
    tl, tr, bl, br = (np.array(c["cam"]) for c in (TL, TR, BL, BR))
    dl, dr = bl - tl, br - tr
    ll, lr = np.linalg.norm(dl), np.linalg.norm(dr)
    if ll < 1e-3 or lr < 1e-3:
        return [], None
    n = int(min(ll, lr) // PITCH)
    absf = (lambda q: T.act(q).tolist()) if T is not None else (lambda q: q.tolist())
    layers = []
    for k in range(1, n + 1):
        a = tl + dl * (k * PITCH / ll)
        b = tr + dr * (k * PITCH / lr)
        mid = (a + b) / 2
        layers.append({"k": k,
                       "left": {"px": project(a), "xyz": absf(a)},
                       "right": {"px": project(b), "xyz": absf(b)},
                       "mid": absf(mid)})
    if n >= 1:
        top_mid = (tl + tr) / 2
        cut1_mid = (tl + dl * (PITCH / ll) + tr + dr * (PITCH / lr)) / 2
        g = (top_mid + cut1_mid) / 2
    else:
        g = (tl + tr + bl + br) / 4
    return layers, {"px": project(g), "xyz": absf(g)}

from ultralytics import FastSAM, YOLO  # noqa: E402
import pinocchio as pin  # noqa: E402

import box_hold  # noqa: E402
import face_plane  # noqa: E402

sys.path.insert(0, str(_HERE / "place_edges"))
import edges as place_edges  # noqa: E402
import targeting  # noqa: E402
from hires_grab import read_hires  # noqa: E402

_held = None  # last unoccluded grasp box (box_hold)


def reproject(box, T):
    """Pixel positions of a held box for the current head pose."""
    if T is None or box["xyz_base"] is None:
        return box  # ponytail: camera-frame hold assumes the head hasn't moved
    Ti = T.inverse()
    px = lambda xyz: project(Ti.act(np.array(xyz)))  # noqa: E731
    b = copy.deepcopy(box)
    for k in b["corners"]:
        k["px"] = px(k["xyz"])
    for L in b["layers"]:
        for s in ("left", "right"):
            L[s]["px"] = px(L[s]["xyz"])
    if b["grasp"]:
        b["grasp"]["px"] = px(b["grasp"]["xyz"])
    b["c"] = tuple(px(b["xyz_base"]))
    us, vs = zip(*(k["px"] for k in b["corners"]))
    b["bbox"] = (min(us), min(vs), max(us), max(vs))
    return b

Path(COORDS_JSON).parent.mkdir(parents=True, exist_ok=True)
model = (FastSAM if "fastsam" in MODEL.lower() else YOLO)(MODEL)

pin_model = pin.buildModelFromUrdf(URDF)
pin_data = pin_model.createData()
CAM_FRAME = pin_model.getFrameId("zed_left_camera")
EE_FRAME = pin_model.getFrameId(f"{TGT_SIDE[0].upper()}_ee")
R_CORR = pin.rpy.rpyToMatrix(*MOUNT_RPY)
_JOINT_GROUPS = {
    "left_arm": [f"L_arm_j{i+1}" for i in range(7)],
    "right_arm": [f"R_arm_j{i+1}" for i in range(7)],
    "torso": [f"torso_j{i+1}" for i in range(3)],
    "head": [f"head_j{i+1}" for i in range(3)],
}

_feed_lock = threading.Lock()
_imu = []                           # (t_rx, acc[3]) of the last ~0.5 s
_joints = {"t": 0.0, "q": None}     # latest robot/joints message
_hand = {"t_ns": None, "t": 0.0, "status": None}  # TGT_SIDE gripper status; t = when its reply last CHANGED
_link = {}                          # keeps the node + subscribers alive


def _on_imu(m):
    try:
        a = np.asarray(m["acc"], float).reshape(3)
    except Exception:
        return
    now = time.time()
    with _feed_lock:
        _imu.append((now, a))
        while _imu and now - _imu[0][0] > 0.5:
            _imu.pop(0)


def _on_joints(m):
    j = m.get("joints", m) if isinstance(m, dict) else None
    if isinstance(j, dict):
        hs = (m.get("hands") or {}).get(TGT_SIDE)  # robot_controller copies the KORAS status here
        with _feed_lock:
            _joints["t"], _joints["q"] = time.time(), j
            if hs and hs.get("t_ns") != _hand["t_ns"]:  # a repeated reply time = monitor stalled
                _hand.update(t_ns=hs.get("t_ns"), t=time.time(), status=hs)


def _robot_link():
    """Subscribe in the background: a missing zenoh setup must never stall the video."""
    try:
        from dexcomm import Node
        from dexcomm.codecs import DictDataCodec, IMUDataCodec
        node = Node(name="seg_overlay")
        _link["subs"] = [node.create_subscriber(IMU_TOPIC, _on_imu, decoder=IMUDataCodec.decode),
                         node.create_subscriber(JOINTS_TOPIC, _on_joints, decoder=DictDataCodec.decode)]
        _link["node"] = node
        print(f"[seg] robot link up: {node.resolve_topic(IMU_TOPIC)}, {node.resolve_topic(JOINTS_TOPIC)}", flush=True)
    except Exception as e:
        print(f"[seg] robot link disabled ({type(e).__name__}: {e}) - no gravity / base frame", flush=True)


threading.Thread(target=_robot_link, daemon=True).start()


def gripper_tilt_deg():
    """Angle between the TGT_SIDE gripper x axis and vertical (deg); valid right after a
    successful cam_to_base_T() (same FK pass). 0.4 deg in the place pose = box upright."""
    return float(np.degrees(np.arccos(np.clip(pin_data.oMf[EE_FRAME].rotation[2, 0], -1.0, 1.0))))


def cam_to_base_T():
    """T(base <- optical camera) from LIVE joints, or None (stale / no feed)."""
    with _feed_lock:
        fresh = time.time() - _joints["t"] < FEED_MAX_AGE_S
        joints = _joints["q"] if fresh else None
    if not joints:
        return None
    q = pin.neutral(pin_model)
    for group, names in _JOINT_GROUPS.items():
        vals = joints.get(group)
        if vals is None or len(vals) == 0:
            continue
        for jname, val in zip(names, vals):
            jid = pin_model.getJointId(jname)
            q[pin_model.idx_qs[jid]] = float(val)
    q = np.clip(q, pin_model.lowerPositionLimit, pin_model.upperPositionLimit)
    pin.framesForwardKinematics(pin_model, pin_data, q)
    return pin_data.oMf[CAM_FRAME] * pin.SE3(R_CORR, np.zeros(3))


def gravity_up(T):
    """(world-up in the camera frame, source). Head IMU first: it sits in the camera,
    so it is right whatever the torso / head joints or the URDF say; FK is the fallback."""
    with _feed_lock:
        acc = [a for _, a in _imu]
        last = _imu[-1][0] if _imu else 0.0
    if len(acc) >= 5 and time.time() - last < FEED_MAX_AGE_S:
        g = np.mean(acc, axis=0)
        gn = float(np.linalg.norm(g))
        if 9.0 < gn < 10.6:  # at rest the accelerometer reads +g along up
            return g / gn, "IMU"
    if T is not None:
        return T.rotation.T @ np.array([0.0, 0.0, 1.0]), "FK"
    return None, "-"


ctx = zmq.Context()


def _sub(addr):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.connect(addr)
    return s


rgb_sub = _sub(RGB_TAP)
_depth_latest = None
_depth_t = 0.0  # arrival time of _depth_latest


def _depth_reader():
    """Keep _depth_latest = newest metric depth frame; reconnect forever."""
    global _depth_latest, _depth_t
    host, port = DEPTH_SRC.rsplit(":", 1)
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=5) as s:
                f = s.makefile("rb")
                first = True
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        break
                    w, h, n = struct.unpack(">HHI", hdr)
                    buf = f.read(n)
                    if len(buf) < n:
                        break
                    # The sender hands out its latest frame on connect - which
                    # is ancient if its grab loop died (camera taken by a 2nd
                    # sender, 2026-11-12). Trust only frames that keep coming.
                    if first:
                        first = False
                        continue
                    _depth_latest = np.frombuffer(buf, dtype="<u2").reshape(h, w)
                    _depth_t = time.time()
        except OSError:
            pass
        _depth_latest = None  # stale depth is worse than none
        time.sleep(1)


threading.Thread(target=_depth_reader, daemon=True).start()

# On-demand scan ("Reach" in the GUI): after the head or torso moved, detect +
# segment again and return the new grasp point - only from frames that arrived
# after the request.
_rescan = {"t": 0.0, "result": None}
_rescan_want, _rescan_done = threading.Event(), threading.Event()


def _rescan_server():
    s = ctx.socket(zmq.REP)
    s.bind(RESCAN_ADDR)
    while True:
        s.recv()
        _rescan_done.clear()
        _rescan["t"] = time.time()
        _rescan_want.set()
        if _rescan_done.wait(10.0):
            s.send_json(_rescan["result"])
        else:
            _rescan_want.clear()
            s.send_json({"ok": False, "error": "no fresh camera + depth frame within 10 s"})


threading.Thread(target=_rescan_server, daemon=True).start()


def _decode(raw):
    w = int.from_bytes(raw[0:2], "big")
    h = int.from_bytes(raw[2:4], "big")
    return np.frombuffer(raw[4:], dtype=np.uint8).reshape(h, w, 3)


gst, gst_wh = None, None


def ensure_gst(w, h):
    global gst, gst_wh
    if gst is not None and gst_wh == (w, h) and gst.poll() is None:
        return gst
    if gst is not None:
        try:
            gst.stdin.close()
            gst.terminate()
        except Exception:
            pass
    cmd = (
        f"gst-launch-1.0 fdsrc do-timestamp=true ! "
        f"rawvideoparse width={w} height={h} format=bgr framerate=30/1 ! "
        f"videoconvert ! x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast key-int-max=30 ! "
        f"rtph264pay pt=96 config-interval=1 ! udpsink host={DEST} port={PORT} sync=false async=false"
    )
    gst = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE)
    gst_wh = (w, h)
    return gst


def depth_at(depth_img, u, v, k=4):
    if depth_img is None:
        return None
    h, w = depth_img.shape[:2]
    y0, y1 = max(0, v - k), min(h, v + k + 1)
    x0, x1 = max(0, u - k), min(w, u + k + 1)
    d = depth_img[y0:y1, x0:x1].astype(np.float32) / 1000.0
    d = d[(d > DEPTH_VALID[0]) & (d < DEPTH_VALID[1])]
    return float(np.median(d)) if d.size >= 5 else None


def fit_aspect(x0, y0, x1, y1, w, h, margin=0.06):
    """Pad bbox by margin, expand to the frame aspect ratio, clamp into frame."""
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * margin; x1 += bw * margin
    y0 -= bh * margin; y1 += bh * margin
    bw, bh = x1 - x0, y1 - y0
    target = w / h
    if bw / max(1e-6, bh) < target:  # too tall -> widen
        grow = bh * target - bw
        x0 -= grow / 2; x1 += grow / 2
    else:                            # too wide -> heighten
        grow = bw / target - bh
        y0 -= grow / 2; y1 += grow / 2
    # shift inside frame, then clamp
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= (x1 - w); x1 = w
    if y1 > h: y0 -= (y1 - h); y1 = h
    return (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))


def _free_face(m, depth_img, cu, cv_, z):
    """Fallback without gravity: free plane through the mask + extreme hull points.

    View-dependent: from above, the top face is in the mask and counts as
    height. -> (centre, corners ordered BR, BL, TL, TR), all camera frame.
    """
    vv, uu = np.nonzero(cv2.erode(m, np.ones((5, 5), np.uint8)))
    zz = depth_img[vv, uu].astype(np.float32) / 1000.0
    ok = (zz > DEPTH_VALID[0]) & (zz < DEPTH_VALID[1])
    uu, vv, zz = uu[ok], vv[ok], zz[ok]
    plane = face_plane.fit(np.stack([(uu - CX) * zz / FX, (vv - CY) * zz / FY, zz], 1))

    def on_face(u, v, fallback_z):
        ray = np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
        p = face_plane.ray_hit(ray, plane) if plane is not None else None
        if p is None or not (DEPTH_VALID[0] < p[2] < DEPTH_VALID[1]):
            p = ray * fallback_z
        return p

    pts = []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2)
        # Extreme hull points along the 4 diagonals. Unlike minAreaRect this
        # keeps the perspective trapezoid of a face seen at an angle.
        for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            px, py = hull[np.argmax(hull[:, 0] * dx + hull[:, 1] * dy)].astype(float)
            su, sv = int(0.85 * px + 0.15 * cu), int(0.85 * py + 0.15 * cv_)
            pts.append(on_face(px, py, depth_at(depth_img, su, sv) or z))
    return on_face(cu, cv_, z), pts


_grav_src = "-"  # where the last scan's gravity came from (shown on the overlay)


def box_sized(size):
    """Front face [width, height] (m) the size of the box pile? None = not measurable."""
    if not BOX_MIN_W:
        return True
    if size is None:
        return False
    w_, h_ = size
    return BOX_MIN_W <= w_ <= BOX_MAX and BOX_MIN_H <= h_ <= BOX_MAX


assert box_sized([0.417, 0.473]) and box_sized([0.41, 0.061]) and box_sized([0.15, 0.14])  # pile / last bundle / small box
assert not box_sized([0.02, 0.02]) and not box_sized(None) and not box_sized([2.0, 0.5])      # speck / no face / wall


def scan(frame, depth_img):
    """FastSAM full frame -> box candidates with camera+base XYZ, focus bbox."""
    h, w = frame.shape[:2]
    res = model(frame, verbose=False, device=0, conf=CONF, retina_masks=True)[0]
    if res.boxes is None or res.masks is None:
        return None, []
    global _grav_src
    T = cam_to_base_T()
    up, _grav_src = gravity_up(T)
    cands = []
    for box, mask in zip(res.boxes, res.masks.data):
        x0, y0, x1, y1 = [int(v) for v in box.xyxy[0]]
        bw, bh = x1 - x0, y1 - y0
        area_frac = (bw * bh) / (w * h)
        if not (MIN_AREA <= area_frac <= 0.35):
            continue
        if CENTER_FRAC and abs((x0 + x1) / 2 - w / 2) > CENTER_FRAC * w / 2:
            continue  # off to the side: not the pile in front
        if bw / w > 0.85 or bh / h > 0.85:
            continue
        if (x0 < 3) + (y0 < 3) + (x1 > w - 3) + (y1 > h - 3) >= 2:
            continue
        if BROWN_MIN_FRAC > 0:
            # only brown (cardboard) objects count as boxes
            iy0, iy1 = y0 + bh // 5, max(y0 + bh // 5 + 1, y1 - bh // 5)
            ix0, ix1 = x0 + bw // 5, max(x0 + bw // 5 + 1, x1 - bw // 5)
            hsv = cv2.cvtColor(frame[iy0:iy1, ix0:ix1], cv2.COLOR_BGR2HSV)
            brown = ((hsv[..., 0] >= BROWN_H[0]) & (hsv[..., 0] <= BROWN_H[1]) &
                     (hsv[..., 1] >= BROWN_S_MIN) & (hsv[..., 2] >= BROWN_V_MIN))
            if float(brown.mean()) < BROWN_MIN_FRAC:
                continue
        cu, cv_ = (x0 + x1) // 2, (y0 + y1) // 2
        z = depth_at(depth_img, cu, cv_)
        if z is None:
            continue
        m = mask.cpu().numpy().astype(np.uint8)
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m, (w, h))
        pts = None
        if up is not None:
            # Upright pile + gravity: the front face is a vertical plane and the
            # box a rectangle on it - same result from any camera height / tilt.
            vv, uu = np.nonzero(m)
            zz = depth_img[vv, uu].astype(np.float32) / 1000.0
            zz[(zz <= DEPTH_VALID[0]) | (zz >= DEPTH_VALID[1])] = 0
            rays = np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones(len(uu))], axis=1)
            face = face_plane.rect(rays, zz, up, z / FX)
            if face is not None:
                pts, xyz_cam = list(face[0]), face[0].mean(axis=0)
        if pts is None:
            xyz_cam, pts = _free_face(m, depth_img, cu, cv_, z)
        xyz_base = T.act(xyz_cam).tolist() if T is not None else None
        corners = [{"px": project(c), "xyz": (T.act(c) if T is not None else c).tolist(),
                    "cam": c.tolist()} for c in pts]
        size = None
        if len(pts) == 4:  # BR, BL, TL, TR -> mean width, mean height (m)
            d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))  # noqa: E731
            size = [round((d(3, 2) + d(0, 1)) / 2, 4), round((d(1, 2) + d(0, 3)) / 2, 4)]
        if not box_sized(size):
            continue
        layers, grasp = layer_lines(corners, T)

        cands.append({"bbox": (x0, y0, x1, y1), "c": (cu, cv_),
                      "xyz_cam": xyz_cam.tolist(), "xyz_base": xyz_base,
                      "corners": corners, "layers": layers, "grasp": grasp, "size_m": size})
    if not cands:
        return None, []
    # Keep only the box PILE: grow a cluster from the candidate nearest the
    # image centre (the pile is straight ahead) and absorb neighbours whose bbox
    # touches the (slightly expanded) union. Seeding from the largest let a big
    # brown thing elsewhere win over the pile. Clutter far from it drops out.
    rest = sorted(cands, key=lambda c: abs(c["c"][0] - w / 2) + abs(c["c"][1] - h / 2))
    cluster = [rest.pop(0)]
    gap = 0.06 * (w + h) / 2  # allowed spacing between pile members (px)
    changed = True
    while changed and rest:
        changed = False
        ux0 = min(c["bbox"][0] for c in cluster) - gap
        uy0 = min(c["bbox"][1] for c in cluster) - gap
        ux1 = max(c["bbox"][2] for c in cluster) + gap
        uy1 = max(c["bbox"][3] for c in cluster) + gap
        for c in rest[:]:
            x0, y0, x1, y1 = c["bbox"]
            if x0 < ux1 and x1 > ux0 and y0 < uy1 and y1 > uy0:
                cluster.append(c)
                rest.remove(c)
                changed = True
    global _held
    cands = sorted(cluster, key=lambda c: c["c"][1])  # topmost (grasp target) first
    # Gripper covering the grasp box -> keep its last full measurement.
    cands[0], _held = box_hold.choose(_held, cands[0])
    if cands[0].get("held"):
        cands[0] = reproject(cands[0], T)
    # Focus = extreme points of the pile, fitted to frame aspect
    xs0 = min(c["bbox"][0] for c in cands); ys0 = min(c["bbox"][1] for c in cands)
    xs1 = max(c["bbox"][2] for c in cands); ys1 = max(c["bbox"][3] for c in cands)
    focus = fit_aspect(xs0, ys0, xs1, ys1, w, h, margin=0.04)
    return focus, cands


_tmode = targeting.Mode()
_edge = {"busy": False, "t": 0.0, "hist": [], "show": None, "last": None, "anchor": np.array(TGT_ANCHOR)}


def _edge_job(T):
    """One HD1200 snapshot -> place-target edges (background thread: ~0.6 s, video keeps going)."""
    try:
        host, port = HIRES_SRC.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=5) as s:
            im, dmm = read_hires(s.makefile("rb"))
        r = place_edges.detect_tracked(im, dmm.astype(np.float32) / 1000.0, T.rotation, T.translation, _edge["anchor"])
    except Exception as e:
        r = {"ok": False, "why": f"{type(e).__name__}: {e}"}
    r["t"] = time.time()
    if r["ok"]:
        hist = (_edge["hist"] + [r])[-EDGE_MEDIAN_N:]
        med = np.median([h_["K"] for h_ in hist], axis=0)  # ~1 in 6 frames picks a wrong LSD piece
        best = min(hist, key=lambda h_: float(np.linalg.norm(np.array(h_["K"]) - med)))
        # box centre = median over the last N (the per-frame floor height wanders +-2.5 cm)
        cs = [h_["center"] for h_ in hist if h_.get("center")]
        if cs:
            cm = np.median(cs, axis=0); cc = T.inverse().act(cm)
            best = {**best, "center_frame": best.get("center"), "center": cm.tolist(), "center_n": len(cs),
                    "px": {**best["px"], "center": [FX * 3 * cc[0] / cc[2] + CX * 3, FY * 2.5 * cc[1] / cc[2] + CY * 2.5]}}
        _edge["hist"], _edge["show"] = hist, best
        if np.linalg.norm(np.array(best["corner"]) - _edge["anchor"]) < 0.05:  # follow, but never jump
            _edge["anchor"] = np.array(best["corner"])
        try:
            with open(PLACE_JSON, "w") as f:
                json.dump({**best, "n": len(hist), "frame": "base"}, f, indent=1)
        except Exception:
            pass
    _edge["last"] = r
    _edge["busy"] = False


def edge_view(frame):
    """Live frame zoomed on the place target + the edges of the shown HD result."""
    h, w = frame.shape[:2]
    E, r = place_edges, _edge["show"]
    if r:  # zoom on the detected target (it moves in the image when the head / torso moves)
        us = [r["px"][k][i] / 3 for k in ("V2", "V3", "D", "K") for i in range(0, len(r["px"][k]), 2)]
        vs = [r["px"][k][i] / 2.5 for k in ("V2", "V3", "D", "K") for i in range(1, len(r["px"][k]), 2)]
        x0, y0, x1, y1 = fit_aspect(min(us) - 45, min(vs) - 35, max(us) + 45, max(vs) + 15, w, h)
    else:
        x0, y0, x1, y1 = fit_aspect(E.U0 / 3 - 50, E.V0 / 2.5 - 40, E.U1 / 3 + 50, E.V1 / 2.5 + 15, w, h)
    sx, sy = w / max(1, x1 - x0), h / max(1, y1 - y0)
    disp = cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_CUBIC)
    P = lambda u, v: (int((u / 3 - x0) * sx), int((v / 2.5 - y0) * sy))  # noqa: E731  HD px -> display

    def put(txt, at, col, sc=0.5):
        cv2.putText(disp, txt, at, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4)
        cv2.putText(disp, txt, at, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1 if sc < 0.6 else 2)

    if r:
        px = r["px"]; K = px["K"]
        for name, s, col in (("V2", px["V2"], (0, 200, 255)), ("H5", px["H5"], (255, 200, 0))):
            cv2.line(disp, P(s[0], s[1]), P(s[2], s[3]), col, 2); put(name, P(s[2] + 4, s[3]), col)
        cv2.line(disp, P(px["D"][0], px["D"][1]), P(*K), (0, 0, 255), 2)          # D, extended to K
        cv2.line(disp, P(*K), P(px["V3"][2], px["V3"][3]), (255, 0, 255), 2)       # V3, extended to K
        put("D", P((px["D"][0] + K[0]) / 2 - 18, (px["D"][1] + K[1]) / 2), (0, 0, 255))
        put("V3", P(px["V3"][2] + 4, px["V3"][3]), (255, 0, 255))
        cv2.circle(disp, P(*K), 7, (0, 255, 255), -1); cv2.circle(disp, P(*K), 7, (0, 0, 0), 1)
        if px.get("B1"):  # floor line + its corner with V3 (V3 drawn down to it)
            cv2.line(disp, P(*px["V3"][2:]), P(*px["Vb"]), (255, 0, 255), 2)
            cv2.line(disp, P(px["B1"][0], px["B1"][1]), P(px["B1"][2], px["B1"][3]), (0, 255, 128), 2); put("B1", P(px["B1"][2] + 4, px["B1"][3]), (0, 255, 128))
        if px.get("center"):
            cu, cv_ = P(*px["center"]); cv2.drawMarker(disp, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
            ce = r["center"]; put(f"CENTER({ce[0]:+.3f},{ce[1]:+.3f},{ce[2]:+.3f})", (cu - 95, cv_ + 26), (0, 0, 255), 0.55)
        k, c = r["K"], r["corner"]
        put(f"K({k[0]:+.3f},{k[1]:+.3f},{k[2]:+.3f})", (P(*K)[0] - 80, P(*K)[1] - 14), (0, 255, 255))
        put(f"C({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})", (P(px["D"][0], px["D"][1])[0] - 150, P(px["D"][0], px["D"][1])[1] + 22), (0, 255, 255))
        put(f"D {100 * r['D_len_m']:.1f}cm yaw {r['D_yaw_deg']:+.1f}  lean {r['lean_deg']:.1f}deg({r.get('lean_src', '')})  n{len(_edge['hist'])}"
            f"  {time.time() - r['t']:.1f}s", (10, 50), (255, 255, 255))
    if r and not r.get("center"):
        put(f"centre: {r.get('center_why', '-')}", (10, 98), (0, 160, 255))
    last = _edge["last"]
    if last is not None and not last.get("ok"):
        put(f"last snapshot: {last.get('why')}", (10, 74), (0, 160, 255))
    put("TARGETING: EDGE", (10, 24), (0, 255, 255), 0.6)
    put(_tmode.why, (10, h - 12), (255, 255, 255), 0.45)
    return disp


def main():
    print(f"[seg] model={MODEL} urdf frame={CAM_FRAME} -> udp {DEST}:{PORT}", flush=True)
    focus, cands, depth_img = None, [], None
    last_scan, n, t0 = 0.0, 0, time.time()
    global _held
    while True:
        frame = _decode(rgb_sub.recv())
        t_frame = time.time()
        h, w = frame.shape[:2]
        depth_img = _depth_latest

        # Targeting System: EDGE (box held + upright) replaces the whole SEG path below
        T_now = cam_to_base_T()
        with _feed_lock:
            hs, age = _hand["status"], time.time() - _hand["t"]
        mode = _tmode.update(hs, age, gripper_tilt_deg() if T_now is not None else None, time.time())
        if mode == "EDGE":
            if _rescan_want.is_set():  # Lotte Reach while a box is held: answer, don't time out
                _rescan["result"] = {"ok": False, "error": "Targeting System is in EDGE mode (box held upright)"}
                _rescan_want.clear()
                _rescan_done.set()
            if not _edge["busy"] and time.time() - _edge["t"] >= EDGE_S and T_now is not None:
                _edge["busy"], _edge["t"] = True, time.time()
                threading.Thread(target=_edge_job, args=(T_now,), daemon=True).start()
            disp = edge_view(frame)
            out = ensure_gst(w, h)
            try:
                out.stdin.write(disp.tobytes())
                out.stdin.flush()
            except BrokenPipeError:
                globals()["gst"] = None
            continue
        if _edge["hist"]:  # back to SEG: the next placement starts a fresh median
            _edge["hist"], _edge["show"], _edge["last"] = [], None, None

        forced = _rescan_want.is_set() and min(t_frame, _depth_t) > _rescan["t"]
        if forced:
            _held = None  # the view changed on purpose: measure afresh, don't hold the old box
        if forced or time.time() - last_scan >= SCAN_S:
            try:
                nf, nc = scan(frame, depth_img)
            except Exception:  # one bad frame must not take HEAD SEG down; log it and keep streaming
                traceback.print_exc()
                nf, nc = None, []
            if forced:
                _rescan["result"] = {"ok": True, "t": time.time(), "grav": _grav_src,
                                     "frame": "base" if nc and nc[0]["xyz_base"] else "camera",
                                     "boxes": nc if nf is not None else []}
                _rescan_want.clear()
                _rescan_done.set()
            if nf is not None:
                focus = nf if focus is None else tuple(
                    int(0.7 * o + 0.3 * n_) for o, n_ in zip(focus, nf))
                cands = nc  # topmost first (scan sorts; keeps a held box at 0)
                try:
                    with open(COORDS_JSON, "w") as f:
                        json.dump({"t": time.time(), "frame": "base" if cands and cands[0]["xyz_base"] else "camera",
                                   "boxes": cands}, f, indent=1)
                except Exception:
                    pass
            last_scan = time.time()

        if focus is not None:
            fx0, fy0, fx1, fy1 = focus
            crop = frame[fy0:fy1, fx0:fx1]
            view = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
            sx, sy = w / max(1, fx1 - fx0), h / max(1, fy1 - fy0)
        else:
            view, (fx0, fy0), (sx, sy) = frame, (0, 0), (1.0, 1.0)

        disp = view.copy()

        def to_disp(px):
            return int((px[0] - fx0) * sx), int((px[1] - fy0) * sy)

        # Candidate bboxes in display coords
        view_boxes = []
        for c in cands:
            x0, y0 = to_disp((c["bbox"][0], c["bbox"][1]))
            x1, y1 = to_disp((c["bbox"][2], c["bbox"][3]))
            view_boxes.append((x0, y0, x1, y1))

        # Segment the zoomed view, but paint ONLY masks that belong to a box
        # candidate (detection center inside a candidate bbox or vice versa).
        res = model(view, verbose=False, device=0, conf=CONF, retina_masks=True)[0]
        PALETTE = [(60, 180, 75), (255, 160, 0), (240, 80, 190), (0, 200, 255)]
        if res.masks is not None and view_boxes:
            for det_box, mask in zip(res.boxes.xyxy, res.masks.data):
                dx0, dy0, dx1, dy1 = [int(t) for t in det_box]
                det_area = max(1, (dx1 - dx0) * (dy1 - dy0))
                owner = None
                for i, (x0, y0, x1, y1) in enumerate(view_boxes):
                    # det must lie mostly INSIDE the candidate bbox — keeps
                    # wall/floor megamasks from being painted as a box.
                    iw = min(dx1, x1) - max(dx0, x0)
                    ih = min(dy1, y1) - max(dy0, y0)
                    if iw > 0 and ih > 0 and (iw * ih) / det_area >= 0.6:
                        owner = i
                        break
                if owner is None:
                    continue
                m = mask.cpu().numpy() > 0.5
                if m.shape != disp.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (w, h)) > 0
                color = np.array(PALETTE[owner % len(PALETTE)], dtype=np.float32)
                disp[m] = (0.55 * disp[m] + 0.45 * color).astype(np.uint8)

        for i, c in enumerate(cands):
            u, v = to_disp(c["c"])
            if not (0 <= u < w and 0 <= v < h):
                continue
            x0, y0, x1, y1 = view_boxes[i]
            cv2.rectangle(disp, (x0, y0), (x1, y1), PALETTE[i % len(PALETTE)], 2)
            tag = "B" if c["xyz_base"] else "C"  # Base or Camera frame
            g = c.get("grasp")
            p = g["xyz"] if g else (c["xyz_base"] or c["xyz_cam"])
            txt = f"box #{i+1} {tag}({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
            if c.get("size_m"):  # width x height of the front face
                txt += f" {c['size_m'][0] * 100:.1f}x{c['size_m'][1] * 100:.1f}cm"
            if i == 0:  # grasp target = top 20-sheet bundle of the topmost box
                gu, gv = to_disp(g["px"]) if g else (u, v)
                cv2.drawMarker(disp, (gu, gv), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
                cv2.circle(disp, (gu, gv), 12, (0, 0, 255), 2)
                txt = ("GRASP HELD " if c.get("held") else "GRASP ") + txt
            # Layer cut lines every PITCH (20 sheets) down both edges
            for L in c.get("layers", []):
                la, lb = to_disp(L["left"]["px"]), to_disp(L["right"]["px"])
                top = L["k"] == 1
                col = (0, 0, 255) if top else (255, 0, 255)
                cv2.line(disp, la, lb, col, 2 if top else 1)
                for pt in (la, lb):
                    cv2.circle(disp, pt, 5, col, -1)
                    cv2.circle(disp, pt, 5, (255, 255, 255), 1)
                mx, my, mz = L["mid"]
                ltxt = f"L{L['k']} {L['k'] * PITCH * 100:.1f}cm ({mx:+.3f},{my:+.3f},{mz:+.3f})"
                lx, ly = la[0] + 8, la[1] - 5
                cv2.putText(disp, ltxt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                cv2.putText(disp, ltxt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            ty = max(16, y0 - 8)
            cv2.putText(disp, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(disp, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255) if i else (0, 200, 255), 1)
            # Corner points with absolute coords
            for j, cr in enumerate(c.get("corners", [])):
                cu_, cv2_ = to_disp(cr["px"])
                if not (0 <= cu_ < w and 0 <= cv2_ < h):
                    continue
                cv2.circle(disp, (cu_, cv2_), 6, (255, 0, 255), -1)
                cv2.circle(disp, (cu_, cv2_), 6, (255, 255, 255), 1)
                cx_, cy_, cz_ = cr["xyz"]
                ctxt = f"P{j+1}({cx_:+.3f},{cy_:+.3f},{cz_:+.3f})"
                tx_ = min(max(2, cu_ - 60), w - 190)
                ty_ = cv2_ - 12 if cv2_ > h // 2 else cv2_ + 20
                cv2.putText(disp, ctxt, (tx_, ty_), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
                cv2.putText(disp, ctxt, (tx_, ty_), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 150, 255), 1)

        label = (f"TARGETING: SEG  boxes: {len(cands)}  x{sx:.1f}  grav:{_grav_src}" if focus is not None
                 else "TARGETING: SEG  NO BOXES - FULL VIEW")
        cv2.putText(disp, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(disp, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out = ensure_gst(w, h)
        try:
            out.stdin.write(disp.tobytes())
            out.stdin.flush()
        except BrokenPipeError:
            globals()["gst"] = None
            continue

        n += 1
        if time.time() - t0 >= 10:
            g = f" top={tuple(round(x,2) for x in (cands[0]['xyz_base'] or cands[0]['xyz_cam']))}" if cands else ""
            print(f"[seg] {n/(time.time()-t0):.1f} fps boxes={len(cands)} focus={focus}{g}", flush=True)
            n, t0 = 0, time.time()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        main()
    except KeyboardInterrupt:
        pass
