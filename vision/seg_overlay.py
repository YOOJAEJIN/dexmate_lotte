#!/usr/bin/env python3
"""Box-pile segmentation + base-frame coordinates + grasp overlay ("HEAD SEG").

  relay 60001 RGB tap (zmq 55601) ─┬─ every SCAN_S: FastSAM full frame
  nano metric depth (TCP 55703, mm) ┘   -> box candidates (area/border/depth)
  backend /robots/joints (poll)    ──> pinocchio FK -> T(base<-zed_left_camera)

  per scan : each box -> centroid px + depth -> camera XYZ -> BASE XYZ
             focus = extreme points of all boxes, padded to the frame aspect
             corners (hull diagonals), 6.1 cm layer lines down both edges,
             grasp = centre of the top 20-sheet bundle -> logs/box_coords.json
  per frame: bicubic zoom to focus -> FastSAM overlay -> markers/labels
             -> x264 RTP udp 50006 -> relay 60006 -> GUI tile "head seg"

Needs a python with torch(CUDA) + ultralytics + opencv + pyzmq + pinocchio
(`pip install pin`). See README.md for the full setup.
"""
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
import urllib.request

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
JOINTS_URL = os.getenv("SEG_JOINTS_URL", "http://127.0.0.1:5006/robots/joints")
_HERE = Path(__file__).resolve().parent
COORDS_JSON = os.getenv("SEG_COORDS_JSON", str(_HERE / "logs" / "box_coords.json"))
URDF = os.getenv(
    "SEG_URDF",
    str(_HERE.parent / "omniteleop/external/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf"),
)
# vega_1p.yaml reach.camera / camera_mount_rpy_corr
# ZED factory calibration SN50691325 [LEFT_CAM_FHD1200] (1920x1200), scaled to the
# 640x480 stream; gstsender resizes non-uniformly (x 1/3, y 0.4) so fx != fy.
FX, FY, CX, CY = 735.25 / 3, 735.141 * 0.4, 992.571 / 3, 618.88 * 0.4
DEPTH_VALID = (0.30, 1.60)
# Pure URDF FK on the real robot. reach's camera_mount_rpy_corr (-0.315, 0.0377,
# 0.0941) was tuned for the Isaac Lab render: on hardware it tilts vertical box
# edges ~20 deg; with zero they come out ~2 deg (best fit roll +0.025 rad).
MOUNT_RPY = (0.0, 0.0, 0.0)
# Cardboard-brown gate (HSV, measured on this robot's lighting 2026-11-10:
# real box H~16 S~130, grey crate S~67 -> saturation separates them).
BROWN_MIN_FRAC = float(os.getenv("SEG_BROWN_MIN", "0.4"))  # 0 disables
BROWN_H = (int(os.getenv("SEG_BROWN_H0", "5")), int(os.getenv("SEG_BROWN_H1", "30")))
BROWN_S_MIN = int(os.getenv("SEG_BROWN_S", "95"))
BROWN_V_MIN = int(os.getenv("SEG_BROWN_V", "50"))
# Layer pitch along the box's vertical edges: 20 cardboard sheets = 6.1 cm.
PITCH = float(os.getenv("SEG_PITCH", "0.061"))


def project(p_cam):
    """Camera-frame 3D point -> pixel (u, v)."""
    return [int(FX * p_cam[0] / p_cam[2] + CX), int(FY * p_cam[1] / p_cam[2] + CY)]


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

Path(COORDS_JSON).parent.mkdir(parents=True, exist_ok=True)
model = (FastSAM if "fastsam" in MODEL.lower() else YOLO)(MODEL)

pin_model = pin.buildModelFromUrdf(URDF)
pin_data = pin_model.createData()
CAM_FRAME = pin_model.getFrameId("zed_left_camera")
R_CORR = pin.rpy.rpyToMatrix(*MOUNT_RPY)
_JOINT_GROUPS = {
    "left_arm": [f"L_arm_j{i+1}" for i in range(7)],
    "right_arm": [f"R_arm_j{i+1}" for i in range(7)],
    "torso": [f"torso_j{i+1}" for i in range(3)],
    "head": [f"head_j{i+1}" for i in range(3)],
}

_joints_cache, _joints_t = None, 0.0


def cam_to_base_T():
    """T(base <- optical camera) from live joints, or None if unavailable."""
    global _joints_cache, _joints_t
    if time.time() - _joints_t > 0.5:
        try:
            with urllib.request.urlopen(JOINTS_URL, timeout=1.5) as r:
                _joints_cache = json.load(r)
        except Exception:
            _joints_cache = _joints_cache  # keep stale joints if poll fails
        _joints_t = time.time()
    if not _joints_cache:
        return None
    q = pin.neutral(pin_model)
    for group, names in _JOINT_GROUPS.items():
        vals = _joints_cache.get(group)
        if not vals:
            continue
        for jname, val in zip(names, vals):
            jid = pin_model.getJointId(jname)
            q[pin_model.idx_qs[jid]] = float(val)
    q = np.clip(q, pin_model.lowerPositionLimit, pin_model.upperPositionLimit)
    pin.framesForwardKinematics(pin_model, pin_data, q)
    return pin_data.oMf[CAM_FRAME] * pin.SE3(R_CORR, np.zeros(3))


ctx = zmq.Context()


def _sub(addr):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.connect(addr)
    return s


rgb_sub = _sub(RGB_TAP)
_depth_latest = None


def _depth_reader():
    """Keep _depth_latest = newest metric depth frame; reconnect forever."""
    global _depth_latest
    host, port = DEPTH_SRC.rsplit(":", 1)
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=5) as s:
                f = s.makefile("rb")
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        break
                    w, h, n = struct.unpack(">HHI", hdr)
                    buf = f.read(n)
                    if len(buf) < n:
                        break
                    _depth_latest = np.frombuffer(buf, dtype="<u2").reshape(h, w)
        except OSError:
            pass
        _depth_latest = None  # stale depth is worse than none
        time.sleep(1)


threading.Thread(target=_depth_reader, daemon=True).start()


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


def scan(frame, depth_img):
    """FastSAM full frame -> box candidates with camera+base XYZ, focus bbox."""
    h, w = frame.shape[:2]
    res = model(frame, verbose=False, device=0, conf=CONF, retina_masks=True)[0]
    if res.boxes is None or res.masks is None:
        return None, []
    T = cam_to_base_T()
    cands = []
    for box, mask in zip(res.boxes, res.masks.data):
        x0, y0, x1, y1 = [int(v) for v in box.xyxy[0]]
        bw, bh = x1 - x0, y1 - y0
        area_frac = (bw * bh) / (w * h)
        if not (0.005 <= area_frac <= 0.35):
            continue
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
        xyz_cam = np.array([(cu - CX) * z / FX, (cv_ - CY) * z / FY, z])
        xyz_base = T.act(xyz_cam).tolist() if T is not None else None

        # Box face corners from the mask's min-area rectangle. Depth is
        # sampled slightly inside each corner (edges have mixed depth).
        corners = []
        m = mask.cpu().numpy().astype(np.uint8)
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m, (w, h))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2)
            # Extreme hull points along the 4 diagonals (BR, BL, TL, TR). Unlike
            # minAreaRect this keeps the perspective trapezoid: with the head
            # looking down, the farther bottom edge is shorter in pixels.
            for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                px, py = hull[np.argmax(hull[:, 0] * dx + hull[:, 1] * dy)].astype(float)
                su, sv = int(0.85 * px + 0.15 * cu), int(0.85 * py + 0.15 * cv_)
                cz = depth_at(depth_img, su, sv) or z
                c_cam = np.array([(px - CX) * cz / FX, (py - CY) * cz / FY, cz])
                c_abs = T.act(c_cam).tolist() if T is not None else c_cam.tolist()
                corners.append({"px": [int(px), int(py)], "xyz": c_abs,
                                "cam": c_cam.tolist()})
        layers, grasp = layer_lines(corners, T)

        cands.append({"bbox": (x0, y0, x1, y1), "c": (cu, cv_),
                      "xyz_cam": xyz_cam.tolist(), "xyz_base": xyz_base,
                      "corners": corners, "layers": layers, "grasp": grasp})
    if not cands:
        return None, []
    # Keep only the box PILE: grow a cluster from the largest candidate and
    # absorb neighbours whose bbox touches the (slightly expanded) union.
    # Desk clutter / monitors far from the pile drop out entirely.
    def _area(c):
        x0, y0, x1, y1 = c["bbox"]
        return (x1 - x0) * (y1 - y0)

    rest = sorted(cands, key=_area, reverse=True)
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
    cands = cluster
    # Focus = extreme points of the pile, fitted to frame aspect
    xs0 = min(c["bbox"][0] for c in cands); ys0 = min(c["bbox"][1] for c in cands)
    xs1 = max(c["bbox"][2] for c in cands); ys1 = max(c["bbox"][3] for c in cands)
    focus = fit_aspect(xs0, ys0, xs1, ys1, w, h, margin=0.04)
    return focus, cands


def main():
    print(f"[seg] model={MODEL} urdf frame={CAM_FRAME} -> udp {DEST}:{PORT}", flush=True)
    focus, cands, depth_img = None, [], None
    last_scan, n, t0 = 0.0, 0, time.time()
    while True:
        frame = _decode(rgb_sub.recv())
        h, w = frame.shape[:2]
        depth_img = _depth_latest

        if time.time() - last_scan >= SCAN_S:
            nf, nc = scan(frame, depth_img)
            if nf is not None:
                focus = nf if focus is None else tuple(
                    int(0.7 * o + 0.3 * n_) for o, n_ in zip(focus, nf))
                cands = sorted(nc, key=lambda c: c["c"][1])  # topmost first
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
            if i == 0:  # grasp target = top 20-sheet bundle of the topmost box
                gu, gv = to_disp(g["px"]) if g else (u, v)
                cv2.drawMarker(disp, (gu, gv), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
                cv2.circle(disp, (gu, gv), 12, (0, 0, 255), 2)
                txt = "GRASP " + txt
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

        label = f"boxes: {len(cands)}  x{sx:.1f}" if focus is not None else "NO BOXES - FULL VIEW"
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
