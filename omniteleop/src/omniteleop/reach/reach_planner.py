#!/usr/bin/env python3
"""Reach planner helper — spawned by the omniteleop backend.

Runs under the pantos_tele virtualenv (torch / transformers / pin / pink) and
reuses pantos_tele's proven Grounding-DINO detection and Pink IK:

    stdin  : JSON request  (image paths on disk, live joint values, settings)
    stdout : JSON response (detected object xyz in the robot base frame and
             arm joint targets for the reach motion)

The planner never touches the robot: joint positions are supplied by the
caller (omniteleop backend reads them from the ``robot/joints`` topic) and the
resulting joint targets are executed through omniteleop's regular command
pipeline (common/input -> command_processor -> robot_controller).

Request schema
--------------
{
  "rgb_path":   str,                    # BGR image file (head_left_rgb frame)
  "depth_path": str,                    # uint16 PNG, depth in millimetres
  "joints": {
    "left_arm":  [7 floats], "right_arm": [7 floats],
    "torso": [3 floats],     "head":      [3 floats]
  },
  "urdf_path": str,                     # vega_1p URDF (needs arm_center,
                                        # L/R_ee, zed_left_camera)
  "intrinsics": {"fx": f, "fy": f, "cx": f, "cy": f},
  "height_above_m": float,              # approach height above the object
  "detect_only": bool,                  # skip IK (detection verification mode)
  "camera_mount_rpy_corr": [r, p, y],   # extra rotation applied to the URDF
                                        # zed_left_camera frame (projection
                                        # calibration, radians)
  "max_bbox_area": int,                 # reject detections larger than this
  "debug_dir": str                      # where to write annotated PNGs
}
"""

import json
import os
import sys
import time
from types import SimpleNamespace

# ── Stage timing (per-invocation profiling) ──────────────────────────────────
# _T0 starts right after the cheap stdlib imports, so "imports" below covers
# the heavy deps (torch/transformers/pin/pink/cv2/pantos). Timings are printed
# to stderr AND embedded in the JSON response under "timings" so the backend
# can surface them in its logs.
_T0 = time.perf_counter()
_T_LAST = _T0
_STAGE_TIMES: dict = {}


def _stage(name: str) -> None:
    """Log and record the time spent since the previous stage marker."""
    global _T_LAST
    now = time.perf_counter()
    dt = now - _T_LAST
    _T_LAST = now
    _STAGE_TIMES[name] = round(dt, 3)
    print(f"[reach_planner] ⏱ {name}: {dt:.2f}s", file=sys.stderr, flush=True)


def _timings_payload() -> dict:
    return {
        "stages": dict(_STAGE_TIMES),
        "total": round(time.perf_counter() - _T0, 3),
    }

# ── pantos_tele import path (must precede pantos imports) ────────────────────
_PANTOS_DIR = os.environ.get("PANTOS_TELE_DIR", "/home/ubuntu/dx/pantos_tele")
if _PANTOS_DIR not in sys.path:
    sys.path.insert(0, _PANTOS_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
import pink  # noqa: E402
import qpsolvers  # noqa: E402
from loguru import logger  # noqa: E402
from pink.tasks import RelativeFrameTask, PostureTask  # noqa: E402

from pantos_tele import config as pconfig  # noqa: E402
from pantos_tele import perception  # noqa: E402
from pantos_tele import ik_utils  # noqa: E402

# stdout carries the JSON request/response protocol — force every log (some
# imported package adds a stdout sink) onto stderr so responses stay clean.
logger.remove()
logger.add(sys.stderr, level="INFO")

_stage("imports (torch/pin/pink/pantos)")

# ── URDF joint groups (indices resolved against the pinocchio model) ─────────
_JOINT_GROUPS = {
    "left_arm": [f"L_arm_j{i + 1}" for i in range(7)],
    "right_arm": [f"R_arm_j{i + 1}" for i in range(7)],
    "torso": [f"torso_j{i + 1}" for i in range(3)],
    "head": [f"head_j{i + 1}" for i in range(3)],
}


def build_q(model: pin.Model, joints: dict) -> np.ndarray:
    """Neutral configuration with arms/torso/head set from the request."""
    q = pin.neutral(model)
    for group, names in _JOINT_GROUPS.items():
        values = joints.get(group)
        if values is None:
            continue
        for jname, val in zip(names, values):
            jid = model.getJointId(jname)
            q[model.idx_qs[jid]] = float(val)
    return np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)


def load_images(rgb_path: str, depth_path: str):
    """Load the RGB frame and the metric depth frame (metres, float32)."""
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise RuntimeError(f"cannot read depth image: {depth_path}")
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return rgb, depth_m


def camera_point_to_base(model, data, q, point_cam, mount_rpy_corr, mount_pos_corr=(0.0, 0.0, 0.0)):
    """Transform an optical-frame camera point to the robot base frame.

    The URDF ``zed_left_camera`` frame follows the ROS optical convention
    (x right, y down, z forward) — the same convention produced by
    ``perception.pixel_to_camera`` — so the transform is plain FK to that
    frame, optionally corrected by ``mount_rpy_corr`` / ``mount_pos_corr``
    (the sim camera's fixed rotation/position offset w.r.t. the URDF frame,
    e.g. the 10 cm forward prim offset in camera_configs.py).
    """
    pin.framesForwardKinematics(model, data, q)
    zed_id = model.getFrameId("zed_left_camera")
    T_base_zed = data.oMf[zed_id]

    R_corr = pin.rpy.rpyToMatrix(*mount_rpy_corr)
    T_base_cam = T_base_zed * pin.SE3(R_corr, np.asarray(mount_pos_corr, dtype=float))
    return T_base_cam.act(point_cam)


def save_debug_image(debug_dir, rgb, bbox, score, label, pixel_uv, xyz_base, all_bboxes):
    """Write an annotated detection PNG (ROI, boxes, target, world xyz)."""
    try:
        from PIL import Image, ImageDraw

        os.makedirs(debug_dir, exist_ok=True)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        img_w, img_h = pil_img.size

        cx, cy = img_w / 2.0, img_h / 2.0
        half_u = (img_w * pconfig.CENTER_ROI_FRAC_U) / 2.0
        half_v = (img_h * pconfig.CENTER_ROI_FRAC_V) / 2.0
        draw.rectangle([cx - half_u, cy - half_v, cx + half_u, cy + half_v],
                       outline=(255, 255, 0), width=2)

        for ab in (all_bboxes or []):
            draw.rectangle(ab, outline=(160, 160, 160), width=1)

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(y1 - 18, 0)), f"{label} {score:.2f}", fill=(0, 255, 0))

        if pixel_uv is not None:
            u, v = pixel_uv
            r = 12
            draw.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=3)
            draw.line([u - r * 2, v, u + r * 2, v], fill=(255, 0, 0), width=2)
            draw.line([u, v - r * 2, u, v + r * 2], fill=(255, 0, 0), width=2)

        if xyz_base is not None:
            wx, wy, wz = xyz_base
            draw.text((10, img_h - 25),
                      f"World: x={wx:+.3f}  y={wy:+.3f}  z={wz:+.3f} m",
                      fill=(255, 80, 80))

        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(debug_dir, f"reach_detection_{ts}.png")
        pil_img.save(path)
        return path
    except Exception as e:  # pragma: no cover - debug only
        logger.warning(f"failed to save debug image: {e}")
        return None


def solve_reach_ik(model, data, q, joints, box_base_xyz, height_above_m):
    """Port of pantos_tele main._reach(): IK toward `height_above_m` above the box.

    Returns (use_left_arm, target_left_arm, target_right_arm, motion_duration).
    The moving arm keeps its current EE orientation; the other arm holds its
    current pose. Targets are expressed in the arm_center frame so torso
    joints stay outside the IK Jacobian.
    """
    # Read the current arm_center / EE poses before building the pink
    # configuration (pink recomputes `data` on every update).
    pin.framesForwardKinematics(model, data, q)
    T_wac = data.oMf[model.getFrameId("arm_center")]
    left_ee_task = RelativeFrameTask(
        "L_ee", root="arm_center",
        position_cost=pconfig.LEFT_EE_POSITION_COST,
        orientation_cost=pconfig.LEFT_EE_ORIENTATION_COST,
    )
    right_ee_task = RelativeFrameTask(
        "R_ee", root="arm_center",
        position_cost=pconfig.RIGHT_EE_POSITION_COST,
        orientation_cost=pconfig.RIGHT_EE_ORIENTATION_COST,
    )
    posture_task = PostureTask(cost=pconfig.POSTURE_COST)
    posture_task.set_target(q)

    solver = qpsolvers.available_solvers[0] if qpsolvers.available_solvers else None
    if pconfig.PREFERRED_QP_SOLVER in qpsolvers.available_solvers:
        solver = pconfig.PREFERRED_QP_SOLVER
    if solver is None:
        raise RuntimeError(
            "no QP solver available in the pantos venv — Pink IK needs one "
            "(e.g. `uv add daqp` in /home/ubuntu/dx/pantos_tele)"
        )

    logger.info(
        f"arm_center: pos=({T_wac.translation[0]:.3f},"
        f"{T_wac.translation[1]:.3f},{T_wac.translation[2]:.3f})"
    )

    box_world = np.array([box_base_xyz[0] + 0.03,
                          box_base_xyz[1],
                          box_base_xyz[2] + height_above_m])
    box_ac = T_wac.inverse().act(box_world)
    logger.info(f"Box in arm_center: x={box_ac[0]:.3f}, y={box_ac[1]:.3f}, z={box_ac[2]:.3f}")

    T_ac_L = T_wac.inverse() * data.oMf[model.getFrameId("L_ee")]
    left_pos, left_rpy = T_ac_L.translation.copy(), pin.rpy.matrixToRpy(T_ac_L.rotation)
    T_ac_R = T_wac.inverse() * data.oMf[model.getFrameId("R_ee")]
    right_pos_cur, right_rpy_cur = T_ac_R.translation.copy(), pin.rpy.matrixToRpy(T_ac_R.rotation)

    use_left_arm = box_ac[1] > 0
    logger.info(f"Using {'LEFT' if use_left_arm else 'RIGHT'} arm (box y={box_ac[1]:+.3f})")
    move_rpy = left_rpy if use_left_arm else right_rpy_cur

    if use_left_arm:
        l_pos, l_rpy = box_ac, move_rpy
        r_pos, r_rpy = right_pos_cur, right_rpy_cur
        move_distance = np.linalg.norm(l_pos - left_pos)
    else:
        l_pos, l_rpy = left_pos, left_rpy
        r_pos, r_rpy = box_ac, move_rpy
        move_distance = np.linalg.norm(r_pos - right_pos_cur)

    motion_duration = float(np.clip(
        move_distance / pconfig.EE_SPEED_M_S, pconfig.MIN_MOTION_S, pconfig.MAX_MOTION_S
    ))
    logger.info(f"Motion duration: {motion_duration:.1f}s (dist={move_distance:.3f}m)")

    # ik_utils.sync_ik_from_live / solve_and_extract expect a context object;
    # provide one seeded with the caller-supplied joint values (no robot).
    configuration = pink.Configuration(model, data, q)
    ctx = SimpleNamespace(
        bot=_BotShim(joints),
        robot_pin=SimpleNamespace(model=model, data=data),
        configuration=configuration,
        left_ee_task=left_ee_task,
        right_ee_task=right_ee_task,
        posture_task=posture_task,
        solver=solver,
        model=model,
        left_arm_indices=[model.getJointId(n) for n in _JOINT_GROUPS["left_arm"]],
        right_arm_indices=[model.getJointId(n) for n in _JOINT_GROUPS["right_arm"]],
        torso_indices=[model.getJointId(n) for n in _JOINT_GROUPS["torso"]],
        control_dt=pconfig.CONTROL_DT,
        stop_event=None,
    )
    _ = ik_utils.sync_ik_from_live(ctx)  # re-sync configuration from shim joints

    waypoint = ik_utils.make_ik_waypoint(l_pos, l_rpy, r_pos, r_rpy, motion_duration)
    tgt_L, tgt_R = ik_utils.solve_and_extract(ctx, waypoint)

    # Target joints + how far they are from the current pose (per arm).
    cur = {"left_arm": np.array(joints["left_arm"]), "right_arm": np.array(joints["right_arm"])}
    for name, tgt in (("left", np.asarray(tgt_L)), ("right", np.asarray(tgt_R))):
        max_d = float(np.max(np.abs(tgt - cur[f"{name}_arm"])))
        logger.info(f"IK target {name}_arm: {[round(v, 3) for v in tgt.tolist()]} "
                    f"(max |Δ| vs current: {max_d:.3f} rad)")

    return use_left_arm, tgt_L.tolist(), tgt_R.tolist(), motion_duration


class _BotShim:
    """Minimal bot-like object for ik_utils.sync_ik_from_live (no hardware)."""

    def __init__(self, joints):
        self._joints = {k: np.asarray(v, dtype=float) for k, v in joints.items()}
        self.left_arm = SimpleNamespace(get_joint_pos=lambda: self._joints["left_arm"])
        self.right_arm = SimpleNamespace(get_joint_pos=lambda: self._joints["right_arm"])
        self.torso = SimpleNamespace(get_joint_pos=lambda: self._joints["torso"])
        self.head = SimpleNamespace(get_joint_pos=lambda: self._joints["head"])


# ─────────────────────────────────────────────────────────────────────────────
# Entry points — one-shot (stdin JSON → stdout JSON, single request) and
# daemon mode (--serve: line-based JSON request/response loop with the URDF
# model and detection model loaded once and cached across requests)
# ─────────────────────────────────────────────────────────────────────────────

_ROBOT_CACHE: dict = {}  # urdf_path -> pin.RobotWrapper (reused across requests)


def _get_robot(urdf_path: str):
    """Load (and cache) the pinocchio model for a URDF path."""
    robot_pin = _ROBOT_CACHE.get(urdf_path)
    if robot_pin is None:
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        package_dir = os.path.abspath(os.path.join(urdf_dir, "..", "..", "..", ".."))
        robot_pin = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[urdf_dir, package_dir],
            root_joint=None,
        )
        _ROBOT_CACHE[urdf_path] = robot_pin
    return robot_pin


def plan(request: dict) -> dict:
    """Process one reach request and return the response dict."""
    global _T_LAST, _STAGE_TIMES, _T0
    _T0 = _T_LAST = time.perf_counter()
    _STAGE_TIMES = {}
    _stage("parse request")

    # Warmup: preload the URDF model and detection model without planning.
    if request.get("warmup"):
        _get_robot(request["urdf_path"])
        _stage("urdf load")
        perception.initialize_detection_model()
        _stage("dino model load")
        return {"ok": True, "warmup": True, "timings": _timings_payload()}

    joints = request["joints"]

    intrinsics = request.get("intrinsics", {})
    # The sim camera differs from the real ZED X Mini — override the pantos
    # module globals used by perception.pixel_to_camera().
    pconfig.ZED_FX = float(intrinsics.get("fx", pconfig.ZED_FX))
    pconfig.ZED_FY = float(intrinsics.get("fy", pconfig.ZED_FY))
    pconfig.ZED_CX = float(intrinsics.get("cx", pconfig.ZED_CX))
    pconfig.ZED_CY = float(intrinsics.get("cy", pconfig.ZED_CY))

    robot_pin = _get_robot(request["urdf_path"])
    _stage("urdf load")
    model, data = robot_pin.model, robot_pin.data
    q = build_q(model, joints)

    rgb, depth_m = load_images(request["rgb_path"], request["depth_path"])
    _stage("load images")

    perception.initialize_detection_model()
    _stage("dino model load")
    result = perception._detect_in_center_roi(
        rgb, depth_m,
        max_bbox_area=int(request.get("max_bbox_area", 40000)),
        allow_fallback=False,
    )
    _stage("dino inference")
    debug_dir = request.get("debug_dir") or os.path.dirname(os.path.abspath(__file__))
    response = {"ok": True, "found": False}

    if result is None:
        logger.info("No valid object detected in the center ROI")
        response["message"] = "no object detected in the center ROI"
        response["timings"] = _timings_payload()
        save_debug_image(debug_dir, rgb, [0, 0, 1, 1], 0.0, "", None, None, [])
        return response

    det, all_bboxes = result
    (u, v), depth, bbox, score, label = det

    # Optional re-sampling: for small/thin objects the H264-compressed depth
    # stream smears the object into the background, so the median patch lands
    # on the surface behind it. Sampling the minimum valid depth inside the
    # bbox recovers the nearest surface (the object itself).
    if request.get("depth_sample") == "min":
        min_m = float(request.get("depth_min_valid_m", 0.30))
        max_m = float(request.get("depth_max_valid_m", 3.0))
        x1, y1, x2, y2 = bbox
        region = depth_m[max(0, y1):y2, max(0, x1):x2]
        valid = region[(region >= min_m) & (region <= max_m) & np.isfinite(region)]
        if valid.size:
            depth = float(valid.min())
            logger.info(f"depth re-sampled (min in bbox): {depth:.3f} m "
                        f"(patch median was {det[1]:.3f} m)")
        else:
            logger.warning("min-depth sampling found no valid values; keeping patch median")
    _stage("depth sampling")

    point_cam = perception.pixel_to_camera(int(u), int(v), float(depth))
    mount_corr = request.get("camera_mount_rpy_corr", [0.0, 0.0, 0.0])
    mount_pos = request.get("camera_mount_pos_corr", [0.0, 0.0, 0.0])
    point_base = camera_point_to_base(model, data, q, point_cam, mount_corr, mount_pos)
    _stage("camera transform")
    logger.info(
        f"Detected '{label}' (conf={score:.2f}) bbox={bbox} "
        f"cam=({point_cam[0]:.3f},{point_cam[1]:.3f},{point_cam[2]:.3f}) "
        f"base=({point_base[0]:.3f},{point_base[1]:.3f},{point_base[2]:.3f})"
    )
    xyz = [float(point_base[0]), float(point_base[1]), float(point_base[2])]

    response.update({
        "found": True,
        "label": str(label),
        "confidence": float(score),
        "bbox": [int(c) for c in bbox],
        "pixel_uv": [int(u), int(v)],
        "depth_m": float(depth),
        "xyz_base": xyz,
    })
    response["debug_image"] = save_debug_image(
        debug_dir, rgb, bbox, score, label, (u, v), xyz, all_bboxes
    )

    if not request.get("detect_only", False):
        use_left, tgt_L, tgt_R, duration = solve_reach_ik(
            model, data, q, joints, xyz, float(request.get("height_above_m", 0.40))
        )
        _stage("ik solve")
        response.update({
            "arm": "left" if use_left else "right",
            "target_left_arm": tgt_L,
            "target_right_arm": tgt_R,
            "motion_duration_s": duration,
        })

    response["timings"] = _timings_payload()
    logger.info(f"⏱ planner total {response['timings']['total']:.2f}s "
                f"| {response['timings']['stages']}")
    return response


def main() -> int:
    """One-shot mode: single JSON request on stdin, single JSON on stdout."""
    response = plan(json.load(sys.stdin))
    print(json.dumps(response))
    return 0


def serve() -> int:
    """Daemon mode: line-delimited JSON requests on stdin, one response per line.

    The URDF model and Grounding DINO weights stay loaded across requests.
    Exits cleanly on stdin EOF (i.e. when the parent backend goes away).
    """
    logger.info("reach planner daemon ready — waiting for requests")
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF: parent closed the pipe
            logger.info("stdin closed — planner daemon exiting")
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            response = plan(json.loads(line))
        except Exception as e:
            logger.exception("reach planner request failed")
            response = {"ok": False, "error": str(e), "timings": _timings_payload()}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        sys.exit(serve())
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("reach planner failed")
        print(json.dumps({"ok": False, "error": str(e), "timings": _timings_payload()}))
        sys.exit(0)
