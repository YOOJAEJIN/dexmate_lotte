# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Detect the conveyor AprilTag with the head camera and print its pose in the robot base frame.

Pipeline (same structure as the real-robot "투입부 좌표계 확정" step):

    left RGB (dexcontrol head_camera)
      -> cv2.aruco tag36h11 detection -> solvePnP (IPPE_SQUARE) -> T_cam_tag
      -> T_base_cam from URDF FK of ``zed_left_camera`` (torso + head joints)
         times the sim camera pitch offset
      -> T_base_tag = T_base_cam @ T_cam_tag

Frames: tag frame is OpenCV aruco (+X image right, +Y image up, +Z toward camera),
identical to the quad in ``assets/objects/apriltag/*.usda``. Camera frame is the
optical frame (+Z forward, +X right, +Y down).

Intrinsics default to the sim camera config (focal 7.6 mm, aperture 20 mm) scaled
to the received image width; override with ``--fx/--fy/--cx/--cy`` for a real ZED.

If the sim is running, ``unitree_sim_isaaclab/apriltag_pose.txt`` (written by
sim_main.py) is used as ground truth and the error is printed.

Example::

    python detect_apriltag.py                 # one shot
    python detect_apriltag.py --loop --hz 2   # continuous
    python detect_apriltag.py --save /tmp/tag.png
    python detect_apriltag.py --cam-pitch-offset-deg 0   # real robot (no sim offset)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402  (URDF_PATH, CAMERA_TIMEOUT)
from _pink_common import connect_camera_robot, disconnect_leave_held  # noqa: E402

CAMERA_LINK = "zed_left_camera"
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]
HEAD_JOINTS = [f"head_j{i}" for i in range(1, 4)]

# Sim camera (tasks/common_config/camera_configs.py get_camera_config defaults).
SIM_FOCAL_MM = 7.6
SIM_APERTURE_MM = 20.0
# CameraCfg.OffsetCfg(rot=(-0.98481, 0.17365, 0, 0), convention="ros") on the
# zed_left_camera prim == rotation of -20 deg about the optical X axis.
SIM_CAM_PITCH_OFFSET_DEG = -20.0

_DEX_ROOT = Path(__file__).resolve().parents[2]


def _default_gt_file() -> Path:
    """``$LOTTE_SIM_DIR/apriltag_pose.txt`` if set, else the newest ``apriltag_pose.txt`` among the
    ``unitree_sim_isaaclab*`` folders (the running sim rewrites it every step)."""
    env = os.environ.get("LOTTE_SIM_DIR")
    if env:
        return Path(env).expanduser() / "apriltag_pose.txt"
    found = [f for f in _DEX_ROOT.glob("unitree_sim_isaaclab*/apriltag_pose.txt") if f.is_file()]
    return max(found, key=lambda f: f.stat().st_mtime) if found else _DEX_ROOT / "unitree_sim_isaaclab_260918" / "apriltag_pose.txt"


DEFAULT_GT_FILE = _default_gt_file()


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
def sim_intrinsics(width: int, height: int) -> np.ndarray:
    """Pinhole K for the sim camera at the given image size (no distortion)."""
    fx = width * SIM_FOCAL_MM / SIM_APERTURE_MM
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


class HeadCameraFK:
    """T_base_cam (optical frame) from torso + head joints via the URDF."""

    def __init__(self, urdf_path: str = config.URDF_PATH, pitch_offset_deg: float = SIM_CAM_PITCH_OFFSET_DEG):
        robot = pin.RobotWrapper.BuildFromURDF(filename=urdf_path, package_dirs=[os.path.dirname(urdf_path)], root_joint=None)
        self.model, self.data = robot.model, robot.data
        self.cam_frame = self.model.getFrameId(CAMERA_LINK)
        self.idx = {n: self.model.idx_qs[self.model.getJointId(n)] for n in TORSO_JOINTS + HEAD_JOINTS}
        self.T_link_opt = np.eye(4)
        self.T_link_opt[:3, :3] = R.from_euler("x", pitch_offset_deg, degrees=True).as_matrix()

    def T_base_cam(self, q_torso, q_head) -> np.ndarray:
        q = pin.neutral(self.model)
        for n, v in zip(TORSO_JOINTS, np.asarray(q_torso, float)):
            q[self.idx[n]] = v
        for n, v in zip(HEAD_JOINTS, np.asarray(q_head, float)):
            q[self.idx[n]] = v
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.cam_frame].homogeneous @ self.T_link_opt


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def make_detector() -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_tag(gray: np.ndarray, K: np.ndarray, tag_size: float, tag_id: int | None, detector) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (T_cam_tag 4x4, corners 4x2, id) for the requested tag, or None."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.flatten()
    pick = [i for i, v in enumerate(ids) if tag_id is None or v == tag_id]
    if not pick:
        return None
    c = corners[pick[0]].reshape(4, 2).astype(np.float64)  # TL, TR, BR, BL
    h = tag_size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.ravel()
    return T, c, int(ids[pick[0]])


def fmt_pose(T: np.ndarray) -> str:
    q = R.from_matrix(T[:3, :3]).as_quat()  # x y z w
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return (f"pos {np.round(T[:3, 3], 4)} m  quat(wxyz) {np.round([q[3], q[0], q[1], q[2]], 4)}  "
            f"rpy {np.round(rpy, 1)} deg")


def load_gt(path: Path) -> np.ndarray | None:
    try:
        vals = [float(v) for v in path.read_text().splitlines()[1].split()]
    except Exception:
        return None
    T = np.eye(4)
    T[:3, 3] = vals[:3]
    T[:3, :3] = R.from_quat([vals[4], vals[5], vals[6], vals[3]]).as_matrix()  # wxyz -> xyzw
    return T


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    dp = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    dR = R.from_matrix(T_est[:3, :3].T @ T_gt[:3, :3]).magnitude()
    return dp, float(np.rad2deg(dR))


def split_stereo_if_needed(rgb: np.ndarray) -> np.ndarray:
    """Side-by-side stereo frames (w/h > 2) -> keep the left half."""
    h, w = rgb.shape[:2]
    if w > 2 * h:
        logger.warning(f"Frame {w}x{h} looks side-by-side stereo; using left half.")
        return rgb[:, : w // 2]
    return rgb


# ---------------------------------------------------------------------------
def main(
    tag_id: int = 0,
    tag_size: float = 0.24,
    loop: bool = False,
    hz: float = 2.0,
    save: str | None = None,
    out: str | None = None,
    gt_file: Path | None = DEFAULT_GT_FILE,
    cam_pitch_offset_deg: float = SIM_CAM_PITCH_OFFSET_DEG,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    camera_timeout: float = 5.0,
    head_pitch: float | None = None,
) -> None:
    """Detect the conveyor AprilTag and report its pose in the robot base frame.

    Args:
        tag_id: tag36h11 id to look for.
        tag_size: Black-square side in metres (must match the spawned asset).
        loop: Keep detecting at ``hz`` until Ctrl-C.
        hz: Loop rate.
        save: Save the annotated camera image to this path (every iteration).
        out: Write the latest base-frame pose to this text file (same format as
            apriltag_pose.txt) for other scripts to consume.
        gt_file: Sim ground-truth file; error is printed when it exists.
        cam_pitch_offset_deg: Rotation about the optical X axis between the URDF
            zed_left_camera frame and the actual optical frame. -20 for the sim
            camera config; use 0 on the real robot.
        fx, fy, cx, cy: Intrinsics override (defaults: sim camera model).
        camera_timeout: Seconds to wait for the head camera stream.
        head_pitch: If given, command head_j1 (rad) before detecting and keep
            yaw/roll. With the torso in the box-hold pose the tag is in view for
            pitch >= about -0.2 (the hold pose uses -0.64, which looks too far down).
    """
    fk = HeadCameraFK(pitch_offset_deg=cam_pitch_offset_deg)
    detector = make_detector()
    bot = connect_camera_robot()
    try:
        cam = bot.sensors.head_camera
        if not cam.wait_for_active(timeout=camera_timeout):
            logger.error("Head camera stream not active.")
            return
        if head_pitch is not None:
            cur = np.asarray(bot.head.get_joint_pos(), float)
            target = np.array([head_pitch, cur[1], cur[2]])
            logger.info(f"Head pitch {cur[0]:+.2f} → {head_pitch:+.2f} rad")
            bot.head.set_joint_pos(target, wait_time=3.0, exit_on_reach=True)
            time.sleep(0.5)  # let the camera stream catch up
        period = 1.0 / max(hz, 0.1)
        while True:
            t0 = time.time()
            rgb = cam.get_obs(obs_keys=["left_rgb"]).get("left_rgb")
            if rgb is None:
                logger.warning("No left_rgb frame yet.")
            else:
                rgb = split_stereo_if_needed(rgb)
                h, w = rgb.shape[:2]
                K = sim_intrinsics(w, h)
                if fx is not None: K[0, 0] = fx
                if fy is not None: K[1, 1] = fy
                if cx is not None: K[0, 2] = cx
                if cy is not None: K[1, 2] = cy
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                q_torso = np.asarray(bot.torso.get_joint_pos(), float)
                q_head = np.asarray(bot.head.get_joint_pos(), float)
                T_base_cam = fk.T_base_cam(q_torso, q_head)
                det = detect_tag(gray, K, tag_size, tag_id, detector)
                if det is None:
                    logger.warning(f"Tag {tag_id} not found in {w}x{h} frame (fx={K[0,0]:.1f}).")
                else:
                    T_cam_tag, corners, found_id = det
                    T_base_tag = T_base_cam @ T_cam_tag
                    side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
                    logger.info(f"tag {found_id}: {side_px:.0f} px, dist {np.linalg.norm(T_cam_tag[:3,3]):.3f} m")
                    logger.info(f"  T_cam_tag : {fmt_pose(T_cam_tag)}")
                    logger.info(f"  T_base_cam: {fmt_pose(T_base_cam)}")
                    logger.info(f"  T_base_tag: {fmt_pose(T_base_tag)}")
                    if gt_file is not None and Path(gt_file).is_file():
                        T_gt = load_gt(Path(gt_file))
                        if T_gt is not None:
                            dp, dr = pose_error(T_base_tag, T_gt)
                            logger.info(f"  vs sim GT : {fmt_pose(T_gt)}")
                            logger.info(f"  error     : {dp * 1000:.1f} mm, {dr:.2f} deg")
                    if out:
                        q = R.from_matrix(T_base_tag[:3, :3]).as_quat()
                        with open(out, "w", encoding="utf-8") as f:
                            f.write("# frame=robot_base  x y z qw qx qy qz\n")
                            f.write(" ".join(f"{v:.6f}" for v in [*T_base_tag[:3, 3], q[3], q[0], q[1], q[2]]) + "\n")
                            f.write(f"# tag family=tag36h11 id={found_id} size={tag_size}\n")
                if save:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if det is not None:
                        cv2.aruco.drawDetectedMarkers(bgr, [det[1].reshape(1, 4, 2).astype(np.float32)], np.array([[det[2]]]))
                        rvec, _ = cv2.Rodrigues(det[0][:3, :3])
                        cv2.drawFrameAxes(bgr, K, None, rvec, det[0][:3, 3], tag_size * 0.5)
                    cv2.imwrite(save, bgr)
            if not loop:
                break
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        disconnect_leave_held(bot)


if __name__ == "__main__":
    tyro.cli(main)
