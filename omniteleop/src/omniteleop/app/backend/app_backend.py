#!/usr/bin/env python3
"""OmniTeleop App Backend.

Orchestrates the full teleoperation stack:
  - Launches leader readers (arm, joycon) and follower processes
    (command_processor, robot_controller) as managed subprocesses.
  - Streams camera images and robot state to the frontend via WebSocket.
  - Optionally runs the MDP recorder when --record-mode is set.

Usage:
    python -m omniteleop.app.backend.app_backend                 # teleop only
    python -m omniteleop.app.backend.app_backend --record-mode   # teleop + recording
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from contextlib import asynccontextmanager
import cv2
import numpy as np
import tyro
from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter
from dexcontrol.robot import Robot
from dexcontrol.core.config import get_robot_config
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger
import uvicorn

from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.record.webrtc_receiver import WebRTCReceiver

from omniteleop.app.backend.state_checker import StateChecker
from omniteleop.app.backend.backend_utils.state_manager import state_manager
from omniteleop.app.backend.backend_utils.state_publisher import state_publisher

# ---------------------------------------------------------------------------
# Optional recorder import — only needed when --record-mode is active
# ---------------------------------------------------------------------------
try:
    from omniteleop.app.backend.mcap_utils import TeleopMcapWriter
    from omniteleop.app.backend.mcap_utils.episode import (
        RobotState,
        RobotAction,
        JointState,
        WrenchState,
    )

    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Predefined poses for /control/predefined (absolute joint targets),
# loaded from predefined_poses.json at the project root
# ---------------------------------------------------------------------------
def _load_predefined_poses() -> Dict[str, Dict[str, list]]:
    """Load predefined poses from predefined_poses.json at the project root.

    Searches the current working directory first (when the backend is
    launched from the project root), then the repository root relative to
    this file (when running from a source checkout). Returns an empty dict
    with a warning when the file is not found or invalid.
    """
    candidates = [
        Path.cwd() / "predefined_poses.json",
        Path(__file__).resolve().parents[4] / "predefined_poses.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    poses = json.load(f)
                logger.info(
                    f"Loaded {len(poses)} predefined poses from {path}"
                )
                return poses
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(f"Failed to parse {path}: {exc}")
                return {}
    logger.warning(
        f"predefined_poses.json not found (looked in: "
        f"{[str(c) for c in candidates]}); predefined pose control disabled"
    )
    return {}


PREDEFINED_POSES = _load_predefined_poses()

class TeleopApp:
    """Manages the full teleoperation stack and exposes a FastAPI server to the frontend."""

    # Sign map for mirrored dual-arm joint motion: right joint delta =
    # sign * left joint delta. Verified against the vega URDF by checking
    # which sign makes the right EE motion the y-mirror of the left EE
    # motion (translation + rotation); consistent with the symmetric
    # init_pos and crouch poses. Only j4 (elbow pitch) moves the same way.
    ARM_MIRROR_SIGNS = [-1, -1, -1, 1, -1, -1, -1]

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
    ):
        self.namespace = namespace
        self.debug = debug
        self.record_mode = False  # set per-launch from /teleop/start body

        self.config = get_config()
        recorder_cfg = self.config.get("recorder", {})

        # ---- paths & rates ------------------------------------------------
        save_dir_str = recorder_cfg.get("save_dir", "recordings")
        save_path = Path(save_dir_str)
        if not save_path.is_absolute():
            save_path = Path.home() / save_path
        self.save_dir = save_path
        self.episode_prefix = recorder_cfg.get("episode_prefix", "episode")
        self.record_rate = recorder_cfg.get("record_rate", 20.0)
        resolution = recorder_cfg.get("image_resolution", [640, 480])
        self.image_resolution = (
            tuple(resolution) if isinstance(resolution, list) else (640, 480)
        )
        self.auto_stop_on_estop = recorder_cfg.get("auto_stop_on_estop", True)

        # ---- component flags -----------------------------------------------
        # Load all component flags from config; any key present in the yaml is
        # respected. Keys not in the yaml default to True for the core joints,
        # False for optional sensors (wrist cameras/wrenches).
        components_cfg = recorder_cfg.get("components", {})
        _defaults: Dict[str, bool] = {
            "left_arm": True,
            "right_arm": True,
            "torso": True,
            "head": True,
            "left_hand": True,
            "right_hand": True,
            "head_left_rgb": True,
            "head_right_rgb": True,
            "left_wrist_rgb": False,
            "right_wrist_rgb": False,
            "left_wrist_wrench": False,
            "right_wrist_wrench": False,
        }
        self.record_components: Dict[str, bool] = {
            **_defaults,
            **{k: bool(v) for k, v in components_cfg.items()},
        }
        self.save_images = any(
            self.record_components.get(k)
            for k in [
                "head_left_rgb",
                "head_right_rgb",
                "left_wrist_rgb",
                "right_wrist_rgb",
            ]
        )

        # ---- subprocess handles --------------------------------------------
        self._procs: List[subprocess.Popen] = []
        self._proc_names: List[str] = []  # parallel name list for each proc
        self._teleop_running = False  # True once start_teleop() succeeds
        self._teleop_starting = False  # True while /teleop/start is spawning/initializing

        # ---- per-process log ring-buffers & subscribers -------------------
        # _proc_logs[name] = deque of {"source": name, "line": text}
        self._proc_logs: Dict[str, List[Dict[str, str]]] = {}
        self._log_lock = threading.Lock()
        self._log_subscribers: List[asyncio.Queue] = []
        self._log_loop: Optional[asyncio.AbstractEventLoop] = None

        # ---- robot / state -------------------------------------------------
        self.robot: Optional[Robot] = None
        self._node = Node(name="app_backend", namespace=namespace)

        # Subscribe to robot joint feedback
        joints_topic = self.config.get_topic("robot_joints")
        self._node.create_subscriber(
            joints_topic,
            callback=self._on_joints_received,
            decoder=DictDataCodec.decode,
        )

        # Subscribe to robot commands — always (estop sync) + record mode (action tracking)
        commands_topic = self.config.get_topic("robot_commands")
        self._node.create_subscriber(
            commands_topic,
            callback=self._on_command_received,
            decoder=DictDataCodec.decode,
        )

        self._node.create_subscriber(
            self.config.get_topic("recorder_control", "recorder/control"),
            callback=self._on_record_received,
            decoder=DictDataCodec.decode,
        )

        # Setup publishers
        self.common_pub = self._node.create_publisher(
            self.config.get_topic("common_input", "common/input"),
            DictDataCodec.encode
        )
        self.record_pub = self._node.create_publisher(
            self.config.get_topic("recorder_control", "recorder/control"),
            DictDataCodec.encode
        )
        self.reset_pub = self._node.create_publisher(
            "reset_pose/cmd",
            DictDataCodec.encode
        )

        # Start dexcomm node in a background thread
        self.node_thread = threading.Thread(target=self._node.spin, daemon=True)
        self.node_thread.start()

        self.state_checker = None  # initialized in _start_teleop_procs
        self.robot_name = os.environ.get("ROBOT_NAME")
        self.robot_config = os.environ.get("ROBOT_CONFIG")
        self.zenoh_config = os.environ.get("ZENOH_CONFIG")

        self.robot_state = "BOOT"
        self._robot_state_lock = threading.Lock()
        self._state_checker_thread: Optional[threading.Thread] = None
        self._state_checker_running = False

        self.frontend_ready = False

        # ---- recording state (only meaningful when record_mode=True) -------
        self.is_recording = False
        self.episode_dir: Optional[Path] = None
        self.mcap_writer = None
        self.episode_start_time: float = 0.0
        self.transitions_in_episode: int = 0
        self.total_episodes: int = 0
        self.total_transitions: int = 0
        self._record_thread: Optional[threading.Thread] = None
        self._record_running = False
        self._action_lock = threading.Lock()
        self.current_action: Dict[str, Any] = {}
        self.last_action: Dict[str, Any] = {}
        self._component_last_seen: Dict[
            str, float
        ] = {}  # epoch time of last command per component
        self._active_component_window = 0.5  # seconds
        self._latest_robot_joints: Dict[str, Any] = {}  # from robot/joints topic
        self._rate_limiter = RateLimiter(self.record_rate)

        # ---- chassis press-and-hold state (dead-man's switch) ---------------
        # Mirrors meta_con BaseController: velocity flows only while the input
        # is active; releasing (or losing the heartbeat) publishes zero velocity.
        # Endpoint and watchdog both run on the asyncio event loop, so no lock.
        self._chassis_hold: Optional[str] = None  # held direction, None if idle
        self._chassis_expire_at: float = 0.0  # heartbeat deadline (epoch)
        self._chassis_heartbeat_timeout = 0.6  # s without heartbeat -> auto stop

        # ---- reach (camera-based object approach) ----------------------------
        self._reach_in_progress = False
        self._reach_receivers: Dict[str, Any] = {}  # "ip:port" -> WebRTCReceiver
        self._reach_lock = threading.Lock()
        # Long-lived planner daemon (pantos venv): keeps Grounding DINO + the
        # URDF model loaded so each reach press only pays the compute time.
        self._reach_planner_proc: Optional[subprocess.Popen] = None
        self._reach_planner_lock = threading.Lock()   # guards spawn/terminate
        self._reach_daemon_io_lock = threading.Lock()  # serializes stdin/stdout

        # ---- FastAPI -------------------------------------------------------
        self.robot_initialized_event: Optional[asyncio.Event] = None
        self.app = FastAPI(title="OmniTeleop App Backend", lifespan=self.lifespan)
        self._setup_fastapi()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Handle startup and shutdown events."""
        self._log_loop = asyncio.get_running_loop()
        self.robot_initialized_event = asyncio.Event()

        # Sink backend's own loguru output into the log stream
        def _backend_log_sink(msg):
            line = self.strip_ansi(msg.rstrip("\n"))
            entry = {"source": "app_backend", "line": line}
            with self._log_lock:
                buf = self._proc_logs.setdefault("app_backend", [])
                buf.append(entry)
                if len(buf) > 500:
                    buf.pop(0)
                subs = list(self._log_subscribers)
            loop = self._log_loop
            if loop and loop.is_running():
                for q in subs:
                    loop.call_soon_threadsafe(self._safe_put_log, q, entry)

        handler_id = logger.add(_backend_log_sink, format="{level}: {message}")

        # Chassis dead-man's switch: auto-stop if press heartbeats stop arriving
        chassis_watchdog_task = asyncio.create_task(self._chassis_watchdog_loop())

        try:
            yield
        finally:
            chassis_watchdog_task.cancel()
            # Lifespan shutdown runs on every graceful uvicorn exit (SIGINT,
            # SIGTERM, SIGHUP). The teleop subprocesses were spawned with
            # start_new_session=True and outlive this process unless stopped
            # here; skipping this orphans them and they keep publishing on
            # the robot's Zenoh topics.
            self._stop_teleop_procs()
            self._stop_reach_planner()
            logger.remove(handler_id)

    # =========================================================================
    # FastAPI setup
    # =========================================================================

    def _setup_fastapi(self):
        app = self.app

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # ------------------------------------------------------------------
        # State callback fed into state_publisher
        # ------------------------------------------------------------------
        async def build_state_dict() -> Dict[str, Any]:
            return {
                "command": "observation",
                "content": self._latest_robot_joints
            }

        state_publisher.set_state_callback(build_state_dict)
        state_publisher.set_connection_callback(self.on_frontend_connected)
        state_publisher.set_disconnection_callback(self.on_frontend_disconnected)

        # ------------------------------------------------------------------
        # REST endpoints
        # ------------------------------------------------------------------

        @app.get("/config")
        async def config():
            # 1. ROBOT_NAME check
            robot_name = os.environ.get("ROBOT_NAME")
            if not robot_name:
                return {
                    "result": "fail",
                    "content": "error: set ROBOT_NAME before launch",
                }

            # 2. ROBOT_CONFIG check
            if not os.environ.get("ROBOT_CONFIG"):
                return {
                    "result": "fail",
                    "content": "error: set ROBOT_CONFIG before launch",
                }

            # 3. ZENOH_CONFIG or 자동으로 찾아서 있는지 check.
            if not os.environ.get("ZENOH_CONFIG"):
                return {
                    "result": "fail",
                    "content": "error: set ZENOH_CONFIG before launch",
                }

            # Check for active robot by running "dextop topic list"
            try:
                proc = await asyncio.create_subprocess_shell(
                    "dextop topic list",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                output = stdout.decode()
                # Check if "/state" appears at least twice
                if output.count(f"{robot_name}/state") < 2:
                    return {
                        "result": "fail",
                        "content": "error: ROBOT is not connected",
                    }
            except Exception as e:
                logger.warning(f"Failed to check active robot via 'dextop topic list': {e}")

            webrtc_cfg = self.config.get("image", {}).get("webrtc", {})
            return {
                "result": "success",
                "content": {
                    "robot_name": robot_name,
                    "robot_config": os.environ.get("ROBOT_CONFIG"),
                    "zenoh_config": os.environ.get("ZENOH_CONFIG"),
                    "ip": webrtc_cfg.get("ip"),
                    "port": webrtc_cfg.get("port"),
                    "position": webrtc_cfg.get("position"),
                }
            }

        @app.get("/health")
        async def health():
            return {"status": "ok", "timestamp": time.time()}

        @app.get("/state")
        async def get_state():
            return {
                "teleop_running": self._teleop_running,
                "is_recording": self.is_recording,
            }

        @app.get("/sensors")
        async def get_sensors():
            sensors = [
                {"id": "camera_1", "data_type": "video/x-motion-jpeg"},
                {"id": "camera_2", "data_type": "video/x-motion-jpeg"},
            ]
            return JSONResponse(content=sensors)

        # -- Teleop controls ------------------------------------------------

        @app.post("/teleop/start")
        async def teleop_start(request: Request):
            """Launch the teleop subprocess stack.

            Body (all fields optional):
            {
                "record_mode": false,
                "env": {
                    "ROBOT_NAME": "...",
                    "ROBOT_CONFIG": "vega_1_f5d6",
                    "ZENOH_CONFIG": "/path/to/zenoh.json"
                }
            }
            """
            # _teleop_running only turns True after the 20s init wait below, so
            # checking it alone lets a second click during that window spawn a
            # duplicate teleop stack — block on the starting flag too, and set
            # it before the first await so concurrent requests can't interleave.
            if self._teleop_running or self._teleop_starting:
                raise HTTPException(status_code=409, detail="Teleop already running")
            self._teleop_starting = True
            try:
                if await state_manager.is_estop():
                    raise HTTPException(
                        status_code=403, detail="Cannot start: E-Stop is active"
                    )
                try:
                    body = await request.json()
                except Exception:
                    body = {}
                self.record_mode = bool(body.get("record_mode", False))
                user_env: Dict[str, str] = {
                    k: str(v) for k, v in (body.get("env") or {}).items() if v
                }
                self._start_teleop_procs(user_env)
                # Preload the reach planner daemon in the background so the
                # first Reach press doesn't pay the model-load cost.
                self._warm_reach_planner_async()
                if self.robot_initialized_event:
                    try:
                        await asyncio.wait_for(self.robot_initialized_event.wait(), timeout=20.0)
                        self._teleop_running = True
                        logger.success("Teleop stack started and initialized")
                        result = "success"
                    except asyncio.TimeoutError:
                        logger.error("Teleop stack failed to initialize within 20s")
                        self._stop_teleop_procs()
                        result = "fail"
                else:
                    # Fallback if event is not available (should not happen with lifespan)
                    self._teleop_running = True
                    result = "success"
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            finally:
                self._teleop_starting = False

            resp = await build_state_dict()
            resp["result"] = result
            return JSONResponse(content=resp)

        @app.post("/teleop/stop")
        async def teleop_stop():
            """Stop the teleop subprocess stack."""
            # Episode teardown must not be able to skip the subprocess kill
            # (an exception there would otherwise leave them orphaned).
            try:
                if self.is_recording:
                    self._end_episode(is_success=False)
            finally:
                self._stop_teleop_procs()
            return JSONResponse(content=await build_state_dict())

        @app.post("/teleop/reset")
        async def teleop_reset():
            self.reset_pub.publish({"reset_category": 1})
            logger.info("Published Reset to reset_pose/cmd 1")
            return {"status": "ok"}

        @app.post("/teleop/resetall")
        async def teleop_resetall():
            self.reset_pub.publish({"reset_category": 2})
            logger.info("Published Reset All to reset_pose/cmd 2")
            return {"status": "ok"}

        # -- Recording controls (only active when record_mode was set at /teleop/start) --

        @app.post("/record/start")
        async def record_start(request: Request):
            if await state_manager.is_estop():
                raise HTTPException(
                    status_code=403, detail="Robot is in emergency stop"
                )
            if self.is_recording:
                raise HTTPException(status_code=409, detail="Already recording")
            try:
                body = await request.json()
            except Exception:
                body = {}

            self.record_pub.publish({"command": "start"})
            self.is_recording = True
            logger.info(f"Starting episode recording")
            # self._start_episode(metadata=body)
            return { "result": "success" }

        @app.post("/record/stop")
        async def record_stop(request: Request):
            if not self.is_recording:
                raise HTTPException(status_code=409, detail="Not currently recording")
            try:
                body = await request.json()
                is_success = body.get("is_success", True)
            except Exception:
                is_success = True

            self.record_pub.publish({"command": "stop" if is_success else "drop"})
            self.is_recording = False
            logger.info(f"Stopping episode recording")
            # episode_id = self._end_episode(is_success=is_success)
            return {"result": "success"}

        # -- control ----------------------------------------------------------
        @app.post("/control/arm/joint")
        async def control_arm_joint(request: Request):
            """Joint jog of the selected arm(s), one joint at a time.

            Body: {"side": "left"|"right"|"both" (default "both"),
            "joint_idx": 0..6, "delta": rad}.

            - "left"/"right": the delta goes to that arm's joint only.
            - "both": the same delta goes to the left arm while the right
              arm receives the mirror-mapped sign so the arms move like
              mirror images of each other.
            """
            try:
                body = await request.json()
                side = body.get("side") or "both"
                if side not in ("left", "right", "both"):
                    raise HTTPException(status_code=400, detail="side must be one of: left, right, both")
                joint_idx = body.get("joint_idx")
                delta = body.get("delta")
                if joint_idx is None or delta is None:
                    raise HTTPException(status_code=400, detail="joint_idx and delta are required")
                if not 0 <= joint_idx < 7:
                    raise HTTPException(status_code=400, detail="joint_idx must be between 0 and 6")

                left_pos = [0.0] * 7
                right_pos = [0.0] * 7
                if side == "left":
                    left_pos[joint_idx] = delta
                elif side == "right":
                    right_pos[joint_idx] = delta
                else:
                    left_pos[joint_idx] = delta
                    right_pos[joint_idx] = delta * self.ARM_MIRROR_SIGNS[joint_idx]

                components = {}
                if side in ("left", "both"):
                    components["left_arm"] = {"mode": "relative", "pos": left_pos}
                if side in ("right", "both"):
                    components["right_arm"] = {"mode": "relative", "pos": right_pos}

                cmd = {
                    "timestamp_ns": time.time_ns(),
                    "components": components,
                }
                self.common_pub.publish(cmd)
                return {
                    "status": "ok",
                    "side": side,
                    "joint_idx": joint_idx,
                    "left_delta": left_pos[joint_idx],
                    "right_delta": right_pos[joint_idx],
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in control_arm_joint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/control/arm/cartesian")
        async def control_arm_cartesian(request: Request):
            """Cartesian jog of the selected arm(s): translation or rotation.

            Body: {"side": "left"|"right"|"both" (default "both"),
            "mode": "pos"|"rot" (default "pos"),
            "rot_frame": "world"|"ee" (default "world", rot only),
            "axis_idx": 0|1|2, "delta": meters (pos) | radians (rot)}.
            Publishes an EE-pose delta command for the selected arm(s) so
            each repeats as "current pose + delta" while the frontend
            button is held. "pos" jogs translation via delta_xyz; "rot"
            rotates via delta_R expressed in the robot base frame
            ("world", delta_R @ current_rot) or the gripper's own frame
            ("ee", current_rot @ delta_R). Position is untouched either way.
            """
            try:
                body = await request.json()
                side = body.get("side") or "both"
                if side not in ("left", "right", "both"):
                    raise HTTPException(status_code=400, detail="side must be one of: left, right, both")
                mode = body.get("mode") or "pos"
                if mode not in ("pos", "rot"):
                    raise HTTPException(status_code=400, detail="mode must be one of: pos, rot")
                rot_frame = body.get("rot_frame") or "world"
                if rot_frame not in ("world", "ee"):
                    raise HTTPException(status_code=400, detail="rot_frame must be one of: world, ee")
                axis_idx = body.get("axis_idx")
                delta = body.get("delta")
                if axis_idx is None or delta is None:
                    raise HTTPException(status_code=400, detail="axis_idx and delta are required")
                if not 0 <= axis_idx < 3:
                    raise HTTPException(status_code=400, detail="axis_idx must be between 0 and 2")

                # delta_xyz always ships (zeros for rot-only jogs) so the
                # command_processor dashboard-jog branch that requires a
                # delta field accepts rotation commands unchanged.
                delta_xyz = [0.0, 0.0, 0.0]
                arm_command = {"command_type": "ee_pose", "delta_xyz": delta_xyz}
                if mode == "pos":
                    delta_xyz[axis_idx] = delta
                else:
                    arm_command["delta_R"] = self._axis_rotation_matrix(axis_idx, delta)
                    arm_command["rot_frame"] = rot_frame
                components = {}
                if side in ("left", "both"):
                    components["left_arm"] = dict(arm_command)
                if side in ("right", "both"):
                    components["right_arm"] = dict(arm_command)

                cmd = {
                    "timestamp_ns": time.time_ns(),
                    "components": components
                }
                self.common_pub.publish(cmd)
                return {"status": "ok", "side": side, "mode": mode, "axis_idx": axis_idx, "delta": delta}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in control_arm_cartesian: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/control/head")
        async def control_head(request: Request):
            try:
                body = await request.json()
                joint_idx = body.get("joint_idx")
                delta = body.get("delta")
                direction = body.get("direction")

                pos = [0.0, 0.0, 0.0]
                if joint_idx is not None and delta is not None:
                    if 0 <= joint_idx < 3:
                        pos[joint_idx] = delta
                    else:
                        raise HTTPException(status_code=400, detail="joint_idx must be between 0 and 2")
                elif direction:
                    sensitivity = 0.05
                    if direction == 'up': pos[0] = sensitivity
                    if direction == 'down': pos[0] = -sensitivity
                    if direction == 'left': pos[1] = sensitivity
                    if direction == 'right': pos[1] = -sensitivity
                else:
                    raise HTTPException(status_code=400, detail="joint_idx/delta or direction is required")

                cmd = {
                    "timestamp_ns": time.time_ns(),
                    "components": {
                        "head": {
                            "mode": "relative",
                            "pos": pos
                        }
                    }
                }
                self.common_pub.publish(cmd)
                return {"status": "ok", "pos": pos}
            except Exception as e:
                logger.error(f"Error in control_head: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/control/torso")
        async def control_torso(request: Request):
            """Torso jog commands.

            Three request shapes:
            - {"joint_idx": 0..2, "delta"}: joint-space jog of torso_jN
              (dashboard "Torso N" selector; 'pos' is applied as joint deltas).
            - {"axis_idx": 0|2, "delta"}: Cartesian head-frame jog via torso
              IK (dashboard "up/down" and "front/back" selector). In the robot
              base frame x is forward and z is up, so axis 0 (pos[0]) drives
              front/back and axis 2 (pos[2]) drives up/down.
            - {"direction": "up|down|left|right"}: legacy direction command.
            """
            try:
                body = await request.json()
                joint_idx = body.get("joint_idx")
                axis_idx = body.get("axis_idx")
                delta = body.get("delta")
                direction = body.get("direction")

                pos = [0.0, 0.0, 0.0]
                use_joint = False
                if joint_idx is not None and delta is not None:
                    if 0 <= joint_idx < 3:
                        pos[joint_idx] = delta
                        use_joint = True
                    else:
                        raise HTTPException(status_code=400, detail="joint_idx must be between 0 and 2")
                elif axis_idx is not None and delta is not None:
                    if axis_idx in (0, 2):
                        pos[axis_idx] = delta
                    else:
                        raise HTTPException(status_code=400, detail="axis_idx must be 0 (front/back) or 2 (up/down)")
                elif direction:
                    sensitivity = 0.015
                    if direction == 'up': pos[2] = sensitivity
                    if direction == 'down': pos[2] = -sensitivity
                    if direction == 'left': pos[0] = -sensitivity
                    if direction == 'right': pos[0] = sensitivity
                else:
                    raise HTTPException(status_code=400, detail="joint_idx/delta, axis_idx/delta or direction is required")

                torso_component = {
                    "mode": "relative",
                    "pos": pos,
                }
                if use_joint:
                    torso_component["use_joint"] = True

                cmd = {
                    "timestamp_ns": time.time_ns(),
                    "components": {
                        "torso": torso_component,
                    }
                }
                self.common_pub.publish(cmd)
                return {"status": "ok", "pos": pos, "use_joint": use_joint}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in control_torso: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @app.post("/control/chassis")
        async def control_chassis(request: Request):
            """Chassis press-and-hold control (dead-man's switch).

            The frontend sends {"direction": ..., "pressed": true,
            "speed": 0.5} while a button is held (repeated as a heartbeat)
            and {"pressed": false} on release. "speed" is the linear
            velocity in m/s (default 0.5 = 1x); angular velocity scales
            proportionally (1.0 rad/s at the default).
            The robot latches the last velocity, so velocity is published only on
            transitions; a watchdog publishes zero velocity if heartbeats stop.
            """
            try:
                body = await request.json()
                direction = body.get("direction")
                if not direction:
                    raise HTTPException(status_code=400, detail="Direction is required")
                pressed = bool(body.get("pressed", True))

                speed = body.get("speed", 0.5)
                try:
                    speed = float(speed)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail="speed must be a number")
                if not 0.0 < speed <= 1.5:
                    raise HTTPException(status_code=400, detail="speed must be in (0, 1.5] m/s")

                if pressed and direction != "stop":
                    # Button held: latch the direction and refresh the heartbeat
                    self._chassis_hold = direction
                    self._chassis_expire_at = (
                        time.time() + self._chassis_heartbeat_timeout
                    )
                    vx, vy, wz = self._chassis_direction_to_velocity(direction, speed)
                    self._publish_chassis_velocity(vx, vy, wz)
                    return {"status": "ok", "direction": direction, "detail": "hold"}

                # Button released (or explicit stop): clear hold, stop chassis
                self._chassis_hold = None
                self._publish_chassis_velocity(0.0, 0.0, 0.0)
                return {"status": "ok", "direction": direction, "detail": "stop"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in control_chassis: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/control/hand")
        async def control_hand(request: Request):
            """Open/close the selected hand(s).

            Body: {"pose": "open"|"close", "side": "left"|"right"|"both"
            (default "both")}. Target positions come from the common hands
            configuration (input_handlers.common.hands).
            """
            try:
                body = await request.json()
                pose = body.get("pose")
                if pose not in ("open", "close"):
                    raise HTTPException(status_code=400, detail="pose must be 'open' or 'close'")
                side = body.get("side") or "both"
                if side not in ("left", "right", "both"):
                    raise HTTPException(status_code=400, detail="side must be one of: left, right, both")

                # Retrieve open/close configurations from common hands configuration
                common_hands = self.config.get("input_handlers", {}).get("common", {}).get("hands", {})
                left_ee_type = common_hands.get("left_ee_type") or common_hands.get("ee_type")
                right_ee_type = common_hands.get("right_ee_type") or common_hands.get("ee_type")

                # Default fallback limits
                left_open_pos = [0.0]
                left_close_pos = [1.0]
                right_open_pos = [0.0]
                right_close_pos = [1.0]

                if left_ee_type:
                    left_cfg = common_hands.get(left_ee_type, {}).get("left", {})
                    left_open_pos = left_cfg.get("open", [0.0])
                    left_close_pos = left_cfg.get("close", [1.0])

                if right_ee_type:
                    right_cfg = common_hands.get(right_ee_type, {}).get("right", {})
                    right_open_pos = right_cfg.get("open", [0.0])
                    right_close_pos = right_cfg.get("close", [1.0])

                hand_pos = {
                    "left": left_open_pos if pose == "open" else left_close_pos,
                    "right": right_open_pos if pose == "open" else right_close_pos,
                }

                components = {}
                if side in ("left", "both"):
                    components["left_hand"] = {"mode": "absolute", "pos": hand_pos["left"]}
                if side in ("right", "both"):
                    components["right_hand"] = {"mode": "absolute", "pos": hand_pos["right"]}

                cmd = {"timestamp_ns": time.time_ns(), "components": components}
                self.common_pub.publish(cmd)
                return {"status": "ok", "pose": pose, "side": side}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in control_hand: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/control/predefined")
        async def control_predefined(request: Request):
            try:
                body = await request.json()
                pose = body.get("pose")
                if not pose:
                    raise HTTPException(status_code=400, detail="Pose name is required")

                now = time.time_ns()

                # Absolute joint targets come from the PREDEFINED_POSES table
                global PREDEFINED_POSES
                pose_components = PREDEFINED_POSES.get(pose)
                if pose_components is None:
                    # Dynamically reload predefined_poses.json in case it was updated
                    PREDEFINED_POSES = _load_predefined_poses()
                    pose_components = PREDEFINED_POSES.get(pose)

                if pose_components is None:
                    raise HTTPException(status_code=400, detail=f"Unknown pose: {pose}")

                cmd = {
                    "timestamp_ns": now,
                    "components": {
                        name: {"mode": "absolute", "pos": list(pos)}
                        for name, pos in pose_components.items()
                    },
                }
                self.common_pub.publish(cmd)

                return {"status": "ok", "pose": pose}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in control_predefined: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # -- Reach (camera-based object approach) ------------------------------
        @app.post("/control/reach")
        async def control_reach(request: Request):
            """Detect an object with the head camera and reach toward it.

            Flow: grab head RGB+depth frames over WebRTC, run the pantos_tele
            Grounding-DINO + Pink-IK planner in a subprocess, then publish the
            resulting arm joint targets on common/input (same pipeline as the
            predefined poses). Body may contain {"detect_only": true} to run
            detection without moving the arms.
            """
            if self._reach_in_progress:
                raise HTTPException(status_code=409, detail="reach already in progress")
            self._reach_in_progress = True
            try:
                return await self._run_reach(request)
            finally:
                self._reach_in_progress = False

        # -- E-Stop ----------------------------------------------------------
        @app.post("/robots/estop")
        async def robots_estop():
            await state_manager.estop()
            if self.is_recording:
                self._end_episode(is_success=False)
            if self.robot and hasattr(self.robot, "estop"):
                try:
                    self.robot.estop.activate()
                except Exception as exc:
                    logger.error(f"Hardware estop failed: {exc}")
            return JSONResponse(content=await build_state_dict())

        @app.delete("/robots/estop")
        async def robots_estop_clear():
            if not await state_manager.is_estop():
                raise HTTPException(status_code=409, detail="Not in emergency stop")
            await state_manager.clear_estop()
            if self.robot and hasattr(self.robot, "estop"):
                try:
                    self.robot.estop.deactivate()
                except Exception as exc:
                    logger.error(f"Hardware estop clear failed: {exc}")
            return JSONResponse(content=await build_state_dict())

        # -- WebSocket endpoints --------------------------------------------
        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await state_publisher.start_stream(websocket)



        @app.websocket("/ws/logs")
        async def ws_logs(websocket: WebSocket):
            """Stream per-process log lines as JSON: {"source": "omni-arm", "line": "..."}"""
            await websocket.accept()
            import json as _json

            q: asyncio.Queue = asyncio.Queue(maxsize=500)
            with self._log_lock:
                # Send buffered history first
                history = []
                for name, lines in self._proc_logs.items():
                    history.extend(lines[-200:])
                self._log_subscribers.append(q)
            try:
                for entry in history:
                    await websocket.send_text(_json.dumps(entry))
                while True:
                    entry = await q.get()
                    await websocket.send_text(_json.dumps(entry))
            except Exception:
                pass
            finally:
                with self._log_lock:
                    try:
                        self._log_subscribers.remove(q)
                    except ValueError:
                        pass

        # -- Serve built frontend if present --------------------------------
        frontend_path = Path("/var/www/html")
        if frontend_path.exists() and (frontend_path / "index.html").exists():

            @app.get("/{full_path:path}")
            async def serve_frontend(full_path: str):
                skip = ["state", "sensors", "health"]
                skip_prefixes = ("robots/", "record/", "teleop/", "ws/")
                if full_path in skip or full_path.startswith(skip_prefixes):
                    raise HTTPException(status_code=404)
                file_path = frontend_path / full_path
                if file_path.exists() and file_path.is_file():
                    return FileResponse(str(file_path))
                return FileResponse(str(frontend_path / "index.html"))

    # =========================================================================
    # Cartesian jog helpers
    # =========================================================================

    @staticmethod
    def _axis_rotation_matrix(axis_idx: int, angle_rad: float) -> List[List[float]]:
        """Rotation matrix about a robot-base-frame coordinate axis (Rodrigues).

        Returns plain Python floats so DictDataCodec/JSON can serialize it.
        """
        axis = np.zeros(3)
        axis[axis_idx] = 1.0
        kx, ky, kz = (float(v) for v in axis)
        c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        t = 1.0 - c
        return [
            [t * kx * kx + c,      t * kx * ky - s * kz,  t * kx * kz + s * ky],
            [t * kx * ky + s * kz, t * ky * ky + c,       t * ky * kz - s * kx],
            [t * kx * kz - s * ky, t * ky * kz + s * kx,  t * kz * kz + c],
        ]

    # =========================================================================
    # Chassis press-and-hold helpers
    # =========================================================================

    @staticmethod
    def _chassis_direction_to_velocity(direction: str, speed: float = 0.5):
        """Map a chassis direction to (vx, vy, wz) at the given linear speed.

        Speed is the linear velocity in m/s and defaults to the meta_con
        BaseController default (0.5 m/s); the angular velocity scales
        proportionally (1.0 rad/s at the default speed).
        """
        angular = speed * 2.0
        vx, vy, wz = 0.0, 0.0, 0.0
        if direction == 'up': vx = speed
        elif direction == 'down': vx = -speed
        elif direction == 'left': vy = speed
        elif direction == 'right': vy = -speed
        elif direction == 'rotate-left': wz = angular
        elif direction == 'rotate-right': wz = -angular
        return vx, vy, wz

    def _publish_chassis_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Publish a chassis velocity command on the common input topic."""
        cmd = {
            "timestamp_ns": time.time_ns(),
            "components": {
                "chassis": {
                    "vx": vx,
                    "vy": vy,
                    "wz": wz,
                }
            },
        }
        self.common_pub.publish(cmd)

    async def _chassis_watchdog_loop(self):
        """Auto-stop the chassis when press heartbeats stop arriving.

        Safety net for a lost release event (network drop, closed tab): if a
        direction is still held but its heartbeat deadline has passed, publish
        zero velocity once and clear the hold.
        """
        try:
            while True:
                await asyncio.sleep(0.15)
                if (
                    self._chassis_hold is not None
                    and time.time() > self._chassis_expire_at
                ):
                    direction = self._chassis_hold
                    self._chassis_hold = None
                    self._publish_chassis_velocity(0.0, 0.0, 0.0)
                    logger.warning(
                        f"Chassis heartbeat for '{direction}' expired - auto-stopped"
                    )
        except asyncio.CancelledError:
            pass

    # =========================================================================
    # Frontend connection lifecycle
    # =========================================================================

    async def on_frontend_connected(self):
        await asyncio.sleep(1)
        self.frontend_ready = True
        logger.info("Frontend ready")

    def on_frontend_disconnected(self):
        self.frontend_ready = False
        logger.info("Frontend disconnected")

    # =========================================================================
    # Teleop subprocess management
    # =========================================================================

    def _start_teleop_procs(self, user_env: Dict[str, str]):
        """Spawn all teleop subprocesses with the given env overrides."""
        # Defense in depth for the endpoint guard. A non-empty _procs list
        # means a stack is running, initializing, or was never stopped after
        # dying. Raise rather than return silently: a silent return would
        # leave the caller waiting on the cleared init event, whose timeout
        # path then kills the existing stack.
        if self._teleop_running or self._procs:
            raise ValueError("Teleop already running")

        env = os.environ.copy()

        # Update env with user_env values if provided
        for key in ["ROBOT_NAME", "ROBOT_CONFIG", "ZENOH_CONFIG"]:
            if user_env.get(key):
                env[key] = user_env[key]

        # Resolve ZENOH_CONFIG if it's just a cert name
        zenoh_val = env.get("ZENOH_CONFIG", "").strip()
        if zenoh_val and not zenoh_val.startswith("/") and not zenoh_val.startswith("."):
            # Treat as certificate name — expand to full path
            zenoh_base = Path.home() / ".dexmate" / "comm" / "zenoh"
            env["ZENOH_CONFIG"] = str(zenoh_base / f"{zenoh_val}.json5")
            logger.info(f"Expanded ZENOH_CONFIG certificate name: {zenoh_val} -> {env['ZENOH_CONFIG']}")

        # Simulation zenoh configs are plain .json5 files; real robots use
        # encrypted .dzcfg. dexcontrol's heartbeat watchdog (os._exit after a
        # 1s /heartbeat gap) guards real hardware, but the IsaacLab sim
        # publishes /heartbeat from its sim loop, which stalls >1s under
        # CPU-render load and would kill robot_controller mid-session — so
        # disable the watchdog for simulation runs only.
        if env.get("ZENOH_CONFIG", "").endswith(".json5"):
            env["DEXCONTROL_DISABLE_HEARTBEAT"] = "1"
            logger.info(
                "Simulation zenoh config detected (.json5) — injecting "
                "DEXCONTROL_DISABLE_HEARTBEAT=1 for teleop subprocesses"
            )

        # Keep instance attributes in sync with the environment used for processes
        self.robot_name = env.get("ROBOT_NAME")
        self.robot_config = env.get("ROBOT_CONFIG")
        self.zenoh_config = env.get("ZENOH_CONFIG")

        # Re-init StateChecker with ROBOT_NAME from env if provided
        robot_name = env.get("ROBOT_NAME", "")

        if not robot_name:
            raise ValueError("ROBOT_NAME is required to start teleop.")
        if not env.get("ROBOT_CONFIG", ""):
            raise ValueError("ROBOT_CONFIG is required to start teleop.")
        if not env.get("ZENOH_CONFIG", ""):
            raise ValueError("ZENOH_CONFIG is required to start teleop.")

        try:
            self._state_checker_running = False
            # Reuse the existing checker when it still targets the same robot: creating a
            # StateChecker opens a new dexcomm Node, which blocks this request handler for
            # ~2 s (mDNS discovery) before any teleop process is even spawned.
            if (
                self.state_checker is not None
                and self.state_checker.robot_name == robot_name
                and self.state_checker.is_connected()
            ):
                logger.info(f"StateChecker reused (robot={robot_name!r})")
            else:
                if self.state_checker:
                    self.state_checker.cleanup()
                self.state_checker = StateChecker(robot_name, rpi_mode=False)
                logger.info(f"StateChecker initialised (robot={robot_name!r})")
            self._state_checker_running = True
            # Assigning a new thread also makes any previous loop thread exit at its next
            # iteration (see the identity check in _state_checker_loop).
            self._state_checker_thread = threading.Thread(
                target=self._state_checker_loop, daemon=True, name="StateCheckerLoop"
            )
            self._state_checker_thread.start()
        except Exception as exc:
            logger.warning(f"StateChecker init failed: {exc}")

        _src = Path(__file__).parents[2]
        ns_args = ["--namespace", self.namespace] if self.namespace else []
        debug_args = ["--debug"] if self.debug else []
        commands = [
            [
                sys.executable,
                str(_src / "follower" / "command_processor.py"),
                *ns_args,
                *debug_args,
            ],
            [
                sys.executable,
                str(_src / "follower" / "robot_controller.py"),
                *ns_args,
                *debug_args,
                # "--interpolation-method",
                # "linear",
            ],
            [
                sys.executable,
                str(_src / "record" / "mdp_recorder.py"),
                *ns_args,
                *debug_args,
            ],
        ]
        if self.config.get("teleop_mode") == "vr":
            commands = [
                [
                    sys.executable,
                    str(_src / "leader" / "vr_reader.py"),
                    *ns_args,
                    *debug_args,
                ],
            ] + commands
        if self.config.get("teleop_mode") == "xsens":
            commands = [
                [
                    sys.executable,
                    str(_src / "leader" / "xsens_reader.py"),
                    *ns_args,
                    *debug_args,
                ],
            ] + commands

        if self.robot_initialized_event:
            self.robot_initialized_event.clear()

        for cmd in commands:
            name = Path(cmd[1]).stem  # e.g. "arm_reader" from "/path/to/arm_reader.py"
            proc_env = env.copy()

            # if name == "command_processor" and proc_env.get("ROBOT_CONFIG") == "vega_1p":
            #     proc_env["ROBOT_CONFIG"] = "vega_1p_gripper"

            logger.info(f"name: {name} ROBOT_NAME: {proc_env.get('ROBOT_NAME')} ROBOT_CONFIG: {proc_env.get('ROBOT_CONFIG')}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    env=proc_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self._procs.append(proc)
                self._proc_names.append(name)
                with self._log_lock:
                    self._proc_logs[name] = []
                threading.Thread(
                    target=self._tail_proc,
                    args=(proc, name),
                    daemon=True,
                    name=f"log-{name}",
                ).start()
                logger.info(f"Started: {' '.join(cmd)}  (pid={proc.pid})")
            except FileNotFoundError:
                logger.error(f"Command not found: {name} — is omniteleop installed?")

        mode_str = " [record mode]" if self.record_mode else ""
        logger.info(f"Teleop stack starting{mode_str}")

    def _stop_teleop_procs(self):
        """Terminate all teleop subprocesses and their entire process groups."""
        import signal

        for proc in self._procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass  # already dead
        # Wait then force-kill any stragglers
        for proc in self._procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
        self._procs.clear()
        self._proc_names.clear()
        self._state_checker_running = False
        self.robot_state = "BOOT"
        self._teleop_running = False
        self.record_mode = False
        self._teleop_env = {}
        logger.info("Teleop stack stopped")

    def _tail_proc(self, proc: subprocess.Popen, name: str):
        """Read stdout/stderr of a subprocess and fan-out to log subscribers."""
        MAX_LINES = 500

        log_file = None
        if name in ["robot_controller", "command_processor"]:
            log_dir = Path(__file__).parents[1] / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(log_dir / f"{name}.log", "a", encoding="utf-8")

        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                if log_file:
                    log_file.write(raw.rstrip("\n") + "\n")
                    log_file.flush()

                line = self.strip_ansi(raw.rstrip("\n"))
                entry = {"source": name, "line": line}
                with self._log_lock:
                    buf = self._proc_logs.get(name)
                    if buf is not None:
                        buf.append(entry)
                        if len(buf) > MAX_LINES:
                            buf.pop(0)
                    subs = list(self._log_subscribers)
                loop = self._log_loop
                if loop and loop.is_running():
                    if name == "robot_controller" and "Robot controller initialized" in line:
                        if self.robot_initialized_event:
                            loop.call_soon_threadsafe(self.robot_initialized_event.set)

                    for q in subs:
                        loop.call_soon_threadsafe(self._safe_put_log, q, entry)
        except Exception:
            pass
        finally:
            if log_file:
                log_file.close()

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences from a string."""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    @staticmethod
    def _safe_put_log(q: asyncio.Queue, entry: Dict[str, str]):
        """Safely put a log entry into an asyncio.Queue from within the event loop."""
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            # Drop the log line if the consumer is too slow
            pass
        except Exception:
            pass

    # =========================================================================
    # Robot / StateChecker background loop
    # =========================================================================

    def initialize(self):
        """Initialize robot interface, state-checker loop, and video publisher."""
        logger.info("Initializing robot interface…")
        self.robot_name = os.environ.get("ROBOT_NAME")
        self.robot_config = os.environ.get("ROBOT_CONFIG")
        self.zenoh_config = os.environ.get("ZENOH_CONFIG")

        if self.state_checker:
            self._state_checker_running = True
            self._state_checker_thread = threading.Thread(
                target=self._state_checker_loop, daemon=True, name="StateCheckerLoop"
            )
            self._state_checker_thread.start()

    def _state_checker_loop(self):
        # The identity check lets a superseded loop thread exit on its own when a new
        # thread has been started (e.g. stop -> start with a reused StateChecker).
        while (
            self._state_checker_running
            and self.state_checker
            and threading.current_thread() is self._state_checker_thread
        ):
            try:
                if self.robot:
                    try:
                        left = self.robot.left_arm.get_joint_pos().tolist()
                        right = self.robot.right_arm.get_joint_pos().tolist()
                        self.state_checker.update_robot_joints(left, right)
                    except Exception:
                        pass
                robot_state = self.state_checker.get_state()
                with self._robot_state_lock:
                    self.robot_state = robot_state
            except Exception as exc:
                logger.error(f"StateChecker loop error: {exc}")
                with self._robot_state_lock:
                    self.robot_state = "ERROR"
            time.sleep(1.0)

    # =========================================================================
    # Robot state observation helpers
    # =========================================================================

    def _get_robot_state(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"joint_pos": {}, "joint_vel": {}, "wrench": {}}
        rc = self.record_components

        # Use joint feedback published by robot_controller (robot/joints topic).
        # Falls back to direct hardware read only if no feedback has arrived yet.
        joints_data = self._latest_robot_joints
        joint_positions = joints_data.get("joints", {})
        joint_velocities = joints_data.get("velocities", {})

        if joint_positions:
            for comp in [
                "left_arm",
                "right_arm",
                "torso",
                "head",
                "left_hand",
                "right_hand",
            ]:
                if rc.get(comp) and comp in joint_positions:
                    obs["joint_pos"][comp] = joint_positions[comp]
            for comp in ["left_arm", "right_arm"]:
                if rc.get(comp) and comp in joint_velocities:
                    obs["joint_vel"][comp] = joint_velocities[comp]
        elif self.robot:
            # Fallback: direct hardware read (before robot_controller starts publishing)
            for comp, getter in [
                ("left_arm", lambda: self.robot.left_arm.get_joint_pos().tolist()),
                ("right_arm", lambda: self.robot.right_arm.get_joint_pos().tolist()),
            ]:
                if rc.get(comp):
                    try:
                        obs["joint_pos"][comp] = getter()
                    except Exception:
                        pass
        if self.robot:
            if rc.get("left_wrist_wrench") and self.robot.left_arm.wrench_sensor:
                w = self.robot.left_arm.wrench_sensor.get_wrench_state()
                obs["wrench"]["left"] = {
                    "force": w[:3].tolist(),
                    "torque": w[3:].tolist(),
                }
            if rc.get("right_wrist_wrench") and self.robot.right_arm.wrench_sensor:
                w = self.robot.right_arm.wrench_sensor.get_wrench_state()
                obs["wrench"]["right"] = {
                    "force": w[:3].tolist(),
                    "torque": w[3:].tolist(),
                }
        return obs

    # =========================================================================
    # Recording (only active when record_mode=True)
    # =========================================================================

    def _on_joints_received(self, data: Dict[str, Any]):
        self._latest_robot_joints = data

    def _on_command_received(self, data: Dict[str, Any]):
        now = time.time()
        with self._action_lock:
            for comp, comp_data in data.get("components", {}).items():
                self.current_action[comp] = comp_data
                self._component_last_seen[comp] = now

        estop_active = data.get("safety_flags", {}).get("emergency_stop", False)

        # Sync estop state from command_processor into state_manager so the GUI reflects it
        loop = self._log_loop
        if loop and loop.is_running():
            # Broadcast action to frontend
            msg = {
                "command": "action",
                "content": data
            }
            asyncio.run_coroutine_threadsafe(state_publisher.broadcast(msg), loop)

            if estop_active:
                asyncio.run_coroutine_threadsafe(state_manager.estop(), loop)
            else:
                asyncio.run_coroutine_threadsafe(state_manager.clear_estop(), loop)

        if self.auto_stop_on_estop and self.is_recording:
            if estop_active:
                self._end_episode(is_success=False)

    def _on_record_received(self, data: Dict[str, Any]):
        # command: recording or save or discard
        loop = self._log_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(state_publisher.broadcast(data), loop)

    def _start_episode(self, metadata: Dict[str, Any]):
        """Initialize and start recording a new episode."""
        if not RECORDER_AVAILABLE:
            logger.error("Recording requested but TeleopMcapWriter is not available (import failed)")
            return

        if self.is_recording:
            logger.warning("Start episode requested while already recording; ignoring.")
            return

        # metadata is the full JSON body from /record/start
        m = metadata.get("metadata", {})
        task_name = m.get("task_name", "teleop_task")
        task_instruction = m.get("task_language_instruction", "operate robot")
        operator = m.get("operator", "unknown")

        ep_num = self._get_next_episode_num()
        self.episode_dir = self.save_dir / f"{self.episode_prefix}_{ep_num:03d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting episode recording: {self.episode_dir}")

        try:
            self.mcap_writer = TeleopMcapWriter(
                output_path=self.episode_dir,
                task_name=task_name,
                task_language_instruction=task_instruction,
                operator=operator,
                record_rate_hz=self.record_rate,
                record_components=self.record_components,
            )
            self.episode_start_time = time.time()
            self.transitions_in_episode = 0
            self.is_recording = True
            self._record_running = True

            self._record_thread = threading.Thread(
                target=self._record_loop, daemon=True, name="RecordLoop"
            )
            self._record_thread.start()
        except Exception as exc:
            logger.error(f"Failed to initialize TeleopMcapWriter: {exc}")
            self.is_recording = False
            self._record_running = False

    def _get_next_episode_num(self) -> int:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Look for existing episode directories
        existing = [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith(self.episode_prefix)]
        numbers = []
        for ep in existing:
            parts = ep.name.split("_")
            try:
                numbers.append(int(parts[1]))
            except (ValueError, IndexError):
                pass
        return max(numbers, default=-1) + 1

    def _record_loop(self):
        try:
            while self._record_running and self.is_recording:
                ts_ns = time.time_ns()
                self._write_state(ts_ns)
                self._write_action(ts_ns)
                self.transitions_in_episode += 1
                self.total_transitions += 1
                self._rate_limiter.sleep()
        except Exception as exc:
            logger.error(f"Record loop crashed: {exc}", exc_info=True)
            self.is_recording = False

    def _write_state(self, ts_ns: int):
        if not self.robot or not self.mcap_writer:
            return
        obs = self._get_robot_state()
        jp = obs.get("joint_pos", {})
        jv = obs.get("joint_vel", {})
        rc = self.record_components

        def js(comp, vel=False):
            if comp not in jp:
                return None
            qpos = np.array(jp[comp], dtype=np.float32)
            qvel = np.array(jv[comp], dtype=np.float32) if vel and comp in jv else None
            return JointState(qpos=qpos, qvel=qvel)

        state_obj = RobotState(
            left_arm=js("left_arm", vel=True) if rc.get("left_arm") else None,
            right_arm=js("right_arm", vel=True) if rc.get("right_arm") else None,
            left_hand=js("left_hand") if rc.get("left_hand") else None,
            right_hand=js("right_hand") if rc.get("right_hand") else None,
            head=js("head") if rc.get("head") else None,
            torso=js("torso") if rc.get("torso") else None,
            left_wrist_wrench=WrenchState(
                force=np.array(obs["wrench"]["left"]["force"], dtype=np.float32),
                torque=np.array(obs["wrench"]["left"]["torque"], dtype=np.float32),
            )
            if "left" in obs.get("wrench", {}) and rc.get("left_wrist_wrench")
            else None,
            right_wrist_wrench=WrenchState(
                force=np.array(obs["wrench"]["right"]["force"], dtype=np.float32),
                torque=np.array(obs["wrench"]["right"]["torque"], dtype=np.float32),
            )
            if "right" in obs.get("wrench", {}) and rc.get("right_wrist_wrench")
            else None,
            timestamp_ns=ts_ns,
        )
        self.mcap_writer.write_state(state_obj, ts_ns)

    def _write_action(self, ts_ns: int):
        if not self.mcap_writer:
            return
        obs = self._get_robot_state() if self.robot else {}
        jp = obs.get("joint_pos", {})
        resolved: Dict[str, Any] = {}
        joint_comps = [
            "left_arm",
            "right_arm",
            "torso",
            "head",
            "left_hand",
            "right_hand",
        ]
        rc = self.record_components

        with self._action_lock:
            for comp in joint_comps:
                if not rc.get(comp):
                    continue
                if comp in self.current_action:
                    resolved[comp] = self.current_action[comp]
                elif comp in self.last_action:
                    resolved[comp] = self.last_action[comp]
                elif comp in jp:
                    resolved[comp] = {"pos": jp[comp]}
                if comp in resolved:
                    self.last_action[comp] = resolved[comp]
            if "chassis" in self.current_action:
                resolved["chassis"] = self.current_action["chassis"]
                self.last_action["chassis"] = self.current_action["chassis"]
            elif "chassis" in self.last_action:
                resolved["chassis"] = self.last_action["chassis"]

        if not resolved:
            return

        def qpos(comp):
            d = resolved.get(comp)
            return np.array(d["pos"], dtype=np.float32) if d and "pos" in d else None

        chassis = resolved.get("chassis", {})
        action_obj = RobotAction(
            left_arm=JointState(qpos=qpos("left_arm"))
            if qpos("left_arm") is not None and rc.get("left_arm")
            else None,
            right_arm=JointState(qpos=qpos("right_arm"))
            if qpos("right_arm") is not None and rc.get("right_arm")
            else None,
            left_hand=JointState(qpos=qpos("left_hand"))
            if qpos("left_hand") is not None and rc.get("left_hand")
            else None,
            right_hand=JointState(qpos=qpos("right_hand"))
            if qpos("right_hand") is not None and rc.get("right_hand")
            else None,
            head=JointState(qpos=qpos("head"))
            if qpos("head") is not None and rc.get("head")
            else None,
            torso=JointState(qpos=qpos("torso"))
            if qpos("torso") is not None and rc.get("torso")
            else None,
            chassis_vx=np.array([chassis["vx"]], dtype=np.float32)
            if "vx" in chassis
            else None,
            chassis_vy=np.array([chassis["vy"]], dtype=np.float32)
            if "vy" in chassis
            else None,
            chassis_wz=np.array([chassis["wz"]], dtype=np.float32)
            if "wz" in chassis
            else None,
            timestamp_ns=ts_ns,
        )
        self.mcap_writer.write_action(action_obj, ts_ns)

    def _end_episode(self, is_success: bool = True) -> Optional[str]:
        import json

        if not self.is_recording:
            return None
        self._record_running = False
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
        self.is_recording = False
        duration = time.time() - self.episode_start_time
        avg_rate = self.transitions_in_episode / duration if duration > 0 else 0

        if self.mcap_writer:
            self.mcap_writer.close(
                success=is_success,
                extra_metadata={"avg_rate": avg_rate, "is_success": is_success},
            )
            self.mcap_writer = None

        if self.episode_dir:
            meta_path = self.episode_dir / "collection-metadata.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "is_success": is_success,
                        "episode_id": self.episode_dir.name,
                        "duration_seconds": duration,
                        "transitions": self.transitions_in_episode,
                        "avg_rate_hz": avg_rate,
                    },
                    f,
                    indent=2,
                )
            state_path = self.episode_dir / ".state.json"
            with open(state_path, "w") as f:
                json.dump({"status": "complete", "is_success": is_success}, f)
            self._fix_ownership(self.episode_dir)

        episode_id = self.episode_dir.name if self.episode_dir else None
        status = "saved" if is_success else "discarded"
        logger.info(
            f"Episode {status}: {episode_id}  ({self.transitions_in_episode} steps, {duration:.1f}s)"
        )
        if is_success:
            self.total_episodes += 1
        self.transitions_in_episode = 0
        return episode_id

    def _fix_ownership(self, path: Path):
        import subprocess as _sp

        if os.getuid() != 0:
            return
        uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
        if uid and gid:
            try:
                _sp.run(
                    ["chown", "-R", f"{uid}:{gid}", str(path)],
                    check=True,
                    capture_output=True,
                )
            except _sp.CalledProcessError:
                pass

    # =========================================================================
    # Reach (camera-based object approach)
    # =========================================================================

    async def _run_reach(self, request: Request) -> Dict[str, Any]:
        """POST /control/reach implementation (guarded by _reach_in_progress)."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        detect_only = bool(body.get("detect_only", False))

        reach_cfg = self.config.get("reach", {}) or {}
        if not reach_cfg.get("enabled", True):
            raise HTTPException(status_code=400, detail="reach is disabled in config")

        # 1. Live joint state from the robot/joints feedback cache
        joints = self._get_reach_joints()
        if joints is None:
            raise HTTPException(
                status_code=409,
                detail="robot joint states unavailable — is the teleop stack running?",
            )

        # 2. Optional head preset so the object lands inside the center ROI
        _t0 = time.perf_counter()
        _reach_timing: Dict[str, float] = {}
        head_preset = reach_cfg.get("head_preset_q")
        if head_preset and not detect_only:
            self.common_pub.publish({
                "timestamp_ns": time.time_ns(),
                "components": {"head": {"mode": "absolute", "pos": list(head_preset)}},
            })
            await asyncio.sleep(float(reach_cfg.get("head_preset_wait_s", 1.5)))
            joints = self._get_reach_joints() or joints
        _reach_timing["head_preset"] = round(time.perf_counter() - _t0, 3)

        # 3. Head camera frames over WebRTC
        _t1 = time.perf_counter()
        try:
            rgb_frame, depth_frame = await asyncio.to_thread(
                self._capture_reach_frames, reach_cfg
            )
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        _reach_timing["frame_capture"] = round(time.perf_counter() - _t1, 3)

        # 4. Persist frames for the planner subprocess
        reach_dir = Path(__file__).resolve().parents[2] / "reach"
        capture_dir = reach_dir / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = capture_dir / "reach_rgb.png"
        depth_path = capture_dir / "reach_depth.png"
        if not cv2.imwrite(str(rgb_path), rgb_frame):
            raise HTTPException(status_code=500, detail=f"failed to save {rgb_path}")
        depth_scale = float(reach_cfg.get("camera", {}).get("depth_scale", 5.0))
        depth_m = depth_frame[..., 0].astype(np.float32) * (depth_scale / 255.0)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise HTTPException(status_code=500, detail=f"failed to save {depth_path}")
        # Viewer-friendly 8-bit copy (16-bit metric PNGs render almost black in
        # image viewers since the 0-5000mm range is only ~8% of the uint16 scale)
        depth_viz = (depth_m / depth_scale * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(capture_dir / "reach_depth_viz.png"), depth_viz)

        # 5. Ask the long-lived planner daemon (Grounding DINO + Pink IK stay
        #    loaded across requests — only the first call pays the load cost)
        urdf_path = self._resolve_reach_urdf(reach_cfg)
        cam_cfg = reach_cfg.get("camera", {})
        payload = {
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "joints": joints,
            "urdf_path": str(urdf_path),
            "intrinsics": {
                "fx": float(cam_cfg.get("fx", 243.2)),
                "fy": float(cam_cfg.get("fy", 243.2)),
                "cx": float(cam_cfg.get("cx", 320.0)),
                "cy": float(cam_cfg.get("cy", 240.0)),
            },
            "height_above_m": float(reach_cfg.get("height_above_m", 0.40)),
            "detect_only": detect_only,
            "camera_mount_rpy_corr": reach_cfg.get("camera_mount_rpy_corr", [0.0, 0.0, 0.0]),
            "camera_mount_pos_corr": reach_cfg.get("camera_mount_pos_corr", [0.0, 0.0, 0.0]),
            "max_bbox_area": int(reach_cfg.get("max_bbox_area", 40000)),
            "depth_sample": reach_cfg.get("depth_sample", "median"),
            "depth_min_valid_m": float(reach_cfg.get("depth_min_valid_m", 0.30)),
            "depth_max_valid_m": float(reach_cfg.get("depth_max_valid_m", 3.0)),
            "debug_dir": str(reach_dir / "debug"),
        }

        timeout_s = float(reach_cfg.get("planner_timeout_s", 90))
        logger.info("🫸 Reach: querying planner daemon …")
        _t2 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._reach_planner_request, reach_cfg, payload),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            # A hung daemon would block the readline worker forever — kill it;
            # the worker unblocks on EOF and the next request respawns cleanly.
            self._stop_reach_planner()
            raise HTTPException(
                status_code=504, detail=f"reach planner timed out after {timeout_s:.0f}s"
            )
        _reach_timing["planner_daemon"] = round(time.perf_counter() - _t2, 3)

        planner_stages = (result.get("timings") or {}).get("stages", {})
        logger.info(
            f"⏱ Reach timing — backend: {_reach_timing} | "
            f"planner stages: {planner_stages}"
        )
        if not result.get("ok"):
            detail = result.get("error") or "reach planner failed"
            raise HTTPException(status_code=500, detail=f"{detail} (see reach/planner_daemon.log)")
        result["backend_timings"] = _reach_timing
        if not result.get("found"):
            return result
        if detect_only:
            return result

        # 6. Execute: publish absolute arm joint targets on the common pipeline
        cmd = {
            "timestamp_ns": time.time_ns(),
            "components": {
                "left_arm": {"mode": "absolute", "pos": result["target_left_arm"]},
                "right_arm": {"mode": "absolute", "pos": result["target_right_arm"]},
            },
        }
        self.common_pub.publish(cmd)
        logger.info(
            f"🫷 Reach: published arm targets "
            f"(arm={result.get('arm')}, xyz={[round(v, 3) for v in result.get('xyz_base', [])]})"
        )
        return result

    # ------------------------------------------------------------------
    # Planner daemon management (pantos venv process kept alive)
    # ------------------------------------------------------------------

    @staticmethod
    def _reach_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "reach"

    def _resolve_reach_urdf(self, reach_cfg: Dict[str, Any]) -> Path:
        urdf_path = Path(reach_cfg.get(
            "urdf_path",
            "external/dexmate-urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf",
        ))
        if not urdf_path.is_absolute():
            urdf_path = Path(__file__).resolve().parents[4] / urdf_path
        return urdf_path

    _DEFAULT_REACH_PLANNER_PYTHON = "/home/ubuntu/dx/pantos_tele/.venv/bin/python"

    def _reach_planner_python(self, reach_cfg: Dict[str, Any]) -> Path:
        """Interpreter (pantos_tele venv) that runs the reach planner daemon."""
        return Path(reach_cfg.get("planner_python", self._DEFAULT_REACH_PLANNER_PYTHON))

    def _ensure_reach_planner(self, reach_cfg: Dict[str, Any]) -> subprocess.Popen:
        """Return a live planner daemon, spawning it if necessary."""
        with self._reach_planner_lock:
            proc = self._reach_planner_proc
            if proc is not None and proc.poll() is None:
                return proc

            planner_python = self._reach_planner_python(reach_cfg)
            planner_path = Path(reach_cfg.get(
                "planner_script", str(self._reach_dir() / "reach_planner.py")
            ))
            pantos_dir = reach_cfg.get("pantos_dir", "/home/ubuntu/dx/pantos_tele")
            log_path = self._reach_dir() / "planner_daemon.log"
            env = {**os.environ, "PANTOS_TELE_DIR": str(pantos_dir)}

            # stderr → file: never fills a pipe (which would block the daemon)
            log_f = open(log_path, "ab")
            try:
                proc = subprocess.Popen(
                    [str(planner_python), str(planner_path), "--serve"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log_f,
                    text=True,
                    bufsize=1,  # line buffered
                    env=env,
                )
            finally:
                log_f.close()
            self._reach_planner_proc = proc
            logger.info(
                f"🫸 Reach: planner daemon spawned (pid={proc.pid}, log={log_path})"
            )
            return proc

    def _reach_planner_request(self, reach_cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one JSON request line to the daemon and read the response line.

        Blocking — call from a worker thread. Retries once if the daemon died
        since the last request (it is respawned by _ensure_reach_planner).
        """
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            proc = self._ensure_reach_planner(reach_cfg)
            try:
                with self._reach_daemon_io_lock:
                    proc.stdin.write(json.dumps(payload) + "\n")
                    proc.stdin.flush()
                    line = proc.stdout.readline()
                    # Skip any stray non-JSON line (defensive: logs must go to
                    # stderr, but never let noise break the protocol).
                    while line and not line.lstrip().startswith("{"):
                        line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("planner daemon closed the connection")
                return json.loads(line)
            except (OSError, RuntimeError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    f"reach planner daemon request failed (attempt {attempt}): {e}"
                )
                self._stop_reach_planner()
        raise RuntimeError(f"reach planner daemon failed: {last_error}")

    def _stop_reach_planner(self) -> None:
        """Terminate the planner daemon (it also exits on stdin EOF)."""
        with self._reach_planner_lock:
            proc = self._reach_planner_proc
            self._reach_planner_proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            logger.info("🫸 Reach: planner daemon stopped")

    def _warm_reach_planner_async(self) -> None:
        """Preload the daemon (URDF + Grounding DINO) in a background thread."""

        def _warm():
            try:
                reach_cfg = self.config.get("reach", {}) or {}
                if not reach_cfg.get("enabled", True):
                    return
                planner_python = self._reach_planner_python(reach_cfg)
                if not planner_python.exists():
                    # Misconfigured reach.planner_python (pantos_tele venv missing):
                    # warn clearly instead of failing every teleop start. Reach
                    # stays unavailable until the interpreter exists.
                    logger.warning(
                        f"🫸 Reach: planner python not found ({planner_python}); "
                        "warmup skipped — check reach.planner_python / pantos_tele venv"
                    )
                    return
                t0 = time.perf_counter()
                result = self._reach_planner_request(
                    reach_cfg, {"warmup": True, "urdf_path": str(self._resolve_reach_urdf(reach_cfg))}
                )
                logger.info(
                    f"🫸 Reach: planner daemon warmed up in "
                    f"{time.perf_counter() - t0:.1f}s (stages: {result.get('timings', {}).get('stages')})"
                )
            except Exception as e:
                logger.warning(f"🫸 Reach: planner warmup failed (will retry on first use): {e}")

        threading.Thread(target=_warm, daemon=True, name="reach-planner-warmup").start()

    def _get_reach_joints(self) -> Optional[Dict[str, List[float]]]:
        """Current arm/torso/head positions from the robot/joints cache."""
        data = self._latest_robot_joints or {}
        joints = data.get("joints", data)
        out: Dict[str, List[float]] = {}
        for key in ("left_arm", "right_arm", "torso", "head"):
            values = joints.get(key) if isinstance(joints, dict) else None
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                return None
            out[key] = [float(v) for v in values]
        return out

    def _get_reach_receiver(self, ip: str, port: int, name: str) -> WebRTCReceiver:
        """Return a (reused or fresh) WebRTC receiver for ip:port."""
        key = f"{ip}:{port}"
        with self._reach_lock:
            receiver = self._reach_receivers.get(key)
            if receiver is not None and not receiver.running:
                receiver.stop()
                receiver = None
            if receiver is None:
                receiver = WebRTCReceiver(url=f"https://{ip}:{port}/offer", name=name)
                receiver.start()
                self._reach_receivers[key] = receiver
            return receiver

    def _capture_reach_frames(self, reach_cfg: Dict[str, Any]):
        """Grab one head RGB + depth frame pair (blocking; call in a thread)."""
        cam_cfg = reach_cfg.get("camera", {})
        webrtc_cfg = self.config.get("image", {}).get("webrtc", {})
        ip = webrtc_cfg.get("ip", "127.0.0.1")
        rgb_port = int(cam_cfg.get("rgb_port", 60001))
        depth_port = int(cam_cfg.get("depth_port", 60003))
        timeout_s = float(reach_cfg.get("frame_timeout_s", 10.0))

        rgb_rx = self._get_reach_receiver(ip, rgb_port, "head_left_rgb")
        depth_rx = self._get_reach_receiver(ip, depth_port, "head_depth")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rgb_frame = rgb_rx.get_latest_frame()
            depth_frame = depth_rx.get_latest_frame()
            if rgb_frame is not None and depth_frame is not None:
                return rgb_frame, depth_frame
            time.sleep(0.2)
        raise RuntimeError(
            f"no head camera frames from https://{ip}:{rgb_port},{depth_port} "
            f"within {timeout_s:.0f}s — is the image server running?"
        )

    # =========================================================================
    # Run
    # =========================================================================

    def cleanup(self):
        self._state_checker_running = False
        if self._state_checker_thread:
            self._state_checker_thread.join(timeout=2.0)
        if self.state_checker:
            try:
                self.state_checker.cleanup()
            except Exception:
                pass
        if self.is_recording:
            self._end_episode(is_success=False)
        self._stop_teleop_procs()
        if self.robot:
            self.robot.shutdown()
        if hasattr(self._node, "shutdown"):
            self._node.shutdown()

    def run(self):
        import os
        import signal

        certs_dir = os.path.join(os.path.dirname(__file__), "..", "certs")
        cert_file = os.path.join(certs_dir, "cert.pem")
        key_file = os.path.join(certs_dir, "key.pem")
        ssl_kwargs = {}
        if os.path.isfile(cert_file) and os.path.isfile(key_file):
            ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=5006, log_level="warning", **ssl_kwargs
        )
        server = uvicorn.Server(config)

        # uvicorn only handles SIGINT/SIGTERM. On SIGHUP (terminal/SSH session
        # closed) the process dies immediately, skipping the lifespan shutdown
        # hook that stops the teleop subprocesses — route SIGHUP to a graceful
        # exit instead. uvicorn's main loop polls should_exit every tick.
        def _graceful_shutdown(signum, frame):
            server.should_exit = True

        signal.signal(signal.SIGHUP, _graceful_shutdown)
        server.run()

# =============================================================================
# Entry point
# =============================================================================

def main(
    namespace: str = "",
    debug: bool = False,
):
    """OmniTeleop App Backend.

    Launches the FastAPI server. Teleoperation processes are started via
    POST /teleop/start from the frontend, which also passes env vars
    (ROBOT_NAME, ROBOT_CONFIG, ZENOH_CONFIG) and the record_mode flag.

    Args:
        namespace: Optional Zenoh namespace prefix.
        debug: Enable verbose logging.
        rpi_mode: RPi mode — leader scripts (arm_reader, joycon_reader) are
            assumed to already be running on a remote machine and will NOT be
            started locally. Only follower scripts are launched.
    """
    setup_logging(debug)
    logger.info("OmniTeleop App Backend starting")

    app = TeleopApp(namespace=namespace, debug=debug)
    try:
        # app.initialize()
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
    finally:
        app.cleanup()

if __name__ == "__main__":
    tyro.cli(main)
