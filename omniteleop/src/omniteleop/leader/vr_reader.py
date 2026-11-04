
from omniteleop.common.schemas import PoseData, ControllerState

# openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out cert.pem
import asyncio
import time
from dataclasses import asdict
from vuer import Vuer
from vuer.schemas import Sphere, MotionControllers, ImageBackground, WebRTCVideoPlane
import numpy as np
from typing import Optional, Any
from scipy.spatial.transform import Rotation as R

from dexcomm import Node
from dexcomm.codecs import DictDataCodec
from dexcomm.utils import RateLimiter
from omniteleop.common import get_config

from omniteleop.common.debug_display import get_debug_display
from loguru import logger
from omniteleop.leader.communication.openxr_socket_client import transform


def safe_mat_update(prev_mat: np.ndarray, mat: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0)."""
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True


class VrReader():
    DEFAULT_PUBLISH_RATE = 40
    DEFAULT_DEBUG_REFRESH_RATE = 10

    def __init__(
            self,
            namespace: str = "",
            publish_rate: Optional[int] = None,
            topic: Optional[str] = None,
            debug: bool = True,
            host: str = "0.0.0.0",
            port: int = 8012,
            cert: str = "cert.pem",
            key: str = "key.pem",
    ) -> None:

        self.node = Node(name="vr_reader", namespace=namespace)

        # Load configuration
        config = get_config()
        self.config = config

        # Use config values with overrides
        self.publish_rate = publish_rate or config.get_rate(
            "input_rate", self.DEFAULT_PUBLISH_RATE
        )
        self.topic = topic or config.get_topic("vr_pose")

        self.publisher: Optional[Any] = None

        self.display_mode = config.get("input_handlers").get("vr").get("display_mode")
        self.image_server = config.get("image").get("server")
        self.webrtc_url = f"https://{config.get('image').get('webrtc').get('ip')}:{config.get('image').get('webrtc').get('port')[0]}/offer"
        self.image_shape = config.get("image").get("shape", [480, 640])
        self.aspect_ratio = self.image_shape[1] / self.image_shape[0]

        logger.info(
            f"[VrReader] config: topic={self.topic} publish_rate={self.publish_rate} "
            f"display_mode={self.display_mode} image_server={self.image_server} "
            f"webrtc_url={self.webrtc_url} host={host} port={port} cert={cert} key={key}"
        )

        # Rate limiter for precise timing
        self.rate_limiter = RateLimiter(self.publish_rate)

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "ReadXr",
                self.publish_rate,
                refresh_rate=self.DEFAULT_DEBUG_REFRESH_RATE,
            )

        # Vuer 앱 설정
        self.app = Vuer(host=host, cert=cert, key=key, port=port)
        self.app.add_handler("CAMERA_MOVE")(self.handle_camera_move)
        self.app.add_handler("CONTROLLER_MOVE")(self.handle_controller_move)

        # Last valid poses for safe_mat_update, initialized to a visible position
        self.last_head_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1.8],
            [0, 0, 1, -0.2],
            [0, 0, 0, 1]
        ])
        self.last_left_pose = np.array([
            [1, 0, 0, -0.3],
            [0, 1, 0, 1.6],
            [0, 0, 1, -1.5],
            [0, 0, 0, 1],
        ])
        self.last_right_pose = np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 1.6],
            [0, 0, 1, -1.5],
            [0, 0, 0, 1],
        ])

        # Controller states
        self.left_controller_state = ControllerState()
        self.right_controller_state = ControllerState()

    def initialize(self) -> bool:
        try:
            self.publisher = self.node.create_publisher(
                self.topic, encoder=DictDataCodec.encode
            )
            logger.info(f"[VrReader] publisher created on topic={self.topic}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
        return True

    # def handle_camera_data(self, data: dict):
    #     """Handle incoming camera data."""
    #     try:
    #         if "left_rgb" in data:
    #             self.latest_left_rgb = data["left_rgb"]
    #         if "right_rgb" in data:
    #             self.latest_right_rgb = data["right_rgb"]
    #     except Exception as e:
    #         logger.error(f"Error handling camera data: {e}")

    async def handle_camera_move(self, event, session):
        try:
            head_pose_raw = event.value["camera"]["matrix"]
            head_pose_mat = np.array(head_pose_raw).reshape(4, 4, order="F")
            self.last_head_pose, _ = safe_mat_update(self.last_head_pose, head_pose_mat)
        except Exception as e:
            logger.error(f"Error in handle_head_move: {e}")

    async def handle_controller_move(self, event, session):
        try:
            # 1. Update states and matrices first (Fix 1-frame delay bug)
            for side in ["left", "right"]:
                state_data = event.value.get(f"{side}State")
                if state_data:
                    c_state = ControllerState(**state_data)
                    if side == "left":
                        self.left_controller_state = c_state
                    else:
                        self.right_controller_state = c_state

                matrix = event.value.get(side)
                if isinstance(matrix, (list, tuple)) and len(matrix) == 16:
                    new_matrix = np.array(matrix).reshape(4, 4, order="F")
                    if side == "left":
                        self.last_left_pose, _ = safe_mat_update(self.last_left_pose, new_matrix)
                    else:
                        self.last_right_pose, _ = safe_mat_update(self.last_right_pose, new_matrix)

            # 3. Batch and optimize UI updates
            ui_updates = []
            for side in ["left", "right"]:
                input_matrix = self.last_left_pose if side == "left" else self.last_right_pose
                is_active = self.left_controller_state.trigger if side == "left" else self.right_controller_state.trigger
                
                pos = input_matrix[:3, 3]
                euler = R.from_matrix(input_matrix[:3, :3]).as_euler("xyz", degrees=False).tolist()
                color = "#00ff00" if is_active else "#ff0000"
                
                ui_updates.append(
                    Sphere(
                        key=f"target_{side}",
                        args=[0.01, 32, 32],  # Reduced segments for efficiency
                        position=pos.tolist(),
                        rotation=euler,
                        materialType="phong",
                        material=dict(color=color, shininess=80),
                    )
                )
            
            # Send both spheres in a single websocket message
            session.upsert @ ui_updates

        except Exception as e:
            logger.error(f"Error in handle_controller_move: {e}")

    async def main_scene(self, session):
        logger.info("[VrReader] client session connected — vuer transport OK")
        logger.info(
            f"[VrReader] main_scene: display_mode={self.display_mode} image_server={self.image_server}"
        )

        session.upsert(
            MotionControllers(
                stream=True,
                key="controllers",
                left=True,
                right=True,
            ),
            to="bgChildren",
        )
        logger.info("[VrReader] MotionControllers upserted — expecting CONTROLLER_MOVE events")

        if self.display_mode == "immersive":
            if self.image_server == "webrtc":
                session.upsert(
                    WebRTCVideoPlane(
                        src=self.webrtc_url,
                        iceServer=None,
                        iceServers=[],
                        key="video-quad",
                        aspect=self.aspect_ratio,
                        height=7,
                    ),
                    to="bgChildren",
                )
                logger.info(f"[VrReader] WebRTCVideoPlane upserted: {self.webrtc_url}")
            else:
                logger.info("[VrReader] image_server != webrtc; skipping WebRTCVideoPlane")
        else:
            logger.info("[VrReader] display_mode != immersive; skipping video plane")

        # Keep coroutine alive; sphere updates are now driven by controller events
        while True:
            if self.publisher:
                try:
                    pose_data = PoseData(
                        timestamp_ns=time.time_ns(),
                        left={"matrix": transform(self.last_left_pose).tolist()},
                        right={"matrix": transform(self.last_right_pose).tolist()},
                        leftState=self.left_controller_state,
                        rightState=self.right_controller_state
                    )
                    self.publisher.publish(asdict(pose_data))
                except Exception:
                    logger.exception("[VrReader] publish loop iteration failed")
            await asyncio.sleep(1 / self.publish_rate)

    def run(self):
        if not self.initialize():
            logger.error("Failed to initialize VrReader")
            return

        self.app.spawn(start=False)(self.main_scene)
        logger.info("[VrReader] starting vuer server")
        self.app.start()

# 서버 실행 후 퀘스트 3 브라우저에서 http://[PC_IP]:8012 접속
def main():
    vr_reader = VrReader()
    vr_reader.run()


if __name__ == "__main__":
    main()

# export type ControllerData = {
#   left?:       Matrix4Tuple;      // array with length==16
#   right?:      Matrix4Tuple;      // array with length==16
#   leftState?:  ControllerStateType;
#   rightState?: ControllerStateType;
# };
#
# export type ControllerStateType = {
#   trigger:    boolean;
#   squeeze:    boolean;
#   touchpad:   boolean;
#   thumbstick: boolean;
#   aButton:    boolean;
#   bButton:    boolean;
#
#   triggerValue:      number;
#   squeezeValue:      number;
#   touchpadValue:   [ number, number ];   // X and Y values for the touchpad
#   thumbstickValue: [ number, number ]; // X and Y values for the thumbstick
#   aButtonValue:      boolean;
#   bButtonValue:      boolean;
# };