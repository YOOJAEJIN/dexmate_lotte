"""VR controller input handler."""

from typing import Dict, Any, Optional
import numpy as np
import time
from dataclasses import asdict

from dexcomm.codecs import DictDataCodec
from dexcomm import Node
from loguru import logger

from omniteleop.common.schemas import PoseData, ControllerState
from omniteleop.follower.input_handlers.base_handler import (
    BaseInputHandler,
    RobotCommand,
    ArmCommandType,
    CommandMode,
)
from omniteleop.follower.input_handlers.control.meta_con.controller import (
    MetaQuestController,
)

class VRHandler(BaseInputHandler):
    """Input handler for VR controllers.

    Subscribes to VR pose data and transforms it into robot commands
    using end-effector pose targets.
    """

    def __init__(self, config: Dict[str, Any], namespace: str = "", motion_manager=None):
        """Initialize VR handler.

        Args:
            config: Configuration dictionary
            namespace: Optional namespace
            motion_manager: Optional motion manager for state sync
        """
        super().__init__(config, namespace)
        self.motion_manager = motion_manager

        # Extract gripper poses from config
        # hands_config = self.config.get("hands", {})
        # self.left_poses = hands_config.get("left", {}).get("poses", {"open": [0.98], "close": [0.12]})
        # self.right_poses = hands_config.get("right", {}).get("poses", {"open": [0.98], "close": [0.12]})

        # Initialize MetaQuestController for control input handling
        self.vr_controller = MetaQuestController(config)


        # Torso control settings
        self.torso_config = self.config.get("torso", {})
        self.torso_sensitivity = self.torso_config.get("sensitivity", 0.015)
        self.torso_deadzone = self.torso_config.get("deadzone", 0.1)

        # Latest received pose data
        self._latest_pose_data: Optional[PoseData] = None

        # Delta control state
        self._base_poses: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._robot_start_poses: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        # Communication node
        self.node = Node(name="vr_handler", namespace=namespace)

        # Pose topic
        self.pose_topic = config.get("topics", {}).get("vr_pose", "vr/pose")

    def initialize(self) -> bool:
        """Initialize subscribers.

        Returns:
            True if initialization successful
        """
        try:
            self.setup_subscribers()
            self.initialized = True
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize VRHandler: {e}")
            return False

    def setup_subscribers(self) -> None:
        """Set up subscription for VR pose data."""
        self.node.create_subscriber(
            self.pose_topic,
            self._on_pose_received,
            decoder=DictDataCodec.decode,
        )
        logger.info(f"VRHandler subscribed to {self.pose_topic}")

    def _on_pose_received(self, data: Dict[str, Any]) -> None:
        """Callback for VR pose data.

        Args:
            data: Dictionary with left/right arm poses (matrices)
        """
        try:
            if not isinstance(data, dict):
                logger.warning(f"Invalid pose data type: {type(data)}")
                return
            
            # left_state = ControllerState(**data.get('leftState', {}))
            # right_state = ControllerState(**data.get('rightState', {}))

            left_state_data = data.get('leftState', {})
            right_state_data = data.get('rightState', {})
            
            if not isinstance(left_state_data, dict) or not isinstance(right_state_data, dict):
                logger.warning("Invalid state data format")
                return
            
            left_state = ControllerState(**left_state_data)
            right_state = ControllerState(**right_state_data)

            pose_data = PoseData(
                timestamp_ns=data.get('timestamp_ns', time.time_ns()),
                # timestamp_ns=data['timestamp_ns'],
                left=data.get('left', {}),
                right=data.get('right', {}),
                leftState=left_state,
                rightState=right_state
            )
            with self._data_lock:
                self._latest_pose_data = pose_data
        except TypeError as e:
            logger.error(f"Failed to deserialize PoseData: {e}")
        except ValueError as e:
            logger.debug(f"Failed to deserialize PoseData (value error): {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in _vr_pose_callback: {e}")

    def process_inputs(self) -> Optional[RobotCommand]:
        """Process latest VR poses into robot commands.

        Uses MetaQuestController for base, torso, and gripper control.
        Handles arm EE pose control via trigger.

        Returns:
            RobotCommand with component targets
        """
        try:
            with self._data_lock:
                if self._latest_pose_data is None:
                    return None
                pose_data = self._latest_pose_data

            # Convert PoseData to VR data dict format for controller efficiently
            vr_data = {
                "leftState": asdict(pose_data.leftState) if pose_data.leftState else {},
                "rightState": asdict(pose_data.rightState) if pose_data.rightState else {}
            }

            # Process with MetaQuestController to get base, torso, gripper commands
            try:
                controller_command = self.vr_controller.process(vr_data)
            except Exception as e:
                logger.error(f"Error processing VR commands: {e}")
                return None

            command = RobotCommand(timestamp_ns=time.time_ns())
            
            # Forward safety and system flags
            command.safety_flags.emergency_stop = controller_command.estop
            command.safety_flags.exit_requested = controller_command.exit_requested
            
            # If E-stop is active, return immediately with flags
            if command.safety_flags.emergency_stop:
                return command

            # Handle base control from MetaQuestController
            if controller_command.base and controller_command.base.active:
                base_cmd = controller_command.base
                command.input_components["chassis"] = {
                    "vx": base_cmd.vx,
                    "vy": base_cmd.vy,
                    "wz": base_cmd.wz,
                }

            # Handle torso control from MetaQuestController
            if controller_command.torso and controller_command.torso.active:
                torso_cmd = controller_command.torso
                command.input_components["torso"] = {
                    "mode": CommandMode.RELATIVE,
                    "pos": [torso_cmd.delta_x, 0.0, torso_cmd.delta_z],
                }

            # Handle gripper control from MetaQuestController
            if controller_command.hands and controller_command.hands.active:
                hands_cmd = controller_command.hands
                
                if hands_cmd.left_positions:
                    command.input_components["left_hand"] = {
                        "mode": CommandMode.ABSOLUTE,
                        "pos": hands_cmd.left_positions,
                    }
                
                if hands_cmd.right_positions:
                    command.input_components["right_hand"] = {
                        "mode": CommandMode.ABSOLUTE,
                        "pos": hands_cmd.right_positions,
                    }

            # Handle arm (EE pose) control via trigger.
            #
            # Delta-IK pattern (mirrors the upstream paddle leader / cartesian solver):
            # On trigger press we capture the VR controller pose as the origin and the
            # robot's current EE pose (via FK) as the reference. Each tick we emit the
            # VR *delta* (delta_xyz, delta_R) relative to that origin, and the
            # command_processor applies it to the reference FK pose before solving IK:
            #     target_pos = ref_pos + delta_xyz
            #     target_rot = delta_R @ ref_rot
            # Position is added (not matrix-multiplied through rotation) so the robot
            # frame's translation axis is not skewed by the EE orientation.
            current_robot_poses = {}
            if self.motion_manager:
                try:
                    current_robot_poses = self.motion_manager.fk(["L_ee", "R_ee"])
                except Exception as e:
                    logger.debug(f"Error getting FK: {e}")

            for side, ee_frame in [("left", "L_ee"), ("right", "R_ee")]:
                try:
                    controller_pose = getattr(pose_data, side, None)
                    controller_state = getattr(pose_data, f"{side}State", None)

                    if controller_pose and controller_state:
                        # Check trigger state for arm control
                        is_active = controller_state.trigger
                        matrix_data = controller_pose.get("matrix")

                        if matrix_data and is_active:
                            current_vr_matrix = np.array(matrix_data)

                            # Set base pose on trigger press
                            if self._base_poses[side] is None:
                                self._base_poses[side] = current_vr_matrix.copy()
                                if ee_frame in current_robot_poses:
                                    # FK returns pin.SE3; snapshot as a plain 4x4
                                    # ndarray so downstream matrix math is pure numpy.
                                    self._robot_start_poses[side] = np.array(
                                        current_robot_poses[ee_frame].np, copy=True
                                    )
                                else:
                                    self._robot_start_poses[side] = np.eye(4)
                                logger.info(f"VR {side} arm control started (trigger pressed)")

                            # Relative VR motion since the origin was captured.
                            # Matches the upstream paddle leader (InterventionTracker):
                            #   delta_xyz = wrist_pos - origin_pos   (world frame)
                            #   delta_R   = wrist_rot @ origin_rot.T
                            # Using a world-frame position difference (not the
                            # translation of cur @ inv(origin)) keeps delta_xyz
                            # axis-aligned with the robot base frame regardless of
                            # the controller's orientation at clutch-in.
                            origin = self._base_poses[side]
                            delta_R = current_vr_matrix[:3, :3] @ origin[:3, :3].T
                            delta_xyz = current_vr_matrix[:3, 3] - origin[:3, 3]

                            command.input_components[f"{side}_arm"] = {
                                "command_type": ArmCommandType.EE_POSE,
                                "pose": self._robot_start_poses[side],
                                "delta_xyz": delta_xyz,
                                "delta_R": delta_R,
                            }
                        else:
                            # Reset on trigger release
                            if self._base_poses[side] is not None:
                                logger.info(f"VR {side} arm control stopped (trigger released)")
                            self._base_poses[side] = None
                            self._robot_start_poses[side] = None
                except Exception as e:
                    logger.debug(f"Error processing {side} arm control: {e}")
                    self._base_poses[side] = None
                    self._robot_start_poses[side] = None

            # Return command if we have any components
            if command.input_components:
                command.debug_info = {
                    "active_modes": self.vr_controller.get_control_mode(vr_data),
                }
                return command
                
            return None
        
        except Exception as e:
            logger.debug(f"Unexpected error in process_inputs: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up resources."""
        self.running = False
        self.node.shutdown()

    def needs_motion_manager(self) -> bool:
        """Indicate that this handler can use a motion manager."""
        return True
