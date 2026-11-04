#!/usr/bin/env python3
"""Command processor with integrated safety validation.

This module processes teleoperation inputs from various sources,
applies safety checks, and publishes validated commands for the robot controller.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional

import numpy as np

from dexcomm import Node
from omniteleop import LIB_PATH
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.log_utils import suppress_loguru_module
from omniteleop.follower.input_handlers.base_handler import CommandMode
from dexcomm.codecs import DictDataCodec
from loguru import logger
from dexmotion.motion_manager import MotionManager
from dexmotion.configs.ik import LocalPinkIKConfig, IKDampingWeightsConfig
from dexcomm import RateLimiter

# Import component processors
from omniteleop.follower.component_processors import (
    ArmProcessor,
    HandProcessor,
    TorsoProcessor,
    HeadProcessor,
    ChassisProcessor,
    SafetyValidator,
)

# Import input handlers
from omniteleop.follower.input_handlers.base_handler import (
    BaseInputHandler,
    RobotCommand,
    ArmCommandType,
)
from omniteleop.follower.input_handlers.exo_joycon_handler import ExoJoyconHandler as StandardExoJoyconHandler
from omniteleop.follower.input_handlers.robotis_exo_joycon_handle import ExoJoyconHandler as RobotisExoJoyconHandler
from omniteleop.follower.input_handlers.vr_handler import VRHandler
from omniteleop.follower.input_handlers.common_handler import CommonHandler
from omniteleop.follower.input_handlers.xsens_handler import XsensHandler


from dexbot_utils import RobotInfo

class CommandProcessor():
    """Processes teleoperation commands with integrated safety validation.

    This class receives commands from various teleoperation input sources,
    applies safety checks including collision detection and joint limits,
    and publishes validated commands for robot control.

    Uses modular component processors for handling different robot parts.

    Attributes:
        config: Configuration dictionary loaded from YAML.
        teleop_mode: Current teleoperation mode (e.g., 'exo_joycon', 'vr').
        motion_manager: MotionManager instance for kinematics and collision checking.
        input_handler: Handler for processing teleoperation inputs.
        processors: Dictionary of component processors.
        safety_validator: Safety validation instance.
    """

    # A rotation jog is considered continuing while identical ticks arrive
    # within this window (frontend hold-repeat sends the same request every
    # 100 ms and sends no stop message on release).
    ROT_JOG_ANCHOR_TIMEOUT_S = 0.3

    def __init__(
        self,
        namespace: str = "",
        debug: bool = False,
        visualize: bool = False,
        sim: bool = False,
        initial_joint_positions: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None,
    ):
        """Initialize the command processor with configuration and handlers.

        Args:
            namespace: Namespace for Zenoh topics.
            debug: Whether to enable debug display output.
            visualize: Whether to enable visualization in simulator.
            sim: Whether to run in simulation mode (skips safety validation).
            initial_joint_positions: Initial joint positions for motion manager.
            config_name: Name of config file (without .yaml extension).

        Raises:
            RuntimeError: If input handler initialization fails.
        """
        # Initialize Node with namespace support
        self.node = Node(name="command_processor", namespace=namespace)
        self.namespace = namespace
        self.sim = sim

        self.robot_info = RobotInfo()
        self.has_torso = self.robot_info.has_torso
        self.has_base = self.robot_info.has_chassis

        # Load configuration
        config_path = None
        if config_name is not None:
            config_path = LIB_PATH / "configs" / f"{config_name}.yaml"
        self.config = get_config(config_path)
        self.running = False
        self.teleop_mode = self.config.get("teleop_mode", "exo_joycon")
        self.rate = self.config.get_rate("command_rate", 40)

        # Initialize safety parameters
        self.safety_config = self.config.get("safety")
        self.bypass_real_robot = self.safety_config.get("bypass_real_robot", False)

        # Initialize motion manager with IK configuration.
        # Elbow-constrained IK (ported from omniteleop-main ElbowConstrainedIKController)
        # pins elbow orientation to resolve the 7-DOF arm null-space ambiguity that
        # otherwise lets the mid-arm joint spin freely during EE pose tracking.
        # Always enabled: elbow frames are included as orientation-only IK targets.
        joints_to_lock = ["BASE"]
        ik_config = self._build_ik_config()

        self._elbow_frames = ["L_arm_l4", "R_arm_l4"]
        target_frames = ["L_ee", "R_ee"] + self._elbow_frames

        mm_kwargs = dict(
            visualizer_type="sapien",
            init_visualizer=visualize,
            joint_regions_to_lock=joints_to_lock,
            initial_joint_configuration_dict=initial_joint_positions,
            custom_local_ik_config=ik_config,
            target_frames=target_frames,
        )
        self.motion_manager = MotionManager(**mm_kwargs)

        # Configure elbow FrameTasks as orientation-only (position_cost=0,
        # orientation_cost=50), mirroring the upstream cartesian solver.
        self._configure_elbow_orientation_only()

        # Get init positions based on robot type
        init_pos_config = self.config.get("init_pos", {})

        # Store init positions for alignment check
        self._init_pos_left_arm = init_pos_config.get("left_arm", [0.0] * 7)
        self._init_pos_right_arm = init_pos_config.get("right_arm", [0.0] * 7)

        # Max allowed difference (rad) between exo and init_pos before motion starts.
        self._align_threshold = 1.0

        # Initialize arms with robot-type-specific positions
        self.motion_manager.left_arm.set_joint_pos(
            init_pos_config.get("left_arm", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.motion_manager.right_arm.set_joint_pos(
            init_pos_config.get("right_arm", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

        # Only initialize torso if robot has torso
        if self.has_torso:
            self.motion_manager.torso.set_joint_pos(
                init_pos_config.get("torso", [0.0, 0.0, 0.0])
            )

        # Initialize component processors
        self._initialize_processors()

        # Initialize safety validator
        self.safety_validator = SafetyValidator(self.config, self.motion_manager)

        # Create input handler
        self.input_handler: Optional[BaseInputHandler] = self._create_input_handler()

        # Initialize common handler for dashboard commands
        self.common_handler = CommonHandler(self.config.get("input_handlers", {}).get("common", {}), self.namespace)

        # Idle resync (pairs with robot_controller's hold_timeout_s). Once a
        # component has not been commanded for longer than hold_timeout_s the
        # follower stops holding it, so another controller may have moved it.
        # Before applying the next relative command, re-sync the processor's
        # virtual state from robot feedback so the jog starts from where the
        # robot actually is instead of snapping back to a stale virtual pose.
        _raw_hold = self.config.get("hold_timeout_s")
        try:
            self._idle_resync_s: Optional[float] = (
                float(_raw_hold) if _raw_hold is not None and float(_raw_hold) > 0 else None
            )
        except (TypeError, ValueError):
            self._idle_resync_s = None
        self._last_output_time: Dict[str, float] = {}

        # Thread-safe storage for robot joint feedback
        self.latest_robot_joints: Optional[Dict[str, Any]] = None
        self._robot_joints_lock = threading.RLock()

        # Control flags
        self.exit_after_publish = False
        self._robot_joint_data_initialized = self.bypass_real_robot
        self._motion_control_started = self.bypass_real_robot
        self._align_checked = self.bypass_real_robot
        self._last_safe_command: Optional[RobotCommand] = None

        # Rotation-jog position anchors per EE frame ("L_ee"/"R_ee"):
        # {"pos": np.ndarray(3), "delta_R": np.ndarray(3,3), "time": float}.
        # See _resolve_dashboard_jog_reference for why rotation-only jogs
        # pin their position reference instead of re-sampling FK each tick.
        self._rot_jog_anchor: Dict[str, Dict[str, Any]] = {}

        # Realignment tracking: when estop is released, re-check proximity
        # to current robot joint positions before resuming motion.
        self._estop_was_active = False
        self._pending_realign = False
        self._last_align_log: Dict[str, float] = {}  # suppress repeated align warnings

        # Set up subscribers and publishers through Dexcomm Node
        self._setup_communication()

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if self.debug:
            rate = self.config.get_rate("command_rate", 40)
            self._debug_display = get_debug_display(
                "CmdSafety", rate, refresh_rate=10, precision=3
            )

        logger.info(f"CommandProcessor initialized with {self.teleop_mode} mode")

    def _build_ik_config(self) -> Optional[LocalPinkIKConfig]:
        """Build optimized IK configuration for real-time control.
        
        Creates LocalPinkIKConfig with carefully tuned damping weights
        for stable inverse kinematics solving.
        
        Returns:
            LocalPinkIKConfig instance or None if setup fails.
        """
        try:
            return LocalPinkIKConfig(
                solver_name="local_pink",
                target_frames=["L_ee", "R_ee"],
                dt=1.0 / max(1, self.config.get_rate("command_rate", 40)),
                avoid_self_collisions=False,
                collision_pairs=0,
                collision_margin=0.01,
                qp_solver="quadprog",
                safety_buffer=0.2,
                damping_weights=IKDampingWeightsConfig(
                    default=1e-3,
                    override={
                        # Torso - high damping for stability
                        "torso_j1": 30000.0,
                        "torso_j2": 30000.0,
                        "torso_j3": 30000.0,
                        # Arms - tuned for smooth reaching
                        "R_arm_j1": 50,
                        "L_arm_j1": 50,
                        "R_arm_j2": 75,
                        "L_arm_j2": 75,
                        "R_arm_j3": 50.0,
                        "L_arm_j3": 50.0,
                        "R_arm_j4": 50.0,
                        "L_arm_j4": 50.0,
                        "R_arm_j5": 120.0,
                        "L_arm_j5": 120.0,
                        "R_arm_j6": 120.0,
                        "L_arm_j6": 120.0,
                        "R_arm_j7": 120.0,
                        "L_arm_j7": 120.0,
                    },
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to create IK config: {e}")
            return None

    def _initialize_processors(self) -> None:
        """Initialize component processors based on robot capabilities."""
        self.processors = {}

        # Arms - always present
        self.processors["left_arm"] = ArmProcessor(
            "left", self.config, self.motion_manager, self.robot_info, self.teleop_mode, rate = self.rate
        )
        self.processors["right_arm"] = ArmProcessor(
            "right", self.config, self.motion_manager, self.robot_info, self.teleop_mode, rate = self.rate
        )

        # Hands - check if component exists
        if self.robot_info.has_component("left_hand"):
            self.processors["left_hand"] = HandProcessor(
                "left",
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )
        if self.robot_info.has_component("right_hand"):
            self.processors["right_hand"] = HandProcessor(
                "right",
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )

        # Single-instance components - check if they exist on robot
        if self.robot_info.has_torso:
            self.processors["torso"] = TorsoProcessor(
                None,
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )
        if self.robot_info.has_head:
            self.processors["head"] = HeadProcessor(
                None,
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )
        if self.robot_info.has_chassis:
            self.processors["chassis"] = ChassisProcessor(
                None,
                self.config,
                self.motion_manager,
                self.robot_info,
                self.teleop_mode,
            )

        logger.debug(f"Initialized processors: {list(self.processors.keys())}")

    def _create_input_handler(self) -> Optional[BaseInputHandler]:
        """Create input handler based on configured teleoperation mode.

        Factory method that creates the appropriate input handler based on the teleop_mode setting.

        Returns:
            BaseInputHandler instance or None if mode is unsupported.

        Note:
            Supported modes are 'exo_joycon', 'robotis_exo', and 'vr'.
        """
        # Get handler configuration from unified structure
        handler_config = self.config.get("input_handlers", {}).get(self.teleop_mode, {}).copy()
        
        # Inject common configuration for shared settings
        handler_config["common"] = self.config.get("input_handlers", {}).get("common", {})


        if self.teleop_mode == "exo_joycon":
            handler_config["topics"] = {
                "exo_joints": self.config.get_topic("exo_joints"),
                "exo_joycon": self.config.get_topic("exo_joycon"),
            }
            handler = StandardExoJoyconHandler(handler_config, self.namespace)
            logger.info("Created ExoJoyconHandler for exoskeleton + JoyCon input")
            return handler
        elif self.teleop_mode == "robotis_exo":
            handler_config["topics"] = {
                "exo_joints": self.config.get_topic("exo_joints"),
                "exo_joycon": self.config.get_topic("exo_joycon"),
            }
            handler = RobotisExoJoyconHandler(handler_config, self.namespace)
            logger.info("Created RobotisExoJoyconHandler for Robotis exoskeleton input")
            return handler
        elif self.teleop_mode == "vr":
            handler = VRHandler(handler_config, self.namespace, self.motion_manager)
            logger.info("Created VRHandler for VR input")
            return handler
        elif self.teleop_mode == "xsens":
            handler = XsensHandler(handler_config, self.namespace, self.motion_manager)
            logger.info("Created XsensHandler for Xsens input")
            return handler

        else:
            logger.error(f"Unsupported teleop mode: {self.teleop_mode}")
            return None

    def _setup_communication(self):
        """Configure Zenoh publishers and subscribers for robot communication.

        Sets up:
        - Subscriber for robot joint feedback
        - Publisher for validated robot commands
        - Input handler initialization

        Raises:
            RuntimeError: If input handler initialization fails.
        """
        # Initialize input handler
        if self.input_handler:
            if not self.input_handler.initialize():
                logger.error("Failed to initialize input handler")
                raise RuntimeError(
                    f"Input handler initialization failed for {self.teleop_mode}"
                )

        # Initialize common handler
        if not self.common_handler.initialize():
            logger.error("Failed to initialize common handler")
            raise RuntimeError("Common handler initialization failed")

        # Subscribe to robot joint feedback
        robot_joints_topic = self.config.get_topic("robot_joints")
        self.robot_joints_sub = self.node.create_subscriber(
            robot_joints_topic,
            self._on_robot_joints_received,
            decoder=DictDataCodec.decode,
        )
        commands_topic = self.config.get_topic("robot_commands")
        self.safe_command_pub = self.node.create_publisher(
            commands_topic,
            encoder=DictDataCodec.encode,
        )

    def _on_robot_joints_received(self, robot_joints_data: Dict[str, Any]) -> None:
        """Store latest robot joint positions and velocities from feedback.

        Thread-safe callback that stores robot joint data for synchronization
        and safety checking.

        Args:
            robot_joints_data: Dictionary containing timestamp, joint positions, and velocities.
        """
        try:
            with self._robot_joints_lock:
                self.latest_robot_joints = robot_joints_data
            if self.latest_robot_joints:
                self._robot_joint_data_initialized = True
        except Exception as e:
            logger.error(f"Error storing robot joints feedback: {e}")

    def process_commands(self) -> Optional[RobotCommand]:
        """Process and validate teleoperation commands.

        Main processing pipeline that:
        1. Gets commands from primary input handler and common handler
        2. Merges commands (primary handler takes precedence)
        3. Processes components via modular processors
        4. Applies safety validation
        5. Returns validated command or last safe command

        Returns:
            Validated RobotCommand or None if no input available.

        Note:
            Falls back to last safe command if current command fails validation.
        """
        # Get command from primary input handler
        if not self.input_handler:
            logger.error("No input handler available")
            primary_command = None
        else:
            primary_command = self.input_handler.process_inputs()

        # Get command from common handler
        common_command = self.common_handler.process_inputs()

        # Merge commands
        command = self._merge_commands(primary_command, common_command)

        if not command:
            return None

        # Process components using modular processors
        self._process_components(command)

        if not self._motion_control_started:
            command = RobotCommand(timestamp_ns=time.time_ns())
            command.safety_flags.emergency_stop = True
            return command

        # Apply safety validation
        if not self.sim:
            self.safety_validator.validate(command)

        # Return validated command or fall back to last safe command
        if command.valid:
            self._last_safe_command = command
            return command
        else:
            # Use last known safe command to maintain stability
            return self._last_safe_command

    def _merge_commands(self, primary: Optional[RobotCommand], common: Optional[RobotCommand]) -> Optional[RobotCommand]:
        """Merge primary and common commands.

        Primary command takes precedence for overlapping components.

        Args:
            primary: Command from primary input handler
            common: Command from common handler

        Returns:
            Merged RobotCommand or None if both are None
        """
        if primary is None:
            return common
        if common is None:
            return primary

        # Use primary command as base
        merged = primary

        # Add components from common if not present in primary
        for comp_name, comp_data in common.input_components.items():
            if comp_name not in merged.input_components:
                merged.input_components[comp_name] = comp_data

        return merged

    def _process_components(self, command: RobotCommand) -> None:
        """Process all components using modular processors.

        Routes input components to appropriate processors. Handles:
        - Emergency stops and exit requests
        - Motion control startup
        - Component-specific processing via processors
        - Special case: dual-arm EE pose IK

        Args:
            command: RobotCommand to process (modified in-place)

        Side Effects:
            - Sets exit_after_publish flag if exit requested
            - Starts motion control when emergency stop is released
            - Syncs motion manager to robot state on startup
            - Modifies command.output_components with processed data
        """
        # Check for exit request flag
        if command.safety_flags.exit_requested:
            logger.critical("Exit requested - will shutdown after publishing")
            self.exit_after_publish = True
            # Ensure emergency stop is also active when exiting
            command.safety_flags.emergency_stop = True

        # Check for emergency stop and motion control start
        if command.safety_flags.emergency_stop:
            # Emergency stop is active, don't start motion control
            pass
        elif not self._motion_control_started:
            # Alignment is only checked on the very first start.
            # After that (e.g. estop toggle), motion control resumes immediately.
            if self._align_checked or self._is_exo_aligned_with_init_pos(command):
                self._motion_control_started = True
                self._align_checked = True
                self._sync_motion_manager_to_robot_state()
                logger.success("Exo joints aligned — motion control started.")
            else:
                command.safety_flags.emergency_stop = True

        # Special case: Dual-arm EE pose when BOTH arms have EE_POSE
        if self._should_process_dual_arm_ee_pose(command):
            self._process_dual_arm_ee_pose(command)
            return

        # Special case: Single-arm EE pose when only ONE arm has EE_POSE
        if self._should_process_single_arm_ee_pose(command):
            self._process_single_arm_ee_pose(command)
            return

        # Process each component independently via processors
        for component_name in list(command.input_components.keys()):
            input_data = command.input_components.get(component_name)
            processor = self.processors.get(component_name)

            # common commnad: hand process가 없는 경우 처리
            if component_name in ["left_hand", "right_hand"] and processor is None:
                command.output_components[component_name] = input_data
                command.input_components.pop(component_name)
                continue

            if processor is None:
                # Unknown component - remove it
                command.input_components.pop(component_name)
                continue

            if not self.robot_info.has_component(component_name):
                # Component disabled for this robot type
                command.input_components.pop(component_name)
                continue

            # Process component and remove from input_components
            command.input_components.pop(component_name)
            if component_name == "torso":
                self._resync_if_idle(component_name, processor)
            processor.process(input_data, command)

    def _should_process_dual_arm_ee_pose(self, command: RobotCommand) -> bool:
        """Check if command requires dual-arm EE pose processing.

        Only process dual-arm IK when BOTH arms have EE_POSE.
        Allows true simultaneous control when both triggers are pressed.

        Args:
            command: RobotCommand to check

        Returns:
            True if both arms have EE_POSE command type
        """
        if 'left_arm' not in command.input_components:
            return False
        if 'right_arm' not in command.input_components:
            return False

        left_data = command.input_components["left_arm"]
        right_data = command.input_components["right_arm"]

        left_type = left_data.get("command_type", ArmCommandType.JOINT)
        right_type = right_data.get("command_type", ArmCommandType.JOINT)

        # Only dual-arm IK when BOTH have EE_POSE
        return (
            left_type == ArmCommandType.EE_POSE and right_type == ArmCommandType.EE_POSE
        )

    def _should_process_single_arm_ee_pose(self, command: RobotCommand) -> bool:
        """Check if command requires single-arm EE pose processing.

        Process single-arm IK when ONLY ONE arm has EE_POSE.
        Allows independent arm control without coupling issues.

        Args:
            command: RobotCommand to check

        Returns:
            True if exactly one arm has EE_POSE command type
        """
        left_has_ee_pose = False
        right_has_ee_pose = False

        if 'left_arm' in command.input_components:
            left_data = command.input_components['left_arm']
            left_type = left_data.get('command_type', ArmCommandType.JOINT)
            left_has_ee_pose = (left_type == ArmCommandType.EE_POSE)

        if 'right_arm' in command.input_components:
            right_data = command.input_components['right_arm']
            right_type = right_data.get('command_type', ArmCommandType.JOINT)
            right_has_ee_pose = (right_type == ArmCommandType.EE_POSE)

        # Only single-arm IK when EXACTLY ONE has EE_POSE
        return (left_has_ee_pose and not right_has_ee_pose) or (right_has_ee_pose and not left_has_ee_pose)

    def _process_dual_arm_ee_pose(self, command: RobotCommand) -> None:
        """Process dual-arm end-effector pose commands with IK.

        Solves IK for both arms simultaneously and applies safety limiting.
        Only called when BOTH arms have EE_POSE targets.

        Uses the delta-IK pattern: each arm's input carries the reference pose
        captured when tracking started (``pose``) plus the relative VR motion
        (``delta_xyz``, ``delta_R``). The target pose is assembled as:
            target_pos = ref_pos + delta_xyz
            target_rot = delta_R @ ref_rot
        Position is added directly so the robot-frame translation axis is not
        skewed by the EE orientation. Falls back to treating ``pose`` as an
        absolute target when no delta fields are present (backward compatible).

        Dashboard cartesian jog commands carry only deltas without a
        reference ``pose``: the reference is resolved per jog type (see
        _resolve_dashboard_jog_reference). Translation jogs re-sample the
        current FK pose of each EE every tick ("current pose + delta"
        semantics); rotation-only jogs pin the position to an anchor
        captured at jog start so the per-tick IK residual cannot
        accumulate into drift.

        Args:
            command: RobotCommand with EE pose targets for both arms
        """
        left_arm_data = command.input_components.pop('left_arm')
        right_arm_data = command.input_components.pop('right_arm')

        # Resolve missing reference poses (dashboard jog) from current FK.
        for arm_data, ee_frame in (
            (left_arm_data, 'L_ee'),
            (right_arm_data, 'R_ee'),
        ):
            if 'pose' in arm_data:
                continue
            if not self._resolve_dashboard_jog_reference(arm_data, ee_frame):
                return

        # Build per-arm target poses from reference + delta (delta-IK pattern).
        target_poses = {
            'L_ee': self._build_ee_target_pose(left_arm_data),
            'R_ee': self._build_ee_target_pose(right_arm_data),
        }

        # Feed elbow orientation targets (pinned to current FK) into a single
        # solve_ik call so the 7-DOF null-space is resolved alongside the EE
        # targets, preventing the mid-arm joint from spinning freely.
        with suppress_loguru_module("dexmotion", enabled=True):
            target_poses.update(self._build_elbow_target_poses())
            qpos_dict, is_collision, _ = (
                self.motion_manager.local_ik_solver.solve_ik(target_poses)
            )
            solution = qpos_dict
            within_limits = True  # local solver does not report limits

        if not solution or not within_limits or is_collision:
            return

        # Get arm processors for safety limiting
        left_processor = self.processors["left_arm"]
        right_processor = self.processors["right_arm"]

        # Extract and limit joint positions
        left_positions = [solution[f"L_arm_j{i + 1}"] for i in range(7)]
        right_positions = [solution[f"R_arm_j{i + 1}"] for i in range(7)]

        safe_left = left_processor.limit_joint_step(left_positions)
        safe_right = right_processor.limit_joint_step(right_positions)

        # Update motion manager
        left_processor.apply_positions(safe_left)
        right_processor.apply_positions(safe_right)

        # Update command
        command.output_components["left_arm"] = {
            "pos": safe_left.tolist(),
            "mode": CommandMode.ABSOLUTE.value,
        }
        command.output_components["right_arm"] = {
            "pos": safe_right.tolist(),
            "mode": CommandMode.ABSOLUTE.value,
        }

    def _resolve_dashboard_jog_reference(self, arm_data: Dict[str, Any], ee_frame: str) -> bool:
        """Resolve the reference pose for a dashboard jog command (no 'pose').

        Translation jogs use the current FK pose as the reference ("current
        pose + delta"). Rotation-only jogs keep the rotation reference on
        current FK (so the rotation accumulates tick by tick) but pin the
        position reference to an anchor captured at jog start: solve_ik only
        approximates the target (~1-2 mm residual per tick), and re-sampling
        the position each tick folds that residual into the next tick's
        reference, accumulating visible drift over a long hold.

        A rotation jog counts as continuing while identical ticks keep
        arriving within ROT_JOG_ANCHOR_TIMEOUT_S, which matches the frontend
        hold-repeat pattern (same request every 100 ms, no stop message).

        Args:
            arm_data: Arm input data with delta_xyz and/or delta_R.
            ee_frame: "L_ee" or "R_ee".

        Returns:
            True when arm_data['pose'] was set; False when no delta is
            present or FK failed (caller should skip the command).
        """
        delta_xyz = arm_data.get('delta_xyz')
        delta_R = arm_data.get('delta_R')
        if delta_xyz is None and delta_R is None:
            return False
        try:
            current = np.array(
                self.motion_manager.fk([ee_frame])[ee_frame].np, dtype=float
            )
        except Exception as e:
            # Skip the command rather than falling back to identity,
            # which would drive the EE toward the world origin.
            logger.debug(f"FK failed for {ee_frame} jog reference: {e}")
            return False

        rot_only = delta_R is not None and (
            delta_xyz is None or not np.any(np.asarray(delta_xyz, dtype=float))
        )
        if not rot_only:
            # Translation intent: drop any stale anchor so the next rotation
            # jog re-anchors at the post-translation position.
            self._rot_jog_anchor.pop(ee_frame, None)
            arm_data['pose'] = current
            return True

        now = time.monotonic()
        anchor = self._rot_jog_anchor.get(ee_frame)
        same_jog = (
            anchor is not None
            and (now - anchor['time']) < self.ROT_JOG_ANCHOR_TIMEOUT_S
            and np.allclose(anchor['delta_R'], delta_R, atol=1e-9)
            and anchor.get('rot_frame') == arm_data.get('rot_frame')
        )
        if not same_jog:
            anchor = {
                'pos': current[:3, 3].copy(),
                'delta_R': np.asarray(delta_R, dtype=float),
                'rot_frame': arm_data.get('rot_frame'),
                'time': now,
            }
            self._rot_jog_anchor[ee_frame] = anchor
        anchor['time'] = now

        reference = current.copy()
        reference[:3, 3] = anchor['pos']
        arm_data['pose'] = reference
        return True

    @staticmethod
    def _build_ee_target_pose(arm_data: Dict[str, Any]) -> np.ndarray:
        """Build an EE target pose from reference pose + VR delta.

        Delta-IK pattern (mirrors the upstream cartesian solver): the reference
        pose is the FK pose captured when tracking started; ``delta_xyz`` and
        ``delta_R`` are the relative VR motion since then. The target is:
            target_pos = ref_pos + delta_xyz
            target_rot = delta_R @ ref_rot   (rot_frame "world", default)
            target_rot = ref_rot @ delta_R   (rot_frame "ee")
        ``rot_frame`` selects which frame ``delta_R`` is expressed in: the
        robot base frame (also how VR deltas are computed) or the EE's own
        frame (dashboard rotation jog about the gripper's axes). Position is
        added directly in either frame.
        When delta fields are absent the ``pose`` is treated as an absolute
        target (backward compatible with non-VR callers).

        Args:
            arm_data: Arm input data with at least a ``pose`` key.

        Returns:
            4x4 homogeneous target pose in the robot base frame.
        """
        pose = np.array(arm_data.get('pose', np.eye(4)), dtype=float)
        delta_xyz = arm_data.get('delta_xyz')
        delta_R = arm_data.get('delta_R')

        if delta_xyz is None and delta_R is None:
            return pose

        target = pose.copy()
        if delta_R is not None:
            delta_R = np.asarray(delta_R, dtype=float)
            if arm_data.get('rot_frame') == 'ee':
                target[:3, :3] = target[:3, :3] @ delta_R
            else:
                target[:3, :3] = delta_R @ target[:3, :3]
        if delta_xyz is not None:
            delta_xyz = np.asarray(delta_xyz, dtype=float)
            target[:3, 3] += delta_xyz
        return target

    def _hold_current_ee_pose(self, ee_frame: str) -> np.ndarray:
        """Return the current EE pose from FK to hold an inactive arm in place.

        Solves FK against the motion manager's current joint state so the arm
        stays where it is instead of being driven to the world origin.

        Args:
            ee_frame: "L_ee" or "R_ee".

        Returns:
            4x4 homogeneous pose; falls back to identity if FK fails.
        """
        try:
            ee_poses = self.motion_manager.fk([ee_frame])
            return np.array(ee_poses[ee_frame].np, dtype=float)
        except Exception as e:
            logger.debug(f"FK failed for {ee_frame} hold pose: {e}")
            return np.eye(4)

    def _configure_elbow_orientation_only(self) -> None:
        """Set elbow FrameTasks to orientation-only (position_cost=0, orientation_cost=50).

        Ported from omniteleop-main ElbowConstrainedIKController: pinning the
        elbow orientation resolves the 7-DOF arm null-space ambiguity (the
        "redundancy circle" the elbow can spin around given only an EE pose),
        which is the root cause of the mid-arm joint spinning during VR control.
        Position is left unconstrained so the elbow can still translate to
        satisfy the EE target; only its direction is held.
        """
        solver = self.motion_manager.local_ik_solver
        for frame_name in self._elbow_frames:
            task = solver.task_dict[frame_name]
            task.position_cost = 0.0
            task.orientation_cost = 50.0
        logger.info(f"Elbow orientation-only IK constraints enabled: {self._elbow_frames}")

    def _build_elbow_target_poses(self) -> Dict[str, np.ndarray]:
        """Build elbow target poses by pinning the current FK orientation.

        Mirrors the fallback branch of the upstream elbow-constrained solver:
        each elbow target keeps its current FK position (irrelevant since
        position_cost=0) and current FK orientation, so the solver holds the
        elbow direction where it is and the null-space cannot drift.

        Returns:
            Dict mapping elbow frame name ("L_arm_l4"/"R_arm_l4") to a 4x4
            homogeneous pose. Empty if FK fails.
        """
        try:
            elbow_poses = self.motion_manager.fk(self._elbow_frames)
        except Exception as e:
            logger.debug(f"FK failed for elbow frames: {e}")
            return {}

        target_poses: Dict[str, np.ndarray] = {}
        for frame_name in self._elbow_frames:
            if frame_name in elbow_poses:
                target_poses[frame_name] = np.array(
                    elbow_poses[frame_name].np, dtype=float
                )
        return target_poses

    def _process_single_arm_ee_pose(self, command: RobotCommand) -> None:
        """Process single-arm end-effector pose commands with IK.

        Solves IK for a single arm independently.
        Called when exactly one arm has EE_POSE target.

        Args:
            command: RobotCommand with EE pose target for one arm
        """
        # Process left arm if it has EE_POSE and right doesn't
        if 'left_arm' in command.input_components:
            left_arm_data = command.input_components.get('left_arm')
            if left_arm_data and left_arm_data.get('command_type') == ArmCommandType.EE_POSE:
                command.input_components.pop('left_arm')
                self._solve_and_apply_single_arm_ik(left_arm_data, 'left', command)
                return

        # Process right arm if it has EE_POSE and left doesn't
        if 'right_arm' in command.input_components:
            right_arm_data = command.input_components.get('right_arm')
            if right_arm_data and right_arm_data.get('command_type') == ArmCommandType.EE_POSE:
                command.input_components.pop('right_arm')
                self._solve_and_apply_single_arm_ik(right_arm_data, 'right', command)
                return

    def _solve_and_apply_single_arm_ik(self, arm_data: Dict[str, Any], side: str, command: RobotCommand) -> None:
        """Solve IK for a single arm and apply safely.

        Uses the delta-IK pattern: the input's ``pose`` is the reference FK
        pose captured when tracking started, and ``delta_xyz`` / ``delta_R``
        carry the relative VR motion. The inactive arm is held at its current
        FK pose (not the world origin) so the simultaneous dual-arm IK solve
        keeps it in place. Dashboard jog references (no ``pose``) are
        resolved by _resolve_dashboard_jog_reference, which anchors the
        position for rotation-only jogs.

        Args:
            arm_data: Arm input data with 'pose' key (reference) and optional
                'delta_xyz' / 'delta_R'.
            side: 'left' or 'right'
            command: RobotCommand to update with results
        """
        ee_frame = 'L_ee' if side == 'left' else 'R_ee'
        inactive_frame = 'R_ee' if side == 'left' else 'L_ee'
        arm_name = f'{side}_arm'

        if 'pose' not in arm_data:
            if not self._resolve_dashboard_jog_reference(arm_data, ee_frame):
                return

        # Active arm: reference pose + delta (delta-IK pattern).
        target_poses = {ee_frame: self._build_ee_target_pose(arm_data)}

        # Inactive arm: hold at current FK pose instead of the world origin.
        # Solving against identity would pull the inactive EE toward the base
        # frame and distort the simultaneous dual-arm solution.
        target_poses[inactive_frame] = self._hold_current_ee_pose(inactive_frame)

        # Add elbow orientation targets (pinned to current FK) so the 7-DOF
        # null-space is resolved together with the EE targets in one solve_ik
        # call. Both arms' elbows are included since the IK is simultaneous.
        with suppress_loguru_module("dexmotion", enabled=True):
            target_poses.update(self._build_elbow_target_poses())
            qpos_dict, is_collision, _ = (
                self.motion_manager.local_ik_solver.solve_ik(target_poses)
            )
            solution = qpos_dict
            within_limits = True  # local solver does not report limits

        if not solution or not within_limits or is_collision:
            return

        # Get arm processor for safety limiting
        processor = self.processors.get(arm_name)
        if not processor:
            return

        # Extract and limit joint positions
        joint_indices = [f"{side[0].upper()}_arm_j{i + 1}" for i in range(7)]
        positions = [solution[joint_idx] for joint_idx in joint_indices]

        safe_positions = processor.limit_joint_step(positions)

        # Update motion manager
        processor.apply_positions(safe_positions)

        # Update command - only active arm
        command.output_components[arm_name] = {
            'pos': safe_positions.tolist(),
            'mode': CommandMode.ABSOLUTE.value,
        }

    def publish_command(self, command: RobotCommand) -> None:
        """Publish validated command to robot controller.

        Converts RobotCommand to dictionary format and publishes
        via Zenoh. Only output components are sent.

        Args:
            command: Validated RobotCommand to publish.

        Message Format:
            {
                'timestamp_ns': int,
                'components': {component_name: component_data},
                'safety_flags': {flag_name: bool}
            }
        """
        # Convert to publishable format
        command_dict = {
            "timestamp_ns": command.timestamp_ns,
            "components": {},
            "safety_flags": {
                "emergency_stop": command.safety_flags.emergency_stop,
                "exit_requested": command.safety_flags.exit_requested,
            },
        }

        # Convert numpy arrays to lists (only output components are published)
        for name, data in command.output_components.items():
            component_data = {}

            # Handle position data
            if "pos" in data:
                component_data["pos"] = (
                    data["pos"].tolist()
                    if isinstance(data["pos"], np.ndarray)
                    else data["pos"]
                )

            # Handle velocity data
            if "vel" in data:
                component_data["vel"] = (
                    data["vel"].tolist()
                    if isinstance(data["vel"], np.ndarray)
                    else data["vel"]
                )

            # Add mode if present
            if "mode" in data:
                component_data["mode"] = data["mode"]

            # Add other fields (for chassis: vx, vy, wz)
            for key in data:
                if key not in ["pos", "vel", "mode"]:
                    component_data[key] = data[key]

            command_dict["components"][name] = component_data
            self._last_output_time[name] = time.perf_counter()

        # Convert numpy types to native Python types (DictDataCodec uses orjson
        # without OPT_SERIALIZE_NUMPY, so numpy types aren't auto-handled)
        command_dict = self._to_json_serializable(command_dict)

        # Publish through Dexcomm
        self.safe_command_pub.publish(command_dict)

    @staticmethod
    def _to_json_serializable(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types.

        DictDataCodec uses orjson without OPT_SERIALIZE_NUMPY flag,
        so numpy scalars (float64, int64, etc.) cause TypeError.
        """
        if isinstance(obj, dict):
            return {
                k: CommandProcessor._to_json_serializable(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [CommandProcessor._to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _is_exo_aligned_with_init_pos(self, command: RobotCommand) -> bool:
        """Checks whether exo arm joints are within threshold of the config init_pos.

        Only runs when teleop_mode is 'exo_joycon'. For other modes (e.g. VR)
        alignment is not required and this returns True immediately.

        Args:
            command: Current RobotCommand containing exo input_components.

        Returns:
            True if aligned (or mode doesn't require check), False otherwise.
        """
        if self.teleop_mode != "exo_joycon":
            return True

        checks = [
            ("left_arm", self._init_pos_left_arm),
            ("right_arm", self._init_pos_right_arm),
        ]
        for arm_name, init_pos in checks:
            arm_data = command.input_components.get(arm_name, {})
            positions = arm_data.get("pos", [])
            if not positions:
                continue
            for i, (exo_val, init_val) in enumerate(zip(positions, init_pos)):
                diff = abs(exo_val - init_val)
                if diff > self._align_threshold:
                    key = f"{arm_name}_{i}"
                    prev = self._last_align_log.get(key, -999)
                    if (
                        abs(diff - prev) > 0.01
                    ):  # only log when value changes meaningfully
                        logger.warning(
                            f"[ALIGN] {arm_name} joint {i + 1} not aligned: "
                            f"exo={exo_val:.3f} init={init_val:.3f} diff={diff:.3f} "
                            f"(threshold {self._align_threshold})"
                        )
                        self._last_align_log[key] = diff
                    return False
        self._last_align_log.clear()
        return True

    def _resync_if_idle(self, component_name: str, processor) -> None:
        """Re-sync one processor's virtual state from robot feedback if idle.

        Only active when ``hold_timeout_s`` is set in the config. A component is
        idle when it produced no output for longer than that timeout (the
        robot_controller has stopped holding it by then).
        """
        if self._idle_resync_s is None:
            return
        last = self._last_output_time.get(component_name)
        if last is not None and (time.perf_counter() - last) <= self._idle_resync_s:
            return
        with self._robot_joints_lock:
            joints = (
                self.latest_robot_joints.get("joints", {})
                if self.latest_robot_joints
                else {}
            )
        if not joints:
            return
        try:
            processor.sync_to_robot_state(joints)
            logger.info(f"[hold] {component_name} idle -> virtual state re-synced from robot feedback")
        except Exception as exc:
            logger.warning(f"[hold] {component_name} resync failed: {exc}")

    def _sync_motion_manager_to_robot_state(self) -> None:
        """Synchronize motion manager with actual robot joint positions.

        Updates motion manager's internal state to match the real robot's
        current configuration. Called when motion control starts.

        Uses component processors to sync each component.
        """
        with self._robot_joints_lock:
            latest_robot_joints = (
                self.latest_robot_joints.copy() if self.latest_robot_joints else None
            )

        if latest_robot_joints is None:
            return

        joint_data = latest_robot_joints.get("joints", {})
        if not joint_data:
            return

        # Sync each component via its processor
        # todo: 시뮬레이션 상에서 상태 동기화가 제대로 동작안해서 주석 처리
        for component_name, processor in self.processors.items():
            if processor.is_enabled():
                processor.sync_to_robot_state(joint_data)

    def run(self) -> None:
        """Execute main command processing loop.

        Runs at configured rate (default 40Hz), processing commands
        and publishing validated outputs until shutdown.

        Processing Steps:
            1. Wait for robot joint data initialization
            2. Process teleoperation commands
            3. Apply safety validation
            4. Publish validated commands
            5. Handle exit requests
            6. Update debug display (if enabled)
        """
        self.running = True
        rate_hz = self.config.get_rate("command_rate", 40)
        rate_limiter = RateLimiter(rate_hz)

        logger.info(f"Starting command & safety processing at {rate_hz}Hz")

        # Start the live display if debug mode
        if self._debug_display:
            self._debug_display.start()

        while not self._robot_joint_data_initialized:
            time.sleep(0.1)

        while self.running:
            command = self.process_commands()

            if command:
                if not self.bypass_real_robot:
                    self.publish_command(command)

                # Check if we should exit after publishing
                if self.exit_after_publish or command.safety_flags.exit_requested:
                    logger.critical("EXIT SIGNAL SENT - Shutting down all systems")
                    self.running = False
                    time.sleep(0.2)  # Small delay to ensure message is sent
                    break

                # Update debug display
                if self._debug_display:
                    # Convert safety flags to dict
                    safety_flags_dict = {
                        "collision_detected": command.safety_flags.collision_detected,
                        "limits_enforced": command.safety_flags.limits_enforced,
                        "emergency_stop_active": command.safety_flags.emergency_stop,
                    }
                    # Add active modes if available
                    if hasattr(command, 'debug_info') and command.debug_info:
                        safety_flags_dict["active_modes"] = command.debug_info.get("active_modes", "none")
                    self._debug_display.print_robot_command(
                        command.output_components, safety_flags_dict
                    )

            # Maintain rate
            rate_limiter.sleep()

        # Cleanup on exit
        logger.info("Command processor loop exited, cleaning up...")
        self.stop()

    def stop(self) -> None:
        """Shutdown command processor cleanly.

        Stops processing loop, cleans up resources, and shuts down
        communication channels.
        """
        self.running = False

        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        # Clean up input handler
        if self.input_handler:
            self.input_handler.cleanup()

        # Clean up common handler
        self.common_handler.cleanup()

        self.node.shutdown()  # Dexcomm Node cleanup
        logger.info("CommandProcessor stopped")

def main(
    namespace: str = "",
    debug: bool = False,
    visualize: bool = False,
    sim: bool = False,
    config_name: Optional[str] = None,
) -> None:
    """Main entry point for command processor.

    Creates and runs the command processor with specified configuration.

    Args:
        namespace: Zenoh namespace for topic isolation.
        debug: Whether to enable debug display output.
        visualize: Whether to visualize the robot.
        sim: Whether to run in simulation mode.
        config_name: Name of config file (without .yaml extension).

    Example:
        $ omni-cmd --namespace /robot1 --debug
    """
    # Setup logging
    logger = setup_logging(debug)
    # Create and run processor (config loaded automatically)
    processor = CommandProcessor(
        namespace=namespace,
        debug=debug,
        visualize=visualize,
        sim=sim,
        config_name=config_name,
    )

    exit_requested = False
    try:
        processor.run()
        exit_requested = processor.exit_after_publish
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt...")
        processor.stop()

    # Force exit to ensure all threads terminate
    if exit_requested:
        import os

        logger.info("Forcing process exit due to exit request")
        os._exit(0)  # Force immediate exit without cleanup (all threads killed)

if __name__ == "__main__":
    import sys
    import tyro

    sys.exit(tyro.cli(main))
