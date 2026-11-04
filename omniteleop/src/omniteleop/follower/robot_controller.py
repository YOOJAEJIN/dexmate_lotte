#!/usr/bin/env python3
"""Robot controller with interpolation and hardware control."""

import os
import sys
import time
import numpy as np
import ruckig
from typing import Optional, Dict, List
import tyro
from enum import Enum

from dexcomm import Node
from dexcomm.utils import RateLimiter
from dexcontrol.robot import Robot
from omniteleop import LIB_PATH
from omniteleop.common import get_config
from omniteleop.common.logging import setup_logging
from omniteleop.common.debug_display import get_debug_display
from omniteleop.common.filters import MultiChannelFilter
from omniteleop.common.trajectory_interpolator import TrajectoryInterpolator
from omniteleop.common.ruckig_trajectory import (
    RuckigArmTrajectoryGenerator,
    RuckigTorsoTrajectoryGenerator,
)
from omniteleop.follower.hand_controllers.hand_factory import HandControllerFactory
from loguru import logger
import threading
from dexbot_utils import RobotInfo
from dexcomm.codecs import DictDataCodec

class RobotMode(Enum):
    RUNNING = "running"
    STOP = "stop"
    EXIT = "exit"


class _RobotWithoutVersionCheck(Robot):
    """dexcontrol Robot that skips the server version-compatibility query.

    ``Robot.__init__`` unconditionally waits up to 5 s for the server's
    version-info service (``wait_for_service(timeout=5.0)``). The simulator does
    not provide that service, so every teleop start paid the full timeout for a
    check that can only end in a warning. Enabled via ``dexcontrol.skip_version_check``
    in the robot config; keep it off on real hardware to get version-mismatch warnings.
    """

    def _check_version_compatibility(self) -> None:
        logger.info(
            "Skipping dexcontrol version compatibility check "
            "(dexcontrol.skip_version_check=true)"
        )

class RobotController:
    """Robot controller with integrated motion interpolation and hardware control."""

    def __init__(
        self,
        namespace: str = "",
        interpolation_method: str = "none",
        history_size: int = 4,
        use_velocity_control: bool = False,
        debug: bool = False,
        publish_telemetry: bool = True,
        config_name: Optional[str] = None,
    ) -> None:
        """Initialize robot controller.

        Args:
            namespace: Optional namespace prefix for Zenoh topics.
            interpolation_method: Method for interpolation ('none', 'linear', 'cubic').
            history_size: Number of past commands to keep for interpolation.
            use_velocity_control: Enable velocity control for smoother motion.
            debug: Enable debug output.
            publish_telemetry: Enable telemetry publishing for visualization.
            config_name: Name of the configuration file (without .yaml extension).

        Note:
            Input rate, control rate, and input topic are loaded from config file.
        """
        # Initialize Node (namespace is handled automatically by Node)
        self.node = Node(name="robot_controller", namespace=namespace)

        # Detect robot type (with or without torso/base)
        self.robot_info = RobotInfo()
        self.has_torso = self.robot_info.has_torso
        self.has_base = self.robot_info.has_chassis

        # Load configuration
        config_path = None
        if config_name is not None:
            config_path = LIB_PATH / "configs" / f"{config_name}.yaml"
        self.config = get_config(config_path)
        logger.info(f"Robot controller configured: {self.config}")

        self.input_rate = self.config.get_rate("input_rate", 40)
        self.control_rate = self.config.get_rate("control_rate", 100)
        self.input_topic = self.config.get_topic("robot_commands")
        self.interpolation_method = interpolation_method
        self.history_size = history_size
        self.use_velocity_control = use_velocity_control

        # Communication
        self.subscriber = None
        self.joint_publisher = None

        # Robot hardware interface
        self.robot = None
        self.left_hand_controller = None
        self.right_hand_controller = None

        # Joint feedback publishing
        self.feedback_rate = self.config.get_rate("feedback_rate", 50)
        self.joint_publish_thread = None
        self.joint_publish_running = False

        # Rate limiter for precise timing
        self.rate_limiter = RateLimiter(self.control_rate)

        # Trajectory interpolator for smooth motion
        self.interpolator = TrajectoryInterpolator(
            method=interpolation_method,
            history_size=history_size,
        )

        # Latest command storage for safety checks and telemetry
        self.latest_command = None

        # Hold timeout. In pass-through mode the control loop re-sends
        # latest_command every tick, so a component commanded once (e.g. a single
        # torso jog) is held forever and blocks any other controller on that
        # topic. When ``hold_timeout_s`` is set in the YAML, a component with no
        # fresh command for that long is dropped from the outgoing stream; the
        # robot/sim keeps its last target on its own. None = original infinite hold.
        self.hold_timeout_s: Optional[float] = None
        _raw_hold = self.config.get("hold_timeout_s")
        if _raw_hold is not None:
            try:
                _hold = float(_raw_hold)
            except (TypeError, ValueError):
                logger.warning(f"Invalid hold_timeout_s={_raw_hold!r}; ignoring")
                _hold = None
            if _hold is not None and _hold <= 0:
                _hold = None
            if _hold is not None and interpolation_method != "none":
                logger.warning(
                    "hold_timeout_s is only supported with interpolation_method='none'; ignoring"
                )
                _hold = None
            self.hold_timeout_s = _hold
        if self.hold_timeout_s is not None:
            logger.info(f"Hold timeout enabled: components idle > {self.hold_timeout_s}s are released")
        self._component_last_update: Dict[str, float] = {}
        self._released_components: set = set()

        # Current state - dict based per component
        self.current_state = {}  # {component: {'pos': [...], 'vel': [...]}}

        # Components that need interpolation
        self.interpolated_components = {"left_arm", "right_arm"}
        # Add torso only if robot has torso
        if self.has_torso:
            self.interpolated_components.add("torso")

        # Components that are passed through directly
        self.direct_components = {"head"}
        # Add chassis only if robot has base
        if self.has_base:
            self.direct_components.add("chassis")
        # Initialize multi-channel filter from config
        filter_config = self.config.get("filters", None)

        # Create the multi-channel filter
        self.filter = MultiChannelFilter(
            filter_config=filter_config, control_rate=self.control_rate
        )

        # Determine hand type from ROBOT_CONFIG env var
        robot_config = os.environ.get("ROBOT_CONFIG", "vega_1_f5d6")
        if "gripper" in robot_config:
            self.hand_type = "gripper"
        elif "f5d6" in robot_config:
            self.hand_type = "f5d6"
        else:
            self.hand_type = None  # No hands (e.g., vega_1, vega_1p, vega_1u)
            common_config = self.config.get("input_handlers", {}).get("common", {})
            hands_common = common_config.get("hands", {})
            
            # Support distinct hand types for left and right side
            self.left_ee_type = hands_common.get("left_ee_type")
            self.right_ee_type = hands_common.get("right_ee_type")

        # Add hands to interpolated components only if robot has hands
        if self.hand_type is not None:
            self.direct_components.add("left_hand")
            self.direct_components.add("right_hand")

        # Log filter configuration
        if filter_config:
            default_type = filter_config.get("default", {}).get("type", "none")
            logger.info(f"Filter configuration loaded - default: {default_type}")
            if "components" in filter_config and filter_config["components"]:
                comp_filters = {}
                for comp, cfg in filter_config["components"].items():
                    comp_filters[comp] = cfg.get("type", "unknown")
                logger.info(f"Component-specific filters: {comp_filters}")

        # Robot state tracking

        # Store home positions for reuse
        self.home_positions = {}

        # Ruckig trajectory generators (initialized after home positions are known)
        self.ruckig_generators: Dict[
            str, RuckigArmTrajectoryGenerator | RuckigTorsoTrajectoryGenerator
        ] = {}

        self._robot_mode = None
        self._is_first_command = True
        self.exit_requested = False

        # Debug mode with efficient display
        self.debug = debug
        self._debug_display = None
        if debug:
            self._debug_display = get_debug_display(
                "Robot", self.control_rate, refresh_rate=10
            )

        # Telemetry publishing for visualization
        self.publish_telemetry = publish_telemetry
        self.telemetry_publisher = None

        logger.info(
            f"Robot controller: {self.input_rate}Hz input -> {self.control_rate}Hz control"
        )

    def initialize(self) -> None:
        """Initialize communication and robot hardware.

        Sets up Zenoh publishers/subscribers and initializes the robot
        hardware interface. Moves robot to home position.
        """
        # Initialize subscriber through Node (namespace handled automatically)
        self.subscriber = self.node.create_subscriber(
            self.input_topic,
            callback=self._on_safe_command,
            decoder=DictDataCodec.decode,
        )

        # Initialize joint feedback publisher
        joint_topic = self.config.get_topic("robot_joints")
        self.joint_publisher = self.node.create_publisher(
            joint_topic,
            encoder=DictDataCodec.encode,
        )
        logger.info(
            f"Publishing robot joints to {self.node.resolve_topic(joint_topic)} "
            f"at {self.feedback_rate}Hz"
        )

        # Initialize telemetry publisher for visualization
        if self.publish_telemetry:
            telemetry_topic = self.config.get_topic("telemetry")
            self.telemetry_publisher = self.node.create_publisher(
                telemetry_topic,
                encoder=DictDataCodec.encode,
            )
            logger.info(
                f"Publishing telemetry to {self.node.resolve_topic(telemetry_topic)}"
            )

        # Initialize Dexcontrol robot
        logger.info("Initializing robot hardware...")
        dexcontrol_cfg = self.config.get("dexcontrol", {}) or {}
        if dexcontrol_cfg.get("skip_version_check", False):
            self.robot = _RobotWithoutVersionCheck()
        else:
            self.robot = Robot()

        # Initialize Hand controllers if pass-through is enabled
        if self.robot.left_arm.enable_ee_pass_through:
            self.left_hand_controller = HandControllerFactory.create_hand_controller(
                self.left_ee_type, self.robot.left_arm, side="left"
            )
            if self.left_hand_controller:
                self.left_hand_controller.activate()
                self.left_hand_controller.start_tracking()

        if self.robot.right_arm.enable_ee_pass_through:
            self.right_hand_controller = HandControllerFactory.create_hand_controller(
                self.right_ee_type, self.robot.right_arm, side="right"
            )
            if self.right_hand_controller:
                self.right_hand_controller.activate()
                self.right_hand_controller.start_tracking()

        # Parse home positions from config (no hardware movement)
        self._parse_home_positions()

        if self.interpolation_method == "ruckig":
            self._init_ruckig_generators()
            self._move_to_home_with_ruckig()
        else:
            self._move_to_home()

        self._robot_mode = RobotMode.RUNNING

        logger.success("Robot controller initialized")

    def _init_ruckig_generators(self) -> None:
        """Initialize per-component ruckig trajectory generators from home positions."""
        control_cycle = 1.0 / self.control_rate

        for component, home_pos in self.home_positions.items():
            qpos = np.array(home_pos)
            if component in {"left_arm", "right_arm"}:
                self.ruckig_generators[component] = RuckigArmTrajectoryGenerator(
                    init_qpos=qpos,
                    control_cycle=control_cycle,
                    safety_factor=1.0,
                )
            elif component == "torso":
                self.ruckig_generators[component] = RuckigTorsoTrajectoryGenerator(
                    init_qpos=qpos,
                    control_cycle=control_cycle,
                )
            # head, hands, chassis are passed through directly

        logger.info(
            f"Ruckig generators initialized for: {list(self.ruckig_generators.keys())}"
        )

    def _move_to_home_with_ruckig(self) -> None:
        """Drive arms and torso to home position using ruckig.

        Seeds each generator from actual hardware position, sets home as target,
        then steps the ruckig loop until ruckig reports Finished for all components.
        """
        rate_limiter = RateLimiter(self.control_rate)

        # Seed generators from actual robot positions and set home as target
        actual_positions = self._get_robot_joint_pos()
        for component, gen in self.ruckig_generators.items():
            current = np.array(
                actual_positions.get(component, self.home_positions[component])
            )
            target = np.array(self.home_positions[component])
            gen.reset(current)
            gen.inp.target_position = target.tolist()
            gen.inp.target_velocity = [0.0] * gen.dof
            gen.inp.target_acceleration = [0.0] * gen.dof

        logger.info("Moving to home position with ruckig...")
        self.robot.estop.deactivate()
        time.sleep(2.0)

        # Track per-component completion
        finished = {comp: False for comp in self.ruckig_generators}

        while not all(finished.values()):
            for component, gen in self.ruckig_generators.items():
                if finished[component]:
                    continue

                result = gen.otg.update(gen.inp, gen.out)
                gen.out.pass_to_input(gen.inp)
                cmd_pos = np.array(gen.out.new_position)

                if component == "torso":
                    self.robot.torso.move_to_joint_pos(cmd_pos)
                elif component in {"left_arm", "right_arm"}:
                    getattr(self.robot, component).move_to_joint_pos(cmd_pos)

                if result != ruckig.Result.Working:
                    finished[component] = True

            rate_limiter.sleep()

        # Open hands only if robot has hands
        if self.hand_type is not None:
            self.robot.left_hand.open_hand()
            self.robot.right_hand.open_hand()
        self.robot.estop.activate()
        logger.info("Home position reached via ruckig")

    def _get_robot_joint_pos(self) -> Dict[str, List[float]]:
        """Get robot joint positions.

        Returns:
            Dictionary mapping component names to joint position lists.
        """
        positions = {
            "left_arm": self.robot.left_arm.get_joint_pos().tolist(),
            "right_arm": self.robot.right_arm.get_joint_pos().tolist(),
            "head": self.robot.head.get_joint_pos().tolist(),
        }

        # Add torso only if robot has torso
        if self.has_torso:
            positions["torso"] = self.robot.torso.get_joint_pos().tolist()

        # Add hands only if robot has hands
        if self.hand_type is not None:
            positions["left_hand"] = self.robot.left_hand.get_joint_pos().tolist()
            positions["right_hand"] = self.robot.right_hand.get_joint_pos().tolist()
        else:
            if self.left_hand_controller:
                positions["left_hand"] = self.left_hand_controller.get_status()
            if self.right_hand_controller:
                positions["right_hand"] = self.right_hand_controller.get_status()

        return positions

    def _get_robot_joint_vel(self) -> Dict[str, List[float]]:
        """Get robot joint velocities.

        Returns:
            Dictionary mapping component names to joint velocity lists.
        """
        velocities = {
            "left_arm": self.robot.left_arm.get_joint_vel().tolist(),
            "right_arm": self.robot.right_arm.get_joint_vel().tolist(),
            "head": self.robot.head.get_joint_vel().tolist(),
        }

        # Add torso only if robot has torso
        if self.has_torso:
            velocities["torso"] = self.robot.torso.get_joint_vel().tolist()

        # Add hands based on hand type
        if not self.robot.left_arm.enable_ee_pass_through:
            velocities["left_hand"] = self.robot.left_hand.get_joint_vel().tolist()
        else:
            if self.left_hand_controller:
                velocities["left_hand"] = [0.0]
        if not self.robot.right_arm.enable_ee_pass_through:
            velocities["right_hand"] = self.robot.right_hand.get_joint_vel().tolist()
        else:
            if self.right_hand_controller:
                velocities["right_hand"] = [0.0]

        return velocities

    def _parse_init_pos(self, component, init_pos: str | list[float]) -> list[float]:
        """Parse initial position from string or list.

        Args:
            component: Robot component with get_predefined_pose method.
            init_pos: Either a predefined pose name or list of joint positions.

        Returns:
            List of joint positions.
        """
        if isinstance(init_pos, str):
            return component.get_predefined_pose(init_pos)
        else:
            return init_pos

    def _parse_home_positions(self) -> None:
        """Parse home positions from config and store for reuse. No hardware movement."""
        init_pos_config = self.config.get("init_pos", {})

        for arm in ["left_arm", "right_arm"]:
            if hasattr(self.robot, arm):
                init_pos = init_pos_config.get(arm)
                if init_pos is not None:
                    robot_component = getattr(self.robot, arm)
                    self.home_positions[arm] = self._parse_init_pos(
                        robot_component, init_pos
                    )

        other_components = ["head", "left_hand", "right_hand"]
        if self.has_torso:
            other_components.insert(0, "torso")

        for component in other_components:
            if hasattr(self.robot, component):
                init_pos = self.config.get_init_pos(component)
                if init_pos is not None:
                    robot_component = getattr(self.robot, component)
                    self.home_positions[component] = self._parse_init_pos(
                        robot_component, init_pos
                    )

        logger.debug(f"Home positions parsed: {list(self.home_positions.keys())}")

    def _move_to_home(self) -> None:
        """Move robot to home position using hardware move_to_joint_pos."""
        self.robot.estop.deactivate()
        time.sleep(0.1)

        # Move torso first if available
        if self.has_torso and "torso" in self.home_positions:
            self.robot.torso.move_to_joint_pos(
                self.home_positions["torso"]
            )

        # Move arms and head
        arm_head_pose = {}
        for component in ["left_arm", "right_arm", "head"]:
            if component in self.home_positions:
                arm_head_pose[component] = self.home_positions[component]

        if arm_head_pose:
            self.robot.move_to_joint_pos(arm_head_pose)
            logger.info("Robot moved to home position")

        # Open hands only if robot has hands
        if self.hand_type is not None:
            self.robot.left_hand.open_hand()
            self.robot.right_hand.open_hand()
        else:
            if self.left_hand_controller:
                self.left_hand_controller.open()
            if self.right_hand_controller:
                self.right_hand_controller.open()
        # self.robot.estop.activate()

    def _on_safe_command(self, data: Dict) -> None:
        """Handle incoming safe command.

        Args:
            data: Command dictionary from command processor with components
                  and safety flags.
        """
        try:
            # Store full command for safety checks and telemetry
            self.latest_command = data

            # Extract positions for interpolation
            timestamp = time.perf_counter()
            positions = {}

            cmd_components = data.get("components", {})
            for component in cmd_components:
                self._component_last_update[component] = timestamp
                if component in self._released_components:
                    self._released_components.discard(component)
                    logger.info(f"[hold] {component} re-acquired (fresh command)")
            for component, comp_data in cmd_components.items():
                # Only add interpolated components to the interpolator
                if component in self.interpolated_components and "pos" in comp_data:
                    positions[component] = np.array(comp_data["pos"])

            # Add to interpolator
            if positions:
                self.interpolator.add_point(timestamp, positions)

        except Exception as e:
            logger.error(f"Error processing safe command: {e}")

    def _compute_interpolated_command(self) -> Optional[Dict]:
        """Compute interpolated command for current time using TrajectoryInterpolator.

        Returns:
            Optional[Dict]: Command dictionary with interpolated components.
        """
        if not self.latest_command:
            return None

        current_time = time.perf_counter()

        # Check for exit signal first (highest priority)
        safety_flags = self.latest_command.get("safety_flags", {})
        if safety_flags.get("exit_requested", False):
            logger.critical("Exit signal received - shutting down immediately")
            self._robot_mode = RobotMode.EXIT
            self.exit_requested = True
            return None

        # Check for emergency stop
        emergency_stop = safety_flags.get("emergency_stop", False)

        if emergency_stop:
            if self._robot_mode != RobotMode.STOP:
                logger.warning("Emergency stop activated")
                self.robot.estop.activate()
                self._robot_mode = RobotMode.STOP
            return None  # Don't process any commands during estop
        elif self._robot_mode == RobotMode.STOP:
            # Emergency stop released
            logger.info("Emergency stop released")
            self.robot.estop.deactivate()
            self.robot.head.set_mode("enable")
            # dexcontrol 0.5: move_to_joint_pos() returns a MotionHandle; the old
            # wait_time/exit_on_reach args are replaced by handle.wait(timeout=...),
            # which blocks until the motion plugin reports completion.
            handle = self.robot.head.move_to_joint_pos(self.home_positions["head"])
            try:
                handle.wait(timeout=5.0)
            except TimeoutError:
                logger.warning("Head move-to-home timed out; continuing.")
            self._robot_mode = RobotMode.RUNNING

        # Get command components
        cmd_components = self.latest_command.get("components", {})

        if not cmd_components:
            return None

        # Final output components
        output_components = {}

        # Handle interpolation based on method
        if self.interpolation_method == "none" or (
            self._is_first_command and self.interpolation_method != "ruckig"
        ):
            # No interpolation: pass through command directly
            for component, data in cmd_components.items():
                formatted_data = {}
                if "pos" in data:
                    formatted_data["pos"] = np.array(data["pos"])
                if "vel" in data:
                    formatted_data["vel"] = np.array(data["vel"])
                # Pass through other fields (like vx, vy, wz for base)
                for key in data:
                    if key not in ["pos", "vel"]:
                        formatted_data[key] = data[key]
                output_components[component] = formatted_data
        elif self.interpolation_method == "ruckig":
            # On first command, reset generators to incoming positions to avoid jumps
            if self._is_first_command:
                for component, gen in self.ruckig_generators.items():
                    if (
                        component in cmd_components
                        and "pos" in cmd_components[component]
                    ):
                        gen.reset(np.array(cmd_components[component]["pos"]))

            # Ruckig jerk-limited interpolation for arms and torso
            for component, gen in self.ruckig_generators.items():
                target = None
                if component in cmd_components and "pos" in cmd_components[component]:
                    target = np.array(cmd_components[component]["pos"])
                pos, vel = gen.update(target)
                output_components[component] = {"pos": pos, "vel": vel}

            # Pass through all other components directly (head, hands, chassis)
            ruckig_components = set(self.ruckig_generators.keys())
            for component, data in cmd_components.items():
                if component not in ruckig_components:
                    formatted_data = {}
                    if "pos" in data:
                        formatted_data["pos"] = np.array(data["pos"])
                    if "vel" in data:
                        formatted_data["vel"] = np.array(data["vel"])
                    for key in data:
                        if key not in ["pos", "vel"]:
                            formatted_data[key] = data[key]
                    output_components[component] = formatted_data
        else:
            # Use interpolator for smooth trajectories
            positions, velocities = self.interpolator.interpolate(
                current_time, compute_velocity=True
            )

            if positions and velocities:
                # Add interpolated components with positions and velocities
                for component in positions:
                    output_components[component] = {
                        "pos": positions[component],
                        "vel": velocities[component],
                    }

            # Add direct-pass components (e.g., chassis) from latest command
            for component, data in cmd_components.items():
                if component in self.direct_components:
                    formatted_data = {}
                    for key, value in data.items():
                        if isinstance(value, list):
                            formatted_data[key] = np.array(value)
                        else:
                            formatted_data[key] = value
                    output_components[component] = formatted_data

        # Release components whose last command is older than hold_timeout_s
        if self.hold_timeout_s is not None and output_components:
            now = time.perf_counter()
            for component in list(output_components):
                age = now - self._component_last_update.get(component, now)
                if age > self.hold_timeout_s:
                    output_components.pop(component)
                    if component not in self._released_components:
                        self._released_components.add(component)
                        logger.info(
                            f"[hold] {component} released after {age:.2f}s without a fresh command"
                        )
            if not output_components:
                self.current_state = {}
                return None

        # Apply smoothing filter only for linear/cubic interpolation
        if output_components and self.interpolation_method in ("linear", "cubic"):
            output_components = self.filter.apply(output_components)

        # Update current state
        self.current_state = output_components

        return {
            "components": output_components,
            "timestamp_ns": time.time_ns(),  # Absolute timestamp for logging/diagnostics
        }

    def _send_robot_command(self, command: Dict):
        """Send command to robot hardware.

        Args:
            command: Command dictionary with components.
        """
        try:
            components = command.get("components", {})

            if not components:
                logger.debug("No components to send")
                return

            # Process base command first (only if robot has base)
            if self.has_base and "chassis" in components:
                self._send_base_command(components["chassis"])

            # Process torso command (only if robot has torso)
            if self.has_torso and "torso" in components:
                self._send_torso_command(components["torso"])

            # Process head command
            if "head" in components:
                self._send_head_command(components["head"])

            # Process arm commands
            if "left_arm" in components:
                self._send_arm_command("left_arm", components["left_arm"])

            if "right_arm" in components:
                self._send_arm_command("right_arm", components["right_arm"])

            if "left_hand" in components:
                self._send_hand_command("left_hand", components["left_hand"])
            if "right_hand" in components:
                self._send_hand_command("right_hand", components["right_hand"])

        except Exception as e:
            component_keys = list(command.get("components", {}).keys())
            logger.error(f"Failed to send robot command for components: {component_keys}", exc_info=True)

    def _publish_telemetry(self, command: Dict):
        """Publish telemetry data for visualization.

        Args:
            command: Command dictionary with components.
        """
        try:
            telemetry_data = {
                "timestamp_ns": time.time_ns(),
                "timestamp": time.time(),  # Keep both for compatibility
                "components": {},
                "robot_state": {},
            }

            # Get command data for each component (after filtering - filtered command)
            components = command.get("components", {})
            for comp_name, comp_data in components.items():
                # Convert numpy arrays to lists for proper serialization
                component_copy = {}
                for key, value in comp_data.items():
                    if isinstance(value, np.ndarray):
                        component_copy[key] = value.tolist()
                    else:
                        component_copy[key] = value
                telemetry_data["components"][comp_name] = component_copy

            # Get current robot state (actual positions and velocities from hardware)
            if self.robot:
                robot_state_pos = self._get_robot_joint_pos()
                robot_state_vel = self._get_robot_joint_vel()

                # Combine positions and velocities into robot_state
                telemetry_data["robot_state"] = {
                    "positions": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in robot_state_pos.items()
                    },
                    "velocities": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in robot_state_vel.items()
                    },
                }

            # Get raw command (before filtering) from latest_command
            if self.latest_command:
                raw_components = self.latest_command.get("components", {})
                raw_command = {}
                for comp_name, comp_data in raw_components.items():
                    if isinstance(comp_data, dict):
                        component_copy = {}
                        for key, value in comp_data.items():
                            if isinstance(value, np.ndarray):
                                component_copy[key] = value.tolist()
                            elif isinstance(value, list):
                                component_copy[key] = value
                            else:
                                component_copy[key] = value
                        raw_command[comp_name] = component_copy
                    else:
                        raw_command[comp_name] = comp_data
                telemetry_data["raw_command"] = raw_command

            # Publish telemetry
            self.telemetry_publisher.publish(telemetry_data)

        except Exception as e:
            logger.error(f"Failed to publish telemetry: {e}", exc_info=True)

    def _sync_robot_arms_with_leader(self, command: Dict):
        """Sync robot arms with leader arms."""
        components = command.get("components", {})
        arm_commands = {}
        if "left_arm" in components:
            arm_commands["left_arm"] = components["left_arm"]["pos"]
        if "right_arm" in components:
            arm_commands["right_arm"] = components["right_arm"]["pos"]
        if arm_commands:
            self.robot.move_to_joint_pos(arm_commands)

    def _send_base_command(self, base_data: Dict):
        """Send command to mobile base.

        Args:
            base_data: Dictionary with vx, vy, wz velocities.
        """
        self.robot.chassis.set_velocity(
            vx=base_data["vx"],
            vy=base_data["vy"],
            wz=base_data["wz"],
        )

    def _send_torso_command(self, torso_data: Dict):
        """Send command to torso joints.

        Args:
            torso_data: Dictionary with position and velocity.
        """
        if self.interpolation_method == "none":
            self.robot.torso.set_joint_pos_vel(
                torso_data["pos"],
                0.2,
                wait_time=0.0,
            )
        else:
            self.robot.torso.set_joint_pos_vel(
                torso_data["pos"],
                torso_data["vel"],
                wait_time=0.0,
            )

    def _send_head_command(self, head_data: Dict):
        """Send command to head joints.

        Uses ``set_joint_pos(wait_time=0.0)`` for the same streaming-control
        reason as ``_send_arm_command``: the head target is updated every tick,
        so planning a full trajectory per call (``move_to_joint_pos``) would
        accumulate queued motions and lag behind the operator.

        Args:
            head_data: Dictionary with position and optional velocity.
        """
        head = getattr(self.robot, "head", None)
        if head is not None:
            if "pos" in head_data:
                positions = head_data["pos"]
                head.set_joint_pos(positions, wait_time=0.0)

    def _send_arm_command(self, arm_name: str, arm_data: Dict):
        """Send command to arm.

        Uses ``set_joint_pos(wait_time=0.0)`` rather than ``move_to_joint_pos``:
        the command stream arrives at ~40Hz and each tick is a new *target*, not
        a self-contained trajectory. ``move_to_joint_pos`` plans a full motion
        per call and returns a ``MotionHandle``; when called every tick the
        queued trajectories pile up and the arm lags far behind the operator
        (observed ~10s delay that resolves the instant the trigger is released
        and the stream stops). ``set_joint_pos`` with ``wait_time=0.0`` only
        updates the target and returns immediately, which is the correct
        semantics for streaming teleoperation.

        Args:
            arm_name: 'left_arm' or 'right_arm'.
            arm_data: Dictionary with position and optional velocity.
        """
        arm = getattr(self.robot, arm_name, None)
        if arm is not None:
            if "pos" in arm_data:
                positions = arm_data["pos"]
                if "vel" in arm_data and self.use_velocity_control:
                    velocities = arm_data["vel"]
                    arm.set_joint_pos_vel(positions, velocities)
                else:
                    arm.set_joint_pos(positions, wait_time=0.0)

    def _send_hand_command(self, hand_name: str, hand_data: Dict):
        """Send command to hand.

        Args:
            hand_name: 'left_hand' or 'right_hand'
            hand_data: Dictionary with position.
        """
        if "pos" not in hand_data:
            return

        positions = hand_data["pos"]

        # Check if Hand controller is used for this hand
        if hand_name == "left_hand" and self.left_hand_controller:
            self.left_hand_controller.set_position(positions)
            return
        elif hand_name == "right_hand" and self.right_hand_controller:
            self.right_hand_controller.set_position(positions)
            return

        # Fallback to standard hand control
        hand = getattr(self.robot, hand_name, None)
        if hand is not None:
            hand.set_joint_pos(positions, relative=False, wait_time=0.0)

    def _publish_joint_feedback(self):
        """Publish robot joint positions and velocities at specified rate."""
        rate_limiter = RateLimiter(self.feedback_rate)
        logger.info(f"Starting joint feedback publishing at {self.feedback_rate}Hz")

        while self.joint_publish_running:
            # Get current joint positions and velocities
            joint_positions = self._get_robot_joint_pos()
            joint_velocities = self._get_robot_joint_vel()

            # Create message with timestamp, positions and velocities
            feedback_msg = {
                "timestamp_ns": time.time_ns(),
                "joints": joint_positions,
                "velocities": joint_velocities,
            }

            # Publish via Zenoh
            self.joint_publisher.publish(feedback_msg)

            # Maintain rate
            rate_limiter.sleep()

    def run(self) -> None:
        """Main control loop.

        Runs at the configured control rate, processes commands, sends to robot,
        and publishes telemetry. Exits when exit signal is received.
        """
        logger.info(f"Starting robot control at {self.control_rate}Hz")

        # Start the joint feedback publishing thread
        self.joint_publish_running = True
        self.joint_publish_thread = threading.Thread(
            target=self._publish_joint_feedback,
            daemon=True,
            name="JointFeedbackPublisher",
        )
        self.joint_publish_thread.start()

        # Start the live display if debug mode
        if self._debug_display:
            self._debug_display.start()

        while not self.exit_requested:
            # Compute interpolated command
            command = self._compute_interpolated_command()

            if command:
                # Send command to robot hardware
                if self._is_first_command:
                    self._sync_robot_arms_with_leader(command)
                    self._is_first_command = False
                else:
                    self._send_robot_command(command)

                # Efficient debug output using rich
                if self._debug_display:
                    self._debug_display.print_robot_command(
                        command.get("components", {})
                    )

                # Publish telemetry for visualization
                if self.telemetry_publisher and command:
                    self._publish_telemetry(command)

            # Use RateLimiter for precise timing
            self.rate_limiter.sleep()

        if self.exit_requested:
            logger.critical("Robot controller loop exited due to exit request")

    def cleanup(self) -> None:
        """Clean up resources.

        Stops joint feedback publishing, debug display, and robot hardware.
        Shuts down all communication channels.
        """
        # Stop the joint feedback publishing thread
        if self.joint_publish_thread:
            self.joint_publish_running = False
            self.joint_publish_thread.join(timeout=1.0)
            logger.info("Joint feedback publishing stopped")

        # Stop the live display if running
        if self._debug_display:
            self._debug_display.stop()

        # Stop hand monitoring
        if self.left_hand_controller:
            self.left_hand_controller.stop_tracking()
        if self.right_hand_controller:
            self.right_hand_controller.stop_tracking()

        # Telemetry publisher cleanup handled by Node

        if self.robot:
            self.robot.shutdown()

        # Node handles cleanup
        self.node.shutdown()

        logger.info("Robot controller cleaned up")

def main(
    namespace: str = "",
    interpolation_method: str = "none",
    use_velocity_control: bool = False,
    debug: bool = True,
    publish_telemetry: bool = False,
    config_name: Optional[str] = None,
):
    """Main entry point for robot controller.

    Filter configuration is loaded from the YAML config file.
    Uses ROBOT_CONFIG env var to select config if config_name is not provided.

    Args:
        namespace: Optional namespace prefix.
        interpolation_method: Method for interpolation ('none', 'linear', 'cubic', 'ruckig').
        use_velocity_control: Enable velocity control for smoother motion.
        debug: Enable debug output.
        publish_telemetry: Enable telemetry publishing for visualization.
        config_name: Config file name (without .yaml). Uses ROBOT_CONFIG env var if None.
    """
    # Setup logging
    logger = setup_logging(debug)
    logger.info(
        f"Starting Dexexo Robot Controller{f' (namespace: {namespace})' if namespace else ''}"
    )

    controller = RobotController(
        namespace=namespace,
        interpolation_method=interpolation_method,
        use_velocity_control=use_velocity_control,
        debug=debug,
        publish_telemetry=publish_telemetry,
        config_name=config_name,
    )

    controller.initialize()

    controller.run()
    controller.cleanup()

    return 0

if __name__ == "__main__":
    sys.exit(tyro.cli(main))
