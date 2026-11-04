import json
import time
from pathlib import Path

import numpy as np
import tyro
from dexcomm import RateLimiter
from dexmotion.motion_manager import MotionManager
from dexmotion.tasks.move_out_of_self_collision_task import MoveOutOfSelfCollisionTask
from dexmotion.tasks.move_to_configuration_task import MoveToConfigurationTask
from dexmotion.utils import robot_utils
from loguru import logger

from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot


class ArmPositionSafeRunner:
    """Move both arms to a target pose with collision-aware planning."""

    def __init__(
        self,
        pose_path: str = "poses/Home_position.json",
        control_hz: int = 250,
        visualize: bool = True,
        collision_escape_time: float = 2.0,
        bot: Robot | None = None,
    ) -> None:
        self.control_hz = control_hz
        self.visualize = visualize
        self.collision_escape_time = collision_escape_time

        self.pose_path = Path(pose_path)
        if not self.pose_path.is_absolute():
            self.pose_path = Path(__file__).resolve().parent / self.pose_path

        self.bot = bot if bot is not None else Robot()
        self.left_arm: Arm = self.bot.left_arm
        self.right_arm: Arm = self.bot.right_arm
        self.rate_limiter = RateLimiter(self.control_hz)

        self.initial_joint_configuration = self._get_joint_pos_for_motion_manager()

        self.motion_manager = MotionManager(
            init_visualizer=self.visualize,
            initial_joint_configuration_dict=self.initial_joint_configuration,
            init_local_ik=False,
        )

        self.move_out_of_self_collision_task = MoveOutOfSelfCollisionTask(
            initial_joint_configuration=self.initial_joint_configuration,
            motion_manager=self.motion_manager,
            range_size=0.3,
            visualize=self.visualize,
        )

    def _get_joint_pos_for_motion_manager(self) -> dict[str, float]:
        components = ["left_arm", "right_arm"]
        if self.bot.has_component("head"):
            components.append("head")
        if self.bot.has_component("torso"):
            components.append("torso")
        return self.bot.get_joint_pos_dict(components)

    def _load_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_path}")

        with self.pose_path.open("r", encoding="utf-8") as f:
            pose = json.load(f)

        if "left" not in pose or "right" not in pose:
            raise ValueError("Pose json must include 'left' and 'right' keys")

        left = np.array(pose["left"], dtype=np.float64)
        right = np.array(pose["right"], dtype=np.float64)

        if left.shape[0] != len(self.left_arm.joint_name):
            raise ValueError(
                f"left pose size mismatch: expected {len(self.left_arm.joint_name)}, got {left.shape[0]}"
            )
        if right.shape[0] != len(self.right_arm.joint_name):
            raise ValueError(
                f"right pose size mismatch: expected {len(self.right_arm.joint_name)}, got {right.shape[0]}"
            )

        return left, right

    def _create_goal_configuration(self, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        goal = dict(zip(self.left_arm.joint_name, left.tolist()))
        goal.update(dict(zip(self.right_arm.joint_name, right.tolist())))
        return goal

    def _execute_trajectory(self, trajectory: list[dict[str, float]]) -> None:
        left_joint_order = self.left_arm.joint_name
        right_joint_order = self.right_arm.joint_name

        for waypoint in trajectory:
            left_q = np.array([waypoint[jn] for jn in left_joint_order], dtype=np.float64)
            right_q = np.array([waypoint[jn] for jn in right_joint_order], dtype=np.float64)
            self.left_arm.set_joint_pos(left_q)
            self.right_arm.set_joint_pos(right_q)
            self.rate_limiter.sleep()

    def _resolve_self_collision_if_needed(self, joint_names: list[str]) -> None:
        is_collision_present, success, collision_free_configuration = (
            self.move_out_of_self_collision_task.run()
        )

        if not is_collision_present:
            return

        if not success or not isinstance(collision_free_configuration, dict):
            raise RuntimeError("Failed to resolve self-collision before planning")

        initial_configuration = self.move_out_of_self_collision_task.initial_joint_configuration
        if not isinstance(initial_configuration, dict):
            raise RuntimeError("Invalid initial configuration from collision task")

        num_steps = int(self.control_hz * self.collision_escape_time)
        interpolated = []
        for i in range(num_steps + 1):
            alpha = i / max(num_steps, 1)
            q = {}
            for joint in joint_names:
                if joint in initial_configuration and joint in collision_free_configuration:
                    q[joint] = (1 - alpha) * initial_configuration[joint] + alpha * collision_free_configuration[joint]
            interpolated.append(q)

        if self.visualize:
            if self.motion_manager.pin_robot is None or self.motion_manager.visualizer is None:
                raise RuntimeError("Visualizer is not initialized")
            self.motion_manager.visualizer.update_motion_plan(
                motion_plan=np.array(
                    [
                        robot_utils.get_qpos_from_joint_dict(self.motion_manager.pin_robot, q)
                        for q in interpolated
                    ]
                ),
                joint_names=joint_names,
                duration=1.0,
            )

        logger.warning("Self-collision detected. Moving to collision-free configuration first.")
        self._execute_trajectory(interpolated)
        time.sleep(1.0)

    def _normalize_waypoints(
        self,
        waypoints: object,
        joint_names: list[str],
    ) -> list[dict[str, float]]:
        """Convert planner outputs into list[dict[str, float]]."""
        if isinstance(waypoints, list):
            if len(waypoints) == 0:
                raise RuntimeError("Planner returned empty waypoint list")
            if isinstance(waypoints[0], dict):
                return [
                    {jn: float(wp[jn]) for jn in joint_names if jn in wp}
                    for wp in waypoints
                ]

        if isinstance(waypoints, np.ndarray):
            if waypoints.ndim != 2:
                raise RuntimeError("Planner returned invalid waypoint array shape")
            return [
                {jn: float(q) for jn, q in zip(joint_names, qpos_array)}
                for qpos_array in waypoints
            ]

        raise RuntimeError(f"Unsupported waypoint type from planner: {type(waypoints).__name__}")

    def _waypoints_to_qs_sample(
        self,
        waypoints: list[dict[str, float]],
        joint_names: list[str],
    ) -> np.ndarray:
        if self.motion_manager.pin_robot is None:
            raise RuntimeError("Robot model is not initialized")
        return np.array(
            [
                robot_utils.get_qpos_from_joint_dict(self.motion_manager.pin_robot, wp)
                for wp in waypoints
            ]
        )

    def _linear_interpolation_waypoints(
        self,
        current_q: dict[str, float],
        goal_q: dict[str, float],
        joint_names: list[str],
    ) -> list[dict[str, float]]:
        max_delta = 0.0
        for jn in joint_names:
            if jn in current_q and jn in goal_q:
                max_delta = max(max_delta, abs(goal_q[jn] - current_q[jn]))

        # Roughly <= 0.03 rad step, with minimum 10 and maximum 300 points.
        num_steps = int(np.clip(np.ceil(max_delta / 0.03), 10, 300))
        waypoints: list[dict[str, float]] = []
        for i in range(num_steps + 1):
            alpha = i / max(num_steps, 1)
            wp = {}
            for jn in joint_names:
                if jn in current_q and jn in goal_q:
                    wp[jn] = (1.0 - alpha) * current_q[jn] + alpha * goal_q[jn]
            waypoints.append(wp)
        return waypoints

    def _plan_with_fallback(
        self,
        current_q: dict[str, float],
        goal_q: dict[str, float],
        joint_names: list[str],
    ) -> list[dict[str, float]]:
        """Plan trajectory with OMPL first, then fallback to interpolation.

        OMPL in some environments can fail due to API mismatch with ompl>=2.0.
        """
        planner = MoveToConfigurationTask(
            initial_joint_configuration=current_q,
            motion_manager=self.motion_manager,
            planner_type="ompl",
            visualize=self.visualize,
        )

        try:
            _, qs_sample, _, _, _ = planner.run(
                start_configuration_dict=current_q,
                goal_configuration_dict=goal_q,
                control_frequency=self.control_hz,
                generate_trajectory=True,
                planner_type="ompl",
            )
            return self._normalize_waypoints(qs_sample, joint_names)
        except Exception as e:
            error_text = str(e)
            if "StateValidityCheckerFn" not in error_text and "ompl" not in error_text.lower():
                raise

            logger.warning(
                "OMPL planning is unavailable in this environment. "
                "Falling back to interpolation planner."
            )

            try:
                self.motion_manager.initialize_interpolation_planner()
                motion_plan = self.motion_manager.plan_to_configuration(
                    start_configuration_dict=current_q,
                    goal_configuration_dict=goal_q,
                    planner_type="interpolation",
                )
                return self._normalize_waypoints(motion_plan, joint_names)
            except Exception as interpolation_error:
                logger.warning(
                    "Interpolation planner is unavailable (%s). "
                    "Falling back to linear joint interpolation.",
                    interpolation_error,
                )
                return self._linear_interpolation_waypoints(
                    current_q=current_q,
                    goal_q=goal_q,
                    joint_names=joint_names,
                )

    def run(self, ask_confirm: bool = True) -> None:
        left_target, right_target = self._load_pose()

        assert self.motion_manager.pin_robot is not None
        joint_names = robot_utils.get_joint_names(self.motion_manager.pin_robot)

        self._resolve_self_collision_if_needed(joint_names)

        current_q = self.bot.get_joint_pos_dict(["left_arm", "right_arm"])
        self.motion_manager.set_joint_pos(current_q)

        goal_q = self._create_goal_configuration(left_target, right_target)
        waypoints = self._plan_with_fallback(
            current_q=current_q,
            goal_q=goal_q,
            joint_names=joint_names,
        )
        qs_sample = self._waypoints_to_qs_sample(waypoints, joint_names)

        if self.visualize:
            if self.motion_manager.visualizer is None:
                raise RuntimeError("Visualizer is not initialized")
            self.motion_manager.visualizer.update_motion_plan(
                motion_plan=qs_sample,
                joint_names=joint_names,
                duration=1.0,
            )

        if ask_confirm:
            user_input = input(
                "Press ENTER to execute planned trajectory (or type 'abort' to cancel): "
            )
            if user_input.strip().lower() == "abort":
                logger.info("Execution aborted by user")
                return

        trajectory = waypoints

        logger.info(f"Executing safe trajectory with {len(trajectory)} waypoints")
        self._execute_trajectory(trajectory)
        logger.success(f"Reached target pose from {self.pose_path}")


def main(
    pose_path: str = "poses/Home_position.json",
    control_hz: int = 250,
    visualize: bool = True,
    collision_escape_time: float = 2.0,
    ask_confirm: bool = True,
) -> None:
    with Robot() as robot:
        runner = ArmPositionSafeRunner(
            pose_path=pose_path,
            control_hz=control_hz,
            visualize=visualize,
            collision_escape_time=collision_escape_time,
            bot=robot,
        )
        runner.run(ask_confirm=ask_confirm)


if __name__ == "__main__":
    tyro.cli(main)
