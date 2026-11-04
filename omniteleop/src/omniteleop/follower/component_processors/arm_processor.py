"""Arm component processor."""

import time
from typing import Dict, Any, List
import numpy as np
import pinocchio as pin
import pink
import qpsolvers
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

from omniteleop.follower.component_processors.base_processor import (
    BaseComponentProcessor,
)
from omniteleop.follower.input_handlers.base_handler import (
    RobotCommand,
    CommandMode,
    ArmCommandType,
)
from dexmotion.utils import robot_utils

CALIB_SECONDS = 2.0
SOFT_START_SECONDS = 3.0
SOFT_START_STEP = 0.02
MAX_JOINT_STEP = 0.15

class ArmIK:
    """IK solver helper for a single robot arm using Pink and Pinocchio."""

    def __init__(self, side: str, urdf_path: str):
        self.side = side
        self.prefix = "L" if side == "left" else "R"
        self.arm_joints = [f"{self.prefix}_arm_j{i}" for i in range(1, 8)]
        
        # Load Pinocchio model from URDF
        full = pin.buildModelFromUrdf(urdf_path)
        
        # Set reference configuration for locked joints (torso/head)
        locked_q_vals = {
            "torso_j1": 0.52, "torso_j2": 1.05, "torso_j3": 0.37,
            "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0
        }
        qref = pin.neutral(full)
        for name, val in locked_q_vals.items():
            if name in full.names:
                qref[full.idx_qs[full.getJointId(name)]] = val
                
        # Lock all joints except target arm joints
        lock_ids = [i for i in range(1, full.njoints) if full.names[i] not in self.arm_joints]
        self.model = pin.buildReducedModel(full, lock_ids, qref)
        
        # Add custom elbow center frame to avoid shoulder asymmetry issues
        j4_name = f"{self.prefix}_arm_j4"
        j5_name = f"{self.prefix}_arm_j5"
        l4_name = f"{self.prefix}_arm_l4"
        elbow_center_name = f"{self.prefix}_elbow_center"
        
        if j4_name in self.model.names and j5_name in self.model.names:
            j4 = self.model.getJointId(j4_name)
            j5 = self.model.getJointId(j5_name)
            y_off = float(self.model.jointPlacements[j5].translation[1])
            fid_l4 = self.model.getFrameId(l4_name)
            placement = self.model.frames[fid_l4].placement * pin.SE3(
                np.eye(3), np.array([0.0, y_off, 0.0]))
            self.model.addFrame(pin.Frame(
                elbow_center_name, j4, fid_l4, placement, pin.FrameType.OP_FRAME))
                
        self.data = self.model.createData()
        
        # Define home configuration for active arm
        home_q_vals = {f"{self.prefix}_arm_j1": 1.57 if side == "left" else -1.57}
        self.q_home = pin.neutral(self.model)
        for name, val in home_q_vals.items():
            if name in self.model.names:
                self.q_home[self.model.idx_qs[self.model.getJointId(name)]] = val
                
        self.q_idx = {n: self.model.idx_qs[self.model.getJointId(n)] for n in self.arm_joints}
        self.config = pink.Configuration(self.model, self.data, self.q_home.copy())
        
        # Apply joint limits
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.95
        self.q_min, self.q_max = mid - half, mid + half
        
        # Frames
        self.ee_frame = f"{self.prefix}_ee"
        self.elbow_frame = elbow_center_name
        self.shoulder_frame = f"{self.prefix}_arm_l1"
        
        # Frame and posture tasks
        self.hand_task = FrameTask(
            self.ee_frame, position_cost=3.0, orientation_cost=0.0, lm_damping=1.0
        )
        self.elbow_task = FrameTask(
            self.elbow_frame, position_cost=1.5, orientation_cost=0.0, lm_damping=1.0
        )
        self.posture = PostureTask(cost=0.05, lm_damping=1.0, gain=0.2)
        self.posture.set_target(self.q_home)
        
        self.solver = "quadprog" if "quadprog" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
        
        # Link lengths
        sh_pos = self.frame_pos(self.shoulder_frame)
        el_pos = self.frame_pos(self.elbow_frame)
        ee_pos = self.frame_pos(self.ee_frame)
        self.shoulder_anchor = sh_pos
        self.robot_len = {
            "upper": np.linalg.norm(el_pos - sh_pos),
            "fore": np.linalg.norm(ee_pos - el_pos)
        }
        
    def frame_pos(self, frame):
        return np.asarray(self.config.get_transform_frame_to_world(frame).translation)
        
    def set_targets(self, hand_pos, elbow_pos, hand_rot=None, use_orientation=False):
        cur_h = self.config.get_transform_frame_to_world(self.ee_frame).rotation
        cur_e = self.config.get_transform_frame_to_world(self.elbow_frame).rotation
        
        if use_orientation and hand_rot is not None:
            self.hand_task.orientation_cost = 0.6
            rot = hand_rot
        else:
            self.hand_task.orientation_cost = 0.0
            rot = cur_h
            
        h_se3 = pin.SE3.Identity()
        h_se3.translation = np.asarray(hand_pos, dtype=float)
        h_se3.rotation = np.asarray(rot, dtype=float)
        
        e_se3 = pin.SE3.Identity()
        e_se3.translation = np.asarray(elbow_pos, dtype=float)
        e_se3.rotation = np.asarray(cur_e, dtype=float)
        
        self.hand_task.set_target(h_se3)
        self.elbow_task.set_target(e_se3)
        
    def solve(self, dt):
        IK_SUBSTEPS = 3
        sub_dt = dt / IK_SUBSTEPS
        for _ in range(IK_SUBSTEPS):
            v = solve_ik(self.config, [self.hand_task, self.elbow_task, self.posture], sub_dt, solver=self.solver)
            self.config.integrate_inplace(v, sub_dt)
        q = np.clip(self.config.q, self.q_min, self.q_max)
        self.config.update(q)
        return {n: float(q[i]) for n, i in self.q_idx.items()}
        
    def seed(self, joint_dict):
        q = self.config.q.copy()
        for n, v in joint_dict.items():
            if n in self.q_idx:
                q[self.q_idx[n]] = v
        q = np.clip(q, self.q_min, self.q_max)
        self.config.update(q)

class ArmProcessor(BaseComponentProcessor):
    """Processes arm commands for left or right arm.

    Supports three command types:
    - Joint control: Direct joint position/velocity commands
    - EE pose control: End-effector pose commands (IK handled externally)
    - XSENS control: Target TF coordinates tracked via Pink IK
    """

    def __init__(self, side: str, config, motion_manager, robot_info, teleop_mode, rate = 20):
        """Initialize arm processor."""
        assert side in ["left", "right"], "Arm side must be 'left' or 'right'"
        super().__init__(side, config, motion_manager, robot_info, teleop_mode)

        self.rate = rate

        # Cache joint names
        self.joint_names = self.robot_info.get_component_joints(f"{side}_arm")

        # Get arm-specific motion manager component
        self.arm = (
            motion_manager.left_arm if side == "left" else motion_manager.right_arm
        )

        # Torso pitch compensation configuration
        handler_config = self.config.get("input_handlers", {}).get(teleop_mode, {})
        leader_arms_config = handler_config.get("leader_arms", {})
        self.compensate_torso_pitch = (
            self.robot_info.has_torso
            and leader_arms_config.get("compensate_torso_pitch", False)
        )

        # Safety limits
        self.max_joint_angle_step = np.deg2rad(10)  # Maximum 10 degrees per step

        # Xsens IK attributes
        self.ik = None
        self.xsens_start_time = None
        self.xsens_calib = {"upper": [], "fore": []}
        self.xsens_scale = None
        self.xsens_prev_pub = None
        self.xsens_tracking_started = False
        self.xsens_track_start_time = None

    @property
    def component_type(self) -> str:
        """Component type identifier."""
        return "arm"

    def process(self, input_data: Dict[str, Any], command: RobotCommand) -> bool:
        """Process arm command."""
        command_type = input_data.get("command_type", ArmCommandType.JOINT)

        if command_type == ArmCommandType.JOINT:
            return self._process_joint_command(input_data, command)
        elif command_type == ArmCommandType.EE_POSE:
            return False
        elif command_type == ArmCommandType.XSENS:
            return self._process_xsens_command(input_data, command)

        return False

    def _process_xsens_command(
        self, input_data: Dict[str, Any], command: RobotCommand
    ) -> bool:
        """Process Xsens TF data into joint angles using Pink IK."""
        from loguru import logger

        sh = np.array(input_data.get("upper_arm_pos", []))
        el = np.array(input_data.get("forearm_pos", []))
        ha = np.array(input_data.get("hand_pos", []))
        hand_rot_flat = input_data.get("hand_rot", [])

        if len(sh) != 3 or len(el) != 3 or len(ha) != 3:
            return False

        # Initialize IK solver if not already done
        if self.ik is None:
            try:
                self.ik = ArmIK(self.side, self.robot_info.urdf_path)
                self.xsens_start_time = time.monotonic()
                logger.info(f"Initialized ArmIK for {self.side} arm using {self.robot_info.urdf_path}")
            except Exception as e:
                logger.error(f"Failed to initialize ArmIK for {self.side} arm: {e}")
                return False

        now = time.monotonic()

        # 1. Calibration Phase
        if self.xsens_scale is None:
            elapsed = now - self.xsens_start_time
            
            if elapsed < CALIB_SECONDS:
                v_se = el - sh
                v_eh = ha - el
                self.xsens_calib["upper"].append(np.linalg.norm(v_se))
                self.xsens_calib["fore"].append(np.linalg.norm(v_eh))
                return True
            
            human_upper = float(np.mean(self.xsens_calib["upper"])) if self.xsens_calib["upper"] else 0.0
            human_fore = float(np.mean(self.xsens_calib["fore"])) if self.xsens_calib["fore"] else 0.0
            
            self.xsens_scale = {
                "upper": self.ik.robot_len["upper"] / human_upper if human_upper > 0.05 else 1.0,
                "fore": self.ik.robot_len["fore"] / human_fore if human_fore > 0.05 else 1.0
            }
            logger.info(f"[{self.side} arm autofit] Calibration complete. Scale: {self.xsens_scale}")

        # 2. Tracking Phase
        if not self.xsens_tracking_started:
            current_positions = self.motion_manager.get_joint_pos_dict()
            current_arm_pos = {name: current_positions.get(name, 0.0) for name in self.joint_names}
            self.ik.seed(current_arm_pos)
            self.xsens_prev_pub = {n: float(current_positions.get(n, self.ik.q_home[self.ik.q_idx[n]])) for n in self.joint_names}
            self.xsens_tracking_started = True
            self.xsens_track_start_time = now
            logger.info(f"[{self.side} arm] Soft-start tracking initiated.")

        v_se = el - sh
        v_eh = ha - el
        R_ALIGN = np.eye(3)

        elbow_target = self.ik.shoulder_anchor + self.xsens_scale["upper"] * (R_ALIGN @ v_se)
        hand_target = elbow_target + self.xsens_scale["fore"] * (R_ALIGN @ v_eh)

        hand_rot = None
        if hand_rot_flat and len(hand_rot_flat) == 9:
            hand_rot = R_ALIGN @ np.array(hand_rot_flat).reshape(3, 3)

        self.ik.set_targets(hand_target, elbow_target, hand_rot, use_orientation=False)
        try:
            sol = self.ik.solve(1.0 / self.rate)
        except Exception as e:
            logger.warning(f"[{self.side} arm] IK solve failed: {e}")
            return False

        ramp = min(1.0, (now - self.xsens_track_start_time) / SOFT_START_SECONDS)
        step_max = SOFT_START_STEP + ramp * (MAX_JOINT_STEP - SOFT_START_STEP)

        positions = []
        for name in self.joint_names:
            val = sol[name]
            prev = self.xsens_prev_pub[name]
            diff = val - prev
            if abs(diff) > step_max:
                val = prev + np.clip(diff, -step_max, step_max)
            self.xsens_prev_pub[name] = val
            positions.append(val)

        # Check for collision
        current_joint_dict = self.motion_manager.get_joint_pos_dict()
        for i, name in enumerate(self.joint_names):
            current_joint_dict[name] = positions[i]
        
        qpos = robot_utils.get_qpos_from_joint_dict(
            self.motion_manager.pin_robot, current_joint_dict
        )
        
        is_collision, _, _ = robot_utils.check_collisions_at_state(
            robot=self.motion_manager.pin_robot,
            qpos=qpos,
            distance_padding=0.01,
            stop_at_first_collision=True
        )
        
        if is_collision:
            logger.warning(f"[{self.side} arm] Collision detected!")
            return False

        if self.side == "left":
            self.motion_manager.left_arm.set_joint_pos(positions)
        else:
            self.motion_manager.right_arm.set_joint_pos(positions)

        command.output_components[self.component_name] = {
            "pos": positions,
            "mode": CommandMode.ABSOLUTE.value,
        }
        return True

    def _process_joint_command(
        self, input_data: Dict[str, Any], command: RobotCommand
    ) -> bool:
        """Process joint position command."""
        mode = input_data.get("mode", CommandMode.ABSOLUTE)
        positions = input_data.get("pos", [])
        velocities = input_data.get("vel", [])

        if not positions:
            return False

        if input_data.get("smooth", False):
            return self._process_smooth_absolute(positions, command)

        if mode == CommandMode.RELATIVE:
            current_positions = self.motion_manager.get_joint_pos_dict()
            current_arm_pos = [
                current_positions.get(name, 0) for name in self.joint_names
            ]
            positions = [
                curr + delta for curr, delta in zip(current_arm_pos, positions)
            ]

        if self.compensate_torso_pitch:
            positions = self._compensate_for_torso_pitch(positions)

        if self.side == "left":
            self.motion_manager.left_arm.set_joint_pos(positions)
        else:
            self.motion_manager.right_arm.set_joint_pos(positions)

        output = {
            "pos": positions,
            "mode": CommandMode.ABSOLUTE.value,
        }
        if velocities:
            output["vel"] = velocities

        command.output_components[self.component_name] = output
        return True

    def _compensate_for_torso_pitch(self, positions: List[float]) -> List[float]:
        """Compensate arm positions for torso pitch."""
        torso_pos = self.motion_manager.torso.get_joint_pos()
        torso_pitch = torso_pos[0] + torso_pos[2] - torso_pos[1]

        compensated = positions.copy()
        pitch_adjustment = torso_pitch if self.side == "left" else -torso_pitch
        compensated[0] += pitch_adjustment

        return compensated

    def limit_joint_step(self, target_positions: List[float]) -> np.ndarray:
        """Limit joint position changes for safety."""
        current = self.arm.get_joint_pos()
        target = np.array(target_positions)
        position_diff = target - current

        step_sizes = np.abs(position_diff)
        alphas = np.minimum(1.0, self.max_joint_angle_step / (step_sizes + 1e-10))
        clipped_position = current + alphas * position_diff

        return clipped_position

    def apply_positions(self, positions: List[float]) -> None:
        """Apply joint positions to motion manager."""
        self.arm.set_joint_pos(positions)

    def _process_smooth_absolute(self, target: List[float], command: RobotCommand) -> bool:
        """Smoothly interpolate arm towards target position."""
        current_positions = self.motion_manager.get_joint_pos_dict()
        current = [current_positions.get(name, target[i]) for i, name in enumerate(self.joint_names)]

        new_pos = [current[i] + (target[i] - current[i]) * 0.1 for i in range(len(target))]
        self.arm.set_joint_pos(new_pos)

        command.output_components[self.component_name] = {
            "pos": new_pos,
            "mode": CommandMode.ABSOLUTE.value,
        }
        return True

    def sync_to_robot_state(self, robot_joints: Dict[str, Any]) -> None:
        """Sync arm to robot state."""
        arm_pos = robot_joints.get(self.component_name, [])
        if arm_pos:
            self.arm.set_joint_pos(arm_pos)
