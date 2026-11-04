"""Configuration file for Pink IK solver.

This file contains paths and configuration parameters for the robot and trajectories.
"""

import os

# URDF file path — dexmate_urdf 패키지 위치에서 동적 해석.
# (호스트 miniconda3 / 컨테이너 /opt/venv 등 환경 무관하게 동작)
import dexmate_urdf

URDF_PATH = os.path.join(
    os.path.dirname(dexmate_urdf.__file__),
    "robots/humanoid/vega_1p/vega_1p_gripper.urdf",
)

# Trajectory file paths (relative to custom_examples directory)
DEFAULT_TRAJECTORY_FILE = "trajectories/VegaTask1-021-Grab.txt"

# IK solver parameters
IK_DT = 0.01  # Time step for IK iterations
IK_MAX_ITERS = 500  # Maximum IK iterations
IK_CONVERGENCE_THRESHOLD = 1e-4  # Convergence threshold

# Task costs
LEFT_EE_POSITION_COST = 2.0  # [cost] / [m]
LEFT_EE_ORIENTATION_COST = 1.0  # [cost] / [rad]
RIGHT_EE_POSITION_COST = 2.0  # [cost] / [m]
RIGHT_EE_ORIENTATION_COST = 1.0  # [cost] / [rad]
POSTURE_COST = 1e-3  # [cost] / [rad]

# Control parameters
CONTROL_DT = 0.02  # Control loop time step (50 Hz)

# Box detection parameters
BOX_DETECTION_MAX_ATTEMPTS = 10  # Maximum attempts to detect box
CAMERA_TIMEOUT = 5.0  # Camera activation timeout (seconds)

# QP solver preference (will use first available if preferred not found)
PREFERRED_QP_SOLVER = "daqp"

# Force-based grasp parameters (used by GRAB trajectory step type)
GRAB_FORCE_THRESHOLD_N = 10.0   # Stop squeezing when either hand exceeds this force (Newtons)
GRAB_MIN_HAND_SEPARATION_M = 0.25  # Back off one step when Y-separation between hands drops below this (metres)

# Force-based reach-forward parameters (used by FORWARD trajectory step type)
FORWARD_FORCE_THRESHOLD_N = 3.0  # Stop reaching forward when the moving hand exceeds this force (Newtons)

# Force-based raise parameters (used by RAISE trajectory step type)
RAISE_FORCE_THRESHOLD_N = 30.0  # Stop raising when force exceeds this value (Newtons)
RAISE_STEP_SIZE_M = 0.005       # Default step size per iteration (metres)
RAISE_MAX_STEPS = 300           # Default maximum raise iterations

# Force-based lower parameters (used by FORCELOWER trajectory step type)
FORCELOWER_FORCE_THRESHOLD_N = 15.0  # Stop lowering once force exceeds this value (Newtons)
FORCELOWER_STEP_SIZE_M = 0.001       # Default step size per iteration (metres)
FORCELOWER_MAX_STEPS = 300           # Default maximum lower iterations

# Force threshold during IK motion steps — pauses and waits for SPACE when exceeded
IK_FORCE_PAUSE_THRESHOLD_N = 300

# Force threshold during ROTATE_OBJ steps — force is tared at rotation start,
# so this measures binding/twist BUILD-UP on the grasped object, not the
# static grasp load. Pauses and waits for SPACE when exceeded.
ROTATE_OBJ_FORCE_PAUSE_N = 50.0
