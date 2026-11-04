"""Simple data schemas for communication between components.

These dataclasses define the structure of messages passed via Zenoh.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ExoJointData:
    """Joint positions and velocities from the exoskeleton."""

    timestamp_ns: int
    left_arm_pos: List[float] = field(
        default_factory=list
    )  # Left arm joint positions in radians
    left_arm_vel: List[float] = field(
        default_factory=list
    )  # Left arm joint velocities in rad/s
    right_arm_pos: List[float] = field(
        default_factory=list
    )  # Right arm joint positions in radians
    right_arm_vel: List[float] = field(
        default_factory=list
    )  # Right arm joint velocities in rad/s

@dataclass
class JoyConData:
    """JoyCon controller inputs."""

    timestamp_ns: int
    left: Dict[str, Any]  # Left controller data
    right: Dict[str, Any]  # Right controller data

@dataclass
class SafeJointCommand:
    """Safety-validated commands for the robot."""

    timestamp_ns: int
    components: Dict[str, Dict[str, List[float]]]  # Component commands
    safety_flags: Dict[str, bool] = field(default_factory=dict)

@dataclass
class ControllerState:
    """Detailed controller input state."""

    trigger: bool = False
    squeeze: bool = False
    touchpad: bool = False
    thumbstick: bool = False
    aButton: bool = False
    bButton: bool = False

    triggerValue: float = 0.0
    squeezeValue: float = 0.0
    touchpadValue: List[float] = field(default_factory=lambda: [0.0, 0.0])
    thumbstickValue: List[float] = field(default_factory=lambda: [0.0, 0.0])
    aButtonValue: bool = False
    bButtonValue: bool = False

@dataclass
class PoseData:
    """Pose data from the VR headset."""

    timestamp_ns: int
    left: Dict[str, Any]
    right: Dict[str, Any]
    leftState: Optional[ControllerState] = None
    rightState: Optional[ControllerState] = None

@dataclass
class XsensArmData:
    """Position and orientation of Xsens arm segments relative to pelvis."""

    upper_arm_pos: List[float] = field(default_factory=list) # [x, y, z]
    forearm_pos: List[float] = field(default_factory=list)   # [x, y, z]
    hand_pos: List[float] = field(default_factory=list)      # [x, y, z]
    hand_rot: List[float] = field(default_factory=list)      # 3x3 rotation matrix flattened (9 floats)
    valid: bool = False

@dataclass
class XsensRosData:
    """Combined Xsens TF data for both arms."""

    timestamp_ns: int
    left: XsensArmData = field(default_factory=XsensArmData)
    right: XsensArmData = field(default_factory=XsensArmData)

