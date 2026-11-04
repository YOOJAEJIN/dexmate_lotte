import sys
import termios
import tty
from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


def wait_for_space():
    """Wait for user to press space bar before continuing."""
    logger.info("Press SPACE to continue to next waypoint (or 'q' to quit)...")
    
    # Save terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        # Set terminal to raw mode
        tty.setraw(fd)
        
        while True:
            char = sys.stdin.read(1)
            if char == ' ':  # Space bar
                logger.info("Continuing to next waypoint...")
                break
            elif char.lower() == 'q':  # Quit
                logger.warning("Quit requested by user")
                raise KeyboardInterrupt
            elif char == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def parse_trajectory_file(filepath: str) -> list[dict[str, Any]]:
    """Parse the trajectory file.

    Supported step types (detected from keyword on first line):

    TORSO step — move torso joints directly (values in degrees):
        q1, q2, q3, TORSO
        duration

    JOINT step — send arm joints directly without IK (values in radians):
        j1, j2, j3, j4, j5, j6, j7, JOINT   # left arm
        j1, j2, j3, j4, j5, j6, j7, JOINT   # right arm
        duration

    CHASSIS step — drive the mobile base (vx m/s forward, vy m/s left, wz rad/s CCW):
        vx, vy, wz, CHASSIS
        duration

    RAISE step — lift one or both arms in +Z until force threshold is reached:
        RAISE, left|right|both[, step_size_m[, max_steps[, force_threshold_N]]]

    FORCELOWER step — lower one arm in −Z until force exceeds threshold (contact):
        FORCELOWER, left|right[, step_size_m[, max_steps[, force_threshold_N]]]

    GRIPPER step — control the KORAS grippers (both hands by default):
        OPEN, GRIPPER[, left|right|both]
        CLOSE, GRIPPER[, left|right|both]
        position, GRIPPER[, left|right|both]
        GRIPPER-OPEN / GRIPPER-CLOSE          # legacy form, acts on both

    ROTATE_OBJ step — rotate a bimanually grasped object as a rigid body.
    Both gripper poses rotate together about the axis through the midpoint
    of the two grippers (arm_center frame), in small IK sub-steps so the
    hand-to-hand transform is preserved throughout the motion:
        ROTATE_OBJ, x|y|z, angle_deg[, n_steps]
        duration
    axis x = forward (side-by-side grasp → vertical); +angle about x lifts
    the LEFT hand (right-hand rule). n_steps defaults to one per ~3°.

    IK step — solve IK for Cartesian end-effector targets (arm_center frame):
        x, y, z, roll, pitch, yaw[, BOX|REL]  # left arm
        x, y, z, roll, pitch, yaw[, BOX|REL]  # right arm
        duration

    All step types accept an optional pause_after (seconds) as a final line.

    Args:
        filepath: Path to the trajectory file.

    Returns:
        List of waypoint dicts, each with a 'type' key ('torso'|'joint'|'ik').
    """
    waypoints = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.strip() for line in lines
             if line.strip() and not line.strip().startswith('#')]

    def _try_pause(idx: int) -> tuple[float, int]:
        """Try to read an optional pause_after at lines[idx]; return (value, new_i)."""
        if idx < len(lines) and ',' not in lines[idx]:
            try:
                return float(lines[idx].strip()), idx + 1
            except ValueError:
                pass
        return 0.0, idx

    i = 0
    while i < len(lines):
        try:
            first_parts = [x.strip() for x in lines[i].split(',')]

            # ── TORSO step: "q1, q2, q3, TORSO[, max_vel]" ───────────────────
            if len(first_parts) in (4, 5) and first_parts[3].upper() == 'TORSO':
                joint_angles_deg = [float(x) for x in first_parts[:3]]
                max_vel = float(first_parts[4]) if len(first_parts) == 5 else None
                duration = float(lines[i + 1].strip())
                pause_after, i = _try_pause(i + 2)
                waypoints.append({
                    'type': 'torso',
                    'joint_angles_deg': joint_angles_deg,
                    'max_vel': max_vel,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── HEAD step: "h1, h2, h3, HEAD" (degrees) ──────────────────────
            # Head joint order is [pitch, yaw, roll]; roll is the 3rd value.
            # A bare '-' keeps the current angle for that joint. Unlike other
            # steps, the HEAD pose is NOT re-aligned afterwards, so it only
            # holds until the next torso/arm waypoint runs align_head_to_forward.
            if len(first_parts) == 4 and first_parts[3].upper() == 'HEAD':
                joint_angles_deg = [np.nan if x.strip() == '-' else float(x) for x in first_parts[:3]]
                duration = float(lines[i + 1].strip())
                pause_after, i = _try_pause(i + 2)
                waypoints.append({
                    'type': 'head',
                    'joint_angles_deg': joint_angles_deg,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── JOINT step: "j1..j7, JOINT" for both arms ─────────────────────
            if len(first_parts) == 8 and first_parts[7].upper() == 'JOINT':
                left_joints = np.array([np.nan if x == '-' else float(x) for x in first_parts[:7]])
                right_parts = [x.strip() for x in lines[i + 1].split(',')]
                right_joints = np.array([np.nan if x == '-' else float(x) for x in right_parts[:7]])
                # Optional concurrent TORSO target before the duration line.
                # "q1, q2, q3, TORSO[, vel]" — 5th field = torso speed (rad/s).
                torso_target_deg = None
                torso_max_vel = None
                dur_idx = i + 2
                nxt = [x.strip() for x in lines[dur_idx].split(',')]
                if len(nxt) in (4, 5) and nxt[3].upper() == 'TORSO':
                    torso_target_deg = [float(x) for x in nxt[:3]]
                    torso_max_vel = float(nxt[4]) if len(nxt) == 5 else None
                    dur_idx += 1
                duration = float(lines[dur_idx].strip())
                pause_after, i = _try_pause(dur_idx + 1)
                waypoints.append({
                    'type': 'joint',
                    'left_arm_joints': left_joints,
                    'right_arm_joints': right_joints,
                    'torso_target_deg': torso_target_deg,
                    'torso_max_vel': torso_max_vel,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── RAISE step: "RAISE, side[, step_size[, max_steps[, force_threshold]]]" ──
            if len(first_parts) >= 2 and first_parts[0].upper() == 'RAISE':
                side            = first_parts[1].strip().lower()
                step_size       = float(first_parts[2]) if len(first_parts) > 2 else None
                max_steps       = int(first_parts[3])   if len(first_parts) > 3 else None
                force_threshold = float(first_parts[4]) if len(first_parts) > 4 else None
                pause_after, i  = _try_pause(i + 1)
                waypoints.append({
                    'type': 'raise',
                    'side': side,
                    'step_size': step_size,
                    'max_steps': max_steps,
                    'force_threshold': force_threshold,
                    'pause_after': pause_after,
                })
                continue

            # ── FORCELOWER step: "FORCELOWER, side[, step_size[, max_steps[, force_threshold]]]" ──
            # Incrementally lowers one arm (−Z in arm_center frame) until force
            # exceeds threshold (contact) or max_steps is reached.
            if len(first_parts) >= 2 and first_parts[0].upper() == 'FORCELOWER':
                side            = first_parts[1].strip().lower()
                step_size       = float(first_parts[2]) if len(first_parts) > 2 else None
                max_steps       = int(first_parts[3])   if len(first_parts) > 3 else None
                force_threshold = float(first_parts[4]) if len(first_parts) > 4 else None
                pause_after, i  = _try_pause(i + 1)
                waypoints.append({
                    'type': 'forcelower',
                    'side': side,
                    'step_size': step_size,
                    'max_steps': max_steps,
                    'force_threshold': force_threshold,
                    'pause_after': pause_after,
                })
                continue

            # ── GRIPPER legacy step: "GRIPPER-OPEN" or "GRIPPER-CLOSE" ─────
            # Acts on BOTH grippers (KORAS gripper on each hand).
            if len(first_parts) == 1 and first_parts[0].upper() in (
                'GRIPPER-OPEN',
                'GRIPPER-CLOSE',
            ):
                command = first_parts[0].split('-', 1)[1].lower()
                pause_after, i = _try_pause(i + 1)
                waypoints.append({
                    'type': 'gripper',
                    'command': command,
                    'side': 'both',
                    'pause_after': pause_after,
                })
                continue

            # ── GRIPPER step: "OPEN|CLOSE|position, GRIPPER[, left|right|both]"
            # Optional 3rd field picks the hand; default is both simultaneously.
            if len(first_parts) in (2, 3) and first_parts[1].upper() == 'GRIPPER':
                side = first_parts[2].lower() if len(first_parts) == 3 else 'both'
                if side not in ('left', 'right', 'both'):
                    raise ValueError(
                        f"Line {i+1}: GRIPPER side must be left, right, or both"
                    )
                gripper_value = first_parts[0].upper()
                waypoint = {
                    'type': 'gripper',
                    'side': side,
                    'pause_after': 0.0,
                }
                if gripper_value in ('OPEN', 'CLOSE'):
                    waypoint['command'] = gripper_value.lower()
                else:
                    position = int(first_parts[0])
                    if not 0 <= position <= 1000:
                        raise ValueError(
                            f"Line {i+1}: GRIPPER position must be in range 0~1000"
                        )
                    waypoint['command'] = 'set-position'
                    waypoint['position'] = position

                pause_after, i = _try_pause(i + 1)
                waypoint['pause_after'] = pause_after
                waypoints.append(waypoint)
                continue

            # ── CHASSIS step: "vx, vy, wz, CHASSIS" ──────────────────────────────
            if len(first_parts) == 4 and first_parts[3].upper() == 'CHASSIS':
                vx = float(first_parts[0])
                vy = float(first_parts[1])
                wz = float(first_parts[2])
                duration = float(lines[i + 1].strip())
                pause_after, i = _try_pause(i + 2)
                waypoints.append({
                    'type': 'chassis',
                    'vx': vx,
                    'vy': vy,
                    'wz': wz,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── GRAB step: "GRAB, <step_size_m>, <max_steps>" ─────────────────
            # Moves both arms inward along Y (left: −Y, right: +Y) by step_size
            # each iteration until either hand reports force ≥ threshold or the
            # hand separation drops below the configured minimum.
            if len(first_parts) >= 1 and first_parts[0].upper() == 'GRAB':
                step_size = float(first_parts[1]) if len(first_parts) > 1 else 0.005
                max_steps = int(first_parts[2]) if len(first_parts) > 2 else 100
                pause_after, i = _try_pause(i + 1)
                waypoints.append({
                    'type': 'grab',
                    'step_size': step_size,
                    'max_steps': max_steps,
                    'pause_after': pause_after,
                })
                continue

            # ── FORWARD step: "FORWARD[, left|right], <step_size_m>, <max_steps>" ──
            # Moves the chosen hand forward (+X in arm_center frame) by
            # step_size each iteration until that hand reports force ≥
            # threshold or max_steps is reached. The other hand holds its pose.
            # The side keyword is optional and defaults to 'right' (backward
            # compatible with "FORWARD, <step_size>, <max_steps>").
            if len(first_parts) >= 1 and first_parts[0].upper() == 'FORWARD':
                rest = first_parts[1:]
                side = 'right'
                if rest and rest[0].lower() in ('left', 'right'):
                    side = rest[0].lower()
                    rest = rest[1:]
                step_size = float(rest[0]) if len(rest) > 0 else 0.001
                max_steps = int(rest[1]) if len(rest) > 1 else 300
                pause_after, i = _try_pause(i + 1)
                waypoints.append({
                    'type': 'forward',
                    'side': side,
                    'step_size': step_size,
                    'max_steps': max_steps,
                    'pause_after': pause_after,
                })
                continue

            # ── REACH step: "REACH, <step_size_m>, <max_steps>" ──────────────
            # Extends the RIGHT hand forward (+X in arm_center frame) by
            # step_size each iteration until the right hand reports force ≥
            # threshold or max_steps is reached. Left hand holds its pose.
            if len(first_parts) >= 1 and first_parts[0].upper() == 'REACH':
                step_size = float(first_parts[1]) if len(first_parts) > 1 else 0.001
                max_steps = int(first_parts[2]) if len(first_parts) > 2 else 300
                pause_after, i = _try_pause(i + 1)
                waypoints.append({
                    'type': 'reach',
                    'step_size': step_size,
                    'max_steps': max_steps,
                    'pause_after': pause_after,
                })
                continue

            # ── ROTATE_OBJ step: "ROTATE_OBJ, x|y|z, angle_deg[, n_steps]" ────
            # Rigid-body rotation of BOTH grippers about the axis through the
            # midpoint of the two gripper positions (arm_center frame). Next
            # line is the duration; optional pause_after line after that.
            if len(first_parts) >= 3 and first_parts[0].upper() == 'ROTATE_OBJ':
                axis = first_parts[1].strip().lower()
                if axis not in ('x', 'y', 'z'):
                    raise ValueError(
                        f"Line {i+1}: ROTATE_OBJ axis must be x, y, or z"
                    )
                angle_deg = float(first_parts[2])
                n_steps = (int(first_parts[3]) if len(first_parts) > 3
                           else max(2, int(round(abs(angle_deg) / 3.0))))
                duration = float(lines[i + 1].strip())
                pause_after, i = _try_pause(i + 2)
                waypoints.append({
                    'type': 'rotate_obj',
                    'axis': axis,
                    'angle_deg': angle_deg,
                    'n_steps': n_steps,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── IK step: "x, y, z, roll, pitch, yaw[, BOX|MAGAZINE|HANDLE|REL]" ──
            # A bare '-' token means "keep current EE value for this field".
            # BOX/MAGAZINE/HANDLE: x,y are offsets from the detected object's
            # position (z is taken from the file as an absolute value).
            def _parse_val(s: str) -> float:
                return np.nan if s.strip() == '-' else float(s)

            left_use_box = False
            left_use_magazine = False
            left_use_handle = False
            left_use_rel = False
            if len(first_parts) == 7:
                flag = first_parts[6].upper()
                left_use_box = flag == 'BOX'
                left_use_magazine = flag == 'MAGAZINE'
                left_use_handle = flag == 'HANDLE'
                left_use_rel = flag == 'REL'
                first_parts = first_parts[:6]
            left_data = [_parse_val(x) for x in first_parts]
            if len(left_data) != 6:
                raise ValueError(f"Line {i+1}: Expected 6 values for left pose")

            right_parts = [x.strip() for x in lines[i + 1].split(',')]
            right_use_box = False
            right_use_magazine = False
            right_use_handle = False
            right_use_rel = False
            if len(right_parts) == 7:
                flag = right_parts[6].upper()
                right_use_box = flag == 'BOX'
                right_use_magazine = flag == 'MAGAZINE'
                right_use_handle = flag == 'HANDLE'
                right_use_rel = flag == 'REL'
                right_parts = right_parts[:6]
            right_data = [_parse_val(x) for x in right_parts]
            if len(right_data) != 6:
                raise ValueError(f"Line {i+2}: Expected 6 values for right pose")

            # Optional concurrent TORSO target: a "q1, q2, q3, TORSO[, vel]" line
            # placed between the right-arm line and the duration moves the torso
            # AT THE SAME TIME as the arms reach their IK targets (degrees).
            # The optional 5th field sets the torso joint speed in rad/s; if
            # omitted, speed is auto-sized to arrive within the duration.
            torso_target_deg = None
            torso_max_vel = None
            dur_idx = i + 2
            nxt = [x.strip() for x in lines[dur_idx].split(',')]
            if len(nxt) in (4, 5) and nxt[3].upper() == 'TORSO':
                torso_target_deg = [float(x) for x in nxt[:3]]
                torso_max_vel = float(nxt[4]) if len(nxt) == 5 else None
                dur_idx += 1

            duration = float(lines[dur_idx].strip())
            pause_after, i = _try_pause(dur_idx + 1)
            left_arr  = np.array(left_data,  dtype=float)
            right_arr = np.array(right_data, dtype=float)
            waypoints.append({
                'type': 'ik',
                'left_pose': {
                    'position': left_arr[:3],
                    'rpy': left_arr[3:6],
                    'use_box': left_use_box,
                    'use_magazine': left_use_magazine,
                    'use_handle': left_use_handle,
                    'use_rel': left_use_rel,
                    'use_cur': bool(np.any(np.isnan(left_arr))),
                },
                'right_pose': {
                    'position': right_arr[:3],
                    'rpy': right_arr[3:6],
                    'use_box': right_use_box,
                    'use_magazine': right_use_magazine,
                    'use_handle': right_use_handle,
                    'use_rel': right_use_rel,
                    'use_cur': bool(np.any(np.isnan(right_arr))),
                },
                'torso_target_deg': torso_target_deg,
                'torso_max_vel': torso_max_vel,
                'duration': duration,
                'pause_after': pause_after,
            })

        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing trajectory at line {i+1}: {e}")
            break

    logger.info(f"Parsed {len(waypoints)} waypoints from trajectory file")
    return waypoints


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw angles to rotation matrix (extrinsic XYZ convention).
    
    Args:
        roll: Rotation around X-axis (radians).
        pitch: Rotation around Y-axis (radians).
        yaw: Rotation around Z-axis (radians).
        
    Returns:
      3x3 rotation matrix.
    """
    # Extrinsic XYZ rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return rot.as_matrix()


def align_head_to_forward(bot, wait_time: float = 6.0) -> None:
    """Adjust head pitch so the robot looks forward in the world frame.

    Keeps torso_pitch_deg + (-head_pitch_deg) ≈ 90 deg, meaning the
    head faces the horizon regardless of how much the torso is tilted.

    Args:
        bot: Connected Robot instance.
        wait_time: Maximum time (seconds) to wait for the head to reach
                   the target position.
    """
    forward_sum_deg = 70.0
    torso_pitch_deg = float(np.rad2deg(bot.torso.pitch_angle))
    current_head_pos = np.asarray(bot.head.get_joint_pos(), dtype=float)
    print(current_head_pos)
    target_head_pos = np.zeros_like(current_head_pos)
    target_head_pos[0] = np.deg2rad(torso_pitch_deg - forward_sum_deg)

    head_error = target_head_pos - current_head_pos
    head_kp = 0.6
    head_min_vel = 0.02
    head_max_vel = 1.0
    head_joint_vel = np.clip(np.abs(head_error) * head_kp, head_min_vel, head_max_vel)

    bot.head.set_joint_pos_vel(
        joint_pos=target_head_pos,
        joint_vel=head_joint_vel,
        wait_time=wait_time,
        exit_on_reach=True,
    )
