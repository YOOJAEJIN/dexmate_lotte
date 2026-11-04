
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os
import traceback

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import contextlib
import time
import sys
import signal
import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from teleimager.image_server import run_isaacsim_server
from dds.dds_create import create_dds_objects,create_dds_objects_replay
# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--action_source", type=str, default="dds", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody", "zenoh"],
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--enable_zenoh_hand", type=str, default=None, choices=["gripper", "vertical", "xhand"], help="enable zenoh hand")
parser.add_argument("--disable_gripper_lock", action="store_true", default=False,
                    help="vertical gripper: do not pin slider joint limits after a settled close command")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="/home/unitree/Code/xr_teleoperate/teleop/utils/data", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=100, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default="assets/model/policy.onnx", help="model path")
parser.add_argument("--reward_interval", type=int, default=10, help="step interval for reward calculation")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")

parser.add_argument("--physics_dt", type=float, default=None, help="physics time step, e.g., 0.005")
parser.add_argument("--render_interval", type=int, default=None, help="render interval steps (>=1)")
parser.add_argument("--camera_write_interval", type=int, default=None, help="camera write interval steps (>=1)")


parser.add_argument("--no_render",action="store_true",default=False,help="disable rendering updates entirely (overrides render interval)",)
parser.add_argument("--public_ip",type=str,default="127.0.0.1",help="public ip")
parser.add_argument("--livestream_type", type=int, default=2, help="livestream type (0: no livestream, 1: WebRTC public network, 2:  WebRTC private network)")

parser.add_argument("--solver_iterations", type=int, default=None, help="physx solver iteration count (e.g., 4)")
parser.add_argument("--gravity_z", type=float, default=None, help="override gravity z (e.g., -9.8)")
parser.add_argument("--skip_cvtcolor", action="store_true", default=False, help="skip cv2.cvtColor if upstream already BGR")

parser.add_argument("--camera_jpeg", action="store_true", default=True, help="enable JPEG compression for camera frames")
parser.add_argument("--camera_jpeg_quality", type=int, default=85, help="JPEG quality (1-100)")

parser.add_argument("--physx_substeps", type=int, default=None, help="physx substeps per step")
parser.add_argument("--camera_include", type=str, default="front_camera,left_wrist_camera,right_wrist_camera", help="comma-separated camera names to enable")
parser.add_argument("--camera_exclude", type=str, default="world_camera", help="comma-separated camera names to disable")
parser.add_argument("--external_image_server", action="store_true", default=False, help="use the standalone image server (image_server_main.py) instead of creating one in-process")

parser.add_argument("--env_reward_interval", type=int, default=5, help="environment reward compute interval (steps)")
parser.add_argument("--seed", type=int, default=42, help="environment seed")
parser.add_argument("--ee_mid_marker", action="store_true", default=False,
                    help="DEPRECATED: the green EE-mid marker was removed; flag is accepted and ignored")
parser.add_argument("--ee_mid_z_offset", type=float, default=0.2, help="DEPRECATED: ignored")




# Keyed-action recordings live in the dexcontrol checkout next to this repo (the planning
# scripts and tools/recctl.py are there). Fall back to ./recordings if there is no dexcontrol.
_default_record_dir = os.path.join(os.path.dirname(project_root), "dexcontrol", "recordings")
if not os.path.isdir(os.path.dirname(_default_record_dir)):
    _default_record_dir = os.path.join(project_root, "recordings")
parser.add_argument("--record_dir", type=str, default=_default_record_dir,
                    help="directory for keyed action segments (R=record, P/O/N=play, see tools/action_recorder.py); "
                         "record_status.txt is written next to it")
parser.add_argument("--no_record_keys", action="store_true", default=False,
                    help="disable the keyed action recorder/player wrapper")
parser.add_argument("--record_idle_hold_s", type=float, default=0.5,
                    help="on playback, cap runs of an unchanged action (planning pauses) to this many seconds")
parser.add_argument("--record_play_speed", type=float, default=1.0,
                    help="initial playback speed multiplier (0.25-8); change at runtime with 'recctl.py speed N'")
parser.add_argument("--no_record_video", action="store_true", default=False,
                    help="do not save an mp4 of each playback (default: saved to <record_dir>/videos/)")
parser.add_argument("--record_video_every", type=int, default=4,
                    help="capture one video frame every N env steps (4 -> 25 fps at 0.01 s steps)")
parser.add_argument("--record_video_size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"),
                    help="playback video resolution (dedicated low-res capture of the viewport camera)")
# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.no_render:
    os.environ["LIVESTREAM"] = str(args_cli.livestream_type)
    os.environ["PUBLIC_IP"] = args_cli.public_ip
else:
    os.environ["LIVESTREAM"] = "0"

if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


import pinocchio                 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_step_reward_value,get_current_rewards



_CARD_BOX_TIP_STATUS_PATH = os.path.join(project_root, "card_box_tip_pos.txt")
# Scene keys the purple mass-centre marker follows, in order of preference: the top 35-sheet merged
# block ``card_block_35`` (its rigid-body origin is the block's geometric centre), else the bottom
# loose card ``object`` (older scenes without merged blocks).
_CARD_BOX_MASS_CENTER_KEYS = ("card_block_35", "object")
# Optional offset of the purple *visual* marker in the tracked body's local frame; None = draw it at
# the exact centre. The top block ``card_block_35`` is 35 x 3 mm = 10.5 cm tall, so a ball with a
# radius above 5.25 cm placed at its centre pokes out of the stack's top face (scene cfg radius 0.08
# -> ~2.75 cm visible). (-0.200, 0, 0) would put it on the block's -X, robot-facing face instead.
# The status file always reports the true centre, not the marker position.
_CARD_BOX_MASS_CENTER_MARKER_LOCAL_OFFSET = None
_CARD_BOX_MASS_CENTER_STATUS_PATH = os.path.join(
    project_root, "card_box_mass_center_pos.txt"
)
# card_box.usd z half-extent 0.0015 * scale_z 1000 → long-axis tip in box frame.
_CARD_BOX_TIP_LOCAL = (0.0, 0.0, 1.5)
# Green marker: red tip shifted by this **world-frame** vector (3 cm along +X).
_APRILTAG_STATUS_PATH = os.path.join(project_root, "apriltag_pose.txt")
# Four card-edge markers (Chilsung scene) in robot base frame, read by
# dexcontrol/lotte_chilsung_ik_pink/6.right_lower_finger_to_cyan.py.
_CARD_EDGE_MARKERS_STATUS_PATH = os.path.join(project_root, "card_edge_markers_pos.txt")


def update_apriltag_status(env) -> None:
    """Write the conveyor AprilTag pose in robot-base and world frames.

    The tag is static; the base-frame pose still changes with the robot, so it
    is rewritten every step. Ground truth for detect_apriltag.py::

        # frame=robot_base  x y z qw qx qy qz
        <pos_b> <quat_b>
        # world <pos_w> <quat_w>
        # tag family=tag36h11 id=<id> size=<m>
    """
    try:
        from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul
        from tasks.common_scene.base_scene_pickplace_chilsungcfg import (
            CONVEYOR_APRILTAG_ID,
            CONVEYOR_APRILTAG_POS,
            CONVEYOR_APRILTAG_ROT,
            CONVEYOR_APRILTAG_SIZE,
        )

        robot = env.scene["robot"]
    except Exception:
        return
    try:
        dev, dt = env.device, robot.data.root_pos_w.dtype
        # Prefer the live scene cfg (the env cfg may re-align the tag, e.g. to the robot's x);
        # fall back to the module constants.
        tag_cfg = getattr(getattr(env.cfg, "scene", None), "conveyor_apriltag", None)
        tag_pos = tuple(tag_cfg.init_state.pos) if tag_cfg is not None else CONVEYOR_APRILTAG_POS
        tag_rot = tuple(tag_cfg.init_state.rot) if tag_cfg is not None else CONVEYOR_APRILTAG_ROT
        tag_pos_w = env.scene.env_origins[0:1] + torch.tensor(tag_pos, device=dev, dtype=dt).unsqueeze(0)
        tag_quat_w = torch.tensor(tag_rot, device=dev, dtype=dt).unsqueeze(0)
        root_pos = robot.data.root_pos_w[0:1]
        root_quat = robot.data.root_quat_w[0:1]
        pos_b = quat_apply_inverse(root_quat, tag_pos_w - root_pos)[0].cpu().tolist()
        quat_b = quat_mul(quat_inv(root_quat), tag_quat_w)[0].cpu().tolist()
        pw = tag_pos_w[0].cpu().tolist()
        qw = tag_quat_w[0].cpu().tolist()
        with open(_APRILTAG_STATUS_PATH, "w", encoding="utf-8") as f:
            f.write("# frame=robot_base  x y z qw qx qy qz\n")
            f.write(" ".join(f"{v:.6f}" for v in pos_b + quat_b) + "\n")
            f.write("# world " + " ".join(f"{v:.6f}" for v in pw + qw) + "\n")
            f.write(f"# tag family=tag36h11 id={CONVEYOR_APRILTAG_ID} size={CONVEYOR_APRILTAG_SIZE}\n")
    except Exception:
        pass


def update_card_box_tip_marker(env) -> None:
    """Snap red tip marker to the live +Z face center of card_box_conveyor.

    The box is dynamic and can settle a few cm after spawn; a fixed marker pose
    then looks slightly high in world Z. Tracking the box root keeps the marker
    on the true end-face center. (The former green tip-offset marker was removed 2026-09-26;
    clients derive the face normal from the known world +X direction and the base yaw fitted
    from the red tip + card edge markers.)
    """
    try:
        box = env.scene["card_box_conveyor"]
        marker = env.scene["card_box_conveyor_local_px_marker"]
    except KeyError:
        return

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    tip_local = torch.tensor(
        _CARD_BOX_TIP_LOCAL, device=env.device, dtype=box.data.root_pos_w.dtype
    ).expand(env.num_envs, -1)
    tip_w = box.data.root_pos_w + quat_apply(box.data.root_quat_w, tip_local)
    pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=tip_w.dtype)
    pose[:, 0:3] = tip_w
    pose[:, 3] = 1.0
    marker.write_root_pose_to_sim(pose)

    # Share tip in **robot base frame** (same frame as MotionManager FK / approach_tip).
    try:
        robot = env.scene["robot"]
        tip_b = quat_apply_inverse(
            robot.data.root_quat_w, tip_w - robot.data.root_pos_w
        )
        t0 = tip_b[0].detach().cpu().tolist()
        tw = tip_w[0].detach().cpu().tolist()
        # conveyor box orientation (w, x, y, z) in the robot base frame and in the world: the tip face normal is the
        # box's local +z, so the approach demo can align the held block with the box's *actual* axes
        from isaaclab.utils.math import quat_inv, quat_mul
        q_bw = quat_mul(quat_inv(robot.data.root_quat_w), box.data.root_quat_w)
        qb = q_bw[0].detach().cpu().tolist()
        qw = box.data.root_quat_w[0].detach().cpu().tolist()
        with open(_CARD_BOX_TIP_STATUS_PATH, "w", encoding="utf-8") as f:
            f.write("# frame=robot_base\n")
            f.write(f"{t0[0]:.6f} {t0[1]:.6f} {t0[2]:.6f}\n")
            f.write(f"# world {tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f}\n")
            f.write("# quat_base " + " ".join(f"{v:.6f}" for v in qb) + "  (w x y z, conveyor box orientation in the robot base frame; tip face normal = local +z)\n")
            f.write("# quat_world " + " ".join(f"{v:.6f}" for v in qw) + "\n")
    except Exception:
        pass


def update_card_stack_edge_markers(env) -> None:
    """Snap the four coloured edge markers to the -X edge of the bottom / middle card.

    Marker set and card-local offsets come from ``CARD_EDGE_MARKERS`` in the Chilsung
    scene config. Each marker follows its card's live root pose (the stack is randomised
    on reset and cards move when picked), with the offset rotated by the card orientation.
    Does nothing for scenes without these prims.
    """
    try:
        from tasks.common_scene.base_scene_pickplace_chilsungcfg import CARD_EDGE_MARKERS
    except Exception:
        return

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    rows: list[tuple[str, torch.Tensor]] = []
    for marker_key, (card_key, local, _color) in CARD_EDGE_MARKERS.items():
        try:
            card = env.scene[card_key]
            marker = env.scene[marker_key]
        except KeyError:
            continue
        local_t = torch.tensor(
            local, device=env.device, dtype=card.data.root_pos_w.dtype
        ).expand(env.num_envs, -1)
        pos_w = card.data.root_pos_w + quat_apply(card.data.root_quat_w, local_t)
        pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=pos_w.dtype)
        pose[:, 0:3] = pos_w
        pose[:, 3] = 1.0
        marker.write_root_pose_to_sim(pose)
        rows.append((marker_key, pos_w[0:1]))

    # Share the four points in **robot base frame** (same convention as the tip files).
    if not rows:
        return
    try:
        robot = env.scene["robot"]
        root_pos, root_quat = robot.data.root_pos_w[0:1], robot.data.root_quat_w[0:1]
        with open(_CARD_EDGE_MARKERS_STATUS_PATH, "w", encoding="utf-8") as f:
            f.write("# frame=robot_base  name x y z  "
                    "(mid_py yellow, mid_ny cyan, bottom_py orange, bottom_ny blue)\n")
            for name, pw in rows:
                pb = quat_apply_inverse(root_quat, pw - root_pos)[0].detach().cpu().tolist()
                f.write(f"{name} {pb[0]:.6f} {pb[1]:.6f} {pb[2]:.6f}\n")
            for name, pw in rows:
                w = pw[0].detach().cpu().tolist()
                f.write(f"# world {name} {w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n")
    except Exception:
        pass


def update_card_box_mass_center_marker(env) -> None:
    """Snap the purple marker (if present) to the tracked box centre and write it to disk.

    Tracks the first scene key of ``_CARD_BOX_MASS_CENTER_KEYS`` that exists (default: the top
    35-sheet block ``card_block_35``; its root position is the block's geometric centre). The
    status file is written even when the visual marker prim is missing, so approach demos
    still get robot-base coordinates.
    """
    box = None
    for key in _CARD_BOX_MASS_CENTER_KEYS:
        try:
            box = env.scene[key]
            break
        except KeyError:
            continue
    if box is None:
        return

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    center_w = box.data.root_pos_w

    try:
        marker = env.scene["card_box_mass_center_marker"]
        marker_w = center_w
        if _CARD_BOX_MASS_CENTER_MARKER_LOCAL_OFFSET is not None:
            local_t = torch.tensor(
                _CARD_BOX_MASS_CENTER_MARKER_LOCAL_OFFSET, device=env.device, dtype=center_w.dtype
            ).expand(env.num_envs, -1)
            marker_w = center_w + quat_apply(box.data.root_quat_w, local_t)
        pose = torch.zeros(
            (env.num_envs, 7), device=env.device, dtype=center_w.dtype
        )
        pose[:, 0:3] = marker_w
        pose[:, 3] = 1.0
        marker.write_root_pose_to_sim(pose)
    except KeyError:
        pass

    try:
        robot = env.scene["robot"]
        center_b = quat_apply_inverse(
            robot.data.root_quat_w, center_w - robot.data.root_pos_w
        )
        c0 = center_b[0].detach().cpu().tolist()
        cw = center_w[0].detach().cpu().tolist()
        # block orientation (w, x, y, z): in the robot base frame and in the world, so the approach demo can
        # square the *block* (not just the gripper) to the conveyor face
        from isaaclab.utils.math import quat_inv, quat_mul
        q_bw = quat_mul(quat_inv(robot.data.root_quat_w), box.data.root_quat_w)
        qb = q_bw[0].detach().cpu().tolist()
        qw = box.data.root_quat_w[0].detach().cpu().tolist()
        with open(_CARD_BOX_MASS_CENTER_STATUS_PATH, "w", encoding="utf-8") as f:
            f.write("# frame=robot_base\n")
            f.write(f"{c0[0]:.6f} {c0[1]:.6f} {c0[2]:.6f}\n")
            f.write(f"# world {cw[0]:.6f} {cw[1]:.6f} {cw[2]:.6f}\n")
            f.write(f"# source {key}\n")
            f.write("# quat_base " + " ".join(f"{v:.6f}" for v in qb) + "  (w x y z, block orientation in the robot base frame)\n")
            f.write("# quat_world " + " ".join(f"{v:.6f}" for v in qw) + "\n")
            if _CARD_BOX_MASS_CENTER_MARKER_LOCAL_OFFSET is not None:
                o = _CARD_BOX_MASS_CENTER_MARKER_LOCAL_OFFSET
                f.write(f"# marker_offset_local {o[0]:.3f} {o[1]:.3f} {o[2]:.3f}  (visual ball only; the point above is the true centre)\n")
    except Exception:
        pass


def setup_signal_handlers(controller,dds_manager=None,image_server=None, zenoh_manager=None):
    """set signal handlers"""
    def signal_handler(signum, frame):
        print(f"\nreceived signal {signum}, stopping controller...")
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")
        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
        except Exception as e:
            print(f"Failed to stop DDS: {e}")
        try:
            if zenoh_manager is not None:
                zenoh_manager.stop()
        except Exception as e:
            print(f"Failed to stop Zenoh: {e}")
        try:
            if image_server is not None:
                image_server.stop()
        except Exception as e:
            print(f"Failed to stop image server: {e}")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



def main():
    """main function"""
    # import cProfile
    # import pstats
    # import io
    # profiler = cProfile.Profile()
    # profiler.enable()
    import os
    import atexit
    try:
        os.setpgrp()
        current_pgid = os.getpgrp()
        print(f"Setting process group: {current_pgid}")
        
        def cleanup_process_group():
            try:
                print(f"Cleaning up process group: {current_pgid}")
                import signal
                os.killpg(current_pgid, signal.SIGTERM)
            except Exception as e:
                print(f"Failed to clean up process group: {e}")
        
        atexit.register(cleanup_process_group)
        
    except Exception as e:
        print(f"Failed to set process group: {e}")
    print("=" * 60)
    print("robot control system started")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return
    
    # create environment
    print("\ncreate environment...")
    try:
        # apply performance optimizations to env_cfg before gym.make
        if args_cli.device == "cpu":
            print("[Performance] CPU device detected, applying automatic optimizations...")
            try:
                # Enable flatcache for better performance on CPU
                if hasattr(env_cfg.sim, "physx"):
                    env_cfg.sim.physx.use_flatcache = True
                    print("  - Enabled PhysX Flatcache")
                
                # Lower solver iterations if not specified
                if args_cli.solver_iterations is None and hasattr(env_cfg.sim, "physx"):
                    env_cfg.sim.physx.solver_iteration_count = 4
                    print("  - Set solver_iteration_count to 4")
                
                # Adjust render interval if not specified
                if args_cli.render_interval is None:
                    if args_cli.enable_cameras:
                        env_cfg.sim.render_interval = 4  # moderate rendering for cameras
                    else:
                        env_cfg.sim.render_interval = 50 # minimal rendering
                    print(f"  - Set default render_interval to {env_cfg.sim.render_interval}")
            except Exception as e:
                print(f"[Performance] Warning: Failed to apply some optimizations: {e}")

        # camera view
        if "Isaac-PickPlace-Pantos-VEGA-1P" in args_cli.task:
            env_cfg.viewer.eye = (-1.3, 2.0, 3.0)
            env_cfg.viewer.lookat = (0.0, -0.3, 0.0)
        elif "Isaac-PickPlace-Chilsung-VEGA-1P" in args_cli.task:
            env_cfg.viewer.eye = (-1.5, -2.5, 2.0)
            env_cfg.viewer.lookat = (0.50, -0.1, 0.0)
        else:
            env_cfg.viewer.eye = (3.0, -2.0, 2.0)  # 카메라 위치
            env_cfg.viewer.lookat = (0.0, 0.0, 0.5)  # 카메라가 바라보는 지점

        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)
        try:
            sensors_dict = getattr(env.scene, "sensors", {})
            if sensors_dict:
                print("Sensors in the environment:")
                for name, sensor in sensors_dict.items():
                    print(name, sensor)
                print("="*60)
        except Exception as e:
            print(f"[sim] failed to list sensors: {e}")
        print(f"\ncreate environment success ...")
        try:
            env._reward_interval = max(1, int(args_cli.env_reward_interval))
            env._reward_counter = 0
            env._reward_last = None
            print(f"[env] reward compute interval set to {env._reward_interval} steps")
        except Exception as e:
            print(f"[env] failed to set reward interval: {e}")
        if args_cli.physics_dt is not None:
            try:
                env.sim.set_substep_time(args_cli.physics_dt)
                print(f"[sim] physics dt set to {args_cli.physics_dt}")
            except Exception:
                try:
                    env.sim.dt = args_cli.physics_dt
                    print(f"[sim] physics dt assigned to env.sim.dt={args_cli.physics_dt}")
                except Exception as e:
                    print(f"[sim] failed to set physics dt: {e}")
        headless_mode = bool(getattr(args_cli, "headless", False))
        render_interval = None
        if args_cli.render_interval is not None:
            try:
                render_interval = max(1, int(args_cli.render_interval))
            except Exception as e:
                print(f"[sim] invalid render_interval value {args_cli.render_interval}: {e}")
        try:
            if args_cli.no_render:
                env.sim.render_interval = 1_000_000
                env.sim.render_mode = "offscreen"
                print("[sim] rendering disabled via --no_render")
            elif headless_mode:
                env.sim.render_mode = "offscreen"
                env.sim.render_interval = render_interval or 1
                print(f"[sim] headless offscreen rendering every {env.sim.render_interval} steps")
            elif render_interval is not None:
                env.sim.render_interval = render_interval
                print(f"[sim] render_interval set to {env.sim.render_interval}")
        except Exception as e:
            print(f"[sim] failed to configure rendering: {e}")
        if args_cli.camera_write_interval is not None:
            try:
                import tasks.common_observations.camera_state as cam_state
                cam_state._camera_cache['write_interval_steps'] = max(1, int(args_cli.camera_write_interval))
                print(f"[camera] write interval steps set to {cam_state._camera_cache['write_interval_steps']}")
            except Exception as e:
                print(f"[camera] failed to set write interval: {e}")

        try:
            if args_cli.solver_iterations is not None:
                env.sim.physx.solver_iteration_count = int(args_cli.solver_iterations)
                print(f"[sim] solver_iteration_count={env.sim.physx.solver_iteration_count}")
            if args_cli.physx_substeps is not None:
                try:
                    env.sim.physx.substeps = int(args_cli.physx_substeps)
                except Exception:
                    try:
                        env.sim.set_substeps(int(args_cli.physx_substeps))
                    except Exception:
                        pass
                print(f"[sim] physx_substeps set to {args_cli.physx_substeps}")
            if args_cli.gravity_z is not None:
                g = float(args_cli.gravity_z)
                env.sim.physx.gravity = (0.0, 0.0, g)
                print(f"[sim] gravity set to {env.sim.physx.gravity}")
        except Exception as e:
            print(f"[sim] failed to set physx params: {e}")
        if args_cli.skip_cvtcolor:
            os.environ["CAMERA_SKIP_CVTCOLOR"] = "1"
        try:
            import tasks.common_observations.camera_state as cam_state
            enable_jpeg = bool(args_cli.camera_jpeg) or (os.getenv("CAMERA_JPEG") == "1")
            jpeg_quality = int(args_cli.camera_jpeg_quality if args_cli.camera_jpeg else os.getenv("CAMERA_JPEG_QUALITY", args_cli.camera_jpeg_quality))
            cam_state.set_writer_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=args_cli.skip_cvtcolor)
            include = [n.strip() for n in (args_cli.camera_include or "").split(',') if n.strip()]
            exclude = [n.strip() for n in (args_cli.camera_exclude or "").split(',') if n.strip()]
            try:
                cam_state.set_camera_allowlist(include)
            except Exception:
                pass
            try:
                sensors_dict = getattr(env.scene, "sensors", {})
                for name, sensor in sensors_dict.items():
                    lname = name.lower()
                    if "camera" not in lname:
                        continue
                    if exclude and name in exclude:
                        for attr_name, value in [("enabled", False), ("is_enabled", False)]:
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, value)
                                except Exception:
                                    pass
                        for meth in ("set_active", "disable", "pause"):
                            if hasattr(sensor, meth):
                                try:
                                    getattr(sensor, meth)(False)
                                except Exception:
                                    pass
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
                    elif include and name not in include:
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
            except Exception as e:
                print(f"[camera] failed to tune sensors: {e}")
        except Exception as e:
            print(f"[camera] failed to apply writer options: {e}")
    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        print(traceback.format_exc())
        return
    
    # get robot stiffness and damping parameters from runtime environment
    print("\n" + "="*60)
    print("🔍 Getting robot stiffness and damping parameters from runtime environment")
    print("="*60)
    
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            print("✅ Successfully got robot parameters!")
        else:
            print("⚠️ Failed to get robot parameters, will try again after environment reset")
    except Exception as e:
        print(f"⚠️ Error getting robot parameters: {e}")
    
    print("="*60)
    
    if not getattr(args_cli, "headless", False) and not args_cli.no_render:
        print("\n")
        print("***  Please left-click on the Sim window to activate rendering. ***")
        print("\n")
    else:
        print("\n")
        print("***  Running without GUI; rendering handled offscreen. ***")
        print("\n")
    # reset environment
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(0.75, 0.75, 0.75),
            intensity=500.0,
            # position=(1.0, 2.0, 3.0),
            radius=0.1,
            enabled=True,
            cast_shadows=True
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,                
            focus_distance=1.2
        )
    env.sim.reset()
    env.reset()
    update_card_box_tip_marker(env)
    update_card_box_mass_center_marker(env)
    update_card_stack_edge_markers(env)
    update_apriltag_status(env)

    if args_cli.ee_mid_marker:
        print("[sim] --ee_mid_marker is deprecated: the green EE-mid marker was removed; ignoring.", flush=True)

    # create simplified control configuration
    try:    
        control_config = ControlConfig(
            step_hz=args_cli.step_hz,
            replay_mode=args_cli.replay_data
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return
    
    # create controller
    dds_manager = None
    zenoh_manager = None
    reset_pose_dds = None
    reset_pose_zenoh = None
    sim_state_dds = None
    image_server = None

    if not args_cli.replay_data:
        if args_cli.external_image_server:
            print("========= external image server mode, skip in-process image server (run image_server_main.py) =========")
        else:
            print("========= create image server =========")
            try:
                image_server = run_isaacsim_server()
            except Exception as e:
                print(f"Failed to create image server: {e}")
                return
            print("========= create image server success =========")
        print("========= create communication (DDS/Zenoh) =========")
        try:
            if args_cli.robot_type == "g129" or args_cli.robot_type == "h1_2":
                reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
            elif args_cli.robot_type == "vega_1p":
                from zenoh.zenoh_manager import create_zenoh_manager
                zenoh_manager = create_zenoh_manager(env, args_cli.robot_type, args_cli.enable_zenoh_hand)
                from zenoh.reset_pose_zenoh import ResetPoseCmdZenoh
                reset_pose_zenoh = ResetPoseCmdZenoh()
                reset_pose_zenoh.setup_subscriber()
                reset_pose_zenoh.subscribing = True
        except Exception as e:
            print(f"Failed to create communication: {e}")
            return
        print("========= create communication success =========")
    else:
        print("========= create dds =========")
        try:
            if args_cli.robot_type != "vega_1p":
                create_dds_objects_replay(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider
    
    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        print(f"args_cli.task: {args_cli.task}")
        if args_cli.robot_type == "g129" or args_cli.robot_type == "h1_2":
            if not args_cli.replay_data and ("Wholebody" in args_cli.task or args_cli.enable_wholebody_dds):
                args_cli.action_source = "dds_wholebody"
                args_cli.enable_wholebody_dds = True
                control_config.use_rl_action_mode = True
        
        action_provider = create_action_provider(env,args_cli)
        if action_provider is None:
            print("action provider creation failed, exiting")
            return
        if not args_cli.no_record_keys and not args_cli.replay_data:
            # Keyed per-motion action recording / playback (R / P / O / N / X keys).
            from tools.action_recorder import SegmentRecorder
            action_provider = SegmentRecorder(
                action_provider, env, args_cli.record_dir,
                idle_hold_s=args_cli.record_idle_hold_s, play_speed=args_cli.record_play_speed,
                record_video=not args_cli.no_record_video, video_every=args_cli.record_video_every,
                video_size=tuple(args_cli.record_video_size),
            )
            action_provider.attach_keyboard()
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return
    
    # set action provider
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)
    print("========= create controller success =========")
    
    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")


    # set signal handlers
    if not args_cli.replay_data:
        setup_signal_handlers(controller,dds_manager,image_server, zenoh_manager)
    else:
        setup_signal_handlers(controller)
        
    print("Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).")
    try:
        # start controller - start asynchronous components
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")
        
        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency
        
        
        reward_interval = max(1, args_cli.reward_interval)

        # use torch.inference_mode() and exception suppression
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and controller.is_running:
                current_time = time.time()
                loop_count += 1
                if not args_cli.replay_data:
                    if args_cli.robot_type == "g129" or args_cli.robot_type == "h1_2":
                        try:
                            env_state = env.scene.get_state()
                            env_state_json =  sim_state_to_json(env_state)
                            sim_state = {"init_state":env_state_json,"task_name":args_cli.task}
                        except Exception as e:
                            print(f"Failed to get env state: {e}")
                            raise e
                        
                        try:
                            sim_state_dds.write_sim_state_data(sim_state)
                        except Exception as e:
                            print(f"Failed to write sim state: {e}")
                            raise e
                        try:
                            reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                        except Exception as e:
                            print(f"Failed to get reset pose command: {e}")
                            raise e
                        
                        if reset_pose_cmd is not None:
                            try:
                                reset_category = reset_pose_cmd.get("reset_category")
                                if (args_cli.enable_wholebody_dds and (reset_category == '1' or reset_category == '2')) or (not args_cli.enable_wholebody_dds and reset_category == '1'):
                                    print("reset object")
                                    env_cfg.event_manager.trigger("reset_object_self", env)
                                    reset_pose_dds.write_reset_pose_command(-1)
                                elif reset_category == '2' and not args_cli.enable_wholebody_dds:
                                    print("reset all")
                                    controller.reset()
                                    env_cfg.event_manager.trigger("reset_all_self", env)
                                    reset_pose_dds.write_reset_pose_command(-1)
                            except Exception as e:
                                print(f"Failed to write reset pose command: {e}")
                                raise e
                    elif args_cli.robot_type == "vega_1p":
                        if zenoh_manager is not None:
                            zenoh_manager.publish_joint_states()
                        if reset_pose_zenoh is not None:
                            try:
                                reset_pose_cmd = reset_pose_zenoh.get_reset_pose_command()
                                if reset_pose_cmd is not None:
                                    reset_category = reset_pose_cmd.get("reset_category")
                                    # Handle both string and integer (e.g. '1', 1, '2', 2)
                                    if str(reset_category) == '1':
                                        print("reset object")
                                        env_cfg.event_manager.trigger("reset_object_self", env)
                                        reset_pose_zenoh.write_reset_pose_command(-1)
                                    elif str(reset_category) == '2':
                                        print("reset all")
                                        controller.reset()
                                        env_cfg.event_manager.trigger("reset_all_self", env)
                                        reset_pose_zenoh.write_reset_pose_command(-1)
                            except Exception as e:
                                print(f"Failed to process reset pose command (Zenoh): {e}")
                    
                    # Compute current reward values manually if needed for debugging
                    try:
                        if (loop_count % reward_interval) == 0:
                            pass
                            # current_reward = get_step_reward_value(env)
                    except Exception as e:
                        print(f"奖励计算失败: {e}")
                        pass
                else:
                    if action_provider.get_start_loop() and data_idx<len(data_json_list):
                        print(f"data_idx: {data_idx}")
                        try:
                            sim_state,task_name = action_provider.load_data(data_json_list[data_idx])
                            if task_name!=args_cli.task:
                                raise ValueError(f" The {task_name} in the dataset is different from the {args_cli.task} being executed .")
                        except Exception as e:
                            print(f"Failed to load data: {e}")
                            raise e
                        try:
                            env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                            env.sim.reset()
                            time.sleep(1)
                            action_provider.start_replay()
                            data_idx+=1
                        except Exception as e:
                            print(f"Failed to start replay: {e}")
                            raise e
                # print(f"env_state: {env_state}")
                # calculate instantaneous loop time
                loop_dt = current_time - last_loop_time
                last_loop_time = current_time
                recent_loop_times.append(loop_dt)
                
                # keep recent 100 loop times
                if len(recent_loop_times) > 100:
                    recent_loop_times.pop(0)
                
                # execute control step (in main thread, support rendering)
                controller.step()

                # Keep red tip marker on the box end-face center (box may settle).
                update_card_box_tip_marker(env)
                update_card_box_mass_center_marker(env)
                update_card_stack_edge_markers(env)
                update_apriltag_status(env)

                # print statistics and loop frequency periodically
                if current_time - last_stats_time >= args_cli.stats_interval:
                    # calculate while loop execution frequency
                    elapsed_time = current_time - loop_start_time
                    loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # calculate moving average frequency (based on recent loop times)
                    if recent_loop_times:
                        avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                        moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                        min_loop_time = min(recent_loop_times)
                        max_loop_time = max(recent_loop_times)
                        max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                        min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                    else:
                        moving_avg_frequency = 0
                        min_freq = max_freq = 0
                    
                    print(f"\n=== While loop execution frequency statistics ===")
                    print(f"loop execution count: {loop_count}")
                    print(f"running time: {elapsed_time:.2f} seconds")
                    print(f"overall average frequency: {loop_frequency:.2f} Hz")
                    print(f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)")
                    print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                    print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                    if recent_loop_times:
                        print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                    print(f"=============================")
                    
                    # print_stats(controller)
                    last_stats_time = current_time
       
                # check environment state
                if env.sim.is_stopped():
                    print("\nenvironment stopped")
                    break
                # rate_limiter.sleep(env)
    except KeyboardInterrupt:
        print("\nuser interrupted program")
    
    except Exception as e:
        print(f"\nprogram exception: {e}")
        print(traceback.format_exc())
    
    finally:
        # clean up resources
        print("\nclean up resources...")
        controller.cleanup()
        if zenoh_manager is not None:
            zenoh_manager.stop()
        if image_server is not None:
            image_server.stop()
        env.close()
        print("cleanup completed")
    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats("time")
    # ps.print_stats(30)  

    # print(s.getvalue())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print("Performing final cleanup...")
        
        # Get current process information
        import os
        import subprocess
        import signal
        import time
        
        current_pid = os.getpid()
        print(f"Current main process PID: {current_pid}")
        
        try:
            # Find all related Python processes
            result = subprocess.run(['pgrep', '-f', 'sim_main.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"Found related processes: {pids}")
                
                for pid in pids:
                    if pid and pid != str(current_pid):
                        try:
                            print(f"Terminating child process: {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            print(f"Process {pid} does not exist")
                        except Exception as e:
                            print(f"Failed to terminate process {pid}: {e}")
                
                # Wait for processes to exit
                time.sleep(2)
                
                # Check if there are any remaining processes, force kill them
                result2 = subprocess.run(['pgrep', '-f', 'sim_main.py'], 
                                       capture_output=True, text=True)
                if result2.returncode == 0:
                    remaining_pids = result2.stdout.strip().split('\n')
                    for pid in remaining_pids:
                        if pid and pid != str(current_pid):
                            try:
                                print(f"Force killing process: {pid}")
                                os.kill(int(pid), signal.SIGKILL)
                            except Exception as e:
                                print(f"Failed to force kill process {pid}: {e}")
                                
        except Exception as e:
            print(f"Error during process cleanup: {e}")
        
        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation application: {e}")
            
        print("Program exit completed")
        
        # Force exit
        os._exit(0)

# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint   --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129




# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody  --robot_type g129 --enable_dex3_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody  --robot_type g129 --enable_inspire_dds 


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint --enable_inspire_dds --robot_type h1_2

# python sim_main.py --device cpu --enable_cameras --task Isaac-Place-VEGA-1P-Joint --action_source zenoh --robot_type vega_1p
# python sim_main.py --device cpu --enable_cameras --task Isaac-PickPlace-Bin-VEGA-1P-Joint --action_source zenoh --robot_type vega_1p
# python sim_main.py --device cpu --enable_cameras --task Isaac-PickPlace-Bin-VEGA-1P-Gripper-Joint --action_source zenoh --enable_zenoh_hand gripper --robot_type vega_1p
# python sim_main.py --device cpu --enable_cameras --task Isaac-PickPlace-Bin-VEGA-1P-Xhand-Joint --action_source zenoh --enable_zenoh_hand xhand --robot_type vega_1p