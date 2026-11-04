# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue
import numpy as np

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")

# decimation = 2 ,  sim.dt = 0.005  인 경우 = 100  Hz
# 화면녹화가 30hz라고 가정하면 write_interval_steps = 100 // 30 = 3.3333333333333335 = 3 이 적당
# 4라면 25hz이므로 화면이 약간 끊기는 현상이 발생할수 있음.
_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'camera_sensors': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 5,
}


_return_placeholder = None
_async_queue = None
_async_thread = None
_async_started = False

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """
    global _return_placeholder
    if _return_placeholder is None:
        _return_placeholder = torch.zeros((1, 480, 640, 3))


    _camera_cache['frame_step'] = (_camera_cache['frame_step'] + 1) % max(1, _camera_cache['write_interval_steps'])


    scene_id = id(env.scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(env.scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        
        # Cache sensor objects for faster access
        if hasattr(env.scene, 'sensors') and env.scene.sensors:
            _camera_cache['camera_sensors'] = [
                sensor for name, sensor in env.scene.sensors.items() 
                if name in _camera_cache['available_cameras']
            ]
        else:
            _camera_cache['camera_sensors'] = []
            
        _camera_cache['last_scene_id'] = scene_id


    if _camera_cache['frame_step'] == 0:
        try:
            # Update only the relevant camera sensors
            if _camera_cache['camera_sensors']:
                dt = getattr(env, 'physics_dt', 0.02)
                for sensor in _camera_cache['camera_sensors']:
                    try:
                        sensor.update(dt, force_recompute=False)
                    except Exception:
                        pass
        except Exception:
            pass
    
        # get the camera images
        images = {}
        # env.sim.render()
        

        camera_keys = _camera_cache['camera_keys']
        # Head camera (front camera)
        if "front_camera" in camera_keys:
            head_image = env.scene["front_camera"].data.output["rgb"][0]  # [batch, height, width, 3]

            if head_image.device.type == 'cpu':
                images["head"] = head_image.numpy()
            else:
                images["head"] = head_image.cpu().numpy()
        
        # Left camera (left wrist camera)
        if "left_wrist_camera" in camera_keys:
            left_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
            if left_image.device.type == 'cpu':
                images["wrist_left"] = left_image.numpy()
            else:
                images["wrist_left"] = left_image.cpu().numpy()
        
        # Right camera (right wrist camera)  
        if "right_wrist_camera" in camera_keys:
            right_image = env.scene["right_wrist_camera"].data.output["rgb"][0]
            if right_image.device.type == 'cpu':
                images["wrist_right"] = right_image.numpy()
            else:
                images["wrist_right"] = right_image.cpu().numpy()

        # modify start -------------------------------------------------------------------------------
        if "head_camera" in camera_keys:
            head_image = env.scene["head_camera"].data.output["rgb"][0]  # [batch, height, width, 3]
            if head_image.device.type == 'cpu':
                images["head"] = head_image.numpy()
            else:
                images["head"] = head_image.cpu().numpy()

        # Head right camera (front right camera)
        if "head_right_camera" in camera_keys:
            head_right_image = env.scene["head_right_camera"].data.output["rgb"][0]  # [batch, height, width, 3]
            if head_right_image.device.type == 'cpu':
                images["head_right"] = head_right_image.numpy()
            else:
                images["head_right"] = head_right_image.cpu().numpy()

        # Head depth image (distance to image plane)
        if "head_camera" in camera_keys:
            head_depth_tensor = env.scene["head_camera"].data.output["distance_to_image_plane"][0]
            if head_depth_tensor.device.type == 'cpu':
                depth_map = head_depth_tensor.numpy()
            else:
                depth_map = head_depth_tensor.cpu().numpy()
            
            # Ensure depth_map is 2D (squeeze extra dimensions like [H, W, 1])
            if depth_map.ndim > 2:
                depth_map = np.squeeze(depth_map)

            # 2. 시각화를 위해 정규화 (예: 0~2미터 범위를 0~255로 변환)
            # 값이 너무 크면 검게 보이므로 적절한 최대 거리(max_depth) 설정 필요
            max_depth = 2.0
            depth_viz = np.clip(depth_map / max_depth * 255, 0, 255).astype(np.uint8)

            # 3. 1채널을 3채널로 복사 (서버 호환성 목적)
            depth_viz_3ch = np.stack([depth_viz] * 3, axis=-1)
            images["head_depth"] = depth_viz_3ch

        # center camera
        if "center_camera" in camera_keys:
            center_image = env.scene["center_camera"].data.output["rgb"][0]  # [batch, height, width, 3]
            if center_image.device.type == 'cpu':
                images["center"] = center_image.numpy()
            else:
                images["center"] = center_image.cpu().numpy()

        # Right camera (right wrist camera)
        if "wrist_right_camera" in camera_keys:
            wrist_right_image = env.scene["wrist_right_camera"].data.output["rgb"][0]
            if wrist_right_image.device.type == 'cpu':
                images["wrist_right"] = wrist_right_image.numpy()
            else:
                images["wrist_right"] = wrist_right_image.cpu().numpy()
        # modify end -------------------------------------------------------------------------------
        
        # if no camera with the specified name is found, try other common camera names
        if not images:

            available_cameras = _camera_cache['available_cameras']
            if available_cameras:
                print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
                
                # if there are available cameras, use the first three as head, left, right
                for i, camera_name in enumerate(available_cameras[:3]):
                    camera_image = env.scene[camera_name].data.output["rgb"][0]
                    
                
                    if camera_image.device.type == 'cpu':
                        numpy_image = camera_image.numpy()
                    else:
                        numpy_image = camera_image.cpu().numpy()
                    
                    if i == 0:
                        images["head"] = numpy_image
                    elif i == 1:
                        images["left"] = numpy_image
                    elif i == 2:
                        images["right"] = numpy_image
    
        if images:
            _ensure_async_started()
            try:
                
                if _async_queue.full():
                    _async_queue.get_nowait()
                _async_queue.put_nowait(images)
            except Exception:
                pass
    elif not _camera_cache.get('available_cameras') and _camera_cache['frame_step'] == 0:
         print("[camera_state] No camera images found in the environment")
    
    return _return_placeholder

