#!/usr/bin/env python3
"""Script to render URDF predefined poses to zero-margin cropped PNG images for frontend UI."""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import yourdfpy

# URDF Path
URDF_PATH = "/home/ubuntu/dex/omniteleop/external/dexmate-urdf/robots/humanoid/vega_1p/vega_1p.urdf"

# Output directory for PNGs
OUTPUT_DIR = Path("/home/ubuntu/dex/omniteleop/src/omniteleop/app/frontend/public/assets/poses")


def _load_predefined_poses():
    """Load predefined poses from predefined_poses.json at the project root.

    Same source as app_backend.py: searches the current working directory
    first, then the repository root relative to this script (scripts/ is
    directly under the project root).
    """
    candidates = [
        Path.cwd() / "predefined_poses.json",
        Path(__file__).resolve().parents[1] / "predefined_poses.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                poses = json.load(f)
            print(f"Loaded {len(poses)} predefined poses from {path}")
            return poses
    raise FileNotFoundError(
        f"predefined_poses.json not found (looked in: {[str(c) for c in candidates]})"
    )


# Predefined poses shared with app_backend.py (predefined_poses.json)
PREDEFINED_POSES = _load_predefined_poses()

LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
HEAD_JOINTS = [f"head_j{i}" for i in range(1, 4)]
TORSO_JOINTS = [f"torso_j{i}" for i in range(1, 4)]

def pose_to_joint_dict(robot, pose_cfg):
    # Initialize ALL actuated joints to 0.0 first so missing components reset to neutral 0
    joint_dict = {name: 0.0 for name in robot.actuated_joint_names}

    if "left_arm" in pose_cfg:
        for name, val in zip(LEFT_ARM_JOINTS, pose_cfg["left_arm"]):
            joint_dict[name] = val
    if "right_arm" in pose_cfg:
        for name, val in zip(RIGHT_ARM_JOINTS, pose_cfg["right_arm"]):
            joint_dict[name] = val
    if "head" in pose_cfg:
        for name, val in zip(HEAD_JOINTS, pose_cfg["head"]):
            joint_dict[name] = val
    if "torso" in pose_cfg:
        for name, val in zip(TORSO_JOINTS, pose_cfg["torso"]):
            joint_dict[name] = val
    return joint_dict

def render_pose(robot, pose_name, joint_dict, output_file):
    robot.update_cfg(joint_dict)
    scene = robot.scene
    combined_mesh = scene.dump(concatenate=True)
    
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Hide grid and axes for clean icon look
    ax.axis('off')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    
    vertices = combined_mesh.vertices
    faces = combined_mesh.faces
    normals = combined_mesh.face_normals

    # Directional light source from front-top-right
    light_dir = np.array([0.4, 0.4, 0.82])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Calculate diffuse (0.6) + ambient (0.4) shading
    dot_products = np.dot(normals, light_dir)
    intensity = 0.4 + 0.6 * np.clip(dot_products, 0.0, 1.0)
    
    # Base color: solid metallic silver / light slate (#cbd5e1 to #f1f5f9)
    base_color = np.array([0.84, 0.88, 0.93])
    face_colors = intensity[:, np.newaxis] * base_color
    face_colors = np.clip(face_colors, 0.0, 1.0)

    # Render ALL faces to guarantee 100% solid surface without holes
    mesh_collection = Poly3DCollection(vertices[faces], alpha=1.0)
    mesh_collection.set_facecolor(face_colors)
    mesh_collection.set_edgecolor('none')
    
    ax.add_collection3d(mesh_collection)
    
    # Calculate bounding box
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    max_range = (max_bounds - min_bounds).max() / 2
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # View angle: front-isometric looking at robot
    ax.view_init(elev=15, azim=-60)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

    # Auto-crop transparent margins tightly using Pillow getbbox()
    img = Image.open(output_file)
    bbox = img.getbbox()
    if bbox:
        cropped_img = img.crop(bbox)
        cropped_img.save(output_file)
        print(f"Saved zero-margin {pose_name} thumbnail ({img.size} -> {cropped_img.size}) to {output_file}")
    else:
        print(f"Saved {pose_name} thumbnail to {output_file}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading URDF from {URDF_PATH}...")
    robot = yourdfpy.URDF.load(URDF_PATH)
    
    for pose_name, pose_cfg in PREDEFINED_POSES.items():
        joint_dict = pose_to_joint_dict(robot, pose_cfg)
        out_path = OUTPUT_DIR / f"{pose_name}.png"
        render_pose(robot, pose_name, joint_dict, out_path)

if __name__ == "__main__":
    main()
