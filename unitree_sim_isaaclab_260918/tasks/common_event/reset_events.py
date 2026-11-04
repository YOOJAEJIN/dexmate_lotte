# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Scene reset helpers that are safe for scenes containing kinematic rigid bodies."""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv


def _is_kinematic(rigid_object) -> bool:
    """True when the asset was spawned with ``rigid_props.kinematic_enabled=True``."""
    props = getattr(getattr(rigid_object, "cfg", None), "spawn", None)
    props = getattr(props, "rigid_props", None)
    return bool(getattr(props, "kinematic_enabled", False))


def reset_scene_to_default_skip_kinematic(
    env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False
) -> None:
    """Same as ``isaaclab.envs.mdp.reset_scene_to_default`` but skips kinematic rigid bodies.

    ``reset_scene_to_default`` writes a root velocity to every rigid object in the scene.
    PhysX rejects that for kinematic bodies and logs
    ``PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!`` (one pair of errors per
    kinematic prim). Kinematic markers in this project are re-positioned every step from
    ``sim_main`` via ``write_root_pose_to_sim``, so they do not need to be reset here at all.
    """
    # rigid bodies (non-kinematic only)
    for rigid_object in env.scene.rigid_objects.values():
        if _is_kinematic(rigid_object):
            continue
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
