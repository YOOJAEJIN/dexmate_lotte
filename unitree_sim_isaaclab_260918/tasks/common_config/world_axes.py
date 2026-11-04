# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""World-frame coordinate axes: a visual-only RGB (= XYZ) triad at the world origin.

Red = +X, green = +Y, blue = +Z, white ball = origin (0, 0, 0).  The prims are plain
meshes (no rigid body, no collider), so they never touch physics or the robot.

Usage inside a scene ``@configclass`` body (same ``locals()`` trick the card stack uses)::

    from tasks.common_config import world_axes_cfgs
    ...
    locals().update(world_axes_cfgs())

The prims live at ``/World/WorldAxes`` (not under ``/World/envs/env_*``), so they mark
the true world frame even with several envs.  Set the environment variable
``WORLD_AXES=0`` to leave them out, e.g. when recording camera data.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

WORLD_AXES_ENABLED = os.environ.get("WORLD_AXES", "1").strip().lower() not in ("0", "false", "off", "no")

# axis name -> (unit direction, display colour)
_AXES = {
    "X": ((1.0, 0.0, 0.0), (1.0, 0.05, 0.05)),
    "Y": ((0.0, 1.0, 0.0), (0.05, 1.0, 0.05)),
    "Z": ((0.0, 0.0, 1.0), (0.10, 0.30, 1.00)),
}
_ORIGIN_COLOR = (1.0, 1.0, 1.0)


def _material(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
    r, g, b = color
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=(r, g, b),
        emissive_color=(0.25 * r, 0.25 * g, 0.25 * b),  # stays readable in shadow (e.g. beside the conveyor)
        roughness=0.6,
        metallic=0.0,
    )


def _visual(prim_path: str, pos: tuple[float, float, float], spawn) -> AssetBaseCfg:
    """Static, physics-free visual prim (AssetBaseCfg is spawned once and never simulated)."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(pos), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=spawn,
    )


def world_axes_cfgs(
    length: float = 1.0 / 3,     # 0.33 m shaft: a 1 m triad spanned the whole conveyor edge
    shaft_radius: float = 0.01,  # tip cone (2.5x) and origin ball scale with this
    prim_path: str = "/World/WorldAxes",
    enabled: bool = WORLD_AXES_ENABLED,
) -> dict[str, AssetBaseCfg]:
    """Scene entries (``name -> AssetBaseCfg``) for the world axis triad.

    Args:
        length: shaft length in metres (cone tip is added on top of this).
        shaft_radius: shaft radius in metres; tip and origin ball scale with it.
        prim_path: USD path of the triad root.  The origin ball *is* the root Xform and the
            six axis prims are its children, so the root exists before they are spawned.
        enabled: return an empty dict (nothing added to the scene) when False.
    """
    if not enabled:
        return {}

    tip_radius = 2.5 * shaft_radius
    tip_height = 3.0 * tip_radius
    cfgs: dict[str, AssetBaseCfg] = {
        "world_axes_origin": _visual(
            prim_path,
            (0.0, 0.0, 0.0),
            sim_utils.SphereCfg(radius=tip_radius, visual_material=_material(_ORIGIN_COLOR)),
        )
    }
    for name, (unit, color) in _AXES.items():
        shaft_center = tuple(u * length / 2 for u in unit)
        tip_center = tuple(u * (length + tip_height / 2) for u in unit)
        cfgs[f"world_axes_{name.lower()}"] = _visual(
            f"{prim_path}/{name}",
            shaft_center,
            sim_utils.CylinderCfg(radius=shaft_radius, height=length, axis=name, visual_material=_material(color)),
        )
        cfgs[f"world_axes_{name.lower()}_tip"] = _visual(
            f"{prim_path}/{name}_tip",
            tip_center,
            sim_utils.ConeCfg(radius=tip_radius, height=tip_height, axis=name, visual_material=_material(color)),
        )
    return cfgs
