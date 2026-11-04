# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Measure the exact height of the card_box stack spawned by ChilsungSceneCfg.

Loads the same scene as the real task (table + 40 stacked card boxes), steps
physics until the stack settles, prints the exact stack height in cm, and
validates the Fabric live-read snippet (the same code can be pasted into the
Script Editor of a running simulation).

Usage (from the project root, same venv as sim_main.py):
    source ~/isaacsim5_1/setup_conda_env.sh
    .venv/bin/python -u tools/measure_card_stack.py --device cpu
"""
import argparse
import os

# PROJECT_ROOT must be set before importing task modules (same as sim_main.py:10)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure card_box stack height")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# this script only needs physics, not rendering
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from tasks.common_scene.base_scene_pickplace_chilsungcfg import ChilsungSceneCfg  # noqa: E402

NUM_CARDS = 40
CARD_NAMES = ["object"] + [f"object{i}" for i in range(1, NUM_CARDS)]
CARD_PRIMS = ["card_box"] + [f"card_box{i}" for i in range(1, NUM_CARDS)]
CARD_THICKNESS = 0.003  # m, verified from card_box.usd bbox (z: -0.0015 ~ +0.0015)
TABLE_TOP_Z = 0.7581    # m, table top surface (see base_scene_pickplace_chilsungcfg.py:22)

# physics settings mirrored from PickPlaceVega1pGripperEnvCfg.__post_init__()
sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005, render_interval=50)
sim_cfg.physx.bounce_threshold_velocity = 0.5
sim_cfg.physx.friction_correlation_distance = 0.00625
if hasattr(sim_cfg.physx, "use_flatcache"):
    sim_cfg.physx.use_flatcache = True
if hasattr(sim_cfg.physx, "solver_iteration_count"):
    sim_cfg.physx.solver_iteration_count = 4

sim = sim_utils.SimulationContext(sim_cfg)

# same scene as the real task, minus remote/render-only assets
scene_cfg = ChilsungSceneCfg(num_envs=1, env_spacing=2.5)
scene_cfg.room_walls = None    # warehouse.usd is loaded from a remote nucleus server
scene_cfg.world_camera = None  # viewport camera needs rendering
scene = InteractiveScene(cfg=scene_cfg)
sim.reset()


def measure(label: str):
    """Print exact stack height from current card root positions (IsaacLab data API)."""
    zs = torch.stack([scene[n].data.root_pos_w[:, 2] for n in CARD_NAMES]).squeeze(1)
    zs_sorted, _ = torch.sort(zs)
    bottom = zs_sorted[0].item() - CARD_THICKNESS / 2
    top = zs_sorted[-1].item() + CARD_THICKNESS / 2
    height = top - bottom
    spacing = zs_sorted[1:] - zs_sorted[:-1]  # center-to-center, ideal settled = 3 mm
    print(f"[{label}]", flush=True)
    print(f"  bottom surface z = {bottom:.4f} m   (table top {TABLE_TOP_Z:.4f} m)", flush=True)
    print(f"  top    surface z = {top:.4f} m", flush=True)
    print(
        f"  stack height       = {height * 100:.2f} cm  (ideal settled: {NUM_CARDS * CARD_THICKNESS * 100:.1f} cm)",
        flush=True,
    )
    print(f"  height above table = {(top - TABLE_TOP_Z) * 100:.2f} cm", flush=True)
    print(
        f"  card-to-card spacing: mean={spacing.mean().item() * 1000:.2f} mm"
        f"  min={spacing.min().item() * 1000:.2f} mm  max={spacing.max().item() * 1000:.2f} mm",
        flush=True,
    )
    print(f"  avg overlap per interface = {(CARD_THICKNESS - spacing.mean().item()) * 1000:+.2f} mm", flush=True)
    print(flush=True)


# measure right after spawn (1 mm gaps still present), then step until settled
measure("t = 0.0 s (right after spawn)")
physics_dt = sim.get_physics_dt()
check_interval = 30   # 0.15 s
total_steps = 60      # 0.30 s (stack is fully static from t=0.25 s in test runs)
for step in range(1, total_steps + 1):
    sim.step()
    scene.update(dt=physics_dt)
    if step % check_interval == 0:
        measure(f"t = {step * physics_dt:.2f} s")


##
# Fabric live-read validation (same pattern the running sim's Script Editor snippet uses)
##
def read_z_from_fabric() -> "list[float]":
    """Read live world z of all card prims directly from Fabric (physics truth).

    Mirrors isaacsim.core internals (XFormPrimView._get_fabric_selection):
    attach a usdrt stage, tag the prims with a temp index attribute,
    SelectPrims on Fabric, then read omni:fabric:worldMatrix via warp.
    """
    import numpy as np
    import omni.usd
    import usdrt
    import warp as wp
    from pxr import UsdUtils

    stage = omni.usd.get_context().get_stage()
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    rt_stage = usdrt.Usd.Stage.Attach(stage_id)

    # refresh computed world matrices in Fabric
    hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
    )
    hierarchy.update_world_xforms()

    # tag the 40 card prims with a temp index attribute (SelectPrims has no path filter)
    view_attr = "tmpStackViewIdx"
    for i, name in enumerate(CARD_PRIMS):
        prim = rt_stage.GetPrimAtPath(usdrt.Sdf.Path(f"/World/envs/env_0/{name}"))
        prim.CreateAttribute(view_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
        prim.GetAttribute(view_attr).Set(i)

    selection = rt_stage.SelectPrims(
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.UInt, view_attr, usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Double3, "xformOp:translate", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
        ],
        device="cpu",
    )
    # render a few frames: the render delegate drives the physics->Fabric transform sync
    for _ in range(5):
        sim.render()
        simulation_app.update()
    hierarchy.update_world_xforms()
    print(f"  [debug] fabric selection count = {selection.GetCount()}", flush=True)
    trans = np.array(wp.fabricarray(selection, "xformOp:translate").numpy())
    mats = np.array(wp.fabricarray(selection, "omni:fabric:worldMatrix").numpy())
    idx = np.array(wp.fabricarray(selection, view_attr).numpy())
    print(f"  [debug] translate shape={trans.shape} mats shape={mats.shape}", flush=True)
    order = idx.argsort()
    zs_translate = trans[order, 2]           # local translate z (env_0 origin = world origin)
    zs_matrix = mats[order, 2, 3] if mats.ndim == 3 else mats[order].reshape(-1, 4, 4)[:, 2, 3]
    print(f"  [debug] first 3 translate = {trans[order][:3].tolist()}", flush=True)
    print(f"  [debug] first 3 matrix z  = {zs_matrix[:3].tolist()}", flush=True)
    return zs_translate.tolist()


print("=" * 60, flush=True)
print("PhysX tensor-API live read (Script Editor snippet equivalence)", flush=True)
try:
    import numpy as np
    from isaacsim.core.simulation_manager import SimulationManager

    sim_view = SimulationManager.get_physics_sim_view()

    def physx_z(name):
        view = sim_view.create_rigid_body_view(f"/World/envs/env_0/{name}")
        if hasattr(view, "get_root_transforms"):  # torch backend
            t = view.get_root_transforms()
        else:  # numpy backend
            t = view.get_transforms()
        return float(np.asarray(t).reshape(-1, 7)[0][2])  # x y z qw qx qy qz

    zs_physx = [physx_z(n) for n in CARD_PRIMS]
    zs_lab2 = torch.stack([scene[n].data.root_pos_w[:, 2] for n in CARD_NAMES]).squeeze(1).cpu().tolist()
    print(f"  card_box   physx z = {zs_physx[0]:.4f} m | isaaclab z = {zs_lab2[0]:.4f} m", flush=True)
    print(f"  card_box39 physx z = {zs_physx[39]:.4f} m | isaaclab z = {zs_lab2[39]:.4f} m", flush=True)
    max_diff = max(abs(a - b) for a, b in zip(zs_physx, zs_lab2))
    print(f"  max |physx - isaaclab| over 40 cards = {max_diff * 1000:.3f} mm", flush=True)
    height_cm = (max(zs_physx) - min(zs_physx) + CARD_THICKNESS) * 100
    print(f"  stack height from PhysX tensor API = {height_cm:.2f} cm", flush=True)
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"  physx read failed: {e}", flush=True)
print("=" * 60, flush=True)

simulation_app.close()
