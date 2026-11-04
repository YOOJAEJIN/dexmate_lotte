# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Parameter sweep for the card_box stack height (target: 12 cm, no explosion).

Rebuilds the REAL ChilsungSceneCfg source with patched numeric parameters,
steps physics until settled, and prints stack height + stability metrics.

Usage (from the project root, same venv as sim_main.py):
    source ~/isaacsim5_1/setup_conda_env.sh
    .venv/bin/python -u tools/sweep_card_stack.py --label v1 \
        --mass 0.02 --corr_dist 0.025 [--base_z 0.7475] [--gap 0.0001] \
        [--pos_iters 32] [--depen 0.05] [--steps 60]
"""
import argparse
import os
import re
import types

# PROJECT_ROOT must be set before importing task modules (same as sim_main.py:10)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="card_box stack parameter sweep")
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--base_z", type=float, default=0.7475)
parser.add_argument("--gap", type=float, default=0.0001)
parser.add_argument("--mass", type=float, default=0.1)
parser.add_argument("--pos_iters", type=int, default=32)
parser.add_argument("--depen", type=float, default=0.05)
parser.add_argument("--corr_dist", type=float, default=0.00625)
parser.add_argument("--rest_offset", type=float, default=None, help="if set, patches rest_offset (m)")
parser.add_argument("--contact_offset", type=float, default=None, help="if set, patches contact_offset (m)")
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

CFG_PATH = os.path.join(PROJECT_ROOT, "tasks/common_scene/base_scene_pickplace_chilsungcfg.py")
CARD_THICKNESS = 0.003
NUM_CARDS = 40

# --- build a tuned copy of the real scene cfg module ---
src = open(CFG_PATH).read()
src = re.sub(r"(CARD_BOX_BASE_Z\s*=\s*)[\d.]+", rf"\g<1>{args_cli.base_z}", src)
src = re.sub(r"(CARD_BOX_GAP\s*=\s*)[\d.]+", rf"\g<1>{args_cli.gap}", src)
src = re.sub(r"(MassPropertiesCfg\(mass\s*=\s*)[\d.]+", rf"\g<1>{args_cli.mass}", src)
src = re.sub(r"(solver_position_iteration_count\s*=\s*)\d+", rf"\g<1>{args_cli.pos_iters}", src)
src = re.sub(r"(max_depenetration_velocity\s*=\s*)[\d.]+", rf"\g<1>{args_cli.depen}", src)
if args_cli.rest_offset is not None:
    src = re.sub(r"(rest_offset\s*=\s*)-?[\d.]+", rf"\g<1>{args_cli.rest_offset}", src)
if args_cli.contact_offset is not None:
    src = re.sub(r"(contact_offset\s*=\s*)[\d.]+", rf"\g<1>{args_cli.contact_offset}", src)
mod = types.ModuleType("chilsung_tuned")
mod.__dict__["__file__"] = CFG_PATH
exec(compile(src, CFG_PATH, "exec"), mod.__dict__)

# physics settings mirrored from PickPlaceVega1pGripperEnvCfg.__post_init__()
sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005, render_interval=50)
sim_cfg.physx.bounce_threshold_velocity = 0.5
sim_cfg.physx.friction_correlation_distance = args_cli.corr_dist
sim = sim_utils.SimulationContext(sim_cfg)

scene_cfg = mod.ChilsungSceneCfg(num_envs=1, env_spacing=2.5)
scene_cfg.room_walls = None    # warehouse.usd is loaded from a remote nucleus server
scene_cfg.world_camera = None  # viewport camera needs rendering
scene = InteractiveScene(cfg=scene_cfg)
sim.reset()

CARD_NAMES = ["object"] + [f"object{i}" for i in range(1, NUM_CARDS)]


def report(label):
    pos = torch.stack([scene[n].data.root_pos_w[:, :] for n in CARD_NAMES]).squeeze(1)  # (40, 3)
    zs_sorted, order = torch.sort(pos[:, 2])
    spawn_xy = torch.tensor([0.0, -0.34], device=pos.device)
    xy_drift = (pos[:, :2] - spawn_xy).abs().amax().item()
    bottom = zs_sorted[0].item() - CARD_THICKNESS / 2
    top = zs_sorted[-1].item() + CARD_THICKNESS / 2
    spacing = (zs_sorted[1:] - zs_sorted[:-1]) * 1000  # mm
    exploded = xy_drift > 0.05 or spacing.max().item() > 3.5 or spacing.min().item() < -1.0
    print(f"[{label}] height={((top - bottom) * 100):.2f} cm | "
          f"spacing mean={spacing.mean().item():.2f} min={spacing.min().item():.2f} max={spacing.max().item():.2f} mm | "
          f"xy_drift={xy_drift * 1000:.1f} mm | exploded={exploded}", flush=True)
    return exploded


report(f"{args_cli.label} t=0.0s (spawn)")
physics_dt = sim.get_physics_dt()
exploded = False
for step in range(1, args_cli.steps + 1):
    sim.step()
    scene.update(dt=physics_dt)
    if step % 30 == 0:
        exploded = exploded or report(f"{args_cli.label} t={step * physics_dt:.2f}s")
if not exploded:
    report(f"{args_cli.label} FINAL")
print(f"{args_cli.label} params: base_z={args_cli.base_z} gap={args_cli.gap} mass={args_cli.mass} "
      f"pos_iters={args_cli.pos_iters} depen={args_cli.depen} corr={args_cli.corr_dist}", flush=True)

# kit shutdown can hang in headless batch runs; results are already flushed, so force-exit
import os as _os

_os._exit(0)
