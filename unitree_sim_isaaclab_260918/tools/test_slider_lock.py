"""Headless check: pinned slider joint limits hold the vertical-gripper fingers against an external push.

실행: PROJECT_ROOT=$PWD .venv/bin/python tools/test_slider_lock.py --headless
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PROJECT_ROOT", REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402

from tasks.common_config.robot_configs import Vega1pRobotPresets  # noqa: E402

sim = SimulationContext(SimulationCfg(dt=0.005, device="cpu"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

cfg = Vega1pRobotPresets.vega_1p_vertical_gripper()
cfg.prim_path = "/World/Robot"
cfg.spawn.func("/World/Robot", cfg.spawn, translation=(0.0, 0.0, 0.04426))
robot = Articulation(cfg)
sim.reset()
dt = sim.get_physics_dt()

jn = list(robot.joint_names)
bn = list(robot.body_names)
ids = [jn.index("R_Slider_1"), jn.index("R_Slider_2")]
ids_t = torch.tensor(ids, dtype=torch.long)
lim0 = robot.data.joint_pos_limits[0, ids].clone()
print("slider limits0:", lim0.tolist(), flush=True)
print("effort limits:", robot.data.joint_effort_limits[0, ids].tolist(), flush=True)
big = [(jn[i], robot.data.joint_pos_limits[0, i].tolist()) for i in range(len(jn)) if robot.data.joint_pos_limits[0, i].abs().max() > 6.3]
print("joints with |limit| > 2pi (PhysX setLimitParams rejects these, harmless):", big, flush=True)

# right-hand Finger_1 body (two grippers share link names -> pick the one on the robot's right, y<0)
paths = list(robot.root_physx_view.link_paths[0])
fingers = [(i, pth) for i, pth in enumerate(paths) if "Finger" in pth]
print("finger links:", fingers, flush=True)
fid = [i for i, pth in fingers if pth.endswith("R_gripper/Finger_1")][0]
pos_w = robot.data.body_pos_w[0]
print("-> right Finger_1 =", fid, paths[fid], "y=", round(pos_w[fid, 1].item(), 3), flush=True)

close_target = torch.tensor([[-1.0, 1.0]])  # same saturated targets omniteleop sends (drive clamps at 23 N)
open_target = torch.zeros(1, 2)


def _body_x_axis_w():
    q = robot.data.body_quat_w[0, fid]  # (w, x, y, z)
    w, x, y, z = q.tolist()
    return torch.tensor([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])


def run(steps, force_n=0.0, target=close_target, sign=-1.0):
    """Push Finger_1 with force_n newton along sign * (body X axis, world frame) via the PhysX view directly."""
    nb = robot.num_bodies
    for _ in range(steps):
        robot.set_joint_position_target(target, joint_ids=ids)
        robot.write_data_to_sim()
        if force_n:
            f = torch.zeros(nb, 3)
            f[fid] = sign * force_n * _body_x_axis_w()
            robot.root_physx_view.apply_forces_and_torques_at_position(
                force_data=f, torque_data=torch.zeros(nb, 3), position_data=None,
                indices=torch.tensor([0]), is_global=True)
        sim.step()
        robot.update(dt)
    return robot.data.joint_pos[0, ids].clone()


def hold_gravity(steps):  # let the robot settle on the ground with default targets
    for _ in range(steps):
        robot.write_data_to_sim(); sim.step(); robot.update(dt)


hold_gravity(200)
p_closed = run(400)
print(f"closed pos          : {[round(v, 5) for v in p_closed.tolist()]}", flush=True)

# --- unlocked: push the long finger toward open with 60 N
print("finger body X axis (world):", [round(v, 3) for v in _body_x_axis_w().tolist()], flush=True)
v = robot.root_physx_view
print("dof max forces  :", v.get_dof_max_forces()[0, ids].tolist(), flush=True)
print("dof stiffness   :", v.get_dof_stiffnesses()[0, ids].tolist(), " damping:", v.get_dof_dampings()[0, ids].tolist(), flush=True)
print("dof limits (phx):", v.get_dof_limits()[0, ids].tolist(), flush=True)
P_MOVE = {}
for sign in (-1.0, +1.0):
    for F in (60.0, 500.0, 2000.0, 5000.0):
        p_push = run(100, force_n=F, sign=sign)
        jf = v.get_dof_projected_joint_forces()[0, ids].tolist()
        vel = robot.data.joint_vel[0, ids].tolist()
        print(f"UNLOCKED push {F:>5.0f}N sign{sign:+.0f}: pos={[round(x, 5) for x in p_push.tolist()]} dS1={p_push[0]-p_closed[0]:+.5f} "
              f"vel={[round(x, 4) for x in vel]} projF={[round(x, 1) for x in jf]}", flush=True)
        run(150, force_n=0.0)
    P_MOVE[sign] = 0.0
    p_push = run(100, force_n=60.0, sign=sign)
    P_MOVE[sign] = (p_push[0] - p_closed[0]).item()
    print(f"UNLOCKED push 60N sign{sign:+.0f}: {[round(v, 5) for v in p_push.tolist()]}  dS1={p_push[0]-p_closed[0]:+.5f} m", flush=True)
    p_back = run(300, force_n=0.0)
    print(f"release -> re-closed: {[round(v, 5) for v in p_back.tolist()]}", flush=True)

OPEN_SIGN = -1.0 if abs(P_MOVE[-1.0]) >= abs(P_MOVE[1.0]) else 1.0
print("opening push sign:", OPEN_SIGN, flush=True)
# --- locked: pin limits at current pos +-0.5 mm
margin = 5e-4
pos = robot.data.joint_pos[0, ids].clone()
lo = torch.maximum(pos - margin, lim0[:, 0]); hi = torch.minimum(pos + margin, lim0[:, 1])
robot.write_joint_position_limit_to_sim(torch.stack([lo, hi], -1).unsqueeze(0), joint_ids=ids, warn_limit_violation=False)
print("pinned limits       :", robot.data.joint_pos_limits[0, ids].tolist(), flush=True)
for F in (60.0, 200.0, 500.0):
    p = run(200, force_n=F, sign=OPEN_SIGN)
    print(f"LOCKED   push {F:>4.0f}N  : {[round(v, 5) for v in p.tolist()]}  dS1={p[0]-pos[0]:+.5f} m", flush=True)
run(50, force_n=0.0)

# --- unlock and open
robot.write_joint_position_limit_to_sim(lim0.unsqueeze(0), joint_ids=ids, warn_limit_violation=False)
p_open = run(400, target=open_target)
print(f"UNLOCKED -> open    : {[round(v, 5) for v in p_open.tolist()]}", flush=True)
print("DONE", flush=True)
os._exit(0)
