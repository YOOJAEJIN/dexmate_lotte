"""Headless check of ZenohActionProvider's slider ratchet lock against a real IsaacLab Articulation.

Covers: immediate engage on a close target, fingers still close fully (one-sided limit),
external opening push held, wheels stay free (sanitized limit write), release + reopen.

실행: source .venv/bin/activate && source <isaacsim>/setup_conda_env.sh && \\
      PROJECT_ROOT=$PWD python tools/test_slider_lock_provider.py --headless
"""
import argparse, os, sys, types
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PROJECT_ROOT", REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args_cli = parser.parse_args()
app = AppLauncher(headless=True).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from tasks.common_config.robot_configs import Vega1pRobotPresets

zm = types.ModuleType("zenoh.zenoh_manager"); zm.get_zenoh_manager = lambda: None
zp = types.ModuleType("zenoh"); zp.zenoh_manager = zm
sys.modules["zenoh"] = zp; sys.modules["zenoh.zenoh_manager"] = zm
from action_provider.action_provider_zenoh import ZenohActionProvider

sim = SimulationContext(SimulationCfg(dt=0.005, device="cpu"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
cfg = Vega1pRobotPresets.vega_1p_vertical_gripper(); cfg.prim_path = "/World/Robot"
cfg.spawn.func("/World/Robot", cfg.spawn, translation=(0.0, 0.0, 0.04426))
robot = Articulation(cfg); sim.reset(); dt = sim.get_physics_dt()

env = types.SimpleNamespace(scene={"robot": robot}, device="cpu")
args = types.SimpleNamespace(robot_type="vega_1p", enable_zenoh_hand="vertical", disable_gripper_lock=False)
prov = ZenohActionProvider(env, args)
print("initialized:", getattr(prov, "_initialized", False), "lock sides:", list(prov._gripper_lock.keys()), flush=True)

jn = list(robot.joint_names); ids = [jn.index("R_Slider_1"), jn.index("R_Slider_2")]
view = robot.root_physx_view
paths = list(view.link_paths[0])
fid = [i for i, pth in enumerate(paths) if pth.endswith("R_gripper/Finger_1")][0]
close_cmd = {"right_hand": [-1.0, 1.0], "left_hand": [-1.0, 1.0]}
open_cmd = {"right_hand": [0.0, 0.0], "left_hand": [0.0, 0.0]}
R = prov._gripper_lock["right_hand"]

def action_for(cmds):
    a = torch.zeros(1, 28)
    for side, tgt in cmds.items():
        m = prov.pos_mapping[side]
        a[0, m["action_indices"]] = torch.tensor(tgt) - m["default_pos"]
    return a

def body_x_axis_w():
    w, x, y, z = robot.data.body_quat_w[0, fid].tolist()
    return torch.tensor([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])

def step(cmds, n, target, force_n=0.0, sign=-1.0):
    """sign=-1 * body X on Finger_1 = opening direction for R_Slider_1 (measured in test_slider_lock.py)."""
    nb = robot.num_bodies
    for _ in range(n):
        robot.set_joint_position_target(torch.tensor([target]), joint_ids=ids)
        robot.write_data_to_sim()
        if force_n:
            f = torch.zeros(nb, 3); f[fid] = sign * force_n * body_x_axis_w()
            view.apply_forces_and_torques_at_position(force_data=f, torque_data=torch.zeros(nb, 3), position_data=None,
                                                      indices=torch.tensor([0]), is_global=True)
        sim.step(); robot.update(dt)
        prov.observe_action(action_for(cmds))

def phx_lim(): return [[round(a, 5), round(b, 5)] for a, b in view.get_dof_limits()[0, ids].tolist()]
def pos(): return [round(v, 5) for v in robot.data.joint_pos[0, ids].tolist()]

for _ in range(200): robot.write_data_to_sim(); sim.step(); robot.update(dt)
print("open pos:", pos(), "limits:", phx_lim(), flush=True)

# 1) close target -> ratchet engages immediately, before the fingers move
step(close_cmd, 1, [-1.0, 1.0])
print("1 step after close target: locked", R["locked"], "pos", pos(), "limits", phx_lim(), flush=True)
assert R["locked"], "ratchet did not engage on the close target"

# 2) fingers must still be able to close fully (one-sided limit)
step(close_cmd, 400, [-1.0, 1.0])
p_closed = robot.data.joint_pos[0, ids].clone()
print("after 2s: pos", pos(), "limits", phx_lim(), "tightened", R["tightened"], flush=True)
assert abs(p_closed[0].item() + 0.082) < 1e-3 and abs(p_closed[1].item() - 0.082) < 1e-3, "fingers could not close"

# 3) opening push while locked -> held within the play
for F in (60.0, 200.0, 500.0):
    step(close_cmd, 200, [-1.0, 1.0], force_n=F)
    d = (robot.data.joint_pos[0, ids] - p_closed).tolist()
    print(f"LOCKED push {F:>4.0f}N opening: pos {pos()} dS1={d[0]:+.5f} m", flush=True)
    assert abs(d[0]) < 1e-3, "finger pushed open while locked"
step(close_cmd, 100, [-1.0, 1.0])

# 4) wheels stay free after the (sanitized) limit writes
wid = [jn.index("R_wheel_j2")]; travel = 0.0
for _ in range(1600):
    robot.set_joint_position_target(torch.tensor([[-1.0, 1.0]]), joint_ids=ids)
    robot.set_joint_velocity_target(torch.tensor([[8.0]]), joint_ids=wid)
    robot.write_data_to_sim(); sim.step(); robot.update(dt)
    prov.observe_action(action_for(close_cmd))
    travel += abs(robot.data.joint_vel[0, wid].item()) * dt
v1 = robot.data.joint_vel[0, wid].item()
print(f"wheel R_wheel_j2 8s @8rad/s: travel={travel:.2f} rad final vel={v1:.2f} phx limits={view.get_dof_limits()[0, wid].tolist()}", flush=True)
assert travel > 2 * 3.14159 * 1.5 and abs(v1) > 1.0, "wheel got limited"

# 5) open target -> release, fingers reopen
step(open_cmd, 5, [0.0, 0.0])
print("after open target: locked", R["locked"], "limits", phx_lim(), flush=True)
assert not R["locked"]
step(open_cmd, 400, [0.0, 0.0])
print("opened pos:", pos(), flush=True)
assert abs(robot.data.joint_pos[0, ids[0]].item()) < 2e-3
print("PROVIDER_OK", flush=True)
os._exit(0)
