"""vega_1p_vertical_gripper 런타임 구동 게인/한계 점검 스크립트 (헤드리스 1회 실행)

실행: PROJECT_ROOT=$PWD .venv/bin/python tools/inspect_runtime_gains.py --headless
"""
import argparse
import os
import sys

# dexmate.py가 import 시점에 PROJECT_ROOT에서 USD 경로를 만들므로 미리 세팅
# (sim_main.py는 자체 세팅하지만 이 스크립트는 어디서 실행되든 스스로 경로를 잡아야 함)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PROJECT_ROOT", REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from tasks.common_config.robot_configs import Vega1pRobotPresets  # noqa: E402

sim = SimulationContext()

cfg = Vega1pRobotPresets.vega_1p_vertical_gripper()
cfg.prim_path = "/World/Robot"
cfg.spawn.func("/World/Robot", cfg.spawn, translation=(0.0, 0.0, 1.0))

robot = Articulation(cfg)
sim.reset()

print("\n===== joint drive state at runtime =====", flush=True)
wanted = ["L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4", "L_arm_j5", "L_arm_j6", "L_arm_j7",
          "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4", "R_arm_j5", "R_arm_j6", "R_arm_j7",
          "R_Slider_1", "R_Slider_2", "L_Slider_1", "L_Slider_2",
          "torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3",
          "B_wheel_j1", "B_wheel_j2"]
jnames = list(robot.data.joint_names[0]) if isinstance(robot.data.joint_names[0], (list, tuple)) else robot.data.joint_names
for n in wanted:
    if n not in jnames:
        print(f"{n:14s} (not a DOF)")
        continue
    i = jnames.index(n)
    k = robot.data.joint_stiffness[0, i].item()
    d = robot.data.joint_damping[0, i].item()
    lim = robot.data.joint_effort_limits[0, i].item()
    print(f"{n:14s} K={k:10.1f}  D={d:8.1f}  effort_limit={lim:.4g}")

print("\n===== all joints with non-inf effort limits =====", flush=True)
lims = robot.data.joint_effort_limits[0]
for i, n in enumerate(jnames):
    if lims[i] < 1e6:
        print(f"{n:20s} effort_limit={lims[i].item():.4g}", flush=True)

with open("/tmp/runtime_gains.txt", "w") as f:
    f.write("done\n")
import os  # noqa: E402

os._exit(0)
