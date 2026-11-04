import numpy as np
from isaacsim.core.simulation_manager import SimulationManager

sim_view = SimulationManager.get_physics_sim_view()

def live_z(name):
    view = sim_view.create_rigid_body_view(f"/World/envs/env_0/{name}")
    if hasattr(view, "get_root_transforms"):   # torch backend
        t = view.get_root_transforms()
    else:                                      # numpy backend (CPU 실행 시)
        t = view.get_transforms()
    return float(np.asarray(t).reshape(-1, 7)[0][2])   # x,y,z,qw,qx,qy,qz

z0, z39 = live_z("card_box"), live_z("card_box39")
print(f"card_box   현재 z = {z0:.4f} m")
print(f"card_box39 현재 z = {z39:.4f} m")
print(f"중심 간 거리         = {(z39 - z0) * 100:.2f} cm")
print(f"스택 전체 높이(+3mm) = {(z39 - z0 + 0.003) * 100:.2f} cm")