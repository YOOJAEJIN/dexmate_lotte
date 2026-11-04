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

z0 = live_z("card_box")
print(f"card_box   현재 z = {z0:.4f} m")