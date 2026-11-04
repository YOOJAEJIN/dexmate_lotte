import argparse

from isaaclab.app import AppLauncher
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

project_root = "/home/ubuntu/unitree_sim_isaaclab"

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",  # use simple room model
        ),
    )

    # pallete = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/pallete",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[1.0, -1.82, 0.1],
    #         rot=[0.0, 1.0, 0.0, 0.0],
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{project_root}/assets/objects/workspace_with_bin/pallet.usd",
    #         scale=(0.02, 0.03, 0.006),
    #     ),
    # )

    bin1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.28, -1.42, 0.11],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.74, -1.42, 0.11],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.28, -1.84, 0.11],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin4 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.74, -1.84, 0.11],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin11 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin11",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.28, -1.42, 0.41],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin21 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin21",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.74, -1.42, 0.41],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin31 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin31",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.28, -1.84, 0.41],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )
    bin41 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bin41",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.74, -1.84, 0.41],
            rot=[0.7071, 0, 0, 0.7071],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/workspace_with_bin/klt.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1/120,
        render_interval=1,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            enable_stabilization=True,
            max_position_iteration_count=8,
            max_velocity_iteration_count=1,
        )
    )
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.0, 1.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

