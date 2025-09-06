from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg
import isaaclab.sim as sim_utils

SELFBALANCING_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="../../../../../../Models/Simple_Robot/Simple_v2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"wheel_left_joint": 0.0, "wheel_right_joint": 0.0}
    ),
    actuators={
        "wheel_drive": DCMotorCfg(
            joint_names_expr=[".*wheel_left_joint", ".*wheel_right_joint"],
            saturation_effort=100.0,     
            velocity_limit=200.0,       
            stiffness=0.0,              
            damping=0.0                 
        )
    }
)