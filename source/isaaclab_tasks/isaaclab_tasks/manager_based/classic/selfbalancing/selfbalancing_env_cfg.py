import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ImuCfg

import isaaclab_tasks.manager_based.classic.selfbalancing.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.selfbalancing import SELFBALANCING_CFG


##
# Scene definition
##


@configclass
class SelfbalancingSceneCfg(InteractiveSceneCfg):
    """Configuration for a selfbalancing scene."""

    # Podloga
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Svjetla
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot
    robot: ArticulationCfg = SELFBALANCING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Senzori
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/imu", debug_vis=True)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["wheel_left_joint", "wheel_right_joint"], scale=10.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # IMU Obeservations
        # Tilt angle estimate (from projected gravity)
        tilt = ObsTerm(func=mdp.imu_projected_gravity)

        # Angular velocity (from IMU, not global base ang_vel)
        ang_vel = ObsTerm(func=mdp.imu_ang_vel)

        wheel_vel = ObsTerm(func=mdp.joint_vel)

        global_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)

        global_pos = ObsTerm(func=mdp.root_pos_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_root = EventTerm(
    func=mdp.events.reset_root_state_uniform,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "pitch":(-0.4, 0.4)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=10.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    upright = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    min_spin = RewTerm(
        func=mdp.rewards.ang_vel_z_l2,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    distance_from_origin = RewTerm(
        func=mdp.rewards.distance_from_env_origin,
        weight=-2.0,  # Negative weight to penalize being far
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    fall_termination = DoneTerm(
        func=mdp.terminations.robot_fallen,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "angle_limit": 0.8,  # ~45 degrees
        }
    )

    terminate_out_of_bounds = DoneTerm(
        func=mdp.terminations.root_distance_from_origin,
        params={"max_distance": 1.5, "asset_cfg": SceneEntityCfg("robot")},
    )


# ##
# # Environment configuration
# ##


@configclass
class SelfbalancingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: SelfbalancingSceneCfg = SelfbalancingSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation