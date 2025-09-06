from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def root_distance_from_origin(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Boolean tensor indicating if root distance exceeds max_distance."""
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.root_pos_w - env.scene.env_origins  # [num_envs, 3]
    dist = torch.norm(diff, dim=1)  # [num_envs]
    return dist >= max_distance  # [num_envs]




def robot_fallen(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    angle_limit: float = 0.5,  # radians
) -> torch.Tensor:
    """Terminate if robot tilt exceeds angle_limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat = asset.data.root_quat_w  # [N, 4]

    # batched world up vectors
    batch_size = root_quat.shape[0]
    world_up = torch.tensor([0, 0, 1], device=root_quat.device, dtype=root_quat.dtype).repeat(batch_size, 1)

    # rotated local up vector
    up_vector = quat_apply(root_quat, world_up)

    # angle between world up and robot up
    cos_angle = torch.clamp(up_vector[:, 2], -1.0, 1.0)
    tilt_angle = torch.acos(cos_angle)

    return tilt_angle >= angle_limit

