# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Custom reward: track a commanded trunk height (crouch/stand control)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for matching the trunk world-z to the commanded height.

    reward = exp(-(z - h_cmd)^2 / std^2), in [0, 1].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = wp.to_torch(asset.data.root_pos_w)[:, 2]
    height_cmd = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(height - height_cmd) / (std**2))
