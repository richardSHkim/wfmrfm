# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Custom command term: a commanded base height (for crouch/stand control).

The stock Isaac Lab velocity task commands only SE(2) velocity (vx, vy, wz).  To give
the Go2 + Airbot Play whole-body-manipulation controller the ability to *crouch* and
*stand* on command (lower/raise the base to bring the arm to a task), we add a scalar
target base height (world-frame z of the trunk) as a separate command term.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class UniformBaseHeightCommand(CommandTerm):
    """Samples a target trunk height (world z, in metres) uniformly per resample.

    Command shape is ``(num_envs, 1)``.  A companion reward (:func:`..rewards.track_base_height_exp`)
    rewards the policy for driving the trunk to this height, so the same policy learns to
    walk/turn/stand *and* crouch to a commanded height.
    """

    cfg: UniformBaseHeightCommandCfg

    def __init__(self, cfg: UniformBaseHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBaseHeightCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tHeight range: {self.cfg.ranges.height}"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        h = wp.to_torch(self.robot.data.root_pos_w)[:, 2]
        self.metrics["error_height"] += torch.abs(h - self.height_command[:, 0]) / max_step

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        # target height is absolute; no post-processing needed.
        pass

    # -- debug visualisation is not needed for a scalar height command --
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UniformBaseHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformBaseHeightCommand`."""

    class_type: type = UniformBaseHeightCommand

    asset_name: str = MISSING
    """Name of the robot asset the height is measured on."""

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """Range for the commanded trunk height (world z, metres)."""

    ranges: Ranges = MISSING
