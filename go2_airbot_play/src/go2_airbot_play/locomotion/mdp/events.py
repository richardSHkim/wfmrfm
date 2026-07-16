# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Custom event: keep the Airbot Play arm *continuously moving* during locomotion training.

Rather than freezing the arm at a single stow pose, we resample a fresh random arm-joint
position target on a short interval.  The arm's PD actuator then drives toward it, so the
arm keeps swinging around throughout the episode.  This makes the arm a *time-varying*
payload — its centre of mass shifts constantly — forcing the lower-body locomotion policy
to be robust to a moving arm (the real whole-body-manipulation scenario), not just to a
single fixed configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def randomize_arm_joint_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_size: float = 0.25,
    range_fraction: float = 0.9,
):
    """Nudge the arm joint targets by a small random step (interval event).

    A **bounded random walk**: each resample moves the target a little away from the arm's
    *current* pose, so the arm drifts smoothly and continuously through its workspace at a
    manipulation-like speed — a realistic, *learnable* time-varying payload.

    (An earlier version sampled fresh absolute targets across most of the joint range every
    ~0.5 s; that whipped the arm at max velocity and the reaction torque toppled the robot
    ~99% of the time — not a physically achievable robustness target.)

    Args:
        env_ids: Envs whose arm target is resampled (``None`` → all envs).
        asset_cfg: Robot asset with ``joint_names`` restricted to the arm joints.
        step_size: Max per-resample change (rad) per joint. Small → gentle continuous drift.
        range_fraction: Fraction (0-1) of each joint's soft-limit span the target may roam,
            centred on the joint midpoint (keeps the arm out of the most extreme poses).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    limits = wp.to_torch(asset.data.soft_joint_pos_limits)  # (num_envs, num_joints, 2)
    lo = limits[env_ids][:, joint_ids, 0]
    hi = limits[env_ids][:, joint_ids, 1]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * range_fraction

    cur = wp.to_torch(asset.data.joint_pos)[env_ids][:, joint_ids]
    delta = (torch.rand_like(cur) * 2.0 - 1.0) * step_size
    target = torch.clamp(cur + delta, mid - half, mid + half)

    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
