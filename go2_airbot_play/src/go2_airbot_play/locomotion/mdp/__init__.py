# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Custom MDP terms for the Go2 + Airbot Play locomotion task.

Re-exports the stock velocity-task MDP terms plus our additions (base-height command,
continuous random arm motion, base-height tracking reward).
"""

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import UniformBaseHeightCommand, UniformBaseHeightCommandCfg  # noqa: F401
from .events import randomize_arm_joint_targets  # noqa: F401
from .rewards import track_base_height_exp  # noqa: F401
