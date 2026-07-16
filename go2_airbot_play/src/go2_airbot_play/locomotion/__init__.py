# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Go2 + Airbot Play locomotion (velocity + crouch) training task.

Registers the gym task via :func:`register`, which is designed to be used as the
``--external_callback go2_airbot_play.locomotion:register`` hook of Isaac Lab's
``scripts/reinforcement_learning/rsl_rl/{train,play}.py`` (returns ``None`` so all CLI
args pass through to Hydra unchanged).
"""

import gymnasium as gym

from . import agents

FLAT_TASK_ID = "Isaac-Velocity-Flat-Go2Airbot-v0"
FLAT_PLAY_TASK_ID = "Isaac-Velocity-Flat-Go2Airbot-Play-v0"
BOOTSTRAP_TASK_ID = "Isaac-Velocity-Flat-Go2Airbot-Bootstrap-v0"
BOOTSTRAP_PLAY_TASK_ID = "Isaac-Velocity-Flat-Go2Airbot-Bootstrap-Play-v0"

_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AirbotPlayFlatPPORunnerCfg"


def register():
    """Register the Go2 + Airbot Play locomotion gym tasks (idempotent)."""
    if FLAT_TASK_ID in gym.registry:
        return None

    for task_id, cfg_cls in [
        (FLAT_TASK_ID, "Go2AirbotPlayFlatEnvCfg"),
        (FLAT_PLAY_TASK_ID, "Go2AirbotPlayFlatEnvCfg_PLAY"),
        (BOOTSTRAP_TASK_ID, "Go2AirbotPlayFlatBootstrapEnvCfg"),
        (BOOTSTRAP_PLAY_TASK_ID, "Go2AirbotPlayFlatBootstrapEnvCfg_PLAY"),
    ]:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{cfg_cls}",
                "rsl_rl_cfg_entry_point": _AGENT,
            },
        )
    return None


# Also register on import (so `import go2_airbot_play.locomotion` alone is enough).
register()
