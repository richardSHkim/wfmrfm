# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL PPO runner config for the Go2 + Airbot Play flat locomotion task."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


@configclass
class Go2AirbotPlayFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_airbot_flat"
        # Harder, multi-objective task (walk + turn + crouch + hold steady under a moving,
        # variably-loaded arm) trained from scratch → larger net + more iterations than the
        # stock 128^3 / 300-iter Go2 flat recipe.
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.max_iterations = 3000
        self.save_interval = 100
