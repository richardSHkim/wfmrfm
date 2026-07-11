# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""External Arena environment for the Go2 + Airbot Play embodiment.

Run a zero-action smoke test with::

    python isaaclab_arena/evaluation/policy_runner.py \
        --policy_type zero_action --num_steps 50 \
        --external_environment_class_path go2_airbot_play.environment:Go2AirbotPlayEnvironment \
        go2_airbot_play_scene

Importing this module registers the ``go2_airbot_play`` embodiment (via the
``@register_asset`` decorator in :mod:`go2_airbot_play.embodiment`).
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Import for side effect: registers the Go2AirbotPlayEmbodiment asset.
from . import embodiment as _embodiment  # noqa: F401


class Go2AirbotPlayEnvironment(ExampleEnvironmentBase):
    """Go2 + Airbot Play standing in the galileo locomanip room (no task)."""

    name: str = "go2_airbot_play_scene"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        light = self.asset_registry.get_asset_by_name("light")()
        embodiment = self.asset_registry.get_asset_by_name("go2_airbot_play")(
            enable_cameras=getattr(args_cli, "enable_cameras", False),
        )

        # NOTE: the embodiment is passed to IsaacLabArenaEnvironment separately, NOT as a
        # scene asset (scene assets are backgrounds/objects/lights).
        scene = Scene(assets=[background, light])

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--background",
            type=str,
            default="galileo_locomanip",
            help="Registered background asset the robot stands in.",
        )
