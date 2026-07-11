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

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Import for side effect: registers the Go2AirbotPlayEmbodiment asset.
from . import embodiment as _embodiment  # noqa: F401


@configclass
class _TimeOutTerminationCfg:
    """A single time-out (truncation) term so episodes end at episode_length_s — this is
    what makes the per-episode camera-video recorder flush an mp4 on reset."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)


class _RobotFollowNoTask(NoTask):
    """NoTask with (a) a robot-anchored viewport camera and (b) a short, time-limited
    episode so the per-episode camera-video recorder flushes an mp4 (writes on reset)."""

    def __init__(self, episode_length_s: float = 1.5):
        super().__init__()
        self.episode_length_s = episode_length_s

    def get_termination_cfg(self):
        return _TimeOutTerminationCfg()

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(
            eye=(-2.0, -1.8, 1.2),
            lookat=(0.0, 0.0, 0.4),
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
        )


class Go2AirbotPlayEnvironment(ExampleEnvironmentBase):
    """Go2 + Airbot Play standing in the galileo locomanip room (no task)."""

    name: str = "go2_airbot_play_scene"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        light = self.asset_registry.get_asset_by_name("light")()
        embodiment = self.asset_registry.get_asset_by_name("go2_airbot_play")(
            enable_cameras=getattr(args_cli, "enable_cameras", False),
        )

        # Scene assets are backgrounds/objects/lights (the embodiment is passed separately,
        # and the flat ground comes from the embodiment scene_config). A background room is
        # optional — cluttered rooms (walls/furniture) can occlude the third-person camera,
        # so the default scene is just ground + light. Pass --background <name> to add one.
        assets = [light]
        if args_cli.background and args_cli.background != "none":
            assets.append(self.asset_registry.get_asset_by_name(args_cli.background)())
        scene = Scene(assets=assets)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=_RobotFollowNoTask(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--background",
            type=str,
            default="none",
            help="Registered background asset to add ('none' = bare ground + light).",
        )
