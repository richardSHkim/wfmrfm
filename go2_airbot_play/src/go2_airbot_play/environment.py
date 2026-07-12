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


class Go2AirbotPlayPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """Go2 + Airbot Play pick-and-place, built on Arena's ``PickAndPlaceTask`` (brings the
    success criterion, the object-focused 3rd-person viewer, and the Mimic hooks for free).

    Objects sit low/in front — reachable by the Airbot arm (base ~0.5 m on the Go2 back,
    reaching down-forward). Placements are provisional and need reach verification.

        policy_runner.py ... --external_environment_class_path \
            go2_airbot_play.environment:Go2AirbotPlayPickAndPlaceEnvironment \
            go2_airbot_play_pick_place
    """

    name: str = "go2_airbot_play_pick_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        light = self.asset_registry.get_asset_by_name("light")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
        embodiment = self.asset_registry.get_asset_by_name("go2_airbot_play")(
            enable_cameras=getattr(args_cli, "enable_cameras", False),
        )

        # Robot at the room origin (galileo floor z=0), facing +x. z=0.40 is the Go2
        # standing base height — spawning the base at z=0 drops the body onto the floor
        # and the robot collapses.
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.40), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        # Pick object + destination on the floor in front of the robot, within arm reach.
        # TODO: verify against the Airbot Play reachable workspace once IK/teleop is wired.
        pick_up_object.set_initial_pose(Pose(position_xyz=(0.40, 0.12, 0.06), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        destination.set_initial_pose(Pose(position_xyz=(0.40, -0.22, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        scene = Scene(assets=[background, light, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object,
                destination,
                background,
                episode_length_s=30.0,
                task_description=(
                    f"Pick up the {args_cli.object.replace('_', ' ')} and place it in the "
                    f"{args_cli.destination.replace('_', ' ')}."
                ),
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="tomato_soup_can")
        parser.add_argument("--destination", type=str, default="blue_sorting_bin")
        parser.add_argument("--background", type=str, default="galileo_locomanip")
