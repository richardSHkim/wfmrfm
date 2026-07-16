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
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg, TerminationTermCfg
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


class Go2AirbotPlayMapleTablePickPlaceEnvironment(ExampleEnvironmentBase):
    """Go2 + Airbot Play pick-and-place on a simple cuboid surface.

    Originally built on Arena's ``maple_table_robolab`` background, but that asset is a BASE
    background whose ``set_initial_pose`` is a no-op on the rendered/physics geometry — the
    table stayed pinned at its native ~0.67 m height (verified via camera-depth back-proj),
    above the Go2 body-mounted front camera (z~0.49), so that camera only ever saw the table
    underside/legs. To get a controllable, camera-visible surface we replace the table with a
    VISIBLE solid cuboid pedestal:

    - ``procedural_table`` is a RIGID kinematic body, so (unlike the BASE background) it
      honors the pose/size we set. We override its spawn with a visible ``CuboidCfg`` that
      spans the floor up to ``--table_z`` (default 0.30 m, i.e. below the front camera) so
      the front camera looks down onto the surface + objects.
    - The Go2 is welded in place (``--fix_base``) in front of the pedestal at ``--robot_x``.
    - Pick object + destination rest on the pedestal top, centred in the front-cam view.

    (Aesthetics were intentionally dropped — a plain box, not a textured table.)

        policy_runner.py ... --external_environment_class_path \
            go2_airbot_play.environment:Go2AirbotPlayMapleTablePickPlaceEnvironment \
            go2_airbot_play_maple_table
    """

    name: str = "go2_airbot_play_maple_table"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        # Surface = a VISIBLE solid cuboid pedestal from the floor up to table_top_z, centred
        # in front of the robot. ``procedural_table`` is RIGID+kinematic, so it actually honors
        # the pose/size we set (the BASE maple_table background did not). Front cam at ~0.49 m
        # looks down onto this surface when table_top_z < ~0.49.
        table_top_z = args_cli.table_z
        # Pedestal pulled close to the robot so its near region is within the Airbot Play
        # reach envelope (shoulder ~(-0.155, 0, 0.611), practical reach ~0.6 m).
        table_center_xy = (0.50, 0.0)
        background = self.asset_registry.get_asset_by_name("procedural_table")()
        background.object_cfg.spawn = sim_utils.CuboidCfg(
            size=(0.5, 0.8, table_top_z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.20)),
            visible=True,
        )
        background.set_initial_pose(
            Pose(position_xyz=(*table_center_xy, table_top_z / 2.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        # Dome light. Optionally drape an HDR environment map over it: this is exactly what
        # provides the photorealistic room/kitchen backdrop in Arena's marketing renders of the
        # maple-table scene (the maple_table.usda itself is only a table + ground plane — the
        # "environment" is the HDR dome, independent of table/robot). 11 HDRs are registered.
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=500.0),
        )
        if args_cli.hdr and args_cli.hdr.lower() != "none":
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
        embodiment = self.asset_registry.get_asset_by_name("go2_airbot_play")(
            enable_cameras=getattr(args_cli, "enable_cameras", False),
        )

        # Anchor the base: weld the Go2 root link to the world at its spawn pose. Without a
        # balance / locomotion controller the arm's mass (folded high on the back) tips the
        # static-PD stand over; fixing the root makes it a stationary tabletop manipulator
        # (legs hold their standing pose, arm is free to move). Pass --no-fix_base once a
        # locomotion/WBC controller is wired to let the quadruped balance on its own.
        if args_cli.fix_base:
            embodiment.scene_config.robot.spawn.articulation_props.fix_root_link = True

        # Arm "ready" pose for tabletop manipulation: pitch the arm down-forward
        # (joint2=-2.0, joint3=1.0; both within limits j2∈[-2.97,0.17], j3∈[-0.09,3.14]) so
        # the gripper hovers ~0.1 m above the objects (EE ~(0.25,0,0.46)) and the wrist camera
        # looks DOWN at them. The embodiment default is all-zeros (arm up), which points the
        # wrist camera away from the workspace. init_state.joint_pos sets both the spawn pose
        # and the zero-action hold target (use_default_offset), so the arm holds this pose.
        # (Override currently DISABLED — arm ready pose left at the embodiment default,
        # all-zeros = arm up. Re-enable the two lines below for the reach-down pose.)
        # arm_ready = dict(embodiment.scene_config.robot.init_state.joint_pos)
        # arm_ready["joint2"] = -2.3; arm_ready["joint3"] = 1.5
        # embodiment.scene_config.robot.init_state.joint_pos = arm_ready

        # Go2 on the floor in front of the table's near edge (x=0.2), facing +x.
        embodiment.set_initial_pose(
            Pose(position_xyz=(args_cli.robot_x, 0.0, 0.40), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        # Objects near the pedestal's FRONT edge (small x), inside the Airbot Play reach
        # envelope (~0.57 m from the shoulder) and under the wrist camera's downward view.
        # They spawn just above the surface and settle onto it.
        obj_x = 0.30
        pick_up_object.set_initial_pose(
            Pose(position_xyz=(obj_x, 0.10, table_top_z + 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        destination.set_initial_pose(
            Pose(position_xyz=(obj_x, -0.14, table_top_z + 0.04), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        scene = Scene(assets=[background, light, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=args_cli.episode_length_s,
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
        parser.add_argument("--object", type=str, default="rubiks_cube_hot3d_robolab")
        parser.add_argument("--destination", type=str, default="bowl_ycb_robolab")
        parser.add_argument(
            "--table_z", type=float, default=0.30,
            help="World z of the cuboid pedestal's TOP surface (keep below front cam ~0.49 to see it).",
        )
        parser.add_argument(
            "--robot_x", type=float, default=-0.15,
            help="Go2 base x on the floor in front of the table (facing +x toward it).",
        )
        parser.add_argument(
            "--hdr", type=str, default=None,
            help=(
                "Registered HDR environment map draped on the dome light for a photorealistic "
                "backdrop (e.g. home_office_robolab, kiara_interior_robolab, wooden_lounge_robolab, "
                "garage_robolab, ...). Omit / 'none' for a plain grey background."
            ),
        )
        parser.add_argument(
            "--episode_length_s", type=float, default=30.0,
            help=(
                "Episode length (s). Lower it (e.g. 4) for quick zero-action placement checks:"
                " the per-episode camera-video recorder only flushes an mp4 on reset."
            ),
        )
        parser.add_argument(
            "--fix_base", action=argparse.BooleanOptionalAction, default=True,
            help=(
                "Weld the Go2 base to the world (default) so it can't topple without a balance"
                " controller. Use --no-fix_base for a free-floating base once WBC is wired."
            ),
        )


@configclass
class _EefIkActionCfg:
    """Differential-IK absolute-EEF arm control + binary parallel gripper.

    This is the action space a GR00T Phase-1 policy drives: the arm is commanded by an
    absolute end-effector POSE (pos + quat, in the robot base frame) solved with damped
    least-squares IK, matching how ``scripts/collect_pickplace_dataset.py`` recorded the
    dataset. Action layout per env step is an 8-vector: ``[x, y, z, qw, qx, qy, qz, grip]``.
    The gripper term is binary (sign of the last element: >0 opens, <=0 closes).
    """

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        body_name="g2_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["g2_joint", "g2_left_joint", "g2_right_joint"],
        open_command_expr={"g2_joint": 0.072, "g2_left_joint": 0.036, "g2_right_joint": 0.036},
        close_command_expr={"g2_joint": 0.0, "g2_left_joint": 0.0, "g2_right_joint": 0.0},
    )


@configclass
class _JointArmActionCfg:
    """Absolute joint-position arm control + binary parallel gripper.

    The action space a GR00T **joint** policy drives: the arm is 6 absolute joint targets
    (``scale=1``, ``use_default_offset=False`` so the command IS the absolute joint position,
    matching the joint-action dataset), no IK. Action layout per step is a 7-vector:
    ``[joint1..joint6, grip]`` (grip binary: >0 opens, <=0 closes).
    """

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["g2_joint", "g2_left_joint", "g2_right_joint"],
        open_command_expr={"g2_joint": 0.072, "g2_left_joint": 0.036, "g2_right_joint": 0.036},
        close_command_expr={"g2_joint": 0.0, "g2_left_joint": 0.0, "g2_right_joint": 0.0},
    )


class Go2AirbotPlayMapleTablePickPlaceEvalEnvironment(Go2AirbotPlayMapleTablePickPlaceEnvironment):
    """Closed-loop **evaluation** variant of the maple-table pick-place env.

    Identical scene / task / objects to the collection env, but swaps the arm to the
    differential-IK absolute-EEF action space (:class:`_EefIkActionCfg`) so a GR00T Phase-1
    policy (arm relative-EEF + gripper) can drive it directly. The base stays welded
    (``--fix_base``, the Phase-1 assumption): no locomotion command is consumed here. The
    ``PickAndPlaceTask`` success termination gives the eval success metric.

        python isaaclab_arena/evaluation/policy_runner.py \
            --external_environment_class_path \
                go2_airbot_play.environment:Go2AirbotPlayMapleTablePickPlaceEvalEnvironment \
            --policy_type go2airbot_gr00t_eef --enable_cameras --num_episodes 10 \
            go2_airbot_play_maple_table_eval
    """

    name: str = "go2_airbot_play_maple_table_eval"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.utils.pose import Pose, PoseRange

        arena_env = super().get_env(args_cli)
        # Set the arm action to match the deployed GR00T policy's action space:
        #   ik_eef  -> differential-IK EEF pose (relative-EEF checkpoints)
        #   joint   -> absolute joint targets   (joint-space checkpoints)
        # Done on the built embodiment before the ArenaEnvBuilder composes the manager cfg.
        if getattr(args_cli, "arm_action", "ik_eef") == "joint":
            arena_env.embodiment.action_config = _JointArmActionCfg()
        else:
            arena_env.embodiment.action_config = _EefIkActionCfg()
        # Render cameras at the dataset resolution (180x320) so the policy sees the same input
        # distribution it was trained on, and to keep per-step ZMQ payloads light.
        cc = arena_env.embodiment.camera_config
        for cam in (cc.wrist_cam, cc.go2_front_cam, cc.overview_cam):
            cam.height = 180
            cam.width = 320
        # Per-episode object-position randomization so the success rate reflects generalization
        # rather than one fixed layout. Re-setting the initial pose to a PoseRange makes the
        # object's reset event a uniform ``randomize_object_pose``. Ranges (world x,y on the
        # pedestal top) mirror scripts/collect_pickplace_dataset.py — the training distribution:
        # cube on the +y half, bowl on the -y half, both within the arm's top-down reach; z held
        # at rest height and orientation left upright (rpy=0). Disable with --no-randomize_objects
        # to keep the parent's FIXED training-like layout (cube (0.30,0.10), bowl (0.30,-0.14)).
        tz = args_cli.table_z
        rest_z = tz + 0.03
        pick_xy = getattr(args_cli, "pick_pos_xy", None)
        place_xy = getattr(args_cli, "place_pos_xy", None)
        if pick_xy or place_xy:
            # Explicit fixed positions (overrides randomization). Used to reproduce a specific
            # dataset episode's layout for a replay control experiment.
            if pick_xy:
                arena_env.scene.assets[args_cli.object].set_initial_pose(
                    Pose(position_xyz=(pick_xy[0], pick_xy[1], rest_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
                )
            if place_xy:
                arena_env.scene.assets[args_cli.destination].set_initial_pose(
                    Pose(position_xyz=(place_xy[0], place_xy[1], rest_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
                )
        elif getattr(args_cli, "randomize_objects", True):
            arena_env.scene.assets[args_cli.object].set_initial_pose(
                PoseRange(position_xyz_min=(0.26, 0.02, rest_z), position_xyz_max=(0.37, 0.26, rest_z))
            )
            arena_env.scene.assets[args_cli.destination].set_initial_pose(
                PoseRange(position_xyz_min=(0.26, -0.26, rest_z), position_xyz_max=(0.37, -0.02, rest_z))
            )
        # else: keep the parent's fixed training-like layout.
        return arena_env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        Go2AirbotPlayMapleTablePickPlaceEnvironment.add_cli_args(parser)
        parser.add_argument(
            "--arm_action", choices=["ik_eef", "joint"], default="ik_eef",
            help="Arm action space: 'ik_eef' (differential-IK EEF pose) or 'joint' (absolute joint targets).",
        )
        parser.add_argument(
            "--randomize_objects", action=argparse.BooleanOptionalAction, default=True,
            help=(
                "Randomize cube/bowl positions per episode over the training-distribution ranges"
                " (default). Use --no-randomize_objects for the fixed training-like layout."
            ),
        )
        parser.add_argument(
            "--pick_pos_xy", type=float, nargs=2, default=None,
            help="Explicit fixed world (x, y) for the pick object (overrides randomization); z = table_z+0.03.",
        )
        parser.add_argument(
            "--place_pos_xy", type=float, nargs=2, default=None,
            help="Explicit fixed world (x, y) for the destination (overrides randomization); z = table_z+0.03.",
        )
