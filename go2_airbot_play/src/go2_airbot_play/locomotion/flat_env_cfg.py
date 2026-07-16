# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Flat-terrain velocity + crouch locomotion env for the Go2 + Airbot Play embodiment.

This reuses Isaac Lab's proven Unitree-Go2 flat velocity task and swaps in our 21-DOF
Go2 + Airbot Play robot, treating the arm as a *moving* payload:

* Action / policy observation are restricted to the **12 leg joints** (arm + gripper are
  NOT policy-controlled) — the leg joints are DOF 0-11 in the exact stock Go2 order.
* The arm is driven by a **continuous random-motion event** (see ``mdp.randomize_arm_joint_targets``)
  so its CoM shifts constantly, and a **random gripper payload** (0 – 1.5 kg, the Airbot
  Play *rated* payload) is added at startup — the lower-body policy must stay stable under
  a moving, variably-loaded arm.
* A **commanded trunk height** (crouch/stand) is added via ``mdp.UniformBaseHeightCommand``
  + a height-tracking reward, on top of the SE(2) velocity command.
* Standing-still envs are increased (station-keeping) so the base holds rock-steady while
  the arm manipulates.

Our trunk body is ``base_link`` (stock Go2 uses ``base``), so the base-referencing
events/terminations are re-pointed accordingly.  Sim-only (no real-robot sim2real DR
beyond the stock friction/mass randomisation).
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,
    EventsCfg,
    ObservationsCfg,
    RewardsCfg,
    StartupEventsCfg,
    TerminationsCfg,
)

from go2_airbot_play.assets.articulation_cfg import GO2_AIRBOT_PLAY_CFG

from . import mdp

# The 12 Go2 leg joints, in the native articulation order (DOF 0-11) — the policy's
# action & proprioception are restricted to these.
LEG_JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]

# Commanded trunk-height range (world z, m). ~0.34 is the free-standing Go2 height;
# 0.20 is a deep crouch. Payload-loaded deep crouch is a stretch target the policy tracks.
HEIGHT_RANGE = (0.20, 0.36)
# Airbot Play rated (stable) payload = 1.5 kg (max 3.5 kg); randomise the gripper load 0..rated.
PAYLOAD_RANGE_KG = (0.0, 1.5)

# Locomotion robot cfg: arm parked in a CoM-centred "tuck" (joint2=-1.0, joint3=2.4 keeps the
# horizontal CoM offset ~0). This is the reset/default pose. In the bootstrap stage the arm
# holds here (a well-balanced static payload); in the full stage the arm is driven away from
# it by the random-motion event. Centring the CoM is what lets a from-scratch policy learn to
# stand — an arm extended forward (all-zeros) or flailing from step 0 topples it instantly.
_LOCO_JOINT_POS = {
    ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0, ".*_calf_joint": -1.5,
    "joint1": 0.0, "joint2": -1.0, "joint3": 2.4, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
    "g2_joint": 0.0,
}
GO2_AIRBOT_PLAY_LOCO_CFG = GO2_AIRBOT_PLAY_CFG.replace(
    init_state=GO2_AIRBOT_PLAY_CFG.init_state.replace(joint_pos=_LOCO_JOINT_POS)
)


@configclass
class Go2AirbotCommandsCfg(CommandsCfg):
    """SE(2) velocity command (inherited) + a commanded trunk height."""

    base_height = mdp.UniformBaseHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        ranges=mdp.UniformBaseHeightCommandCfg.Ranges(height=HEIGHT_RANGE),
    )


@configclass
class Go2AirbotPolicyCfg(ObservationsCfg.PolicyCfg):
    """Policy obs (inherited) + the commanded trunk height so the policy can crouch on demand."""

    base_height_command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_height"})


@configclass
class Go2AirbotObservationsCfg(ObservationsCfg):
    policy: Go2AirbotPolicyCfg = Go2AirbotPolicyCfg()


@configclass
class Go2AirbotRewardsCfg(RewardsCfg):
    """Stock velocity rewards + trunk-height tracking (crouch/stand)."""

    track_base_height = RewTerm(
        func=mdp.track_base_height_exp,
        weight=2.0,
        params={"command_name": "base_height", "std": 0.06},
    )


@configclass
class Go2AirbotTerminationsCfg(TerminationsCfg):
    """Contact-free fall detection.

    Our URDF->USD conversion nests the rigid bodies by the kinematic tree
    (``base_link/FL_hip/.../FL_foot``), whereas Isaac Lab's ContactSensor requires bodies as
    direct children of the robot root.  So instead of the contact-based ``base_contact``
    termination we detect falls from the root state: the trunk tipping over, or dropping
    below any plausible (even fully-crouched) height.
    """

    # NOTE: the inherited ``base_contact`` term is nulled in the env's __post_init__ (after
    # the stock Go2 __post_init__, which still configures it, has run).
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 1.0})
    root_too_low = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.12})


@configclass
class Go2AirbotEventsCfg(EventsCfg, StartupEventsCfg):
    """Go2 event tuning (re-pointed to ``base_link``) + moving arm + random gripper payload."""

    # Continuous random arm motion: every ~0.2-0.5 s nudge the arm targets by a small step so
    # the arm drifts smoothly through its workspace (manipulation-speed time-varying CoM
    # disturbance — a learnable robustness target, not a violent whip).
    arm_random_motion = EventTerm(
        func=mdp.randomize_arm_joint_targets,
        mode="interval",
        interval_range_s=(0.2, 0.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "step_size": 0.3,
            "range_fraction": 0.9,
        },
    )
    # Random object held in the gripper (0 = empty hand, up to the Airbot Play rated payload).
    add_gripper_payload = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="g2_base_link"),
            "mass_distribution_params": PAYLOAD_RANGE_KG,
            "operation": "add",
        },
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Mirror the stock Go2 event tuning, re-pointed to our trunk body ``base_link``.
        self.push_robot = None
        self.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.base_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        # base mass randomisation on the trunk (not the default "base").
        self.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.base_com = None


@configclass
class Go2AirbotPlayFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    # Concrete (non-preset) managers replacing the stock Go2 ones.
    observations: Go2AirbotObservationsCfg = Go2AirbotObservationsCfg()
    commands: Go2AirbotCommandsCfg = Go2AirbotCommandsCfg()
    events: Go2AirbotEventsCfg = Go2AirbotEventsCfg()
    rewards: Go2AirbotRewardsCfg = Go2AirbotRewardsCfg()
    terminations: Go2AirbotTerminationsCfg = Go2AirbotTerminationsCfg()

    # When False (bootstrap stage), the arm is a *static* CoM-centred payload and no gripper
    # payload is randomised — so a from-scratch policy can learn to stand/walk/crouch first.
    # When True (full stage, warm-started from the bootstrap policy), the arm swings randomly
    # and a 0-1.5 kg gripper payload is added, making the policy robust to a moving, loaded arm.
    enable_arm_disturbance: bool = True

    def __post_init__(self):
        super().__post_init__()

        # -- robot: 21-DOF Go2 + Airbot Play (arm parked in CoM-centred tuck) --
        self.scene.robot = GO2_AIRBOT_PLAY_LOCO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # -- staged curriculum: disable the arm disturbance for the bootstrap stage --
        if not self.enable_arm_disturbance:
            self.events.arm_random_motion = None
            self.events.add_gripper_payload = None

        # -- contact-free: our nested USD body layout is incompatible with the ContactSensor
        #    (bodies aren't direct children of the robot root), so drop the contact sensor and
        #    the reward that needs it; falls are caught by orientation/height terminations. --
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.terminations.base_contact = None

        # -- action: only the 12 leg joints (arm/gripper are payload, not policy-controlled) --
        self.actions.joint_pos = base_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEG_JOINTS,
            scale=0.25,
            use_default_offset=True,
            preserve_order=True,
        )

        # -- proprioception restricted to the leg joints (matches the action interface) --
        legs = SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": legs}
        self.observations.policy.joint_vel.params = {"asset_cfg": legs}
        # flat terrain: no height scan (super() already set it None on our obs group)

        # -- station-keeping: many more standing-still envs (base holds steady for manipulation) --
        self.commands.base_velocity.rel_standing_envs = 0.2
        # headless training: command debug-vis markers pull in omni.kit (unavailable when the
        # cfg is imported before the app) and are not needed — disable them.
        self.commands.base_velocity.debug_vis = False


@configclass
class Go2AirbotPlayFlatBootstrapEnvCfg(Go2AirbotPlayFlatEnvCfg):
    """Stage 1: static CoM-centred arm, no gripper payload — learn to stand/walk/turn/crouch."""

    enable_arm_disturbance: bool = False


@configclass
class Go2AirbotPlayFlatEnvCfg_PLAY(Go2AirbotPlayFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class Go2AirbotPlayFlatBootstrapEnvCfg_PLAY(Go2AirbotPlayFlatBootstrapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
