# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab ``ArticulationCfg`` for the Go2 + Airbot Play + wrist D435i embodiment.

The config binds our converted USD (``third_party/go2_airbot_play_description``) to
three actuator groups:

- ``base_legs``  – the 12 Unitree Go2 leg joints, using the exact native
  ``UNITREE_GO2_CFG`` DC-motor gains so a Go2 locomotion RL policy transfers unchanged.
- ``arm``        – the 6 Airbot Play arm joints, position-tracking PD (provisional gains;
  see the TODO — replace with the Airbot Play datasheet torque/velocity limits).
- ``gripper``    – the Airbot Play parallel gripper. ``g2_joint`` is the driver
  (opening 0..0.072 m); ``g2_left_joint`` / ``g2_right_joint`` mimic it at x0.5 in the
  URDF but load as independent DOFs, so the control layer must couple them.
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

# --------------------------------------------------------------------------------------
# USD path. The asset lives in the ``go2_airbot_play_description`` submodule; resolve it
# by walking up from this file until the submodule is found (robust to package nesting /
# install location). Override with the GO2_AIRBOT_PLAY_USD env var when the asset is
# mounted elsewhere (e.g. a different path inside the Arena container).
# --------------------------------------------------------------------------------------
_USD_REL = (
    "third_party/go2_airbot_play_description/usd/go2_airbot_play.usd"
    "/go2_airbot_play_flat/go2_airbot_play_flat.usda"
)


def _resolve_usd_path() -> str:
    override = os.environ.get("GO2_AIRBOT_PLAY_USD")
    if override:
        return override
    for base in Path(__file__).resolve().parents:
        cand = base / _USD_REL
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"Could not locate {_USD_REL} above {__file__}; set GO2_AIRBOT_PLAY_USD."
    )


GO2_AIRBOT_PLAY_USD_PATH: str = _resolve_usd_path()

GO2_AIRBOT_PLAY_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_AIRBOT_PLAY_USD_PATH,
        activate_contact_sensors=True,  # Go2 feet contacts for locomotion
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            # --- Go2 legs: native UNITREE_GO2_CFG standing pose ---
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # --- Airbot Play arm: neutral "ready" pose — all joints 0 (arm extended forward,
            # gripper over the workspace in front of the robot; within joint limits) ---
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            # --- gripper: closed ---
            "g2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Go2 legs — EXACT native UNITREE_GO2_CFG values (RL-policy transfer).
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        # Airbot Play arm — position-tracking PD. Gains must be stiff enough to HOLD the
        # arm against gravity: at stiffness=150/effort=18 the arm collapsed to its joint
        # limit under zero-action (verified). These firmer gains hold the ready pose.
        # TODO: replace with the Airbot Play datasheet torque/velocity limits + tuned gains
        # (or add explicit gravity compensation) once available.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit=87.0,
            velocity_limit_sim=3.14,
            stiffness=800.0,
            damping=40.0,
        ),
        # Airbot Play parallel gripper (driver + mimic fingers, coupled by control layer).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["g2_joint", "g2_left_joint", "g2_right_joint"],
            effort_limit=100.0,
            velocity_limit_sim=10.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)
"""Articulation config for the Go2 + Airbot Play + wrist D435i robot."""
