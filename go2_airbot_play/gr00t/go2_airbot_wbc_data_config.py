# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 ``NEW_EMBODIMENT`` modality config for the Go2 + Airbot Play decoupled-WBC robot.

This is the single modality configuration used for BOTH fine-tuning
(``launch_finetune.py --modality-config-path ...``) and closed-loop deployment
(Arena ``Gr00tRemoteClosedloopPolicy``, ``modality_config_path``). It encodes the
decoupled whole-body-control (WBC) action contract:

    GR00T (this VLA)  ->  arm (relative EEF) + gripper (absolute)
                          + base_height_command (absolute)
                          + navigate_command   (absolute [vx, vy, wz])
                              |
                              v
    the base commands feed our ALREADY-TRAINED RL locomotion policy
    (``eval/locomotion/.../exported/policy.onnx``), whose obs expects exactly
    ``velocity_commands(3) = [vx, vy, wz]`` + ``base_height_command(1)``.
    The 12 leg joints are produced by that policy and are NOT a VLA target.

Design (see ``README.md`` for the full rationale):

* **arm = RELATIVE + EEF + XYZ_ROT6D (9D)** — GR00T N1.7's headline representation
  (README.md #1 "Relative EEF Action Space"): deltas from the current pose, the
  representation shared with the 20K-hour human-video prior, best cross-embodiment
  generalization. The dataset stores the *absolute* 9D pose (xyz + rot6d) **in the
  ``base_link`` (arm-mount) frame, not world** — the Go2 base moves, so a world-frame EEF
  drifts and couples the arm to locomotion; ``base_link`` frame keeps it bounded and
  decoupled. The processor converts absolute->relative at train/inference time (verified in
  ``gr00t/data/state_action/state_action_processor.py:apply_action``); we never pre-store
  relative.
* **gripper = ABSOLUTE + NON_EEF** — binary open/close works better absolute.
* **base_height_command / navigate_command = ABSOLUTE + NON_EEF** — setpoints /
  velocity commands, inherently per-step absolute values (mirrors the G1
  ``unitree_g1_sim_wbc_config`` decoupled-WBC precedent).

``modality_keys`` here MUST match the keys in the dataset's ``meta/modality.json``
(the canonical file is ``modality.json`` in this directory, installed by the
``add_single_arm_eef.py`` post-process).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Action prediction horizon. 16 matches the SO-100 new-embodiment tutorial and the
# ``open_loop_eval.py`` default; the G1 whole-body config uses 50. If you change this you
# MUST regenerate stats: ``python gr00t/data/stats.py --dataset-path <ds> --embodiment-tag NEW_EMBODIMENT``.
ACTION_HORIZON = 16

go2_airbot_wbc_config = {
    # Two views: Go2 front (exterior) + wrist Intel RealSense D435i. NOTE: the Arena
    # HDF5->LeRobot converter emits only one POV video, so this full-WBC datagen path needs
    # the multi-cam converter extension (the direct convert_pickplace_to_lerobot.py already
    # writes both views).
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    # Proprioception: the absolute EEF pose (reference for the relative arm action), the arm
    # joint angles, and the gripper opening. Mirrors the DROID ``oxe_droid`` state layout
    # (eef_9d + gripper + joints) but single-arm.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm_eef", "arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "arm_eef",
            "gripper",
            "base_height_command",
            "navigate_command",
        ],
        action_configs=[
            # arm_eef: 9D end-effector pose (xyz + rot6d), controlled relative to the
            # current pose. state_key points at the matching absolute EEF proprio.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="arm_eef",
            ),
            # gripper: absolute opening target (0 = closed .. ~0.072 m = open).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # base_height_command: absolute target base height [~0.20 .. 0.36] m -> RL loco policy.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # navigate_command: absolute base velocity command [vx, vy, wz] -> RL loco policy.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Always register a custom embodiment under NEW_EMBODIMENT.
register_modality_config(go2_airbot_wbc_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
