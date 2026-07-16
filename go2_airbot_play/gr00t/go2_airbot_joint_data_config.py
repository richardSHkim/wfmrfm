# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 NEW_EMBODIMENT modality config — Go2 + Airbot Play **ABSOLUTE JOINT** (Phase-1).

Joint-space variant of the arm+gripper contract: the arm is 6 **absolute joint** targets and the
gripper is 1 absolute opening — no EEF, no IK, no rot6d, no relative reference. This mirrors the
proven small-data recipe (a prior N1.5 pick-place succeeded with ~20 teleop episodes / 3000 steps
using a joint-based action space) and removes two variables that hurt the relative-EEF closed-loop:
the in-the-loop IK solve and the documented relative-action drift.

Contrast with ``go2_airbot_arm_data_config.py`` (relative-EEF):
* action = ``arm`` (6, ABSOLUTE, NON_EEF, DEFAULT) + ``gripper`` (1, ABSOLUTE, NON_EEF, DEFAULT).
* state  = ``arm`` (6) + ``gripper`` (1) — no ``arm_eef`` (a relative action needs an EEF reference
  via ``state_key``; an absolute joint action does not, so EEF proprio is dropped).
* ABSOLUTE rep ⇒ **no ``relative_stats.json`` needed** — plain ``generate_stats`` (min/max) suffices,
  which also side-steps the EEF relative-stats ``NotImplementedError`` on older Isaac-GR00T.

Deploy: pair with the joint-action eval env (arm = ``JointPositionActionCfg``, no IK) — the model
outputs joint targets that go straight to the arm actuators.
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

ACTION_HORIZON = 16

go2_airbot_joint_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            # arm: 6 absolute joint targets (joint1..joint6), joint-space control.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper: absolute opening target (0 = closed .. ~0.072 m = open).
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

register_modality_config(go2_airbot_joint_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
