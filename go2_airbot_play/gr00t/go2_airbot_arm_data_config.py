# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 NEW_EMBODIMENT modality config — Go2 + Airbot Play **Phase-1 (static base)**.

Arm-manipulation-only subset of the full decoupled-WBC contract
(``go2_airbot_wbc_data_config.py``): the arm (relative EEF) + gripper, with NO
``base_height_command`` / ``navigate_command``. Use this for datasets collected with the Go2
base fixed (e.g. ``scripts/collect_pickplace_dataset.py``, ``fix_base=True``), where the base
commands have no signal — including them would be a constant column that breaks min-max
normalization. When mobile-base teleop data exists, switch to the full WBC config.

Same arm representation as the full config (relative EEF, XYZ_ROT6D, base_link frame) so a
Phase-1 checkpoint is a valid warm-start for the full-WBC fine-tune.
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

go2_airbot_arm_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm_eef", "arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["arm_eef", "gripper"],
        action_configs=[
            # arm_eef: 9D EEF pose (xyz + rot6d) in base_link frame, controlled relative.
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
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(go2_airbot_arm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
