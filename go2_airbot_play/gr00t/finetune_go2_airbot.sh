#!/usr/bin/env bash
# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tune GR00T N1.7 on a Go2 + Airbot Play dataset (NEW_EMBODIMENT).
# Run from the GR00T repo root with its uv env available:
#   cd Isaac-GR00T && bash <this>/finetune_go2_airbot.sh <lerobot_dir> [output_dir] [modality_config.py]
#
# The modality config MUST match the dataset's meta/modality.json:
#   * Phase-1 static-base pick-place (arm+gripper only) -> go2_airbot_arm_data_config.py
#   * Full decoupled-WBC (adds base_height/navigate)    -> go2_airbot_wbc_data_config.py  (default)
# Prerequisite: run gr00t/data/stats.py on the dataset first (generates meta/stats.json +
# relative_stats.json for the chosen embodiment tag).
set -euo pipefail

PREP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${1:?usage: finetune_go2_airbot.sh <lerobot_dataset_dir> [output_dir] [modality_config.py]}"
OUTPUT_DIR="${2:-/mnt/nas2/users/shkim/work/projects/wfmrfm/scratchpad_out/gr00t_go2_airbot}"
MODALITY_CONFIG="${3:-${PREP_DIR}/go2_airbot_wbc_data_config.py}"

export NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 20000 \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    --use-wandb

# Open-loop sanity eval on a training trajectory (see finetune_new_embodiment.md).
# Use the modality-keys of the chosen config: Phase-1 -> "arm_eef gripper";
# full WBC -> "arm_eef gripper base_height_command navigate_command".
#   uv run python gr00t/eval/open_loop_eval.py \
#       --dataset-path "${DATASET_PATH}" \
#       --embodiment-tag NEW_EMBODIMENT \
#       --model-path "${OUTPUT_DIR}/checkpoint-20000" \
#       --traj-ids 0 --action-horizon 16 --steps 400 \
#       --modality-keys arm_eef gripper
