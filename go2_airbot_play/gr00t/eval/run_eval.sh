#!/usr/bin/env bash
# Copyright (c) 2026, WFMRFM project.
# SPDX-License-Identifier: Apache-2.0
#
# Closed-loop GR00T evaluation for the Go2 + Airbot Play pick-place env, run INSIDE the Arena
# Docker container. Starts a GR00T policy server (dataset replay OR a real checkpoint), waits
# for it, then runs Arena's policy_runner with our IK-EEF eval env + custom closed-loop policy,
# and finally stops the server. Produces per-episode results + an HTML report + camera videos.
#
# Usage (from the repo root, inside the container):
#   # Replay rehearsal (no trained model) — validates the whole pipeline:
#   bash go2_airbot_play/gr00t/eval/run_eval.sh replay \
#       scratchpad_out/datasets/go2airbot_pickplace/lerobot 10
#
#   # Real checkpoint (when it arrives):
#   bash go2_airbot_play/gr00t/eval/run_eval.sh model /path/to/checkpoint-XXXX 10
#
# Args: <mode: replay|model|client> <dataset_dir|checkpoint_dir|-> [num_episodes=10] [port=5555]
#
# Remote server (server on another machine, e.g. the training server running GR00T's own
# run_gr00t_server.py): set REMOTE_HOST to that host and use mode "client" — this script then
# does NOT start/stop a local server, it only connects the Arena eval client to REMOTE_HOST:PORT.
#   REMOTE_HOST=10.0.0.5 bash .../run_eval.sh client - 20 5555
set -euo pipefail

MODE="${1:?mode required: replay|model|client}"
SRC="${2:?dataset dir (replay) / checkpoint dir (model) / '-' (client)}"
NUM_EPISODES="${3:-10}"
PORT="${4:-5555}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
export PYTHONPATH="${REPO}/go2_airbot_play/src:${PYTHONPATH:-}"

SERVE="${REPO}/go2_airbot_play/gr00t/eval/serve_go2airbot_gr00t.py"
RUNNER="${REPO}/third_party/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py"
# Default policy = relative-EEF client; override with POLICY_TYPE for the joint client, e.g.
#   POLICY_TYPE=go2_airbot_play.eval.gr00t_joint_policy.Go2AirbotJointClosedloopPolicy
# (pair with EXTRA_ENV_ARGS="--arm_action joint").
POLICY="${POLICY_TYPE:-go2_airbot_play.eval.gr00t_eef_policy.Go2AirbotEefClosedloopPolicy}"
ENV_CLASS="go2_airbot_play.environment:Go2AirbotPlayMapleTablePickPlaceEvalEnvironment"
ENV_NAME="go2_airbot_play_maple_table_eval"
OUT="${REPO}/scratchpad_out/eval/go2airbot_gr00t"

if [[ "${MODE}" == "client" || "${REMOTE_HOST}" != "localhost" ]]; then
  # --- Remote server: do NOT manage a local server; just connect to REMOTE_HOST:PORT. --------
  echo "[run_eval] client-only mode -> remote GR00T server at ${REMOTE_HOST}:${PORT}"
else
  # --- 0. pre-clean any stale local server on this port (a prior run SIGKILLed can't run its
  #        trap, so its server may still hold the port) ----------------------------------------
  pkill -9 -f "serve_go2airbot_gr00t.py" 2>/dev/null || true
  sleep 1
  # --- 1. start the GR00T server in the background -------------------------------------------
  if [[ "${MODE}" == "replay" ]]; then
    "${PY}" "${SERVE}" --dataset-path "${SRC}" --execution-horizon 16 --port "${PORT}" \
      --video-backend "${VIDEO_BACKEND:-opencv}" &
  else
    "${PY}" "${SERVE}" --model-path "${SRC}" --port "${PORT}" &
  fi
  SERVER_PID=$!
  trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
fi

# --- 2. wait until the server answers a ping ------------------------------------------------
echo "[run_eval] waiting for GR00T server on ${REMOTE_HOST}:${PORT} ..."
"${PY}" - "${REMOTE_HOST}" "${PORT}" <<'PYWAIT'
import sys, time
from gr00t.policy.server_client import PolicyClient
host, port = sys.argv[1], int(sys.argv[2])
for i in range(120):
    c = PolicyClient(host=host, port=port)
    if c.ping():
        print(f"[run_eval] server ready after {i}s"); sys.exit(0)
    time.sleep(1)
sys.exit("[run_eval] server did not become ready in 120s")
PYWAIT

# --- 3. run the closed-loop rollout ---------------------------------------------------------
# table_z / robot_x / object / destination match scripts/collect_pickplace_dataset.py so the
# scene the policy sees matches the training distribution.
# Set NUM_STEPS=<n> to bound the run by steps (smoke test) instead of episodes.
if [[ -n "${NUM_STEPS:-}" ]]; then
  LEN_ARGS=(--num_steps "${NUM_STEPS}")
else
  LEN_ARGS=(--num_episodes "${NUM_EPISODES}")
fi
# NOTE: for an external environment Arena registers the env NAME as an argparse subcommand,
# so main-parser flags (policy/runner/shared) go BEFORE the env name and the env-specific
# flags (--object/--table_z/...) go AFTER it (they live on the subparser).
"${PY}" "${RUNNER}" \
  --policy_type "${POLICY}" \
  --external_environment_class_path "${ENV_CLASS}" \
  --enable_cameras --headless --num_envs 1 \
  "${LEN_ARGS[@]}" \
  --remote_host "${REMOTE_HOST}" --remote_port "${PORT}" ${EXTRA_POLICY_ARGS:-} \
  --record_camera_video --output_base_dir "${OUT}" \
  "${ENV_NAME}" \
  --object rubiks_cube_hot3d_robolab --destination bowl_ycb_robolab \
  --table_z 0.40 --robot_x -0.15 ${EXTRA_ENV_ARGS:-}

echo "[run_eval] done. Report + videos under: ${OUT}"
