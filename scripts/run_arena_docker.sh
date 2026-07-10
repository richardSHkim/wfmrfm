#!/usr/bin/env bash
#
# run_arena_docker.sh — IsaacLab-Arena 컨테이너를 헤드리스 서버에서 한 번에 실행.
#
# (헤드리스 관련: run_docker.sh 의 `xhost +local:docker` 줄은 X 서버가 없으면 실패해
# `set -e` 때문에 컨테이너가 뜨기 전에 종료됐다. 이제 그 줄은 run_docker.sh 에서 직접
# 주석 처리했으므로 이 래퍼에서 따로 처리하지 않는다.)
#
# 사용법:
#   scripts/run_arena_docker.sh                 # 기본 마운트로 실행/진입
#   CUDA_VISIBLE_DEVICES=1 scripts/run_arena_docker.sh
#   scripts/run_arena_docker.sh -r              # 이미지 강제 재빌드 등 run_docker.sh 플래그 전달
#   DATASETS_DIR=/data MODELS_DIR=/models scripts/run_arena_docker.sh
#
# 환경변수:
#   DATASETS_DIR / MODELS_DIR / EVAL_DIR  — 호스트 마운트 경로 (기본: <프로젝트 루트>/datasets|models|eval)
#   CUDA_VISIBLE_DEVICES                  — 설정돼 있으면 그대로 상속 (GPU 선택)
#   CLAUDE_CFG                            — Claude Code 설정 디렉터리 (기본: /mnt/nas2/users/shkim/cache/.claude)
#                                           동일 경로로 마운트 + CLAUDE_CONFIG_DIR 로 연결된다.
#   EXTRA_MOUNTS                          — 추가로 "동일 절대경로"에 bind-mount 할 호스트 경로들 (공백 구분)
#   EXTRA_ENV                             — 컨테이너로 전달할 env 목록 (공백 구분, "VAR=값" 또는 "VAR")
#
# 추가 인자(-r, -R, -s <suffix>, 또는 컨테이너에서 실행할 명령)는 run_docker.sh 로 그대로 전달된다.
#
# Claude Code:
#   claude 바이너리는 이미지(Dockerfile.isaaclab_arena)에 이미 설치돼 있으므로 재설치가
#   필요 없다. 이 래퍼는 claude 설정/인증/메모리 디렉터리와 이 프로젝트 루트를 호스트와
#   "동일한 절대경로"로 컨테이너에 마운트하도록 run_docker.sh 의 EXTRA_MOUNTS 에 넘긴다.
#   동일 경로로 얹기 때문에 memory 프로젝트 슬러그가 실제 경로와 맞아떨어지고 ~/.claude
#   기본 위치도 그대로 동작한다. 컨테이너 안에서 이 프로젝트를 작업하려면 그 경로로 cd:
#     cd <프로젝트 루트> && claude

set -euo pipefail

# 이 스크립트 위치 기준으로 IsaacLab-Arena 저장소 루트를 찾는다 (CWD 무관).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
ARENA_DIR="$REPO_ROOT/third_party/IsaacLab-Arena"

if [ ! -x "$ARENA_DIR/docker/run_docker.sh" ]; then
    echo "[run-arena] ERROR: $ARENA_DIR/docker/run_docker.sh 를 찾을 수 없다." >&2
    echo "[run-arena] IsaacLab-Arena 서브모듈이 체크아웃돼 있는지 확인:" >&2
    echo "[run-arena]   git submodule update --init third_party/IsaacLab-Arena" >&2
    exit 1
fi

# 호스트 마운트 경로 (없으면 run_docker.sh 가 해당 볼륨을 조용히 건너뛴다).
# 기본값은 이 프로젝트 루트 기준 — 데이터/모델/eval 산출물을 저장소 옆에 모아둔다.
DATASETS_DIR="${DATASETS_DIR:-$REPO_ROOT/datasets}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
EVAL_DIR="${EVAL_DIR:-$REPO_ROOT/eval}"

# --- Claude Code 설정/프로젝트 마운트 --------------------------------------
# claude 바이너리는 이미지에 설치돼 있다. 여기서는 설정(인증·memory)과 이 프로젝트
# 루트를 호스트와 동일한 절대경로로 컨테이너에 넣는다. run_docker.sh 가 EXTRA_MOUNTS
# 를 읽어 각 경로를 `-v <path>:<path>` 로 마운트한다(존재하지 않으면 건너뜀).
# CLAUDE_CFG 는 컨테이너 기본 위치(~/.claude)가 아니므로, 동일 경로로 마운트한 뒤
# CLAUDE_CONFIG_DIR 로 그 경로를 가리켜 claude 가 실제로 사용하게 한다(EXTRA_ENV).
CLAUDE_CFG="${CLAUDE_CFG:-/mnt/nas2/users/shkim/cache/.claude}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-} $REPO_ROOT $CLAUDE_CFG"
EXTRA_ENV="${EXTRA_ENV:-} CLAUDE_CONFIG_DIR=$CLAUDE_CFG"
export EXTRA_MOUNTS EXTRA_ENV
echo "[run-arena] 동일 경로 마운트(EXTRA_MOUNTS): $REPO_ROOT  $CLAUDE_CFG"
echo "[run-arena] 컨테이너 env(EXTRA_ENV): CLAUDE_CONFIG_DIR=$CLAUDE_CFG"
# ---------------------------------------------------------------------------

# run_docker.sh 는 현재 작업 디렉터리를 컨테이너에 마운트하고 컨테이너 이름도
# 저장소 디렉터리명에서 유도하므로, 반드시 Arena 저장소 루트에서 실행해야 한다.
cd "$ARENA_DIR"

echo "[run-arena] Arena 저장소: $ARENA_DIR"
echo "[run-arena] 마운트: datasets=$DATASETS_DIR  models=$MODELS_DIR  eval=$EVAL_DIR"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "[run-arena] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec ./docker/run_docker.sh \
    -d "$DATASETS_DIR" \
    -m "$MODELS_DIR" \
    -e "$EVAL_DIR" \
    "$@"
