# Isaac Lab / IsaacLab-Arena Docker 세팅 (superb-tony)

호스트: `superb-tony` · GPU: RTX A5000 ×4 (RT Core O → Isaac Sim 6.0 렌더링 지원)
프로젝트 경로: `/mnt/nas2/users/shkim/work/projects/wfmrfm`

> **중요**: 아래 명령은 **호스트 셸**에서 `docker` 그룹 권한(또는 `sudo`)으로 실행한다.
> Claude 작업 세션은 docker 접근 권한이 없는 컨테이너 안이라 여기선 빌드가 안 된다.
> Arena는 빌드/실행 헬퍼(`docker/run_docker.sh`)를 제공하므로 Dockerfile을 직접 다룰 필요는 없다.

---

## 0. 사전 점검

```bash
# 드라이버 (이미 OK: 580.126.09 / CUDA 13.0)
nvidia-smi

# docker 설치/데몬/권한
docker --version
docker info --format '{{.ServerVersion}}'      # 데몬 도달 확인
groups | grep -q docker && echo "docker group OK" || echo "need docker group or sudo"

# 디스크 여유 — 이미지가 수십 GB. docker data-root 파티션 여유가 충분한지 확인
docker info --format 'Docker Root Dir: {{.DockerRootDir}}'
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

여유가 부족하면 data-root를 큰 파티션으로 옮긴다 (`/etc/docker/daemon.json`의 `"data-root"`).

---

## 1. Docker Engine (미설치 시)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"      # 로그아웃/재로그인 후 그룹 반영
```

## 2. NVIDIA Container Toolkit (필수 — `--runtime=nvidia` 용)

```bash
# 리포 등록
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# docker에 nvidia 런타임 등록 후 데몬 재시작
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# GPU 패스스루 확인
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 3. NGC 로그인 (베이스 이미지 pull)

베이스 이미지: `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` (Dockerfile의 `ARG BASE_IMAGE`).

```bash
# NGC API key: https://ngc.nvidia.com/setup/api-key
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC_API_KEY>
```

> `-dev2` 태그는 NVIDIA dev 이미지라 특정 NGC org 접근이 필요할 수 있다.
> pull이 막히면 NGC 접근 권한을 확인하거나 정식 릴리스 태그로 `BASE_IMAGE`를 바꾼다.

---

## 4. Arena 컨테이너 빌드 + 진입

```bash
cd /mnt/nas2/users/shkim/work/projects/wfmrfm/third_party/IsaacLab-Arena

# 최초 빌드(오래 걸림) + 컨테이너 실행/진입. 리포 루트가 /workspaces/isaaclab_arena 로 마운트됨.
# 데이터/모델/eval 호스트 디렉터리를 함께 마운트하려면 -d / -m / -e 사용.
./docker/run_docker.sh \
  -d "$HOME/datasets" \
  -m "$HOME/models" \
  -e "$HOME/eval"

# 옵션:
#   -r   이미지 강제 재빌드
#   -R   캐시 없이 강제 재빌드
#   -s <suffix>  컨테이너 이름 접미사(여러 개 동시 실행)
#   -h   도움말
```

`run_docker.sh`가 붙이는 실행 플래그: `--privileged --runtime=nvidia --gpus=all
--net=host --ipc=host` + X11/SSH/gh/cache 마운트. 이미 떠 있으면 자동으로 attach.

### GPU 선택 (권장)

GPU 1이 `ERR!` 상태이므로 정상 GPU만 노출하는 게 안전하다. 호스트에서:

```bash
export NVIDIA_VISIBLE_DEVICES=0        # 또는 여유 있는 GPU 번호
```
또는 `run_docker.sh`의 `--gpus=all`을 `--gpus '"device=0"'`로 수정.

### 헤드리스 서버 (X 서버 없음) — `xhost` 우회

SSH/tmux로만 접속하는 헤드리스 서버(`superb-tony`가 이 경우)에는 X 서버가 없다.
`run_docker.sh`는 `docker run` **직전**에 항상 `xhost +local:docker`를 호출하는데,
스크립트 맨 위 `set -e` 때문에 X 서버가 없으면 여기서 스크립트가 죽고 **컨테이너가
아예 뜨지 않는다**. 증상:

```text
xhost:  unable to open display ""
# → 이후 docker run 이 실행되지 않음 (컨테이너 미생성)
```

헤드리스로 돌릴 거라도 이 줄을 통과해야 컨테이너가 뜬다. `xhost`의 유일한 목적은
X 서버 접근 권한을 여는 것인데 헤드리스(영상 녹화/WebRTC)에는 X가 필요 없으므로,
`xhost` 호출을 그냥 성공시켜 건너뛰면 된다. **설치·권한·스크립트 수정 없이** 되는
가장 간단한 방법은 `xhost`를 no-op bash 함수로 가리는 것:

```bash
xhost() { return 0; }        # 실제 X 서버 없이 xhost 호출을 no-op 처리
export -f xhost              # 자식 bash(run_docker.sh)로 함수 전파

CUDA_VISIBLE_DEVICES=1 ./docker/run_docker.sh \
  -d "$HOME/datasets" -m "$HOME/models" -e "$HOME/eval"
```

`run_docker.sh`가 `#!/bin/bash`라 `export -f`한 함수를 상속받아 외부 `xhost` 대신
이 함수를 호출한다.

대안 — 다른 셸이거나 함수 방식이 꺼려지면 PATH에 스텁을 배치:

```bash
mkdir -p ~/.local/fakebin
printf '#!/bin/sh\nexit 0\n' > ~/.local/fakebin/xhost && chmod +x ~/.local/fakebin/xhost
PATH="$HOME/.local/fakebin:$PATH" CUDA_VISIBLE_DEVICES=1 ./docker/run_docker.sh \
  -d "$HOME/datasets" -m "$HOME/models" -e "$HOME/eval"
```

> 근본책은 `run_docker.sh`의 `xhost +local:docker`를 `xhost +local:docker || true`로
> 바꾸는 것이지만, `AGENTS.md`가 `docker/` 변경은 사전 합의를 요구하므로 위 우회를
> 우선 사용한다. (Xvfb를 띄워 `DISPLAY`를 채우는 방법도 되지만 `xvfb` 설치가 필요하다.)

헤드리스 서버에서 시뮬레이터 화면을 확인하려면 GUI 대신 다음 두 경로를 쓴다:

- **영상 녹화 + HTML 리포트** (권장): eval 러너에 `--enable_cameras --record_camera_video
  --serve_evaluation_report` → 브라우저로 `http://<서버IP>:8000` 리포트 확인.
- **WebRTC 라이브스트림** (실시간 관찰용): `LIVESTREAM=2 ENABLE_CAMERAS=1`로 실행 후
  Isaac Sim WebRTC Streaming Client로 접속(시그널 49100/TCP, 미디어 47998/UDP;
  `run_docker.sh`가 `--net=host`라 포트는 그대로 노출됨).

---

## 5. Phase 1 Sanity (컨테이너 안에서)

컨테이너 안에선 `python`이 `/isaac-sim/python.sh`로 alias 돼 있다.
외부에서 `docker exec`로 돌릴 땐 명시 경로를 쓴다.

```bash
# 카메라(렌더) 없는 테스트부터 — GPU 렌더 파이프라인 문제와 분리
cd /workspaces/isaaclab_arena
pytest -m "not with_cameras"

# 그다음 렌더 포함 스모크 (RT 코어/Vulkan 동작 확인)
pytest -m "with_cameras"
```

호스트에서 exec로 돌리는 형태(컨테이너 이름은 `dev-container` 스킬로 조회, 하드코딩 금지):

```bash
docker exec "$ARENA_CONTAINER" su "$(id -un)" -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest -m 'not with_cameras'"
```

Sanity 통과 후 → `zero_action` 스모크 eval, 이어서 Go2+Airbot embodiment/task 스켈레톤 작업으로 진행.

---

## 트러블슈팅

- **`could not select device driver ... nvidia`** → 2단계 미완료. `nvidia-ctk runtime configure` 후 `docker restart`.
- **베이스 이미지 pull 403/denied** → 3단계 NGC 접근 권한 문제.
- **Vulkan/RT 관련 렌더 실패** → GPU 1(ERR) 배제, 정상 GPU 지정. `--gpus '"device=N"'`.
- **디스크 부족(빌드 중 no space)** → docker data-root 여유 확보(0단계).
- **`xhost: unable to open display ""` 후 컨테이너 미생성** → 헤드리스 서버라 X 서버가 없어 `run_docker.sh`가 `xhost` 줄(`set -e`)에서 죽은 것. 4장 "헤드리스 서버 — `xhost` 우회"의 `xhost` no-op 방법으로 해결.
- **X11/디스플레이 에러(렌더 테스트)** → 헤드리스면 `pytest -m "not with_cameras"`로 먼저 검증. 렌더/카메라 확인은 영상 녹화 또는 WebRTC 스트리밍(4장) 경로 사용.
