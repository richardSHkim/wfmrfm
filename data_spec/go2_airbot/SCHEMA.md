# Go2 + Airbot LeRobot 수집 스키마 (초안)

배포 정책 = **GR00T N1.7**. 동일 원본으로 Cosmos WFM/FDM/IDM(+Policy 실험)도 파생.
LeRobot **v2.1** (GR00T 네이티브; Cosmos는 v2.x/v3.0 모두 수용).

## 원칙
- **모든 값 absolute 저장** → 로더/프로세서가 relative로 변환 (GR00T `data_config.md:202`, Cosmos `pose_abs_to_rel`).
- **joint + EEF 둘 다 저장** → 주경로(relative EEF)와 부경로(Cosmos joint_pos+use_state) 양쪽 커버.
- 모든 카메라·state·action **동일 타임스탬프 동기화** (FDM/IDM는 프레임-동작 정렬이 생명).

## 디렉토리 레이아웃 (v2.1)
```
go2_airbot_dataset/
├── meta/
│   ├── info.json          # features 스키마 (이 초안)
│   ├── modality.json      # GR00T 슬라이싱 (이 초안)
│   ├── episodes.jsonl     # {episode_index, tasks, length} 라인별
│   ├── tasks.jsonl        # {task_index, task} 언어 지시 사전
│   └── stats.json         # 정규화 통계 (수집 후 gr00t/data/stats.py 생성)
├── data/
│   └── chunk-000/episode_000000.parquet   # per-frame state/action/index/task_index...
└── videos/
    └── chunk-000/
        ├── observation.images.wrist/episode_000000.mp4
        ├── observation.images.exterior_left/episode_000000.mp4
        └── observation.images.exterior_right/episode_000000.mp4
```

## observation.state — 41D (absolute proprioception)
| idx | 키 | dim | 설명 |
|----|----|----|----|
| 0:9   | arm_eef_9d          | 9  | Airbot EEF pose = xyz(3)+rot6d(6), **FK로 산출**, base 프레임 |
| 9:10  | gripper_position    | 1  | 그리퍼 개방량 (정규화 0~1 또는 m) |
| 10:16 | arm_joint_position  | 6  | Airbot 6관절 각도 (rad) |
| 16:28 | leg_joint_position  | 12 | Go2 12관절 (FR/FL/RR/RL × hip/thigh/calf, rad) |
| 28:35 | base_pose           | 7  | Go2 base 위치 xyz(3) + quat wxyz(4), odom 프레임 |
| 35:41 | base_velocity       | 6  | base 선속도(3) + 각속도(3) |

## action — 32D (absolute commands)
| idx | 키 | dim | 소비자 | 설명 |
|----|----|----|----|----|
| 0:9   | arm_eef_9d            | 9  | **GR00T 팔 (RELATIVE)** / Cosmos WFM·FDM·IDM | 명령 EEF pose (명령 joint의 FK) |
| 9:10  | gripper_position      | 1  | **GR00T (ABSOLUTE)** | 명령 그리퍼 |
| 10:16 | arm_joint_position    | 6  | Cosmos Policy(joint_pos+use_state) 부경로 | 명령 팔 관절 |
| 16:28 | leg_joint_position    | 12 | RL 로코모션 / Cosmos 19D / FDM·IDM 컨텍스트 | 명령 다리 관절 (**GR00T VLA 타깃 아님**) |
| 28:31 | base_velocity_command | 3  | **GR00T 로코 인터페이스** (Phase2) | vx, vy, wz 고수준 명령 |
| 31:32 | base_height_command   | 1  | 선택 | 목표 base 높이 |

## 소비자별 사용 슬라이스
- **GR00T N1.7 (배포)**: action = `arm_eef_9d`(RELATIVE,EEF) + `gripper_position`(ABSOLUTE) + `base_velocity_command`(ABSOLUTE, Phase2). 다리 제외.
- **Cosmos WFM**: 영상 + 언어캡션만 (action 무시). Qwen3-VL로 caption 생성.
- **Cosmos FDM/IDM**: 기본 relative EEF = `arm_eef_9d`+`gripper` (10D `[pos3,rot6d6,grip1]`).
- **Cosmos Policy (실험)**: `joint_pos` 경로 = `leg_joint`(12)+`arm_joint`(6)+`gripper`(1) = raw 19D absolute + `use_state`.

## 카메라
- `wrist` (Airbot 손목) — 필수
- `exterior_left`, `exterior_right` (Go2 몸체 장착) — Cosmos `concat_view`(wrist 상단 + L/R 하단) 재현용
- native ≥640×480 저장 후 소비자별 다운샘플 (Cosmos 480p=640×360, GR00T 224/256)
- 원본 30fps → Cosmos 15fps / GR00T 20~30 리샘플

## Phase 처리
- **Phase 1** (팔 매니퓰, Go2 정지): leg_joint = 고정 standing pose, base_velocity/command = 0 (스키마 유지 위해 상수 기록).
- **Phase 2** (decoupled loco-manip): leg는 RL 컨트롤러 출력 기록, base_velocity_command = 조작/RL 명령.

## 수집 후 파이프라인
1. episodes.jsonl / tasks.jsonl 작성, parquet + mp4 저장 (LeRobotDataset API 사용 권장).
2. `python gr00t/data/stats.py --dataset-path <path> --embodiment-tag NEW_EMBODIMENT` → stats.json/relative_stats.json.
3. GR00T data_config.py에 ActionConfig 정의 (arm=RELATIVE+EEF, gripper=ABSOLUTE, base_vel=ABSOLUTE).
4. Cosmos: `caption_from_video`→`captions_to_sft_jsonl` (WFM), domain_utils.py `go2_airbot`=19 등록 + normalizer_stats/go2_airbot.json (FDM/IDM/Policy).

## 확인 필요 (수집 착수 전)
- gripper 단위/범위 (정규화 vs 미터), rot6d 규약 (열-major 6D 표준).
- base_pose 소스 (Go2 odom/IMU 융합 정확도) — Phase2 이동수집 신뢰도.
- EEF FK 기준 프레임 (Airbot base 링크) 및 tool 오프셋.
