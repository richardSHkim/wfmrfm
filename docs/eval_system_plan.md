# Go2 + Airbot Play 평가 시스템 구축 계획

**목표**: 데이터 수집/학습 이전에 **평가 시스템을 먼저** 구축한다.
- **RFM = Isaac GR00T N1.7** (배포 policy) → **IsaacLab-Arena** 백본으로 closed-loop 평가
- **WFM = NVIDIA Cosmos3** (world model) → Cosmos 메트릭 harness로 평가
- **Embodiment**: Unitree Go2 (12-DoF 다리) + Airbot Play (6-DoF 팔 + 그리퍼). 휴머노이드 아님.
- **아키텍처**: decoupled — GR00T가 팔/그리퍼, RL policy가 locomotion. Arena의 G1 loco-manip
  NEW_EMBODIMENT 패턴(WBC command 인터페이스)에 ~1:1 매핑.

**핵심 원칙**
1. 초기엔 실제 모델/데이터 없이 **sanity policy**(zero_action / ReplayPolicy / random)로 harness 완성 → 나중에 모델·데이터를 "꽂기만" 하면 되게.
2. **WFM eval은 RT 코어 불필요** → RFM eval(Isaac Sim, A5000 렌더 의존)과 **병렬 진행**.

---

## Phase 0 — 환경 부트스트랩 (진행 중)

Isaac Lab / Arena Docker 빌드 → sanity. RT 렌더가 A5000에서 도는지 검증.

- [진행중] `./docker/run_docker.sh` 빌드
- [ ] `pytest -m "not with_cameras"` (렌더 무관 로직 검증)
- [ ] `pytest -m "with_cameras"` (RT 코어 / Vulkan 렌더 파이프라인 검증)
- [ ] `zero_action` 스모크 eval — Arena env가 뜨고 스텝/리포트가 도는지
- 주의: GPU 1(`ERR!`) 배제, 여유 VRAM 확인 (`docs/isaac_docker_setup.md`)

**완료 기준**: 렌더 포함 테스트 통과 + zero_action eval이 success_rate/영상/JSONL 리포트를 뱉음.

---

## Phase 1 — Embodiment 에셋 준비 (Isaac Sim 의존)

Go2+Airbot 합성 articulation을 만든다. Docker 필요(Isaac Lab URDF→USD 컨버터).

- [ ] Airbot Play URDF 확보 → Isaac Lab converter로 **URDF→USD**
- [ ] Go2 USD 재사용 (Isaac Lab 기본 제공 Unitree Go2)
- [ ] Airbot을 Go2 base에 마운트 → **합성 USD/articulation** (마운트 프레임/오프셋 정의)
- [ ] **Joint order 확정**: 12 leg + 6 arm + 1 gripper, 인덱스 매핑표
- 참고: `third_party` 내 `go2_arx`(legged-robots-manipulation) 템플릿, RoboDuet EE-tracking

**완료 기준**: Arena 씬에 로드되어 물리적으로 안정(넘어지지 않음)하게 서 있는 Go2+Airbot USD.

---

## Phase 2 — RFM eval 통합 (IsaacLab-Arena)

Arena out-of-tree 확장 패키지로 Go2+Airbot embodiment + task를 등록하고 GR00T를 붙인다.

- [ ] 확장 패키지 스캐폴드 `isaaclab_arena_go2airbot/`
      (`@register_*` + `--external_environment_class_path`)
- [ ] **Embodiment** 정의 — G1 loco-manip 참고. decoupled: GR00T=팔/그리퍼,
      RL=locomotion. command 인터페이스(navigate/base_height/… 중 필요한 것) 정의
- [ ] **GR00T modality config** (NEW_EMBODIMENT 태그) — state/action 키, 팔+그리퍼(+base vel)
- [ ] **Joint-order YAML** — GR00T action space ↔ Isaac Lab joint index 매핑
- [ ] **Loco-manip task** 정의 + **success 기준** (예: 목표 위치 이동 + 물체 pick&place)
- [ ] GR00T 연결 — Arena `Gr00tRemoteClosedloopPolicy` (ZMQ). **GR00T 서버는 A100/타 노드,
      sim은 A5000** 분리(client/server)
- [ ] Locomotion policy: 초기엔 placeholder/Isaac Lab 기본 Go2 velocity task 사전학습 policy
- [ ] 메트릭: Arena pooled `success_rate` + HTML/영상/JSONL 리포트

**완료 기준**: ReplayPolicy/zero_action으로 loco-manip task가 end-to-end 돌고 success_rate가 집계됨.
(실제 GR00T 체크포인트는 데이터/학습 이후 교체)

---

## Phase 3 — WFM eval harness (Cosmos) · **병렬 가능, RT 불필요**

Cosmos3 world model 평가. Isaac Sim 없이 A100/A5000 compute만으로 개발 가능.

- [ ] Cosmos `metrics.py`의 미연결 유틸 wiring:
      PSNR / dynamic-PSNR / SSIM, action MSE/MAE/grouped/geodesic rotation error
- [ ] **FVD 추가** (WFM 대표 헤드라인 메트릭 — 현재 Cosmos에 없음)
- [ ] **FDM/IDM 수치 harness** 신규 구축 (forward/inverse dynamics 정합성)
- [ ] **Controllability**: IDM-decode + Genie 스타일 PSNR-delta
- [ ] **Physical plausibility**: `eval_videophy2.py` (Cosmos-Reason critic)
- [ ] **Downstream utility (북극성 지표)**: DreamGen Bench —
      dream 데이터 → GR00T → success_rate delta. RFM eval과 WFM eval을 잇는 고리.

**완료 기준**: 합성/placeholder 비디오 쌍에 대해 위 메트릭이 수치로 산출 + 리포트.

---

## Phase 4 — 통합 harness

- [ ] 공통 eval config / 데이터 스펙(`data_spec/go2_airbot`) / 리포트 포맷 통일
- [ ] DreamGen 루프 배선: WFM(dream 생성) → GR00T(학습/평가) → SR — RFM·WFM eval 연결
- [ ] CI/재현 스크립트

---

## 의존성 / 병렬화 요약

```
Phase 0 (docker+sanity) ─┬─> Phase 1 (에셋) ──> Phase 2 (RFM eval)
                         │
Phase 3 (WFM eval) ──────┴─(독립, 지금 시작 가능)──> Phase 4 (통합)
```

- Phase 0/1/2는 Isaac Sim(A5000 렌더) 라인.
- Phase 3(WFM)은 독립 라인 — docker 빌드 기다리는 동안 착수 가능.
- Phase 4에서 DreamGen으로 두 라인이 만남.

## 열린 결정 사항

1. **GR00T 추론 서버 위치** — A100 노드 vs 로컬 A5000 (ZMQ split). VRAM 여유 고려.
2. **Locomotion policy** — Isaac Lab 기본 Go2 velocity 사전학습 재사용 vs 자체 학습.
3. **Success 기준 상세** — loco-manip task의 성공 판정 정의.
4. **초기 real 데이터** — open-loop MSE용. 데이터 수집이 eval 시스템 이후라 초기엔 합성/placeholder.
```
