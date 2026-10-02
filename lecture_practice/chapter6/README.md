# 6장 실습: 변화 탐지 결과의 정책 집계와 개발 가능 부지 탐색

이 실습은 `docs/ch06.md`와 `lecture/ch06.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `lecture_practice/chapter6/results/`에 저장되어 있습니다.

두 개의 분석이 들어 있습니다.

- **분석 1 (공공)** — 변화 마스크를 평가·필터링해 행정구역별 우선순위표로 옮긴다.
- **분석 2 (비즈니스)** — 유휴 부지 마스크에서 형상 피처를 뽑아 개발 실사 후보 목록을 자른다.

기본 실습은 지금처럼 CPU에서 실행합니다. GPU가 있는 수강생은 선택 실습 `6-0a`에서 작은 U-Net을 직접 학습해 변화 마스크를 만들 수 있습니다. 두 경로는 같은 입력 자료를 쓰지만, 선택 실습은 기존 마스크나 기본 실습의 결과를 덮어쓰지 않습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `lecture_practice/README.md`의 설치 지침을 먼저 따르세요.
- 분석 1의 `data/change_masks.npz`와 분석 2의 `data/urban_block.npz`는 저장소에 포함되어 있습니다. 별도의 데이터 준비 스크립트는 실행하지 않습니다.
- U-Net 선택 실습에는 PyTorch가 필요합니다. `python lecture_practice/setup_torch.py`로 컴퓨터에 맞는 CUDA·MPS·CPU 빌드를 설치하고, `python lecture_practice/check_env.py`의 `딥러닝 장치` 항목을 확인하세요.

실습 파일 (실행 순서대로)

- `lecture_practice/chapter6/code/6-0a-unet-change-detection.py` — 선택 실습. 두 시점 NDVI를 입력으로 받는 소형 U-Net 학습, 공간 검증 영역의 IoU·Dice 계산
- `lecture_practice/chapter6/code/6-1-change-detection-policy.py` — IoU·정밀도·재현율 평가, 거짓변화 필터, 행정구역 집계
- `lecture_practice/chapter6/code/6-2-site-sourcing.py` — 객체화·형상 피처·3단 요건 필터·면적 편향 보정·대조군 3종·임계값 비용 곡선·실사 순서

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python lecture_practice/chapter6/code/6-1-change-detection-policy.py
python lecture_practice/chapter6/code/6-2-site-sourcing.py
```

GPU U-Net 선택 실습 (NVIDIA GPU)

```bash
python lecture_practice/setup_torch.py
python lecture_practice/check_env.py
python lecture_practice/chapter6/code/6-0a-unet-change-detection.py --device cuda
```

Apple Silicon Mac은 마지막 명령의 `cuda`를 `mps`로 바꿉니다. 장치를 자동 선택하려면 `--device auto`를 쓰며, GPU가 없는 컴퓨터에서도 `--device cpu`로 구조를 확인할 수 있습니다. `cuda`나 `mps`를 명시했는데 해당 장치를 사용할 수 없으면 코드는 CPU로 전환하지 않고 중단합니다.

U-Net 성공 여부 확인

- 첫 출력의 `장치`가 NVIDIA에서는 `cuda`, Apple Silicon에서는 `mps`인지 확인합니다.
- 학습 중 `train loss`가 출력되고 마지막에 검증 영역의 `IoU`와 `Dice`가 나오면 학습과 추론을 모두 실행한 것입니다.
- `results/6-0a-unet-cuda.png` 또는 `results/6-0a-unet-mps.png`에서 두 시점 NDVI, 정답, U-Net 확률 지도를 비교합니다.
- 이 값은 한 개의 합성 장면을 공간 분할해 얻으므로 실제 위성영상 성능으로 해석하지 않습니다.

예상 결과(검증 포인트) — 분석 1

- 래스터 `128×128`(픽셀 10m = 0.01ha), 진짜 변화 `2,001px`, 예측 변화 `3,022px`
- 필터 전: IoU `0.525` | 정밀도 `0.572` | 재현율 `0.865` | Dice `0.689`
- 필터 후(NDVI 차분 > 0.25 + 최소면적 ≥ 1ha): IoU `0.865` | 정밀도 `1.000` | 재현율 `0.865` | Dice `0.927`
- 행정구역 우선순위 1위: `구역 9`, 변화 면적 `2.93ha`
- 탐지된 개발 변화 총 `17.3ha`

> 분석 1과 분석 2의 입력 자료는 저장소에 포함되어 있습니다. 값이 달라졌다면 `data/change_masks.npz` 또는 `data/urban_block.npz` 파일을 먼저 확인하세요.

예상 결과(검증 포인트) — 분석 2

- 래스터 `512×512`(픽셀 1m = 1㎡, 26.2ha), 유휴 부지 정답 `32필지`, 그중 요건을 모두 채우는 우량 부지 `10필지`
- 검출된 필지의 면적 상대오차 중앙값 `−0.125` — 경계 침식이 면적을 한쪽으로 깎습니다
- 요건 필터 깔때기: 객체 `40` → 면적 `20` → 접도 `9` → 최소폭 `8`
- 보정 전 후보 `8건`, 우량 부지 오탈락 `5건` → 보정 후 후보 `11건`, 오탈락 `2건`
- 대조군 방향: **C1(침식 제거)에서 오탈락이 0으로 사라지고, C2(오탐 제거)에서 헛후보가 0으로 사라진다.** C3(과분할 제거)은 검출 필지를 30 → 32로 회복시키지만 후보 수와 오탈락은 바꾸지 않는다
- 면적 임계의 컷라인: 놓침 비용이 실사 비용의 1배면 최적 임계 `1,450㎡`, 3배 이상이면 `725㎡`

검증 팁

- 분석 1: 정밀도가 오르는 대신 재현율은 그대로이거나 조금 떨어집니다. **필터는 공짜가 아니라 맞바꿈**이라는 점이 핵심입니다.
- 분석 2: 접도 판정의 허용오차를 0으로 바꾸면 진짜 필지가 전부 탈락하고 오탐만 남습니다. 기하 조건의 허용오차는 마스크의 위치 오차보다 커야 합니다.
- 두 실습 모두 **시뮬레이션**입니다. 정답을 일부러 심어 두었기 때문에 채점이 가능합니다. 실제 과제에는 이런 정답지가 없다는 점을 전제로 결과를 읽으세요.
- IoU는 반씩만 겹쳐도 0.33 수준으로 떨어지는 박한 지표입니다. 값이 낮게 느껴져도 계산이 틀린 것이 아닙니다.
- 분석 2의 면적 편향 12.5%는 **우리가 심은 설계값**의 결과이지 실제 세그멘테이션 모델의 성능이 아닙니다.

결과 파일

- `lecture_practice/chapter6/results/6-1-change-detection-policy.log` — 분석 1 실행 로그
- `lecture_practice/chapter6/results/change_admin_priority.csv` — 행정구역별 변화 면적·우선순위
- `lecture_practice/chapter6/results/6-2-site-sourcing.log` — 분석 2 실행 로그
- `lecture_practice/chapter6/results/ch6_site_candidates.csv` — 후보 부지 표(추정·보정 면적, 형상 피처, 방문 순위)
- `lecture_practice/chapter6/results/ch6_missed_sites.csv` — 오탈락한 우량 부지와 보정 후 잔존 여부
- `lecture_practice/chapter6/results/ch6_threshold_cost.csv` — 면적 임계별 후보 수·오탈락 수·비용
- `lecture_practice/chapter6/results/6-2-site-sourcing-map.png` — 정답과 후보 지도

연관 자료

- 교재: `docs/ch06.md` — 세그멘테이션·파운데이션 모델·SAM
- 강의: `lecture/ch06.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `lecture_practice/chapter6/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
