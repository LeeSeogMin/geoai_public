# 7장 실습: 자율 GIS 질의 검증, 시공간 예측 불확실성, 그리고 발주 결정

이 실습은 `docs/ch07.md`와 `lecture/ch07.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `lecture_practice/chapter7/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `lecture_practice/README.md`의 설치 지침을 먼저 따르세요.
- 이 장이 쓰는 데이터 여섯 개(`pop_grid.parquet`, `libraries.parquet`, `schools.parquet`, `traffic_volume.npy`, `store_demand.parquet`, `store_demand_homoskedastic.parquet`)는 **모두 저장소의 `data/` 폴더에 포함되어 있습니다.** 저장소를 받으면 함께 내려오므로 따로 만들 것이 없습니다.
- 7-1·7-2는 교통량·GIS 레이어 데이터를 씁니다. 7-3은 **다른 목적의 실습**으로, 점포 6곳의 도시락·우산 판매 수요를 예측해 발주량을 결정합니다. 두 실습은 서로 다른 데이터를 쓰며 서로 의존하지 않습니다.
- `7-2`는 LSTM을 학습합니다. CPU로 1~2분이면 끝납니다.
- `7-3`은 품목 2개 × 데이터셋 2종으로 LSTM을 네 번 학습합니다. CPU로 40초쯤 걸립니다. 가속기(CUDA·MPS)를 쓰면 커널 차이로 소수점 아래가 달라지므로, 본문 수치와 맞추기 위해 **CPU로 고정**해 두었습니다(`PIN_CPU = True`).
- **외부 LLM API를 호출하지 않습니다.** `7-1`은 자연어 질의를 규칙으로 파싱하고 검증 관문이 어떻게 작동하는지를 보여 주는 시연이며, API 키가 필요 없습니다.

실습 파일 (실행 순서대로)

- `lecture_practice/chapter7/code/7-1-autonomous-gis-query.py` — 자연어 공간 질의와 검증 관문의 차단 사례 (`data/`의 레이어 카탈로그 사용)
- `lecture_practice/chapter7/code/7-2-spatiotemporal-uncertainty.py` — LSTM + MC Dropout 시공간 예측과 예측구간 (`data/`의 교통량 시계열 사용). **목적: 교통량 예측**
- `lecture_practice/chapter7/code/7-3-demand-newsvendor.py` — LSTM + 3분할 정규화 conformal → 임계비 발주 결정과 세 정책의 실현 손익 (`data/`의 점포 수요 패널 사용). **목적: 점포 수요 예측 → 발주량 결정** (교통량과 무관)
- `lecture_practice/chapter7/code/7-0-simdata-prep.py` — `data/` 폴더의 7-1·7-2용 데이터를 다시 만드는 스크립트(교수자용). 학생은 실행할 필요가 없습니다
- `lecture_practice/chapter7/code/7-0b-demand-simdata.py` — `data/` 폴더의 7-3용 수요 패널(이분산 본 데이터·등분산 대조군 C1)을 다시 만드는 스크립트(교수자용). 학생은 실행할 필요가 없습니다

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python lecture_practice/chapter7/code/7-1-autonomous-gis-query.py
python lecture_practice/chapter7/code/7-2-spatiotemporal-uncertainty.py
python lecture_practice/chapter7/code/7-3-demand-newsvendor.py
```

예상 결과(검증 포인트)

- 7-1 질의 `4건` 중 성공 `1건`, 검증 단계가 차단 `3건`
  - 성공: 도서관 1km 이내 인구 약 `73,713명`
  - 차단 사유: 존재하지 않는 레이어(지하철역) / 좌표계 불일치(EPSG:4326 vs EPSG:5179) / 모호한 질의
- 7-2 점추정 RMSE = `12.82 대/시간`
- 7-2 불확실성: epistemic `7.23` + aleatoric `11.69`(epistemic 비중 `28%`), 90% 구간 포함률 `91.5%`, 평균 구간 폭 `45.18`
- 7-3 3분할 conformal: 시험 집합 90% 구간 포함률이 도시락 `89.3%`, 우산 `89.6%` — 목표 `90%` 부근
- 7-3 임계비: 도시락 `0.375`(평균 발주 `81.8`, 실현 서비스 수준 `34.9%`), 우산 `0.909`(평균 발주 `25.1`, `92.8%`)
  - 방향으로 확인할 것: **도시락의 발주 분위는 0.5보다 낮고 우산은 0.9보다 높다**
- 7-3 세 정책 총손실(도시락): P2 구간 상한 `119,353,200`원 > P1 점추정 `46,110,000`원 > P3 newsvendor `45,666,000`원
  - 방향으로 확인할 것: **도시락에서 P2(구간 상한)가 P3보다 크게 비싸다**(약 2.6배)
- 7-3 대조군 C1(등분산): 구간 폭 변동계수가 `0.364 → 0.032`(도시락), `0.473 → 0.062`(우산)로 무너진다
  - 방향으로 확인할 것: **등분산 데이터에서는 정규화와 비정규화의 차이가 사라진다**
- 7-3 대조군 C2(비정규화): 주변 포함률은 정규화와 비슷하지만 수요 3분위별 편차가 `14.9%p`·`16.4%p`(정규화는 `5.7%p`·`1.5%p`)

7-1·7-2의 기존 결과가 그대로인지 확인하기

- `7-0b`와 `7-3`은 나중에 추가된 코드입니다. `7-0b`는 `7-0`과 **완전히 분리된 난수열**(별도 파일·별도 시드)을 쓰므로 7-1·7-2의 결과에 영향을 주지 않습니다.
- `7-3`의 입력(점포 수요 패널)과 `7-1`·`7-2`의 입력(레이어 카탈로그·교통량)은 서로 다른 파일이라 한쪽을 다시 만들어도 다른 쪽이 흔들리지 않습니다.
- 코드를 고친 뒤에는 위 7-1·7-2 검증 포인트(73,713명 / RMSE 12.82 / 7.23·11.69 / 91.5% / 45.18)가 그대로인지 먼저 확인하세요. 이 수치들은 다른 장에서도 참조합니다.

검증 팁

- 7-1에서 확인할 것은 성공 건수가 아니라 **차단된 3건이 왜 차단됐는가**입니다. 검증 관문이 없었다면 세 질의 모두 그럴듯한 숫자를 뱉었을 것입니다.
- 7-2는 신경망 학습이라 RMSE와 구간 폭이 실행마다 소폭 흔들립니다. 확인해야 할 방향은 **포함률이 목표 90% 근처**, **aleatoric이 epistemic보다 큼** 두 가지입니다.
- 7-3도 마찬가지로 금액의 절대값보다 **방향**이 중요합니다. 두 품목의 임계비가 0.5의 반대편에 있고, 그래서 발주가 점추정의 위아래로 갈리는 것이 요점입니다.
- 7-3의 원가·판가는 **설명을 위해 정한 가정값**입니다. 어느 업종의 실제 원가율도 나타내지 않습니다.
- 실습 데이터는 교육용으로 준비된 것입니다. 레이어 카탈로그·교통량 시계열·수요 패널 모두 저장소의 `data/` 폴더로 제공됩니다. 어느 지역·업종의 실제 통계도 나타내지 않습니다.

결과 파일

- `lecture_practice/chapter7/results/7-1-autonomous-gis-query.log`, `7-2-spatiotemporal-uncertainty.log`, `7-3-demand-newsvendor.log` — 실행 로그
- `lecture_practice/chapter7/results/autonomous_gis_query_log.csv` — 질의 처리 기록
- `lecture_practice/chapter7/results/traffic_uncertainty.csv` — 예측값과 신뢰구간
- `lecture_practice/chapter7/results/ch7_conformal_coverage.csv` — 데이터·품목·conformal 방식별 포함률과 조건부 진단
- `lecture_practice/chapter7/results/ch7_newsvendor_policy.csv` — 정책 × 품목 실현 손익
- `lecture_practice/chapter7/results/ch7_newsvendor_sensitivity.csv` — 가정 임계비별 발주량과 총손실
- `lecture_practice/chapter7/results/7-3-demand-newsvendor.png` — 세 정책의 발주선과 실현 수요

연관 자료

- 교재: `docs/ch07.md` — LLM·자율 GIS·시공간 예측
- 강의: `lecture/ch07.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `lecture_practice/chapter7/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
