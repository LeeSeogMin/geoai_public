# 7장 실습: 자율 GIS 질의 검증, 시공간 예측 불확실성, 그리고 발주 결정

이 실습은 `docs/ch07.md`와 `lecture/chapter7.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `lecture_practice/chapter7/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `lecture_practice/README.md`의 설치 지침을 먼저 따르세요.
- `data/` 폴더의 파일은 저장소에 포함되지 않습니다. **`7-0-simdata-prep.py`와 `7-0b-demand-simdata.py`를 먼저 실행**해야 나머지가 돌아갑니다. 두 준비 스크립트는 서로 다른 난수열을 쓰므로 실행 순서를 바꿔도 결과가 흔들리지 않습니다.
- `7-2`는 LSTM을 학습합니다. CPU로 1~2분이면 끝납니다.
- `7-3`은 품목 2개 × 데이터셋 2종으로 LSTM을 네 번 학습합니다. CPU로 40초쯤 걸립니다. 가속기(CUDA·MPS)를 쓰면 커널 차이로 소수점 아래가 달라지므로, 본문 수치와 맞추기 위해 **CPU로 고정**해 두었습니다(`PIN_CPU = True`).
- **외부 LLM API를 호출하지 않습니다.** `7-1`은 자연어 질의를 규칙으로 파싱하고 검증 관문이 어떻게 작동하는지를 보여 주는 시연이며, API 키가 필요 없습니다.

실습 파일 (실행 순서대로)

- `lecture_practice/chapter7/code/7-0-simdata-prep.py` — 레이어 카탈로그·교통량 시계열 생성 (7-1·7-2용)
- `lecture_practice/chapter7/code/7-0b-demand-simdata.py` — 점포 6곳 × 두 품목 × 1,460일 수요 패널 생성 (7-3용). 이분산 패널과 대조군 C1(등분산) 패널을 함께 저장
- `lecture_practice/chapter7/code/7-1-autonomous-gis-query.py` — 자연어 공간 질의와 검증 관문의 차단 사례
- `lecture_practice/chapter7/code/7-2-spatiotemporal-uncertainty.py` — LSTM + MC Dropout 시공간 예측과 예측구간
- `lecture_practice/chapter7/code/7-3-demand-newsvendor.py` — LSTM + 3분할 정규화 conformal → 임계비 발주 결정과 세 정책의 실현 손익

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python lecture_practice/chapter7/code/7-0-simdata-prep.py
python lecture_practice/chapter7/code/7-0b-demand-simdata.py
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
- 코드를 고친 뒤에는 위 7-1·7-2 검증 포인트(73,713명 / RMSE 12.82 / 7.23·11.69 / 91.5% / 45.18)가 그대로인지 먼저 확인하세요. 이 수치들은 다른 장에서도 참조합니다.

검증 팁

- 7-1에서 확인할 것은 성공 건수가 아니라 **차단된 3건이 왜 차단됐는가**입니다. 검증 관문이 없었다면 세 질의 모두 그럴듯한 숫자를 뱉었을 것입니다.
- 7-2는 신경망 학습이라 RMSE와 구간 폭이 실행마다 소폭 흔들립니다. 확인해야 할 방향은 **포함률이 목표 90% 근처**, **aleatoric이 epistemic보다 큼** 두 가지입니다.
- 7-3도 마찬가지로 금액의 절대값보다 **방향**이 중요합니다. 두 품목의 임계비가 0.5의 반대편에 있고, 그래서 발주가 점추정의 위아래로 갈리는 것이 요점입니다.
- 7-3의 원가·판가는 **설명을 위해 정한 가정값**입니다. 어느 업종의 실제 원가율도 나타내지 않습니다.
- 세 실습 모두 **시뮬레이션 데이터**를 씁니다. 교통량과 수요 패널은 합성 시계열이고, 레이어 카탈로그도 실습용으로 만든 것입니다.

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
- 강의: `lecture/chapter7.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `lecture_practice/chapter7/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
