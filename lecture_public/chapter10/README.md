# 10장 실습: 생활SOC 접근성·민원 예측·규제경계 RDD·배달 권역 최적화

이 실습은 `docs/ch10.md`와 `lecture/chapter10.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `practice/chapter10/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `practice/README.md`의 설치 지침을 먼저 따르세요. **신규 의존성은 없습니다** — 최적화는 `scipy.optimize.milp`(HiGHS)로 풀고, 좌표 변환은 이미 설치된 `pyproj`를 씁니다.
- `data/` 폴더의 파일은 저장소에 포함되지 않습니다.
  - 10-1~10-3(시뮬레이션): **`10-0-simdata-prep.py`를 먼저 실행**해야 나머지가 돌아갑니다.
  - 10-4(실데이터): **`10-0b-delivery-data-prep.py`를 먼저 실행**해 스냅샷을 만들어야 합니다. 이 준비 스크립트는 아래 원자료 두 개를 필요로 합니다.

원자료 내려받기 (10-4 실습에만 필요)

두 포털 모두 자바스크립트로 다운로드를 처리해 자동 내려받기가 막혀 있으므로 한 번은 직접 받아야 합니다. 받은 CSV는 저장소 안 아무 장의 `data/raw/` 아래에 두면 준비 스크립트가 찾습니다(용량이 크므로 `.gitignore`가 제외합니다).

1. **소상공인시장진흥공단 상가(상권)정보** — https://www.data.go.kr/data/15083033/fileData.do (로그인·인증키 불필요, 서울 분할 파일)
2. **서울 생활인구(행정동 단위)** — https://data.seoul.go.kr/dataList/OA-14991/S/1/datasetView.do (월별 파일 한 개. 집계구 단위 OA-14979가 아니라 **행정동 단위**를 받습니다)

API 인증키는 쓰지 않습니다. 키가 필요한 경로로 확장할 경우 키를 코드에 넣지 말고 `.env`에 두세요(저장소의 `.gitignore`가 `.env`를 제외합니다).

실습 파일 (실행 순서대로)

- `practice/chapter10/code/10-0-simdata-prep.py` — 격자 인구·생활SOC·민원 데이터 생성(시뮬레이션)
- `practice/chapter10/code/10-1-living-soc-accessibility.py` — 접근성·형평성 진단과 우선 공급 후보지(AI 없이 결정론적 계산)
- `practice/chapter10/code/10-2-civil-complaint-forecast.py` — 다음 주 민원 시공간 예측
- `practice/chapter10/code/10-3-rdd-regulation-boundary.py` — 규제경계 회귀불연속(RDD)
- `practice/chapter10/code/10-0b-delivery-data-prep.py` — **실데이터 스냅샷 생성**(원자료 필요, 1회)
- `practice/chapter10/code/10-4-delivery-zone-optimization.py` — 배달 권역 최적화와 원가 컷라인(스냅샷만 읽음)

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python practice/chapter10/code/10-0-simdata-prep.py
python practice/chapter10/code/10-1-living-soc-accessibility.py
python practice/chapter10/code/10-2-civil-complaint-forecast.py
python practice/chapter10/code/10-3-rdd-regulation-boundary.py
python practice/chapter10/code/10-0b-delivery-data-prep.py        # 원자료 있을 때 1회
python practice/chapter10/code/10-4-delivery-zone-optimization.py
```

스냅샷(`data/delivery_grid.parquet`)이 이미 만들어져 있으면 취득 단계를 건너뛰고 `10-4`만 실행해도 같은 수치가 재현됩니다.

예상 결과(검증 포인트)

- 10-1 단위: 격자 `300개`(셀 500m), 총 인구 `1,870,844명`
- 10-1 종합 접근성: 인구가중 평균 `21.4분`, Gini `0.246`, 15분 초과 격자 `258개`(인구의 `69.5%`)
- 10-2 데이터: 격자 `256개` × `40주`, 총 민원 `126,117건`
- 10-2 예측: 시간 시차만 R² `0.753` → 공간 시차 추가 R² `0.813` (개선 `+0.059`)
- 10-3 설계: **참 규제효과 = `−60.0` 만원/㎡**, 표본 `2,000`
- 10-3 추정: 지역선형(h=1.0km) `−54.92`, 대역폭 0.5~1.0km에서 `−55 ~ −58`로 안정. 전역 다항식은 차수에 따라 `−63.2 ~ −53.6`로 요동
- 10-0b 자료: 서울 종로구·서대문구·은평구, 활동 격자 `245개`(셀 500m), 행정동 `47개`, 생활 인프라 POI `3,288개`. 평시 생활인구 `1,121,348명` → 저녁 `1,030,742명`
- 10-4 최적화: p-median 평균 도달 `2.96분`(5분 커버 `86.6%`) / 최대커버 `3.13분`(`88.8%`, 거점 3곳이 다름) / 휴리스틱은 목적함수 `+0.738%`
- 10-4 컷라인: 손익분기 편도 주행시간 `3.8~7.7분`(묶음계수 1.0~2.0), 컷라인 밖 격자 `83개`(`33.9%`)인데 인구는 `6.6%`
- 10-4 공급: 평시 `6.6%` → 피크(용량제약+할증) `54.4%`. 라이더 ±40%에서 `34.4% ~ 56.6%`
- 10-4 중첩: 컷라인 밖 × 인프라 열위 Jaccard `0.408`(인프라 열위 격자의 `67.7%`가 컷라인 밖)
- 10-4 대조군: 균등 배분 `9.5%` / 수요 무작위화 200회 Jaccard 평균 `0.277` [`0.228`, `0.333`], 경험적 p `0.000`
- 10-4 퇴화 검사: 단순 규칙(음식점 상위·수요 상위)은 목적함수 `+57~62%`, 컷라인 밖 인구 `25%` 안팎
- 10-4 민감도: 컷라인 밖 인구 `2.0% ~ 30.1%`(기준 `6.6%`). 가장 크게 흔드는 것은 임률이 아니라 묶음계수

검증 팁

- 10-1에는 **일부러 AI를 쓰지 않습니다.** 접근성·형평성은 거리 계산으로 완결되는 문제라, 학습 모델을 붙이면 설명력과 책임성만 잃습니다. AI가 정당해지는 지점은 10-2의 미래 예측입니다.
- 10-3에서 확인할 것은 추정값 하나가 아니라 **대역폭을 바꿔도 값이 버티는가**입니다. 좁은 대역폭에서 안정적이고 넓히면 흔들리는 패턴이 정상입니다.
- 10-1~10-3은 **시뮬레이션**입니다. 10-3은 참 효과를 심어 두어 추정 오차를 채점할 수 있습니다.
- 10-4는 **실데이터**지만 원가 파라미터(수수료·임률·고정시간·변동비·묶음계수·라이더 수)는 공개 통계가 없어 **전부 가정값**입니다. 이 실습의 숫자는 파라미터를 바꾸면 함께 움직입니다 — 절대 금액이 아니라 **민감도 범위와 순위**를 읽으세요. 어느 설정에서든 먼저 잘리는 곳은 멀고 성긴 격자입니다.
- 10-4에서 **스냅샷 날짜가 다르면 수치가 달라지는 것이 정상입니다.** 재현의 기준은 동결된 스냅샷(`data/delivery_snapshot_meta.json`의 취득일·해시)이며, 확인할 것은 숫자의 일치가 아니라 구조의 재현입니다 — 목적함수를 바꾸면 배치가 바뀌는가, 컷라인 밖이 멀고 성긴 곳인가, 대조군에서 중첩이 내려가는가.
- 10-4의 주행시간은 실제 도로망이 아니라 **직선거리 × 우회계수 1.35**입니다(맨해튼 거리로 한 번 더 검산). 하천·급경사가 만드는 국소 우회는 담지 못하므로, 특정 동네를 서비스에서 빼는 판단의 근거로 쓰지 마세요.

결과 파일

- `practice/chapter10/results/10-1~10-4 *.log` — 실행 로그(+ `.evidence.json` 증거)
- `living_soc_priority.csv`, `complaint_response_priority.csv`, `complaint_response_priority_uncertainty.csv` — 우선순위표
- `10-3-rdd-discontinuity.png`, `10-3-rdd-summary.txt` — RDD 산출물
- `delivery_zone_cells.csv`(격자별 원가·기여이익·컷라인 판정), `delivery_zone_summary.csv`(시나리오 요약), `10-4-delivery-zone.png` — 배달 권역 산출물

연관 자료

- 교재: `docs/ch10.md` — 도시·생활권·민원 행정과 GeoAI
- 강의: `lecture/chapter10.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `practice/chapter10/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
- 10-0b가 "원자료를 찾지 못했다"로 멈추면, 위 두 포털에서 받은 CSV가 `practice/*/data/raw/` 아래에 있는지 확인하세요(하위 폴더도 탐색합니다).
