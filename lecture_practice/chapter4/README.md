# 4장 실습: 공간 피처·공간 교차검증·앙상블·군집·취약성 예측·상권 기대 공급량

이 실습은 `docs/ch04.md`와 `lecture/chapter4.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `lecture_practice/chapter4/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `lecture_practice/README.md`의 설치 지침을 먼저 따르세요.
- `data/` 폴더의 위성영상·벡터 파일은 저장소에 포함되지 않습니다(`.gitignore`). **아래 0단계 스크립트를 먼저 실행해 데이터를 만들어야** 나머지 실습이 돌아갑니다.
- `4-0-data-download.py`는 Planetary Computer에서 실제 Sentinel-2 장면을 내려받으므로 인터넷 연결과 수백 MB의 여유 공간이 필요합니다.
- `4-0-market-data-prep.py`는 **국내 공개 데이터 원본 CSV 두 개**(소상공인 상가정보 서울분, 서울 생활인구 행정동)를 필요로 합니다. 두 포털 모두 자바스크립트로 다운로드를 처리해 자동 내려받기가 막히므로 한 번은 직접 받아야 합니다. 절차는 `data/raw/README.md` 참조. **인증키는 필요 없습니다.** 14장도 같은 원자료를 쓰므로, 이미 받아 두었다면 그대로 재사용합니다.

실습 파일 (실행 순서대로)

- `lecture_practice/chapter4/code/4-0-simdata-prep.py` — 4-1·4-2·4-5용 벡터·격자 데이터 생성
- `lecture_practice/chapter4/code/4-0-data-download.py` — 4-3·4-4용 실제 Sentinel-2 L2A + ESA WorldCover 내려받기
- `lecture_practice/chapter4/code/4-0-raster-inspection.py` — 내려받은 위성영상 메타데이터·밴드 확인
- `lecture_practice/chapter4/code/4-1-spatial-features.py` — 건물 필지에서 형태·거리·위상 피처 추출
- `lecture_practice/chapter4/code/4-2-spatial-cv.py` — 공간 자기상관(Moran's I)과 Random/Block/Cluster CV 비교
- `lecture_practice/chapter4/code/4-3-ensemble-comparison.py` — RF·XGBoost·LightGBM 비교(실제 위성 픽셀)
- `lecture_practice/chapter4/code/4-4-spatial-clustering.py` — K-means·DBSCAN 비지도 군집
- `lecture_practice/chapter4/code/4-5-grid-vulnerability-shap.py` — 격자 취약성 예측 + 공간 CV + SHAP (4.6 분석 1)
- `lecture_practice/chapter4/code/4-0-market-data-prep.py` — 상가정보·생활인구를 서울 행정동 단위로 집계
- `lecture_practice/chapter4/code/4-6-store-location-supply.py` — 행정동 카페 기대 공급량 + 자치구 단위 CV + SHAP + 라벨 셔플 대조군 (4.6 분석 2)

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python lecture_practice/chapter4/code/4-0-simdata-prep.py
python lecture_practice/chapter4/code/4-0-data-download.py
python lecture_practice/chapter4/code/4-0-raster-inspection.py
python lecture_practice/chapter4/code/4-1-spatial-features.py
python lecture_practice/chapter4/code/4-2-spatial-cv.py
python lecture_practice/chapter4/code/4-3-ensemble-comparison.py
python lecture_practice/chapter4/code/4-4-spatial-clustering.py
python lecture_practice/chapter4/code/4-5-grid-vulnerability-shap.py
python lecture_practice/chapter4/code/4-0-market-data-prep.py
python lecture_practice/chapter4/code/4-6-store-location-supply.py
```

예상 결과(검증 포인트)

- 0단계 생성 데이터: 건물 `200개`, 지하철역 `3개`, 공간 포인트 `500개`, 격자 `400개(20×20)`
- 내려받은 장면: `S2B_MSIL2A_20210407T021559_R003_T52SCG` (2021-04-07, 구름 `0.44%`), `12밴드 1314×1338`, `EPSG:32652`, 해상도 `10m`
- 4-1 공간 피처: 건물 200개에서 피처 `7개`, 면적 범위 `167.0 ~ 3393.9 m²`
- 4-2 공간 자기상관: Moran's I `0.8660` (p = `0.0010`)
- 4-2 CV 비교(R²): Random `0.869` / Block `0.798` / Cluster `0.349` — Random이 Block보다 `+0.071` 높음
- 4-2 시드 반복(잡음 바닥, 4시드 × 3축 = 12회): Random−Block `+0.031 ~ +0.123`(12회 전부 양수 — **방향만 재현**), Random−Cluster 낙폭 `0.517 ~ 1.747`(**방향·크기 모두 재현**). 축별 낙폭 폭은 모델 시드 `0.041`, 분할 시드 `0.210`, **생성 시드 `1.227`**로 생성 축이 지배합니다. 생성식 대조 최대 차이 `0.00e+00`
- 4-3 모델 비교(Macro-F1, Random → Block): RF `0.763 → 0.753`, XGBoost `0.772 → 0.760`, LightGBM `0.768 → 0.755`
- 4-3 시드 반복(잡음 바닥, 3모델 × 4시드 × 3방식 = 36회): 과대추정 `+0.005 ~ +0.022` — **36회 전부 양수라 방향은 재현되지만 크기는 재현되지 않습니다.** 축별 범위는 모델 시드만 `+0.006~+0.017`, 블록→폴드 배정까지 `+0.005~+0.022`, 픽셀 표집 시드 `+0.009~+0.016`. 모델 간 과대추정 순위는 반복마다 바뀝니다
- 4-4 군집: K-means(K=5) 실루엣 `0.365`, 실제 토지피복과의 ARI `0.329`·NMI `0.431`, DBSCAN 군집 `2개`·노이즈 `123개(3.1%)`
- 4-5 취약성 예측: Moran's I `0.817`, RF Random `0.639` / Block `0.535` / Cluster `0.373`, SHAP 1위 피처 `elderly (4.003)`, 잔차 Moran's I `0.663`
- 4-5 시드 반복(잡음 바닥, 2모델 × 4시드 × 3축 = 24회): Random−Cluster 낙폭 `−0.511 ~ −0.165` — **24회 전부 음수라 방향과 크기를 함께 말할 수 있습니다.** 축별 RF 범위는 모델 시드 `−0.266~−0.253`, 분할 시드 `−0.266~−0.197`, 생성 시드 `−0.269~−0.165`로 **생성 축이 가장 크게 흔듭니다**(폭 0.104 대 0.013). 생성식 대조(시드 42 재생성 vs 저장 자료) 최대 차이 `0.00e+00`
- 4-0-market 집계(2026년 6월 기준분): 서울 점포 `554,092개` → 소분류 '카페' `22,739개(4.1%)`, 행정동 `427개` 집계 후 생활인구와 `97.2%` 결합 → 최종 `415개 행정동 / 25개 자치구`
- 4-6 공간 시차: 수요·위치만 `0.656` → 이웃 정보 추가 `0.660`(개선 `+0.003`) — **거의 오르지 않는 것이 정상입니다**
- 4-6 검증 설계: 무작위 CV `0.685` vs 자치구 GroupKFold `0.660`(과대추정 `+0.025`). 폴드별 `0.564~0.726`
- 4-6 시드 반복(잡음 바닥): 모델 시드만 교체하면 개선 `+0.003~+0.008` / 과대추정 `+0.019~+0.026`, 폴드 배정 시드까지 교체하면 개선 `−0.001~+0.018` / 과대추정 `−0.006~+0.034` — **두 차이 모두 부호가 뒤집힙니다.** 폴드 간 R² SD는 자치구 CV `0.060`(배정 고정) / `0.189`(배정 교체)
- 4-6 대조군: 결과변수 셔플 20회 R² `−0.141`(SD `0.045`) — 실제 `0.660`과 확연히 갈립니다
- 4-6 SHAP: 낮 생활인구 `18.434` > 주야 인구비 `12.401` > 이웃 동 카페 수 `4.486`
- 4-6 공급 격차: 기대보다 적은 1위 서대문구 북아현동(실제 `16` vs 기대 `176.0`), 많은 1위 마포구 서교동(실제 `535` vs 기대 `238.5`)

검증 팁

- `[Errno 2] No such file or directory ... data/...`가 나오면 0단계 스크립트를 건너뛴 것입니다. `4-0-simdata-prep.py`부터 다시 실행하세요.
- 4-2·4-5의 핵심은 절대 점수가 아니라 **Random CV가 Block/Cluster CV보다 높게 나온다**는 방향입니다. 숫자가 소수점 아래에서 조금 달라도 이 부등호가 유지되면 정상입니다.
- 4-2는 8절 시드 반복 12회 때문에 **약 55초**가 걸립니다(1~7절만은 20초 남짓). 8절 첫 줄의 **생성식 대조가 `0.00e+00`이 아니면** 4-0-simdata-prep.py의 `prepare_spatial_points()`가 바뀐 것이므로, 4-2의 `regenerate_points()`를 같은 식(난수 소비 순서 x → y → 잡음)으로 맞춘 뒤 다시 실행하세요.
- 4-2에서 **`+0.071`은 재현 확인 항목이 아닙니다.** 8절이 보여 주듯 Random−Block은 `+0.031~+0.123`으로 흩어지므로 양수라는 방향만 확인합니다. 반면 Random−Cluster 낙폭은 12회 모두 `0.517` 이상이라 크기까지 확인해도 됩니다.
- 4-5는 6절 시드 반복 24회 때문에 **약 1분 반**이 걸립니다(1~5절만은 20초 남짓). 6절 첫 줄의 **생성식 대조가 `0.00e+00`이 아니면** 4-0-simdata-prep.py의 `prepare_grid()`가 바뀐 것이므로, 4-5의 `regenerate_grid()`를 같은 식으로 맞춘 뒤 다시 실행하세요.
- 4-3은 9절의 시드 반복 36회 때문에 **약 5분 반**이 걸립니다(1~8절만은 30초 남짓). LightGBM의 `X does not have valid feature names` 경고는 예측 단계의 알림이며 결과에 영향을 주지 않습니다.
- 4-3에서 확인할 것은 **과대추정이 양수라는 방향**이지 그 크기가 아닙니다. 9절이 보여 주듯 크기는 `+0.005~+0.022`로 벌어지므로, `+0.010`이 재현되지 않아도 정상입니다. 같은 이유로 **세 모델 중 어느 쪽이 가장 크게 부풀려지는지는 재현 확인 항목이 아닙니다.**
- 4-3의 경작지 F1이 낮은 것은 모델 결함이 아니라 **4월 초 장면의 나지 상태와 WorldCover 연간 라벨이 어긋나기 때문**입니다. 로그의 "이 실습의 범위와 한계"를 함께 읽으세요.
- 4-6은 **실데이터라 절대 수치가 시점에 따라 달라집니다.** 공개 데이터가 갱신되면 점포 수와 순위가 바뀝니다. 확인할 것은 숫자 자체가 아니라 방향입니다 — ① 공간 시차 개선이 폴드 간 SD보다 훨씬 작을 것 ② 셔플 대조군이 음수일 것 ③ SHAP 1위가 낮 생활인구일 것. **"자치구 CV가 무작위 CV보다 낮을 것"은 확인 항목이 아닙니다** — [6]절이 보여 주듯 이 부등호는 폴드 배정 시드에 따라 뒤집히므로, 재현 여부를 이 부호로 판정하면 안 됩니다. 실행한 원본 파일명과 처리 일자는 `data/SOURCES_market.txt`에 기록됩니다.
- 국내 공공 CSV는 `cp949`와 `utf-8`이 섞여 있습니다. 4-0-market은 둘을 차례로 시도하므로 미리 변환할 필요가 없습니다. 열 이름이 갱신으로 바뀌면 **조용히 잘못된 열을 쓰지 않고 실제 열 목록을 보여 주며 멈춥니다.**

결과 파일

- `lecture_practice/chapter4/results/*.log` — 스크립트별 실행 로그
- `lecture_practice/chapter4/results/ch4_cv_comparison.csv` — CV 기법 비교 표
- `lecture_practice/chapter4/results/ch4_vulnerability_priority.csv` — 격자 우선순위 표
- `lecture_practice/chapter4/results/ch4_dong_supply_gap.csv` — 행정동별 기대 공급량과 공급 격차
- `lecture_practice/chapter4/results/ch4_supply_cv_comparison.csv` — 검증 설계별 성능 비교
- `lecture_practice/chapter4/results/ch4_supply_gap_map.png` — 공급 격차 지도(그림 4.3)

연관 자료

- 교재: `docs/ch04.md` — 공간단위 머신러닝과 지역 분석
- 강의: `lecture/chapter4.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `lecture_practice/chapter4/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
