# 12장 실습: 복지 사각지대·감염병 확산·DML·공간 스캔 통계·동물병원 입지

이 실습은 `docs/ch12.md`와 `lecture/chapter12.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `practice/chapter12/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `practice/README.md`의 설치 지침을 먼저 따르세요.
- `data/` 폴더의 파일은 저장소에 포함되지 않습니다. **`12-0-simdata-prep.py`를 먼저 실행**해야 12-1~12-4가 돌아갑니다.
- **12-5만 국내 공개 실데이터를 씁니다.** 원자료를 한 번 직접 내려받아야 하며, 절차는 `data/raw/README.md`에 있습니다. 14장 원자료를 이미 받아 두었다면 다시 받지 않아도 됩니다(준비 스크립트가 자동으로 찾습니다).

실습 파일 (실행 순서대로)

- `code/12-0-simdata-prep.py` — 읍면동·격자 단위 복지·보건 합성 데이터 생성 (12-1~12-4용)
- `code/12-1-welfare-blindspot-priority.py` — 사각지대 위험 예측과 방문상담 우선순위
- `code/12-2-infectious-spatiotemporal-risk.py` — 감염병 시공간 확산 예측
- `code/12-3-dml-highdim-confounding.py` — 고차원·비선형 교란과 이중기계학습(DML)
- `code/12-4-satscan-cluster.py` — 공간 스캔 통계로 군집 탐지
- `code/12-0b-vetcare-data-prep.py` — **실데이터** 준비: 서울 동물병원·행정동 심야 인구·후보지 격자
- `code/12-5-unmet-demand-siting.py` — 미충족 수요와 제약 하 순차 입지 선택 (비즈니스 분석)

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python practice/chapter12/code/12-0-simdata-prep.py
python practice/chapter12/code/12-1-welfare-blindspot-priority.py
python practice/chapter12/code/12-2-infectious-spatiotemporal-risk.py
python practice/chapter12/code/12-3-dml-highdim-confounding.py
python practice/chapter12/code/12-4-satscan-cluster.py
python practice/chapter12/code/12-0b-vetcare-data-prep.py   # 원자료 필요
python practice/chapter12/code/12-5-unmet-demand-siting.py
```

예상 결과(검증 포인트)

- 12-1 단위: 읍면동 `300개`. 접근성 취약지 `15개`(5.0%), 취약지 위험 `13.56%` vs 그 외 `10.88%`
- 12-1 예측: 비공간 R² `0.681` → 공간 시차 추가 `0.737`(개선 `+0.056`)
- 12-2 데이터: 격자 `256개` × `40주`, 총 발생 `14,345건`
- 12-2 예측: 시간 시차만 R² `0.694` → 공간 시차 추가 `0.776`(개선 `+0.082`)
- 12-3 설계: **참 효과 θ = `−2.00`**. 순진 OLS `−0.579`(편향 +1.421) vs DML 교차적합 `−1.874`(편향 +0.126)
- 12-4 설계: 참 군집 상대위험 `RR = 3.0`, 단위 `21개`. 탐지 결과 관측 RR `2.52`, LLR `819.57`, p `0.0010`, 재현율·정밀도 모두 `1.00`
- 12-0b 실데이터: 서울 점포 `554,092개` 중 업종 소분류 '동물병원' `905개`(좌표 결측 0%), 행정동 `415개` 결합, 심야 생활인구 합계 `9,617,177명`, 후보지 격자 `5,462개`
- 12-5 진단: 추정 수요 `1,000,186마리`, 미충족 수요가 있는 행정동 `239개`(57.6%), 미충족 합계 `108,080마리`(전체의 10.8%)
- 12-5 선택: 탐욕 순차 10곳의 실현 포획 `19,893마리`, 자기잠식 손실률 `1.5%`, 손익분기 포획 `721마리/월`
- 12-5 대조: 미충족 수요 순위 상위 10곳은 실현 `13,399마리`·자기잠식 `18.3%` → 탐욕이 `+6,494마리`(+48.5%)
- 12-5 대조군: 인구 상위 10곳은 실현 `13,003마리`. 이격 1km 규칙을 씌운 순위표는 `14,051마리`
- 12-5 체제: 용량 배수 `1.8 → 1.4 → 1.0`에서 두 전략의 차이가 `+6,494 → +2,300 → +0`마리

검증 팁

- **가장 중요한 줄**은 12-5의 격차 분해입니다. 탐욕과 순위표의 차이(+6,494마리) 중 자기잠식이 설명하는 몫은 `+2,632`뿐이고, 더 큰 몫은 '자리 자체의 포획력'(`+8,770`)입니다. 이 분해를 빼면 "자기잠식 때문"이라는 틀린 설명을 하게 됩니다.
- 12-5의 `단계별 동점 후보 수`를 꼭 보세요. 용량 상한에 걸리는 자리끼리는 한계 이득이 같아져 **순서 자체에 정보가 없습니다.** 용량 배수를 1.0으로 낮추면 동점이 1,797개로 늘고 입지의 우열이 사라집니다.
- 12-5는 실데이터라 **공개 데이터가 갱신되면 숫자가 달라집니다.** 확인할 것은 숫자의 일치가 아니라 구조의 재현입니다(`data/raw/README.md` 마지막 절 참조).
- 12-3의 핵심은 추정값 자체가 아니라 **순진 OLS가 참값의 절반도 못 잡는다**는 방향입니다. 교란이 굽어 있는데 직선으로 통제하면 효과가 0쪽으로 끌려옵니다.
- 12-4의 몬테카를로 검정은 999회 순열이라 실행에 시간이 걸립니다. p값이 `0.001`로 고정되는 것은 999회 중 한 번도 관측 LLR을 넘지 못했다는 뜻입니다.
- 12-1 ~ 12-4는 **시뮬레이션**입니다. 참 효과·참 군집을 심어 두었기 때문에 "방법이 심은 것을 되찾는가"를 채점할 수 있습니다. 실제 데이터 실증은 별개 문제입니다.

결과 파일

- `results/12-1 ~ 12-5 *.log` — 실행 로그
- `welfare_blindspot_priority.csv`, `welfare_blindspot_priority_uncertainty.csv`, `infectious_region_priority.csv`, `satscan_cluster.csv` — 우선순위·탐지 결과
- `12-3-dml-orthogonal-residual.png`, `12-3-dml-summary.txt` — DML 산출물
- `vet_site_selection.csv`, `vet_unmet_demand.csv`, `vet_strategy_comparison.csv`, `vet_capacity_regimes.csv`, `vet_sensitivity.csv`, `12-5-vet-siting-map.png` — 동물병원 입지 산출물

연관 자료

- 교재: `docs/ch12.md` — 사회복지·보건·감염병 정책과 GeoAI
- 강의: `lecture/chapter12.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `practice/chapter12/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
- 12-0b가 "[중단] 상가(상권)정보 파일을 찾지 못했다"로 멈추면 `data/raw/README.md`의 내려받기 절차를 먼저 밟으세요.
