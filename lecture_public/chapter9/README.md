# 9장 실습: 보호구역 효과 추정, 이질적 처치효과 기반 배분, 그리고 실사 컷오프 결정

이 실습은 `docs/ch09.md`와 `lecture/chapter9.md`의 내용을 코드로 확인하는 목적입니다. 결과 로그는 `practice/chapter9/results/`에 저장되어 있습니다.

요구사항

- 루트에 `.venv` 가상환경이 생성되어 있고 활성화되어 있어야 합니다. 아직이면 `practice/README.md`의 설치 지침을 먼저 따르세요.
- `data/` 폴더의 파일은 저장소에 포함되지 않습니다. **`9-0-simdata-prep.py`를 먼저 실행**해야 나머지가 돌아갑니다.
- 네트워크는 필요하지 않습니다. 세 분석 모두 저장된 합성 자료만 씁니다.

실습 파일 (실행 순서대로)

- `practice/chapter9/code/9-0-simdata-prep.py` — 세 데이터 생성(보호구역 패널, 참 CATE 격자, 조달 구역)
- `practice/chapter9/code/9-1-protected-area-forest-loss.py` — 단순 비교 vs DID, 공간 ML 벌채위험 예측, 신뢰등급
- `practice/chapter9/code/9-2-causal-heterogeneity.py` — R-러너 CATE, GATES·BLP 이질성 검정, 배분 전략 비교
- `practice/chapter9/code/9-3-supply-chain-due-diligence.py` — 비대칭 손실 하의 실사 컷오프, 예산 곡선, 대조군

실행 방법 (Windows cmd/PowerShell / macOS Linux)

```bash
python practice/chapter9/code/9-0-simdata-prep.py
python practice/chapter9/code/9-1-protected-area-forest-loss.py
python practice/chapter9/code/9-2-causal-heterogeneity.py
python practice/chapter9/code/9-3-supply-chain-due-diligence.py
```

예상 결과(검증 포인트)

- 9-0 자료: 보호구역 패널 `6000행`(300단위×20년, 보호 150) / 참 CATE 격자 `1600개`(보호 692) / 조달 구역 `240개`(참 위반확률 평균 `0.128`, 실현 위반 `25건`)
- 9-1 설계: 단위 `300개`(보호 150), 2001~2020년, 지정 2011년, **진짜 효과 = `−0.80 %p`**
- 9-1 추정 비교: 단순 비교 `−1.306`(오차 −0.506) / DID 전체 `−0.822`(−0.022) / DID 매칭 `−0.804`(−0.004)
- 9-1 공간 ML: 공간 피처 포함 R² `0.307` vs 제외 R² `−0.080` — 공간 시차 기여 `+0.386`
- 9-1 불확실성: 90% 예측구간 경험적 포함률 `0.907`, 구간 반폭 중앙값 `0.511`(범위 `0.125`~`1.327`), 신뢰등급 높음/중간/낮음 각 `100`개
- 9-2 이질성: 참 CATE 상관 r `0.744` / GATES 1분위 − 5분위 차이 `+0.587 %p`(z = +5.52) / BLP 보정기울기 β1 `+0.519`
- 9-2 배분 전략별 총 저감손실: CATE 표적화 `123.12 %p` > 무작위 `86.17` > 사후손실 표적화 `83.28` (oracle `135.13`)
- 9-3 관측: 손실률 중위 `0.51%`, 90분위 `1.58%`, 최대 `8.761%`. 손실 면적 상위 10%(24구역)가 전체 손실 면적의 `39.6%`
- 9-3 비용비 10:1: 확률 컷오프 `0.0909`, 실사 `123`건, 기대 총비용 `149.5`(실사 1건 = 1단위), 놓친 위반 기대 `5.21`건
- 9-3 규칙 비교(R=10, 예산 K=30): 비용 최적 `149.5` < 예산 상위 K `205.7` < 전수 실사 `209.2` < 동수 손실률 컷오프 `219.8` < 제도 기본선 9% `233.3` < 무작위 `291.6` < 무실사 `307.8`
- 9-3 예산 곡선: 무실사 `307.8` → K\*=`123`건에서 `149.5`. 절감의 90%는 K=`66`건에서 달성. 현 예산 K=30은 절감의 `64.5%`만 얻음
- 9-3 대조군: 신호 0 세계에서 (위험순위 − 무작위) = `+0.0`(SD 27.2) / 신호 있는 세계에서는 `−89.0`(SD 32.0)
- 9-3 대조군: 비용 대칭(R=1)이면 확률 컷오프가 정확히 `0.5000`으로 이동, 실사 `7`건
- 9-3 민감도: 위험함수 기울기 ±50% → 기대 총비용 초과 `+1.3`·`+4.2`단위 / 비용비 오지정 → 초과 `+9.4`~`+116.0`단위

검증 팁

- 9-2에서 가장 중요한 줄은 **"손실이 큰 곳부터 보호"(83.28)가 무작위 배분(86.17)보다 못하다**는 대소 관계입니다. 손실이 큰 곳과 보호가 잘 듣는 곳이 다르기 때문입니다. 숫자가 조금 달라져도 이 부등호가 유지되는지 확인하세요.
- 9-3에서 가장 중요한 줄은 **신호 0 대조군**입니다. 컷오프 최적화와 위험순위가 좋아 보이는 이유가 정말 산림손실 신호 때문인지, 신호를 끊은 세계에서 확인합니다. 신호 0 세계에서 이득이 `+0.0` 근처로 사라지지 않으면 결과를 신호의 덕으로 해석할 수 없습니다.
- 9-3의 두 민감도를 꼭 견주어 보세요. 위험함수를 ±50% 틀린 대가(1~4단위)가 비용비를 한 자릿수 틀린 대가(최대 116단위)보다 훨씬 작습니다. 기대비용 곡선이 최적점 근처에서 평평하기 때문입니다.
- 세 실습 모두 **진짜값(정책효과·CATE·위반확률)을 심어 둔 시뮬레이션**입니다. 그래서 채점이 가능합니다. 실제 환경 평가와 실제 공급망 실사에는 정답지가 없다는 점이 핵심 메시지입니다.
- 9-3은 GFW/Hansen 실데이터를 쓰지 않습니다. 라이선스(CC BY 4.0)는 문제가 없지만 하위 행정구역별 연간 집계를 API 키 없이 받아 낼 고정 경로가 없어 재현이 불가능했습니다. 대신 손실률 분포의 오른쪽 꼬리 구조만 로그정규로 반영했습니다.
- 9-2는 교차적합이 들어가 실행에 수십 초가 걸립니다. 9-3은 스윕 계산뿐이라 몇 초면 끝납니다.

결과 파일

- `practice/chapter9/results/9-1-*.log`, `9-2-*.log`, `9-3-*.log` — 실행 로그
- `practice/chapter9/results/spatial_unit_priority.csv`, `admin_area_priority.csv`, `admin_area_priority_uncertainty.csv` — 우선순위표
- `practice/chapter9/results/9-2-cate-heterogeneity.png`, `9-2-cate-summary.txt` — 이질성 분석 산출물
- `practice/chapter9/results/9-3-due-diligence-cutoff.png` — 위험함수·컷오프와 예산 곡선
- `practice/chapter9/results/9-3-cost-ratio-cutoff.csv`, `9-3-rule-comparison.csv`, `9-3-calibration-sensitivity.csv`, `9-3-cost-ratio-misspecification.csv`, `9-3-null-control.csv` — 실사 결정 산출물

연관 자료

- 교재: `docs/ch09.md` — 환경·기후 정책과 GeoAI
- 강의: `lecture/chapter9.md` — 강의용 설명과 활동 지침

문제 발생 시

- 실행 로그와 `practice/chapter9/results/*.evidence.json`을 함께 첨부해 이 저장소 이슈 또는 수업 게시판에 올려 주세요.
