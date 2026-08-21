"""
9-0. 실습 데이터 준비: 환경·기후 정책 분석용 합성 데이터셋 저장
================================================================
이 장의 9-1·9-2 실습이 쓰는 데이터셋을 미리 만들어 `data/` 폴더에 저장한다.
학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를 준비하고, 이후 각
실습 코드는 저장된 파일을 불러와 분석만 수행한다.

9장은 인과 이질성(CATE)과 정책효과 식별을 시연하는 장이라 **합성이 본질**이다.
보호구역의 참 정책효과·참 CATE는 현실에서 관측할 수 없으므로, 선택 편향·평행추세·
정책효과·이질성을 명시적 파라미터로 심은 합성 자료로 추정량의 작동을 검증한다.
(실제 국내 분석에서는 환경부·산림청 보호지역·산림 공간자료, 토지피복도, 개발행위
자료, 위성 기반 산림 변화 지표(Hansen GFC / Global Forest Watch)를 결합한다.)

여기서 만드는 세 데이터:
- 보호구역 산림손실 패널(9-1): 선택 편향·평행추세·정책효과·노이즈를 심은 300단위 20년 패널.
- 참 CATE 격자 자료(9-2): 집행역량이 조절하는 참 CATE와 선택 편향을 심은 40×40 격자 단면.
- 조달 구역 실사 자료(9-3): 산림손실률과 참 위반확률을 심은 240개 해외 조달 구역 단면.

데이터는 합성이지만, **분석과 추정 결과는 실제 계산값**이다(가짜 아님).

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter9/code/9-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def prepare_forest_panel():
    """보호구역 selection bias를 포함한 산림손실 패널 데이터 생성(9-1).

    핵심 설계: '개발 압력(pressure)'이 높을수록 손실이 크고, 보호구역은 압력이
    낮은 곳에 우선 지정된다 → 단순 비교는 편향된다. 여기에 공통 시간추세(평행추세의
    근거)와 처리군·사후에만 걸리는 진짜 정책효과, 관측 노이즈를 명시적으로 심는다.
    """
    rng = np.random.default_rng(42)

    # 시뮬레이션 파라미터 (정책 시나리오)
    N_UNITS = 300          # 분석 단위(산림 격자) 수
    N_PROTECTED = 150      # 보호구역(처리군)
    YEARS = np.arange(2001, 2021)   # 20년
    TREAT_YEAR = 2011      # 보호구역 지정 발효 연도
    TRUE_EFFECT = -0.80    # 진짜 정책 효과: 보호로 연간 손실률 0.80%p 감소 (음수=저감)
    BASE_LOSS = 2.5        # 기준 연간 산림손실률(%) 평균
    PRESSURE_COEF = 2.0    # 개발 압력이 손실률에 미치는 영향 (혼란변수)
    TIME_TREND = 0.05      # 공통 시간 추세(평행추세 가정의 근거)
    NOISE_SD = 0.40        # 관측 노이즈
    SPATIAL_COEF = 1.5     # 관측되지 않는 공간 효과의 손실 기여(공간 시차 피처가 잡아냄)
    GRID_COLS = 15         # 300단위를 20행×15열 격자로 배치(공간 명시적 피처용, 결정론적)

    # 격자 좌표 기반 매끄러운 공간장 2개 생성 (공간 자기상관 = Tobler 제1법칙).
    g_rows = N_UNITS // GRID_COLS
    yy, xx = np.meshgrid(np.linspace(0, 1, g_rows),
                         np.linspace(0, 1, GRID_COLS), indexing="ij")

    def smooth_field(fx, fy, noise_sd):
        f = (np.sin(2 * np.pi * fx * xx) + np.cos(2 * np.pi * fy * yy)
             + noise_sd * rng.normal(0, 1, (g_rows, GRID_COLS)))
        f = f.reshape(-1)
        return (f - f.min()) / (f.max() - f.min())

    # 개발 압력: 공간 자기상관을 가진 관측 가능 혼란변수(피처로 사용).
    pressure = smooth_field(1.5, 1.2, 0.35)
    # 관측되지 않는 공간 효과(토양·접근성 등): 피처에 없음 → 공간 시차로만 포착 가능.
    spatial_effect = smooth_field(2.3, 1.8, 0.35)

    # 보호구역 선정: 압력이 낮을수록 지정 확률↑ (selection bias의 원천)
    select_prob = 1.0 - pressure              # 저압력일수록 선정 가능성↑
    order = np.argsort(-select_prob + rng.normal(0, 0.15, N_UNITS))
    treated = np.zeros(N_UNITS, dtype=int)
    treated[order[:N_PROTECTED]] = 1          # 상위 N_PROTECTED개를 보호구역으로

    # 단위별 기준 손실률: 압력 + 관측 안 되는 공간 효과 + 개체 임의효과
    unit_base = (BASE_LOSS + PRESSURE_COEF * pressure
                 + SPATIAL_COEF * spatial_effect + rng.normal(0, 0.30, N_UNITS))

    admin_area = np.array([f"A{(i // 30) + 1:02d}" for i in range(N_UNITS)])
    habitat_value = np.clip(1.0 - pressure + rng.normal(0, 0.12, N_UNITS), 0, 1)

    rows = []
    for i in range(N_UNITS):
        for t in YEARS:
            post = 1 if t >= TREAT_YEAR else 0
            # 손실률 = 단위기준 + 공통 시간추세 + 정책효과(처리군 & 사후) + 노이즈
            loss = (
                unit_base[i]
                + TIME_TREND * (t - YEARS[0])
                + TRUE_EFFECT * (treated[i] * post)
                + rng.normal(0, NOISE_SD)
            )
            loss = max(loss, 0.0)   # 손실률은 음수가 될 수 없음
            rows.append((i, admin_area[i], t, treated[i], post, pressure[i], habitat_value[i], loss))

    df = pd.DataFrame(
        rows,
        columns=[
            "unit", "admin_area", "year", "treated", "post",
            "pressure", "habitat_value", "loss",
        ],
    )
    # 부동소수점을 정확히 보존하도록 Parquet로 저장(추정 재현성 확보)
    df.to_parquet(DATA_DIR / "forest_loss_panel.parquet", index=False)
    print(f"  산림손실 패널 {len(df)}행({N_UNITS}단위×{len(YEARS)}년, 보호 {N_PROTECTED})"
          f" → forest_loss_panel.parquet")


def prepare_cate_grid():
    """참 CATE와 선택편향을 심은 합성 보호구역 단면 자료 생성(9-2).

    핵심 설계:
      - enforce(집행역량)가 처치효과 τ의 조절변수다: 집행역량이 높을수록 보호가
        벌채를 더 많이 막는다(τ 큼). enforce는 손실 '수준'에는 영향이 없다.
      - pressure(개발압력)은 손실 '수준'(μ0)과 보호 '지정'(선택편향)을 결정한다.
        압력이 높으면 손실이 크지만, 보호 지정 확률은 낮다(저압력이 보호됨).
      - 결정적으로 enforce와 pressure는 서로 다른 공간장이다 → 손실이 큰 곳(고압력)과
        보호가 잘 듣는 곳(고집행역량)이 어긋난다. 관측 손실만으로는 후자를 못 찾는다.
      - 잡음 공변량(elevation·slope)은 결과·처치·CATE 어디에도 영향이 없다
        → RF가 이들로 '헛이질성'을 만들지 않는지 점검하는 대조군.
      - 교란은 전부 관측 가능(무교란 성립). CATE 복원 검증이 목적이므로 미관측
        교란은 심지 않는다(실제 상황의 미관측 교란은 본문 한계에서 명시).
    """
    rng = np.random.default_rng(42)

    # DGP 파라미터 (참 CATE를 심은 합성 보호구역 자료)
    GRID_ROWS, GRID_COLS = 40, 40      # 1,600 격자
    N = GRID_ROWS * GRID_COLS
    NOISE_SD = 0.5                     # 결과 관측 잡음
    # 참 CATE: τ(x) = −(TAU0 + TAU_E·enforce)  (음수 = 손실 저감)
    TAU0, TAU_E = 0.20, 1.40

    def smooth_field(freq_x, freq_y, noise_sd):
        """격자 좌표 기반 매끄러운 공간장을 0~1로 정규화(공간 자기상관=Tobler 제1법칙)."""
        yy, xx = np.meshgrid(
            np.linspace(0, 1, GRID_ROWS), np.linspace(0, 1, GRID_COLS), indexing="ij"
        )
        f = (np.sin(2 * np.pi * freq_x * xx) + np.cos(2 * np.pi * freq_y * yy)
             + noise_sd * rng.normal(0, 1, (GRID_ROWS, GRID_COLS)))
        f = f.reshape(-1)
        return (f - f.min()) / (f.max() - f.min())

    row = np.repeat(np.arange(GRID_ROWS), GRID_COLS)
    col = np.tile(np.arange(GRID_COLS), GRID_ROWS)

    pressure = smooth_field(1.5, 1.2, 0.30)        # 손실 수준 driver & 선택편향 교란
    enforce = smooth_field(2.1, 1.7, 0.30)         # 처치효과 조절변수(수준과 독립)
    habitat = np.clip(1.0 - pressure + rng.normal(0, 0.10, N), 0, 1)  # 서식지 가치
    elevation = smooth_field(2.8, 0.7, 0.30)       # 잡음 공변량(무관)
    slope = smooth_field(0.6, 3.1, 0.30)           # 잡음 공변량(무관)

    # 참 CATE: 집행역량이 높을수록 보호효과가 크다(음의 저감). 다른 변수와 독립.
    true_tau = -(TAU0 + TAU_E * enforce)

    # 성향점수(선택편향): 저압력·고서식지일수록 보호 지정 확률↑(9.6과 동형).
    logit = -2.2 * (pressure - 0.5) + 1.0 * (habitat - 0.5)
    e_true = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.05, 0.95)  # positivity 유지
    treated = (rng.uniform(0, 1, N) < e_true).astype(int)

    # 기저결과 μ0=E[Y(0)|x]: 손실 수준은 개발압력이 좌우(집행역량 enforce는 무관).
    mu0 = 2.4 + 2.6 * pressure - 0.6 * habitat + rng.normal(0, 0.30, N)
    y = mu0 + true_tau * treated + rng.normal(0, NOISE_SD, N)
    y = np.maximum(y, 0.0)  # 손실률은 음수가 될 수 없음

    df = pd.DataFrame({
        "row": row, "col": col,
        "pressure": pressure, "enforce": enforce, "habitat": habitat,
        "elevation": elevation, "slope": slope,
        "treated": treated, "e_true": e_true, "true_tau": true_tau, "y": y,
    })
    # 부동소수점을 정확히 보존하도록 Parquet로 저장(추정 재현성 확보)
    df.to_parquet(DATA_DIR / "cate_grid.parquet", index=False)
    print(f"  참 CATE 격자 {len(df)}개({GRID_ROWS}×{GRID_COLS}, 보호 {int(treated.sum())})"
          f" → cate_grid.parquet")


def prepare_sourcing_districts():
    """해외 조달 구역의 산림손실률과 참 위반확률을 심은 단면 자료 생성(9-3).

    문제 상황: 국내 식품·목재 기업이 표준위험 생산국의 하위 행정구역에서 원료를
    조달한다. 각 구역의 연간 산림손실률은 위성으로 관측되지만, 그 구역에서 실제로
    산림전용 위반이 있었는가는 관측되지 않는다(공개 라벨이 없다).

    설계 원칙 — 관측 가능한 층과 관측 불가능한 층을 나눈다:
      - 관측 가능: 연간 산림손실률(loss_rate_pct), 산림 면적, 거버넌스 지표.
        손실률은 로그정규로 두어 '소수 구역에 손실이 집중되는 오른쪽 꼬리'만 반영한다.
        특정 국가·연도의 실제 손실률을 재현하려는 것이 아니다(공개 자료의 무인증
        재현 경로가 없어 전량 합성으로 갔다. 근거: content/research/ch9-business-*.md).
      - 관측 불가: 실제 위반 여부(violation)와 참 위반확률(p_true).
        여기서 위험함수의 계수는 **추정하는 것이 아니라 선언한다.** 규칙으로 정의한
        값을 지도학습으로 복원하면 순환논증이기 때문이다. 선언한 가정은 9-3의
        민감도 분석과 대조군으로 되갚는다.

    아래 계수는 9-3-supply-chain-due-diligence.py와 동일한 값을 쓴다(같은 위험함수).
    """
    rng = np.random.default_rng(42)

    N_DISTRICTS = 240                 # 조달 구역 수
    N_COUNTRIES = 4                   # 표준위험 생산국 4개국
    LOSS_LOG_MEDIAN = np.log(0.55)    # 연간 산림손실률 중위 0.55%
    LOSS_LOG_SD = 0.95                # 로그 표준편차(오른쪽 꼬리)
    AREA_LOG_MEDIAN = np.log(50_000)  # 구역 산림면적 중위 5만 ha
    AREA_LOG_SD = 0.60
    # 위반 위험함수(선언값): logit p = B0 + B_LOSS·z(log 손실률) + B_GOV·(0.5 − 거버넌스)·2
    B0, B_LOSS, B_GOV = -2.30, 1.10, 1.60

    loss_rate = np.exp(rng.normal(LOSS_LOG_MEDIAN, LOSS_LOG_SD, N_DISTRICTS))
    forest_area = np.exp(rng.normal(AREA_LOG_MEDIAN, AREA_LOG_SD, N_DISTRICTS))
    # 거버넌스 지표: 0에 가까울수록 토지·산림 행정이 약하다(관측 가능한 구역 특성)
    governance = rng.beta(5.0, 5.0, N_DISTRICTS)

    # 표준화된 log 손실률 — 손실률의 '상대 위치'가 위험을 좌우하게 만든다
    z_loss = (np.log(loss_rate) - LOSS_LOG_MEDIAN) / LOSS_LOG_SD
    logit = B0 + B_LOSS * z_loss + B_GOV * (0.5 - governance) * 2.0
    p_true = 1.0 / (1.0 + np.exp(-logit))
    violation = (rng.uniform(0, 1, N_DISTRICTS) < p_true).astype(int)

    df = pd.DataFrame({
        "district": [f"D{i + 1:03d}" for i in range(N_DISTRICTS)],
        "country": [f"C{(i % N_COUNTRIES) + 1}" for i in range(N_DISTRICTS)],
        "forest_area_ha": forest_area,
        "loss_rate_pct": loss_rate,
        "governance": governance,
        "p_true": p_true,
        "violation": violation,
    })
    df.to_parquet(DATA_DIR / "sourcing_districts.parquet", index=False)
    print(f"  조달 구역 {len(df)}개(참 위반확률 평균 {p_true.mean():.3f},"
          f" 실현 위반 {int(violation.sum())}건) → sourcing_districts.parquet")


def main():
    print("=" * 60)
    print("9장 실습 데이터 준비 (9-1 · 9-2 · 9-3)")
    print("=" * 60)
    prepare_forest_panel()
    prepare_cate_grid()
    prepare_sourcing_districts()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
