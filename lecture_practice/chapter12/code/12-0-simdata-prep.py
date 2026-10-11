"""
13-0. 실습 데이터 준비: 지역균형발전·인구감소 분석용 데이터셋 저장
=====================================================================
이 장의 13-1·13-2·13-3 실습이 쓰는 데이터셋을 미리 만들어 `data/` 폴더에
저장한다. 학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를 준비하고,
이후 각 실습 코드는 저장된 파일을 불러와 분석만 수행한다.

여기서 만드는 세 데이터는 실제 공개 데이터를 그대로 구하기 어려운 대상이라
현실적인 값과 공간 구조를 갖도록 합성한 교육용 데이터다(개인 식별 불가, 집계 단위).
- 시군구 인구·경제·접근성 지표(소멸위험·취약성 유형화 실습, 13-1)
- 야간조명 경제 프록시·행정지표·미래 쇠퇴(측정오차 보정·쇠퇴 예측 실습, 13-2)
- 지자체 야간조명 경제지수 패널(합성통제 인과추정 실습, 13-3)

13-3의 패널은 *참 효과 α를 아는 요인모형 DGP*로, 추정량의 타당성을 검증하기 위한
위조 대조군 시연이다(합성이 본질). 설계 근거는 아래 make_scm_panel 주석 참조.

각 데이터셋의 난수 스트림은 서로 독립이다(생성 함수마다 별도의
np.random.default_rng(42)를 두어, 원래의 파일별 시드 스트림을 그대로 보존한다).
실수 값은 Parquet(float64)로 저장해 부동소수점을 정확히 보존한다(모델 재현성).

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python lecture_practice/chapter13/code/13-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

GRID_ROWS, GRID_COLS = 20, 15
N = GRID_ROWS * GRID_COLS               # 300개 시군구·읍면동


# ============================================================
# 13-1: 시군구 인구·경제·접근성 지표
# ============================================================
def prepare_decline_regions():
    """시군구 인구·경제·접근성 지표 생성(13-1 소멸위험·취약성·유형화).

    핵심 설계: 외곽(수도권에서 먼) 지역일수록 고령화·청년유출·접근성 악화가
    동반된다(공간 자기상관). 도시→농산어촌 경사를 행 인덱스로 근사.
    """
    rng = np.random.default_rng(42)
    rows, cols = np.divmod(np.arange(N), GRID_COLS)
    # 도시성(urbanity): 좌상단을 도심으로, 우하단을 농산어촌으로
    urbanity = 1.0 - (rows / (GRID_ROWS - 1) * 0.6 + cols / (GRID_COLS - 1) * 0.4)
    urbanity = urbanity + rng.normal(0, 0.08, N)

    # 인구구조: 도시일수록 청년 비중↑, 농촌일수록 고령↑
    women_2039 = np.clip(0.10 + 0.10 * urbanity + rng.normal(0, 0.02, N), 0.02, 0.30)
    elderly_65 = np.clip(0.30 - 0.18 * urbanity + rng.normal(0, 0.03, N), 0.08, 0.45)
    # 지방소멸위험지수 = (20–39세 여성) / (65세 이상)  (낮을수록 위험)
    decline_risk_index = women_2039 / elderly_65

    youth_netmove = 5 * urbanity - 3 + rng.normal(0, 1.2, N)        # %, 음수=유출
    service_access = np.clip(40 - 28 * urbanity + rng.normal(0, 5, N), 5, 70)  # 분, 클수록 취약
    biz_density = np.clip(20 + 60 * urbanity + rng.normal(0, 8, N), 3, None)    # 개/㎢

    df = pd.DataFrame({
        "region_id": np.arange(N), "row": rows, "col": cols,
        "women_2039": women_2039, "elderly_65": elderly_65,
        "decline_risk_index": decline_risk_index,
        "youth_netmove": youth_netmove, "service_access": service_access,
        "biz_density": biz_density,
    })
    df.to_parquet(DATA_DIR / "decline_regions.parquet", index=False)
    print(f"  시군구 지표 {len(df)}개(20×15) → decline_regions.parquet")


# ============================================================
# 13-2: 야간조명 경제 프록시·행정지표·미래 쇠퇴
# ============================================================
def _neighbor_mean(rows, cols, values):
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid[rows, cols] = values
    out = np.zeros(N)
    for i, (r, c) in enumerate(zip(rows, cols)):
        neigh = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                neigh.append(grid[rr, cc])
        out[i] = np.nanmean(neigh)
    return out


def prepare_econ_regions():
    """진짜 경제활동(관측 불가) → 야간조명(측정오차) + 행정지표 + 미래 쇠퇴(13-2).

    야간조명은 진짜 경제활동에 측정오차를 더한 거친 프록시(Henderson 2012)로,
    분석에서 attenuation(회귀계수가 0쪽으로 수축하는 측정오차 편의) 보정을 시연한다.
    미래 쇠퇴율에는 인접 시군구의 쇠퇴가 번지는 공간 spillover를 더한다.
    """
    MEAS_SD = 0.55          # 야간조명 측정오차 표준편차(거친 프록시) — 클수록 attenuation↑
    rng = np.random.default_rng(42)
    rows, cols = np.divmod(np.arange(N), GRID_COLS)
    # 진짜 경제활동(표준화): 공간 자기상관 패턴 (관측되지 않음)
    true_econ = (np.sin(rows / 3.0) + np.cos(cols / 2.5) + rng.normal(0, 0.4, N))
    true_econ = (true_econ - true_econ.mean()) / true_econ.std()

    # 야간조명: 진짜 경제활동 + 측정오차 (거친 프록시, Henderson 2012)
    nightlight = true_econ + rng.normal(0, MEAS_SD, N)

    # 행정지표(관측): 진짜 경제와 상관 + 자체 잡음
    aging = np.clip(0.25 - 0.06 * true_econ + rng.normal(0, 0.03, N), 0.08, 0.45)
    biz = np.clip(40 + 12 * true_econ + rng.normal(0, 6, N), 5, None)

    # 미래 쇠퇴율(%, 클수록 쇠퇴): 경제 낮고 고령화 높을수록 쇠퇴 + 공간 spillover
    decline = (-2.0 * true_econ + 8.0 * aging + rng.normal(0, 0.8, N))
    decline = decline + 0.3 * _neighbor_mean(rows, cols, decline)

    df = pd.DataFrame({
        "region_id": np.arange(N), "row": rows, "col": cols,
        "true_econ": true_econ, "nightlight": nightlight,
        "aging": aging, "biz": biz, "decline": decline,
    })
    df.to_parquet(DATA_DIR / "econ_regions.parquet", index=False)
    print(f"  야간조명·쇠퇴 지표 {len(df)}개(20×15) → econ_regions.parquet")


# ============================================================
# 13-3: 지자체 야간조명 경제지수 패널 (요인모형 SCM DGP)
# ============================================================
# 단위 J+1개(처치 1 + 기증 풀 J). 시점 T개, 처치 시점 T0 이후 처치 단위에 참 효과 α.
# 결과 Y = 기저 + λ(단위 적재)·μ(공통요인) + 잡음. 단위: 야간조명 경제지수(기준 100).
J = 30                  # 기증 풀(통제 지자체) 수
N_UNITS = J + 1         # 처치 단위 1 + 기증 풀 J
T = 30                  # 총 연도 수
T0 = 20                 # 처치 시점(21년차부터 처치). 사전 20 + 사후 10
K = 3                   # 잠재 공통요인 수
TRUE_EFFECT_FINAL = 8.0  # 참 효과: 처치 후 마지막 해 +8 지수(기금 투입 → 경제 상승)
NOISE_SD = 1.3          # 관측 잡음(작아야 사전 적합 양호)


def make_scm_panel():
    """합성 지자체 패널 생성(요인모형 — Abadie et al. 2010의 SCM 정당화 구조).

    처치 단위의 기저·적재를 몇몇 기증 단위의 볼록조합으로 설정 → 좋은 합성 대조가
    기증 풀의 볼록껍질 안에 실제로 존재하도록(SCM이 복원할 수 있는 조건). 참 효과는
    처치 후 선형 증가(αₜ: 0 → TRUE_EFFECT_FINAL)로 심는다.

    이 패널은 위조 대조군 시연을 위한 합성 자료가 본질이다: 참 효과 α를 알기에
    추정량이 (a) 사전기간을 잘 적합하고 (b) 처치 후 효과를 복원하는지 검증할 수 있다.

    반환:
      Y      : (N_UNITS, T) 관측 결과. 행 0 = 처치 단위.
      cov    : (N_UNITS, 2) 시불변 공변량(고령화율·청년인구비 프록시).
      alpha_t: (T,) 처치 단위에 심은 참 효과(사전기간 0).
      w_true : (N_UNITS,) DGP 가중치를 **전체-단위 프레임**(index 0=처치, 1..J=기증)으로
               정렬한 벡터.
    """
    rng = np.random.default_rng(42)
    # 공통요인 μ(t,k): 매끄러운 시간 궤적(추세·순환·완만한 드리프트)
    t = np.arange(T)
    mu = np.column_stack([
        0.6 * t / T,                      # 완만한 상승 추세
        np.sin(2 * np.pi * t / 14),       # 경기 순환
        np.cos(2 * np.pi * t / 9) * 0.7,  # 짧은 순환
    ])  # (T, K)

    # 기증 단위 적재 λ(j,k) ≥ 0, 기저, 공변량
    donor_load = rng.uniform(0.5, 3.0, size=(J, K))
    donor_base = rng.uniform(94.0, 106.0, size=J)
    donor_cov = np.column_stack([
        rng.uniform(0.12, 0.40, J),       # 고령화율
        rng.uniform(0.08, 0.25, J),       # 청년인구비
    ])

    # 처치 단위 = 4개 기증 단위의 볼록조합(볼록껍질 안 → 좋은 합성 존재)
    support = np.array([2, 7, 15, 23])
    w_true = np.zeros(J)
    w_true[support] = [0.40, 0.30, 0.20, 0.10]
    treat_load = w_true @ donor_load
    treat_base = w_true @ donor_base
    treat_cov = w_true @ donor_cov

    # 결과 조립: Y = 기저 + λ·μᵀ + 잡음
    Y = np.empty((N_UNITS, T))
    Y[0] = treat_base + treat_load @ mu.T + rng.normal(0, NOISE_SD, T)
    Y[1:] = donor_base[:, None] + donor_load @ mu.T + rng.normal(0, NOISE_SD, (J, T))

    cov = np.vstack([treat_cov, donor_cov])

    # 참 효과: 사전 0, 처치 후 선형 증가
    alpha_t = np.zeros(T)
    post = np.arange(T0, T)
    alpha_t[T0:] = TRUE_EFFECT_FINAL * (post - T0 + 1) / (T - T0)
    Y[0] += alpha_t  # 처치 단위에만 심음

    # DGP 가중치를 전체-단위 프레임(0=처치, 기증 d → index d+1)으로 정렬
    w_true_full = np.zeros(N_UNITS)
    w_true_full[1:] = w_true

    return Y, cov, alpha_t, w_true_full


def prepare_scm_panel():
    """SCM 패널(Y·공변량·참효과·DGP 가중치)을 Parquet로 저장(13-3).

    - scm_Y.parquet     : 단위(N_UNITS)×연도(T). 열 t0..t{T-1}, 행 0=처치 단위.
    - scm_units.parquet : 단위별 시불변 정보(공변량 2열 + DGP 참 가중치).
    - scm_alpha.parquet : 연도별 참 효과 αₜ(사전기간 0).
    Parquet(float64)로 저장해 SLSQP 최적화 입력을 바이트 단위로 보존한다.
    """
    Y, cov, alpha_t, w_true = make_scm_panel()

    pd.DataFrame(Y, columns=[f"t{i}" for i in range(T)]).to_parquet(
        DATA_DIR / "scm_Y.parquet", index=False)
    pd.DataFrame({
        "cov_aging": cov[:, 0], "cov_youth": cov[:, 1], "w_true": w_true,
    }).to_parquet(DATA_DIR / "scm_units.parquet", index=False)
    pd.DataFrame({"alpha_t": alpha_t}).to_parquet(
        DATA_DIR / "scm_alpha.parquet", index=False)
    print(f"  SCM 패널 {N_UNITS}단위×{T}년 → scm_Y.parquet·scm_units.parquet·scm_alpha.parquet")


def main():
    print("=" * 64)
    print("13장 실습 데이터 준비 (13-1 · 13-2 · 13-3)")
    print("=" * 64)
    prepare_decline_regions()
    prepare_econ_regions()
    prepare_scm_panel()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
