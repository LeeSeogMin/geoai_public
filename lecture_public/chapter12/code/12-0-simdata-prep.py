"""
12-0. 실습 데이터 준비: 사회복지·보건·감염병 정책 분석용 합성 데이터셋 저장
==========================================================================
이 장의 12-1·12-2·12-3·12-4 실습이 쓰는 데이터셋을 미리 만들어 `data/` 폴더에
저장한다. 학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를 준비하고,
이후 각 실습 코드는 저장된 파일을 불러와 분석만 수행한다.

네 데이터는 모두 실제 정책 미시데이터를 그대로 구하기 어렵거나(개인정보·목적제한),
'참값(신호)을 아는 상태로 방법을 검증'해야 하는 대상이라, 현실적인 값과 공간·시간
구조를 갖도록 합성한 교육용 데이터다(개인 식별 불가, 집계 단위, seed 42).

여기서 만드는 네 데이터와 그 설계 의도:

1) 복지·의료 사각지대 읍면동(12-1) — welfare_units.parquet
   미발굴 사각지대 위험은 그 지역의 위기정보뿐 아니라 '인접 지역의 위기정보·접근성'에
   spillover 의존한다(복지 전달체계는 권역 단위로 작동하고, 고립·자원 부족은 광역적으로
   나타나기 때문). '미관측 지역 고립효과 U'를 공간 자기상관을 갖도록 주입해, 자기 피처만
   쓰면 이 spillover를 놓치고 공간 시차 피처가 이를 회복하도록 설계했다.

2) 감염병 시공간 확산(12-2) — infectious_incidence.parquet, infectious_pop.parquet
   격자 SIRS 모형으로 감염 '주간 신규 발생' 시계열을 만든다. 신규 감염은 자기 감염자 +
   '이웃 감염자'에서 비롯되는 공간 확산을 그리며, 회복(I→R)·면역 소실(R→S)로 격자가
   포화되지 않고 풍토병 정상상태로 안정된다. 이웃의 직전 발생(공간 시차)이 다음 주
   신규 발생을 예측하도록 설계했다(2층 정당성 검증의 신호).

3) 고차원·비선형 교란 DML 표본(12-3) — dml_sample.parquet
   부분선형모형 Y = θ·D + g(X) + ε의 합성 표본. 결과 기저 g(X)와 처치확률 m(X)가
   '같은 비선형 교란 c(X)'(계단·V자 꺾임·포화곡선)를 공유해, 개입(D=1)이 위기 심각
   영역에 집중되게 했다(선택편향). 참 처치효과 θ=−2.0을 심어, 추정량(순진 OLS vs DML)이
   이를 복원하는지 검증한다. 이 예제는 '고차원 교란 통제'의 시연이라 합성이 본질이며,
   참값을 알아야 방법의 옳고 그름을 판정할 수 있다(construct validation).

4) 공간 스캔 통계 SaTScan 격자(12-4) — satscan_cases.parquet
   인구·좌표를 만들고, 특정 원형 영역에 상대위험 RR=3.0을 높인 발생을 심는다. 귀무(균일)
   에서는 발생이 인구에 비례하고, 참 군집 안에서만 '비정상 밀집'이 나타나도록 설계해
   SaTScan이 그 군집(중심·구성단위)을 회복하는지 검증한다.

부동소수점을 정확히 보존하도록 표 데이터는 Parquet로 저장한다(CSV로 저장하면 반올림으로
결정론적 재현이 깨진다). 각 생성 함수는 독립적으로 seed 42의 난수생성기를 새로 만들어
쓰므로, 저장 데이터는 실행 환경과 무관하게 동일하다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter12/code/12-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ===========================================================
# 1) 복지·의료 사각지대 읍면동 (12-1)
# ===========================================================
def prepare_welfare_units():
    """읍면동 단위 취약계층·위기정보·접근성을 생성해 저장(12-1).

    핵심 설계: 미발굴 사각지대 위험은 그 지역의 위기정보뿐 아니라 '인접 지역의
    위기정보·접근성'에 spillover 의존한다. → 자기 피처만 쓰면 이 spillover를 놓치고,
    공간 시차 피처가 이를 회복한다(규칙2의 비교 설계).
    """
    GRID_ROWS, GRID_COLS = 20, 15          # 300개 읍면동을 20×15 격자로 배치(공간 이웃 정의)
    N = GRID_ROWS * GRID_COLS
    rng = np.random.default_rng(42)

    def grid_neighbor_mean(rows, cols, values):
        """4-이웃 평균(시뮬레이션 내부용). 단위가 속한 권역의 spillover 계산."""
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

    rows, cols = np.divmod(np.arange(N), GRID_COLS)

    # 미관측 지역 고립효과 U: 공간 자기상관(저주파 패턴) — 관측되지 않음
    U = (np.sin(rows / 3.0) + np.cos(cols / 2.5)
         + rng.normal(0, 0.3, N))
    U = (U - U.mean()) / U.std()

    elderly = np.clip(0.18 + 0.05 * U + rng.normal(0, 0.05, N), 0.05, 0.45)
    single = np.clip(0.30 + 0.04 * U + rng.normal(0, 0.06, N), 0.10, 0.60)
    basic = np.clip(0.06 + 0.03 * U + rng.normal(0, 0.025, N), 0.01, 0.20)
    pop = np.clip(rng.normal(12000, 4000, N), 2000, 30000).round()

    # 의료·복지 이동시간(분): 외곽일수록 멀다 + 충분한 개별(idiosyncratic) 변동
    medical_access = np.clip(15 + 8 * U + rng.normal(0, 6, N), 3, 70)
    welfare_access = np.clip(12 + 6 * U + rng.normal(0, 5, N), 2, 60)
    # 위기정보 밀도(‰): 개별 변동을 크게 줘 '자기≠이웃'이 되도록(공간 시차가 별도 정보)
    crisis = np.clip(8 + 30 * basic + 6 * single + rng.normal(0, 3.0, N), 1, None)

    # 이웃 spillover(권역적 고립·자원부족): 관측되지만 '자기 피처'에는 없는 정보
    neigh_crisis = grid_neighbor_mean(rows, cols, crisis)
    neigh_access = grid_neighbor_mean(rows, cols, medical_access)

    # 진짜 사각지대 위험(미발굴 위기가구 비율, %):
    #   자기 위기정보·접근성 + 이웃 spillover + 취약계층 + 비선형 상호작용
    true_risk = (
        0.12 * crisis + 0.22 * neigh_crisis          # spillover 항(이웃)이 더 큼
        + 0.04 * medical_access + 0.10 * neigh_access
        + 14.0 * elderly + 8.0 * single
        + 0.004 * np.maximum(neigh_crisis - 15, 0) * neigh_access  # 비선형 권역 상호작용
        + rng.normal(0, 1.2, N)
    )
    df = pd.DataFrame({
        "unit_id": np.arange(N), "row": rows, "col": cols, "pop": pop,
        "elderly": elderly, "single": single, "basic": basic,
        "crisis": crisis, "medical_access": medical_access,
        "welfare_access": welfare_access, "blindspot_risk": true_risk,
    })
    df.to_parquet(DATA_DIR / "welfare_units.parquet", index=False)
    print(f"  복지 사각지대 읍면동 {len(df)}개(20×15 격자) → welfare_units.parquet")


# ===========================================================
# 2) 감염병 시공간 확산 (12-2)
# ===========================================================
def prepare_infectious():
    """격자 SIRS 모형으로 감염 '주간 신규 발생' 시계열을 생성해 저장(12-2).

    핵심 설계: 신규 감염은 자기 감염자 + '이웃 감염자'에서 비롯되는 공간 확산을 그린다.
      - 회복(I→R)과 면역 소실(R→S)이 있어 격자가 포화되지 않고 풍토병 정상상태로 안정된다.
      - 이웃의 직전 발생(공간 시차)이 다음 주 신규 발생을 예측한다 → 2층 정당(규칙2).
    """
    GRID = 16                       # 16×16 격자 = 256개 분석 단위
    T = 40                          # 40주(앞 10주 burn-in 후 정상상태 사용)
    BETA = 0.18                     # 전파율(자기 감염압)
    KAPPA = 0.10                    # 이웃 감염압 가중(공간 확산의 핵심)
    GAMMA = 0.45                    # 주간 회복률(I→R)
    OMEGA = 0.08                    # 면역 소실률(R→S) → 풍토병 정상상태 유지, 폭발 방지
    POP_CELL = 2000                 # 격자당 인구
    rng = np.random.default_rng(42)

    def neighbor_sum(mat):
        """각 격자의 4-이웃 합(공간 시차). mat: (GRID,GRID)."""
        s = np.zeros_like(mat)
        s[1:, :] += mat[:-1, :]; s[:-1, :] += mat[1:, :]
        s[:, 1:] += mat[:, :-1]; s[:, :-1] += mat[:, 1:]
        return s

    S = np.full((GRID, GRID), float(POP_CELL))
    I = np.zeros((GRID, GRID))
    R = np.zeros((GRID, GRID))
    I[3, 3] = 20; I[12, 11] = 15            # 2개 진원지
    S[3, 3] -= 20; S[12, 11] -= 15
    incidence = np.zeros((T, GRID, GRID))   # 주간 신규 발생(예측 대상)
    for t in range(T):
        force = (BETA * I + KAPPA * neighbor_sum(I)) / POP_CELL
        prob = 1.0 - np.exp(-force)                       # 감염 확률(비선형 포화)
        new_inf = rng.binomial(np.maximum(S, 0).astype(int), np.clip(prob, 0, 1))
        recover = rng.binomial(np.maximum(I, 0).astype(int), GAMMA)
        waning = rng.binomial(np.maximum(R, 0).astype(int), OMEGA)
        S = S - new_inf + waning
        I = I + new_inf - recover
        R = R + recover - waning
        incidence[t] = new_inf
    pop = np.full((GRID, GRID), POP_CELL / 1000.0)        # 상대 인구(피처용)

    # 3D 발생 시계열을 long 형식 표로 저장(Parquet, float64 정확 보존)
    tt, rr, cc = np.meshgrid(np.arange(T), np.arange(GRID), np.arange(GRID), indexing="ij")
    inc_df = pd.DataFrame({
        "t": tt.ravel(), "row": rr.ravel(), "col": cc.ravel(),
        "cases": incidence.ravel(),
    })
    inc_df.to_parquet(DATA_DIR / "infectious_incidence.parquet", index=False)

    pr, pc = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    pop_df = pd.DataFrame({"row": pr.ravel(), "col": pc.ravel(), "pop": pop.ravel()})
    pop_df.to_parquet(DATA_DIR / "infectious_pop.parquet", index=False)
    print(f"  감염병 시공간 발생 {GRID}×{GRID}격자×{T}주(총 {int(incidence.sum())}건) "
          f"→ infectious_incidence.parquet, infectious_pop.parquet")


# ===========================================================
# 3) 고차원·비선형 교란 DML 표본 (12-3)
# ===========================================================
def prepare_dml_sample():
    """부분선형모형 합성 표본을 생성해 저장(12-3).

    결과 기저 g(X)와 처치확률 m(X)가 '같은 비선형 교란 c(X)'를 공유한다 → 개입(D=1)이
    위기 심각 영역(c 큰 곳)에 집중(선택편향). c(X)는 가법 비선형 주효과(계단·꺾임·포화)라
    선형회귀(OLS)로는 못 걷어내지만 랜덤포레스트는 정확히 추정한다. 참 θ=−2.0을 심는다.
    """
    N = 6000
    P = 20                  # 위기지표 차원(고차원; c에 쓰이는 것은 소수, 나머지는 잡음지표)
    TRUE_THETA = -2.0       # 참 처치효과: 개입이 위기점수를 2.0 낮춤(음의 효과=개선)
    NOISE_SD = 1.0          # 결과 잡음 표준편차
    CLIP = 3.0              # 성향점수 로짓 절단(positivity 보장: m∈[0.05,0.95])
    rng = np.random.default_rng(42)

    def confounder(X):
        """공유 비선형 교란 c(X). 가법 '주효과'만으로 구성(상호작용 아님).

        각 항은 선형회귀로는 못 맞추지만 트리 기반 학습기는 정확히 추정하는 형태다:
          1[X0>0]        : 계단(제도 문턱 넘김)
          1[X1>0.7]      : 계단(고위험 구간 진입)
          |X2|           : V자 꺾임(양극단일수록 위험)
          logistic(3·X3) : 단조 S자 포화곡선
        상수는 대략 평균 0으로 중심화.
        """
        return (
            1.2 * (X[:, 0] > 0.0).astype(float)
            + 1.0 * (X[:, 1] > 0.7).astype(float)
            + 0.9 * np.abs(X[:, 2])
            + 1.0 / (1.0 + np.exp(-3.0 * X[:, 3]))
            - 1.6
        )

    def nonlinear_g(X):
        """결과 기저 g(X) = E[Y|X, D=0]. 공유 교란 c(X)의 2배 + 약한 선형항 몇 개."""
        return 2.0 * confounder(X) + 0.5 * X[:, 5] - 0.4 * X[:, 6]

    def propensity_logit(X):
        """처치확률 로짓 = m(X)의 선형예측자. g(X)와 같은 교란 c(X)를 공유(선택편향 원천)."""
        return 2.0 * confounder(X)

    X = rng.standard_normal((N, P))
    # 처치 배정: 비선형 성향점수 m(X)에서 베르누이 추출(위기 심각 영역에 집중)
    # positivity(중첩)를 위해 로짓을 ±CLIP로 절단 → 극단 확정처치 방지.
    m = 1.0 / (1.0 + np.exp(-np.clip(propensity_logit(X), -CLIP, CLIP)))
    D = (rng.uniform(size=N) < m).astype(float)
    # 결과: 참 처치효과 θ·D + 비선형 교란 g(X) + 잡음
    Y = TRUE_THETA * D + nonlinear_g(X) + rng.normal(0, NOISE_SD, N)

    # nuisance 정확도 점검에 필요한 참 함수값 ℓ(X)=E[Y|X]=θ·m(X)+g(X)도 함께 저장
    ey_true = TRUE_THETA * m + nonlinear_g(X)

    cols = {f"x{j:02d}": X[:, j] for j in range(P)}
    cols.update({"D": D, "Y": Y, "m": m, "ey_true": ey_true})
    df = pd.DataFrame(cols)
    df.to_parquet(DATA_DIR / "dml_sample.parquet", index=False)
    print(f"  DML 합성 표본 N={N}, 위기지표 p={P}(+D,Y,m,ey_true) → dml_sample.parquet")


# ===========================================================
# 4) 공간 스캔 통계 SaTScan 격자 (12-4)
# ===========================================================
def prepare_satscan_cases():
    """인구·좌표와 참 군집(RR을 높인 원형 영역)을 심은 발생을 생성해 저장(12-4).

    귀무(균일)에서는 발생이 인구에 비례한다. 참 군집 안에서는 위험을 RR배로 높여
    '비정상 밀집'을 심는다. SaTScan이 이 군집을 회복하는지가 검증 목표다.
    """
    GRID_ROWS, GRID_COLS = 20, 15          # 300개 단위(격자)
    N = GRID_ROWS * GRID_COLS
    BASE_RATE = 0.02                       # 인구당 기저 발생률
    TRUE_RR = 3.0                          # 참 군집의 상대위험(고위험)
    CLUSTER_CENTER = (6.0, 4.0)            # 참 군집 중심(row, col)
    CLUSTER_RADIUS = 2.6                   # 참 군집 반경(격자 단위)
    rng = np.random.default_rng(42)

    rows, cols = np.divmod(np.arange(N), GRID_COLS)
    coords = np.column_stack([rows, cols]).astype(float)

    # 인구: 완만한 공간 변동(도심일수록 많음)을 준 뒤 양수로 클립
    pop = np.clip(1500 + 400 * np.sin(rows / 3.0) + 300 * np.cos(cols / 2.5)
                  + rng.normal(0, 200, N), 300, None)

    # 참 군집: 중심에서 반경 이내 단위에 상대위험 RR
    d_center = np.hypot(coords[:, 0] - CLUSTER_CENTER[0],
                        coords[:, 1] - CLUSTER_CENTER[1])
    in_cluster = d_center <= CLUSTER_RADIUS
    rr = np.where(in_cluster, TRUE_RR, 1.0)

    # 발생: 포아송(인구 × 기저율 × 상대위험)
    obs = rng.poisson(pop * BASE_RATE * rr).astype(float)

    df = pd.DataFrame({
        "unit_id": np.arange(N),
        "row": coords[:, 0], "col": coords[:, 1],
        "pop": pop, "obs": obs, "in_cluster": in_cluster,
    })
    df.to_parquet(DATA_DIR / "satscan_cases.parquet", index=False)
    print(f"  SaTScan 감염병 격자 {N}개(20×15), 참 군집 {int(in_cluster.sum())}개 "
          f"→ satscan_cases.parquet")


def main():
    print("=" * 64)
    print("실습 데이터 준비 (12-1 · 12-2 · 12-3 · 12-4)")
    print("=" * 64)
    prepare_welfare_units()
    prepare_infectious()
    prepare_dml_sample()
    prepare_satscan_cases()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
