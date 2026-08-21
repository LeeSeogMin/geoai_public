"""
8-0. 실습 데이터 준비: 인과추론용 패널 데이터 생성·저장
====================================================
이 장의 8-1·8-2 실습이 쓰는 패널 데이터를 미리 만들어 `data/` 폴더에 저장한다.
학습자는 실습 전에 이 스크립트를 한 번 실행해 데이터를 준비하고, 이후 각 실습
코드는 저장된 파일을 불러와 추정·분석만 수행한다.

8장은 인과추론(DID/DML)을 가르치는 방법론 허브다. 인과추론 실습은 본질적으로
**진짜 처치효과(ATT)를 미리 알고 그 값을 추정량이 얼마나 잘 회복하는지 검증하는
설계**이므로, 데이터는 진짜값을 심을 수 있는 통제된 시뮬레이션을 쓴다(개인 식별
불가). 실제 정책평가에서는 진짜값이 없으며, 이 데이터는 방법의 작동 원리를 드러
내는 실험실 역할만 한다. 실제로는 통계청 SGIS 격자통계 + VIIRS 야간조명 + 행정
개발사업 자료를 결합한다.

여기서 만드는 세 데이터:
- 야간조명 패널 + 고차원 공변량 + 선택적 사업 배치 (8-1, DID/DML)
- 시차 도입(staggered adoption) 패널 + 동적 처치효과 (8-2, TWFE vs CS)
- 신규 출점 격자 패널 세 세계(관측·잠식 0·효과 0) (8-3, IPW-DID + 도넛 설계)

부동소수점을 정확히 보존해 추정치 재현성을 확보하도록 패널은 Parquet로 저장한다.

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter8/code/8-0-simdata-prep.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _store_dgp import make_store_panel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ===========================================================
# 8-1 데이터: 야간조명 패널 + 고차원 공변량 + 선택적 사업 배치
# ===========================================================
# 시뮬레이션 파라미터
GRID_COLS = 15
N_REGIONS = 300                       # 20행 × 15열 격자(읍면동 가정)
YEARS = np.arange(2016, 2024)         # 8년
TREAT_YEAR = 2020                     # 개발사업 착수
TRUE_ATT = 0.50                       # 진짜 정책효과: 야간조명 성장 +0.50(로그 radiance)
N_COVAR = 8                           # 고차원 공변량 수


def prepare_nightlight():
    """야간조명 패널 + 고차원(일부 공간) 공변량 + 선택적 사업 배치를 저장한다(8-1).

    사업은 잠재 성장력이 높은 지역에 선택적으로 배치된다(selection on observables).
    진짜 정책효과(ATT)는 야간조명 성장 +0.50으로 결정론적으로 심어, 뒤의 추정량이
    이 값을 얼마나 회복하는지 검증한다.
    """
    RNG = np.random.default_rng(42)

    def smooth_field(fx, fy, sd):
        """격자 위 매끄러운 공간장(공간 자기상관) → [0,1]."""
        g_rows = N_REGIONS // GRID_COLS
        yy, xx = np.meshgrid(np.linspace(0, 1, g_rows),
                             np.linspace(0, 1, GRID_COLS), indexing="ij")
        f = (np.sin(2 * np.pi * fx * xx) + np.cos(2 * np.pi * fy * yy)
             + sd * RNG.normal(0, 1, (g_rows, GRID_COLS))).reshape(-1)
        return (f - f.min()) / (f.max() - f.min())

    # 고차원 공변량: 2개는 공간 자기상관(접근성·도시화), 나머지는 일반
    cov = {}
    cov["x_urban"] = smooth_field(1.5, 1.2, 0.3)     # 도시화(공간)
    cov["x_access"] = smooth_field(2.2, 1.8, 0.3)    # 교통 접근성(공간)
    for k in range(N_COVAR - 2):
        cov[f"x{k}"] = RNG.uniform(0, 1, N_REGIONS)
    X = pd.DataFrame(cov)

    # 잠재 성장력(비선형): 도시화·접근성의 상호작용 등 → 선형통제로 못 잡음
    latent = (1.2 * X["x_urban"] * X["x_access"]
              + 0.8 * X["x_urban"] ** 2 + 0.4 * X["x0"])
    latent = (latent - latent.mean()) / latent.std()

    # 사업 배치: 잠재 성장력이 높은 곳에 선택적으로(selection on observables)
    prop = 1 / (1 + np.exp(-1.3 * latent + RNG.normal(0, 0.5, N_REGIONS)))
    treated = (prop > np.median(prop)).astype(int)

    base_light = 2.0 + 1.5 * X["x_urban"] + RNG.normal(0, 0.2, N_REGIONS)

    rows = []
    for i in range(N_REGIONS):
        for t in YEARS:
            post = 1 if t >= TREAT_YEAR else 0
            # 공통 추세 + (사후) 잠재 성장력에 비례하는 시변 교란 + 정책효과 + 노이즈
            light = (base_light[i]
                     + 0.04 * (t - YEARS[0])                       # 공통 추세(평행)
                     + post * 0.35 * latent[i]                     # 시변 교란(선택 편향원)
                     + TRUE_ATT * treated[i] * post                # 진짜 정책효과
                     + RNG.normal(0, 0.15))
            rows.append((i, t, post, treated[i], light))

    panel = pd.DataFrame(rows, columns=["region", "year", "post", "treated", "light"])

    # 부동소수점을 정확히 보존하도록 Parquet로 저장(추정치 재현성 확보)
    panel.to_parquet(DATA_DIR / "nightlight_panel.parquet", index=False)
    X.to_parquet(DATA_DIR / "nightlight_covariates.parquet", index=False)
    print(f"  야간조명 패널 {len(panel)}행(격자 {N_REGIONS}×{len(YEARS)}년) → nightlight_panel.parquet")
    print(f"  고차원 공변량 {X.shape[1]}개 → nightlight_covariates.parquet")


# ===========================================================
# 8-2 데이터: 시차 도입(staggered) 패널 + 동적 처치효과
# ===========================================================
# 코호트: 조기 도입(g=3), 후기 도입(g=5), 미도입(never). 각 100단위.
# 기간 t = 0..7 (예: 2016~2023). 미도입은 g=99로 표기(사실상 무한).
N_PER = 100
T = 8
COHORTS = {3: N_PER, 5: N_PER, 99: N_PER}  # 99 = never-treated
THETA = 0.30  # 처치 후 1기당 효과 증가분(동적 효과) — 두 코호트 공통


def prepare_staggered():
    """시차 도입 + 동적(시간에 따라 커지는) 처치효과 패널을 저장한다(8-2).

    처치효과는 처치 후 경과기간에 비례해 커지도록(1기당 +0.30) 결정론적으로 심어,
    진짜 overall ATT를 알고 TWFE와 CS group-time 추정량을 비교한다.
    """
    RNG = np.random.default_rng(42)

    rows = []
    uid = 0
    for g, n in COHORTS.items():
        for _ in range(n):
            alpha = RNG.normal(0.0, 1.0)  # 단위 고정효과
            for t in range(T):
                lam = 0.10 * t  # 공통 시간추세
                # 동적 처치효과: 처치 후 경과기간(t-g+1)에 비례해 커짐
                if g != 99 and t >= g:
                    tau = THETA * (t - g + 1)
                else:
                    tau = 0.0
                eps = RNG.normal(0.0, 0.30)
                y = alpha + lam + tau + eps
                rows.append((uid, t, g, int(g != 99 and t >= g), tau, y))
            uid += 1

    panel = pd.DataFrame(rows, columns=["unit", "time", "cohort", "D", "tau", "y"])
    panel.to_parquet(DATA_DIR / "staggered_panel.parquet", index=False)
    print(f"  시차 도입 패널 {len(panel)}행({len(COHORTS)}코호트×{N_PER}단위×{T}기) → staggered_panel.parquet")


# ===========================================================
# 8-3 데이터: 신규 출점 격자 패널 — 관측 세계 + 대조군 두 세계
# ===========================================================
def prepare_store_opening():
    """출점 효과 예제의 격자 패널 세 개와 참값을 저장한다(8-3).

    세 세계는 시드가 같아 공변량과 매장 배치가 완전히 동일하다. 다른 것은 심어
    둔 메커니즘뿐이다 — 그래야 추정치의 차이를 메커니즘 탓으로 돌릴 수 있다.

    - 세계 A(관측): 내생적 배치 + 시변 교란 + 자기잠식
    - 세계 B(잠식 0): 자기잠식만 끈다 → 도넛 설계의 이득이 정말 잠식에서 온 것인지
    - 세계 C(효과 0): 총효과·자기잠식을 함께 끈다 → 절차 자체에 편향이 있는지
    """
    worlds = {
        "A": dict(with_effect=True, with_cannibalization=True),
        "B": dict(with_effect=True, with_cannibalization=False),
        "C": dict(with_effect=False, with_cannibalization=False),
    }
    truths = {}
    for tag, opt in worlds.items():
        out = make_store_panel(seed=42, **opt)
        out["panel"].to_parquet(DATA_DIR / f"store_opening_panel_{tag}.parquet", index=False)
        truths[tag] = out["truth"]
        if tag == "A":
            out["cells"].to_parquet(DATA_DIR / "store_opening_cells.parquet", index=False)

    (DATA_DIR / "store_opening_truth.json").write_text(
        json.dumps(truths, ensure_ascii=False, indent=2), encoding="utf-8")

    tA = truths["A"]
    print(f"  출점 패널 {tA['panel_rows']}행"
          f"(브랜드 상권 격자 {tA['n_cells_universe']}개 × {tA['months']}개월)"
          f" × 3세계 → store_opening_panel_A/B/C.parquet")
    print(f"  격자 속성 {tA['n_cells_universe']}행 → store_opening_cells.parquet")
    print(f"  처치 격자 {tA['n_treated_cells']} · 잠식 링 격자 {tA['n_ring_cells']}"
          f" · 원거리 격자 {tA['n_far_cells']}")
    print(f"  참값 — 총효과 {tA['true_att_total']:+.3f} ·"
          f" 잠식 최대 {tA['true_kappa_peak']:.3f}(링 평균 {tA['true_kappa_ring_mean']:.3f}) ·"
          f" 순효과 {tA['true_net_index']:+.2f} 매출지수"
          f"({tA['true_net_pct_of_zone']:+.2f}%) → store_opening_truth.json")


def main():
    print("=" * 60)
    print("실습 데이터 준비 (8-1 야간조명 DID/DML · 8-2 시차 도입 DID · 8-3 출점 도넛 DID)")
    print("=" * 60)
    prepare_nightlight()
    prepare_staggered()
    prepare_store_opening()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
