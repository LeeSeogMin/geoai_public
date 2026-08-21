"""
11-0. 실습 데이터 준비: 재난안전·치안·보험 리스크 분석용 시뮬레이션 데이터 저장
=================================================================
이 장의 11-1(침수위험)·11-2(예측적 치안)·11-3(보험 리스크) 분석이 쓰는
시뮬레이션 데이터를 미리 만들어 `data/` 폴더에 저장한다. 학습자는 분석 전에
이 스크립트를 한 번 실행해 데이터를 준비하고, 이후 각 분석 코드는 저장된 파일을
불러와 분석만 수행한다.

여기서 만드는 세 데이터는 실제 공개 데이터를 그대로 구하기 어려운 대상이라
현실적인 값과 공간 구조를 갖도록 합성한 교육용 데이터다(개인 식별 불가, 격자 집계).
  - 침수 지형·강우·침수심 격자(11-1): 하천 골짜기 지형에서 저지대×고강우의
    비선형과 이웃 저지대로의 공간 흐름으로 생성한 침수심.
  - 예측적 치안의 '진짜 범죄율' 필드(11-2): 관측되지 않는 진짜 범죄율과 집단·역사적
    단속 편향. 되먹임 시뮬레이션(단속→기록→예측→재배치)은 분석 코드가 수행한다.
  - 보험 건물 데이터(11-3): 500개 건물에 소득·보험 가입·청구 강도·진짜 침수확률을
    부여. 핵심 DGP는 저소득→저지대 거주→보험 미가입→청구 기록 부재의 상관.

실제 분석에서는 강우레이더·DEM·토지피복·하천망과 SAR 침수 관측(11-1),
치안·인구·환경 자료(11-2), 보험 청구·지형·건물대장 자료(11-3)를 사용한다.
시뮬레이션을 쓰는 이유는 생성 규칙을 우리가 알고 있어야 모델이 그 규칙을
제대로 포착했는지 채점할 수 있기 때문이다.

부동소수점을 정확히 보존하도록 Parquet로 저장한다(모델 재현성 확보).

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter11/code/11-0-simdata-prep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# =====================================================================
# 11-1 침수: 지형·강우·침수심 격자
# =====================================================================
GRID_ROWS, GRID_COLS = 20, 15
N1 = GRID_ROWS * GRID_COLS


def _neighbor_mean(rows, cols, values):
    """4-이웃 평균. 미관측 공간 흐름(상류→하류)의 대리."""
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    grid[rows, cols] = values
    out = np.zeros(N1)
    for i, (r, c) in enumerate(zip(rows, cols)):
        neigh = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                neigh.append(grid[rr, cc])
        out[i] = np.nanmean(neigh)
    return out


def prepare_flood():
    """지형·강우·침수위험 격자를 만들어 저장한다(11-1).

    핵심 설계: 하천이 격자 중앙 열을 따라 흐른다(저지대 골짜기).
    침수위험은 (저표고)×(고강우)×(불투수면)의 비선형 + 이웃 저지대로의 공간 흐름.
    """
    RNG = np.random.default_rng(42)
    rows, cols = np.divmod(np.arange(N1), GRID_COLS)

    # 표고: 하천(중앙 열)에서 멀수록 높음 + 완만한 공간 변동
    river_col = (GRID_COLS - 1) / 2.0
    elevation = (np.abs(cols - river_col) * 6.0
                 + np.sin(rows / 3.0) * 5.0
                 + RNG.normal(0, 3.0, N1))
    elevation = elevation - elevation.min()           # 0 기준
    river_dist = np.abs(cols - river_col)

    # 경사: 표고의 국소 차이 근사
    slope = np.abs(elevation - _neighbor_mean(rows, cols, elevation))

    # 불투수면(도심일수록 높음, 하천 주변 저지대에 시가지) + 강우(북동 방향 경도)
    imperv = np.clip(0.6 - 0.02 * river_dist + RNG.normal(0, 0.12, N1), 0.05, 0.95)
    rainfall = np.clip(80 + 2.0 * rows + 1.5 * (GRID_COLS - cols)
                       + RNG.normal(0, 8, N1), 40, 200)   # mm

    # 진짜 침수심(cm): 저지대 비선형 + 불투수면 + 강우 상호작용
    low = np.maximum(30 - elevation, 0)               # 저지대 가중(표고 낮을수록↑)
    true_depth = (
        0.6 * low
        + 0.05 * low * (rainfall / 100.0)             # 저지대×강우 비선형 상호작용
        + 12.0 * imperv
        + RNG.normal(0, 2.5, N1)
    )
    # 공간 흐름: 이웃(특히 상류)의 침수심이 하류로 모임
    true_depth = true_depth + 0.35 * _neighbor_mean(rows, cols, np.maximum(true_depth, 0))
    true_depth = np.clip(true_depth, 0, None)

    df = pd.DataFrame({
        "cell": np.arange(N1), "row": rows, "col": cols,
        "elevation": elevation, "slope": slope, "river_dist": river_dist,
        "imperv": imperv, "rainfall": rainfall, "flood_depth": true_depth,
    })
    df.to_parquet(DATA_DIR / "flood_terrain.parquet", index=False)
    print(f"  침수 격자 {len(df)}개(20×15) → flood_terrain.parquet")


# =====================================================================
# 11-2 예측적 치안: 관측되지 않는 '진짜 범죄율' 필드
# =====================================================================
GRID2 = 16
N2 = GRID2 * GRID2


def smooth_field(fr, fc, noise_sd, rng):
    rr, cc = np.meshgrid(np.linspace(0, 1, GRID2), np.linspace(0, 1, GRID2), indexing="ij")
    f = np.sin(2 * np.pi * fr * rr) + np.cos(2 * np.pi * fc * cc) + rng.normal(0, noise_sd, (GRID2, GRID2))
    return (f - f.min()) / (f.max() - f.min())


def prepare_policing():
    """예측적 치안 시뮬레이션의 '진짜 범죄율' 필드와 집단·편향 구조를 저장한다(11-2).

    진짜 범죄율(관측 불가)은 공간 패턴을 갖고, 두 집단(A·B)의 진짜율은 거의 같다.
    다만 group A(좌측 절반)는 과거 단속이 집중되어 기록이 부풀려져 있다(역사적 편향).
    단속→기록→예측→재배치의 되먹임 루프(포아송 적발 과정)는 분석 코드가 수행한다.
    """
    RNG = np.random.default_rng(42)
    # 진짜 범죄율(관측 불가): 공간 패턴, 평균 ~8건
    true_rate = 4 + 8 * smooth_field(1.2, 1.0, 0.25, RNG).reshape(-1)

    # 집단(예: 인구·사회경제 속성)과 '역사적 과잉단속' 편향:
    #   group A(좌측 절반)는 과거 단속이 집중되어 기록이 부풀려져 있다.
    rr, cc = np.divmod(np.arange(N2), GRID2)
    group_A = cc < (GRID2 // 2)                      # 과잉단속 집단
    hist_bias = np.where(group_A, 1.8, 0.8)          # 진짜율은 같아도 기록은 편향

    df = pd.DataFrame({
        "row": rr, "col": cc,
        "true_rate": true_rate, "group_A": group_A, "hist_bias": hist_bias,
    })
    df.to_parquet(DATA_DIR / "policing_field.parquet", index=False)
    print(f"  치안 필드 {len(df)}개(16×16) → policing_field.parquet")


# =====================================================================
# 11-3 보험 리스크: 건물별 홍수 리스크와 보험 가입 편향
# =====================================================================
N_BUILDINGS = 500


def prepare_insurance():
    """보험 리스크 분석용 건물 데이터를 만들어 저장한다(11-3).

    핵심 DGP: 보험 가입이 소득에 비례하고, 저소득 가구가 저지대에 거주한다.
    라벨 편향을 두 층으로 나누어 심는다.
      (1) 선택: 청구 기록은 가입 건물에서만 관측된다(미가입은 학습에서 제외).
      (2) 라벨 왜곡: 가입자 중 저소득 가구는 소액 피해를 신고하지 않는다.
    (2)가 11-2의 '순찰이 적발 기록을 만드는 것'과 같은 뼈대이며, 소득이 피처에
    없으므로 모델이 스스로 보정할 수 없다.
    """
    RNG = np.random.default_rng(2024)

    # 침수 격자 불러와서 격자-건물 매핑
    flood_df = pd.read_parquet(DATA_DIR / "flood_terrain.parquet")
    n_grids = len(flood_df)

    # 건물별 소속 격자: 격자당 대략 균등 (일부 격자에 더 많이)
    grid_ids = RNG.choice(n_grids, size=N_BUILDINGS, replace=True)

    # 격자 정보 가져오기
    grid_elev = flood_df["elevation"].values[grid_ids]
    grid_depth = flood_df["flood_depth"].values[grid_ids]

    # --- 건물 속성 ---
    # 지하층 유무: 저지대일수록 반지하 비율 높음 (저지대 = 저렴 = 저소득)
    elev_norm = (grid_elev - grid_elev.min()) / (grid_elev.max() - grid_elev.min() + 1e-8)
    p_basement = np.clip(0.4 - 0.35 * elev_norm, 0.02, 0.45)
    floor_type = np.zeros(N_BUILDINGS, dtype=int)  # 0=없음, 1=반지하, 2=완전지하
    for i in range(N_BUILDINGS):
        r = RNG.random()
        if r < p_basement[i] * 0.3:
            floor_type[i] = 2  # 완전지하
        elif r < p_basement[i]:
            floor_type[i] = 1  # 반지하

    building_age = np.clip(RNG.normal(25, 15, N_BUILDINGS), 1, 70).astype(int)
    structure_type = RNG.choice(["RC", "steel", "wood"], N_BUILDINGS, p=[0.5, 0.3, 0.2])

    # --- 소득과 보험 가입 ---
    # 소득: 표고와 양(+) 상관 (고지대 = 주거비 높음 = 고소득)
    income_score = 0.4 * elev_norm + 0.6 * RNG.random(N_BUILDINGS)
    income_score = (income_score - income_score.min()) / (income_score.max() - income_score.min())
    low_income = income_score < 0.35  # 하위 35%

    # 보험 가입: 소득이 높을수록 가입 확률 높음
    p_insured = np.clip(0.3 + 0.5 * income_score + RNG.normal(0, 0.08, N_BUILDINGS), 0.05, 0.95)
    insured = RNG.random(N_BUILDINGS) < p_insured

    # --- 진짜 침수 확률과 청구 기록 ---
    # 진짜 30년 침수 확률: 격자 침수심에 건물 특성 결합
    base_prob = np.clip(grid_depth / 60.0, 0, 0.95)  # 침수심 비례
    # 지하층이 있으면 피해 확률 증가
    basement_mult = np.where(floor_type == 2, 1.4, np.where(floor_type == 1, 1.2, 1.0))
    # 목조는 피해 취약
    struct_mult = np.where(structure_type == "wood", 1.15,
                           np.where(structure_type == "steel", 1.05, 1.0))
    true_flood_prob = np.clip(base_prob * basement_mult * struct_mult
                              + RNG.normal(0, 0.03, N_BUILDINGS), 0.01, 0.95)

    # 청구 강도(claim intensity): "과거 10년간 청구 금액 / 보험 가액"을 모사한 연속값.
    # 진짜 침수 확률에 비례한다.
    claim_intensity = np.clip(
        true_flood_prob * 0.85 + RNG.normal(0, 0.06, N_BUILDINGS), 0, 1)

    # 라벨 편향의 두 원천 — 둘을 구분해서 심는다.
    #
    # (1) 선택(selection): 미가입자는 청구 기록 자체가 없다 → 학습 표본에서 빠진다.
    #     피처(표고 등)는 관측되므로 원리상 보정 가능한 공변량 이동에 가깝다.
    #
    # (2) 라벨 왜곡(label distortion): 가입자 안에서도 저소득 가구는 자기부담금
    #     부담과 보험료 할증 우려 때문에 소액 피해를 신고하지 않는다. 그 결과
    #     "피해가 있었으나 기록되지 않은" 사례가 저소득에 몰린다. 소득은 보험사의
    #     피처에 들어 있지 않으므로 이 왜곡은 모델이 스스로 보정할 수 없다.
    #     11-2에서 '순찰이 적발 기록을 만드는' 것과 같은 뼈대다 — 기록이 현상이
    #     아니라 '기록하는 행위'의 산물이 된다.
    SMALL_CLAIM_CUT = 0.25       # 이 미만이면 '소액 피해'
    UNDERREPORT_P = 0.70         # 저소득 가구의 소액 피해 미신고 확률
    small_claim = claim_intensity < SMALL_CLAIM_CUT
    underreported = small_claim & low_income & (RNG.random(N_BUILDINGS) < UNDERREPORT_P)
    claim_recorded = np.where(underreported, 0.0, claim_intensity)

    # 미가입자의 청구 기록은 관측 불가 → NaN
    claim_observed = np.where(insured, claim_recorded, np.nan)

    df = pd.DataFrame({
        "building_id": np.arange(N_BUILDINGS),
        "grid_id": grid_ids,
        "elevation": grid_elev,
        "grid_flood_depth": grid_depth,
        "floor_type": floor_type,
        "building_age": building_age,
        "structure_type": structure_type,
        "income_score": income_score,
        "low_income": low_income,
        "insured": insured,
        "claim_observed": claim_observed,
        # 가입 여부와 무관하게 '기록되었을' 청구값. 대조군(null control) 전용으로,
        # 선택편향의 몫만 분리해 재기 위해 저장한다. 보험사는 볼 수 없는 값이다.
        "claim_recorded_latent": claim_recorded,
        "underreported": underreported,
        "true_flood_prob": true_flood_prob,
    })
    df.to_parquet(DATA_DIR / "insurance_buildings.parquet", index=False)
    print(f"  보험 건물 {len(df)}개 → insurance_buildings.parquet")
    print(f"    가입률: {insured.mean():.1%}, 저소득 비율: {low_income.mean():.1%}")
    print(f"    소액 피해 미신고(가입자 중): "
          f"{(underreported & insured).sum()}건 "
          f"(저소득 가입자의 {(underreported & insured).sum() / max((low_income & insured).sum(), 1):.1%})")
    print(f"    저소득 가입률: {insured[low_income].mean():.1%}, "
          f"고소득 가입률: {insured[~low_income].mean():.1%}")


def main():
    print("=" * 60)
    print("실습 데이터 준비 (11-1 침수 · 11-2 예측적 치안 · 11-3 보험 리스크)")
    print("=" * 60)
    prepare_flood()
    prepare_policing()
    prepare_insurance()
    print("\n[완료] 실습 데이터를 data/ 폴더에 저장했다.")


if __name__ == "__main__":
    main()
