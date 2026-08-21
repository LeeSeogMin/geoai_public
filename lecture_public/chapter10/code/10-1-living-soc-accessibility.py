"""
10장 실습 1: 생활SOC 접근성 형평성과 우선 공급 후보지 (1층 기준선)
=====================================================================
도시·생활권 정책 질문: "생활SOC 접근성 격차는 어디인가? 어디에 먼저 공급해야 하는가?"

3층 모델에서의 위치 — 이 분석은 의도적으로 [1층 GIS]에 머문다:
  [1층 GIS]  격자별 인구 + 다종 생활SOC(도서관·돌봄·공원·체육) 위치 →
             최근접 이동시간 접근성, 서비스 권역, 임계값 미달 비율, 형평성(Gini)
  [2층 AI ]  없음 — 접근성 격차 '진단'은 네트워크/버퍼 분석으로 충분하다.
             AI를 억지로 붙이지 않는다(규칙2: AI 정당화가 안 되면 기준선으로 명시).
  [3층 정책] 접근성 최하위 ∩ 인구 상위 격자 → 생활SOC 우선 공급 후보지 + 형평성 지표

교육 의도(GeoAI 정체성 게이트 5번 / harness §7.5):
  "언제 AI를 쓰지 *않는가*"를 가르친다. 모든 공간 문제가 AI를 필요로 하지 않는다.
  민원 시공간 예측(10-2)은 비선형 시공간 확산 때문에 2층 AI가 정당화되지만,
  접근성 진단은 결정론적 거리·시간 계산으로 충분하므로 기준선(1층)으로 둔다.

데이터: 미리 준비한 격자 인구·생활SOC 시설 데이터를 불러와 사용(개인 식별 불가,
  격자 집계). 거리·시간·형평성은 불러온 데이터로 계산한 실제 값이다.
  실제 분석에서는 SGIS 통계격자 인구 + 생활SOC 시설 위치(공공데이터포털)를 사용한다.

실행:
    python 10-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 10-1-living-soc-accessibility.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ===========================================================
# 1. 격자 파라미터 (도시 격자)
# ===========================================================
GRID_ROWS, GRID_COLS = 20, 15        # 300개 격자를 20×15로 배치
N = GRID_ROWS * GRID_COLS
CELL_KM = 0.5                        # 격자 한 변 = 500 m
WALK_KMH = 4.0                       # 도보 속도 4 km/h
WALK_15MIN = 15.0                    # 생활SOC '10분 생활권' 정책 목표 → 보수적으로 15분 임계값

# 생활SOC 4종과 종류별 시설 수(도시 전역에 분산 공급, 종류별 공급량 차이)
SOC_TYPES = {"도서관": 6, "돌봄": 10, "공원": 12, "체육": 8}


def gini(x):
    """접근성(이동시간) 분포의 Gini 계수. 0=완전 균등, 1=완전 불균등.

    표준 정의(정렬 후 누적합 기반). 이동시간 Gini가 높을수록 접근성 격차가 크다.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def load_city():
    """미리 준비한 격자 인구와 생활SOC 시설 위치를 불러온다.

    데이터의 설계 의도(현실적 불평등 주입): 인구는 도심(중앙)에 집중되고,
    시설은 인구가 많은 곳에 우선 공급되는 경향이 있다(예산·수요 논리).
    → 외곽 저밀도 지역이 접근성 사각지대가 되는 구조를 담고 있다.
    """
    grid_path = DATA_DIR / "soc_grid.parquet"
    fac_path = DATA_DIR / "soc_facilities.parquet"
    if not grid_path.exists() or not fac_path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {grid_path}\n먼저 실행: python 10-0-simdata-prep.py")

    grid = pd.read_parquet(grid_path)
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    pop = grid["pop"].to_numpy()

    fac = pd.read_parquet(fac_path)
    # SOC_TYPES 정의 순서를 유지해 종류별 접근성 출력 순서를 고정한다.
    facilities = {soc: fac.loc[fac["soc"] == soc, "facility_cell"].to_numpy()
                  for soc in SOC_TYPES}
    return rows, cols, pop, facilities


def nearest_walk_time(rows, cols, facility_idx):
    """각 격자 → 해당 종류 최근접 시설까지 도보 이동시간(분).

    격자 간 유클리드 거리(셀 단위)를 km로 환산 후 도보 시간으로 변환.
    """
    fr, fc = rows[facility_idx], cols[facility_idx]
    # (N, n_fac) 거리 행렬 → 최소
    dr = rows[:, None] - fr[None, :]
    dc = cols[:, None] - fc[None, :]
    dist_cells = np.sqrt(dr ** 2 + dc ** 2)
    nearest_km = dist_cells.min(axis=1) * CELL_KM
    return nearest_km / WALK_KMH * 60.0


def main():
    print("=" * 64)
    print("생활SOC 접근성 형평성과 우선 공급 후보지 (10-1, 1층 기준선)")
    print("=" * 64)
    rows, cols, pop, facilities = load_city()
    print(f"\n분석 단위: {N}개 격자(20×15, 셀 500m), 총 인구 {int(pop.sum()):,}명")
    print(f"생활SOC: " + ", ".join(f"{k} {len(v)}개소" for k, v in facilities.items()))

    # -------- [1층 GIS] 종류별 접근성 + 형평성 --------
    print("\n[1층 GIS] 생활SOC 종류별 도보 접근성(분) — 결정론적 거리 계산")
    df = pd.DataFrame({"cell": np.arange(N), "row": rows, "col": cols, "pop": pop})
    access_cols = []
    for soc, idx in facilities.items():
        t = nearest_walk_time(rows, cols, idx)
        col = f"access_{soc}"
        df[col] = t
        access_cols.append(col)
        # 인구가중 평균 접근성 + 15분 초과(사각지대) 인구 비율
        wmean = np.average(t, weights=pop)
        over = t > WALK_15MIN
        pop_over = pop[over].sum() / pop.sum() * 100
        print(f"  {soc:4s}: 인구가중 평균 {wmean:5.1f}분 | "
              f"15분 초과 사각지대 인구 {pop_over:4.1f}% | Gini {gini(t):.3f}")

    # 종합 접근성(4종 평균) + 종합 형평성
    df["access_composite"] = df[access_cols].mean(axis=1)
    comp = df["access_composite"].values
    print(f"\n  종합 접근성(4종 평균): 인구가중 평균 {np.average(comp, weights=pop):.1f}분, "
          f"Gini {gini(comp):.3f}")
    print(f"  종합 15분 초과 격자: {(comp > WALK_15MIN).sum()}개 "
          f"(인구의 {pop[comp > WALK_15MIN].sum()/pop.sum()*100:.1f}%)")

    # -------- [AI 정당화 — 역방향] --------
    print("\n[2층 AI 판단] 이 진단에 AI가 필요한가? → 아니오(기준선으로 명시)")
    print("  접근성·서비스 권역·형평성은 거리·시간의 결정론적 계산으로 완결된다.")
    print("  학습 모델을 붙여도 설명력만 떨어지고 책임성 추적이 어려워진다(규칙2).")
    print("  → AI는 민원 시공간 '예측'(10-2)처럼 비선형·미래 추정이 필요할 때 정당화된다.")

    # -------- [3층 정책] 우선 공급 후보지 --------
    print("\n[3층 정책] 생활SOC 우선 공급 후보지 (접근성 사각 ∩ 인구 多)")
    # 결손점수 = 임계값 초과분 × 인구(많은 사람이 오래 걸릴수록 우선)
    df["deficit_min"] = np.maximum(comp - WALK_15MIN, 0.0)
    df["priority_score"] = df["deficit_min"] * df["pop"]
    df["priority_rank"] = df["priority_score"].rank(ascending=False).astype(int)
    cand = df[df["deficit_min"] > 0].sort_values("priority_score", ascending=False)
    show = cand[["cell", "row", "col", "pop", "access_composite",
                 "deficit_min", "priority_score", "priority_rank"]].head(10).round(2)
    print(f"  사각지대 후보 {len(cand)}개 중 상위 10:")
    print(show.to_string(index=False))

    csv_path = RESULTS_DIR / "living_soc_priority.csv"
    df.sort_values("priority_score", ascending=False).round(4).to_csv(csv_path, index=False)
    print(f"\n  전체 우선순위표 저장 → {csv_path.name}")
    print("  ※ 형평(equity) ≠ 균등(equality): 수요(인구)가 큰 사각지대에 먼저 공급.")
    print("    Gini는 공급 전후 형평성 변화를 추적하는 정책 모니터링 지표로 쓴다.")

    print("\n[완료] 생활SOC 접근성 형평성 분석을 마쳤다(1층 기준선).")


if __name__ == "__main__":
    main()
