"""
12장 실습 2: 감염병 시공간 확산 위험 예측과 보건소 권역 대응 우선순위
=========================================================================
보건 정책 질문: "감염병 위험은 어떻게 퍼지는가? 다음 주 어느 권역에 대응 자원을 보낼 것인가?"

3층 모델:
  [1층 GIS]  격자·보건소 권역별 발생 집계
  [2층 AI ]  시공간 확산 위험 예측: 시간 시차(자기 과거) + 공간 시차(이웃 과거) 피처 →
             시공간 피처 유무 성능 비교(=AI 정당성). 비선형 확산이 단순 추세보다 나은지 검증(심화)
  [3층 정책] 보건소 권역별 다음 기간 대응 자원 배분 우선순위 CSV

핵심 설계(공간 명시성·정당성):
  - 감염은 이웃 격자의 직전 발생에 의존해 비선형 확산한다(SaTScan 시공간 군집의 직관).
  - '공간 시차(이웃 t-1 발생)'가 (설계된) 확산을 회복함을 보인다 = 방법 검증(규칙2의 비교 설계).
    실데이터에서 시공간 AI가 정당한지는 같은 비교를 실데이터로 수행해야 입증된다.
  - 단순 시간 추세(자기 과거만)와 비교해 공간 시차가 예측을 개선하는지 실제 측정.

데이터: 미리 준비한 교육용 합성 데이터를 불러와 사용(집계 단위, 개인 식별 불가).
       분석은 실제 계산값. 실제 분석에서는 질병관리청 감염병 신고자료·유동인구·시설
       자료를 비식별 집계로 사용한다.
       (SIRS 시뮬레이터의 설계 의도와 생성은 12-0-simdata-prep.py 참고)

실행:
    python 12-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 12-2-infectious-spatiotemporal-risk.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GRID = 16                       # 16×16 격자 = 256개 분석 단위
N = GRID * GRID
T = 40                          # 40주(앞 10주 burn-in 후 정상상태 사용)
BURN = 10                       # burn-in: 도입기를 버리고 풍토병 정상상태만 분석(스케일 안정)
N_REGIONS = 8                   # 보건소 권역 수


def neighbor_sum(mat):
    """각 격자의 4-이웃 합(공간 시차). mat: (GRID,GRID)."""
    s = np.zeros_like(mat)
    s[1:, :] += mat[:-1, :]; s[:-1, :] += mat[1:, :]
    s[:, 1:] += mat[:, :-1]; s[:, :-1] += mat[:, 1:]
    return s


def load_outbreak():
    """미리 준비한 감염 발생 시계열과 인구를 (T,GRID,GRID)·(GRID,GRID)로 복원한다.

    이 데이터는 격자 SIRS 시뮬레이터로 생성했으며, 신규 감염이 자기 감염자와
    '이웃 감염자' 양쪽에서 비롯되는 공간 확산을 담고 있다(도입기 burn-in 이후 풍토병
    정상상태). 이웃의 직전 발생(공간 시차)이 다음 주 발생을 예측하는지가 분석의 초점이다.
    데이터 생성의 설계 의도는 12-0-simdata-prep.py에 있다.
    """
    inc_path = DATA_DIR / "infectious_incidence.parquet"
    pop_path = DATA_DIR / "infectious_pop.parquet"
    if not inc_path.exists() or not pop_path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {inc_path}\n먼저 실행: python 12-0-simdata-prep.py")
    inc_df = pd.read_parquet(inc_path).sort_values(["t", "row", "col"])
    incidence = inc_df["cases"].values.reshape(T, GRID, GRID)
    pop_df = pd.read_parquet(pop_path).sort_values(["row", "col"])
    pop = pop_df["pop"].values.reshape(GRID, GRID)
    return incidence, pop


def build_features(cases, pop):
    """시공간 피처 테이블: (단위×시점) 행, 다음 시점 발생 예측."""
    rows = []
    for t in range(BURN + 2, T - 1):  # burn-in 이후 정상상태만. t로 t+1 예측, t-2까지 이력
        prev = cases[t - 1]; prev2 = cases[t - 2]; cur = cases[t]
        sp_cur = neighbor_sum(cur); sp_prev = neighbor_sum(prev)
        target = cases[t + 1]
        for r in range(GRID):
            for c in range(GRID):
                rows.append({
                    "t": t, "row": r, "col": c, "pop": pop[r, c],
                    "self_t": cur[r, c], "self_t1": prev[r, c], "self_t2": prev2[r, c],
                    "neigh_t": sp_cur[r, c], "neigh_t1": sp_prev[r, c],
                    # 보건소 권역: 격자를 2행×4열 블록으로 분할 = 8개 권역(0~7)
                    "region": (r // (GRID // 2)) * 4 + (c // (GRID // 4)),
                    "target_next": target[r, c],
                })
    return pd.DataFrame(rows)


def time_block_eval(df, feat_cols):
    """시간 블록 분할: 앞 70% 기간 학습 → 뒤 30% 예측(미래 누수 차단)."""
    cut = int(df["t"].quantile(0.7))
    tr, te = df["t"] <= cut, df["t"] > cut
    m = GradientBoostingRegressor(random_state=42, n_estimators=300,
                                  max_depth=3, learning_rate=0.05)
    m.fit(df.loc[tr, feat_cols], df.loc[tr, "target_next"])
    pred = m.predict(df.loc[te, feat_cols])
    y = df.loc[te, "target_next"]
    return r2_score(y, pred), mean_absolute_error(y, pred), m, te


def main():
    cases, pop = load_outbreak()
    df = build_features(cases, pop)
    print("=" * 64)
    print("감염병 시공간 확산 위험 예측과 보건소 권역 우선순위 (12-2)")
    print("=" * 64)
    print(f"\n분석 단위: {N}개 격자 × {T}주, 보건소 권역 {N_REGIONS}개 (집계, 개인 식별 불가)")
    print(f"총 발생: {cases.sum():.0f}건, 피크 주 발생: {cases.sum(axis=(1,2)).max():.0f}건")

    # -------- [1층 GIS] 권역별 누적 발생 집계 --------
    print("\n[1층 GIS] 보건소 권역별 누적 발생 집계")
    region_total = df.groupby("region")["target_next"].sum().sort_values(ascending=False)
    for reg, tot in region_total.items():
        print(f"  권역 {int(reg)}: 누적 {tot:.0f}건")

    # -------- [2층 AI] 시공간 피처 유무 비교 --------
    print("\n[2층 AI] 다음 주 발생 예측: 시공간 피처의 정보 가치")
    temporal_only = ["pop", "self_t", "self_t1", "self_t2"]          # 자기 과거만(단순 추세)
    spatiotemporal = temporal_only + ["neigh_t", "neigh_t1"]         # +공간 시차

    r2_t, mae_t, _, _ = time_block_eval(df, temporal_only)
    r2_st, mae_st, model_st, te_mask = time_block_eval(df, spatiotemporal)
    print(f"  시간 시차만(자기 과거)   R² = {r2_t:6.3f} | MAE = {mae_t:.3f}")
    print(f"  +공간 시차(이웃 과거)    R² = {r2_st:6.3f} | MAE = {mae_st:.3f}  "
          f"(개선 R² {r2_st-r2_t:+.3f}, MAE {mae_t-mae_st:+.3f})")
    if r2_st - r2_t > 0.02:
        print("  → 공간 시차가 (설계된) 이웃 의존 확산을 회복 = 방법 검증(실데이터 실증은 별도, 규칙2)")
    else:
        print("  → 공간 시차 개선 미미 → 단순 시계열로 충분")

    imp = pd.Series(model_st.feature_importances_, index=spatiotemporal).sort_values(ascending=False)
    print("\n  피처 중요도(설명가능성):")
    for name, val in imp.items():
        print(f"    {name:10s} {val:.3f}")

    # -------- [3층 정책] 권역별 대응 우선순위 --------
    print("\n[3층 정책] 다음 주 보건소 권역별 대응 자원 배분 우선순위")
    df_te = df.loc[te_mask].copy()
    df_te["pred_next"] = model_st.predict(df_te[spatiotemporal])
    last_t = df_te["t"].max()
    snap = df_te[df_te["t"] == last_t]
    region_risk = (snap.groupby("region")["pred_next"].sum()
                   .sort_values(ascending=False))
    region_df = region_risk.reset_index()
    region_df.columns = ["region", "pred_next_week_cases"]
    region_df["priority_rank"] = region_df["pred_next_week_cases"].rank(ascending=False).astype(int)
    print(region_df.round(2).to_string(index=False))
    csv_path = RESULTS_DIR / "infectious_region_priority.csv"
    region_df.round(3).to_csv(csv_path, index=False)
    print(f"\n  권역 우선순위 저장 → {csv_path.name}")
    print("  ※ 비식별 집계 단위 분석. 신고자료 원본은 개인정보·역학조사 목적 제한 준수.")

    print("\n[완료] 감염병 시공간 확산 위험 분석을 마쳤다.")


if __name__ == "__main__":
    main()
