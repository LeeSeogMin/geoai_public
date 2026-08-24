"""
11장 실습 1: 침수위험 공간 예측과 대응 우선순위 (2층 AI 정당)
=================================================================
재난안전 정책 질문: "어디가 먼저, 더 깊이 침수되는가? 한정된 대비 자원을 어디에 먼저 둘까?"

3층 모델:
  [1층 GIS]  격자별 표고·경사·불투수면·강우·하천거리 → 저지대 침수 노출 단순 식별
  [2층 AI ]  침수위험 공간 예측: 공간 시차(이웃/상류 위험) + 비선형(저지대×고강우) +
             공간 블록 CV + 공간 피처 유무 성능 비교(=AI 정당성) + 피처 중요도(설명가능성)
  [3층 정책] 격자·행정구역별 침수위험 대응 우선순위 CSV

AI 정당성(규칙2):
  - 침수는 지형-수문의 '비선형'(저지대일수록, 고강우일수록 급증)이고
    물이 인접 저지대로 모이는 '공간 흐름(spillover)'을 가진다.
  - 단순 선형/비공간 모델은 이 상호작용·흐름을 놓친다 → 공간 명시적 ML 정당.
  - 영상 측면에서는 SAR(Sentinel-1)이 구름을 관통해 홍수 범위를 탐지하고
    U-Net/SAM 세그멘테이션(6장)으로 침수역을 자동 추출한다(여기서는 위험 예측에 집중).

데이터: 미리 준비한 침수 지형·강우·침수심 격자를 불러와 사용(개인 식별 불가, 격자 집계).
  분석·추정은 실제 계산값. 실제 분석에서는 강우레이더·DEM·토지피복·하천망과
  SAR 침수 관측을 사용한다.
실행:
    python 11-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 11-1-flood-risk-priority.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GRID_ROWS, GRID_COLS = 20, 15
N = GRID_ROWS * GRID_COLS
N_ADMIN = 6                       # 행정구역 6개(열 기준 분할)


def _neighbor_mean(rows, cols, values):
    """4-이웃 평균. 미관측 공간 흐름(상류→하류)의 대리."""
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


def load_terrain():
    """미리 준비한 침수 지형·강우·침수심 격자를 불러온다.

    생성 규칙(11-0-simdata-prep.py): 하천이 격자 중앙 열을 따라 흐르는 골짜기
    지형에서, 침수위험은 (저표고)×(고강우)×(불투수면)의 비선형에 이웃 저지대로의
    공간 흐름을 더해 만들었다. 이 규칙을 우리가 알고 있어야 모델이 규칙을
    제대로 포착했는지 채점할 수 있다.
    """
    path = DATA_DIR / "flood_terrain.parquet"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 11-0-simdata-prep.py")
    return pd.read_parquet(path)


def spatial_lag(df, col):
    return _neighbor_mean(df["row"].values, df["col"].values, df[col].values)


def spatial_block_folds(df, n_blocks=5):
    block = (df["col"].values / (GRID_COLS / n_blocks)).astype(int)
    block = np.clip(block, 0, n_blocks - 1)
    for b in range(n_blocks):
        test = block == b
        yield ~test, test


def cv_r2(X, y, folds):
    preds = np.full(len(y), np.nan)
    for tr, te in folds:
        m = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return r2_score(y, preds)


def main():
    print("=" * 64)
    print("침수위험 공간 예측과 대응 우선순위 (11-1, 2층 AI)")
    print("=" * 64)
    df = load_terrain()
    print(f"\n분석 단위: {N}개 격자(20×15), 행정구역 {N_ADMIN}개, 개인 식별 불가")

    # -------- [1층 GIS] 저지대 침수 노출 --------
    print("\n[1층 GIS] 저지대 침수 노출 (표고 하위 25%)")
    low_q = df["elevation"].quantile(0.25)
    low = df["elevation"] <= low_q
    print(f"  저지대 격자: {low.sum()}개 | 평균 침수심 {df.loc[low,'flood_depth'].mean():.1f}cm "
          f"vs 그 외 {df.loc[~low,'flood_depth'].mean():.1f}cm")
    print("  → 표고만으로는 강우·불투수면·상류 흐름의 비선형을 놓침 → 2층 AI 동기")

    # -------- [2층 AI] 공간 명시적 침수위험 예측 --------
    print("\n[2층 AI] 침수위험 예측: 공간 피처 유무 + 검증 방식 비교")
    base_cols = ["elevation", "slope", "river_dist", "imperv", "rainfall"]
    df["depth_lag"] = spatial_lag(df, "flood_depth")     # 이웃 침수심(공간 흐름)
    df["elev_lag"] = spatial_lag(df, "elevation")
    spatial_cols = base_cols + ["depth_lag", "elev_lag"]

    y = df["flood_depth"].values
    folds = list(spatial_block_folds(df))
    r2_base = cv_r2(df[base_cols].values, y, folds)
    r2_spat = cv_r2(df[spatial_cols].values, y, folds)
    rand_folds = list(KFold(5, shuffle=True, random_state=42).split(df[spatial_cols].values))
    r2_rand = cv_r2(df[spatial_cols].values, y, rand_folds)

    print(f"  공간 블록 CV | 비공간 피처     R² = {r2_base:6.3f}")
    print(f"  공간 블록 CV | +공간 시차 피처 R² = {r2_spat:6.3f}  (개선 {r2_spat-r2_base:+.3f})")
    print(f"  랜덤    CV | +공간 시차 피처 R² = {r2_rand:6.3f}  "
          f"(공간 누수 과대추정 {r2_rand-r2_spat:+.3f})")
    if r2_spat - r2_base > 0.02:
        print("  → 공간 시차(상류 흐름)가 예측을 개선 = 공간 명시적 AI 정당화(규칙2).")
    else:
        print("  → 공간 피처 개선 미미 → 단순 모델로 충분(AI 강제 금지, 규칙2).")

    full = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                     max_depth=3, learning_rate=0.05).fit(
        df[spatial_cols].values, y)
    imp = pd.Series(full.feature_importances_, index=spatial_cols).sort_values(ascending=False)
    print("\n  피처 중요도(설명가능성):")
    for name, val in imp.items():
        print(f"    {name:11s} {val:.3f}")

    # -------- [3층 정책] 우선순위 --------
    print("\n[3층 정책] 침수위험 대응 우선순위")
    df["pred_depth"] = full.predict(df[spatial_cols].values)
    df["priority_rank"] = df["pred_depth"].rank(ascending=False).astype(int)
    df["admin"] = (df["col"].values / (GRID_COLS / N_ADMIN)).astype(int).clip(0, N_ADMIN - 1)

    top = (df.sort_values("pred_depth", ascending=False)
             [["cell", "row", "col", "elevation", "imperv", "rainfall",
               "pred_depth", "priority_rank"]].head(10).round(2))
    print("  격자 우선순위 상위 10:")
    print(top.to_string(index=False))

    admin = (df.groupby("admin")
               .agg(mean_pred_depth=("pred_depth", "mean"),
                    high_risk_cells=("pred_depth", lambda s: int((s > df["pred_depth"]
                                     .quantile(0.75)).sum())))
               .sort_values("mean_pred_depth", ascending=False).round(2))
    admin["priority_rank"] = range(1, len(admin) + 1)
    print("\n  행정구역 우선순위:")
    print(admin.to_string())

    cell_csv = RESULTS_DIR / "flood_risk_priority.csv"
    admin_csv = RESULTS_DIR / "flood_admin_priority.csv"
    df.sort_values("pred_depth", ascending=False).round(4).to_csv(cell_csv, index=False)
    admin.to_csv(admin_csv)
    print(f"\n  격자 우선순위 저장 → {cell_csv.name}")
    print(f"  행정구역 우선순위 저장 → {admin_csv.name}")
    print("  ※ 침수위험 지도는 건축 규제·홍수보험·대피계획의 공간 근거가 된다.")

    # -------- [불확실성] 정규화 conformal 예측구간·신뢰등급(OOF 근사) --------
    # 우선순위 점추정만으로 자원을 배분하면 '얼마나 틀릴지 모른 채 확신하는'(7.5절) 위험이 있다.
    # conformal 예측(Lei et al. 2018)은 분포가정 없이 예측구간을 주고, 지역 난이도로
    # 정규화(Romano et al. 2019 계열)하면 이질적 불확실성을 반영한다. 단, 아래는 엄밀한
    # 분할 conformal(학습/보정/시험 3분할)이 아니라, 공간 블록 CV의 OOF 잔차를 σ̂ 적합과
    # 캘리브레이션에 함께 쓰는 '교육용 근사'다(포함률은 다소 낙관적).
    print("\n[불확실성] 정규화 conformal 예측구간(90%, OOF 근사)과 신뢰등급")
    X_all = df[spatial_cols].values
    # 1) 공간 블록 CV의 out-of-fold(OOF) 예측 → 누수 없는 정직한 잔차
    oof = np.full(N, np.nan)
    for tr, te in folds:
        m = GradientBoostingRegressor(random_state=42, n_estimators=200,
                                      max_depth=3, learning_rate=0.05).fit(X_all[tr], y[tr])
        oof[te] = m.predict(X_all[te])
    resid = np.abs(y - oof)
    # 2) 지역 난이도 σ̂(x): OOF 절대잔차를 특징으로 회귀(이질적 불확실성 추정)
    sigma = np.clip(GradientBoostingRegressor(random_state=1, n_estimators=150, max_depth=3,
                    learning_rate=0.05).fit(X_all, resid).predict(X_all), 1e-3, None)
    # 3) 정규화 conformal 점수의 k번째 순서통계(k=⌈(n+1)(1-α)⌉, 분할 conformal 관례)
    alpha = 0.10
    scores = np.sort(resid / sigma)
    k = int(np.ceil((N + 1) * (1 - alpha)))
    q = scores[min(k, N) - 1]                       # k>N이면 최댓값(구간 무한대 회피)
    half = q * sigma                                # 단위별 예측구간 반폭(이질적)
    cover = float(np.mean(resid <= half))           # OOF 경험적 포함률(≈0.90 목표)
    df["pi_half"] = half
    df["pi_low"] = np.clip(df["pred_depth"] - half, 0, None)
    df["pi_high"] = df["pred_depth"] + half
    t1, t2 = np.quantile(half, [1 / 3, 2 / 3])      # 신뢰등급: 반폭 3분위(좁을수록 신뢰↑)
    df["confidence"] = np.where(half <= t1, "높음", np.where(half <= t2, "중간", "낮음"))
    print(f"  90% 예측구간 경험적 포함률(OOF) = {cover:.3f}  (목표 0.90)")
    print(f"  구간 반폭(cm): 중앙값 {np.median(half):.1f}, 범위 [{half.min():.1f}, {half.max():.1f}]")
    print("  주의: σ̂를 같은 OOF 잔차로 적합했으므로 포함률은 다소 낙관적이다.")
    print("        엄밀한 실무는 학습/보정/시험 3분할 split conformal을 쓴다(7.5절·보론).")

    # -------- [적용가능영역 AoA] Meyer & Pebesma 2021 아이디어의 간이 구현 --------
    # 모델은 학습분포와 닮은 곳에서만 신뢰할 수 있다. 특징공간에서 크게 벗어난 격자를
    # '적용가능영역 밖'으로 표기한다. 정석(CAST 패키지)은 학습 DI 분포를 교차검증으로
    # 구해 임계값을 잡지만, 여기서는 개념을 보이는 근사다.
    print("\n[적용가능영역 AoA] 특징공간 비유사도로 신뢰 밖 영역 표기(개념 근사)")
    w = full.feature_importances_
    Xw = ((X_all - X_all.mean(0)) / (X_all.std(0) + 1e-9)) * np.sqrt(w)  # 표준화+중요도 가중
    Dm = cdist(Xw, Xw)
    mean_pair = Dm[np.triu_indices(N, 1)].mean()    # 학습 데이터 평균 pairwise 거리(정규화 기준)
    np.fill_diagonal(Dm, np.inf)
    di = Dm.min(1) / mean_pair                       # DI ≈ 최근접 특징거리 / 평균 pairwise(Meyer&Pebesma식)
    q1, q3 = np.quantile(di, [0.25, 0.75])
    thr = q3 + 1.5 * (q3 - q1)                        # 임계값: DI 분포의 outlier 경계(근사)
    df["aoa_di"] = di
    df["inside_aoa"] = di <= thr
    n_out = int((~df["inside_aoa"]).sum())
    print(f"  DI 임계값(Q3+1.5·IQR) = {thr:.2f} | 적용가능영역 밖 격자 = {n_out}개")
    print("  → AoA 밖 격자는 신뢰가 낮아 현장확인 전에는 우선순위를 확정하지 않는다.")

    # -------- 우선순위표에 불확실성·AoA 열 추가 --------
    admin2 = (df.groupby("admin")
                .agg(mean_pred_depth=("pred_depth", "mean"),
                     mean_pi_half=("pi_half", "mean"),
                     low_conf_cells=("confidence", lambda s: int((s == "낮음").sum())),
                     aoa_out_cells=("inside_aoa", lambda s: int((~s).sum())))
                .sort_values("mean_pred_depth", ascending=False).round(2))
    admin2["priority_rank"] = range(1, len(admin2) + 1)
    a1, a2 = np.quantile(admin2["mean_pi_half"], [1 / 3, 2 / 3])
    admin2["confidence"] = np.where(admin2["mean_pi_half"] <= a1, "높음",
                                    np.where(admin2["mean_pi_half"] <= a2, "중간", "낮음"))
    print("\n  불확실성·AoA를 반영한 행정구역 우선순위:")
    print(admin2.to_string())
    admin2.to_csv(RESULTS_DIR / "flood_admin_priority_uncertainty.csv")
    print("  → 예측 침수심이 높아도 신뢰등급이 '낮음'이거나 AoA 밖이면, 확정 배분 전에")
    print("     추가 관측·현장확인으로 보완한다(점추정 맹신 방지).")

    print("\n[완료] 침수위험 공간 예측·대응 우선순위 분석을 마쳤다.")


if __name__ == "__main__":
    main()
